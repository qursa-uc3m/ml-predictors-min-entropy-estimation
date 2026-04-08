"""Orchestrates teacher -> student distillation using any model module's public API."""

import os
import sys
import time
from timeit import default_timer as timer

import numpy as np
import torch
from tqdm import tqdm

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log


class DistillationTrainer:

    def __init__(self, strategy, distillation_config=None):
        self.strategy = strategy
        self.distillation_config = distillation_config or {}

    def _build_config(self, module, run_params):
        return module.get_config(
            module.MODEL_NAME,
            run_params["filename"],
            run_params["generator"],
            run_params["seqlen"],
            run_params["step"],
            run_params["num_bytes"],
            run_params["target_bits"],
            run_params["train_ratio"],
            run_params["test_ratio"],
            run_params["learning_rate"],
            run_params["batch_size"],
            run_params["epochs"],
            run_params["is_autoregressive"],
            run_params["evaluate_all_bits"],
        )

    def _train_student(
        self,
        module,
        student,
        teacher,
        config,
        train_data,
        eval_data,
        device,
        evaluation_checkpoints,
        target_bits,
    ):
        epochs = self.distillation_config.get("epochs", config["epochs"])
        lr = self.distillation_config.get("learning_rate", config["learning_rate"])
        accumulation_steps = self.distillation_config.get("accumulation_steps", 4)

        bytes_processed = 0
        next_checkpoint_idx = 0
        partial_evals = []
        start = timer()

        student.to(device)
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda")

        strategy_name = self.strategy.__class__.__name__
        print("-" * 40)
        print(f"Distillation [{strategy_name}]")
        print("-" * 40)
        nice_log(f"Starting distillation for {epochs} epoch(s)...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            student.train()
            optimizer.zero_grad()

            for i, (x, y) in enumerate(tqdm(train_data)):
                x = x.to(device)
                y = y.to(device)
                bytes_processed += (x.shape[0] * x.shape[1]) // 8

                while (
                    next_checkpoint_idx < len(evaluation_checkpoints)
                    and bytes_processed >= evaluation_checkpoints[next_checkpoint_idx]
                ):
                    eval_results = module.evaluate_model(
                        student, config, eval_data, device, target_bits=target_bits
                    )
                    partial_evals.append(
                        {"eval": eval_results, "bytes_processed_eval": bytes_processed}
                    )
                    next_checkpoint_idx += 1
                    time.sleep(60)

                if (
                    torch.isnan(x).any()
                    or torch.isinf(x).any()
                    or torch.isnan(y).any()
                    or torch.isinf(y).any()
                ):
                    raise ValueError("Invalid values found in input data")

                with torch.amp.autocast("cuda"):
                    loss = self.strategy.compute_loss(
                        student, teacher, x, y, config, device
                    )

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_data), 1)
            nice_log(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        training_time = float(timer() - start) / 60
        nice_log(f"Distillation completed in {training_time:.2f} minutes")
        return training_time, partial_evals

    def run(self, module, run_params):
        target_bits = run_params["target_bits"]
        model_size_params = run_params["model_size_parameters"]
        evaluation_checkpoints = run_params.get("evaluation_checkpoints", [])

        config = self._build_config(module, run_params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nice_log(f"Using device: {device}")

        train_data, eval_data = module.load_and_prepare_data(config)
        steps_per_epoch = len(train_data)
        validation_steps = len(eval_data)

        # Teacher
        nice_log("Phase 1: Training teacher...")
        teacher = module.build_model(
            config, **model_size_params, target_bits=target_bits
        )
        model_parameters = module.log_model_parameters(teacher)

        teacher_time, _ = module.train_model(
            teacher,
            config,
            train_data,
            device,
            evaluation_checkpoints,
            eval_data,
            target_bits=target_bits,
        )
        module.save_model(teacher, config["weights_path"])

        teacher_eval = module.evaluate_model(
            teacher, config, eval_data, device, target_bits=target_bits
        )
        nice_log(
            f"Teacher p_ml: {teacher_eval['p_ml']:.5f}, "
            f"CE loss: {teacher_eval['bin_cross-entropy_loss']:.5f}"
        )

        # Student
        nice_log("Phase 2: Distilling student...")
        student = module.build_model(
            config, **model_size_params, target_bits=target_bits
        )

        distill_time, distill_evals = self._train_student(
            module,
            student,
            teacher,
            config,
            train_data,
            eval_data,
            device,
            evaluation_checkpoints,
            target_bits,
        )

        student_weights = config["weights_path"].replace(".pth", "_student.pth")
        module.save_model(student, student_weights)

        student_eval = module.evaluate_model(
            student, config, eval_data, device, target_bits=target_bits
        )
        distill_evals.append(
            {"eval": student_eval, "bytes_processed_eval": run_params["num_bytes"]}
        )
        nice_log(
            f"Student p_ml: {student_eval['p_ml']:.5f}, "
            f"CE loss: {student_eval['bin_cross-entropy_loss']:.5f}"
        )

        total_train_samples = (
            int(run_params["num_bytes"] * run_params["train_ratio"])
            - int(np.ceil(run_params["seqlen"] / 8))
        ) // run_params["step"]

        return {
            "training_time": teacher_time + distill_time,
            "teacher_training_time": teacher_time,
            "distillation_training_time": distill_time,
            "distillation_strategy": self.strategy.__class__.__name__,
            "eval_results": distill_evals,
            "teacher_eval": teacher_eval,
            "total_parameters": model_parameters[0],
            "trainable_parameters": model_parameters[1],
            "non_trainable_parameters": model_parameters[2],
            "total_train_samples": total_train_samples,
            "training_data_size": total_train_samples * run_params["seqlen"],
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
        }
