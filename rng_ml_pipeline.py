import os
import csv
import time
import argparse
import itertools
import subprocess
import numpy as np

from utils.nice_log import nice_log
from gbarp_gen.python import (
    gbAR,
    point_to_point_alpha,
    constant_alpha,
    exponentially_decreasing_alpha,
    gaussian_alpha,
)
from entropy_limits import ar_min_entropy_limit
from parsers.entropy_parsers import parse_entropy_output

OUTPUT_FILE_PATH = "./results"
ENTROPY_TEST_BINARY = "./SP800-90B_EntropyAssessment/cpp/ea_non_iid"
MIN_EVAL_ORDER = 4
EVALS_PER_ORDER = 2


class NistEntropyAssessment:
    """Runs NIST SP800-90B entropy assessment as a background process."""
    
    def __init__(self, sample_file, binary_path=ENTROPY_TEST_BINARY):
        self.sample_file = sample_file
        self.binary_path = binary_path
        self.command = [binary_path, "-a", "-v", sample_file]
        self._process = None
    
    def start(self):
        self._process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return self
    
    def wait(self):
        if self._process is None:
            raise RuntimeError("Must call start() before wait()")
        
        stdout, stderr = self._process.communicate()
        
        if self._process.returncode != 0:
            raise subprocess.CalledProcessError(
                self._process.returncode, 
                self.command, 
                output=stdout, 
                stderr=stderr
            )
        
        print("-" * 40)
        print("NIST SP800-90B Assessment")
        print("-" * 40)
        print(stdout)
        return parse_entropy_output(stdout)


def calculate_p_c(random_bytes, num_bytes=10**4):
    """Compute max bit bias: max(P(0), P(1))."""
    random_bits = np.unpackbits(np.frombuffer(random_bytes[:num_bytes], dtype=np.uint8))
    p_zeroes = np.sum(random_bits == 0) / len(random_bits)
    return max(p_zeroes, 1 - p_zeroes)


def experimental_min_entropy(p_ml, target_bits=1):
    return -np.log2(p_ml) / target_bits


def write_results_to_csv(output_dict, results_dir):
    output_file = f"{results_dir}/results.csv"
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(output_dict.keys())
        writer.writerow(output_dict.values())


def save_random_data(data, data_target_file, sample_target_file, sample_size=10**7):
    """Save full data and a sample for NIST assessment."""
    with open(data_target_file, "wb") as f:
        f.write(data)
    with open(sample_target_file, "wb") as f:
        f.write(data[:sample_size])


def check_test_to_classes_ratio(
    num_bytes, test_ratio, train_ratio, target_bits, seqlen, step
):
    threshold_ratio = 1
    total_elements = num_bytes

    train_sequences = (
        int(total_elements * train_ratio) - int(np.ceil(seqlen / 8))
    ) // step
    test_sequences = (test_ratio / train_ratio) * train_sequences
    test_to_classes_ratio = test_sequences / (2**target_bits)

    if test_to_classes_ratio < threshold_ratio:
        required_num_bytes = threshold_ratio * (2**target_bits) * step * (
            train_ratio / test_ratio
        ) + int(np.ceil(seqlen / 8))
        required_num_bytes = int(np.ceil(required_num_bytes))
        raise ValueError(
            f"Test to classes ratio must be greater than 1. Increase the number of bytes to at least {required_num_bytes}."
        )


def generate_gbAR_random_bytes(alpha_scaling_factor, data_param_dict, beta, num_bytes):
    distance_scale_p = data_param_dict["distance_scale_p"]
    autocorrelation_function = data_param_dict["autocorrelation_function"]
    if autocorrelation_function == "point-to-point":
        alpha = point_to_point_alpha(distance_scale_p, alpha_scaling_factor)
    elif autocorrelation_function == "constant":
        alpha = constant_alpha(distance_scale_p, alpha_scaling_factor)
    elif autocorrelation_function == "exponential":
        alpha = exponentially_decreasing_alpha(
            distance_scale_p,
            alpha_scaling_factor,
            decay_rate=data_param_dict["exponential_decay_rate"],
        )
    elif autocorrelation_function == "gaussian":
        alpha = gaussian_alpha(
            distance_scale_p,
            alpha_scaling_factor,
            data_param_dict["gaussian_sigma"],
            0.001,
        )
    elif "constant_" in autocorrelation_function:
        signs = data_param_dict["signs"]
        alpha = constant_alpha(distance_scale_p, alpha_scaling_factor, signs=signs)
    else:
        raise ValueError("Unknown autocorrelation function.")

    assert beta >= 0
    assert np.sum(np.abs(alpha)) + beta - 1 < 1e-10
    return gbAR(alpha, beta, num_bytes), alpha


def generate_evaluation_checkpoints(start_order, end_order, num_points_per_order=2):
    evaluation_checkpoints = np.concatenate(
        [
            np.logspace(
                order, order + 1, num=num_points_per_order, endpoint=False
            ).astype(int)
            for order in range(start_order, end_order)
        ]
    )

    return evaluation_checkpoints


class ModelRunner:
    """Runs ML models (GPT-2, nanoGPT, or RCNN) with model-specific configurations."""
    
    MODEL_DEFAULTS = {
        "gpt2": {
            "batch_size": 8,
            "model_size_parameters": lambda p: {
                "n_positions": p["seqlen"],
                "n_ctx": p["seqlen"],
                "n_embd": 256,   # embedding dimension
                "n_layer": 3,    # transformer layers
                "n_head": 4,     # attention heads
            },
            "remove_keys": [],
        },
        "nanogpt": {
            "batch_size": 8,
            "model_size_parameters": lambda p: {
                "block_size": p["seqlen"],
                "n_embd": 256,   # embedding dimension
                "n_layer": 3,    # transformer layers
                "n_head": 4,     # attention heads
                "dropout": 0.0,
            },
            "remove_keys": [],
        },
        "rcnn": {
            "batch_size": 2 * 10**3,
            "model_size_parameters": lambda p: {"scale_factor": 1},
            "remove_keys": ["is_autoregressive", "evaluate_all_bits"],
        },
    }
    
    def __init__(self, model_name):
        if model_name not in self.MODEL_DEFAULTS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_DEFAULTS.keys())}")
        self.model_name = model_name
        self.config = self.MODEL_DEFAULTS[model_name]
        self._module = None
    
    def _load_module(self):
        if self._module is None:
            if self.model_name == "gpt2":
                from models.gpt2 import rng_gpt2 as module
            elif self.model_name == "nanogpt":
                from models.nanogpt import rng_nanogpt as module
            elif self.model_name == "rcnn":
                from models.rcnn import rng_rcnn as module
            self._module = module
        return self._module
    
    def run(self, params, distillation_mode=None, distillation_config=None):
        module = self._load_module()
        run_params = params.copy()
        
        for key in self.config["remove_keys"]:
            run_params.pop(key, None)
        
        if run_params.get("batch_size") is None:
            run_params["batch_size"] = self.config["batch_size"]
        
        run_params["model_size_parameters"] = self.config["model_size_parameters"](run_params)
        
        if distillation_mode is not None:
            from distillation import get_strategy, DistillationTrainer
            strategy = get_strategy(distillation_mode, **(distillation_config or {}))
            trainer = DistillationTrainer(strategy, distillation_config)
            return trainer.run(module, run_params)
        
        return module.main(**run_params)


def main(model_param_dict, data_param_dict, model_name, hardware,
         gpu_cooldown=0, distillation_mode=None, distillation_config=None):
    print("=" * 60)
    nice_log(f"Running model [{model_name}]", color="green")
    print(f"  Data: {model_param_dict['num_bytes']} bytes, "
          f"target_bits={data_param_dict['target_bits']}, "
          f"corr_intensities={data_param_dict['corr_intensities']}")
    print(f"  Model params: seqlen={model_param_dict['seqlen']}, "
          f"step={model_param_dict['step']}, "
          f"batch_size={model_param_dict['batch_size']}, "
          f"epochs={model_param_dict['epochs']}, "
          f"lr={model_param_dict['learning_rate']}")
    print(f"  Training: train_ratio={model_param_dict['train_ratio']}, "
          f"is_autoregressive={model_param_dict['is_autoregressive']}, "
          f"evaluate_all_bits={model_param_dict['evaluate_all_bits']}")
    print("=" * 60)
    
    results_dir = f"{OUTPUT_FILE_PATH}/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    data_target_file = f"{results_dir}/random_bytes.bin"
    sample_target_file = f"{results_dir}/random_bytes_sample.bin"
    model_param_dict["filename"] = data_target_file
    model_runner = ModelRunner(model_name)
    total_runs = len(data_param_dict["target_bits"]) * len(data_param_dict["corr_intensities"])
    
    for target_bits, corr_intensity in itertools.product(
        data_param_dict["target_bits"], data_param_dict["corr_intensities"]
    ):
        model_param_dict["target_bits"] = target_bits
        alpha_scaling_factor = corr_intensity
        beta = 1 - alpha_scaling_factor
        random_bytes, alpha = generate_gbAR_random_bytes(
            alpha_scaling_factor,
            data_param_dict,
            beta,
            model_param_dict["num_bytes"],
        )
        save_random_data(random_bytes, data_target_file, sample_target_file)
        
        p_c_source = calculate_p_c(random_bytes)
        min_entropy_th = ar_min_entropy_limit(beta)
        
        nist = NistEntropyAssessment(sample_target_file).start()
        ml_results = model_runner.run(
            model_param_dict,
            distillation_mode=distillation_mode,
            distillation_config=distillation_config,
        )
        entropies_dict = nist.wait()

        base_result = {
            "model": model_name,
            "nn_info_unit": "bit",
            "hardware": hardware,
            "is_autoregressive": model_param_dict["is_autoregressive"],
            "evaluate_all_bits": model_param_dict["evaluate_all_bits"],
            "num_bytes": model_param_dict["num_bytes"],
            "target_bits": target_bits,
            "seqlen": model_param_dict["seqlen"],
            "step": model_param_dict["step"],
            "train_ratio": model_param_dict["train_ratio"],
            "learning_rate": model_param_dict["learning_rate"],
            "batch_size": model_param_dict["batch_size"],
            "epochs": model_param_dict["epochs"],
            "corr_intensity": f"{corr_intensity:.3f}",
            "autocorrelation_function": data_param_dict["autocorrelation_function"],
            "distance_scale_p": data_param_dict["distance_scale_p"],
            "exponential_decay_rate": data_param_dict["exponential_decay_rate"],
            "gaussian_sigma": data_param_dict["gaussian_sigma"],
            "p_c_source": p_c_source,
            "min_entropy_th": min_entropy_th,
            "distillation_mode": distillation_mode or "-",
            **entropies_dict,
        }

        ml_info = {k: v for k, v in ml_results.items()
                   if k not in ("eval_results", "teacher_eval")}

        for partial_eval in ml_results["eval_results"]:
            eval_result = partial_eval["eval"]
            training_time = ml_info.get("training_time")
            evaluation_time = eval_result.get("evaluation_time")

            output_dict = {
                **base_result,
                **ml_info,
                **eval_result,
                "training_time": f"{training_time:.2f}" if training_time is not None else "-",
                "evaluation_time": f"{evaluation_time:.2f}" if evaluation_time is not None else "-",
                "bytes_processed_eval": partial_eval["bytes_processed_eval"],
                "min_entropy_estimated": experimental_min_entropy(eval_result["p_ml"], target_bits),
            }

            # Include teacher metrics when in distillation mode
            teacher_eval = ml_results.get("teacher_eval")
            if teacher_eval is not None:
                output_dict["teacher_p_ml"] = teacher_eval["p_ml"]
                output_dict["teacher_ce_loss"] = teacher_eval["bin_cross-entropy_loss"]

            write_results_to_csv(output_dict, results_dir)

        os.remove(data_target_file)
        os.remove(sample_target_file)

        if gpu_cooldown > 0 and total_runs > 1:
            time.sleep(gpu_cooldown)

    print("=" * 60)
    nice_log(f"Finished model [{model_name}]", color="green")
    print("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the main function with custom parameters."
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Model name (default: gpt2)"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Hardware being used (for example: RTX3060Ti or g5.xlarge)",
    )
    parser.add_argument(
        "--corr_intensities",
        type=float,
        nargs="+",
        default=None,
        help="Correlation intensities (default: None)",
    )
    parser.add_argument(
        "--num_bytes",
        type=int,
        default=100000,
        help="Number of bytes (default: 100000)",
    )
    parser.add_argument(
        "--target_bits", type=int, default=None, nargs="+", help="Target bits"
    )
    parser.add_argument(
        "--seqlen", type=int, default=100, help="Max length in bits (default: 100)"
    )
    parser.add_argument("--step", type=int, default=None, help="Step (default: None)")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train ratio (default: 0.8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (default: 10000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs (default: 1)"
    )
    parser.add_argument(
        "--distance_scale_p", type=int, default=1, help="Distance scale p (default: 1)"
    )
    parser.add_argument(
        "--autocorrelation_function",
        type=str,
        default="point-to-point",
        help="Correlation law (default: point-to-point) (exponential, gaussian, point-to-point)",
    )
    parser.add_argument(
        "--signs", type=int, default=None, nargs="+", help="Signs (default: None)"
    )
    parser.add_argument(
        "--is_autoregressive",
        action="store_true",
        help="Enable autoregressive mode (default: False)",
    )
    parser.add_argument(
        "--evaluate_all_bits",
        action="store_true",
        help="Evaluate all bits (default: False)",
    )
    parser.add_argument(
        "--gpu_cooldown",
        type=int,
        default=0,
        help="Seconds to wait between runs for GPU cooldown (default: 0, use 180 for production)",
    )
    parser.add_argument(
        "--distillation_mode",
        type=str,
        default=None,
        choices=["rad", "vad", "irbc"],
        help="Distillation strategy: rad (REINFORCE), vad (Gumbel-Softmax), irbc (Behaviour Cloning)",
    )
    parser.add_argument(
        "--distillation_steps",
        type=int,
        default=5,
        help="Number of autoregressive rollout steps for distillation (default: 5)",
    )
    parser.add_argument(
        "--distillation_lr",
        type=float,
        default=None,
        help="Learning rate for student distillation (default: same as --learning_rate)",
    )
    parser.add_argument(
        "--distillation_epochs",
        type=int,
        default=None,
        help="Epochs for student distillation (default: same as --epochs)",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.hardware is None:
        raise ValueError(
            "Hardware must be specified (for example: RTX3060Ti or g5.xlarge)"
        )

    if args.num_bytes < 10**5:
        raise ValueError("Number of bytes must be at least 10**5 bytes")

    if args.step is None:
        step = args.seqlen
    else:
        step = args.step

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("Train ratio must be greater than 0 and lower than 1")

    if args.autocorrelation_function not in [
        "exponential",
        "gaussian",
        "point-to-point",
        "constant",
    ]:
        raise ValueError("Unknown autocorrelation function.")

    if args.corr_intensities is None:
        corr_intensities = np.linspace(0, 0.99, num=10)
    else:
        for corr_intensity in args.corr_intensities:
            if corr_intensity < 0 or corr_intensity > 1:
                raise ValueError("Correlation intensity must be between 0 and 1")
        corr_intensities = args.corr_intensities

    if args.target_bits is None:
        target_bits = [1]
    else:
        for target_bit in args.target_bits:
            if target_bit < 1 or (target_bit > args.seqlen - 1):
                raise ValueError("target_bits must be between 1 and seqlen - 1")
        target_bits = args.target_bits

    exponential_decay_rate = "-"
    gaussian_sigma = "-"
    if args.autocorrelation_function == "exponential":
        exponential_decay_rate = 1 / 10
    elif args.autocorrelation_function == "gaussian":
        gaussian_sigma = args.distance_scale_p / 100

    if args.autocorrelation_function == "constant" and args.signs is not None:
        if len(args.signs) != args.distance_scale_p:
            raise ValueError("Number of signs must be equal to the distance scale p")
        for sign in args.signs:
            if sign == 1:
                args.autocorrelation_function += "_+"
            elif sign == -1:
                args.autocorrelation_function += "_-"
            else:
                raise ValueError("Signs must be either 1 or -1")

    for target_bit_n in target_bits:
        check_test_to_classes_ratio(
            args.num_bytes,
            1 - args.train_ratio,
            args.train_ratio,
            target_bit_n,
            args.seqlen,
            step,
        )

    upper_order = int(np.floor(np.log10(1000000)))
    evaluation_checkpoints = []

    model_param_dict = {
        "generator": f"ar_{args.num_bytes}",
        "num_bytes": int(args.num_bytes),
        "seqlen": args.seqlen,
        "step": step,
        "train_ratio": args.train_ratio,
        "test_ratio": 1 - args.train_ratio,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "evaluation_checkpoints": evaluation_checkpoints,
        "is_autoregressive": args.is_autoregressive,
        "evaluate_all_bits": args.evaluate_all_bits,
    }

    data_param_dict = {
        "num_bytes": args.num_bytes,
        "target_bits": target_bits,
        "corr_intensities": corr_intensities,
        "distance_scale_p": args.distance_scale_p,
        "autocorrelation_function": args.autocorrelation_function,
        "signs": args.signs,
        "exponential_decay_rate": exponential_decay_rate,
        "gaussian_sigma": gaussian_sigma,
    }

    # Build distillation config from CLI args (only if distillation_mode is set)
    distillation_config = None
    if args.distillation_mode is not None:
        distillation_config = {
            "num_steps": args.distillation_steps,
        }
        if args.distillation_lr is not None:
            distillation_config["learning_rate"] = args.distillation_lr
        if args.distillation_epochs is not None:
            distillation_config["epochs"] = args.distillation_epochs

    main(
        model_param_dict,
        data_param_dict,
        args.model_name,
        args.hardware,
        gpu_cooldown=args.gpu_cooldown,
        distillation_mode=args.distillation_mode,
        distillation_config=distillation_config,
    )
