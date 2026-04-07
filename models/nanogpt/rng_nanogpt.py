"""
nanoGPT model for RNG min-entropy estimation.

This module implements binary sequence prediction using a minimal GPT architecture
based on Andrej Karpathy's nanoGPT. It follows the same interface as the GPT-2
model for seamless integration with the pipeline.

Key features:
- Minimal transformer architecture optimized for binary sequences
- Flash Attention support (PyTorch >= 2.0)
- Mixed precision training with torch.amp.autocast
- Memory-mapped file support for large datasets
- Compatible with the existing evaluation pipeline
"""

# Standard library imports
import os
import sys
import time
from timeit import default_timer as timer

# Related third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Local application/library specific imports
# wrapper.py provides our interface around official nanoGPT
from .wrapper import NanoGPTWrapper, create_nanogpt_config
from .argparser.argparser import parse_arguments
from .data_proc.data_proc import load_and_prepare_data, NBitsTokenizer
from .inference.inference import (
    autoregressive_inference,
    multitoken_inference,
    binary_inference,
)
from .aux.aux import binary_entropy, log_model_parameters, get_config

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log

MODEL_NAME = "nanogpt"


def save_model(model, filename):
    """Save model state dict."""
    torch.save(model.state_dict(), filename)


def build_model(
    config,
    block_size=512,
    n_embd=256,
    n_layer=6,
    n_head=4,
    dropout=0.0,
    target_bits=1,
):
    """
    Build nanoGPT model with given configuration.
    
    Args:
        config: Training configuration dictionary
        block_size: Maximum sequence length (context window)
        n_embd: Embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        dropout: Dropout rate
        target_bits: Number of target bits for vocabulary size
    
    Returns:
        NanoGPTWrapper model instance
    """
    if config["is_autoregressive"]:
        vocab_size = 2
    else:
        vocab_size = 2**target_bits

    gpt_config = create_nanogpt_config(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True,
    )
    
    model = NanoGPTWrapper(gpt_config)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    return model


def train_model(
    model,
    config,
    data,
    device,
    evaluation_checkpoints,
    eval_data,
    target_bits=1,
    accumulation_steps=4,
):
    """
    Train the nanoGPT model.
    
    Uses:
    - torch.amp.autocast for mixed precision training
    - torch.amp.GradScaler for gradient scaling
    - Gradient accumulation for effective larger batch sizes
    - Gradient clipping for training stability
    
    Args:
        model: nanoGPT model
        config: Training configuration
        data: Training DataLoader
        device: Torch device
        evaluation_checkpoints: List of byte counts for intermediate evaluations
        eval_data: Evaluation DataLoader
        target_bits: Number of target bits
        accumulation_steps: Number of steps for gradient accumulation
    
    Returns:
        training_time: Time taken for training in minutes
        partial_evals: List of evaluation results at checkpoints
    """
    bytes_processed = 0
    next_checkpoint_idx = 0
    partial_evals = []
    start = timer()
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    if config["is_autoregressive"]:
        target_bits = 1
        
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    print("-" * 40)
    print("Training")
    print("-" * 40)
    nice_log(f"Starting training for {config['epochs']} epoch(s)...")
    model.train()

    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, (x, y) in enumerate(tqdm(data)):
            x = x.to(device)
            y = y.to(device)

            number_of_bytes_in_batch = (x.shape[0] * x.shape[1]) // 8
            bytes_processed += number_of_bytes_in_batch

            # Intermediate evaluation checkpoints
            while (
                next_checkpoint_idx < len(evaluation_checkpoints)
                and bytes_processed >= evaluation_checkpoints[next_checkpoint_idx]
            ):
                eval_results = evaluate_model(
                    model, config, eval_data, device, target_bits=config["target_bits"]
                )
                checkpoint_data = {
                    "eval": eval_results,
                    "bytes_processed_eval": bytes_processed,
                }
                partial_evals.append(checkpoint_data)
                next_checkpoint_idx += 1
                time.sleep(60)

            # Input validation
            if (
                torch.isnan(x).any()
                or torch.isinf(x).any()
                or torch.isnan(y).any()
                or torch.isinf(y).any()
            ):
                raise Exception("Invalid values found in input data")

            # Mixed precision training
            with torch.amp.autocast("cuda"):
                output = model(x)
                logits = output.logits.view(-1, 2**target_bits)
                target = y.view(-1)
                loss = loss_fn(logits, target)

            # Gradient scaling for mixed precision
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

    training_time = float(timer() - start) / 60
    nice_log(f"Training completed in {training_time:.2f} minutes")
    return training_time, partial_evals


def evaluate_model(model, config, data, device, target_bits=1):
    """
    Evaluate the nanoGPT model.
    
    Args:
        model: nanoGPT model
        config: Model configuration
        data: Evaluation DataLoader
        device: Torch device
        target_bits: Number of target bits
    
    Returns:
        Dictionary with evaluation metrics:
        - evaluation_time: Time taken
        - p_ml: ML predictor accuracy
        - p_g: Random guessing probability
        - p_c_pred: Bit bias in predictions
        - p_e: Entropy of predictions
        - bin_cross-entropy_loss: Average cross-entropy loss
    """
    start = timer()
    model.eval()

    total = correct = 0
    total_cross_entropy = 0
    all_binary_predictions = []

    loss_fn = torch.nn.CrossEntropyLoss()
    if target_bits > 1:
        tokenizer = NBitsTokenizer(n_bits=target_bits)

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(data)):
            x = x.to(device)
            y = y.to(device)
            
            if target_bits == 1:
                binary_predictions, loss, correct, total = binary_inference(
                    model, x, y, loss_fn, correct, total
                )
            else:
                if config["is_autoregressive"]:
                    binary_predictions, loss, correct, total = autoregressive_inference(
                        model, x, y, target_bits, loss_fn, correct, total, config, device
                    )
                else:
                    binary_predictions, loss, correct, total = multitoken_inference(
                        model, x, y, target_bits, loss_fn, correct, total, config, tokenizer
                    )

            all_binary_predictions.append(binary_predictions)
            total_cross_entropy += loss.item()

    concatenated_binary_predictions = torch.cat(all_binary_predictions, dim=0)
    
    # Calculate bit bias on predictions
    n_zeroes = (concatenated_binary_predictions == 0).sum().item()
    total_bits = concatenated_binary_predictions.numel()
    p_c_zeroes = n_zeroes / total_bits
    p_c_pred = max(p_c_zeroes, 1 - p_c_zeroes)
    
    p_zeroes = n_zeroes / total_bits
    overall_entropy = binary_entropy(p_zeroes)

    average_cross_entropy = total_cross_entropy / len(data)

    p_ml = correct / total
    p_g = 1 / (2**target_bits)

    evaluation_time = float(timer() - start) / 60

    results = {
        "evaluation_time": evaluation_time,
        "p_ml": p_ml,
        "p_g": p_g,
        "p_c_pred": p_c_pred,
        "p_e": overall_entropy,
        "bin_cross-entropy_loss": average_cross_entropy,
    }
    nice_log(
        f"Evaluation completed in {evaluation_time:.2f} minutes - p_ml: {p_ml:.5f}, p_g: {p_g:.5f}, p_c_pred: {p_c_pred:.5f}, "
        f"Predictions Entropy: {overall_entropy:.5f}, Cross-Entropy Loss: {average_cross_entropy:.5f}"
    )

    return results


def main(
    filename,
    generator,
    seqlen,
    step,
    num_bytes,
    target_bits,
    train_ratio,
    test_ratio,
    learning_rate,
    batch_size,
    epochs,
    is_autoregressive,
    evaluate_all_bits,
    model_size_parameters,
    evaluation_checkpoints=[],
):
    """
    Main entry point for nanoGPT training and evaluation.
    
    Args:
        filename: Path to binary data file
        generator: Generator name for output paths
        seqlen: Sequence length in bits
        step: Step size for sliding window
        num_bytes: Total bytes to process
        target_bits: Number of bits to predict
        train_ratio: Training data ratio
        test_ratio: Test data ratio
        learning_rate: Learning rate for optimizer
        batch_size: Batch size
        epochs: Number of training epochs
        is_autoregressive: Use autoregressive mode
        evaluate_all_bits: Evaluate all bits vs last only
        model_size_parameters: Dict with block_size, n_embd, n_layer, n_head
        evaluation_checkpoints: Byte counts for intermediate evaluations
    
    Returns:
        Dictionary with training results and metrics
    """
    config = get_config(
        MODEL_NAME,
        filename,
        generator,
        seqlen,
        step,
        num_bytes,
        target_bits,
        train_ratio,
        test_ratio,
        learning_rate,
        batch_size,
        epochs,
        is_autoregressive,
        evaluate_all_bits,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nice_log(f"Using device: {device}")

    # Load and prepare data
    train_data, eval_data = load_and_prepare_data(config)
    steps_per_epoch = len(train_data)
    validation_steps = len(eval_data)

    # Build model
    model = build_model(config, **model_size_parameters, target_bits=target_bits)
    model_parameters = log_model_parameters(model)

    # Train model
    training_time, partial_evals = train_model(
        model,
        config,
        train_data,
        device,
        evaluation_checkpoints,
        eval_data,
        target_bits=target_bits,
    )
    
    # Save model
    save_model(model, config["weights_path"])
    
    # Final evaluation
    print("-" * 40)
    print("Evaluation")
    print("-" * 40)
    nice_log("Starting evaluation...")
    eval_results = evaluate_model(
        model, config, eval_data, device, target_bits=target_bits
    )
    final_checkpoint_data = {
        "eval": eval_results,
        "bytes_processed_eval": num_bytes,
    }
    partial_evals.append(final_checkpoint_data)

    total_train_samples = (
        int(num_bytes * train_ratio) - int(np.ceil(seqlen / 8))
    ) // step
    
    output_dict = {
        "training_time": training_time,
        "eval_results": partial_evals,
        "total_parameters": model_parameters[0],
        "trainable_parameters": model_parameters[1],
        "non_trainable_parameters": model_parameters[2],
        "total_train_samples": total_train_samples,
        "training_data_size": total_train_samples * seqlen,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }

    return output_dict


if __name__ == "__main__":
    args = parse_arguments()

    # Default model size parameters for nanoGPT
    model_size_parameters = {
        "block_size": args.seqlen,
        "n_embd": 256,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.0,
    }

    results = main(
        args.filename,
        args.generator,
        args.seqlen,
        args.step,
        args.num_bytes,
        args.target_bits,
        args.train_ratio,
        args.test_ratio,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.is_autoregressive,
        args.evaluate_all_bits,
        model_size_parameters,
        evaluation_checkpoints=args.evaluation_checkpoints,
    )
