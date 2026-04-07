"""
Inference functions for nanoGPT model.

Provides three inference modes:
- binary_inference: Single bit prediction (target_bits=1)
- multitoken_inference: Multi-bit token prediction
- autoregressive_inference: Sequential bit prediction for multi-bit targets
"""

import torch


def binary_inference(model, x, y, loss_fn, correct, total):
    """
    Binary inference for single-bit prediction.
    
    Args:
        model: nanoGPT model (wrapper with .logits interface)
        x: Input tensor (B, T)
        y: Target tensor (B, T)
        loss_fn: Loss function
        correct: Running count of correct predictions
        total: Running count of total predictions
    
    Returns:
        binary_predictions: Predicted bits
        loss: Cross-entropy loss
        correct: Updated correct count
        total: Updated total count
    """
    logits = model(x).logits
    probs = torch.softmax(logits, dim=-1)
    predicted = torch.argmax(probs, dim=-1)
    
    # Flatten for CrossEntropyLoss
    loss = loss_fn(logits.view(-1, 2), y.view(-1))
    
    correct += (predicted == y).sum().item()
    total += y.numel()

    binary_predictions = predicted.cpu()
    return binary_predictions, loss, correct, total


def multitoken_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, tokenizer
):
    """
    Multi-token inference for multi-bit prediction.
    
    Args:
        model: nanoGPT model
        x: Input tensor (B, T)
        y: Target tensor (B, T) with class indices
        target_bits: Number of target bits
        loss_fn: Loss function
        correct: Running correct count
        total: Running total count
        config: Model configuration
        tokenizer: NBitsTokenizer for detokenization
    
    Returns:
        binary_predictions: Predicted bits
        loss: Cross-entropy loss
        correct: Updated correct count
        total: Updated total count
    """
    logits = model(x).logits
    target = y.view(-1)
    
    # Flatten logits for loss calculation
    flattened_logits = logits.view(-1, 2**target_bits)
    loss = loss_fn(flattened_logits, target)

    if config["evaluate_all_bits"]:
        predicted = flattened_logits.argmax(dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        binary_predictions = tokenizer.detokenize(predicted.cpu().tolist())
    else:
        # Evaluate only the last token's logits for each batch item
        last_logits = logits[:, -1, :]
        last_predicted = last_logits.argmax(dim=1)
        last_target = y[:, -1]
        
        correct += (last_predicted == last_target).sum().item()
        total += x.size(0)
        
        binary_predictions = tokenizer.detokenize(last_predicted.cpu().tolist())

    binary_predictions = torch.tensor(
        [int(bit) for bit in binary_predictions], dtype=torch.int32
    )

    return binary_predictions, loss, correct, total


def eval_multi(predictions, target, target_bits, correct, total, config):
    """Helper function for multi-bit evaluation."""
    if config["evaluate_all_bits"]:
        correct += (predictions == target).sum().item()
        total += target.numel()
    else:
        correct += (predictions[:, -1] == target[:, -1]).sum().item()
        total += target.size(0)
        
    return correct, total


def autoregressive_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, device
):
    """
    Autoregressive inference for multi-bit prediction.
    
    Generates bits one at a time, feeding each prediction back as input.
    
    Args:
        model: nanoGPT model
        x: Input tensor (B, T)
        y: Target tensor (B, T) with class indices
        target_bits: Number of bits to generate
        loss_fn: Loss function
        correct: Running correct count
        total: Running total count
        config: Model configuration
        device: Torch device
    
    Returns:
        binary_predictions: Predicted bits
        loss: Dummy loss (0.0 since we can't compute true cross-entropy for hard predictions)
        correct: Updated correct count
        total: Updated total count
    """
    target = y
    x_current = x.to(device)
    predictions_sequence = []

    for step in range(target_bits):
        logits = model(x_current).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_bits = torch.argmax(probs, dim=-1)

        # Shift sequence: remove first element, append new prediction
        x_current = torch.cat([x_current[:, 1:], predicted_bits[:, -1:]], dim=1)
        
        predicted_bits_expanded = predicted_bits.unsqueeze(-1)
        predictions_sequence.append(predicted_bits_expanded)

    concatenated_bits = torch.cat(predictions_sequence, dim=-1)
    
    # Convert bit sequence to decimal indices
    decimal_predictions = torch.sum(
        concatenated_bits
        * 2 ** torch.arange(target_bits - 1, -1, -1, device=device).unsqueeze(0).unsqueeze(0),
        dim=-1,
    )
    
    # One-hot encode predictions
    predictions = torch.nn.functional.one_hot(
        decimal_predictions, num_classes=2**target_bits
    ).to(torch.float)
    
    # Return dummy loss since we can't compute true cross-entropy for hard predictions
    loss = torch.tensor(0.0, device=device)

    correct, total = eval_multi(
        decimal_predictions, target, target_bits, correct, total, config
    )

    binary_predictions = concatenated_bits.view(-1).cpu()

    return binary_predictions, loss, correct, total


if __name__ == "__main__":
    print("This is a module with inference mechanisms for the nanoGPT model.")
