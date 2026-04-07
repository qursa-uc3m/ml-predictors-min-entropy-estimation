"""
Wrapper for official nanoGPT model.

This module provides a wrapper around Karpathy's nanoGPT implementation
to make it compatible with our training/evaluation pipeline interface.

The official nanoGPT model.py must be downloaded first using:
    ./installation_scripts/nanogpt_installation.sh
"""

import torch
import torch.nn as nn

# Import from official nanoGPT
from .model import GPT, GPTConfig


class NanoGPTOutput:
    """Simple output wrapper to provide .logits attribute for compatibility."""
    
    def __init__(self, logits):
        self.logits = logits


class NanoGPTWrapper(nn.Module):
    """
    Wrapper to provide HuggingFace-style interface for compatibility with existing code.
    
    This wrapper makes nanoGPT compatible with the existing inference functions
    that expect model(x).logits pattern (like GPT-2 from transformers).
    
    The official nanoGPT returns (logits, loss) tuple from forward(), and only
    computes logits for the last position during inference. This wrapper:
    1. Returns full sequence logits (not just last position)
    2. Wraps output in an object with .logits attribute
    """
    
    def __init__(self, config):
        super().__init__()
        self.gpt = GPT(config)
        self.config = config
    
    def forward(self, x):
        """
        Forward pass returning an object with .logits attribute.
        
        Args:
            x: Input tensor of shape (B, T)
        
        Returns:
            Object with logits attribute of shape (B, T, vocab_size)
        """
        device = x.device
        b, t = x.size()
        
        # Get embeddings and run through transformer
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.gpt.transformer.wte(x)
        pos_emb = self.gpt.transformer.wpe(pos)
        hidden = self.gpt.transformer.drop(tok_emb + pos_emb)
        
        for block in self.gpt.transformer.h:
            hidden = block(hidden)
        
        hidden = self.gpt.transformer.ln_f(hidden)
        
        # Get logits for ALL positions (not just last like in inference mode)
        logits = self.gpt.lm_head(hidden)
        
        return NanoGPTOutput(logits)
    
    def parameters(self, recurse=True):
        return self.gpt.parameters(recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        return self.gpt.named_parameters(prefix, recurse)
    
    def train(self, mode=True):
        self.gpt.train(mode)
        return self
    
    def eval(self):
        self.gpt.eval()
        return self
    
    def to(self, device):
        self.gpt.to(device)
        return self
    
    def state_dict(self):
        return self.gpt.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.gpt.load_state_dict(state_dict)


def create_nanogpt_config(
    block_size=512,
    vocab_size=2,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    bias=True,
):
    """
    Create a GPTConfig for binary sequence prediction.
    
    Args:
        block_size: Maximum sequence length (context window)
        vocab_size: Vocabulary size (2 for binary, 2^n for n-bit tokens)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout rate
        bias: Whether to use bias in Linear and LayerNorm layers
    
    Returns:
        GPTConfig instance
    """
    return GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )


if __name__ == "__main__":
    # Test the wrapper
    config = create_nanogpt_config(
        block_size=128,
        vocab_size=2,
        n_layer=3,
        n_head=4,
        n_embd=128,
    )
    
    model = NanoGPTWrapper(config)
    
    # Test forward pass
    x = torch.randint(0, 2, (4, 64))  # batch of 4, sequence length 64
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {output.logits.shape}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        x = x.cuda()
        output = model(x)
        print(f"CUDA output logits shape: {output.logits.shape}")
