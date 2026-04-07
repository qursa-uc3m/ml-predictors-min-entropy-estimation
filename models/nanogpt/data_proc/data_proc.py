"""
Data processing module for nanoGPT.

Reuses the BinaryDataset implementation from GPT-2 with memory-mapped file support
and optimized DataLoader configuration.
"""

import os
import mmap
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class BinaryTokenizer:
    """Simple binary (0/1) tokenizer."""
    
    def __init__(self):
        self.stoi = {"0": 0, "1": 1}
        self.itos = {0: "0", 1: "1"}

    def tokenize(self, binary_data):
        if isinstance(binary_data, str):
            return [self.stoi[b] for b in binary_data]
        return binary_data

    def detokenize(self, tokens):
        return "".join([self.itos[t] for t in tokens])


class NBitsTokenizer:
    """N-bits tokenizer for multi-bit targets."""
    
    def __init__(self, n_bits):
        self.target_bits = n_bits

    def _create_stoi_mapping(self, n_bits):
        max_val = 2**n_bits
        stoi = {"{:0{}b}".format(i, n_bits): i for i in range(max_val)}
        return stoi

    def tokenize(self, binary_data):
        assert (
            len(binary_data) % self.target_bits == 0
        ), "Length of binary_data must be multiple of n_bits."

        return [
            int(binary_data[i : i + self.target_bits], 2)
            for i in range(0, len(binary_data), self.target_bits)
        ]

    def detokenize(self, tokens):
        return "".join(
            [format(token, "0{}b".format(self.target_bits)) for token in tokens]
        )


class BinaryDataset(Dataset):
    """
    Dataset for binary sequence prediction with configurable target bits.
    Uses memory-mapped files for efficient data access.
    """
    
    def __init__(
        self, data_source, config, start_index=0, end_index=None, is_training=False
    ):
        self.data_source = data_source
        self.config = config
        self.start_index = start_index
        self.end_index = end_index
        self.is_training = is_training
        
        self.data = None
        self.file_handle = None
        
        if not isinstance(self.data_source, str):
             self.data = data_source
             if self.end_index is None:
                 self.end_index = len(self.data)
        
        if self.end_index is None:
             if self.data is not None:
                 self.end_index = len(self.data)
             else:
                 raise ValueError("end_index must be provided if data_source is a file path")

        # Calculate total bits available in this slice
        self.num_bits = (self.end_index - self.start_index) * 8
        
        if is_training and config["is_autoregressive"]:
            self.target_bits = 1
        else:
            self.target_bits = config["target_bits"]
        
        self.seqlen = config["seqlen"]
        self.step = config["step"]
        self.batch_size = config["batch_size"]
        self.num_classes = 2 ** self.target_bits
        self.powers = 2 ** np.arange(self.target_bits - 1, -1, -1)
        
        # Precompute target indices offset matrix
        self.target_offsets = np.arange(self.seqlen)[:, None] + np.arange(1, self.target_bits + 1)

    def _get_data(self):
        if self.data is None:
            if isinstance(self.data_source, str):
                self.file_handle = open(self.data_source, "rb")
                self.data = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                raise ValueError("Data source is not a file path and data is None")
        return self.data

    def __len__(self):
        total_steps = (self.num_bits - self.seqlen - self.target_bits) // self.step
        num_batches = total_steps // self.batch_size
        return max(0, num_batches * self.batch_size)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        
        # Global bit index relative to the start of this dataset slice
        bit_start_rel = idx * self.step
        bit_end_rel = bit_start_rel + self.seqlen + self.target_bits
        
        # Absolute bit index in the source data
        bit_start_abs = self.start_index * 8 + bit_start_rel
        bit_end_abs = self.start_index * 8 + bit_end_rel
        
        # Byte indices
        byte_start = bit_start_abs // 8
        byte_end = (bit_end_abs + 7) // 8
        
        # Read bytes
        data = self._get_data()
        data_slice = data[byte_start:byte_end]
        
        # Unpack
        np_data = np.frombuffer(data_slice, dtype=np.uint8)
        bits = np.unpackbits(np_data)
        
        # Extract exact bits
        offset = bit_start_abs % 8
        length = bit_end_rel - bit_start_rel
        bits_slice = bits[offset : offset + length]
        
        input_sequence = torch.from_numpy(bits_slice[:self.seqlen].astype(np.int64))
        
        target_bits_matrix = bits_slice[self.target_offsets]
        target_indices = np.dot(target_bits_matrix, self.powers)
        
        target_sequence = torch.from_numpy(target_indices.astype(np.int64))
        
        return input_sequence, target_sequence


def load_and_prepare_data(config, num_workers=4):
    """
    Load binary data and prepare DataLoaders for training and evaluation.
    
    Uses optimized DataLoader settings:
    - num_workers for parallel data loading
    - pin_memory for faster GPU transfers
    - persistent_workers to avoid worker restart overhead
    """
    file_path = config["filename"]
    file_size = os.path.getsize(file_path)
    
    num_bytes = config["num_bytes"]
    if num_bytes > file_size:
        num_bytes = file_size

    train_length = int(num_bytes * config["train_ratio"])

    train_dataset = BinaryDataset(
        file_path, config, start_index=0, end_index=train_length, is_training=True
    )
    eval_dataset = BinaryDataset(
        file_path, config, start_index=train_length, end_index=num_bytes
    )

    return (
        DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        ),
        DataLoader(
            eval_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        ),
    )


if __name__ == "__main__":
    print("This is a module for loading and preparing data for the nanoGPT model.")
