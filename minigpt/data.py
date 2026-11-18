"""Data: token ids stored as raw uint16 files, read with memmap, and random (x, y) batches."""
import numpy as np
import torch


def load_split(path):
    """Open a .bin file of uint16 ids without loading it into RAM."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, block_size, batch_size, device="cpu", generator=None):
    """Sample batch_size random windows from data (1D array of ids).

    Returns x, y of shape (batch_size, block_size), int64, with y shifted by one token:
    y[:, t] is the token that follows x[:, t]. Each window gives block_size
    next-token predictions at once thanks to the causal mask.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(np.asarray(data[i:i + block_size], dtype=np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64)) for i in ix.tolist()])
    if device.startswith("cuda"):
        # pinned memory + non_blocking lets the copy overlap with GPU work
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
