"""pytest tests/test_data.py"""
import numpy as np
import torch
from minigpt.data import get_batch


def test_shapes_dtype_and_shift():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, block_size=16, batch_size=8)
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted by one token"
    assert torch.equal(y, x + 1), "with data = arange, each row is a contiguous slice"


def test_positions_in_bounds():
    data = np.arange(20, dtype=np.uint16)
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        x, y = get_batch(data, block_size=19, batch_size=4, generator=g)
        assert int(y.max()) <= 19 and int(x.min()) >= 0


def test_generator_is_reproducible():
    data = np.arange(500, dtype=np.uint16)
    a = get_batch(data, 8, 4, generator=torch.Generator().manual_seed(3))[0]
    b = get_batch(data, 8, 4, generator=torch.Generator().manual_seed(3))[0]
    assert torch.equal(a, b)
