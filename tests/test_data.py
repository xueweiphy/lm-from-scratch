"""data_loading: shapes, targets offset by one, every legal start reachable and none beyond
(mirrors the CS336 A1 test_get_batch)."""

from collections import Counter

import numpy as np
import torch

from training import data_loading


def test_data_loading():
    dataset = np.arange(0, 100, dtype=np.uint16)
    context_length, batch_size = 7, 32
    starts = Counter()
    for _ in range(500):
        x, y = data_loading(dataset, batch_size, context_length, device="cpu")
        assert x.shape == y.shape == (batch_size, context_length)
        assert x.dtype == y.dtype == torch.long
        assert torch.equal(x + 1, y)
        starts.update(x[:, 0].tolist())
    assert min(starts) == 0
    assert max(starts) == len(dataset) - context_length - 1


def test_data_loading_memmap(tmp_path):
    path = tmp_path / "ids.npy"
    np.save(path, np.arange(1000, dtype=np.uint16))
    data = np.load(path, mmap_mode="r")
    x, y = data_loading(data, 4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
