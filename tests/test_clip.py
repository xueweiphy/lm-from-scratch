"""gradient_clipping must match torch.nn.utils.clip_grad_norm_ (mirrors the CS336 A1 test),
including a frozen parameter with no gradient."""

import numpy
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from training import gradient_clipping


def _grads_after(clip_fn, tensors, max_norm):
    params = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    params[-1].requires_grad_(False)
    torch.cat(params).sum().backward()
    clip_fn(params, max_norm)
    return [torch.clone(p.grad) for p in params if p.grad is not None]


def test_gradient_clipping_matches_pytorch():
    torch.manual_seed(0)
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    for max_norm in (1e-2, 1e3):  # clipping active / inactive
        expected = _grads_after(clip_grad_norm_, tensors, max_norm)
        actual = _grads_after(gradient_clipping, tensors, max_norm)
        assert len(expected) == len(actual) == 5
        for e, a in zip(expected, actual):
            numpy.testing.assert_allclose(e.numpy(), a.numpy(), atol=1e-5)
