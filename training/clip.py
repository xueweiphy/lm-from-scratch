"""Gradient clipping by global l2 norm (CS336 A1, gradient_clipping).

The norm is taken over all gradients together, as one concatenated vector.
If it exceeds maxnorm, every gradient is scaled in place by maxnorm / (norm + eps).
Parameters without a gradient (requires_grad=False) are skipped.
"""

import torch


def gradient_clipping (  parameters, maxnorm, eps = 1e-6):
    gradlist = [p.grad for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum((g**2).sum() for g in gradlist))

    if norm < maxnorm :
        return
    else :
        for g in gradlist :
            g.mul_ ( maxnorm / ( norm + eps ) )
