"""Cosine learning-rate schedule with linear warmup (CS336 A1, learning_rate_schedule).

    t <  Tw       : linear warmup, alpha = t / Tw * alpha_max
    Tw <= t <= Tc : cosine anneal from alpha_max down to alpha_min
    t >  Tc       : alpha_min
"""

import math


def learning_rate_schedule ( t, alpha_max, alpha_min, Tw, Tc ) :
    """Learning rate at iteration t.

    Args:
        t:         iteration number (0-based)
        alpha_max: peak learning rate, reached at t = Tw
        alpha_min: final learning rate, reached at t = Tc and held afterwards
        Tw:        number of warmup iterations
        Tc:        iteration at which cosine annealing ends
    """
    if t < Tw :
        out = t / Tw * alpha_max
    elif t <= Tc :
        out = alpha_min + 0.5* ( 1 + math.cos ( ( t - Tw ) * math.pi / ( Tc - Tw ) ) )* ( alpha_max - alpha_min )
    else :
        out = alpha_min

    return out
