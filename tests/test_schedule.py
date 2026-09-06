"""Checks learning_rate_schedule against the reference values from the CS336 A1 test suite
(alpha_max=1, alpha_min=0.1, Tw=7, Tc=21)."""

import math

from training import learning_rate_schedule

EXPECTED = [
    0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714,
    0.7142857142857143, 0.8571428571428571, 1.0, 0.9887175604818206, 0.9554359905560885,
    0.9018241671106134, 0.8305704108364301, 0.7452476826029011, 0.6501344202803414, 0.55,
    0.44986557971965857, 0.3547523173970989, 0.26942958916356996, 0.19817583288938662,
    0.14456400944391146, 0.11128243951817937, 0.1, 0.1, 0.1, 0.1,
]


def test_learning_rate_schedule():
    for t, expected in enumerate(EXPECTED):
        actual = learning_rate_schedule(t, alpha_max=1, alpha_min=0.1, Tw=7, Tc=21)
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (t, actual, expected)


def test_boundaries():
    # warmup meets the cosine branch at Tw, and the cosine branch meets the floor at Tc
    assert learning_rate_schedule(7, 1, 0.1, 7, 21) == 1.0
    assert math.isclose(learning_rate_schedule(21, 1, 0.1, 7, 21), 0.1)
    assert learning_rate_schedule(100, 1, 0.1, 7, 21) == 0.1
