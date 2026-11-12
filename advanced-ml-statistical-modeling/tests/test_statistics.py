import numpy as np
from train import bootstrap_ci, permutation_p


def test_confidence_interval():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([.1, .2, .8, .7, .3, .9, .2, .6])
    lo, hi = bootstrap_ci(y, p, n=100)
    assert 0 <= lo <= hi <= 1


def test_identical_models():
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .9, .2, .8])
    assert permutation_p(y, p, p, repeats=50) == 1
