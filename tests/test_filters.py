import random
import statistics

import pytest

from gesture_mouse.filters import OneEuroFilter


def test_first_value_passes_through():
    assert OneEuroFilter()(0.7, 0.0) == 0.7


def test_converges_to_constant_input():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(0.0, 0.0)
    out = 0.0
    for i in range(1, 120):
        out = f(1.0, i / 30)
    assert out == pytest.approx(1.0, abs=1e-3)


def test_reduces_jitter():
    rng = random.Random(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = [0.5 + rng.uniform(-0.02, 0.02) for _ in range(200)]
    out = [f(x, i / 30) for i, x in enumerate(raw)]

    assert statistics.pstdev(out[50:]) < statistics.pstdev(raw[50:]) / 2


def test_reset_forgets_state():
    f = OneEuroFilter()
    f(0.0, 0.0)
    f(0.0, 0.1)
    f.reset()
    assert f(5.0, 0.2) == 5.0
