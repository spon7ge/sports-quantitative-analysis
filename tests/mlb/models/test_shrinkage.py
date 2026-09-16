"""Shrinkage formula and broadcasting."""

from __future__ import annotations

import numpy as np
from src.mlb.models.shrinkage import shrink_rate


def test_shrink_rate_formula() -> None:
    expected = (10.0 + 0.2 * 50.0) / (100.0 + 50.0)
    assert shrink_rate(10, 100, 0.2, 50) == expected


def test_shrink_rate_zero_trials_returns_prior() -> None:
    assert shrink_rate(5, 0, 0.25, 100) == 0.25
    assert shrink_rate(0, -3, 0.3, 10) == 0.3


def test_shrink_rate_broadcasting() -> None:
    out = shrink_rate([1.0, 2.0], [10.0, 0.0], 0.3, 20.0)
    assert isinstance(out, np.ndarray)
    assert out[0] == (1.0 + 0.3 * 20.0) / (10.0 + 20.0)
    assert out[1] == 0.3
    grid = shrink_rate(np.array([[1.0, 2.0], [3.0, 4.0]]), 10.0, 0.2, 5.0)
    assert grid.shape == (2, 2)
    assert grid[1, 1] == (4.0 + 0.2 * 5.0) / (10.0 + 5.0)
