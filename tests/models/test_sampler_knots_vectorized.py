"""The vectorized knot builders must match the original per-point loops."""

import numpy as np
import pytest

from models.shared.minutes_sampler import _clip_breakpoints
from models.shared.ppm_sampler import _zero_atom_knots


def _clip_breakpoints_loop(u, q, lo, hi):
    u = np.asarray(u, dtype=float)
    q = np.asarray(q, dtype=float)
    out_u, out_q = [], []

    def append(point_u, point_q):
        clipped = float(np.clip(point_q, lo, hi))
        if out_u and point_u < out_u[-1]:
            raise ValueError("quantile knots decreased in u")
        if out_u and point_u == out_u[-1]:
            out_q[-1] = clipped
            return
        if out_q and clipped < out_q[-1] - 1e-8:
            raise ValueError("clipped quantile function decreased")
        out_u.append(float(point_u))
        out_q.append(clipped)

    append(float(u[0]), float(q[0]))
    for index in range(1, len(u)):
        u0, u1 = float(u[index - 1]), float(u[index])
        q0, q1 = float(q[index - 1]), float(q[index])
        if q0 != q1:
            for level in (lo, hi):
                if (q0 < level < q1) or (q1 < level < q0):
                    t = (level - q0) / (q1 - q0)
                    append(u0 + t * (u1 - u0), level)
        append(u1, q1)
    clipped_q = np.asarray(out_q, dtype=float)
    np.maximum.accumulate(clipped_q, out=clipped_q)
    return np.asarray(out_u, dtype=float), clipped_q


def _zero_atom_knots_loop(q05, table, zeros):
    n = len(table)
    atom = zeros / n
    u_atom = atom * 0.05
    positive = table[zeros:]
    m = len(positive)
    local = np.linspace(0.0, 1.0, m)
    u_positive = (atom + local * (1.0 - atom)) * 0.05
    q_positive = q05 * positive
    u_knots = [0.0, u_atom, float(np.nextafter(u_atom, 1.0))]
    q_knots = [0.0, 0.0, float(q_positive[0])]
    for index in range(1, m - 1):
        u_knots.append(float(u_positive[index]))
        q_knots.append(float(q_positive[index]))
    return np.asarray(u_knots), np.asarray(q_knots)


@pytest.mark.parametrize("seed", range(25))
def test_clip_breakpoints_matches_loop(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 400))
    u = np.sort(rng.random(n))
    u[rng.random(n) < 0.05] = u[0]
    u = np.sort(u)
    q = np.sort(rng.normal(20, 15, n))
    q[rng.random(n) < 0.1] = 0.0
    q = np.sort(q)
    lo, hi = 0.0, float(rng.choice([30.0, 48.0, 60.0, 6.0]))
    expected = _clip_breakpoints_loop(u, q, lo, hi)
    actual = _clip_breakpoints(u, q, lo, hi)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_clip_breakpoints_rejects_decreasing_u():
    with pytest.raises(ValueError):
        _clip_breakpoints(np.array([0.0, 0.5, 0.4]), np.array([1.0, 2.0, 3.0]), 0.0, 10.0)


@pytest.mark.parametrize("zeros", [1, 5, 200])
def test_zero_atom_knots_matches_loop(zeros):
    rng = np.random.default_rng(zeros)
    table = np.sort(np.concatenate([np.zeros(zeros), rng.random(500), [1.0]]))
    expected = _zero_atom_knots_loop(0.3, table, zeros)
    actual = _zero_atom_knots(0.3, table, zeros)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
