"""Over and push probabilities from the raw quantile CDF."""

import numpy as np
import pytest
from models.shared.points_calibration import quantile_line_probabilities

TAUS = np.array([0.2, 0.5, 0.8])
KNOTS = np.array([[10.0, 20.0, 30.0]])


def test_half_point_over_is_one_minus_the_cdf_at_the_floor():
    over, push = quantile_line_probabilities(19.5, KNOTS, TAUS)

    # F(19) sits 0.9 of the way from q_0.20=10 to q_0.50=20.
    np.testing.assert_allclose(over, [0.53])
    np.testing.assert_allclose(push, [0.0])


def test_integer_line_keeps_the_atom_out_of_the_over():
    over, push = quantile_line_probabilities(20.0, KNOTS, TAUS)

    # F(20) = 0.5 and F(19) = 0.47, so the atom at 20 is 0.03.
    np.testing.assert_allclose(over, [0.5])
    np.testing.assert_allclose(push, [0.03])
    np.testing.assert_allclose(over + push, [0.53])


def test_over_and_push_leave_a_nonnegative_under():
    lines = np.array([4.5, 19.5, 20.0, 40.5])
    over, push = quantile_line_probabilities(lines, np.repeat(KNOTS, 4, axis=0), TAUS)
    under = 1.0 - over - push

    np.testing.assert_allclose(over + push + under, np.ones(4))
    assert np.all(over >= 0.0) and np.all(over <= 1.0)
    assert np.all(push >= 0.0) and np.all(under >= -1e-12)
    # Below q_0.20 and above q_0.80 the CDF stays at the end taus.
    np.testing.assert_allclose(over[[0, 3]], [0.8, 0.2])
    np.testing.assert_allclose(push[[0, 3]], [0.0, 0.0])


def test_one_line_broadcasts_across_rows():
    grid = np.tile(KNOTS, (3, 1))

    over, push = quantile_line_probabilities(19.5, grid, TAUS)

    np.testing.assert_allclose(over, [0.53, 0.53, 0.53])
    np.testing.assert_allclose(push, [0.0, 0.0, 0.0])


def test_a_crossed_grid_is_refused():
    crossed = np.array([[30.0, 10.0, 20.0]])

    with pytest.raises(ValueError, match="nondecreasing"):
        quantile_line_probabilities(15.5, crossed, TAUS)


def test_a_non_finite_line_is_refused():
    with pytest.raises(ValueError, match="line"):
        quantile_line_probabilities(np.array([np.nan]), KNOTS, TAUS)


def test_one_grid_broadcasts_across_lines():
    over, push = quantile_line_probabilities(np.array([19.5, 20.0]), KNOTS, TAUS)

    np.testing.assert_allclose(over, [0.53, 0.5])
    np.testing.assert_allclose(push, [0.0, 0.03])


def test_a_row_mismatch_is_refused():
    grid = np.tile(KNOTS, (2, 1))

    with pytest.raises(ValueError, match="line"):
        quantile_line_probabilities(np.array([10.5, 12.5, 14.5]), grid, TAUS)
