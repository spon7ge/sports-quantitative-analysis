"""Nested isotonic map of the points CDF. Holdout dates are refused."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_isotonic_pulls_a_low_outcome_onto_zero():
    from models.shared.points_calibration import (
        apply_points_cdf_calibrator,
        fit_points_cdf_calibrator,
    )

    pmf = np.tile(np.array([0.9, 0.1]), (8, 1))
    y = np.zeros(8)
    dates = pd.to_datetime(["2024-01-01"] * 8)
    calibrator = fit_points_cdf_calibrator(np.cumsum(pmf, axis=1), y, dates)
    calibrated = apply_points_cdf_calibrator(np.cumsum(pmf, axis=1), calibrator)
    np.testing.assert_allclose(calibrated[:, 0], 1.0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)


def test_isotonic_leaves_a_matched_two_point_pmf_in_place():
    from models.shared.points_calibration import (
        apply_points_cdf_calibrator,
        fit_points_cdf_calibrator,
    )

    pmf = np.tile(np.array([0.5, 0.5]), (8, 1))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dates = pd.to_datetime(["2024-01-01"] * 8)
    cdf = np.cumsum(pmf, axis=1)
    calibrator = fit_points_cdf_calibrator(cdf, y, dates)
    calibrated = apply_points_cdf_calibrator(cdf, calibrator)
    np.testing.assert_allclose(calibrated, pmf)


def test_calibrated_cdf_stays_monotone_and_ends_at_one():
    from models.shared.points_calibration import (
        apply_points_cdf_calibrator,
        fit_points_cdf_calibrator,
    )

    rng = np.random.default_rng(0)
    pmf = rng.dirichlet(np.ones(5), size=12)
    y = rng.integers(0, 5, size=12)
    dates = pd.to_datetime(["2024-03-01"] * 12)
    calibrator = fit_points_cdf_calibrator(np.cumsum(pmf, axis=1), y, dates)
    calibrated = apply_points_cdf_calibrator(np.cumsum(pmf, axis=1), calibrator)
    cdf = np.cumsum(calibrated, axis=1)
    assert np.all(np.diff(cdf, axis=1) >= -1e-12)
    np.testing.assert_allclose(cdf[:, -1], 1.0)


def test_fit_rejects_a_holdout_date():
    from models.shared.points_calibration import fit_points_cdf_calibrator

    cdf = np.array([[0.2, 1.0]])
    with pytest.raises(ValueError, match="holdout"):
        fit_points_cdf_calibrator(
            cdf,
            np.array([0]),
            pd.to_datetime(["2026-10-20"]),
        )


def test_holdout_start_moves_with_the_caller():
    from models.shared.points_calibration import fit_points_cdf_calibrator

    cdf = np.array([[0.2, 1.0]])
    y = np.array([0])
    dates = pd.to_datetime(["2025-10-21"])
    fit_points_cdf_calibrator(cdf, y, dates, holdout_start="2026-10-20")
    with pytest.raises(ValueError, match="holdout"):
        fit_points_cdf_calibrator(cdf, y, dates, holdout_start="2025-10-21")


def test_nested_calibrator_is_fit_on_earlier_folds_only():
    from models.shared.points_calibration import nested_points_calibrators

    pmf = np.tile(np.array([0.9, 0.1]), (4, 1))
    y = np.zeros(4)
    folds = np.array([1, 1, 2, 2])
    dates = pd.to_datetime(["2024-01-01"] * 4)
    nested = nested_points_calibrators(pmf, y, folds, dates, folds=(2,))
    np.testing.assert_allclose(nested.pmf[:2], pmf[:2])
    np.testing.assert_allclose(nested.pmf[2:, 0], 1.0)
    assert nested.evaluated.tolist() == [False, False, True, True]
    assert list(nested.curves) == [2]


def test_calibrator_is_kept_only_when_both_upper_bounds_are_negative():
    from models.shared.points_calibration import keep_calibrator

    assert keep_calibrator(rps_ci_high=-0.01, reliability_ci_high=-0.001)
    assert not keep_calibrator(rps_ci_high=-0.01, reliability_ci_high=0.001)
    assert not keep_calibrator(rps_ci_high=0.0, reliability_ci_high=-0.001)


def test_curve_is_in_the_unit_interval():
    from models.shared.points_calibration import (
        calibrator_curve,
        fit_points_cdf_calibrator,
    )

    pmf = np.tile(np.array([0.9, 0.1]), (8, 1))
    dates = pd.to_datetime(["2024-01-01"] * 8)
    calibrator = fit_points_cdf_calibrator(
        np.cumsum(pmf, axis=1), np.zeros(8), dates
    )
    curve = calibrator_curve(calibrator, np.linspace(0.0, 1.0, 11))
    assert curve.shape == (11,)
    assert np.all(curve >= 0.0) and np.all(curve <= 1.0)
    assert np.all(np.diff(curve) >= -1e-12)
