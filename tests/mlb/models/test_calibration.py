"""PIT range, CRPS point mass, and monotone recalibration."""

from __future__ import annotations

import numpy as np
from src.mlb.models.calibration import (
    apply_recalibration,
    calibration_slope_intercept,
    fit_pit_recalibration,
    randomized_pit,
)
from src.mlb.models.metrics import discrete_crps, pmf_nll
from src.mlb.models.pmf import negative_binomial_pmf
from src.mlb.models.props import prop_probabilities


def test_randomized_pit_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    mu = rng.uniform(3.0, 9.0, size=30)
    pmf = negative_binomial_pmf(mu, 0.15, 15, 0.001)
    y = rng.poisson(mu)
    pit = randomized_pit(pmf, y, np.random.default_rng(7))
    assert pit.shape == (30,)
    assert np.all((pit >= 0.0) & (pit <= 1.0))


def test_discrete_crps_zero_on_point_mass() -> None:
    pmf = np.zeros(17)
    pmf[4] = 1.0
    assert discrete_crps(pmf, 4)[0] == 0.0
    tail = np.zeros(17)
    tail[-1] = 1.0
    assert discrete_crps(tail, 20)[0] == 0.0
    nll = pmf_nll(pmf, 4)
    assert nll.shape == (1,)
    assert nll[0] < 1e-8


def test_recalibration_keeps_threshold_probabilities_ordered() -> None:
    rng = np.random.default_rng(0)
    mu = rng.uniform(3.0, 10.0, size=40)
    pmf = negative_binomial_pmf(mu, 0.2, 15, 0.001)
    y = np.clip(rng.poisson(mu), 0, 25)
    recal = fit_pit_recalibration(pmf, y)
    adjusted = apply_recalibration(recal, pmf)
    over_4 = np.atleast_1d(prop_probabilities(adjusted, 4.5)["p_over"])
    over_6 = np.atleast_1d(prop_probabilities(adjusted, 6.5)["p_over"])
    assert np.all(over_4 + 1e-9 >= over_6)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(adjusted >= -1e-12)


def test_calibration_slope_intercept_returns_two_floats() -> None:
    rng = np.random.default_rng(1)
    mu = rng.uniform(4.0, 8.0, size=60)
    pmf = negative_binomial_pmf(mu, 0.12, 15, 0.001)
    y = rng.poisson(mu)
    intercept, slope = calibration_slope_intercept(pmf, y)
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
