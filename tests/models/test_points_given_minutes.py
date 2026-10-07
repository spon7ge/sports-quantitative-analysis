import numpy as np
import pyvinecopulib as pv

from models.shared.points_given_minutes import (
    CountModel,
    conditional_points_pmf,
    copula_corner_lifts,
    corner_lifts,
    count_pit,
    fit_count_model,
)


def _simulated(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(2, 40, n)
    rate = rng.uniform(0.2, 0.8, n)
    truth = CountModel(log_kappa=0.1, beta=0.9, log_alpha=np.log(0.15))
    mu = truth.mean(minutes, rate)
    size = 1 / truth.alpha(minutes)
    points = rng.negative_binomial(size, size / (size + mu))
    return minutes, rate, points, truth, rng


def test_fit_recovers_parameters():
    minutes, rate, points, truth, _ = _simulated()
    fit = fit_count_model(minutes, points, rate, flexible=False)
    assert abs(fit.log_kappa - truth.log_kappa) < 0.1
    assert abs(fit.beta - truth.beta) < 0.05
    assert abs(fit.log_alpha - truth.log_alpha) < 0.2
    assert fit.gamma == 0.0 and fit.beta2 == 0.0


def test_flexible_fit_recovers_minutes_dependent_dispersion():
    rng = np.random.default_rng(3)
    minutes = rng.uniform(2, 40, 40_000)
    rate = rng.uniform(0.2, 0.8, 40_000)
    truth = CountModel(log_kappa=0.0, beta=1.0, log_alpha=np.log(0.12), alpha_minutes=-0.6)
    mu = truth.mean(minutes, rate)
    size = 1 / truth.alpha(minutes)
    points = rng.negative_binomial(size, size / (size + mu))
    fit = fit_count_model(minutes, points, rate)
    assert abs(fit.alpha_minutes - truth.alpha_minutes) < 0.15
    assert abs(fit.beta2) < 0.05


def test_count_pit_is_uniform_under_the_true_model():
    minutes, rate, points, truth, rng = _simulated()
    pit = count_pit(points, truth.mean(minutes, rate), truth.alpha(minutes), rng.random(len(points)))
    assert abs(pit.mean() - 0.5) < 0.01
    assert abs(12 * pit.var() - 1.0) < 0.03


def test_independence_copula_matches_no_copula():
    model = CountModel(0.0, 1.0, np.log(0.2))
    u = (np.arange(20) + 0.5) / 20
    minutes = np.tile(np.linspace(5, 35, 20), (3, 1))
    rate = np.array([0.3, 0.5, 0.7])
    plain = conditional_points_pmf(minutes, u, rate, model, kmax=60)
    indep = conditional_points_pmf(minutes, u, rate, model, kmax=60, bicop=pv.Bicop())
    np.testing.assert_allclose(plain.sum(axis=1), 1.0)
    np.testing.assert_allclose(plain, indep, atol=1e-9)


def test_positive_copula_shifts_points_down_when_minutes_are_low():
    model = CountModel(0.0, 1.0, np.log(0.2))
    u = np.array([0.05])
    minutes = np.array([[20.0]])
    clayton = pv.Bicop(family=pv.families.clayton, parameters=np.array([[2.0]]))
    plain = conditional_points_pmf(minutes, u, np.array([0.5]), model, kmax=60)
    linked = conditional_points_pmf(minutes, u, np.array([0.5]), model, kmax=60, bicop=clayton)
    ks = np.arange(plain.shape[1])
    assert linked @ ks < plain @ ks


def test_corner_lifts_and_copula_lifts_under_independence():
    rng = np.random.default_rng(1)
    u, v = rng.random(200_000), rng.random(200_000)
    for row in corner_lifts(u, v, [0.1]):
        assert abs(row["lift"] - 1.0) < 0.1
    for row in copula_corner_lifts(pv.Bicop(), [0.1, 0.02]):
        assert abs(row["lift"] - 1.0) < 1e-9
