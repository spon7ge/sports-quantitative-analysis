import numpy as np
from scipy.stats import nbinom, poisson

from src.mlb.models.nb_scores import half_point_probabilities, nb2_crps, nb2_nll, nb2_quantile


def _poisson_discrete_crps(y: int, mu: float, survival_limit: float = 1e-10) -> float:
    total = 0.0
    k = 0
    survival = 1.0
    while survival >= survival_limit:
        cdf = poisson.cdf(k, mu)
        hit = 1.0 if y <= k else 0.0
        total += (cdf - hit) ** 2
        survival = 1.0 - cdf
        k += 1
    return total


def test_quantile_can_exceed_15_and_matches_the_cdf():
    mu = np.array([30.0])
    alpha = np.array([0.2])
    q90 = int(nb2_quantile(mu, alpha, 0.90)[0])
    assert q90 > 15
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    assert nbinom.cdf(q90, r, p)[0] >= 0.90
    assert nbinom.cdf(q90 - 1, r, p)[0] < 0.90


def test_nll_uses_the_log_pmf_at_the_observed_count():
    loss = nb2_nll(np.array([4]), np.array([5.0]), np.array([0.5]))
    r = 1.0 / 0.5
    p = 1.0 / (1.0 + 0.5 * 5.0)
    assert loss == pytest_approx(-nbinom.logpmf(4, r, p))


def test_half_point_line_has_no_push():
    over, under = half_point_probabilities(np.array([6.0]), np.array([0.4]), 5.5)
    assert over[0] + under[0] == pytest_approx(1.0)


def test_crps_stops_when_survival_is_negligible():
    score = nb2_crps(np.array([5]), np.array([6.0]), np.array([0.4]))
    assert np.isfinite(score)
    poisson_limit = nb2_crps(np.array([3]), np.array([3.0]), np.array([1e-6]))
    assert poisson_limit == pytest_approx(_poisson_discrete_crps(3, 3.0), abs=1e-3)
    point_mass = nb2_crps(np.array([0]), np.array([1e-6]), np.array([1e-6]))
    assert point_mass == pytest_approx(0.0, abs=1e-3)


def pytest_approx(value, **kwargs):
    import pytest

    return pytest.approx(value, **kwargs)
