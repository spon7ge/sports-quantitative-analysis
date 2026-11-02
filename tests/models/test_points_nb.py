"""NB2 over-probabilities with the mean fixed at a point forecast."""

import numpy as np
from models.shared.points_nb import fit_nb2_alpha, over_probability
from scipy import stats


def test_dispersion_fit_recovers_the_simulated_alpha():
    rng = np.random.default_rng(0)
    mu = rng.uniform(4, 30, 20_000)
    alpha = 0.25
    size = 1.0 / alpha
    points = rng.negative_binomial(size, size / (size + mu))

    fitted = fit_nb2_alpha(points, mu)

    assert abs(fitted - alpha) < 0.02


def test_over_probability_matches_the_survival_function():
    mu = np.array([10.0, 22.0])
    alpha = 0.1
    line = np.array([9.5, 21.5])
    size = 1.0 / alpha
    probability = size / (size + mu)

    expected = stats.nbinom.sf(np.floor(line), size, probability)

    np.testing.assert_allclose(over_probability(mu, alpha, line), expected)


def test_half_point_over_and_under_cover_every_outcome():
    mu = np.array([8.0, 15.0, 27.5])
    alpha = 0.2
    line = np.array([7.5, 14.5, 26.5])
    over = over_probability(mu, alpha, line)
    size = 1.0 / alpha
    under = stats.nbinom.cdf(np.floor(line), size, size / (size + mu))

    np.testing.assert_allclose(over + under, 1.0)
    assert np.all((over > 0) & (over < 1))
