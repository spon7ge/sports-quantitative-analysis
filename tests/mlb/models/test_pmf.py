"""Negative-binomial PMF support, tail collapse, and mean."""

from __future__ import annotations

import numpy as np
from src.mlb.models.pmf import negative_binomial_pmf, pmf_cdf


def test_pmf_sums_to_one_and_mean_near_mu() -> None:
    mu = np.array([2.0, 6.0, 9.0])
    pmf = negative_binomial_pmf(mu, 0.12, 15, 0.001)
    assert pmf.shape == (3, 17)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-10)
    support = np.arange(16)
    mean = pmf[:, :16] @ support + pmf[:, -1] * 16.0
    np.testing.assert_allclose(mean, mu, rtol=0.02, atol=0.15)
    cdf = pmf_cdf(pmf)
    np.testing.assert_allclose(cdf[:, -1], 1.0, atol=1e-10)


def test_pmf_tail_collapse_keeps_width() -> None:
    pmf = negative_binomial_pmf(35.0, 0.08, 15, 0.001)
    assert pmf.shape == (1, 17)
    assert pmf[0, -1] > 0.001
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-10)
