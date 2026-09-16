"""Empirical-Bayes rate shrinkage."""

from __future__ import annotations

import numpy as np


def shrink_rate(
    successes,
    trials,
    prior_mean,
    prior_strength,
) -> float | np.ndarray:
    """Posterior mean of a Beta/binomial rate: (s + m k) / (n + k).

    If ``trials`` is 0, return ``prior_mean``. Trials are clipped at 0.
    """
    successes_arr = np.asarray(successes, dtype=float)
    trials_arr = np.clip(np.asarray(trials, dtype=float), 0.0, None)
    prior_mean_arr = np.asarray(prior_mean, dtype=float)
    prior_strength_arr = np.asarray(prior_strength, dtype=float)
    shrunk = (successes_arr + prior_mean_arr * prior_strength_arr) / (
        trials_arr + prior_strength_arr
    )
    result = np.where(trials_arr == 0, prior_mean_arr, shrunk)
    if np.ndim(result) == 0:
        return float(result)
    return np.asarray(result, dtype=float)
