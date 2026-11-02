"""NB2 probabilities from a fixed points mean.

The mean is the point forecast. Dispersion is one shared number,
fit where the caller chooses — walk-forward rows, not the holdout.
Variance is ``mu + alpha * mu**2``.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from models.shared.points_given_minutes import nb_cdf, nb_logpmf


def fit_nb2_alpha(points, mu) -> float:
    """Maximum likelihood of one NB2 dispersion with the mean held fixed."""
    y = np.asarray(points, dtype=float)
    mean = np.asarray(mu, dtype=float)
    if y.shape != mean.shape:
        raise ValueError("points and mu must have the same shape")
    if y.size == 0:
        raise ValueError("need at least one row to fit dispersion")
    if (
        not np.isfinite(y).all()
        or np.any(y < 0)
        or np.any(np.abs(y - np.rint(y)) > 1e-6)
    ):
        raise ValueError("points must be finite non-negative integers")
    if not np.isfinite(mean).all() or np.any(mean <= 0):
        raise ValueError("mu must be finite and positive")

    def nll(log_alpha: float) -> float:
        return -float(np.mean(nb_logpmf(y, mean, np.exp(log_alpha))))

    result = optimize.minimize_scalar(nll, bounds=(-8.0, 3.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"dispersion fit failed: {result.message}")
    return float(np.exp(result.x))


def over_probability(mu, alpha, line) -> np.ndarray:
    """P(points > line). A half-point line has no push."""
    mean = np.asarray(mu, dtype=float)
    level = np.asarray(line, dtype=float)
    mean, level = np.broadcast_arrays(mean, level)
    if not np.isfinite(mean).all() or np.any(mean <= 0):
        raise ValueError("mu must be finite and positive")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    if not np.isfinite(level).all():
        raise ValueError("line must be finite")
    return 1.0 - nb_cdf(np.floor(level), mean, float(alpha))
