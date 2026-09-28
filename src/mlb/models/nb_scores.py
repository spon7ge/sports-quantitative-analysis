from __future__ import annotations

import numpy as np
from scipy.stats import nbinom


def _r_p(mu, alpha):
    mu_arr = np.clip(np.atleast_1d(np.asarray(mu, dtype=float)), 1e-12, None)
    alpha_arr = np.clip(np.atleast_1d(np.asarray(alpha, dtype=float)), 1e-12, None)
    alpha_arr = np.broadcast_to(alpha_arr, mu_arr.shape)
    return 1.0 / alpha_arr, 1.0 / (1.0 + alpha_arr * mu_arr)


def nb2_quantile(mu, alpha, level: float) -> np.ndarray:
    r, p = _r_p(mu, alpha)
    return nbinom.ppf(level, r, p).astype(int)


def nb2_nll(y, mu, alpha) -> float:
    r, p = _r_p(mu, alpha)
    y_arr = np.asarray(y, dtype=int)
    return float(-np.mean(nbinom.logpmf(y_arr, r, p)))


def nb2_crps(y, mu, alpha, survival_limit: float = 1e-10) -> float:
    r, p = _r_p(mu, alpha)
    y_arr = np.atleast_1d(np.asarray(y, dtype=int))
    total = np.zeros(y_arr.shape[0], dtype=float)
    k = 0
    survival = np.ones(y_arr.shape[0], dtype=float)
    while np.any(survival >= survival_limit):
        cdf = nbinom.cdf(k, r, p)
        hit = (y_arr <= k).astype(float)
        total += (cdf - hit) ** 2
        survival = 1.0 - cdf
        k += 1
        if k > 100000:
            raise RuntimeError("CRPS support did not reach the survival limit")
    return float(np.mean(total))


def half_point_probabilities(mu, alpha, line: float):
    if float(line) != int(line) + 0.5:
        raise ValueError("line must be a half point")
    r, p = _r_p(mu, alpha)
    under = nbinom.cdf(int(np.floor(line)), r, p)
    over = 1.0 - under
    return over, under
