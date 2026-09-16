"""Negative-binomial PMF on a finite K support with a collapsed tail."""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

_MAX_SUPPORT_EXTENSION = 2048


def as_pmf_matrix(pmf) -> np.ndarray:
    """Return a float matrix with shape (n, k_max + 2)."""
    arr = np.asarray(pmf, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"pmf must be 1-d or 2-d, got shape {arr.shape}")
    return arr


def negative_binomial_pmf(mu, alpha, k_max, tail_threshold) -> np.ndarray:
    """NB2 PMF on {0, ..., k_max} plus a collapsed P(K > k_max) column.

    Variance is ``mu + alpha * mu^2``. SciPy mapping: ``r = 1/alpha``,
    ``p = 1 / (1 + alpha * mu)``. Returned width is always ``k_max + 2``.
    If tail mass at ``k_max`` exceeds ``tail_threshold``, the support is
    extended internally and the extra mass is collapsed into the tail.
    """
    mu_arr = np.clip(np.atleast_1d(np.asarray(mu, dtype=float)), 1e-12, 200.0)
    alpha_arr = np.clip(np.asarray(alpha, dtype=float), 1e-12, None)
    alpha_arr = np.broadcast_to(np.atleast_1d(alpha_arr), mu_arr.shape)
    k_max_i = int(k_max)
    threshold = float(tail_threshold)
    r = 1.0 / alpha_arr
    p = 1.0 / (1.0 + alpha_arr * mu_arr)

    k_ext = k_max_i
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        survival = nbinom.sf(k_ext, r, p)
        step = 32
        while np.any(survival > threshold) and k_ext < k_max_i + _MAX_SUPPORT_EXTENSION:
            k_ext = min(k_ext + step, k_max_i + _MAX_SUPPORT_EXTENSION)
            survival = nbinom.sf(k_ext, r, p)
            step = min(step * 2, 256)

        ks = np.arange(k_ext + 1, dtype=int)
        full = nbinom.pmf(ks[None, :], r[:, None], p[:, None])
        full = np.clip(np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

        body = full[:, : k_max_i + 1]
        if k_ext > k_max_i:
            tail = full[:, k_max_i + 1 :].sum(axis=1) + nbinom.sf(k_ext, r, p)
        else:
            tail = nbinom.sf(k_max_i, r, p)
    tail = np.clip(np.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    out = np.concatenate([body, tail[:, None]], axis=1)
    out = np.clip(out, 0.0, None)
    row_sum = out.sum(axis=1, keepdims=True)
    out = out / np.maximum(row_sum, 1e-15)
    return out


def pmf_cdf(pmf) -> np.ndarray:
    """Row-wise CDF, including the tail column as P(K >= k_max + 1)."""
    mat = np.asarray(pmf, dtype=float)
    return np.cumsum(mat, axis=-1)
