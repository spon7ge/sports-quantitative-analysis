"""Proper scoring rules on finite strikeout PMFs."""

from __future__ import annotations

import numpy as np

from src.mlb.models.pmf import as_pmf_matrix


def _point_mass(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, width = pmf.shape
    k_max = width - 2
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if y_arr.shape[0] != n:
        raise ValueError("pmf rows and y length must match")
    p_y = np.zeros(n, dtype=float)
    in_support = (y_arr >= 0) & (y_arr <= k_max)
    rows = np.nonzero(in_support)[0]
    p_y[in_support] = pmf[rows, y_arr[in_support]]
    in_tail = y_arr > k_max
    p_y[in_tail] = pmf[in_tail, -1]
    return p_y


def pmf_nll(pmf, y) -> np.ndarray:
    """Negative log likelihood: ``-log(p_y + 1e-12)``."""
    mat = as_pmf_matrix(pmf)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if mat.shape[0] == 1 and y_arr.shape[0] > 1:
        mat = np.repeat(mat, y_arr.shape[0], axis=0)
    return -np.log(_point_mass(mat, y_arr) + 1e-12)


def discrete_crps(pmf, y) -> np.ndarray:
    """Discrete CRPS on k = 0..k_max: ``sum_k (F(k) - 1{y <= k})^2``.

    Tail mass is already inside F(k_max); it is not scored as a second bin.
    A point mass on ``y`` in ``0..k_max`` (or all mass in the tail when
    ``y > k_max``) yields CRPS 0.
    """
    mat = as_pmf_matrix(pmf)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if mat.shape[0] == 1 and y_arr.shape[0] > 1:
        mat = np.repeat(mat, y_arr.shape[0], axis=0)
    k_max = mat.shape[1] - 2
    cdf = np.cumsum(mat[:, : k_max + 1], axis=1)
    ks = np.arange(k_max + 1)
    indicator = (y_arr[:, None] <= ks[None, :]).astype(float)
    return np.sum((cdf - indicator) ** 2, axis=1)
