"""Over/under/push probabilities from a discrete strikeout PMF."""

from __future__ import annotations

import math

import numpy as np

from src.mlb.models.pmf import as_pmf_matrix, pmf_cdf


def _cdf_at(cdf: np.ndarray, k_max: int, k: int) -> np.ndarray:
    """P(K <= k). Tail mass lives above k_max, so F(k_max) = 1 - tail."""
    if k < 0:
        return np.zeros(cdf.shape[0], dtype=float)
    if k <= k_max:
        return cdf[:, k]
    return np.ones(cdf.shape[0], dtype=float)


def _mass_at(pmf: np.ndarray, k_max: int, k: int) -> np.ndarray:
    if k < 0:
        return np.zeros(pmf.shape[0], dtype=float)
    if k <= k_max:
        return pmf[:, k]
    if k == k_max + 1:
        return pmf[:, -1]
    return np.zeros(pmf.shape[0], dtype=float)


def _maybe_squeeze(values: np.ndarray, squeeze: bool) -> float | np.ndarray:
    if squeeze:
        return float(values.reshape(-1)[0])
    return values


def prop_probabilities(pmf, line: float) -> dict:
    """Settle an over/under line against the finite PMF (tail is > k_max).

    Half-integer L: ``p_over = 1 - F(floor(L))``, ``p_under = F(floor(L))``,
    ``p_push = 0``. Integer m: ``p_over = 1 - F(m)``, ``p_push = P(K=m)``,
    ``p_under = F(m - 1)``.
    """
    mat = as_pmf_matrix(pmf)
    squeeze = np.asarray(pmf).ndim == 1
    k_max = mat.shape[1] - 2
    cdf = pmf_cdf(mat)
    line_f = float(line)

    if math.isfinite(line_f) and float(line_f).is_integer():
        m = int(round(line_f))
        p_over = 1.0 - _cdf_at(cdf, k_max, m)
        p_push = _mass_at(mat, k_max, m)
        p_under = _cdf_at(cdf, k_max, m - 1)
    else:
        floor_l = int(np.floor(line_f))
        p_over = 1.0 - _cdf_at(cdf, k_max, floor_l)
        p_under = _cdf_at(cdf, k_max, floor_l)
        p_push = np.zeros(mat.shape[0], dtype=float)

    return {
        "p_over": _maybe_squeeze(np.asarray(p_over, dtype=float), squeeze),
        "p_under": _maybe_squeeze(np.asarray(p_under, dtype=float), squeeze),
        "p_push": _maybe_squeeze(np.asarray(p_push, dtype=float), squeeze),
    }
