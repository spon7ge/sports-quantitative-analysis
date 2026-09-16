"""PIT diagnostics and monotone CDF recalibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from src.mlb.models.pmf import as_pmf_matrix, pmf_cdf
from src.mlb.models.props import prop_probabilities


@dataclass
class Recalibrator:
    """Monotone map from predicted CDF values to recalibrated CDF values."""

    isotonic: IsotonicRegression


def randomized_pit(pmf, y, rng) -> np.ndarray:
    """Randomized PIT: ``F(y-1) + U * (F(y) - F(y-1))`` with ``F(-1) = 0``.

    Outcomes above ``k_max`` use the lumped tail bin.
    """
    mat = as_pmf_matrix(pmf)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if mat.shape[0] == 1 and y_arr.shape[0] > 1:
        mat = np.repeat(mat, y_arr.shape[0], axis=0)
    n = y_arr.shape[0]
    if mat.shape[0] != n:
        raise ValueError("pmf rows and y length must match")
    k_max = mat.shape[1] - 2
    cdf = pmf_cdf(mat)
    rows = np.arange(n)
    f_y = np.zeros(n, dtype=float)
    f_ym1 = np.zeros(n, dtype=float)

    in_support = (y_arr >= 0) & (y_arr <= k_max)
    f_y[in_support] = cdf[rows[in_support], y_arr[in_support]]
    positive = in_support & (y_arr > 0)
    f_ym1[positive] = cdf[rows[positive], y_arr[positive] - 1]

    in_tail = y_arr > k_max
    f_y[in_tail] = 1.0
    f_ym1[in_tail] = cdf[rows[in_tail], k_max]

    u = np.asarray(rng.random(n), dtype=float)
    pit = f_ym1 + u * np.clip(f_y - f_ym1, 0.0, None)
    return np.clip(pit, 0.0, 1.0)


def calibration_slope_intercept(pmf, y) -> tuple[float, float]:
    """Logistic calibration of P(K > 5.5) vs the realized over-5.5 indicator.

    Returns ``(intercept, slope)`` from ``logit P = a + b * logit(p_over_5.5)``.
    """
    mat = as_pmf_matrix(pmf)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if mat.shape[0] == 1 and y_arr.shape[0] > 1:
        mat = np.repeat(mat, y_arr.shape[0], axis=0)
    props = prop_probabilities(mat, 5.5)
    p_over = np.atleast_1d(np.asarray(props["p_over"], dtype=float))
    outcome = (y_arr > 5.5).astype(float)
    p_clip = np.clip(p_over, 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p_clip / (1.0 - p_clip))
    design = np.column_stack([np.ones(len(outcome)), logit_p])
    try:
        result = sm.Logit(outcome, design).fit(disp=0, maxiter=200)
        params = np.asarray(result.params, dtype=float).reshape(-1)
        return float(params[0]), float(params[1])
    except Exception:
        coef, *_ = np.linalg.lstsq(design, outcome, rcond=None)
        return float(coef[0]), float(coef[1])


def _cdf_at_y(cdf: np.ndarray, y: np.ndarray, k_max: int) -> np.ndarray:
    n = y.shape[0]
    out = np.zeros(n, dtype=float)
    in_support = (y >= 0) & (y <= k_max)
    rows = np.arange(n)
    out[in_support] = cdf[rows[in_support], y[in_support]]
    out[y > k_max] = 1.0
    return out


def fit_pit_recalibration(pmf, y) -> Recalibrator:
    """Isotonic map from predicted F(y) to the empirical CDF of those values."""
    mat = as_pmf_matrix(pmf)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if mat.shape[0] == 1 and y_arr.shape[0] > 1:
        mat = np.repeat(mat, y_arr.shape[0], axis=0)
    k_max = mat.shape[1] - 2
    pred = _cdf_at_y(pmf_cdf(mat), y_arr, k_max)
    n = pred.shape[0]
    order = np.argsort(pred, kind="mergesort")
    empirical = np.empty(n, dtype=float)
    empirical[order] = (np.arange(1, n + 1, dtype=float)) / (n + 1.0)
    iso = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    iso.fit(pred, empirical)
    return Recalibrator(isotonic=iso)


def apply_recalibration(recal, pmf) -> np.ndarray:
    """Apply a single increasing CDF transform, then renormalize the PMF."""
    mat = as_pmf_matrix(pmf)
    cdf = pmf_cdf(mat)
    mapped = np.asarray(recal.isotonic.predict(cdf.reshape(-1)), dtype=float)
    new_cdf = mapped.reshape(cdf.shape)
    new_cdf = np.maximum.accumulate(np.clip(new_cdf, 0.0, 1.0), axis=1)
    new_cdf[:, -1] = 1.0
    new_pmf = np.empty_like(mat)
    new_pmf[:, 0] = new_cdf[:, 0]
    new_pmf[:, 1:] = np.diff(new_cdf, axis=1)
    new_pmf = np.clip(new_pmf, 0.0, None)
    new_pmf = new_pmf / np.maximum(new_pmf.sum(axis=1, keepdims=True), 1e-15)
    if np.asarray(pmf).ndim == 1:
        return new_pmf[0]
    return new_pmf
