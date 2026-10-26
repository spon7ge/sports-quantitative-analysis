"""Isotonic map from a points CDF to a recalibrated CDF.

Pairs are pooled over every integer threshold. A date on or after the
2026-27 opener is rejected, so this cannot be fit on the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HOLDOUT_START = np.datetime64("2026-10-20")


@dataclass
class NestedCalibration:
    pmf: np.ndarray
    curves: dict[int, np.ndarray]
    evaluated: np.ndarray
    calibrators: dict[int, IsotonicRegression]


def fit_points_cdf_calibrator(cdf, y, dates, holdout_start=HOLDOUT_START) -> IsotonicRegression:
    """Fit g on (F̂(j), 1[y ≤ j]) for every integer j in the CDF support."""
    values = np.asarray(cdf, dtype=float)
    outcome = np.rint(np.asarray(y, dtype=float)).astype(int)
    when = np.asarray(pd.to_datetime(dates), dtype="datetime64[ns]")
    if values.ndim != 2:
        raise ValueError("cdf")
    if outcome.shape[0] != values.shape[0] or when.shape[0] != values.shape[0]:
        raise ValueError("rows")
    if not np.isfinite(values).all():
        raise ValueError("non-finite cdf")
    if np.any(when >= np.datetime64(holdout_start)):
        raise ValueError("holdout")
    width = values.shape[1]
    indicators = (outcome[:, None] <= np.arange(width)[None, :]).astype(float)
    calibrator = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    calibrator.fit(values.ravel(), indicators.ravel())
    return calibrator


def apply_points_cdf_calibrator(cdf, calibrator) -> np.ndarray:
    """Map the CDF, pin the last knot at 1, and difference back to a PMF."""
    values = np.asarray(cdf, dtype=float)
    mapped = np.asarray(calibrator.predict(values.ravel()), dtype=float).reshape(values.shape)
    mapped = np.maximum.accumulate(np.clip(mapped, 0.0, 1.0), axis=1)
    mapped[:, -1] = 1.0
    pmf = np.empty_like(mapped)
    pmf[:, 0] = mapped[:, 0]
    pmf[:, 1:] = np.diff(mapped, axis=1)
    pmf = np.clip(pmf, 0.0, None)
    totals = pmf.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("empty pmf")
    return pmf / totals


def calibrator_curve(calibrator, points) -> np.ndarray:
    """g at the requested CDF knots, for the stability check."""
    grid = np.asarray(points, dtype=float)
    return np.asarray(calibrator.predict(grid), dtype=float)


def nested_points_calibrators(
    pmf,
    y,
    fold_ids,
    dates,
    *,
    folds=(3, 4, 5),
    holdout_start=HOLDOUT_START,
    curve_points=None,
) -> NestedCalibration:
    """Fit g on folds before k and apply it only to fold k.

    Rows in earlier folds keep their original PMF. The returned calibrators
    are the ones printed to check that g is stable across k.
    """
    raw = np.asarray(pmf, dtype=float)
    outcome = np.asarray(y)
    folds_now = np.asarray(fold_ids)
    when = np.asarray(dates)
    if raw.shape[0] != folds_now.shape[0]:
        raise ValueError("rows")
    calibrated = np.array(raw, dtype=float, copy=True)
    evaluated = np.zeros(raw.shape[0], dtype=bool)
    curves: dict[int, np.ndarray] = {}
    calibrators: dict[int, IsotonicRegression] = {}
    grid = np.linspace(0.0, 1.0, 11) if curve_points is None else np.asarray(curve_points, dtype=float)
    cdf = np.cumsum(raw, axis=1)
    for fold_id in folds:
        fit_mask = folds_now < int(fold_id)
        apply_mask = folds_now == int(fold_id)
        if not np.any(fit_mask) or not np.any(apply_mask):
            raise ValueError(f"fold {fold_id}")
        calibrator = fit_points_cdf_calibrator(
            cdf[fit_mask],
            outcome[fit_mask],
            when[fit_mask],
            holdout_start=holdout_start,
        )
        calibrators[int(fold_id)] = calibrator
        curves[int(fold_id)] = calibrator_curve(calibrator, grid)
        calibrated[apply_mask] = apply_points_cdf_calibrator(cdf[apply_mask], calibrator)
        evaluated |= apply_mask
    return NestedCalibration(
        pmf=calibrated,
        curves=curves,
        evaluated=evaluated,
        calibrators=calibrators,
    )


def keep_calibrator(*, rps_ci_high: float, reliability_ci_high: float) -> bool:
    """Keep g only when both nested differences sit entirely below zero."""
    return float(rps_ci_high) < 0.0 and float(reliability_ci_high) < 0.0
