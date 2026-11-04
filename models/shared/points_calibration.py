"""CDF recalibration for integer points.

``fit_points_cdf_calibrator`` pools every integer threshold. The line
map is one randomized PIT per row, then isotonic regression of the
empirical CDF of those values on the values themselves. A date on or
after the 2026-27 opener is rejected, so neither fit can see the holdout.
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


def quantile_cdf(points, knots, taus) -> np.ndarray:
    """F(x) by linear interpolation of a monotone quantile grid.

    Outside the grid, the CDF stays at the end taus.
    """
    point = np.asarray(points, dtype=float).reshape(-1)
    grid = np.asarray(knots, dtype=float)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    if grid.ndim != 2 or grid.shape != (point.shape[0], levels.shape[0]):
        raise ValueError("cdf knots do not match thresholds")
    if levels.size < 2 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("taus must be strictly increasing")
    if not np.isfinite(grid).all() or not np.isfinite(point).all():
        raise ValueError("cdf inputs must be finite")
    if np.any(np.diff(grid, axis=1) < 0.0):
        raise ValueError("knots must be nondecreasing")
    index = np.clip(
        np.sum(grid <= point[:, None], axis=1) - 1, 0, levels.shape[0] - 2
    )
    left = np.take_along_axis(grid, index[:, None], axis=1)[:, 0]
    right = np.take_along_axis(grid, index[:, None] + 1, axis=1)[:, 0]
    low = levels[index]
    high = levels[index + 1]
    width = np.maximum(right - left, 1e-8)
    weight = np.clip((point - left) / width, 0.0, 1.0)
    out = low + weight * (high - low)
    out = np.where(point <= grid[:, 0], levels[0], out)
    out = np.where(point >= grid[:, -1], levels[-1], out)
    return out


def quantile_line_probabilities(line, knots, taus) -> tuple[np.ndarray, np.ndarray]:
    """P(points > line) and P(points == line) from the raw quantile CDF.

    Over is ``1 - F(floor(line))``. A half-point line has no push. An
    integer line pushes, and that atom is ``F(k) - F(k - 1)``, so it is
    not counted in the over. Outside the knot grid, F stays at the end
    taus, the same rule as ``quantile_cdf``.
    """
    level = np.asarray(line, dtype=float).reshape(-1)
    grid = np.asarray(knots, dtype=float)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    if grid.ndim != 2:
        raise ValueError("knots")
    if level.size == 1 and grid.shape[0] != 1:
        level = np.repeat(level, grid.shape[0])
    elif grid.shape[0] == 1 and level.size != 1:
        grid = np.repeat(grid, level.size, axis=0)
    if level.shape[0] != grid.shape[0]:
        raise ValueError("line")
    if level.size == 0:
        return level.copy(), level.copy()
    if not np.isfinite(level).all():
        raise ValueError("line")
    nearest = np.rint(level)
    integer = np.abs(level - nearest) <= 1e-6
    settled = np.where(integer, nearest, np.floor(level))
    at = quantile_cdf(settled, grid, taus)
    left = quantile_cdf(settled - 1.0, grid, taus)
    over = 1.0 - at
    push = np.where(integer, np.clip(at - left, 0.0, None), 0.0)
    return over, push


def _integer_points(y) -> np.ndarray:
    raw = np.asarray(y, dtype=float).reshape(-1)
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("points")
    nearest = np.rint(raw)
    if float(np.max(np.abs(raw - nearest))) > 1e-6:
        raise ValueError("points must be integers")
    return nearest


def deterministic_quantile_pit(y, knots, taus) -> np.ndarray:
    """Midpoint of the integer atom, ``(F(y - 1) + F(y)) / 2``.

    This is the randomized PIT at ``U = 1/2``. A fresh uniform would give
    the same row a different value whenever a later call scored a subset.
    """
    outcome = _integer_points(y)
    midpoint = np.full(outcome.shape[0], 0.5)
    return randomized_quantile_pit(outcome, knots, taus, midpoint)


def randomized_quantile_pit(y, knots, taus, uniform) -> np.ndarray:
    """u = F(y - 1) + U * (F(y) - F(y - 1)).

    ``U`` is one draw per row on [0, 1]. The atom at an integer point is
    spread across that interval so the PIT of a discrete total can be uniform.
    """
    outcome = _integer_points(y)
    draw = np.asarray(uniform, dtype=float).reshape(-1)
    if draw.shape[0] != outcome.shape[0]:
        raise ValueError("uniform draw")
    if not np.isfinite(draw).all() or np.any((draw < 0.0) | (draw > 1.0)):
        raise ValueError("uniform draw must lie in [0, 1]")
    grid = np.asarray(knots, dtype=float)
    f_y = quantile_cdf(outcome, grid, taus)
    f_left = quantile_cdf(outcome - 1.0, grid, taus)
    return f_left + draw * np.clip(f_y - f_left, 0.0, None)


def fit_pit_recalibrator(u) -> IsotonicRegression:
    """Isotonic regression of the empirical CDF of ``u`` on ``u``.

    The fitted map ``R`` satisfies ``R(F(x))`` as the calibrated CDF.
    """
    values = np.asarray(u, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("pit sample")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pit sample must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_u = values[order]
    empirical = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_u[stop] == sorted_u[start]:
            stop += 1
        empirical[start:stop] = stop / values.size
        start = stop
    model = IsotonicRegression(
        increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    model.fit(sorted_u, empirical)
    return model


def _holdout_dates(dates, holdout_start):
    when = np.asarray(pd.to_datetime(dates), dtype="datetime64[ns]")
    if np.any(when >= np.datetime64(holdout_start, "ns")):
        raise ValueError("holdout rows are not calibration data")
    return when


def fit_randomized_pit_map(
    y, knots, taus, uniform, dates, holdout_start=HOLDOUT_START
) -> IsotonicRegression:
    """Fit ``R`` on randomized PITs. Holdout dates are refused."""
    outcome = np.asarray(y, dtype=float).reshape(-1)
    when = _holdout_dates(dates, holdout_start)
    if when.shape[0] != outcome.shape[0]:
        raise ValueError("dates")
    u = randomized_quantile_pit(y, knots, taus, uniform)
    return fit_pit_recalibrator(u)


def _inverse_isotonic(calibrator, alpha) -> np.ndarray:
    """Smallest model probability whose calibrated CDF reaches ``alpha``."""
    xs = np.asarray(calibrator.X_thresholds_, dtype=float)
    ys = np.asarray(calibrator.y_thresholds_, dtype=float)
    keep = np.ones(xs.size, dtype=bool)
    if xs.size > 1:
        keep[1:] = np.diff(ys) > 1e-15
    return np.interp(np.asarray(alpha, dtype=float), ys[keep], xs[keep])


def probability_over(line, knots, taus, calibrator) -> np.ndarray:
    """P(over L) = 1 - R(F(L))."""
    raw = np.asarray(line, dtype=float).reshape(-1)
    grid = np.asarray(knots, dtype=float)
    if grid.ndim != 2:
        raise ValueError("knots")
    if not np.isfinite(raw).all():
        raise ValueError("line")
    if grid.shape[0] == 1 and raw.shape[0] > 1:
        grid = np.repeat(grid, raw.shape[0], axis=0)
    elif raw.shape[0] == 1 and grid.shape[0] > 1:
        raw = np.repeat(raw, grid.shape[0])
    if raw.shape[0] != grid.shape[0]:
        raise ValueError("line")
    model_cdf = quantile_cdf(raw, grid, taus)
    calibrated = np.asarray(calibrator.predict(model_cdf), dtype=float)
    return 1.0 - calibrated


def calibrated_quantile_grid(knots, taus, calibrator) -> np.ndarray:
    """Quantiles of the recalibrated CDF, ``F^{-1}(R^{-1}(tau))``."""
    grid = np.asarray(knots, dtype=float)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    if grid.ndim != 2 or grid.shape[1] != levels.shape[0]:
        raise ValueError("knots")
    if levels.size < 2 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("taus must be strictly increasing")
    if not np.isfinite(grid).all():
        raise ValueError("knots")
    targets = _inverse_isotonic(calibrator, levels)
    return np.vstack([np.interp(targets, levels, row) for row in grid])


def pit_map_from_thresholds(pit_x, pit_y) -> IsotonicRegression:
    """Rebuild ``R`` from the stored isotonic knots."""
    xs = np.asarray(pit_x, dtype=float).reshape(-1)
    ys = np.asarray(pit_y, dtype=float).reshape(-1)
    if xs.size == 0 or xs.shape != ys.shape:
        raise ValueError("pit map")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("pit map")
    if np.any(np.diff(xs) < 0.0) or np.any((ys < 0.0) | (ys > 1.0)):
        raise ValueError("pit map")
    model = IsotonicRegression(
        increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    model.fit(xs, ys)
    return model


def pit_recalibration_payload(
    calibrator, taus, dates, *, target: str, clip_low: float, clip_high: float
) -> dict:
    """Bundle the PIT map. ``R`` is stored as its isotonic knots."""
    when = pd.to_datetime(np.asarray(dates).reshape(-1))
    if len(when) == 0 or bool(when.isna().any()):
        raise ValueError("dates")
    levels = np.asarray(taus, dtype=float).reshape(-1)
    return {
        "method": "pit_cdf",
        "pit_x": np.asarray(calibrator.X_thresholds_, dtype=float),
        "pit_y": np.asarray(calibrator.y_thresholds_, dtype=float),
        "taus": levels,
        "target": str(target),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "date_start": pd.Timestamp(when.min()).date().isoformat(),
        "date_end": pd.Timestamp(when.max()).date().isoformat(),
    }


def pit_walk_forward(
    y, Q, taus, fold_ids, rng, *, eval_mask=None, dates=None
) -> pd.DataFrame:
    """Fit ``R`` on earlier folds and score pinball on the next.

    ``R`` is the isotonic map of the midpoint PIT, so the fit does not
    draw a uniform. ``eval_mask`` restricts the scored rows and does not
    change the fit. Coverage jitter is one ``U(-1/2, 1/2)`` per row from
    ``rng``, then masked with the scored rows. Subsetting first and
    drawing again would give those rows a different jitter.
    """
    from models.shared.quantile_offsets import (  # noqa: PLC0415
        calibration_table,
        line_band_gap,
    )

    outcome = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(Q, dtype=float)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    folds = np.asarray(fold_ids)
    if folds.shape[0] != outcome.shape[0]:
        raise ValueError("fold_ids")
    if dates is not None:
        when = _holdout_dates(dates, HOLDOUT_START)
        if when.shape[0] != outcome.shape[0]:
            raise ValueError("dates")
    if eval_mask is None:
        mask = np.ones(outcome.shape[0], dtype=bool)
    else:
        mask = np.asarray(eval_mask, dtype=bool)
        if mask.shape[0] != outcome.shape[0]:
            raise ValueError("eval_mask")
    jitter = np.asarray(rng.uniform(-0.5, 0.5, outcome.shape[0]), dtype=float)
    if jitter.shape != outcome.shape or not np.isfinite(jitter).all():
        raise ValueError("jitter")
    rows: list[dict[str, float | int]] = []
    for fold_id in np.unique(folds)[1:]:
        past = folds < fold_id
        current = (folds == fold_id) & mask
        if not bool(past.any()) or not bool(current.any()):
            continue
        calibrator = fit_pit_recalibrator(
            deterministic_quantile_pit(outcome[past], grid[past], levels)
        )
        adjusted = calibrated_quantile_grid(grid[current], levels, calibrator)
        hits = (outcome[current] + jitter[current])[:, None] <= adjusted
        table = calibration_table(outcome[current], adjusted, levels, rng, hits=hits)
        rows.append(
            {
                "fold": int(fold_id),
                "n": int(current.sum()),
                "max_abs_gap": line_band_gap(table),
                "mean_pinball": float(table["pinball"].mean()),
            }
        )
    return pd.DataFrame(rows)
