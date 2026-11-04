"""Per-quantile offsets in units of each row's predicted width.

An additive shift on the 10–90 interval can hit 80% coverage while the
quantiles around the median, where the lines sit, stay wrong. These
offsets are one number per tau. Multiplying by the row's q10–q90 width
makes the same correction small for a bench player and large for a scorer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_WIDTH_FLOOR = 1.0


def _as_grid(y, Q, taus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcome = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(Q, dtype=float)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    if grid.ndim != 2 or grid.shape != (outcome.shape[0], levels.shape[0]):
        raise ValueError("Q must have shape (n, len(taus))")
    if levels.size == 0 or not np.all((levels > 0.0) & (levels < 1.0)):
        raise ValueError("taus must be in (0, 1)")
    if not np.isfinite(grid).all() or not np.isfinite(outcome).all():
        raise ValueError("y and Q must be finite")
    return outcome, grid, levels


def interval_width(Q, taus, *, floor: float = _WIDTH_FLOOR) -> np.ndarray:
    """q90 - q10 for each row, with a floor so a zero-width row cannot divide."""
    _outcome, grid, levels = _as_grid(np.zeros(len(Q)), Q, taus)
    low = int(np.abs(levels - 0.10).argmin())
    high = int(np.abs(levels - 0.90).argmin())
    if low == high:
        raise ValueError("taus must include distinct knots near 0.10 and 0.90")
    return np.maximum(grid[:, high] - grid[:, low], float(floor))


def row_mean_pinball(y, Q, taus) -> np.ndarray:
    """Pinball averaged across the quantile grid, one value per row.

    The row vector is what a date-level standard error has to average.
    Averaging it again over rows is ``mean_pinball``.
    """
    outcome, grid, levels = _as_grid(y, Q, taus)
    residual = outcome[:, None] - grid
    loss = np.maximum(levels * residual, (levels - 1.0) * residual)
    return loss.mean(axis=1)


def mean_pinball(y, Q, taus) -> float:
    """Mean pinball over rows and the whole quantile grid."""
    return float(row_mean_pinball(y, Q, taus).mean())


def jittered_hits(y, Q, taus, rng) -> np.ndarray:
    """Boolean coverage of the jittered outcome at each quantile.

    Same draw ``calibration_table`` uses. Shape is ``(n, len(taus))``.
    """
    outcome, grid, _levels = _as_grid(y, Q, taus)
    jitter = np.asarray(rng.uniform(-0.5, 0.5, outcome.shape[0]), dtype=float)
    if jitter.shape != outcome.shape or not np.isfinite(jitter).all():
        raise ValueError("jitter")
    return (outcome + jitter)[:, None] <= grid


def _as_hits(hits, n_levels: int) -> np.ndarray:
    covered = np.asarray(hits)
    if covered.ndim != 2 or covered.shape[1] != n_levels or covered.shape[0] == 0:
        raise ValueError("hits")
    if covered.dtype == bool:
        return covered
    values = np.asarray(covered, dtype=float)
    if not np.isfinite(values).all() or np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("hits")
    return values.astype(bool)


def calibration_table(y, Q, taus, rng, *, hits=None) -> pd.DataFrame:
    """Per-tau coverage on jittered outcomes, plus pinball on the raw outcomes.

    Integer points make ``y <= q`` true whenever q lands on that integer.
    Jittering by less than half a point is the continuous target an x.5
    line is actually pricing. Pinball stays on the recorded total.
    ``hits`` reuses a ``jittered_hits`` draw so a bootstrap of those hits
    describes the same coverage table.
    """
    outcome, grid, levels = _as_grid(y, Q, taus)
    if hits is None:
        covered = jittered_hits(outcome, grid, levels, rng)
    else:
        covered = _as_hits(hits, levels.shape[0])
        if covered.shape[0] != outcome.shape[0]:
            raise ValueError("hits")
    residual = outcome[:, None] - grid
    pinball = np.maximum(levels * residual, (levels - 1.0) * residual).mean(axis=0)
    coverage = covered.mean(axis=0)
    return pd.DataFrame(
        {
            "tau": levels,
            "coverage": coverage,
            "gap": coverage - levels,
            "pinball": pinball,
        }
    )


def reliability_pairs(y, Q, taus, rng) -> tuple[np.ndarray, np.ndarray]:
    """Binary hits and nominal probabilities for a reliability diagram.

    One pair per game and tau. The outcome is 1 when the jittered point is
    at or below that quantile, and the predicted probability is tau. The
    jitter is the same draw ``calibration_table`` uses.
    """
    outcome, grid, levels = _as_grid(y, Q, taus)
    jitter = np.asarray(rng.uniform(-0.5, 0.5, outcome.shape[0]), dtype=float)
    if jitter.shape != outcome.shape or not np.isfinite(jitter).all():
        raise ValueError("jitter")
    covered = (outcome + jitter)[:, None] <= grid
    predicted = np.broadcast_to(levels, covered.shape).reshape(-1)
    return covered.reshape(-1).astype(float), predicted.copy()


def quantile_reliability_bins(taus) -> np.ndarray:
    """Edges that put each tau in its own bin, from 0 through 1.

    The cuts between knots are the midpoints, so a reliability diagram draws
    one point per trained quantile.
    """
    levels = np.asarray(taus, dtype=float).reshape(-1)
    if levels.size == 0 or not np.all((levels > 0.0) & (levels < 1.0)):
        raise ValueError("taus must be in (0, 1)")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("taus must be strictly increasing")
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    return np.concatenate(([0.0], midpoints, [1.0]))


def line_band_gap(table: pd.DataFrame, *, low: float = 0.3, high: float = 0.7) -> float:
    """Largest absolute coverage gap for taus between ``low`` and ``high``."""
    band = table.loc[table["tau"].between(low, high), "gap"]
    if band.empty:
        raise ValueError("no taus in the line band")
    return float(band.abs().max())


def cluster_gap_band(
    covered,
    taus,
    clusters,
    rng,
    *,
    low: float = 0.3,
    high: float = 0.7,
    n_boot: int = 2000,
) -> dict[str, float | int | bool]:
    """Sampling noise of the line-band coverage gap, resampling whole clusters.

    Teammates share a game's pace and blowout, so an independent-row
    binomial interval treats dependent rows as extra samples. Each draw
    resamples cluster ids with replacement and keeps every row of a drawn
    cluster, weighted by that cluster's row count. Pass a game id, or a
    date when the game id is not on the row. The half-width is the 95th
    percentile, over taus in ``[low, high]``, of the bootstrap max
    ``|coverage - observed coverage|``. ``inside`` is true when the
    observed ``max |coverage - tau|`` is no larger than that half-width.
    Games from one player on different clusters are not held together.
    """
    levels = np.asarray(taus, dtype=float).reshape(-1)
    if levels.size == 0 or not np.all((levels > 0.0) & (levels < 1.0)):
        raise ValueError("taus must be in (0, 1)")
    draws = int(n_boot)
    if draws <= 0:
        raise ValueError("n_boot")
    chosen = (levels >= low) & (levels <= high)
    if not bool(np.any(chosen)):
        raise ValueError("no taus in the line band")
    hits = _as_hits(covered, levels.shape[0]).astype(float)[:, chosen]
    labels = np.asarray(clusters)
    if labels.shape != (hits.shape[0],) or bool(pd.isna(labels).any()):
        raise ValueError("clusters")
    codes, _uniques = pd.factorize(labels, sort=False)
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise ValueError("clusters")
    cluster_n = np.bincount(codes, minlength=n_clusters).astype(float)
    cluster_sum = np.column_stack(
        [
            np.bincount(codes, weights=hits[:, column], minlength=n_clusters)
            for column in range(hits.shape[1])
        ]
    )
    point = cluster_sum.sum(axis=0) / float(cluster_n.sum())
    max_abs_gap = float(np.max(np.abs(point - levels[chosen])))
    picked = np.asarray(rng.integers(0, n_clusters, size=(draws, n_clusters)))
    weights = np.zeros((draws, n_clusters), dtype=np.int32)
    np.add.at(
        weights,
        (np.repeat(np.arange(draws), n_clusters), picked.ravel()),
        1,
    )
    denom = weights @ cluster_n
    boot = (weights @ cluster_sum) / denom[:, None]
    deviate = np.max(np.abs(boot - point), axis=1)
    half_width = float(np.quantile(deviate, 0.95))
    return {
        "max_abs_gap": max_abs_gap,
        "half_width": half_width,
        "n": int(hits.shape[0]),
        "n_clusters": n_clusters,
        "inside": bool(max_abs_gap <= half_width),
    }


def fit_offsets(y, Q, taus, rng) -> np.ndarray:
    """Per-tau offsets in units of each row's q10–q90 width."""
    outcome, grid, levels = _as_grid(y, Q, taus)
    jitter = np.asarray(rng.uniform(-0.5, 0.5, outcome.shape[0]), dtype=float)
    if jitter.shape != outcome.shape or not np.isfinite(jitter).all():
        raise ValueError("jitter")
    width = interval_width(grid, levels)
    residual = ((outcome + jitter)[:, None] - grid) / width[:, None]
    return np.array(
        [
            float(np.quantile(residual[:, column], level))
            for column, level in enumerate(levels)
        ]
    )


def apply_offsets(Q, c, taus) -> np.ndarray:
    """Add ``c * width`` to each quantile, then sort the row."""
    _outcome, grid, levels = _as_grid(np.zeros(len(Q)), Q, taus)
    offsets = np.asarray(c, dtype=float).reshape(-1)
    if offsets.shape != levels.shape or not np.isfinite(offsets).all():
        raise ValueError("offsets")
    shifted = grid + offsets * interval_width(grid, levels)[:, None]
    shifted.sort(axis=1)
    return shifted


def scaled_offset_payload(
    offsets,
    taus,
    dates,
    *,
    target: str,
    clip_low: float,
    clip_high: float,
) -> dict:
    """Bundle fields for a width-scaled offset vector. No group table."""
    when = pd.to_datetime(np.asarray(dates).reshape(-1))
    if len(when) == 0 or bool(when.isna().any()):
        raise ValueError("dates")
    vector = np.asarray(offsets, dtype=float).reshape(-1)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    if vector.shape != levels.shape or not np.isfinite(vector).all():
        raise ValueError("offsets")
    return {
        "method": "scaled_quantile",
        "offsets": vector,
        "taus": levels,
        "width_floor": _WIDTH_FLOOR,
        "target": str(target),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "date_start": pd.Timestamp(when.min()).date().isoformat(),
        "date_end": pd.Timestamp(when.max()).date().isoformat(),
    }


def offset_walk_forward(
    y,
    Q,
    taus,
    fold_ids,
    rng,
    *,
    eval_mask=None,
    low: float = 0.3,
    high: float = 0.7,
) -> pd.DataFrame:
    """Fit offsets on earlier folds and score the coverage gap on the next.

    ``eval_mask`` restricts the scored rows (the early-season slice) and
    does not change which rows the offset is fit on.
    """
    outcome, grid, levels = _as_grid(y, Q, taus)
    folds = np.asarray(fold_ids)
    if folds.shape[0] != outcome.shape[0]:
        raise ValueError("fold_ids")
    if eval_mask is None:
        mask = np.ones(outcome.shape[0], dtype=bool)
    else:
        mask = np.asarray(eval_mask, dtype=bool)
        if mask.shape[0] != outcome.shape[0]:
            raise ValueError("eval_mask")
    rows: list[dict[str, float | int]] = []
    for fold_id in np.unique(folds)[1:]:
        past = folds < fold_id
        current = (folds == fold_id) & mask
        if not bool(past.any()) or not bool(current.any()):
            continue
        fitted = fit_offsets(outcome[past], grid[past], levels, rng)
        adjusted = apply_offsets(grid[current], fitted, levels)
        table = calibration_table(outcome[current], adjusted, levels, rng)
        rows.append(
            {
                "fold": int(fold_id),
                "n": int(current.sum()),
                "max_abs_gap": line_band_gap(table, low=low, high=high),
                "mean_pinball": float(table["pinball"].mean()),
            }
        )
    return pd.DataFrame(rows)
