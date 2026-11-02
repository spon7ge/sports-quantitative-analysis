"""Correlation groups and drop-group ablation summaries.

Leave-one-out ablation scores every member of a redundant group near
zero, because the others fill in. Dropping the group is the question
that asks whether the whole block is replaceable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from models.shared.metrics import interval_coverage
from models.shared.quantile_offsets import row_mean_pinball

COVERAGE_NOMINAL = 0.80


def feature_groups(
    frame: pd.DataFrame,
    min_abs_rho: float = 0.8,
) -> list[list[str]]:
    """Cluster columns by absolute Spearman correlation.

    Average linkage joins two columns when their distance ``1 - |rho|``
    is at most ``1 - min_abs_rho``. Members keep the frame's column order.
    """
    columns = [str(column) for column in frame.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate feature names")
    if not 0.0 < float(min_abs_rho) < 1.0:
        raise ValueError("min_abs_rho must be between 0 and 1")
    if len(columns) <= 1:
        return [columns] if columns else []

    rho = frame.corr(method="spearman").abs().fillna(0.0)
    matrix = rho.to_numpy(dtype=float)
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 1.0)
    distance = np.clip(1.0 - matrix, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=1.0 - float(min_abs_rho),
        criterion="distance",
    )
    groups: list[list[str]] = []
    for label in pd.unique(labels):
        groups.append([column for column, item in zip(columns, labels) if item == label])
    return groups


def date_mean_se(values, dates) -> tuple[float, float]:
    """Mean of per-date averages, and its standard error.

    Rows from the same game move together, so the independent unit is
    the date, not the player-game. ``se`` uses the sample standard
    deviation of those daily means.
    """
    observed = np.asarray(values, dtype=float)
    if observed.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if observed.size == 0 or not np.isfinite(observed).all():
        raise ValueError("values must be finite")
    when = np.asarray(dates)
    if when.shape != observed.shape:
        raise ValueError("dates must align with values")
    daily = pd.Series(observed).groupby(when, sort=False).mean()
    if len(daily) < 2:
        raise ValueError("standard error needs at least two dates")
    center = float(daily.mean())
    se = float(daily.std(ddof=1) / np.sqrt(len(daily)))
    return center, se


def group_ablation(
    frame: pd.DataFrame,
    features: Sequence[str],
    groups: Sequence[Sequence[str]],
    folds: Sequence[dict],
    fit_predict: Callable,
    taus,
    seeds: Sequence[int] = (0, 1, 2),
    *,
    target_col: str = "pts",
    date_col: str = "game_date",
) -> pd.DataFrame:
    """Drop-group-and-retrain delta in mean pinball, one row per group and fold.

    ``fit_predict(train, val, columns, seed)`` returns sorted quantiles
    with shape ``(n_val, len(taus))``. Delta is loss without the group
    minus loss with every feature, so a positive delta means the group
    helps. ``coverage_gap_delta`` is the same difference for the absolute
    gap between q10–q90 coverage and 80%. A positive coverage delta means
    removing the group made the interval less calibrated.
    """
    levels = np.asarray(taus, dtype=float).reshape(-1)
    low = int(np.abs(levels - 0.10).argmin())
    high = int(np.abs(levels - 0.90).argmin())
    if low == high:
        raise ValueError("taus must include distinct knots near 0.10 and 0.90")
    names = [str(column) for column in features]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names")
    records: list[dict[str, object]] = []
    for fold in folds:
        train = frame.loc[fold["train_mask"]]
        val = frame.loc[fold["val_mask"]]
        outcome = val[target_col].to_numpy(dtype=float)
        dates = val[date_col].to_numpy()

        def scored(columns: Sequence[str]) -> tuple[np.ndarray, float]:
            row_losses = []
            gaps = []
            for seed in seeds:
                raw = np.asarray(
                    fit_predict(train, val, list(columns), seed),
                    dtype=float,
                )
                if raw.ndim != 2 or raw.shape != (len(val), levels.size):
                    raise ValueError(
                        f"fit_predict returned {raw.shape}, expected "
                        f"({len(val)}, {levels.size})"
                    )
                grid = np.sort(raw, axis=1)
                row_losses.append(row_mean_pinball(outcome, grid, levels))
                coverage = interval_coverage(outcome, grid[:, low], grid[:, high])
                gaps.append(abs(coverage - COVERAGE_NOMINAL))
            return np.mean(row_losses, axis=0), float(np.mean(gaps))

        base_rows, base_gap = scored(names)
        for group in groups:
            dropped = [column for column in group if column in names]
            if not dropped:
                continue
            kept = [column for column in names if column not in set(dropped)]
            rows, gap = scored(kept)
            delta, se = date_mean_se(rows - base_rows, dates)
            records.append(
                {
                    "group": "+".join(dropped),
                    "fold": fold["label"],
                    "delta": delta,
                    "se": se,
                    "coverage_gap_delta": gap - base_gap,
                }
            )
    return pd.DataFrame(records)


def decide_groups(
    records: pd.DataFrame,
    *,
    min_positive_folds: int = 3,
    se_multiple: float = 2.0,
) -> pd.DataFrame:
    """Keep, drop, or leave a group alone.

    Keep when the mean fold delta is above ``se_multiple`` standard
    errors of that mean and the delta is positive on at least
    ``min_positive_folds`` folds. Drop only when the delta is <= 0 on
    most folds and the average coverage gap does not get worse. Anything
    in between stays, including a group that is flat on pinball but
    widens the coverage gap when removed.
    """
    required = {"group", "fold", "delta", "coverage_gap_delta"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = []
    for group, part in records.groupby("group", sort=False):
        deltas = part["delta"].to_numpy(dtype=float)
        coverage = part["coverage_gap_delta"].to_numpy(dtype=float)
        if deltas.size == 0 or not np.isfinite(deltas).all() or not np.isfinite(coverage).all():
            raise ValueError(f"non-finite ablation rows for {group}")
        n_folds = int(deltas.size)
        mean_delta = float(deltas.mean())
        se = (
            float(deltas.std(ddof=1) / np.sqrt(n_folds))
            if n_folds > 1
            else float("nan")
        )
        n_positive = int(np.sum(deltas > 0))
        n_nonpositive = int(np.sum(deltas <= 0))
        coverage_delta = float(coverage.mean())
        beats_noise = bool(
            np.isfinite(se)
            and mean_delta > float(se_multiple) * se
            and n_positive >= int(min_positive_folds)
        )
        if beats_noise:
            decision = "keep"
        elif n_nonpositive > n_folds / 2 and coverage_delta <= 0.0:
            decision = "drop"
        else:
            decision = "inconclusive"
        rows.append(
            {
                "group": group,
                "delta": mean_delta,
                "se": se,
                "n_positive": n_positive,
                "n_folds": n_folds,
                "coverage_gap_delta": coverage_delta,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)
