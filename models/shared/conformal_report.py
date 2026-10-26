"""Walk-forward conformal tables shared by the minutes and rate notebooks.

The offset math lives in ``models.shared.conformal``. This module turns
an out-of-fold frame into the go/no-go table and the nested backtest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.shared.conformal import apply_offsets, fit_offsets
from models.shared.metrics import paired_date_bootstrap, pinball_loss
from models.shared.minutes_sampler import row_crps

MONTHS = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun")
_WIS_INTERVALS = (
    (0.80, "q_0.40", "q_0.60"),
    (0.60, "q_0.30", "q_0.70"),
    (0.40, "q_0.20", "q_0.80"),
    (0.20, "q_0.10", "q_0.90"),
    (0.10, "q_0.05", "q_0.95"),
)


@dataclass(frozen=True)
class ConformalSpec:
    """Target, lag, tiers, and clip for one conformal report."""

    target_col: str
    lag_col: str
    tiers: tuple[tuple[str, float | None, float | None], ...]
    clip_low: float
    clip_high: float
    min_n: int = 300


def calibration_table(
    y,
    dates,
    base: pd.DataFrame,
    group,
    *,
    spec: ConformalSpec,
    starting,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Per-knot gap and 80/90 coverage, with a date-clustered interval."""
    actual = np.asarray(y, dtype=float)
    when = pd.to_datetime(np.asarray(dates)).to_numpy()
    labels = np.asarray(group)
    knots = _knot_columns(base)
    levels = {key: float(key.split("_", 1)[1]) for key in knots}
    played = np.asarray(starting)
    slices: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(actual), dtype=bool)),
        ("starter", played == 1),
        ("bench", played == 0),
    ]
    for name in _group_names(spec):
        slices.append((name, labels == name))
    month = pd.to_datetime(when).strftime("%b")
    for name in MONTHS:
        mask = month == name
        if np.any(mask):
            slices.append((name, mask))
    rows = [
        row
        for name, mask in slices
        if np.any(mask)
        for row in _slice_rows(
            name, mask, actual, when, base, knots, levels, n_boot, seed
        )
    ]
    return pd.DataFrame(rows)


def nested_offset_backtest(
    oof: pd.DataFrame,
    base: pd.DataFrame,
    group,
    spec: ConformalSpec,
    *,
    folds: Sequence[int] = (3, 4, 5),
    n_boot: int,
    seed: int,
) -> dict:
    """Fit offsets on folds before k and score fold k.

    ``scores['wis_diff']`` and ``scores['crps_diff']`` are calibrated minus
    raw. Lower is better.
    """
    if len(oof) != len(base) or len(oof) != len(group):
        raise ValueError("oof, base, and group lengths differ")
    if spec.target_col not in oof.columns:
        raise KeyError(spec.target_col)
    labels = np.asarray(group)
    knots = _knot_columns(base)
    levels = {key: float(key.split("_", 1)[1]) for key in knots}
    level_grid = np.array([levels[key] for key in knots])
    y = oof[spec.target_col].to_numpy(dtype=float)
    when_all = pd.to_datetime(oof["game_date"]).to_numpy()
    fold_id = oof["fold_id"].to_numpy()
    group_order = _group_names(spec)
    nested_rows = []
    score_rows = []
    delta_rows = []
    count_rows = []
    offset_tables = {}
    for k in folds:
        fit_mask = fold_id < k
        test_mask = fold_id == k
        offsets = fit_offsets(
            base.loc[fit_mask],
            y[fit_mask],
            labels[fit_mask],
            levels,
            min_n=spec.min_n,
            group_order=group_order,
        )
        counts = pd.Series(labels[fit_mask]).value_counts()
        for name, count in counts.items():
            count_rows.append(
                {
                    "fold": k,
                    "group": str(name),
                    "n": int(count),
                    "fit": int(count) >= spec.min_n,
                }
            )
        shown = offsets.copy()
        shown.insert(0, "n", counts.reindex(shown.index).fillna(0).astype(int))
        offset_tables[k] = shown
        test_labels = labels[test_mask]
        actual = y[test_mask]
        when = when_all[test_mask]
        raw = base.loc[test_mask, knots].reset_index(drop=True)
        reorder = _reorder_rate(raw, test_labels, offsets, knots)
        calibrated = apply_offsets(
            raw,
            test_labels,
            offsets,
            spec.clip_high,
            lower=spec.clip_low,
        )
        for version, frame in (("raw", raw), ("cal", calibrated)):
            blocks = [("all", np.ones(len(frame), dtype=bool))]
            for name in pd.unique(test_labels):
                blocks.append((str(name), test_labels == name))
            for name, mask in blocks:
                part = frame.loc[mask]
                outcome = actual[mask]
                cov80, width80 = _band(outcome, part, 0.10, 0.90)
                cov90, width90 = _band(outcome, part, 0.05, 0.95)
                nested_rows.append(
                    {
                        "fold": k,
                        "version": version,
                        "group": name,
                        "n": int(mask.sum()),
                        "mean_abs_delta": _mean_abs_delta(outcome, part, levels),
                        "cov80": cov80,
                        "width80": width80,
                        "cov90": cov90,
                        "width90": width90,
                        "pinball_q50": pinball_loss(outcome, part["q_0.50"], 0.50),
                        "reorder": reorder,
                    }
                )
                for key, tau in levels.items():
                    delta_rows.append(
                        {
                            "fold": k,
                            "version": version,
                            "group": name,
                            "knot": key,
                            "delta": float(np.mean(outcome <= part[key].to_numpy()))
                            - tau,
                        }
                    )
        wis_raw = _wis_values(actual, _qmap(raw, knots))
        wis_cal = _wis_values(actual, _qmap(calibrated, knots))
        crps_raw = row_crps(actual, raw[knots].to_numpy(dtype=float), level_grid)
        crps_cal = row_crps(
            actual, calibrated[knots].to_numpy(dtype=float), level_grid
        )
        wis_ci = paired_date_bootstrap(
            wis_raw, wis_cal, when, n_boot=n_boot, seed=seed
        )
        crps_ci = paired_date_bootstrap(
            crps_raw, crps_cal, when, n_boot=n_boot, seed=seed
        )
        score_rows.append(
            {
                "fold": k,
                "n": int(test_mask.sum()),
                "reorder": reorder,
                "pinball_q50_raw": pinball_loss(actual, raw["q_0.50"], 0.50),
                "pinball_q50_cal": pinball_loss(actual, calibrated["q_0.50"], 0.50),
                "wis_raw": float(np.mean(wis_raw)),
                "wis_cal": float(np.mean(wis_cal)),
                "wis_diff": wis_ci["diff"],
                "wis_ci_low": wis_ci["ci_low"],
                "wis_ci_high": wis_ci["ci_high"],
                "crps_raw": float(np.mean(crps_raw)),
                "crps_cal": float(np.mean(crps_cal)),
                "crps_diff": crps_ci["diff"],
                "crps_ci_low": crps_ci["ci_low"],
                "crps_ci_high": crps_ci["ci_high"],
            }
        )
    return {
        "nested": pd.DataFrame(nested_rows),
        "deltas": pd.DataFrame(delta_rows),
        "scores": pd.DataFrame(score_rows),
        "offset_tables": offset_tables,
        "counts": pd.DataFrame(count_rows),
        "flags": offset_stability_flags(offset_tables),
    }


def offset_stability_flags(offset_tables: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Knots whose offset range exceeds the mean absolute offset."""
    long_offsets = []
    for k, table in offset_tables.items():
        body = table.drop(columns="n") if "n" in table.columns else table
        for grp, row in body.iterrows():
            for key, value in row.items():
                long_offsets.append(
                    {"k": k, "group": grp, "knot": key, "offset": float(value)}
                )
    if not long_offsets:
        return pd.DataFrame(columns=["group", "knot", "swing", "mean_abs", "offsets"])
    offset_long = pd.DataFrame(long_offsets)
    flag_rows = []
    for (grp, key), part in offset_long.groupby(["group", "knot"], sort=False):
        values = part["offset"].to_numpy(dtype=float)
        if values.size < 2:
            continue
        swing = float(values.max() - values.min())
        magnitude = float(np.mean(np.abs(values)))
        if swing > magnitude:
            flag_rows.append(
                {
                    "group": grp,
                    "knot": key,
                    "swing": swing,
                    "mean_abs": magnitude,
                    "offsets": ", ".join(f"{value:.3f}" for value in values),
                }
            )
    return pd.DataFrame(flag_rows)


def _group_names(spec: ConformalSpec) -> tuple[str, ...]:
    return ("cold_start", *[name for name, _low, _high in spec.tiers])


def _knot_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame.columns if str(column).startswith("q_")),
        key=lambda column: float(str(column).split("_", 1)[1]),
    )


def _gap(indicator, when, target: float, n_boot: int, seed: int) -> dict:
    covered = np.asarray(indicator, dtype=float)
    if covered.size == 0:
        raise ValueError("empty slice")
    boot = paired_date_bootstrap(
        np.full(covered.shape[0], float(target)),
        covered,
        when,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "n": int(boot["n"]),
        "empirical": float(np.mean(covered)),
        "target": float(target),
        "delta": float(boot["diff"]),
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
    }


def _slice_rows(
    name, mask, actual, when, frame, knots, levels, n_boot, seed
) -> list[dict]:
    chosen = np.asarray(mask, dtype=bool)
    outcome = actual[chosen]
    dated = when[chosen]
    grid = frame.loc[:, knots].to_numpy(dtype=float)[chosen]
    rows = []
    for column, (key, tau) in enumerate(levels.items()):
        row = _gap(outcome <= grid[:, column], dated, tau, n_boot, seed)
        row.update({"slice": name, "check": key})
        rows.append(row)
    for check, low_key, high_key, target in (
        ("coverage_80", "q_0.10", "q_0.90", 0.80),
        ("coverage_90", "q_0.05", "q_0.95", 0.90),
    ):
        if low_key not in knots or high_key not in knots:
            continue
        low = knots.index(low_key)
        high = knots.index(high_key)
        inside = (outcome >= grid[:, low]) & (outcome <= grid[:, high])
        row = _gap(inside, dated, target, n_boot, seed)
        row.update({"slice": name, "check": check})
        rows.append(row)
    return rows


def _mean_abs_delta(actual, frame: pd.DataFrame, levels: Mapping[str, float]) -> float:
    gaps = [
        abs(float(np.mean(actual <= frame[key].to_numpy(dtype=float))) - tau)
        for key, tau in levels.items()
    ]
    return float(np.mean(gaps))


def _band(actual, frame: pd.DataFrame, low: float, high: float) -> tuple[float, float]:
    low_key, high_key = f"q_{low:.2f}", f"q_{high:.2f}"
    if low_key not in frame.columns or high_key not in frame.columns:
        return float("nan"), float("nan")
    lo = frame[low_key].to_numpy(dtype=float)
    hi = frame[high_key].to_numpy(dtype=float)
    inside = (actual >= lo) & (actual <= hi)
    return float(np.mean(inside)), float(np.mean(hi - lo))


def _qmap(frame: pd.DataFrame, knots: Sequence[str]) -> dict[str, np.ndarray]:
    return {key: frame[key].to_numpy(dtype=float) for key in knots}


def _reorder_rate(frame, labels, offsets: pd.DataFrame, knots: Sequence[str]) -> float:
    grid = frame.loc[:, list(knots)].to_numpy(dtype=float)
    add = np.zeros_like(grid)
    if len(offsets):
        lookup = {str(name): row for row, name in enumerate(offsets.index)}
        matched = np.array([lookup.get(str(label), -1) for label in labels])
        known = matched >= 0
        add[known] = offsets.loc[:, list(knots)].to_numpy(dtype=float)[matched[known]]
    shifted = grid + add
    return float(np.mean(np.any(np.diff(shifted, axis=1) < 0, axis=1)))


def _interval_score(y, lower, upper, alpha: float) -> np.ndarray:
    actual = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    below = actual < lo
    above = actual > hi
    return (
        width
        + (2.0 / alpha) * (lo - actual) * below
        + (2.0 / alpha) * (actual - hi) * above
    )


def _wis_values(y, qmap: Mapping[str, np.ndarray]) -> np.ndarray:
    actual = np.asarray(y, dtype=float)
    present = [
        (alpha, qmap[low], qmap[high])
        for alpha, low, high in _WIS_INTERVALS
        if low in qmap and high in qmap
    ]
    total = 0.5 * np.abs(actual - np.asarray(qmap["q_0.50"], dtype=float))
    for alpha, lower, upper in present:
        total = total + (alpha / 2.0) * _interval_score(actual, lower, upper, alpha)
    return total / (len(present) + 0.5)
