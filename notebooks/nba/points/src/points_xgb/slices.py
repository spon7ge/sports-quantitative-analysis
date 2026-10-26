"""Holdout slices, monotone rearrangement, and the cold-start fallback."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from points_xgb.config import TARGET_COL
from points_xgb.paths import ensure_repo_on_path
from points_xgb.scoring import wis_values

ensure_repo_on_path()
from models.shared.artifacts import monotonize_quantiles  # noqa: E402
from models.shared.metrics import interval_coverage  # noqa: E402


def slice_score(
    name: str,
    mask,
    y_all,
    qmap: Mapping[str, np.ndarray],
) -> dict:
    chosen = np.asarray(mask, dtype=bool)
    n = int(chosen.sum())
    if n == 0:
        return {"slice": name, "n": 0}
    actual = np.asarray(y_all, dtype=float)[chosen]
    part = {
        key: np.asarray(values, dtype=float)[chosen] for key, values in qmap.items()
    }
    error = actual - part["q_0.50"]
    return {
        "slice": name,
        "n": n,
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "p_actual_le_q50": float(np.mean(actual <= part["q_0.50"])),
        "cov_80": interval_coverage(actual, part["q_0.10"], part["q_0.90"]),
        "mean_width_80": float(np.mean(part["q_0.90"] - part["q_0.10"])),
        "wis": float(np.mean(wis_values(actual, part))),
    }


def regular_season_blocks(dates) -> dict[str, pd.Series]:
    """Calendar blocks inside one regular season.

    Cuts use the calendar year of the last date, so a 2025-26 holdout
    that ends in April 2026 uses March and April 2026.
    """
    stamps = pd.to_datetime(dates)
    year = int(stamps.max().year)
    march = pd.Timestamp(year=year, month=3, day=1)
    mid_march = pd.Timestamp(year=year, month=3, day=15)
    april = pd.Timestamp(year=year, month=4, day=1)
    return {
        "October–February": stamps < march,
        "March 1–14": (stamps >= march) & (stamps < mid_march),
        "March 15–31": (stamps >= mid_march) & (stamps < april),
        "April regular season": stamps >= april,
    }


def monotone_shift(
    preds: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, float, float]:
    """Sort each row's quantiles. The saved trees stay raw."""
    ordered = sorted(preds, key=lambda key: float(str(key).split("_", 1)[1]))
    raw = {key: np.asarray(preds[key], dtype=float) for key in ordered}
    grid = np.column_stack([raw[key] for key in ordered])
    before = float(np.mean(np.any(np.diff(grid, axis=1) < 0, axis=1)))
    mono = monotonize_quantiles(raw)
    mono_grid = np.column_stack([mono[key] for key in ordered])
    after = float(np.mean(np.any(np.diff(mono_grid, axis=1) < 0, axis=1)))
    shift = pd.DataFrame(
        [
            {
                "quantile": key,
                "mean_abs_shift": float(np.mean(np.abs(mono[key] - raw[key]))),
                "share_changed": float(np.mean(mono[key] != raw[key])),
            }
            for key in ordered
        ]
    )
    return mono, shift, before, after


def role_quantile_fallback(
    ppm_df: pd.DataFrame,
    preds_mono: Mapping[str, np.ndarray],
    starting,
    cold,
) -> dict[str, np.ndarray]:
    """Replace cold-start rows with training-pool quantiles for that role.

    A row is cold when ``pts_per_min_lag_1`` is missing. Starters and bench
    players each get the empirical quantiles of training-pool rates in that role.
    """
    ordered = list(preds_mono)
    history = {}
    for role_value, role_name in ((1, "starter"), (0, "bench")):
        rates = ppm_df.loc[ppm_df["starting"] == role_value, TARGET_COL].to_numpy(
            dtype=float
        )
        rates = rates[np.isfinite(rates)]
        history[role_name] = {
            key: float(np.quantile(rates, float(str(key).split("_", 1)[1])))
            for key in ordered
        }
    fallback = {key: np.asarray(preds_mono[key], dtype=float).copy() for key in ordered}
    starter = np.asarray(starting, dtype=int) == 1
    bench = np.asarray(starting, dtype=int) == 0
    cold_mask = np.asarray(cold, dtype=bool)
    for key in ordered:
        fallback[key][cold_mask & starter] = history["starter"][key]
        fallback[key][cold_mask & bench] = history["bench"][key]
    return fallback


def role_reading(row: pd.Series) -> str:
    gap = row["p_actual_le_q50"] - 0.50
    if abs(gap) <= 0.02:
        return "q50 is near 50%, so the mean bias is skew around a calibrated median"
    if gap > 0:
        return "q50 is too high: actuals fall under the median more often than 50%"
    return "q50 is too low: actuals fall under the median less often than 50%"
