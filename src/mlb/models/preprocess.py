"""Strikeout-frame aliases, rest encoding, and usable-feature selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.mlb.schemas import STRIKEOUT_FEATURE_COLUMNS

REST_CAP_DAYS = 14.0
DEFAULT_BF_EXPOSURE = 22.0

BINARY_COUNT_FEATURES: frozenset[str] = frozenset(
    {
        "standard_rest",
        "extended_rest",
        "long_absence",
        "first_start_or_missing_history",
        "pitcher_throws_L",
        "is_home",
        "is_opener",
        "is_restricted",
        "is_il_return",
    }
)
BINARY_STRIKEOUT_FEATURES = BINARY_COUNT_FEATURES

_STRIKEOUT_ALIASES: dict[str, tuple[str, ...]] = {
    "bf_mean_5": ("bf_mean_5", "bf_per_start_5"),
    "bf_sd_5": ("bf_sd_5",),
    "early_exit_rate_5": ("early_exit_rate_5",),
    "predicted_bf_oof": ("predicted_bf_oof", "expected_bf_oof"),
}

_WORKLOAD_ALIASES: dict[str, tuple[str, ...]] = {
    "bf_mean_5": ("bf_mean_5", "bf_per_start_5"),
    "bf_sd_5": ("bf_sd_5",),
    "early_exit_rate_5": ("early_exit_rate_5",),
    "pitches_per_start_5": ("pitches_per_start_5",),
    "outs_per_start_5": ("outs_per_start_5",),
    "pitches_per_bf_5": ("pitches_per_bf_5",),
}

_BF_EXPOSURE_ALIASES: tuple[str, ...] = (
    "predicted_bf_oof",
    "expected_bf_oof",
    "bf_mean_5",
    "bf_per_start_5",
)


@dataclass
class FeatureSelection:
    retained: tuple[str, ...]
    dropped: dict[str, str]
    medians: dict[str, float]
    centers: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)


def encode_rest_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Bounded rest plus mutually exclusive absence flags."""
    out = frame.copy()
    if "rest_days" in out.columns:
        rest = pd.to_numeric(out["rest_days"], errors="coerce")
    else:
        rest = pd.Series(np.nan, index=out.index)
    missing = rest.isna()
    out["first_start_or_missing_history"] = missing.astype(float)
    out["rest_days_capped"] = np.minimum(rest, REST_CAP_DAYS)
    out["standard_rest"] = ((rest >= 4.0) & (rest <= 6.0)).fillna(False).astype(float)
    out["extended_rest"] = (
        ((rest >= 7.0) & (rest <= REST_CAP_DAYS)).fillna(False).astype(float)
    )
    out["long_absence"] = (rest > REST_CAP_DAYS).fillna(False).astype(float)
    out.loc[
        missing,
        ["standard_rest", "extended_rest", "long_absence"],
    ] = 0.0
    return out


def _first_populated(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().any():
            return values
    return None


def prepare_strikeout_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Alias rolling workload names and attach rest encodings."""
    out = frame.copy()
    for dest, sources in _STRIKEOUT_ALIASES.items():
        if dest in out.columns and pd.to_numeric(out[dest], errors="coerce").notna().any():
            continue
        populated = _first_populated(out, sources)
        if populated is not None:
            out[dest] = populated
    return encode_rest_features(out)


def prepare_workload_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Alias lagged BF inputs and attach rest encodings for nb_bf_v1."""
    out = encode_rest_features(frame)
    for dest, sources in _WORKLOAD_ALIASES.items():
        if dest in out.columns and pd.to_numeric(out[dest], errors="coerce").notna().any():
            continue
        populated = _first_populated(out, sources)
        if populated is not None:
            out[dest] = populated
    if "pitches_per_bf_5" not in out.columns or pd.to_numeric(
        out["pitches_per_bf_5"], errors="coerce"
    ).isna().all():
        pitches = _first_populated(out, ("pitches_per_start_5",))
        bf = _first_populated(out, ("bf_mean_5", "bf_per_start_5"))
        if pitches is not None and bf is not None:
            out["pitches_per_bf_5"] = pitches / bf.clip(lower=1e-6)
    return out


def bf_exposure(frame: pd.DataFrame) -> np.ndarray:
    """Pregame expected BF. Never uses realized Game N ``batters_faced``."""
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in _BF_EXPOSURE_ALIASES:
        if name not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[name], errors="coerce")
        values = values.where(values.notna(), candidate)
    filled = values.fillna(DEFAULT_BF_EXPOSURE).to_numpy(dtype=float)
    filled = np.nan_to_num(
        filled, nan=DEFAULT_BF_EXPOSURE, posinf=DEFAULT_BF_EXPOSURE, neginf=DEFAULT_BF_EXPOSURE
    )
    return np.clip(filled, 1.0, 60.0)


def bf_exposure_offset(frame: pd.DataFrame) -> np.ndarray:
    """``log(predicted_bf_oof)`` with lagged-BF fallback."""
    return np.log(bf_exposure(frame))


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _is_binary(filled: np.ndarray) -> bool:
    unique = set(np.unique(np.round(filled, 8)))
    return unique.issubset({0.0, 1.0})


def select_usable_features(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | None = None,
    *,
    min_std: float = 1e-12,
) -> FeatureSelection:
    """Drop all-missing and training-fold zero-variance columns."""
    candidates = columns if columns is not None else STRIKEOUT_FEATURE_COLUMNS
    retained: list[str] = []
    dropped: dict[str, str] = {}
    medians: dict[str, float] = {}
    for column in candidates:
        values = _numeric_column(frame, column)
        if values.notna().sum() == 0:
            dropped[column] = "all_missing"
            continue
        median = float(values.median())
        if not np.isfinite(median):
            median = 0.0
        filled = values.fillna(median).to_numpy(dtype=float)
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.std(filled, ddof=0)) < min_std:
            dropped[column] = "zero_variance"
            continue
        retained.append(column)
        medians[column] = median
    collinear_kept: list[str] = []
    current: np.ndarray | None = None
    for column in retained:
        values = _numeric_column(frame, column).fillna(medians[column])
        filled = np.nan_to_num(
            values.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(-1, 1)
        candidate = filled if current is None else np.hstack([current, filled])
        intercept = np.ones((candidate.shape[0], 1), dtype=float)
        rank = int(np.linalg.matrix_rank(np.hstack([intercept, candidate]), tol=1e-8))
        if rank == 1 + candidate.shape[1]:
            collinear_kept.append(column)
            current = candidate
        else:
            dropped[column] = "collinear"
    return FeatureSelection(tuple(collinear_kept), dropped, medians)


def compute_standardization(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    medians: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Training-fold mean/sd for continuous columns only."""
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in columns:
        values = _numeric_column(frame, column).fillna(medians.get(column, 0.0))
        filled = np.nan_to_num(
            values.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if column in BINARY_COUNT_FEATURES or _is_binary(filled):
            continue
        centers[column] = float(np.mean(filled))
        std = float(np.std(filled, ddof=0))
        scales[column] = std if std >= 1e-8 else 1.0
    return centers, scales
