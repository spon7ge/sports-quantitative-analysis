"""Joint out-of-sample minutes and rate quantile knots.

Walk-forward rows come from fold models. Holdout rows come from the
frozen minutes bundle and the saved rate bundle. The parquet is written
once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from models.shared.minutes_sampler import QUANTILE_LEVELS

HOLDOUT_WINDOW_ID = -1
RATE_CAP = 6.0

_IDENTITY = (
    "game_id",
    "player_id",
    "game_date",
    "window_id",
    "is_holdout",
    "minutes",
    "pts",
)
_MERGE_KEYS = ("game_id", "player_id", "window_id")
_MUST_MATCH = ("game_date", "is_holdout", "minutes", "pts")


def capped_rate(
    points,
    minutes,
    cap: float = RATE_CAP,
) -> np.ndarray:
    """Training and scoring label ``min(points / minutes, cap)``."""
    points = np.asarray(points, dtype=float)
    minutes = np.asarray(minutes, dtype=float)
    if points.shape != minutes.shape:
        raise ValueError("points and minutes must have the same shape")
    if np.any(~np.isfinite(minutes)) or np.any(minutes <= 0):
        raise ValueError("rate rows require minutes > 0")
    return np.minimum(points / minutes, cap)


def quantile_columns(levels: Sequence[float] = QUANTILE_LEVELS) -> list[str]:
    return [f"q_{level:.2f}" for level in levels]


def oos_columns(levels: Sequence[float] = QUANTILE_LEVELS) -> list[str]:
    minutes = [f"minutes_q_{level:.2f}" for level in levels]
    rate = [f"rate_q_{level:.2f}" for level in levels]
    return [
        "game_id",
        "player_id",
        "game_date",
        "window_id",
        "is_holdout",
        *minutes,
        *rate,
        "minutes",
        "pts",
    ]


def oof_prediction_frame(
    oof: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> pd.DataFrame:
    """Validation-fold predictions plus realized minutes and points.

    ``oof['minutes']`` is the walk-forward training target, which is the
    capped rate for the rate model. Realized minutes are read from
    ``identity``.
    """
    if oof.index.duplicated().any():
        raise ValueError("walk-forward frame has duplicate rows")
    columns = quantile_columns(levels)
    missing = [column for column in columns if column not in oof.columns]
    if missing:
        raise KeyError(f"walk-forward frame missing {missing}")
    needed = ["game_id", "player_id", "game_date", "minutes", "pts"]
    aligned = identity.loc[oof.index, needed]
    if len(aligned) != len(oof):
        raise ValueError("identity index does not cover walk-forward rows")
    out = aligned.copy()
    out["window_id"] = oof["fold_id"].to_numpy()
    out["is_holdout"] = False
    for column in columns:
        out[column] = oof[column].to_numpy()
    return out.reset_index(drop=True)


def holdout_prediction_frame(
    holdout_df: pd.DataFrame,
    preds: Mapping[str, np.ndarray] | pd.DataFrame,
    *,
    levels: Sequence[float] = QUANTILE_LEVELS,
    window_id: int = HOLDOUT_WINDOW_ID,
) -> pd.DataFrame:
    """Holdout-season predictions. The frame must already be holdout rows."""
    columns = quantile_columns(levels)
    out = holdout_df.loc[
        :, ["game_id", "player_id", "game_date", "minutes", "pts"]
    ].copy()
    out["window_id"] = window_id
    out["is_holdout"] = True
    for column in columns:
        if isinstance(preds, pd.DataFrame):
            values = preds[column].to_numpy(dtype=float)
        else:
            values = np.asarray(preds[column], dtype=float)
        if len(values) != len(out):
            raise ValueError("prediction rows do not match holdout rows")
        out[column] = values
    return out.reset_index(drop=True)


def assemble_oos_predictions(
    preholdout_minutes: pd.DataFrame,
    preholdout_rate: pd.DataFrame,
    holdout_minutes: pd.DataFrame,
    holdout_rate: pd.DataFrame,
    *,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> pd.DataFrame:
    """One row per player-game with sorted minutes knots and rate knots.

    Pre-holdout rows are walk-forward validation folds. The same
    ``(game_id, player_id)`` must not also appear in the holdout block:
    a final bundle has already seen those rows.
    """
    _require_fold_rows(preholdout_minutes, "minutes")
    _require_fold_rows(preholdout_rate, "rate")
    _require_holdout_rows(holdout_minutes, "minutes")
    _require_holdout_rows(holdout_rate, "rate")
    overlap = _player_games(preholdout_minutes) & _player_games(holdout_minutes)
    overlap |= _player_games(preholdout_rate) & _player_games(holdout_rate)
    if overlap:
        raise ValueError(
            "pre-holdout and holdout share player-games; "
            "final-bundle predictions on pre-holdout rows are in-sample"
        )

    pre = _join_targets(preholdout_minutes, preholdout_rate, levels)
    held = _join_targets(holdout_minutes, holdout_rate, levels)
    table = pd.concat([pre, held], ignore_index=True)
    table = table.sort_values(
        ["game_date", "game_id", "player_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    _validate_oos_frame(table, levels)
    return table


def write_oos_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the joint OOS table. An existing file is left untouched."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; out-of-sample predictions are never revised"
        )
    _validate_oos_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _player_games(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["game_id"].astype(str),
            frame["player_id"].astype(str),
            strict=True,
        )
    )


def _require_fold_rows(frame: pd.DataFrame, name: str) -> None:
    if frame["is_holdout"].astype(bool).any():
        raise ValueError(f"{name} pre-holdout rows cannot be marked holdout")
    if (frame["window_id"].astype(int) <= 0).any():
        raise ValueError(
            f"{name} pre-holdout window_id must be a walk-forward fold"
        )


def _require_holdout_rows(frame: pd.DataFrame, name: str) -> None:
    if not frame["is_holdout"].astype(bool).all():
        raise ValueError(f"{name} holdout rows must be marked holdout")
    if not (frame["window_id"].astype(int) == HOLDOUT_WINDOW_ID).all():
        raise ValueError(f"{name} holdout window_id must be the holdout sentinel")


def _join_targets(
    minutes: pd.DataFrame,
    rate: pd.DataFrame,
    levels: Sequence[float],
) -> pd.DataFrame:
    qcols = quantile_columns(levels)
    left = minutes.reset_index(drop=True)
    right = rate.reset_index(drop=True)
    keys = list(_MERGE_KEYS)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("duplicate player-game in a prediction frame")
    merged = left.merge(
        right,
        on=list(_MERGE_KEYS),
        how="outer",
        suffixes=("", "_rate"),
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError(
            "minutes and rate predictions do not cover the same player-games"
        )
    _assert_same_realized(merged)
    minute_names = [f"minutes_q_{level:.2f}" for level in levels]
    rate_names = [f"rate_q_{level:.2f}" for level in levels]
    minute_grid = _sorted_grid(merged, qcols)
    rate_grid = _sorted_grid(merged, [f"{column}_rate" for column in qcols])
    out = merged.loc[:, list(_IDENTITY)].copy()
    out[minute_names] = minute_grid
    out[rate_names] = rate_grid
    out["window_id"] = out["window_id"].astype(int)
    out["is_holdout"] = out["is_holdout"].astype(bool)
    return out.loc[:, oos_columns(levels)]


def _assert_same_realized(merged: pd.DataFrame) -> None:
    for column in _MUST_MATCH:
        left = merged[column]
        right = merged[f"{column}_rate"]
        if column in ("minutes", "pts"):
            same = np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=False,
            )
        elif column == "game_date":
            same = pd.to_datetime(left).equals(pd.to_datetime(right))
        else:
            same = left.astype(bool).equals(right.astype(bool))
        if not same:
            raise ValueError(f"{column} disagrees between minutes and rate rows")


def _sorted_grid(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    grid = np.array(frame.loc[:, list(columns)].to_numpy(dtype=float), copy=True)
    if not np.isfinite(grid).all():
        raise ValueError("quantile knots must be finite")
    grid.sort(axis=1)
    return grid


def _validate_oos_frame(
    frame: pd.DataFrame,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> None:
    expected = oos_columns(levels)
    if list(frame.columns) != expected:
        raise ValueError("out-of-sample columns are not the saved contract")
    if frame.duplicated(["game_id", "player_id"]).any():
        raise ValueError("duplicate game_id, player_id in out-of-sample table")
    minutes = frame["minutes"].to_numpy(dtype=float)
    if np.any(~np.isfinite(minutes)) or np.any(minutes <= 0):
        raise ValueError("realized minutes must be positive")
    holdout = frame["is_holdout"].astype(bool).to_numpy()
    window = frame["window_id"].astype(int).to_numpy()
    if np.any(holdout & (window != HOLDOUT_WINDOW_ID)):
        raise ValueError("holdout rows must use the holdout window id")
    if np.any(~holdout & (window <= 0)):
        raise ValueError("pre-holdout window_id must be a walk-forward fold")
    for prefix in ("minutes", "rate"):
        cols = [f"{prefix}_q_{level:.2f}" for level in levels]
        grid = frame.loc[:, cols].to_numpy(dtype=float)
        if not np.isfinite(grid).all():
            raise ValueError(f"{prefix} knots must be finite")
        if np.any(np.diff(grid, axis=1) < 0):
            raise ValueError(f"{prefix} knots must be sorted along tau")
