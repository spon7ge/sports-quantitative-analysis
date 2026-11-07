"""Official logs and an upcoming slate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_events.checks import _normalize_ids
from game_events.constants import (
    _EVENT_COLUMNS,
    _EVENT_IDS,
    _EVENT_TEAM_COLUMNS,
)


def _prepare_event_logs(logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(logs, pd.DataFrame) or logs.empty:
        raise ValueError("logs must be a non-empty DataFrame")
    required = (*_EVENT_IDS, *_EVENT_COLUMNS, *_EVENT_TEAM_COLUMNS)
    missing = [column for column in required if column not in logs.columns]
    if "min" not in logs.columns and "minutes" not in logs.columns:
        missing.append("min")
    if missing:
        raise ValueError(f"logs need {missing}")
    frame = logs.copy()
    for column in ("player_id", "game_id", "team_id"):
        frame[column] = _normalize_ids(frame[column], column).to_numpy()
    frame["game_date"] = pd.to_datetime(
        frame["game_date"], errors="coerce"
    ).dt.normalize()
    if frame["game_date"].isna().any():
        raise ValueError("game_date has missing or unparseable values")
    frame["minutes"] = _log_minutes(frame)
    for column in (*_EVENT_COLUMNS, *_EVENT_TEAM_COLUMNS):
        frame[column] = _event_nonnegative(frame[column], column)
    _reject_event_bounds(frame)

    keys = ["game_id", "player_id", "team_id"]
    _reject_conflicts(
        frame,
        keys,
        (*_EVENT_COLUMNS, *_EVENT_TEAM_COLUMNS, "minutes", "game_date"),
        "player-game",
    )
    _reject_conflicts(
        frame, ["game_id", "team_id"], (*_EVENT_TEAM_COLUMNS, "game_date"), "team-game"
    )
    _reject_conflicts(frame, ["game_id"], ("game_date",), "game")
    frame = frame.drop_duplicates(keys)
    if frame.duplicated(["player_id", "game_date"]).any():
        raise ValueError("logs have two games for one player on one date")
    if frame.groupby("game_id")["team_id"].nunique().gt(2).any():
        raise ValueError("logs have more than two teams in one game")
    return frame.reset_index(drop=True)


def _reject_conflicts(
    frame: pd.DataFrame, keys: list[str], columns: tuple, label: str
) -> None:
    counts = frame.groupby(keys, dropna=False)[list(columns)].nunique(dropna=False)
    conflicted = [column for column in columns if (counts[column] > 1).any()]
    if conflicted:
        raise ValueError(f"logs have conflicting {conflicted} for one {label}")


def _parse_minutes(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    text = values.astype("string").str.strip().fillna("")
    parts = text.str.extract(r"^(\d+):([0-5]?\d(?:\.\d+)?)$")
    clock = (
        pd.to_numeric(parts[0], errors="coerce")
        + pd.to_numeric(parts[1], errors="coerce") / 60.0
    ).astype(float)
    out = numeric.where(numeric.notna(), clock)
    if (text.ne("") & out.isna()).any():
        raise ValueError(f"{name} has unparseable values")
    if (~np.isfinite(out.fillna(0.0)) | (out < 0)).any():
        raise ValueError(f"{name} must be nonnegative and finite")
    return out


def _log_minutes(frame: pd.DataFrame) -> pd.Series:
    """Official minutes. Missing minutes stay missing, not zero."""
    parsed = [
        _parse_minutes(frame[column], column)
        for column in ("min", "minutes")
        if column in frame.columns
    ]
    if len(parsed) == 2:
        first, second = parsed
        both = first.notna() & second.notna()
        if (np.abs(first[both] - second[both]) > 1.0 / 60.0 + 1e-9).any():
            raise ValueError("min and minutes disagree")
        return first.fillna(second)
    return parsed[0]


def _event_nonnegative(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    present = values.notna() & values.astype("string").str.strip().fillna("").ne("")
    if (present & numeric.isna()).any():
        raise ValueError(f"{name} must be numeric")
    if (numeric < 0).any() or np.isinf(numeric).any():
        raise ValueError(f"{name} must be nonnegative and finite")
    return numeric


def _reject_event_bounds(frame: pd.DataFrame) -> None:
    pairs = (
        ("team_fgm", "team_fga"),
        ("team_ast", "team_fgm"),
        ("team_stl", "opp_tov"),
        ("team_blk", "opp_fga"),
        ("ast", "team_ast"),
        ("oreb", "team_oreb"),
        ("stl", "team_stl"),
        ("blk", "team_blk"),
    )
    for made, attempts in pairs:
        both = frame[made].notna() & frame[attempts].notna()
        if (frame.loc[both, made] > frame.loc[both, attempts]).any():
            raise ValueError(f"{made} cannot exceed {attempts}")


def _prepare_event_slate(slate: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(slate, pd.DataFrame) or slate.empty:
        raise ValueError("slate must be a non-empty DataFrame")
    needed = ("player_id", "game_date", "team_id")
    missing = [column for column in needed if column not in slate.columns]
    if missing:
        raise ValueError(f"slate needs {missing}")
    out = slate.copy()
    for column in ("player_id", "team_id"):
        out[column] = _normalize_ids(out[column], f"slate {column}").to_numpy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    if out["game_date"].isna().any():
        raise ValueError("slate game_date has missing or unparseable values")
    if out.duplicated(["player_id", "game_date"]).any():
        raise ValueError("slate has two rows for one player on one date")
    return out.reset_index(drop=True)
