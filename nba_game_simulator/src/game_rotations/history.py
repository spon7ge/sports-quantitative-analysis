"""Training labels and live role rows."""
from __future__ import annotations

import pandas as pd
from role_context import (
    _minutes,
    _reject_minute_conflicts,
    build_role_history,
    role_for_slate,
)

from game_rotations.checks import _frame, _normalize_keys

def training_minutes(logs: pd.DataFrame) -> pd.DataFrame:
    """Prior-only role features joined to official positive-minute labels."""
    history = build_role_history(logs).copy().drop(columns="minutes", errors="ignore")
    keys = ["player_id", "game_id", "team_id"]
    _normalize_keys(history, keys)
    if history.duplicated(keys).any():
        raise ValueError("role history contains repeated player-game rows")
    played = _played_minutes(logs)
    merged = history.merge(played, on=keys, how="left", validate="one_to_one")
    return merged.reset_index(drop=True)


def live_minutes(logs: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Latest completed appearance stays in the upcoming player's window."""
    return role_for_slate(logs, slate)

def _played_minutes(logs):
    _frame(logs, "logs")
    frame = logs.copy()
    keys = ["player_id", "game_id", "team_id"]

    _normalize_keys(frame, keys)
    frame["minutes"] = _minutes(frame)
    _reject_minute_conflicts(frame)

    conflicts = frame.groupby(keys)["minutes"].nunique(dropna=True)

    if conflicts.gt(1).any():
        raise ValueError("official logs contain conflicting minute labels")

    return frame.groupby(
        keys,
        as_index=False,
        sort=False,
    )["minutes"].first()
