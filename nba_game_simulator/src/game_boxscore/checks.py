"""Player identifiers shared by the pipeline and the assembly checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _normalize_id(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("IDs cannot be booleans")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        raise ValueError(f"ID {value!r} is not an integer")
    text = str(value).strip()
    if not text:
        raise ValueError("IDs cannot be blank")
    return text


def _player_id_tuple(players: Any, label: str) -> tuple:
    if isinstance(players, pd.DataFrame):
        if "player_id" not in players.columns:
            raise ValueError(f"{label} need player_id")
        values = players["player_id"].tolist()
    else:
        values = list(players)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(
        value is None or (isinstance(value, float) and np.isnan(value))
        for value in values
    ):
        raise ValueError(f"{label} have missing player IDs")
    ids = tuple(_normalize_id(value) for value in values)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} list the same player twice")
    return ids


def _reject_shared_players(player_ids: Mapping[str, tuple]) -> None:
    shared = set(player_ids["home"]) & set(player_ids["away"])
    if shared:
        raise ValueError(f"players appear on both teams: {sorted(shared)}")
