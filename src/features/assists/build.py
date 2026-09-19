"""Build leakage-safe pregame assists features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.assists.columns import CURRENT_ASSISTS_FEATURES
from src.features.assists.player import add_player_features
from src.features.assists.team import add_team_features

_BUILDER_TEMPS = (
    "_builder_row",
    "_started_obs",
    "_ast_obs",
    "_minutes_obs",
)


def canonical_minutes(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(
            frame["minutes"], errors="coerce"
        ).astype("float64")
    if "min" in frame.columns:
        return pd.to_numeric(
            frame["min"], errors="coerce"
        ).astype("float64")
    raise ValueError(
        "canonical minutes column missing: need minutes or min"
    )


def canonical_ast(frame: pd.DataFrame) -> pd.Series:
    if "ast" in frame.columns:
        return pd.to_numeric(
            frame["ast"], errors="coerce"
        ).astype("float64")
    if "assists" in frame.columns:
        return pd.to_numeric(
            frame["assists"], errors="coerce"
        ).astype("float64")
    return pd.Series(
        np.nan, index=frame.index, dtype="float64"
    )


def add_assists_features(frame: pd.DataFrame) -> pd.DataFrame:
    original_index = frame.index
    input_columns = set(frame.columns)
    work = frame.copy()

    saved_temps = {
        name: work[name].to_numpy()
        for name in _BUILDER_TEMPS
        if name in input_columns
    }

    work["_builder_row"] = np.arange(len(work), dtype=np.int64)
    work["_minutes_obs"] = canonical_minutes(work)
    work["_ast_obs"] = canonical_ast(work)
    for name in CURRENT_ASSISTS_FEATURES:
        work[name] = np.nan
        work[name] = work[name].astype("float64")
    work = add_player_features(work)
    work = add_team_features(work)
    work = work.sort_values("_builder_row")
    row_ids = work["_builder_row"].to_numpy()
    for name, values in saved_temps.items():
        work[name] = values[row_ids]
    drop = [
        name
        for name in work.columns
        if name not in input_columns
        and name not in CURRENT_ASSISTS_FEATURES
    ]
    work = work.drop(columns=drop)
    work.index = original_index
    return work
