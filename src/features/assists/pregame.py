"""Pregame assists rows from history strictly before an as-of date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.assists.build import add_assists_features
from src.features.assists.columns import CURRENT_ASSISTS_FEATURES

OUTCOME_COLUMNS = (
    "ast",
    "assists",
    "min",
    "minutes",
    "min_sec",
    "start_position",
    "team_ast",
    "team_fgm",
    "team_pace",
    "opp_ast",
    "opp_pace",
    "pass",
    "tchs",
    "sast",
    "ftast",
)


def appearances_before(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.loc[dates < cutoff].copy()


def blank_outcomes(row: pd.Series) -> pd.Series:
    result = row.copy()
    for column in OUTCOME_COLUMNS:
        if column not in result.index:
            continue
        if column == "start_position":
            result[column] = ""
        else:
            result[column] = np.nan
    return result


def build_pregame_assists_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    prior = appearances_before(history, as_of)
    if candidates.empty:
        result = candidates.copy()
        for column in CURRENT_ASSISTS_FEATURES:
            result[column] = pd.Series(
                index=result.index, dtype="float64"
            )
        return result

    dated = candidates.copy()
    dated["game_date"] = pd.to_datetime(
        dated["game_date"], errors="coerce"
    )
    blanked = dated.apply(blank_outcomes, axis=1)
    parts: list[pd.DataFrame] = []
    for _, group in blanked.groupby("game_date", sort=True):
        panel = pd.concat([prior, group], ignore_index=True)
        featured = add_assists_features(panel)
        keys = group[["game_id", "player_id"]]
        matched = featured.merge(
            keys,
            on=["game_id", "player_id"],
            how="inner",
        )
        parts.append(matched)
    stacked = pd.concat(parts, ignore_index=True)
    keep = [
        column
        for column in stacked.columns
        if column in set(candidates.columns) | set(CURRENT_ASSISTS_FEATURES)
    ]
    restored = candidates[["game_id", "player_id"]].merge(
        stacked[keep],
        on=["game_id", "player_id"],
        how="left",
    )
    restored.index = candidates.index
    return restored
