"""Build leakage-safe pregame minutes features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import TIER1_MINUTES_FEATURES
from .environment import add_environment_features
from .player import (
    add_observation_columns,
    add_player_role_features,
    add_season_and_stint_features,
    add_usage_features,
)
from .schedule import (
    add_player_schedule_features,
    add_team_schedule_features,
)
from .target import reconcile_target_minutes
from .team import (
    add_team_history_features,
    build_team_games,
    merge_team_and_opponent,
)


def add_minutes_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add shifted pregame features for an XGBoost minutes model.

    Trailing player windows cross seasons. Season-to-date
    and schedule-density features reset by season. Current
    box, team, opponent, and starter fields are never used
    as features. The model remains appearance-conditional
    unless a documented DNP universe is supplied later.
    """
    result = frame.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"],
        errors="coerce",
    )
    result["target_minutes"] = reconcile_target_minutes(
        result
    )
    result["minutes"] = result["target_minutes"]
    result = result.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    result = add_observation_columns(result)
    result = add_player_role_features(result)
    result = add_season_and_stint_features(result)
    result = add_usage_features(result)

    teams = build_team_games(result)
    teams = add_team_history_features(teams)
    teams = add_team_schedule_features(teams)
    result = merge_team_and_opponent(result, teams)
    result = add_player_schedule_features(result)
    result = add_environment_features(result)

    for column in TIER1_MINUTES_FEATURES:
        if column not in result:
            result[column] = np.nan

    temporary = [
        column
        for column in result.columns
        if column.startswith("_")
    ]
    return result.drop(columns=temporary)
