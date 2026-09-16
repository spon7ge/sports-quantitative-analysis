"""Team-game histories and opponent pregame snapshots."""

from __future__ import annotations

import pandas as pd

from .rolling import (
    numeric_column,
    prior_expanding,
    prior_roll,
)

TEAM_KEY = ["team_id"]
TEAM_SEASON = ["team_id", "season_year"]
TEAM_SOURCE_COLUMNS = [
    "season_year",
    "game_id",
    "game_date",
    "team_id",
    "opp_team_id",
    "team_pace",
    "team_net_rating",
]


def build_team_games(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        column
        for column in TEAM_SOURCE_COLUMNS
        if column in frame.columns
    ]
    teams = frame[columns].copy()
    teams["team_pace"] = numeric_column(
        teams,
        "team_pace",
    )
    teams["team_net_rating"] = numeric_column(
        teams,
        "team_net_rating",
    )
    return (
        teams.drop_duplicates(
            ["team_id", "game_id"]
        )
        .sort_values(
            ["team_id", "game_date", "game_id"]
        )
        .reset_index(drop=True)
    )


def add_team_history_features(
    teams: pd.DataFrame,
) -> pd.DataFrame:
    result = teams.copy()
    result["team_pace_mean_10"] = prior_roll(
        result,
        "team_pace",
        TEAM_KEY,
        10,
        "mean",
    )
    result["team_net_rating_mean_10"] = prior_roll(
        result,
        "team_net_rating",
        TEAM_KEY,
        10,
        "mean",
    )
    result["team_net_rating_season"] = prior_expanding(
        result,
        "team_net_rating",
        TEAM_SEASON,
        "mean",
    )
    return result


def merge_team_and_opponent(
    frame: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_columns = [
        "game_id",
        "team_id",
        "team_pace_mean_10",
        "team_net_rating_mean_10",
        "team_net_rating_season",
        "team_days_since_prev_game",
        "is_back_to_back",
        "team_games_prev_3d",
        "team_games_prev_5d",
    ]
    available = [
        column
        for column in snapshot_columns
        if column in teams.columns
    ]
    result = frame.merge(
        teams[available],
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )

    opp_rename = {
        "team_id": "opp_team_id",
        "team_pace_mean_10": "opp_pace_mean_10",
        "team_net_rating_mean_10": "opp_net_rating_mean_10",
    }
    opp_columns = [
        column
        for column in (
            "game_id",
            "team_id",
            "team_pace_mean_10",
            "team_net_rating_mean_10",
        )
        if column in teams.columns
    ]
    opponent = teams[opp_columns].rename(
        columns=opp_rename
    )
    result = result.merge(
        opponent,
        on=["game_id", "opp_team_id"],
        how="left",
        validate="many_to_one",
    )
    result["expected_possessions"] = (
        result["team_pace_mean_10"]
        + result["opp_pace_mean_10"]
    ) / 2
    return result
