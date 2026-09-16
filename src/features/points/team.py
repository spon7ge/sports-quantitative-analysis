"""Team offense and opponent defense snapshots."""

from __future__ import annotations

import pandas as pd

from src.features.minutes.rolling import (
    numeric_column,
    prior_roll,
)

TEAM_KEY = ["team_id"]
TEAM_SOURCE_COLUMNS = [
    "season_year",
    "game_id",
    "game_date",
    "team_id",
    "opp_team_id",
    "team_off_rating",
    "team_def_rating",
]


def add_team_scoring_context(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    teams = _build_team_games(frame)
    teams["team_off_rating_mean_10"] = prior_roll(
        teams,
        "team_off_rating",
        TEAM_KEY,
        10,
        "mean",
    )
    teams["team_def_rating_mean_10"] = prior_roll(
        teams,
        "team_def_rating",
        TEAM_KEY,
        10,
        "mean",
    )
    return _merge_offense_and_defense(frame, teams)


def _build_team_games(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        column
        for column in TEAM_SOURCE_COLUMNS
        if column in frame.columns
    ]
    teams = frame[columns].copy()
    teams["team_off_rating"] = numeric_column(
        teams,
        "team_off_rating",
    )
    teams["team_def_rating"] = numeric_column(
        teams,
        "team_def_rating",
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


def _merge_offense_and_defense(
    frame: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    own_columns = [
        "game_id",
        "team_id",
        "team_off_rating_mean_10",
    ]
    result = frame.merge(
        teams[own_columns],
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    opponent = teams[
        [
            "game_id",
            "team_id",
            "team_def_rating_mean_10",
        ]
    ].rename(
        columns={
            "team_id": "opp_team_id",
            "team_def_rating_mean_10": (
                "opp_def_rating_mean_10"
            ),
        }
    )
    return result.merge(
        opponent,
        on=["game_id", "opp_team_id"],
        how="left",
        validate="many_to_one",
    )
