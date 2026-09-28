"""Pregame assists-per-minute features.

Player windows cross seasons. Season-to-date rate, days since
the previous game, and season game counts reset by season.
Team and opponent windows are earlier team-games only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.minutes.rolling import (
    numeric_column,
    prior_expanding,
    prior_roll,
    prior_sum,
    ratio,
)

PLAYER_KEY = ["player_id"]
SEASON_PLAYER = ["player_id", "season_year"]
TEAM_KEY = ["team_id"]

_PLAYER_OUTPUTS = (
    "is_home",
    "days_since_previous_game",
    "season_games_prior",
    "player_ast_rate_last5",
    "player_ast_rate_last10",
    "player_ast_rate_season_to_date",
    "player_ast_pct_last10",
    "player_tchs_per_min_last10",
    "player_pass_per_min_last10",
    "player_pass_per_touch_last10",
    "player_sast_per_min_last10",
    "player_usg_pct_last10",
    "player_min_last5",
)

_OWN_TEAM_OUTPUTS = (
    "team_pace_last10",
    "team_ast_pct_last10",
    "team_efg_pct_last10",
)

_OPP_OUTPUTS = (
    "opp_pace_last10",
    "opp_def_rating_last10",
    "opp_ast_allowed_per_100_last10",
)


def add_ast_rate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the rolling columns used by the assists-per-minute model."""
    original_index = frame.index
    result = frame.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"],
        errors="coerce",
    )
    result["_ast_rate_row"] = np.arange(len(result), dtype=np.int64)
    result = result.sort_values(
        ["player_id", "game_date", "game_id", "_ast_rate_row"],
        kind="mergesort",
    )
    result = _add_player_rates(result)
    result = _add_team_rates(result)
    result = result.sort_values("_ast_rate_row", kind="mergesort")
    result = result.drop(columns="_ast_rate_row")
    result.index = original_index
    return result


def _add_player_rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_minutes_obs"] = numeric_column(
        result,
        "minutes",
        fallback="min",
    )
    result["_ast_obs"] = numeric_column(
        result,
        "ast",
        fallback="assists",
    )
    for source, temp in (
        ("ast_pct", "_ast_pct_obs"),
        ("tchs", "_tchs_obs"),
        ("pass", "_pass_obs"),
        ("sast", "_sast_obs"),
        ("usg_pct", "_usg_obs"),
    ):
        result[temp] = numeric_column(result, source)

    ast_minutes_5 = _minutes_where_finite(
        result, "_ast_obs", 5
    )
    ast_minutes_10 = _minutes_where_finite(
        result, "_ast_obs", 10
    )
    result["player_ast_rate_last5"] = ratio(
        prior_sum(result, "_ast_obs", PLAYER_KEY, 5),
        ast_minutes_5,
    )
    result["player_ast_rate_last10"] = ratio(
        prior_sum(result, "_ast_obs", PLAYER_KEY, 10),
        ast_minutes_10,
    )
    result["player_ast_rate_season_to_date"] = ratio(
        prior_expanding(
            result, "_ast_obs", SEASON_PLAYER, "sum"
        ),
        _minutes_where_finite_expanding(result, "_ast_obs"),
    )
    result["player_ast_pct_last10"] = _weighted_mean(
        result, "_ast_pct_obs", 10
    )
    result["player_tchs_per_min_last10"] = ratio(
        prior_sum(result, "_tchs_obs", PLAYER_KEY, 10),
        _minutes_where_finite(result, "_tchs_obs", 10),
    )
    result["player_pass_per_min_last10"] = ratio(
        prior_sum(result, "_pass_obs", PLAYER_KEY, 10),
        _minutes_where_finite(result, "_pass_obs", 10),
    )
    result["player_pass_per_touch_last10"] = ratio(
        prior_sum(result, "_pass_obs", PLAYER_KEY, 10),
        prior_sum(result, "_tchs_obs", PLAYER_KEY, 10),
    )
    result["player_sast_per_min_last10"] = ratio(
        prior_sum(result, "_sast_obs", PLAYER_KEY, 10),
        _minutes_where_finite(result, "_sast_obs", 10),
    )
    result["player_usg_pct_last10"] = _weighted_mean(
        result, "_usg_obs", 10
    )
    result["player_min_last5"] = prior_roll(
        result,
        "_minutes_obs",
        PLAYER_KEY,
        5,
        "mean",
    )
    result["days_since_previous_game"] = (
        result.groupby(SEASON_PLAYER, sort=False)["game_date"]
        .diff()
        .dt.days
        .astype("float64")
    )
    result["season_games_prior"] = result.groupby(
        SEASON_PLAYER,
        sort=False,
    ).cumcount()
    result["is_home"] = _is_home(result)
    result["player_min_last5"] = result["player_min_last5"].astype(
        "float64"
    )
    temps = [
        "_minutes_obs",
        "_ast_obs",
        "_ast_pct_obs",
        "_tchs_obs",
        "_pass_obs",
        "_sast_obs",
        "_usg_obs",
    ]
    return result.drop(columns=temps)


def _minutes_where_finite(
    frame: pd.DataFrame,
    stat: str,
    window: int,
) -> pd.Series:
    frame[f"{stat}_min"] = frame["_minutes_obs"].where(
        frame[stat].notna()
    )
    total = prior_sum(frame, f"{stat}_min", PLAYER_KEY, window)
    frame.drop(columns=f"{stat}_min", inplace=True)
    return total


def _minutes_where_finite_expanding(
    frame: pd.DataFrame,
    stat: str,
) -> pd.Series:
    frame[f"{stat}_min"] = frame["_minutes_obs"].where(
        frame[stat].notna()
    )
    total = prior_expanding(
        frame,
        f"{stat}_min",
        SEASON_PLAYER,
        "sum",
    )
    frame.drop(columns=f"{stat}_min", inplace=True)
    return total


def _weighted_mean(
    frame: pd.DataFrame,
    stat: str,
    window: int,
) -> pd.Series:
    frame[f"{stat}_x_min"] = frame[stat] * frame["_minutes_obs"]
    weighted = ratio(
        prior_sum(frame, f"{stat}_x_min", PLAYER_KEY, window),
        _minutes_where_finite(frame, stat, window),
    )
    frame.drop(columns=f"{stat}_x_min", inplace=True)
    return weighted


def _is_home(frame: pd.DataFrame) -> pd.Series:
    if "matchup" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return (
        frame["matchup"]
        .astype("string")
        .str.contains(r"\bvs\.", regex=True)
        .astype(float)
    )


def _add_team_rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(
        columns=[
            column
            for column in (*_OWN_TEAM_OUTPUTS, *_OPP_OUTPUTS)
            if column in frame.columns
        ]
    )
    teams = _team_games(result)
    teams["team_pace_last10"] = prior_roll(
        teams, "team_pace", TEAM_KEY, 10, "mean"
    )
    teams["team_ast_pct_last10"] = prior_roll(
        teams, "team_ast_pct", TEAM_KEY, 10, "mean"
    )
    teams["team_efg_pct_last10"] = prior_roll(
        teams, "team_efg_pct", TEAM_KEY, 10, "mean"
    )
    teams["team_def_rating_last10"] = prior_roll(
        teams, "team_def_rating", TEAM_KEY, 10, "mean"
    )
    teams["_ast_allowed_per_100"] = ratio(
        teams["opp_ast"],
        teams["team_poss"],
    ) * 100.0
    teams["team_ast_allowed_per_100_last10"] = prior_roll(
        teams,
        "_ast_allowed_per_100",
        TEAM_KEY,
        10,
        "mean",
    )

    own = teams[
        ["game_id", "team_id", *_OWN_TEAM_OUTPUTS]
    ]
    result = result.merge(
        own,
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    opponent = teams[
        [
            "game_id",
            "team_id",
            "team_pace_last10",
            "team_def_rating_last10",
            "team_ast_allowed_per_100_last10",
        ]
    ].rename(
        columns={
            "team_id": "opp_team_id",
            "team_pace_last10": "opp_pace_last10",
            "team_def_rating_last10": "opp_def_rating_last10",
            "team_ast_allowed_per_100_last10": (
                "opp_ast_allowed_per_100_last10"
            ),
        }
    )
    return result.merge(
        opponent,
        on=["game_id", "opp_team_id"],
        how="left",
        validate="many_to_one",
    )


def _team_games(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player_id",
        "team_id",
        "game_id",
        "game_date",
        "opp_team_id",
        "team_pace",
        "team_ast_pct",
        "team_efg_pct",
        "team_def_rating",
        "opp_ast",
        "team_poss",
    ]
    present = [column for column in columns if column in frame.columns]
    teams = frame[present].copy()
    for column in (
        "team_pace",
        "team_ast_pct",
        "team_efg_pct",
        "team_def_rating",
        "opp_ast",
        "team_poss",
    ):
        teams[column] = numeric_column(teams, column)
    missing_poss = teams["team_poss"].isna()
    teams.loc[missing_poss, "team_poss"] = teams.loc[
        missing_poss, "team_pace"
    ]
    if "player_id" not in teams.columns:
        teams["player_id"] = np.nan
    return (
        teams.sort_values(
            ["team_id", "game_id", "player_id"],
            kind="mergesort",
        )
        .drop_duplicates(["team_id", "game_id"])
        .sort_values(
            ["team_id", "game_date", "game_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
