"""Scoring-rate and shot-profile features from rolling sums."""

from __future__ import annotations

import pandas as pd

from src.features.minutes.rolling import (
    numeric_column,
    prior_expanding,
    prior_shift,
    prior_sum,
    ratio,
)

PLAYER_KEY = ["player_id"]
SEASON_PLAYER = ["player_id", "season_year"]


def add_rate_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    for column in ("fga", "fg3_a", "fta", "team_fga"):
        result[column] = numeric_column(result, column)

    minutes_5 = prior_sum(
        result,
        "target_minutes",
        PLAYER_KEY,
        5,
    )
    minutes_10 = prior_sum(
        result,
        "target_minutes",
        PLAYER_KEY,
        10,
    )
    minutes_20 = prior_sum(
        result,
        "target_minutes",
        PLAYER_KEY,
        20,
    )
    fga_10 = prior_sum(result, "fga", PLAYER_KEY, 10)
    fta_20 = prior_sum(result, "fta", PLAYER_KEY, 20)
    fga_20 = prior_sum(result, "fga", PLAYER_KEY, 20)

    result["pts_per_min_10"] = ratio(
        prior_sum(result, "pts", PLAYER_KEY, 10),
        minutes_10,
    )
    result["pts_per_min_20"] = ratio(
        prior_sum(result, "pts", PLAYER_KEY, 20),
        minutes_20,
    )
    season_pts = prior_expanding(
        result,
        "pts",
        SEASON_PLAYER,
        "sum",
    )
    season_minutes = prior_expanding(
        result,
        "target_minutes",
        SEASON_PLAYER,
        "sum",
    )
    result["season_pts_per_min"] = ratio(
        season_pts,
        season_minutes,
    )
    result = _add_stint_rate(result)

    if "fga_per_min_10" not in result:
        result["fga_per_min_10"] = ratio(
            fga_10,
            minutes_10,
        )
    result["fg3a_per_min_10"] = ratio(
        prior_sum(result, "fg3_a", PLAYER_KEY, 10),
        minutes_10,
    )
    result["fta_per_min_10"] = ratio(
        prior_sum(result, "fta", PLAYER_KEY, 10),
        minutes_10,
    )
    result["three_attempt_rate_10"] = ratio(
        prior_sum(result, "fg3_a", PLAYER_KEY, 10),
        fga_10,
    )
    result["free_throw_rate_10"] = ratio(
        prior_sum(result, "fta", PLAYER_KEY, 10),
        fga_10,
    )
    result["ts_agg_20"] = ratio(
        prior_sum(result, "pts", PLAYER_KEY, 20),
        2 * (fga_20 + 0.44 * fta_20),
    )
    result["player_fga_share_10"] = ratio(
        fga_10,
        prior_sum(result, "team_fga", PLAYER_KEY, 10),
    )

    fga_per_min_5 = ratio(
        prior_sum(result, "fga", PLAYER_KEY, 5),
        minutes_5,
    )
    season_fga_per_min = ratio(
        prior_expanding(
            result,
            "fga",
            SEASON_PLAYER,
            "sum",
        ),
        season_minutes,
    )
    result["fga_per_min_5_minus_season"] = (
        fga_per_min_5 - season_fga_per_min
    )
    return result


def _add_stint_rate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    previous_team = prior_shift(
        result,
        "team_id",
        PLAYER_KEY,
    )
    same_team = result["team_id"].eq(previous_team)
    result["_new_stint"] = (
        ~same_team.fillna(False)
    ).astype(int)
    result["_stint_id"] = result.groupby(
        PLAYER_KEY,
        sort=False,
    )["_new_stint"].cumsum()
    stint_key = ["player_id", "_stint_id"]
    result["current_team_pts_per_min"] = ratio(
        prior_expanding(
            result,
            "pts",
            stint_key,
            "sum",
        ),
        prior_expanding(
            result,
            "target_minutes",
            stint_key,
            "sum",
        ),
    )
    return result
