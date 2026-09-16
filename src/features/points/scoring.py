"""Shifted historical scoring features."""

from __future__ import annotations

import pandas as pd

from src.features.minutes.rolling import (
    numeric_column,
    prior_ewm,
    prior_expanding,
    prior_roll,
    prior_shift,
    prior_sum,
    ratio,
)

PLAYER_KEY = ["player_id"]
SEASON_PLAYER = ["player_id", "season_year"]


def add_scoring_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["pts"] = numeric_column(result, "pts")
    if "_appeared_obs" not in result:
        minutes = numeric_column(
            result,
            "target_minutes",
            fallback="minutes",
        )
        result["_appeared_obs"] = (
            minutes.gt(0)
            .astype(float)
            .where(minutes.notna())
        )

    result["pts_lag_1"] = prior_shift(
        result,
        "pts",
        PLAYER_KEY,
    )
    for window in (3, 10, 20):
        result[f"pts_mean_{window}"] = prior_roll(
            result,
            "pts",
            PLAYER_KEY,
            window,
            "mean",
        )
    result["pts_ewm_hl_3"] = prior_ewm(
        result,
        "pts",
        PLAYER_KEY,
        halflife=3,
    )
    result["pts_std_10"] = prior_roll(
        result,
        "pts",
        PLAYER_KEY,
        10,
        "std",
        min_periods=2,
    )
    result["active_pts_mean_10"] = ratio(
        prior_sum(result, "pts", PLAYER_KEY, 10),
        prior_sum(
            result,
            "_appeared_obs",
            PLAYER_KEY,
            10,
        ),
    )
    result["season_pts_mean"] = prior_expanding(
        result,
        "pts",
        SEASON_PLAYER,
        "mean",
    )
    return result
