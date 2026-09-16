"""Pregame environment, market, trend, and interaction features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import SEASON_TYPE_CODES
from .rolling import numeric_column


def add_environment_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["is_home"] = _is_home(result)
    result["season_type_cat"] = _season_type(result)
    result["game_total"] = numeric_column(
        result,
        "game_total",
    )
    result["team_spread_canonical"] = numeric_column(
        result,
        "player_team_spread",
    )
    result["abs_spread"] = result[
        "team_spread_canonical"
    ].abs()
    result["implied_team_total"] = (
        result["game_total"] / 2
        - result["team_spread_canonical"] / 2
    )
    result["market_missing"] = (
        result[["game_total", "team_spread_canonical"]]
        .isna()
        .any(axis=1)
        .astype(float)
    )
    result["min_mean_3_minus_10"] = (
        result["min_mean_3"] - result["min_mean_10"]
    )
    result["min_mean_10_minus_season"] = (
        result["min_mean_10"] - result["season_min_mean"]
    )
    result["start_rate_x_abs_spread"] = (
        result["start_rate_10"] * result["abs_spread"]
    )
    result["min10_x_expected_possessions"] = (
        result["min_mean_10"]
        * result["expected_possessions"]
    )
    return result


def _is_home(frame: pd.DataFrame) -> pd.Series:
    if "matchup" not in frame:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype="float64",
        )
    return (
        frame["matchup"]
        .astype("string")
        .str.contains(r"\bvs\.", regex=True)
        .astype(float)
    )


def _season_type(frame: pd.DataFrame) -> pd.Series:
    if "season_type" not in frame:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype="float64",
        )
    return (
        frame["season_type"]
        .astype("string")
        .map(SEASON_TYPE_CODES)
        .astype("float64")
    )
