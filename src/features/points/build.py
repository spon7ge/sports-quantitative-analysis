"""Build leakage-safe pregame points features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.minutes import add_minutes_features

from .columns import TIER1_POINTS_FEATURES
from .early import add_early_season_points
from .interactions import add_points_trend_features
from .position import add_position_encoding
from .ranks import add_role_ranks
from .rates import add_rate_features
from .scoring import add_scoring_features
from .team import add_team_scoring_context


def add_points_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add shifted pregame features for a stacked points model.

    Historical minutes, scoring, usage, and team stats are
    shift-then-rolled. Current-game box scores, minutes,
    and opponent results are never used as features.
    ``predicted_minutes_oof`` is left missing until
    chronological out-of-fold minute predictions are joined.
    """
    result = add_minutes_features(frame)
    result = add_scoring_features(result)
    result = add_rate_features(result)
    result = add_role_ranks(result)
    result = add_team_scoring_context(result)
    result = add_points_trend_features(result)
    result = add_early_season_points(result)
    result["min_sd_10"] = result["min_std_10"]
    result["minutes_ewm_5"] = result["min_ewm_hl_5"]
    result["minutes_ewm_10"] = result["min_ewm_hl_10"]
    result["minutes_ewm_20"] = result["min_ewm_hl_20"]
    result["minutes_rolling_5"] = result["min_mean_5"]
    result["minutes_rolling_10"] = result["min_mean_10"]
    result["minutes_rolling_20"] = result["min_mean_20"]
    result["minutes_rolling_3"] = result["min_mean_3"]
    result["opp_pace_10"] = result["opp_pace_mean_10"]
    result["team_pace_10"] = result["team_pace_mean_10"]
    result["expected_pace"] = (
        result["team_pace_10"] + result["opp_pace_10"]
    ) / 2
    result["expected_pace_median"] = (
        result["team_pace_median_10"]
        + result["opp_pace_median_10"]
    ) / 2
    result = add_position_encoding(result)

    for column in TIER1_POINTS_FEATURES:
        if column not in result:
            result[column] = np.nan

    temporary = [
        column
        for column in result.columns
        if column.startswith("_")
    ]
    return result.drop(columns=temporary)
