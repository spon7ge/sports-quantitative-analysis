"""Build leakage-safe pregame points features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.minutes import add_minutes_features

from .columns import TIER1_POINTS_FEATURES
from .interactions import add_points_trend_features
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
    result = add_team_scoring_context(result)
    result = add_points_trend_features(result)

    for column in TIER1_POINTS_FEATURES:
        if column not in result:
            result[column] = np.nan

    temporary = [
        column
        for column in result.columns
        if column.startswith("_")
    ]
    return result.drop(columns=temporary)
