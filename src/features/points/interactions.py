"""Points trends and stacked minutes interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_points_trend_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["pts_mean_3_minus_10"] = (
        result["pts_mean_3"] - result["pts_mean_10"]
    )
    result["ppm_10_minus_season"] = (
        result["pts_per_min_10"]
        - result["season_pts_per_min"]
    )
    result["usage_x_expected_possessions"] = (
        result["usg_wmean_10"]
        * result["expected_possessions"]
    )
    if "predicted_minutes_oof" not in result:
        result["predicted_minutes_oof"] = np.nan
    return add_stacked_interactions(result)


def add_stacked_interactions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["expected_points_rate"] = (
        result["predicted_minutes_oof"]
        * result["pts_per_min_10"]
    )
    result["expected_attempt_volume"] = (
        result["predicted_minutes_oof"]
        * result["fga_per_min_10"]
    )
    return result
