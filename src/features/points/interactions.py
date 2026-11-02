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
    # Last 3 appearances vs season-to-date average (current game excluded).
    result["min_trend"] = (
        result["min_mean_3"] - result["season_min_mean"]
    )
    result["pts_trend"] = (
        result["pts_mean_3"] - result["season_pts_mean"]
    )
    result["cfga_trend"] = (
        result["cfga_rolling_3"] - result["season_cfga_mean"]
    )
    result["ufga_trend"] = (
        result["ufga_rolling_3"] - result["season_ufga_mean"]
    )
    result["usg_trend"] = (
        result["usg_rolling_3"] - result["season_usg_mean"]
    )
    result["usage_x_expected_possessions"] = (
        result["usg_wmean_10"]
        * result["expected_possessions"]
    )
    # Usage stays on its stored percent scale. True shooting is the
    # prior-10 aggregate ratio. Both already exclude the current game.
    result["usg_x_ts_pct_10"] = (
        result["usg_wmean_10"] * result["ts_pct_10"]
    )
    # Same prior-10 product. The points notebook lists it as ts_x_usg.
    result["ts_x_usg"] = result["usg_x_ts_pct_10"]
    # Trees split on one column at a time, so the market total and the
    # spread rarely become "implied team total" on their own. The share
    # times that total is a projected shot volume.
    result["fga_share_x_implied_total"] = (
        result["player_fga_share_10"] * result["implied_team_total"]
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
