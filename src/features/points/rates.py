"""Scoring-rate and shot-profile features from rolling sums."""

from __future__ import annotations

import pandas as pd

from src.features.minutes.rolling import (
    numeric_column,
    prior_ewm,
    prior_expanding,
    prior_outside_bounds,
    prior_quantile,
    prior_roll,
    prior_shift,
    prior_sum,
    prior_trim_mean,
    ratio,
)

PLAYER_KEY = ["player_id"]
SEASON_PLAYER = ["player_id", "season_year"]
# Short stints produce extreme per-game rates (6 pts in 2 min = 3.0).
RATE_MIN_MINUTES = 10
# Per-game ppm CV with the 10-minute floor (2024-25): median ~0.48;
# 0.50 flags ~45% of players and ~15% of 10+ ppg scorers.
SEASON_PPM_VOL_CAP = 0.50


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
    for halflife in (10, 20):
        result[f"pts_per_min_ewm_hl_{halflife}"] = ratio(
            prior_ewm(result, "pts", PLAYER_KEY, halflife=halflife),
            prior_ewm(
                result,
                "target_minutes",
                PLAYER_KEY,
                halflife=halflife,
            ),
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
    result = _add_ppm_distribution(result)

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


def _add_ppm_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    minutes = numeric_column(result, "target_minutes")
    result["_ppm_obs"] = ratio(
        numeric_column(result, "pts"),
        minutes,
    ).where(minutes.ge(RATE_MIN_MINUTES))

    ppm_mean_10 = prior_roll(result, "_ppm_obs", PLAYER_KEY, 10, "mean")
    ppm_std_10 = prior_roll(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        10,
        "std",
        min_periods=2,
    )
    result["ppm_vol_10"] = ratio(ppm_std_10, ppm_mean_10)
    result["ppm_p20_20"] = prior_quantile(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        20,
        0.20,
    )
    result["ppm_p80_20"] = prior_quantile(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        20,
        0.80,
    )
    result["ppm_p80_minus_p20_20"] = (
        result["ppm_p80_20"] - result["ppm_p20_20"]
    )
    result["ppm_trim_mean_20"] = prior_trim_mean(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        20,
    )
    result["ppm_floor_10"] = prior_roll(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        10,
        "min",
        min_periods=2,
    )
    result["ppm_ceiling_10"] = prior_roll(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        10,
        "max",
        min_periods=2,
    )
    result["ppm_span_10"] = (
        result["ppm_ceiling_10"] - result["ppm_floor_10"]
    )
    result["ppm_lag_outside_10"] = prior_outside_bounds(
        result,
        "_ppm_obs",
        PLAYER_KEY,
        10,
    )

    season_ppm_mean = prior_expanding(
        result,
        "_ppm_obs",
        SEASON_PLAYER,
        "mean",
    )
    result["season_ppm_std"] = prior_expanding(
        result,
        "_ppm_obs",
        SEASON_PLAYER,
        "std",
        min_periods=2,
    )
    result["season_ppm_vol"] = ratio(
        result["season_ppm_std"],
        season_ppm_mean,
    )
    result["ppm_unstable"] = (
        result["season_ppm_vol"]
        .gt(SEASON_PPM_VOL_CAP)
        .astype(float)
        .where(result["season_ppm_vol"].notna())
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
