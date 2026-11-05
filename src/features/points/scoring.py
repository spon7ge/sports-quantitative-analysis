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
    result["pts_median_10"] = prior_roll(
        result,
        "pts",
        PLAYER_KEY,
        10,
        "median",
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
    return add_raw_box_history(result)


# The points model uses half-lives 5, 10, and 20. Half-life 10
# still feeds the shooting-percentage ratios.
RAW_EWM_HALFLIVES = (5, 10, 20)
_RAW_COUNT_COLUMNS = (
    "pts",
    "fga",
    "fta",
    "fg3_a",
    "fgm",
    "fg3_m",
    "ftm",
    "poss",
    "tchs",
    "pfd",
    "cfga",
    "ufga",
    "two_pa",
    "two_pm",
)
# Feature prefix differs from the box-score column for threes.
_ROLLING_NAMES = (
    ("pts", "pts"),
    ("fga", "fga"),
    ("fta", "fta"),
    ("fg3_a", "fg3a"),
    ("two_pa", "two_pa"),
    ("poss", "poss"),
    ("tchs", "tchs"),
    ("pfd", "pfd"),
    ("cfga", "cfga"),
    ("ufga", "ufga"),
)


def add_raw_box_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Prior-only raw box-score level and shooting efficiency.

    Counts are an EWMA of the stat itself. Shooting percentages
    are ratios of those EWMs, not an EWMA of the game percentage:
    a 1-for-1 night is not the same evidence as 10-for-20, and a
    night with no attempts has no percentage. The current game
    is excluded by ``prior_shift`` / ``prior_ewm`` / ``prior_roll``.
    """
    result = frame.copy()
    for column in ("pts", "fga", "fta", "fg3_a", "fgm", "fg3_m", "ftm",
                   "poss", "tchs", "pfd", "cfga", "ufga"):
        result[column] = numeric_column(result, column)
    # Two-point attempts / makes from the box score identity.
    result["two_pa"] = result["fga"] - result["fg3_a"]
    result["two_pm"] = result["fgm"] - result["fg3_m"]
    result["fga_lag_1"] = prior_shift(result, "fga", PLAYER_KEY)

    smoothed_by_halflife = {
        halflife: {
            column: prior_ewm(
                result,
                column,
                PLAYER_KEY,
                halflife=halflife,
            )
            for column in _RAW_COUNT_COLUMNS
        }
        for halflife in RAW_EWM_HALFLIVES
    }
    for halflife, smoothed_at in smoothed_by_halflife.items():
        result[f"pts_ewm_hl_{halflife}"] = smoothed_at["pts"]
        result[f"fga_ewm_hl_{halflife}"] = smoothed_at["fga"]
        result[f"fta_ewm_hl_{halflife}"] = smoothed_at["fta"]
        result[f"fg3a_ewm_hl_{halflife}"] = smoothed_at["fg3_a"]
        result[f"two_pa_ewm_hl_{halflife}"] = smoothed_at["two_pa"]
        result[f"poss_ewm_hl_{halflife}"] = smoothed_at["poss"]
        result[f"tchs_ewm_hl_{halflife}"] = smoothed_at["tchs"]
        result[f"pfd_ewm_hl_{halflife}"] = smoothed_at["pfd"]
        result[f"cfga_ewm_hl_{halflife}"] = smoothed_at["cfga"]
        result[f"ufga_ewm_hl_{halflife}"] = smoothed_at["ufga"]
    smoothed = smoothed_by_halflife[10]
    # Windows 5, 10, and 20 are the points-model contract.
    # Window 3 remains for the older mean-model columns.
    for source, name in _ROLLING_NAMES:
        for window in (5, 10, 20, 3):
            result[f"{name}_rolling_{window}"] = prior_roll(
                result,
                source,
                PLAYER_KEY,
                window,
                "mean",
            )
    # Unweighted mean of prior usage. usg_wmean_10 weights the same
    # games by minutes, so a short stint moves this column more.
    result["usg_pct"] = numeric_column(result, "usg_pct")
    result["usg_rolling_3"] = prior_roll(
        result,
        "usg_pct",
        PLAYER_KEY,
        3,
        "mean",
    )
    result["usg_rolling_10"] = prior_roll(
        result,
        "usg_pct",
        PLAYER_KEY,
        10,
        "mean",
    )
    result["usg_median_10"] = prior_roll(
        result,
        "usg_pct",
        PLAYER_KEY,
        10,
        "median",
    )
    for source, name in (
        ("two_pa", "two_pa"),
        ("fg3_a", "fg3a"),
        ("fta", "fta"),
    ):
        result[f"{name}_median_10"] = prior_roll(
            result,
            source,
            PLAYER_KEY,
            10,
            "median",
        )
    # Extremes of the prior ten games. The current game is excluded
    # by prior_roll, same as the rolling means above.
    for source, name in (
        ("cfga", "cfga"),
        ("ufga", "ufga"),
        ("fta", "fta"),
        ("fg3_a", "fg3a"),
        ("tchs", "tchs"),
        ("usg_pct", "usg_pct"),
    ):
        result[f"{name}_min_10"] = prior_roll(
            result,
            source,
            PLAYER_KEY,
            10,
            "min",
        )
        result[f"{name}_max_10"] = prior_roll(
            result,
            source,
            PLAYER_KEY,
            10,
            "max",
        )
    result["fg_pct_ewm_hl_10"] = ratio(
        smoothed["fgm"],
        smoothed["fga"],
    )
    result["fg3_pct_ewm_hl_10"] = ratio(
        smoothed["fg3_m"],
        smoothed["fg3_a"],
    )
    result["ft_pct_ewm_hl_10"] = ratio(
        smoothed["ftm"],
        smoothed["fta"],
    )
    # Ratio of EWMs (hl=10), not an EWM of game percentages —
    # same construction as the *_ewm_hl_10 shooting rates above.
    result["two_pct_smoothed"] = ratio(
        smoothed["two_pm"],
        smoothed["two_pa"],
    )
    result["three_pct_smoothed"] = ratio(
        smoothed["fg3_m"],
        smoothed["fg3_a"],
    )
    result["ft_pct_smoothed"] = ratio(
        smoothed["ftm"],
        smoothed["fta"],
    )
    # Same ratio-of-EWMs construction as two/three/ft_pct_smoothed.
    result["efg_pct_smoothed"] = ratio(
        smoothed["fgm"] + 0.5 * smoothed["fg3_m"],
        smoothed["fga"],
    )
    result["ts_pct_smoothed"] = ratio(
        smoothed["pts"],
        2 * (smoothed["fga"] + 0.44 * smoothed["fta"]),
    )
    result["efg_pct_ewm_hl_10"] = result["efg_pct_smoothed"]
    result["ts_pct_ewm_hl_10"] = result["ts_pct_smoothed"]

    result["fga_std_10"] = prior_roll(
        result,
        "fga",
        PLAYER_KEY,
        10,
        "std",
        min_periods=2,
    )
    result["fga_sd_10"] = result["fga_std_10"]
    if "pts_std_10" not in result:
        result["pts_std_10"] = prior_roll(
            result,
            "pts",
            PLAYER_KEY,
            10,
            "std",
            min_periods=2,
        )
    result["pts_sd_10"] = result["pts_std_10"]
    # Season-to-date means for contested / uncontested / usage trends.
    result["season_cfga_mean"] = prior_expanding(
        result,
        "cfga",
        SEASON_PLAYER,
        "mean",
    )
    result["season_ufga_mean"] = prior_expanding(
        result,
        "ufga",
        SEASON_PLAYER,
        "mean",
    )
    result["season_usg_mean"] = prior_expanding(
        result,
        "usg_pct",
        SEASON_PLAYER,
        "mean",
    )
    return result
