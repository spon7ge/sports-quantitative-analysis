"""Player-game minutes, role, usage, and stint features."""

from __future__ import annotations

import pandas as pd

from .rolling import (
    group_keys,
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


def add_observation_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    minutes = result["target_minutes"]
    result["_appeared_obs"] = (
        minutes.gt(0).astype(float).where(minutes.notna())
    )
    start = (
        result["start_position"]
        if "start_position" in result
        else pd.Series("", index=result.index)
    )
    start_text = start.astype("string").str.strip()
    result["_started_obs"] = (
        start_text.notna()
        & start_text.ne("")
        & start_text.ne("nan")
        & start_text.ne("<NA>")
    ).astype(float)
    result["_ge30_obs"] = (
        minutes.ge(30).astype(float).where(minutes.notna())
    )
    result["_usg_x_min"] = (
        numeric_column(result, "usg_pct") * minutes
    )
    result["_activity_obs"] = _activity(result)
    return result


def add_player_role_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    minutes = "target_minutes"

    result["min_lag_1"] = prior_shift(
        result,
        minutes,
        PLAYER_KEY,
    )
    for window in (3, 10, 20):
        result[f"min_mean_{window}"] = prior_roll(
            result,
            minutes,
            PLAYER_KEY,
            window,
            "mean",
        )

    result["min_std_10"] = prior_roll(
        result,
        minutes,
        PLAYER_KEY,
        10,
        "std",
        min_periods=2,
    )
    result["min_ewm_hl_3"] = prior_ewm(
        result,
        minutes,
        PLAYER_KEY,
        halflife=3,
    )
    result["min_ge_30_rate_10"] = prior_roll(
        result,
        "_ge30_obs",
        PLAYER_KEY,
        10,
        "mean",
    )
    result["start_rate_10"] = prior_roll(
        result,
        "_started_obs",
        PLAYER_KEY,
        10,
        "mean",
    )
    return result


def add_season_and_stint_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    minutes = "target_minutes"

    result["season_rows_prior"] = result.groupby(
        SEASON_PLAYER,
        sort=False,
    ).cumcount()
    appeared = prior_shift(
        result,
        "_appeared_obs",
        SEASON_PLAYER,
    ).fillna(0).astype(float)
    result["season_appearances_prior"] = appeared.groupby(
        group_keys(result, SEASON_PLAYER),
        sort=False,
    ).cumsum()
    result["season_min_mean"] = prior_expanding(
        result,
        minutes,
        SEASON_PLAYER,
        "mean",
    )
    result["season_min_std"] = prior_expanding(
        result,
        minutes,
        SEASON_PLAYER,
        "std",
        min_periods=2,
    )
    result["season_start_rate"] = prior_expanding(
        result,
        "_started_obs",
        SEASON_PLAYER,
        "mean",
    )

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
    result["current_team_rows_prior"] = result.groupby(
        stint_key,
        sort=False,
    ).cumcount()
    result["current_team_min_mean"] = prior_expanding(
        result,
        minutes,
        stint_key,
        "mean",
    )
    return _add_prior_season(result)


def add_usage_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    minutes_sum_10 = prior_sum(
        result,
        "target_minutes",
        PLAYER_KEY,
        10,
    )
    minutes_sum_5 = prior_sum(
        result,
        "target_minutes",
        PLAYER_KEY,
        5,
    )
    result["usg_wmean_10"] = ratio(
        prior_sum(result, "_usg_x_min", PLAYER_KEY, 10),
        minutes_sum_10,
    )
    usg_wmean_5 = ratio(
        prior_sum(result, "_usg_x_min", PLAYER_KEY, 5),
        minutes_sum_5,
    )
    season_usg = ratio(
        prior_expanding(
            result,
            "_usg_x_min",
            SEASON_PLAYER,
            "sum",
        ),
        prior_expanding(
            result,
            "target_minutes",
            SEASON_PLAYER,
            "sum",
        ),
    )
    result["usg_5_minus_season"] = (
        usg_wmean_5 - season_usg
    )

    sources = {
        "tchs": "touches_per_min_10",
        "pass": "passes_per_min_10",
        "fga": "fga_per_min_10",
        "_activity_obs": "activity_per_min_10",
    }
    for source, output in sources.items():
        result[source] = numeric_column(result, source)
        result[output] = ratio(
            prior_sum(result, source, PLAYER_KEY, 10),
            minutes_sum_10,
        )

    assists = numeric_column(
        result,
        "assists",
        fallback="ast",
    )
    result["_assists_obs"] = assists
    result["assists_per_min_10"] = ratio(
        prior_sum(result, "_assists_obs", PLAYER_KEY, 10),
        minutes_sum_10,
    )
    return result


def _activity(frame: pd.DataFrame) -> pd.Series:
    parts = pd.concat(
        [
            numeric_column(frame, column)
            for column in ("reb", "stl", "blk")
        ],
        axis=1,
    )
    return parts.sum(axis=1, min_count=1)


def _add_prior_season(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(
            ["player_id", "season_year"],
            as_index=False,
            sort=False,
        )
        .agg(
            completed_season_min_mean=(
                "target_minutes",
                "mean",
            ),
            completed_season_appearances=(
                "_appeared_obs",
                "sum",
            ),
        )
        .sort_values(["player_id", "season_year"])
    )
    summary["_season_start"] = _season_start(
        summary["season_year"]
    )
    grouped = summary.groupby("player_id", sort=False)
    summary["prior_season_min_mean"] = grouped[
        "completed_season_min_mean"
    ].shift(1)
    summary["prior_season_appearances"] = grouped[
        "completed_season_appearances"
    ].shift(1)
    summary["_prior_start"] = grouped[
        "_season_start"
    ].shift(1)
    adjacent = summary["_season_start"].eq(
        summary["_prior_start"] + 1
    )
    summary["prior_season_min_mean"] = summary[
        "prior_season_min_mean"
    ].where(adjacent)
    summary["prior_season_appearances"] = summary[
        "prior_season_appearances"
    ].where(adjacent)

    return frame.merge(
        summary[
            [
                "player_id",
                "season_year",
                "prior_season_min_mean",
                "prior_season_appearances",
            ]
        ],
        on=["player_id", "season_year"],
        how="left",
        validate="many_to_one",
    )


def _season_start(season_year: pd.Series) -> pd.Series:
    return pd.to_numeric(
        season_year.astype(str).str.slice(0, 4),
        errors="coerce",
    )
