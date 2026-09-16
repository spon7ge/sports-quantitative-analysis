"""Rest and schedule-density features.

Team and player density windows reset by season so
the offseason is not treated as extraordinary rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_team_schedule_features(
    teams: pd.DataFrame,
) -> pd.DataFrame:
    result = teams.sort_values(
        ["team_id", "season_year", "game_date", "game_id"]
    ).copy()
    result["team_days_since_prev_game"] = (
        result.groupby(
            ["team_id", "season_year"],
            sort=False,
        )["game_date"]
        .diff()
        .dt.days
    )
    result["is_back_to_back"] = (
        result["team_days_since_prev_game"]
        .eq(1)
        .astype(float)
    )
    result["team_games_prev_3d"] = _group_lookback(
        result,
        ["team_id", "season_year"],
        days=3,
    )
    result["team_games_prev_5d"] = _group_lookback(
        result,
        ["team_id", "season_year"],
        days=5,
    )
    return result


def add_player_schedule_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    days = pd.Series(np.nan, index=result.index)
    appearances = pd.Series(np.nan, index=result.index)

    grouped = result.groupby(
        ["player_id", "season_year"],
        sort=False,
    )
    for _, group in grouped:
        appeared = (
            group["_appeared_obs"]
            .fillna(0)
            .astype(float)
            .gt(0)
        )
        prior_appearance = (
            group["game_date"]
            .where(appeared)
            .shift(1)
            .ffill()
        )
        gap = (
            group["game_date"] - prior_appearance
        ).dt.days.clip(upper=30)
        days.loc[group.index] = gap
        dates = pd.to_datetime(
            group["game_date"]
        ).to_numpy()
        appearances.loc[group.index] = _lookback_count(
            dates,
            counted=dates[appeared.to_numpy()],
            days=7,
        )

    result["player_days_since_appearance"] = days
    result["player_appearances_prev_7d"] = appearances
    return result


def _group_lookback(
    frame: pd.DataFrame,
    by: list[str],
    *,
    days: int,
) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index)
    for _, group in frame.groupby(by, sort=False):
        dates = pd.to_datetime(
            group["game_date"]
        ).to_numpy()
        values.loc[group.index] = _lookback_count(
            dates,
            counted=dates,
            days=days,
        )
    return values


def _lookback_count(
    dates: np.ndarray,
    *,
    counted: np.ndarray,
    days: int,
) -> np.ndarray:
    if len(dates) == 0:
        return np.array([], dtype=float)
    start = dates - np.timedelta64(days, "D")
    if len(counted) == 0:
        return np.zeros(len(dates), dtype=float)
    left = np.searchsorted(counted, start, side="left")
    right = np.searchsorted(counted, dates, side="left")
    return (right - left).astype(float)
