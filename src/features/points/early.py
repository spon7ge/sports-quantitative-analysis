"""Pregame context for the first games of a season.

A 10-game window is mostly last season, or missing, until a player
has played 10 games. These columns tell the model which of those
two cases it is in. They read earlier appearances only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.minutes.rolling import numeric_column


def add_early_season_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Add games played, a team-change flag, and last season's points.

    ``games_played_this_season`` counts earlier appearances with
    minutes in this season. ``prior_season_pts_mean`` is that player's
    mean points last season, and only when last season is the adjacent
    year. ``changed_team`` is 1 when today's team is not the team from
    the last game of that prior season. A rookie, or a player who sat
    out the previous year, gets a missing team-change flag rather than
    a false 0.
    """
    required = (
        "player_id",
        "season_year",
        "team_id",
        "game_date",
        "game_id",
        "pts",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"early-season features need {missing}")

    result = frame.copy()
    ordered = result.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="mergesort",
    )
    minutes = numeric_column(ordered, "minutes", fallback="target_minutes")
    if bool(minutes.isna().all()):
        raise ValueError("early-season features need minutes")
    appeared = minutes.gt(0) & minutes.notna()
    season_key = [ordered["player_id"], ordered["season_year"]]
    prior_appearance = (
        appeared.astype(float).groupby(season_key, sort=False).shift(1).fillna(0.0)
    )
    result["games_played_this_season"] = prior_appearance.groupby(
        season_key,
        sort=False,
    ).cumsum()

    played = ordered.loc[
        appeared, ["player_id", "season_year", "team_id", "pts", "game_date", "game_id"]
    ]
    if played.empty:
        result["prior_season_pts_mean"] = np.nan
        result["changed_team"] = np.nan
        return result

    played = played.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="mergesort",
    )
    summary = played.groupby(
        ["player_id", "season_year"],
        as_index=False,
        sort=False,
    ).agg(pts_mean=("pts", "mean"))
    last_team = played.groupby(["player_id", "season_year"], sort=False).tail(1)
    summary = summary.merge(
        last_team[["player_id", "season_year", "team_id"]],
        on=["player_id", "season_year"],
        how="left",
        validate="one_to_one",
    )
    summary["_season_start"] = _season_start(summary["season_year"])
    summary = summary.sort_values(["player_id", "_season_start"], kind="mergesort")
    grouped = summary.groupby("player_id", sort=False)
    summary["prior_season_pts_mean"] = grouped["pts_mean"].shift(1)
    summary["prior_team_id"] = grouped["team_id"].shift(1)
    summary["_prior_start"] = grouped["_season_start"].shift(1)
    adjacent = summary["_season_start"].eq(summary["_prior_start"] + 1)
    summary["prior_season_pts_mean"] = summary["prior_season_pts_mean"].where(adjacent)
    summary["prior_team_id"] = summary["prior_team_id"].where(adjacent)

    carry = ordered[["player_id", "season_year", "team_id"]].copy()
    carry["_row"] = ordered.index
    brought = carry.merge(
        summary[
            [
                "player_id",
                "season_year",
                "prior_season_pts_mean",
                "prior_team_id",
            ]
        ],
        on=["player_id", "season_year"],
        how="left",
        validate="many_to_one",
    ).set_index("_row")
    result["prior_season_pts_mean"] = brought["prior_season_pts_mean"]
    changed = brought["team_id"].ne(brought["prior_team_id"]).astype(float)
    result["changed_team"] = changed.where(brought["prior_team_id"].notna())
    return result


def _season_start(season_year: pd.Series) -> pd.Series:
    """First year of a label such as ``2023-24``. Adjacent seasons differ by 1."""
    return pd.to_numeric(season_year.astype(str).str.slice(0, 4), errors="coerce")
