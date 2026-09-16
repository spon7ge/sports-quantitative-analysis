"""Pregame points rows from history strictly before an as-of date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.points.build import add_points_features

OUTCOME_COLUMNS = (
    "minutes",
    "min",
    "min_sec",
    "pts",
    "fgm",
    "fga",
    "fg3_m",
    "fg3_a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "usg_pct",
    "tchs",
    "pass",
    "assists",
    "start_position",
    "wl",
    "plus_minus",
    "available_flag",
    "comment",
    "team_pace",
    "team_net_rating",
    "team_off_rating",
    "team_def_rating",
    "opp_pace",
    "opp_net_rating",
    "opp_def_rating",
    "team_fga",
    "team_pts",
    "opp_pts",
)


def appearances_before(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Keep completed appearances strictly before the quote as-of date."""
    cutoff = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.loc[dates < cutoff].copy()


def blank_outcomes(row: pd.Series) -> pd.Series:
    result = row.copy()
    for column in OUTCOME_COLUMNS:
        if column in result.index:
            result[column] = np.nan
    if "start_position" in result.index:
        result["start_position"] = ""
    if "comment" in result.index:
        result["comment"] = ""
    return result


def build_pregame_points_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Feature each candidate using only appearances before ``as_of``.

    Candidate rows on different future dates do not observe each other.
    Game N box scores on the candidates are ignored.
    """
    prior = appearances_before(history, as_of)
    if candidates.empty:
        return candidates.copy()

    parts: list[pd.DataFrame] = []
    dated = candidates.copy()
    dated["game_date"] = pd.to_datetime(
        dated["game_date"], errors="coerce"
    )
    for _, group in dated.groupby("game_date", sort=True):
        panel = pd.concat([prior, group], ignore_index=True)
        featured = add_points_features(panel)
        parts.append(
            featured.loc[featured["game_id"].isin(group["game_id"])]
        )
    return pd.concat(parts, ignore_index=True)
