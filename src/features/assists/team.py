"""Null-aware team-game snapshots and scheduled-opponent asof joins."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

_STAT_COLUMNS = (
    "team_ast",
    "team_fgm",
    "team_pace",
    "opp_ast",
)


def add_team_features(work: pd.DataFrame) -> pd.DataFrame:
    result = work.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"], errors="coerce"
    )
    snapshots = _team_snapshots(result)
    snapshots["team_id"] = _numeric_id(snapshots["team_id"])
    left = result[
        ["_builder_row", "game_date", "team_id", "opp_team_id"]
    ].copy()
    left["team_id"] = _numeric_id(left["team_id"])
    left["opp_team_id"] = _numeric_id(left["opp_team_id"])

    own_columns = [
        "game_date",
        "team_id",
        "team_ast_mean_10",
        "team_fgm_mean_10",
        "team_pace_mean_10",
    ]
    own = pd.merge_asof(
        left.sort_values("game_date", kind="mergesort"),
        snapshots[own_columns].sort_values(
            "game_date", kind="mergesort"
        ),
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
        left_by="team_id",
        right_by="team_id",
    )

    opponent = snapshots[
        [
            "game_date",
            "team_id",
            "opp_ast_mean_10",
            "team_pace_mean_10",
        ]
    ].rename(
        columns={
            "opp_ast_mean_10": "opponent_team_ast_allowed_mean_10",
            "team_pace_mean_10": "opponent_team_pace_mean_10",
        }
    )
    opponent = pd.merge_asof(
        left.sort_values("game_date", kind="mergesort"),
        opponent.sort_values("game_date", kind="mergesort"),
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
        left_by="opp_team_id",
        right_by="team_id",
    )

    for column in (
        "team_ast_mean_10",
        "team_fgm_mean_10",
        "team_pace_mean_10",
    ):
        result[column] = result["_builder_row"].map(
            own.set_index("_builder_row")[column]
        )
    for column in (
        "opponent_team_ast_allowed_mean_10",
        "opponent_team_pace_mean_10",
    ):
        result[column] = result["_builder_row"].map(
            opponent.set_index("_builder_row")[column]
        )
    return result


def _team_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    games = _reduce_team_games(frame)
    games = games.sort_values(
        ["team_id", "game_date", "game_id"],
        kind="mergesort",
    )
    for source, output in (
        ("team_ast", "team_ast_mean_10"),
        ("team_fgm", "team_fgm_mean_10"),
        ("team_pace", "team_pace_mean_10"),
        ("opp_ast", "opp_ast_mean_10"),
    ):
        games[output] = np.nan
        for _, group in games.groupby("team_id", sort=False):
            window: deque[float] = deque(maxlen=10)
            for index, value in group[source].items():
                if np.isfinite(value):
                    window.append(float(value))
                games.at[index, output] = (
                    np.nan
                    if not window
                    else float(np.mean(window))
                )
    return games


def _reduce_team_games(frame: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "team_id",
        "game_id",
        "game_date",
        "opp_team_id",
        "player_id",
        *_STAT_COLUMNS,
    ]
    present = [column for column in needed if column in frame.columns]
    team_rows = frame[present].copy()
    if "player_id" not in team_rows.columns:
        team_rows["player_id"] = np.nan
    team_rows = team_rows.sort_values(
        ["team_id", "game_id", "player_id"],
        kind="mergesort",
    )

    def first_nonnull(series: pd.Series):
        valid = series.dropna()
        return valid.iloc[0] if len(valid) else np.nan

    def first_finite(series: pd.Series):
        values = pd.to_numeric(series, errors="coerce")
        finite = values[np.isfinite(values)]
        return finite.iloc[0] if len(finite) else np.nan

    aggregations = {
        "game_date": ("game_date", first_nonnull),
        "opp_team_id": ("opp_team_id", first_nonnull),
    }
    for column in _STAT_COLUMNS:
        if column in team_rows.columns:
            aggregations[column] = (column, first_finite)
    games = team_rows.groupby(
        ["team_id", "game_id"],
        sort=False,
    ).agg(**aggregations)
    for column in _STAT_COLUMNS:
        if column not in games.columns:
            games[column] = np.nan
    return games.reset_index()


def _numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")
