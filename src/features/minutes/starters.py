"""Starter-availability features for the minutes model.

``starts_lN``, ``min_avg_lN`` and ``starter_rank`` use prior team games
only. ``starter_min_missing_teammates`` and ``main_starters_out`` read
whether each main starter appeared in *this* game, so they are known at
bet time only once inactives are confirmed. Roster membership comes from
each player's first and last game with the team that season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STARTER_LOOKBACK_GAMES = 10
MAIN_STARTER_SLOTS = 5


def add_starter_features(
    frame: pd.DataFrame,
    lookback: int = STARTER_LOOKBACK_GAMES,
    slots: int = MAIN_STARTER_SLOTS,
) -> pd.DataFrame:
    """Merge starter-availability columns onto ``frame`` by team-game-player."""
    d = frame.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["mins"] = d["minutes"]
    d["started"] = d["starting"].fillna(0).astype(int)
    d["played"] = (d["mins"] > 0).astype(int)

    sched = (
        d[["season_year", "team_id", "game_id", "game_date"]]
        .drop_duplicates(["team_id", "game_id"])
        .sort_values(["team_id", "game_date"])
    )
    sched["tg_num"] = sched.groupby(["season_year", "team_id"]).cumcount()

    tenure = (
        d.groupby(["season_year", "team_id", "player_id"])["game_date"]
        .agg(first="min", last="max")
        .reset_index()
    )
    grid = tenure.merge(sched, on=["season_year", "team_id"])
    grid = grid[
        (grid["game_date"] >= grid["first"]) & (grid["game_date"] <= grid["last"])
    ]
    grid = grid.merge(
        d[["team_id", "game_id", "player_id", "mins", "started", "played"]],
        on=["team_id", "game_id", "player_id"],
        how="left",
    )
    grid[["mins", "started", "played"]] = grid[["mins", "started", "played"]].fillna(0)
    grid = grid.sort_values(["season_year", "team_id", "player_id", "tg_num"])

    g = grid.groupby(["season_year", "team_id", "player_id"])
    grid["starts_lN"] = g["started"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=1).sum()
    )
    grid["mins_played"] = grid["mins"].where(grid["played"] == 1)
    grid["min_avg_lN"] = g["mins_played"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=1).mean()
    )

    grid = grid.sort_values(
        ["team_id", "game_id", "starts_lN", "min_avg_lN"],
        ascending=[True, True, False, False],
    )
    grid["starter_rank"] = grid.groupby(["team_id", "game_id"]).cumcount() + 1
    grid["is_main_starter"] = (
        (grid["starter_rank"] <= slots) & (grid["starts_lN"] > 0)
    ).astype(int)

    main = grid[grid["is_main_starter"] == 1]
    team_feats = (
        main.assign(missing_min=main["min_avg_lN"].where(main["played"] == 0, 0))
        .groupby(["team_id", "game_id"])
        .agg(
            main_starters_playing=("played", "sum"),
            main_starters_defined=("player_id", "size"),
            starter_min_missing=("missing_min", "sum"),
        )
        .reset_index()
    )

    out = frame.merge(team_feats, on=["team_id", "game_id"], how="left").merge(
        grid[
            [
                "team_id",
                "game_id",
                "player_id",
                "is_main_starter",
                "min_avg_lN",
                "starter_rank",
            ]
        ],
        on=["team_id", "game_id", "player_id"],
        how="left",
    )
    # The player's own absence is excluded so his label cannot leak in.
    own_missing = np.where(
        (out["is_main_starter"] == 1) & (out["minutes"].fillna(0) == 0),
        out["min_avg_lN"].fillna(0),
        0,
    )
    out["starter_min_missing_teammates"] = out["starter_min_missing"] - own_missing
    out["main_starters_out"] = out["main_starters_defined"] - out["main_starters_playing"]
    return out
