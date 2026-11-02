"""Within-team ranks of prior minutes, points, shot volume, usage, and TS%."""

from __future__ import annotations

import pandas as pd

_RANK_SOURCES = (
    ("min_rank_10", "min_mean_10"),
    ("pts_rank_10", "pts_mean_10"),
    ("fga_rank_10", "fga_rolling_10"),
    ("usg_rank_10", "usg_rolling_10"),
    ("ts_rank_10", "ts_pct_10"),
)
_KEYS = ["season_year", "team_id", "game_id", "player_id"]


def add_role_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 is the highest prior value on that team's roster.

    The ranked values are the existing last-10 features, which
    already exclude the current game. A player with no prior
    games stays unranked. Someone who rests between his first
    and last game with the team still takes a spot: the value
    is the prior his next appearance will use, and it is not
    written back onto an appearance that has no history yet.
    """
    missing = [source for _, source in _RANK_SOURCES if source not in frame.columns]
    if missing:
        raise ValueError(f"role ranks need prior features: {missing}")
    if frame.duplicated(_KEYS).any():
        raise ValueError("duplicate player games")

    sources = [source for _, source in _RANK_SOURCES]
    grid = _roster_grid(frame)
    stats = frame.loc[:, [*_KEYS, *sources]].copy()
    stats["_appeared"] = True
    grid = grid.merge(stats, on=_KEYS, how="left")
    grid["_appeared"] = grid["_appeared"].eq(True)
    grid = grid.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="mergesort",
    )
    for source in sources:
        carried = grid.groupby("player_id", sort=False)[source].bfill()
        grid[source] = grid[source].where(grid["_appeared"], carried)

    for name, source in _RANK_SOURCES:
        grid[name] = grid.groupby(["team_id", "game_id"], sort=False)[source].rank(
            ascending=False,
            method="dense",
        )
    ranked = grid.loc[
        grid["_appeared"],
        ["team_id", "game_id", "player_id", *[name for name, _ in _RANK_SOURCES]],
    ]
    return frame.merge(
        ranked,
        on=["team_id", "game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )


def _roster_grid(frame: pd.DataFrame) -> pd.DataFrame:
    schedule = frame.loc[
        :,
        ["season_year", "team_id", "game_id", "game_date"],
    ].drop_duplicates(["team_id", "game_id"])
    tenure = (
        frame.groupby(["season_year", "team_id", "player_id"], sort=False)["game_date"]
        .agg(first="min", last="max")
        .reset_index()
    )
    grid = tenure.merge(schedule, on=["season_year", "team_id"], how="inner")
    covered = grid["game_date"].ge(grid["first"]) & grid["game_date"].le(grid["last"])
    return grid.loc[
        covered,
        ["season_year", "team_id", "game_id", "game_date", "player_id"],
    ].copy()
