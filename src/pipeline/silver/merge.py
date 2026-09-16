"""Merge raw player, team, and tracking datasets."""

from __future__ import annotations

import pandas as pd

from .columns import (
    assign_playing_minutes,
    drop_silver_junk,
    normalize_columns,
    prepare_tracking,
)

def merge_gamelogs(
    player_base: pd.DataFrame,
    player_adv: pd.DataFrame | None,
    team_base: pd.DataFrame,
    team_adv: pd.DataFrame | None,
    start_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    player_base = normalize_columns(player_base)
    team_base = normalize_columns(team_base)
    player_adv = normalize_columns(player_adv)
    team_adv = normalize_columns(team_adv)
    start_positions = prepare_tracking(start_positions)

    _require_columns(
        player_base,
        {"game_id", "player_id", "team_id"},
        "player_base",
    )
    _require_columns(
        team_base,
        {"game_id", "team_id"},
        "team_base",
    )

    players = _merge_player_frames(player_base, player_adv)
    players = _merge_tracking(players, start_positions)

    teams = _merge_team_frames(team_base, team_adv)
    own_team = _prefix_team_columns(teams, "team")
    opponent = _prefix_team_columns(teams, "opp")

    result = players.merge(
        own_team,
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        opponent,
        on="game_id",
        how="left",
        validate="many_to_many",
    )

    result = result.loc[result["team_id"] != result["opp_team_id"]]

    return assign_playing_minutes(
        drop_silver_junk(result)
    ).reset_index(drop=True)

def _merge_player_frames(
    base: pd.DataFrame,
    advanced: pd.DataFrame,
) -> pd.DataFrame:
    if advanced.empty:
        return base.copy()

    _require_columns(
        advanced,
        {"game_id", "player_id", "team_id"},
        "player_adv",
    )

    return base.merge(
        advanced,
        on=["game_id", "player_id", "team_id"],
        how="left",
        suffixes=("", "_adv"),
        validate="one_to_one",
    )

def _merge_tracking(
    players: pd.DataFrame,
    tracking: pd.DataFrame,
) -> pd.DataFrame:
    if tracking.empty:
        return players

    _require_columns(
        tracking,
        {"game_id", "player_id"},
        "start_positions",
    )

    tracking = tracking.drop_duplicates(
        subset=["game_id", "player_id"],
        keep="last",
    )

    tracking_columns = ["game_id", "player_id"] + [
        column
        for column in tracking.columns
        if column not in {"game_id", "player_id"}
        and column not in players.columns
    ]

    return players.merge(
        tracking[tracking_columns],
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

def _merge_team_frames(
    base: pd.DataFrame,
    advanced: pd.DataFrame,
) -> pd.DataFrame:
    if advanced.empty:
        return base.copy()

    _require_columns(
        advanced,
        {"game_id", "team_id"},
        "team_adv",
    )

    return base.merge(
        advanced,
        on=["game_id", "team_id"],
        how="left",
        suffixes=("_base", "_adv"),
        validate="one_to_one",
    )

def _prefix_team_columns(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    renamed = {}

    for column in frame.columns:
        if column == "game_id":
            renamed[column] = "game_id"
        elif column == "team_id":
            renamed[column] = (
                "team_id" if prefix == "team" else "opp_team_id"
            )
        else:
            renamed[column] = f"{prefix}_{column}"

    return frame.rename(columns=renamed)

def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    dataset: str,
) -> None:
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{dataset} is missing required columns: {sorted(missing)}"
        )