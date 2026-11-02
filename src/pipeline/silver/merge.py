"""Merge raw player, team, and tracking datasets."""

from __future__ import annotations

import logging

import pandas as pd

from .columns import (
    assign_playing_minutes,
    drop_silver_junk,
    normalize_columns,
    parse_minutes,
    prepare_tracking,
)

logger = logging.getLogger(__name__)

# Counting stats a feed sometimes splits onto a second 0-minute row.
# Rates, ranks, and minutes stay on the row for the team that played.
_COUNTING_STATS = (
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "blka",
    "pf",
    "pfd",
    "pts",
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
    players = _collapse_split_player_games(players)
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

    return assign_playing_minutes(drop_silver_junk(result)).reset_index(drop=True)


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


def _collapse_split_player_games(players: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per player-game when the feed splits the box score.

    A player cannot play for two teams in one game. The live feed
    sometimes parks a fragment, such as a free throw, on the opponent
    with zero minutes. Those counts are added to the row that has
    playing time, and the empty row is dropped.
    """
    keys = ["game_id", "player_id"]
    if not players.duplicated(keys).any():
        return players

    result = players.copy()
    minutes = _row_minutes(result)
    duplicate = result.duplicated(keys, keep=False)
    dropped = 0

    for (game_id, player_id), group in result.loc[duplicate].groupby(
        keys,
        sort=False,
    ):
        played = group.index[minutes.loc[group.index].gt(0)]
        if len(played) != 1:
            raise ValueError(
                "player has more than one positive-minute row in a game: "
                f"game_id={game_id} player_id={player_id}"
            )
        keep = played[0]
        extras = group.index.difference([keep])
        for column in _COUNTING_STATS:
            if column not in result.columns:
                continue
            extra = pd.to_numeric(
                result.loc[extras, column],
                errors="coerce",
            ).fillna(0)
            current = pd.to_numeric(
                result.at[keep, column],
                errors="coerce",
            )
            result.at[keep, column] = (
                0.0 if pd.isna(current) else float(current)
            ) + float(extra.sum())
        dropped += len(extras)
        team = (
            result.at[keep, "team_abbreviation"]
            if "team_abbreviation" in result.columns
            else result.at[keep, "team_id"]
        )
        logger.warning(
            "Collapsed %d zero-minute box row(s) for player %s in game %s onto team %s",
            len(extras),
            player_id,
            game_id,
            team,
        )

    logger.warning(
        "Dropped %d split box rows so each player has one row per game",
        dropped,
    )
    extras = duplicate & minutes.fillna(0).le(0)
    return result.loc[~extras].reset_index(drop=True)


def _row_minutes(frame: pd.DataFrame) -> pd.Series:
    for column in ("min", "minutes"):
        if column not in frame:
            continue
        values = parse_minutes(frame[column])
        if values.notna().any():
            return values
    return pd.Series(0.0, index=frame.index)


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
        if column not in {"game_id", "player_id"} and column not in players.columns
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
            renamed[column] = "team_id" if prefix == "team" else "opp_team_id"
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
        raise ValueError(f"{dataset} is missing required columns: {sorted(missing)}")
