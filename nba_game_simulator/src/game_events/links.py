"""Play-by-play links for assists, rebound chances, blocks, and steals.

A made shot records the shooter, the zone, and the passer when the make is
assisted. A miss is a rebound chance for the five on each side of the floor.
A block is marked on that shot. A steal is marked on a live-ball turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from possession_table import _parse_game, _prepare, _prepare_starters

_LINEUP = tuple(f"{side}_{index}" for side in ("off", "def") for index in range(1, 6))
SHOT_COLUMNS = (
    "game_id",
    "shooter_id",
    "zone",
    "made",
    "assisted",
    "passer_id",
    "blocked",
    "blocker_id",
    "offense_team_id",
    "defense_team_id",
    *_LINEUP,
)
REBOUND_COLUMNS = (
    "game_id",
    "shooter_id",
    "zone",
    "outcome",
    "rebounder_id",
    "offense_team_id",
    "defense_team_id",
    *_LINEUP,
)
TURNOVER_COLUMNS = (
    "game_id",
    "player_id",
    "subtype",
    "live_ball",
    "stealer_id",
    "offense_team_id",
    "defense_team_id",
    *_LINEUP,
)
_NULLABLE = {
    "passer_id",
    "blocker_id",
    "rebounder_id",
    "stealer_id",
    "player_id",
}


@dataclass(frozen=True)
class EventLinks:
    """Shot, rebound-chance, and turnover rows from one play-by-play feed."""

    shots: pd.DataFrame
    rebounds: pd.DataFrame
    turnovers: pd.DataFrame


def event_links(pbp: pd.DataFrame, starters: pd.DataFrame | None = None) -> EventLinks:
    """One row per shot, missed-shot rebound chance, and turnover.

    Lineups are the five on the floor when the event happened. ``outcome``
    on a rebound chance is ``oreb``, ``dreb``, ``team``, or ``unresolved``.
    A team rebound is why player rebounds do not add up to the misses.
    ``live_ball`` is true only for the Bad Pass and Lost Ball labels.
    """
    frame = _prepare(pbp)
    starter_rows = _prepare_starters(starters)
    shots: list[dict] = []
    rebounds: list[dict] = []
    turnovers: list[dict] = []
    for game_id, game in frame.groupby("gameId", sort=False):
        parser = _parse_game(str(game_id), game, starter_rows)
        shots.extend(parser.shot_links)
        rebounds.extend(parser.rebound_links)
        turnovers.extend(parser.turnover_links)
    return EventLinks(
        _frame(shots, SHOT_COLUMNS),
        _frame(rebounds, REBOUND_COLUMNS),
        _frame(turnovers, TURNOVER_COLUMNS),
    )


def _frame(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=list(columns))
    for column in columns:
        if column in _NULLABLE or column.endswith("_id") and column != "game_id":
            if column == "game_id":
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "Int64"
            )
    return frame
