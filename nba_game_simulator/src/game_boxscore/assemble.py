"""Stack one game's minutes, shots, and events into a box score."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from game_boxscore.checks import _normalize_id, _player_id_tuple, _reject_shared_players
from game_boxscore.constants import (
    _COUNT_STATS,
    _EVENT_STATS,
    _MINUTE_TOLERANCE,
    _SHOT_STATS,
    SIDES,
)
from game_boxscore.types import BoxScoreSet, TeamBoxScore


def assemble_box_score(
    environment: pd.DataFrame,
    players: Mapping[str, Any],
    minutes: Mapping[str, Any],
    shots: Mapping[str, Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
    *,
    availability: Mapping[str, np.ndarray] | None = None,
    check_score: bool = True,
) -> BoxScoreSet:
    """Stack one game's minutes, shots, and events. Row ``k`` is one outcome.

    Teams are keyed ``home`` and ``away``. ``players[side]`` is the player
    order (a sequence or a frame with ``player_id``). ``shots`` is Layer 4's
    output and ``events`` is Layer 5's; Layer 5's ``AST`` and ``OREB`` are
    the box-score values. With ``check_score``, team points must equal the
    rounded Layer 1 score in every row.
    """
    if not isinstance(environment, pd.DataFrame) or environment.empty:
        raise ValueError("environment must be a non-empty DataFrame")
    for label, value in (
        ("players", players),
        ("minutes", minutes),
        ("shots", shots),
        ("events", events),
    ):
        if not isinstance(value, Mapping) or set(value) != set(SIDES):
            raise ValueError(f"{label} must be keyed by home and away")
    for column in ("team_pts", "opp_pts", "is_home", "game_minutes"):
        if column not in environment.columns:
            raise ValueError(f"environment needs {column}")
    n_sims = len(environment)
    game_minutes = pd.to_numeric(environment["game_minutes"]).to_numpy(dtype=float)

    teams: dict[str, TeamBoxScore] = {}
    for side in SIDES:
        ids = _player_id_tuple(players[side], f"{side} players")
        n_players = len(ids)
        stats = {"MIN": _minute_matrix(minutes[side], ids, environment.index, side)}
        for stat in _SHOT_STATS:
            stats[stat] = _count_matrix(
                shots[side], stat, n_sims, n_players, f"{side} shots"
            )
        for stat in _EVENT_STATS:
            stats[stat] = _count_matrix(
                events[side], stat, n_sims, n_players, f"{side} events"
            )
        _check_minutes(side, stats["MIN"], game_minutes)
        _check_player_identities(side, stats)
        mask = (
            None
            if availability is None
            else np.broadcast_to(
                np.asarray(availability[side], dtype=bool), (n_sims, n_players)
            )
        )
        _check_on_floor(side, stats, mask)
        teams[side] = TeamBoxScore(player_ids=ids, stats=stats)

    _reject_shared_players({side: teams[side].player_ids for side in SIDES})
    _check_cross_team(teams)
    if check_score:
        _check_score(teams, environment)
    return BoxScoreSet(
        teams=teams,
        n_sims=n_sims,
        environment=environment,
        metadata={"assembled": True},
    )


def _minute_matrix(value: Any, ids: tuple, index: pd.Index, side: str) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        columns = tuple(_normalize_id(column) for column in value.columns)
        if columns != ids:
            raise ValueError(f"{side} minute columns must match the player order")
        if not value.index.equals(index):
            raise ValueError(f"{side} minute rows must keep the Layer 1 index")
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{side} MIN must be numeric") from exc
    if arr.shape != (len(index), len(ids)):
        raise ValueError(f"{side} MIN must have shape ({len(index)}, {len(ids)})")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{side} MIN must be nonnegative and finite")
    return arr.copy()


def _count_matrix(
    source: Mapping[str, Any], stat: str, n_sims: int, n_players: int, label: str
) -> np.ndarray:
    if stat not in source:
        raise ValueError(f"{label} need {stat}")
    try:
        arr = np.asarray(source[stat], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} {stat} must be numeric") from exc
    if arr.shape != (n_sims, n_players):
        raise ValueError(f"{label} {stat} must have shape ({n_sims}, {n_players})")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        raise ValueError(f"{label} {stat} must be nonnegative whole numbers")
    return arr.astype(np.int64)


def _check_minutes(side: str, played: np.ndarray, game_minutes: np.ndarray) -> None:
    budget = 5.0 * game_minutes
    if not np.allclose(played.sum(axis=1), budget, atol=_MINUTE_TOLERANCE):
        raise ValueError(f"{side} minutes must sum to 5 * game_minutes in every row")
    if np.any(played > game_minutes[:, None] + _MINUTE_TOLERANCE):
        raise ValueError(f"{side} has a player above the game length")


def _check_player_identities(side: str, stats: dict[str, np.ndarray]) -> None:
    rules = (
        ("FGA = 2PA + 3PA", stats["FGA"] == stats["2PA"] + stats["3PA"]),
        ("FGM = 2PM + 3PM", stats["FGM"] == stats["2PM"] + stats["3PM"]),
        ("2PM <= 2PA", stats["2PM"] <= stats["2PA"]),
        ("3PM <= 3PA", stats["3PM"] <= stats["3PA"]),
        ("FTM <= FTA", stats["FTM"] <= stats["FTA"]),
        (
            "PTS = 2 * 2PM + 3 * 3PM + FTM",
            stats["PTS"] == 2 * stats["2PM"] + 3 * stats["3PM"] + stats["FTM"],
        ),
        ("REB = OREB + DREB", stats["REB"] == stats["OREB"] + stats["DREB"]),
        (
            "AST <= teammates' FGM",
            stats["AST"] <= stats["FGM"].sum(axis=1, keepdims=True) - stats["FGM"],
        ),
    )
    for rule, holds in rules:
        if not np.all(holds):
            raise ValueError(f"{side} breaks {rule}")


def _check_on_floor(
    side: str, stats: dict[str, np.ndarray], mask: np.ndarray | None
) -> None:
    played = stats["MIN"] > 0.0
    if mask is not None and np.any(played & ~mask):
        raise ValueError(f"{side} unavailable players received minutes")
    for stat in _COUNT_STATS:
        if np.any((stats[stat] > 0) & ~played):
            raise ValueError(f"{side} players without minutes recorded {stat}")


def _check_cross_team(teams: dict[str, TeamBoxScore]) -> None:
    for side, other in (("home", "away"), ("away", "home")):
        own, opp = teams[side].stats, teams[other].stats
        misses = own["FGA"].sum(axis=1) - own["FGM"].sum(axis=1)
        if np.any(own["OREB"].sum(axis=1) + opp["DREB"].sum(axis=1) > misses):
            raise ValueError(f"rebounds on {side} misses exceed those misses")
        if np.any(opp["BLK"].sum(axis=1) > misses):
            raise ValueError(f"blocks on {side} shots exceed its misses")
        if np.any(opp["STL"].sum(axis=1) > own["TOV"].sum(axis=1)):
            raise ValueError(f"steals exceed {side} turnovers")


def _check_score(teams: dict[str, TeamBoxScore], environment: pd.DataFrame) -> None:
    is_home = pd.to_numeric(environment["is_home"]).to_numpy() == 1
    team_pts = pd.to_numeric(environment["team_pts"]).to_numpy(dtype=float)
    opp_pts = pd.to_numeric(environment["opp_pts"]).to_numpy(dtype=float)
    targets = {
        "home": np.rint(np.where(is_home, team_pts, opp_pts)),
        "away": np.rint(np.where(is_home, opp_pts, team_pts)),
    }
    for side in SIDES:
        points = teams[side].stats["PTS"].sum(axis=1)
        wrong = points != targets[side]
        if np.any(wrong):
            raise ValueError(
                f"{side} points differ from the Layer 1 score "
                f"in {int(wrong.sum())} simulations"
            )
