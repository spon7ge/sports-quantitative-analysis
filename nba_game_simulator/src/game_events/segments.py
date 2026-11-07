"""On-court weights from Layer 3 five-player schedules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from game_events.checks import _normalize_id
from game_events.constants import _MINUTE_TOLERANCE, _TIME_TOLERANCE
from game_events.simulate import _Side


@dataclass(frozen=True)
class _Schedule:
    periods: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    active: np.ndarray
    lengths: dict[int, float]


def _schedule_rows(
    schedules: Any, n_sims: int, name: str, index: pd.Index | None
) -> list:
    if isinstance(schedules, pd.Series):
        if index is not None and not schedules.index.equals(index):
            raise ValueError(f"{name} segments must share the minute index")
        values = schedules.tolist()
    else:
        try:
            values = list(schedules)
        except TypeError as exc:
            raise ValueError(
                f"{name} segments must be one schedule per simulation"
            ) from exc
    if len(values) != n_sims:
        raise ValueError(f"{name} segments must have {n_sims} schedules")
    return values


def _parse_schedule(
    schedule: Any, index: Mapping[str, int], n_players: int, name: str, row: int
) -> _Schedule:
    where = f"{name} segments row {row}"
    try:
        segments = list(schedule)
    except TypeError as exc:
        raise ValueError(f"{where} must be a list of segments") from exc
    if not segments:
        raise ValueError(f"{where} is empty")
    periods, starts, ends, columns = [], [], [], []
    for segment in segments:
        try:
            period = segment["period"]
            start = float(segment["start_min"])
            end = float(segment["end_min"])
            players = [
                player
                if isinstance(player, str) and player in index
                else _normalize_id(player)
                for player in segment["players"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where} has a malformed segment") from exc
        if isinstance(period, (bool, np.bool_)) or int(period) != period or period < 1:
            raise ValueError(f"{where} has an invalid period")
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end):
            raise ValueError(f"{where} needs 0 <= start_min < end_min")
        if len(players) != 5 or len(set(players)) != 5:
            raise ValueError(f"{where} needs five distinct players")
        try:
            columns.append([index[player] for player in players])
        except KeyError as exc:
            unknown = [player for player in players if player not in index]
            raise ValueError(
                f"{where} names players outside the roster: {unknown}"
            ) from exc
        periods.append(int(period))
        starts.append(start)
        ends.append(end)

    periods = np.asarray(periods, dtype=np.int64)
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    active = np.zeros((len(segments), n_players), dtype=bool)
    active[np.arange(len(segments))[:, None], np.asarray(columns)] = True
    order = np.lexsort((starts, periods))
    periods, starts, ends, active = (
        periods[order],
        starts[order],
        ends[order],
        active[order],
    )
    first = np.r_[True, periods[1:] != periods[:-1]]
    last = np.r_[periods[1:] != periods[:-1], True]
    gaps = np.abs(starts[1:] - ends[:-1])[~first[1:]]
    if np.any(np.abs(starts[first]) > _TIME_TOLERANCE) or np.any(
        gaps > _TIME_TOLERANCE
    ):
        raise ValueError(f"{where} has gaps or overlaps within a period")
    lengths = dict(zip(periods[last].tolist(), ends[last].tolist()))
    return _Schedule(periods, starts, ends, active, lengths)


def _merge_schedules(
    first: _Schedule, second: _Schedule, row: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split both schedules at their combined substitution boundaries."""
    if set(first.lengths) != set(second.lengths) or any(
        abs(first.lengths[p] - second.lengths[p]) > _TIME_TOLERANCE
        for p in first.lengths
    ):
        raise ValueError(f"segments row {row}: teams cover different game time")
    ordered = sorted(first.lengths)
    offsets = dict(
        zip(
            ordered,
            np.concatenate([[0.0], np.cumsum([first.lengths[p] for p in ordered])]),
        )
    )

    def on_clock(schedule: _Schedule) -> tuple[np.ndarray, np.ndarray]:
        shift = np.array([offsets[int(p)] for p in schedule.periods])
        return schedule.starts + shift, schedule.ends + shift

    first_start, first_end = on_clock(first)
    second_start, second_end = on_clock(second)
    bounds = np.unique(
        np.round(np.concatenate([first_start, first_end, second_start, second_end]), 9)
    )
    durations = np.diff(bounds)
    keep = durations > _TIME_TOLERANCE
    mids = (bounds[:-1] + bounds[1:])[keep] / 2.0
    first_k = np.searchsorted(first_start, mids, side="right") - 1
    second_k = np.searchsorted(second_start, mids, side="right") - 1
    return durations[keep], first.active[first_k], second.active[second_k]


def _interval_shares(active: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``[sim, interval, player]`` split of one event across the active five."""
    weighted = active * weight[:, None, :]
    total = weighted.sum(axis=2, keepdims=True)
    count = active.sum(axis=2, keepdims=True)
    equal = np.divide(active, count, out=np.zeros(active.shape), where=count > 0)
    return np.where(total > 0.0, weighted / np.where(total > 0.0, total, 1.0), equal)


def _event_timing(
    durations: np.ndarray, active: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """``[sim, interval]`` distribution of a team's events, given who produced them.

    Each player's events are spread uniformly over that player's on-court
    time. Without events, time is the only weight.
    """
    on_court = durations[:, :, None] * active
    player_time = on_court.sum(axis=1, keepdims=True)
    spread = np.divide(
        on_court, player_time, out=np.zeros_like(on_court), where=player_time > 0.0
    )
    total = counts.sum(axis=1)
    mix = np.divide(
        counts, total[:, None], out=np.zeros(counts.shape), where=total[:, None] > 0
    )
    timing = np.einsum("skp,sp->sk", spread, mix)
    by_time = durations / durations.sum(axis=1, keepdims=True)
    return np.where(total[:, None] > 0, timing, by_time)


def _assist_pairs(
    durations: np.ndarray, active: np.ndarray, creation: np.ndarray, chunk: int = 512
) -> np.ndarray:
    """``[sim, scorer, assister]`` weights over teammates on the scorer's floor time."""
    n_sims, _, n_players = active.shape
    not_self = 1.0 - np.eye(n_players)
    out = np.zeros((n_sims, n_players, n_players))
    for begin in range(0, n_sims, chunk):
        part = slice(begin, begin + chunk)
        on = active[part]
        weighted = on[:, :, None, :] * creation[part, None, None, :] * not_self
        total = weighted.sum(axis=3, keepdims=True)
        shares = np.divide(
            weighted, total, out=np.zeros_like(weighted), where=total > 0.0
        )
        scorer_time = durations[part, :, None] * on
        pairs = np.einsum("skj,skji->sji", scorer_time, shares)
        time = scorer_time.sum(axis=1)[:, :, None]
        out[part] = np.divide(pairs, time, out=np.zeros_like(pairs), where=time > 0.0)
    return out


def _segment_arrays(
    sides: list[_Side],
    segments: Mapping[str, Any],
    indexes: list[pd.Index | None],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Merged interval durations and on-court masks, padded to one width."""
    n_sims = sides[0].fga.shape[0]
    rows = [
        _schedule_rows(segments[side.name], n_sims, side.name, indexes[i])
        for i, side in enumerate(sides)
    ]
    lookup = [{pid: j for j, pid in enumerate(side.player_ids)} for side in sides]
    merged = []
    for row in range(n_sims):
        parsed = [
            _parse_schedule(
                rows[i][row], lookup[i], len(side.player_ids), side.name, row
            )
            for i, side in enumerate(sides)
        ]
        merged.append(_merge_schedules(parsed[0], parsed[1], row))
    width = max(len(durations) for durations, _, _ in merged)
    durations = np.zeros((n_sims, width))
    actives = [
        np.zeros((n_sims, width, len(side.player_ids)), dtype=bool) for side in sides
    ]
    for row, (row_durations, first, second) in enumerate(merged):
        k = len(row_durations)
        durations[row, :k] = row_durations
        actives[0][row, :k] = first
        actives[1][row, :k] = second
    for i, side in enumerate(sides):
        implied = np.einsum("sk,skp->sp", durations, actives[i])
        bad = ~np.isclose(implied, side.minutes, atol=_MINUTE_TOLERANCE).all(axis=1)
        if np.any(bad):
            row = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{side.name} segments row {row} disagree with Layer 2 minutes"
            )
    return durations, actives


def _apply_segments(
    sides: list[_Side],
    segments: Mapping[str, Any],
    indexes: list[pd.Index | None],
) -> list[_Side]:
    """Replace minute-based weights with on-court weights from Layer 3."""
    durations, actives = _segment_arrays(sides, segments, indexes)
    out = []
    for i, side in enumerate(sides):
        other = sides[1 - i]
        active, other_active = actives[i], actives[1 - i]
        missed_on = _event_timing(durations, other_active, other.fga - other.fgm)
        turned_on = _event_timing(durations, other_active, other.tov)
        own_missed_on = _event_timing(durations, active, side.fga - side.fgm)

        def spread(
            timing: np.ndarray, stat: str, intensity=side.intensity
        ) -> np.ndarray:
            shares = _interval_shares(active, intensity[stat])
            return np.einsum("sk,skp->sp", timing, shares)

        updated = dict(
            side.weights,
            DREB=spread(missed_on, "DREB"),
            BLK=spread(missed_on, "BLK"),
            STL=spread(turned_on, "STL"),
            OREB=spread(own_missed_on, "OREB"),
        )
        if side.ast is None:
            updated["AST"] = _assist_pairs(durations, active, side.intensity["AST"])
        out.append(replace(side, weights=updated))
    return out
