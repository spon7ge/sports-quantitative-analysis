"""Five-player schedules that realize a minute vector."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from role_context import OT_MIN, REG_MIN

from game_rotations.checks import (
    _as_float,
    _binary,
    _frame,
    _number,
    _one_game,
    _one_team,
    _player_key,
    _required,
)
from game_rotations.constants import _SCRIPT_INPUT, _TOL
from game_rotations.minutes import constrain, scale_mu
from game_rotations.model import _recent_mean
from game_rotations.roster import (
    _availability_matrix,
    _available_flags,
    _roster_keys,
)
from game_rotations.script import _period_lengths, _script_arrays

def segments_from_minutes(
    role: pd.DataFrame,
    environment: pd.DataFrame,
    minute_draws,
    *,
    availability=None,
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
) -> list[list[dict]]:
    """Layer 3 schedules from existing Layer 2 draws; no new minute model.

    Returns one schedule per environment row, preserving row order and the
    supplied player minutes. The environment must use this roster's side.
    """
    _frame(role, "roster")
    _frame(environment, "environment")
    _required(environment, _SCRIPT_INPUT)
    _one_game(environment)
    _roster_keys(role)
    _one_team(role)

    if "is_home" in role and "is_home" in environment:
        side = _binary(role["is_home"], "roster is_home")
        env_side = _binary(environment["is_home"], "environment is_home")

        if len(set(side)) != 1 or len(set(env_side)) != 1 or side[0] != env_side[0]:
            raise ValueError("environment must use the roster's perspective")

    counts, _, margin, _, cap = _script_arrays(
        len(environment),
        _as_float(environment["n_ot"]),
        _as_float(environment["team_margin"]),
        5 * _as_float(environment["game_minutes"]),
        _as_float(environment["game_minutes"]),
        regulation_minutes=regulation_minutes,
        ot_minutes=ot_minutes,
    )

    active = _availability_matrix(role, availability, len(environment))

    if np.any(active.sum(axis=1) < 5):
        raise ValueError("every scenario needs at least five available players")

    targets = np.asarray(_target_array(role, environment, minute_draws), dtype=float)
    _validate_targets(targets, len(environment), len(role), cap, active)

    result = []

    for i in range(len(environment)):
        scenario = role.copy()
        scenario["available"] = active[i]

        result.append(
            build_rotation_segments(
                scenario,
                game_minutes=cap[i],
                n_ot=counts[i],
                team_margin=margin[i],
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
                target_minutes=targets[i],
            )
        )

    return result


def _target_array(role, environment, target):
    if isinstance(target, pd.DataFrame):
        if not target.index.equals(environment.index):
            raise ValueError("target minute rows must retain the environment index")

        _required(role, ["player_id"])
        columns = [_player_key(value) for value in target.columns]

        if columns != [_player_key(value) for value in role["player_id"]]:
            raise ValueError("target minute columns must match roster player order")

        return target.to_numpy(dtype=float)

    return target

def build_rotation_segments(
    role: pd.DataFrame,
    *,
    game_minutes: float,
    n_ot: float,
    team_margin: float,
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
    target_minutes=None,
) -> list[dict]:
    """Realize a feasible minute vector as five-player segments exactly.

    Without target_minutes, use the player's recent minutes as deterministic
    weights. The final margin does not supply an observed score trajectory.
    """
    _frame(role, "roster")
    _one_team(role)
    _roster_keys(role)
    _number(team_margin, "team_margin")

    periods = _period_lengths(
        game_minutes,
        n_ot,
        regulation_minutes,
        ot_minutes,
    )
    active = _available_flags(role)

    if active.sum() < 5:
        raise ValueError("a rotation needs at least five available players")

    duration = float(game_minutes)

    if target_minutes is None:
        weights = _recent_mean(role, 18.0)[None, :]

        target = constrain(
            scale_mu(
                weights,
                [5 * duration],
                availability=active[None, :],
            ),
            np.ones_like(weights),
            [5 * duration],
            [duration],
            availability=active[None, :],
        )[0]
    else:
        target = np.asarray(target_minutes, dtype=float)

    _validate_targets(
        target[None, :] if target.ndim == 1 else target,
        1,
        len(role),
        [duration],
        active[None, :],
    )

    target = target.reshape(-1)
    players = role["player_id"].tolist()

    if "is_starter" in role:
        starters = _binary(role["is_starter"], "is_starter") & active & (target > 0)
    else:
        starters = np.zeros(len(role), dtype=bool)

    order = sorted(
        range(len(role)),
        key=lambda i: (not starters[i], -target[i], _player_key(players[i])),
    )
    opening = [i for i in order if active[i] and target[i] > 0][:5]
    other = [i for i in range(len(role)) if i not in opening]

    start_duration = min(
        periods[0][1] / 4,
        min(target[i] for i in opening),
        duration - max((target[i] for i in other), default=0.0),
    )

    global_segments = []
    remaining = target.copy()
    offset = max(0.0, start_duration)

    if offset > 1e-10:
        global_segments.append((0.0, offset, opening))
        remaining[opening] -= offset
    else:
        offset = 0.0

    horizon = duration - offset

    if horizon > 1e-10:
        lanes = [[] for _ in range(5)]
        cursor, lane = 0.0, 0

        for player in order:
            length = max(remaining[player], 0.0)

            while length > 1e-10:
                if lane >= 5:
                    if length <= _TOL:
                        break
                    raise ValueError("cannot pack the supplied minute totals")

                start = cursor
                piece = min(length, horizon - start)

                if piece <= 1e-10:
                    lane += 1
                    cursor = 0.0
                    continue

                end = min(start + piece, horizon)
                lanes[lane].append((start, end, player))
                cursor = end
                length -= piece

                if horizon - cursor <= 1e-10:
                    lane += 1
                    cursor = 0.0

        boundaries = sorted(
            {
                0.0,
                horizon,
                *(v for lane in lanes for a, b, _ in lane for v in (a, b)),
            }
        )

        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            if right - left <= 1e-10:
                continue

            middle = (left + right) / 2
            five = [
                player
                for lane in lanes
                for a, b, player in lane
                if a <= middle < b
            ]

            if len(five) != 5 or len(set(five)) != 5:
                raise ValueError("supplied minute totals did not yield a feasible five")

            global_segments.append((offset + left, offset + right, five))

    segments = []
    period_offset = 0.0

    for period, length in periods:
        for start, end, five in global_segments:
            left = max(start, period_offset)
            right = min(end, period_offset + length)

            if right - left > 1e-10:
                segments.append(
                    {
                        "period": period,
                        "start_min": left - period_offset,
                        "end_min": right - period_offset,
                        "players": [players[i] for i in five],
                    }
                )

        period_offset += length

    issues = _segment_issues(
        segments,
        role,
        game_minutes=duration,
        n_ot=n_ot,
        regulation_minutes=regulation_minutes,
        ot_minutes=ot_minutes,
    )

    if issues:
        raise ValueError(issues[0])

    if not np.allclose(_exposure(segments, players), target, rtol=0, atol=_TOL):
        raise ValueError("segments changed the supplied player minutes")

    return segments


def invalid_segment_count(
    segments,
    role: pd.DataFrame,
    *,
    game_minutes: float,
    n_ot: float,
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
) -> int:
    return len(
        _segment_issues(
            segments,
            role,
            game_minutes=game_minutes,
            n_ot=n_ot,
            regulation_minutes=regulation_minutes,
            ot_minutes=ot_minutes,
        )
    )


def _segment_minutes(
    roster,
    *,
    n,
    n_ot,
    margin,
    cap,
    segments,
    regulation_minutes,
    ot_minutes,
    availability=None,
    target_minutes=None,
):
    schedules = _broadcast_schedules(segments, n)
    active = _availability_matrix(roster, availability, n)

    if target_minutes is not None:
        targets = np.asarray(target_minutes, dtype=float)
        _validate_targets(targets, n, len(roster), cap, active)
    else:
        targets = None

    result = np.empty((n, len(roster)))

    for i in range(n):
        scenario = roster.copy()
        scenario["available"] = active[i]
        schedule = schedules[i]

        if schedule is None:
            schedule = build_rotation_segments(
                scenario,
                game_minutes=cap[i],
                n_ot=n_ot[i],
                team_margin=margin[i],
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
                target_minutes=None if targets is None else targets[i],
            )

        issues = _segment_issues(
            schedule,
            scenario,
            game_minutes=cap[i],
            n_ot=n_ot[i],
            regulation_minutes=regulation_minutes,
            ot_minutes=ot_minutes,
        )

        if issues:
            raise ValueError(f"simulation {i}: {issues[0]}")

        result[i] = _exposure(schedule, roster["player_id"].tolist())

    _validate_targets(result, n, len(roster), cap, active)
    return result


def _broadcast_schedules(segments, n):
    if segments is None:
        return [None] * n

    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("segments must be a nonempty schedule or list of schedules")

    if isinstance(segments[0], (list, tuple)):
        if len(segments) != n:
            raise ValueError("one rotation schedule is required per simulation")
        return list(segments)

    return [segments] * n


def _segment_issues(
    segments,
    role,
    *,
    game_minutes,
    n_ot,
    regulation_minutes,
    ot_minutes,
):
    _frame(role, "roster")

    periods = dict(
        _period_lengths(
            game_minutes,
            n_ot,
            regulation_minutes,
            ot_minutes,
        )
    )
    roster_keys, available_keys = _roster_keys(role)

    if not isinstance(segments, (list, tuple)):
        raise ValueError("segments must be a list of rotation segments")

    issues, normalized = [], []

    for segment in segments:
        item, issue = _inspect_segment(
            segment,
            roster_keys,
            available_keys,
            periods,
        )

        if issue is None:
            normalized.append(item)
        else:
            issues.append(issue)

    coverage = _coverage_issue(normalized, periods)

    if coverage is not None:
        issues.append(coverage)

    if normalized:
        exposure = _exposure(normalized, role["player_id"].tolist())

        if abs(exposure.sum() - 5 * game_minutes) > _TOL:
            issues.append("segment durations do not conserve the team minute budget")

        if (exposure > game_minutes + _TOL).any():
            issues.append("a player exceeds game duration")

    return issues


def _inspect_segment(segment, roster_keys, available_keys, periods):
    if not isinstance(segment, Mapping):
        return None, "segment must be a mapping"

    try:
        period_value = float(segment["period"])

        if not np.isfinite(period_value) or period_value != round(period_value):
            return None, "period must be a finite integer"

        period = int(period_value)

        if period not in periods:
            return None, "period is outside the script"

        start = float(segment["start_min"])
        end = float(segment["end_min"])

    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "segment has invalid period or time fields"

    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None, "duration must be finite and positive"

    if start < 0 or end > periods[period]:
        return None, "segment is outside its period"

    players = segment.get("players")

    if not isinstance(players, (list, tuple)) or len(players) != 5:
        return None, "a rotation segment needs five players"

    try:
        keys = [_player_key(player) for player in players]
    except ValueError:
        return None, "invalid player identifier"

    if len(set(keys)) != 5:
        return None, "players in a segment must be distinct"

    if any(key not in roster_keys for key in keys):
        return None, "player is not on the roster"

    if any(key not in available_keys for key in keys):
        return None, "unavailable player takes a segment"

    return {
        "period": period,
        "start_min": start,
        "end_min": end,
        "players": list(players),
    }, None


def _coverage_issue(segments, periods):
    for period, length in periods.items():
        spans = sorted(
            (s for s in segments if s["period"] == period),
            key=lambda s: (s["start_min"], s["end_min"]),
        )
        cursor = 0.0

        for segment in spans:
            delta = segment["start_min"] - cursor

            if abs(delta) > 1e-8:
                return f"period {period} has a {'gap' if delta > 0 else 'overlap'}"

            cursor = segment["end_min"]

        if abs(cursor - length) > 1e-8:
            return f"segments do not cover period {period}"

    return None


def _exposure(segments, player_ids):
    lookup = {_player_key(player): i for i, player in enumerate(player_ids)}
    minutes = np.zeros(len(player_ids))

    for segment in segments:
        duration = segment["end_min"] - segment["start_min"]

        for player in segment["players"]:
            minutes[lookup[_player_key(player)]] += duration

    return minutes


def _validate_targets(target, n, k, cap, active):
    target = np.asarray(target, dtype=float)
    cap = np.asarray(cap, dtype=float)

    if target.shape != (n, k) or not np.isfinite(target).all():
        raise ValueError(
            "target_minutes must be a finite (n_simulations, n_players) matrix"
        )

    if (target < 0).any() or (target > cap[:, None] + _TOL).any():
        raise ValueError("target minutes exceed a player's game-duration cap")

    if np.any(target[~active] != 0):
        raise ValueError("unavailable players must have exactly zero target minutes")

    if not np.allclose(target.sum(axis=1), 5 * cap, rtol=0, atol=_TOL):
        raise ValueError("target minutes must sum to five times game duration")
