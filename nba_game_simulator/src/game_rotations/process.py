"""Layer 2 and Layer 3 as one within-game process.

A player's minutes are the time he survives on the floor. Fouls, the
margin path, stint length, timeouts, and a rare injury or ejection all
pull him off that clock. Points that accrue only while he is on it then
move with his minutes. A whole-game template is one sample. Each
player-tick is a sample for these hazards.

The margin series is an input. Layer 1's final margin is the last tick,
and closing lineups read the last six minutes rather than that tick alone.
This module does not invent a path from the final margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from game_rotations.checks import _binary, _player_key
from role_context import OT_MIN, REG_MIN

# Six personal fouls is the league disqualification. Spread across 48
# minutes, that is the pace a player can keep without sitting.
FOUL_OUT = 6
FOUL_BUDGET = 6.0
# Two points of cushion so a starter does not re-enter on the same
# crossing that just sent him to the bench.
BLOWOUT_HYSTERESIS = 2.0
# Without a short rest, the starter is the first eligible name and the
# substitution would not change the floor.
REST_MINUTES = 2.0
PERIOD_MIN = REG_MIN / 4
# Fixed scale so the threshold's level and slope are identified.
# A free steepness trades off one-for-one with both.
FIT_STEEPNESS = 2.0
# A handful of ejections should not become a large probability.
EXIT_PRIOR_HITS = 0.5
EXIT_PRIOR_GAMES = 20.0
RETURN_PRIOR_HITS = 1.0
RETURN_PRIOR_CHANCES = 2.0
# On the standardized design. A timeout that always coincides with the sub
# would otherwise push its slope without a bound.
_RIDGE = 1e-3
_REASONS = frozenset(
    {"rotation", "foul_trouble", "blowout", "injury", "ejection", "censored"}
)
_TICK_COLUMNS = (
    "game_id",
    "team_id",
    "player_id",
    "stint_minutes",
    "fatigue_minutes",
    "timeout",
    "rotation_sub",
    "fouls",
    "elapsed_minutes",
    "foul_trouble_exit",
    "margin",
    "minutes_left",
    "is_starter",
    "blowout_exit",
    "injury_exit",
    "ejection_exit",
    "blowout_benched",
    "inside_threshold",
    "returned",
)
_STINT_COLUMNS = (
    "game_id",
    "team_id",
    "player_id",
    "is_starter",
    "start_elapsed",
    "end_elapsed",
    "game_minutes",
    "margin",
    "fouls",
    "timeout",
    "reason",
)


@dataclass(frozen=True)
class StintHazard:
    """Logit of a rotation sub on one tick the player is still at risk.

    The four numbers are the intercept and the slopes on minutes into the
    current stint, minutes already played in the game, and a timeout tick.
    """

    intercept: float
    stint_minutes: float
    fatigue: float
    timeout: float


@dataclass(frozen=True)
class FoulProcess:
    """Logit of sitting, given how far the foul count is ahead of pace."""

    intercept: float
    ahead: float


@dataclass(frozen=True)
class BlowoutRule:
    """Starters leave once the lead clears cushion + per_minute * minutes left.

    exit_steepness is the logit slope on that gap. return_probability is
    the chance, per tick, that a benched starter comes back once the lead
    is back inside the threshold by the hysteresis.
    """

    cushion: float
    per_minute: float
    exit_steepness: float
    return_probability: float


@dataclass(frozen=True)
class ExitMixture:
    """One latent per player-game, not a residual in the minute equation.

    injury_rate and ejection_rate are probabilities. An exit time is then
    uniform between exit_lo and exit_hi as fractions of the regulation game.
    """

    injury_rate: float
    ejection_rate: float
    exit_lo: float
    exit_hi: float


@dataclass(frozen=True)
class ClosingRule:
    """Closers follow the share of the late window inside close_margin.

    late_minutes is six because that is the window coaches use to choose
    a closing five. The final margin is only the last tick of that window.
    """

    late_minutes: float = 6.0
    close_margin: float = 8.0
    closeness_share: float = 0.5


@dataclass(frozen=True)
class RotationProcess:
    stint: StintHazard
    foul: FoulProcess
    blowout: BlowoutRule
    exits: ExitMixture
    closing: ClosingRule


@dataclass(frozen=True)
class RotationDraw:
    """One row per simulation, players in roster order.

    late_closeness is a property of the supplied path, so it is shared.
    """

    minutes: np.ndarray
    fouls: np.ndarray
    points: np.ndarray
    exit_reason: tuple[tuple[str, ...], ...]
    segments: tuple[tuple[dict, ...], ...]
    late_closeness: float
    n_ticks: int


def ahead_of_foul_pace(fouls, elapsed_minutes) -> np.ndarray:
    """Fouls above the straight line from zero to a foul-out at the buzzer."""

    count = np.asarray(fouls, dtype=float)
    elapsed = np.asarray(elapsed_minutes, dtype=float)
    if not np.isfinite(count).all() or not np.isfinite(elapsed).all():
        raise ValueError("fouls and elapsed minutes must be finite")
    if (elapsed < 0).any():
        raise ValueError("elapsed minutes must be nonnegative")
    return count - FOUL_BUDGET * elapsed / REG_MIN


def blowout_threshold(minutes_left, rule: BlowoutRule) -> np.ndarray:
    """Lead, in points, at which starters become expendable."""

    _check_blowout(rule)
    left = np.asarray(minutes_left, dtype=float)
    if not np.isfinite(left).all() or (left < 0).any():
        raise ValueError("minutes_left must be finite and nonnegative")
    return rule.cushion + rule.per_minute * left


def late_closeness(margin_path, dt: float, rule: ClosingRule) -> float:
    """Share of the late window whose absolute margin is still close.

    A zero-length window has no blowout ticks in it, so the share is 1.
    The simulator still requires a positive window before it changes closers.
    """

    path = _path(margin_path)
    step = _dt(dt)
    _check_closing(rule)
    if rule.late_minutes == 0:
        return 1.0
    elapsed = np.arange(len(path)) * step
    late = elapsed >= (len(path) * step - rule.late_minutes)
    if not late.any():
        raise ValueError("late window is longer than the margin path")
    return float(np.mean(np.abs(path[late]) <= rule.close_margin))


def expand_stints(stints: pd.DataFrame, *, dt: float = 1.0) -> pd.DataFrame:
    """One player-tick per minute on the floor.

    The sub reason is written on the last tick only. Earlier ticks are
    survivals. Margin and fouls are held at the stint's ending values:
    a path that moves inside the stint has to be passed as ticks instead.
    """

    frame = _table(stints, _STINT_COLUMNS, "stints")
    step = _dt(dt)
    rows = []
    for stint in frame.itertuples(index=False):
        reason = str(stint.reason)
        if reason not in _REASONS:
            raise ValueError(f"unknown stint reason {stint.reason!r}")
        start = float(stint.start_elapsed)
        end = float(stint.end_elapsed)
        game_minutes = float(stint.game_minutes)
        if not all(np.isfinite(value) for value in (start, end, game_minutes)):
            raise ValueError("stint times must be finite")
        if start < 0 or end <= start or end > game_minutes + 1e-8:
            raise ValueError("stint must sit inside the game")
        if abs((end - start) / step - round((end - start) / step)) > 1e-8:
            raise ValueError("stint length must be a whole number of ticks")
        cursor = start
        while cursor < end - 1e-8:
            last = cursor + step >= end - 1e-8
            rows.append(
                {
                    "game_id": stint.game_id,
                    "team_id": stint.team_id,
                    "player_id": stint.player_id,
                    "stint_minutes": cursor - start,
                    "fatigue_minutes": cursor,
                    "timeout": float(stint.timeout) if last else 0.0,
                    "rotation_sub": float(last and reason == "rotation"),
                    "fouls": float(stint.fouls),
                    "elapsed_minutes": cursor,
                    "foul_trouble_exit": float(last and reason == "foul_trouble"),
                    "margin": float(stint.margin),
                    "minutes_left": game_minutes - cursor,
                    "is_starter": stint.is_starter,
                    "blowout_exit": float(last and reason == "blowout"),
                    "injury_exit": float(last and reason == "injury"),
                    "ejection_exit": float(last and reason == "ejection"),
                    "blowout_benched": 0.0,
                    "inside_threshold": 0.0,
                    "returned": 0.0,
                }
            )
            cursor += step
    if not rows:
        raise ValueError("stints did not contain any time on the floor")
    return pd.DataFrame(rows)


def fit_rotation_process(ticks: pd.DataFrame) -> tuple[RotationProcess, dict]:
    """Fit hazards on player-ticks. Closing thresholds stay structural.

    The late window and the 8-point band are the coaching rule, not a
    coefficient. Estimating them from the same ticks would turn the
    conditioner into another fitted template.
    """

    frame = _table(ticks, _TICK_COLUMNS, "player ticks")
    _binary(frame["timeout"], "timeout")
    rotation_sub = _binary(frame["rotation_sub"], "rotation_sub")
    foul_exit = _binary(frame["foul_trouble_exit"], "foul_trouble_exit")
    starter = _binary(frame["is_starter"], "is_starter")
    blowout_exit = _binary(frame["blowout_exit"], "blowout_exit")
    injury = _binary(frame["injury_exit"], "injury_exit")
    ejection = _binary(frame["ejection_exit"], "ejection_exit")
    benched = _binary(frame["blowout_benched"], "blowout_benched")
    inside = _binary(frame["inside_threshold"], "inside_threshold")
    returned = _binary(frame["returned"], "returned")
    for name in (
        "stint_minutes",
        "fatigue_minutes",
        "fouls",
        "elapsed_minutes",
        "margin",
        "minutes_left",
    ):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        frame[name] = values

    stint_beta = _logit_mle(
        np.column_stack(
            [
                np.ones(len(frame)),
                frame["stint_minutes"],
                frame["fatigue_minutes"],
                frame["timeout"].to_numpy(dtype=float),
            ]
        ),
        rotation_sub.astype(float),
    )
    ahead = ahead_of_foul_pace(frame["fouls"], frame["elapsed_minutes"])
    foul_beta = _logit_mle(
        np.column_stack([np.ones(len(frame)), ahead]),
        foul_exit.astype(float),
    )
    starter_mask = np.asarray(starter, dtype=bool)
    cushion, per_minute = _fit_blowout(
        frame["margin"].to_numpy(dtype=float)[starter_mask],
        frame["minutes_left"].to_numpy(dtype=float)[starter_mask],
        blowout_exit[starter_mask].astype(float),
    )
    chances = benched & inside
    returns = float(returned[chances].sum()) if chances.any() else 0.0
    return_probability = (returns + RETURN_PRIOR_HITS) / (
        float(chances.sum()) + RETURN_PRIOR_CHANCES
    )
    exits = _fit_exits(frame, injury, ejection)
    process = RotationProcess(
        stint=StintHazard(*map(float, stint_beta)),
        foul=FoulProcess(*map(float, foul_beta)),
        blowout=BlowoutRule(cushion, per_minute, FIT_STEEPNESS, return_probability),
        exits=exits,
        closing=ClosingRule(),
    )
    _check_process(process)
    team_games = frame.groupby(["game_id", "team_id"], sort=False).ngroups
    diagnostics = {
        "n_player_ticks": int(len(frame)),
        "n_team_games": int(team_games),
        "ticks_per_team_game": float(len(frame) / team_games),
    }
    return process, diagnostics


def minutes_from_process(process, role, environment, rng):
    """Minutes and segments for one roster, one path per Layer 1 row.

    Unavailable players are left off the floor. The returned minute frame
    keeps every player on ``role``, in that order, so Layer 4 can align it.
    """

    paths = environment.attrs.get("margin_path")
    if not isinstance(paths, (list, tuple)) or len(paths) != len(environment):
        raise ValueError("environment needs one margin path per simulation")
    if "player_id" not in role.columns or "is_starter" not in role.columns:
        raise ValueError("role needs player_id and is_starter")
    available = (
        role["available"].to_numpy() == 1
        if "available" in role.columns
        else np.ones(len(role), dtype=bool)
    )
    active = role.loc[available].reset_index(drop=True)
    if len(active) < 5:
        raise ValueError("rotation process needs five available players")
    roster = pd.DataFrame(
        {
            "player_id": active["player_id"].tolist(),
            "is_starter": active["is_starter"].to_numpy(),
        }
    )
    if {"pf_mean_10", "min_mean_10"} <= set(active.columns):
        played = pd.to_numeric(active["min_mean_10"], errors="coerce").to_numpy(
            dtype=float
        )
        fouls = pd.to_numeric(active["pf_mean_10"], errors="coerce").to_numpy(
            dtype=float
        )
        played = np.where(np.isfinite(played), played, 0.0)
        fouls = np.where(np.isfinite(fouls), fouls, 0.0)
        roster["foul_rate"] = np.clip(fouls / np.maximum(played, 1.0), 0.0, 0.5)
    total = np.zeros((len(environment), len(role)))
    schedules = []
    positions = {
        player: index for index, player in enumerate(role["player_id"].tolist())
    }
    columns = [positions[player] for player in roster["player_id"]]
    for row, path in enumerate(paths):
        draw = simulate_rotations(process, roster, path, rng)
        total[row, columns] = draw.minutes[0]
        schedules.append(draw.segments[0])
    minutes = pd.DataFrame(
        total, index=environment.index, columns=list(role["player_id"])
    )
    return minutes, schedules


def simulate_rotations(
    process: RotationProcess,
    roster: pd.DataFrame,
    margin_path,
    rng: np.random.Generator,
    *,
    n: int = 1,
    timeouts=None,
    dt: float = 1.0,
) -> RotationDraw:
    """Draw minutes and segments for one team, conditional on the path.

    n repeats share the path and differ only in the hazards. Pass a new
    path for a different script. Positive margin means this team leads.
    """

    if not isinstance(process, RotationProcess):
        raise ValueError("process must be a RotationProcess")
    _check_process(process)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise ValueError("n must be a positive integer")
    step = _dt(dt)
    path = _path(margin_path)
    length = len(path) * step
    _check_duration(length)
    marks = _timeouts(timeouts)
    players, starter, foul_rate, scoring, injury_rate, ejection_rate = _roster(roster)
    if len(players) < 5:
        raise ValueError("every rotation needs at least five players")
    if injury_rate is None:
        injury_rate = np.full(len(players), process.exits.injury_rate)
    if ejection_rate is None:
        ejection_rate = np.full(len(players), process.exits.ejection_rate)
    if np.any(injury_rate + ejection_rate > 1):
        raise ValueError("injury and ejection rates must sum to at most 1")
    closeness = late_closeness(path, step, process.closing)
    close_late = (
        process.closing.late_minutes > 0
        and closeness >= process.closing.closeness_share
    )
    minutes = np.zeros((int(n), len(players)))
    fouls = np.zeros((int(n), len(players)))
    points = np.zeros((int(n), len(players)))
    reasons = []
    schedules = []
    for _ in range(int(n)):
        played, foul_count, scored, reason, schedule = _one_game(
            process,
            players,
            starter,
            foul_rate,
            scoring,
            injury_rate,
            ejection_rate,
            path,
            marks,
            step,
            close_late,
            rng,
        )
        minutes[len(reasons)] = played
        fouls[len(reasons)] = foul_count
        points[len(reasons)] = scored
        reasons.append(tuple(reason))
        schedules.append(tuple(schedule))
    return RotationDraw(
        minutes=minutes,
        fouls=fouls,
        points=points,
        exit_reason=tuple(reasons),
        segments=tuple(schedules),
        late_closeness=closeness,
        n_ticks=len(path),
    )


def _one_game(
    process,
    players,
    starter,
    foul_rate,
    scoring,
    injury_rate,
    ejection_rate,
    path,
    timeouts,
    dt,
    close_late,
    rng,
):
    count = len(players)
    length = len(path) * dt
    kind, exit_at = _draw_exits(
        injury_rate,
        ejection_rate,
        process.exits,
        length,
        rng,
    )
    gone = np.zeros(count, dtype=bool)
    foul_sit = np.zeros(count, dtype=bool)
    blowout_out = np.zeros(count, dtype=bool)
    rest_until = np.full(count, -1.0)
    stint = np.zeros(count)
    fatigue = np.zeros(count)
    fouls = np.zeros(count)
    minutes = np.zeros(count)
    points = np.zeros(count)
    reason = [""] * count
    on = _compose(~gone, starter)
    pieces = []
    for tick, margin in enumerate(path):
        elapsed = tick * dt
        left = length - elapsed
        due = ~gone & (elapsed >= exit_at)
        _mark(reason, _take_gone(gone, due), kind)
        on &= ~gone
        pace = FOUL_BUDGET * elapsed / REG_MIN
        foul_sit &= fouls > pace
        closing_now = close_late and elapsed >= length - process.closing.late_minutes
        if closing_now:
            blowout_out[starter & ~gone] = False
            rest_until[starter & ~gone & ~foul_sit] = -1.0
        uniforms = rng.random((4, count))
        ahead = fouls - pace
        # Zero fouls is not foul trouble. The hazard starts only once the
        # count is ahead of the six-foul pace.
        leave_foul = (
            on
            & (ahead > 0)
            & (
                uniforms[0]
                < _sigmoid(process.foul.intercept + process.foul.ahead * ahead)
            )
        )
        foul_sit |= leave_foul & (fouls < FOUL_OUT)
        threshold = process.blowout.cushion + process.blowout.per_minute * left
        gap = abs(float(margin)) - threshold
        leave_blow = (
            on
            & starter
            & (uniforms[1] < _sigmoid(process.blowout.exit_steepness * gap))
        )
        if closing_now:
            leave_blow = np.zeros(count, dtype=bool)
        blowout_out |= leave_blow
        timeout = float(np.any((timeouts >= elapsed) & (timeouts < elapsed + dt)))
        eta = (
            process.stint.intercept
            + process.stint.stint_minutes * stint
            + process.stint.fatigue * fatigue
            + process.stint.timeout * timeout
        )
        protected = starter if closing_now else np.zeros(count, dtype=bool)
        # A rotation sub is voluntary. Decline it when the bench cannot
        # keep five players eligible; a foul or a blowout is not optional.
        wants_rest = np.flatnonzero(
            on & ~protected & ~leave_foul & ~leave_blow & (uniforms[2] < _sigmoid(eta))
        )
        for index in wants_rest:
            rest_until[index] = elapsed + dt + REST_MINUTES
            still_eligible = (
                ~gone & ~foul_sit & ~blowout_out & (elapsed + 1e-9 >= rest_until)
            )
            if int(still_eligible.sum()) < 5:
                rest_until[index] = -1.0
        inside = abs(float(margin)) + BLOWOUT_HYSTERESIS < threshold
        come_back = (
            blowout_out & inside & (uniforms[3] < process.blowout.return_probability)
        )
        blowout_out &= ~come_back
        _release_until_five(
            gone, foul_sit, blowout_out, rest_until, starter, elapsed
        )
        eligible = ~gone & ~foul_sit & ~blowout_out & (elapsed + 1e-9 >= rest_until)
        on = _compose(eligible, starter)
        minutes[on] += dt
        fatigue[on] += dt
        stint = np.where(on, stint + dt, 0.0)
        fouls += rng.poisson(foul_rate * dt) * on
        points += rng.poisson(scoring * dt) * on
        fouled_out = (fouls >= FOUL_OUT) & ~gone
        _mark(reason, _take_gone(gone, fouled_out), np.full(count, "foul_out"))
        pieces.append(_piece(tick, dt, on, players))
    if abs(minutes.sum() - 5 * length) > 1e-6:
        raise ValueError("rotation failed to keep five players on the clock")
    return minutes, fouls, points, reason, _collapse(pieces)


def _draw_exits(injury_rate, ejection_rate, exits, length, rng):
    draw = rng.random(len(injury_rate))
    clock = rng.random(len(injury_rate))
    ejected = draw < ejection_rate
    injured = ~ejected & (draw < ejection_rate + injury_rate)
    fraction = exits.exit_lo + (exits.exit_hi - exits.exit_lo) * clock
    # The stored fraction is of regulation. An overtime game can run past it,
    # and a fraction of 1 still means the end of regulation.
    exit_at = np.full(len(injury_rate), np.inf)
    exit_at[injured | ejected] = np.minimum(fraction[injured | ejected], 1.0) * REG_MIN
    exit_at = np.minimum(exit_at, length)
    kind = np.full(len(injury_rate), "", dtype=object)
    kind[injured] = "injury"
    kind[ejected] = "ejection"
    return kind, exit_at


def _take_gone(gone, newly):
    """Mark players gone while five others can still take the floor."""

    taken = np.zeros(len(gone), dtype=bool)
    for index in np.flatnonzero(np.asarray(newly, dtype=bool) & ~gone):
        if int((~gone).sum()) - 1 < 5:
            break
        gone[index] = True
        taken[index] = True
    return taken


def _release_until_five(gone, foul_sit, blowout_out, rest_until, starter, elapsed):
    """Bring sitters back when a blowout, foul, or rest would leave four.

    An injury, an ejection, and a sixth foul stay on ``gone``. Rest goes
    first, then a blowout, then foul trouble. A non-starter returns before
    a starter, so the blowout benching holds when the bench is deep enough.
    """

    def eligible_count():
        available = (
            ~gone
            & ~foul_sit
            & ~blowout_out
            & (elapsed + 1e-9 >= rest_until)
        )
        return int(available.sum())

    def ordered(mask):
        index = np.flatnonzero(mask)
        rank = starter[index].astype(np.int8)
        return index[np.argsort(rank, kind="stable")]

    for index in ordered(~gone & (rest_until > elapsed)):
        if eligible_count() >= 5:
            return
        rest_until[index] = -1.0
    for index in ordered(blowout_out & ~gone):
        if eligible_count() >= 5:
            return
        blowout_out[index] = False
    for index in ordered(foul_sit & ~gone):
        if eligible_count() >= 5:
            return
        foul_sit[index] = False


def _compose(eligible, starter):
    chosen = np.zeros(len(eligible), dtype=bool)
    starters = np.flatnonzero(eligible & starter)
    chosen[starters[:5]] = True
    holes = 5 - int(chosen.sum())
    if holes:
        others = np.flatnonzero(eligible & ~chosen)
        if len(others) < holes:
            raise ValueError("fewer than five eligible players")
        chosen[others[:holes]] = True
    if int(chosen.sum()) != 5:
        raise ValueError("fewer than five eligible players")
    return chosen


def _piece(tick, dt, on, players):
    elapsed = tick * dt
    if elapsed < REG_MIN:
        period = int(elapsed // PERIOD_MIN) + 1
        origin = (period - 1) * PERIOD_MIN
    else:
        overtime = int((elapsed - REG_MIN) // OT_MIN)
        period = 5 + overtime
        origin = REG_MIN + overtime * OT_MIN
    names = [players[index] for index in np.flatnonzero(on)]
    return {
        "period": period,
        "start_min": elapsed - origin,
        "end_min": elapsed + dt - origin,
        "players": names,
    }


def _collapse(pieces):
    merged = []
    for piece in pieces:
        previous = merged[-1] if merged else None
        if (
            previous
            and previous["period"] == piece["period"]
            and previous["players"] == piece["players"]
            and abs(previous["end_min"] - piece["start_min"]) < 1e-8
        ):
            previous["end_min"] = piece["end_min"]
        else:
            merged.append(dict(piece))
    return merged


def _mark(reason, mask, labels):
    for index in np.flatnonzero(mask):
        if reason[index] == "":
            reason[index] = str(labels[index])


def _fit_blowout(margin, minutes_left, labels):
    if len(labels) < 20:
        raise ValueError("blowout threshold needs at least twenty starter ticks")
    if not np.isfinite(margin).all() or not np.isfinite(minutes_left).all():
        raise ValueError("blowout covariates must be finite")
    best = None
    absolute = np.abs(margin)
    for cushion in np.arange(4.0, 26.0, 2.0):
        for per_minute in np.arange(0.2, 1.21, 0.1):
            gap = absolute - cushion - per_minute * minutes_left
            probability = np.clip(_sigmoid(FIT_STEEPNESS * gap), 1e-6, 1 - 1e-6)
            nll = -np.sum(
                labels * np.log(probability)
                + (1.0 - labels) * np.log(1.0 - probability)
            )
            if best is None or nll < best[0]:
                best = (float(nll), float(cushion), float(per_minute))
    return best[1], best[2]


def _fit_exits(frame, injury, ejection) -> ExitMixture:
    grouped = (
        pd.DataFrame(
            {
                "game_id": frame["game_id"].to_numpy(),
                "team_id": frame["team_id"].to_numpy(),
                "player_id": frame["player_id"].to_numpy(),
                "injury_exit": injury,
                "ejection_exit": ejection,
                "elapsed_minutes": frame["elapsed_minutes"].to_numpy(),
            }
        )
        .groupby(["game_id", "team_id", "player_id"], sort=False)
        .agg(
            injury_exit=("injury_exit", "max"),
            ejection_exit=("ejection_exit", "max"),
        )
    )
    both = grouped["injury_exit"] & grouped["ejection_exit"]
    if bool(both.any()):
        raise ValueError("a player-game cannot be both an injury and an ejection")
    games = float(len(grouped))
    injury_rate = (float(grouped["injury_exit"].sum()) + EXIT_PRIOR_HITS) / (
        games + EXIT_PRIOR_GAMES
    )
    ejection_rate = (float(grouped["ejection_exit"].sum()) + EXIT_PRIOR_HITS) / (
        games + EXIT_PRIOR_GAMES
    )
    if injury_rate + ejection_rate > 1:
        scale = injury_rate + ejection_rate
        injury_rate /= scale
        ejection_rate /= scale
    exited = np.asarray(injury | ejection, dtype=bool)
    if int(exited.sum()) >= 5:
        fraction = np.clip(
            frame["elapsed_minutes"].to_numpy(dtype=float)[exited] / REG_MIN,
            0.0,
            1.0,
        )
        lo, hi = np.quantile(fraction, [0.1, 0.9])
    else:
        lo, hi = 0.15, 0.70
    return ExitMixture(float(injury_rate), float(ejection_rate), float(lo), float(hi))


def _logit_mle(features, labels) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("hazard features must align with labels")
    if len(labels) <= features.shape[1]:
        raise ValueError("need more player-ticks than hazard coefficients")
    if not np.isfinite(features).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("hazard features and labels must be finite 0/1 outcomes")
    # Center the slopes so a timeout column that matches the label does not
    # send the intercept and the slope off in opposite directions.
    design = np.asarray(features, dtype=float)
    center = np.zeros(design.shape[1])
    scale = np.ones(design.shape[1])
    if design.shape[1] > 1:
        center[1:] = design[:, 1:].mean(axis=0)
        scale[1:] = design[:, 1:].std(axis=0)
        scale[scale < 1e-8] = 1.0
        design = design.copy()
        design[:, 1:] = (features[:, 1:] - center[1:]) / scale[1:]
    beta = np.zeros(design.shape[1])
    for _ in range(80):
        eta = design @ beta
        probability = _sigmoid(eta)
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        working = eta + (labels - probability) / weight
        weighted = design.T * weight
        hessian = weighted @ design
        hessian.flat[:: design.shape[1] + 1] += _RIDGE
        updated = np.linalg.solve(hessian, weighted @ working)
        if float(np.max(np.abs(updated - beta))) < 1e-8:
            beta = updated
            break
        beta = updated
    else:
        raise ValueError("hazard fit did not converge")
    raw = beta.copy()
    raw[1:] = beta[1:] / scale[1:]
    raw[0] = beta[0] - float(np.dot(beta[1:], center[1:] / scale[1:]))
    return raw


def _roster(roster):
    if not isinstance(roster, pd.DataFrame) or roster.empty:
        raise ValueError("roster must be a non-empty DataFrame")
    missing = [name for name in ("player_id", "is_starter") if name not in roster]
    if missing:
        raise ValueError(f"roster needs {missing}")
    players = roster["player_id"].tolist()
    keys = [_player_key(player) for player in players]
    if len(set(keys)) != len(keys):
        raise ValueError("roster lists the same player twice")
    starter = _binary(roster["is_starter"], "is_starter")
    if int(starter.sum()) > 5:
        raise ValueError("a roster cannot list more than five starters")
    if len(roster) < 5:
        raise ValueError("every rotation needs at least five players")
    # A pregame scratch is not an in-game exit. Drop him from the roster.
    if "available" in roster:
        listed = _binary(roster["available"], "available")
        if not bool(listed.all()):
            raise ValueError("mark a pregame scratch by dropping him from the roster")
    foul_rate = _rate(roster, "foul_rate", 0.04, len(roster))
    scoring = _rate(roster, "points_per_minute", 0.0, len(roster))
    injury = _rate(roster, "injury_rate", None, len(roster))
    ejection = _rate(roster, "ejection_rate", None, len(roster))
    return players, starter, foul_rate, scoring, injury, ejection


def _rate(roster, name, default, n):
    if name not in roster:
        if default is None:
            return None
        return np.full(n, default)
    values = pd.to_numeric(roster[name], errors="coerce").to_numpy(dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    if name in {"injury_rate", "ejection_rate"} and (values > 1).any():
        raise ValueError(f"{name} must be between 0 and 1")
    return values


def _timeouts(timeouts):
    if timeouts is None:
        return np.array([], dtype=float)
    marks = np.asarray(timeouts, dtype=float).reshape(-1)
    if not np.isfinite(marks).all() or (marks < 0).any():
        raise ValueError("timeouts must be finite and nonnegative")
    return marks


def _path(margin_path):
    path = np.asarray(margin_path, dtype=float)
    if path.ndim != 1 or path.size == 0 or not np.isfinite(path).all():
        raise ValueError("margin path must be a finite one-dimensional series")
    return path


def _dt(dt):
    step = float(dt)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("dt must be a positive finite number")
    return step


def _check_duration(length):
    overtime = (length - REG_MIN) / OT_MIN
    if length + 1e-8 < REG_MIN or abs(overtime - round(overtime)) > 1e-8:
        raise ValueError(
            "margin path must cover 48 minutes plus whole overtime periods"
        )


def _check_process(process: RotationProcess):
    for name, value in (
        ("stint intercept", process.stint.intercept),
        ("stint slope", process.stint.stint_minutes),
        ("fatigue slope", process.stint.fatigue),
        ("timeout slope", process.stint.timeout),
        ("foul intercept", process.foul.intercept),
        ("foul slope", process.foul.ahead),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    _check_blowout(process.blowout)
    _check_exits(process.exits)
    _check_closing(process.closing)


def _check_blowout(rule: BlowoutRule):
    if not isinstance(rule, BlowoutRule):
        raise ValueError("blowout rule has the wrong type")
    if (
        not np.isfinite(rule.cushion)
        or not np.isfinite(rule.per_minute)
        or not np.isfinite(rule.exit_steepness)
        or rule.exit_steepness <= 0
        or not 0 <= rule.return_probability <= 1
    ):
        raise ValueError("blowout rule has an invalid threshold or return rate")


def _check_exits(exits: ExitMixture):
    if not isinstance(exits, ExitMixture):
        raise ValueError("exit mixture has the wrong type")
    if (
        not 0 <= exits.injury_rate <= 1
        or not 0 <= exits.ejection_rate <= 1
        or exits.injury_rate + exits.ejection_rate > 1
        or not 0 <= exits.exit_lo <= exits.exit_hi <= 1
    ):
        raise ValueError(
            "exit mixture rates must be probabilities that sum to at most 1"
        )


def _check_closing(rule: ClosingRule):
    if not isinstance(rule, ClosingRule):
        raise ValueError("closing rule has the wrong type")
    if (
        not np.isfinite(rule.late_minutes)
        or rule.late_minutes < 0
        or not np.isfinite(rule.close_margin)
        or rule.close_margin < 0
        or not 0 <= rule.closeness_share <= 1
    ):
        raise ValueError(
            "closing rule needs a nonnegative window and a share in [0, 1]"
        )


def _table(frame, columns, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} need column {missing[0]}")
    return frame.copy()


def _sigmoid(values):
    raw = np.asarray(values, dtype=float)
    flat = np.atleast_1d(raw)
    out = np.empty(flat.shape, dtype=float)
    positive = flat >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))
    exponential = np.exp(flat[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    if raw.shape == ():
        return float(out[0])
    return out.reshape(raw.shape)
