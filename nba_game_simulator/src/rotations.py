"""Minutes for one team inside an attached script, from each player's role.

A training row is a game already played. The minute window is shifted, so
it stops before that game, and ``minutes`` is what he played. A live row is
a game that has not been played. The same window is not shifted again, so
the latest completed game stays in. The script is a Layer 1 draw already
on the player's side: his team's margin, overtime, and game length. The
draw sums to five times that length, and no player plays more than that
length.

A rotation segment is five players on the floor for a positive duration.
Segment minutes are the sum of those durations, so the team minute budget
holds without a later rescale. They are not stored on the environment row
and they are not a Layer 4 production multiplier. A later random rotation
would still be one draw per environment row, conditional on that script and
on the role: posted start, minute history, availability, and the blowout
flag. It would not walk the live score, the foul count, or the game clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from role_context import (
    BLOWOUT_MARGIN,
    OPP_HISTORY,
    OT_MIN,
    PLAYER_HISTORY,
    REG_MIN,
    SCRIPT,
    SLATE_KNOWN,
    TEAM_HISTORY,
    _minutes,
    _reject_minute_conflicts,
    build_role_history,
    role_for_slate,
)
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MU_BINS = np.array([0.0, 8.0, 16.0, 24.0, 30.0, 99.0])
PREGAME = (*PLAYER_HISTORY, *TEAM_HISTORY, *OPP_HISTORY, *SLATE_KNOWN)
CONDITIONAL = (*PREGAME, *SCRIPT)
_INTERACTIONS = (
    "min_mean_x_abs_margin",
    "min_mean_x_blowout",
    "min_mean_x_n_ot",
    "starter_x_blowout",
)
_FEATURES = (*CONDITIONAL, *_INTERACTIONS)
_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_SIGMA_FLOOR = 0.05
_SCRIPT_INPUT = ("team_margin", "n_ot", "game_minutes")


def training_minutes(logs: pd.DataFrame) -> pd.DataFrame:
    """One player-game. The window excludes that game. ``minutes`` is the result."""
    history = build_role_history(logs)
    if "minutes" in history.columns:
        history = history.drop(columns=["minutes"])
    history["game_id"] = history["game_id"].astype(str)
    played = _played_minutes(logs)
    played["game_id"] = played["game_id"].astype(str)
    merged = history.merge(
        played,
        on=["player_id", "game_id", "team_id"],
        how="left",
        validate="one_to_one",
    )
    columns = [*history.columns, "minutes"]
    return merged.loc[:, columns].reset_index(drop=True)


def live_minutes(logs: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """One upcoming player. The latest completed game stays in the window."""
    return role_for_slate(logs, slate)


def fit_rotation(roles: pd.DataFrame) -> RotationSimulator:
    """Fit on shifted rows, score noise on the latest slice, then refit on all."""
    if not isinstance(roles, pd.DataFrame) or roles.empty:
        raise ValueError("roles must be a non-empty DataFrame")
    if "minutes" not in roles.columns:
        raise ValueError("training roles need minutes from the game being scored")
    missing = [column for column in CONDITIONAL if column not in roles.columns]
    if missing:
        raise ValueError(f"training roles need {missing}")
    usable = roles.dropna(subset=["minutes", "min_mean_10"]).copy()
    minutes = pd.to_numeric(usable["minutes"], errors="coerce")
    recent = pd.to_numeric(usable["min_mean_10"], errors="coerce")
    usable = usable.loc[np.isfinite(minutes) & np.isfinite(recent)].copy()
    if len(usable) < 2:
        raise ValueError("need at least 2 player-games with a prior minute window")
    if "game_date" in usable.columns:
        usable = usable.sort_values(["game_date", "player_id"], kind="mergesort")
    usable = usable.reset_index(drop=True)
    played = pd.to_numeric(usable["minutes"], errors="coerce")
    fallback_mu = float(played.median())
    fallback_sd = float(played.std(ddof=0))
    if not np.isfinite(fallback_sd) or fallback_sd < 1.0:
        fallback_sd = 1.0
    learned = usable.iloc[: max(len(usable) - len(usable) // 4, 2)]
    held = usable.iloc[len(learned) :]
    if held.empty:
        alpha = 1.0
        model = _ConditionalMinutes.fit(usable, alpha)
        sigma_rows = usable
        sigma_cond = _sigma_from_model(model, sigma_rows)
    else:
        alpha = _choose_alpha(learned, held)
        probe = _ConditionalMinutes.fit(learned, alpha)
        sigma_rows = held
        sigma_cond = _sigma_from_model(probe, held)
        model = _ConditionalMinutes.fit(usable, alpha)
    sigma_mu = pd.to_numeric(sigma_rows["min_mean_10"], errors="coerce").to_numpy()
    sigma_resid = _as_float(sigma_rows["minutes"]) - sigma_mu
    return RotationSimulator(
        pre_model=_RecentMinutes(),
        cond_model=model,
        sigma_pre=_sigma_table(sigma_mu, sigma_resid),
        sigma_cond=sigma_cond,
        fallback_mu=fallback_mu,
        fallback_sd=fallback_sd,
    )


def allocate_rotation(
    simulator: RotationSimulator,
    role: pd.DataFrame,
    environment: pd.DataFrame,
    rng: np.random.Generator,
    *,
    method: str = "state",
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
    segments=None,
) -> np.ndarray:
    """Draw one team's minutes from a script already on the player's side.

    ``method="segments"`` accumulates minutes from rotation segments. Pass
    ``segments`` to use one schedule, or leave it empty and a deterministic
    schedule is built from the role and each row's script.
    """
    if not isinstance(simulator, RotationSimulator):
        raise ValueError("simulator must be a RotationSimulator")
    if not isinstance(environment, pd.DataFrame) or environment.empty:
        raise ValueError("environment must be a non-empty DataFrame")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    missing = [column for column in _SCRIPT_INPUT if column not in environment.columns]
    if missing:
        raise ValueError(f"environment needs {missing}")
    margin = pd.to_numeric(environment["team_margin"], errors="coerce").to_numpy()
    n_ot = pd.to_numeric(environment["n_ot"], errors="coerce").to_numpy(dtype=float)
    cap = pd.to_numeric(environment["game_minutes"], errors="coerce").to_numpy()
    if np.isnan(margin).any() or np.isnan(n_ot).any() or np.isnan(cap).any():
        raise ValueError("environment script has missing values")
    return simulator.simulate(
        role,
        method=method,
        n=len(environment),
        n_ot=n_ot,
        margin=margin,
        team_total=5.0 * cap,
        cap=cap,
        rng=rng,
        regulation_minutes=regulation_minutes,
        ot_minutes=ot_minutes,
        segments=segments,
    )


def build_rotation_segments(
    role: pd.DataFrame,
    *,
    game_minutes: float,
    n_ot: float,
    team_margin: float,
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
) -> list[dict]:
    """Five-player segments whose durations add up to this script.

    Posted start, availability, and the blowout flag on ``team_margin`` pick
    the fives. The schedule is deterministic, so minute history is unused.
    A later random rotation would draw one schedule per environment row from
    that history and from this same script, still without walking the live
    score.
    """
    if not isinstance(role, pd.DataFrame) or role.empty:
        raise ValueError("roster must be a non-empty DataFrame")
    margin = float(team_margin)
    if not np.isfinite(margin):
        raise ValueError("team margin must be finite")
    periods = _period_lengths(game_minutes, n_ot, regulation_minutes, ot_minutes)
    available, starters = _rotation_pools(role)
    if len(available) < 5:
        raise ValueError(
            "roster cannot fill the team minute budget without passing the cap"
        )
    opening = _five(starters, available)
    opening_keys = {_player_key(player) for player in opening}
    rest = [player for player in available if _player_key(player) not in opening_keys]
    blowout = abs(margin) >= BLOWOUT_MARGIN
    segments = []
    for period, length in periods:
        for start, end, preferred in _stints(period, length, opening, rest, blowout):
            segments.append(
                {
                    "period": period,
                    "start_min": start,
                    "end_min": end,
                    "players": _five(preferred, available),
                }
            )
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
    """Segment and coverage problems in one schedule. Zero is feasible."""
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
    roster: pd.DataFrame,
    *,
    n: int,
    n_ot,
    margin,
    cap,
    segments,
    regulation_minutes: float,
    ot_minutes: float,
) -> np.ndarray:
    if "player_id" not in roster.columns:
        raise ValueError("roster needs player_id")
    player_ids = roster["player_id"].tolist()
    if int(_available_flags(roster).sum()) < 5:
        raise ValueError(
            "roster cannot fill the team minute budget without passing the cap"
        )
    schedules = _broadcast_schedules(segments, n)
    rows = []
    for index in range(n):
        schedule = schedules[index]
        if schedule is None:
            schedule = build_rotation_segments(
                roster,
                game_minutes=float(cap[index]),
                n_ot=float(n_ot[index]),
                team_margin=float(margin[index]),
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
            )
        issues = _segment_issues(
            schedule,
            roster,
            game_minutes=float(cap[index]),
            n_ot=float(n_ot[index]),
            regulation_minutes=regulation_minutes,
            ot_minutes=ot_minutes,
        )
        if issues:
            raise ValueError(issues[0])
        rows.append(_exposure(schedule, player_ids))
    minutes = np.vstack(rows)
    budget = 5.0 * np.asarray(cap, dtype=float)
    bound = np.asarray(cap, dtype=float)[:, None]
    if np.any(np.abs(minutes.sum(axis=1) - budget) > 1e-5):
        raise ValueError("rotation could not meet the team minute budget")
    if np.any(minutes < -1e-9) or np.any(minutes > bound + 1e-6):
        raise ValueError("a player exceeds game minutes")
    return minutes


def _broadcast_schedules(segments, n: int):
    if segments is None:
        return [None] * n
    if isinstance(segments, (str, bytes)) or not isinstance(segments, (list, tuple)):
        raise ValueError("segments must be a list of rotation segments")
    if len(segments) == 0:
        raise ValueError("rotation schedule is empty")
    if isinstance(segments[0], (list, tuple)):
        if len(segments) != n:
            raise ValueError("one rotation schedule is required per simulation")
        return list(segments)
    return [segments for _ in range(n)]


def _segment_issues(
    segments,
    role: pd.DataFrame,
    *,
    game_minutes: float,
    n_ot: float,
    regulation_minutes: float,
    ot_minutes: float,
) -> list[str]:
    if not isinstance(role, pd.DataFrame) or role.empty:
        raise ValueError("roster must be a non-empty DataFrame")
    periods = dict(_period_lengths(game_minutes, n_ot, regulation_minutes, ot_minutes))
    roster_keys, available_keys = _roster_keys(role)
    if isinstance(segments, (str, bytes)) or not isinstance(segments, (list, tuple)):
        raise ValueError("segments must be a list of rotation segments")
    issues = []
    normalized = []
    for segment in segments:
        item, issue = _inspect_segment(segment, roster_keys, available_keys, periods)
        if issue is None:
            normalized.append(item)
        else:
            issues.append(issue)
    coverage = _coverage_issue(normalized, periods)
    if coverage is not None:
        issues.append(coverage)
    return issues


def _inspect_segment(segment, roster_keys: set[str], available_keys: set[str], periods):
    if not isinstance(segment, dict):
        return None, "a rotation segment needs five players"
    try:
        period_value = float(segment["period"])
        period = int(round(period_value))
    except (KeyError, TypeError, ValueError):
        return None, "period is outside the script"
    if abs(period_value - period) > 1e-6 or period not in periods:
        return None, "period is outside the script"
    try:
        start = float(segment["start_min"])
        end = float(segment["end_min"])
    except (KeyError, TypeError, ValueError):
        return None, "duration must be positive"
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None, "duration must be positive"
    if start < -1e-6:
        return None, "segment starts before the period"
    if end > periods[period] + 1e-6:
        return None, "segment runs past the period"
    players = segment.get("players")
    if isinstance(players, (str, bytes)) or not isinstance(players, (list, tuple)):
        return None, "a rotation segment needs five players"
    players = list(players)
    if len(players) != 5:
        return None, "a rotation segment needs five players"
    try:
        keys = [_player_key(player) for player in players]
    except ValueError:
        return None, "player is not on the active roster"
    if len(set(keys)) != 5:
        return None, "players in a rotation segment must be distinct"
    if any(key not in roster_keys for key in keys):
        return None, "player is not on the active roster"
    if any(key not in available_keys for key in keys):
        return None, "an unavailable player cannot take a rotation segment"
    return (
        {
            "period": period,
            "start_min": start,
            "end_min": end,
            "players": players,
        },
        None,
    )


def _coverage_issue(segments, periods: dict[int, float]) -> str | None:
    grouped = {period: [] for period in periods}
    for segment in segments:
        grouped[segment["period"]].append(segment)
    for period, length in periods.items():
        spans = sorted(
            grouped[period], key=lambda item: (item["start_min"], item["end_min"])
        )
        if not spans:
            return "segments do not cover the script's periods"
        cursor = 0.0
        for segment in spans:
            if segment["start_min"] > cursor + 1e-6:
                return f"period {period} has a gap"
            if segment["start_min"] < cursor - 1e-6:
                return f"period {period} overlaps"
            cursor = segment["end_min"]
        if cursor < length - 1e-6:
            return f"period {period} has a gap"
        if cursor > length + 1e-6:
            return f"period {period} overlaps"
    return None


def _period_lengths(game_minutes, n_ot, regulation_minutes, ot_minutes):
    """Regulation is four periods. Overtime periods exist only on this script."""
    if ot_minutes <= 0 or regulation_minutes <= 0:
        raise ValueError("regulation length and overtime length must be positive")
    count = _whole_ot(n_ot)
    length = float(game_minutes)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("game minutes must be positive")
    expected = float(regulation_minutes) + count * float(ot_minutes)
    if abs(length - expected) > 1e-6:
        raise ValueError("game minutes do not match the script's overtime")
    # Four periods, so a 48-minute regulation script is four 12-minute periods.
    quarter = float(regulation_minutes) / 4.0
    periods = [(index, quarter) for index in range(1, 5)]
    periods.extend((5 + extra, float(ot_minutes)) for extra in range(count))
    return periods


def _whole_ot(n_ot) -> int:
    number = float(n_ot)
    if not np.isfinite(number) or number < 0 or abs(number - round(number)) > 1e-6:
        raise ValueError("n_ot must be a nonnegative whole number")
    return int(round(number))


def _rotation_pools(role: pd.DataFrame):
    if "player_id" not in role.columns:
        raise ValueError("roster needs player_id")
    ids = role["player_id"].tolist()
    keys = [_player_key(player) for player in ids]
    if len(set(keys)) != len(keys):
        raise ValueError("roster lists the same player twice")
    flags = _available_flags(role)
    if "is_starter" in role.columns:
        starter_values = pd.to_numeric(role["is_starter"], errors="coerce").to_numpy(
            dtype=float
        )
        starter_flags = (
            np.where(np.isfinite(starter_values), starter_values, 0.0) == 1.0
        )
    else:
        starter_flags = np.zeros(len(ids), dtype=bool)
    available = [player for player, flag in zip(ids, flags) if flag]
    starters = [
        player
        for player, flag, starter in zip(ids, flags, starter_flags)
        if flag and starter
    ]
    return available, starters


def _roster_keys(role: pd.DataFrame) -> tuple[set[str], set[str]]:
    if "player_id" not in role.columns:
        raise ValueError("roster needs player_id")
    ids = role["player_id"].tolist()
    keys = [_player_key(player) for player in ids]
    if len(set(keys)) != len(keys):
        raise ValueError("roster lists the same player twice")
    flags = _available_flags(role)
    available = {key for key, flag in zip(keys, flags) if flag}
    return set(keys), available


def _available_flags(role: pd.DataFrame) -> np.ndarray:
    if "available" not in role.columns:
        return np.ones(len(role), dtype=bool)
    values = pd.to_numeric(role["available"], errors="coerce").to_numpy(dtype=float)
    if values.shape != (len(role),) or np.any(~np.isfinite(values)):
        raise ValueError("available must contain only 0 or 1")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("available must contain only 0 or 1")
    return values == 1.0


def _five(preferred, pool) -> list:
    chosen = []
    seen = set()
    for player in list(preferred) + list(pool):
        key = _player_key(player)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(player)
        if len(chosen) == 5:
            return chosen
    raise ValueError(
        "roster cannot fill the team minute budget without passing the cap"
    )


def _stints(period: int, length: float, opening, rest, blowout: bool):
    if period >= 5:
        preferred = list(rest) + list(opening) if blowout else list(opening)
        return [(0.0, float(length), preferred)]
    half = length / 2.0
    late = 3.0 * length / 4.0
    bounds = ((0.0, half), (half, late), (late, float(length)))
    if period in (1, 3):
        choices = (
            list(opening),
            list(opening[:3]) + list(rest[:2]),
            list(opening[1:]) + list(rest[:1]),
        )
    elif blowout and period == 4:
        choices = (
            list(rest) + list(opening),
            list(rest[1:]) + list(opening),
            list(rest) + list(opening),
        )
    else:
        choices = (
            list(opening[:3]) + list(rest[:2]),
            list(opening[3:]) + list(rest[:3]),
            list(opening),
        )
    return [(start, end, choices[index]) for index, (start, end) in enumerate(bounds)]


def _exposure(segments, player_ids) -> np.ndarray:
    index = {
        _player_key(player): position for position, player in enumerate(player_ids)
    }
    minutes = np.zeros(len(player_ids), dtype=float)
    for segment in segments:
        duration = float(segment["end_min"]) - float(segment["start_min"])
        for player in segment["players"]:
            minutes[index[_player_key(player)]] += duration
    return minutes


def _player_key(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("player_id must name a player")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        raise ValueError("player_id must name a player")
    return text


def scale_mu(mu, total):
    """Expected minutes rescaled so each simulation sums to that team total."""
    mu = np.atleast_2d(np.asarray(mu, dtype=float)).copy()
    totals = np.asarray(total, dtype=float).reshape(-1)
    weight = mu.sum(axis=1, keepdims=True)
    usable = np.where(weight > 0, mu, 1.0)
    weight = usable.sum(axis=1)
    return usable * (totals / weight)[:, None]


def draw_minutes(mean, sd, cap, rng: np.random.Generator) -> np.ndarray:
    """Truncated normal on ``[0, cap]`` whose mean is the requested minute total.

    Clipping a plain normal at zero pulls a low minute total upward. Matching
    the mean inside the cap keeps a 4-minute player at 4 minutes.
    """
    mean, sd, cap = np.broadcast_arrays(
        np.asarray(mean, dtype=float),
        np.asarray(sd, dtype=float),
        np.asarray(cap, dtype=float),
    )
    mean = np.array(mean, dtype=float, copy=True)
    sd = np.maximum(sd, 1e-6)
    cap = np.maximum(np.array(cap, dtype=float, copy=True), 0.0)
    target = np.clip(mean, 0.0, cap)
    out = target.copy()
    free = (target > 1e-6) & (target < cap - 1e-6)
    if not np.any(free):
        return out
    loc = _location_for_mean(target[free], sd[free], cap[free])
    scale = sd[free]
    upper = cap[free]
    alpha = norm.cdf((0.0 - loc) / scale)
    beta = norm.cdf((upper - loc) / scale)
    span = np.maximum(beta - alpha, 1e-12)
    ticks = alpha + rng.random(loc.shape) * span
    ticks = np.clip(ticks, 1e-8, 1.0 - 1e-8)
    out[free] = np.clip(loc + scale * norm.ppf(ticks), 0.0, upper)
    return out


def sim_gaussian(mean_mat, sd, total, cap, rng):
    """Truncated noise, then the team budget and the per-player cap."""
    mean_mat = np.atleast_2d(np.asarray(mean_mat, dtype=float))
    total = np.asarray(total, dtype=float).reshape(-1)
    cap = np.asarray(cap, dtype=float).reshape(-1)
    if cap.shape != total.shape:
        raise ValueError("cap and the team minute budget must have the same length")
    if mean_mat.shape[0] not in (1, total.shape[0]):
        raise ValueError("expected minutes and the team total do not line up")
    if mean_mat.shape[0] == 1 and total.shape[0] != 1:
        mean_mat = np.repeat(mean_mat, total.shape[0], axis=0)
    spread = _as_player_matrix(sd, mean_mat.shape)
    raw = draw_minutes(mean_mat, spread, cap[:, None], rng)
    return constrain(raw, np.maximum(spread, 1e-6) ** 2, total, cap)


def constrain(raw, w, total, cap, iters=100):
    """Project minutes onto ``sum = total`` and ``0 <= minutes <= cap``.

    ``minutes = clip(raw + λ w, 0, cap)``. λ is the one number per simulation
    that hits the team total. The sum is monotone in λ, so bisection finds it,
    including for players who start the step sitting on 0 or on the cap.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    n, k = raw.shape
    total = np.asarray(total, dtype=float).reshape(-1)
    cap = np.asarray(cap, dtype=float).reshape(-1)
    weight = _as_player_matrix(w, raw.shape)
    if total.shape != (n,) or cap.shape != (n,):
        raise ValueError("constraint shapes do not line up")
    if not np.any(weight > 0):
        weight = np.ones_like(raw)
    weight = np.maximum(weight, 1e-9)
    ceiling = cap * k
    if np.any(total < -1e-6) or np.any(total > ceiling + 1e-6):
        raise ValueError("team minute budget is outside what the cap can hold")
    total = np.clip(total, 0.0, ceiling)
    bound = cap[:, None]
    lo = np.min(-raw / weight, axis=1)
    hi = np.max((bound - raw) / weight, axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        projected = np.clip(raw + mid[:, None] * weight, 0.0, bound)
        over = projected.sum(axis=1) > total
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    minutes = np.clip(raw + (0.5 * (lo + hi))[:, None] * weight, 0.0, bound)
    gap = np.max(np.abs(minutes.sum(axis=1) - total))
    if gap > 1e-5:
        raise ValueError("rotation could not meet the team minute budget")
    return minutes


class RotationSimulator:
    """Conditional minutes from a player's role, then the team constraint.

    ``alpha``, ``margin_resid``, and ``p_ot`` are accepted and ignored. The
    margin and the length come from the attached script on each call.
    """

    def __init__(
        self,
        pre_model=None,
        cond_model=None,
        sigma_pre=None,
        sigma_cond=None,
        alpha=None,
        margin_resid=None,
        p_ot=None,
        fallback_mu=24.0,
        fallback_sd=6.0,
    ):
        del alpha, margin_resid, p_ot
        if not np.isfinite(fallback_mu) or fallback_mu < 0:
            raise ValueError("fallback_mu must be a non-negative finite number")
        if not np.isfinite(fallback_sd) or fallback_sd <= 0:
            raise ValueError("fallback_sd must be a positive finite number")
        self.pre = pre_model
        self.cond = cond_model
        self.sig_pre = None if sigma_pre is None else np.asarray(sigma_pre, dtype=float)
        self.sig_cond = (
            None if sigma_cond is None else np.asarray(sigma_cond, dtype=float)
        )
        self.fb_mu = float(fallback_mu)
        self.fb_sd = float(fallback_sd)

    def pregame_mu(self, roster: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Minutes from the player's recent window, before the script."""
        mu = np.full(len(roster), self.fb_mu, dtype=float)
        known = self._known(roster)
        if known.any() and self.pre is not None:
            rows = roster.iloc[np.flatnonzero(known)]
            predicted = np.asarray(self.pre.predict(rows), dtype=float).reshape(-1)
            if predicted.shape != (int(known.sum()),):
                raise ValueError("pregame model returned the wrong number of minutes")
            predicted = np.where(np.isfinite(predicted), predicted, self.fb_mu)
            mu[known] = predicted
        return np.clip(mu, 0, None), known

    def simulate(
        self,
        roster: pd.DataFrame,
        method: str = "state",
        n: int = 2000,
        seed: int = 0,
        *,
        n_ot=None,
        margin=None,
        team_total=None,
        cap=None,
        rng: np.random.Generator | None = None,
        regulation_minutes: float = REG_MIN,
        ot_minutes: float = OT_MIN,
        segments=None,
    ) -> np.ndarray:
        """One team, one script. Returns ``(n, players)`` minutes.

        ``margin`` is that team's final margin. The other team passes the
        same draw with the sign flipped before this call.
        ``method="segments"`` returns exposure to rotation segments and does
        not rescale those minutes.
        """
        if not isinstance(roster, pd.DataFrame) or roster.empty:
            raise ValueError("roster must be a non-empty DataFrame")
        if "team_id" in roster.columns and roster["team_id"].nunique(dropna=True) > 1:
            raise ValueError("a rotation is one team")
        if method not in {"state", "gaussian", "segments"}:
            raise ValueError("method must be state, gaussian, or segments")
        if segments is not None and method != "segments":
            raise ValueError("segments are only used when method is segments")
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
            raise ValueError("n must be a positive integer")
        n = int(n)
        if rng is None:
            rng = np.random.default_rng(seed)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy Generator")
        n_ot, ot, margin, total, cap = _script_arrays(
            n,
            n_ot,
            margin,
            team_total,
            cap,
            regulation_minutes=regulation_minutes,
            ot_minutes=ot_minutes,
        )
        if np.any(total > len(roster) * cap + 1e-6):
            raise ValueError(
                "roster cannot fill the team minute budget without passing the cap"
            )
        if method == "segments":
            # Exposure already meets the budget. scale_mu would move players
            # off the segments that produced these minutes.
            return _segment_minutes(
                roster,
                n=n,
                n_ot=n_ot,
                margin=margin,
                cap=cap,
                segments=segments,
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
            )
        mu, known = self.pregame_mu(roster)
        mu_state = np.tile(mu, (n, 1))
        if method == "state" and self.cond is not None and known.any():
            mu_state[:, known] = self._conditional(roster, known, margin, n_ot, ot, cap)
        mu_state = np.clip(mu_state, 0, None)
        table = self.sig_cond if method == "state" else self.sig_pre
        sd = np.full_like(mu_state, self.fb_sd)
        if known.any():
            sd[:, known] = self._sigma(mu_state[:, known], table)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, self.fb_sd)
        return sim_gaussian(scale_mu(mu_state, total), sd, total, cap, rng)

    def _conditional(self, roster, known, margin, n_ot, ot, cap) -> np.ndarray:
        base = roster.iloc[np.flatnonzero(known)].reset_index(drop=True)
        k = len(base)
        n = len(margin)
        frame = pd.concat([base] * n, ignore_index=True)
        sim = np.repeat(np.arange(n), k)
        frame["team_final_margin"] = margin[sim]
        frame["abs_final_margin"] = np.abs(margin)[sim]
        frame["n_ot"] = n_ot[sim]
        frame["went_ot"] = ot.astype(float)[sim]
        frame["game_minutes"] = cap[sim]
        frame["blowout"] = (np.abs(margin)[sim] >= BLOWOUT_MARGIN).astype(float)
        missing = [column for column in SCRIPT if column not in frame.columns]
        if missing:
            raise ValueError(f"script features missing: {missing}")
        predicted = np.asarray(self.cond.predict(frame), dtype=float).reshape(n, k)
        if predicted.shape != (n, k) or not np.isfinite(predicted).all():
            raise ValueError("conditional minutes model returned a non-finite value")
        return predicted

    def _known(self, roster: pd.DataFrame) -> np.ndarray:
        if "min_mean_10" not in roster.columns:
            return np.zeros(len(roster), dtype=bool)
        values = pd.to_numeric(roster["min_mean_10"], errors="coerce")
        return values.notna().to_numpy()

    def _sigma(self, mu: np.ndarray, table) -> np.ndarray:
        mu = np.asarray(mu, dtype=float)
        if table is None or np.asarray(table).size == 0:
            return np.full(mu.shape, self.fb_sd)
        values = np.asarray(table, dtype=float).reshape(-1)
        index = np.clip(np.digitize(mu, MU_BINS) - 1, 0, len(values) - 1)
        return values[index]


class _RecentMinutes:
    """Pregame minutes are the player's own recent mean, already shifted or not."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if "min_mean_10" not in frame.columns:
            raise ValueError("role needs min_mean_10")
        return _as_float(frame["min_mean_10"])


class _ConditionalMinutes:
    """Minutes from the role and the script, with the script scaled by role.

    ``min_mean_10`` stays in the regression so a long recent run can revert.
    Products with the margin, the blowout, and overtime let a starter and a
    reserve move by different amounts. Features are standardized, and the
    penalty is the one that scored best on the later games.
    """

    def __init__(self, pipeline: Pipeline, medians: dict[str, float], alpha: float):
        self.pipeline = pipeline
        self.medians = medians
        self.alpha = alpha
        self.columns = _FEATURES

    @classmethod
    def fit(cls, roles: pd.DataFrame, alpha: float) -> _ConditionalMinutes:
        medians = _medians(roles)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=float(alpha))),
            ]
        )
        model.fit(_design(roles, medians), _as_float(roles["minutes"]))
        return cls(model, medians, float(alpha))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in CONDITIONAL if column not in frame.columns]
        if missing:
            raise ValueError(f"conditional minutes are missing {missing}")
        designed = _design(frame, self.medians)
        return np.asarray(self.pipeline.predict(designed), dtype=float)


def _choose_alpha(learned: pd.DataFrame, held: pd.DataFrame) -> float:
    target = _as_float(held["minutes"])
    best_alpha = _ALPHAS[0]
    best_mse = np.inf
    for alpha in _ALPHAS:
        model = _ConditionalMinutes.fit(learned, alpha)
        mse = float(np.mean((model.predict(held) - target) ** 2))
        if mse < best_mse - 1e-8:
            best_alpha = alpha
            best_mse = mse
    return best_alpha


def _sigma_from_model(model: _ConditionalMinutes, rows: pd.DataFrame) -> np.ndarray:
    predicted = model.predict(rows)
    return _sigma_table(predicted, _as_float(rows["minutes"]) - predicted)


def _medians(frame: pd.DataFrame) -> dict[str, float]:
    values = frame.loc[:, list(CONDITIONAL)].apply(pd.to_numeric, errors="coerce")
    medians = values.median(numeric_only=True).fillna(0.0)
    return {column: float(medians[column]) for column in CONDITIONAL}


def _design(frame: pd.DataFrame, medians: dict[str, float]) -> np.ndarray:
    values = frame.loc[:, list(CONDITIONAL)].apply(pd.to_numeric, errors="coerce")
    for column, fill in medians.items():
        values[column] = values[column].fillna(fill)
    recent = values["min_mean_10"]
    values["min_mean_x_abs_margin"] = recent * values["abs_final_margin"]
    values["min_mean_x_blowout"] = recent * values["blowout"]
    values["min_mean_x_n_ot"] = recent * values["n_ot"]
    values["starter_x_blowout"] = values["is_starter"] * values["blowout"]
    return values.loc[:, list(_FEATURES)].to_numpy(dtype=float)


def _played_minutes(logs: pd.DataFrame) -> pd.DataFrame:
    frame = logs.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["minutes"] = _minutes(frame)
    _reject_minute_conflicts(frame)
    keys = ["player_id", "game_id", "team_id"]
    return frame.loc[:, [*keys, "minutes"]].drop_duplicates(keys)


def _script_arrays(
    n,
    n_ot,
    margin,
    team_total,
    cap,
    *,
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
):
    if margin is None or team_total is None or cap is None:
        raise ValueError(
            "rotation needs the attached script: margin, team minute budget, and cap"
        )
    if ot_minutes <= 0 or regulation_minutes <= 0:
        raise ValueError("regulation length and overtime length must be positive")
    margin = np.asarray(margin, dtype=float).reshape(-1)
    total = np.asarray(team_total, dtype=float).reshape(-1)
    cap = np.asarray(cap, dtype=float).reshape(-1)
    if n_ot is None:
        n_ot = np.maximum(cap - regulation_minutes, 0.0) / ot_minutes
    else:
        n_ot = np.asarray(n_ot, dtype=float).reshape(-1)
    ot = n_ot > 1e-9
    shapes = (margin.shape, total.shape, cap.shape, n_ot.shape)
    if any(shape != (n,) for shape in shapes):
        raise ValueError("margin, n_ot, team minute budget, and cap must have length n")
    missing = (
        np.isnan(margin).any()
        or np.isnan(total).any()
        or np.isnan(cap).any()
        or np.isnan(n_ot).any()
    )
    if missing:
        raise ValueError("attached script has missing values")
    if np.any(cap <= 0) or np.any(total <= 0):
        raise ValueError("game minutes and the team minute budget must be positive")
    return n_ot, ot, margin, total, cap


def _as_float(values) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _as_player_matrix(values, shape: tuple[int, int]) -> np.ndarray:
    n, k = shape
    values = np.asarray(values, dtype=float)
    if values.shape == (k,):
        return np.broadcast_to(values, (n, k)).copy()
    if values.shape == (n, k):
        return values.copy()
    if values.shape == (1, k):
        return np.repeat(values, n, axis=0)
    raise ValueError("minutes, spread, and the team total do not line up")


def _location_for_mean(target, scale, upper, iters: int = 60) -> np.ndarray:
    lo = target - 16.0 * scale
    hi = target + 16.0 * scale
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = _truncated_mean(mid, scale, upper)
        hi = np.where(got > target, mid, hi)
        lo = np.where(got > target, lo, mid)
    return 0.5 * (lo + hi)


def _truncated_mean(loc, scale, upper) -> np.ndarray:
    alpha = (0.0 - loc) / scale
    beta = (upper - loc) / scale
    mass = np.maximum(norm.cdf(beta) - norm.cdf(alpha), 1e-12)
    return loc + scale * (norm.pdf(alpha) - norm.pdf(beta)) / mass


def _sigma_table(mu: np.ndarray, resid: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    resid = np.asarray(resid, dtype=float)
    overall = float(np.std(resid)) if len(resid) else _SIGMA_FLOOR
    if not np.isfinite(overall) or overall < _SIGMA_FLOOR:
        overall = _SIGMA_FLOOR
    bins = np.clip(np.digitize(mu, MU_BINS) - 1, 0, len(MU_BINS) - 2)
    table = []
    for index in range(len(MU_BINS) - 1):
        piece = resid[bins == index]
        value = float(np.std(piece)) if len(piece) >= 8 else overall
        if not np.isfinite(value) or value < _SIGMA_FLOOR:
            value = overall
        table.append(value)
    return np.asarray(table, dtype=float)
