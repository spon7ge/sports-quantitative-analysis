"""Who plays, before the team minute budget is split.

A roster with close and blowout rates is drawn from the script margin.
Posted starters stay in, ruled-out players stay out, and a short draw is
filled back to seven. Without those rates, only players under
``LOW_MINUTE_CUTOFF`` are drawn from a shrunk appearance rate. Logs record
an appearance and not a DNP, so that average is minutes when he played.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_rotations.checks import _frame, _required
from game_rotations.constants import (
    _PRIOR_MINUTES,
    _PRIOR_RATE,
    _TOL,
    APPEARANCE_WINDOW,
    CLOSE_MINUTES,
    LOW_MINUTE_CUTOFF,
    MIN_ACTIVE,
    PLAY_PRIOR_STRENGTH,
)
from role_context import BLOWOUT_MARGIN, _minutes


def play_prior(minutes) -> np.ndarray:
    """Appearance rate for a minute tier, from seasons through 2024-25."""
    values = np.asarray(minutes, dtype=float)
    if values.shape == () and not np.isfinite(values):
        raise ValueError("minutes for the play prior must be finite")
    if values.shape != () and not np.isfinite(values).all():
        raise ValueError("minutes for the play prior must be finite")
    clipped = np.clip(values, _PRIOR_MINUTES[0], _PRIOR_MINUTES[-1])
    return np.interp(clipped, _PRIOR_MINUTES, _PRIOR_RATE)


def shrink_play_rate(appearances, games, minutes, *, strength=PLAY_PRIOR_STRENGTH):
    """(appearances + k * prior) / (games + k). Zero games returns the prior."""
    apps = np.asarray(appearances, dtype=float)
    count = np.asarray(games, dtype=float)
    apps, count = np.broadcast_arrays(apps, count)
    if (
        not np.isfinite(apps).all()
        or not np.isfinite(count).all()
        or (apps < 0).any()
        or (count < 0).any()
    ):
        raise ValueError("appearances and games must be finite and nonnegative")
    if np.any(apps > count + _TOL):
        raise ValueError("appearances cannot exceed team games")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("prior strength must be a positive finite number")
    prior = play_prior(minutes)
    return (apps + float(strength) * prior) / (count + float(strength))


def play_rates(logs: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Shrunk rate from team games before each roster row."""
    return AppearanceIndex(logs).rates(roster)


class AppearanceIndex:
    """Team games and appearances, keyed for repeated roster lookups."""

    def __init__(self, logs: pd.DataFrame):
        _frame(logs, "logs")
        _required(logs, ["player_id", "team_id", "game_id", "game_date"])
        played = logs.copy()
        played["game_date"] = pd.to_datetime(played["game_date"]).dt.normalize()
        if played["game_date"].isna().any():
            raise ValueError("game_date has unparseable values")
        played["minutes"] = _minutes(played)
        played = played.loc[played["minutes"].gt(0)].copy()
        played["team_key"] = played["team_id"].map(_key)
        played["player_key"] = played["player_id"].map(_key)
        games = (
            played.groupby(["team_key", "game_id"], sort=False)
            .agg(game_date=("game_date", "min"))
            .reset_index()
            .sort_values(["team_key", "game_date", "game_id"], kind="mergesort")
        )
        self._games: dict = {}
        for team, block in games.groupby("team_key", sort=False):
            self._games[team] = list(
                zip(
                    block["game_date"].to_numpy(),
                    block["game_id"].astype(str).to_numpy(),
                    strict=True,
                )
            )
        appeared: dict = {}
        for row in played.itertuples(index=False):
            appeared.setdefault((row.player_key, row.team_key), set()).add(
                str(row.game_id)
            )
        self._appeared = appeared

    def rates(self, roster: pd.DataFrame) -> pd.DataFrame:
        _frame(roster, "roster")
        _required(roster, ["player_id", "team_id", "game_date", "min_mean_10"])
        slate_dates = pd.to_datetime(roster["game_date"]).dt.normalize()
        if slate_dates.isna().any():
            raise ValueError("roster game_date has unparseable values")
        minutes = roster["min_mean_10"].to_numpy(dtype=float)
        if not np.isfinite(minutes).all() or (minutes < 0).any():
            raise ValueError("min_mean_10 must be finite and nonnegative")
        apps = np.zeros(len(roster), dtype=float)
        games = np.zeros(len(roster), dtype=float)
        for i, row in enumerate(roster.itertuples(index=False)):
            team = _key(row.team_id)
            player = _key(row.player_id)
            seen = self._appeared.get((player, team), set())
            history = [
                (day, game_id)
                for day, game_id in self._games.get(team, ())
                if day < slate_dates.iloc[i]
            ]
            joined = [day for day, game_id in history if game_id in seen]
            if joined:
                first = min(joined)
                history = [(day, game_id) for day, game_id in history if day >= first]
            else:
                history = []
            window = history[-APPEARANCE_WINDOW:]
            games[i] = len(window)
            apps[i] = sum(game_id in seen for _, game_id in window)
        return pd.DataFrame(
            {
                "player_id": roster["player_id"].to_numpy(),
                "team_id": roster["team_id"].to_numpy(),
                "game_date": slate_dates.to_numpy(),
                "appearances": apps,
                "games": games,
                "play_rate": shrink_play_rate(apps, games, minutes),
            }
        )


def script_probability(close_rate, blowout_rate, margin):
    """Close games use close_rate. A blowout uses either appearance."""
    close = np.asarray(close_rate, dtype=float)
    blow = np.asarray(blowout_rate, dtype=float)
    margin = np.asarray(margin, dtype=float)
    _probability(close, "close_rate")
    _probability(blow, "blowout_rate")
    if not np.isfinite(margin).all():
        raise ValueError("margin must be finite")
    union = 1.0 - (1.0 - close) * (1.0 - blow)
    if np.ndim(margin) == 0:
        return np.where(np.abs(margin) >= BLOWOUT_MARGIN, union, close)
    # One probability per script and player, including when those counts match.
    return np.where(
        (np.abs(margin) >= BLOWOUT_MARGIN)[:, None],
        np.atleast_1d(union)[None, :],
        np.atleast_1d(close)[None, :],
    )


def script_rates(exposure: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Shrunk close and blowout rates from the team's last parsed games.

    The window starts at the player's first parsed appearance for that team.
    Player logs do not record games he was listed out, so those games cannot
    be skipped. The blowout rate's denominator is games whose margin exceeded
    the blowout cutoff, not every game in the window.
    """
    played = _exposure_frame(exposure)
    slate_dates = pd.to_datetime(roster["game_date"]).dt.normalize()
    if slate_dates.isna().any():
        raise ValueError("roster game_date has unparseable values")
    _required(roster, ["player_id", "team_id", "game_date"])
    team_games = _team_games(played)
    appearances = _appearances(played)
    close_hits = np.zeros(len(roster))
    games = np.zeros(len(roster))
    garbage_hits = np.zeros(len(roster))
    blowout_games = np.zeros(len(roster))
    for i, row in enumerate(roster.itertuples(index=False)):
        team = _same_key(row.team_id)
        player = _same_key(row.player_id)
        seen = appearances.get((player, team), {})
        history = [
            item
            for item in team_games.get(team, ())
            if item[0] < slate_dates.iloc[i]
        ]
        joined = [item[0] for item in history if item[1] in seen]
        if joined:
            first = min(joined)
            history = [item for item in history if item[0] >= first]
        else:
            history = []
        window = history[-APPEARANCE_WINDOW:]
        games[i] = len(window)
        for _, game_id, reached in window:
            close_min, garbage = seen.get(game_id, (0.0, False))
            close_hits[i] += close_min >= CLOSE_MINUTES
            blowout_games[i] += reached
            garbage_hits[i] += reached and garbage
    return pd.DataFrame(
        {
            "player_id": roster["player_id"].to_numpy(),
            "team_id": roster["team_id"].to_numpy(),
            "game_date": slate_dates.to_numpy(),
            "close_hits": close_hits,
            "games": games,
            "garbage_hits": garbage_hits,
            "blowout_games": blowout_games,
            "close_rate": (close_hits + 1) / (games + 2),
            "blowout_rate": (garbage_hits + 1) / (blowout_games + 2),
        }
    )


def draw_script_availability(roster, margin, base, rng: np.random.Generator):
    """Who plays this script. Starters stay in. Ruled-out players stay out."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    _frame(roster, "roster")
    _required(roster, ["close_rate", "blowout_rate"])
    base = np.asarray(base, dtype=bool)
    margin = np.asarray(margin, dtype=float).reshape(-1)
    if base.ndim != 2 or margin.shape != (base.shape[0],):
        raise ValueError("margin and availability must share one script count")
    n, k = base.shape
    if len(roster) != k:
        raise ValueError("availability must have one column per roster player")
    close = np.asarray(roster["close_rate"], dtype=float).reshape(-1)
    blow = np.asarray(roster["blowout_rate"], dtype=float).reshape(-1)
    if close.shape != (k,) or blow.shape != (k,):
        raise ValueError("script rates must have one value per player")
    starter = _marked(roster, "is_starter", k)
    prob = np.asarray(script_probability(close, blow, margin), dtype=float)
    if prob.shape != (n, k):
        raise ValueError("script probabilities must align with availability")
    chosen = base & (starter | (_uniforms(rng, (n, k)) < prob))
    for i in range(n):
        _fill_to_seven(chosen[i], base[i], prob[i])
    if np.any(chosen.sum(axis=1) < 5):
        raise ValueError("every scenario needs at least five available players")
    return chosen


def draw_availability(roster, weights, base, total, rng: np.random.Generator):
    """Per-script mask. Low-minute players sit unless the budget needs them."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    _frame(roster, "roster")
    _required(roster, ["play_rate", "min_mean_10"])
    weights = np.asarray(weights, dtype=float)
    base = np.asarray(base, dtype=bool)
    total = np.asarray(total, dtype=float).reshape(-1)
    if weights.ndim != 2 or base.shape != weights.shape:
        raise ValueError("weights and availability must share one roster shape")
    n, k = weights.shape
    if total.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights and the team budget must align and be finite")
    if not np.isfinite(total).all() or (total < 0).any():
        raise ValueError("team budget must be finite and nonnegative")
    rate = np.asarray(roster["play_rate"], dtype=float).reshape(-1)
    minutes = np.asarray(roster["min_mean_10"], dtype=float).reshape(-1)
    if rate.shape != (k,) or minutes.shape != (k,):
        raise ValueError("play_rate and min_mean_10 must have one value per player")
    if (
        not np.isfinite(rate).all()
        or (rate < 0).any()
        or (rate > 1).any()
        or not np.isfinite(minutes).all()
        or (minutes < 0).any()
    ):
        raise ValueError("play_rate must be a probability and minutes nonnegative")
    low = minutes < LOW_MINUTE_CUTOFF
    low &= ~_marked(roster, "is_starter", k)
    if "probable" in roster.columns:
        low &= ~_marked(roster, "probable", k)
    stay = rng.random((n, k)) < rate
    active = base & (~low | stay)
    priority = np.where(low, rate, -1.0)
    order = np.argsort(-priority, kind="mergesort")
    factors = np.empty(n, dtype=float)
    for i in range(n):
        chosen = active[i].copy()
        _fill_to_five(chosen, base[i], order, priority)
        _fill_to_budget(chosen, base[i], order, priority, weights[i], total[i])
        active[i] = chosen
        covered = float(weights[i, chosen].sum())
        factors[i] = total[i] / covered if covered > _TOL else np.inf
    if np.any(active.sum(axis=1) < 5):
        raise ValueError("every scenario needs at least five available players")
    return active, factors


def _fill_to_five(chosen, base, order, priority) -> None:
    while int(chosen.sum()) < 5:
        added = _next_player(chosen, base, order, priority, None)
        if added is None:
            return
        chosen[added] = True


def _fill_to_budget(chosen, base, order, priority, weights, total) -> None:
    for _ in range(len(chosen)):
        if float(weights[chosen].sum()) + _TOL >= total:
            return
        added = _next_player(chosen, base, order, priority, weights)
        if added is None:
            return
        chosen[added] = True


def _next_player(chosen, base, order, priority, weights):
    for player in order:
        if chosen[player] or not base[player] or priority[player] < 0:
            continue
        if weights is not None and weights[player] <= 0:
            continue
        return int(player)
    return None


def _marked(roster: pd.DataFrame, column: str, count: int) -> np.ndarray:
    if column not in roster.columns:
        return np.zeros(count, dtype=bool)
    values = np.asarray(roster[column], dtype=float).reshape(-1)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"{column} must be a finite flag for each player")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError(f"{column} must contain only 0 or 1")
    return values == 1.0


def _exposure_frame(exposure: pd.DataFrame) -> pd.DataFrame:
    _frame(exposure, "exposure")
    _required(
        exposure,
        [
            "player_id",
            "team_id",
            "game_id",
            "game_date",
            "close_min",
            "garbage_app",
            "reached_20",
        ],
    )
    played = exposure.copy()
    played["game_date"] = pd.to_datetime(played["game_date"]).dt.normalize()
    if played["game_date"].isna().any():
        raise ValueError("game_date has unparseable values")
    played["close_min"] = pd.to_numeric(played["close_min"], errors="coerce")
    if (
        not np.isfinite(played["close_min"]).all()
        or (played["close_min"] < 0).any()
    ):
        raise ValueError("close_min must be finite and nonnegative")
    played["garbage_app"] = _flags(played["garbage_app"], "garbage_app")
    played["reached_20"] = _flags(played["reached_20"], "reached_20")
    if np.any(played["garbage_app"] & ~played["reached_20"]):
        raise ValueError("garbage time requires the margin to have exceeded 20")
    return played


def _team_games(played: pd.DataFrame) -> dict:
    games = (
        played.groupby(["team_id", "game_id"], sort=False)
        .agg(
            game_date=("game_date", "min"),
            reached_20=("reached_20", "max"),
        )
        .reset_index()
        .sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    )
    out: dict = {}
    for team, block in games.groupby("team_id", sort=False):
        out[_same_key(team)] = list(
            zip(
                block["game_date"].to_numpy(),
                block["game_id"].map(_key).to_numpy(),
                block["reached_20"].to_numpy(dtype=bool),
                strict=True,
            )
        )
    return out


def _appearances(played: pd.DataFrame) -> dict:
    grouped = played.groupby(["player_id", "team_id", "game_id"], sort=False)
    close = grouped["close_min"].sum()
    garbage = grouped["garbage_app"].max()
    out: dict = {}
    for (player, team, game), minutes in close.items():
        out.setdefault((_same_key(player), _same_key(team)), {})[_key(game)] = (
            float(minutes),
            bool(garbage.loc[(player, team, game)]),
        )
    return out


def _uniforms(rng, size):
    return rng.random(size)


def _fill_to_seven(chosen, eligible, priority) -> None:
    order = np.argsort(-np.asarray(priority, dtype=float), kind="mergesort")
    for player in order:
        if int(chosen.sum()) >= MIN_ACTIVE:
            return
        if chosen[player] or not eligible[player]:
            continue
        chosen[player] = True


def _probability(values, name) -> None:
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{name} must be a probability")


def _flags(values, name) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(values.dtype, pd.BooleanDtype) or values.dtype == bool:
        if values.isna().any():
            raise ValueError(f"{name} must be true or false")
        return values.to_numpy(dtype=bool)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must be true or false")
    flags = numeric.to_numpy(dtype=float)
    if np.any((flags != 0.0) & (flags != 1.0)):
        raise ValueError(f"{name} must be true or false")
    return flags == 1.0


def _same_key(value):
    key = _key(value)
    if isinstance(key, str) and key.isdigit():
        return int(key)
    return key


def _key(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("identifier cannot be boolean")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        raise ValueError("missing identifier")
    return text
