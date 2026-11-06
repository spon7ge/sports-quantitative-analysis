"""Layer 5: assists, rebounds, and defense from shared game events.

Player totals are allocations of events the game already produced.
Missed shots become rebounds, made field goals become assists, opponent
turnovers become steals, and opponent attempts become blocks. An injury
changes who is eligible and how the weights are split. It does not create
a new pool of events for the replacement.

Assists use the joint model. For each made shot by player j, the make is
assisted with probability ``assisted_rate[j]``, and that assist is given
to a teammate. Independent ``Binomial(team FGM, p_i)`` draws can exceed
team makes and can credit the scorer, so they are not used.

Teammate shooting efficiency enters through the made-field-goal counts
supplied by Layer 4. This layer does not resimulate whether the shot falls.
``creation_rate`` combines ball-handling and shot creation. On-court
sharing beyond the availability flag is the caller's ``lineup_factors``.

``allocate_events`` credits players from their recent rates. The shot and
turnover counts still come from outside this layer. A training rate stops
before the game being scored. A live rate keeps the latest completed game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from opportunity_shares import opportunity_share_rows, opportunity_shares
from possession_allocation import (
    allocate_from_pool,
    multinomial_rows,
    renormalize_props,
)
from role_context import _before, _minutes, _reject_minute_conflicts, _through

_STATS = ("AST", "OREB", "DREB", "REB", "STL", "BLK")
_REQUIRED_TEAM_FIELDS = (
    "player_ids",
    "fga",
    "fgm",
    "tov",
    "available",
    "projected_minutes",
    "reference_minutes",
    "oreb_rate",
    "dreb_rate",
    "creation_rate",
    "assisted_rate",
    "stl_rate",
    "blk_rate",
)


def _invariant(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Invariant violated: {message}")


def _as_count_matrix(value: Any, n_sims: int, n_players: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer matrix") from exc
    if arr.shape != (n_sims, n_players):
        raise ValueError(f"{name} must have shape ({n_sims}, {n_players})")
    if (
        np.any(~np.isfinite(arr))
        or np.any(arr < 0)
        or np.any(arr != np.floor(arr))
        or np.any(arr > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must be a nonnegative integer matrix")
    return arr.astype(np.int64)


def _as_probability(value: Any, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1]") from exc
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _as_unit_interval_vector(value: Any, n_players: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain values in [0, 1]") from exc
    if (
        arr.shape != (n_players,)
        or np.any(~np.isfinite(arr))
        or np.any((arr < 0.0) | (arr > 1.0))
    ):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return arr


def _as_context(value: Any, n_players: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "context_factors must contain nonnegative finite values"
        ) from exc
    if arr.ndim != 1 or arr.shape != (n_players,):
        raise ValueError(
            "context_factors must be a one-dimensional player-aligned array"
        )
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("context_factors must contain nonnegative finite values")
    return arr


def event_shares(
    rates: np.ndarray,
    available: np.ndarray,
    projected_minutes: np.ndarray,
    reference_minutes: np.ndarray,
    role_factors: np.ndarray | None = None,
    lineup_factors: np.ndarray | None = None,
    context_factors: np.ndarray | None = None,
    minutes_exponent: float = 1.0,
    allow_empty: bool = False,
) -> np.ndarray:
    """Normalize event weights. ``rates`` are the pre-minutes baseline.

    ``context_factors`` is the event-specific context: opponent shot profile
    for rebounds, or matchup context for steals and blocks. It multiplies
    lineup composition; it does not replace it.
    """
    lineup = lineup_factors
    if context_factors is not None:
        context = _as_context(context_factors, len(np.asarray(rates)))
        if lineup_factors is None:
            lineup = context
        else:
            # Validate lineup before multiplying. A negative lineup and a
            # negative context would otherwise cancel and look legal.
            try:
                lineup_values = np.asarray(lineup_factors, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "lineup_factors must contain nonnegative finite values"
                ) from exc
            if (
                lineup_values.shape != context.shape
                or np.any(~np.isfinite(lineup_values))
                or np.any(lineup_values < 0.0)
            ):
                raise ValueError(
                    "lineup_factors must contain nonnegative finite values"
                )
            lineup = lineup_values * context
    projected = np.asarray(projected_minutes, dtype=float)
    share = opportunity_share_rows if projected.ndim == 2 else opportunity_shares
    return share(
        base_weights=rates,
        available=available,
        projected_minutes=projected_minutes,
        reference_minutes=reference_minutes,
        role_factors=role_factors,
        lineup_factors=lineup,
        minutes_exponent=minutes_exponent,
        allow_empty=allow_empty,
    )


def allocate_joint_assists(
    fgm: np.ndarray,
    assisted_rate: np.ndarray,
    creation_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Credit each assisted make to a teammate of the scorer.

    Returns ``(assists, pairs)`` where ``pairs[s, i, j]`` is the number of
    sim-``s`` baskets scored by ``j`` and assisted by ``i``. The scorer's
    own weight is removed before the draw, so the diagonal is zero.
    """
    made = np.asarray(fgm)
    if made.ndim != 2:
        raise ValueError("FGM must be a two-dimensional count matrix")
    n_sims, n_players = made.shape
    made = _as_count_matrix(fgm, n_sims, n_players, "FGM")
    assisted_rate = _as_unit_interval_vector(assisted_rate, n_players, "assisted_rate")
    creation = np.asarray(creation_weights, dtype=float)
    if creation.ndim == 1:
        creation_ok = creation.shape == (n_players,)
    elif creation.ndim == 2:
        creation_ok = creation.shape == (n_sims, n_players)
    else:
        creation_ok = False
    if (
        not creation_ok
        or np.any(~np.isfinite(creation))
        or np.any(creation < 0.0)
    ):
        raise ValueError("creation_weights must contain nonnegative finite values")

    pairs = np.zeros((n_sims, n_players, n_players), dtype=np.int64)
    for scorer in range(n_players):
        # A player cannot assist their own field goal. If nobody else has
        # creation weight, the make is unassisted and no draw is taken.
        eligible = creation.copy()
        if eligible.ndim == 1:
            eligible[scorer] = 0.0
            if float(eligible.sum()) <= 0.0:
                continue
            assisted = rng.binomial(
                made[:, scorer], float(assisted_rate[scorer])
            ).astype(np.int64)
            pairs[:, :, scorer] = multinomial_rows(
                assisted, renormalize_props(eligible), rng
            )
            continue
        eligible[:, scorer] = 0.0
        row_sum = eligible.sum(axis=1)
        assisted = rng.binomial(
            made[:, scorer], float(assisted_rate[scorer])
        ).astype(np.int64)
        assisted = np.where(row_sum > 0.0, assisted, 0)
        if np.all(assisted == 0):
            continue
        probs = np.zeros_like(eligible)
        positive = row_sum > 0.0
        probs[positive] = eligible[positive] / row_sum[positive, None]
        # Empty rows still need a legal distribution; their assisted count is 0.
        probs[~positive] = 1.0 / n_players
        pairs[:, :, scorer] = multinomial_rows(assisted, probs, rng)

    assists = pairs.sum(axis=2)
    diagonal = pairs[:, np.arange(n_players), np.arange(n_players)]
    _invariant(bool(np.all(diagonal == 0)), "a player assisted their own field goal")
    _invariant(
        bool(np.all(assists.sum(axis=1) <= made.sum(axis=1))),
        "assists exceed made field goals",
    )
    return assists, pairs


def _allocate(
    pool: np.ndarray,
    shares: np.ndarray,
    rng: np.random.Generator,
    stat: str,
) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if shares.ndim == 1:
        n_players = shares.shape[0]
        row_ok = np.array([float(shares.sum()) > 0.0])
        probs = shares
    else:
        n_players = shares.shape[1]
        row_sum = shares.sum(axis=1)
        row_ok = row_sum > 0.0
        probs = np.zeros_like(shares)
        probs[row_ok] = shares[row_ok] / row_sum[row_ok, None]
        if np.any(~row_ok):
            probs[~row_ok] = 1.0 / n_players
    allocated = np.zeros((pool.shape[0], n_players), dtype=np.int64)
    if np.all(pool == 0):
        return allocated
    if shares.ndim == 1 and not bool(row_ok[0]):
        raise ValueError(f"No eligible players to allocate {stat}")
    if shares.ndim == 2 and np.any((pool > 0) & ~row_ok):
        raise ValueError(f"No eligible players to allocate {stat}")
    if shares.ndim == 1:
        total = float(shares.sum())
        probs = shares if np.isclose(total, 1.0) else shares / total
    allocated = allocate_from_pool(pool, probs, rng)
    _invariant(
        bool(np.array_equal(allocated.sum(axis=1), pool)),
        f"{stat} allocation does not match its team pool",
    )
    return allocated


def _split_rebounds(
    misses: np.ndarray,
    p_rebound: float,
    p_oreb: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split missed shots into offensive rebounds, defensive rebounds, and dead balls.

    ``live ~ Binomial(misses, p_rebound)`` is the spec's team-rebound draw.
    ``oreb ~ Binomial(live, p_oreb)`` gives the offense its share; the rest
    of the live rebounds are defensive. Dead balls are the misses left over.
    """
    live = rng.binomial(misses, p_rebound).astype(np.int64)
    offensive = rng.binomial(live, p_oreb).astype(np.int64)
    defensive = live - offensive
    _invariant(
        bool(np.all(offensive + defensive == live)),
        "rebound split lost a live rebound",
    )
    _invariant(bool(np.all(live <= misses)), "live rebounds exceed missed shots")
    return offensive, defensive


def _block_pool(
    attempts: np.ndarray,
    misses: np.ndarray,
    p_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw blocks from shot attempts, then drop any that would block a make.

    A block is recorded on a field-goal attempt, and the attempt is missed.
    The binomial can otherwise exceed the misses already simulated upstream.
    """
    drawn = rng.binomial(attempts, p_block).astype(np.int64)
    return np.minimum(drawn, misses)


@dataclass(frozen=True)
class _PreparedTeam:
    name: str
    player_ids: tuple
    fga: np.ndarray
    fgm: np.ndarray
    tov: np.ndarray
    assisted_rate: np.ndarray
    oreb_shares: np.ndarray
    dreb_shares: np.ndarray
    creation_weights: np.ndarray
    stl_shares: np.ndarray
    blk_shares: np.ndarray


def _prepare_team(name: str, raw: dict, n_sims: int) -> _PreparedTeam:
    missing = [field for field in _REQUIRED_TEAM_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"{name} missing {missing}")

    player_ids = tuple(raw["player_ids"])
    n_players = len(player_ids)
    if n_players == 0:
        raise ValueError(f"{name} must include at least one player")

    fga = _as_count_matrix(raw["fga"], n_sims, n_players, "FGA")
    fgm = _as_count_matrix(raw["fgm"], n_sims, n_players, "FGM")
    tov = _as_count_matrix(raw["tov"], n_sims, n_players, "TOV")
    if np.any(fgm > fga):
        raise ValueError("FGM cannot exceed FGA")

    available = np.asarray(raw["available"], dtype=float)
    if available.shape != (n_players,) or np.any(
        (available != 0.0) & (available != 1.0)
    ):
        raise ValueError("available must contain only 0 or 1")
    if np.any((available == 0.0) & ((fga > 0) | (fgm > 0) | (tov > 0))):
        raise ValueError("unavailable players cannot record FGA, FGM, or TOV")

    assisted_rate = _as_unit_interval_vector(
        raw["assisted_rate"], n_players, "assisted_rate"
    )
    minutes_exponent = raw.get("minutes_exponent", 1.0)
    role = raw.get("role_factors")
    lineup = raw.get("lineup_factors")
    shot_profile = raw.get("opponent_shot_factors")
    matchup = raw.get("matchup_factors")

    def shares(rates: Any, *, context: Any = None, use_role: bool = True) -> np.ndarray:
        return event_shares(
            rates=rates,
            available=available,
            projected_minutes=raw["projected_minutes"],
            reference_minutes=raw["reference_minutes"],
            role_factors=role if use_role else None,
            lineup_factors=lineup,
            context_factors=context,
            minutes_exponent=minutes_exponent,
            allow_empty=True,
        )

    return _PreparedTeam(
        name=name,
        player_ids=player_ids,
        fga=fga,
        fgm=fgm,
        tov=tov,
        assisted_rate=assisted_rate,
        oreb_shares=shares(raw["oreb_rate"], context=shot_profile),
        dreb_shares=shares(raw["dreb_rate"], context=shot_profile),
        # Creation is not scaled by the rebound/defense role factor. A bigger
        # rebounding role should not hand the same player more assists.
        creation_weights=shares(raw["creation_rate"], use_role=False),
        stl_shares=shares(raw["stl_rate"], context=matchup),
        blk_shares=shares(raw["blk_rate"], context=matchup),
    )


@dataclass(frozen=True)
class TeamSharedStats:
    player_ids: tuple
    stats: dict[str, np.ndarray]
    ast_pairs: np.ndarray


@dataclass(frozen=True)
class SharedEventResult:
    teams: dict[str, TeamSharedStats]
    metadata: dict[str, dict[str, np.ndarray]]


def _empty_stats(n_sims: int, n_players: int) -> dict[str, np.ndarray]:
    return {stat: np.zeros((n_sims, n_players), dtype=np.int64) for stat in _STATS}


def simulate_shared_events(
    teams: dict[str, dict],
    n_sims: int,
    rng: np.random.Generator,
    *,
    p_rebound: float = 0.90,
    p_oreb: float = 0.27,
    p_steal: float = 0.55,
    p_block: float = 0.06,
) -> SharedEventResult:
    """Allocate one game's shared events to both teams.

    Team order follows ``teams`` insertion order and is part of the random
    stream. Defaults are typical NBA magnitudes, not estimates from a sample:
    most misses produce a rebound, about 27% of live rebounds are offensive,
    a bit over half of turnovers are steals, and about 6% of attempts are blocks.
    """
    if (
        isinstance(n_sims, bool)
        or not isinstance(n_sims, (int, np.integer))
        or n_sims <= 0
    ):
        raise ValueError("n_sims must be a positive integer")
    if len(teams) != 2:
        raise ValueError("shared events require exactly two teams")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")

    p_rebound = _as_probability(p_rebound, "p_rebound")
    p_oreb = _as_probability(p_oreb, "p_oreb")
    p_steal = _as_probability(p_steal, "p_steal")
    p_block = _as_probability(p_block, "p_block")

    prepared = [_prepare_team(name, raw, int(n_sims)) for name, raw in teams.items()]
    stats = {
        team.name: _empty_stats(int(n_sims), len(team.player_ids))
        for team in prepared
    }
    pairs = {
        team.name: np.zeros(
            (int(n_sims), len(team.player_ids), len(team.player_ids)),
            dtype=np.int64,
        )
        for team in prepared
    }
    metadata: dict[str, dict[str, np.ndarray]] = {
        "oreb_pool": {},
        "dreb_pool": {},
        "stl_pool": {},
        "blk_pool": {},
    }

    for index, offense in enumerate(prepared):
        defense = prepared[1 - index]
        misses = offense.fga.sum(axis=1) - offense.fgm.sum(axis=1)
        oreb_pool, dreb_pool = _split_rebounds(misses, p_rebound, p_oreb, rng)
        stats[offense.name]["OREB"] = _allocate(
            oreb_pool, offense.oreb_shares, rng, "OREB"
        )
        stats[defense.name]["DREB"] = _allocate(
            dreb_pool, defense.dreb_shares, rng, "DREB"
        )
        metadata["oreb_pool"][offense.name] = oreb_pool
        metadata["dreb_pool"][defense.name] = dreb_pool

    for team in prepared:
        assists, team_pairs = allocate_joint_assists(
            team.fgm,
            team.assisted_rate,
            team.creation_weights,
            rng,
        )
        stats[team.name]["AST"] = assists
        pairs[team.name] = team_pairs

    for index, offense in enumerate(prepared):
        defense = prepared[1 - index]
        steal_pool = rng.binomial(offense.tov.sum(axis=1), p_steal).astype(np.int64)
        stats[defense.name]["STL"] = _allocate(
            steal_pool, defense.stl_shares, rng, "STL"
        )
        metadata["stl_pool"][defense.name] = steal_pool

        misses = offense.fga.sum(axis=1) - offense.fgm.sum(axis=1)
        block_pool = _block_pool(offense.fga.sum(axis=1), misses, p_block, rng)
        stats[defense.name]["BLK"] = _allocate(
            block_pool, defense.blk_shares, rng, "BLK"
        )
        metadata["blk_pool"][defense.name] = block_pool

    result_teams: dict[str, TeamSharedStats] = {}
    for team in prepared:
        team_stats = stats[team.name]
        team_stats["REB"] = team_stats["OREB"] + team_stats["DREB"]
        result_teams[team.name] = TeamSharedStats(
            player_ids=team.player_ids,
            stats=team_stats,
            ast_pairs=pairs[team.name],
        )
    return SharedEventResult(teams=result_teams, metadata=metadata)


EVENT_RATES = (
    "ast_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
    "assisted_rate",
    "p_rebound",
    "p_oreb",
    "p_steal",
    "p_block",
)
_PLAYER_EVENT_RATES = EVENT_RATES[:5]
_TEAM_EVENT_RATES = EVENT_RATES[5:]
_EVENT_COLUMNS = ("ast", "oreb", "dreb", "stl", "blk")
_EVENT_TEAM_COLUMNS = (
    "team_fga",
    "team_fgm",
    "team_ast",
    "team_oreb",
    "team_stl",
    "team_blk",
    "opp_dreb",
    "opp_tov",
    "opp_fga",
)
_EVENT_IDS = ("player_id", "game_id", "game_date", "team_id")
_SHOT_KEYS = ("FGA", "FGM", "TOV")


@dataclass(frozen=True)
class EventModel:
    """League fallback for a player or team with no prior window."""

    ast_per_min: float
    oreb_per_min: float
    dreb_per_min: float
    stl_per_min: float
    blk_per_min: float
    assisted_rate: float
    p_rebound: float
    p_oreb: float
    p_steal: float
    p_block: float


def training_events(logs: pd.DataFrame) -> pd.DataFrame:
    """One player-game. Event rates stop before that game."""
    frame = _prepare_event_logs(logs)
    players = _player_event_rates(frame, "before")
    teams = _team_event_rates(frame, "before")
    merged = players.merge(
        teams[["game_id", "team_id", *_TEAM_EVENT_RATES]],
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    columns = [*_EVENT_IDS, *EVENT_RATES]
    return merged.loc[:, columns].reset_index(drop=True)


def live_events(logs: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """One upcoming player. The latest completed game stays in the rates."""
    prepared = _prepare_event_logs(logs)
    upcoming = _prepare_event_slate(slate)
    parts = []
    for date, rows in upcoming.groupby("game_date", sort=False):
        prior = prepared.loc[prepared["game_date"] < date].copy()
        parts.append(_live_event_rows(prior, rows.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


def fit_events(roles: pd.DataFrame) -> EventModel:
    """Median rate from rows that already have a window."""
    if not isinstance(roles, pd.DataFrame) or roles.empty:
        raise ValueError("roles must be a non-empty DataFrame")
    missing = [column for column in EVENT_RATES if column not in roles.columns]
    if missing:
        raise ValueError(f"roles need {missing}")
    fallback = {}
    for column in EVENT_RATES:
        values = pd.to_numeric(roles[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"no usable {column}")
        fallback[column] = float(np.median(finite))
    for column in _TEAM_EVENT_RATES:
        if not 0.0 <= fallback[column] <= 1.0:
            raise ValueError(f"{column} must be in [0, 1]")
    for column in _PLAYER_EVENT_RATES:
        if fallback[column] < 0.0:
            raise ValueError(f"{column} must be nonnegative")
    return EventModel(**fallback)


def allocate_events(
    model: EventModel,
    profiles: dict[str, pd.DataFrame],
    shots: dict[str, dict[str, np.ndarray]],
    minutes: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """Credit both teams for events the shot draw already produced."""
    if not isinstance(model, EventModel):
        raise ValueError("model must be an EventModel")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    if (
        not isinstance(profiles, dict)
        or not isinstance(shots, dict)
        or not isinstance(minutes, dict)
        or set(profiles) != set(shots)
        or set(profiles) != set(minutes)
    ):
        raise ValueError("profiles, shots, and minutes must name the same teams")
    if len(profiles) != 2:
        raise ValueError("shared events require exactly two teams")

    sides = [
        _prepare_side(name, profiles[name], shots[name], minutes[name], model)
        for name in profiles
    ]
    stats = {
        side.name: _empty_stats(side.fga.shape[0], side.fga.shape[1]) for side in sides
    }
    for index, offense in enumerate(sides):
        defense = sides[1 - index]
        misses = offense.fga.sum(axis=1) - offense.fgm.sum(axis=1)
        oreb_pool, dreb_pool = _split_rebounds(
            misses, offense.p_rebound, offense.p_oreb, rng
        )
        stats[offense.name]["OREB"] = _allocate(
            oreb_pool, offense.oreb_shares, rng, "OREB"
        )
        stats[defense.name]["DREB"] = _allocate(
            dreb_pool, defense.dreb_shares, rng, "DREB"
        )
        steal_pool = rng.binomial(offense.tov.sum(axis=1), defense.p_steal)
        stats[defense.name]["STL"] = _allocate(
            steal_pool.astype(np.int64), defense.stl_shares, rng, "STL"
        )
        block_pool = _block_pool(
            offense.fga.sum(axis=1), misses, defense.p_block, rng
        )
        stats[defense.name]["BLK"] = _allocate(
            block_pool, defense.blk_shares, rng, "BLK"
        )
    for side in sides:
        assists, _pairs = allocate_joint_assists(
            side.fgm, side.assisted_rate, side.creation, rng
        )
        stats[side.name]["AST"] = assists
        stats[side.name]["REB"] = stats[side.name]["OREB"] + stats[side.name]["DREB"]
    return stats


@dataclass(frozen=True)
class _Side:
    name: str
    fga: np.ndarray
    fgm: np.ndarray
    tov: np.ndarray
    assisted_rate: np.ndarray
    p_rebound: float
    p_oreb: float
    p_steal: float
    p_block: float
    creation: np.ndarray
    oreb_shares: np.ndarray
    dreb_shares: np.ndarray
    stl_shares: np.ndarray
    blk_shares: np.ndarray


def _prepare_side(name, profile, shot, played, model: EventModel) -> _Side:
    if not isinstance(profile, pd.DataFrame) or profile.empty:
        raise ValueError(f"{name} profile must be a non-empty DataFrame")
    minutes = np.asarray(played, dtype=float)
    if minutes.ndim != 2:
        raise ValueError(f"{name} minutes must have shape (n_sims, n_players)")
    n_sims, n_players = minutes.shape
    if len(profile) != n_players:
        raise ValueError(f"{name} profile must have one row per minute column")
    if np.any(~np.isfinite(minutes)) or np.any(minutes < 0):
        raise ValueError(f"{name} minutes must be nonnegative and finite")
    if not isinstance(shot, dict):
        raise ValueError(f"{name} shots must include {_SHOT_KEYS}")
    missing = [key for key in _SHOT_KEYS if key not in shot]
    if missing:
        raise ValueError(f"{name} shots need {missing}")
    fga = _as_count_matrix(shot["FGA"], n_sims, n_players, "FGA")
    fgm = _as_count_matrix(shot["FGM"], n_sims, n_players, "FGM")
    tov = _as_count_matrix(shot["TOV"], n_sims, n_players, "TOV")
    if np.any(fgm > fga):
        raise ValueError("FGM cannot exceed FGA")
    rates = {
        column: _filled_event_rate(profile, column, getattr(model, column))
        for column in EVENT_RATES
    }
    _reject_bad_event_rates(rates)
    assisted = _shared_event_rate(rates["assisted_rate"], "assisted_rate")
    return _Side(
        name=name,
        fga=fga,
        fgm=fgm,
        tov=tov,
        assisted_rate=np.full(n_players, assisted),
        p_rebound=_shared_event_rate(rates["p_rebound"], "p_rebound"),
        p_oreb=_shared_event_rate(rates["p_oreb"], "p_oreb"),
        p_steal=_shared_event_rate(rates["p_steal"], "p_steal"),
        p_block=_shared_event_rate(rates["p_block"], "p_block"),
        creation=rates["ast_per_min"][None, :] * minutes,
        oreb_shares=_event_shares(rates["oreb_per_min"], minutes),
        dreb_shares=_event_shares(rates["dreb_per_min"], minutes),
        stl_shares=_event_shares(rates["stl_per_min"], minutes),
        blk_shares=_event_shares(rates["blk_per_min"], minutes),
    )


def _prepare_event_logs(logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(logs, pd.DataFrame) or logs.empty:
        raise ValueError("logs must be a non-empty DataFrame")
    required = (*_EVENT_IDS, *_EVENT_COLUMNS, *_EVENT_TEAM_COLUMNS)
    missing = [column for column in required if column not in logs.columns]
    if "min" not in logs.columns and "minutes" not in logs.columns:
        missing.append("min")
    if missing:
        raise ValueError(f"logs need {missing}")
    frame = logs.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    if frame["game_date"].isna().any():
        raise ValueError("game_date has unparseable values")
    frame["game_id"] = frame["game_id"].astype(str)
    frame["minutes"] = _minutes(frame)
    _reject_minute_conflicts(frame)
    for column in (*_EVENT_COLUMNS, *_EVENT_TEAM_COLUMNS):
        frame[column] = _event_nonnegative(frame[column], column)
    _reject_event_bounds(frame)
    keys = ["game_id", "player_id", "team_id"]
    counted = (*_EVENT_COLUMNS, *_EVENT_TEAM_COLUMNS, "minutes")
    for column in counted:
        counts = frame.groupby(keys, dropna=False)[column].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(f"logs have conflicting {column} for one player-game")
    for column in _EVENT_TEAM_COLUMNS:
        counts = frame.groupby(["game_id", "team_id"], dropna=False)[column].nunique(
            dropna=False
        )
        if (counts > 1).any():
            raise ValueError(f"logs have conflicting {column} for one team-game")
    return frame.drop_duplicates(keys).reset_index(drop=True)


def _prepare_event_slate(slate: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(slate, pd.DataFrame) or slate.empty:
        raise ValueError("slate must be a non-empty DataFrame")
    needed = ("player_id", "game_date", "team_id")
    missing = [column for column in needed if column not in slate.columns]
    if missing:
        raise ValueError(f"slate needs {missing}")
    out = slate.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    if out["game_date"].isna().any():
        raise ValueError("slate game_date has unparseable values")
    if out.duplicated(["player_id", "game_date"]).any():
        raise ValueError("slate has two rows for one player on one date")
    return out.reset_index(drop=True)


def _player_event_rates(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    work = frame.sort_values(["player_id", "game_date", "game_id"])
    work = work.reset_index(drop=True)
    sources = {
        "ast_per_min": _event_ratio(work["ast"], work["minutes"]),
        "oreb_per_min": _event_ratio(work["oreb"], work["minutes"]),
        "dreb_per_min": _event_ratio(work["dreb"], work["minutes"]),
        "stl_per_min": _event_ratio(work["stl"], work["minutes"]),
        "blk_per_min": _event_ratio(work["blk"], work["minutes"]),
    }
    for name, game_rate in sources.items():
        work[f"_{name}"] = game_rate
        work[name] = _event_window(work, "player_id", f"_{name}", window)
    return work


def _team_event_rates(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    teams = (
        frame.sort_values(["team_id", "game_date", "game_id", "player_id"])
        .groupby(["game_id", "team_id"], as_index=False)
        .agg(
            game_date=("game_date", "first"),
            team_fga=("team_fga", "first"),
            team_fgm=("team_fgm", "first"),
            team_ast=("team_ast", "first"),
            team_oreb=("team_oreb", "first"),
            team_stl=("team_stl", "first"),
            team_blk=("team_blk", "first"),
            opp_dreb=("opp_dreb", "first"),
            opp_tov=("opp_tov", "first"),
            opp_fga=("opp_fga", "first"),
        )
        .sort_values(["team_id", "game_date", "game_id"])
        .reset_index(drop=True)
    )
    misses = teams["team_fga"] - teams["team_fgm"]
    live = teams["team_oreb"] + teams["opp_dreb"]
    # Missed free throws are in the rebound count, so a field-goal miss can
    # look like it produced more than one rebound. Cap the rate at one.
    teams["_p_rebound"] = _event_ratio(live, misses).clip(upper=1.0)
    teams["_p_oreb"] = _event_ratio(teams["team_oreb"], live)
    teams["_assisted_rate"] = _event_ratio(teams["team_ast"], teams["team_fgm"])
    teams["_p_steal"] = _event_ratio(teams["team_stl"], teams["opp_tov"])
    teams["_p_block"] = _event_ratio(teams["team_blk"], teams["opp_fga"])
    for name in _TEAM_EVENT_RATES:
        teams[name] = _event_window(teams, "team_id", f"_{name}", window)
    return teams


def _live_event_rows(prior: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    base = slate.loc[:, ["player_id", "game_date", "team_id"]].copy()
    if prior.empty:
        for column in EVENT_RATES:
            base[column] = np.nan
        return base
    players = _player_event_rates(prior, "through")
    last = players.groupby("player_id", sort=False).tail(1)
    teams = _team_event_rates(prior, "through")
    last_team = teams.groupby("team_id", sort=False).tail(1)
    out = base.merge(
        last[["player_id", *_PLAYER_EVENT_RATES]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        last_team[["team_id", *_TEAM_EVENT_RATES]],
        on="team_id",
        how="left",
        validate="many_to_one",
    )
    return out.loc[:, ["player_id", "game_date", "team_id", *EVENT_RATES]]


def _event_window(frame, entity: str, column: str, window: str) -> np.ndarray:
    if window == "before":
        return _before(frame, entity, column, "mean", 1)
    if window == "through":
        return _through(frame, entity, column, "mean", 1)
    raise ValueError(f"unknown window {window}")


def _event_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=numer.index, dtype=float)
    ok = numer.notna() & denom.notna() & (denom > 0)
    out.loc[ok] = numer.loc[ok].to_numpy() / denom.loc[ok].to_numpy()
    return out


def _event_nonnegative(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    if (present & numeric.isna()).any():
        raise ValueError(f"{name} must be numeric")
    if (numeric < 0).fillna(False).any():
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _reject_event_bounds(frame: pd.DataFrame) -> None:
    pairs = (
        ("team_fgm", "team_fga"),
        ("team_ast", "team_fgm"),
        ("team_stl", "opp_tov"),
        ("team_blk", "opp_fga"),
    )
    for made, attempts in pairs:
        both = frame[made].notna() & frame[attempts].notna()
        if (frame.loc[both, made] > frame.loc[both, attempts]).any():
            raise ValueError(f"{made} cannot exceed {attempts}")


def _filled_event_rate(
    profile: pd.DataFrame, column: str, fallback: float
) -> np.ndarray:
    if column not in profile.columns:
        return np.full(len(profile), fallback, dtype=float)
    values = pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    values = values.copy()
    values[missing] = fallback
    return values


def _reject_bad_event_rates(rates: dict[str, np.ndarray]) -> None:
    for column in _TEAM_EVENT_RATES:
        if np.any((rates[column] < 0.0) | (rates[column] > 1.0)):
            raise ValueError(f"{column} must be in [0, 1]")
    for column in _PLAYER_EVENT_RATES:
        if np.any(rates[column] < 0.0):
            raise ValueError(f"{column} must be nonnegative")


def _shared_event_rate(values: np.ndarray, name: str) -> float:
    if float(np.max(values) - np.min(values)) > 1e-8:
        raise ValueError(f"players on one team must share one {name}")
    return float(values[0])


def _event_shares(per_minute: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    weights = per_minute[None, :] * minutes
    total = weights.sum(axis=1, keepdims=True)
    shares = np.zeros_like(weights)
    positive = total.ravel() > 0.0
    shares[positive] = weights[positive] / total[positive]
    return shares
