"""Explicit-probability draw used by the box score and by the fitted path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from game_events.checks import (
    _as_count_matrix,
    _as_factor,
    _as_probability,
    _as_unit_interval_vector,
    _invariant,
    _normalize_ids,
)
from game_events.constants import (
    _REQUIRED_TEAM_FIELDS,
    _STATS,
    _STREAM_PURPOSES,
    _UPSTREAM_STATS,
    _WEIGHTED_STATS,
)
from game_events.draws import _allocate, _overdispersed_binomial, _streams
from game_events.shares import _rebound_pools, allocate_joint_assists, event_shares


@dataclass(frozen=True)
class _Side:
    name: str
    player_ids: tuple
    fga: np.ndarray
    fgm: np.ndarray
    tov: np.ndarray
    assisted_rate: np.ndarray
    weights: dict[str, np.ndarray]
    oreb: np.ndarray | None = None
    ast: np.ndarray | None = None
    rates: dict[str, float] | None = None
    minutes: np.ndarray | None = None
    # Rate times factor before exposure; segments turn this into weights.
    intensity: dict[str, np.ndarray] | None = None


def _empty_stats(n_sims: int, n_players: int) -> dict[str, np.ndarray]:
    return {stat: np.zeros((n_sims, n_players), dtype=np.int64) for stat in _STATS}


def _draw_events(
    sides: list[_Side],
    probs: list[dict[str, Any]],
    rng: np.random.Generator,
    pool_dispersion: Mapping[str, float] | None = None,
    share_concentration: Mapping[str, float | None] | None = None,
) -> tuple[dict, dict, dict]:
    """Draw every event pool and allocate it. ``probs[i]`` is offense ``i``."""
    pool_dispersion = pool_dispersion or {}
    share_concentration = share_concentration or {}
    streams = _streams(
        rng, [(purpose, i) for i in range(2) for purpose in _STREAM_PURPOSES]
    )
    n_sims = sides[0].fga.shape[0]
    stats = {side.name: _empty_stats(n_sims, len(side.player_ids)) for side in sides}
    pools: dict[str, dict[str, np.ndarray]] = {stat: {} for stat in _WEIGHTED_STATS}
    sources: dict[str, dict[str, str]] = {side.name: {} for side in sides}
    pairs: dict[str, np.ndarray | None] = {}

    for index, offense in enumerate(sides):
        defense = sides[1 - index]
        d = 1 - index
        p = probs[index]
        misses = offense.fga.sum(axis=1) - offense.fgm.sum(axis=1)
        upstream = None if offense.oreb is None else offense.oreb.sum(axis=1)
        oreb_pool, dreb_pool = _rebound_pools(
            misses,
            upstream,
            p["p_rebound"],
            p["p_oreb"],
            streams[("rebounds", index)],
            pool_dispersion,
        )
        if offense.oreb is None:
            stats[offense.name]["OREB"] = _allocate(
                oreb_pool,
                offense.weights["OREB"],
                streams[("OREB", index)],
                "OREB",
                share_concentration.get("OREB"),
            )
            sources[offense.name]["OREB"] = "layer5"
        else:
            stats[offense.name]["OREB"] = offense.oreb
            sources[offense.name]["OREB"] = "layer4"
        stats[defense.name]["DREB"] = _allocate(
            dreb_pool,
            defense.weights["DREB"],
            streams[("DREB", d)],
            "DREB",
            share_concentration.get("DREB"),
        )

        steal_pool = _overdispersed_binomial(
            offense.tov.sum(axis=1),
            p["p_steal"],
            pool_dispersion.get("p_steal", 0.0),
            streams[("steals", d)],
        )
        stats[defense.name]["STL"] = _allocate(
            steal_pool,
            defense.weights["STL"],
            streams[("STL", d)],
            "STL",
            share_concentration.get("STL"),
        )
        # A block is recorded on a missed field-goal attempt.
        block_pool = _overdispersed_binomial(
            misses,
            p["p_block"],
            pool_dispersion.get("p_block", 0.0),
            streams[("blocks", d)],
        )
        stats[defense.name]["BLK"] = _allocate(
            block_pool,
            defense.weights["BLK"],
            streams[("BLK", d)],
            "BLK",
            share_concentration.get("BLK"),
        )

        pools["OREB"][offense.name] = oreb_pool
        pools["DREB"][defense.name] = dreb_pool
        pools["STL"][defense.name] = steal_pool
        pools["BLK"][defense.name] = block_pool

    for index, side in enumerate(sides):
        if side.ast is None:
            stream = streams[("AST", index)]
            rate = side.assisted_rate
            rho = pool_dispersion.get("assisted_rate", 0.0)
            base = float(rate.mean())
            if rho > 0.0 and 0.0 < base < 1.0:
                # One shared assisted-rate draw per game keeps scorers coherent.
                strength = (1.0 - rho) / rho
                game_rate = stream.beta(
                    base * strength, (1.0 - base) * strength, size=n_sims
                )
                rate = np.clip(rate[None, :] * (game_rate / base)[:, None], 0.0, 1.0)
            assists, team_pairs = allocate_joint_assists(
                side.fgm, rate, side.weights["AST"], stream
            )
            sources[side.name]["AST"] = "layer5"
        else:
            assists, team_pairs = side.ast, None
            sources[side.name]["AST"] = "layer4"
        team_stats = stats[side.name]
        team_stats["AST"] = assists
        team_stats["REB"] = team_stats["OREB"] + team_stats["DREB"]
        pools["AST"][side.name] = assists.sum(axis=1)
        pairs[side.name] = team_pairs

    for index, offense in enumerate(sides):
        defense = sides[1 - index]
        misses = offense.fga.sum(axis=1) - offense.fgm.sum(axis=1)
        offense_reb = stats[offense.name]["OREB"].sum(axis=1)
        defense_reb = stats[defense.name]["DREB"].sum(axis=1)
        _invariant(
            bool(np.all(offense_reb + defense_reb <= misses)),
            "rebounds exceed missed shots",
        )
        _invariant(
            bool(np.all(stats[defense.name]["BLK"].sum(axis=1) <= misses)),
            "blocks exceed missed shots",
        )
        _invariant(
            bool(
                np.all(
                    stats[defense.name]["STL"].sum(axis=1) <= offense.tov.sum(axis=1)
                )
            ),
            "steals exceed turnovers",
        )
    return stats, {"pools": pools, "sources": sources}, pairs


def _validate_upstream(
    name: str,
    key: str,
    value: np.ndarray,
    fga: np.ndarray,
    fgm: np.ndarray,
) -> None:
    if key == "AST":
        teammates_made = fgm.sum(axis=1, keepdims=True) - fgm
        if np.any(value > teammates_made):
            raise ValueError(f"{name} AST exceeds teammates' made field goals")
    if key == "OREB":
        misses = fga.sum(axis=1) - fgm.sum(axis=1)
        if np.any(value.sum(axis=1) > misses):
            raise ValueError(f"{name} OREB exceeds missed field goals")


def _reject_shared_players(sides: list[_Side]) -> None:
    shared = set(sides[0].player_ids) & set(sides[1].player_ids)
    if shared:
        raise ValueError(f"players appear on both teams: {sorted(shared)}")


@dataclass(frozen=True)
class TeamSharedStats:
    player_ids: tuple
    stats: dict[str, np.ndarray]
    # ``None`` when assists came from Layer 4, which supplies no scorer pairs.
    ast_pairs: np.ndarray | None


@dataclass(frozen=True)
class SharedEventResult:
    teams: dict[str, TeamSharedStats]
    metadata: dict[str, dict[str, np.ndarray]]


def _prepare_team(name: str, raw: dict, n_sims: int) -> _Side:
    missing = [field for field in _REQUIRED_TEAM_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"{name} missing {missing}")

    player_ids = tuple(_normalize_ids(list(raw["player_ids"]), f"{name} player_ids"))
    n_players = len(player_ids)
    if n_players == 0:
        raise ValueError(f"{name} must include at least one player")
    if len(set(player_ids)) != n_players:
        raise ValueError(f"{name} player_ids must be unique")

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
    upstream = {
        key: _as_count_matrix(raw[key.lower()], n_sims, n_players, key)
        for key in _UPSTREAM_STATS
        if key.lower() in raw
    }
    for key, value in {"FGA": fga, "FGM": fgm, "TOV": tov, **upstream}.items():
        if np.any((available == 0.0) & (value > 0)):
            raise ValueError(f"unavailable players cannot record {key}")
    for key, value in upstream.items():
        _validate_upstream(name, key, value, fga, fgm)

    assisted_rate = _as_unit_interval_vector(
        raw["assisted_rate"], n_players, "assisted_rate"
    )
    minutes_exponent = raw.get("minutes_exponent", 1.0)
    role = raw.get("role_factors")
    lineup = raw.get("lineup_factors")
    shot_profile = raw.get("opponent_shot_factors")
    matchup = raw.get("matchup_factors")

    def shares(rates: Any, *, context: Any = None, use_role: bool = True) -> np.ndarray:
        values = event_shares(
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
        return _as_factor(values, n_sims, n_players, f"{name} shares")

    return _Side(
        name=name,
        player_ids=player_ids,
        fga=fga,
        fgm=fgm,
        tov=tov,
        assisted_rate=assisted_rate,
        weights={
            "OREB": shares(raw["oreb_rate"], context=shot_profile),
            "DREB": shares(raw["dreb_rate"], context=shot_profile),
            # Creation is not scaled by the rebound/defense role factor. A
            # bigger rebounding role should not hand the same player more
            # assists.
            "AST": shares(raw["creation_rate"], use_role=False),
            "STL": shares(raw["stl_rate"], context=matchup),
            "BLK": shares(raw["blk_rate"], context=matchup),
        },
        oreb=upstream.get("OREB"),
        ast=upstream.get("AST"),
    )


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
    """Allocate one game's shared events with explicit probabilities.

    Team order follows ``teams`` insertion order and is part of the random
    stream. Defaults are typical NBA magnitudes, not estimates from a sample.
    ``p_block`` is per field-goal attempt; it is converted to a per-miss
    probability in each simulation so blocks never exceed misses. Optional
    ``oreb`` and ``ast`` matrices are reused instead of drawn. There is no
    fitted dispersion on this path.
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

    sides = [_prepare_team(name, raw, int(n_sims)) for name, raw in teams.items()]
    _reject_shared_players(sides)
    probs = []
    for offense in sides:
        attempts = offense.fga.sum(axis=1)
        misses = attempts - offense.fgm.sum(axis=1)
        per_miss = np.divide(
            p_block * attempts,
            misses,
            out=np.zeros(attempts.shape, dtype=float),
            where=misses > 0,
        )
        probs.append(
            {
                "p_rebound": p_rebound,
                "p_oreb": p_oreb,
                "p_steal": p_steal,
                "p_block": np.minimum(per_miss, 1.0),
            }
        )
    stats, details, pairs = _draw_events(sides, probs, rng)
    metadata = {
        "oreb_pool": details["pools"]["OREB"],
        "dreb_pool": details["pools"]["DREB"],
        "stl_pool": details["pools"]["STL"],
        "blk_pool": details["pools"]["BLK"],
        "ast_pool": details["pools"]["AST"],
        "sources": details["sources"],
    }
    result_teams = {
        side.name: TeamSharedStats(
            player_ids=side.player_ids,
            stats=stats[side.name],
            ast_pairs=pairs[side.name],
        )
        for side in sides
    }
    return SharedEventResult(teams=result_teams, metadata=metadata)
