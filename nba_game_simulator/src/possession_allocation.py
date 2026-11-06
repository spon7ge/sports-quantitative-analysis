"""Layer 4: allocate team offensive opportunities to players."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opportunity_shares import opportunity_shares


def as_nonneg_int_vector(value: Any, n_sims: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    is_scalar = arr.ndim == 0
    if is_scalar:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
        if arr.shape[0] != n_sims:
            raise ValueError(f"{name} must be scalar or length n_sims={n_sims}")

    try:
        numeric = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative integers") from exc
    int64_info = np.iinfo(np.int64)
    if (
        np.any(~np.isfinite(numeric))
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric < 0)
        or np.any(numeric > int64_info.max)
    ):
        raise ValueError(f"{name} must be nonnegative integers")
    integers = numeric.astype(np.int64)
    if is_scalar:
        return np.full(n_sims, integers[0], dtype=np.int64)
    return integers


_PLAYER_ALIGNED_FIELDS = (
    "base_weights",
    "available",
    "projected_minutes",
    "reference_minutes",
    "p_3pa",
    "p_3p",
    "p_2p",
    "fta_per_fga",
    "ft_pct",
    "tov_rate",
    "assist_propensity",
    "oreb_propensity",
)
_OPTIONAL_PLAYER_ALIGNED_FIELDS = ("role_factors", "lineup_factors")
_PROBABILITY_FIELDS = ("p_3pa", "p_3p", "p_2p", "ft_pct")
_NONNEGATIVE_PLAYER_FIELDS = (
    "fta_per_fga",
    "tov_rate",
    "assist_propensity",
    "oreb_propensity",
)
_DISPERSION_FIELDS = ("fta", "ast", "oreb", "tov")


def _as_finite_float_array(value: Any, field: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite numeric values") from exc
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{field} must contain finite numeric values")
    return arr


def _validate_player_inputs(
    players: dict[str, Any],
    n_players: int,
    team_name: str,
) -> None:
    for field in _PLAYER_ALIGNED_FIELDS:
        value = np.asarray(players[field])
        if value.ndim != 1 or value.shape[0] != n_players:
            raise ValueError(
                f"{field} for {team_name!r} must have length {n_players}"
            )
    for field in _OPTIONAL_PLAYER_ALIGNED_FIELDS:
        if players.get(field) is None:
            continue
        value = np.asarray(players[field])
        if value.ndim != 1 or value.shape[0] != n_players:
            raise ValueError(
                f"{field} for {team_name!r} must have length {n_players}"
            )

    for field in _PROBABILITY_FIELDS:
        values = _as_finite_float_array(players[field], field)
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{field} must contain values in [0, 1]")
    for field in _NONNEGATIVE_PLAYER_FIELDS:
        values = _as_finite_float_array(players[field], field)
        if np.any(values < 0):
            raise ValueError(f"{field} must contain nonnegative values")

    for field in ("assist_rate", "oreb_rate"):
        value = _as_finite_float_array(
            players.get(field, 0.55 if field == "assist_rate" else 0.25),
            field,
        )
        if value.ndim != 0 or value < 0:
            raise ValueError(f"{field} must be a nonnegative scalar")

    dispersion = players.get("dispersion", {})
    for field in _DISPERSION_FIELDS:
        value = _as_finite_float_array(dispersion.get(field, 0.05), field)
        if value.ndim != 0 or value < 0:
            raise ValueError(f"dispersion {field} must be a nonnegative scalar")


def sample_negbin(
    mu: np.ndarray | float,
    alpha: float,
    rng: np.random.Generator,
    size: tuple[int, ...] | None = None,
) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float)
    if size is None:
        size = mu_arr.shape
    mu_arr = np.broadcast_to(mu_arr, size)
    if alpha <= 0:
        return rng.poisson(np.maximum(mu_arr, 0.0), size=size).astype(np.int64)
    # NB2: mean mu, var = mu + alpha * mu^2
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * np.maximum(mu_arr, 0.0))
    return rng.negative_binomial(n, p, size=size).astype(np.int64)


def multinomial_rows(
    totals: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    totals = np.asarray(totals, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be 1-d for v1 multinomial_rows")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("probs must sum to 1")
    out = np.zeros((totals.shape[0], probs.shape[0]), dtype=np.int64)
    for i, n in enumerate(totals):
        if n < 0:
            raise ValueError("totals must be nonnegative")
        out[i] = rng.multinomial(int(n), probs)
    return out


def allocate_and_shoot(
    team_fga: np.ndarray,
    shares: np.ndarray,
    p_3pa: np.ndarray,
    p_3p: np.ndarray,
    p_2p: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    fga = multinomial_rows(team_fga, shares, rng)
    n_sims, n_players = fga.shape
    three_pa = np.zeros_like(fga)
    two_pm = np.zeros_like(fga)
    three_pm = np.zeros_like(fga)
    for j in range(n_players):
        three_pa[:, j] = rng.binomial(fga[:, j], float(p_3pa[j]))
        three_pm[:, j] = rng.binomial(three_pa[:, j], float(p_3p[j]))
        two_pa_j = fga[:, j] - three_pa[:, j]
        two_pm[:, j] = rng.binomial(two_pa_j, float(p_2p[j]))
    two_pa = fga - three_pa
    fgm = two_pm + three_pm
    return {
        "FGA": fga,
        "2PA": two_pa,
        "2PM": two_pm,
        "3PA": three_pa,
        "3PM": three_pm,
        "FGM": fgm,
    }


def renormalize_props(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError("allocation weights must sum to a positive value")
    return w / total


def simulate_fta_ftm(
    fga: np.ndarray,
    fta_per_fga: np.ndarray,
    ft_pct: np.ndarray,
    alpha_fta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mu = fga.astype(float) * np.asarray(fta_per_fga, dtype=float)[None, :]
    fta = sample_negbin(mu, alpha_fta, rng, size=fga.shape)
    ftm = np.zeros_like(fta)
    for j in range(fga.shape[1]):
        ftm[:, j] = rng.binomial(fta[:, j], float(ft_pct[j]))
    return fta, ftm


def allocate_from_pool(
    pool_totals: np.ndarray,
    alloc_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return multinomial_rows(pool_totals, alloc_probs, rng)


@dataclass(frozen=True)
class TeamSimResult:
    player_ids: tuple
    opportunity_shares: np.ndarray
    stats: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    teams: dict[str, TeamSimResult]


def simulate_game(
    team_inputs: dict[str, dict],
    player_inputs: dict[str, dict],
    n_sims: int,
    rng: np.random.Generator,
) -> SimulationResult:
    """Simulate player box-score statistics from team opportunity pools."""
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    team_results: dict[str, TeamSimResult] = {}
    for team_name, team in team_inputs.items():
        if team_name not in player_inputs:
            raise ValueError(f"player_inputs missing team {team_name!r}")

        players = player_inputs[team_name]
        player_ids = tuple(players["player_ids"])
        _validate_player_inputs(players, len(player_ids), team_name)
        shares = opportunity_shares(
            base_weights=players["base_weights"],
            available=players["available"],
            projected_minutes=players["projected_minutes"],
            reference_minutes=players["reference_minutes"],
            role_factors=players.get("role_factors"),
            lineup_factors=players.get("lineup_factors"),
            minutes_exponent=players.get("minutes_exponent", 1.0),
        )
        available = np.asarray(players["available"], dtype=float)
        if len(player_ids) != shares.shape[0]:
            raise ValueError(
                f"player_ids for {team_name!r} must match opportunity inputs"
            )

        team_fga = as_nonneg_int_vector(team["team_fga"], n_sims, "team_fga")
        shot = allocate_and_shoot(
            team_fga=team_fga,
            shares=shares,
            p_3pa=players["p_3pa"],
            p_3p=players["p_3p"],
            p_2p=players["p_2p"],
            rng=rng,
        )

        dispersion = players.get("dispersion", {})
        fta, ftm = simulate_fta_ftm(
            fga=shot["FGA"],
            fta_per_fga=players["fta_per_fga"],
            ft_pct=players["ft_pct"],
            alpha_fta=dispersion.get("fta", 0.05),
            rng=rng,
        )

        team_fgm = shot["FGM"].sum(axis=1)
        team_miss = team_fga - team_fgm

        ast_draw = sample_negbin(
            team_fgm * players.get("assist_rate", 0.55),
            dispersion.get("ast", 0.05),
            rng,
        )
        ast_pool = np.minimum(ast_draw, team_fgm)
        if np.all(ast_pool == 0):
            ast = np.zeros((n_sims, len(player_ids)), dtype=np.int64)
        else:
            ast = allocate_from_pool(
                ast_pool,
                renormalize_props(
                    np.asarray(players["assist_propensity"], dtype=float)
                    * available
                ),
                rng,
            )

        oreb_pool = sample_negbin(
            team_miss * players.get("oreb_rate", 0.25),
            dispersion.get("oreb", 0.05),
            rng,
        )
        if np.all(oreb_pool == 0):
            oreb = np.zeros((n_sims, len(player_ids)), dtype=np.int64)
        else:
            oreb = allocate_from_pool(
                oreb_pool,
                renormalize_props(
                    np.asarray(players["oreb_propensity"], dtype=float)
                    * available
                ),
                rng,
            )

        metadata: dict[str, Any] = {}
        if "team_tov" in team:
            tov_pool = as_nonneg_int_vector(team["team_tov"], n_sims, "team_tov")
        else:
            possessions = as_nonneg_int_vector(
                team["possessions"], n_sims, "possessions"
            )
            weighted_tov_rate = float(
                np.sum(shares * np.asarray(players["tov_rate"], dtype=float))
            )
            tov_pool = sample_negbin(
                possessions * weighted_tov_rate,
                dispersion.get("tov", 0.05),
                rng,
            )
            metadata["tov_pool_approximate"] = True
        tov = allocate_from_pool(
            tov_pool,
            renormalize_props(
                shares * np.asarray(players["tov_rate"], dtype=float)
            ),
            rng,
        )

        pts = 2 * shot["2PM"] + 3 * shot["3PM"] + ftm
        stats = {
            "FGA": shot["FGA"],
            "FGM": shot["FGM"],
            "2PA": shot["2PA"],
            "2PM": shot["2PM"],
            "3PA": shot["3PA"],
            "3PM": shot["3PM"],
            "FTA": fta,
            "FTM": ftm,
            "PTS": pts,
            "AST": ast,
            "TOV": tov,
            "OREB": oreb,
        }
        team_results[team_name] = TeamSimResult(
            player_ids=player_ids,
            opportunity_shares=shares,
            stats=stats,
            metadata=metadata,
        )

    return SimulationResult(teams=team_results)
