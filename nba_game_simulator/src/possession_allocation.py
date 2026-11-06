"""Layer 4: allocate one team's shots and turnovers.

``simulate_game`` splits a supplied attempt pool. ``allocate_opportunities``
builds that pool from the possession count already drawn and the minutes
already assigned, using each player's recent shot profile. When the scenario
score is supplied, makes and free throws move so the points sum to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from opportunity_shares import opportunity_share_rows, opportunity_shares
from role_context import _before, _minutes, _reject_minute_conflicts, _through


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
    n_sims: int,
) -> None:
    for field in _PLAYER_ALIGNED_FIELDS:
        value = np.asarray(players[field])
        if field == "projected_minutes" and value.ndim == 2:
            if value.shape != (n_sims, n_players):
                raise ValueError(
                    f"projected_minutes for {team_name!r} must have shape "
                    f"({n_sims}, {n_players})"
                )
            continue
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
    if probs.ndim == 1:
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError("probs must sum to 1")
        row_probs = None
        n_players = probs.shape[0]
    elif probs.ndim == 2:
        if probs.shape[0] != totals.shape[0]:
            raise ValueError("probs must have one row per simulation")
        if not np.allclose(probs.sum(axis=1), 1.0):
            raise ValueError("each probs row must sum to 1")
        row_probs = probs
        n_players = probs.shape[1]
    else:
        raise ValueError("probs must be 1-d or shaped (n_sims, n_players)")
    out = np.zeros((totals.shape[0], n_players), dtype=np.int64)
    for i, n in enumerate(totals):
        if n < 0:
            raise ValueError("totals must be nonnegative")
        weights = probs if row_probs is None else row_probs[i]
        out[i] = rng.multinomial(int(n), weights)
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
        _validate_player_inputs(players, len(player_ids), team_name, n_sims)
        share_kwargs = dict(
            base_weights=players["base_weights"],
            available=players["available"],
            projected_minutes=players["projected_minutes"],
            reference_minutes=players["reference_minutes"],
            role_factors=players.get("role_factors"),
            lineup_factors=players.get("lineup_factors"),
            minutes_exponent=players.get("minutes_exponent", 1.0),
        )
        projected = np.asarray(players["projected_minutes"], dtype=float)
        if projected.ndim == 2:
            shares = opportunity_share_rows(**share_kwargs)
        else:
            shares = opportunity_shares(**share_kwargs)
        available = np.asarray(players["available"], dtype=float)
        share_players = shares.shape[0] if shares.ndim == 1 else shares.shape[1]
        if len(player_ids) != share_players:
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
            tov_rate = np.asarray(players["tov_rate"], dtype=float)
            if shares.ndim == 1:
                weighted_tov_rate = float(np.sum(shares * tov_rate))
                tov_mean = possessions * weighted_tov_rate
            else:
                tov_mean = possessions * np.sum(shares * tov_rate, axis=1)
            tov_pool = sample_negbin(
                tov_mean,
                dispersion.get("tov", 0.05),
                rng,
            )
            metadata["tov_pool_approximate"] = True
        tov_rate = np.asarray(players["tov_rate"], dtype=float)
        tov_weights = shares * tov_rate
        if tov_weights.ndim == 1:
            tov_probs = renormalize_props(tov_weights)
        else:
            row_total = tov_weights.sum(axis=1, keepdims=True)
            if np.any(row_total <= 0):
                raise ValueError("allocation weights must sum to a positive value")
            tov_probs = tov_weights / row_total
        tov = allocate_from_pool(tov_pool, tov_probs, rng)

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


OPPORTUNITY_RATES = (
    "fga_per_min",
    "p_3pa",
    "p_3p",
    "p_2p",
    "fta_per_fga",
    "ft_pct",
    "tov_per_min",
    "fga_per_poss",
    "tov_per_poss",
)
_PLAYER_RATES = (
    "fga_per_min",
    "p_3pa",
    "p_3p",
    "p_2p",
    "fta_per_fga",
    "ft_pct",
    "tov_per_min",
)
_TEAM_RATES = ("fga_per_poss", "tov_per_poss")
_PROBABILITY_RATES = ("p_3pa", "p_3p", "p_2p", "ft_pct")
_SHOT_COLUMNS = ("fga", "fgm", "fg3_a", "fg3_m", "fta", "ftm", "tov")
_TEAM_COLUMNS = ("team_fga", "team_poss", "team_tov")
_ID_COLUMNS = ("player_id", "game_id", "game_date", "team_id")
_MADE_WITHIN = (
    ("fgm", "fga"),
    ("fg3_a", "fga"),
    ("fg3_m", "fg3_a"),
    ("fg3_m", "fgm"),
    ("ftm", "fta"),
)


@dataclass(frozen=True)
class OpportunityModel:
    """League fallback for a player or team with no prior window."""

    fga_per_min: float
    p_3pa: float
    p_3p: float
    p_2p: float
    fta_per_fga: float
    ft_pct: float
    tov_per_min: float
    fga_per_poss: float
    tov_per_poss: float
    alpha: float


def training_opportunities(logs: pd.DataFrame) -> pd.DataFrame:
    """One player-game. Shot rates stop before that game."""
    frame = _prepare_opportunity_logs(logs)
    players = _player_rates(frame, "before")
    teams = _team_rates(frame, "before")
    merged = players.merge(
        teams[["game_id", "team_id", *_TEAM_RATES]],
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    return merged.loc[:, [*_ID_COLUMNS, *OPPORTUNITY_RATES]].reset_index(drop=True)


def live_opportunities(logs: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """One upcoming player. The latest completed game stays in the rates."""
    prepared = _prepare_opportunity_logs(logs)
    upcoming = _prepare_opportunity_slate(slate)
    parts = []
    for date, rows in upcoming.groupby("game_date", sort=False):
        prior = prepared.loc[prepared["game_date"] < date].copy()
        parts.append(_live_opportunity_rows(prior, rows.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


def fit_opportunities(roles: pd.DataFrame, alpha: float = 0.03) -> OpportunityModel:
    """Median rate from rows that already have a window. ``alpha`` is NegBin noise."""
    if not isinstance(roles, pd.DataFrame) or roles.empty:
        raise ValueError("roles must be a non-empty DataFrame")
    missing = [column for column in OPPORTUNITY_RATES if column not in roles.columns]
    if missing:
        raise ValueError(f"roles need {missing}")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a nonnegative finite number") from exc
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a nonnegative finite number")
    fallback = {}
    for column in OPPORTUNITY_RATES:
        values = pd.to_numeric(roles[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"no usable {column}")
        fallback[column] = float(np.median(finite))
    return OpportunityModel(alpha=alpha, **fallback)


def allocate_opportunities(
    model: OpportunityModel,
    profile: pd.DataFrame,
    minutes: np.ndarray,
    possessions: np.ndarray,
    rng: np.random.Generator,
    *,
    team_points=None,
) -> dict[str, np.ndarray]:
    """Shots and turnovers for one team. Possessions and minutes are already drawn.

    ``team_points`` is that side's scenario score. Makes and free throws move
    so the players' points sum to its nearest whole number. Attempts stay on
    the possession draw.
    """
    if not isinstance(model, OpportunityModel):
        raise ValueError("model must be an OpportunityModel")
    if not isinstance(profile, pd.DataFrame) or profile.empty:
        raise ValueError("profile must be a non-empty DataFrame")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    played = np.asarray(minutes, dtype=float)
    if played.ndim != 2:
        raise ValueError("minutes must have shape (n_sims, n_players)")
    n_sims, n_players = played.shape
    if len(profile) != n_players:
        raise ValueError("profile must have one row per minute column")
    if np.any(~np.isfinite(played)) or np.any(played < 0):
        raise ValueError("minutes must be nonnegative and finite")
    owned = np.asarray(possessions, dtype=float).reshape(-1)
    if owned.shape != (n_sims,):
        raise ValueError(f"possessions must have length {n_sims}")
    if np.any(~np.isfinite(owned)) or np.any(owned < 0):
        raise ValueError("possessions must be nonnegative and finite")

    rates = {
        column: _filled_rate(profile, column, getattr(model, column))
        for column in OPPORTUNITY_RATES
    }
    _reject_bad_rates(rates)
    fga_per_poss = _shared_rate(rates["fga_per_poss"], "fga_per_poss")
    tov_per_poss = _shared_rate(rates["tov_per_poss"], "tov_per_poss")
    shares = _weights(rates["fga_per_min"], played, "shot rate")
    team_fga = sample_negbin(owned * fga_per_poss, model.alpha, rng)
    shot = allocate_and_shoot(
        team_fga,
        shares,
        rates["p_3pa"],
        rates["p_3p"],
        rates["p_2p"],
        rng,
    )
    fta, ftm = simulate_fta_ftm(
        shot["FGA"],
        rates["fta_per_fga"],
        rates["ft_pct"],
        model.alpha,
        rng,
    )
    if team_points is not None:
        shot, fta, ftm = _match_scenario_points(
            shot,
            fta,
            ftm,
            shares,
            _whole_score(team_points, n_sims),
            rng,
        )
    tov_shares = _weights(rates["tov_per_min"], played, "turnover rate")
    team_tov = sample_negbin(owned * tov_per_poss, model.alpha, rng)
    turnovers = multinomial_rows(team_tov, tov_shares, rng)
    return {
        **shot,
        "FTA": fta,
        "FTM": ftm,
        "PTS": 2 * shot["2PM"] + 3 * shot["3PM"] + ftm,
        "TOV": turnovers,
        "team_fga": team_fga,
        "team_tov": team_tov,
        "shares": shares,
    }


def _whole_score(team_points, n_sims: int) -> np.ndarray:
    try:
        points = np.asarray(team_points, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("team points must be nonnegative and finite") from exc
    if points.shape != (n_sims,):
        raise ValueError(f"team points must have length {n_sims}")
    if np.any(~np.isfinite(points)) or np.any(points < 0):
        raise ValueError("team points must be nonnegative and finite")
    return np.floor(points + 0.5).astype(np.int64)


def _match_scenario_points(shot, fta, ftm, shares, target, rng):
    """Move makes and free throws until each row sums to its scenario score.

    Field-goal attempts stay put. A one-point remainder uses a free throw, so
    a two or a three that would step past the score is taken back and the
    difference is made up at the line.
    """
    two_pm = np.array(shot["2PM"], dtype=np.int64, copy=True)
    three_pm = np.array(shot["3PM"], dtype=np.int64, copy=True)
    two_pa = np.asarray(shot["2PA"], dtype=np.int64)
    three_pa = np.asarray(shot["3PA"], dtype=np.int64)
    fta = np.array(fta, dtype=np.int64, copy=True)
    ftm = np.array(ftm, dtype=np.int64, copy=True)
    shares = np.asarray(shares, dtype=float)
    for index in range(two_pm.shape[0]):
        current = 2 * int(two_pm[index].sum()) + 3 * int(three_pm[index].sum())
        current += int(ftm[index].sum())
        delta = int(target[index]) - current
        made = int(two_pm[index].sum() + three_pm[index].sum() + ftm[index].sum())
        limit = abs(delta) + 2 * made + 3
        steps = 0
        while delta != 0:
            steps += 1
            if steps > limit:
                raise ValueError("scenario points could not be matched")
            weight = shares[index]
            if delta > 0:
                missed_threes = three_pa[index] - three_pm[index]
                missed_twos = two_pa[index] - two_pm[index]
                missed_free_throws = fta[index] - ftm[index]
                if delta >= 3 and np.any(missed_threes > 0):
                    chosen = _pick_player(rng, missed_threes, weight)
                    three_pm[index, chosen] += 1
                    delta -= 3
                elif delta >= 2 and np.any(missed_twos > 0):
                    chosen = _pick_player(rng, missed_twos, weight)
                    two_pm[index, chosen] += 1
                    delta -= 2
                elif np.any(missed_free_throws > 0):
                    chosen = _pick_player(rng, missed_free_throws, weight)
                    ftm[index, chosen] += 1
                    delta -= 1
                else:
                    chosen = _pick_player(rng, weight, weight)
                    fta[index, chosen] += 1
                    ftm[index, chosen] += 1
                    delta -= 1
            else:
                if delta <= -3 and np.any(three_pm[index] > 0):
                    chosen = _pick_player(rng, three_pm[index], weight)
                    three_pm[index, chosen] -= 1
                    delta += 3
                elif delta <= -2 and np.any(two_pm[index] > 0):
                    chosen = _pick_player(rng, two_pm[index], weight)
                    two_pm[index, chosen] -= 1
                    delta += 2
                elif np.any(ftm[index] > 0):
                    chosen = _pick_player(rng, ftm[index], weight)
                    ftm[index, chosen] -= 1
                    delta += 1
                elif np.any(two_pm[index] > 0):
                    chosen = _pick_player(rng, two_pm[index], weight)
                    two_pm[index, chosen] -= 1
                    delta += 2
                elif np.any(three_pm[index] > 0):
                    chosen = _pick_player(rng, three_pm[index], weight)
                    three_pm[index, chosen] -= 1
                    delta += 3
                else:
                    raise ValueError("scenario points could not be matched")
    updated = dict(shot)
    updated["2PM"] = two_pm
    updated["3PM"] = three_pm
    updated["FGM"] = two_pm + three_pm
    return updated, fta, ftm


def _pick_player(rng, eligible, weight) -> int:
    room = np.maximum(np.asarray(eligible, dtype=float), 0.0)
    prior = np.maximum(np.asarray(weight, dtype=float), 0.0)
    scores = room * prior
    if float(scores.sum()) <= 0:
        scores = room
    total = float(scores.sum())
    if total <= 0:
        raise ValueError("scenario points could not be matched")
    return int(rng.choice(scores.shape[0], p=scores / total))


def _prepare_opportunity_logs(logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(logs, pd.DataFrame) or logs.empty:
        raise ValueError("logs must be a non-empty DataFrame")
    required = (*_ID_COLUMNS, *_SHOT_COLUMNS, *_TEAM_COLUMNS)
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
    for column in (*_SHOT_COLUMNS, *_TEAM_COLUMNS):
        frame[column] = _nonnegative(frame[column], column)
    _reject_makes(frame)
    keys = ["game_id", "player_id", "team_id"]
    for column in (*_SHOT_COLUMNS, *_TEAM_COLUMNS, "minutes"):
        counts = frame.groupby(keys, dropna=False)[column].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(f"logs have conflicting {column} for one player-game")
    for column in _TEAM_COLUMNS:
        counts = frame.groupby(["game_id", "team_id"], dropna=False)[column].nunique(
            dropna=False
        )
        if (counts > 1).any():
            raise ValueError(f"logs have conflicting {column} for one team-game")
    return frame.drop_duplicates(keys).reset_index(drop=True)


def _prepare_opportunity_slate(slate: pd.DataFrame) -> pd.DataFrame:
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


def _player_rates(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    work = frame.sort_values(["player_id", "game_date", "game_id"])
    work = work.reset_index(drop=True)
    work["fga_per_min_game"] = _ratio(work["fga"], work["minutes"])
    work["p_3pa_game"] = _ratio(work["fg3_a"], work["fga"])
    work["p_3p_game"] = _ratio(work["fg3_m"], work["fg3_a"])
    work["p_2p_game"] = _ratio(work["fgm"] - work["fg3_m"], work["fga"] - work["fg3_a"])
    work["fta_per_fga_game"] = _ratio(work["fta"], work["fga"])
    work["ft_pct_game"] = _ratio(work["ftm"], work["fta"])
    work["tov_per_min_game"] = _ratio(work["tov"], work["minutes"])
    sources = {
        "fga_per_min": "fga_per_min_game",
        "p_3pa": "p_3pa_game",
        "p_3p": "p_3p_game",
        "p_2p": "p_2p_game",
        "fta_per_fga": "fta_per_fga_game",
        "ft_pct": "ft_pct_game",
        "tov_per_min": "tov_per_min_game",
    }
    for name, source in sources.items():
        work[name] = _window(work, "player_id", source, window)
    return work


def _team_rates(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    teams = (
        frame.sort_values(["team_id", "game_date", "game_id", "player_id"])
        .groupby(["game_id", "team_id"], as_index=False)
        .agg(
            game_date=("game_date", "first"),
            team_fga=("team_fga", "first"),
            team_poss=("team_poss", "first"),
            team_tov=("team_tov", "first"),
        )
        .sort_values(["team_id", "game_date", "game_id"])
        .reset_index(drop=True)
    )
    teams["fga_per_poss_game"] = _ratio(teams["team_fga"], teams["team_poss"])
    teams["tov_per_poss_game"] = _ratio(teams["team_tov"], teams["team_poss"])
    teams["fga_per_poss"] = _window(teams, "team_id", "fga_per_poss_game", window)
    teams["tov_per_poss"] = _window(teams, "team_id", "tov_per_poss_game", window)
    return teams


def _live_opportunity_rows(prior: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    base = slate.loc[:, ["player_id", "game_date", "team_id"]].copy()
    if prior.empty:
        for column in OPPORTUNITY_RATES:
            base[column] = np.nan
        return base
    players = _player_rates(prior, "through")
    last = players.groupby("player_id", sort=False).tail(1)
    teams = _team_rates(prior, "through")
    last_team = teams.groupby("team_id", sort=False).tail(1)
    out = base.merge(
        last[["player_id", *_PLAYER_RATES]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        last_team[["team_id", *_TEAM_RATES]],
        on="team_id",
        how="left",
        validate="many_to_one",
    )
    return out.loc[:, ["player_id", "game_date", "team_id", *OPPORTUNITY_RATES]]


def _window(frame: pd.DataFrame, entity: str, column: str, window: str) -> np.ndarray:
    if window == "before":
        return _before(frame, entity, column, "mean", 1)
    if window == "through":
        return _through(frame, entity, column, "mean", 1)
    raise ValueError(f"unknown window {window}")


def _ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=numer.index, dtype=float)
    ok = numer.notna() & denom.notna() & (denom > 0)
    out.loc[ok] = numer.loc[ok].to_numpy() / denom.loc[ok].to_numpy()
    return out


def _nonnegative(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    if (present & numeric.isna()).any():
        raise ValueError(f"{name} must be numeric")
    if (numeric < 0).fillna(False).any():
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _reject_makes(frame: pd.DataFrame) -> None:
    for made, attempts in _MADE_WITHIN:
        both = frame[made].notna() & frame[attempts].notna()
        if (frame.loc[both, made] > frame.loc[both, attempts]).any():
            raise ValueError(f"{made} cannot exceed {attempts}")


def _filled_rate(profile: pd.DataFrame, column: str, fallback: float) -> np.ndarray:
    if column not in profile.columns:
        return np.full(len(profile), fallback, dtype=float)
    values = pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    if values.shape != (len(profile),):
        raise ValueError(f"{column} must have one value per player")
    missing = ~np.isfinite(values)
    values = values.copy()
    values[missing] = fallback
    return values


def _reject_bad_rates(rates: dict[str, np.ndarray]) -> None:
    for column in _PROBABILITY_RATES:
        if np.any((rates[column] < 0) | (rates[column] > 1)):
            raise ValueError(f"{column} must be in [0, 1]")
    for column, values in rates.items():
        if column in _PROBABILITY_RATES:
            continue
        if np.any(values < 0):
            raise ValueError(f"{column} must be nonnegative")


def _shared_rate(values: np.ndarray, name: str) -> float:
    if float(np.max(values) - np.min(values)) > 1e-8:
        raise ValueError(f"players on one team must share one {name}")
    return float(values[0])


def _weights(per_minute: np.ndarray, minutes: np.ndarray, name: str) -> np.ndarray:
    weights = per_minute[None, :] * minutes
    total = weights.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError(f"no player has minutes and a {name}")
    return weights / total
