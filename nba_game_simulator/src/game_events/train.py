"""League rates, window selection, and dispersion from official logs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from game_events.constants import (
    _EVENT_COLUMNS,
    _MAX_POOL_RHO,
    _MIN_DISPERSION_GROUPS,
    _MIN_TUNING_DATES,
    _PLAYER_SOURCES,
    _RATE_WEIGHT_STAT,
    _TEAM_GRID,
    EVENT_MODEL_VERSION,
    PLAYER_EVENT_RATES,
    PLAYER_PRIOR_MINUTES,
    PLAYER_WINDOW,
    PLAYER_WINDOW_GRID,
    POOL_DISPERSION_KEYS,
    TEAM_EVENT_RATES,
    TEAM_WINDOW,
)
from game_events.draws import _log5
from game_events.logs import _prepare_event_logs
from game_events.model import EventModel
from game_events.rates import (
    _player_counts,
    _player_rate_components,
    _player_rates,
    _targets,
    _team_counts,
    _team_rates,
)
from game_events.strength import predictive_nll, tune_strength


def fit_events(
    logs: pd.DataFrame,
    *,
    tune: bool = True,
    validation_fraction: float = 0.25,
) -> EventModel:
    """Fit league rates, rate settings, and dispersion from official logs.

    League rates are summed events over summed exposure. With ``tune``, each
    player rate's shrinkage and the shared player window are chosen by
    one-step-ahead predictive loss on dates before the final
    ``validation_fraction``. Team windows still use that final slice. Pool
    dispersion and player-share concentration are fit afterward, on the
    games after a short burn-in, using the chosen strengths.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    frame = _prepare_event_logs(logs)
    played = _player_counts(frame)
    teams = _team_counts(frame)
    fitted: dict[str, float] = {}
    for rate in PLAYER_EVENT_RATES:
        denom = float(played[f"{rate}_d"].sum())
        if denom <= 0.0:
            raise ValueError(f"no usable {rate}")
        fitted[rate] = float(played[f"{rate}_n"].sum()) / denom
    for rate in TEAM_EVENT_RATES:
        denom = float(teams[f"{rate}_d"].sum())
        if denom <= 0.0:
            raise ValueError(f"no usable {rate}")
        fitted[rate] = min(float(teams[f"{rate}_n"].sum()) / denom, 1.0)
    cutoff = frame["game_date"].max().date().isoformat()
    base = EventModel(version=EVENT_MODEL_VERSION, cutoff=cutoff, **fitted)

    dates = np.sort(frame["game_date"].unique())
    settings = {
        "player_window": PLAYER_WINDOW,
        "player_prior_minutes": PLAYER_PRIOR_MINUTES,
        "team_window": TEAM_WINDOW,
        "team_prior_scale": 1.0,
    }
    tuning: dict[str, Any] = {"tuned": False}
    if tune and len(dates) >= _MIN_TUNING_DATES:
        validation_start = pd.Timestamp(
            dates[int(len(dates) * (1.0 - validation_fraction))]
        )
        settings, tuning = _tune_settings(played, teams, validation_start)
        tuning["validation_start"] = validation_start.date().isoformat()
        tuning["tuned"] = True

    burn_in = pd.Timestamp(dates[min(len(dates) - 1, len(dates) // 10)])
    tuned = replace(base, **settings)
    pool_dispersion = _fit_pool_dispersion(teams, tuned, burn_in)
    share_concentration = _fit_share_concentration(played, tuned, burn_in)
    return replace(
        tuned,
        pool_dispersion=pool_dispersion,
        share_concentration=share_concentration,
        tuning=tuning,
    )


def _tune_settings(
    played: pd.DataFrame, teams: pd.DataFrame, validation_start: pd.Timestamp
) -> tuple[dict, dict]:
    """Player strengths on training dates; team grid on the validation slice."""
    player = _tune_player_strengths(played, validation_start)
    team_targets = _targets(teams, "team_id")
    team_valid = (teams["game_date"] >= validation_start).to_numpy()
    team_scores = {}
    for window in _TEAM_GRID[0]:
        for scale in _TEAM_GRID[1]:
            rates = _team_rates(teams, team_targets, window, scale)
            team_scores[(window, scale)] = _team_log_likelihood(
                teams, rates, team_valid
            )

    best_team = max(team_scores, key=team_scores.get)
    settings = {
        "player_window": player["player_window"],
        "player_prior_minutes": player["player_prior_minutes"],
        "team_window": int(best_team[0]),
        "team_prior_scale": float(best_team[1]),
    }
    report = {
        "player_predictive_nll": player["predictive_nll"],
        "team_log_likelihood": {f"{w}/{k}": v for (w, k), v in team_scores.items()},
    }
    return settings, report


def _tune_player_strengths(played: pd.DataFrame, validation_start) -> dict:
    """One window, and one strength per rate, from pre-validation games."""
    best_total = np.inf
    best_window: int | None = PLAYER_WINDOW
    best_strength = {rate: float(PLAYER_PRIOR_MINUTES) for rate in PLAYER_EVENT_RATES}
    scores: dict[str, float] = {}
    train = (played["game_date"] < validation_start).to_numpy()
    targets = _targets(played, "player_id")
    for window in PLAYER_WINDOW_GRID:
        components = _player_rate_components(played, targets, window)
        strength: dict[str, float] = {}
        total = 0.0
        for rate in PLAYER_EVENT_RATES:
            n = components[f"{rate}_n"].to_numpy(dtype=float)[train]
            d = components[f"{rate}_d"].to_numpy(dtype=float)[train]
            t = components[f"{rate}_t"].to_numpy(dtype=float)[train]
            x_f = played[f"{rate}_n"].to_numpy(dtype=float)[train]
            e_f = played[f"{rate}_d"].to_numpy(dtype=float)[train]
            k = tune_strength(n, d, t, x_f, e_f)
            strength[rate] = k
            use = _predictive_rows(n, d, t, x_f, e_f)
            if np.any(use):
                total += predictive_nll(
                    np.log(k), n[use], d[use], t[use], x_f[use], e_f[use]
                )
        label = "all" if window is None else str(window)
        scores[label] = float(total)
        if total < best_total:
            best_total = float(total)
            best_window = window
            best_strength = strength
    return {
        "player_window": best_window,
        "player_prior_minutes": best_strength,
        "predictive_nll": scores,
    }


def _predictive_rows(n, d, t, x_f, e_f) -> np.ndarray:
    return (
        np.isfinite(n)
        & np.isfinite(d)
        & np.isfinite(t)
        & np.isfinite(x_f)
        & np.isfinite(e_f)
        & (t >= 0.0)
        & (e_f > 0.0)
        & ((n > 0.0) | (t > 0.0))
    )


def _team_log_likelihood(
    teams: pd.DataFrame, rates: pd.DataFrame, valid: np.ndarray
) -> float:
    total = 0.0
    for rate in TEAM_EVENT_RATES:
        trials = teams[f"{rate}_d"].to_numpy()
        successes = np.minimum(teams[f"{rate}_n"].to_numpy(), trials)
        p = np.clip(rates[rate].to_numpy(), 1e-9, 1.0 - 1e-9)
        use = valid & (trials > 0) & np.isfinite(p)
        total += float(
            np.sum(
                successes[use] * np.log(p[use])
                + (trials[use] - successes[use]) * np.log1p(-p[use])
            )
        )
    return total


def _estimate_rho(successes: Any, trials: Any, p: Any) -> float:
    """Method-of-moments beta-binomial correlation, clipped to a safe range."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    p = np.asarray(p, dtype=float)
    use = np.isfinite(successes) & np.isfinite(trials) & np.isfinite(p)
    use &= (trials > 1) & (p > 0.0) & (p < 1.0)
    if use.sum() < _MIN_DISPERSION_GROUPS:
        return 0.0
    n, d, q = np.minimum(successes[use], trials[use]), trials[use], p[use]
    binomial = d * q * (1.0 - q)
    excess = float(np.sum((n - d * q) ** 2) - np.sum(binomial))
    scale = float(np.sum(d * (d - 1.0) * q * (1.0 - q)))
    if scale <= 0.0:
        return 0.0
    return float(np.clip(excess / scale, 0.0, _MAX_POOL_RHO))


def _estimate_concentration(
    events: np.ndarray, shares: np.ndarray, pool: np.ndarray
) -> float | None:
    """Dirichlet-multinomial concentration matching the observed spread.

    Solves ``sum((x - N p)^2) = sum(N p (1 - p) (N + a) / (1 + a))``.
    ``None`` means the split is no wider than multinomial.
    """
    base = pool * shares * (1.0 - shares)
    observed = float(np.sum((events - pool * shares) ** 2))
    if base.sum() <= 0.0 or observed <= float(base.sum()):
        return None

    def gap(alpha: float) -> float:
        return float(np.sum(base * (pool + alpha) / (1.0 + alpha))) - observed

    low, high = 1e-3, 1e6
    if gap(high) > 0.0:
        return None
    if gap(low) < 0.0:
        return low
    for _ in range(80):
        middle = float(np.sqrt(low * high))
        if gap(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def _fit_pool_dispersion(
    teams: pd.DataFrame, model: EventModel, burn_in: pd.Timestamp
) -> dict[str, float]:
    targets = _targets(teams, "team_id")
    rates = _team_rates(teams, targets, model.team_window, model.team_prior_scale)
    for rate in TEAM_EVENT_RATES:
        rates[rate] = rates[rate].fillna(getattr(model, rate))
    frame = pd.concat([teams.reset_index(drop=True), rates.add_prefix("r_")], axis=1)
    opponent = frame[
        ["game_id", "team_id", *[f"r_{rate}" for rate in TEAM_EVENT_RATES]]
    ]
    opponent = opponent.rename(
        columns={
            "team_id": "opp_id",
            **{f"r_{rate}": f"o_{rate}" for rate in TEAM_EVENT_RATES},
        }
    )
    frame = frame.merge(opponent, on="game_id")
    frame = frame.loc[
        (frame["team_id"] != frame["opp_id"]) & (frame["game_date"] >= burn_in)
    ]
    if frame.empty:
        return {key: 0.0 for key in POOL_DISPERSION_KEYS}

    misses = frame["team_fga"] - frame["team_fgm"]
    opp_misses = frame["team_fga_o"] - frame["team_fgm_o"]
    p_reb = frame["r_p_rebound"].to_numpy()
    p_or = _log5(frame["r_p_oreb"], frame["o_p_oreb_allowed"], model.p_oreb)
    remaining = np.divide(
        p_reb * (1.0 - p_or),
        1.0 - p_reb * p_or,
        out=np.zeros_like(p_reb),
        where=(1.0 - p_reb * p_or) > 0.0,
    )
    observations = {
        "assisted_rate": (
            frame["team_ast"],
            frame["team_fgm"],
            frame["r_assisted_rate"],
        ),
        "p_rebound": (frame["team_oreb"] + frame["opp_dreb"], misses, p_reb),
        "p_oreb": (frame["team_oreb"], frame["team_oreb"] + frame["opp_dreb"], p_or),
        "dreb_given_oreb": (
            frame["opp_dreb"],
            (misses - frame["team_oreb"]).clip(lower=0),
            remaining,
        ),
        "p_steal": (
            frame["team_stl"],
            frame["opp_tov"],
            _log5(frame["r_p_steal"], frame["o_p_steal_against"], model.p_steal),
        ),
        "p_block": (
            frame["team_blk"],
            opp_misses,
            _log5(frame["r_p_block"], frame["o_p_block_against"], model.p_block),
        ),
    }
    return {key: _estimate_rho(*values) for key, values in observations.items()}


def _fit_share_concentration(
    played: pd.DataFrame, model: EventModel, burn_in: pd.Timestamp
) -> dict[str, float | None]:
    targets = _targets(played, "player_id")
    rates = _player_rates(
        played, targets, model.player_window, model.player_prior_minutes
    )
    work = pd.concat(
        [
            played[
                ["game_id", "team_id", "game_date", "minutes", *_EVENT_COLUMNS]
            ].reset_index(drop=True),
            rates,
        ],
        axis=1,
    )
    work = work.loc[work["game_date"] >= burn_in]
    out: dict[str, float | None] = {}
    for stat, rate in _RATE_WEIGHT_STAT.items():
        column = _PLAYER_SOURCES[rate]
        part = work[["game_id", "team_id", "minutes", column, rate]].copy()
        part[rate] = part[rate].fillna(getattr(model, rate))
        incomplete = (
            part[column]
            .isna()
            .groupby([part["game_id"], part["team_id"]], sort=False)
            .transform("any")
        )
        part = part.loc[~incomplete.astype(bool)]
        if part.empty:
            out[stat] = None
            continue
        part["weight"] = part[rate] * part["minutes"]
        group = part.groupby(["game_id", "team_id"], sort=False)
        part["pool"] = group[column].transform("sum")
        part["weight_total"] = group["weight"].transform("sum")
        part["size"] = group[column].transform("size")
        part = part.loc[
            (part["pool"] > 0) & (part["weight_total"] > 0) & (part["size"] > 1)
        ]
        if part.groupby(["game_id", "team_id"]).ngroups < _MIN_DISPERSION_GROUPS:
            out[stat] = None
            continue
        out[stat] = _estimate_concentration(
            part[column].to_numpy(dtype=float),
            (part["weight"] / part["weight_total"]).to_numpy(dtype=float),
            part["pool"].to_numpy(dtype=float),
        )
    return out
