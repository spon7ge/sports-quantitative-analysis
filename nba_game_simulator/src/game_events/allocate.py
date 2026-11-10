"""Credit both teams from Layer 2 minutes, Layer 3 segments, and Layer 4 shots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from game_events.checks import (
    _as_count_matrix,
    _as_factor,
    _as_probability,
    _normalize_ids,
)
from game_events.constants import (
    _RATE_WEIGHT_STAT,
    _SHOT_KEYS,
    _UPSTREAM_STATS,
    _WEIGHTED_STATS,
    PLAYER_EVENT_RATES,
    TEAM_EVENT_RATES,
)
from game_events.credit import (
    CHANCE_RATE_COLUMNS,
    PASSER_PRIOR_ASSISTS,
    CreditRates,
    assisted_matrix,
    passer_counts_matrix,
    passer_tensor,
    profile_with_credit,
)
from game_events.draws import _log5
from game_events.model import EventModel
from game_events.segments import _apply_segments
from game_events.simulate import (
    _draw_events,
    _optional_counts,
    _reject_shared_players,
    _Side,
    _validate_upstream,
)
from possession_table import ZONES


def allocate_events(
    model: EventModel,
    profiles: dict[str, pd.DataFrame],
    shots: dict[str, Mapping[str, np.ndarray]],
    minutes: dict[str, Any],
    rng: np.random.Generator,
    *,
    segments: dict[str, Any] | None = None,
    availability: dict[str, np.ndarray] | None = None,
    factors: dict[str, dict[str, Any]] | None = None,
    resimulate: tuple[str, ...] = (),
    dispersion: bool = True,
    return_details: bool = False,
    credit: CreditRates | None = None,
):
    """Credit both teams for events the Layer 4 draw already produced.

    ``profiles[team]`` has one row per player with ``player_id`` and the
    ``EVENT_RATES`` columns from ``training_events`` or ``live_events``.
    Missing rates use the model's league value. ``minutes[team]`` is the
    Layer 2 matrix (array or DataFrame whose columns are player IDs) in the
    same player order. ``shots[team]`` is Layer 4's player output; its
    ``AST`` and ``OREB`` are reused unless named in ``resimulate``.

    ``segments[team]`` is Layer 3's schedule list, one per simulation row.
    With segments, events go only to players on the floor when they happened,
    and the segment minutes must match ``minutes``. Without segments, weights
    are rate times total minutes.

    ``availability[team]`` is the shared bool mask, ``(n_sims, n_players)``
    or ``(n_players,)``. ``factors[team][stat]`` multiplies that stat's
    player weights, for lineup or matchup context. ``dispersion=False``
    ignores the fitted dispersion.

    ``credit`` is the play-by-play rate table from ``credit_rates``. It
    replaces the team assisted rate with P(assisted | shooter, zone) and
    P(passer | shooter, lineup), rebound weights with rebounds per chance
    on the floor, blocks with missed rim attempts, and steals with live-ball
    turnovers. The shot dict then needs ``LIVE_TOV`` and either ``RIM_FGA``
    or ``MISS_BY_ZONE``, and ``ZONE_FGM`` when assists are drawn here.

    When ``ZONE_FGM`` is present, assists are linked to those makes and a
    supplied ``AST`` array is not reused. When ``MISS_BY_ZONE`` is present,
    offensive rebounds are drawn from those misses and a supplied ``OREB``
    array is not reused.

    Row ``k`` of every output is simulation row ``k`` of the inputs.
    """
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
    for label, extra in (("availability", availability), ("factors", factors)):
        if extra is not None and (
            not isinstance(extra, dict) or not set(extra) <= set(profiles)
        ):
            raise ValueError(f"{label} must be keyed by the same team names")
    if segments is not None and (
        not isinstance(segments, dict) or set(segments) != set(profiles)
    ):
        raise ValueError("segments must cover both teams")
    unknown = set(resimulate) - set(_UPSTREAM_STATS)
    if unknown:
        raise ValueError(
            f"resimulate accepts only {_UPSTREAM_STATS}, got {sorted(unknown)}"
        )

    if credit is not None and not isinstance(credit, CreditRates):
        raise ValueError("credit must be CreditRates")
    sides = []
    indexes = []
    for name in profiles:
        profile = profiles[name]
        redraw_assists = "AST" not in shots[name] or "AST" in resimulate
        if credit is not None:
            _require_credit_shots(name, shots[name], redraw_assists)
            profile = profile_with_credit(profile, credit)
        side, index = _prepare_side(
            name,
            profile,
            shots[name],
            minutes[name],
            model,
            None if availability is None else availability.get(name),
            None if factors is None else factors.get(name),
            frozenset(resimulate),
        )
        if credit is not None:
            side = replace(
                side,
                credit_rates=True,
                p_assisted=assisted_matrix(credit, side.player_ids),
                passer_counts=passer_counts_matrix(credit, side.player_ids),
                passer_strength=PASSER_PRIOR_ASSISTS,
            )
        sides.append(side)
        indexes.append(index)
    if sides[0].fga.shape[0] != sides[1].fga.shape[0]:
        raise ValueError("both teams must have the same number of simulations")
    if (
        indexes[0] is not None
        and indexes[1] is not None
        and not indexes[0].equals(indexes[1])
    ):
        raise ValueError("home and away minute rows must share the simulation index")
    _reject_shared_players(sides)
    if segments is not None:
        sides = _apply_segments(sides, segments, indexes)
    sides = [_apply_passers(side) for side in sides]

    probs = [
        _matchup_probabilities(model, sides[index], sides[1 - index])
        for index in range(2)
    ]
    pool_dispersion = model.pool_dispersion if dispersion else {}
    concentration = model.share_concentration if dispersion else {}
    stats, details, _pairs = _draw_events(
        sides, probs, rng, pool_dispersion, concentration
    )
    if not return_details:
        return stats
    details["probabilities"] = {
        side.name: probs[index] for index, side in enumerate(sides)
    }
    details["player_ids"] = {side.name: side.player_ids for side in sides}
    details["segments_used"] = segments is not None
    details["dispersion"] = {
        "pool": dict(pool_dispersion),
        "share_concentration": dict(concentration),
    }
    return stats, details


def _matchup_probabilities(
    model: EventModel, offense: _Side, defense: _Side
) -> dict[str, float]:
    """Event probabilities for ``offense``'s possessions against ``defense``.

    Play-by-play rates are already per chance, per rim attempt, or per
    live-ball turnover. Mixing them with the box-score allowed rates would
    put them back on the wrong denominator.
    """
    if offense.credit_rates or defense.credit_rates:
        return {
            "p_rebound": offense.rates["p_rebound"],
            "p_oreb": offense.rates["p_oreb"],
            "p_steal": defense.rates["p_steal"],
            "p_block": defense.rates["p_block"],
        }
    return {
        "p_rebound": offense.rates["p_rebound"],
        "p_oreb": _log5(
            offense.rates["p_oreb"], defense.rates["p_oreb_allowed"], model.p_oreb
        ),
        "p_steal": _log5(
            defense.rates["p_steal"], offense.rates["p_steal_against"], model.p_steal
        ),
        "p_block": _log5(
            defense.rates["p_block"], offense.rates["p_block_against"], model.p_block
        ),
    }


def _prepare_side(
    name: str,
    profile: pd.DataFrame,
    shot: Mapping[str, Any],
    played: Any,
    model: EventModel,
    availability: Any,
    factors: Mapping[str, Any] | None,
    resimulate: frozenset,
) -> tuple[_Side, pd.Index | None]:
    if not isinstance(profile, pd.DataFrame) or profile.empty:
        raise ValueError(f"{name} profile must be a non-empty DataFrame")
    if "player_id" not in profile.columns:
        raise ValueError(f"{name} profile needs player_id")
    player_ids = tuple(_normalize_ids(profile["player_id"], f"{name} player_id"))
    if len(set(player_ids)) != len(player_ids):
        raise ValueError(f"{name} profile player_id values must be unique")
    if "team_id" in profile.columns and profile["team_id"].nunique(dropna=False) > 1:
        raise ValueError(f"{name} profile mixes team_id values")

    index = None
    if isinstance(played, pd.DataFrame):
        columns = tuple(_normalize_ids(list(played.columns), f"{name} minute columns"))
        if columns != player_ids:
            raise ValueError(f"{name} minute columns must match profile player order")
        index = played.index
    try:
        minutes = np.asarray(played, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} minutes must be numeric") from exc
    if minutes.ndim != 2 or minutes.shape[1] != len(player_ids):
        raise ValueError(f"{name} minutes must have shape (n_sims, {len(player_ids)})")
    if np.any(~np.isfinite(minutes)) or np.any(minutes < 0):
        raise ValueError(f"{name} minutes must be nonnegative and finite")
    n_sims, n_players = minutes.shape

    if not isinstance(shot, Mapping):
        raise ValueError(f"{name} shots must include {_SHOT_KEYS}")
    missing = [key for key in _SHOT_KEYS if key not in shot]
    if missing:
        raise ValueError(f"{name} shots need {missing}")
    fga = _as_count_matrix(shot["FGA"], n_sims, n_players, f"{name} FGA")
    fgm = _as_count_matrix(shot["FGM"], n_sims, n_players, f"{name} FGM")
    tov = _as_count_matrix(shot["TOV"], n_sims, n_players, f"{name} TOV")
    if np.any(fgm > fga):
        raise ValueError(f"{name} FGM cannot exceed FGA")

    on_floor = minutes > 0.0
    if availability is not None:
        mask = np.asarray(availability)
        if mask.dtype != bool or mask.shape not in ((n_players,), (n_sims, n_players)):
            raise ValueError(f"{name} availability must be a player-aligned bool mask")
        mask = np.broadcast_to(mask, (n_sims, n_players))
        if np.any(on_floor & ~mask):
            raise ValueError(f"{name} unavailable players received minutes")
        on_floor &= mask

    zone_makes = "ZONE_FGM" in shot
    zone_misses = "MISS_BY_ZONE" in shot
    upstream = {
        key: _as_count_matrix(shot[key], n_sims, n_players, f"{name} {key}")
        for key in _UPSTREAM_STATS
        if key in shot
        and key not in resimulate
        and not (key == "AST" and zone_makes)
        and not (key == "OREB" and zone_misses)
    }
    for key, value in {"FGA": fga, "TOV": tov, **upstream}.items():
        if np.any((value > 0) & ~on_floor):
            raise ValueError(f"{name} players without minutes recorded {key}")
    for key, value in upstream.items():
        _validate_upstream(name, key, value, fga, fgm)

    rate_names = list(PLAYER_EVENT_RATES)
    rate_names.extend(
        column for column in CHANCE_RATE_COLUMNS.values() if column in profile.columns
    )
    rates = {
        column: _filled_event_rate(
            profile,
            column,
            getattr(model, column) if column in PLAYER_EVENT_RATES else 0.0,
        )
        for column in rate_names
    }
    for column, values in rates.items():
        if np.any(values < 0.0):
            raise ValueError(f"{name} {column} must be nonnegative")
    team_rates = {
        column: _team_rate(profile, column, getattr(model, column), name)
        for column in TEAM_EVENT_RATES
    }

    factors = factors or {}
    unknown = set(factors) - set(_WEIGHTED_STATS)
    if unknown:
        raise ValueError(f"{name} factors accept only {_WEIGHTED_STATS}")
    exposure = minutes * on_floor
    intensity = {}
    weights = {}
    for stat, rate in _RATE_WEIGHT_STAT.items():
        chance = CHANCE_RATE_COLUMNS.get(stat)
        rate_name = chance if chance in rates else rate
        values = np.broadcast_to(rates[rate_name][None, :], (n_sims, n_players)).copy()
        if stat in factors:
            values = values * _as_factor(
                factors[stat], n_sims, n_players, f"{name} {stat} factor"
            )
        intensity[stat] = values
        weights[stat] = values * exposure

    side = _Side(
        name=name,
        player_ids=player_ids,
        fga=fga,
        fgm=fgm,
        tov=tov,
        assisted_rate=np.full(n_players, team_rates["assisted_rate"]),
        weights=weights,
        oreb=upstream.get("OREB"),
        ast=upstream.get("AST"),
        rates=team_rates,
        minutes=minutes,
        intensity=intensity,
        zone_fgm=_zone_makes(name, shot, n_sims, n_players, fgm),
        rim_fga=_optional_counts(shot, "RIM_FGA", n_sims, n_players, fga, "FGA"),
        live_tov=_optional_counts(shot, "LIVE_TOV", n_sims, n_players, tov, "TOV"),
        on_floor=on_floor,
        miss_by_zone=_misses_by_zone(name, shot, n_sims, fga, fgm),
        p_oreb_zone=_zone_oreb_rates(profile, name),
    )
    return side, index


def _require_credit_shots(
    name: str, shot: Mapping[str, Any], redraw_assists: bool
) -> None:
    missing = ["LIVE_TOV"] if "LIVE_TOV" not in shot else []
    if "RIM_FGA" not in shot and "MISS_BY_ZONE" not in shot:
        missing.append("RIM_FGA")
    if redraw_assists and "ZONE_FGM" not in shot:
        missing.append("ZONE_FGM")
    if missing:
        raise ValueError(f"{name} event credit needs {missing}")


def _misses_by_zone(name, shot, n_sims, fga, fgm):
    if "MISS_BY_ZONE" not in shot:
        return None
    try:
        values = np.asarray(shot["MISS_BY_ZONE"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} MISS_BY_ZONE must be a count array") from exc
    if values.shape != (n_sims, len(ZONES)):
        raise ValueError(
            f"{name} MISS_BY_ZONE must have shape ({n_sims}, {len(ZONES)})"
        )
    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError(f"{name} MISS_BY_ZONE must be a nonnegative integer array")
    values = values.astype(np.int64)
    misses = fga.sum(axis=1) - fgm.sum(axis=1)
    if np.any(values.sum(axis=1) != misses):
        raise ValueError(f"{name} MISS_BY_ZONE must add up to missed field goals")
    return values


def _zone_oreb_rates(profile, name):
    columns = [f"{zone}_p_oreb" for zone in ZONES]
    if any(column not in profile.columns for column in columns):
        return None
    rates = np.array(
        [_team_rate(profile, column, np.nan, name) for column in columns],
        dtype=float,
    )
    if not np.isfinite(rates).all():
        return None
    return rates


def _zone_makes(name, shot, n_sims, n_players, fgm):
    if "ZONE_FGM" not in shot:
        return None
    try:
        zone_fgm = np.asarray(shot["ZONE_FGM"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} ZONE_FGM must be a count array") from exc
    if zone_fgm.shape != (n_sims, n_players, len(ZONES)):
        raise ValueError(
            f"{name} ZONE_FGM must have shape ({n_sims}, {n_players}, {len(ZONES)})"
        )
    if (
        np.any(~np.isfinite(zone_fgm))
        or np.any(zone_fgm < 0)
        or np.any(zone_fgm != np.floor(zone_fgm))
    ):
        raise ValueError(f"{name} ZONE_FGM must be a nonnegative integer array")
    zone_fgm = zone_fgm.astype(np.int64)
    if np.any(zone_fgm.sum(axis=2) != fgm):
        raise ValueError(f"{name} ZONE_FGM must add up to FGM")
    return zone_fgm


def _apply_passers(side: _Side) -> _Side:
    """Replace creation weights with P(passer | shooter, lineup)."""
    if side.passer_counts is None or side.ast is not None:
        return side
    creation = (
        side.intensity["AST"] if side.intensity is not None else side.weights["AST"]
    )
    if creation.ndim != 2:
        raise ValueError(f"{side.name} assist intensity must be per player")
    on_floor = side.on_floor
    if on_floor is None:
        on_floor = np.ones(creation.shape, dtype=bool)
    if side.weights["AST"].ndim == 3:
        # Segments already zero teammates who were not on the floor with the
        # shooter. The pair counts then pick which of those teammates passed.
        tensor = side.weights["AST"] * (
            side.passer_counts[None, :, :] + side.passer_strength
        )
        tensor = np.array(tensor, copy=True)
        diagonal = np.arange(tensor.shape[-1])
        tensor[:, diagonal, diagonal] = 0.0
    else:
        tensor = passer_tensor(
            creation, side.passer_counts, side.passer_strength, on_floor
        )
    return replace(side, weights={**side.weights, "AST": tensor})


def _filled_event_rate(
    profile: pd.DataFrame, column: str, fallback: float
) -> np.ndarray:
    if column not in profile.columns:
        return np.full(len(profile), fallback, dtype=float)
    values = pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
    values = values.copy()
    values[~np.isfinite(values)] = fallback
    return values


def _team_rate(profile: pd.DataFrame, column: str, fallback: float, name: str) -> float:
    values = _filled_event_rate(profile, column, np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(fallback)
    if float(finite.max() - finite.min()) > 1e-8:
        raise ValueError(f"{name} players must share one {column}")
    return _as_probability(finite[0], f"{name} {column}")
