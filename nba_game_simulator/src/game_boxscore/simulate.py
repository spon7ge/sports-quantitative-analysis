"""One pass through layers 1–5, one row per simulated game."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from game_boxscore.assemble import assemble_box_score
from game_boxscore.checks import _player_id_tuple, _reject_shared_players
from game_boxscore.constants import (
    _ENVIRONMENT_COLUMNS,
    _LAYER4_EVENT_INPUTS,
    _LAYER4_STATS,
    _VIEW_PAIRS,
    LAYER_MODULES,
    SIDES,
)
from game_boxscore.types import BoxScoreSet, PipelineModels, TeamInputs


def simulate_box_scores(
    models: PipelineModels,
    home: TeamInputs,
    away: TeamInputs,
    date: Any,
    n_sims: int,
    rng: np.random.Generator,
    *,
    total: float | None = None,
    spread: float | None = None,
    availability: dict[str, np.ndarray] | None = None,
    condition_on_score: bool = True,
) -> BoxScoreSet:
    """Run layers 1–5 once and keep one row per simulated game.

    Layer 1 is drawn once from the home perspective and every later layer
    reads the same rows. ``spread`` uses the negative-favorite convention.
    ``availability`` is the shared bool mask per side, ``(n_sims, n_players)``
    or ``(n_players,)``, in roster order; without it every listed player is
    available. Each layer gets its own random stream derived from ``rng``.
    """
    if not isinstance(models, PipelineModels):
        raise ValueError("models must be PipelineModels")
    for label, inputs in (("home", home), ("away", away)):
        if not isinstance(inputs, TeamInputs):
            raise ValueError(f"{label} must be TeamInputs")
    if home.name == away.name:
        raise ValueError("home and away must be different teams")
    if (
        isinstance(n_sims, bool)
        or not isinstance(n_sims, (int, np.integer))
        or int(n_sims) <= 0
    ):
        raise ValueError("n_sims must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    n_sims = int(n_sims)
    game_date = _game_date(date)
    for label in ("environment", "minutes", "rotation", "opportunities", "events"):
        _reject_stale_model(label, getattr(models, label), game_date)

    inputs = {"home": home, "away": away}
    player_ids = {side: _aligned_player_ids(side, inputs[side]) for side in SIDES}
    _reject_shared_players(player_ids)
    availability = _shared_availability(availability, player_ids, n_sims)
    rows = {side: _rows_for_side(inputs[side].rows, side) for side in SIDES}
    streams, seeds = _layer_streams(rng)

    environment = _layer("environment")(
        models.environment,
        home.name,
        away.name,
        date,
        total=total,
        spread=spread,
        n=n_sims,
        is_home=True,
        team_lineup=home.lineup,
        opponent_lineup=away.lineup,
        rng=streams["environment"],
    )
    environment = _check_environment(environment, n_sims)

    mask_kwargs = {} if availability is None else {"availability": availability}
    minutes_result = _layer("minutes")(
        models.minutes,
        rows["home"],
        rows["away"],
        environment,
        seed=seeds["minutes"],
        **mask_kwargs,
    )
    minutes = {side: minutes_result[side] for side in SIDES}

    segments = None
    if models.rotation is not None:
        segments = {}
        for side in SIDES:
            side_mask = (
                {} if availability is None else {"availability": availability[side]}
            )
            segments[side] = _layer("segments")(
                models.rotation,
                rows[side],
                _team_view(environment, side),
                minutes[side],
                game_date=date,
                seed=seeds[f"segments_{side}"],
                **side_mask,
            )

    layer4_kwargs: dict[str, Any] = {
        "condition_on_score": condition_on_score,
        **mask_kwargs,
    }
    if segments is not None:
        layer4_kwargs.update(
            home_segments=segments["home"], away_segments=segments["away"]
        )
    opportunities = _layer("opportunities")(
        models.opportunities,
        home.opportunity_profile,
        away.opportunity_profile,
        minutes["home"],
        minutes["away"],
        environment,
        streams["opportunities"],
        **layer4_kwargs,
    )
    shots = {
        side: _shots_for_events(_side_output(opportunities, side), f"Layer 4 {side}")
        for side in SIDES
    }

    event_stats, event_details = _layer("events")(
        models.events,
        {side: inputs[side].event_profile for side in SIDES},
        shots,
        minutes,
        streams["events"],
        segments=segments,
        availability=availability,
        return_details=True,
    )

    box = assemble_box_score(
        environment,
        player_ids,
        minutes,
        shots,
        event_stats,
        availability=availability,
        check_score=condition_on_score,
    )
    metadata = {
        **box.metadata,
        "team_names": {"home": home.name, "away": away.name},
        "game_date": game_date.date().isoformat(),
        "sim_ids": environment.index.to_numpy(),
        "seeds": seeds,
        "segments_used": segments is not None,
        "condition_on_score": condition_on_score,
        "availability_supplied": availability is not None,
        "models": {
            label: _model_stamp(getattr(models, label))
            for label in (
                "environment",
                "minutes",
                "rotation",
                "opportunities",
                "events",
            )
        },
        "event_sources": event_details.get("sources"),
        "score_pool_resamples": _score_resamples(opportunities),
    }
    return replace(box, metadata=metadata)


def _layer(key: str):
    module_name, function_name = LAYER_MODULES[key]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{key} layer module {module_name!r} not found; set LAYER_MODULES[{key!r}]"
        ) from exc
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise ImportError(f"{module_name} has no {function_name}") from exc


def _layer_streams(rng: np.random.Generator) -> tuple[dict, dict]:
    """Independent streams so one layer's draw count cannot shift another's."""
    names = (
        "environment",
        "minutes",
        "segments_home",
        "segments_away",
        "opportunities",
        "events",
    )
    root = np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
    children = dict(zip(names, root.spawn(len(names))))
    streams = {
        name: np.random.default_rng(children[name])
        for name in ("environment", "opportunities", "events")
    }
    # Layers 2 and 3 take integer seeds.
    seeds = {
        name: int(children[name].generate_state(1, dtype=np.uint32)[0])
        for name in ("minutes", "segments_home", "segments_away")
    }
    return streams, seeds


def _game_date(date: Any) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(date)
    except (TypeError, ValueError) as exc:
        raise ValueError("date must be a parseable date") from exc
    if pd.isna(stamp):
        raise ValueError("date must be a parseable date")
    return stamp.normalize()


def _model_cutoff(model: Any) -> pd.Timestamp | None:
    for name in ("cutoff_date", "cutoff", "training_cutoff", "model_cutoff"):
        value = (
            model.get(name)
            if isinstance(model, Mapping)
            else getattr(model, name, None)
        )
        if value is not None:
            return pd.Timestamp(value).normalize()
    return None


def _reject_stale_model(label: str, model: Any, game_date: pd.Timestamp) -> None:
    if model is None:
        return
    cutoff = _model_cutoff(model)
    if cutoff is not None and game_date <= cutoff:
        raise ValueError(
            f"{label} model cutoff {cutoff.date()} is not before "
            f"the game date {game_date.date()}"
        )


def _model_stamp(model: Any) -> dict | None:
    if model is None:
        return None
    version = (
        model.get("version")
        if isinstance(model, Mapping)
        else getattr(model, "version", None)
    )
    cutoff = _model_cutoff(model)
    return {
        "type": type(model).__name__,
        "version": version,
        "cutoff": None if cutoff is None else cutoff.date().isoformat(),
    }


def _aligned_player_ids(side: str, inputs: TeamInputs) -> tuple:
    """Every layer must list the same players in the same order."""
    sources = {
        "rows": inputs.rows,
        "opportunity_profile": inputs.opportunity_profile,
        "event_profile": inputs.event_profile,
    }
    ids = {
        label: _player_id_tuple(frame, f"{side} {label}")
        for label, frame in sources.items()
    }
    reference = ids["rows"]
    for label, values in ids.items():
        if values != reference:
            raise ValueError(f"{side} {label} player order differs from rows")
    if len(reference) < 5:
        raise ValueError(f"{side} needs at least five players")
    return reference


def _rows_for_side(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    expected = 1 if side == "home" else 0
    if "is_home" in rows.columns:
        values = pd.to_numeric(rows["is_home"], errors="coerce")
        if not values.eq(expected).all():
            raise ValueError(f"{side} rows must have is_home = {expected}")
        return rows
    return rows.assign(is_home=expected)


def _shared_availability(
    availability: Any, player_ids: dict[str, tuple], n_sims: int
) -> dict[str, np.ndarray] | None:
    if availability is None:
        return None
    if not isinstance(availability, Mapping) or set(availability) != set(SIDES):
        raise ValueError("availability must have home and away masks")
    out = {}
    for side in SIDES:
        n_players = len(player_ids[side])
        mask = np.asarray(availability[side])
        if mask.dtype != bool or mask.shape not in ((n_players,), (n_sims, n_players)):
            raise ValueError(
                f"{side} availability must be bool ({n_sims}, {n_players}) "
                f"or ({n_players},)"
            )
        mask = np.broadcast_to(mask, (n_sims, n_players)).copy()
        if np.any(mask.sum(axis=1) < 5):
            raise ValueError(f"{side} availability leaves fewer than five players")
        out[side] = mask
    return out


def _check_environment(environment: Any, n_sims: int) -> pd.DataFrame:
    if not isinstance(environment, pd.DataFrame) or environment.empty:
        raise ValueError("Layer 1 must return a non-empty DataFrame")
    missing = [
        column for column in _ENVIRONMENT_COLUMNS if column not in environment.columns
    ]
    if missing:
        raise ValueError(f"Layer 1 output needs {missing}")
    if len(environment) != n_sims:
        raise ValueError(f"Layer 1 returned {len(environment)} rows, expected {n_sims}")
    if not environment.index.is_unique:
        raise ValueError("Layer 1 simulation index must be unique")
    values = environment.loc[:, list(_ENVIRONMENT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Layer 1 output must be numeric and finite")
    if not values["is_home"].eq(1).all():
        raise ValueError("Layer 1 was asked for the home perspective")
    expected = 48.0 + 5.0 * values["n_ot"]
    if not np.allclose(values["game_minutes"], expected):
        raise ValueError("Layer 1 game_minutes must equal 48 + 5 * n_ot")
    return environment


def _team_view(environment: pd.DataFrame, side: str) -> pd.DataFrame:
    """The Layer 1 rows from ``side``'s perspective, same index and duration."""
    is_home = pd.to_numeric(environment["is_home"]).to_numpy() == 1
    want_home = side == "home"
    flip = is_home != want_home
    if not flip.any():
        return environment
    view = environment.copy()
    for first, second in _VIEW_PAIRS:
        view.loc[flip, first] = environment.loc[flip, second]
        view.loc[flip, second] = environment.loc[flip, first]
    view.loc[flip, "team_margin"] = -environment.loc[flip, "team_margin"]
    view["is_home"] = int(want_home)
    return view


def _side_output(result: Any, side: str) -> Any:
    if isinstance(result, Mapping):
        if side not in result:
            raise ValueError(f"Layer 4 output has no {side} entry")
        return result[side]
    if hasattr(result, side):
        return getattr(result, side)
    raise ValueError(f"Layer 4 output has no {side} entry")


def _shot_source(output: Any, label: str) -> Mapping:
    if isinstance(output, Mapping) and "FGA" in output:
        return output
    if isinstance(output, Mapping) and isinstance(output.get("stats"), Mapping):
        return output["stats"]
    if isinstance(getattr(output, "stats", None), Mapping):
        return output.stats
    raise ValueError(f"{label} must map stat names to player arrays")


def _shots_for_events(output: Any, label: str) -> dict[str, np.ndarray]:
    """Counting stats plus the zone arrays Layer 5 consumes.

    Assists and offensive rebounds are left for Layer 5 when the zone makes
    and the misses by zone are present. Layer 4's copies of those two stats
    are not forwarded.
    """

    source = _shot_source(output, label)
    shots = _stat_arrays(output, _LAYER4_STATS, label)
    for key in _LAYER4_EVENT_INPUTS:
        if key in source:
            shots[key] = np.asarray(source[key])
    if "ZONE_FGM" in source:
        shots.pop("AST", None)
    if "MISS_BY_ZONE" in source:
        shots.pop("OREB", None)
    return shots


def _stat_arrays(output: Any, keys: tuple, label: str) -> dict[str, np.ndarray]:
    source = _shot_source(output, label)
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError(f"{label} needs {missing}")
    return {key: np.asarray(source[key]) for key in keys}


def _score_resamples(result: Any) -> Any:
    if isinstance(result, Mapping) and "score_pool_resamples" in result:
        return result["score_pool_resamples"]
    found = {}
    for side in SIDES:
        try:
            output = _side_output(result, side)
        except ValueError:
            continue
        value = (
            output.get("score_pool_resamples") if isinstance(output, Mapping) else None
        )
        if value is None:
            value = getattr(output, "score_pool_resamples", None)
        if value is not None:
            found[side] = value
    return found or None
