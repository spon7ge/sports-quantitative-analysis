"""Fitted league rates, windows, and dispersion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from game_events.checks import _as_probability, _positive_float, _positive_int
from game_events.constants import (
    _MAX_POOL_RHO,
    _WEIGHTED_STATS,
    EVENT_MODEL_VERSION,
    PLAYER_EVENT_RATES,
    PLAYER_PRIOR_MINUTES,
    PLAYER_WINDOW,
    POOL_DISPERSION_KEYS,
    TEAM_EVENT_RATES,
    TEAM_WINDOW,
)
from game_events.strength import prior_minutes_map


@dataclass(frozen=True)
class EventModel:
    """League rates, selected rate settings, and fitted dispersion.

    ``pool_dispersion`` holds beta-binomial intra-pool correlations.
    ``share_concentration`` holds Dirichlet concentrations; ``None`` means
    a plain multinomial split.
    """

    version: int
    cutoff: str
    ast_per_min: float
    oreb_per_min: float
    dreb_per_min: float
    stl_per_min: float
    blk_per_min: float
    assisted_rate: float
    p_rebound: float
    p_oreb: float
    p_oreb_allowed: float
    p_steal: float
    p_steal_against: float
    p_block: float
    p_block_against: float
    player_window: int | None = PLAYER_WINDOW
    team_window: int = TEAM_WINDOW
    # A number stands for every rate. A map is one strength per rate.
    player_prior_minutes: float | dict[str, float] = PLAYER_PRIOR_MINUTES
    team_prior_scale: float = 1.0
    pool_dispersion: dict = field(default_factory=dict)
    share_concentration: dict = field(default_factory=dict)
    tuning: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != EVENT_MODEL_VERSION:
            raise ValueError(
                f"event model version {self.version} != {EVENT_MODEL_VERSION}; refit"
            )
        pd.Timestamp(self.cutoff)
        for column in PLAYER_EVENT_RATES:
            value = getattr(self, column)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{column} must be nonnegative and finite")
        for column in TEAM_EVENT_RATES:
            _as_probability(getattr(self, column), column)
        if self.player_window is not None:
            _positive_int(self.player_window, "player_window")
        _positive_int(self.team_window, "team_window")
        object.__setattr__(
            self, "player_prior_minutes", prior_minutes_map(self.player_prior_minutes)
        )
        _positive_float(self.team_prior_scale, "team_prior_scale")
        for key, rho in self.pool_dispersion.items():
            if key not in POOL_DISPERSION_KEYS:
                raise ValueError(f"unknown pool dispersion {key}")
            if not 0.0 <= float(rho) <= _MAX_POOL_RHO:
                raise ValueError(
                    f"pool dispersion {key} must be in [0, {_MAX_POOL_RHO}]"
                )
        for key, alpha in self.share_concentration.items():
            if key not in _WEIGHTED_STATS:
                raise ValueError(f"unknown share concentration {key}")
            if alpha is not None:
                _positive_float(alpha, f"share concentration {key}")

    @property
    def cutoff_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.cutoff).normalize()


def save_event_model(model: EventModel, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(model), allow_nan=False, indent=2))


def load_event_model(path: str | Path) -> EventModel:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("version") != EVENT_MODEL_VERSION:
        raise ValueError(f"expected event model version {EVENT_MODEL_VERSION}; refit")
    return EventModel(**data)
