"""Fitted opportunity rates and one simulation result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from game_opportunities.checks import _number
from game_opportunities.constants import (
    _PROBABILITIES,
    MODEL_VERSION,
    OPPORTUNITY_RATES,
)


@dataclass(frozen=True)
class OpportunityModel:
    fga_per_min: float
    p_3pa: float
    p_3p: float
    p_2p: float
    fta_per_fga: float
    ft_pct: float
    tov_per_min: float
    fga_per_poss: float
    tov_per_poss: float

    alpha: float = 0.03
    fta_per_min: float | None = None
    fta_per_poss: float | None = None
    ast_per_min: float = 0.10
    oreb_per_min: float = 0.04
    tov_rho: float = 0.01
    assist_rate: float = 0.55
    oreb_rate: float = 0.25
    assist_rho: float = 0.0
    oreb_rho: float = 0.0
    training_cutoff: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = MODEL_VERSION

    def __post_init__(self):
        if self.fta_per_min is None:
            object.__setattr__(
                self,
                "fta_per_min",
                self.fga_per_min * self.fta_per_fga,
            )

        if self.fta_per_poss is None:
            object.__setattr__(
                self,
                "fta_per_poss",
                self.fga_per_poss * self.fta_per_fga,
            )

        for name in OPPORTUNITY_RATES:
            value = _number(
                getattr(self, name),
                name,
                nonnegative=True,
            )
            if name in _PROBABILITIES and value > 1:
                raise ValueError(f"{name} must be in [0, 1]")

        _number(self.alpha, "alpha", nonnegative=True)

        if (
            not 0
            <= _number(
                self.tov_rho,
                "tov_rho",
                nonnegative=True,
            )
            < 1
        ):
            raise ValueError("tov_rho must be in [0, 1)")

        for name in ("assist_rate", "oreb_rate"):
            if _number(getattr(self, name), name, nonnegative=True) > 1:
                raise ValueError(f"{name} must be in [0, 1]")

        for name in ("assist_rho", "oreb_rho"):
            if _number(getattr(self, name), name, nonnegative=True) >= 1:
                raise ValueError(f"{name} must be in [0, 1)")

        if self.version != MODEL_VERSION:
            raise ValueError("unsupported opportunity model version")


@dataclass(frozen=True)
class TeamSimResult:
    player_ids: tuple
    opportunity_shares: np.ndarray
    stats: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    teams: dict[str, TeamSimResult]


def _model(value):
    if (
        not isinstance(value, OpportunityModel)
        or getattr(value, "version", None) != MODEL_VERSION
    ):
        raise ValueError("model must be a supported OpportunityModel; refit old models")
