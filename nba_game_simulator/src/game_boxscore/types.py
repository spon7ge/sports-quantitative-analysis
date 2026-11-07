"""Inputs, fitted models, props, and the assembled box score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TeamInputs:
    """One team's inputs, all in the same player order.

    ``rows`` are the Layer 2 and Layer 3 roster rows. ``opportunity_profile``
    is the Layer 4 profile and ``event_profile`` the Layer 5 profile.
    ``lineup`` is the starter list Layer 1 uses for lineup features.
    """

    name: str
    rows: pd.DataFrame
    opportunity_profile: pd.DataFrame
    event_profile: pd.DataFrame
    lineup: list | None = None


@dataclass(frozen=True)
class PipelineModels:
    environment: Any
    minutes: Any
    opportunities: Any
    events: Any
    # Without a rotation model, layers 4 and 5 run on total minutes only.
    rotation: Any = None


@dataclass(frozen=True)
class Prop:
    team: str
    player_id: Any
    stat: str
    line: float
    side: str = "over"


@dataclass(frozen=True)
class PropProbability:
    joint: float
    marginals: tuple[float, ...]
    independent_product: float
    n_sims: int
    pushes: tuple[float, ...] = ()
    joint_standard_error: float = 0.0


@dataclass(frozen=True)
class OverPrice:
    """One over: win, loss, and push shares, plus profit per dollar staked."""

    p_over: float
    p_under: float
    p_push: float
    ev: float


@dataclass(frozen=True)
class PortfolioRisk:
    """Equal one-dollar overs on the same simulated games.

    ``risk`` is the standard deviation of the combined profit. A push
    returns the stake, so it adds nothing to profit.
    """

    n: int
    expected_profit: float
    risk: float
    p_loss: float
    joint: float
    independent_product: float


@dataclass(frozen=True)
class TeamBoxScore:
    player_ids: tuple
    stats: dict[str, np.ndarray]


@dataclass(frozen=True)
class BoxScoreSet:
    teams: dict[str, TeamBoxScore]
    n_sims: int
    environment: pd.DataFrame
    metadata: dict = field(default_factory=dict)
