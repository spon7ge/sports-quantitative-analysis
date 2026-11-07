"""Layer 4: prior-only opportunity rates and conditional player box scores.

Player rates use sums of events divided by sums of exposure. Team pools are a
box-score approximation. Layer 3 segments can keep events on the court.
"""

from __future__ import annotations

from game_opportunities.allocate import allocate_opportunities
from game_opportunities.checks import as_nonneg_int_vector
from game_opportunities.constants import MODEL_VERSION, OPPORTUNITY_RATES
from game_opportunities.counts import (
    allocate_and_shoot,
    allocate_from_pool,
    multinomial_rows,
    renormalize_props,
    sample_negbin,
    simulate_fta_ftm,
)
from game_opportunities.model import OpportunityModel, SimulationResult, TeamSimResult
from game_opportunities.segments import allocate_game_opportunities
from game_opportunities.simulate import simulate_game
from game_opportunities.train import (
    fit_opportunities,
    live_opportunities,
    training_opportunities,
)

__all__ = [
    "MODEL_VERSION",
    "OPPORTUNITY_RATES",
    "OpportunityModel",
    "SimulationResult",
    "TeamSimResult",
    "allocate_and_shoot",
    "allocate_from_pool",
    "allocate_game_opportunities",
    "allocate_opportunities",
    "as_nonneg_int_vector",
    "fit_opportunities",
    "live_opportunities",
    "multinomial_rows",
    "renormalize_props",
    "sample_negbin",
    "simulate_fta_ftm",
    "simulate_game",
    "training_opportunities",
]
