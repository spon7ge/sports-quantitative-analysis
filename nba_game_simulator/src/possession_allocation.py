"""Layer 4 entry points.

The implementation lives in game_opportunities. Existing imports stay here.
"""

from game_opportunities import (
    OpportunityModel,
    allocate_and_shoot,
    allocate_from_pool,
    allocate_game_opportunities,
    allocate_opportunities,
    as_nonneg_int_vector,
    fit_opportunities,
    live_opportunities,
    multinomial_rows,
    renormalize_props,
    sample_negbin,
    simulate_fta_ftm,
    simulate_game,
    training_opportunities,
)

__all__ = [
    "OpportunityModel",
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
