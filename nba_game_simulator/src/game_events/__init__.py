"""Layer 5: rebounds, steals, blocks, and assists from shared game events.

Player totals are allocations of events the game already produced. Layer 4
owns attempts, makes, turnovers, points, and, when it supplies them, assists
and offensive rebounds. Layer 5 adds the defensive rebounds, steals, and
blocks those quantities imply. Assists and offensive rebounds are reused
from Layer 4 when that draw supplied them.

With play-by-play links, an assist is P(assisted | shooter, zone) and then
P(passer | shooter, lineup). A rebound chance is a miss while the player was
on the floor. A block is a rim attempt. A steal is a live-ball turnover.
Without those links, weights are the box-score rates.

With Layer 3 segments, an event is credited only to players on the floor
when it happened. Without segments, weights are rate times Layer 2 minutes.
"""

from __future__ import annotations

from game_events.allocate import allocate_events
from game_events.constants import EVALUATED_STATS, EVENT_MODEL_VERSION, EVENT_RATES
from game_events.credit import CreditRates, credit_rates
from game_events.evaluate import score_event_forecasts, summarize_event_scores
from game_events.links import EventLinks, event_links
from game_events.model import EventModel, load_event_model, save_event_model
from game_events.rates import live_events, training_events
from game_events.shares import allocate_joint_assists, event_shares
from game_events.simulate import (
    SharedEventResult,
    TeamSharedStats,
    simulate_shared_events,
)
from game_events.train import fit_events

__all__ = [
    "EVENT_MODEL_VERSION",
    "EVENT_RATES",
    "EVALUATED_STATS",
    "CreditRates",
    "EventLinks",
    "EventModel",
    "SharedEventResult",
    "TeamSharedStats",
    "allocate_events",
    "allocate_joint_assists",
    "credit_rates",
    "event_links",
    "event_shares",
    "fit_events",
    "live_events",
    "load_event_model",
    "save_event_model",
    "score_event_forecasts",
    "simulate_shared_events",
    "summarize_event_scores",
    "training_events",
]
