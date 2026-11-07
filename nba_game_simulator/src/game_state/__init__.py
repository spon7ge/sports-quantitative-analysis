"""Layer 2: shared game context -> conditional roster minutes.

Requires PBP plus official player logs with game/player/team IDs,
game date, and minutes.

Use simulate_minutes() with Layer 1 draws.
Margins and overtime are never redrawn here.

Historical injury snapshots are not inferred from who ultimately played.
"""
from __future__ import annotations

from game_state.constants import (
    BLOWOUT_MARGIN,
    CARRY_GAMES,
    LARGE_LEAD,
    LATE_SECONDS,
    PRE,
    RECENCY_DECAY,
    REG_SECONDS,
    REQUIRED_PBP,
    ROLE_EDGES,
    SCRIPT_EDGES,
    STATE,
)
from game_state.evaluate import evaluate_pipeline
from game_state.features import add_pregame_features, build_player_frame
from game_state.load import (
    game_ids,
    load_pbp,
    load_player_logs,
    minutes_value,
    read_files,
)
from game_state.replay import intervals, name_keys, period_bounds, replay_game
from game_state.roster import historical_roster, pregame_players
from game_state.simulate import allocate_minutes, simulate_minutes
from game_state.tables import build_tables, starters_on_floor_table
from game_state.train import (
    date_split,
    new_model,
    role_index,
    script_library,
    train_models,
)

__all__ = [
    "BLOWOUT_MARGIN",
    "CARRY_GAMES",
    "LARGE_LEAD",
    "LATE_SECONDS",
    "PRE",
    "RECENCY_DECAY",
    "REG_SECONDS",
    "REQUIRED_PBP",
    "ROLE_EDGES",
    "SCRIPT_EDGES",
    "STATE",
    "add_pregame_features",
    "allocate_minutes",
    "build_player_frame",
    "build_tables",
    "date_split",
    "evaluate_pipeline",
    "game_ids",
    "historical_roster",
    "intervals",
    "load_pbp",
    "load_player_logs",
    "minutes_value",
    "name_keys",
    "new_model",
    "period_bounds",
    "pregame_players",
    "read_files",
    "replay_game",
    "role_index",
    "script_library",
    "simulate_minutes",
    "starters_on_floor_table",
    "train_models",
]
