"""Layer 1: one shared game environment per simulation.

Train from player parquet logs or load a saved model. Live spreads belong to
the target team: negative means favored. --spread-convention describes logs.
Overtime is a pregame approximation, not a tied-score possession simulation.
Retrain existing models: this simplified version writes model version 3.
"""
from __future__ import annotations

from game_enviroment.constants import ENVIRONMENT_COLUMNS, FEATURES, REST_CAP
from game_enviroment.features import build_features
from game_enviroment.games import build_game_table, load_gamelogs, season_of
from game_enviroment.backtest import run_backtest
from game_enviroment.rotation import (
    build_rotation_history,
    sample_rotation,
)
from game_enviroment.simulate import (
    load_model,
    print_sim_summary,
    save_model,
    simulate_game,
)
from game_enviroment.slate import load_latest_slate, load_live_slate, player_slate
from game_enviroment.train import train

__all__ = [
    "ENVIRONMENT_COLUMNS",
    "FEATURES",
    "REST_CAP",
    "build_features",
    "build_game_table",
    "load_gamelogs",
    "load_latest_slate",
    "load_live_slate",
    "build_rotation_history",
    "load_model",
    "player_slate",
    "run_backtest",
    "sample_rotation",
    "print_sim_summary",
    "save_model",
    "season_of",
    "simulate_game",
    "train",
]
