"""Layer 5 rate names, windows, and column sets."""

from __future__ import annotations

EVENT_MODEL_VERSION = 2

_STATS = ("AST", "OREB", "DREB", "REB", "STL", "BLK")
_WEIGHTED_STATS = ("AST", "OREB", "DREB", "STL", "BLK")
_UPSTREAM_STATS = ("AST", "OREB")
_SHOT_KEYS = ("FGA", "FGM", "TOV")

PLAYER_EVENT_RATES = (
    "ast_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
)
TEAM_EVENT_RATES = (
    "assisted_rate",
    "p_rebound",
    "p_oreb",
    "p_oreb_allowed",
    "p_steal",
    "p_steal_against",
    "p_block",
    "p_block_against",
)
EVENT_RATES = (*PLAYER_EVENT_RATES, *TEAM_EVENT_RATES)
POOL_DISPERSION_KEYS = (
    "assisted_rate",
    "p_rebound",
    "p_oreb",
    "dreb_given_oreb",
    "p_steal",
    "p_block",
)

_PLAYER_SOURCES = {
    "ast_per_min": "ast",
    "oreb_per_min": "oreb",
    "dreb_per_min": "dreb",
    "stl_per_min": "stl",
    "blk_per_min": "blk",
}
_RATE_WEIGHT_STAT = {
    "AST": "ast_per_min",
    "OREB": "oreb_per_min",
    "DREB": "dreb_per_min",
    "STL": "stl_per_min",
    "BLK": "blk_per_min",
}
_EVENT_IDS = ("player_id", "game_id", "game_date", "team_id")
_EVENT_COLUMNS = ("ast", "oreb", "dreb", "stl", "blk")
_EVENT_TEAM_COLUMNS = (
    "team_fga",
    "team_fgm",
    "team_ast",
    "team_oreb",
    "team_stl",
    "team_blk",
    "opp_dreb",
    "opp_tov",
    "opp_fga",
)

PLAYER_WINDOW = 10
TEAM_WINDOW = 20
PLAYER_PRIOR_MINUTES = 60.0
# Roughly three team-games of each rate's denominator. ``team_prior_scale``
# multiplies all of them.
_TEAM_PRIOR_TRIALS = {
    "assisted_rate": 120.0,
    "p_rebound": 140.0,
    "p_oreb": 130.0,
    "p_oreb_allowed": 130.0,
    "p_steal": 45.0,
    "p_steal_against": 45.0,
    "p_block": 140.0,
    "p_block_against": 140.0,
}
_PLAYER_GRID = ((5, 10, 20), (30.0, 60.0, 120.0, 240.0))
_TEAM_GRID = ((10, 20, 40), (0.5, 1.0, 2.0))
_MIN_TUNING_DATES = 8
_MIN_DISPERSION_GROUPS = 50
_MAX_POOL_RHO = 0.5
_MINUTE_TOLERANCE = 1e-3
_TIME_TOLERANCE = 1e-6

EVALUATED_STATS = _STATS
_STREAM_PURPOSES = (
    "rebounds",
    "OREB",
    "DREB",
    "steals",
    "STL",
    "blocks",
    "BLK",
    "AST",
)
_REQUIRED_TEAM_FIELDS = (
    "player_ids",
    "fga",
    "fgm",
    "tov",
    "available",
    "projected_minutes",
    "reference_minutes",
    "oreb_rate",
    "dreb_rate",
    "creation_rate",
    "assisted_rate",
    "stl_rate",
    "blk_rate",
)
