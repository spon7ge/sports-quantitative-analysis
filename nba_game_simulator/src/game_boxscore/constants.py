"""Layer 6 sides, upstream modules, and box-score stat names."""

from __future__ import annotations

SIDES = ("home", "away")

# Module and function for each upstream layer. Edit the module names to
# match the project's file names.
LAYER_MODULES = {
    "environment": ("game_enviroment", "simulate_game"),
    "minutes": ("minutes", "simulate_minutes"),
    "segments": ("pbp_rotation", "learned_segments_from_minutes"),
    "opportunities": ("opportunities", "allocate_game_opportunities"),
    "events": ("shared_events", "allocate_events"),
}

_SHOT_STATS = ("FGA", "FGM", "2PA", "2PM", "3PA", "3PM", "FTA", "FTM", "PTS", "TOV")
_EVENT_STATS = ("AST", "OREB", "DREB", "REB", "STL", "BLK")
_LAYER4_STATS = (*_SHOT_STATS, "AST", "OREB")
# Count arrays Layer 5 reads directly. They are not box-score columns.
_LAYER4_EVENT_INPUTS = (
    "ZONE_FGA",
    "ZONE_FGM",
    "LIVE_TOV",
    "DEAD_TOV",
    "MISS_BY_ZONE",
)
_COUNT_STATS = (*_SHOT_STATS, *_EVENT_STATS)
_ENVIRONMENT_COLUMNS = (
    "team_poss",
    "opp_poss",
    "pace",
    "team_pts",
    "opp_pts",
    "game_total",
    "is_home",
    "n_ot",
    "game_minutes",
    "team_margin",
)
# Swapped when changing a Layer 1 row from one team's view to the other's.
_VIEW_PAIRS = (("team_poss", "opp_poss"), ("team_pts", "opp_pts"))
_COMBOS = {
    "PRA": ("PTS", "REB", "AST"),
    "PR": ("PTS", "REB"),
    "PA": ("PTS", "AST"),
    "RA": ("REB", "AST"),
    "STOCKS": ("STL", "BLK"),
}
_DOUBLE_STATS = ("PTS", "REB", "AST", "STL", "BLK")
_MINUTE_TOLERANCE = 1e-3
