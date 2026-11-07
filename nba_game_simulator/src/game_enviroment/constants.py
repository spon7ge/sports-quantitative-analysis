from __future__ import annotations

from lineup import LINEUP_GAME_FEATURES

REG_MIN, OT_MIN, REST_CAP = 48.0, 5.0, 7

REQUIRED = [
    "season_year", "game_id", "game_date", "team_id", "team_abbreviation",
    "matchup", "min", "team_poss", "opp_poss", "team_pts", "opp_pts",
]
OPTIONAL = [
    "season_type", "player_id", "start_position", "game_total", "team_spread",
    "player_team_spread", "pace", "off_rating", "def_rating",
    "team_pace", "team_off_rating", "team_def_rating",
]
FEATURES = [
    "pace_h", "pace_a", "b2b_h", "b2b_a", "rested_h", "rested_a",
    "total_dev", "abs_spread", "total_missing", "spread_missing",
    *LINEUP_GAME_FEATURES,
]
SPREAD_EDGES = [0.0, 3.0, 6.0, 9.0, float("inf")]
ENVIRONMENT_COLUMNS = [
    "team_poss", "opp_poss", "pace", "team_pts", "opp_pts", "game_total",
    "is_home", "n_ot", "game_minutes", "team_margin",
]
