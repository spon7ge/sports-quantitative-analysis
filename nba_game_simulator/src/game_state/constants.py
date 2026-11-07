from __future__ import annotations

REG_SECONDS = 2880
BLOWOUT_MARGIN = 20
RECENCY_DECAY = 0.7
CARRY_GAMES = 15
LATE_SECONDS = 360
LARGE_LEAD = 15

PRE = [
    "roll_min",
    "roll_min_sd",
    "roll_start",
    "roll_pf",
    "is_starter",
    "is_home",
    "team_exp_margin",
    "early_season",
    "team_change",
]
STATE = [
    "team_final_margin",
    "abs_final_margin",
    "game_minutes",
    "late_lead_min",
    "late_trail_min",
]

REQUIRED_PBP = {
    "gameId", "actionId", "clock", "period", "teamId", "personId",
    "playerName", "scoreHome", "scoreAway", "location",
    "description", "actionType", "subType",
}

ROLE_EDGES = [12, 26]
SCRIPT_EDGES = [-20, -10, 0, 10, 20]
