"""Layer 4 rate names, defaults, and column aliases."""

from __future__ import annotations

MODEL_VERSION = 3

_TOL = 1e-6

# Share of turnovers that are live-ball when the profile has no shrunk rate.
# Not an estimate from a sample. A profile column replaces it.
LIVE_TOV_PRIOR = 0.5

_MAX_POOL = 10000

_DEFAULTS = {
    "fga_per_min": 0.40,
    "p_3pa": 0.40,
    "p_3p": 0.36,
    "p_2p": 0.54,
    "fta_per_fga": 0.25,
    "ft_pct": 0.78,
    "tov_per_min": 0.06,
    "fta_per_min": 0.10,
    "ast_per_min": 0.10,
    "oreb_per_min": 0.04,
    "fga_per_poss": 0.88,
    "tov_per_poss": 0.13,
    "fta_per_poss": 0.22,
}

_PLAYER_SPEC = {
    "fga_per_min": ("fga", "minutes"),
    "p_3pa": ("fg3_a", "fga"),
    "p_3p": ("fg3_m", "fg3_a"),
    "p_2p": ("fg2_m", "fg2_a"),
    "fta_per_fga": ("fta", "fga"),
    "ft_pct": ("ftm", "fta"),
    "tov_per_min": ("tov", "minutes"),
    "fta_per_min": ("fta", "minutes"),
    "ast_per_min": ("ast", "minutes"),
    "oreb_per_min": ("oreb", "minutes"),
}

_TEAM_SPEC = {
    "fga_per_poss": ("team_fga", "team_poss"),
    "tov_per_poss": ("team_tov", "team_poss"),
    "fta_per_poss": ("team_fta", "team_poss"),
}

_PLAYER_RATES = tuple(_PLAYER_SPEC)

_TEAM_RATES = tuple(_TEAM_SPEC)

OPPORTUNITY_RATES = (*_PLAYER_RATES, *_TEAM_RATES)

_PROBABILITIES = {
    "p_3pa",
    "p_3p",
    "p_2p",
    "ft_pct",
    "tov_per_poss",
}

_IDS = ("player_id", "game_id", "game_date", "team_id")

_SHOTS = ("fga", "fgm", "fg3_a", "fg3_m", "fta", "ftm", "tov")

_TEAM = ("team_fga", "team_poss", "team_tov")

_ALIASES = {
    "player_id": ("PLAYER_ID", "playerId", "personId"),
    "game_id": ("GAME_ID", "gameId"),
    "game_date": ("GAME_DATE", "gameDate"),
    "team_id": ("TEAM_ID", "teamId"),
    "minutes": ("MIN", "min", "MINUTES"),
    "fga": ("FGA", "FG_A"),
    "fgm": ("FGM", "FG_M"),
    "fg3_a": ("FG3A", "FG3_A"),
    "fg3_m": ("FG3M", "FG3_M"),
    "fta": ("FTA", "FT_A"),
    "ftm": ("FTM", "FT_M"),
    "tov": ("TOV", "TO"),
    "ast": ("AST",),
    "oreb": ("OREB",),
    "team_fga": ("TEAM_FGA", "TEAM_FG_A"),
    "team_poss": ("TEAM_POSS", "team_possessions"),
    "team_tov": ("TEAM_TOV", "TEAM_TO"),
    "team_fta": ("TEAM_FTA", "TEAM_FT_A"),
    "game_minutes": ("GAME_MINUTES",),
}
