"""Minutes feature manifests.

The default list is appearance-conditional: silver
gamelogs almost never include documented DNPs, so
DNP-rate features are omitted until a pregame
candidate universe exists.
"""

from __future__ import annotations

# Production selected list (notebook current37). Interactions
# start_rate_x_abs_spread and min10_x_expected_possessions stay.
CURRENT_MINUTES_FEATURES = [
    "min_lag_1",
    "min_mean_3",
    "min_mean_10",
    "min_mean_20",
    "min_ewm_hl_3",
    "min_std_10",
    "start_rate_10",
    "season_rows_prior",
    "season_appearances_prior",
    "season_min_mean",
    "season_min_std",
    "season_start_rate",
    "prior_season_min_mean",
    "prior_season_appearances",
    "current_team_min_mean",
    "current_team_rows_prior",
    "usg_wmean_10",
    "fga_per_min_10",
    "assists_per_min_10",
    "team_pace_mean_10",
    "team_net_rating_mean_10",
    "team_net_rating_season",
    "opp_pace_mean_10",
    "opp_net_rating_mean_10",
    "expected_possessions",
    "team_days_since_prev_game",
    "team_games_prev_3d",
    "player_days_since_appearance",
    "player_appearances_prev_7d",
    "is_home",
    "team_spread_canonical",
    "abs_spread",
    "implied_team_total",
    "min_mean_3_minus_10",
    "min_mean_10_minus_season",
    "start_rate_x_abs_spread",
    "min10_x_expected_possessions",
]
CURRENT_MINUTES_37 = CURRENT_MINUTES_FEATURES

ROLE_TAIL_MINUTES_FEATURES = [
    *CURRENT_MINUTES_FEATURES,
    "min_ge_30_rate_10",
    "is_back_to_back",
    "team_games_prev_5d",
    "usg_5_minus_season",
]

# Lean ablation of CURRENT. Dropped redundant volume / team-rate
# extras: min_mean_20, season_rows_prior, season_appearances_prior,
# prior_season_appearances, current_team_rows_prior, fga_per_min_10,
# assists_per_min_10, team_pace_mean_10, team_net_rating_mean_10,
# team_net_rating_season, opp_pace_mean_10, opp_net_rating_mean_10.
# Kept minutes recency, start_rate_10, season_min_mean, usg_wmean_10,
# expected_possessions, is_home, spreads, and both interactions.
LEAN_MINUTES_FEATURES = [
    "min_lag_1",
    "min_mean_3",
    "min_mean_10",
    "min_ewm_hl_3",
    "min_std_10",
    "start_rate_10",
    "season_min_mean",
    "season_min_std",
    "season_start_rate",
    "prior_season_min_mean",
    "current_team_min_mean",
    "usg_wmean_10",
    "expected_possessions",
    "team_days_since_prev_game",
    "team_games_prev_3d",
    "player_days_since_appearance",
    "player_appearances_prev_7d",
    "is_home",
    "team_spread_canonical",
    "abs_spread",
    "implied_team_total",
    "min_mean_3_minus_10",
    "min_mean_10_minus_season",
    "start_rate_x_abs_spread",
    "min10_x_expected_possessions",
]

TIER1_MINUTES_FEATURES = [
    "min_lag_1",
    "min_mean_3",
    "min_mean_10",
    "min_mean_20",
    "min_ewm_hl_3",
    "min_std_10",
    "min_ge_30_rate_10",
    "start_rate_10",
    "season_rows_prior",
    "season_appearances_prior",
    "season_min_mean",
    "season_min_std",
    "season_start_rate",
    "prior_season_min_mean",
    "prior_season_appearances",
    "current_team_min_mean",
    "current_team_rows_prior",
    "usg_wmean_10",
    "touches_per_min_10",
    "passes_per_min_10",
    "fga_per_min_10",
    "assists_per_min_10",
    "activity_per_min_10",
    "team_pace_mean_10",
    "team_net_rating_mean_10",
    "team_net_rating_season",
    "opp_pace_mean_10",
    "opp_net_rating_mean_10",
    "expected_possessions",
    "team_days_since_prev_game",
    "is_back_to_back",
    "team_games_prev_3d",
    "team_games_prev_5d",
    "player_days_since_appearance",
    "player_appearances_prev_7d",
    "is_home",
    "season_type_cat",
    "game_total",
    "team_spread_canonical",
    "abs_spread",
    "implied_team_total",
    "market_missing",
    "min_mean_3_minus_10",
    "min_mean_10_minus_season",
    "usg_5_minus_season",
    "start_rate_x_abs_spread",
    "min10_x_expected_possessions",
]

DNP_FEATURES = [
    "dnp_rate_10",
    "active_min_mean_10",
]

SEASON_TYPE_CODES = {
    "Regular Season": 0,
    "Playoffs": 1,
    "Play-In": 2,
    "PlayIn": 2,
}
