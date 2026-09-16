"""Points feature manifests.

The default list is the stacked minutes-to-points
model. Actual Game N minutes are never a feature;
use chronological out-of-fold minute predictions.
DNP-rate features are omitted until a pregame
candidate universe exists.
"""

from __future__ import annotations

PLAYING_TIME_FEATURES = [
    "predicted_minutes_oof",
    "min_lag_1",
    "min_mean_3",
    "min_mean_10",
    "min_ewm_hl_3",
    "min_std_10",
    "start_rate_10",
    "season_min_mean",
    "current_team_min_mean",
]

SCORING_FEATURES = [
    "pts_lag_1",
    "pts_mean_3",
    "pts_mean_10",
    "pts_mean_20",
    "pts_ewm_hl_3",
    "active_pts_mean_10",
    "pts_std_10",
    "season_pts_mean",
]

RATE_FEATURES = [
    "pts_per_min_10",
    "pts_per_min_20",
    "season_pts_per_min",
    "current_team_pts_per_min",
    "fga_per_min_10",
    "fg3a_per_min_10",
    "fta_per_min_10",
    "usg_wmean_10",
    "ts_agg_20",
    "three_attempt_rate_10",
    "free_throw_rate_10",
    "touches_per_min_10",
    "player_fga_share_10",
]

CONTEXT_FEATURES = [
    "team_pace_mean_10",
    "team_off_rating_mean_10",
    "opp_pace_mean_10",
    "opp_def_rating_mean_10",
    "expected_possessions",
    "game_total",
    "team_spread_canonical",
    "abs_spread",
    "implied_team_total",
    "team_days_since_prev_game",
    "is_back_to_back",
    "player_days_since_appearance",
    "is_home",
    "season_type_cat",
]

TREND_FEATURES = [
    "pts_mean_3_minus_10",
    "ppm_10_minus_season",
    "fga_per_min_5_minus_season",
]

STACKED_INTERACTIONS = [
    "expected_points_rate",
    "expected_attempt_volume",
    "usage_x_expected_possessions",
]

DIRECT_POINTS_FEATURES = (
    PLAYING_TIME_FEATURES[1:]
    + SCORING_FEATURES
    + RATE_FEATURES
    + CONTEXT_FEATURES
    + TREND_FEATURES
    + ["usage_x_expected_possessions"]
)

TIER1_POINTS_FEATURES = (
    PLAYING_TIME_FEATURES
    + SCORING_FEATURES
    + RATE_FEATURES
    + CONTEXT_FEATURES
    + TREND_FEATURES
    + STACKED_INTERACTIONS
)

# Production notebook current41 mean-model contract.
# Omits optional add-ins min_mean_10 / pts_mean_10 and
# treats pts_std_10 as a sampler-conditioning candidate
# rather than a mean feature.
CURRENT_POINTS_FEATURES = (
    "predicted_minutes_oof",
    "min_lag_1",
    "min_mean_3",
    "min_ewm_hl_3",
    "min_std_10",
    "start_rate_10",
    "current_team_min_mean",
    "pts_lag_1",
    "pts_mean_3",
    "pts_mean_20",
    "pts_ewm_hl_3",
    "active_pts_mean_10",
    "season_pts_mean",
    "pts_per_min_10",
    "pts_per_min_20",
    "current_team_pts_per_min",
    "fga_per_min_10",
    "fta_per_min_10",
    "usg_wmean_10",
    "ts_agg_20",
    "three_attempt_rate_10",
    "free_throw_rate_10",
    "player_fga_share_10",
    "team_pace_mean_10",
    "team_off_rating_mean_10",
    "opp_pace_mean_10",
    "opp_def_rating_mean_10",
    "expected_possessions",
    "team_spread_canonical",
    "abs_spread",
    "implied_team_total",
    "team_days_since_prev_game",
    "is_back_to_back",
    "player_days_since_appearance",
    "is_home",
    "season_type_cat",
    "pts_mean_3_minus_10",
    "ppm_10_minus_season",
    "expected_points_rate",
    "expected_attempt_volume",
    "usage_x_expected_possessions",
)
CURRENT_POINTS_41 = CURRENT_POINTS_FEATURES

# Lean drops four suspected-noise columns that are not
# the minutes stack, not the three STAY volume
# interactions, and not fga_per_min_10 (kept so
# expected_attempt_volume can still be computed).
# Dropped: longer-window scoring/rate duplicates
# (pts_mean_20, pts_per_min_20), the overlapping
# current-team rate (current_team_pts_per_min), and
# the unstable attempt-share (player_fga_share_10).
_LEAN_POINTS_DROPS = frozenset(
    {
        "pts_mean_20",
        "pts_per_min_20",
        "current_team_pts_per_min",
        "player_fga_share_10",
    }
)
LEAN_POINTS_FEATURES = tuple(
    column
    for column in CURRENT_POINTS_FEATURES
    if column not in _LEAN_POINTS_DROPS
)
LEAN_POINTS_37 = LEAN_POINTS_FEATURES


def _with_column_after(
    columns: tuple[str, ...],
    after: str,
    column: str,
) -> tuple[str, ...]:
    items = list(columns)
    items.insert(items.index(after) + 1, column)
    return tuple(items)


# Optional add-ins the notebook dropped from current41.
CURRENT_PLUS_PTS_MEAN_10 = _with_column_after(
    CURRENT_POINTS_FEATURES,
    "pts_mean_3",
    "pts_mean_10",
)
CURRENT_PLUS_MIN_MEAN_10 = _with_column_after(
    CURRENT_POINTS_FEATURES,
    "min_mean_3",
    "min_mean_10",
)

# Sampler-conditioning only; not a mean-model contract.
POINTS_SAMPLER_FEATURES = ("pts_std_10",)
