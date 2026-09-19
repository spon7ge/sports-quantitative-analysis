"""Assists feature manifests."""

from __future__ import annotations

CURRENT_ASSISTS_FEATURES = [
    "predicted_minutes_oof",
    "start_rate_10",
    "ast_mean_10",
    "ast_mean_20",
    "assists_per_min_10",
    "team_ast_mean_10",
    "team_fgm_mean_10",
    "team_pace_mean_10",
    "opponent_team_ast_allowed_mean_10",
    "opponent_team_pace_mean_10",
    "days_rest",
    "is_home",
]

ASSISTS_FEATURE_CHALLENGERS = {
    "exposure_history": ["min_mean_10"],
    "usage_role": ["usg_wmean_10"],
    "position_role": ["position_guard_prior"],
    "assist_efficiency": ["ast_pct_wmean_10"],
    "passing_tracking": ["passes_per_min_10"],
    "opponent_make_environment": [
        "opponent_team_fgm_allowed_mean_10",
    ],
}
