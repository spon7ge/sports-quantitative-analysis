"""Points feature list, rate target, and the train-only Tukey fence."""

from __future__ import annotations

import pandas as pd
import pytest

from points_xgb.config import PTS_FEATURES, RATE_TIERS
from points_xgb.features import apply_train_tukey, assert_no_leak, assign_rate_target
from models.shared.metrics import RATE_Q50_TIERS
from models.shared.splits import tukey_fences


def test_feature_list_matches_the_model_and_rejects_leakage():
    assert PTS_FEATURES == [
        "pts_per_min_ewm_hl_10",
        "player_fga_share_10",
        "ppm_trim_mean_20",
        "current_team_pts_per_min",
        "usg_wmean_10",
        "season_pts_per_min",
        "starting",
        "fta_per_min_10",
        "fga_per_min_10",
        "fg3a_per_min_10",
        "pts_per_min_mean_vs_opp",
        "fga_per_min_mean_vs_opp",
        "fta_per_min_mean_vs_opp",
        "fg3a_per_min_mean_vs_opp",
        "ppm_unstable",
        "opp_def_rating_mean_10",
        "min_mean_10",
        "min_sd_10",
        "minutes_ewm_10",
        "ts_pct_10",
        "opp_pace_10",
        "ppm_sd_10",
    ]
    assert_no_leak(PTS_FEATURES)
    with pytest.raises(AssertionError):
        assert_no_leak([*PTS_FEATURES, "minutes"])


def test_rate_tier_names_match_the_shared_cuts():
    assert list(RATE_TIERS) == [name for name, _low, _high in RATE_Q50_TIERS]
    values = pd.Series([0.29, 0.30, 0.50, 0.70])
    assert list(RATE_TIERS["<0.30"](values)) == [True, False, False, False]
    assert list(RATE_TIERS["0.30-0.50"](values)) == [False, True, False, False]
    assert list(RATE_TIERS["0.50-0.70"](values)) == [False, False, True, False]
    assert list(RATE_TIERS["0.70+"](values)) == [False, False, False, True]


def test_train_tukey_fence_ignores_a_holdout_rate_of_100():
    train = pd.DataFrame({"pts_per_min": [0.2, 0.3, 0.4, 0.5, 0.35]})
    holdout = pd.DataFrame({"pts_per_min": [100.0]})
    kept, low, high, n_dropped = apply_train_tukey(train)
    expected_low, expected_high = tukey_fences(train["pts_per_min"])
    moved_low, moved_high = tukey_fences(
        pd.concat([train, holdout], ignore_index=True)["pts_per_min"]
    )
    assert (low, high) == (expected_low, expected_high)
    assert (low, high) != (moved_low, moved_high)
    assert n_dropped == 0
    assert len(kept) == len(train)


def test_assign_rate_target_caps_and_lags_within_player():
    frame = pd.DataFrame(
        {
            "player_id": [1, 1],
            "game_id": [10, 11],
            "season_year": ["2024-25", "2024-25"],
            "game_date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "pts": [18.0, 10.0],
            "minutes": [2.0, 20.0],
        }
    )
    rated = assign_rate_target(frame)
    assert rated["pts_per_min"].tolist() == [0.5, 6.0]
    assert rated["pts_per_min_lag_1"].iloc[0] != rated["pts_per_min_lag_1"].iloc[0]
    assert rated["pts_per_min_lag_1"].iloc[1] == pytest.approx(0.5)
