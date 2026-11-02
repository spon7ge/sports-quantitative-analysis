"""Points feature list, rate target, and the train-only Tukey fence."""

from __future__ import annotations

import pandas as pd
import pytest
from models.shared.metrics import RATE_Q50_TIERS
from models.shared.splits import tukey_fences
from points_xgb.config import RATE_TIERS
from points_xgb.features import (
    apply_train_tukey,
    assert_no_leak,
    assign_rate_target,
    build_model_frame,
)


def test_leak_guard_rejects_game_outcomes():
    assert_no_leak(["pts_rolling_10", "starting"])
    with pytest.raises(AssertionError):
        assert_no_leak(["pts_rolling_10", "minutes"])


def test_model_frame_adds_minutes_availability_features():
    frame = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 1, 2, 3],
            "team_id": [10, 10, 10, 10, 10, 10],
            "opp_team_id": [20, 20, 20, 20, 20, 20],
            "game_id": [1, 1, 1, 2, 2, 2],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03", "2024-01-03"]
            ),
            "season_year": ["2023-24"] * 6,
            "season_type": ["Regular Season"] * 6,
            "matchup": ["BOS vs. NYK"] * 6,
            "minutes": [30.0, 20.0, 10.0, 28.0, 12.0, 0.0],
            "start_position": ["G", "F", "C", "G", "F", ""],
            "pts": [20.0, 8.0, 4.0, 18.0, 6.0, 0.0],
            "fga": [15.0, 6.0, 3.0, 14.0, 5.0, 0.0],
            "player_team_spread": [-4.5, -4.5, -4.5, -3.0, -3.0, -3.0],
            "game_total": [226.0, 226.0, 226.0, 218.0, 218.0, 218.0],
        }
    )
    featured = build_model_frame(frame)
    game2 = featured.loc[featured["game_id"].eq(2)].set_index("player_id")
    assert game2["total_stars_active"].tolist() == [2, 2, 2]
    assert game2.loc[1, "starter_min_missing_teammates"] == pytest.approx(10.0)
    assert game2.loc[2, "starter_min_missing_teammates"] == pytest.approx(10.0)
    assert game2.loc[3, "starter_min_missing_teammates"] == pytest.approx(0.0)


def test_model_frame_carries_minutes_role_and_market_columns():
    frame = pd.DataFrame(
        {
            "player_id": [1, 2, 1, 2],
            "team_id": [10, 10, 10, 10],
            "opp_team_id": [20, 20, 20, 20],
            "game_id": [1, 1, 2, 2],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]
            ),
            "season_year": ["2023-24"] * 4,
            "season_type": ["Regular Season"] * 4,
            "matchup": ["BOS vs. NYK"] * 4,
            "minutes": [30.0, 12.0, 28.0, 10.0],
            "start_position": ["G", "", "G", ""],
            "pts": [20.0, 4.0, 18.0, 6.0],
            "fga": [15.0, 4.0, 14.0, 5.0],
            "player_team_spread": [-4.5, -4.5, -3.0, -3.0],
            "game_total": [226.0, 226.0, 218.0, 218.0],
        }
    )
    featured = build_model_frame(frame)
    second = featured.loc[featured["game_id"].eq(2)].set_index("player_id")
    assert second.loc[1, "starter_rank"] == 1
    assert second.loc[2, "starter_rank"] == 2
    assert second.loc[1, "start_rate_10"] == pytest.approx(1.0)
    assert second.loc[1, "player_team_spread"] == pytest.approx(-3.0)
    assert second.loc[1, "game_total"] == pytest.approx(218.0)


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
