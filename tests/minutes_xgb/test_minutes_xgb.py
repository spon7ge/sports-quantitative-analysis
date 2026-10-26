"""Behavior extracted from the minutes XGBoost notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from minutes_xgb.baseline import clustered_mae_bootstrap, point_row
from minutes_xgb.config import MIN_FEATURES
from minutes_xgb.data import data_quality, load_regular_season
from minutes_xgb.errors import build_miss_frame
from minutes_xgb.features import add_total_stars_active
from minutes_xgb.importance import feature_consensus
from minutes_xgb.paths import repo_root
from minutes_xgb.scoring import (
    interval_score,
    quantile_calibration,
    quantile_crossing,
    score_predictions,
    wis_values,
)
from minutes_xgb.slices import slice_score


def test_selected_features_include_home_and_opponent_history() -> None:
    assert "is_home" in MIN_FEATURES
    assert "min_mean_vs_opp" in MIN_FEATURES
    assert "games_vs_opp" in MIN_FEATURES
    assert len(MIN_FEATURES) == 16


def test_repo_root_finds_pyproject() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()


def test_total_stars_active_counts_only_top_three_who_played() -> None:
    frame = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1, 2],
            "game_id": [10, 10, 10, 10, 11],
            "starter_rank": [1, 2, 3, 4, 1],
            "minutes": [30, 0, 20, 35, np.nan],
        }
    )
    featured = add_total_stars_active(frame)
    assert featured["total_stars_active"].tolist() == [2, 2, 2, 2, 0]


def test_load_regular_season_flags_starters(tmp_path: Path) -> None:
    season = tmp_path / "data" / "silver" / "nba" / "2019-20" / "regular_season"
    season.mkdir(parents=True)
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "game_id": [10, 10],
            "minutes": [30, 12],
            "start_position": [" G ", ""],
        }
    ).to_parquet(season / "player_gamelogs.parquet")

    loaded = load_regular_season(tmp_path, seasons=("2019-20",))

    assert loaded["starting"].tolist() == [1, 0]
    assert len(loaded) == 2


def test_data_quality_counts_duplicate_player_games() -> None:
    frame = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "game_id": [10, 10, 11],
            "minutes": [10.0, 20.0, 30.0],
            "starting": [1, 1, 0],
        }
    )
    stats = data_quality(frame)
    assert stats["rows"] == 3
    assert stats["duplicate_player_games"] == 1
    assert stats["players"] == 2


def test_score_predictions_pinball_and_coverage() -> None:
    y_true = np.array([0.0, 10.0])
    preds = {
        "q_0.10": np.array([-1.0, -1.0]),
        "q_0.50": np.array([0.0, 0.0]),
        "q_0.90": np.array([1.0, 1.0]),
    }
    pinball, coverage = score_predictions(y_true, preds, "fold")
    q50 = pinball.loc[pinball["quantile"].eq(0.50), "pinball"].iloc[0]
    assert q50 == 2.5
    band = coverage.loc[coverage["interval"].eq("80%"), "coverage"].iloc[0]
    assert band == 0.5


def test_quantile_crossing_counts_an_inverted_row() -> None:
    preds = {
        "q_0.10": np.array([10.0, 10.0]),
        "q_0.50": np.array([20.0, 5.0]),
        "q_0.90": np.array([30.0, 30.0]),
    }
    n_crossed, n_rows, pairs = quantile_crossing(preds)
    assert n_rows == 2
    assert n_crossed == 1
    pair = pairs.loc[
        pairs["lower"].eq("q_0.10") & pairs["upper"].eq("q_0.50"),
        "crossing_rate",
    ].iloc[0]
    assert pair == 0.5


def test_calibration_delta_is_empirical_minus_target() -> None:
    table = quantile_calibration(
        np.array([1.0, 2.0]),
        {"q_0.50": np.array([0.0, 3.0])},
    )
    assert table["empirical"].iloc[0] == 0.5
    assert table["delta"].iloc[0] == 0.0


def test_interval_score_penalizes_a_miss_below_the_band() -> None:
    inside = interval_score(
        np.array([10.0]),
        np.array([8.0]),
        np.array([12.0]),
        0.80,
    )
    below = interval_score(
        np.array([5.0]),
        np.array([8.0]),
        np.array([12.0]),
        0.80,
    )
    assert inside[0] == 4.0
    assert below[0] == 4.0 + (2.0 / 0.80) * 3.0


def test_wis_uses_the_median_and_one_present_interval() -> None:
    qmap = {
        "q_0.50": np.array([10.0]),
        "q_0.40": np.array([8.0]),
        "q_0.60": np.array([12.0]),
    }
    value = wis_values(np.array([10.0]), qmap)
    assert value[0] == (0.80 / 2.0) * 4.0 / 1.5


def test_clustered_bootstrap_is_zero_when_the_forecast_matches() -> None:
    y = np.array([10.0, 10.0, 20.0, 20.0])
    dates = np.array(["a", "a", "b", "b"])
    draws = clustered_mae_bootstrap(
        y,
        {"perfect": y.copy(), "off": y + 2.0},
        dates,
        n_boot=20,
        seed=0,
    )
    assert draws["perfect"].shape == (20,)
    assert np.all(draws["perfect"] == 0.0)
    assert np.all(draws["off"] == 2.0)


def test_point_row_bias_is_actual_minus_predicted() -> None:
    row = point_row("model", np.array([10.0, 14.0]), np.array([12.0, 10.0]))
    assert row["mae"] == 3.0
    assert row["mean_bias"] == 1.0


def test_slice_score_reports_an_empty_slice() -> None:
    empty = slice_score(
        "April",
        np.array([False, False]),
        np.array([10.0, 12.0]),
        {"q_0.50": np.array([9.0, 11.0])},
    )
    assert empty == {"slice": "April", "n": 0}


def test_miss_hint_marks_a_starter_who_fouled_out() -> None:
    history = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [7],
            "team_abbreviation": ["BOS"],
            "matchup": ["BOS vs. NYK"],
            "wl": ["W"],
            "pf": [6],
            "pfd": [1],
            "pts": [8],
            "plus_minus": [-3],
            "team_pts": [110],
            "opp_pts": [100],
            "abs_spread": [4.5],
            "minutes": [20.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [7],
            "player_name": ["A"],
            "game_date": ["2026-01-01"],
            "minutes": [20.0],
            "starting": [1],
            "pts": [8],
            "min_lag_1": [32.0],
        }
    )
    preds = {
        "q_0.10": np.array([28.0]),
        "q_0.50": np.array([32.0]),
        "q_0.90": np.array([36.0]),
    }
    miss = build_miss_frame(history, holdout, preds)
    assert miss["hint"].iloc[0] == "fouled out"
    assert miss["error"].iloc[0] == -12.0


def test_feature_consensus_keeps_a_significant_positive_ablation() -> None:
    shap = pd.Series({"a": 2.0, "b": 1.0})
    permutation = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [0.2, 0.0],
            "std": [0.01, 0.0],
        }
    )
    importances = np.array(
        [
            [0.20, 0.21, 0.19],
            [0.00, 0.00, 0.00],
        ]
    )
    ablation = pd.DataFrame(
        {"feature": ["a", "b"], "delta_pinball": [0.1, -0.1]}
    )
    audit = feature_consensus(
        ["a", "b"],
        shap,
        permutation,
        importances,
        ablation,
    )
    kept = audit.set_index("feature")["keep"]
    assert bool(kept["a"])
    assert not bool(kept["b"])


def test_oos_predictions_keep_raw_knots_and_add_calibrated_knots():
    from minutes_xgb.artifacts import build_oos_predictions
    from minutes_xgb.config import QUANTILES

    rows = 5
    identity = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(rows)],
            "player_id": np.arange(rows),
            "game_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "minutes": np.full(rows, 20.0),
            "pts": np.full(rows, 8.0),
            "starting": np.zeros(rows, dtype=int),
        }
    )
    oof = identity.copy()
    oof["fold_id"] = np.arange(1, rows + 1)
    oof["early_stop"] = "train_tail"
    oof["train_tail_frac"] = 0.10
    for level in QUANTILES:
        oof[f"q_{level:.2f}"] = 10.0 + level
    holdout = identity.copy()
    holdout["game_date"] = pd.date_range("2025-10-21", periods=rows, freq="D")
    preds = {f"q_{level:.2f}": np.full(rows, 15.0) for level in QUANTILES}
    oof_cal = oof[[f"q_{level:.2f}" for level in QUANTILES]] + 1.0
    preds_cal = {key: values + 1.0 for key, values in preds.items()}

    out = build_oos_predictions(
        {"oof": oof},
        identity,
        holdout,
        preds,
        oof_cal=oof_cal,
        preds_ho_cal=preds_cal,
    )

    raw = out.loc[~out["is_holdout"], "q_0.50"]
    calibrated = out.loc[~out["is_holdout"], "cal_q_0.50"]
    np.testing.assert_allclose(raw, 10.5)
    np.testing.assert_allclose(calibrated, 11.5)
    np.testing.assert_allclose(out.loc[out["is_holdout"], "q_0.50"], 15.0)
    np.testing.assert_allclose(out.loc[out["is_holdout"], "cal_q_0.50"], 16.0)
    assert len(out) == 10


def test_oos_predictions_accept_four_contiguous_folds_and_reject_gaps():
    from minutes_xgb.artifacts import build_oos_predictions
    from minutes_xgb.config import QUANTILES

    def frame(fold_ids):
        rows = len(fold_ids)
        identity = pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(rows)],
                "player_id": np.arange(rows),
                "game_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "minutes": np.full(rows, 20.0),
                "pts": np.full(rows, 8.0),
                "starting": np.zeros(rows, dtype=int),
            }
        )
        oof = identity.copy()
        oof["fold_id"] = fold_ids
        oof["early_stop"] = "train_tail"
        oof["train_tail_frac"] = 0.10
        for level in QUANTILES:
            oof[f"q_{level:.2f}"] = 10.0 + level
        holdout = identity.iloc[:1].copy()
        preds = {f"q_{level:.2f}": np.full(1, 15.0) for level in QUANTILES}
        return {"oof": oof}, identity, holdout, preds

    out = build_oos_predictions(*frame([1, 2, 3, 4]))
    assert sorted(out.loc[~out["is_holdout"], "window_id"].unique()) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="walk-forward folds"):
        build_oos_predictions(*frame([1, 2, 3]))
    with pytest.raises(ValueError, match="walk-forward folds"):
        build_oos_predictions(*frame([1, 2, 3, 5]))
