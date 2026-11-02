"""Rate conformal offsets stay off the 2025-26 holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from models.shared.conformal_report import ConformalSpec
from points_xgb.calibration import (
    calibrated_tail_frame,
    fold_base_knots,
    oof_leakage_table,
    oof_player_games,
    require_preholdout,
)

POINTS_SPEC = ConformalSpec(
    target_col="pts",
    lag_col="pts_lag_1",
    tiers=(
        ("<8", None, 8.0),
        ("8-15", 8.0, 15.0),
        ("15-25", 15.0, 25.0),
        ("25+", 25.0, None),
    ),
    clip_low=0.0,
    clip_high=100.0,
)


def test_require_preholdout_rejects_opening_night():
    oof = pd.DataFrame({"game_date": pd.to_datetime(["2025-10-21"])})
    require_preholdout(oof)
    oof = pd.DataFrame({"game_date": pd.to_datetime(["2026-10-20"])})
    with pytest.raises(ValueError, match="holdout"):
        require_preholdout(oof)


def test_fold_base_knots_fill_a_cold_start_from_that_folds_rates():
    ppm = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "starting": [1, 1, 1],
            "pts_per_min": [0.2, 0.6, 0.1],
            "pts_per_min_lag_1": [0.1, 0.2, np.nan],
        }
    )
    oof = ppm.iloc[[2]].copy()
    oof["fold_id"] = 1
    oof["q_0.10"] = 1.0
    oof["q_0.50"] = 1.0
    oof["q_0.90"] = 1.0
    mask = pd.Series([True, True, False], index=ppm.index)

    base, group = fold_base_knots(oof, ppm, [mask])

    assert list(group) == ["cold_start"]
    assert base.loc[oof.index[0], "q_0.50"] == pytest.approx(0.4)

    oof["game_date"] = pd.Timestamp("2025-10-21")
    fold_base_knots(oof, ppm, [mask])
    oof["game_date"] = pd.Timestamp("2026-10-20")
    with pytest.raises(ValueError, match="holdout"):
        fold_base_knots(oof, ppm, [mask])


def test_fold_base_knots_reads_points_when_the_rate_column_is_absent():
    ppm = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "starting": [1, 1, 1],
            "pts": [10.0, 30.0, 4.0],
            "pts_lag_1": [8.0, 10.0, np.nan],
        }
    )
    oof = ppm.iloc[[2]].copy()
    oof["fold_id"] = 1
    oof["q_0.10"] = 99.0
    oof["q_0.50"] = 99.0
    oof["q_0.90"] = 99.0
    mask = pd.Series([True, True, False], index=ppm.index)

    base, group = fold_base_knots(oof, ppm, [mask], spec=POINTS_SPEC)

    assert list(group) == ["cold_start"]
    assert base.loc[oof.index[0], "q_0.50"] == pytest.approx(20.0)


def test_duplicate_oof_player_games_raise():
    ppm = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 10],
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        }
    )
    oof = ppm.iloc[[0, 1]].copy()
    with pytest.raises(RuntimeError, match="duplicated"):
        oof_player_games(oof, ppm)


def test_leakage_table_rejects_a_validation_date_in_the_training_tail():
    ppm = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "player_id": [10, 10, 10, 10],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
        }
    )
    oof = ppm.iloc[[3]].copy()
    oof["fold_id"] = 1
    oof["train_tail_frac"] = 0.10
    train_mask = pd.Series([True, True, True, False], index=ppm.index)
    folds = [
        {
            "fold": 1,
            "train_dates": list(ppm["game_date"].iloc[:3]),
            "val_dates": [ppm["game_date"].iloc[2]],
            "train_mask": train_mask,
        }
    ]
    with pytest.raises(RuntimeError, match="leaks"):
        oof_leakage_table(oof, ppm, folds)

    folds[0]["val_dates"] = [ppm["game_date"].iloc[3]]
    table = oof_leakage_table(oof, ppm, folds)
    assert table.loc[0, "val_in_train"] == 0
    assert table.loc[0, "val_in_stop"] == 0
    assert bool(table.loc[0, "dates_match"])


def test_calibrated_tail_frame_drops_fold_five_and_names_the_rate():
    oof = pd.DataFrame(
        {
            "fold_id": [1, 5],
            "game_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "starting": [1, 0],
            "pts_per_min": [0.4, 0.8],
        }
    )
    knots = pd.DataFrame({"q_0.50": [0.3, 0.7]}, index=oof.index)
    out = calibrated_tail_frame(oof, knots)
    assert out["fold_id"].tolist() == [1]
    assert out["rate"].iloc[0] == pytest.approx(0.4)
    assert "pts_per_min" not in out.columns
    assert out["q_0.50"].iloc[0] == pytest.approx(0.3)
