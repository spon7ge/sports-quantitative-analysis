"""Points notebook helpers score raw points, not points per minute."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from points_xgb.artifacts import exploratory_metadata
from points_xgb.baseline import compare_to_baselines
from points_xgb.data import data_quality, format_data_quality
from points_xgb.eda import format_rate_outliers, rate_outlier_summary
from points_xgb.errors import build_miss_frame, miss_view


def _quantiles(values: list[float]) -> dict[str, np.ndarray]:
    prediction = np.asarray(values, dtype=float)
    return {
        f"q_{level:.2f}": prediction
        for level in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    }


def test_baselines_score_points_when_the_rate_columns_are_absent():
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-10-21", "2026-10-21"]),
            "starting": [1, 0],
            "pts": [20.0, 8.0],
            "pts_lag_1": [18.0, 6.0],
            "season_pts_mean": [17.0, 7.0],
            "pts_median_10": [16.0, 5.0],
        }
    )
    report = compare_to_baselines(
        frame["pts"],
        _quantiles([19.0, 7.0]),
        frame,
        frame,
        [{"fold": 1, "val_mask": pd.Series([True, True]), "pinball": 1.0}],
        target_col="pts",
        tiers={
            "<8": lambda values: np.asarray(values) < 8,
            "8+": lambda values: np.asarray(values) >= 8,
        },
        baseline_cols={
            "Last game": "pts_lag_1",
            "Season-to-date": "season_pts_mean",
            "Last-10 median": "pts_median_10",
        },
        error_cutoffs=(5, 10),
        error_unit="points",
        n_boot=20,
        seed=1,
    )
    point = report["point_table"].set_index("model")
    assert point.loc["Last game", "mae"] == pytest.approx(2.0)
    assert report["fold_table"].loc[0, "last_10_median"] == pytest.approx(3.5)
    assert list(report["tail_table"]["threshold"]) == [">5 points", ">10 points"]
    assert "pts/min" not in " ".join(report["tail_table"]["threshold"])


def test_miss_frame_measures_points():
    source = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [2],
            "team_abbreviation": ["BOS"],
            "matchup": ["BOS vs. NYK"],
            "wl": ["W"],
            "pf": [2],
            "pfd": [1],
            "plus_minus": [5],
            "team_pts": [110],
            "opp_pts": [100],
            "abs_spread": [3.5],
            "minutes": [30.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [2],
            "game_date": pd.to_datetime(["2026-10-21"]),
            "player_name": ["A"],
            "starting": [1],
            "pts": [22.0],
            "minutes": [30.0],
            "pts_lag_1": [4.0],
            "season_pts_mean": [12.0],
        }
    )
    preds = {
        "q_0.10": np.array([8.0]),
        "q_0.50": np.array([14.0]),
        "q_0.90": np.array([20.0]),
    }
    miss = build_miss_frame(
        source,
        holdout,
        preds,
        target_col="pts",
        lag_col="pts_lag_1",
        unit="points",
    )
    view = miss_view(
        miss,
        target_col="pts",
        lag_col="pts_lag_1",
        season_col="season_pts_mean",
        unit="points",
    )
    assert miss["error"].iloc[0] == pytest.approx(8.0)
    assert miss["hint"].iloc[0] == "unexpected start"
    assert "actual" in view.columns
    assert "actual_rate" not in view.columns
    assert view.loc[0, "last_game"] == pytest.approx(4.0)


def test_data_quality_can_omit_the_rate_lines():
    frame = pd.DataFrame(
        {
            "player_id": [1, 1],
            "game_id": [1, 2],
            "minutes": [20.0, 10.0],
            "pts": [10.0, 30.0],
        }
    )
    text = format_data_quality(data_quality(frame), include_rate=False)
    assert "per minute" not in text
    assert "Max points: 30.000" in text


def test_outlier_summary_can_describe_points_without_a_rate_cap():
    frame = pd.DataFrame({"pts": [0.0, 10.0, 20.0, 40.0]})
    stats = rate_outlier_summary(frame, "pts", unit="points")
    text = format_rate_outliers(stats)
    assert "points-per-minute" not in text
    assert "cap" not in text
    assert stats["max"] == pytest.approx(40.0)


def test_saved_metadata_records_the_points_label():
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-01", "2026-10-21"]),
        }
    )
    metadata = exploratory_metadata(
        frame.iloc[:1],
        frame.iloc[1:],
        {"n": 1, "pinball": 1.5},
        [{"pinball": 1.5}],
        features=["pts_ewm_hl_10"],
        label="pts",
        label_rule="pts, minutes > 0",
    )
    assert metadata["label"] == "pts"
    assert metadata["label_rule"] == "pts, minutes > 0"
    assert metadata["features"] == ["pts_ewm_hl_10"]
