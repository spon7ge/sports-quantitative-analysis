"""Pinball, crossing, and calibrated OOS columns for the rate notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from points_xgb.artifacts import build_oos_predictions
from points_xgb.config import QUANTILES
from points_xgb.scoring import quantile_crossing, score_predictions


def test_one_miss_at_the_median_has_pinball_one_half():
    pinball, _coverage = score_predictions(
        [1.0],
        {"q_0.50": np.array([0.0])},
        "holdout",
    )
    row = pinball.loc[pinball["quantile"].eq(0.50)].iloc[0]
    assert row["pinball"] == 0.5


def test_a_decreasing_pair_is_one_crossed_row():
    n_crossed, n_rows, pairs = quantile_crossing(
        {"q_0.10": np.array([1.0]), "q_0.50": np.array([0.0])}
    )
    assert (n_crossed, n_rows) == (1, 1)
    assert pairs.iloc[0]["crossing_rate"] == 1.0


def test_oos_frame_keeps_raw_knots_and_adds_calibrated_knots():
    identity = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": ["p1", "p2"],
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "minutes": [20.0, 30.0],
            "pts": [10.0, 12.0],
            "starting": [1, 0],
        }
    )
    levels = [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
    ]
    oof = pd.DataFrame({f"q_{level:.2f}": [0.4, 0.5] for level in levels})
    oof["fold_id"] = [1, 2]
    oof["starting"] = [1, 0]
    oof["early_stop"] = ["train_tail", "train_tail"]
    oof["train_tail_frac"] = [0.1, 0.1]
    oof.index = identity.index
    extra_dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
    more = pd.DataFrame(
        {
            "game_id": ["g3", "g4", "g5"],
            "player_id": ["p3", "p4", "p5"],
            "game_date": extra_dates,
            "minutes": [10.0, 11.0, 12.0],
            "pts": [1.0, 2.0, 3.0],
            "starting": [1, 0, 1],
        }
    )
    identity = pd.concat([identity, more], ignore_index=True)
    oof = pd.DataFrame({f"q_{level:.2f}": np.full(5, 0.4) for level in levels})
    oof["fold_id"] = [1, 2, 3, 4, 5]
    oof["starting"] = identity["starting"].to_numpy()
    oof["early_stop"] = "train_tail"
    oof["train_tail_frac"] = 0.1
    holdout = identity.iloc[:1].copy()
    preds = {f"q_{level:.2f}": np.array([0.2]) for level in levels}
    calibrated = oof[[column for column in oof.columns if column.startswith("q_")]].copy()
    calibrated["q_0.50"] = np.arange(5, dtype=float)
    ho_cal = {f"q_{level:.2f}": np.array([0.9]) for level in levels}
    frame = build_oos_predictions(
        {"oof": oof},
        identity,
        holdout,
        preds,
        oof_cal=calibrated,
        preds_ho_cal=ho_cal,
    )
    assert frame["q_0.50"].iloc[0] == 0.4
    assert frame["cal_q_0.50"].iloc[0] == 0.0
    assert frame["cal_q_0.50"].iloc[-1] == 0.9
    assert frame["minutes"].iloc[0] == 20.0


def test_oos_predictions_accept_four_contiguous_folds_and_reject_gaps():
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
            oof[f"q_{level:.2f}"] = 0.4
        holdout = identity.iloc[:1].copy()
        preds = {f"q_{level:.2f}": np.full(1, 0.5) for level in QUANTILES}
        return {"oof": oof}, identity, holdout, preds

    out = build_oos_predictions(*frame([1, 2, 3, 4]))
    assert sorted(out.loc[~out["is_holdout"], "window_id"].unique()) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="walk-forward folds"):
        build_oos_predictions(*frame([1, 2, 3]))
    with pytest.raises(ValueError, match="walk-forward folds"):
        build_oos_predictions(*frame([1, 2, 3, 5]))
