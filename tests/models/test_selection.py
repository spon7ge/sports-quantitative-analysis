"""Group ablation statistics. Fits stay outside these tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.shared.quantile_offsets import row_mean_pinball
from models.shared.selection import (
    date_mean_se,
    decide_groups,
    feature_groups,
    group_ablation,
)


def test_identical_features_cluster_and_an_independent_one_stays_alone():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=80)
    frame = pd.DataFrame(
        {
            "volume": shared,
            "volume_copy": shared,
            "other": rng.normal(size=80),
        }
    )
    groups = feature_groups(frame, min_abs_rho=0.8)
    members = {tuple(group) for group in groups}
    assert ("volume", "volume_copy") in members
    assert ("other",) in members


def test_date_mean_weights_a_date_once():
    # Two rows on the first date average to 1. The second date is 4.
    delta, se = date_mean_se(
        [0.0, 2.0, 4.0],
        ["2024-01-01", "2024-01-01", "2024-01-02"],
    )
    assert delta == pytest.approx(2.5)
    assert se == pytest.approx(np.std([1.0, 4.0], ddof=1) / np.sqrt(2))


def test_row_mean_pinball_matches_the_hand_value():
    # Residuals 1 and -2. Pinball 0.25 and 0.5. The row mean is 0.375.
    value = row_mean_pinball(
        np.array([1.0]),
        np.array([[0.0, 3.0]]),
        np.array([0.25, 0.75]),
    )
    assert value.tolist() == [0.375]


def test_group_ablation_charges_a_group_the_model_cannot_replace():
    frame = pd.DataFrame(
        {
            "pts": [0.0, 1.0, 2.0, 3.0],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "useful": [1.0, 1.0, 1.0, 1.0],
            "noise": [0.0, 0.0, 0.0, 0.0],
        }
    )
    fold = {
        "label": "2023-24",
        "train_mask": np.array([True, True, False, False]),
        "val_mask": np.array([False, False, True, True]),
    }
    taus = np.array([0.10, 0.50, 0.90])
    seen = []

    def fit_predict(train, val, cols, seed):
        seen.append(seed)
        pred = np.repeat(val["pts"].to_numpy()[:, None], len(taus), axis=1)
        if "useful" not in cols:
            pred = pred + 1.0
        return pred

    records = group_ablation(
        frame,
        ["useful", "noise"],
        [["useful"], ["noise"]],
        [fold],
        fit_predict,
        taus,
        seeds=(0, 1),
    )
    useful = records.loc[records["group"].eq("useful")].iloc[0]
    noise = records.loc[records["group"].eq("noise")].iloc[0]
    assert useful["delta"] == pytest.approx(0.5)
    assert useful["coverage_gap_delta"] == pytest.approx(0.6)
    assert noise["delta"] == pytest.approx(0.0)
    assert seen == [0, 1, 0, 1, 0, 1]


def test_keep_requires_two_se_and_three_positive_folds():
    records = pd.DataFrame(
        {
            "group": ["volume"] * 4 + ["flat"] * 4,
            "fold": [1, 2, 3, 4, 1, 2, 3, 4],
            "delta": [0.02, 0.02, 0.02, 0.02, 0.001, -0.001, 0.0, 0.0005],
            "coverage_gap_delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    decision = decide_groups(records).set_index("group")
    assert decision.loc["volume", "decision"] == "keep"
    assert decision.loc["flat", "decision"] == "inconclusive"


def test_drop_requires_nonpositive_folds_and_no_worse_coverage():
    records = pd.DataFrame(
        {
            "group": ["dead"] * 4 + ["tail"] * 4,
            "fold": [1, 2, 3, 4, 1, 2, 3, 4],
            "delta": [-0.01, -0.01, 0.0, -0.01, -0.01, -0.01, 0.0, -0.01],
            "coverage_gap_delta": [-0.01, 0.0, -0.01, 0.0, 0.02, 0.02, 0.02, 0.02],
        }
    )
    decision = decide_groups(records).set_index("group")
    assert decision.loc["dead", "decision"] == "drop"
    assert decision.loc["tail", "decision"] == "inconclusive"
