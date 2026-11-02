"""The audit table ranks features. It does not drop them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from points_xgb.importance import feature_consensus, shap_center_and_width


def test_consensus_does_not_drop_a_replaceable_high_importance_feature():
    features = ["pts_ewm_hl_10", "starting"]
    shap_mean = pd.Series({"pts_ewm_hl_10": 3.0, "starting": 1.0})
    permutation = pd.DataFrame(
        {
            "feature": features,
            "importance": [0.75, 0.20],
            "std": [0.01, 0.01],
        }
    )
    importances = np.array(
        [
            [0.75, 0.74, 0.76, 0.75, 0.75],
            [0.20, 0.19, 0.21, 0.20, 0.20],
        ]
    )
    ablation = pd.DataFrame(
        {
            "feature": features,
            "delta_pinball": [-0.0004, 0.0301],
        }
    )
    audit = feature_consensus(
        features,
        shap_mean,
        permutation,
        importances,
        ablation,
    )
    assert "keep" not in audit.columns
    assert set(audit["feature"]) == set(features)


def test_width_importance_is_the_tail_shap_gap():
    table = shap_center_and_width(
        {
            "q_0.10": np.array([[0.0, 1.0], [0.0, 1.0]]),
            "q_0.50": np.array([[5.0, 0.0], [5.0, 0.0]]),
            "q_0.90": np.array([[2.0, 1.0], [2.0, 1.0]]),
        },
        ["min_sd_10", "pts_ewm_hl_10"],
    )
    assert table.loc["min_sd_10", "center"] == 5.0
    assert table.loc["min_sd_10", "width"] == 2.0
    assert table.loc["pts_ewm_hl_10", "width"] == 0.0
    assert table.index[0] == "min_sd_10"
