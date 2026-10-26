"""Walk-forward stores the outcome under the requested column name."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.shared.train import run_walk_forward

_PARAMS = {
    "objective": "reg:quantileerror",
    "n_estimators": 2,
    "max_depth": 1,
    "learning_rate": 0.1,
    "early_stopping_rounds": 1,
    "n_jobs": 1,
    "random_state": 0,
}


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "game_date": dates,
            "starting": np.zeros(len(dates), dtype=int),
            "x": np.arange(len(dates), dtype=float),
            "pts_per_min": np.linspace(0.1, 1.0, len(dates)),
        }
    )


def test_walk_forward_writes_the_requested_outcome_column():
    frame = _frame()
    features = frame[["x"]]
    outcome = frame["pts_per_min"]
    named = run_walk_forward(
        features,
        outcome,
        frame,
        xgb_params=_PARAMS,
        quantiles=[0.5],
        n_folds=1,
        target_col="pts_per_min",
    )
    assert "pts_per_min" in named["oof"].columns
    assert "minutes" not in named["oof"].columns

    default = run_walk_forward(
        features,
        outcome,
        frame,
        xgb_params=_PARAMS,
        quantiles=[0.5],
        n_folds=1,
    )
    assert "minutes" in default["oof"].columns
    assert "pts_per_min" not in default["oof"].columns
