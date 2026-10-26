"""Holdout early stopping uses the last training dates, not the last rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.shared import train as train_mod
from models.shared.train import evaluate_holdout


def test_holdout_stops_on_the_last_training_dates_and_not_the_holdout(monkeypatch):
    seen = {}

    class _FakeBooster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, eval_set=None, verbose=False):
            seen["eval_index"] = eval_set[0][0].index.to_numpy().copy()
            seen["eval_y"] = np.asarray(eval_set[0][1], dtype=float).copy()

        def predict(self, X):
            return np.full(len(X), 15.0)

    monkeypatch.setattr(train_mod, "XGBRegressor", _FakeBooster)
    rows = []
    for day in pd.date_range("2024-01-01", periods=9, freq="D"):
        for _ in range(20):
            rows.append({"game_date": day, "a": 1.0, "minutes": 10.0, "starting": 1})
    last = pd.Timestamp("2024-01-10")
    for _ in range(2):
        rows.append({"game_date": last, "a": 1.0, "minutes": 10.0, "starting": 0})
    train = pd.DataFrame(rows)
    holdout = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2025-10-21", "2025-10-22", "2025-10-23"]),
            "a": [1.0, 1.0, 1.0],
            "minutes": [40.0, 41.0, 42.0],
            "starting": [1, 0, 1],
        }
    )

    result = evaluate_holdout(
        train,
        holdout,
        features=["a"],
        target_col="minutes",
        xgb_params={"n_estimators": 1, "random_state": 0},
        quantiles=[0.10, 0.50, 0.90],
        train_tail_frac=0.10,
    )

    eval_dates = set(pd.to_datetime(train.loc[seen["eval_index"], "game_date"]))
    assert eval_dates == {last}
    assert len(seen["eval_index"]) == 2
    assert 40.0 not in set(seen["eval_y"])
    assert len(result["preds_ho"]["q_0.50"]) == 3
