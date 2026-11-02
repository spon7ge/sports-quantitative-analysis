"""Early stopping picks a round count, then the model refits on the full window."""

from __future__ import annotations

import numpy as np
import pandas as pd
from models.shared import train as train_mod
from models.shared.train import fit_quantile_models


def test_stopping_tail_is_the_latest_regular_season_dates_then_refit(monkeypatch):
    constructed = []

    class _Fake:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.best_iteration = 3
            constructed.append(self)

        def fit(self, X, y, eval_set=None, verbose=False):
            self.fit_index = np.asarray(X.index).copy()
            self.eval_index = (
                None
                if eval_set is None
                else np.asarray(eval_set[0][0].index).copy()
            )

        def predict(self, X):
            self.predict_index = np.asarray(X.index).copy()
            return np.zeros(len(X))

    monkeypatch.setattr(train_mod, "XGBRegressor", _Fake)
    # Appearance order is not calendar order. The last two calendar dates
    # are a regular-season date and a playoff date. The tail must be the
    # latest regular-season date, which is Jan 3, not the playoff.
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2024-01-04", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "a": [1.0, 1.0, 1.0, 1.0],
            "pts": [10.0, 11.0, 12.0, 13.0],
            "season_type": ["Playoffs", "Regular Season", "Regular Season", "Regular Season"],
        }
    )
    predict = pd.DataFrame({"a": [9.0]}, index=[99])
    regular = frame["season_type"].eq("Regular Season").to_numpy()

    _models, preds = fit_quantile_models(
        frame[["a"]],
        frame["pts"],
        frame[["a"]],
        frame["pts"],
        xgb_params={"n_estimators": 20, "early_stopping_rounds": 5, "random_state": 0},
        quantiles=[0.50],
        early_stop="train_tail",
        train_dates=frame["game_date"],
        train_tail_frac=0.5,
        regular_mask=regular,
        X_predict=predict,
    )

    stopped, refit = constructed
    assert "early_stopping_rounds" not in refit.kwargs
    # Three regular dates, tail fraction 0.5 → one stopping date (Jan 3).
    # best_iteration 3, full window 4 rows, early-stopping fit 2 rows (Jan 1–2).
    assert refit.kwargs["n_estimators"] == int((3 + 1) * 4 / 2)
    assert set(pd.to_datetime(frame.loc[stopped.eval_index, "game_date"])) == {
        pd.Timestamp("2024-01-03")
    }
    assert set(pd.to_datetime(frame.loc[stopped.fit_index, "game_date"])) == {
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    }
    assert set(refit.fit_index) == set(frame.index)
    assert list(refit.predict_index) == [99]
    assert len(preds["q_0.50"]) == 1
