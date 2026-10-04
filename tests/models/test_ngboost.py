import numpy as np
import pandas as pd
import pytest
from models.shared.ngboost import (
    evaluate_holdout,
    fit_ngboost,
    run_walk_forward,
)


def _minutes_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = pd.Series(18.0 + 3.0 * x + rng.normal(scale=0.4, size=n), name="minutes")
    X = pd.DataFrame({"a": x})
    return X, y


def test_fit_returns_ordered_quantiles_around_the_minutes_level():
    X, y = _minutes_frame()
    predict = pd.DataFrame({"a": [0.0, 0.5, -0.5]})
    models, preds = fit_ngboost(
        X,
        y,
        X.iloc[:12],
        y.iloc[:12],
        quantiles=[0.05, 0.10, 0.50, 0.90, 0.95],
        ngb_params={
            "n_estimators": 30,
            "learning_rate": 0.05,
            "verbose": False,
            "random_state": 0,
            "early_stopping_rounds": 5,
        },
        X_predict=predict,
    )
    assert list(models) == ["q_0.05", "q_0.10", "q_0.50", "q_0.90", "q_0.95"]
    assert len(preds["q_0.50"]) == 3
    np.testing.assert_allclose(models["q_0.50"].predict(predict), preds["q_0.50"])
    assert np.all(preds["q_0.05"] < preds["q_0.50"])
    assert np.all(preds["q_0.50"] < preds["q_0.95"])
    assert abs(float(np.mean(preds["q_0.50"])) - 18.0) < 3.0


def test_train_tail_stops_on_the_last_training_dates(monkeypatch):
    import models.shared.ngboost as ngb_mod

    seen = {}
    real = ngb_mod.NGBRegressor

    class _Recording(real):
        def fit(self, X, Y, X_val=None, Y_val=None, **kwargs):
            seen["y_val"] = np.asarray(Y_val, dtype=float).copy()
            return super().fit(X, Y, X_val=X_val, Y_val=Y_val, **kwargs)

    monkeypatch.setattr(ngb_mod, "NGBRegressor", _Recording)
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([10.0, 11.0, 12.0, 13.0])
    predict = pd.DataFrame({"a": [9.0, 8.0]})
    _, preds = fit_ngboost(
        X,
        y,
        predict,
        pd.Series([30.0, 31.0]),
        quantiles=[0.50],
        ngb_params={"n_estimators": 3, "verbose": False, "random_state": 0},
        early_stop="train_tail",
        train_dates=dates,
        train_tail_frac=0.5,
        X_predict=predict,
    )
    np.testing.assert_array_equal(seen["y_val"], [12.0, 13.0])
    assert len(preds["q_0.50"]) == 2


def test_missing_params_and_unknown_early_stop_raise_before_fit(monkeypatch):
    import models.shared.ngboost as ngb_mod

    def boom(*args, **kwargs):
        raise AssertionError("booster constructed")

    monkeypatch.setattr(ngb_mod, "NGBRegressor", boom)
    X, y = _minutes_frame(4)
    with pytest.raises(ValueError, match="ngb_params"):
        fit_ngboost(X, y, X, y, quantiles=[0.50])
    with pytest.raises(ValueError, match="unknown early_stop"):
        fit_ngboost(
            X,
            y,
            X,
            y,
            quantiles=[0.50],
            ngb_params={"n_estimators": 1},
            early_stop="holdout",
        )
    with pytest.raises(ValueError, match="two training dates"):
        fit_ngboost(
            X.iloc[:1],
            y.iloc[:1],
            X.iloc[:1],
            y.iloc[:1],
            quantiles=[0.50],
            ngb_params={"n_estimators": 1},
            early_stop="train_tail",
            train_dates=pd.to_datetime(["2024-01-01"]),
        )


def _dated_frame(n_dates=10):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for day in dates:
        for player, minutes in ((1, 12.0), (2, 28.0)):
            rows.append(
                {
                    "game_date": day,
                    "a": float(player),
                    "minutes": minutes,
                    "starting": player % 2,
                }
            )
    frame = pd.DataFrame(rows)
    return frame, frame[["a"]], frame["minutes"]


def test_walk_forward_stacks_validation_quantiles_only():
    frame, X, y = _dated_frame(10)
    result = run_walk_forward(
        X,
        y,
        frame,
        ngb_params={
            "n_estimators": 8,
            "verbose": False,
            "random_state": 0,
            "early_stopping_rounds": 2,
        },
        quantiles=[0.10, 0.50, 0.90],
        n_folds=1,
        train_frac=0.5,
        step_frac=0.2,
        early_stop="train_tail",
    )
    oof = result["oof"]
    assert set(oof["fold_id"]) == {1}
    assert (oof["early_stop"] == "train_tail").all()
    assert "q_0.50" in oof.columns
    assert set(pd.to_datetime(oof["game_date"])) == set(
        pd.to_datetime(frame.loc[result["last_fold"]["val_mask"], "game_date"])
    )
    assert oof["game_date"].min() > result["fold_ranges"][1]["train_end"]


def test_holdout_predicts_the_held_out_rows(monkeypatch):
    import models.shared.ngboost as ngb_mod

    seen = {}
    real = ngb_mod.NGBRegressor

    class _Recording(real):
        def fit(self, X, Y, X_val=None, Y_val=None, **kwargs):
            seen["y_val"] = np.asarray(Y_val, dtype=float).copy()
            return super().fit(X, Y, X_val=X_val, Y_val=Y_val, **kwargs)

    monkeypatch.setattr(ngb_mod, "NGBRegressor", _Recording)
    train, X_train, y_train = _dated_frame(8)
    holdout = train.iloc[:3].copy()
    holdout["minutes"] = [40.0, 41.0, 42.0]
    holdout["game_date"] = pd.to_datetime(["2025-10-21", "2025-10-22", "2025-10-23"])
    result = evaluate_holdout(
        train,
        holdout,
        features=["a"],
        target_col="minutes",
        ngb_params={"n_estimators": 4, "verbose": False, "random_state": 0},
        quantiles=[0.10, 0.50, 0.90],
        es_frac=0.75,
    )
    assert len(result["preds_ho"]["q_0.50"]) == 3
    assert 40.0 not in set(seen["y_val"])
    assert set(np.round(seen["y_val"], 5)).issubset({12.0, 28.0})
