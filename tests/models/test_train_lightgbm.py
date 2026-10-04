import numpy as np
import pandas as pd
import pytest

from models.shared import train as train_mod
from models.shared.train import (
    evaluate_holdout,
    fit_quantile_lightgbm,
    run_timeseries_cv,
    run_walk_forward,
    tune_lgb_quantile,
)


class _FakeLGBM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callbacks = None
        self.eval_rows = None

    def fit(self, X, y, eval_set=None, callbacks=None):
        self.eval_rows = len(eval_set[0][0])
        self.callbacks = callbacks

    def predict(self, X):
        return np.arange(len(X), dtype=float)


def _patch_booster(monkeypatch):
    monkeypatch.setattr(train_mod, "LGBMRegressor", _FakeLGBM)
    monkeypatch.setattr(
        train_mod,
        "early_stopping",
        lambda rounds, verbose=False: ("early_stopping", rounds, verbose),
    )


def _frame(n=4):
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    y = pd.Series(np.arange(n, dtype=float) + 1.0)
    return X, y


def test_fit_sets_alpha_per_quantile_and_predicts_requested_frame(monkeypatch):
    _patch_booster(monkeypatch)
    X, y = _frame(4)
    predict = pd.DataFrame({"a": [9.0, 8.0, 7.0]})
    models, preds = fit_quantile_lightgbm(
        X, y, X, y,
        quantiles=[0.05, 0.10, 0.50, 0.90, 0.95],
        lgb_params={"n_estimators": 3, "early_stopping_rounds": 7},
        X_predict=predict,
    )
    assert list(models) == ["q_0.05", "q_0.10", "q_0.50", "q_0.90", "q_0.95"]
    assert [models[key].kwargs["alpha"] for key in models] == [
        0.05, 0.10, 0.50, 0.90, 0.95,
    ]
    assert all("early_stopping_rounds" not in models[key].kwargs for key in models)
    assert models["q_0.50"].callbacks == [("early_stopping", 7, False)]
    assert len(preds["q_0.50"]) == 3


def test_fit_without_early_stopping_passes_no_callback(monkeypatch):
    _patch_booster(monkeypatch)
    X, y = _frame()
    models, _ = fit_quantile_lightgbm(
        X, y, X, y,
        quantiles=[0.50],
        lgb_params={"n_estimators": 3},
    )
    assert models["q_0.50"].callbacks is None


def test_alpha_in_shared_params_raises_before_booster(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("booster constructed")

    monkeypatch.setattr(train_mod, "LGBMRegressor", boom)
    X, y = _frame()
    with pytest.raises(ValueError, match="alpha"):
        fit_quantile_lightgbm(
            X, y, X, y,
            quantiles=[0.50],
            lgb_params={"alpha": 0.2, "n_estimators": 1},
        )


def test_missing_lgb_params_raises():
    X, y = _frame()
    with pytest.raises(ValueError, match="lgb_params"):
        fit_quantile_lightgbm(X, y, X, y, quantiles=[0.50])


def test_unknown_early_stop_raises():
    X, y = _frame()
    with pytest.raises(ValueError, match="unknown early_stop"):
        fit_quantile_lightgbm(
            X, y, X, y,
            quantiles=[0.50],
            lgb_params={"n_estimators": 1},
            early_stop="holdout",
        )


def test_train_tail_one_date_raises_before_any_booster(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("booster constructed")

    monkeypatch.setattr(train_mod, "LGBMRegressor", boom)
    X, y = _frame(1)
    with pytest.raises(ValueError, match="two training dates"):
        fit_quantile_lightgbm(
            X, y, X, y,
            quantiles=[0.50],
            lgb_params={"n_estimators": 1},
            early_stop="train_tail",
            train_dates=pd.to_datetime(["2024-01-01"]),
        )


def test_tune_locks_fixed_settings_and_q50_alpha(monkeypatch):
    constructed: list[float] = []

    class _RecordingLGBM(_FakeLGBM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            constructed.append(kwargs["alpha"])

        def predict(self, X):
            return np.full(len(X), 10.0)

    monkeypatch.setattr(train_mod, "LGBMRegressor", _RecordingLGBM)
    monkeypatch.setattr(
        train_mod,
        "early_stopping",
        lambda rounds, verbose=False: ("early_stopping", rounds, verbose),
    )
    X, y = _frame(8)
    result = tune_lgb_quantile(
        X, y, n_trials=1, n_splits=2, seed=42, show_progress_bar=False,
    )
    params = result["best_params"]
    assert "alpha" not in params
    assert params["objective"] == "quantile"
    assert params["n_jobs"] == -1
    assert params["random_state"] == 42
    assert params["verbose"] == -1
    assert params["bagging_freq"] == 1
    assert params["early_stopping_rounds"] == 50
    assert set(constructed) == {0.50}
    assert isinstance(result["best_value"], float)


def test_tune_fixed_params_override_defaults(monkeypatch):
    monkeypatch.setattr(train_mod, "LGBMRegressor", _FakeLGBM)
    monkeypatch.setattr(
        train_mod,
        "early_stopping",
        lambda rounds, verbose=False: ("early_stopping", rounds, verbose),
    )
    X, y = _frame(8)
    result = tune_lgb_quantile(
        X, y,
        n_trials=1,
        n_splits=2,
        show_progress_bar=False,
        fixed_params={"verbose": 0},
    )
    assert result["best_params"]["verbose"] == 0


def _dated_frame(n_dates=12):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for day in dates:
        for player in (1, 2):
            rows.append({
                "game_date": day,
                "a": float(player),
                "minutes": 10.0 + player,
                "starting": player % 2,
            })
    frame = pd.DataFrame(rows)
    return frame, frame[["a"]], frame["minutes"]


def test_timeseries_cv_fits_lightgbm_on_each_split(monkeypatch):
    _patch_booster(monkeypatch)
    frame, X, y = _dated_frame(9)
    results = run_timeseries_cv(
        X, y, frame,
        lgb_params={"n_estimators": 3, "early_stopping_rounds": 2},
        n_splits=2,
        quantiles=[0.10, 0.50, 0.90],
    )
    assert len(results) == 2
    assert all(np.isfinite(row["pinball"]) for row in results)
    assert all(np.isfinite(row["coverage_80pct"]) for row in results)


def test_walk_forward_fits_lightgbm_on_the_train_tail(monkeypatch):
    _patch_booster(monkeypatch)
    frame, X, y = _dated_frame(10)
    result = run_walk_forward(
        X, y, frame,
        lgb_params={"n_estimators": 3},
        quantiles=[0.50],
        n_folds=1,
        train_frac=0.5,
        step_frac=0.2,
        early_stop="train_tail",
    )
    oof = result["oof"]
    assert set(oof["fold_id"]) == {1}
    assert (oof["early_stop"] == "train_tail").all()
    assert "q_0.50" in oof.columns
    assert result["models_last"]["q_0.50"].eval_rows < len(X)


def test_holdout_fits_lightgbm_on_the_train_tail_and_predicts_holdout(monkeypatch):
    seen = {}

    class _Recording(_FakeLGBM):
        def fit(self, X, y, eval_set=None, callbacks=None):
            seen["eval_y"] = np.asarray(eval_set[0][1], dtype=float).copy()
            return super().fit(X, y, eval_set=eval_set, callbacks=callbacks)

    monkeypatch.setattr(train_mod, "LGBMRegressor", _Recording)
    monkeypatch.setattr(
        train_mod,
        "early_stopping",
        lambda rounds, verbose=False: ("early_stopping", rounds, verbose),
    )
    train, _, _ = _dated_frame(10)
    holdout = train.iloc[:4].copy()
    holdout["minutes"] = [40.0, 41.0, 42.0, 43.0]
    result = evaluate_holdout(
        train,
        holdout,
        features=["a"],
        target_col="minutes",
        lgb_params={"n_estimators": 3, "early_stopping_rounds": 2},
        quantiles=[0.10, 0.50, 0.90],
        es_frac=0.75,
        wf_results=None,
    )
    assert len(result["preds_ho"]["q_0.50"]) == 4
    assert result["models_ho"]["q_0.50"].kwargs["alpha"] == 0.50
    assert 40.0 not in set(seen["eval_y"])
    assert set(seen["eval_y"]).issubset(set(train["minutes"]))


def test_holdout_rejects_both_or_neither_model():
    train, _, _ = _dated_frame(6)
    holdout = train.iloc[:2].copy()
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_holdout(
            train, holdout, features=["a"], target_col="minutes", quantiles=[0.50],
        )
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_holdout(
            train,
            holdout,
            features=["a"],
            target_col="minutes",
            xgb_params={"n_estimators": 1},
            lgb_params={"n_estimators": 1},
            quantiles=[0.50],
        )


def test_cv_rejects_both_or_neither_model():
    frame, X, y = _dated_frame(6)
    with pytest.raises(ValueError, match="exactly one"):
        run_timeseries_cv(X, y, frame, n_splits=2, quantiles=[0.50])
    with pytest.raises(ValueError, match="exactly one"):
        run_walk_forward(
            X, y, frame,
            xgb_params={"n_estimators": 1},
            lgb_params={"n_estimators": 1},
            n_folds=1,
            quantiles=[0.50],
        )


def test_tune_rejects_bad_inputs():
    X, y = _frame(4)
    with pytest.raises(ValueError, match="same length"):
        tune_lgb_quantile(X, y.iloc[:3], n_trials=1, n_splits=2)
    with pytest.raises(ValueError, match="n_splits"):
        tune_lgb_quantile(X.iloc[:2], y.iloc[:2], n_trials=1, n_splits=2)
    with pytest.raises(ValueError, match="quantile_alpha"):
        tune_lgb_quantile(X, y, n_trials=1, n_splits=2, quantile_alpha=0.0)
    with pytest.raises(ValueError, match="quantile_alpha"):
        tune_lgb_quantile(X, y, n_trials=1, n_splits=2, quantile_alpha=1.0)
