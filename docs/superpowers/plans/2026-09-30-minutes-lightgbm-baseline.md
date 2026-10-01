# Minutes LightGBM Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the minutes notebook's q50 linear model with a five-quantile LightGBM peer, tuned on the training pool only, and compare that peer to the locked XGBoost holdout.

**Architecture:** `models/shared/train.py` gains `fit_quantile_lightgbm` and `tune_lgb_quantile` beside the XGBoost helpers. A private `_fit_lgb_regressor` owns the quantile `alpha` and the early-stopping callback. The minutes notebook locks the winning parameters, fits on the first 90% of `ppm_df`, early-stops on the last 10%, and scores the 2025-26 holdout. `models/shared/baselines.py` stays.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn `TimeSeriesSplit`, Optuna, LightGBM 4.7.0, pytest, the existing minutes notebook.

## Global Constraints

- Pin `lightgbm==4.7.0` in `requirements.txt` next to `xgboost==3.4.1`
- Quantiles the notebook fits: `0.05, 0.10, 0.50, 0.90, 0.95` (`QUANTILES`)
- `fit_quantile_lightgbm` default quantiles are `DEFAULT_QUANTILES` in `models/shared/train.py` (the eleven-level list)
- Tune q50 only, on `ppm_df[MIN_FEATURES]` and `ppm_df[TARGET_COL]`, `n_trials=40`, `n_splits=4`, `seed=42`
- Search: `n_estimators` int 500–2000, `num_leaves` int 15–127, `max_depth` int 3–12, `learning_rate` float 0.01–0.2 log, `subsample` float 0.5–1.0, `colsample_bytree` float 0.5–1.0, `reg_alpha` float 1e-3–5.0 log, `reg_lambda` float 1e-3–5.0 log, `min_child_samples` int 10–200
- Fixed tune settings: `objective="quantile"`, `n_jobs=-1`, `random_state=seed`, `verbose=-1`, `bagging_freq=1`, `early_stopping_rounds=50`
- `alpha` is never in the shared parameter dictionary
- Holdout cutoff is `es_cutoff = int(len(ppm_df) * 0.90)`; the holdout is not an `eval_set`
- No median imputer and no scaler
- A negative `xgb_minus_lgb` means the XGBoost pinball is lower
- Last-game minutes, season-to-date average, and EWMA stay
- The date walk-forward fold table stays XGBoost-only
- Do not retune XGBoost, do not edit `models/shared/baselines.py`, do not save a new model artifact
- Pytest monkeypatches `LGBMRegressor` and does not run the 40-trial search

## File map

- Modify: `requirements.txt` — pin LightGBM
- Modify: `models/shared/train.py` — fitter, tuner, private booster helper
- Modify: `models/shared/__init__.py` — export both public functions
- Create: `tests/models/test_train_lightgbm.py` — fake-booster tests
- Modify: `notebooks/nba/minutes/minutes.ipynb` — locked params, Model 2, comparison, writeup

---

### Task 1: LightGBM quantile fitter

**Files:**
- Modify: `requirements.txt`
- Modify: `models/shared/train.py`
- Modify: `models/shared/__init__.py`
- Create: `tests/models/test_train_lightgbm.py`
- Test: `tests/models/test_train_lightgbm.py`

**Interfaces:**
- Consumes: `DEFAULT_QUANTILES` already defined in `models/shared/train.py`
- Produces:
  - `_fit_lgb_regressor(params: dict, alpha: float, X_fit, y_fit, X_es, y_es) -> LGBMRegressor`
  - `fit_quantile_lightgbm(X_train, y_train, X_val, y_val, *, quantiles: Sequence[float] | None = None, lgb_params: dict | None = None, early_stop: str = "validation", train_dates=None, train_tail_frac: float = 0.10, X_predict=None) -> tuple[dict[str, Any], dict[str, np.ndarray]]`
  - Prediction keys `q_{alpha:.2f}`

- [ ] **Step 1: Pin LightGBM and install it**

In `requirements.txt`, immediately after `xgboost==3.4.1`, add:

```text
lightgbm==4.7.0
```

Run:

```powershell
.venv\Scripts\python.exe -m pip install lightgbm==4.7.0
```

Expected: `Successfully installed lightgbm-4.7.0` (or `Requirement already satisfied`).

- [ ] **Step 2: Write the failing fitter tests**

Create `tests/models/test_train_lightgbm.py`:

```python
import numpy as np
import pandas as pd
import pytest

from models.shared import train as train_mod
from models.shared.train import fit_quantile_lightgbm


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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/models/test_train_lightgbm.py -v
```

Expected: FAIL with `ImportError` or `cannot import name 'fit_quantile_lightgbm'`.

- [ ] **Step 4: Implement the fitter**

In `models/shared/train.py`, add this import beside the XGBoost import:

```python
from lightgbm import LGBMRegressor, early_stopping
```

Add this helper and function after `fit_quantile_models`. Do not change `fit_quantile_models`.

```python
def _fit_lgb_regressor(params: dict, alpha: float, X_fit, y_fit, X_es, y_es):
    """Fit one quantile LightGBM model. ``alpha`` is not taken from ``params``."""
    if "alpha" in params:
        raise ValueError("alpha is set per quantile, not in lgb_params")
    cleaned = dict(params)
    rounds = cleaned.pop("early_stopping_rounds", None)
    model = LGBMRegressor(**cleaned, alpha=alpha)
    fit_kwargs: dict[str, Any] = {"eval_set": [(X_es, y_es)]}
    if rounds is not None:
        fit_kwargs["callbacks"] = [early_stopping(int(rounds), verbose=False)]
    model.fit(X_fit, y_fit, **fit_kwargs)
    return model


def fit_quantile_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    quantiles: Sequence[float] | None = None,
    lgb_params: dict | None = None,
    early_stop: str = "validation",
    train_dates: pd.Series | np.ndarray | Sequence | None = None,
    train_tail_frac: float = 0.10,
    X_predict: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Train one LightGBM quantile model per quantile; return models + preds."""
    if early_stop == "train_tail":
        if train_dates is None:
            raise ValueError("train_tail requires train_dates")
        dates = pd.to_datetime(np.asarray(train_dates))
        if len(dates) != len(X_train):
            raise ValueError("train_dates must align with X_train rows")
        unique_dates = pd.unique(dates)
        n_unique = len(unique_dates)
        if n_unique < 2:
            raise ValueError("train_tail requires at least two training dates")
        n_stop = int(np.floor(n_unique * train_tail_frac))
        if n_stop < 1:
            n_stop = 1
        if n_stop >= n_unique:
            raise ValueError(
                f"train_tail n_stop={n_stop} must be strictly less than "
                f"n_unique={n_unique}"
            )
        stop_dates = set(unique_dates[-n_stop:])
        fit_mask = ~pd.Series(dates).isin(stop_dates).to_numpy()
        stop_mask = pd.Series(dates).isin(stop_dates).to_numpy()
        X_fit = X_train.iloc[fit_mask] if hasattr(X_train, "iloc") else X_train[fit_mask]
        y_fit = y_train.iloc[fit_mask] if hasattr(y_train, "iloc") else y_train[fit_mask]
        X_es = X_train.iloc[stop_mask] if hasattr(X_train, "iloc") else X_train[stop_mask]
        y_es = y_train.iloc[stop_mask] if hasattr(y_train, "iloc") else y_train[stop_mask]
        predict_X = X_val if X_predict is None else X_predict
    elif early_stop == "validation":
        X_fit, y_fit = X_train, y_train
        X_es, y_es = X_val, y_val
        predict_X = X_val if X_predict is None else X_predict
    else:
        raise ValueError(f"unknown early_stop={early_stop!r}")

    if lgb_params is None:
        raise ValueError("lgb_params is required (define in the prop notebook)")
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    params = dict(lgb_params)
    models: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for q in quantiles:
        model = _fit_lgb_regressor(params, q, X_fit, y_fit, X_es, y_es)
        key = f"q_{q:.2f}"
        models[key] = model
        preds[key] = np.asarray(model.predict(predict_X), dtype=float)
    return models, preds
```

In `models/shared/__init__.py`, add `fit_quantile_lightgbm` to the `models.shared.train` import and to `__all__` immediately after `fit_quantile_models`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/models/test_train_lightgbm.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 6: Commit**

```powershell
git add -- requirements.txt models/shared/train.py models/shared/__init__.py tests/models/test_train_lightgbm.py
git commit -m @"
Add a LightGBM quantile fitter beside the XGBoost one.
"@
```

---

### Task 2: Training-pool tuner

**Files:**
- Modify: `models/shared/train.py`
- Modify: `models/shared/__init__.py`
- Modify: `tests/models/test_train_lightgbm.py`
- Test: `tests/models/test_train_lightgbm.py`

**Interfaces:**
- Consumes: `_fit_lgb_regressor` from Task 1; `pinball_loss` from `models.shared.metrics`
- Produces: `tune_lgb_quantile(X, y, *, n_trials: int = 40, n_splits: int = 4, quantile_alpha: float = 0.50, seed: int = 42, fixed_params: dict | None = None, show_progress_bar: bool = True) -> dict[str, Any]` with keys `best_params`, `best_value`, `study`

- [ ] **Step 1: Write the failing tuner tests**

Append to `tests/models/test_train_lightgbm.py`:

```python
from models.shared.train import tune_lgb_quantile


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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/models/test_train_lightgbm.py::test_tune_locks_fixed_settings_and_q50_alpha tests/models/test_train_lightgbm.py::test_tune_rejects_bad_inputs -v
```

Expected: FAIL with `cannot import name 'tune_lgb_quantile'`.

- [ ] **Step 3: Implement the tuner**

Add this function to `models/shared/train.py` after `tune_xgb_quantile`. Do not change `tune_xgb_quantile`.

```python
def tune_lgb_quantile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 40,
    n_splits: int = 4,
    quantile_alpha: float = 0.50,
    seed: int = 42,
    fixed_params: dict | None = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Optuna-tune one LightGBM quantile model via TimeSeriesSplit pinball loss.

    Use train-pool ``X`` / ``y`` only — never the locked holdout.
    Returns merged ``best_params`` (ready for ``LGB_PARAMS``), ``best_value``,
    and the Optuna ``study``. ``alpha`` is not part of ``best_params``.
    """
    import optuna

    from models.shared.metrics import pinball_loss

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) <= n_splits:
        raise ValueError(f"len(X)={len(X)} must be greater than n_splits={n_splits}")
    if not 0.0 < quantile_alpha < 1.0:
        raise ValueError("quantile_alpha must be inside (0, 1)")

    fixed = {
        "objective": "quantile",
        "n_jobs": -1,
        "random_state": seed,
        "verbose": -1,
        "bagging_freq": 1,
        "early_stopping_rounds": 50,
        **(fixed_params or {}),
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: Any) -> float:
        params = {
            **fixed,
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        }
        losses: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = _fit_lgb_regressor(params, quantile_alpha, X_tr, y_tr, X_val, y_val)
            losses.append(pinball_loss(y_val, model.predict(X_val), quantile_alpha))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best_params = {**fixed, **study.best_params}
    print(f"Best pinball (mean TSCV, q={quantile_alpha}): {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "study": study,
    }
```

In `models/shared/__init__.py`, add `tune_lgb_quantile` to the `models.shared.train` import and to `__all__` immediately after `tune_xgb_quantile`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/models/test_train_lightgbm.py -v
```

Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```powershell
git add -- models/shared/train.py models/shared/__init__.py tests/models/test_train_lightgbm.py
git commit -m @"
Tune LightGBM quantile models on the training pool only.
"@
```

---

### Task 3: Notebook comparison source

**Files:**
- Modify: `notebooks/nba/minutes/minutes.ipynb` cells 2, 3, 20, 27, 28, and 30
- Test: source check below. Do not execute the notebook in this task.

**Interfaces:**
- Consumes: `fit_quantile_lightgbm` and `tune_lgb_quantile` from Task 2. Session names already in the notebook: `ppm_df`, `ppm_holdout`, `MIN_FEATURES`, `TARGET_COL`, `QUANTILES`, `HOLDOUT_SEASON`, `SEED`, `score_predictions`, `ho_pinball`, `ho_coverage`, `y_ho`, `preds_ho`, `wf_results`, `interval_coverage`, `pinball_loss`.
- Produces: `LGB_PARAMS` (set to `None` until Task 4), `preds_lgb`, `LGB_NAME = "Quantile LightGBM"`, `lgb_q50_name`

- [ ] **Step 1: Drop the linear imports**

In cell 2, delete these three lines:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
```

- [ ] **Step 2: Lock a placeholder and import the fitter**

In cell 3, immediately after the `XGB_PARAMS = dict(...)` block, add:

```python
# Filled by the locked tune_lgb_quantile run. Do not hand-edit.
LGB_PARAMS = None
```

Replace the train import with:

```python
from models.shared.train import (
    evaluate_holdout,
    fit_quantile_lightgbm,
    run_timeseries_cv,
    run_walk_forward,
    tune_lgb_quantile,
)
```

- [ ] **Step 3: Comment the LightGBM tune beside the XGBoost tune**

Replace cell 20 with:

```python
# Hyperparameter tuning
# Leave commented if you don't want to retune the model

# import importlib
# import models.shared.train as train

# importlib.reload(train)
# tune_xgb_quantile = train.tune_xgb_quantile

# result = tune_xgb_quantile(X, y, n_trials=50, n_splits=3)
# XGB_PARAMS = result["best_params"]

# lgb_result = tune_lgb_quantile(
#     ppm_df[MIN_FEATURES],
#     ppm_df[TARGET_COL],
#     n_trials=40,
#     n_splits=4,
#     seed=SEED,
# )
# LGB_PARAMS = lgb_result["best_params"]
```

- [ ] **Step 4: Replace Model 2**

Replace cell 27 with:

```python
# Model 2: LightGBM quantile regression

es_cutoff = int(len(ppm_df) * 0.90)
fit_df = ppm_df.iloc[:es_cutoff]
stop_df = ppm_df.iloc[es_cutoff:]
models_lgb, _ = fit_quantile_lightgbm(
    fit_df[MIN_FEATURES],
    fit_df[TARGET_COL],
    stop_df[MIN_FEATURES],
    stop_df[TARGET_COL],
    quantiles=QUANTILES,
    lgb_params=LGB_PARAMS,
)
X_lgb_ho = ppm_holdout[MIN_FEATURES]
y_lgb_ho = ppm_holdout[TARGET_COL].to_numpy(dtype=float)
preds_lgb = {key: model.predict(X_lgb_ho) for key, model in models_lgb.items()}

lgb_pinball, lgb_coverage = score_predictions(
    y_lgb_ho, preds_lgb, f"{HOLDOUT_SEASON} lightgbm"
)

cmp_pinball = (
    ho_pinball[["quantile", "pinball"]]
    .merge(
        lgb_pinball[["quantile", "pinball"]],
        on="quantile",
        suffixes=("_xgb", "_lgb"),
    )
)
cmp_pinball["xgb_minus_lgb"] = (
    cmp_pinball["pinball_xgb"] - cmp_pinball["pinball_lgb"]
)
cmp_coverage = (
    ho_coverage[["interval", "coverage", "target"]]
    .merge(
        lgb_coverage[["interval", "coverage"]],
        on="interval",
        suffixes=("_xgb", "_lgb"),
    )
)

print(
    f"{HOLDOUT_SEASON} holdout — LightGBM quantile vs XGBoost\n"
    f"  train rows {len(ppm_df):,} | fit rows {es_cutoff:,} | "
    f"early-stop rows {len(ppm_df) - es_cutoff:,} | holdout rows {len(y_lgb_ho):,}\n"
    "  xgb_minus_lgb < 0 means the XGBoost pinball is lower"
)
display(cmp_pinball.round(4))
display(cmp_coverage.round(4))
```

- [ ] **Step 5: Restore interval sharpness**

Replace cell 28 with:

```python
def interval_sharpness(y_true, preds):
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for low, high, nominal in ((0.10, 0.90, 0.80), (0.05, 0.95, 0.90)):
        low_key, high_key = f"q_{low:.2f}", f"q_{high:.2f}"
        width = (
            np.asarray(preds[high_key], dtype=float)
            - np.asarray(preds[low_key], dtype=float)
        )
        rows.append({
            "interval": f"{nominal:.0%}",
            "coverage": interval_coverage(y_true, preds[low_key], preds[high_key]),
            "target": nominal,
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)),
        })
    return pd.DataFrame(rows)

sharp_xgb = interval_sharpness(y_ho, preds_ho)
sharp_lgb = interval_sharpness(y_lgb_ho, preds_lgb)

cmp_sharpness = (
    sharp_xgb[["interval", "target", "coverage", "mean_width", "median_width"]]
    .merge(
        sharp_lgb[["interval", "coverage", "mean_width", "median_width"]],
        on="interval",
        suffixes=("_xgb", "_lgb"),
    )
)
cmp_sharpness["mean_width_xgb_minus_lgb"] = (
    cmp_sharpness["mean_width_xgb"] - cmp_sharpness["mean_width_lgb"]
)

print(
    f"{HOLDOUT_SEASON} holdout — interval sharpness (minutes)\n"
    "  mean_width_xgb_minus_lgb < 0 means the XGBoost interval is narrower"
)
display(cmp_sharpness.round(4))
```

- [ ] **Step 6: Point the baseline cell at LightGBM**

In cell 30, apply these replacements and no others. Leave `baselines`, the walk-forward fold loop, and `WIS_INTERVALS` as they are.

```python
LGB_NAME = "Quantile LightGBM"
```

replaces `LIN_NAME = "Linear quantile"`.

```python
    LGB_NAME: {key: np.asarray(pred, dtype=float) for key, pred in preds_lgb.items()},
```

replaces the `LIN_NAME` / `preds_lin` entry.

```python
lgb_q50_name = f"{LGB_NAME} q50"
```

replaces `lin_q50_name = f"{LIN_NAME} q50"`.

Replace every remaining `lin_q50_name` with `lgb_q50_name`, every remaining `LIN_NAME` with `LGB_NAME`, `lin_dist` with `lgb_dist`, `lin_cal` with `lgb_cal`, and `lin_delta` with `lgb_delta`.

Replace the three print strings:

```python
    f"LightGBM q50 pinball {pinball_loss(y_paired, paired_preds[lgb_q50_name], 0.50):.4f}. "
```

```python
    "LightGBM is scored on the holdout, not inside these folds. "
```

```python
    "XGBoost minus LightGBM WIS "
    f"{xgb_dist['wis'] - lgb_dist['wis']:+.4f}. "
    "Negative is the better distribution score."
```

The last block deletes the sentence that said the linear row has q50 only. Both models now have the five quantile keys, so WIS uses the intervals whose endpoints are present plus the q50 term.

```python
    f"  XGBoost q50 vs LightGBM q50: delta MAE {lgb_delta['delta_mae']:+.3f} "
```

```python
    f"  WIS {xgb_dist['wis']:.3f} (XGBoost) vs {lgb_dist['wis']:.3f} (LightGBM) "
```

```python
    f"{lgb_dist['quantile_crossing']:.2%} (LightGBM)"
```

- [ ] **Step 7: Check the notebook source**

Run:

```powershell
.venv\Scripts\python.exe -c @'
import json
nb = json.load(open(r"notebooks/nba/minutes/minutes.ipynb", encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
for gone in ("QuantileRegressor", "SimpleImputer", "StandardScaler", "preds_lin", "LIN_NAME", "xgb_minus_linear", "pinball_linear"):
    assert gone not in src, gone
for present in ("fit_quantile_lightgbm", "tune_lgb_quantile", "LGB_PARAMS = None", "preds_lgb", "LGB_NAME = \"Quantile LightGBM\"", "xgb_minus_lgb", "mean_width_xgb_minus_lgb", "LightGBM is scored on the holdout"):
    assert present in src, present
print("notebook source ok")
'@
```

Expected: `notebook source ok`.

- [ ] **Step 8: Commit**

```powershell
git add -- notebooks/nba/minutes/minutes.ipynb
git commit -m @"
Compare the minutes holdout to LightGBM instead of linear quantile regression.
"@
```

---

### Task 4: Lock the search and refresh the writeup

**Files:**
- Modify: `notebooks/nba/minutes/minutes.ipynb` cell 3 (`LGB_PARAMS`), cell 43 (conclusion)
- Test: source check below. This task runs the real 40-trial search once.

**Interfaces:**
- Consumes: `tune_lgb_quantile`, `ppm_df`, `MIN_FEATURES`, `TARGET_COL`, `SEED`, plus the session state from cells 21–23 (`preds_ho`, `y_ho`, `ho_pinball`, `ho_coverage`, `wf_results`)
- Produces: numeric `LGB_PARAMS` equal to `tune_lgb_quantile(...)["best_params"]`, refreshed outputs for cells 27, 28, and 30, and a conclusion that names LightGBM

Do not call `tune_xgb_quantile`. Do not save a joblib.

- [ ] **Step 1: Rebuild the session through the XGBoost holdout**

Execute notebook cells 2, 3, 5, 9, 10, 11, 13, 21, 22, and 23 in order, in one kernel. Those cells load data, build `ppm_df`, and fit the locked XGBoost holdout. Cell 3 still has `LGB_PARAMS = None` at this point. That is expected. Do not execute cell 27 yet.

- [ ] **Step 2: Run the search once**

In that same kernel:

```python
lgb_result = tune_lgb_quantile(
    ppm_df[MIN_FEATURES],
    ppm_df[TARGET_COL],
    n_trials=40,
    n_splits=4,
    seed=SEED,
    show_progress_bar=True,
)
assert "alpha" not in lgb_result["best_params"]
assert lgb_result["best_params"]["objective"] == "quantile"
print(lgb_result["best_value"])
print(lgb_result["best_params"])
```

Expected: one printed pinball and one parameter dictionary. The holdout frame is not passed in.

- [ ] **Step 3: Write the winner into the config cell**

Replace `LGB_PARAMS = None` in cell 3 with a `LGB_PARAMS = dict(...)` literal that contains every key and value from `lgb_result["best_params"]`. Use `repr` for the floats so they are not rounded. Keep the comment `# Filled by the locked tune_lgb_quantile run. Do not hand-edit.`

Confirm the literal contains `objective`, `early_stopping_rounds`, `bagging_freq`, and the nine searched keys, and does not contain `alpha`.

- [ ] **Step 4: Score the holdout**

Re-execute cell 3 so the kernel sees the literal, then execute cells 27, 28, and 30. Cell 27 must not be edited to call Optuna. Save `notebooks/nba/minutes/minutes.ipynb` after those cells finish so the stored outputs are the new displays. Expected displays:

- `cmp_pinball` columns `quantile`, `pinball_xgb`, `pinball_lgb`, `xgb_minus_lgb`
- `cmp_coverage` columns `interval`, `coverage_xgb`, `target`, `coverage_lgb`
- `cmp_sharpness` with `mean_width_xgb_minus_lgb`
- point table rows for last-game, season-to-date, EWMA, `Quantile LightGBM q50`, and `Quantile XGBoost q50`
- a walk-forward table whose model column is still XGBoost only

- [ ] **Step 5: Rewrite the conclusion from those tables**

In the same kernel, print the replacement markdown:

```python
def _mae(name):
    return float(point_table.loc[point_table["model"] == name, "mae"].iloc[0])

xgb_mae = _mae(xgb_q50_name)
lgb_mae = _mae(lgb_q50_name)
ewma_mae = _mae("EWMA minutes")
season_mae = _mae("Season-to-date average")
last_mae = _mae("Last-game minutes")
lgb_delta = delta_table.loc[
    delta_table["comparison"] == f"XGBoost q50 − {lgb_q50_name}"
].iloc[0]
ewma_delta = delta_table.loc[
    delta_table["comparison"] == "XGBoost q50 − EWMA minutes"
].iloc[0]
q50_pinball = float(ho_pinball.loc[ho_pinball["quantile"] == 0.50, "pinball"].iloc[0])
cov_80 = float(ho_coverage.loc[ho_coverage["interval"] == "80%", "coverage"].iloc[0])
cov_90 = float(ho_coverage.loc[ho_coverage["interval"] == "90%", "coverage"].iloc[0])
n_paired = int(paired.sum())
width_80 = float(cmp_sharpness.loc[cmp_sharpness["interval"] == "80%", "mean_width_xgb"].iloc[0])
width_90 = float(cmp_sharpness.loc[cmp_sharpness["interval"] == "90%", "mean_width_xgb"].iloc[0])
print(
    f"q50 pinball {q50_pinball:.3f} (MAE {2 * q50_pinball:.2f}). "
    f"80% coverage {cov_80:.1%}. 90% coverage {cov_90:.1%}. "
    f"On the {n_paired:,} paired rows, MAE is {xgb_mae:.3f} (XGBoost), "
    f"{lgb_mae:.3f} (LightGBM q50), {ewma_mae:.3f} (EWMA), "
    f"{season_mae:.3f} (season-to-date), and {last_mae:.3f} (last game). "
    f"XGBoost minus EWMA MAE {ewma_delta['delta_mae']:+.2f} "
    f"(CI {ewma_delta['ci_low']:+.2f} to {ewma_delta['ci_high']:+.2f}). "
    f"XGBoost minus LightGBM MAE {lgb_delta['delta_mae']:+.2f} "
    f"(CI {lgb_delta['ci_low']:+.2f} to {lgb_delta['ci_high']:+.2f}). "
    f"WIS {xgb_dist['wis']:.3f} (XGBoost) vs {lgb_dist['wis']:.3f} (LightGBM). "
    f"Quantile crossing {xgb_dist['quantile_crossing']:.2%} (XGBoost), "
    f"{lgb_dist['quantile_crossing']:.2%} (LightGBM). "
    f"XGBoost mean width 80% {width_80:.1f} minutes, 90% {width_90:.1f}."
)
```

Paste that sentence into cell 43 in place of the clauses that cite linear q50. In the selected-model bullet, the recommendation bullet, and the primary-test bullet, name LightGBM q50 instead of linear q50. State the MAE ordering from `point_table` rather than keeping the old claim that XGBoost is lowest if LightGBM is lower. Leave the slice findings and the Next Steps list unchanged. Leave the saved-model path `models/saved_models/min_nba_model_2026-04-12.joblib` unchanged.

- [ ] **Step 6: Check the saved notebook**

Run:

```powershell
.venv\Scripts\python.exe -c @'
import json
nb = json.load(open(r"notebooks/nba/minutes/minutes.ipynb", encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
assert "LGB_PARAMS = None" not in src
assert "linear q50" not in src
assert "Linear quantile" not in src
model2 = next(c for c in nb["cells"] if "".join(c["source"]).startswith("# Model 2: LightGBM"))
blob = json.dumps(model2.get("outputs", []))
assert "pinball_lgb" in blob
print("locked params and writeup ok")
'@
```

Expected: `locked params and writeup ok`.

- [ ] **Step 7: Commit**

```powershell
git add -- notebooks/nba/minutes/minutes.ipynb
git commit -m @"
Lock the tuned LightGBM minutes baseline and record the holdout comparison.
"@
```
