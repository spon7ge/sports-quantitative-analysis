# Minutes LightGBM baseline

Date: 2026-09-30
Approved: replace minutes Model 2 and the XGBoost model baseline with a five-quantile LightGBM peer. Last-game, season-to-date, and EWMA stay. The linear helper in `baselines.py` stays.

## Goal

`notebooks/nba/minutes/minutes.ipynb` currently compares a tuned five-quantile XGBoost with a q50-only linear `QuantileRegressor`. That linear fit is median-imputed and scaled, trained on every training row, and has no interval predictions, so coverage, width, and crossing for that row are blank.

Replace it with LightGBM trained the way XGBoost is trained: one shared parameter dictionary, Optuna-tuned at q50 on the training pool, then reused for 0.05, 0.10, 0.50, 0.90, and 0.95. Score both boosters on the 2025-26 holdout. A negative `xgb_minus_lgb` means the XGBoost score is lower.

This does not retune XGBoost, does not put LightGBM inside the date walk-forward table, and does not save a new model artifact.

## Pieces

| Piece | Path | Role |
|---|---|---|
| Dependency | `requirements.txt` | Pin `lightgbm==4.7.0` next to `xgboost==3.4.1` |
| Fit and tune | `models/shared/train.py` | `fit_quantile_lightgbm`, `tune_lgb_quantile` |
| Exports | `models/shared/__init__.py` | Export both functions beside the XGBoost helpers |
| Tests | `tests/models/test_train_lightgbm.py` | Fake `LGBMRegressor`. No real booster, no 40-trial search |
| Notebook | `notebooks/nba/minutes/minutes.ipynb` | Locked `LGB_PARAMS`, Model 2, comparison, writeup |

`models/shared/baselines.py` is unchanged. Points and assists notebooks are unchanged. Last-game minutes, season-to-date average, and EWMA stay in the point table and the paired-MAE bootstrap.

## Tuning

`tune_lgb_quantile(X, y, *, n_trials=40, n_splits=4, quantile_alpha=0.50, seed=42, fixed_params=None, show_progress_bar=True)` matches `tune_xgb_quantile`.

It sees only the training-pool features and minutes. The 2025-26 holdout is not an argument. `TimeSeriesSplit` is the search split. The date walk-forward folds are not.

Raise `ValueError` when `len(X) != len(y)`, when `len(X) <= n_splits`, or when `quantile_alpha` is outside `(0, 1)`.

Each trial builds `LGBMRegressor` with `objective="quantile"` and `alpha=quantile_alpha`. The validation fold is the `eval_set` and the scored fold. Early stopping is 50 rounds. The score is mean q50 pinball across splits. The study minimizes that score with a TPE sampler seeded by `seed`. This is the same early-stopping pattern as `tune_xgb_quantile`.

Fixed settings, overridable through `fixed_params`:

- `objective="quantile"`
- `n_jobs=-1`
- `random_state=seed`
- `verbose=-1`
- `bagging_freq=1`
- `early_stopping_rounds=50`

Search space:

| Parameter | Range |
|---|---|
| `n_estimators` | int, 500–2000 |
| `num_leaves` | int, 15–127 |
| `max_depth` | int, 3–12 |
| `learning_rate` | float, 0.01–0.2, log |
| `subsample` | float, 0.5–1.0 |
| `colsample_bytree` | float, 0.5–1.0 |
| `reg_alpha` | float, 1e-3–5.0, log |
| `reg_lambda` | float, 1e-3–5.0, log |
| `min_child_samples` | int, 10–200 |

`alpha` is not in the search and not in the returned parameter dictionary. The return value matches `tune_xgb_quantile`: `best_params` (fixed settings merged with the winning trial), `best_value`, and the Optuna `study`.

Implementation runs `tune_lgb_quantile` once on `ppm_df[MIN_FEATURES]` and `ppm_df[TARGET_COL]` and writes the returned `best_params` into `LGB_PARAMS` in the config cell. The tune call stays commented, same as the XGBoost tune cell. Later runs of Model 2 do not call Optuna.

## Fitting

`fit_quantile_lightgbm` has the same signature and return shape as `fit_quantile_models`: one regressor per quantile, prediction keys `q_{alpha:.2f}`, and a prediction array per key. When `quantiles` is omitted it uses the same `DEFAULT_QUANTILES` as `fit_quantile_models`. The notebook passes its own `QUANTILES`. `lgb_params` replaces `xgb_params` and is required.

`early_stop` accepts `"validation"` and `"train_tail"` and uses the same date-window checks as `fit_quantile_models`. `train_tail` raises `ValueError` before any booster is constructed when dates are missing, misaligned, or cover fewer than two distinct days, and when the stop window would consume every date. Any other `early_stop` value raises `ValueError`.

`early_stopping_rounds`, when present, is removed from the constructor dictionary and passed as `lightgbm.early_stopping(rounds, verbose=False)`. When it is absent, the fit uses no early-stopping callback. `alpha` in the shared dictionary raises `ValueError` before any booster is constructed. Each model sets its own `alpha` to that quantile.

Feature NaNs are passed through. There is no median imputer and no scaler. A LightGBM rejection propagates.

## Holdout score

Model 2 uses the same row-order cutoff as `evaluate_holdout` with `es_frac=0.90`. `ppm_df` is already in the order that function expects. The notebook does not sort it again.

```text
es_cutoff = int(len(ppm_df) * 0.90)
fit on ppm_df.iloc[:es_cutoff]
early-stop on ppm_df.iloc[es_cutoff:]
predict ppm_holdout[MIN_FEATURES]
```

The holdout is not an `eval_set`. Quantiles are the notebook's `QUANTILES`: 0.05, 0.10, 0.50, 0.90, 0.95. Predictions are stored as `preds_lgb`.

## Notebook comparison

Remove the `QuantileRegressor`, `SimpleImputer`, and `StandardScaler` imports. They are used only by Model 2.

The config cell holds `LGB_PARAMS` beside `XGB_PARAMS`. A commented `tune_lgb_quantile(...)` call sits beside the commented XGBoost tune call.

The pinball table columns are `pinball_xgb`, `pinball_lgb`, and `xgb_minus_lgb`. The printed line says a negative difference means the XGBoost pinball is lower.

Uncomment the Model 2 coverage merge. Overlapping `coverage` columns become `coverage_xgb` and `coverage_lgb`. `interval` and `target` stay unsuffixed.

Uncomment `interval_sharpness` and its comparison. Overlapping `coverage`, `mean_width`, and `median_width` columns get `_xgb` and `_lgb` suffixes. `mean_width_xgb_minus_lgb` is `mean_width_xgb` minus `mean_width_lgb`. A negative width difference means the XGBoost interval is narrower.

In the baseline section, `LIN_NAME` and `preds_lin` become `LGB_NAME = "Quantile LightGBM"` and `preds_lgb`. Point accuracy, the date-clustered MAE difference, WIS, calibration, and the closing writeup use that name. LightGBM interval columns fill in because the five quantile keys exist. The walk-forward fold table stays XGBoost-only. Its note says LightGBM is scored on the holdout, not inside those folds.

Stored cell outputs and the closing numeric writeup are refreshed from this holdout run.

## Tests

`tests/models/test_train_lightgbm.py` monkeypatches `LGBMRegressor`.

- Five requested quantiles produce keys `q_0.05`, `q_0.10`, `q_0.50`, `q_0.90`, `q_0.95`, each model receives its own `alpha`, and `predict` is called on the requested frame.
- `early_stopping_rounds` becomes an early-stopping callback and is not a constructor argument.
- `alpha` inside `lgb_params` raises `ValueError` before a booster is constructed.
- Missing `lgb_params` raises `ValueError`.
- An unknown `early_stop` raises `ValueError`.
- `train_tail` with one date raises `ValueError` before a booster is constructed.
- `tune_lgb_quantile` on a tiny frame, one Optuna trial, two `TimeSeriesSplit` folds, and the fake regressor returns `best_params` that contain the fixed settings and do not contain `alpha`. The fake regressor is constructed with `alpha=0.50`.
- `tune_lgb_quantile` raises `ValueError` for unequal `X` and `y`, too few rows for the split, and a quantile outside `(0, 1)`.

The 40-trial minutes search and the notebook rerun are not part of pytest.
