# Points quantile notebook package

Date: 2026-10-26  
Status: approved  
Approved decisions: option B (package plus conformal); one shared conformal implementation with a minutes adapter and a rate adapter.

## Goal

Make `notebooks/nba/points/points_xgboost.ipynb` follow the training sequence in `notebooks/nba/minutes/02_minutes_xgboost.ipynb`, and move the reusable points logic into `notebooks/nba/points/src/points_xgb` so a later notebook can import it.

The points target stays capped points per minute. The new sections are the walk-forward conformal offsets and the date-tail comparison of raw knots beside calibrated knots, using rate tiers and the rate cap.

This slice does not retrain the model and does not replace the frozen conclusion in the notebook. Those cells stay empty until someone runs the notebook.

## Scope

### In

- A `points_xgb` package that owns the logic currently inlined in `points_xgboost.ipynb`.
- Notebook cells that call that package, in the same section order as `02_minutes_xgboost.ipynb`.
- Rate conformal offsets fit only on walk-forward out-of-fold rows.
- A date-tail holdout display: raw booster knots beside calibrated knots.
- Shared conformal code that still serves the minutes notebook through the current `minutes_xgb.calibration` signatures.
- Tests for the rate spec and for the minute defaults that already exist.

### Out

- Retraining, retuning, or editing `PTS_FEATURES`.
- Changing minute pinball, coverage, clip `[0.001, max_minutes]`, or minute tier cuts.
- Importing `minutes_xgb` from the points package.
- Overwriting `models/saved_models/pts_nba_tails_2026-04-12.joblib`. That file stays the uncalibrated four-fold sidecar written by `scripts/build_oos_table.py`.
- Changing `scripts/build_oos_table.py`. Extra `cal_q_*` columns on the rate parquet are ignored because that script selects its own column list.
- Committing model bundles, tail sidecars, or parquet predictions.
- Rewording the "minute tiers by predicted q50" print inside `score_quantile_fold`.

## Rate spec

One definition, `RATE_Q50_TIERS`, lives in `models/shared/metrics.py` next to `MINUTE_Q50_TIERS`:

| Group | Predicted base q50 |
|---|---|
| `cold_start` | `pts_per_min_lag_1` is missing, regardless of q50 |
| `<0.30` | q50 < 0.30 |
| `0.30-0.50` | 0.30 ≤ q50 < 0.50 |
| `0.50-0.70` | 0.50 ≤ q50 < 0.70 |
| `0.70+` | q50 ≥ 0.70 |

Cuts are left-closed, matching the minute tiers. `points_xgb.config.RATE_TIERS` is the lambda map `run_timeseries_cv` and `run_walk_forward` already receive, built from these same names and cuts. A test locks the two to each other.

| Fact | Value |
|---|---|
| Outcome column | `pts_per_min` |
| Cold-start lag | `pts_per_min_lag_1` |
| Clip after offset and sort | `[0, 6]` |
| Cap | `models.shared.oos.RATE_CAP` (6) |
| Floor | 0, so a scoreless game stays a real zero |
| `min_n` | 300 |
| Offset sample | walk-forward OOF only |
| Holdout | 2025-26, scored after the offset fit |
| Role fallback | empirical quantiles of `pts_per_min` by `starting`, from that fold's training rows for OOF base knots, and from the whole pre-holdout pool for the holdout fallback |

`run_walk_forward` gains `target_col: str = "minutes"`. The outcome is stored as `oof[target_col]`. The default keeps the minutes column the minutes notebook already reads. The points notebook passes `target_col="pts_per_min"` and does not also write a column named `minutes` for the rate. Realized minutes in the OOS export still come from the identity frame inside `oof_prediction_frame`. Update that function's docstring so it no longer says `oof["minutes"]` holds the capped rate.

## Shared conformal interface

`models.shared.conformal` keeps the offset math. Existing minute keyword arguments keep working.

- `calibration_group(q50, lag, tiers=MINUTE_Q50_TIERS)`. The second argument is the cold-start lag. A missing lag is `cold_start`. Keyword callers that pass the lag positionally stay valid. The parameter name stays `min_lag_1` so current keyword uses do not break.
- `base_knots(..., target_col="minutes")` and `role_quantile_table(..., target_col="minutes")` read that column from history. The default column is `minutes`.
- `apply_offsets(base, group, offsets, max_minutes, *, lower=1e-3)`. The fourth argument is the inclusive upper clip. The name stays `max_minutes`. `lower` defaults to `1e-3`. A rate call passes `RATE_CAP` and `lower=0`.
- `calibration_payload` gains optional `target`, `lag_col`, `tiers`, and `clip_low`, with minute defaults: target `minutes`, lag `min_lag_1`, `MINUTE_Q50_TIERS`, `clip_low=1e-3`. It still writes `max_minutes` equal to the upper clip so a minute bundle saved after this change loads the way it does today.

The walk-forward report moves to `models.shared.conformal_report`. A frozen `ConformalSpec` carries `target_col`, `lag_col`, `tiers`, `clip_low`, `clip_high`, and `min_n`. `calibration_table` and `nested_offset_backtest` take that spec. Group slices in the table follow `spec.tiers` plus `cold_start`, then starter, bench, and month. `nested_offset_backtest` reads `oof[spec.target_col]` and clips with `spec.clip_low` and `spec.clip_high`. Score differences stay calibrated minus raw. Lower WIS, CRPS, and q50 pinball are better. CRPS uses the existing knot `row_crps`.

`minutes_xgb.calibration.calibration_table` and `nested_offset_backtest` keep the signatures `02_minutes_xgboost.ipynb` calls, including `max_minutes`. They build a minute spec (`clip_low=1e-3`, `clip_high=max_minutes`, minute tiers, target `minutes`, lag `min_lag_1`) and call the shared report.

`models.shared.artifacts._check_calibration`:

- No calibration key: load succeeds, as today.
- Booster fingerprint must still match.
- Missing `target`, or `target == "minutes"`: stored tiers must equal `MINUTE_Q50_TIERS`.
- `target == "pts_per_min"`: stored tiers must equal `RATE_Q50_TIERS`.
- Any other target: `ValueError`.

`predict_calibrated_quantiles` selects `bundle["feature_names"]` from the frame, so the lag column may sit beside the model features. Points per minute lag is not a model feature. It reads `lag_col` (default `min_lag_1`), clips with `clip_low` (default `1e-3`) and `clip_high` (default `max_minutes`), and groups with the stored tiers. A minute bundle that only has `max_minutes` and `min_lag_1` keeps the path it uses now. A missing lag column raises `KeyError`.

## Files

| Path | Owns |
|---|---|
| `notebooks/nba/points/src/points_xgb/__init__.py` | Notebook exports |
| `.../config.py` | Holdout season, ids, `PTS_FEATURES`, excluded columns, `XGB_PARAMS`, `QUANTILES`, `RATE_TIERS`, `RATE_CAP`, `SEED`, `N_BOOT`, `ARTIFACT_STEM` |
| `.../paths.py` | Package path and repo root, same rules as `minutes_xgb.paths` |
| `.../data.py` | Load the seven regular-season gamelogs, flag starters, points quality summary |
| `.../features.py` | `add_points_features`, positive-minute filter, capped target, naive lag / season mean / EWMA columns, train-pool Tukey fence |
| `.../eda.py` | Missingness, rate outlier summary, season coverage |
| `.../scoring.py` | Pinball and coverage tables, calibration deltas, crossing, predicted-rate tiers, sharpness, WIS, knot CRPS summary |
| `.../baseline.py` | q50 compared with last game, season-to-date mean, and EWMA, date-clustered bootstrap (`N_BOOT=2000`, seed 42) |
| `.../importance.py` | Mean absolute SHAP, permutation importance, consensus |
| `.../slices.py` | Slice score and regular-season month blocks |
| `.../errors.py` | Holdout miss frame. Score, fouls, and plus/minus are display context, never features |
| `.../artifacts.py` | Exploratory metadata, loaded holdout scores, OOS frame with optional `cal_q_*`, write `data/oos/nba/rate_predictions.parquet` |
| `.../calibration.py` | Rate spec, `require_preholdout`, OOF base knots, go/no-go table, nested offset backtest, holdout raw-vs-calibrated summary, tail sidecar |
| `models/shared/conformal.py` | Parameterized grouping, base knots, offsets, payload |
| `models/shared/conformal_report.py` | `ConformalSpec`, calibration table, nested backtest |
| `models/shared/metrics.py` | `RATE_Q50_TIERS` |
| `models/shared/train.py` | `run_walk_forward(..., target_col="minutes")` |
| `models/shared/artifacts.py` | Tier check by target; calibrated predict reads lag and clip from the payload |
| `notebooks/nba/minutes/src/minutes_xgb/calibration.py` | Minute wrapper. Public signatures unchanged |
| `notebooks/nba/points/points_xgboost.ipynb` | Thin orchestration |
| `tests/points_xgb/` | Rate package tests |
| `tests/models/test_conformal.py` | Minute defaults still pass; add a rate clip and rate group case |
| `tests/models/test_bundle_calibration.py` | Minute bundle still refused on a bad tier; rate bundle accepted only with `RATE_Q50_TIERS` |

## Notebook shape

Import `notebooks/nba/points/src` the same way the minutes notebook imports `notebooks/nba/minutes/src`. The first code cell puts that directory on `sys.path`, then imports `points_xgb`.

| Section | Notebook does | Package does |
|---|---|---|
| Header | Keep the current question, metrics, and data contract. Refresh the date. State that conformal is fit on walk-forward OOF and that 2025-26 is scored once | — |
| Setup | Import the package and the shared train, split, and artifact helpers | Config constants |
| Data | `load_regular_season`, print the quality summary | Load and quality |
| Features | `build_model_frame`, assert the locked feature list | Feature build and leak check |
| Train / validation / test | Keep the current design markdown, including the train-pool Tukey fence and the untouched holdout | — |
| Filter and split | Positive minutes, capped target, naive columns, `prepare_splits`, Tukey on the train pool only | Those transforms |
| EDA | Call the EDA helpers and `analyze_correlations` | Tables and plots |
| Models | Commented `tune_xgb_quantile`. Then `run_timeseries_cv` and `run_walk_forward` with `target_col="pts_per_min"` | — |
| Conformal calibration | Display the leakage table, go/no-go table, nested folds 3–5, and offset-stability flags | Rate spec on OOF only |
| Walk-forward score | `score_predictions` on the last fold's raw knots | Scoring |
| Holdout | `evaluate_holdout`, then raw pinball and coverage | — |
| Date-tail, raw beside calibrated | Print the side-by-side tables and the date-clustered calibrated-minus-raw intervals | Fit offsets on all OOF folds, apply to the holdout, score both versions |
| Baseline | `compare_to_baselines` | Baseline report |
| Feature analysis | SHAP, permutation, consensus | Importance |
| Slice analysis | Calendar and role slices on the raw holdout | Slice scores |
| Error analysis | Show the miss frame | Miss frame |
| Save | Metadata, `save_model_bundle(..., calibration=calibration)`, OOS parquet including `cal_q_*` | Metadata, OOS rows, tail sidecar |

Cells stay calls, displays, and the feature-list guard. Metric definitions live in the package.

### Date-tail tail tables

The nested miss-rate display uses folds 3, 4, and 5. Tables for fold k are fit on calibrated folds before k. Tail ratios use `minutes_sampler.tail_miss_rates` on the rate outcome. That function only compares `y` with `q_0.05` and `q_0.95`.

The file written for later loading is `models/saved_models/pts_nba_tails_cal_2026-04-12.joblib`. It uses `ppm_sampler.build_tail_tables` on calibrated OOF rows renamed to the column `rate`, folds `(1, 2, 3, 4)`, so `ppm_sampler.load_tail_sidecar` accepts it. Fold 5 is scored in the notebook and is not part of that file. The uncalibrated sidecar `pts_nba_tails_2026-04-12.joblib` is not written here.

## Data flow

1. Load regular-season gamelogs and flag starters from `start_position` in `{G, F, C}`.
2. `add_points_features`. Refuse the feature list if it contains `minutes`, `pts`, or `predicted_minutes_oof`, or if a selected column is absent.
3. Keep `minutes > 0`. Set `pts_per_min = min(pts / minutes, 6)`. Add `pts_per_min_lag_1`, `season_pts_per_min_mean`, and keep `pts_per_min_ewm_hl_10` as a naive baseline column. Those three are not model features, except `pts_per_min_ewm_hl_10`, which is already in `PTS_FEATURES`.
4. `prepare_splits` holds out `2025-26`. Tukey fences are fit on the train pool only. Rows on the fence stay. The holdout keeps every appearance. The fence sample and the holdout index are disjoint; a test with an extreme holdout rate shows the fence does not move.
5. Time-series CV and five date walk-forward folds use the saved `XGB_PARAMS`. Walk-forward writes `oof["pts_per_min"]`.
6. For each OOF fold, base knots are the sorted booster output. A row with missing `pts_per_min_lag_1` is replaced by that fold's training-role quantiles. The early-stopping tail is part of that history and is not an offset score. Groups come from base q50.
7. `fit_offsets` on folds before k, `apply_offsets` on fold k, for k in `{3, 4, 5}`. The same all-fold offset table is applied to the 2025-26 base knots. Nothing in the offset fit includes a 2025-26 row.
8. The bundle stores that all-fold payload. `save_model_bundle` adds the booster fingerprint, as it does for minutes.

## Error handling

- Non-finite scores, an empty offset sample, or a conformal rank outside `1..n` raise `ValueError`, as they do now.
- A group with fewer than `min_n` rows is omitted and receives a zero offset at apply time.
- A cold-start row whose `starting` is neither 0 nor 1 raises `ValueError`.
- `clip_high < clip_low`, or a non-finite bound, raises `ValueError`. `clip_low` may be 0.
- `points_xgb.calibration.require_preholdout` raises `ValueError` if any OOF `game_date` is on or after 2025-10-21. The rate `fold_base_knots` and rate `nested_offset_backtest` call it. `fit_offsets` itself does not learn that date, so the existing minute unit tests with synthetic dates stay valid.
- `predict_calibrated_quantiles` raises `KeyError` when the payload's lag column is absent.
- A calibration payload whose tiers do not match its target is refused on load.
- Duplicate OOF player-games raise `RuntimeError`, matching the minutes leakage check.
- A walk-forward fold whose validation dates intersect its training dates or its early-stopping tail raises `RuntimeError`.

## Testing

Tests call the public functions. Expected numbers are literals from a tiny hand-built frame, not a reimplementation of the function under test.

- A q50 of 0.30 is `0.30-0.50`. A q50 of 0.50 is `0.50-0.70`. A q50 of 0.70 is `0.70+`. A missing lag is `cold_start` even when q50 is inside a tier.
- A rate offset that would push a knot below 0 or above 6 clips to `[0, 6]`. A knot that is already 0 stays 0.
- The same minute offset case already in `test_conformal.py` still clips to `[0.001, max_minutes]`.
- `require_preholdout` raises on a row dated 2025-10-21. `fold_base_knots` on that frame raises before any offset is fit.
- Tukey fences fit on a train pool are unchanged when a holdout rate of 100 is present.
- `PTS_FEATURES` is exactly `pts_per_min_ewm_hl_10`, `player_fga_share_10`, `ppm_trim_mean_20`, `current_team_pts_per_min`, `usg_wmean_10`, `season_pts_per_min`, `starting`, `fta_per_min_10`, `fga_per_min_10`, `fg3a_per_min_10`, `ppm_unstable`, `opp_def_rating_mean_10`, `min_mean_10`. The leak check rejects `minutes`, `pts`, and `predicted_minutes_oof`.
- `run_walk_forward(..., target_col="pts_per_min")` on 10 one-row dates, `n_folds=1`, and `n_estimators=2` writes `pts_per_min` and does not write `minutes`. The default `target_col` on that same frame writes `minutes`.
- A minute bundle whose stored tiers are not `MINUTE_Q50_TIERS` still fails to load. A rate bundle loads when its tiers are `RATE_Q50_TIERS` and fails when they are the minute tiers.
- `minutes_xgb` calibration tests still pass without edits to their expected minute groups or clip.

## Completion

The work is done when `pytest tests/points_xgb tests/models/test_conformal.py tests/models/test_bundle_calibration.py tests/minutes_xgb/test_calibration.py` passes, and the points notebook's code cells call `points_xgb` instead of defining the scoring, baseline, slice, error, and artifact helpers inline. The full XGBoost walk-forward is not part of that check.
