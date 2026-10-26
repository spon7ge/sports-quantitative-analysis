# Points quantile notebook package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the points quantile notebook the same training sequence as `02_minutes_xgboost.ipynb`, with reusable code in `points_xgb` and conformal offsets adapted to rate tiers and the rate cap.

**Architecture:** `models.shared.conformal` grows optional rate arguments and keeps minute defaults. The walk-forward report moves to `models.shared.conformal_report.ConformalSpec`. `minutes_xgb.calibration` stays the minutes adapter. `points_xgb` is the rate adapter and the notebook only calls it.

**Tech Stack:** Python, pandas, numpy, xgboost, pytest. Notebook package under `notebooks/nba/points/src`. Tests under `tests/points_xgb` and `tests/models`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-26-points-xgboost-package-design.md`
- Rate tiers: `cold_start`, `<0.30` (q50 < 0.30), `0.30-0.50` (0.30 ≤ q50 < 0.50), `0.50-0.70` (0.50 ≤ q50 < 0.70), `0.70+` (q50 ≥ 0.70)
- Outcome column `pts_per_min`. Lag `pts_per_min_lag_1`. Clip `[0, 6]`. `min_n` 300
- Minute callers keep `max_minutes`, `min_lag_1`, clip lower `1e-3`, and `MINUTE_Q50_TIERS`
- Do not retrain. Do not edit the frozen conclusion numbers. Do not overwrite `pts_nba_tails_2026-04-12.joblib`
- Calibrated sidecar: `models/saved_models/pts_nba_tails_cal_2026-04-12.joblib`, folds `(1, 2, 3, 4)`
- `PTS_FEATURES` is the 13 names in the spec
- Do not `git commit` unless the user asks (skip every Commit step until then)
- Tests: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest`
- Work in this checkout on branch `feature/points-xgb-package`. Do not create a worktree: the points notebook and minutes package are uncommitted here

## File map

- Modify: `models/shared/metrics.py` — add `RATE_Q50_TIERS`
- Modify: `models/shared/conformal.py` — tiers, `target_col`, `lower`, payload fields
- Create: `models/shared/conformal_report.py` — `ConformalSpec`, calibration table, nested backtest
- Modify: `models/shared/artifacts.py` — tier check by target; predict reads lag and clip
- Modify: `models/shared/train.py` — `run_walk_forward(..., target_col="minutes")`
- Modify: `models/shared/oos.py` — docstring only
- Modify: `notebooks/nba/minutes/src/minutes_xgb/calibration.py` — call the shared report; signatures unchanged
- Create: `notebooks/nba/points/src/points_xgb/` — config, paths, data, features, eda, scoring, baseline, importance, slices, errors, artifacts, calibration
- Modify: `notebooks/nba/points/points_xgboost.ipynb` — thin caller
- Create: `tests/points_xgb/`
- Modify: `tests/models/test_conformal.py`, `tests/models/test_bundle_calibration.py`

---

### Task 1: Rate tiers and grouping

**Files:**
- Modify: `models/shared/metrics.py`
- Modify: `models/shared/conformal.py`
- Test: `tests/models/test_conformal.py`

**Interfaces:**
- Produces: `RATE_Q50_TIERS` and `calibration_group(q50, min_lag_1, tiers=MINUTE_Q50_TIERS)`
- Consumes: nothing

- [ ] **Step 1: Write the failing test**

```python
def test_rate_groups_use_left_closed_cuts_and_a_missing_lag_is_cold_start():
    from models.shared.metrics import RATE_Q50_TIERS

    labels = calibration_group(
        [0.29, 0.30, 0.50, 0.70, 0.10],
        [1.0, 1.0, 1.0, 1.0, np.nan],
        tiers=RATE_Q50_TIERS,
    )
    assert list(labels) == ["<0.30", "0.30-0.50", "0.50-0.70", "0.70+", "cold_start"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_conformal.py::test_rate_groups_use_left_closed_cuts_and_a_missing_lag_is_cold_start -v`
Expected: FAIL because `tiers` is unexpected or `RATE_Q50_TIERS` is missing

- [ ] **Step 3: Implement**

Add `RATE_Q50_TIERS` beside `MINUTE_Q50_TIERS`. `calibration_group` loops over the `tiers` argument instead of the minute constant. Default stays `MINUTE_Q50_TIERS`.

- [ ] **Step 4: Run the minute and rate group tests**

Run: `.venv/bin/python -m pytest tests/models/test_conformal.py -q`
Expected: PASS

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 2: Clip floor

**Files:**
- Modify: `models/shared/conformal.py` (`apply_offsets`)
- Test: `tests/models/test_conformal.py`

**Interfaces:**
- Produces: `apply_offsets(..., max_minutes, *, lower=1e-3)`
- Consumes: Task 1

- [ ] **Step 1: Write the failing test**

A knot of `-1` with offset `0` and `lower=0`, `max_minutes=6` becomes `0`. A knot of `9` becomes `6`. A knot of `0` stays `0`.

- [ ] **Step 2: Run and confirm FAIL** (`lower` unexpected, or clip stays `1e-3`)

- [ ] **Step 3: Implement** `np.clip(..., lower, max_minutes)`. Reject non-finite bounds and `max_minutes < lower`.

- [ ] **Step 4: Existing clip test still expects minimum `1e-3` when `lower` is omitted. Both pass.**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 3: Role quantiles for a named target

**Files:**
- Modify: `models/shared/conformal.py`
- Test: `tests/models/test_conformal.py`

**Interfaces:**
- Produces: `role_quantile_table(history, levels, target_col="minutes")` and `base_knots(..., target_col="minutes")`
- Consumes: Task 1

- [ ] **Step 1: Failing test.** History column `pts_per_min` with starter values `[0.2, 0.6]` yields starter q50 `0.4` when `target_col="pts_per_min"`.

- [ ] **Step 2: Confirm FAIL**

- [ ] **Step 3: Read `target_col` instead of a hardcoded `minutes` column. Default remains `minutes`.**

- [ ] **Step 4: Existing history test and the new rate test pass**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 4: Payload, load check, calibrated predict

**Files:**
- Modify: `models/shared/conformal.py` (`calibration_payload`)
- Modify: `models/shared/artifacts.py`
- Test: `tests/models/test_conformal.py`, `tests/models/test_bundle_calibration.py`

**Interfaces:**
- Produces: payload keys `target`, `lag_col`, `clip_low`, plus existing `max_minutes`. `_check_calibration` accepts `pts_per_min` only with `RATE_Q50_TIERS`. `predict_calibrated_quantiles` selects feature columns and reads `lag_col`, `clip_low`, `clip_high`.
- Consumes: Tasks 1–3

- [ ] **Step 1: Failing tests**

Minute payload from the existing call still has `max_minutes == 63` and now also `target == "minutes"`, `lag_col == "min_lag_1"`, `clip_low == 1e-3`.

Rate bundle with `RATE_Q50_TIERS` loads. The same bundle with minute tiers raises `ValueError` matching `tiers`. A minute bundle with rate tiers still raises.

Predict on a frame whose columns are `starting` plus `pts_per_min_lag_1`, with `feature_names == ["starting"]`, reads the lag and clips to `[0, 6]`.

- [ ] **Step 2: Confirm FAIL**

- [ ] **Step 3: Implement the payload defaults and the loader branch. Missing `target` means minutes.**

- [ ] **Step 4: `pytest tests/models/test_conformal.py tests/models/test_bundle_calibration.py -q` passes**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 5: Shared report and the minutes adapter

**Files:**
- Create: `models/shared/conformal_report.py`
- Modify: `notebooks/nba/minutes/src/minutes_xgb/calibration.py`
- Test: `tests/minutes_xgb/test_calibration.py`

**Interfaces:**
- Produces:
  - `ConformalSpec(target_col, lag_col, tiers, clip_low, clip_high, min_n=300)`
  - `calibration_table(y, dates, base, group, *, spec, starting, n_boot, seed)`
  - `nested_offset_backtest(oof, base, group, spec, *, folds=(3, 4, 5), n_boot, seed)`
  - Minutes wrappers keep `calibration_table(...)` and `nested_offset_backtest(..., max_minutes=)`
- Consumes: Tasks 1–4

- [ ] **Step 1: Move the report body. Replace `oof["minutes"]` with `oof[spec.target_col]`, `_GROUPS` with `("cold_start", *[name for name, _, _ in spec.tiers])`, and `apply_offsets(..., max_minutes)` with `apply_offsets(..., spec.clip_high, lower=spec.clip_low)`.**

- [ ] **Step 2: Minutes wrappers build the minute spec and call the shared functions. `fold_base_knots` stays in `minutes_xgb` and still reads `minutes` and `min_lag_1`.**

- [ ] **Step 3: `pytest tests/minutes_xgb/test_calibration.py tests/models/test_conformal.py -q` passes with no edits to expected minute numbers**

- [ ] **Step 4: Commit** — skip unless the user asks

### Task 6: Walk-forward outcome column

**Files:**
- Modify: `models/shared/train.py`
- Modify: `models/shared/oos.py` docstring
- Test: `tests/models/test_walk_forward_target.py`

**Interfaces:**
- Produces: `run_walk_forward(..., target_col: str = "minutes")` writes `oof[target_col]` only
- Consumes: nothing from Tasks 1–5

- [ ] **Step 1: Failing test.** Ten one-row dates, one numeric feature, `n_folds=1`, `n_estimators=2`, `target_col="pts_per_min"`. OOF has `pts_per_min` and no `minutes` column. A second call with the default has `minutes`.

- [ ] **Step 2: Confirm FAIL** (column is always `minutes`)

- [ ] **Step 3: `fold_oof[target_col] = y_val.to_numpy()`. Reject an empty `target_col`.**

- [ ] **Step 4: Test passes**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 7: Points package scaffold, features, and fence

**Files:**
- Create: `notebooks/nba/points/src/points_xgb/{__init__,config,paths,data,features}.py`
- Create: `tests/points_xgb/conftest.py`
- Create: `tests/points_xgb/test_features.py`

**Interfaces:**
- Produces:
  - `PTS_FEATURES` exactly the 13 spec names
  - `assert_no_leak(features)`
  - `RATE_TIERS` lambdas built from `RATE_Q50_TIERS`
  - `assign_rate_target(frame) -> frame` with `pts_per_min`, `pts_per_min_lag_1`, `season_pts_per_min_mean`
  - `apply_train_tukey(ppm_df, target_col="pts_per_min") -> (frame, low, high, n_dropped)`
- Consumes: `RATE_Q50_TIERS`, `capped_rate`, `tukey_fences`, `prior_shift`, `prior_expanding`

- [ ] **Step 1: Tests for the feature list, leak rejection, tier name parity, and a Tukey fence that ignores a holdout rate of 100**

- [ ] **Step 2: Confirm FAIL**

- [ ] **Step 3: Implement config, paths, data load, and features**

- [ ] **Step 4: `pytest tests/points_xgb/test_features.py -q` passes**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 8: Rate calibration adapter

**Files:**
- Create: `notebooks/nba/points/src/points_xgb/calibration.py`
- Test: `tests/points_xgb/test_calibration.py`

**Interfaces:**
- Produces:
  - `RATE_SPEC: ConformalSpec`
  - `require_preholdout(oof)`
  - `fold_base_knots(oof, ppm_df, train_masks)` using `pts_per_min` and `pts_per_min_lag_1`
  - `nested_offset_backtest(...)` filling `RATE_SPEC`
  - `holdout_calibrated(preds_ho, ppm_holdout, offsets, role_quantiles) -> (base, groups, calibrated)`
  - `calibrated_tail_frame(oof, calibrated_knots) -> frame` with column `rate` and folds 1–4 only
- Consumes: Task 5 spec and Task 7

- [ ] **Step 1: Tests.** Date `2025-10-21` raises. A fold history of rates `[0.2, 0.6]` for a cold starter writes q50 `0.4`. `calibrated_tail_frame` drops fold 5 and names the outcome `rate`.

- [ ] **Step 2: Confirm FAIL**

- [ ] **Step 3: Implement the adapter. Do not import `minutes_xgb`.**

- [ ] **Step 4: `pytest tests/points_xgb/test_calibration.py tests/minutes_xgb/test_calibration.py -q` passes**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 9: Scoring, baselines, slices, errors, EDA, importance, artifacts

**Files:**
- Create: the remaining `points_xgb` modules
- Test: `tests/points_xgb/test_scoring.py`

**Interfaces:**
- Produces the same public names the minutes package uses, with rate wording: `score_predictions`, `quantile_calibration`, `quantile_crossing`, `predicted_rate_tiers`, `interval_sharpness`, `compare_to_baselines`, `slice_score`, `build_miss_frame`, `exploratory_metadata`, `build_oos_predictions`, `write_oos_predictions`
- `build_oos_predictions` writes `cal_q_*` when both calibrated frames are passed, and reads realized minutes from the identity frame
- Consumes: Tasks 7–8

- [ ] **Step 1: A scoring test with `y = [0, 1]` and `q_0.50 = [0, 0]` has pinball `0.5` on the one miss. A crossing test with knots `[1, 0]` reports one crossed row. An OOS frame with two raw rows and two calibrated rows gains `cal_q_0.50`.**

- [ ] **Step 2: Confirm FAIL**

- [ ] **Step 3: Move the points notebook helpers into these modules without changing formulas. Baseline comparators are `pts_per_min_lag_1`, `season_pts_per_min_mean`, and `pts_per_min_ewm_hl_10`.**

- [ ] **Step 4: `pytest tests/points_xgb -q` passes**

- [ ] **Step 5: Commit** — skip unless the user asks

### Task 10: Thin the points notebook

**Files:**
- Modify: `notebooks/nba/points/points_xgboost.ipynb`
- Modify: `notebooks/nba/points/src/points_xgb/__init__.py`

**Interfaces:**
- Consumes: every `points_xgb` export the notebook calls

- [ ] **Step 1: Rewrite cells to the spec's section order.** Setup inserts `notebooks/nba/points/src` on `sys.path`. Models call `run_walk_forward(..., target_col="pts_per_min")`. Conformal and date-tail call the rate adapter. Save passes `calibration=calibration` and writes `data/oos/nba/rate_predictions.parquet` plus `pts_nba_tails_cal_2026-04-12.joblib`.

- [ ] **Step 2: No code cell defines `score_predictions`, `compare_to_baselines` helpers, `slice_score`, or `_iso_date`.**

- [ ] **Step 3: `pytest tests/points_xgb tests/models/test_conformal.py tests/models/test_bundle_calibration.py tests/minutes_xgb/test_calibration.py tests/models/test_walk_forward_target.py -q` passes**

- [ ] **Step 4: Commit** — skip unless the user asks

## Self-review

- Spec coverage: tiers (Task 1), clip (Task 2), role quantiles (Task 3), payload and predict (Task 4), report (Task 5), walk-forward column (Task 6), Tukey and features (Task 7), preholdout and tail folds (Task 8), notebook helpers (Task 9), notebook shape (Task 10).
- Minute numerical defaults stay behind default arguments and the existing minutes tests.
- No full XGBoost walk-forward of the real panel is required for completion.
