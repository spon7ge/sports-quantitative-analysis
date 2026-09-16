# NBA player-props context

Hobby NBA player-props project. The live market is **full-game points**. Minutes exist so points can be a distribution, not a point forecast. Assists is next and should copy this stack, not invent a new one.

**Primary quality bar:** 80% interval coverage ≈ 0.80, PIT ≈ Uniform(0,1) (mean ~0.5, std ~0.29), lower NLL.

**Secondary:** MAE / RMSE.

**Betting:** joint simulation → no-vig EV is downstream. Do not train or promote on CLV, two-night ROI, or priced-edge buckets.

Related architecture: [`system_architecture.md`](system_architecture.md). Frozen design notes live under [`superpowers/specs/`](superpowers/specs/). Readiness audit: [`prompts/assists-readiness-4-agents.md`](prompts/assists-readiness-4-agents.md).

---

## Current freeze (2026-09-15)

| Piece | Production | Status |
|---|---|---|
| Minutes mean | 37 features (`current37`) | Frozen |
| Minutes distribution | Stratified residual bootstrap, bins `[0, 12, 18, 24, 30, 36, 64)` | Frozen |
| Points mean | 41 features (`current41`) | Frozen |
| Joint coupling | `PRODUCTION_JOINT_VARIANT = "current"` | Frozen. Challengers deferred or rejected on preholdout. |
| Holdout | `2025-26` | **Closed** for selection / promotion. Open only for a one-shot final eval. |
| Fitted through | `2019-20`–`2024-25` appearances with `minutes > 0` | |

Do not refit production minutes-mean, minutes-distribution, or points-mean to chase holdout. `2025-26` already revealed the clipping / location hypothesis and cannot independently validate a fix. Next promotion window is `2026-27`.

Role at train and price time is `start_rate_10 >= 0.5`. Confirmed starting fives are **not** in the model. A Rotowire lineup overlay is a later pricing project, not a retraining project.

Assists is not started. There is no `src/features/assists/`, no assists notebook, and no assists artifact. Do not revive the deleted `XGBoostPropModel` stub.

---

## How work actually happens

1. **Data.** Bronze game logs / tracking / positions / Rotowire totals → silver `player_gamelogs.parquet` per season.
2. **Features.** Notebooks and trainers call `add_minutes_features` / `add_points_features`. Shift-then-roll, pregame only. Game N box scores are never features.
3. **Mean models.** Fit in `notebooks/nba/minutes/xgboost_model.ipynb` and `notebooks/nba/points/xgboost_model.ipynb`. Appearance-conditional (`minutes > 0`). Objective `reg:squarederror`. Chronological last 20% of training dates for residual calibration, early stop, then refit.
4. **Minutes distribution.** Same mean model, residual pools stratified by predicted-minute bin. Notebook: `xgboost_distributions.ipynb`.
5. **Points stack.** Expanding-window OOF minute means become `predicted_minutes_oof`. Recompute minutes-dependent points features (`expected_points_rate`, `expected_attempt_volume`, …). Fit the points mean.
6. **Joint calibration.** Offline only. `train_joint_calibration` copies production settings into throwaway fold models, fits overlay \(g\), slope \(\beta(z)\), and scoring-only \(\epsilon\) pools. Does not overwrite the three production mean/distribution artifacts. Notebook: `notebooks/nba/points/joint_calibration.ipynb`.
7. **Pricing / audit.** Load the four-artifact bundle (fingerprint-checked). Pregame rows use history **strictly before** the quote as-of date. Simulate minutes + points, then price half-point PTS over/under. Scripts: `example_set.py`, `snapshot_audit.py`. Quotes live in `data/odds/`.

Holdout scoring for frozen `current` is in `notebooks/nba/test.ipynb`. Location diagnostics: `scripts/diagnose_points_location.py` → `artifacts/models/points/location_diagnostics.json`. Variant freeze record: `scripts/run_joint_variant_preholdout.py` → `artifacts/models/points/joint_variant_preholdout.json`.

---

## Shared training mechanics

Both mean models:

- Universe: silver NBA regular-season appearances, `minutes > 0`.
- Silver seasons on disk: `2019-20` … `2025-26`. Trainers that select models must not read `2025-26` rows.
- `XGBoostConfig` defaults: `n_estimators=750`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=10`, `subsample=0.80`, `colsample_bytree=0.80`, `reg_alpha=0.10`, `reg_lambda=5.0`, `early_stopping_rounds=100`, seed `42`.
- OOF: `expanding_window_splits` (`folds=5`, `minimum_training_dates=60`).
- Uncertainty: residual bootstrap. Minutes pools are stratified; points scoring residuals live on the joint artifact, not on the points mean joblib.
- Predictions clip at 0. Minutes also clip at `maximum_minutes("nba")` (63).

Approximate sizes from the saved metas: ~150k train appearances, ~27k `2025-26` holdout appearances.

---

## Minutes model

- Target: `minutes` | played.
- Production contract: `CURRENT_MINUTES_FEATURES` (37). Named `current37` in `MINUTES_FEATURE_CONTRACTS`.
- Alternate contracts exist (`role_tail`, `lean`, `tier1`) for ablation only. Do not swap production without a preholdout distribution gate.
- Starter vs bench: `expected_role(start_rate_10)` with threshold `0.5`.
- Distribution artifact: `artifacts/models/minutes/xgboost_minutes_distribution.joblib`. Bin edges are part of the joint contract; do not change them without retraining coupling.

---

## Points model

- Target: `pts` | played.
- Production contract: `CURRENT_POINTS_FEATURES` (41). Named `current41`.
- `predicted_minutes_oof` is required. Actual Game N minutes are never a feature.
- Lean (`lean37`) drops `pts_mean_20`, `pts_per_min_20`, `current_team_pts_per_min`, `player_fga_share_10`. Not production.
- Inference stacks the **scalar** minute mean \(\hat M\) into the points booster, then rebuilds `expected_points_rate` and `expected_attempt_volume`. Minute **draws** never enter the booster.

---

## Joint minutes → points

Nightly / snapshot simulation:

\[
P^{(s)} = \max\bigl(0,\; \hat P + g(\cdot) + \beta_b\bigl(M^{(s)} - \bar M^{(s)}\bigr) + \epsilon^{(s)}\bigr)
\]

- \(\hat M, \hat P\): production mean models.
- \(g\): ridge overlay on `minutes_shock` (\(\hat M - \texttt{min_mean_10}\)), `pts_per_min_10`, `usg_wmean_10`, `start_rate_10`.
- \(M^{(s)}\): \(\hat M\) plus a draw from the minutes residual pool for \(\hat M\)'s bin, clipped to `[0, 63]`.
- \(\beta_b\): minutes-bin slope, shrunk, clipped to `[0, 2.5]`.
- \(\epsilon^{(s)}\): scoring-only residual from the \(\mu = \max(0, \hat P + g)\) bin (and role when the pool is hierarchical).
- \(\bar M^{(s)}\): mean of the **drawn** minute vector for that row (simulation shock center), not the full-pool `expected_minutes` used for diagnostics.

`residual_around_p` is a baseline that shows coupling has value. It is not production.

Promotion rule (preholdout only): a challenger must beat `current` and `residual_around_p` on NLL, win most folds, keep 80% coverage in band, and not wreck PIT / role gap / width. Last run (`g_off_shrunk_role`) failed that gate.

---

## Pricing

Supported market today: **full-game PTS**, paired over/under, **half-point lines only**. Integer lines return `UNSUPPORTED_INTEGER_LINE`.

Quotes: `data/odds/NBA_US_*.csv`. Features for a snapshot use box scores strictly before the pull date; the as-of game’s outcomes are blanked.

DNPs void; they are not priced as zeros. Predictions are conditional on appearance.

Two snapshot nights are a pipeline/settlement check, not a bankroll test.

---

## Known gaps (do not “fix” on holdout)

- Zero-point appearances pull PIT down; clipping at 0 lifts low-\(\hat P\) means.
- Minutes error still dominates points error when \(|M - \hat M|\) is large. That is why coupling exists; it is not a reason to pass minute draws through the points trees.
- No confirmed lineup / injury feed in training or pricing.
- No assists / rebounds stack yet.
- No git repo. These docs are the source of truth until one exists.

---

## Next: assists

Copy the points workflow:

1. Feature EDA notebook (see `notebooks/nba/points/points_role_eda.ipynb`).
2. `src/features/assists/` shift-then-roll builder. Stack on OOF minutes. Never Game N minutes.
3. Mean model, `2025-26` closed for selection.
4. Residual distribution after the mean is trustworthy.
5. Minutes → assists coupling last.

Silver already has `ast` / `assists`, `ast_pct`, `ast_ratio`, `pass`, `tchs`. Minutes already builds `assists_per_min_10`, `passes_per_min_10`, `touches_per_min_10`. Assists is a lower-mean, more discrete count than points (~2.5 vs ~10.7).

---

## File map

| Path | Role |
|---|---|
| `src/pipeline/` | Bronze fetch, silver merge, positions, Rotowire totals |
| `src/features/minutes/` | Causal minutes features + contracts |
| `src/features/points/` | Causal points features + contracts + pregame as-of builder |
| `src/models/xgboost_models/` | Mean models, joint train/sim/price, bundle, snapshot audit |
| `src/models/odds.py` | American odds, no-vig, EV |
| `src/models/settlement.py` | DNP void, minute caps |
| `notebooks/nba/minutes/` | Train + distribution + role EDA |
| `notebooks/nba/points/` | Train + joint calibration + role EDA |
| `notebooks/nba/test.ipynb` | Frozen-stack holdout / snapshot grading |
| `scripts/` | One-shot diagnostics that wrote freeze artifacts |
| `artifacts/models/` | joblib + meta + freeze JSON |
| `data/silver/nba/<season>/regular_season/player_gamelogs.parquet` | Model panel |
| `data/odds/` | Book snapshots |

Tests: `tests/` (193 passing after the 2026-09-15 cleanup). Run `.venv/bin/python -m pytest tests -q`.
