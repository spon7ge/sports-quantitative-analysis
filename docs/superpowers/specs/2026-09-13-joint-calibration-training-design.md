# Joint minutes → points calibration (offline training)

Date: 2026-09-13  
Approved: coupling-only training stage. One `src` trainer, one diagnostics notebook, one additive artifact. Production minutes-mean, minutes-distribution, and points-mean models are not refit. Nightly prediction is out of scope until this training path exists.

## Goal

Fit a joint calibration that treats the saved points XGBoost as the marginal scoring anchor and the saved minutes distribution as the future exposure driver.

Training produces honest pre-holdout OOF minute and point means, a role-shock overlay \(g\), a minutes-to-points slope \(\beta(z)\), and scoring-only residual pools \(\epsilon\). Nightly simulation (later) will draw minutes from the existing distribution artifact and apply

\[
P^{(s)}
=
\max\bigl(
0,\;
\hat P + g(\cdot)
+
\beta_b\bigl(M^{(s)}-\mathbb{E}[M^{(s)}]\bigr)
+
\epsilon^{(s)}
\bigr)
\]

without passing minute **draws** through the points booster and without adding raw points residuals (those still contain unexpected-minutes variance).

\(g\) itself may be negative. Only the corrected mean \(\hat\mu_i = \max(0, \hat P_i + g_i)\) is clipped.

## Paths

| Path | This spec |
|---|---|
| Offline training | In scope |
| Nightly prediction | Out of scope; loads this artifact later plus existing minutes-distribution and points-mean artifacts |

## Universe

- League: NBA regular season silver gamelogs.
- Seasons loaded for features: `2019-20` through `2024-25`.
- Rows: appearances with `minutes > 0`.
- **`2025-26` is closed.** Trainer and notebook assert that no input row has `season_year == "2025-26"`. That season is opened once, later, for final evaluation.

Two residual universes:

| Universe | Rule |
|---|---|
| `base_oof_eligible` | Finite `minutes_hat` and `points_hat` |
| `coupling_oof_eligible` | `base_oof_eligible` **and** finite cross-fitted `g_hat` and `beta_hat` |

The first **base** OOF fold is the earliest block with both hats. It has no earlier eligible residuals, so it cannot train overlay or \(\beta\). Label it `coupling_warmup`. For coupling fold \(j \ge 2\), fit overlay then \(\beta\) on prior `base_oof_eligible` rows and predict fold \(j\).

Build \(\epsilon\) pools only from `coupling_oof_eligible`. Warm-up and other holes are dropped, not imputed.

Using realized Game N minutes to form \(u_i\) is valid: it is an observed outcome for residual calibration, not a predictive feature.

## Inputs (not produced here)

| Artifact | Role |
|---|---|
| `artifacts/models/minutes/xgboost_minutes.joblib` | Production minutes **mean** settings (feature list, clip, XGBoost config) copied into fold models |
| `artifacts/models/minutes/xgboost_minutes_distribution.joblib` | Bin **edges** for \(\beta(z)\); pools themselves are not rebuilt |
| `artifacts/models/points/xgboost_points.joblib` | Production points **mean** settings copied into fold models |

Fingerprint **all three** with a content hash. Paths, tree counts, and feature lists are metadata, not identity. Load **fails closed** if any hash does not match the file on disk.

Fold models use the exact production feature lists, clipping rules, and XGBoost hyperparameters. They are throwaway OOF instruments; they do not overwrite the artifacts above.

Minutes residual pools by predicted-minutes bin stay on the distribution artifact. This job only needs OOF minute **means** to form \(u_i\).

## Training order

1. Build causal pregame features (`add_points_features`) for `2019-20`–`2024-25`. Shift-then-roll features may be computed once on the panel. Every **learned** transform is fold-scoped.
2. Expanding-window OOF minute means \(\hat M_i\).
3. Insert \(\hat M_i\) as `predicted_minutes_oof` and recompute every minutes-dependent points feature (`expected_points_rate`, `expected_attempt_volume`, …).
4. Expanding-window OOF point means \(\hat P_i\) on those features (same date-fold cuts as minutes).
5. Align on `appearance_key_columns = (season_year, game_id, player_id)`:

\[
u_i = M_i - \hat M_i, \qquad v_i = P_i - \hat P_i
\]

   `base_oof_eligible` is the set with finite hats.

6. Nested OOF overlay \(\hat g_i\): for coupling fold \(j \ge 2\), fit \(g\) on prior `base_oof_eligible` rows (pregame inputs only) and predict fold \(j\). Fold 1 is `coupling_warmup` (`missing_overlay_hat`).
7. Nested OOF \(\beta\): on the same coupling folds, after that fold’s \(\hat g\), fit bin-specific \(\beta\) on prior rows’ \(v - \hat g\) and apply to fold \(j\). Fold 1 is `missing_beta_hat`.

\[
v_i - \hat g_i = \beta_{b(i)} u_i + \epsilon_i
\]

8. Cross-fitted scoring-only residuals on `coupling_oof_eligible` only:

\[
\epsilon_i = P_i - (\hat P_i + \hat g_i) - \beta_{b(i)} u_i
\]

9. Refit **final** overlay on all `base_oof_eligible` rows. Refit **final** \(\beta\) on residuals from that **final** overlay (\(v - g_{\text{final}}\)), not from cross-fitted \(\hat g\). Keep cross-fitted \(\epsilon_i\) as the simulation pools. Do not replace pools with in-sample residuals from the final refits.
10. Save `artifacts/models/points/joint_calibration.joblib`.

Overlay is fit **before** \(\beta\) and \(\epsilon\) on every coupling fold. Otherwise \(\epsilon\) pools retain systematic role-shock error that \(g\) later removes, and nightly draws are miscentered or too wide.

## OOF folds

- Shared date-fold cuts for minutes OOF and points OOF. Independent splitters are not allowed. Coupling folds reuse those same valid-date blocks: block 1 = `coupling_warmup`; blocks \(j \ge 2\) are coupling folds.
- Generator: existing `expanding_window_splits` (`folds=5`, `minimum_training_dates=60`).
- Persist the **realized** fold date boundaries and per-fold train/valid row counts.
- Persist `base_oof_max_training_date` (max date used to train any minutes/points fold model) and `final_calibration_max_date` (max `game_date` among rows used to fit final \(g\) and \(\beta\)). Do not store a single ambiguous `training_max_game_date`.
- Random seed default `42`.
- Earliest dates with no base hats: exclusion `warmup`. First base fold with hats but no nested \(g\)/\(\beta\): `coupling_warmup` / `missing_overlay_hat` / `missing_beta_hat`.

## Overlay \(g\)

Pregame features, fixed order:

1. `minutes_shock` = \(\hat M_i - \texttt{min_mean_10}\)
2. `pts_per_min_10`
3. `usg_wmean_10`
4. `start_rate_10`

Target: \(v_i = P_i - \hat P_i\).

Fold-scoped preprocessing: median impute, then standardize with that fold’s mean/std (store the same statistics on the final overlay for nightly). Ridge penalty `alpha = 1.0` on standardized features. Bound **only** `minutes_shock` at ≥ 0 (bounded least squares). Other coefficients may be negative. Do not clip-after without refitting intercept and remaining coefficients with the clipped shock coefficient held fixed.

\(g\) is unconstrained in sign. Clip only when forming \(\hat\mu_i = \max(0, \hat P_i + g_i)\).

Store complete preprocessing with the overlay: feature order, imputation, scaling, transforms, coefficient constraints, and the \(\hat\mu\) clip rule.

## \(\beta(z)\)

- \(z\): predicted-minutes bin using **exact** edges from the minutes-distribution artifact (currently `[0, 12, 18, 24, 30, 36, 64)`).
- Through-origin OLS per bin on \((u, v-g)\), shrunk toward the global through-origin slope with shrinkage \(k = 20\): \(\beta_b^\star = \frac{n_b}{n_b+k}\beta_b + \frac{k}{n_b+k}\beta_{\text{global}}\). Clip \(\beta^\star\) to \([0, 2.5]\).
- Cross-fitted \(\beta_{b(i)}\) (from prior base-OOF rows, using that fold’s \(\hat g\)) is what enters \(\epsilon_i\).
- Final \(\beta\) (nightly) is fit on all `base_oof_eligible` rows against \(v - g_{\text{final}}\).

## Scoring-only \(\epsilon\) pools

`coupling_oof_eligible` only. Stratify by the overlay-corrected OOF mean

\[
\hat\mu_i = \max(0, \hat P_i + \hat g_i)
\]

not raw \(\hat P_i\). Edges: `[0, 8, 14, 20, 28, inf)` so the last bin is exhaustive.

Construction:

1. Winsorize \(\epsilon\) at documented quantiles (same spirit as minutes: keep rotation shocks, drop data errors; default 0.5% / 99.5%).
2. Require a minimum pool size (default 40). Empty/thin bins fall back to the concatenated eligible \(\epsilon\).
3. Center each pool so \(\mathbb{E}[\epsilon \mid \text{bin}] = 0\) if \(\hat\mu\) is the intended marginal mean. Store the **removed pool means** and **raw (pre-center) standard deviations** for diagnostics.

These pools are cross-fitted. Final overlay/\(\beta\) refits do not rebuild them.

## Artifact: `artifacts/models/points/joint_calibration.joblib`

Additive. Nightly (later) loads this file plus the minutes-distribution and points-mean artifacts.

The OOF panel is **not** stored. Field name is `appearance_key_columns`; appearance keys themselves are absent.

Required payload:

- `schema_version`
- `holdout_season` = `"2025-26"`
- `fitted_through` = `"2024-25"`
- `base_oof_max_training_date`
- `final_calibration_max_date`
- `random_seed`
- `appearance_key_columns`
- Fold records: date boundaries, train/valid row counts per **base** fold; coupling-fold flags (`coupling_warmup` vs fitted)
- `minutes_bins`, final `beta_global` / `beta_by_bin`, shrinkage, plus **cross-fitted** \(\beta\) metadata used for \(\epsilon\)
- Overlay: final nightly model **and** a record that \(\epsilon\) used cross-fitted \(g\); full preprocessing contract
- `epsilon_pools`, `epsilon_bins`, winsorization, centering flag, removed means, raw stds, min-pool-size, fallback rule, `epsilon_source = "cross_fitted"`
- `n_preholdout_appearances`
- `n_base_oof_eligible`
- `n_coupling_oof_eligible`
- `oof_coverage` = `n_coupling_oof_eligible / n_preholdout_appearances`
- `exclusions`: counts by reason (`warmup`, `coupling_warmup`, `missing_minutes_hat`, `missing_points_hat`, `missing_overlay_hat`, `missing_beta_hat`, `invalid_feature`)
- Compatible **minutes-mean**, **minutes-distribution**, and **points-mean** content hashes, plus path / feature list / `training_cutoff` / trees as metadata

Load **fails closed** if any of the three content hashes does not match the file on disk.

## Code layout

- Trainer: `src/models/xgboost_models/joint_calibration.py`
- Tests: `tests/models/test_joint_calibration.py`
- Notebook: `notebooks/nba/points/joint_calibration.ipynb`

The notebook calls the trainer. It does not reimplement OOF or pooling.

## Notebook diagnostics

Both trainer and notebook assert no `2025-26` rows.

Overlay-on vs overlay-off comparisons use **cross-fitted** \(g_i\) on `coupling_oof_eligible` only. Report residual mean, MAE, 80% residual width, and the same metrics by expected role-shock bucket (expected starter vs actual starter, same labels as the minutes role eval). Width alone is not success if bias worsens.

**Gate (not literal zero deterioration):** a slice counts only if `n >= 200`. Overlay-on passes if MAE ≤ overlay-off MAE + `0.05` points **and** |residual mean| ≤ overlay-off |residual mean| + `0.05` points, both overall and in every eligible role-shock slice.

Also show: \(\beta\) by bin, overlay coefficients, \(\epsilon\) pool sizes / removed means / raw stds, exclusion counts, `n_preholdout_appearances`, `n_base_oof_eligible`, `n_coupling_oof_eligible`, `oof_coverage`.

## Out of scope

- Nightly prediction path
- Four-variant 2025-26 benchmark
- Rebuilding or overwriting minutes residual pools
- Passing simulated minute draws through the current points XGBoost
- Dumping the OOF panel beside the joblib
- A fully conditional \(P \mid M\) model (separate later revision of the “Game N minutes are never features” contract)

## Success

- `base_oof_eligible` rows have finite \(u, v\); `coupling_oof_eligible` rows have finite \(g, \beta, \epsilon\)
- First base fold is excluded from \(\epsilon\) pools (`coupling_warmup`)
- Final \(g\) and \(\beta\) are stored for nightly; final \(\beta\) uses \(v - g_{\text{final}}\); \(\epsilon\) pools are cross-fitted, centered, and built only from `coupling_oof_eligible`
- Load fails if minutes-mean, minutes-distribution, or points-mean hashes drift
- `2025-26` never appears in trainer inputs
- Overlay-on OOF meets the MAE/bias tolerance gate on slices with \(n \ge 200\)
