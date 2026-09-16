# Minutes distribution role eval

Date: 2026-09-13  
Approved: notebook-only holdout evaluation of the fitted XGBoost minutes **distribution**, sliced by expected role, actual tip role, and role shock. Not MAE. Not implied points.

## Goal

Minutes are a volume input to points markets (pts/min × minutes). This eval asks whether actual minutes were a believable draw from the pregame distribution, separately for starters and bench. It does not ask whether the point prediction was close, and it does not score points lines.

## Universe

- Notebook: `notebooks/nba/minutes/xgboost_model.ipynb`
- Model: the **already-fitted** `XGBoostMinutesModel` (current 47-feature list, no retrain, no reduced list)
- Rows: 2025-26 holdout `test` (appearances with `minutes > 0`)
- Draws: `model.simulate(test, simulations=2_000)` — 10,000 is too heavy for ~27k rows
- Outcome: same `minutes` / `target_minutes` column used in training

## Role labels

Labels are slices only. Actual starter status is never a feature.

| Label | Rule |
|---|---|
| Expected starter | `start_rate_10 ≥ 0.5` |
| Expected bench | `start_rate_10 < 0.5` |
| Actual starter | non-empty `start_position` (same rule as `_started_obs`) |
| Actual bench | empty / missing `start_position` |
| Role shock | expected starter ≠ actual starter, and both labels exist |

Missing `start_rate_10`: drop from expected-role tables (typical first appearances); keep in the overall row. Missing `start_position`: actual bench.

## Scores

No MAE, RMSE, or R².

| Metric | Meaning | Read |
|---|---|---|
| 80% coverage | actual between 10th and 90th simulated percentiles | should be ~0.80; low = too tight; high = too wide |
| PIT | share of draws strictly below actual, plus half the ties | well-calibrated ⇒ Uniform(0,1), mean ~0.5 |
| NLL | probability the distribution put on the realized minute total | lower is better when comparing groups |

Use `interval_coverage` and `negative_log_likelihood` from `src.models.evaluation`. Compute PIT in the notebook. Skip CRPS: the pairwise implementation cannot run on 26k × 2,000 draws.

PIT reading:

- Mass near 0: over-assigned minutes
- Mass near 1: under-assigned minutes
- U-shape: bands too narrow
- Peak in the middle: bands too wide

## Notebook cells

1. Labels + simulate
2. Metrics table: `n`, coverage_80, PIT mean, PIT std, NLL for all / expected starter / expected bench / actual starter / actual bench / role shock
3. PIT histograms: expected starter vs expected bench (pregame betting view)
4. PIT histograms: actual starter vs actual bench (lineup-known view)
5. Role-shock table: expected starter who sat vs expected bench who started

## Out of scope

Retrain, reduced feature list, implied points (minutes × pts/min), market minutes lines, rank-order of the roster, new `src` module, CRPS on the full holdout.

## Success

Coverage near 0.80 and roughly flat PIT histograms in the expected-starter and expected-bench groups. If bench coverage collapses or role-shock PIT is extreme, the volume module is not yet a trustworthy input to points.
