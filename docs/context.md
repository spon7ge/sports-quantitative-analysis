# NBA player-props context

Hobby NBA player-props project. The live market is **full-game points**. Minutes exist so points can be a distribution, not a point forecast.

**Primary quality bar:** quantile calibration. Pinball at each trained level, 80% interval coverage near 0.80, 90% coverage near 0.90, and an integer points PMF whose probabilities match outcomes.

**Secondary:** MAE of q50 against last-game, season-to-date mean, and EWMA.

**Betting:** line probabilities come from the points PMF. Do not train or promote on CLV or ROI. Working spec: [`../PROPSIM_SPEC.MD`](../PROPSIM_SPEC.MD). Architecture: [`system_architecture.md`](system_architecture.md).

The mean-plus-residual stack (`src/models/xgboost_models/`, joint overlay / slope / residual pools) has been removed. Do not recreate it.

---

## Current models

| Piece | Where | Status |
|---|---|---|
| Minutes quantiles | `notebooks/nba/minutes/min_nba_model.ipynb` | Frozen. Bundle `models/saved_models/min_nba_model_2026-04-12.joblib` |
| Minutes distribution | `models/shared/minutes_sampler.py` | Frozen tails `min_nba_tails_2026-04-12.joblib` |
| Points-per-minute quantiles | `notebooks/nba/points/pts_nba_model.ipynb` | Notebook still fits `pts` until the retarget. Label will be `min(pts / minutes, 6.0)` |
| Rate distribution, copula, points PMF | `models/shared/ppm_sampler.py`, `copula.py`, `simulate.py` | Not written. Spec is `PROPSIM_SPEC.MD` |
| Holdout | `2025-26` | Closed for selection, tuning, and tail edits |

Both models are `reg:quantileerror` with levels `0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95`. Training goes through `models/shared/`. Appearance filter is `minutes > 0`.

The minutes feature list is `MIN_FEATURES` in the minutes notebook. The rate model's feature list is whatever `pts_nba_model.ipynb` trains after the retarget. `DIRECT_POINTS_FEATURES`, `CURRENT_MINUTES_FEATURES`, and `CURRENT_POINTS_FEATURES` are older contracts.

---

## How work actually happens

1. **Data.** Bronze game logs / tracking / positions / Rotowire totals → silver `player_gamelogs.parquet` per season.
2. **Features.** `add_minutes_features` / `add_points_features`. Shift-then-roll, pregame only. Game N box scores are never features.
3. **Quantile models.** Fit in `min_nba_model.ipynb` and `pts_nba_model.ipynb`. Walk-forward on the pre-holdout pool, then one holdout score. Save with `save_model_bundle`.
4. **Minutes distribution.** Inverse of the 11 knots, with out-of-fold tails by starter/bench. A minutes line is `probability_below_line`, not a draw count.
5. **Points distribution.** Rate sampler, then an OOS copula with minutes, then `points = minutes × rate` rounded to an integer PMF.

Holdout season `2025-26` is not a place to retune. Next promotion window is `2026-27`.

---

## Shared training mechanics

- Universe: silver NBA regular-season appearances, `minutes > 0`.
- Silver seasons: `2019-20` through `2025-26`. Selection must not read `2025-26`.
- Walk-forward: `date_walk_forward_folds` (date blocks, no row from a date split across train and val).
- Quantile crossing is fixed by sorting the row (`monotonize_quantiles`).
- Minutes knots are floored at `1e-3` inside `prepare_quantile_grid`. Draws clip to `[0, maximum_minutes("nba")]` (63). The grid itself is not clipped to 63.

---

## Pricing

Supported market to build: **full-game PTS**, half-point lines, probabilities from the integer PMF.

DNPs void; they are not priced as zeros. Predictions are conditional on appearance.

---

## Known gaps

- The points notebook still trains `pts`. The rate retarget, its retune, and its tail tables are not done.
- No copula or integer points PMF yet.
- No confirmed lineup / injury feed in training.
- Assists has a feature builder and `notebooks/nba/assists/ast_nba_model.ipynb`. It is not part of the points PMF.

---

## File map

| Path | Role |
|---|---|
| `src/pipeline/` | Bronze fetch, silver merge, positions, Rotowire totals |
| `src/features/minutes/` | Causal minutes features |
| `src/features/points/` | Causal points features and pregame as-of builder |
| `models/shared/` | Quantile train, splits, metrics, bundles, minutes sampler |
| `src/models/odds.py` | American odds, no-vig, EV |
| `src/models/settlement.py` | DNP void, minute caps |
| `src/models/evaluation.py` | Draw-based NLL, CRPS, coverage |
| `notebooks/nba/minutes/min_nba_model.ipynb` | Minutes quantile model |
| `notebooks/nba/points/pts_nba_model.ipynb` | Points quantile model |
| `models/saved_models/` | Quantile bundles and minutes tail sidecar |
| `data/silver/nba/<season>/regular_season/player_gamelogs.parquet` | Model panel |
