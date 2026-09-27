# propsim V1 spec

Goal: for every NBA player-game, an integer PMF of points whose out-of-sample
probabilities are calibrated, e.g. P(Over 21.5). Point accuracy is secondary.
There is no odds history to train on, so V1 is judged against outcomes and
against the naive baselines already scored in the minutes and points notebooks.

Two learners. Minutes stays the frozen quantile model. The points notebook
is retargeted from `pts` to points per minute. Do not add a third learner,
and do not revive the removed mean-plus-residual stack
(`XGBoostMinutesModel`, `XGBoostPointsModel`, joint overlay `g` / slope
`β` / residual `ε`).

## Models

Both are XGBoost quantile regressions (`reg:quantileerror`) fit by
`models/shared/train.py` on pregame features. Each row gets these 11
conditional quantiles, in this order:

    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95

`np.linspace(0.05, 0.95, 11)` is a different grid. Do not use it.

Minutes — `notebooks/nba/minutes/min_nba_model.ipynb`

- Target: `minutes`. Rows with `minutes == 0` are dropped.
- Features: the `MIN_FEATURES` list in that notebook, from
  `add_minutes_features`. That list is the trained contract.
  `CURRENT_MINUTES_FEATURES` is an older 37-column manifest and is not
  this model.
- Distribution: `models/shared/minutes_sampler.py`, with tail sidecar
  `models/saved_models/min_nba_tails_2026-04-12.joblib`.
- Bundle: `models/saved_models/min_nba_model_2026-04-12.joblib`.
- Frozen. Do not refit it on the 2025-26 holdout.

Points per minute — `notebooks/nba/points/pts_nba_model.ipynb`

- Target: `min(pts / minutes, 6.0)` on rows with `minutes > 0`. The cap
  is the training label. Short stints stay in the sample. The notebook
  still fits `pts` until this retarget lands. After it lands, that `pts`
  bundle is not the V1 rate model.
- `prepare_splits` names the train frame `ppm_df`. That frame holds
  whatever `target_col` the notebook passes. Set `target_col` to the
  capped rate.
- Features: the list in the notebook after the retarget, from
  `add_points_features`. Pregame only. Game N minutes and Game N points
  are not features. No `predicted_minutes_oof`.
  `DIRECT_POINTS_FEATURES` and `CURRENT_POINTS_FEATURES` are the old
  direct-points contracts.
- The hyperparameters in the notebook are the minutes search, copied as a
  starting point. Retune with `tune_xgb_quantile` on the pre-holdout
  pool only, against the capped rate.
- Bundle stem comes from the notebook, written by `save_model_bundle` to
  `models/saved_models/{stem}_{holdout_end_date}.joblib`.

## Data

Silver player-game panel:

`data/silver/nba/<season>/regular_season/player_gamelogs.parquet`

Seasons on disk: `2019-20` through `2025-26`. Call `add_minutes_features`
or `add_points_features` on that panel. Holdout season is `2025-26`,
split by `models/shared/splits.py`. Tuning, tail tables, and model
selection must not read holdout rows.

## Layout

Stay in the current packages. Do not add a `propsim/` tree.

```
src/features/minutes/                 pregame minutes features
src/features/points/                  pregame points features
models/shared/train.py                walk-forward fit, holdout eval
models/shared/splits.py               season holdout, date walk-forward
models/shared/artifacts.py            bundle save/load, monotonize, predict
models/shared/minutes_sampler.py      minutes quantile function
models/shared/ppm_sampler.py          rate quantile function (not written)
models/shared/copula.py               OOS minutes/rate PIT pairs (not written)
models/shared/simulate.py             product → integer points PMF (not written)
models/shared/metrics.py              pinball, interval coverage
models/saved_models/                  quantile bundles and tail sidecars
notebooks/nba/minutes/min_nba_model.ipynb
notebooks/nba/points/pts_nba_model.ipynb
tests/
```

`ppm_sampler.py` follows the minutes sampler's contract: same 11 levels,
sort along tau, out-of-fold tails, exact inversion. It consumes the rate
notebook's knots. Sampled rates are capped at 6.0. Writing it does not
retune the minutes trees. The rate trees are whatever the retargeted
notebook saves.

## Definitions

- OOS: predicted by a model whose training rows all have `game_date`
  before the scored game. Walk-forward validation windows and the
  2025-26 holdout both qualify. Stored predictions are never revised.
- Minutes tier: bucket the row by predicted minutes q50 with the
  notebook cuts `<15`, `15-24`, `24-31`, `31+`. Never use realized
  minutes.
- Rate model is scored on the capped points-per-minute label. The
  integer points PMF is scored on realized points. Slice that PMF by
  the minutes tiers above. Never slice on realized minutes or realized
  points.
- Naive baselines: last game, season-to-date mean, and EWMA with
  halflife 3. On the rate model those baselines are rates. On the PMF
  they are points.

## Rules

1. As-of only. Every feature, model, tail table, and report uses only
   information available before tip. Walk-forward only. The 2025-26
   holdout is closed for selection, tuning, and tail-table edits.
   The minutes bundle and its tail sidecar stay frozen.

2. Drop rows with 0 minutes everywhere. Books void those bets. Keep
   injury exits and short stints. Do not drop rate-model rows for any
   other minutes cutoff. The rate label is `min(pts / minutes, 6.0)`.

3. Sort each row's 11 predictions along the tau axis before a
   probability is read (`monotonize_quantiles`, or
   `prepare_quantile_grid` for minutes). Minutes knots are then floored
   at `1e-3`. Do not clamp a query outside q05/q95 by interpolating the
   11 knots alone. Each map interpolates only for `u` in `[0.05, 0.95]`.
   Below 0.05 and above 0.95 it uses that target's out-of-fold tail
   tables. After the rate map, cap the rate at 6.0.

4. Copula pairs are the OOS probability-integral-transform pair
   `(F_minutes(minutes), F_rate(rate))`. `F` is the CDF of the same
   quantile function used for sampling, tails included. A flat piece
   uses a randomized PIT. Store the pairs raw. Do not rank-transform
   them. A new row draws one stored pair and pushes each uniform
   through that row's own quantile function.

5. One integer PMF per player-game. A draw is
   `X = minutes_draw * rate_draw`, then `k = floor(X + 0.5)`. Every
   points probability, mean, PIT, and score comes from that PMF.
   Expected points is the PMF mean, never q50 minutes times q50 rate.
   Half-point line probabilities are the PMF mass on the matching
   integers. Minutes lines stay on the minutes quantile function
   (`probability_below_line`). They are not a points price.

6. Seed randomness per `(game_id, player_id)` with a stable hash
   (`hashlib`), so reruns are identical. The minutes sampler already
   does this for minute draws. When the quantile function can be
   inverted exactly, the inverse is the price and the draws are only a
   check that they match it.

Style: numpy, pandas, scipy, xgboost; small plain functions; pytest
tests for every module that is added.
