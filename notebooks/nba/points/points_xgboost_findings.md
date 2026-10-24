# Points per minute: findings

Source: `notebooks/nba/points/points_xgboost.ipynb`. This is the frozen run in that notebook. Quantile XGBoost, `reg:quantileerror`, depth 3, 1,695 trees, quantiles 0.05 through 0.95. The label is `min(points / minutes, 6.0)` on rows with a positive parsed clock.

The question was whether this model predicts the next game's points per minute more accurately than last game, the season-to-date average, and an EWMA of recent points per minute. Primary metric is q50 pinball. Lower is better.

## Holdout

2025-26, 21 Oct 2025 through 12 Apr 2026, 26,648 appearances. The holdout keeps rates that training deleted.

| Check | Result | Target |
| --- | --- | --- |
| q50 pinball | 0.096 | lower |
| q10–q90 coverage | 79.6% | 80% |
| q05–q95 coverage | 89.5% | 90% |
| Raw quantile crossing | 3.56% (949 rows) | none |

Pinball by quantile: q05 0.019, q10 0.034, q20 0.060, q30 0.078, q40 0.090, q50 0.096, q60 0.095, q70 0.088, q80 0.074, q90 0.050, q95 0.032.

| Slice | n | q50 pinball | 80% coverage |
| --- | --- | --- | --- |
| Starters | 12,300 | 0.079 | 78.1% |
| Bench | 14,348 | 0.109 | 80.9% |
| Predicted q50 under 0.30 | 6,628 | 0.110 | 84.2% |
| 0.30–0.50 | 13,813 | 0.093 | 78.4% |
| 0.50–0.70 | 5,086 | 0.086 | 78.3% |
| 0.70 and up | 1,121 | 0.091 | 72.9% |

Aggregate 80% coverage is close to 80%. The high predicted-rate slice is not: 72.9% of those 1,121 rows fall inside the intended 80% interval. The share of all holdout outcomes at or below q95 is 93.15% against a 95% knot, so more outcomes sit above q95 than the knot implies. Ties, particularly at zero points, affect an exact reading of those shares. A correct quantile only has to satisfy P(Y < q) ≤ τ ≤ P(Y ≤ q).

Outcomes are at or below predicted q50 in 48.2% of appearances.

## Paired baselines

On the 26,066 holdout rows where last-game, season-to-date, and EWMA rates all exist, the comparison uses the same rows for every predictor. Paired q50 pinball is 0.0949. The notebook states that this equals half the q50 MAE, so the paired q50 MAE is 0.1898 at that printed precision. The displayed three-decimal table is:

| Predictor | MAE |
| --- | --- |
| Last-game rate | 0.258 |
| Season-to-date rate | 0.197 |
| EWMA rate | 0.194 |
| Quantile XGBoost q50 | 0.190 |

The unrounded XGBoost-minus-EWMA MAE, printed to three decimals, is −0.004. Its 95% date-clustered bootstrap interval is [−0.006, −0.003]. That interval resamples game dates, with every player-game on a drawn date kept together. It does not capture all dependence across a player's successive games.

The same paired differences versus the other baselines, also printed to three decimals, are −0.068 versus last game, interval [−0.071, −0.066], and −0.008 versus the season-to-date rate, interval [−0.009, −0.006].

Reported WIS on the 11 knots is 0.130. That WIS is not the same functional as the EWMA point-forecast MAE of 0.194. The printed "CRPS" is twice the unweighted mean pinball at those knots. The knots are not equally spaced, so that mean is not a quadrature of the CRPS integral.

The largest absolute gap between a knot and its empirical share of outcomes at or below that knot is 2.0 percentage points.

## Walk-forward

Five date blocks inside the pre-holdout pool, after the Tukey filter. Each block is the next 10% of train-pool dates after a 50% date window. Early stopping uses the last 10% of that fold's training dates, not the scored block.

Mean q50 pinball is 0.088 ± 0.001. Mean 80% coverage is 80.9% ± 0.1%.

| Fold | Window | q50 pinball | 80% coverage |
| --- | --- | --- | --- |
| 1 | 2022-11-02 → 2023-02-07 | 0.086 | 81.1% |
| 2 | 2023-02-07 → 2023-12-05 | 0.088 | 81.1% |
| 3 | 2023-12-05 → 2024-03-17 | 0.088 | 80.8% |
| 4 | 2024-03-17 → 2025-01-02 | 0.089 | 80.9% |
| 5 | 2025-01-02 → 2025-04-11 | 0.088 | 80.9% |

XGBoost had lower reported MAE in all five folds, but fold comparisons were not restricted to identical rows. Model MAE is twice that fold's pinball on every validation row. Each baseline MAE drops rows where that baseline is missing. The paired holdout comparison is stronger evidence.

The holdout pinball of 0.096 is 0.008 above the walk-forward mean. Those two numbers are also on different populations: walk-forward drops 2,032 of 150,062 pre-holdout rates outside the Tukey fence [−0.32, 1.15], and the holdout keeps every appearance. An absolute pinball tolerance of 0.25 is far larger than either loss, so a pass/fail check at that tolerance does not say whether the gap is small. A future acceptance rule should be specified before evaluation and tied to baseline improvement, calibration, and the intended population.

## Rows with no history

104 holdout rows have no prior game. Figures below are the three-decimal table.

| Rule | MAE | Bias | WIS |
| --- | --- | --- | --- |
| Quantile tree | 0.301 | 0.173 | 0.205 |
| Role-quantile fallback | 0.309 | 0.019 | 0.210 |

The fallback's mean residual is smaller. Its MAE and WIS are worse. The saved predictions are the raw trees, not the sorted quantiles and not this fallback. Sorting does not keep q05 from going negative.

## What the features are

Thirteen columns: `pts_per_min_ewm_hl_10`, `player_fga_share_10`, `ppm_trim_mean_20`, `current_team_pts_per_min`, `usg_wmean_10`, `season_pts_per_min`, `starting`, `fta_per_min_10`, `fga_per_min_10`, `fg3a_per_min_10`, `ppm_unstable`, `opp_def_rating_mean_10`, `min_mean_10`.

The rolling rates are prior games only. `starting` is this game's recorded lineup, so it is available only after that lineup exists. Live rows use projected lineups instead.

## Conclusion

In this frozen evaluation, Quantile XGBoost modestly improves median points-per-minute accuracy over EWMA on paired holdout appearances. Aggregate interval coverage is near nominal, but high predicted-rate players have substantial undercoverage. Outcome-based filtering and the mismatch between historical actual lineups and live projected lineups limit interpretation. The run supports further forecasting research; it does not establish a betting edge.
