# MLB strikeout walk-forward

Date: 2026-09-27
Status: approved. Histories use an earlier official date, feature snapshots stay immutable, and scores use the analytical negative binomial.

Scope: workload negative binomial, stored pregame batters-faced offset, and strikeout negative binomial on season-relative 28-day blocks. 2020 locks the shrinkage and rest settings. 2021–2024 is the reported evaluation. 2025 is the holdout.

## Goal

Replace the MLB strikeout training and evaluation path with a chronological walk-forward. The production model remains a negative-binomial GLM. Each start gets a pregame batters-faced prediction from an earlier workload fit, and that prediction is the hard offset in the strikeout model.

Poisson XGBoost and the five quantile trees are a later pass. They are not part of this deliverable.

## Placement

| Piece | Path | Role |
|---|---|---|
| Walk-forward driver | `src/mlb/evaluation/strikeout_walk_forward.py` | Blocks, fits, stored offsets, metrics |
| Feature builder | `src/mlb/pipeline/walk_forward_features.py` | Point-in-time regular-season histories |
| Season calendar | `src/mlb/data/regular_season_calendar.csv` | Official open and close dates, 2018–2025 |
| Tests | `tests/mlb/evaluation/test_strikeout_walk_forward.py` | Temporal integrity on synthetic starts |
| Report | `artifacts/mlb/walk_forward/` | Block rows and aggregate tables |

The driver calls the existing `fit_nb2`. It calls `negative_binomial_pmf` only to persist the `0..15` reporting mass. Scoring uses the analytical NB2 distribution described below. It does not change `config/mlb.yaml` folds, and it does not replace `python -m src.mlb backtest` in this pass. L2 stays at the current config values: workload `1.0`, strikeout `2.0`. Those values are not searched.

## Population

`pitcher_starts` keeps every stored start, including the postseason. A start is model-eligible when its official `game_date` is inside that season’s regular-season window:

```text
regular_season_open_date <= game_date <= regular_season_close_date
```

The calendar sidecar has one row per season:

```text
season, regular_season_open_date, regular_season_close_date
```

Those dates are the first and last completed `gameType="R"` games on the MLB schedule. The sidecar is generated from that rule, reviewed, and checked in. A training run reads the file. It does not call the schedule API. Opening Day is that official open date, not the earliest date present in `pitcher_starts`. A season in 2018–2025 that is missing from the sidecar fails the run. Ineligible starts stay in storage with `model_eligible=false`. They are not targets, and they do not enter rolling windows, season-to-date sums, rest gaps, or league priors.

`game_date` assigns the season, the block, and which earlier starts are visible. A prior start enters a row’s history only when `prior.game_date < current.game_date`. `scheduled_start_utc` orders starts on different dates. It does not prove that an earlier scheduled game has finished, so a same-day result is never used, including a doubleheader and a same-day opponent start. 2026 is absent from this run. The calendar sidecar must have one row per season from 2018 through 2025, unique seasons, parseable dates, and `open <= close`. A violation fails the run before any fit.

## Blocks

The block index resets every season:

```text
block_index = (game_date - regular_season_open_date).days // 28
```

Block 0 is `[open, open + 28 days)`. Block 1 is `[open + 28, open + 56 days)`. The last block ends on the official close date and may be shorter than 28 days. Every start that shares an official `game_date` is in the same block. The block id is `(season, block_index)`.

A block’s training rows are the eligible starts whose `game_date` is strictly before that block’s first `game_date`. Coefficients, standardization, the league strikeout rate, the population workload means, and the first-start rest value come from those rows only. They stay fixed until every row in the block has been predicted.

A row’s lagged sums may include an eligible start from an earlier date in the same block. They may not include a start on the row’s own `game_date`. The workload prediction for that row uses the frozen coefficients and those updated histories. That prediction is the row’s stored offset. Coefficients are not refit inside the block. After the block is scored, its observed batters faced and strikeouts are available to later blocks.

Workload fits begin once the training history reaches the existing minimum, `workload_min_train_starts` (24). Starts before that minimum are not predicted. 2018 is the bootstrap exception: it has no earlier season in this dataset, so an unreported 2018 row may use eligible 2018 starts with an earlier `game_date` as its prior. Those 2018 rows are not scored, are not strikeout training rows, and are not rewritten with a later season’s priors.

## Timeline

| Season | Workload model | Strikeout model |
|---|---|---|
| 2018 | Initialization. Not reported. | No training rows and no score. |
| 2019 | Walk-forward offsets, stored once. | Those rows become the first training history. Not scored. |
| 2020 | Selects `κ` and the rest cap. | Selects `m`. Tuning report only. |
| 2021–2024 | Expanding 28-day blocks. | Reported walk-forward. |
| 2025 | Same blocks, locked settings. | Holdout. No tuning and no feature selection. |
| 2026 | Excluded. | Excluded. |

The workload model and the strikeout model share block ids and row membership wherever both are scored.

## 2020 lock

Grids:

```python
M_GRID = [25, 50, 75, 100, 150]          # prior batters faced
KAPPA_GRID = [1, 2, 3, 4, 6]             # prior starts
REST_CAP_GRID = [14, 21, 28, 35]         # days
```

Stage 1 candidates are pairs `(κ, rest cap)`. Stage 2 candidates are values of `m` under the winning pair. Each stage-1 pair runs an expanding walk-forward from the 2018 warm-up through the 2020 blocks. Training rows and predicted rows for that pair use the same `κ` and rest cap. The selection score uses 2020 validation rows only, pooled across blocks. A short final block does not count as much as a full one.

Stage 1 chooses the pair with the lowest batters-faced MAE on those pooled 2020 rows. MAE is the mean of `abs(predicted_bf - batters_faced)`. A tie breaks toward the smaller `κ`, then the smaller rest cap. Stage 2 applies each candidate `m` to the stored raw strikeout and opponent sums and the stored block league rate. It does not recompute those sums or that rate. It keeps the winning workload offsets and chooses the `m` with the lowest mean strikeout negative log-likelihood on the pooled 2020 strikeout rows. A tie breaks toward the smaller `m`. Discrete CRPS and calibration-in-the-large are written next to the choice. They do not change it. The strikeout rows in that score are 2020 starts whose offsets were stored by the winning workload pair. 2018 starts are not in the strikeout fit.

Candidate runs keep their snapshots in memory. After `m` is chosen, the winning 2019 and 2020 snapshots are written once, with smoothed rates computed from the winning `m`. Later blocks append new rows. A second write for the same `(pitcher_id, game_pk)` raises, including a write of the same value.

After 2020 these stay fixed: `m`, `κ`, the rest cap, the feature definitions, and the L2 penalties. Each later block refits coefficients on the stored historical feature values plus the new block’s own point-in-time rows. The league rate and population means used for a new row come from pre-block history. They do not replace the priors already stored on older rows.

## Workload model

Target: that start’s `batters_faced`. Family: the existing NB2 fit, with dispersion estimated on the training rows. The linear predictor has no exposure offset.

```python
WORKLOAD_FEATURES = [
    "home_flag",
    "days_rest_capped",
    "no_prior_regular_start",
    "long_layoff_flag",
    "season_starts_prior",
    "pitcher_prior_regular_starts",
    "pitcher_bf_mean_last3_smoothed",
    "pitcher_bf_mean_last10_smoothed",
    "pitcher_outs_mean_last5_smoothed",
    "pitcher_pitches_mean_last5_smoothed",
]
```

`home_flag` is the start’s `is_home`.

Every predicted row from 2019 onward stores one immutable snapshot. The key is `(pitcher_id, game_pk)`.

```text
predicted_bf_oof
workload_train_cutoff
walk_forward_block
workload_model_version
league_k_per_bf_prior
population_bf_mean
population_outs_mean
population_pitches_mean
rest_median_capped
```

The snapshot also stores the raw window sums, the history counts, and the workload and strikeout features before standardization. A later fit reads those stored feature values for old rows and standardizes them with the current training block’s centers and scales. It does not recompute an old row’s sums, league prior, population means, rest median, or smoothed rates. A new row uses the priors frozen for its block. Its raw sums still respect `prior.game_date < current.game_date`.

`predicted_bf_oof` is clipped at `1e-6` only when the strikeout model takes its log. The stored value is the model’s prediction before that clip.

## Strikeout model

Target: that start’s `strikeouts`. Family: NB2.

```text
log E[K] = log(predicted_bf_oof) + intercept + X β
```

`predicted_bf_oof` is the value stored for that row when the row was in a workload validation block. The current game’s batters faced, outs, and pitches are not features and are not the offset.

```python
NB_RATE_FEATURES = [
    "home_flag",
    "season_starts_prior",
    "pitcher_prior_regular_starts",
    "pitcher_prior_bf_last10",
    "pitcher_k_per_bf_last3_smoothed",
    "pitcher_k_per_bf_last10_smoothed",
    "pitcher_k_per_bf_season_to_date_smoothed",
    "opponent_k_rate_vs_starters_last10_smoothed",
    "opponent_k_rate_vs_starters_season_to_date_smoothed",
    "opponent_prior_starts_observed",
]
```

K/9 is not in this feature list. The training matrix for both models is the stored point-in-time snapshot. Dispersion `α` is the value estimated by `fit_nb2` for that training block. The parameter-count fallback `α = 1e-4` fails the block. A 2020 candidate that hits it is invalid and is not selected. Its rows are not dropped to produce a partial score. A reported 2021–2024 block or a 2025 holdout block that hits it aborts the run. The method-of-moments estimate is usable and is recorded as such. `α` is refit with the coefficients. It is not one of the frozen 2020 settings.

Continuous columns are standardized with the training block’s mean and standard deviation. The workload model leaves `home_flag`, `no_prior_regular_start`, and `long_layoff_flag` on the 0/1 scale. The strikeout model leaves `home_flag` on the 0/1 scale. The same training centers and scales are applied to the block being predicted. A declared column with zero variance in a training block is omitted from that block’s fit and recorded. It is not median-filled.

## Histories

Rolling windows are built on the eligible table only. A start is eligible for a row when its `game_date` is strictly earlier. `scheduled_start_utc` breaks ties across different dates and supplies the rest gap. It never admits a same-day start. A last-N window uses however many earlier eligible starts exist, including starts from a previous regular season. A window with four prior starts uses those four starts.

Season-to-date sums include only earlier eligible starts from the same season. With no same-season history, the season-to-date rate uses the cross-season smoothed last-10 rate. With no cross-season history either, it uses the league prior. `season_starts_prior` remains 0 in both fallback cases.

Strikeout rates are ratios of prior sums, then shrunk:

```text
smoothed K/BF = (K_prior + m * r0) / (BF_prior + m)
```

`r0` is the league strikeouts-per-batter-faced stored for that row’s block. The same shrinkage is applied to opponent rates. An opponent rate uses earlier-date eligible starts whose `opponent_team_id` equals the upcoming opponent. `opponent_prior_starts_observed` is how many of those starts entered the last-10 window, from 0 to 10. `pitcher_prior_bf_last10` is the raw batters-faced sum in the pitcher’s last-10 window, before `m` is added.

Workload means shrink by start count:

```text
smoothed mean = (sum + κ * μ) / (n + κ)
```

`μ` is the matching population mean stored for that row’s block: batters faced, outs, or pitches. `pitcher_prior_regular_starts` counts every earlier-date eligible start.

Rest uses the previous eligible start on an earlier `game_date`. The gap is the difference in days between the two `scheduled_start_utc` values, `total_seconds / 86400`. When several starts share that previous date, the gap uses the latest `scheduled_start_utc` among them. A pitcher with one has:

```text
days_rest_raw = days between the two scheduled_start_utc values
days_rest_capped = min(days_rest_raw, rest_cap)
long_layoff_flag = 1 when days_rest_raw >= rest_cap
no_prior_regular_start = 0
```

A pitcher with none has `no_prior_regular_start = 1`, `long_layoff_flag = 0`, and `days_rest_capped` equal to the median of the capped rest values on the training rows. The cap in that median is the candidate cap during 2020 and the locked cap afterward. That median is the only imputed value. Partial windows are not median-filled and are not filled with zero.

## Outputs

For each scored strikeout row the driver emits:

```text
predicted_bf
predicted_strikeout_mean
negative_binomial_dispersion
probability mass for K = 0 .. k_max and the existing tail bin
q10, q25, q50, q75, q90
over and under probabilities at 4.5, 5.5, and 6.5
```

The persisted mass is the existing `negative_binomial_pmf` with `k_max = 15`, kept as the reporting format `0..15` plus `P(K > 15)`. Quantiles, negative log-likelihood, discrete CRPS, and over/under probabilities do not read that truncated matrix.

Quantiles use the analytical NB2 CDF. With the same parameterization as `negative_binomial_pmf`, `p = 1 / (1 + α μ)` and `r = 1 / α`, the reported quantile is the smallest integer `k` whose CDF is at least that level. That `k` may be greater than 15. Median MAE uses this `q50`.

Negative log-likelihood uses a stable NB2 log-PMF at the observed count. It is not the log of a probability taken from the truncated reporting matrix.

Discrete CRPS sums `(F(k) - 1{y <= k})^2` from `k = 0` upward until the survival probability above the last included `k` is below `1e-10`. `F` is the analytical CDF.

Interval coverage is the share of actual strikeout counts inside the closed interval. The reported intervals are q25–q75, nominal 50%, and q10–q90, nominal 80%. A closed integer interval often covers more than its nominal probability. That excess is expected and is not, by itself, a miscalibration flag. Over/under probabilities use the analytical CDF and the existing half-point settlement. There is no push on these lines.

## Report

2019 writes workload MAE and mean bias, labeled as offset generation. Bias is the mean of `predicted_bf - batters_faced`.

2020 writes the locked `m`, `κ`, and rest cap, with the 2020 MAE, negative log-likelihood, CRPS, and calibration-in-the-large beside them. It is labeled as tuning.

Each 2021–2024 block, and the aggregate of those blocks, writes:

```text
training period
validation period
training rows
validation rows
workload MAE
workload bias
strikeout negative log-likelihood
discrete CRPS
median MAE
coverage of q25–q75 and of q10–q90
empirical CDF rate at q10, q25, q50, q75, and q90
Brier score and calibration-in-the-large at 4.5, 5.5, and 6.5
```

Calibration-in-the-large at a line is the mean predicted over-probability against the observed over rate. Quantile calibration is the share of outcomes at or below the predicted quantile, reported next to the nominal level. The 2021–2024 aggregate and the 2025 aggregate also write reliability bins `[0.0, 0.1), [0.1, 0.2), …, [0.9, 1.0]`. A bin with fewer than 200 rows is omitted. A single block reports calibration-in-the-large only. 2025 repeats the same block report under the locked settings and is labeled as the holdout.

## Temporal tests

Synthetic starts must show all of the following:

- Every history used for a row has `game_date` strictly before that row’s `game_date`. A same-day start contributes nothing.
- An ineligible start changes no window, rest gap, season count, or league prior.
- The strikeout design matrix has no current-game batters faced, outs, or pitches.
- Each strikeout training offset equals the one stored snapshot for that `(pitcher_id, game_pk)`. A second write raises.
- Each historical training feature equals the value stored when that row was predicted. Rebuilding the row with a later block’s prior is a failure.
- Workload and strikeout scores for a reported block use the same rows.
- League rate, population means, standardization, and the rest median attached to a block use only eligible rows with `game_date` before that block’s first `game_date`.
- No outcome changes its own prediction or a prediction at an earlier cutoff. Outcomes from eligible earlier dates may update row-level histories and workload offsets for later starts, including later starts in the same block. Model coefficients, scaling parameters, league priors, population means, and locked settings remain fixed throughout the block.

## Later pass

After this baseline is frozen:

- Poisson XGBoost with `log(predicted_bf_oof)` as `base_margin`.
- Five quantile models at 0.10, 0.25, 0.50, 0.75, and 0.90, with predicted batters faced as a pregame feature.
- Same rows and blocks as this walk-forward.
- A challenger is promoted only when probability calibration or a distributional score improves. RMSE alone does not promote it.

K/9, `season_year`, `opponent_team_id` as a category, and a free batters-faced elasticity are also later ablations. They are not in the first feature lists.
