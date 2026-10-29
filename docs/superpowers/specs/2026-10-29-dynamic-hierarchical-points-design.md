# Dynamic Hierarchical Points Model Design

## Objective

Turn `notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb` into a
standalone, reproducible notebook for evaluating and operating the dynamic
hierarchical NBA points model. The notebook must not depend on executing
`bayesian_models.ipynb` first.

The evaluation output must support a deployment decision. Calibration
diagnostics therefore use chronological out-of-sample predictions rather than
residuals from games used to fit the static hierarchy.

## Boundaries

- Put reusable model and diagnostic logic in a tested Python module under
  `src/models/`.
- Keep data loading, configuration, report assembly, and visualization in the
  notebook.
- Preserve the current model structure: a partially pooled pregame regression
  plus a player latent state updated after each observed game.
- Do not add posterior simulation solely for diagnostic 13. Report that
  draw-based metrics are unavailable when no draws are supplied.
- Do not add market pricing or bet-selection logic in this change.

## Data and Causal Features

Load regular-season player game logs from the configured season parquet files.
Build the same causal predictors used by `bayesian_models.ipynb`:

- previous-game minutes, FGA, and FTA;
- rest days within season;
- opponent defensive rating using only prior opponent games;
- home indicator;
- position, team, opponent, and player identifiers.

Rows with invalid outcomes or required identifiers are rejected. Missing
numeric predictors are imputed from training-window medians, and the trace
records the number imputed. Every filter reports its row count.

Derive `new_team_flag` by comparing the current team with the player's prior
appearance. Derive `rookie_flag` as no prior modeled appearance before the
current season. These definitions are explicit model-history flags, not
official NBA rookie designations.

## Model

The static hierarchy is the conjugate Bayesian regression from
`bayesian_models.ipynb`: shared standardized slopes and partially pooled
position, team, and player effects. The dynamic forecast excludes the static
player coefficient from the base mean and represents player ability with the
latent state instead.

For each player and game:

1. Add per-game process variance after the player's first modeled game.
2. At a season transition, add `OFFSEASON_DRIFT_SD ** 2` once instead of the
   ordinary per-game drift for that transition.
3. Form the pregame prediction from causal features and the state before seeing
   the outcome.
4. Set predictive variance to observation variance plus state variance.
5. Record the forecast and diagnostics.
6. Reveal points and perform the Kalman update.

The default state drift SD remains 0.35 points and offseason drift SD remains
2.0 points. Both are notebook configuration values and are not tuned on the
reported evaluation seasons.

## Walk-Forward Evaluation

Use season-by-season evaluation. For each evaluation season:

1. Fit the static hierarchy on the immediately preceding two complete seasons.
2. Initialize and sequentially filter player states over that same training
   window.
3. Inflate returning-player state variance at the evaluation-season boundary.
4. Predict every eligible evaluation game before updating with its result.

Evaluation seasons are those with two available prior seasons. The notebook
prints the exact train/evaluation seasons and row counts for every fold.
Hyperparameters remain fixed across folds.

The canonical prediction trace contains:

- season, date, game and player identifiers;
- team, opponent, and position;
- points, base mean, predictive mean, innovation, and standardized innovation
  `z`;
- state mean and variance before and after the update;
- observation variance, predictive variance, and Kalman gain;
- games in model before the prediction and game number within season;
- returning-player, new-team, rookie, and imputation flags.

Assertions enforce chronological ordering, one prediction per eligible row,
finite positive variances, and no evaluation outcomes in a fold's fit.

## Diagnostics

### Core calibration

Report overall mean z, SD(z), RMSE, MAE, empirical 50%, 80%, 90%, and 95%
interval coverage, and a PIT histogram under the model's Student-t predictive
distribution. Each metric names the evaluation seasons and sample size.

### Variance decomposition

Bin `games_in_model_before` as 0–9, 10–29, 30–59, and 60+. For each tier,
report sample size, mean state-variance share, mean observation-variance share,
SD(z), and interval coverage. The shares use predictive variance as the
denominator. Compare low-game and high-game coverage without asserting that
equal coverage proves correct uncertainty.

### Shrinkage

By games-played tier, report mean z and mean signed innovation. Separately bin
`state_before` into deciles and report mean innovation. Flag the bottom and top
deciles when the innovation points significantly back toward zero. This
diagnoses persistent under-shrinkage; it does not automatically change the
state mean.

### Offseason behavior

For returning players, compare the first 5, 10, and 20 games within each season
with all later games. Report n, mean z, SD(z), and interval coverage. Also
report the slope, correlation, and uncertainty for early-season innovation
against `state_before`. SD(z) above one suggests insufficient variance
inflation; below one suggests excessive inflation. A significant negative
innovation/state relationship suggests missing offseason mean reversion.

### Group calibration

For opponent, team, and position, report n, mean z, SD(z), their approximate
standard errors, and flags for deviations greater than two standard errors
from targets zero and one. Groups with fewer than 30 observations are shown
but marked low-sample rather than treated as actionable. Include new-team and
rookie slices.

### Gain

Plot mean Kalman gain against games in model using the same games-played tiers
and a finer binned view. Mark season-boundary observations separately. Report
unexpected upward jumps outside season boundaries; offseason jumps are
expected only to the extent implied by the configured variance inflation.

### Posterior draws

If a compatible draws column is supplied, compute empirical PIT and CRPS from
draws and compare them with the analytic Student-t approximation. Otherwise,
print a clear skipped status. No synthetic draws are presented as posterior
draws.

## Interfaces

The reusable module exposes pure or deterministic functions for:

- feature-frame construction;
- hierarchical fitting and prediction-row construction;
- one Kalman update;
- filtering a training window;
- generating one chronological out-of-sample fold;
- assembling all walk-forward folds;
- each diagnostic summary.

Configuration is passed explicitly with a small immutable configuration
object. Functions receive data frames and return data frames or typed result
objects; they do not read project-global notebook variables.

## Failure Handling

Fail loudly for missing required columns, nonpositive variances, duplicate
game/player rows, invalid season ordering, empty folds, and nonfinite outputs.
Optional diagnostics return a structured skipped reason when their optional
inputs are absent. Small groups are retained with warnings rather than silently
dropped.

## Tests and Verification

Write tests before implementation for:

- exact Kalman update arithmetic;
- predictive variance decomposition;
- ordinary drift versus one-time season-boundary inflation;
- prediction-before-update causal ordering;
- no evaluation season in a fold's fit;
- games-played tiers and state deciles;
- returning-player, new-team, and model-rookie flags;
- group-calibration standard-error flags;
- graceful skipping of draw-based diagnostics.

Use a synthetic multi-player, multi-team, multi-season frame so tests do not
depend on local parquet data. After unit tests pass, execute the notebook
end-to-end against the local data and verify that all output tables are finite,
fold dates are chronological, and the live forecast is clearly separated from
walk-forward evaluation.

## Deployment Gate

The notebook informs but does not automatically approve deployment. Before live
use, require:

- no causal-order or leakage assertion failures;
- enough observations in the intended deployment slices;
- interval coverage and SD(z) acceptable overall and by experience tier;
- no unexplained group bias or offseason mean-reversion signal;
- a documented decision on any flagged diagnostic.
