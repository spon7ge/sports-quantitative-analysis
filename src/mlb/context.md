# MLB strikeout model

Starting-pitcher strikeout distributions. Versions in `src/mlb/__init__.py` and `config/mlb.yaml`:

| Piece | Version | Code |
| --- | --- | --- |
| Feature set | `k_mvp_v1` | `pipeline/gamelog_features.py`, `pipeline/features.py` |
| Workload | `nb_bf_v1` | `models/workload.py` |
| Strikeouts | `nb_k_v1` | `models/strikeouts.py` |

Config defaults live in `config/mlb.yaml`. The 2026 quote backtest uses `config/mlb_hf_2026.yaml`. Column contracts are in `schemas.py`. A longer operator guide is `docs/mlb/README.md`.

The live card (`notebooks/mlb/live.ipynb`) scores today's slate with the saved `nb_k_v1` artifact. Lineup slots and the Log5 walk in that notebook are a sketch. They are not inputs to the GLM.

## How a prediction is made

Two models, both negative binomial NB2 (`Var = μ + α μ²`).

1. **Workload (`nb_bf_v1`)** predicts expected batters faced from lagged start history, plus a logit for an early exit (fewer than `early_exit_bf` batters, default 15). Pitches and outs are that BF mean times the training-fold pitches-per-BF and outs-per-BF ratios.
2. **Strikeouts (`nb_k_v1`)** predict K given that expected BF. With `strikeout_free_bf_coef: false` (the default), batters faced is an offset with coefficient fixed at 1:

```
log E[K] = log(predicted_bf_oof) + intercept + β · x
```

`predicted_bf_oof` is an out-of-fold workload prediction. Realized Game N `batters_faced` is never the offset. If the OOF value is missing, the offset falls back through `expected_bf_oof`, then `bf_mean_5` / `bf_per_start_5`, then a constant 22, clipped to [1, 60].

The mean is turned into a probability mass function on K = 0 … `k_max` (15) plus a tail bin `P(K > 15)`. If the coefficient covariance is usable, the PMF is the average of `posterior_draws` (64) coefficient draws; otherwise it is the point estimate. From that PMF the code writes `expected_k`, `variance_k`, an 80% interval (`pi_lower`, `pi_upper`), and over/under probabilities at the configured lines (4.5, 5.5, 6.5). Any other line is `P(K > line)` from the same PMF, which is what the live notebook does.

Continuous features are standardized with the training-fold mean and standard deviation. Binary features are left as 0/1. At fit time, columns that are all missing, constant, or collinear with the columns already kept are dropped. L2 penalties: workload 1.0, strikeouts 2.0. The intercept is not penalized.

Workload predictions that enter the strikeout model are out of fold. `add_oof_workload_features` refits on history strictly before each game date: every 7 days until 500 starts, then every 28 days. It will not fit until `workload_min_train_starts` (24).

## Features the strikeout model uses

Declared in `STRIKEOUT_FEATURE_COLUMNS`. These are the only regressors in `nb_k_v1`, plus the BF offset above.

| Feature | Meaning |
| --- | --- |
| `k_bf_shrunk_365` | Pitcher K/BF over the prior 365 days, shrunk to the league |
| `k_bf_shrunk_60` | Same, prior 60 days |
| `rest_days_capped` | Days since previous start, capped at 14 |
| `extended_rest` | 1 if rest is 7–14 days |
| `long_absence` | 1 if rest is more than 14 days |
| `first_start_or_missing_history` | 1 if there is no previous start |
| `pitcher_throws_L` | 1 if the pitcher throws left |
| `is_home` | 1 if the start is at home |

Shrinkage is `(strikeouts + m · k) / (BF + k)` with prior strength `k = pitcher_k_prior_strength` (100). `m` is the expanding league K/BF from starts strictly before this one (fallback 0.22). Zero trials returns the prior. On the game-log path, the windows are built in `build_gamelog_feature_rows` and exclude the current start.

Rest flags are mutually exclusive and come from `encode_rest_features`. Missing rest sets the missing-history flag and zeros the other rest flags. `standard_rest` (4–6 days) is encoded but is not a strikeout regressor.

`k_bf_shrunk_prior2` (previous two seasons, same shrinkage) is computed on the feature row and used by the Marcel-style baseline. It is not in `STRIKEOUT_FEATURE_COLUMNS`.

## Features the workload model uses

Declared in `WORKLOAD_FEATURE_COLUMNS`. All of these are lagged: rolling windows are `shift(1)` before the roll, so the start being predicted is not in its own history.

| Feature | Meaning |
| --- | --- |
| `bf_mean_5` | Mean batters faced over the previous 5 starts |
| `bf_sd_5` | SD of batters faced over the previous 5 starts |
| `early_exit_rate_5` | Share of the previous 5 starts with BF under `early_exit_bf` |
| `pitches_per_start_5` | Mean pitches over the previous 5 starts |
| `outs_per_start_5` | Mean outs over the previous 5 starts |
| `pitches_per_bf_5` | Mean pitches per batter over the previous 5 starts |
| `rest_days_capped`, `extended_rest`, `long_absence`, `first_start_or_missing_history` | Same rest encoding as above |
| `is_opener`, `is_restricted`, `is_il_return` | Role flags. The game-log builder sets these to 0 |
| `is_home` | Home start |

The game-log builder also stores 3- and 10-start means (`bf_per_start_{3,10}`, and the same for pitches and outs) plus `pitches_last_start`. Those are on the feature row. The workload GLM only sees the 5-start block and the flags above.

## What is computed and not in the GLM

`FEATURE_VALUE_COLUMNS` is the full feature-row schema. The Statcast builder (`build_feature_rows`) fills plate-discipline rates (CSW, whiff, chase, zone, swing, called strike at 300 pitches, 750 pitches, and 365 days), pitch mix and fastball velocity, opponent and lineup K rates, and role flags. The strikeout GLM does not take those columns.

The path that actually trains and scores today is the game-log path. `_model_feature_rows` uses `build_gamelog_feature_rows` whenever `pitch_events` is empty, which is the stored 2018–2025 starter logs and the live notebook. On that path the Statcast columns are empty and `missing_plate_discipline`, `missing_opponent`, `missing_lineup`, and `missing_stuff` are 1. Lineup slot rates (`lineup_slots`, frozen K/PA by batting-order slot) are a separate table for the notebook sketch. They are not strikeout features.

Sportsbook prices are never model features. Quotes are attached after the PMF exists.

## Data the current fit is built from

`ingest-gamelogs` pulls MLB Stats API starter box scores for seasons 2018–2025 (`TRAIN_SEASONS` in `pipeline/gamelogs.py`). 2026 is held out. Each start becomes a `pitcher_starts` row and a `pregame_snapshots` row whose cutoff is `scheduled_start_utc - forecast_horizon_hours` (default 2 hours).

Feature rows are one row per pitcher-game. History for a row is strictly earlier starts for that pitcher. League shrinkage uses only starts with an earlier event time.

## How to use it

From the repo root, with the project venv:

```bash
python -m src.mlb --help
```

Train on cached 2018–2025 starter logs (network only if the cache is missing):

```bash
python -m src.mlb ingest-gamelogs
python -m src.mlb train-workload
python -m src.mlb train-strikeouts
```

Artifacts land in `artifacts/mlb/workload/nb_bf_v1.pkl` and `artifacts/mlb/strikeouts/nb_k_v1.pkl`.

Chronological backtest on the stored tables, using the folds in `config/mlb.yaml`:

```bash
python -m src.mlb backtest
```

2026 Hugging Face quote backtest (separate folds in `config/mlb_hf_2026.yaml`, selected inside the command):

```bash
python -m src.mlb ingest-hf-props
python -m src.mlb backtest --hf-props
```

Predict at a cutoff. If a strikeout artifact exists, `predict` loads the newest one under `artifact_dir`. Otherwise it refits on rows strictly before the cutoff.

```bash
python -m src.mlb predict --cutoff 2026-09-23T21:00:00Z
```

Predictions are written to `artifacts/mlb/predictions.parquet`. Columns are `PREDICTION_COLUMNS`: identity, `expected_bf`, `expected_k`, `variance_k`, `pi_lower`, `pi_upper`, `pmf_00` … `pmf_15`, `pmf_tail`, and `p_over_*` / `p_under_*` for 4.5, 5.5, and 6.5.

Quotes and a daily card:

```bash
python -m src.mlb import-quotes --path path/to/quotes.csv
python -m src.mlb compare-market --predictions artifacts/mlb/predictions.parquet --quotes path/to/quotes.csv
python -m src.mlb daily-report --cutoff 2026-09-23T21:00:00Z --out artifacts/mlb/daily
```

`--fixture` swaps in `tests/mlb/fixtures` and does not hit the network. `backtest --fixture` is the synthetic smoke test. `predict --fixture` refits on the fixture instead of loading the saved model.

### Live slate

`notebooks/mlb/live.ipynb` is the daily card:

1. Load `artifacts/mlb/strikeouts/nb_k_v1.pkl`.
2. Pull the latest strikeout quotes (Supabase by default, or local JSON when `ODDS_SOURCE = "json"`).
3. Append today's probable starters onto the 2018–2025 game logs with K and BF set to 0 so they cannot leak into their own features.
4. Build game-log features and OOF workload, then `predict_strikeout_pmf`.
5. Price the book's main line from the PMF and print expected value versus the American odds.

Set `FETCH = False` to stay on local tables. The Log5 `walk_expected_k` cell is labeled not wired. It does not change `nb_k_v1`.

### Grading and backtest notebooks

- `notebooks/mlb/backtest.ipynb` scores a cached quote window. It does not refit and does not overwrite the 2018–2025 store.
- `notebooks/mlb/grade.ipynb` grades closing lines against box-score strikeouts.
- `notebooks/mlb/strikeout_metrics.ipynb` inspects the saved `nb_k_v1` coefficients.

## Leakage rules

- A feature for a start uses only events and ingestions before that start's cutoff.
- Workload features on the strikeout model are out-of-fold.
- Folds are chronological expanding windows. Random splits are not used.
- Eventual starters, final lineups, and later quotes are not backfilled into the feature row.
