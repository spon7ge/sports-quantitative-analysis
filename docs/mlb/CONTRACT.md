# MLB MVP shared contract

Binding names for the three implementation agents. Do not rename columns or functions.

## Identifiers

- `pitcher_id`, `batter_id`, `player_id`: MLB person IDs (`int64`)
- `game_pk`: MLB game PK (`int64`)
- `team_id`, `venue_id`: MLB IDs (`int64`)
- Datetimes: pandas `datetime64[us, UTC]`, suffix `_utc`
- Cutoff comparisons are **strict**: `event_time_utc < cutoff` AND `ingested_at_utc < cutoff`

## Versions

- `parser_version`: `1.0.0`
- `feature_set_version`: `k_mvp_v1`
- `model_version`: `nb_k_v1`
- `workload_model_version`: `nb_bf_v1`

## Tables

See `src/mlb/schemas.py` for ordered columns and dtypes. Required tables:

`raw_snapshots`, `game_versions`, `pitch_events`, `plate_appearances`,
`pitcher_starts`, `pregame_snapshots`, `feature_rows`, `market_quotes`,
`id_map`.

## Feature columns (`k_mvp_v1`)

Computed on `pregame_snapshots` rows. Workload OOF columns may be filled later by the modeling layer.

Pitcher K rates (empirical Bayes):

- `k_bf_shrunk_60`, `k_bf_shrunk_365`, `k_bf_shrunk_prior2`
- `n_eff_k_bf_60`, `n_eff_k_bf_365`, `n_eff_k_bf_prior2`

Plate discipline (trailing 300 pitches, 750 pitches, 365-day baseline):

- `{csw,whiff,chase,zone,swing,called_strike}_{300,750,365}`

Workload history (trailing 3/5/10 starts):

- `{bf,pitches,outs}_per_start_{3,5,10}`
- `pitches_last_start`, `rest_days`

OOF workload (filled by modeling; pipeline may leave null):

- `expected_bf_oof`, `bf_sd_oof`, `expected_pitches_oof`, `expected_outs_oof`
- `p_early_exit_oof`

Opponent / lineup:

- `opp_k_rate_vs_hand_shrunk`, `n_eff_opp_k`
- `lineup_k_rate_shrunk`, `lineup_state_code`
- `pitcher_throws_L`, `expected_rhb_share`

Stuff / mix:

- `fb_velo_300`, `fb_velo_365`, `fb_velo_delta`
- `ff_share_300`, `bb_share_300`, `os_share_300`
- `ff_share_delta`, `bb_share_delta`, `os_share_delta`

Role / context:

- `is_opener`, `is_il_return`, `is_restricted`, `is_home`
- `venue_id`, `season`, `rules_era`
- `starter_state_code`

Missingness: `missing_{group}` flags listed in `MISSINGNESS_FLAGS`.
Provenance: `max_input_event_time_utc`, `max_source_ingestion_time_utc`.

## Prediction row

See `PREDICTION_COLUMNS` in `schemas.py`. PMF keys are `pmf_00` … `pmf_15` plus `pmf_tail` (P(K>=16) after any support extension is collapsed into tail if k_max=15).

## Agent 1 signatures

```python
def ingest_statcast(config, *, start_date, end_date, http=None) -> pd.DataFrame
def ingest_schedule(config, *, game_date, http=None) -> pd.DataFrame
def ingest_lineups(config, *, game_pk, http=None) -> pd.DataFrame
def ingest_chadwick(config, *, http=None) -> pd.DataFrame
def snapshot_raw(config, source, params, payload, fetched_at) -> str  # snapshot_id
def parse_statcast(raw_payload, snapshot_id, ingested_at) -> tuple[pd.DataFrame, pd.DataFrame]
def parse_schedule(raw_payload, snapshot_id, ingested_at) -> pd.DataFrame  # game_versions
def build_pitcher_starts(plate_appearances, pitch_events, game_versions) -> pd.DataFrame
def build_feature_rows(tables: dict[str, pd.DataFrame], config) -> pd.DataFrame
def assert_no_leakage(feature_rows, tables) -> None
def load_fixture_tables(root: Path | None = None) -> dict[str, pd.DataFrame]
```

`http` is a callable `(url, params) -> bytes` so tests inject fixtures. Default client must sleep `config.rate_limit_seconds` and send a research User-Agent.

Pitch families: `ff` = FF/SI/FC, `bb` = SL/ST/SV/KN/CS, `os` = CH/FS/CU/KC/EP. Other types ignore in shares denominator using classified pitches only.

Switch hitters: `batter_stand` vs LHP is R, vs RHP is L, unless a pitch-level stand is already present.

Rules era: `2015-2019` → `pre2020`, `2020` → `2020`, `2021-2022` → `2021_22`, `2023+` → `pitch_clock`.

## Agent 2 signatures

```python
def shrink_rate(successes, trials, prior_mean, prior_strength) -> float | np.ndarray
def fit_workload(train: pd.DataFrame, config) -> WorkloadModel
def predict_workload(model, frame: pd.DataFrame) -> pd.DataFrame  # expected_bf, bf_sd, expected_pitches, expected_outs, p_early_exit
def add_oof_workload_features(starts: pd.DataFrame, feature_rows: pd.DataFrame, config) -> pd.DataFrame
def fit_strikeouts(train: pd.DataFrame, config) -> StrikeoutModel
def predict_strikeout_pmf(model, frame: pd.DataFrame, config) -> pd.DataFrame  # mu, variance, pmf_*, pi_*, p_over_*
def negative_binomial_pmf(mu, alpha, k_max, tail_threshold) -> np.ndarray  # shape (n, k_max+2) last col tail
def pmf_cdf(pmf) -> np.ndarray
def prop_probabilities(pmf, line: float) -> dict  # p_over, p_under, p_push
def randomized_pit(pmf, y, rng) -> np.ndarray
def discrete_crps(pmf, y) -> np.ndarray
def pmf_nll(pmf, y) -> np.ndarray
def calibration_slope_intercept(pmf, y) -> tuple[float, float]
def fit_pit_recalibration(pmf, y) -> Recalibrator
def apply_recalibration(recal, pmf) -> np.ndarray
def save_model(path, bundle) -> None
def load_model(path) -> dict
```

Negative binomial NB2: `Var = mu + alpha * mu^2`, `alpha > 0`. SciPy mapping:

- `r = 1 / alpha`
- `p = 1 / (1 + alpha * mu)`
- `P(K=k) = nbinom.pmf(k, r, p)`

Do **not** add simulated BF noise on top of the count model. Parameter uncertainty: draw `beta ~ MVN(hat, cov)` (`config.posterior_draws`, seed 42), average PMFs. If covariance is unavailable, skip draws and use the MLE PMF.

Workload OOF: expanding folds by `game_date` with `config.workload_min_train_starts`. Never write in-sample fitted BF into `expected_bf_oof`.

Early-exit threshold: `config.early_exit_bf` (15). `p_early_exit` from a regularized logit on the same OOF schedule.

## Agent 3 signatures

```python
def chronological_folds(dates, config) -> list[tuple[np.ndarray, np.ndarray]]  # train idx, test idx
def fit_baselines(train, test, config) -> dict[str, pd.DataFrame]
def run_backtest(tables, config) -> dict
def american_to_implied(odds) -> float
def decimal_to_implied(odds) -> float
def implied_to_decimal(p) -> float
def no_vig(p_over, p_under) -> tuple[float, float]
def expected_profit(p_win, p_loss, decimal_odds, *, p_push=0.0) -> float
def import_quotes(path, config) -> pd.DataFrame
def compare_market(predictions, quotes, config) -> pd.DataFrame
def write_daily_report(predictions, path) -> Path
def main(argv: list[str] | None = None) -> int
```

CLI subcommands (argparse):

- `ingest-statcast --start YYYY-MM-DD --end YYYY-MM-DD`
- `snapshot-schedule --date YYYY-MM-DD`
- `snapshot-lineups --game-pk INT`
- `build-features --cutoff ISO8601`
- `train-workload`
- `train-strikeouts`
- `backtest`
- `predict --cutoff ISO8601`
- `import-quotes --path PATH`
- `compare-market --predictions PATH --quotes PATH`
- `daily-report --cutoff ISO8601 --out PATH`

`--config` optional, default `config/mlb.yaml`. `--fixture` uses bundled synthetic tables (no network).

## Baselines

Keys: `league_nb`, `rolling_k`, `k9_workload`, `shrunk_kbf`, `pitcher_opp`, `marcel`, `market` (only if quotes exist for that row).

Marcel: 5/4/3 weights on prior-2-season / 365 / 60-day K/BF, regress to league mean with strength `config.pitcher_k_prior_strength`.

## Scoring

Primary: mean PMF NLL. Also discrete CRPS, MAE, RMSE, Brier at each configured line, 80% coverage and mean width, randomized PIT mean, calibration slope/intercept.

Market metrics are paired log score and Brier vs de-vigged quote probabilities. Label ROI as `quoted_price_simulation`.
