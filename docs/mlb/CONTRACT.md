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
`id_map`, `batter_pas`, `lineup_slots`.

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
- `bf_mean_5`, `bf_sd_5`, `early_exit_rate_5`, `pitches_per_bf_5`
- `rest_days_capped`, `standard_rest`, `extended_rest`, `long_absence`, `first_start_or_missing_history`

OOF workload (filled by modeling; pipeline may leave null):

- `expected_bf_oof`, `predicted_bf_oof`, `bf_sd_oof`, `expected_pitches_oof`, `expected_outs_oof`
- `p_early_exit_oof`

`predicted_bf_oof` is the chronological `nb_bf_v1` prediction and is an alias of `expected_bf_oof`. Strikeouts use `log(predicted_bf_oof)` as an exposure offset, never realized Game N BF.

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

## Lineup slots and batter PBP (public API)

Constants and URLs:

```python
LIVE_FEED_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
PBP_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
BOXSCORE_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

STRIKEOUT_EVENT_TYPES = frozenset({
    "strikeout",
    "strikeout_double_play",
    "strikeout_triple_play",
})

KPA_WINDOWS_DAYS = (60, 365)
KPA_PRIOR_SEASONS = 2
BOXSCORE_00_LEAD = pd.Timedelta(hours=24)
RATE_VERSION_PREFIX = "kpa_"
PITCHER_POSITIONS = frozenset({"P"})
```

Functions:

```python
def is_strikeout(event_type: str, *, events: frozenset[str] = STRIKEOUT_EVENT_TYPES) -> int:
    """Derived at read. Never stored on batter_pas."""

def parse_play_by_play(
    raw_payload: str | bytes,
    snapshot_id: str,
    ingested_at: datetime | pd.Timestamp,
) -> pd.DataFrame: ...

def ingest_play_by_play(
    config: MlbConfig,
    *,
    game_pks: list[int],
    http: HttpFn | None = None,
) -> pd.DataFrame: ...

def parse_starting_nine(raw_payload: str | bytes, game_pk: int | None = None) -> pd.DataFrame:
    """00-filter 1–9. Not teams.*.battingOrder."""

def freeze_lineup_slot_rates(
    slots: pd.DataFrame,
    batter_pas: pd.DataFrame,
    people: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame: ...

def ingest_lineup_slots(
    config: MlbConfig,
    *,
    game_pks: list[int],
    provenance: str,  # "live_feed" | "boxscore_00"
    http: HttpFn | None = None,
) -> pd.DataFrame: ...

def shrink_batter_k_pa(
    pas: pd.DataFrame,
    *,
    batter_id: int,
    opposing_pitcher_hand: str,
    bats: str,
    cutoff: pd.Timestamp,
    league: LeagueKPa,
    config: MlbConfig,
) -> dict[str, dict[str, float]]:
    """Keyed by window name '60' | '365' | 'prior2'.
    Each value has k_pa_vs_hand_shrunk, k_pa_overall_shrunk,
    pa_vs_hand, pa_all."""

def league_platoon_odds_ratio(
    *,
    bats: str,
    league_k_pa_cell: float,
    league_k_pa_bats: float,
) -> float:
    """1.0 only when bats is missing or empty.
    For L/R/S, callers pass cells computed inside that bats value."""

def rate_version(config: MlbConfig) -> str: ...

def lineup_identity_mismatch_rate(live: pd.DataFrame, official: pd.DataFrame) -> float:
    """Join on (game_pk, team_id, slot). Identities only. No K, no quotes."""

def write_lineup_coverage(slots: pd.DataFrame, skips: pd.DataFrame, *, season: int) -> dict: ...

def assert_lineup_coverage(coverage: dict, *, season: int) -> None:
    """Fail on zero announced rows or complete-nine rate below floor."""
```

CLI (lineup slice):

```text
python -m src.mlb ingest-play-by-play --start-season 2018 --end-season 2025
python -m src.mlb ingest-lineup-slots --start-season 2018 --end-season 2025
python -m src.mlb snapshot-lineups --game-pk INT
```

`snapshot-lineups` writes `lineup_slots` with `provenance=live_feed` through `parse_starting_nine`. `--fixture` reads `tests/mlb/fixtures/raw/` and never uses the network.

**Not wired this slice:** `STRIKEOUT_FEATURE_COLUMNS`, `fit_strikeouts`, and `nb_k_v1` unchanged. Game-log `pregame_from_starts` keeps `lineup_state = "team_fallback"` and `lineup_batter_ids_json = "[]"`.

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

Workload history used by `nb_bf_v1` (lagged only):

- `bf_mean_5`, `bf_sd_5`, `early_exit_rate_5`
- `{pitches,outs}_per_start_5`, `pitches_per_bf_5`
- `rest_days_capped`, `extended_rest`, `long_absence`, `first_start_or_missing_history`
- `is_opener`, `is_restricted`, `is_il_return`, `is_home`

Strikeout NB2: `Var = mu + alpha * mu^2`, `alpha > 0`. SciPy mapping:

- `r = 1 / alpha`
- `p = 1 / (1 + alpha * mu)`
- `P(K=k) = nbinom.pmf(k, r, p)`

\[
\log E[K] = \log(\widehat{BF}_{\text{oof}}) + X\beta
\]

`X` is pitcher skill and context (`k_bf_shrunk_*`, rest/absence flags, hand, home). The offset is `log(predicted_bf_oof)`, never realized Game N BF. Do **not** add simulated BF noise on top of the count model. Parameter uncertainty: draw `beta ~ MVN(hat, cov)` (`config.posterior_draws`, seed 42), average PMFs. If covariance is unavailable, skip draws and use the MLE PMF.

Workload OOF: expanding folds by `game_date` with `config.workload_min_train_starts`. Never write in-sample fitted BF into `expected_bf_oof` / `predicted_bf_oof`.

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
- `ingest-play-by-play --start-season INT --end-season INT`
- `ingest-lineup-slots --start-season INT --end-season INT`
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
