# MLB starting-pitcher strikeout MVP

Hobby strikeout distributions for scheduled MLB starters. This stack lives beside the NBA project and does **not** share learners, features, or artifacts with it.

For each starter at a prediction cutoff it emits E[K], P(K=k), an 80% interval, and over/under probabilities. Optional sportsbook quotes are compared as a **quoted-price simulation**, not as realized ROI or a betting edge.

## Installation

Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run MLB tests with the same interpreter:

```bash
.venv/bin/python -m pytest tests/mlb -q
```

## Data-source caveats

- Pitch-level history comes from **Baseball Savant CSV** exports and schedule/lineup snapshots from the **MLB Stats API**.
- There is **no FanGraphs scrape** in this MVP. Chadwick / id-map files are local.
- `ingest-statcast`, `snapshot-schedule`, and `snapshot-lineups` call those public endpoints unless you pass `--fixture`, which reads `tests/mlb/fixtures/raw/` and never uses the network.
- Production ingest sleeps `rate_limit_seconds` (default 1.0s) and sends the configured research User-Agent. Do not hammer public endpoints.
- Tests and `backtest --fixture` use bundled synthetic tables under `tests/mlb/fixtures/`.
- `ingest-gamelogs` pulls **2018–2025** MLB starter box scores (2026 held out for a later Hugging Face quote backtest). `backtest --hf-props` is the 2026 odds eval and is not required to train.

Column-level documentation: [`data_dictionary.yaml`](data_dictionary.yaml). Binding names: [`CONTRACT.md`](CONTRACT.md).

## Commands

Global flags: `--config` (default `config/mlb.yaml`), `--fixture` (load synthetic tables, no network).

```bash
python -m src.mlb --help

python -m src.mlb ingest-statcast --start YYYY-MM-DD --end YYYY-MM-DD
python -m src.mlb ingest-statcast --fixture --start YYYY-MM-DD --end YYYY-MM-DD
python -m src.mlb snapshot-schedule --date YYYY-MM-DD
python -m src.mlb snapshot-lineups --game-pk INT
python -m src.mlb build-features --cutoff ISO8601
python -m src.mlb train-workload
python -m src.mlb train-strikeouts
python -m src.mlb backtest --fixture
python -m src.mlb ingest-gamelogs
python -m src.mlb train-strikeouts
python -m src.mlb ingest-hf-props
python -m src.mlb backtest --hf-props
python -m src.mlb predict --cutoff ISO8601
python -m src.mlb import-quotes --path PATH
python -m src.mlb compare-market --predictions PATH --quotes PATH
python -m src.mlb daily-report --cutoff ISO8601 --out PATH
```

`--fixture` on `predict` trains on the synthetic tables and predicts the last fixture date. `daily-report` writes a CSV of prediction rows and a markdown/text report with expected K and over/under probabilities.

If a sibling pipeline or model module has not landed yet, ingest/train/predict commands raise a clear `Missing module src.mlb...` error.

## Schema pointer

Ordered columns and dtypes live in `src/mlb/schemas.py` (`TABLE_SCHEMAS`). Required tables:

`raw_snapshots`, `game_versions`, `pitch_events`, `plate_appearances`, `pitcher_starts`, `pregame_snapshots`, `feature_rows`, `market_quotes`, `id_map`.

Prediction rows use `PREDICTION_COLUMNS` (PMF keys `pmf_00` … `pmf_15` plus `pmf_tail`).

## Feature definitions (`k_mvp_v1`)

Computed on `pregame_snapshots` rows. Events and ingestions must both precede the cutoff.

- Pitcher K rates (empirical Bayes): `k_bf_shrunk_{60,365,prior2}`, `n_eff_k_bf_{60,365,prior2}`
- Plate discipline (trailing 300 / 750 pitches and 365-day): `{csw,whiff,chase,zone,swing,called_strike}_{300,750,365}`
- Workload history (trailing 3/5/10 starts): `{bf,pitches,outs}_per_start_{3,5,10}`, `pitches_last_start`, `rest_days`
- OOF workload (filled by the modeling layer): `expected_bf_oof`, `bf_sd_oof`, `expected_pitches_oof`, `expected_outs_oof`, `p_early_exit_oof`
- Opponent / lineup: `opp_k_rate_vs_hand_shrunk`, `n_eff_opp_k`, `lineup_k_rate_shrunk`, `lineup_state_code`, `pitcher_throws_L`, `expected_rhb_share`
- Stuff / mix: `fb_velo_{300,365,delta}`, `{ff,bb,os}_share_300`, `{ff,bb,os}_share_delta`
- Role / context: `is_opener`, `is_il_return`, `is_restricted`, `is_home`, `venue_id`, `season`, `rules_era`, `starter_state_code`
- Missingness flags in `MISSINGNESS_FLAGS`; provenance `max_input_event_time_utc`, `max_source_ingestion_time_utc`

## Horizons

`prediction_cutoff_utc = scheduled_start_utc - forecast_horizon_hours` (default 2 hours). Quotes fetched at or after that cutoff are not attached to that forecast. `quote_latency_seconds` further delays quote availability and uses the worst (highest implied) over price among quotes in `(cutoff - latency, cutoff)`.

## Modeling assumptions

- Strikeouts are Negative Binomial NB2: `Var = mu + alpha * mu^2`, `alpha > 0`. Support is `0 … k_max` with a tail bin `P(K >= k_max + 1)` (`k_max = 15`).
- Workload features used by the strikeout model are **out-of-fold**. In-sample fitted BF is never written to `expected_bf_oof`.
- The count model is **not** a BF offset plus independent K|BF noise, and the MVP does not add simulated BF noise on top of the NB.
- Sportsbook lines are not baseball-model features.

## Leakage controls

Cutoff comparisons are strict: `event_time_utc < cutoff` and `ingested_at_utc < cutoff`. No eventual starters, final lineups, realized weather, or later quotes. Folds are chronological expanding windows (`config.folds`); random splits are forbidden. The 2018 test fold does not contain 2017 dates.

## Backtest interpretation

Primary score is mean PMF negative log likelihood. Also reported: discrete CRPS, MAE, RMSE, Brier at 4.5 / 5.5 / 6.5, 80% coverage and mean width, randomized PIT mean.

Market metrics are paired log score and Brier versus **de-vigged** quote probabilities. Simulated expected value is labeled `roi_type='quoted_price_simulation'`. That is forecast quality versus a quoted price, **not profitability** and not realized ROI.

## Known limitations

- Synthetic fixtures are small and are not a stand-in for 2015–present Statcast.
- No umpire, weather, catcher, or batter-specific Poisson-binomial layer.
- Openers, IL returns, and restricted pitchers are flagged, not fully modeled.
- Quote files are user-supplied; books, limits, and voids are not a live feed.
- Integer K lines can push; half-point lines in the default config do not.

## Daily operation

1. Snapshot schedule and lineups for the day; ingest recent Statcast if needed.
2. `build-features --cutoff` at `scheduled start - forecast_horizon_hours`.
3. `predict --cutoff` using the latest artifacts under `artifact_dir` (or `--fixture` to train on synthetics).
4. Optionally `import-quotes` then `compare-market`.
5. `daily-report --cutoff --out PATH` for CSV + markdown.

## Responsible use

This is a **hobby research** project. Nothing here is gambling advice, a recommendation to wager, or a claim that a quoted-price simulation can be captured in the market. Bankroll, limits, latency, and void rules are out of scope.
