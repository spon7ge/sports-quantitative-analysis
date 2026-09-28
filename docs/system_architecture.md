# System architecture

How the NBA points stack is wired. Working rules and freeze status live in [`context.md`](context.md). This file is the layer map: what each box owns, what it may see, and how training differs from a quote snapshot.

```
nba_api / BRef / Rotowire
        │
        ▼
   data/bronze          scrapers + GameLogs fetch
        │
        ▼
   data/silver          one panel row per player-game
        │
        ▼
   features             shift-then-roll, pregame only
        │
        ├── minutes quantile model (11 knots)
        │         │
        │         ▼
        │   minutes sampler (inverse CDF + OOS tails)
        │
        └── rate quantile model (11 knots, points per minute)
                  │
                  ▼
            OOS copula  →  minutes × rate  →  integer points PMF
```

Nothing below silver should read bronze. Nothing in features should use Game N outcomes. A points total is a minutes draw times a rate draw from the copula. Game N minutes are still not a feature.

---

## 1. Ingest

| Layer | Code | Writes | Notes |
|---|---|---|---|
| Bronze fetch | `src/pipeline/bronze/` | `data/bronze/*.parquet` | `player_base`, `player_adv`, `team_base`, `team_adv`, tracking (`start_positions`) |
| Positions | `src/scrapers/basketball_reference_nba_positions.py` | `data/bronze/player_positions/` | Name → position for silver enrichment |
| Rotowire | `src/scrapers/rotowire_nba_odds.py` | `data/bronze/rotowire/` | **Game totals and home lines**, not starting lineups |
| Silver build | `src/pipeline/silver/` (`build.py`, `merge.py`, `columns.py`, `positions.py`, `rotowire.py`) | `data/silver/nba/<season>/regular_season/player_gamelogs.parquet` | CLI facade: `src/pipeline/clean.py` |

Silver is the modeling table: box + advanced + tracking + team/opponent + positions + pregame spread/total. Canonical minutes from `min` / `minutes`. Assists land as `ast` (and tracking `assists`).

WNBA fetch/scrape still exists in the pipeline. There is no WNBA model or notebook.

---

## 2. Features

Two builders, one leakage rule: **every learned or rolling transform is prior-only**.

| Builder | Entry | Depends on |
|---|---|---|
| Minutes | `add_minutes_features` | player role/usage, team/opponent history, schedule, environment |
| Points | `add_points_features` | minutes builder, then scoring, rates, team scoring context, trends |

The lists the quantile models train on live in the notebooks, not in the older manifests:

- Minutes: `MIN_FEATURES` in `notebooks/nba/minutes/min_nba_model.ipynb`
- Rate: the feature list in `pts_nba_model.ipynb` for `min(pts / minutes, 6.0)`

`CURRENT_MINUTES_FEATURES` (37) and `CURRENT_POINTS_FEATURES` (41) are the removed mean-model contracts. `add_points_features` still leaves `predicted_minutes_oof` missing. The points quantile model does not fill it.

### Pregame / as-of inference

`src/features/points/pregame.py`:

1. Keep appearances with `game_date < as_of`.
2. Append tonight’s candidate rows with outcomes blanked (`pts`, `minutes`, `start_position`, team ratings, …).
3. Run `add_points_features` on that concatenated panel so tonight cannot leak.

`starting` in the notebooks is a confirmed start flag from `start_position` in `{G, F, C}`. That flag is known only once a lineup is posted. `start_rate_10` is the trailing share of prior games with a non-empty `start_position` and is available pregame.

---

## 3. Models

The learners are the two quantile notebooks, fit through `models/shared/`. The package `src/models/xgboost_models/` (mean models, residual pools, joint calibration, snapshot pricing) has been removed. Do not recreate it.

| Module | Owns |
|---|---|
| `models/shared/train.py` | Quantile fit, time-series CV, walk-forward, holdout |
| `models/shared/splits.py` | Season holdout and date-blocked walk-forward |
| `models/shared/artifacts.py` | Bundle save/load, `monotonize_quantiles`, `predict_quantiles` |
| `models/shared/minutes_sampler.py` | Minutes quantile function, OOS tails, line probability |
| `models/shared/ppm_sampler.py` | Rate quantile function, OOS tails, cap at 6.0 |
| `models/shared/oos.py` | Joint OOS minutes and rate knots, one row per player-game |
| `models/shared/copula.py` | OOS PIT pairs; `IndependentCopula` and `EmpiricalCopula` |
| `models/shared/simulate.py` | `minutes × rate` → integer points PMF and line probabilities |
| `models/shared/metrics.py` | Pinball, interval coverage, and points-PMF scores |
| `src/models/evaluation.py` | Draw-based NLL, CRPS, coverage |
| `src/models/odds.py` | American → decimal / implied / no-vig / EV |
| `src/models/settlement.py` | DNP voids, regulation vs max minutes |

---

## 4. Artifacts

| File | What |
|---|---|
| `models/saved_models/min_nba_model_2026-04-12.joblib` | Frozen minutes quantile bundle |
| `models/saved_models/min_nba_tails_2026-04-12.joblib` | Frozen minutes tail sidecar |
| `models/saved_models/pts_nba_model_2026-04-12.joblib` | Rate quantile bundle, label `min(pts / minutes, 6.0)` |
| `models/saved_models/pts_nba_tails_2026-04-12.joblib` | Rate tail sidecar, fit on pre-holdout OOS only |
| `data/oos/nba/minutes_rate_oos.parquet` | Walk-forward and 2025-26 knots for both targets |
| `reports/dependence.md` | Pre-holdout PIT-pair dependence by predicted-minutes tier |

Load a bundle with `load_model_bundle` and a sidecar with that sampler's `load_tail_sidecar`. A knot-level mismatch with `QUANTILE_LEVELS` raises. The dependence report describes the stored pairs. It is not a points-PMF calibration.

---

## 5. Train vs infer

### Train (notebooks)

`min_nba_model.ipynb` and `pts_nba_model.ipynb` fit on silver appearances with `minutes > 0`, holding out `2025-26`. Walk-forward folds stay inside the train pool. The minutes model and its tail sidecar are frozen. The points notebook trains `min(pts / minutes, 6.0)` and writes the rate bundle. The same run stores joint OOS knots: walk-forward validation rows for both targets, and holdout rows from the frozen minutes bundle plus the saved rate bundle. Rate tails are fit from pre-holdout OOS only.

### Infer

1. Load both quantile bundles and both tail sidecars.
2. Build pregame features as of the quote timestamp.
3. `predict_quantiles` for minutes and for the rate. Sort each row's 11 knots along tau before a probability is read.
4. A minutes line is `probability_below_line` on the minutes sampler. That is not a points price.
5. Points: `build_pit_pairs` on the OOS table. `EmpiricalCopula` keeps pairs with `game_date` before the quote and drops holdout pairs unless `include_holdout` is set. A tier with fewer than 3,000 pairs draws from every remaining tier. `points_pmf` takes 10,000 seeded draws, inverts both quantile functions, caps the rate at 6, and sets `k = floor(minutes × rate + 0.5)` with `k` clipped at 90. `line_probs` reads the half-point over/under from that PMF. Expected points is the PMF mean.

`IndependentCopula` draws the two uniforms separately. `notebooks/nba/points/pmf_eval.ipynb` scores it as model A against the empirical copula as model B. The priced path is B.

---

## 6. Leakage and settlement boundaries

| Allowed | Not allowed |
|---|---|
| Trailing player/team windows that skip the current row | Game N minutes, points, `start_position`, usage, team ratings as features |
| Rotowire **pregame** spread / total as context | Tonight’s box score via a same-day silver row |
| Realized minutes and points as labels; `pts / minutes` capped at 6 as the rate label; those labels inside OOS tail tables and copula pairs | Using Game N minutes or points as features |
| History strictly before quote `as_of` | Any appearance with `game_date >= as_of` in the feature panel |
| DNP → void | Pricing a DNP as a zero |

Predictions are **conditional on appearance**. The candidate universe is “players with a quote / players who played,” not a full roster with explicit DNP rows. DNP-rate features are omitted until that universe exists.

---

## 7. Evaluation gates

Quantile promotion stays pinball and interval coverage from `models/shared/metrics.py`:

- Pinball at each of the 11 levels
- 80% interval (q0.10–q0.90) coverage near 0.80
- 90% interval (q0.05–q0.95) coverage near 0.90

The integer points PMF is scored in `pmf_eval.ipynb` by `evaluate_points_pmfs`. Dev is the 2024-25 OOS rows, with B's pool rebuilt at each game date from pre-holdout pairs only. Holdout `2025-26` runs once after that, and B's pool is every pre-holdout pair. Reports are overall and by predicted minutes q50 (`<15`, `15-24`, `24-31`, `31+`):

- Randomized PIT of the PMF, P(points = 0), mean log score, and RPS
- Log score and RPS differences (B minus A) with a 95% interval from a bootstrap over game dates
- Pseudo-lines around `floor(EWMA halflife 3) + 0.5`, plus that line ±4 and ±8, floored at 0.5
- MAE of the PMF median and RMSE of the PMF mean against last game, season-to-date mean, and EWMA halflife 3

A flag needs both a tolerance miss (PIT mean ±0.01, 12×variance ±0.05, tail shares ±1 percentage point, rate gaps ±1.5 percentage points) and a gap of more than 2 SE. Each pseudo-line is its own sample of player-games.

q50 MAE against last-game, season mean, and EWMA is diagnostic. It does not select a model.

`2025-26` is a sealed holdout. Tail tables and tuning use preholdout walk-forward rows only.

---

## 8. Tests and runtime

- Tests: `tests/features`, `tests/models`, `tests/pipeline`. No network in unit tests.
- Env: `.venv`, `requirements.txt` (pandas, numpy, pyarrow, xgboost, nba_api, playwright, …).
- There is no scheduler. A quote loads both bundles and both sidecars, then prices points with `points_pmf`.

---

## 9. Out of scope (current architecture)

- A second counting stat on this PMF. Assists (`notebooks/nba/assists/ast_nba_model.ipynb`) would be its own rate model and its own minutes-rate copula
- Confirmed lineup overlay
- Non-PTS markets and period props. A whole-number points line pushes on that integer; the scored market is the half-point line
- Bankroll / Kelly as a training objective
- WNBA modeling
- Production MLOps (serving, monitoring, automated retrains)
