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
        ├── minutes mean ──────────────┐
        │         │                    │
        │         ▼                    │
        │   minutes residual pools     │
        │         │                    │
        │         ▼                    │
        └── points mean ◄── OOF hatM ──┤
                  │                    │
                  ▼                    │
            joint calibration          │
            (g, β, ε)                  │
                  │                    │
                  ▼                    │
         four-artifact bundle ◄────────┘
                  │
                  ▼
         JointPointsSimulator
                  │
                  ▼
         joint pricing  →  snapshot audit
```

Nothing below silver should read bronze. Nothing in features should use Game N outcomes. Nothing in simulation should pass minute **draws** into the points booster.

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

Contracts (the lists the boosters actually use) live in:

- `src/features/minutes/columns.py` — production `CURRENT_MINUTES_FEATURES` (37)
- `src/features/points/columns.py` — production `CURRENT_POINTS_FEATURES` (41)

`add_points_features` leaves `predicted_minutes_oof` missing. Trainers fill it with chronological OOF minute means, then `add_stacked_interactions` rebuilds volume terms.

### Pregame / as-of inference

`src/features/points/pregame.py`:

1. Keep appearances with `game_date < as_of`.
2. Append tonight’s candidate rows with outcomes blanked (`pts`, `minutes`, `start_position`, team ratings, …).
3. Run `add_points_features` on that concatenated panel so tonight cannot leak.

Candidate construction (name → `player_id`, last team, schedule) is `src/models/xgboost_models/example_set.py`.

Role is not a lineup feed. `start_rate_10` is the trailing share of prior games with a non-empty `start_position`. `expected_role` maps `>= 0.5` to starter.

---

## 3. Models

All production learners are under `src/models/xgboost_models/`. Shared fit/simulate: `core.py` (`CalibratedXGBoostRegressor`, `XGBoostConfig`, expanding-window OOF).

| Module | Owns |
|---|---|
| `minutes.py` | Mean minutes, stratified residual pools, role helper, feature-contract names |
| `points.py` | Mean points, OOF minute/point hats, points residual helpers |
| `joint_calibration.py` | Offline \(g\), \(\beta\), \(\epsilon\); variant freeze; fail-closed if `2025-26` is in the train frame |
| `joint_variant_eval.py` | Preholdout challenger scoring and promotion gate |
| `joint_simulation.py` | Appearance-conditional draws from the saved bundle |
| `joint_pricing.py` | Half-point full-game PTS over/under from one draw cloud |
| `artifact_bundle.py` | Load four joblibs, content-hash fingerprints, schema / pool checks |
| `example_set.py` | Snapshot quote → candidate rows → priced slate |
| `snapshot_audit.py` | Pre-tip quote audit (pipeline check, not a profitability test) |
| `evaluation.py` | Coverage, PIT, NLL, role slices |
| `tuning.py` | Expanding-window split helper |

Supporting, not learners:

- `src/models/odds.py` — American → decimal / implied / no-vig / EV
- `src/models/settlement.py` — DNP voids, regulation vs max minutes

Deleted (do not resurrect): baseline Empirical Bayes, Bayesian props, stub `XGBoostPropModel`.

---

## 4. Artifacts

Production bundle, all fingerprint-checked together:

| File | What |
|---|---|
| `artifacts/models/minutes/xgboost_minutes.joblib` | Minutes mean |
| `artifacts/models/minutes/xgboost_minutes_distribution.joblib` | Same mean + stratified residual pools + bin edges |
| `artifacts/models/points/xgboost_points.joblib` | Points mean |
| `artifacts/models/points/joint_calibration.joblib` | Overlay, \(\beta\), \(\epsilon\), hashes of the three inputs |

Sidecars: `*_meta.json`, `joint_variant_preholdout.json`, `location_diagnostics.json`.

Load path: `load_joint_points_bundle(...)`. Mismatched hash, schema, bins, or empty pools → `ArtifactIncompatibilityError` / `NO_QUOTE`. Fail closed.

`joblib` files are gitignored (`artifacts/models/` in `.gitignore`). They still exist on disk and are required to price.

---

## 5. Train vs infer

### Train (notebooks)

Minutes and points notebooks fit on silver through `2024-25` (holdout season held out of the **saved** model). Joint calibration:

1. Copy production hyperparameters / feature lists into fold models.
2. Expanding-window OOF \(\hat M\), then \(\hat P\) on stacked features.
3. Nested OOF \(g\) then \(\beta\) (first base fold is `coupling_warmup`).
4. \(\epsilon\) only from rows with hats **and** cross-fitted \(g\), \(\beta\).
5. Refit final \(g\) / \(\beta\) on all `base_oof_eligible`. Keep cross-fitted \(\epsilon\) pools.
6. Write `joint_calibration.joblib` only.

Fold models are throwaway. They must not overwrite the three production joblibs.

### Infer (snapshot / holdout replay)

1. Load the four-artifact bundle.
2. Build pregame features as of the quote timestamp.
3. `JointPointsSimulator`:
   - \(\hat M\) from minutes mean, clipped `[0, 63]`
   - \(\hat P\) from points mean with **scalar** \(\hat M\) stacked in
   - \(g(\hat M, \text{pregame})\)
   - minute draws from the distribution artifact
   - points draws: \(\max(0, \hat P + g + \beta (M^{(s)} - \bar M^{(s)}) + \epsilon^{(s)})\)
4. `price_joint_points_market` maps that cloud onto a half-point line.

Holdout replay (`notebooks/nba/test.ipynb`) uses realized appearance rows, not book names. Snapshot pricing uses `data/odds/` quotes mapped onto `player_id`.

---

## 6. Leakage and settlement boundaries

| Allowed | Not allowed |
|---|---|
| Trailing player/team windows that skip the current row | Game N minutes, points, `start_position`, usage, team ratings as features |
| Rotowire **pregame** spread / total as context | Tonight’s box score via a same-day silver row |
| Realized Game N minutes inside **calibration residuals** \(u_i = M_i - \hat M_i\) | Passing those realized minutes (or draws) through the points trees at infer |
| History strictly before quote `as_of` | Any appearance with `game_date >= as_of` in the feature panel |
| DNP → void | Pricing a DNP as a zero |

Predictions are **conditional on appearance**. The candidate universe is “players with a quote / players who played,” not a full roster with explicit DNP rows. DNP-rate features are omitted until that universe exists.

---

## 7. Evaluation gates

Distribution metrics (`src/models/evaluation.py`) are the promotion language:

- 80% coverage target 0.80 (tight band 0.78–0.82)
- PIT mean ~0.50
- NLL (lower better; only comparable on the same frame and draw count)
- Starter vs bench coverage gap

Mean MAE is diagnostic. Snapshot edge/EV tables are diagnostic. Neither selects a model.

`2025-26` is a sealed holdout. Variant selection uses preholdout OOF folds only.

---

## 8. Tests and runtime

- Tests: `tests/features`, `tests/models`, `tests/pipeline`. No network in unit tests.
- Env: `.venv`, `requirements.txt` (pandas, numpy, pyarrow, xgboost, nba_api, playwright, …).
- There is no git remote and no scheduler. “Nightly” in older specs means the infer path that a later job would call; today that path is snapshot scripts plus notebooks.

---

## 9. Out of scope (current architecture)

- Assists / rebounds models
- Confirmed lineup overlay
- Integer lines, non-PTS markets, period props
- Bankroll / Kelly as a training objective
- WNBA modeling
- Production MLOps (serving, monitoring, automated retrains)
