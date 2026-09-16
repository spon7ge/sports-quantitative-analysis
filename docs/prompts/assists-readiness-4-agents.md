# 4-agent audit: is the stack good enough to start assists?

Read [`docs/context.md`](../context.md) and [`docs/system_architecture.md`](../system_architecture.md) first. Those are the source of truth. This prompt does **not** retrain, promote, or implement assists.

**Question:** Is the frozen minutes + points + joint stack good enough to copy, and can work on an assists **mean model** start now?

**“Start assists” means:** feature EDA + `src/features/assists/` + a mean-model notebook, stacked on existing OOF minutes. It does **not** mean assists pricing, joint coupling, or a production artifact.

**Quality bar (already frozen; do not reopen holdout for selection):** 80% coverage ≈ 0.80, PIT ≈ Uniform(0,1) (mean ~0.5), lower NLL. MAE is secondary. Snapshot EV/CLV is not a gate.

---

## How to run

Launch **four parallel Task agents** (or four Cursor chats) using the paste blocks below. When all four return, run the **merge** paste in a fifth chat.

Do not let agents edit production joblibs, notebooks, or `src/`. Audit only. Cite files and numbers.

`2025-26` is **closed for selection**. Agents may read already-written diagnostics (`location_diagnostics.json`, `joint_variant_preholdout.json`, persisted notebook outputs, `*_meta.json`). They must not fit, tune, or pick a winner on holdout rows.

Known gaps in `context.md` (zero-point PIT, clip uplift, no lineup feed, no git) are **not automatic blockers**. A blocker is something that would make an assists mean model inherit a broken method, leak, or missing exposure driver.

---

## Verdict language (merge only)

| Verdict | Meaning |
|---|---|
| **START** | Minutes exposure is copyable; points pattern is sound enough to clone; silver has assist labels/features. Start EDA this session. |
| **FIX THEN START** | One or more **blockers** in the current stack. List the fixes. Assists EDA waits until those land. |
| **DO NOT START** | The shared method is not trustworthy yet (leakage, missing minutes distribution, no way to stack OOF minutes, etc.). |

Non-blockers (unless an agent finds they actually break stacking): confirmed lineups, integer PTS lines, two-night ROI, deferred joint challengers, frozen clip bias on points, assists coupling.

---

## Agent 1 — Minutes exposure

**Mandate:** Is the frozen minutes mean + distribution a good enough **shared exposure driver** for another count stat?

**Own (read only):**

- `src/features/minutes/`
- `src/models/xgboost_models/minutes.py`
- `src/models/xgboost_models/core.py` (fit / residual / OOF behavior used by minutes)
- `notebooks/nba/minutes/`
- `artifacts/models/minutes/`
- `tests/models/test_minutes_features.py`
- `tests/models/test_minutes_pools.py`
- `docs/superpowers/specs/2026-09-13-minutes-distribution-role-eval-design.md`

**Do not** re-score joint points or hunt assists columns.

**Answer:**

1. Production contract vs artifact vs notebook: do `current37`, meta, and joblib agree?
2. Distribution: bins, pool sizes, role slices. Coverage / PIT / NLL from **existing** notebook outputs or tests — do not retrain.
3. Would stacking `predicted_minutes_oof` into assists be valid with this artifact, or is minutes calibration too broken by role/bin?
4. `start_rate_10` as role: fine to copy for assists training, or a minutes-specific landmine?
5. Verdict for *this slice*: **OK TO STACK** / **FIX FIRST** / **UNKNOWN**, with evidence.

**Return:** ≤ 40 lines. Paths + numbers. No feature-wishlist unless it blocks stacking.

---

## Agent 2 — Points template (mean + leakage)

**Mandate:** Is the points **method** (not the last MAE) a template assists should copy?

**Own (read only):**

- `src/features/points/`
- `src/models/xgboost_models/points.py`
- `notebooks/nba/points/xgboost_model.ipynb`
- `notebooks/nba/points/points_role_eda.ipynb`
- `artifacts/models/points/xgboost_points_meta.json`
- `tests/models/test_points_features.py`
- `tests/models/test_points_pools.py`
- `tests/features/test_pregame_points.py`

**Do not** re-litigate joint \(g\)/\(\beta\)/\(\epsilon\) except where the points mean contract depends on OOF minutes.

**Answer:**

1. Leakage: shift-then-roll, Game N minutes never a feature, pregame blanking includes `ast` already or not.
2. Stacking protocol: how `predicted_minutes_oof` is built and how minutes-dependent features are recomputed. Copyable for assists (`ast_per_min` × hatM, etc.)?
3. Contract: `current41` vs notebook vs joblib. Any silent mismatch assists would repeat?
4. What assists should **not** copy (points-specific scoring features, clip quirks, `pts_std_10` sampler note).
5. Verdict for *this slice*: **COPY** / **COPY WITH CHANGES** / **DO NOT COPY**, plus the changes if any.

**Return:** ≤ 40 lines. Explicit “copy these steps” vs “do not copy this.”

---

## Agent 3 — Joint coupling + infer path

**Mandate:** Is frozen `current` good enough that we should **stop iterating points joint** and treat coupling as a later assists stage?

**Own (read only):**

- `src/models/xgboost_models/joint_calibration.py`
- `src/models/xgboost_models/joint_simulation.py`
- `src/models/xgboost_models/joint_pricing.py`
- `src/models/xgboost_models/joint_variant_eval.py`
- `src/models/xgboost_models/artifact_bundle.py`
- `src/models/xgboost_models/example_set.py`
- `src/models/xgboost_models/snapshot_audit.py`
- `src/models/evaluation.py`
- `src/models/odds.py`
- `src/models/settlement.py`
- `notebooks/nba/points/joint_calibration.ipynb`
- `notebooks/nba/test.ipynb`
- `scripts/`
- `artifacts/models/points/joint_variant_preholdout.json`
- `artifacts/models/points/location_diagnostics.json`
- `artifacts/pricing/`
- `tests/models/test_joint_*.py`
- `tests/models/xgboost_models/`
- `docs/superpowers/specs/2026-09-13-joint-calibration-training-design.md`

**Do not** propose a new joint variant to promote on `2025-26`.

**Answer:**

1. Freeze: does code (`PRODUCTION_JOINT_VARIANT`), preholdout JSON, and `context.md` agree that `current` stays?
2. On **preholdout** evidence, is the distribution bar roughly met (coverage band, PIT, NLL vs `residual_around_p`)? Quote the JSON.
3. Holdout diagnostics: what is already revealed (clipping, zeros, role gap) vs what would be cheating to “fix” now.
4. Infer path: bundle fingerprints, pregame as-of, simulator does not pass minute draws through the points booster, half-point PTS only. Any **broken** plumbing (not a missing feature) that would also break assists later?
5. Is joint coupling a **prerequisite** for starting an assists mean model, or only for later assists distributions/pricing?
6. Verdict for *this slice*: **STOP ITERATING / COUPLING LATER** / **MUST FIX JOINT FIRST** / **MUST FIX INFER PATH FIRST**.

**Return:** ≤ 40 lines. Quote freeze numbers. Separate “mean-model start” from “assists pricing.”

---

## Agent 4 — Assists unblocking

**Mandate:** Ignoring whether points is “done,” what is actually required to **start** assists, and is it present?

**Own (read only):**

- `docs/context.md` (Next: assists)
- `src/pipeline/silver/columns.py`
- `src/features/minutes/player.py` (assist/pass/touch rates already built)
- `src/features/points/pregame.py` (`OUTCOME_COLUMNS`)
- `data/silver/nba/2024-25/regular_season/player_gamelogs.parquet` (schema + null rates for `ast`, `assists`, `ast_pct`, `ast_ratio`, `pass`, `tchs`, `sast`, `ftast`, `team_ast`, `opp_ast`)
- `notebooks/nba/points/points_role_eda.ipynb` (as the EDA template)
- `src/models/xgboost_models/minutes.py` / `points.py` only as **templates**, not quality audits
- Confirm **absence** of `src/features/assists/`, assists notebooks, assists artifacts

**Do not** grade minutes coverage or joint NLL. Take other agents’ quality as out of scope except “OOF minutes join keys exist.”

**Answer:**

1. Silver: target column, coverage, zeros, tracking usefulness (`pass`/`tchs` missingness).
2. Reuse: which minutes/points helpers (rolling, pregame blanking, OOF join on `season_year, game_id, player_id`) already work.
3. Missing code: list only what the first EDA + feature builder needs, not a full joint trainer.
4. Assists-specific modeling risks (low mean, discreteness, teammate collinearity) — flags for EDA, not blockers unless data is unusable.
5. First three concrete steps if START.
6. Verdict for *this slice*: **DATA READY** / **DATA GAP** / **TEMPLATE GAP**.

**Return:** ≤ 40 lines. Include actual null rates / means from parquet if the file exists.

---

## Merge (after all four)

You are the merge agent. Read the four reports plus `docs/context.md`. Do not reopen files unless a report contradicts the docs.

Produce:

### 1. Verdict

One of **START** / **FIX THEN START** / **DO NOT START** in the first line, then 3–6 sentences.

### 2. Scorecard

| Slice | Agent verdict | Blocker for assists *mean* start? | Notes |
|---|---|---|---|
| Minutes exposure | | yes/no | |
| Points template | | yes/no | |
| Joint + infer | | yes/no | |
| Assists data | | yes/no | |

### 3. If START

Ordered next actions (max 5). First action should be the EDA notebook, not coupling.

### 4. If not START

Ordered fixes (max 5). Each must be a blocker from the table, not a wishlist.

### 5. Explicit non-blockers

Bullet the things we will **not** wait on (lineups, joint challengers, PTS clip bias, assists pricing, git, …).

Do not implement. Do not launch assists code in this merge.
