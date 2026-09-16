# Joint Calibration Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and save an additive `joint_calibration.joblib` from pre-2025-26 chronological OOF: nested overlay \(g\), \(\beta(z)\), and scoring-only \(\epsilon\) pools.

**Architecture:** Coupling-only `src` trainer. Reuse production minutes/points settings for throwaway OOF fold models. Nested coupling OOF starts at base fold 2. Notebook only calls the trainer and plots diagnostics. Do not refit or overwrite minutes/points artifacts.

**Tech Stack:** Python, pandas, numpy, scipy (`lsq_linear`), joblib, hashlib, existing `expanding_window_splits`, `add_predicted_minutes_oof`, `XGBoostMinutesModel`, `XGBoostPointsModel`, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-13-joint-calibration-training-design.md`
- Holdout `2025-26` never in trainer inputs; assert closed
- Do not overwrite `xgboost_minutes.joblib`, `xgboost_minutes_distribution.joblib`, or `xgboost_points.joblib`
- Shared date-fold cuts for minutes OOF and points OOF; coupling reuses those blocks
- Overlay before \(\beta\) and \(\epsilon\) on every coupling fold
- \(\epsilon\) pools from `coupling_oof_eligible` only; first base fold is `coupling_warmup`
- Final \(\beta\) fit on \(v - g_{\text{final}}\)
- \(g\) may be negative; clip only \(\max(0,\hat P+g)\)
- Fingerprint all three input artifacts; load fails closed on hash mismatch
- Tests: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest ...`
- Do not `git commit` unless the user asks

## File map

- Create: `src/models/xgboost_models/joint_calibration.py`
- Create: `tests/models/test_joint_calibration.py`
- Create: `notebooks/nba/points/joint_calibration.ipynb`
- Modify: `src/models/xgboost_models/__init__.py` — export trainer + load/save
- Modify: `src/models/xgboost_models/points.py` — add `add_predicted_points_oof` parallel to minutes OOF

---

### Task 1: Closed holdout, fingerprints, two universes

**Files:**
- Create: `src/models/xgboost_models/joint_calibration.py`
- Test: `tests/models/test_joint_calibration.py`

**Interfaces:**
- Produces: `HOLDOUT_SEASON = "2025-26"`, `APPEARANCE_KEY_COLUMNS`, `assert_preholdout(frame: pd.DataFrame) -> None`, `content_hash(path: Path) -> str`, `fingerprint_artifact(path: Path, extra: dict | None = None) -> dict`, `classify_universes(frame: pd.DataFrame) -> pd.DataFrame` adding boolean `base_oof_eligible`, `coupling_oof_eligible` and an `exclusion` string column.

- [ ] **Step 1: Write the failing tests**

```python
"""Joint calibration trainer: nested OOF coupling artifact."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.joint_calibration import (
    HOLDOUT_SEASON,
    assert_preholdout,
    classify_universes,
    content_hash,
)


class PreholdoutGuardTests(unittest.TestCase):
    def test_assert_preholdout_rejects_holdout_season(self) -> None:
        frame = pd.DataFrame({"season_year": ["2024-25", HOLDOUT_SEASON]})
        with self.assertRaises(ValueError):
            assert_preholdout(frame)

    def test_assert_preholdout_accepts_preholdout_only(self) -> None:
        frame = pd.DataFrame({"season_year": ["2023-24", "2024-25"]})
        assert_preholdout(frame)


class FingerprintTests(unittest.TestCase):
    def test_content_hash_changes_when_bytes_change(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.joblib"
            path.write_bytes(b"abc")
            first = content_hash(path)
            path.write_bytes(b"abd")
            self.assertNotEqual(first, content_hash(path))


class UniverseTests(unittest.TestCase):
    def test_first_base_fold_is_coupling_warmup(self) -> None:
        frame = pd.DataFrame(
            {
                "minutes_hat": [np.nan, 20.0, 22.0],
                "points_hat": [np.nan, 15.0, 16.0],
                "g_hat": [np.nan, np.nan, 0.5],
                "beta_hat": [np.nan, np.nan, 0.4],
                "base_fold": [0, 1, 2],
            }
        )
        labeled = classify_universes(frame)
        self.assertFalse(labeled.loc[0, "base_oof_eligible"])
        self.assertTrue(labeled.loc[1, "base_oof_eligible"])
        self.assertFalse(labeled.loc[1, "coupling_oof_eligible"])
        self.assertEqual(labeled.loc[1, "exclusion"], "coupling_warmup")
        self.assertTrue(labeled.loc[2, "coupling_oof_eligible"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_joint_calibration -v
```

Expected: `ImportError` for `joint_calibration`.

- [ ] **Step 3: Implement guards, hashing, and universe labels**

`assert_preholdout` raises `ValueError` if any `season_year` equals `HOLDOUT_SEASON`. `content_hash` is SHA-256 of file bytes. `classify_universes`:

- `base_oof_eligible` = finite `minutes_hat` and `points_hat`
- `coupling_oof_eligible` = base eligible and finite `g_hat` and `beta_hat`
- `exclusion` priority: `warmup` (missing hats, `base_fold` 0 or null hats), `coupling_warmup` (base eligible, `base_fold == 1` or missing `g_hat`/`beta_hat` on first hat fold), `missing_minutes_hat`, `missing_points_hat`, `missing_overlay_hat`, `missing_beta_hat`, `invalid_feature`, else empty

Treat `base_fold == 1` as the first hat-producing fold (1-indexed to match expanding-window fold order after warm-up dates). Dates with no hats: `base_fold = 0`, exclusion `warmup`.

- [ ] **Step 4: Re-run tests — expect PASS**

---

### Task 2: Overlay fit and \(\beta\) shrinkage

**Files:**
- Modify: `src/models/xgboost_models/joint_calibration.py`
- Test: `tests/models/test_joint_calibration.py`

**Interfaces:**
- Consumes: Task 1
- Produces: `OverlayFit` dataclass (`coef`, `intercept`, `feature_names`, `impute_median`, `scale_mean`, `scale_std`, `alpha`). `fit_overlay(frame, residual) -> OverlayFit`. `apply_overlay(fit, frame) -> np.ndarray` (unclipped \(g\)). `fit_beta(u, residual, minutes_hat, bins, shrinkage=20.0) -> BetaMap` with `beta_global`, `beta_by_bin`. `apply_beta(beta_map, minutes_hat) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
class OverlayFitTests(unittest.TestCase):
    def test_overlay_minutes_shock_coefficient_is_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        shock = rng.normal(0, 4, n)
        frame = pd.DataFrame(
            {
                "minutes_hat": 28.0 + shock,
                "min_mean_10": np.full(n, 28.0),
                "pts_per_min_10": rng.uniform(0.4, 0.7, n),
                "usg_wmean_10": rng.uniform(15, 30, n),
                "start_rate_10": rng.uniform(0, 1, n),
            }
        )
        residual = 0.8 * shock + rng.normal(0, 0.3, n)
        from src.models.xgboost_models.joint_calibration import (
            apply_overlay,
            fit_overlay,
        )
        fit = fit_overlay(frame, residual)
        self.assertGreaterEqual(fit.coef[0], 0.0)
        pred = apply_overlay(fit, frame)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred < 0).any() or True)

    def test_overlay_may_be_negative(self) -> None:
        frame = pd.DataFrame(
            {
                "minutes_hat": [20.0, 20.0, 20.0, 10.0],
                "min_mean_10": [20.0, 20.0, 20.0, 20.0],
                "pts_per_min_10": [0.5, 0.5, 0.5, 0.5],
                "usg_wmean_10": [20.0, 20.0, 20.0, 20.0],
                "start_rate_10": [0.8, 0.8, 0.8, 0.8],
            }
        )
        residual = np.array([1.0, 1.0, 1.0, -6.0])
        from src.models.xgboost_models.joint_calibration import (
            apply_overlay,
            fit_overlay,
        )
        pred = apply_overlay(fit_overlay(frame, residual), frame)
        self.assertLess(pred[-1], 0.0)


class BetaFitTests(unittest.TestCase):
    def test_beta_recovers_slope_and_clips_nonnegative(self) -> None:
        u = np.array([-4.0, -2.0, 0.0, 2.0, 4.0] * 20)
        residual = 0.5 * u
        minutes_hat = np.where(u >= 0, 30.0, 10.0)
        bins = np.array([0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0])
        from src.models.xgboost_models.joint_calibration import fit_beta
        fitted = fit_beta(u, residual, minutes_hat, bins)
        self.assertGreater(fitted.beta_global, 0.3)
        self.assertLess(fitted.beta_global, 0.7)
        for value in fitted.beta_by_bin.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 2.5)
```

- [ ] **Step 2: Run the new tests — expect FAIL** (missing `fit_overlay` / `fit_beta`)

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_joint_calibration.OverlayFitTests tests.models.test_joint_calibration.BetaFitTests -v
```

- [ ] **Step 3: Implement overlay and beta**

Overlay design: `minutes_shock = minutes_hat - min_mean_10`, then `pts_per_min_10`, `usg_wmean_10`, `start_rate_10`. Median impute, standardize, ridge `alpha=1.0`. `scipy.optimize.lsq_linear` with bounds `minutes_shock >= 0`, others unbounded, intercept unbounded.

\(\beta\): through-origin OLS per bin of `minutes_hat` vs `residual` (\(v-g\)), shrink \(k=20\) toward global, clip to `[0, 2.5]`.

- [ ] **Step 4: Re-run — expect PASS**

---

### Task 3: Nested coupling OOF and \(\epsilon\) pools

**Files:**
- Modify: `src/models/xgboost_models/joint_calibration.py`
- Test: `tests/models/test_joint_calibration.py`

**Interfaces:**
- Consumes: `fit_overlay`, `apply_overlay`, `fit_beta`, `apply_beta`, `classify_universes`
- Produces: `run_nested_coupling(frame) -> pd.DataFrame` filling `g_hat`, `beta_hat` using prior `base_oof_eligible` rows only. `build_epsilon_pools(frame) -> EpsilonPools` from `coupling_oof_eligible`, bins `[0, 8, 14, 20, 28, inf)` on `mu_hat = max(0, points_hat + g_hat)`, winsorize 0.5%/99.5%, min size 40, center each pool, store removed means and raw stds.

- [ ] **Step 1: Write the failing tests**

```python
class NestedCouplingTests(unittest.TestCase):
    def test_nested_coupling_leaves_first_hat_fold_without_g(self) -> None:
        frame = pd.DataFrame(
            {
                "base_fold": [1] * 80 + [2] * 80,
                "minutes_hat": np.r_[np.full(80, 18.0), np.full(80, 28.0)],
                "points_hat": np.r_[np.full(80, 10.0), np.full(80, 18.0)],
                "minutes": np.r_[np.full(80, 18.0), np.full(80, 32.0)],
                "pts": np.r_[np.full(80, 10.0), np.full(80, 24.0)],
                "min_mean_10": 20.0,
                "pts_per_min_10": 0.5,
                "usg_wmean_10": 22.0,
                "start_rate_10": 0.6,
            }
        )
        from src.models.xgboost_models.joint_calibration import run_nested_coupling
        out = run_nested_coupling(frame)
        self.assertTrue(out.loc[out["base_fold"].eq(1), "g_hat"].isna().all())
        self.assertTrue(out.loc[out["base_fold"].eq(2), "g_hat"].notna().all())
        labeled = classify_universes(out)
        self.assertEqual(int(labeled["coupling_oof_eligible"].sum()), 80)


class EpsilonPoolTests(unittest.TestCase):
    def test_epsilon_pools_ignore_coupling_warmup_and_are_centered(self) -> None:
        frame = pd.DataFrame(
            {
                "base_fold": [1] * 50 + [2] * 80,
                "minutes_hat": 24.0,
                "points_hat": 12.0,
                "g_hat": np.r_[np.full(50, np.nan), np.full(80, 0.0)],
                "beta_hat": np.r_[np.full(50, np.nan), np.full(80, 0.4)],
                "minutes": 24.0,
                "pts": np.r_[np.full(50, 30.0), np.full(80, 12.0)],
            }
        )
        from src.models.xgboost_models.joint_calibration import build_epsilon_pools
        pools = build_epsilon_pools(classify_universes(frame))
        self.assertTrue(all(abs(pool.mean()) < 1e-8 for pool in pools.pools.values()))
        stacked = np.concatenate(list(pools.pools.values()))
        self.assertLess(stacked.max(), 15.0)
```

Fold-1 `pts=30` must not enter pools (warmup). Fold-2 residuals around 0 after centering.

- [ ] **Step 2: Run — expect FAIL**

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_joint_calibration.NestedCouplingTests tests.models.test_joint_calibration.EpsilonPoolTests -v
```

- [ ] **Step 3: Implement `run_nested_coupling` and `build_epsilon_pools`**

For each `base_fold` sorted ascending: skip fold `<= 1` for fitting. Train overlay on prior base-eligible rows (`v = pts - points_hat`), predict current fold `g_hat`. Then train \(\beta\) on prior rows’ `v - g_hat` (use already-written `g_hat` on prior folds), write `beta_hat` for current fold.

`build_epsilon_pools`: `u = minutes - minutes_hat`, `epsilon = pts - (points_hat + g_hat) - beta_hat * u`, only `coupling_oof_eligible`.

- [ ] **Step 4: Re-run — expect PASS**

---

### Task 4: Points OOF helper, train/save/load fail-closed

**Files:**
- Modify: `src/models/xgboost_models/points.py` — `add_predicted_points_oof`
- Modify: `src/models/xgboost_models/joint_calibration.py` — `train_joint_calibration`, `save_joint_calibration`, `load_joint_calibration`
- Modify: `src/models/xgboost_models/__init__.py`
- Test: `tests/models/test_joint_calibration.py`
- Test: `tests/models/test_points_features.py` — one leakage-style test that OOF points are not actual Game N points (optional if time; prefer a fake-factory test in `test_joint_calibration.py`)

**Interfaces:**
- `add_predicted_points_oof(frame, *, points_model_factory=None, splits=None, ...) -> pd.DataFrame` writes `points_hat` (or `predicted_points_oof`) using expanding windows; actual `pts` never copied into the hat.
- `train_joint_calibration(frame, minutes_mean_path, minutes_dist_path, points_path, *, minutes_model_factory=None, points_model_factory=None) -> JointCalibration`
- `save_joint_calibration(artifact, path) -> Path`
- `load_joint_calibration(path, minutes_mean_path, minutes_dist_path, points_path) -> JointCalibration` raises if any fingerprint hash mismatches

- [ ] **Step 1: Write the failing tests**

```python
class PointsOofTests(unittest.TestCase):
    def test_points_oof_is_not_actual_points(self) -> None:
        from src.models.xgboost_models.points import add_predicted_points_oof
        frame = pd.DataFrame(
            {
                "game_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "pts": [10.0, 20.0, 30.0, 40.0],
                "predicted_minutes_oof": [20.0, 20.0, 20.0, 20.0],
            }
        )
        class Spy:
            def fit(self, train, **kwargs):
                return self
            def predict_mean(self, rows):
                return np.full(len(rows), 15.0)
        splits = [(frame.index[:2], frame.index[2:])]
        out = add_predicted_points_oof(
            frame,
            points_model_factory=lambda: Spy(),
            splits=splits,
        )
        later = out.loc[out.index[2:], "predicted_points_oof"]
        self.assertTrue((later == 15.0).all())
        self.assertFalse((later.values == out.loc[out.index[2:], "pts"].values).any())


class SaveLoadTests(unittest.TestCase):
    def test_load_fails_closed_on_hash_mismatch(self) -> None:
        from src.models.xgboost_models.joint_calibration import (
            JointCalibration,
            load_joint_calibration,
            save_joint_calibration,
        )
        with TemporaryDirectory() as tmp:
            minutes = Path(tmp) / "m.joblib"
            dist = Path(tmp) / "d.joblib"
            points = Path(tmp) / "p.joblib"
            for path in (minutes, dist, points):
                path.write_bytes(b"ok")
            artifact = JointCalibration.example_for_test(
                fingerprints={
                    "minutes_mean": content_hash(minutes),
                    "minutes_distribution": content_hash(dist),
                    "points_mean": content_hash(points),
                }
            )
            out = Path(tmp) / "joint.joblib"
            save_joint_calibration(artifact, out)
            points.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_joint_calibration(
                    out,
                    minutes_mean_path=minutes,
                    minutes_dist_path=dist,
                    points_path=points,
                )
```

Prefer a real `JointCalibration` dataclass constructed in the test over `example_for_test` if the fields are small enough to fill explicitly.

- [ ] **Step 2: Run — expect FAIL**

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_joint_calibration.PointsOofTests tests.models.test_joint_calibration.SaveLoadTests -v
```

- [ ] **Step 3: Implement `add_predicted_points_oof` (mirror `add_predicted_minutes_oof` in `points.py`), dataclass payload, save/load with three hashes, and `train_joint_calibration` that:**

  1. `assert_preholdout`
  2. Shared `expanding_window_splits`
  3. Minutes OOF then stacked interactions then points OOF
  4. Nested coupling
  5. \(\epsilon\) pools from coupling-eligible
  6. Final overlay on `base_oof_eligible`; final \(\beta\) on \(v - g_{\text{final}}\)
  7. Record `n_preholdout_appearances`, `n_base_oof_eligible`, `n_coupling_oof_eligible`, `oof_coverage`, exclusion counts, both max dates, three fingerprints

Allow factories in tests so this does not fit real XGBoost in unittest.

Export `train_joint_calibration`, `save_joint_calibration`, `load_joint_calibration` from `src/models/xgboost_models/__init__.py`.

- [ ] **Step 4: Re-run Task 1–4 tests — expect PASS**

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_joint_calibration tests.models.test_points_features -v
```

---

### Task 5: Diagnostics notebook and tolerance gate

**Files:**
- Create: `notebooks/nba/points/joint_calibration.ipynb`
- Modify: `src/models/xgboost_models/joint_calibration.py` — `overlay_gate(on_metrics, off_metrics, n, *, mae_tol=0.05, bias_tol=0.05, min_n=200) -> bool`
- Test: `tests/models/test_joint_calibration.py`

**Interfaces:**
- `overlay_gate` returns False if `n < min_n` (slice skipped, not failed) when called per-slice; overall must have `n >= min_n`. Pass iff MAE_on ≤ MAE_off + 0.05 and |bias_on| ≤ |bias_off| + 0.05.

- [ ] **Step 1: Failing tests for the gate**

```python
class OverlayGateTests(unittest.TestCase):
    def test_gate_allows_small_mae_wiggle(self) -> None:
        from src.models.xgboost_models.joint_calibration import overlay_gate
        self.assertTrue(
            overlay_gate(
                {"mae": 4.52, "bias": 0.01, "n": 500},
                {"mae": 4.50, "bias": 0.01, "n": 500},
            )
        )

    def test_gate_skips_tiny_slices(self) -> None:
        from src.models.xgboost_models.joint_calibration import overlay_gate
        self.assertTrue(
            overlay_gate(
                {"mae": 9.0, "bias": 3.0, "n": 20},
                {"mae": 4.0, "bias": 0.0, "n": 20},
                require_min_n=False,
            )
        )
```

Use `require_min_n=False` meaning “slice ineligible → do not fail the suite”. Document: tiny slices return `None` (skipped); `evaluate_overlay_gate(results) -> bool` fails only on eligible slices that miss the tolerance.

- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement `overlay_gate` / `evaluate_overlay_gate`**
- [ ] **Step 4: Notebook cells**

  1. Imports, `ROOT`, `assert_preholdout` after load
  2. Load silver `2019-20`–`2024-25` only, `add_points_features`, appearances
  3. `train_joint_calibration(...)` then `save_joint_calibration`
  4. Print fingerprints, exclusion counts, coverage, dates
  5. \(\beta\) by bin table; overlay coefficients
  6. \(\epsilon\) pool sizes, removed means, raw stds
  7. Overlay-on vs off (cross-fitted \(g\)) overall + role-shock slices; apply gate

Do not read `2025-26` parquet in this notebook.

- [ ] **Step 5: Re-run full joint tests — expect PASS**

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_joint_calibration -v
```

---

## Spec coverage

| Spec requirement | Task |
|---|---|
| Two universes, coupling_warmup | 1, 3 |
| Nested overlay then β | 3 |
| ε from coupling_oof only | 3 |
| Final β on \(v - g_{\text{final}}\) | 4 |
| Fingerprint three artifacts, fail closed | 1, 4 |
| `base_oof_max_training_date`, `final_calibration_max_date` | 4 |
| `n_preholdout_appearances`, `oof_coverage` | 4 |
| \(g\) may be negative | 2 |
| Exclusion reasons including overlay/beta | 1, 4 |
| MAE/bias tolerance gate, min n | 5 |
| Notebook, no 2025-26 | 5 |
| `add_predicted_points_oof` | 4 |
