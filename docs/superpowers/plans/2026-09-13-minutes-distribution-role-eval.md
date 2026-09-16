# Minutes Distribution Role Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a holdout distribution eval in `xgboost_model.ipynb` that checks whether actual minutes were a believable draw from the fitted XGBoost minutes distribution, sliced by expected starter/bench, actual tip starter/bench, and role shock.

**Architecture:** Keep scoring in `src.models.evaluation` (add PIT next to coverage and NLL). Drive the eval from new cells at the end of the existing minutes XGBoost notebook, using the already-fitted model and 2025-26 `test` frame. No retrain, no reduced feature list, no implied points.

**Tech Stack:** Python, pandas, numpy, matplotlib, seaborn, existing `XGBoostMinutesModel.simulate`, `src.models.evaluation`, unittest.

## Global Constraints

- Notebook only for the eval UI: `notebooks/nba/minutes/xgboost_model.ipynb`
- Use the already-fitted `XGBoostMinutesModel` (47-feature `TIER1_MINUTES_FEATURES`); do not retrain
- Do not switch to `reduced_features`
- Holdout rows: 2025-26 `test` appearances (`minutes > 0`)
- `model.simulate(test, simulations=2_000)` — not 10,000
- Outcome column: `minutes`
- No MAE, RMSE, or R² in this eval
- Skip CRPS on the full holdout
- Actual starter status is a slice only, never a feature
- Expected starter: `start_rate_10 >= 0.5`
- Actual starter: non-empty `start_position` (same rule as `_started_obs` in `src/features/minutes/player.py`)
- Missing `start_rate_10`: exclude from expected-role tables; keep in overall
- Missing / empty `start_position`: actual bench
- No new `src` package; PIT may be added to existing `src/models/evaluation.py`
- This workspace currently has no git repository; skip `git commit` if `.git` is absent

## File map

- Modify: `src/models/evaluation.py` — add `probability_integral_transform`
- Modify: `tests/models/test_evaluation.py` — PIT unit tests (ties, below, above)
- Modify: `notebooks/nba/minutes/xgboost_model.ipynb` — five new cells at the end (labels+simulate, metrics table, expected PIT histograms, actual PIT histograms, role-shock table)

---

### Task 1: Probability integral transform

**Files:**
- Modify: `src/models/evaluation.py`
- Test: `tests/models/test_evaluation.py`

**Interfaces:**
- Consumes: `_align(samples, actual)` already in `src/models/evaluation.py`
- Produces: `probability_integral_transform(samples: np.ndarray, actual: np.ndarray) -> np.ndarray` of shape `(n_rows,)`. PIT = mean(draws < actual) + 0.5 * mean(draws == actual) per row.

- [ ] **Step 1: Write the failing test**

Add this method to `EvaluationMetricTests` in `tests/models/test_evaluation.py` and import `probability_integral_transform`:

```python
def test_probability_integral_transform_handles_ties(self) -> None:
    actual = np.array([10.0, 2.0, 20.0])
    samples = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    pit = probability_integral_transform(samples, actual)
    np.testing.assert_allclose(pit, [0.5, 0.375, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_evaluation.EvaluationMetricTests.test_probability_integral_transform_handles_ties -v
```

Expected: FAIL with `ImportError` or `AttributeError` for `probability_integral_transform`.

- [ ] **Step 3: Write minimal implementation**

In `src/models/evaluation.py`, add:

```python
def probability_integral_transform(
    samples: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    """Row-wise PIT with half-credit for ties."""
    draws, outcomes = _align(samples, actual)
    below = np.mean(draws < outcomes[:, None], axis=1)
    equal = np.mean(draws == outcomes[:, None], axis=1)
    return below + 0.5 * equal
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m unittest tests.models.test_evaluation -v
```

Expected: all tests PASS, including `test_probability_integral_transform_handles_ties`.

- [ ] **Step 5: Commit**

Skip if `/Users/alexgonzalez/Documents/nba_quant/.git` does not exist. Otherwise:

```bash
git add src/models/evaluation.py tests/models/test_evaluation.py
git commit -m "$(cat <<'EOF'
Add PIT scoring for simulated minutes distributions.

EOF
)"
```

---

### Task 2: Labels and simulate

**Files:**
- Modify: `notebooks/nba/minutes/xgboost_model.ipynb` (new cell at the first empty index at the end; currently cell 9 if empty, else append)

**Interfaces:**
- Consumes: fitted `model`, `test` DataFrame from earlier notebook cells; `model.simulate(frame, simulations=int | None = None) -> np.ndarray` shaped `(n_rows, simulations)`
- Produces: `eval_frame` (copy of `test` with `expected_starter`, `actual_starter`, `role_shock`, `has_expected_role`); `samples` ndarray `(len(eval_frame), 2000)`; `actual` float ndarray

- [ ] **Step 1: Add the labels + simulate cell**

Use EditNotebook. If the last cell is empty, edit it; otherwise create a new cell at the end.

```python
from src.models.evaluation import (
    interval_coverage,
    negative_log_likelihood,
    probability_integral_transform,
)

eval_frame = test.copy()
start_text = (
    eval_frame["start_position"].astype("string").str.strip()
)
eval_frame["actual_starter"] = (
    start_text.notna()
    & start_text.ne("")
    & start_text.ne("nan")
    & start_text.ne("<NA>")
)
eval_frame["has_expected_role"] = eval_frame["start_rate_10"].notna()
eval_frame["expected_starter"] = (
    eval_frame["start_rate_10"] >= 0.5
)
eval_frame["role_shock"] = (
    eval_frame["has_expected_role"]
    & (
        eval_frame["expected_starter"]
        != eval_frame["actual_starter"]
    )
)

actual = eval_frame["minutes"].to_numpy(dtype=float)
samples = model.simulate(eval_frame, simulations=2_000)

print(f"Simulated draws: {samples.shape}")
print(
    "Expected starter rows: "
    f"{int(eval_frame.loc[eval_frame['has_expected_role'], 'expected_starter'].sum()):,}"
)
print(f"Actual starter rows: {int(eval_frame['actual_starter'].sum()):,}")
print(f"Role shock rows: {int(eval_frame['role_shock'].sum()):,}")
print(
    "Missing start_rate_10: "
    f"{int((~eval_frame['has_expected_role']).sum()):,}"
)
```

- [ ] **Step 2: Run the cell**

Expected stdout includes `Simulated draws: (26648, 2000)` or the current `test` length, plus the four counts. Do not train a new model.

- [ ] **Step 3: Commit**

Skip if no `.git`. Otherwise commit the notebook with message `Add minutes holdout simulation and role labels.`

---

### Task 3: Metrics table

**Files:**
- Modify: `notebooks/nba/minutes/xgboost_model.ipynb` (next new cell)

**Interfaces:**
- Consumes: `eval_frame`, `samples`, `actual`, `probability_integral_transform`, `interval_coverage`, `negative_log_likelihood`
- Produces: `distribution_metrics` DataFrame with columns `split`, `n`, `coverage_80`, `pit_mean`, `pit_std`, `nll`

- [ ] **Step 1: Add the metrics cell**

```python
def slice_metrics(
    mask: pd.Series,
    split: str,
) -> dict:
    row_mask = mask.to_numpy()
    slice_samples = samples[row_mask]
    slice_actual = actual[row_mask]
    pit = probability_integral_transform(
        slice_samples,
        slice_actual,
    )
    return {
        "split": split,
        "n": int(row_mask.sum()),
        "coverage_80": interval_coverage(
            slice_samples,
            slice_actual,
        ),
        "pit_mean": float(pit.mean()),
        "pit_std": float(pit.std()),
        "nll": negative_log_likelihood(
            slice_samples,
            slice_actual,
        ),
    }

expected_known = eval_frame["has_expected_role"]
distribution_metrics = pd.DataFrame(
    [
        slice_metrics(
            pd.Series(True, index=eval_frame.index),
            "all",
        ),
        slice_metrics(
            expected_known & eval_frame["expected_starter"],
            "expected_starter",
        ),
        slice_metrics(
            expected_known & ~eval_frame["expected_starter"],
            "expected_bench",
        ),
        slice_metrics(
            eval_frame["actual_starter"],
            "actual_starter",
        ),
        slice_metrics(
            ~eval_frame["actual_starter"],
            "actual_bench",
        ),
        slice_metrics(
            eval_frame["role_shock"],
            "role_shock",
        ),
    ]
)
distribution_metrics.round(3)
```

- [ ] **Step 2: Run the cell**

Expected: six-row table. `coverage_80` on `all` should be a fraction between 0 and 1 (target ~0.80, not a pass/fail). `pit_mean` near 0.5 is well calibrated. No MAE columns.

- [ ] **Step 3: Commit**

Skip if no `.git`. Otherwise commit with message `Add minutes distribution coverage and PIT table.`

---

### Task 4: Pregame PIT histograms

**Files:**
- Modify: `notebooks/nba/minutes/xgboost_model.ipynb` (next new cell)

**Interfaces:**
- Consumes: `eval_frame`, `samples`, `actual`, `probability_integral_transform`
- Produces: two PIT histograms — expected starter vs expected bench

- [ ] **Step 1: Add the pregame histogram cell**

```python
eval_frame["pit"] = probability_integral_transform(
    samples,
    actual,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for axis, title, mask in (
    (
        axes[0],
        "Expected starter",
        eval_frame["has_expected_role"]
        & eval_frame["expected_starter"],
    ),
    (
        axes[1],
        "Expected bench",
        eval_frame["has_expected_role"]
        & ~eval_frame["expected_starter"],
    ),
):
    sns.histplot(
        eval_frame.loc[mask, "pit"],
        bins=20,
        binrange=(0, 1),
        ax=axis,
        color="steelblue",
    )
    axis.axhline(
        len(eval_frame.loc[mask]) / 20,
        color="black",
        linestyle="--",
        linewidth=1,
        label="uniform",
    )
    axis.set_title(title)
    axis.set_xlabel("PIT")
    axis.set_xlim(0, 1)
fig.suptitle(
    "Pregame role · minutes PIT · 2025-26 holdout",
    y=1.02,
)
fig.tight_layout()
plt.show()
```

- [ ] **Step 2: Run the cell**

Expected: two histograms on [0, 1]. Flat ≈ calibrated. Mass at 0 = over-assigned minutes; mass at 1 = under-assigned; U-shape = too narrow.

- [ ] **Step 3: Commit**

Skip if no `.git`. Otherwise commit with message `Add pregame starter vs bench PIT histograms.`

---

### Task 5: Actual-lineup PIT histograms

**Files:**
- Modify: `notebooks/nba/minutes/xgboost_model.ipynb` (next new cell)

**Interfaces:**
- Consumes: `eval_frame["pit"]`, `eval_frame["actual_starter"]`
- Produces: two PIT histograms — actual starter vs actual bench

- [ ] **Step 1: Add the actual-lineup histogram cell**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for axis, title, mask in (
    (axes[0], "Actual starter", eval_frame["actual_starter"]),
    (axes[1], "Actual bench", ~eval_frame["actual_starter"]),
):
    sns.histplot(
        eval_frame.loc[mask, "pit"],
        bins=20,
        binrange=(0, 1),
        ax=axis,
        color="indianred",
    )
    axis.axhline(
        len(eval_frame.loc[mask]) / 20,
        color="black",
        linestyle="--",
        linewidth=1,
        label="uniform",
    )
    axis.set_title(title)
    axis.set_xlabel("PIT")
    axis.set_xlim(0, 1)
fig.suptitle(
    "Actual tip role · minutes PIT · 2025-26 holdout",
    y=1.02,
)
fig.tight_layout()
plt.show()
```

- [ ] **Step 2: Run the cell**

Expected: two histograms. This is the “lineup already known” view. Do not treat it as the pregame betting view.

- [ ] **Step 3: Commit**

Skip if no `.git`. Otherwise commit with message `Add actual starter vs bench PIT histograms.`

---

### Task 6: Role-shock table

**Files:**
- Modify: `notebooks/nba/minutes/xgboost_model.ipynb` (next new cell)

**Interfaces:**
- Consumes: `eval_frame`, `slice_metrics` from Task 3
- Produces: table with `expected_starter_sat` and `expected_bench_started`

- [ ] **Step 1: Add the role-shock cell**

```python
sat = (
    eval_frame["has_expected_role"]
    & eval_frame["expected_starter"]
    & ~eval_frame["actual_starter"]
)
started = (
    eval_frame["has_expected_role"]
    & ~eval_frame["expected_starter"]
    & eval_frame["actual_starter"]
)

role_shock_metrics = pd.DataFrame(
    [
        slice_metrics(sat, "expected_starter_sat"),
        slice_metrics(started, "expected_bench_started"),
    ]
)
print(
    "Role shock is expected role ≠ actual tip starter. "
    "Coverage collapse or PIT piled at 0/1 means the "
    "volume module does not widen when the rotation moves."
)
role_shock_metrics.round(3)
```

- [ ] **Step 2: Run the cell**

Expected: two-row table. Interpret against the spec success line: if these slices have coverage far below 0.80 or PIT mean far from 0.5, minutes uncertainty is not yet a trustworthy points input when the lineup is in flux.

- [ ] **Step 3: Commit**

Skip if no `.git`. Otherwise commit with message `Add role-shock minutes distribution table.`

---

## Self-review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| Fitted model, no retrain, 47 features | Task 2 |
| 2025-26 `test`, `simulations=2_000` | Task 2 |
| Expected starter `start_rate_10 >= 0.5` | Task 2 |
| Actual starter from `start_position` | Task 2 |
| Missing `start_rate_10` excluded from expected tables | Tasks 2–4 |
| Missing `start_position` = actual bench | Task 2 |
| Coverage 80, PIT with half-ties, NLL | Tasks 1, 3 |
| No MAE / RMSE / R² | Task 3 |
| Skip CRPS | no CRPS call in any cell |
| Pregame histograms | Task 4 |
| Actual-lineup histograms | Task 5 |
| Role-shock sat vs started | Task 6 |
| No implied points, no reduced list, no new package | all tasks |

**Placeholder scan:** none.

**Type consistency:** `slice_metrics(mask: pd.Series, split: str) -> dict` is defined in Task 3 and reused in Task 6. `eval_frame["pit"]` is written in Task 4 and read in Task 5. `samples` is `(n, 2000)`.
