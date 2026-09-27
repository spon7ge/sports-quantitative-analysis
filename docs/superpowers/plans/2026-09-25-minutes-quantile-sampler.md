# Minutes Quantile Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an inverse-transform minute quantile function `Q` from the frozen 11-knot minutes model, with out-of-fold tail shapes, so a minutes line is priced by inverting `Q` and draws exist only for later combination with other random quantities.

**Architecture:** `models/shared/minutes_sampler.py` owns grid prep, `quantile_minutes`, tail tables, the sidecar, `sample_minutes`, line probabilities, and CRPS. `models/shared/train.py` learns to early-stop walk-forward folds on the last 10% of training dates and to keep every validation fold's raw knots. The minutes notebook builds the sidecar and reports tail shape plus one 2025-26 CRPS. The residual-bootstrap simulator is not modified.

**Tech Stack:** Python, numpy, pandas, joblib, hashlib, pytest, existing XGBoost quantile training in `models/shared/train.py`.

## Global Constraints

- Quantile levels, in order: `0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95`
- Knot floor: `1e-3` via `max(q, 1e-3)` after one sort. No second sort
- Holdout boundary: game dates must be strictly before `2025-10-21`
- Finished minutes clip to `[0, maximum_minutes("nba")]` (63). The grid is not clipped
- Tail build uses `early_stop="train_tail"` at 10% of training dates. Default walk-forward stays `"validation"`
- Miss-rate band on each requested fold and each tail: 4% to 6%, pooled across groups
- `starting` groups are `0` and `1`. The builder allocates both before scanning rows
- A blank box-score `start_position` is already coded as bench in the notebook. The sampler does not code a missing draw-time `starting` as `0`
- Sidecar load accepts fold ids exactly `(1, 2, 3, 4)`
- Draws are not a price. `P(M < L) = inf{u : Q(u) >= L}`, with `0` at or below the bottom of `Q` and `1` above the top
- Minute RNG material is `seed|minutes|player_id|game_id` after id canonicalization. Do not change `JointPointsSimulator._rng`
- 2025-26 knots for the record come from `preds_ho`, not `preds_ho_live`. Do not edit tables after seeing that score
- Do not rewrite `models/saved_models/min_nba_model_2026-04-12.joblib`
- Tests must not fit XGBoost

## File map

- Create: `models/shared/minutes_sampler.py` — prep, map, tables, sidecar, draws, line probability, CRPS, tail-bin shares
- Create: `tests/models/test_minutes_sampler.py` — synthetic tests
- Modify: `models/shared/train.py` — `early_stop` on `fit_quantile_models` and `run_walk_forward`; retain every fold's raw predictions
- Modify: `models/shared/__init__.py` — export the public sampler functions
- Modify: `notebooks/nba/minutes/min_nba_model.ipynb` — diagnostic report, sidecar build, one 2025-26 record
- The mean-plus-residual simulator was later removed. This plan does not recreate it.

---

### Task 1: Grid prep and quantile map

**Files:**
- Create: `models/shared/minutes_sampler.py`
- Create: `tests/models/test_minutes_sampler.py`
- Modify: `models/shared/__init__.py`

**Interfaces:**
- Consumes: `maximum_minutes` from `src.models.settlement`
- Produces:
  - `QUANTILE_LEVELS: tuple[float, ...]`
  - `KNOT_FLOOR = 1e-3`
  - `@dataclass MinuteTailTables` with field `arrays: dict[tuple[str, int], np.ndarray]`
  - `prepare_quantile_grid(grids: np.ndarray) -> np.ndarray` shape `(rows, 11)`
  - `empirical_quantile(v: np.ndarray, table: np.ndarray) -> np.ndarray`
  - `quantile_minutes(u, grids, lower_groups, upper_groups, tables) -> np.ndarray` shape `(rows, draws)`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_minutes_sampler.py`:

```python
import numpy as np
import pytest

from models.shared.minutes_sampler import (
    KNOT_FLOOR,
    QUANTILE_LEVELS,
    MinuteTailTables,
    prepare_quantile_grid,
    quantile_minutes,
)


def _grid(values):
    grid = np.asarray(values, dtype=float)
    assert grid.shape == (11,)
    return grid.reshape(1, -1)


def _tables(lower0, upper0, lower1, upper1):
    return MinuteTailTables(
        arrays={
            ("lower", 0): np.asarray(lower0, dtype=float),
            ("upper", 0): np.asarray(upper0, dtype=float),
            ("lower", 1): np.asarray(lower1, dtype=float),
            ("upper", 1): np.asarray(upper1, dtype=float),
        }
    )


SORTED = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
TABLES = _tables([0.2, 1.0], [0.0, 4.0], [0.5, 1.0], [0.0, 10.0])


def test_prepare_floors_without_reordering_levels():
    raw = np.array([-0.1, 0.0005, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    prepared = prepare_quantile_grid(raw)
    assert prepared.shape == (1, 11)
    assert prepared[0, 0] == KNOT_FLOOR
    assert prepared[0, 1] == KNOT_FLOOR
    assert prepared[0, 2] == 1


def test_prepare_rejects_non_finite_knot():
    raw = SORTED.copy()
    raw[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        prepare_quantile_grid(raw)


def test_quantile_minutes_hits_all_eleven_stored_levels():
    u = np.asarray(QUANTILE_LEVELS, dtype=float).reshape(1, -1)
    got = quantile_minutes(u, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    np.testing.assert_allclose(got[0], SORTED)


def test_lower_tail_scales_with_q05_and_upper_tail_shifts_with_q95():
    u = np.array([[0.025, 0.975]])
    base = quantile_minutes(u, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    doubled = SORTED.copy()
    doubled[0] = 20
    scaled = quantile_minutes(u, _grid(doubled), np.array([1]), np.array([1]), TABLES)
    shifted = SORTED.copy()
    shifted[-1] = 35
    moved = quantile_minutes(u, _grid(shifted), np.array([1]), np.array([1]), TABLES)
    np.testing.assert_allclose(scaled[0, 0], 2 * base[0, 0])
    np.testing.assert_allclose(moved[0, 1], base[0, 1] + 5)


def test_quantile_minutes_is_non_decreasing_and_reads_each_tail_group():
    u = np.linspace(0, 1, 21).reshape(1, -1)
    got = quantile_minutes(u, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    assert np.all(np.diff(got[0]) >= -1e-12)
    low = np.array([[0.025]])
    bench = quantile_minutes(low, _grid(SORTED), np.array([0]), np.array([0]), TABLES)
    starter = quantile_minutes(low, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    assert bench[0, 0] != pytest.approx(starter[0, 0])
    mixed = quantile_minutes(
        np.array([[0.025, 0.975]]),
        _grid(SORTED),
        np.array([0]),
        np.array([1]),
        TABLES,
    )
    assert mixed[0, 0] == pytest.approx(bench[0, 0])
    assert mixed[0, 1] != pytest.approx(
        quantile_minutes(np.array([[0.975]]), _grid(SORTED), np.array([0]), np.array([0]), TABLES)[0, 0]
    )


def test_draw_above_63_clips_to_63():
    tall = SORTED.copy()
    tall[-1] = 80
    got = quantile_minutes(np.array([[0.99]]), _grid(tall), np.array([1]), np.array([1]), TABLES)
    assert got[0, 0] == 63


def test_bad_u_knot_count_and_unknown_group_raise():
    with pytest.raises(ValueError, match="knot"):
        quantile_minutes(np.array([[0.5]]), SORTED[:10].reshape(1, -1), np.array([1]), np.array([1]), TABLES)
    with pytest.raises(ValueError, match="outside"):
        quantile_minutes(np.array([[1.1]]), _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    with pytest.raises(ValueError, match="group"):
        quantile_minutes(np.array([[0.025]]), _grid(SORTED), np.array([2]), np.array([1]), TABLES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `models.shared.minutes_sampler`.

- [ ] **Step 3: Implement prep and the map**

Create `models/shared/minutes_sampler.py`:

```python
"""Inverse-transform minutes quantile function. Draws are not a price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.settlement import maximum_minutes

QUANTILE_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
KNOT_FLOOR = 1e-3


@dataclass
class MinuteTailTables:
    arrays: dict[tuple[str, int], np.ndarray]


def prepare_quantile_grid(grids: np.ndarray) -> np.ndarray:
    """Sort once, then floor every knot at 1e-3. No second sort."""
    values = np.asarray(grids, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError(f"knot count must be {len(QUANTILE_LEVELS)}")
    if not np.isfinite(values).all():
        raise ValueError("non-finite knot")
    ordered = np.sort(values, axis=1)
    return np.maximum(ordered, KNOT_FLOOR)


def empirical_quantile(v: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Linear empirical quantile. Probability 0 is table[0], probability 1 is table[-1]."""
    xp = np.linspace(0.0, 1.0, len(table))
    return np.interp(np.asarray(v, dtype=float), xp, np.asarray(table, dtype=float))


def _lookup(tables: MinuteTailTables, tail: str, group: int) -> np.ndarray:
    try:
        return tables.arrays[(tail, int(group))]
    except KeyError as exc:
        raise ValueError(f"unknown group {group} for {tail}") from exc


def _row_quantile(u, prepared, lower_group, upper_group, tables):
    q05 = prepared[0]
    q95 = prepared[-1]
    out = np.empty(len(u), dtype=float)
    lower = u < 0.05
    upper = u > 0.95
    middle = ~lower & ~upper
    if np.any(lower):
        table = _lookup(tables, "lower", lower_group)
        out[lower] = q05 * empirical_quantile(u[lower] / 0.05, table)
    if np.any(upper):
        table = _lookup(tables, "upper", upper_group)
        out[upper] = q95 + empirical_quantile((u[upper] - 0.95) / 0.05, table)
    if np.any(middle):
        out[middle] = np.interp(u[middle], QUANTILE_LEVELS, prepared)
    return out


def quantile_minutes(u, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """Map uniforms of shape (rows, draws) through each row's Q."""
    uniforms = np.asarray(u, dtype=float)
    if uniforms.ndim != 2:
        raise ValueError("u must have shape (rows, draws)")
    if not np.isfinite(uniforms).all() or np.any(uniforms < 0) or np.any(uniforms > 1):
        raise ValueError("u outside [0, 1]")
    prepared = prepare_quantile_grid(grids)
    if prepared.shape[0] != uniforms.shape[0]:
        raise ValueError("grids and u must have the same number of rows")
    lower_ids = np.asarray(lower_groups)
    upper_ids = np.asarray(upper_groups)
    out = np.empty_like(uniforms, dtype=float)
    for index in range(prepared.shape[0]):
        out[index] = _row_quantile(
            uniforms[index],
            prepared[index],
            int(lower_ids[index]),
            int(upper_ids[index]),
            tables,
        )
    cap = maximum_minutes("nba")
    return np.clip(out, 0.0, cap)
```

Add the three names to the import block and `__all__` in `models/shared/__init__.py`: `KNOT_FLOOR`, `QUANTILE_LEVELS`, `MinuteTailTables`, `prepare_quantile_grid`, `quantile_minutes`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/shared/minutes_sampler.py models/shared/__init__.py tests/models/test_minutes_sampler.py
git commit -m "$(cat <<'EOF'
Add the minutes quantile map.

Q interpolates each row's own knots between 0.05 and 0.95 and reads group-specific tail tables outside that band.
EOF
)"
```

---

### Task 2: Tail table builder

**Files:**
- Modify: `models/shared/minutes_sampler.py`
- Modify: `tests/models/test_minutes_sampler.py`
- Modify: `models/shared/__init__.py`

**Interfaces:**
- Consumes: `prepare_quantile_grid`, `QUANTILE_LEVELS`, `MinuteTailTables`, `quantile_minutes`
- Produces: `build_tail_tables(oof, *, folds, fold_ranges, grouping=None) -> MinuteTailTables`
  - `oof` columns: `q_0.05` ... `q_0.95`, `fold_id`, `game_date`, `minutes`, `starting`, `early_stop`, `train_tail_frac`
  - `fold_ranges`: `{fold_id: {"train_start", "train_end", "val_start", "val_end"}}`
  - Returned tables also carry `floor`, `quantile_levels`, `early_stop`, `train_tail_frac`, `holdout_start`, `grouping`, `groups`, `folds`, `fold_ranges`, `oof`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_minutes_sampler.py`. Keep the Task 1 `MinuteTailTables` tests working by giving the dataclass defaults for every new field so existing `_tables()` still constructs.

```python
import pandas as pd

from models.shared.minutes_sampler import HOLDOUT_START, build_tail_tables


def _knot_frame(n, q05=10.0, q95=30.0):
    frame = pd.DataFrame(index=np.arange(n))
    for level in QUANTILE_LEVELS:
        if level <= 0.05:
            value = q05
        elif level >= 0.95:
            value = q95
        else:
            value = q05 + (q95 - q05) * (level - 0.05) / 0.90
        frame[f"q_{level:.2f}"] = value
    return frame


def _valid_oof():
    """100 rows, fold 1, pooled tail rate 5%. Group 1 has only r=0.5 and e=10."""
    starter = _knot_frame(20)
    starter["minutes"] = 20.0
    starter.loc[0, "minutes"] = 5.0
    starter.loc[1, "minutes"] = 40.0
    starter["starting"] = 1
    bench = _knot_frame(80)
    bench["minutes"] = 20.0
    bench.loc[0:3, "minutes"] = 2.0
    bench.loc[4:7, "minutes"] = 36.0
    bench["starting"] = 0
    oof = pd.concat([starter, bench], ignore_index=True)
    oof["fold_id"] = 1
    oof["game_date"] = pd.Timestamp("2024-01-15")
    oof["early_stop"] = "train_tail"
    oof["train_tail_frac"] = 0.10
    ranges = {1: {"train_start": "2023-10-01", "train_end": "2024-01-01", "val_start": "2024-01-02", "val_end": "2024-01-15"}}
    return oof, ranges


def test_builder_pins_and_tail_limits_come_from_build_tail_tables():
    oof, ranges = _valid_oof()
    tables = build_tail_tables(oof, folds=[1], fold_ranges=ranges)
    lower = tables.arrays[("lower", 1)]
    upper = tables.arrays[("upper", 1)]
    assert lower[-1] == 1
    assert upper[0] == 0
    assert set(np.round(lower, 5)) == {0.5, 1.0}
    assert set(np.round(upper, 5)) == {0.0, 10.0}
    row = oof.iloc[[0]]
    grids = row[[f"q_{level:.2f}" for level in QUANTILE_LEVELS]].to_numpy(dtype=float)
    eps = 1e-6
    got = quantile_minutes(
        np.array([[0.05 - eps, 0.95 + eps]]),
        grids,
        np.array([1]),
        np.array([1]),
        tables,
    )
    q05 = prepare_quantile_grid(grids)[0, 0]
    q95 = prepare_quantile_grid(grids)[0, -1]
    assert abs(got[0, 0] - q05) / q05 < 1e-3
    assert abs(got[0, 1] - q95) <= 1e-3
    stripped = MinuteTailTables(arrays=dict(tables.arrays))
    stripped.arrays[("lower", 1)] = lower[:-1]
    stripped.arrays[("upper", 1)] = upper[1:]
    bare = quantile_minutes(
        np.array([[0.05 - eps, 0.95 + eps]]),
        grids,
        np.array([1]),
        np.array([1]),
        stripped,
    )
    assert bare[0, 0] == pytest.approx(0.5 * q05)
    assert bare[0, 1] == pytest.approx(q95 + 10)


def test_missing_requested_fold_raises_before_miss_rate():
    oof, ranges = _valid_oof()
    oof["fold_id"] = 4
    oof["minutes"] = 20.0
    with pytest.raises(ValueError, match="fold"):
        build_tail_tables(oof, folds=[1, 2, 3, 4], fold_ranges=ranges)


def test_miss_rate_outside_band_raises():
    oof, ranges = _valid_oof()
    oof["minutes"] = 1.0
    with pytest.raises(ValueError, match="miss rate"):
        build_tail_tables(oof, folds=[1], fold_ranges=ranges)


def test_group_missing_from_the_data_raises_at_build():
    oof, ranges = _valid_oof()
    oof = oof.loc[oof["starting"] == 1].reset_index(drop=True)
    oof["minutes"] = 20.0
    oof.loc[0, "minutes"] = 5.0
    oof.loc[1, "minutes"] = 40.0
    with pytest.raises(ValueError, match="pin"):
        build_tail_tables(oof, folds=[1], fold_ranges=ranges)


def test_builder_rejects_a_non_finite_knot():
    oof, ranges = _valid_oof()
    oof.loc[0, "q_0.50"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_tail_tables(oof, folds=[1], fold_ranges=ranges)


def test_builder_rejects_bad_rows_and_rules():
    oof, ranges = _valid_oof()
    non_positive = oof.copy()
    non_positive.loc[0, "minutes"] = 0
    with pytest.raises(ValueError, match="y > 0"):
        build_tail_tables(non_positive, folds=[1], fold_ranges=ranges)
    late = oof.copy()
    late["game_date"] = HOLDOUT_START
    with pytest.raises(ValueError, match="2025-10-21"):
        build_tail_tables(late, folds=[1], fold_ranges=ranges)
    optimistic = oof.copy()
    optimistic["early_stop"] = "validation"
    with pytest.raises(ValueError, match="train_tail"):
        build_tail_tables(optimistic, folds=[1], fold_ranges=ranges)
    bad_role = oof.copy()
    bad_role.loc[0, "starting"] = 2
    with pytest.raises(ValueError, match="starting"):
        build_tail_tables(bad_role, folds=[1], fold_ranges=ranges)
    with pytest.raises(ValueError, match="grouping"):
        build_tail_tables(
            oof,
            folds=[1],
            fold_ranges=ranges,
            grouping={"lower": "q50_tier", "upper": "starting"},
        )
```

Update `_tables()` from Task 1 so `MinuteTailTables(arrays=...)` still works. Give every new dataclass field a default.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q -k "builder or missing_requested or miss_rate or group_missing or builder_rejects"`

Expected: FAIL with `ImportError` for `build_tail_tables`.

- [ ] **Step 3: Implement the builder**

Extend `MinuteTailTables` with defaults:

```python
@dataclass
class MinuteTailTables:
    arrays: dict[tuple[str, int], np.ndarray]
    floor: float = KNOT_FLOOR
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS
    early_stop: str = "train_tail"
    train_tail_frac: float = 0.10
    holdout_start: str = "2025-10-21"
    grouping: dict[str, str] | None = None
    groups: tuple[int, ...] = (0, 1)
    folds: tuple[int, ...] = ()
    fold_ranges: dict | None = None
    oof: object | None = None
```

Add `HOLDOUT_START = np.datetime64("2025-10-21")` and `build_tail_tables`. Rules, in order:

1. Default grouping is `{"lower": "starting", "upper": "starting"}`. Any other rule name raises `ValueError("unknown grouping")`.
2. `early_stop` must be `"train_tail"` and `train_tail_frac` must be `0.10` on every row. Otherwise raise `ValueError` mentioning `train_tail`.
3. If any requested fold id is absent, raise `ValueError` mentioning `fold` before counting misses.
4. `starting` must be numeric 0 or 1. Otherwise raise `ValueError` mentioning `starting`.
5. `minutes > 0`. Otherwise raise `ValueError("y > 0")`.
6. `game_date < HOLDOUT_START`. Otherwise raise `ValueError` mentioning `2025-10-21`.
7. Prep the 11 raw knots with `prepare_quantile_grid`.
8. For each requested fold, lower-tail rate `mean(y < q05)` and upper-tail rate `mean(y > q95)` must sit in `[0.04, 0.06]`. Otherwise raise `ValueError` mentioning `miss rate`.
9. Allocate `(lower, 0)`, `(lower, 1)`, `(upper, 0)`, `(upper, 1)` as empty lists. Append `y / q05` when `y < q05`, and `y - q95` when `y > q95`, to that row's group.
10. Append pin `1` to each lower list and `0` to each upper list, sort, and raise `ValueError` mentioning `pin` if a list has length 1.

Return a `MinuteTailTables` whose `folds` are the requested ids in order and whose `oof` is the selected rows.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/shared/minutes_sampler.py models/shared/__init__.py tests/models/test_minutes_sampler.py
git commit -m "$(cat <<'EOF'
Build minutes tail tables from out-of-fold misses.

The builder keeps both starting groups, pins the knots, and rejects a fold that is missing or whose tail rate is off 5%.
EOF
)"
```

---

### Task 3: Honest walk-forward predictions

**Files:**
- Modify: `models/shared/train.py`
- Modify: `tests/models/test_minutes_sampler.py`

**Interfaces:**
- Consumes: `date_walk_forward_folds` in `models/shared/splits.py`
- Produces:
  - `fit_quantile_models(..., early_stop="validation", train_dates=None, train_tail_frac=0.10, X_predict=None)`
  - `run_walk_forward(..., early_stop="validation", train_tail_frac=0.10)` adds return keys `oof` and `fold_ranges`
  - Default `early_stop="validation"` still fits and predicts the validation rows, so existing callers keep their numbers

- [ ] **Step 1: Write the failing tests**

Append:

```python
from models.shared import train as train_mod
from models.shared.train import fit_quantile_models, run_walk_forward


class _FakeBooster:
    def __init__(self, **kwargs):
        self.eval_rows = None

    def fit(self, X, y, eval_set=None, verbose=False):
        self.eval_rows = len(eval_set[0][0])

    def predict(self, X):
        return np.arange(len(X), dtype=float)


def test_train_tail_one_date_raises_before_any_booster(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("booster constructed")

    monkeypatch.setattr(train_mod, "XGBRegressor", boom)
    X = pd.DataFrame({"a": [1.0]})
    y = pd.Series([2.0])
    with pytest.raises(ValueError, match="two training dates"):
        fit_quantile_models(
            X, y, X, y,
            xgb_params={"n_estimators": 1},
            quantiles=[0.5],
            early_stop="train_tail",
            train_dates=pd.to_datetime(["2024-01-01"]),
        )


def test_train_tail_walk_forward_stops_on_training_dates_and_keeps_every_fold(monkeypatch):
    monkeypatch.setattr(train_mod, "XGBRegressor", _FakeBooster)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for day in dates:
        for player in (1, 2):
            rows.append({"game_date": day, "a": 1.0, "minutes": 10.0, "starting": player % 2})
    frame = pd.DataFrame(rows)
    X = frame[["a"]]
    y = frame["minutes"]
    result = run_walk_forward(
        X, y, frame,
        xgb_params={"n_estimators": 1},
        quantiles=[0.5],
        n_folds=1,
        train_frac=0.5,
        step_frac=0.2,
        early_stop="train_tail",
    )
    oof = result["oof"]
    assert set(oof["fold_id"]) == {1}
    assert (oof["early_stop"] == "train_tail").all()
    assert 1 in result["fold_ranges"]
    assert "q_0.50" in oof.columns
    assert len(oof) == int(frame["game_date"].isin(frame["game_date"].unique()[5:7]).sum())
    assert result["models_last"]["q_0.50"].eval_rows < 5
```

`date_walk_forward_folds` with 10 unique dates, `train_frac=0.5`, `step_frac=0.2` yields one fold: train dates `0:5`, validation dates `5:7`. The OOF length assertion uses that split. `fit_quantile_models` stores boosters under `q_0.50`. The fake booster's `eval_rows` must be less than 5, because early stopping uses only the last 10% of those five training dates (at least one date, not all five).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q -k "train_tail"`

Expected: FAIL with `TypeError` because `early_stop` is not a parameter yet.

- [ ] **Step 3: Implement the stopping split**

In `fit_quantile_models`, add keyword-only `early_stop="validation"`, `train_dates=None`, `train_tail_frac=0.10`, and `X_predict=None`.

When `early_stop == "train_tail"`:

- If `train_dates` has fewer than two unique values, raise `ValueError("train_tail requires at least two training dates")` before reading `xgb_params` and before constructing `XGBRegressor`.
- `n_stop = int(np.floor(n_unique * train_tail_frac))`. If `n_stop < 1`, set it to 1. If `n_stop >= n_unique`, raise `ValueError`.
- The last `n_stop` unique dates are the early-stopping rows. The earlier unique dates are the fit rows. A whole date stays on one side.
- Predict `X_predict`.

When `early_stop == "validation"`, keep today's behavior: fit `X_train`, `eval_set` is `(X_eval, y_eval)`, predict `X_eval` unless `X_predict` is passed.

In `run_walk_forward`, add the same `early_stop` and `train_tail_frac` arguments, defaulting to `"validation"` and `0.10`. On every fold, append the validation rows to `oof` with raw prediction columns, `fold_id`, `game_date`, `minutes`, `starting`, `early_stop`, and `train_tail_frac`. Fill `fold_ranges` from that fold's first and last train date and first and last validation date. For `"train_tail"`, pass the fold's training frame and `train_dates` into `fit_quantile_models` and set `X_predict` to the validation features. Return `oof` and `fold_ranges` in addition to the existing keys. Do not drop `preds_last`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py tests/models -q -k "train_tail or minutes_sampler"`

Expected: PASS. The old residual-bootstrap tests are gone with `src/models/xgboost_models/`.

- [ ] **Step 5: Commit**

```bash
git add models/shared/train.py tests/models/test_minutes_sampler.py
git commit -m "$(cat <<'EOF'
Keep honest walk-forward quantile predictions.

Tail builds can early-stop on the last 10% of training dates, and every validation fold retains its raw knots.
EOF
)"
```

---

### Task 4: Sidecar and draw-time groups

**Files:**
- Modify: `models/shared/minutes_sampler.py`
- Modify: `tests/models/test_minutes_sampler.py`
- Modify: `models/shared/__init__.py`

**Interfaces:**
- Consumes: `MinuteTailTables`, `build_tail_tables`
- Produces:
  - `save_tail_sidecar(tables, path) -> Path`
  - `load_tail_sidecar(path) -> MinuteTailTables`
  - `groups_for_frame(starting, tables) -> tuple[np.ndarray, np.ndarray]` named `(lower_groups, upper_groups)`

- [ ] **Step 1: Write the failing tests**

```python
from models.shared.minutes_sampler import groups_for_frame, load_tail_sidecar, save_tail_sidecar


def _four_fold_tables():
    frames = []
    ranges = {}
    for fold in (1, 2, 3, 4):
        oof, one_range = _valid_oof()
        oof["fold_id"] = fold
        oof["game_date"] = pd.Timestamp("2024-01-01") + pd.Timedelta(days=fold)
        frames.append(oof)
        ranges[fold] = {
            "train_start": "2023-10-01",
            "train_end": f"2024-01-0{fold}",
            "val_start": f"2024-01-1{fold}",
            "val_end": f"2024-01-2{fold}",
        }
    oof = pd.concat(frames, ignore_index=True)
    return build_tail_tables(oof, folds=[1, 2, 3, 4], fold_ranges=ranges)


def test_loader_accepts_folds_1_through_4_only(tmp_path):
    tables = _four_fold_tables()
    path = save_tail_sidecar(tables, tmp_path / "tails.joblib")
    loaded = load_tail_sidecar(path)
    assert loaded.folds == (1, 2, 3, 4)
    partial, ranges = _valid_oof()
    partial_tables = build_tail_tables(partial, folds=[1], fold_ranges=ranges)
    partial_path = save_tail_sidecar(partial_tables, tmp_path / "partial.joblib")
    with pytest.raises(ValueError, match="fold"):
        load_tail_sidecar(partial_path)


def test_loader_rejects_a_different_floor_or_level_list(tmp_path):
    tables = _four_fold_tables()
    path = save_tail_sidecar(tables, tmp_path / "tails.joblib")
    payload = __import__("joblib").load(path)
    payload["floor"] = 0.01
    __import__("joblib").dump(payload, path)
    with pytest.raises(ValueError, match="floor"):
        load_tail_sidecar(path)
    payload = __import__("joblib").load(save_tail_sidecar(tables, tmp_path / "levels.joblib"))
    payload["quantile_levels"] = list(np.linspace(0.05, 0.95, 11))
    __import__("joblib").dump(payload, tmp_path / "levels.joblib")
    with pytest.raises(ValueError, match="quantile"):
        load_tail_sidecar(tmp_path / "levels.joblib")


def test_missing_starting_raises_at_draw_time_and_is_not_bench():
    tables = _four_fold_tables()
    with pytest.raises(ValueError, match="starting"):
        groups_for_frame(pd.Series([pd.NA, 1]), tables)
    lower, upper = groups_for_frame(pd.Series([0, 1]), tables)
    np.testing.assert_array_equal(lower, [0, 1])
    np.testing.assert_array_equal(upper, [0, 1])
    tables.grouping = {"lower": "q50_tier", "upper": "starting"}
    with pytest.raises(ValueError, match="grouping"):
        groups_for_frame(pd.Series([0, 1]), tables)
```

`save_tail_sidecar` may write a one-fold payload. `load_tail_sidecar` is what rejects it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q -k "loader or missing_starting"`

Expected: FAIL with `ImportError` for `save_tail_sidecar`.

- [ ] **Step 3: Implement save, load, and group assignment**

`save_tail_sidecar` writes a dict with `arrays`, `floor`, `quantile_levels`, `early_stop`, `train_tail_frac`, `holdout_start`, `grouping`, `groups`, `folds`, `fold_ranges`, and `oof`. It does not require four folds.

`load_tail_sidecar` raises unless:

- `floor == KNOT_FLOOR`
- `tuple(quantile_levels) == QUANTILE_LEVELS`
- `tuple(folds) == (1, 2, 3, 4)`
- both grouping values equal `"starting"`

`groups_for_frame` reads `tables.grouping`. An unknown rule raises `ValueError` mentioning `grouping`. For `"starting"`, coerce with `pd.to_numeric(..., errors="coerce")` and raise `ValueError` mentioning `starting` if any value is missing or not in `{0, 1}`. Return the same integer array for both tails today.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/shared/minutes_sampler.py models/shared/__init__.py tests/models/test_minutes_sampler.py
git commit -m "$(cat <<'EOF'
Load minutes tail sidecars only when the rules match.

Draw-time starting must be 0 or 1, and a saved table set has to come from folds 1 through 4.
EOF
)"
```

---

### Task 5: Draws for later combination

**Files:**
- Modify: `models/shared/minutes_sampler.py`
- Modify: `tests/models/test_minutes_sampler.py`
- Modify: `models/shared/__init__.py`

**Interfaces:**
- Consumes: `quantile_minutes`, `canonical` ids
- Produces:
  - `canonical_id(value) -> str`
  - `sample_minutes(grids, lower_groups, upper_groups, player_ids, game_ids, tables, *, seed=42, draws=10_000) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
from models.shared.minutes_sampler import canonical_id, sample_minutes


def test_ids_strip_before_digit_check_and_share_a_draw_vector():
    assert canonical_id(" 0021900001 ") == canonical_id(21900001) == "21900001"
    grids = np.vstack([SORTED, SORTED + 15])
    groups = np.array([1, 1])
    kwargs = dict(
        grids=grids,
        lower_groups=groups,
        upper_groups=groups,
        tables=TABLES,
        seed=42,
        draws=32,
    )
    padded = sample_minutes(player_ids=["p", "p"], game_ids=[" 0021900001 ", "g2"], **kwargs)
    plain = sample_minutes(player_ids=["p", "p"], game_ids=[21900001, "g2"], **kwargs)
    np.testing.assert_allclose(padded, plain)


def test_shuffle_keeps_player_games_and_streams_are_not_identical():
    grids = np.vstack([SORTED, SORTED + 15])
    groups = np.array([1, 0])
    first = sample_minutes(grids, groups, groups, ["a", "b"], ["g1", "g2"], TABLES, seed=7, draws=10_000)
    second = sample_minutes(grids[::-1], groups[::-1], groups[::-1], ["b", "a"], ["g2", "g1"], TABLES, seed=7, draws=10_000)
    np.testing.assert_allclose(first[0], second[1])
    np.testing.assert_allclose(first[1], second[0])
    assert abs(np.corrcoef(first[0], first[1])[0, 1]) < 0.05
    assert not np.allclose(first[0], first[1])
    labeled = sample_minutes(grids[:1], groups[:1], groups[:1], ["a"], ["g1"], TABLES, seed=7, draws=64)
    from hashlib import sha256
    material = f"7|{canonical_id('a')}|{canonical_id('g1')}".encode()
    unlabeled_seed = int.from_bytes(sha256(material).digest()[:8], "little")
    unlabeled = np.random.default_rng(unlabeled_seed).random(64)
    assert not np.allclose(labeled[0], unlabeled)


def test_median_draw_tracks_each_rows_q50_and_missing_id_raises():
    low = SORTED.copy()
    high = SORTED.copy()
    high[:] = SORTED + 15
    grids = np.vstack([low, high])
    groups = np.array([1, 1])
    draws = sample_minutes(grids, groups, groups, ["a", "b"], ["g1", "g2"], TABLES, seed=3, draws=10_000)
    assert abs(np.median(draws[0]) - low[5]) < abs(np.median(draws[0]) - high[5])
    assert abs(np.median(draws[1]) - high[5]) < abs(np.median(draws[1]) - low[5])
    with pytest.raises(ValueError, match="id"):
        sample_minutes(grids[:1], groups[:1], groups[:1], [None], ["g1"], TABLES, seed=1, draws=4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q -k "ids_strip or shuffle_keeps or median_draw"`

Expected: FAIL with `ImportError` for `sample_minutes`.

- [ ] **Step 3: Implement canonical ids and sample_minutes**

`canonical_id` strips strings first. A missing value (`None`, NA, blank, or a NaN float) raises `ValueError` mentioning `id`. After stripping, an integer, an integer-valued float, or a string matching `^[+-]?\d+$` becomes `str(int(value))` with no leading zeros. Every other stripped string is returned as that string.

`sample_minutes` builds one generator per row from SHA-256 of `f"{seed}|minutes|{canonical_id(player_id)}|{canonical_id(game_id)}"`, using the first 8 bytes as the NumPy seed, matching the byte order in `JointPointsSimulator._rng`. Draw `i` is the `i`-th `Generator.random()` value. Stack those uniforms into shape `(rows, draws)` and call `quantile_minutes`. Do not implement `Q` inside `sample_minutes`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: PASS. The 10,000-draw correlation test should be stable at seed 7. If it fails once, do not loosen the 0.05 cutoff; check that the two rows are not sharing a generator.

- [ ] **Step 5: Commit**

```bash
git add models/shared/minutes_sampler.py models/shared/__init__.py tests/models/test_minutes_sampler.py
git commit -m "$(cat <<'EOF'
Sample minutes on a separate random stream.

Canonical player and game ids keep draws stable across row order without matching the points simulator's uniforms.
EOF
)"
```

---

### Task 6: Price a line from Q, and score Q with CRPS

**Files:**
- Modify: `models/shared/minutes_sampler.py`
- Modify: `tests/models/test_minutes_sampler.py`
- Modify: `models/shared/__init__.py`

**Interfaces:**
- Consumes: `quantile_minutes`, `prepare_quantile_grid`, `empirical_quantile`, `QUANTILE_LEVELS`
- Produces:
  - `probability_below_line(grids, lower_groups, upper_groups, tables, line) -> np.ndarray` shape `(rows,)`
  - `row_crps(y, q_at_u, u) -> np.ndarray` shape `(rows,)`
  - `tail_bin_shares(y, edge_values, *, tail) -> pd.DataFrame` with columns `bin`, `share_of_rows`, `share_of_misses`

- [ ] **Step 1: Write the failing tests**

```python
from models.shared.metrics import pinball_loss
from models.shared.minutes_sampler import probability_below_line, row_crps, tail_bin_shares


def test_probability_below_line_inverts_q_without_draws():
    grids = _grid(SORTED)
    groups = np.array([1])
    at_median = probability_below_line(grids, groups, groups, TABLES, line=20)
    assert at_median[0] == pytest.approx(0.50)
    # Group 1 lower table starts at 0.5, so Q(0) = 0.5 * q05 = 5.
    assert probability_below_line(grids, groups, groups, TABLES, line=5)[0] == 0
    assert probability_below_line(grids, groups, groups, TABLES, line=80)[0] == 1


def test_row_crps_at_the_eleven_levels_is_twice_mean_pinball():
    y = np.array([21.0])
    u = np.asarray(QUANTILE_LEVELS, dtype=float)
    q = np.broadcast_to(SORTED, (1, 11)).copy()
    got = row_crps(y, q, u)
    manual = [pinball_loss(y, np.array([knot]), level) for level, knot in zip(u, SORTED)]
    assert got[0] == pytest.approx(2 * np.mean(manual))


def test_tail_bin_shares_are_a_fraction_of_misses():
    y = np.array([1, 2, 3, 4, 5, 20, 20, 20, 20, 20], dtype=float)
    edges = np.array([1.5, 2.5, 3.5, 4.5, 6.0])
    report = tail_bin_shares(y, np.broadcast_to(edges, (10, 5)), tail="lower")
    assert report["share_of_misses"].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
    assert report["share_of_rows"].sum() == pytest.approx(0.5)
```

For the lower tail, `edge_values` columns are `Q(0.01) ... Q(0.05)`. Misses are `y < edge_values[:, -1]`. Bin 1 is `y < edge0`. Bin `k` for `k > 1` is `edge[k-2] <= y < edge[k-1]`. `share_of_misses` divides by the miss count. `share_of_rows` divides by all rows. Add this upper-tail test in the same step. Edges are `q95, Q(0.96), Q(0.97), Q(0.98), Q(0.99)`. Misses are `y > q95`. One held-out interior row keeps the miss share at 0.2:

```python
def test_upper_tail_bin_shares_are_a_fraction_of_misses():
    y = np.array([31, 33, 35, 37, 39, 20], dtype=float)
    edges = np.array([30, 32, 34, 36, 38], dtype=float)
    report = tail_bin_shares(y, np.broadcast_to(edges, (6, 5)), tail="upper")
    assert report["share_of_misses"].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q -k "probability_below or row_crps or tail_bin"`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement inversion, CRPS, and bin shares**

`probability_below_line` builds breakpoints for one row:

- Lower tail: `u = linspace(0, 1, n) * 0.05` excluding the right endpoint `0.05`, values `q05 * empirical_quantile(linspace without 1, lower_table)`
- Middle: `u = QUANTILE_LEVELS`, values = prepared knots
- Upper tail: `u = 0.95 + linspace(0, 1, n) * 0.05` excluding the left endpoint `0.95`, values `q95 + empirical_quantile(linspace without 0, upper_table)`

If `line <= breakpoints_q[0]`, return `0`. If `line > breakpoints_q[-1]`, return `1`. Otherwise return the linear `u` on the piece where `q` crosses `line`, which is `inf{u : Q(u) >= line}`.

`row_crps` uses one observation's pinball, the same formula as `pinball_loss`: residual `y - q`, then `alpha * residual` when residual is nonnegative and `(alpha - 1) * residual` otherwise. `CRPS_i = 2 * mean over u`. Do not average across rows inside the function.

`tail_bin_shares` returns one row per bin with `share_of_rows` and `share_of_misses`. An empty miss set yields NaN miss shares rather than a divide by zero.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/shared/minutes_sampler.py models/shared/__init__.py tests/models/test_minutes_sampler.py
git commit -m "$(cat <<'EOF'
Price minutes lines by inverting Q.

CRPS and tail-bin shares score that same quantile function, so a line does not depend on draw noise.
EOF
)"
```

---

### Task 7: Notebook record

**Files:**
- Modify: `notebooks/nba/minutes/min_nba_model.ipynb`
- Test: the functions from Task 6. This task does not refit the model.

**Interfaces:**
- Consumes: `build_tail_tables`, `quantile_minutes`, `row_crps`, `tail_bin_shares`, `save_tail_sidecar`, `run_walk_forward`, `QUANTILE_LEVELS`, `prepare_quantile_grid`
- Produces: three new cells at the end of `notebooks/nba/minutes/min_nba_model.ipynb`, and later the sidecar `models/saved_models/min_nba_tails_2026-04-12.joblib` when a person runs the refit cell

- [ ] **Step 1: Add the cells**

Do not execute the refit. It fits 11 quantile models on four folds.

Cell A, markdown: fold 1–2 tail shape is a diagnostic. The sidecar is folds 1–4. The 2025-26 numbers are a record and are not a reason to edit tables, grouping, the floor, or the stopping setting.

Cell B, code, writes the diagnostic and the sidecar. It assumes `ppm_df`, `X`, `y`, `MIN_FEATURES`, `QUANTILES`, and `XGB_PARAMS` already exist in the notebook. Call:

```python
tail_wf = run_walk_forward(
    X, y, ppm_df,
    xgb_params=XGB_PARAMS,
    quantiles=QUANTILES,
    early_stop="train_tail",
    train_tail_frac=0.10,
)
diagnostic = build_tail_tables(
    tail_wf["oof"],
    folds=[1, 2],
    fold_ranges=tail_wf["fold_ranges"],
)
sidecar_tables = build_tail_tables(
    tail_wf["oof"],
    folds=[1, 2, 3, 4],
    fold_ranges=tail_wf["fold_ranges"],
)
save_tail_sidecar(
    sidecar_tables,
    ROOT / "models/saved_models/min_nba_tails_2026-04-12.joblib",
)
```

Then score folds 3–4 only, in memory, with `diagnostic`. For a slice, build `u` of shape `(rows, 5)` at `0.01, 0.02, 0.03, 0.04, 0.05` and at `0.95, 0.96, 0.97, 0.98, 0.99`. Pass those through `quantile_minutes` with groups from `groups_for_frame`. Call `tail_bin_shares`. Repeat for `starting == 0`, `starting == 1`, and predicted-q50 tiers `<15`, `15-24`, `24-31`, `31+` using the prepared `q_0.50` on those rows. Do not cross the two splits.

Interval: resample `game_date` with replacement 2,000 times, the same cluster weights as `clustered_mae_bootstrap` in this notebook, and take the 2.5 and 97.5 percentiles of each share. Print both `share_of_rows` and `share_of_misses`. The shape under test is `share_of_misses`, with a target near 20% per bin.

Cell C, code, scores 2025-26 from `preds_ho` and `sidecar_tables`. Do not read `preds_ho_live`.

```python
levels = np.asarray(QUANTILE_LEVELS)
grids = np.column_stack([preds_ho[f"q_{level:.2f}"] for level in levels])
u11 = np.broadcast_to(levels, (len(grids), 11))
groups = groups_for_frame(ppm_holdout["starting"], sidecar_tables)
q11 = quantile_minutes(u11, grids, groups[0], groups[1], sidecar_tables)
eleven = float(np.mean(row_crps(np.asarray(y_ho), q11, levels)))
```

Compare `eleven` with `3.092`. Also compute the same mean from raw `grids` passed straight to `row_crps` with no prep. The two means may differ only on rows where `prepare_quantile_grid(grids)` changed a knot. If the gap is larger than the mean absolute row-CRPS gap on those rows, raise `RuntimeError` and stop. Do not edit the sidecar.

Then the 999-point CRPS on `u = 0.001, ..., 0.999`, plus the same bin report as cell B on the holdout, using `sidecar_tables`. Print the number. Leave the tables unchanged.

- [ ] **Step 2: Check the notebook calls the tested functions**

Run: `.venv/bin/python -m pytest tests/models/test_minutes_sampler.py -q`

Expected: PASS. Do not start the walk-forward refit from pytest.

- [ ] **Step 3: Commit the notebook cells**

```bash
git add notebooks/nba/minutes/min_nba_model.ipynb
git commit -m "$(cat <<'EOF'
Add the minutes tail-shape report and the 2025-26 CRPS record.

The notebook scores Q directly and writes the four-fold sidecar only after a train-tail walk-forward.
EOF
)"
```

The sidecar file appears only after the refit cell is run by hand. Do not commit that joblib unless it is produced and the user asks. `models/saved_models/` is the existing artifact location; leave the frozen model joblib untouched.
