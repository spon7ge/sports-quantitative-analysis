# MLB Strikeout Walk-Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a chronological workload-offset and negative-binomial strikeout walk-forward that stores immutable pregame snapshots and scores 2021–2024, with 2025 held out.

**Architecture:** A checked-in regular-season calendar assigns 28-day blocks. Features for a start use only eligible starts on earlier official dates. Each 2019-and-later start stores one snapshot. Later fits read that snapshot. Coefficients are refit every block. `m`, `κ`, and the rest cap are chosen on 2020 and then frozen. Quantiles, log-likelihood, and CRPS use the analytical NB2 distribution. The existing year-fold backtest stays in place.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, statsmodels via the existing `fit_nb2`, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-27-mlb-strikeout-walk-forward-design.md`
- History rule: `prior.game_date < current.game_date`. Same-day results are never used.
- `scheduled_start_utc` orders different dates and supplies the rest gap. It does not admit a same-day start.
- Block training rows: `game_date` strictly before the block’s first `game_date`.
- Within a block, an earlier official date may update raw sums and the stored batters-faced offset for a later start. Coefficients, scaling, league priors, population means, the rest median, and locked settings stay fixed.
- One snapshot per `(pitcher_id, game_pk)`. A second write raises, including a write of the same value.
- Hard offset: `log(max(predicted_bf_oof, 1e-6))`. Current-game batters faced, outs, and pitches are not features and are not the offset.
- `m` grid: `25, 50, 75, 100, 150`. `κ` grid: `1, 2, 3, 4, 6`. Rest-cap grid: `14, 21, 28, 35`.
- 2018 is bootstrap only. 2019 stores workload offsets. 2020 tunes. 2021–2024 is the report. 2025 is the holdout. 2026 is excluded.
- L2 stays `workload_l2 = 1.0` and `strikeout_l2 = 2.0`.
- Parameter-count fallback `α = 1e-4` fails the block. Method-of-moments `α` is allowed.
- Persisted strikeout mass is `negative_binomial_pmf` with `k_max = 15`. Scoring does not read that matrix.
- Analytical NB2: `r = 1 / α`, `p = 1 / (1 + α μ)`.
- CRPS continues until survival probability `< 1e-10`.
- Lines: `4.5, 5.5, 6.5`. Quantiles: `0.10, 0.25, 0.50, 0.75, 0.90`.
- First-start rest is the median of capped observed rest under the candidate cap.
- Closed integer intervals may cover more than their nominal probability. Report both. Do not flag the excess by itself.
- Calibration-in-the-large is the mean predicted over-probability versus the observed over rate. Aggregate reliability bins are `[0.0, 0.1), …, [0.9, 1.0]`. Omit a bin with fewer than 200 rows.
- Do not add Poisson XGBoost, quantile trees, K/9, `season_year`, categorical `opponent_team_id`, or a free batters-faced elasticity.
- Do not change `config/mlb.yaml` folds or `python -m src.mlb backtest`.
- Tests use synthetic frames. No network.

---

### Task 1: Regular-season calendar

**Files:**
- Create: `src/mlb/evaluation/walk_forward_calendar.py`
- Create: `src/mlb/data/regular_season_calendar.csv`
- Test: `tests/mlb/evaluation/test_walk_forward_calendar.py`

**Interfaces:**
- Consumes: schedule rows with `game_type`, `abstract_game_state`, `official_date`, `season`
- Produces: `load_regular_season_calendar(path) -> pd.DataFrame`, `validate_regular_season_calendar(frame, seasons=range(2018, 2026)) -> None`, `calendar_from_schedule(schedule) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
import pytest

from src.mlb.evaluation.walk_forward_calendar import (
    calendar_from_schedule,
    validate_regular_season_calendar,
)


def test_calendar_keeps_final_regular_season_games_only():
    schedule = pd.DataFrame(
        [
            {"season": 2018, "official_date": "2018-03-29", "game_type": "R", "abstract_game_state": "Final"},
            {"season": 2018, "official_date": "2018-03-28", "game_type": "S", "abstract_game_state": "Final"},
            {"season": 2018, "official_date": "2018-10-05", "game_type": "F", "abstract_game_state": "Final"},
            {"season": 2018, "official_date": "2018-09-30", "game_type": "R", "abstract_game_state": "Final"},
            {"season": 2019, "official_date": "2019-03-20", "game_type": "R", "abstract_game_state": "Final"},
            {"season": 2019, "official_date": "2019-09-29", "game_type": "R", "abstract_game_state": "Final"},
        ]
    )
    calendar = calendar_from_schedule(schedule)
    row_2018 = calendar.loc[calendar["season"] == 2018].iloc[0]
    assert row_2018["regular_season_open_date"] == "2018-03-29"
    assert row_2018["regular_season_close_date"] == "2018-09-30"


def test_calendar_rejects_a_missing_season():
    frame = pd.DataFrame(
        {
            "season": [2018],
            "regular_season_open_date": ["2018-03-29"],
            "regular_season_close_date": ["2018-09-30"],
        }
    )
    with pytest.raises(ValueError, match="2019"):
        validate_regular_season_calendar(frame, seasons=range(2018, 2020))


def test_calendar_rejects_open_after_close():
    frame = pd.DataFrame(
        {
            "season": list(range(2018, 2026)),
            "regular_season_open_date": ["2018-04-02"] + ["2019-03-20"] * 7,
            "regular_season_close_date": ["2018-04-01"] + ["2019-09-29"] * 7,
        }
    )
    with pytest.raises(ValueError, match="open"):
        validate_regular_season_calendar(frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/evaluation/test_walk_forward_calendar.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write the calendar module**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

CALENDAR_COLUMNS = (
    "season",
    "regular_season_open_date",
    "regular_season_close_date",
)


def calendar_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    frame = schedule.copy()
    regular = frame["game_type"].astype(str).eq("R")
    final = frame["abstract_game_state"].astype(str).isin({"Final", "Completed"})
    kept = frame.loc[regular & final].copy()
    if kept.empty:
        return pd.DataFrame(columns=list(CALENDAR_COLUMNS))
    kept["official_date"] = kept["official_date"].astype(str)
    grouped = kept.groupby("season", as_index=False).agg(
        regular_season_open_date=("official_date", "min"),
        regular_season_close_date=("official_date", "max"),
    )
    grouped["season"] = grouped["season"].astype(int)
    return grouped.loc[:, list(CALENDAR_COLUMNS)].sort_values("season")


def validate_regular_season_calendar(
    frame: pd.DataFrame,
    seasons: range = range(2018, 2026),
) -> None:
    required = set(seasons)
    present = set(int(s) for s in frame["season"])
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"calendar missing seasons: {missing}")
    if frame["season"].duplicated().any():
        raise ValueError("calendar seasons must be unique")
    opened = pd.to_datetime(frame["regular_season_open_date"], errors="coerce")
    closed = pd.to_datetime(frame["regular_season_close_date"], errors="coerce")
    if opened.isna().any() or closed.isna().any():
        raise ValueError("calendar dates must parse")
    if (opened > closed).any():
        raise ValueError("calendar open must be on or before close")


def load_regular_season_calendar(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in CALENDAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"calendar missing columns: {missing}")
    out = frame.loc[:, list(CALENDAR_COLUMNS)].copy()
    out["season"] = out["season"].astype(int)
    out["regular_season_open_date"] = out["regular_season_open_date"].astype(str)
    out["regular_season_close_date"] = out["regular_season_close_date"].astype(str)
    validate_regular_season_calendar(out)
    return out
```

Build `src/mlb/data/regular_season_calendar.csv` with `calendar_from_schedule` on MLB Stats API schedule payloads. Keep `gameType == "R"` and a final abstract state. Do not hand-type the dates. `load_regular_season_calendar` on that file must succeed for 2018–2025.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/evaluation/test_walk_forward_calendar.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/evaluation/walk_forward_calendar.py src/mlb/data/regular_season_calendar.csv tests/mlb/evaluation/test_walk_forward_calendar.py
git commit -m "$(cat <<'EOF'
Add the regular-season calendar used by strikeout walk-forward blocks.

EOF
)"
```

### Task 2: Season-relative 28-day blocks

**Files:**
- Create: `src/mlb/evaluation/walk_forward_blocks.py`
- Test: `tests/mlb/evaluation/test_walk_forward_blocks.py`

**Interfaces:**
- Consumes: `regular_season_open_date`, `regular_season_close_date` from Task 1
- Produces: `assign_walk_forward_blocks(starts, calendar) -> pd.DataFrame` with `model_eligible` and `walk_forward_block`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from src.mlb.evaluation.walk_forward_blocks import assign_walk_forward_blocks


def test_blocks_reset_each_season_and_drop_postseason():
    calendar = pd.DataFrame(
        {
            "season": [2019, 2020],
            "regular_season_open_date": ["2019-03-28", "2020-07-23"],
            "regular_season_close_date": ["2019-04-30", "2020-09-27"],
        }
    )
    starts = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 1, 1],
            "game_pk": [10, 11, 12, 13],
            "season": [2019, 2019, 2019, 2020],
            "game_date": ["2019-03-28", "2019-04-25", "2019-05-02", "2020-07-23"],
        }
    )
    out = assign_walk_forward_blocks(starts, calendar)
    assert out.loc[0, "model_eligible"]
    assert out.loc[0, "walk_forward_block"] == "2019-0"
    assert out.loc[1, "walk_forward_block"] == "2019-1"
    assert not out.loc[2, "model_eligible"]
    assert out.loc[3, "walk_forward_block"] == "2020-0"
    assert out.loc[out["game_date"] == "2019-03-28", "walk_forward_block"].nunique() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/evaluation/test_walk_forward_blocks.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the block assignment**

```python
from __future__ import annotations

import pandas as pd


def assign_walk_forward_blocks(
    starts: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    out = starts.copy()
    cal = calendar.copy()
    cal["season"] = cal["season"].astype(int)
    out = out.merge(cal, on="season", how="left")
    game_date = pd.to_datetime(out["game_date"])
    opened = pd.to_datetime(out["regular_season_open_date"])
    closed = pd.to_datetime(out["regular_season_close_date"])
    out["model_eligible"] = (
        opened.notna() & closed.notna() & game_date.ge(opened) & game_date.le(closed)
    )
    day_offset = (game_date - opened).dt.days
    block_index = (day_offset // 28).astype("Int64")
    label = out["season"].astype("Int64").astype(str) + "-" + block_index.astype(str)
    out["walk_forward_block"] = label.where(out["model_eligible"], pd.NA)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/evaluation/test_walk_forward_blocks.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/evaluation/walk_forward_blocks.py tests/mlb/evaluation/test_walk_forward_blocks.py
git commit -m "$(cat <<'EOF'
Assign regular-season starts to season-relative 28-day blocks.

EOF
)"
```

### Task 3: Analytical NB2 scores

**Files:**
- Create: `src/mlb/models/nb_scores.py`
- Test: `tests/mlb/models/test_nb_scores.py`

**Interfaces:**
- Consumes: scipy `nbinom` with `r = 1/α`, `p = 1/(1+αμ)`
- Produces: `nb2_quantile(mu, alpha, level) -> np.ndarray`, `nb2_nll(y, mu, alpha) -> float`, `nb2_crps(y, mu, alpha, survival_limit=1e-10) -> float`, `half_point_probabilities(mu, alpha, line) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from scipy.stats import nbinom

from src.mlb.models.nb_scores import half_point_probabilities, nb2_crps, nb2_nll, nb2_quantile


def test_quantile_can_exceed_15_and_matches_the_cdf():
    mu = np.array([30.0])
    alpha = np.array([0.2])
    q90 = int(nb2_quantile(mu, alpha, 0.90)[0])
    assert q90 > 15
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    assert nbinom.cdf(q90, r, p)[0] >= 0.90
    assert nbinom.cdf(q90 - 1, r, p)[0] < 0.90


def test_nll_uses_the_log_pmf_at_the_observed_count():
    loss = nb2_nll(np.array([4]), np.array([5.0]), np.array([0.5]))
    r = 1.0 / 0.5
    p = 1.0 / (1.0 + 0.5 * 5.0)
    assert loss == pytest_approx(-nbinom.logpmf(4, r, p))


def test_half_point_line_has_no_push():
    over, under = half_point_probabilities(np.array([6.0]), np.array([0.4]), 5.5)
    assert over[0] + under[0] == pytest_approx(1.0)


def pytest_approx(value):
    import pytest
    return pytest.approx(value)
```

Add `test_crps_stops_when_survival_is_negligible`: a one-row call returns a finite float. `y=3`, `mu=3`, `alpha=1e-6` is the Poisson limit (NB2 variance stays near `mu`), so assert that score against an independent Poisson(3) discrete CRPS, not against 0. A separate call with `y=0`, `mu=1e-6`, `alpha=1e-6` is near 0.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/models/test_nb_scores.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the scorer**

```python
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom


def _r_p(mu, alpha):
    mu_arr = np.clip(np.atleast_1d(np.asarray(mu, dtype=float)), 1e-12, None)
    alpha_arr = np.clip(np.atleast_1d(np.asarray(alpha, dtype=float)), 1e-12, None)
    alpha_arr = np.broadcast_to(alpha_arr, mu_arr.shape)
    return 1.0 / alpha_arr, 1.0 / (1.0 + alpha_arr * mu_arr)


def nb2_quantile(mu, alpha, level: float) -> np.ndarray:
    r, p = _r_p(mu, alpha)
    return nbinom.ppf(level, r, p).astype(int)


def nb2_nll(y, mu, alpha) -> float:
    r, p = _r_p(mu, alpha)
    y_arr = np.asarray(y, dtype=int)
    return float(-np.mean(nbinom.logpmf(y_arr, r, p)))


def nb2_crps(y, mu, alpha, survival_limit: float = 1e-10) -> float:
    r, p = _r_p(mu, alpha)
    y_arr = np.atleast_1d(np.asarray(y, dtype=int))
    total = np.zeros(y_arr.shape[0], dtype=float)
    k = 0
    survival = np.ones(y_arr.shape[0], dtype=float)
    while np.any(survival >= survival_limit):
        cdf = nbinom.cdf(k, r, p)
        hit = (y_arr <= k).astype(float)
        total += (cdf - hit) ** 2
        survival = 1.0 - cdf
        k += 1
        if k > 100000:
            raise RuntimeError("CRPS support did not reach the survival limit")
    return float(np.mean(total))


def half_point_probabilities(mu, alpha, line: float):
    if float(line) != int(line) + 0.5:
        raise ValueError("line must be a half point")
    r, p = _r_p(mu, alpha)
    under = nbinom.cdf(int(np.floor(line)), r, p)
    over = 1.0 - under
    return over, under
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/models/test_nb_scores.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/models/nb_scores.py tests/mlb/models/test_nb_scores.py
git commit -m "$(cat <<'EOF'
Score strikeout negative binomials with the analytical distribution.

EOF
)"
```

### Task 4: Reject an unestimated dispersion

**Files:**
- Modify: `src/mlb/models/workload.py` (`Nb2Fit`, `_extract_nb2_params`, `fit_nb2`)
- Create: `src/mlb/evaluation/walk_forward_fit.py`
- Test: `tests/mlb/evaluation/test_walk_forward_fit.py`

**Interfaces:**
- Consumes: `fit_nb2`, `design_matrix`, `compute_standardization`
- Produces: `Nb2Fit.dispersion_estimated: bool`, `fit_walk_forward_nb2(train, target, features, binary, l2, offset=None) -> WalkForwardFit`, `predict_walk_forward_mean(model, frame, offset=None) -> np.ndarray`

`WalkForwardFit` fields: `feature_names`, `coef`, `alpha`, `method`, `centers`, `scales`, `medians`, `dispersion_estimated`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from src.mlb.evaluation.walk_forward_fit import (
    UnestimatedDispersion,
    fit_walk_forward_nb2,
    predict_walk_forward_mean,
)


def test_fit_records_an_estimated_dispersion_and_predicts_positive_means():
    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "home_flag": rng.integers(0, 2, n),
            "rate": rng.normal(0.22, 0.03, n),
            "batters_faced": rng.poisson(22, n),
        }
    )
    model = fit_walk_forward_nb2(
        frame,
        target="batters_faced",
        features=("home_flag", "rate"),
        binary=("home_flag",),
        l2=1.0,
    )
    assert model.dispersion_estimated
    pred = predict_walk_forward_mean(model, frame)
    assert pred.shape == (n,)
    assert np.all(pred > 0)


def test_parameter_count_fallback_raises(monkeypatch):
    from src.mlb.models.workload import Nb2Fit

    def fake_fit(*args, **kwargs):
        return Nb2Fit(
            coef=np.array([1.0]),
            alpha=1e-4,
            cov=None,
            method="glm",
            dispersion_estimated=False,
        )

    monkeypatch.setattr(
        "src.mlb.evaluation.walk_forward_fit.fit_nb2",
        fake_fit,
    )
    frame = pd.DataFrame({"home_flag": [1, 0], "y": [4, 5]})
    with pytest.raises(UnestimatedDispersion):
        fit_walk_forward_nb2(
            frame,
            target="y",
            features=("home_flag",),
            binary=("home_flag",),
            l2=1.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/evaluation/test_walk_forward_fit.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Record dispersion provenance and fit the declared columns**

Add `dispersion_estimated: bool = True` to `Nb2Fit`. Change `_extract_nb2_params` to return a fourth bool: `True` when `params.size == p + 1`, `False` when `params.size == p`. `fit_nb2` stores that bool. The moments return path sets `dispersion_estimated=True`.

`fit_walk_forward_nb2` drops a feature whose training standard deviation is below `1e-12`, records it on the model, and does not median-fill it. Binary columns are not standardized. Other columns use the training mean and standard deviation. The design matrix still includes the intercept through `design_matrix`. Offset, when provided, is passed to `fit_nb2`. After the fit, `dispersion_estimated is False` raises `UnestimatedDispersion`.

`predict_walk_forward_mean` rebuilds the design matrix with the training centers, scales, and medians, adds the offset when present, and returns `exp(X @ coef)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/evaluation/test_walk_forward_fit.py tests/mlb/models/test_glm_fit_contract.py -q`

Expected: PASS. The existing GLM contract tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb/models/workload.py src/mlb/evaluation/walk_forward_fit.py tests/mlb/evaluation/test_walk_forward_fit.py
git commit -m "$(cat <<'EOF'
Fail a walk-forward block when negative-binomial dispersion was not estimated.

EOF
)"
```

### Task 5: Date-strict point-in-time features

**Files:**
- Create: `src/mlb/pipeline/walk_forward_features.py`
- Test: `tests/mlb/pipeline/test_walk_forward_features.py`

**Interfaces:**
- Consumes: eligible starts with `pitcher_id`, `game_pk`, `game_date`, `scheduled_start_utc`, `season`, `is_home`, `opponent_team_id`, `strikeouts`, `batters_faced`, `outs`, `pitches`
- Produces: `WORKLOAD_FEATURES`, `NB_RATE_FEATURES`, `feature_block(history, block, *, m, kappa, rest_cap) -> pd.DataFrame`

`home_flag` is the row’s `is_home`. The feature tuples are exactly:

```python
WORKLOAD_FEATURES = (
    "home_flag",
    "days_rest_capped",
    "no_prior_regular_start",
    "long_layoff_flag",
    "season_starts_prior",
    "pitcher_prior_regular_starts",
    "pitcher_bf_mean_last3_smoothed",
    "pitcher_bf_mean_last10_smoothed",
    "pitcher_outs_mean_last5_smoothed",
    "pitcher_pitches_mean_last5_smoothed",
)
NB_RATE_FEATURES = (
    "home_flag",
    "season_starts_prior",
    "pitcher_prior_regular_starts",
    "pitcher_prior_bf_last10",
    "pitcher_k_per_bf_last3_smoothed",
    "pitcher_k_per_bf_last10_smoothed",
    "pitcher_k_per_bf_season_to_date_smoothed",
    "opponent_k_rate_vs_starters_last10_smoothed",
    "opponent_k_rate_vs_starters_season_to_date_smoothed",
    "opponent_prior_starts_observed",
)
```

`feature_block` returns one row per block start. Priors come only from `history`. Raw sums for a row include `history` plus block starts whose `game_date` is strictly earlier. Same-day starts contribute nothing.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from src.mlb.pipeline.walk_forward_features import feature_block


def _start(**kwargs):
    base = {
        "pitcher_id": 1,
        "game_pk": 1,
        "season": 2021,
        "game_date": "2021-04-01",
        "scheduled_start_utc": "2021-04-01T20:00:00Z",
        "is_home": 1,
        "opponent_team_id": 10,
        "strikeouts": 6,
        "batters_faced": 20,
        "outs": 15,
        "pitches": 80,
    }
    base.update(kwargs)
    return base


def test_same_day_doubleheader_does_not_enter_history():
    history = pd.DataFrame([_start(game_pk=1, game_date="2021-04-01", strikeouts=10, batters_faced=25)])
    block = pd.DataFrame(
        [
            _start(game_pk=2, game_date="2021-04-10", scheduled_start_utc="2021-04-10T17:00:00Z", strikeouts=4, batters_faced=18),
            _start(game_pk=3, game_date="2021-04-10", scheduled_start_utc="2021-04-10T23:00:00Z", strikeouts=9, batters_faced=22),
        ]
    )
    out = feature_block(history, block, m=50, kappa=3, rest_cap=30)
    assert out.loc[out["game_pk"] == 3, "pitcher_prior_regular_starts"].iloc[0] == 1
    assert out.loc[out["game_pk"] == 2, "pitcher_prior_regular_starts"].iloc[0] == 1


def test_earlier_date_in_the_block_updates_raw_sums_not_the_prior():
    history = pd.DataFrame(
        [_start(game_pk=1, game_date="2021-04-01", strikeouts=5, batters_faced=20)]
    )
    block = pd.DataFrame(
        [
            _start(game_pk=2, game_date="2021-04-06", strikeouts=1, batters_faced=10),
            _start(game_pk=3, game_date="2021-04-11", strikeouts=8, batters_faced=24),
        ]
    )
    out = feature_block(history, block, m=100, kappa=2, rest_cap=30)
    first_prior = out.loc[out["game_pk"] == 2, "league_k_per_bf_prior"].iloc[0]
    second_prior = out.loc[out["game_pk"] == 3, "league_k_per_bf_prior"].iloc[0]
    assert first_prior == second_prior == 5 / 20
    assert out.loc[out["game_pk"] == 3, "pitcher_prior_regular_starts"].iloc[0] == 2


def test_first_start_rest_is_the_median_of_capped_training_rest():
    history = pd.DataFrame(
        [
            _start(pitcher_id=7, game_pk=1, game_date="2021-04-01"),
            _start(pitcher_id=7, game_pk=2, game_date="2021-04-20", scheduled_start_utc="2021-04-20T20:00:00Z"),
        ]
    )
    block = pd.DataFrame([_start(pitcher_id=9, game_pk=3, game_date="2021-05-01")])
    out = feature_block(history, block, m=50, kappa=3, rest_cap=14)
    row = out.iloc[0]
    assert row["no_prior_regular_start"] == 1
    assert row["days_rest_capped"] == 14
```

The April 1 to April 20 gap is 19 days, capped at 14. That single capped value is the median.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/pipeline/test_walk_forward_features.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `feature_block`**

Use `src.mlb.models.shrinkage.shrink_rate` for K/BF. Workload means use the same formula: `(sum + κ μ) / (n + κ)`.

League rate is `sum(strikeouts) / sum(batters_faced)` on `history`. Population means are the means of `batters_faced`, `outs`, and `pitches` on `history`. Rest median is the median of capped gaps inside `history`, using the same earlier-date rule and the function’s `rest_cap`. A history row with no earlier date does not contribute a rest value.

Season-to-date sums reset by `season`. When the season count is 0, the season-to-date smoothed rate equals the cross-season last-10 smoothed rate, and `season_starts_prior` stays 0. Last-N windows use however many earlier-date starts exist. Opponent windows filter `opponent_team_id` first. `opponent_prior_starts_observed` is the count inside the last-10 opponent window, from 0 to 10.

Sort earlier starts by `game_date`, then `scheduled_start_utc`. The previous rest date is the latest earlier `game_date`. If that date has several starts, use the latest `scheduled_start_utc` on it. The gap is `total_seconds / 86400`.

Return the workload columns, the strikeout columns, the raw sums needed to reapply `m` (`pitcher_k_last3`, `pitcher_bf_last3`, `pitcher_k_last10`, `pitcher_bf_last10`, `pitcher_k_season`, `pitcher_bf_season`, `opponent_k_last10`, `opponent_bf_last10`, `opponent_k_season`, `opponent_bf_season`), and the prior columns.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/pipeline/test_walk_forward_features.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/walk_forward_features.py tests/mlb/pipeline/test_walk_forward_features.py
git commit -m "$(cat <<'EOF'
Build strikeout walk-forward features from earlier official dates only.

EOF
)"
```

### Task 6: Immutable snapshot store

**Files:**
- Create: `src/mlb/evaluation/walk_forward_snapshot.py`
- Test: `tests/mlb/evaluation/test_walk_forward_snapshot.py`

**Interfaces:**
- Consumes: feature rows from Task 5
- Produces: `SnapshotStore.write(frame) -> None`, `SnapshotStore.frame() -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
import pytest

from src.mlb.evaluation.walk_forward_snapshot import SnapshotStore


def test_second_write_of_the_same_start_raises():
    store = SnapshotStore()
    row = pd.DataFrame([{"pitcher_id": 1, "game_pk": 10, "predicted_bf_oof": 21.0}])
    store.write(row)
    with pytest.raises(ValueError, match="game_pk=10"):
        store.write(row)


def test_store_returns_the_original_feature_value():
    store = SnapshotStore()
    store.write(pd.DataFrame([{"pitcher_id": 1, "game_pk": 10, "league_k_per_bf_prior": 0.22}]))
    assert store.frame().iloc[0]["league_k_per_bf_prior"] == 0.22
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/evaluation/test_walk_forward_snapshot.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the store**

`write` concatenates rows. Before concatenating, it rejects any `(pitcher_id, game_pk)` already present. The error names both ids. `frame` returns a copy.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/evaluation/test_walk_forward_snapshot.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/evaluation/walk_forward_snapshot.py tests/mlb/evaluation/test_walk_forward_snapshot.py
git commit -m "$(cat <<'EOF'
Reject a second walk-forward snapshot for the same pitcher start.

EOF
)"
```

### Task 7: Expanding block walker

**Files:**
- Create: `src/mlb/evaluation/strikeout_walk_forward.py`
- Test: `tests/mlb/evaluation/test_strikeout_walk_forward.py`

**Interfaces:**
- Consumes: Tasks 2–6
- Produces: `walk_blocks(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts) -> pd.DataFrame`

`walk_blocks` returns the snapshot rows. It does not write the `SnapshotStore`. The caller writes the winning frame once. The walker processes eligible blocks in `(season, block_index)` order through `through_season`.

For a 2018 block, priors for each official date are the eligible 2018 starts on earlier dates. Those rows are stored as feature snapshots only when they have finite workload features. They do not receive `predicted_bf_oof`. They are not strikeout training rows.

For 2019 and later, a block is fit only when at least `workload_min_train_starts` (24) stored finite workload rows have an earlier `game_date`. The workload fit reads stored workload features. It does not rebuild them. Prediction features for the block come from `feature_block`, using the pre-block snapshot rows as `history` for the priors and the block’s own earlier dates for raw sums. Each predicted row is written once, including `predicted_bf_oof`.

When `score_strikeouts` is true, the strikeout fit reads stored strikeout features and stored `predicted_bf_oof` from earlier dates only. The offset is `log(max(predicted_bf_oof, 1e-6))`. 2018 rows are excluded. A block that raises `UnestimatedDispersion` records `fit_status="unestimated_dispersion"` and does not drop rows to invent a score.

- [ ] **Step 1: Write the failing test**

Build 30 pitcher starts on distinct April 2018 dates and 10 starts on distinct April 2019 dates. Give every start positive batters faced and strikeouts. Use one opponent id.

```python
def test_2019_offset_ignores_a_same_day_result():
    frame = walk_blocks(
        starts,
        calendar,
        m=50,
        kappa=3,
        rest_cap=30,
        through_season=2019,
        score_strikeouts=False,
    )
    april_10 = frame.loc[frame["game_date"] == "2019-04-10"]
    assert april_10["predicted_bf_oof"].notna().all()
    assert frame.loc[frame["season"] == 2018, "predicted_bf_oof"].isna().all()
```

Add a second start on `2019-04-10` with a later `scheduled_start_utc`. Assert both rows have the same `pitcher_prior_regular_starts`.

Add `test_later_block_keeps_the_2019_league_prior`: after walking through a 2020 block, the stored 2019 `league_k_per_bf_prior` is unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/evaluation/test_strikeout_walk_forward.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `walk_blocks`**

Order eligible rows by `game_date`, then `scheduled_start_utc`. Group by `walk_forward_block`. For each block:

1. Split stored rows into `game_date < block_open` and the block itself.
2. Fit the workload model on the earlier stored rows when 24 finite rows exist.
3. Call `feature_block` with those earlier rows as history.
4. Predict batters faced for 2019 and later.
5. Append the block’s feature rows to the result frame. For 2019 and later, include `predicted_bf_oof`. For 2018, leave `predicted_bf_oof` null.
6. If scoring strikeouts, fit on earlier result rows that have `predicted_bf_oof`, predict the block, and include `predicted_strikeout_mean`, `negative_binomial_dispersion`, and the `k_max = 15` reporting mass from `negative_binomial_pmf` on those same rows. When `score_strikeouts` is false, omit them.

Do not call `feature_block` again for a row already present in the result. `tail_mass_threshold` is the existing config value `0.001`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/evaluation/test_strikeout_walk_forward.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/evaluation/strikeout_walk_forward.py tests/mlb/evaluation/test_strikeout_walk_forward.py
git commit -m "$(cat <<'EOF'
Walk strikeout blocks forward without rewriting stored pregame features.

EOF
)"
```

### Task 8: 2020 lock and the reported metrics

**Files:**
- Modify: `src/mlb/evaluation/strikeout_walk_forward.py`
- Test: `tests/mlb/evaluation/test_strikeout_walk_forward_report.py`

**Interfaces:**
- Consumes: `walk_blocks`, `nb2_nll`, `nb2_crps`, `nb2_quantile`, `half_point_probabilities`
- Produces: `select_2020(starts, calendar) -> Lock`, `score_locked_seasons(starts, calendar, lock, seasons) -> pd.DataFrame`

`Lock` fields: `m`, `kappa`, `rest_cap`.

- [ ] **Step 1: Write the failing test**

```python
def test_2020_tie_breaks_toward_the_smaller_kappa_then_the_smaller_cap(monkeypatch):
    calls = []

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        calls.append((kappa, rest_cap, m, score_strikeouts))
        return pd.DataFrame([
            {
                "pitcher_id": 1,
                "game_pk": 1,
                "season": 2020,
                "predicted_bf_oof": 20.0,
                "batters_faced": 20,
                "strikeouts": 5,
                "pitcher_k_last10": 5.0,
                "pitcher_bf_last10": 20.0,
                "pitcher_k_last3": 5.0,
                "pitcher_bf_last3": 20.0,
                "pitcher_k_season": 5.0,
                "pitcher_bf_season": 20.0,
                "opponent_k_last10": 5.0,
                "opponent_bf_last10": 20.0,
                "opponent_k_season": 5.0,
                "opponent_bf_season": 20.0,
                "league_k_per_bf_prior": 0.25,
            }
        ])

    monkeypatch.setattr(
        "src.mlb.evaluation.strikeout_walk_forward.walk_blocks",
        fake_walk,
    )
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.kappa == 1
    assert lock.rest_cap == 14
```

The fake walk makes every pair’s MAE identical, so the tie break is the assertion. Stage 2 may use the same fake store. Give every `m` the same NLL by using one stored row, and assert `lock.m == 25`.

Add `test_report_labels_calibration_in_the_large` on a hand-built prediction frame of 5 rows. The report column is `calibration_in_the_large_4_5`, equal to mean predicted over-probability minus the observed over rate is not required. Store both `mean_predicted_over_4_5` and `observed_over_rate_4_5`. Coverage columns are `coverage_q25_q75` and `coverage_q10_q90`, with `nominal_q25_q75 = 0.50` and `nominal_q10_q90 = 0.80`. No pass/fail flag.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mlb/evaluation/test_strikeout_walk_forward_report.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement selection and the report**

`select_2020` loops `κ`, then rest cap. It calls `walk_blocks(..., through_season=2020, score_strikeouts=False)`. MAE is the mean absolute error of `predicted_bf_oof` against `batters_faced` on season 2020 rows, pooled. Keep the pair with the lowest MAE, then the smaller `κ`, then the smaller cap. Keep that pair’s frame.

Stage 2 does not call `walk_blocks` again. For each `m`, recompute smoothed K/BF columns from the kept raw sums and `league_k_per_bf_prior`, then score 2020 strikeout rows with a strikeout fit that uses those columns and the stored offsets. NLL is `nb2_nll` on the pooled 2020 rows. Keep the lowest NLL, then the smaller `m`. Write that one frame through `SnapshotStore.write` once. Discard the other frames.

`score_locked_seasons` calls `walk_blocks` with the locked values, `score_strikeouts=True`, and `through_season=max(seasons)`. It returns one row per block plus an aggregate row. Columns:

```text
walk_forward_block
role
train_start
train_end
validation_start
validation_end
n_train
n_validation
workload_mae
workload_bias
strikeout_nll
discrete_crps
median_mae
coverage_q25_q75
nominal_q25_q75
coverage_q10_q90
nominal_q10_q90
empirical_cdf_q10
empirical_cdf_q25
empirical_cdf_q50
empirical_cdf_q75
empirical_cdf_q90
brier_4_5
brier_5_5
brier_6_5
mean_predicted_over_4_5
observed_over_rate_4_5
mean_predicted_over_5_5
observed_over_rate_5_5
mean_predicted_over_6_5
observed_over_rate_6_5
```

`role` is `offset_generation` for 2019, `tuning` for 2020, `reported` for 2021–2024, and `holdout` for 2025. Workload bias is mean `predicted_bf_oof - batters_faced`. Median MAE uses `nb2_quantile(..., 0.50)`. Brier score is the mean squared error of the analytical over-probability against the 0/1 over outcome. A reported or holdout block that raises `UnestimatedDispersion` aborts. A 2020 candidate that raises it is skipped and cannot win.

Aggregate reliability bins are a second frame from `reliability_bins(predictions, line)` for the pooled reported rows and, separately, the pooled holdout rows. Bin edges are `0.0, 0.1, ..., 1.0`. A bin with fewer than 200 rows is absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mlb/evaluation/test_strikeout_walk_forward_report.py tests/mlb/evaluation/test_strikeout_walk_forward.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/evaluation/strikeout_walk_forward.py tests/mlb/evaluation/test_strikeout_walk_forward_report.py
git commit -m "$(cat <<'EOF'
Lock strikeout shrinkage on 2020 and report the later walk-forward blocks.

EOF
)"
```

### Task 9: Temporal integrity suite

**Files:**
- Test: `tests/mlb/evaluation/test_strikeout_temporal.py`

**Interfaces:**
- Consumes: `feature_block`, `walk_blocks`, `SnapshotStore`, `assign_walk_forward_blocks`

- [ ] **Step 1: Write the failing tests**

Cover every rule in the spec’s temporal tests with synthetic starts:

- A postseason start dated after `regular_season_close_date` does not change `pitcher_prior_regular_starts` or `league_k_per_bf_prior` for the next regular-season start.
- The strikeout feature list returned for a row does not contain `batters_faced`, `outs`, or `pitches` of that same `game_pk`.
- `SnapshotStore.write` on `walk_blocks`’s frame, called a second time with that same frame, raises `ValueError`.
- After walking two blocks, copying a 2025 strikeout onto a 2024 stored row is not what the code does. Instead, mutate the source start’s 2025 strikeouts before a second fresh walk and assert the first 2025 block’s stored `league_k_per_bf_prior` equals the prior from rows before that block, not a prior that includes that block’s own strikeouts.
- An earlier date inside one block changes the later date’s `pitcher_prior_regular_starts` and `predicted_bf_oof` inputs, while both rows store the same `league_k_per_bf_prior`.

- [ ] **Step 2: Run the suite**

Run: `pytest tests/mlb/evaluation/test_strikeout_temporal.py tests/mlb/pipeline/test_walk_forward_features.py tests/mlb/evaluation/test_walk_forward_snapshot.py -q`

Expected: PASS. These tests exercise the modules from Tasks 5–8. If a temporal assertion fails, fix the walker or the feature builder. Do not weaken the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/mlb/evaluation/test_strikeout_temporal.py
git commit -m "$(cat <<'EOF'
Lock the strikeout walk-forward against same-day and future-prior leakage.

EOF
)"
```
