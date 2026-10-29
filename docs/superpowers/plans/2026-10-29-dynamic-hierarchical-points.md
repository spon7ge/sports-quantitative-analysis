# Dynamic Hierarchical Points Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested dynamic hierarchical NBA points model, a leakage-free season walk-forward trace, deployment diagnostics, and a standalone reporting notebook.

**Architecture:** Put deterministic feature, hierarchy, state-filtering, and walk-forward logic in `src/models/dynamic_points.py`; put trace summaries in `src/models/dynamic_points_diagnostics.py`. Keep local parquet discovery, configuration, display, and plots in the notebook so production logic can be tested without executing notebook cells.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy sparse/statistics, matplotlib, seaborn, pytest, Ruff, mypy, Jupyter.

## Global Constraints

- Use only information available before each predicted game.
- Fit each evaluation fold on the immediately preceding two complete seasons.
- Keep `state_drift_sd=0.35`, `offseason_drift_sd=2.0`, player prior precision 10, team prior precision 20, and position prior precision 20.
- Add offseason variance exactly once per returning player at each season boundary without moving the state mean.
- Predict first, record the trace, then update with observed points.
- Predictive scale squared is observation variance plus state variance.
- Log every excluded or imputed row count; never silently drop data.
- Do not create posterior draws. Draw diagnostics must return a structured skipped result when draws are absent.
- Do not add market-pricing or bet-selection logic.
- Preserve unrelated working-tree and staged changes.

## File Structure

- Create `src/models/dynamic_points.py`: causal frame construction, hierarchy fit, Kalman state transitions, fold prediction, and walk-forward orchestration.
- Create `src/models/dynamic_points_diagnostics.py`: calibration, variance, shrinkage, offseason, group, gain, and optional draw summaries.
- Create `tests/models/test_dynamic_points.py`: deterministic model, leakage, boundary, and trace tests using synthetic data.
- Create `tests/models/test_dynamic_points_diagnostics.py`: deterministic diagnostic-summary tests.
- Replace `notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb`: standalone configuration, data loading, evaluation, diagnostics, plots, and live-state report.

---

### Task 1: Causal frame and state primitives

**Files:**
- Create: `src/models/dynamic_points.py`
- Create: `tests/models/test_dynamic_points.py`

**Interfaces:**
- Produces: `DynamicPointsConfig`, `DynamicState`, `KalmanUpdate`, `build_causal_frame`, `kalman_update`, `predictive_scale`.
- Consumes: raw regular-season game-log columns documented in `REQUIRED_RAW_COLUMNS`.

- [ ] **Step 1: Write failing primitive and feature tests**

Create `tests/models/test_dynamic_points.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dynamic_points import (
    DynamicPointsConfig,
    build_causal_frame,
    kalman_update,
    predictive_scale,
)


def _games() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, start in (
        ("2022-23", "2022-10-01"),
        ("2023-24", "2023-10-01"),
        ("2024-25", "2024-10-01"),
    ):
        for game in range(3):
            for player_id, team_id, opponent_id, position in (
                (1, 10, 20, "PG"),
                (2, 20, 10, "C"),
            ):
                rows.append(
                    {
                        "season_year": season,
                        "season_type": "Regular Season",
                        "game_date": pd.Timestamp(start) + pd.Timedelta(days=game * 2),
                        "game_id": f"{season}-{game}",
                        "player_id": player_id,
                        "player_name": f"Player {player_id}",
                        "team_id": team_id,
                        "opp_team_id": opponent_id,
                        "pos": position,
                        "matchup": "AAA vs. BBB" if team_id == 10 else "BBB @ AAA",
                        "minutes": 30.0 + player_id + game,
                        "fga": 10.0 + player_id + game,
                        "fta": 2.0 + game,
                        "pts": 15.0 + 2 * player_id + game,
                        "team_def_rating": 110.0 + team_id / 10 + game,
                    }
                )
    return pd.DataFrame(rows)


def test_kalman_update_matches_closed_form() -> None:
    result = kalman_update(
        mean=3.0,
        variance=4.0,
        observation=20.0,
        base_mean=12.0,
        observation_variance=16.0,
    )
    assert result.gain == pytest.approx(0.2)
    assert result.innovation == pytest.approx(5.0)
    assert result.mean == pytest.approx(4.0)
    assert result.variance == pytest.approx(3.2)


def test_predictive_scale_adds_variances() -> None:
    assert predictive_scale(9.0, 16.0) == pytest.approx(5.0)


def test_causal_features_use_only_prior_games() -> None:
    frame, audit = build_causal_frame(_games())
    player = frame.loc[frame["player_id"].eq(1)].sort_values("game_date")
    assert np.isnan(player.iloc[0]["min_lag_1"])
    assert player.iloc[1]["min_lag_1"] == pytest.approx(player.iloc[0]["minutes"])
    assert player.iloc[1]["fga_lag_1"] == pytest.approx(player.iloc[0]["fga"])
    assert player.iloc[1]["fta_lag_1"] == pytest.approx(player.iloc[0]["fta"])
    assert audit["input_rows"] == len(_games())
    assert audit["output_rows"] == len(_games())


def test_team_and_model_rookie_flags_follow_prior_history() -> None:
    games = _games()
    mask = games["player_id"].eq(1) & games["season_year"].eq("2024-25")
    games.loc[mask, ["team_id", "opp_team_id"]] = [30, 40]
    frame, _ = build_causal_frame(games)
    player = frame.loc[frame["player_id"].eq(1)].sort_values("game_date")
    first_2024 = player.loc[player["season_year"].eq("2024-25")].iloc[0]
    assert bool(first_2024["new_team_flag"])
    assert not bool(first_2024["rookie_flag"])
    first_2022 = player.loc[player["season_year"].eq("2022-23")].iloc[0]
    assert bool(first_2022["rookie_flag"])


def test_duplicate_player_game_is_rejected() -> None:
    duplicate = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate player/game"):
        build_causal_frame(duplicate)


def test_config_rejects_nonpositive_uncertainty() -> None:
    with pytest.raises(ValueError, match="positive"):
        DynamicPointsConfig(state_drift_sd=0.0)
```

- [ ] **Step 2: Run tests and confirm the expected import failure**

Run:

```bash
.venv/bin/python -m pytest tests/models/test_dynamic_points.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.models.dynamic_points'`.

- [ ] **Step 3: Implement the primitives and causal frame**

Create `src/models/dynamic_points.py` with these public definitions and validation:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu

FEATURE_COLUMNS = (
    "min_lag_1",
    "fga_lag_1",
    "fta_lag_1",
    "rest_days",
    "opp_def_prior",
    "is_home",
)
REQUIRED_RAW_COLUMNS = {
    "season_year",
    "season_type",
    "game_date",
    "game_id",
    "player_id",
    "player_name",
    "team_id",
    "opp_team_id",
    "pos",
    "matchup",
    "minutes",
    "fga",
    "fta",
    "pts",
    "team_def_rating",
}


@dataclass(frozen=True)
class DynamicPointsConfig:
    state_drift_sd: float = 0.35
    offseason_drift_sd: float = 2.0
    player_prior_precision: float = 10.0
    team_prior_precision: float = 20.0
    position_prior_precision: float = 20.0
    slope_prior_sd: float = 2.5
    variance_prior_alpha: float = 2.0
    variance_prior_beta: float = 50.0
    train_window_seasons: int = 2
    min_group_n: int = 30

    def __post_init__(self) -> None:
        positive = (
            self.state_drift_sd,
            self.offseason_drift_sd,
            self.player_prior_precision,
            self.team_prior_precision,
            self.position_prior_precision,
            self.slope_prior_sd,
            self.variance_prior_alpha,
            self.variance_prior_beta,
        )
        if any(value <= 0 or not math.isfinite(value) for value in positive):
            raise ValueError("uncertainty and prior values must be positive and finite")
        if self.train_window_seasons < 1 or self.min_group_n < 2:
            raise ValueError("season window and minimum group size are invalid")


@dataclass(frozen=True)
class DynamicState:
    mean: float
    variance: float
    games: int
    season: str | None = None


@dataclass(frozen=True)
class KalmanUpdate:
    mean: float
    variance: float
    gain: float
    innovation: float


@dataclass(frozen=True)
class HierarchicalFit:
    center: np.ndarray
    scale: np.ndarray
    medians: np.ndarray
    coef: np.ndarray
    factor: SuperLU
    alpha: float
    beta: float
    position_map: dict[str, int]
    team_map: dict[Any, int]
    player_map: dict[Any, int]

    @property
    def df(self) -> float:
        return 2.0 * self.alpha

    @property
    def observation_variance(self) -> float:
        return self.beta / (self.alpha - 1.0)


def kalman_update(
    mean: float,
    variance: float,
    observation: float,
    base_mean: float,
    observation_variance: float,
) -> KalmanUpdate:
    if variance <= 0 or observation_variance <= 0:
        raise ValueError("state and observation variances must be positive")
    innovation = observation - (base_mean + mean)
    gain = variance / (variance + observation_variance)
    return KalmanUpdate(
        mean=mean + gain * innovation,
        variance=(1.0 - gain) * variance,
        gain=gain,
        innovation=innovation,
    )


def predictive_scale(state_variance: float, observation_variance: float) -> float:
    if state_variance <= 0 or observation_variance <= 0:
        raise ValueError("state and observation variances must be positive")
    return math.sqrt(state_variance + observation_variance)


def build_causal_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = REQUIRED_RAW_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = games.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise")
    frame = frame.loc[frame["season_type"].eq("Regular Season")].copy()
    if frame.duplicated(["player_id", "game_id"]).any():
        raise ValueError("duplicate player/game rows")
    frame = frame.sort_values(
        ["player_id", "game_date", "game_id"], kind="mergesort"
    )
    for source, target in (
        ("minutes", "min_lag_1"),
        ("fga", "fga_lag_1"),
        ("fta", "fta_lag_1"),
    ):
        numeric = pd.to_numeric(frame[source], errors="coerce")
        frame[target] = numeric.groupby(frame["player_id"], sort=False).shift(1)
    frame["rest_days"] = (
        frame.groupby(["player_id", "season_year"], sort=False)["game_date"]
        .diff()
        .dt.days
    )
    frame["is_home"] = (
        frame["matchup"].astype(str).str.contains("vs", case=False).astype(float)
    )
    teams = (
        frame[["team_id", "game_id", "game_date", "team_def_rating"]]
        .drop_duplicates(["team_id", "game_id"])
        .sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    )
    defense = pd.to_numeric(teams["team_def_rating"], errors="coerce")
    prior = defense.groupby(teams["team_id"], sort=False).shift(1)
    teams["opp_def_prior"] = (
        prior.groupby(teams["team_id"], sort=False)
        .rolling(10, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    opponent = teams[["game_id", "team_id", "opp_def_prior"]].rename(
        columns={"team_id": "opp_team_id"}
    )
    frame = frame.merge(
        opponent,
        on=["game_id", "opp_team_id"],
        how="left",
        validate="many_to_one",
    )
    prior_team = frame.groupby("player_id", sort=False)["team_id"].shift(1)
    frame["new_team_flag"] = prior_team.notna() & frame["team_id"].ne(prior_team)
    first_season = frame.groupby("player_id", sort=False)["season_year"].transform(
        "first"
    )
    frame["rookie_flag"] = frame["season_year"].eq(first_season)
    frame["season_game_number"] = (
        frame.groupby(["player_id", "season_year"], sort=False).cumcount() + 1
    )
    return frame, {
        "input_rows": int(len(games)),
        "postseason_rows_removed": int(len(games) - len(frame)),
        "output_rows": int(len(frame)),
    }
```

- [ ] **Step 4: Run primitive tests**

Run:

```bash
.venv/bin/python -m pytest tests/models/test_dynamic_points.py -q
```

Expected: all six tests pass.

- [ ] **Step 5: Lint the new files**

Run:

```bash
.venv/bin/ruff check src/models/dynamic_points.py tests/models/test_dynamic_points.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit the primitive layer**

```bash
git add src/models/dynamic_points.py tests/models/test_dynamic_points.py
git commit -m "feat: add dynamic points state primitives"
```

---

### Task 2: Hierarchical fit and chronological walk-forward trace

**Files:**
- Modify: `src/models/dynamic_points.py`
- Modify: `tests/models/test_dynamic_points.py`

**Interfaces:**
- Consumes: `DynamicPointsConfig`, `DynamicState`, `KalmanUpdate`, `FEATURE_COLUMNS`.
- Produces: `fit_hierarchy(frame, config) -> HierarchicalFit`, `filter_training_states`, `predict_fold`, and `walk_forward_trace`.

- [ ] **Step 1: Add failing walk-forward tests**

Append to `tests/models/test_dynamic_points.py`:

```python
from src.models.dynamic_points import walk_forward_trace


def test_walk_forward_is_out_of_sample_and_predicts_before_update() -> None:
    frame, _ = build_causal_frame(_games())
    trace, audits = walk_forward_trace(
        frame,
        evaluation_seasons=("2024-25",),
        config=DynamicPointsConfig(train_window_seasons=2),
    )
    assert set(trace["season_year"]) == {"2024-25"}
    assert audits[0]["train_seasons"] == ("2022-23", "2023-24")
    assert audits[0]["evaluation_season"] == "2024-25"
    first = trace.sort_values(["game_date", "game_id", "player_id"]).iloc[0]
    expected_innovation = first["points"] - first["predictive_mean"]
    assert first["innovation"] == pytest.approx(expected_innovation)
    expected_after = (
        first["state_before"] + first["kalman_gain"] * first["innovation"]
    )
    assert first["state_after"] == pytest.approx(expected_after)


def test_offseason_variance_is_added_once_without_moving_mean() -> None:
    frame, _ = build_causal_frame(_games())
    config = DynamicPointsConfig(
        state_drift_sd=0.5,
        offseason_drift_sd=2.0,
        train_window_seasons=2,
    )
    trace, _ = walk_forward_trace(
        frame,
        evaluation_seasons=("2024-25",),
        config=config,
    )
    player = trace.loc[trace["player_id"].eq(1)].sort_values("game_date")
    first, second = player.iloc[0], player.iloc[1]
    assert bool(first["season_boundary"])
    assert first["state_before"] == pytest.approx(first["state_end_training"])
    assert first["state_var_before"] == pytest.approx(
        first["state_var_end_training"] + config.offseason_drift_sd**2
    )
    assert not bool(second["season_boundary"])
    assert second["state_var_before"] == pytest.approx(
        first["state_var_after"] + config.state_drift_sd**2
    )


def test_trace_contains_finite_variance_and_identity_columns() -> None:
    frame, _ = build_causal_frame(_games())
    trace, _ = walk_forward_trace(
        frame,
        evaluation_seasons=("2024-25",),
        config=DynamicPointsConfig(),
    )
    required = {
        "team_id",
        "opp_team_id",
        "position",
        "state_var_before",
        "obs_var",
        "predictive_var",
        "games_in_model_before",
        "returning_player",
        "new_team_flag",
        "rookie_flag",
        "z",
        "pit",
    }
    assert required.issubset(trace.columns)
    assert np.isfinite(trace[list(required & set(trace.select_dtypes("number")))]).all().all()
    assert (trace["predictive_var"] > 0).all()
    assert np.allclose(
        trace["predictive_var"],
        trace["state_var_before"] + trace["obs_var"],
    )


def test_unknown_evaluation_season_is_rejected() -> None:
    frame, _ = build_causal_frame(_games())
    with pytest.raises(ValueError, match="evaluation season"):
        walk_forward_trace(
            frame,
            evaluation_seasons=("2030-31",),
            config=DynamicPointsConfig(),
        )
```

- [ ] **Step 2: Run the new tests and confirm missing API failures**

Run:

```bash
.venv/bin/python -m pytest tests/models/test_dynamic_points.py -q
```

Expected: collection fails because `walk_forward_trace` is not defined.

- [ ] **Step 3: Implement hierarchy helpers**

Add private helpers `_category_map`, `_one_hot`, `_impute_features`,
`_design_row`, and public `fit_hierarchy`. Use the exact design matrix from
`bayesian_models.ipynb`: intercept, standardized common features, then
position/team/player one-hot blocks. Set precisions from `DynamicPointsConfig`.
The implementation must:

```python
def _category_map(values: pd.Series) -> dict[Any, int]:
    unique = sorted(values.dropna().unique().tolist())
    return {value: index for index, value in enumerate(unique)}


def _one_hot(values: pd.Series, mapping: dict[Any, int]) -> sparse.csr_matrix:
    columns = np.array([mapping[value] for value in values])
    rows = np.arange(len(values))
    return sparse.csr_matrix(
        (np.ones(len(values)), (rows, columns)),
        shape=(len(values), len(mapping)),
    )


def fit_hierarchy(
    frame: pd.DataFrame,
    config: DynamicPointsConfig,
) -> tuple[HierarchicalFit, dict[str, int]]:
    required = set(FEATURE_COLUMNS) | {"pts", "player_id", "pos", "team_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing hierarchy columns: {sorted(missing)}")
    usable = frame.loc[pd.to_numeric(frame["minutes"], errors="coerce").gt(0)].copy()
    medians = usable.loc[:, FEATURE_COLUMNS].median().to_numpy(dtype=float)
    if not np.isfinite(medians).all():
        raise ValueError("a feature has no finite training median")
    raw = usable.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missing_features = int(raw.isna().sum().sum())
    raw = raw.fillna(dict(zip(FEATURE_COLUMNS, medians, strict=True)))
    values = raw.to_numpy(dtype=float)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("a hierarchy feature has no variation")
    positions = usable["pos"].astype(str)
    position_map = _category_map(positions)
    team_map = _category_map(usable["team_id"])
    player_map = _category_map(usable["player_id"])
    common = sparse.csr_matrix(
        np.column_stack([np.ones(len(usable)), (values - center) / scale])
    )
    design = sparse.hstack(
        [
            common,
            _one_hot(positions, position_map),
            _one_hot(usable["team_id"], team_map),
            _one_hot(usable["player_id"], player_map),
        ],
        format="csc",
    )
    precision = np.concatenate(
        [
            [1e-6],
            np.full(len(FEATURE_COLUMNS), 1.0 / config.slope_prior_sd**2),
            np.full(len(position_map), config.position_prior_precision),
            np.full(len(team_map), config.team_prior_precision),
            np.full(len(player_map), config.player_prior_precision),
        ]
    )
    posterior_precision = (
        design.T @ design + sparse.diags(precision, format="csc")
    ).tocsc()
    factor = splu(posterior_precision)
    y = pd.to_numeric(usable["pts"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("training outcomes must be finite")
    xty = np.asarray(design.T @ y).ravel()
    coef = factor.solve(xty)
    alpha = config.variance_prior_alpha + len(usable) / 2
    beta = config.variance_prior_beta + 0.5 * float(y @ y - coef @ xty)
    if alpha <= 1 or beta <= 0 or not np.isfinite(coef).all():
        raise ValueError("hierarchical posterior is not finite")
    return (
        HierarchicalFit(
            center=center,
            scale=scale,
            medians=medians,
            coef=coef,
            factor=factor,
            alpha=alpha,
            beta=beta,
            position_map=position_map,
            team_map=team_map,
            player_map=player_map,
        ),
        {
            "candidate_rows": int(len(frame)),
            "fitted_rows": int(len(usable)),
            "zero_minute_rows_removed": int(len(frame) - len(usable)),
            "imputed_feature_cells": missing_features,
        },
    )
```

Add `_design_row`, `_impute_row_features`, and `_base_prediction` with explicit
zero effects for unknown categories:

```python
def _impute_row_features(row: object, medians: np.ndarray) -> np.ndarray:
    raw = np.array([getattr(row, name) for name in FEATURE_COLUMNS], dtype=float)
    missing = ~np.isfinite(raw)
    raw[missing] = medians[missing]
    return raw


def _design_row(
    fit: HierarchicalFit,
    raw: np.ndarray,
    position: object,
    team_id: object,
    player_id: object | None = None,
) -> np.ndarray:
    n_common = 1 + len(FEATURE_COLUMNS)
    n_position = len(fit.position_map)
    n_team = len(fit.team_map)
    row = np.zeros(len(fit.coef))
    row[0] = 1.0
    row[1:n_common] = (raw - fit.center) / fit.scale
    position_key = str(position)
    if position_key in fit.position_map:
        row[n_common + fit.position_map[position_key]] = 1.0
    if team_id in fit.team_map:
        row[n_common + n_position + fit.team_map[team_id]] = 1.0
    if player_id in fit.player_map:
        row[
            n_common + n_position + n_team + fit.player_map[player_id]
        ] = 1.0
    return row


def _base_prediction(
    fit: HierarchicalFit,
    raw: np.ndarray,
    position: object,
    team_id: object,
) -> float:
    return float(_design_row(fit, raw, position, team_id) @ fit.coef)
```

- [ ] **Step 4: Implement training-state filtering**

Add:

```python
def filter_training_states(
    fit: HierarchicalFit,
    frame: pd.DataFrame,
    config: DynamicPointsConfig,
) -> dict[object, DynamicState]:
    states: dict[object, DynamicState] = {}
    initial_variance = fit.observation_variance / config.player_prior_precision
    ordered = frame.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    )
    for row in ordered.itertuples(index=False):
        state = states.get(row.player_id, DynamicState(0.0, initial_variance, 0))
        variance = state.variance
        if state.games:
            variance += (
                config.offseason_drift_sd**2
                if state.season != row.season_year
                else config.state_drift_sd**2
            )
        raw = _impute_row_features(row, fit.medians)
        base = _base_prediction(fit, raw, row.pos, row.team_id)
        updated = kalman_update(
            state.mean,
            variance,
            float(row.pts),
            base,
            fit.observation_variance,
        )
        states[row.player_id] = DynamicState(
            updated.mean,
            updated.variance,
            state.games + 1,
            str(row.season_year),
        )
    return states
```

- [ ] **Step 5: Implement one fold and orchestration**

Add `predict_fold` and `walk_forward_trace`. `predict_fold` must snapshot
`state_end_training`/`state_var_end_training`, add offseason variance only for
returning players' first evaluation appearance, and emit all canonical trace
columns before applying `kalman_update`. Calculate:

```python
scale = predictive_scale(variance, fit.observation_variance)
predictive_mean = base + state.mean
innovation = points - predictive_mean
z = innovation / scale
pit = float(stats.t.cdf(points, fit.df, loc=predictive_mean, scale=scale))
```

`walk_forward_trace` must find each evaluation season's index in chronological
season order, take exactly `config.train_window_seasons` preceding seasons,
fit only those rows, and assert:

```python
if set(train_seasons) & {evaluation_season}:
    raise AssertionError("evaluation season leaked into training")
if train["game_date"].max() >= evaluation["game_date"].min():
    raise AssertionError("training dates must precede evaluation dates")
```

Return `(pd.concat(fold_traces, ignore_index=True), fold_audits)`, where each
audit records train seasons, evaluation season, candidate rows, fitted rows,
imputed feature cells, and predicted rows.

- [ ] **Step 6: Run model tests**

Run:

```bash
.venv/bin/python -m pytest tests/models/test_dynamic_points.py -q
```

Expected: all ten tests pass.

- [ ] **Step 7: Run static checks**

Run:

```bash
.venv/bin/ruff check src/models/dynamic_points.py tests/models/test_dynamic_points.py
.venv/bin/mypy src/models/dynamic_points.py
```

Expected: both commands exit 0.

- [ ] **Step 8: Commit the walk-forward model**

```bash
git add src/models/dynamic_points.py tests/models/test_dynamic_points.py
git commit -m "feat: add dynamic points walk-forward trace"
```

---

### Task 3: Deployment diagnostic summaries

**Files:**
- Create: `src/models/dynamic_points_diagnostics.py`
- Create: `tests/models/test_dynamic_points_diagnostics.py`

**Interfaces:**
- Consumes: canonical trace returned by `walk_forward_trace`.
- Produces: `core_calibration`, `variance_decomposition`, `shrinkage_summary`, `offseason_summary`, `group_calibration`, `gain_summary`, `posterior_draw_summary`.

- [ ] **Step 1: Write failing diagnostic tests**

Create `tests/models/test_dynamic_points_diagnostics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dynamic_points_diagnostics import (
    core_calibration,
    games_played_tier,
    group_calibration,
    offseason_summary,
    posterior_draw_summary,
    shrinkage_summary,
    variance_decomposition,
)


def _trace() -> pd.DataFrame:
    n = 80
    games = np.arange(n)
    state = np.linspace(-4.0, 4.0, n)
    innovation = -0.25 * state + np.sin(games)
    state_var = np.where(games < 10, 9.0, 1.0)
    obs_var = np.full(n, 16.0)
    scale = np.sqrt(state_var + obs_var)
    return pd.DataFrame(
        {
            "season_year": np.where(games < 40, "2024-25", "2025-26"),
            "player_id": games % 4,
            "team_id": games % 2,
            "opp_team_id": (games + 1) % 2,
            "position": np.where(games % 2, "G", "C"),
            "points": 20.0 + innovation,
            "predictive_mean": np.full(n, 20.0),
            "innovation": innovation,
            "z": innovation / scale,
            "pit": np.linspace(0.01, 0.99, n),
            "state_before": state,
            "state_var_before": state_var,
            "obs_var": obs_var,
            "predictive_var": state_var + obs_var,
            "kalman_gain": state_var / (state_var + obs_var),
            "games_in_model_before": games,
            "season_game_number": games % 20 + 1,
            "returning_player": np.ones(n, dtype=bool),
            "season_boundary": (games % 20) == 0,
            "new_team_flag": (games % 17) == 0,
            "rookie_flag": (games % 19) == 0,
            "df": np.full(n, 100.0),
        }
    )


def test_games_played_tiers_have_requested_boundaries() -> None:
    tier = games_played_tier(pd.Series([0, 9, 10, 29, 30, 59, 60]))
    assert tier.astype(str).tolist() == [
        "0-9", "0-9", "10-29", "10-29", "30-59", "30-59", "60+"
    ]


def test_variance_shares_sum_to_one_and_keep_coverage() -> None:
    result = variance_decomposition(_trace())
    assert np.allclose(
        result["state_variance_share"] + result["observation_variance_share"],
        1.0,
    )
    assert {"coverage_50", "coverage_80", "coverage_90", "coverage_95"}.issubset(
        result.columns
    )


def test_shrinkage_detects_regression_toward_zero() -> None:
    by_tier, by_decile = shrinkage_summary(_trace())
    assert {"mean_z", "mean_innovation"}.issubset(by_tier.columns)
    assert by_decile.iloc[0]["mean_innovation"] > 0
    assert by_decile.iloc[-1]["mean_innovation"] < 0
    assert bool(by_decile.iloc[0]["toward_zero"])
    assert bool(by_decile.iloc[-1]["toward_zero"])


def test_offseason_summary_compares_early_and_late() -> None:
    windows, relationship = offseason_summary(_trace())
    assert set(windows["window"]) == {5, 10, 20}
    assert set(windows["period"]) == {"early", "rest"}
    assert relationship["n"] > 0
    assert relationship["slope"] < 0


def test_group_flags_respect_minimum_sample() -> None:
    result = group_calibration(_trace(), "position", min_n=50)
    assert {"mean_z_se", "sd_z_se", "mean_flag", "sd_flag", "low_sample"}.issubset(
        result.columns
    )
    assert result["low_sample"].all()
    assert not result["actionable_flag"].any()


def test_core_metrics_name_sample_and_seasons() -> None:
    result = core_calibration(_trace())
    assert result["n"] == 80
    assert result["seasons"] == ("2024-25", "2025-26")
    assert 0 <= result["coverage_80"] <= 1


def test_draw_metrics_skip_cleanly_without_draws() -> None:
    result = posterior_draw_summary(_trace(), draws=None)
    assert result == {"status": "skipped", "reason": "posterior draws unavailable"}
```

- [ ] **Step 2: Run tests and confirm the expected import failure**

Run:

```bash
.venv/bin/python -m pytest tests/models/test_dynamic_points_diagnostics.py -q
```

Expected: collection fails with `ModuleNotFoundError` for the diagnostics module.

- [ ] **Step 3: Implement common calibration helpers**

Create `src/models/dynamic_points_diagnostics.py`. Validate required columns
at each public boundary. Implement:

```python
def games_played_tier(games: pd.Series) -> pd.Categorical:
    return pd.cut(
        games,
        bins=[-1, 9, 29, 59, np.inf],
        labels=["0-9", "10-29", "30-59", "60+"],
    )


def _coverage(frame: pd.DataFrame, level: float) -> float:
    quantile = stats.t.ppf(
        (1.0 + level) / 2.0,
        frame["df"].to_numpy(dtype=float),
    )
    half_width = quantile * np.sqrt(frame["predictive_var"].to_numpy(dtype=float))
    error = np.abs(frame["innovation"].to_numpy(dtype=float))
    return float(np.mean(error <= half_width))


def core_calibration(trace: pd.DataFrame) -> dict[str, object]:
    z = trace["z"].to_numpy(dtype=float)
    innovation = trace["innovation"].to_numpy(dtype=float)
    result: dict[str, object] = {
        "n": len(trace),
        "seasons": tuple(sorted(trace["season_year"].astype(str).unique())),
        "mean_z": float(np.mean(z)),
        "sd_z": float(np.std(z, ddof=1)),
        "rmse": float(np.sqrt(np.mean(innovation**2))),
        "mae": float(np.mean(np.abs(innovation))),
    }
    for level in (0.50, 0.80, 0.90, 0.95):
        result[f"coverage_{int(level * 100)}"] = _coverage(trace, level)
    result["pit_histogram"] = np.histogram(
        trace["pit"].to_numpy(dtype=float),
        bins=np.linspace(0.0, 1.0, 11),
    )[0]
    return result
```

- [ ] **Step 4: Implement variance and shrinkage summaries**

`variance_decomposition` groups by `games_played_tier` and reports `n`,
state/observation shares, `mean_z`, `sd_z`, and all four coverage levels.

`shrinkage_summary` returns two frames. The tier frame reports `n`, `mean_z`,
and `mean_innovation`. The decile frame uses
`pd.qcut(state_before, q=10, duplicates="drop")`, reports innovation mean,
standard error, and:

```python
summary["toward_zero"] = (
    ((summary["mean_state"] < 0) & (summary["mean_innovation"] > 0))
    | ((summary["mean_state"] > 0) & (summary["mean_innovation"] < 0))
)
summary["significant"] = (
    summary["mean_innovation"].abs() > 2.0 * summary["innovation_se"]
)
summary["under_shrinkage_flag"] = (
    summary["extreme_decile"] & summary["toward_zero"] & summary["significant"]
)
```

Only the first and last populated deciles set `extreme_decile=True`.

- [ ] **Step 5: Implement offseason, group, and gain summaries**

`offseason_summary` filters `returning_player`, loops over windows `(5, 10, 20)`,
and emits separate early/rest rows with n, mean z, SD(z), and four coverage
levels. Fit `scipy.stats.linregress(state_before, innovation)` on rows with
`season_game_number <= 20`; return n, slope, intercept, correlation, p-value,
and slope standard error.

`group_calibration(trace, group_column, min_n)` must calculate:

```python
summary["mean_z_se"] = summary["sd_z"] / np.sqrt(summary["n"])
summary["sd_z_se"] = summary["sd_z"] / np.sqrt(2.0 * (summary["n"] - 1))
summary["mean_flag"] = summary["mean_z"].abs() > 2.0 * summary["mean_z_se"]
summary["sd_flag"] = (
    (summary["sd_z"] - 1.0).abs() > 2.0 * summary["sd_z_se"]
)
summary["low_sample"] = summary["n"] < min_n
summary["actionable_flag"] = (
    ~summary["low_sample"] & (summary["mean_flag"] | summary["sd_flag"])
)
```

`gain_summary` returns a tier summary, a 20-quantile games summary, and a
boundary/non-boundary summary. Flag positive non-boundary gain jumps exceeding
two standard errors between adjacent fine bins; do not flag season-boundary
jumps as unexplained.

- [ ] **Step 6: Implement optional draw metrics**

`posterior_draw_summary(trace, draws)` must return the exact skipped dictionary
when `draws is None`. Otherwise require a finite two-dimensional array shaped
`(len(trace), n_draws)`, use `src.models.evaluation.crps` and
`probability_integral_transform`, and compare them with Gaussian approximation:

```python
gaussian_pit = stats.norm.cdf(trace["z"].to_numpy(dtype=float))
draw_pit = probability_integral_transform(
    draws,
    trace["points"].to_numpy(dtype=float),
)
```

Use the closed-form Gaussian CRPS for each row and report draw CRPS, Gaussian
CRPS, their difference, draw PIT mean/SD, Gaussian PIT mean/SD, and PIT
mean/SD differences.

- [ ] **Step 7: Run diagnostic tests**

Run:

```bash
.venv/bin/python -m pytest tests/models/test_dynamic_points_diagnostics.py -q
```

Expected: all seven tests pass.

- [ ] **Step 8: Run all focused checks**

Run:

```bash
.venv/bin/python -m pytest \
  tests/models/test_dynamic_points.py \
  tests/models/test_dynamic_points_diagnostics.py -q
.venv/bin/ruff check \
  src/models/dynamic_points.py \
  src/models/dynamic_points_diagnostics.py \
  tests/models/test_dynamic_points.py \
  tests/models/test_dynamic_points_diagnostics.py
.venv/bin/mypy \
  src/models/dynamic_points.py \
  src/models/dynamic_points_diagnostics.py
```

Expected: all commands exit 0 with no warnings.

- [ ] **Step 9: Commit diagnostics**

```bash
git add \
  src/models/dynamic_points_diagnostics.py \
  tests/models/test_dynamic_points_diagnostics.py
git commit -m "feat: add dynamic points diagnostics"
```

---

### Task 4: Standalone evaluation and live-state notebook

**Files:**
- Modify: `notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb`

**Interfaces:**
- Consumes: `build_causal_frame`, `walk_forward_trace`, all diagnostic functions.
- Produces: a standalone executed report; no Python module imports from another notebook.

- [ ] **Step 1: Replace notebook setup cells**

Use the notebook editor, not raw JSON editing. Replace the current cells with:

1. Markdown title, model equation, status, leakage contract, and explicit
   distinction between walk-forward evaluation and the final live state.
2. Imports and project-root discovery:

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
if not (ROOT / "pyproject.toml").exists():
    raise RuntimeError("run this notebook from inside the nba_quant repository")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dynamic_points import (
    DynamicPointsConfig,
    build_causal_frame,
    walk_forward_trace,
)
from src.models.dynamic_points_diagnostics import (
    core_calibration,
    gain_summary,
    group_calibration,
    offseason_summary,
    posterior_draw_summary,
    shrinkage_summary,
    variance_decomposition,
)
```

3. Configuration:

```python
CONFIG = DynamicPointsConfig(
    state_drift_sd=0.35,
    offseason_drift_sd=2.0,
    train_window_seasons=2,
    min_group_n=30,
)
SEASON_PATHS = sorted(
    ROOT.glob("data/silver/nba/*/regular_season/player_gamelogs.parquet")
)
if len(SEASON_PATHS) < CONFIG.train_window_seasons + 1:
    raise ValueError("need at least three complete season parquet files")
```

- [ ] **Step 2: Add data and walk-forward cells**

Load each parquet, retain regular-season rows, concatenate, build the causal
frame, and print the returned audit. Determine evaluation seasons from sorted
available seasons after the first two:

```python
available_seasons = sorted(
    causal["season_year"].astype(str).unique(),
    key=lambda value: int(value.split("-")[0]),
)
evaluation_seasons = tuple(available_seasons[CONFIG.train_window_seasons :])
trace, fold_audits = walk_forward_trace(
    causal,
    evaluation_seasons=evaluation_seasons,
    config=CONFIG,
)
display(pd.DataFrame(fold_audits))
```

Assert the trace is nonempty, finite for numeric diagnostic columns, and has
one row per player/game/fold.

- [ ] **Step 3: Add core, variance, and shrinkage report cells**

Display `core_calibration(trace)`, the PIT histogram, variance decomposition,
games-tier coverage comparison, tier shrinkage, and state-decile shrinkage.
Plots must label sample size and evaluation seasons. Add horizontal reference
lines at z mean 0 and z SD 1.

- [ ] **Step 4: Add offseason, group, and gain report cells**

Display all early windows and the innovation/state relationship. Run group
calibration for `opp_team_id`, `team_id`, and `position`; display actionable
flags first while retaining low-sample rows. Display separate new-team and
rookie slices by calling a two-level boolean-slice helper or grouping those
columns directly.

Plot mean gain against fine games-in-model bins and overlay season-boundary
means. Print any unexplained non-boundary jump rows.

- [ ] **Step 5: Add draw status and final live-state section**

Call:

```python
draw_report = posterior_draw_summary(trace, draws=None)
print("Posterior-draw diagnostics:", draw_report)
```

For the live state, use all but the latest available season as the training
window only when exactly two prior seasons are available; otherwise use the
latest two complete prior seasons. Generate the latest season causally with
the same `walk_forward_trace` API and display each player's most recent state,
state SD, games in model, and latest predictive diagnostics. Label this as
state monitoring, not a future-game forecast, because no future schedule/role
row is supplied.

- [ ] **Step 6: Execute the notebook end to end**

Run:

```bash
mkdir -p .superpowers/scratch
.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb \
  --output "$PWD/.superpowers/scratch/dynamic_hierarchical_bayesian_model.executed.ipynb" \
  --ExecutePreprocessor.timeout=1800
```

Expected: exit 0 and an executed scratch notebook. Do not commit the executed
artifact.

- [ ] **Step 7: Inspect execution output for failures and sample sizes**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(".superpowers/scratch/dynamic_hierarchical_bayesian_model.executed.ipynb")
notebook = json.loads(path.read_text())
errors = [
    output
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
assert not errors, errors
assert any(
    "Posterior-draw diagnostics:" in "".join(output.get("text", []))
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
)
print(f"executed_cells={sum(cell.get('execution_count') is not None for cell in notebook['cells'])}")
print("notebook_errors=0")
PY
```

Expected: `notebook_errors=0`.

- [ ] **Step 8: Commit the standalone notebook**

```bash
git add notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb
git commit -m "feat: add dynamic points evaluation notebook"
```

---

### Task 5: Full verification and deployment report

**Files:**
- Modify only if verification exposes a defect:
  `src/models/dynamic_points.py`,
  `src/models/dynamic_points_diagnostics.py`,
  `tests/models/test_dynamic_points.py`,
  `tests/models/test_dynamic_points_diagnostics.py`,
  `notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb`

**Interfaces:**
- Consumes: completed model, diagnostics, and notebook.
- Produces: fresh test/static-check evidence and an honest deployment-readiness summary.

- [ ] **Step 1: Run focused and adjacent tests**

```bash
.venv/bin/python -m pytest \
  tests/models/test_dynamic_points.py \
  tests/models/test_dynamic_points_diagnostics.py \
  tests/models/test_evaluation.py \
  tests/models/test_season_folds.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run final lint and type checks**

```bash
.venv/bin/ruff check \
  src/models/dynamic_points.py \
  src/models/dynamic_points_diagnostics.py \
  tests/models/test_dynamic_points.py \
  tests/models/test_dynamic_points_diagnostics.py
.venv/bin/mypy \
  src/models/dynamic_points.py \
  src/models/dynamic_points_diagnostics.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Re-execute the final notebook**

Repeat Task 4 Step 6 after all fixes. Expected: exit 0 with no notebook error
outputs.

- [ ] **Step 4: Verify leakage and trace invariants from the executed report**

Confirm every fold audit has disjoint train/evaluation seasons, training dates
strictly before evaluation dates, positive prediction counts, and reported
imputation counts. Confirm `predictive_var == state_var_before + obs_var` and
all z/PIT values are finite.

- [ ] **Step 5: Report results without declaring automatic deployment approval**

Report:

- exact evaluation seasons and sample size;
- overall mean z, SD(z), and 80%/90%/95% coverage;
- worst experience-tier coverage;
- shrinkage extreme-decile flags;
- offseason SD(z) and innovation/state slope;
- actionable team/opponent/position flags;
- whether gain behavior has unexplained jumps;
- draw diagnostic skipped status;
- test, lint, type-check, and notebook-execution commands.

If any deployment gate in the design spec fails, state that the model is not
ready for live use and name the failed gate. Do not tune hyperparameters on
these reported evaluation seasons.

- [ ] **Step 6: Commit verification-only fixes, if any**

If Step 1–4 required code changes, rerun the failing check, then all checks, and
commit only those focused files:

```bash
git add \
  src/models/dynamic_points.py \
  src/models/dynamic_points_diagnostics.py \
  tests/models/test_dynamic_points.py \
  tests/models/test_dynamic_points_diagnostics.py \
  notebooks/nba/points/dynamic_hierarchical_bayesian_model.ipynb
git commit -m "fix: harden dynamic points evaluation"
```

If no files changed, do not create an empty commit.
