# Layer 4 Possession Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a NumPy Layer 4 module that allocates supplied team FGA across players, simulates shooting and secondary offensive stats, and runs a Boston 10k-sim demo with means/sds/percentiles.

**Architecture:** Pure functions under `nba_game_simulator/src/`: `opportunity_shares` for weight normalization; `possession_allocation.simulate_game` as the single public entry point; `rate_builders` optional PBP helper; FGA volume generation only in `demos/bos_layer4_demo.py`. Allocated pools (FGA/AST/OREB/TOV) reconcile via multinomial; FTA/FTM are player-generated NegBin/Binomial.

**Tech Stack:** Python 3.12+, NumPy, pytest. Optional pandas only inside `rate_builders.py` / demo summaries.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-05-layer4-possession-allocation-design.md`
- Core never generates `team_fga`; demo uses NegBin outside core
- `base_weights` = baseline share **before** minutes adjustment
- Fail loudly on invalid inputs; only soft constraint is `AST_pool = min(NB_draw, team_fgm)`
- No Dirichlet-multinomial, hierarchical Bayes, possession-by-possession engine, or plotting in `src/`
- Do not `git commit` unless the user asks (skip every Commit step until then)
- Tests: `.venv/bin/python -m pytest`
- Work in this checkout; do not create a worktree

## File map

- Create: `nba_game_simulator/src/__init__.py`
- Create: `nba_game_simulator/src/opportunity_shares.py`
- Create: `nba_game_simulator/src/possession_allocation.py`
- Create: `nba_game_simulator/src/rate_builders.py`
- Create: `nba_game_simulator/demos/bos_layer4_demo.py`
- Create: `tests/game_simulator/test_opportunity_shares.py`
- Create: `tests/game_simulator/test_possession_allocation.py`
- Existing: `tests/game_simulator/conftest.py` (already puts `nba_game_simulator/src` on `sys.path`)

---

### Task 1: Opportunity shares

**Files:**
- Create: `nba_game_simulator/src/__init__.py`
- Create: `nba_game_simulator/src/opportunity_shares.py`
- Create: `tests/game_simulator/test_opportunity_shares.py`

**Interfaces:**
- Produces:
  ```python
  def opportunity_shares(
      base_weights: np.ndarray,
      available: np.ndarray,
      projected_minutes: np.ndarray,
      reference_minutes: np.ndarray,
      role_factors: np.ndarray | None = None,
      lineup_factors: np.ndarray | None = None,
      minutes_exponent: float = 1.0,
  ) -> np.ndarray:  # shape (n_players,), sums to 1
  ```
- Consumes: nothing

- [ ] **Step 1: Write the failing tests**

```python
# tests/game_simulator/test_opportunity_shares.py
from __future__ import annotations

import numpy as np
import pytest

from opportunity_shares import opportunity_shares


def test_shares_normalize_and_zero_unavailable():
    p = opportunity_shares(
        base_weights=np.array([0.31, 0.27, 0.19, 0.15, 0.08]),
        available=np.array([1, 1, 1, 0, 1]),
        projected_minutes=np.array([36.0, 34.0, 32.0, 28.0, 20.0]),
        reference_minutes=np.array([36.0, 34.0, 32.0, 28.0, 20.0]),
    )
    assert p.shape == (5,)
    assert p[3] == 0.0
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0


def test_proj_equals_ref_shares_proportional_to_base_times_availability():
    w = np.array([0.4, 0.4, 0.2])
    p = opportunity_shares(
        base_weights=w,
        available=np.array([1.0, 1.0, 1.0]),
        projected_minutes=np.array([30.0, 30.0, 30.0]),
        reference_minutes=np.array([30.0, 30.0, 30.0]),
    )
    assert np.allclose(p, w / w.sum())


def test_higher_minute_ratio_increases_share():
    p = opportunity_shares(
        base_weights=np.array([0.5, 0.5]),
        available=np.array([1.0, 1.0]),
        projected_minutes=np.array([40.0, 20.0]),
        reference_minutes=np.array([30.0, 30.0]),
    )
    assert p[0] > p[1]


def test_all_unavailable_raises():
    with pytest.raises(ValueError, match="No available players"):
        opportunity_shares(
            base_weights=np.array([0.5, 0.5]),
            available=np.array([0.0, 0.0]),
            projected_minutes=np.array([30.0, 30.0]),
            reference_minutes=np.array([30.0, 30.0]),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_opportunity_shares.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'opportunity_shares'`

- [ ] **Step 3: Implement**

```python
# nba_game_simulator/src/__init__.py
"""NBA Monte Carlo game simulator layers."""

# nba_game_simulator/src/opportunity_shares.py
"""Layer 4 opportunity-share adjustment.

``base_weights`` are baseline shares *before* the minutes adjustment.
"""

from __future__ import annotations

import numpy as np


def opportunity_shares(
    base_weights: np.ndarray,
    available: np.ndarray,
    projected_minutes: np.ndarray,
    reference_minutes: np.ndarray,
    role_factors: np.ndarray | None = None,
    lineup_factors: np.ndarray | None = None,
    minutes_exponent: float = 1.0,
) -> np.ndarray:
    base_weights = np.asarray(base_weights, dtype=float)
    available = np.asarray(available, dtype=float)
    projected_minutes = np.asarray(projected_minutes, dtype=float)
    reference_minutes = np.asarray(reference_minutes, dtype=float)
    n = base_weights.shape[0]
    if role_factors is None:
        role_factors = np.ones(n, dtype=float)
    else:
        role_factors = np.asarray(role_factors, dtype=float)
    if lineup_factors is None:
        lineup_factors = np.ones(n, dtype=float)
    else:
        lineup_factors = np.asarray(lineup_factors, dtype=float)

    minute_ratio = np.divide(
        projected_minutes,
        reference_minutes,
        out=np.zeros_like(projected_minutes, dtype=float),
        where=reference_minutes > 0,
    )
    adjusted = (
        base_weights
        * available
        * np.power(minute_ratio, minutes_exponent)
        * role_factors
        * lineup_factors
    )
    total = float(adjusted.sum())
    if total <= 0:
        raise ValueError("No available players with positive opportunity weight")
    return adjusted / total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_opportunity_shares.py -v`

Expected: PASS

- [ ] **Step 5: Commit** — skip unless the user asks

---

### Task 2: Validation helpers and NegBin / multinomial utilities

**Files:**
- Create: `nba_game_simulator/src/possession_allocation.py` (helpers only in this task)
- Create: `tests/game_simulator/test_possession_allocation.py` (helper tests)

**Interfaces:**
- Produces:
  ```python
  def as_nonneg_int_vector(value, n_sims: int, name: str) -> np.ndarray: ...
  def sample_negbin(mu, alpha: float, rng: np.random.Generator, size=None) -> np.ndarray: ...
  def multinomial_rows(totals: np.ndarray, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
  ```
- Consumes: nothing from Task 1 yet

NegBin parameterization (NB2): `Var = μ + α μ²`. When `α <= 0`, use Poisson(`μ`). NumPy mapping: `n = 1/α`, `p = 1 / (1 + α μ)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/game_simulator/test_possession_allocation.py
from __future__ import annotations

import numpy as np
import pytest

from possession_allocation import (
    as_nonneg_int_vector,
    multinomial_rows,
    sample_negbin,
)


def test_as_nonneg_int_vector_broadcasts_scalar():
    out = as_nonneg_int_vector(90, n_sims=3, name="team_fga")
    assert out.dtype == np.int64
    assert np.array_equal(out, [90, 90, 90])


def test_as_nonneg_int_vector_rejects_negative_and_float():
    with pytest.raises(ValueError, match="team_fga"):
        as_nonneg_int_vector(-1, n_sims=2, name="team_fga")
    with pytest.raises(ValueError, match="team_fga"):
        as_nonneg_int_vector(np.array([1.5, 2.0]), n_sims=2, name="team_fga")


def test_as_nonneg_int_vector_rejects_wrong_length():
    with pytest.raises(ValueError, match="team_fga"):
        as_nonneg_int_vector(np.array([80, 90]), n_sims=3, name="team_fga")


def test_multinomial_rows_sum_to_totals():
    rng = np.random.default_rng(0)
    totals = np.array([10, 0, 25], dtype=np.int64)
    probs = np.array([0.4, 0.35, 0.25])
    out = multinomial_rows(totals, probs, rng)
    assert out.shape == (3, 3)
    assert np.array_equal(out.sum(axis=1), totals)


def test_sample_negbin_poisson_limit_nonnegative():
    rng = np.random.default_rng(1)
    draws = sample_negbin(mu=np.full(1000, 5.0), alpha=0.0, rng=rng)
    assert draws.shape == (1000,)
    assert draws.min() >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py -v`

Expected: FAIL with import / missing symbols

- [ ] **Step 3: Implement helpers**

```python
# nba_game_simulator/src/possession_allocation.py
"""Layer 4: allocate team offensive opportunities to players."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opportunity_shares import opportunity_shares


def as_nonneg_int_vector(value: Any, n_sims: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = np.full(n_sims, int(arr), dtype=np.int64)
    else:
        arr = np.asarray(arr).reshape(-1)
        if arr.shape[0] != n_sims:
            raise ValueError(f"{name} must be scalar or length n_sims={n_sims}")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.allclose(arr, np.round(arr)):
            raise ValueError(f"{name} must be nonnegative integers")
        arr = np.round(arr).astype(np.int64)
    else:
        arr = arr.astype(np.int64)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative integers")
    return arr


def sample_negbin(
    mu: np.ndarray | float,
    alpha: float,
    rng: np.random.Generator,
    size: tuple[int, ...] | None = None,
) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float)
    if size is None:
        size = mu_arr.shape
    mu_arr = np.broadcast_to(mu_arr, size)
    if alpha <= 0:
        return rng.poisson(np.maximum(mu_arr, 0.0), size=size).astype(np.int64)
    # NB2: mean mu, var = mu + alpha * mu^2
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * np.maximum(mu_arr, 0.0))
    return rng.negative_binomial(n, p, size=size).astype(np.int64)


def multinomial_rows(
    totals: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    totals = np.asarray(totals, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be 1-d for v1 multinomial_rows")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("probs must sum to 1")
    out = np.zeros((totals.shape[0], probs.shape[0]), dtype=np.int64)
    for i, n in enumerate(totals):
        if n < 0:
            raise ValueError("totals must be nonnegative")
        out[i] = rng.multinomial(int(n), probs)
    return out
```

- [ ] **Step 4: Run helper tests**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py -v`

Expected: PASS for the five helper tests

- [ ] **Step 5: Commit** — skip unless the user asks

---

### Task 3: FGA allocation + shooting outcomes

**Files:**
- Modify: `nba_game_simulator/src/possession_allocation.py`
- Modify: `tests/game_simulator/test_possession_allocation.py`

**Interfaces:**
- Produces (internal, used by `simulate_game`):
  ```python
  def allocate_and_shoot(
      team_fga: np.ndarray,          # (n_sims,)
      shares: np.ndarray,            # (n_players,)
      p_3pa: np.ndarray,
      p_3p: np.ndarray,
      p_2p: np.ndarray,
      rng: np.random.Generator,
  ) -> dict[str, np.ndarray]:  # FGA,2PA,2PM,3PA,3PM,FGM each (n_sims, n_players)
  ```
- Consumes: `multinomial_rows`

- [ ] **Step 1: Write the failing tests**

```python
from possession_allocation import allocate_and_shoot


def test_fga_rows_sum_to_team_total():
    rng = np.random.default_rng(2)
    team_fga = np.array([80, 90, 100], dtype=np.int64)
    shares = np.array([0.31, 0.27, 0.19, 0.15, 0.08])
    out = allocate_and_shoot(
        team_fga,
        shares,
        p_3pa=np.full(5, 0.4),
        p_3p=np.full(5, 0.35),
        p_2p=np.full(5, 0.55),
        rng=rng,
    )
    assert np.array_equal(out["FGA"].sum(1), team_fga)
    assert np.array_equal(out["2PA"] + out["3PA"], out["FGA"])
    assert np.all(out["3PM"] <= out["3PA"])
    assert np.all(out["2PM"] <= out["2PA"])
    assert np.array_equal(out["FGM"], out["2PM"] + out["3PM"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py::test_fga_rows_sum_to_team_total -v`

Expected: FAIL with `ImportError` / missing `allocate_and_shoot`

- [ ] **Step 3: Implement `allocate_and_shoot`**

```python
def allocate_and_shoot(
    team_fga: np.ndarray,
    shares: np.ndarray,
    p_3pa: np.ndarray,
    p_3p: np.ndarray,
    p_2p: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    fga = multinomial_rows(team_fga, shares, rng)
    n_sims, n_players = fga.shape
    three_pa = np.zeros_like(fga)
    two_pm = np.zeros_like(fga)
    three_pm = np.zeros_like(fga)
    for j in range(n_players):
        three_pa[:, j] = rng.binomial(fga[:, j], float(p_3pa[j]))
        three_pm[:, j] = rng.binomial(three_pa[:, j], float(p_3p[j]))
        two_pa_j = fga[:, j] - three_pa[:, j]
        two_pm[:, j] = rng.binomial(two_pa_j, float(p_2p[j]))
    two_pa = fga - three_pa
    fgm = two_pm + three_pm
    return {
        "FGA": fga,
        "2PA": two_pa,
        "2PM": two_pm,
        "3PA": three_pa,
        "3PM": three_pm,
        "FGM": fgm,
    }
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py::test_fga_rows_sum_to_team_total -v`

Expected: PASS

- [ ] **Step 5: Commit** — skip unless the user asks

---

### Task 4: FTA/FTM and AST/OREB/TOV pools

**Files:**
- Modify: `nba_game_simulator/src/possession_allocation.py`
- Modify: `tests/game_simulator/test_possession_allocation.py`

**Interfaces:**
- Produces:
  ```python
  def simulate_fta_ftm(fga, fta_per_fga, ft_pct, alpha_fta, rng) -> tuple[np.ndarray, np.ndarray]
  def allocate_from_pool(pool_totals, alloc_probs, rng) -> np.ndarray
  def renormalize_props(weights: np.ndarray) -> np.ndarray
  ```
- Consumes: `sample_negbin`, `multinomial_rows`

- [ ] **Step 1: Write the failing tests**

```python
from possession_allocation import allocate_from_pool, renormalize_props, simulate_fta_ftm


def test_fta_ftm_player_generated_not_team_sum_constrained():
    rng = np.random.default_rng(3)
    fga = np.array([[10, 5], [8, 8]], dtype=np.int64)
    fta, ftm = simulate_fta_ftm(
        fga,
        fta_per_fga=np.array([0.3, 0.2]),
        ft_pct=np.array([0.8, 0.75]),
        alpha_fta=0.1,
        rng=rng,
    )
    assert fta.shape == fga.shape
    assert np.all(ftm <= fta)


def test_allocate_from_pool_sums_and_ast_cap_pattern():
    rng = np.random.default_rng(4)
    pool = np.array([10, 7], dtype=np.int64)
    props = renormalize_props(np.array([0.5, 0.3, 0.2]))
    out = allocate_from_pool(pool, props, rng)
    assert np.array_equal(out.sum(1), pool)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py::test_fta_ftm_player_generated_not_team_sum_constrained tests/game_simulator/test_possession_allocation.py::test_allocate_from_pool_sums_and_ast_cap_pattern -v`

Expected: FAIL (missing symbols)

- [ ] **Step 3: Implement**

```python
def renormalize_props(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError("allocation weights must sum to a positive value")
    return w / total


def simulate_fta_ftm(
    fga: np.ndarray,
    fta_per_fga: np.ndarray,
    ft_pct: np.ndarray,
    alpha_fta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mu = fga.astype(float) * np.asarray(fta_per_fga, dtype=float)[None, :]
    fta = sample_negbin(mu, alpha_fta, rng, size=fga.shape)
    ftm = np.zeros_like(fta)
    for j in range(fga.shape[1]):
        ftm[:, j] = rng.binomial(fta[:, j], float(ft_pct[j]))
    return fta, ftm


def allocate_from_pool(
    pool_totals: np.ndarray,
    alloc_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return multinomial_rows(pool_totals, alloc_probs, rng)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py::test_fta_ftm_player_generated_not_team_sum_constrained tests/game_simulator/test_possession_allocation.py::test_allocate_from_pool_sums_and_ast_cap_pattern -v`

Expected: PASS

- [ ] **Step 5: Commit** — skip unless the user asks

---

### Task 5: `simulate_game` public API + `SimulationResult`

**Files:**
- Modify: `nba_game_simulator/src/possession_allocation.py`
- Modify: `tests/game_simulator/test_possession_allocation.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class TeamSimResult:
      player_ids: np.ndarray
      opportunity_shares: np.ndarray
      FGA: np.ndarray
      FGM: np.ndarray
      two_pa: np.ndarray  # expose as fields matching spec names
      # ... 2PA,2PM,3PA,3PM,FTA,FTM,PTS,AST,TOV,OREB
      metadata: dict  # e.g. {"tov_pool_approximate": True}

  @dataclass
  class SimulationResult:
      teams: dict[str, TeamSimResult]

  def simulate_game(
      team_inputs: dict[str, dict],
      player_inputs: dict[str, dict],
      n_sims: int,
      rng: np.random.Generator,
  ) -> SimulationResult: ...
  ```
- Consumes: `opportunity_shares`, all helpers from Tasks 2–4

Use dict keys in `TeamSimResult` arrays with exact names: `FGA`, `FGM`, `2PA`, `2PM`, `3PA`, `3PM`, `FTA`, `FTM`, `PTS`, `AST`, `TOV`, `OREB` — store as attributes with those names where valid (`two_pa` alias not required if using a dict payload). Prefer a dataclass with attribute names `fga`, `fgm`, … **or** a nested dict `arrays`. Spec asks for predictable shapes; implement:

```python
@dataclass(frozen=True)
class TeamSimResult:
    player_ids: tuple[str, ...]
    opportunity_shares: np.ndarray
    stats: dict[str, np.ndarray]  # keys listed above
    metadata: dict[str, Any]
```

- [ ] **Step 1: Write the failing integration tests**

```python
from possession_allocation import simulate_game


def _bos_players():
    return {
        "player_ids": np.array(["Tatum", "Brown", "White", "Porzingis", "Other"]),
        "base_weights": np.array([0.31, 0.27, 0.19, 0.15, 0.08]),
        "available": np.ones(5),
        "projected_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "reference_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "p_3pa": np.array([0.45, 0.40, 0.55, 0.35, 0.40]),
        "p_3p": np.array([0.37, 0.35, 0.40, 0.37, 0.34]),
        "p_2p": np.array([0.55, 0.54, 0.52, 0.58, 0.50]),
        "fta_per_fga": np.array([0.28, 0.25, 0.15, 0.30, 0.18]),
        "ft_pct": np.array([0.83, 0.78, 0.88, 0.84, 0.75]),
        "tov_rate": np.array([0.12, 0.11, 0.09, 0.10, 0.10]),
        "assist_propensity": np.array([0.25, 0.20, 0.35, 0.10, 0.10]),
        "oreb_propensity": np.array([0.15, 0.15, 0.10, 0.40, 0.20]),
        "assist_rate": 0.60,
        "oreb_rate": 0.28,
        "dispersion": {"fta": 0.05, "ast": 0.05, "oreb": 0.05, "tov": 0.05},
    }


def test_simulate_game_reconciles_fga_and_ast_cap():
    rng = np.random.default_rng(5)
    n_sims = 200
    result = simulate_game(
        team_inputs={
            "BOS": {
                "possessions": 101,
                "team_fga": np.full(n_sims, 90, dtype=np.int64),
            }
        },
        player_inputs={"BOS": _bos_players()},
        n_sims=n_sims,
        rng=rng,
    )
    bos = result.teams["BOS"]
    fga = bos.stats["FGA"]
    assert np.array_equal(fga.sum(1), np.full(n_sims, 90))
    assert np.all(bos.stats["AST"].sum(1) <= bos.stats["FGM"].sum(1))
    assert np.array_equal(
        bos.stats["PTS"],
        2 * bos.stats["2PM"] + 3 * bos.stats["3PM"] + bos.stats["FTM"],
    )


def test_simulate_game_reproducible_with_seed():
    def run(seed):
        return simulate_game(
            team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
            player_inputs={"BOS": _bos_players()},
            n_sims=50,
            rng=np.random.default_rng(seed),
        )

    a = run(99)
    b = run(99)
    assert np.array_equal(a.teams["BOS"].stats["PTS"], b.teams["BOS"].stats["PTS"])


def test_provisional_tov_metadata_when_team_tov_omitted():
    result = simulate_game(
        team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
        player_inputs={"BOS": _bos_players()},
        n_sims=20,
        rng=np.random.default_rng(6),
    )
    assert result.teams["BOS"].metadata.get("tov_pool_approximate") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py::test_simulate_game_reconciles_fga_and_ast_cap tests/game_simulator/test_possession_allocation.py::test_simulate_game_reproducible_with_seed tests/game_simulator/test_possession_allocation.py::test_provisional_tov_metadata_when_team_tov_omitted -v`

Expected: FAIL (`simulate_game` missing)

- [ ] **Step 3: Implement `simulate_game`**

Implementation outline (write the full function in the module):

1. For each team key in `team_inputs` (must exist in `player_inputs`):
2. `shares = opportunity_shares(...)` from player fields.
3. `team_fga = as_nonneg_int_vector(team["team_fga"], n_sims, "team_fga")`.
4. `shot = allocate_and_shoot(...)`.
5. `fta, ftm = simulate_fta_ftm(...)`.
6. `team_fgm = shot["FGM"].sum(1)`; `team_miss = team_fga - team_fgm`.
7. AST pool: `mu_ast = team_fgm * assist_rate`; draw NB; `pool = minimum(draw, team_fgm)`; allocate with `renormalize_props(assist_propensity)`.
8. OREB pool: `mu_oreb = team_miss * oreb_rate`; draw NB; allocate with `renormalize_props(oreb_propensity)`.
9. TOV: if `team_tov` in team inputs, use it; else provisional `NB(poss * sum(shares * tov_rate), α)` and set metadata flag; allocate with `renormalize_props(shares * tov_rate)`.
10. `PTS = 2*2PM + 3*3PM + FTM`.
11. Pack `TeamSimResult` / `SimulationResult`.

Default missing dispersion alphas to `0.05`. Default `assist_rate=0.55`, `oreb_rate=0.25` if omitted.

- [ ] **Step 4: Run integration tests**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_possession_allocation.py -v`

Expected: all PASS

- [ ] **Step 5: Commit** — skip unless the user asks

---

### Task 6: Optional PBP rate builder (minimal)

**Files:**
- Create: `nba_game_simulator/src/rate_builders.py`
- Modify: `tests/game_simulator/test_possession_allocation.py` (or create `test_rate_builders.py` with a tiny synthetic frame)

**Interfaces:**
- Produces:
  ```python
  def build_player_rates_from_pbp(
      pbp: "pd.DataFrame",
      team_tricode: str,
      player_name_map: dict[int, str] | None = None,
  ) -> dict[str, np.ndarray | float]:
  ```
  Returns a partial `PlayerInputs`-compatible dict: `player_ids`, `base_weights` from FGA share, `p_3pa`, `p_3p`, `p_2p`, and simple rate stubs where countable. Does not invent minutes; caller supplies minutes/availability.

- [ ] **Step 1: Write a failing unit test on a synthetic DataFrame**

```python
# tests/game_simulator/test_rate_builders.py
import numpy as np
import pandas as pd
import pytest

from rate_builders import build_player_rates_from_pbp


def test_build_player_rates_from_synthetic_pbp():
    rows = [
        dict(teamTricode="BOS", personId=1, isFieldGoal=1, shotValue=3, shotResult="Made", actionType="Made Shot", subType="Jump Shot"),
        dict(teamTricode="BOS", personId=1, isFieldGoal=1, shotValue=2, shotResult="Missed", actionType="Missed Shot", subType="Layup"),
        dict(teamTricode="BOS", personId=2, isFieldGoal=1, shotValue=2, shotResult="Missed", actionType="Missed Shot", subType="Jump Shot"),
    ]
    rates = build_player_rates_from_pbp(pd.DataFrame(rows), team_tricode="BOS")
    assert set(rates["player_ids"].tolist()) == {1, 2}
    assert pytest.approx(float(rates["base_weights"].sum()), abs=1e-9) == 1.0
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_rate_builders.py -v`

Expected: FAIL missing module

- [ ] **Step 3: Implement minimal builder**

Count FGA / 3PA / makes by `personId` for `teamTricode`. `base_weights = player_fga / team_fga`. `p_3pa = 3PA/FGA` (0 if FGA=0). `p_3p = 3PM/3PA`, `p_2p = 2PM/2PA` with safe zeros. Leave other rates to caller defaults in the demo. Keep the function short; no season joins.

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_rate_builders.py -v`

Expected: PASS

- [ ] **Step 5: Commit** — skip unless the user asks

---

### Task 7: Boston demo + summary tables

**Files:**
- Create: `nba_game_simulator/demos/__init__.py` (empty)
- Create: `nba_game_simulator/demos/bos_layer4_demo.py`

**Interfaces:**
- Produces demo-only helpers:
  ```python
  def draw_team_fga(possessions, fga_per_poss, alpha, n_sims, rng, family="negbin") -> np.ndarray
  def summarize_player_stats(stats: dict[str, np.ndarray], player_ids) -> "pd.DataFrame"
  ```
- Consumes: `simulate_game` from Task 5

- [ ] **Step 1: Implement demo helpers and `__main__`**

```python
# nba_game_simulator/demos/bos_layer4_demo.py
"""Standalone Layer 4 Boston demo. Generates team FGA outside the core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from possession_allocation import sample_negbin, simulate_game


def draw_team_fga(possessions, fga_per_poss, alpha, n_sims, rng, family="negbin"):
    mu = float(possessions) * float(fga_per_poss)
    if family == "poisson":
        return sample_negbin(mu, alpha=0.0, rng=rng, size=(n_sims,))
    return sample_negbin(mu, alpha=alpha, rng=rng, size=(n_sims,))


def summarize_player_stats(stats, player_ids):
    rows = []
    for j, pid in enumerate(player_ids):
        for name, arr in stats.items():
            col = arr[:, j]
            rows.append(
                {
                    "player": pid,
                    "stat": name,
                    "mean": float(col.mean()),
                    "sd": float(col.std(ddof=0)),
                    "p25": float(np.percentile(col, 25)),
                    "p50": float(np.percentile(col, 50)),
                    "p75": float(np.percentile(col, 75)),
                }
            )
    return pd.DataFrame(rows)


def main():
    n_sims = 10_000
    rng = np.random.default_rng(42)
    possessions = 101
    team_fga = draw_team_fga(
        possessions, fga_per_poss=0.88, alpha=0.03, n_sims=n_sims, rng=rng
    )
    players = {
        "player_ids": np.array(["Tatum", "Brown", "White", "Porzingis", "Other"]),
        "base_weights": np.array([0.31, 0.27, 0.19, 0.15, 0.08]),
        "available": np.ones(5),
        "projected_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "reference_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "p_3pa": np.array([0.45, 0.40, 0.55, 0.35, 0.40]),
        "p_3p": np.array([0.37, 0.35, 0.40, 0.37, 0.34]),
        "p_2p": np.array([0.55, 0.54, 0.52, 0.58, 0.50]),
        "fta_per_fga": np.array([0.28, 0.25, 0.15, 0.30, 0.18]),
        "ft_pct": np.array([0.83, 0.78, 0.88, 0.84, 0.75]),
        "tov_rate": np.array([0.12, 0.11, 0.09, 0.10, 0.10]),
        "assist_propensity": np.array([0.25, 0.20, 0.35, 0.10, 0.10]),
        "oreb_propensity": np.array([0.15, 0.15, 0.10, 0.40, 0.20]),
        "assist_rate": 0.60,
        "oreb_rate": 0.28,
        "dispersion": {"fta": 0.05, "ast": 0.05, "oreb": 0.05, "tov": 0.05},
    }
    result = simulate_game(
        team_inputs={"BOS": {"possessions": possessions, "team_fga": team_fga}},
        player_inputs={"BOS": players},
        n_sims=n_sims,
        rng=rng,
    )
    bos = result.teams["BOS"]
    assert np.array_equal(bos.stats["FGA"].sum(1), team_fga)
    assert np.all(bos.stats["AST"].sum(1) <= bos.stats["FGM"].sum(1))
    summary = summarize_player_stats(bos.stats, bos.player_ids)
    print(summary.pivot(index="player", columns="stat", values="mean"))
    print(summary[summary["stat"].isin(["PTS", "FGA", "3PM", "FTM", "AST", "TOV", "OREB"])]
          .sort_values(["player", "stat"])
          .to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run: `.venv/bin/python nba_game_simulator/demos/bos_layer4_demo.py`

Expected: prints mean pivot and percentile table; exits 0; no assertion errors

- [ ] **Step 3: Run full Layer 4 test suite**

Run: `.venv/bin/python -m pytest tests/game_simulator/test_opportunity_shares.py tests/game_simulator/test_possession_allocation.py tests/game_simulator/test_rate_builders.py -v`

Expected: PASS

- [ ] **Step 4: Commit** — skip unless the user asks

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| Opportunity shares formula + defaults | Task 1 |
| `team_fga` scalar/vector validation | Task 2, 5 |
| Multinomial FGA allocation | Task 3, 5 |
| 2PA/3PA + makes | Task 3 |
| FTA NegBin / FTM Binomial (player-generated) | Task 4, 5 |
| AST/OREB/TOV team pools + multinomial | Task 4, 5 |
| AST ≤ TeamFGM | Task 5 |
| Provisional TOV metadata | Task 5 |
| Single `simulate_game` entry point | Task 5 |
| Optional PBP rate builder | Task 6 |
| Boston NegBin demo + summaries | Task 7 |
| Reproducibility / reconciliation tests | Tasks 1–5, 7 |

## Plan self-review notes

- No TBD placeholders in task steps.
- Helper signatures in Tasks 2–4 match Task 5 consumption (`sample_negbin`, `multinomial_rows`, `allocate_and_shoot`, `simulate_fta_ftm`, `allocate_from_pool`, `renormalize_props`).
- Duplicate TOV row in the spec table is intentional consolidation in Task 5 (one TOV path).
- Commits deferred until the user asks, matching prior repo plan convention.
