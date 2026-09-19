# Assists Feature Builder (current12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship leakage-safe `add_assists_features` / `build_pregame_assists_features` that emit `CURRENT_ASSISTS_FEATURES` (12) on synthetic frames.

**Architecture:** Dedicated `src/features/assists/` that reuses `numeric_column` / `ratio` only. Player windows are last-K qualifying priors (deque), not independent `prior_roll`. Team snapshots are null-aware, inclusive trailing-10, looked up with `merge_asof(allow_exact_matches=False)` after a global `game_date` sort. Align-back is positional `_row`, never label reindex.

**Tech Stack:** Python, pandas, numpy, unittest, pytest runner, existing `src/features/minutes/rolling.py`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-19-assists-feature-builder-design.md`
- Do not call `add_minutes_features` or modify `src/features/minutes/player.py`
- Do not call `prior_sum` / `prior_roll` on two series independently for the rate
- Official `ast` (not tracking-first `assists`); raise if neither `minutes` nor `min` exists
- Overwrite the 12 contract columns unconditionally; `predicted_minutes_oof` is `float64` IEEE NaN
- Preserve input row order and index via `_row = np.arange(len(frame))`; never `reindex` on labels
- Opponent join: `left_by="opp_team_id", right_by="team_id"`; sort both sides by `game_date` globally before `merge_asof`
- Only `days_rest` is season-scoped; team-game table does not store `season_year`
- Synthetic frames only; no silver, no `2025-26` reads, no joblib
- Tests: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py tests/features/test_pregame_assists.py -q`
- Do not `git commit` unless the user asks (skip every Commit step until then)

## File map

- Create: `src/features/assists/columns.py`
- Create: `src/features/assists/player.py`
- Create: `src/features/assists/team.py`
- Create: `src/features/assists/build.py`
- Create: `src/features/assists/pregame.py`
- Create: `src/features/assists/__init__.py`
- Create: `tests/features/test_assists_features.py`
- Create: `tests/features/test_pregame_assists.py`

---

### Task 1: Package, contract, skeleton builder

**Files:**
- Create: `src/features/assists/columns.py`
- Create: `src/features/assists/build.py`
- Create: `src/features/assists/__init__.py`
- Test: `tests/features/test_assists_features.py`

**Interfaces:**
- Produces: `CURRENT_ASSISTS_FEATURES: list[str]` (12 names, this order), `ASSISTS_FEATURE_CHALLENGERS: dict[str, list[str]]`, `add_assists_features(frame: pd.DataFrame) -> pd.DataFrame`
- Consumes: nothing from later tasks

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_assists_features.py`:

```python
"""Causal assists features must not read the current game."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.assists import (
    ASSISTS_FEATURE_CHALLENGERS,
    CURRENT_ASSISTS_FEATURES,
    add_assists_features,
)

CONTRACT = [
    "predicted_minutes_oof",
    "start_rate_10",
    "ast_mean_10",
    "ast_mean_20",
    "assists_per_min_10",
    "team_ast_mean_10",
    "team_fgm_mean_10",
    "team_pace_mean_10",
    "opponent_team_ast_allowed_mean_10",
    "opponent_team_pace_mean_10",
    "days_rest",
    "is_home",
]


def _row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    ast: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    season_year: str = "2024-25",
    start_position: str = "G",
    matchup: str = "DET vs. CLE",
    team_ast: float = 20.0,
    team_fgm: float = 40.0,
    team_pace: float = 100.0,
    opp_ast: float = 22.0,
    assists: float | None = None,
) -> dict:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": season_year,
        "matchup": matchup,
        "minutes": minutes,
        "min": minutes,
        "min_sec": "00:00",
        "start_position": start_position,
        "ast": ast,
        "assists": ast if assists is None else assists,
        "team_ast": team_ast,
        "team_fgm": team_fgm,
        "team_pace": team_pace,
        "opp_ast": opp_ast,
        "opp_pace": team_pace,
        "pass": 10.0,
        "tchs": 20.0,
        "sast": 1.0,
        "ftast": 1.0,
    }


class ContractTests(unittest.TestCase):
    def test_current_assists_features_are_the_twelve(self) -> None:
        self.assertEqual(list(CURRENT_ASSISTS_FEATURES), CONTRACT)
        self.assertEqual(
            ASSISTS_FEATURE_CHALLENGERS["passing_tracking"],
            ["passes_per_min_10"],
        )

    def test_predicted_minutes_oof_is_float64_nan(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=5, game_id=1)]
        )
        featured = add_assists_features(frame)
        self.assertIn("predicted_minutes_oof", featured.columns)
        self.assertEqual(
            featured["predicted_minutes_oof"].dtype,
            np.float64,
        )
        self.assertTrue(
            featured["predicted_minutes_oof"].isna().all()
        )
        self.assertFalse(
            featured["predicted_minutes_oof"].isna().astype("object").eq(pd.NA).any()
        )

    def test_missing_minutes_column_raises(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=5, game_id=1)]
        ).drop(columns=["minutes", "min"])
        with self.assertRaises(ValueError):
            add_assists_features(frame)

    def test_preserves_duplicate_unsorted_index(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-05", 30, ast=8, game_id=3),
                _row(1, "2024-01-01", 10, ast=2, game_id=1),
                _row(1, "2024-01-03", 20, ast=4, game_id=2),
            ]
        )
        frame.index = pd.Index([7, 7, 2], name="dup")
        featured = add_assists_features(frame)
        self.assertTrue(featured.index.equals(frame.index))
        self.assertEqual(
            list(featured["game_id"]),
            [3, 1, 2],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::ContractTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.assists'`

- [ ] **Step 3: Write minimal implementation**

`src/features/assists/columns.py`:

```python
"""Assists feature manifests."""

from __future__ import annotations

CURRENT_ASSISTS_FEATURES = [
    "predicted_minutes_oof",
    "start_rate_10",
    "ast_mean_10",
    "ast_mean_20",
    "assists_per_min_10",
    "team_ast_mean_10",
    "team_fgm_mean_10",
    "team_pace_mean_10",
    "opponent_team_ast_allowed_mean_10",
    "opponent_team_pace_mean_10",
    "days_rest",
    "is_home",
]

ASSISTS_FEATURE_CHALLENGERS = {
    "exposure_history": ["min_mean_10"],
    "usage_role": ["usg_wmean_10"],
    "position_role": ["position_guard_prior"],
    "assist_efficiency": ["ast_pct_wmean_10"],
    "passing_tracking": ["passes_per_min_10"],
    "opponent_make_environment": [
        "opponent_team_fgm_allowed_mean_10",
    ],
}
```

`src/features/assists/build.py`:

```python
"""Build leakage-safe pregame assists features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.assists.columns import CURRENT_ASSISTS_FEATURES

_BUILDER_TEMPS = (
    "_row",
    "_started_obs",
    "_ast_obs",
    "_minutes_obs",
)


def canonical_minutes(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "min" in frame.columns:
        return pd.to_numeric(frame["min"], errors="coerce")
    raise ValueError(
        "canonical minutes column missing: need minutes or min"
    )


def canonical_ast(frame: pd.DataFrame) -> pd.Series:
    if "ast" in frame.columns:
        return pd.to_numeric(frame["ast"], errors="coerce")
    if "assists" in frame.columns:
        return pd.to_numeric(frame["assists"], errors="coerce")
    return pd.Series(
        np.nan, index=frame.index, dtype="float64"
    )


def add_assists_features(frame: pd.DataFrame) -> pd.DataFrame:
    original_index = frame.index
    work = frame.copy()
    work["_row"] = np.arange(len(work), dtype=np.int64)
    work["_minutes_obs"] = canonical_minutes(work)
    work["_ast_obs"] = canonical_ast(work)
    work["predicted_minutes_oof"] = np.asarray(
        np.nan, dtype="float64"
    )
    work["predicted_minutes_oof"] = np.nan
    work["predicted_minutes_oof"] = work[
        "predicted_minutes_oof"
    ].astype("float64")
    work = work.sort_values("_row")
    drop = [name for name in _BUILDER_TEMPS if name in work.columns]
    work = work.drop(columns=drop)
    work.index = original_index
    return work
```

`src/features/assists/__init__.py`:

```python
"""Causal assists features for pregame models."""

from .build import add_assists_features
from .columns import (
    ASSISTS_FEATURE_CHALLENGERS,
    CURRENT_ASSISTS_FEATURES,
)

__all__ = [
    "ASSISTS_FEATURE_CHALLENGERS",
    "CURRENT_ASSISTS_FEATURES",
    "add_assists_features",
]
```

Do not loop `CURRENT_ASSISTS_FEATURES` to fill NaNs yet — later tests must fail until those columns are computed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::ContractTests -v`

Expected: PASS

- [ ] **Step 5: Commit**

Skip unless the user asked. If asked:

```bash
git add src/features/assists tests/features/test_assists_features.py
git commit -m "$(cat <<'EOF'
feat: add assists feature contract and builder skeleton

EOF
)"
```

---

### Task 2: Player windows (start rate, ast means, days_rest, is_home)

**Files:**
- Create: `src/features/assists/player.py`
- Modify: `src/features/assists/build.py`
- Test: `tests/features/test_assists_features.py`

**Interfaces:**
- Consumes: `canonical_minutes` / `canonical_ast` via `work["_minutes_obs"]` and `work["_ast_obs"]`; `_row` already assigned
- Produces: `add_player_features(work: pd.DataFrame) -> pd.DataFrame` writing `start_rate_10`, `ast_mean_10`, `ast_mean_20`, `days_rest`, `is_home`

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_assists_features.py`:

```python
class PlayerWindowTests(unittest.TestCase):
    def test_days_rest_opener_and_blank_minutes_candidate(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=5, game_id=1),
                _row(1, "2024-01-04", 24, ast=6, game_id=2),
                _row(1, "2024-01-07", np.nan, ast=np.nan, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertTrue(pd.isna(featured.iloc[0]["days_rest"]))
        self.assertEqual(featured.iloc[1]["days_rest"], 3)
        self.assertEqual(featured.iloc[2]["days_rest"], 3)

    def test_days_rest_resets_by_season(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-04-10", 20, ast=5, game_id=1,
                    season_year="2023-24",
                ),
                _row(
                    1, "2024-10-20", 20, ast=5, game_id=2,
                    season_year="2024-25",
                ),
            ]
        )
        featured = add_assists_features(frame)
        self.assertTrue(pd.isna(featured.iloc[1]["days_rest"]))

    def test_ast_mean_ignores_current_and_nan_ast(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=4, game_id=1),
                _row(1, "2024-01-03", 20, ast=np.nan, game_id=2),
                _row(1, "2024-01-05", 20, ast=10, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertEqual(featured.iloc[2]["ast_mean_10"], 4.0)
        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "ast"] = 99
        mutated_featured = add_assists_features(mutated)
        self.assertEqual(
            mutated_featured.iloc[2]["ast_mean_10"],
            featured.iloc[2]["ast_mean_10"],
        )

    def test_start_rate_missing_start_is_non_start_slot(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    start_position="G",
                ),
                _row(
                    1, "2024-01-03", 20, ast=4, game_id=2,
                    start_position="",
                ),
                _row(1, "2024-01-05", 20, ast=4, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertEqual(featured.iloc[2]["start_rate_10"], 0.5)

    def test_is_home_from_matchup(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    matchup="DET vs. CLE",
                ),
                _row(
                    1, "2024-01-03", 20, ast=4, game_id=2,
                    matchup="DET @ CLE",
                ),
            ]
        )
        featured = add_assists_features(frame)
        self.assertEqual(featured.iloc[0]["is_home"], 1.0)
        self.assertEqual(featured.iloc[1]["is_home"], 0.0)
        del featured
        missing = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=4, game_id=1)]
        ).drop(columns=["matchup"])
        featured = add_assists_features(missing)
        self.assertTrue(pd.isna(featured.iloc[0]["is_home"]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::PlayerWindowTests -v`

Expected: FAIL with `KeyError: 'days_rest'` (skeleton does not write player columns)

- [ ] **Step 3: Write minimal implementation**

`src/features/assists/player.py`:

```python
"""Player trailing assists, start rate, rest, and home flag."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def add_player_features(work: pd.DataFrame) -> pd.DataFrame:
    result = work.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"], errors="coerce"
    )
    result["start_rate_10"] = np.nan
    result["ast_mean_10"] = np.nan
    result["ast_mean_20"] = np.nan
    result["days_rest"] = np.nan
    result["is_home"] = _is_home(result)

    ordered = result.sort_values(
        ["player_id", "game_date", "game_id", "_row"]
    )
    start_by_row = {}
    ast10_by_row = {}
    ast20_by_row = {}
    rest_by_row = {}

    for _, group in ordered.groupby("player_id", sort=False):
        starts: deque[float] = deque(maxlen=10)
        asts10: deque[float] = deque(maxlen=10)
        asts20: deque[float] = deque(maxlen=20)
        last_appearance_by_season: dict[object, pd.Timestamp] = {}
        for rec in group.to_dict("records"):
            row_id = rec["_row"]
            minutes = rec["_minutes_obs"]
            ast = rec["_ast_obs"]
            appeared = np.isfinite(minutes) and minutes > 0
            if len(starts) == 0:
                start_by_row[row_id] = np.nan
            else:
                start_by_row[row_id] = float(np.mean(starts))
            if len(asts10) == 0:
                ast10_by_row[row_id] = np.nan
            else:
                ast10_by_row[row_id] = float(np.mean(asts10))
            if len(asts20) == 0:
                ast20_by_row[row_id] = np.nan
            else:
                ast20_by_row[row_id] = float(np.mean(asts20))

            season = rec["season_year"]
            prev = last_appearance_by_season.get(season)
            if prev is None:
                rest_by_row[row_id] = np.nan
            else:
                rest_by_row[row_id] = float(
                    (rec["game_date"] - prev).days
                )

            if appeared:
                starts.append(_started(rec.get("start_position")))
                if np.isfinite(ast):
                    asts10.append(float(ast))
                    asts20.append(float(ast))
                last_appearance_by_season[season] = rec["game_date"]

    result["start_rate_10"] = result["_row"].map(start_by_row)
    result["ast_mean_10"] = result["_row"].map(ast10_by_row)
    result["ast_mean_20"] = result["_row"].map(ast20_by_row)
    result["days_rest"] = result["_row"].map(rest_by_row)
    return result


def _started(value: object) -> float:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
    if text in {"", "nan", "<NA>", "None"}:
        return 0.0
    return 1.0


def _is_home(frame: pd.DataFrame) -> pd.Series:
    if "matchup" not in frame.columns:
        return pd.Series(
            np.nan, index=frame.index, dtype="float64"
        )
    return (
        frame["matchup"]
        .astype("string")
        .str.contains(r"\bvs\.", regex=True)
        .astype(float)
    )
```

In `src/features/assists/build.py`, after setting `_ast_obs`, call player features before sorting back:

```python
from src.features.assists.player import add_player_features
```

Inside `add_assists_features`, after `_ast_obs` assignment:

```python
    work = add_player_features(work)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::ContractTests tests/features/test_assists_features.py::PlayerWindowTests -v`

Expected: PASS

- [ ] **Step 5: Commit**

Skip unless the user asked.

---

### Task 3: Paired `assists_per_min_10`

**Files:**
- Modify: `src/features/assists/player.py`
- Test: `tests/features/test_assists_features.py`

**Interfaces:**
- Consumes: `add_player_features` from Task 2
- Produces: `assists_per_min_10` on the same frame; qualifying prior = minutes finite and `> 0` **and** finite `ast`; both sums use those same rows

- [ ] **Step 1: Write the failing test**

```python
class PairedRateTests(unittest.TestCase):
    def test_clean_history_is_sum_ast_over_sum_minutes(self) -> None:
        rows = [
            _row(1, f"2024-01-{day:02d}", 10.0 + i, ast=float(i + 1), game_id=i)
            for i, day in enumerate(range(1, 12), start=1)
        ]
        frame = pd.DataFrame(rows)
        featured = add_assists_features(frame)
        last = featured.iloc[-1]
        prior = frame.iloc[:-1]
        expected = prior["ast"].sum() / prior["minutes"].sum()
        self.assertAlmostEqual(
            last["assists_per_min_10"], expected
        )

    def test_nan_ast_hole_is_excluded_from_both(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 10, ast=5, game_id=1),
                _row(1, "2024-01-03", 20, ast=np.nan, game_id=2),
                _row(1, "2024-01-05", 30, ast=9, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertAlmostEqual(
            featured.iloc[2]["assists_per_min_10"],
            5.0 / 10.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::PairedRateTests -v`

Expected: FAIL with `KeyError: 'assists_per_min_10'`

- [ ] **Step 3: Write minimal implementation**

Replace `add_player_features` in `src/features/assists/player.py` with this complete function (same helpers `_started` / `_is_home`). Add a `pairs` deque of `(ast, minutes)` with `maxlen=10`. Do not `prior_sum` ast and minutes separately.

```python
def add_player_features(work: pd.DataFrame) -> pd.DataFrame:
    result = work.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"], errors="coerce"
    )
    result["start_rate_10"] = np.nan
    result["ast_mean_10"] = np.nan
    result["ast_mean_20"] = np.nan
    result["assists_per_min_10"] = np.nan
    result["days_rest"] = np.nan
    result["is_home"] = _is_home(result)

    ordered = result.sort_values(
        ["player_id", "game_date", "game_id", "_row"]
    )
    start_by_row: dict = {}
    ast10_by_row: dict = {}
    ast20_by_row: dict = {}
    rate_by_row: dict = {}
    rest_by_row: dict = {}

    for _, group in ordered.groupby("player_id", sort=False):
        starts: deque[float] = deque(maxlen=10)
        asts10: deque[float] = deque(maxlen=10)
        asts20: deque[float] = deque(maxlen=20)
        pairs: deque[tuple[float, float]] = deque(maxlen=10)
        last_appearance_by_season: dict[object, pd.Timestamp] = {}
        for rec in group.to_dict("records"):
            row_id = rec["_row"]
            minutes = rec["_minutes_obs"]
            ast = rec["_ast_obs"]
            appeared = np.isfinite(minutes) and minutes > 0
            start_by_row[row_id] = (
                np.nan if len(starts) == 0 else float(np.mean(starts))
            )
            ast10_by_row[row_id] = (
                np.nan if len(asts10) == 0 else float(np.mean(asts10))
            )
            ast20_by_row[row_id] = (
                np.nan if len(asts20) == 0 else float(np.mean(asts20))
            )
            if len(pairs) == 0:
                rate_by_row[row_id] = np.nan
            else:
                ast_sum = sum(item[0] for item in pairs)
                min_sum = sum(item[1] for item in pairs)
                rate_by_row[row_id] = (
                    np.nan if min_sum == 0 else ast_sum / min_sum
                )
            season = rec["season_year"]
            prev = last_appearance_by_season.get(season)
            rest_by_row[row_id] = (
                np.nan
                if prev is None
                else float((rec["game_date"] - prev).days)
            )
            if appeared:
                starts.append(_started(rec.get("start_position")))
                if np.isfinite(ast):
                    asts10.append(float(ast))
                    asts20.append(float(ast))
                    pairs.append((float(ast), float(minutes)))
                last_appearance_by_season[season] = rec["game_date"]

    result["start_rate_10"] = result["_row"].map(start_by_row)
    result["ast_mean_10"] = result["_row"].map(ast10_by_row)
    result["ast_mean_20"] = result["_row"].map(ast20_by_row)
    result["assists_per_min_10"] = result["_row"].map(rate_by_row)
    result["days_rest"] = result["_row"].map(rest_by_row)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::PairedRateTests tests/features/test_assists_features.py::PlayerWindowTests -v`

Expected: PASS

- [ ] **Step 5: Commit**

Skip unless the user asked.

---

### Task 4: Team-game snapshots and asof joins

**Files:**
- Create: `src/features/assists/team.py`
- Modify: `src/features/assists/build.py`
- Test: `tests/features/test_assists_features.py`

**Interfaces:**
- Consumes: `work` with `_row`, `team_id`, `opp_team_id`, `game_id`, `game_date`, `player_id`, `team_ast`, `team_fgm`, `team_pace`, `opp_ast`
- Produces: `add_team_features(work: pd.DataFrame) -> pd.DataFrame` writing `team_ast_mean_10`, `team_fgm_mean_10`, `team_pace_mean_10`, `opponent_team_ast_allowed_mean_10`, `opponent_team_pace_mean_10`

- [ ] **Step 1: Write the failing tests**

```python
class TeamSnapshotTests(unittest.TestCase):
    def test_null_aware_reduction_skips_nan_teammate(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    **_row(1, "2024-01-01", 20, ast=4, game_id=1),
                    "team_ast": np.nan,
                },
                {
                    **_row(2, "2024-01-01", 20, ast=5, game_id=1),
                    "team_ast": 25.0,
                },
                _row(1, "2024-01-03", 20, ast=6, game_id=2),
            ]
        )
        featured = add_assists_features(frame)
        tonight = featured.loc[featured["game_id"].eq(2)].iloc[0]
        self.assertEqual(tonight["team_ast_mean_10"], 25.0)

    def test_scheduled_opponent_not_player_opp_history(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    team_id=100, opp_team_id=200,
                    team_pace=90.0, opp_ast=5.0,
                ),
                _row(
                    2, "2024-01-01", 20, ast=8, game_id=1,
                    team_id=200, opp_team_id=100,
                    team_pace=110.0, opp_ast=20.0,
                ),
                _row(
                    1, "2024-01-03", 20, ast=4, game_id=2,
                    team_id=100, opp_team_id=200,
                    team_pace=91.0, opp_ast=6.0,
                ),
                _row(
                    2, "2024-01-03", 20, ast=8, game_id=2,
                    team_id=200, opp_team_id=100,
                    team_pace=111.0, opp_ast=100.0,
                ),
                _row(
                    1, "2024-01-05", 20, ast=4, game_id=3,
                    team_id=100, opp_team_id=200,
                    team_pace=92.0, opp_ast=7.0,
                ),
                _row(
                    2, "2024-01-05", 20, ast=8, game_id=3,
                    team_id=200, opp_team_id=100,
                    team_pace=112.0, opp_ast=8.0,
                ),
            ]
        )
        featured = add_assists_features(frame)
        tonight = featured.loc[
            featured["game_id"].eq(3) & featured["player_id"].eq(1)
        ].iloc[0]
        self.assertEqual(
            tonight["opponent_team_ast_allowed_mean_10"],
            60.0,
        )
        self.assertNotEqual(
            tonight["team_pace_mean_10"],
            tonight["opponent_team_pace_mean_10"],
        )
        double_shift = 20.0
        self.assertNotEqual(
            tonight["opponent_team_ast_allowed_mean_10"],
            double_shift,
        )
        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "opp_ast"] = 999
        mutated.loc[mutated["game_id"].eq(3), "team_pace"] = 999
        mutated_featured = add_assists_features(mutated)
        mutated_tonight = mutated_featured.loc[
            mutated_featured["game_id"].eq(3)
            & mutated_featured["player_id"].eq(1)
        ].iloc[0]
        self.assertEqual(
            mutated_tonight["opponent_team_ast_allowed_mean_10"],
            tonight["opponent_team_ast_allowed_mean_10"],
        )
        self.assertEqual(
            mutated_tonight["team_pace_mean_10"],
            tonight["team_pace_mean_10"],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::TeamSnapshotTests -v`

Expected: FAIL with `KeyError: 'team_ast_mean_10'`

- [ ] **Step 3: Write minimal implementation**

`src/features/assists/team.py`:

```python
"""Null-aware team-game snapshots and scheduled-opponent asof joins."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


_STAT_COLUMNS = (
    "team_ast",
    "team_fgm",
    "team_pace",
    "opp_ast",
)


def add_team_features(work: pd.DataFrame) -> pd.DataFrame:
    result = work.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"], errors="coerce"
    )
    snapshots = _team_snapshots(result)
    left = result[
        ["_row", "game_date", "team_id", "opp_team_id"]
    ].copy()

    own_cols = [
        "game_date",
        "team_id",
        "team_ast_mean_10",
        "team_fgm_mean_10",
        "team_pace_mean_10",
    ]
    own = pd.merge_asof(
        left.sort_values("game_date"),
        snapshots[own_cols].sort_values("game_date"),
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
        left_by="team_id",
        right_by="team_id",
    )
    opp_right = snapshots[
        [
            "game_date",
            "team_id",
            "opp_ast_mean_10",
            "team_pace_mean_10",
        ]
    ].rename(
        columns={
            "opp_ast_mean_10": "opponent_team_ast_allowed_mean_10",
            "team_pace_mean_10": "opponent_team_pace_mean_10",
        }
    )
    opp = pd.merge_asof(
        left.sort_values("game_date"),
        opp_right.sort_values("game_date"),
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
        left_by="opp_team_id",
        right_by="team_id",
    )
    for column in (
        "team_ast_mean_10",
        "team_fgm_mean_10",
        "team_pace_mean_10",
    ):
        result[column] = result["_row"].map(
            own.set_index("_row")[column]
        )
    for column in (
        "opponent_team_ast_allowed_mean_10",
        "opponent_team_pace_mean_10",
    ):
        result[column] = result["_row"].map(
            opp.set_index("_row")[column]
        )
    return result


def _team_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    games = _reduce_team_games(frame)
    games = games.sort_values(
        ["team_id", "game_date", "game_id"]
    )
    for source, output in (
        ("team_ast", "team_ast_mean_10"),
        ("team_fgm", "team_fgm_mean_10"),
        ("team_pace", "team_pace_mean_10"),
        ("opp_ast", "opp_ast_mean_10"),
    ):
        games[output] = np.nan
        for _, group in games.groupby("team_id", sort=False):
            window: deque[float] = deque(maxlen=10)
            for idx, value in group[source].items():
                if np.isfinite(value):
                    window.append(float(value))
                games.at[idx, output] = (
                    np.nan if len(window) == 0 else float(np.mean(window))
                )
    return games


def _reduce_team_games(frame: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "team_id",
        "game_id",
        "game_date",
        "opp_team_id",
        "player_id",
        *_STAT_COLUMNS,
    ]
    present = [name for name in needed if name in frame.columns]
    tmp = frame[present].copy()
    if "player_id" not in tmp.columns:
        tmp["player_id"] = np.nan
    tmp = tmp.sort_values(
        ["team_id", "game_id", "player_id"]
    )

    def first_nonnull(series: pd.Series):
        valid = series.dropna()
        return valid.iloc[0] if len(valid) else np.nan

    def first_finite(series: pd.Series):
        values = pd.to_numeric(series, errors="coerce")
        finite = values[np.isfinite(values)]
        return finite.iloc[0] if len(finite) else np.nan

    aggregations: dict[str, tuple] = {
        "game_date": ("game_date", first_nonnull),
        "opp_team_id": ("opp_team_id", first_nonnull),
    }
    for column in _STAT_COLUMNS:
        if column in tmp.columns:
            aggregations[column] = (column, first_finite)
    aggregated = tmp.groupby(
        ["team_id", "game_id"], sort=False
    ).agg(**aggregations)
    for column in _STAT_COLUMNS:
        if column not in aggregated.columns:
            aggregated[column] = np.nan
    return aggregated.reset_index()

Wire into `add_assists_features` after `add_player_features`:

```python
from src.features.assists.team import add_team_features
...
    work = add_player_features(work)
    work = add_team_features(work)
```

If `set_index("_row")` fails because asof dropped `_row`, keep `_row` on `left` (it is already there) so both `own` and `opp` retain it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::TeamSnapshotTests tests/features/test_assists_features.py::PlayerWindowTests -v`

Expected: PASS. `opponent_team_ast_allowed_mean_10` for tonight is `(20 + 100) / 2 = 60`, not `20` (double-shift) and not player 1’s `opp_ast` mean `(5+6)/2`.

- [ ] **Step 5: Commit**

Skip unless the user asked.

---

### Task 5: Leakage aliases, overwrite, full contract columns

**Files:**
- Modify: `src/features/assists/build.py` only if a contract column is still missing
- Test: `tests/features/test_assists_features.py`

**Interfaces:**
- Consumes: `add_assists_features` from Tasks 1–4
- Produces: tests proving Game N mutations and alias blanks do not change the 12 features; overwrite of an incoming `assists_per_min_10`

- [ ] **Step 1: Write the failing tests**

```python
_FEATURE_COLUMNS = list(CURRENT_ASSISTS_FEATURES)


class LeakageTests(unittest.TestCase):
    def test_mutating_game_n_does_not_change_features(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=4, game_id=1),
                _row(1, "2024-01-03", 24, ast=6, game_id=2),
            ]
        )
        featured = add_assists_features(frame)
        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(2), "ast"] = 99
        mutated.loc[mutated["game_id"].eq(2), "minutes"] = 99
        mutated.loc[mutated["game_id"].eq(2), "team_ast"] = 99
        mutated.loc[mutated["game_id"].eq(2), "opp_ast"] = 99
        mutated.loc[mutated["game_id"].eq(2), "start_position"] = "C"
        mutated_featured = add_assists_features(mutated)
        left = featured.loc[featured["game_id"].eq(2)].iloc[0]
        right = mutated_featured.loc[
            mutated_featured["game_id"].eq(2)
        ].iloc[0]
        for column in _FEATURE_COLUMNS:
            self._assert_same(left[column], right[column], column)

    def test_blanking_both_aliases_does_not_change_features(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=4, game_id=1),
                _row(1, "2024-01-03", 24, ast=6, game_id=2),
            ]
        )
        featured = add_assists_features(frame)
        blanked = frame.copy()
        for column in ("ast", "assists", "min", "minutes"):
            blanked.loc[blanked["game_id"].eq(2), column] = np.nan
        blanked_featured = add_assists_features(blanked)
        left = featured.loc[featured["game_id"].eq(2)].iloc[0]
        right = blanked_featured.loc[
            blanked_featured["game_id"].eq(2)
        ].iloc[0]
        for column in _FEATURE_COLUMNS:
            self._assert_same(left[column], right[column], column)

    def test_overwrites_existing_assists_per_min_10(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=4, game_id=1)]
        )
        frame["assists_per_min_10"] = 999.0
        featured = add_assists_features(frame)
        self.assertTrue(
            pd.isna(featured.iloc[0]["assists_per_min_10"])
        )

    def _assert_same(self, left, right, column: str) -> None:
        if pd.isna(left) and pd.isna(right):
            return
        self.assertEqual(left, right, msg=column)
```

If Task 4 already makes mutation tests pass, this step still fails until `assists_per_min_10` overwrite is guaranteed (it should already be, because the builder writes the column). Run anyway — TDD requires seeing the new tests execute.

- [ ] **Step 2: Run tests**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py::LeakageTests -v`

Expected: PASS if Tasks 2–4 are correct. If `test_overwrites_existing_assists_per_min_10` fails because the incoming column is left untouched, assign every name in `CURRENT_ASSISTS_FEATURES` after helpers return (already true for `assists_per_min_10` via `add_player_features`).

- [ ] **Step 3: Fix only if a test failed**

If overwrite failed, at the end of `add_assists_features` before dropping temps:

```python
    for column in CURRENT_ASSISTS_FEATURES:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = work[column]
```

Do not skip writing `assists_per_min_10` when it already exists.

- [ ] **Step 4: Re-run the full assists feature file**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

Skip unless the user asked.

---

### Task 6: Pregame as-of builder

**Files:**
- Create: `src/features/assists/pregame.py`
- Modify: `src/features/assists/__init__.py`
- Test: `tests/features/test_pregame_assists.py`

**Interfaces:**
- Consumes: `add_assists_features(frame: pd.DataFrame) -> pd.DataFrame`
- Produces: `OUTCOME_COLUMNS: tuple[str, ...]`, `appearances_before(frame, as_of) -> pd.DataFrame`, `blank_outcomes(row: pd.Series) -> pd.Series`, `build_pregame_assists_features(history, candidates, *, as_of) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_pregame_assists.py`:

```python
"""Pregame assists features stay frozen at the quote as-of date."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.assists.pregame import (
    appearances_before,
    build_pregame_assists_features,
)


def _row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    ast: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    team_pace: float = 90.0,
    opp_ast: float = 20.0,
) -> dict:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": "2024-25",
        "matchup": "DET vs. CLE",
        "minutes": minutes,
        "min": minutes,
        "min_sec": "20:00",
        "start_position": "G",
        "ast": ast,
        "assists": ast,
        "team_ast": 20.0,
        "team_fgm": 40.0,
        "team_pace": team_pace,
        "opp_ast": opp_ast,
        "opp_pace": team_pace,
        "pass": 10.0,
        "tchs": 20.0,
        "sast": 1.0,
        "ftast": 1.0,
    }


class AsOfCutoffTests(unittest.TestCase):
    def test_drops_games_on_or_after_as_of(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2026-05-08", 30, ast=5, game_id=1),
                _row(1, "2026-05-09", 40, ast=99, game_id=2),
            ]
        )
        prior = appearances_before(frame, "2026-05-09")
        self.assertEqual(len(prior), 1)
        self.assertEqual(int(prior.iloc[0]["game_id"]), 1)

    def test_candidate_ignores_same_day_box_and_keeps_index(self) -> None:
        history = pd.DataFrame(
            [
                _row(1, "2026-04-01", 32, ast=8, game_id=1),
                _row(1, "2026-05-09", 40, ast=99, game_id=2),
            ]
        )
        candidate = pd.DataFrame(
            [_row(1, "2026-05-09", 0, ast=0, game_id=99)]
        )
        candidate.index = pd.Index([42])
        featured = build_pregame_assists_features(
            history, candidate, as_of="2026-05-09"
        )
        self.assertEqual(list(featured.index), [42])
        self.assertEqual(featured.iloc[0]["ast_mean_10"], 8.0)
        self.assertTrue(
            pd.isna(featured.iloc[0]["predicted_minutes_oof"])
        )

    def test_later_candidate_does_not_see_earlier_candidate(self) -> None:
        history = pd.DataFrame(
            [_row(1, "2026-04-01", 32, ast=8, game_id=1)]
        )
        candidates = pd.DataFrame(
            [
                _row(1, "2026-05-09", 40, ast=50, game_id=9),
                _row(1, "2026-05-11", 40, ast=50, game_id=11),
            ]
        )
        featured = build_pregame_assists_features(
            history, candidates, as_of="2026-05-09"
        )
        later = featured.loc[featured["game_id"].eq(11)].iloc[0]
        self.assertEqual(later["ast_mean_10"], 8.0)

    def test_one_sided_slate_still_gets_opponent_history(self) -> None:
        history = pd.DataFrame(
            [
                _row(
                    1, "2026-04-01", 20, ast=4, game_id=1,
                    team_id=100, opp_team_id=200,
                    team_pace=90.0, opp_ast=5.0,
                ),
                _row(
                    2, "2026-04-01", 20, ast=8, game_id=1,
                    team_id=200, opp_team_id=100,
                    team_pace=110.0, opp_ast=40.0,
                ),
            ]
        )
        candidate = pd.DataFrame(
            [
                _row(
                    1, "2026-04-03", np.nan, ast=np.nan, game_id=9,
                    team_id=100, opp_team_id=200,
                    team_pace=np.nan, opp_ast=np.nan,
                )
            ]
        )
        featured = build_pregame_assists_features(
            history, candidate, as_of="2026-04-03"
        )
        self.assertEqual(
            featured.iloc[0]["opponent_team_ast_allowed_mean_10"],
            40.0,
        )
        self.assertNotEqual(
            featured.iloc[0]["team_pace_mean_10"],
            featured.iloc[0]["opponent_team_pace_mean_10"],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_pregame_assists.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `src.features.assists.pregame`

- [ ] **Step 3: Write minimal implementation**

`src/features/assists/pregame.py`:

```python
"""Pregame assists rows from history strictly before an as-of date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.assists.build import add_assists_features
from src.features.assists.columns import CURRENT_ASSISTS_FEATURES

OUTCOME_COLUMNS = (
    "ast",
    "assists",
    "min",
    "minutes",
    "min_sec",
    "start_position",
    "team_ast",
    "team_fgm",
    "team_pace",
    "opp_ast",
    "opp_pace",
    "pass",
    "tchs",
    "sast",
    "ftast",
)


def appearances_before(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.loc[dates < cutoff].copy()


def blank_outcomes(row: pd.Series) -> pd.Series:
    result = row.copy()
    for column in OUTCOME_COLUMNS:
        if column not in result.index:
            continue
        if column == "start_position":
            result[column] = ""
        else:
            result[column] = np.nan
    return result


def build_pregame_assists_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    prior = appearances_before(history, as_of)
    if candidates.empty:
        return candidates.copy()

    dated = candidates.copy()
    dated["game_date"] = pd.to_datetime(
        dated["game_date"], errors="coerce"
    )
    blanked = dated.apply(blank_outcomes, axis=1)
    parts: list[pd.DataFrame] = []
    for _, group in blanked.groupby("game_date", sort=True):
        panel = pd.concat([prior, group], ignore_index=True)
        featured = add_assists_features(panel)
        keys = group[["game_id", "player_id"]]
        matched = featured.merge(
            keys,
            on=["game_id", "player_id"],
            how="inner",
        )
        parts.append(matched)
    stacked = pd.concat(parts, ignore_index=True)
    keep = [
        column
        for column in stacked.columns
        if column in set(candidates.columns) | set(CURRENT_ASSISTS_FEATURES)
    ]
    restored = candidates[["game_id", "player_id"]].merge(
        stacked[keep],
        on=["game_id", "player_id"],
        how="left",
    )
    restored.index = candidates.index
    return restored
```

Do not blank `matchup`, `game_date`, `season_year`, `team_id`, `opp_team_id`, `player_id`, or `game_id`.

Export `build_pregame_assists_features` from `src/features/assists/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/features/test_assists_features.py tests/features/test_pregame_assists.py tests/features/test_pregame_points.py tests/models/test_points_features.py -q`

Expected: PASS (assists files green; existing points tests still pass)

- [ ] **Step 5: Commit**

Skip unless the user asked.

---

## Spec coverage

| Spec requirement | Task |
|---|---|
| `CURRENT_ASSISTS_FEATURES` order + challenger names | 1 |
| `predicted_minutes_oof` float64 NaN | 1 |
| Missing minutes/`min` raises | 1 |
| Positional `_row` align-back, duplicate index | 1 |
| Appearance vs output panel; `days_rest` opener / season reset / blank minutes | 2 |
| `start_rate_10` among appearances; missing start = 0 slot | 2 |
| `ast_mean_10/20` drop NaN ast | 2 |
| `is_home` `\bvs\.`; missing column NaN | 2 |
| Paired `assists_per_min_10` + NaN-ast hole | 3 |
| Null-aware team reduction | 4 |
| Opponent `merge_asof` left_by/right_by; pace two keys; includes-self / no double-shift | 4 |
| Shift-before-lookup + blank both aliases + overwrite | 5 |
| Pregame `ignore_index`, `(game_id, player_id)` restore, per-date loop, one-sided slate | 6 |
| Blank list includes `sast`/`ftast`; do not blank matchup/ids | 6 |

Out of scope (no task): Poisson booster, PIT, coupling, pricing, challenger columns, silver/`2025-26`.
