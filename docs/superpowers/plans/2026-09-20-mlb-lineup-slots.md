# Posted Lineup Slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist cutoff-safe starting nines as a per-slot K/PA vector plus Stats API play-by-play batter history, without wiring `nb_k_v1`.

**Architecture:** Gzipped live-feed snapshots parse into `batter_pas` (facts) and `lineup_slots` (as-of cards). Hierarchical leave-one-split-out shrinkage freezes 60/365/prior-2 vs-hand and overall rates onto nine long rows. `observed_before_cutoff` and `rate_version` keep leak bits and frozen math from collapsing into one column.

**Tech Stack:** Python 3.12, pandas, pyarrow, pytest, argparse, gzip, Stats API (`v1.1/game/{pk}/feed/live`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-20-mlb-lineup-slots-design.md` (review 2). Binding names and dtypes live there.
- Do not change `STRIKEOUT_FEATURE_COLUMNS`, `fit_strikeouts`, `nb_k_v1`, `pregame_from_starts` (`team_fallback` / `"[]"`), or `gamelog_features.missing_lineup = 1`.
- Do not implement Log5, BF mixture, or Poisson-binomial.
- Network tests use inline JSON fixtures; never hit statsapi from pytest.
- Cutoff joins: `lineup_slots.ingested_at_utc < cutoff`; `batter_pas.event_time_utc < cutoff` only (not `ingested_at_utc`). Rate windows drop `event_time_imputed == 1`.
- `boxscore_00` stamp is `original_scheduled_start - 24h`, not `start - forecast_horizon_hours`.
- Python 3.12; imports `from src.mlb...`; datetimes `datetime64[us, UTC]`.
- Commits use a HEREDOC message; do not skip hooks.

## File map

| File | Responsibility |
|---|---|
| `src/mlb/schemas.py` | `BATTER_PA_COLUMNS`, `LINEUP_SLOT_COLUMNS`, `TABLE_SCHEMAS` |
| `src/mlb/config.py`, `config/mlb.yaml` | `batter_hand_prior_strength: 400` |
| `src/mlb/models/batter_rates.py` | `is_strikeout`, `rate_version`, `LeagueKPa`, platoon OR, shrink, league eligibility |
| `src/mlb/pipeline/pbp.py` | Parse/ingest PBP, gzip cache, `pa_id` dedupe |
| `src/mlb/pipeline/lineup_slots.py` | 00-filter, freeze, coverage, as-of select, mismatch rate |
| `src/mlb/pipeline/parse.py` | `parse_lineups` delegates Stats API payloads to 00-filter |
| `src/mlb/cli.py` | `ingest-play-by-play`, `ingest-lineup-slots`; `snapshot-lineups` writes `lineup_slots` |
| `docs/mlb/data_dictionary.yaml`, `docs/mlb/CONTRACT.md` | Tables, Stanek, Ohtani, as-of notes |
| Tests under `tests/mlb/pipeline/` and `tests/mlb/models/` | Spec tests 1–19 |

---

### Task 1: Schemas and hand-prior config

**Files:**
- Modify: `src/mlb/schemas.py` (after `ID_MAP_COLUMNS`)
- Modify: `src/mlb/config.py` (`MlbConfig` + `load_config`)
- Modify: `config/mlb.yaml`
- Test: `tests/mlb/pipeline/test_lineup_schemas.py`

**Interfaces:**
- Consumes: existing `UTC_DTYPE`, `TABLE_SCHEMAS`
- Produces: `BATTER_PA_COLUMNS`, `LINEUP_SLOT_COLUMNS` with spec dtypes. `MlbConfig.batter_hand_prior_strength: float = 400.0`

- [ ] **Step 1: Write the failing test**

```python
from src.mlb.config import load_config
from src.mlb.schemas import BATTER_PA_COLUMNS, LINEUP_SLOT_COLUMNS, TABLE_SCHEMAS


def test_batter_pas_has_no_is_strikeout_column() -> None:
    assert "is_strikeout" not in BATTER_PA_COLUMNS
    assert "event_type" in BATTER_PA_COLUMNS
    assert "event_time_imputed" in BATTER_PA_COLUMNS
    assert "is_pitcher_in_game" in BATTER_PA_COLUMNS


def test_lineup_slots_stores_three_windows_and_overall() -> None:
    for window in ("60", "365", "prior2"):
        assert f"k_pa_vs_hand_shrunk_{window}" in LINEUP_SLOT_COLUMNS
        assert f"k_pa_overall_shrunk_{window}" in LINEUP_SLOT_COLUMNS
        assert f"pa_vs_hand_{window}" in LINEUP_SLOT_COLUMNS
        assert f"pa_all_{window}" in LINEUP_SLOT_COLUMNS
    assert "n_eff_hand" not in LINEUP_SLOT_COLUMNS
    assert "observed_before_cutoff" in LINEUP_SLOT_COLUMNS
    assert "opposing_pitcher_hand" in LINEUP_SLOT_COLUMNS
    assert TABLE_SCHEMAS["batter_pas"] is BATTER_PA_COLUMNS
    assert TABLE_SCHEMAS["lineup_slots"] is LINEUP_SLOT_COLUMNS


def test_config_loads_hand_prior() -> None:
    assert load_config().batter_hand_prior_strength == 400.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_schemas.py -v`

Expected: FAIL with `ImportError` or `AttributeError` (`BATTER_PA_COLUMNS` / `batter_hand_prior_strength`).

- [ ] **Step 3: Write minimal implementation**

Add to `schemas.py` (use `UTC_DTYPE` already defined):

```python
BATTER_PA_COLUMNS: dict[str, str] = {
    "pa_id": "string",
    "game_pk": "int64",
    "at_bat_index": "int64",
    "batter_id": "int64",
    "pitcher_id": "int64",
    "pitcher_hand": "string",
    "batter_bats": "string",
    "batter_stand": "string",
    "event_type": "string",
    "event_time_utc": UTC_DTYPE,
    "event_time_imputed": "int64",
    "is_pitcher_in_game": "int64",
    "ingested_at_utc": UTC_DTYPE,
    "snapshot_id": "string",
}

LINEUP_SLOT_COLUMNS: dict[str, str] = {
    "game_pk": "int64",
    "team_id": "int64",
    "side": "string",
    "slot": "int64",
    "batter_id": "int64",
    "slot_is_pitcher": "int64",
    "k_pa_vs_hand_shrunk_60": "float64",
    "k_pa_vs_hand_shrunk_365": "float64",
    "k_pa_vs_hand_shrunk_prior2": "float64",
    "k_pa_overall_shrunk_60": "float64",
    "k_pa_overall_shrunk_365": "float64",
    "k_pa_overall_shrunk_prior2": "float64",
    "pa_vs_hand_60": "float64",
    "pa_vs_hand_365": "float64",
    "pa_vs_hand_prior2": "float64",
    "pa_all_60": "float64",
    "pa_all_365": "float64",
    "pa_all_prior2": "float64",
    "opposing_pitcher_hand": "string",
    "vs_pitcher_id": "Int64",
    "lineup_state": "string",
    "observed_before_cutoff": "int64",
    "provenance": "string",
    "rate_version": "string",
    "ingested_at_utc": UTC_DTYPE,
    "snapshot_id": "string",
    "season": "int64",
}
```

Register both in `TABLE_SCHEMAS`. Add `batter_hand_prior_strength: float = 400.0` to `MlbConfig` and `load_config` (`raw.get("batter_hand_prior_strength", 400.0)`). Add `batter_hand_prior_strength: 400.0` to `config/mlb.yaml` next to `batter_k_prior_strength`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_schemas.py tests/mlb -q --tb=line -k "not backtest"`

Expected: new tests PASS; existing MLB tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlb/schemas.py src/mlb/config.py config/mlb.yaml tests/mlb/pipeline/test_lineup_schemas.py
git commit -m "$(cat <<'EOF'
feat(mlb): add batter_pas and lineup_slots schemas

EOF
)"
```

---

### Task 2: Derived strikeout flag and rate_version

**Files:**
- Create: `src/mlb/models/batter_rates.py`
- Test: `tests/mlb/models/test_batter_rates.py`

**Interfaces:**
- Consumes: `MlbConfig.batter_k_prior_strength`, `batter_hand_prior_strength`
- Produces:
  - `STRIKEOUT_EVENT_TYPES: frozenset[str]`
  - `is_strikeout(event_type: str, *, events: frozenset[str] = STRIKEOUT_EVENT_TYPES) -> int`
  - `rate_version(config: MlbConfig) -> str`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import replace
from src.mlb.config import load_config
from src.mlb.models.batter_rates import is_strikeout, rate_version


def test_is_strikeout_derived_from_event_set() -> None:
    assert is_strikeout("strikeout") == 1
    assert is_strikeout("walk") == 0
    assert is_strikeout("strikeout_double_play") == 1
    assert is_strikeout("strikeout", events=frozenset({"walk"})) == 0


def test_rate_version_changes_with_priors_not_l2() -> None:
    config = load_config()
    base = rate_version(config)
    assert base.startswith("kpa_")
    assert len(base) == 16
    bumped = replace(config, batter_hand_prior_strength=401.0)
    assert rate_version(bumped) != base
    l2 = replace(config, strikeout_l2=99.0)
    assert rate_version(l2) == base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py::test_is_strikeout_derived_from_event_set tests/mlb/models/test_batter_rates.py::test_rate_version_changes_with_priors_not_l2 -v`

Expected: FAIL `ModuleNotFoundError: batter_rates`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.mlb.config import MlbConfig

STRIKEOUT_EVENT_TYPES = frozenset({
    "strikeout",
    "strikeout_double_play",
    "strikeout_triple_play",
})
RATE_VERSION_PREFIX = "kpa_"


def is_strikeout(
    event_type: str,
    *,
    events: frozenset[str] = STRIKEOUT_EVENT_TYPES,
) -> int:
    return int(str(event_type) in events)


def rate_version(config: MlbConfig) -> str:
    payload = {
        "exclude_pitcher_batters": "pbp_pitcher_ids_minus_two_way",
        "hand_prior": float(config.batter_hand_prior_strength),
        "overall": "leave_one_split_out",
        "overall_prior": float(config.batter_k_prior_strength),
        "platoon": "odds_ratio",
        "prior_seasons": 2,
        "strikeout_events": sorted(STRIKEOUT_EVENT_TYPES),
        "windows_days": [60, 365],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return RATE_VERSION_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py::test_is_strikeout_derived_from_event_set tests/mlb/models/test_batter_rates.py::test_rate_version_changes_with_priors_not_l2 -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/models/batter_rates.py tests/mlb/models/test_batter_rates.py
git commit -m "$(cat <<'EOF'
feat(mlb): hash k/PA rate_version and derive is_strikeout at read

EOF
)"
```

---

### Task 3: Platoon odds ratio

**Files:**
- Modify: `src/mlb/models/batter_rates.py`
- Test: `tests/mlb/models/test_batter_rates.py`

**Interfaces:**
- Produces: `league_platoon_odds_ratio(*, bats: str, league_k_pa_cell: float, league_k_pa_bats: float) -> float`

- [ ] **Step 1: Write the failing test**

```python
from src.mlb.models.batter_rates import league_platoon_odds_ratio


def test_missing_bats_ratio_is_one() -> None:
    assert league_platoon_odds_ratio(bats="", league_k_pa_cell=0.28, league_k_pa_bats=0.22) == 1.0
    assert league_platoon_odds_ratio(bats="S", league_k_pa_cell=0.24, league_k_pa_bats=0.22) != 1.0


def test_switch_uses_s_cells_not_rhb_offset() -> None:
    rhb = league_platoon_odds_ratio(bats="R", league_k_pa_cell=0.26, league_k_pa_bats=0.22)
    switch = league_platoon_odds_ratio(bats="S", league_k_pa_cell=0.23, league_k_pa_bats=0.22)
    assert switch != rhb
    assert switch > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py::test_missing_bats_ratio_is_one tests/mlb/models/test_batter_rates.py::test_switch_uses_s_cells_not_rhb_offset -v`

Expected: FAIL `league_platoon_odds_ratio` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def _odds(p: float) -> float:
    clipped = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return clipped / (1.0 - clipped)


def league_platoon_odds_ratio(
    *,
    bats: str,
    league_k_pa_cell: float,
    league_k_pa_bats: float,
) -> float:
    if bats is None or str(bats).strip() in {"", "nan", "<NA>", "None"}:
        return 1.0
    return _odds(league_k_pa_cell) / _odds(league_k_pa_bats)
```

Note: `bats="S"` is a real cell, not the missing branch.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py::test_missing_bats_ratio_is_one tests/mlb/models/test_batter_rates.py::test_switch_uses_s_cells_not_rhb_offset -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/models/batter_rates.py tests/mlb/models/test_batter_rates.py
git commit -m "$(cat <<'EOF'
feat(mlb): platoon odds-ratio prior uses bats-specific cells

EOF
)"
```

---

### Task 4: Leave-one-split-out shrink_batter_k_pa

**Files:**
- Modify: `src/mlb/models/batter_rates.py`
- Test: `tests/mlb/models/test_batter_rates.py`

**Interfaces:**
- Consumes: `shrink_rate` from `src.mlb.models.shrinkage`, `is_strikeout`, `league_platoon_odds_ratio`
- Produces:
```python
@dataclass(frozen=True)
class LeagueKPa:
    overall: float
    by_bats: dict[str, float]
    by_bats_hand: dict[tuple[str, str], float]

def shrink_batter_k_pa(
    pas: pd.DataFrame,
    *,
    batter_id: int,
    opposing_pitcher_hand: str,
    bats: str,
    cutoff: pd.Timestamp,
    league: LeagueKPa,
    config: MlbConfig,
) -> dict[str, dict[str, float]]:
```
  Keys `'60' | '365' | 'prior2'`. Each value: `k_pa_vs_hand_shrunk`, `k_pa_overall_shrunk`, `pa_vs_hand`, `pa_all`.

- [ ] **Step 1: Write the failing test**

Build a 10-row frame: batter 1, 6 PA vs R (2 K), 4 PA vs L (0 K), `event_type` strikeout/out, `event_time_imputed=0`, times before cutoff. `LeagueKPa(overall=0.22, by_bats={"R": 0.22}, by_bats_hand={("R","R"): 0.26, ("R","L"): 0.18})`.

```python
def test_zero_vs_l_returns_prior_not_nan(mlb_config) -> None:
    # only vs-R rows for batter 1; ask for vs L
    out = shrink_batter_k_pa(..., opposing_pitcher_hand="L", bats="R", ...)
    assert out["365"]["pa_vs_hand"] == 0.0
    assert out["365"]["k_pa_vs_hand_shrunk"] == pytest.approx(out["365"]["k_pa_overall_shrunk"] * 0 + out["365"]["k_pa_vs_hand_shrunk"])
    assert math.isfinite(out["365"]["k_pa_vs_hand_shrunk"])


def test_overall_excludes_the_split(mlb_config) -> None:
    # vs-R K/PA = 0.5 on 4 PA, vs-L = 0 on 4 PA, league 0.22, priors huge so posterior ~ prior
    # more useful: small priors via replace(config, batter_k_prior_strength=1, batter_hand_prior_strength=1)
    # overall for vs-R must use the vs-L counts only (leave-one-split-out)
```

Implement the second test as: with `batter_k_prior_strength=1` and `batter_hand_prior_strength=1`, vs-R overall (rest) equals shrink of the vs-L successes only, not all 8 PA. Assert `k_pa_overall_shrunk` differs from shrinking `k_all/pa_all` to league.

Also: unknown `opposing_pitcher_hand=""` still fills `k_pa_overall_shrunk_*` and sets vs-hand to NaN.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py -k shrink -v`

Expected: FAIL `shrink_batter_k_pa` not defined.

- [ ] **Step 3: Write minimal implementation**

Filter `pas` to `batter_id`, `event_time_utc < cutoff`, `event_time_imputed == 0`. Windows: 60/365 days before cutoff; `prior2` = season in `{cutoff.year-1, cutoff.year-2}` (use `event_time_utc` year). Derive K via `is_strikeout(event_type)`. `pa_hand` / `k_hand` where `pitcher_hand == opposing_pitcher_hand`. `pa_rest = pa_all - pa_hand`. `overall = shrink_rate(k_rest, pa_rest, league.overall, config.batter_k_prior_strength)`. If `opposing_pitcher_hand` empty, vs-hand values `nan` and skip split shrink. Else `ratio = league_platoon_odds_ratio(bats=bats, league_k_pa_cell=league.by_bats_hand.get((bats, hand), league.overall), league_k_pa_bats=league.by_bats.get(bats, league.overall))`, convert overall through odds × ratio, then `shrink_rate(k_hand, pa_hand, prior_mean, config.batter_hand_prior_strength)`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/models/batter_rates.py tests/mlb/models/test_batter_rates.py
git commit -m "$(cat <<'EOF'
feat(mlb): hierarchical batter K/PA with leave-one-split-out overall

EOF
)"
```

---

### Task 5: parse_play_by_play

**Files:**
- Create: `src/mlb/pipeline/pbp.py`
- Test: `tests/mlb/pipeline/test_pbp.py`

**Interfaces:**
- Consumes: `BATTER_PA_COLUMNS`, `coerce_frame`
- Produces: `parse_play_by_play(raw_payload, snapshot_id, ingested_at, *, scheduled_start=None, people=None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

Inline live-feed JSON with two `allPlays`: atBatIndex 0 strikeout, atBatIndex 1 walk; matchup batter/pitcher/hands; `about.startTime` set. Assert `pa_id == "746327_0"`, no `is_strikeout` column, `event_type` stored, `is_pitcher_in_game` 1 for a batter who also pitches in the payload, `ingested_at` equals the wall-clock argument (not event time).

Second test: missing start/end, `scheduled_start` supplied → `event_time_imputed==1` and `event_time_utc == scheduled_start + 4h`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_pbp.py -v`

Expected: FAIL import.

- [ ] **Step 3: Write minimal implementation**

Walk `liveData.plays.allPlays` or top-level `allPlays`. `event_type` from `result.eventType` or `result.event`. Hands from `matchup.pitchHand.code` / `pitcher.p_throws` and `batSide.code`. After building rows, set `is_pitcher_in_game` per `game_pk` if `batter_id` is in that game’s `pitcher_id` set. Join `batter_bats` from `people` on `mlb_id` if provided, else `""`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_pbp.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/pbp.py tests/mlb/pipeline/test_pbp.py
git commit -m "$(cat <<'EOF'
feat(mlb): parse Stats API play-by-play into batter_pas

EOF
)"
```

---

### Task 6: 00-filter starting nine

**Files:**
- Create: `src/mlb/pipeline/lineup_slots.py` (`parse_starting_nine`)
- Modify: `src/mlb/pipeline/parse.py` (`parse_lineups` Stats API branch calls `parse_starting_nine` and maps `slot` → `batting_slot` for list compatibility)
- Test: `tests/mlb/pipeline/test_lineup_slots.py`

**Interfaces:**
- Produces: `parse_starting_nine(raw_payload, game_pk=None) -> pd.DataFrame` with `game_pk, team_id, side, slot, batter_id, slot_is_pitcher, season` (`season` from `gameData.game.season` when present, else 0)

- [ ] **Step 1: Write the failing test**

Fixture: home `battingOrder` array is the *final* nine (includes PH id `641658` in slot 2). Players dict has `100…900` starters including Witt `677951` at `200`, and `201` Hampson `641658`. Assert parsed slot 2 is `677951`, not `641658`. `slot_is_pitcher==0`.

Stanek fixture: away slot 9 `592773` position `P`, PH `623205` battingOrder `901`. Assert slot 9 is Stanek, `slot_is_pitcher==1`, PH absent.

Ohtani fixture: batter `660271` battingOrder `100` position `DH`, also appears as a pitcher elsewhere in the box. Assert slot 1 `slot_is_pitcher==0`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py -k "starting_nine or stanek or ohtani" -v`

Expected: FAIL import / wrong slot 2.

- [ ] **Step 3: Write minimal implementation**

For each side, iterate `teams.{side}.players`. Keep `str(battingOrder).endswith("0")` and `battingOrder` not empty. Slot = `int(str(battingOrder)[0])` for 3-digit codes, else `int(battingOrder)` if 1–9. If two 00-codes share a slot, keep the one whose position is not `P` when the other is `P` (two-way); otherwise first seen. Never read `teams.*.battingOrder` for identity.

`parse_lineups`: if payload is a list, keep today’s `LINEUP_COLUMNS` behavior (fixture `lineups.json`). Else return `parse_starting_nine` renamed to `batting_slot`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py tests/mlb/pipeline/test_parse.py tests/mlb/pipeline/test_ingest.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/lineup_slots.py src/mlb/pipeline/parse.py tests/mlb/pipeline/test_lineup_slots.py
git commit -m "$(cat <<'EOF'
feat(mlb): parse starting nines from battingOrder 00 codes

EOF
)"
```

---

### Task 7: League eligibility (relievers out, two-way in)

**Files:**
- Modify: `src/mlb/models/batter_rates.py`
- Test: `tests/mlb/models/test_batter_rates.py`

**Interfaces:**
- Produces: `league_eligible_pas(pas: pd.DataFrame, nines: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

Three PAs in game 1: starter-pitcher batting (`is_pitcher_in_game=1`, 00-slot P), reliever PH (`is_pitcher_in_game=1`, not in nines), Ohtani DH (`is_pitcher_in_game=1`, 00-slot DH `slot_is_pitcher=0`). Assert league-eligible contains only the Ohtani row.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py::test_league_keeps_two_way_drops_relievers -v`

Expected: FAIL function missing.

- [ ] **Step 3: Write minimal implementation**

Two-way keys = `(game_pk, batter_id)` from `nines` where `slot_is_pitcher == 0`. Keep a PA if `is_pitcher_in_game == 0` or `(game_pk, batter_id)` in two-way keys.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/models/test_batter_rates.py::test_league_keeps_two_way_drops_relievers -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/models/batter_rates.py tests/mlb/models/test_batter_rates.py
git commit -m "$(cat <<'EOF'
feat(mlb): exclude pitcher PAs from league K/PA except two-way hitters

EOF
)"
```

---

### Task 8: freeze_lineup_slot_rates

**Files:**
- Modify: `src/mlb/pipeline/lineup_slots.py`
- Test: `tests/mlb/pipeline/test_lineup_slots.py`

**Interfaces:**
- Consumes: `shrink_batter_k_pa`, `league_eligible_pas`, `rate_version`, `LeagueKPa`
- Produces: `freeze_lineup_slot_rates(slots, batter_pas, people, config, *, cutoff, opposing_pitcher_hand, vs_pitcher_id) -> pd.DataFrame` filling all `k_pa_*` / `pa_*` columns and `rate_version`

- [ ] **Step 1: Write the failing test**

Minimal nines + PAs. Assert 365 vs-hand finite; empty opposing hand still writes `k_pa_overall_shrunk_365`; imputed PA rows do not enter counts.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py::test_freeze_rates_skips_imputed -v`

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Build `LeagueKPa` from `league_eligible_pas` with `event_time_utc < cutoff` and `event_time_imputed==0`: overall K/PA; by `batter_bats`; by `(batter_bats, pitcher_hand)`. For each slot row, `bats` from `people`, call `shrink_batter_k_pa`, unpack three windows onto columns.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py -k freeze -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/lineup_slots.py tests/mlb/pipeline/test_lineup_slots.py
git commit -m "$(cat <<'EOF'
feat(mlb): freeze three-window K/PA vectors onto lineup slots

EOF
)"
```

---

### Task 9: ingest_play_by_play (gzip, wall-clock dedupe)

**Files:**
- Modify: `src/mlb/pipeline/pbp.py`
- Test: `tests/mlb/pipeline/test_pbp.py`

**Interfaces:**
- Consumes: `snapshot_raw` or local gzip under `config.raw_dir / "mlb_pbp"`
- Produces: `ingest_play_by_play(config, *, game_pks, http=None, people=None) -> pd.DataFrame` writing `batter_pas` via store concat then `drop_duplicates("pa_id", keep="last")` after sorting `ingested_at_utc`

- [ ] **Step 1: Write the failing test**

`http=None` reads a gzip (or fixture JSON) from `tmp_path` config. First ingest 2 PAs. Second ingest same `pa_id` with later `ingested_at`. Assert one row, later timestamp wins. Assert no `is_strikeout` column on store read.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_pbp.py::test_reingest_keeps_latest_wall_clock -v`

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Write `{game_pk}.json.gz`. Parse with wall-clock `ingested_at`. Append to existing `batter_pas`, sort, `drop_duplicates("pa_id", keep="last")`, `store.write_table`. Prefer live-feed URL; if plays missing, GET `PBP_URL_TEMPLATE`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_pbp.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/pbp.py tests/mlb/pipeline/test_pbp.py
git commit -m "$(cat <<'EOF'
feat(mlb): idempotent gzip play-by-play ingest

EOF
)"
```

---

### Task 10: ingest_lineup_slots (24h stamp, DH skip, coverage)

**Files:**
- Modify: `src/mlb/pipeline/lineup_slots.py`
- Test: `tests/mlb/pipeline/test_lineup_slots.py`

**Interfaces:**
- Consumes: `parse_starting_nine`, `freeze_lineup_slot_rates`, `game_versions`
- Produces:
  - `BOXSCORE_00_LEAD = pd.Timedelta(hours=24)`
  - `ingest_lineup_slots(config, *, game_pks, provenance, http=None) -> pd.DataFrame`
  - `write_lineup_coverage(slots, skips, *, season) -> dict`
  - `assert_lineup_coverage(coverage, *, season) -> None`
  - `is_dummy_dh2_start(start, game1_start) -> bool`

- [ ] **Step 1: Write the failing tests**

1. `boxscore_00` `ingested_at == start - 24h`. Replace config `forecast_horizon_hours=3`; cutoff = start-3h; assert `ingested_at < cutoff`. A stamp of `start-2h` would fail that cutoff — do not use it.
2. DH2 NaT start → zero `boxscore_00` rows, skip reason `dh2_dummy_start`. Same `game_pk` with `provenance=live_feed` and wall clock before cutoff still writes 9 rows.
3. `observed_before_cutoff == (provenance == "live_feed")`.
4. Coverage: healthy 2 sides × 9 = 18 slots, `n_boxscore_ok=1`, complete-nine 2/2; `assert_lineup_coverage` passes. Zero slots raises. 1 complete side of 2 raises (`0.5 < 0.95`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py -k "stamp or dh2 or coverage or observed" -v`

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Earliest `game_versions.scheduled_start_utc` for original start. Dummy DH2: `doubleheader==2` and (NaT or midnight UTC or equal to game 1 start on same date). `boxscore_00` ingest time = start − 24h. Live: wall clock; drop if `>= cutoff`. Probable opposing starter: import `select_game_version` from `src.mlb.pipeline.features` (already latest `valid_from_utc < cutoff`) and take `probable_home_pitcher_id` / `probable_away_pitcher_id` opposite the batting side; hand from `people.throws`. Skip incomplete nines (`len != 9` distinct slots) with `incomplete_nine`. No boxscore → `no_boxscore`. After each season, `assert_lineup_coverage`. Resume: skip existing `(game_pk, team_id, slot, rate_version, provenance)`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/lineup_slots.py tests/mlb/pipeline/test_lineup_slots.py
git commit -m "$(cat <<'EOF'
feat(mlb): ingest lineup_slots with 24h retrospective stamp and coverage floors

EOF
)"
```

---

### Task 11: As-of select, probable pitcher, mismatch rate, live re-freeze clock

**Files:**
- Modify: `src/mlb/pipeline/lineup_slots.py`
- Test: `tests/mlb/pipeline/test_lineup_slots.py`

**Interfaces:**
- Produces:
  - `select_lineup_slots(slots, *, cutoff, rate_version) -> pd.DataFrame` — filter version first, then `ingested_at_utc < cutoff`, latest per `(game_pk, team_id, slot, rate_version)`
  - `select_probable_pitcher(game_versions, game_pk, cutoff, *, batting_is_home: bool) -> tuple[int | None, str]` — latest `valid_from_utc < cutoff`
  - `earliest_scheduled_start(game_versions, game_pk) -> pd.Timestamp`
  - `lineup_identity_mismatch_rate(live, official) -> float` join `(game_pk, team_id, slot)`
  - `copy_live_ingest_clock(existing, new_rows) -> pd.DataFrame` — live re-freeze copies original `ingested_at_utc`

- [ ] **Step 1: Write the failing tests**

Mismatch: two teams, identical nines, join without `team_id` would be 50%; with `team_id` rate is 0.

Select: two `rate_version`s; requesting A does not return B.

Probable: versions at T-10d pitcher 1, T-1h pitcher 2; cutoff T; assert pitcher 2. Earliest start still T-10d’s `scheduled_start_utc` if that start is earlier.

Live re-freeze: original ingest 12:00, new freeze 18:00, cutoff 13:00; written `ingested_at` stays 12:00 so it remains `< cutoff`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py -k "mismatch or select_lineup or probable or live_clock" -v`

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Reuse the same validity window logic as `select_game_version` in `src/mlb/pipeline/features.py` for probable (latest `valid_from`). Earliest start = `min(scheduled_start_utc)` for `game_pk`. Mismatch: inner join on `(game_pk, team_id, slot)`, mean of `batter_id` inequality among joined rows; empty join returns `float("nan")`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_lineup_slots.py tests/mlb/models/test_batter_rates.py tests/mlb/pipeline/test_pbp.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/pipeline/lineup_slots.py tests/mlb/pipeline/test_lineup_slots.py
git commit -m "$(cat <<'EOF'
feat(mlb): version-first slot select and team-keyed identity mismatch

EOF
)"
```

---

### Task 12: CLI wiring

**Files:**
- Modify: `src/mlb/cli.py` (parsers, handlers, `_PIPELINE_MODULES` if needed)
- Modify: `src/mlb/pipeline/__init__.py` exports
- Modify: `tests/mlb/test_cli.py`

**Interfaces:**
- Produces: subcommands `ingest-play-by-play --start-season INT --end-season INT`, `ingest-lineup-slots --start-season INT --end-season INT`; `snapshot-lineups` writes `lineup_slots` with `provenance=live_feed`

- [ ] **Step 1: Write the failing test**

Extend `test_help_lists_subcommands` with `ingest-play-by-play` and `ingest-lineup-slots`. Fixture CLI: `--fixture snapshot-lineups --game-pk 500043` with tmp config writes `lineup_slots` part-0 (may be empty if fixture nine missing — then assert the command returns 0 and does not raise). Prefer asserting help text first.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/test_cli.py::test_help_lists_subcommands -v`

Expected: FAIL missing subcommand names.

- [ ] **Step 3: Write minimal implementation**

Register parsers and handlers calling `ingest_play_by_play` / `ingest_lineup_slots`. `snapshot-lineups` calls `ingest_lineup_slots(..., provenance="live_feed", game_pks=[args.game_pk])`. `--fixture` keeps `http=None`. Game_pk lists for season ingest: unique `game_pk` from stored `pitcher_starts` or `game_versions` in that season; if empty, no-op return 0.

Print the coverage dict after `ingest-lineup-slots`. After a real backfill (not pytest), also print slot-1–9 std of `k_pa_vs_hand_shrunk_365` vs raw `k/pa_all_365` grouped by `pa_all_365` decile (diagnostic, stdout only).

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb/test_cli.py tests/mlb -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb/cli.py src/mlb/pipeline/__init__.py tests/mlb/test_cli.py
git commit -m "$(cat <<'EOF'
feat(mlb): CLI ingest for play-by-play and lineup slots

EOF
)"
```

---

### Task 13: Docs and regression guards

**Files:**
- Modify: `docs/mlb/data_dictionary.yaml`
- Modify: `docs/mlb/CONTRACT.md`
- Test: `tests/mlb/pipeline/test_gamelog_untouched.py`

**Interfaces:**
- Consumes: none
- Produces: dictionary lines for Stanek, Ohtani, as-of columns, inherited 225, raw PA counts; CONTRACT signatures matching the spec API

- [ ] **Step 1: Write the failing test**

```python
from src.mlb.pipeline.hf_tables import pregame_from_starts
from src.mlb.schemas import STRIKEOUT_FEATURE_COLUMNS


def test_gamelog_pregame_still_team_fallback(mlb_config, fixture_tables) -> None:
    starts = fixture_tables["pitcher_starts"]
    pre = pregame_from_starts(starts, mlb_config)
    assert set(pre["lineup_state"].unique()) == {"team_fallback"}
    assert pre["lineup_batter_ids_json"].eq("[]").all()


def test_strikeout_feature_columns_unchanged() -> None:
    assert "k_bf_shrunk_365" in STRIKEOUT_FEATURE_COLUMNS
    assert "k_pa_vs_hand_shrunk_365" not in STRIKEOUT_FEATURE_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlb/pipeline/test_gamelog_untouched.py -v`

Expected: FAIL until file exists; after file exists this should already PASS if earlier tasks kept the constraint. If it fails because someone wired features, stop and revert that wiring.

- [ ] **Step 3: Write dictionary and CONTRACT**

Required dictionary lines (spec):

- Stanek 2019-05-15: 00-codes keep the opener; PBP-first-nine is not the starting nine.
- Ohtani 2022+: two-way carve-out; `slot_is_pitcher` follows batting-slot position.
- `observed_before_cutoff=0` is more informative than live; do not group by `lineup_state` alone.
- `batter_k_prior_strength=225` inherited, not NLL-tuned. `pa_*` are raw counts.
- `batter_pas.ingested_at_utc` is write recency; lineup as-of is `lineup_slots.ingested_at_utc`; PA as-of is `event_time_utc`.

Copy public signatures from the spec into `CONTRACT.md`. Add `batter_pas` and `lineup_slots` to the required-tables list.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/mlb -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/mlb/data_dictionary.yaml docs/mlb/CONTRACT.md tests/mlb/pipeline/test_gamelog_untouched.py
git commit -m "$(cat <<'EOF'
docs(mlb): document lineup slot vectors and keep nb_k_v1 unwired

EOF
)"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| 00-filter, never `battingOrder` array | 6 |
| Stanek / Ohtani tests | 6, 7 |
| No stored `is_strikeout` | 1, 2, 5 |
| PBP join `event_time` only; wall-clock dedupe | 5, 9 |
| Imputed +4h excluded from windows | 5, 8 |
| `start − 24h` stamp; horizon 3h still visible | 10 |
| DH2 dummy skip; live still writes | 10 |
| Coverage floor 0.95 + skip reasons | 10 |
| `observed_before_cutoff` invariant | 10 |
| Dedupe `(game_pk, team_id, slot, rate_version)` | 11 |
| Mismatch join includes `team_id` | 11 |
| Latest-before-cutoff probable; earliest start | 11 |
| Live re-freeze copies clock | 11 |
| Leave-one-split-out; S cells; missing bats = 1.0 | 3, 4 |
| Store 60/365/prior2 + overall | 1, 8 |
| Reliever vs two-way league series | 7 |
| gzip / prefer live feed | 9 |
| `season` from `gameData.game.season` | 6 |
| CLI | 12 |
| Docs + `nb_k_v1` / `team_fallback` untouched | 13 |
| Spread diagnostic stdout | 12 |
| Log5 / mixture / `STRIKEOUT_FEATURE_COLUMNS` | 13 guards; not implemented |

No TBD. `LeagueKPa`, `shrink_batter_k_pa` return dict, and `select_lineup_slots` names are used consistently across tasks.
