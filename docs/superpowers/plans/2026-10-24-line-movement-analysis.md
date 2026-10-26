# Line Movement Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a thin dated notebook plus a local `line_movement` package that loads seven books’ prop JSON, validates/normalizes quotes, matches markets, detects movements, scores leader/follower relationships, and writes a daily report.

**Architecture:** All reusable logic lives under `notebooks/nba/line_movement_analysis/src/line_movement/`. The notebook only configures `DATE`, imports the package, displays tables/charts, and triggers report writes. Top-level `src/odds/` (DB upserts) is not modified. Tests live in `tests/line_movement/` and put the local package on `sys.path` via conftest.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, seaborn, pytest. Local JSON under `data/props/*/nba/`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-24-line-movement-analysis-design.md`
- Package root: `notebooks/nba/line_movement_analysis/src/line_movement/` (not top-level `src/`)
- Do not modify `src/odds/**` or DB upsert paths
- Do not invent PrizePicks sides; missing-side rows stay in DQ and drop out of match / leader-follower
- Market group key: `(player_norm, event_key, prop, period, side, line)`
- Continuity key (line moves + episodes): `(player_norm, event_key, prop, period, side)`
- Follower window default: 30 minutes; also emit uncapped next-mover diagnostics
- Schedule filter: game `start_time` in [12:00, 19:00] America/Los_Angeles on the analysis date
- Filename `snapshot_at` is Pacific and is the ordering clock
- Tests: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/line_movement/ -q`
- Do not `git commit` unless the user asks (skip every Commit step until then)

## File map

- Create: `notebooks/nba/line_movement_analysis/src/line_movement/__init__.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/paths.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/normalize.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/load.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/validate.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/match.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/detect.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/leaders.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/report.py`
- Create: `tests/line_movement/conftest.py`
- Create: `tests/line_movement/test_normalize.py`
- Create: `tests/line_movement/test_detect.py`
- Create: `tests/line_movement/test_leaders.py`
- Create: `tests/line_movement/test_match.py`
- Modify: `notebooks/nba/line_movement_analysis/line_movement_exploration_2026-10-24.ipynb`
- Runtime (not hand-authored): `notebooks/nba/line_movement_analysis/reports/<date>/`

---

### Task 1: Package path + normalize helpers

**Files:**
- Create: `tests/line_movement/conftest.py`
- Create: `tests/line_movement/test_normalize.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/paths.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/normalize.py`
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/__init__.py`

**Interfaces:**
- Produces:
  - `PACKAGE_SRC: Path` via `paths.package_src()` / conftest insert
  - `normalize_player(name: str | None) -> str | None`
  - `normalize_prop(stat: str | None) -> str | None`
  - `normalize_side(side: str | None) -> str | None`  # only `over`/`under` or None
  - `normalize_period(period: str | None) -> str`  # default `"game"`
  - `normalize_event_key(event_name: str | None) -> str | None`
  - `DEFAULT_FOLLOW_WINDOW = timedelta(minutes=30)`
- Consumes: nothing

- [ ] **Step 1: Write the failing tests**

Create `tests/line_movement/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

_PKG_SRC = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "nba"
    / "line_movement_analysis"
    / "src"
)
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))
```

Create `tests/line_movement/test_normalize.py`:

```python
from __future__ import annotations

from line_movement.normalize import (
    normalize_event_key,
    normalize_period,
    normalize_player,
    normalize_prop,
    normalize_side,
)


def test_normalize_player_strips_suffix_and_case():
    assert normalize_player("  Luka Dončić Jr. ") == "luka doncic"
    assert normalize_player(None) is None


def test_normalize_prop_aliases():
    assert normalize_prop("Points") == "points"
    assert normalize_prop("player_points") == "points"
    assert normalize_prop("Pts + Rebs + Asts") == "pra"
    assert normalize_prop(None) is None


def test_normalize_side_and_period():
    assert normalize_side("Over") == "over"
    assert normalize_side("u") == "under"
    assert normalize_side(None) is None
    assert normalize_period(None) == "game"
    assert normalize_period("1H") == "1h"


def test_normalize_event_key_orders_teams():
    assert normalize_event_key("Lakers @ Celtics") == normalize_event_key(
        "Celtics vs Lakers"
    )
    assert normalize_event_key(None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/line_movement/test_normalize.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'line_movement'` (or import error for missing symbols).

- [ ] **Step 3: Implement paths + normalize**

Create `notebooks/nba/line_movement_analysis/src/line_movement/paths.py`:

```python
from __future__ import annotations

from pathlib import Path


def package_src() -> Path:
    return Path(__file__).resolve().parents[1]


def analysis_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not find repo root (pyproject.toml)")
```

Create `notebooks/nba/line_movement_analysis/src/line_movement/normalize.py` implementing:

- Unicode NFKD → ASCII fold for players; lowercase; strip; drop trailing `jr`/`sr`/`ii`/`iii`/`iv`
- Prop map covering at least: points, rebounds, assists, threes / 3-pointers / made threes → `threes`, pra / pts+rebs+asts → `pra`, steals, blocks, turnovers; unknown non-empty stats → lowercase snake (`re.sub(r"[^a-z0-9]+", "_", ...)`)
- Side: `over`/`o` → `over`; `under`/`u` → `under`; else None
- Period: None/blank → `"game"`; else lowercase stripped token
- Event key: extract two team tokens from patterns `A @ B`, `A vs B`, `A vs. B`, `A at B`; normalize each team string (lowercase, strip city/mascot punctuation lightly); sort the two tokens alphabetically; join with `"__"`. If fewer than two teams parseable, return lowercase collapsed `event_name` or None

Create `__init__.py` exporting the normalize helpers and `DEFAULT_FOLLOW_WINDOW`:

```python
from datetime import timedelta

from line_movement.normalize import (
    normalize_event_key,
    normalize_period,
    normalize_player,
    normalize_prop,
    normalize_side,
)

DEFAULT_FOLLOW_WINDOW = timedelta(minutes=30)

__all__ = [
    "DEFAULT_FOLLOW_WINDOW",
    "normalize_event_key",
    "normalize_period",
    "normalize_player",
    "normalize_prop",
    "normalize_side",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/line_movement/test_normalize.py -v`

Expected: PASS

- [ ] **Step 5: Commit** (skip unless user asks)

---

### Task 2: Load seven books into a raw frame

**Files:**
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/load.py`
- Create: `tests/line_movement/test_load.py`
- Modify: `notebooks/nba/line_movement_analysis/src/line_movement/__init__.py`

**Interfaces:**
- Consumes: `paths.repo_root`
- Produces:
  - `BOOKS: tuple[tuple[str, str, str], ...]`  # (book, relpath, kind)
  - `snapshot_at(path: Path) -> datetime`
  - `prop_files(folder: Path, date: str, kind: str) -> list[Path]`
  - `load_book(book: str, folder: Path, kind: str, date: str) -> pd.DataFrame`
  - `load_all_books(date: str, *, root: Path | None = None) -> pd.DataFrame`
  - Raw columns before normalize: `book, snapshot_at, fetched_at, source_file, player, stat, line, side, american_price, decimal_price, implied_probability, market_name, event_name, start_time, line_updated_at, is_main, odds_type`

Port flattening logic from the current notebook cell (picks / projections / games). Keep behavior:

- Filename stamp → Pacific `snapshot_at`
- `games` kind: only `*_props.json`
- non-games: exclude `*_team.json`
- Expand over/under american prices for games payloads
- Numeric coerce `line` / `american_price`; UTC parse datetimes

- [ ] **Step 1: Write the failing test**

Create `tests/line_movement/test_load.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from line_movement.load import BOOKS, load_all_books, prop_files, snapshot_at
from line_movement.paths import repo_root


def test_books_has_seven_providers():
    assert len(BOOKS) == 7
    assert {b[0] for b in BOOKS} == {
        "fanduel",
        "draftkings",
        "pinnacle",
        "novig",
        "prophetx",
        "underdog",
        "prizepicks",
    }


def test_snapshot_at_is_pacific():
    path = Path("fanduel_nba_2026-10-24_120245.json")
    ts = snapshot_at(path)
    assert str(ts.tzinfo) == "America/Los_Angeles" or ts.tzname() in {"PDT", "PST"}
    assert ts.hour == 12 and ts.minute == 2


def test_load_all_books_smoke_2026_10_24():
    root = repo_root()
    # Skip if data not present in this checkout
    sample = root / "data/props/fanduel/nba"
    if not any(sample.glob("*2026-10-24*.json")):
        return
    frame = load_all_books("2026-10-24", root=root)
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["book"].unique()) == {b[0] for b in BOOKS}
    assert "snapshot_at" in frame.columns
    assert len(frame) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/line_movement/test_load.py::test_books_has_seven_providers -v`

Expected: FAIL (`ModuleNotFoundError` or missing `load`)

- [ ] **Step 3: Implement `load.py`**

Move the notebook’s `BOOKS`, `snapshot_at`, `prop_files`, flatten helpers, and `load_book` into `load.py`. Add:

```python
def load_all_books(date: str, *, root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root()
    frames = [
        load_book(book, root / rel, kind, date) for book, rel, kind in BOOKS
    ]
    return pd.concat(frames, ignore_index=True)
```

Export `load_all_books`, `BOOKS` from `__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/line_movement/test_load.py -v`

Expected: PASS (smoke test runs against real data when present)

- [ ] **Step 5: Commit** (skip unless user asks)

---

### Task 3: Normalize frame + validate + schedule filter

**Files:**
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/validate.py`
- Create: `tests/line_movement/test_validate.py`
- Modify: `notebooks/nba/line_movement_analysis/src/line_movement/normalize.py` (add `normalize_quotes`)
- Modify: `notebooks/nba/line_movement_analysis/src/line_movement/__init__.py`

**Interfaces:**
- Produces:
  - `normalize_quotes(raw: pd.DataFrame) -> pd.DataFrame`  
    Adds `player_norm`, `prop`, `period`, `side` (normalized), `event_key`; keeps raw columns.
  - `QualityReport` dataclass or typed dict with keys: `missing_rates_by_book`, `duplicate_count`, `schema_flags`, `dropped_by_schedule`, `snapshot_span_by_book`
  - `validate_quotes(quotes: pd.DataFrame) -> QualityReport`
  - `filter_schedule(quotes: pd.DataFrame, *, date: str, start_hour: int = 12, end_hour: int = 19, tz: str = "America/Los_Angeles") -> tuple[pd.DataFrame, int]`  
    Keeps rows whose `start_time` is null **or** local time on `date` in `[start_hour, end_hour)`; returns `(kept, n_dropped)`. Spec wants analysis on games in window — **drop** rows with known `start_time` outside the window; **keep** null `start_time` but count them under `schema_flags["missing_start_time"]`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import pandas as pd

from line_movement.normalize import normalize_quotes
from line_movement.validate import filter_schedule, validate_quotes


def _raw_row(**overrides):
    base = {
        "book": "fanduel",
        "snapshot_at": pd.Timestamp("2026-10-24 12:00:00", tz="America/Los_Angeles"),
        "player": "Luka Doncic",
        "stat": "Points",
        "line": 28.5,
        "side": "Over",
        "american_price": -110,
        "event_name": "DAL @ BOS",
        "start_time": pd.Timestamp("2026-10-24 17:00:00", tz="UTC"),
        "fetched_at": pd.NaT,
        "source_file": "x.json",
        "line_updated_at": pd.NaT,
        "is_main": True,
        "decimal_price": None,
        "implied_probability": None,
        "market_name": None,
        "odds_type": None,
    }
    base.update(overrides)
    return base


def test_normalize_quotes_adds_keys():
    raw = pd.DataFrame([_raw_row()])
    out = normalize_quotes(raw)
    assert out.loc[0, "player_norm"] == "luka doncic"
    assert out.loc[0, "prop"] == "points"
    assert out.loc[0, "side"] == "over"
    assert out.loc[0, "period"] == "game"
    assert out.loc[0, "event_key"] is not None


def test_filter_schedule_drops_late_games():
    raw = pd.DataFrame(
        [
            _raw_row(
                start_time=pd.Timestamp("2026-10-24 16:00:00", tz="UTC")
            ),  # 9am PT — drop
            _raw_row(
                start_time=pd.Timestamp("2026-10-24 20:00:00", tz="UTC")
            ),  # 1pm PT — keep
            _raw_row(
                start_time=pd.Timestamp("2026-10-25 02:00:00", tz="UTC")
            ),  # 7pm PT — drop (end exclusive 19:00)
        ]
    )
    quotes = normalize_quotes(raw)
    kept, dropped = filter_schedule(quotes, date="2026-10-24")
    assert dropped == 2
    assert len(kept) == 1


def test_validate_flags_missing_side():
    raw = pd.DataFrame(
        [_raw_row(book="prizepicks", side=None, american_price=None)]
    )
    quotes = normalize_quotes(raw)
    report = validate_quotes(quotes)
    assert report.missing_rates_by_book["prizepicks"]["side"] == 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `/Users/alexgonzalez/Documents/nba_quant/.venv/bin/python -m pytest tests/line_movement/test_validate.py -v`

- [ ] **Step 3: Implement `normalize_quotes`, `validate_quotes`, `filter_schedule`**

`validate_quotes` must compute:

- missing rate by book for `player`, `prop`, `side`, `line`, `american_price`, `event_key`
- duplicate count on `(book, snapshot_at, player_norm, event_key, prop, period, side, line)`
- schema flags: books with all-null `side`, all-null `is_main`, all-null `american_price`
- snapshot min/max per book

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit** (skip unless user asks)

---

### Task 4: Match equivalent markets

**Files:**
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/match.py`
- Create: `tests/line_movement/test_match.py`
- Modify: `__init__.py`

**Interfaces:**
- Produces:
  - `MARKET_KEY_COLS = ("player_norm", "event_key", "prop", "period", "side", "line")`
  - `attach_market_keys(quotes: pd.DataFrame) -> pd.DataFrame`  
    Adds `market_key` (tuple or stable string) and `continuity_key`.
  - `build_market_groups(quotes: pd.DataFrame) -> pd.DataFrame`  
    One row per market_key with: key cols, `n_books`, `books` (sorted join), `n_rows`, `matchable` (bool: all key parts non-null and `n_books >= 1`).
  - `cross_book_groups(groups: pd.DataFrame) -> pd.DataFrame`  
    Filter `n_books >= 2`.

Rows missing any market key part are assigned `market_key=None` and excluded from `cross_book_groups`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pandas as pd

from line_movement.match import build_market_groups, cross_book_groups
from line_movement.normalize import normalize_quotes


def test_cross_book_group_requires_two_books():
    rows = [
        {
            "book": "fanduel",
            "snapshot_at": pd.Timestamp("2026-10-24 12:00:00", tz="America/Los_Angeles"),
            "player": "A",
            "stat": "points",
            "line": 20.5,
            "side": "over",
            "event_name": "AAA @ BBB",
            "american_price": -110,
            "start_time": pd.NaT,
            "fetched_at": pd.NaT,
            "source_file": "a",
            "line_updated_at": pd.NaT,
            "is_main": True,
            "decimal_price": None,
            "implied_probability": None,
            "market_name": None,
            "odds_type": None,
        },
        {
            "book": "draftkings",
            "snapshot_at": pd.Timestamp("2026-10-24 12:00:00", tz="America/Los_Angeles"),
            "player": "A",
            "stat": "points",
            "line": 20.5,
            "side": "over",
            "event_name": "BBB vs AAA",
            "american_price": -115,
            "start_time": pd.NaT,
            "fetched_at": pd.NaT,
            "source_file": "b",
            "line_updated_at": pd.NaT,
            "is_main": True,
            "decimal_price": None,
            "implied_probability": None,
            "market_name": None,
            "odds_type": None,
        },
        {
            "book": "prizepicks",
            "snapshot_at": pd.Timestamp("2026-10-24 12:00:00", tz="America/Los_Angeles"),
            "player": "A",
            "stat": "points",
            "line": 20.5,
            "side": None,
            "event_name": None,
            "american_price": None,
            "start_time": pd.NaT,
            "fetched_at": pd.NaT,
            "source_file": "c",
            "line_updated_at": pd.NaT,
            "is_main": None,
            "decimal_price": None,
            "implied_probability": None,
            "market_name": None,
            "odds_type": "standard",
        },
    ]
    quotes = normalize_quotes(pd.DataFrame(rows))
    groups = build_market_groups(quotes)
    cross = cross_book_groups(groups)
    assert len(cross) == 1
    assert cross.iloc[0]["n_books"] == 2
```

- [ ] **Step 2–4:** Implement, run pytest until PASS, skip commit unless asked.

---

### Task 5: Detect line and price changes

**Files:**
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/detect.py`
- Create: `tests/line_movement/test_detect.py`
- Modify: `__init__.py`

**Interfaces:**
- Produces: `detect_movements(quotes: pd.DataFrame) -> pd.DataFrame`

Movement log columns:

`book, player_norm, event_key, prop, period, side, series` (`price`|`line`), `prev_line, new_line, prev_american_price, new_american_price, change_type, line_up, line_down, price_shortened, price_lengthened, snapshot_at, prev_snapshot_at, lag_seconds, line_choice`

Rules (exact):

1. **Price series:** groupby `(book, player_norm, event_key, prop, period, side, line)` sort `snapshot_at`; emit when `american_price` changes vs previous row (`pd.isna` XOR or value differs by > 1e-9). `change_type='price'`. Set price direction flags from american_price decrease/increase.
2. **Line continuity series:** for each snapshot and continuity key, pick one active line: prefer `is_main == True`; else latest non-null `line_updated_at`; else median `line`. Set `line_choice` to `main` / `updated_at` / `median`. Groupby continuity + book; emit when chosen `line` changes; `change_type='line'`; set `line_up`/`line_down`.
3. If a single observation changes both chosen line and price on the price-series key, emit one row with `change_type='both'` only when comparing the price-series rows that also changed line identity — keep implementation simple: emit separate `price` and `line` rows; do not require a merged `both` row if dual emission is clearer. Spec allows `both`; prefer emitting **one row per series** (`price` and/or `line`) and set `change_type` to that series name. Update spec compliance note in module docstring: dual rows replace `both`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import pandas as pd

from line_movement.detect import detect_movements
from line_movement.match import attach_market_keys
from line_movement.normalize import normalize_quotes


def _q(book, snap, line, price, **kw):
    row = {
        "book": book,
        "snapshot_at": pd.Timestamp(snap, tz="America/Los_Angeles"),
        "player": "Test Player",
        "stat": "points",
        "line": line,
        "side": "over",
        "american_price": price,
        "event_name": "AAA @ BBB",
        "start_time": pd.NaT,
        "fetched_at": pd.NaT,
        "source_file": "x",
        "line_updated_at": pd.NaT,
        "is_main": True,
        "decimal_price": None,
        "implied_probability": None,
        "market_name": None,
        "odds_type": None,
    }
    row.update(kw)
    return row


def test_price_only_move():
    raw = pd.DataFrame(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 20.5, -120),
        ]
    )
    quotes = attach_market_keys(normalize_quotes(raw))
    moves = detect_movements(quotes)
    assert len(moves) == 1
    assert moves.iloc[0]["change_type"] == "price"
    assert bool(moves.iloc[0]["price_shortened"]) is True


def test_no_move_when_identical():
    raw = pd.DataFrame(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 20.5, -110),
        ]
    )
    quotes = attach_market_keys(normalize_quotes(raw))
    assert detect_movements(quotes).empty


def test_line_continuity_move():
    raw = pd.DataFrame(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 21.5, -110),
        ]
    )
    quotes = attach_market_keys(normalize_quotes(raw))
    moves = detect_movements(quotes)
    line_moves = moves[moves["change_type"] == "line"]
    assert len(line_moves) == 1
    assert bool(line_moves.iloc[0]["line_up"]) is True
```

- [ ] **Step 2–4:** Implement `detect_movements`, pytest PASS, skip commit unless asked.

---

### Task 6: Leader / follower episodes and pairwise stats

**Files:**
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/leaders.py`
- Create: `tests/line_movement/test_leaders.py`
- Modify: `__init__.py`

**Interfaces:**
- Consumes: movement log from Task 5; `DEFAULT_FOLLOW_WINDOW`
- Produces:
  - `build_episodes(movements: pd.DataFrame, *, window: timedelta = DEFAULT_FOLLOW_WINDOW) -> pd.DataFrame`  
    Columns: `episode_id, player_norm, event_key, prop, period, side, direction, leader_book, leader_snapshot_at, change_type`
  - `build_follows(movements, episodes, *, window) -> pd.DataFrame`  
    Columns: `episode_id, follower_book, follower_snapshot_at, lag_seconds, within_window, direction`
  - `pairwise_stats(follows: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame`  
    Columns: `leader_book, follower_book, lead_count, follow_count_windowed, follow_rate_windowed, median_lag_windowed_sec, mean_lag_windowed_sec, median_lag_uncapped_sec`

**Episode algorithm (deterministic):**

1. Work on movements that have non-null continuity fields.
2. Derive `direction` string: `line_up` / `line_down` / `price_shortened` / `price_lengthened` (first true flag; if line and price both set, prefer line direction).
3. Sort by continuity key, `snapshot_at`, `book`.
4. For each continuity key + direction stream: start new episode when gap from current episode leader time > `window` (or no open episode). Leader = earliest row in cluster.
5. Later different-book same direction rows attach as follows; retain only the
   earliest row per `(episode_id, follower_book)`. `within_window = lag <=
   window`; uncapped and windowed views both use that first follow.

`lead_count` for pair (L,F): episodes where `leader_book==L`.  
`follow_count_windowed`: follows with `leader_book==L`, `follower_book==F`, `within_window`.  
`follow_rate_windowed`: `follow_count_windowed / lead_count` (NaN if lead_count==0).  
Uncapped median lag uses all follows for that pair regardless of window.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from line_movement.leaders import build_episodes, build_follows, pairwise_stats


def _move(book, snap, **flags):
    base = {
        "book": book,
        "player_norm": "test player",
        "event_key": "aaa__bbb",
        "prop": "points",
        "period": "game",
        "side": "over",
        "change_type": "price",
        "line_up": False,
        "line_down": False,
        "price_shortened": False,
        "price_lengthened": False,
        "snapshot_at": pd.Timestamp(snap, tz="America/Los_Angeles"),
        "prev_snapshot_at": pd.NaT,
        "lag_seconds": None,
        "prev_line": 20.5,
        "new_line": 20.5,
        "prev_american_price": -110,
        "new_american_price": -120,
        "series": "price",
        "line_choice": None,
    }
    base.update(flags)
    return base


def test_follower_inside_and_outside_window():
    moves = pd.DataFrame(
        [
            _move("fanduel", "2026-10-24 12:00:00", price_shortened=True),
            _move("draftkings", "2026-10-24 12:15:00", price_shortened=True),
            _move("pinnacle", "2026-10-24 13:00:00", price_shortened=True),
        ]
    )
    window = timedelta(minutes=30)
    episodes = build_episodes(moves, window=window)
    follows = build_follows(moves, episodes, window=window)
    assert list(episodes["leader_book"]) == ["fanduel", "pinnacle"]
    assert list(follows["follower_book"]) == ["draftkings"]
    assert follows.iloc[0]["within_window"]

    stats = pairwise_stats(follows, episodes)
    fd = stats[
        (stats["leader_book"] == "fanduel")
        & (stats["follower_book"] == "draftkings")
    ].iloc[0]
    assert fd["follow_count_windowed"] == 1
    assert fd["median_lag_windowed_sec"] == 15 * 60
```

- [ ] **Step 2–4:** Implement, pytest PASS, skip commit unless asked.

---

### Task 7: Daily report writers

**Files:**
- Create: `notebooks/nba/line_movement_analysis/src/line_movement/report.py`
- Create: `tests/line_movement/test_report.py`
- Modify: `__init__.py`

**Interfaces:**
- Produces:
  - `report_dir(date: str, *, analysis_root: Path | None = None) -> Path`
  - `save_daily_report(*, date: str, quotes, groups, movements, episodes, follows, pairwise, quality: QualityReport, analysis_root: Path | None = None) -> Path`  
    Writes under `reports/<date>/`:
    - `quotes_normalized.csv` (csv default; try parquet if `pyarrow`/`fastparquet` importable, else csv only — always write csv for reliability)
    - `market_groups.csv`
    - `movements.csv`
    - `episodes.csv`
    - `follows.csv`
    - `pairwise_stats.csv`
    - `quality_report.json`
    - `coverage_by_book.png`
    - `movements_over_time.png`
    - `pairwise_lag_heatmap.png` (seaborn heatmap of median_lag_windowed_sec; empty-safe)
  - `plot_coverage(quotes, ax=None)`, `plot_movements_over_time(movements, ax=None)`, `plot_pairwise_lag(pairwise, ax=None)` returning the matplotlib Axes

- [ ] **Step 1: Write failing test** that calls `save_daily_report` with tiny frames into `tmp_path` (monkeypatch `analysis_root`), asserts expected filenames exist and `quality_report.json` parses.

- [ ] **Step 2–4:** Implement with matplotlib Agg backend in tests (`matplotlib.use("Agg")` at top of test module), PASS, skip commit unless asked.

---

### Task 8: Rewrite notebook sections 01–06

**Files:**
- Modify: `notebooks/nba/line_movement_analysis/line_movement_exploration_2026-10-24.ipynb`

**Interfaces:**
- Consumes: public exports from `line_movement`
- Produces: executed narrative for 2026-10-24; artifacts under `reports/2026-10-24/`

- [ ] **Step 1: Clear inline loader cells** — keep markdown **01** research question; replace **Data & Setup** with path bootstrap + imports only:

```python
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ANALYSIS_SRC = Path("src").resolve()
if not (ANALYSIS_SRC / "line_movement").is_dir():
    ANALYSIS_SRC = Path.cwd() / "notebooks/nba/line_movement_analysis/src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from line_movement import (
    DEFAULT_FOLLOW_WINDOW,
    build_episodes,
    build_follows,
    build_market_groups,
    cross_book_groups,
    detect_movements,
    filter_schedule,
    load_all_books,
    normalize_quotes,
    pairwise_stats,
    save_daily_report,
    validate_quotes,
)
from line_movement.match import attach_market_keys
from line_movement.paths import analysis_root, repo_root

DATE = "2026-10-24"
ROOT = repo_root()
sns.set_theme(style="whitegrid")
```

- [ ] **Step 2: Section 02 cells**

Markdown: `## 02 — Load and validate data`

Code:

```python
raw = load_all_books(DATE, root=ROOT)
quotes = normalize_quotes(raw)
quotes, dropped = filter_schedule(quotes, date=DATE)
quality = validate_quotes(quotes)
quality.dropped_by_schedule = dropped  # if mutable dataclass field
summary = (
    quotes.groupby("book", as_index=False)
    .agg(
        files=("source_file", "nunique"),
        rows=("player", "size"),
        players=("player_norm", "nunique"),
        props=("prop", "nunique"),
        first_snapshot=("snapshot_at", "min"),
        last_snapshot=("snapshot_at", "max"),
    )
)
summary
```

Show `quality.missing_rates_by_book` (as DataFrame).

- [ ] **Step 3: Section 03**

Markdown: `## 03 — Match equivalent markets`

```python
quotes = attach_market_keys(quotes)
groups = build_market_groups(quotes)
cross = cross_book_groups(groups)
cross.sort_values("n_books", ascending=False).head(20)
```

- [ ] **Step 4: Section 04**

Markdown: `## 04 — Detect line and price changes`

```python
movements = detect_movements(quotes)
movements.head(20)
```

- [ ] **Step 5: Section 05**

Markdown: `## 05 — Analyze leader/follower relationships`

```python
episodes = build_episodes(movements, window=DEFAULT_FOLLOW_WINDOW)
follows = build_follows(movements, episodes, window=DEFAULT_FOLLOW_WINDOW)
pairwise = pairwise_stats(follows, episodes)
display(episodes.head(20))
display(pairwise.sort_values("follow_count_windowed", ascending=False).head(20))
```

- [ ] **Step 6: Section 06**

Markdown: `## 06 — Daily report`  
Note in markdown: scrape resolution is ~15 minutes; first-observed ≠ proof of true leadership.

```python
out = save_daily_report(
    date=DATE,
    quotes=quotes,
    groups=groups,
    movements=movements,
    episodes=episodes,
    follows=follows,
    pairwise=pairwise,
    quality=quality,
    analysis_root=analysis_root(),
)
print(out)
```

- [ ] **Step 7: Manual verification**

From repo root (or analysis dir), run the notebook top-to-bottom (Jupyter or `jupyter nbconvert --execute` if available). Confirm:

- `reports/2026-10-24/` contains the CSV/JSON/PNG files
- movement log non-empty
- pairwise table non-empty for books with sides
- PrizePicks missing-side rate near 1.0 in quality report

- [ ] **Step 8: Commit** (skip unless user asks)

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Load seven providers’ JSON | 2 |
| Validate timestamps/duplicates/missing/schema | 3 |
| One normalized dataset | 3 |
| Standardize player/event/prop/period/side/line | 1, 3, 4 |
| Comparable market groups | 4 |
| Detect line/price changes vs previous snapshot | 5 |
| Leader/follower + windowed + uncapped | 6 |
| Daily report tables/charts/DQ/save | 7, 8 |
| Package under notebook `src/` | 1–7 |
| Thin notebook 02–06 | 8 |
| Unit tests normalize/detect/leaders | 1, 5, 6 |
| Do not invent PrizePicks sides | 3, 4, 8 |
| Do not touch `src/odds/` | global |

## Self-review notes

- `change_type='both'`: Task 5 deliberately emits separate `price`/`line` rows instead of a merged `both` to keep detection simple; document in `detect.py` docstring (spec-compatible via dual emission).
- Underdog often lacks `event_name` in raw picks — those rows will fail `event_key` and drop from cross-book groups; DQ must surface this (Task 3 schema flags).
- Schedule filter uses end-exclusive 19:00 Pacific so a 7:00 PM tip is excluded; if product intent was inclusive end, change `end_hour` handling in one place (`filter_schedule`).
