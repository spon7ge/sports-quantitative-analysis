# Rotowire NBA Lineup Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On each NBA odds run, fetch RotoWire lineups once and write one timestamped JSON file under `data/props/rotowire_nba_lineup/`.

**Architecture:** `NBADailyLineups.lineup_snapshot` parses HTML already in memory into the props payload. `src.scrapers.nba.nba_rotowire_lineup` performs the single GET, stamps Pacific time, and atomically writes the file. `run_all_odds` registers that module as the NBA job `nba_rotowire_lineup`, immediately before PrizePicks in the NBA list.

**Tech Stack:** Python, pytest, BeautifulSoup, requests (already used by `nba_lineups.py`). No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-24-rotowire-nba-lineup-design.md`.
- One GET of `https://www.rotowire.com/basketball/nba-lineups.php` per invocation. No book scraper fetches this page.
- Directory `data/props/rotowire_nba_lineup/`. Filename `rotowire_nba_YYYY-MM-DD_HHMMSS.json`. Clock `America/Los_Angeles`. Example `rotowire_nba_2026-10-24_223512.json`.
- `fetched_at` is that same instant as ISO-8601 with the Pacific offset (`2026-10-24T22:35:12-07:00` on that example).
- Temp file is `.<filename>.tmp` in the same directory, then rename. Failed request writes no file.
- `status` is `confirmed` when the chip has class `is-confirmed` or its text is `Confirmed Lineup`. Every other chip, including `Expected Lineup`, and a missing chip, is `projected`.
- `lineup` is every `li.lineup__player` before the first `li.lineup__title`, in order. No cap at five. No play-likelihood filter. Entries are `name` and `position` only.
- `may_not_play` is every `li.lineup__player` after that title, including `hide` / `is-ofs`. `badge` is the `.lineup__inj` text unchanged, or `null`. A starter-row badge is not copied into `may_not_play`.
- NBA cards only (`div.lineup.is-nba`). Do not use `_matchup_cards()` for the snapshot: that helper falls back to every `div.lineup` when no NBA card exists, which would keep a WNBA-only page.
- Do not change `expected_starters_by_abbr`, `posted_starting_flags`, or `src/live/nba_lineup.py`.
- No Supabase write. No second output format.
- `--fail-fast` behavior in `run_all_odds` stays as it is.
- Tests use fixture HTML. No live RotoWire call from pytest.
- Commits use a HEREDOC message. Do not skip hooks.

## File map

| File | Responsibility |
|---|---|
| `src/scrapers/nba/nba_lineups.py` | `lineup_snapshot(fetched_at: str) -> dict` and `_side_snapshot`. Reads `self.soup` only. |
| `src/scrapers/nba/nba_rotowire_lineup.py` | Filename, Pacific `fetched_at`, atomic write, `run()` / `main()`. One `NBADailyLineups()` call. |
| `src/scrapers/run_all_odds.py` | NBA job `nba_rotowire_lineup` before `nba_prizepick`. |
| `tests/scrapers/test_nba_rotowire_lineup.py` | Snapshot shape, filename, atomic write, single fetch. |
| `tests/scrapers/test_run_all_odds.py` | Job name, module, league, pool, order. |
| `tests/scrapers/test_nba_lineups.py` | Unchanged. Must still pass. |

---

### Task 1: Parse a lineup snapshot from HTML

**Files:**
- Modify: `src/scrapers/nba/nba_lineups.py` (add helpers after `_class_tokens`, add `lineup_snapshot` and `_side_snapshot` at the end of `NBADailyLineups`)
- Test: `tests/scrapers/test_nba_rotowire_lineup.py`
- Test: `tests/scrapers/test_nba_lineups.py` (run only; do not edit)

**Interfaces:**
- Consumes: `NBADailyLineups(html=...)`, `NBA_LINEUPS_URL`, `_is_nba_lineup_card`, `_class_tokens`, `_side_nickname`, `_resolve_abbr`
- Produces: `NBADailyLineups.lineup_snapshot(self, fetched_at: str) -> dict` with keys `source`, `url`, `fetched_at`, `games`. Each game has `away` and `home`. Each side has `team: str | None`, `status: str`, `lineup: list[dict]`, `may_not_play: list[dict]`. Lineup dict keys are `name` and `position`. May-not-play dict keys are `name`, `position`, and `badge`.

- [ ] **Step 1: Write the failing test**

Create `tests/scrapers/test_nba_rotowire_lineup.py`:

```python
"""RotoWire lineup props snapshot. Fixture HTML only; no network."""

from src.scrapers.nba.nba_lineups import NBA_LINEUPS_URL, NBADailyLineups

FETCHED_AT = "2026-10-24T22:35:12-07:00"

PAGE = """
<div class="lineup is-nba">
  <a class="lineup__mteam is-visit white">Celtics</a>
  <a class="lineup__team is-visit"><span class="lineup__abbr">BOS</span></a>
  <a class="lineup__mteam is-home white">76ers</a>
  <a class="lineup__team is-home"><span class="lineup__abbr">PHI</span></a>
  <ul class="lineup__list is-visit">
    <li class="lineup__status is-confirmed">Confirmed Lineup</li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jayson Tatum">Tatum</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jaylen Brown">Brown</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Derrick White">White</a><span class="lineup__pos">PG</span><span class="lineup__inj">GTD</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jrue Holiday">Holiday</a>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Al Horford">Horford</a><span class="lineup__pos">C</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Payton Pritchard">Pritchard</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__player has-injury-status" title="Very Unlikely To Play">
      <a title="Kristaps Porzingis">Porzingis</a><span class="lineup__pos">C</span><span class="lineup__inj">Out</span>
    </li>
    <li class="lineup__player" title="Toss Up To Play">
      <a title="Neemias Queta">Queta</a><span class="lineup__pos">C</span>
    </li>
  </ul>
  <ul class="lineup__list is-home">
    <li class="lineup__status is-expected">Expected Lineup</li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Tyrese Maxey">Maxey</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="VJ Edgecombe">Edgecombe</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jaylen Brown">Brown</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Dean Wade">Wade</a><span class="lineup__pos">PF</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Joel Embiid">Embiid</a><span class="lineup__pos">C</span><span class="lineup__inj">Doubt</span>
    </li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Joel Embiid">Embiid</a><span class="lineup__pos">C</span><span class="lineup__inj">Doubt</span>
    </li>
    <li class="lineup__player has-injury-status hide is-ofs" title="Very Likely To Play">
      <a title="Ariel Hukporti">Hukporti</a><span class="lineup__pos">C</span><span class="lineup__inj">OFS</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Jabari Walker">Walker</a><span class="lineup__pos">F</span><span class="lineup__inj">Ques</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Unlikely To Play">
      <a title="Dominick Barlow">Barlow</a><span class="lineup__pos">C</span><span class="lineup__inj">Out</span>
    </li>
  </ul>
</div>
<div class="lineup is-nba">
  <a class="lineup__mteam is-visit">Trail Blazers</a>
  <a class="lineup__mteam is-home">Kings</a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Anfernee Simons">Simons</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Scoot Henderson">Henderson</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jerami Grant">Grant</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Deandre Ayton">Ayton</a><span class="lineup__pos">C</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Shaedon Sharpe">Sharpe</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player"><span class="lineup__pos">G</span></li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__no">None</li>
  </ul>
  <ul class="lineup__list is-home">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="De'Aaron Fox">Fox</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Domantas Sabonis">Sabonis</a><span class="lineup__pos">C</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Keegan Murray">Murray</a><span class="lineup__pos">PF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="DeMar DeRozan">DeRozan</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Malik Monk">Monk</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__player has-injury-status" title="Very Unlikely To Play">
      <a title="Kevin Huerter">Huerter</a><span class="lineup__pos">G</span><span class="lineup__inj">Out</span>
    </li>
  </ul>
</div>
<div class="lineup is-wnba">
  <a class="lineup__mteam is-visit white">Aces</a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="A'ja Wilson">Wilson</a><span class="lineup__pos">F</span>
    </li>
  </ul>
</div>
"""

WNBA_ONLY = """
<div class="lineup is-wnba">
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="A'ja Wilson">Wilson</a>
    </li>
  </ul>
</div>
"""


def test_snapshot_reads_confirmed_projected_and_may_not_play() -> None:
    payload = NBADailyLineups(html=PAGE).lineup_snapshot(FETCHED_AT)

    assert payload["source"] == "rotowire"
    assert payload["url"] == NBA_LINEUPS_URL
    assert payload["fetched_at"] == FETCHED_AT
    assert len(payload["games"]) == 2

    away = payload["games"][0]["away"]
    assert away["team"] == "BOS"
    assert away["status"] == "confirmed"
    assert away["lineup"] == [
        {"name": "Jayson Tatum", "position": "SF"},
        {"name": "Jaylen Brown", "position": "SG"},
        {"name": "Derrick White", "position": "PG"},
        {"name": "Jrue Holiday", "position": None},
        {"name": "Al Horford", "position": "C"},
        {"name": "Payton Pritchard", "position": "PG"},
    ]
    assert away["may_not_play"] == [
        {"name": "Kristaps Porzingis", "position": "C", "badge": "Out"},
        {"name": "Neemias Queta", "position": "C", "badge": None},
    ]
    assert all(set(player) == {"name", "position"} for player in away["lineup"])

    home = payload["games"][0]["home"]
    assert home["team"] == "PHI"
    assert home["status"] == "projected"
    assert [player["name"] for player in home["lineup"]] == [
        "Tyrese Maxey",
        "VJ Edgecombe",
        "Jaylen Brown",
        "Dean Wade",
        "Joel Embiid",
    ]
    assert "badge" not in home["lineup"][-1]
    assert home["may_not_play"] == [
        {"name": "Joel Embiid", "position": "C", "badge": "Doubt"},
        {"name": "Ariel Hukporti", "position": "C", "badge": "OFS"},
        {"name": "Jabari Walker", "position": "F", "badge": "Ques"},
        {"name": "Dominick Barlow", "position": "C", "badge": "Out"},
    ]


def test_missing_status_chip_is_projected_and_nickname_maps_to_abbr() -> None:
    game = NBADailyLineups(html=PAGE).lineup_snapshot(FETCHED_AT)["games"][1]
    assert game["away"]["team"] == "POR"
    assert game["away"]["status"] == "projected"
    assert [player["name"] for player in game["away"]["lineup"]] == [
        "Anfernee Simons",
        "Scoot Henderson",
        "Jerami Grant",
        "Deandre Ayton",
        "Shaedon Sharpe",
    ]
    assert game["away"]["may_not_play"] == []
    assert game["home"]["team"] == "SAC"
    assert game["home"]["status"] == "projected"
    assert game["home"]["may_not_play"] == [
        {"name": "Kevin Huerter", "position": "G", "badge": "Out"}
    ]


def test_confirmed_text_without_the_class_is_confirmed() -> None:
    html = """
    <div class="lineup is-nba">
      <a class="lineup__team is-visit"><span class="lineup__abbr">MIA</span></a>
      <ul class="lineup__list is-visit">
        <li class="lineup__status">Confirmed Lineup</li>
        <li class="lineup__player"><a title="Bam Adebayo">Bam</a><span class="lineup__pos">C</span></li>
      </ul>
      <ul class="lineup__list is-home"></ul>
    </div>
    """
    game = NBADailyLineups(html=html).lineup_snapshot(FETCHED_AT)["games"][0]
    assert game["away"]["status"] == "confirmed"
    assert game["away"]["lineup"] == [{"name": "Bam Adebayo", "position": "C"}]
    assert game["away"]["may_not_play"] == []
    assert game["home"]["team"] is None
    assert game["home"]["status"] == "projected"
    assert game["home"]["lineup"] == []
    assert game["home"]["may_not_play"] == []


def test_wnba_cards_and_empty_pages_produce_no_games() -> None:
    names = []
    for game in NBADailyLineups(html=PAGE).lineup_snapshot(FETCHED_AT)["games"]:
        for side in ("away", "home"):
            names.extend(player["name"] for player in game[side]["lineup"])
            names.extend(player["name"] for player in game[side]["may_not_play"])
    assert "A'ja Wilson" not in names

    empty = NBADailyLineups(html="<div></div>").lineup_snapshot(FETCHED_AT)
    assert empty == {
        "source": "rotowire",
        "url": NBA_LINEUPS_URL,
        "fetched_at": FETCHED_AT,
        "games": [],
    }
    assert NBADailyLineups(html=WNBA_ONLY).lineup_snapshot(FETCHED_AT)["games"] == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/scrapers/test_nba_rotowire_lineup.py -v`

Expected: FAIL with `AttributeError: 'NBADailyLineups' object has no attribute 'lineup_snapshot'`

- [ ] **Step 3: Write the minimal implementation**

In `src/scrapers/nba/nba_lineups.py`, add these functions immediately after `_class_tokens`:

```python
def _chip_status(item) -> str:
    tokens = _class_tokens(item.get("class"))
    text = item.get_text(strip=True)
    if "is-confirmed" in tokens or text == "Confirmed Lineup":
        return "confirmed"
    return "projected"


def _snapshot_player(item, *, with_badge: bool) -> dict | None:
    link = item.find("a")
    if link is None:
        return None
    name = (link.get("title") or link.get_text(strip=True) or "").strip()
    if not name:
        return None
    pos_el = item.find(class_="lineup__pos")
    position = pos_el.get_text(strip=True) if pos_el else ""
    player = {"name": name, "position": position or None}
    if with_badge:
        inj_el = item.find(class_="lineup__inj")
        badge = inj_el.get_text(strip=True) if inj_el else ""
        player["badge"] = badge or None
    return player
```

Add these methods at the end of class `NBADailyLineups`, after `getOutPlayers`:

```python
    def lineup_snapshot(self, fetched_at: str) -> dict:
        """Props payload for NBA cards already loaded in ``self.soup``.

        Does not fetch. WNBA cards are ignored, including on a page that has
        no NBA card. ``_matchup_cards`` is the wrong helper here: with no NBA
        card it falls back to every ``div.lineup``.
        """
        cards = self.soup.find_all("div", class_=_is_nba_lineup_card)
        return {
            "source": "rotowire",
            "url": self.url,
            "fetched_at": fetched_at,
            "games": [
                {
                    "away": self._side_snapshot(card, "away"),
                    "home": self._side_snapshot(card, "home"),
                }
                for card in cards
            ],
        }

    def _side_snapshot(self, matchup, side: str) -> dict:
        list_cls = (
            "lineup__list is-visit" if side == "away" else "lineup__list is-home"
        )
        lineup_list = matchup.find("ul", {"class": list_cls})
        nickname = self._side_nickname(matchup, side)
        status = "projected"
        lineup: list[dict] = []
        may_not_play: list[dict] = []
        after_title = False
        if lineup_list is not None:
            for item in lineup_list.find_all("li", recursive=False):
                tokens = _class_tokens(item.get("class"))
                if "lineup__status" in tokens:
                    status = _chip_status(item)
                    continue
                if "lineup__title" in tokens:
                    after_title = True
                    continue
                if "lineup__player" not in tokens:
                    continue
                player = _snapshot_player(item, with_badge=after_title)
                if player is None:
                    continue
                if after_title:
                    may_not_play.append(player)
                else:
                    lineup.append(player)
        return {
            "team": self._resolve_abbr(matchup, side, nickname),
            "status": status,
            "lineup": lineup,
            "may_not_play": may_not_play,
        }
```

Do not edit `getDict`, `_ordered_expected_starters`, `expected_starters_by_abbr`, or `posted_starting_flags`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scrapers/test_nba_rotowire_lineup.py tests/scrapers/test_nba_lineups.py -v`

Expected: PASS, including the existing starter-flag tests.

- [ ] **Step 5: Commit**

```bash
git add src/scrapers/nba/nba_lineups.py tests/scrapers/test_nba_rotowire_lineup.py
git commit -m "$(cat <<'EOF'
Add a RotoWire lineup snapshot parsed from the page already in memory.

EOF
)"
```

---

### Task 2: Write one timestamped JSON file from a single fetch

**Files:**
- Create: `src/scrapers/nba/nba_rotowire_lineup.py`
- Modify: `tests/scrapers/test_nba_rotowire_lineup.py` (append the writer tests)

**Interfaces:**
- Consumes: `NBADailyLineups.lineup_snapshot(fetched_at: str) -> dict` from Task 1. `ensure_repo_on_path` from `src.scrapers.nba.paths`.
- Produces:
  - `OUTPUT_TZ = ZoneInfo("America/Los_Angeles")`
  - `snapshot_filename(moment: datetime) -> str`
  - `fetched_at_text(moment: datetime) -> str`
  - `output_directory(root: Path | None = None) -> Path`
  - `write_snapshot(payload: dict, directory: Path, moment: datetime) -> Path`
  - `run(moment: datetime, directory: Path, fetch: Callable[[], NBADailyLineups] | None = None) -> int`
  - `main() -> int` calls `run(datetime.now(OUTPUT_TZ), output_directory())`

- [ ] **Step 1: Write the failing test**

Add the imports next to the imports already at the top of `tests/scrapers/test_nba_rotowire_lineup.py`. Append the tests at the bottom of that file:

```python
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.scrapers.nba.nba_rotowire_lineup import (
    fetched_at_text,
    output_directory,
    run,
    snapshot_filename,
    write_snapshot,
)

PACIFIC = ZoneInfo("America/Los_Angeles")
MOMENT = datetime(2026, 10, 24, 22, 35, 12, tzinfo=PACIFIC)


def test_filename_and_fetched_at_use_pacific_time() -> None:
    assert snapshot_filename(MOMENT) == "rotowire_nba_2026-10-24_223512.json"
    assert fetched_at_text(MOMENT) == "2026-10-24T22:35:12-07:00"
    utc = datetime(2026, 10, 25, 5, 35, 12, tzinfo=ZoneInfo("UTC"))
    assert snapshot_filename(utc) == "rotowire_nba_2026-10-24_223512.json"
    assert fetched_at_text(utc) == "2026-10-24T22:35:12-07:00"


def test_output_directory_is_the_props_folder() -> None:
    assert output_directory().parts[-3:] == ("data", "props", "rotowire_nba_lineup")


def test_write_snapshot_renames_temp_into_place(tmp_path) -> None:
    older = tmp_path / "rotowire_nba_2026-10-24_220000.json"
    older.write_text("{}\n", encoding="utf-8")
    payload = NBADailyLineups(html="<div></div>").lineup_snapshot(fetched_at_text(MOMENT))
    path = write_snapshot(payload, tmp_path, MOMENT)
    assert path == tmp_path / "rotowire_nba_2026-10-24_223512.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert older.read_text(encoding="utf-8") == "{}\n"
    assert set(tmp_path.iterdir()) == {older, path}


def test_write_snapshot_removes_temp_when_rename_fails(tmp_path, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "src.scrapers.nba.nba_rotowire_lineup.os.replace",
        boom,
    )
    payload = {"source": "rotowire", "url": NBA_LINEUPS_URL, "fetched_at": FETCHED_AT, "games": []}
    with pytest.raises(OSError, match="disk full"):
        write_snapshot(payload, tmp_path, MOMENT)
    assert list(tmp_path.iterdir()) == []


def test_run_fetches_once_and_writes_the_payload(tmp_path) -> None:
    calls = {"n": 0}

    class FakeLineups:
        def lineup_snapshot(self, fetched_at: str) -> dict:
            return {
                "source": "rotowire",
                "url": NBA_LINEUPS_URL,
                "fetched_at": fetched_at,
                "games": [],
            }

    def fetch():
        calls["n"] += 1
        return FakeLineups()

    assert run(MOMENT, tmp_path, fetch) == 0
    assert calls["n"] == 1
    written = json.loads(
        (tmp_path / "rotowire_nba_2026-10-24_223512.json").read_text(encoding="utf-8")
    )
    assert written["fetched_at"] == "2026-10-24T22:35:12-07:00"
    assert written["games"] == []
    assert not (tmp_path / ".rotowire_nba_2026-10-24_223512.json.tmp").exists()


def test_run_writes_nothing_when_the_fetch_fails(tmp_path) -> None:
    def fetch():
        raise RuntimeError("rotowire 503")

    assert run(MOMENT, tmp_path, fetch) == 1
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/scrapers/test_nba_rotowire_lineup.py::test_filename_and_fetched_at_use_pacific_time -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.scrapers.nba.nba_rotowire_lineup'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/scrapers/nba/nba_rotowire_lineup.py`:

```python
"""Write one RotoWire NBA lineup snapshot.

Usage:
    python -m src.scrapers.nba.nba_rotowire_lineup
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.scrapers.nba.nba_lineups import NBADailyLineups
from src.scrapers.nba.paths import ensure_repo_on_path

OUTPUT_TZ = ZoneInfo("America/Los_Angeles")


def snapshot_filename(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    pacific = moment.astimezone(OUTPUT_TZ)
    return pacific.strftime("rotowire_nba_%Y-%m-%d_%H%M%S.json")


def fetched_at_text(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    return moment.astimezone(OUTPUT_TZ).isoformat(timespec="seconds")


def output_directory(root: Path | None = None) -> Path:
    base = root if root is not None else ensure_repo_on_path(__file__)
    return base / "data" / "props" / "rotowire_nba_lineup"


def write_snapshot(payload: dict, directory: Path, moment: datetime) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = snapshot_filename(moment)
    final = directory / filename
    temporary = directory / f".{filename}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return final


def run(
    moment: datetime,
    directory: Path,
    fetch: Callable[[], NBADailyLineups] | None = None,
) -> int:
    """Fetch the lineup page once and write one snapshot. 0 on success, 1 on failure."""
    loader = fetch or NBADailyLineups
    try:
        lineups = loader()
        payload = lineups.lineup_snapshot(fetched_at_text(moment))
        write_snapshot(payload, directory, moment)
    except Exception as exc:
        print(f"nba_rotowire_lineup failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    return run(datetime.now(OUTPUT_TZ), output_directory())


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scrapers/test_nba_rotowire_lineup.py tests/scrapers/test_nba_lineups.py -v`

Expected: PASS. `test_run_writes_nothing_when_the_fetch_fails` prints `nba_rotowire_lineup failed: rotowire 503` to stderr. That line is the failure report, not a test failure.

- [ ] **Step 5: Commit**

```bash
git add src/scrapers/nba/nba_rotowire_lineup.py tests/scrapers/test_nba_rotowire_lineup.py
git commit -m "$(cat <<'EOF'
Write each RotoWire lineup fetch to a timestamped props JSON file.

EOF
)"
```

---

### Task 3: Run the lineup job with the NBA odds scrapers

**Files:**
- Modify: `src/scrapers/run_all_odds.py` (`_league_jobs`, about lines 52–74)
- Modify: `tests/scrapers/test_run_all_odds.py` (`NBA_NAMES`, the default-order assertions, `test_classify_job_pools`)

**Interfaces:**
- Consumes: module path `src.scrapers.nba.nba_rotowire_lineup` and `main()` from Task 2. The runner invokes it as `python -m src.scrapers.nba.nba_rotowire_lineup` through the existing `ScraperJob` path. No new `run_all_odds` function.
- Produces: job name `nba_rotowire_lineup`, league `nba`, module `src.scrapers.nba.nba_rotowire_lineup`, `env is None`, pool `http`. Absent from MLB. On a full run it is the last name before `mlb_prizepick` and `nba_prizepick`.

- [ ] **Step 1: Write the failing test**

In `tests/scrapers/test_run_all_odds.py`, change `NBA_NAMES` to:

```python
NBA_NAMES = [
    "nba_novig",
    "nba_prophetx",
    "nba_underdog",
    "nba_fanduel",
    "nba_draftking",
    "nba_pinnacle",
    "nba_rotowire_lineup",
    "nba_prizepick",
]
```

In `test_resolve_jobs_default_runs_mlb_then_nba_with_prizepicks_last`, after the `nba_draftking` assertion, add:

```python
    assert by_name["nba_rotowire_lineup"].module == (
        "src.scrapers.nba.nba_rotowire_lineup"
    )
    assert by_name["nba_rotowire_lineup"].league == "nba"
    assert by_name["nba_rotowire_lineup"].env is None
    names = [j.name for j in jobs]
    assert names[names.index("mlb_prizepick") - 1] == "nba_rotowire_lineup"
    assert "nba_rotowire_lineup" not in [j.name for j in resolve_jobs(league="mlb")]
```

In `test_classify_job_pools`, add:

```python
    assert pools["nba_rotowire_lineup"] == "http"
```

Leave `test_resolve_jobs_default_runs_mlb_then_nba_with_prizepicks_last`'s existing equality

```python
    assert [j.name for j in jobs] == (
        MLB_NAMES[:-1] + NBA_NAMES[:-1] + ["mlb_prizepick", "nba_prizepick"]
    )
```

as it is. With `nba_rotowire_lineup` placed before `nba_prizepick` in `NBA_NAMES`, that equality is the order check. The new index assertion makes the "last name before PrizePicks" requirement obvious when it fails.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/scrapers/test_run_all_odds.py::test_resolve_jobs_default_runs_mlb_then_nba_with_prizepicks_last tests/scrapers/test_run_all_odds.py::test_classify_job_pools -v`

Expected: FAIL. The default-order assertion does not contain `nba_rotowire_lineup`, or `by_name["nba_rotowire_lineup"]` raises `KeyError`.

- [ ] **Step 3: Write the minimal implementation**

Replace `_league_jobs` in `src/scrapers/run_all_odds.py` with:

```python
def _league_jobs(league: League) -> tuple[ScraperJob, ...]:
    """One book order per league. PrizePicks stays last inside the league."""
    package = f"src.scrapers.{league}"
    jobs = [
        ScraperJob(f"{league}_novig", league, f"{package}.{league}_novig"),
        ScraperJob(f"{league}_prophetx", league, f"{package}.{league}_prophetx"),
        ScraperJob(f"{league}_underdog", league, f"{package}.{league}_underdog"),
        ScraperJob(f"{league}_fanduel", league, f"{package}.{league}_fanduel"),
        ScraperJob(f"{league}_draftking", league, f"{package}.{league}_draftking"),
        ScraperJob(
            f"{league}_pinnacle",
            league,
            f"{package}.{league}_pinnacle",
            env={"PINNACLE_LEAGUES": league},
        ),
    ]
    if league == "nba":
        jobs.append(
            ScraperJob(
                "nba_rotowire_lineup",
                "nba",
                "src.scrapers.nba.nba_rotowire_lineup",
            )
        )
    jobs.append(
        ScraperJob(
            f"{league}_prizepick",
            league,
            f"{package}.{league}_prizepick",
            # HTTP attempt first; browser fallback is queued separately on exit 3.
            env={"PRIZEPICKS_NO_BROWSER": "1"},
        )
    )
    return tuple(jobs)
```

Do not change `resolve_jobs`, `classify_pool`, or `--fail-fast`. The new job has no `env`, so `classify_pool` already returns `"http"`. `_prizepicks_last` already moves only names ending in `prizepick`, so this job stays immediately before the two PrizePicks jobs on a full run.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scrapers/test_run_all_odds.py tests/scrapers/test_nba_rotowire_lineup.py tests/scrapers/test_nba_lineups.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scrapers/run_all_odds.py tests/scrapers/test_run_all_odds.py
git commit -m "$(cat <<'EOF'
Scrape RotoWire lineups once whenever the NBA odds scrapers run.

EOF
)"
```
