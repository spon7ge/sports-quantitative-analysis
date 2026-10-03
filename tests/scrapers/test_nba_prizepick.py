"""NBA PrizePicks scraper defaults and league routing."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.odds.load_snapshots import _prizepicks_table
from src.odds.quote_specs import QUOTE_SPECS
from src.scrapers.nba.nba_prizepick import (
    _CLEARANCE_PROBE_LEAGUE_ID,
    _DEFAULT_OUTPUT_DIR,
    DEFAULT_LEAGUE_NAMES,
    DEFAULT_LEAGUES,
    extract_projections,
    resolve_output_path,
)
from src.scrapers.nba.paths import repo_root


def test_nba_prizepick_defaults() -> None:
    root = repo_root()
    assert DEFAULT_LEAGUES == (("NBA", 7),)
    assert DEFAULT_LEAGUE_NAMES == ("NBA",)
    assert _CLEARANCE_PROBE_LEAGUE_ID == 3
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "prizepicks")


def test_nba_prizepicks_table_and_quote_spec() -> None:
    assert _prizepicks_table("nba") == "nba_prizepicks"
    assert _prizepicks_table("mlb") == "mlb_prizepicks"
    assert _prizepicks_table("wnba") == "wnba_prizepicks"
    assert QUOTE_SPECS["nba_prizepicks"] is QUOTE_SPECS["mlb_prizepicks"]


def test_extract_projections_stamps_nba() -> None:
    payload = {
        "included": [
            {
                "type": "new_player",
                "id": "p1",
                "attributes": {"name": "Nikola Jokic"},
            }
        ],
        "data": [
            {
                "type": "projection",
                "attributes": {
                    "line_score": 27.5,
                    "stat_type": "Points",
                    "odds_type": "standard",
                    "updated_at": "2026-10-03T00:00:00Z",
                },
                "relationships": {"new_player": {"data": {"id": "p1"}}},
            }
        ],
    }

    rows = extract_projections(payload)
    assert len(rows) == 1
    assert rows[0].player == "Nikola Jokic"
    assert rows[0].stat_type == "Points"
    assert rows[0].line_score == 27.5
    assert rows[0].league == "NBA"


def test_resolve_output_path_uses_nba_slug(monkeypatch) -> None:
    monkeypatch.delenv("PRIZEPICKS_OUTPUT", raising=False)
    when = datetime(2026, 10, 3, 14, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    path = resolve_output_path("NBA", when=when)
    assert path.endswith("prizepicks_nba_2026-10-03_143000.json")
    assert path.startswith(_DEFAULT_OUTPUT_DIR)


def test_nba_prizepick_imports_as_loose_script() -> None:
    """`python nba_prizepick.py` from src/scrapers/nba must resolve paths.py."""
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_prizepick.py', run_name='not_main'); "
        "assert ns.get('_ROOT'); "
        "assert ns['DEFAULT_LEAGUES'] == (('NBA', 7),)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
