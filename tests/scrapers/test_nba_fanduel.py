"""FanDuel NBA event discovery, prop parsing, and league routing."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.odds.load_snapshots import _fanduel_table
from src.odds.quote_specs import QUOTE_SPECS
from src.odds.snapshot_rows import fanduel_picks_to_rows
from src.scrapers.nba.nba_fanduel import (
    _DEFAULT_CONFIG,
    _DEFAULT_OUTPUT_DIR,
    Event,
    extract_event_picks,
    fetch_events,
    is_game_market,
    match_stat,
    resolve_output_path,
    sport_to_league,
)
from src.scrapers.nba.paths import repo_root


def _page_payload() -> dict:
    return {
        "attachments": {
            "competitions": {
                "10547864": {
                    "name": "NBA",
                    "competitionId": 10547864,
                    "eventTypeId": 7522,
                },
                "11196870": {
                    "name": "MLB",
                    "competitionId": 11196870,
                    "eventTypeId": 7511,
                },
            },
            "events": {
                "1": {
                    "eventId": 34316771,
                    "name": "Denver Nuggets @ Oklahoma City Thunder",
                    "competitionId": 10547864,
                    "openDate": "2026-10-03T23:30:00.000Z",
                },
                "2": {
                    "eventId": 34800001,
                    "name": "Cincinnati Reds @ Atlanta Braves",
                    "competitionId": 11196870,
                    "openDate": "2026-10-03T23:20:00.000Z",
                },
            },
            "markets": {},
        }
    }


def _counters() -> dict[str, int]:
    return {
        "added": 0,
        "suspended_markets": 0,
        "suspended_runners": 0,
        "game_markets": 0,
        "exact_markets": 0,
        "no_stat": 0,
        "no_player": 0,
        "no_price": 0,
        "skipped_sport": 0,
    }


def test_nba_fanduel_defaults() -> None:
    root = repo_root()
    assert _DEFAULT_CONFIG["sport_allowlist"] == ["NBA"]
    assert _DEFAULT_CONFIG["fd_page_id"] == "nba"
    assert _DEFAULT_CONFIG["fd_competition_names"] == ["NBA"]
    assert "player-points" in _DEFAULT_CONFIG["fd_event_tabs"]
    assert "pitcher-props" not in _DEFAULT_CONFIG["fd_event_tabs"]
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "fanduel")
    assert sport_to_league("NBA") == "nba"
    assert sport_to_league("MLB") is None


def test_nba_fanduel_table_and_quote_spec() -> None:
    assert _fanduel_table("nba") == "nba_fanduel"
    assert _fanduel_table("mlb") == "mlb_fanduel"
    assert QUOTE_SPECS["nba_fanduel"] is QUOTE_SPECS["mlb_fanduel"]


def test_fetch_events_keeps_nba_and_drops_mlb() -> None:
    cfg = {
        "fd_region": "ny",
        "fd_page_id": "nba",
        "fd_api_key": "test",
        "fd_competition_names": ["NBA"],
        "fd_timeout": 30,
        "fd_max_retries": 1,
    }
    with patch("src.scrapers.nba.nba_fanduel.fetch_json", return_value=_page_payload()):
        events = fetch_events(MagicMock(), cfg)

    assert [e.event_id for e in events] == ["34316771"]
    assert events[0].competition == "NBA"
    assert events[0].name == "Denver Nuggets @ Oklahoma City Thunder"


def test_match_stat_prefers_combos_and_threes() -> None:
    assert match_stat("Pts + Reb + Ast") == "points_rebounds_assists"
    assert match_stat("Pts + Reb") == "points_rebounds"
    assert match_stat("Made Threes") == "three_pointers_made"
    assert match_stat("Points") == "points"
    assert match_stat("Strikeouts") is None


def test_game_total_is_not_a_player_prop() -> None:
    assert is_game_market("Total Points")
    assert is_game_market("Alternate Total Points")
    assert not is_game_market("Nikola Jokic - Total Points")


def test_extract_event_picks_keeps_points_and_drops_game_total() -> None:
    event = Event(
        event_id="1",
        name="Denver Nuggets @ Oklahoma City Thunder",
        start_time="2026-10-03T23:30:00.000Z",
        competition="NBA",
    )
    markets = [
        {
            "marketName": "Nikola Jokic - Points",
            "marketStatus": "OPEN",
            "runners": [
                {
                    "runnerName": "Over",
                    "handicap": 27.5,
                    "winRunnerOdds": {
                        "americanDisplayOdds": {"americanOddsInt": -115},
                    },
                },
                {
                    "runnerName": "Under",
                    "handicap": 27.5,
                    "winRunnerOdds": {
                        "americanDisplayOdds": {"americanOddsInt": -105},
                    },
                },
            ],
        },
        {
            "marketName": "Total Points",
            "marketStatus": "OPEN",
            "runners": [
                {
                    "runnerName": "Over",
                    "handicap": 220.5,
                    "winRunnerOdds": {
                        "americanDisplayOdds": {"americanOddsInt": -110},
                    },
                }
            ],
        },
    ]
    picks = extract_event_picks(
        markets, event, {}, "2026-10-03T20:00:00Z", "NBA", _counters(), set()
    )
    assert {(p.full_name, p.stat_name, p.choice, p.stat_value) for p in picks} == {
        ("Nikola Jokic", "points", "over", 27.5),
        ("Nikola Jokic", "points", "under", 27.5),
    }

    rows = fanduel_picks_to_rows(
        [p.to_dict() for p in picks],
        league="nba",
        scraped_at=datetime(2026, 10, 3, tzinfo=ZoneInfo("UTC")),
    )
    assert {row["market_type"] for row in rows} == {"player_points"}
    assert {row["league"] for row in rows} == {"nba"}


def test_resolve_output_path_uses_nba_slug(monkeypatch) -> None:
    monkeypatch.delenv("FANDUEL_OUTPUT", raising=False)
    when_path = resolve_output_path("NBA")
    assert when_path.startswith(_DEFAULT_OUTPUT_DIR)
    assert "fanduel_nba_" in when_path


def test_nba_fanduel_imports_as_loose_script() -> None:
    """`python nba_fanduel.py` from src/scrapers/nba must resolve paths.py."""
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_fanduel.py', run_name='not_main'); "
        "assert ns.get('_ROOT'); "
        "assert ns['_DEFAULT_CONFIG']['fd_page_id'] == 'nba'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
