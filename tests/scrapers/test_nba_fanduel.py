"""FanDuel NBA event discovery, prop parsing, and league routing."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.odds.load_snapshots import _fanduel_table
from src.odds.quote_specs import QUOTE_SPECS
from src.odds.snapshot_rows import fanduel_picks_to_rows
from src.scrapers.arbitrage import quotes_from_payload
from src.scrapers.nba.nba_fanduel import (
    _DEFAULT_CONFIG,
    _DEFAULT_OUTPUT_DIR,
    Event,
    ExportData,
    extract_event_picks,
    extract_game_lines,
    fetch_events,
    is_game_market,
    match_stat,
    resolve_output_path,
    save_picks,
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
                    "openDate": "2099-10-03T23:30:00.000Z",
                },
                "2": {
                    "eventId": 34800001,
                    "name": "Cincinnati Reds @ Atlanta Braves",
                    "competitionId": 11196870,
                    "openDate": "2099-10-03T23:20:00.000Z",
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
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "fanduel" / "nba")
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


def _priced(name: str, handicap: float, american: int) -> dict:
    return {
        "runnerName": name,
        "handicap": handicap,
        "runnerStatus": "ACTIVE",
        "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": american}},
    }


def _game() -> Event:
    return Event(
        event_id="36244906",
        name="Minnesota Timberwolves @ Cleveland Cavaliers",
        start_time="2026-10-26T23:13:00.000Z",
        competition="NBA",
    )


def test_extract_game_lines_keeps_full_game_moneyline_spread_and_total() -> None:
    event = _game()
    markets = [
        {
            "marketName": "Moneyline",
            "marketType": "MONEY_LINE",
            "marketStatus": "OPEN",
            "eventId": event.event_id,
            "runners": [
                _priced("Minnesota Timberwolves", 0, -285),
                _priced("Cleveland Cavaliers", 0, 230),
            ],
        },
        {
            "marketName": "Spread",
            "marketType": "MATCH_HANDICAP_(2-WAY)",
            "marketStatus": "OPEN",
            "eventId": event.event_id,
            "runners": [
                _priced("Minnesota Timberwolves", -6.5, -110),
                _priced("Cleveland Cavaliers", 6.5, -110),
            ],
        },
        {
            "marketName": "Spread",
            "marketType": "MATCH_HANDICAP_(2-WAY)",
            "marketStatus": "OPEN",
            "eventId": event.event_id,
            "runners": [
                _priced("Minnesota Timberwolves", -10.5, -250),
                _priced("Cleveland Cavaliers", 10.5, 190),
            ],
        },
        {
            "marketName": "Total Points",
            "marketType": "TOTAL_POINTS_(OVER/UNDER)",
            "marketStatus": "OPEN",
            "eventId": event.event_id,
            "runners": [
                _priced("Over", 230.5, -118),
                _priced("Under", 230.5, -112),
            ],
        },
        {
            "marketName": "1st Half Total Points",
            "marketType": "TOTAL_POINTS_(OVER/UNDER)",
            "marketStatus": "OPEN",
            "eventId": event.event_id,
            "runners": [
                _priced("Over", 115.5, -110),
                _priced("Under", 115.5, -110),
            ],
        },
        {
            "marketName": "Alternate Spread",
            "marketType": "MATCH_HANDICAP_(2-WAY)",
            "marketStatus": "OPEN",
            "eventId": event.event_id,
            "runners": [
                _priced("Minnesota Timberwolves", -3.5, 140),
                _priced("Cleveland Cavaliers", 3.5, -170),
            ],
        },
        {
            "marketName": "26-27 NBA Finals Winner",
            "marketType": "NBA_FINALS_WINNER",
            "marketStatus": "OPEN",
            "eventId": "34409523",
            "runners": [_priced("Oklahoma City Thunder", 0, 390)],
        },
        {
            "marketName": "Moneyline",
            "marketType": "MONEY_LINE",
            "marketStatus": "SUSPENDED",
            "eventId": event.event_id,
            "runners": [
                _priced("Minnesota Timberwolves", 0, 100),
                _priced("Cleveland Cavaliers", 0, 100),
            ],
        },
    ]

    lines = extract_game_lines(markets, [event])
    saved = [line.to_dict() for line in lines]
    assert saved == [
        {
            "market": "moneyline",
            "event_name": event.name,
            "start_time": event.start_time,
            "side": "Minnesota Timberwolves",
            "line": None,
            "american": -285,
        },
        {
            "market": "moneyline",
            "event_name": event.name,
            "start_time": event.start_time,
            "side": "Cleveland Cavaliers",
            "line": None,
            "american": 230,
        },
        {
            "market": "spread",
            "event_name": event.name,
            "start_time": event.start_time,
            "side": "Minnesota Timberwolves",
            "line": -6.5,
            "american": -110,
        },
        {
            "market": "spread",
            "event_name": event.name,
            "start_time": event.start_time,
            "side": "Cleveland Cavaliers",
            "line": 6.5,
            "american": -110,
        },
        {
            "market": "total",
            "event_name": event.name,
            "start_time": event.start_time,
            "side": "over",
            "line": 230.5,
            "american": -118,
        },
        {
            "market": "total",
            "event_name": event.name,
            "start_time": event.start_time,
            "side": "under",
            "line": 230.5,
            "american": -112,
        },
    ]
    quotes = quotes_from_payload("fanduel", "nba", {"picks": [], "game_lines": saved})
    assert {quote.kind for quote in quotes} == {"moneyline", "spread", "total"}
    assert {quote.line for quote in quotes if quote.kind == "spread"} == {-6.5, 6.5}


def test_save_picks_writes_game_lines_beside_props(tmp_path: Path) -> None:
    path = tmp_path / "fanduel_nba.json"
    event = _game()
    lines = extract_game_lines(
        [
            {
                "marketName": "Total Points",
                "marketType": "TOTAL_POINTS_(OVER/UNDER)",
                "marketStatus": "OPEN",
                "eventId": event.event_id,
                "runners": [
                    _priced("Over", 220.5, -105),
                    _priced("Under", 220.5, -115),
                ],
            }
        ],
        [event],
    )
    save_picks([], str(path), game_lines=lines)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["picks"] == []
    assert payload["count"] == 0
    assert payload["game_lines"] == [line.to_dict() for line in lines]
    assert ExportData(picks=[], game_lines=[]).to_dict()["game_lines"] == []


def test_resolve_output_path_uses_nba_slug(monkeypatch) -> None:
    monkeypatch.delenv("FANDUEL_OUTPUT", raising=False)
    when_path = resolve_output_path("NBA")
    assert Path(when_path).parent == Path(_DEFAULT_OUTPUT_DIR)
    assert "fanduel_nba_" in Path(when_path).name

    other = resolve_output_path("MLB")
    assert Path(other).parent.name == "mlb"
    assert Path(other).parent.parent == Path(_DEFAULT_OUTPUT_DIR).parent


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
