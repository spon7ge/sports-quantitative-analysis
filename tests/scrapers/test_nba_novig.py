"""Novig NBA fetch uses public REST events + allowlisted EventMarkets_Query."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from src.odds.load_snapshots import _novig_props_table, _novig_team_table
from src.odds.quote_specs import QUOTE_SPECS
from src.scrapers.nba import nba_novig as novig
from src.scrapers.nba.nba_novig import (
    PROP_TYPE_TO_STAT,
    TRADING_PAGE_URL,
    _DEFAULT_OUTPUT_DIR,
    extract_props,
    extract_team_markets,
    fetch_event_markets,
    fetch_nba_events,
    graphql,
    normalize_event,
    pick_main_spread,
)
from src.scrapers.nba.paths import repo_root


def test_nba_novig_targets_nba_page_and_dir() -> None:
    root = repo_root()
    assert TRADING_PAGE_URL.endswith("/nbx/v1/trading/NBA/page")
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "novig" / "nba")
    assert PROP_TYPE_TO_STAT["POINTS"] == "points"
    assert PROP_TYPE_TO_STAT["THREE_POINTERS_MADE"] == "three_pointers_made"
    assert PROP_TYPE_TO_STAT["POINTS_REBOUNDS_ASSISTS"] == "points_rebounds_assists"
    assert "HITS" not in PROP_TYPE_TO_STAT


def test_nba_novig_tables_and_quote_specs() -> None:
    assert _novig_props_table("nba") == "nba_novig"
    assert _novig_team_table("nba") == "nba_novig_team"
    assert _novig_props_table("mlb") == "mlb_novig"
    assert _novig_props_table("wnba") == "wnba_novig"
    assert QUOTE_SPECS["nba_novig"] is QUOTE_SPECS["mlb_novig"]
    assert QUOTE_SPECS["nba_novig_team"] is QUOTE_SPECS["mlb_novig_team"]


def test_extract_props_uses_player_full_name() -> None:
    markets = [
        {
            "id": "m1",
            "type": "POINTS",
            "strike": 27.5,
            "player": {"id": "p1", "full_name": "Nikola Jokic"},
            "outcomes": [
                {"description": "Over 27.5", "available": 0.45},
                {"description": "Under 27.5", "available": 0.55},
            ],
        },
        {
            "id": "m2",
            "type": "HITS",
            "strike": 1.5,
            "player": {"id": "p2", "full_name": "Shohei Ohtani"},
            "outcomes": [
                {"description": "Over 1.5", "available": 0.5},
                {"description": "Under 1.5", "available": 0.5},
            ],
        },
    ]
    rows = extract_props(markets)
    assert len(rows) == 1
    assert rows[0]["player"] == "Nikola Jokic"
    assert rows[0]["stat"] == "points"
    assert rows[0]["line"] == 27.5


def test_main_spread_is_the_even_line_not_run_line() -> None:
    markets = [
        {
            "type": "SPREAD",
            "strike": 1.5,
            "outcomes": [
                {"description": "Lakers -1.5", "available": 0.2},
                {"description": "Celtics +1.5", "available": 0.8},
            ],
        },
        {
            "type": "SPREAD",
            "strike": -5.5,
            "outcomes": [
                {"description": "Lakers -5.5", "available": 0.49},
                {"description": "Celtics +5.5", "available": 0.51},
            ],
        },
    ]
    main = pick_main_spread(markets)
    assert main is not None
    assert main["strike"] == -5.5
    assert "spread" in extract_team_markets(markets)
    assert "run_line" not in extract_team_markets(markets)


def test_events_from_trading_page_reads_game_cards() -> None:
    payload = {
        "target": "NBA",
        "sections": [
            {
                "title": "Games",
                "content": {
                    "type": "components",
                    "components": [
                        {
                            "type": "game_event_card",
                            "eventId": "evt-1",
                            "scheduledStart": "2026-10-03T23:15:00.000Z",
                            "league": "NBA",
                            "isLive": False,
                            "homeTeam": {"id": "h1", "name": "Boston Celtics"},
                            "awayTeam": {"id": "a1", "name": "Los Angeles Lakers"},
                            "scoreboard": {
                                "status": "OPEN_PREGAME",
                                "title": "Los Angeles Lakers @ Boston Celtics",
                            },
                        },
                        {"type": "featured_parlay_carousel", "cards": []},
                    ],
                },
            }
        ],
    }
    events = novig.events_from_trading_page(payload)
    assert len(events) == 1
    event = events[0]
    assert event["id"] == "evt-1"
    assert event["description"] == "Los Angeles Lakers @ Boston Celtics"
    assert event["status"] == "OPEN_PREGAME"
    assert event["game"]["homeTeam"]["name"] == "Boston Celtics"
    assert event["game"]["league"] == "NBA"
    base = normalize_event(event)
    assert base["event_id"] == "evt-1"
    assert base["competitors"][0]["name"] == "Boston Celtics"


def test_fetch_nba_events_uses_trading_page() -> None:
    session = MagicMock()
    trading = MagicMock()
    trading.raise_for_status = MagicMock()
    trading.json.return_value = {
        "sections": [
            {
                "content": {
                    "components": [
                        {
                            "type": "game_event_card",
                            "eventId": "evt-rest",
                            "scheduledStart": "2026-10-03T23:15:00.000Z",
                            "isLive": True,
                            "homeTeam": {"id": "h", "name": "Home"},
                            "awayTeam": {"id": "a", "name": "Away"},
                            "scoreboard": {
                                "status": "OPEN_INGAME",
                                "title": "Away @ Home",
                            },
                        }
                    ]
                }
            }
        ]
    }
    session.get.return_value = trading

    events = fetch_nba_events(session)

    assert [e["id"] for e in events] == ["evt-rest"]
    url = session.get.call_args.args[0]
    assert "/nbx/v1/trading/NBA/page" in url
    session.post.assert_not_called()


def test_fetch_event_markets_posts_event_markets_query() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": {
            "event": [
                {
                    "id": "evt-1",
                    "markets": [
                        {
                            "id": "m1",
                            "type": "POINTS",
                            "strike": 27.5,
                            "player": {"id": "p1", "full_name": "Nikola Jokic"},
                            "outcomes": [
                                {"description": "Over 27.5", "available": 0.48},
                            ],
                        }
                    ],
                }
            ]
        }
    }
    session.post.return_value = resp

    markets = fetch_event_markets(session, "evt-1")
    assert len(markets) == 1
    payload = session.post.call_args.kwargs["json"]
    assert payload["operationName"] == "EventMarkets_Query"
    assert "EventMarkets_Query" in payload["query"]
    assert "GetNbaEvents" not in payload["query"]
    assert payload["variables"]["eventId"] == "evt-1"


def test_graphql_sends_allowlisted_event_markets_operation() -> None:
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": {"event": []}}
    session.post.return_value = resp

    graphql(
        session,
        "query EventMarkets_Query { __typename }",
        {"eventId": "abc"},
        operation_name="EventMarkets_Query",
    )

    payload = session.post.call_args.kwargs["json"]
    assert payload["operationName"] == "EventMarkets_Query"
    assert payload["variables"]["eventId"] == "abc"


def test_nba_novig_imports_as_loose_script() -> None:
    """`python nba_novig.py` from src/scrapers/nba must resolve paths.py."""
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_novig.py', run_name='not_main'); "
        "assert ns.get('_ROOT'); "
        "assert ns['TRADING_PAGE_URL'].endswith('/trading/NBA/page')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
