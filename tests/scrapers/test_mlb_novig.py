"""Novig MLB fetch uses public REST events + allowlisted EventMarkets_Query."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.scrapers.mlb import mlb_novig as novig
from src.scrapers.mlb.mlb_novig import (
    extract_props,
    fetch_event_markets,
    fetch_mlb_events,
    graphql,
    normalize_event,
)


def test_extract_props_uses_player_full_name() -> None:
    markets = [
        {
            "id": "m1",
            "type": "HITS",
            "strike": 1.5,
            "player": {"id": "p1", "full_name": "Jose Trevino"},
            "outcomes": [
                {"description": "Over 1.5", "available": 0.45},
                {"description": "Under 1.5", "available": 0.55},
            ],
        }
    ]
    rows = extract_props(markets)
    assert len(rows) == 1
    assert rows[0]["player"] == "Jose Trevino"
    assert rows[0]["stat"] == "hits"
    assert rows[0]["line"] == 1.5


def test_events_from_trading_page_reads_game_cards() -> None:
    payload = {
        "target": "MLB",
        "sections": [
            {
                "title": "Games",
                "content": {
                    "type": "components",
                    "components": [
                        {
                            "type": "game_event_card",
                            "eventId": "evt-1",
                            "scheduledStart": "2026-09-22T23:15:00.000Z",
                            "league": "MLB",
                            "isLive": False,
                            "homeTeam": {"id": "h1", "name": "Atlanta Braves"},
                            "awayTeam": {"id": "a1", "name": "Cincinnati Reds"},
                            "scoreboard": {
                                "status": "OPEN_PREGAME",
                                "title": "Cincinnati Reds @ Atlanta Braves",
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
    assert event["description"] == "Cincinnati Reds @ Atlanta Braves"
    assert event["status"] == "OPEN_PREGAME"
    assert event["game"]["homeTeam"]["name"] == "Atlanta Braves"
    assert event["game"]["awayTeam"]["name"] == "Cincinnati Reds"
    assert event["game"]["scheduled_start"] == "2026-09-22T23:15:00.000Z"
    base = normalize_event(event)
    assert base["event_id"] == "evt-1"
    assert base["competitors"][0]["name"] == "Atlanta Braves"


def test_fetch_mlb_events_uses_trading_page_not_blocked_graphql() -> None:
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
                            "scheduledStart": "2026-09-22T23:15:00.000Z",
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

    events = fetch_mlb_events(session)

    assert [e["id"] for e in events] == ["evt-rest"]
    session.get.assert_called()
    url = session.get.call_args.args[0]
    assert "/nbx/v1/trading/MLB/page" in url
    session.post.assert_not_called()


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
                            "type": "HOME_RUNS",
                            "strike": 0.5,
                            "player": {"id": "p1", "full_name": "Jose Trevino"},
                            "outcomes": [
                                {"description": "Over 0.5", "available": 0.12},
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
    assert "GetEventMarkets" not in payload["query"]
    assert "GetMlbEvents" not in payload["query"]
    assert payload["variables"]["eventId"] == "evt-1"
