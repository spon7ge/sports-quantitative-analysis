"""FanDuel MLB event discovery and repo-root setup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.scrapers.mlb.mlb_fanduel import (
    _DEFAULT_OUTPUT_DIR,
    _ROOT,
    fetch_events,
)
from src.scrapers.mlb.paths import repo_root


def _page_payload() -> dict:
    """Shape of the live content-managed-page payload: names live on competitions."""
    return {
        "attachments": {
            "competitions": {
                "11196870": {
                    "name": "MLB",
                    "competitionId": 11196870,
                    "eventTypeId": 7511,
                },
                "11195275": {
                    "name": "MLB - Player Markets",
                    "competitionId": 11195275,
                    "eventTypeId": 7511,
                },
                "12765173": {
                    "name": "MLB Futures",
                    "competitionId": 12765173,
                    "eventTypeId": 7511,
                },
            },
            "events": {
                "1": {
                    "eventId": 34800001,
                    "name": "Cincinnati Reds @ Atlanta Braves",
                    "competitionId": 11196870,
                    "openDate": "2026-09-22T23:20:00.000Z",
                },
                "2": {
                    "eventId": 28197722,
                    "name": "MLB Player Markets",
                    "competitionId": 11195275,
                    "openDate": "2099-01-01T00:00:00.000Z",
                },
                "3": {
                    "eventId": 34821316,
                    "name": "MLB Futures",
                    "competitionId": 12765173,
                    "openDate": "2099-01-01T00:00:00.000Z",
                },
            },
            "markets": {},
        }
    }


def test_fanduel_root_is_repo_root_not_src() -> None:
    root = repo_root()
    assert _ROOT == str(root)
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "fanduel")


def test_fetch_events_joins_competition_name_from_attachments() -> None:
    cfg = {
        "fd_region": "ny",
        "fd_page_id": "mlb",
        "fd_api_key": "test",
        "fd_competition_names": ["MLB"],
        "fd_timeout": 30,
        "fd_max_retries": 1,
    }
    with patch("src.scrapers.mlb.mlb_fanduel.fetch_json", return_value=_page_payload()):
        events = fetch_events(MagicMock(), cfg)

    assert [e.event_id for e in events] == ["34800001"]
    assert events[0].competition == "MLB"
    assert events[0].name == "Cincinnati Reds @ Atlanta Braves"


def test_fetch_events_keeps_legacy_competition_name_on_event() -> None:
    payload = {
        "attachments": {
            "competitions": {},
            "events": {
                "1": {
                    "eventId": 1,
                    "name": "Yankees @ Red Sox",
                    "competitionName": "MLB",
                    "openDate": "2026-09-22T23:00:00.000Z",
                }
            },
            "markets": {},
        }
    }
    cfg = {
        "fd_region": "ny",
        "fd_page_id": "mlb",
        "fd_api_key": "test",
        "fd_competition_names": ["MLB"],
        "fd_timeout": 30,
        "fd_max_retries": 1,
    }
    with patch("src.scrapers.mlb.mlb_fanduel.fetch_json", return_value=payload):
        events = fetch_events(MagicMock(), cfg)

    assert [e.event_id for e in events] == ["1"]
    assert events[0].competition == "MLB"


def test_parse_runner_selection_reads_trailing_over_under() -> None:
    from src.scrapers.mlb.mlb_fanduel import parse_runner_selection

    assert parse_runner_selection("Over", 4.5) == ("over", 4.5)
    assert parse_runner_selection("Under", 4.5) == ("under", 4.5)
    assert parse_runner_selection("Carlos Rodon Over", 4.5) == ("over", 4.5)
    assert parse_runner_selection("Carlos Rodon Under", 4.5) == ("under", 4.5)


def test_extract_event_picks_keeps_player_name_and_side() -> None:
    from src.scrapers.mlb.mlb_fanduel import Event, extract_event_picks

    event = Event(
        event_id="1",
        name="Tampa Bay Rays @ New York Yankees",
        start_time="2026-09-22T17:06:00.000Z",
        competition="MLB",
    )
    markets = [
        {
            "marketName": "Carlos Rodon - Strikeouts",
            "marketStatus": "OPEN",
            "runners": [
                {
                    "runnerName": "Carlos Rodon Over",
                    "handicap": 4.5,
                    "winRunnerOdds": {
                        "americanDisplayOdds": {"americanOddsInt": -154},
                        "trueOdds": {"decimalOdds": {"decimalOdds": 1.6494}},
                    },
                },
                {
                    "runnerName": "Carlos Rodon Under",
                    "handicap": 4.5,
                    "winRunnerOdds": {
                        "americanDisplayOdds": {"americanOddsInt": 120},
                        "trueOdds": {"decimalOdds": {"decimalOdds": 2.2}},
                    },
                },
            ],
        }
    ]
    counters = {
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
    picks = extract_event_picks(
        markets, event, {}, "2026-09-22T04:00:00Z", "MLB", counters, set()
    )
    assert {(p.full_name, p.choice, p.stat_value, p.american_price) for p in picks} == {
        ("Carlos Rodon", "over", 4.5, -154),
        ("Carlos Rodon", "under", 4.5, 120),
    }
