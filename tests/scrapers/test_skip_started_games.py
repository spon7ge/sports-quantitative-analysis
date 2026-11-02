"""Started games are left out of odds scrapes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.scrapers.game_clock import game_has_started
from src.scrapers.mlb.mlb_draftking import (
    fetch_subcategory_offers as fetch_mlb_dk_offers,
)
from src.scrapers.mlb.mlb_fanduel import fetch_events as fetch_mlb_fd_events
from src.scrapers.mlb.mlb_novig import _fetch_graphql_snapshots as fetch_mlb_novig
from src.scrapers.mlb.mlb_pinnacle import (
    game_urls_from_league_matchups as mlb_pinnacle_urls,
)
from src.scrapers.mlb.mlb_prizepick import extract_projections as extract_mlb_pp
from src.scrapers.mlb.mlb_prophetx import pregame_events as mlb_px_pregame
from src.scrapers.mlb.mlb_underdog import extract_picks as extract_mlb_ud
from src.scrapers.nba.nba_draftking import Subcategory
from src.scrapers.nba.nba_draftking import (
    fetch_subcategory_offers as fetch_nba_dk_offers,
)
from src.scrapers.nba.nba_fanduel import events_from_page
from src.scrapers.nba.nba_novig import _fetch_graphql_snapshots as fetch_nba_novig
from src.scrapers.nba.nba_pinnacle import (
    game_urls_from_league_matchups as nba_pinnacle_urls,
)
from src.scrapers.nba.nba_prizepick import extract_projections as extract_nba_pp
from src.scrapers.nba.nba_prophetx import pregame_events as nba_px_pregame
from src.scrapers.nba.nba_underdog import extract_picks as extract_nba_ud

NOW = datetime(2026, 10, 27, 1, 30, tzinfo=timezone.utc)
PAST = "2026-10-26T23:00:00Z"
FUTURE = "2099-01-01T00:00:00Z"


def test_game_has_started_by_time_or_live_status() -> None:
    assert game_has_started(PAST, now=NOW)
    assert game_has_started("2026-10-26T23:30:00.0000000Z", now=NOW)
    assert not game_has_started(FUTURE, now=NOW)
    assert not game_has_started(None, now=NOW)
    assert not game_has_started("", status="not_started", now=NOW)
    assert not game_has_started(FUTURE, status="OPEN_PREGAME", now=NOW)
    assert game_has_started(FUTURE, status="OPEN_INGAME", now=NOW)
    assert game_has_started(PAST, status="OPEN_PREGAME", now=NOW)
    assert game_has_started(1_700_000_000_000, now=NOW)


def test_exact_start_instant_is_still_pregame() -> None:
    same = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert not game_has_started(same, now=NOW)
    assert game_has_started(same, now=NOW + timedelta(seconds=1))
    assert game_has_started(same, status="in_progress", now=NOW)


def test_fanduel_discovery_drops_started_nba_games() -> None:
    payload = {
        "attachments": {
            "competitions": {"1": {"name": "NBA", "competitionId": 1}},
            "events": {
                "a": {
                    "eventId": 1,
                    "name": "Live @ Game",
                    "competitionId": 1,
                    "openDate": PAST,
                    "inPlay": True,
                },
                "b": {
                    "eventId": 2,
                    "name": "Later @ Game",
                    "competitionId": 1,
                    "openDate": FUTURE,
                },
            },
            "markets": {},
        }
    }
    events = events_from_page(payload, {"fd_competition_names": ["NBA"]})
    assert [event.event_id for event in events] == ["2"]


def test_fanduel_discovery_drops_started_mlb_games() -> None:
    payload = {
        "attachments": {
            "competitions": {},
            "events": {
                "1": {
                    "eventId": 1,
                    "name": "Live @ Game",
                    "competitionName": "MLB",
                    "openDate": PAST,
                },
                "2": {
                    "eventId": 2,
                    "name": "Later @ Game",
                    "competitionName": "MLB",
                    "openDate": FUTURE,
                },
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
    from unittest.mock import patch

    with patch("src.scrapers.mlb.mlb_fanduel.fetch_json", return_value=payload):
        events = fetch_mlb_fd_events(MagicMock(), cfg)
    assert [event.event_id for event in events] == ["2"]


def _dk_payload() -> dict:
    return {
        "events": [
            {
                "id": "live",
                "name": "A @ B",
                "startEventDate": PAST,
                "status": "STARTED",
            },
            {
                "id": "later",
                "name": "C @ D",
                "startEventDate": FUTURE,
                "status": "NOT_STARTED",
            },
        ],
        "markets": [
            {"id": "m-live", "eventId": "live", "name": "Points"},
            {"id": "m-later", "eventId": "later", "name": "Points"},
        ],
        "selections": [],
    }


def _dk_cfg(group: str) -> dict:
    return {
        "dk_event_group": group,
        "dk_api_base": "https://example.test",
        "dk_timeout": 1,
        "dk_max_retries": 1,
        "dk_request_delay": 0,
    }


def test_draftkings_fetch_drops_started_games() -> None:
    from unittest.mock import patch

    sub = Subcategory("1215", "12488", "Games", "Points")
    events: dict = {}
    with patch(
        "src.scrapers.nba.nba_draftking.fetch_json", return_value=_dk_payload()
    ):
        offers = fetch_nba_dk_offers(
            MagicMock(), _dk_cfg("42648"), sub, events
        )
    assert [offer["eventId"] for offer in offers] == ["later"]
    assert list(events) == ["later"]

    mlb_events: dict = {}
    with patch(
        "src.scrapers.mlb.mlb_draftking.fetch_json", return_value=_dk_payload()
    ):
        mlb_offers = fetch_mlb_dk_offers(
            MagicMock(), _dk_cfg("84240"), Subcategory("1", "2", "Games", "Hits"), mlb_events
        )
    assert [offer["eventId"] for offer in mlb_offers] == ["later"]
    assert list(mlb_events) == ["later"]


def test_novig_skips_market_fetch_for_started_games() -> None:
    from unittest.mock import patch

    events = [
        {
            "id": "live",
            "status": "OPEN_INGAME",
            "game": {"scheduled_start": FUTURE},
        },
        {
            "id": "later",
            "status": "OPEN_PREGAME",
            "game": {"scheduled_start": FUTURE},
        },
    ]
    calls: list[str] = []

    def fake_markets(_session, event_id):
        calls.append(event_id)
        return []

    with patch(
        "src.scrapers.nba.nba_novig.fetch_nba_events", return_value=events
    ), patch(
        "src.scrapers.nba.nba_novig.fetch_event_markets", side_effect=fake_markets
    ), patch(
        "src.scrapers.nba.nba_novig.build_game_snapshots", return_value=([], [])
    ):
        fetch_nba_novig(MagicMock())
    assert calls == ["later"]

    mlb_calls: list[str] = []
    with patch(
        "src.scrapers.mlb.mlb_novig.fetch_mlb_events", return_value=events
    ), patch(
        "src.scrapers.mlb.mlb_novig.fetch_event_markets",
        side_effect=lambda _session, event_id: mlb_calls.append(event_id) or [],
    ), patch(
        "src.scrapers.mlb.mlb_novig.build_game_snapshots", return_value=([], [])
    ):
        fetch_mlb_novig(MagicMock())
    assert mlb_calls == ["later"]


def test_prophetx_keeps_only_pregame_events() -> None:
    events = [
        {"id": 1, "scheduled": PAST, "status": "live"},
        {"id": 2, "scheduled": FUTURE, "status": "not_started"},
    ]
    assert [event["id"] for event in nba_px_pregame(events, now=NOW)] == [2]
    assert [event["id"] for event in mlb_px_pregame(events, now=NOW)] == [2]


def _underdog_payload() -> dict:
    return {
        "players": [
            {
                "id": "p1",
                "position_id": "pos",
                "team_id": "t1",
                "first_name": "Nikola",
                "last_name": "Jokic",
            },
            {
                "id": "p2",
                "position_id": "pos",
                "team_id": "t2",
                "first_name": "Jamal",
                "last_name": "Murray",
            },
        ],
        "games": [
            {"id": "g1", "sport_id": "NBA", "scheduled_at": PAST, "status": "scoring"},
            {"id": "g2", "sport_id": "NBA", "scheduled_at": FUTURE, "status": "scheduled"},
        ],
        "appearances": [
            {
                "id": "a1",
                "match_id": "g1",
                "player_id": "p1",
                "position_id": "pos",
                "team_id": "t1",
            },
            {
                "id": "a2",
                "match_id": "g2",
                "player_id": "p2",
                "position_id": "pos",
                "team_id": "t2",
            },
        ],
        "over_under_lines": [
            {
                "status": "active",
                "stat_value": "27.5",
                "over_under": {
                    "appearance_stat": {"stat": "points", "appearance_id": "a1"}
                },
                "options": [{"appearance_id": "a1", "choice": "higher", "american_price": -115}],
            },
            {
                "status": "active",
                "stat_value": "19.5",
                "over_under": {
                    "appearance_stat": {"stat": "points", "appearance_id": "a2"}
                },
                "options": [{"appearance_id": "a2", "choice": "higher", "american_price": -110}],
            },
        ],
    }


def test_underdog_skips_started_games() -> None:
    nba = extract_nba_ud(_underdog_payload(), frozenset({"NBA"}))
    assert [pick.full_name for pick in nba] == ["Jamal Murray"]

    mlb_payload = _underdog_payload()
    for game in mlb_payload["games"]:
        game["sport_id"] = "MLB"
    mlb = extract_mlb_ud(mlb_payload, frozenset({"MLB"}))
    assert [pick.full_name for pick in mlb] == ["Jamal Murray"]


def _prizepick_payload() -> dict:
    def row(player_id: str, name: str, start: str, status: str) -> tuple[dict, dict]:
        included = {"type": "new_player", "id": player_id, "attributes": {"name": name}}
        data = {
            "type": "projection",
            "attributes": {
                "line_score": 20.5,
                "stat_type": "Points",
                "odds_type": "standard",
                "updated_at": FUTURE,
                "start_time": start,
                "status": status,
            },
            "relationships": {"new_player": {"data": {"id": player_id}}},
        }
        return included, data

    live_player, live_row = row("p1", "Live Player", PAST, "in_progress")
    later_player, later_row = row("p2", "Later Player", FUTURE, "pre_game")
    return {"included": [live_player, later_player], "data": [live_row, later_row]}


def test_prizepicks_skips_started_games() -> None:
    nba = extract_nba_pp(_prizepick_payload())
    assert [row.player for row in nba] == ["Later Player"]
    mlb = extract_mlb_pp(_prizepick_payload(), league="MLB")
    assert [row.player for row in mlb] == ["Later Player"]


def _pinnacle_rows() -> list[dict]:
    def row(matchup_id: int, start: str, status: str) -> dict:
        return {
            "type": "matchup",
            "id": matchup_id,
            "startTime": start,
            "status": status,
            "participants": [
                {"alignment": "away", "name": "Los Angeles Lakers"},
                {"alignment": "home", "name": "Boston Celtics"},
            ],
        }

    return [row(1, PAST, "started"), row(2, FUTURE, "pending")]


def test_pinnacle_urls_skip_started_games() -> None:
    nba = nba_pinnacle_urls(_pinnacle_rows(), "nba")
    assert nba == [
        "https://www.pinnacle.com/en/basketball/nba/los-angeles-lakers-vs-boston-celtics/2/#all"
    ]
    mlb = mlb_pinnacle_urls(_pinnacle_rows(), "mlb")
    assert mlb == [
        "https://www.pinnacle.com/en/baseball/mlb/los-angeles-lakers-vs-boston-celtics/2/#all"
    ]
