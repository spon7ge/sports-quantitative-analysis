"""Novig parallel event-market fetch and DB failure status."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.scrapers.nba import nba_novig as novig


def test_fetch_graphql_snapshots_uses_worker_pool() -> None:
    events = [{"id": f"e{i}", "description": f"g{i}"} for i in range(4)]
    calls: list[str] = []

    def fake_markets(session, event_id):
        calls.append(event_id)
        return [{"id": event_id, "type": "MONEY"}]

    with patch.object(novig, "fetch_nba_events", return_value=events), patch.object(
        novig, "fetch_event_markets", side_effect=fake_markets
    ), patch.object(
        novig,
        "build_game_snapshots",
        return_value=([], []),
    ), patch.dict("os.environ", {"NOVIG_MARKET_WORKERS": "3"}):
        novig._fetch_graphql_snapshots(MagicMock())

    assert set(calls) == {"e0", "e1", "e2", "e3"}


def test_load_supabase_snapshots_returns_false_on_error() -> None:
    with patch(
        "src.odds.load_snapshots.load_novig_props_snapshot",
        side_effect=TimeoutError("statement timeout"),
    ):
        ok, rows = novig.load_supabase_snapshots([], [])
    assert ok is False
    assert rows == 0
