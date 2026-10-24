"""FanDuel parallel event/tab fetch and Retry-After backoff."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from src.scrapers.nba.nba_fanduel import (
    Event,
    _DEFAULT_CONFIG,
    extract_picks,
    fetch_json,
)


def test_default_delay_is_zero_and_concurrency_set() -> None:
    assert float(_DEFAULT_CONFIG.get("fd_request_delay") or 0) == 0.0
    assert int(_DEFAULT_CONFIG.get("fd_concurrency") or 0) >= 6


def test_fetch_json_honors_retry_after(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "src.scrapers.nba.nba_fanduel.time.sleep", lambda s: sleeps.append(s)
    )

    class Resp:
        status_code = 429
        headers = {"Retry-After": "1.25"}

        def raise_for_status(self):
            raise requests.HTTPError("429", response=self)

        def json(self):
            return {}

    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    ok.raise_for_status = MagicMock()
    ok.json = MagicMock(return_value={"ok": True})

    session = MagicMock()
    session.get.side_effect = [Resp(), ok]

    out = fetch_json(
        session, "https://example.com", {}, timeout=5, max_retries=3, label="t"
    )
    assert out == {"ok": True}
    assert sleeps and sleeps[0] == 1.25


def test_extract_picks_schedules_event_tabs_concurrently() -> None:
    events = [
        Event(event_id="1", name="A @ B", start_time=None, competition="NBA"),
        Event(event_id="2", name="C @ D", start_time=None, competition="NBA"),
    ]
    cfg = {
        **_DEFAULT_CONFIG,
        "fd_concurrency": 4,
        "fd_event_tabs": ["player-points", "player-rebounds"],
        "fd_request_delay": 0,
    }
    calls: list[tuple[str, str | None]] = []

    def fake_fetch(session, cfg, event, tab):
        calls.append((event.event_id, tab))
        return [{"marketId": f"{event.event_id}-{tab}", "marketName": "x"}], {}

    with patch(
        "src.scrapers.nba.nba_fanduel.fetch_event_tab_markets",
        side_effect=fake_fetch,
    ), patch(
        "src.scrapers.nba.nba_fanduel.extract_event_picks",
        return_value=[],
    ):
        extract_picks(MagicMock(), cfg, events, frozenset({"NBA"}))

    assert len(calls) == 4
    assert {(e, t) for e, t in calls} == {
        ("1", "player-points"),
        ("1", "player-rebounds"),
        ("2", "player-points"),
        ("2", "player-rebounds"),
    }
