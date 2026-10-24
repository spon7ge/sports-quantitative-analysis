"""ProphetX concurrent market groups and period toggle."""

from __future__ import annotations

from unittest.mock import patch

from src.scrapers.nba import nba_prophetx as px


def test_include_period_markets_flag(monkeypatch) -> None:
    monkeypatch.setenv("PROPHETX_INCLUDE_PERIOD_MARKETS", "0")
    assert px._include_period_markets() is False
    monkeypatch.setenv("PROPHETX_INCLUDE_PERIOD_MARKETS", "1")
    assert px._include_period_markets() is True


def test_run_fetches_market_groups_concurrently(monkeypatch) -> None:
    monkeypatch.setenv("PROPHETX_INCLUDE_PERIOD_MARKETS", "1")
    calls: list[str] = []

    def fake_fetch(session, event_ids, **kwargs):
        if kwargs.get("market_types"):
            calls.append("team")
        elif "period" in str(kwargs.get("market_sub_types", "")).lower() or (
            kwargs.get("market_sub_types")
            and any(s in kwargs["market_sub_types"] for s in px.PERIOD_MARKET_SUBTYPES)
        ):
            calls.append("period")
        else:
            calls.append("props")
        return []

    monkeypatch.setattr(px, "fetch_nba_events", lambda session: [{"id": 1}])
    monkeypatch.setattr(px, "fetch_markets_for_events", fake_fetch)
    monkeypatch.setattr(
        px, "build_game_snapshots", lambda *a, **k: ([{"props": []}], [])
    )
    monkeypatch.setattr(px, "resolve_props_output_path", lambda: "/tmp/px_props.json")
    monkeypatch.setattr(
        px, "write_snapshots", lambda *a, **k: ("/tmp/px_props.json", "/tmp/px_team.json")
    )
    monkeypatch.setattr(px, "load_supabase_snapshots", lambda *a, **k: (True, 0))

    try:
        px.run()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")

    assert set(calls) == {"team", "period", "props"}
