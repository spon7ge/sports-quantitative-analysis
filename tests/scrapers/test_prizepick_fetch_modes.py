"""PrizePicks HTTP-only / browser-only modes for the concurrent runner."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("module_path", [
    "src.scrapers.nba.nba_prizepick",
    "src.scrapers.mlb.mlb_prizepick",
])
def test_no_browser_exits_with_need_browser_code(monkeypatch, module_path) -> None:
    mod = __import__(module_path, fromlist=["fetch_projections_payloads"])

    monkeypatch.setenv("PRIZEPICKS_NO_BROWSER", "1")
    monkeypatch.delenv("PRIZEPICKS_BROWSER_ONLY", raising=False)
    monkeypatch.setattr(mod, "resolve_leagues", lambda names: (("NBA", 7),))
    monkeypatch.setattr(mod, "try_fetch_with_curl_cffi", lambda leagues: None)
    monkeypatch.setattr(mod, "try_fetch_with_requests", lambda leagues: None)

    def boom(_leagues):
        raise AssertionError("playwright must not run when NO_BROWSER=1")

    monkeypatch.setattr(mod, "try_fetch_with_playwright", boom)

    with pytest.raises(SystemExit) as excinfo:
        mod.fetch_projections_payloads()
    assert excinfo.value.code == mod.PRIZEPICKS_NEED_BROWSER_EXIT


@pytest.mark.parametrize("module_path", [
    "src.scrapers.nba.nba_prizepick",
    "src.scrapers.mlb.mlb_prizepick",
])
def test_browser_only_skips_http(monkeypatch, module_path) -> None:
    mod = __import__(module_path, fromlist=["fetch_projections_payloads"])

    monkeypatch.setenv("PRIZEPICKS_BROWSER_ONLY", "1")
    monkeypatch.delenv("PRIZEPICKS_NO_BROWSER", raising=False)
    monkeypatch.setattr(mod, "resolve_leagues", lambda names: (("NBA", 7),))

    def http_boom(_leagues):
        raise AssertionError("HTTP must not run when BROWSER_ONLY=1")

    monkeypatch.setattr(mod, "try_fetch_with_curl_cffi", http_boom)
    monkeypatch.setattr(mod, "try_fetch_with_requests", http_boom)
    monkeypatch.setattr(
        mod, "try_fetch_with_playwright", lambda leagues: {"NBA": {"data": []}}
    )

    out = mod.fetch_projections_payloads()
    assert out == {"NBA": {"data": []}}
