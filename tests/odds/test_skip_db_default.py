"""Supabase odds writes are opt-in via ODDS_WRITE_DB."""

from __future__ import annotations

from src.odds.load_snapshots import _skip_db


def test_skip_db_default_without_write_flag(monkeypatch) -> None:
    monkeypatch.delenv("ODDS_WRITE_DB", raising=False)
    monkeypatch.delenv("PROPHETX_SKIP_DB", raising=False)
    assert _skip_db("PROPHETX_SKIP_DB") is True


def test_skip_db_allows_write_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("ODDS_WRITE_DB", "1")
    monkeypatch.delenv("PROPHETX_SKIP_DB", raising=False)
    assert _skip_db("PROPHETX_SKIP_DB") is False
    monkeypatch.setenv("PROPHETX_SKIP_DB", "1")
    assert _skip_db("PROPHETX_SKIP_DB") is True
