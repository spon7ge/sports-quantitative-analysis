"""Index migration covers every QuoteSpec table identity."""

from __future__ import annotations

from pathlib import Path

from src.odds.quote_specs import QUOTE_SPECS

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "sql"
    / "migrations"
    / "20261023_odds_latest_quote_indexes.sql"
)


def test_migration_indexes_every_quote_spec_table() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    missing = [table for table in QUOTE_SPECS if f"odds.{table}" not in sql]
    assert missing == [], f"tables missing from index migration: {missing}"


def test_migration_uses_concurrent_builds() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "CREATE INDEX CONCURRENTLY" in sql
    assert "scraped_at DESC" in sql
