"""PrizePicks optional demon/goblin filter before DB load."""

from __future__ import annotations

from src.scrapers.nba.nba_prizepick import (
    Projection,
    excluded_odds_types,
    filter_projections_for_db,
)


def test_excluded_odds_types_empty_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PRIZEPICKS_EXCLUDE_ODDS_TYPES", raising=False)
    assert excluded_odds_types() == frozenset()


def test_filter_projections_for_db_drops_demons(monkeypatch) -> None:
    monkeypatch.setenv("PRIZEPICKS_EXCLUDE_ODDS_TYPES", "demon,goblin")
    rows = [
        Projection("A", "Points", 20.5, "standard", "", "NBA"),
        Projection("B", "Points", 22.5, "demon", "", "NBA"),
        Projection("C", "Points", 18.5, "goblin", "", "NBA"),
    ]
    kept = filter_projections_for_db(rows)
    assert [p.player for p in kept] == ["A"]
