"""DraftKings scraper JSON → odds.mlb_draftkings rows."""

from __future__ import annotations

from datetime import datetime, timezone

from src.odds.quote_specs import QUOTE_SPECS
from src.odds.snapshot_rows import draftkings_picks_to_rows


def test_mlb_draftkings_uses_book_props_quote_spec() -> None:
    spec = QUOTE_SPECS["mlb_draftkings"]
    assert spec.identity_cols == (
        "league",
        "player_name",
        "market_type",
        "side",
    )
    assert spec.compare_cols == ("line_score", "american_price")


def test_draftkings_picks_to_rows_maps_strikeout_milestones() -> None:
    scraped_at = datetime(2026, 9, 22, 4, 21, 30, tzinfo=timezone.utc)
    rows = draftkings_picks_to_rows(
        [
            {
                "full_name": "Carlos Rodon",
                "stat_name": "strikeouts",
                "stat_value": 4.5,
                "choice": "over",
                "american_price": -154,
            }
        ],
        league="mlb",
        scraped_at=scraped_at,
    )
    assert rows == [
        {
            "league": "mlb",
            "player_name": "Carlos Rodon",
            "market_type": "player_strikeouts",
            "stat_category": "strikeouts",
            "side": "over",
            "line_score": 4.5,
            "american_price": -154,
            "scraped_at": scraped_at,
        }
    ]


def test_load_draftkings_snapshot_skip_db(monkeypatch) -> None:
    monkeypatch.setenv("DRAFTKINGS_SKIP_DB", "1")
    from src.odds.load_snapshots import load_draftkings_snapshot

    n = load_draftkings_snapshot(
        [{"full_name": "A", "stat_name": "strikeouts", "stat_value": 1.5, "choice": "over", "american_price": -110}],
        league="mlb",
    )
    assert n == 0
