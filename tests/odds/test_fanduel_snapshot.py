"""FanDuel scraper JSON → odds.mlb_fanduel rows."""

from __future__ import annotations

from datetime import datetime, timezone

from src.odds.quote_specs import QUOTE_SPECS
from src.odds.snapshot_rows import fanduel_picks_to_rows


def test_mlb_fanduel_uses_book_props_quote_spec() -> None:
    spec = QUOTE_SPECS["mlb_fanduel"]
    assert spec.identity_cols == (
        "league",
        "player_name",
        "market_type",
        "side",
    )
    assert spec.compare_cols == ("line_score", "american_price")


def test_fanduel_picks_to_rows_maps_strikeouts_and_recovers_side() -> None:
    scraped_at = datetime(2026, 9, 22, 4, 21, 30, tzinfo=timezone.utc)
    rows = fanduel_picks_to_rows(
        [
            {
                "full_name": "Carlos Rodon Over",
                "stat_name": "strikeouts",
                "stat_value": 4.5,
                "choice": "over",
                "american_price": -154,
            },
            {
                "full_name": "Carlos Rodon Under",
                "stat_name": "strikeouts",
                "stat_value": 4.5,
                "choice": "over",
                "american_price": 120,
            },
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
        },
        {
            "league": "mlb",
            "player_name": "Carlos Rodon",
            "market_type": "player_strikeouts",
            "stat_category": "strikeouts",
            "side": "under",
            "line_score": 4.5,
            "american_price": 120,
            "scraped_at": scraped_at,
        },
    ]


def test_load_fanduel_snapshot_skip_db(monkeypatch) -> None:
    monkeypatch.setenv("FANDUEL_SKIP_DB", "1")
    from src.odds.load_snapshots import load_fanduel_snapshot

    n = load_fanduel_snapshot(
        [{"full_name": "A", "stat_name": "strikeouts", "stat_value": 1.5, "choice": "over", "american_price": -110}],
        league="mlb",
    )
    assert n == 0
