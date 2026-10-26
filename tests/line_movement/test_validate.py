from __future__ import annotations

import pandas as pd
from line_movement.normalize import normalize_quotes
from line_movement.validate import filter_schedule, validate_quotes


def _raw_row(**overrides):
    base = {
        "book": "fanduel",
        "snapshot_at": pd.Timestamp(
            "2026-10-24 12:00:00", tz="America/Los_Angeles"
        ),
        "player": "Luka Doncic",
        "stat": "Points",
        "line": 28.5,
        "side": "Over",
        "american_price": -110,
        "event_name": "DAL @ BOS",
        "start_time": pd.Timestamp("2026-10-24 17:00:00", tz="UTC"),
        "fetched_at": pd.NaT,
        "source_file": "x.json",
        "line_updated_at": pd.NaT,
        "is_main": True,
        "decimal_price": None,
        "implied_probability": None,
        "market_name": None,
        "odds_type": None,
    }
    base.update(overrides)
    return base


def test_normalize_quotes_adds_keys():
    raw = pd.DataFrame([_raw_row()])
    out = normalize_quotes(raw)
    assert out.loc[0, "player_norm"] == "luka doncic"
    assert out.loc[0, "prop"] == "points"
    assert out.loc[0, "side"] == "over"
    assert out.loc[0, "period"] == "game"
    assert out.loc[0, "event_key"] is not None


def test_filter_schedule_drops_late_games():
    raw = pd.DataFrame(
        [
            _raw_row(
                start_time=pd.Timestamp("2026-10-24 16:00:00", tz="UTC")
            ),  # 9am PT — drop
            _raw_row(
                start_time=pd.Timestamp("2026-10-24 20:00:00", tz="UTC")
            ),  # 1pm PT — keep
            _raw_row(
                start_time=pd.Timestamp("2026-10-25 02:00:00", tz="UTC")
            ),  # 7pm PT — drop (end exclusive 19:00)
        ]
    )
    quotes = normalize_quotes(raw)
    kept, dropped = filter_schedule(quotes, date="2026-10-24")
    assert dropped == 2
    assert len(kept) == 1


def test_validate_flags_missing_side():
    raw = pd.DataFrame(
        [_raw_row(book="prizepicks", side=None, american_price=None)]
    )
    quotes = normalize_quotes(raw)
    report = validate_quotes(quotes)
    assert report.missing_rates_by_book["prizepicks"]["side"] == 1.0


def test_filter_schedule_keeps_missing_start_time():
    quotes = normalize_quotes(pd.DataFrame([_raw_row(start_time=pd.NaT)]))

    kept, dropped = filter_schedule(quotes, date="2026-10-24")

    assert len(kept) == 1
    assert dropped == 0


def test_validate_reports_duplicates_schema_flags_and_snapshot_span():
    snapshot = pd.Timestamp("2026-10-24 12:00:00", tz="America/Los_Angeles")
    raw = pd.DataFrame(
        [
            _raw_row(
                book="prizepicks",
                snapshot_at=snapshot,
                side=None,
                american_price=None,
                is_main=None,
                start_time=pd.NaT,
            ),
            _raw_row(
                book="prizepicks",
                snapshot_at=snapshot,
                side=None,
                american_price=None,
                is_main=None,
                start_time=pd.NaT,
            ),
        ]
    )

    report = validate_quotes(normalize_quotes(raw))

    assert report.duplicate_count == 1
    assert report.schema_flags == {
        "all_null_side": ["prizepicks"],
        "all_null_is_main": ["prizepicks"],
        "all_null_american_price": ["prizepicks"],
        "missing_start_time": 2,
        "books_without_event_key_overlap": ["prizepicks"],
    }
    assert report.snapshot_span_by_book["prizepicks"] == {
        "min": snapshot,
        "max": snapshot,
    }


def test_validate_flags_books_without_event_key_overlap():
    raw = pd.DataFrame(
        [
            _raw_row(book="fanduel", event_name="Milwaukee Bucks @ Boston Celtics"),
            _raw_row(book="draftkings", event_name="MIL Bucks @ Boston Celtics"),
            _raw_row(book="pinnacle", event_name="Dallas Mavericks @ Phoenix Suns"),
        ]
    )

    report = validate_quotes(normalize_quotes(raw))

    assert report.schema_flags["books_without_event_key_overlap"] == ["pinnacle"]
