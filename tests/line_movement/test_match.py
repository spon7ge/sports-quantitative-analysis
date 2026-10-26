from __future__ import annotations

import pandas as pd
from line_movement import (
    MARKET_KEY_COLS,
    attach_market_keys,
    build_market_groups,
    cross_book_groups,
    normalize_quotes,
)


def test_matching_interfaces_are_exported():
    assert MARKET_KEY_COLS == (
        "player_norm",
        "event_key",
        "prop",
        "period",
        "side",
        "line",
    )
    assert callable(attach_market_keys)
    assert callable(build_market_groups)
    assert callable(cross_book_groups)


def test_market_keys_track_exact_market_and_line_continuity():
    quotes = pd.DataFrame(
        {
            "player_norm": ["a", "a", "a"],
            "event_key": ["aaa__bbb", "aaa__bbb", None],
            "prop": ["points", "points", "points"],
            "period": ["game", "game", "game"],
            "side": ["over", "over", None],
            "line": [20.5, 21.5, 20.5],
        }
    )

    keyed = attach_market_keys(quotes)

    assert "market_key" not in quotes.columns
    assert keyed.loc[0, "market_key"] != keyed.loc[1, "market_key"]
    assert keyed.loc[0, "continuity_key"] == keyed.loc[1, "continuity_key"]
    assert keyed.loc[2, "market_key"] is None
    assert keyed.loc[2, "continuity_key"] is None


def test_market_groups_count_distinct_books_and_exclude_incomplete_keys():
    quotes = pd.DataFrame(
        {
            "player_norm": ["a", "a", "a", "a"],
            "event_key": ["aaa__bbb", "aaa__bbb", "aaa__bbb", None],
            "prop": ["points"] * 4,
            "period": ["game"] * 4,
            "side": ["over", "over", "over", None],
            "line": [20.5] * 4,
            "book": ["fanduel", "fanduel", "draftkings", "prizepicks"],
        }
    )

    groups = build_market_groups(quotes)
    complete = groups.loc[groups["matchable"]].iloc[0]

    assert complete["n_books"] == 2
    assert complete["books"] == "draftkings,fanduel"
    assert complete["n_rows"] == 3
    assert groups["matchable"].tolist() == [True, False]
    assert len(cross_book_groups(groups)) == 1


def test_cross_book_group_requires_two_books():
    rows = [
        {
            "book": "fanduel",
            "snapshot_at": pd.Timestamp(
                "2026-10-24 12:00:00", tz="America/Los_Angeles"
            ),
            "player": "A",
            "stat": "points",
            "line": 20.5,
            "side": "over",
            "event_name": "AAA @ BBB",
            "american_price": -110,
            "start_time": pd.NaT,
            "fetched_at": pd.NaT,
            "source_file": "a",
            "line_updated_at": pd.NaT,
            "is_main": True,
            "decimal_price": None,
            "implied_probability": None,
            "market_name": None,
            "odds_type": None,
        },
        {
            "book": "draftkings",
            "snapshot_at": pd.Timestamp(
                "2026-10-24 12:00:00", tz="America/Los_Angeles"
            ),
            "player": "A",
            "stat": "points",
            "line": 20.5,
            "side": "over",
            "event_name": "BBB vs AAA",
            "american_price": -115,
            "start_time": pd.NaT,
            "fetched_at": pd.NaT,
            "source_file": "b",
            "line_updated_at": pd.NaT,
            "is_main": True,
            "decimal_price": None,
            "implied_probability": None,
            "market_name": None,
            "odds_type": None,
        },
        {
            "book": "prizepicks",
            "snapshot_at": pd.Timestamp(
                "2026-10-24 12:00:00", tz="America/Los_Angeles"
            ),
            "player": "A",
            "stat": "points",
            "line": 20.5,
            "side": None,
            "event_name": None,
            "american_price": None,
            "start_time": pd.NaT,
            "fetched_at": pd.NaT,
            "source_file": "c",
            "line_updated_at": pd.NaT,
            "is_main": None,
            "decimal_price": None,
            "implied_probability": None,
            "market_name": None,
            "odds_type": "standard",
        },
    ]
    quotes = normalize_quotes(pd.DataFrame(rows))
    groups = build_market_groups(quotes)
    cross = cross_book_groups(groups)
    assert len(cross) == 1
    assert cross.iloc[0]["n_books"] == 2


def test_cross_book_groups_excludes_incomplete_side_even_with_two_books():
    quotes = pd.DataFrame(
        {
            "player_norm": ["a", "a"],
            "event_key": ["aaa__bbb", "aaa__bbb"],
            "prop": ["points", "points"],
            "period": ["game", "game"],
            "side": [None, None],
            "line": [20.5, 20.5],
            "book": ["fanduel", "draftkings"],
        }
    )

    groups = build_market_groups(quotes)
    incomplete = groups.iloc[0]

    assert incomplete["n_books"] == 2
    assert not incomplete["matchable"]
    assert cross_book_groups(groups).empty


def test_cross_book_groups_excludes_single_book_complete_market():
    quotes = pd.DataFrame(
        {
            "player_norm": ["a"],
            "event_key": ["aaa__bbb"],
            "prop": ["points"],
            "period": ["game"],
            "side": ["over"],
            "line": [20.5],
            "book": ["fanduel"],
        }
    )

    groups = build_market_groups(quotes)
    complete = groups.iloc[0]

    assert complete["matchable"]
    assert complete["n_books"] == 1
    assert cross_book_groups(groups).empty
