from __future__ import annotations

import pandas as pd
from line_movement import detect_movements as exported_detect_movements
from line_movement.detect import detect_movements
from line_movement.match import attach_market_keys
from line_movement.normalize import normalize_quotes


def _q(book, snap, line, price, **kw):
    row = {
        "book": book,
        "snapshot_at": pd.Timestamp(snap, tz="America/Los_Angeles"),
        "player": "Test Player",
        "stat": "points",
        "line": line,
        "side": "over",
        "american_price": price,
        "event_name": "AAA @ BBB",
        "start_time": pd.NaT,
        "fetched_at": pd.NaT,
        "source_file": "x",
        "line_updated_at": pd.NaT,
        "is_main": True,
        "decimal_price": None,
        "implied_probability": None,
        "market_name": None,
        "odds_type": None,
    }
    row.update(kw)
    return row


def _quotes(rows):
    return attach_market_keys(normalize_quotes(pd.DataFrame(rows)))


def test_detect_movements_is_exported():
    assert exported_detect_movements is detect_movements


def test_price_only_move():
    quotes = _quotes(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 20.5, -120),
        ]
    )

    moves = detect_movements(quotes)

    assert len(moves) == 1
    move = moves.iloc[0]
    assert move["series"] == "price"
    assert move["change_type"] == "price"
    assert bool(move["price_shortened"]) is True
    assert move["prev_american_price"] == -110
    assert move["new_american_price"] == -120
    assert move["lag_seconds"] == 900


def test_no_move_when_identical():
    quotes = _quotes(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 20.5, -110),
        ]
    )

    assert detect_movements(quotes).empty


def test_line_continuity_move():
    quotes = _quotes(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 21.5, -110),
        ]
    )

    moves = detect_movements(quotes)
    line_moves = moves[moves["change_type"] == "line"]

    assert len(line_moves) == 1
    move = line_moves.iloc[0]
    assert move["series"] == "line"
    assert bool(move["line_up"]) is True
    assert move["prev_line"] == 20.5
    assert move["new_line"] == 21.5
    assert move["line_choice"] == "main"


def test_zero_line_is_excluded_from_line_continuity():
    quotes = _quotes(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _q("fanduel", "2026-10-24 12:15:00", 0.0, -110),
        ]
    )

    line_moves = detect_movements(quotes).query("change_type == 'line'")

    assert line_moves.empty


def test_price_nan_transition_is_a_move():
    quotes = _quotes(
        [
            _q("fanduel", "2026-10-24 12:00:00", 20.5, None),
            _q("fanduel", "2026-10-24 12:15:00", 20.5, -110),
        ]
    )

    moves = detect_movements(quotes)

    assert len(moves) == 1
    assert moves.iloc[0]["change_type"] == "price"
    assert pd.isna(moves.iloc[0]["prev_american_price"])
    assert not bool(moves.iloc[0]["price_shortened"])
    assert not bool(moves.iloc[0]["price_lengthened"])


def test_line_choice_prefers_main_quote():
    quotes = _quotes(
        [
            _q(
                "fanduel",
                "2026-10-24 12:00:00",
                20.5,
                -110,
                is_main=False,
                line_updated_at=pd.Timestamp("2026-10-24 11:59:00"),
            ),
            _q("fanduel", "2026-10-24 12:00:00", 21.5, -115, is_main=True),
            _q("fanduel", "2026-10-24 12:15:00", 22.5, -115, is_main=True),
        ]
    )

    line_moves = detect_movements(quotes).query("change_type == 'line'")

    assert len(line_moves) == 1
    assert line_moves.iloc[0]["prev_line"] == 21.5
    assert line_moves.iloc[0]["new_line"] == 22.5
    assert line_moves.iloc[0]["line_choice"] == "main"


def test_line_choice_uses_latest_update_then_median_fallback():
    updated_quotes = _quotes(
        [
            _q(
                "fanduel",
                "2026-10-24 12:00:00",
                20.5,
                -110,
                is_main=False,
                line_updated_at=pd.Timestamp("2026-10-24 11:55:00"),
            ),
            _q(
                "fanduel",
                "2026-10-24 12:00:00",
                21.5,
                -110,
                is_main=False,
                line_updated_at=pd.Timestamp("2026-10-24 11:59:00"),
            ),
            _q(
                "fanduel",
                "2026-10-24 12:15:00",
                22.5,
                -110,
                is_main=False,
                line_updated_at=pd.Timestamp("2026-10-24 12:14:00"),
            ),
        ]
    )
    median_quotes = _quotes(
        [
            _q(
                "draftkings",
                "2026-10-24 12:00:00",
                19.5,
                -110,
                is_main=False,
            ),
            _q(
                "draftkings",
                "2026-10-24 12:00:00",
                20.5,
                -110,
                is_main=False,
            ),
            _q(
                "draftkings",
                "2026-10-24 12:00:00",
                21.5,
                -110,
                is_main=False,
            ),
            _q(
                "draftkings",
                "2026-10-24 12:15:00",
                22.5,
                -110,
                is_main=False,
            ),
        ]
    )

    updated_move = detect_movements(updated_quotes).query("change_type == 'line'").iloc[0]
    median_move = detect_movements(median_quotes).query("change_type == 'line'").iloc[0]

    assert updated_move["prev_line"] == 21.5
    assert updated_move["line_choice"] == "updated_at"
    assert median_move["prev_line"] == 20.5
    assert median_move["line_choice"] == "median"
