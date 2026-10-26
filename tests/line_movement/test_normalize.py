from __future__ import annotations

from line_movement.normalize import (
    normalize_event_key,
    normalize_period,
    normalize_player,
    normalize_prop,
    normalize_side,
)


def test_normalize_player_strips_suffix_and_case():
    assert normalize_player("  Luka Dončić Jr. ") == "luka doncic"
    assert normalize_player(None) is None


def test_normalize_prop_aliases():
    assert normalize_prop("Points") == "points"
    assert normalize_prop("player_points") == "points"
    assert normalize_prop("Pts + Rebs + Asts") == "pra"
    assert normalize_prop(None) is None


def test_normalize_side_and_period():
    assert normalize_side("Over") == "over"
    assert normalize_side("u") == "under"
    assert normalize_side(None) is None
    assert normalize_period(None) == "game"
    assert normalize_period("1H") == "1h"


def test_normalize_event_key_orders_teams():
    assert normalize_event_key("Lakers @ Celtics") == normalize_event_key(
        "Celtics vs Lakers"
    )
    assert normalize_event_key(None) is None


def test_normalize_event_key_joins_draftkings_team_abbreviations():
    assert normalize_event_key(
        "MIL Bucks @ PHI 76ers"
    ) == normalize_event_key("Milwaukee Bucks @ Philadelphia 76ers")
    assert normalize_event_key(
        "DAL Mavericks @ SA Spurs"
    ) == normalize_event_key("Dallas Mavericks @ San Antonio Spurs")
