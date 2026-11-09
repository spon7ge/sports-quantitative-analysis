"""Layer 1 reads the team close. Layer 3 reads the confirmed Rotowire five."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from slate_inputs import (
    load_rotowire_closing_lineups,
    load_team_closing_scripts,
    match_starter_ids,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _team_close(path: Path) -> None:
    _write(
        path,
        {
            "source": "Pinnacle",
            "league": "NBA",
            "date": "2026-11-06",
            "timezone": "America/Los_Angeles",
            "games": [
                {
                    "game": "Boston Celtics vs Charlotte Hornets",
                    "closing_fetched_at": "2026-11-07T00:00:55Z",
                    "next_fetched_at": "2026-11-07T00:15:10Z",
                    "opening_markets": [
                        {
                            "market": "spread",
                            "side": "Charlotte Hornets",
                            "line": 8.0,
                            "american_price": 101,
                        },
                        {
                            "market": "total",
                            "side": "over",
                            "line": 210.5,
                            "american_price": -102,
                        },
                    ],
                    "markets": [
                        {
                            "market": "spread",
                            "side": "Boston Celtics",
                            "line": -10.5,
                            "american_price": -106,
                        },
                        {
                            "market": "spread",
                            "side": "Charlotte Hornets",
                            "line": 10.5,
                            "american_price": -104,
                        },
                        {
                            "market": "spread",
                            "side": "Charlotte Hornets",
                            "line": 3.5,
                            "american_price": 470,
                        },
                        {
                            "market": "total",
                            "side": "over",
                            "line": 209.0,
                            "american_price": -112,
                        },
                        {
                            "market": "total",
                            "side": "under",
                            "line": 209.0,
                            "american_price": -108,
                        },
                        {
                            "market": "total",
                            "side": "Over 199.5",
                            "line": 199.5,
                            "american_price": 400,
                        },
                        {
                            "market": "total",
                            "side": "under",
                            "line": 199.5,
                            "american_price": -600,
                        },
                    ],
                }
            ],
        },
    )


def _lineups(path: Path, *, home_closing: list[str] | None) -> None:
    _write(
        path,
        {
            "source": "RotoWire",
            "league": "NBA",
            "date": "2026-11-06",
            "timezone": "America/Los_Angeles",
            "games": [
                {
                    "away": "BOS",
                    "home": "CHA",
                    "away_opening": {
                        "fetched_at": "2026-11-06T08:45:00Z",
                        "status": "projected",
                        "lineup": [{"name": "Hugo Gonzalez", "position": "SG"}],
                    },
                    "away_closing": {
                        "fetched_at": "2026-11-07T03:15:01Z",
                        "status": "confirmed",
                        "lineup": [
                            {"name": "Derrick White", "position": "PG"},
                            {"name": "Jordan Walsh", "position": "SG"},
                        ],
                    },
                    "home_opening": {
                        "fetched_at": "2026-11-06T08:45:00Z",
                        "status": "projected",
                        "lineup": [{"name": "Opening Home", "position": "PG"}],
                    },
                    "home_closing": None
                    if home_closing is None
                    else {
                        "fetched_at": "2026-11-07T03:15:01Z",
                        "status": "confirmed",
                        "lineup": [
                            {"name": name, "position": "PG"} for name in home_closing
                        ],
                    },
                }
            ],
        },
    )


def test_layer1_uses_the_closing_spread_and_total_not_the_open(tmp_path: Path):
    path = tmp_path / "pinnacle_nba_team_closing_2026-11-06.json"
    _team_close(path)
    scripts, audit = load_team_closing_scripts(path, "2026-11-06")
    script = scripts[("bos", "cha")]
    assert script.total == 209.0
    assert script.home_spread == 10.5
    assert audit["alternate_lines_dropped"] == 2
    assert audit["opening_ignored"] == 1


def test_a_close_from_another_pacific_day_is_dropped(tmp_path: Path):
    path = tmp_path / "pinnacle_nba_team_closing_2026-11-06.json"
    _team_close(path)
    payload = json.loads(path.read_text())
    payload["games"][0]["closing_fetched_at"] = "2026-11-07T12:00:00Z"
    path.write_text(json.dumps(payload))
    scripts, audit = load_team_closing_scripts(path, "2026-11-06")
    assert scripts == {}
    assert audit["timestamp_mismatch"] == 1


def test_layer3_uses_the_confirmed_five_and_not_the_opening(tmp_path: Path):
    path = tmp_path / "rotowire_nba_lineup_closing_2026-11-06.json"
    _lineups(path, home_closing=["LaMelo Ball"])
    lineups, audit = load_rotowire_closing_lineups(path, "2026-11-06")
    away, home = lineups[("bos", "cha")]
    assert away == ("Derrick White", "Jordan Walsh")
    assert home == ("LaMelo Ball",)
    assert "Hugo Gonzalez" not in away
    assert audit["sides_missing_close"] == 0

    _lineups(path, home_closing=None)
    lineups, audit = load_rotowire_closing_lineups(path, "2026-11-06")
    assert ("bos", "cha") not in lineups
    assert audit["sides_missing_close"] == 1


def test_starter_names_map_onto_the_pre_slate_roster():
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Derrick White", "Jordan Walsh", "Someone Else"],
        }
    )
    starters, missing = match_starter_ids(
        roster, ["Derrick White", "Jordan Walsh", "Not On Roster"]
    )
    assert starters == {1, 2}
    assert missing == ["Not On Roster"]
    with pytest.raises(ValueError, match="five"):
        match_starter_ids(
            roster,
            ["Derrick White", "Jordan Walsh", "A", "B", "C", "D"],
        )
