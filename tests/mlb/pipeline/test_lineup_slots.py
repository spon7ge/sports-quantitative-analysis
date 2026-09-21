"""Starting-nine parser tests."""

from __future__ import annotations

import json

from src.mlb.pipeline.lineup_slots import parse_starting_nine
from src.mlb.pipeline.parse import parse_lineups


def _player(player_id: int, batting_order: str, position: str) -> dict[str, object]:
    return {
        "person": {"id": player_id},
        "battingOrder": batting_order,
        "position": {"abbreviation": position},
    }


def _payload(
    *,
    home_players: dict[str, dict[str, object]] | None = None,
    away_players: dict[str, dict[str, object]] | None = None,
    home_order: list[int] | None = None,
    pitchers: list[dict[str, int]] | None = None,
    season: str = "2024",
) -> str:
    return json.dumps(
        {
            "gamePk": 745001,
            "gameData": {"game": {"pk": 745001, "season": season}},
            "liveData": {
                "boxscore": {
                    "teams": {
                        "home": {
                            "team": {"id": 118},
                            "players": home_players or {},
                            "battingOrder": home_order or [],
                            "pitchers": pitchers or [],
                        },
                        "away": {
                            "team": {"id": 139},
                            "players": away_players or {},
                            "battingOrder": [],
                        },
                    }
                }
            },
        }
    )


def test_parse_starting_nine_uses_00_codes_not_final_batting_order() -> None:
    starters = {
        f"ID{player_id}": _player(player_id, f"{slot}00", "SS" if slot == 2 else "OF")
        for slot, player_id in enumerate(
            [111001, 677951, 111003, 111004, 111005, 111006, 111007, 111008, 111009],
            start=1,
        )
    }
    starters["ID641658"] = _player(641658, "201", "2B")
    payload = _payload(
        home_players=starters,
        home_order=[
            111001,
            641658,
            111003,
            111004,
            111005,
            111006,
            111007,
            111008,
            111009,
        ],
    )

    parsed = parse_starting_nine(payload)
    slot_two = parsed.loc[(parsed["side"] == "home") & (parsed["slot"] == 2)].iloc[0]

    assert int(slot_two["batter_id"]) == 677951
    assert int(slot_two["slot_is_pitcher"]) == 0
    assert 641658 not in set(parsed["batter_id"])
    assert int(slot_two["season"]) == 2024

    compatible = parse_lineups(payload)
    assert list(compatible.columns) == [
        "game_pk",
        "team_id",
        "batter_id",
        "batting_slot",
        "side",
    ]
    assert int(compatible.loc[compatible["batting_slot"] == 2, "batter_id"].iloc[0]) == 677951


def test_parse_starting_nine_stanek_keeps_opener_and_excludes_pinch_hitter() -> None:
    away_players = {
        f"ID{player_id}": _player(player_id, f"{slot}00", "P" if slot == 9 else "OF")
        for slot, player_id in enumerate(
            [222001, 222002, 222003, 222004, 222005, 222006, 222007, 222008, 592773],
            start=1,
        )
    }
    away_players["ID623205"] = _player(623205, "901", "SS")

    parsed = parse_starting_nine(_payload(away_players=away_players))
    slot_nine = parsed.loc[(parsed["side"] == "away") & (parsed["slot"] == 9)].iloc[0]

    assert int(slot_nine["batter_id"]) == 592773
    assert int(slot_nine["slot_is_pitcher"]) == 1
    assert 623205 not in set(parsed["batter_id"])


def test_parse_starting_nine_ohtani_uses_batting_position_not_pitcher_list() -> None:
    home_players = {"ID660271": _player(660271, "100", "DH")}

    parsed = parse_starting_nine(
        _payload(home_players=home_players, pitchers=[{"id": 660271}], season="")
    )
    ohtani = parsed.iloc[0]

    assert int(ohtani["batter_id"]) == 660271
    assert int(ohtani["slot_is_pitcher"]) == 0
    assert int(ohtani["season"]) == 0


def test_parse_starting_nine_prefers_non_pitcher_for_duplicate_00_slot() -> None:
    players = {
        "ID900001": _player(900001, "100", "P"),
        "ID900002": _player(900002, "100", "DH"),
    }

    parsed = parse_starting_nine(_payload(home_players=players), game_pk=800001)

    assert len(parsed) == 1
    assert int(parsed.iloc[0]["game_pk"]) == 800001
    assert int(parsed.iloc[0]["batter_id"]) == 900002
    assert int(parsed.iloc[0]["slot_is_pitcher"]) == 0
