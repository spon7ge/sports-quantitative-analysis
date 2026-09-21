"""Starting-nine parser tests."""

from __future__ import annotations

import json
import math

import pandas as pd
from src.mlb.config import load_config
from src.mlb.pipeline.lineup_slots import (
    freeze_lineup_slot_rates,
    parse_starting_nine,
)
from src.mlb.pipeline.parse import parse_lineups
from src.mlb.schemas import LINEUP_SLOT_COLUMNS


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


def test_freeze_rates_skips_imputed() -> None:
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    slots = pd.DataFrame(
        [
            {
                "game_pk": 745001,
                "team_id": 118,
                "side": "home",
                "slot": 1,
                "batter_id": 101,
                "slot_is_pitcher": 0,
                "season": 2026,
            }
        ]
    )
    batter_pas = pd.DataFrame(
        [
            (1, 101, "R", "R", "strikeout", cutoff - pd.Timedelta(days=1), 0, 0),
            (2, 101, "L", "R", "strikeout", cutoff - pd.Timedelta(days=2), 1, 0),
            (3, 202, "R", "L", "out", cutoff - pd.Timedelta(days=3), 0, 0),
        ],
        columns=[
            "game_pk",
            "batter_id",
            "pitcher_hand",
            "batter_bats",
            "event_type",
            "event_time_utc",
            "event_time_imputed",
            "is_pitcher_in_game",
        ],
    )
    people = pd.DataFrame([{"mlb_id": 101, "bats": "R"}])

    frozen = freeze_lineup_slot_rates(
        slots,
        batter_pas,
        people,
        load_config(),
        cutoff=cutoff,
        opposing_pitcher_hand="R",
        vs_pitcher_id=9001,
    )

    assert list(frozen.columns) == list(LINEUP_SLOT_COLUMNS)
    assert math.isfinite(float(frozen.iloc[0]["k_pa_vs_hand_shrunk_365"]))
    assert float(frozen.iloc[0]["pa_all_365"]) == 1.0
    assert float(frozen.iloc[0]["pa_vs_hand_365"]) == 1.0
    assert float(frozen.iloc[0]["k_pa_overall_shrunk_365"]) == 0.5
    assert int(frozen.iloc[0]["vs_pitcher_id"]) == 9001
    assert str(frozen.iloc[0]["rate_version"]).startswith("kpa_")
    assert {column: str(dtype) for column, dtype in frozen.dtypes.items()} == {
        column: dtype for column, dtype in LINEUP_SLOT_COLUMNS.items()
    }

    unknown_hand = freeze_lineup_slot_rates(
        slots,
        batter_pas,
        people,
        load_config(),
        cutoff=cutoff,
        opposing_pitcher_hand="",
        vs_pitcher_id=None,
    )
    assert math.isfinite(float(unknown_hand.iloc[0]["k_pa_overall_shrunk_365"]))


def test_freeze_league_rates_use_trailing_365_days() -> None:
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    slots = pd.DataFrame(
        [
            {
                "game_pk": 745001,
                "team_id": 118,
                "side": "home",
                "slot": 1,
                "batter_id": 101,
                "slot_is_pitcher": 0,
                "season": 2026,
            }
        ]
    )
    columns = [
        "game_pk",
        "batter_id",
        "pitcher_hand",
        "batter_bats",
        "event_type",
        "event_time_utc",
        "event_time_imputed",
        "is_pitcher_in_game",
    ]
    baseline = pd.DataFrame(
        [
            (1, 101, "R", "R", "strikeout", cutoff - pd.Timedelta(days=1), 0, 0),
            (2, 202, "R", "L", "out", cutoff - pd.Timedelta(days=2), 0, 0),
        ],
        columns=columns,
    )
    people = pd.DataFrame([{"mlb_id": 101, "bats": "R"}])

    def frozen_overall(extra_age_days: int) -> float:
        extra = pd.DataFrame(
            [
                (
                    3,
                    303,
                    "R",
                    "R",
                    "strikeout",
                    cutoff - pd.Timedelta(days=extra_age_days),
                    0,
                    0,
                )
            ],
            columns=columns,
        )
        frozen = freeze_lineup_slot_rates(
            slots,
            pd.concat([baseline, extra], ignore_index=True),
            people,
            load_config(),
            cutoff=cutoff,
            opposing_pitcher_hand="R",
            vs_pitcher_id=9001,
        )
        return float(frozen.iloc[0]["k_pa_overall_shrunk_365"])

    outside_window = frozen_overall(366)
    inside_window = frozen_overall(364)

    assert outside_window == 0.5
    assert inside_window == 2.0 / 3.0
