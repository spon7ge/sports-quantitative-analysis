"""Starting-nine parser tests."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pandas as pd
import pytest
from src.mlb.config import load_config
from src.mlb.pipeline.lineup_slots import (
    BOXSCORE_00_LEAD,
    assert_lineup_coverage,
    freeze_lineup_slot_rates,
    ingest_lineup_slots,
    is_dummy_dh2_start,
    parse_starting_nine,
    write_lineup_coverage,
)
from src.mlb.pipeline.parse import parse_lineups
from src.mlb.schemas import (
    BATTER_PA_COLUMNS,
    GAME_VERSION_COLUMNS,
    ID_MAP_COLUMNS,
    LINEUP_SLOT_COLUMNS,
    coerce_frame,
)
from src.mlb.storage import MlbStore


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


def _complete_payload(game_pk: int = 745001, season: str = "2026") -> str:
    return _payload(
        home_players={
            f"ID{100 + slot}": _player(100 + slot, f"{slot}00", "OF")
            for slot in range(1, 10)
        },
        away_players={
            f"ID{200 + slot}": _player(200 + slot, f"{slot}00", "OF")
            for slot in range(1, 10)
        },
        season=season,
    ).replace('"gamePk": 745001', f'"gamePk": {game_pk}').replace(
        '"pk": 745001', f'"pk": {game_pk}'
    )


def _home_nine_payload(game_pk: int, season: str = "2026") -> str:
    payload = _payload(
        home_players={
            f"ID{100 + slot}": _player(100 + slot, f"{slot}00", "OF")
            for slot in range(1, 10)
        },
        season=season,
    )
    return payload.replace('"gamePk": 745001', f'"gamePk": {game_pk}').replace(
        '"pk": 745001', f'"pk": {game_pk}'
    )


def _ingest_config(tmp_path) -> object:
    return replace(
        load_config(),
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        forecast_horizon_hours=3,
    )


def _seed_ingest_tables(config, *, game_pk: int, start: object, doubleheader: int = 1):
    store = MlbStore(config)
    versions = pd.DataFrame(
        [
            {
                "game_pk": game_pk,
                "scheduled_start_utc": start,
                "status": "Scheduled",
                "home_team_id": 118,
                "away_team_id": 139,
                "venue_id": 1,
                "doubleheader": doubleheader,
                "probable_home_pitcher_id": 901,
                "probable_away_pitcher_id": 902,
                "valid_from_utc": "2026-01-01T00:00:00Z",
                "valid_to_utc": pd.NaT,
                "snapshot_id": "schedule",
            }
        ]
    )
    store.write_table("game_versions", coerce_frame(versions, GAME_VERSION_COLUMNS))
    people = pd.DataFrame(
        [
            {"mlb_id": player_id, "key_mlbam": player_id, "key_fangraphs": pd.NA,
             "key_bbref": "", "key_retro": "", "name": "", "bats": "R", "throws": "R"}
            for player_id in [*range(101, 110), *range(201, 210), 901, 902]
        ]
    )
    store.write_table("id_map", coerce_frame(people, ID_MAP_COLUMNS))
    cutoff = pd.Timestamp("2026-07-01T16:00:00Z")
    pas = pd.DataFrame(
        [
            {
                "pa_id": f"pa-{player_id}",
                "game_pk": 1,
                "at_bat_index": player_id,
                "batter_id": player_id,
                "pitcher_id": 999,
                "pitcher_hand": "R",
                "batter_bats": "R",
                "batter_stand": "R",
                "event_type": "strikeout" if player_id % 2 else "field_out",
                "event_time_utc": cutoff - pd.Timedelta(days=1),
                "event_time_imputed": 0,
                "is_pitcher_in_game": 0,
                "ingested_at_utc": cutoff,
                "snapshot_id": "pbp",
            }
            for player_id in [*range(101, 110), *range(201, 210)]
        ]
    )
    store.write_table("batter_pas", coerce_frame(pas, BATTER_PA_COLUMNS))


def test_ingest_boxscore_stamp_is_24h_and_observed_false(tmp_path) -> None:
    game_pk = 745001
    start = pd.Timestamp("2026-07-01T19:00:00Z")
    config = _ingest_config(tmp_path)
    _seed_ingest_tables(config, game_pk=game_pk, start=start)
    fixture = config.raw_dir / "mlb_lineup_slots" / f"{game_pk}.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(_complete_payload(game_pk))

    written = ingest_lineup_slots(
        config, game_pks=[game_pk], provenance="boxscore_00", http=None
    )

    assert len(written) == 18
    assert set(written["ingested_at_utc"]) == {start - BOXSCORE_00_LEAD}
    assert (written["ingested_at_utc"] < start - pd.Timedelta(hours=3)).all()
    assert written["observed_before_cutoff"].eq(0).all()


def test_ingest_resume_uses_stored_slots_for_coverage(tmp_path) -> None:
    game_pk = 745001
    config = _ingest_config(tmp_path)
    _seed_ingest_tables(
        config,
        game_pk=game_pk,
        start=pd.Timestamp("2026-07-01T19:00:00Z"),
    )
    fixture = config.raw_dir / "mlb_lineup_slots" / f"{game_pk}.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(_complete_payload(game_pk))

    first = ingest_lineup_slots(
        config, game_pks=[game_pk], provenance="boxscore_00", http=None
    )
    second = ingest_lineup_slots(
        config, game_pks=[game_pk], provenance="boxscore_00", http=None
    )

    assert len(first) == 18
    assert second.empty
    assert len(MlbStore(config).read_table("lineup_slots")) == 18


def test_dummy_dh2_detection_and_boxscore_skip(tmp_path, monkeypatch) -> None:
    assert is_dummy_dh2_start(pd.NaT, pd.Timestamp("2026-07-01T17:00:00Z"))
    assert is_dummy_dh2_start(
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-01T17:00:00Z"),
    )

    config = _ingest_config(tmp_path)
    dummy_start = pd.Timestamp("2026-07-01T17:00:00Z")
    assert is_dummy_dh2_start(dummy_start, dummy_start)
    _seed_ingest_tables(
        config, game_pk=745002, start=dummy_start, doubleheader=2
    )
    store = MlbStore(config)
    versions = store.read_table("game_versions")
    game1 = versions.iloc[0].copy()
    game1["game_pk"] = 745001
    game1["doubleheader"] = 1
    store.write_table(
        "game_versions",
        coerce_frame(
            pd.concat([versions, game1.to_frame().T], ignore_index=True),
            GAME_VERSION_COLUMNS,
        ),
    )
    fixture = config.raw_dir / "mlb_lineup_slots" / "745002.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(_complete_payload(745002))

    with pytest.raises(ValueError, match="zero slots"):
        ingest_lineup_slots(
            config, game_pks=[745002], provenance="boxscore_00", http=None
        )
    coverage = json.loads(
        (config.artifact_dir / "lineup_coverage_2026.json").read_text()
    )
    assert coverage["n_skip_dh2_dummy_start"] == 1

    monkeypatch.setattr(
        "src.mlb.pipeline.lineup_slots._now_utc",
        lambda: pd.Timestamp("2026-07-01T13:00:00Z"),
    )
    fixture.write_text(_home_nine_payload(745002))
    live = ingest_lineup_slots(
        config, game_pks=[745002], provenance="live_feed", http=None
    )
    assert len(live) == 9
    assert live["observed_before_cutoff"].eq(1).all()
    assert len(MlbStore(config).read_table("lineup_slots")) == 9


def test_nat_dh2_skips_boxscore_but_live_uses_corrected_start(
    tmp_path, monkeypatch
) -> None:
    game_pk = 745003
    config = _ingest_config(tmp_path)
    _seed_ingest_tables(config, game_pk=game_pk, start=pd.NaT, doubleheader=2)
    store = MlbStore(config)
    versions = store.read_table("game_versions")
    corrected = versions.iloc[0].copy()
    corrected["scheduled_start_utc"] = pd.Timestamp("2026-07-01T19:00:00Z")
    corrected["valid_from_utc"] = pd.Timestamp("2026-06-01T00:00:00Z")
    store.write_table(
        "game_versions",
        coerce_frame(
            pd.concat([versions, corrected.to_frame().T], ignore_index=True),
            GAME_VERSION_COLUMNS,
        ),
    )
    fixture = config.raw_dir / "mlb_lineup_slots" / f"{game_pk}.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(_complete_payload(game_pk))

    with pytest.raises(ValueError, match="zero slots"):
        ingest_lineup_slots(
            config, game_pks=[game_pk], provenance="boxscore_00", http=None
        )
    coverage = json.loads(
        (config.artifact_dir / "lineup_coverage_2026.json").read_text()
    )
    assert coverage["n_skip_dh2_dummy_start"] == 1
    assert MlbStore(config).read_table("lineup_slots").empty

    monkeypatch.setattr(
        "src.mlb.pipeline.lineup_slots._now_utc",
        lambda: pd.Timestamp("2026-07-01T15:00:00Z"),
    )
    fixture.write_text(_home_nine_payload(game_pk))
    live = ingest_lineup_slots(
        config, game_pks=[game_pk], provenance="live_feed", http=None
    )

    assert len(live) == 9
    assert live["observed_before_cutoff"].eq(1).all()
    assert len(MlbStore(config).read_table("lineup_slots")) == 9


def test_lineup_coverage_passes_healthy_and_fails_low_or_zero() -> None:
    slots = pd.DataFrame(
        [{"game_pk": 1, "team_id": team, "slot": slot, "season": 2026}
         for team in (10, 20) for slot in range(1, 10)]
    )
    skips = pd.DataFrame(columns=["game_pk", "season", "reason"])
    coverage = write_lineup_coverage(slots, skips, season=2026)
    assert coverage["n_slots_written"] == 18
    assert coverage["n_boxscore_ok"] == 1
    assert coverage["n_sides_complete_nine"] == 2
    assert_lineup_coverage(coverage, season=2026)

    with pytest.raises(ValueError):
        assert_lineup_coverage(
            write_lineup_coverage(slots.iloc[0:0], skips, season=2026), season=2026
        )
    low = dict(coverage, n_sides_complete_nine=1)
    with pytest.raises(ValueError):
        assert_lineup_coverage(low, season=2026)
