"""Statcast and schedule parser tests."""

from __future__ import annotations

from datetime import UTC, datetime

from src.mlb.config import load_config
from src.mlb.pipeline.parse import parse_schedule, parse_statcast
from src.mlb.schemas import (
    GAME_VERSION_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
)

SAVANT_CSV = """game_pk,pitcher,batter,pitch_type,release_speed,pfx_x,pfx_z,plate_x,plate_z,description,events,stand,p_throws,at_bat_number,pitch_number,game_date
123,111001,222001,FF,95.0,0.1,1.0,0.2,2.5,called_strike,,L,L,1,1,2017-04-07
123,111001,222001,FF,96.0,0.1,1.0,0.2,2.5,swinging_strike,strikeout,L,L,1,2,2017-04-07
123,111001,222002,SL,88.0,-0.4,0.2,2.0,2.4,foul,,R,L,2,1,2017-04-07
"""


def test_parse_statcast_from_fixture_csv_has_required_columns() -> None:
    config = load_config()
    payload = (config.fixture_dir / "raw" / "statcast_sample.csv").read_text()
    ingested = datetime(2017, 4, 8, tzinfo=UTC)
    pitches, pas = parse_statcast(payload, "snap-test", ingested)
    for column in PITCH_EVENT_COLUMNS:
        assert column in pitches.columns
    for column in PLATE_APPEARANCE_COLUMNS:
        assert column in pas.columns
    assert len(pitches) > 0
    assert len(pas) > 0
    assert pitches["snapshot_id"].eq("snap-test").all()


def test_parse_statcast_derives_flags_from_savant_description() -> None:
    ingested = datetime(2017, 4, 8, tzinfo=UTC)
    pitches, pas = parse_statcast(SAVANT_CSV, "snap-savant", ingested)
    first = pitches.sort_values(["at_bat_number", "pitch_number"]).iloc[0]
    second = pitches.sort_values(["at_bat_number", "pitch_number"]).iloc[1]
    chase_pitch = pitches.loc[pitches["at_bat_number"] == 2].iloc[0]

    assert int(first["is_called_strike"]) == 1
    assert int(first["is_swing"]) == 0
    assert int(first["is_in_zone"]) == 1
    assert int(second["is_whiff"]) == 1
    assert int(second["is_swing"]) == 1
    assert int(chase_pitch["is_swing"]) == 1
    assert int(chase_pitch["is_in_zone"]) == 0
    assert int(chase_pitch["is_chase"]) == 1
    assert pitches["pitcher_id"].iloc[0] == 111001
    assert pitches["batter_stand"].iloc[0] == "L"
    assert pitches["pitcher_hand"].iloc[0] == "L"

    strikeout_pa = pas.loc[pas["at_bat_number"] == 1].iloc[0]
    assert int(strikeout_pa["is_strikeout"]) == 1
    assert int(strikeout_pa["pitches_in_pa"]) == 2
    assert len(pas) == 2


def test_parse_schedule_from_fixture_json() -> None:
    config = load_config()
    payload = (config.fixture_dir / "raw" / "schedule.json").read_text()
    ingested = datetime(2017, 4, 5, tzinfo=UTC)
    versions = parse_schedule(payload, "snap-sched", ingested)
    for column in GAME_VERSION_COLUMNS:
        assert column in versions.columns
    assert versions["snapshot_id"].eq("snap-sched").all()
    assert 500001 in set(versions["game_pk"].astype(int))
