"""Switch-hitter stand and pitcher-hand tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from src.mlb.pipeline.features import build_feature_rows, resolved_batter_stand
from src.mlb.schemas import (
    ID_MAP_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PITCHER_START_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
    PREGAME_SNAPSHOT_COLUMNS,
    coerce_frame,
    empty_frame,
)


def test_switch_hitter_stands_right_vs_lhp() -> None:
    assert resolved_batter_stand("S", "L") == "R"
    assert resolved_batter_stand("S", "R") == "L"
    assert resolved_batter_stand("L", "L") == "L"
    assert resolved_batter_stand("S", "L", pitch_stand="L") == "L"


def test_pitcher_throws_l_copied_from_pregame(fixture_tables, mlb_config) -> None:
    features = build_feature_rows(fixture_tables, mlb_config)
    pregame = fixture_tables["pregame_snapshots"]
    merged = features.merge(
        pregame[["pregame_id", "pitcher_hand"]],
        on="pregame_id",
        how="left",
    )
    assert (merged.loc[merged["pitcher_hand"] == "L", "pitcher_throws_L"] == 1).all()
    assert (merged.loc[merged["pitcher_hand"] == "R", "pitcher_throws_L"] == 0).all()


def test_opponent_k_rate_uses_matching_hand_only(mlb_config) -> None:
    cutoff = datetime(2018, 6, 1, 21, tzinfo=UTC)
    ingested = datetime(2017, 1, 1, tzinfo=UTC)
    pas = [
        {
            "game_pk": 10,
            "pa_id": "10-1",
            "at_bat_number": 1,
            "pitcher_id": 111002,
            "batter_id": 222001,
            "result": "strikeout",
            "is_strikeout": 1,
            "batting_slot": 1,
            "pitcher_hand": "L",
            "batter_stand": "R",
            "tto_number": 1,
            "pitches_in_pa": 4,
            "event_time_utc": datetime(2018, 5, 1, 23, tzinfo=UTC),
            "ingested_at_utc": ingested,
            "snapshot_id": "s",
        },
        {
            "game_pk": 11,
            "pa_id": "11-1",
            "at_bat_number": 1,
            "pitcher_id": 111002,
            "batter_id": 222001,
            "result": "out",
            "is_strikeout": 0,
            "batting_slot": 1,
            "pitcher_hand": "R",
            "batter_stand": "L",
            "tto_number": 1,
            "pitches_in_pa": 3,
            "event_time_utc": datetime(2018, 5, 2, 23, tzinfo=UTC),
            "ingested_at_utc": ingested,
            "snapshot_id": "s",
        },
    ]
    starts = [
        {
            "pitcher_id": 111002,
            "game_pk": 10,
            "game_date": "2018-05-01",
            "season": 2018,
            "strikeouts": 1,
            "batters_faced": 1,
            "pitches": 4,
            "outs": 1,
            "role": "starter",
            "is_home": 1,
            "opponent_team_id": 136,
            "team_id": 133,
            "venue_id": 1000,
            "pitcher_hand": "L",
            "scheduled_start_utc": datetime(2018, 5, 1, 23, tzinfo=UTC),
            "event_time_utc": datetime(2018, 5, 1, 23, tzinfo=UTC),
            "ingested_at_utc": ingested,
            "doubleheader": 0,
        },
        {
            "pitcher_id": 111002,
            "game_pk": 11,
            "game_date": "2018-05-02",
            "season": 2018,
            "strikeouts": 0,
            "batters_faced": 1,
            "pitches": 3,
            "outs": 1,
            "role": "starter",
            "is_home": 1,
            "opponent_team_id": 136,
            "team_id": 133,
            "venue_id": 1000,
            "pitcher_hand": "R",
            "scheduled_start_utc": datetime(2018, 5, 2, 23, tzinfo=UTC),
            "event_time_utc": datetime(2018, 5, 2, 23, tzinfo=UTC),
            "ingested_at_utc": ingested,
            "doubleheader": 0,
        },
    ]
    pregame = {
        "pregame_id": "hand-test",
        "game_pk": 12,
        "pitcher_id": 111001,
        "prediction_cutoff_utc": cutoff,
        "forecast_horizon_hours": 2.0,
        "scheduled_start_utc": datetime(2018, 6, 1, 23, tzinfo=UTC),
        "game_date": "2018-06-01",
        "season": 2018,
        "starter_state": "probable",
        "lineup_state": "team_fallback",
        "roster_state": "active",
        "is_opener": 0,
        "is_il_return": 0,
        "is_restricted": 0,
        "is_home": 1,
        "team_id": 133,
        "opponent_team_id": 136,
        "venue_id": 1000,
        "pitcher_hand": "L",
        "expected_rhb_share": 0.5,
        "lineup_batter_ids_json": "[]",
        "source_snapshot_ids_json": "[]",
    }
    id_map = {
        "mlb_id": 222001,
        "key_mlbam": 222001,
        "key_fangraphs": 1,
        "key_bbref": "bat",
        "key_retro": "B",
        "name": "Switch",
        "bats": "S",
        "throws": "R",
    }
    tables = {
        "pregame_snapshots": coerce_frame(
            pd.DataFrame([pregame]), PREGAME_SNAPSHOT_COLUMNS
        ),
        "pitcher_starts": coerce_frame(pd.DataFrame(starts), PITCHER_START_COLUMNS),
        "plate_appearances": coerce_frame(pd.DataFrame(pas), PLATE_APPEARANCE_COLUMNS),
        "pitch_events": empty_frame(PITCH_EVENT_COLUMNS),
        "game_versions": pd.DataFrame(),
        "id_map": coerce_frame(pd.DataFrame([id_map]), ID_MAP_COLUMNS),
    }
    features = build_feature_rows(tables, mlb_config)
    assert features["n_eff_opp_k"].iloc[0] == 1
    # Only the LHP PA (a strikeout) should count vs this LHP.
    assert features["opp_k_rate_vs_hand_shrunk"].iloc[0] > 0.2
    assert resolved_batter_stand("S", "L") == "R"
