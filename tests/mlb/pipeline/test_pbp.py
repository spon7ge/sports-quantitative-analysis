import gzip
import json
from dataclasses import replace

import pandas as pd
from src.mlb.config import load_config
from src.mlb.pipeline import pbp
from src.mlb.pipeline.pbp import ingest_play_by_play, parse_play_by_play
from src.mlb.schemas import BATTER_PA_COLUMNS
from src.mlb.storage import MlbStore


def test_parse_play_by_play_retains_events_and_wall_clock_ingestion() -> None:
    payload = """
    {
      "gamePk": 746327,
      "liveData": {
        "plays": {
          "allPlays": [
            {
              "atBatIndex": 0,
              "result": {"eventType": "strikeout"},
              "about": {"startTime": "2024-04-01T17:05:00Z"},
              "matchup": {
                "batter": {"id": 200},
                "pitcher": {"id": 300},
                "pitchHand": {"code": "R"},
                "batSide": {"code": "L"}
              }
            },
            {
              "atBatIndex": 1,
              "result": {"event": "walk"},
              "about": {"endTime": "2024-04-01T17:11:00Z"},
              "matchup": {
                "batter": {"id": 400},
                "pitcher": {"id": 200, "p_throws": "L"},
                "batSide": {"code": "R"}
              }
            }
          ]
        }
      }
    }
    """
    people = pd.DataFrame(
        [
            {"mlb_id": 200, "bats": "S"},
            {"mlb_id": 400, "bats": "R"},
        ]
    )
    ingested_at = pd.Timestamp("2026-09-20T23:59:00Z")

    result = parse_play_by_play(
        payload,
        "snapshot-1",
        ingested_at,
        people=people,
    )

    assert result.columns.tolist() == list(BATTER_PA_COLUMNS)
    assert result["pa_id"].tolist() == ["746327_0", "746327_1"]
    assert result["event_type"].tolist() == ["strikeout", "walk"]
    assert "is_strikeout" not in result.columns
    assert result["pitcher_hand"].tolist() == ["R", "L"]
    assert result["batter_bats"].tolist() == ["S", "R"]
    assert result["batter_stand"].tolist() == ["L", "R"]
    assert result["is_pitcher_in_game"].tolist() == [1, 0]
    assert result["event_time_imputed"].tolist() == [0, 0]
    assert result["ingested_at_utc"].tolist() == [ingested_at, ingested_at]
    assert result.loc[0, "ingested_at_utc"] != result.loc[0, "event_time_utc"]
    assert str(result["event_time_utc"].dtype) == "datetime64[us, UTC]"
    assert str(result["ingested_at_utc"].dtype) == "datetime64[us, UTC]"


def test_parse_play_by_play_imputes_missing_event_time_four_hours_late() -> None:
    payload = """
    {
      "gameData": {"game": {"pk": 746328}},
      "allPlays": [
        {
          "atBatIndex": 7,
          "result": {"eventType": "hit_by_pitch"},
          "about": {},
          "matchup": {
            "batter": {"id": 500},
            "pitcher": {"id": 600},
            "pitchHand": {"code": "R"},
            "batSide": {"code": "R"}
          }
        }
      ]
    }
    """
    scheduled_start = pd.Timestamp("2024-04-02T18:00:00Z")

    result = parse_play_by_play(
        payload,
        "snapshot-2",
        pd.Timestamp("2026-09-21T00:00:00Z"),
        scheduled_start=scheduled_start,
    )

    assert result.loc[0, "event_time_imputed"] == 1
    assert result.loc[0, "event_time_utc"] == scheduled_start + pd.Timedelta(hours=4)
    assert result.loc[0, "batter_bats"] == ""


def test_reingest_keeps_latest_wall_clock(tmp_path, monkeypatch) -> None:
    config = replace(
        load_config(),
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
    )
    game_pk = 746327
    raw_path = config.raw_dir / "mlb_pbp" / f"{game_pk}.json.gz"
    raw_path.parent.mkdir(parents=True)

    def payload(plays: list[dict]) -> bytes:
        return json.dumps({"gamePk": game_pk, "allPlays": plays}).encode()

    first_plays = [
        {
            "atBatIndex": index,
            "result": {"eventType": event_type},
            "about": {"startTime": f"2024-04-01T17:0{index}:00Z"},
            "matchup": {
                "batter": {"id": 200 + index},
                "pitcher": {"id": 300},
            },
        }
        for index, event_type in enumerate(("field_out", "single"))
    ]
    with gzip.open(raw_path, "wb") as handle:
        handle.write(payload(first_plays))

    ingested_times = iter(
        [
            pd.Timestamp("2026-09-20T23:59:00Z"),
            pd.Timestamp("2026-09-21T00:01:00Z"),
        ]
    )
    monkeypatch.setattr(pbp, "_now_utc", lambda: next(ingested_times))

    first = ingest_play_by_play(config, game_pks=[game_pk])
    assert len(first) == 2

    updated_play = first_plays[0] | {"result": {"eventType": "strikeout"}}
    with gzip.open(raw_path, "wb") as handle:
        handle.write(payload([updated_play]))
    ingest_play_by_play(config, game_pks=[game_pk])

    stored = MlbStore(config).read_table("batter_pas")
    assert stored["pa_id"].tolist() == [f"{game_pk}_1", f"{game_pk}_0"]
    latest = stored.loc[stored["pa_id"] == f"{game_pk}_0"].iloc[0]
    assert latest["event_type"] == "strikeout"
    assert latest["ingested_at_utc"] == pd.Timestamp("2026-09-21T00:01:00Z")
    assert "is_strikeout" not in stored.columns
