import pandas as pd
from src.mlb.pipeline.pbp import parse_play_by_play
from src.mlb.schemas import BATTER_PA_COLUMNS


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
