"""Quote evaluation windows: one game, one week, one month, or the full HF span."""

from __future__ import annotations

import pandas as pd
import pytest
from src.mlb.pipeline.hf_tables import (
    assemble_windowed_hf_tables,
    map_quotes_to_starts,
    select_quote_window,
)


def _quotes() -> pd.DataFrame:
    rows = []
    # March 21–27 (first week), March 31, April 2, May 1
    stamps = [
        ("2026-03-21 17:05:00+00:00", "m~week-a", 101, "max meyer"),
        ("2026-03-21 17:05:00+00:00", "m~week-a", 101, "andrew abbott"),
        ("2026-03-22 20:10:00+00:00", "m~week-b", 102, "max meyer"),
        ("2026-03-27 23:40:00+00:00", "m~week-c", 103, "max meyer"),
        ("2026-03-28 00:10:00+00:00", "m~next-week", 104, "max meyer"),
        ("2026-03-31 23:46:00+00:00", "m~month-end", 105, "max meyer"),
        ("2026-04-02 23:10:00+00:00", "m~april", 106, "max meyer"),
        ("2026-05-01 23:10:00+00:00", "m~may", 107, "max meyer"),
    ]
    for start, gid, pk, player in stamps:
        rows.append(
            {
                "game_id": gid,
                "game_pk": pk,
                "player": player,
                "player_key": player,
                "book": "pinnacle",
                "line": 5.5,
                "over_price": 1.91,
                "under_price": 1.91,
                "start_time": pd.Timestamp(start),
                "fetched_at_utc": pd.Timestamp(start) - pd.Timedelta(hours=3),
                "result": 6.0,
            }
        )
    return pd.DataFrame(rows)


def test_default_week_is_seven_days_from_first_quote() -> None:
    picked, meta = select_quote_window(_quotes(), "week")
    assert meta["window"] == "week"
    assert meta["n_quotes"] == 4
    assert set(picked["game_id"]) == {"m~week-a", "m~week-b", "m~week-c"}
    assert "m~next-week" not in set(picked["game_id"])


def test_game_window_keeps_first_game_on_origin_date() -> None:
    picked, meta = select_quote_window(_quotes(), "game")
    assert meta["n_quotes"] == 2
    assert set(picked["game_id"]) == {"m~week-a"}
    assert meta["n_games"] == 1


def test_game_window_can_pin_game_pk() -> None:
    picked, meta = select_quote_window(_quotes(), "game", game_pk=105)
    assert meta["n_quotes"] == 1
    assert int(picked.iloc[0]["game_pk"]) == 105


def test_month_window_is_rest_of_calendar_month() -> None:
    picked, meta = select_quote_window(_quotes(), "month")
    assert meta["n_quotes"] == 6
    assert "m~april" not in set(picked["game_id"])
    assert "m~month-end" in set(picked["game_id"])


def test_month_window_honors_window_start() -> None:
    picked, meta = select_quote_window(_quotes(), "month", window_start="2026-04-01")
    assert meta["n_quotes"] == 1
    assert picked.iloc[0]["game_id"] == "m~april"


def test_full_window_keeps_every_quote() -> None:
    quotes = _quotes()
    picked, meta = select_quote_window(quotes, "full")
    assert meta["n_quotes"] == len(quotes)
    assert meta["n_games"] == quotes["game_pk"].nunique()


def test_one_week_alias_matches_week() -> None:
    a, _ = select_quote_window(_quotes(), "1 week")
    b, _ = select_quote_window(_quotes(), "week")
    assert list(a["game_id"]) == list(b["game_id"])


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window must be one of"):
        select_quote_window(_quotes(), "season")


def test_assemble_windowed_tables_does_not_need_store(mlb_config) -> None:
    quotes = _quotes().drop(columns=["game_pk"])
    start = pd.Timestamp("2026-03-21 17:05:00", tz="UTC")
    starts = pd.DataFrame(
        {
            "pitcher_id": [676974, 671096, 676974],
            "game_pk": [101, 101, 102],
            "game_date": ["2026-03-21", "2026-03-21", "2026-03-22"],
            "season": [2026, 2026, 2026],
            "strikeouts": [6, 5, 7],
            "batters_faced": [22, 21, 23],
            "pitches": [90, 88, 91],
            "outs": 15,
            "role": "starter",
            "is_home": [1, 0, 1],
            "opponent_team_id": 115,
            "team_id": [146, 113, 146],
            "venue_id": 1,
            "pitcher_hand": ["R", "L", "R"],
            "scheduled_start_utc": [
                start,
                start,
                start + pd.Timedelta(days=1),
            ],
            "event_time_utc": [
                start,
                start,
                start + pd.Timedelta(days=1),
            ],
            "ingested_at_utc": [
                start,
                start,
                start + pd.Timedelta(days=1),
            ],
            "doubleheader": 0,
            "player_key": ["max meyer", "andrew abbott", "max meyer"],
        }
    )
    people = pd.DataFrame(
        {
            "mlb_id": [676974, 671096],
            "key_mlbam": [676974, 671096],
            "key_fangraphs": [0, 0],
            "key_bbref": ["", ""],
            "key_retro": ["", ""],
            "name": ["Max Meyer", "Andrew Abbott"],
            "bats": ["R", "L"],
            "throws": ["R", "L"],
            "is_pitcher": [1, 1],
            "player_key": ["max meyer", "andrew abbott"],
        }
    )
    schedule = pd.DataFrame(
        {
            "game_pk": [101, 102],
            "scheduled_start_utc": [start, start + pd.Timedelta(days=1)],
            "status": ["Final", "Final"],
            "home_team_id": [146, 146],
            "away_team_id": [115, 115],
            "venue_id": [1, 1],
            "doubleheader": [0, 0],
            "probable_home_pitcher_id": [676974, 676974],
            "probable_away_pitcher_id": [671096, 0],
            "official_date": ["2026-03-21", "2026-03-22"],
        }
    )
    mapped = map_quotes_to_starts(quotes, starts)
    assert not mapped.empty
    tables, meta = assemble_windowed_hf_tables(
        mlb_config,
        window="game",
        quotes=quotes,
        starts=starts,
        people=people,
        schedule=schedule,
    )
    assert meta["window"] == "game"
    assert meta["n_mapped"] == 2
    assert tables["market_quotes"].empty is False
    assert set(tables["pitcher_starts"]["pitcher_id"]) <= {676974, 671096}
    assert "pitch_events" in tables
    assert tables["pitch_events"].empty
