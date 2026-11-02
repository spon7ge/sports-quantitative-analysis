"""Closing-line features use every game before the slate and none from it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.features.points import add_points_features
from src.features.points.closing import (
    features_before,
    load_closing_points,
    match_player_ids,
)


def _clock(minutes: float) -> str:
    whole = int(minutes)
    seconds = int(round((minutes - whole) * 60))
    return f"{whole}:{seconds:02d}"


def _row(player_id, game_date, minutes, pts, game_id, player_name="Player One"):
    return {
        "player_id": player_id,
        "player_name": player_name,
        "team_id": 100,
        "opp_team_id": 200,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": "2026-27",
        "season_type": "Regular Season",
        "matchup": "DET vs. CLE",
        "minutes": minutes,
        "min": minutes,
        "min_sec": _clock(minutes),
        "start_position": "G",
        "pts": pts,
        "fgm": 5,
        "fga": 12,
        "fg3_m": 2,
        "fg3_a": 4,
        "ftm": 2,
        "fta": 3,
        "usg_pct": 25.0,
        "tchs": 40,
        "pass": 20,
        "assists": 4,
        "reb": 5,
        "stl": 1,
        "blk": 1,
        "team_pace": 100.0,
        "team_net_rating": 1.0,
        "team_off_rating": 110.0,
        "team_def_rating": 108.0,
        "team_fga": 88,
        "opp_pace": 100.0,
        "opp_net_rating": -1.0,
        "opp_def_rating": 108.0,
        "player_team_spread": -3.0,
        "game_total": 220.0,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
        "pfd": 2,
        "cfga": 6,
        "ufga": 6,
    }


def test_window_includes_every_game_before_the_slate_and_not_the_slate():
    history = pd.DataFrame(
        [
            _row(1, "2026-10-20", 10, 10, 1),
            _row(1, "2026-10-22", 20, 20, 2),
            _row(1, "2026-10-24", 30, 30, 3),
            _row(1, "2026-10-26", 40, 99, 4),
        ]
    )

    featured = features_before(history, "2026-10-26", add_points_features)

    assert len(featured) == 1
    row = featured.iloc[0]
    assert row["history_through"] == pd.Timestamp("2026-10-24")
    assert row["pts_rolling_10"] == 20
    assert row["min_lag_1"] == 30
    assert pd.isna(row["pts"])
    assert pd.isna(row["player_team_spread"])
    assert pd.isna(row["starting"])


def test_last_played_game_is_not_shifted_out():
    history = pd.DataFrame(
        [
            _row(1, "2026-10-20", 10, 10, 1),
            _row(1, "2026-10-22", 20, 40, 2),
        ]
    )

    featured = features_before(history, "2026-10-26", add_points_features)

    assert featured.iloc[0]["pts_rolling_3"] == 25
    assert featured.iloc[0]["min_lag_1"] == 20


def test_name_match_strips_suffix_and_accent():
    history = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Jaren Jackson Jr.", "Alperen Sengun", "Jabari Walker"],
        }
    )
    closing = pd.DataFrame(
        {
            "player_name": ["Jaren Jackson", "Alperen Şengün", "Someone Else"],
        }
    )

    matched = match_player_ids(closing, history)

    assert matched["player_id"].tolist()[:2] == [1, 2]
    assert pd.isna(matched.loc[2, "player_id"])


def test_closing_loader_keeps_standard_points_and_the_slate_date(tmp_path: Path):
    payload = {
        "date": "2026-10-26",
        "players": [
            {
                "player": "Cade Cunningham",
                "lines": [
                    {"stat_type": "Points", "line_score": 24.5, "odds_type": "standard"},
                    {"stat_type": "Points", "line_score": 30.5, "odds_type": "demon"},
                    {"stat_type": "Rebounds", "line_score": 5.5, "odds_type": "standard"},
                ],
            }
        ],
    }
    path = tmp_path / "prizepicks_nba_closing_2026-10-26.json"
    path.write_text(json.dumps(payload))

    loaded = load_closing_points(tmp_path)

    assert loaded.to_dict(orient="records") == [
        {
            "slate_date": pd.Timestamp("2026-10-26"),
            "player_name": "Cade Cunningham",
            "line": 24.5,
        }
    ]
    assert np.issubdtype(loaded["slate_date"].dtype, np.datetime64)
