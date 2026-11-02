"""Last-3 vs season-average trend features exclude the current game."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.points.build import add_points_features


def _row(
    *,
    game: int,
    pts: float,
    minutes: float,
    cfga: float,
    ufga: float,
    usg: float,
) -> dict:
    day = pd.Timestamp("2025-11-01") + pd.Timedelta(days=game)
    return {
        "player_id": 1,
        "player_name": "Test Player",
        "team_id": 10,
        "team_abbreviation": "DET",
        "opp_team_id": 20,
        "game_id": f"g{game}",
        "game_date": day,
        "season_year": "2025-26",
        "season_type": "Regular Season",
        "matchup": "DET vs. CLE",
        "minutes": minutes,
        "min": minutes,
        "min_sec": f"{int(minutes)}:00",
        "start_position": "G",
        "starting": 1.0,
        "pts": pts,
        "fgm": 4,
        "fga": 10,
        "fg3_m": 1,
        "fg3_a": 3,
        "ftm": 2,
        "fta": 2,
        "usg_pct": usg,
        "tchs": 40,
        "pass": 15,
        "pfd": 2,
        "poss": 70,
        "cfga": cfga,
        "ufga": ufga,
        "assists": 3,
        "reb": 4,
        "stl": 1,
        "blk": 0,
        "team_pace": 100.0,
        "team_net_rating": 1.0,
        "team_off_rating": 112.0,
        "team_def_rating": 110.0,
        "team_fga": 90,
        "opp_pace": 100.0,
        "opp_net_rating": -1.0,
        "opp_def_rating": 110.0,
        "player_team_spread": -2.5,
        "game_total": 224.0,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
    }


def test_last3_minus_season_trends_exclude_current_game() -> None:
    # Games 0-4 set the history. Game 5 is the row we inspect.
    history = [
        _row(game=0, pts=10, minutes=20, cfga=4, ufga=6, usg=0.20),
        _row(game=1, pts=12, minutes=22, cfga=5, ufga=7, usg=0.22),
        _row(game=2, pts=14, minutes=24, cfga=6, ufga=8, usg=0.24),
        _row(game=3, pts=20, minutes=30, cfga=8, ufga=10, usg=0.30),
        _row(game=4, pts=22, minutes=32, cfga=9, ufga=11, usg=0.32),
        _row(game=5, pts=99, minutes=99, cfga=99, ufga=99, usg=0.99),
    ]
    featured = add_points_features(pd.DataFrame(history))
    row = featured.iloc[-1]

    # Season means through games 0-4 (current game excluded).
    season_pts = np.mean([10, 12, 14, 20, 22])
    season_min = np.mean([20, 22, 24, 30, 32])
    season_cfga = np.mean([4, 5, 6, 8, 9])
    season_ufga = np.mean([6, 7, 8, 10, 11])
    season_usg = np.mean([0.20, 0.22, 0.24, 0.30, 0.32])
    last3_pts = np.mean([14, 20, 22])
    last3_min = np.mean([24, 30, 32])
    last3_cfga = np.mean([6, 8, 9])
    last3_ufga = np.mean([8, 10, 11])
    last3_usg = np.mean([0.24, 0.30, 0.32])

    assert row["pts_trend"] == np.float64(last3_pts - season_pts)
    assert row["min_trend"] == np.float64(last3_min - season_min)
    assert row["cfga_trend"] == np.float64(last3_cfga - season_cfga)
    assert row["ufga_trend"] == np.float64(last3_ufga - season_ufga)
    assert row["usg_trend"] == np.float64(last3_usg - season_usg)
    # Current-game box scores must not leak into the trends.
    assert row["pts_trend"] != 99 - season_pts
    assert row["usg_trend"] != 0.99 - season_usg
