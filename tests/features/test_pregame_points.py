"""Pregame points features stay frozen at the quote as-of date."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.points.pregame import (
    appearances_before,
    build_pregame_points_features,
)


def _clock(minutes: float) -> str:
    whole = int(minutes)
    seconds = int(round((minutes - whole) * 60))
    return f"{whole}:{seconds:02d}"


def _row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    pts: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    player_name: str = "Cade Cunningham",
) -> dict:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "team_id": team_id,
        "team_abbreviation": "DET",
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": "2025-26",
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
    }


class AsOfCutoffTests(unittest.TestCase):
    def test_drops_games_on_or_after_as_of(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2026-05-08", 30, pts=20, game_id=1),
                _row(1, "2026-05-09", 40, pts=99, game_id=2),
            ]
        )
        prior = appearances_before(frame, "2026-05-09")
        self.assertEqual(len(prior), 1)
        self.assertEqual(int(prior.iloc[0]["game_id"]), 1)

    def test_candidate_features_ignore_same_day_box_scores(self) -> None:
        history = pd.DataFrame(
            [
                _row(1, "2026-04-01", 32, pts=18, game_id=1),
                _row(1, "2026-05-09", 40, pts=99, game_id=2),
            ]
        )
        candidate = pd.DataFrame(
            [
                {
                    **_row(1, "2026-05-09", 0, pts=0, game_id=99),
                    "minutes": np.nan,
                    "min": np.nan,
                    "min_sec": "",
                    "pts": np.nan,
                    "start_position": "",
                    "season_type": "Playoffs",
                }
            ]
        )
        featured = build_pregame_points_features(
            history,
            candidate,
            as_of="2026-05-09",
        )
        self.assertEqual(len(featured), 1)
        self.assertEqual(featured.iloc[0]["pts_lag_1"], 18)
        self.assertEqual(featured.iloc[0]["min_lag_1"], 32)
        self.assertNotEqual(featured.iloc[0]["pts_lag_1"], 99)
