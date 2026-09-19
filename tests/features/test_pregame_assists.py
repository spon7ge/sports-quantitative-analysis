"""Pregame assists features stay frozen at the quote as-of date."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.assists import CURRENT_ASSISTS_FEATURES
from src.features.assists.pregame import (
    appearances_before,
    build_pregame_assists_features,
)


def _row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    ast: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    team_pace: float = 90.0,
    opp_ast: float = 20.0,
) -> dict:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": "2024-25",
        "matchup": "DET vs. CLE",
        "minutes": minutes,
        "min": minutes,
        "min_sec": "20:00",
        "start_position": "G",
        "ast": ast,
        "assists": ast,
        "team_ast": 20.0,
        "team_fgm": 40.0,
        "team_pace": team_pace,
        "opp_ast": opp_ast,
        "opp_pace": team_pace,
        "pass": 10.0,
        "tchs": 20.0,
        "sast": 1.0,
        "ftast": 1.0,
    }


class AsOfCutoffTests(unittest.TestCase):
    def test_empty_candidates_include_float64_feature_columns(self) -> None:
        history = pd.DataFrame(
            [_row(1, "2026-04-01", 32, ast=8, game_id=1)]
        )
        candidates = pd.DataFrame(
            columns=["game_id", "player_id", "game_date"]
        )

        featured = build_pregame_assists_features(
            history, candidates, as_of="2026-05-09"
        )

        self.assertIsNot(featured, candidates)
        self.assertTrue(featured.empty)
        for column in CURRENT_ASSISTS_FEATURES:
            self.assertIn(column, featured.columns)
            self.assertEqual(featured[column].dtype, np.float64)

    def test_drops_games_on_or_after_as_of(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2026-05-08", 30, ast=5, game_id=1),
                _row(1, "2026-05-09", 40, ast=99, game_id=2),
            ]
        )
        prior = appearances_before(frame, "2026-05-09")
        self.assertEqual(len(prior), 1)
        self.assertEqual(int(prior.iloc[0]["game_id"]), 1)

    def test_candidate_ignores_same_day_box_and_keeps_index(self) -> None:
        history = pd.DataFrame(
            [
                _row(1, "2026-04-01", 32, ast=8, game_id=1),
                _row(1, "2026-05-09", 40, ast=99, game_id=2),
            ]
        )
        candidate = pd.DataFrame(
            [_row(1, "2026-05-09", 0, ast=0, game_id=99)]
        )
        candidate.index = pd.Index([42])
        featured = build_pregame_assists_features(
            history, candidate, as_of="2026-05-09"
        )
        self.assertEqual(list(featured.index), [42])
        self.assertEqual(featured.iloc[0]["ast_mean_10"], 8.0)
        self.assertTrue(
            pd.isna(featured.iloc[0]["predicted_minutes_oof"])
        )

    def test_later_candidate_does_not_see_earlier_candidate(self) -> None:
        history = pd.DataFrame(
            [_row(1, "2026-04-01", 32, ast=8, game_id=1)]
        )
        candidates = pd.DataFrame(
            [
                _row(1, "2026-05-09", 40, ast=50, game_id=9),
                _row(1, "2026-05-11", 40, ast=50, game_id=11),
            ]
        )
        featured = build_pregame_assists_features(
            history, candidates, as_of="2026-05-09"
        )
        later = featured.loc[featured["game_id"].eq(11)].iloc[0]
        self.assertEqual(later["ast_mean_10"], 8.0)

    def test_one_sided_slate_still_gets_opponent_history(self) -> None:
        history = pd.DataFrame(
            [
                _row(
                    1, "2026-04-01", 20, ast=4, game_id=1,
                    team_id=100, opp_team_id=200,
                    team_pace=90.0, opp_ast=5.0,
                ),
                _row(
                    2, "2026-04-01", 20, ast=8, game_id=1,
                    team_id=200, opp_team_id=100,
                    team_pace=110.0, opp_ast=40.0,
                ),
            ]
        )
        candidate = pd.DataFrame(
            [
                _row(
                    1, "2026-04-03", np.nan, ast=np.nan, game_id=9,
                    team_id=100, opp_team_id=200,
                    team_pace=np.nan, opp_ast=np.nan,
                )
            ]
        )
        featured = build_pregame_assists_features(
            history, candidate, as_of="2026-04-03"
        )
        self.assertEqual(
            featured.iloc[0]["opponent_team_ast_allowed_mean_10"],
            40.0,
        )
        self.assertNotEqual(
            featured.iloc[0]["team_pace_mean_10"],
            featured.iloc[0]["opponent_team_pace_mean_10"],
        )
