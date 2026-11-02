"""A player has one box row per game, even when the feed splits it."""

from __future__ import annotations

import unittest

import pandas as pd
from src.pipeline.silver.merge import merge_gamelogs


def _teams() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022600124", "0022600124"],
            "team_id": [1610612756, 1610612760],
        }
    )


def _tracking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022600124"],
            "player_id": [1626164],
            "position": ["G"],
        }
    )


class SplitPlayerGameTests(unittest.TestCase):
    def test_zero_minute_split_line_folds_onto_the_team_that_played(self) -> None:
        players = pd.DataFrame(
            {
                "game_id": ["0022600124", "0022600124", "0022600124"],
                "player_id": [1626164, 1626164, 99],
                "player_name": ["Devin Booker", "Devin Booker", "Other"],
                "team_id": [1610612756, 1610612760, 1610612756],
                "team_abbreviation": ["PHX", "OKC", "PHX"],
                "min": [33.79, 0.0, 20.0],
                "pts": [2, 1, 8],
                "ftm": [0, 1, 0],
                "fta": [0, 1, 0],
                "fgm": [1, 0, 3],
                "off_rating": [113.0, 100.0, 110.0],
            }
        )

        result = merge_gamelogs(
            players,
            pd.DataFrame(),
            _teams(),
            pd.DataFrame(),
            _tracking(),
        )
        booker = result.loc[result["player_id"].eq(1626164)]

        self.assertEqual(len(booker), 1)
        self.assertEqual(booker["team_id"].iloc[0], 1610612756)
        self.assertEqual(booker["pts"].iloc[0], 3)
        self.assertEqual(booker["ftm"].iloc[0], 1)
        self.assertEqual(booker["fta"].iloc[0], 1)
        self.assertEqual(booker["fgm"].iloc[0], 1)
        self.assertEqual(booker["off_rating"].iloc[0], 113.0)
        other = result.loc[result["player_id"].eq(99)]
        self.assertEqual(other["pts"].iloc[0], 8)

    def test_two_played_rows_for_one_player_game_are_rejected(self) -> None:
        players = pd.DataFrame(
            {
                "game_id": ["0022600124", "0022600124"],
                "player_id": [1626164, 1626164],
                "team_id": [1610612756, 1610612760],
                "min": [20.0, 10.0],
                "pts": [5, 4],
            }
        )

        with self.assertRaisesRegex(ValueError, "positive-minute"):
            merge_gamelogs(
                players,
                pd.DataFrame(),
                _teams(),
                pd.DataFrame(),
                _tracking(),
            )
