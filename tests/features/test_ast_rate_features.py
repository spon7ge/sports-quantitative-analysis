"""Assists-per-minute features use only earlier games."""

from __future__ import annotations

import unittest

import pandas as pd

from src.features.assists.rate import (
    _OPP_OUTPUTS,
    _OWN_TEAM_OUTPUTS,
    _PLAYER_OUTPUTS,
    add_ast_rate_features,
)


def _row(
    player_id: int,
    game_date: str,
    *,
    game_id: int,
    minutes: float = 20.0,
    ast: float = 4.0,
    team_id: int = 10,
    opp_team_id: int = 20,
    season_year: str = "2024-25",
    matchup: str = "DET vs. CLE",
    ast_pct: float = 0.25,
    tchs: float = 40.0,
    passes: float = 20.0,
    sast: float = 2.0,
    usg_pct: float = 0.20,
    team_pace: float = 100.0,
    team_ast_pct: float = 0.60,
    team_efg_pct: float = 0.54,
    team_def_rating: float = 112.0,
    opp_ast: float = 25.0,
    team_poss: float = 100.0,
) -> dict:
    return {
        "player_id": player_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": season_year,
        "minutes": minutes,
        "ast": ast,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "matchup": matchup,
        "ast_pct": ast_pct,
        "tchs": tchs,
        "pass": passes,
        "sast": sast,
        "usg_pct": usg_pct,
        "team_pace": team_pace,
        "team_ast_pct": team_ast_pct,
        "team_efg_pct": team_efg_pct,
        "team_def_rating": team_def_rating,
        "opp_ast": opp_ast,
        "team_poss": team_poss,
    }


class AstRateFeatureTests(unittest.TestCase):
    def test_outputs_are_prior_only_and_keep_row_order(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-05", game_id=2, ast=8, minutes=10),
                _row(1, "2024-01-01", game_id=1, ast=4, minutes=20),
            ]
        )
        featured = add_ast_rate_features(frame)

        self.assertEqual(list(featured.index), [0, 1])
        later = featured.iloc[0]
        earlier = featured.iloc[1]
        self.assertTrue(pd.isna(earlier["player_ast_rate_last5"]))
        self.assertAlmostEqual(later["player_ast_rate_last5"], 0.2)
        self.assertAlmostEqual(later["player_ast_rate_last10"], 0.2)
        self.assertAlmostEqual(later["player_min_last5"], 20.0)
        self.assertAlmostEqual(later["player_ast_pct_last10"], 0.25)
        self.assertAlmostEqual(later["player_pass_per_touch_last10"], 0.5)
        self.assertEqual(earlier["season_games_prior"], 0)
        self.assertEqual(later["season_games_prior"], 1)
        self.assertTrue(pd.isna(earlier["days_since_previous_game"]))
        self.assertEqual(later["days_since_previous_game"], 4.0)
        self.assertEqual(later["is_home"], 1.0)

    def test_current_game_does_not_enter_the_rate(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", game_id=1, ast=4, minutes=20),
                _row(1, "2024-01-03", game_id=2, ast=9, minutes=10),
            ]
        )
        featured = add_ast_rate_features(frame)
        frame.loc[1, "ast"] = 99
        frame.loc[1, "minutes"] = 50
        mutated = add_ast_rate_features(frame)
        self.assertAlmostEqual(
            featured.loc[1, "player_ast_rate_last10"],
            0.2,
        )
        self.assertAlmostEqual(
            mutated.loc[1, "player_ast_rate_last10"],
            featured.loc[1, "player_ast_rate_last10"],
        )

    def test_last5_drops_the_game_before_the_window(self) -> None:
        rows = []
        for game_id, ast in enumerate((1, 2, 3, 4, 5, 100), start=1):
            rows.append(
                _row(
                    1,
                    f"2024-01-{game_id:02d}",
                    game_id=game_id,
                    ast=float(ast),
                    minutes=10,
                )
            )
        rows.append(
            _row(1, "2024-01-07", game_id=7, ast=0, minutes=10)
        )
        featured = add_ast_rate_features(pd.DataFrame(rows))
        # Prior five games are ast 2,3,4,5,100.
        self.assertAlmostEqual(
            featured.loc[6, "player_ast_rate_last5"],
            114 / 50,
        )

    def test_season_rate_resets_and_trailing_rate_does_not(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1,
                    "2024-04-01",
                    game_id=1,
                    season_year="2023-24",
                    ast=10,
                    minutes=10,
                ),
                _row(
                    1,
                    "2024-10-22",
                    game_id=2,
                    season_year="2024-25",
                    ast=1,
                    minutes=10,
                ),
            ]
        )
        featured = add_ast_rate_features(frame)
        opener = featured.iloc[1]
        self.assertAlmostEqual(opener["player_ast_rate_last10"], 1.0)
        self.assertTrue(pd.isna(opener["player_ast_rate_season_to_date"]))
        self.assertEqual(opener["season_games_prior"], 0)
        self.assertTrue(pd.isna(opener["days_since_previous_game"]))

    def test_team_and_opponent_windows_exclude_the_current_game(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    2,
                    "2024-01-01",
                    game_id=1,
                    team_id=20,
                    opp_team_id=30,
                    team_pace=90,
                    team_def_rating=108,
                    opp_ast=30,
                    team_poss=100,
                    team_ast_pct=0.70,
                    team_efg_pct=0.50,
                ),
                _row(
                    1,
                    "2024-01-03",
                    game_id=2,
                    team_id=10,
                    opp_team_id=20,
                    team_pace=110,
                    team_def_rating=120,
                    opp_ast=99,
                    team_poss=80,
                    team_ast_pct=0.40,
                    team_efg_pct=0.61,
                    matchup="DET @ CLE",
                ),
                _row(
                    3,
                    "2024-01-03",
                    game_id=2,
                    team_id=20,
                    opp_team_id=10,
                    team_pace=110,
                    team_def_rating=99,
                    opp_ast=5,
                    team_poss=80,
                    matchup="CLE vs. DET",
                ),
            ]
        )
        featured = add_ast_rate_features(frame)
        current = featured.iloc[1]
        self.assertTrue(pd.isna(current["team_pace_last10"]))
        self.assertAlmostEqual(current["opp_pace_last10"], 90.0)
        self.assertAlmostEqual(current["opp_def_rating_last10"], 108.0)
        self.assertAlmostEqual(
            current["opp_ast_allowed_per_100_last10"],
            30.0,
        )
        self.assertEqual(current["is_home"], 0.0)

        frame.loc[1, "team_pace"] = 999
        frame.loc[1, "opp_ast"] = 1
        mutated = add_ast_rate_features(frame)
        self.assertAlmostEqual(mutated.iloc[1]["opp_pace_last10"], 90.0)
        self.assertAlmostEqual(
            mutated.iloc[1]["opp_ast_allowed_per_100_last10"],
            30.0,
        )

    def test_second_team_game_uses_only_the_first(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1,
                    "2024-01-01",
                    game_id=1,
                    team_id=10,
                    team_pace=100,
                    team_ast_pct=0.50,
                    team_efg_pct=0.40,
                ),
                _row(
                    1,
                    "2024-01-03",
                    game_id=2,
                    team_id=10,
                    team_pace=120,
                    team_ast_pct=0.80,
                    team_efg_pct=0.70,
                ),
            ]
        )
        featured = add_ast_rate_features(frame)
        self.assertAlmostEqual(featured.loc[1, "team_pace_last10"], 100)
        self.assertAlmostEqual(
            featured.loc[1, "team_ast_pct_last10"], 0.50
        )
        self.assertAlmostEqual(
            featured.loc[1, "team_efg_pct_last10"], 0.40
        )

    def test_contract_columns_are_present(self) -> None:
        featured = add_ast_rate_features(
            pd.DataFrame([_row(1, "2024-01-01", game_id=1)])
        )
        for column in (
            *_PLAYER_OUTPUTS,
            *_OWN_TEAM_OUTPUTS,
            *_OPP_OUTPUTS,
        ):
            self.assertIn(column, featured.columns)


if __name__ == "__main__":
    unittest.main()
