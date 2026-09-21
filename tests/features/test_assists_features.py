"""Causal assists features must not read the current game."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.assists import (
    ASSISTS_FEATURE_CHALLENGERS,
    CURRENT_ASSISTS_FEATURES,
    add_assists_features,
)

CONTRACT = [
    "predicted_minutes_oof",
    "start_rate_10",
    "ast_mean_10",
    "ast_mean_20",
    "assists_per_min_10",
    "team_ast_mean_10",
    "team_fgm_mean_10",
    "team_pace_mean_10",
    "opponent_team_ast_allowed_mean_10",
    "opponent_team_pace_mean_10",
    "days_rest",
    "is_home",
]


def _row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    ast: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    season_year: str = "2024-25",
    start_position: str = "G",
    matchup: str = "DET vs. CLE",
    team_ast: float = 20.0,
    team_fgm: float = 40.0,
    team_pace: float = 100.0,
    opp_ast: float = 22.0,
    assists: float | None = None,
) -> dict:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": season_year,
        "matchup": matchup,
        "minutes": minutes,
        "min": minutes,
        "min_sec": "00:00",
        "start_position": start_position,
        "ast": ast,
        "assists": ast if assists is None else assists,
        "team_ast": team_ast,
        "team_fgm": team_fgm,
        "team_pace": team_pace,
        "opp_ast": opp_ast,
        "opp_pace": team_pace,
        "pass": 10.0,
        "tchs": 20.0,
        "sast": 1.0,
        "ftast": 1.0,
    }


class ContractTests(unittest.TestCase):
    def test_current_assists_features_are_the_twelve(self) -> None:
        self.assertEqual(list(CURRENT_ASSISTS_FEATURES), CONTRACT)
        self.assertEqual(
            ASSISTS_FEATURE_CHALLENGERS["passing_tracking"],
            ["passes_per_min_10"],
        )

    def test_predicted_minutes_oof_is_float64_nan(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=5, game_id=1)]
        )
        featured = add_assists_features(frame)
        self.assertIn("predicted_minutes_oof", featured.columns)
        self.assertEqual(
            featured["predicted_minutes_oof"].dtype,
            np.float64,
        )
        self.assertTrue(
            featured["predicted_minutes_oof"].isna().all()
        )
        self.assertFalse(
            featured["predicted_minutes_oof"].isna().astype("object").eq(pd.NA).any()
        )

    def test_missing_minutes_column_raises(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=5, game_id=1)]
        ).drop(columns=["minutes", "min"])
        with self.assertRaises(ValueError):
            add_assists_features(frame)

    def test_nullable_minutes_and_ast_produce_features(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=5, game_id=1),
                _row(1, "2024-01-03", 24, ast=6, game_id=2),
            ]
        )
        frame["minutes"] = frame["minutes"].astype("Float64")
        frame["ast"] = frame["ast"].astype("Int64")

        featured = add_assists_features(frame)

        self.assertEqual(featured.iloc[1]["ast_mean_10"], 5.0)
        self.assertEqual(
            featured.iloc[1]["assists_per_min_10"],
            5.0 / 20.0,
        )

    def test_preserves_input_minutes_obs(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=5, game_id=1)]
        )
        frame["_minutes_obs"] = 77.0
        featured = add_assists_features(frame)
        self.assertEqual(featured["_minutes_obs"].iloc[0], 77.0)

    def test_overwrites_contract_and_preserves_caller_row(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=5, game_id=1)]
        )
        frame["assists_per_min_10"] = 999.0
        frame["_row"] = 42
        featured = add_assists_features(frame)
        self.assertTrue(
            np.isnan(featured["assists_per_min_10"].iloc[0])
        )
        self.assertEqual(featured["_row"].iloc[0], 42)

    def test_preserves_duplicate_unsorted_index(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-05", 30, ast=8, game_id=3),
                _row(1, "2024-01-01", 10, ast=2, game_id=1),
                _row(1, "2024-01-03", 20, ast=4, game_id=2),
            ]
        )
        frame.index = pd.Index([7, 7, 2], name="dup")
        featured = add_assists_features(frame)
        self.assertTrue(featured.index.equals(frame.index))
        self.assertEqual(
            list(featured["game_id"]),
            [3, 1, 2],
        )


class PlayerWindowTests(unittest.TestCase):
    def test_days_rest_opener_and_blank_minutes_candidate(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=5, game_id=1),
                _row(1, "2024-01-04", 24, ast=6, game_id=2),
                _row(1, "2024-01-07", np.nan, ast=np.nan, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertTrue(pd.isna(featured.iloc[0]["days_rest"]))
        self.assertEqual(featured.iloc[1]["days_rest"], 3)
        self.assertEqual(featured.iloc[2]["days_rest"], 3)

    def test_days_rest_resets_by_season(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-04-10", 20, ast=5, game_id=1,
                    season_year="2023-24",
                ),
                _row(
                    1, "2024-10-20", 20, ast=5, game_id=2,
                    season_year="2024-25",
                ),
            ]
        )
        featured = add_assists_features(frame)
        self.assertTrue(pd.isna(featured.iloc[1]["days_rest"]))

    def test_ast_mean_ignores_current_and_nan_ast(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=4, game_id=1),
                _row(1, "2024-01-03", 20, ast=np.nan, game_id=2),
                _row(1, "2024-01-05", 20, ast=10, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertEqual(featured.iloc[2]["ast_mean_10"], 4.0)
        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "ast"] = 99
        mutated_featured = add_assists_features(mutated)
        self.assertEqual(
            mutated_featured.iloc[2]["ast_mean_10"],
            featured.iloc[2]["ast_mean_10"],
        )

    def test_start_rate_missing_start_is_non_start_slot(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    start_position="G",
                ),
                _row(
                    1, "2024-01-03", 20, ast=4, game_id=2,
                    start_position="",
                ),
                _row(1, "2024-01-05", 20, ast=4, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertEqual(featured.iloc[2]["start_rate_10"], 0.5)

    def test_is_home_from_matchup(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    matchup="DET vs. CLE",
                ),
                _row(
                    1, "2024-01-03", 20, ast=4, game_id=2,
                    matchup="DET @ CLE",
                ),
            ]
        )
        featured = add_assists_features(frame)
        self.assertEqual(featured.iloc[0]["is_home"], 1.0)
        self.assertEqual(featured.iloc[1]["is_home"], 0.0)
        del featured
        missing = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=4, game_id=1)]
        ).drop(columns=["matchup"])
        featured = add_assists_features(missing)
        self.assertTrue(pd.isna(featured.iloc[0]["is_home"]))


class PairedRateTests(unittest.TestCase):
    def test_clean_history_is_sum_ast_over_sum_minutes(self) -> None:
        rows = [
            _row(1, f"2024-01-{day:02d}", 10.0 + i, ast=float(i + 1), game_id=i)
            for i, day in enumerate(range(1, 12), start=1)
        ]
        frame = pd.DataFrame(rows)
        featured = add_assists_features(frame)
        last = featured.iloc[-1]
        prior = frame.iloc[:-1]
        expected = prior["ast"].sum() / prior["minutes"].sum()
        self.assertAlmostEqual(
            last["assists_per_min_10"], expected
        )

    def test_nan_ast_hole_is_excluded_from_both(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 10, ast=5, game_id=1),
                _row(1, "2024-01-03", 20, ast=np.nan, game_id=2),
                _row(1, "2024-01-05", 30, ast=9, game_id=3),
            ]
        )
        featured = add_assists_features(frame)
        self.assertAlmostEqual(
            featured.iloc[2]["assists_per_min_10"],
            5.0 / 10.0,
        )


class TeamSnapshotTests(unittest.TestCase):
    def test_null_aware_reduction_skips_nan_teammate(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    **_row(1, "2024-01-01", 20, ast=4, game_id=1),
                    "team_ast": np.nan,
                },
                {
                    **_row(2, "2024-01-01", 20, ast=5, game_id=1),
                    "team_ast": 25.0,
                },
                _row(1, "2024-01-03", 20, ast=6, game_id=2),
            ]
        )
        featured = add_assists_features(frame)
        tonight = featured.loc[featured["game_id"].eq(2)].iloc[0]
        self.assertEqual(tonight["team_ast_mean_10"], 25.0)

    def test_scheduled_opponent_not_player_opp_history(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    team_id=100, opp_team_id=200,
                    team_pace=90.0, opp_ast=5.0,
                ),
                _row(
                    2, "2024-01-01", 20, ast=8, game_id=1,
                    team_id=200, opp_team_id=100,
                    team_pace=110.0, opp_ast=20.0,
                ),
                _row(
                    1, "2024-01-03", 20, ast=4, game_id=2,
                    team_id=100, opp_team_id=200,
                    team_pace=91.0, opp_ast=6.0,
                ),
                _row(
                    2, "2024-01-03", 20, ast=8, game_id=2,
                    team_id=200, opp_team_id=100,
                    team_pace=111.0, opp_ast=100.0,
                ),
                _row(
                    1, "2024-01-05", 20, ast=4, game_id=3,
                    team_id=100, opp_team_id=200,
                    team_pace=92.0, opp_ast=7.0,
                ),
                _row(
                    2, "2024-01-05", 20, ast=8, game_id=3,
                    team_id=200, opp_team_id=100,
                    team_pace=112.0, opp_ast=8.0,
                ),
            ]
        )
        featured = add_assists_features(frame)
        tonight = featured.loc[
            frame["game_id"].eq(3) & frame["player_id"].eq(1)
        ].iloc[0]
        self.assertEqual(
            tonight["opponent_team_ast_allowed_mean_10"],
            60.0,
        )
        self.assertNotEqual(
            tonight["team_pace_mean_10"],
            tonight["opponent_team_pace_mean_10"],
        )
        self.assertNotEqual(
            tonight["opponent_team_ast_allowed_mean_10"],
            20.0,
        )

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "opp_ast"] = 999
        mutated.loc[mutated["game_id"].eq(3), "team_pace"] = 999
        mutated_featured = add_assists_features(mutated)
        mutated_tonight = mutated_featured.loc[
            mutated["game_id"].eq(3) & mutated["player_id"].eq(1)
        ].iloc[0]
        self.assertEqual(
            mutated_tonight["opponent_team_ast_allowed_mean_10"],
            tonight["opponent_team_ast_allowed_mean_10"],
        )
        self.assertEqual(
            mutated_tonight["team_pace_mean_10"],
            tonight["team_pace_mean_10"],
        )

    def test_missing_opponent_id_does_not_crash_asof_join(self) -> None:
        history = pd.DataFrame(
            [
                _row(
                    1, "2024-01-01", 20, ast=4, game_id=1,
                    team_id=100, opp_team_id=200,
                ),
                _row(
                    2, "2024-01-01", 20, ast=8, game_id=1,
                    team_id=200, opp_team_id=100,
                ),
            ]
        )
        history["team_id"] = history["team_id"].astype("Int64")
        history["opp_team_id"] = history["opp_team_id"].astype("Int64")
        candidate = pd.DataFrame(
            [
                {
                    **_row(
                        1, "2024-01-05", np.nan, ast=np.nan, game_id=99,
                        team_id=100, opp_team_id=200,
                    ),
                    "opp_team_id": pd.NA,
                }
            ]
        )
        candidate["team_id"] = candidate["team_id"].astype(object)
        candidate["opp_team_id"] = candidate["opp_team_id"].astype(object)
        featured = add_assists_features(
            pd.concat([history, candidate], ignore_index=True)
        )
        tonight = featured.iloc[-1]
        self.assertTrue(
            pd.isna(tonight["opponent_team_ast_allowed_mean_10"])
        )
        self.assertTrue(
            pd.isna(tonight["opponent_team_pace_mean_10"])
        )
        self.assertTrue(np.isfinite(tonight["team_ast_mean_10"]))


_FEATURE_COLUMNS = list(CURRENT_ASSISTS_FEATURES)


class LeakageTests(unittest.TestCase):
    def test_mutating_game_n_does_not_change_features(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=4, game_id=1),
                _row(1, "2024-01-03", 24, ast=6, game_id=2),
            ]
        )
        featured = add_assists_features(frame)
        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(2), "ast"] = 99
        mutated.loc[mutated["game_id"].eq(2), "minutes"] = 99
        mutated.loc[mutated["game_id"].eq(2), "team_ast"] = 99
        mutated.loc[mutated["game_id"].eq(2), "opp_ast"] = 99
        mutated.loc[mutated["game_id"].eq(2), "start_position"] = ""
        mutated_featured = add_assists_features(mutated)
        left = featured.loc[featured["game_id"].eq(2)].iloc[0]
        right = mutated_featured.loc[
            mutated_featured["game_id"].eq(2)
        ].iloc[0]
        for column in _FEATURE_COLUMNS:
            self._assert_same(left[column], right[column], column)

    def test_blanking_both_aliases_does_not_change_features(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2024-01-01", 20, ast=4, game_id=1),
                _row(1, "2024-01-03", 24, ast=6, game_id=2),
            ]
        )
        featured = add_assists_features(frame)
        blanked = frame.copy()
        for column in ("ast", "assists", "min", "minutes"):
            blanked.loc[blanked["game_id"].eq(2), column] = np.nan
        blanked_featured = add_assists_features(blanked)
        left = featured.loc[featured["game_id"].eq(2)].iloc[0]
        right = blanked_featured.loc[
            blanked_featured["game_id"].eq(2)
        ].iloc[0]
        for column in _FEATURE_COLUMNS:
            self._assert_same(left[column], right[column], column)

    def test_overwrites_existing_assists_per_min_10(self) -> None:
        frame = pd.DataFrame(
            [_row(1, "2024-01-01", 20, ast=4, game_id=1)]
        )
        frame["assists_per_min_10"] = 999.0
        featured = add_assists_features(frame)
        self.assertTrue(
            pd.isna(featured.iloc[0]["assists_per_min_10"])
        )

    def _assert_same(self, left, right, column: str) -> None:
        if pd.isna(left) and pd.isna(right):
            return
        self.assertEqual(left, right, msg=column)
