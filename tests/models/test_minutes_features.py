"""Causal minutes features must not read the current game."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.minutes import (
    CURRENT_MINUTES_37,
    CURRENT_MINUTES_FEATURES,
    LEAN_MINUTES_FEATURES,
    ROLE_TAIL_MINUTES_FEATURES,
    TIER1_MINUTES_FEATURES,
    add_minutes_features,
)
from src.models.xgboost_models.minutes import (
    DEFAULT_MINUTES_FEATURES,
    MINUTES_FEATURE_CONTRACTS,
    XGBoostMinutesModel,
)


FORBIDDEN_FEATURE_COLUMNS = {
    "min",
    "min_sec",
    "minutes",
    "target_minutes",
    "start_position",
    "available_flag",
    "comment",
    "wl",
    "is_starter",
    "player_id",
    "team_id",
    "opp_team_id",
    "game_id",
    "player_name",
    "fgm_pg",
    "fga_pg",
    "team_count",
    "team_spread",
}


def _clock(minutes: float) -> str:
    whole = int(minutes)
    seconds = int(round((minutes - whole) * 60))
    if seconds == 60:
        whole += 1
        seconds = 0
    return f"{whole}:{seconds:02d}"


def _rows(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["min_sec"] = [
        _clock(value) for value in frame["minutes"]
    ]
    frame["min"] = frame["minutes"]
    return frame


class MinutesFeatureLeakageTests(unittest.TestCase):
    def test_rolling_minutes_ignore_the_current_game(self) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 10, game_id=1),
                _player_row(1, "2024-01-03", 20, game_id=2),
                _player_row(1, "2024-01-05", 30, game_id=3),
            ]
        )
        featured = add_minutes_features(frame)
        third = featured.loc[
            featured["game_id"].eq(3)
        ].iloc[0]

        self.assertEqual(third["min_lag_1"], 20)
        self.assertEqual(third["min_mean_3"], 15)
        self.assertEqual(third["min_mean_10"], 15)

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "minutes"] = 99
        mutated.loc[mutated["game_id"].eq(3), "min"] = 99
        mutated.loc[mutated["game_id"].eq(3), "min_sec"] = "99:00"
        mutated_featured = add_minutes_features(mutated)
        mutated_third = mutated_featured.loc[
            mutated_featured["game_id"].eq(3)
        ].iloc[0]

        self.assertEqual(
            mutated_third["min_lag_1"],
            third["min_lag_1"],
        )
        self.assertEqual(
            mutated_third["min_mean_3"],
            third["min_mean_3"],
        )

    def test_start_rate_ignores_current_start_position(self) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    30,
                    game_id=1,
                    start_position="G",
                ),
                _player_row(
                    1,
                    "2024-01-03",
                    28,
                    game_id=2,
                    start_position="G",
                ),
                _player_row(
                    1,
                    "2024-01-05",
                    12,
                    game_id=3,
                    start_position="",
                ),
            ]
        )
        featured = add_minutes_features(frame)
        third = featured.loc[
            featured["game_id"].eq(3)
        ].iloc[0]
        self.assertEqual(third["start_rate_10"], 1.0)
        self.assertNotIn("is_starter", featured.columns)

    def test_trailing_minutes_cross_seasons_season_mean_resets(
        self,
    ) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-04-10",
                    40,
                    game_id=1,
                    season_year="2023-24",
                ),
                _player_row(
                    1,
                    "2024-10-20",
                    18,
                    game_id=2,
                    season_year="2024-25",
                ),
            ]
        )
        featured = add_minutes_features(frame)
        opener = featured.loc[
            featured["game_id"].eq(2)
        ].iloc[0]

        self.assertEqual(opener["min_lag_1"], 40)
        self.assertTrue(np.isnan(opener["season_min_mean"]))
        self.assertEqual(opener["season_rows_prior"], 0)
        self.assertEqual(opener["prior_season_min_mean"], 40)

    def test_nullable_team_id_does_not_break_stints(self) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 20, game_id=1),
                _player_row(1, "2024-01-03", 22, game_id=2),
            ]
        )
        frame["team_id"] = frame["team_id"].astype("Int64")
        frame.loc[0, "team_id"] = pd.NA
        featured = add_minutes_features(frame)
        self.assertEqual(len(featured), 2)
        self.assertFalse(
            featured["current_team_rows_prior"].isna().all()
        )

    def test_team_pace_is_not_rolled_on_player_duplicates(
        self,
    ) -> None:
        records = []
        for player_id in range(1, 11):
            records.append(
                _player_row(
                    player_id,
                    "2024-01-01",
                    20,
                    game_id=1,
                    team_id=100,
                    opp_team_id=200,
                    team_pace=90,
                    team_net_rating=5,
                )
            )
            records.append(
                _player_row(
                    player_id,
                    "2024-01-03",
                    22,
                    game_id=2,
                    team_id=100,
                    opp_team_id=200,
                    team_pace=110,
                    team_net_rating=8,
                )
            )
        featured = add_minutes_features(_rows(records))
        game_two = featured.loc[featured["game_id"].eq(2)]

        self.assertTrue(
            game_two["team_pace_mean_10"].eq(90).all()
        )

    def test_opponent_features_use_opponent_prior_history(
        self,
    ) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    30,
                    game_id=1,
                    team_id=100,
                    opp_team_id=200,
                    team_pace=90,
                    team_net_rating=4,
                ),
                _player_row(
                    2,
                    "2024-01-01",
                    28,
                    game_id=1,
                    team_id=200,
                    opp_team_id=100,
                    team_pace=105,
                    team_net_rating=-3,
                ),
                _player_row(
                    1,
                    "2024-01-04",
                    32,
                    game_id=2,
                    team_id=100,
                    opp_team_id=200,
                    team_pace=999,
                    team_net_rating=999,
                    opp_pace=999,
                    opp_net_rating=999,
                ),
                _player_row(
                    2,
                    "2024-01-04",
                    26,
                    game_id=2,
                    team_id=200,
                    opp_team_id=100,
                    team_pace=100,
                    team_net_rating=-1,
                ),
            ]
        )
        featured = add_minutes_features(frame)
        home = featured.loc[
            featured["player_id"].eq(1)
            & featured["game_id"].eq(2)
        ].iloc[0]

        self.assertEqual(home["opp_pace_mean_10"], 105)
        self.assertEqual(home["team_pace_mean_10"], 90)
        self.assertEqual(home["expected_possessions"], 97.5)

    def test_market_and_schedule_features(self) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    30,
                    game_id=1,
                    matchup="BOS vs. NYK",
                    player_team_spread=-6.5,
                    game_total=220,
                ),
                _player_row(
                    1,
                    "2024-01-02",
                    28,
                    game_id=2,
                    matchup="BOS @ NYK",
                    player_team_spread=4.0,
                    game_total=np.nan,
                ),
            ]
        )
        featured = add_minutes_features(frame)
        second = featured.loc[
            featured["game_id"].eq(2)
        ].iloc[0]

        self.assertEqual(second["is_home"], 0)
        self.assertEqual(second["is_back_to_back"], 1)
        self.assertEqual(
            second["team_days_since_prev_game"],
            1,
        )
        self.assertEqual(second["team_games_prev_3d"], 1)
        self.assertEqual(second["abs_spread"], 4.0)
        self.assertEqual(second["market_missing"], 1)
        first = featured.loc[
            featured["game_id"].eq(1)
        ].iloc[0]
        self.assertEqual(first["is_home"], 1)
        self.assertAlmostEqual(
            first["implied_team_total"],
            113.25,
        )

    def test_default_feature_list_is_causal(self) -> None:
        overlap = FORBIDDEN_FEATURE_COLUMNS.intersection(
            DEFAULT_MINUTES_FEATURES
        )
        self.assertEqual(overlap, set())
        self.assertNotIn("dnp_rate_10", DEFAULT_MINUTES_FEATURES)
        self.assertIn("min_mean_10", DEFAULT_MINUTES_FEATURES)
        self.assertIn(
            "team_spread_canonical",
            DEFAULT_MINUTES_FEATURES,
        )

        model = XGBoostMinutesModel(league="nba")
        self.assertEqual(
            model.feature_columns,
            DEFAULT_MINUTES_FEATURES,
        )

    def test_current_contract_is_the_constructor_default(self) -> None:
        self.assertEqual(
            DEFAULT_MINUTES_FEATURES,
            CURRENT_MINUTES_FEATURES,
        )
        self.assertEqual(CURRENT_MINUTES_FEATURES, CURRENT_MINUTES_37)
        self.assertEqual(len(CURRENT_MINUTES_FEATURES), 37)
        self.assertEqual(len(DEFAULT_MINUTES_FEATURES), 37)

        model = XGBoostMinutesModel(league="nba")
        self.assertEqual(
            model.feature_columns,
            CURRENT_MINUTES_FEATURES,
        )
        self.assertEqual(model.feature_contract_name, "current37")

    def test_role_tail_appends_four_columns(self) -> None:
        extra = [
            "min_ge_30_rate_10",
            "is_back_to_back",
            "team_games_prev_5d",
            "usg_5_minus_season",
        ]
        self.assertEqual(
            ROLE_TAIL_MINUTES_FEATURES[:37],
            CURRENT_MINUTES_FEATURES,
        )
        self.assertEqual(ROLE_TAIL_MINUTES_FEATURES[37:], extra)
        self.assertTrue(
            set(CURRENT_MINUTES_FEATURES).issubset(
                ROLE_TAIL_MINUTES_FEATURES
            )
        )
        self.assertEqual(len(ROLE_TAIL_MINUTES_FEATURES), 41)

    def test_named_contracts_exclude_leakage_columns(self) -> None:
        self.assertGreaterEqual(len(LEAN_MINUTES_FEATURES), 24)
        self.assertLessEqual(len(LEAN_MINUTES_FEATURES), 28)
        self.assertEqual(len(TIER1_MINUTES_FEATURES), 47)
        for name, columns in MINUTES_FEATURE_CONTRACTS.items():
            overlap = FORBIDDEN_FEATURE_COLUMNS.intersection(
                columns
            )
            self.assertEqual(overlap, set(), msg=name)

        custom = XGBoostMinutesModel(
            league="nba",
            feature_columns=["min_lag_1", "is_home"],
        )
        self.assertEqual(custom.feature_contract_name, "custom")
        matched = XGBoostMinutesModel(
            league="nba",
            feature_columns=list(ROLE_TAIL_MINUTES_FEATURES),
        )
        self.assertEqual(matched.feature_contract_name, "role_tail")


def _player_row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    season_year: str = "2023-24",
    start_position: str = "G",
    matchup: str = "BOS vs. NYK",
    team_pace: float = 100.0,
    team_net_rating: float = 1.0,
    opp_pace: float = 100.0,
    opp_net_rating: float = -1.0,
    player_team_spread: float = -3.0,
    game_total: float = 220.0,
) -> dict:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": season_year,
        "season_type": "Regular Season",
        "matchup": matchup,
        "minutes": minutes,
        "start_position": start_position,
        "usg_pct": 25.0,
        "tchs": 40,
        "pass": 20,
        "fga": 12,
        "assists": 4,
        "reb": 5,
        "stl": 1,
        "blk": 1,
        "team_pace": team_pace,
        "team_net_rating": team_net_rating,
        "opp_pace": opp_pace,
        "opp_net_rating": opp_net_rating,
        "player_team_spread": player_team_spread,
        "game_total": game_total,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
    }


if __name__ == "__main__":
    unittest.main()
