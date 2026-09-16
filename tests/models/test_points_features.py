"""Causal points features must not read the current game."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.points import (
    CURRENT_PLUS_MIN_MEAN_10,
    CURRENT_PLUS_PTS_MEAN_10,
    CURRENT_POINTS_41,
    CURRENT_POINTS_FEATURES,
    LEAN_POINTS_37,
    LEAN_POINTS_FEATURES,
    POINTS_SAMPLER_FEATURES,
    add_points_features,
)
from src.models.xgboost_models.points import (
    DEFAULT_POINTS_FEATURES,
    DIRECT_POINTS_FEATURES,
    XGBoostPointsModel,
    add_predicted_minutes_oof,
)


FORBIDDEN_FEATURE_COLUMNS = {
    "pts",
    "min",
    "min_sec",
    "minutes",
    "target_minutes",
    "fgm",
    "fga",
    "fg3_m",
    "fg3_a",
    "ftm",
    "fta",
    "usg_pct",
    "tchs",
    "poss",
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
    "team_fga",
    "team_off_rating",
    "team_def_rating",
    "opp_def_rating",
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


class PointsFeatureLeakageTests(unittest.TestCase):
    def test_scoring_features_ignore_the_current_game(self) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 10, pts=10, game_id=1),
                _player_row(1, "2024-01-03", 20, pts=20, game_id=2),
                _player_row(1, "2024-01-05", 30, pts=30, game_id=3),
            ]
        )
        featured = add_points_features(frame)
        third = featured.loc[
            featured["game_id"].eq(3)
        ].iloc[0]

        self.assertEqual(third["pts_lag_1"], 20)
        self.assertEqual(third["pts_mean_3"], 15)
        self.assertEqual(third["pts_mean_10"], 15)
        self.assertEqual(third["min_lag_1"], 20)

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "pts"] = 99
        mutated.loc[mutated["game_id"].eq(3), "minutes"] = 99
        mutated.loc[mutated["game_id"].eq(3), "min"] = 99
        mutated.loc[mutated["game_id"].eq(3), "min_sec"] = "99:00"
        mutated.loc[mutated["game_id"].eq(3), "fga"] = 99
        mutated.loc[mutated["game_id"].eq(3), "usg_pct"] = 99
        mutated_featured = add_points_features(mutated)
        mutated_third = mutated_featured.loc[
            mutated_featured["game_id"].eq(3)
        ].iloc[0]

        for column in (
            "pts_lag_1",
            "pts_mean_3",
            "pts_per_min_10",
            "min_lag_1",
            "fga_per_min_10",
            "usg_wmean_10",
        ):
            self.assertEqual(
                mutated_third[column],
                third[column],
                column,
            )

    def test_rates_use_rolling_sums_not_mean_of_ratios(self) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    10,
                    pts=10,
                    game_id=1,
                    fga=10,
                    fg3_a=9,
                    fta=2,
                    team_fga=50,
                ),
                _player_row(
                    1,
                    "2024-01-03",
                    20,
                    pts=20,
                    game_id=2,
                    fga=20,
                    fg3_a=1,
                    fta=4,
                    team_fga=80,
                ),
                _player_row(
                    1,
                    "2024-01-05",
                    30,
                    pts=8,
                    game_id=3,
                    fga=4,
                    fg3_a=0,
                    fta=0,
                    team_fga=90,
                ),
            ]
        )
        featured = add_points_features(frame)
        third = featured.loc[
            featured["game_id"].eq(3)
        ].iloc[0]

        self.assertAlmostEqual(third["pts_per_min_10"], 1.0)
        self.assertAlmostEqual(
            third["three_attempt_rate_10"],
            10 / 30,
        )
        self.assertAlmostEqual(
            third["free_throw_rate_10"],
            6 / 30,
        )
        self.assertAlmostEqual(
            third["player_fga_share_10"],
            30 / 130,
        )
        expected_ts = 30 / (
            2 * (30 + 0.44 * 6)
        )
        self.assertAlmostEqual(
            third["ts_agg_20"],
            expected_ts,
        )

    def test_season_scoring_resets_trailing_minutes_cross_seasons(
        self,
    ) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-04-10",
                    40,
                    pts=40,
                    game_id=1,
                    season_year="2023-24",
                ),
                _player_row(
                    1,
                    "2024-10-20",
                    18,
                    pts=12,
                    game_id=2,
                    season_year="2024-25",
                ),
            ]
        )
        featured = add_points_features(frame)
        opener = featured.loc[
            featured["game_id"].eq(2)
        ].iloc[0]

        self.assertEqual(opener["pts_lag_1"], 40)
        self.assertEqual(opener["min_lag_1"], 40)
        self.assertTrue(np.isnan(opener["season_pts_mean"]))
        self.assertTrue(np.isnan(opener["season_pts_per_min"]))
        self.assertTrue(np.isnan(opener["season_min_mean"]))

    def test_current_team_pts_per_min_resets_on_stint(self) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    20,
                    pts=20,
                    game_id=1,
                    team_id=100,
                ),
                _player_row(
                    1,
                    "2024-01-03",
                    10,
                    pts=2,
                    game_id=2,
                    team_id=200,
                ),
            ]
        )
        featured = add_points_features(frame)
        second = featured.loc[
            featured["game_id"].eq(2)
        ].iloc[0]
        self.assertTrue(
            np.isnan(second["current_team_pts_per_min"])
        )

    def test_opponent_defense_uses_opponent_prior_history(
        self,
    ) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    30,
                    pts=20,
                    game_id=1,
                    team_id=100,
                    opp_team_id=200,
                    team_off_rating=110,
                    team_def_rating=108,
                    team_pace=90,
                ),
                _player_row(
                    2,
                    "2024-01-01",
                    28,
                    pts=18,
                    game_id=1,
                    team_id=200,
                    opp_team_id=100,
                    team_off_rating=102,
                    team_def_rating=115,
                    team_pace=105,
                ),
                _player_row(
                    1,
                    "2024-01-04",
                    32,
                    pts=24,
                    game_id=2,
                    team_id=100,
                    opp_team_id=200,
                    team_off_rating=999,
                    team_def_rating=999,
                    team_pace=999,
                    opp_def_rating=999,
                    opp_pace=999,
                ),
                _player_row(
                    2,
                    "2024-01-04",
                    26,
                    pts=16,
                    game_id=2,
                    team_id=200,
                    opp_team_id=100,
                    team_off_rating=100,
                    team_def_rating=112,
                    team_pace=100,
                ),
            ]
        )
        featured = add_points_features(frame)
        home = featured.loc[
            featured["player_id"].eq(1)
            & featured["game_id"].eq(2)
        ].iloc[0]

        self.assertEqual(home["opp_def_rating_mean_10"], 115)
        self.assertEqual(home["team_off_rating_mean_10"], 110)
        self.assertEqual(home["opp_pace_mean_10"], 105)
        self.assertEqual(home["team_pace_mean_10"], 90)

    def test_market_and_schedule_features(self) -> None:
        frame = _rows(
            [
                _player_row(
                    1,
                    "2024-01-01",
                    30,
                    pts=22,
                    game_id=1,
                    matchup="BOS vs. NYK",
                    player_team_spread=-6.5,
                    game_total=220,
                ),
                _player_row(
                    1,
                    "2024-01-02",
                    28,
                    pts=18,
                    game_id=2,
                    matchup="BOS @ NYK",
                    player_team_spread=4.0,
                    game_total=np.nan,
                ),
            ]
        )
        featured = add_points_features(frame)
        second = featured.loc[
            featured["game_id"].eq(2)
        ].iloc[0]

        self.assertEqual(second["is_home"], 0)
        self.assertEqual(second["is_back_to_back"], 1)
        first = featured.loc[
            featured["game_id"].eq(1)
        ].iloc[0]
        self.assertEqual(first["is_home"], 1)
        self.assertAlmostEqual(
            first["implied_team_total"],
            113.25,
        )

    def test_default_feature_list_is_causal_and_stacked(
        self,
    ) -> None:
        overlap = FORBIDDEN_FEATURE_COLUMNS.intersection(
            DEFAULT_POINTS_FEATURES
        )
        self.assertEqual(overlap, set())
        self.assertNotIn("dnp_rate_10", DEFAULT_POINTS_FEATURES)
        self.assertIn(
            "predicted_minutes_oof",
            DEFAULT_POINTS_FEATURES,
        )
        self.assertIn(
            "expected_points_rate",
            DEFAULT_POINTS_FEATURES,
        )
        self.assertNotIn(
            "predicted_minutes_oof",
            DIRECT_POINTS_FEATURES,
        )

        model = XGBoostPointsModel(league="nba")
        self.assertEqual(
            model.feature_columns,
            DEFAULT_POINTS_FEATURES,
        )
        self.assertEqual(
            model.regressor.objective,
            "reg:squarederror",
        )


class PointsFeatureContractTests(unittest.TestCase):
    def test_default_equals_current_41(self) -> None:
        self.assertEqual(
            DEFAULT_POINTS_FEATURES,
            list(CURRENT_POINTS_FEATURES),
        )
        self.assertEqual(len(CURRENT_POINTS_FEATURES), 41)
        self.assertEqual(len(CURRENT_POINTS_41), 41)
        model = XGBoostPointsModel(league="nba")
        self.assertEqual(
            model.feature_columns,
            list(CURRENT_POINTS_FEATURES),
        )
        self.assertEqual(
            model.feature_contract_name,
            "current41",
        )

    def test_lean_keeps_volume_stack(self) -> None:
        self.assertEqual(len(LEAN_POINTS_FEATURES), 37)
        self.assertEqual(len(LEAN_POINTS_37), 37)
        for column in (
            "predicted_minutes_oof",
            "expected_points_rate",
            "expected_attempt_volume",
            "usage_x_expected_possessions",
            "fga_per_min_10",
        ):
            self.assertIn(column, LEAN_POINTS_FEATURES)

    def test_pts_std_10_is_not_a_mean_feature(self) -> None:
        self.assertNotIn(
            "pts_std_10",
            CURRENT_POINTS_FEATURES,
        )
        self.assertNotIn("pts_std_10", LEAN_POINTS_FEATURES)
        self.assertIn("pts_std_10", POINTS_SAMPLER_FEATURES)

    def test_plus_contracts_include_optional_windows(
        self,
    ) -> None:
        self.assertIn(
            "pts_mean_10",
            CURRENT_PLUS_PTS_MEAN_10,
        )
        self.assertNotIn(
            "pts_mean_10",
            CURRENT_POINTS_FEATURES,
        )
        pts_index = list(
            CURRENT_PLUS_PTS_MEAN_10
        ).index("pts_mean_10")
        self.assertEqual(
            CURRENT_PLUS_PTS_MEAN_10[pts_index - 1],
            "pts_mean_3",
        )
        self.assertIn(
            "min_mean_10",
            CURRENT_PLUS_MIN_MEAN_10,
        )
        self.assertNotIn(
            "min_mean_10",
            CURRENT_POINTS_FEATURES,
        )
        min_index = list(
            CURRENT_PLUS_MIN_MEAN_10
        ).index("min_mean_10")
        self.assertEqual(
            CURRENT_PLUS_MIN_MEAN_10[min_index - 1],
            "min_mean_3",
        )

    def test_actual_game_minutes_are_not_features(
        self,
    ) -> None:
        forbidden = {"minutes", "min", "start_position"}
        for contract in (
            DEFAULT_POINTS_FEATURES,
            CURRENT_POINTS_FEATURES,
            LEAN_POINTS_FEATURES,
        ):
            self.assertEqual(
                forbidden.intersection(contract),
                set(),
            )

    def test_custom_feature_list_is_named_custom(
        self,
    ) -> None:
        model = XGBoostPointsModel(
            league="nba",
            feature_columns=["pts_mean_3"],
        )
        self.assertEqual(
            model.feature_contract_name,
            "custom",
        )


class PredictedMinutesOofTests(unittest.TestCase):
    def test_oof_minutes_are_not_actual_game_minutes(self) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 10, pts=8, game_id=1),
                _player_row(1, "2024-01-03", 20, pts=16, game_id=2),
                _player_row(1, "2024-01-05", 30, pts=24, game_id=3),
                _player_row(1, "2024-01-07", 40, pts=32, game_id=4),
            ]
        )
        featured = add_points_features(frame)
        train = featured.index[featured["game_id"].isin([1, 2])]
        valid = featured.index[featured["game_id"].isin([3, 4])]
        spy = _SpyMinutesModel()

        stacked = add_predicted_minutes_oof(
            featured,
            minutes_model_factory=lambda: spy,
            splits=[(train, valid)],
        )
        later = stacked.loc[
            stacked["game_id"].isin([3, 4])
        ]

        self.assertTrue(
            later["predicted_minutes_oof"].eq(15.0).all()
        )
        self.assertFalse(
            later["predicted_minutes_oof"].eq(
                later["target_minutes"]
            ).any()
        )
        self.assertEqual(spy.fitted_game_ids, [{1, 2}])
        self.assertAlmostEqual(
            later.iloc[0]["expected_points_rate"],
            15.0 * later.iloc[0]["pts_per_min_10"],
        )


class _SpyMinutesModel:
    def __init__(self) -> None:
        self.fitted_game_ids: list[set] = []

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str = "minutes",
        date_column: str = "game_date",
    ) -> _SpyMinutesModel:
        self.fitted_game_ids.append(set(frame["game_id"]))
        return self

    def predict_mean(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), 15.0)


def _player_row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    pts: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    season_year: str = "2023-24",
    start_position: str = "G",
    matchup: str = "BOS vs. NYK",
    team_pace: float = 100.0,
    team_net_rating: float = 1.0,
    team_off_rating: float = 110.0,
    team_def_rating: float = 108.0,
    opp_pace: float = 100.0,
    opp_net_rating: float = -1.0,
    opp_def_rating: float = 108.0,
    player_team_spread: float = -3.0,
    game_total: float = 220.0,
    fga: float = 12,
    fg3_a: float = 4,
    fta: float = 3,
    team_fga: float = 88,
    usg_pct: float = 25.0,
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
        "pts": pts,
        "fgm": 5,
        "fga": fga,
        "fg3_m": 2,
        "fg3_a": fg3_a,
        "ftm": 2,
        "fta": fta,
        "usg_pct": usg_pct,
        "tchs": 40,
        "pass": 20,
        "assists": 4,
        "reb": 5,
        "stl": 1,
        "blk": 1,
        "team_pace": team_pace,
        "team_net_rating": team_net_rating,
        "team_off_rating": team_off_rating,
        "team_def_rating": team_def_rating,
        "team_fga": team_fga,
        "opp_pace": opp_pace,
        "opp_net_rating": opp_net_rating,
        "opp_def_rating": opp_def_rating,
        "player_team_spread": player_team_spread,
        "game_total": game_total,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
    }


if __name__ == "__main__":
    unittest.main()
