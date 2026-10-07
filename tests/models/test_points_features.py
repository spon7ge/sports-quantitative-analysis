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
    DIRECT_POINTS_FEATURES,
    LEAN_POINTS_37,
    LEAN_POINTS_FEATURES,
    POINTS_SAMPLER_FEATURES,
    add_points_features,
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

    def test_typical_band_and_recent_bounds_ignore_the_current_game(
        self,
    ) -> None:
        points = [10, 20, 30, 40, 100, 32, 28]
        frame = _rows(
            [
                _player_row(
                    1,
                    f"2024-01-{2 * index + 1:02d}",
                    30,
                    pts=value,
                    game_id=index + 1,
                )
                for index, value in enumerate(points)
            ]
        )
        featured = add_points_features(frame)
        by_game = featured.set_index("game_id")
        # 30 minutes each, so prior ppm is 10, 20, 30, 40, 100 over 30.
        # Linear 20th/80th are 18/30 and 52/30; inside are 20, 30, 40.
        row = by_game.loc[6]
        self.assertAlmostEqual(row["ppm_p20_20"], 18 / 30)
        self.assertAlmostEqual(row["ppm_p80_20"], 52 / 30)
        self.assertAlmostEqual(row["ppm_p80_minus_p20_20"], 34 / 30)
        self.assertAlmostEqual(row["ppm_trim_mean_20"], 1.0)
        self.assertAlmostEqual(row["ppm_floor_10"], 10 / 30)
        self.assertAlmostEqual(row["ppm_ceiling_10"], 100 / 30)
        self.assertAlmostEqual(row["ppm_span_10"], 3.0)
        self.assertEqual(row["ppm_lag_outside_10"], 1)
        self.assertEqual(by_game.loc[7, "ppm_lag_outside_10"], 0)
        self.assertTrue(np.isnan(by_game.loc[1, "ppm_p20_20"]))
        self.assertTrue(np.isnan(by_game.loc[1, "ppm_floor_10"]))

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(6), "pts"] = 1
        mutated_row = add_points_features(mutated).set_index("game_id").loc[6]
        self.assertAlmostEqual(mutated_row["ppm_trim_mean_20"], 1.0)
        self.assertAlmostEqual(mutated_row["ppm_span_10"], 3.0)
        self.assertEqual(mutated_row["ppm_lag_outside_10"], 1)

    def test_ppm_distribution_skips_short_stints(self) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 20, pts=10, game_id=1),
                _player_row(1, "2024-01-03", 2, pts=8, game_id=2),
                _player_row(1, "2024-01-05", 20, pts=20, game_id=3),
                _player_row(1, "2024-01-07", 20, pts=12, game_id=4),
            ]
        )
        row = add_points_features(frame).set_index("game_id").loc[4]

        # The 2-minute, 4.0 ppm stint is excluded; priors are 0.5 and 1.0.
        self.assertAlmostEqual(row["ppm_floor_10"], 0.5)
        self.assertAlmostEqual(row["ppm_ceiling_10"], 1.0)

    def test_points_volatility_is_prior_dispersion_over_prior_mean(
        self,
    ) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 30, pts=10, game_id=1),
                _player_row(1, "2024-01-03", 30, pts=30, game_id=2),
                _player_row(1, "2024-01-05", 30, pts=20, game_id=3),
            ]
        )
        by_game = add_points_features(frame).set_index("game_id")

        self.assertTrue(np.isnan(by_game.loc[1, "ppm_vol_10"]))
        self.assertTrue(np.isnan(by_game.loc[2, "ppm_vol_10"]))
        # Prior pts 10 and 30 over 30 minutes; CV is scale-free.
        self.assertAlmostEqual(
            by_game.loc[3, "ppm_vol_10"],
            (200 ** 0.5) / 20,
        )

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "pts"] = 99
        self.assertAlmostEqual(
            add_points_features(mutated)
            .set_index("game_id")
            .loc[3, "ppm_vol_10"],
            by_game.loc[3, "ppm_vol_10"],
        )

    def test_season_volatility_cap_flags_unstable_scorers(self) -> None:
        stable_pts = [20, 22, 21, 23, 19, 21]
        unstable_pts = [4, 28, 2, 30, 6, 18]
        frame = _rows(
            [
                _player_row(
                    player_id,
                    f"2024-01-{2 * index + 1:02d}",
                    30,
                    pts=value,
                    game_id=10 * player_id + index + 1,
                )
                for player_id, series in ((1, stable_pts), (2, unstable_pts))
                for index, value in enumerate(series)
            ]
        )
        by_game = add_points_features(frame).set_index("game_id")
        stable = by_game.loc[16]
        unstable = by_game.loc[26]

        self.assertAlmostEqual(stable["season_ppm_vol"], (2.5 ** 0.5) / 21)
        self.assertEqual(stable["ppm_unstable"], 0)
        self.assertAlmostEqual(
            unstable["season_ppm_vol"],
            (190.0 ** 0.5) / 14,
        )
        self.assertEqual(unstable["ppm_unstable"], 1)
        self.assertTrue(np.isnan(by_game.loc[11, "season_ppm_vol"]))
        self.assertTrue(np.isnan(by_game.loc[11, "ppm_unstable"]))

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(26), "pts"] = 40
        mutated_row = add_points_features(mutated).set_index("game_id").loc[26]
        self.assertEqual(mutated_row["ppm_unstable"], 1)
        self.assertAlmostEqual(
            mutated_row["season_ppm_vol"],
            unstable["season_ppm_vol"],
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

    def test_ewm_pts_per_min_is_ratio_of_prior_ewms(self) -> None:
        frame = _rows(
            [
                _player_row(1, "2024-01-01", 10, pts=10, game_id=1),
                _player_row(1, "2024-01-03", 20, pts=40, game_id=2),
                _player_row(1, "2024-01-05", 30, pts=8, game_id=3),
            ]
        )
        by_game = add_points_features(frame).set_index("game_id")

        self.assertTrue(np.isnan(by_game.loc[1, "pts_per_min_ewm_hl_10"]))
        self.assertAlmostEqual(by_game.loc[2, "pts_per_min_ewm_hl_10"], 1.0)
        for halflife in (10, 20):
            alpha = 1 - np.exp(np.log(0.5) / halflife)
            self.assertAlmostEqual(
                by_game.loc[3, f"pts_per_min_ewm_hl_{halflife}"],
                (10 + alpha * 30) / (10 + alpha * 10),
            )

        mutated = frame.copy()
        mutated.loc[mutated["game_id"].eq(3), "pts"] = 99
        mutated.loc[mutated["game_id"].eq(3), "minutes"] = 1
        mutated.loc[mutated["game_id"].eq(3), "min"] = 1
        mutated.loc[mutated["game_id"].eq(3), "min_sec"] = "1:00"
        mutated_row = add_points_features(mutated).set_index("game_id").loc[3]
        for halflife in (10, 20):
            column = f"pts_per_min_ewm_hl_{halflife}"
            self.assertAlmostEqual(mutated_row[column], by_game.loc[3, column])

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
            CURRENT_POINTS_FEATURES
        )
        self.assertEqual(overlap, set())
        self.assertNotIn("dnp_rate_10", CURRENT_POINTS_FEATURES)
        self.assertIn(
            "predicted_minutes_oof",
            CURRENT_POINTS_FEATURES,
        )
        self.assertIn(
            "expected_points_rate",
            CURRENT_POINTS_FEATURES,
        )
        self.assertNotIn(
            "predicted_minutes_oof",
            DIRECT_POINTS_FEATURES,
        )


class PointsFeatureContractTests(unittest.TestCase):
    def test_current_contract_is_41_columns(self) -> None:
        self.assertEqual(len(CURRENT_POINTS_FEATURES), 41)
        self.assertEqual(len(CURRENT_POINTS_41), 41)
        self.assertEqual(CURRENT_POINTS_41, CURRENT_POINTS_FEATURES)

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
            CURRENT_POINTS_FEATURES,
            DIRECT_POINTS_FEATURES,
            LEAN_POINTS_FEATURES,
        ):
            self.assertEqual(
                forbidden.intersection(contract),
                set(),
            )

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
