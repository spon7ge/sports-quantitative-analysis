"""Joint minutes → points simulator (appearance-conditional)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.models.settlement import maximum_minutes
from src.models.xgboost_models.joint_simulation import (
    JointPointsSimulator,
    build_points_inference_features,
    expected_minutes,
    select_bin,
)
from tests.models.xgboost_models.joint_fixtures import (
    MINUTES_BINS,
    centered_pool,
    epsilon_pools,
    feature_row,
    load_bundle,
    make_overlay,
    minutes_pools,
    write_bundle,
)

NBA_MAX_MINUTES = maximum_minutes("nba")


def _asymmetric_pool() -> np.ndarray:
    raw = np.array([-8.0] * 39 + [8.0 * 39], dtype=float)
    return raw - raw.mean()


class MinutesMeanTests(unittest.TestCase):
    def test_predicts_scalar_hatm_clipped_to_nba_range(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(Path(tmp), minutes_value=80.0)
            )
            simulator = JointPointsSimulator(bundle)
            hat_m = simulator.predict_minutes_mean(feature_row())
        self.assertIsInstance(hat_m, float)
        self.assertEqual(hat_m, NBA_MAX_MINUTES)

        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(Path(tmp), minutes_value=-4.0)
            )
            hat_m = JointPointsSimulator(bundle).predict_minutes_mean(
                feature_row()
            )
        self.assertEqual(hat_m, 0.0)


class BinSelectionTests(unittest.TestCase):
    def test_left_closed_right_open_boundaries(self) -> None:
        expected = {
            0.0: 0,
            12.0: 1,
            18.0: 2,
            24.0: 3,
            30.0: 4,
            36.0: 5,
            63.0: 5,
        }
        for value, bin_id in expected.items():
            self.assertEqual(
                select_bin(value, MINUTES_BINS),
                bin_id,
                msg=f"hatM={value}",
            )


class MinutesExpectationTests(unittest.TestCase):
    def test_asymmetric_pool_post_clip_mean_differs_from_hatm(
        self,
    ) -> None:
        hat_m = 5.0
        pool = _asymmetric_pool()
        post_clip = expected_minutes(hat_m, pool)
        self.assertNotAlmostEqual(post_clip, hat_m)
        self.assertAlmostEqual(
            post_clip,
            float(np.mean(np.clip(hat_m + pool, 0.0, NBA_MAX_MINUTES))),
        )

    def test_centering_ignores_draw_count(self) -> None:
        with TemporaryDirectory() as tmp:
            pools = minutes_pools()
            pools[0] = _asymmetric_pool()
            bundle = load_bundle(
                write_bundle(
                    Path(tmp),
                    minutes_value=5.0,
                    minutes_residual_pools=pools,
                )
            )
            simulator = JointPointsSimulator(bundle, n_draws=50)
            row = feature_row()
            few = simulator.simulate(row, n_draws=20)
            many = simulator.simulate(row, n_draws=400)
        self.assertAlmostEqual(few.expected_minutes, many.expected_minutes)
        self.assertNotAlmostEqual(few.expected_minutes, few.hat_m)

    def test_beta_perturbation_has_zero_distributional_expectation(
        self,
    ) -> None:
        hat_m = 5.0
        pool = _asymmetric_pool()
        draws = np.clip(hat_m + pool, 0.0, NBA_MAX_MINUTES)
        shock_center = float(draws.mean())
        beta = 0.45
        self.assertAlmostEqual(
            float(np.mean(beta * (draws - shock_center))),
            0.0,
            places=12,
        )


class PointsFeatureTests(unittest.TestCase):
    def test_inference_uses_scalar_hatm_not_stale_or_actual_minutes(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            simulator = JointPointsSimulator(bundle)
            row = feature_row(
                predicted_minutes_oof=99.0,
                minutes=41.0,
                pts=33.0,
                is_starter=1,
            )
            result = simulator.simulate(row, n_draws=32)
            seen = bundle.points_mean.regressor.model.seen

        self.assertEqual(len(seen), 1)
        passed = seen[0]
        self.assertEqual(len(passed), 1)
        self.assertEqual(
            float(passed.iloc[0]["predicted_minutes_oof"]),
            result.hat_m,
        )
        self.assertNotEqual(result.hat_m, 99.0)
        self.assertNotEqual(result.hat_m, 41.0)
        self.assertNotIn("minutes", passed.columns)
        self.assertFalse(
            np.allclose(
                passed["predicted_minutes_oof"].to_numpy(),
                result.minute_draws,
            )
        )
        self.assertAlmostEqual(
            float(passed.iloc[0]["expected_points_rate"]),
            result.hat_m * 0.55,
        )
        self.assertAlmostEqual(
            float(passed.iloc[0]["expected_attempt_volume"]),
            result.hat_m * 0.45,
        )

    def test_feature_vector_and_hatp_are_deterministic(self) -> None:
        hat_m = 24.0
        row = pd.Series(feature_row())
        features = build_points_inference_features(
            row,
            hat_m,
            feature_columns=[
                "predicted_minutes_oof",
                "min_mean_10",
                "pts_per_min_10",
                "fga_per_min_10",
                "expected_points_rate",
                "expected_attempt_volume",
            ],
        )
        self.assertEqual(
            float(features.iloc[0]["predicted_minutes_oof"]),
            24.0,
        )
        self.assertAlmostEqual(
            float(features.iloc[0]["expected_points_rate"]),
            24.0 * 0.55,
        )
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(Path(tmp), points_value=18.0)
            )
            hat_p = JointPointsSimulator(bundle).predict_points_mean(
                feature_row(),
                hat_m=24.0,
            )
        self.assertEqual(hat_p, 18.0)


class OverlayAndResidualTests(unittest.TestCase):
    def test_overlay_g_is_not_clipped(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(
                    Path(tmp),
                    overlay=make_overlay(intercept=-4.0, coef=np.zeros(4)),
                )
            )
            result = JointPointsSimulator(bundle).simulate(
                feature_row(),
                n_draws=16,
            )
        self.assertLess(result.g, 0.0)
        self.assertEqual(result.g, -4.0)

    def test_epsilon_pool_uses_clipped_mu(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(
                    Path(tmp),
                    points_value=2.0,
                    overlay=make_overlay(intercept=-5.0, coef=np.zeros(4)),
                )
            )
            result = JointPointsSimulator(bundle).simulate(
                feature_row(),
                n_draws=16,
            )
        self.assertEqual(result.mu, 0.0)
        self.assertEqual(result.epsilon_bin, 0)

    def test_epsilon_samples_are_not_recentered(self) -> None:
        pool = centered_pool()
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(
                    Path(tmp),
                    overlay=make_overlay(intercept=0.0, coef=np.zeros(4)),
                    epsilon=epsilon_pools(lambda: pool.copy()),
                    beta_by_bin={
                        bin_id: 0.0 for bin_id in range(6)
                    },
                )
            )
            result = JointPointsSimulator(
                bundle,
                seed=7,
            ).simulate(feature_row(), n_draws=200)
        unique = np.unique(np.round(result.epsilon_draws, 10))
        self.assertTrue(
            np.all(np.isin(unique, np.round(pool, 10)))
        )

    def test_negative_base_uses_hatp_plus_g_then_clip(self) -> None:
        zeros = np.zeros(40)
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(
                    Path(tmp),
                    minutes_value=24.0,
                    points_value=2.0,
                    overlay=make_overlay(intercept=-5.0, coef=np.zeros(4)),
                    minutes_residual_pools=minutes_pools(lambda: zeros.copy()),
                    epsilon=epsilon_pools(
                        lambda: np.array([-2.0] * 20 + [2.0] * 20)
                    ),
                    beta_by_bin={bin_id: 0.0 for bin_id in range(6)},
                )
            )
            result = JointPointsSimulator(bundle).simulate(
                feature_row(),
                n_draws=200,
            )
        self.assertEqual(result.g, -5.0)
        self.assertEqual(result.mu, 0.0)
        self.assertTrue(np.all(result.point_draws == 0.0))


class DrawIdentityTests(unittest.TestCase):
    def test_row_order_does_not_change_player_draws(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            simulator = JointPointsSimulator(bundle, seed=42, n_draws=64)
            first = feature_row(player_id=1, game_id="g-a")
            second = feature_row(player_id=2, game_id="g-b")
            a_then_b = simulator.simulate([first, second])
            extra = feature_row(player_id=9, game_id="g-z")
            shuffled = simulator.simulate([second, extra, first, first])

        by_key = {
            (item.player_id, item.game_id): item
            for item in a_then_b
        }
        shuffled_first = [
            item
            for item in shuffled
            if (item.player_id, item.game_id) == (1, "g-a")
        ]
        np.testing.assert_array_equal(
            by_key[(1, "g-a")].point_draws,
            shuffled_first[0].point_draws,
        )
        np.testing.assert_array_equal(
            shuffled_first[0].point_draws,
            shuffled_first[1].point_draws,
        )

    def test_draws_are_finite_nonnegative_with_clip_rates(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            result = JointPointsSimulator(bundle).simulate(
                feature_row(),
                n_draws=100,
            )
        self.assertTrue(np.isfinite(result.minute_draws).all())
        self.assertTrue(np.isfinite(result.point_draws).all())
        self.assertGreaterEqual(result.minute_draws.min(), 0.0)
        self.assertLessEqual(result.minute_draws.max(), NBA_MAX_MINUTES)
        self.assertGreaterEqual(result.point_draws.min(), 0.0)
        self.assertGreaterEqual(result.minutes_clip_rate, 0.0)
        self.assertLessEqual(result.minutes_clip_rate, 1.0)
        self.assertGreaterEqual(result.points_clip_rate, 0.0)
        self.assertEqual(result.n_draws, 100)
        self.assertEqual(result.seed, 42)
        self.assertEqual(result.bundle_hash, bundle.bundle_hash)
        self.assertTrue(np.isfinite(result.hat_m))
        self.assertTrue(np.isfinite(result.hat_p))
        self.assertTrue(np.isfinite(result.g))
        self.assertTrue(np.isfinite(result.mu))
        self.assertTrue(np.isfinite(result.beta))


class SampleMeanShockTests(unittest.TestCase):
    def test_clipped_sample_mean_centers_beta_shock(self) -> None:
        mild = np.array([-3.0] * 30 + [9.0] * 10)
        zeros = np.zeros(40)
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(
                write_bundle(
                    Path(tmp),
                    minutes_value=24.0,
                    points_value=18.0,
                    overlay=make_overlay(intercept=1.0, coef=np.zeros(4)),
                    minutes_residual_pools=minutes_pools(
                        lambda: mild.copy()
                    ),
                    epsilon=epsilon_pools(lambda: zeros.copy()),
                    beta_by_bin={bin_id: 0.8 for bin_id in range(6)},
                )
            )
            result = JointPointsSimulator(bundle).simulate(
                feature_row(),
                n_draws=200,
            )
        shock_center = float(result.minute_draws.mean())
        shock = result.beta * (result.minute_draws - shock_center)
        self.assertAlmostEqual(float(shock.mean()), 0.0, places=10)
        self.assertGreaterEqual(result.point_draws.min(), 0.0)
        self.assertAlmostEqual(
            float(result.point_draws.mean()),
            result.hat_p + result.g,
            places=6,
        )

    def test_missing_overlay_and_beta_are_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            bundle.joint_calibration.overlay = None
            bundle.joint_calibration.beta = None
            result = JointPointsSimulator(bundle).simulate(
                feature_row(),
                n_draws=32,
            )
        self.assertEqual(result.g, 0.0)
        self.assertEqual(result.beta, 0.0)


class SimulatePointsTests(unittest.TestCase):
    def test_simulate_points_returns_float32_matrix(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            simulator = JointPointsSimulator(bundle)
            rows = pd.DataFrame([feature_row(), feature_row()])
            draws = simulator.simulate_points(rows, n_draws=16)
        self.assertEqual(draws.shape, (2, 16))
        self.assertEqual(draws.dtype, np.float32)
        self.assertTrue(np.all(draws >= 0.0))

    def test_batch_means_match_single_row(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            simulator = JointPointsSimulator(bundle)
            rows = pd.DataFrame(
                [
                    feature_row(min_mean_10=30.0),
                    feature_row(min_mean_10=18.0),
                ]
            )
            hat_m = simulator.predict_minutes_means(rows)
            hat_p = simulator.predict_points_means(rows, hat_m=hat_m)
        for index, row in rows.iterrows():
            self.assertEqual(
                hat_m[index],
                simulator.predict_minutes_mean(row),
            )
            self.assertEqual(
                hat_p[index],
                simulator.predict_points_mean(row, hat_m=float(hat_m[index])),
            )

