"""Standalone points residual pools stay role-aware."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.points import (
    EXPECTED_STARTER_THRESHOLD,
    XGBoostPointsModel,
    build_points_residual_pools,
    expected_role,
)


class ExpectedRoleTests(unittest.TestCase):
    def test_threshold_and_missing(self) -> None:
        roles = expected_role(
            np.array([1.0, 0.5, 0.49, np.nan])
        )
        np.testing.assert_array_equal(
            roles,
            [1, 1, 0, -1],
        )
        self.assertEqual(EXPECTED_STARTER_THRESHOLD, 0.5)


class SoftRolePoolTests(unittest.TestCase):
    def test_soft_role_uses_starter_and_bench_locations(
        self,
    ) -> None:
        residuals = np.concatenate(
            [np.full(40, 10.0), np.full(40, -10.0)]
        )
        start_rate = np.concatenate(
            [np.ones(40), np.zeros(40)]
        )
        pools = build_points_residual_pools(
            residuals=residuals,
            start_rate_10=start_rate,
            hat_p=np.full(80, 16.0),
        )
        model = XGBoostPointsModel(league="nba")
        model.predict_mean = (  # type: ignore[method-assign]
            lambda rows: np.full(len(rows), 20.0)
        )
        model.set_residual_pools(pools)

        rows = pd.DataFrame(
            {"start_rate_10": [1.0, 0.0]}
        )
        samples = model.simulate(rows, simulations=800)
        self.assertTrue(np.allclose(samples[0], 30.0))
        self.assertTrue(np.allclose(samples[1], 10.0))

    def test_missing_start_rate_uses_global(self) -> None:
        residuals = np.concatenate(
            [np.full(40, 10.0), np.full(40, -10.0)]
        )
        start_rate = np.concatenate(
            [np.ones(40), np.zeros(40)]
        )
        pools = build_points_residual_pools(
            residuals=residuals,
            start_rate_10=start_rate,
            hat_p=np.zeros(80),
        )
        model = XGBoostPointsModel(league="nba")
        model.predict_mean = (  # type: ignore[method-assign]
            lambda rows: np.full(len(rows), 20.0)
        )
        model.set_residual_pools(pools)
        rows = pd.DataFrame({"start_rate_10": [np.nan]})
        samples = model.simulate(rows, simulations=2_000)
        residual_mean = samples.mean() - 20.0
        self.assertGreater(residual_mean, -4.0)
        self.assertLess(residual_mean, 4.0)
        self.assertGreater(samples.std(), 8.0)


class RoleByHatPoolTests(unittest.TestCase):
    def test_role_x_hat_p_backs_off_tiny_cells(self) -> None:
        residuals = np.concatenate(
            [
                np.full(50, 10.0),
                np.full(50, -10.0),
                np.array([50.0]),
            ]
        )
        start_rate = np.concatenate(
            [np.ones(50), np.zeros(50), np.ones(1)]
        )
        hat_p = np.concatenate(
            [
                np.full(50, 5.0),
                np.full(50, 5.0),
                np.array([25.0]),
            ]
        )
        pools = build_points_residual_pools(
            residuals=residuals,
            start_rate_10=start_rate,
            hat_p=hat_p,
            scheme="role_x_hat_p",
            min_pool_size=40,
        )
        cell = pools.pools.get((1, 3))
        self.assertIsNotNone(cell)
        self.assertEqual(len(cell), 1)

        model = XGBoostPointsModel(league="nba")
        model.predict_mean = (  # type: ignore[method-assign]
            lambda rows: np.full(len(rows), 25.0)
        )
        model.set_residual_pools(pools)
        rows = pd.DataFrame({"start_rate_10": [1.0]})
        samples = model.simulate(rows, simulations=1_200)
        residual_mean = samples.mean() - 25.0
        self.assertGreater(residual_mean, 8.0)
        self.assertLess(residual_mean, 14.0)

    def test_missing_start_rate_uses_global(self) -> None:
        residuals = np.concatenate(
            [np.full(50, 10.0), np.full(50, -10.0)]
        )
        start_rate = np.concatenate(
            [np.ones(50), np.zeros(50)]
        )
        pools = build_points_residual_pools(
            residuals=residuals,
            start_rate_10=start_rate,
            hat_p=np.full(100, 12.0),
            scheme="role_x_hat_p",
        )
        model = XGBoostPointsModel(league="nba")
        model.predict_mean = (  # type: ignore[method-assign]
            lambda rows: np.full(len(rows), 12.0)
        )
        model.set_residual_pools(pools)
        rows = pd.DataFrame({"start_rate_10": [np.nan]})
        samples = model.simulate(rows, simulations=2_000)
        residual_mean = samples.mean() - 12.0
        self.assertGreater(residual_mean, -3.0)
        self.assertLess(residual_mean, 3.0)


class PoolConstructionTests(unittest.TestCase):
    def test_build_does_not_need_joint_or_pts_std(
        self,
    ) -> None:
        residuals = np.concatenate(
            [np.full(40, 4.0), np.full(40, -4.0)]
        )
        start_rate = np.concatenate(
            [np.ones(40), np.zeros(40)]
        )
        pools = build_points_residual_pools(
            residuals=residuals,
            start_rate_10=start_rate,
            hat_p=np.full(80, 18.0),
        )
        self.assertIn((-1, -1), pools.pools)
        self.assertEqual(pools.scheme, "soft_role")
        self.assertNotIn("pts_std_10", dir(pools))
        self.assertNotIn(
            "joint_residual",
            pools.__dataclass_fields__,
        )


if __name__ == "__main__":
    unittest.main()
