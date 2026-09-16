"""Hierarchical minute-bin × pregame-role residual pools."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.minutes import (
    EXPECTED_STARTER_THRESHOLD,
    HierarchicalMinutePools,
    XGBoostMinutesModel,
    build_hierarchical_minute_pools,
    expected_role,
)


class _CenterFromFeature:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(frame["pred"], dtype=float)


def _minutes_model_with_center_feature() -> XGBoostMinutesModel:
    model = XGBoostMinutesModel(
        league="nba",
        feature_columns=["pred"],
    )
    model.regressor.model = _CenterFromFeature()
    model.regressor.calibration_residuals = np.zeros(40)
    return model


class ExpectedRoleTests(unittest.TestCase):
    def test_threshold_and_missing(self) -> None:
        roles = expected_role(
            np.array([0.9, 0.5, 0.49, np.nan])
        )
        np.testing.assert_array_equal(roles, [1, 1, 0, -1])
        self.assertEqual(EXPECTED_STARTER_THRESHOLD, 0.5)


class HierarchicalMinutePoolTests(unittest.TestCase):
    def test_starter_and_bench_draw_from_role_leaves(self) -> None:
        residuals = np.concatenate(
            [np.full(50, 5.0), np.full(50, -3.0)]
        )
        minutes_hat = np.full(100, 10.0)
        start_rate = np.concatenate(
            [np.ones(50), np.zeros(50)]
        )
        pools = build_hierarchical_minute_pools(
            residuals=residuals,
            minutes_hat=minutes_hat,
            start_rate_10=start_rate,
            bins=np.array([0.0, 20.0, 64.0]),
            min_pool_size=40,
        )
        model = _minutes_model_with_center_feature()
        model.set_hierarchical_residual_pools(pools)
        rows = pd.DataFrame(
            {
                "pred": [10.0, 10.0],
                "start_rate_10": [1.0, 0.0],
            }
        )
        draws = model.simulate(rows, simulations=40)
        np.testing.assert_allclose(draws[0], 15.0)
        np.testing.assert_allclose(draws[1], 7.0)

    def test_small_leaf_backoffs_to_parent_bin(self) -> None:
        residuals = np.concatenate(
            [np.full(10, 7.0), np.full(50, -4.0)]
        )
        minutes_hat = np.full(60, 10.0)
        start_rate = np.concatenate(
            [np.ones(10), np.zeros(50)]
        )
        pools = build_hierarchical_minute_pools(
            residuals=residuals,
            minutes_hat=minutes_hat,
            start_rate_10=start_rate,
            bins=np.array([0.0, 20.0, 64.0]),
            min_pool_size=40,
        )
        self.assertNotIn((1, 0), pools.leaf_pools)
        self.assertIn(0, pools.parent_pools)

        model = _minutes_model_with_center_feature()
        model.set_hierarchical_residual_pools(pools)
        rows = pd.DataFrame(
            {
                "pred": [10.0],
                "start_rate_10": [1.0],
            }
        )
        draws = model.simulate(rows, simulations=80)
        unique = set(np.round(draws[0] - 10.0, 8).tolist())
        self.assertNotEqual(unique, {7.0})
        self.assertTrue(unique <= {7.0, -4.0})

    def test_missing_start_rate_backoffs_to_bin_pool(self) -> None:
        residuals = np.concatenate(
            [np.full(50, 5.0), np.full(50, -3.0)]
        )
        minutes_hat = np.full(100, 10.0)
        start_rate = np.concatenate(
            [np.ones(50), np.zeros(50)]
        )
        pools = build_hierarchical_minute_pools(
            residuals=residuals,
            minutes_hat=minutes_hat,
            start_rate_10=start_rate,
            bins=np.array([0.0, 20.0, 64.0]),
            min_pool_size=40,
        )
        model = _minutes_model_with_center_feature()
        model.set_hierarchical_residual_pools(pools)
        rows = pd.DataFrame({"pred": [10.0]})
        draws = model.simulate(rows, simulations=200)
        unique = set(np.round(draws[0] - 10.0, 8).tolist())
        self.assertEqual(unique, {5.0, -3.0})

    def test_raw_residuals_are_not_sqrt_scaled(self) -> None:
        residuals = np.full(50, 4.0)
        minutes_hat = np.full(50, 16.0)
        start_rate = np.ones(50)
        pools = build_hierarchical_minute_pools(
            residuals=residuals,
            minutes_hat=minutes_hat,
            start_rate_10=start_rate,
            bins=np.array([0.0, 20.0, 64.0]),
            min_pool_size=40,
        )
        model = _minutes_model_with_center_feature()
        self.assertEqual(model.regressor.residual_scaling, "raw")
        model.set_hierarchical_residual_pools(pools)
        rows = pd.DataFrame(
            {
                "pred": [16.0],
                "start_rate_10": [1.0],
            }
        )
        draws = model.simulate(rows, simulations=20)
        np.testing.assert_allclose(draws[0], 20.0)

    def test_hierarchical_pools_do_not_center_residuals(self) -> None:
        residuals = np.full(50, 3.0)
        pools = build_hierarchical_minute_pools(
            residuals=residuals,
            minutes_hat=np.full(50, 8.0),
            start_rate_10=np.ones(50),
            bins=np.array([0.0, 20.0, 64.0]),
        )
        leaf = pools.leaf_pools[(1, 0)]
        self.assertAlmostEqual(float(np.mean(leaf)), 3.0)


class HierarchicalPoolTypeTests(unittest.TestCase):
    def test_set_hierarchical_returns_model(self) -> None:
        pools = HierarchicalMinutePools(
            bins=np.array([0.0, 20.0, 64.0]),
            leaf_pools={},
            parent_pools={0: np.array([-1.0, -1.0, -1.0])},
            global_pool=np.array([-1.0, -1.0, -1.0]),
            min_pool_size=40,
            shrinkage=20.0,
        )
        model = _minutes_model_with_center_feature()
        returned = model.set_hierarchical_residual_pools(pools)
        self.assertIs(returned, model)


if __name__ == "__main__":
    unittest.main()
