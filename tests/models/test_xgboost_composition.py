"""XGBoost minutes allow overtime and keep stratified residual pools."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.minutes import XGBoostMinutesModel
from src.models.settlement import maximum_minutes


class XGBoostCompositionTests(unittest.TestCase):
    def test_minutes_model_allows_overtime(self) -> None:
        model = XGBoostMinutesModel(league="nba")
        self.assertEqual(
            model.regressor.maximum_prediction,
            maximum_minutes("nba"),
        )


class StratifiedMinutesSamplerTests(unittest.TestCase):
    def test_simulate_draws_from_matching_minute_bin(self) -> None:
        model = _minutes_model_with_center_feature()
        model.set_stratified_residual_pools(
            {
                0: np.full(80, -4.0),
                1: np.full(80, 3.0),
            },
            np.array([0.0, 20.0, 64.0]),
        )
        rows = pd.DataFrame({"pred": [8.0, 34.0]})
        draws = model.simulate(rows, simulations=50)
        self.assertEqual(draws.shape, (2, 50))
        np.testing.assert_allclose(draws[0], 4.0)
        np.testing.assert_allclose(draws[1], 37.0)

    def test_save_load_round_trip_keeps_stratified_pools(self) -> None:
        model = _minutes_model_with_center_feature()
        model.set_stratified_residual_pools(
            {
                0: np.array([-2.0, -2.0, -2.0]),
                1: np.array([5.0, 5.0, 5.0]),
            },
            np.array([0.0, 20.0, 64.0]),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "xgboost_minutes_distribution.joblib"
            model.save(path)
            loaded = XGBoostMinutesModel.load(path)

        rows = pd.DataFrame({"pred": [10.0, 30.0]})
        draws = loaded.simulate(rows, simulations=20)
        np.testing.assert_allclose(draws[0], 8.0)
        np.testing.assert_allclose(draws[1], 35.0)


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


if __name__ == "__main__":
    unittest.main()
