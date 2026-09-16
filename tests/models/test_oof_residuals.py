"""Expanding-window OOF residuals are chronological."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.core import (
    CalibratedXGBoostRegressor,
    XGBoostConfig,
    _expanding_oof_residuals,
)


def _linear_frame(
    *,
    n_dates: int,
    rows_per_date: int,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    records = []
    for date in dates:
        for _ in range(rows_per_date):
            x = rng.normal()
            noise = rng.normal(scale=0.1)
            records.append(
                {
                    "game_date": date,
                    "x": x,
                    "z": rng.normal(),
                    "y": 0.5 * x + noise,
                }
            )
    return pd.DataFrame(records)


class ExpandingOOFHelperTests(unittest.TestCase):
    def test_helper_enforces_chronology(self) -> None:
        frame = _linear_frame(n_dates=80, rows_per_date=2)
        target = frame["y"].to_numpy()
        dates = frame["game_date"].to_numpy()
        max_train_by_fold: dict[int, pd.Timestamp] = {}

        def predict_fn(
            train: pd.DataFrame,
            valid: pd.DataFrame,
        ) -> np.ndarray:
            train_max = pd.Timestamp(train["game_date"].max())
            valid_min = pd.Timestamp(valid["game_date"].min())
            self.assertLess(train_max, valid_min)
            fold = len(max_train_by_fold)
            max_train_by_fold[fold] = train_max
            return np.full(len(valid), float(train["y"].mean()))

        result = _expanding_oof_residuals(
            frame,
            target,
            dates,
            predict_fn,
        )
        self.assertEqual(len(result.residuals), len(result.predictions))
        self.assertEqual(len(result.residuals), len(result.fold_ids))
        self.assertGreater(len(result.residuals), 0)
        self.assertTrue(np.all(result.fold_ids >= 0))

    def test_helper_raises_on_leaking_split(self) -> None:
        frame = _linear_frame(n_dates=12, rows_per_date=2)
        leaking = [
            (frame.index[6:], frame.index[:6]),
        ]

        def predict_fn(
            train: pd.DataFrame,
            valid: pd.DataFrame,
        ) -> np.ndarray:
            return np.zeros(len(valid))

        with self.assertRaises(ValueError) as raised:
            _expanding_oof_residuals(
                frame,
                frame["y"].to_numpy(),
                frame["game_date"].to_numpy(),
                predict_fn,
                splits=leaking,
            )
        self.assertIn("chronology", str(raised.exception).lower())


class ExpandingOOFFitTests(unittest.TestCase):
    def test_fit_uses_expanding_oof_when_dates_allow(self) -> None:
        frame = _linear_frame(n_dates=80, rows_per_date=3)
        model = CalibratedXGBoostRegressor(
            ["x", "z"],
            config=XGBoostConfig(
                n_estimators=20,
                early_stopping_rounds=5,
                min_child_weight=1.0,
                max_depth=2,
                learning_rate=0.1,
            ),
        )
        model.fit(frame, target_column="y")
        self.assertEqual(model.residual_source, "expanding_oof")
        self.assertIsNotNone(model.calibration_residuals)
        self.assertIsNotNone(model._oof_predictions)
        self.assertIsNotNone(model.oof_fold_ids)
        self.assertIsNotNone(model._oof_dates)
        self.assertIsNotNone(model._oof_max_train_date_by_fold)

        for fold_id, residual_date in zip(
            model.oof_fold_ids,
            model._oof_dates,
        ):
            max_train = model._oof_max_train_date_by_fold[int(fold_id)]
            self.assertLess(
                pd.Timestamp(max_train),
                pd.Timestamp(residual_date),
            )

    def test_fit_falls_back_when_dates_are_few(self) -> None:
        frame = _linear_frame(n_dates=20, rows_per_date=8)
        model = CalibratedXGBoostRegressor(
            ["x", "z"],
            config=XGBoostConfig(
                n_estimators=20,
                early_stopping_rounds=5,
                min_child_weight=1.0,
                max_depth=2,
                learning_rate=0.1,
            ),
        )
        model.fit(frame, target_column="y")
        self.assertEqual(
            model.residual_source,
            "early_stopping_holdout",
        )
        self.assertGreaterEqual(len(model.calibration_residuals), 30)


if __name__ == "__main__":
    unittest.main()
