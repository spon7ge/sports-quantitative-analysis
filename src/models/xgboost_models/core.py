"""Shared training and probability logic for XGBoost models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class XGBoostConfig:
    # M-T2: check whether 750 is cap-binding before raising it.
    n_estimators: int = 750
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 10.0
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    reg_alpha: float = 0.10
    reg_lambda: float = 5.0
    calibration_fraction: float = 0.20
    early_stopping_rounds: int = 100
    simulations: int = 10_000
    random_seed: int = 42
    oof_folds: int = 5
    oof_minimum_training_dates: int = 60


@dataclass(frozen=True)
class ExpandingOOFResult:
    residuals: np.ndarray
    predictions: np.ndarray
    fold_ids: np.ndarray
    dates: np.ndarray
    max_train_date_by_fold: dict[int, pd.Timestamp]

class CalibratedXGBoostRegressor:
    """XGBoost point model with out-of-time residual simulation."""

    def __init__(
        self,
        feature_columns: list[str],
        *,
        config: XGBoostConfig | None = None,
        objective: str = "reg:squarederror",
        residual_scaling: str = "raw",
        minimum_prediction: float = 0.0,
        maximum_prediction: float | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.config = config or XGBoostConfig()
        self.objective = objective
        self.residual_scaling = residual_scaling
        self.minimum_prediction = minimum_prediction
        self.maximum_prediction = maximum_prediction

        self.model: XGBRegressor | None = None
        self.calibration_residuals: np.ndarray | None = None
        self.training_cutoff: pd.Timestamp | None = None
        self.residual_pools: dict[int, np.ndarray] | None = None
        self.residual_pool_bins: np.ndarray | None = None
        self.residual_source: str | None = None
        self._oof_predictions: np.ndarray | None = None
        self.oof_fold_ids: np.ndarray | None = None
        self._oof_dates: np.ndarray | None = None
        self._oof_max_train_date_by_fold: dict[int, pd.Timestamp] | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        date_column: str = "game_date",
        calibrate_residuals: bool = True,
    ) -> CalibratedXGBoostRegressor:
        """Train and calibrate using a chronological split.

        Set ``calibrate_residuals=False`` for throwaway OOF
        mean models that never simulate.
        """
        self._validate_columns(
            frame,
            {
                target_column,
                date_column,
                *self.feature_columns,
            },
        )

        data = frame.copy()
        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        )
        data[target_column] = pd.to_numeric(
            data[target_column],
            errors="coerce",
        )

        data = data.dropna(
            subset=[date_column, target_column]
        )
        data = data.sort_values(date_column)

        unique_dates = np.sort(
            data[date_column].unique()
        )

        if len(unique_dates) < 10:
            raise ValueError(
                "At least 10 distinct dates are required."
            )

        split_index = int(
            len(unique_dates)
            * (1 - self.config.calibration_fraction)
        )
        split_index = min(
            max(split_index, 1),
            len(unique_dates) - 1,
        )

        cutoff = pd.Timestamp(
            unique_dates[split_index]
        )
        self.training_cutoff = cutoff

        development = data.loc[
            data[date_column] < cutoff
        ]
        calibration = data.loc[
            data[date_column] >= cutoff
        ]

        if development.empty or calibration.empty:
            raise ValueError(
                "Chronological split produced an empty partition."
            )

        development_x = self._feature_matrix(
            development
        )
        calibration_x = self._feature_matrix(
            calibration
        )

        candidate = self._new_model(
            early_stopping=True
        )

        candidate.fit(
            development_x,
            development[target_column],
            eval_set=[
                (
                    calibration_x,
                    calibration[target_column],
                )
            ],
            verbose=False,
        )

        calibration_prediction = candidate.predict(
            calibration_x
        )
        calibration_prediction = self._clip(
            calibration_prediction
        )

        self.calibration_residuals = (
            self._calculate_residuals(
                calibration[target_column].to_numpy(),
                calibration_prediction,
            )
        )
        self.residual_source = "early_stopping_holdout"

        best_iteration = getattr(
            candidate,
            "best_iteration",
            None,
        )

        estimator_count = (
            best_iteration + 1
            if best_iteration is not None
            else self.config.n_estimators
        )

        self.model = self._new_model(
            early_stopping=False,
            n_estimators=estimator_count,
        )

        self.model.fit(
            self._feature_matrix(data),
            data[target_column],
            verbose=False,
        )

        if calibrate_residuals:
            self._replace_residuals_with_expanding_oof(
                data,
                target_column=target_column,
                date_column=date_column,
                n_estimators=estimator_count,
            )

        return self

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "Call fit() before predict()."
            )

        predictions = self.model.predict(
            self._feature_matrix(frame)
        )

        return self._clip(predictions)

    def simulate(
        self,
        frame: pd.DataFrame,
        *,
        simulations: int | None = None,
        random_seed: int | None = None,
        centers: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an array shaped `(rows, simulations)`."""
        if self.calibration_residuals is None:
            raise RuntimeError(
                "Call fit() before simulate()."
            )

        if centers is None:
            centers = self.predict(frame)

        centers = np.asarray(centers, dtype=float)
        sample_count = (
            simulations or self.config.simulations
        )

        rng = np.random.default_rng(
            random_seed or self.config.random_seed
        )

        if (
            centers.ndim == 1
            and self.residual_pools is not None
        ):
            return self._simulate_stratified(
                centers,
                sample_count=sample_count,
                rng=rng,
            )

        if centers.ndim == 2:
            sampled_residuals = rng.choice(
                self.calibration_residuals,
                size=centers.shape,
                replace=True,
            )
            residual_scale = np.sqrt(
                np.maximum(centers, 1.0)
            ) if self.residual_scaling == "sqrt_mean" else 1.0
            samples = centers + (
                sampled_residuals * residual_scale
            )
            return self._clip(samples)

        sampled_residuals = rng.choice(
            self.calibration_residuals,
            size=(len(centers), sample_count),
            replace=True,
        )

        if self.residual_scaling == "sqrt_mean":
            scale = np.sqrt(
                np.maximum(centers, 1.0)
            )[:, None]
            samples = centers[:, None] + (
                sampled_residuals * scale
            )
        else:
            samples = (
                centers[:, None]
                + sampled_residuals
            )

        return self._clip(samples)

    def set_residual_pools(
        self,
        pools: dict[int, np.ndarray],
        bins: np.ndarray,
    ) -> CalibratedXGBoostRegressor:
        """Attach heteroskedastic residual pools for simulate()."""
        if len(pools) == 0:
            raise ValueError("residual pools must not be empty")

        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(
                "residual pool bins must be a 1-d increasing edge array"
            )
        if np.any(np.diff(edges) <= 0):
            raise ValueError(
                "residual pool bins must be strictly increasing"
            )

        self.residual_pool_bins = edges
        self.residual_pools = {
            int(bin_id): np.asarray(pool, dtype=float)
            for bin_id, pool in pools.items()
            if len(pool) > 0
        }
        if not self.residual_pools:
            raise ValueError("residual pools must not be empty")
        return self

    def _simulate_stratified(
        self,
        centers: np.ndarray,
        *,
        sample_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if (
            self.residual_pools is None
            or self.residual_pool_bins is None
        ):
            raise RuntimeError(
                "set_residual_pools() before stratified simulate()."
            )

        bins = np.clip(
            np.digitize(centers, self.residual_pool_bins) - 1,
            0,
            len(self.residual_pool_bins) - 2,
        )
        fallback = np.concatenate(
            list(self.residual_pools.values())
        )
        draws = np.empty((len(centers), sample_count), dtype=float)
        for bin_id in np.unique(bins):
            mask = bins == bin_id
            pool = self.residual_pools.get(int(bin_id), fallback)
            sampled = rng.choice(
                pool,
                size=(int(mask.sum()), sample_count),
                replace=True,
            )
            draws[mask] = centers[mask, None] + sampled
        return self._clip(draws)

    def _calculate_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> np.ndarray:
        residuals = actual - predicted

        if self.residual_scaling == "sqrt_mean":
            residuals = residuals / np.sqrt(
                np.maximum(predicted, 1.0)
            )

        residuals = residuals[
            np.isfinite(residuals)
        ]

        if len(residuals) < 30:
            raise ValueError(
                "At least 30 calibration residuals are required."
            )

        # Remove only extreme data errors while retaining genuine
        # injuries, foul trouble, and rotation volatility.
        lower, upper = np.quantile(
            residuals,
            [0.005, 0.995],
        )

        return np.clip(
            residuals,
            lower,
            upper,
        )

    def _new_model(
        self,
        *,
        early_stopping: bool,
        n_estimators: int | None = None,
    ) -> XGBRegressor:
        arguments = {
            "objective": self.objective,
            "n_estimators": (
                n_estimators
                or self.config.n_estimators
            ),
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "min_child_weight": (
                self.config.min_child_weight
            ),
            "subsample": self.config.subsample,
            "colsample_bytree": (
                self.config.colsample_bytree
            ),
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.config.random_seed,
            "tree_method": "hist",
            "n_jobs": -1,
        }
        if self.objective == "reg:squarederror":
            arguments["eval_metric"] = "mae"

        if early_stopping:
            arguments["early_stopping_rounds"] = (
                self.config.early_stopping_rounds
            )

        return XGBRegressor(**arguments)

    def _feature_matrix(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_columns(
            frame,
            set(self.feature_columns),
        )

        matrix = frame[
            self.feature_columns
        ].copy()

        for column in matrix.columns:
            matrix[column] = pd.to_numeric(
                matrix[column],
                errors="coerce",
            )

        return matrix

    def _clip(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        upper = (
            self.maximum_prediction
            if self.maximum_prediction is not None
            else np.inf
        )

        return np.clip(
            values,
            self.minimum_prediction,
            upper,
        )

    @staticmethod
    def _validate_columns(
        frame: pd.DataFrame,
        required: set[str],
    ) -> None:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

    def _replace_residuals_with_expanding_oof(
        self,
        data: pd.DataFrame,
        *,
        target_column: str,
        date_column: str,
        n_estimators: int,
    ) -> None:
        unique_dates = np.sort(data[date_column].unique())
        minimum_training_dates = (
            self.config.oof_minimum_training_dates
        )
        if len(unique_dates) < minimum_training_dates + 5:
            return

        def predict_fn(
            train: pd.DataFrame,
            valid: pd.DataFrame,
        ) -> np.ndarray:
            fold_model = self._new_model(
                early_stopping=False,
                n_estimators=n_estimators,
            )
            fold_model.fit(
                self._feature_matrix(train),
                train[target_column],
                verbose=False,
            )
            return self._clip(
                fold_model.predict(
                    self._feature_matrix(valid)
                )
            )

        try:
            result = _expanding_oof_residuals(
                data,
                data[target_column].to_numpy(),
                data[date_column].to_numpy(),
                predict_fn,
                folds=self.config.oof_folds,
                minimum_training_dates=minimum_training_dates,
            )
        except ValueError as exc:
            if "chronology" in str(exc).lower():
                raise
            return

        if len(result.residuals) < 30:
            return

        actuals = result.residuals + result.predictions
        try:
            calibrated = self._calculate_residuals(
                actuals,
                result.predictions,
            )
        except ValueError:
            return

        self.calibration_residuals = calibrated
        self.residual_source = "expanding_oof"
        self._oof_predictions = result.predictions
        self.oof_fold_ids = result.fold_ids
        self._oof_dates = result.dates
        self._oof_max_train_date_by_fold = (
            result.max_train_date_by_fold
        )


def _assert_fold_chronology(
    train_dates: np.ndarray,
    valid_dates: np.ndarray,
) -> None:
    if len(train_dates) == 0 or len(valid_dates) == 0:
        raise ValueError(
            "OOF fold produced an empty partition."
        )
    train_max = pd.Timestamp(np.max(train_dates))
    valid_min = pd.Timestamp(np.min(valid_dates))
    if train_max >= valid_min:
        raise ValueError(
            "OOF fold violates chronology: "
            "max(train date) must be < min(valid date)."
        )


def _expanding_oof_residuals(
    frame: pd.DataFrame,
    target: np.ndarray,
    dates: np.ndarray,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    *,
    folds: int = 5,
    minimum_training_dates: int = 60,
    splits: Sequence[tuple[pd.Index, pd.Index]] | None = None,
) -> ExpandingOOFResult:
    """Expanding-window OOF residuals with chronology checks.

    ``predict_fn(train_frame, valid_frame)`` returns predictions
    for ``valid_frame``. Every residual comes from a model that
    never trained on that row's date or later.
    """
    target_arr = np.asarray(target, dtype=float)
    date_arr = pd.to_datetime(np.asarray(dates))
    if len(target_arr) != len(frame) or len(date_arr) != len(frame):
        raise ValueError(
            "target and dates must align with frame"
        )

    work = frame.copy()
    work["_oof_target"] = target_arr
    work["_oof_date"] = date_arr

    if splits is None:
        from src.models.xgboost_models.tuning import (
            expanding_window_splits,
        )

        splits = expanding_window_splits(
            work,
            date_column="_oof_date",
            folds=folds,
            minimum_training_dates=minimum_training_dates,
        )

    if not splits:
        raise ValueError("No expanding OOF splits.")

    oof_residuals: list[np.ndarray] = []
    oof_predictions: list[np.ndarray] = []
    oof_fold_ids: list[np.ndarray] = []
    oof_dates: list[np.ndarray] = []
    max_train_date_by_fold: dict[int, pd.Timestamp] = {}

    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        train = work.loc[train_idx]
        valid = work.loc[valid_idx]
        train_dates = train["_oof_date"].to_numpy()
        valid_dates = valid["_oof_date"].to_numpy()
        _assert_fold_chronology(train_dates, valid_dates)

        predictions = np.asarray(
            predict_fn(train, valid),
            dtype=float,
        )
        if len(predictions) != len(valid):
            raise ValueError(
                "predict_fn must return one prediction per valid row"
            )

        actuals = valid["_oof_target"].to_numpy(dtype=float)
        oof_residuals.append(actuals - predictions)
        oof_predictions.append(predictions)
        oof_fold_ids.append(
            np.full(len(valid), fold_id, dtype=int)
        )
        oof_dates.append(valid_dates)
        max_train_date_by_fold[fold_id] = pd.Timestamp(
            np.max(train_dates)
        )

    return ExpandingOOFResult(
        residuals=np.concatenate(oof_residuals),
        predictions=np.concatenate(oof_predictions),
        fold_ids=np.concatenate(oof_fold_ids),
        dates=np.concatenate(oof_dates),
        max_train_date_by_fold=max_train_date_by_fold,
    )