"""Shared fake artifacts for joint points tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.xgboost_models.artifact_bundle import (
    JointPointsArtifactBundle,
    load_joint_points_bundle,
)
from src.models.xgboost_models.joint_calibration import (
    HOLDOUT_SEASON,
    BetaMap,
    EpsilonPools,
    JointCalibration,
    OverlayFit,
    content_hash,
    save_joint_calibration,
)
from src.models.xgboost_models.minutes import XGBoostMinutesModel
from src.models.xgboost_models.points import XGBoostPointsModel

MINUTES_FEATURES = [
    "min_mean_10",
    "start_rate_10",
    "usg_wmean_10",
]
POINTS_FEATURES = [
    "predicted_minutes_oof",
    "min_mean_10",
    "pts_per_min_10",
    "fga_per_min_10",
    "usg_wmean_10",
    "start_rate_10",
    "expected_points_rate",
    "expected_attempt_volume",
]
MINUTES_BINS = np.array(
    [0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0]
)
EPSILON_BINS = np.array([0.0, 8.0, 14.0, 20.0, 28.0, np.inf])


class ConstantModel:
    def __init__(self, value: float, n_estimators: int) -> None:
        self.value = value
        self.n_estimators = n_estimators
        self.seen: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.seen.append(frame.copy())
        return np.full(len(frame), self.value)


def centered_pool(size: int = 40, scale: float = 2.0) -> np.ndarray:
    raw = np.linspace(-scale, scale, size)
    return raw - raw.mean()


def minutes_pools(
    factory=None,
) -> dict[int, np.ndarray]:
    build = factory or centered_pool
    return {
        bin_id: build()
        for bin_id in range(len(MINUTES_BINS) - 1)
    }


def epsilon_pools(
    factory=None,
) -> dict[int, np.ndarray]:
    build = factory or (lambda: centered_pool(scale=1.5))
    return {
        bin_id: build()
        for bin_id in range(len(EPSILON_BINS) - 1)
    }


def make_minutes_model(
    *,
    value: float = 24.0,
    n_estimators: int = 750,
    with_pools: bool = False,
    pools: dict[int, np.ndarray] | None = None,
) -> XGBoostMinutesModel:
    model = XGBoostMinutesModel(
        league="nba",
        feature_columns=list(MINUTES_FEATURES),
    )
    model.regressor.model = ConstantModel(value, n_estimators)
    model.regressor.training_cutoff = pd.Timestamp("2024-03-18")
    model.regressor.calibration_residuals = centered_pool()
    if with_pools or pools is not None:
        model.set_stratified_residual_pools(
            pools if pools is not None else minutes_pools(),
            MINUTES_BINS,
        )
    return model


def make_points_model(
    *,
    value: float = 18.0,
    n_estimators: int = 581,
) -> XGBoostPointsModel:
    model = XGBoostPointsModel(
        league="nba",
        feature_columns=list(POINTS_FEATURES),
    )
    model.regressor.model = ConstantModel(value, n_estimators)
    model.regressor.training_cutoff = pd.Timestamp("2024-03-18")
    model.regressor.calibration_residuals = centered_pool()
    return model


def make_overlay(
    *,
    intercept: float = 0.0,
    coef: np.ndarray | None = None,
) -> OverlayFit:
    return OverlayFit(
        coef=np.asarray(
            coef if coef is not None else [0.4, 0.1, 0.05, -0.2],
            dtype=float,
        ),
        intercept=float(intercept),
        feature_names=(
            "minutes_shock",
            "pts_per_min_10",
            "usg_wmean_10",
            "start_rate_10",
        ),
        impute_median=np.zeros(4),
        scale_mean=np.zeros(4),
        scale_std=np.ones(4),
        alpha=1.0,
    )


def make_joint_calibration(
    minutes_mean_path: Path,
    minutes_dist_path: Path,
    points_path: Path,
    *,
    overlay: OverlayFit | None = None,
    epsilon: dict[int, np.ndarray] | None = None,
    beta_by_bin: dict[int, float] | None = None,
) -> JointCalibration:
    n_minute_bins = len(MINUTES_BINS) - 1
    return JointCalibration(
        schema_version=1,
        holdout_season=HOLDOUT_SEASON,
        fitted_through="2024-25",
        fingerprints={
            "minutes_mean": {
                "content_hash": content_hash(minutes_mean_path),
                "feature_columns": list(MINUTES_FEATURES),
                "training_cutoff": "2024-03-18 00:00:00",
                "n_estimators": 750,
            },
            "minutes_distribution": {
                "content_hash": content_hash(minutes_dist_path),
                "feature_columns": list(MINUTES_FEATURES),
                "training_cutoff": "2024-03-18 00:00:00",
                "n_estimators": 750,
            },
            "points_mean": {
                "content_hash": content_hash(points_path),
                "feature_columns": list(POINTS_FEATURES),
                "training_cutoff": "2024-03-18 00:00:00",
                "n_estimators": 581,
            },
        },
        minutes_bins=MINUTES_BINS.copy(),
        overlay=overlay or make_overlay(),
        beta=BetaMap(
            beta_global=0.45,
            beta_by_bin=beta_by_bin
            or {bin_id: 0.45 for bin_id in range(n_minute_bins)},
            bins=MINUTES_BINS.copy(),
            shrinkage=20.0,
        ),
        epsilon=EpsilonPools(
            pools=epsilon if epsilon is not None else epsilon_pools(),
            bins=EPSILON_BINS.copy(),
            removed_means={
                bin_id: 0.0
                for bin_id in range(len(EPSILON_BINS) - 1)
            },
            raw_stds={
                bin_id: 1.0
                for bin_id in range(len(EPSILON_BINS) - 1)
            },
            winsor=(0.005, 0.995),
            min_pool_size=40,
            centered=True,
        ),
        epsilon_source="cross_fitted",
    )


def write_bundle(
    directory: Path,
    *,
    minutes_value: float = 24.0,
    points_value: float = 18.0,
    overlay: OverlayFit | None = None,
    minutes_residual_pools: dict[int, np.ndarray] | None = None,
    epsilon: dict[int, np.ndarray] | None = None,
    beta_by_bin: dict[int, float] | None = None,
) -> dict[str, Path]:
    paths = {
        "minutes_mean": directory / "xgboost_minutes.joblib",
        "minutes_distribution": (
            directory / "xgboost_minutes_distribution.joblib"
        ),
        "points_mean": directory / "xgboost_points.joblib",
        "joint_calibration": directory / "joint_calibration.joblib",
    }
    make_minutes_model(value=minutes_value).save(paths["minutes_mean"])
    make_minutes_model(
        value=minutes_value,
        with_pools=True,
        pools=minutes_residual_pools,
    ).save(paths["minutes_distribution"])
    joblib.dump(
        make_points_model(value=points_value),
        paths["points_mean"],
    )
    save_joint_calibration(
        make_joint_calibration(
            paths["minutes_mean"],
            paths["minutes_distribution"],
            paths["points_mean"],
            overlay=overlay,
            epsilon=epsilon,
            beta_by_bin=beta_by_bin,
        ),
        paths["joint_calibration"],
    )
    return paths


def refresh_hashes(paths: dict[str, Path]) -> None:
    artifact = joblib.load(paths["joint_calibration"])
    artifact.fingerprints["minutes_mean"]["content_hash"] = (
        content_hash(paths["minutes_mean"])
    )
    artifact.fingerprints["minutes_distribution"]["content_hash"] = (
        content_hash(paths["minutes_distribution"])
    )
    artifact.fingerprints["points_mean"]["content_hash"] = (
        content_hash(paths["points_mean"])
    )
    save_joint_calibration(artifact, paths["joint_calibration"])


def load_bundle(paths: dict[str, Path]) -> JointPointsArtifactBundle:
    return load_joint_points_bundle(
        minutes_mean_path=paths["minutes_mean"],
        minutes_dist_path=paths["minutes_distribution"],
        points_path=paths["points_mean"],
        joint_calibration_path=paths["joint_calibration"],
    )


def feature_row(**overrides) -> dict:
    row = {
        "player_id": 2544,
        "game_id": "0022400123",
        "season_year": "2024-25",
        "min_mean_10": 32.0,
        "start_rate_10": 0.85,
        "usg_wmean_10": 27.0,
        "pts_per_min_10": 0.55,
        "fga_per_min_10": 0.45,
        "predicted_minutes_oof": 99.0,
        "minutes": 40.0,
        "pts": 30.0,
        "is_starter": 1,
    }
    row.update(overrides)
    return row
