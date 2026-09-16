"""Fingerprint-verified four-artifact bundle for joint points."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np

from src.models.xgboost_models.joint_calibration import (
    HOLDOUT_SEASON,
    JointCalibration,
    content_hash,
    load_joint_calibration,
)
from src.models.xgboost_models.minutes import XGBoostMinutesModel
from src.models.xgboost_models.points import XGBoostPointsModel

KNOWN_SCHEMA_VERSIONS = frozenset({1})
FITTED_THROUGH = "2024-25"
POOL_CENTER_TOLERANCE = 1e-6


class ArtifactIncompatibilityError(ValueError):
    """Fingerprint, schema, or pool failure. Maps to NO_QUOTE."""

    status = "NO_QUOTE"


@dataclass(frozen=True)
class JointPointsArtifactBundle:
    minutes_mean: XGBoostMinutesModel
    minutes_distribution: XGBoostMinutesModel
    points_mean: XGBoostPointsModel
    joint_calibration: JointCalibration
    bundle_hash: str


def load_joint_points_bundle(
    *,
    minutes_mean_path: str | Path,
    minutes_dist_path: str | Path,
    points_path: str | Path,
    joint_calibration_path: str | Path,
) -> JointPointsArtifactBundle:
    minutes_mean_path = Path(minutes_mean_path)
    minutes_dist_path = Path(minutes_dist_path)
    points_path = Path(points_path)
    joint_calibration_path = Path(joint_calibration_path)
    try:
        calibration = load_joint_calibration(
            joint_calibration_path,
            minutes_mean_path=minutes_mean_path,
            minutes_dist_path=minutes_dist_path,
            points_path=points_path,
        )
    except ValueError as exc:
        raise ArtifactIncompatibilityError(str(exc)) from exc

    minutes_mean = _load_minutes_model(minutes_mean_path)
    minutes_distribution = _load_minutes_model(minutes_dist_path)
    points_mean = _load_points_model(points_path)

    _validate_metadata(calibration)
    _validate_features(
        minutes_mean,
        minutes_distribution,
        points_mean,
        calibration,
    )
    _validate_minutes_bins(minutes_distribution, calibration)
    _validate_residual_pools(
        minutes_distribution.regressor.residual_pools,
        minutes_distribution.regressor.residual_pool_bins,
        name="minutes",
        require_centered=False,
    )
    if calibration.epsilon is None:
        raise ArtifactIncompatibilityError(
            "joint calibration is missing epsilon pools"
        )
    _validate_residual_pools(
        calibration.epsilon.pools,
        calibration.epsilon.bins,
        name="epsilon",
        require_centered=True,
    )
    if not calibration.epsilon.centered:
        raise ArtifactIncompatibilityError(
            "epsilon pools are not marked centered"
        )

    bundle_hash = _bundle_hash(
        minutes_mean_path,
        minutes_dist_path,
        points_path,
        joint_calibration_path,
    )
    return JointPointsArtifactBundle(
        minutes_mean=minutes_mean,
        minutes_distribution=minutes_distribution,
        points_mean=points_mean,
        joint_calibration=calibration,
        bundle_hash=bundle_hash,
    )


def require_pool(
    pools: dict[int, np.ndarray] | None,
    bin_id: int,
    *,
    name: str,
    require_centered: bool | None = None,
) -> np.ndarray:
    """Return the exact bin pool or abort. No neighbor fallback."""
    if pools is None or int(bin_id) not in pools:
        raise ArtifactIncompatibilityError(
            f"missing {name} residual pool {int(bin_id)}"
        )
    pool = np.asarray(pools[int(bin_id)], dtype=float)
    centered = (
        name == "epsilon" if require_centered is None else require_centered
    )
    _assert_valid_pool(
        pool,
        name=name,
        bin_id=int(bin_id),
        require_centered=centered,
    )
    return pool


def _load_minutes_model(path: Path) -> XGBoostMinutesModel:
    try:
        return XGBoostMinutesModel.load(path)
    except (TypeError, ValueError) as exc:
        raise ArtifactIncompatibilityError(
            f"incompatible minutes artifact at {path}"
        ) from exc


def _load_points_model(path: Path) -> XGBoostPointsModel:
    loaded = joblib.load(path)
    if isinstance(loaded, dict):
        loaded = loaded.get("model")
    if isinstance(loaded, XGBoostPointsModel):
        return loaded
    raise ArtifactIncompatibilityError(
        f"incompatible points artifact at {path}"
    )


def _validate_metadata(calibration: JointCalibration) -> None:
    if calibration.schema_version not in KNOWN_SCHEMA_VERSIONS:
        raise ArtifactIncompatibilityError(
            f"unsupported schema_version {calibration.schema_version}"
        )
    if calibration.holdout_season != HOLDOUT_SEASON:
        raise ArtifactIncompatibilityError(
            "holdout_season metadata is incompatible"
        )
    if calibration.fitted_through != FITTED_THROUGH:
        raise ArtifactIncompatibilityError(
            "fitted_through metadata is incompatible"
        )
    if calibration.overlay is None:
        raise ArtifactIncompatibilityError("missing overlay")
    if calibration.beta is None:
        raise ArtifactIncompatibilityError("missing beta map")
    if calibration.epsilon is None:
        raise ArtifactIncompatibilityError("missing epsilon pools")


def _validate_features(
    minutes_mean: XGBoostMinutesModel,
    minutes_distribution: XGBoostMinutesModel,
    points_mean: XGBoostPointsModel,
    calibration: JointCalibration,
) -> None:
    minutes_mean_features = list(minutes_mean.feature_columns)
    minutes_dist_features = list(minutes_distribution.feature_columns)
    if minutes_mean_features != minutes_dist_features:
        raise ArtifactIncompatibilityError(
            "minutes mean and distribution feature names or order differ"
        )
    _assert_fingerprint_features(
        calibration,
        "minutes_mean",
        minutes_mean_features,
        minutes_mean,
    )
    _assert_fingerprint_features(
        calibration,
        "minutes_distribution",
        minutes_dist_features,
        minutes_distribution,
    )
    _assert_fingerprint_features(
        calibration,
        "points_mean",
        list(points_mean.feature_columns),
        points_mean,
    )


def _assert_fingerprint_features(
    calibration: JointCalibration,
    key: str,
    observed: list[str],
    model: object,
) -> None:
    stored = calibration.fingerprints.get(key)
    if not isinstance(stored, dict):
        return
    expected = stored.get("feature_columns")
    if expected is not None and list(expected) != list(observed):
        raise ArtifactIncompatibilityError(
            f"{key} feature names or order do not match the fingerprint"
        )
    expected_trees = stored.get("n_estimators")
    observed_trees = _n_estimators(model)
    if (
        expected_trees is not None
        and observed_trees is not None
        and int(expected_trees) != int(observed_trees)
    ):
        raise ArtifactIncompatibilityError(
            f"{key} n_estimators metadata is incompatible"
        )
    expected_cutoff = stored.get("training_cutoff")
    observed_cutoff = _training_cutoff(model)
    if (
        expected_cutoff is not None
        and observed_cutoff is not None
        and str(expected_cutoff) != str(observed_cutoff)
    ):
        raise ArtifactIncompatibilityError(
            f"{key} training_cutoff metadata is incompatible"
        )


def _validate_minutes_bins(
    minutes_distribution: XGBoostMinutesModel,
    calibration: JointCalibration,
) -> None:
    dist_bins = minutes_distribution.regressor.residual_pool_bins
    if dist_bins is None or calibration.minutes_bins is None:
        raise ArtifactIncompatibilityError(
            "minutes bin edges are missing"
        )
    if not np.array_equal(
        np.asarray(dist_bins, dtype=float),
        np.asarray(calibration.minutes_bins, dtype=float),
    ):
        raise ArtifactIncompatibilityError(
            "minutes bin edges do not match the joint calibration"
        )
    if calibration.beta is not None and not np.array_equal(
        np.asarray(calibration.beta.bins, dtype=float),
        np.asarray(calibration.minutes_bins, dtype=float),
    ):
        raise ArtifactIncompatibilityError(
            "beta bin edges do not match minutes bins"
        )


def _validate_residual_pools(
    pools: dict[int, np.ndarray] | None,
    bins: np.ndarray | None,
    *,
    name: str,
    require_centered: bool = False,
) -> None:
    if bins is None:
        raise ArtifactIncompatibilityError(f"missing {name} bin edges")
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ArtifactIncompatibilityError(
            f"{name} bin edges are invalid"
        )
    for bin_id in range(len(edges) - 1):
        require_pool(
            pools,
            bin_id,
            name=name,
            require_centered=require_centered,
        )


def _assert_valid_pool(
    pool: np.ndarray,
    *,
    name: str,
    bin_id: int,
    require_centered: bool = False,
) -> None:
    if pool.size == 0:
        raise ArtifactIncompatibilityError(
            f"empty {name} residual pool {bin_id}"
        )
    if not np.isfinite(pool).all():
        raise ArtifactIncompatibilityError(
            f"nonfinite {name} residual pool {bin_id}"
        )
    if (
        require_centered
        and abs(float(np.mean(pool))) > POOL_CENTER_TOLERANCE
    ):
        raise ArtifactIncompatibilityError(
            f"uncentered {name} residual pool {bin_id}"
        )


def _n_estimators(model: object) -> int | None:
    regressor = getattr(model, "regressor", None)
    booster = getattr(regressor, "model", None)
    value = getattr(booster, "n_estimators", None)
    if value is None:
        return None
    return int(value)


def _training_cutoff(model: object) -> str | None:
    regressor = getattr(model, "regressor", None)
    cutoff = getattr(regressor, "training_cutoff", None)
    if cutoff is None:
        return None
    return str(cutoff)


def _bundle_hash(*paths: Path) -> str:
    material = "|".join(content_hash(path) for path in paths)
    return sha256(material.encode("utf-8")).hexdigest()
