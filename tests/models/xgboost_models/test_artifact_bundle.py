"""Fingerprint-verified joint points artifact bundle."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np

from src.models.xgboost_models.artifact_bundle import (
    ArtifactIncompatibilityError,
    JointPointsArtifactBundle,
)
from src.models.xgboost_models.joint_calibration import (
    JointCalibration,
    save_joint_calibration,
)
from src.models.xgboost_models.minutes import XGBoostMinutesModel
from src.models.xgboost_models.points import XGBoostPointsModel

from tests.models.xgboost_models.joint_fixtures import (
    centered_pool,
    load_bundle,
    make_minutes_model,
    refresh_hashes,
    write_bundle,
)


class BundleLoadTests(unittest.TestCase):
    def test_loads_exact_four_artifact_bundle(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            bundle = load_bundle(paths)

        self.assertIsInstance(bundle, JointPointsArtifactBundle)
        self.assertIsInstance(
            bundle.minutes_mean,
            XGBoostMinutesModel,
        )
        self.assertIsInstance(
            bundle.minutes_distribution,
            XGBoostMinutesModel,
        )
        self.assertIsInstance(
            bundle.points_mean,
            XGBoostPointsModel,
        )
        self.assertIsInstance(
            bundle.joint_calibration,
            JointCalibration,
        )
        self.assertEqual(len(bundle.bundle_hash), 64)
        self.assertTrue(
            set(bundle.bundle_hash) <= set("0123456789abcdef")
        )


class FingerprintMismatchTests(unittest.TestCase):
    def test_minutes_mean_fingerprint_mismatch_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            paths["minutes_mean"].write_bytes(
                paths["minutes_mean"].read_bytes() + b"x"
            )
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_minutes_distribution_fingerprint_mismatch_aborts(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            paths["minutes_distribution"].write_bytes(
                paths["minutes_distribution"].read_bytes() + b"x"
            )
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_points_mean_fingerprint_mismatch_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            paths["points_mean"].write_bytes(
                paths["points_mean"].read_bytes() + b"x"
            )
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)


class SchemaValidationTests(unittest.TestCase):
    def test_mismatched_feature_names_abort(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model()
            model.feature_columns = [
                "min_mean_10",
                "start_rate_10",
                "other_feature",
            ]
            model.regressor.feature_columns = list(
                model.feature_columns
            )
            model.save(paths["minutes_mean"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_mismatched_feature_order_abort(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model()
            model.feature_columns = [
                "start_rate_10",
                "min_mean_10",
                "usg_wmean_10",
            ]
            model.regressor.feature_columns = list(
                model.feature_columns
            )
            model.save(paths["minutes_mean"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_mismatched_minutes_bin_edges_abort(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model(with_pools=True)
            model.regressor.residual_pool_bins = np.array(
                [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 64.0]
            )
            model.save(paths["minutes_distribution"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_mismatched_artifact_metadata_abort(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            artifact = joblib.load(paths["joint_calibration"])
            artifact.holdout_season = "2024-25"
            save_joint_calibration(
                artifact,
                paths["joint_calibration"],
            )
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)


class ResidualPoolValidationTests(unittest.TestCase):
    def test_missing_residual_pool_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model(with_pools=True)
            del model.regressor.residual_pools[2]
            model.save(paths["minutes_distribution"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_empty_residual_pool_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model(with_pools=True)
            model.regressor.residual_pools[1] = np.array([], dtype=float)
            model.save(paths["minutes_distribution"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_nonfinite_residual_pool_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model(with_pools=True)
            pool = centered_pool()
            pool[0] = np.nan
            model.regressor.residual_pools[0] = pool
            model.save(paths["minutes_distribution"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_uncentered_epsilon_pool_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            artifact = joblib.load(paths["joint_calibration"])
            artifact.epsilon.pools[0] = np.full(40, 3.0)
            save_joint_calibration(
                artifact,
                paths["joint_calibration"],
            )
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError):
                load_bundle(paths)

    def test_uncentered_minutes_pool_is_allowed(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model(with_pools=True)
            model.regressor.residual_pools[0] = np.full(40, 3.0)
            model.save(paths["minutes_distribution"])
            refresh_hashes(paths)
            bundle = load_bundle(paths)
        self.assertIsNotNone(bundle)
    def test_does_not_fall_back_to_neighboring_pool(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = write_bundle(Path(tmp))
            model = make_minutes_model(with_pools=True)
            del model.regressor.residual_pools[4]
            model.save(paths["minutes_distribution"])
            refresh_hashes(paths)
            with self.assertRaises(ArtifactIncompatibilityError) as ctx:
                load_bundle(paths)
            self.assertNotIn("fallback", str(ctx.exception).lower())
