"""Joint calibration trainer: nested OOF coupling artifact."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.joint_calibration import (
    CURRENT_OVERLAY_FEATURES,
    HOLDOUT_SEASON,
    JOINT_VARIANTS,
    PREGAME_OVERLAY_FEATURES,
    JointCalibration,
    apply_overlay,
    assert_preholdout,
    build_epsilon_pools,
    classify_universes,
    content_hash,
    distribution_gate,
    evaluate_distribution_gate,
    evaluate_overlay_gate,
    fit_beta,
    fit_joint_variant,
    fit_overlay,
    load_joint_calibration,
    overlay_gate,
    run_nested_coupling,
    save_joint_calibration,
    select_joint_variant,
)
from src.models.xgboost_models.points import add_predicted_points_oof


class PreholdoutGuardTests(unittest.TestCase):
    def test_assert_preholdout_rejects_holdout_season(self) -> None:
        frame = pd.DataFrame({"season_year": ["2024-25", HOLDOUT_SEASON]})
        with self.assertRaises(ValueError):
            assert_preholdout(frame)

    def test_assert_preholdout_accepts_preholdout_only(self) -> None:
        frame = pd.DataFrame({"season_year": ["2023-24", "2024-25"]})
        assert_preholdout(frame)


class FingerprintTests(unittest.TestCase):
    def test_content_hash_changes_when_bytes_change(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.joblib"
            path.write_bytes(b"abc")
            first = content_hash(path)
            path.write_bytes(b"abd")
            self.assertNotEqual(first, content_hash(path))


class UniverseTests(unittest.TestCase):
    def test_first_base_fold_is_coupling_warmup(self) -> None:
        frame = pd.DataFrame(
            {
                "minutes_hat": [np.nan, 20.0, 22.0],
                "points_hat": [np.nan, 15.0, 16.0],
                "g_hat": [np.nan, np.nan, 0.5],
                "beta_hat": [np.nan, np.nan, 0.4],
                "base_fold": [0, 1, 2],
            }
        )
        labeled = classify_universes(frame)
        self.assertFalse(labeled.loc[0, "base_oof_eligible"])
        self.assertTrue(labeled.loc[1, "base_oof_eligible"])
        self.assertFalse(labeled.loc[1, "coupling_oof_eligible"])
        self.assertEqual(labeled.loc[1, "exclusion"], "coupling_warmup")
        self.assertTrue(labeled.loc[2, "coupling_oof_eligible"])
        self.assertEqual(labeled.loc[0, "exclusion"], "warmup")


class OverlayFitTests(unittest.TestCase):
    def test_overlay_minutes_shock_coefficient_is_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        shock = rng.normal(0, 4, n)
        frame = pd.DataFrame(
            {
                "minutes_hat": 28.0 + shock,
                "min_mean_10": np.full(n, 28.0),
                "pts_per_min_10": rng.uniform(0.4, 0.7, n),
                "usg_wmean_10": rng.uniform(15, 30, n),
                "start_rate_10": rng.uniform(0, 1, n),
            }
        )
        residual = 0.8 * shock + rng.normal(0, 0.3, n)
        fit = fit_overlay(frame, residual)
        self.assertGreaterEqual(fit.coef[0], 0.0)
        pred = apply_overlay(fit, frame)
        self.assertTrue(np.isfinite(pred).all())

    def test_overlay_may_be_negative(self) -> None:
        frame = pd.DataFrame(
            {
                "minutes_hat": [20.0, 20.0, 20.0, 10.0],
                "min_mean_10": [20.0, 20.0, 20.0, 20.0],
                "pts_per_min_10": [0.5, 0.5, 0.5, 0.5],
                "usg_wmean_10": [20.0, 20.0, 20.0, 20.0],
                "start_rate_10": [0.8, 0.8, 0.8, 0.8],
            }
        )
        residual = np.array([1.0, 1.0, 1.0, -6.0])
        pred = apply_overlay(fit_overlay(frame, residual), frame)
        self.assertLess(pred[-1], 0.0)


class BetaFitTests(unittest.TestCase):
    def test_beta_recovers_slope_and_clips_nonnegative(self) -> None:
        u = np.array([-4.0, -2.0, 0.0, 2.0, 4.0] * 20)
        residual = 0.5 * u
        minutes_hat = np.where(u >= 0, 30.0, 10.0)
        bins = np.array([0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0])
        fitted = fit_beta(u, residual, minutes_hat, bins)
        self.assertGreater(fitted.beta_global, 0.3)
        self.assertLess(fitted.beta_global, 0.7)
        for value in fitted.beta_by_bin.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 2.5)


class NestedCouplingTests(unittest.TestCase):
    def test_nested_coupling_leaves_first_hat_fold_without_g(self) -> None:
        frame = pd.DataFrame(
            {
                "base_fold": [1] * 80 + [2] * 80,
                "minutes_hat": np.r_[np.full(80, 18.0), np.full(80, 28.0)],
                "points_hat": np.r_[np.full(80, 10.0), np.full(80, 18.0)],
                "minutes": np.r_[np.full(80, 18.0), np.full(80, 32.0)],
                "pts": np.r_[np.full(80, 10.0), np.full(80, 24.0)],
                "min_mean_10": 20.0,
                "pts_per_min_10": 0.5,
                "usg_wmean_10": 22.0,
                "start_rate_10": 0.6,
            }
        )
        out = run_nested_coupling(frame)
        self.assertTrue(out.loc[out["base_fold"].eq(1), "g_hat"].isna().all())
        self.assertTrue(out.loc[out["base_fold"].eq(2), "g_hat"].notna().all())
        labeled = classify_universes(out)
        self.assertEqual(int(labeled["coupling_oof_eligible"].sum()), 80)


class EpsilonPoolTests(unittest.TestCase):
    def test_epsilon_pools_ignore_coupling_warmup_and_are_centered(self) -> None:
        frame = pd.DataFrame(
            {
                "base_fold": [1] * 50 + [2] * 80,
                "minutes_hat": 24.0,
                "points_hat": 12.0,
                "g_hat": np.r_[np.full(50, np.nan), np.full(80, 0.0)],
                "beta_hat": np.r_[np.full(50, np.nan), np.full(80, 0.4)],
                "minutes": 24.0,
                "pts": np.r_[np.full(50, 30.0), np.full(80, 12.0)],
            }
        )
        pools = build_epsilon_pools(classify_universes(frame))
        self.assertTrue(
            all(abs(pool.mean()) < 1e-8 for pool in pools.pools.values())
        )
        stacked = np.concatenate(list(pools.pools.values()))
        self.assertLess(np.abs(stacked).max(), 1.0)


class PointsOofTests(unittest.TestCase):
    def test_points_oof_is_not_actual_points(self) -> None:
        frame = pd.DataFrame(
            {
                "game_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "pts": [10.0, 20.0, 30.0, 40.0],
                "predicted_minutes_oof": [20.0, 20.0, 20.0, 20.0],
            }
        )

        class Spy:
            def fit(self, train, **kwargs):
                return self

            def predict_mean(self, rows):
                return np.full(len(rows), 15.0)

        splits = [(frame.index[:2], frame.index[2:])]
        out = add_predicted_points_oof(
            frame,
            points_model_factory=lambda: Spy(),
            splits=splits,
        )
        later = out.loc[out.index[2:], "predicted_points_oof"]
        self.assertTrue((later == 15.0).all())
        self.assertFalse(
            (later.values == out.loc[out.index[2:], "pts"].values).any()
        )


class SaveLoadTests(unittest.TestCase):
    def test_load_fails_closed_on_hash_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            minutes = Path(tmp) / "m.joblib"
            dist = Path(tmp) / "d.joblib"
            points = Path(tmp) / "p.joblib"
            for path in (minutes, dist, points):
                path.write_bytes(b"ok")
            artifact = JointCalibration(
                schema_version=1,
                holdout_season=HOLDOUT_SEASON,
                fitted_through="2024-25",
                fingerprints={
                    "minutes_mean": content_hash(minutes),
                    "minutes_distribution": content_hash(dist),
                    "points_mean": content_hash(points),
                },
            )
            out = Path(tmp) / "joint.joblib"
            save_joint_calibration(artifact, out)
            points.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_joint_calibration(
                    out,
                    minutes_mean_path=minutes,
                    minutes_dist_path=dist,
                    points_path=points,
                )

    def test_load_accepts_dict_fingerprints(self) -> None:
        with TemporaryDirectory() as tmp:
            minutes = Path(tmp) / "m.joblib"
            dist = Path(tmp) / "d.joblib"
            points = Path(tmp) / "p.joblib"
            for path in (minutes, dist, points):
                path.write_bytes(b"ok")
            artifact = JointCalibration(
                fingerprints={
                    "minutes_mean": {
                        "content_hash": content_hash(minutes),
                        "n_estimators": 750,
                    },
                    "minutes_distribution": {
                        "content_hash": content_hash(dist),
                    },
                    "points_mean": {
                        "content_hash": content_hash(points),
                    },
                }
            )
            out = Path(tmp) / "joint.joblib"
            save_joint_calibration(artifact, out)
            loaded = load_joint_calibration(
                out,
                minutes_mean_path=minutes,
                minutes_dist_path=dist,
                points_path=points,
            )
            self.assertEqual(loaded.fingerprints["minutes_mean"]["n_estimators"], 750)
            points.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_joint_calibration(
                    out,
                    minutes_mean_path=minutes,
                    minutes_dist_path=dist,
                    points_path=points,
                )


class OverlayGateTests(unittest.TestCase):
    def test_gate_allows_small_mae_wiggle(self) -> None:
        self.assertTrue(
            overlay_gate(
                {"mae": 4.52, "bias": 0.01, "n": 500},
                {"mae": 4.50, "bias": 0.01, "n": 500},
            )
        )

    def test_gate_skips_tiny_slices(self) -> None:
        self.assertIsNone(
            overlay_gate(
                {"mae": 9.0, "bias": 3.0, "n": 20},
                {"mae": 4.0, "bias": 0.0, "n": 20},
            )
        )
        self.assertTrue(
            evaluate_overlay_gate(
                [
                    (
                        {"mae": 9.0, "bias": 3.0, "n": 20},
                        {"mae": 4.0, "bias": 0.0, "n": 20},
                    )
                ]
            )
        )


def _gate_metrics(**overrides) -> dict:
    payload = {
        "nll": 1.0,
        "coverage_80": 0.80,
        "pit_mean": 0.50,
        "width_80": 10.0,
    }
    payload.update(overrides)
    return payload


def _variant_panel(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    folds = np.repeat([1, 2, 3], n // 3)
    n = len(folds)
    minutes_hat = rng.uniform(10.0, 34.0, n)
    u = rng.normal(0.0, 3.0, n)
    start_rate = np.r_[np.full(n // 2, 0.8), np.full(n - n // 2, 0.2)]
    return pd.DataFrame(
        {
            "base_fold": folds,
            "minutes_hat": minutes_hat,
            "points_hat": 0.5 * minutes_hat,
            "minutes": minutes_hat + u,
            "pts": 0.5 * minutes_hat + 0.4 * u + rng.normal(0, 0.4, n),
            "min_mean_10": minutes_hat - rng.normal(0, 1.0, n),
            "pts_per_min_10": rng.uniform(0.4, 0.7, n),
            "usg_wmean_10": rng.uniform(15, 30, n),
            "start_rate_10": start_rate,
        }
    )


class PregameOverlayTests(unittest.TestCase):
    def test_pregame_overlay_skips_minutes_columns(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        frame = pd.DataFrame(
            {
                "minutes_hat": rng.uniform(10, 36, n),
                "minutes": rng.uniform(8, 40, n),
                "min_mean_10": np.full(n, 24.0),
                "pts_per_min_10": rng.uniform(0.3, 0.8, n),
                "usg_wmean_10": rng.uniform(12, 32, n),
                "start_rate_10": rng.uniform(0, 1, n),
            }
        )
        residual = (
            3.0 * frame["pts_per_min_10"].to_numpy()
            + rng.normal(0, 0.2, n)
        )
        fit = fit_overlay(
            frame,
            residual,
            feature_names=PREGAME_OVERLAY_FEATURES,
        )
        self.assertEqual(fit.feature_names, PREGAME_OVERLAY_FEATURES)
        self.assertEqual(len(fit.coef), 3)
        self.assertNotIn("minutes_shock", fit.feature_names)
        pred = apply_overlay(fit, frame)
        swapped = frame.copy()
        swapped["minutes_hat"] = 3.0
        swapped["minutes"] = 41.0
        swapped["min_mean_10"] = 50.0
        np.testing.assert_allclose(
            pred, apply_overlay(fit, swapped), atol=1e-12
        )

    def test_current_overlay_default_keeps_minutes_shock(self) -> None:
        self.assertEqual(
            CURRENT_OVERLAY_FEATURES[0],
            "minutes_shock",
        )


class BetaModeTests(unittest.TestCase):
    def test_beta_recovers_realized_minute_shock(self) -> None:
        minutes_hat = np.full(200, 24.0)
        u = np.linspace(-8.0, 8.0, 200)
        residual = 0.65 * u
        fitted = fit_beta(u, residual, minutes_hat)
        self.assertAlmostEqual(fitted.beta_global, 0.65, places=2)

    def test_beta_modes_differ_on_slope_by_bin(self) -> None:
        n = 80
        u_low = np.linspace(-4.0, 4.0, n)
        u_high = np.linspace(-4.0, 4.0, n)
        residual = np.r_[0.2 * u_low, 1.8 * u_high]
        u = np.r_[u_low, u_high]
        minutes_hat = np.r_[np.full(n, 8.0), np.full(n, 32.0)]
        bins = np.array([0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0])
        global_fit = fit_beta(
            u, residual, minutes_hat, bins, mode="global"
        )
        binwise = fit_beta(
            u, residual, minutes_hat, bins, mode="binwise"
        )
        shrunk = fit_beta(
            u, residual, minutes_hat, bins, mode="shrunk"
        )
        self.assertTrue(
            all(
                value == global_fit.beta_global
                for value in global_fit.beta_by_bin.values()
            )
        )
        self.assertAlmostEqual(binwise.beta_by_bin[0], 0.2, places=2)
        self.assertAlmostEqual(binwise.beta_by_bin[4], 1.8, places=2)
        self.assertGreater(shrunk.beta_by_bin[0], binwise.beta_by_bin[0])
        self.assertLess(shrunk.beta_by_bin[4], binwise.beta_by_bin[4])
        self.assertNotAlmostEqual(
            shrunk.beta_by_bin[0],
            shrunk.beta_by_bin[4],
            places=2,
        )


class EpsilonConstructionTests(unittest.TestCase):
    def test_current_epsilon_is_centered_coupling_residual(self) -> None:
        n_warm, n_fit = 50, 80
        u = np.full(n_fit, 2.0)
        frame = pd.DataFrame(
            {
                "base_fold": [1] * n_warm + [2] * n_fit,
                "minutes_hat": 24.0,
                "points_hat": 12.0,
                "g_hat": np.r_[np.full(n_warm, np.nan), np.full(n_fit, 1.0)],
                "beta_hat": np.r_[np.full(n_warm, np.nan), np.full(n_fit, 0.5)],
                "minutes": np.r_[np.full(n_warm, 24.0), 24.0 + u],
                "pts": np.r_[
                    np.full(n_warm, 30.0),
                    12.0 + 1.0 + 0.5 * u + 0.4,
                ],
                "start_rate_10": 0.8,
            }
        )
        labeled = classify_universes(frame)
        pools = build_epsilon_pools(labeled)
        self.assertTrue(pools.centered)
        self.assertNotEqual(pools.epsilon_source, "standalone_points")
        self.assertTrue(
            all(abs(pool.mean()) < 1e-8 for pool in pools.pools.values())
        )

    def test_g_off_epsilon_may_keep_shrunk_location(self) -> None:
        n_warm, n_fit = 50, 80
        frame = pd.DataFrame(
            {
                "base_fold": [1] * n_warm + [2] * n_fit,
                "minutes_hat": 24.0,
                "points_hat": 12.0,
                "g_hat": np.r_[np.full(n_warm, np.nan), np.full(n_fit, 0.0)],
                "beta_hat": np.r_[np.full(n_warm, np.nan), np.full(n_fit, 0.0)],
                "minutes": 24.0,
                "pts": np.r_[np.full(n_warm, 30.0), np.full(n_fit, 16.0)],
                "start_rate_10": 0.2,
            }
        )
        pools = build_epsilon_pools(
            classify_universes(frame),
            overlay_enabled=False,
        )
        stacked = np.concatenate(list(pools.pools.values()))
        self.assertFalse(pools.centered)
        self.assertGreater(float(stacked.mean()), 1.0)
        self.assertLess(float(stacked.mean()), 4.0)

    def test_epsilon_ignores_standalone_points_residuals(self) -> None:
        n_warm, n_fit = 50, 80
        frame = pd.DataFrame(
            {
                "base_fold": [1] * n_warm + [2] * n_fit,
                "minutes_hat": 24.0,
                "points_hat": 12.0,
                "g_hat": np.r_[np.full(n_warm, np.nan), np.full(n_fit, 0.0)],
                "beta_hat": np.r_[np.full(n_warm, np.nan), np.full(n_fit, 0.0)],
                "minutes": 24.0,
                "pts": np.r_[np.full(n_warm, 30.0), np.full(n_fit, 12.0)],
                "calibration_residuals": 999.0,
                "standalone_points_residual": -500.0,
            }
        )
        pools = build_epsilon_pools(classify_universes(frame))
        stacked = np.concatenate(list(pools.pools.values()))
        self.assertLess(np.abs(stacked).max(), 1.0)

    def test_epsilon_rejects_standalone_source(self) -> None:
        frame = pd.DataFrame(
            {
                "base_fold": [2] * 80,
                "minutes_hat": 24.0,
                "points_hat": 12.0,
                "g_hat": 0.0,
                "beta_hat": 0.0,
                "minutes": 24.0,
                "pts": 12.0,
            }
        )
        with self.assertRaises(ValueError):
            build_epsilon_pools(
                classify_universes(frame),
                epsilon_source="standalone_points",
            )


class VariantMatrixTests(unittest.TestCase):
    def test_residual_around_p_has_zero_g_and_beta(self) -> None:
        overlay, beta, pools, meta = fit_joint_variant(
            _variant_panel(),
            "residual_around_p",
        )
        self.assertIsNone(overlay)
        self.assertIsNone(beta)
        self.assertEqual(meta["variant"], "residual_around_p")
        self.assertIn("residual_around_p", JOINT_VARIANTS)

    def test_current_variant_keeps_minutes_shock_overlay(self) -> None:
        overlay, beta, pools, meta = fit_joint_variant(
            _variant_panel(),
            "current",
        )
        self.assertIsNotNone(overlay)
        self.assertIsNotNone(beta)
        self.assertEqual(
            overlay.feature_names,
            CURRENT_OVERLAY_FEATURES,
        )
        self.assertTrue(pools.centered)

    def test_g_on_shrunk_role_uses_pregame_overlay(self) -> None:
        overlay, beta, pools, meta = fit_joint_variant(
            _variant_panel(),
            "g_on_shrunk_role",
        )
        self.assertEqual(
            overlay.feature_names,
            PREGAME_OVERLAY_FEATURES,
        )
        self.assertTrue(pools.centered)
        self.assertTrue(pools.role_aware)
        self.assertIsNotNone(pools.role_pools)


class DistributionGateTests(unittest.TestCase):
    def test_nll_improvement_is_required(self) -> None:
        self.assertFalse(
            distribution_gate(_gate_metrics(nll=1.0), _gate_metrics())
        )
        self.assertTrue(
            distribution_gate(_gate_metrics(nll=0.9), _gate_metrics())
        )

    def test_mae_only_improvement_does_not_promote(self) -> None:
        candidate = _gate_metrics(nll=1.1, mae=3.0)
        baseline = _gate_metrics(nll=1.0, mae=4.0)
        self.assertFalse(distribution_gate(candidate, baseline))

    def test_coverage_stays_in_band(self) -> None:
        self.assertFalse(
            distribution_gate(
                _gate_metrics(nll=0.9, coverage_80=0.70),
                _gate_metrics(),
            )
        )
        self.assertTrue(
            distribution_gate(
                _gate_metrics(nll=0.9, coverage_80=0.80),
                _gate_metrics(),
            )
        )

    def test_width_does_not_explode(self) -> None:
        self.assertFalse(
            distribution_gate(
                _gate_metrics(nll=0.9, width_80=20.0),
                _gate_metrics(width_80=10.0),
            )
        )

    def test_role_gap_must_shrink_when_provided(self) -> None:
        baseline = _gate_metrics(
            starter_coverage_80=0.90,
            bench_coverage_80=0.60,
        )
        better = _gate_metrics(
            nll=0.9,
            starter_coverage_80=0.81,
            bench_coverage_80=0.79,
        )
        worse = _gate_metrics(
            nll=0.9,
            starter_coverage_80=0.95,
            bench_coverage_80=0.50,
        )
        self.assertTrue(distribution_gate(better, baseline))
        self.assertFalse(distribution_gate(worse, baseline))

    def test_select_joint_variant_skips_failing_candidates(self) -> None:
        residual = [_gate_metrics()]
        failing = [_gate_metrics(nll=1.4)]
        chosen = select_joint_variant(
            {
                "residual_around_p": residual,
                "g_off_global": failing,
                "g_off_shrunk_role": failing,
                "g_on_shrunk_role": failing,
            }
        )
        self.assertEqual(chosen, "current")

    def test_evaluate_distribution_gate_majority(self) -> None:
        pairs = [
            (_gate_metrics(nll=0.9), _gate_metrics()),
            (_gate_metrics(nll=0.9), _gate_metrics()),
            (_gate_metrics(nll=1.015), _gate_metrics()),
        ]
        self.assertTrue(evaluate_distribution_gate(pairs))
        veto = [
            (_gate_metrics(nll=0.9), _gate_metrics()),
            (_gate_metrics(nll=1.05), _gate_metrics()),
        ]
        self.assertFalse(evaluate_distribution_gate(veto))


if __name__ == "__main__":
    unittest.main()
