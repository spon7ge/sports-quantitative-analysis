"""Preholdout joint-variant scoring uses earlier-fold coupling only."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.joint_variant_eval import (
    PROMOTION_CANDIDATE,
    _assert_earlier_fold_only,
    clipping_uplift_by_band,
    decide_promotion,
    decompose_location,
    fit_fold_variant,
    fit_preholdout_variant,
    pit_decile_table,
    score_samples,
    score_variant_folds,
    signed_error_by_band,
    simulate_fold,
    simulate_residual_around_p,
)


def _panel(n_per_fold: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2021-01-01")
    for fold in (1, 2, 3):
        for index in range(n_per_fold):
            date = start + pd.Timedelta(days=(fold - 1) * 120 + index)
            shock = rng.normal(0, 3)
            hat_m = 22.0 + 6.0 * (index % 2)
            minutes = hat_m + shock
            hat_p = 0.5 * hat_m
            pts = hat_p + 0.4 * shock + rng.normal(0, 1.5)
            start_rate = 0.8 if index % 2 == 0 else 0.2
            rows.append(
                {
                    "season_year": "2023-24",
                    "game_id": fold * 1000 + index,
                    "player_id": index,
                    "game_date": date,
                    "minutes": minutes,
                    "pts": pts,
                    "minutes_hat": hat_m,
                    "points_hat": hat_p,
                    "base_fold": fold,
                    "start_rate_10": start_rate,
                    "min_mean_10": 20.0,
                    "pts_per_min_10": 0.5,
                    "usg_wmean_10": 22.0,
                }
            )
    return pd.DataFrame(rows)


class ChronologyTests(unittest.TestCase):
    def test_prior_dates_must_precede_valid_dates(self) -> None:
        panel = _panel()
        fit = fit_fold_variant(
            panel, fold=2, variant="g_off_shrunk_role"
        )
        self.assertLess(fit.prior_max_date, fit.valid_min_date)

    def test_leaking_dates_raise(self) -> None:
        prior = pd.DataFrame(
            {"game_date": pd.to_datetime(["2022-01-10", "2022-01-20"])}
        )
        valid = pd.DataFrame(
            {"game_date": pd.to_datetime(["2022-01-15"])}
        )
        with self.assertRaises(ValueError):
            _assert_earlier_fold_only(prior, valid)


class SimulationTests(unittest.TestCase):
    def test_g_off_variant_fits_from_earlier_fold_only(self) -> None:
        panel = _panel()
        valid = panel.loc[panel["base_fold"].eq(2)].copy()
        fit = fit_fold_variant(
            panel, fold=2, variant="g_off_shrunk_role"
        )
        draws = simulate_fold(valid, fit, n_draws=200, seed=42)
        self.assertEqual(draws.shape, (len(valid), 200))
        self.assertTrue(np.all(draws >= 0))
        self.assertIsNone(fit.overlay)
        self.assertIsNotNone(fit.beta)
        self.assertTrue(fit.epsilon.role_aware)

    def test_identical_seeds_are_reproducible(self) -> None:
        panel = _panel()
        valid = panel.loc[panel["base_fold"].eq(3)].copy()
        fit = fit_fold_variant(panel, fold=3, variant="current")
        first = simulate_fold(valid, fit, n_draws=50, seed=42)
        second = simulate_fold(valid, fit, n_draws=50, seed=42)
        np.testing.assert_allclose(first, second)

    def test_simulate_fold_can_return_preclip_draws(self) -> None:
        panel = _panel()
        valid = panel.loc[panel["base_fold"].eq(2)].copy()
        valid["points_hat"] = 1.5
        fit = fit_fold_variant(panel, fold=2, variant="current")
        clipped, raw = simulate_fold(
            valid,
            fit,
            n_draws=400,
            seed=7,
            return_raw=True,
        )
        self.assertEqual(clipped.shape, raw.shape)
        self.assertTrue(np.all(clipped >= 0.0))
        self.assertLess(float(raw.min()), 0.0)
        np.testing.assert_allclose(clipped, np.maximum(0.0, raw))
        self.assertGreater(
            float(clipped.mean()),
            float(raw.mean()),
        )


class BundleFitTests(unittest.TestCase):
    def test_fold_fit_from_bundle_simulates(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.models.xgboost_models.joint_variant_eval import (
            fold_fit_from_bundle,
        )
        from tests.models.xgboost_models.joint_fixtures import (
            feature_row,
            load_bundle,
            write_bundle,
        )

        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            fit = fold_fit_from_bundle(bundle)
            valid = pd.DataFrame([feature_row(), feature_row()])
            valid["minutes_hat"] = 24.0
            valid["points_hat"] = 18.0
            clipped, raw = simulate_fold(
                valid,
                fit,
                n_draws=32,
                seed=3,
                return_raw=True,
            )
        self.assertEqual(clipped.shape, (2, 32))
        self.assertTrue(np.all(clipped >= 0.0))
        self.assertEqual(fit.variant, "current")
        self.assertIsNotNone(fit.epsilon)


class LocationDiagnosticTests(unittest.TestCase):
    def test_decompose_location_splits_mean_sources(self) -> None:
        actual = np.array([4.0, 6.0, 8.0])
        hat_p = np.array([5.0, 7.0, 9.0])
        g = np.array([0.2, 0.0, -0.1])
        mean_pre = np.array([5.1, 6.8, 8.7])
        mean_post = np.array([5.4, 6.9, 8.8])
        parts = decompose_location(
            actual,
            hat_p,
            g,
            mean_pre,
            mean_post,
        )
        self.assertAlmostEqual(
            parts["error_vs_hat_p"],
            float(np.mean(actual - hat_p)),
        )
        self.assertAlmostEqual(
            parts["overlay_shift"],
            float(np.mean(g)),
        )
        self.assertAlmostEqual(
            parts["residual_location"],
            float(np.mean(mean_pre - (hat_p + g))),
        )
        self.assertAlmostEqual(
            parts["clipping_uplift"],
            float(np.mean(mean_post - mean_pre)),
        )
        self.assertAlmostEqual(
            parts["error_vs_post_clip"],
            parts["error_vs_hat_p"]
            - parts["overlay_shift"]
            - parts["residual_location"]
            - parts["clipping_uplift"],
        )

    def test_clipping_uplift_by_band_concentrates_at_low_hat_p(self) -> None:
        hat_p = np.array([3.0, 4.0, 12.0, 22.0])
        mean_pre = np.array([2.0, 3.0, 12.0, 22.0])
        mean_post = np.array([3.0, 4.0, 12.05, 22.0])
        table = clipping_uplift_by_band(
            mean_pre,
            mean_post,
            hat_p,
            edges=np.array([0.0, 8.0, 14.0, 20.0, 28.0, np.inf]),
        )
        low = table.loc[table["band"].eq("[0, 8)")].iloc[0]
        mid = table.loc[table["band"].eq("[8, 14)")].iloc[0]
        self.assertAlmostEqual(low["clipping_uplift"], 1.0)
        self.assertAlmostEqual(mid["clipping_uplift"], 0.05)

    def test_shift_raw_matches_target_postclip_mean(self) -> None:
        from src.models.xgboost_models.joint_variant_eval import (
            shift_raw_to_postclip_mean,
        )

        raw = np.array(
            [
                [-4.0, -1.0, 0.5, 2.0, 5.0],
                [8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        )
        target = np.array([1.5, 10.0])
        shifted = shift_raw_to_postclip_mean(raw, target)
        post = np.maximum(0.0, shifted).mean(axis=1)
        np.testing.assert_allclose(post, target, atol=1e-6)


class ScoringTests(unittest.TestCase):
    def test_score_variant_folds_skips_warmup_fold(self) -> None:
        panel = _panel()
        metrics = score_variant_folds(
            panel,
            variants=("residual_around_p", "current"),
            n_draws=40,
            seed=42,
        )
        folds = [row["fold"] for row in metrics["current"]]
        self.assertEqual(folds, [2, 3])
        self.assertNotIn(1, folds)

    def test_score_samples_has_no_mae_key(self) -> None:
        samples = np.array([[8.0, 10.0, 12.0, 9.0]])
        metrics = score_samples(
            samples,
            np.array([10.0]),
            np.array([0.9]),
        )
        self.assertNotIn("mae", metrics)
        self.assertIn("nll", metrics)
        self.assertIn("role_gap", metrics)


def _fold(nll, coverage, pit, width, starter, bench, shape=0.05):
    return {
        "n": 1000,
        "nll": nll,
        "coverage_80": coverage,
        "pit_mean": pit,
        "width_80": width,
        "starter_coverage_80": starter,
        "bench_coverage_80": bench,
        "role_gap": abs(starter - bench),
        "pit_shape": shape,
        "pit_hist": [100] * 10,
        "starter_width_80": width,
        "bench_width_80": width,
        "fold": 2,
    }


class PromotionTests(unittest.TestCase):
    def test_failed_nll_retains_current(self) -> None:
        current = _fold(3.10, 0.80, 0.50, 12.0, 0.73, 0.86)
        residual = _fold(3.20, 0.79, 0.50, 11.5, 0.72, 0.85)
        candidate = _fold(3.15, 0.80, 0.50, 12.0, 0.76, 0.82)
        decision = decide_promotion(
            {
                "current": [current, current],
                "residual_around_p": [residual, residual],
                PROMOTION_CANDIDATE: [candidate, candidate],
            }
        )
        self.assertEqual(decision.selected, "current")
        self.assertFalse(decision.passed)

    def test_passing_gates_promotes_candidate(self) -> None:
        current = _fold(3.20, 0.80, 0.50, 12.0, 0.73, 0.86)
        residual = _fold(3.25, 0.79, 0.50, 11.8, 0.72, 0.85)
        candidate = _fold(3.10, 0.80, 0.50, 12.1, 0.77, 0.82, shape=0.04)
        decision = decide_promotion(
            {
                "current": [current, current],
                "residual_around_p": [residual, residual],
                PROMOTION_CANDIDATE: [candidate, candidate],
            }
        )
        self.assertTrue(decision.passed)
        self.assertEqual(decision.selected, PROMOTION_CANDIDATE)

    def test_width_increase_beyond_two_percent_retains_current(self) -> None:
        current = _fold(3.20, 0.80, 0.50, 10.0, 0.79, 0.81)
        residual = _fold(3.25, 0.79, 0.50, 10.0, 0.79, 0.81)
        candidate = _fold(3.10, 0.80, 0.50, 10.5, 0.80, 0.81)
        decision = decide_promotion(
            {
                "current": [current, current],
                "residual_around_p": [residual, residual],
                PROMOTION_CANDIDATE: [candidate, candidate],
            }
        )
        self.assertEqual(decision.selected, "current")
        self.assertFalse(decision.passed)


class HoldoutDiagnosticTests(unittest.TestCase):
    def test_signed_error_by_band_groups_predicted_points(self) -> None:
        table = signed_error_by_band(
            actual=np.array([4.0, 6.0, 16.0, 18.0]),
            predicted=np.array([5.0, 7.0, 15.0, 17.0]),
            centers=np.array([5.0, 6.0, 15.0, 16.0]),
            edges=np.array([0.0, 8.0, 20.0]),
        )
        self.assertEqual(table["n"].tolist(), [2, 2])
        np.testing.assert_allclose(
            table["mean_signed_error"].to_numpy(),
            [-1.0, 1.0],
        )

    def test_pit_decile_table_shares_sum_to_one(self) -> None:
        pit = np.linspace(0.05, 0.95, 20)
        table = pit_decile_table(pit, bins=10)
        self.assertEqual(len(table), 10)
        np.testing.assert_allclose(table["share"].sum(), 1.0)

    def test_residual_around_p_uses_hat_p_only(self) -> None:
        panel = _panel()
        epsilon = fit_preholdout_variant(panel, "residual_around_p")
        hat_p = panel["points_hat"].to_numpy(dtype=float)
        draws = simulate_residual_around_p(
            hat_p,
            epsilon,
            n_draws=64,
            seed=42,
        )
        self.assertEqual(draws.shape, (len(panel), 64))
        self.assertEqual(draws.dtype, np.float32)
        self.assertTrue(np.all(draws >= 0.0))
        again = simulate_residual_around_p(
            hat_p,
            epsilon,
            n_draws=64,
            seed=42,
        )
        np.testing.assert_array_equal(draws, again)
