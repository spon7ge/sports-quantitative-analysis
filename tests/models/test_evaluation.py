"""Distribution scoring metrics for chronological model comparison."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.evaluation import (
    brier_score,
    crps,
    interval_coverage,
    interval_width,
    negative_log_likelihood,
    pit_histogram,
    pit_shape_penalty,
    probability_integral_transform,
    randomized_probability_integral_transform,
    role_coverage_gap,
    score_distribution,
    score_role_distribution,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_ensemble_has_low_crps_and_nll(self) -> None:
        actual = np.array([10, 10, 10], dtype=float)
        samples = np.full((3, 500), 10.0)
        self.assertLess(crps(samples, actual), 0.05)
        self.assertLess(
            negative_log_likelihood(samples, actual),
            0.2,
        )

    def test_interval_coverage_and_brier(self) -> None:
        rng = np.random.default_rng(0)
        actual = np.array([5.0, 15.0, 25.0])
        samples = np.stack(
            [
                rng.normal(5, 1, 2_000),
                rng.normal(15, 1, 2_000),
                rng.normal(25, 1, 2_000),
            ]
        )
        self.assertGreater(
            interval_coverage(samples, actual),
            0.9,
        )
        self.assertLess(
            brier_score(samples, actual, line=20.0),
            0.05,
        )

    def test_probability_integral_transform_handles_ties(self) -> None:
        actual = np.array([10.0, 2.0, 20.0])
        samples = np.array(
            [
                [10.0, 10.0, 10.0, 10.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        pit = probability_integral_transform(samples, actual)
        np.testing.assert_allclose(pit, [0.5, 0.375, 1.0])

    def test_randomized_pit_matches_half_credit_without_ties(self) -> None:
        actual = np.array([2.0, 20.0])
        samples = np.array(
            [
                [1.0, 3.0, 5.0, 7.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        half = probability_integral_transform(samples, actual)
        randomized = randomized_probability_integral_transform(
            samples,
            actual,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(randomized, half)

    def test_randomized_pit_spreads_zero_mass(self) -> None:
        actual = np.zeros(4)
        samples = np.zeros((4, 8))
        randomized = randomized_probability_integral_transform(
            samples,
            actual,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(np.all((randomized >= 0.0) & (randomized <= 1.0)))
        self.assertGreater(float(np.std(randomized)), 0.2)
        half = probability_integral_transform(samples, actual)
        np.testing.assert_allclose(half, np.full(4, 0.5))

    def test_randomized_pit_has_same_expectation_as_half_credit(self) -> None:
        rng = np.random.default_rng(2)
        actual = np.array([0.0, 0.0, 4.0, 12.0])
        samples = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 4.0, 6.0],
                [8.0, 10.0, 12.0, 14.0, 16.0],
            ]
        )
        half = probability_integral_transform(samples, actual)
        draws = np.stack(
            [
                randomized_probability_integral_transform(
                    samples,
                    actual,
                    rng=rng,
                )
                for _ in range(2_000)
            ]
        )
        np.testing.assert_allclose(draws.mean(axis=0), half, atol=0.03)

    def test_score_distribution_returns_required_metrics(self) -> None:
        samples = np.array([[1.0, 2.0, 3.0, 4.0]])
        scores = score_distribution(
            samples,
            np.array([3.0]),
            line=2.5,
        )
        self.assertEqual(
            set(scores),
            {
                "negative_log_likelihood",
                "crps",
                "coverage_80",
                "brier",
            },
        )

    def test_interval_width_grows_with_spread(self) -> None:
        rng = np.random.default_rng(0)
        narrow = rng.normal(10.0, 1.0, size=(20, 2_000))
        wide = rng.normal(10.0, 5.0, size=(20, 2_000))
        self.assertLess(
            interval_width(narrow),
            interval_width(wide),
        )
        self.assertGreater(interval_width(wide), 5.0)

    def test_pit_shape_is_near_zero_when_uniform(
        self,
    ) -> None:
        rng = np.random.default_rng(1)
        uniform = rng.uniform(0.0, 1.0, size=8_000)
        piled = np.full(8_000, 0.05)
        self.assertEqual(
            pit_histogram(uniform, bins=10).sum(),
            8_000,
        )
        self.assertLess(pit_shape_penalty(uniform), 0.05)
        self.assertGreater(
            pit_shape_penalty(piled),
            pit_shape_penalty(uniform),
        )

    def test_role_coverage_gap_when_starters_miss(
        self,
    ) -> None:
        samples = np.full((4, 400), 10.0)
        actual = np.array([100.0, 100.0, 10.0, 10.0])
        start_rate = np.array([1.0, 0.9, 0.0, 0.1])
        gap = role_coverage_gap(
            samples,
            actual,
            start_rate,
        )
        self.assertEqual(gap["starter_coverage"], 0.0)
        self.assertEqual(gap["bench_coverage"], 1.0)
        self.assertEqual(gap["gap"], 1.0)
        self.assertEqual(gap["n_starter"], 2)
        self.assertEqual(gap["n_bench"], 2)

        scores = score_role_distribution(
            samples,
            actual,
            start_rate,
            line=9.5,
        )
        self.assertIn("width_80", scores)
        self.assertIn("pit_mean", scores)
        self.assertIn("pit_std", scores)
        self.assertIn("pit_shape", scores)
        self.assertIn("starter_coverage", scores)
        self.assertIn("negative_log_likelihood", scores)
        self.assertIn("crps", scores)
        self.assertIn("coverage_80", scores)
        self.assertIn("brier", scores)

    def test_score_distribution_can_skip_crps(self) -> None:
        scores = score_distribution(
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            np.array([3.0]),
            include_crps=False,
        )
        self.assertNotIn("crps", scores)
        self.assertIn("negative_log_likelihood", scores)
        self.assertIn("coverage_80", scores)


if __name__ == "__main__":
    unittest.main()
