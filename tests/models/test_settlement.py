"""Settlement rules: DNP voids, overtime counts, missing players skip."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.settlement import (
    VoidedBetError,
    assert_bet_settles,
    expected_counts_from_minutes,
    maximum_minutes,
    regulation_minutes,
)


class SettlementRuleTests(unittest.TestCase):
    def test_nba_maximum_minutes_includes_overtime(self) -> None:
        self.assertEqual(regulation_minutes("nba"), 48)
        self.assertGreater(maximum_minutes("nba"), 48)
        self.assertEqual(maximum_minutes("nba"), 63)

    def test_wnba_maximum_minutes_includes_overtime(self) -> None:
        self.assertEqual(regulation_minutes("wnba"), 40)
        self.assertGreater(maximum_minutes("wnba"), 40)
        self.assertEqual(maximum_minutes("wnba"), 55)

    def test_known_dnp_voids_the_bet(self) -> None:
        with self.assertRaises(VoidedBetError):
            assert_bet_settles(is_dnp=True)

        with self.assertRaises(VoidedBetError):
            assert_bet_settles(availability_probability=0.0)

    def test_playing_player_settles(self) -> None:
        probability = assert_bet_settles(
            availability_probability=0.85
        )
        self.assertEqual(probability, 0.85)

    def test_expected_counts_scale_with_sampled_minutes(self) -> None:
        predicted_mean = np.array([20.0, 10.0])
        projected_minutes = np.array([20.0, 10.0])
        minute_samples = np.array(
            [
                [20.0, 40.0],
                [10.0, 5.0],
            ]
        )

        expected = expected_counts_from_minutes(
            predicted_mean,
            projected_minutes,
            minute_samples,
        )

        np.testing.assert_allclose(
            expected,
            np.array(
                [
                    [20.0, 40.0],
                    [10.0, 5.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
