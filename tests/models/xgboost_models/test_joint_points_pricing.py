"""Price one paired full-game PTS market from joint draws."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.models.odds import (
    american_to_implied_probability,
    expected_value,
    remove_vig,
)
from src.models.xgboost_models.joint_pricing import (
    CanonicalPlayerPointsMarket,
    SKIP_NO_FEATURE,
    SKIP_OUT,
    UNSUPPORTED_INTEGER_LINE,
    canonicalize_player_points_market,
    price_joint_points_market,
    price_one_player_points_line,
)
from src.models.xgboost_models.joint_simulation import (
    JointPointsSimulator,
    JointSimulation,
)
from tests.models.xgboost_models.joint_fixtures import (
    feature_row,
    load_bundle,
    write_bundle,
)


def _market(**overrides) -> CanonicalPlayerPointsMarket:
    payload = dict(
        book="fanduel",
        event="0022400123",
        player_id=2544,
        game_id="0022400123",
        stat="PTS",
        period="full_game",
        line=9.5,
        over_odds=122,
        under_odds=-162,
        quote_ts="2024-12-01T17:00:00Z",
    )
    payload.update(overrides)
    return CanonicalPlayerPointsMarket(**payload)


def _result(*, p_over: float, n: int = 100) -> JointSimulation:
    n_over = int(round(p_over * n))
    draws = np.concatenate(
        [np.full(n_over, 20.0), np.full(n - n_over, 5.0)]
    )
    return JointSimulation(
        player_id=2544,
        game_id="0022400123",
        hat_m=24.0,
        expected_minutes=24.0,
        hat_p=12.0,
        g=0.0,
        mu=12.0,
        beta=0.45,
        beta_bin=3,
        minutes_bin=3,
        epsilon_bin=1,
        minute_draws=np.full(n, 24.0),
        point_draws=draws,
        epsilon_draws=np.zeros(n),
        minutes_clip_rate=0.0,
        points_clip_rate=0.0,
        seed=42,
        n_draws=n,
        bundle_hash="abc",
    )


class CanonicalMarketTests(unittest.TestCase):
    def test_rejects_unpaired_sides(self) -> None:
        over = {
            "book": "fanduel",
            "event": "e1",
            "player_id": 1,
            "game_id": "g1",
            "stat": "PTS",
            "period": "full_game",
            "line": 9.5,
            "side": "over",
            "odds": 122,
            "quote_ts": "t0",
        }
        under = dict(over)
        under["side"] = "under"
        under["odds"] = -162
        market = canonicalize_player_points_market(over, under)
        self.assertEqual(market.line, 9.5)

        broken = dict(under)
        broken["book"] = "draftkings"
        with self.assertRaises(ValueError):
            canonicalize_player_points_market(over, broken)


class OddsMathTests(unittest.TestCase):
    def test_hand_calculated_plus_122_minus_162(self) -> None:
        over_raw = american_to_implied_probability(122)
        under_raw = american_to_implied_probability(-162)
        self.assertAlmostEqual(over_raw, 0.45045, places=5)
        self.assertAlmostEqual(under_raw, 0.61832, places=5)
        over_vf, under_vf = remove_vig(122, -162)
        self.assertAlmostEqual(over_vf, 0.4215, places=4)
        self.assertAlmostEqual(under_vf, 0.5785, places=4)

        priced = price_joint_points_market(_result(p_over=0.5), _market())
        p_over = priced["over"]["model_probability"]
        p_under = priced["under"]["model_probability"]
        self.assertAlmostEqual(
            priced["over"]["ev"],
            p_over * 1.22 - (1 - p_over),
        )
        self.assertAlmostEqual(
            priced["under"]["ev"],
            p_under * (100 / 162) - (1 - p_under),
        )
        self.assertAlmostEqual(
            priced["under"]["ev"],
            expected_value(p_under, p_over, -162),
        )


class ProbabilityTests(unittest.TestCase):
    def test_half_point_probabilities_sum_to_one(self) -> None:
        priced = price_joint_points_market(_result(p_over=0.37), _market())
        self.assertAlmostEqual(
            priced["over"]["model_probability"]
            + priced["under"]["model_probability"],
            1.0,
        )

    def test_over_probability_is_nonincreasing_in_the_line(self) -> None:
        draws = _result(p_over=0.6)
        low = price_joint_points_market(draws, _market(line=8.5))
        high = price_joint_points_market(draws, _market(line=24.5))
        self.assertGreaterEqual(
            low["over"]["model_probability"],
            high["over"]["model_probability"],
        )

    def test_integer_line_is_unsupported(self) -> None:
        priced = price_joint_points_market(
            _result(p_over=0.5),
            _market(line=10.0),
        )
        self.assertEqual(priced["status"], UNSUPPORTED_INTEGER_LINE)

    def test_ranks_by_ev_and_picks_at_most_one_side(self) -> None:
        priced = price_joint_points_market(_result(p_over=1.0), _market())
        self.assertEqual(priced["status"], "PRICED")
        self.assertEqual(priced["selected_side"], "over")
        self.assertGreater(priced["over"]["ev"], priced["under"]["ev"])

    def test_positive_vig_free_edge_with_nonpositive_ev_is_no_bet(
        self,
    ) -> None:
        priced = price_joint_points_market(
            _result(p_over=0.43),
            _market(),
        )
        self.assertGreater(priced["over"]["edge"], 0.0)
        self.assertLessEqual(priced["over"]["ev"], 0.0)
        self.assertLessEqual(priced["under"]["ev"], 0.0)
        self.assertEqual(priced["status"], "NO_BET")
        self.assertIsNone(priced["selected_side"])


class SkipTests(unittest.TestCase):
    def test_availability_zero_skips_without_simulation(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            simulator = JointPointsSimulator(bundle)
            with patch.object(
                simulator,
                "simulate",
                side_effect=AssertionError("simulate must not run"),
            ):
                priced = price_one_player_points_line(
                    simulator,
                    feature_row(),
                    _market(),
                    availability_probability=0.0,
                )
        self.assertEqual(priced["status"], SKIP_OUT)

    def test_missing_features_never_become_zero_points(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = load_bundle(write_bundle(Path(tmp)))
            simulator = JointPointsSimulator(bundle)
            priced = price_one_player_points_line(
                simulator,
                {"player_id": 1, "game_id": "g"},
                _market(),
            )
        self.assertEqual(priced["status"], SKIP_NO_FEATURE)
        self.assertNotIn("point_draws", priced)
        self.assertNotEqual(priced["status"], "PRICED")
