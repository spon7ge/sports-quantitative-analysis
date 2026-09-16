"""Pipeline-integrity audit: one row per player/line/book."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.xgboost_models.joint_simulation import JointSimulation
from src.models.xgboost_models.snapshot_audit import (
    EDGE_BUCKETS,
    assumed_tip_utc,
    clip_mean_point_draws,
    draw_line_probabilities,
    edge_bucket,
    expected_return_per_unit,
    market_over_probability,
    select_snapshot_quotes,
    settle_sides,
    summarize_current_vs_clip,
    summarize_current_vs_market,
    summarize_edge_buckets,
    summarize_integrity,
    summarize_side_split,
)


def _sim(*, raw: np.ndarray, hat_p: float, g: float = 0.0) -> JointSimulation:
    n = len(raw)
    return JointSimulation(
        player_id=1,
        game_id="g",
        hat_m=24.0,
        expected_minutes=24.0,
        hat_p=hat_p,
        g=g,
        mu=max(0.0, hat_p + g),
        beta=0.0,
        beta_bin=0,
        minutes_bin=0,
        epsilon_bin=0,
        minute_draws=np.full(n, 24.0),
        point_draws=np.maximum(0.0, raw),
        epsilon_draws=raw - hat_p - g,
        minutes_clip_rate=0.0,
        points_clip_rate=float(np.mean(raw < 0.0)),
        seed=0,
        n_draws=n,
        bundle_hash="t",
    )


def _quote(
    *,
    book: str = "FanDuel",
    name: str = "Jay Huff",
    side: str = "Over",
    line: float = 10.5,
    odds: int = -110,
    commence: str = "2026-02-11",
    last_update: str = "2026-02-10T22:46:47Z",
    pulled: str = "2026-02-10 14:47:44",
) -> dict:
    return {
        "BOOKMAKER": book,
        "CATEGORY": "player_points",
        "NAME": name,
        "OVER/UNDER": side,
        "LINE": line,
        "ODDS": odds,
        "COMMENCE_TIME": commence,
        "LAST_UPDATE": last_update,
        "DATA_PULLED_AT": pulled,
    }


class ProbabilityAndEvTests(unittest.TestCase):
    def test_over_under_push_sum_to_one(self) -> None:
        draws = np.array([9.0, 10.0, 10.0, 12.0])
        probs = draw_line_probabilities(draws, 10.0)
        self.assertAlmostEqual(probs["p_over"], 0.25)
        self.assertAlmostEqual(probs["p_under"], 0.25)
        self.assertAlmostEqual(probs["p_push"], 0.5)
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_half_point_line_has_no_push_mass(self) -> None:
        draws = np.array([9.0, 10.0, 11.0])
        probs = draw_line_probabilities(draws, 10.5)
        self.assertAlmostEqual(probs["p_push"], 0.0)
        self.assertAlmostEqual(probs["p_over"], 1 / 3)
        self.assertAlmostEqual(probs["p_under"], 2 / 3)

    def test_no_vig_is_raw_over_over_sum(self) -> None:
        # +122 / -162 from the existing odds tests.
        p_over = market_over_probability(122, -162)
        self.assertAlmostEqual(p_over, 0.4215, places=4)

    def test_expected_return_credits_push_as_zero(self) -> None:
        # +100: win +1, lose -1, push 0.
        ev = expected_return_per_unit(
            p_win=0.4,
            p_lose=0.5,
            p_push=0.1,
            american_odds=100,
        )
        self.assertAlmostEqual(ev, -0.1)


class SettlementTests(unittest.TestCase):
    def test_dnp_voids_both_sides_not_under(self) -> None:
        settled = settle_sides(pts=0.0, minutes=0.0, line=10.5)
        self.assertEqual(settled["played"], False)
        self.assertEqual(settled["status"], "DNP")
        self.assertEqual(settled["over_result"], "void")
        self.assertEqual(settled["under_result"], "void")

    def test_integer_line_push(self) -> None:
        settled = settle_sides(pts=10.0, minutes=24.0, line=10.0)
        self.assertEqual(settled["played"], True)
        self.assertEqual(settled["status"], "played")
        self.assertEqual(settled["over_result"], "push")
        self.assertEqual(settled["under_result"], "push")

    def test_half_point_over_and_under(self) -> None:
        over = settle_sides(pts=11.0, minutes=20.0, line=10.5)
        self.assertEqual(over["over_result"], "win")
        self.assertEqual(over["under_result"], "loss")
        under = settle_sides(pts=10.0, minutes=20.0, line=10.5)
        self.assertEqual(under["over_result"], "loss")
        self.assertEqual(under["under_result"], "win")


class SnapshotCutoffTests(unittest.TestCase):
    def test_assumed_tip_is_7pm_et(self) -> None:
        tip = assumed_tip_utc("2026-02-11")
        self.assertEqual(str(tip), "2026-02-12 00:00:00+00:00")

    def test_keeps_last_quote_at_or_before_60_minutes(self) -> None:
        quotes = pd.DataFrame(
            [
                _quote(side="Over", odds=100, last_update="2026-02-11T21:00:00Z"),
                _quote(side="Under", odds=-120, last_update="2026-02-11T21:00:00Z"),
                _quote(side="Over", odds=110, last_update="2026-02-11T23:30:00Z"),
                _quote(side="Under", odds=-130, last_update="2026-02-11T23:30:00Z"),
                _quote(side="Over", odds=120, last_update="2026-02-12T00:30:00Z"),
                _quote(side="Under", odds=-140, last_update="2026-02-12T00:30:00Z"),
            ]
        )
        snapshot, closing = select_snapshot_quotes(quotes, minutes_before_tip=60)
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(
            set(snapshot["LAST_UPDATE"].astype(str)),
            {"2026-02-11 21:00:00+00:00"},
        )
        self.assertEqual(int(snapshot.loc[snapshot["OVER/UNDER"].eq("Over"), "ODDS"].iloc[0]), 100)
        self.assertEqual(len(closing), 2)
        self.assertEqual(
            set(closing["LAST_UPDATE"].astype(str)),
            {"2026-02-11 23:30:00+00:00"},
        )
        self.assertEqual(int(closing.loc[closing["OVER/UNDER"].eq("Over"), "ODDS"].iloc[0]), 110)

    def test_single_file_snapshot_is_also_closing(self) -> None:
        quotes = pd.DataFrame(
            [
                _quote(side="Over", odds=104),
                _quote(side="Under", odds=-128),
            ]
        )
        snapshot, closing = select_snapshot_quotes(quotes, minutes_before_tip=60)
        self.assertEqual(len(snapshot), 2)
        self.assertTrue(snapshot["LAST_UPDATE"].equals(closing["LAST_UPDATE"]))


class ClipMeanTests(unittest.TestCase):
    def test_shifted_draws_match_unclipped_target_mean(self) -> None:
        raw = np.array([-4.0, -1.0, 2.0, 8.0])
        hat_p = 2.0
        sim = _sim(raw=raw, hat_p=hat_p)
        shifted = clip_mean_point_draws(sim)
        self.assertAlmostEqual(float(shifted.mean()), hat_p, places=6)
        self.assertTrue(np.all(shifted >= 0.0))
        current = np.maximum(0.0, raw)
        self.assertGreater(float(current.mean()), hat_p)


class ReportTests(unittest.TestCase):
    def test_edge_bucket_edges(self) -> None:
        self.assertEqual(edge_bucket(0.019), "<2%")
        self.assertEqual(edge_bucket(0.02), "2–5%")
        self.assertEqual(edge_bucket(0.05), "5–10%")
        self.assertEqual(edge_bucket(0.10), ">10%")
        self.assertEqual(EDGE_BUCKETS, ("<2%", "2–5%", "5–10%", ">10%"))

    def test_integrity_flags_odds_after_tip_and_dnp_void(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "odds_before_tip": False,
                    "features_as_of_before_game": True,
                    "over_result": "void",
                    "under_result": "void",
                    "played": False,
                    "line": 10.5,
                    "actual_points": 0.0,
                    "status": "DNP",
                    "current_over_edge": 0.01,
                },
                {
                    "odds_before_tip": True,
                    "features_as_of_before_game": True,
                    "over_result": "win",
                    "under_result": "loss",
                    "played": True,
                    "line": 10.5,
                    "actual_points": 18.0,
                    "status": "played",
                    "current_over_edge": 0.22,
                },
            ]
        )
        report = summarize_integrity(rows)
        self.assertEqual(report["n_rows"], 2)
        self.assertEqual(report["odds_after_tip"], 1)
        self.assertEqual(report["dnp_voids"], 1)
        self.assertEqual(report["dnp_graded_as_zero"], 0)
        self.assertEqual(report["half_point_pushes"], 0)
        self.assertEqual(report["implausible_over_edges_gt_10pct"], 1)

    def test_current_vs_market_and_clip_and_side_split(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "player_name": "Low",
                    "book": "FanDuel",
                    "line": 8.5,
                    "hat_p": 6.0,
                    "current_p_over": 0.55,
                    "current_p_under": 0.45,
                    "current_p_push": 0.0,
                    "clip_mean_p_over": 0.40,
                    "clip_mean_p_under": 0.60,
                    "p_market_over": 0.48,
                    "current_over_edge": 0.07,
                    "current_under_edge": -0.07,
                    "over_result": "loss",
                    "under_result": "win",
                    "played": True,
                },
                {
                    "player_name": "Star",
                    "book": "DraftKings",
                    "line": 28.5,
                    "hat_p": 30.0,
                    "current_p_over": 0.52,
                    "current_p_under": 0.48,
                    "current_p_push": 0.0,
                    "clip_mean_p_over": 0.51,
                    "clip_mean_p_under": 0.49,
                    "p_market_over": 0.50,
                    "current_over_edge": 0.02,
                    "current_under_edge": -0.02,
                    "over_result": "win",
                    "under_result": "loss",
                    "played": True,
                },
            ]
        )
        vs_market = summarize_current_vs_market(rows, n_largest=2)
        self.assertAlmostEqual(vs_market["mean_over_edge"], 0.045)
        self.assertEqual(vs_market["largest"][0]["player_name"], "Low")

        vs_clip = summarize_current_vs_clip(rows)
        self.assertAlmostEqual(vs_clip["mean_over_shift"], 0.08)
        low = vs_clip["low_scorer_overs"]
        self.assertEqual(low["n"], 1)
        self.assertAlmostEqual(low["mean_over_shift"], 0.15)

        buckets = summarize_edge_buckets(rows)
        self.assertEqual(buckets["over"]["5–10%"]["n"], 1)
        self.assertEqual(buckets["over"]["5–10%"]["wins"], 0)
        self.assertEqual(buckets["over"]["5–10%"]["losses"], 1)
        self.assertEqual(buckets["over"]["2–5%"]["wins"], 1)

        sides = summarize_side_split(rows)
        self.assertEqual(sides["over"]["n_positive_edge"], 2)
        self.assertGreater(sides["over"]["mean_edge"], sides["under"]["mean_edge"])
