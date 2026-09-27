"""Calibration by predicted-minutes tier, pre-holdout OOS and holdout apart."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.shared.metrics import calibration_by_minutes_tier, score_quantile_fold
from models.shared.oos import QUANTILE_LEVELS, oos_columns


def _row(
    *,
    minutes_q50: float,
    outcome_minutes: float,
    points: float,
    is_holdout: bool,
    spread: float = 1.0,
) -> dict:
    row = {
        "game_id": "g",
        "player_id": 1,
        "game_date": pd.Timestamp("2024-01-01"),
        "window_id": -1 if is_holdout else 1,
        "is_holdout": is_holdout,
        "minutes": outcome_minutes,
        "pts": points,
    }
    for level in QUANTILE_LEVELS:
        row[f"minutes_q_{level:.2f}"] = minutes_q50 + (level - 0.50) * 20
        row[f"rate_q_{level:.2f}"] = 0.50 + (level - 0.50) * spread
    return row


def test_tiers_follow_predicted_minutes_q50_and_flag_gaps_over_two_points():
    # Predicted q50 is 10, so the row is in <15 even though he played 40.
    low = _row(minutes_q50=10.0, outcome_minutes=40.0, points=20.0, is_holdout=False)
    # Realized minutes would have put him in 31+; the tier must not follow that.
    low["game_id"] = "low"
    low["player_id"] = 1
    high = _row(minutes_q50=33.0, outcome_minutes=5.0, points=1.0, is_holdout=False)
    high["game_id"] = "high"
    high["player_id"] = 2
    # Holdout row stays in its own split.
    held = _row(minutes_q50=20.0, outcome_minutes=20.0, points=10.0, is_holdout=True)
    held["game_id"] = "held"
    held["player_id"] = 3
    frame = pd.DataFrame([low, high, held])
    report = calibration_by_minutes_tier(frame)

    assert set(report["split"]) == {"pre-holdout OOS", "holdout"}
    pre = report.loc[report["split"].eq("pre-holdout OOS") & report["target"].eq("minutes")]
    assert set(pre["tier"]) == {"<15", "31+"}
    assert "15-24" not in set(pre["tier"])
    low_tier = pre.loc[pre["tier"].eq("<15") & pre["check"].eq("q0.05")]
    # Outcome 40 is above every knot around q50=10, so the empirical share is 0.
    assert low_tier["empirical"].iloc[0] == 0.0
    assert bool(low_tier["flag"].iloc[0])
    assert low_tier["ideal"].iloc[0] == 0.05

    held_rate = report.loc[
        report["split"].eq("holdout")
        & report["target"].eq("rate")
        & report["tier"].eq("15-24")
        & report["check"].eq("Q20-Q80")
    ]
    # points/minutes = 0.5, and q20/q80 bracket 0.5 when spread is 1.
    assert held_rate["empirical"].iloc[0] == 1.0
    assert held_rate["ideal"].iloc[0] == 0.60
    assert bool(held_rate["flag"].iloc[0])


def test_rate_score_uses_the_capped_label():
    row = _row(minutes_q50=20.0, outcome_minutes=1.0, points=12.0, is_holdout=False, spread=0.0)
    row["game_id"] = "cap"
    # Every rate knot is 0.50. Capped label is 6.0, so nothing falls at or below 0.50.
    frame = pd.DataFrame([row])
    report = calibration_by_minutes_tier(frame)
    share = report.loc[
        report["target"].eq("rate") & report["check"].eq("q0.95"),
        "empirical",
    ].iloc[0]
    assert share == 0.0


def test_gap_of_exactly_two_points_is_not_flagged():
    rows = []
    for index in range(50):
        row = _row(
            minutes_q50=20.0,
            outcome_minutes=10.0 if index < 26 else 30.0,
            points=10.0,
            is_holdout=False,
        )
        row["game_id"] = f"g{index}"
        row["player_id"] = index
        for level in QUANTILE_LEVELS:
            row[f"minutes_q_{level:.2f}"] = 20.0
        rows.append(row)
    report = calibration_by_minutes_tier(pd.DataFrame(rows))
    q50 = report.loc[
        report["target"].eq("minutes") & report["check"].eq("q0.50")
    ].iloc[0]
    assert q50["empirical"] == 0.52
    assert q50["ideal"] == 0.50
    assert abs(q50["gap"] - 0.02) < 1e-12
    assert not bool(q50["flag"])

    rows[26]["minutes"] = 10.0
    report = calibration_by_minutes_tier(pd.DataFrame(rows))
    q50 = report.loc[
        report["target"].eq("minutes") & report["check"].eq("q0.50")
    ].iloc[0]
    assert q50["empirical"] == 0.54
    assert bool(q50["flag"])


def test_report_columns_cover_the_stored_knots():
    row = _row(minutes_q50=20.0, outcome_minutes=20.0, points=10.0, is_holdout=False)
    frame = pd.DataFrame([row])
    assert set(oos_columns()).issuperset(
        {f"minutes_q_{level:.2f}" for level in QUANTILE_LEVELS}
    )
    report = calibration_by_minutes_tier(frame)
    checks = set(report["check"])
    for level in QUANTILE_LEVELS:
        assert f"q{level:.2f}" in checks
    assert {"Q20-Q80", "Q10-Q90", "Q05-Q95"} <= checks
    coverage = report.loc[report["check"].eq("Q10-Q90")].iloc[0]
    assert coverage["ideal"] == 0.80
    assert 0.0 <= coverage["empirical"] <= 1.0


def test_explicit_empty_tiers_skip_the_minute_defaults():
    actual = np.array([1.0, 2.0, 3.0])
    preds = {"q_0.50": np.array([1.0, 2.0, 3.0])}
    metrics = score_quantile_fold(
        actual, preds, fold_label="t", tiers={}, verbose=False
    )
    assert not any(str(key).startswith("pinball_<") for key in metrics)
