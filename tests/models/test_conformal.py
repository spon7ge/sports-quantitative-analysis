"""One-sided conformal offsets for the minutes quantile knots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.shared.conformal import (
    apply_offsets,
    base_knots,
    calibration_group,
    calibration_payload,
    conformal_offset,
    fit_offsets,
    role_quantile_table,
)

LEVELS = {
    "q_0.05": 0.05,
    "q_0.10": 0.10,
    "q_0.20": 0.20,
    "q_0.30": 0.30,
    "q_0.40": 0.40,
    "q_0.50": 0.50,
    "q_0.60": 0.60,
    "q_0.70": 0.70,
    "q_0.80": 0.80,
    "q_0.90": 0.90,
    "q_0.95": 0.95,
}

# Knots of a normal with mean 30 and sd 8, shifted down by 12 minutes.
_Z = {
    0.05: -1.6448536269514722,
    0.10: -1.2815515655446004,
    0.20: -0.8416212335729143,
    0.30: -0.5244005127080407,
    0.40: -0.2533471031357997,
    0.50: 0.0,
    0.60: 0.2533471031357997,
    0.70: 0.5244005127080407,
    0.80: 0.8416212335729143,
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
}


def test_floor_and_ceil_pick_the_order_statistic_and_reject_a_rank_outside_the_sample():
    scores = np.array([4.0, 1.0, 3.0, 2.0])

    assert conformal_offset(scores, 0.50) == 3.0
    assert conformal_offset(scores, 0.25) == 1.0
    assert conformal_offset(scores, 0.75) == 4.0
    assert conformal_offset(scores, 0.40) == 2.0
    assert conformal_offset(scores, 0.80) == 4.0

    with pytest.raises(ValueError, match="k="):
        conformal_offset(scores, 0.10)
    with pytest.raises(ValueError, match="k="):
        conformal_offset(scores, 0.95)


def test_exchangeable_knots_land_near_their_levels_after_the_offset():
    rng = np.random.default_rng(42)
    n_cal, n_test = 20_000, 50_000
    y_cal = rng.normal(30.0, 8.0, size=n_cal)
    y_test = rng.normal(30.0, 8.0, size=n_test)
    raw = {key: np.full(n_cal, 18.0 + 4.0 * _Z[tau]) for key, tau in LEVELS.items()}
    base = pd.DataFrame(raw)
    group = np.full(n_cal, "15-24", dtype=object)
    offsets = fit_offsets(base, y_cal, group, LEVELS, min_n=300)

    test = pd.DataFrame(
        {key: np.full(n_test, 18.0 + 4.0 * _Z[tau]) for key, tau in LEVELS.items()}
    )
    calibrated = apply_offsets(
        test,
        np.full(n_test, "15-24", dtype=object),
        offsets,
        max_minutes=63.0,
    )

    raw_median = float(np.mean(y_test <= test["q_0.50"].to_numpy()))
    assert abs(raw_median - 0.50) > 0.20
    for key, tau in LEVELS.items():
        coverage = float(np.mean(y_test <= calibrated[key].to_numpy()))
        assert abs(coverage - tau) < 0.02


def test_apply_offsets_sorts_clips_and_leaves_an_unknown_group_unshifted():
    base = pd.DataFrame(
        {
            "q_0.10": [10.0, 5.0, 0.0],
            "q_0.50": [20.0, 6.0, 0.0],
            "q_0.90": [30.0, 7.0, 100.0],
        }
    )
    offsets = pd.DataFrame(
        {"q_0.10": [5.0], "q_0.50": [-1.0], "q_0.90": [-20.0]},
        index=["15-24"],
    )

    out = apply_offsets(
        base,
        ["15-24", "nope", "15-24"],
        offsets,
        max_minutes=63.0,
    )

    assert list(out.columns) == ["q_0.10", "q_0.50", "q_0.90"]
    assert np.all(np.diff(out.to_numpy(), axis=1) >= 0)
    np.testing.assert_allclose(out.iloc[1].to_numpy(), [5.0, 6.0, 7.0])
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [10.0, 15.0, 19.0])
    np.testing.assert_allclose(out.iloc[2].to_numpy(), [1e-3, 5.0, 63.0])
    assert out.to_numpy().min() >= 1e-3
    assert out.to_numpy().max() <= 63.0


def test_role_quantiles_read_a_named_target_column():
    history = pd.DataFrame(
        {"starting": [1, 1, 0], "pts_per_min": [0.2, 0.6, 0.1]}
    )
    table = role_quantile_table(history, [0.50], target_col="pts_per_min")
    assert table.loc["starter", "q_0.50"] == pytest.approx(0.4)


def test_rate_clip_keeps_zero_and_stops_at_the_cap():
    base = pd.DataFrame({"q_0.50": [-1.0, 0.0, 9.0]})
    offsets = pd.DataFrame({"q_0.50": [0.0]}, index=["<0.30"])
    out = apply_offsets(
        base,
        ["<0.30", "<0.30", "<0.30"],
        offsets,
        max_minutes=6.0,
        lower=0.0,
    )
    np.testing.assert_allclose(out["q_0.50"].to_numpy(), [0.0, 0.0, 6.0])


def test_rate_groups_use_left_closed_cuts_and_a_missing_lag_is_cold_start():
    from models.shared.metrics import RATE_Q50_TIERS

    labels = calibration_group(
        [0.29, 0.30, 0.50, 0.70, 0.10],
        [1.0, 1.0, 1.0, 1.0, np.nan],
        tiers=RATE_Q50_TIERS,
    )
    assert list(labels) == ["<0.30", "0.30-0.50", "0.50-0.70", "0.70+", "cold_start"]


def test_small_groups_are_omitted_and_cold_start_is_not_a_minute_tier():
    y = np.arange(1, 6, dtype=float)
    base = pd.DataFrame({"q_0.50": np.zeros(5)})
    group = np.array(["15-24"] * 4 + ["cold_start"])
    empty = fit_offsets(base, y, group, {"q_0.50": 0.50}, min_n=300)
    assert list(empty.columns) == ["q_0.50"]
    assert empty.empty

    labels = calibration_group(
        [14.9, 15.0, 24.0, 31.0, 40.0, np.nan],
        [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
    )
    assert list(labels) == ["<15", "15-24", "24-31", "31+", "cold_start", "cold_start"]


def test_base_knots_sort_tree_rows_and_replace_only_cold_starts_from_history():
    preds = pd.DataFrame(
        {
            "q_0.90": [50.0, 8.0],
            "q_0.10": [100.0, 9.0],
            "q_0.50": [0.0, 1.0],
        }
    )
    history = pd.DataFrame(
        {
            "starting": [1, 1, 0, 0],
            "minutes": [10.0, 30.0, 2.0, 4.0],
        }
    )

    out = base_knots(preds, [1, 0], [np.nan, 12.0], history)

    assert list(out.columns) == ["q_0.10", "q_0.50", "q_0.90"]
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [12.0, 20.0, 28.0])
    np.testing.assert_allclose(out.iloc[1].to_numpy(), [1.0, 8.0, 9.0])


def test_stored_role_quantiles_match_the_history_fallback():
    preds = pd.DataFrame({"q_0.10": [100.0], "q_0.50": [0.0], "q_0.90": [50.0]})
    history = pd.DataFrame(
        {"starting": [1, 1, 0, 0], "minutes": [10.0, 30.0, 2.0, 4.0]}
    )
    table = role_quantile_table(history, [0.10, 0.50, 0.90])
    from_history = base_knots(preds, [1], [np.nan], history)
    from_table = base_knots(preds, [1], [np.nan], role_quantiles=table)

    np.testing.assert_allclose(table.loc["starter"].to_numpy(), [12.0, 20.0, 28.0])
    np.testing.assert_allclose(from_table.to_numpy(), from_history.to_numpy())
    np.testing.assert_allclose(from_table.iloc[0].to_numpy(), [12.0, 20.0, 28.0])


def test_calibration_payload_records_every_group_and_the_date_span():
    offsets = pd.DataFrame({"q_0.50": [0.2]}, index=pd.Index(["<15"], name="group"))
    groups = np.array(["<15", "<15", "cold_start"])
    dates = pd.to_datetime(["2024-01-02", "2024-03-01", "2024-01-02"])
    role = pd.DataFrame({"q_0.50": [20.0, 8.0]}, index=["starter", "bench"])
    payload = calibration_payload(
        offsets,
        groups,
        dates,
        min_n=300,
        role_quantiles=role,
        max_minutes=63.0,
    )

    assert payload["row_counts"]["<15"] == 2
    assert payload["row_counts"]["cold_start"] == 1
    assert payload["row_counts"]["31+"] == 0
    assert payload["date_start"] == "2024-01-02"
    assert payload["date_end"] == "2024-03-01"
    assert payload["min_n"] == 300
    assert payload["group_definition"]["tiers"][0][0] == "<15"
    assert payload["target"] == "minutes"
    assert payload["lag_col"] == "min_lag_1"
    assert payload["clip_low"] == pytest.approx(1e-3)
    assert payload["max_minutes"] == 63.0
    assert "fingerprint" not in payload
