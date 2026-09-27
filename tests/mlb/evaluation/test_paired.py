"""Paired loss deltas vs shrunk_kbf with Diebold-Mariano HAC SEs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.mlb.evaluation.backtest import (
    format_paired_comparison_table,
    paired_loss_stats,
    run_backtest,
)


def test_paired_loss_stats_iid_se_and_ci() -> None:
    diffs = np.array([0.2, -0.1, 0.0, 0.1, -0.05, 0.15])
    stats = paired_loss_stats(diffs)
    n = diffs.size
    expected_mean = float(diffs.mean())
    expected_se = float(diffs.std(ddof=1) / np.sqrt(n))
    assert stats["n"] == n
    assert stats["mean_diff"] == pytest.approx(expected_mean)
    assert stats["paired_se"] == pytest.approx(expected_se)
    assert stats["dm_stat"] == pytest.approx(expected_mean / stats["hac_se"])
    half = 1.96 * float(stats["hac_se"])
    assert stats["ci_low"] == pytest.approx(expected_mean - half)
    assert stats["ci_high"] == pytest.approx(expected_mean + half)


def test_hac_se_inflates_under_positive_autocorrelation() -> None:
    rng = np.random.default_rng(7)
    n = 400
    noise = rng.normal(size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + noise[i]
    iid = paired_loss_stats(noise)
    serial = paired_loss_stats(ar)
    assert serial["hac_se"] > iid["hac_se"]
    assert serial["hac_se"] > serial["paired_se"]


def test_zero_mean_series_is_marked_inside_noise() -> None:
    stats = paired_loss_stats(np.zeros(50))
    assert stats["mean_diff"] == 0.0
    assert stats["inside_noise"] is True
    assert stats["ci_low"] <= 0.0 <= stats["ci_high"]


def test_large_shift_is_outside_noise() -> None:
    stats = paired_loss_stats(np.full(80, -0.4))
    assert stats["inside_noise"] is False
    assert stats["ci_high"] < 0.0


def test_format_table_marks_inside_noise() -> None:
    rows = pd.DataFrame(
        [
            {
                "fold": "eval_2022",
                "model": "strikeout_nb",
                "reference": "shrunk_kbf",
                "metric": "pmf_nll",
                "n": 50,
                "mean_diff": -0.01,
                "paired_se": 0.02,
                "hac_se": 0.025,
                "dm_stat": -0.4,
                "ci_low": -0.06,
                "ci_high": 0.04,
                "inside_noise": True,
            },
            {
                "fold": "eval_2022",
                "model": "strikeout_nb",
                "reference": "shrunk_kbf",
                "metric": "discrete_crps",
                "n": 50,
                "mean_diff": -0.08,
                "paired_se": 0.01,
                "hac_se": 0.012,
                "dm_stat": -6.7,
                "ci_low": -0.104,
                "ci_high": -0.056,
                "inside_noise": False,
            },
        ]
    )
    text = format_paired_comparison_table(rows)
    assert "strikeout_nb vs shrunk_kbf" in text
    assert "INSIDE_NOISE" in text
    assert "pmf_nll" in text
    assert "discrete_crps" in text


def test_backtest_emits_paired_deltas_vs_shrunk_kbf(fixture_tables, mlb_config) -> None:
    result = run_backtest(fixture_tables, mlb_config)
    paired = result["paired_scores"]
    assert not paired.empty
    assert set(paired["reference"].unique()) == {"shrunk_kbf"}
    assert {"pmf_nll", "discrete_crps"} <= set(paired["metric"])
    assert "strikeout_nb" in set(paired["model"])
    assert "shrunk_kbf" not in set(paired["model"])
    required = {
        "fold",
        "model",
        "reference",
        "metric",
        "n",
        "mean_diff",
        "paired_se",
        "hac_se",
        "dm_stat",
        "ci_low",
        "ci_high",
        "inside_noise",
    }
    assert required <= set(paired.columns)
    nll = paired.loc[paired["metric"] == "pmf_nll"]
    assert (nll["n"] > 0).all()
    assert nll["paired_se"].notna().all()
