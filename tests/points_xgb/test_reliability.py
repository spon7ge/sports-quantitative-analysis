"""Reliability curve: nominal tau against jittered coverage."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from points_xgb.eda import plot_reliability


def test_reliability_curve_plots_coverage_against_nominal_tau():
    perfect = pd.DataFrame(
        {"tau": [0.9, 0.1, 0.5], "coverage": [0.9, 0.1, 0.5], "gap": [0, 0, 0]}
    )
    shifted = pd.DataFrame(
        {"tau": [0.1, 0.5, 0.9], "coverage": [0.2, 0.5, 0.8]}
    )

    fig = plot_reliability({"perfect": perfect, "shifted": shifted})
    lines = fig.axes[0].lines

    assert [line.get_label() for line in lines] == [
        "perfect calibration",
        "perfect",
        "shifted",
    ]
    assert lines[1].get_xdata().tolist() == [0.1, 0.5, 0.9]
    assert lines[1].get_ydata().tolist() == [0.1, 0.5, 0.9]
    assert lines[2].get_ydata().tolist() == [0.2, 0.5, 0.8]
    plt.close(fig)


def test_reliability_curve_rejects_a_table_without_coverage():
    with pytest.raises(ValueError, match="coverage"):
        plot_reliability({"bad": pd.DataFrame({"tau": [0.5]})})
