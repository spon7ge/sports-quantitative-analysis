"""Width-scaled quantile offsets and jittered coverage."""

from __future__ import annotations

import numpy as np
from models.shared.metrics import score_quantile_fold
from models.shared.quantile_offsets import (
    apply_offsets,
    calibration_table,
    fit_offsets,
    line_band_gap,
    mean_pinball,
    offset_walk_forward,
    quantile_reliability_bins,
    reliability_pairs,
)


class _Jitter:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def uniform(self, low, high, size):
        assert low == -0.5 and high == 0.5
        assert int(size) == len(self.values)
        return self.values.copy()


def test_reliability_pairs_match_jittered_coverage():
    y = np.array([1.0, 4.0])
    grid = np.array([[0.0, 1.0, 3.0], [2.0, 4.0, 6.0]])
    taus = np.array([0.25, 0.50, 0.75])
    jitter = [0.25, -0.25]
    table = calibration_table(y, grid, taus, _Jitter(jitter))
    outcomes, predicted = reliability_pairs(y, grid, taus, _Jitter(jitter))
    assert outcomes.shape == predicted.shape == (6,)
    for level, coverage in zip(table["tau"], table["coverage"], strict=True):
        mask = predicted == level
        assert int(mask.sum()) == len(y)
        assert outcomes[mask].mean() == coverage


def test_reliability_diagram_keeps_one_point_and_count_per_tau():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ml_insights as mli

    y = np.array([1.0, 4.0, 0.0])
    grid = np.array([[0.0, 1.0, 3.0], [2.0, 4.0, 6.0], [0.5, 1.5, 2.5]])
    taus = np.array([0.25, 0.50, 0.75])
    jitter = [0.0, 0.0, 0.0]
    table = calibration_table(y, grid, taus, _Jitter(jitter))
    outcomes, predicted = reliability_pairs(y, grid, taus, _Jitter(jitter))
    diagram = mli.plot_reliability_diagram(
        outcomes,
        predicted,
        bins=quantile_reliability_bins(taus),
        show_histogram=True,
        ci_ref="point",
    )
    plt.close("all")
    order = np.argsort(diagram["pred_probs"])
    np.testing.assert_allclose(diagram["pred_probs"][order], taus)
    np.testing.assert_allclose(diagram["emp_probs"][order], table["coverage"])
    np.testing.assert_allclose(diagram["bin_counts"], len(y))


def test_quantile_bins_put_each_tau_in_its_own_interval():
    taus = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    bins = quantile_reliability_bins(taus)
    index = np.digitize(taus, bins)
    assert len(set(index.tolist())) == len(taus)
    assert int(index.min()) >= 1
    assert int(index.max()) < len(bins)


def test_calibration_table_uses_jitter_for_coverage_and_raw_y_for_pinball():
    # yj = 1.25. Coverage is 0, 0, 1. Pinball is hand-computed on y = 1.
    table = calibration_table(
        np.array([1.0]),
        np.array([[0.0, 1.0, 3.0]]),
        np.array([0.25, 0.50, 0.75]),
        _Jitter([0.25]),
    )
    assert table["coverage"].tolist() == [0.0, 0.0, 1.0]
    assert table["gap"].tolist() == [-0.25, -0.50, 0.25]
    assert table["pinball"].tolist() == [0.25, 0.0, 0.5]
    assert line_band_gap(table, low=0.25, high=0.50) == 0.50


def test_offsets_are_in_units_of_each_rows_width():
    y = np.array([0.0, 10.0])
    grid = np.array([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]])
    taus = np.array([0.10, 0.50, 0.90])
    offsets = fit_offsets(y, grid, taus, _Jitter([0.0, 0.0]))
    # Residuals / width 10, then the tau quantile of each column.
    np.testing.assert_allclose(offsets, [0.1, 0.0, -0.1])

    shifted = apply_offsets(grid, offsets, taus)
    np.testing.assert_allclose(shifted, [[1.0, 5.0, 9.0], [1.0, 5.0, 9.0]])

    narrow = np.array([[0.0, 1.0, 2.0]])
    wide = np.array([[0.0, 10.0, 20.0]])
    same = np.array([0.0, 0.0, 0.2])
    narrow_shift = apply_offsets(narrow, same, taus) - narrow
    wide_shift = apply_offsets(wide, same, taus) - wide
    np.testing.assert_allclose(wide_shift, narrow_shift * 10)


def test_apply_offsets_sorts_a_crossed_row():
    taus = np.array([0.10, 0.50, 0.90])
    crossed = np.array([[5.0, 0.0, 4.0]])
    ordered = apply_offsets(crossed, np.zeros(3), taus)
    assert np.all(np.diff(ordered, axis=1) >= 0)


def test_walk_forward_fits_on_earlier_folds_and_scores_the_next():
    taus = np.array([0.10, 0.50, 0.90])
    # Every row is the same shape. Fold 2 is scored from offsets fit on fold 1.
    grid = np.tile(np.array([[0.0, 5.0, 10.0]]), (6, 1))
    y = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    folds = np.array([1, 1, 1, 2, 2, 2])
    gaps = offset_walk_forward(y, grid, taus, folds, np.random.default_rng(0))
    assert gaps["fold"].tolist() == [2]
    assert 0.0 <= float(gaps["max_abs_gap"].iloc[0]) <= 1.0

    early = np.array([False, False, False, True, False, False])
    early_gaps = offset_walk_forward(
        y, grid, taus, folds, np.random.default_rng(0), eval_mask=early
    )
    assert early_gaps["n"].tolist() == [1]


def test_fold_score_headline_is_mean_pinball_across_the_grid():
    # q0.10 pinball is 0.1. q0.90 pinball is 0.5. The mean is 0.3.
    metrics = score_quantile_fold(
        np.array([0.0, 2.0]),
        {"q_0.10": np.array([0.0, 0.0]), "q_0.90": np.array([1.0, 1.0])},
        fold_label="toy",
        verbose=False,
    )
    assert metrics["mean_pinball"] == 0.3
    assert metrics["pinball_q_0.10"] == 0.1
    assert metrics["pinball_q_0.90"] == 0.5


def test_mean_pinball_is_the_grid_average():
    # One row, d = [1, -2]. Pinball 0.25 and 0.5. Mean 0.375.
    value = mean_pinball(
        np.array([1.0]),
        np.array([[0.0, 3.0]]),
        np.array([0.25, 0.75]),
    )
    assert value == 0.375
