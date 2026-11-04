"""PIT recalibration: one isotonic map of the randomized CDF."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from models.shared.artifacts import (
    load_model_bundle,
    predict_calibrated_quantiles,
    save_model_bundle,
)
from models.shared.points_calibration import (
    calibrated_quantile_grid,
    deterministic_quantile_pit,
    fit_pit_recalibrator,
    fit_randomized_pit_map,
    pit_walk_forward,
    probability_over,
    randomized_quantile_pit,
)
from models.shared.quantile_offsets import calibration_table, line_band_gap
from xgboost import XGBRegressor

TAUS = np.array([0.2, 0.5, 0.8])


def test_deterministic_pit_is_the_midpoint_of_the_atom():
    knots = np.array([[10.0, 20.0, 30.0]])
    y = np.array([20.0])

    mid = deterministic_quantile_pit(y, knots, TAUS)

    # F(19) = 0.47 and F(20) = 0.5, so the midpoint is 0.485.
    np.testing.assert_allclose(mid, [0.485])


def test_randomized_pit_spreads_the_integer_atom():
    knots = np.array([[10.0, 20.0, 30.0]])
    y = np.array([20.0])

    low = randomized_quantile_pit(y, knots, TAUS, np.array([0.0]))
    high = randomized_quantile_pit(y, knots, TAUS, np.array([1.0]))
    mid = randomized_quantile_pit(y, knots, TAUS, np.array([0.25]))

    # F(19) interpolates between q_0.20=10 and q_0.50=20: weight 0.9.
    # F(20) = 0.5. u = 0.47 + U * 0.03.
    np.testing.assert_allclose(low, [0.47])
    np.testing.assert_allclose(high, [0.50])
    np.testing.assert_allclose(mid, [0.4775])


def test_isotonic_map_is_the_empirical_cdf_of_u():
    uniform = np.array([0.2, 0.4, 0.6, 0.8])

    calibrator = fit_pit_recalibrator(uniform)

    np.testing.assert_allclose(
        calibrator.predict([0.2, 0.4, 0.5, 0.6, 0.8]),
        [0.25, 0.5, 0.625, 0.75, 1.0],
    )


def test_probability_over_is_one_minus_the_recalibrated_cdf():
    knots = np.array([[10.0, 20.0, 30.0]])
    calibrator = fit_pit_recalibrator(np.array([0.2, 0.4, 0.6, 0.8]))

    over = probability_over(20.0, knots, TAUS, calibrator)

    # F(20) = 0.5 and R(0.5) = 0.625, so P(over) = 0.375.
    np.testing.assert_allclose(over, [0.375])


def test_identity_map_leaves_the_quantile_grid_in_place():
    taus = np.array([0.25, 0.5, 0.75])
    knots = np.array([[10.0, 20.0, 30.0]])
    calibrator = fit_pit_recalibrator(np.array([0.25, 0.5, 0.75, 1.0]))

    out = calibrated_quantile_grid(knots, taus, calibrator)

    np.testing.assert_allclose(out, knots)


def test_a_low_pit_pulls_every_calibrated_quantile_down_to_q_low():
    taus = np.array([0.25, 0.5, 0.75])
    knots = np.tile(np.array([[20.0, 30.0, 40.0]]), (4, 1))
    y = np.full(4, 10.0)

    calibrator = fit_randomized_pit_map(
        y,
        knots,
        taus,
        np.zeros(4),
        pd.to_datetime(["2024-01-01"] * 4),
    )
    out = calibrated_quantile_grid(knots[:1], taus, calibrator)

    np.testing.assert_allclose(out, [[20.0, 20.0, 20.0]])
    np.testing.assert_allclose(
        probability_over(20.0, knots[:1], taus, calibrator), [0.0]
    )


def test_holdout_dates_are_refused():
    knots = np.array([[10.0, 20.0, 30.0]])
    with pytest.raises(ValueError, match="holdout"):
        fit_randomized_pit_map(
            np.array([20.0]),
            knots,
            TAUS,
            np.array([0.5]),
            pd.to_datetime(["2026-10-20"]),
        )


def test_walk_forward_fits_the_map_on_earlier_folds_only():
    taus = np.array([0.2, 0.5, 0.8])
    knots = np.tile(np.array([[0.0, 1.0, 2.0]]), (8, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    scored = pit_walk_forward(
        y, knots, taus, folds, np.random.default_rng(0)
    )

    assert scored["fold"].tolist() == [1]
    assert scored["n"].tolist() == [4]
    # Earlier fold: every PIT is 0.2, so R sends the whole grid to q_0.20 = 0.
    # Pinball of y=2 against [0, 0, 0] is 1. Fitting on the later fold would
    # score 0.5, because that fold's own PIT sits at the upper knot.
    np.testing.assert_allclose(scored["mean_pinball"], [1.0])


def test_walk_forward_map_ignores_the_rng_seed():
    # y=10 sits on a flat upper knot, so a random U moves the PIT on
    # [0.47, 0.80]. Pinball does not use the coverage jitter, so two
    # seeds must still fit the same map.
    taus = np.array([0.2, 0.5, 0.8])
    knots = np.tile(np.array([[0.0, 10.0, 10.0]]), (8, 1))
    y = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 5.0, 10.0])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    first = pit_walk_forward(y, knots, taus, folds, np.random.default_rng(0))
    second = pit_walk_forward(y, knots, taus, folds, np.random.default_rng(1))

    np.testing.assert_allclose(first["mean_pinball"], second["mean_pinball"])


def test_eval_mask_reuses_the_same_row_jitter():
    # Dropping rows from fold 1 must not move fold 2. A draw sized to
    # the scored rows consumes the generator, so the next fold's map
    # and its coverage jitter both change.
    taus = np.array([0.2, 0.5, 0.8])
    knots = np.tile(np.array([[0.0, 10.0, 10.0]]), (12, 1))
    y = np.array([0, 0, 10, 10, 0, 10, 0, 10, 0, 10, 5, 10], dtype=float)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    mask = np.ones(12, dtype=bool)
    mask[[4, 5]] = False

    full = pit_walk_forward(y, knots, taus, folds, np.random.default_rng(0))
    sliced = pit_walk_forward(
        y, knots, taus, folds, np.random.default_rng(0), eval_mask=mask
    )

    full_fold2 = full.loc[full["fold"] == 2].iloc[0]
    sliced_fold2 = sliced.loc[sliced["fold"] == 2].iloc[0]
    assert int(sliced_fold2["n"]) == 4
    np.testing.assert_allclose(sliced_fold2["mean_pinball"], full_fold2["mean_pinball"])
    np.testing.assert_allclose(sliced_fold2["max_abs_gap"], full_fold2["max_abs_gap"])


def test_scored_subset_uses_the_full_sample_jitter():
    taus = np.array([0.2, 0.5, 0.8])
    knots = np.tile(np.array([[0.0, 10.0, 10.0]]), (8, 1))
    y = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 5.0, 10.0])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    early = np.array([False, False, False, False, True, False, True, False])

    scored = pit_walk_forward(
        y, knots, taus, folds, np.random.default_rng(0), eval_mask=early
    )
    jitter = np.random.default_rng(0).uniform(-0.5, 0.5, y.shape[0])
    calibrator = fit_pit_recalibrator(
        deterministic_quantile_pit(y[folds == 0], knots[folds == 0], taus)
    )
    adjusted = calibrated_quantile_grid(knots[early], taus, calibrator)
    hits = (y[early] + jitter[early])[:, None] <= adjusted
    table = calibration_table(y[early], adjusted, taus, None, hits=hits)

    np.testing.assert_allclose(scored["n"], [2])
    np.testing.assert_allclose(scored["mean_pinball"], [table["pinball"].mean()])
    np.testing.assert_allclose(scored["max_abs_gap"], [line_band_gap(table)])


class _Column:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values.copy()


def test_saved_pit_map_sorts_then_recalibrates_and_clips():
    bundle = {
        "quantile_models": {
            "q_0.25": _Column([40.0]),
            "q_0.50": _Column([80.0]),
            "q_0.75": _Column([20.0]),
        },
        "feature_names": ["starting"],
        "calibration": {
            "method": "pit_cdf",
            "pit_x": np.array([0.25]),
            "pit_y": np.array([1.0]),
            "taus": np.array([0.25, 0.5, 0.75]),
            "clip_low": 0.0,
            "clip_high": 15.0,
            "target": "pts",
        },
    }

    out = predict_calibrated_quantiles(bundle, pd.DataFrame({"starting": [1.0]}))

    np.testing.assert_allclose(out.to_numpy(), [[15.0, 15.0, 15.0]])


def test_pit_payload_loads_without_a_group_table(tmp_path):
    frame = pd.DataFrame({"starting": np.arange(8, dtype=float)})
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        n_estimators=2,
        max_depth=1,
        random_state=0,
    )
    model.fit(frame, np.arange(8, dtype=float))
    train = pd.DataFrame(
        {"game_date": pd.to_datetime(["2024-04-01"]), "starting": [1.0]}
    )
    holdout = pd.DataFrame(
        {"game_date": pd.to_datetime(["2025-10-21"]), "starting": [1.0]}
    )
    path = save_model_bundle(
        {"q_0.50": model},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="pts_pit",
        calibration={
            "method": "pit_cdf",
            "pit_x": np.array([0.25, 1.0]),
            "pit_y": np.array([0.25, 1.0]),
            "taus": np.array([0.25, 0.75]),
            "clip_low": 0.0,
            "clip_high": 100.0,
            "target": "pts",
        },
        save_dir=tmp_path,
    )

    loaded = load_model_bundle(path)

    assert loaded["calibration"]["method"] == "pit_cdf"
    np.testing.assert_allclose(loaded["calibration"]["pit_x"], [0.25, 1.0])
    np.testing.assert_allclose(loaded["calibration"]["pit_y"], [0.25, 1.0])
