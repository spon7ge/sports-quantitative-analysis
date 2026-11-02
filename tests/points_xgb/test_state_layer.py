"""Residual-layer rules: earlier folds only, noisy games move the state less."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from points_xgb.config import QUANTILES
from points_xgb.state_layer import (
    FilterPath,
    ProcessNoise,
    adjust_quantiles,
    align_market,
    apply_layer,
    choose_sweep,
    assign_groups,
    attach_context,
    audit_predictions,
    filter_player,
    fit_group_prior,
    fit_process_variances,
    layer_decision,
    leave_one_player_mean,
    noise_scale,
    normal_scores,
    opening_quotes,
    p_over,
    p_over_state,
    pair_points_quotes,
    prediction_hash,
    prepare_residuals,
    probability_clv,
    residual_dependence,
    shrink_variance,
    signal_remains,
    unscored_played_games,
    walk_forward_quantiles,
    window_states,
)


def test_noise_scale_divides_the_80_percent_width_by_2_56():
    scale, n_floored = noise_scale(np.array([10.0, 0.0]), np.array([12.56, 0.0]))
    assert scale[0] == pytest.approx(1.0)
    assert scale[1] == pytest.approx(0.5)
    assert n_floored == 1


def test_audit_rejects_a_duplicated_player_game():
    frame = _predictions()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        audit_predictions(frame)


def test_audit_rejects_oof_rows_before_2022_23():
    frame = _predictions()
    frame.loc[0, "game_date"] = pd.Timestamp("2021-12-01")
    with pytest.raises(ValueError, match="2022-23"):
        audit_predictions(frame)


def test_walk_forward_offsets_ignore_the_fold_being_scored():
    taus = np.array([0.10, 0.50, 0.90])
    grid = np.tile(np.array([8.0, 10.0, 12.0]), (4, 1))
    windows = np.array([1, 1, 2, 2])
    early = walk_forward_quantiles(
        np.array([20.0, 20.0, 0.0, 0.0]), grid, taus, windows, seed=0
    )
    later = walk_forward_quantiles(
        np.array([20.0, 20.0, 40.0, 40.0]), grid, taus, windows, seed=0
    )
    np.testing.assert_allclose(early[2:], later[2:])
    np.testing.assert_allclose(early[:2], grid[:2])
    assert early[2, 1] > grid[2, 1]


def test_a_noisier_game_moves_the_state_less():
    quiet = _one_game(obs_sd=2.0)
    noisy = _one_game(obs_sd=10.0)
    assert quiet.post_mean[0] > noisy.post_mean[0]
    assert quiet.gain[0] > noisy.gain[0]


def test_the_current_game_does_not_enter_its_own_prediction():
    path = _one_game(obs_sd=2.0, residual=5.0, initial_mean=1.5)
    assert path.pred_mean[0] == pytest.approx(1.5)
    assert path.post_mean[0] != pytest.approx(1.5)


def test_offseason_adds_offseason_variance_once():
    path = filter_player(
        residual=np.array([0.0, 0.0]),
        obs_sd=np.array([1.0, 1.0]),
        season=np.array(["2023-24", "2024-25"]),
        team=np.array(["A", "A"]),
        rest_days=np.array([np.nan, 120.0]),
        per_game=np.array([0.01, 0.01]),
        offseason=9.0,
        initial=1.0,
        initial_mean=0.0,
        reset_on_new_team=True,
    )
    assert path.pred_var[1] == pytest.approx(0.5 + 9.0)


def test_a_new_team_resets_variance_up_to_the_prior():
    path = filter_player(
        residual=np.array([0.0, 0.0]),
        obs_sd=np.array([1.0, 1.0]),
        season=np.array(["2023-24", "2023-24"]),
        team=np.array(["A", "B"]),
        rest_days=np.array([np.nan, 2.0]),
        per_game=np.array([0.01, 0.01]),
        offseason=9.0,
        initial=4.0,
        initial_mean=0.0,
        reset_on_new_team=True,
    )
    assert path.pred_var[1] == pytest.approx(4.0)


def test_a_longer_rest_adds_more_within_season_variance():
    short = _rest_path(2.0)
    long = _rest_path(10.0)
    assert long.pred_var[1] > short.pred_var[1]
    assert short.pred_var[1] == pytest.approx(0.5 + 1.0)
    assert long.pred_var[1] == pytest.approx(0.5 + 5.0)


def test_static_prior_leaves_a_player_out_of_his_own_group_mean():
    residuals = np.array([0.0] * 40 + [8.0])
    players = np.array(["other"] * 40 + ["hot"])
    groups = np.array(["PG|bench|BOS"] * 41)
    coarse = np.array(["PG|bench"] * 41)
    weights = np.ones(41)
    mean = leave_one_player_mean(
        residuals, weights, players, groups, coarse, "hot", "PG|bench|BOS", "PG|bench", min_n=10
    )
    assert mean == pytest.approx(0.0)


def test_process_fit_ignores_rows_outside_the_mask():
    frame = _constant_frame()
    mask = frame["window_id"].eq(1)
    prior = fit_group_prior(frame.loc[mask], min_n=5)
    original = fit_process_variances(
        frame.loc[mask],
        prior,
        q_grid=(0.25, 4.0),
        q_off_grid=(1.0,),
        p0_grid=(4.0,),
        group_q_grid=None,
    )
    poisoned = frame.copy()
    poisoned.loc[poisoned["window_id"].eq(2), "residual"] = 50.0
    again = fit_process_variances(
        poisoned.loc[mask],
        prior,
        q_grid=(0.25, 4.0),
        q_off_grid=(1.0,),
        p0_grid=(4.0,),
        group_q_grid=None,
    )
    assert original.per_game == again.per_game
    assert original.offseason == again.offseason
    assert original.initial == again.initial


def test_a_random_walk_prefers_more_drift_than_white_noise():
    calm = fit_process_variances(
        _constant_frame().loc[lambda frame: frame["window_id"].eq(1)],
        fit_group_prior(_constant_frame().loc[lambda frame: frame["window_id"].eq(1)], min_n=5),
        q_grid=(0.25, 4.0),
        q_off_grid=(1.0,),
        p0_grid=(1.0,),
        group_q_grid=None,
    )
    walking = fit_process_variances(
        _walk_frame(),
        fit_group_prior(_walk_frame(), min_n=5),
        q_grid=(0.25, 4.0),
        q_off_grid=(1.0,),
        p0_grid=(1.0,),
        group_q_grid=None,
    )
    assert walking.per_game > calm.per_game


def test_group_drift_shrinks_toward_the_global_value():
    assert shrink_variance(4.0, 30, 1.0, prior_n=30) == pytest.approx(2.5)


def test_integer_line_matches_the_half_point_above_it():
    knots = np.array([[10.0, 20.0, 30.0]])
    taus = np.array([0.10, 0.50, 0.90])
    assert p_over(np.array([20.0]), knots, taus) == pytest.approx(
        p_over(np.array([20.5]), knots, taus)
    )


def test_persistent_residuals_have_dependence_and_noise_does_not():
    persistent = residual_dependence(_player_effects(offset=3.0))
    assert persistent["acf_1"] > 0.5
    assert persistent["icc"] > 0.5
    noise = residual_dependence(_player_effects(offset=0.0))
    assert abs(noise["acf_1"]) < 0.05
    assert noise["icc"] < 0.05


def test_gate_stops_only_when_both_summaries_are_near_zero():
    assert signal_remains(0.01, 0.01) is False
    assert signal_remains(0.05, 0.0) is True
    assert signal_remains(0.0, 0.05) is True
    assert signal_remains(float("nan"), 0.2) is False


def test_normal_score_is_zero_at_the_median_and_uses_the_tail_past_q95():
    from scipy.stats import norm

    knots = np.tile(np.linspace(10.0, 30.0, len(QUANTILES)), (1, 1))
    median = float(knots[0, QUANTILES.index(0.5)])
    q95 = float(knots[0, -1])
    tails = _tails_with_upper_excess(q95)
    pit, z = normal_scores(
        np.array([median, q95 + 20.0]),
        knots.repeat(2, axis=0),
        np.array([0.0, 0.0]),
        tails,
        np.array([0, 0]),
    )
    assert pit[0] == pytest.approx(0.5)
    assert z[0] == pytest.approx(0.0)
    assert pit[1] > 0.95
    assert z[1] > norm.ppf(0.95)
    assert z[1] > 0


def test_a_later_window_does_not_build_its_tail_from_its_own_misses():
    frame = _scored_frame()
    frame.loc[frame["window_id"].eq(2), "pts"] = 80.0
    _out, info = prepare_residuals(frame, seed=0)
    later = info["tails"][2]
    earlier = fit_reference_tail(frame.loc[frame["window_id"].eq(1)])
    np.testing.assert_allclose(later.arrays[("upper", 0)], earlier.arrays[("upper", 0)])


def test_season_boundary_shrinks_the_mean_toward_the_group():
    path = filter_player(
        residual=np.array([4.0, 0.0]),
        obs_sd=np.array([1.0, 1.0]),
        season=np.array(["2023-24", "2024-25"]),
        team=np.array(["A", "A"]),
        rest_days=np.array([np.nan, 120.0]),
        per_game=np.array([0.0, 0.0]),
        offseason=0.0,
        initial=1.0,
        initial_mean=4.0,
        reset_on_new_team=False,
        group_mean=np.array([0.0, 0.0]),
        group_tau2=np.array([1.0, 1.0]),
    )
    assert path.post_mean[0] == pytest.approx(4.0)
    assert path.pred_mean[1] == pytest.approx(8.0 / 3.0)
    assert path.pred_var[1] == pytest.approx(0.5)


def test_zero_state_probability_matches_the_calibrated_distribution():
    knots = np.tile(np.linspace(10.0, 30.0, len(QUANTILES)), (1, 1))
    taus = np.asarray(QUANTILES, dtype=float)
    tails = _tails_with_upper_excess(float(knots[0, -1]))
    base = p_over(np.array([20.5]), knots, taus)
    same = p_over_state(
        np.array([20.5]),
        knots,
        taus,
        state_mean=np.array([0.0]),
        state_var=np.array([0.0]),
        obs_var=1.0,
        tails=tails,
        groups=np.array([0]),
    )
    higher = p_over_state(
        np.array([20.5]),
        knots,
        taus,
        state_mean=np.array([0.8]),
        state_var=np.array([0.0]),
        obs_var=1.0,
        tails=tails,
        groups=np.array([0]),
    )
    assert same[0] == pytest.approx(base[0])
    assert higher[0] > same[0]


def test_zero_state_leaves_the_quantile_grid_unchanged():
    knots = np.tile(np.linspace(10.0, 30.0, len(QUANTILES)), (2, 1))
    taus = np.asarray(QUANTILES, dtype=float)
    tails = {1: _tails_with_upper_excess(30.0)}
    adjusted = adjust_quantiles(
        knots,
        taus,
        state_mean=np.zeros(2),
        state_var=np.zeros(2),
        obs_sd=np.ones(2),
        tails=tails,
        windows=np.array([1, 1]),
        groups=np.array([0, 0]),
    )
    np.testing.assert_allclose(adjusted, knots)


def test_zero_drift_is_the_static_hierarchical_filter():
    frame = _constant_frame().copy()
    frame["game_id"] = np.arange(len(frame)).astype(str)
    prior = fit_group_prior(frame, min_n=5)
    process = ProcessNoise(
        per_game=1.0,
        offseason=4.0,
        initial=2.0,
        loglik=0.0,
        group_per_game={"PG|rookie=0|new=0": 1.0},
        obs_var=1.0,
    )
    static = apply_layer(frame, prior, process, dynamic=False)
    zero = apply_layer(frame, prior, process.scaled(0.0), dynamic=True)
    np.testing.assert_allclose(zero["state_mean"], static["state_mean"])
    np.testing.assert_allclose(zero["state_var"], static["state_var"])
    drifting = apply_layer(frame, prior, process, dynamic=True)
    assert not np.allclose(drifting["state_var"], zero["state_var"])


def test_sweep_keeps_a_small_drift_only_when_it_beats_the_static_model():
    hierarchical = choose_sweep(
        _sweep(
            (1.0, -0.01, 0.02),
            (0.25, -0.04, -0.01),
            (0.0, -0.03, -0.01),
            best_vs_c=(0.25, -0.02, 0.01),
        )
    )
    assert hierarchical == (
        "hierarchical",
        "zero-drift hierarchical model beats the calibrated quantiles; the interval sits below zero",
    )
    dynamic = choose_sweep(
        _sweep(
            (1.0, -0.02, 0.01),
            (0.25, -0.05, -0.02),
            (0.0, -0.01, 0.01),
            best_vs_c=(0.25, -0.04, -0.01),
        )
    )
    assert dynamic[0] == "dynamic"
    dropped = choose_sweep(
        _sweep(
            (1.0, 0.02, 0.04),
            (0.25, 0.01, 0.03),
            (0.0, -0.01, 0.02),
            best_vs_c=(0.25, -0.03, -0.01),
        )
    )
    assert dropped[0] == "drop"


def test_groups_are_position_rookie_and_new_team():
    frame = pd.DataFrame(
        {
            "pos": ["PG", "C"],
            "rookie": [True, False],
            "new_team": [False, True],
            "starting": [1, 0],
        }
    )
    grouped = assign_groups(frame)
    assert grouped.loc[0, "coarse_group"] == "PG"
    assert grouped.loc[0, "fine_group"] == "PG|rookie=1|new=0"
    assert grouped.loc[1, "fine_group"] == "C|rookie=0|new=1"


def test_unscored_played_games_are_counted_and_not_returned_as_rows():
    played = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": ["1", "1"],
            "game_date": pd.to_datetime(["2023-11-01", "2023-11-03"]),
            "minutes": [30.0, 12.0],
        }
    )
    counted = unscored_played_games(_predictions().iloc[[0]], played)
    assert counted["unscored"] == 1
    assert counted["scored"] == 1


def test_context_join_raises_when_a_prediction_has_no_gamelog():
    context = pd.DataFrame(
        {
            "game_id": ["g1"],
            "player_id": ["1"],
            "season_year": ["2023-24"],
            "team_id": ["10"],
            "pos": ["PG"],
            "player_name": ["A"],
        }
    )
    with pytest.raises(ValueError, match="unmatched"):
        attach_context(_predictions(), context)


def test_priced_quotes_keep_one_points_pair_and_drop_other_stats():
    quotes = pair_points_quotes(
        pd.DataFrame(
            {
                "book": ["pinnacle"] * 4,
                "slate_date": pd.to_datetime(["2026-10-22"] * 4),
                "player_name": ["A", "A", "A", "A"],
                "line": [20.5, 20.5, 15.5, 15.5],
                "side": ["over", "under", "over", "under"],
                "american": [-110, -110, 100, -140],
                "stat_type": ["points", "points", "assists", "assists"],
            }
        )
    )
    assert len(quotes) == 1
    assert quotes.iloc[0]["line"] == pytest.approx(20.5)
    assert quotes.iloc[0]["over_american"] == -110


def test_clv_is_missing_when_the_line_moved():
    clv = probability_clv(
        open_over=np.array([0.45, 0.40]),
        close_over=np.array([0.55, 0.60]),
        take_over=np.array([True, True]),
        line_unchanged=np.array([True, False]),
    )
    assert clv[0] == pytest.approx(0.10)
    assert np.isnan(clv[1])


def test_opening_quote_is_the_earliest_snapshot_that_lists_the_player():
    early = _snapshot(
        datetime(2026, 10, 22, 18, tzinfo=UTC),
        ("A", "points", 10.5, "over", -120),
        ("A", "points", 10.5, "under", 100),
    )
    late = _snapshot(
        datetime(2026, 10, 22, 22, tzinfo=UTC),
        ("A", "points", 12.5, "over", -110),
        ("A", "points", 12.5, "under", -110),
    )
    quotes = opening_quotes([late, early], book="pinnacle")
    assert len(quotes) == 1
    assert quotes.iloc[0]["line"] == pytest.approx(10.5)


def test_keep_needs_a_real_pinball_gain_and_a_market_gain():
    keep = dict(
        pinball_low=-0.02,
        pinball_high=-0.01,
        market_n=500,
        market_low=-0.02,
        market_high=-0.01,
        shuffle_delta=-0.001,
        real_delta=-0.015,
        narrow=False,
    )
    assert layer_decision(**keep) == ("keep", "pinball and closing-line log loss both improve")
    assert layer_decision(**{**keep, "pinball_high": 0.01})[0] == "drop"
    assert layer_decision(**{**keep, "shuffle_delta": -0.02})[0] == "drop"
    assert layer_decision(**{**keep, "narrow": True})[0] == "drop"
    assert layer_decision(**{**keep, "market_n": 10})[0] == "drop"


def test_later_window_residuals_do_not_change_an_earlier_state():
    frame = _constant_frame().copy()
    frame["game_id"] = np.arange(len(frame)).astype(str)
    extra = frame.loc[frame["window_id"].eq(2)].copy()
    extra["window_id"] = 3
    extra["season_year"] = "2024-25"
    extra["game_date"] = extra["game_date"] + pd.Timedelta(days=400)
    extra["game_id"] = extra["game_id"] + "x"
    frame = pd.concat([frame, extra], ignore_index=True)
    grid = dict(q_grid=(0.25,), q_off_grid=(1.0,), p0_grid=(1.0,), group_q_grid=None)
    original = window_states(frame, (2,), dynamic=True, **grid)
    poisoned = frame.copy()
    poisoned.loc[poisoned["window_id"].eq(3), "residual"] = 25.0
    again = window_states(poisoned, (2,), dynamic=True, **grid)
    np.testing.assert_allclose(original["state_mean"], again["state_mean"])
    np.testing.assert_allclose(original["state_var"], again["state_var"])


def test_prediction_hash_changes_when_the_median_changes():
    frame = _predictions()
    other = frame.copy()
    other["q_0.50"] = other["q_0.50"] + 1
    assert prediction_hash(frame) != prediction_hash(other)


def _one_game(obs_sd: float, residual: float = 5.0, initial_mean: float = 0.0) -> FilterPath:
    return filter_player(
        residual=np.array([residual]),
        obs_sd=np.array([obs_sd]),
        season=np.array(["2023-24"]),
        team=np.array(["A"]),
        rest_days=np.array([np.nan]),
        per_game=np.array([0.0]),
        offseason=1.0,
        initial=4.0,
        initial_mean=initial_mean,
        reset_on_new_team=False,
    )


def _rest_path(rest_days: float) -> FilterPath:
    return filter_player(
        residual=np.array([0.0, 0.0]),
        obs_sd=np.array([1.0, 1.0]),
        season=np.array(["2023-24", "2023-24"]),
        team=np.array(["A", "A"]),
        rest_days=np.array([np.nan, rest_days]),
        per_game=np.array([1.0, 1.0]),
        offseason=0.0,
        initial=1.0,
        initial_mean=0.0,
        reset_on_new_team=False,
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": ["1", "2"],
            "game_date": pd.to_datetime(["2023-11-01", "2026-10-22"]),
            "window_id": [2, -1],
            "is_holdout": [False, True],
            "pts": [10, 12],
            "q_0.10": [6.0, 7.0],
            "q_0.50": [10.0, 11.0],
            "q_0.90": [14.0, 15.0],
        }
    )


def _constant_frame() -> pd.DataFrame:
    rows = []
    for window, season, day in ((1, "2022-23", 1), (2, "2023-24", 1)):
        for player in range(6):
            for game in range(8):
                rows.append(
                    {
                        "player_id": str(player),
                        "game_date": pd.Timestamp(f"2023-01-{day:02d}")
                        + pd.Timedelta(days=game + window),
                        "season_year": season,
                        "team_id": "BOS",
                        "coarse_group": "PG|bench",
                        "fine_group": "PG|bench|BOS",
                        "residual": 0.0,
                        "obs_sd": 2.0,
                        "rest_days": np.nan if game == 0 else 2.0,
                        "window_id": window,
                    }
                )
    return pd.DataFrame(rows)


def _walk_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    level = 0.0
    for game in range(40):
        level += float(rng.normal(0, 2.0))
        rows.append(
            {
                "player_id": "1",
                "game_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=game),
                "season_year": "2022-23",
                "team_id": "BOS",
                "coarse_group": "PG|bench",
                "fine_group": "PG|bench|BOS",
                "residual": level,
                "obs_sd": 1.0,
                "rest_days": np.nan if game == 0 else 1.0,
                "window_id": 1,
            }
        )
    return pd.DataFrame(rows)


def _player_effects(offset: float) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for player in range(80):
        level = offset if player % 2 == 0 else -offset
        for game in range(20):
            rows.append(
                {
                    "player_id": player,
                    "z": level + float(rng.normal(0, 0.2 if offset else 1.0)),
                    "season_boundary": game == 0 and False,
                    "early": game < 2,
                    "tier": "10-20",
                }
            )
    return pd.DataFrame(rows)


def test_align_market_keeps_the_quote_name():
    quotes = pd.DataFrame(
        {
            "book": ["pinnacle"],
            "slate_date": [pd.Timestamp("2026-10-22")],
            "player_name": ["John Smith"],
            "line": [12.5],
            "over_american": [-110],
            "under_american": [-110],
            "novig_over": [0.5],
        }
    )
    games = pd.DataFrame(
        {
            "player_id": ["7"],
            "player_name": ["John Smith"],
            "game_id": ["g1"],
            "game_date": [pd.Timestamp("2026-10-22")],
            "pts": [14],
        }
    )
    matched, counts = align_market(quotes, games)
    assert counts["matched"] == 1
    assert matched.loc[0, "player_name"] == "John Smith"
    assert "player_name_x" not in matched.columns


def _sweep(*rows, best_vs_c):
    records = []
    for scale, delta_b, ci_high_b in rows:
        delta_c, ci_high_c = (float("nan"), float("nan"))
        if scale == best_vs_c[0]:
            delta_c, ci_high_c = best_vs_c[1], best_vs_c[2]
        records.append(
            {
                "scale": scale,
                "delta_b": delta_b,
                "ci_high_b": ci_high_b,
                "delta_c": delta_c,
                "ci_high_c": ci_high_c,
            }
        )
    return pd.DataFrame(records)


def _tails_with_upper_excess(q95: float):
    from points_xgb.state_layer import PointTails

    del q95
    return PointTails(
        arrays={
            ("lower", 0): np.array([1.0]),
            ("upper", 0): np.array([0.0, 10.0, 20.0, 40.0]),
        }
    )


def _knot_row() -> np.ndarray:
    return np.linspace(10.0, 30.0, len(QUANTILES))


def _scored_frame() -> pd.DataFrame:
    knots = _knot_row()
    rows = []
    for window, count in ((1, 4), (2, 2)):
        for game in range(count):
            row = {
                "player_id": "1",
                "game_id": f"{window}-{game}",
                "game_date": pd.Timestamp("2023-01-01")
                + pd.Timedelta(days=int(window) * 30 + game),
                "season_year": "2022-23" if window == 1 else "2023-24",
                "window_id": window,
                "pts": float(knots[QUANTILES.index(0.5)]),
                "starting": 0,
            }
            for level, knot in zip(QUANTILES, knots, strict=True):
                row[f"b_q_{level:.2f}"] = float(knot)
            rows.append(row)
    return pd.DataFrame(rows)


def fit_reference_tail(frame: pd.DataFrame):
    from points_xgb.state_layer import fit_point_tails

    knots = np.column_stack(
        [frame[f"b_q_{level:.2f}"].to_numpy(dtype=float) for level in QUANTILES]
    )
    return fit_point_tails(
        frame["pts"].to_numpy(dtype=float),
        knots,
        frame["starting"].to_numpy(dtype=int),
    )


def _snapshot(fetched_at, *lines):
    listed = tuple(
        SimpleNamespace(
            player=player,
            stat_type=stat,
            line_score=line,
            side=side,
            american_price=price,
        )
        for player, stat, line, side, price in lines
    )
    return SimpleNamespace(fetched_at=fetched_at, lines=listed)
