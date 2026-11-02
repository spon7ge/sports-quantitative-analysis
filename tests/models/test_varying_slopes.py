"""Player-level slopes stay off unless a toggle says otherwise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytensor.graph.traversal import ancestors
from src.models.hierarchical_points import (
    VARYING_SLOPE_NAMES,
    VARYING_SLOPE_SD,
    SlopeConfig,
    build_design,
    build_model,
    default_slope_config,
    expanded_covariates,
    shared_slope_config,
    transform_design,
)
from src.models.slope_comparison import (
    holdout_elpd,
    last_games_mask,
    modeled_holdout_mask,
    plot_shrinkage,
    pointwise_elpd,
    recommend_slope,
    sampling_diagnostics,
    sd_summary,
    slope_stems,
)
from tests.models.test_hierarchical_points import _context_games, _games


def _names(variable) -> set[str]:
    return {node.name for node in ancestors([variable]) if getattr(node, "name", None)}


def _call(fn, point: dict[str, np.ndarray]) -> np.ndarray:
    needed = [
        inp.variable.name
        for inp, container in zip(fn.f.maker.inputs, fn.f.input_storage)
        if not container.implicit
    ]
    return np.asarray(fn({name: point[name] for name in needed}), dtype=float)


def _model(design, slopes: SlopeConfig | None = None, **kwargs):
    return build_model(
        design.data,
        design.n_players,
        design.n_teams,
        design.n_pos,
        design.n_blocks,
        minutes_columns=design.minutes_columns,
        rate_columns=design.rate_columns,
        slopes=slopes,
        **kwargs,
    )


def test_default_rollout_varies_only_usage_without_the_star() -> None:
    config = default_slope_config()
    assert config.enabled("usg_stars_out")
    assert not any(
        config.enabled(name) for name in VARYING_SLOPE_NAMES if name != "usg_stars_out"
    )
    assert config.use_role_tier is False
    assert config.varying_sd == pytest.approx(VARYING_SLOPE_SD)


def test_omitting_the_config_keeps_the_shared_coefficient_vector() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    plain = build_model(
        design.data,
        design.n_players,
        design.n_teams,
        design.n_pos,
        design.n_blocks,
    )
    shared = _model(design, shared_slope_config())
    assert set(plain.named_vars) == set(shared.named_vars)
    assert "spread_z" not in plain.named_vars
    width = design.data["X_min"].shape[1]
    assert plain.initial_point()["b_min"].shape == (width,)
    assert shared.initial_point()["b_min"].shape == (width,)


def test_a_false_toggle_does_not_add_that_slope() -> None:
    pytest.importorskip("pymc")
    design = build_design(_context_games(), covariates=expanded_covariates())
    config = SlopeConfig(
        varying={
            "usg_stars_out": True,
            "usg_stars_in": False,
            "stars_out": False,
            "spread": False,
        }
    )
    model = _model(design, config)
    assert "fga2_usg_stars_out" in model.named_vars
    assert "fga2_usg_stars_in" not in model.named_vars
    assert "stars_out_z" not in model.named_vars
    assert "spread_z" not in model.named_vars
    assert "opp_def_prior_z" not in model.named_vars
    assert "points_trend_z" not in model.named_vars
    assert "fga2_sd_drift" in model.named_vars
    rate_width = design.data["X_rate"].shape[1]
    assert model.initial_point()["b_fga2"].shape == (rate_width - 1,)
    assert np.asarray(model.initial_point()["fga2_sd_drift_log__"]).shape == ()


def test_spread_slope_is_noncentered_and_moves_only_that_player() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    config = SlopeConfig(
        varying={name: name == "spread" for name in VARYING_SLOPE_NAMES}
    )
    model = _model(design, config)
    assert float(model["spread_sd"].owner.inputs[3].data) == pytest.approx(
        VARYING_SLOPE_SD
    )
    assert float(model["spread_mu"].owner.inputs[2].data) == pytest.approx(0.0)
    assert float(model["spread_mu"].owner.inputs[3].data) == pytest.approx(3.0)
    assert model.initial_point()["b_min"].shape == (design.data["X_min"].shape[1] - 1,)
    assert "spread" in _names(model["minutes"])
    # Attempts are downstream of sampled minutes, so this slope reaches fga2
    # through minutes. It does not become an attempt coefficient.
    assert "fga2_spread" not in model.named_vars
    assert model.initial_point()["b_fga2"].shape == (design.data["X_rate"].shape[1],)
    assert "spread_tier_sd" not in model.named_vars

    minutes_mean = model.compile_fn(
        model.replace_rvs_by_values([model["minutes"].owner.inputs[2]])[0]
    )
    beta_fn = model.compile_fn(model.replace_rvs_by_values([model["spread"]])[0])
    point = model.initial_point()
    point["spread_mu"] = np.array(0.0)
    point["spread_z"] = np.zeros(design.n_players)
    baseline = _call(minutes_mean, point)
    base_beta = _call(beta_fn, point)
    player = int(design.data["player"][0])
    point["spread_z"] = np.array(point["spread_z"], copy=True)
    point["spread_z"][player] = 1.0
    shifted = _call(minutes_mean, point)
    high_beta = _call(beta_fn, point)
    delta = float(high_beta[player] - base_beta[player])
    assert delta > 0
    others = np.ones(design.n_players, dtype=bool)
    others[player] = False
    np.testing.assert_allclose(high_beta[others], base_beta[others])
    column = design.minutes_columns.index("spread")
    touched = design.data["player"] == player
    expected = delta * design.data["X_min"][touched, column]
    np.testing.assert_allclose(shifted[touched] - baseline[touched], expected)
    np.testing.assert_allclose(shifted[~touched], baseline[~touched])


def test_usage_slope_changes_only_that_attempt_equation() -> None:
    pytest.importorskip("pymc")
    design = build_design(_context_games(), covariates=expanded_covariates())
    config = default_slope_config()
    model = _model(design, config)
    assert slope_stems(config, design.minutes_columns, design.rate_columns) == [
        "fga2_usg_stars_out",
        "fga3_usg_stars_out",
        "fta_usg_stars_out",
    ]
    assert "fga2_usg_stars_out" in _names(model["fga2"])
    assert "fga2_usg_stars_out" not in _names(model["minutes"])
    assert float(model["fga2_usg_stars_out_mu"].owner.inputs[3].data) == pytest.approx(
        0.2
    )
    # NegativeBinomial stores p = sigmoid(log alpha - log mu) at this input.
    # A change in log rate moves logit(p) by the opposite amount.
    rate_p = model.compile_fn(
        model.replace_rvs_by_values([model["fga2"].owner.inputs[3]])[0]
    )
    beta_fn = model.compile_fn(
        model.replace_rvs_by_values([model["fga2_usg_stars_out"]])[0]
    )
    point = model.initial_point()
    for stat in ("fga2", "fga3", "fta"):
        point[f"{stat}_usg_stars_out_mu"] = np.array(0.0)
        point[f"{stat}_usg_stars_out_z"] = np.zeros(design.n_players)
    baseline = _call(rate_p, point)
    base_beta = _call(beta_fn, point)
    player = int(design.data["player"][0])
    point["fga2_usg_stars_out_z"] = np.array(point["fga2_usg_stars_out_z"], copy=True)
    point["fga2_usg_stars_out_z"][player] = 1.0
    shifted = _call(rate_p, point)
    high_beta = _call(beta_fn, point)
    delta = float(high_beta[player] - base_beta[player])
    column = design.rate_columns.index("usg_stars_out")
    touched = design.data["player"] == player
    logit_shift = np.log(shifted / (1.0 - shifted)) - np.log(
        baseline / (1.0 - baseline)
    )
    np.testing.assert_allclose(
        logit_shift[touched],
        -delta * design.data["X_rate"][touched, column],
    )
    np.testing.assert_allclose(shifted[~touched], baseline[~touched])


def test_role_tier_adds_a_second_level_and_stays_off_by_default() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    off = _model(design, SlopeConfig(varying={"spread": True}))
    assert "role" not in off.coords
    config = SlopeConfig(varying={"spread": True}, use_role_tier=True)
    model = _model(design, config)
    assert list(model.coords["role"]) == ["bench", "rotation", "starter"]
    assert "spread_sd" not in model.named_vars
    assert "spread_player_sd" in model.named_vars
    assert "spread_tier_sd" in model.named_vars
    assert float(model["spread_tier_sd"].owner.inputs[3].data) == pytest.approx(0.05)
    beta_fn = model.compile_fn(model.replace_rvs_by_values([model["spread"]])[0])
    point = model.initial_point()
    point["spread_mu"] = np.array(0.0)
    point["spread_z"] = np.zeros(design.n_players)
    point["spread_tier_z"] = np.zeros(3)
    baseline = _call(beta_fn, point)
    point["spread_tier_z"] = np.array([0.0, 0.0, 1.0])
    shifted = _call(beta_fn, point)
    # Player 1 averages 30 minutes (starter). Player 3 averages 21 (rotation).
    starter = list(design.player_ids).index(1)
    rotation = list(design.player_ids).index(3)
    assert shifted[starter] - baseline[starter] > 0
    np.testing.assert_allclose(shifted[rotation], baseline[rotation])


def test_unknown_or_absent_varying_slopes_are_rejected() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    with pytest.raises(ValueError, match="unknown varying"):
        _model(design, SlopeConfig(varying={"not_a_slope": True}))
    with pytest.raises(ValueError, match="usg_stars_out"):
        _model(design, default_slope_config())
    with pytest.raises(ValueError, match="minutes_columns"):
        build_model(
            design.data,
            design.n_players,
            design.n_teams,
            design.n_pos,
            design.n_blocks,
            slopes=default_slope_config(),
        )
    with pytest.raises(ValueError, match="varying_sd"):
        _model(design, SlopeConfig(varying={"spread": True}, varying_sd=0))


def test_recommendation_requires_a_better_score_and_a_nonzero_sd() -> None:
    away = np.array([0.02, 0.03, 0.04, 0.05])
    collapsed = np.array([0.001, 0.002, 0.004, 0.008])
    assert recommend_slope(-10.0, -9.0, away) == "vary"
    assert recommend_slope(-10.0, -9.0, collapsed) == "share"
    assert recommend_slope(-9.0, -10.0, away) == "share"
    summary = sd_summary(away)
    assert summary["q05"] > 0.01
    assert summary["fraction_below_near_zero"] == 0
    assert pointwise_elpd(np.array([[1.0, 3.0], [3.0, 5.0]])) == pytest.approx(6.0)


def test_last_games_mask_keeps_a_player_with_too_few_games_in_train() -> None:
    players = np.array([1, 1, 1, 2])
    dates = pd.to_datetime(["2024-10-01", "2024-10-03", "2024-10-05", "2024-10-02"])
    games = np.array(["a", "b", "c", "d"])
    mask = last_games_mask(players, dates, games, n_games=1)
    assert mask.tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="n_games"):
        last_games_mask(players, dates, games, n_games=0)


def test_holdout_skips_a_trailing_game_with_no_market() -> None:
    frame = _games().reset_index(drop=True)
    latest = frame.loc[frame["player_id"].eq(1)].iloc[[-1]].copy()
    latest["game_date"] = pd.Timestamp("2024-10-06")
    latest["game_id"] = "missing-market"
    latest["game_total"] = np.nan
    frame = pd.concat([frame, latest], ignore_index=True)
    raw = last_games_mask(frame["player_id"], frame["game_date"], frame["game_id"], 1)
    assert raw[frame["game_id"].eq("missing-market").to_numpy()].all()
    mask, audit = modeled_holdout_mask(frame, 1)
    assert audit["unscored_trailing_rows"] == 1
    assert audit["missing_market_trailing_rows"] == 1
    held = frame.loc[mask]
    player_holdout = held.loc[held["player_id"].eq(1), "game_date"]
    assert list(player_holdout) == [pd.Timestamp("2024-10-04")]
    assert "missing-market" not in set(frame.loc[mask, "game_id"])
    design = build_design(frame.loc[~mask].copy())
    rows, scored = transform_design(
        frame, design, keep=pd.Series(mask, index=frame.index)
    )
    assert scored["scoring_rows"] == int(mask.sum())
    assert rows["minutes"].shape[0] == int(mask.sum())


def test_holdout_season_leaves_earlier_seasons_in_training() -> None:
    history = _games()
    history["season_year"] = "2024-25"
    current = history.loc[history["player_id"].eq(1)].copy()
    current["season_year"] = "2026-27"
    current["game_date"] = pd.to_datetime(current["game_date"]) + pd.Timedelta(days=400)
    current["game_id"] = current["game_id"].astype(str) + "-now"
    frame = pd.concat([history, current], ignore_index=True)
    mask, _audit = modeled_holdout_mask(frame, 1, season="2026-27")
    held = frame.loc[mask]
    assert set(held["season_year"]) == {"2026-27"}
    assert held["player_id"].nunique() == 1
    assert (
        held["game_date"].max()
        == frame.loc[frame["season_year"].eq("2026-27"), "game_date"].max()
    )


def test_shrinkage_plot_writes_a_file(tmp_path: Path) -> None:
    draws = np.array([[0.0, 1.0], [0.2, 1.4], [0.1, 0.8]])
    path = tmp_path / "shrinkage.png"
    plot_shrinkage(draws, np.array([0.4, 0.5, 0.6]), path, title="spread")
    assert path.stat().st_size > 0


def test_sampling_diagnostics_report_rhat_ess_and_divergences() -> None:
    pytest.importorskip("arviz")
    import arviz as az

    rng = np.random.default_rng(0)
    idata = az.from_dict(
        {
            "posterior": {
                "spread_sd": rng.uniform(0.02, 0.06, size=(2, 40)),
                "spread_z": rng.normal(size=(2, 40, 2)),
            },
            "sample_stats": {"diverging": np.zeros((2, 40), dtype=bool)},
        }
    )
    idata.sample_stats["diverging"][0, 0] = True
    table = sampling_diagnostics(idata, ["spread_sd", "spread_z"])
    assert set(table["parameter"]).issuperset({"spread_sd"})
    assert int(table["divergences"].iloc[0]) == 1
    assert np.isfinite(table.loc[table["parameter"].eq("spread_sd"), "r_hat"]).all()
    assert np.isfinite(table.loc[table["parameter"].eq("spread_sd"), "ess_bulk"]).all()


def test_holdout_elpd_scores_the_last_game_per_player() -> None:
    pytest.importorskip("pymc")
    import pymc as pm

    frame = _games().reset_index(drop=True)
    mask = last_games_mask(frame["player_id"], frame["game_date"], frame["game_id"], 1)
    design = build_design(frame.loc[~mask].copy(), block_days=2)
    model = build_model(
        design.data,
        design.n_players,
        design.n_teams,
        design.n_pos,
        design.n_blocks,
    )
    with model:
        idata = pm.sample(
            draws=4,
            tune=4,
            chains=1,
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
    rows, _audit = transform_design(
        frame, design, keep=pd.Series(mask, index=frame.index)
    )
    score = holdout_elpd(model, idata, rows)
    assert rows["minutes"].shape[0] == 2
    assert np.isfinite(score)
