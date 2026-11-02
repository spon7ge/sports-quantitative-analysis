"""Hierarchical points model: scoring rules, causal features, and the shot chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.evaluation import crps
from src.models.hierarchical_points import (
    MINUTES_UPPER,
    CovariateSpec,
    HierarchicalDesign,
    build_design,
    build_model,
    crps_ensemble,
    expanded_covariates,
    prob_over,
    simulate_points,
    transform_design,
)


def test_crps_ensemble_matches_the_sorted_draw_identity() -> None:
    draws = np.array([1.0, 3.0, 2.0])
    # E|X-y| = 2/3. Half the pairwise gap is 4/9. CRPS = 2/9.
    assert crps_ensemble(draws, 2.0) == pytest.approx(2.0 / 9.0)
    assert crps_ensemble(draws, 2.0) == pytest.approx(
        crps(draws.reshape(1, -1), np.array([2.0]))
    )


def test_crps_ensemble_rejects_empty_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        crps_ensemble(np.array([1.0, np.nan]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        crps_ensemble(np.array([1.0, 2.0]), np.inf)
    with pytest.raises(ValueError, match="finite"):
        crps_ensemble(np.array([]), 1.0)


def test_prob_over_leaves_pushes_out_of_both_sides() -> None:
    draws = np.array([[10.0, 10.0, 11.0], [9.0, 10.0, 12.0]])
    over, under = prob_over(draws, 10)
    np.testing.assert_allclose(over, [1.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(under, [0.0, 1.0 / 3.0])


def test_half_point_line_partitions_every_draw() -> None:
    draws = np.array([[10.0, 11.0, 12.0]])
    over, under = prob_over(draws, np.array([10.5]))
    assert over.item() + under.item() == pytest.approx(1.0)
    assert over.item() == pytest.approx(2.0 / 3.0)


def test_prob_over_rejects_a_mismatched_line() -> None:
    with pytest.raises(ValueError, match="line"):
        prob_over(np.ones((2, 4)), np.array([1.0]))


def _box(
    *,
    day: str,
    player_id: int,
    team_id: int,
    opp_team_id: int,
    minutes: float,
    fga: int,
    fg3_a: int,
    fta: int,
    fgm: int,
    fg3_m: int,
    ftm: int,
    spread: float,
    total: float,
    pace: float,
    home: bool,
    start: str,
    pos: str | None = "PG",
    season_year: str = "2024-25",
    game_id: str | None = None,
    usg: float | None = None,
    defense: float | None = None,
) -> dict[str, object]:
    points = 2 * (fgm - fg3_m) + 3 * fg3_m + ftm
    matchup = (
        f"T{team_id} vs. T{opp_team_id}" if home else f"T{team_id} @ T{opp_team_id}"
    )
    row: dict[str, object] = {
        "season_year": season_year,
        "game_date": pd.Timestamp(day),
        "game_id": game_id or f"{day}-{team_id}",
        "player_id": player_id,
        "team_id": team_id,
        "opp_team_id": opp_team_id,
        "pos": pos,
        "matchup": matchup,
        "minutes": minutes,
        "fga": fga,
        "fgm": fgm,
        "fg3_a": fg3_a,
        "fg3_m": fg3_m,
        "fta": fta,
        "ftm": ftm,
        "pts": points,
        "start_position": start,
        "game_total": total,
        "player_team_spread": spread,
        "team_pace": pace,
    }
    if usg is not None:
        row["usg_pct"] = usg
    if defense is not None:
        row["team_def_rating"] = defense
    return row


def _games() -> pd.DataFrame:
    rows = [
        _box(
            day="2024-10-01",
            player_id=1,
            team_id=10,
            opp_team_id=20,
            minutes=30,
            fga=10,
            fg3_a=4,
            fta=4,
            fgm=5,
            fg3_m=2,
            ftm=3,
            spread=-3.5,
            total=220,
            pace=100,
            home=True,
            start="G",
            pos="PG",
        ),
        _box(
            day="2024-10-01",
            player_id=2,
            team_id=20,
            opp_team_id=10,
            minutes=0,
            fga=0,
            fg3_a=0,
            fta=0,
            fgm=0,
            fg3_m=0,
            ftm=0,
            spread=3.5,
            total=220,
            pace=98,
            home=False,
            start="",
            pos="C",
        ),
        _box(
            day="2024-10-01",
            player_id=3,
            team_id=20,
            opp_team_id=10,
            minutes=20,
            fga=8,
            fg3_a=1,
            fta=2,
            fgm=4,
            fg3_m=1,
            ftm=2,
            spread=3.5,
            total=220,
            pace=98,
            home=False,
            start="",
            pos="C",
        ),
        _box(
            day="2024-10-02",
            player_id=1,
            team_id=10,
            opp_team_id=20,
            minutes=32,
            fga=12,
            fg3_a=5,
            fta=2,
            fgm=6,
            fg3_m=2,
            ftm=2,
            spread=-4,
            total=224,
            pace=110,
            home=True,
            start="G",
            pos="PG",
        ),
        _box(
            day="2024-10-02",
            player_id=3,
            team_id=20,
            opp_team_id=10,
            minutes=18,
            fga=6,
            fg3_a=2,
            fta=4,
            fgm=3,
            fg3_m=1,
            ftm=3,
            spread=4,
            total=224,
            pace=105,
            home=False,
            start="C",
            pos="C",
        ),
        _box(
            day="2024-10-04",
            player_id=1,
            team_id=10,
            opp_team_id=20,
            minutes=28,
            fga=9,
            fg3_a=3,
            fta=5,
            fgm=4,
            fg3_m=1,
            ftm=4,
            spread=-2,
            total=218,
            pace=102,
            home=False,
            start="G",
            pos="SG",
        ),
        _box(
            day="2024-10-04",
            player_id=3,
            team_id=20,
            opp_team_id=10,
            minutes=24,
            fga=11,
            fg3_a=4,
            fta=1,
            fgm=6,
            fg3_m=2,
            ftm=1,
            spread=2,
            total=218,
            pace=108,
            home=True,
            start="",
            pos="C",
        ),
    ]
    return pd.DataFrame(rows)


def _raw_column(design: HierarchicalDesign, group: str, name: str) -> np.ndarray:
    if group == "minutes":
        index = design.minutes_columns.index(name)
        return (
            design.data["X_min"][:, index] * design.minutes_scale[index]
            + design.minutes_center[index]
        )
    index = design.rate_columns.index(name)
    return (
        design.data["X_rate"][:, index] * design.rate_scale[index]
        + design.rate_center[index]
    )


def test_recent_games_pull_the_minutes_trail() -> None:
    design = build_design(_games())
    trails = _raw_column(design, "minutes", "minutes_trail")
    # Player 1's modeled games follow 30 minutes, then 30 and 32.
    # The second trail sits closer to 32 than the flat mean of 31.
    assert trails[0] == pytest.approx(30)
    assert 31 < trails[1] < 32
    assert abs(trails[1] - 32) < abs(trails[1] - 30)


def test_last_season_minutes_carry_and_opening_night_rest_stays_seven() -> None:
    earlier = _box(
        day="2025-04-10",
        player_id=1,
        team_id=10,
        opp_team_id=20,
        minutes=30,
        fga=10,
        fg3_a=2,
        fta=2,
        fgm=5,
        fg3_m=1,
        ftm=2,
        spread=-3,
        total=220,
        pace=100,
        home=True,
        start="G",
        season_year="2024-25",
    )
    opener = _box(
        day="2025-10-22",
        player_id=1,
        team_id=10,
        opp_team_id=20,
        minutes=28,
        fga=10,
        fg3_a=2,
        fta=2,
        fgm=5,
        fg3_m=1,
        ftm=2,
        spread=-3,
        total=220,
        pace=102,
        home=True,
        start="G",
        season_year="2025-26",
    )
    design = build_design(pd.DataFrame([earlier, opener]))
    assert design.audit["output_rows"] == 1
    # Each season debut is an opener, including the 2024-25 game that is
    # dropped because it has no prior minutes.
    assert design.audit["season_opener_rest_filled"] == 2
    assert _raw_column(design, "minutes", "minutes_trail")[0] == pytest.approx(30)
    assert _raw_column(design, "minutes", "rest_days")[0] == pytest.approx(7)


def test_design_keeps_prior_information_and_logs_drops() -> None:
    design = build_design(_games())
    assert design.audit["nonpositive_minute_rows_removed"] == 1
    assert design.audit["no_prior_minutes_rows_removed"] == 2
    assert design.audit["no_prior_pace_rows_removed"] == 0
    assert design.audit["season_opener_rest_filled"] == 2
    assert design.audit["players_with_mixed_positions"] == 1
    assert design.audit["output_rows"] == 4
    assert list(design.player_ids) == [1, 3]
    trails = _raw_column(design, "minutes", "minutes_trail")
    paces = _raw_column(design, "rate", "pace_prior")
    # One prior game is that game. Two prior games sit closer to the newer one
    # than the flat mean (31 minutes, 105 and 101.5 pace).
    assert trails[0] == pytest.approx(30)
    assert trails[2] == pytest.approx(20)
    assert 31 < trails[1] < 32
    assert 18 < trails[3] < 19
    assert paces[0] == pytest.approx(100)
    assert paces[2] == pytest.approx(98)
    assert 105 < paces[1] < 110
    assert 101.5 < paces[3] < 105
    np.testing.assert_allclose(_raw_column(design, "minutes", "b2b"), [1, 0, 1, 0])
    np.testing.assert_allclose(_raw_column(design, "rate", "is_home"), [1, 0, 0, 1])
    np.testing.assert_array_equal(design.data["fga2"], [7, 6, 4, 7])
    np.testing.assert_array_equal(design.data["fgm2"], [4, 3, 2, 4])
    np.testing.assert_array_equal(design.points, [16, 13, 10, 15])
    assert design.data["pos_of_player"].shape == (design.n_players,)
    assert design.positions[design.data["pos_of_player"][0]] == "PG"


def test_current_game_minutes_and_pace_do_not_change_covariates() -> None:
    original = build_design(_games())
    tampered = _games()
    last = tampered["game_date"].eq(pd.Timestamp("2024-10-04"))
    tampered.loc[last, "minutes"] = tampered.loc[last, "minutes"] + 15
    tampered.loc[last, "team_pace"] = tampered.loc[last, "team_pace"] + 20
    rebuilt = build_design(tampered)
    np.testing.assert_allclose(original.data["X_min"], rebuilt.data["X_min"])
    np.testing.assert_allclose(original.data["X_rate"], rebuilt.data["X_rate"])


def test_market_gaps_are_counted_and_removed() -> None:
    games = _games()
    games.loc[games["game_date"].eq(pd.Timestamp("2024-10-04")), "game_total"] = np.nan
    design = build_design(games)
    assert design.audit["missing_market_rows_removed"] == 2
    assert design.audit["output_rows"] == 2


def test_impossible_makes_raise() -> None:
    games = _games()
    games.loc[0, "fgm"] = 20
    with pytest.raises(ValueError, match="fgm2"):
        build_design(games)


def test_points_that_disagree_with_the_shot_sum_raise() -> None:
    games = _games()
    games.loc[0, "pts"] = 0
    with pytest.raises(ValueError, match="pts"):
        build_design(games)


def test_minutes_above_the_truncation_bound_raise() -> None:
    games = _games()
    games.loc[4, "minutes"] = MINUTES_UPPER + 1
    with pytest.raises(ValueError, match="minutes"):
        build_design(games)


def test_transform_reuses_the_training_scaler_and_clamps_future_blocks() -> None:
    history = _games()
    design = build_design(history)
    future = _box(
        day="2024-11-15",
        player_id=1,
        team_id=10,
        opp_team_id=20,
        minutes=30,
        fga=10,
        fg3_a=2,
        fta=2,
        fgm=5,
        fg3_m=1,
        ftm=2,
        spread=-6,
        total=230,
        pace=120,
        home=True,
        start="G",
        pos="PG",
    )
    frame = pd.concat([history, pd.DataFrame([future])], ignore_index=True)
    keep = frame["game_date"].eq(pd.Timestamp("2024-11-15"))
    scored, audit = transform_design(frame, design, keep)
    spread_index = design.minutes_columns.index("spread")
    expected = (-6 - design.minutes_center[spread_index]) / design.minutes_scale[
        spread_index
    ]
    assert scored["X_min"][0, spread_index] == pytest.approx(expected)
    assert scored["block"].item() == design.n_blocks - 1
    assert audit["block_indices_clamped"] == 1
    assert scored["player"].shape == (1,)


def test_transform_rejects_an_unseen_player() -> None:
    design = build_design(_games())
    debut = _box(
        day="2024-10-05",
        player_id=99,
        team_id=10,
        opp_team_id=20,
        minutes=15,
        fga=6,
        fg3_a=1,
        fta=2,
        fgm=3,
        fg3_m=1,
        ftm=1,
        spread=-1,
        total=220,
        pace=101,
        home=True,
        start="",
        pos="PG",
    )
    nxt = _box(
        day="2024-10-06",
        player_id=99,
        team_id=10,
        opp_team_id=20,
        minutes=16,
        fga=7,
        fg3_a=2,
        fta=2,
        fgm=3,
        fg3_m=1,
        ftm=2,
        spread=-1,
        total=221,
        pace=101,
        home=True,
        start="G",
        pos="PG",
    )
    frame = pd.concat([_games(), pd.DataFrame([debut, nxt])], ignore_index=True)
    keep = frame["game_date"].eq(pd.Timestamp("2024-10-06"))
    with pytest.raises(ValueError, match="player"):
        transform_design(frame, design, keep)


def test_build_model_rejects_a_per_game_position_index() -> None:
    design = build_design(_games())
    broken = dict(design.data)
    broken["pos_of_player"] = np.zeros(design.audit["output_rows"], dtype=int)
    with pytest.raises(ValueError, match="pos_of_player"):
        build_model(
            broken,
            design.n_players,
            design.n_teams,
            design.n_pos,
            design.n_blocks,
        )


def test_minutes_mean_adds_the_player_block_drift() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    model = build_model(
        design.data,
        design.n_players,
        design.n_teams,
        design.n_pos,
        design.n_blocks,
    )
    assert "min_drift" in model.named_vars
    assert "min_sd_z" in model.named_vars
    assert "min_sd_drift_z" in model.named_vars
    init_dist = model["min_drift"].owner.inputs[0]
    init_sigma = init_dist.owner.inputs[-1].owner.inputs[0]
    assert float(init_sigma.data) == pytest.approx(2.0)

    # TruncatedNormal inputs are rng, size, location, scale, lower, upper.
    mu = model.replace_rvs_by_values([model["minutes"].owner.inputs[2]])[0]
    minutes_mean = model.compile_fn(mu)
    point = {
        "min_mu0": np.array(20.0),
        "min_player": np.zeros(design.n_players),
        "b_min": np.zeros(design.data["X_min"].shape[1]),
        "min_drift": np.zeros((design.n_players, design.n_blocks)),
    }
    needed = [
        inp.variable.name
        for inp, container in zip(
            minutes_mean.f.maker.inputs, minutes_mean.f.input_storage
        )
        if not container.implicit
    ]
    baseline = np.asarray(
        minutes_mean({name: point[name] for name in needed}), dtype=float
    )

    player = int(design.data["player"][0])
    block = int(design.data["block"][0])
    point["min_drift"] = np.array(point["min_drift"], copy=True)
    point["min_drift"][player, block] = 7.0
    shifted = np.asarray(
        minutes_mean({name: point[name] for name in needed}), dtype=float
    )
    touched = (design.data["player"] == player) & (design.data["block"] == block)
    assert touched.any()
    assert (~touched).any()
    np.testing.assert_allclose(shifted[touched] - baseline[touched], 7.0)
    np.testing.assert_allclose(shifted[~touched], baseline[~touched])


def _call_model(model, fn, point: dict[str, np.ndarray]) -> np.ndarray:
    needed = [
        inp.variable.name
        for inp, container in zip(fn.f.maker.inputs, fn.f.input_storage)
        if not container.implicit
    ]
    return np.asarray(fn({name: point[name] for name in needed}), dtype=float)


def test_minutes_noise_changes_only_that_player() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    model = build_model(
        design.data,
        design.n_players,
        design.n_teams,
        design.n_pos,
        design.n_blocks,
    )
    sigma = model.replace_rvs_by_values([model["minutes"].owner.inputs[3]])[0]
    minutes_sd = model.compile_fn(sigma)
    point = model.initial_point()
    point["min_sd_z"] = np.zeros(design.n_players)
    baseline = _call_model(model, minutes_sd, point)
    player = int(design.data["player"][0])
    point["min_sd_z"] = np.array(point["min_sd_z"], copy=True)
    point["min_sd_z"][player] = 1.0
    shifted = _call_model(model, minutes_sd, point)
    touched = design.data["player"] == player
    assert np.all(shifted[touched] > baseline[touched])
    np.testing.assert_allclose(shifted[~touched], baseline[~touched])

    drift_sd = model.replace_rvs_by_values([model["min_sd_drift"]])[0]
    drift_scale = model.compile_fn(drift_sd)
    point["min_sd_drift_z"] = np.zeros(design.n_players)
    base_scale = _call_model(model, drift_scale, point)
    point["min_sd_drift_z"] = np.array(point["min_sd_drift_z"], copy=True)
    point["min_sd_drift_z"][player] = 1.0
    moved_scale = _call_model(model, drift_scale, point)
    assert moved_scale[player] > base_scale[player]
    others = np.ones(design.n_players, dtype=bool)
    others[player] = False
    np.testing.assert_allclose(moved_scale[others], base_scale[others])


def _context_games() -> pd.DataFrame:
    # Shared game ids so both teams are one game. Player 1 is the only
    # teammate whose prior minutes clear 24. He sits on g1.
    rows = []
    played = {
        1: {0: 10, 2: 10, 3: 40, 4: 12},
        3: {0: 11, 1: 20, 2: 12, 3: 18, 4: 16},
        2: {0: 8, 1: 9, 2: 10, 3: 11, 4: 12},
    }
    defenses = {10: [112, 108, 110, 109, 111], 20: [105, 80, 101, 99, 100]}
    paces = {10: [100, 102, 101, 104, 103], 20: [98, 120, 99, 97, 96]}
    usage = {1: 0.30, 3: [0.18, 0.25, 0.16, 0.20, 0.19], 2: 0.22}
    days = ["2024-10-01", "2024-10-03", "2024-10-05", "2024-10-07", "2024-10-09"]
    teams = {
        1: (10, 20, True, "G", "PG", 36),
        3: (10, 20, True, "", "SG", 22),
        2: (20, 10, False, "F", "PF", 30),
    }
    for index, day in enumerate(days):
        for player_id, schedule in played.items():
            if index not in schedule:
                continue
            team_id, opp_team_id, home, start, pos, minutes = teams[player_id]
            pts = schedule[index]
            usg = usage[player_id]
            usg_value = usg[index] if isinstance(usg, list) else usg
            rows.append(
                _box(
                    day=day,
                    game_id=f"g{index}",
                    player_id=player_id,
                    team_id=team_id,
                    opp_team_id=opp_team_id,
                    minutes=minutes,
                    fga=0,
                    fg3_a=0,
                    fta=pts,
                    fgm=0,
                    fg3_m=0,
                    ftm=pts,
                    spread=-3 if home else 3,
                    total=220,
                    pace=paces[team_id][index],
                    home=home,
                    start=start,
                    pos=pos,
                    usg=usg_value,
                    defense=defenses[team_id][index],
                )
            )
    return pd.DataFrame(rows)


def test_context_covariates_use_prior_games_and_the_selected_spec() -> None:
    spec = expanded_covariates()
    design = build_design(_context_games(), covariates=spec)
    assert design.minutes_columns == spec.minutes
    assert design.rate_columns == spec.rate

    def _row(design, player_id: int, game_id: str) -> np.ndarray:
        return (design.row_player_ids == player_id) & (design.row_game_ids == game_id)

    sat = _row(design, 3, "g1")
    last = _row(design, 3, "g4")
    assert sat.sum() == 1
    assert _raw_column(design, "minutes", "stars_out")[sat] == pytest.approx(1)
    assert _raw_column(design, "rate", "opp_def_prior")[sat] == pytest.approx(105)
    assert _raw_column(design, "rate", "opp_pace_prior")[sat] == pytest.approx(98)
    assert _raw_column(design, "rate", "usg_stars_in")[sat] == pytest.approx(0.18)
    assert design.audit["usg_stars_out_filled"] >= 1
    # Priors for player 3 entering g4: points 11, 20, 12, 18.
    # The newest game weighs 1. Each game further back weighs 0.5 ** (1/4).
    decay = 0.5 ** (1.0 / 4.0)

    def _weighted(values: list[float]) -> float:
        ages = np.arange(len(values) - 1, -1, -1)
        return float(np.average(values, weights=decay**ages))

    short = _weighted([20, 12, 18])
    long = _weighted([11, 20, 12, 18])
    assert short != pytest.approx((20 + 12 + 18) / 3)
    assert _raw_column(design, "minutes", "points_trend")[last] == pytest.approx(
        short - long
    )
    assert _raw_column(design, "rate", "usg_stars_out")[last] == pytest.approx(0.25)

    tampered = _context_games()
    current = tampered["game_id"].eq("g1") & tampered["team_id"].eq(20)
    tampered.loc[current, "team_def_rating"] = 50
    tampered.loc[current, "team_pace"] = 70
    role = tampered["game_id"].eq("g1") & tampered["player_id"].eq(3)
    tampered.loc[role, "usg_pct"] = 0.5
    rebuilt = build_design(tampered, covariates=spec)
    sat_again = _row(rebuilt, 3, "g1")
    assert _raw_column(rebuilt, "rate", "opp_def_prior")[sat_again] == pytest.approx(
        105
    )
    assert _raw_column(rebuilt, "rate", "opp_pace_prior")[sat_again] == pytest.approx(
        98
    )
    assert _raw_column(rebuilt, "rate", "usg_stars_in")[sat_again] == pytest.approx(
        0.18
    )


def test_unknown_covariate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="opp_def_prior"):
        build_design(
            _games(),
            covariates=CovariateSpec(minutes=("spread", "opp_def_prior")),
        )
    with pytest.raises(ValueError, match="unknown"):
        build_design(
            _context_games(),
            covariates=CovariateSpec(minutes=("not_a_column",)),
        )


def test_posterior_predictive_follows_minutes_into_makes() -> None:
    pytest.importorskip("pymc")
    design = build_design(_games(), block_days=2)
    model = build_model(
        design.data,
        design.n_players,
        design.n_teams,
        design.n_pos,
        design.n_blocks,
    )
    import pymc as pm

    with model:
        prior = pm.sample_prior_predictive(draws=40, random_seed=0)
    minutes = prior.prior_predictive["minutes"].values
    attempts = prior.prior_predictive["fga2"].values
    makes = prior.prior_predictive["fgm2"].values
    assert minutes.shape[-1] == design.audit["output_rows"]
    assert np.all((minutes > 0) & (minutes <= MINUTES_UPPER))
    assert np.all(makes <= attempts)
    # Two-point attempts per minute are centered at exp(log 0.22). A model
    # that ignored minutes would land near 0.22 attempts total, not per minute.
    attempts_per_minute = attempts.mean() / minutes.mean()
    assert 0.08 < attempts_per_minute < 0.6

    with model:
        idata = pm.sample(
            draws=15,
            tune=15,
            chains=1,
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
    subset = {
        key: value[:2] for key, value in design.data.items() if key != "pos_of_player"
    }
    subset["pos_of_player"] = design.data["pos_of_player"]
    points = simulate_points(model, idata, subset, random_seed=0)
    assert points.shape[0] == 2
    assert points.shape[1] == 15
    assert np.all(np.isfinite(points))
    assert np.all(points >= 0)
    assert points.mean() > 1
