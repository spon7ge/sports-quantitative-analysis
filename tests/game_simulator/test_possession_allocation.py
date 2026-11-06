from __future__ import annotations

import numpy as np
import pytest

from possession_allocation import (
    allocate_and_shoot,
    allocate_from_pool,
    as_nonneg_int_vector,
    multinomial_rows,
    renormalize_props,
    sample_negbin,
    simulate_game,
    simulate_fta_ftm,
)


def test_as_nonneg_int_vector_broadcasts_scalar():
    out = as_nonneg_int_vector(90.0, n_sims=3, name="team_fga")
    assert out.dtype == np.int64
    assert np.array_equal(out, [90, 90, 90])


def test_as_nonneg_int_vector_rejects_negative_and_float():
    with pytest.raises(ValueError, match="team_fga"):
        as_nonneg_int_vector(-1, n_sims=2, name="team_fga")
    with pytest.raises(ValueError, match="team_fga"):
        as_nonneg_int_vector(np.array([1.5, 2.0]), n_sims=2, name="team_fga")


def test_as_nonneg_int_vector_rejects_wrong_length():
    with pytest.raises(ValueError, match="team_fga"):
        as_nonneg_int_vector(np.array([80, 90]), n_sims=3, name="team_fga")


def test_multinomial_rows_sum_to_totals():
    rng = np.random.default_rng(0)
    totals = np.array([10, 0, 25], dtype=np.int64)
    probs = np.array([0.4, 0.35, 0.25])
    out = multinomial_rows(totals, probs, rng)
    assert out.shape == (3, 3)
    assert np.array_equal(out.sum(axis=1), totals)


def test_sample_negbin_poisson_limit_nonnegative():
    rng = np.random.default_rng(1)
    draws = sample_negbin(mu=np.full(1000, 5.0), alpha=0.0, rng=rng)
    assert draws.shape == (1000,)
    assert draws.min() >= 0


def test_fga_rows_sum_to_team_total():
    rng = np.random.default_rng(2)
    team_fga = np.array([80, 90, 100], dtype=np.int64)
    shares = np.array([0.31, 0.27, 0.19, 0.15, 0.08])
    out = allocate_and_shoot(
        team_fga,
        shares,
        p_3pa=np.full(5, 0.4),
        p_3p=np.full(5, 0.35),
        p_2p=np.full(5, 0.55),
        rng=rng,
    )
    assert np.array_equal(out["FGA"].sum(1), team_fga)
    assert np.array_equal(out["2PA"] + out["3PA"], out["FGA"])
    assert np.all(out["3PM"] <= out["3PA"])
    assert np.all(out["2PM"] <= out["2PA"])
    assert np.array_equal(out["FGM"], out["2PM"] + out["3PM"])


def test_fta_ftm_player_generated_not_team_sum_constrained():
    rng = np.random.default_rng(3)
    fga = np.array([[10, 5], [8, 8]], dtype=np.int64)
    fta, ftm = simulate_fta_ftm(
        fga,
        fta_per_fga=np.array([0.3, 0.2]),
        ft_pct=np.array([0.8, 0.75]),
        alpha_fta=0.1,
        rng=rng,
    )
    assert fta.shape == fga.shape
    assert np.all(ftm <= fta)


def test_allocate_from_pool_sums_and_ast_cap_pattern():
    rng = np.random.default_rng(4)
    pool = np.array([10, 7], dtype=np.int64)
    props = renormalize_props(np.array([0.5, 0.3, 0.2]))
    out = allocate_from_pool(pool, props, rng)
    assert np.array_equal(out.sum(1), pool)


def _bos_players():
    return {
        "player_ids": np.array(["Tatum", "Brown", "White", "Porzingis", "Other"]),
        "base_weights": np.array([0.31, 0.27, 0.19, 0.15, 0.08]),
        "available": np.ones(5),
        "projected_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "reference_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "p_3pa": np.array([0.45, 0.40, 0.55, 0.35, 0.40]),
        "p_3p": np.array([0.37, 0.35, 0.40, 0.37, 0.34]),
        "p_2p": np.array([0.55, 0.54, 0.52, 0.58, 0.50]),
        "fta_per_fga": np.array([0.28, 0.25, 0.15, 0.30, 0.18]),
        "ft_pct": np.array([0.83, 0.78, 0.88, 0.84, 0.75]),
        "tov_rate": np.array([0.12, 0.11, 0.09, 0.10, 0.10]),
        "assist_propensity": np.array([0.25, 0.20, 0.35, 0.10, 0.10]),
        "oreb_propensity": np.array([0.15, 0.15, 0.10, 0.40, 0.20]),
        "assist_rate": 0.60,
        "oreb_rate": 0.28,
        "dispersion": {"fta": 0.05, "ast": 0.05, "oreb": 0.05, "tov": 0.05},
    }


def test_simulate_game_reconciles_fga_and_ast_cap():
    rng = np.random.default_rng(5)
    n_sims = 200
    result = simulate_game(
        team_inputs={
            "BOS": {
                "possessions": 101,
                "team_fga": np.full(n_sims, 90, dtype=np.int64),
            }
        },
        player_inputs={"BOS": _bos_players()},
        n_sims=n_sims,
        rng=rng,
    )
    bos = result.teams["BOS"]
    fga = bos.stats["FGA"]
    assert np.array_equal(fga.sum(1), np.full(n_sims, 90))
    assert np.all(bos.stats["AST"].sum(1) <= bos.stats["FGM"].sum(1))
    assert np.array_equal(
        bos.stats["PTS"],
        2 * bos.stats["2PM"] + 3 * bos.stats["3PM"] + bos.stats["FTM"],
    )


def test_simulate_game_reproducible_with_seed():
    def run(seed):
        return simulate_game(
            team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
            player_inputs={"BOS": _bos_players()},
            n_sims=50,
            rng=np.random.default_rng(seed),
        )

    a = run(99)
    b = run(99)
    assert np.array_equal(a.teams["BOS"].stats["PTS"], b.teams["BOS"].stats["PTS"])


def test_simulate_game_rejects_fractional_scalar_team_fga():
    with pytest.raises(ValueError, match="team_fga"):
        simulate_game(
            team_inputs={"BOS": {"possessions": 101, "team_fga": 1.5}},
            player_inputs={"BOS": _bos_players()},
            n_sims=1,
            rng=np.random.default_rng(100),
        )


def test_simulate_game_rejects_misaligned_assist_propensity():
    players = _bos_players()
    players["assist_propensity"] = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="assist_propensity"):
        simulate_game(
            team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
            player_inputs={"BOS": players},
            n_sims=1,
            rng=np.random.default_rng(101),
        )


def test_simulate_game_rejects_negative_p_3p():
    players = _bos_players()
    players["p_3p"][0] = -0.01

    with pytest.raises(ValueError, match="p_3p"):
        simulate_game(
            team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
            player_inputs={"BOS": players},
            n_sims=1,
            rng=np.random.default_rng(102),
        )


def test_provisional_tov_metadata_when_team_tov_omitted():
    result = simulate_game(
        team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
        player_inputs={"BOS": _bos_players()},
        n_sims=20,
        rng=np.random.default_rng(6),
    )
    assert result.teams["BOS"].metadata.get("tov_pool_approximate") is True


def test_unavailable_player_receives_no_opportunities():
    players = _bos_players()
    players["available"][0] = 0
    result = simulate_game(
        team_inputs={"BOS": {"possessions": 101, "team_fga": 88}},
        player_inputs={"BOS": players},
        n_sims=200,
        rng=np.random.default_rng(7),
    )

    stats = result.teams["BOS"].stats
    for stat in ("FGA", "AST", "OREB", "TOV"):
        assert np.all(stats[stat][:, 0] == 0), stat


def test_zero_ast_and_oreb_pools_skip_propensity_normalization():
    players = _bos_players()
    players["available"][0] = 0
    players["assist_propensity"] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    players["oreb_propensity"] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    result = simulate_game(
        team_inputs={"BOS": {"possessions": 101, "team_fga": 0}},
        player_inputs={"BOS": players},
        n_sims=10,
        rng=np.random.default_rng(8),
    )

    bos = result.teams["BOS"]
    assert not np.any(bos.stats["AST"])
    assert not np.any(bos.stats["OREB"])


def test_supplied_team_tov_reconciles_without_approximate_metadata():
    n_sims = 20
    team_tov = np.arange(n_sims, dtype=np.int64)
    result = simulate_game(
        team_inputs={"BOS": {"team_fga": 88, "team_tov": team_tov}},
        player_inputs={"BOS": _bos_players()},
        n_sims=n_sims,
        rng=np.random.default_rng(9),
    )

    bos = result.teams["BOS"]
    assert np.array_equal(bos.stats["TOV"].sum(axis=1), team_tov)
    assert bos.metadata.get("tov_pool_approximate", False) is False
