from __future__ import annotations

import numpy as np
import pytest

from possession_allocation import simulate_game
from shared_events import allocate_joint_assists, event_shares, simulate_shared_events


def test_event_shares_normalize_and_zero_unavailable():
    shares = event_shares(
        rates=np.array([0.40, 0.25, 0.20, 0.15]),
        available=np.array([1, 1, 0, 1]),
        projected_minutes=np.array([32.0, 28.0, 24.0, 20.0]),
        reference_minutes=np.array([32.0, 28.0, 24.0, 20.0]),
    )
    assert shares.shape == (4,)
    assert shares[2] == 0.0
    assert shares.sum() == pytest.approx(1.0)


def test_event_shares_minutes_role_and_shot_profile_move_weight():
    rates = np.array([0.2, 0.2])
    minutes = np.array([30.0, 30.0])
    base = event_shares(
        rates=rates,
        available=np.array([1.0, 1.0]),
        projected_minutes=minutes,
        reference_minutes=minutes,
    )
    more_minutes = event_shares(
        rates=rates,
        available=np.array([1.0, 1.0]),
        projected_minutes=np.array([40.0, 20.0]),
        reference_minutes=minutes,
    )
    bigger_role = event_shares(
        rates=rates,
        available=np.array([1.0, 1.0]),
        projected_minutes=minutes,
        reference_minutes=minutes,
        role_factors=np.array([1.0, 2.0]),
    )
    rim_heavy = event_shares(
        rates=rates,
        available=np.array([1.0, 1.0]),
        projected_minutes=minutes,
        reference_minutes=minutes,
        context_factors=np.array([1.0, 3.0]),
    )
    assert np.allclose(base, [0.5, 0.5])
    assert more_minutes[0] > more_minutes[1]
    assert bigger_role[1] > base[1]
    assert rim_heavy[1] > base[1]


def test_event_shares_rejects_negative_context_even_if_lineup_is_also_negative():
    kwargs = dict(
        rates=np.array([0.5, 0.5]),
        available=np.ones(2),
        projected_minutes=np.array([30.0, 30.0]),
        reference_minutes=np.array([30.0, 30.0]),
    )
    with pytest.raises(ValueError, match="context_factors"):
        event_shares(**kwargs, context_factors=np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="lineup_factors"):
        event_shares(
            **kwargs,
            lineup_factors=np.array([-1.0, -1.0]),
            context_factors=np.array([2.0, 2.0]),
        )
    with pytest.raises(ValueError):
        event_shares(
            **kwargs,
            lineup_factors=np.array([-1.0, -1.0]),
            context_factors=np.array([-2.0, -2.0]),
        )


def test_event_shares_allow_empty_returns_zeros():
    shares = event_shares(
        rates=np.array([0.0, 0.0]),
        available=np.array([1.0, 1.0]),
        projected_minutes=np.array([30.0, 30.0]),
        reference_minutes=np.array([30.0, 30.0]),
        allow_empty=True,
    )
    assert np.array_equal(shares, [0.0, 0.0])


def test_joint_assists_credit_only_teammates_of_the_scorer():
    rng = np.random.default_rng(1)
    fgm = np.array([[5, 7], [2, 0]], dtype=np.int64)
    ast, pairs = allocate_joint_assists(
        fgm=fgm,
        assisted_rate=np.array([1.0, 1.0]),
        creation_weights=np.array([1.0, 0.0]),
        rng=rng,
    )
    assert np.array_equal(ast, [[7, 0], [0, 0]])
    assert np.array_equal(pairs[:, 0, 1], [7, 0])
    assert np.array_equal(pairs[:, 0, 0], [0, 0])
    assert np.all(pairs[:, 1, :] == 0)
    assert np.all(ast.sum(1) <= fgm.sum(1))


def test_joint_assists_split_a_scorers_makes_by_creation_weight():
    rng = np.random.default_rng(2)
    n_sims = 2000
    fgm = np.column_stack(
        [
            np.zeros(n_sims, dtype=np.int64),
            np.full(n_sims, 10, dtype=np.int64),
            np.zeros(n_sims, dtype=np.int64),
        ]
    )
    ast, pairs = allocate_joint_assists(
        fgm=fgm,
        assisted_rate=np.array([0.0, 1.0, 0.0]),
        creation_weights=np.array([0.8, 0.0, 0.2]),
        rng=rng,
    )
    assert np.array_equal(ast.sum(1), np.full(n_sims, 10))
    assert np.all(pairs[:, 1, 1] == 0)
    assert ast[:, 0].mean() == pytest.approx(8.0, abs=0.2)
    assert ast[:, 2].mean() == pytest.approx(2.0, abs=0.2)


def test_no_eligible_assister_leaves_makes_unassisted():
    ast, pairs = allocate_joint_assists(
        fgm=np.array([[6]], dtype=np.int64),
        assisted_rate=np.array([1.0]),
        creation_weights=np.array([1.0]),
        rng=np.random.default_rng(3),
    )
    assert np.array_equal(ast, [[0]])
    assert np.all(pairs == 0)


_NEXT_PLAYER = 0


def _team(n_sims, n_players, *, fga, fgm, tov, **overrides):
    global _NEXT_PLAYER

    def _matrix(value):
        if np.ndim(value) == 0:
            return np.full((n_sims, n_players), value, dtype=np.int64)
        return np.asarray(value, dtype=np.int64)

    ids = np.array([f"P{_NEXT_PLAYER + i}" for i in range(n_players)])
    _NEXT_PLAYER += n_players
    team = {
        "player_ids": ids,
        "fga": _matrix(fga),
        "fgm": _matrix(fgm),
        "tov": _matrix(tov),
        "available": np.ones(n_players),
        "projected_minutes": np.full(n_players, 30.0),
        "reference_minutes": np.full(n_players, 30.0),
        "oreb_rate": np.ones(n_players),
        "dreb_rate": np.ones(n_players),
        "creation_rate": np.ones(n_players),
        "assisted_rate": np.ones(n_players),
        "stl_rate": np.ones(n_players),
        "blk_rate": np.ones(n_players),
    }
    team.update(overrides)
    return team


def test_rebounds_partition_misses_between_the_two_teams():
    n_sims = 30
    result = simulate_shared_events(
        {
            "BOS": _team(n_sims, 2, fga=10, fgm=4, tov=1),
            "NYK": _team(n_sims, 2, fga=8, fgm=3, tov=2),
        },
        n_sims=n_sims,
        rng=np.random.default_rng(4),
        p_rebound=1.0,
        p_oreb=1.0,
        p_steal=0.0,
        p_block=0.0,
    )
    bos, nyk = result.teams["BOS"], result.teams["NYK"]
    # Each BOS player misses 6 shots, so the team misses 12. All of them
    # are offensive rebounds when p_oreb is 1.
    assert np.array_equal(bos.stats["OREB"].sum(1), np.full(n_sims, 12))
    assert np.all(nyk.stats["DREB"].sum(1) == 0)
    assert np.array_equal(bos.stats["REB"], bos.stats["OREB"] + bos.stats["DREB"])

    defensive = simulate_shared_events(
        {
            "BOS": _team(n_sims, 2, fga=10, fgm=4, tov=1),
            "NYK": _team(n_sims, 2, fga=8, fgm=3, tov=2),
        },
        n_sims=n_sims,
        rng=np.random.default_rng(5),
        p_rebound=1.0,
        p_oreb=0.0,
    )
    assert np.all(defensive.teams["BOS"].stats["OREB"].sum(1) == 0)
    assert np.array_equal(defensive.teams["NYK"].stats["DREB"].sum(1), np.full(n_sims, 12))
    assert np.array_equal(
        defensive.teams["NYK"].stats["DREB"].sum(1),
        defensive.metadata["dreb_pool"]["NYK"],
    )


def test_steals_and_blocks_come_from_opponent_events():
    n_sims = 15
    bos = _team(n_sims, 2, fga=0, fgm=0, tov=0)
    bos["blk_rate"] = np.array([1.0, 0.0])
    bos["stl_rate"] = np.array([0.0, 1.0])
    nyk_fga = np.array([[6, 4]] * n_sims, dtype=np.int64)
    nyk_fgm = np.zeros_like(nyk_fga)
    nyk_tov = np.array([[3, 1]] * n_sims, dtype=np.int64)
    made = simulate_shared_events(
        {
            "BOS": bos,
            "NYK": _team(n_sims, 2, fga=nyk_fga, fgm=nyk_fga, tov=nyk_tov),
        },
        n_sims=n_sims,
        rng=np.random.default_rng(6),
        p_rebound=1.0,
        p_steal=1.0,
        p_block=1.0,
    )
    # Every NYK shot falls, so there is nothing to block. Every NYK turnover
    # is a BOS steal, and none of BOS's own (zero) turnovers are.
    assert np.all(made.teams["BOS"].stats["BLK"].sum(1) == 0)
    assert np.array_equal(made.teams["BOS"].stats["STL"].sum(1), np.full(n_sims, 4))
    assert np.all(made.teams["BOS"].stats["STL"][:, 0] == 0)
    assert np.all(made.teams["BOS"].stats["STL"][:, 1] == 4)
    assert np.all(made.teams["NYK"].stats["STL"].sum(1) == 0)

    missed = simulate_shared_events(
        {
            "BOS": bos,
            "NYK": _team(n_sims, 2, fga=nyk_fga, fgm=nyk_fgm, tov=nyk_tov),
        },
        n_sims=n_sims,
        rng=np.random.default_rng(7),
        p_steal=1.0,
        p_block=1.0,
    )
    assert np.array_equal(missed.teams["BOS"].stats["BLK"].sum(1), np.full(n_sims, 10))
    assert np.all(missed.teams["BOS"].stats["BLK"][:, 0] == 10)
    assert np.all(missed.teams["BOS"].stats["BLK"][:, 1] == 0)


def test_matchup_context_changes_who_is_credited():
    n_sims = 400
    bos = _team(n_sims, 2, fga=0, fgm=0, tov=0)
    bos["dreb_rate"] = np.array([1.0, 1.0])
    bos["stl_rate"] = np.array([1.0, 1.0])
    bos["opponent_shot_factors"] = np.array([1.0, 3.0])
    bos["matchup_factors"] = np.array([3.0, 1.0])
    nyk = _team(n_sims, 1, fga=20, fgm=0, tov=10)
    result = simulate_shared_events(
        {"BOS": bos, "NYK": nyk},
        n_sims=n_sims,
        rng=np.random.default_rng(8),
        p_rebound=1.0,
        p_oreb=0.0,
        p_steal=1.0,
        p_block=0.0,
    )
    dreb = result.teams["BOS"].stats["DREB"]
    stl = result.teams["BOS"].stats["STL"]
    assert dreb.sum(1).mean() == pytest.approx(20.0)
    assert dreb[:, 1].mean() == pytest.approx(15.0, abs=0.5)
    assert stl[:, 0].mean() == pytest.approx(7.5, abs=0.4)


def test_injury_reallocates_existing_events():
    """Starting PG is out. Team pools stay tied to the same events.

    Bench assist totals are not the claim: losing the primary creator can
    hand the bench a slice of his assists even while their creation share
    falls. Rebounds have no such transfer, so the bench rebound count falls.
    """
    n_sims = 400
    names = np.array(
        ["Starter PG", "Backup PG", "Starting SG", "Center", "Bench guard"]
    )
    base_min = np.array([34.0, 22.0, 35.0, 31.0, 17.0])
    inj_min = np.array([0.0, 32.0, 37.0, 33.0, 11.0])
    creation = np.array([0.40, 0.25, 0.20, 0.08, 0.07])
    reb = np.array([0.05, 0.08, 0.10, 0.45, 0.12])
    available = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    injured = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    base_creation = event_shares(
        rates=creation,
        available=available,
        projected_minutes=base_min,
        reference_minutes=base_min,
    )
    inj_creation = event_shares(
        rates=creation,
        available=injured,
        projected_minutes=inj_min,
        reference_minutes=base_min,
    )
    base_reb = event_shares(
        rates=reb,
        available=available,
        projected_minutes=base_min,
        reference_minutes=base_min,
    )
    inj_reb = event_shares(
        rates=reb,
        available=injured,
        projected_minutes=inj_min,
        reference_minutes=base_min,
    )
    assert inj_creation[0] == 0.0
    assert inj_creation[1] > base_creation[1]
    assert inj_creation[2] > base_creation[2]
    assert inj_creation[4] < base_creation[4]
    assert inj_reb[3] > base_reb[3]
    assert inj_reb[4] < base_reb[4]

    fga = np.zeros((n_sims, 5), dtype=np.int64)
    fgm = np.zeros((n_sims, 5), dtype=np.int64)
    tov = np.zeros((n_sims, 5), dtype=np.int64)
    fga[:, 1:] = 8
    fgm[:, 1:] = 4
    tov[:, 1:] = 1
    opponent = _team(n_sims, 2, fga=15, fgm=5, tov=2)

    def run(minutes, active, seed):
        team = _team(
            n_sims,
            5,
            fga=fga,
            fgm=fgm,
            tov=tov,
            player_ids=names,
            available=active,
            projected_minutes=minutes,
            reference_minutes=base_min,
            oreb_rate=reb,
            dreb_rate=reb,
            creation_rate=creation,
            assisted_rate=np.ones(5),
            stl_rate=creation,
            blk_rate=reb,
        )
        return simulate_shared_events(
            {"BOS": team, "NYK": opponent},
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
            p_rebound=1.0,
            p_oreb=0.0,
            p_steal=1.0,
            p_block=1.0,
        )

    baseline = run(base_min, available, 11)
    injury = run(inj_min, injured, 12)
    base_stats = baseline.teams["BOS"].stats
    inj_stats = injury.teams["BOS"].stats

    for stat in ("AST", "OREB", "DREB", "REB", "STL", "BLK"):
        assert np.all(inj_stats[stat][:, 0] == 0), stat

    team_fgm = fgm.sum(1)
    # Opponent misses 20 shots. All of them are BOS defensive rebounds.
    assert np.array_equal(base_stats["AST"].sum(1), team_fgm)
    assert np.array_equal(inj_stats["AST"].sum(1), team_fgm)
    assert np.array_equal(base_stats["DREB"].sum(1), np.full(n_sims, 20))
    assert np.array_equal(inj_stats["DREB"].sum(1), np.full(n_sims, 20))
    assert np.array_equal(base_stats["STL"].sum(1), np.full(n_sims, 4))
    assert np.array_equal(inj_stats["STL"].sum(1), np.full(n_sims, 4))
    assert inj_stats["AST"][:, 1].mean() > base_stats["AST"][:, 1].mean()
    assert inj_stats["AST"][:, 2].mean() > base_stats["AST"][:, 2].mean()
    assert inj_stats["DREB"][:, 3].mean() > base_stats["DREB"][:, 3].mean()
    assert inj_stats["DREB"][:, 4].mean() < base_stats["DREB"][:, 4].mean()


def test_shared_events_are_reproducible():
    def run(seed):
        return simulate_shared_events(
            {
                "BOS": _team(25, 3, fga=9, fgm=4, tov=2),
                "NYK": _team(25, 2, fga=8, fgm=3, tov=1),
            },
            n_sims=25,
            rng=np.random.default_rng(seed),
            p_rebound=0.9,
            p_oreb=0.28,
            p_steal=0.55,
            p_block=0.06,
        )

    a = run(21)
    b = run(21)
    for stat in ("AST", "OREB", "DREB", "STL", "BLK"):
        assert np.array_equal(a.teams["BOS"].stats[stat], b.teams["BOS"].stats[stat])


def test_fractional_probabilities_respect_event_caps():
    n_sims = 200
    result = simulate_shared_events(
        {
            "BOS": _team(n_sims, 3, fga=12, fgm=5, tov=3),
            "NYK": _team(n_sims, 3, fga=11, fgm=4, tov=2),
        },
        n_sims=n_sims,
        rng=np.random.default_rng(22),
        p_rebound=0.9,
        p_oreb=0.27,
        p_steal=0.55,
        p_block=0.06,
    )
    bos_miss = 3 * (12 - 5)
    nyk_miss = 3 * (11 - 4)
    bos_tov = 3 * 3
    assert np.all(result.teams["BOS"].stats["OREB"].sum(1) + result.teams["NYK"].stats["DREB"].sum(1) <= bos_miss)
    assert np.all(result.teams["NYK"].stats["OREB"].sum(1) + result.teams["BOS"].stats["DREB"].sum(1) <= nyk_miss)
    assert np.all(result.teams["NYK"].stats["STL"].sum(1) <= bos_tov)
    assert np.all(result.teams["BOS"].stats["BLK"].sum(1) <= nyk_miss)
    assert np.all(result.teams["BOS"].stats["AST"].sum(1) <= 3 * 5)


def test_shared_events_reject_incoherent_inputs():
    with pytest.raises(ValueError, match="exactly two teams"):
        simulate_shared_events(
            {"BOS": _team(1, 1, fga=1, fgm=0, tov=0)},
            n_sims=1,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="p_block"):
        simulate_shared_events(
            {
                "BOS": _team(1, 1, fga=1, fgm=0, tov=0),
                "NYK": _team(1, 1, fga=1, fgm=0, tov=0),
            },
            n_sims=1,
            rng=np.random.default_rng(1),
            p_block=1.2,
        )
    bos = _team(2, 2, fga=4, fgm=5, tov=0)
    with pytest.raises(ValueError, match="FGM"):
        simulate_shared_events(
            {"BOS": bos, "NYK": _team(2, 2, fga=4, fgm=1, tov=0)},
            n_sims=2,
            rng=np.random.default_rng(1),
        )
    unavailable = _team(2, 2, fga=4, fgm=1, tov=0, available=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="unavailable"):
        simulate_shared_events(
            {"BOS": unavailable, "NYK": _team(2, 2, fga=4, fgm=1, tov=0)},
            n_sims=2,
            rng=np.random.default_rng(1),
        )


def test_no_rebounders_raises_when_misses_exist():
    bos = _team(4, 2, fga=5, fgm=0, tov=0, oreb_rate=np.zeros(2), available=np.array([1.0, 1.0]))
    # dreb_rate still positive so the opponent can take defensive rebounds;
    # the offense itself has nobody to collect an offensive rebound.
    bos["oreb_rate"] = np.zeros(2)
    with pytest.raises(ValueError, match="OREB"):
        simulate_shared_events(
            {"BOS": bos, "NYK": _team(4, 2, fga=5, fgm=1, tov=0)},
            n_sims=4,
            rng=np.random.default_rng(9),
            p_rebound=1.0,
            p_oreb=1.0,
        )


def test_layer4_shooting_feeds_shared_events():
    n_sims = 80
    rng = np.random.default_rng(30)
    players = {
        "player_ids": np.array(["A", "B", "C"]),
        "base_weights": np.array([0.5, 0.3, 0.2]),
        "available": np.ones(3),
        "projected_minutes": np.array([34.0, 30.0, 22.0]),
        "reference_minutes": np.array([34.0, 30.0, 22.0]),
        "p_3pa": np.full(3, 0.4),
        "p_3p": np.full(3, 0.36),
        "p_2p": np.full(3, 0.52),
        "fta_per_fga": np.full(3, 0.2),
        "ft_pct": np.full(3, 0.8),
        "tov_rate": np.full(3, 0.1),
        "assist_propensity": np.array([0.5, 0.3, 0.2]),
        "oreb_propensity": np.array([0.2, 0.3, 0.5]),
    }
    shot = simulate_game(
        team_inputs={
            "BOS": {"possessions": 100, "team_fga": 90, "team_tov": 14},
            "NYK": {"possessions": 100, "team_fga": 88, "team_tov": 15},
        },
        player_inputs={"BOS": players, "NYK": {**players, "player_ids": np.array(["D", "E", "F"])}},
        n_sims=n_sims,
        rng=rng,
    )

    def events(name, ids):
        stats = shot.teams[name].stats
        return _team(
            n_sims,
            3,
            fga=stats["FGA"],
            fgm=stats["FGM"],
            tov=stats["TOV"],
            player_ids=np.array(ids),
            creation_rate=np.array([0.5, 0.3, 0.2]),
            assisted_rate=np.array([0.4, 0.7, 0.65]),
            oreb_rate=np.array([0.2, 0.3, 0.5]),
            dreb_rate=np.array([0.25, 0.35, 0.4]),
            stl_rate=np.array([0.4, 0.35, 0.25]),
            blk_rate=np.array([0.1, 0.2, 0.7]),
        )

    shared = simulate_shared_events(
        {"BOS": events("BOS", ["A", "B", "C"]), "NYK": events("NYK", ["D", "E", "F"])},
        n_sims=n_sims,
        rng=np.random.default_rng(31),
    )
    bos_fgm = shot.teams["BOS"].stats["FGM"].sum(1)
    bos_miss = shot.teams["BOS"].stats["FGA"].sum(1) - bos_fgm
    bos_tov = shot.teams["BOS"].stats["TOV"].sum(1)
    assert np.all(shared.teams["BOS"].stats["AST"].sum(1) <= bos_fgm)
    assert np.all(
        shared.teams["BOS"].stats["OREB"].sum(1) + shared.teams["NYK"].stats["DREB"].sum(1)
        <= bos_miss
    )
    assert np.all(shared.teams["NYK"].stats["STL"].sum(1) <= bos_tov)
    assert np.all(shared.teams["BOS"].stats["BLK"].sum(1) <= shot.teams["NYK"].stats["FGA"].sum(1) - shot.teams["NYK"].stats["FGM"].sum(1))
    assert shared.teams["BOS"].player_ids == ("A", "B", "C")
