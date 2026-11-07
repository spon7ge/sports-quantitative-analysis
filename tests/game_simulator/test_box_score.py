"""Layer 6 reads one box score off a single pass through layers 1–5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from game_boxscore import (
    Prop,
    assemble_box_score,
    format_outcome,
    prop_probability,
    simulate_box_scores,
)
from game_enviroment import FEATURES
from game_state import PRE
from game_rotations import RotationSimulator


def _environment_model(*, p_ot: float = 0.0) -> dict:
    def team(abbr: str) -> dict:
        return {
            "abbr": abbr,
            "num": 0.0,
            "den": 0.0,
            "last_date": "2024-04-10",
            "last_season": "2023-24",
        }

    return {
        "config": {"season_carry": 0.5, "prior_weight": 10.0},
        "state": {
            "league_poss": 100.0,
            "league_pts": 220.0,
            "teams": {"1": team("BOS"), "2": team("NYK")},
        },
        "model": {"name": "league_prior"},
        "features": [name for name in FEATURES if not str(name).startswith("lineup_")],
        "margin": {"a": 0.0, "b": 0.0, "sd": 12.0, "nu": 8.0, "scale": 12.0},
        "script": {"kappa": 0.0},
        "residual_t": {"nu": 30.0, "scale": 1.0},
        "ot": {
            "p_ot_overall": p_ot,
            "p_ot_by_spread_bucket": [p_ot, p_ot, p_ot, p_ot],
            "p_more_given_ot": 0.0,
        },
        "split": {"values": [0], "probs": [1.0]},
    }


def _roster(ids: list[str], minutes: list[float], **overrides) -> dict:
    players = len(ids)
    weights = np.full(players, 1.0 / players)
    roster = {
        "player_ids": np.array(ids),
        "expected_minutes": np.array(minutes, dtype=float),
        "minutes_sd": np.full(players, 1.0),
        "base_weights": weights.copy(),
        "available": np.ones(players),
        "reference_minutes": np.array(minutes, dtype=float),
        "p_3pa": np.full(players, 0.35),
        "p_3p": np.full(players, 0.36),
        "p_2p": np.full(players, 0.52),
        "fta_per_fga": np.full(players, 0.2),
        "ft_pct": np.full(players, 0.8),
        "tov_rate": np.full(players, 0.1),
        "assist_propensity": weights.copy(),
        "oreb_propensity": weights.copy(),
        "oreb_rate": np.linspace(0.1, 0.4, players),
        "dreb_rate": np.linspace(0.1, 0.5, players),
        "creation_rate": np.linspace(0.3, 0.05, players),
        "assisted_rate": np.full(players, 0.6),
        "stl_rate": np.ones(players),
        "blk_rate": np.linspace(0.05, 0.4, players),
    }
    roster.update(overrides)
    return roster


def _rosters(**home_overrides) -> dict:
    home = _roster(
        ["A", "B", "C", "D", "E"],
        [36, 34, 30, 24, 16],
        **home_overrides,
    )
    away = _roster(["V", "W", "X", "Y", "Z"], [34, 32, 30, 26, 18])
    return {"BOS": home, "NYK": away}


def _simulate(n_sims: int, seed: int, **kwargs):
    return simulate_box_scores(
        environment_model=_environment_model(),
        home="BOS",
        away="NYK",
        date="2026-11-05",
        total=220.0,
        spread=0.0,
        rosters=_rosters(),
        n_sims=n_sims,
        rng=np.random.default_rng(seed),
        fga_per_poss=1.0,
        alpha_fga=0.0,
        **kwargs,
    )


def test_minutes_sum_to_the_layer1_game_length():
    box = _simulate(40, seed=1)
    game_minutes = box.environment["game_minutes"].to_numpy()
    for team in ("BOS", "NYK"):
        assert np.allclose(box.teams[team].stats["MIN"].sum(axis=1), 5.0 * game_minutes)


def test_shot_volume_tracks_layer1_possessions():
    box = _simulate(400, seed=2)
    poss = np.rint(box.environment["team_poss"].to_numpy())
    fga = box.teams["BOS"].stats["FGA"].sum(axis=1)
    assert abs(float(fga.mean()) - float(poss.mean())) < 4.0


def test_box_score_uses_layer5_events_and_layer4_points():
    box = _simulate(30, seed=3, p_rebound=1.0, p_oreb=0.0, p_steal=1.0, p_block=1.0)
    bos, nyk = box.teams["BOS"].stats, box.teams["NYK"].stats
    assert np.array_equal(bos["PTS"], 2 * bos["2PM"] + 3 * bos["3PM"] + bos["FTM"])
    assert np.array_equal(bos["REB"], bos["OREB"] + bos["DREB"])
    assert np.all(bos["AST"].sum(axis=1) <= bos["FGM"].sum(axis=1))
    misses = nyk["FGA"].sum(axis=1) - nyk["FGM"].sum(axis=1)
    assert np.all(bos["DREB"].sum(axis=1) <= misses)
    assert np.all(bos["BLK"].sum(axis=1) <= misses)
    assert np.all(bos["STL"].sum(axis=1) <= nyk["TOV"].sum(axis=1))


def test_zero_assisted_rate_beats_layer4_assist_pool():
    box = _simulate(20, seed=4)
    # Rebuild is unnecessary: the default assisted rate is positive.
    # A dedicated run with no assists must stay at zero even though Layer 4
    # still draws its own assist pool from made shots.
    quiet = simulate_box_scores(
        environment_model=_environment_model(),
        home="BOS",
        away="NYK",
        date="2026-11-05",
        total=220.0,
        spread=0.0,
        rosters=_rosters(assisted_rate=np.zeros(5)),
        n_sims=15,
        rng=np.random.default_rng(5),
    )
    assert np.all(quiet.teams["BOS"].stats["AST"] == 0)
    assert box.n_sims == 20


def test_same_generator_seed_reproduces_the_box_score():
    first = _simulate(25, seed=6)
    second = _simulate(25, seed=6)
    other = _simulate(25, seed=7)
    assert np.array_equal(first.teams["BOS"].stats["PTS"], second.teams["BOS"].stats["PTS"])
    assert np.array_equal(first.teams["NYK"].stats["REB"], second.teams["NYK"].stats["REB"])
    assert not np.array_equal(first.teams["BOS"].stats["PTS"], other.teams["BOS"].stats["PTS"])


def test_rotation_simulator_takes_the_layer1_minute_budget():
    # This simulator's own overtime rate is 1, which would budget 265 minutes.
    # Layer 1 is configured with no overtime, so the box score must sum to 240.
    simulator = RotationSimulator(
        pre_model=None,
        cond_model=None,
        sigma_pre=np.array([2.0]),
        sigma_cond=np.array([2.0]),
        alpha=10.0,
        margin_resid=np.array([0.0]),
        p_ot=1.0,
        fallback_mu=30.0,
        fallback_sd=1.0,
    )
    frame = pd.DataFrame({column: [np.nan] for column in PRE})
    # Five identical rows; features are missing so the fallback minutes are used.
    frames = pd.concat([frame] * 5, ignore_index=True)
    frames["is_starter"] = 1
    home = _roster(["A", "B", "C", "D", "E"], [36, 34, 30, 24, 16])
    home["minute_frame"] = frames
    away = _roster(["V", "W", "X", "Y", "Z"], [34, 32, 30, 26, 18])
    away["minute_frame"] = frames.copy()
    box = simulate_box_scores(
        environment_model=_environment_model(p_ot=0.0),
        home="BOS",
        away="NYK",
        date="2026-11-05",
        total=220.0,
        spread=0.0,
        rosters={"BOS": home, "NYK": away},
        n_sims=12,
        rng=np.random.default_rng(8),
        rotation_simulator=simulator,
    )
    assert np.allclose(box.teams["BOS"].stats["MIN"].sum(axis=1), 240.0)
    assert np.allclose(box.environment["game_minutes"], 48.0)


def test_one_outcome_is_a_single_simulation_row():
    box = _simulate(5, seed=9)
    rows = format_outcome(box, sim=0)
    assert len(rows) == 10
    bos = [row for row in rows if row["team"] == "BOS"]
    assert [row["player"] for row in bos] == ["A", "B", "C", "D", "E"]
    assert bos[0]["PTS"] == box.teams["BOS"].stats["PTS"][0, 0]
    assert bos[0]["MIN"] == box.teams["BOS"].stats["MIN"][0, 0]
    with pytest.raises(ValueError, match="sim"):
        format_outcome(box, sim=5)


def test_joint_probability_uses_the_same_simulation_index():
    from game_boxscore import BoxScoreSet, TeamBoxScore

    points = np.array([[30, 10], [30, 10], [10, 30], [10, 30]], dtype=float)
    box = BoxScoreSet(
        teams={
            "BOS": TeamBoxScore(
                player_ids=("A", "B"),
                stats={"PTS": points},
            )
        },
        n_sims=4,
        environment=pd.DataFrame(),
        metadata={},
    )
    same_direction = prop_probability(
        box,
        [
            Prop("BOS", "A", "PTS", 20.5),
            Prop("BOS", "B", "PTS", 20.5),
        ],
    )
    # The two series never clear 20.5 together, so the joint is 0.
    # The independent product is 0.5 * 0.5.
    assert same_direction.marginals == (0.5, 0.5)
    assert same_direction.independent_product == pytest.approx(0.25)
    assert same_direction.joint == 0.0

    copied = points.copy()
    copied[:, 1] = copied[:, 0]
    tied = BoxScoreSet(
        teams={
            "BOS": TeamBoxScore(player_ids=("A", "B"), stats={"PTS": copied}),
        },
        n_sims=4,
        environment=pd.DataFrame(),
        metadata={},
    )
    together = prop_probability(
        tied,
        [Prop("BOS", "A", "PTS", 20.5), Prop("BOS", "B", "PTS", 20.5)],
    )
    assert together.joint == 0.5
    assert together.joint != together.independent_product


def test_over_line_is_strict_for_half_point_props():
    from game_boxscore import BoxScoreSet, TeamBoxScore

    box = BoxScoreSet(
        teams={
            "BOS": TeamBoxScore(
                player_ids=("A",),
                stats={"PTS": np.array([[27], [28]], dtype=float)},
            )
        },
        n_sims=2,
        environment=pd.DataFrame(),
        metadata={},
    )
    result = prop_probability(box, [Prop("BOS", "A", "PTS", 27.5)])
    assert result.joint == 0.5
    assert result.marginals == (0.5,)
    assert result.independent_product == 0.5


def test_pipeline_rejects_a_missing_environment_and_a_bad_prop():
    with pytest.raises(ValueError, match="environment"):
        simulate_box_scores(
            home="BOS",
            away="NYK",
            date="2026-11-05",
            rosters=_rosters(),
            n_sims=2,
            rng=np.random.default_rng(1),
        )
    box = _simulate(4, seed=10)
    with pytest.raises(ValueError, match="player"):
        prop_probability(box, [Prop("BOS", "Nobody", "PTS", 10.5)])


def _counts(values: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.array(values, copy=True) for name in (
        "FGA", "FGM", "2PA", "2PM", "3PA", "3PM", "FTA", "FTM", "PTS", "TOV",
        "AST", "OREB", "DREB", "REB", "STL", "BLK",
    )}


def _drawn_game():
    environment = pd.DataFrame({"game_minutes": [48.0, 48.0, 48.0, 53.0]})
    players = {
        "HOM": pd.DataFrame({"player_id": [1, 2]}),
        "AWY": pd.DataFrame({"player_id": [3]}),
    }
    points = np.array([[30, 10], [30, 10], [10, 30], [10, 30]], dtype=float)
    rebounds = np.array([[5, 1], [5, 1], [1, 5], [1, 5]], dtype=float)
    minutes = {
        "HOM": np.array([[30, 18], [10, 38], [36, 12], [40, 13]], dtype=float),
        "AWY": np.array([[28], [30], [22], [35]], dtype=float),
    }
    home_shots = _counts(np.zeros_like(points))
    home_shots["PTS"] = points
    away_shots = _counts(np.zeros((4, 1)))
    away_shots["PTS"] = np.array([[8], [12], [4], [20]], dtype=float)
    home_events = _counts(np.zeros_like(points))
    home_events["REB"] = rebounds
    away_events = _counts(np.zeros((4, 1)))
    return environment, players, minutes, {
        "HOM": home_shots,
        "AWY": away_shots,
    }, {
        "HOM": home_events,
        "AWY": away_events,
    }


def test_assembled_box_keeps_the_simulation_row():
    environment, players, minutes, shots, events = _drawn_game()
    box = assemble_box_score(environment, players, minutes, shots, events)
    assert np.array_equal(box.teams["HOM"].stats["PTS"], shots["HOM"]["PTS"])
    assert np.array_equal(box.teams["HOM"].stats["MIN"], minutes["HOM"])
    assert box.teams["HOM"].stats["REB"][2, 1] == 5
    assert box.teams["HOM"].player_ids == (1, 2)
    assert box.n_sims == 4
    priced = prop_probability(
        box,
        [Prop("HOM", 1, "PTS", 20.5), Prop("HOM", 2, "PTS", 20.5)],
    )
    assert priced.marginals == (0.5, 0.5)
    assert priced.independent_product == pytest.approx(0.25)
    assert priced.joint == 0.0
    row = format_outcome(box, 2, stats=("MIN", "PTS", "REB"))
    assert [item["player"] for item in row if item["team"] == "HOM"] == [1, 2]
    assert row[1]["PTS"] == 30
    assert row[1]["MIN"] == 12
    assert row[1]["REB"] == 5


def test_assembly_rejects_a_row_that_does_not_match():
    environment, players, minutes, shots, events = _drawn_game()
    short = environment.iloc[:3].reset_index(drop=True)
    with pytest.raises(ValueError, match="environment"):
        assemble_box_score(short, players, minutes, shots, events)
    broken = {team: dict(frame) for team, frame in shots.items()}
    broken["HOM"]["PTS"] = shots["HOM"]["PTS"][:3]
    with pytest.raises(ValueError, match="PTS"):
        assemble_box_score(environment, players, minutes, broken, events)
    missing = {team: dict(frame) for team, frame in shots.items()}
    del missing["HOM"]["3PM"]
    with pytest.raises(ValueError, match="3PM"):
        assemble_box_score(environment, players, minutes, missing, events)


def test_assembly_rejects_a_duplicate_player_and_a_single_team():
    environment, players, minutes, shots, events = _drawn_game()
    players["HOM"] = pd.DataFrame({"player_id": [1, 1]})
    with pytest.raises(ValueError, match="player"):
        assemble_box_score(environment, players, minutes, shots, events)
    with pytest.raises(ValueError, match="two teams"):
        assemble_box_score(
            environment,
            {"HOM": players["AWY"]},
            {"HOM": minutes["AWY"]},
            {"HOM": shots["AWY"]},
            {"HOM": events["AWY"]},
        )
