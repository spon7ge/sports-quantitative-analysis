"""Layer 5 assigns assists, rebounds, steals, and blocks from events already drawn.

Training rates stop before the game being scored. Live rates keep the latest
completed game. Shots and turnovers are inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shared_events import (
    allocate_events,
    fit_events,
    live_events,
    training_events,
)


def _row(**overrides) -> dict:
    row = {
        "player_id": 1,
        "game_id": "g1",
        "game_date": "2024-01-01",
        "team_id": 10,
        "min": 20.0,
        "ast": 4.0,
        "oreb": 2.0,
        "dreb": 6.0,
        "stl": 1.0,
        "blk": 1.0,
        "team_fga": 50.0,
        "team_fgm": 40.0,
        "team_ast": 20.0,
        "team_oreb": 3.0,
        "team_stl": 5.0,
        "team_blk": 2.0,
        "opp_dreb": 6.0,
        "opp_tov": 10.0,
        "opp_fga": 40.0,
    }
    row.update(overrides)
    return row


def _logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(),
            _row(
                game_id="g2",
                game_date="2024-01-15",
                min=20.0,
                ast=8.0,
                oreb=4.0,
                dreb=2.0,
                stl=3.0,
                blk=3.0,
                team_fga=30.0,
                team_fgm=20.0,
                team_ast=16.0,
                team_oreb=2.0,
                team_stl=8.0,
                team_blk=4.0,
                opp_dreb=8.0,
                opp_tov=10.0,
                opp_fga=40.0,
            ),
        ]
    )


def _slate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-02-01",
                "team_id": 10,
                "opp_team_id": 20,
                "is_home": 1,
                "is_starter": 1,
            }
        ]
    )


def _players(team_rates: dict, personal: list[dict]) -> pd.DataFrame:
    rows = []
    for player in personal:
        row = dict(team_rates)
        row.update(player)
        rows.append(row)
    return pd.DataFrame(rows)


def _shots(n_sims: int, fga, fgm, tov) -> dict[str, np.ndarray]:
    return {
        "FGA": np.tile(np.asarray(fga, dtype=np.int64), (n_sims, 1)),
        "FGM": np.tile(np.asarray(fgm, dtype=np.int64), (n_sims, 1)),
        "TOV": np.tile(np.asarray(tov, dtype=np.int64), (n_sims, 1)),
    }


def test_training_rates_exclude_the_game_being_scored():
    trained = training_events(_logs())
    game2 = trained.loc[trained["game_id"].eq("g2")].iloc[0]
    assert game2["ast_per_min"] == pytest.approx(4.0 / 20.0)
    assert game2["oreb_per_min"] == pytest.approx(2.0 / 20.0)
    assert game2["dreb_per_min"] == pytest.approx(6.0 / 20.0)
    assert game2["stl_per_min"] == pytest.approx(1.0 / 20.0)
    assert game2["blk_per_min"] == pytest.approx(1.0 / 20.0)
    assert game2["assisted_rate"] == pytest.approx(20.0 / 40.0)
    assert game2["p_rebound"] == pytest.approx(9.0 / 10.0)
    assert game2["p_oreb"] == pytest.approx(3.0 / 9.0)
    assert game2["p_steal"] == pytest.approx(5.0 / 10.0)
    assert game2["p_block"] == pytest.approx(2.0 / 40.0)
    opening = trained.loc[trained["game_id"].eq("g1")].iloc[0]
    assert np.isnan(opening["ast_per_min"])
    assert np.isnan(opening["p_rebound"])


def test_extra_rebounds_do_not_push_the_rate_above_one():
    logs = pd.DataFrame(
        [
            _row(
                team_fga=10.0,
                team_fgm=8.0,
                team_ast=4.0,
                team_oreb=2.0,
                opp_dreb=2.0,
            ),
            _row(game_id="g2", game_date="2024-01-15"),
        ]
    )
    trained = training_events(logs)
    game2 = trained.loc[trained["game_id"].eq("g2")].iloc[0]
    assert game2["p_rebound"] == pytest.approx(1.0)
    assert game2["p_oreb"] == pytest.approx(0.5)


def test_live_rates_keep_the_latest_completed_game():
    live = live_events(_logs(), _slate()).iloc[0]
    assert live["ast_per_min"] == pytest.approx((0.2 + 0.4) / 2.0)
    assert live["assisted_rate"] == pytest.approx((0.5 + 0.8) / 2.0)
    assert live["p_rebound"] == pytest.approx((0.9 + 1.0) / 2.0)
    assert live["p_oreb"] == pytest.approx((1.0 / 3.0 + 0.2) / 2.0)
    assert live["p_steal"] == pytest.approx((0.5 + 0.8) / 2.0)
    assert live["p_block"] == pytest.approx((0.05 + 0.1) / 2.0)


def test_live_rates_stop_before_the_slate_date():
    future = _row(game_id="g3", game_date="2024-02-01", ast=99.0, min=10.0)
    live = live_events(pd.concat([_logs(), pd.DataFrame([future])]), _slate())
    assert live.iloc[0]["ast_per_min"] == pytest.approx(0.3)


def test_events_come_from_the_shots_already_drawn():
    offense = _players(
        {
            "assisted_rate": 1.0,
            "p_rebound": 1.0,
            "p_oreb": 1.0,
            "p_steal": 0.0,
            "p_block": 0.0,
        },
        [
            {"player_id": 1, "ast_per_min": 0.0, "oreb_per_min": 0.5},
            {"player_id": 2, "ast_per_min": 1.0, "oreb_per_min": 0.5},
        ],
    )
    defense = _players(
        {
            "assisted_rate": 0.0,
            "p_rebound": 0.0,
            "p_oreb": 0.0,
            "p_steal": 1.0,
            "p_block": 1.0,
        },
        [
            {"player_id": 3, "stl_per_min": 0.0, "blk_per_min": 1.0, "dreb_per_min": 1.0},
            {"player_id": 4, "stl_per_min": 1.0, "blk_per_min": 0.0, "dreb_per_min": 1.0},
        ],
    )
    model = fit_events(training_events(_logs()))
    n_sims = 40
    minutes = {
        "OFF": np.tile(np.array([[30.0, 10.0]]), (n_sims, 1)),
        "DEF": np.tile(np.array([[24.0, 24.0]]), (n_sims, 1)),
    }
    drawn = allocate_events(
        model,
        {"OFF": offense, "DEF": defense},
        {
            "OFF": _shots(n_sims, [10, 10], [10, 0], [4, 0]),
            "DEF": _shots(n_sims, [6, 4], [0, 0], [3, 1]),
        },
        minutes,
        np.random.default_rng(4),
    )
    off, defender = drawn["OFF"], drawn["DEF"]
    assert np.array_equal(off["AST"][:, 1], np.full(n_sims, 10))
    assert np.all(off["AST"][:, 0] == 0)
    assert np.array_equal(off["OREB"].sum(axis=1), np.full(n_sims, 10))
    assert np.all(defender["DREB"].sum(axis=1) == 0)
    assert np.array_equal(defender["STL"].sum(axis=1), np.full(n_sims, 4))
    assert np.all(defender["STL"][:, 0] == 0)
    assert np.array_equal(defender["BLK"].sum(axis=1), np.full(n_sims, 10))
    assert np.all(defender["BLK"][:, 1] == 0)
    assert np.array_equal(off["REB"], off["OREB"] + off["DREB"])


def test_a_make_is_not_a_block_and_each_offense_keeps_its_own_rebound_rate():
    model = fit_events(training_events(_logs()))
    n_sims = 15
    both = {
        "assisted_rate": 0.0,
        "p_steal": 0.0,
        "p_block": 1.0,
        "ast_per_min": 0.0,
        "oreb_per_min": 1.0,
        "dreb_per_min": 1.0,
        "stl_per_min": 0.0,
        "blk_per_min": 1.0,
    }
    home = _players({**both, "p_rebound": 1.0, "p_oreb": 1.0}, [{"player_id": 1}])
    away = _players({**both, "p_rebound": 0.0, "p_oreb": 0.0}, [{"player_id": 2}])
    minutes = {
        "HOM": np.ones((n_sims, 1)),
        "AWY": np.ones((n_sims, 1)),
    }
    drawn = allocate_events(
        model,
        {"HOM": home, "AWY": away},
        {
            "HOM": _shots(n_sims, [10], [6], [0]),
            "AWY": _shots(n_sims, [8], [4], [0]),
        },
        minutes,
        np.random.default_rng(5),
    )
    assert np.array_equal(drawn["AWY"]["BLK"].sum(axis=1), np.full(n_sims, 4))
    assert np.array_equal(drawn["HOM"]["OREB"].sum(axis=1), np.full(n_sims, 4))
    assert np.all(drawn["AWY"]["OREB"].sum(axis=1) == 0)
    assert np.all(drawn["HOM"]["DREB"].sum(axis=1) == 0)


def test_minutes_split_a_rebound_pool():
    model = fit_events(training_events(_logs()))
    n_sims = 2000
    offense = _players(
        {"p_rebound": 1.0, "p_oreb": 1.0, "assisted_rate": 0.0, "p_steal": 0.0, "p_block": 0.0},
        [
            {"player_id": 1, "oreb_per_min": 0.4},
            {"player_id": 2, "oreb_per_min": 0.4},
        ],
    )
    defense = _players(
        {"p_rebound": 0.0, "p_oreb": 0.0, "assisted_rate": 0.0, "p_steal": 0.0, "p_block": 0.0},
        [{"player_id": 3, "dreb_per_min": 0.2}, {"player_id": 4, "dreb_per_min": 0.2}],
    )
    drawn = allocate_events(
        model,
        {"OFF": offense, "DEF": defense},
        {
            "OFF": _shots(n_sims, [10, 10], [0, 0], [0, 0]),
            "DEF": _shots(n_sims, [0, 0], [0, 0], [0, 0]),
        },
        {
            "OFF": np.tile(np.array([[30.0, 10.0]]), (n_sims, 1)),
            "DEF": np.tile(np.array([[20.0, 0.0]]), (n_sims, 1)),
        },
        np.random.default_rng(6),
    )
    oreb = drawn["OFF"]["OREB"]
    assert np.array_equal(oreb.sum(axis=1), np.full(n_sims, 20))
    assert oreb[:, 0].mean() / 20.0 == pytest.approx(0.75, abs=0.03)
    assert np.all(drawn["DEF"]["DREB"][:, 1] == 0)


def test_missing_history_uses_the_fitted_fallback():
    model = fit_events(training_events(_logs()))
    assert model.ast_per_min == pytest.approx(0.2)
    assert model.p_rebound == pytest.approx(0.9)
    n_sims = 1500
    blank = pd.DataFrame([{"player_id": 7}, {"player_id": 8}])
    other = pd.DataFrame([{"player_id": 9}])
    drawn = allocate_events(
        model,
        {"HOM": blank, "AWY": other},
        {
            "HOM": _shots(n_sims, [5, 5], [0, 0], [0, 0]),
            "AWY": _shots(n_sims, [0], [0], [4]),
        },
        {
            "HOM": np.tile(np.array([[20.0, 20.0]]), (n_sims, 1)),
            "AWY": np.ones((n_sims, 1)) * 30.0,
        },
        np.random.default_rng(7),
    )
    misses = 10.0
    expected = misses * model.p_rebound * model.p_oreb
    assert drawn["HOM"]["OREB"].sum(axis=1).mean() == pytest.approx(expected, abs=0.4)
    assert drawn["HOM"]["OREB"][:, 0].mean() == pytest.approx(
        drawn["HOM"]["OREB"][:, 1].mean(), abs=0.3
    )


def test_conflicting_team_assists_raise():
    logs = pd.DataFrame([_row(player_id=1, team_ast=20.0), _row(player_id=2, team_ast=10.0)])
    with pytest.raises(ValueError, match="team_ast"):
        training_events(logs)


def test_allocation_rejects_shots_that_do_not_match_minutes():
    model = fit_events(training_events(_logs()))
    profile = pd.DataFrame([{"player_id": 1}, {"player_id": 2}])
    with pytest.raises(ValueError, match="FGA"):
        allocate_events(
            model,
            {"HOM": profile, "AWY": profile.iloc[:1]},
            {
                "HOM": _shots(3, [1, 1], [0, 0], [0, 0]),
                "AWY": _shots(3, [1], [0], [0]),
            },
            {
                "HOM": np.ones((4, 2)),
                "AWY": np.ones((4, 1)),
            },
            np.random.default_rng(8),
        )
