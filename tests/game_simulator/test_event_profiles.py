"""Layer 5 assigns assists, rebounds, steals, and blocks from events already drawn.

Training rates stop before the game being scored. Live rates keep the latest
completed game. Shots and turnovers are inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shared_events import (
    EventModel,
    allocate_events,
    fit_events,
    live_events,
    save_event_model,
    load_event_model,
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
                oreb=1.0,
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


def _model(**overrides) -> EventModel:
    """League rates at 0 make log5 return the profile rate unchanged."""
    values = {
        "version": 2,
        "cutoff": "2024-01-01",
        "ast_per_min": 0.2,
        "oreb_per_min": 0.1,
        "dreb_per_min": 0.3,
        "stl_per_min": 0.05,
        "blk_per_min": 0.05,
        "assisted_rate": 0.5,
        "p_rebound": 0.9,
        "p_oreb": 0.0,
        "p_oreb_allowed": 0.0,
        "p_steal": 0.0,
        "p_steal_against": 0.0,
        "p_block": 0.0,
        "p_block_against": 0.0,
    }
    values.update(overrides)
    return EventModel(**values)


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
    # Blocks are the opponent's misses. This log has no opponent row.
    assert np.isnan(game2["p_block"])
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
    assert live["ast_per_min"] == pytest.approx((4.0 + 8.0) / 40.0)
    assert live["assisted_rate"] == pytest.approx((20.0 + 16.0) / (40.0 + 20.0))
    assert live["p_rebound"] == pytest.approx((9.0 + 10.0) / (10.0 + 10.0))
    assert live["p_oreb"] == pytest.approx((3.0 + 2.0) / (9.0 + 10.0))
    assert live["p_steal"] == pytest.approx((5.0 + 8.0) / (10.0 + 10.0))
    assert np.isnan(live["p_block"])


def test_player_rates_shrink_toward_the_same_position():
    """A center with no minutes takes the center rate, not the league rate.

    2025-26 centers blocked about 0.042 per minute and the league about
    0.020. Power forwards were near the league rate, so they stay their
    own position. A blank start_position does not erase a listed position.
    """
    prior = "2026-11-01"
    upcoming = "2026-11-15"
    quiet = {
        "ast": 0.0,
        "oreb": 0.0,
        "dreb": 0.0,
        "stl": 0.0,
        "team_blk": 6.0,
    }
    logs = pd.DataFrame(
        [
            _row(
                player_id=1,
                game_id="g1",
                game_date=prior,
                min=30.0,
                blk=6.0,
                pos="C",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=2,
                game_id="g1",
                game_date=prior,
                min=30.0,
                blk=0.0,
                pos="PG",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=3,
                game_id="g1",
                game_date=prior,
                min=30.0,
                blk=1.5,
                pos="PF",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=9,
                game_id="g1",
                game_date=prior,
                min=30.0,
                blk=6.0,
                pos="",
                start_position="G",
                **quiet,
            ),
            _row(
                player_id=4,
                game_id="g2",
                game_date=upcoming,
                min=0.0,
                blk=0.0,
                pos="C",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=5,
                game_id="g2",
                game_date=upcoming,
                min=0.0,
                blk=0.0,
                pos="PG",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=6,
                game_id="g2",
                game_date=upcoming,
                min=0.0,
                blk=0.0,
                pos="PF",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=7,
                game_id="g2",
                game_date=upcoming,
                min=0.0,
                blk=0.0,
                pos="",
                start_position="",
                **quiet,
            ),
            _row(
                player_id=10,
                game_id="g2",
                game_date=upcoming,
                min=0.0,
                blk=0.0,
                pos="",
                start_position="G",
                **quiet,
            ),
            _row(
                player_id=11,
                game_id="g2",
                game_date=upcoming,
                min=0.0,
                blk=0.0,
                pos="",
                start_position="C",
                **quiet,
            ),
        ]
    )
    trained = training_events(logs)
    upcoming_rows = trained.loc[trained["game_id"].eq("g2")].set_index("player_id")
    # 6/30. The league pool is (6 + 0 + 1.5 + 6) / 120 = 0.1125.
    assert upcoming_rows.loc["4", "blk_per_min"] == pytest.approx(0.2)
    assert upcoming_rows.loc["5", "blk_per_min"] == pytest.approx(0.0)
    assert upcoming_rows.loc["6", "blk_per_min"] == pytest.approx(1.5 / 30.0)
    assert upcoming_rows.loc["7", "blk_per_min"] == pytest.approx(13.5 / 120.0)
    # Start slot G pools point guards and the unlabeled guard.
    assert upcoming_rows.loc["10", "blk_per_min"] == pytest.approx((0.0 + 6.0) / 60.0)
    assert upcoming_rows.loc["11", "blk_per_min"] == pytest.approx(0.2)


def test_current_season_outweighs_earlier_seasons():
    """A 2026-27 slate weights that season above 2025-26 and 2024-25.

    Per-minute weights are 1, 0.1, and 0.01. Equal weights would leave the
    rookie at 12/60 = 0.2. Games after the slate date stay out.
    """
    quiet = {"oreb": 0.0, "dreb": 0.0, "stl": 0.0, "blk": 0.0, "pos": "PG"}
    logs = pd.DataFrame(
        [
            _row(
                player_id=1,
                game_id="old",
                game_date="2024-11-01",
                min=20.0,
                ast=0.0,
                **quiet,
            ),
            _row(
                player_id=1,
                game_id="mid",
                game_date="2025-11-01",
                min=20.0,
                ast=4.0,
                **quiet,
            ),
            _row(
                player_id=1,
                game_id="now",
                game_date="2026-11-01",
                min=20.0,
                ast=8.0,
                **quiet,
            ),
            _row(
                player_id=1,
                game_id="later",
                game_date="2026-12-01",
                min=20.0,
                ast=0.0,
                **quiet,
            ),
            _row(
                player_id=9,
                game_id="slate",
                game_date="2026-11-15",
                min=0.0,
                ast=0.0,
                **quiet,
            ),
        ]
    )
    trained = training_events(logs)
    rookie = trained.loc[trained["player_id"].eq("9")].iloc[0]
    # 0*0.01 + 4*0.1 + 8*1 over 20*0.01 + 20*0.1 + 20*1.
    assert rookie["ast_per_min"] == pytest.approx(8.4 / 22.2)
    veteran = trained.loc[trained["game_id"].eq("now")].iloc[0]
    # His own prior is 4 assists in 40 minutes. December is not in yet.
    # League target is 0.4/2.2, shrunk with 60 prior minutes.
    assert veteran["ast_per_min"] == pytest.approx(41.0 / 275.0)


def test_team_rates_use_the_same_season_weights():
    logs = pd.DataFrame(
        [
            _row(
                game_id="old",
                game_date="2024-11-01",
                ast=0.0,
                team_ast=0.0,
                team_fgm=40.0,
                team_fga=50.0,
            ),
            _row(
                game_id="mid",
                game_date="2025-11-01",
                ast=0.0,
                team_ast=20.0,
                team_fgm=40.0,
                team_fga=50.0,
            ),
            _row(game_id="now", game_date="2026-11-01"),
        ]
    )
    trained = training_events(logs)
    current = trained.loc[trained["game_id"].eq("now")].iloc[0]
    # League assists/makes = (0*0.01 + 20*0.1) / (40*0.01 + 40*0.1) = 2/4.4.
    # Team window is both prior games, 20/80, shrunk with 120 trials.
    assert current["assisted_rate"] == pytest.approx(41.0 / 110.0)


def test_a_center_prior_decays_older_center_seasons():
    quiet = {"ast": 0.0, "oreb": 0.0, "dreb": 0.0, "stl": 0.0, "pos": "C"}
    logs = pd.DataFrame(
        [
            _row(
                player_id=1,
                game_id="old",
                game_date="2025-11-01",
                min=30.0,
                blk=0.0,
                team_blk=6.0,
                **quiet,
            ),
            _row(
                player_id=1,
                game_id="now",
                game_date="2026-11-01",
                min=30.0,
                blk=6.0,
                team_blk=6.0,
                **quiet,
            ),
            _row(
                player_id=9,
                game_id="slate",
                game_date="2026-11-15",
                min=0.0,
                blk=0.0,
                team_blk=6.0,
                **quiet,
            ),
        ]
    )
    trained = training_events(logs)
    rookie = trained.loc[trained["player_id"].eq("9")].iloc[0]
    # (0*0.1 + 6*1) / (30*0.1 + 30*1). Equal weights would be 6/60.
    assert rookie["blk_per_min"] == pytest.approx(6.0 / 33.0)


def test_live_slate_inherits_a_listed_position_from_earlier_games():
    logs = pd.DataFrame(
        [
            _row(
                player_id=1,
                game_date="2026-11-01",
                min=30.0,
                blk=6.0,
                ast=0.0,
                oreb=0.0,
                dreb=0.0,
                stl=0.0,
                team_blk=6.0,
                pos="C",
                start_position="",
            ),
            _row(
                player_id=2,
                game_date="2026-11-01",
                min=30.0,
                blk=0.0,
                ast=0.0,
                oreb=0.0,
                dreb=0.0,
                stl=0.0,
                team_blk=6.0,
                pos="PG",
                start_position="",
            ),
        ]
    )
    slate = pd.DataFrame(
        [{"player_id": 1, "game_date": "2026-11-15", "team_id": 10}]
    )
    live = live_events(logs, slate).iloc[0]
    # Own 6/30 shrunk toward centers (6/30), not the 6/60 league rate.
    assert live["blk_per_min"] == pytest.approx(0.2)


def test_live_rates_stop_before_the_slate_date():
    future = _row(
        game_id="g3",
        game_date="2024-02-01",
        ast=99.0,
        min=10.0,
        team_ast=99.0,
        team_fgm=99.0,
        team_fga=100.0,
    )
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
    model = _model()
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
    model = _model()
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
    model = _model()
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
    model = _model(p_oreb=1.0 / 3.0, p_oreb_allowed=1.0 / 3.0)
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
    model = _model()
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


def _paired_logs() -> pd.DataFrame:
    """Two games, both teams, so opponent misses are the block denominator."""
    home = dict(
        player_id=1,
        team_id=10,
        min=20.0,
        ast=4.0,
        oreb=1.0,
        dreb=5.0,
        stl=1.0,
        blk=1.0,
        team_fga=40.0,
        team_fgm=20.0,
        team_ast=10.0,
        team_oreb=4.0,
        team_stl=4.0,
        team_blk=2.0,
        opp_dreb=8.0,
        opp_tov=8.0,
        opp_fga=50.0,
    )
    away = dict(
        player_id=2,
        team_id=20,
        min=30.0,
        ast=6.0,
        oreb=2.0,
        dreb=7.0,
        stl=2.0,
        blk=3.0,
        team_fga=50.0,
        team_fgm=30.0,
        team_ast=15.0,
        team_oreb=8.0,
        team_stl=3.0,
        team_blk=4.0,
        opp_dreb=4.0,
        opp_tov=10.0,
        opp_fga=40.0,
    )
    rows = []
    for game_id, game_date, home_ast, home_blk in (
        ("g1", "2024-01-01", 4.0, 2.0),
        ("g2", "2024-01-15", 8.0, 10.0),
    ):
        rows.append(
            {**home, "game_id": game_id, "game_date": game_date, "ast": home_ast, "team_blk": home_blk}
        )
        rows.append({**away, "game_id": game_id, "game_date": game_date})
    return pd.DataFrame(rows)


def test_blocks_use_the_opponents_misses_and_stop_before_the_game():
    trained = training_events(_paired_logs())
    game2 = trained.loc[
        trained["game_id"].eq("g2") & trained["player_id"].eq("1")
    ].iloc[0]
    # Game 2's own assist rate would be 8/20. The rate stops at game 1.
    assert game2["ast_per_min"] == pytest.approx(0.2)
    # Game 1 is 2 blocks on 20 opponent misses, shrunk toward the 0.15 league rate.
    assert game2["p_block"] == pytest.approx(23.0 / 160.0)
    opening = trained.loc[trained["game_id"].eq("g1")]
    assert opening["p_block"].isna().all()


def test_fit_events_uses_exposure_sums_from_both_teams():
    model = fit_events(_paired_logs(), tune=False)
    assert model.cutoff == "2024-01-15"
    assert model.ast_per_min == pytest.approx(24.0 / 100.0)
    assert model.assisted_rate == pytest.approx(0.5)
    assert model.p_block == pytest.approx(0.25)
    assert model.tuning["tuned"] is False
    assert all(value == 0.0 for value in model.pool_dispersion.values())


def test_live_rates_include_both_completed_games():
    slate = pd.DataFrame(
        [{"player_id": 1, "game_date": "2024-02-01", "team_id": 10}]
    )
    live = live_events(_paired_logs(), slate).iloc[0]
    assert live["ast_per_min"] == pytest.approx((12.0 + 60.0 * 0.24) / 100.0)
    assert live["p_block"] == pytest.approx(47.0 / 180.0)


def test_layer4_assists_and_offensive_rebounds_are_reused():
    n_sims = 2
    supplied_ast = np.array([[1, 0], [0, 2]], dtype=np.int64)
    supplied_oreb = np.array([[1, 0], [0, 1]], dtype=np.int64)
    shots = _shots(n_sims, [5, 5], [4, 4], [1, 0])
    shots["AST"] = supplied_ast
    shots["OREB"] = supplied_oreb
    shots["PTS"] = np.zeros_like(supplied_ast)
    profiles = {
        "OFF": _players(
            {"assisted_rate": 0.0, "p_rebound": 1.0, "p_oreb": 0.0, "p_steal": 0.0, "p_block": 0.0},
            [{"player_id": 1, "ast_per_min": 1.0}, {"player_id": 2, "ast_per_min": 1.0}],
        ),
        "DEF": _players(
            {"assisted_rate": 0.0, "p_rebound": 0.0, "p_oreb": 0.0, "p_steal": 1.0, "p_block": 1.0},
            [{"player_id": 3}],
        ),
    }
    minutes = {"OFF": np.full((n_sims, 2), 20.0), "DEF": np.full((n_sims, 1), 20.0)}
    kept = allocate_events(_model(), profiles, {"OFF": shots, "DEF": _shots(n_sims, [0], [0], [0])}, minutes, np.random.default_rng(1))
    assert np.array_equal(kept["OFF"]["AST"], supplied_ast)
    assert np.array_equal(kept["OFF"]["OREB"], supplied_oreb)
    redrawn = allocate_events(
        _model(),
        profiles,
        {"OFF": shots, "DEF": _shots(n_sims, [0], [0], [0])},
        minutes,
        np.random.default_rng(1),
        resimulate=("AST", "OREB"),
    )
    assert np.all(redrawn["OFF"]["AST"] == 0)
    assert np.all(redrawn["OFF"]["OREB"] == 0)


def test_segments_credit_only_the_defender_who_shared_the_floor():
    n_sims = 3
    offense_ids = ["H0", "H1", "H2", "H3", "H4", "H5"]
    defense_ids = ["D0", "D1", "D2", "D3", "D4", "D5"]

    def span(players, start, end):
        return {"period": 1, "start_min": start, "end_min": end, "players": players}

    offense_schedule = [
        span(offense_ids[:5], 0.0, 6.0),
        span(["H5", "H1", "H2", "H3", "H4"], 6.0, 12.0),
    ]
    defense_schedule = [
        span(defense_ids[:5], 0.0, 6.0),
        span(["D5", "D1", "D2", "D3", "D4"], 6.0, 12.0),
    ]
    offense = _players(
        {"p_rebound": 1.0, "p_oreb": 0.0, "assisted_rate": 0.0, "p_steal": 0.0, "p_block": 0.0},
        [{"player_id": pid, "oreb_per_min": 1.0} for pid in offense_ids],
    )
    defense = _players(
        {"p_rebound": 0.0, "p_oreb": 0.0, "assisted_rate": 0.0, "p_steal": 1.0, "p_block": 1.0},
        [
            {
                "player_id": pid,
                "dreb_per_min": 1.0 if pid in {"D0", "D5"} else 0.0,
                "stl_per_min": 1.0 if pid in {"D0", "D5"} else 0.0,
                "blk_per_min": 1.0 if pid in {"D0", "D5"} else 0.0,
            }
            for pid in defense_ids
        ],
    )
    played = np.array([6.0, 12.0, 12.0, 12.0, 12.0, 6.0])
    drawn, details = allocate_events(
        _model(),
        {"OFF": offense, "DEF": defense},
        {
            "OFF": _shots(n_sims, [10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]),
            "DEF": _shots(n_sims, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        },
        {"OFF": np.tile(played, (n_sims, 1)), "DEF": np.tile(played, (n_sims, 1))},
        np.random.default_rng(9),
        segments={
            "OFF": [offense_schedule] * n_sims,
            "DEF": [defense_schedule] * n_sims,
        },
        return_details=True,
    )
    assert details["segments_used"] is True
    assert np.array_equal(drawn["DEF"]["DREB"][:, 0], np.full(n_sims, 10))
    assert np.array_equal(drawn["DEF"]["STL"][:, 0], np.full(n_sims, 4))
    assert np.array_equal(drawn["DEF"]["BLK"][:, 0], np.full(n_sims, 10))
    assert np.all(drawn["DEF"]["DREB"][:, 1:] == 0)
    assert np.all(drawn["DEF"]["STL"][:, 1:] == 0)
    assert np.all(drawn["DEF"]["BLK"][:, 1:] == 0)


def test_a_zero_minute_player_receives_nothing():
    n_sims = 20
    drawn = allocate_events(
        _model(),
        {
            "OFF": _players(
                {"p_rebound": 1.0, "p_oreb": 1.0, "assisted_rate": 0.0, "p_steal": 0.0, "p_block": 0.0},
                [{"player_id": 1, "oreb_per_min": 5.0}, {"player_id": 2, "oreb_per_min": 5.0}],
            ),
            "DEF": _players(
                {"p_rebound": 0.0, "p_oreb": 0.0, "assisted_rate": 0.0, "p_steal": 0.0, "p_block": 0.0},
                [{"player_id": 3}],
            ),
        },
        {
            "OFF": _shots(n_sims, [4, 0], [0, 0], [0, 0]),
            "DEF": _shots(n_sims, [0], [0], [0]),
        },
        {"OFF": np.tile(np.array([[12.0, 0.0]]), (n_sims, 1)), "DEF": np.ones((n_sims, 1))},
        np.random.default_rng(3),
    )
    assert np.array_equal(drawn["OFF"]["OREB"][:, 0], np.full(n_sims, 4))
    assert np.all(drawn["OFF"]["OREB"][:, 1] == 0)


def test_the_same_player_cannot_appear_on_both_teams():
    profile = pd.DataFrame([{"player_id": 1}])
    with pytest.raises(ValueError, match="both teams"):
        allocate_events(
            _model(),
            {"HOM": profile, "AWY": profile.copy()},
            {"HOM": _shots(1, [1], [0], [0]), "AWY": _shots(1, [1], [0], [0])},
            {"HOM": np.ones((1, 1)), "AWY": np.ones((1, 1))},
            np.random.default_rng(1),
        )


def test_event_model_round_trip(tmp_path):
    model = fit_events(_paired_logs(), tune=False)
    path = tmp_path / "events.json"
    save_event_model(model, path)
    loaded = load_event_model(path)
    assert loaded == model


def test_forecast_scores_a_known_count():
    from shared_events import score_event_forecasts, summarize_event_scores

    stats = {"HOM": {"AST": np.array([[1], [3], [2]])}}
    actual = {"HOM": pd.DataFrame([{"player_id": 7, "ast": 2}])}
    scores = score_event_forecasts(stats, {"HOM": [7]}, actual, game_id="g1")
    assert len(scores) == 1
    assert scores.iloc[0]["actual"] == 2
    assert bool(scores.iloc[0]["covered"]) is True
    summary = summarize_event_scores(scores)
    assert summary.iloc[0]["stat"] == "AST"
    assert summary.iloc[0]["n"] == 1
