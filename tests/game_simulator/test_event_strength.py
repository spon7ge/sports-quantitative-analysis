"""Per-rate shrinkage is chosen by the next game's predictive likelihood.

Count rates use a gamma-Poisson predictive. Attempt rates use a beta-binomial
predictive. A saved model may still store one number; that number applies to
every rate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from game_events.model import EventModel
from game_events.strength import (
    deviance_by_exposure,
    predictive_attempt_nll,
    predictive_nll,
    tune_attempt_strength,
    tune_strength,
)
from scipy import stats
from shared_events import (
    fit_events,
    load_event_model,
    save_event_model,
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


def _model(**overrides) -> EventModel:
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


def test_predictive_nll_matches_the_negative_binomial():
    n = np.array([4.0, 0.0])
    d = np.array([10.0, 5.0])
    t = np.array([0.2, 0.5])
    x_f = np.array([3.0, 1.0])
    e_f = np.array([8.0, 4.0])
    k = 1.0
    alpha = n + k * t
    beta = d + k
    probability = beta / (beta + e_f)
    expected = -float(stats.nbinom.logpmf(x_f, alpha, probability).sum())
    assert predictive_nll(0.0, n, d, t, x_f, e_f) == pytest.approx(expected)


def test_attempt_nll_matches_the_beta_binomial():
    n = np.array([4.0, 1.0])
    d = np.array([10.0, 8.0])
    t = np.array([0.3, 0.4])
    x_f = np.array([3.0, 2.0])
    e_f = np.array([8.0, 5.0])
    k = 2.0
    alpha = n + k * t
    beta = (d - n) + k * (1.0 - t)
    expected = -float(stats.betabinom.logpmf(x_f, e_f, alpha, beta).sum())
    assert predictive_attempt_nll(np.log(k), n, d, t, x_f, e_f) == pytest.approx(
        expected
    )


def test_strength_stays_small_when_the_next_games_match_the_raw_rate():
    # Prior window is 0 events in 50 minutes. The target is 1 per minute.
    # The next games have 0 events, so shrinkage toward the target hurts.
    rows = 40
    strength = tune_strength(
        np.zeros(rows),
        np.full(rows, 50.0),
        np.ones(rows),
        np.zeros(rows),
        np.full(rows, 10.0),
    )
    assert strength == pytest.approx(np.exp(-2.0), rel=1e-3)


def test_strength_grows_when_the_next_games_match_the_target():
    # Prior window is empty of events. The next games land on the target rate.
    rows = 40
    strength = tune_strength(
        np.zeros(rows),
        np.full(rows, 50.0),
        np.full(rows, 0.5),
        np.full(rows, 5.0),
        np.full(rows, 10.0),
    )
    assert strength == pytest.approx(np.exp(10.0), rel=1e-3)


def test_attempt_strength_grows_when_makes_match_the_target():
    rows = 30
    strength = tune_attempt_strength(
        np.zeros(rows),
        np.full(rows, 40.0),
        np.full(rows, 0.5),
        np.full(rows, 10.0),
        np.full(rows, 20.0),
    )
    assert strength == pytest.approx(np.exp(10.0), rel=1e-3)


def test_a_scalar_prior_still_loads_as_every_rate():
    model = _model()
    payload = {
        **model.__dict__,
        "player_prior_minutes": 45.0,
        "pool_dispersion": {},
        "share_concentration": {},
        "tuning": {},
    }
    loaded = EventModel(**payload)
    assert loaded.player_prior_minutes["ast_per_min"] == pytest.approx(45.0)
    assert loaded.player_prior_minutes["blk_per_min"] == pytest.approx(45.0)


def test_each_rate_uses_its_own_strength():
    quiet = {"oreb": 0.0, "dreb": 0.0, "stl": 0.0, "pos": "PG", "start_position": ""}
    logs = pd.DataFrame(
        [
            _row(
                player_id=1,
                game_id="g1",
                game_date="2026-11-01",
                min=20.0,
                ast=10.0,
                blk=0.0,
                team_ast=10.0,
                team_blk=6.0,
                **quiet,
            ),
            _row(
                player_id=2,
                game_id="g1",
                game_date="2026-11-01",
                min=30.0,
                ast=0.0,
                blk=6.0,
                team_ast=10.0,
                team_blk=6.0,
                **quiet,
            ),
            _row(
                player_id=1,
                game_id="g2",
                game_date="2026-11-15",
                min=20.0,
                ast=0.0,
                blk=0.0,
                team_ast=10.0,
                team_blk=6.0,
                **quiet,
            ),
        ]
    )
    # League before game 2 is 10 assists and 6 blocks in 50 minutes.
    model = _model(
        player_prior_minutes={"ast_per_min": 60.0, "blk_per_min": 600.0},
    )
    game2 = training_events(logs, model)
    game2 = game2.loc[game2["player_id"].eq("1") & game2["game_id"].eq("g2")].iloc[0]
    assert game2["ast_per_min"] == pytest.approx((10.0 + 60.0 * 0.2) / 80.0)
    assert game2["blk_per_min"] == pytest.approx((0.0 + 600.0 * 0.12) / 620.0)


def test_an_open_window_keeps_every_earlier_game():
    rows = []
    for index in range(3):
        rows.append(
            _row(
                player_id=1,
                game_id=f"g{index}",
                game_date=f"2026-11-{index + 1:02d}",
                min=10.0,
                ast=10.0 if index == 0 else 0.0,
                oreb=0.0,
                dreb=0.0,
                stl=0.0,
                blk=0.0,
            )
        )
    logs = pd.DataFrame(rows)
    recent = training_events(logs, _model(player_window=1))
    career = training_events(logs, _model(player_window=None))
    last_recent = recent.loc[recent["game_id"].eq("g2")].iloc[0]
    last_career = career.loc[career["game_id"].eq("g2")].iloc[0]
    # Window 1 sees only the scoreless middle game. The open window also
    # sees the 10 assists from the first game. League history is the same
    # 10 assists in 20 minutes either way.
    assert last_recent["ast_per_min"] == pytest.approx((0.0 + 60.0 * 0.5) / 70.0)
    assert last_career["ast_per_min"] == pytest.approx(0.5)


def test_saved_scalar_round_trip_fills_every_rate(tmp_path):
    model = fit_events(_two_team_logs(["2026-11-01", "2026-11-02"]), tune=False)
    path = tmp_path / "events.json"
    save_event_model(model, path)
    payload = json.loads(path.read_text())
    payload["player_prior_minutes"] = 45.0
    path.write_text(json.dumps(payload))
    loaded = load_event_model(path)
    assert set(loaded.player_prior_minutes) == {
        "ast_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
    }
    assert loaded.player_prior_minutes["stl_per_min"] == pytest.approx(45.0)


def _two_team_logs(dates, shocked=()):
    """Home pair plus an away player, so every team rate has a denominator."""
    shocked = set(shocked)
    home = dict(
        team_id=10,
        team_fga=40.0,
        team_fgm=20.0,
        team_ast=15.0,
        team_oreb=4.0,
        team_stl=4.0,
        team_blk=2.0,
        opp_dreb=8.0,
        opp_tov=8.0,
        opp_fga=50.0,
        pos="PG",
        start_position="",
    )
    away = dict(
        team_id=20,
        team_fga=50.0,
        team_fgm=30.0,
        team_ast=10.0,
        team_oreb=8.0,
        team_stl=3.0,
        team_blk=2.0,
        opp_dreb=4.0,
        opp_tov=10.0,
        opp_fga=40.0,
        pos="",
        start_position="",
    )
    rows = []
    for index, day in enumerate(dates):
        shock = day in shocked
        rows.append(
            _row(
                player_id=1,
                game_id=f"g{index}",
                game_date=day,
                min=10.0,
                ast=5.0 if shock else 0.0,
                oreb=0.0,
                dreb=0.0,
                stl=0.0,
                blk=0.0,
                **home,
            )
        )
        rows.append(
            _row(
                player_id=2,
                game_id=f"g{index}",
                game_date=day,
                min=10.0,
                ast=10.0,
                oreb=0.0,
                dreb=0.0,
                stl=0.0,
                blk=1.0,
                **home,
            )
        )
        rows.append(
            _row(
                player_id=3,
                game_id=f"g{index}",
                game_date=day,
                min=10.0,
                ast=0.0,
                oreb=0.0,
                dreb=0.0,
                stl=0.0,
                blk=0.0,
                **away,
            )
        )
    return pd.DataFrame(rows)


def test_strengths_are_fit_on_training_dates_and_dispersion_is_refit():
    dates = [f"2026-01-{day:02d}" for day in range(1, 17)]
    # The last half lands on the league rate. The first half is scoreless,
    # which wants almost no shrinkage. Those later games are the holdout.
    logs = _two_team_logs(dates, dates[8:])
    model = fit_events(logs, validation_fraction=0.5)
    assert isinstance(model.player_prior_minutes, dict)
    assert model.player_prior_minutes["ast_per_min"] < 1.0
    assert model.player_window in (20, 40, 82, None)
    assert set(model.pool_dispersion) == {
        "assisted_rate",
        "p_rebound",
        "p_oreb",
        "dreb_given_oreb",
        "p_steal",
        "p_block",
    }
    assert set(model.share_concentration) == {"AST", "OREB", "DREB", "STL", "BLK"}
    assert model.tuning["tuned"] is True


def test_deviance_is_split_by_prior_exposure():
    n = np.array([0.0, 0.0])
    d = np.array([50.0, 1500.0])
    t = np.array([0.2, 0.2])
    x_f = np.array([1.0, 2.0])
    e_f = np.array([10.0, 10.0])
    k = np.exp(0.0)
    alpha = n + k * t
    beta = d + k
    probability = beta / (beta + e_f)
    nll = -stats.nbinom.logpmf(x_f, alpha, probability)
    buckets = deviance_by_exposure(n, d, t, x_f, e_f, 0.0)
    assert buckets["under_200"] == pytest.approx(float(2.0 * nll[0]))
    assert buckets["200_to_1000"] is None
    assert buckets["over_1000"] == pytest.approx(float(2.0 * nll[1]))
