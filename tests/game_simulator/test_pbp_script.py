"""Layer 1 pace and overtime use play-by-play, not the box estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from possession_table import possession_table

from game_enviroment.pbp_script import (
    apply_possession_counts,
    box_possession_estimate,
    choose_ot_blend,
    fit_ot_path,
    fit_pace_state,
    late_checkpoints,
    late_margin_paths,
    ot_path_probability,
    script_tables,
)
from game_enviroment.script import _draw

GAME = "0022500001"
HOME, AWAY = 100, 200
NAMES = {
    1: "Alpha",
    2: "Bravo",
    3: "Charlie",
    4: "Delta",
    5: "Echo",
    6: "Foxtrot",
    7: "Golf",
    8: "Hector",
    9: "India",
    10: "Juliet",
}


def starters():
    return pd.DataFrame(
        [
            {
                "game_id": GAME,
                "team_id": HOME if player < 6 else AWAY,
                "player_id": player,
                "player_name": NAMES[player],
            }
            for player in range(1, 11)
        ]
    )


def ev(**kwargs):
    team = kwargs.get("team", 0)
    number = kwargs["n"]
    return {
        "gameId": kwargs.get("game", GAME),
        "actionNumber": number,
        "clock": kwargs.get("clock", "PT12M00.00S"),
        "period": kwargs.get("period", 1),
        "teamId": team,
        "personId": kwargs.get("player", 0),
        "playerName": kwargs.get("name", ""),
        "teamTricode": "HOM" if team == HOME else "AWY" if team == AWAY else "",
        "xLegacy": kwargs.get("x", 0),
        "yLegacy": kwargs.get("y", 0),
        "shotDistance": kwargs.get("dist", 0),
        "scoreHome": kwargs.get("home", ""),
        "scoreAway": kwargs.get("away", ""),
        "location": "h" if team == HOME else "v" if team == AWAY else "",
        "description": kwargs.get("description", ""),
        "actionType": kwargs.get("action", ""),
        "subType": kwargs.get("sub", ""),
        "shotValue": kwargs.get("value", 0),
    }


def _made(number, team, player, name, clock, home, away):
    return ev(
        n=number,
        action="Made Shot",
        sub="Layup Shot",
        team=team,
        player=player,
        name=name,
        clock=clock,
        value=2,
        dist=2,
        y=10,
        home=home,
        away=away,
        description=f"{name} 2' Layup",
    )


def _two_trips():
    return [
        ev(n=1, action="period", sub="start", description="Start of 1st Period"),
        ev(
            n=2,
            action="Jump Ball",
            team=HOME,
            player=1,
            name="Alpha",
            description="Jump Ball Alpha vs. Foxtrot: Tip to Alpha",
        ),
        _made(3, HOME, 1, "Alpha", "PT11M40.00S", "2", "0"),
        _made(4, AWAY, 6, "Foxtrot", "PT11M20.00S", "2", "2"),
        ev(n=5, action="period", sub="end", clock="PT00M00.00S"),
    ]


def _rows(game, half, margin, seconds, n):
    return pd.DataFrame(
        {
            "game_id": game,
            "half": half,
            "offense_margin": margin,
            "seconds": seconds,
        },
        index=range(n),
    )


def test_box_estimate_uses_the_free_throw_weight():
    assert box_possession_estimate(80, 20, 10, 12) == pytest.approx(80 + 0.44 * 20 - 10 + 12)


def test_play_by_play_count_replaces_the_box_estimate():
    rows = _two_trips()
    counted = possession_table(pd.DataFrame(rows), starters())
    home_trips = int((counted.offense_team_id == HOME).sum())
    away_trips = int((counted.offense_team_id == AWAY).sum())
    estimate = float(box_possession_estimate(80, 20, 10, 12))
    assert home_trips != pytest.approx(estimate)

    tables = script_tables(pd.DataFrame(rows), starters())
    counts = tables["counts"]
    assert int(counts.loc[counts.team_id.eq(HOME), "poss"].iloc[0]) == home_trips
    assert int(counts.loc[counts.team_id.eq(AWAY), "poss"].iloc[0]) == away_trips

    games = pd.DataFrame(
        {
            "game_id": [GAME],
            "home_id": [HOME],
            "away_id": [AWAY],
            "home_poss": [estimate],
            "away_poss": [estimate],
            "nominal_min": [48.0],
            "poss": [estimate],
            "poss48": [estimate],
        }
    )
    # Scale the clock length so a two-trip sample is not rejected as junk pace.
    games["nominal_min"] = 48.0 * (home_trips + away_trips) / 2.0 / 100.0
    replaced = apply_possession_counts(games, counts)

    assert replaced.loc[0, "poss_source"] == "pbp"
    assert replaced.loc[0, "home_poss"] == home_trips
    assert replaced.loc[0, "away_poss"] == away_trips
    assert replaced.loc[0, "poss48"] == pytest.approx(100.0)


def test_a_junk_count_keeps_the_box_estimate():
    estimate = float(box_possession_estimate(80, 20, 10, 12))
    games = pd.DataFrame(
        {
            "game_id": [GAME, "0022500002"],
            "home_id": [HOME, HOME],
            "away_id": [AWAY, AWAY],
            "home_poss": [estimate, estimate],
            "away_poss": [estimate, 101.0],
            "nominal_min": [48.0, 48.0],
            "poss": [estimate, estimate],
            "poss48": [estimate, estimate],
        }
    )
    counts = pd.DataFrame(
        {
            "game_id": [GAME, GAME, "0022500002"],
            "team_id": [HOME, AWAY, HOME],
            "poss": [1, 1, 96],
        }
    )

    replaced = apply_possession_counts(games, counts)

    assert replaced.loc[0, "poss_source"] == "box"
    assert replaced.loc[0, "home_poss"] == pytest.approx(estimate)
    assert replaced.loc[1, "poss_source"] == "box"
    assert replaced.loc[1, "away_poss"] == pytest.approx(101.0)


def test_a_game_that_does_not_parse_is_skipped_and_reported():
    good = pd.DataFrame(_two_trips())
    bad = pd.DataFrame([ev(n=1, game="0022500099", action="period", sub="start")])
    tables = script_tables(pd.concat([good, bad], ignore_index=True), starters())

    assert set(tables["counts"].game_id) == {GAME}
    assert tables["failed"] == [
        {"game_id": "0022500099", "error": "0022500099 expected two teams, found []"}
    ]


def test_leader_holds_the_ball_longer_and_halves_differ():
    exposure = pd.concat(
        [
            _rows("g1", 1, 0, 10.0, 80),
            _rows("g1", 2, 0, 16.0, 80),
            _rows("g1", 2, 10, 24.0, 80),
            _rows("g1", 2, -10, 12.0, 80),
            _rows("holdout", 1, 0, 40.0, 80),
            _rows("holdout", 2, 0, 40.0, 80),
        ],
        ignore_index=True,
    )
    fitted = fit_pace_state(exposure, ["g1"])
    multiplier = np.asarray(fitted["multiplier"])
    lead = fitted["lead_seconds"][3]
    trail = fitted["trail_seconds"][3]

    assert lead > trail
    assert multiplier[0, 0] > multiplier[1, 0]
    assert (multiplier[0, 0] + multiplier[1, 0]) / 2 == pytest.approx(1.0)
    # Margin 10 is the 8-12 bucket. Those trips take longer than a tied trip.
    assert multiplier[1, 3] < multiplier[1, 0]
    assert fit_pace_state(exposure)["multiplier"] != fitted["multiplier"]


def test_late_checkpoints_follow_the_second_half():
    margin_6, margin_2 = late_checkpoints(8.0, 2.0)
    assert margin_6 == pytest.approx(5.0)
    assert margin_2 == pytest.approx(3.0)


def test_late_margin_path_reads_the_clock_and_the_final_score():
    rows = [
        ev(n=1, period=4, clock="PT06M05.00S", home="20", away="10", action="Made Shot"),
        ev(n=2, period=4, clock="PT05M50.00S", action="Timeout"),
        ev(n=3, period=4, clock="PT02M10.00S", home="24", away="20", action="Made Shot"),
        ev(n=4, period=4, clock="PT01M50.00S", action="Timeout"),
        ev(n=5, period=4, clock="PT00M00.00S", home="28", away="26", action="period", sub="end"),
    ]
    path = late_margin_paths(pd.DataFrame(rows)).iloc[0]

    assert path.margin_6 == 10
    assert path.margin_2 == 4
    assert path.reg_margin == 2
    assert path.final_margin == 2
    assert path.went_ot == 0


def test_overtime_probability_uses_the_path_and_the_final_score():
    close = _paths(0, 0, 0, 1, 40, "train")
    close_reg = _paths(0, 0, 0, 0, 40, "train")
    blowout = _paths(15, 15, 15, 0, 40, "train")
    fitted = fit_ot_path(pd.concat([close, close_reg, blowout], ignore_index=True))
    edges = np.asarray(fitted["edges"])
    close_bucket = int(np.searchsorted(edges, 0, side="right") - 1)
    far_bucket = int(np.searchsorted(edges, 15, side="right") - 1)

    assert fitted["trajectory"][close_bucket][close_bucket] > fitted["trajectory"][far_bucket][far_bucket]
    assert fitted["final"][close_bucket] > fitted["final"][far_bucket]

    both = ot_path_probability(
        {**fitted, "blend": 0.5},
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    far_path = ot_path_probability({**fitted, "blend": 1.0}, 15.0, 15.0, 0.0)
    close_final = ot_path_probability({**fitted, "blend": 0.0}, 15.0, 15.0, 0.0)
    assert both[0] == pytest.approx(
        0.5 * fitted["trajectory"][close_bucket][close_bucket]
        + 0.5 * fitted["final"][close_bucket]
    )
    assert far_path < both[0]
    assert close_final > far_path

    validation = pd.concat(
        [
            _paths(0, 0, 15, 1, 20, "val"),
            _paths(15, 15, 0, 0, 20, "val"),
        ],
        ignore_index=True,
    )
    blend = choose_ot_blend(
        fitted,
        pd.concat([close, close_reg, blowout, validation], ignore_index=True),
        validation.game_id,
    )
    assert blend == 1.0


def test_draw_slows_a_lead_and_overtime_follows_the_simulated_path():
    edges = [0.0, 1.0, 4.0, 8.0, 13.0, 21.0, 1.0e9]
    slow = [1.0, 1.0, 0.6, 0.6, 0.6, 0.6]
    pace_state = {
        "edges": edges,
        "multiplier": [[1.0] * 6, slow],
    }
    ot_path = {
        "edges": edges,
        "trajectory": _matrix(1.0, 0.0),
        "final": [0.5] * 6,
        "blend": 1.0,
    }
    model = _state_model(pace_state, ot_path)
    tied = _draw(model, _row(1.1, 1.1), 30, np.random.default_rng(1))
    lead = _draw(model, _row(1.5, 0.5), 30, np.random.default_rng(1))

    assert np.allclose(tied["n_ot"], 1)
    assert np.allclose(tied["home_poss"], 100.0 * 53.0 / 48.0)
    assert np.allclose(lead["n_ot"], 0)
    assert np.allclose(lead["home_poss"], 80.0)
    assert lead["home_poss"].mean() < tied["home_poss"].mean()


def test_state_pace_spreads_possession_totals():
    edges = [0.0, 1.0, 4.0, 8.0, 13.0, 21.0, 1.0e9]
    flat = {"edges": edges, "multiplier": [[1.0] * 6, [1.0] * 6]}
    varied = {"edges": edges, "multiplier": [[1.0] * 6, [1.0, 0.2, 0.2, 0.2, 0.2, 0.2]]}
    ot_path = {"edges": edges, "trajectory": _matrix(0.0, 0.0), "final": [0.0] * 6, "blend": 1.0}
    model = _state_model(flat, ot_path)
    model["scoring"]["residual"] = {"nu": 30.0, "scale": 8.0, "rho": 0.0}
    steady = _draw(model, _row(1.1, 1.1), 800, np.random.default_rng(3))
    model["pace_state"] = varied
    swung = _draw(model, _row(1.1, 1.1), 800, np.random.default_rng(3))

    assert steady["home_poss"].std() < 1e-6
    assert swung["home_poss"].std() > 5
    assert swung["game_total"].std() > steady["game_total"].std()


def _paths(margin_6, margin_2, final_margin, went_ot, n, prefix):
    return pd.DataFrame(
        {
            "game_id": [f"{prefix}-{i}" for i in range(n)],
            "margin_6": margin_6,
            "margin_2": margin_2,
            "final_margin": final_margin,
            "went_ot": went_ot,
        }
    )


def _matrix(tied, other):
    table = [[other] * 6 for _ in range(6)]
    table[0][0] = tied
    return table


def _row(home_ppp, away_ppp):
    return pd.DataFrame(
        {
            "lg_poss": [100.0],
            "home_spread": [0.0],
            "total": [np.nan],
            "lineup_off_rating_h": [home_ppp],
            "lineup_off_rating_a": [away_ppp],
            "lineup_def_rating_h": [0.0],
            "lineup_def_rating_a": [0.0],
        }
    )


def _state_model(pace_state, ot_path):
    return {
        "residual_t": {"nu": 30.0, "scale": 0.0},
        "model": {"name": "league_prior", "team_ids": [], "beta": [], "lam": 0.0},
        "pace_state": pace_state,
        "ot_path": ot_path,
        "ot": {
            "p_ot_overall": 0.0,
            "p_ot_by_spread_bucket": [0.0, 0.0, 0.0, 0.0],
            "p_more_given_ot": 0.0,
        },
        "split": {"values": [0.0], "probs": [1.0]},
        "scoring": {
            "beta": [0.0, 1.0, 0.0, 0.0],
            "off_center": 0.0,
            "def_center": 0.0,
            "market_weight": 0.0,
            "residual": {"nu": 30.0, "scale": 0.0, "rho": 0.0},
        },
    }
