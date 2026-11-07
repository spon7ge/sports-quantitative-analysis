"""Prior-only rotation minutes, shared availability, and leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from game_enviroment.rotation import (
    allocate_regulation_minutes,
    build_rotation_history,
    choose_starters,
    classify_status,
    overtime_player_minutes,
    sample_rotation,
)


def _players():
    return pd.DataFrame(
        [
            _player(1, 40, "C", starter=True, confirmed=True),
            _player(2, 40, "G", starter=True, confirmed=True),
            _player(3, 40, "G", starter=True, confirmed=True),
            _player(4, 40, "F", starter=True, confirmed=True),
            _player(5, 40, "F", starter=True, confirmed=True),
            _player(6, 25, "G", starter=False),
            _player(7, 15, "F", starter=False),
        ]
    )


def _player(pid, minutes, position, *, starter, confirmed=False, status="available", team=1):
    return {
        "player_id": pid,
        "team_id": team,
        "preferred_minutes": minutes,
        "position": position,
        "status": status,
        "is_starter": starter,
        "starter_confirmed": confirmed,
        "pace": 100.0 + pid,
        "off_rating": 110.0,
        "def_rating": 108.0,
    }


def test_valid_rotation_sums_to_240_and_out_is_zero():
    players = _players()
    players.loc[players.player_id.eq(7), "status"] = "out"
    minutes, info = allocate_regulation_minutes(
        players.preferred_minutes.to_numpy(float),
        eligible=players.status.map(classify_status).ne("out").to_numpy(),
        locked=players.starter_confirmed.to_numpy()
        & players.status.map(classify_status).ne("out").to_numpy(),
    )

    assert info["valid"]
    assert minutes.sum() == pytest.approx(240.0)
    assert np.all((minutes >= 0) & (minutes <= 48))
    assert minutes[players.player_id.eq(7).to_numpy()][0] == 0
    # Confirmed starters stay at their pre-removal minutes when the bench can absorb the out.
    for pid in (1, 2, 3, 4, 5):
        assert minutes[players.player_id.eq(pid).to_numpy()][0] == pytest.approx(40.0)


def test_fewer_than_five_eligible_players_is_flagged():
    preferred = np.array([30.0, 30.0, 30.0, 30.0])
    minutes, info = allocate_regulation_minutes(
        preferred, eligible=np.ones(4, dtype=bool)
    )

    assert info["valid"] is False
    assert minutes.sum() != pytest.approx(240.0)


def test_out_starter_is_replaced_by_position_and_bench_minutes():
    players = _players()
    players.loc[players.player_id.eq(5), "status"] = "out"
    # Player 6 has more bench minutes but plays G. Player 7 is the F.
    starters = choose_starters(players)

    assert 5 not in starters
    assert 7 in starters
    assert 6 not in starters
    assert 1 in starters and 4 in starters


def test_unknown_status_is_preserved_and_gtd_draw_is_shared():
    assert classify_status("OUT") == "out"
    assert classify_status("confirmed") == "available"
    assert classify_status("GTD") == "gtd"
    assert classify_status("day-to-day") == "unknown"

    home = _players()
    home["side"] = "home"
    home.loc[home.player_id.eq(6), "status"] = "gtd"
    away = _players()
    away["team_id"] = 2
    away["player_id"] = away["player_id"] + 100
    away["side"] = "away"
    away.loc[away.player_id.eq(106), "status"] = "gtd"
    away.loc[away.player_id.eq(107), "status"] = "scratched"
    both = pd.concat([home, away], ignore_index=True)

    calls = []

    class _Rng:
        def random(self, size):
            calls.append(int(size))
            return np.array([0.1, 0.9])

    players, features = sample_rotation(
        both, n=1, rng=_Rng(), gtd_probability=0.5
    )

    assert calls == [2]
    gtd = players.loc[players.status.eq("gtd")].sort_values("player_id")
    assert list(gtd.resolved_status) == ["available", "out"]
    scratched = players.loc[players.player_id.eq(107)].iloc[0]
    assert scratched.status == "scratched"
    assert scratched.resolved_status == "unknown"
    assert scratched.projected_reg_minutes > 0
    assert set(features[0]) >= {
        "rotation_pace_h",
        "rotation_pace_a",
        "rotation_off_rating_h",
        "rotation_def_rating_a",
    }
    assert players.sim.nunique() == 1


def test_overtime_minutes_add_25_per_period_without_changing_regulation():
    reg = np.array([48.0, 48.0, 48.0, 48.0, 48.0, 0.0])
    full = overtime_player_minutes(reg, n_ot=2)

    assert full.sum() == pytest.approx(240.0 + 50.0)
    assert np.all(full[:5] >= reg[:5])
    assert full[5] == 0


def test_start_position_marks_historical_starters_and_bench():
    rows = []
    for pid, position in ((1, "G"), (2, "G"), (3, "F"), (4, "F"), (5, "C")):
        rows.append(_log("g1", "2024-01-01", 10, pid, 30, 100, position))
    rows.append(_log("g1", "2024-01-01", 10, 6, 12, 90, ""))
    rows.append(_log("g1", "2024-01-01", 10, 7, 8, 90, "  "))
    players, _teams = build_rotation_history(pd.DataFrame(rows), prior_weight=0.0)
    game = players.loc[players.game_id.eq("g1")]

    starters = set(game.loc[game.is_expected_starter, "player_id"])
    bench = set(game.loc[~game.is_expected_starter, "player_id"])
    assert starters == {1, 2, 3, 4, 5}
    assert bench == {6, 7}
    assert game.projected_reg_minutes.sum() == pytest.approx(240.0)
    assert game.loc[game.player_id.eq(6), "role"].iloc[0] == "bench"


def test_current_game_and_future_games_do_not_change_earlier_minutes():
    rows = []
    prior = (
        (1, 40, 110, "G"),
        (2, 36, 108, "G"),
        (3, 32, 104, "F"),
        (4, 28, 100, "F"),
        (5, 24, 96, "C"),
        (6, 40, 80, "G"),
        (7, 40, 80, "F"),
    )
    for pid, minutes, pace, position in prior:
        rows.append(_log("g1", "2024-01-01", 10, pid, minutes, pace, position))
        rows.append(_log("g1", "2024-01-01", 20, pid + 10, minutes, pace - 5, position))
    for pid, _minutes, _pace, position in prior:
        rows.append(_log("g2", "2024-01-03", 10, pid, 5, 100, position))
        rows.append(_log("g2", "2024-01-03", 20, pid + 10, 5, 90, position))
    rows.append(_log("g2", "2024-01-03", 10, 99, 48, 140, "G"))
    for pid, _minutes, _pace, position in prior:
        rows.append(_log("g3", "2024-01-05", 10, pid, 48, 180, position))
        rows.append(_log("g3", "2024-01-05", 20, pid + 10, 48, 180, position))
    logs = pd.DataFrame(rows)
    players, teams = build_rotation_history(logs, roster_window=5, prior_weight=0.0)
    game2 = players.loc[players.game_id.eq("g2") & players.team_id.eq(10)]
    # Player 99 started (start_position G) but has no earlier games.
    # The projection uses the starter prior, not tonight's 48 minutes or 140 pace.
    assert 99 in set(game2.player_id)
    debut = game2.loc[game2.player_id.eq(99)].iloc[0]
    assert bool(debut.is_expected_starter)
    assert debut.projected_reg_minutes == pytest.approx(32.0 * 240.0 / 272.0)
    assert debut.pace == pytest.approx((110 + 108 + 104 + 100 + 96 + 80 + 80) / 7)
    assert game2.loc[game2.player_id.eq(1), "projected_reg_minutes"].iloc[0] == pytest.approx(
        40.0 * 240.0 / 272.0
    )
    assert game2.projected_reg_minutes.sum() == pytest.approx(240.0)
    team_pace = (110 + 108 + 104 + 100 + 96 + 80 + 80) / 7
    expected_pace = (
        40 * 110 + 36 * 108 + 32 * 104 + 28 * 100 + 24 * 96 + 40 * 80 + 40 * 80
        + 32 * team_pace
    ) / 272
    feat_now = teams.loc[teams.game_id.eq("g2") & teams.team_id.eq(10)].iloc[0]
    assert feat_now.rotation_pace == pytest.approx(expected_pace)

    altered = logs.copy()
    altered.loc[altered.game_id.eq("g3"), "min"] = 1.0
    altered.loc[altered.game_id.eq("g3"), "pace"] = 70.0
    players2, teams2 = build_rotation_history(
        altered, roster_window=5, prior_weight=0.0
    )
    left = players.loc[players.game_id.eq("g2"), ["player_id", "team_id", "projected_reg_minutes"]]
    right = players2.loc[players2.game_id.eq("g2"), ["player_id", "team_id", "projected_reg_minutes"]]
    merged = left.merge(right, on=["player_id", "team_id"], suffixes=("_a", "_b"))
    assert np.allclose(
        merged.projected_reg_minutes_a, merged.projected_reg_minutes_b
    )
    feat = teams.loc[teams.game_id.eq("g2") & teams.team_id.eq(10)].iloc[0]
    feat2 = teams2.loc[teams2.game_id.eq("g2") & teams2.team_id.eq(10)].iloc[0]
    assert feat.rotation_pace == pytest.approx(feat2.rotation_pace)
    assert feat.availability_source == "prior_only"


def _log(game_id, date, team, pid, minutes, pace, position):
    return {
        "season_year": "2023-24",
        "season_type": "Regular Season",
        "game_id": game_id,
        "game_date": date,
        "team_id": team,
        "player_id": pid,
        "min": minutes,
        "start_position": position,
        "pace": pace,
        "off_rating": pace + 10,
        "def_rating": 110.0,
    }
