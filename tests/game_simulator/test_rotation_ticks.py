"""Play-by-play stints feed the foul, blowout, and exit hazards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from game_enviroment.script import margin_paths
from game_rotations import fit_rotation_process, minutes_from_process, rotation_ticks
from game_rotations.process import RotationProcess


def test_margin_path_follows_the_half_and_ends_on_the_final_margin():
    paths = margin_paths(
        final=np.array([6.0]),
        game_minutes=np.array([53.0]),
        h1=np.array([10.0]),
        regulation=np.array([2.0]),
    )

    assert len(paths) == 1
    assert len(paths[0]) == 53
    assert paths[0][0] == pytest.approx(0.0)
    assert paths[0][24] == pytest.approx(10.0)
    assert paths[0][48] == pytest.approx(2.0)
    assert paths[0][-1] == pytest.approx(6.0)


def test_flat_margin_path_is_the_final_margin_for_the_whole_game():
    paths = margin_paths(np.array([-4.0]), np.array([48.0]))

    assert np.all(paths[0] == pytest.approx(-4.0))
    assert len(paths[0]) == 48


def _logs():
    names = {
        "1": "Alpha One",
        "2": "Bravo Two",
        "3": "Charlie Three",
        "4": "Delta Four",
        "5": "Echo Five",
        "6": "Foxtrot Six",
    }
    rows = []
    for player, name in names.items():
        rows.append(
            {
                "game_id": "g1",
                "player_id": player,
                "team_id": "10",
                "game_date": "2026-01-01",
                "min": 12.0 if player != "6" else 6.0,
                "player_name": name,
                "start_position": "" if player == "6" else "G",
            }
        )
    return pd.DataFrame(rows)


def _row(number, clock, player, action, description, home=0, away=0, sub=""):
    return {
        "game_id": "g1",
        "action_number": number,
        "period": 1,
        "clock": clock,
        "team_id": "10",
        "player_id": player,
        "action_type": action,
        "sub_type": sub,
        "score_home": home,
        "score_away": away,
        "description": description,
    }


def test_a_foul_exit_is_labeled_and_fits_the_process():
    rows = [
        _row(1, "PT12M00.00S", "1", "made shot", "Alpha One makes a shot", 2, 0),
        _row(2, "PT06M00.00S", "1", "foul", "Alpha One P.FOUL", 2, 0, "personal"),
        _row(3, "PT06M00.00S", "1", "foul", "Alpha One P.FOUL", 2, 0, "personal"),
        _row(
            4,
            "PT06M00.00S",
            "6",
            "substitution",
            "SUB: Foxtrot Six FOR Alpha One",
            2,
            0,
        ),
        _row(5, "PT00M00.00S", "2", "made shot", "Bravo Two makes a shot", 4, 0),
    ]
    ticks = rotation_ticks(pd.DataFrame(rows), _logs())
    process, diagnostics = fit_rotation_process(ticks)

    foul = ticks.loc[ticks["player_id"].eq("1") & ticks["foul_trouble_exit"].eq(1)]
    assert len(foul) == 1
    assert diagnostics["n_player_ticks"] > diagnostics["n_team_games"]
    assert isinstance(process, RotationProcess)


def test_minutes_follow_each_simulation_margin_path():
    from game_rotations.process import (
        BlowoutRule,
        ClosingRule,
        ExitMixture,
        FoulProcess,
        StintHazard,
    )

    process = RotationProcess(
        stint=StintHazard(-20.0, 0.0, 0.0, 0.0),
        foul=FoulProcess(-20.0, 0.0),
        blowout=BlowoutRule(80.0, 0.0, 1.0, 0.0),
        exits=ExitMixture(0.0, 0.0, 0.2, 0.5),
        closing=ClosingRule(0.0, 8.0, 0.5),
    )
    role = pd.DataFrame(
        {
            "player_id": ["1", "2", "3", "4", "5", "6"],
            "is_starter": [1, 1, 1, 1, 1, 0],
            "available": [1, 1, 1, 1, 1, 1],
        }
    )
    environment = pd.DataFrame({"game_minutes": [48.0, 48.0]})
    environment.attrs["margin_path"] = [
        np.zeros(48),
        np.full(48, 4.0),
    ]
    minutes, schedules = minutes_from_process(
        process, role, environment, np.random.default_rng(0)
    )

    assert minutes.shape == (2, 6)
    assert np.allclose(minutes.sum(axis=1), 5 * 48)
    assert len(schedules) == 2
    assert set(schedules[0][0]["players"]) == {"1", "2", "3", "4", "5"}
