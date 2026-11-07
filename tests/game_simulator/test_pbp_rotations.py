"""Play-by-play templates arrange an existing Layer 2 minute draw.

Reconstruction keeps only team-games whose intervals match official minutes.
Fitting reports a date-separated retrieval diagnostic. Scheduling must not
change the supplied minute totals.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from game_rotations import (
    fit_pbp_rotation,
    learned_segments_from_minutes,
    reconstruct_rotation_segments,
)

_SRC = Path(__file__).resolve().parents[2] / "nba_game_simulator" / "src"
_HOME = "10"
_AWAY = "20"
_HOME_NAMES = {
    "1": "Alpha One",
    "2": "Bravo Two",
    "3": "Charlie Three",
    "4": "Delta Four",
    "5": "Echo Five",
    "6": "Foxtrot Six",
}
_AWAY_NAMES = {
    "11": "Golf One",
    "12": "Hotel Two",
    "13": "India Three",
    "14": "Juliet Four",
    "15": "Kilo Five",
}


def _minutes(player_id: str) -> float:
    if player_id == "5":
        return 36.0
    if player_id == "6":
        return 12.0
    return 48.0


def _logs(game_id: str, game_date: str) -> list[dict]:
    rows = []
    for player_id, name in _HOME_NAMES.items():
        rows.append(
            {
                "game_id": game_id,
                "player_id": player_id,
                "team_id": _HOME,
                "game_date": game_date,
                "minutes": _minutes(player_id),
                "is_starter": player_id != "6",
                "player_name": name,
            }
        )
    for player_id, name in _AWAY_NAMES.items():
        rows.append(
            {
                "game_id": game_id,
                "player_id": player_id,
                "team_id": _AWAY,
                "game_date": game_date,
                "minutes": 48.0,
                "is_starter": True,
                "player_name": name,
            }
        )
    return rows


def _pbp(
    game_id: str, *, sub_description: str = "SUB: Foxtrot Six FOR Echo Five"
) -> list[dict]:
    rows = []
    action = 1
    for period in range(1, 5):
        rows.append(
            _event(game_id, action, period, "12:00", None, None, "period", "start")
        )
        action += 1
        rows.append(
            _event(
                game_id,
                action,
                period,
                "3:00",
                _HOME,
                None,
                "substitution",
                "",
                sub_description,
            )
        )
        action += 1
        rows.append(
            _event(game_id, action, period, "0:00", None, None, "period", "end")
        )
        action += 1
    return rows


def _event(
    game_id,
    action,
    period,
    clock,
    team_id,
    player_id,
    action_type,
    sub_type,
    description="",
):
    return {
        "game_id": game_id,
        "action_number": action,
        "period": period,
        "clock": clock,
        "team_id": team_id,
        "player_id": player_id,
        "action_type": action_type,
        "sub_type": sub_type,
        "score_home": 100,
        "score_away": 90,
        "description": description,
    }


def _games(game_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"game_id": game_ids, "home_team_id": _HOME})


def _history():
    game_ids = ["g1", "g2", "g3"]
    dates = ["2024-01-01", "2024-01-03", "2024-01-05"]
    logs = pd.DataFrame(
        [
            row
            for game_id, day in zip(game_ids, dates, strict=True)
            for row in _logs(game_id, day)
        ]
    )
    pbp = pd.DataFrame([row for game_id in game_ids for row in _pbp(game_id)])
    return pbp, logs, _games(game_ids)


def _exposure(schedule, player_ids):
    minutes = dict.fromkeys(player_ids, 0.0)
    for segment in schedule:
        duration = segment["end_min"] - segment["start_min"]
        assert len(set(segment["players"])) == 5
        for player in segment["players"]:
            minutes[player] += duration
    return minutes


def _covers(schedule, n_ot=0):
    for period in range(1, 5 + n_ot):
        spans = sorted(
            (segment for segment in schedule if segment["period"] == period),
            key=lambda segment: segment["start_min"],
        )
        cursor = 0.0
        expected = 12.0 if period <= 4 else 5.0
        for segment in spans:
            assert segment["start_min"] == pytest.approx(cursor)
            assert segment["end_min"] > cursor
            cursor = segment["end_min"]
        assert cursor == pytest.approx(expected)


def test_reconstruct_keeps_team_games_that_match_official_minutes():
    pbp, logs, games = _history()

    result = reconstruct_rotation_segments(pbp, logs, games=games)

    assert result.issues.empty
    home = result.comparisons.loc[result.comparisons["team_id"].eq(_HOME)]
    assert home["validated"].all()
    one_game = home.loc[home["game_id"].eq("g1")].set_index("player_id")
    assert one_game.loc["1", "reconstructed_minutes"] == pytest.approx(48)
    assert one_game.loc["5", "reconstructed_minutes"] == pytest.approx(36)
    assert one_game.loc["6", "reconstructed_minutes"] == pytest.approx(12)
    assert set(result.segments.loc[result.segments["validated"], "game_id"]) == {
        "g1",
        "g2",
        "g3",
    }


def test_reconstruct_records_an_unreadable_substitution():
    logs = pd.DataFrame(_logs("g1", "2024-01-01"))
    pbp = pd.DataFrame(_pbp("g1", sub_description="not a substitution"))

    result = reconstruct_rotation_segments(pbp, logs, games=_games(["g1"]))

    home = result.comparisons.loc[result.comparisons["team_id"].eq(_HOME)]
    assert not home["validated"].any()
    assert result.issues["reason"].str.contains("unrecognized substitution").any()


def test_conflicting_official_minutes_are_rejected():
    logs = pd.DataFrame(_logs("g1", "2024-01-01"))
    conflict = logs.iloc[[0]].copy()
    conflict["minutes"] = 10.0
    pbp = pd.DataFrame(_pbp("g1"))

    with pytest.raises(ValueError, match="conflicting official player-game labels"):
        reconstruct_rotation_segments(
            pbp,
            pd.concat([logs, conflict], ignore_index=True),
            games=_games(["g1"]),
        )


def test_fit_reports_retrieval_distance_on_a_later_date():
    pbp, logs, games = _history()
    reconstruction = reconstruct_rotation_segments(pbp, logs, games=games)

    model = fit_pbp_rotation(reconstruction, holdout_fraction=0.2)

    assert model.training_cutoff == "2024-01-05"
    assert model.diagnostics["holdout_first_date"] == "2024-01-05"
    assert model.diagnostics["training_team_games"] == 4
    assert model.diagnostics["holdout_team_games"] == 2
    assert model.diagnostics["nearest_boundary_mae_minutes"] == pytest.approx(0)
    # Retrieval keeps the earliest same-margin template. That date still uses
    # the default 18-minute prior, so later role bins do not match it.
    assert model.diagnostics["role_signature_mae"] == pytest.approx(2.4375)
    assert "not a pregame forecast" in model.diagnostics["metric_scope"]


def test_learned_segments_preserve_the_supplied_minutes():
    pbp, logs, games = _history()
    model = fit_pbp_rotation(reconstruct_rotation_segments(pbp, logs, games=games))
    roster = pd.DataFrame(
        {
            "player_id": list(_HOME_NAMES),
            "team_id": _HOME,
            "is_starter": [player != "6" for player in _HOME_NAMES],
            "min_mean_10": 18.0,
        }
    )
    environment = pd.DataFrame(
        {
            "team_margin": [10.0],
            "n_ot": [0],
            "game_minutes": [48.0],
            "game_date": ["2024-02-01"],
        }
    )
    draws = pd.DataFrame(
        [[_minutes(player) for player in roster["player_id"]]],
        columns=roster["player_id"],
        index=environment.index,
    )

    schedules = learned_segments_from_minutes(
        model,
        roster,
        environment,
        draws,
        seed=0,
    )

    assert len(schedules) == 1
    played = _exposure(schedules[0], list(_HOME_NAMES))
    for player, minutes in played.items():
        assert minutes == pytest.approx(_minutes(player))
    _covers(schedules[0])
    assert schedules[0][0]["period"] == 1
    assert schedules[0][0]["start_min"] == pytest.approx(0)
    assert set(schedules[0][0]["players"]) == {"1", "2", "3", "4", "5"}


def test_learned_segments_refuse_dates_inside_the_template_window():
    pbp, logs, games = _history()
    model = fit_pbp_rotation(reconstruct_rotation_segments(pbp, logs, games=games))
    roster = pd.DataFrame({"player_id": list(_HOME_NAMES), "team_id": _HOME})
    environment = pd.DataFrame(
        {"team_margin": [10.0], "n_ot": [0], "game_minutes": [48.0]}
    )
    draws = np.array([[_minutes(player) for player in _HOME_NAMES]])

    with pytest.raises(ValueError, match="training cutoff"):
        learned_segments_from_minutes(
            model,
            roster,
            environment,
            draws,
            game_date="2024-01-05",
        )


def test_learned_segments_reject_an_optimization_that_is_too_large():
    pbp, logs, games = _history()
    model = fit_pbp_rotation(reconstruct_rotation_segments(pbp, logs, games=games))
    roster = pd.DataFrame({"player_id": list(_HOME_NAMES), "team_id": _HOME})
    environment = pd.DataFrame(
        {
            "team_margin": [10.0],
            "n_ot": [0],
            "game_minutes": [48.0],
            "game_date": ["2024-02-01"],
        }
    )
    draws = np.array([[_minutes(player) for player in _HOME_NAMES]])

    with pytest.raises(ValueError, match="too large"):
        learned_segments_from_minutes(
            model,
            roster,
            environment,
            draws,
            max_variables=10,
        )


def test_cli_writes_the_reconstruction_audit(tmp_path):
    pbp, logs, games = _history()
    pbp_path = tmp_path / "pbp.csv"
    logs_path = tmp_path / "logs.csv"
    games_path = tmp_path / "games.csv"
    out = tmp_path / "out"
    pbp.to_csv(pbp_path, index=False)
    logs.to_csv(logs_path, index=False)
    games.to_csv(games_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "game_rotations",
            "--pbp",
            str(pbp_path),
            "--logs",
            str(logs_path),
            "--games",
            str(games_path),
            "--output",
            str(out),
        ],
        check=False,
        cwd=_SRC,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    diagnostics = json.loads((out / "diagnostics.json").read_text())
    assert diagnostics["validated_team_games"] == 6
    assert diagnostics["training_cutoff"] == "2024-01-05"
    assert (out / "minute_comparisons.csv").exists()
    assert (out / "reconstruction_issues.csv").exists()
    assert (out / "historical_segments.csv").exists()
    assert (out / "pbp_rotation_model.joblib").exists()
