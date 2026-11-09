"""A slate date is scored with models fit only on earlier games."""

from __future__ import annotations

import pandas as pd
import pytest

from slate_backtest import (
    assert_before,
    environment_cutoff,
    history_before,
    parse_slate_dates,
    played_dates,
    played_on,
)


def _logs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_date": [
                "2025-11-04",
                "2025-11-05",
                "2025-11-06",
                "2025-11-07",
            ],
            "team_pts": [100, 110, 140, 90],
        }
    )


def test_history_stops_the_day_before_the_slate():
    history = history_before(_logs(), "2025-11-06")
    assert list(history["game_id"]) == ["g1", "g2"]
    assert history["game_date"].max() == pd.Timestamp("2025-11-05")


def test_same_day_outcomes_never_enter_history():
    logs = _logs()
    history = history_before(logs, "2025-11-06")
    logs.loc[logs["game_id"].eq("g3"), "team_pts"] = 200
    again = history_before(logs, "2025-11-06")
    assert history["team_pts"].tolist() == again["team_pts"].tolist()
    assert 200 not in history["team_pts"].tolist()


def test_played_on_is_only_the_slate():
    played = played_on(_logs(), "2025-11-06")
    assert list(played["game_id"]) == ["g3"]


def test_cutoff_on_the_slate_is_rejected():
    with pytest.raises(ValueError, match="environment"):
        assert_before("2025-11-06", "2025-11-06", "environment")


def test_environment_cutoff_is_the_latest_team_date():
    model = {
        "state": {
            "teams": {
                "1": {"last_date": "2025-11-04"},
                "2": {"last_date": "2025-11-05"},
            }
        }
    }
    assert environment_cutoff(model) == pd.Timestamp("2025-11-05")


def test_played_dates_skip_calendar_days_with_no_games():
    logs = _logs().loc[lambda frame: frame["game_date"] != "2025-11-06"]
    dates = played_dates(logs, "2025-11-04", "2025-11-08")
    assert dates == [
        pd.Timestamp("2025-11-04"),
        pd.Timestamp("2025-11-05"),
        pd.Timestamp("2025-11-07"),
    ]


def test_played_dates_reject_a_window_with_no_games():
    with pytest.raises(ValueError, match="no completed games"):
        played_dates(_logs(), "2025-11-08", "2025-11-10")


def test_slate_dates_are_unique_and_ordered():
    dates = parse_slate_dates(["2025-11-06", "2025-11-05", "2025-11-06"])
    assert dates == [
        pd.Timestamp("2025-11-05"),
        pd.Timestamp("2025-11-06"),
    ]
