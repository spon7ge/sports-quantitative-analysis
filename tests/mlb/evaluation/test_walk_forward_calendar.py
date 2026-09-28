import pandas as pd
import pytest

from src.mlb.evaluation.walk_forward_calendar import (
    calendar_from_schedule,
    validate_regular_season_calendar,
)


def test_calendar_keeps_final_regular_season_games_only():
    schedule = pd.DataFrame(
        [
            {"season": 2018, "official_date": "2018-03-29", "game_type": "R", "abstract_game_state": "Final"},
            {"season": 2018, "official_date": "2018-03-28", "game_type": "S", "abstract_game_state": "Final"},
            {"season": 2018, "official_date": "2018-10-05", "game_type": "F", "abstract_game_state": "Final"},
            {"season": 2018, "official_date": "2018-09-30", "game_type": "R", "abstract_game_state": "Final"},
            {"season": 2019, "official_date": "2019-03-20", "game_type": "R", "abstract_game_state": "Final"},
            {"season": 2019, "official_date": "2019-09-29", "game_type": "R", "abstract_game_state": "Final"},
        ]
    )
    calendar = calendar_from_schedule(schedule)
    row_2018 = calendar.loc[calendar["season"] == 2018].iloc[0]
    assert row_2018["regular_season_open_date"] == "2018-03-29"
    assert row_2018["regular_season_close_date"] == "2018-09-30"


def test_calendar_rejects_a_missing_season():
    frame = pd.DataFrame(
        {
            "season": [2018],
            "regular_season_open_date": ["2018-03-29"],
            "regular_season_close_date": ["2018-09-30"],
        }
    )
    with pytest.raises(ValueError, match="2019"):
        validate_regular_season_calendar(frame, seasons=range(2018, 2020))


def test_calendar_rejects_open_after_close():
    frame = pd.DataFrame(
        {
            "season": list(range(2018, 2026)),
            "regular_season_open_date": ["2018-04-02"] + ["2019-03-20"] * 7,
            "regular_season_close_date": ["2018-04-01"] + ["2019-09-29"] * 7,
        }
    )
    with pytest.raises(ValueError, match="open"):
        validate_regular_season_calendar(frame)
