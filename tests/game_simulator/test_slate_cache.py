"""Stored slate sims are reused until a new date appears."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from slate_cache import missing_slates, read_slates, save_slate


def _frames(day: str, total: float) -> dict[str, pd.DataFrame]:
    samples = np.array([total, total + 1.0])
    return {
        "game_sims": pd.DataFrame(
            {
                "slate_date": [day],
                "game_id": ["g1"],
                "history_through": ["2025-11-04"],
                "home_samples": [samples],
            }
        ),
        "player_sims": pd.DataFrame(
            {"slate_date": [day], "game_id": ["g1"], "player_name": ["A"]}
        ),
        "game_actuals": pd.DataFrame(
            {"slate_date": [day], "game_id": ["g1"], "home_pts": [total]}
        ),
        "player_actuals": pd.DataFrame(
            {"slate_date": [day], "player_name": ["A"], "PTS": [20.0]}
        ),
        "stopped": pd.DataFrame(columns=["slate_date", "game", "error"]),
    }


def test_saved_slate_is_reused_and_a_new_date_is_missing(tmp_path):
    save_slate(tmp_path, "2025-11-05", **_frames("2025-11-05", 210.0))
    dates = [
        pd.Timestamp("2025-11-05"),
        pd.Timestamp("2025-11-06"),
    ]

    assert missing_slates(tmp_path, dates) == [pd.Timestamp("2025-11-06")]
    stored = read_slates(tmp_path, dates[:1])
    assert stored["game_sims"]["home_samples"].iloc[0].tolist() == [210.0, 211.0]
    assert list(stored["game_actuals"]["home_pts"]) == [210.0]


def test_read_slates_keeps_calendar_order(tmp_path):
    save_slate(tmp_path, "2025-11-06", **_frames("2025-11-06", 220.0))
    save_slate(tmp_path, "2025-11-05", **_frames("2025-11-05", 210.0))
    stored = read_slates(
        tmp_path,
        [pd.Timestamp("2025-11-05"), pd.Timestamp("2025-11-06")],
    )
    assert list(stored["game_sims"]["slate_date"]) == ["2025-11-05", "2025-11-06"]


def test_rows_from_another_date_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="2025-11-05"):
        save_slate(tmp_path, "2025-11-05", **_frames("2025-11-06", 220.0))
