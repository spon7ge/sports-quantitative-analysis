"""Date splits keep within-date row order so two calls share windows."""

from __future__ import annotations

import pandas as pd

from models.shared.splits import season_holdout_split


def test_holdout_split_preserves_within_date_order():
    frame = pd.DataFrame(
        {
            "season_year": ["2024-25"] * 4,
            "game_date": pd.to_datetime(
                ["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"]
            ),
            "player_id": [9, 2, 1, 8],
        }
    )
    train, _holdout = season_holdout_split(frame, holdout_season="2025-26")
    jan1 = train.loc[train["game_date"] == "2024-01-01", "player_id"].tolist()
    assert jan1 == [2, 1]
