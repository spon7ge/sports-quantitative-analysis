"""Date splits keep within-date row order so two calls share windows."""

from __future__ import annotations

import pandas as pd

import numpy as np
import pytest

from models.shared.splits import season_holdout_split, tukey_fences


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


def test_tukey_fences_mark_only_the_extreme_tail():
    rates = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2.5])
    low, high = tukey_fences(rates)
    # Q1=0.375, Q3=0.725, IQR=0.35 → fences [-0.15, 1.25]
    assert low == pytest.approx(-0.15)
    assert high == pytest.approx(1.25)
    inside = (rates >= low) & (rates <= high)
    assert inside.tolist() == [True, True, True, True, True, True, True, False]


def test_tukey_fences_on_a_short_series():
    low, high = tukey_fences([1.0, 2.0, 3.0, 4.0])
    # Q1=1.75, Q3=3.25, IQR=1.5
    assert low == pytest.approx(-0.5)
    assert high == pytest.approx(5.5)


def test_tukey_fences_reject_missing_rates():
    with pytest.raises(ValueError, match="finite"):
        tukey_fences([0.4, np.nan])
