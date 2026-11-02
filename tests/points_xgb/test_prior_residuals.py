"""Prior residual summaries use earlier predictions only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from points_xgb.errors import prior_residual_features


def _rows(residuals: list[float], player_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": player_id,
            "game_id": np.arange(len(residuals)),
            "game_date": pd.date_range("2026-01-01", periods=len(residuals), freq="D"),
            "pts": residuals,
            "q50": np.zeros(len(residuals)),
        }
    )


def test_windows_wait_for_enough_prior_games() -> None:
    features = prior_residual_features(_rows([1, 2, 3, 4, 5, 10]))
    assert features["residual_count"].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.isnan(features["residual_median_5"].iloc[:5]).all()
    assert features["residual_median_5"].iloc[5] == pytest.approx(3.0)
    assert np.isnan(features["residual_median_10"]).all()
    assert np.isnan(features["residual_mad_10"]).all()


def test_median_and_mad_use_the_last_window_only() -> None:
    residuals = list(range(1, 12))
    features = prior_residual_features(_rows(residuals))
    # Entering the last row, the prior residuals are 1..10.
    assert features["residual_count"].iloc[-1] == 10
    assert features["residual_median_10"].iloc[-1] == pytest.approx(5.5)
    assert features["residual_median_5"].iloc[-1] == pytest.approx(8.0)
    assert features["residual_mad_10"].iloc[-1] == pytest.approx(2.5)
    assert np.isnan(features["residual_median_20"].iloc[-1])


def test_a_later_game_does_not_change_an_earlier_summary() -> None:
    short = prior_residual_features(_rows([1, 2, 3, 4, 5, 6]))
    long = prior_residual_features(_rows([1, 2, 3, 4, 5, 6, 100]))
    pd.testing.assert_series_equal(
        short["residual_median_5"],
        long["residual_median_5"].iloc[:6],
        check_names=False,
    )


def test_players_do_not_share_residual_history() -> None:
    frame = pd.concat([_rows([1, 1, 1, 1, 1, 9]), _rows([4, 4, 4, 4, 4, 4], 2)])
    features = prior_residual_features(frame.reset_index(drop=True))
    assert features["residual_median_5"].iloc[5] == pytest.approx(1.0)
    assert features["residual_median_5"].iloc[11] == pytest.approx(4.0)
