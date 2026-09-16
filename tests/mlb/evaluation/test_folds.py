"""Chronological fold construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.mlb.evaluation.folds import chronological_folds


def test_chronological_folds_expanding_no_shuffle(mlb_config) -> None:
    dates = ["2017-05-01", "2018-05-01", "2018-06-01", "2019-05-01"]
    folds = chronological_folds(dates, mlb_config)
    assert len(folds) == 2
    train0, test0 = folds[0]
    train1, test1 = folds[1]
    np.testing.assert_array_equal(train0, np.array([0]))
    np.testing.assert_array_equal(test0, np.array([1, 2]))
    np.testing.assert_array_equal(train1, np.array([0, 1, 2]))
    np.testing.assert_array_equal(test1, np.array([3]))
    assert list(train0) == sorted(train0)
    assert list(test0) == sorted(test0)


def test_2018_test_fold_excludes_2017_dates(fixture_tables, mlb_config) -> None:
    dates = fixture_tables["pitcher_starts"]["game_date"]
    folds = chronological_folds(dates, mlb_config)
    _train_idx, test_idx = folds[0]
    test_dates = pd.Series(dates).iloc[test_idx].astype(str)
    assert not test_dates.str.startswith("2017").any()
    assert test_dates.str.startswith("2018").all()
    train_dates = pd.Series(dates).iloc[_train_idx].astype(str)
    assert train_dates.str.startswith("2017").all()
