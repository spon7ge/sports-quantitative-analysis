"""Fixture backtest wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.mlb.evaluation.backtest import run_backtest
from src.mlb.evaluation.folds import chronological_folds
from src.mlb.schemas import PMF_COLUMNS


def test_run_backtest_fixture_keys_and_pmf(fixture_tables, mlb_config) -> None:
    result = run_backtest(fixture_tables, mlb_config)
    assert {"folds", "scores", "baseline_scores", "predictions"} <= set(result)
    preds = result["predictions"]
    assert not preds.empty
    pmf = preds.loc[:, list(PMF_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)
    fold_names = {row["name"] for row in result["folds"]}
    assert "dev_2018" in fold_names
    assert "dev_2019" in fold_names


def test_backtest_does_not_use_late_quotes(fixture_tables, mlb_config) -> None:
    result = run_backtest(fixture_tables, mlb_config)
    market = result.get("market")
    if market is not None and not market.empty and "quote_id" in market.columns:
        assert "q-late-probe" not in set(market["quote_id"].astype(str))
    preds = result["predictions"]
    quotes = fixture_tables["market_quotes"]
    probe = quotes.loc[quotes["quote_id"] == "q-late-probe"].iloc[0]
    if "quote_id" in preds.columns:
        matched = preds.loc[
            (preds["game_pk"] == probe["game_pk"])
            & (preds["pitcher_id"] == probe["pitcher_id"])
        ]
        assert "q-late-probe" not in set(matched["quote_id"].dropna().astype(str))


def test_2018_fold_has_no_2017_test_dates(fixture_tables, mlb_config) -> None:
    dates = fixture_tables["pitcher_starts"]["game_date"]
    _train, test_idx = chronological_folds(dates, mlb_config)[0]
    test_dates = pd.Series(dates).iloc[test_idx].astype(str)
    assert not test_dates.str.startswith("2017").any()
