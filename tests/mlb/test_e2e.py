"""End-to-end fixture backtest: PMFs, quote cutoff, chronological folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.mlb.evaluation.backtest import run_backtest
from src.mlb.evaluation.folds import chronological_folds
from src.mlb.evaluation.market import compare_market
from src.mlb.schemas import PMF_COLUMNS


def test_e2e_run_backtest_on_fixture(fixture_tables, mlb_config) -> None:
    result = run_backtest(fixture_tables, mlb_config)
    preds = result["predictions"]
    assert not preds.empty
    pmf = preds.loc[:, list(PMF_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)

    quotes = fixture_tables["market_quotes"]
    probe = quotes.loc[quotes["quote_id"] == "q-late-probe"].iloc[0]
    pregame = fixture_tables["pregame_snapshots"]
    snap = pregame.loc[
        (pregame["game_pk"] == probe["game_pk"])
        & (pregame["pitcher_id"] == probe["pitcher_id"])
    ].iloc[0]
    pred_row = pd.DataFrame(
        {
            "pitcher_id": [int(probe["pitcher_id"])],
            "game_pk": [int(probe["game_pk"])],
            "prediction_cutoff_utc": [snap["prediction_cutoff_utc"]],
            **{
                name: [float(preds[name].mean()) if name in preds.columns else 0.0]
                for name in PMF_COLUMNS
            },
        }
    )
    compared = compare_market(pred_row, quotes, mlb_config)
    if "quote_id" in compared.columns:
        assert "q-late-probe" not in set(compared["quote_id"].astype(str))
    market = result.get("market")
    if (
        market is not None
        and not getattr(market, "empty", True)
        and "quote_id" in market.columns
    ):
        assert "q-late-probe" not in set(market["quote_id"].astype(str))

    dates = fixture_tables["pitcher_starts"]["game_date"]
    _train_idx, test_idx = chronological_folds(dates, mlb_config)[0]
    test_dates = pd.Series(dates).iloc[test_idx].astype(str)
    assert not test_dates.str.startswith("2017").any()
    fold_2018 = next(row for row in result["folds"] if row["name"] == "dev_2018")
    assert fold_2018["n_test"] > 0
