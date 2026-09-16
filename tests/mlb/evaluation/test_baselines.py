"""Baseline predictors emit test-row PMFs that sum to one."""

from __future__ import annotations

import numpy as np
from src.mlb.evaluation.baselines import fit_baselines
from src.mlb.schemas import PMF_COLUMNS


def test_fit_baselines_keys_and_pmf(fixture_tables, mlb_config) -> None:
    starts = fixture_tables["pitcher_starts"]
    train = starts.loc[starts["game_date"] <= "2017-12-31"].copy()
    test = starts.loc[
        (starts["game_date"] >= "2018-01-01") & (starts["game_date"] <= "2018-12-31")
    ].copy()
    out = fit_baselines(train, test, mlb_config)
    expected_keys = {
        "league_nb",
        "rolling_k",
        "k9_workload",
        "shrunk_kbf",
        "pitcher_opp",
        "marcel",
    }
    assert expected_keys <= set(out)
    assert "market" not in out
    for name, frame in out.items():
        assert len(frame) == len(test), name
        assert "expected_k" in frame.columns
        pmf = frame.loc[:, list(PMF_COLUMNS)].to_numpy(dtype=float)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(np.isfinite(frame["expected_k"]))
        assert np.all(frame["expected_k"] > 0)


def test_market_baseline_only_when_quotes_present(fixture_tables, mlb_config) -> None:
    starts = fixture_tables["pitcher_starts"]
    quotes = fixture_tables["market_quotes"]
    train = starts.loc[starts["game_date"] <= "2018-12-31"].copy()
    test = starts.loc[starts["season"] == 2019].copy()
    quoted = test.merge(
        quotes[
            [
                "game_pk",
                "pitcher_id",
                "line",
                "over_price",
                "under_price",
                "price_format",
            ]
        ],
        on=["game_pk", "pitcher_id"],
        how="left",
    )
    out = fit_baselines(train, quoted, mlb_config)
    assert "market" in out
    assert len(out["market"]) == int(quoted["over_price"].notna().sum())
    pmf = out["market"].loc[:, list(PMF_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)
