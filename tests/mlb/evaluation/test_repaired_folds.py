"""Acceptance checks for the repaired strikeout GLM on stored gamelog tables."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from src.mlb.cli import _load_tables
from src.mlb.config import FoldWindow, load_config
from src.mlb.evaluation.backtest import run_backtest


def _expanding_config():
    config = load_config()
    tables, _store = _load_tables(config, fixture=False)
    starts = tables.get("pitcher_starts")
    if starts is None or len(starts) < 10_000:
        pytest.skip("stored 2018–2025 starter logs are not available")
    seasons = pd.to_numeric(starts["season"], errors="coerce")
    if seasons.min() > 2018 or seasons.max() < 2025:
        pytest.skip("stored starter logs do not cover 2018–2025")
    folds = tuple(
        FoldWindow(
            name=f"eval_{year}",
            train_end=f"{year - 1}-12-31",
            test_start=f"{year}-01-01",
            test_end=f"{year}-12-31",
        )
        for year in range(2021, 2026)
    )
    return replace(config, folds=folds), tables


def test_repaired_2021_2025_folds_are_genuine_glms() -> None:
    config, tables = _expanding_config()
    result = run_backtest(tables, config)
    scores = result["scores"]
    baselines = result["baseline_scores"]
    preds = result["predictions"]
    assert not scores.empty
    assert set(scores["fold"]) == {f"eval_{year}" for year in range(2021, 2026)}
    assert (scores["method"] == "glm").all()
    for row in result["folds"]:
        if row["n_test"] > 0:
            assert row["method"] == "glm"
            assert row["matrix_rank"] == 1 + len(row["retained_features"])
            assert row["retained_features"]

    frozen = preds.loc[preds["fold"].isin(["eval_2023", "eval_2024", "eval_2025"])]
    assert frozen["expected_k"].std() > 0.25
    league = baselines.loc[
        (baselines["model"] == "league_nb")
        & (baselines["fold"].isin(["eval_2023", "eval_2024", "eval_2025"]))
    ]
    glm = scores.loc[scores["fold"].isin(["eval_2023", "eval_2024", "eval_2025"])]
    assert float(glm["pmf_nll"].mean()) < float(league["pmf_nll"].mean())
    league_mu = float(
        preds.loc[preds["fold"].isin(["eval_2023", "eval_2024", "eval_2025"]), "expected_k"].mean()
    )
    assert not np.allclose(frozen["expected_k"].to_numpy(), league_mu, atol=0.05)

    for name in ("shrunk_kbf", "marcel", "league_nb"):
        assert name in set(baselines["model"])
