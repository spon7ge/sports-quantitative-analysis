"""Evaluation panel merge must not suffix overlapping context columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.mlb.evaluation.backtest import REQUIRED_PANEL_FEATURES, _evaluation_panel
from src.mlb.models.workload import GlmFitError
from src.mlb.models.strikeouts import fit_strikeouts


def _tables_with_overlap() -> dict[str, pd.DataFrame]:
    starts = pd.DataFrame(
        {
            "pitcher_id": [1, 2],
            "game_pk": [10, 11],
            "game_date": ["2024-04-01", "2024-04-02"],
            "season": [2024, 2024],
            "strikeouts": [6, 8],
            "batters_faced": [24, 26],
            "pitches": [90, 95],
            "outs": [18, 19],
            "role": ["starter", "starter"],
            "is_home": [1, 0],
            "opponent_team_id": [2, 3],
            "team_id": [4, 5],
            "venue_id": [7, 8],
            "pitcher_hand": ["R", "L"],
            "scheduled_start_utc": pd.to_datetime(
                ["2024-04-01 23:00:00", "2024-04-02 23:00:00"], utc=True
            ),
            "event_time_utc": pd.to_datetime(
                ["2024-04-01 23:00:00", "2024-04-02 23:00:00"], utc=True
            ),
            "ingested_at_utc": pd.to_datetime(
                ["2024-04-01 23:00:00", "2024-04-02 23:00:00"], utc=True
            ),
            "doubleheader": [0, 0],
        }
    )
    features = pd.DataFrame(
        {
            "pitcher_id": [1, 2],
            "game_pk": [10, 11],
            "is_home": [1.0, 0.0],
            "season": [2024, 2024],
            "venue_id": [7, 8],
            "bf_mean_5": [23.0, 25.0],
            "bf_sd_5": [3.0, 2.5],
            "early_exit_rate_5": [0.1, 0.2],
            "k_bf_shrunk_365": [0.22, 0.28],
            "k_bf_shrunk_60": [0.21, 0.27],
            "rest_days": [5.0, 6.0],
            "pitcher_throws_L": [0.0, 1.0],
        }
    )
    return {"pitcher_starts": starts, "feature_rows": features}


def test_evaluation_panel_keeps_unsuffixed_context_columns(mlb_config) -> None:
    tables = _tables_with_overlap()
    panel = _evaluation_panel(tables, tables["feature_rows"], mlb_config)
    for column in ("is_home", "season", "venue_id"):
        assert column in panel.columns
        assert f"{column}_x" not in panel.columns
        assert f"{column}_y" not in panel.columns
    assert set(panel["is_home"].tolist()) == {0.0, 1.0}
    missing = [name for name in REQUIRED_PANEL_FEATURES if name not in panel.columns]
    assert missing == []


def test_evaluation_panel_raises_on_disagreement(mlb_config) -> None:
    tables = _tables_with_overlap()
    tables["feature_rows"] = tables["feature_rows"].copy()
    tables["feature_rows"].loc[0, "is_home"] = 0.0
    with pytest.raises(ValueError, match="is_home"):
        _evaluation_panel(tables, tables["feature_rows"], mlb_config)


def test_backtest_records_glm_method(fixture_tables, mlb_config) -> None:
    from src.mlb.evaluation.backtest import run_backtest

    result = run_backtest(fixture_tables, mlb_config)
    assert not result["scores"].empty
    assert (result["scores"]["method"] == "glm").all()
    for row in result["folds"]:
        if row["n_test"] > 0:
            assert row["method"] == "glm"


def test_fit_error_terminates_instead_of_publishing_scores(mlb_config) -> None:
    rng = np.random.default_rng(3)
    n = 40
    train = pd.DataFrame(
        {
            "strikeouts": rng.poisson(5.0, n),
            "bf_mean_5": np.zeros(n),
            "bf_sd_5": np.zeros(n),
            "early_exit_rate_5": np.zeros(n),
            "k_bf_shrunk_365": np.zeros(n),
            "k_bf_shrunk_60": np.zeros(n),
            "rest_days": np.full(n, 5.0),
            "pitcher_throws_L": np.zeros(n),
            "is_home": np.zeros(n),
        }
    )
    with pytest.raises(GlmFitError):
        fit_strikeouts(train, mlb_config, fold="broken_2025")
