"""Season folds validate every recent season and leave postseason rows out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from models.shared import train as train_mod
from models.shared.splits import (
    date_walk_forward_folds,
    early_season_mask,
    season_walk_forward_folds,
)
from models.shared.train import run_walk_forward


def _panel() -> pd.DataFrame:
    rows = []
    for season, start in (
        ("2021-22", "2021-10-01"),
        ("2022-23", "2022-10-01"),
        ("2023-24", "2023-10-01"),
    ):
        for team in (1, 2):
            for game in range(3):
                rows.append(
                    {
                        "season_year": season,
                        "game_date": pd.Timestamp(start) + pd.Timedelta(days=game),
                        "team_id": team,
                        "game_id": f"{season}-{team}-{game}",
                        "season_type": "Regular Season" if game < 2 else "Playoffs",
                    }
                )
    return pd.DataFrame(rows)


def test_expanding_folds_validate_each_requested_season_and_drop_playoffs():
    frame = _panel()
    folds = list(
        season_walk_forward_folds(
            frame,
            val_seasons=("2022-23", "2023-24"),
            season_type_col="season_type",
        )
    )

    assert [fold["fold"] for fold in folds] == [1, 2]
    assert folds[0]["label"] == "2021-22..2021-22 -> 2022-23"
    assert folds[1]["label"] == "2021-22..2022-23 -> 2023-24"
    assert int(folds[0]["train_mask"].sum()) < int(folds[1]["train_mask"].sum())

    for fold, season in zip(folds, ("2022-23", "2023-24"), strict=True):
        train_seasons = set(frame.loc[fold["train_mask"], "season_year"])
        val = frame.loc[fold["val_mask"]]
        assert train_seasons.isdisjoint({season})
        assert set(val["season_year"]) == {season}
        assert set(val["season_type"]) == {"Regular Season"}
        assert int(fold["n_val_postseason"]) == 2
        assert pd.Timestamp(fold["val_dates"][0]) < pd.Timestamp(fold["val_dates"][-1])
        assert not np.any(fold["train_mask"] & fold["val_mask"])


def test_sliding_window_keeps_only_the_latest_train_seasons():
    frame = _panel()
    folds = list(
        season_walk_forward_folds(
            frame,
            val_seasons=("2023-24",),
            max_train_seasons=1,
            season_type_col="season_type",
        )
    )
    trained = set(frame.loc[folds[0]["train_mask"], "season_year"])
    assert trained == {"2022-23"}


def test_a_recent_season_left_out_of_validation_is_an_error():
    frame = _panel()
    with pytest.raises(ValueError, match="2023-24"):
        list(
            season_walk_forward_folds(
                frame,
                val_seasons=("2022-23",),
                season_type_col="season_type",
            )
        )


def test_unknown_season_type_is_an_error():
    frame = _panel()
    frame.loc[0, "season_type"] = "Postseason"
    with pytest.raises(ValueError, match="Postseason"):
        list(
            season_walk_forward_folds(
                frame,
                val_seasons=("2022-23", "2023-24"),
                season_type_col="season_type",
            )
        )


def test_early_slice_is_the_first_ten_games_of_each_team_season():
    rows = []
    for season in ("2023-24", "2024-25"):
        for team in (10, 20):
            for game in range(12):
                rows.append(
                    {
                        "season_year": season,
                        "team_id": team,
                        "game_id": f"{season}-{team}-{game}",
                        "game_date": pd.Timestamp("2023-10-01")
                        + pd.Timedelta(days=game + (0 if season == "2023-24" else 400)),
                        "player_id": 1,
                    }
                )
    frame = pd.DataFrame(rows)
    # A second player on game 0 must share the team's flag.
    extra = frame.iloc[[0]].copy()
    extra["player_id"] = 2
    frame = pd.concat([extra, frame], ignore_index=True)

    early = early_season_mask(frame, n_games=10)
    assert early.index.equals(frame.index)
    flagged = frame.loc[early]
    assert set(flagged.groupby(["season_year", "team_id"])["game_id"].nunique()) == {10}
    late = frame.loc[~early, "game_id"]
    assert set(late) == {
        "2023-24-10-10",
        "2023-24-10-11",
        "2023-24-20-10",
        "2023-24-20-11",
        "2024-25-10-10",
        "2024-25-10-11",
        "2024-25-20-10",
        "2024-25-20-11",
    }
    game0 = frame["game_id"].eq("2023-24-10-0")
    assert early.loc[game0].all()


def test_walk_forward_scores_the_early_slice_and_leaves_playoffs_out(monkeypatch):
    class _Fake:
        def __init__(self, **kwargs):
            self.best_iteration = 0

        def fit(self, X, y, eval_set=None, verbose=False):
            return None

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(train_mod, "XGBRegressor", _Fake)
    rows = []
    for season, start in (("2022-23", "2022-10-01"), ("2023-24", "2023-10-01")):
        for game in range(12):
            rows.append(
                {
                    "season_year": season,
                    "season_type": "Playoffs" if game == 11 else "Regular Season",
                    "game_date": pd.Timestamp(start) + pd.Timedelta(days=game),
                    "team_id": 7,
                    "game_id": f"{season}-{game}",
                    "starting": 1,
                    "x": 1.0,
                    "pts": float(game),
                }
            )
    frame = pd.DataFrame(rows)
    folds = list(
        season_walk_forward_folds(
            frame,
            val_seasons=("2023-24",),
            season_type_col="season_type",
        )
    )
    result = run_walk_forward(
        frame[["x"]],
        frame["pts"],
        frame,
        xgb_params={"n_estimators": 2, "early_stopping_rounds": 1},
        quantiles=[0.50],
        folds=folds,
        early_team_games=10,
        target_col="pts",
    )
    assert set(frame.loc[result["oof"].index, "season_type"]) == {"Regular Season"}
    assert len(result["oof"]) == 11
    assert result["wf_results"][0]["early_n"] == 10
    assert "early_mean_pinball" in result["wf_results"][0]


def test_date_folds_use_chronological_dates():
    dates = pd.to_datetime(
        ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]
    )
    # The frame itself stays sorted; unique() order is what the sort protects.
    frame = pd.DataFrame(
        {
            "game_date": np.sort(dates),
            "row": np.arange(5),
        }
    )
    folds = list(date_walk_forward_folds(frame, train_frac=0.4, step_frac=0.2, n_folds=2))
    for fold in folds:
        assert list(fold["val_dates"]) == sorted(fold["val_dates"])
        assert fold["train_dates"][-1] < fold["val_dates"][0]
