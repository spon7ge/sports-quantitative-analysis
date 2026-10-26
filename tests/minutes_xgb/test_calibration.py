"""Walk-forward conformal checks extracted from the minutes notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minutes_xgb.calibration import (
    calibration_table,
    fold_base_knots,
    nested_offset_backtest,
    offset_stability_flags,
    oof_leakage_table,
    oof_player_games,
)


def _ppm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(6)],
            "player_id": np.arange(6),
            "game_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ]
            ),
            "starting": [1, 1, 1, 1, 1, 0],
            "minutes": [10.0, 30.0, 2.0, 4.0, 20.0, 12.0],
            "min_lag_1": [1.0, 1.0, 1.0, 1.0, np.nan, 5.0],
        }
    )


def test_oof_player_games_reject_a_repeated_player_game():
    ppm = _ppm()
    oof = ppm.iloc[[4, 4]].copy()
    with pytest.raises(RuntimeError, match="duplicated"):
        oof_player_games(oof, ppm)


def test_leakage_table_rejects_a_validation_date_in_the_training_tail():
    ppm = _ppm().iloc[:4].copy()
    oof = ppm.iloc[[3]].copy()
    oof["fold_id"] = 1
    oof["train_tail_frac"] = 0.10
    train_mask = pd.Series([True, True, True, False], index=ppm.index)
    folds = [
        {
            "fold": 1,
            "train_dates": list(ppm["game_date"].iloc[:3]),
            "val_dates": [ppm["game_date"].iloc[2]],
            "train_mask": train_mask,
        }
    ]
    with pytest.raises(RuntimeError, match="leaks"):
        oof_leakage_table(oof, ppm, folds)

    folds[0]["val_dates"] = [ppm["game_date"].iloc[3]]
    table = oof_leakage_table(oof, ppm, folds)
    assert table.loc[0, "val_in_train"] == 0
    assert table.loc[0, "val_in_stop"] == 0
    assert bool(table.loc[0, "dates_match"])
    assert str(table.loc[0, "train_end"]) == "2024-01-03"
    assert str(table.loc[0, "val_start"]) == "2024-01-04"


def test_fold_base_knots_use_that_folds_history_only():
    ppm = _ppm()
    oof = ppm.iloc[[4, 5]].copy()
    oof["fold_id"] = 1
    oof["q_0.90"] = [50.0, 8.0]
    oof["q_0.10"] = [100.0, 9.0]
    oof["q_0.50"] = [0.0, 1.0]
    mask = pd.Series([True, True, False, False, False, False], index=ppm.index)
    base, group = fold_base_knots(oof, ppm, [mask])

    np.testing.assert_allclose(base.loc[oof.index[0]].to_numpy(), [12.0, 20.0, 28.0])
    np.testing.assert_allclose(base.loc[oof.index[1]].to_numpy(), [1.0, 8.0, 9.0])
    assert list(group) == ["cold_start", "<15"]

    cold = ppm.iloc[[4]].copy()
    cold["fold_id"] = 1
    cold["q_0.10"] = 0.0
    cold["q_0.50"] = 0.0
    cold["q_0.90"] = 0.0
    fold2 = pd.Series([False, False, True, True, False, False], index=ppm.index)
    base2, _ = fold_base_knots(cold, ppm, [fold2])
    np.testing.assert_allclose(base2.iloc[0].to_numpy(), [2.2, 3.0, 3.8])


def test_calibration_table_reports_the_gap_from_the_nominal_level():
    dates = np.array(["2024-01-01", "2024-01-01"], dtype="datetime64[ns]")
    base = pd.DataFrame(
        {
            "q_0.10": [1.0, 1.0],
            "q_0.50": [1.0, 1.0],
            "q_0.90": [1.0, 1.0],
        }
    )
    table = calibration_table(
        np.zeros(2),
        dates,
        base,
        np.array(["<15", "<15"]),
        starting=np.array([1, 0]),
        n_boot=20,
        seed=0,
    )
    overall = table.loc[table["slice"].eq("overall") & table["check"].eq("q_0.50")].iloc[0]
    assert overall["empirical"] == 1.0
    assert overall["delta"] == pytest.approx(0.5)


def test_nested_offset_is_fit_on_earlier_folds_only():
    fold1 = pd.DataFrame(
        {
            "fold_id": 1,
            "minutes": np.full(400, 11.0),
            "game_date": pd.Timestamp("2024-01-01"),
            "q_0.10": np.full(400, 10.0),
            "q_0.50": np.full(400, 10.0),
            "q_0.90": np.full(400, 10.0),
        }
    )
    fold2 = pd.DataFrame(
        {
            "fold_id": 2,
            "minutes": np.full(20, 10.0),
            "game_date": pd.Timestamp("2024-02-01"),
            "q_0.10": np.full(20, 10.0),
            "q_0.50": np.full(20, 10.0),
            "q_0.90": np.full(20, 10.0),
        }
    )
    oof = pd.concat([fold1, fold2], ignore_index=True)
    knots = ["q_0.10", "q_0.50", "q_0.90"]
    base = oof[knots].copy()
    group = np.array(["15-24"] * len(oof))
    backtest = nested_offset_backtest(
        oof,
        base,
        group,
        folds=(2,),
        min_n=300,
        max_minutes=63.0,
        n_boot=20,
        seed=0,
    )
    assert backtest["offset_tables"][2].loc["15-24", "q_0.50"] == pytest.approx(1.0)
    scored = backtest["scores"].iloc[0]
    assert scored["pinball_q50_raw"] == pytest.approx(0.0)
    assert scored["pinball_q50_cal"] == pytest.approx(0.5)


def test_offset_swing_larger_than_its_mean_is_flagged():
    def _table(value: float) -> pd.DataFrame:
        frame = pd.DataFrame({"q_0.50": [value]}, index=pd.Index(["<15"], name="group"))
        frame.insert(0, "n", 400)
        return frame

    flagged = offset_stability_flags({3: _table(1.0), 4: _table(1.0), 5: _table(5.0)})
    assert list(flagged["knot"]) == ["q_0.50"]
    assert flagged.iloc[0]["swing"] == pytest.approx(4.0)

    stable = offset_stability_flags({3: _table(1.0), 4: _table(1.1), 5: _table(1.0)})
    assert stable.empty
