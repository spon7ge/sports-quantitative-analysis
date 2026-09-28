import numpy as np
import pandas as pd
import pytest

from src.mlb.evaluation import strikeout_walk_forward
from src.mlb.evaluation.strikeout_walk_forward import (
    Lock,
    block_metrics,
    reliability_bins,
    score_locked_seasons,
    select_2020,
)
from src.mlb.evaluation.walk_forward_fit import UnestimatedDispersion, WalkForwardFit
from src.mlb.evaluation.walk_forward_snapshot import SnapshotStore
from src.mlb.models.nb_scores import half_point_probabilities, nb2_quantile


def _intercept_only(train, target, features, binary, l2, offset=None):
    return WalkForwardFit(
        feature_names=(),
        coef=np.array([np.log(0.25)]),
        alpha=0.1,
        method="fake",
        centers={},
        scales={},
        medians={},
        dispersion_estimated=True,
    )


def test_2020_tie_breaks_toward_the_smaller_kappa_then_the_smaller_cap(monkeypatch):
    calls = []

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        calls.append((kappa, rest_cap, m, score_strikeouts))
        return pd.DataFrame([
            {
                "pitcher_id": 1,
                "game_pk": 1,
                "season": 2020,
                "predicted_bf_oof": 20.0,
                "batters_faced": 20,
                "strikeouts": 5,
                "pitcher_k_last10": 5.0,
                "pitcher_bf_last10": 20.0,
                "pitcher_k_last3": 5.0,
                "pitcher_bf_last3": 20.0,
                "pitcher_k_season": 5.0,
                "pitcher_bf_season": 20.0,
                "opponent_k_last10": 5.0,
                "opponent_bf_last10": 20.0,
                "opponent_k_season": 5.0,
                "opponent_bf_season": 20.0,
                "league_k_per_bf_prior": 0.25,
            }
        ])

    monkeypatch.setattr(
        "src.mlb.evaluation.strikeout_walk_forward.walk_blocks",
        fake_walk,
    )
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", _intercept_only)
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.kappa == 1
    assert lock.rest_cap == 14
    assert lock.m == 25
    assert len(calls) == 20
    assert all(m == 25 and not scored for _, _, m, scored in calls)
    assert isinstance(lock.store, SnapshotStore)
    assert len(lock.store.frame()) == 1


def test_2020_skips_a_candidate_whose_dispersion_is_unestimated(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        if kappa == 1:
            raise UnestimatedDispersion("no dispersion")
        return pd.DataFrame([
            {
                "pitcher_id": 1,
                "game_pk": 1,
                "season": 2020,
                "predicted_bf_oof": 20.0 + rest_cap,
                "batters_faced": 20,
                "strikeouts": 5,
                "pitcher_k_last10": 5.0,
                "pitcher_bf_last10": 20.0,
                "pitcher_k_last3": 5.0,
                "pitcher_bf_last3": 20.0,
                "pitcher_k_season": 0.0,
                "pitcher_bf_season": 0.0,
                "opponent_k_last10": 5.0,
                "opponent_bf_last10": 20.0,
                "opponent_k_season": 0.0,
                "opponent_bf_season": 0.0,
                "league_k_per_bf_prior": 0.25,
            }
        ])

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", _intercept_only)
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.kappa == 2
    assert lock.rest_cap == 14
    stored = lock.store.frame()
    assert stored.loc[0, "pitcher_k_per_bf_season_to_date_smoothed"] == pytest.approx(
        stored.loc[0, "pitcher_k_per_bf_last10_smoothed"]
    )


def _predictions():
    return pd.DataFrame({
        "predicted_bf_oof": [20.0, 22.0, 24.0, 18.0, 25.0],
        "batters_faced": [21, 20, 25, 18, 27],
        "strikeouts": [3, 6, 8, 2, 5],
        "predicted_strikeout_mean": [4.0, 5.0, 6.0, 4.5, 5.5],
        "negative_binomial_dispersion": [0.1] * 5,
    })


def test_report_labels_calibration_in_the_large():
    predictions = _predictions()
    metrics = block_metrics(predictions)
    mu = predictions["predicted_strikeout_mean"].to_numpy()
    alpha = predictions["negative_binomial_dispersion"].to_numpy()
    y = predictions["strikeouts"].to_numpy()
    over, _ = half_point_probabilities(mu, alpha, 4.5)
    assert metrics["mean_predicted_over_4_5"] == pytest.approx(over.mean())
    assert metrics["observed_over_rate_4_5"] == pytest.approx((y > 4.5).mean())
    q25, q75 = nb2_quantile(mu, alpha, 0.25), nb2_quantile(mu, alpha, 0.75)
    q10, q90 = nb2_quantile(mu, alpha, 0.10), nb2_quantile(mu, alpha, 0.90)
    assert metrics["coverage_q25_q75"] == pytest.approx(((y >= q25) & (y <= q75)).mean())
    assert metrics["coverage_q10_q90"] == pytest.approx(((y >= q10) & (y <= q90)).mean())
    assert metrics["nominal_q25_q75"] == 0.50
    assert metrics["nominal_q10_q90"] == 0.80
    assert metrics["workload_bias"] == pytest.approx(-0.4)
    assert "calibration_in_the_large_4_5" not in metrics


def test_reliability_bins_drop_a_bin_under_200_rows():
    predictions = pd.DataFrame({
        "strikeouts": [5] * 250,
        "predicted_strikeout_mean": [5.0] * 250,
        "negative_binomial_dispersion": [0.1] * 250,
    })
    bins = reliability_bins(predictions, 4.5)
    assert list(bins.columns) == [
        "bin_left", "bin_right", "n", "mean_predicted_over", "observed_over_rate"
    ]
    assert len(bins) == 1
    assert bins.loc[0, "n"] == 250
    assert bins.loc[0, "observed_over_rate"] == 1.0
    assert reliability_bins(predictions.head(199), 4.5).empty


def _scored_rows(season, block, dates, status="ok"):
    rows = _predictions().iloc[: len(dates)].copy()
    rows["pitcher_id"] = 1
    rows["game_pk"] = [f"{block}-{i}" for i in range(len(dates))]
    rows["season"] = season
    rows["walk_forward_block"] = block
    rows["game_date"] = dates
    rows["fit_status"] = status
    return rows


def test_locked_report_has_one_row_per_block_and_an_aggregate(monkeypatch):
    seen = {}

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        seen.update(m=m, kappa=kappa, rest_cap=rest_cap, through=through_season, scored=score_strikeouts)
        return pd.concat([
            _scored_rows(2020, "2020-01", ["2020-07-24", "2020-07-25"]),
            _scored_rows(2021, "2021-01", ["2021-04-01", "2021-04-02", "2021-04-03"]),
            _scored_rows(2021, "2021-02", ["2021-04-29", "2021-04-30"]),
        ], ignore_index=True)

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    lock = Lock(m=50, kappa=2, rest_cap=21, store=SnapshotStore())
    report = score_locked_seasons(pd.DataFrame(), pd.DataFrame(), lock, [2021])
    assert seen == {"m": 50, "kappa": 2, "rest_cap": 21, "through": 2021, "scored": True}
    assert list(report["walk_forward_block"]) == ["2021-01", "2021-02", "all"]
    assert list(report["role"]) == ["reported", "reported", "aggregate"]
    second = report.iloc[1]
    assert second["n_train"] == 5
    assert second["train_start"] == "2020-07-24"
    assert second["train_end"] == "2021-04-03"
    assert second["validation_start"] == "2021-04-29"
    assert second["n_validation"] == 2
    assert report.iloc[2]["n_validation"] == 5


def test_reported_block_with_unestimated_dispersion_aborts(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return _scored_rows(2021, "2021-01", ["2021-04-01"], status="unestimated_dispersion")

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    lock = Lock(m=50, kappa=2, rest_cap=21, store=SnapshotStore())
    with pytest.raises(UnestimatedDispersion):
        score_locked_seasons(pd.DataFrame(), pd.DataFrame(), lock, [2021])
