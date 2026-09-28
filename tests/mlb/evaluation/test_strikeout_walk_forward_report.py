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


def _tuning_row(season, game_pk, predicted_bf, *, season_sums=5.0, status="ok"):
    return {
        "pitcher_id": 1,
        "game_pk": game_pk,
        "season": season,
        "walk_forward_block": f"{season}-01",
        "game_date": f"{season}-07-24",
        "fit_status": status,
        "predicted_bf_oof": predicted_bf,
        "batters_faced": 20,
        "strikeouts": 5,
        "home_flag": 1,
        "season_starts_prior": 3,
        "pitcher_prior_regular_starts": 10,
        "pitcher_prior_bf_last10": 200.0,
        "opponent_prior_starts_observed": 10,
        "pitcher_k_last10": 5.0,
        "pitcher_bf_last10": 20.0,
        "pitcher_k_last3": 5.0,
        "pitcher_bf_last3": 20.0,
        "pitcher_k_season": season_sums,
        "pitcher_bf_season": 4 * season_sums,
        "opponent_k_last10": 5.0,
        "opponent_bf_last10": 20.0,
        "opponent_k_season": season_sums,
        "opponent_bf_season": 4 * season_sums,
        "league_k_per_bf_prior": 0.25,
    }


def test_2020_tie_breaks_toward_the_smaller_kappa_then_the_smaller_cap(monkeypatch):
    calls = []

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        calls.append((kappa, rest_cap, m, score_strikeouts))
        return pd.DataFrame([
            _tuning_row(2019, 0, 20.0),
            _tuning_row(2020, 1, 20.0),
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
    assert len(lock.store.frame()) == 2


def test_2020_skips_a_candidate_whose_dispersion_is_unestimated(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        if kappa == 2:
            raise UnestimatedDispersion("no dispersion")
        if kappa == 1:
            return pd.DataFrame([
                _tuning_row(2019, 0, 20.0, season_sums=0.0),
                _tuning_row(2020, 1, 20.0, season_sums=0.0),
                _tuning_row(2020, 2, np.nan, season_sums=0.0, status="unestimated_dispersion"),
            ])
        return pd.DataFrame([
            _tuning_row(2019, 0, 20.0, season_sums=0.0),
            _tuning_row(2020, 1, 20.0 + rest_cap, season_sums=0.0),
        ])

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", _intercept_only)
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.kappa == 3
    assert lock.rest_cap == 14
    stored = lock.store.frame()
    assert stored.loc[1, "pitcher_k_per_bf_season_to_date_smoothed"] == pytest.approx(
        stored.loc[1, "pitcher_k_per_bf_last10_smoothed"]
    )


def test_2020_skips_an_m_without_legal_strikeout_training_rows(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return pd.DataFrame([_tuning_row(2020, 1, 20.0)])

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("strikeout model fit on the rows it scores")

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", fail_if_fit)
    with pytest.raises(UnestimatedDispersion, match="strikeout candidate"):
        select_2020(pd.DataFrame(), pd.DataFrame())


def test_2020_strikeout_blocks_refit_on_rows_dated_before_each_block(monkeypatch):
    seen = []

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        rows = [_tuning_row(2019, 0, 20.0), _tuning_row(2020, 1, 20.0), _tuning_row(2020, 2, 20.0)]
        rows[2].update(walk_forward_block="2020-02", game_date="2020-08-21")
        return pd.DataFrame(rows)

    def recording_fit(train, target, features, binary, l2, offset=None):
        seen.append(sorted(train["game_pk"]))
        return _intercept_only(train, target, features, binary, l2, offset)

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", recording_fit)
    select_2020(pd.DataFrame(), pd.DataFrame())
    assert seen[:2] == [[0], [0, 1]]


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
