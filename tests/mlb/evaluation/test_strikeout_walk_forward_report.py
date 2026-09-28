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
from src.mlb.models.workload import GlmFitError


def _intercept_only(train, target, features, binary, l2, offset=None, require_convergence=False):
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


def _rows_2019(**kwargs):
    return [_tuning_row(2019, -i, 20.0, **kwargs) for i in range(3)]


def test_2020_tie_breaks_toward_the_smaller_kappa_then_the_smaller_cap(monkeypatch):
    calls = []

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        calls.append((kappa, rest_cap, m, score_strikeouts))
        return pd.DataFrame([*_rows_2019(), _tuning_row(2020, 1, 20.0)])

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
    assert len(lock.store.frame()) == 4


def test_2020_skips_a_candidate_whose_dispersion_is_unestimated(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        if kappa == 2:
            raise UnestimatedDispersion("no dispersion")
        if kappa == 1:
            return pd.DataFrame([
                *_rows_2019(season_sums=0.0),
                _tuning_row(2020, 1, 20.0, season_sums=0.0),
                _tuning_row(2020, 2, np.nan, season_sums=0.0, status="unestimated_dispersion"),
            ])
        return pd.DataFrame([
            *_rows_2019(season_sums=0.0),
            _tuning_row(2020, 1, 20.0 + rest_cap, season_sums=0.0),
        ])

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", _intercept_only)
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.kappa == 3
    assert lock.rest_cap == 14
    stored = lock.store.frame()
    row = stored[stored["season"] == 2020].iloc[0]
    assert row["pitcher_k_per_bf_season_to_date_smoothed"] == pytest.approx(
        row["pitcher_k_per_bf_last10_smoothed"]
    )


def test_2020_skips_a_workload_candidate_whose_glm_fit_raises(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        if kappa == 1:
            raise GlmFitError("rank deficient")
        return pd.DataFrame([*_rows_2019(), _tuning_row(2020, 1, 20.0 + kappa)])

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", _intercept_only)
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.kappa == 2


def test_2020_skips_an_m_whose_strikeout_glm_fit_raises(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return pd.DataFrame([*_rows_2019(), _tuning_row(2020, 1, 20.0)])

    calls = []

    def raise_first_m(train, target, features, binary, l2, offset=None, require_convergence=False):
        assert require_convergence is True
        calls.append(target)
        if len(calls) == 1:
            raise GlmFitError("rank deficient")
        return _intercept_only(train, target, features, binary, l2, offset)

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", raise_first_m)
    lock = select_2020(pd.DataFrame(), pd.DataFrame())
    assert lock.m == 50


def test_2020_skips_an_m_without_legal_strikeout_training_rows(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return pd.DataFrame([_tuning_row(2020, 1, 20.0)])

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("strikeout model fit on the rows it scores")

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", fail_if_fit)
    with pytest.raises(UnestimatedDispersion, match="strikeout candidate"):
        select_2020(pd.DataFrame(), pd.DataFrame())


def test_2020_does_not_fit_strikeouts_on_fewer_than_three_training_rows(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return pd.DataFrame([*_rows_2019()[:2], _tuning_row(2020, 1, 20.0)])

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("strikeout model fit on fewer than three rows")

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", fail_if_fit)
    with pytest.raises(UnestimatedDispersion, match="strikeout candidate"):
        select_2020(pd.DataFrame(), pd.DataFrame())


def test_2020_strikeout_blocks_refit_on_rows_dated_before_each_block(monkeypatch):
    seen = []

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        rows = [*_rows_2019(), _tuning_row(2020, 1, 20.0), _tuning_row(2020, 2, 20.0)]
        rows[-1].update(walk_forward_block="2020-02", game_date="2020-08-21")
        return pd.DataFrame(rows)

    def recording_fit(train, target, features, binary, l2, offset=None, require_convergence=False):
        seen.append((sorted(train["game_pk"]), train["game_date"].max()))
        return _intercept_only(train, target, features, binary, l2, offset)

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", recording_fit)
    select_2020(pd.DataFrame(), pd.DataFrame())
    (first_pks, first_last), (second_pks, second_last) = seen[:2]
    assert first_pks == [-2, -1, 0]
    assert second_pks == [-2, -1, 0, 1]
    assert first_last < "2020-07-24"
    assert second_last < "2020-08-21"


def test_duplicate_feature_dropped_before_the_fit_is_recorded_on_the_model(monkeypatch):
    seen = {}

    def fake_fit(train, target, features, binary, l2, offset=None, require_convergence=False):
        seen["features"] = features
        model = _intercept_only(train, target, features, binary, l2, offset)
        model.dropped_features = ("constant",)
        return model

    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", fake_fit)
    train = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "constant": [0, 0, 0], "y": [1, 2, 3]})
    model = strikeout_walk_forward._fit_distinct(train, "y", ("a", "b", "constant"), (), 1.0)
    assert seen["features"] == ("a", "constant")
    assert model.dropped_features == ("b", "constant")


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
    assert list(report["role"]) == ["reported", "reported", "reported"]
    second = report.iloc[1]
    assert second["n_train"] == 5
    assert second["train_start"] == "2020-07-24"
    assert second["train_end"] == "2021-04-03"
    assert second["validation_start"] == "2021-04-29"
    assert second["n_validation"] == 2
    aggregate = report.iloc[2]
    assert aggregate["n_validation"] == 5
    pooled = fake_walk(None, None, m=50, kappa=2, rest_cap=21, through_season=2021, score_strikeouts=True)
    expected = block_metrics(pooled[pooled["season"] == 2021])
    for column, value in expected.items():
        assert aggregate[column] == pytest.approx(value, nan_ok=True)


def _many_rows(season, block, date, n, mu):
    return pd.DataFrame({
        "pitcher_id": 1,
        "game_pk": [f"{block}-{i}" for i in range(n)],
        "season": season,
        "walk_forward_block": block,
        "game_date": date,
        "fit_status": "ok",
        "predicted_bf_oof": 22.0,
        "batters_faced": 22,
        "strikeouts": [i % 10 for i in range(n)],
        "predicted_strikeout_mean": mu,
        "negative_binomial_dispersion": 0.1,
    })


def test_multi_role_report_keeps_one_aggregate_and_reliability_per_role(monkeypatch):
    reported = _many_rows(2024, "2024-01", "2024-04-01", 250, 3.0)
    holdout = _many_rows(2025, "2025-01", "2025-04-01", 250, 8.0)

    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return pd.concat([reported, holdout], ignore_index=True)

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    lock = Lock(m=50, kappa=2, rest_cap=21, store=SnapshotStore())
    report = score_locked_seasons(pd.DataFrame(), pd.DataFrame(), lock, [2024, 2025])
    aggregates = report[report["walk_forward_block"] == "all"]
    assert list(aggregates["role"]) == ["reported", "holdout"]
    assert "aggregate" not in set(report["role"])
    for role, rows in (("reported", reported), ("holdout", holdout)):
        row = aggregates[aggregates["role"] == role].iloc[0]
        assert row["n_validation"] == 250
        for column, value in block_metrics(rows).items():
            assert row[column] == pytest.approx(value, nan_ok=True)

    reliability = report.attrs["reliability"]
    assert set(reliability) == {"reported", "holdout"}
    for role, rows in (("reported", reported), ("holdout", holdout)):
        at_45 = reliability[role]
        at_45 = at_45[at_45["line"] == 4.5].drop(columns="line").reset_index(drop=True)
        pd.testing.assert_frame_equal(at_45, reliability_bins(rows, 4.5), check_dtype=False)
        assert at_45["n"].sum() == 250
    assert not reliability["reported"]["mean_predicted_over"].equals(
        reliability["holdout"]["mean_predicted_over"]
    )


def test_locked_report_rejects_a_season_outside_2019_to_2025(monkeypatch):
    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", lambda *a, **k: pytest.fail("walked"))
    lock = Lock(m=50, kappa=2, rest_cap=21, store=SnapshotStore())
    for season in (2018, 2026):
        with pytest.raises(ValueError, match="outside"):
            score_locked_seasons(pd.DataFrame(), pd.DataFrame(), lock, [2021, season])


def test_reported_block_with_unestimated_dispersion_aborts(monkeypatch):
    def fake_walk(starts, calendar, *, m, kappa, rest_cap, through_season, score_strikeouts):
        return _scored_rows(2021, "2021-01", ["2021-04-01"], status="unestimated_dispersion")

    monkeypatch.setattr(strikeout_walk_forward, "walk_blocks", fake_walk)
    lock = Lock(m=50, kappa=2, rest_cap=21, store=SnapshotStore())
    with pytest.raises(UnestimatedDispersion):
        score_locked_seasons(pd.DataFrame(), pd.DataFrame(), lock, [2021])
