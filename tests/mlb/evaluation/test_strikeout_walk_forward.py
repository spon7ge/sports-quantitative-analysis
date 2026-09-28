import numpy as np
import pandas as pd

from src.mlb.evaluation import strikeout_walk_forward
from src.mlb.evaluation.strikeout_walk_forward import walk_blocks
from src.mlb.evaluation.walk_forward_fit import UnestimatedDispersion, WalkForwardFit
from src.mlb.schemas import PMF_COLUMNS


def _start(rng, game_pk, date, hour=17, is_home=None):
    bf = int(rng.integers(14, 32))
    return {
        "pitcher_id": 1,
        "game_pk": game_pk,
        "game_date": date,
        "scheduled_start_utc": pd.Timestamp(f"{date} {hour:02d}:05", tz="UTC"),
        "season": int(date[:4]),
        "is_home": bool(rng.integers(0, 2)) if is_home is None else is_home,
        "opponent_team_id": 100,
        "strikeouts": int(rng.integers(1, max(2, bf // 3))),
        "batters_faced": bf,
        "outs": int(rng.integers(9, 21)),
        "pitches": int(rng.integers(60, 110)),
    }


def _starts(with_2020=False):
    rng = np.random.default_rng(7)
    rows = []
    pk = 1000
    for day in range(1, 31):
        rows.append(_start(rng, pk, f"2018-04-{day:02d}"))
        pk += 1
    for day in range(1, 11):
        rows.append(_start(rng, pk, f"2019-04-{day:02d}"))
        pk += 1
    rows.append(_start(rng, pk, "2019-04-10", hour=23))
    pk += 1
    if with_2020:
        for day in range(1, 6):
            rows.append(_start(rng, pk, f"2020-07-{day + 23:02d}"))
            pk += 1
    return pd.DataFrame(rows)


def _calendar(with_2020=False):
    rows = [
        {"season": 2018, "regular_season_open_date": "2018-03-29", "regular_season_close_date": "2018-09-30"},
        {"season": 2019, "regular_season_open_date": "2019-03-28", "regular_season_close_date": "2019-09-29"},
    ]
    if with_2020:
        rows.append(
            {"season": 2020, "regular_season_open_date": "2020-07-23", "regular_season_close_date": "2020-09-27"}
        )
    return pd.DataFrame(rows)


def _walk(through_season, score_strikeouts=False, with_2020=False):
    return walk_blocks(
        _starts(with_2020),
        _calendar(with_2020),
        m=50,
        kappa=3,
        rest_cap=30,
        through_season=through_season,
        score_strikeouts=score_strikeouts,
    )


def test_2019_offset_ignores_a_same_day_result():
    frame = _walk(2019)
    april_10 = frame.loc[frame["game_date"] == "2019-04-10"]
    assert len(april_10) == 2
    assert april_10["predicted_bf_oof"].notna().all()
    assert april_10["pitcher_prior_regular_starts"].nunique() == 1
    assert frame.loc[frame["season"] == 2018, "predicted_bf_oof"].isna().all()


def test_2018_first_date_is_not_stored_and_later_dates_are_stored_once():
    frame = _walk(2019)
    season_2018 = frame.loc[frame["season"] == 2018]
    assert "2018-04-01" not in set(season_2018["game_date"])
    assert len(season_2018) == 29
    assert not frame.duplicated(["pitcher_id", "game_pk"]).any()
    assert "batters_faced" in frame.columns
    assert "scheduled_start_utc" in frame.columns


def test_unscored_walk_omits_strikeout_columns():
    frame = _walk(2019)
    for column in ("predicted_strikeout_mean", "negative_binomial_dispersion", *PMF_COLUMNS):
        assert column not in frame.columns


def test_later_block_keeps_the_2019_league_prior():
    stopped = _walk(2019)
    extended = _walk(2020, with_2020=True)
    assert (extended["season"] == 2020).any()
    key = ["pitcher_id", "game_pk"]
    before = stopped.loc[stopped["season"] == 2019].set_index(key)["league_k_per_bf_prior"]
    after = extended.loc[extended["season"] == 2019].set_index(key)["league_k_per_bf_prior"]
    pd.testing.assert_series_equal(before.sort_index(), after.sort_index())


def test_scored_walk_adds_strikeout_columns_without_2018_training_rows():
    frame = _walk(2019, score_strikeouts=True)
    for column in ("predicted_strikeout_mean", "negative_binomial_dispersion", *PMF_COLUMNS, "fit_status"):
        assert column in frame.columns
    season_2019 = frame.loc[frame["season"] == 2019]
    assert season_2019["predicted_bf_oof"].notna().all()
    assert (season_2019["fit_status"] == "insufficient_history").all()
    assert frame["predicted_strikeout_mean"].isna().all()


def _patch_strikeout_fit(monkeypatch, strikeout_fit):
    real_fit = strikeout_walk_forward.fit_walk_forward_nb2
    seen = {}

    def fake_fit(train, target, features, binary, l2, offset=None):
        if target != "strikeouts":
            return real_fit(train, target, features, binary, l2, offset=offset)
        seen["seasons"] = set(train["season"])
        seen["offset"] = offset
        seen["predicted_bf_oof"] = train["predicted_bf_oof"].to_numpy(dtype=float)
        return strikeout_fit()

    monkeypatch.setattr(strikeout_walk_forward, "fit_walk_forward_nb2", fake_fit)
    return seen


def test_scored_block_uses_the_bf_offset_and_reports_a_pmf(monkeypatch):
    def intercept_only():
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

    seen = _patch_strikeout_fit(monkeypatch, intercept_only)
    frame = _walk(2020, score_strikeouts=True, with_2020=True)
    assert seen["seasons"] == {2019}
    np.testing.assert_allclose(seen["offset"], np.log(seen["predicted_bf_oof"]))
    block_2020 = frame.loc[frame["season"] == 2020]
    assert (block_2020["fit_status"] == "ok").all()
    np.testing.assert_allclose(
        block_2020["predicted_strikeout_mean"], 0.25 * block_2020["predicted_bf_oof"]
    )
    assert (block_2020["negative_binomial_dispersion"] == 0.1).all()
    np.testing.assert_allclose(block_2020.loc[:, list(PMF_COLUMNS)].sum(axis=1), 1.0, atol=1e-9)


def test_unestimated_strikeout_dispersion_keeps_rows_unscored(monkeypatch):
    def raise_unestimated():
        raise UnestimatedDispersion("no dispersion")

    _patch_strikeout_fit(monkeypatch, raise_unestimated)
    frame = _walk(2020, score_strikeouts=True, with_2020=True)
    block_2020 = frame.loc[frame["season"] == 2020]
    assert len(block_2020) == 5
    assert (block_2020["fit_status"] == "unestimated_dispersion").all()
    assert block_2020["predicted_bf_oof"].notna().all()
    assert block_2020.loc[:, ["predicted_strikeout_mean", *PMF_COLUMNS]].isna().all().all()
