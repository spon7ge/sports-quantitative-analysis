import numpy as np
import pandas as pd
import pytest

from models.shared.minutes_sampler import (
    HOLDOUT_START,
    KNOT_FLOOR,
    QUANTILE_LEVELS,
    MinuteTailTables,
    build_tail_tables,
    groups_for_frame,
    load_tail_sidecar,
    prepare_quantile_grid,
    quantile_minutes,
    save_tail_sidecar,
)


def _grid(values):
    grid = np.asarray(values, dtype=float)
    assert grid.shape == (11,)
    return grid.reshape(1, -1)


def _tables(lower0, upper0, lower1, upper1):
    return MinuteTailTables(
        arrays={
            ("lower", 0): np.asarray(lower0, dtype=float),
            ("upper", 0): np.asarray(upper0, dtype=float),
            ("lower", 1): np.asarray(lower1, dtype=float),
            ("upper", 1): np.asarray(upper1, dtype=float),
        }
    )


SORTED = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
TABLES = _tables([0.2, 1.0], [0.0, 4.0], [0.5, 1.0], [0.0, 10.0])


def test_prepare_floors_without_reordering_levels():
    raw = np.array([-0.1, 0.0005, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    prepared = prepare_quantile_grid(raw)
    assert prepared.shape == (1, 11)
    assert prepared[0, 0] == KNOT_FLOOR
    assert prepared[0, 1] == KNOT_FLOOR
    assert prepared[0, 2] == 1


def test_prepare_rejects_non_finite_knot():
    raw = SORTED.copy()
    raw[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        prepare_quantile_grid(raw)


def test_quantile_minutes_hits_all_eleven_stored_levels():
    u = np.asarray(QUANTILE_LEVELS, dtype=float).reshape(1, -1)
    got = quantile_minutes(u, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    np.testing.assert_allclose(got[0], SORTED)


def test_lower_tail_scales_with_q05_and_upper_tail_shifts_with_q95():
    u = np.array([[0.025, 0.975]])
    grid = np.array([5, *SORTED[1:]], dtype=float)
    base = quantile_minutes(u, _grid(grid), np.array([1]), np.array([1]), TABLES)
    doubled = grid.copy()
    doubled[0] = 2 * grid[0]
    scaled = quantile_minutes(u, _grid(doubled), np.array([1]), np.array([1]), TABLES)
    shifted = grid.copy()
    shifted[-1] = 35
    moved = quantile_minutes(u, _grid(shifted), np.array([1]), np.array([1]), TABLES)
    np.testing.assert_allclose(scaled[0, 0], 2 * base[0, 0])
    np.testing.assert_allclose(moved[0, 1], base[0, 1] + 5)


def test_quantile_minutes_is_non_decreasing_and_reads_each_tail_group():
    u = np.linspace(0, 1, 21).reshape(1, -1)
    got = quantile_minutes(u, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    assert np.all(np.diff(got[0]) >= -1e-12)
    low = np.array([[0.025]])
    bench = quantile_minutes(low, _grid(SORTED), np.array([0]), np.array([0]), TABLES)
    starter = quantile_minutes(low, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    assert bench[0, 0] != pytest.approx(starter[0, 0])
    mixed = quantile_minutes(
        np.array([[0.025, 0.975]]),
        _grid(SORTED),
        np.array([0]),
        np.array([1]),
        TABLES,
    )
    assert mixed[0, 0] == pytest.approx(bench[0, 0])
    assert mixed[0, 1] != pytest.approx(
        quantile_minutes(np.array([[0.975]]), _grid(SORTED), np.array([0]), np.array([0]), TABLES)[0, 0]
    )


def test_draw_above_63_clips_to_63():
    tall = SORTED.copy()
    tall[-1] = 80
    got = quantile_minutes(np.array([[0.99]]), _grid(tall), np.array([1]), np.array([1]), TABLES)
    assert got[0, 0] == 63


def test_bad_u_knot_count_and_unknown_group_raise():
    with pytest.raises(ValueError, match="knot"):
        quantile_minutes(np.array([[0.5]]), SORTED[:10].reshape(1, -1), np.array([1]), np.array([1]), TABLES)
    with pytest.raises(ValueError, match="outside"):
        quantile_minutes(np.array([[1.1]]), _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    with pytest.raises(ValueError, match="group"):
        quantile_minutes(np.array([[0.025]]), _grid(SORTED), np.array([2]), np.array([1]), TABLES)


def _knot_frame(n, q05=10.0, q95=30.0):
    frame = pd.DataFrame(index=np.arange(n))
    for level in QUANTILE_LEVELS:
        if level <= 0.05:
            value = q05
        elif level >= 0.95:
            value = q95
        else:
            value = q05 + (q95 - q05) * (level - 0.05) / 0.90
        frame[f"q_{level:.2f}"] = value
    return frame


def _valid_oof():
    """100 rows, fold 1, pooled tail rate 5%. Group 1 has only r=0.5 and e=10."""
    starter = _knot_frame(20)
    starter["minutes"] = 20.0
    starter.loc[0, "minutes"] = 5.0
    starter.loc[1, "minutes"] = 40.0
    starter["starting"] = 1
    bench = _knot_frame(80)
    bench["minutes"] = 20.0
    bench.loc[0:3, "minutes"] = 2.0
    bench.loc[4:7, "minutes"] = 36.0
    bench["starting"] = 0
    oof = pd.concat([starter, bench], ignore_index=True)
    oof["fold_id"] = 1
    oof["game_date"] = pd.Timestamp("2024-01-15")
    oof["early_stop"] = "train_tail"
    oof["train_tail_frac"] = 0.10
    ranges = {1: {"train_start": "2023-10-01", "train_end": "2024-01-01", "val_start": "2024-01-02", "val_end": "2024-01-15"}}
    return oof, ranges


def test_builder_pins_and_tail_limits_come_from_build_tail_tables():
    oof, ranges = _valid_oof()
    tables = build_tail_tables(oof, folds=[1], fold_ranges=ranges)
    lower = tables.arrays[("lower", 1)]
    upper = tables.arrays[("upper", 1)]
    assert lower[-1] == 1
    assert upper[0] == 0
    assert set(np.round(lower, 5)) == {0.5, 1.0}
    assert set(np.round(upper, 5)) == {0.0, 10.0}
    row = oof.iloc[[0]]
    grids = row[[f"q_{level:.2f}" for level in QUANTILE_LEVELS]].to_numpy(dtype=float)
    eps = 1e-6
    got = quantile_minutes(
        np.array([[0.05 - eps, 0.95 + eps]]),
        grids,
        np.array([1]),
        np.array([1]),
        tables,
    )
    q05 = prepare_quantile_grid(grids)[0, 0]
    q95 = prepare_quantile_grid(grids)[0, -1]
    assert abs(got[0, 0] - q05) / q05 < 1e-3
    assert abs(got[0, 1] - q95) <= 1e-3
    stripped = MinuteTailTables(arrays=dict(tables.arrays))
    stripped.arrays[("lower", 1)] = lower[:-1]
    stripped.arrays[("upper", 1)] = upper[1:]
    bare = quantile_minutes(
        np.array([[0.05 - eps, 0.95 + eps]]),
        grids,
        np.array([1]),
        np.array([1]),
        stripped,
    )
    assert bare[0, 0] == pytest.approx(0.5 * q05)
    assert bare[0, 1] == pytest.approx(q95 + 10)


def test_missing_requested_fold_raises_before_miss_rate():
    oof, ranges = _valid_oof()
    oof["fold_id"] = 4
    oof["minutes"] = 20.0
    with pytest.raises(ValueError, match="fold"):
        build_tail_tables(oof, folds=[1, 2, 3, 4], fold_ranges=ranges)


def test_miss_rate_outside_band_raises():
    oof, ranges = _valid_oof()
    oof["minutes"] = 1.0
    with pytest.raises(ValueError, match="miss rate"):
        build_tail_tables(oof, folds=[1], fold_ranges=ranges)


def test_group_missing_from_the_data_raises_at_build():
    oof, ranges = _valid_oof()
    oof = oof.loc[oof["starting"] == 1].reset_index(drop=True)
    oof["minutes"] = 20.0
    oof.loc[0, "minutes"] = 5.0
    oof.loc[1, "minutes"] = 40.0
    with pytest.raises(ValueError, match="pin"):
        build_tail_tables(oof, folds=[1], fold_ranges=ranges)


def test_builder_rejects_a_non_finite_knot():
    oof, ranges = _valid_oof()
    oof.loc[0, "q_0.50"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_tail_tables(oof, folds=[1], fold_ranges=ranges)


def test_builder_rejects_bad_rows_and_rules():
    oof, ranges = _valid_oof()
    non_positive = oof.copy()
    non_positive.loc[0, "minutes"] = 0
    with pytest.raises(ValueError, match="y > 0"):
        build_tail_tables(non_positive, folds=[1], fold_ranges=ranges)
    late = oof.copy()
    late["game_date"] = HOLDOUT_START
    with pytest.raises(ValueError, match="2025-10-21"):
        build_tail_tables(late, folds=[1], fold_ranges=ranges)
    optimistic = oof.copy()
    optimistic["early_stop"] = "validation"
    with pytest.raises(ValueError, match="train_tail"):
        build_tail_tables(optimistic, folds=[1], fold_ranges=ranges)
    bad_role = oof.copy()
    bad_role.loc[0, "starting"] = 2
    with pytest.raises(ValueError, match="starting"):
        build_tail_tables(bad_role, folds=[1], fold_ranges=ranges)
    with pytest.raises(ValueError, match="grouping"):
        build_tail_tables(
            oof,
            folds=[1],
            fold_ranges=ranges,
            grouping={"lower": "q50_tier", "upper": "starting"},
        )


from models.shared import train as train_mod
from models.shared.train import fit_quantile_models, run_walk_forward


class _FakeBooster:
    def __init__(self, **kwargs):
        self.eval_rows = None

    def fit(self, X, y, eval_set=None, verbose=False):
        self.eval_rows = len(eval_set[0][0])

    def predict(self, X):
        return np.arange(len(X), dtype=float)


def test_train_tail_one_date_raises_before_any_booster(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("booster constructed")

    monkeypatch.setattr(train_mod, "XGBRegressor", boom)
    X = pd.DataFrame({"a": [1.0]})
    y = pd.Series([2.0])
    with pytest.raises(ValueError, match="two training dates"):
        fit_quantile_models(
            X, y, X, y,
            xgb_params={"n_estimators": 1},
            quantiles=[0.5],
            early_stop="train_tail",
            train_dates=pd.to_datetime(["2024-01-01"]),
        )


def test_train_tail_walk_forward_stops_on_training_dates_and_keeps_every_fold(monkeypatch):
    monkeypatch.setattr(train_mod, "XGBRegressor", _FakeBooster)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for day in dates:
        for player in (1, 2):
            rows.append({"game_date": day, "a": 1.0, "minutes": 10.0, "starting": player % 2})
    frame = pd.DataFrame(rows)
    X = frame[["a"]]
    y = frame["minutes"]
    result = run_walk_forward(
        X, y, frame,
        xgb_params={"n_estimators": 1},
        quantiles=[0.5],
        n_folds=1,
        train_frac=0.5,
        step_frac=0.2,
        early_stop="train_tail",
    )
    oof = result["oof"]
    assert set(oof["fold_id"]) == {1}
    assert (oof["early_stop"] == "train_tail").all()
    assert 1 in result["fold_ranges"]
    assert "q_0.50" in oof.columns
    assert len(oof) == int(frame["game_date"].isin(frame["game_date"].unique()[5:7]).sum())
    assert result["models_last"]["q_0.50"].eval_rows < 5


def _four_fold_tables():
    frames = []
    ranges = {}
    for fold in (1, 2, 3, 4):
        oof, one_range = _valid_oof()
        oof["fold_id"] = fold
        oof["game_date"] = pd.Timestamp("2024-01-01") + pd.Timedelta(days=fold)
        frames.append(oof)
        ranges[fold] = {
            "train_start": "2023-10-01",
            "train_end": f"2024-01-0{fold}",
            "val_start": f"2024-01-1{fold}",
            "val_end": f"2024-01-2{fold}",
        }
    oof = pd.concat(frames, ignore_index=True)
    return build_tail_tables(oof, folds=[1, 2, 3, 4], fold_ranges=ranges)


def test_loader_accepts_folds_1_through_4_only(tmp_path):
    tables = _four_fold_tables()
    path = save_tail_sidecar(tables, tmp_path / "tails.joblib")
    loaded = load_tail_sidecar(path)
    assert loaded.folds == (1, 2, 3, 4)
    partial, ranges = _valid_oof()
    partial_tables = build_tail_tables(partial, folds=[1], fold_ranges=ranges)
    partial_path = save_tail_sidecar(partial_tables, tmp_path / "partial.joblib")
    with pytest.raises(ValueError, match="fold"):
        load_tail_sidecar(partial_path)


def test_loader_rejects_a_different_floor_or_level_list(tmp_path):
    tables = _four_fold_tables()
    path = save_tail_sidecar(tables, tmp_path / "tails.joblib")
    payload = __import__("joblib").load(path)
    payload["floor"] = 0.01
    __import__("joblib").dump(payload, path)
    with pytest.raises(ValueError, match="floor"):
        load_tail_sidecar(path)
    payload = __import__("joblib").load(save_tail_sidecar(tables, tmp_path / "levels.joblib"))
    payload["quantile_levels"] = list(np.linspace(0.05, 0.95, 11))
    __import__("joblib").dump(payload, tmp_path / "levels.joblib")
    with pytest.raises(ValueError, match="quantile"):
        load_tail_sidecar(tmp_path / "levels.joblib")


def test_missing_starting_raises_at_draw_time_and_is_not_bench():
    tables = _four_fold_tables()
    with pytest.raises(ValueError, match="starting"):
        groups_for_frame(pd.Series([pd.NA, 1]), tables)
    lower, upper = groups_for_frame(pd.Series([0, 1]), tables)
    np.testing.assert_array_equal(lower, [0, 1])
    np.testing.assert_array_equal(upper, [0, 1])
    tables.grouping = {"lower": "q50_tier", "upper": "starting"}
    with pytest.raises(ValueError, match="grouping"):
        groups_for_frame(pd.Series([0, 1]), tables)

