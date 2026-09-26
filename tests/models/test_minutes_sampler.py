import numpy as np
import pytest

from models.shared.minutes_sampler import (
    KNOT_FLOOR,
    QUANTILE_LEVELS,
    MinuteTailTables,
    prepare_quantile_grid,
    quantile_minutes,
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
    base = quantile_minutes(u, _grid(SORTED), np.array([1]), np.array([1]), TABLES)
    doubled = SORTED.copy()
    doubled[0] = 20
    scaled = quantile_minutes(u, _grid(doubled), np.array([1]), np.array([1]), TABLES)
    shifted = SORTED.copy()
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
