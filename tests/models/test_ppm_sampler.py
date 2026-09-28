"""Rate quantile function: body knots, scoreless atom, cap at 6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from models.shared.minutes_sampler import QUANTILE_LEVELS
from models.shared.oos import RATE_CAP, capped_rate
from models.shared.ppm_sampler import (
    HOLDOUT_START,
    build_tail_tables,
    cdf,
    cdf_left,
    fit_preholdout_tails,
    load_tail_sidecar,
    ppf,
    prepare_quantile_grid,
    randomized_pit,
    sample_rate,
    save_tail_sidecar,
    starting_from_gamelogs,
)

ROOT = Path(__file__).resolve().parents[2]
TAILS_PATH = ROOT / "models" / "saved_models" / "pts_nba_tails_2026-04-12.joblib"
OOS_PATH = ROOT / "data" / "oos" / "nba" / "minutes_rate_oos.parquet"


def _grid(values):
    grid = np.asarray(values, dtype=float)
    assert grid.shape == (11,)
    return grid.reshape(1, -1)


def _tables(lower0, upper0, lower1, upper1):
    from models.shared.ppm_sampler import RateTailTables

    return RateTailTables(
        arrays={
            ("lower", 0): np.asarray(lower0, dtype=float),
            ("upper", 0): np.asarray(upper0, dtype=float),
            ("lower", 1): np.asarray(lower1, dtype=float),
            ("upper", 1): np.asarray(upper1, dtype=float),
        }
    )


# Group 1 has no exact zero. Group 0's lower table is 3/5 zeros.
ZERO_TABLE = np.array([0.0, 0.0, 0.0, 0.5, 1.0])
TABLES = _tables(ZERO_TABLE, [0.0, 0.4], [0.4, 1.0], [0.0, 0.5])
INTERIOR = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20])


def test_prepare_sorts_and_keeps_exact_zero():
    raw = np.array([0.4, 0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    prepared = prepare_quantile_grid(raw)
    assert prepared[0, 0] == 0.0
    assert prepared[0, 1] == 0.2
    assert np.all(np.diff(prepared[0]) >= 0)


def test_ppf_at_the_eleven_levels_returns_sorted_knots():
    raw = np.array([1.2, 0.2, 0.5, 0.3, 0.8, 0.4, 0.9, 0.6, 1.0, 0.7, 1.1])
    u = np.asarray(QUANTILE_LEVELS, dtype=float).reshape(1, -1)
    got = ppf(u, _grid(raw), np.array([1]), np.array([1]), TABLES)
    np.testing.assert_allclose(got[0], np.sort(raw))


def test_floor_and_cap_apply_after_the_map():
    raw = np.array([-0.4, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 7.5])
    prepared = prepare_quantile_grid(raw)
    assert prepared[0, 0] == -0.4
    assert prepared[0, -1] == 7.5
    u = np.asarray(QUANTILE_LEVELS, dtype=float).reshape(1, -1)
    got = ppf(u, _grid(raw), np.array([1]), np.array([1]), TABLES)
    assert got[0, 0] == 0.0
    assert got[0, -1] == RATE_CAP
    assert np.max(got) <= RATE_CAP
    assert np.min(got) >= 0.0


def test_zero_probability_matches_the_lower_tail_table():
    """Three exact zeros out of five table entries, and q05 is strictly positive."""
    share = float(np.mean(ZERO_TABLE == 0.0))
    assert share == pytest.approx(3 / 5)
    grids = _grid(INTERIOR)
    groups = np.array([0])
    mass = cdf(np.array([[0.0]]), grids, groups, groups, TABLES) - cdf_left(
        np.array([[0.0]]), grids, groups, groups, TABLES
    )
    assert mass[0, 0] == pytest.approx(0.05 * share)
    # The atom is the table, not a tied knot: every stored knot is positive.
    assert prepare_quantile_grid(INTERIOR)[0, 0] > 0
    assert ppf(np.array([[0.0]]), grids, groups, groups, TABLES)[0, 0] == 0.0
    assert ppf(np.array([[0.05 * share]]), grids, groups, groups, TABLES)[0, 0] == 0.0
    assert ppf(np.array([[0.05 * share + 1e-4]]), grids, groups, groups, TABLES)[0, 0] > 0


def test_one_table_zero_is_returned_without_a_tied_knot():
    tables = _tables([0.0, 0.5, 1.0], [0.0, 0.4], [0.4, 1.0], [0.0, 0.5])
    grids = _grid(INTERIOR)
    groups = np.array([0])
    assert ppf(np.array([[0.0]]), grids, groups, groups, tables)[0, 0] == 0.0
    share = 1 / 3
    mass = cdf(np.array([[0.0]]), grids, groups, groups, tables) - cdf_left(
        np.array([[0.0]]), grids, groups, groups, tables
    )
    assert mass[0, 0] == pytest.approx(0.05 * share)


def test_tail_probabilities_are_five_percent_when_bounds_do_not_bind():
    grids = _grid(INTERIOR)
    groups = np.array([1])
    q05 = INTERIOR[0]
    q95 = INTERIOR[-1]
    assert 0 < q05 < q95 < RATE_CAP
    # Group 1's lower table starts at 0.4, so the floor does not bind.
    # Its upper excess tops out at 0.5, so q95 + excess stays under 6.
    below = cdf_left(np.array([[q05]]), grids, groups, groups, TABLES)
    above = 1.0 - cdf(np.array([[q95]]), grids, groups, groups, TABLES)
    assert below[0, 0] == pytest.approx(0.05)
    assert above[0, 0] == pytest.approx(0.05)


def test_cap_atom_keeps_the_upper_five_percent_above_q95():
    grids = _grid(INTERIOR)
    groups = np.array([0])
    # Upper table [0, 10] pushes the raw tail past 6 while q95 stays below 6.
    tables = _tables(ZERO_TABLE, [0.0, 10.0], [0.4, 1.0], [0.0, 10.0])
    q95 = INTERIOR[-1]
    assert q95 < RATE_CAP
    above = 1.0 - cdf(np.array([[q95]]), grids, groups, groups, tables)
    assert above[0, 0] == pytest.approx(0.05)
    at_cap = cdf(np.array([[RATE_CAP]]), grids, groups, groups, tables) - cdf_left(
        np.array([[RATE_CAP]]), grids, groups, groups, tables
    )
    assert at_cap[0, 0] > 0
    assert ppf(np.array([[1.0]]), grids, groups, groups, tables)[0, 0] == RATE_CAP


def test_inverse_brackets_ppf_including_flats():
    tied = INTERIOR.copy()
    tied[3:7] = tied[4]
    q05_zero = INTERIOR.copy()
    q05_zero[0] = 0.0
    q05_zero[1] = 0.0
    rows = np.vstack([INTERIOR, tied, q05_zero])
    groups = np.array([1, 1, 0])
    rng = np.random.default_rng(4)
    u = np.vstack([
        np.linspace(0, 1, 64),
        rng.random(64),
        rng.random(64),
    ])
    y = ppf(u, rows, groups, groups, TABLES)
    left = cdf_left(y, rows, groups, groups, TABLES)
    right = cdf(y, rows, groups, groups, TABLES)
    assert np.all(left <= u + 1e-8)
    assert np.all(u <= right + 1e-8)
    assert np.all(y >= 0)
    assert np.all(y <= RATE_CAP)


def test_randomized_pits_of_draws_are_uniform():
    smooth = INTERIOR.copy()
    tied = INTERIOR.copy()
    tied[4:8] = tied[5]
    scoreless_knots = INTERIOR.copy()
    scoreless_knots[0] = 0.0
    scoreless_knots[1] = 0.0
    rows = np.vstack([smooth, tied, scoreless_knots, INTERIOR])
    # Row 3 reads the zero atom in group 0. The others use group 1,
    # except the scoreless knots, which are zero whichever table is scaled.
    groups = np.array([1, 1, 1, 0])
    rng = np.random.default_rng(26)
    draws = 8_000
    u = rng.random((4, draws))
    atom = rng.random((4, draws))
    y = ppf(u, rows, groups, groups, TABLES)
    pits = randomized_pit(y, atom, rows, groups, groups, TABLES)
    for index in range(rows.shape[0]):
        assert kstest(pits[index], "uniform").pvalue > 0.01
        assert np.all((pits[index] >= 0) & (pits[index] <= 1))


def _knot_frame(n, q05=0.4, q95=1.2):
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
    """100 rows, fold 1, pooled tail rate 5%. One scoreless game in group 0."""
    starter = _knot_frame(20)
    starter["rate"] = 0.8
    starter.loc[0, "rate"] = 0.1
    starter.loc[1, "rate"] = 1.8
    starter["starting"] = 1
    bench = _knot_frame(80)
    bench["rate"] = 0.8
    bench.loc[0, "rate"] = 0.0
    bench.loc[1:3, "rate"] = 0.1
    bench.loc[4:7, "rate"] = 1.6
    bench["starting"] = 0
    oof = pd.concat([starter, bench], ignore_index=True)
    oof["fold_id"] = 1
    oof["game_date"] = pd.Timestamp("2024-01-15")
    oof["is_holdout"] = False
    oof["game_id"] = [f"g{i}" for i in range(len(oof))]
    oof["player_id"] = np.arange(len(oof)).astype(str)
    ranges = {1: {"val_start": "2024-01-15", "val_end": "2024-01-15"}}
    return oof, ranges


def test_builder_records_exact_scoreless_zeros():
    oof, ranges = _valid_oof()
    tables = build_tail_tables(oof, folds=[1], fold_ranges=ranges)
    lower = tables.arrays[("lower", 0)]
    assert lower[0] == 0.0
    assert lower[-1] == 1.0
    assert np.sum(lower == 0.0) >= 1
    assert tables.cap == RATE_CAP
    grids = _grid(INTERIOR)
    share = float(np.mean(lower == 0.0))
    mass = cdf(np.array([[0.0]]), grids, np.array([0]), np.array([0]), tables)
    mass = mass - cdf_left(np.array([[0.0]]), grids, np.array([0]), np.array([0]), tables)
    assert mass[0, 0] == pytest.approx(0.05 * share)


def test_builder_rejects_holdout_rows():
    oof, ranges = _valid_oof()
    late = oof.copy()
    late["game_date"] = HOLDOUT_START
    with pytest.raises(ValueError, match="2025-10-21"):
        build_tail_tables(late, folds=[1], fold_ranges=ranges)
    marked = oof.copy()
    marked["is_holdout"] = True
    with pytest.raises(ValueError, match="holdout"):
        build_tail_tables(marked, folds=[1], fold_ranges=ranges)


def test_fit_drops_holdout_before_building():
    oof, _ranges = _valid_oof()
    oof["window_id"] = oof["fold_id"]
    hold = oof.iloc[[0]].copy()
    hold["game_date"] = pd.Timestamp("2025-11-01")
    hold["is_holdout"] = True
    hold["game_id"] = "hold"
    hold["rate"] = RATE_CAP
    for level in QUANTILE_LEVELS:
        hold[f"q_{level:.2f}"] = 0.2
    both = pd.concat([oof, hold], ignore_index=True)
    starting = both[["game_id", "player_id", "starting"]]
    # fit_preholdout_tails reads rate_q_* from an OOS-shaped frame. Adapt.
    renamed = both.rename(columns={f"q_{level:.2f}": f"rate_q_{level:.2f}" for level in QUANTILE_LEVELS})
    renamed["pts"] = renamed["rate"]
    renamed["minutes"] = 1.0
    tables = fit_preholdout_tails(renamed, starting, folds=[1])
    assert pd.to_datetime(tables.oof["game_date"]).max() < pd.Timestamp("2025-10-21")
    assert "hold" not in set(tables.oof["game_id"].astype(str))
    assert 5.8 not in set(np.round(tables.arrays[("upper", 1)], 5))


def test_loader_accepts_only_the_four_fold_cap(tmp_path):
    frames = []
    ranges = {}
    for fold in (1, 2, 3, 4):
        oof, _ = _valid_oof()
        oof["fold_id"] = fold
        oof["game_date"] = pd.Timestamp("2024-01-01") + pd.Timedelta(days=fold)
        frames.append(oof)
        ranges[fold] = {"val_start": "2024-01-01", "val_end": "2024-01-15"}
    tables = build_tail_tables(pd.concat(frames, ignore_index=True), folds=[1, 2, 3, 4], fold_ranges=ranges)
    path = save_tail_sidecar(tables, tmp_path / "pts_nba_tails_2026-04-12.joblib")
    loaded = load_tail_sidecar(path)
    assert loaded.folds == (1, 2, 3, 4)
    assert loaded.cap == RATE_CAP
    partial, one_range = _valid_oof()
    partial_tables = build_tail_tables(partial, folds=[1], fold_ranges=one_range)
    partial_path = save_tail_sidecar(partial_tables, tmp_path / "partial.joblib")
    with pytest.raises(ValueError, match="fold"):
        load_tail_sidecar(partial_path)
    payload = __import__("joblib").load(path)
    payload["cap"] = 5.0
    __import__("joblib").dump(payload, path)
    with pytest.raises(ValueError, match="cap"):
        load_tail_sidecar(path)


def test_rate_stream_is_labeled_separately_from_minutes():
    grids = _grid(INTERIOR)
    groups = np.array([1])
    kwargs = dict(grids=grids, lower_groups=groups, upper_groups=groups, tables=TABLES, seed=7, draws=32)
    labeled = sample_rate(player_ids=["a"], game_ids=["g1"], **kwargs)
    from hashlib import sha256

    from models.shared.minutes_sampler import canonical_id

    material = f"7|rate|{canonical_id('a')}|g1".encode()
    stream = int.from_bytes(sha256(material).digest()[:8], "little")
    uniforms = np.random.default_rng(stream).random(32).reshape(1, -1)
    np.testing.assert_allclose(labeled, ppf(uniforms, grids, groups, groups, TABLES))
    minutes_material = material.replace(b"|rate|", b"|minutes|")
    assert minutes_material != material


def test_saved_sidecar_is_the_preholdout_rate_fit():
    oos = pd.read_parquet(OOS_PATH)
    loaded = load_tail_sidecar(TAILS_PATH)
    pre = oos.loc[~oos["is_holdout"].astype(bool)].copy()
    hold = oos.loc[oos["is_holdout"].astype(bool)]
    saved_keys = set(zip(loaded.oof["game_id"].astype(str), loaded.oof["player_id"].astype(str)))
    pre_keys = set(zip(pre["game_id"].astype(str), pre["player_id"].astype(str)))
    hold_keys = set(zip(hold["game_id"].astype(str), hold["player_id"].astype(str)))
    assert saved_keys == pre_keys
    assert saved_keys.isdisjoint(hold_keys)
    assert pd.to_datetime(loaded.oof["game_date"]).max() < pd.Timestamp("2025-10-21")
    assert loaded.folds == (1, 2, 3, 4)
    for key, values in loaded.arrays.items():
        if key[0] == "lower":
            assert values[0] == 0.0
            assert values[-1] == 1.0
        else:
            assert values[0] == 0.0
    rebuilt = build_tail_tables(
        loaded.oof,
        folds=[1, 2, 3, 4],
        fold_ranges=loaded.fold_ranges,
    )
    for key in loaded.arrays:
        np.testing.assert_allclose(rebuilt.arrays[key], loaded.arrays[key])


def test_saved_sidecar_matches_a_fresh_preholdout_fit():
    oos = pd.read_parquet(OOS_PATH)
    parts = []
    silver_root = ROOT / "data" / "silver" / "nba"
    for path in sorted(silver_root.glob("*/regular_season/player_gamelogs.parquet")):
        parts.append(pd.read_parquet(path, columns=["game_id", "player_id", "start_position"]))
    starting = starting_from_gamelogs(pd.concat(parts, ignore_index=True))
    fresh = fit_preholdout_tails(oos, starting)
    loaded = load_tail_sidecar(TAILS_PATH)
    for key in fresh.arrays:
        np.testing.assert_allclose(loaded.arrays[key], fresh.arrays[key])
    y = capped_rate(fresh.oof["pts"], fresh.oof["minutes"])
    np.testing.assert_allclose(y, fresh.oof["rate"].to_numpy(dtype=float))
