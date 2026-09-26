"""Inverse-transform minutes quantile function. Draws are not a price."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.settlement import maximum_minutes

_INT_STRING = re.compile(r"^[+-]?\d+$")

QUANTILE_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
KNOT_FLOOR = 1e-3
HOLDOUT_START = np.datetime64("2025-10-21")
_DEFAULT_GROUPING = {"lower": "starting", "upper": "starting"}
_MISS_RATE_LO = 0.04
_MISS_RATE_HI = 0.06


@dataclass
class MinuteTailTables:
    arrays: dict[tuple[str, int], np.ndarray]
    floor: float = KNOT_FLOOR
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS
    early_stop: str = "train_tail"
    train_tail_frac: float = 0.10
    holdout_start: str = "2025-10-21"
    grouping: dict[str, str] | None = None
    groups: tuple[int, ...] = (0, 1)
    folds: tuple[int, ...] = ()
    fold_ranges: dict | None = None
    oof: object | None = None


def prepare_quantile_grid(grids: np.ndarray) -> np.ndarray:
    """Sort once, then floor every knot at 1e-3. No second sort."""
    values = np.asarray(grids, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError(f"knot count must be {len(QUANTILE_LEVELS)}")
    if not np.isfinite(values).all():
        raise ValueError("non-finite knot")
    ordered = np.sort(values, axis=1)
    return np.maximum(ordered, KNOT_FLOOR)


def empirical_quantile(v: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Linear empirical quantile. Probability 0 is table[0], probability 1 is table[-1]."""
    xp = np.linspace(0.0, 1.0, len(table))
    return np.interp(np.asarray(v, dtype=float), xp, np.asarray(table, dtype=float))


def _lookup(tables: MinuteTailTables, tail: str, group: int) -> np.ndarray:
    try:
        return tables.arrays[(tail, int(group))]
    except KeyError as exc:
        raise ValueError(f"unknown group {group} for {tail}") from exc


def _row_quantile(u, prepared, q05, q95, lower_group, upper_group, tables):
    out = np.empty(len(u), dtype=float)
    lower = u < 0.05
    upper = u > 0.95
    middle = ~lower & ~upper
    if np.any(lower):
        table = _lookup(tables, "lower", lower_group)
        out[lower] = q05 * empirical_quantile(u[lower] / 0.05, table)
    if np.any(upper):
        table = _lookup(tables, "upper", upper_group)
        out[upper] = q95 + empirical_quantile((u[upper] - 0.95) / 0.05, table)
    if np.any(middle):
        out[middle] = np.interp(u[middle], QUANTILE_LEVELS, prepared)
    return out


def quantile_minutes(u, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """Map uniforms of shape (rows, draws) through each row's Q."""
    uniforms = np.asarray(u, dtype=float)
    if uniforms.ndim != 2:
        raise ValueError("u must have shape (rows, draws)")
    if not np.isfinite(uniforms).all() or np.any(uniforms < 0) or np.any(uniforms > 1):
        raise ValueError("u outside [0, 1]")
    prepared = prepare_quantile_grid(grids)
    if prepared.shape[0] != uniforms.shape[0]:
        raise ValueError("grids and u must have the same number of rows")
    lower_ids = np.asarray(lower_groups)
    upper_ids = np.asarray(upper_groups)
    out = np.empty_like(uniforms, dtype=float)
    for index in range(prepared.shape[0]):
        out[index] = _row_quantile(
            uniforms[index],
            prepared[index],
            prepared[index, 0],
            prepared[index, -1],
            int(lower_ids[index]),
            int(upper_ids[index]),
            tables,
        )
    cap = maximum_minutes("nba")
    return np.clip(out, 0.0, cap)


def _row_breakpoints(prepared, lower_group, upper_group, tables):
    """Build (u, q) knots for one row's Q, including both tails."""
    q05 = float(prepared[0])
    q95 = float(prepared[-1])
    lower_table = _lookup(tables, "lower", lower_group)
    upper_table = _lookup(tables, "upper", upper_group)
    n_lower = len(lower_table)
    n_upper = len(upper_table)
    lower_grid = np.linspace(0.0, 1.0, n_lower)
    upper_grid = np.linspace(0.0, 1.0, n_upper)
    lower_u = lower_grid[:-1] * 0.05
    lower_q = q05 * empirical_quantile(lower_grid[:-1], lower_table)
    middle_u = np.asarray(QUANTILE_LEVELS, dtype=float)
    middle_q = np.asarray(prepared, dtype=float)
    upper_u = 0.95 + upper_grid[1:] * 0.05
    upper_q = q95 + empirical_quantile(upper_grid[1:], upper_table)
    u = np.concatenate([lower_u, middle_u, upper_u])
    q = np.concatenate([lower_q, middle_q, upper_q])
    return u, q


def _invert_q(u, q, line: float) -> float:
    """P(M < L) = inf{u : Q(u) >= L}."""
    if line <= q[0]:
        return 0.0
    if line > q[-1]:
        return 1.0
    index = int(np.searchsorted(q, line, side="left"))
    q0 = float(q[index - 1])
    q1 = float(q[index])
    u0 = float(u[index - 1])
    u1 = float(u[index])
    if q1 == q0:
        return u0
    return u0 + (line - q0) / (q1 - q0) * (u1 - u0)


def probability_below_line(grids, lower_groups, upper_groups, tables, line) -> np.ndarray:
    """Price P(M < line) by inverting each row's Q. No draws."""
    prepared = prepare_quantile_grid(grids)
    lower_ids = np.asarray(lower_groups)
    upper_ids = np.asarray(upper_groups)
    threshold = float(line)
    out = np.empty(prepared.shape[0], dtype=float)
    for index in range(prepared.shape[0]):
        u, q = _row_breakpoints(
            prepared[index],
            int(lower_ids[index]),
            int(upper_ids[index]),
            tables,
        )
        out[index] = _invert_q(u, q, threshold)
    return out


def row_crps(y, q_at_u, u) -> np.ndarray:
    """Per-row CRPS from pinball at levels u. Does not average across rows."""
    actual = np.asarray(y, dtype=float)
    knots = np.asarray(q_at_u, dtype=float)
    levels = np.asarray(u, dtype=float)
    if knots.ndim == 1:
        knots = knots.reshape(1, -1)
    residual = actual.reshape(-1, 1) - knots
    loss = np.where(residual >= 0, levels * residual, (levels - 1.0) * residual)
    return 2.0 * np.mean(loss, axis=1)


def tail_bin_shares(y, edge_values, *, tail) -> pd.DataFrame:
    """Fraction of rows and of tail misses falling in each edge bin."""
    actual = np.asarray(y, dtype=float)
    edges = np.asarray(edge_values, dtype=float)
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    n_rows, n_bins = edges.shape
    counts = np.zeros(n_bins, dtype=float)
    if tail == "lower":
        misses = actual < edges[:, -1]
        for bin_index in range(n_bins):
            if bin_index == 0:
                in_bin = actual < edges[:, 0]
            else:
                in_bin = (actual >= edges[:, bin_index - 1]) & (actual < edges[:, bin_index])
            counts[bin_index] = float(np.sum(in_bin))
    elif tail == "upper":
        misses = actual > edges[:, 0]
        for bin_index in range(n_bins):
            if bin_index == 0:
                in_bin = actual > edges[:, -1]
            else:
                lo = edges[:, n_bins - bin_index - 1]
                hi = edges[:, n_bins - bin_index]
                in_bin = (actual > lo) & (actual <= hi)
            counts[bin_index] = float(np.sum(in_bin))
    else:
        raise ValueError(f"unknown tail {tail}")
    miss_count = float(np.sum(misses))
    share_of_rows = counts / float(n_rows)
    if miss_count == 0:
        share_of_misses = np.full(n_bins, np.nan, dtype=float)
    else:
        share_of_misses = counts / miss_count
    return pd.DataFrame(
        {
            "bin": np.arange(1, n_bins + 1),
            "share_of_rows": share_of_rows,
            "share_of_misses": share_of_misses,
        }
    )


def build_tail_tables(oof, *, folds, fold_ranges, grouping=None) -> MinuteTailTables:
    """Build pinned lower/upper empirical tables from out-of-fold misses."""
    rules = dict(_DEFAULT_GROUPING) if grouping is None else dict(grouping)
    if rules != _DEFAULT_GROUPING:
        raise ValueError("unknown grouping")

    if not ((oof["early_stop"] == "train_tail").all() and (oof["train_tail_frac"] == 0.10).all()):
        raise ValueError("train_tail")

    present = set(oof["fold_id"].tolist())
    missing = [fold_id for fold_id in folds if fold_id not in present]
    if missing:
        raise ValueError(f"fold missing: {missing}")

    starting = oof["starting"].to_numpy()
    if not np.issubdtype(np.asarray(starting).dtype, np.number):
        raise ValueError("starting")
    if not np.isin(starting, [0, 1]).all():
        raise ValueError("starting")

    minutes = oof["minutes"].to_numpy(dtype=float)
    if not np.all(minutes > 0):
        raise ValueError("y > 0")

    game_dates = np.asarray(oof["game_date"], dtype="datetime64[ns]")
    if not np.all(game_dates < HOLDOUT_START):
        raise ValueError("2025-10-21")

    knot_cols = [f"q_{level:.2f}" for level in QUANTILE_LEVELS]
    prepared = prepare_quantile_grid(oof[knot_cols].to_numpy(dtype=float))

    selected_mask = oof["fold_id"].isin(list(folds)).to_numpy()
    selected = oof.loc[selected_mask].copy()
    prepared_sel = prepared[selected_mask]
    q05_sel = prepared_sel[:, 0]
    q95_sel = prepared_sel[:, -1]
    y_sel = selected["minutes"].to_numpy(dtype=float)
    starting_sel = selected["starting"].to_numpy(dtype=int)
    fold_sel = selected["fold_id"].to_numpy()

    for fold_id in folds:
        fold_mask = fold_sel == fold_id
        y_fold = y_sel[fold_mask]
        q05_fold = q05_sel[fold_mask]
        q95_fold = q95_sel[fold_mask]
        lower_rate = float(np.mean(y_fold < q05_fold))
        upper_rate = float(np.mean(y_fold > q95_fold))
        if not (_MISS_RATE_LO <= lower_rate <= _MISS_RATE_HI and _MISS_RATE_LO <= upper_rate <= _MISS_RATE_HI):
            raise ValueError(
                f"miss rate outside [0.04, 0.06] for fold {fold_id}: "
                f"lower={lower_rate:.4f} upper={upper_rate:.4f}"
            )

    buckets: dict[tuple[str, int], list[float]] = {
        ("lower", 0): [],
        ("lower", 1): [],
        ("upper", 0): [],
        ("upper", 1): [],
    }
    for index in range(len(selected)):
        group = int(starting_sel[index])
        y = y_sel[index]
        if y < q05_sel[index]:
            buckets[("lower", group)].append(y / q05_sel[index])
        if y > q95_sel[index]:
            buckets[("upper", group)].append(y - q95_sel[index])

    arrays: dict[tuple[str, int], np.ndarray] = {}
    for key, values in buckets.items():
        tail, _group = key
        if tail == "lower":
            values = [*values, 1.0]
        else:
            values = [*values, 0.0]
        if len(values) == 1:
            raise ValueError(f"pin-only table for {key}")
        arrays[key] = np.sort(np.asarray(values, dtype=float))

    return MinuteTailTables(
        arrays=arrays,
        floor=KNOT_FLOOR,
        quantile_levels=QUANTILE_LEVELS,
        early_stop="train_tail",
        train_tail_frac=0.10,
        holdout_start="2025-10-21",
        grouping=rules,
        groups=(0, 1),
        folds=tuple(folds),
        fold_ranges=fold_ranges,
        oof=selected,
    )


def save_tail_sidecar(tables: MinuteTailTables, path) -> Path:
    """Persist a MinuteTailTables payload. Does not require four folds."""
    out = Path(path)
    payload = {
        "arrays": tables.arrays,
        "floor": tables.floor,
        "quantile_levels": tables.quantile_levels,
        "early_stop": tables.early_stop,
        "train_tail_frac": tables.train_tail_frac,
        "holdout_start": tables.holdout_start,
        "grouping": tables.grouping,
        "groups": tables.groups,
        "folds": tables.folds,
        "fold_ranges": tables.fold_ranges,
        "oof": tables.oof,
    }
    joblib.dump(payload, out)
    return out


def load_tail_sidecar(path) -> MinuteTailTables:
    """Load a sidecar and reject anything that is not the pinned four-fold contract."""
    payload = joblib.load(path)
    if payload.get("floor") != KNOT_FLOOR:
        raise ValueError("floor")
    if tuple(payload.get("quantile_levels", ())) != QUANTILE_LEVELS:
        raise ValueError("quantile")
    if tuple(payload.get("folds", ())) != (1, 2, 3, 4):
        raise ValueError("fold")
    grouping = payload.get("grouping") or {}
    if grouping.get("lower") != "starting" or grouping.get("upper") != "starting":
        raise ValueError("grouping")
    return MinuteTailTables(
        arrays=payload["arrays"],
        floor=payload["floor"],
        quantile_levels=tuple(payload["quantile_levels"]),
        early_stop=payload.get("early_stop", "train_tail"),
        train_tail_frac=payload.get("train_tail_frac", 0.10),
        holdout_start=payload.get("holdout_start", "2025-10-21"),
        grouping=dict(grouping),
        groups=tuple(payload.get("groups", (0, 1))),
        folds=tuple(payload["folds"]),
        fold_ranges=payload.get("fold_ranges"),
        oof=payload.get("oof"),
    )


def groups_for_frame(starting, tables: MinuteTailTables) -> tuple[np.ndarray, np.ndarray]:
    """Map draw-time starting flags to lower/upper group ids."""
    rules = tables.grouping or {}
    if rules.get("lower") != "starting" or rules.get("upper") != "starting":
        raise ValueError("grouping")
    values = pd.to_numeric(starting, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("starting")
    groups = values.astype(int).to_numpy()
    return groups, groups.copy()


def canonical_id(value) -> str:
    """Normalize player/game ids for stable per-row RNG streams."""
    if value is None:
        raise ValueError("id")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError("id")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("id")
        if _INT_STRING.fullmatch(stripped):
            return str(int(stripped))
        return stripped
    if pd.isna(value):
        raise ValueError("id")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        raise ValueError("id")
    text = str(value).strip()
    if not text:
        raise ValueError("id")
    if _INT_STRING.fullmatch(text):
        return str(int(text))
    return text


def sample_minutes(
    grids,
    lower_groups,
    upper_groups,
    player_ids,
    game_ids,
    tables,
    *,
    seed=42,
    draws=10_000,
) -> np.ndarray:
    """Draw uniforms on a labeled minutes stream, then map through Q."""
    players = list(player_ids)
    games = list(game_ids)
    if len(players) != len(games):
        raise ValueError("id")
    uniforms = np.empty((len(players), int(draws)), dtype=float)
    for index, (player_id, game_id) in enumerate(zip(players, games)):
        material = (
            f"{seed}|minutes|{canonical_id(player_id)}|{canonical_id(game_id)}"
        ).encode("utf-8")
        digest = sha256(material).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        uniforms[index] = rng.random(int(draws))
    return quantile_minutes(uniforms, grids, lower_groups, upper_groups, tables)
