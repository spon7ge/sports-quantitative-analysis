"""Inverse-transform points-per-minute quantile function. Draws are not a price.

The body interpolates the 11 sorted knots on [0.05, 0.95]. Outside that
band the map uses out-of-fold tail tables. Scoreless ratios are an atom
at 0 whose probability is their share of the lower-tail table. The map
is clipped to [0, 6] afterwards, so the cap is an atom at 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.shared.minutes_sampler import (
    HOLDOUT_START,
    QUANTILE_LEVELS,
    _cdf_from_knots,
    _cdf_left_from_knots,
    _clip_breakpoints,
    _lookup,
    _reshape_outcomes,
    _row_breakpoints,
    canonical_id,
    groups_for_frame as _groups_for_frame,
)
from models.shared.oos import RATE_CAP, capped_rate

RATE_FLOOR = 0.0
_DEFAULT_GROUPING = {"lower": "starting", "upper": "starting"}


@dataclass
class RateTailTables:
    arrays: dict[tuple[str, int], np.ndarray]
    floor: float = RATE_FLOOR
    cap: float = RATE_CAP
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS
    holdout_start: str = "2025-10-21"
    grouping: dict[str, str] | None = None
    groups: tuple[int, ...] = (0, 1)
    folds: tuple[int, ...] = ()
    fold_ranges: dict | None = None
    oof: object | None = None
    source: str = "preholdout_oos"


def prepare_quantile_grid(grids: np.ndarray) -> np.ndarray:
    """Sort along tau. Knots are not floored; the cap is applied after the map."""
    values = np.asarray(grids, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError(f"knot count must be {len(QUANTILE_LEVELS)}")
    if not np.isfinite(values).all():
        raise ValueError("non-finite knot")
    return np.sort(values, axis=1)


def _zero_atom_knots(q05: float, table: np.ndarray, zeros: int):
    """Lower-tail knots whose atom at 0 has mass ``zeros / len(table)``."""
    n = len(table)
    atom = zeros / n
    u_atom = atom * 0.05
    positive = table[zeros:]
    if len(positive) == 0:
        return (
            np.array([0.0, np.nextafter(0.05, 0.0)], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )
    m = len(positive)
    local = np.linspace(0.0, 1.0, m)
    u_positive = (atom + local * (1.0 - atom)) * 0.05
    q_positive = q05 * positive
    u_knots = np.concatenate(
        [[0.0, u_atom, float(np.nextafter(u_atom, 1.0))], u_positive[1 : m - 1]]
    )
    q_knots = np.concatenate([[0.0, 0.0, float(q_positive[0])], q_positive[1 : m - 1]])
    return u_knots.astype(float), q_knots.astype(float)


def _rate_knots(prepared_row, lower_group, upper_group, tables):
    u, q = _row_breakpoints(prepared_row, int(lower_group), int(upper_group), tables)
    q05 = float(prepared_row[0])
    table = np.sort(np.asarray(_lookup(tables, "lower", int(lower_group)), dtype=float))
    zeros = int(np.sum(table == 0.0))
    if zeros > 0 and q05 > 0.0:
        u_low, q_low = _zero_atom_knots(q05, table, zeros)
        cut = int(np.searchsorted(u, 0.05, side="left"))
        u = np.concatenate([u_low, u[cut:]])
        q = np.concatenate([q_low, q[cut:]])
    return _clip_breakpoints(u, q, RATE_FLOOR, RATE_CAP)


def _validate_uniforms(u, n_rows: int) -> np.ndarray:
    uniforms = np.asarray(u, dtype=float)
    if uniforms.ndim != 2:
        raise ValueError("u must have shape (rows, draws)")
    if not np.isfinite(uniforms).all() or np.any(uniforms < 0) or np.any(uniforms > 1):
        raise ValueError("u outside [0, 1]")
    if uniforms.shape[0] != n_rows:
        raise ValueError("grids and u must have the same number of rows")
    return uniforms


def ppf(u, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """Quantile function on [0, 6]. ``u`` is caller-supplied. No RNG."""
    prepared = prepare_quantile_grid(grids)
    uniforms = _validate_uniforms(u, prepared.shape[0])
    lower_ids = np.asarray(lower_groups)
    upper_ids = np.asarray(upper_groups)
    out = np.empty(uniforms.shape, dtype=float)
    for index in range(prepared.shape[0]):
        knots_u, knots_q = _rate_knots(
            prepared[index],
            int(lower_ids[index]),
            int(upper_ids[index]),
            tables,
        )
        out[index] = np.interp(uniforms[index], knots_u, knots_q)
    return out


def quantile_rate(u, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """Alias of ``ppf``."""
    return ppf(u, grids, lower_groups, upper_groups, tables)


def _apply_inverse(y, grids, lower_groups, upper_groups, tables, inverse):
    prepared = prepare_quantile_grid(grids)
    values, squeeze = _reshape_outcomes(y, prepared.shape[0])
    lower_ids = np.asarray(lower_groups)
    upper_ids = np.asarray(upper_groups)
    out = np.empty(values.shape, dtype=float)
    for index in range(prepared.shape[0]):
        knots_u, knots_q = _rate_knots(
            prepared[index],
            int(lower_ids[index]),
            int(upper_ids[index]),
            tables,
        )
        out[index] = inverse(knots_u, knots_q, values[index])
    if squeeze:
        return out[:, 0]
    return out


def cdf_left(y, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """P(rate < y) for the clipped quantile function. No RNG."""
    return _apply_inverse(y, grids, lower_groups, upper_groups, tables, _cdf_left_from_knots)


def cdf(y, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """P(rate <= y) for the clipped quantile function. No RNG."""
    return _apply_inverse(y, grids, lower_groups, upper_groups, tables, _cdf_from_knots)


def randomized_pit(y, atom_u, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """PIT on a flat piece. ``atom_u`` is caller-supplied. No RNG."""
    left = cdf_left(y, grids, lower_groups, upper_groups, tables)
    right = cdf(y, grids, lower_groups, upper_groups, tables)
    draw = np.asarray(atom_u, dtype=float)
    if draw.shape != left.shape:
        raise ValueError("u")
    if not np.isfinite(draw).all() or np.any(draw < 0) or np.any(draw > 1):
        raise ValueError("u outside [0, 1]")
    return left + draw * (right - left)


def probability_below_line(grids, lower_groups, upper_groups, tables, line) -> np.ndarray:
    """Price P(rate < line) by inverting each row's clipped Q. No draws."""
    n_rows = prepare_quantile_grid(grids).shape[0]
    return cdf_left(np.full(n_rows, float(line)), grids, lower_groups, upper_groups, tables)


def build_tail_tables(oof, *, folds, fold_ranges, grouping=None) -> RateTailTables:
    """Build pinned lower/upper tables from pre-holdout out-of-fold rates.

    Scoreless games (``rate == 0``) are kept. They enter the lower table
    as exact zeros. Holdout rows are rejected.
    """
    rules = dict(_DEFAULT_GROUPING) if grouping is None else dict(grouping)
    if rules != _DEFAULT_GROUPING:
        raise ValueError("unknown grouping")
    if "is_holdout" in oof.columns and oof["is_holdout"].astype(bool).any():
        raise ValueError("holdout")

    present = set(oof["fold_id"].tolist())
    missing = [fold_id for fold_id in folds if fold_id not in present]
    if missing:
        raise ValueError(f"fold missing: {missing}")

    starting = oof["starting"].to_numpy()
    if not np.issubdtype(np.asarray(starting).dtype, np.number):
        raise ValueError("starting")
    if not np.isin(starting, [0, 1]).all():
        raise ValueError("starting")

    rate = oof["rate"].to_numpy(dtype=float)
    if not np.isfinite(rate).all() or np.any(rate < 0) or np.any(rate > RATE_CAP):
        raise ValueError("rate")

    game_dates = np.asarray(pd.to_datetime(oof["game_date"]), dtype="datetime64[ns]")
    if not np.all(game_dates < HOLDOUT_START):
        raise ValueError("2025-10-21")

    knot_cols = [f"q_{level:.2f}" for level in QUANTILE_LEVELS]
    prepared = prepare_quantile_grid(oof[knot_cols].to_numpy(dtype=float))

    selected_mask = oof["fold_id"].isin(list(folds)).to_numpy()
    selected = oof.loc[selected_mask].copy()
    prepared_sel = prepared[selected_mask]
    q05_sel = prepared_sel[:, 0]
    q95_sel = prepared_sel[:, -1]
    y_sel = selected["rate"].to_numpy(dtype=float)
    starting_sel = selected["starting"].to_numpy(dtype=int)

    buckets: dict[tuple[str, int], list[float]] = {
        ("lower", 0): [],
        ("lower", 1): [],
        ("upper", 0): [],
        ("upper", 1): [],
    }
    for index in range(len(selected)):
        group = int(starting_sel[index])
        y = float(y_sel[index])
        q05 = float(q05_sel[index])
        q95 = float(q95_sel[index])
        if y < q05:
            buckets[("lower", group)].append(0.0 if y == 0.0 else y / q05)
        if y > q95:
            buckets[("upper", group)].append(y - q95)

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

    return RateTailTables(
        arrays=arrays,
        floor=RATE_FLOOR,
        cap=RATE_CAP,
        quantile_levels=QUANTILE_LEVELS,
        holdout_start="2025-10-21",
        grouping=rules,
        groups=(0, 1),
        folds=tuple(folds),
        fold_ranges=fold_ranges,
        oof=selected,
        source="preholdout_oos",
    )


def save_tail_sidecar(tables: RateTailTables, path) -> Path:
    """Persist a RateTailTables payload. Does not require four folds."""
    out = Path(path)
    payload = {
        "arrays": tables.arrays,
        "floor": tables.floor,
        "cap": tables.cap,
        "quantile_levels": tables.quantile_levels,
        "holdout_start": tables.holdout_start,
        "grouping": tables.grouping,
        "groups": tables.groups,
        "folds": tables.folds,
        "fold_ranges": tables.fold_ranges,
        "oof": tables.oof,
        "source": tables.source,
    }
    joblib.dump(payload, out)
    return out


def load_tail_sidecar(path) -> RateTailTables:
    """Load a sidecar and reject anything other than the four-fold pre-holdout fit."""
    payload = joblib.load(path)
    if payload.get("cap") != RATE_CAP:
        raise ValueError("cap")
    if payload.get("floor") != RATE_FLOOR:
        raise ValueError("floor")
    if tuple(payload.get("quantile_levels", ())) != QUANTILE_LEVELS:
        raise ValueError("quantile")
    if tuple(payload.get("folds", ())) != (1, 2, 3, 4):
        raise ValueError("fold")
    grouping = payload.get("grouping") or {}
    if grouping.get("lower") != "starting" or grouping.get("upper") != "starting":
        raise ValueError("grouping")
    oof = payload.get("oof")
    if oof is not None and len(oof):
        dates = np.asarray(pd.to_datetime(oof["game_date"]), dtype="datetime64[ns]")
        if not np.all(dates < HOLDOUT_START):
            raise ValueError("2025-10-21")
        if "is_holdout" in getattr(oof, "columns", []) and oof["is_holdout"].astype(bool).any():
            raise ValueError("holdout")
    return RateTailTables(
        arrays=payload["arrays"],
        floor=payload["floor"],
        cap=payload["cap"],
        quantile_levels=tuple(payload["quantile_levels"]),
        holdout_start=payload.get("holdout_start", "2025-10-21"),
        grouping=dict(grouping),
        groups=tuple(payload.get("groups", (0, 1))),
        folds=tuple(payload["folds"]),
        fold_ranges=payload.get("fold_ranges"),
        oof=oof,
        source=payload.get("source", "preholdout_oos"),
    )


def groups_for_frame(starting, tables: RateTailTables):
    """Map draw-time starting flags to lower/upper group ids."""
    return _groups_for_frame(starting, tables)


def starting_from_gamelogs(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Starter flag from the box-score ``start_position``, one row per player-game."""
    text = gamelogs["start_position"].astype("string").str.strip()
    frame = pd.DataFrame(
        {
            "game_id": gamelogs["game_id"].astype(str),
            "player_id": gamelogs["player_id"].astype(str),
            "starting": text.isin(["G", "F", "C"]).astype(int),
        }
    )
    return frame.drop_duplicates(["game_id", "player_id"])


def _preholdout_oof(oos: pd.DataFrame, starting: pd.DataFrame) -> pd.DataFrame:
    if "is_holdout" not in oos.columns:
        raise ValueError("holdout")
    pre = oos.loc[~oos["is_holdout"].astype(bool)].copy()
    if pre.empty:
        raise ValueError("pre-holdout")
    pre = pre.drop(columns=["starting"], errors="ignore")
    flags = starting.copy()
    flags["game_id"] = flags["game_id"].astype(str)
    flags["player_id"] = flags["player_id"].astype(str)
    flags = flags.drop_duplicates(["game_id", "player_id"])
    pre["game_id"] = pre["game_id"].astype(str)
    pre["player_id"] = pre["player_id"].astype(str)
    merged = pre.merge(
        flags[["game_id", "player_id", "starting"]],
        on=["game_id", "player_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["starting"].isna().any():
        raise ValueError("starting")
    fold_col = "window_id" if "window_id" in merged.columns else "fold_id"
    frame = pd.DataFrame(
        {f"q_{level:.2f}": merged[f"rate_q_{level:.2f}"].to_numpy(dtype=float) for level in QUANTILE_LEVELS}
    )
    frame["rate"] = capped_rate(merged["pts"], merged["minutes"])
    frame["starting"] = merged["starting"].astype(int).to_numpy()
    frame["fold_id"] = merged[fold_col].astype(int).to_numpy()
    frame["game_date"] = pd.to_datetime(merged["game_date"]).to_numpy()
    frame["is_holdout"] = False
    frame["game_id"] = merged["game_id"].to_numpy()
    frame["player_id"] = merged["player_id"].to_numpy()
    frame["pts"] = merged["pts"].to_numpy()
    frame["minutes"] = merged["minutes"].to_numpy()
    return frame


def fit_preholdout_tails(oos, starting, *, folds=(1, 2, 3, 4)) -> RateTailTables:
    """Tail tables from pre-holdout OOS rate knots only. Holdout rows are dropped."""
    frame = _preholdout_oof(oos, starting)
    fold_list = [int(fold) for fold in folds]
    ranges = {}
    for fold in fold_list:
        dates = pd.to_datetime(frame.loc[frame["fold_id"] == fold, "game_date"])
        if dates.empty:
            ranges[fold] = {}
            continue
        ranges[fold] = {
            "val_start": str(dates.min().date()),
            "val_end": str(dates.max().date()),
        }
    return build_tail_tables(frame, folds=fold_list, fold_ranges=ranges)


def tails_path_for_bundle(bundle_path) -> Path:
    """``pts_nba_model_<date>.joblib`` -> ``pts_nba_tails_<date>.joblib``."""
    path = Path(bundle_path)
    if "_model_" not in path.name:
        raise ValueError("bundle")
    return path.with_name(path.name.replace("_model_", "_tails_"))


def sample_rate(
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
    """Draw uniforms on a labeled rate stream, then map through Q."""
    players = list(player_ids)
    games = list(game_ids)
    if len(players) != len(games):
        raise ValueError("id")
    uniforms = np.empty((len(players), int(draws)), dtype=float)
    for index, (player_id, game_id) in enumerate(zip(players, games)):
        material = (
            f"{seed}|rate|{canonical_id(player_id)}|{canonical_id(game_id)}"
        ).encode("utf-8")
        digest = sha256(material).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        uniforms[index] = rng.random(int(draws))
    return ppf(uniforms, grids, lower_groups, upper_groups, tables)
