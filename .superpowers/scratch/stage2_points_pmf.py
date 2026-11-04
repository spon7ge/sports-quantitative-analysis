"""Stage 2 conditional points PMF. Walk-forward OOF only.

Does not score 2025-26, does not fit on it, and does not write an
existing bundle or OOS parquet. Grid size and rounding are chosen here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".superpowers" / "scratch"))

import stage1_points_pmf as s1  # noqa: E402
from models.shared.minutes_sampler import (  # noqa: E402
    QUANTILE_LEVELS,
    groups_for_frame as minutes_groups,
    load_tail_sidecar as load_minutes_tails,
    ppf as minutes_ppf,
)
from models.shared.points_pmf import (  # noqa: E402
    build_ppm_tails,
    points_pmf,
    ppm_prediction_frame,
    ppm_quantile_function,
)
from models.shared.simulate import points_pmf as baseline_pmf  # noqa: E402

OUT = ROOT / ".superpowers" / "scratch" / "stage2"
HOLDOUT_START = np.datetime64("2025-10-21")
KMAX = 90
TOLERANCE = 0.001
GRID_ROWS = 300
BATCH = 250
LEVELS = list(QUANTILE_LEVELS)


def _q(level: float) -> str:
    return f"q_{level:.2f}"


def load_scored_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Conditional OOF joined to minutes knots. Comparison rows also have rate knots."""
    oof = pd.read_parquet(OUT / "oof.parquet")
    frame_path = OUT / "frame.parquet"
    features = joblib.load(OUT / "folds" / "fold_1.joblib")["features"]
    base = pd.read_parquet(frame_path, columns=["game_id", "player_id", *features])
    extra = [column for column in features if column not in oof.columns]
    oof = oof.merge(base.loc[:, ["game_id", "player_id", *extra]], on=["game_id", "player_id"], how="left")
    if len(oof) != len(pd.read_parquet(OUT / "oof.parquet")):
        raise AssertionError("feature join duplicated OOF rows")
    dates = s1._naive_dates(oof["game_date"])
    if np.any(dates >= HOLDOUT_START):
        raise AssertionError("holdout date in the conditional OOF")

    minutes = pd.read_parquet(
        ROOT / "data" / "oos" / "nba" / "minutes_predictions.parquet",
        columns=["game_id", "player_id", "is_holdout", *[_q(level) for level in LEVELS]],
    )
    minutes = minutes.loc[~minutes["is_holdout"].astype(bool)].drop(columns=["is_holdout"])
    minutes = minutes.rename(columns={_q(level): f"minutes_q_{level:.2f}" for level in LEVELS})
    oof = oof.merge(minutes, on=["game_id", "player_id"], how="left", validate="one_to_one")
    if oof["minutes_q_0.50"].isna().any():
        raise AssertionError("conditional OOF row missing minutes knots")

    rate = pd.read_parquet(
        ROOT / "data" / "oos" / "nba" / "rate_predictions.parquet",
        columns=["game_id", "player_id", "is_holdout", *[_q(level) for level in LEVELS]],
    )
    rate = rate.loc[~rate["is_holdout"].astype(bool)].drop(columns=["is_holdout"])
    rate = rate.rename(columns={_q(level): f"rate_q_{level:.2f}" for level in LEVELS})
    comparison = oof.merge(rate, on=["game_id", "player_id"], how="inner", validate="one_to_one")
    comparison = comparison.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    info = {
        "conditional_oof": int(len(oof)),
        "comparison": int(len(comparison)),
        "minutes_only": int(len(oof) - len(comparison)),
    }
    print(
        f"conditional OOF {info['conditional_oof']:,}  "
        f"shared with the baseline {info['comparison']:,}  "
        f"minutes-only {info['minutes_only']:,}",
        flush=True,
    )
    return oof, comparison, info


def fit_tails(oof: pd.DataFrame) -> tuple[object, dict]:
    """Tails on folds 1–4. A lower miss with q05 <= 0 cannot form a ratio."""
    chosen = oof.loc[oof["fold_id"].isin([1, 2, 3, 4])].copy()
    knots = np.sort(chosen.loc[:, [_q(level) for level in LEVELS]].to_numpy(dtype=float), axis=1)
    outcome = chosen["pts_per_min"].to_numpy(dtype=float)
    q05 = knots[:, 0]
    q95 = knots[:, -1]
    lower = outcome < q05
    upper = outcome > q95
    invalid = lower & (q05 <= 0)
    report = {"folds": [1, 2, 3, 4], "rows": int(len(chosen)), "invalid_lower": int(invalid.sum())}
    for role, value in (("bench", 0), ("starters", 1)):
        mask = chosen["starting"].to_numpy() == value
        report[role] = {
            "n": int(mask.sum()),
            "lower_miss": float(np.mean(lower[mask])) if mask.any() else None,
            "upper_miss": float(np.mean(upper[mask])) if mask.any() else None,
        }
        print(
            f"  {role} tail miss lower {report[role]['lower_miss']:.3%} "
            f"upper {report[role]['upper_miss']:.3%} n={report[role]['n']:,}",
            flush=True,
        )
    usable = chosen.loc[~invalid].copy()
    usable["ppm"] = usable["pts_per_min"]
    tails = build_ppm_tails(usable, folds=(1, 2, 3, 4))
    return tails, report


def _models() -> dict:
    loaded = {}
    for fold_id in range(1, 6):
        loaded[fold_id] = joblib.load(OUT / "folds" / f"fold_{fold_id}.joblib")
    return loaded


def _minutes_q(knots: np.ndarray, starting, tables, g_m: int) -> np.ndarray:
    uniforms = np.tile((np.arange(g_m) + 0.5) / g_m, (len(knots), 1))
    lower, upper = minutes_groups(starting, tables)
    return minutes_ppf(uniforms, knots, lower, upper, tables)


def _ppm_q(frame: pd.DataFrame, minutes_q: np.ndarray, models, tails, g_p: int) -> np.ndarray:
    n_rows, g_m = minutes_q.shape
    knots = np.empty((n_rows, g_m, len(LEVELS)), dtype=float)
    features = models[1]["features"]
    for fold_id, indexer in frame.groupby("fold_id", sort=False).groups.items():
        positions = np.asarray(list(indexer), dtype=int)
        payload = models[int(fold_id)]
        batched = ppm_prediction_frame(frame.iloc[positions], minutes_q[positions], features)
        predictors = batched.loc[:, features]
        for index, level in enumerate(LEVELS):
            predicted = payload["models"][_q(level)].predict(predictors)
            knots[positions, :, index] = np.asarray(predicted, dtype=float).reshape(len(positions), g_m)
    flat = knots.reshape(n_rows * g_m, len(LEVELS))
    uniforms = np.tile((np.arange(g_p) + 0.5) / g_p, (n_rows * g_m, 1))
    groups = np.repeat(frame["starting"].to_numpy(dtype=int), g_m)
    values = ppm_quantile_function(uniforms, flat, LEVELS, tails, groups)
    return values.reshape(n_rows, g_m, g_p)


def _both(frame, minute_knots, models, tails, minutes_tables, g_m: int, g_p: int):
    minutes_q = _minutes_q(minute_knots, frame["starting"], minutes_tables, g_m)
    ppm = _ppm_q(frame, minutes_q, models, tails, g_p)
    return {
        "rint": points_pmf(minutes_q, ppm, KMAX, "rint"),
        "floor": points_pmf(minutes_q, ppm, KMAX, "floor"),
    }


def _minute_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, [f"minutes_q_{level:.2f}" for level in LEVELS]].to_numpy(dtype=float)


def grid_search(comparison: pd.DataFrame, models, tails, minutes_tables) -> dict:
    rng = np.random.default_rng(42)
    picked = np.sort(rng.choice(len(comparison), size=GRID_ROWS, replace=False))
    sample = comparison.iloc[picked].reset_index(drop=True)
    knots = _minute_matrix(sample)
    g_m, g_p = 40, 200
    started = time.perf_counter()
    current = _both(sample, knots, models, tails, minutes_tables, g_m, g_p)
    elapsed = time.perf_counter() - started
    rows, ks = s1.evaluation_lines(s1.score_pmf_batch(current["rint"], sample["pts"].to_numpy(), np.zeros(len(sample)))["median"])
    history = []
    chosen = {"g_m": g_m, "g_p": g_p, "seconds_300": elapsed}
    while g_m < 320 and g_p < 1600:
        nxt_m, nxt_p = g_m * 2, g_p * 2
        started = time.perf_counter()
        refined = _both(sample, knots, models, tails, minutes_tables, nxt_m, nxt_p)
        refined_seconds = time.perf_counter() - started
        change = {}
        for rounding in ("rint", "floor"):
            before = s1.survival_at(current[rounding], rows, ks)
            after = s1.survival_at(refined[rounding], rows, ks)
            change[rounding] = float(np.max(np.abs(after - before)))
        step = {
            "from": [g_m, g_p],
            "to": [nxt_m, nxt_p],
            "max_abs": change,
            "seconds": refined_seconds,
        }
        history.append(step)
        print(
            f"grid {g_m}x{g_p} -> {nxt_m}x{nxt_p}  "
            f"max |dP| rint {change['rint']:.5f} floor {change['floor']:.5f}  "
            f"{refined_seconds:.1f}s",
            flush=True,
        )
        if change["rint"] < TOLERANCE and change["floor"] < TOLERANCE:
            chosen = {"g_m": g_m, "g_p": g_p, "seconds_300": elapsed}
            break
        current = refined
        g_m, g_p = nxt_m, nxt_p
        elapsed = refined_seconds
        chosen = {"g_m": g_m, "g_p": g_p, "seconds_300": elapsed}
    else:
        print("grid did not reach 0.1pp; keeping the finest computed size", flush=True)
    report = {"chosen": chosen, "steps": history, "n": GRID_ROWS, "tolerance": TOLERANCE}
    (OUT / "grid.json").write_text(json.dumps(report, indent=2))
    print(f"chosen grid G_M={chosen['g_m']} G_P={chosen['g_p']}  {chosen['seconds_300']:.1f}s for {GRID_ROWS}", flush=True)
    return report


def _accumulate(frame, models, tails, minutes_tables, g_m: int, g_p: int) -> dict[str, np.ndarray]:
    n_rows = len(frame)
    stored = {
        rounding: np.empty((n_rows, KMAX + 1), dtype=np.float32)
        for rounding in ("rint", "floor")
    }
    knots = _minute_matrix(frame)
    started = time.perf_counter()
    for start in range(0, n_rows, BATCH):
        stop = min(start + BATCH, n_rows)
        batch = frame.iloc[start:stop].reset_index(drop=True)
        both = _both(batch, knots[start:stop], models, tails, minutes_tables, g_m, g_p)
        for rounding, pmf in both.items():
            stored[rounding][start:stop] = pmf
        print(f"  conditional {stop:,}/{n_rows:,}  {time.perf_counter() - started:.0f}s", flush=True)
    return stored


def _line_table(pmf, centers, y, dates) -> list[dict]:
    rows = []
    y = np.asarray(y, dtype=float)
    center = np.rint(np.asarray(centers, dtype=float)).astype(int)
    specs = [(f"median{offset:+d}", center + offset) for offset in s1.OFFSETS]
    specs.extend((str(line), np.full(len(y), int(line))) for line in s1.ALT_LINES)
    for name, line in specs:
        keep = line >= 0
        if not np.any(keep):
            continue
        index = np.flatnonzero(keep)
        predicted = s1.survival_at(pmf, index, line[keep])
        observed = (y[keep] >= line[keep]).astype(float)
        pred_ci = s1.cluster_mean_ci(predicted, dates[keep])
        obs_ci = s1.cluster_mean_ci(observed, dates[keep])
        rows.append(
            {
                "line": name,
                "n": int(keep.sum()),
                "predicted": pred_ci["estimate"],
                "predicted_ci": [pred_ci["ci_low"], pred_ci["ci_high"]],
                "observed": obs_ci["estimate"],
                "observed_ci": [obs_ci["ci_low"], obs_ci["ci_high"]],
                "gap": obs_ci["estimate"] - pred_ci["estimate"],
            }
        )
    return rows


def _mae(table: list[dict]) -> float:
    return float(np.mean([abs(row["gap"]) for row in table]))


def _baseline(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    cached = OUT / "baseline_scores.npz"
    if cached.exists():
        loaded = np.load(cached)
        print("loaded cached baseline scores", flush=True)
        return {key: loaded[key] for key in loaded.files}
    minutes_tables = load_minutes_tails(ROOT / "models" / "saved_models" / "min_nba_tails_2026-04-12.joblib")
    rate_tables = s1.load_rate_tails(ROOT / "models" / "saved_models" / "pts_nba_tails_2026-04-12.joblib")
    atom = np.random.default_rng(s1.PIT_SEED).random(len(frame))
    scores = {key: np.empty(len(frame), dtype=float) for key in (
        "rps", "log_score", "pit", "mean", "median", "p0", "y0"
    )}
    rename = frame.rename(columns={f"minutes_q_{level:.2f}": f"minutes_q_{level:.2f}" for level in LEVELS})
    started = time.perf_counter()
    for start in range(0, len(frame), s1.BATCH):
        stop = min(start + s1.BATCH, len(frame))
        batch = rename.iloc[start:stop]
        pmf, _mean, _median = baseline_pmf(
            batch,
            n_draws=s1.N_DRAWS,
            kmax=KMAX,
            minutes_tables=minutes_tables,
            rate_tables=rate_tables,
            seed=42,
        )
        scored = s1.score_pmf_batch(pmf, batch["pts"].to_numpy(dtype=float), atom[start:stop])
        for key, values in scored.items():
            scores[key][start:stop] = values
        print(f"  baseline {stop:,}/{len(frame):,}  {time.perf_counter() - started:.0f}s", flush=True)
    np.savez(cached, **scores)
    return scores


def _score_block(pmf, y, atom) -> dict[str, np.ndarray]:
    return s1.score_pmf_batch(np.asarray(pmf, dtype=float), y, atom)


def _ci(values, dates) -> dict:
    row = s1.cluster_mean_ci(values, dates)
    return row


def run_score() -> None:
    oof, comparison, info = load_scored_frame()
    tails, tail_report = fit_tails(oof)
    minutes_tables = load_minutes_tails(ROOT / "models" / "saved_models" / "min_nba_tails_2026-04-12.joblib")
    models = _models()
    grid = grid_search(comparison, models, tails, minutes_tables)
    g_m = int(grid["chosen"]["g_m"])
    g_p = int(grid["chosen"]["g_p"])

    print("baseline", flush=True)
    base = _baseline(comparison)
    print("conditional full OOF", flush=True)
    stored = _accumulate(comparison, models, tails, minutes_tables, g_m, g_p)
    y = comparison["pts"].to_numpy(dtype=float)
    dates = comparison["game_date"]
    atom = np.random.default_rng(42).random(len(comparison))
    conditional = {
        rounding: _score_block(stored[rounding], y, atom) for rounding in ("rint", "floor")
    }
    tables = {
        rounding: _line_table(stored[rounding], base["median"], y, dates.to_numpy())
        for rounding in ("rint", "floor")
    }
    mae = {rounding: _mae(table) for rounding, table in tables.items()}
    chosen_rounding = min(mae, key=mae.get)
    print(f"rounding MAE rint {mae['rint']:.5f} floor {mae['floor']:.5f} -> {chosen_rounding}", flush=True)

    chosen = conditional[chosen_rounding]
    comparison = comparison.copy()
    comparison["tier"] = s1.tier_labels(comparison["minutes_q_0.50"].to_numpy(dtype=float))
    columns = {
        "rps": chosen["rps"],
        "log_score": chosen["log_score"],
        "mean_bias": chosen["mean"] - y,
        "median_bias": chosen["median"] - y,
        "p0": chosen["p0"],
        "y0": chosen["y0"],
    }
    base_columns = {
        "rps": base["rps"],
        "log_score": base["log_score"],
        "mean_bias": base["mean"] - y,
        "median_bias": base["median"] - y,
        "p0": base["p0"],
        "y0": base["y0"],
    }
    diff_columns = {
        key: columns[key] - base_columns[key]
        for key in ("rps", "log_score", "mean_bias", "median_bias", "p0")
    }
    counts, _edges = np.histogram(chosen["pit"], bins=np.linspace(0.0, 1.0, 11))
    summary = json.loads((OUT / "train_summary.json").read_text())
    predicted, observed, when = _reliability_inputs(
        stored[chosen_rounding], base["median"], y, dates.to_numpy()
    )
    payload = {
        "rows": info,
        "pinball_q50_by_fold": summary["pinball_q50_by_fold"],
        "holdout_pinball_q50_not_used": _holdout_pinball(),
        "tails": tail_report,
        "grid": grid,
        "rounding_mae": mae,
        "rounding": chosen_rounding,
        "conditional": s1.metric_table(comparison, columns),
        "baseline": s1.metric_table(comparison, base_columns),
        "difference": s1.metric_table(comparison, diff_columns),
        "lines": tables[chosen_rounding],
        "line_tables": tables,
        "reliability": s1.decile_reliability(predicted, observed, when),
        "pit_shares": (counts / counts.sum()).tolist(),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"rounding": chosen_rounding, "mae": mae, "grid": grid["chosen"]}, indent=2), flush=True)
    _print_slices(payload)


def _holdout_pinball() -> float | None:
    path = OUT / "holdout_models.joblib"
    if not path.exists():
        return None
    return float(joblib.load(path)["pinball_q50"])


def _pooled_lines(pmf, centers, y):
    rows, ks = s1.evaluation_lines(centers)
    return s1.survival_at(pmf, rows, ks), (np.asarray(y)[rows] >= ks).astype(float)


def _reliability_inputs(pmf, centers, y, dates):
    predicted, observed = _pooled_lines(pmf, centers, y)
    rows, _ks = s1.evaluation_lines(centers)
    return predicted, observed, np.asarray(dates)[rows]


def _print_slices(payload: dict) -> None:
    print(f"{'slice':<10} {'base RPS':>10} {'new RPS':>10} {'dRPS':>10} {'new log':>10}", flush=True)
    base = {row["slice"]: row for row in payload["baseline"]}
    diff = {row["slice"]: row for row in payload["difference"]}
    for row in payload["conditional"]:
        name = row["slice"]
        print(
            f"{name:<10} {base[name]['rps']['estimate']:10.3f} "
            f"{row['rps']['estimate']:10.3f} {diff[name]['rps']['estimate']:10.3f} "
            f"{row['log_score']['estimate']:10.3f}",
            flush=True,
        )


if __name__ == "__main__":
    run_score()
