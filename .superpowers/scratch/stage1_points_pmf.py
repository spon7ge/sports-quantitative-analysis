"""Stage 1 baseline for the independent minutes x PPM points PMF.

Scores the saved 2026-04-12 pipeline on the shared walk-forward windows.
Does not fit, does not read the 2025-26 holdout into a metric, and does
not write a saved bundle or an existing OOS parquet.

The working tree sets ``RATE_CAP`` to 2. The saved tails and the label
rule on ``pts_nba_model_2026-04-12.joblib`` use 6. This process restores
6 for the sampler globals so the scored PMF is that saved pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import models.shared.oos as oos
import models.shared.ppm_sampler as ppm_sampler
import models.shared.simulate as simulate

SAVED_CAP = 6.0
oos.RATE_CAP = SAVED_CAP
ppm_sampler.RATE_CAP = SAVED_CAP
simulate.RATE_CAP = SAVED_CAP

from models.shared.metrics import MINUTE_Q50_TIERS  # noqa: E402
from models.shared.minutes_sampler import (  # noqa: E402
    QUANTILE_LEVELS,
    cdf as minutes_cdf,
    groups_for_frame as minutes_groups,
    load_tail_sidecar as load_minutes_tails,
)
from models.shared.ppm_sampler import (  # noqa: E402
    cdf as rate_cdf,
    groups_for_frame as rate_groups,
    load_tail_sidecar as load_rate_tails,
)
from models.shared.simulate import points_pmf  # noqa: E402
from src.pipeline.silver.columns import assign_playing_minutes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".superpowers" / "scratch" / "stage1_points"
HOLDOUT_START = np.datetime64("2025-10-21")
TRAIN_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)
N_BOOT = 2_000
BOOT_SEED = 42
PIT_SEED = 42
PMF_SEEDS = (42, 43)
N_DRAWS = 10_000
KMAX = 90
LOG_FLOOR = 1e-6
OFFSETS = (-6, -3, 0, 3, 6)
ALT_LINES = (10, 20, 30, 40)
BATCH = 2_000


def _qcols(prefix: str) -> list[str]:
    return [f"{prefix}q_{level:.2f}" for level in QUANTILE_LEVELS]


def _naive_dates(values) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(np.asarray(values)))
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.to_numpy(dtype="datetime64[ns]")


def cluster_mean_ci(values, dates, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """95% percentile interval for a mean, resampling whole game dates."""
    sample = np.asarray(values, dtype=float).reshape(-1)
    when = _naive_dates(dates)
    if sample.shape[0] != when.shape[0]:
        raise ValueError("dates")
    finite = np.isfinite(sample)
    sample = sample[finite]
    when = when[finite]
    if sample.size == 0:
        raise ValueError("empty")
    codes, _uniques = pd.factorize(when, sort=False)
    n_clusters = int(codes.max()) + 1
    cluster_n = np.bincount(codes, minlength=n_clusters).astype(float)
    cluster_sum = np.bincount(codes, weights=sample, minlength=n_clusters)
    point = float(sample.mean())
    if n_clusters == 1 or n_boot <= 0:
        return _ci_row(point, point, point, sample.size, n_clusters)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = cluster_sum[draws].sum(axis=1) / cluster_n[draws].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return _ci_row(point, float(low), float(high), sample.size, n_clusters)


def _ci_row(point, low, high, n, n_dates) -> dict:
    return {
        "estimate": point,
        "ci_low": low,
        "ci_high": high,
        "n": int(n),
        "n_dates": int(n_dates),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import rankdata

    if left.size < 3:
        return float("nan")
    rank_l = rankdata(left, method="average")
    rank_r = rankdata(right, method="average")
    if np.std(rank_l) == 0 or np.std(rank_r) == 0:
        return float("nan")
    return float(np.corrcoef(rank_l, rank_r)[0, 1])


def cluster_spearman_ci(left, right, dates, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Spearman rho with a date-clustered percentile interval."""
    u = np.asarray(left, dtype=float)
    v = np.asarray(right, dtype=float)
    when = _naive_dates(dates)
    finite = np.isfinite(u) & np.isfinite(v)
    u, v, when = u[finite], v[finite], when[finite]
    point = _spearman(u, v)
    codes, _uniques = pd.factorize(when, sort=False)
    n_clusters = int(codes.max()) + 1
    groups = [np.flatnonzero(codes == code) for code in range(n_clusters)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[int(code)] for code in chosen])
        boots[draw] = _spearman(u[idx], v[idx])
        if draw == 0 or (draw + 1) % 500 == 0:
            print(f"    spearman bootstrap {draw + 1}/{n_boot}", flush=True)
    low, high = np.nanquantile(boots, [0.025, 0.975])
    return _ci_row(point, float(low), float(high), u.size, n_clusters)


def walk_forward_blocks(unique_dates: np.ndarray, *, n_folds: int = 5):
    """Same date blocks as ``date_walk_forward_folds`` plus the train tail."""
    n_dates = len(unique_dates)
    train_window = round(n_dates * 0.50)
    step = round(n_dates * 0.10)
    blocks = []
    for fold in range(n_folds):
        start = fold * step
        train_end = start + train_window
        val_end = train_end + step
        if val_end > n_dates:
            break
        train_dates = unique_dates[start:train_end]
        val_dates = unique_dates[train_end:val_end]
        n_stop = max(1, int(np.floor(len(train_dates) * 0.10)))
        blocks.append(
            {
                "fold": fold + 1,
                "train": set(train_dates.tolist()),
                "early_stop": set(train_dates[-n_stop:].tolist()),
                "val": set(val_dates.tolist()),
                "train_end": pd.Timestamp(train_dates[-1]).date().isoformat(),
                "val_start": pd.Timestamp(val_dates[0]).date().isoformat(),
                "val_end": pd.Timestamp(val_dates[-1]).date().isoformat(),
            }
        )
    return blocks


def appearance_dates(root: Path) -> np.ndarray:
    """Unique game dates with a positive parsed minute total, train seasons."""
    frames = []
    for season in TRAIN_SEASONS:
        path = (
            root
            / "data"
            / "silver"
            / "nba"
            / season
            / "regular_season"
            / "player_gamelogs.parquet"
        )
        names = set(pq.read_schema(path).names)
        columns = [
            name
            for name in ("game_date", "min", "minutes", "min_sec")
            if name in names
        ]
        frame = pd.read_parquet(path, columns=columns)
        frame = assign_playing_minutes(frame)
        frames.append(frame.loc[frame["minutes"] > 0, ["game_date"]])
    dates = pd.to_datetime(pd.concat(frames, ignore_index=True)["game_date"])
    unique = np.sort(dates.dt.tz_localize(None).unique())
    return unique.astype("datetime64[ns]")


def load_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame.loc[~frame["is_holdout"].astype(bool)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["player_id"] = frame["player_id"].astype(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame.reset_index(drop=True)


def join_oof(minutes: pd.DataFrame, rate: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = ["game_id", "player_id"]
    if minutes.duplicated(keys).any() or rate.duplicated(keys).any():
        raise AssertionError("duplicate game_id, player_id inside a model OOF")
    merged = minutes.merge(rate, on=keys, how="outer", indicator=True, suffixes=("_m", "_r"))
    counts = merged["_merge"].value_counts().to_dict()
    both = merged.loc[merged["_merge"] == "both"].copy()
    if (both["window_id_m"] != both["window_id_r"]).any():
        raise AssertionError("a player-game is in two different walk-forward windows")
    if not pd.to_datetime(both["game_date_m"]).equals(pd.to_datetime(both["game_date_r"])):
        raise AssertionError("game_date disagrees")
    if not np.allclose(both["minutes_m"], both["minutes_r"]):
        raise AssertionError("minutes disagree")
    if not np.allclose(both["pts_m"], both["pts_r"]):
        raise AssertionError("points disagree")
    if not both["starting_m"].astype(int).equals(both["starting_r"].astype(int)):
        raise AssertionError("starting disagrees")
    if both.duplicated(keys).any():
        raise AssertionError("duplicate game_id, player_id in the joint OOF")
    holdout = _naive_dates(both["game_date_m"]) >= HOLDOUT_START
    if holdout.any():
        raise AssertionError("holdout date in the joint OOF")

    out = pd.DataFrame(
        {
            "game_id": both["game_id"].to_numpy(),
            "player_id": both["player_id"].to_numpy(),
            "game_date": pd.to_datetime(both["game_date_m"]).to_numpy(),
            "window_id": both["window_id_m"].astype(int).to_numpy(),
            "starting": both["starting_m"].astype(int).to_numpy(),
            "minutes": both["minutes_m"].to_numpy(dtype=float),
            "pts": both["pts_m"].to_numpy(dtype=float),
        }
    )
    for level in QUANTILE_LEVELS:
        out[f"minutes_q_{level:.2f}"] = both[f"q_{level:.2f}_m"].to_numpy(dtype=float)
        out[f"rate_q_{level:.2f}"] = both[f"q_{level:.2f}_r"].to_numpy(dtype=float)
    out["ppm"] = out["pts"] / out["minutes"]
    out = out.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    info = {
        "minutes_oof": int(len(minutes)),
        "rate_oof": int(len(rate)),
        "joint": int(len(out)),
        "minutes_only": int(counts.get("left_only", 0)),
        "rate_only": int(counts.get("right_only", 0)),
    }
    return out, info


def assert_fold_dates(frame: pd.DataFrame, blocks: list[dict]) -> None:
    """No OOF date is in that fold's training dates or early-stopping dates."""
    by_fold = {block["fold"]: block for block in blocks}
    if set(frame["window_id"].unique()) != set(by_fold):
        raise AssertionError(
            f"windows {sorted(frame['window_id'].unique())} != {[b['fold'] for b in blocks]}"
        )
    for fold, block in by_fold.items():
        val = set(_naive_dates(frame.loc[frame["window_id"] == fold, "game_date"]).tolist())
        if val != block["val"]:
            missing = len(block["val"] - val)
            extra = len(val - block["val"])
            raise AssertionError(
                f"fold {fold} validation dates differ (missing {missing}, extra {extra})"
            )
        leaked = val & (block["train"] | block["early_stop"])
        if leaked:
            raise AssertionError(f"fold {fold} leaks {len(leaked)} dates")
        if not block["early_stop"] <= block["train"]:
            raise AssertionError(f"fold {fold} early-stopping dates leave the train window")


def tier_labels(predicted_q50: np.ndarray) -> np.ndarray:
    labels = np.empty(predicted_q50.shape[0], dtype=object)
    seen = np.zeros(predicted_q50.shape[0], dtype=bool)
    for name, low, high in MINUTE_Q50_TIERS:
        mask = np.ones(predicted_q50.shape[0], dtype=bool)
        if low is not None:
            mask &= predicted_q50 >= low
        if high is not None:
            mask &= predicted_q50 < high
        labels[mask] = name
        seen |= mask
    if not seen.all():
        raise AssertionError("predicted minutes q50 fell outside the tier cuts")
    return labels


def slice_masks(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    masks = [("all", np.ones(len(frame), dtype=bool))]
    for role, value in (("bench", 0), ("starters", 1)):
        masks.append((role, frame["starting"].to_numpy() == value))
    tiers = frame["tier"].to_numpy()
    for name, _low, _high in MINUTE_Q50_TIERS:
        masks.append((name, tiers == name))
    return masks


def survival_at(pmf: np.ndarray, row_index: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """P(points >= k) for ``(row_index, k)`` pairs into ``pmf``."""
    width = pmf.shape[1]
    cdf = np.cumsum(pmf, axis=1)
    index = np.clip(ks.astype(int) - 1, 0, width - 1)
    probability = 1.0 - cdf[row_index.astype(int), index]
    probability = np.where(ks <= 0, 1.0, probability)
    probability = np.where(ks >= width, 0.0, probability)
    return probability


def score_pmf_batch(pmf: np.ndarray, y: np.ndarray, atom: np.ndarray) -> dict[str, np.ndarray]:
    """RPS, negative log score, randomized PIT, mean, median, P(points = 0)."""
    if not np.allclose(pmf.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("PMF rows do not sum to 1")
    width = pmf.shape[1]
    y_int = np.rint(y).astype(int)
    if np.any(np.abs(y - y_int) > 1e-6):
        raise AssertionError("points are not integers")
    clipped = np.clip(y_int, 0, width - 1)
    ks = np.arange(width)
    cdf = np.cumsum(pmf, axis=1)
    indicator = (y_int[:, None] <= ks[None, :]).astype(float)
    # Mass clipped into the last bin still counts as that integer for the
    # indicator only when the outcome is inside the support.
    rps = np.sum((cdf - indicator) ** 2, axis=1)
    outside = (y_int < 0) | (y_int >= width)
    probability = pmf[np.arange(len(y_int)), clipped]
    probability = np.where(outside, 0.0, probability)
    log_score = -np.log(np.maximum(probability, LOG_FLOOR))
    cdf_left = np.zeros(len(y_int), dtype=float)
    inside = (y_int >= 1) & (y_int < width)
    at = y_int[inside] - 1
    cdf_left[inside] = cdf[np.flatnonzero(inside), at]
    pit = np.where(y_int >= width, 1.0, cdf_left + atom * probability)
    pit = np.where(y_int < 0, 0.0, pit)
    mean = pmf @ ks.astype(float)
    median = np.argmax(cdf >= 0.5 - 1e-12, axis=1).astype(int)
    return {
        "rps": rps,
        "log_score": log_score,
        "pit": pit,
        "mean": mean,
        "median": median.astype(float),
        "p0": pmf[:, 0],
        "y0": (y_int == 0).astype(float),
    }


def evaluation_lines(median: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer lines from the rounded PMF median and the fixed alt ladder.

    A negative threshold is dropped. Duplicate lines on one row are kept
    once. Returns ``(row_index, k)``.
    """
    rows = []
    ks = []
    for index, center in enumerate(np.rint(median).astype(int)):
        chosen = []
        for offset in OFFSETS:
            line = int(center + offset)
            if line >= 0:
                chosen.append(line)
        chosen.extend(int(line) for line in ALT_LINES)
        for line in dict.fromkeys(chosen):
            rows.append(index)
            ks.append(line)
    return np.asarray(rows, dtype=int), np.asarray(ks, dtype=int)


def decile_reliability(predicted, observed, dates) -> list[dict]:
    """Predicted vs observed in deciles of predicted P(points >= k)."""
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    when = np.asarray(dates)
    try:
        bins = pd.qcut(pred, 10, labels=False, duplicates="drop")
    except ValueError:
        return []
    bins = np.asarray(bins)
    rows = []
    for bin_id in range(int(np.nanmax(bins)) + 1):
        mask = bins == bin_id
        if not mask.any():
            continue
        pred_ci = cluster_mean_ci(pred[mask], when[mask])
        obs_ci = cluster_mean_ci(obs[mask], when[mask])
        rows.append(
            {
                "decile": int(bin_id + 1),
                "n": int(mask.sum()),
                "predicted": pred_ci["estimate"],
                "predicted_ci": [pred_ci["ci_low"], pred_ci["ci_high"]],
                "observed": obs_ci["estimate"],
                "observed_ci": [obs_ci["ci_low"], obs_ci["ci_high"]],
                "gap": obs_ci["estimate"] - pred_ci["estimate"],
            }
        )
    return rows


def coverage_table(frame: pd.DataFrame) -> list[dict]:
    """PPM 80% interval coverage by starting x realized-minutes octile."""
    low = frame["rate_q_0.10"].to_numpy(dtype=float)
    high = frame["rate_q_0.90"].to_numpy(dtype=float)
    inside = (frame["ppm"].to_numpy() >= low) & (frame["ppm"].to_numpy() <= high)
    crossed = low > high
    octile = pd.qcut(frame["minutes"], 8, labels=False, duplicates="drop")
    edges = pd.qcut(frame["minutes"], 8, duplicates="drop")
    categories = edges.cat.categories
    rows = []
    for role, value in (("all", None), ("bench", 0), ("starters", 1)):
        role_mask = np.ones(len(frame), dtype=bool) if value is None else (
            frame["starting"].to_numpy() == value
        )
        for bin_id, interval in enumerate(categories):
            mask = role_mask & (np.asarray(octile) == bin_id)
            if not mask.any():
                continue
            ci = cluster_mean_ci(inside[mask].astype(float), frame.loc[mask, "game_date"])
            rows.append(
                {
                    "starting": role,
                    "octile": int(bin_id + 1),
                    "minutes": f"{interval.left:.1f}–{interval.right:.1f}",
                    "coverage": ci["estimate"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n": ci["n"],
                    "n_dates": ci["n_dates"],
                    "crossed": int(crossed[mask].sum()),
                }
            )
    return rows


def metric_table(frame: pd.DataFrame, columns: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, mask in slice_masks(frame):
        if int(mask.sum()) == 0:
            continue
        dates = frame.loc[mask, "game_date"]
        record = {"slice": name, "n": int(mask.sum())}
        for key, values in columns.items():
            record[key] = cluster_mean_ci(values[mask], dates)
        rows.append(record)
    return rows


def historical_line_inventory(root: Path) -> dict:
    """What player-points lines exist, and whether they cover the OOF window."""
    notes = []
    quotes = (
        root
        / "notebooks"
        / "nba"
        / "line_movement_analysis"
        / "reports"
        / "2026-10-24"
        / "quotes_normalized.parquet"
    )
    if quotes.exists():
        frame = pd.read_parquet(quotes, columns=["prop", "start_time", "line"])
        points = frame.loc[frame["prop"].astype(str).eq("points")]
        start = pd.to_datetime(frame["start_time"], utc=True)
        notes.append(
            {
                "path": str(quotes.relative_to(root)),
                "rows": int(len(frame)),
                "points_rows": int(len(points)),
                "start_min": str(start.min()),
                "start_max": str(start.max()),
                "covers_oof": bool(start.min() < pd.Timestamp("2025-10-21", tz="UTC")),
            }
        )
    odds = root / "data" / "odds"
    if odds.exists():
        for path in sorted(odds.glob("*.csv")):
            sample = pd.read_csv(path, nrows=5)
            notes.append(
                {
                    "path": str(path.relative_to(root)),
                    "columns": list(sample.columns),
                    "covers_oof": False,
                }
            )
    team = root / "data" / "backtest" / "teamOdds.csv"
    if team.exists():
        notes.append(
            {
                "path": str(team.relative_to(root)),
                "kind": "team odds, not player points",
                "covers_oof": False,
            }
        )
    return {"files": notes, "usable_for_oof_lines": False}


def bundle_card(path: Path) -> dict:
    import joblib

    bundle = joblib.load(path)
    meta = bundle.get("metadata") or {}
    model = bundle["quantile_models"]["q_0.50"]
    params = model.get_params()
    keep = (
        "objective",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_child_weight",
        "early_stopping_rounds",
        "random_state",
    )
    folds = []
    for row in bundle["fold_metrics"]:
        folds.append(
            {
                "fold": row["fold"],
                "train_start": str(pd.Timestamp(row["train_start"]).date()),
                "train_end": str(pd.Timestamp(row["train_end"]).date()),
                "val_end": str(pd.Timestamp(row["val_end"]).date()),
            }
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "features": list(bundle["feature_names"]),
        "label": meta.get("label"),
        "label_rule": meta.get("label_rule"),
        "holdout_season": meta.get("holdout_season"),
        "train_range": meta.get("train_range"),
        "holdout_range": meta.get("holdout_range"),
        "params": {key: params.get(key) for key in keep},
        "seed": meta.get("seed"),
        "n_folds_recorded": len(folds),
        "folds": folds,
        "sample_weight": None,
    }


def predict_pits(frame: pd.DataFrame, minutes_tables, rate_tables) -> tuple[np.ndarray, np.ndarray]:
    minute_grid = frame[_qcols("minutes_")].to_numpy(dtype=float)
    rate_grid = frame[_qcols("rate_")].to_numpy(dtype=float)
    minute_g = minutes_groups(frame["starting"], minutes_tables)
    rate_g = rate_groups(frame["starting"], rate_tables)
    u_m = np.empty(len(frame), dtype=float)
    u_p = np.empty(len(frame), dtype=float)
    started = time.perf_counter()
    for start in range(0, len(frame), BATCH):
        stop = min(start + BATCH, len(frame))
        sl = slice(start, stop)
        u_m[sl] = minutes_cdf(
            frame["minutes"].to_numpy()[sl],
            minute_grid[sl],
            minute_g[0][sl],
            minute_g[1][sl],
            minutes_tables,
        )
        u_p[sl] = rate_cdf(
            frame["ppm"].to_numpy()[sl],
            rate_grid[sl],
            rate_g[0][sl],
            rate_g[1][sl],
            rate_tables,
        )
        print(f"  PIT {stop:,}/{len(frame):,}  {time.perf_counter() - started:.0f}s", flush=True)
    return u_m, u_p


def run_pmfs(frame: pd.DataFrame, minutes_tables, rate_tables):
    """Seed-42 scores plus P(points >= k) under seeds 42 and 43."""
    atom = np.random.default_rng(PIT_SEED).random(len(frame))
    scores = {key: np.empty(len(frame), dtype=float) for key in (
        "rps", "log_score", "pit", "mean", "median", "p0", "y0"
    )}
    p42_parts = []
    p43_parts = []
    row_parts = []
    k_parts = []
    started = time.perf_counter()
    for start in range(0, len(frame), BATCH):
        stop = min(start + BATCH, len(frame))
        batch = frame.iloc[start:stop]
        y = batch["pts"].to_numpy(dtype=float)
        pmf42, _mean, _median = points_pmf(
            batch,
            n_draws=N_DRAWS,
            kmax=KMAX,
            minutes_tables=minutes_tables,
            rate_tables=rate_tables,
            seed=PMF_SEEDS[0],
        )
        scored = score_pmf_batch(pmf42, y, atom[start:stop])
        for key, values in scored.items():
            scores[key][start:stop] = values
        rows, ks = evaluation_lines(scored["median"])
        p42 = survival_at(pmf42, rows, ks)
        pmf43, _mean43, _median43 = points_pmf(
            batch,
            n_draws=N_DRAWS,
            kmax=KMAX,
            minutes_tables=minutes_tables,
            rate_tables=rate_tables,
            seed=PMF_SEEDS[1],
        )
        p43 = survival_at(pmf43, rows, ks)
        p42_parts.append(p42)
        p43_parts.append(p43)
        row_parts.append(rows + start)
        k_parts.append(ks)
        print(
            f"  PMF {stop:,}/{len(frame):,}  {time.perf_counter() - started:.0f}s",
            flush=True,
        )
    return scores, (
        np.concatenate(row_parts),
        np.concatenate(k_parts),
        np.concatenate(p42_parts),
        np.concatenate(p43_parts),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading OOF", flush=True)
    minutes = load_oof(ROOT / "data/oos/nba/minutes_predictions.parquet")
    rate = load_oof(ROOT / "data/oos/nba/rate_predictions.parquet")
    frame, join_info = join_oof(minutes, rate)
    print(join_info, flush=True)

    print("reconstructing date blocks", flush=True)
    unique_dates = appearance_dates(ROOT)
    blocks = walk_forward_blocks(unique_dates)
    assert_fold_dates(frame, blocks)
    if not (minutes["early_stop"] == "train_tail").all():
        raise AssertionError("minutes OOF early_stop is not train_tail")
    if not np.allclose(minutes["train_tail_frac"].astype(float), 0.10):
        raise AssertionError("minutes OOF train_tail_frac is not 0.10")

    date_sets = {}
    for window, group in frame.groupby("window_id"):
        date_sets[int(window)] = set(_naive_dates(group["game_date"]).tolist())
    for window, group in rate.groupby("window_id"):
        rate_dates = set(_naive_dates(group["game_date"]).tolist())
        if rate_dates != date_sets[int(window)]:
            raise AssertionError(f"rate window {window} dates differ from the joint")

    frame["tier"] = tier_labels(frame["minutes_q_0.50"].to_numpy(dtype=float))
    crossed = frame["rate_q_0.10"].to_numpy() > frame["rate_q_0.90"].to_numpy()
    print(f"rate q10 > q90: {int(crossed.sum())}", flush=True)

    print("loading tails", flush=True)
    minutes_tables = load_minutes_tails(
        ROOT / "models/saved_models/min_nba_tails_2026-04-12.joblib"
    )
    rate_tables = load_rate_tails(
        ROOT / "models/saved_models/pts_nba_tails_2026-04-12.joblib"
    )

    print("PIT", flush=True)
    u_m, u_p = predict_pits(frame, minutes_tables, rate_tables)
    frame["u_m"] = u_m
    frame["u_p"] = u_p

    print("dependence", flush=True)
    dependence = {
        "all": cluster_spearman_ci(u_m, u_p, frame["game_date"]),
    }
    for role, value in (("bench", 0), ("starters", 1)):
        mask = frame["starting"].to_numpy() == value
        dependence[role] = cluster_spearman_ci(
            u_m[mask], u_p[mask], frame.loc[mask, "game_date"]
        )
        print(role, dependence[role], flush=True)
    print("all", dependence["all"], flush=True)

    print("coverage", flush=True)
    coverage = coverage_table(frame)
    for row in coverage:
        if row["starting"] == "all":
            print(
                f"  octile {row['octile']} {row['minutes']}: "
                f"{row['coverage']:.3f} ({row['ci_low']:.3f}, {row['ci_high']:.3f}) n={row['n']}",
                flush=True,
            )
    (OUT / "partial.json").write_text(
        json.dumps(
            {"join": join_info, "dependence": dependence, "coverage": coverage},
            indent=2,
            default=_json_default,
        )
    )

    print("PMF", flush=True)
    scores, lines = run_pmfs(frame, minutes_tables, rate_tables)
    row_index, ks, p42, p43 = lines
    gap = np.abs(p42 - p43)
    noise = {
        "seeds": list(PMF_SEEDS),
        "n_draws": N_DRAWS,
        "n_lines": int(gap.size),
        "mean_abs": float(gap.mean()),
        "max_abs": float(gap.max()),
        "p95_abs": float(np.quantile(gap, 0.95)),
    }
    print("noise", noise, flush=True)

    y = frame["pts"].to_numpy(dtype=float)
    bias_mean = scores["mean"] - y
    bias_median = scores["median"] - y
    zero_gap = scores["p0"] - scores["y0"]
    metrics = metric_table(
        frame,
        {
            "rps": scores["rps"],
            "log_score": scores["log_score"],
            "bias_mean": bias_mean,
            "bias_median": bias_median,
            "p0_predicted": scores["p0"],
            "p0_realized": scores["y0"],
            "p0_gap": zero_gap,
        },
    )

    line_dates = frame["game_date"].to_numpy()[row_index]
    observed = (frame["pts"].to_numpy()[row_index] >= ks).astype(float)
    reliability = decile_reliability(p42, observed, line_dates)
    pit = scores["pit"]
    hist, edges = np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))
    pit_hist = [
        {
            "bin": f"{edges[i]:.1f}–{edges[i + 1]:.1f}",
            "count": int(hist[i]),
            "share": float(hist[i] / hist.sum()),
        }
        for i in range(len(hist))
    ]

    centers = scores["median"]
    key = pd.MultiIndex.from_arrays([row_index, ks])
    stored = pd.Series(p42, index=key)
    def _at(line_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(len(frame))
        want = pd.MultiIndex.from_arrays([idx, line_values.astype(int)])
        found = stored.reindex(want)
        mask = found.notna().to_numpy()
        return (
            found.to_numpy()[mask],
            (frame["pts"].to_numpy()[mask] >= line_values[mask]).astype(float),
            frame.loc[mask, "game_date"],
        )

    by_line = []
    for offset in OFFSETS:
        line_values = np.rint(centers).astype(int) + int(offset)
        eligible = line_values >= 0
        pred, obs, dates = _at(np.where(eligible, line_values, -1))
        # _at on -1 misses the index, which drops ineligible rows.
        by_line.append(
            {
                "line": f"median{offset:+d}",
                "predicted": cluster_mean_ci(pred, dates),
                "observed": cluster_mean_ci(obs, dates),
            }
        )
    for alt in ALT_LINES:
        pred, obs, dates = _at(np.full(len(frame), alt, dtype=int))
        by_line.append(
            {
                "line": str(alt),
                "predicted": cluster_mean_ci(pred, dates),
                "observed": cluster_mean_ci(obs, dates),
            }
        )

    results = {
        "join": join_info,
        "folds": [
            {
                "fold": block["fold"],
                "train_end": block["train_end"],
                "val_start": block["val_start"],
                "val_end": block["val_end"],
                "n_train_dates": len(block["train"]),
                "n_early_stop_dates": len(block["early_stop"]),
                "n_val_dates": len(block["val"]),
            }
            for block in blocks
        ],
        "rate_q10_above_q90": int(crossed.sum()),
        "dependence": dependence,
        "coverage": coverage,
        "noise": noise,
        "metrics": metrics,
        "reliability": reliability,
        "by_line": by_line,
        "pit_histogram": pit_hist,
        "lines": historical_line_inventory(ROOT),
        "ppm_bundle": bundle_card(ROOT / "models/saved_models/pts_nba_model_2026-04-12.joblib"),
        "cap": SAVED_CAP,
        "rounding": "floor(minutes * rate + 0.5)",
        "n_draws": N_DRAWS,
        "kmax": KMAX,
    }
    destination = OUT / "results.json"
    destination.write_text(json.dumps(results, indent=2, default=_json_default))
    frame[
        ["game_id", "player_id", "game_date", "window_id", "starting", "minutes", "pts", "ppm", "tier", "u_m", "u_p"]
    ].to_parquet(OUT / "joint_oof_keys.parquet", index=False)
    print(f"wrote {destination}", flush=True)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
