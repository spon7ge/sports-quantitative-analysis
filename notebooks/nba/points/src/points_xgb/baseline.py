"""Holdout comparison against last-game, season-to-date, and EWMA rates."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from points_xgb.config import RATE_TIERS, N_BOOT, SEED, TARGET_COL, XGB_NAME
from points_xgb.paths import ensure_repo_on_path
from points_xgb.scoring import WIS_INTERVALS, interval_stats, wis_values

ensure_repo_on_path()
from models.shared.metrics import pinball_loss  # noqa: E402


def point_row(name: str, y, pred) -> dict[str, float | str]:
    error = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    absolute = np.abs(error)
    return {
        "model": name,
        "mae": float(absolute.mean()),
        "median_ae": float(np.median(absolute)),
        "mean_bias": float(error.mean()),
    }


def clustered_mae_bootstrap(
    y,
    predictors: Mapping[str, np.ndarray],
    dates,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample game dates, then recompute MAE inside each draw."""
    codes, _ = pd.factorize(dates, sort=False)
    n_clusters = int(codes.max()) + 1
    cluster_n = np.bincount(codes).astype(float)
    actual = np.asarray(y, dtype=float)
    cluster_sum = {
        name: np.bincount(codes, weights=np.abs(actual - np.asarray(pred, dtype=float)))
        for name, pred in predictors.items()
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    weights = np.zeros((n_boot, n_clusters), dtype=np.int32)
    np.add.at(
        weights,
        (np.repeat(np.arange(n_boot), n_clusters), draws.ravel()),
        1,
    )
    denom = weights @ cluster_n
    return {name: (weights @ total) / denom for name, total in cluster_sum.items()}


def _distribution_row(name: str, y, qmap: Mapping[str, np.ndarray]) -> dict:
    band_80 = interval_stats(y, qmap, 0.10, 0.90)
    band_90 = interval_stats(y, qmap, 0.05, 0.95)
    keys = sorted(qmap, key=lambda key: float(str(key).split("_", 1)[1]))
    grid = np.column_stack([qmap[key] for key in keys])
    if grid.shape[1] < 2:
        crossed = 0.0
    else:
        crossed = float(np.mean(np.any(np.diff(grid, axis=1) < 0, axis=1)))
    return {
        "model": name,
        "cov_80": band_80["cov"],
        "width_80_mean": band_80["width_mean"],
        "width_80_median": band_80["width_median"],
        "cov_90": band_90["cov"],
        "width_90_mean": band_90["width_mean"],
        "width_90_median": band_90["width_median"],
        "wis": float(np.mean(wis_values(y, qmap))),
        "quantile_crossing": crossed,
    }


def _subgroup_rows(
    group: str,
    mask,
    y_all,
    qmap: Mapping[str, np.ndarray],
) -> list[dict]:
    base = np.asarray(mask, dtype=bool) & np.isfinite(y_all)
    use = base & np.isfinite(qmap["q_0.50"])
    n = int(use.sum())
    if n == 0:
        return []
    actual = y_all[use]
    part = {key: values[use] for key, values in qmap.items()}
    error = actual - part["q_0.50"]
    band_80 = interval_stats(actual, part, 0.10, 0.90)
    band_90 = interval_stats(actual, part, 0.05, 0.95)
    return [
        {
            "model": XGB_NAME,
            "group": group,
            "n": n,
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "wis": float(np.mean(wis_values(actual, part))),
            "cov_80": band_80["cov"],
            "width_80_mean": band_80["width_mean"],
            "width_80_median": band_80["width_median"],
            "cov_90": band_90["cov"],
            "width_90_mean": band_90["width_mean"],
            "width_90_median": band_90["width_median"],
        }
    ]


def compare_to_baselines(
    y_ho,
    preds_ho: Mapping[str, np.ndarray],
    ppm_holdout: pd.DataFrame,
    ppm_df: pd.DataFrame,
    wf_results: list[dict],
    *,
    target_col: str = TARGET_COL,
    tiers: Mapping[str, Callable] = RATE_TIERS,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Point, distribution, calibration, subgroup, and tail tables.

    Baselines are last-game rate, season-to-date mean, and a 10-game
    EWMA. The MAE interval resamples game dates. Rows missing any
    baseline are dropped from the paired point comparison only.
    """
    y_all = np.asarray(y_ho, dtype=float)
    qmap = {key: np.asarray(pred, dtype=float) for key, pred in preds_ho.items()}
    if len(next(iter(qmap.values()))) != len(y_all):
        raise ValueError(f"{XGB_NAME} predictions do not match the holdout length")
    baselines = {
        "Last-game rate": ppm_holdout["pts_per_min_lag_1"].to_numpy(dtype=float),
        "Season-to-date rate": ppm_holdout["season_pts_per_min_mean"].to_numpy(dtype=float),
        "EWMA rate": ppm_holdout["pts_per_min_ewm_hl_10"].to_numpy(dtype=float),
    }
    paired = np.isfinite(y_all) & np.isfinite(qmap["q_0.50"])
    for pred in baselines.values():
        paired &= np.isfinite(pred)

    y_paired = y_all[paired]
    xgb_q50_name = f"{XGB_NAME} q50"
    paired_preds = {xgb_q50_name: qmap["q_0.50"][paired]}
    paired_preds.update({name: pred[paired] for name, pred in baselines.items()})
    dates_paired = pd.to_datetime(ppm_holdout["game_date"]).to_numpy()[paired]
    boot_mae = clustered_mae_bootstrap(
        y_paired, paired_preds, dates_paired, n_boot, seed
    )

    point_order = [
        "Last-game rate",
        "Season-to-date rate",
        "EWMA rate",
        xgb_q50_name,
    ]
    point_table = pd.DataFrame(
        [point_row(name, y_paired, paired_preds[name]) for name in point_order]
    )
    best_name = min(
        baselines,
        key=lambda name: point_table.loc[point_table["model"] == name, "mae"].iloc[0],
    )
    model_boot = boot_mae[xgb_q50_name]
    model_mae = float(
        point_table.loc[point_table["model"] == xgb_q50_name, "mae"].iloc[0]
    )
    delta_table = pd.DataFrame(
        [
            {
                "comparison": f"XGBoost q50 − {name}",
                "delta_mae": model_mae
                - float(point_table.loc[point_table["model"] == name, "mae"].iloc[0]),
                "ci_low": float(np.quantile(model_boot - boot_mae[name], 0.025)),
                "ci_high": float(np.quantile(model_boot - boot_mae[name], 0.975)),
            }
            for name in point_order[:-1]
        ]
    )

    fold_rows = []
    for fold in wf_results:
        part = ppm_df.loc[fold["val_mask"]]
        y_fold = part[target_col].to_numpy(dtype=float)
        fold_row: dict[str, float | int | str] = {
            "fold": fold["fold"],
            "n": int(len(part)),
            "model_mae": 2.0 * float(fold["pinball"]),
        }
        for name, column in (
            ("last_game", "pts_per_min_lag_1"),
            ("season_average", "season_pts_per_min_mean"),
            ("ewma", "pts_per_min_ewm_hl_10"),
        ):
            pred = part[column].to_numpy(dtype=float)
            ok = np.isfinite(y_fold) & np.isfinite(pred)
            fold_row[name] = float(np.mean(np.abs(y_fold[ok] - pred[ok])))
        fold_row["best_baseline"] = min(
            fold_row["last_game"], fold_row["season_average"], fold_row["ewma"]
        )
        fold_row["model_minus_best"] = (
            float(fold_row["model_mae"]) - float(fold_row["best_baseline"])
        )
        fold_rows.append(fold_row)
    fold_table = pd.DataFrame(fold_rows)

    distribution = pd.DataFrame([_distribution_row(XGB_NAME, y_all, qmap)])
    alpha_of = {key: float(str(key).split("_", 1)[1]) for key in qmap}
    calibration = pd.DataFrame(
        [
            {
                "model": XGB_NAME,
                "quantile": f"q{alpha_of[key] * 100:02.0f}",
                "expected": alpha_of[key],
                "observed": float(np.mean(y_all <= qmap[key])),
                "difference": float(np.mean(y_all <= qmap[key]) - alpha_of[key]),
            }
            for key in sorted(qmap, key=lambda key: float(str(key).split("_", 1)[1]))
        ]
    )

    starting = ppm_holdout["starting"].to_numpy()
    months = pd.to_datetime(ppm_holdout["game_date"]).dt.to_period("M").astype(str)
    xgb_q50 = qmap["q_0.50"]
    subgroup_rows: list[dict] = []
    subgroup_rows.extend(
        _subgroup_rows("Confirmed starter", starting == 1, y_all, qmap)
    )
    subgroup_rows.extend(_subgroup_rows("Bench", starting == 0, y_all, qmap))
    for name, rule in tiers.items():
        subgroup_rows.extend(
            _subgroup_rows(f"Predicted q50 {name}", rule(xgb_q50), y_all, qmap)
        )
    for month in sorted(months.unique()):
        subgroup_rows.extend(
            _subgroup_rows(str(month), months.to_numpy() == month, y_all, qmap)
        )

    absolute_error = np.abs(y_all - qmap["q_0.50"])
    tail_table = pd.DataFrame(
        [
            {
                "model": XGB_NAME,
                "threshold": f">{cutoff:g} pts/min",
                "rate": float(np.mean(absolute_error > cutoff)),
                "n": int(np.sum(absolute_error > cutoff)),
            }
            for cutoff in (0.15, 0.25, 0.40)
        ]
    )

    xgb_dist = distribution.iloc[0]
    best_mae = float(point_table.loc[point_table["model"] == best_name, "mae"].iloc[0])
    best_delta = delta_table.loc[
        delta_table["comparison"] == f"XGBoost q50 − {best_name}"
    ].iloc[0]
    xgb_cal = calibration
    q50_rate = float(xgb_cal.loc[xgb_cal["quantile"] == "q50", "observed"].iloc[0])
    max_gap = float(xgb_cal["difference"].abs().max())
    q50_pinball = pinball_loss(y_paired, paired_preds[xgb_q50_name], 0.50)
    wis_knots = [
        f"{low}–{high}"
        for _alpha, low, high in WIS_INTERVALS
        if low in qmap and high in qmap
    ]
    checks = [
        f"{XGB_NAME} q50 observed rate {q50_rate:.1%}  (target about 48–52%)",
        f"{XGB_NAME} largest |quantile gap| {max_gap:.1%}",
        (
            f"XGBoost q50 vs {best_name}: delta MAE {best_delta['delta_mae']:+.3f} "
            f"[{best_delta['ci_low']:+.3f}, {best_delta['ci_high']:+.3f}]"
        ),
        f"WIS {xgb_dist['wis']:.3f} (XGBoost) vs best baseline CRPS/MAE {best_mae:.3f}",
        (
            "walk-forward folds with XGBoost MAE below the best baseline: "
            f"{int((fold_table['model_minus_best'] < 0).sum())} / {len(fold_table)}"
        ),
        f"raw quantile crossing {xgb_dist['quantile_crossing']:.2%}",
    ]
    return {
        "paired_n": int(paired.sum()),
        "dropped_n": int((~paired).sum()),
        "n_boot": n_boot,
        "q50_pinball": q50_pinball,
        "point_table": point_table,
        "delta_table": delta_table,
        "fold_table": fold_table,
        "wis_knots": wis_knots,
        "distribution": distribution,
        "best_name": best_name,
        "best_mae": best_mae,
        "calibration": calibration,
        "subgroup_table": pd.DataFrame(subgroup_rows),
        "tail_table": tail_table,
        "checks": checks,
    }
