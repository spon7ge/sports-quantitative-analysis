"""Decompose joint points location bias. Holdout is diagnostic only.

Preholdout expanding folds fit coupling on earlier folds and score the
next fold. A clip-mean-matching challenger is scored on those same draws
without refitting β or role-aware variance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/alexgonzalez/Documents/nba_quant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.points import add_points_features
from src.models.evaluation import (
    pit_shape_penalty,
    probability_integral_transform,
    randomized_probability_integral_transform,
)
from src.models.xgboost_models.artifact_bundle import load_joint_points_bundle
from src.models.xgboost_models.joint_calibration import (
    DEFAULT_MINUTES_BINS,
    EPSILON_BINS,
    HOLDOUT_SEASON,
)
from src.models.xgboost_models.joint_simulation import JointPointsSimulator
from src.models.xgboost_models.joint_variant_eval import (
    EVAL_SEED,
    N_DRAWS,
    OOF_PANEL_ARTIFACT,
    clipping_uplift_by_band,
    decompose_location,
    fit_fold_variant,
    fold_fit_from_bundle,
    load_oof_panel,
    overlay_shift,
    score_samples,
    scored_folds,
    shift_raw_to_postclip_mean,
    signed_error_by_band,
    simulate_fold,
)

OUT_PATH = (
    ROOT / "artifacts" / "models" / "points" / "location_diagnostics.json"
)
MINUTES_MEAN = ROOT / "artifacts" / "models" / "minutes" / "xgboost_minutes.joblib"
MINUTES_DIST = (
    ROOT
    / "artifacts"
    / "models"
    / "minutes"
    / "xgboost_minutes_distribution.joblib"
)
POINTS_PATH = ROOT / "artifacts" / "models" / "points" / "xgboost_points.joblib"
JOINT_CAL = ROOT / "artifacts" / "models" / "points" / "joint_calibration.joblib"
SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]
MINUTES_ERROR_BINS = np.array([-np.inf, -8.0, -3.0, 3.0, 8.0, np.inf])
PIT_RNG = np.random.default_rng(EVAL_SEED + 13)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _table(frame: pd.DataFrame) -> list[dict]:
    return _clean(frame.to_dict(orient="records"))


def _pit_summary(samples: np.ndarray, actual: np.ndarray) -> dict:
    half = probability_integral_transform(samples, actual)
    randomized = randomized_probability_integral_transform(
        samples,
        actual,
        rng=PIT_RNG,
    )
    return {
        "half_credit_mean": float(np.mean(half)),
        "half_credit_std": float(np.std(half)),
        "half_credit_shape": pit_shape_penalty(half),
        "randomized_mean": float(np.mean(randomized)),
        "randomized_std": float(np.std(randomized)),
        "randomized_shape": pit_shape_penalty(randomized),
    }


def _band_label(value: float, edges: np.ndarray) -> str:
    bounds = np.asarray(edges, dtype=float)
    bin_id = int(
        np.clip(np.digitize(value, bounds) - 1, 0, len(bounds) - 2)
    )
    lo = bounds[bin_id]
    hi = bounds[bin_id + 1]
    lo_label = "-∞" if not np.isfinite(lo) else f"{lo:g}"
    hi_label = "∞" if not np.isfinite(hi) else f"{hi:g}"
    return f"[{lo_label}, {hi_label})"


def _slice_location(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for name, part in frame.groupby(column, dropna=False, sort=True):
        parts = decompose_location(
            part["actual"].to_numpy(dtype=float),
            part["hat_p"].to_numpy(dtype=float),
            part["g"].to_numpy(dtype=float),
            part["mean_pre"].to_numpy(dtype=float),
            part["mean_post"].to_numpy(dtype=float),
        )
        parts["slice"] = str(name)
        parts["pit_half_mean"] = float(part["pit_half"].mean())
        parts["pit_rand_mean"] = float(part["pit_rand"].mean())
        rows.append(parts)
    return pd.DataFrame(rows)


def _row_frame(
    valid: pd.DataFrame,
    fit,
    clipped: np.ndarray,
    raw: np.ndarray,
) -> pd.DataFrame:
    actual = pd.to_numeric(valid["pts"], errors="coerce").to_numpy(
        dtype=float
    )
    hat_p = pd.to_numeric(valid["points_hat"], errors="coerce").to_numpy(
        dtype=float
    )
    hat_m = pd.to_numeric(valid["minutes_hat"], errors="coerce").to_numpy(
        dtype=float
    )
    g = overlay_shift(valid, fit)
    mean_pre = raw.mean(axis=1)
    mean_post = clipped.mean(axis=1)
    dates = pd.to_datetime(valid["game_date"])
    minutes_error = (
        pd.to_numeric(valid["minutes"], errors="coerce").to_numpy(dtype=float)
        - hat_m
    )
    start_rate = pd.to_numeric(
        valid["start_rate_10"], errors="coerce"
    ).to_numpy(dtype=float)
    half = probability_integral_transform(clipped, actual)
    randomized = randomized_probability_integral_transform(
        clipped,
        actual,
        rng=PIT_RNG,
    )
    frame = pd.DataFrame(
        {
            "actual": actual,
            "hat_p": hat_p,
            "hat_m": hat_m,
            "g": g,
            "hat_p_plus_g": hat_p + g,
            "mean_pre": mean_pre,
            "mean_post": mean_post,
            "clipping_uplift": mean_post - mean_pre,
            "error_post": actual - mean_post,
            "minutes_error": minutes_error,
            "start_rate_10": start_rate,
            "role": np.where(start_rate >= 0.5, "starter", "bench"),
            "zero_points": np.where(actual <= 0.0, "zero", "positive"),
            "season_year": valid["season_year"].astype("string").to_numpy(),
            "month": dates.dt.to_period("M").astype("string").to_numpy(),
            "hat_p_band": [
                _band_label(value, EPSILON_BINS) for value in hat_p
            ],
            "hat_m_band": [
                _band_label(value, DEFAULT_MINUTES_BINS) for value in hat_m
            ],
            "minutes_error_band": [
                _band_label(value, MINUTES_ERROR_BINS)
                for value in minutes_error
            ],
            "pit_half": half,
            "pit_rand": randomized,
        }
    )
    return frame


def _pack_frame(frame: pd.DataFrame, samples: np.ndarray) -> dict:
    actual = frame["actual"].to_numpy(dtype=float)
    start_rate = frame["start_rate_10"].to_numpy(dtype=float)
    location = decompose_location(
        actual,
        frame["hat_p"].to_numpy(dtype=float),
        frame["g"].to_numpy(dtype=float),
        frame["mean_pre"].to_numpy(dtype=float),
        frame["mean_post"].to_numpy(dtype=float),
    )
    return {
        "n": int(len(frame)),
        "location": location,
        "pit": _pit_summary(samples, actual),
        "metrics": score_samples(samples, actual, start_rate),
        "hat_p_signed_error": _table(
            signed_error_by_band(
                actual,
                frame["mean_post"].to_numpy(dtype=float),
                frame["hat_p"].to_numpy(dtype=float),
                EPSILON_BINS,
            )
        ),
        "hat_m_signed_error": _table(
            signed_error_by_band(
                actual,
                frame["mean_post"].to_numpy(dtype=float),
                frame["hat_m"].to_numpy(dtype=float),
                DEFAULT_MINUTES_BINS,
            )
        ),
        "clipping_by_hat_p": _table(
            clipping_uplift_by_band(
                frame["mean_pre"].to_numpy(dtype=float),
                frame["mean_post"].to_numpy(dtype=float),
                frame["hat_p"].to_numpy(dtype=float),
                EPSILON_BINS,
            )
        ),
        "by_role": _table(_slice_location(frame, "role")),
        "by_zero_points": _table(_slice_location(frame, "zero_points")),
        "by_hat_p_band": _table(_slice_location(frame, "hat_p_band")),
        "by_season": _table(_slice_location(frame, "season_year")),
        "by_month": _table(_slice_location(frame, "month")),
        "by_minutes_error": _table(
            _slice_location(frame, "minutes_error_band")
        ),
    }


def load_holdout() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = (
            ROOT
            / "data"
            / "silver"
            / "nba"
            / season
            / "regular_season"
            / "player_gamelogs.parquet"
        )
        frames.append(pd.read_parquet(path))
    panel = pd.concat(frames, ignore_index=True)
    panel = add_points_features(panel)
    appearances = panel.loc[panel["minutes"].gt(0)].copy()
    appearances["game_date"] = pd.to_datetime(appearances["game_date"])
    holdout = appearances.loc[
        appearances["season_year"].astype("string").eq(HOLDOUT_SEASON)
    ].copy()
    print(
        f"Holdout appearances: {len(holdout):,}  "
        f"{holdout['game_date'].min().date()} → "
        f"{holdout['game_date'].max().date()}",
        flush=True,
    )
    return holdout


def diagnose_holdout() -> dict:
    holdout = load_holdout()
    bundle = load_joint_points_bundle(
        minutes_mean_path=MINUTES_MEAN,
        minutes_dist_path=MINUTES_DIST,
        points_path=POINTS_PATH,
        joint_calibration_path=JOINT_CAL,
    )
    sim = JointPointsSimulator(bundle, n_draws=N_DRAWS)
    print("Predicting production hats...", flush=True)
    hat_m = sim.predict_minutes_means(holdout)
    hat_p = sim.predict_points_means(holdout, hat_m=hat_m)
    holdout = holdout.copy()
    holdout["minutes_hat"] = hat_m
    holdout["points_hat"] = hat_p
    fit = fold_fit_from_bundle(bundle)
    print(
        f"Simulating holdout current  n={len(holdout):,}  draws={N_DRAWS}",
        flush=True,
    )
    clipped, raw = simulate_fold(
        holdout,
        fit,
        n_draws=N_DRAWS,
        seed=EVAL_SEED,
        return_raw=True,
    )
    frame = _row_frame(holdout, fit, clipped, raw)
    packed = _pack_frame(frame, clipped)
    target = np.maximum(0.0, frame["hat_p_plus_g"].to_numpy(dtype=float))
    matched_raw = shift_raw_to_postclip_mean(raw, target)
    matched = np.maximum(0.0, matched_raw)
    packed["clip_match_metrics"] = score_samples(
        matched,
        frame["actual"].to_numpy(dtype=float),
        frame["start_rate_10"].to_numpy(dtype=float),
    )
    packed["clip_match_pit"] = _pit_summary(
        matched,
        frame["actual"].to_numpy(dtype=float),
    )
    packed["clip_match_location"] = decompose_location(
        frame["actual"].to_numpy(dtype=float),
        frame["hat_p"].to_numpy(dtype=float),
        frame["g"].to_numpy(dtype=float),
        matched_raw.mean(axis=1),
        matched.mean(axis=1),
    )
    return packed


def diagnose_preholdout() -> dict:
    panel = load_oof_panel(OOF_PANEL_ARTIFACT)
    folds = scored_folds(panel)
    fold_rows = []
    parts = []
    for fold in folds:
        valid = panel.loc[
            panel["base_fold"].eq(fold)
            & np.isfinite(panel["minutes_hat"])
            & np.isfinite(panel["points_hat"])
        ].copy()
        print(
            f"Preholdout fold={fold}  n={len(valid):,}  draws={N_DRAWS}",
            flush=True,
        )
        fit = fit_fold_variant(panel, fold=fold, variant="current")
        clipped, raw = simulate_fold(
            valid,
            fit,
            n_draws=N_DRAWS,
            seed=EVAL_SEED,
            return_raw=True,
        )
        frame = _row_frame(valid, fit, clipped, raw)
        actual = frame["actual"].to_numpy(dtype=float)
        start_rate = frame["start_rate_10"].to_numpy(dtype=float)
        current_metrics = score_samples(clipped, actual, start_rate)
        target = np.maximum(0.0, frame["hat_p_plus_g"].to_numpy(dtype=float))
        matched_raw = shift_raw_to_postclip_mean(raw, target)
        matched = np.maximum(0.0, matched_raw)
        matched_metrics = score_samples(matched, actual, start_rate)
        location = decompose_location(
            actual,
            frame["hat_p"].to_numpy(dtype=float),
            frame["g"].to_numpy(dtype=float),
            frame["mean_pre"].to_numpy(dtype=float),
            frame["mean_post"].to_numpy(dtype=float),
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "n": int(len(valid)),
                "prior_max_date": str(fit.prior_max_date),
                "valid_min_date": str(fit.valid_min_date),
                "location": location,
                "pit": _pit_summary(clipped, actual),
                "current": current_metrics,
                "clip_match": matched_metrics,
                "clip_match_pit": _pit_summary(matched, actual),
                "nll_delta": (
                    matched_metrics["nll"] - current_metrics["nll"]
                ),
                "width_ratio": (
                    matched_metrics["width_80"]
                    / current_metrics["width_80"]
                    if current_metrics["width_80"]
                    else None
                ),
            }
        )
        parts.append(frame)
    pooled = pd.concat(parts, ignore_index=True)
    actual = pooled["actual"].to_numpy(dtype=float)
    packed = {
        "n": int(len(pooled)),
        "location": decompose_location(
            actual,
            pooled["hat_p"].to_numpy(dtype=float),
            pooled["g"].to_numpy(dtype=float),
            pooled["mean_pre"].to_numpy(dtype=float),
            pooled["mean_post"].to_numpy(dtype=float),
        ),
        "hat_p_signed_error": _table(
            signed_error_by_band(
                actual,
                pooled["mean_post"].to_numpy(dtype=float),
                pooled["hat_p"].to_numpy(dtype=float),
                EPSILON_BINS,
            )
        ),
        "clipping_by_hat_p": _table(
            clipping_uplift_by_band(
                pooled["mean_pre"].to_numpy(dtype=float),
                pooled["mean_post"].to_numpy(dtype=float),
                pooled["hat_p"].to_numpy(dtype=float),
                EPSILON_BINS,
            )
        ),
        "by_role": _table(_slice_location(pooled, "role")),
        "by_zero_points": _table(_slice_location(pooled, "zero_points")),
        "by_hat_p_band": _table(_slice_location(pooled, "hat_p_band")),
        "by_season": _table(_slice_location(pooled, "season_year")),
        "by_minutes_error": _table(
            _slice_location(pooled, "minutes_error_band")
        ),
        "folds": _clean(fold_rows),
        "nll_wins": int(
            sum(row["nll_delta"] < 0 for row in fold_rows)
        ),
        "n_folds": int(len(fold_rows)),
    }
    return packed


def main() -> None:
    print("=== Holdout location diagnostics ===", flush=True)
    holdout = diagnose_holdout()
    print(
        "Holdout signed error vs post-clip "
        f"{holdout['location']['error_vs_post_clip']:.4f}  "
        "clipping uplift "
        f"{holdout['location']['clipping_uplift']:.4f}",
        flush=True,
    )
    print("=== Preholdout expanding folds ===", flush=True)
    preholdout = diagnose_preholdout()
    print(
        "Preholdout signed error vs post-clip "
        f"{preholdout['location']['error_vs_post_clip']:.4f}  "
        "clipping uplift "
        f"{preholdout['location']['clipping_uplift']:.4f}",
        flush=True,
    )
    payload = {
        "n_draws": int(N_DRAWS),
        "seed": int(EVAL_SEED),
        "holdout": holdout,
        "preholdout": preholdout,
        "note": (
            "2025-26 revealed the bias hypothesis and cannot independently "
            "validate a fix. Freeze current vs challenger for 2026-27."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(_clean(payload), indent=2))
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
