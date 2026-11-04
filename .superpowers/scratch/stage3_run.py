"""Stage 3: direct points baseline, nested CDF calibration, one holdout score.

The calibrator is accepted or rejected on walk-forward folds 3–5 before
any 2026-27 number is computed. Nothing is fit on the holdout.
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
import stage2_points_pmf as s2  # noqa: E402
from models.shared.artifacts import save_model_bundle  # noqa: E402
from models.shared.minutes_sampler import load_tail_sidecar as load_minutes_tails  # noqa: E402
from models.shared.points_calibration import (  # noqa: E402
    apply_points_cdf_calibrator,
    fit_points_cdf_calibrator,
    keep_calibrator,
    nested_points_calibrators,
)
from models.shared.points_pmf import check_feature_list, points_pmf, ppm_quantile_function  # noqa: E402
from models.shared.points_pricing import save_points_pricing  # noqa: E402
from models.shared.simulate import points_pmf as baseline_pmf  # noqa: E402
from models.shared.splits import date_walk_forward_folds  # noqa: E402
from models.shared.train import fit_quantile_models  # noqa: E402
from train_conditional_ppm import (  # noqa: E402
    FEATURES,
    QUANTILES,
    XGB_PARAMS,
    _assert_same_windows,
    _oof_frame,
    _split,
    prepare_frame,
)

OUT = ROOT / ".superpowers" / "scratch" / "stage3"
STAGE2 = ROOT / ".superpowers" / "scratch" / "stage2"
DIRECT_FEATURES = check_feature_list(
    [column for column in FEATURES if column != "minutes_given"],
    conditional=False,
)
check_feature_list(FEATURES, conditional=True)
KMAX = 90
G_M = 320
G_P = 1600
MINUTES_BUNDLE = ROOT / "models" / "saved_models" / "min_nba_model_2026-04-12.joblib"
MINUTES_TAILS = ROOT / "models" / "saved_models" / "min_nba_tails_2026-04-12.joblib"
RATE_TAILS = ROOT / "models" / "saved_models" / "pts_nba_tails_2026-04-12.joblib"


def _fit_direct_fold(train: pd.DataFrame, fold_info: dict) -> pd.DataFrame:
    train_rows = train.loc[fold_info["train_mask"]]
    val_rows = train.loc[fold_info["val_mask"]]
    overlap = set(pd.to_datetime(train_rows["game_date"]).dt.normalize()) & set(
        pd.to_datetime(val_rows["game_date"]).dt.normalize()
    )
    if overlap:
        raise AssertionError(f"fold {fold_info['fold']} shares a date with its training rows")
    models, preds = fit_quantile_models(
        train_rows.loc[:, DIRECT_FEATURES],
        train_rows["pts"],
        val_rows.loc[:, DIRECT_FEATURES],
        val_rows["pts"],
        quantiles=QUANTILES,
        xgb_params=dict(XGB_PARAMS),
        early_stop="train_tail",
        train_dates=train_rows["game_date"],
        train_tail_frac=0.10,
        X_predict=val_rows.loc[:, DIRECT_FEATURES],
    )
    if not np.isfinite(preds["q_0.50"]).all():
        raise AssertionError("non-finite direct prediction")
    from models.shared.metrics import pinball_loss

    pinball = pinball_loss(val_rows["pts"], preds["q_0.50"], 0.50)
    print(f"  fold {fold_info['fold']} points pinball q50 {pinball:.4f} n={len(val_rows):,}", flush=True)
    destination = OUT / "folds" / f"fold_{fold_info['fold']}.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": models, "features": DIRECT_FEATURES, "pinball_q50": pinball},
        destination,
    )
    part = _oof_frame(val_rows, preds, fold_info["fold"])
    part.to_parquet(OUT / "folds" / f"oof_{fold_info['fold']}.parquet", index=False)
    del models
    return part


def train_direct(frame: pd.DataFrame) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    oof_path = OUT / "oof.parquet"
    if oof_path.exists() and (OUT / "holdout_models.joblib").exists():
        oof = pd.read_parquet(oof_path)
        print(f"loaded direct OOF {len(oof):,}", flush=True)
        return oof
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        print(f"loaded direct OOF {len(oof):,}", flush=True)
        _fit_direct_holdout(frame)
        return oof
    train_pool, holdout = _split(frame)
    sample = train_pool.head(1500)
    smoke_params = dict(XGB_PARAMS)
    smoke_params["n_estimators"] = 30
    smoke_params["early_stopping_rounds"] = 5
    _models, smoke_preds = fit_quantile_models(
        sample.loc[:, DIRECT_FEATURES],
        sample["pts"],
        sample.tail(200).loc[:, DIRECT_FEATURES],
        sample.tail(200)["pts"],
        quantiles=[0.50],
        xgb_params=smoke_params,
        early_stop="train_tail",
        train_dates=sample["game_date"],
        train_tail_frac=0.10,
        X_predict=sample.tail(200).loc[:, DIRECT_FEATURES],
    )
    if not np.isfinite(smoke_preds["q_0.50"]).all():
        raise AssertionError("non-finite direct smoke prediction")
    print("direct smoke predictions are finite", flush=True)
    parts = []
    for fold_info in date_walk_forward_folds(train_pool, n_folds=5):
        part_path = OUT / "folds" / f"oof_{fold_info['fold']}.parquet"
        if part_path.exists():
            parts.append(pd.read_parquet(part_path))
            print(f"fold {fold_info['fold']} already fit", flush=True)
            continue
        started = time.perf_counter()
        print(fold_info["label"], flush=True)
        parts.append(_fit_direct_fold(train_pool, fold_info))
        print(f"  elapsed {time.perf_counter() - started:.0f}s", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    _assert_same_windows(oof)
    oof.to_parquet(oof_path, index=False)
    _fit_direct_holdout(frame)
    return oof


def _fit_direct_holdout(frame: pd.DataFrame) -> None:
    if (OUT / "holdout_models.joblib").exists():
        return
    train_pool, holdout = _split(frame)
    print("direct holdout fit", flush=True)
    models, preds = fit_quantile_models(
        train_pool.loc[:, DIRECT_FEATURES],
        train_pool["pts"],
        train_pool.loc[:, DIRECT_FEATURES],
        train_pool["pts"],
        quantiles=QUANTILES,
        xgb_params=dict(XGB_PARAMS),
        early_stop="train_tail",
        train_dates=train_pool["game_date"],
        train_tail_frac=0.10,
        X_predict=holdout.loc[:, DIRECT_FEATURES],
    )
    from models.shared.metrics import pinball_loss

    pinball = pinball_loss(holdout["pts"], preds["q_0.50"], 0.50)
    print(f"  holdout points pinball q50 {pinball:.4f} (not used to choose)", flush=True)
    joblib.dump(
        {"models": models, "features": DIRECT_FEATURES, "pinball_q50": pinball},
        OUT / "holdout_models.joblib",
    )
    _oof_frame(holdout, preds, -1).to_parquet(OUT / "holdout_predictions.parquet", index=False)


def _rint_pmf(frame, models, tails, minutes_tables, path: Path) -> np.ndarray:
    done_flag = path.with_suffix(".done")
    progress = path.with_suffix(".progress")
    knots = s2._minute_matrix(frame)
    if done_flag.exists():
        print(f"loaded {path.name}", flush=True)
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and progress.exists():
        stored = np.lib.format.open_memmap(path, mode="r+", dtype=np.float32, shape=(len(frame), KMAX + 1))
        start_at = int(progress.read_text())
    else:
        stored = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(frame), KMAX + 1))
        start_at = 0
    started = time.perf_counter()
    for start in range(start_at, len(frame), s2.BATCH):
        stop = min(start + s2.BATCH, len(frame))
        batch = frame.iloc[start:stop].reset_index(drop=True)
        minutes_q = s2._minutes_q(knots[start:stop], batch["starting"], minutes_tables, G_M)
        ppm = s2._ppm_q(batch, minutes_q, models, tails, G_P)
        stored[start:stop] = points_pmf(minutes_q, ppm, KMAX, "rint")
        stored.flush()
        progress.write_text(str(stop))
        print(f"  {path.name} {stop:,}/{len(frame):,}  {time.perf_counter() - started:.0f}s", flush=True)
    done_flag.write_text("ok")
    return np.asarray(stored)


def _direct_pmf(knots, starting, tails, grid: int) -> np.ndarray:
    knots = np.asarray(knots, dtype=float)
    starting = np.asarray(starting)
    stored = np.empty((len(knots), KMAX + 1), dtype=np.float32)
    for start in range(0, len(knots), 4000):
        stop = min(start + 4000, len(knots))
        uniforms = np.tile((np.arange(grid) + 0.5) / grid, (stop - start, 1))
        values = ppm_quantile_function(
            uniforms,
            knots[start:stop],
            QUANTILES,
            tails,
            np.asarray(starting[start:stop], dtype=int),
        )
        ones = np.ones((stop - start, grid, 1), dtype=float)
        stored[start:stop] = points_pmf(values, ones, KMAX, "rint")
    return stored


def _choose_direct_grid(knots, starting, tails) -> int:
    grid = 1000
    current = _direct_pmf(knots, starting, tails, grid)
    rows, ks = s1.evaluation_lines(
        s1.score_pmf_batch(current, np.zeros(len(knots)), np.zeros(len(knots)))["median"]
    )
    # The lines above use a zero outcome only to read the median. Recompute
    # from the PMF median directly.
    median = np.argmax(np.cumsum(current, axis=1) >= 0.5 - 1e-12, axis=1).astype(float)
    rows, ks = s1.evaluation_lines(median)
    while grid < 8000:
        refined = _direct_pmf(knots, starting, tails, grid * 2)
        before = s1.survival_at(current, rows, ks)
        after = s1.survival_at(refined, rows, ks)
        change = float(np.max(np.abs(after - before)))
        print(f"direct grid {grid} -> {grid * 2}  max |dP| {change:.5f}", flush=True)
        if change < 0.001:
            return grid
        current = refined
        rows, ks = s1.evaluation_lines(median)
        grid *= 2
    return grid


def _line_brier(pmf, centers, y):
    rows, ks = s1.evaluation_lines(centers)
    predicted = s1.survival_at(pmf, rows, ks)
    observed = (np.asarray(y)[rows] >= ks).astype(float)
    return (predicted - observed) ** 2, rows


def _decide(comparison, raw_pmf) -> dict:
    folds = comparison["fold_id"].to_numpy(dtype=int)
    y = comparison["pts"].to_numpy(dtype=float)
    dates = comparison["game_date"]
    centers = np.load(STAGE2 / "baseline_scores.npz")["median"]
    if len(centers) != len(comparison):
        raise AssertionError("baseline median rows do not match the comparison frame")
    nested_fit = nested_points_calibrators(raw_pmf, y, folds, dates, folds=(3, 4, 5))
    calibrated = nested_fit.pmf
    nested = nested_fit.evaluated
    curves = {
        str(fold_id): [round(float(value), 4) for value in curve]
        for fold_id, curve in nested_fit.curves.items()
    }
    for fold_id, curve in curves.items():
        print(f"g on folds before {fold_id}: {curve}", flush=True)
    atom = np.random.default_rng(42).random(int(nested.sum()))
    raw_scores = s1.score_pmf_batch(raw_pmf[nested], y[nested], atom)
    cal_scores = s1.score_pmf_batch(calibrated[nested], y[nested], atom)
    rps_ci = s1.cluster_mean_ci(cal_scores["rps"] - raw_scores["rps"], dates[nested])
    raw_brier, raw_rows = _line_brier(raw_pmf, centers, y)
    cal_brier, _cal_rows = _line_brier(calibrated, centers, y)
    keep_lines = nested[raw_rows]
    brier_ci = s1.cluster_mean_ci(
        cal_brier[keep_lines] - raw_brier[keep_lines],
        dates.to_numpy()[raw_rows[keep_lines]],
    )
    kept = keep_calibrator(
        rps_ci_high=rps_ci["ci_high"],
        reliability_ci_high=brier_ci["ci_high"],
    )
    print(
        f"nested dRPS {rps_ci['estimate']:.4f} ({rps_ci['ci_low']:.4f}, {rps_ci['ci_high']:.4f})  "
        f"dBrier {brier_ci['estimate']:.5f} ({brier_ci['ci_low']:.5f}, {brier_ci['ci_high']:.5f})  "
        f"kept={kept}",
        flush=True,
    )
    production = None
    if kept:
        production = fit_points_cdf_calibrator(
            np.cumsum(raw_pmf, axis=1),
            y,
            dates,
        )
    decision = {
        "kept": kept,
        "curves": curves,
        "rps_difference": rps_ci,
        "brier_difference": brier_ci,
        "nested_rows": int(nested.sum()),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    if production is not None:
        joblib.dump(production, OUT / "calibrator.joblib")
    return decision, production, calibrated, nested


def _holdout_frame(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = pd.read_parquet(ROOT / "data" / "oos" / "nba" / "minutes_predictions.parquet")
    rate = pd.read_parquet(ROOT / "data" / "oos" / "nba" / "rate_predictions.parquet")
    minutes = minutes.loc[minutes["is_holdout"].astype(bool)]
    rate = rate.loc[rate["is_holdout"].astype(bool)]
    levels = s2.LEVELS
    minutes = minutes.rename(columns={f"q_{level:.2f}": f"minutes_q_{level:.2f}" for level in levels})
    rate = rate.rename(columns={f"q_{level:.2f}": f"rate_q_{level:.2f}" for level in levels})
    keep_minutes = ["game_id", "player_id", "game_date", "starting", "minutes", "pts", *[
        f"minutes_q_{level:.2f}" for level in levels
    ]]
    keep_rate = ["game_id", "player_id", *[f"rate_q_{level:.2f}" for level in levels]]
    joined = minutes.loc[:, keep_minutes].merge(rate.loc[:, keep_rate], on=["game_id", "player_id"], how="inner")
    extra = [column for column in FEATURES if column not in joined.columns]
    joined = joined.merge(frame.loc[:, ["game_id", "player_id", *extra]], on=["game_id", "player_id"], how="left")
    dates = s1._naive_dates(joined["game_date"])
    if np.any(dates < np.datetime64("2026-10-20")):
        raise AssertionError("pre-holdout row in the holdout score")
    joined = joined.sort_values(["game_date", "game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    print(f"holdout rows {len(joined):,}", flush=True)
    return joined


def _score_table(frame, pmf, atom) -> list[dict]:
    y = frame["pts"].to_numpy(dtype=float)
    scored = s1.score_pmf_batch(np.asarray(pmf, dtype=float), y, atom)
    labeled = frame.copy()
    labeled["tier"] = s1.tier_labels(labeled["minutes_q_0.50"].to_numpy(dtype=float))
    columns = {
        "rps": scored["rps"],
        "log_score": scored["log_score"],
        "mean_bias": scored["mean"] - y,
        "median_bias": scored["median"] - y,
        "p0": scored["p0"],
        "y0": scored["y0"],
    }
    return s1.metric_table(labeled, columns), scored


def _direct_knots(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, [f"q_{level:.2f}" for level in QUANTILES]].to_numpy(dtype=float)


def main() -> None:
    frame = prepare_frame()
    direct = train_direct(frame)
    _oof, comparison, info = s2.load_scored_frame()
    minutes_tables = load_minutes_tails(MINUTES_TAILS)
    tails, tail_report = s2.fit_tails(pd.read_parquet(STAGE2 / "oof.parquet"))
    print(f"ppm tails {tail_report}", flush=True)

    knot_columns = [f"q_{level:.2f}" for level in QUANTILES]
    direct_cmp = comparison.loc[:, [
        "game_id", "player_id", "starting", "pts", "game_date", "fold_id", "minutes_q_0.50"
    ]].merge(
        direct.loc[:, ["game_id", "player_id", *knot_columns]],
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if direct_cmp["q_0.50"].isna().any():
        raise AssertionError("direct knots missing")
    tail_frame = direct.loc[direct["fold_id"].isin([1, 2, 3, 4])].reset_index(drop=True)
    tail_frame["ppm"] = tail_frame["pts"]
    knots = np.sort(tail_frame.loc[:, [f"q_{level:.2f}" for level in QUANTILES]].to_numpy(dtype=float), axis=1)
    invalid = (tail_frame["pts"].to_numpy(dtype=float) < knots[:, 0]) & (knots[:, 0] <= 0)
    print(f"direct tail rows dropped for q05<=0: {int(invalid.sum()):,}", flush=True)
    from models.shared.points_pmf import build_ppm_tails

    direct_tails = build_ppm_tails(tail_frame.loc[~invalid].copy(), folds=(1, 2, 3, 4))
    rng = np.random.default_rng(42)
    picked = np.sort(rng.choice(len(direct_cmp), size=300, replace=False))
    direct_grid = _choose_direct_grid(
        _direct_knots(direct_cmp.iloc[picked]),
        direct_cmp.iloc[picked]["starting"],
        direct_tails,
    )
    print(f"direct grid {direct_grid}", flush=True)
    direct_oof = _direct_pmf(_direct_knots(direct_cmp), direct_cmp["starting"], direct_tails, direct_grid)

    models = s2._models()
    raw_path = OUT / "conditional_oof_rint.npy"
    raw = _rint_pmf(comparison, models, tails, minutes_tables, raw_path)
    decision, production, _nested_pmf, nested = _decide(comparison, raw)

    print("holdout score", flush=True)
    holdout = _holdout_frame(frame)
    holdout_direct = pd.read_parquet(OUT / "holdout_predictions.parquet")
    holdout = holdout.merge(
        holdout_direct.loc[:, ["game_id", "player_id", *[f"q_{level:.2f}" for level in QUANTILES]]],
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if holdout["q_0.50"].isna().any():
        raise AssertionError("direct holdout knots missing")
    atom = np.random.default_rng(42).random(len(holdout))
    rate_tables = s1.load_rate_tails(RATE_TAILS)
    current, _mean, current_median = baseline_pmf(
        holdout,
        n_draws=s1.N_DRAWS,
        kmax=KMAX,
        minutes_tables=minutes_tables,
        rate_tables=rate_tables,
        seed=42,
    )
    holdout = holdout.copy()
    holdout["fold_id"] = -1
    payload = joblib.load(STAGE2 / "holdout_models.joblib")
    conditional = _rint_pmf(holdout, {-1: payload, 1: payload}, tails, minutes_tables, OUT / "conditional_holdout_rint.npy")
    # _ppm_q reads models[1]["features"] and models[fold]. fold is -1, so both keys are required.
    tables = {}
    tables["current"], _current_scores = _score_table(holdout, current, atom)
    tables["conditional"], conditional_scores = _score_table(holdout, conditional, atom)
    if production is not None:
        calibrated = apply_points_cdf_calibrator(np.cumsum(np.asarray(conditional, dtype=float), axis=1), production)
        tables["conditional_calibrated"], _cal_scores = _score_table(holdout, calibrated, atom)
    direct_holdout_pmf = _direct_pmf(_direct_knots(holdout), holdout["starting"], direct_tails, direct_grid)
    tables["direct"], _direct_scores = _score_table(holdout, direct_holdout_pmf, atom)
    lines = s2._line_table(conditional, current_median, holdout["pts"].to_numpy(dtype=float), holdout["game_date"].to_numpy())
    # OOF direct on the comparison rows, same baseline lines.
    oof_atom = np.random.default_rng(42).random(len(comparison))
    direct_oof_table, _direct_oof_scores = _score_table(comparison, direct_oof, oof_atom)
    nested_frame = comparison.loc[nested].copy()
    raw_nested_table, _raw_nested = _score_table(nested_frame, raw[nested], np.random.default_rng(42).random(int(nested.sum())))
    payload_out = {
        "rows": info,
        "direct_grid": direct_grid,
        "direct_features": DIRECT_FEATURES,
        "hyperparameters": "saved PPM bundle, not retuned for points",
        "decision": decision,
        "direct_oof": direct_oof_table,
        "nested_raw": raw_nested_table,
        "holdout": tables,
        "holdout_conditional_lines": lines,
        "holdout_rows": int(len(holdout)),
    }
    (OUT / "results.json").write_text(json.dumps(payload_out, indent=2))
    _package(payload, production, tails)
    print(json.dumps({"kept": decision["kept"], "holdout_rows": len(holdout)}, indent=2), flush=True)


def _package(conditional_payload, calibrator, tails) -> None:
    saved = ROOT / "models" / "saved_models"
    conditional_path = saved / "pts_nba_conditional_2026-04-12.joblib"
    direct_path = saved / "pts_nba_direct_2026-04-12.joblib"
    pricing_path = saved / "pts_nba_pricing_2026-04-12.joblib"
    for path in (conditional_path, direct_path, pricing_path):
        if path.exists():
            raise AssertionError(f"refusing to overwrite {path}")
    train = pd.DataFrame({"game_date": pd.to_datetime(["2025-04-13"])})
    holdout = pd.DataFrame({"game_date": pd.to_datetime(["2026-04-12"])})
    save_model_bundle(
        conditional_payload["models"],
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball_q50": json.loads((STAGE2 / "train_summary.json").read_text())["pinball_q50_by_fold"]}],
        ho_metrics={"note": "holdout pinball was not used to select"},
        features=conditional_payload["features"],
        artifact_stem="pts_nba_conditional",
        metadata={"grid_m": G_M, "grid_p": G_P, "rounding": "rint", "k_max": KMAX, "target": "pts_per_min"},
        save_dir=saved,
    )
    direct_payload = joblib.load(OUT / "holdout_models.joblib")
    save_model_bundle(
        direct_payload["models"],
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"note": "see stage3 oof parquet"}],
        ho_metrics={"note": "holdout pinball was not used to select"},
        features=direct_payload["features"],
        artifact_stem="pts_nba_direct",
        metadata={"target": "pts", "features": "pregame only"},
        save_dir=saved,
    )
    minutes = joblib.load(MINUTES_BUNDLE)
    conditional = joblib.load(conditional_path)
    save_points_pricing(
        pricing_path,
        minutes_bundle=MINUTES_BUNDLE,
        ppm_bundle=conditional_path,
        minutes_models=minutes["quantile_models"],
        minutes_features=minutes["feature_names"],
        ppm_models=conditional["quantile_models"],
        ppm_features=conditional["feature_names"],
        tails=tails,
        g_m=G_M,
        g_p=G_P,
        rounding="rint",
        k_max=KMAX,
        calibrator=calibrator,
        minutes_tails=MINUTES_TAILS,
    )
    print(f"wrote {pricing_path}", flush=True)


if __name__ == "__main__":
    if "--score" not in sys.argv:
        raise SystemExit(
            "Pass --score to fit the direct baseline, the nested calibrator, and the one holdout look."
        )
    main()
