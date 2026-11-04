"""Rebuild points_xgboost.ipynb as a thin caller of points_xgb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/alexgonzalez/Documents/nba_quant")
OLD = json.loads(
    (ROOT / "notebooks/nba/points/points_xgboost.ipynb").read_text()
)


def src(index: int) -> str:
    return "".join(OLD["cells"][index]["source"])


header = src(0).replace("**Last Updated:** 10/3/26", "**Last Updated:** 10/26/26")
note = (
    "Conformal offsets are fit on walk-forward out-of-fold rows only. "
    "The 2025-26 holdout is scored after that fit and is never used to choose an offset."
)
guard = "2025-26 is scored once and never used to tune"
if guard in header and note not in header:
    header = header.replace(guard, guard + "\n\n" + note)
timing = src(10)
conclusion_heading = src(42) or "### Conclusion"
conclusion = src(43)

SETUP = '''import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_SRC = Path("src").resolve()
if not (ANALYSIS_SRC / "points_xgb").is_dir():
    ANALYSIS_SRC = Path.cwd() / "notebooks/nba/points/src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from points_xgb.paths import ensure_repo_on_path, repo_root

ensure_repo_on_path()

from points_xgb import (
    ARTIFACT_STEM,
    HOLDOUT_SEASON,
    ID_COLS,
    LAG_COL,
    N_BOOT,
    PREDICTIONS_PATH,
    PTS_FEATURES,
    QUANTILES,
    RATE_CAP,
    RATE_SPEC,
    RATE_TIERS,
    ROLE_COL,
    SEED,
    TAIL_SIDECAR,
    TARGET_COL,
    XGB_PARAMS,
    apply_train_tukey,
    assert_no_leak,
    assign_rate_target,
    build_miss_frame,
    build_model_frame,
    build_oos_predictions,
    calibrated_tail_frame,
    calibration_table,
    compare_to_baselines,
    data_quality,
    exploratory_metadata,
    feature_consensus,
    feature_missingness,
    feature_outlier_table,
    fit_rate_offsets,
    fold_base_knots,
    format_data_quality,
    format_rate_outliers,
    holdout_calibrated,
    holdout_point_distribution,
    holdout_role_quantiles,
    interval_sharpness,
    load_regular_season,
    loaded_holdout_scores,
    mean_absolute_shap,
    miss_view,
    missingness_summary,
    monotone_shift,
    nested_offset_backtest,
    permutation_importance_table,
    plot_missingness,
    plot_rate,
    plot_shap,
    positive_minutes,
    predicted_rate_tiers,
    quantile_calibration,
    quantile_crossing,
    rate_outlier_summary,
    rate_payload,
    regular_season_blocks,
    role_reading,
    rotation_team_games,
    score_predictions,
    season_coverage,
    season_missingness,
    slice_score,
    tail_calibration_rows,
    top_rate_rows,
    wis_values,
    write_oos_predictions,
)
from models.shared.analysis import analyze_correlations, run_feature_ablation
from models.shared.artifacts import load_model_bundle, save_model_bundle
from models.shared.conformal import apply_offsets
from models.shared.metrics import paired_date_bootstrap
from models.shared.minutes_sampler import tail_miss_rates, within_miss_rate_band
from models.shared.ppm_sampler import build_tail_tables, save_tail_sidecar
from models.shared.splits import prepare_splits
from models.shared.train import evaluate_holdout, run_timeseries_cv, run_walk_forward

ROOT = repo_root()
pd.set_option("display.max_columns", None)
'''

DATE_TAIL = '''# Date-tail holdout, raw knots beside calibrated knots.
# Offsets are fit on every walk-forward validation fold. Nothing here is
# fit on the 2025-26 rows. Raw is the booster output. Calibrated is
# sort, cold-start fallback, offset, sort, clip to [0, 6].

oof = wf["oof"].copy()
knots = [f"q_{level:.2f}" for level in QUANTILES]
base_oof, group_oof = fold_base_knots(
    oof, ppm_df, [result["train_mask"] for result in wf_results]
)
offsets_all = fit_rate_offsets(base_oof, oof, group_oof)
oof_cal = apply_offsets(
    base_oof, group_oof, offsets_all, RATE_SPEC.clip_high, lower=RATE_SPEC.clip_low
)
role_quantiles = holdout_role_quantiles(ppm_df)
base_ho, group_ho, preds_ho_cal = holdout_calibrated(
    preds_ho, ppm_holdout, offsets_all, role_quantiles
)
calibration = rate_payload(offsets_all, group_oof, oof["game_date"], role_quantiles)
print("OOF calibration rows", calibration["date_start"], "→", calibration["date_end"])
print("row counts", calibration["row_counts"])
display(offsets_all.round(3))

raw_map = {key: np.asarray(preds_ho[key], dtype=float) for key in knots}
cal_map = {key: preds_ho_cal[key].to_numpy(dtype=float) for key in knots}
base_map = {key: base_ho[key].to_numpy(dtype=float) for key in knots}
shift = offsets_all.reindex(pd.Index(group_ho)).fillna(0.0)
shift.index = base_ho.index
added = base_ho[knots].to_numpy() + shift[knots].to_numpy()
reorder = float(np.mean(np.any(np.diff(added, axis=1) < 0, axis=1)))
raw_cross, raw_n, _pairs = quantile_crossing(raw_map)
cal_cross, _, _pairs = quantile_crossing(cal_map)
print(
    f"Crossing rows: raw {raw_cross}/{raw_n} ({raw_cross / raw_n:.2%}), "
    f"calibrated {cal_cross}/{raw_n} ({cal_cross / raw_n:.2%}). "
    f"Second-sort reorder after the offset: {reorder:.2%}."
)

calibration_side = pd.concat(
    [
        quantile_calibration(y_ho, raw_map).assign(version="raw"),
        quantile_calibration(y_ho, cal_map).assign(version="calibrated"),
    ],
    ignore_index=True,
)
print("Holdout calibration. delta = P(y ≤ q) − τ")
display(
    calibration_side.pivot(index="quantile", columns="version", values="delta").round(4)
)

tiers = pd.concat(
    [
        predicted_rate_tiers(y_ho, raw_map, RATE_TIERS).assign(version="raw"),
        predicted_rate_tiers(y_ho, cal_map, RATE_TIERS).assign(version="calibrated"),
    ],
    ignore_index=True,
)
print("Predicted-rate tiers")
display(tiers.round(4))

point_table = pd.DataFrame(
    [
        holdout_point_distribution("raw", y_ho, raw_map),
        holdout_point_distribution("calibrated", y_ho, cal_map),
    ]
)
tail_table = pd.DataFrame(
    tail_calibration_rows("raw", y_ho, raw_map)
    + tail_calibration_rows("calibrated", y_ho, cal_map)
)
sharp = pd.concat(
    [
        interval_sharpness(y_ho, raw_map).assign(version="raw"),
        interval_sharpness(y_ho, cal_map).assign(version="calibrated"),
    ],
    ignore_index=True,
)
print("q50 RMSE and 11-knot CRPS. CRPS is twice the mean pinball at the trained knots.")
display(point_table.round(4))
print("Tail calibration. delta = empirical − nominal")
display(tail_table.round(4))
print("Interval sharpness")
display(sharp.round(4))

dates = pd.to_datetime(ppm_holdout["game_date"])
starting = ppm_holdout["starting"].to_numpy()
lag = ppm_holdout[LAG_COL].to_numpy(dtype=float)
masks = {
    **regular_season_blocks(dates),
    "Confirmed starter": starting == 1,
    "Bench": starting == 0,
    "No prior game": ~np.isfinite(lag),
}
slice_rows = []
for name, mask in masks.items():
    chosen = np.asarray(mask, dtype=bool)
    slice_rows.append({**slice_score(name, chosen, y_ho, raw_map), "version": "raw"})
    slice_rows.append(
        {**slice_score(name, chosen, y_ho, cal_map), "version": "calibrated"}
    )
    if name == "No prior game":
        slice_rows.append(
            {**slice_score(name, chosen, y_ho, base_map), "version": "base fallback"}
        )
print("Calendar, role, and cold-start slices")
display(pd.DataFrame(slice_rows).round(4))


def _row_pinball(actual, pred, alpha=0.50):
    residual = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    return np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)


pin_ci = paired_date_bootstrap(
    _row_pinball(y_ho, raw_map["q_0.50"]),
    _row_pinball(y_ho, cal_map["q_0.50"]),
    dates,
    n_boot=N_BOOT,
    seed=SEED,
)
wis_ci = paired_date_bootstrap(
    wis_values(y_ho, raw_map),
    wis_values(y_ho, cal_map),
    dates,
    n_boot=N_BOOT,
    seed=SEED,
)
actual = np.asarray(y_ho, dtype=float)
in_raw = ((actual >= raw_map["q_0.10"]) & (actual <= raw_map["q_0.90"])).astype(float)
in_cal = ((actual >= cal_map["q_0.10"]) & (actual <= cal_map["q_0.90"])).astype(float)
cov_ci = paired_date_bootstrap(in_raw, in_cal, dates, n_boot=N_BOOT, seed=SEED)
print(
    "Date-clustered 95% CI, calibrated − raw. "
    f"q50 pinball {pin_ci['diff']:+.4f} "
    f"[{pin_ci['ci_low']:+.4f}, {pin_ci['ci_high']:+.4f}]. "
    f"WIS {wis_ci['diff']:+.4f} "
    f"[{wis_ci['ci_low']:+.4f}, {wis_ci['ci_high']:+.4f}]. "
    f"80% coverage {cov_ci['diff']:+.4f} "
    f"[{cov_ci['ci_low']:+.4f}, {cov_ci['ci_high']:+.4f}]."
)

print("\\nTail miss rates on the next fold. Tables are fit on earlier calibrated folds only.")
miss_rows = []
for k in (3, 4, 5):
    past = oof["fold_id"].to_numpy() < k
    offsets_k = fit_rate_offsets(
        base_oof.loc[past].reset_index(drop=True),
        oof.loc[past].reset_index(drop=True),
        group_oof[past],
    )
    test = oof["fold_id"].to_numpy() == k
    test_cal = apply_offsets(
        base_oof.loc[test], group_oof[test], offsets_k, RATE_SPEC.clip_high, lower=0.0
    )
    raw_lower, raw_upper = tail_miss_rates(
        oof.loc[test, TARGET_COL],
        base_oof.loc[test, "q_0.05"],
        base_oof.loc[test, "q_0.95"],
    )
    cal_lower, cal_upper = tail_miss_rates(
        oof.loc[test, TARGET_COL],
        test_cal["q_0.05"],
        test_cal["q_0.95"],
    )
    miss_rows.append(
        {
            "fold": k,
            "raw_lower": raw_lower,
            "raw_upper": raw_upper,
            "cal_lower": cal_lower,
            "cal_upper": cal_upper,
            "raw_in_band": within_miss_rate_band(raw_lower, raw_upper),
            "cal_in_band": within_miss_rate_band(cal_lower, cal_upper),
        }
    )
display(pd.DataFrame(miss_rows).round(4))

tail_frame = calibrated_tail_frame(oof, oof_cal)
tail_tables = build_tail_tables(
    tail_frame,
    folds=[1, 2, 3, 4],
    fold_ranges=wf["fold_ranges"],
)
tail_path = ROOT / TAIL_SIDECAR
save_tail_sidecar(tail_tables, tail_path)
print(f"Wrote calibrated tail tables to {tail_path.relative_to(ROOT)}")
'''

CELLS = [
    ("markdown", header),
    ("markdown", "### Setup and Configurations"),
    ("code", SETUP),
    ("code", 'print(f"Holdout season: {HOLDOUT_SEASON}")\nprint(f"Rate cap: {RATE_CAP}")\nprint(f"Quantiles: {QUANTILES}")'),
    ("markdown", "### Data"),
    ("code", "df = load_regular_season(ROOT)\ndf.sample(5)"),
    ("code", "print(format_data_quality(data_quality(df)))"),
    ("markdown", "### Features"),
    ("code", "df = build_model_frame(df)"),
    (
        "code",
        """assert_no_leak(PTS_FEATURES)
missing = [column for column in PTS_FEATURES if column not in df.columns]
if missing:
    raise KeyError(f"Rate features missing from the panel: {missing}")
print(f"Number of features: {len(PTS_FEATURES)}")
print(", ".join(PTS_FEATURES))
print(f"Number of nulls: {int(df[PTS_FEATURES].isna().sum().sum())}")
""",
    ),
    ("markdown", timing),
    (
        "code",
        """df = positive_minutes(df)
df = assign_rate_target(df)
print(f"After filtering appearances: {len(df):,} rows")
print(df[TARGET_COL].describe().round(3))

splits = prepare_splits(
    df,
    holdout_season=HOLDOUT_SEASON,
    features=PTS_FEATURES,
    target_col=TARGET_COL,
    id_cols=ID_COLS,
    role_col=ROLE_COL,
    extra_keep=[
        "minutes",
        "pts",
        "pts_per_min_lag_1",
        "season_pts_per_min_mean",
        "pts_per_min_ewm_hl_10",
    ],
)
ppm_df = splits["ppm_df"]
ppm_holdout = splits["ppm_holdout"]
print(f"Train pool (excl. {HOLDOUT_SEASON}) : {len(ppm_df):,} rows")
print(f"Holdout   ({HOLDOUT_SEASON})        : {len(ppm_holdout):,} rows")

ppm_df, fence_low, fence_high, n_dropped = apply_train_tukey(ppm_df)
X = ppm_df[splits["features"]].copy()
y = ppm_df[TARGET_COL]
print(
    f"Train Tukey fence [{fence_low:.2f}, {fence_high:.2f}]: "
    f"dropped {n_dropped:,} extreme rates"
)
print(f"Train pool after outlier drop: {len(ppm_df):,} rows")
print(f"Holdout still includes every appearance: {len(ppm_holdout):,} rows")
""",
    ),
    ("markdown", "### EDA"),
    (
        "code",
        """miss = feature_missingness(df, PTS_FEATURES)
print(missingness_summary(df, PTS_FEATURES))
display(miss)
season_miss = season_missingness(df, miss)
if not season_miss.empty:
    display(season_miss.round(3))
plot_missingness(miss)
plt.show()
""",
    ),
    (
        "code",
        """outlier_stats = rate_outlier_summary(df, TARGET_COL, RATE_CAP)
print(format_rate_outliers(outlier_stats))
display(top_rate_rows(df, TARGET_COL))
plot_rate(df, TARGET_COL, outlier_stats["fence_high"])
plt.show()
display(feature_outlier_table(df, PTS_FEATURES))
""",
    ),
    (
        "code",
        """coverage = season_coverage(df, PTS_FEATURES)
display(coverage)
""",
    ),
    (
        "code",
        """correlated_features = analyze_correlations(
    df[PTS_FEATURES].dropna(),
    threshold=0.85,
)
""",
    ),
    ("markdown", "### Models"),
    (
        "code",
        """# Hyperparameter tuning
# Leave commented if you don't want to retune the model

# import importlib
# import models.shared.train as train
#
# importlib.reload(train)
# tune_xgb_quantile = train.tune_xgb_quantile
#
# result = tune_xgb_quantile(X, y, n_trials=50, n_splits=3)
# XGB_PARAMS = result["best_params"]
""",
    ),
    (
        "code",
        """tscv_results = run_timeseries_cv(
    X, y, ppm_df,
    xgb_params=XGB_PARAMS,
    role_col=ROLE_COL,
    tiers=RATE_TIERS,
    quantiles=QUANTILES,
)
wf = run_walk_forward(
    X, y, ppm_df,
    xgb_params=XGB_PARAMS,
    role_col=ROLE_COL,
    tiers=RATE_TIERS,
    quantiles=QUANTILES,
    n_folds=5,
    target_col=TARGET_COL,
)

wf_results = wf["wf_results"]
models_last = wf["models_last"]
preds_last = wf["preds_last"]
X_val_last = wf["X_val_last"]
y_val_last = wf["y_val_last"]
starting_last = wf["starting_last"]
last_fold = wf["last_fold"]
""",
    ),
    (
        "markdown",
        "### Conformal calibration\n\nWalk-forward out-of-fold rows only. Nothing here is fit on the 2025-26 holdout.\n",
    ),
    (
        "code",
        """oof = wf["oof"].copy()
base_oof, group_oof = fold_base_knots(
    oof, ppm_df, [result["train_mask"] for result in wf_results]
)
print(pd.Series(group_oof).value_counts().to_string())
y_oof = oof[TARGET_COL].to_numpy(dtype=float)
dates_oof = pd.to_datetime(oof["game_date"]).to_numpy()
starting_oof = oof["starting"].to_numpy()
print(
    f"Go/no-go on base knots. delta = P(y <= q) - tau, "
    f"with {N_BOOT:,} date-clustered resamples, seed {SEED}."
)
go_nogo = calibration_table(
    y_oof,
    dates_oof,
    base_oof.reset_index(drop=True),
    group_oof,
    starting=starting_oof,
    n_boot=N_BOOT,
    seed=SEED,
)
display(go_nogo.round(4))

backtest = nested_offset_backtest(
    oof.reset_index(drop=True),
    base_oof.reset_index(drop=True),
    group_oof,
    n_boot=N_BOOT,
    seed=SEED,
)
for fold_id, table in backtest["offset_tables"].items():
    print(f"\\nOffsets fit on folds before {fold_id}")
    display(table.round(3))
print(
    "\\nNested backtest, folds 3-5. diff = calibrated - raw. "
    "Lower WIS, CRPS, and pinball are better."
)
display(backtest["nested"].round(4))
display(backtest["scores"].round(4))
if backtest["flags"].empty:
    print("No group/knot swings by more than its mean absolute offset.")
else:
    display(backtest["flags"].round(3))
""",
    ),
    (
        "code",
        """wf_pinball, wf_coverage = score_predictions(
    y_val_last, preds_last, last_fold["fold"]
)
display(wf_pinball.round(4))
display(wf_coverage.round(4))
""",
    ),
    (
        "code",
        """holdout = evaluate_holdout(
    ppm_df,
    ppm_holdout,
    features=PTS_FEATURES,
    target_col=TARGET_COL,
    xgb_params=XGB_PARAMS,
    role_col=ROLE_COL,
    tiers=RATE_TIERS,
    wf_results=wf_results,
    quantiles=QUANTILES,
    fold_label=f"{HOLDOUT_SEASON} Blind Holdout",
)
models_ho = holdout["models_ho"]
preds_ho = holdout["preds_ho"]
ho_metrics = holdout["ho_metrics"]
X_ho = holdout["X_ho"]
y_ho = holdout["y_ho"]

ho_pinball, ho_coverage = score_predictions(
    y_ho, preds_ho, f"{HOLDOUT_SEASON} holdout"
)
display(ho_pinball.round(4))
display(ho_coverage.round(4))
""",
    ),
    (
        "markdown",
        "### Date-tail holdout, raw beside calibrated\n\nEarly stopping uses the last 10% of training dates. Offsets are the all-fold walk-forward table. The 2025-26 rows are scored, not fit.\n",
    ),
    ("code", DATE_TAIL),
    (
        "code",
        """calibration_raw = quantile_calibration(y_ho, preds_ho)
display(calibration_raw.round(4))
n_crossed, n_rows, crossing_pairs = quantile_crossing(preds_ho)
print(f"Crossing rows: {n_crossed}/{n_rows}")
display(crossing_pairs.round(4))
display(predicted_rate_tiers(y_ho, preds_ho, RATE_TIERS).round(4))
display(interval_sharpness(y_ho, preds_ho).round(4))
""",
    ),
    ("markdown", "### Baseline"),
    (
        "code",
        """report = compare_to_baselines(
    y_ho,
    preds_ho,
    ppm_holdout,
    ppm_df,
    wf_results,
    target_col=TARGET_COL,
    tiers=RATE_TIERS,
    n_boot=N_BOOT,
    seed=SEED,
)
print(
    f"Identical holdout rows: {report['paired_n']:,} | "
    f"dropped {report['dropped_n']:,}"
)
display(report["point_table"].round(4))
display(report["delta_table"].round(4))
display(report["fold_table"].round(4))
display(report["distribution"].round(4))
display(report["calibration"].round(4))
display(report["subgroup_table"].round(3))
display(report["tail_table"].round(4))
for line in report["checks"]:
    print(line)
""",
    ),
    ("markdown", "### Features Analysis"),
    (
        "code",
        """feat_imp = mean_absolute_shap(models_last["q_0.50"], X_val_last)
plot_shap(feat_imp)
plt.show()
""",
    ),
    (
        "code",
        """pi_df, perm_result = permutation_importance_table(
    models_last["q_0.50"],
    X_val_last,
    y_val_last,
)
print(pi_df.to_string())
""",
    ),
    (
        "code",
        """train_mask_last = last_fold["train_mask"]
val_mask_last = last_fold["val_mask"]
ablation_df = run_feature_ablation(
    PTS_FEATURES,
    X.loc[train_mask_last],
    ppm_df.loc[train_mask_last, TARGET_COL],
    X.loc[val_mask_last],
    ppm_df.loc[val_mask_last, TARGET_COL],
    xgb_params=XGB_PARAMS,
)
""",
    ),
    (
        "code",
        """feature_audit = feature_consensus(
    list(X_val_last.columns),
    feat_imp,
    pi_df,
    perm_result.importances,
    ablation_df,
)
print(feature_audit.to_string())
""",
    ),
    ("markdown", "### Slice Analysis"),
    (
        "code",
        """dates = pd.to_datetime(ppm_holdout["game_date"])
raw_q = {key: np.asarray(pred, dtype=float) for key, pred in preds_ho.items()}
blocks = regular_season_blocks(dates)
print("April is late regular season, not the postseason")
holdout_season = df.loc[df["season_year"].astype("string").eq(str(HOLDOUT_SEASON))]
print(holdout_season["season_type"].astype("string").value_counts(dropna=False).to_string())
calendar = pd.DataFrame(
    [slice_score(name, mask, y_ho, raw_q) for name, mask in blocks.items()]
)
display(calendar.round(3))

starting = ppm_holdout["starting"].to_numpy()
role = pd.DataFrame(
    [
        slice_score("Confirmed starter", starting == 1, y_ho, raw_q),
        slice_score("Bench", starting == 0, y_ho, raw_q),
    ]
)
display(role.round(3))
for _, row in role.iterrows():
    if row["n"] == 0:
        continue
    print(
        f"  {row['slice']}: bias {row['bias']:+.3f} pts/min, "
        f"P(actual <= q50) {row['p_actual_le_q50']:.1%}. {role_reading(row)}."
    )

preds_ho_mono, shift, before, after = monotone_shift(raw_q)
print(f"Crossing before {before:.2%} of rows; after rearrangement {after:.2%}.")
display(shift.round(4))
""",
    ),
    ("markdown", "### Error Analysis"),
    (
        "code",
        """miss = build_miss_frame(df, ppm_holdout, preds_ho)
way_off = miss["abs_error"] >= 0.25
print(
    f"{HOLDOUT_SEASON} holdout: {int(way_off.sum()):,} / {len(miss):,} rows "
    f"missed by 0.25+ points per minute ({way_off.mean():.1%}). "
    f"Median |error| {miss['abs_error'].median():.1f}."
)
display(miss_view(miss.nsmallest(20, "error")))
display(miss_view(miss.nlargest(20, "error")))
biggest = pd.concat(
    [miss.nsmallest(20, "error"), miss.nlargest(20, "error")],
    ignore_index=True,
)
print(biggest["hint"].replace("", "unlabeled").value_counts().to_string())
games = rotation_team_games(miss)
display(games.head(15).round(1))
""",
    ),
    ("markdown", "### Save Exploratory Artifact"),
    ("code", "metadata = exploratory_metadata(ppm_df, ppm_holdout, ho_metrics, wf_results)\ndisplay(metadata)"),
    (
        "code",
        """save_path = save_model_bundle(
    models_ho,
    train_df=ppm_df,
    holdout_df=ppm_holdout,
    wf_results=wf_results,
    ho_metrics=ho_metrics,
    features=PTS_FEATURES,
    artifact_stem=ARTIFACT_STEM,
    metadata=metadata,
    calibration=calibration,
)
bundle = load_model_bundle(save_path)
assert list(X_ho.columns) == list(bundle["feature_names"])
scores = loaded_holdout_scores(bundle["quantile_models"], X_ho, y_ho)
print("Saved quantiles:", ", ".join(scores["quantiles"]))
print(f"Loaded model pinball q50 (holdout): {scores['pinball_q50']:.3f}")
print(f"Loaded model 80% coverage (holdout): {scores['coverage_80']:.1%}")
print(f"Trained on data up to      : {bundle['train_end'].date()}")
print(f"Holdout through            : {bundle['val_end'].date()}")
""",
    ),
    (
        "code",
        """predictions = build_oos_predictions(
    wf, ppm_df, ppm_holdout, preds_ho, oof_cal=oof_cal, preds_ho_cal=preds_ho_cal
)
path = write_oos_predictions(predictions, ROOT / PREDICTIONS_PATH)
print(f"Wrote {len(predictions):,} rows to {path.relative_to(ROOT)}")
cal_cols = [column for column in predictions.columns if column.startswith("cal_q_")]
print(f"Calibrated knot columns: {len(cal_cols)}")
predictions.groupby("window_id").agg(
    rows=("game_id", "size"),
    first=("game_date", "min"),
    last=("game_date", "max"),
)
""",
    ),
    ("markdown", conclusion_heading),
    ("markdown", conclusion),
]


def as_cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        cell["id"] = None
    return cell


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": OLD.get("metadata", {}),
    "cells": [as_cell(kind, source) for kind, source in CELLS],
}
for index, cell in enumerate(notebook["cells"]):
    cell.pop("id", None)
    if cell["cell_type"] == "code" and "def score_predictions" in "".join(cell["source"]):
        raise SystemExit(f"inline score_predictions remains in cell {index}")
    if cell["cell_type"] == "code" and "def slice_score" in "".join(cell["source"]):
        raise SystemExit(f"inline slice_score remains in cell {index}")
    if cell["cell_type"] == "code" and "def _iso_date" in "".join(cell["source"]):
        raise SystemExit(f"inline _iso_date remains in cell {index}")

out = ROOT / "notebooks/nba/points/points_xgboost.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {len(notebook['cells'])} cells")
