# 8c. Date-tail holdout, raw knots beside calibrated knots.
# Offsets are fit on every walk-forward validation fold. Nothing here is
# fit on the 2025-26 rows. Raw is the booster output. Calibrated is
# sort, cold-start fallback, offset, sort, clip.

from minutes_xgb.config import MIN_TIERS, N_BOOT, QUANTILES, SEED
from minutes_xgb.scoring import (
    holdout_point_distribution,
    interval_sharpness,
    predicted_minute_tiers,
    quantile_calibration,
    quantile_crossing,
    tail_calibration_rows,
    wis_values,
)
from minutes_xgb.slices import regular_season_blocks, slice_score
from models.shared.conformal import (
    apply_offsets,
    base_knots,
    calibration_group,
    calibration_payload,
    fit_offsets,
    role_quantile_table,
)
from models.shared.metrics import paired_date_bootstrap
from models.shared.minutes_sampler import (
    build_tail_tables,
    save_tail_sidecar,
    tail_miss_rates,
    within_miss_rate_band,
)
from src.models.settlement import maximum_minutes

MIN_N = 300
MAX_MINUTES = maximum_minutes("nba")
oof = wf["oof"].copy()
knots = [f"q_{level:.2f}" for level in QUANTILES]
levels = {key: float(key.split("_", 1)[1]) for key in knots}

print(
    "Previous row-tail holdout (saved notebook run, last 10% of rows): "
    "q50 pinball 2.220, 80% coverage 80.0%, 90% coverage 89.6%."
)
print(
    "Date-tail holdout (last 10% of training dates): "
    f"q50 pinball {ho_metrics['pinball']:.3f}, "
    f"80% coverage {ho_metrics['coverage_80pct']:.1%}, "
    f"90% coverage {ho_metrics['coverage_90pct']:.1%}."
)

parts = []
for fold_id, part in oof.groupby("fold_id", sort=True):
    history = ppm_df.loc[
        wf_results[int(fold_id) - 1]["train_mask"], ["starting", "minutes"]
    ]
    mask = oof["fold_id"].eq(fold_id)
    folded = base_knots(
        {key: oof.loc[mask, key].to_numpy() for key in knots},
        oof.loc[mask, "starting"],
        ppm_df.loc[oof.index[mask.to_numpy()], "min_lag_1"],
        history,
    )
    folded.index = oof.index[mask.to_numpy()]
    parts.append(folded)
base_oof = pd.concat(parts).loc[oof.index]
lag_oof = ppm_df.loc[oof.index, "min_lag_1"]
group_oof = calibration_group(base_oof["q_0.50"], lag_oof)
offsets_all = fit_offsets(base_oof, oof["minutes"], group_oof, levels, min_n=MIN_N)
oof_cal = apply_offsets(base_oof, group_oof, offsets_all, MAX_MINUTES)
role_quantiles = role_quantile_table(ppm_df[["starting", "minutes"]], QUANTILES)
base_ho = base_knots(
    preds_ho,
    ppm_holdout["starting"],
    ppm_holdout["min_lag_1"],
    role_quantiles=role_quantiles,
)
group_ho = calibration_group(base_ho["q_0.50"], ppm_holdout["min_lag_1"])
preds_ho_cal = apply_offsets(base_ho, group_ho, offsets_all, MAX_MINUTES)
calibration = calibration_payload(
    offsets_all,
    group_oof,
    oof["game_date"],
    min_n=MIN_N,
    role_quantiles=role_quantiles,
    max_minutes=MAX_MINUTES,
)
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
raw_cross, raw_n, _ = quantile_crossing(raw_map)
cal_cross, _, _ = quantile_crossing(cal_map)
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
        predicted_minute_tiers(y_ho, raw_map, MIN_TIERS).assign(version="raw"),
        predicted_minute_tiers(y_ho, cal_map, MIN_TIERS).assign(version="calibrated"),
    ],
    ignore_index=True,
)
print("Predicted-minute tiers")
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
lag = ppm_holdout["min_lag_1"].to_numpy(dtype=float)
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
in_raw = (
    (np.asarray(y_ho, dtype=float) >= raw_map["q_0.10"])
    & (np.asarray(y_ho, dtype=float) <= raw_map["q_0.90"])
).astype(float)
in_cal = (
    (np.asarray(y_ho, dtype=float) >= cal_map["q_0.10"])
    & (np.asarray(y_ho, dtype=float) <= cal_map["q_0.90"])
).astype(float)
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

print(
    "\nTail miss rates on the next fold. Tables are fit on earlier calibrated folds only."
)
miss_rows = []
for k in (3, 4, 5):
    past_ids = list(range(1, k))
    past = oof["fold_id"].isin(past_ids).to_numpy()
    offsets_k = fit_offsets(
        base_oof.loc[past],
        oof.loc[past, "minutes"],
        group_oof[past],
        levels,
        min_n=MIN_N,
    )
    past_cal = apply_offsets(
        base_oof.loc[past], group_oof[past], offsets_k, MAX_MINUTES
    )
    past_frame = oof.loc[
        past,
        [
            "fold_id",
            "game_date",
            "minutes",
            "starting",
            "early_stop",
            "train_tail_frac",
        ],
    ].copy()
    for column in knots:
        past_frame[column] = past_cal[column].to_numpy()
    build_tail_tables(
        past_frame.reset_index(drop=True),
        folds=past_ids,
        fold_ranges=wf["fold_ranges"],
        check_miss_rate=False,
    )
    test = oof["fold_id"].to_numpy() == k
    test_cal = apply_offsets(
        base_oof.loc[test], group_oof[test], offsets_k, MAX_MINUTES
    )
    raw_lower, raw_upper = tail_miss_rates(
        oof.loc[test, "minutes"],
        base_oof.loc[test, "q_0.05"],
        base_oof.loc[test, "q_0.95"],
    )
    cal_lower, cal_upper = tail_miss_rates(
        oof.loc[test, "minutes"],
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
miss_rates = pd.DataFrame(miss_rows)
display(miss_rates.round(4))

ship = oof[
    ["fold_id", "game_date", "minutes", "starting", "early_stop", "train_tail_frac"]
].copy()
for column in knots:
    ship[column] = oof_cal[column].to_numpy()
tail_tables = build_tail_tables(
    ship.reset_index(drop=True),
    folds=[1, 2, 3, 4, 5],
    fold_ranges=wf["fold_ranges"],
    check_miss_rate=False,
)
tail_path = ROOT / "models/saved_models/min_nba_tails_cal_2026-04-12.joblib"
save_tail_sidecar(tail_tables, tail_path)
print(f"Wrote calibrated tail tables to {tail_path.relative_to(ROOT)}")
