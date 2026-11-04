# 8b. One-sided conformal offsets on walk-forward OOF.
# Base knots are the sorted booster output, with each fold's cold-start
# role quantiles taken from that fold's training dates (the early-stopping
# tail is included there, and is not used as a calibration score).

from minutes_xgb.config import N_BOOT, SEED
from minutes_xgb.scoring import wis_values
from models.shared.conformal import (
    apply_offsets,
    base_knots,
    calibration_group,
    fit_offsets,
)
from models.shared.metrics import paired_date_bootstrap, pinball_loss
from models.shared.minutes_sampler import row_crps
from models.shared.splits import date_walk_forward_folds
from src.models.settlement import maximum_minutes

MIN_N = 300
MAX_MINUTES = maximum_minutes("nba")
NESTED_FOLDS = (3, 4, 5)
MONTHS = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun")

oof = wf["oof"].copy()
print("oof columns:", list(oof.columns))
try:
    print("direct duplicated:", int(oof.duplicated(["game_id", "player_id"]).sum()))
except KeyError as exc:
    missing = list(getattr(exc, "args", [exc])[0])
    print(f"wf['oof'] has no {missing} columns; joining ids from the train pool")
ids = ppm_df.loc[oof.index, ["game_id", "player_id"]]
dup = int(ids.duplicated(["game_id", "player_id"]).sum())
print("duplicated game_id, player_id:", dup)
print("oof index duplicated:", int(oof.index.duplicated().sum()))
if dup or oof.index.duplicated().any():
    raise RuntimeError("OOF player-games are duplicated; not deduping")

tail_frac = float(pd.unique(oof["train_tail_frac"])[0])
n_folds = int(oof["fold_id"].nunique())
leak_rows = []
for fold in date_walk_forward_folds(ppm_df, n_folds=n_folds):
    part = oof.loc[oof["fold_id"].eq(fold["fold"])]
    oof_dates = set(pd.to_datetime(part["game_date"]))
    train_dates = set(pd.to_datetime(np.asarray(fold["train_dates"])))
    val_dates = set(pd.to_datetime(np.asarray(fold["val_dates"])))
    unique = pd.unique(pd.to_datetime(ppm_df.loc[fold["train_mask"], "game_date"]))
    n_stop = max(1, int(np.floor(len(unique) * tail_frac)))
    stop = set(pd.to_datetime(unique[-n_stop:]))
    leak_rows.append(
        {
            "fold": fold["fold"],
            "train_end": pd.Timestamp(fold["train_dates"][-1]).date(),
            "val_start": pd.Timestamp(fold["val_dates"][0]).date(),
            "val_end": pd.Timestamp(fold["val_dates"][-1]).date(),
            "n": int(len(part)),
            "val_in_train": len(oof_dates & train_dates),
            "val_in_stop": len(oof_dates & stop),
            "dates_match": oof_dates == val_dates,
        }
    )
    if oof_dates & train_dates or oof_dates & stop or oof_dates != val_dates:
        raise RuntimeError(f"fold {fold['fold']} leaks or does not match its dates")
leak = pd.DataFrame(leak_rows)
display(leak)

fold_ids = sorted(int(value) for value in oof["fold_id"].unique())
if fold_ids != list(range(1, len(wf_results) + 1)):
    raise RuntimeError("walk-forward fold ids are not 1..n in wf_results order")
knots = sorted(
    (column for column in oof.columns if str(column).startswith("q_")),
    key=lambda column: float(str(column).split("_", 1)[1]),
)
levels = {key: float(str(key).split("_", 1)[1]) for key in knots}
level_grid = np.array([levels[key] for key in knots])

parts = []
for fold_id, part in oof.groupby("fold_id", sort=True):
    history = ppm_df.loc[
        wf_results[int(fold_id) - 1]["train_mask"], ["starting", "minutes"]
    ]
    built = base_knots(
        part[knots],
        part["starting"].to_numpy(),
        ppm_df.loc[part.index, "min_lag_1"].to_numpy(),
        history,
    )
    parts.append(built)
base = pd.concat(parts).reindex(oof.index)
if not np.isfinite(base.to_numpy(dtype=float)).all():
    raise RuntimeError("base knots are not finite")
lag = ppm_df.loc[oof.index, "min_lag_1"].to_numpy()
group = calibration_group(base["q_0.50"].to_numpy(), lag)
y = oof["minutes"].to_numpy(dtype=float)
dates = pd.to_datetime(oof["game_date"]).to_numpy()
print(pd.Series(group).value_counts().to_string())
oof = oof.reset_index(drop=True)
base = base.reset_index(drop=True)


def _gap(indicator, when, target):
    covered = np.asarray(indicator, dtype=float)
    if covered.size == 0:
        raise ValueError("empty slice")
    boot = paired_date_bootstrap(
        np.full(covered.shape[0], float(target)),
        covered,
        when,
        n_boot=N_BOOT,
        seed=SEED,
    )
    return {
        "n": int(boot["n"]),
        "empirical": float(np.mean(covered)),
        "target": float(target),
        "delta": float(boot["diff"]),
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
    }


def _slice_rows(name, mask, frame):
    chosen = np.asarray(mask, dtype=bool)
    actual = y[chosen]
    when = dates[chosen]
    grid = frame.loc[:, knots].to_numpy(dtype=float)[chosen]
    rows = []
    for column, (key, tau) in enumerate(levels.items()):
        row = _gap(actual <= grid[:, column], when, tau)
        row.update({"slice": name, "check": key})
        rows.append(row)
    for check, low, high, target in (
        ("coverage_80", knots.index("q_0.10"), knots.index("q_0.90"), 0.80),
        ("coverage_90", knots.index("q_0.05"), knots.index("q_0.95"), 0.90),
    ):
        inside = (actual >= grid[:, low]) & (actual <= grid[:, high])
        row = _gap(inside, when, target)
        row.update({"slice": name, "check": check})
        rows.append(row)
    return rows


slices = [("overall", np.ones(len(oof), dtype=bool))]
starting = oof["starting"].to_numpy()
slices.append(("starter", starting == 1))
slices.append(("bench", starting == 0))
for name in ("cold_start", "<15", "15-24", "24-31", "31+"):
    slices.append((name, group == name))
month = pd.to_datetime(oof["game_date"]).dt.strftime("%b")
for name in MONTHS:
    mask = (month == name).to_numpy()
    if mask.any():
        slices.append((name, mask))

print(
    f"\nGo/no-go on base knots. delta = P(y <= q) - tau, "
    f"with {N_BOOT:,} date-clustered resamples, seed {SEED}."
)
go_nogo = pd.DataFrame(
    [row for name, mask in slices if mask.any() for row in _slice_rows(name, mask, base)]
)
display(go_nogo.round(4))


def _mean_abs_delta(actual, frame):
    gaps = [
        abs(float(np.mean(actual <= frame[key].to_numpy(dtype=float))) - tau)
        for key, tau in levels.items()
    ]
    return float(np.mean(gaps))


def _band(actual, frame, low, high):
    lo = frame[f"q_{low:.2f}"].to_numpy(dtype=float)
    hi = frame[f"q_{high:.2f}"].to_numpy(dtype=float)
    inside = (actual >= lo) & (actual <= hi)
    return float(np.mean(inside)), float(np.mean(hi - lo))


def _qmap(frame):
    return {key: frame[key].to_numpy(dtype=float) for key in knots}


def _shift(frame, labels, offsets):
    grid = frame[knots].to_numpy(dtype=float)
    add = np.zeros_like(grid)
    if len(offsets):
        lookup = {str(name): row for row, name in enumerate(offsets.index)}
        matched = np.array([lookup.get(str(label), -1) for label in labels])
        known = matched >= 0
        add[known] = offsets.loc[:, knots].to_numpy(dtype=float)[matched[known]]
    return grid + add


nested_rows = []
score_rows = []
delta_rows = []
offset_tables = {}
for k in NESTED_FOLDS:
    fit_mask = oof["fold_id"].to_numpy() < k
    test_mask = oof["fold_id"].to_numpy() == k
    offsets = fit_offsets(
        base.loc[fit_mask], y[fit_mask], group[fit_mask], levels, min_n=MIN_N
    )
    counts = pd.Series(group[fit_mask]).value_counts()
    shown = offsets.copy()
    shown.insert(0, "n", counts.reindex(shown.index).fillna(0).astype(int))
    offset_tables[k] = shown
    print(f"\nOffsets fit on folds before {k}")
    print("rows per group:\n" + counts.sort_index().to_string())
    omitted = counts[counts < MIN_N]
    if len(omitted):
        print("omitted (n < min_n):\n" + omitted.to_string())
    display(shown.round(3))

    labels = group[test_mask]
    actual = y[test_mask]
    when = dates[test_mask]
    raw = base.loc[test_mask, knots].reset_index(drop=True)
    added = _shift(raw, labels, offsets)
    reorder = float(np.mean(np.any(np.diff(added, axis=1) < 0, axis=1)))
    cal = apply_offsets(raw, labels, offsets, MAX_MINUTES)
    versions = {"raw": raw, "cal": cal}
    for version, frame in versions.items():
        blocks = [("all", np.ones(len(frame), dtype=bool))]
        for name in pd.unique(labels):
            blocks.append((str(name), labels == name))
        for name, mask in blocks:
            part = frame.loc[mask]
            outcome = actual[mask]
            cov80, width80 = _band(outcome, part, 0.10, 0.90)
            cov90, width90 = _band(outcome, part, 0.05, 0.95)
            nested_rows.append(
                {
                    "fold": k,
                    "version": version,
                    "group": name,
                    "n": int(mask.sum()),
                    "mean_abs_delta": _mean_abs_delta(outcome, part),
                    "cov80": cov80,
                    "width80": width80,
                    "cov90": cov90,
                    "width90": width90,
                    "pinball_q50": pinball_loss(outcome, part["q_0.50"], 0.50),
                    "reorder": reorder,
                }
            )
            for key, tau in levels.items():
                delta_rows.append(
                    {
                        "fold": k,
                        "version": version,
                        "group": name,
                        "knot": key,
                        "delta": float(np.mean(outcome <= part[key].to_numpy())) - tau,
                    }
                )
    wis_raw = wis_values(actual, _qmap(raw))
    wis_cal = wis_values(actual, _qmap(cal))
    crps_raw = row_crps(actual, raw[knots].to_numpy(dtype=float), level_grid)
    crps_cal = row_crps(actual, cal[knots].to_numpy(dtype=float), level_grid)
    wis_ci = paired_date_bootstrap(wis_raw, wis_cal, when, n_boot=N_BOOT, seed=SEED)
    crps_ci = paired_date_bootstrap(crps_raw, crps_cal, when, n_boot=N_BOOT, seed=SEED)
    score_rows.append(
        {
            "fold": k,
            "n": int(test_mask.sum()),
            "reorder": reorder,
            "pinball_q50_raw": pinball_loss(actual, raw["q_0.50"], 0.50),
            "pinball_q50_cal": pinball_loss(actual, cal["q_0.50"], 0.50),
            "wis_raw": float(np.mean(wis_raw)),
            "wis_cal": float(np.mean(wis_cal)),
            "wis_diff": wis_ci["diff"],
            "wis_ci_low": wis_ci["ci_low"],
            "wis_ci_high": wis_ci["ci_high"],
            "crps_raw": float(np.mean(crps_raw)),
            "crps_cal": float(np.mean(crps_cal)),
            "crps_diff": crps_ci["diff"],
            "crps_ci_low": crps_ci["ci_low"],
            "crps_ci_high": crps_ci["ci_high"],
        }
    )

nested = pd.DataFrame(nested_rows)
deltas = pd.DataFrame(delta_rows)
scores = pd.DataFrame(score_rows)
print("\nNested backtest, folds 3-5. diff = calibrated - raw. Lower WIS, CRPS, and pinball are better.")
display(nested.round(4))
display(deltas.pivot_table(index=["fold", "version", "group"], columns="knot", values="delta").round(4))
display(scores.round(4))

flag_rows = []
long_offsets = []
for k, table in offset_tables.items():
    for grp, row in table.drop(columns="n").iterrows():
        for key, value in row.items():
            long_offsets.append(
                {"k": k, "group": grp, "knot": key, "offset": float(value)}
            )
offset_long = pd.DataFrame(long_offsets)
for (grp, key), part in offset_long.groupby(["group", "knot"], sort=False):
    values = part["offset"].to_numpy(dtype=float)
    if values.size < 2:
        continue
    swing = float(values.max() - values.min())
    magnitude = float(np.mean(np.abs(values)))
    if swing > magnitude:
        flag_rows.append(
            {
                "group": grp,
                "knot": key,
                "swing": swing,
                "mean_abs": magnitude,
                "offsets": ", ".join(f"{value:.3f}" for value in values),
            }
        )
offset_flags = pd.DataFrame(flag_rows)
print("\nOffset stability across fits on folds before k = 3, 4, 5")
if offset_flags.empty:
    print("No group/knot swings by more than its mean absolute offset.")
else:
    display(offset_flags.round(3))
