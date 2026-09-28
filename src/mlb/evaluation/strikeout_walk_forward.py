"""Expanding 28-day block walk that stores each pregame feature row once."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mlb.evaluation.walk_forward_blocks import assign_walk_forward_blocks
from src.mlb.evaluation.walk_forward_fit import (
    UnestimatedDispersion,
    fit_walk_forward_nb2,
    predict_walk_forward_mean,
)
from src.mlb.evaluation.walk_forward_snapshot import SnapshotStore
from src.mlb.models.nb_scores import (
    half_point_probabilities,
    nb2_crps,
    nb2_nll,
    nb2_quantile,
)
from src.mlb.models.pmf import negative_binomial_pmf
from src.mlb.models.shrinkage import shrink_rate
from src.mlb.pipeline.walk_forward_features import (
    NB_RATE_FEATURES,
    OUTPUT_COLUMNS,
    REQUIRED_COLUMNS,
    WORKLOAD_FEATURES,
    feature_block,
)
from src.mlb.schemas import PMF_COLUMNS

FIRST_SEASON = 2018
WORKLOAD_MIN_TRAIN_STARTS = 24
WORKLOAD_BINARY = ("home_flag", "no_prior_regular_start", "long_layoff_flag")
WORKLOAD_L2 = 1.0
STRIKEOUT_BINARY = ("home_flag",)
STRIKEOUT_L2 = 2.0
OFFSET_FLOOR = 1e-6
PMF_K_MAX = 15
TAIL_MASS_THRESHOLD = 0.001

STATUS_OK = "ok"
STATUS_UNESTIMATED = "unestimated_dispersion"
STATUS_INSUFFICIENT = "insufficient_history"
STATUS_FEATURES_ONLY = "features_only"


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(dtype=float)).all(axis=1), index=frame.index)


FEATURE_ONLY_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in ("pitcher_id", "game_pk")]
RAW_COLUMNS = list(REQUIRED_COLUMNS) + ["walk_forward_block"]


def _attach(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Raw start columns plus feature columns; ``feature_block`` keeps block row order."""
    raw = raw.loc[:, RAW_COLUMNS].reset_index(drop=True)
    features = features.loc[:, FEATURE_ONLY_COLUMNS].reset_index(drop=True)
    out = pd.concat([raw, features], axis=1)
    out["predicted_bf_oof"] = np.nan
    return out


def _first_season_rows(
    raw: pd.DataFrame, stored: list[pd.DataFrame], *, m: float, kappa: float, rest_cap: float
) -> None:
    """Date-level features for the first season; history is its earlier eligible starts."""
    for date in sorted(raw["game_date"].unique()):
        history = raw[raw["game_date"] < date]
        if history.empty:
            continue
        today = raw[raw["game_date"] == date]
        try:
            features = feature_block(history, today, m=m, kappa=kappa, rest_cap=rest_cap)
        except ValueError as exc:
            if "rest median" not in str(exc):
                raise
            # First starts cannot get a finite rest yet, so only pitchers seen earlier qualify.
            seen = today["pitcher_id"].isin(history["pitcher_id"])
            if not seen.any():
                continue
            today = today[seen]
            features = feature_block(history, today, m=m, kappa=kappa, rest_cap=rest_cap)
        rows = _attach(today, features)
        rows = rows[_finite(rows, WORKLOAD_FEATURES)]
        if not rows.empty:
            rows["fit_status"] = STATUS_FEATURES_ONLY
            stored.append(rows)


def _distinct_features(train: pd.DataFrame, features: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a feature whose training column equals an earlier one.

    First-season history makes ``pitcher_prior_regular_starts`` equal ``season_starts_prior``.
    """
    kept: list[str] = []
    for column in features:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        if any(
            np.array_equal(values, pd.to_numeric(train[k], errors="coerce").to_numpy(dtype=float))
            for k in kept
        ):
            continue
        kept.append(column)
    return tuple(kept)


def _fit_workload(history: pd.DataFrame):
    train = history[_finite(history, WORKLOAD_FEATURES)]
    if len(train) < WORKLOAD_MIN_TRAIN_STARTS:
        return None, STATUS_INSUFFICIENT
    try:
        model = fit_walk_forward_nb2(
            train,
            "batters_faced",
            _distinct_features(train, WORKLOAD_FEATURES),
            WORKLOAD_BINARY,
            WORKLOAD_L2,
        )
    except UnestimatedDispersion:
        return None, STATUS_UNESTIMATED
    return model, STATUS_OK


def _offset(predicted_bf: pd.Series) -> np.ndarray:
    return np.log(np.maximum(predicted_bf.to_numpy(dtype=float), OFFSET_FLOOR))


def _score_strikeouts(history: pd.DataFrame, rows: pd.DataFrame) -> str:
    train = history[
        (history["season"] != FIRST_SEASON)
        & history["predicted_bf_oof"].notna()
        & _finite(history, NB_RATE_FEATURES)
    ]
    if train.empty:
        return STATUS_INSUFFICIENT
    try:
        model = fit_walk_forward_nb2(
            train,
            "strikeouts",
            _distinct_features(train, NB_RATE_FEATURES),
            STRIKEOUT_BINARY,
            STRIKEOUT_L2,
            offset=_offset(train["predicted_bf_oof"]),
        )
    except UnestimatedDispersion:
        return STATUS_UNESTIMATED
    mu = predict_walk_forward_mean(model, rows, offset=_offset(rows["predicted_bf_oof"]))
    rows["predicted_strikeout_mean"] = mu
    rows["negative_binomial_dispersion"] = model.alpha
    pmf = negative_binomial_pmf(mu, model.alpha, PMF_K_MAX, TAIL_MASS_THRESHOLD)
    rows[list(PMF_COLUMNS)] = pmf
    return STATUS_OK


def walk_blocks(
    starts: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    m: float,
    kappa: float,
    rest_cap: float,
    through_season: int,
    score_strikeouts: bool,
) -> pd.DataFrame:
    """Snapshot rows for eligible blocks in ``(season, block_index)`` order.

    Each start is featurized once and never rebuilt. The caller writes the frame.
    """
    assigned = assign_walk_forward_blocks(starts, calendar)
    eligible = assigned[assigned["model_eligible"] & (assigned["season"] <= through_season)].copy()
    eligible["game_date"] = pd.to_datetime(eligible["game_date"]).dt.strftime("%Y-%m-%d")
    eligible["_sort_start"] = pd.to_datetime(eligible["scheduled_start_utc"], utc=True)
    eligible = eligible.sort_values(["game_date", "_sort_start"], kind="mergesort").drop(
        columns="_sort_start"
    )
    eligible = eligible.loc[:, RAW_COLUMNS].reset_index(drop=True)
    label_parts = eligible["walk_forward_block"].str.split("-", expand=True).astype(int)
    eligible["_season"] = label_parts[0]
    eligible["_block_index"] = label_parts[1]

    stored: list[pd.DataFrame] = []
    first = eligible[eligible["_season"] == FIRST_SEASON].drop(columns=["_season", "_block_index"])
    if not first.empty:
        _first_season_rows(first, stored, m=m, kappa=kappa, rest_cap=rest_cap)

    later = eligible[eligible["_season"] > FIRST_SEASON]
    for (_, _), block in later.groupby(["_season", "_block_index"], sort=True):
        block = block.drop(columns=["_season", "_block_index"])
        history = pd.concat(stored, ignore_index=True)
        block_open = block["game_date"].min()
        history = history[history["game_date"] < block_open]

        model, status = _fit_workload(history)
        features = feature_block(history, block, m=m, kappa=kappa, rest_cap=rest_cap)
        rows = _attach(block, features)
        if model is not None:
            rows["predicted_bf_oof"] = predict_walk_forward_mean(model, rows)
            if score_strikeouts:
                status = _score_strikeouts(history, rows)
        rows["fit_status"] = status
        stored.append(rows)

    if stored:
        result = pd.concat(stored, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=RAW_COLUMNS + FEATURE_ONLY_COLUMNS + ["predicted_bf_oof", "fit_status"]
        )
    if score_strikeouts:
        for column in ("predicted_strikeout_mean", "negative_binomial_dispersion", *PMF_COLUMNS):
            if column not in result.columns:
                result[column] = np.nan
    result["season"] = result["season"].astype(int)
    result["predicted_bf_oof"] = result["predicted_bf_oof"].astype(float)
    return result


TUNING_SEASON = 2020
M_GRID = (25, 50, 75, 100, 150)
KAPPA_GRID = (1, 2, 3, 4, 6)
REST_CAP_GRID = (14, 21, 28, 35)
OVER_LINES = (4.5, 5.5, 6.5)
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
RELIABILITY_MIN_ROWS = 200
SEASON_ROLES = {
    2019: "offset_generation",
    2020: "tuning",
    2021: "reported",
    2022: "reported",
    2023: "reported",
    2024: "reported",
    2025: "holdout",
}
ABORTING_ROLES = ("reported", "holdout")

REPORT_COLUMNS = [
    "walk_forward_block",
    "role",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "n_train",
    "n_validation",
    "workload_mae",
    "workload_bias",
    "strikeout_nll",
    "discrete_crps",
    "median_mae",
    "coverage_q25_q75",
    "nominal_q25_q75",
    "coverage_q10_q90",
    "nominal_q10_q90",
    "empirical_cdf_q10",
    "empirical_cdf_q25",
    "empirical_cdf_q50",
    "empirical_cdf_q75",
    "empirical_cdf_q90",
    "brier_4_5",
    "brier_5_5",
    "brier_6_5",
    "mean_predicted_over_4_5",
    "observed_over_rate_4_5",
    "mean_predicted_over_5_5",
    "observed_over_rate_5_5",
    "mean_predicted_over_6_5",
    "observed_over_rate_6_5",
]


@dataclass
class Lock:
    m: float
    kappa: float
    rest_cap: float
    store: SnapshotStore


def _line_label(line: float) -> str:
    return str(line).replace(".", "_")


def _workload_mae(frame: pd.DataFrame, season: int) -> float:
    rows = frame[
        (frame["season"] == season)
        & frame["predicted_bf_oof"].notna()
        & frame["batters_faced"].notna()
    ]
    if rows.empty:
        return float("nan")
    error = rows["predicted_bf_oof"].to_numpy(dtype=float) - rows["batters_faced"].to_numpy(dtype=float)
    return float(np.mean(np.abs(error)))


def _resmooth(frame: pd.DataFrame, m: float) -> pd.DataFrame:
    """K/BF columns rebuilt at prior strength ``m`` from the stored raw sums."""
    out = frame.copy()
    league = out["league_k_per_bf_prior"].to_numpy(dtype=float)

    def rate(k: str, bf: str) -> np.ndarray:
        return np.asarray(shrink_rate(out[k].to_numpy(dtype=float), out[bf].to_numpy(dtype=float), league, m))

    pitcher_last10 = rate("pitcher_k_last10", "pitcher_bf_last10")
    pitcher_season = rate("pitcher_k_season", "pitcher_bf_season")
    opponent_last10 = rate("opponent_k_last10", "opponent_bf_last10")
    opponent_season = rate("opponent_k_season", "opponent_bf_season")
    out["pitcher_k_per_bf_last3_smoothed"] = rate("pitcher_k_last3", "pitcher_bf_last3")
    out["pitcher_k_per_bf_last10_smoothed"] = pitcher_last10
    out["pitcher_k_per_bf_season_to_date_smoothed"] = np.where(
        out["pitcher_bf_season"].to_numpy(dtype=float) == 0, pitcher_last10, pitcher_season
    )
    out["opponent_k_rate_vs_starters_last10_smoothed"] = opponent_last10
    out["opponent_k_rate_vs_starters_season_to_date_smoothed"] = np.where(
        out["opponent_bf_season"].to_numpy(dtype=float) == 0, opponent_last10, opponent_season
    )
    return out


def _workload_valid(frame: pd.DataFrame, season: int) -> bool:
    rows = frame[frame["season"] == season]
    return bool(
        not rows.empty
        and not (rows["fit_status"] == STATUS_UNESTIMATED).any()
        and rows["predicted_bf_oof"].notna().all()
    )


def _tuning_nll(frame: pd.DataFrame) -> float:
    """Pooled 2020 strikeout NLL, each block fit only on rows dated before it opens.

    NaN when a block has no legal training rows or an unestimated dispersion.
    """
    tuning = frame[frame["season"] == TUNING_SEASON]
    if tuning.empty:
        return float("nan")
    scored_blocks: list[pd.DataFrame] = []
    for label in sorted(tuning["walk_forward_block"].unique(), key=_block_order):
        rows = tuning[tuning["walk_forward_block"] == label].copy()
        history = frame[frame["game_date"] < rows["game_date"].min()]
        if _score_strikeouts(history, rows) != STATUS_OK:
            return float("nan")
        scored_blocks.append(rows)
    scored = pd.concat(scored_blocks, ignore_index=True)
    return nb2_nll(
        scored["strikeouts"].to_numpy(dtype=int),
        scored["predicted_strikeout_mean"].to_numpy(dtype=float),
        scored["negative_binomial_dispersion"].to_numpy(dtype=float),
    )


def select_2020(starts: pd.DataFrame, calendar: pd.DataFrame) -> Lock:
    """Lock kappa and rest cap on 2020 workload MAE, then m on 2020 strikeout NLL."""
    best_workload: tuple[float, float, float, pd.DataFrame] | None = None
    for kappa in KAPPA_GRID:
        for rest_cap in REST_CAP_GRID:
            try:
                frame = walk_blocks(
                    starts,
                    calendar,
                    m=M_GRID[0],
                    kappa=kappa,
                    rest_cap=rest_cap,
                    through_season=TUNING_SEASON,
                    score_strikeouts=False,
                )
            except UnestimatedDispersion:
                continue
            if not _workload_valid(frame, TUNING_SEASON):
                continue
            mae = _workload_mae(frame, TUNING_SEASON)
            if not np.isfinite(mae):
                continue
            if best_workload is None or mae < best_workload[0]:
                best_workload = (mae, kappa, rest_cap, frame)
    if best_workload is None:
        raise UnestimatedDispersion("no 2020 workload candidate produced a finite MAE")
    _, kappa, rest_cap, kept = best_workload

    best_strikeout: tuple[float, float, pd.DataFrame] | None = None
    for m in M_GRID:
        candidate = _resmooth(kept, m)
        nll = _tuning_nll(candidate)
        if not np.isfinite(nll):
            continue
        if best_strikeout is None or nll < best_strikeout[0]:
            best_strikeout = (nll, m, candidate)
    if best_strikeout is None:
        raise UnestimatedDispersion("no 2020 strikeout candidate produced a finite NLL")
    _, m, winner = best_strikeout

    store = SnapshotStore()
    store.write(winner)
    return Lock(m=m, kappa=kappa, rest_cap=rest_cap, store=store)


def block_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Workload and strikeout scores for one set of validation rows."""
    metrics: dict[str, float] = {column: float("nan") for column in REPORT_COLUMNS[8:]}
    metrics["nominal_q25_q75"] = 0.50
    metrics["nominal_q10_q90"] = 0.80

    workload = predictions[
        predictions["predicted_bf_oof"].notna() & predictions["batters_faced"].notna()
    ]
    if not workload.empty:
        error = workload["predicted_bf_oof"].to_numpy(dtype=float) - workload["batters_faced"].to_numpy(dtype=float)
        metrics["workload_mae"] = float(np.mean(np.abs(error)))
        metrics["workload_bias"] = float(np.mean(error))

    if "predicted_strikeout_mean" not in predictions.columns:
        return metrics
    scored = predictions[
        predictions["predicted_strikeout_mean"].notna()
        & predictions["negative_binomial_dispersion"].notna()
        & predictions["strikeouts"].notna()
    ]
    if scored.empty:
        return metrics
    y = scored["strikeouts"].to_numpy(dtype=int)
    mu = scored["predicted_strikeout_mean"].to_numpy(dtype=float)
    alpha = scored["negative_binomial_dispersion"].to_numpy(dtype=float)
    metrics["strikeout_nll"] = nb2_nll(y, mu, alpha)
    metrics["discrete_crps"] = nb2_crps(y, mu, alpha)
    quantiles = {level: nb2_quantile(mu, alpha, level) for level in QUANTILE_LEVELS}
    metrics["median_mae"] = float(np.mean(np.abs(quantiles[0.50] - y)))
    metrics["coverage_q25_q75"] = float(np.mean((y >= quantiles[0.25]) & (y <= quantiles[0.75])))
    metrics["coverage_q10_q90"] = float(np.mean((y >= quantiles[0.10]) & (y <= quantiles[0.90])))
    for level, values in quantiles.items():
        metrics[f"empirical_cdf_q{round(level * 100):02d}"] = float(np.mean(y <= values))
    for line in OVER_LINES:
        label = _line_label(line)
        over, _ = half_point_probabilities(mu, alpha, line)
        hit = (y > line).astype(float)
        metrics[f"brier_{label}"] = float(np.mean((over - hit) ** 2))
        metrics[f"mean_predicted_over_{label}"] = float(np.mean(over))
        metrics[f"observed_over_rate_{label}"] = float(np.mean(hit))
    return metrics


def reliability_bins(predictions: pd.DataFrame, line: float) -> pd.DataFrame:
    """Over-probability reliability on tenths; bins under 200 rows are absent."""
    columns = ["bin_left", "bin_right", "n", "mean_predicted_over", "observed_over_rate"]
    scored = predictions[
        predictions["predicted_strikeout_mean"].notna()
        & predictions["negative_binomial_dispersion"].notna()
        & predictions["strikeouts"].notna()
    ]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    over, _ = half_point_probabilities(
        scored["predicted_strikeout_mean"].to_numpy(dtype=float),
        scored["negative_binomial_dispersion"].to_numpy(dtype=float),
        line,
    )
    hit = (scored["strikeouts"].to_numpy(dtype=float) > line).astype(float)
    edges = np.round(np.linspace(0.0, 1.0, 11), 10)
    index = np.clip(np.searchsorted(edges, over, side="right") - 1, 0, len(edges) - 2)
    records = []
    for b in range(len(edges) - 1):
        inside = index == b
        n = int(inside.sum())
        if n < RELIABILITY_MIN_ROWS:
            continue
        records.append({
            "bin_left": float(edges[b]),
            "bin_right": float(edges[b + 1]),
            "n": n,
            "mean_predicted_over": float(np.mean(over[inside])),
            "observed_over_rate": float(np.mean(hit[inside])),
        })
    return pd.DataFrame.from_records(records, columns=columns)


def _block_order(label: str) -> tuple[int, int]:
    season, index = label.split("-")
    return int(season), int(index)


def score_locked_seasons(
    starts: pd.DataFrame,
    calendar: pd.DataFrame,
    lock: Lock,
    seasons,
) -> pd.DataFrame:
    """One report row per walk-forward block in ``seasons`` plus a pooled aggregate row."""
    seasons = sorted(int(s) for s in seasons)
    frame = walk_blocks(
        starts,
        calendar,
        m=lock.m,
        kappa=lock.kappa,
        rest_cap=lock.rest_cap,
        through_season=max(seasons),
        score_strikeouts=True,
    )
    in_scope = frame[frame["season"].isin(seasons)]
    labels = sorted(in_scope["walk_forward_block"].unique(), key=_block_order)

    records = []
    for label in labels:
        block = in_scope[in_scope["walk_forward_block"] == label]
        season = int(block["season"].iloc[0])
        role = SEASON_ROLES.get(season, "reported")
        if role in ABORTING_ROLES and (block["fit_status"] == STATUS_UNESTIMATED).any():
            raise UnestimatedDispersion(
                f"strikeout dispersion was not estimated for {role} block {label}"
            )
        block_open = block["game_date"].min()
        train = frame[frame["game_date"] < block_open]
        records.append({
            "walk_forward_block": label,
            "role": role,
            "train_start": train["game_date"].min() if not train.empty else None,
            "train_end": train["game_date"].max() if not train.empty else None,
            "validation_start": block_open,
            "validation_end": block["game_date"].max(),
            "n_train": len(train),
            "n_validation": len(block),
            **block_metrics(block),
        })

    if labels:
        last_open = in_scope.loc[in_scope["walk_forward_block"] == labels[-1], "game_date"].min()
        union_train = frame[frame["game_date"] < last_open]
        records.append({
            "walk_forward_block": "all",
            "role": "aggregate",
            "train_start": union_train["game_date"].min() if not union_train.empty else None,
            "train_end": union_train["game_date"].max() if not union_train.empty else None,
            "validation_start": in_scope["game_date"].min(),
            "validation_end": in_scope["game_date"].max(),
            "n_train": len(union_train),
            "n_validation": len(in_scope),
            **block_metrics(in_scope),
        })
    return pd.DataFrame.from_records(records, columns=REPORT_COLUMNS)
