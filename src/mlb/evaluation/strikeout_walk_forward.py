"""Expanding 28-day block walk that stores each pregame feature row once."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlb.evaluation.walk_forward_blocks import assign_walk_forward_blocks
from src.mlb.evaluation.walk_forward_fit import (
    UnestimatedDispersion,
    fit_walk_forward_nb2,
    predict_walk_forward_mean,
)
from src.mlb.models.pmf import negative_binomial_pmf
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
