"""NGBoost minutes regression with the quantile prediction contract.

One fitted distribution yields every quantile. Callers still receive
``q_0.05``-style models whose ``predict`` reads that quantile, so
walk-forward scoring and ``save_model_bundle`` stay the same as XGBoost.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext, redirect_stdout
from typing import Any

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from sklearn.model_selection import TimeSeriesSplit

from models.shared.metrics import DEFAULT_MIN_TIERS, score_quantile_fold
from models.shared.splits import date_walk_forward_folds
from models.shared.train import DEFAULT_QUANTILES


class QuantileNGBoost:
    """Predict one quantile from a shared NGBoost distribution."""

    def __init__(self, model: NGBRegressor, alpha: float):
        self.model = model
        self.alpha = alpha
        best = getattr(model, "best_val_loss_itr", None)
        self.best_iteration = None if best is None else int(best)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        max_iter = None if self.best_iteration is None else self.best_iteration + 1
        dist = self.model.pred_dist(X, max_iter=max_iter)
        return np.asarray(dist.ppf(self.alpha), dtype=float)


def _fit_frames(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    early_stop: str,
    train_dates: pd.Series | np.ndarray | Sequence | None,
    train_tail_frac: float,
    X_predict: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    if early_stop == "train_tail":
        if train_dates is None:
            raise ValueError("train_tail requires train_dates")
        dates = pd.to_datetime(np.asarray(train_dates))
        if len(dates) != len(X_train):
            raise ValueError("train_dates must align with X_train rows")
        unique_dates = pd.unique(dates)
        n_unique = len(unique_dates)
        if n_unique < 2:
            raise ValueError("train_tail requires at least two training dates")
        n_stop = int(np.floor(n_unique * train_tail_frac))
        if n_stop < 1:
            n_stop = 1
        if n_stop >= n_unique:
            raise ValueError(
                f"train_tail n_stop={n_stop} must be strictly less than "
                f"n_unique={n_unique}"
            )
        stop_dates = set(unique_dates[-n_stop:])
        position = pd.Series(dates)
        fit_mask = ~position.isin(stop_dates).to_numpy()
        stop_mask = position.isin(stop_dates).to_numpy()
        X_fit = X_train.iloc[fit_mask]
        y_fit = y_train.iloc[fit_mask]
        X_es = X_train.iloc[stop_mask]
        y_es = y_train.iloc[stop_mask]
        predict_X = X_val if X_predict is None else X_predict
        return X_fit, y_fit, X_es, y_es, predict_X
    if early_stop == "validation":
        predict_X = X_val if X_predict is None else X_predict
        return X_train, y_train, X_val, y_val, predict_X
    raise ValueError(f"unknown early_stop={early_stop!r}")


def fit_ngboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    quantiles: Sequence[float] | None = None,
    ngb_params: dict | None = None,
    early_stop: str = "validation",
    train_dates: pd.Series | np.ndarray | Sequence | None = None,
    train_tail_frac: float = 0.10,
    X_predict: pd.DataFrame | None = None,
) -> tuple[dict[str, QuantileNGBoost], dict[str, np.ndarray]]:
    """Fit one Normal NGBoost model and read each quantile from it."""
    if ngb_params is None:
        raise ValueError("ngb_params is required (define in the prop notebook)")
    X_fit, y_fit, X_es, y_es, predict_X = _fit_frames(
        X_train,
        y_train,
        X_val,
        y_val,
        early_stop=early_stop,
        train_dates=train_dates,
        train_tail_frac=train_tail_frac,
        X_predict=X_predict,
    )
    params = dict(ngb_params)
    rounds = params.pop("early_stopping_rounds", None)
    params.setdefault("verbose", False)
    model = NGBRegressor(**params, early_stopping_rounds=rounds)
    # NGBoost prints one validation line even when verbose is False.
    quiet = nullcontext() if params["verbose"] else redirect_stdout(io.StringIO())
    with quiet:
        model.fit(X_fit, y_fit, X_val=X_es, Y_val=y_es, early_stopping_rounds=rounds)

    levels = list(quantiles or DEFAULT_QUANTILES)
    max_iter = None
    if model.best_val_loss_itr is not None:
        max_iter = int(model.best_val_loss_itr) + 1
    dist = model.pred_dist(predict_X, max_iter=max_iter)
    models: dict[str, QuantileNGBoost] = {}
    preds: dict[str, np.ndarray] = {}
    for alpha in levels:
        key = f"q_{alpha:.2f}"
        models[key] = QuantileNGBoost(model, alpha)
        preds[key] = np.asarray(dist.ppf(alpha), dtype=float)
    return models, preds


def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    *,
    ngb_params: dict,
    role_col: str = "starting",
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    n_splits: int = 5,
    quantiles: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Phase 1 — TimeSeriesSplit, same row cuts as the XGBoost path."""
    print("── Phase 1: TimeSeriesSplit (NGBoost) ──────────────────────────")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        models, preds = fit_ngboost(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[val_idx],
            y.iloc[val_idx],
            quantiles=quantiles,
            ngb_params=ngb_params,
        )
        results.append(
            score_quantile_fold(
                y.iloc[val_idx].to_numpy(),
                preds,
                fold_label=f"TSCV fold {fold + 1}",
                starting=train_df[role_col].iloc[val_idx].to_numpy(),
                models=models,
                tiers=tiers,
            )
        )
    pinballs = [row["pinball"] for row in results]
    print("\nTimeSeriesSplit Summary (NGBoost)")
    print(f"  Pinball q50 : {np.mean(pinballs):.3f} ± {np.std(pinballs):.3f}")
    return results


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    *,
    ngb_params: dict,
    role_col: str = "starting",
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    n_folds: int = 4,
    train_frac: float = 0.50,
    step_frac: float = 0.10,
    quantiles: Sequence[float] | None = None,
    early_stop: str = "train_tail",
    train_tail_frac: float = 0.10,
) -> dict[str, Any]:
    """Phase 2 — date walk-forward. Same folds as ``train.run_walk_forward``."""
    print("\n── Phase 2: Walk-Forward Validation (NGBoost, date-based) ──────")
    if not train_df["game_date"].is_monotonic_increasing:
        raise ValueError("train_df must be sorted by game_date")

    unique_dates = train_df["game_date"].unique()
    n_dates = len(unique_dates)
    train_window = round(n_dates * train_frac)
    step_size = round(n_dates * step_frac)
    tier_map = dict(DEFAULT_MIN_TIERS if tiers is None else tiers)
    print(f"Unique game dates in pool : {n_dates}")
    print(
        f"Training window           : {train_window} dates "
        f"(~{train_window / n_dates:.0%})"
    )
    print(f"Step size                 : {step_size} dates (~{step_size / n_dates:.0%})")

    wf_results: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_ranges: dict[int, dict[str, Any]] = {}
    models_last = preds_last = X_val_last = y_val_last = starting_last = None

    for fold_info in date_walk_forward_folds(
        train_df,
        train_frac=train_frac,
        step_frac=step_frac,
        n_folds=n_folds,
    ):
        train_mask = fold_info["train_mask"]
        val_mask = fold_info["val_mask"]
        train_dates = fold_info["train_dates"]
        val_dates = fold_info["val_dates"]
        fold_id = fold_info["fold"]
        X_tr, X_val = X[train_mask], X[val_mask]
        y_tr, y_val = y[train_mask], y[val_mask]
        starting_val = train_df[role_col][val_mask].to_numpy()

        fit_kwargs: dict[str, Any] = {
            "quantiles": quantiles,
            "ngb_params": ngb_params,
            "early_stop": early_stop,
            "train_tail_frac": train_tail_frac,
        }
        if early_stop == "train_tail":
            fit_kwargs["train_dates"] = train_df.loc[train_mask, "game_date"]
            fit_kwargs["X_predict"] = X_val

        models, preds = fit_ngboost(X_tr, y_tr, X_val, y_val, **fit_kwargs)
        metrics = score_quantile_fold(
            y_val.to_numpy(),
            preds,
            fold_label=fold_info["label"],
            starting=starting_val,
            models=models,
            tiers=tier_map,
        )
        metrics.update(
            {
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "val_end": val_dates[-1],
                "train_mask": train_mask,
                "val_mask": val_mask,
            }
        )
        wf_results.append(metrics)

        fold_oof = pd.DataFrame({key: preds[key] for key in preds}, index=X_val.index)
        fold_oof["fold_id"] = fold_id
        fold_oof["game_date"] = train_df.loc[val_mask, "game_date"].to_numpy()
        fold_oof["minutes"] = y_val.to_numpy()
        fold_oof["starting"] = starting_val
        fold_oof["early_stop"] = early_stop
        fold_oof["train_tail_frac"] = train_tail_frac
        oof_parts.append(fold_oof)
        fold_ranges[fold_id] = {
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_start": val_dates[0],
            "val_end": val_dates[-1],
        }
        models_last = models
        preds_last = preds
        X_val_last = X_val
        y_val_last = y_val
        starting_last = starting_val

    empty = {
        "wf_results": [],
        "models_last": None,
        "preds_last": None,
        "X_val_last": None,
        "y_val_last": None,
        "starting_last": None,
        "last_fold": None,
        "oof": pd.DataFrame(),
        "fold_ranges": {},
    }
    if not wf_results:
        print("No walk-forward folds produced.")
        return empty

    pinballs = [row["pinball"] for row in wf_results]
    print(f"\n{'─' * 60}")
    print(f"Walk-Forward Summary ({len(wf_results)}/{n_folds} folds)")
    print(f"  Pinball q50 : {np.mean(pinballs):.3f} ± {np.std(pinballs):.3f}")
    print(f"\n{'─' * 60}")
    print("Per-fold pinball (q50) and 80% coverage:")
    for row in wf_results:
        print(
            f"  {row['fold']:<45} {row['pinball']:>10.3f}  "
            f"{row.get('coverage_80pct', float('nan')):>9.1%}"
        )

    oof = pd.concat(oof_parts, axis=0) if oof_parts else pd.DataFrame()
    return {
        "wf_results": wf_results,
        "models_last": models_last,
        "preds_last": preds_last,
        "X_val_last": X_val_last,
        "y_val_last": y_val_last,
        "starting_last": starting_last,
        "last_fold": wf_results[-1],
        "oof": oof,
        "fold_ranges": fold_ranges,
    }


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    ngb_params: dict,
    role_col: str = "starting",
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    wf_results: list[dict[str, Any]] | None = None,
    quantiles: Sequence[float] | None = None,
    fold_label: str = "Blind Holdout",
    es_frac: float = 0.90,
) -> dict[str, Any]:
    """Blind holdout. Early-stopping rows stay inside the train pool."""
    features = list(features)
    print(f"── {fold_label} ─────────────────────────────────────────────────")
    print(
        f"  Train pool: {es_frac:.0%} rows for fit, remainder for early-stopping only; "
        "predict holdout (never in eval_set)."
    )
    X_train_full = train_df[features]
    y_train_full = train_df[target_col]
    X_ho = holdout_df[features]
    y_ho = holdout_df[target_col]
    cutoff = int(len(X_train_full) * es_frac)
    models_ho, _ = fit_ngboost(
        X_train_full.iloc[:cutoff],
        y_train_full.iloc[:cutoff],
        X_train_full.iloc[cutoff:],
        y_train_full.iloc[cutoff:],
        quantiles=quantiles,
        ngb_params=ngb_params,
    )
    preds_ho = {key: models_ho[key].predict(X_ho) for key in models_ho}
    ho_metrics = score_quantile_fold(
        y_ho.to_numpy(),
        preds_ho,
        fold_label=fold_label,
        starting=holdout_df[role_col].to_numpy(),
        models=models_ho,
        tiers=tiers,
    )
    if wf_results:
        wf_pinball = float(np.mean([row["pinball"] for row in wf_results]))
        ho_pinball = ho_metrics["pinball"]
        print(f"\n{'─' * 55}")
        print(f"  Walk-forward pinball q50 (mean) : {wf_pinball:.3f}")
        print(f"  Holdout pinball q50             : {ho_pinball:.3f}")
        gap = ho_pinball - wf_pinball
        print(
            f"  Pinball gap                     : {gap:+.3f}  "
            f"{'⚠ investigate' if abs(gap) > 0.25 else '✓ acceptable'}"
        )
    return {
        "models_ho": models_ho,
        "preds_ho": preds_ho,
        "ho_metrics": ho_metrics,
        "X_ho": X_ho,
        "y_ho": y_ho,
    }
