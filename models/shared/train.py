"""Quantile model fitting and CV / holdout orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from models.shared.metrics import DEFAULT_MIN_TIERS, score_quantile_fold
from models.shared.splits import date_walk_forward_folds

DEFAULT_QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    quantiles: Sequence[float] | None = None,
    xgb_params: dict | None = None,
    early_stop: str = "validation",
    train_dates: pd.Series | np.ndarray | Sequence | None = None,
    train_tail_frac: float = 0.10,
    X_predict: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Train one XGB quantile model per quantile; return models + preds."""
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
        fit_mask = ~pd.Series(dates).isin(stop_dates).to_numpy()
        stop_mask = pd.Series(dates).isin(stop_dates).to_numpy()
        X_fit = X_train.iloc[fit_mask] if hasattr(X_train, "iloc") else X_train[fit_mask]
        y_fit = y_train.iloc[fit_mask] if hasattr(y_train, "iloc") else y_train[fit_mask]
        X_es = X_train.iloc[stop_mask] if hasattr(X_train, "iloc") else X_train[stop_mask]
        y_es = y_train.iloc[stop_mask] if hasattr(y_train, "iloc") else y_train[stop_mask]
        predict_X = X_val if X_predict is None else X_predict
    elif early_stop == "validation":
        X_fit, y_fit = X_train, y_train
        X_es, y_es = X_val, y_val
        predict_X = X_val if X_predict is None else X_predict
    else:
        raise ValueError(f"unknown early_stop={early_stop!r}")

    if xgb_params is None:
        raise ValueError("xgb_params is required (define in the prop notebook)")
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    params = dict(xgb_params)
    models: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for q in quantiles:
        m = XGBRegressor(**params, quantile_alpha=q)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        key = f"q_{q:.2f}"
        models[key] = m
        preds[key] = m.predict(predict_X)
    return models, preds


def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    *,
    xgb_params: dict,
    role_col: str = "starting",
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    n_splits: int = 5,
    quantiles: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Phase 1 — TimeSeriesSplit for feature / hyperparam comparison."""
    print("── Phase 1: TimeSeriesSplit ─────────────────────────────────────────")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        starting_val = train_df[role_col].iloc[val_idx].values
        models, preds = fit_quantile_models(
            X_tr, y_tr, X_val, y_val,
            quantiles=quantiles, xgb_params=xgb_params,
        )
        metrics = score_quantile_fold(
            y_val.values, preds,
            fold_label=f"TSCV fold {fold + 1}",
            starting=starting_val,
            models=models,
            tiers=tiers,
        )
        results.append(metrics)

    pinballs = [r["pinball"] for r in results]
    covs = [r["coverage_80pct"] for r in results]
    print("\nTimeSeriesSplit Summary")
    print(f"  Pinball q50 : {np.mean(pinballs):.3f} ± {np.std(pinballs):.3f}")
    print(f"  Coverage 80%: {np.mean(covs):.1%} ± {np.std(covs):.1%}  (target 80%)")
    return results


def tune_xgb_quantile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 40,
    n_splits: int = 4,
    quantile_alpha: float = 0.50,
    seed: int = 42,
    fixed_params: dict | None = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Optuna-tune one XGB quantile model via TimeSeriesSplit pinball loss.

    Use train-pool ``X`` / ``y`` only — never the locked holdout.
    Returns merged ``best_params`` (ready for ``XGB_PARAMS``), ``best_value``,
    and the Optuna ``study``.
    """
    import optuna

    from models.shared.metrics import pinball_loss

    fixed = {
        "objective": "reg:quantileerror",
        "n_jobs": -1,
        "random_state": seed,
        "early_stopping_rounds": 50,
        **(fixed_params or {}),
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: Any) -> float:
        params = {
            **fixed,
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        losses: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            m = XGBRegressor(**params, quantile_alpha=quantile_alpha)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            losses.append(pinball_loss(y_val, m.predict(X_val), quantile_alpha))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best_params = {**fixed, **study.best_params}
    print(f"Best pinball (mean TSCV, q={quantile_alpha}): {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "study": study,
    }


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    *,
    xgb_params: dict,
    role_col: str = "starting",
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    n_folds: int = 4,
    train_frac: float = 0.50,
    step_frac: float = 0.10,
    quantiles: Sequence[float] | None = None,
    early_stop: str = "train_tail",
    train_tail_frac: float = 0.10,
) -> dict[str, Any]:
    """Phase 2 — date-based walk-forward validation (production simulation)."""
    print("\n── Phase 2: Walk-Forward Validation (date-based) ───────────────────")
    if not train_df["game_date"].is_monotonic_increasing:
        raise ValueError("train_df must be sorted by game_date")

    unique_dates = train_df["game_date"].unique()
    n_dates = len(unique_dates)
    train_window = round(n_dates * train_frac)
    step_size = round(n_dates * step_frac)
    tier_map = dict(DEFAULT_MIN_TIERS if tiers is None else tiers)

    print(f"Unique game dates in pool : {n_dates}")
    print(f"Training window           : {train_window} dates (~{train_window / n_dates:.0%})")
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
        starting_val = train_df[role_col][val_mask].values

        fit_kwargs: dict[str, Any] = {
            "quantiles": quantiles,
            "xgb_params": xgb_params,
            "early_stop": early_stop,
            "train_tail_frac": train_tail_frac,
        }
        if early_stop == "train_tail":
            fit_kwargs["train_dates"] = train_df.loc[train_mask, "game_date"]
            fit_kwargs["X_predict"] = X_val

        models, preds = fit_quantile_models(
            X_tr, y_tr, X_val, y_val,
            **fit_kwargs,
        )
        metrics = score_quantile_fold(
            y_val.values, preds,
            fold_label=fold_info["label"],
            starting=starting_val,
            models=models,
            tiers=tier_map,
        )
        metrics.update({
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_end": val_dates[-1],
            "train_mask": train_mask,
            "val_mask": val_mask,
        })
        wf_results.append(metrics)

        fold_oof = pd.DataFrame({k: preds[k] for k in preds}, index=X_val.index)
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

    empty_return = {
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
        return empty_return

    pinballs = [r["pinball"] for r in wf_results]
    covs = [r.get("coverage_80pct", float("nan")) for r in wf_results]
    cov_mean = float(np.nanmean(covs)) if np.any(~np.isnan(covs)) else float("nan")
    cov_std = float(np.nanstd(covs)) if np.any(~np.isnan(covs)) else float("nan")

    print(f"\n{'─' * 60}")
    print(f"Walk-Forward Summary ({len(wf_results)}/{n_folds} folds)")
    print(f"  Pinball q50 : {np.mean(pinballs):.3f} ± {np.std(pinballs):.3f}")
    if np.isnan(cov_mean):
        print("  Coverage 80%: n/a (q10/q90 not fit)")
    else:
        print(f"  Coverage 80%: {cov_mean:.1%} ± {cov_std:.1%}  (target 80%)")

    print(f"\n{'─' * 60}")
    print("Per-fold pinball (q50) and 80% coverage:")
    print(f"  {'Fold':<45} {'Pinball':>10}  {'Coverage':>10}  {'Starters':>10}  {'Bench':>10}")
    for r in wf_results:
        s_cov = r.get("coverage_Starters", float("nan"))
        b_cov = r.get("coverage_Bench", float("nan"))
        cov80 = r.get("coverage_80pct", float("nan"))
        print(
            f"  {r['fold']:<45} {r['pinball']:>10.3f}  "
            f"{cov80:>9.1%}  "
            f"{s_cov:>9.1%}  {b_cov:>9.1%}"
        )

    print("\nPer-fold pinball (q50) by tier:")
    tier_keys = list(tier_map.keys())
    header = f"  {'Fold':<45} " + "  ".join(f"{t:>10}" for t in tier_keys)
    print(header)
    for r in wf_results:
        row = f"  {r['fold']:<45} "
        row += "  ".join(
            f"{r.get(f'pinball_{t}', float('nan')):>10.3f}" for t in tier_keys
        )
        print(row)

    max_pinball = max(pinballs) if pinballs else 1.0
    print("\nPinball q50 over folds (stable = good, rising = model aging):")
    for r in wf_results:
        bar = "█" * int((r["pinball"] / max_pinball) * 20) if max_pinball else ""
        print(f"  {r['fold']:<45} {r['pinball']:.3f}  {bar}")

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
    xgb_params: dict,
    role_col: str = "starting",
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    wf_results: list[dict[str, Any]] | None = None,
    quantiles: Sequence[float] | None = None,
    fold_label: str = "Blind Holdout",
    es_frac: float = 0.90,
) -> dict[str, Any]:
    """Blind holdout evaluation on the held-out season."""
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
    starting_ho = holdout_df[role_col].values

    es_cutoff = int(len(X_train_full) * es_frac)
    X_es_train = X_train_full.iloc[:es_cutoff]
    y_es_train = y_train_full.iloc[:es_cutoff]
    X_es_val = X_train_full.iloc[es_cutoff:]
    y_es_val = y_train_full.iloc[es_cutoff:]

    models_ho, _ = fit_quantile_models(
        X_es_train, y_es_train, X_es_val, y_es_val,
        quantiles=quantiles, xgb_params=xgb_params,
    )
    preds_ho = {k: models_ho[k].predict(X_ho) for k in models_ho}

    ho_metrics = score_quantile_fold(
        y_ho.values, preds_ho,
        fold_label=fold_label,
        starting=starting_ho,
        models=models_ho,
        tiers=tiers,
    )

    if wf_results:
        wf_pinball = float(np.mean([r["pinball"] for r in wf_results]))
        wf_coverage = float(np.mean([r["coverage_80pct"] for r in wf_results]))
        ho_pinball = ho_metrics["pinball"]
        ho_coverage = ho_metrics["coverage_80pct"]
        pinball_gap = ho_pinball - wf_pinball
        print(f"\n{'─' * 55}")
        print(f"  Walk-forward pinball q50 (mean) : {wf_pinball:.3f}")
        print(f"  Holdout pinball q50             : {ho_pinball:.3f}")
        print(
            f"  Pinball gap                     : {pinball_gap:+.3f}  "
            f"{'⚠ investigate' if abs(pinball_gap) > 0.25 else '✓ acceptable'}"
        )
        print(f"  Walk-forward 80% coverage (mean): {wf_coverage:.1%}")
        print(f"  Holdout 80% coverage            : {ho_coverage:.1%}")

    return {
        "models_ho": models_ho,
        "preds_ho": preds_ho,
        "ho_metrics": ho_metrics,
        "X_ho": X_ho,
        "y_ho": y_ho,
    }
