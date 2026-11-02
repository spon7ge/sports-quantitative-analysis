"""Quantile model fitting and CV / holdout orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from models.shared.metrics import DEFAULT_MIN_TIERS, score_quantile_fold
from models.shared.splits import (
    date_walk_forward_folds,
    early_season_mask,
    regular_season_mask,
)

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
    regular_mask: np.ndarray | None = None,
    X_predict: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Train one XGB quantile model per quantile; return models + preds.

    ``X_predict`` is scored after fitting. It is not an early-stopping
    set and it is not used to impute. With ``train_tail``, the round
    count comes from the latest regular-season dates, then each model
    is refit on the full training window for that many trees.
    """
    refit = False
    es_fit_mask: np.ndarray | None = None
    if early_stop == "train_tail":
        if train_dates is None:
            raise ValueError("train_tail requires train_dates")
        dates = pd.to_datetime(np.asarray(train_dates))
        if len(dates) != len(X_train):
            raise ValueError("train_dates must align with X_train rows")
        if regular_mask is None:
            eligible_rows = np.ones(len(dates), dtype=bool)
        else:
            eligible_rows = np.asarray(regular_mask, dtype=bool)
            if eligible_rows.shape != (len(dates),):
                raise ValueError("regular_mask must align with X_train rows")
        if not bool(eligible_rows.any()):
            raise ValueError("train_tail requires a regular-season row")
        eligible_dates = np.sort(pd.to_datetime(pd.unique(dates[eligible_rows])))
        n_unique = len(eligible_dates)
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
        tail_start = eligible_dates[-n_stop]
        stop_mask = eligible_rows & (dates >= tail_start)
        # Rows on or after the tail, including a postseason that follows
        # it, are not used to grow the trees that pick the round count.
        es_fit_mask = np.asarray(dates < tail_start, dtype=bool)
        if not bool(es_fit_mask.any()) or not bool(stop_mask.any()):
            raise ValueError("train_tail split is empty")
        print(
            "  Non-regular rows excluded from the stopping tail: "
            f"{int((~eligible_rows).sum())}"
        )
        def _take(frame, mask):
            if hasattr(frame, "iloc"):
                return frame.iloc[mask]
            return frame[mask]

        X_fit = _take(X_train, es_fit_mask)
        y_fit = _take(y_train, es_fit_mask)
        X_es = _take(X_train, stop_mask)
        y_es = _take(y_train, stop_mask)
        predict_X = X_val if X_predict is None else X_predict
        refit = True
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
        stopped = XGBRegressor(**params, quantile_alpha=q)
        stopped.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        key = f"q_{q:.2f}"
        model = stopped
        if refit:
            best = getattr(stopped, "best_iteration", None)
            if best is None or es_fit_mask is None:
                raise RuntimeError(f"{key} early stopping did not set best_iteration")
            n_fit = int(np.asarray(es_fit_mask).sum())
            n_trees = max(int((int(best) + 1) * len(X_train) / n_fit), 1)
            final_params = dict(params)
            final_params.pop("early_stopping_rounds", None)
            final_params["n_estimators"] = n_trees
            model = XGBRegressor(**final_params, quantile_alpha=q)
            model.fit(X_train, y_train)
            model.early_stop_iteration = int(best)
            model.stopping_rows = int(np.asarray(stop_mask).sum())
        models[key] = model
        preds[key] = model.predict(predict_X)
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
    print("── Phase 1: TimeSeriesSplit (XGBoost) ──────────────────────────")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        starting_val = train_df[role_col].iloc[val_idx].values
        models, preds = fit_quantile_models(
            X_tr, y_tr, X_val, y_val,
            quantiles=quantiles,
            xgb_params=xgb_params,
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
    covs = [r.get("coverage_80pct", float("nan")) for r in results]
    print("\nTimeSeriesSplit Summary (XGBoost)")
    print(f"  Pinball q50 : {np.mean(pinballs):.3f} ± {np.std(pinballs):.3f}")
    if np.any(~np.isnan(covs)):
        print(
            f"  Coverage 80%: {np.nanmean(covs):.1%} ± {np.nanstd(covs):.1%}  "
            "(target 80%)"
        )
    else:
        print("  Coverage 80%: n/a (q10/q90 not fit)")
    return results


def tune_xgb_quantile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 40,
    n_splits: int = 4,
    quantile_alpha: float = 0.50,
    quantiles: Sequence[float] | None = None,
    seed: int = 42,
    fixed_params: dict | None = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Optuna-tune one XGB quantile model via TimeSeriesSplit pinball loss.

    Pass ``quantiles`` to score the mean pinball across the grid. A single
    ``quantile_alpha`` is kept for a one-knot search. Use train-pool ``X``
    / ``y`` only — never the locked holdout. Returns merged ``best_params``
    (ready for ``XGB_PARAMS``), ``best_value``, and the Optuna ``study``.
    """
    import optuna

    from models.shared.metrics import pinball_loss
    from models.shared.quantile_offsets import mean_pinball

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
        if quantiles is None:
            levels = [float(quantile_alpha)]
        else:
            levels = [float(level) for level in quantiles]
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            columns = []
            for level in levels:
                m = XGBRegressor(**params, quantile_alpha=level)
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                columns.append(m.predict(X_val))
            if len(levels) == 1:
                losses.append(pinball_loss(y_val, columns[0], levels[0]))
            else:
                losses.append(
                    mean_pinball(y_val, np.column_stack(columns), np.asarray(levels))
                )
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
    target_col: str = "minutes",
    folds: Sequence[Mapping[str, Any]] | None = None,
    early_team_games: int | None = None,
) -> dict[str, Any]:
    """Phase 2 — walk-forward validation (production simulation).

    ``folds`` overrides the date-percentage windows. Season folds from
    ``season_walk_forward_folds`` use the same dict. ``early_team_games``
    scores the first team games of each validation season a second time.
    """
    outcome_col = str(target_col)
    if not outcome_col:
        raise ValueError("target_col")
    print("\n── Phase 2: Walk-Forward Validation (XGBoost) ─────────────────")
    if not train_df["game_date"].is_monotonic_increasing:
        raise ValueError("train_df must be sorted by game_date")

    tier_map = dict(DEFAULT_MIN_TIERS if tiers is None else tiers)
    if folds is None:
        unique_dates = np.sort(pd.to_datetime(train_df["game_date"]).unique())
        n_dates = len(unique_dates)
        train_window = round(n_dates * train_frac)
        step_size = round(n_dates * step_frac)
        print(f"Unique game dates in pool : {n_dates}")
        window_share = train_window / n_dates
        step_share = step_size / n_dates
        print(f"Training window           : {train_window} dates (~{window_share:.0%})")
        print(f"Step size                 : {step_size} dates (~{step_share:.0%})")
        fold_list = list(
            date_walk_forward_folds(
                train_df,
                train_frac=train_frac,
                step_frac=step_frac,
                n_folds=n_folds,
            )
        )
    else:
        fold_list = list(folds)
        n_folds = len(fold_list)
        print(f"Supplied folds            : {n_folds}")

    wf_results: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_ranges: dict[int, dict[str, Any]] = {}
    models_last = preds_last = X_val_last = y_val_last = starting_last = None
    early_rows = (
        None
        if early_team_games is None
        else early_season_mask(train_df, n_games=int(early_team_games)).to_numpy()
    )

    for fold_info in fold_list:
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
            "early_stop": early_stop,
            "train_tail_frac": train_tail_frac,
        }
        if early_stop == "train_tail":
            fit_kwargs["train_dates"] = train_df.loc[train_mask, "game_date"]
            fit_kwargs["X_predict"] = X_val
            if "season_type" in train_df.columns:
                fit_kwargs["regular_mask"] = regular_season_mask(
                    train_df.loc[train_mask, "season_type"]
                )

        models, preds = fit_quantile_models(
            X_tr, y_tr, X_val, y_val,
            xgb_params=xgb_params,
            **fit_kwargs,
        )
        metrics = score_quantile_fold(
            y_val.values, preds,
            fold_label=fold_info["label"],
            starting=starting_val,
            models=models,
            tiers=tier_map,
        )
        if early_rows is not None:
            on_val = np.asarray(val_mask, dtype=bool)
            chosen = np.asarray(early_rows, dtype=bool)[on_val]
            if bool(chosen.any()):
                early_metrics = score_quantile_fold(
                    np.asarray(y_val)[chosen],
                    {key: np.asarray(value)[chosen] for key, value in preds.items()},
                    fold_label=(
                        f"{fold_info['label']} | first {early_team_games} team games"
                    ),
                    starting=np.asarray(starting_val)[chosen],
                    tiers=tier_map,
                )
                metrics["early_n"] = early_metrics["n"]
                metrics["early_mean_pinball"] = early_metrics["mean_pinball"]
                metrics["early_pinball"] = early_metrics["pinball"]
                metrics["early_coverage_80pct"] = early_metrics.get(
                    "coverage_80pct", float("nan")
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
        fold_oof[outcome_col] = y_val.to_numpy()
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
        "folds": fold_list,
    }
    if not wf_results:
        print("No walk-forward folds produced.")
        return empty_return

    pinballs = [r["pinball"] for r in wf_results]
    means = [r.get("mean_pinball", float("nan")) for r in wf_results]
    covs = [r.get("coverage_80pct", float("nan")) for r in wf_results]
    cov_mean = float(np.nanmean(covs)) if np.any(~np.isnan(covs)) else float("nan")
    cov_std = float(np.nanstd(covs)) if np.any(~np.isnan(covs)) else float("nan")

    print(f"\n{'─' * 60}")
    print(f"Walk-Forward Summary ({len(wf_results)}/{n_folds} folds)")
    print(f"  Mean pinball  : {np.mean(means):.3f} ± {np.std(means):.3f}")
    print(f"  Pinball q50   : {np.mean(pinballs):.3f} ± {np.std(pinballs):.3f}")
    if np.isnan(cov_mean):
        print("  Coverage 80%: n/a (q10/q90 not fit)")
    else:
        print(f"  Coverage 80%: {cov_mean:.1%} ± {cov_std:.1%}  (target 80%)")

    print(f"\n{'─' * 60}")
    print("Per-fold mean pinball, q50 pinball, and 80% coverage:")
    print(
        f"  {'Fold':<45} {'Mean':>10}  {'q50':>10}  {'Coverage':>10}  "
        f"{'Starters':>10}  {'Bench':>10}"
    )
    for r in wf_results:
        s_cov = r.get("coverage_Starters", float("nan"))
        b_cov = r.get("coverage_Bench", float("nan"))
        cov80 = r.get("coverage_80pct", float("nan"))
        print(
            f"  {r['fold']:<45} {r.get('mean_pinball', float('nan')):>10.3f}  "
            f"{r['pinball']:>10.3f}  "
            f"{cov80:>9.1%}  "
            f"{s_cov:>9.1%}  {b_cov:>9.1%}"
        )
    if any("early_n" in r for r in wf_results):
        print(f"\nFirst {early_team_games} team games of each validation season:")
        for r in wf_results:
            if "early_n" not in r:
                continue
            early_cov = r.get("early_coverage_80pct", float("nan"))
            print(
                f"  {r['fold']:<45} n={r['early_n']:5d}  "
                f"mean pinball {r['early_mean_pinball']:.3f}  "
                f"80% coverage {early_cov:.1%}"
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

    max_pinball = max(means) if means else 1.0
    print("\nMean pinball over folds (stable = good, rising = model aging):")
    for r in wf_results:
        value = r.get("mean_pinball", float("nan"))
        bar = "█" * int((value / max_pinball) * 20) if max_pinball else ""
        print(f"  {r['fold']:<45} {value:.3f}  {bar}")

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
        "folds": fold_list,
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
    train_tail_frac: float = 0.10,
) -> dict[str, Any]:
    """Blind holdout evaluation on the held-out season.

    The round count comes from the last ``train_tail_frac`` of
    regular-season training dates. The model is then refit on the full
    window. The holdout is predicted and is not the stopping set.
    """
    features = list(features)
    print(f"── {fold_label} ─────────────────────────────────────────────────")
    print(
        "  Round count from the last "
        f"{train_tail_frac:.0%} of regular-season training dates; "
        "refit on the full window; predict holdout (never in eval_set)."
    )

    X_train_full = train_df[features]
    y_train_full = train_df[target_col]
    X_ho = holdout_df[features]
    y_ho = holdout_df[target_col]
    starting_ho = holdout_df[role_col].values

    holdout_kwargs: dict[str, Any] = {}
    if "season_type" in train_df.columns:
        holdout_kwargs["regular_mask"] = regular_season_mask(train_df["season_type"])
    models_ho, pred_frame = fit_quantile_models(
        X_train_full,
        y_train_full,
        X_train_full,
        y_train_full,
        quantiles=quantiles,
        xgb_params=xgb_params,
        early_stop="train_tail",
        train_dates=train_df["game_date"],
        train_tail_frac=train_tail_frac,
        X_predict=X_ho,
        **holdout_kwargs,
    )
    preds_ho = {
        key: np.asarray(value, dtype=float) for key, value in pred_frame.items()
    }

    ho_metrics = score_quantile_fold(
        y_ho.values, preds_ho,
        fold_label=fold_label,
        starting=starting_ho,
        models=models_ho,
        tiers=tiers,
    )

    early_cols = {"team_id", "game_id", "season_year"}
    if early_cols.issubset(holdout_df.columns):
        early = early_season_mask(holdout_df, n_games=10).to_numpy()
        if bool(early.any()):
            early_metrics = score_quantile_fold(
                y_ho.to_numpy()[early],
                {key: np.asarray(value)[early] for key, value in preds_ho.items()},
                fold_label=f"{fold_label} | first 10 team games",
                starting=np.asarray(starting_ho)[early],
                tiers=tiers,
            )
            ho_metrics["early_n"] = early_metrics["n"]
            ho_metrics["early_mean_pinball"] = early_metrics["mean_pinball"]
            ho_metrics["early_pinball"] = early_metrics["pinball"]
            ho_metrics["early_coverage_80pct"] = early_metrics.get(
                "coverage_80pct", float("nan")
            )

    if wf_results:
        wf_mean = float(np.mean([r.get("mean_pinball", np.nan) for r in wf_results]))
        wf_pinball = float(np.mean([r["pinball"] for r in wf_results]))
        wf_coverage = float(np.mean([r["coverage_80pct"] for r in wf_results]))
        ho_mean = ho_metrics["mean_pinball"]
        ho_pinball = ho_metrics["pinball"]
        ho_coverage = ho_metrics["coverage_80pct"]
        pinball_gap = ho_pinball - wf_pinball
        print(f"\n{'─' * 55}")
        print(f"  Walk-forward mean pinball       : {wf_mean:.3f}")
        print(f"  Holdout mean pinball            : {ho_mean:.3f}")
        print(f"  Walk-forward pinball q50 (mean) : {wf_pinball:.3f}")
        print(f"  Holdout pinball q50             : {ho_pinball:.3f}")
        print(
            f"  q50 pinball gap                 : {pinball_gap:+.3f}  "
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
