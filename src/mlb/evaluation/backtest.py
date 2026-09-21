"""Chronological strikeout backtest: model, baselines, and optional market compare."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.evaluation.baselines import fit_baselines, prediction_frame_from_mu
from src.mlb.evaluation.folds import chronological_folds
from src.mlb.evaluation.market import compare_market
from src.mlb.markets.quotes import select_quotes_asof
from src.mlb.models.calibration import (
    calibration_slope_intercept,
    randomized_pit,
)
from src.mlb.models.metrics import discrete_crps, pmf_nll
from src.mlb.models.preprocess import prepare_strikeout_frame
from src.mlb.schemas import FEATURE_ROW_COLUMNS, PMF_COLUMNS

LOGGER = logging.getLogger(__name__)

REQUIRED_PANEL_FEATURES: tuple[str, ...] = (
    "bf_mean_5",
    "bf_sd_5",
    "early_exit_rate_5",
    "k_bf_shrunk_365",
    "k_bf_shrunk_60",
    "rest_days",
    "pitcher_throws_L",
    "is_home",
)
_CONTEXT_COLUMNS: tuple[str, ...] = ("is_home", "season", "venue_id")

_PIPELINE_MODULES = (
    "src.mlb.pipeline",
    "src.mlb.pipeline.features",
    "src.mlb.pipeline.build",
    "src.mlb.pipeline.ingest",
    "src.mlb.pipeline.gamelog_features",
)
_MODEL_MODULES = (
    "src.mlb.models",
    "src.mlb.models.strikeouts",
    "src.mlb.models.workload",
    "src.mlb.models.pmf",
    "src.mlb.models.scoring",
    "src.mlb.models.nb",
)


def _load_symbol(name: str, modules: tuple[str, ...]) -> Callable | None:
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _pmf_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in PMF_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"predictions missing PMF columns: {missing}")
    return frame.loc[:, list(PMF_COLUMNS)].to_numpy(dtype=float)


def _brier_at_line(pmf: np.ndarray, y: np.ndarray, line: float) -> np.ndarray:
    k_max = pmf.shape[1] - 2
    support = np.concatenate(
        [np.arange(k_max + 1, dtype=float), np.array([float(k_max + 1)])]
    )
    p_over = pmf @ (support > line).astype(float)
    return (p_over - (np.asarray(y, dtype=float) > line).astype(float)) ** 2


def _score_predictions(
    predictions: pd.DataFrame,
    y: np.ndarray,
    config: MlbConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    pmf = _pmf_matrix(predictions)
    nll = np.asarray(pmf_nll(pmf, y), dtype=float)
    crps = np.asarray(discrete_crps(pmf, y), dtype=float)
    expected = pd.to_numeric(predictions["expected_k"], errors="coerce").to_numpy(
        dtype=float
    )
    mae = np.abs(expected - y)
    rmse = (expected - y) ** 2
    pit = np.asarray(randomized_pit(pmf, y, rng), dtype=float)

    lower = pd.to_numeric(predictions.get("pi_lower"), errors="coerce").to_numpy(
        dtype=float
    )
    upper = pd.to_numeric(predictions.get("pi_upper"), errors="coerce").to_numpy(
        dtype=float
    )
    covered = (y >= lower) & (y <= upper)
    width = upper - lower

    scores: dict[str, float] = {
        "n": float(len(y)),
        "pmf_nll": float(np.mean(nll)),
        "discrete_crps": float(np.mean(crps)),
        "mae": float(np.mean(mae)),
        "rmse": float(np.sqrt(np.mean(rmse))),
        "coverage_80": float(np.mean(covered))
        if np.isfinite(covered).any()
        else float("nan"),
        "width_80": float(np.nanmean(width)),
        "pit_mean": float(np.mean(pit)),
    }
    for line in config.lines:
        scores[f"brier_{str(line).replace('.', '_')}"] = float(
            np.mean(_brier_at_line(pmf, y, float(line)))
        )
    try:
        intercept, slope = calibration_slope_intercept(pmf, y)
        scores["calibration_slope"] = float(slope)
        scores["calibration_intercept"] = float(intercept)
    except Exception:
        pass
    return scores


def _skeleton_feature_rows(
    tables: dict[str, pd.DataFrame], config: MlbConfig
) -> pd.DataFrame:
    """Join spine from pregame snapshots. Not a feature builder."""
    from src.mlb.schemas import empty_frame

    pregame = tables.get("pregame_snapshots")
    if pregame is None or pregame.empty:
        starts = tables["pitcher_starts"]
        keys = pd.DataFrame(
            {
                "pitcher_id": starts["pitcher_id"].to_numpy(),
                "game_pk": starts["game_pk"].to_numpy(),
                "prediction_cutoff_utc": pd.to_datetime(
                    starts["scheduled_start_utc"], utc=True
                )
                - pd.Timedelta(hours=config.forecast_horizon_hours),
                "feature_set_version": config.feature_set_version,
                "pregame_id": starts["game_pk"].astype(str)
                + "-"
                + starts["pitcher_id"].astype(str),
            }
        )
    else:
        keys = pd.DataFrame(
            {
                "pitcher_id": pregame["pitcher_id"].to_numpy(),
                "game_pk": pregame["game_pk"].to_numpy(),
                "prediction_cutoff_utc": pregame["prediction_cutoff_utc"].to_numpy(),
                "feature_set_version": config.feature_set_version,
                "pregame_id": pregame["pregame_id"].to_numpy(),
            }
        )
    spine = empty_frame(FEATURE_ROW_COLUMNS)
    return pd.concat([spine, keys], ignore_index=True)


def _coalesce_overlap_column(
    panel: pd.DataFrame, column: str, right_name: str
) -> pd.DataFrame:
    if right_name not in panel.columns:
        return panel
    if column not in panel.columns:
        panel[column] = panel[right_name]
        return panel.drop(columns=[right_name])
    left = pd.to_numeric(panel[column], errors="coerce")
    right = pd.to_numeric(panel[right_name], errors="coerce")
    both = left.notna() & right.notna()
    if both.any():
        left_v = left.loc[both].to_numpy(dtype=float)
        right_v = right.loc[both].to_numpy(dtype=float)
        if not np.allclose(left_v, right_v, equal_nan=True):
            n_bad = int((~np.isclose(left_v, right_v, equal_nan=True)).sum())
            raise ValueError(
                f"{column} disagrees between feature rows and pitcher starts "
                f"({n_bad} rows)"
            )
    panel[column] = left.where(left.notna(), right)
    return panel.drop(columns=[right_name])


def _assert_panel_features(panel: pd.DataFrame) -> None:
    suffixed = [
        name
        for name in panel.columns
        if any(
            name == f"{column}_{suffix}"
            for column in _CONTEXT_COLUMNS
            for suffix in ("x", "y", "right")
        )
    ]
    if suffixed:
        raise ValueError(f"evaluation panel has suffixed context columns: {suffixed}")
    missing: list[str] = []
    core = (
        "k_bf_shrunk_365",
        "k_bf_shrunk_60",
        "rest_days",
        "pitcher_throws_L",
        "is_home",
    )
    for name in core:
        if name not in panel.columns:
            missing.append(name)
    if "bf_mean_5" not in panel.columns and "bf_per_start_5" not in panel.columns:
        missing.append("bf_mean_5")
    if missing:
        raise ValueError(f"evaluation panel missing required features: {missing}")


def _evaluation_panel(
    tables: dict[str, pd.DataFrame],
    feature_rows: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame:
    starts = tables["pitcher_starts"].copy()
    pregame = tables.get("pregame_snapshots")
    start_keep = [
        c
        for c in (
            "pitcher_id",
            "game_pk",
            "game_date",
            "season",
            "strikeouts",
            "batters_faced",
            "pitches",
            "outs",
            "role",
            "is_home",
            "opponent_team_id",
            "team_id",
            "venue_id",
            "pitcher_hand",
            "scheduled_start_utc",
            "event_time_utc",
            "ingested_at_utc",
            "doubleheader",
        )
        if c in starts.columns
    ]
    panel = feature_rows.merge(
        starts[start_keep],
        on=["pitcher_id", "game_pk"],
        how="left",
        suffixes=("", "_right"),
    )
    for column in _CONTEXT_COLUMNS:
        panel = _coalesce_overlap_column(panel, column, f"{column}_right")
        panel = _coalesce_overlap_column(panel, column, f"{column}_x")
        if f"{column}_y" in panel.columns:
            panel = _coalesce_overlap_column(panel, column, f"{column}_y")
    leftover_right = [c for c in panel.columns if c.endswith("_right")]
    for column in leftover_right:
        base = column[: -len("_right")]
        if base in panel.columns:
            panel[base] = panel[base].where(panel[base].notna(), panel[column])
            panel = panel.drop(columns=[column])
        else:
            panel = panel.rename(columns={column: base})
    if pregame is not None and not pregame.empty:
        pre_keep = [
            c
            for c in (
                "pitcher_id",
                "game_pk",
                "prediction_cutoff_utc",
                "forecast_horizon_hours",
                "scheduled_start_utc",
                "game_date",
                "season",
                "is_opener",
                "is_il_return",
                "is_restricted",
                "expected_rhb_share",
                "starter_state",
                "lineup_state",
                "source_snapshot_ids_json",
            )
            if c in pregame.columns
        ]
        extra = [c for c in pre_keep if c not in {"pitcher_id", "game_pk"}]
        need = ["pitcher_id", "game_pk"] + [
            c for c in extra if c not in panel.columns or panel[c].isna().all()
        ]
        panel = panel.merge(
            pregame[need].drop_duplicates(["pitcher_id", "game_pk"]),
            on=["pitcher_id", "game_pk"],
            how="left",
            suffixes=("", "_pre"),
        )
        if "game_date" not in panel.columns and "game_date_pre" in panel.columns:
            panel["game_date"] = panel["game_date_pre"]
        if (
            "prediction_cutoff_utc" in panel.columns
            and panel["prediction_cutoff_utc"].isna().any()
        ):
            if "prediction_cutoff_utc_pre" in panel.columns:
                panel["prediction_cutoff_utc"] = panel["prediction_cutoff_utc"].fillna(
                    panel["prediction_cutoff_utc_pre"]
                )
        for column in _CONTEXT_COLUMNS:
            panel = _coalesce_overlap_column(panel, column, f"{column}_pre")
    if "game_date" not in panel.columns:
        raise ValueError("evaluation panel requires game_date")
    if "forecast_horizon_hours" not in panel.columns:
        panel["forecast_horizon_hours"] = config.forecast_horizon_hours
    panel = prepare_strikeout_frame(panel)
    _assert_panel_features(panel)
    return panel


def _attach_quotes_to_test(
    test: pd.DataFrame,
    quotes: pd.DataFrame | None,
    config: MlbConfig,
) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return test
    if "prediction_cutoff_utc" not in test.columns:
        return test
    selected = select_quotes_asof(test, quotes, config)
    if selected.empty:
        return test
    quote_cols = [
        c
        for c in (
            "_pred_idx",
            "quote_id",
            "sportsbook",
            "line",
            "over_price",
            "under_price",
            "price_format",
            "fetched_at_utc",
            "quote_status",
            "jurisdiction",
            "rule_version",
        )
        if c in selected.columns
    ]
    if "_pred_idx" not in selected.columns:
        keys = ["pitcher_id", "game_pk"]
        merged = test.merge(
            selected[keys + [c for c in quote_cols if c not in keys]],
            on=keys,
            how="left",
        )
        return merged
    test = test.copy().reset_index(drop=True)
    test["_pred_idx"] = np.arange(len(test), dtype=np.int64)
    attach = selected[quote_cols].drop_duplicates("_pred_idx")
    merged = test.merge(attach, on="_pred_idx", how="left")
    return merged.drop(columns=["_pred_idx"])


def _ensure_prediction_pmf(
    frame: pd.DataFrame, test: pd.DataFrame, config: MlbConfig
) -> pd.DataFrame:
    if all(c in frame.columns for c in PMF_COLUMNS) and "expected_k" in frame.columns:
        out = frame.copy()
        for column in (
            "pitcher_id",
            "game_pk",
            "prediction_cutoff_utc",
            "game_date",
            "season",
            "strikeouts",
        ):
            if column not in out.columns and column in test.columns:
                out[column] = test[column].to_numpy()
        return out.reset_index(drop=True)
    mu = pd.to_numeric(
        frame.get("expected_k", frame.get("mu")), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(mu).any():
        mu = np.full(len(test), 5.0)
    alpha = 1e-4
    if "variance_k" in frame.columns:
        var = pd.to_numeric(frame["variance_k"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_est = (var - mu) / np.clip(mu**2, 1e-8, None)
        alpha = float(np.nanmedian(np.clip(alpha_est, 1e-6, None)))
    packed = prediction_frame_from_mu(
        test,
        mu,
        alpha,
        config,
        model_version=str(
            frame.get("model_version", [config.model_version]).iloc[0]
            if hasattr(frame.get("model_version"), "iloc")
            else config.model_version
        ),
    )
    for column in frame.columns:
        if column not in packed.columns:
            packed[column] = frame[column].to_numpy()
    return packed


def _model_feature_rows(
    tables: dict[str, pd.DataFrame], config: MlbConfig
) -> pd.DataFrame:
    pitches = tables.get("pitch_events")
    gamelog = tables.get("pitcher_starts")
    feature_rows: pd.DataFrame | None = None
    if pitches is None or pitches.empty:
        from src.mlb.pipeline.gamelog_features import build_gamelog_feature_rows

        if gamelog is not None and not gamelog.empty:
            feature_rows = build_gamelog_feature_rows(tables, config)
    if feature_rows is None:
        build_feature_rows = _load_symbol("build_feature_rows", _PIPELINE_MODULES)
        if build_feature_rows is not None:
            feature_rows = build_feature_rows(tables, config)
        else:
            feature_rows = tables.get("feature_rows")
            if feature_rows is None or feature_rows.empty:
                feature_rows = _skeleton_feature_rows(tables, config)
    add_oof = _load_symbol("add_oof_workload_features", _MODEL_MODULES)
    starts = tables["pitcher_starts"]
    if add_oof is not None:
        feature_rows = add_oof(starts, feature_rows, config)
    return feature_rows


def run_backtest(tables: dict[str, pd.DataFrame], config: MlbConfig) -> dict[str, Any]:
    """Chronological folds, strikeout/baseline scores, and quote compare."""
    rng = np.random.default_rng(config.seed)
    fit_strikeouts = _load_symbol("fit_strikeouts", _MODEL_MODULES)
    predict_pmf = _load_symbol("predict_strikeout_pmf", _MODEL_MODULES)
    feature_rows = _model_feature_rows(tables, config)

    panel = _evaluation_panel(tables, feature_rows, config)
    dates = panel["game_date"].astype(str)
    folds = chronological_folds(dates, config)
    quotes = tables.get("market_quotes")

    from src.mlb.models.strikeouts import format_glm_diagnostics
    from src.mlb.models.workload import GlmFitError

    fold_records: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    baseline_score_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    for (train_idx, test_idx), window in zip(folds, config.folds, strict=False):
        train = panel.iloc[train_idx].copy()
        test = panel.iloc[test_idx].copy()
        fold_records.append(
            {
                "name": window.name,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            }
        )
        if test.empty:
            continue
        test = _attach_quotes_to_test(test, quotes, config)
        baselines = fit_baselines(train, test, config)

        if fit_strikeouts is not None and predict_pmf is not None:
            train_fit = (
                train.dropna(subset=["strikeouts"])
                if "strikeouts" in train.columns
                else train
            )
            try:
                model = fit_strikeouts(train_fit, config, fold=window.name)
            except GlmFitError as exc:
                diag = dict(getattr(exc, "diagnostics", None) or {})
                fold_records[-1]["glm_diagnostics"] = diag
                fold_records[-1]["method"] = str(diag.get("method", "glm"))
                LOGGER.error(format_glm_diagnostics(window.name, diag))
                raise
            if str(model.method) != "glm":
                diag = dict(model.extra.get("glm_diagnostics") or {})
                fold_records[-1]["glm_diagnostics"] = diag
                LOGGER.error(format_glm_diagnostics(window.name, diag))
                raise RuntimeError(
                    f"fold {window.name} produced method={model.method!r}; "
                    "intercept-only fallback cannot be scored as strikeout_nb"
                )
            raw_pred = predict_pmf(model, test, config)
            fold_pred = _ensure_prediction_pmf(raw_pred, test, config)
            fold_pred["fit_method"] = model.method
            diag = dict(model.extra.get("glm_diagnostics") or {})
            fold_records[-1]["method"] = model.method
            fold_records[-1]["retained_features"] = list(model.feature_names)
            fold_records[-1]["dropped_features"] = dict(model.dropped_features)
            fold_records[-1]["matrix_rank"] = int(model.matrix_rank)
            fold_records[-1]["glm_diagnostics"] = diag
            LOGGER.info(format_glm_diagnostics(window.name, diag))
        else:
            fold_pred = baselines["league_nb"].copy()
            fold_pred["model_version"] = "baseline_league_nb_fallback"
            fold_pred["fit_method"] = "baseline"
            fold_records[-1]["method"] = "baseline"

        fold_pred["fold"] = window.name
        if "strikeouts" not in fold_pred.columns and "strikeouts" in test.columns:
            fold_pred["strikeouts"] = test["strikeouts"].to_numpy()
        if "game_date" not in fold_pred.columns:
            fold_pred["game_date"] = test["game_date"].to_numpy()
        prediction_parts.append(fold_pred)

        y = pd.to_numeric(test["strikeouts"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y)
        if finite.any():
            scores = _score_predictions(
                fold_pred.loc[finite].reset_index(drop=True), y[finite], config, rng
            )
            scores["fold"] = window.name
            scores["model"] = "strikeout_nb"
            scores["method"] = str(fold_records[-1].get("method", "glm"))
            score_rows.append(scores)
            for name, frame in baselines.items():
                aligned = frame.reset_index(drop=True)
                if "strikeouts" in test.columns and len(aligned) == len(test):
                    yy = y
                    mask = finite
                    scored = aligned.loc[mask].reset_index(drop=True)
                    y_use = yy[mask]
                elif "strikeouts" in aligned.columns:
                    y_use = pd.to_numeric(
                        aligned["strikeouts"], errors="coerce"
                    ).to_numpy(dtype=float)
                    mask = np.isfinite(y_use)
                    scored = aligned.loc[mask].reset_index(drop=True)
                    y_use = y_use[mask]
                else:
                    continue
                if len(scored) == 0:
                    continue
                bscores = _score_predictions(scored, y_use, config, rng)
                bscores["fold"] = window.name
                bscores["model"] = name
                baseline_score_rows.append(bscores)

    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    scores = pd.DataFrame(score_rows) if score_rows else pd.DataFrame()
    baseline_scores = (
        pd.DataFrame(baseline_score_rows) if baseline_score_rows else pd.DataFrame()
    )

    market = pd.DataFrame()
    if quotes is not None and not quotes.empty and not predictions.empty:
        market = compare_market(predictions, quotes, config)
        if not market.empty:
            key_cols = [c for c in ("pitcher_id", "game_pk") if c in market.columns]
            attach_cols = [
                c
                for c in (
                    "market_p_over",
                    "market_p_under",
                    "market_disagreement",
                    "quote_id",
                    "roi_type",
                )
                if c in market.columns
            ]
            if key_cols and attach_cols:
                predictions = predictions.merge(
                    market[key_cols + attach_cols].drop_duplicates(key_cols),
                    on=key_cols,
                    how="left",
                    suffixes=("", "_quoted"),
                )

    return {
        "folds": fold_records,
        "scores": scores,
        "baseline_scores": baseline_scores,
        "predictions": predictions,
        "market": market,
    }
