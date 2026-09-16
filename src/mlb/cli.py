"""Argparse CLI for the MLB starter strikeout stack: ``python -m src.mlb``."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.mlb.config import DEFAULT_CONFIG_PATH, MlbConfig, load_config
from src.mlb.evaluation.backtest import run_backtest
from src.mlb.evaluation.market import compare_market
from src.mlb.evaluation.report import write_daily_report
from src.mlb.fixtures import load_fixture_tables
from src.mlb.markets.quotes import import_quotes
from src.mlb.schemas import TABLE_SCHEMAS
from src.mlb.storage import MlbStore

_PIPELINE_MODULES = (
    "src.mlb.pipeline",
    "src.mlb.pipeline.ingest",
    "src.mlb.pipeline.features",
    "src.mlb.pipeline.build",
    "src.mlb.pipeline.statcast",
    "src.mlb.pipeline.schedule",
)
_MODEL_MODULES = (
    "src.mlb.models",
    "src.mlb.models.strikeouts",
    "src.mlb.models.workload",
    "src.mlb.models.serialize",
    "src.mlb.models.io",
    "src.mlb.models.pmf",
)


class MissingMlbModuleError(RuntimeError):
    """Raised when a sibling pipeline/model symbol has not landed yet."""


def _load_symbol(name: str, modules: tuple[str, ...]) -> Callable:
    tried: list[str] = []
    for module_name in modules:
        tried.append(module_name)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    expected = ", ".join(modules)
    raise MissingMlbModuleError(
        f"Missing module {modules[0]}.{name} (also searched {expected})"
    )


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config/mlb.yaml",
    )
    shared.add_argument(
        "--fixture",
        action="store_true",
        help="Use bundled synthetic tables (no network)",
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.mlb",
        description="MLB starting-pitcher strikeout MVP",
        parents=[shared],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_statcast = sub.add_parser(
        "ingest-statcast", parents=[shared], help="Ingest Baseball Savant CSV"
    )
    p_statcast.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_statcast.add_argument("--end", required=True, help="YYYY-MM-DD")

    p_sched = sub.add_parser(
        "snapshot-schedule", parents=[shared], help="Snapshot MLB Stats API schedule"
    )
    p_sched.add_argument("--date", required=True, help="YYYY-MM-DD")

    p_lu = sub.add_parser(
        "snapshot-lineups", parents=[shared], help="Snapshot lineups for a game"
    )
    p_lu.add_argument("--game-pk", type=int, required=True)

    p_feat = sub.add_parser(
        "build-features", parents=[shared], help="Build cutoff-strict feature rows"
    )
    p_feat.add_argument("--cutoff", required=True, help="ISO-8601 cutoff")

    sub.add_parser(
        "train-workload", parents=[shared], help="Fit the OOF workload model"
    )
    sub.add_parser(
        "train-strikeouts", parents=[shared], help="Fit the strikeout NB model"
    )
    p_back = sub.add_parser(
        "backtest",
        parents=[shared],
        help="Chronological backtest on stored, fixture, or HF 2026 tables",
    )
    p_back.add_argument(
        "--hf-props",
        action="store_true",
        help="SmartStake HF quotes (Mar-Jul 2026) plus Stats API starter logs",
    )
    sub.add_parser(
        "ingest-gamelogs",
        parents=[shared],
        help="Fetch 2018-2025 MLB starter game logs for training",
    )
    sub.add_parser(
        "ingest-hf-props",
        parents=[shared],
        help="Cache HF strikeout quotes and MLB starter logs",
    )

    p_pred = sub.add_parser(
        "predict", parents=[shared], help="Predict PMFs at a cutoff"
    )
    p_pred.add_argument("--cutoff", help="ISO-8601 cutoff")

    p_quotes = sub.add_parser(
        "import-quotes", parents=[shared], help="Import CSV/JSON quotes"
    )
    p_quotes.add_argument("--path", required=True)

    p_cmp = sub.add_parser(
        "compare-market", parents=[shared], help="Compare predictions to as-of quotes"
    )
    p_cmp.add_argument("--predictions", required=True)
    p_cmp.add_argument("--quotes", required=True)

    p_rep = sub.add_parser(
        "daily-report", parents=[shared], help="Write CSV + markdown daily report"
    )
    p_rep.add_argument("--cutoff", required=True, help="ISO-8601 cutoff")
    p_rep.add_argument("--out", required=True)
    return parser


def _load_tables(
    config: MlbConfig, *, fixture: bool
) -> tuple[dict[str, pd.DataFrame], MlbStore]:
    store = MlbStore(config)
    if fixture:
        tables = load_fixture_tables()
        for name, frame in tables.items():
            if name in TABLE_SCHEMAS and not frame.empty:
                store.write_table(name, frame)
        return tables, store
    tables = {
        name: store.read_table(name) for name in TABLE_SCHEMAS if name != "predictions"
    }
    return tables, store


def _read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(source)
    return pd.read_parquet(source)


def _latest_artifact(directory: Path, prefixes: tuple[str, ...]) -> Path | None:
    if not directory.exists():
        return None
    candidates: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pkl", ".joblib", ".json", ".pickle"}:
            continue
        blob = str(path).lower()
        if any(prefix.lower() in blob for prefix in prefixes):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _write_predictions(
    store: MlbStore, frame: pd.DataFrame, name: str = "backtest_predictions.parquet"
) -> Path:
    path = store.config.artifact_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _http_client(args: argparse.Namespace, config: MlbConfig):
    if getattr(args, "fixture", False):
        return None
    from src.mlb.pipeline.http import HttpClient

    return HttpClient(config)


def _cmd_ingest_statcast(args: argparse.Namespace, config: MlbConfig) -> int:
    ingest = _load_symbol("ingest_statcast", _PIPELINE_MODULES)
    frame = ingest(
        config,
        start_date=args.start,
        end_date=args.end,
        http=_http_client(args, config),
    )
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        store = MlbStore(config)
        name = "pitch_events" if "pitch_id" in frame.columns else "raw_snapshots"
        store.write_table(name, frame)
        print(f"Wrote {len(frame)} rows to {name}")
    else:
        print("ingest-statcast completed")
    return 0


def _cmd_snapshot_schedule(args: argparse.Namespace, config: MlbConfig) -> int:
    ingest = _load_symbol("ingest_schedule", _PIPELINE_MODULES)
    frame = ingest(config, game_date=args.date, http=_http_client(args, config))
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        MlbStore(config).write_table("game_versions", frame)
        print(f"Wrote {len(frame)} game_versions rows")
    else:
        print("snapshot-schedule completed")
    return 0


def _cmd_snapshot_lineups(args: argparse.Namespace, config: MlbConfig) -> int:
    ingest = _load_symbol("ingest_lineups", _PIPELINE_MODULES)
    frame = ingest(config, game_pk=args.game_pk, http=_http_client(args, config))
    print(f"snapshot-lineups returned {0 if frame is None else len(frame)} rows")
    return 0


def _cmd_build_features(args: argparse.Namespace, config: MlbConfig) -> int:
    build = _load_symbol("build_feature_rows", _PIPELINE_MODULES)
    tables, store = _load_tables(config, fixture=args.fixture)
    rows = build(tables, config)
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    if "prediction_cutoff_utc" in rows.columns:
        rows = rows.loc[
            pd.to_datetime(rows["prediction_cutoff_utc"], utc=True) <= cutoff
        ]
    leakage = None
    try:
        leakage = _load_symbol("assert_no_leakage", _PIPELINE_MODULES)
    except MissingMlbModuleError:
        leakage = None
    if leakage is not None:
        leakage(rows, tables)
    store.write_table("feature_rows", rows)
    print(f"Wrote {len(rows)} feature_rows")
    return 0


def _prepare_train_frame(
    tables: dict[str, pd.DataFrame], config: MlbConfig
) -> pd.DataFrame:
    from src.mlb.evaluation.backtest import _evaluation_panel, _skeleton_feature_rows

    pitches = tables.get("pitch_events")
    if pitches is None or pitches.empty:
        from src.mlb.pipeline.gamelog_features import build_gamelog_feature_rows

        feature_rows = build_gamelog_feature_rows(tables, config)
        return _evaluation_panel(tables, feature_rows, config)

    build = None
    try:
        build = _load_symbol("build_feature_rows", _PIPELINE_MODULES)
    except MissingMlbModuleError:
        build = None
    if build is not None:
        feature_rows = build(tables, config)
    else:
        feature_rows = tables.get("feature_rows")
        if feature_rows is None or feature_rows.empty:
            feature_rows = _skeleton_feature_rows(tables, config)
    add_oof = None
    try:
        add_oof = _load_symbol("add_oof_workload_features", _MODEL_MODULES)
    except MissingMlbModuleError:
        add_oof = None
    if add_oof is not None:
        feature_rows = add_oof(tables["pitcher_starts"], feature_rows, config)
    return _evaluation_panel(tables, feature_rows, config)


def _cmd_train_workload(args: argparse.Namespace, config: MlbConfig) -> int:
    fit_workload = _load_symbol("fit_workload", _MODEL_MODULES)
    save_model = _load_symbol("save_model", _MODEL_MODULES)
    tables, store = _load_tables(config, fixture=args.fixture)
    frame = _prepare_train_frame(tables, config)
    model = fit_workload(frame, config)
    path = (
        store.config.artifact_dir / "workload" / f"{config.workload_model_version}.pkl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(path, model)
    print(f"Wrote workload model {path}")
    return 0


def _cmd_train_strikeouts(args: argparse.Namespace, config: MlbConfig) -> int:
    fit_strikeouts = _load_symbol("fit_strikeouts", _MODEL_MODULES)
    save_model = _load_symbol("save_model", _MODEL_MODULES)
    tables, store = _load_tables(config, fixture=args.fixture)
    frame = _prepare_train_frame(tables, config)
    model = fit_strikeouts(
        frame.dropna(subset=["strikeouts"]) if "strikeouts" in frame.columns else frame,
        config,
    )
    path = store.config.artifact_dir / "strikeouts" / f"{config.model_version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(path, model)
    fit_frame = (
        frame.dropna(subset=["strikeouts"])
        if "strikeouts" in frame.columns
        else frame
    )
    n = len(fit_frame)
    if "season" in frame.columns:
        seasons = pd.to_numeric(frame["season"], errors="coerce")
        print(
            f"Fit {n} starts across {int(seasons.min())}-{int(seasons.max())} "
            f"({int(seasons.nunique())} seasons)"
        )
    print(f"Wrote strikeout model {path}")
    return 0


def _cmd_ingest_gamelogs(args: argparse.Namespace, config: MlbConfig) -> int:
    from src.mlb.pipeline.gamelogs import TRAIN_SEASONS
    from src.mlb.pipeline.hf_tables import (
        assemble_gamelog_train_tables,
        write_hf_tables,
    )

    _ = args
    tables = assemble_gamelog_train_tables(config)
    write_hf_tables(tables, config)
    starts = tables["pitcher_starts"]
    seasons = sorted(pd.to_numeric(starts["season"], errors="coerce").dropna().unique())
    print(
        f"Cached {len(starts)} starter rows for seasons "
        f"{int(min(seasons))}-{int(max(seasons))} "
        f"(requested {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}; 2026 held out)"
    )
    print(starts.groupby("season").size().to_string())
    return 0


def _cmd_ingest_hf_props(args: argparse.Namespace, config: MlbConfig) -> int:
    from src.mlb.pipeline.hf_tables import assemble_hf_backtest_tables, write_hf_tables

    _ = args
    tables = assemble_hf_backtest_tables(config)
    write_hf_tables(tables, config)
    quotes = tables["market_quotes"]
    starts = tables["pitcher_starts"]
    print(f"Cached {len(starts)} pitcher starts and {len(quotes)} HF as-of quotes")
    return 0


def _cmd_backtest(args: argparse.Namespace, config: MlbConfig) -> int:
    if getattr(args, "hf_props", False) and args.fixture:
        raise SystemExit("use only one of --fixture or --hf-props")
    if getattr(args, "hf_props", False):
        from src.mlb.pipeline.hf_tables import (
            HF_2026_FOLDS,
            assemble_hf_backtest_tables,
            write_hf_tables,
        )

        config = replace(config, folds=HF_2026_FOLDS)
        tables = assemble_hf_backtest_tables(config)
        store = write_hf_tables(tables, config)
    else:
        tables, store = _load_tables(config, fixture=args.fixture)
    result = run_backtest(tables, config)
    pred_path = _write_predictions(store, result["predictions"])
    scores_path = store.write_json(
        store.config.artifact_dir / "backtest_scores.json",
        {
            "folds": result["folds"],
            "scores": result["scores"].to_dict(orient="records")
            if isinstance(result["scores"], pd.DataFrame)
            else result["scores"],
            "baseline_scores": result["baseline_scores"].to_dict(orient="records")
            if isinstance(result["baseline_scores"], pd.DataFrame)
            else result["baseline_scores"],
        },
    )
    print(f"Wrote predictions {pred_path}")
    print(f"Wrote scores {scores_path}")
    if isinstance(result["scores"], pd.DataFrame) and not result["scores"].empty:
        print(result["scores"].to_string(index=False))
    market = result.get("market")
    if isinstance(market, pd.DataFrame) and not market.empty:
        market_path = store.config.artifact_dir / "market_comparison.parquet"
        market_path.parent.mkdir(parents=True, exist_ok=True)
        market.to_parquet(market_path, index=False)
        print(f"Wrote quoted-price simulation {market_path}")
        cols = [
            c
            for c in (
                "paired_log_score_model",
                "paired_log_score_market",
                "brier_model",
                "brier_market",
            )
            if c in market.columns
        ]
        if cols:
            print("quoted-price simulation means (not ROI)")
            print(market[cols].mean(numeric_only=True).to_string())
    return 0


def _predict_frame(
    tables: dict[str, pd.DataFrame],
    config: MlbConfig,
    cutoff: pd.Timestamp,
    *,
    train_if_fixture: bool,
) -> pd.DataFrame:
    panel = _prepare_train_frame(tables, config)
    cutoffs = pd.to_datetime(panel["prediction_cutoff_utc"], utc=True)
    target = panel.loc[cutoffs == cutoff]
    if target.empty:
        same_day = cutoffs.dt.strftime("%Y-%m-%d") == cutoff.strftime("%Y-%m-%d")
        target = panel.loc[same_day]
    if target.empty:
        raise ValueError(f"No rows at cutoff {cutoff.isoformat()}")

    fit_strikeouts = _load_symbol("fit_strikeouts", _MODEL_MODULES)
    predict_pmf = _load_symbol("predict_strikeout_pmf", _MODEL_MODULES)
    load_model = None
    try:
        load_model = _load_symbol("load_model", _MODEL_MODULES)
    except MissingMlbModuleError:
        load_model = None

    model = None
    artifact = _latest_artifact(
        config.artifact_dir, ("strikeouts", config.model_version, "nb_k")
    )
    if artifact is not None and load_model is not None and not train_if_fixture:
        loaded = load_model(artifact)
        try:
            from src.mlb.models.serialize import as_strikeout_model

            model = as_strikeout_model(loaded)
        except (KeyError, TypeError, ValueError):
            model = None
    if model is None:
        train = panel.loc[cutoffs < cutoff]
        if train.empty:
            train = panel.loc[
                pd.to_datetime(panel["game_date"]) < cutoff.strftime("%Y-%m-%d")
            ]
        model = fit_strikeouts(
            train.dropna(subset=["strikeouts"])
            if "strikeouts" in train.columns
            else train,
            config,
        )
    preds = predict_pmf(model, target, config)
    from src.mlb.evaluation.backtest import _ensure_prediction_pmf

    return _ensure_prediction_pmf(preds, target, config)


def _cmd_predict(args: argparse.Namespace, config: MlbConfig) -> int:
    tables, store = _load_tables(config, fixture=args.fixture)
    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
    else:
        if not args.fixture:
            raise SystemExit("predict requires --cutoff unless --fixture is set")
        pregame = tables["pregame_snapshots"]
        cutoff = pd.to_datetime(pregame["prediction_cutoff_utc"], utc=True).max()
    preds = _predict_frame(tables, config, cutoff, train_if_fixture=args.fixture)
    path = _write_predictions(store, preds, name="predictions.parquet")
    print(f"Wrote {len(preds)} predictions to {path}")
    return 0


def _cmd_import_quotes(args: argparse.Namespace, config: MlbConfig) -> int:
    imported = import_quotes(args.path, config)
    store = MlbStore(config)
    existing = store.read_table("market_quotes")
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, imported], ignore_index=True)
        combined = combined.drop_duplicates("quote_id", keep="last")
    else:
        combined = imported
    store.write_table("market_quotes", combined)
    print(f"Imported {len(imported)} quotes ({len(combined)} stored)")
    return 0


def _cmd_compare_market(args: argparse.Namespace, config: MlbConfig) -> int:
    predictions = _read_frame(args.predictions)
    quotes = (
        import_quotes(args.quotes, config)
        if Path(args.quotes).suffix.lower() in {".csv", ".json"}
        else _read_frame(args.quotes)
    )
    compared = compare_market(predictions, quotes, config)
    store = MlbStore(config)
    path = store.config.artifact_dir / "market_comparison.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    compared.to_parquet(path, index=False)
    print(f"Wrote {len(compared)} compared rows to {path}")
    return 0


def _cmd_daily_report(args: argparse.Namespace, config: MlbConfig) -> int:
    tables, store = _load_tables(config, fixture=args.fixture)
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    pred_path = store.config.artifact_dir / "predictions.parquet"
    predictions: pd.DataFrame | None = None
    if pred_path.exists():
        loaded = pd.read_parquet(pred_path)
        if "prediction_cutoff_utc" in loaded.columns:
            stamps = pd.to_datetime(loaded["prediction_cutoff_utc"], utc=True)
            subset = loaded.loc[stamps == cutoff]
            if subset.empty:
                subset = loaded.loc[
                    stamps.dt.strftime("%Y-%m-%d") == cutoff.strftime("%Y-%m-%d")
                ]
            predictions = subset if not subset.empty else loaded
        else:
            predictions = loaded
    if predictions is None or predictions.empty:
        try:
            predictions = _predict_frame(
                tables, config, cutoff, train_if_fixture=args.fixture
            )
        except MissingMlbModuleError:
            from src.mlb.evaluation.baselines import fit_baselines

            panel = _prepare_train_frame(tables, config)
            cutoffs = pd.to_datetime(panel["prediction_cutoff_utc"], utc=True)
            target = panel.loc[
                cutoffs.dt.strftime("%Y-%m-%d") == cutoff.strftime("%Y-%m-%d")
            ]
            train = panel.loc[cutoffs < cutoff]
            if target.empty:
                raise
            predictions = fit_baselines(
                train if not train.empty else panel, target, config
            )["marcel"]
    report = write_daily_report(predictions, args.out)
    print(f"Wrote daily report {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    globals_parser = argparse.ArgumentParser(add_help=False)
    globals_parser.add_argument("--config", default=None)
    globals_parser.add_argument("--fixture", action="store_true")
    global_args, _rest = globals_parser.parse_known_args(raw)
    parser = build_parser()
    args = parser.parse_args(argv)
    if global_args.config:
        args.config = global_args.config
    args.fixture = bool(global_args.fixture or getattr(args, "fixture", False))
    config = load_config(args.config)
    handlers = {
        "ingest-statcast": _cmd_ingest_statcast,
        "snapshot-schedule": _cmd_snapshot_schedule,
        "snapshot-lineups": _cmd_snapshot_lineups,
        "build-features": _cmd_build_features,
        "train-workload": _cmd_train_workload,
        "train-strikeouts": _cmd_train_strikeouts,
        "backtest": _cmd_backtest,
        "ingest-gamelogs": _cmd_ingest_gamelogs,
        "ingest-hf-props": _cmd_ingest_hf_props,
        "predict": _cmd_predict,
        "import-quotes": _cmd_import_quotes,
        "compare-market": _cmd_compare_market,
        "daily-report": _cmd_daily_report,
    }
    try:
        return handlers[args.command](args, config)
    except MissingMlbModuleError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
