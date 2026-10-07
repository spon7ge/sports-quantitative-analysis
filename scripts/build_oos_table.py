"""Build the joint minutes x rate OOS table and refit both tail sidecars.

Inputs are the prediction files written by the export cells in
``minutes_xgboost.ipynb`` and ``points_xgboost.ipynb``. Walk-forward rows
keep their fold id as ``window_id``; holdout rows use the holdout sentinel.

Tail sidecars stay on the pinned four-fold contract (folds 1-4), so
window 5 is out of sample for the tails as well as the copula.

    .venv/bin/python scripts/build_oos_table.py --replace

Without ``--replace`` the script stops before writing if any output exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.shared import minutes_sampler, ppm_sampler  # noqa: E402
from models.shared.oos import (  # noqa: E402
    assemble_oos_predictions,
    oos_columns,
    quantile_columns,
    write_oos_parquet,
)

TAIL_FOLDS = (1, 2, 3, 4)
KEYS = ["game_id", "player_id", "window_id"]
PREDICTION_COLUMNS = [
    "game_id",
    "player_id",
    "game_date",
    "window_id",
    "is_holdout",
    "minutes",
    "pts",
    *quantile_columns(),
]


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["game_id"] = frame["game_id"].astype(str)
    frame["player_id"] = frame["player_id"].astype(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame["window_id"] = frame["window_id"].astype(int)
    frame["is_holdout"] = frame["is_holdout"].astype(bool)
    return frame


def align(
    minutes: pd.DataFrame, rate: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep player-games both models predicted in the same window."""
    common = minutes[KEYS].merge(rate[KEYS], on=KEYS, how="inner")
    kept_minutes = minutes.merge(common, on=KEYS, how="inner")
    kept_rate = rate.merge(common, on=KEYS, how="inner")
    print(
        f"{label}: {len(common):,} shared rows "
        f"(dropped {len(minutes) - len(common):,} minutes-only, "
        f"{len(rate) - len(common):,} rate-only)"
    )
    return kept_minutes[PREDICTION_COLUMNS], kept_rate[PREDICTION_COLUMNS]


def fold_ranges(frame: pd.DataFrame, folds) -> dict[int, dict[str, str]]:
    ranges = {}
    for fold in folds:
        dates = frame.loc[frame["window_id"] == fold, "game_date"]
        ranges[fold] = {
            "val_start": str(dates.min().date()),
            "val_end": str(dates.max().date()),
        }
    return ranges


def minutes_tails(minutes: pd.DataFrame) -> minutes_sampler.MinuteTailTables:
    pre = minutes.loc[~minutes["is_holdout"]]
    oof = pre[quantile_columns()].copy()
    oof["minutes"] = pre["minutes"].to_numpy(dtype=float)
    oof["starting"] = pre["starting"].astype(int).to_numpy()
    oof["fold_id"] = pre["window_id"].to_numpy()
    oof["game_date"] = pre["game_date"].to_numpy()
    oof["early_stop"] = pre["early_stop"].to_numpy()
    oof["train_tail_frac"] = pre["train_tail_frac"].astype(float).to_numpy()
    return minutes_sampler.build_tail_tables(
        oof,
        folds=list(TAIL_FOLDS),
        fold_ranges=fold_ranges(pre, TAIL_FOLDS),
    )


def check_writable(paths: list[Path], replace: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not replace:
        names = "\n  ".join(str(path.relative_to(ROOT)) for path in existing)
        raise SystemExit(f"Refusing to overwrite without --replace:\n  {names}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--minutes",
        type=Path,
        default=ROOT / "data/oos/nba/minutes_predictions.parquet",
    )
    parser.add_argument(
        "--rate", type=Path, default=ROOT / "data/oos/nba/rate_predictions.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/oos/nba/minutes_rate_oos.parquet"
    )
    parser.add_argument(
        "--minutes-tails",
        type=Path,
        default=ROOT / "models/saved_models/min_nba_tails_2026-04-12.joblib",
    )
    parser.add_argument(
        "--rate-tails",
        type=Path,
        default=ROOT / "models/saved_models/pts_nba_tails_2026-04-12.joblib",
    )
    parser.add_argument(
        "--replace", action="store_true", help="delete and rewrite existing outputs"
    )
    args = parser.parse_args()
    for name in ("minutes", "rate", "out", "minutes_tails", "rate_tails"):
        setattr(args, name, getattr(args, name).resolve())

    minutes = load_predictions(args.minutes)
    rate = load_predictions(args.rate)

    pre_min, pre_rate = align(
        minutes.loc[~minutes["is_holdout"]],
        rate.loc[~rate["is_holdout"]],
        "walk-forward",
    )
    ho_min, ho_rate = align(
        minutes.loc[minutes["is_holdout"]], rate.loc[rate["is_holdout"]], "holdout"
    )

    table = assemble_oos_predictions(pre_min, pre_rate, ho_min, ho_rate)
    assert list(table.columns) == oos_columns()
    summary = table.groupby("window_id").agg(
        rows=("game_id", "size"),
        first=("game_date", "min"),
        last=("game_date", "max"),
    )
    print(summary.to_string())

    min_tables = minutes_tails(minutes)
    starting = minutes[["game_id", "player_id", "starting"]]
    rate_tables = ppm_sampler.fit_preholdout_tails(table, starting, folds=TAIL_FOLDS)

    outputs = [args.out, args.minutes_tails, args.rate_tails]
    check_writable(outputs, args.replace)
    for path in outputs:
        path.unlink(missing_ok=True)
    write_oos_parquet(table, args.out)
    minutes_sampler.save_tail_sidecar(min_tables, args.minutes_tails)
    ppm_sampler.save_tail_sidecar(rate_tables, args.rate_tails)
    minutes_sampler.load_tail_sidecar(args.minutes_tails)
    ppm_sampler.load_tail_sidecar(args.rate_tails)
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
