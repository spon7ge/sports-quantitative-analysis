"""Load SmartStake Hugging Face player-prop ticks into quote rows.

The public dataset is minute-level. This module keeps the last pre-start
tick per (game, player, book, line, side), pairs over/under, and emits at
most one as-of quote per starter. Sportsbook prices are not baseball
features.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.schemas import MARKET_QUOTE_COLUMNS, coerce_frame

HF_DATASET_ID = "SmartStake/mlb-player-props"
HF_STRIKEOUT_MARKET = "player strikeouts"
HF_MONTHS = ("2026-03", "2026-04", "2026-05", "2026-06", "2026-07")
PREFERRED_BOOKS = (
    "pinnacle",
    "draftkings",
    "fanduel",
    "bet365",
    "caesars",
    "betmgm",
    "hard_rock",
)
DEFAULT_TARGET_LINE = 5.5
MIN_STARTER_LINE = 3.5
MAX_STARTER_LINE = 12.5


def hf_local_dir(config: MlbConfig) -> Path:
    return config.raw_dir / "hf_mlb_player_props"


def ensure_hf_dataset(config: MlbConfig) -> Path:
    """Download the hive-partitioned parquet dataset if it is not cached."""
    dest = hf_local_dir(config)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("mon=*/*.parquet")):
        return dest
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(dest),
        max_workers=2,
    )
    return dest


def parquet_glob(root: Path) -> str:
    return str(root / "mon=*/*.parquet")


def collapse_last_pre_start(
    frame: pd.DataFrame,
    *,
    horizon_hours: float = 2.0,
) -> pd.DataFrame:
    """Keep the last tick strictly before ``start_time - horizon`` per quote key."""
    if frame.empty:
        return frame.copy()
    ticks = frame.copy()
    ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)
    ticks["start_time"] = pd.to_datetime(ticks["start_time"], utc=True)
    cutoff = ticks["start_time"] - pd.Timedelta(hours=float(horizon_hours))
    ticks = ticks.loc[ticks["ts"] < cutoff]
    if ticks.empty:
        return ticks
    ticks["player_key"] = ticks["player"].astype(str).str.lower()
    ticks = ticks.sort_values("ts")
    return (
        ticks.groupby(
            ["game_id", "player_key", "book", "line", "side"],
            as_index=False,
            sort=False,
        )
        .tail(1)
        .drop(columns=["player_key"])
        .reset_index(drop=True)
    )


def load_collapsed_strikeout_ticks(
    root: Path,
    *,
    min_line: float = MIN_STARTER_LINE,
    max_line: float = MAX_STARTER_LINE,
    horizon_hours: float = 2.0,
) -> pd.DataFrame:
    """Scan local hive parquet and return last pre-cutoff strikeout ticks."""
    source = parquet_glob(root)
    hours = int(round(float(horizon_hours)))
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            WITH so AS (
                SELECT
                    game_id,
                    start_time,
                    player,
                    line,
                    side,
                    book,
                    ts,
                    odds,
                    result,
                    won
                FROM read_parquet('{source}', hive_partitioning=true)
                WHERE market = '{HF_STRIKEOUT_MARKET}'
                  AND ts < start_time - INTERVAL '{hours} hours'
                  AND line >= {float(min_line)}
                  AND line <= {float(max_line)}
            ),
            ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY game_id, lower(player), book, line, side
                        ORDER BY ts DESC
                    ) AS rn
                FROM so
            )
            SELECT
                game_id, start_time, player, line, side, book, ts, odds, result, won
            FROM ranked
            WHERE rn = 1
            """
        ).df()
    finally:
        con.close()


def pair_over_under(ticks: pd.DataFrame) -> pd.DataFrame:
    """Inner-join over and under decimal odds at the same key."""
    if ticks.empty:
        return ticks.copy()
    frame = ticks.copy()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame["player_key"] = frame["player"].astype(str).str.lower()
    over = frame.loc[frame["side"] == "over"].copy()
    under = frame.loc[frame["side"] == "under"].copy()
    keys = ["game_id", "player_key", "book", "line"]
    paired = over.merge(
        under,
        on=keys,
        how="inner",
        suffixes=("_over", "_under"),
    )
    if paired.empty:
        return paired
    ts_over = pd.to_datetime(paired["ts_over"], utc=True)
    ts_under = pd.to_datetime(paired["ts_under"], utc=True)
    start = pd.to_datetime(
        paired["start_time_over"].where(
            paired["start_time_over"].notna(), paired["start_time_under"]
        ),
        utc=True,
    )
    fetched = pd.concat([ts_over, ts_under], axis=1).max(axis=1)
    result = paired["result_over"].where(
        paired["result_over"].notna(), paired["result_under"]
    )
    player = paired["player_over"].where(
        paired["player_over"].notna(), paired["player_under"]
    )
    return pd.DataFrame(
        {
            "game_id": paired["game_id"].astype(str),
            "player": player.astype(str),
            "player_key": paired["player_key"].astype(str),
            "book": paired["book"].astype(str),
            "line": pd.to_numeric(paired["line"], errors="coerce"),
            "over_price": pd.to_numeric(paired["odds_over"], errors="coerce"),
            "under_price": pd.to_numeric(paired["odds_under"], errors="coerce"),
            "start_time": start,
            "fetched_at_utc": fetched,
            "result": pd.to_numeric(result, errors="coerce"),
        }
    )


def _book_rank(book: pd.Series) -> pd.Series:
    ranks = {name: i for i, name in enumerate(PREFERRED_BOOKS)}
    mapped = book.astype(str).str.lower().map(ranks)
    return mapped.fillna(len(PREFERRED_BOOKS) + 1)


def select_asof_strikeout_quotes(
    paired: pd.DataFrame,
    *,
    target_line: float = DEFAULT_TARGET_LINE,
) -> pd.DataFrame:
    """One paired quote per (game_id, player): nearest line, preferred book."""
    if paired.empty:
        return paired.copy()
    frame = paired.copy()
    over_ok = pd.to_numeric(frame["over_price"], errors="coerce")
    under_ok = pd.to_numeric(frame["under_price"], errors="coerce")
    frame = frame.loc[
        (over_ok > 1.05)
        & (over_ok < 8.0)
        & (under_ok > 1.05)
        & (under_ok < 8.0)
    ]
    if frame.empty:
        return frame
    frame["player_key"] = frame["player"].astype(str).str.lower()
    preferred = frame["book"].astype(str).str.lower().isin(PREFERRED_BOOKS)
    group_key = frame["game_id"].astype(str) + "|" + frame["player_key"]
    has_pref = preferred.groupby(group_key).transform("any")
    frame = frame.loc[~has_pref | preferred]
    frame["line_distance"] = (
        pd.to_numeric(frame["line"], errors="coerce") - float(target_line)
    ).abs()
    frame["book_rank"] = _book_rank(frame["book"])
    frame["fetched_at_utc"] = pd.to_datetime(frame["fetched_at_utc"], utc=True)
    frame = frame.sort_values(
        ["game_id", "player_key", "line_distance", "book_rank", "fetched_at_utc"],
        ascending=[True, True, True, True, False],
    )
    return (
        frame.drop_duplicates(["game_id", "player_key"], keep="first")
        .drop(columns=["line_distance", "book_rank"])
        .reset_index(drop=True)
    )


def quotes_from_paired(
    paired: pd.DataFrame,
    *,
    game_pk: pd.Series,
    pitcher_id: pd.Series,
) -> pd.DataFrame:
    """Coerce paired SmartStake rows onto ``MARKET_QUOTE_COLUMNS``."""
    if paired.empty:
        return coerce_frame(paired, MARKET_QUOTE_COLUMNS)
    frame = paired.copy()
    frame["game_pk"] = pd.to_numeric(game_pk, errors="coerce")
    frame["pitcher_id"] = pd.to_numeric(pitcher_id, errors="coerce")
    frame = frame.dropna(subset=["game_pk", "pitcher_id"])
    if frame.empty:
        return coerce_frame(frame, MARKET_QUOTE_COLUMNS)
    frame["quote_id"] = (
        "hf-"
        + frame["game_pk"].astype("int64").astype(str)
        + "-"
        + frame["pitcher_id"].astype("int64").astype(str)
        + "-"
        + frame["book"].astype(str)
        + "-"
        + frame["line"].astype(str)
    )
    frame["sportsbook"] = frame["book"].astype(str)
    frame["jurisdiction"] = "US"
    frame["price_format"] = "decimal"
    frame["quote_status"] = "open"
    frame["rule_version"] = "k_standard_v1"
    frame["fill_flag"] = pd.NA
    frame["rejected_flag"] = pd.NA
    frame["limit"] = float("nan")
    frame["void_flag"] = pd.NA
    frame["settled_flag"] = (
        frame["result"].notna().astype("Int64")
        if "result" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    )
    return coerce_frame(frame, MARKET_QUOTE_COLUMNS)


def load_hf_strikeout_quotes(config: MlbConfig) -> pd.DataFrame:
    """Cached last-pre-start paired strikeout quotes (unmapped ids)."""
    root = ensure_hf_dataset(config)
    cache = config.raw_dir / "hf_strikeout_asof.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ticks = load_collapsed_strikeout_ticks(root)
    paired = pair_over_under(ticks)
    selected = select_asof_strikeout_quotes(paired)
    cache.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(cache, index=False)
    return selected
