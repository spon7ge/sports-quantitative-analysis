"""Assemble MLB tables for the 2026 Hugging Face strikeout backtest."""

from __future__ import annotations

import pandas as pd

from src.mlb.config import FoldWindow, MlbConfig
from src.mlb.markets.hf_props import (
    load_hf_strikeout_quotes,
    quotes_from_paired,
)
from src.mlb.pipeline.gamelogs import (
    TRAIN_SEASONS,
    attach_schedule_times,
    default_http,
    game_versions_from_schedule,
    id_map_from_people,
    load_or_fetch_people,
    load_or_fetch_schedule,
    load_or_fetch_starter_logs,
    normalize_player_name,
)
from src.mlb.pipeline.http import HttpFn
from src.mlb.schemas import (
    MARKET_QUOTE_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
    PREGAME_SNAPSHOT_COLUMNS,
    RAW_SNAPSHOT_COLUMNS,
    coerce_frame,
    empty_frame,
)
from src.mlb.storage import MlbStore

HF_2026_FOLDS = (
    FoldWindow("hf_apr_2026", "2026-03-31", "2026-04-01", "2026-04-30"),
    FoldWindow("hf_may_2026", "2026-04-30", "2026-05-01", "2026-05-31"),
    FoldWindow("hf_jun_2026", "2026-05-31", "2026-06-01", "2026-06-30"),
    FoldWindow("hf_jul_2026", "2026-06-30", "2026-07-01", "2026-07-15"),
)


def map_quotes_to_starts(
    quotes: pd.DataFrame,
    starts: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``game_pk`` / ``pitcher_id`` using name + date, then start time."""
    if quotes.empty or starts.empty:
        out = quotes.copy()
        out["game_pk"] = pd.Series(dtype="int64")
        out["pitcher_id"] = pd.Series(dtype="int64")
        return out
    q = quotes.copy()
    q["player_key"] = q["player"].map(normalize_player_name)
    q["start_time"] = pd.to_datetime(q["start_time"], utc=True)
    q["game_date"] = q["start_time"].dt.strftime("%Y-%m-%d")
    s = starts.copy()
    if "player_key" not in s.columns or s["player_key"].eq("").all():
        raise ValueError("starter logs need player_key for name matching")
    s["game_date"] = s["game_date"].astype(str)
    s["scheduled_start_utc"] = pd.to_datetime(s["scheduled_start_utc"], utc=True)
    merged = q.merge(
        s[
            [
                "player_key",
                "game_date",
                "game_pk",
                "pitcher_id",
                "scheduled_start_utc",
            ]
        ],
        on=["player_key", "game_date"],
        how="inner",
    )
    if merged.empty:
        return merged
    merged["start_delta"] = (
        merged["scheduled_start_utc"] - merged["start_time"]
    ).abs()
    merged = merged.sort_values(["game_id", "player_key", "start_delta"])
    return merged.drop_duplicates(["game_id", "player_key"], keep="first")


def pregame_from_starts(starts: pd.DataFrame, config: MlbConfig) -> pd.DataFrame:
    if starts.empty:
        return empty_frame(PREGAME_SNAPSHOT_COLUMNS)
    frame = starts.copy()
    scheduled = pd.to_datetime(frame["scheduled_start_utc"], utc=True)
    cutoff = scheduled - pd.Timedelta(hours=float(config.forecast_horizon_hours))
    frame["pregame_id"] = (
        frame["game_pk"].astype(str) + "-" + frame["pitcher_id"].astype(str)
    )
    frame["prediction_cutoff_utc"] = cutoff
    frame["forecast_horizon_hours"] = float(config.forecast_horizon_hours)
    frame["starter_state"] = "probable"
    frame["lineup_state"] = "team_fallback"
    frame["roster_state"] = "unknown"
    frame["is_opener"] = 0
    frame["is_il_return"] = 0
    frame["is_restricted"] = 0
    frame["expected_rhb_share"] = 0.58
    frame["lineup_batter_ids_json"] = "[]"
    frame["source_snapshot_ids_json"] = "[]"
    return coerce_frame(frame, PREGAME_SNAPSHOT_COLUMNS)


def assemble_gamelog_train_tables(
    config: MlbConfig,
    *,
    http: HttpFn | None = None,
    starts: pd.DataFrame | None = None,
    people: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """2018–2025 starter logs for model fitting. Holds out 2026."""
    getter = http or default_http(config)
    if people is None:
        people = load_or_fetch_people(config, getter, seasons=TRAIN_SEASONS)
    if schedule is None:
        schedule = load_or_fetch_schedule(config, getter, seasons=TRAIN_SEASONS)
    if starts is None:
        starts = load_or_fetch_starter_logs(
            config, getter, people, seasons=TRAIN_SEASONS
        )
        starts = attach_schedule_times(starts, schedule)
    else:
        starts = attach_schedule_times(starts, schedule)
    if "season" in starts.columns:
        starts = starts.loc[pd.to_numeric(starts["season"], errors="coerce") <= 2025]
    versions = game_versions_from_schedule(schedule)
    id_map = id_map_from_people(people)
    pregame = pregame_from_starts(starts, config)
    return {
        "raw_snapshots": empty_frame(RAW_SNAPSHOT_COLUMNS),
        "game_versions": versions,
        "pitch_events": empty_frame(PITCH_EVENT_COLUMNS),
        "plate_appearances": empty_frame(PLATE_APPEARANCE_COLUMNS),
        "pitcher_starts": starts,
        "pregame_snapshots": pregame,
        "market_quotes": empty_frame(MARKET_QUOTE_COLUMNS),
        "id_map": id_map,
    }


def assemble_hf_backtest_tables(
    config: MlbConfig,
    *,
    http: HttpFn | None = None,
    quotes: pd.DataFrame | None = None,
    starts: pd.DataFrame | None = None,
    people: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build cutoff-safe tables from HF quotes + Stats API starter logs.

    Pass ``quotes`` / ``starts`` to skip the network (tests). Live calls cache
    under ``data/mlb/raw_snapshots``.
    """
    getter = http or default_http(config)
    if people is None:
        people = load_or_fetch_people(config, getter)
    if schedule is None:
        schedule = load_or_fetch_schedule(config, getter)
    if starts is None:
        starts = load_or_fetch_starter_logs(config, getter, people)
        starts = attach_schedule_times(starts, schedule)
    elif (
        "scheduled_start_utc" not in starts.columns
        or pd.to_datetime(starts["scheduled_start_utc"], utc=True).isna().all()
    ):
        starts = attach_schedule_times(starts, schedule)
    else:
        starts = starts.copy()
        starts["scheduled_start_utc"] = pd.to_datetime(
            starts["scheduled_start_utc"], utc=True
        )
        if "event_time_utc" not in starts.columns:
            starts["event_time_utc"] = starts["scheduled_start_utc"]
        if "ingested_at_utc" not in starts.columns:
            starts["ingested_at_utc"] = starts["scheduled_start_utc"]
        starts = attach_schedule_times(starts, schedule.iloc[0:0])

    starts = starts.loc[starts["game_date"].astype(str) <= "2026-07-15"].copy()

    if quotes is None:
        quotes = load_hf_strikeout_quotes(config)
    mapped = map_quotes_to_starts(quotes, starts)
    if mapped.empty:
        market_quotes = quotes_from_paired(
            mapped,
            game_pk=pd.Series(dtype="int64"),
            pitcher_id=pd.Series(dtype="int64"),
        )
    else:
        market_quotes = quotes_from_paired(
            mapped,
            game_pk=mapped["game_pk"],
            pitcher_id=mapped["pitcher_id"],
        )
    versions = game_versions_from_schedule(schedule)
    id_map = id_map_from_people(people)
    pregame = pregame_from_starts(starts, config)
    return {
        "raw_snapshots": empty_frame(RAW_SNAPSHOT_COLUMNS),
        "game_versions": versions,
        "pitch_events": empty_frame(PITCH_EVENT_COLUMNS),
        "plate_appearances": empty_frame(PLATE_APPEARANCE_COLUMNS),
        "pitcher_starts": starts,
        "pregame_snapshots": pregame,
        "market_quotes": market_quotes,
        "id_map": id_map,
    }


def write_hf_tables(
    tables: dict[str, pd.DataFrame],
    config: MlbConfig,
) -> MlbStore:
    store = MlbStore(config)
    for name, frame in tables.items():
        if name in (
            "game_versions",
            "pitcher_starts",
            "pregame_snapshots",
            "market_quotes",
            "id_map",
            "pitch_events",
            "plate_appearances",
            "raw_snapshots",
        ):
            store.write_table(name, frame)
    return store
