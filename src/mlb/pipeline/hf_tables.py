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
    cache_path,
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

EVAL_WINDOWS = ("game", "week", "month", "full")


def select_quote_window(
    quotes: pd.DataFrame,
    window: str = "week",
    *,
    window_start: str | None = None,
    game_pk: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Slice as-of quotes to one game, week, month, or the full HF span.

    ``window_start`` is an optional ISO date. When omitted, the earliest quote
    ``start_time`` is used. ``game`` keeps the first ``game_pk`` (or ``game_id``)
    on that date unless ``game_pk`` is passed.
    """
    if quotes is None or quotes.empty:
        raise ValueError("quotes are empty; cannot select an evaluation window")
    raw = str(window).strip().lower()
    aliases = {
        "1 game": "game",
        "one game": "game",
        "1 week": "week",
        "one week": "week",
        "1 month": "month",
        "one month": "month",
        "all": "full",
        "full dataset": "full",
        "dataset": "full",
    }
    kind = aliases.get(raw, raw)
    if kind not in EVAL_WINDOWS:
        raise ValueError(f"window must be one of {EVAL_WINDOWS}, got {window!r}")
    frame = quotes.copy()
    times = pd.to_datetime(frame["start_time"], utc=True)
    frame["_start"] = times
    valid = frame.loc[times.notna()].sort_values("_start")
    if valid.empty:
        raise ValueError("quotes have no valid start_time values")
    origin = (
        pd.Timestamp(window_start, tz="UTC")
        if window_start
        else valid["_start"].iloc[0]
    )
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    else:
        origin = origin.tz_convert("UTC")
    on_or_after = valid.loc[valid["_start"] >= origin]
    if on_or_after.empty:
        on_or_after = valid
        origin = valid["_start"].iloc[0]
    first = on_or_after.iloc[0]

    if kind == "full":
        picked = valid
    elif kind == "game":
        if game_pk is not None and "game_pk" in valid.columns:
            picked = valid.loc[pd.to_numeric(valid["game_pk"], errors="coerce") == int(game_pk)]
            if picked.empty:
                raise ValueError(f"no quotes for game_pk={game_pk}")
        elif "game_pk" in first.index and pd.notna(first.get("game_pk")):
            picked = valid.loc[
                pd.to_numeric(valid["game_pk"], errors="coerce") == int(first["game_pk"])
            ]
        else:
            gid = str(first["game_id"])
            picked = valid.loc[valid["game_id"].astype(str) == gid]
    elif kind == "week":
        start_day = origin.normalize()
        end_day = start_day + pd.Timedelta(days=7)
        picked = valid.loc[(valid["_start"] >= start_day) & (valid["_start"] < end_day)]
    else:
        start_day = origin.normalize()
        if start_day.month == 12:
            month_end = start_day.replace(year=start_day.year + 1, month=1, day=1)
        else:
            month_end = start_day.replace(month=start_day.month + 1, day=1)
        picked = valid.loc[(valid["_start"] >= start_day) & (valid["_start"] < month_end)]

    if picked.empty:
        raise ValueError(f"window={kind!r} selected zero quotes")
    meta = {
        "window": kind,
        "start": str(picked["_start"].min()),
        "end": str(picked["_start"].max()),
        "n_quotes": int(len(picked)),
        "n_games": int(
            picked["game_pk"].nunique()
            if "game_pk" in picked.columns and picked["game_pk"].notna().any()
            else picked["game_id"].nunique()
        ),
        "n_players": int(
            picked["player_key"].nunique()
            if "player_key" in picked.columns
            else picked["player"].nunique()
        ),
    }
    return picked.drop(columns=["_start"]).reset_index(drop=True), meta


def load_cached_hf_history(
    config: MlbConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 2018–2025 + 2026 starter logs from parquet caches.

    Does not write the MlbStore. History is ``starter_logs_2018_2025.parquet``;
    2026 rows come from ``starter_logs.parquet``.
    """
    hist_path = cache_path(config, "starter_logs_2018_2025.parquet")
    live_path = cache_path(config, "starter_logs.parquet")
    frames: list[pd.DataFrame] = []
    if hist_path.exists():
        frames.append(pd.read_parquet(hist_path))
    if live_path.exists():
        live = pd.read_parquet(live_path)
        live = live.loc[pd.to_numeric(live["season"], errors="coerce") == 2026]
        if not live.empty:
            frames.append(live)
    if not frames:
        raise FileNotFoundError(
            "missing starter log caches under data/mlb/raw_snapshots/mlb_gamelogs"
        )
    starts = pd.concat(frames, ignore_index=True)
    starts = starts.drop_duplicates(["pitcher_id", "game_pk"], keep="last")

    sched_frames = [
        pd.read_parquet(cache_path(config, name))
        for name in ("schedule_2018_2025.parquet", "schedule.parquet")
        if cache_path(config, name).exists()
    ]
    schedule = (
        pd.concat(sched_frames, ignore_index=True).drop_duplicates("game_pk", keep="last")
        if sched_frames
        else pd.DataFrame()
    )
    starts = attach_schedule_times(starts, schedule)

    people_frames = [
        pd.read_parquet(cache_path(config, name))
        for name in ("people_2018_2025.parquet", "people.parquet")
        if cache_path(config, name).exists()
    ]
    people = (
        pd.concat(people_frames, ignore_index=True).drop_duplicates("mlb_id", keep="last")
        if people_frames
        else pd.DataFrame()
    )
    return starts, schedule, people


def assemble_windowed_hf_tables(
    config: MlbConfig,
    *,
    window: str = "week",
    window_start: str | None = None,
    game_pk: int | None = None,
    quotes: pd.DataFrame | None = None,
    starts: pd.DataFrame | None = None,
    people: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """In-memory HF tables for one quote window. Never writes the store.

    Feature rows use each quoted pitcher's full start history. Scoring and
    ``market_quotes`` are limited to the selected window.
    """
    if starts is None or people is None or schedule is None:
        cached_starts, cached_sched, cached_people = load_cached_hf_history(config)
        if starts is None:
            starts = cached_starts
        if schedule is None:
            schedule = cached_sched
        if people is None:
            people = cached_people
    elif (
        "scheduled_start_utc" not in starts.columns
        or pd.to_datetime(starts["scheduled_start_utc"], utc=True).isna().all()
    ):
        starts = attach_schedule_times(starts, schedule)

    if quotes is None:
        quotes = load_hf_strikeout_quotes(config)
    windowed, meta = select_quote_window(
        quotes,
        window,
        window_start=window_start,
        game_pk=game_pk,
    )
    mapped = map_quotes_to_starts(windowed, starts)
    if mapped.empty:
        raise ValueError(
            "window quotes did not match any starter logs; check player_key coverage"
        )
    ids = pd.to_numeric(mapped["pitcher_id"], errors="coerce").dropna().astype("int64")
    feature_starts = starts.loc[
        pd.to_numeric(starts["pitcher_id"], errors="coerce").isin(set(ids))
    ].copy()
    market_quotes = quotes_from_paired(
        mapped,
        game_pk=mapped["game_pk"],
        pitcher_id=mapped["pitcher_id"],
    )
    meta = dict(meta)
    meta["n_mapped"] = int(len(mapped))
    meta["n_feature_starts"] = int(len(feature_starts))
    meta["match_rate"] = float(len(mapped) / max(len(windowed), 1))
    tables = {
        "raw_snapshots": empty_frame(RAW_SNAPSHOT_COLUMNS),
        "game_versions": game_versions_from_schedule(schedule),
        "pitch_events": empty_frame(PITCH_EVENT_COLUMNS),
        "plate_appearances": empty_frame(PLATE_APPEARANCE_COLUMNS),
        "pitcher_starts": feature_starts,
        "pregame_snapshots": pregame_from_starts(feature_starts, config),
        "market_quotes": market_quotes,
        "id_map": id_map_from_people(people),
    }
    return tables, meta


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
    sort_keys = ["game_id", "player_key", "start_delta"]
    dedupe_keys = ["game_id", "player_key"]
    if "book" in merged.columns:
        sort_keys = ["game_id", "player_key", "book", "start_delta"]
        dedupe_keys = ["game_id", "player_key", "book"]
    if "line" in merged.columns:
        sort_keys = [k for k in sort_keys if k != "start_delta"] + ["line", "start_delta"]
        dedupe_keys = dedupe_keys + ["line"]
    merged = merged.sort_values(sort_keys)
    return merged.drop_duplicates(dedupe_keys, keep="first")


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
