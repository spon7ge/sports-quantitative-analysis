"""Snapshot and ingest Statcast, schedule, lineups, and Chadwick payloads."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.pipeline.http import HttpFn
from src.mlb.pipeline.parse import (
    parse_lineups,
    parse_people,
    parse_schedule,
    parse_statcast,
)
from src.mlb.schemas import (
    ID_MAP_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
    RAW_SNAPSHOT_COLUMNS,
    coerce_frame,
)
from src.mlb.storage import MlbStore

STATCAST_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LINEUP_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
CHADWICK_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people.csv"
)


def _payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return payload.encode("utf-8")


def _params_json(params: dict[str, Any] | None) -> str:
    return json.dumps(params or {}, sort_keys=True, default=str)


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _append_table(store: MlbStore, name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    existing = store.read_table(name)
    combined = pd.concat([existing, frame], ignore_index=True)
    store.write_table(name, combined)


def snapshot_raw(
    config: MlbConfig,
    source: str,
    params: dict[str, Any] | None,
    payload: str | bytes,
    fetched_at: datetime | pd.Timestamp,
) -> str:
    payload_b = _payload_bytes(payload)
    params_json = _params_json(params)
    snapshot_id = hashlib.sha256(
        source.encode("utf-8") + params_json.encode("utf-8") + payload_b
    ).hexdigest()[:16]
    payload_hash = hashlib.sha256(payload_b).hexdigest()
    safe_source = source.replace("/", "_")
    dest_dir = config.raw_dir / safe_source
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload_path = dest_dir / snapshot_id
    payload_path.write_bytes(payload_b)

    store = MlbStore(config)
    row = coerce_frame(
        pd.DataFrame(
            [
                {
                    "snapshot_id": snapshot_id,
                    "source": source,
                    "request_params_json": params_json,
                    "fetched_at_utc": fetched_at,
                    "payload_hash": payload_hash,
                    "parser_version": config.parser_version,
                    "payload_path": str(payload_path),
                }
            ]
        ),
        RAW_SNAPSHOT_COLUMNS,
    )
    _append_table(store, "raw_snapshots", row)
    return snapshot_id


def _read_fixture(config: MlbConfig, name: str) -> bytes:
    path = Path(config.fixture_dir) / "raw" / name
    return path.read_bytes()


def _lineup_rows_from_pregame(
    pregame: pd.DataFrame, game_pk: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = pregame.loc[pregame["game_pk"] == game_pk]
    for rec in subset.to_dict(orient="records"):
        raw = rec.get("lineup_batter_ids_json") or "[]"
        try:
            ids = json.loads(raw) if not isinstance(raw, list) else raw
        except json.JSONDecodeError:
            ids = []
        opponent = int(rec["opponent_team_id"])
        side = "away" if int(rec["is_home"]) == 1 else "home"
        for slot, batter_id in enumerate(ids, start=1):
            rows.append(
                {
                    "game_pk": int(game_pk),
                    "team_id": opponent,
                    "batter_id": int(batter_id),
                    "batting_slot": slot,
                    "side": side,
                }
            )
    return rows


def ingest_statcast(
    config: MlbConfig,
    *,
    start_date: str,
    end_date: str,
    http: HttpFn | None = None,
) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "game_date_gt": str(start_date),
        "game_date_lt": str(end_date),
    }
    if http is None:
        payload = _read_fixture(config, "statcast_sample.csv")
    else:
        payload = http(STATCAST_URL, params)
    fetched_at = _now_utc()
    snapshot_id = snapshot_raw(config, "statcast", params, payload, fetched_at)
    pitches, plate_appearances = parse_statcast(payload, snapshot_id, fetched_at)
    store = MlbStore(config)
    _append_table(store, "pitch_events", coerce_frame(pitches, PITCH_EVENT_COLUMNS))
    _append_table(
        store,
        "plate_appearances",
        coerce_frame(plate_appearances, PLATE_APPEARANCE_COLUMNS),
    )
    return pitches


def ingest_schedule(
    config: MlbConfig,
    *,
    game_date: str,
    http: HttpFn | None = None,
) -> pd.DataFrame:
    params = {
        "sportId": 1,
        "date": str(game_date),
        "hydrate": "probablePitcher,venue,team",
    }
    if http is None:
        payload = _read_fixture(config, "schedule.json")
    else:
        payload = http(SCHEDULE_URL, params)
    fetched_at = _now_utc()
    snapshot_id = snapshot_raw(config, "schedule", params, payload, fetched_at)
    versions = parse_schedule(payload, snapshot_id, fetched_at)
    store = MlbStore(config)
    _append_table(store, "game_versions", versions)
    return versions


def ingest_lineups(
    config: MlbConfig,
    *,
    game_pk: int,
    http: HttpFn | None = None,
) -> pd.DataFrame:
    params = {"game_pk": int(game_pk)}
    payload: bytes | None = None
    if http is None:
        raw_dir = Path(config.fixture_dir) / "raw"
        for name in (f"lineups_{game_pk}.json", "lineups.json"):
            path = raw_dir / name
            if path.exists():
                payload = path.read_bytes()
                break
        if payload is None:
            from src.mlb.fixtures import load_fixture_tables

            tables = load_fixture_tables(config.fixture_dir)
            pregame = tables["pregame_snapshots"]
            rows = _lineup_rows_from_pregame(pregame, int(game_pk))
            return pd.DataFrame(rows)
    else:
        payload = http(LINEUP_URL_TEMPLATE.format(game_pk=int(game_pk)), params)
    fetched_at = _now_utc()
    snapshot_raw(config, "lineups", params, payload, fetched_at)
    return parse_lineups(payload, game_pk=int(game_pk))


def ingest_chadwick(
    config: MlbConfig,
    *,
    http: HttpFn | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {}
    if http is None:
        payload = _read_fixture(config, "people.csv")
    else:
        payload = http(CHADWICK_URL, None)
    fetched_at = _now_utc()
    snapshot_raw(config, "chadwick", params, payload, fetched_at)
    people = parse_people(payload)
    store = MlbStore(config)
    store.write_table("id_map", coerce_frame(people, ID_MAP_COLUMNS))
    return people
