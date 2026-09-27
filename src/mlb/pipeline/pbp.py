"""Parse Stats API play-by-play into the batter plate-appearance table."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.pipeline.http import HttpFn
from src.mlb.pipeline.ingest import snapshot_raw
from src.mlb.schemas import BATTER_PA_COLUMNS, coerce_frame
from src.mlb.storage import MlbStore

LIVE_FEED_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
PBP_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _as_text(raw_payload: str | bytes) -> str:
    if isinstance(raw_payload, bytes):
        return raw_payload.decode("utf-8-sig")
    return raw_payload


def _game_pk(data: dict[str, Any]) -> Any:
    game = (data.get("gameData") or {}).get("game") or {}
    return data.get("gamePk") if data.get("gamePk") is not None else game.get("pk")


def _has_plays(raw_payload: str | bytes) -> bool:
    data = json.loads(_as_text(raw_payload))
    if not isinstance(data, dict):
        return False
    live_data = data.get("liveData") or {}
    return (live_data.get("plays") or {}).get("allPlays") is not None or data.get(
        "allPlays"
    ) is not None


def _with_game_pk(raw_payload: str | bytes, game_pk: int) -> str | bytes:
    data = json.loads(_as_text(raw_payload))
    if not isinstance(data, dict) or _game_pk(data) is not None:
        return raw_payload
    data["gamePk"] = int(game_pk)
    return json.dumps(data)


def _payload_scheduled_start(raw_payload: str | bytes) -> Any:
    data = json.loads(_as_text(raw_payload))
    if not isinstance(data, dict):
        return None
    return ((data.get("gameData") or {}).get("datetime") or {}).get("dateTime")


def _stored_scheduled_start(
    game_versions: pd.DataFrame, game_pk: int
) -> pd.Timestamp:
    if game_versions.empty:
        return pd.NaT
    starts = pd.to_datetime(
        game_versions.loc[
            game_versions["game_pk"] == int(game_pk), "scheduled_start_utc"
        ],
        utc=True,
    )
    return starts.min()


def _local_payload(config: MlbConfig, game_pk: int) -> bytes:
    gzip_path = config.raw_dir / "mlb_pbp" / f"{game_pk}.json.gz"
    if gzip_path.exists():
        with gzip.open(gzip_path, "rb") as handle:
            return handle.read()

    fixture_dir = Path(config.fixture_dir) / "raw"
    for name in (f"{game_pk}.json", f"pbp_{game_pk}.json", "pbp.json"):
        fixture_path = fixture_dir / name
        if fixture_path.exists():
            return fixture_path.read_bytes()
    raise FileNotFoundError(f"No local play-by-play payload for game {game_pk}")


def ingest_play_by_play(
    config: MlbConfig,
    *,
    game_pks: list[int],
    http: HttpFn | None = None,
    people: pd.DataFrame | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Ingest game PAs, retaining the latest wall-clock version of each PA."""
    store = MlbStore(config)
    game_versions = store.read_table("game_versions")
    if people is None:
        people = store.read_table("id_map")
    frames: list[pd.DataFrame] = []
    for done, game_pk in enumerate(game_pks, start=1):
        if progress is not None:
            progress(done, int(game_pk))
        params = {"game_pk": int(game_pk)}
        ingested_at = _now_utc()
        if http is None:
            payload = _local_payload(config, int(game_pk))
        else:
            payload = http(
                LIVE_FEED_URL_TEMPLATE.format(game_pk=int(game_pk)),
                params,
            )
            if not _has_plays(payload):
                snapshot_raw(
                    config,
                    "mlb_pbp",
                    params,
                    payload,
                    ingested_at,
                )
                payload = http(
                    PBP_URL_TEMPLATE.format(game_pk=int(game_pk)),
                    params,
                )

        snapshot_id = snapshot_raw(
            config,
            "mlb_pbp",
            params,
            payload,
            ingested_at,
        )
        gzip_path = config.raw_dir / "mlb_pbp" / f"{game_pk}.json.gz"
        gzip_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gzip_path, "wb") as handle:
            handle.write(payload if isinstance(payload, bytes) else payload.encode())
        frames.append(
            parse_play_by_play(
                _with_game_pk(payload, int(game_pk)),
                snapshot_id,
                ingested_at,
                scheduled_start=(
                    _payload_scheduled_start(payload)
                    or _stored_scheduled_start(game_versions, int(game_pk))
                ),
                people=people,
            )
        )

    parsed = (
        pd.concat(frames, ignore_index=True)
        if frames
        else coerce_frame(pd.DataFrame(), BATTER_PA_COLUMNS)
    )
    if parsed.empty:
        return parsed

    combined = pd.concat([store.read_table("batter_pas"), parsed], ignore_index=True)
    combined = combined.sort_values("ingested_at_utc", kind="stable")
    combined = combined.drop_duplicates("pa_id", keep="last").reset_index(drop=True)
    store.write_table("batter_pas", coerce_frame(combined, BATTER_PA_COLUMNS))
    return parsed


def _batter_bats(people: pd.DataFrame | None) -> dict[int, str]:
    if people is None or not {"mlb_id", "bats"}.issubset(people.columns):
        return {}

    lookup: dict[int, str] = {}
    for mlb_id, bats in people[["mlb_id", "bats"]].itertuples(index=False):
        if pd.isna(mlb_id):
            continue
        lookup[int(mlb_id)] = "" if pd.isna(bats) else str(bats)
    return lookup


def parse_play_by_play(
    raw_payload: str | bytes,
    snapshot_id: str,
    ingested_at: datetime | pd.Timestamp,
    *,
    scheduled_start: datetime | pd.Timestamp | None = None,
    people: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Parse one live-feed or play-by-play payload into one row per PA."""
    data = json.loads(_as_text(raw_payload))
    if not isinstance(data, dict):
        return coerce_frame(pd.DataFrame(), BATTER_PA_COLUMNS)

    live_data = data.get("liveData") or {}
    plays = (live_data.get("plays") or {}).get("allPlays")
    if plays is None:
        plays = data.get("allPlays") or []

    game_pk = _game_pk(data)
    ingested = pd.to_datetime(ingested_at, utc=True)
    fallback_time = (
        pd.to_datetime(scheduled_start, utc=True) + pd.Timedelta(hours=4)
        if scheduled_start is not None
        else pd.NaT
    )
    bats_by_id = _batter_bats(people)
    rows: list[dict[str, Any]] = []

    for play in plays:
        matchup = play.get("matchup") or {}
        batter = matchup.get("batter") or {}
        pitcher = matchup.get("pitcher") or {}
        result = play.get("result") or {}
        about = play.get("about") or {}
        at_bat_index = play.get("atBatIndex")
        batter_id = batter.get("id")
        pitcher_id = pitcher.get("id")
        if None in (game_pk, at_bat_index, batter_id, pitcher_id):
            continue

        event_time = about.get("startTime") or about.get("endTime")
        event_time_imputed = int(event_time is None)
        if event_time is None:
            event_time = fallback_time

        pitch_hand = (matchup.get("pitchHand") or {}).get("code")
        if pitch_hand is None:
            pitch_hand = pitcher.get("p_throws") or ""

        rows.append(
            {
                "pa_id": f"{game_pk}_{at_bat_index}",
                "game_pk": game_pk,
                "at_bat_index": at_bat_index,
                "batter_id": batter_id,
                "pitcher_id": pitcher_id,
                "pitcher_hand": pitch_hand,
                "batter_bats": bats_by_id.get(int(batter_id), ""),
                "batter_stand": (matchup.get("batSide") or {}).get("code") or "",
                "event_type": result.get("eventType") or result.get("event") or "",
                "event_time_utc": event_time,
                "event_time_imputed": event_time_imputed,
                "is_pitcher_in_game": 0,
                "ingested_at_utc": ingested,
                "snapshot_id": snapshot_id,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        pitcher_ids = {
            game: set(group["pitcher_id"]) for game, group in frame.groupby("game_pk")
        }
        frame["is_pitcher_in_game"] = [
            int(batter_id in pitcher_ids[game])
            for game, batter_id in zip(
                frame["game_pk"], frame["batter_id"], strict=True
            )
        ]
    return coerce_frame(frame, BATTER_PA_COLUMNS)
