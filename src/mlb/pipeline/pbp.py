"""Parse Stats API play-by-play into the batter plate-appearance table."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from src.mlb.schemas import BATTER_PA_COLUMNS, coerce_frame


def _as_text(raw_payload: str | bytes) -> str:
    if isinstance(raw_payload, bytes):
        return raw_payload.decode("utf-8-sig")
    return raw_payload


def _game_pk(data: dict[str, Any]) -> Any:
    game = (data.get("gameData") or {}).get("game") or {}
    return data.get("gamePk") if data.get("gamePk") is not None else game.get("pk")


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
            game: set(group["pitcher_id"])
            for game, group in frame.groupby("game_pk")
        }
        frame["is_pitcher_in_game"] = [
            int(batter_id in pitcher_ids[game])
            for game, batter_id in zip(
                frame["game_pk"], frame["batter_id"], strict=True
            )
        ]
    return coerce_frame(frame, BATTER_PA_COLUMNS)
