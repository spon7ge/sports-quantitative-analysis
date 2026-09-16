"""Parse Statcast CSV, schedule JSON, lineups, and Chadwick people files."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from src.mlb.schemas import (
    GAME_VERSION_COLUMNS,
    ID_MAP_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
    coerce_frame,
    empty_frame,
)

SWING_PATTERN = r"swinging|foul|hit_into_play"
WHIFF_PATTERN = r"swinging_strike"
CALLED_STRIKE_PATTERN = r"called_strike"
STRIKEOUT_PATTERN = r"strikeout"
ZONE_HALF_WIDTH = 0.83
ZONE_Z_LOW = 1.5
ZONE_Z_HIGH = 3.5

COLUMN_ALIASES = {
    "pitcher": "pitcher_id",
    "batter": "batter_id",
    "stand": "batter_stand",
    "p_throws": "pitcher_hand",
    "description": "pitch_result",
    "events": "pa_result",
}

LINEUP_COLUMNS = ("game_pk", "team_id", "batter_id", "batting_slot", "side")


def _as_text(raw_payload: str | bytes) -> str:
    if isinstance(raw_payload, bytes):
        return raw_payload.decode("utf-8-sig")
    return raw_payload


def _as_utc(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if isinstance(ts, pd.Series):
        return ts
    return pd.Timestamp(ts)


def _opt_int(value: Any) -> Any:
    if value is None or value == "" or pd.isna(value):
        return pd.NA
    return int(float(value))


def _series_or_default(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    return pd.to_numeric(_series_or_default(frame, name, default), errors="coerce")


def parse_statcast(
    raw_payload: str | bytes,
    snapshot_id: str,
    ingested_at: datetime | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = _as_text(raw_payload).strip()
    if not text:
        return empty_frame(PITCH_EVENT_COLUMNS), empty_frame(PLATE_APPEARANCE_COLUMNS)

    frame = pd.read_csv(pd.io.common.StringIO(text))
    if frame.empty:
        return empty_frame(PITCH_EVENT_COLUMNS), empty_frame(PLATE_APPEARANCE_COLUMNS)

    had_description = "description" in frame.columns
    renamed = {src: dst for src, dst in COLUMN_ALIASES.items() if src in frame.columns}
    frame = frame.rename(columns=renamed)
    if "pitcher_id" not in frame.columns and "pitcher" in frame.columns:
        frame["pitcher_id"] = frame["pitcher"]
    if "batter_id" not in frame.columns and "batter" in frame.columns:
        frame["batter_id"] = frame["batter"]

    frame = frame.dropna(subset=["game_pk"])
    if frame.empty:
        return empty_frame(PITCH_EVENT_COLUMNS), empty_frame(PLATE_APPEARANCE_COLUMNS)

    ingested = _as_utc(ingested_at)
    if "event_time_utc" in frame.columns:
        event_time = pd.to_datetime(frame["event_time_utc"], utc=True)
    elif "game_date" in frame.columns:
        event_time = pd.to_datetime(frame["game_date"], utc=True)
    else:
        event_time = pd.Series(ingested, index=frame.index)

    at_bat_number = (
        pd.to_numeric(_series_or_default(frame, "at_bat_number", 1), errors="coerce")
        .fillna(1)
        .astype("int64")
    )
    pitch_number = (
        pd.to_numeric(_series_or_default(frame, "pitch_number", 1), errors="coerce")
        .fillna(1)
        .astype("int64")
    )
    game_pk = pd.to_numeric(frame["game_pk"], errors="coerce")
    pitcher_id = pd.to_numeric(
        _series_or_default(frame, "pitcher_id", pd.NA), errors="coerce"
    )
    batter_id = pd.to_numeric(
        _series_or_default(frame, "batter_id", pd.NA), errors="coerce"
    )

    if "pitch_id" in frame.columns:
        pitch_id = frame["pitch_id"].astype(str)
    else:
        pitch_id = (
            game_pk.astype("Int64").astype(str)
            + "-"
            + at_bat_number.astype(str)
            + "-"
            + pitch_number.astype(str)
        )

    plate_x = _numeric(frame, "plate_x", pd.NA)
    plate_z = _numeric(frame, "plate_z", pd.NA)
    in_zone = (
        (plate_x.abs() <= ZONE_HALF_WIDTH)
        & (plate_z >= ZONE_Z_LOW)
        & (plate_z <= ZONE_Z_HIGH)
    ).fillna(False)

    description = None
    if had_description and "pitch_result" in frame.columns:
        description = frame["pitch_result"].fillna("").astype(str).str.lower()

    if description is not None and description.str.len().gt(0).any():
        is_swing = description.str.contains(SWING_PATTERN, regex=True, na=False)
        is_whiff = description.str.contains(WHIFF_PATTERN, regex=True, na=False)
        is_called_strike = description.str.contains(
            CALLED_STRIKE_PATTERN, regex=True, na=False
        )
    else:
        is_swing = (
            pd.to_numeric(_series_or_default(frame, "is_swing", 0), errors="coerce")
            .fillna(0)
            .astype(bool)
        )
        is_whiff = (
            pd.to_numeric(_series_or_default(frame, "is_whiff", 0), errors="coerce")
            .fillna(0)
            .astype(bool)
        )
        is_called_strike = (
            pd.to_numeric(
                _series_or_default(frame, "is_called_strike", 0), errors="coerce"
            )
            .fillna(0)
            .astype(bool)
        )

    is_chase = is_swing & ~in_zone

    if "pitch_result" in frame.columns:
        pitch_result = frame["pitch_result"].astype(str)
    elif description is not None:
        pitch_result = description
    else:
        pitch_result = pd.Series("", index=frame.index)

    if "pa_result" in frame.columns:
        pa_result = frame["pa_result"].fillna("").astype(str)
    else:
        pa_result = pd.Series("", index=frame.index)

    if "ingested_at_utc" in frame.columns:
        ingested_at_utc = pd.to_datetime(frame["ingested_at_utc"], utc=True)
    else:
        ingested_at_utc = pd.Series(ingested, index=frame.index)

    if "source_vintage" in frame.columns:
        source_vintage = frame["source_vintage"].astype(str)
    else:
        source_vintage = pd.Series("statcast", index=frame.index)

    pitcher_hand = _series_or_default(frame, "pitcher_hand", "").astype(str)
    batter_stand = _series_or_default(frame, "batter_stand", "").astype(str)

    pitches = pd.DataFrame(
        {
            "game_pk": game_pk,
            "pitch_id": pitch_id,
            "at_bat_number": at_bat_number,
            "pitch_number": pitch_number,
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "event_time_utc": event_time,
            "pitch_result": pitch_result,
            "pa_result": pa_result,
            "pitcher_hand": pitcher_hand,
            "batter_stand": batter_stand,
            "release_speed": _numeric(frame, "release_speed", pd.NA),
            "pfx_x": _numeric(frame, "pfx_x", pd.NA),
            "pfx_z": _numeric(frame, "pfx_z", pd.NA),
            "plate_x": plate_x,
            "plate_z": plate_z,
            "pitch_type": _series_or_default(frame, "pitch_type", "").astype(str),
            "is_swing": is_swing.astype("int64"),
            "is_whiff": is_whiff.astype("int64"),
            "is_called_strike": is_called_strike.astype("int64"),
            "is_in_zone": in_zone.astype("int64"),
            "is_chase": is_chase.astype("int64"),
            "source_vintage": source_vintage,
            "ingested_at_utc": ingested_at_utc,
            "snapshot_id": snapshot_id,
        }
    )
    pitches = pitches.dropna(subset=["pitcher_id", "batter_id"])
    pitches = coerce_frame(pitches, PITCH_EVENT_COLUMNS)

    if pitches.empty:
        return pitches, empty_frame(PLATE_APPEARANCE_COLUMNS)

    ordered = pitches.sort_values(
        ["game_pk", "at_bat_number", "pitch_number", "event_time_utc"]
    )
    last_pitch = ordered.groupby(["game_pk", "at_bat_number"], as_index=False).tail(1)
    pitch_counts = (
        ordered.groupby(["game_pk", "at_bat_number"], as_index=False)
        .size()
        .rename(columns={"size": "pitches_in_pa"})
    )
    last_pitch = last_pitch.merge(
        pitch_counts, on=["game_pk", "at_bat_number"], how="left"
    )
    last_pitch = last_pitch.sort_values(["game_pk", "pitcher_id", "at_bat_number"])
    pa_index = last_pitch.groupby(["game_pk", "pitcher_id"]).cumcount()
    tto_number = (pa_index // 9) + 1
    batting_slot = ((last_pitch["at_bat_number"].astype("int64") - 1) % 9) + 1
    result = last_pitch["pa_result"].fillna("").astype(str)
    is_strikeout = (
        result.str.lower().str.contains(STRIKEOUT_PATTERN, regex=True, na=False)
    ).astype("int64")
    pa_id = (
        last_pitch["game_pk"].astype(str)
        + "-"
        + last_pitch["pitcher_id"].astype(str)
        + "-"
        + last_pitch["at_bat_number"].astype(str)
    )
    plate_appearances = pd.DataFrame(
        {
            "game_pk": last_pitch["game_pk"],
            "pa_id": pa_id,
            "at_bat_number": last_pitch["at_bat_number"],
            "pitcher_id": last_pitch["pitcher_id"],
            "batter_id": last_pitch["batter_id"],
            "result": result,
            "is_strikeout": is_strikeout,
            "batting_slot": batting_slot.astype("int64"),
            "pitcher_hand": last_pitch["pitcher_hand"],
            "batter_stand": last_pitch["batter_stand"],
            "tto_number": tto_number.astype("int64"),
            "pitches_in_pa": last_pitch["pitches_in_pa"].fillna(1).astype("int64"),
            "event_time_utc": last_pitch["event_time_utc"],
            "ingested_at_utc": last_pitch["ingested_at_utc"],
            "snapshot_id": snapshot_id,
        }
    )
    return pitches, coerce_frame(plate_appearances, PLATE_APPEARANCE_COLUMNS)


def parse_schedule(
    raw_payload: str | bytes,
    snapshot_id: str,
    ingested_at: datetime | pd.Timestamp,
) -> pd.DataFrame:
    data = json.loads(_as_text(raw_payload))
    ingested = _as_utc(ingested_at)
    rows: list[dict[str, Any]] = []
    fixture_games = (
        isinstance(data, dict)
        and isinstance(data.get("games"), list)
        and "dates" not in data
    )
    if fixture_games:
        for game in data["games"]:
            rows.append(_schedule_row_from_flat(game, snapshot_id, ingested))
    else:
        dates = data.get("dates", []) if isinstance(data, dict) else []
        for date_block in dates:
            for game in date_block.get("games", []):
                rows.append(_schedule_row_from_statsapi(game, snapshot_id, ingested))
    if not rows:
        return empty_frame(GAME_VERSION_COLUMNS)
    return coerce_frame(pd.DataFrame(rows), GAME_VERSION_COLUMNS)


def _schedule_row_from_flat(
    game: dict[str, Any],
    snapshot_id: str,
    ingested: pd.Timestamp,
) -> dict[str, Any]:
    valid_from = game.get("valid_from_utc") or ingested
    valid_to = game.get("valid_to_utc")
    if valid_to == "":
        valid_to = pd.NaT
    return {
        "game_pk": _opt_int(game.get("game_pk")),
        "scheduled_start_utc": game.get("scheduled_start_utc") or ingested,
        "status": str(game.get("status", "scheduled")),
        "home_team_id": _opt_int(game.get("home_team_id")),
        "away_team_id": _opt_int(game.get("away_team_id")),
        "venue_id": _opt_int(game.get("venue_id")),
        "doubleheader": int(float(game.get("doubleheader") or 0)),
        "probable_home_pitcher_id": _opt_int(game.get("probable_home_pitcher_id")),
        "probable_away_pitcher_id": _opt_int(game.get("probable_away_pitcher_id")),
        "valid_from_utc": valid_from,
        "valid_to_utc": valid_to,
        "snapshot_id": snapshot_id,
    }


def _schedule_row_from_statsapi(
    game: dict[str, Any],
    snapshot_id: str,
    ingested: pd.Timestamp,
) -> dict[str, Any]:
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    status = game.get("status") or {}
    double_header = str(game.get("doubleHeader", "N")).upper()
    doubleheader = 0 if double_header in {"N", "N/A", ""} else 1
    if str(game.get("gameNumber", "1")) == "2":
        doubleheader = 2
    state = status.get("detailedState") or status.get("abstractGameState")
    home_pitcher = (home.get("probablePitcher") or {}).get("id")
    away_pitcher = (away.get("probablePitcher") or {}).get("id")
    return {
        "game_pk": _opt_int(game.get("gamePk")),
        "scheduled_start_utc": game.get("gameDate") or ingested,
        "status": str(state or "scheduled"),
        "home_team_id": _opt_int((home.get("team") or {}).get("id")),
        "away_team_id": _opt_int((away.get("team") or {}).get("id")),
        "venue_id": _opt_int((game.get("venue") or {}).get("id")),
        "doubleheader": doubleheader,
        "probable_home_pitcher_id": _opt_int(home_pitcher),
        "probable_away_pitcher_id": _opt_int(away_pitcher),
        "valid_from_utc": ingested,
        "valid_to_utc": pd.NaT,
        "snapshot_id": snapshot_id,
    }


def parse_lineups(raw_payload: str | bytes, game_pk: int | None = None) -> pd.DataFrame:
    data = json.loads(_as_text(raw_payload))
    if isinstance(data, list):
        frame = pd.DataFrame(data)
        keep = [col for col in LINEUP_COLUMNS if col in frame.columns]
        return frame[keep] if keep else pd.DataFrame(columns=list(LINEUP_COLUMNS))
    if isinstance(data, dict) and isinstance(data.get("players"), list):
        return pd.DataFrame(data["players"])

    resolved_pk = game_pk
    if resolved_pk is None and isinstance(data, dict):
        game_info = (data.get("gameData") or {}).get("game", {})
        resolved_pk = data.get("gamePk") or game_info.get("pk")

    box: dict[str, Any] = {}
    if isinstance(data, dict):
        live = data.get("liveData") or {}
        box = live.get("boxscore") or data.get("boxscore") or data

    teams = box.get("teams") if isinstance(box, dict) else None
    if not teams:
        return pd.DataFrame(columns=list(LINEUP_COLUMNS))

    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = teams.get(side) or {}
        team_id = (team.get("team") or {}).get("id")
        order = team.get("battingOrder") or []
        for slot, player_key in enumerate(order, start=1):
            batter_id = int(str(player_key).replace("ID", ""))
            rows.append(
                {
                    "game_pk": resolved_pk,
                    "team_id": team_id,
                    "batter_id": batter_id,
                    "batting_slot": slot,
                    "side": side,
                }
            )
    return pd.DataFrame(rows, columns=list(LINEUP_COLUMNS))


def parse_people(raw_payload: str | bytes) -> pd.DataFrame:
    text = _as_text(raw_payload).strip()
    if not text:
        return empty_frame(ID_MAP_COLUMNS)
    frame = pd.read_csv(pd.io.common.StringIO(text))
    if frame.empty:
        return empty_frame(ID_MAP_COLUMNS)
    if "key_mlbam" not in frame.columns:
        first = frame.columns[0]
        frame = frame.rename(columns={first: "key_mlbam"})
    mlbam = pd.to_numeric(frame["key_mlbam"], errors="coerce")
    fangraphs = frame["key_fangraphs"] if "key_fangraphs" in frame.columns else pd.NA
    out = pd.DataFrame(
        {
            "mlb_id": mlbam,
            "key_mlbam": mlbam,
            "key_fangraphs": pd.to_numeric(fangraphs, errors="coerce"),
            "key_bbref": frame["key_bbref"] if "key_bbref" in frame.columns else pd.NA,
            "key_retro": frame["key_retro"] if "key_retro" in frame.columns else pd.NA,
            "name": frame["name"] if "name" in frame.columns else pd.NA,
            "bats": frame["bats"] if "bats" in frame.columns else pd.NA,
            "throws": frame["throws"] if "throws" in frame.columns else pd.NA,
        }
    )
    out = out.dropna(subset=["mlb_id"])
    return coerce_frame(out, ID_MAP_COLUMNS)
