from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from game_state.constants import REQUIRED_PBP


def read_files(spec):
    paths = sorted(glob.glob(str(spec)))
    if not paths:
        raise FileNotFoundError(f"No files match {spec}")

    frames = [
        pd.read_parquet(path)
        if path.lower().endswith(".parquet")
        else pd.read_csv(path, dtype=str)
        for path in paths
    ]
    return pd.concat(frames, ignore_index=True)


def game_ids(values):
    ids = (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    if not ids.str.fullmatch(r"\d{10}", na=False).all():
        raise ValueError("Invalid NBA game IDs")
    return ids


def minutes_value(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    text = str(value).strip()
    if text.startswith("PT"):
        match = re.fullmatch(
            r"PT(?:(\d+)M)?(?:([\d.]+)S)?", text
        )
        if not match:
            return np.nan
        return float(match[1] or 0) + float(match[2] or 0) / 60

    try:
        if ":" in text:
            parts = text.split(":")
            return (
                float(parts[0]) + float(parts[1]) / 60
                if len(parts) == 2 else np.nan
            )
        return float(text)
    except ValueError:
        return np.nan


def load_player_logs(spec):
    d = read_files(spec)

    aliases = {
        "gameId": ["game_id", "GAME_ID", "Game_ID"],
        "personId": ["player_id", "PLAYER_ID", "Player_ID"],
        "teamId": ["team_id", "TEAM_ID"],
        "game_date": ["GAME_DATE"],
        "minutes": ["min", "MIN"],
        "playerName": ["player_name", "PLAYER_NAME"],
        "pf": ["PF"],
        "starter": ["is_starter"],
        "start_position": ["START_POSITION"],
        "spread": ["team_spread", "player_team_spread"],
        "total": ["game_total"],
    }
    for target, names in aliases.items():
        if target not in d:
            found = next(
                (name for name in names if name in d), None
            )
            if found:
                d = d.rename(columns={found: target})

    required = {
        "gameId", "personId", "teamId", "game_date", "minutes"
    }
    if not required.issubset(d):
        raise ValueError(
            f"Player logs need {sorted(required - set(d))}"
        )

    d["gameId"] = game_ids(d.gameId)
    d = d[d.gameId.str.startswith("002")].copy()
    d["game_date"] = pd.to_datetime(
        d.game_date, errors="coerce"
    ).dt.normalize()

    for name in ("personId", "teamId"):
        d[name] = pd.to_numeric(d[name], errors="coerce")
    d["minutes"] = d.minutes.map(minutes_value)

    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["game_date", "personId", "teamId"]
    )
    d[["personId", "teamId"]] = d[
        ["personId", "teamId"]
    ].astype("int64")
    d = d[d.personId.gt(0) & d.teamId.gt(0)].copy()

    for name in ("pf", "spread", "total", "team_poss"):
        d[name] = pd.to_numeric(
            d.get(name, np.nan), errors="coerce"
        )

    if "starter" in d:
        d["starter"] = pd.to_numeric(
            d.starter, errors="coerce"
        )
    elif "start_position" in d:
        d["starter"] = (
            d.start_position.fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(float)
        )
    else:
        d["starter"] = np.nan

    if "playerName" not in d:
        d["playerName"] = d.personId.astype(str)

    key = ["gameId", "teamId", "personId"]
    if d.groupby(key).minutes.nunique().gt(1).any():
        raise ValueError("Conflicting official minute labels")

    return d.drop_duplicates(key).sort_values(
        ["game_date", "gameId", "personId"]
    )


def load_pbp(spec):
    d = read_files(spec)
    if not REQUIRED_PBP.issubset(d):
        raise ValueError(
            f"PBP needs {sorted(REQUIRED_PBP - set(d))}"
        )

    d["gameId"] = game_ids(d.gameId)
    d = d[d.gameId.str.startswith("002")].copy()

    identifiers = ["actionId", "period", "teamId", "personId"]
    for name in identifiers:
        d[name] = pd.to_numeric(d[name], errors="coerce")
    if d[identifiers].isna().any().any():
        raise ValueError(
            "Missing/non-numeric PBP identifiers or periods"
        )
    d[identifiers] = d[identifiers].astype("int64")
    d["rem"] = d.clock.map(minutes_value) * 60

    for name in (
        "playerName", "description", "actionType",
        "subType", "location",
    ):
        d[name] = d[name].fillna("").astype(str)

    d["scoreHome"] = pd.to_numeric(
        d.scoreHome, errors="coerce"
    )
    d["scoreAway"] = pd.to_numeric(
        d.scoreAway, errors="coerce"
    )

    return d.drop_duplicates(
        ["gameId", "actionId"], keep="last"
    ).sort_values(["gameId", "period", "actionId"])
