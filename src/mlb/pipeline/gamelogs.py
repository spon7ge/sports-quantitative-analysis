"""MLB Stats API people, schedule, and pitching game logs for the HF backtest."""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.pipeline.http import HttpFn
from src.mlb.schemas import (
    GAME_VERSION_COLUMNS,
    ID_MAP_COLUMNS,
    PITCHER_START_COLUMNS,
    coerce_frame,
    empty_frame,
)

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/sports/1/players"
PEOPLE_BATCH_URL = "https://statsapi.mlb.com/api/v1/people"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
TRAIN_SEASONS = tuple(range(2018, 2026))
TRAIN_END_DATE = "2025-12-31"
BATCH_SIZE = 40

HttpGetter = Callable[[str, dict | None], bytes]


def normalize_player_name(name: object) -> str:
    """Lowercase ASCII name without punctuation or generational suffixes."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text).strip()


def curl_http(url: str, params: dict | None = None, *, user_agent: str) -> bytes:
    """GET via curl so Stats API calls work when urllib SSL verify fails."""
    full = url
    if params:
        full = f"{url}?{urlencode(params, doseq=True)}"
    completed = subprocess.run(
        ["curl", "-gsS", "-A", user_agent, "--max-time", "90", full],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def default_http(config: MlbConfig) -> HttpFn:
    def _call(url: str, params: dict | None = None) -> bytes:
        return curl_http(url, params, user_agent=config.user_agent)

    return _call


def _load_json(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def fetch_people(
    http: HttpGetter,
    *,
    season: int,
) -> pd.DataFrame:
    payload = _load_json(http(PEOPLE_URL, {"season": season}))
    rows: list[dict[str, Any]] = []
    for person in payload.get("people") or []:
        position = person.get("primaryPosition") or {}
        rows.append(
            {
                "mlb_id": int(person["id"]),
                "key_mlbam": int(person["id"]),
                "key_fangraphs": pd.NA,
                "key_bbref": pd.NA,
                "key_retro": pd.NA,
                "name": person.get("fullName"),
                "bats": (person.get("batSide") or {}).get("code"),
                "throws": (person.get("pitchHand") or {}).get("code"),
                "is_pitcher": int(
                    str(position.get("abbreviation") or "").upper() == "P"
                ),
                "player_key": normalize_player_name(person.get("fullName")),
            }
        )
    if not rows:
        return empty_frame(ID_MAP_COLUMNS)
    frame = pd.DataFrame(rows)
    return frame


def parse_game_log_splits(
    splits: Iterable[dict[str, Any]],
    *,
    pitcher_id: int,
    pitcher_hand: str,
    season: int,
) -> list[dict[str, Any]]:
    """Convert Stats API pitching ``splits`` into pitcher-start dicts."""
    rows: list[dict[str, Any]] = []
    for split in splits:
        stat = split.get("stat") or {}
        games_started = int(stat.get("gamesStarted") or 0)
        if games_started < 1:
            continue
        game = split.get("game") or {}
        game_pk = game.get("gamePk")
        if game_pk is None:
            continue
        date = str(split.get("date") or "")
        team = split.get("team") or {}
        opponent = split.get("opponent") or {}
        innings = str(stat.get("inningsPitched") or "0")
        try:
            if "." in innings:
                whole, frac = innings.split(".", 1)
                outs = int(whole) * 3 + int(frac)
            else:
                outs = int(float(innings) * 3)
        except (TypeError, ValueError):
            outs = int(stat.get("outs") or 0)
        rows.append(
            {
                "pitcher_id": int(pitcher_id),
                "game_pk": int(game_pk),
                "game_date": date,
                "season": int(split.get("season") or season),
                "strikeouts": int(stat.get("strikeOuts") or 0),
                "batters_faced": int(stat.get("battersFaced") or 0),
                "pitches": int(stat.get("numberOfPitches") or 0),
                "outs": int(stat.get("outs") or outs),
                "role": "starter",
                "is_home": int(bool(split.get("isHome"))),
                "opponent_team_id": int(opponent.get("id") or 0),
                "team_id": int(team.get("id") or 0),
                "venue_id": 0,
                "pitcher_hand": pitcher_hand or "",
                "doubleheader": int(game.get("gameNumber") or 1) - 1,
                "player_key": "",
            }
        )
    return rows


def fetch_pitcher_game_logs(
    http: HttpGetter,
    pitcher_ids: list[int],
    *,
    season: int,
    hands: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Batch-hydrate pitching game logs and keep games started."""
    hands = hands or {}
    rows: list[dict[str, Any]] = []
    unique_ids = [int(i) for i in dict.fromkeys(pitcher_ids)]
    for start in range(0, len(unique_ids), BATCH_SIZE):
        chunk = unique_ids[start : start + BATCH_SIZE]
        ids = ",".join(str(i) for i in chunk)
        hydrate = f"stats(group=[pitching],type=[gameLog],season={int(season)})"
        url = f"{PEOPLE_BATCH_URL}?personIds={ids}&hydrate={hydrate}"
        payload = _load_json(http(url, None))
        for person in payload.get("people") or []:
            pitcher_id = int(person["id"])
            hand = str(
                hands.get(pitcher_id)
                or (person.get("pitchHand") or {}).get("code")
                or ""
            )
            for block in person.get("stats") or []:
                rows.extend(
                    parse_game_log_splits(
                        block.get("splits") or [],
                        pitcher_id=pitcher_id,
                        pitcher_hand=hand,
                        season=season,
                    )
                )
    if not rows:
        return empty_frame(PITCHER_START_COLUMNS)
    return pd.DataFrame(rows)


def parse_schedule_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            status = (game.get("status") or {}).get("abstractGameState")
            teams = game.get("teams") or {}
            home = (teams.get("home") or {}).get("team") or {}
            away = (teams.get("away") or {}).get("team") or {}
            venue = game.get("venue") or {}
            home_pitcher = (teams.get("home") or {}).get("probablePitcher") or {}
            away_pitcher = (teams.get("away") or {}).get("probablePitcher") or {}
            probable_home = home_pitcher.get("id")
            probable_away = away_pitcher.get("id")
            rows.append(
                {
                    "game_pk": int(game["gamePk"]),
                    "scheduled_start_utc": game.get("gameDate"),
                    "status": str(status or ""),
                    "home_team_id": int(home.get("id") or 0),
                    "away_team_id": int(away.get("id") or 0),
                    "venue_id": int(venue.get("id") or 0),
                    "doubleheader": int(str(game.get("doubleHeader") or "N") != "N"),
                    "probable_home_pitcher_id": probable_home,
                    "probable_away_pitcher_id": probable_away,
                    "official_date": game.get("officialDate") or day.get("date"),
                }
            )
    return rows


def fetch_schedule(
    http: HttpGetter,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    payload = _load_json(
        http(
            SCHEDULE_URL,
            {
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
            },
        )
    )
    rows = parse_schedule_games(payload)
    if not rows:
        return empty_frame(GAME_VERSION_COLUMNS)
    return pd.DataFrame(rows)


def attach_schedule_times(
    starts: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Fill scheduled start, venue, and cutoff-safe ingest times from schedule.

    Extra columns such as ``player_key`` are preserved for name matching.
    """
    if starts.empty:
        return starts.copy()
    frame = starts.copy()
    if schedule is not None and not schedule.empty:
        sched = schedule.drop_duplicates("game_pk").rename(
            columns={
                "scheduled_start_utc": "sched_start",
                "venue_id": "sched_venue",
                "doubleheader": "sched_dh",
            }
        )
        keep = [
            c
            for c in ("game_pk", "sched_start", "sched_venue", "sched_dh")
            if c in sched.columns
        ]
        frame = frame.merge(sched[keep], on="game_pk", how="left")
        if "sched_start" in frame.columns:
            frame["scheduled_start_utc"] = pd.to_datetime(
                frame["sched_start"], utc=True
            )
        if "sched_venue" in frame.columns:
            frame["venue_id"] = pd.to_numeric(
                frame["sched_venue"], errors="coerce"
            ).fillna(pd.to_numeric(frame.get("venue_id", 0), errors="coerce"))
        if "sched_dh" in frame.columns:
            frame["doubleheader"] = pd.to_numeric(
                frame["sched_dh"], errors="coerce"
            ).fillna(pd.to_numeric(frame.get("doubleheader", 0), errors="coerce"))
        frame = frame.drop(
            columns=["sched_start", "sched_venue", "sched_dh"], errors="ignore"
        )
    if "scheduled_start_utc" not in frame.columns:
        frame["scheduled_start_utc"] = pd.NaT
    scheduled = pd.to_datetime(frame["scheduled_start_utc"], utc=True)
    missing = scheduled.isna()
    if missing.any():
        fallback = pd.to_datetime(
            frame.loc[missing, "game_date"], utc=True
        ) + pd.Timedelta(hours=17)
        scheduled = scheduled.copy()
        scheduled.loc[missing] = fallback
    frame["scheduled_start_utc"] = scheduled
    frame["event_time_utc"] = scheduled
    frame["ingested_at_utc"] = scheduled
    if "game_date" not in frame.columns or frame["game_date"].isna().any():
        frame["game_date"] = scheduled.dt.strftime("%Y-%m-%d")
    frame["venue_id"] = pd.to_numeric(frame.get("venue_id", 0), errors="coerce").fillna(
        0
    )
    frame["doubleheader"] = pd.to_numeric(
        frame.get("doubleheader", 0), errors="coerce"
    ).fillna(0)
    extra = [c for c in frame.columns if c not in PITCHER_START_COLUMNS]
    coerced = coerce_frame(frame, PITCHER_START_COLUMNS)
    for column in extra:
        coerced[column] = frame[column].to_numpy()
    return coerced


def cache_path(config: MlbConfig, name: str) -> Path:
    path = config.raw_dir / "mlb_gamelogs"
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def load_or_fetch_people(
    config: MlbConfig,
    http: HttpGetter,
    *,
    seasons: tuple[int, ...] = TRAIN_SEASONS,
) -> pd.DataFrame:
    dest = cache_path(config, f"people_{seasons[0]}_{seasons[-1]}.parquet")
    if dest.exists():
        return pd.read_parquet(dest)
    frames = [fetch_people(http, season=season) for season in seasons]
    people = pd.concat(frames, ignore_index=True)
    people = people.drop_duplicates("mlb_id", keep="last")
    people.to_parquet(dest, index=False)
    return people


def _schedule_windows(seasons: tuple[int, ...]) -> tuple[tuple[str, str], ...]:
    windows: list[tuple[str, str]] = []
    for year in seasons:
        windows.append((f"{year}-03-01", f"{year}-07-15"))
        windows.append((f"{year}-07-16", f"{year}-11-15"))
    return tuple(windows)


def load_or_fetch_schedule(
    config: MlbConfig,
    http: HttpGetter,
    *,
    seasons: tuple[int, ...] = TRAIN_SEASONS,
) -> pd.DataFrame:
    dest = cache_path(config, f"schedule_{seasons[0]}_{seasons[-1]}.parquet")
    if dest.exists():
        return pd.read_parquet(dest)
    frames = [
        fetch_schedule(http, start_date=start, end_date=end)
        for start, end in _schedule_windows(seasons)
    ]
    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule.drop_duplicates("game_pk", keep="last")
    schedule.to_parquet(dest, index=False)
    return schedule


def load_or_fetch_starter_logs(
    config: MlbConfig,
    http: HttpGetter,
    people: pd.DataFrame,
    *,
    seasons: tuple[int, ...] = TRAIN_SEASONS,
) -> pd.DataFrame:
    dest = cache_path(config, f"starter_logs_{seasons[0]}_{seasons[-1]}.parquet")
    if dest.exists():
        return pd.read_parquet(dest)
    pitchers = people.loc[people["is_pitcher"] == 1, "mlb_id"].astype(int).tolist()
    hands = {
        int(row.mlb_id): str(row.throws or "")
        for row in people.itertuples(index=False)
    }
    frames = [
        fetch_pitcher_game_logs(http, pitchers, season=season, hands=hands)
        for season in seasons
    ]
    logs = pd.concat(frames, ignore_index=True)
    if not logs.empty:
        name_map = people.set_index("mlb_id")["player_key"]
        logs["player_key"] = logs["pitcher_id"].map(name_map).fillna("")
        logs = logs.loc[logs["game_date"].astype(str) <= TRAIN_END_DATE].copy()
    logs.to_parquet(dest, index=False)
    return logs


def game_versions_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return empty_frame(GAME_VERSION_COLUMNS)
    frame = schedule.copy()
    start = pd.to_datetime(frame["scheduled_start_utc"], utc=True)
    frame["scheduled_start_utc"] = start
    frame["valid_from_utc"] = start - pd.Timedelta(days=7)
    frame["valid_to_utc"] = pd.NaT
    frame["snapshot_id"] = "mlb-schedule-" + frame["game_pk"].astype(str)
    return coerce_frame(frame, GAME_VERSION_COLUMNS)


def id_map_from_people(people: pd.DataFrame) -> pd.DataFrame:
    if people.empty:
        return empty_frame(ID_MAP_COLUMNS)
    return coerce_frame(people, ID_MAP_COLUMNS)
