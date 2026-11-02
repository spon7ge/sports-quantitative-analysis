"""Newest points board from each saved bookmaker snapshot.

The slate date is the game's Pacific date. Snapshot files from an earlier
fetch are ignored. Closing files are ignored. The returned markets are the
same two-way main points lines used in the closing backtest.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
from src.models.dynamic_points_closing import select_main_points
from src.scrapers.nba.nba_lineups import NBA_TEAM_ABBREVIATIONS

_PACIFIC = "America/Los_Angeles"
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{6})")
_POINTS = frozenset({"points", "pts"})
_BOOKS = (
    ("prizepicks", "prizepicks/nba", "projections"),
    ("underdog", "underdogs/nba", "picks"),
    ("novig", "novig/nba", "games"),
    ("prophetx", "prophetx/nba", "games"),
    ("pinnacle", "pinnacle/nba", "games"),
    ("fanduel", "fanduel/nba", "picks"),
    ("draftkings", "draftkings/nba", "picks"),
)


def latest_points_markets(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Main points lines on the newest snapshot for every book.

    The second frame maps each player to one event label, preferring a
    label that marks the home team with ``@``.
    """
    props = Path(root)
    sides: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    files: dict[str, str] = {}
    missing: list[str] = []
    for book, relative, kind in _BOOKS:
        path = _newest_snapshot(props / relative)
        if path is None:
            missing.append(book)
            continue
        files[book] = path.name
        payload = json.loads(path.read_text())
        fetched = payload.get("fetched_at")
        for row in _points_rows(payload, kind):
            slate = _pacific_date(row.get("start_time") or fetched)
            sides.append(
                {
                    "slate_date": slate,
                    "book": book,
                    "player_name": row["player"],
                    "line": row["line"],
                    "side": row.get("side"),
                    "american_price": row.get("american_price"),
                }
            )
            if row.get("event_name"):
                events.append(
                    {
                        "player_name": row["player"],
                        "event_name": row["event_name"],
                        "slate_date": slate,
                    }
                )
    if not sides:
        raise ValueError("no latest points snapshots were found")
    markets, audit = select_main_points(pd.DataFrame(sides))
    audit = {
        **audit,
        "source_files": files,
        "missing_books": tuple(missing),
    }
    return markets, _player_events(events), audit


def event_context(event: object, team_abbreviation: object) -> tuple[str | None, float]:
    """Opponent abbreviation and home flag from ``Away @ Home``.

    A label without ``@`` does not identify the home team, so the home flag
    stays missing and the caller can impute it.
    """
    if event is None or (isinstance(event, float) and math.isnan(event)):
        text = ""
    else:
        text = str(event)
    team = "" if team_abbreviation is None else str(team_abbreviation).upper()
    if "@" not in text or not team:
        return None, float("nan")
    away_text, home_text = (part.strip() for part in text.split("@", 1))
    away = _abbreviation(away_text)
    home = _abbreviation(home_text)
    if team == home and away:
        return away, 1.0
    if team == away and home:
        return home, 0.0
    return None, float("nan")


def _newest_snapshot(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    stamped: list[tuple[str, Path]] = []
    for path in folder.glob("*.json"):
        if not path.is_file() or path.name.endswith("_team.json"):
            continue
        if "closing" in path.parts:
            continue
        match = _STAMP.search(path.name)
        if match is None:
            continue
        stamped.append((f"{match.group(1)}_{match.group(2)}", path))
    if not stamped:
        return None
    return max(stamped, key=lambda item: item[0])[1]


def _points_rows(payload: dict, kind: str) -> list[dict[str, object]]:
    if kind == "projections":
        return _projection_points(payload.get("projections") or [])
    if kind == "picks":
        return _pick_points(payload.get("picks") or [])
    if kind == "games":
        return _game_points(payload.get("games") or [])
    raise ValueError(f"unknown snapshot kind: {kind}")


def _projection_points(rows: list[dict]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for row in rows:
        stat = str(row.get("stat_type") or "").strip().lower()
        if stat not in _POINTS or row.get("odds_type") != "standard":
            continue
        kept.append(
            {
                "player": row.get("player"),
                "line": float(row["line_score"]),
                "side": None,
                "american_price": None,
                "event_name": None,
                "start_time": None,
            }
        )
    return kept


def _pick_points(rows: list[dict]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for row in rows:
        stat = str(row.get("stat_name") or "").strip().lower()
        if stat not in _POINTS:
            continue
        kept.append(
            {
                "player": row.get("full_name"),
                "line": float(row["stat_value"]),
                "side": str(row.get("choice") or "").lower() or None,
                "american_price": row.get("american_price"),
                "event_name": row.get("event_name"),
                "start_time": row.get("start_time"),
            }
        )
    return kept


def _game_points(games: list[dict]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for game in games:
        event = game.get("name") or game.get("label")
        start = game.get("scheduled") or game.get("start_time")
        for prop in game.get("props") or []:
            stat = str(prop.get("stat") or "").strip().lower()
            if stat not in _POINTS:
                continue
            base = {
                "player": prop.get("player"),
                "line": float(prop["line"]),
                "event_name": event,
                "start_time": start,
            }
            priced = False
            for side in ("over", "under"):
                price = prop.get(f"american_{side}")
                nested = prop.get(side)
                if price is None and isinstance(nested, dict):
                    price = nested.get("american")
                if price is None:
                    continue
                priced = True
                kept.append({**base, "side": side, "american_price": price})
            if not priced:
                kept.append({**base, "side": None, "american_price": None})
    return kept


def _pacific_date(value: object) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("a points snapshot needs a start time or fetch time")
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(_PACIFIC).normalize().tz_localize(None)


def _player_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["player_name", "event_name", "slate_date"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    frame["_home_marked"] = frame["event_name"].astype(str).str.contains("@")
    frame = frame.sort_values("_home_marked", ascending=False, kind="mergesort")
    kept = frame.drop_duplicates("player_name").drop(columns="_home_marked")
    return kept.reset_index(drop=True)


def _abbreviation(side: str) -> str | None:
    lowered = side.lower()
    matches = [
        (nickname, abbr)
        for nickname, abbr in NBA_TEAM_ABBREVIATIONS.items()
        if nickname.lower() in lowered
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]
