from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from line_movement.paths import repo_root

PACIFIC = ZoneInfo("America/Los_Angeles")
_SNAPSHOT_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{6})")

# Kind is the top-level list that holds player props in that book's JSON.
BOOKS: tuple[tuple[str, str, str], ...] = (
    ("fanduel", "data/props/fanduel/nba", "picks"),
    ("draftkings", "data/props/draftkings/nba", "picks"),
    ("pinnacle", "data/props/pinnacle/nba", "games"),
    ("novig", "data/props/novig/nba", "games"),
    ("prophetx", "data/props/prophetx/nba", "games"),
    ("underdog", "data/props/underdogs/nba", "picks"),
    ("prizepicks", "data/props/prizepicks/nba", "projections"),
)


def snapshot_at(path: Path) -> datetime:
    """Parse the Pacific snapshot clock encoded in a source filename."""
    match = _SNAPSHOT_STAMP.search(path.name)
    if match is None:
        raise ValueError(f"no snapshot stamp in {path.name}")
    naive = datetime.strptime(
        f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H%M%S"
    )
    return naive.replace(tzinfo=PACIFIC)


def prop_files(folder: Path, date: str, kind: str) -> list[Path]:
    files = [path for path in folder.glob(f"*{date}*.json") if path.is_file()]
    if kind == "games":
        files = [path for path in files if path.name.endswith("_props.json")]
    else:
        files = [path for path in files if not path.name.endswith("_team.json")]
    if not files:
        raise FileNotFoundError(f"no {date} prop json in {folder}")
    return sorted(files)


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _side(value: object) -> str | None:
    text = _text(value)
    return text.lower() if text else None


def _american(prop: dict, side: str):
    flat = prop.get(f"american_{side}")
    if flat is not None:
        return flat
    nested = prop.get(side)
    if isinstance(nested, dict):
        return nested.get("american")
    return None


def _pick_rows(picks: list[dict]) -> list[dict]:
    rows = []
    for pick in picks:
        rows.append(
            {
                "player": _text(pick.get("full_name")),
                "stat": _text(pick.get("stat_name")),
                "line": pick.get("stat_value"),
                "side": _side(pick.get("choice")),
                "american_price": pick.get("american_price"),
                "decimal_price": pick.get("decimal_price"),
                "implied_probability": pick.get("implied_probability"),
                "payout_multiplier": pick.get("payout_multiplier"),
                "market_name": _text(pick.get("market_name")),
                "event_name": _text(pick.get("event_name")),
                "start_time": pick.get("start_time"),
                "line_updated_at": pick.get("updated_at"),
                "is_main": None,
                "odds_type": None,
            }
        )
    return rows


def _projection_rows(projections: list[dict]) -> list[dict]:
    rows = []
    for projection in projections:
        rows.append(
            {
                "player": _text(projection.get("player")),
                "stat": _text(projection.get("stat_type")),
                "line": projection.get("line_score"),
                "side": None,
                "american_price": None,
                "market_name": None,
                "event_name": None,
                "start_time": None,
                "line_updated_at": projection.get("updated_at"),
                "is_main": None,
                "odds_type": _text(projection.get("odds_type")),
            }
        )
    return rows


def _game_rows(games: list[dict]) -> list[dict]:
    rows = []
    for game in games:
        event_name = _text(game.get("name") or game.get("label"))
        start_time = game.get("scheduled") or game.get("start_time")
        for prop in game.get("props") or []:
            base = {
                "player": _text(prop.get("player")),
                "stat": _text(prop.get("stat")),
                "line": prop.get("line"),
                "market_name": _text(prop.get("sub_type")),
                "event_name": event_name,
                "start_time": start_time,
                "line_updated_at": None,
                "is_main": prop.get("is_main"),
                "odds_type": None,
            }
            priced = False
            for side in ("over", "under"):
                price = _american(prop, side)
                if price is None:
                    continue
                priced = True
                rows.append({**base, "side": side, "american_price": price})
            if not priced:
                rows.append({**base, "side": None, "american_price": None})
    return rows


def flatten_payload(payload: dict, kind: str) -> list[dict]:
    if kind == "picks":
        return _pick_rows(payload.get("picks") or [])
    if kind == "projections":
        return _projection_rows(payload.get("projections") or [])
    if kind == "games":
        return _game_rows(payload.get("games") or [])
    raise ValueError(f"unknown prop kind: {kind}")


def load_book(book: str, folder: Path, kind: str, date: str) -> pd.DataFrame:
    frames = []
    for path in prop_files(folder, date, kind):
        payload = json.loads(path.read_text())
        frame = pd.DataFrame(flatten_payload(payload, kind))
        if frame.empty:
            continue
        frame.insert(0, "book", book)
        frame.insert(1, "snapshot_at", snapshot_at(path))
        frame.insert(2, "fetched_at", payload.get("fetched_at"))
        frame.insert(3, "source_file", path.name)
        frames.append(frame)
    if not frames:
        raise ValueError(f"{book} files for {date} contained no props")
    out = pd.concat(frames, ignore_index=True)
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    prices = out["american_price"].astype("string").str.replace("+", "", regex=False)
    out["american_price"] = pd.to_numeric(prices, errors="coerce")
    out["start_time"] = pd.to_datetime(out["start_time"], utc=True, errors="coerce")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], utc=True, errors="coerce")
    out["line_updated_at"] = pd.to_datetime(
        out["line_updated_at"], utc=True, errors="coerce"
    )
    return out


def load_all_books(date: str, *, root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root()
    frames = [
        load_book(book, root / rel, kind, date) for book, rel, kind in BOOKS
    ]
    return pd.concat(frames, ignore_index=True)
