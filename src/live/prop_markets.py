"""Main points, rebounds, assists, steals, and blocks from one saved bookmaker."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.live.latest_points import (
    _BOOKS,
    _newest_snapshot,
    _pacific_date,
    _points_rows,
)
from src.models.dynamic_points_closing import select_main_points
from src.pipeline.silver.positions import player_name_keys

_STAT_NAMES = {
    "PTS": frozenset({"points", "pts"}),
    "REB": frozenset({"rebounds", "reb"}),
    "AST": frozenset({"assists", "ast"}),
    "STL": frozenset({"steals", "stl"}),
    "BLK": frozenset({"blocks", "blk", "blocked shots"}),
}
_ACCEPTED = frozenset().union(*_STAT_NAMES.values())
_MARKET_COLUMNS = [
    "slate_date",
    "book",
    "player_name",
    "stat",
    "line",
    "over_american",
    "under_american",
]
_ROSTER_COLUMNS = ("team", "side", "player_id", "player_name")


def book_names() -> tuple[str, ...]:
    """Books with a saved NBA prop snapshot folder."""
    return tuple(book for book, _, _ in _BOOKS)


def latest_prop_markets(
    root: str | Path, book: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """One main line per player for points, rebounds, assists, steals, and blocks.

    PrizePicks keeps the standard line and prices both sides at -119.
    Every other book keeps the two-way line whose over and under prices
    are closest. Combo markets such as points plus rebounds stay out.
    """
    requested = str(book).strip().lower()
    folders = {name: (relative, kind) for name, relative, kind in _BOOKS}
    if requested not in folders:
        known = ", ".join(book_names())
        raise ValueError(f"unknown book {book}; choose one of {known}")
    relative, kind = folders[requested]
    path = _newest_snapshot(Path(root) / relative)
    if path is None:
        raise ValueError(f"no {requested} snapshot in {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fetched = payload.get("fetched_at")
    raw = _points_rows(payload, kind, _ACCEPTED)
    frames: list[pd.DataFrame] = []
    for stat, names in _STAT_NAMES.items():
        sides = [
            {
                "slate_date": _pacific_date(row.get("start_time") or fetched),
                "book": requested,
                "player_name": row["player"],
                "line": row["line"],
                "side": row.get("side"),
                "american_price": row.get("american_price"),
            }
            for row in raw
            if row["stat"] in names
        ]
        markets, _audit = select_main_points(pd.DataFrame(sides))
        if markets.empty:
            continue
        markets = markets.copy()
        markets["stat"] = stat
        markets["over_american"] = markets["over_american"].round().astype(int)
        markets["under_american"] = markets["under_american"].round().astype(int)
        frames.append(markets)
    audit = {"book": requested, "source_file": path.name, "n": 0}
    if not frames:
        return pd.DataFrame(columns=_MARKET_COLUMNS), audit
    out = pd.concat(frames, ignore_index=True).loc[:, _MARKET_COLUMNS]
    audit["n"] = int(len(out))
    return out, audit


def attach_roster(markets: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Keep lines whose player is on this game's roster.

    Matching ignores accents and a trailing generational suffix. The
    returned name and id are the roster's, so they line up with the box.
    """
    missing = [column for column in _ROSTER_COLUMNS if column not in roster.columns]
    if missing:
        raise ValueError(f"roster needs {missing}")
    keys: dict[str, tuple[str, str, str, str]] = {}
    for record in roster.loc[:, list(_ROSTER_COLUMNS)].itertuples(index=False):
        player_id = _player_id(record.player_id)
        identity = (
            str(record.team),
            str(record.side),
            player_id,
            str(record.player_name),
        )
        for key in player_name_keys(record.player_name):
            previous = keys.get(key)
            if previous is not None and previous[2] != player_id:
                raise ValueError(f"two roster players share {record.player_name}")
            keys[key] = identity
    columns = [*_MARKET_COLUMNS, "team", "side", "player_id"]
    rows: list[dict[str, object]] = []
    if not markets.empty:
        for market in markets.itertuples(index=False):
            identity = None
            for key in player_name_keys(market.player_name):
                if key in keys:
                    identity = keys[key]
                    break
            if identity is None:
                continue
            team, side, player_id, player_name = identity
            rows.append(
                {
                    "slate_date": market.slate_date,
                    "book": market.book,
                    "player_name": player_name,
                    "stat": market.stat,
                    "line": market.line,
                    "over_american": market.over_american,
                    "under_american": market.under_american,
                    "team": team,
                    "side": side,
                    "player_id": player_id,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _player_id(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and float(number).is_integer():
        return str(int(number))
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("roster player id is blank")
    return text
