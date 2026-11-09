"""Slate inputs that are known at the close and are not final scores.

Layer 1 is conditioned on one book's closing spread and total. Layer 3
marks the Rotowire confirmed five as the starters. Opening lines and
opening lineups are not read. A missing close stays missing.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.models.odds import american_to_implied_probability
from src.scrapers.arbitrage import canonical_team, player_key

PACIFIC = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True)
class ClosingScript:
    """Home spread uses the negative-favorite convention. Total is the points line."""

    total: float
    home_spread: float


def load_team_closing_scripts(
    path: Path, slate
) -> tuple[dict[tuple[str, str], ClosingScript], dict[str, int]]:
    """Closing spread and total for each game. Keys are ``(away_id, home_id)``."""
    payload = _payload(Path(path))
    day = _require_slate(payload, Path(path), slate)
    audit = {
        "games": 0,
        "games_missing_line": 0,
        "timestamp_mismatch": 0,
        "unmatched_name": 0,
        "alternate_lines_dropped": 0,
        "opening_ignored": 0,
    }
    scripts: dict[tuple[str, str], ClosingScript] = {}
    for game in _games(payload, Path(path)):
        audit["games"] += 1
        if "opening_markets" in game:
            audit["opening_ignored"] += 1
        if _close_stamp(game, day, Path(path), audit) is None:
            continue
        parsed = _script(game, Path(path), audit)
        if parsed is None:
            audit["games_missing_line"] += 1
            continue
        key, script, dropped = parsed
        if key in scripts:
            raise ValueError(f"{Path(path).name} lists {key[0]} at {key[1]} twice")
        scripts[key] = script
        audit["alternate_lines_dropped"] += dropped
    return scripts, audit


def load_rotowire_closing_lineups(
    path: Path, slate
) -> tuple[
    dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]], dict[str, int]
]:
    """Confirmed names only. A side without a close keeps the game out."""
    payload = _payload(Path(path))
    day = _require_slate(payload, Path(path), slate)
    audit = {"games": 0, "sides_missing_close": 0, "timestamp_mismatch": 0}
    lineups: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for game in _games(payload, Path(path)):
        audit["games"] += 1
        away_id = _abbr_id(game.get("away"), Path(path))
        home_id = _abbr_id(game.get("home"), Path(path))
        if away_id is None or home_id is None or away_id == home_id:
            raise ValueError(f"{Path(path).name} has an unreadable matchup")
        away = _closing_names(game.get("away_closing"), day, Path(path), audit)
        home = _closing_names(game.get("home_closing"), day, Path(path), audit)
        if away is None or home is None:
            audit["sides_missing_close"] += 1
            continue
        key = (away_id, home_id)
        if key in lineups:
            raise ValueError(f"{Path(path).name} lists {away_id} at {home_id} twice")
        lineups[key] = (away, home)
    return lineups, audit


def match_starter_ids(
    roster: pd.DataFrame, names: Sequence[str]
) -> tuple[set[int], list[str]]:
    """Map a confirmed five onto roster ids. Names with no unique row stay unmatched."""
    if not isinstance(roster, pd.DataFrame):
        raise ValueError("roster must be a DataFrame")
    missing_columns = {"player_id", "player_name"} - set(roster.columns)
    if missing_columns:
        raise ValueError(f"roster needs {sorted(missing_columns)}")
    posted = [str(name).strip() for name in names if str(name).strip()]
    if len(posted) > 5:
        raise ValueError("a closing lineup lists more than five players")
    by_key: dict[str, list[int]] = {}
    for row in roster.itertuples(index=False):
        key = player_key(str(row.player_name))
        if not key:
            continue
        by_key.setdefault(key, []).append(int(row.player_id))
    starters: set[int] = set()
    missing: list[str] = []
    for name in posted:
        hits = by_key.get(player_key(name), [])
        if len(hits) != 1:
            missing.append(name)
            continue
        starters.add(hits[0])
    return starters, missing


def _script(game: dict, path: Path, audit: dict[str, int]):
    away_id, home_id = _matchup(str(game.get("game") or ""), path, audit)
    if away_id is None or home_id is None:
        return None
    markets = game.get("markets")
    if not isinstance(markets, list):
        raise ValueError(f"{path.name} markets must be a list")
    spreads = []
    totals = []
    for market in markets:
        if not isinstance(market, dict):
            raise ValueError(f"{path.name} market must be an object")
        name = str(market.get("market") or "").strip().lower()
        if name == "spread":
            spreads.append(market)
        elif name == "total":
            totals.append(market)
    home_spread, spread_dropped = _home_spread(spreads, home_id, path)
    total, total_dropped = _total(totals, path)
    if home_spread is None or total is None:
        return None
    return (
        (away_id, home_id),
        ClosingScript(total, home_spread),
        spread_dropped + total_dropped,
    )


def _home_spread(markets: list[dict], home_id: str, path: Path):
    rows = []
    for market in markets:
        side = canonical_team(str(market.get("side") or ""), "nba")
        line = _number(market.get("line"), path)
        price = _number(market.get("american_price"), path)
        if side is None or line is None or price is None:
            continue
        rows.append((side, line, price))
    groups: dict[float, list[tuple[str, float, float]]] = {}
    for side, line, price in rows:
        groups.setdefault(round(abs(line), 4), []).append((side, line, price))
    pairs = []
    for group in groups.values():
        sides = {side for side, _line, _price in group}
        if len(sides) < 2:
            continue
        home = [row for row in group if row[0] == home_id]
        if len(home) != 1:
            continue
        pairs.append(home[0])
    if not pairs:
        return None, 0
    chosen = min(pairs, key=lambda row: _gap(rows, abs(row[1])))
    dropped = sum(
        1
        for group in groups.values()
        if round(abs(chosen[1]), 4) != round(abs(group[0][1]), 4)
    )
    return chosen[1], dropped


def _total(markets: list[dict], path: Path):
    rows = []
    for market in markets:
        side = str(market.get("side") or "").strip().lower().split(" ", 1)[0]
        line = _number(market.get("line"), path)
        price = _number(market.get("american_price"), path)
        if side not in {"over", "under"} or line is None or price is None:
            continue
        rows.append((side, line, price))
    groups: dict[float, list[tuple[str, float, float]]] = {}
    for side, line, price in rows:
        groups.setdefault(round(line, 4), []).append((side, line, price))
    pairs = []
    for group in groups.values():
        sides = {side for side, _line, _price in group}
        if sides >= {"over", "under"}:
            pairs.append(group)
    if not pairs:
        return None, 0
    chosen = min(pairs, key=_rung_gap)
    line = float(chosen[0][1])
    dropped = sum(
        len(group)
        for group in groups.values()
        if round(group[0][1], 4) != round(line, 4)
    )
    return line, 1 if dropped else 0


def _gap(rows, line: float) -> tuple[float, float]:
    prices = [
        price
        for _side, number, price in rows
        if math.isclose(abs(number), line, abs_tol=1e-6)
    ]
    return _price_gap(prices), abs(line)


def _rung_gap(rows) -> tuple[float, float]:
    return _price_gap([price for _side, _line, price in rows]), abs(float(rows[0][1]))


def _price_gap(prices: list[float]) -> float:
    if len(prices) < 2:
        return 1.0
    implied = [american_to_implied_probability(price) for price in prices[:2]]
    return abs(implied[0] - implied[1])


def _closing_names(board, day: pd.Timestamp, path: Path, audit: dict[str, int]):
    if board is None:
        return None
    if not isinstance(board, dict):
        raise ValueError(f"{path.name} closing lineup must be an object")
    if _close_stamp(board, day, path, audit) is None:
        return None
    lineup = board.get("lineup")
    if not isinstance(lineup, list) or not lineup:
        return None
    names = []
    for player in lineup:
        if not isinstance(player, dict):
            raise ValueError(f"{path.name} closing player must be an object")
        name = str(player.get("name") or "").strip()
        if not name:
            raise ValueError(f"{path.name} closing player is missing a name")
        names.append(name)
    return tuple(names)


def _matchup(label: str, path: Path, audit: dict[str, int]):
    parts = re.split(
        r"\s+(?:@|at|vs\.?|v)\s+",
        label.strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(parts) != 2:
        audit["unmatched_name"] += 1
        return None, None
    away_id = canonical_team(parts[0], "nba")
    home_id = canonical_team(parts[1], "nba")
    if away_id is None or home_id is None or away_id == home_id:
        audit["unmatched_name"] += 1
        return None, None
    return away_id, home_id


def _abbr_id(value, path: Path) -> str | None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path.name} is missing a team")
    team_id = canonical_team(value.strip(), "nba")
    if team_id is None:
        raise ValueError(f"{path.name} has an unknown team {value.strip()}")
    return team_id


def _payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _require_slate(payload: dict, path: Path, slate) -> pd.Timestamp:
    if payload.get("timezone") != "America/Los_Angeles":
        raise ValueError(f"{path.name} closing date must be Pacific")
    day = pd.Timestamp(slate).normalize()
    file_day = pd.Timestamp(payload.get("date")).normalize()
    if pd.isna(day) or pd.isna(file_day) or file_day != day:
        raise ValueError(f"{path.name} is not the close for {day.date()}")
    return day


def _games(payload: dict, path: Path) -> list:
    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError(f"{path.name} games must be a list")
    for game in games:
        if not isinstance(game, dict):
            raise ValueError(f"{path.name} game must be an object")
    return games


def _close_stamp(record: dict, slate: pd.Timestamp, path: Path, audit: dict[str, int]):
    raw = record.get("closing_fetched_at") or record.get("fetched_at")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{path.name} is missing closing_fetched_at")
    fetched = _aware(raw, path)
    if fetched.astimezone(PACIFIC).date() != slate.date():
        audit["timestamp_mismatch"] += 1
        return None
    nxt = record.get("next_fetched_at")
    if nxt is not None and (not isinstance(nxt, str) or _aware(nxt, path) <= fetched):
        audit["timestamp_mismatch"] += 1
        return None
    return fetched


def _aware(value: str, path: Path) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{path.name} has an unreadable timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{path.name} timestamps must be timezone-aware")
    return parsed


def _number(value, path: Path) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} has a non-numeric line") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path.name} has a non-numeric line")
    return number
