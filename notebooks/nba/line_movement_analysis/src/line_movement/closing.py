"""Closing lines from the 15-minute snapshot boards.

PrizePicks, Underdog, Novig, ProphetX, and Pinnacle drop a player once
the game starts. The closing line is the last snapshot that still lists
that player when the next snapshot, within one fetch, no longer does.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
# One 15-minute fetch, plus the late polls observed in this archive (about 21 minutes).
MAX_GAP = timedelta(minutes=22)


def _snapshot_name(prefix: str, *, props: bool) -> re.Pattern[str]:
    suffix = "_props" if props else ""
    return re.compile(
        rf"{prefix}_nba_\d{{4}}-\d{{2}}-\d{{2}}_\d{{6}}{suffix}\.json$"
    )


@dataclass(frozen=True)
class _Book:
    source: str
    kind: str
    pattern: re.Pattern[str]
    closing_name: str


_BOOKS: dict[str, _Book] = {
    "prizepicks": _Book(
        "PrizePicks",
        "projections",
        _snapshot_name("prizepicks", props=False),
        "prizepicks_nba_closing_{date}.json",
    ),
    "underdog": _Book(
        "Underdog",
        "picks",
        _snapshot_name("underdog", props=False),
        "underdog_nba_closing_{date}.json",
    ),
    "novig": _Book(
        "Novig",
        "games",
        _snapshot_name("novig", props=True),
        "novig_nba_closing_{date}.json",
    ),
    "prophetx": _Book(
        "ProphetX",
        "games",
        _snapshot_name("prophetx", props=True),
        "prophetx_nba_closing_{date}.json",
    ),
    "pinnacle": _Book(
        "Pinnacle",
        "games",
        _snapshot_name("pinnacle", props=True),
        "pinnacle_nba_closing_{date}.json",
    ),
}
_LEFT_OUT = (
    "no_later_snapshot",
    "next_fetch_too_late",
    "still_listed_on_next_fetch",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ListedLine:
    player: str
    stat_type: str
    line_score: float | int | None
    odds_type: str | None
    updated_at: str | None
    league: str | None
    side: str | None = None
    american_price: int | float | None = None


@dataclass(frozen=True)
class Snapshot:
    fetched_at: datetime
    lines: tuple[ListedLine, ...]


@dataclass(frozen=True)
class PlayerClose:
    player: str
    closing_fetched_at: datetime
    next_fetched_at: datetime
    lines: tuple[ListedLine, ...]


@dataclass(frozen=True)
class ClosingBoards:
    by_date: dict[date, tuple[PlayerClose, ...]]
    left_out: dict[date, dict[str, int]]
    skipped_blank_lines: int


def closing_boards(
    snapshots: list[Snapshot] | tuple[Snapshot, ...],
    *,
    max_gap: timedelta = MAX_GAP,
) -> ClosingBoards:
    """Last listing on a Pacific date when the next fetch drops that player."""
    if max_gap <= timedelta(0):
        raise ValueError("max_gap must be positive")
    ordered = tuple(sorted(snapshots, key=lambda snap: snap.fetched_at))
    for snap in ordered:
        if snap.fetched_at.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")

    presence: dict[str, list[int]] = defaultdict(list)
    lines_at: dict[tuple[str, int], list[ListedLine]] = defaultdict(list)
    skipped = 0
    for index, snap in enumerate(ordered):
        for line in snap.lines:
            if not line.player.strip():
                skipped += 1
                continue
            indexes = presence[line.player]
            if not indexes or indexes[-1] != index:
                indexes.append(index)
            lines_at[(line.player, index)].append(line)

    last_on_date: dict[tuple[str, date], int] = {}
    for player, indexes in presence.items():
        for index in indexes:
            day = ordered[index].fetched_at.astimezone(PACIFIC).date()
            key = (player, day)
            if key not in last_on_date or index > last_on_date[key]:
                last_on_date[key] = index

    included: dict[date, list[PlayerClose]] = defaultdict(list)
    left: dict[date, Counter[str]] = defaultdict(Counter)
    present_at = {player: set(indexes) for player, indexes in presence.items()}
    for (player, day), index in last_on_date.items():
        if index + 1 >= len(ordered):
            left[day]["no_later_snapshot"] += 1
            continue
        if (index + 1) in present_at[player]:
            left[day]["still_listed_on_next_fetch"] += 1
            continue
        gap = ordered[index + 1].fetched_at - ordered[index].fetched_at
        if gap <= timedelta(0) or gap > max_gap:
            left[day]["next_fetch_too_late"] += 1
            continue
        included[day].append(
            PlayerClose(
                player=player,
                closing_fetched_at=ordered[index].fetched_at,
                next_fetched_at=ordered[index + 1].fetched_at,
                lines=tuple(sorted(lines_at[(player, index)], key=_line_sort_key)),
            )
        )

    by_date: dict[date, tuple[PlayerClose, ...]] = {}
    left_out: dict[date, dict[str, int]] = {}
    for day in sorted(set(included) | set(left)):
        by_date[day] = tuple(
            sorted(included.get(day, []), key=lambda row: row.player.casefold())
        )
        counts = left.get(day, Counter())
        left_out[day] = {reason: int(counts[reason]) for reason in _LEFT_OUT}
    return ClosingBoards(
        by_date=by_date,
        left_out=left_out,
        skipped_blank_lines=skipped,
    )


def load_prizepicks_snapshots(folder: Path) -> list[Snapshot]:
    """Load ``prizepicks_nba_YYYY-MM-DD_HHMMSS.json`` files from ``folder``."""
    return load_book_snapshots("prizepicks", folder)


def load_book_snapshots(book: str, folder: Path) -> list[Snapshot]:
    """Load one book's NBA prop snapshots from ``folder``."""
    spec = _require_book(book)
    paths = sorted(
        path
        for path in folder.glob("*.json")
        if path.is_file() and spec.pattern.search(path.name)
    )
    if not paths:
        raise FileNotFoundError(f"no {book} NBA snapshots in {folder}")
    snapshots: list[Snapshot] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} must be a JSON object")
        snapshots.append(
            _snapshot_from_payload(payload, source=path.name, kind=spec.kind)
        )
    return snapshots


def snapshot_from_payload(payload: dict, *, source: str) -> Snapshot:
    """PrizePicks projection snapshot. Other books use ``load_book_snapshots``."""
    return _snapshot_from_payload(payload, source=source, kind="projections")


def _snapshot_from_payload(payload: dict, *, source: str, kind: str) -> Snapshot:
    raw = payload.get("fetched_at")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{source} is missing fetched_at")
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        fetched_at = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{source} has an unparseable fetched_at") from exc
    if fetched_at.tzinfo is None:
        raise ValueError(f"{source} fetched_at must be timezone-aware")
    return Snapshot(
        fetched_at=fetched_at,
        lines=_lines_from_payload(payload, source=source, kind=kind),
    )


def write_closing_files(
    folder: Path,
    out_dir: Path,
    *,
    book: str = "prizepicks",
    max_gap: timedelta = MAX_GAP,
) -> list[Path]:
    """Write one closing-line JSON per Pacific date that has at least one player."""
    spec = _require_book(book)
    boards = closing_boards(load_book_snapshots(book, folder), max_gap=max_gap)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for day, rows in boards.by_date.items():
        if not rows:
            logger.info(
                "no closing players on %s left_out=%s",
                day.isoformat(),
                boards.left_out[day],
            )
            continue
        path = out_dir / spec.closing_name.format(date=day.isoformat())
        payload = _day_payload(day, rows, boards, max_gap, source=spec.source)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(path)
        logger.info(
            "wrote %s players=%s left_out=%s",
            path.name,
            len(rows),
            boards.left_out[day],
        )
    if boards.skipped_blank_lines:
        logger.info("skipped %s blank player names", boards.skipped_blank_lines)
    return written


def standard_closing_points(folder: Path, player: str) -> dict[str, float]:
    """Pacific date to the standard Points closing line for one player.

    Demon, goblin, and combined-stat lines are ignored. A player name must
    match exactly, so a combo listing does not count as that player.
    """
    if not player.strip():
        raise ValueError("player name is blank")
    paths = sorted(
        path
        for path in folder.glob("prizepicks_nba_closing_*.json")
        if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"no PrizePicks closing files in {folder}")
    lines: dict[str, float] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} must be a JSON object")
        day = payload.get("date")
        if not isinstance(day, str) or not day.strip():
            raise ValueError(f"{path.name} is missing date")
        matches = _standard_points_lines(
            payload.get("players"), player, source=path.name
        )
        if not matches:
            continue
        if len(matches) > 1:
            raise ValueError(
                f"{player} has {len(matches)} standard points lines on {day}"
            )
        if day in lines:
            raise ValueError(f"duplicate closing file for {day}")
        lines[day] = matches[0]
    return lines


def _standard_points_lines(players: object, player: str, *, source: str) -> list[float]:
    if players is None:
        return []
    if not isinstance(players, list):
        raise ValueError(f"{source} players must be a list")
    matches: list[float] = []
    for index, row in enumerate(players):
        if not isinstance(row, dict):
            raise ValueError(f"{source} player {index} must be an object")
        if row.get("player") != player:
            continue
        raw_lines = row.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError(f"{source} lines for {player} must be a list")
        for line in raw_lines:
            if not isinstance(line, dict):
                raise ValueError(f"{source} has a non-object line for {player}")
            if line.get("stat_type") != "Points" or line.get("odds_type") != "standard":
                continue
            score = line.get("line_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{source} has a non-numeric points line for {player}")
            matches.append(float(score))
    return matches


def _day_payload(
    day: date,
    rows: tuple[PlayerClose, ...],
    boards: ClosingBoards,
    max_gap: timedelta,
    *,
    source: str,
) -> dict:
    return {
        "source": source,
        "league": "NBA",
        "date": day.isoformat(),
        "timezone": "America/Los_Angeles",
        "max_gap_minutes": max_gap.total_seconds() / 60,
        "rule": (
            "Included when this player's last listing on this date is followed, "
            "within max_gap_minutes, by a snapshot that no longer lists them. "
            "That snapshot is the last fetch before the scraper drops the game "
            "at start time."
        ),
        "player_count": len(rows),
        "left_out": boards.left_out[day],
        "players": [_player_payload(row) for row in rows],
    }


def _player_payload(row: PlayerClose) -> dict:
    gap_minutes = (row.next_fetched_at - row.closing_fetched_at).total_seconds() / 60
    return {
        "player": row.player,
        "closing_fetched_at": _stamp(row.closing_fetched_at),
        "next_fetched_at": _stamp(row.next_fetched_at),
        "gap_minutes": round(gap_minutes, 1),
        "lines": [
            {
                "stat_type": line.stat_type,
                "line_score": line.line_score,
                "odds_type": line.odds_type,
                "updated_at": line.updated_at,
                "league": line.league,
                "side": line.side,
                "american_price": line.american_price,
            }
            for line in row.lines
        ],
    }


def _require_book(book: str) -> _Book:
    try:
        return _BOOKS[book]
    except KeyError as exc:
        known = ", ".join(sorted(_BOOKS))
        raise ValueError(f"unknown book {book!r}; expected one of {known}") from exc


def _lines_from_payload(
    payload: dict,
    *,
    source: str,
    kind: str,
) -> tuple[ListedLine, ...]:
    if kind == "projections":
        return _object_lines(
            payload.get("projections") or [],
            source=source,
            label="projection",
            builder=_listed_line,
        )
    if kind == "picks":
        return _object_lines(
            payload.get("picks") or [],
            source=source,
            label="pick",
            builder=_pick_line,
        )
    if kind == "games":
        raw = payload.get("games") or []
        if not isinstance(raw, list):
            raise ValueError(f"{source} games must be a list")
        lines: list[ListedLine] = []
        for index, game in enumerate(raw):
            if not isinstance(game, dict):
                raise ValueError(f"{source} game {index} must be an object")
            lines.extend(_game_lines(game, source=f"{source} game {index}"))
        return tuple(lines)
    raise ValueError(f"unknown prop kind: {kind}")


def _object_lines(
    raw: object,
    *,
    source: str,
    label: str,
    builder,
) -> tuple[ListedLine, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{source} {label}s must be a list")
    lines: list[ListedLine] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{source} {label} {index} must be an object")
        lines.append(builder(row, source=f"{source} {label} {index}"))
    return tuple(lines)


def _pick_line(row: dict, *, source: str) -> ListedLine:
    return _listed_line(
        {
            "player": row.get("full_name"),
            "stat_type": row.get("stat_name"),
            "line_score": row.get("stat_value"),
            "updated_at": row.get("updated_at"),
            "side": row.get("choice"),
            "american_price": row.get("american_price"),
        },
        source=source,
    )


def _game_lines(game: dict, *, source: str) -> list[ListedLine]:
    raw = game.get("props") or []
    if not isinstance(raw, list):
        raise ValueError(f"{source} props must be a list")
    lines: list[ListedLine] = []
    for index, prop in enumerate(raw):
        if not isinstance(prop, dict):
            raise ValueError(f"{source} prop {index} must be an object")
        where = f"{source} prop {index}"
        priced: list[tuple[str, int | float]] = []
        for side in ("over", "under"):
            raw_price = _side_price(prop, side)
            if raw_price is None:
                continue
            priced.append((side, _american_number(raw_price, source=where)))
        base = {
            "player": prop.get("player"),
            "stat_type": prop.get("stat"),
            "line_score": prop.get("line"),
        }
        if not priced:
            lines.append(_listed_line(base, source=where))
            continue
        for side, price in priced:
            row = {**base, "side": side, "american_price": price}
            lines.append(_listed_line(row, source=where))
    return lines


def _side_price(prop: dict, side: str) -> object:
    flat = prop.get(f"american_{side}")
    if flat is not None:
        return flat
    nested = prop.get(side)
    if isinstance(nested, dict):
        return nested.get("american")
    return None


def _american_number(value: object, *, source: str) -> int | float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{source} has a non-numeric american price")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace("+", "")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{source} has a non-numeric american price") from exc
    if number.is_integer():
        return int(number)
    return number


def _listed_line(row: dict, *, source: str) -> ListedLine:
    score = row.get("line_score")
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError as exc:
            raise ValueError(f"{source} has a non-numeric line_score") from exc
    elif score is not None and not isinstance(score, (int, float)):
        raise ValueError(f"{source} has a non-numeric line_score")
    if isinstance(score, bool):
        raise ValueError(f"{source} has a non-numeric line_score")
    side = _text(row.get("side"))
    price = row.get("american_price")
    return ListedLine(
        player=str(row.get("player") or "").strip(),
        stat_type=str(row.get("stat_type") or ""),
        line_score=score,
        odds_type=_text(row.get("odds_type")),
        updated_at=_text(row.get("updated_at")),
        league=_text(row.get("league")),
        side=side.lower() if side else None,
        american_price=(
            None if price is None else _american_number(price, source=source)
        ),
    )


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _line_sort_key(line: ListedLine) -> tuple:
    score = line.line_score
    return (
        line.stat_type,
        line.side or "",
        line.odds_type or "",
        score is None,
        score if isinstance(score, (int, float)) else str(score),
    )


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
