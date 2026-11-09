"""Closing lines from the 15-minute snapshot boards.

When a game has a start time, the close is the last snapshot with
``fetched_at`` at or before that start. A later snapshot is not a close.
Boards with no start time still use the next-fetch drop: the last listing
followed, within one fetch, by a snapshot that no longer lists the player.

Novig, ProphetX, and Pinnacle team closes read ``*_team.json``. FanDuel and
DraftKings read ``game_lines`` in the main snapshot. PrizePicks and Underdog
have no team board.
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
    "fanduel": _Book(
        "FanDuel",
        "picks",
        _snapshot_name("fanduel", props=False),
        "fanduel_nba_closing_{date}.json",
    ),
    "draftkings": _Book(
        "DraftKings",
        "picks",
        _snapshot_name("draftkings", props=False),
        "draftkings_nba_closing_{date}.json",
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
def _team_snapshot_name(prefix: str) -> re.Pattern[str]:
    return re.compile(
        rf"{prefix}_nba_\d{{4}}-\d{{2}}-\d{{2}}_\d{{6}}_team\.json$"
    )


_FULL_GAME_MARKETS = ("moneyline", "spread", "total")
_NO_TEAM_BOARD = frozenset({"prizepicks", "underdog"})


@dataclass(frozen=True)
class _TeamBook:
    source: str
    kind: str
    pattern: re.Pattern[str]
    closing_name: str


_TEAM_BOOKS: dict[str, _TeamBook] = {
    "fanduel": _TeamBook(
        "FanDuel",
        "game_lines",
        _snapshot_name("fanduel", props=False),
        "fanduel_nba_team_closing_{date}.json",
    ),
    "draftkings": _TeamBook(
        "DraftKings",
        "game_lines",
        _snapshot_name("draftkings", props=False),
        "draftkings_nba_team_closing_{date}.json",
    ),
    "novig": _TeamBook(
        "Novig",
        "team_games",
        _team_snapshot_name("novig"),
        "novig_nba_team_closing_{date}.json",
    ),
    "prophetx": _TeamBook(
        "ProphetX",
        "team_games",
        _team_snapshot_name("prophetx"),
        "prophetx_nba_team_closing_{date}.json",
    ),
    "pinnacle": _TeamBook(
        "Pinnacle",
        "team_games",
        _team_snapshot_name("pinnacle"),
        "pinnacle_nba_team_closing_{date}.json",
    ),
}
_LEFT_OUT = (
    "no_later_snapshot",
    "next_fetch_too_late",
    "still_listed_on_next_fetch",
    "after_start",
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
    label: str | None = None
    starts_at: datetime | None = None


@dataclass(frozen=True)
class Snapshot:
    fetched_at: datetime
    lines: tuple[ListedLine, ...]


@dataclass(frozen=True)
class PlayerClose:
    player: str
    closing_fetched_at: datetime
    next_fetched_at: datetime | None
    lines: tuple[ListedLine, ...]
    opening_fetched_at: datetime
    opening_lines: tuple[ListedLine, ...]


@dataclass(frozen=True)
class ClosingBoards:
    by_date: dict[date, tuple[PlayerClose, ...]]
    left_out: dict[date, dict[str, int]]
    skipped_blank_lines: int


def _listing_key(line: ListedLine) -> str:
    """Separate the same player when one snapshot lists two games."""
    if line.starts_at is None:
        return line.player
    return f"{line.player}\x1f{line.starts_at.isoformat()}"


def _include_pre_start(
    player: str,
    identity: str,
    indexes: list[int],
    ordered: tuple[Snapshot, ...],
    lines_at: dict[tuple[str, int], list[ListedLine]],
    included: dict[date, list[PlayerClose]],
    left: dict[date, Counter[str]],
) -> None:
    """Keep the last listing at or before this game's start time."""
    pre: list[int] = []
    start: datetime | None = None
    for index in indexes:
        lines = lines_at[(identity, index)]
        listed_start = _one_start(lines, player)
        if listed_start is None:
            continue
        start = listed_start
        if ordered[index].fetched_at <= listed_start:
            pre.append(index)
    if start is None:
        return
    if not pre:
        left[start.astimezone(PACIFIC).date()]["after_start"] += 1
        return
    index = pre[-1]
    close_day = ordered[index].fetched_at.astimezone(PACIFIC).date()
    open_index = next(
        candidate
        for candidate in pre
        if ordered[candidate].fetched_at.astimezone(PACIFIC).date() == close_day
    )
    nxt = ordered[index + 1].fetched_at if index + 1 < len(ordered) else None
    included[close_day].append(
        PlayerClose(
            player=player,
            closing_fetched_at=ordered[index].fetched_at,
            next_fetched_at=nxt,
            lines=tuple(sorted(lines_at[(identity, index)], key=_line_sort_key)),
            opening_fetched_at=ordered[open_index].fetched_at,
            opening_lines=tuple(
                sorted(lines_at[(identity, open_index)], key=_line_sort_key)
            ),
        )
    )


def _one_start(lines: list[ListedLine], player: str) -> datetime | None:
    starts = {line.starts_at for line in lines if line.starts_at is not None}
    if not starts:
        return None
    if len(starts) != 1:
        raise ValueError(f"{player} has more than one start time in one snapshot")
    start = next(iter(starts))
    if start.tzinfo is None:
        raise ValueError(f"{player} start time must be timezone-aware")
    return start


def closing_boards(
    snapshots: list[Snapshot] | tuple[Snapshot, ...],
    *,
    max_gap: timedelta = MAX_GAP,
) -> ClosingBoards:
    """Last pre-start listing, or the last listing before a player drops."""
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
            identity = _listing_key(line)
            indexes = presence[identity]
            if not indexes or indexes[-1] != index:
                indexes.append(index)
            lines_at[(identity, index)].append(line)

    started = {
        player
        for (player, _index), lines in lines_at.items()
        if any(line.starts_at is not None for line in lines)
    }
    first_on_date: dict[tuple[str, date], int] = {}
    last_on_date: dict[tuple[str, date], int] = {}
    for player, indexes in presence.items():
        if player in started:
            continue
        for index in indexes:
            day = ordered[index].fetched_at.astimezone(PACIFIC).date()
            key = (player, day)
            if key not in first_on_date or index < first_on_date[key]:
                first_on_date[key] = index
            if key not in last_on_date or index > last_on_date[key]:
                last_on_date[key] = index

    included: dict[date, list[PlayerClose]] = defaultdict(list)
    left: dict[date, Counter[str]] = defaultdict(Counter)
    for identity in started:
        _include_pre_start(
            lines_at[(identity, presence[identity][0])][0].player,
            identity,
            presence[identity],
            ordered,
            lines_at,
            included,
            left,
        )
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
        open_index = first_on_date[(player, day)]
        included[day].append(
            PlayerClose(
                player=player,
                closing_fetched_at=ordered[index].fetched_at,
                next_fetched_at=ordered[index + 1].fetched_at,
                lines=tuple(sorted(lines_at[(player, index)], key=_line_sort_key)),
                opening_fetched_at=ordered[open_index].fetched_at,
                opening_lines=tuple(
                    sorted(lines_at[(player, open_index)], key=_line_sort_key)
                ),
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


def write_team_closing_files(
    folder: Path,
    out_dir: Path,
    *,
    book: str,
    max_gap: timedelta = MAX_GAP,
) -> list[Path]:
    """Write one full-game team close per Pacific date.

    The folder is the book's snapshot directory. ``out_dir`` is where
    ``team_closing_lines`` files are written.
    """
    spec = _require_team_book(book)
    boards = closing_boards(load_team_snapshots(book, folder), max_gap=max_gap)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for day, rows in boards.by_date.items():
        if not rows:
            continue
        path = out_dir / spec.closing_name.format(date=day.isoformat())
        path.write_text(
            json.dumps(
                _team_day_payload(day, rows, boards, max_gap, source=spec.source),
                indent=2,
            )
            + "\n"
        )
        written.append(path)
        logger.info("wrote %s games=%s", path.name, len(rows))
    return written


def load_team_snapshots(book: str, folder: Path) -> list[Snapshot]:
    """Load one book's NBA team snapshots from ``folder``."""
    spec = _require_team_book(book)
    paths = sorted(
        path
        for path in folder.glob("*.json")
        if path.is_file() and spec.pattern.search(path.name)
    )
    if not paths:
        raise FileNotFoundError(f"no {book} NBA team snapshots in {folder}")
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
            "When the game has a start time, the close is the last snapshot "
            "with fetched_at at or before that start. A later snapshot is "
            "dropped. Listings with no start time are included when the last "
            "listing on this date is followed, within max_gap_minutes, by a "
            "snapshot that no longer lists the player. The opening line is the "
            "first listing on this Pacific date."
        ),
        "player_count": len(rows),
        "left_out": boards.left_out[day],
        "players": [_player_payload(row) for row in rows],
    }


def _player_payload(row: PlayerClose) -> dict:
    return {
        "player": row.player,
        "opening_fetched_at": _stamp(row.opening_fetched_at),
        "closing_fetched_at": _stamp(row.closing_fetched_at),
        **_next_fetch_fields(row),
        "opening_lines": [_prop_line_fields(line) for line in row.opening_lines],
        "lines": [_prop_line_fields(line) for line in row.lines],
    }


def _next_fetch_fields(row: PlayerClose) -> dict:
    if row.next_fetched_at is None:
        return {"next_fetched_at": None, "gap_minutes": None}
    gap_minutes = (row.next_fetched_at - row.closing_fetched_at).total_seconds() / 60
    return {
        "next_fetched_at": _stamp(row.next_fetched_at),
        "gap_minutes": round(gap_minutes, 1),
    }


def _prop_line_fields(line: ListedLine) -> dict:
    return {
        "stat_type": line.stat_type,
        "line_score": line.line_score,
        "odds_type": line.odds_type,
        "updated_at": line.updated_at,
        "league": line.league,
        "side": line.side,
        "american_price": line.american_price,
    }


def _require_team_book(book: str) -> _TeamBook:
    if book in _NO_TEAM_BOARD:
        raise ValueError(f"{book} has no team snapshots")
    try:
        return _TEAM_BOOKS[book]
    except KeyError as exc:
        known = ", ".join(sorted(_TEAM_BOOKS))
        raise ValueError(
            f"unknown book {book!r}; expected one of {known}"
        ) from exc


def _team_day_payload(
    day: date,
    rows: tuple[PlayerClose, ...],
    boards: ClosingBoards,
    max_gap: timedelta,
    *,
    source: str,
) -> dict:
    ordered = sorted(rows, key=lambda row: (_team_label(row).casefold(), row.player))
    return {
        "source": source,
        "league": "NBA",
        "date": day.isoformat(),
        "timezone": "America/Los_Angeles",
        "max_gap_minutes": max_gap.total_seconds() / 60,
        "rule": (
            "When the game has a start time, the close is the last snapshot "
            "with fetched_at at or before that start. A later snapshot is "
            "dropped. Games with no start time are included when the last "
            "listing on this date is followed, within max_gap_minutes, by a "
            "snapshot that no longer lists the game. Markets are the full-game "
            "moneyline, spread, and total. The opening line is the first "
            "listing on this Pacific date."
        ),
        "game_count": len(ordered),
        "left_out": boards.left_out[day],
        "games": [_team_game_payload(row) for row in ordered],
    }


def _team_game_payload(row: PlayerClose) -> dict:
    return {
        "game": _team_label(row),
        "game_id": row.player,
        "opening_fetched_at": _stamp(row.opening_fetched_at),
        "closing_fetched_at": _stamp(row.closing_fetched_at),
        **_next_fetch_fields(row),
        "opening_markets": [_market_fields(line) for line in row.opening_lines],
        "markets": [_market_fields(line) for line in row.lines],
    }


def _market_fields(line: ListedLine) -> dict:
    return {
        "market": line.stat_type,
        "side": line.side,
        "line": line.line_score,
        "american_price": line.american_price,
    }


def _team_label(row: PlayerClose) -> str:
    for line in row.lines:
        if line.label:
            return line.label
    return row.player


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
    if kind == "team_games":
        return _team_games_from_payload(payload, source=source)
    if kind == "game_lines":
        return _flat_game_lines(payload, source=source)
    raise ValueError(f"unknown prop kind: {kind}")


def _team_games_from_payload(payload: dict, *, source: str) -> tuple[ListedLine, ...]:
    raw = payload.get("games") or []
    if not isinstance(raw, list):
        raise ValueError(f"{source} games must be a list")
    lines: list[ListedLine] = []
    for index, game in enumerate(raw):
        if not isinstance(game, dict):
            raise ValueError(f"{source} game {index} must be an object")
        where = f"{source} game {index}"
        key, label = _game_identity(game)
        if not key:
            continue
        markets = game.get("team_markets") or {}
        if not isinstance(markets, dict):
            raise ValueError(f"{where} team_markets must be an object")
        lines.extend(
            _markets_to_lines(
                markets,
                key=key,
                label=label,
                source=where,
                starts_at=_game_start(game, where),
            )
        )
    return tuple(lines)


def _flat_game_lines(payload: dict, *, source: str) -> tuple[ListedLine, ...]:
    raw = payload.get("game_lines") or []
    if not isinstance(raw, list):
        raise ValueError(f"{source} game_lines must be a list")
    lines: list[ListedLine] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{source} game line {index} must be an object")
        market = _market_name(row.get("market"), source=f"{source} game line {index}")
        if market is None:
            continue
        event = _text(row.get("event_name"))
        if not event:
            raise ValueError(f"{source} game line {index} is missing event_name")
        lines.append(
            _team_quote(
                key=event,
                label=event,
                market=market,
                side=row.get("side"),
                line=row.get("line"),
                american=row.get("american"),
                source=f"{source} game line {index}",
            )
        )
    return tuple(lines)


def _markets_to_lines(
    markets: dict,
    *,
    key: str,
    label: str,
    source: str,
    starts_at: datetime | None = None,
) -> list[ListedLine]:
    lines: list[ListedLine] = []
    for market in _FULL_GAME_MARKETS:
        raw = markets.get(market) or []
        if not raw:
            continue
        if not isinstance(raw, list):
            raise ValueError(f"{source} {market} must be a list")
        if raw and isinstance(raw[0], dict) and "lines" in raw[0]:
            for index, row in enumerate(_main_period_lines(raw, source=f"{source} {market}")):
                lines.append(
                    _team_quote(
                        key=key,
                        label=label,
                        market=market,
                        side=row.get("team") or row.get("side"),
                        line=row.get("points"),
                        american=row.get("american"),
                        source=f"{source} {market} line {index}",
                        starts_at=starts_at,
                    )
                )
            continue
        for index, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ValueError(f"{source} {market} {index} must be an object")
            lines.append(
            _team_quote(
                key=key,
                label=label,
                market=market,
                side=row.get("name"),
                line=row.get("line"),
                american=row.get("american"),
                source=f"{source} {market} {index}",
                starts_at=starts_at,
            )
            )
    return lines


def _main_period_lines(blocks: list, *, source: str) -> list:
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"{source} block {index} must be an object")
        if block.get("is_alternate"):
            continue
        period = block.get("period", 0)
        if isinstance(period, bool) or not isinstance(period, (int, float)) or period != 0:
            continue
        raw = block.get("lines") or []
        if not isinstance(raw, list):
            raise ValueError(f"{source} lines must be a list")
        for line_index, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ValueError(f"{source} line {line_index} must be an object")
        return raw
    return []


def _team_quote(
    *,
    key: str,
    label: str,
    market: str,
    side: object,
    line: object,
    american: object,
    source: str,
    starts_at: datetime | None = None,
) -> ListedLine:
    posted = _text(side)
    if not posted:
        raise ValueError(f"{source} is missing a side")
    if american is None:
        raise ValueError(f"{source} is missing an american price")
    return ListedLine(
        player=key,
        stat_type=market,
        line_score=_line_score(line, source=source),
        odds_type=None,
        updated_at=None,
        league=None,
        side=posted,
        american_price=_american_number(american, source=source),
        label=label,
        starts_at=starts_at,
    )


def _market_name(value: object, *, source: str) -> str | None:
    text = _text(value)
    if text is None:
        raise ValueError(f"{source} is missing market")
    market = text.casefold()
    if market not in _FULL_GAME_MARKETS:
        return None
    return market


def _game_identity(game: dict) -> tuple[str, str]:
    event_id = game.get("event_id")
    if event_id is not None and str(event_id).strip():
        key = str(event_id).strip()
        return key, _text(game.get("name")) or key
    matchup_id = game.get("matchup_id")
    if matchup_id is not None and str(matchup_id).strip():
        key = str(matchup_id).strip()
        return key, _text(game.get("label")) or key
    label = _text(game.get("name")) or _text(game.get("label")) or ""
    return label, label


def _line_score(value: object, *, source: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"{source} has a non-numeric line") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} has a non-numeric line")
    return value


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
    starts_at = _game_start(game, source)
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
            lines.append(_listed_line(base, source=where, starts_at=starts_at))
            continue
        for side, price in priced:
            row = {**base, "side": side, "american_price": price}
            lines.append(_listed_line(row, source=where, starts_at=starts_at))
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


def _game_start(game: dict, source: str) -> datetime | None:
    raw = game.get("start_time")
    if raw is None:
        raw = game.get("scheduled")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{source} start time must be a timestamp")
    return _parse_time(raw, source=f"{source} start time")


def _parse_time(value: str, *, source: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{source} has an unparseable timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{source} must be timezone-aware")
    return parsed


def _listed_line(
    row: dict,
    *,
    source: str,
    starts_at: datetime | None = None,
) -> ListedLine:
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
        starts_at=starts_at,
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
