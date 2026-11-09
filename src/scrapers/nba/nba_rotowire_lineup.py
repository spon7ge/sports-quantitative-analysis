"""Write one RotoWire NBA lineup snapshot, and the day's open and close.

Usage:
    python -m src.scrapers.nba.nba_rotowire_lineup
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.scrapers.nba.nba_lineups import NBADailyLineups
from src.scrapers.nba.paths import ensure_repo_on_path

OUTPUT_TZ = ZoneInfo("America/Los_Angeles")
_SNAPSHOT_NAME = re.compile(r"rotowire_nba_\d{4}-\d{2}-\d{2}_\d{6}\.json$")


def snapshot_filename(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    pacific = moment.astimezone(OUTPUT_TZ)
    return pacific.strftime("rotowire_nba_%Y-%m-%d_%H%M%S.json")


def fetched_at_text(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    return moment.astimezone(OUTPUT_TZ).isoformat(timespec="seconds")


def output_directory(root: Path | None = None) -> Path:
    base = root if root is not None else ensure_repo_on_path(__file__)
    return base / "data" / "props" / "rotowire_nba_lineup"


def write_snapshot(payload: dict, directory: Path, moment: datetime) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = snapshot_filename(moment)
    final = directory / filename
    temporary = directory / f".{filename}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return final


def run(
    moment: datetime,
    directory: Path,
    fetch: Callable[[], NBADailyLineups] | None = None,
) -> int:
    """Fetch the lineup page once and write one snapshot. 0 on success, 1 on failure."""
    loader = fetch or NBADailyLineups
    try:
        lineups = loader()
        payload = lineups.lineup_snapshot(fetched_at_text(moment))
        write_snapshot(payload, directory, moment)
    except Exception as exc:
        print(f"nba_rotowire_lineup failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


def main() -> int:
    return run(datetime.now(OUTPUT_TZ), output_directory())


@dataclass(frozen=True)
class _PostedLineup:
    fetched_at: datetime
    status: str
    lineup: tuple[dict[str, str | None], ...]


def write_lineup_closing_files(folder: Path, out_dir: Path) -> list[Path]:
    """Write one file per Pacific date that has a confirmed lineup.

    Opening is the first non-empty lineup posted for that team on the date.
    Closing is the last lineup marked confirmed that date. A game is included
    when at least one side has a confirmed lineup.
    """
    by_date = _lineups_by_date(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for day in sorted(by_date):
        games = _closing_games(by_date[day])
        if not games:
            continue
        path = out_dir / f"rotowire_nba_lineup_closing_{day.isoformat()}.json"
        path.write_text(
            json.dumps(_lineup_day_payload(day, games), indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def _lineups_by_date(
    folder: Path,
) -> dict[date, dict[tuple[str, str], dict[str, list[_PostedLineup]]]]:
    paths = sorted(
        path
        for path in folder.glob("*.json")
        if path.is_file() and _SNAPSHOT_NAME.search(path.name)
    )
    if not paths:
        raise FileNotFoundError(f"no RotoWire lineup snapshots in {folder}")
    by_date: dict[date, dict[tuple[str, str], dict[str, list[_PostedLineup]]]] = (
        defaultdict(lambda: defaultdict(lambda: {"away": [], "home": []}))
    )
    for path in paths:
        payload = _read_snapshot(path)
        fetched_at = _fetched_at(payload.get("fetched_at"), source=path.name)
        games = payload.get("games")
        if not isinstance(games, list):
            raise ValueError(f"{path.name} games must be a list")
        seen: set[tuple[str, str]] = set()
        for index, game in enumerate(games):
            if not isinstance(game, dict):
                raise ValueError(f"{path.name} game {index} must be an object")
            posted = _posted_game(
                game,
                source=f"{path.name} game {index}",
                fetched_at=fetched_at,
            )
            if posted is None:
                continue
            key, away, home = posted
            if key in seen:
                raise ValueError(f"{path.name} lists {key[0]} at {key[1]} twice")
            seen.add(key)
            day = by_date[fetched_at.astimezone(OUTPUT_TZ).date()][key]
            if away is not None:
                day["away"].append(away)
            if home is not None:
                day["home"].append(home)
    return by_date


def _read_snapshot(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _fetched_at(raw: object, *, source: str) -> datetime:
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
    return fetched_at


def _posted_game(
    game: dict,
    *,
    source: str,
    fetched_at: datetime,
) -> tuple[tuple[str, str], _PostedLineup | None, _PostedLineup | None] | None:
    away = _posted_side(game.get("away"), source=f"{source} away", fetched_at=fetched_at)
    home = _posted_side(game.get("home"), source=f"{source} home", fetched_at=fetched_at)
    if away is None and home is None:
        return None
    away_team = _team_name(game.get("away"), source=f"{source} away")
    home_team = _team_name(game.get("home"), source=f"{source} home")
    return (away_team, home_team), away, home


def _team_name(side: object, *, source: str) -> str:
    if not isinstance(side, dict):
        raise ValueError(f"{source} must be an object")
    team = side.get("team")
    if not isinstance(team, str) or not team.strip():
        raise ValueError(f"{source} is missing team")
    return team.strip()


def _posted_side(
    side: object,
    *,
    source: str,
    fetched_at: datetime,
) -> _PostedLineup | None:
    if not isinstance(side, dict):
        raise ValueError(f"{source} must be an object")
    lineup = _lineup_players(side.get("lineup"), source=source)
    if not lineup:
        return None
    status = side.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError(f"{source} is missing status")
    return _PostedLineup(
        fetched_at=fetched_at,
        status=status.strip(),
        lineup=lineup,
    )


def _lineup_players(raw: object, *, source: str) -> tuple[dict[str, str | None], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{source} lineup must be a list")
    players: list[dict[str, str | None]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{source} player {index} must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source} player {index} is missing name")
        position = row.get("position")
        if position is not None and not isinstance(position, str):
            raise ValueError(f"{source} player {index} has a non-text position")
        posted = position.strip() if isinstance(position, str) else ""
        players.append({"name": name.strip(), "position": posted or None})
    return tuple(players)


def _closing_games(
    games: dict[tuple[str, str], dict[str, list[_PostedLineup]]],
) -> list[dict]:
    rows: list[dict] = []
    for (away, home), sides in games.items():
        away_posts = sorted(sides["away"], key=lambda post: post.fetched_at)
        home_posts = sorted(sides["home"], key=lambda post: post.fetched_at)
        away_close = _last_confirmed(away_posts)
        home_close = _last_confirmed(home_posts)
        if away_close is None and home_close is None:
            continue
        rows.append(
            {
                "away": away,
                "home": home,
                "away_opening": _board(away_posts[0]) if away_posts else None,
                "away_closing": _board(away_close) if away_close else None,
                "home_opening": _board(home_posts[0]) if home_posts else None,
                "home_closing": _board(home_close) if home_close else None,
            }
        )
    rows.sort(key=lambda row: (row["away"], row["home"]))
    return rows


def _last_confirmed(posts: list[_PostedLineup]) -> _PostedLineup | None:
    chosen: _PostedLineup | None = None
    for post in posts:
        if post.status == "confirmed":
            chosen = post
    return chosen


def _board(post: _PostedLineup) -> dict:
    return {
        "fetched_at": post.fetched_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": post.status,
        "lineup": [dict(player) for player in post.lineup],
    }


def _lineup_day_payload(day: date, games: list[dict]) -> dict:
    return {
        "source": "RotoWire",
        "league": "NBA",
        "date": day.isoformat(),
        "timezone": "America/Los_Angeles",
        "rule": (
            "Opening is the first non-empty lineup posted for that team on this "
            "Pacific date. Closing is the last lineup marked confirmed that date."
        ),
        "game_count": len(games),
        "games": games,
    }


if __name__ == "__main__":
    raise SystemExit(main())
