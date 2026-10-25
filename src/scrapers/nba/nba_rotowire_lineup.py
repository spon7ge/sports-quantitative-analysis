"""Write one RotoWire NBA lineup snapshot.

Usage:
    python -m src.scrapers.nba.nba_rotowire_lineup
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.scrapers.nba.nba_lineups import NBADailyLineups
from src.scrapers.nba.paths import ensure_repo_on_path

OUTPUT_TZ = ZoneInfo("America/Los_Angeles")


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


if __name__ == "__main__":
    raise SystemExit(main())
