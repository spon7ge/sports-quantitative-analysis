"""Whether a sportsbook game has already started.

A missing or unparseable start is treated as not started, so one bad timestamp
does not drop the slate. An explicit live or final status still drops the game.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

_FRACTION = re.compile(r"(\.\d+)")

# Books spell the same states differently. Pregame labels are absent on purpose:
# a past start time still counts as started even when status says not started.
_STARTED_STATUSES = frozenset(
    {
        "live",
        "inplay",
        "in_play",
        "inprogress",
        "in_progress",
        "started",
        "open_ingame",
        "ongoing",
        "playing",
        "scoring",
        "closed",
        "ended",
        "final",
        "finished",
        "settled",
        "completed",
        "complete",
        "resulted",
        "in_game",
        "ingame",
    }
)


def _status_token(status: object) -> str:
    text = str(status or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def parse_start(raw: object) -> datetime | None:
    """Parse an ISO timestamp or epoch seconds/millis. None when it is not a time."""
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if value > 10_000_000_000:
            value /= 1000.0
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = _FRACTION.sub(lambda match: match.group(1)[:7], text, count=1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def game_has_started(
    start: object = None,
    *,
    status: object = None,
    now: datetime | None = None,
) -> bool:
    """True when the game is live, finished, or its start is before now.

    A start equal to the scrape time is still pregame. The line at that
    instant can be the closing line. Anything later is in progress.
    """
    if _status_token(status) in _STARTED_STATUSES:
        return True
    parsed = parse_start(start)
    if parsed is None:
        return False
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return parsed.astimezone(UTC) < moment.astimezone(UTC)
