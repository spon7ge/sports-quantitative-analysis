"""Parse starting lineup slots from Stats API boxscores."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

LINEUP_SLOT_COLUMNS = (
    "game_pk",
    "team_id",
    "side",
    "slot",
    "batter_id",
    "slot_is_pitcher",
    "season",
)


def _as_text(raw_payload: str | bytes) -> str:
    if isinstance(raw_payload, bytes):
        return raw_payload.decode("utf-8-sig")
    return raw_payload


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_starting_nine(
    raw_payload: str | bytes,
    game_pk: int | None = None,
) -> pd.DataFrame:
    """Return each side's original 1-9 from player ``battingOrder`` 00 codes."""
    data = json.loads(_as_text(raw_payload))
    if not isinstance(data, dict):
        return pd.DataFrame(columns=list(LINEUP_SLOT_COLUMNS))

    game = (data.get("gameData") or {}).get("game") or {}
    resolved_pk = (
        game_pk if game_pk is not None else data.get("gamePk") or game.get("pk")
    )
    season = _as_int(game.get("season"))

    live = data.get("liveData") or {}
    box = live.get("boxscore") or data.get("boxscore") or data
    teams = box.get("teams") if isinstance(box, dict) else None
    if not isinstance(teams, dict):
        return pd.DataFrame(columns=list(LINEUP_SLOT_COLUMNS))

    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = teams.get(side) or {}
        team_id = (team.get("team") or {}).get("id")
        selected: dict[int, dict[str, Any]] = {}
        players = team.get("players") or {}
        if not isinstance(players, dict):
            continue

        for player_key, player in players.items():
            if not isinstance(player, dict):
                continue
            batting_order = player.get("battingOrder")
            if batting_order is None or batting_order == "":
                continue
            order_text = str(batting_order)
            if not order_text.endswith("0"):
                continue
            if len(order_text) == 3:
                slot = _as_int(order_text[0])
            elif len(order_text) == 1:
                slot = _as_int(order_text)
            else:
                continue
            if slot not in range(1, 10):
                continue

            position = ((player.get("position") or {}).get("abbreviation") or "")
            is_pitcher = int(str(position).upper() == "P")
            person = player.get("person") or {}
            batter_id = person.get("id")
            if batter_id is None:
                batter_id = str(player_key).removeprefix("ID")
            candidate = {
                "game_pk": resolved_pk,
                "team_id": team_id,
                "side": side,
                "slot": slot,
                "batter_id": _as_int(batter_id),
                "slot_is_pitcher": is_pitcher,
                "season": season,
            }
            current = selected.get(slot)
            if current is None or (
                current["slot_is_pitcher"] == 1 and candidate["slot_is_pitcher"] == 0
            ):
                selected[slot] = candidate

        rows.extend(selected[slot] for slot in sorted(selected))

    return pd.DataFrame(rows, columns=list(LINEUP_SLOT_COLUMNS))
