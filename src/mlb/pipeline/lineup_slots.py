"""Parse starting lineup slots from Stats API boxscores."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.models.batter_rates import (
    LeagueKPa,
    is_strikeout,
    league_eligible_pas,
    rate_version,
    shrink_batter_k_pa,
)
from src.mlb.schemas import LINEUP_SLOT_COLUMNS, coerce_frame

STARTING_NINE_COLUMNS = (
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
        return pd.DataFrame(columns=list(STARTING_NINE_COLUMNS))

    game = (data.get("gameData") or {}).get("game") or {}
    resolved_pk = (
        game_pk if game_pk is not None else data.get("gamePk") or game.get("pk")
    )
    season = _as_int(game.get("season"))

    live = data.get("liveData") or {}
    box = live.get("boxscore") or data.get("boxscore") or data
    teams = box.get("teams") if isinstance(box, dict) else None
    if not isinstance(teams, dict):
        return pd.DataFrame(columns=list(STARTING_NINE_COLUMNS))

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

    return pd.DataFrame(rows, columns=list(STARTING_NINE_COLUMNS))


def _league_k_pa(pas: pd.DataFrame) -> LeagueKPa:
    strikeouts = pas["event_type"].map(is_strikeout)
    overall = float(strikeouts.mean())
    if pd.isna(overall):
        raise ValueError("cannot freeze lineup rates without eligible league PAs")

    rates = pas.assign(_is_strikeout=strikeouts)
    bats_rates = (
        rates.groupby("batter_bats", dropna=False)["_is_strikeout"]
        .mean()
        .rename("rate")
        .reset_index()
    )
    cell_rates = (
        rates.groupby(["batter_bats", "pitcher_hand"], dropna=False)[
            "_is_strikeout"
        ]
        .mean()
        .rename("rate")
        .reset_index()
    )
    return LeagueKPa(
        overall=overall,
        by_bats={
            str(row.batter_bats): float(row.rate)
            for row in bats_rates.itertuples(index=False)
        },
        by_bats_hand={
            (str(row.batter_bats), str(row.pitcher_hand)): float(row.rate)
            for row in cell_rates.itertuples(index=False)
        },
    )


def freeze_lineup_slot_rates(
    slots: pd.DataFrame,
    batter_pas: pd.DataFrame,
    people: pd.DataFrame,
    config: MlbConfig,
    *,
    cutoff: pd.Timestamp,
    opposing_pitcher_hand: str,
    vs_pitcher_id: int | None,
) -> pd.DataFrame:
    """Freeze cutoff-safe three-window batter K/PA rates onto lineup slots."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    event_times = pd.to_datetime(batter_pas["event_time_utc"], utc=True)
    league_pas = batter_pas.loc[
        (event_times < cutoff)
        & (event_times >= cutoff - pd.Timedelta(days=365))
        & (batter_pas["event_time_imputed"] == 0)
    ].copy()
    league_pas = league_eligible_pas(league_pas, slots)
    league = _league_k_pa(league_pas)

    people_id_column = "mlb_id" if "mlb_id" in people else "batter_id"
    bats_by_id = (
        people.drop_duplicates(people_id_column)
        .set_index(people_id_column)["bats"]
        .to_dict()
    )
    hand = str(opposing_pitcher_hand or "")
    frozen = slots.copy()
    frozen["opposing_pitcher_hand"] = hand
    frozen["vs_pitcher_id"] = vs_pitcher_id
    frozen["rate_version"] = rate_version(config)

    for index, slot in frozen.iterrows():
        batter_id = int(slot["batter_id"])
        bats_value = bats_by_id.get(batter_id, "")
        bats = "" if pd.isna(bats_value) else str(bats_value)
        rates = shrink_batter_k_pa(
            batter_pas,
            batter_id=batter_id,
            opposing_pitcher_hand=hand,
            bats=bats,
            cutoff=cutoff,
            league=league,
            config=config,
        )
        for window, values in rates.items():
            for metric, value in values.items():
                frozen.at[index, f"{metric}_{window}"] = value

    return coerce_frame(frozen, LINEUP_SLOT_COLUMNS)
