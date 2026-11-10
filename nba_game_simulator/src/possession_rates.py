"""Rates a possession table can support.

Usage is field-goal attempts divided by offensive possessions with that player
on the floor. With-teammate usage is shrunk hard toward the player's own
unconditional rate, because the conditional sample is small and was previously
measured only while the star was beside him. Each shot zone is shrunk on its
own, toward the league rate in the same table. Garbage possessions and
possessions whose lineup changed mid-trip are left out unless a garbage weight
is supplied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pbp_guard import rows_before
from possession_table import ZONES

# A few hundred possessions of prior keeps a short on/off split from moving
# the rate. Eighty attempts does the same for one zone.
USAGE_PRIOR_POSSESSIONS = 400.0
ZONE_PRIOR_ATTEMPTS = 80.0

# Eighty free-throw possessions of prior keeps a short foul sample near the league.
FTA_PRIOR_POSSESSIONS = 80.0

_OFFENSE = ("off_1", "off_2", "off_3", "off_4", "off_5")
_TRIP_KINDS = ("shooting2", "shooting3", "and1", "bonus")


def usage_rate(
    table: pd.DataFrame,
    player_id: int,
    *,
    garbage_weight: float = 0.0,
) -> float:
    """Field-goal attempts per offensive possession on the floor."""

    weights, on = _exposure(table, player_id, garbage_weight)
    possessions = float(weights[on].sum())
    if possessions <= 0:
        raise ValueError(f"player {player_id} has no offensive possessions")

    return _field_goals(table, weights, on, player_id) / possessions


def shrunk_usage(
    table: pd.DataFrame,
    player_id: int,
    teammate_id: int,
    *,
    teammate_on: bool,
    prior_possessions: float = USAGE_PRIOR_POSSESSIONS,
    garbage_weight: float = 0.0,
    before=None,
) -> float:
    """Usage with or without one teammate, pulled toward the unconditional rate.

    ``before`` keeps possessions whose date is strictly earlier. The slate
    date itself is not a with/without sample.
    """

    if before is not None:
        table = rows_before(table, before)
    if player_id == teammate_id:
        raise ValueError("teammate must be a different player")
    if prior_possessions < 0:
        raise ValueError("prior_possessions must be nonnegative")

    weights, on = _exposure(table, player_id, garbage_weight)
    unconditional_possessions = float(weights[on].sum())
    if unconditional_possessions <= 0:
        raise ValueError(f"player {player_id} has no offensive possessions")

    unconditional_attempts = _field_goals(table, weights, on, player_id)
    unconditional = unconditional_attempts / unconditional_possessions
    mate = _on_offense(table, teammate_id)
    if not teammate_on:
        mate = ~mate

    conditional = on & mate
    conditional_possessions = float(weights[conditional].sum())
    conditional_attempts = _field_goals(table, weights, conditional, player_id)
    return (conditional_attempts + prior_possessions * unconditional) / (
        conditional_possessions + prior_possessions
    )


def zone_make_rates(
    table: pd.DataFrame,
    player_id: int,
    *,
    prior_attempts: float = ZONE_PRIOR_ATTEMPTS,
    garbage_weight: float = 0.0,
) -> dict[str, float]:
    """Make rate in each zone, shrunk toward that zone's league rate."""

    if prior_attempts < 0:
        raise ValueError("prior_attempts must be nonnegative")

    weights = _weights(table, garbage_weight)
    player = _zone_counts(table, weights, player_id)
    league = _zone_counts(table, weights, None)
    rates = {}

    for zone in ZONES:
        league_makes, league_attempts = league[zone]
        prior_rate = league_makes / league_attempts if league_attempts else 0.0
        makes, attempts = player[zone]
        denominator = attempts + prior_attempts
        if denominator <= 0:
            raise ValueError(f"no attempts to shrink for {zone}")
        rates[zone] = (makes + prior_attempts * prior_rate) / denominator

    return rates


def zone_shot_mix(
    table: pd.DataFrame,
    player_id: int,
    *,
    prior_attempts: float = ZONE_PRIOR_ATTEMPTS,
    garbage_weight: float = 0.0,
) -> dict[str, float]:
    """Share of a player's shots in each zone, shrunk toward the league mix."""

    if prior_attempts < 0:
        raise ValueError("prior_attempts must be nonnegative")

    weights = _weights(table, garbage_weight)
    player = _zone_counts(table, weights, player_id)
    league = _zone_counts(table, weights, None)
    league_total = sum(league[zone][1] for zone in ZONES)
    if league_total <= 0:
        raise ValueError("the table has no field-goal attempts")

    player_total = sum(player[zone][1] for zone in ZONES)
    denominator = player_total + prior_attempts
    mix = {}
    for zone in ZONES:
        league_share = league[zone][1] / league_total
        mix[zone] = (player[zone][1] + prior_attempts * league_share) / denominator
    return mix


def ft_structure(
    table: pd.DataFrame,
    player_id: int,
    *,
    garbage_weight: float = 0.0,
) -> dict:
    """Shooting-foul, and-one, and bonus trips per offensive possession.

    And-ones keep the zone of the made shot. Shooting fouls are split by
    whether the trip was two or three throws, which is the shot value the
    whistle replaced. Bonus trips are the non-shooting fouls that sent him
    to the line and left his team with the ball.
    """

    weights, on = _exposure(table, player_id, garbage_weight)
    possessions = float(weights[on].sum())
    if possessions <= 0:
        raise ValueError(f"player {player_id} has no offensive possessions")

    _require(table, ("ft_kinds", "ft_player_ids", "ft_zones"))
    league_poss = float(weights.sum())
    league_trips, league_zones = _trip_tally(table, weights, None)
    trips, and1_zones = _trip_tally(table, weights, player_id)
    and1_total = sum(and1_zones.values())
    league_and1 = sum(league_zones.values())
    share = {}
    for zone in ZONES:
        league_share = league_zones[zone] / league_and1 if league_and1 else 0.0
        share[zone] = (and1_zones[zone] + ZONE_PRIOR_ATTEMPTS * league_share) / (
            and1_total + ZONE_PRIOR_ATTEMPTS
        )

    rates = {}
    for kind in _TRIP_KINDS:
        league_rate = league_trips[kind] / league_poss if league_poss else 0.0
        rates[f"{kind}_per_poss"] = (
            trips[kind] + FTA_PRIOR_POSSESSIONS * league_rate
        ) / (possessions + FTA_PRIOR_POSSESSIONS)
    rates["and1_zone_share"] = share
    return rates


def possession_player_rates(table: pd.DataFrame, player_ids) -> pd.DataFrame:
    """Shot and foul rates, one row per player, for the Layer 4 draw.

    ``fga_per_min`` holds field goals per offensive possession, shrunk toward
    the table's average player usage. The allocator multiplies that by minutes
    on the floor, so the share is usage times playing time. Zone shares and
    make rates are the four-zone draw. Free throws per possession are shrunk
    toward the table's own rate.
    """

    if not isinstance(table, pd.DataFrame):
        raise ValueError("rates need a possession DataFrame")

    columns = [
        "player_id",
        "fga_per_min",
        "p_3pa",
        "p_2p",
        "p_3p",
        "fta_per_min",
        "ft_pct",
        *(f"{zone}_share" for zone in ZONES),
        *(f"{zone}_fg_pct" for zone in ZONES),
    ]
    ids: list[int] = []
    for player in player_ids:
        player = int(player)
        if player not in ids:
            ids.append(player)
    if not ids:
        return pd.DataFrame(columns=columns)

    weights = _weights(table, 0.0)
    league_poss = float(weights.sum())
    league_usage = _league_usage(table, weights)
    league_fta, league_ftm = _ft_counts(table, weights, None)
    league_fta_per = league_fta / league_poss if league_poss else 0.0
    league_ft_pct = league_ftm / league_fta if league_fta else 0.0
    rows = []

    for player in ids:
        _, on = _exposure(table, player, 0.0)
        possessions = float(weights[on].sum())
        if possessions <= 0:
            continue
        attempts = _field_goals(table, weights, on, player)
        usage = (attempts + USAGE_PRIOR_POSSESSIONS * league_usage) / (
            possessions + USAGE_PRIOR_POSSESSIONS
        )
        mix = zone_shot_mix(table, player)
        makes = zone_make_rates(table, player)
        two = mix["rim"] + mix["mid"]
        three = mix["corner3"] + mix["atb3"]
        p_2p = (
            (mix["rim"] * makes["rim"] + mix["mid"] * makes["mid"]) / two
            if two
            else 0.0
        )
        p_3p = (
            (mix["corner3"] * makes["corner3"] + mix["atb3"] * makes["atb3"]) / three
            if three
            else 0.0
        )
        fta, ftm = _ft_counts(table, weights, player)
        fta_per = (fta + FTA_PRIOR_POSSESSIONS * league_fta_per) / (
            possessions + FTA_PRIOR_POSSESSIONS
        )
        ft_denominator = fta + FTA_PRIOR_POSSESSIONS * league_fta_per
        ft_pct = (
            (ftm + FTA_PRIOR_POSSESSIONS * league_fta_per * league_ft_pct)
            / ft_denominator
            if ft_denominator
            else np.nan
        )
        row = {
            "player_id": player,
            "fga_per_min": usage,
            "p_3pa": three,
            "p_2p": p_2p,
            "p_3p": p_3p,
            "fta_per_min": fta_per,
            "ft_pct": ft_pct,
        }
        row.update({f"{zone}_share": mix[zone] for zone in ZONES})
        row.update({f"{zone}_fg_pct": makes[zone] for zone in ZONES})
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def _exposure(table: pd.DataFrame, player_id: int, garbage_weight: float):
    _require(table, (*_OFFENSE, "garbage", "shot_player_ids"))
    return _weights(table, garbage_weight), _on_offense(table, player_id)


def _weights(table: pd.DataFrame, garbage_weight: float) -> np.ndarray:
    if garbage_weight < 0 or garbage_weight > 1:
        raise ValueError("garbage_weight must be between 0 and 1")

    weights = np.ones(len(table), dtype=float)
    weights[table["garbage"].to_numpy(dtype=bool)] = garbage_weight
    if "lineup_changed" in table.columns:
        weights[table["lineup_changed"].to_numpy(dtype=bool)] = 0.0
    return weights


def _on_offense(table: pd.DataFrame, player_id: int) -> np.ndarray:
    mask = np.zeros(len(table), dtype=bool)
    for column in _OFFENSE:
        mask |= table[column].to_numpy() == player_id
    return mask


def _league_usage(table: pd.DataFrame, weights: np.ndarray) -> float:
    """Field goals per offensive player-possession. Five men share each trip."""

    shots = 0.0
    slots = 0.0
    shooters = table["shot_player_ids"]
    for index in range(len(table)):
        weight = float(weights[index])
        if weight == 0.0:
            continue
        shots += weight * len(list(shooters.iat[index]))
        slots += weight * len(_OFFENSE)
    if slots <= 0.0:
        return 0.0
    return shots / slots


def _trip_tally(table, weights, player_id: int | None):
    trips = {kind: 0.0 for kind in _TRIP_KINDS}
    zones = {zone: 0.0 for zone in ZONES}
    for index in range(len(table)):
        weight = float(weights[index])
        if weight == 0.0:
            continue
        for kind, player, zone in zip(
            table["ft_kinds"].iat[index],
            table["ft_player_ids"].iat[index],
            table["ft_zones"].iat[index],
            strict=True,
        ):
            if player_id is not None and player != player_id:
                continue
            if kind not in trips:
                continue
            trips[kind] += weight
            if kind == "and1" and zone in zones:
                zones[zone] += weight
    return trips, zones


def _field_goals(table, weights, mask, player_id: int) -> float:
    total = 0.0
    shooters = table["shot_player_ids"]
    for index in np.flatnonzero(mask):
        weight = float(weights[index])
        if weight == 0:
            continue
        attempts = sum(1 for player in shooters.iat[index] if player == player_id)
        total += weight * attempts
    return total


def _zone_counts(table, weights, player_id: int | None) -> dict[str, list[float]]:
    _require(table, ("shot_player_ids", "shot_zones", "shot_made", "garbage"))
    counts = {zone: [0.0, 0.0] for zone in ZONES}

    for index in range(len(table)):
        weight = float(weights[index])
        if weight == 0:
            continue
        for player, zone, made in zip(
            table["shot_player_ids"].iat[index],
            table["shot_zones"].iat[index],
            table["shot_made"].iat[index],
            strict=True,
        ):
            if player_id is not None and player != player_id:
                continue
            if zone not in counts:
                raise ValueError(f"unknown shot zone {zone!r}")
            counts[zone][1] += weight
            if made:
                counts[zone][0] += weight

    return counts


def _ft_counts(table, weights, player_id: int | None) -> tuple[float, float]:
    _require(table, ("ft_player_ids", "ft_fta", "ft_ftm"))
    attempts = 0.0
    makes = 0.0

    for index in range(len(table)):
        weight = float(weights[index])
        if weight == 0:
            continue
        for player, taken, made in zip(
            table["ft_player_ids"].iat[index],
            table["ft_fta"].iat[index],
            table["ft_ftm"].iat[index],
            strict=True,
        ):
            if player_id is not None and player != player_id:
                continue
            attempts += weight * taken
            makes += weight * made

    return attempts, makes


def _require(table: pd.DataFrame, columns) -> None:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("rates need a possession DataFrame")

    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"possession table needs {missing}")
