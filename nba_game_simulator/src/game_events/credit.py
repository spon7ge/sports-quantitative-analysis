"""Rates and draws that follow the play-by-play links.

Assists are two draws: the make is assisted with probability
P(assisted | shooter, zone), then the passer is chosen with
P(passer | shooter, lineup). Rebound weights are offensive rebounds per
team miss faced on the floor, not per minute. Blocks use rim attempts.
Steals use live-ball turnovers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from game_events.checks import _normalize_id
from game_events.links import EventLinks
from game_events.shares import allocate_joint_assists
from possession_table import ZONES

_OFFENSE = tuple(f"off_{index}" for index in range(1, 6))
_DEFENSE = tuple(f"def_{index}" for index in range(1, 6))
# A shooter-zone cell and a rebound chance are small. Forty observations of
# the league rate keep a couple of makes from becoming the probability.
ASSIST_PRIOR_MAKES = 40.0
CHANCE_PRIOR = 40.0
# Pseudo-assists spread across the teammates on the floor, so one observed
# pass does not name the only possible passer.
PASSER_PRIOR_ASSISTS = 4.0
CHANCE_RATE_COLUMNS = {
    "OREB": "oreb_per_chance",
    "DREB": "dreb_per_chance",
    "STL": "stl_per_live",
    "BLK": "blk_per_rim",
}


@dataclass(frozen=True)
class CreditRates:
    """Counts behind the event-level probabilities.

    ``assisted`` is one row per shooter and zone. ``passers`` is one row per
    shooter-passer pair. Player rebound, block, and steal rows use chances
    faced on the floor as the denominator. ``teams`` holds the pool rates.
    """

    assisted: pd.DataFrame
    passers: pd.DataFrame
    rebounds: pd.DataFrame
    blocks: pd.DataFrame
    steals: pd.DataFrame
    teams: pd.DataFrame


def credit_rates(links: EventLinks) -> CreditRates:
    """Turn event links into per-player and per-team rates."""
    if not isinstance(links, EventLinks):
        raise ValueError("links must be EventLinks")
    return CreditRates(
        assisted=_assisted(links.shots),
        passers=_passers(links.shots),
        rebounds=_rebounds(links.rebounds),
        blocks=_blocks(links.shots),
        steals=_steals(links.turnovers),
        teams=_teams(links),
    )


def passer_tensor(
    creation: np.ndarray,
    counts: np.ndarray,
    strength: float,
    on_floor: np.ndarray,
) -> np.ndarray:
    """``(sim, shooter, passer)`` weights for P(passer | shooter, lineup).

    ``counts[shooter, passer]`` is how often that passer assisted that
    shooter. ``strength`` is a pseudo-count added to every teammate on the
    floor, scaled by ``creation`` when creation is not flat, so an unseen
    teammate can still pass. A player off the floor has weight zero, and a
    player cannot assist himself.
    """
    creation = np.asarray(creation, dtype=float)
    counts = np.asarray(counts, dtype=float)
    on_floor = np.asarray(on_floor, dtype=bool)
    if creation.ndim != 2:
        raise ValueError("creation must have shape (simulations, players)")
    n_sims, n_players = creation.shape
    if counts.shape != (n_players, n_players):
        raise ValueError("passer counts must have shape (players, players)")
    if on_floor.shape != (n_sims, n_players):
        raise ValueError("on_floor must have shape (simulations, players)")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("passer strength must be nonnegative")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("passer counts must be nonnegative")
    if np.any(~np.isfinite(creation)) or np.any(creation < 0.0):
        raise ValueError("creation must be nonnegative")

    affinity = np.broadcast_to(counts, (n_sims, n_players, n_players)).copy()
    if strength > 0.0:
        totals = creation.sum(axis=1, keepdims=True)
        share = np.divide(
            creation, totals, out=np.zeros_like(creation), where=totals > 0
        )
        affinity = affinity + strength * share[:, None, :]
    weights = affinity * on_floor[:, None, :]
    diagonal = np.arange(n_players)
    weights[:, diagonal, diagonal] = 0.0
    return weights


def allocate_zone_assists(
    zone_fgm: np.ndarray,
    p_assisted: np.ndarray,
    passer_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Assist each zone's makes, then hand the assist to that shooter's lineup.

    ``zone_fgm`` is ``(simulations, players, zones)`` in ``ZONES`` order.
    ``p_assisted`` is ``(players, zones)``. ``passer_weights`` is
    ``(simulations, shooter, passer)``.
    """
    made = np.asarray(zone_fgm)
    if made.ndim != 3:
        raise ValueError("zone makes must have shape (simulations, players, zones)")
    n_sims, n_players, n_zones = made.shape
    if n_zones != len(ZONES):
        raise ValueError(f"zone makes must use {ZONES}")
    rate = np.asarray(p_assisted, dtype=float)
    if rate.shape != (n_players, n_zones):
        raise ValueError("p_assisted must have shape (players, zones)")
    if np.any(~np.isfinite(rate)) or np.any((rate < 0.0) | (rate > 1.0)):
        raise ValueError("p_assisted must contain values in [0, 1]")
    weights = np.asarray(passer_weights, dtype=float)
    if weights.shape != (n_sims, n_players, n_players):
        raise ValueError(
            "passer weights must have shape (simulations, shooter, passer)"
        )
    pairs = np.zeros((n_sims, n_players, n_players), dtype=np.int64)
    for zone in range(n_zones):
        if not np.any(made[:, :, zone]):
            continue
        _assists, zone_pairs = allocate_joint_assists(
            made[:, :, zone], rate[:, zone], weights, rng
        )
        pairs += zone_pairs
    return pairs.sum(axis=2), pairs


def profile_with_credit(profile: pd.DataFrame, rates: CreditRates) -> pd.DataFrame:
    """Add chance rates and the team's event probabilities to a roster profile."""
    if not isinstance(profile, pd.DataFrame) or profile.empty:
        raise ValueError("profile must be a non-empty DataFrame")
    if "player_id" not in profile.columns:
        raise ValueError("profile needs player_id")
    frame = profile.copy()
    keys = [_normalize_id(player) for player in frame["player_id"]]
    frame["oreb_per_chance"] = _shrunk_chance(
        rates.rebounds, keys, "oreb", "off_chances"
    )
    frame["dreb_per_chance"] = _shrunk_chance(
        rates.rebounds, keys, "dreb", "def_chances"
    )
    frame["blk_per_rim"] = _shrunk_chance(rates.blocks, keys, "blk", "rim_faced")
    frame["stl_per_live"] = _shrunk_chance(rates.steals, keys, "stl", "live_faced")
    if "team_id" in frame.columns:
        team = _team_row(rates, frame["team_id"].iloc[0])
        if team is not None:
            for column in ("p_rebound", "p_oreb", "p_block", "p_steal"):
                frame[column] = float(team[column])
            for zone in ZONES:
                column = f"{zone}_p_oreb"
                if column in team.index:
                    frame[column] = float(team[column])
    return frame


def assisted_matrix(rates: CreditRates, player_ids: tuple) -> np.ndarray:
    """``(players, zones)`` assisted-make probabilities for this roster."""
    keys = {_normalize_id(player): index for index, player in enumerate(player_ids)}
    out = np.zeros((len(player_ids), len(ZONES)))
    assisted = rates.assisted
    for zone_index, zone in enumerate(ZONES):
        part = (
            assisted.loc[assisted["zone"].eq(zone)] if not assisted.empty else assisted
        )
        makes = float(part["makes"].sum()) if len(part) else 0.0
        league = float(part["assisted"].sum()) / makes if makes else 0.0
        out[:, zone_index] = league
        for row in part.itertuples(index=False):
            index = keys.get(_normalize_id(row.player_id))
            if index is not None:
                out[index, zone_index] = (
                    float(row.assisted) + ASSIST_PRIOR_MAKES * league
                ) / (float(row.makes) + ASSIST_PRIOR_MAKES)
    return out


def passer_counts_matrix(rates: CreditRates, player_ids: tuple) -> np.ndarray:
    """``(shooter, passer)`` assist counts limited to this roster."""
    keys = {_normalize_id(player): index for index, player in enumerate(player_ids)}
    out = np.zeros((len(player_ids), len(player_ids)))
    for row in rates.passers.itertuples(index=False):
        shooter = keys.get(_normalize_id(row.shooter_id))
        passer = keys.get(_normalize_id(row.passer_id))
        if shooter is None or passer is None or shooter == passer:
            continue
        out[shooter, passer] += float(row.assists)
    return out


def _assisted(shots: pd.DataFrame) -> pd.DataFrame:
    columns = ["player_id", "zone", "makes", "assisted", "p_assisted"]
    makes = shots.loc[shots["made"].astype(bool)] if not shots.empty else shots
    if makes.empty:
        return pd.DataFrame(columns=columns)
    grouped = makes.groupby(["shooter_id", "zone"], sort=False)
    out = grouped.agg(
        makes=("made", "size"), assisted=("assisted", "sum")
    ).reset_index()
    out = out.rename(columns={"shooter_id": "player_id"})
    out["p_assisted"] = out["assisted"] / out["makes"]
    return out.loc[:, columns]


def _passers(shots: pd.DataFrame) -> pd.DataFrame:
    columns = ["passer_id", "shooter_id", "assists"]
    if shots.empty:
        return pd.DataFrame(columns=columns)
    paired = shots.loc[shots["passer_id"].notna()]
    if paired.empty:
        return pd.DataFrame(columns=columns)
    return (
        paired.groupby(["passer_id", "shooter_id"], sort=False)
        .size()
        .reset_index(name="assists")
        .loc[:, columns]
    )


def _rebounds(chances: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player_id",
        "off_chances",
        "oreb",
        "oreb_per_chance",
        "def_chances",
        "dreb",
        "dreb_per_chance",
    ]
    offensive: dict[int, int] = {}
    defensive: dict[int, int] = {}
    grabbed_off: dict[int, int] = {}
    grabbed_def: dict[int, int] = {}
    for row in chances.itertuples(index=False):
        for player in _five(row, _OFFENSE):
            offensive[player] = offensive.get(player, 0) + 1
            if row.outcome == "oreb" and _same(row.rebounder_id, player):
                grabbed_off[player] = grabbed_off.get(player, 0) + 1
        for player in _five(row, _DEFENSE):
            defensive[player] = defensive.get(player, 0) + 1
            if row.outcome == "dreb" and _same(row.rebounder_id, player):
                grabbed_def[player] = grabbed_def.get(player, 0) + 1
    rows = []
    for player in sorted(set(offensive) | set(defensive)):
        off_chances = offensive.get(player, 0)
        def_chances = defensive.get(player, 0)
        oreb = grabbed_off.get(player, 0)
        dreb = grabbed_def.get(player, 0)
        rows.append(
            {
                "player_id": player,
                "off_chances": off_chances,
                "oreb": oreb,
                "oreb_per_chance": oreb / off_chances if off_chances else np.nan,
                "def_chances": def_chances,
                "dreb": dreb,
                "dreb_per_chance": dreb / def_chances if def_chances else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _blocks(shots: pd.DataFrame) -> pd.DataFrame:
    columns = ["player_id", "rim_faced", "blk", "blk_per_rim"]
    faced: dict[int, int] = {}
    blocks: dict[int, int] = {}
    rim = shots.loc[shots["zone"].eq("rim")] if not shots.empty else shots
    for row in rim.itertuples(index=False):
        for player in _five(row, _DEFENSE):
            faced[player] = faced.get(player, 0) + 1
            if bool(row.blocked) and _same(row.blocker_id, player):
                blocks[player] = blocks.get(player, 0) + 1
    rows = [
        {
            "player_id": player,
            "rim_faced": count,
            "blk": blocks.get(player, 0),
            "blk_per_rim": blocks.get(player, 0) / count,
        }
        for player, count in faced.items()
    ]
    return pd.DataFrame(rows, columns=columns)


def _steals(turnovers: pd.DataFrame) -> pd.DataFrame:
    columns = ["player_id", "live_faced", "stl", "stl_per_live"]
    faced: dict[int, int] = {}
    steals: dict[int, int] = {}
    live = (
        turnovers.loc[turnovers["live_ball"].astype(bool)]
        if not turnovers.empty
        else turnovers
    )
    for row in live.itertuples(index=False):
        for player in _five(row, _DEFENSE):
            faced[player] = faced.get(player, 0) + 1
            if _same(row.stealer_id, player):
                steals[player] = steals.get(player, 0) + 1
    rows = [
        {
            "player_id": player,
            "live_faced": count,
            "stl": steals.get(player, 0),
            "stl_per_live": steals.get(player, 0) / count,
        }
        for player, count in faced.items()
    ]
    return pd.DataFrame(rows, columns=columns)


def _teams(links: EventLinks) -> pd.DataFrame:
    columns = [
        "team_id",
        "misses",
        "player_rebounds",
        "oreb",
        "team_rebounds",
        "rim_attempts",
        "blocks",
        "live_turnovers",
        "steals",
        "p_rebound",
        "p_oreb",
        "p_block",
        "p_steal",
        *(f"{zone}_p_oreb" for zone in ZONES),
    ]
    found: set[int] = set()
    for frame in (links.shots, links.rebounds, links.turnovers):
        for column in ("offense_team_id", "defense_team_id"):
            if column in frame.columns:
                found.update(int(team) for team in frame[column].dropna().unique())
    rows = []
    for team in sorted(found):
        misses = links.rebounds.loc[links.rebounds["offense_team_id"].eq(team)]
        n_miss = int(len(misses))
        player_rebounds = (
            int(misses["outcome"].isin(["oreb", "dreb"]).sum()) if n_miss else 0
        )
        oreb = int(misses["outcome"].eq("oreb").sum()) if n_miss else 0
        team_rebounds = int(misses["outcome"].eq("team").sum()) if n_miss else 0
        league_oreb = oreb / n_miss if n_miss else 0.0
        zone_oreb = _zone_oreb_rates(misses, league_oreb)
        rim = links.shots.loc[
            links.shots["zone"].eq("rim") & links.shots["defense_team_id"].eq(team)
        ]
        rim_attempts = int(len(rim))
        blocks = int(rim["blocked"].astype(bool).sum()) if rim_attempts else 0
        live = links.turnovers.loc[
            links.turnovers["live_ball"].astype(bool)
            & links.turnovers["defense_team_id"].eq(team)
        ]
        live_turnovers = int(len(live))
        steals = int(live["stealer_id"].notna().sum()) if live_turnovers else 0
        rows.append(
            {
                "team_id": team,
                "misses": n_miss,
                "player_rebounds": player_rebounds,
                "oreb": oreb,
                "team_rebounds": team_rebounds,
                "rim_attempts": rim_attempts,
                "blocks": blocks,
                "live_turnovers": live_turnovers,
                "steals": steals,
                "p_rebound": player_rebounds / n_miss if n_miss else 0.0,
                "p_oreb": oreb / player_rebounds if player_rebounds else 0.0,
                "p_block": blocks / rim_attempts if rim_attempts else 0.0,
                "p_steal": steals / live_turnovers if live_turnovers else 0.0,
                **zone_oreb,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _zone_oreb_rates(misses: pd.DataFrame, league: float) -> dict[str, float]:
    """P(offensive rebound | miss in this zone), shrunk toward the team rate."""

    rates = {}
    for zone in ZONES:
        if misses.empty or "zone" not in misses.columns:
            made = 0
            chances = 0
        else:
            zone_rows = misses.loc[misses["zone"].eq(zone)]
            chances = int(len(zone_rows))
            made = int(zone_rows["outcome"].eq("oreb").sum()) if chances else 0
        rates[f"{zone}_p_oreb"] = (made + CHANCE_PRIOR * league) / (
            chances + CHANCE_PRIOR
        )
    return rates


def _five(row, columns: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(getattr(row, column)) for column in columns)


def _same(left, right: int) -> bool:
    return pd.notna(left) and int(left) == int(right)


def _shrunk_chance(
    table: pd.DataFrame,
    keys: list[str],
    events: str,
    chances: str,
) -> list[float]:
    """Events per chance, pulled toward the rate in this table."""

    if table.empty or events not in table.columns or chances not in table.columns:
        return [0.0] * len(keys)

    indexed = table.copy()
    indexed["_key"] = indexed["player_id"].map(_normalize_id)
    indexed = indexed.drop_duplicates("_key").set_index("_key")
    event_total = float(pd.to_numeric(indexed[events], errors="coerce").fillna(0).sum())
    chance_total = float(
        pd.to_numeric(indexed[chances], errors="coerce").fillna(0).sum()
    )
    league = event_total / chance_total if chance_total else 0.0
    values = []
    for key in keys:
        if key not in indexed.index:
            values.append(league)
            continue
        event = _count(indexed.at[key, events])
        chance = _count(indexed.at[key, chances])
        values.append((event + CHANCE_PRIOR * league) / (chance + CHANCE_PRIOR))
    return values


def _count(value) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return 0.0
    return float(number)


def _team_row(rates: CreditRates, team_id) -> pd.Series | None:
    if rates.teams.empty:
        return None
    key = _normalize_id(team_id)
    teams = rates.teams.copy()
    teams["_key"] = teams["team_id"].map(_normalize_id)
    matched = teams.loc[teams["_key"].eq(key)]
    if matched.empty:
        return None
    return matched.iloc[0]
