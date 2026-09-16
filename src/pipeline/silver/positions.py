"""Load and apply player-position enrichment."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Literal

import pandas as pd

LeagueKey = Literal["nba", "wnba"]

logger = logging.getLogger(__name__)

FINE_POSITIONS = frozenset({"PG", "SG", "SF", "PF", "C"})

WNBA_POSITIONS = FINE_POSITIONS | frozenset(
    {"G", "F", "G-F", "F-G", "F-C", "C-F"}
)

POSITION_ALIASES = {
    "Point Guard": "PG",
    "Shooting Guard": "SG",
    "Small Forward": "SF",
    "Power Forward": "PF",
    "Center": "C",
    "Guard": "G",
    "Forward": "F",
    "Guard-Forward": "G-F",
    "Forward-Guard": "F-G",
    "Forward-Center": "F-C",
    "Center-Forward": "C-F",
}

GENERATIONAL_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")
IGNORED_POSITION_NAMES = frozenset({"league average"})

def load_player_positions(
    season: str,
    *,
    league: LeagueKey,
    positions_dir: str | Path,
) -> pd.DataFrame:
    positions_dir = Path(positions_dir)
    path = _resolve_positions_path(
        season,
        league,
        positions_dir,
    )

    if path is None:
        logger.warning(
            "No player-position file found for %s %s",
            league.upper(),
            season,
        )
        return pd.DataFrame()

    frame = pd.read_csv(path, encoding="utf-8-sig")

    if "pos" not in frame:
        logger.warning("%s has no pos column", path)
        return pd.DataFrame()

    name_columns = [
        column
        for column in ("name_s26", "name")
        if column in frame
    ]

    if not name_columns:
        logger.warning("%s has no player-name column", path)
        return pd.DataFrame()

    lookup = _build_name_lookup(frame, name_columns)

    logger.info(
        "Loaded %d player positions from %s",
        len(lookup),
        path,
    )

    return lookup

def enrich_positions(
    frame: pd.DataFrame,
    positions: pd.DataFrame | None,
    *,
    league: LeagueKey,
) -> pd.DataFrame:
    result = frame.copy()
    allowed = (
        WNBA_POSITIONS
        if league == "wnba"
        else FINE_POSITIONS
    )

    result["pos"] = pd.Series(
        None,
        index=result.index,
        dtype="object",
    )

    if (
        positions is not None
        and not positions.empty
        and "player_name" in result
    ):
        lookup = positions.copy()
        lookup["position_csv"] = lookup["position_csv"].map(
            lambda value: canonical_position(value, allowed)
        )
        lookup = lookup.dropna(subset=["position_csv"])

        position_map = lookup.set_index("name_key")["position_csv"].to_dict()
        result["pos"] = result["player_name"].map(
            lambda name: _match_position(name, position_map)
        )

    if "start_position" in result:
        tracking_positions = result["start_position"].map(
            lambda value: canonical_position(value, allowed)
        )
        result["pos"] = result["pos"].fillna(tracking_positions)

    return result

def normalize_player_name(value: object) -> str:
    if not isinstance(value, str):
        return ""

    normalized = unicodedata.normalize("NFKD", value)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace(".", "")

    return " ".join(ascii_name.lower().strip().split())


def player_name_keys(value: object) -> tuple[str, ...]:
    primary = normalize_player_name(value)
    if not primary or primary in IGNORED_POSITION_NAMES:
        return ()

    keys = [primary]
    stripped = GENERATIONAL_SUFFIX.sub("", primary).strip()
    if stripped and stripped != primary:
        keys.append(stripped)

    return tuple(keys)


def _build_name_lookup(
    frame: pd.DataFrame,
    name_columns: list[str],
) -> pd.DataFrame:
    exact: dict[str, str] = {}
    alias_positions: dict[str, set[str]] = {}

    for column in name_columns:
        for name, position in zip(frame[column], frame["pos"]):
            keys = player_name_keys(name)
            if not keys or pd.isna(position):
                continue

            position_text = str(position).strip()
            if not position_text:
                continue

            primary, *aliases = keys
            exact.setdefault(primary, position_text)
            for alias in aliases:
                alias_positions.setdefault(alias, set()).add(position_text)

    merged = dict(exact)
    for alias, positions in alias_positions.items():
        if alias in merged or len(positions) != 1:
            continue
        merged[alias] = next(iter(positions))

    return pd.DataFrame(
        {
            "name_key": list(merged),
            "position_csv": list(merged.values()),
        }
    )


def _match_position(
    name: object,
    position_map: dict[str, str],
) -> str | None:
    for key in player_name_keys(name):
        matched = position_map.get(key)
        if matched is not None:
            return matched
    return None

def canonical_position(
    value: object,
    allowed: frozenset[str],
) -> str | None:
    if value is None or pd.isna(value):
        return None

    raw = str(value).strip()

    if not raw:
        return None

    canonical = POSITION_ALIASES.get(raw, raw.upper())
    return canonical if canonical in allowed else None

def _resolve_positions_path(
    season: str,
    league: LeagueKey,
    positions_dir: Path,
) -> Path | None:
    if league == "wnba":
        year = str(season).strip()[:4]
        candidate = positions_dir / f"wnba_{year}_players.csv"
        return candidate if candidate.exists() else None

    start_year = int(str(season).split("-")[0])
    candidate = positions_dir / f"nba_{start_year + 1}_players.csv"

    if candidate.exists():
        return candidate

    fallback = positions_dir / "player_positions.csv"
    return fallback if fallback.exists() else None