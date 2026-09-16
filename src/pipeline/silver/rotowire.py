"""Load and apply NBA Rotowire betting-line enrichment."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

LeagueKey = Literal["nba", "wnba"]

logger = logging.getLogger(__name__)

TEAM_ABBREVIATION_ALIASES = {
    "PHO": "PHX",
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
    "NY": "NYK",
    "UTAH": "UTA",
    "WSH": "WAS",
    "CHA": "CHO",
}

def load_rotowire(
    season: str,
    *,
    league: LeagueKey,
    rotowire_dir: str | Path,
    auto_scrape: bool = False,
) -> pd.DataFrame:
    if league != "nba":
        return pd.DataFrame()

    year = str(season).split("-")[0]
    path = Path(rotowire_dir) / f"rotowire_nba_{year}.csv"

    if not path.exists() and auto_scrape:
        _scrape_rotowire(year, path)

    if not path.exists():
        logger.warning(
            "Rotowire file not found: %s; skipping enrichment",
            path,
        )
        return pd.DataFrame()

    frame = pd.read_csv(path)
    required = {"Game", "Tipoff"}

    if not required.issubset(frame.columns):
        logger.warning(
            "Rotowire file is missing columns: %s",
            sorted(required - set(frame.columns)),
        )
        return pd.DataFrame()

    games = frame["Game"].str.split(
        r"\s*@\s*",
        n=1,
        expand=True,
        regex=True,
    )

    if games.shape[1] != 2:
        logger.warning("Rotowire Game values have an unexpected format")
        return pd.DataFrame()

    frame["rw_away"] = games[0].str.strip().map(
        normalize_team_abbreviation
    )
    frame["rw_home"] = games[1].str.strip().map(
        normalize_team_abbreviation
    )
    frame["rw_date"] = _parse_tipoff_dates(frame, int(year))

    output_columns = [
        "rw_date",
        "rw_away",
        "rw_home",
        "Over_Under",
        "Home_Line",
    ]

    missing = set(output_columns) - set(frame.columns)

    if missing:
        logger.warning(
            "Rotowire file is missing columns: %s",
            sorted(missing),
        )
        return pd.DataFrame()

    return frame[output_columns].rename(
        columns={
            "Over_Under": "game_total",
            "Home_Line": "team_spread",
        }
    )

def enrich_rotowire(
    frame: pd.DataFrame,
    rotowire: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or rotowire.empty:
        return frame.copy()

    if not {"game_date", "matchup"}.issubset(frame.columns):
        logger.warning(
            "game_date or matchup is missing; skipping Rotowire"
        )
        return frame.copy()

    result = frame.copy()
    lines = rotowire.copy()

    result["_rw_date"] = pd.to_datetime(
        result["game_date"],
        errors="coerce",
    ).dt.normalize()

    lines["rw_date"] = pd.to_datetime(
        lines["rw_date"],
        errors="coerce",
    ).dt.normalize()

    matchups = result["matchup"].map(parse_matchup)
    result["_rw_away"] = matchups.map(lambda value: value[0])
    result["_rw_home"] = matchups.map(lambda value: value[1])

    merged = result.merge(
        lines,
        left_on=["_rw_date", "_rw_away", "_rw_home"],
        right_on=["rw_date", "rw_away", "rw_home"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    unmatched_rate = float(merged["_merge"].eq("left_only").mean())
    if unmatched_rate:
        logger.warning(
            "Rotowire unmatched %.1f%% of rows; check season file and archive gaps",
            100 * unmatched_rate,
        )

    home_line = pd.to_numeric(merged["team_spread"], errors="coerce")
    if "team_abbreviation" in merged:
        player_team = merged["team_abbreviation"].map(
            normalize_team_abbreviation
        )
        home = player_team.eq(merged["_rw_home"])
        away = player_team.eq(merged["_rw_away"])
        signed = home_line.where(home, -home_line.where(away))
        merged["player_team_spread"] = signed
    else:
        merged["player_team_spread"] = pd.NA

    return merged.drop(
        columns=[
            "_rw_date",
            "_rw_away",
            "_rw_home",
            "rw_date",
            "rw_away",
            "rw_home",
            "_merge",
        ]
    )

def parse_matchup(
    matchup: object,
) -> tuple[str | None, str | None]:
    value = str(matchup)

    if " @ " in value:
        away, home = value.split(" @ ", 1)
    elif " vs. " in value:
        home, away = value.split(" vs. ", 1)
    else:
        return None, None

    return (
        normalize_team_abbreviation(away),
        normalize_team_abbreviation(home),
    )

def normalize_team_abbreviation(value: object) -> str:
    abbreviation = str(value).strip()
    return TEAM_ABBREVIATION_ALIASES.get(
        abbreviation,
        abbreviation,
    )

TIPOFF_FORMAT = "%b %d %I:%M %p %Y"
_MONTH_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_tipoff_dates(
    frame: pd.DataFrame,
    default_year: int,
) -> pd.Series:
    if "Season" in frame:
        years = pd.to_numeric(
            frame["Season"],
            errors="coerce",
        ).fillna(default_year).astype(int)
    else:
        years = pd.Series(
            default_year,
            index=frame.index,
        )

    tipoff = frame["Tipoff"].astype(str).str.strip()
    month = (
        tipoff.str.extract(r"^([A-Za-z]+)", expand=False)
        .str.lower()
        .map(_MONTH_NUMBER)
    )
    rolls_forward = month.between(1, 7)
    calendar_year = years.where(~rolls_forward, years + 1)

    parsed = pd.to_datetime(
        tipoff + " " + calendar_year.astype(str),
        format=TIPOFF_FORMAT,
        errors="coerce",
    )

    return parsed.dt.normalize()

def _scrape_rotowire(
    season: str,
    output_path: Path,
) -> None:
    try:
        from src.scrapers.rotowire_nba_odds import run_scrape

        output_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(
            run_scrape(
                season=season,
                output_file=output_path,
            )
        )
    except Exception:
        logger.exception(
            "Rotowire auto-scrape failed for season %s",
            season,
        )