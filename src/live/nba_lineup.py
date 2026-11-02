"""RotoWire NBA projected starting fives."""

from __future__ import annotations

from src.pipeline.silver.positions import player_name_keys
from src.scrapers.nba.nba_lineups import NBADailyLineups


def projected_lineups(html: str | None = None) -> dict[str, list[str]]:
    """Expected starting five for each posted team.

    Keys are the RotoWire abbreviations. Values are the five names in
    lineup order. Pass ``html`` to parse a saved page instead of fetching.
    """
    lineups = NBADailyLineups(html=html)
    starters = lineups.expected_starters_by_abbr()
    return {
        abbreviation: [str(player["name"]) for player in five]
        for abbreviation, five in starters.items()
    }


def get_starter_status(
    name: str,
    date: str,
    lineups: dict[str, list[str]] | None = None,
) -> int:
    """1 when RotoWire lists the player in a starting five, otherwise 0.

    ``date`` is the slate being priced. The posted page is the current
    slate, so a passed-in ``lineups`` map is what a past date must use.
    """
    del date
    posted = projected_lineups() if lineups is None else lineups
    wanted = set(player_name_keys(name))
    if not wanted:
        return 0
    for starters in posted.values():
        for starter in starters:
            if wanted & set(player_name_keys(starter)):
                return 1
    return 0
