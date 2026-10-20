"""RotoWire NBA projected starting fives."""

from __future__ import annotations

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
