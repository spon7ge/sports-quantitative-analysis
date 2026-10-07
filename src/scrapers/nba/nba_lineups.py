"""RotoWire daily NBA starting lineups.

The expected five is the posted starting lineup. A player in that five has
``starting`` 1. Everyone else on the team has 0.

Example::

    lineups = NBADailyLineups()
    starters = lineups.expected_starters_by_abbr()
"""

from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.pipeline.silver.positions import player_name_keys
from src.pipeline.silver.rotowire import normalize_team_abbreviation

NBA_LINEUPS_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

# First token of a RotoWire matchup label → NBA abbreviation.
NBA_TEAM_ABBREVIATIONS = {
    "Hawks": "ATL",
    "Celtics": "BOS",
    "Nets": "BKN",
    "Hornets": "CHA",
    "Bulls": "CHI",
    "Cavaliers": "CLE",
    "Mavericks": "DAL",
    "Nuggets": "DEN",
    "Pistons": "DET",
    "Warriors": "GSW",
    "Rockets": "HOU",
    "Pacers": "IND",
    "Clippers": "LAC",
    "Lakers": "LAL",
    "Grizzlies": "MEM",
    "Heat": "MIA",
    "Bucks": "MIL",
    "Timberwolves": "MIN",
    "Pelicans": "NOP",
    "Knicks": "NYK",
    "Thunder": "OKC",
    "Magic": "ORL",
    "76ers": "PHI",
    "Suns": "PHX",
    "Trail": "POR",
    "Kings": "SAC",
    "Spurs": "SAS",
    "Raptors": "TOR",
    "Jazz": "UTA",
    "Wizards": "WAS",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_EXPECTED_TITLES = {"Very Likely To Play", "Likely To Play", "Toss Up To Play"}
_GTD_TITLES = {"Likely To Play", "Toss Up To Play"}
_GTD_STATUS = {"gtd", "ques", "questionable", "doubtful"}


def _class_tokens(classes) -> set[str]:
    if isinstance(classes, str):
        return set(classes.split())
    if isinstance(classes, (list, tuple)):
        return set(classes)
    return set()


def _is_nba_lineup_card(classes) -> bool:
    tokens = _class_tokens(classes)
    return "lineup" in tokens and "is-nba" in tokens


class NBADailyLineups:
    """Scrape RotoWire expected NBA starting fives."""

    def __init__(self, html: str | None = None, url: str | None = None):
        self.url = url or NBA_LINEUPS_URL
        self.data: list[dict] = []
        self.soup = (
            BeautifulSoup(html, "html.parser")
            if html is not None
            else self._get_soup()
        )

    def __str__(self) -> str:
        blocks = []
        for index, matchup in enumerate(self.data):
            lines = [f"Matchup {index + 1}"]
            for side in ("away", "home"):
                team = matchup[side]["team"]
                abbr = matchup[side].get("abbr") or ""
                label = f"{team} ({abbr})" if abbr else team
                lines.append(f"{side} team: {label}")
                for bucket in ("confirmed", "gtd", "questionable", "out"):
                    lines.append(bucket)
                    lines.extend(matchup[side][bucket])
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _get_soup(self) -> BeautifulSoup:
        response = requests.get(
            self.url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _matchup_cards(self):
        cards = self.soup.find_all("div", class_=_is_nba_lineup_card)
        if cards:
            return cards
        return [
            card
            for card in self.soup.find_all("div", class_="lineup")
            if "lineup__" not in " ".join(card.get("class") or [])
        ]

    @staticmethod
    def _side_abbr(matchup, side: str) -> str | None:
        team_cls = "is-visit" if side == "away" else "is-home"
        team_el = matchup.select_one(f"a.lineup__team.{team_cls}")
        if team_el is None:
            return None
        abbr_el = team_el.select_one(".lineup__abbr")
        if abbr_el is None:
            return None
        text = abbr_el.get_text(strip=True)
        return text.upper() if text else None

    @staticmethod
    def _side_nickname(matchup, side: str) -> str:
        exact = (
            "lineup__mteam is-visit white"
            if side == "away"
            else "lineup__mteam is-home white"
        )
        element = matchup.find("a", {"class": exact})
        if element is None:
            selector = (
                "a.lineup__mteam.is-visit"
                if side == "away"
                else "a.lineup__mteam.is-home"
            )
            element = matchup.select_one(selector)
        if element is None:
            return ""
        text = element.get_text(strip=True)
        return text.split(None, 1)[0] if text else ""

    def _resolve_abbr(self, matchup, side: str, nickname: str) -> str | None:
        abbr = self._side_abbr(matchup, side)
        if abbr:
            return abbr
        return NBA_TEAM_ABBREVIATIONS.get(nickname)

    def _get_injured_players(self, lineup_list) -> dict[str, set]:
        questionable: set[str] = set()
        out: set[str] = set()
        if lineup_list is None:
            return {"questionable": questionable, "out": out}

        injured_items = lineup_list.find_all(
            "li", class_=lambda value: value and "has-injury-status" in value
        )
        for item in injured_items:
            player_elem = item.find("a")
            if not player_elem:
                continue
            player_name = player_elem.get("title") or player_elem.get_text(strip=True)
            status_span = item.find("span", class_="lineup__inj")
            if status_span:
                status_text = status_span.get_text(strip=True).lower()
                if status_text == "out":
                    out.add(player_name)
                elif status_text in _GTD_STATUS:
                    questionable.add(player_name)
            elif "unlikely" in (item.get("title") or "").lower():
                out.add(player_name)
        return {"questionable": questionable, "out": out}

    @staticmethod
    def _ordered_expected_starters(lineup_list) -> list[dict[str, object]]:
        if lineup_list is None:
            return []
        starters: list[dict[str, object]] = []
        for item in lineup_list.find_all("li", recursive=False):
            classes = item.get("class") or []
            if "lineup__title" in classes:
                break
            if "lineup__player" not in classes:
                continue
            title = item.get("title") or ""
            if title not in _EXPECTED_TITLES:
                continue
            link = item.find("a")
            if not link:
                continue
            name = (link.get("title") or link.get_text(strip=True) or "").strip()
            if not name:
                continue
            pos_el = item.find(class_="lineup__pos")
            position = pos_el.get_text(strip=True) if pos_el else None
            inj_el = item.find(class_="lineup__inj")
            inj_text = (inj_el.get_text(strip=True) if inj_el else "").casefold()
            gtd = (
                title in _GTD_TITLES
                or "has-injury-status" in classes
                or inj_text in _GTD_STATUS
            )
            starters.append(
                {
                    "name": name,
                    "position": position or None,
                    "gtd": gtd,
                }
            )
            if len(starters) == 5:
                break
        return starters

    @staticmethod
    def _players_by_title(lineup_list, titles: list[str] | str) -> set[str]:
        if lineup_list is None:
            return set()
        if isinstance(titles, str):
            titles = [titles]
        found: set[str] = set()
        for title in titles:
            for item in lineup_list.find_all("li", {"title": title}):
                classes = item.get("class") or []
                if item.a and "has-injury-status" not in classes:
                    name = item.a.get("title") or item.a.get_text(strip=True)
                    if name:
                        found.add(name)
        return found

    def getDict(self) -> list[dict]:
        """Parse NBA matchup cards into ``self.data`` and return it."""
        self.data = []
        for matchup in self._matchup_cards():
            away_list = matchup.find("ul", {"class": "lineup__list is-visit"})
            home_list = matchup.find("ul", {"class": "lineup__list is-home"})
            away_nick = self._side_nickname(matchup, "away")
            home_nick = self._side_nickname(matchup, "home")
            away_injured = self._get_injured_players(away_list)
            home_injured = self._get_injured_players(home_list)
            self.data.append(
                {
                    "away": {
                        "team": away_nick,
                        "abbr": self._resolve_abbr(matchup, "away", away_nick),
                        "confirmed": self._players_by_title(away_list, "Very Likely To Play"),
                        "gtd": self._players_by_title(
                            away_list, ["Toss Up To Play", "Likely To Play"]
                        ),
                        "questionable": away_injured["questionable"],
                        "out": away_injured["out"],
                    },
                    "home": {
                        "team": home_nick,
                        "abbr": self._resolve_abbr(matchup, "home", home_nick),
                        "confirmed": self._players_by_title(
                            home_list, ["Very Likely To Play", "Likely To Play"]
                        ),
                        "gtd": self._players_by_title(home_list, "Toss Up To Play"),
                        "questionable": home_injured["questionable"],
                        "out": home_injured["out"],
                    },
                }
            )
        return self.data

    def expected_starters_by_abbr(self) -> dict[str, list[dict[str, object]]]:
        """Ordered expected fives keyed by team abbreviation."""
        if not self.data:
            self.getDict()
        result: dict[str, list[dict[str, object]]] = {}
        for matchup_el, parsed in zip(self._matchup_cards(), self.data):
            for side, list_cls in (
                ("away", "lineup__list is-visit"),
                ("home", "lineup__list is-home"),
            ):
                abbr = parsed[side].get("abbr")
                if not abbr:
                    continue
                starters = self._ordered_expected_starters(
                    matchup_el.find("ul", {"class": list_cls})
                )
                if starters:
                    result[abbr] = starters
        return result

    def posted_teams(self) -> set[str]:
        """Normalized abbreviations for teams with a game on the RotoWire slate."""
        if not self.data:
            self.getDict()
        teams: set[str] = set()
        for matchup in self.data:
            for side in ("away", "home"):
                abbr = matchup[side].get("abbr") or NBA_TEAM_ABBREVIATIONS.get(
                    matchup[side]["team"]
                )
                if abbr:
                    teams.add(normalize_team_abbreviation(abbr))
        return teams

    def getQuestionablePlayers(self) -> dict[str, list[str]]:
        if not self.data:
            self.getDict()
        return _players_by_status(self.data, "questionable")

    def getOutPlayers(self) -> dict[str, list[str]]:
        if not self.data:
            self.getDict()
        return _players_by_status(self.data, "out")


def _players_by_status(matchups: list[dict], status: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for matchup in matchups:
        for side in ("away", "home"):
            abbr = matchup[side].get("abbr") or NBA_TEAM_ABBREVIATIONS.get(
                matchup[side]["team"]
            )
            names = list(matchup[side][status])
            if abbr and names:
                grouped[abbr] = names
    return grouped


def starter_name_keys(
    starters: dict[str, list[dict[str, object]]],
) -> dict[str, set[str]]:
    """Map a normalized team abbreviation to name keys in its expected five."""
    lookup: dict[str, set[str]] = {}
    for abbr, players in starters.items():
        team = normalize_team_abbreviation(abbr)
        names: set[str] = set()
        for player in players:
            names.update(player_name_keys(player.get("name")))
        if team and names:
            lookup[team] = names
    return lookup


def posted_starting_flags(
    names,
    abbreviations,
    starters: dict[str, list[dict[str, object]]],
    posted_teams: set[str],
):
    """1 when the player is in that team's RotoWire five, otherwise 0.

    A team with no posted game is missing, so the flag is null and the caller
    skips that player. A posted game with the player off the five is 0.
    """
    names = pd.Series(names)
    abbreviations = pd.Series(abbreviations, index=names.index)
    lookup = starter_name_keys(starters)
    posted = {normalize_team_abbreviation(team) for team in posted_teams}
    teams = abbreviations.map(normalize_team_abbreviation)
    flags = []
    for name, team in zip(names, teams):
        if team not in posted:
            flags.append(pd.NA)
            continue
        flags.append(int(bool(set(player_name_keys(name)) & lookup.get(team, set()))))
    return pd.Series(flags, index=names.index, dtype="Int64")


if __name__ == "__main__":
    scraper = NBADailyLineups()
    starters = scraper.expected_starters_by_abbr()
    print(f"NBA games: {len(scraper.data)}")
    for abbr, five in sorted(starters.items()):
        names = ", ".join(player["name"] for player in five)
        print(f"{abbr}: {names}")
    print("Questionable:", scraper.getQuestionablePlayers())
    print("Out:", scraper.getOutPlayers())
