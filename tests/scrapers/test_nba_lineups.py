"""RotoWire NBA expected fives set the posted starting flag."""

import pandas as pd

from src.scrapers.nba.nba_lineups import (
    NBADailyLineups,
    posted_starting_flags,
)

CARD = """
<div class="lineup is-nba">
  <a class="lineup__mteam is-visit white">Celtics</a>
  <a class="lineup__team is-visit"><span class="lineup__abbr">BOS</span></a>
  <a class="lineup__mteam is-home white">Hornets</a>
  <a class="lineup__team is-home"><span class="lineup__abbr">CHA</span></a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jayson Tatum">Tatum</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Jaylen Brown">Brown</a></li>
    <li class="lineup__player" title="Likely To Play">
      <a title="Derrick White">White</a><span class="lineup__inj">GTD</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Jrue Holiday">Holiday</a></li>
    <li class="lineup__player" title="Toss Up To Play"><a title="Al Horford">Horford</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Payton Pritchard">Pritchard</a></li>
    <li class="lineup__title">Injuries</li>
    <li class="lineup__player has-injury-status" title="Unlikely To Play">
      <a title="Kristaps Porzingis">Porzingis</a><span class="lineup__inj">OUT</span>
    </li>
  </ul>
  <ul class="lineup__list is-home">
    <li class="lineup__player" title="Very Likely To Play"><a title="LaMelo Ball">Ball</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Brandon Miller">Miller</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Miles Bridges">Bridges</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Mark Williams">Williams</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Kelly Oubre Jr.">Oubre</a></li>
  </ul>
</div>
<div class="lineup is-wnba">
  <a class="lineup__mteam is-visit white">Aces</a>
  <a class="lineup__team is-visit"><span class="lineup__abbr">LVA</span></a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play"><a title="A'ja Wilson">Wilson</a></li>
  </ul>
</div>
"""

NICKNAME_ONLY = """
<div class="lineup">
  <a class="lineup__mteam is-visit">Trail Blazers</a>
  <a class="lineup__mteam is-home">Kings</a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play"><a title="Anfernee Simons">Simons</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Scoot Henderson">Henderson</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Jerami Grant">Grant</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Deandre Ayton">Ayton</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Shaedon Sharpe">Sharpe</a></li>
  </ul>
  <ul class="lineup__list is-home">
    <li class="lineup__player" title="Very Likely To Play"><a title="De'Aaron Fox">Fox</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Domantas Sabonis">Sabonis</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Keegan Murray">Murray</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Malik Monk">Monk</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Harrison Barnes">Barnes</a></li>
  </ul>
</div>
"""


def test_expected_five_keeps_gtd_and_drops_the_sixth_man() -> None:
    lineups = NBADailyLineups(html=CARD)
    starters = lineups.expected_starters_by_abbr()

    assert set(starters) == {"BOS", "CHA"}
    boston = starters["BOS"]
    assert [player["name"] for player in boston] == [
        "Jayson Tatum",
        "Jaylen Brown",
        "Derrick White",
        "Jrue Holiday",
        "Al Horford",
    ]
    assert boston[2]["gtd"] is True
    assert boston[0]["position"] == "SF"
    assert lineups.getOutPlayers()["BOS"] == ["Kristaps Porzingis"]


def test_nickname_maps_to_an_abbreviation_when_the_page_omits_one() -> None:
    starters = NBADailyLineups(html=NICKNAME_ONLY).expected_starters_by_abbr()
    assert set(starters) == {"POR", "SAC"}
    assert starters["POR"][0]["name"] == "Anfernee Simons"


def test_posted_starting_flag_is_one_only_inside_the_five() -> None:
    lineups = NBADailyLineups(html=CARD)
    flags = posted_starting_flags(
        ["Jayson Tatum", "Payton Pritchard", "Kelly Oubre Jr", "Kristaps Porzingis"],
        ["BOS", "BOS", "CHO", "BOS"],
        lineups.expected_starters_by_abbr(),
        lineups.posted_teams(),
    )
    assert flags.tolist() == [1, 0, 1, 0]


def test_team_without_a_posted_game_is_left_blank() -> None:
    lineups = NBADailyLineups(html=CARD)
    flags = posted_starting_flags(
        pd.Series(["Jalen Brunson", "Payton Pritchard"]),
        pd.Series(["NYK", "BOS"]),
        lineups.expected_starters_by_abbr(),
        lineups.posted_teams(),
    )
    assert pd.isna(flags.iloc[0])
    assert int(flags.iloc[1]) == 0
    assert lineups.posted_teams() == {"BOS", "CHO"}
