"""RotoWire projected fives as abbreviation → five names."""

from src.live.nba_lineup import projected_lineups

CARD = """
<div class="lineup is-nba">
  <a class="lineup__mteam is-visit white">Celtics</a>
  <a class="lineup__team is-visit"><span class="lineup__abbr">BOS</span></a>
  <a class="lineup__mteam is-home white">Hornets</a>
  <a class="lineup__team is-home"><span class="lineup__abbr">CHA</span></a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jayson Tatum">Tatum</a>
    </li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Jaylen Brown">Brown</a></li>
    <li class="lineup__player" title="Likely To Play"><a title="Derrick White">White</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Jrue Holiday">Holiday</a></li>
    <li class="lineup__player" title="Toss Up To Play"><a title="Al Horford">Horford</a></li>
    <li class="lineup__player" title="Very Likely To Play"><a title="Payton Pritchard">Pritchard</a></li>
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
  <a class="lineup__team is-visit"><span class="lineup__abbr">LVA</span></a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play"><a title="A'ja Wilson">Wilson</a></li>
  </ul>
</div>
"""


def test_projected_lineups_map_each_abbreviation_to_five_names() -> None:
    lineups = projected_lineups(html=CARD)

    assert lineups == {
        "BOS": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Derrick White",
            "Jrue Holiday",
            "Al Horford",
        ],
        "CHA": [
            "LaMelo Ball",
            "Brandon Miller",
            "Miles Bridges",
            "Mark Williams",
            "Kelly Oubre Jr.",
        ],
    }
