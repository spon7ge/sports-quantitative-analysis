"""RotoWire lineup props snapshot. Fixture HTML only; no network."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.scrapers.nba.nba_lineups import NBA_LINEUPS_URL, NBADailyLineups
from src.scrapers.nba.nba_rotowire_lineup import (
    fetched_at_text,
    output_directory,
    run,
    snapshot_filename,
    write_snapshot,
)

PACIFIC = ZoneInfo("America/Los_Angeles")
MOMENT = datetime(2026, 10, 24, 22, 35, 12, tzinfo=PACIFIC)

FETCHED_AT = "2026-10-24T22:35:12-07:00"

PAGE = """
<div class="lineup is-nba">
  <a class="lineup__mteam is-visit white">Celtics</a>
  <a class="lineup__team is-visit"><span class="lineup__abbr">BOS</span></a>
  <a class="lineup__mteam is-home white">76ers</a>
  <a class="lineup__team is-home"><span class="lineup__abbr">PHI</span></a>
  <ul class="lineup__list is-visit">
    <li class="lineup__status is-confirmed">Confirmed Lineup</li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jayson Tatum">Tatum</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jaylen Brown">Brown</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Derrick White">White</a><span class="lineup__pos">PG</span><span class="lineup__inj">GTD</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jrue Holiday">Holiday</a>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Al Horford">Horford</a><span class="lineup__pos">C</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Payton Pritchard">Pritchard</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__player has-injury-status" title="Very Unlikely To Play">
      <a title="Kristaps Porzingis">Porzingis</a><span class="lineup__pos">C</span><span class="lineup__inj">Out</span>
    </li>
    <li class="lineup__player" title="Toss Up To Play">
      <a title="Neemias Queta">Queta</a><span class="lineup__pos">C</span>
    </li>
  </ul>
  <ul class="lineup__list is-home">
    <li class="lineup__status is-expected">Expected Lineup</li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Tyrese Maxey">Maxey</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="VJ Edgecombe">Edgecombe</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jaylen Brown">Brown</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Dean Wade">Wade</a><span class="lineup__pos">PF</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Joel Embiid">Embiid</a><span class="lineup__pos">C</span><span class="lineup__inj">Doubt</span>
    </li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Joel Embiid">Embiid</a><span class="lineup__pos">C</span><span class="lineup__inj">Doubt</span>
    </li>
    <li class="lineup__player has-injury-status hide is-ofs" title="Very Likely To Play">
      <a title="Ariel Hukporti">Hukporti</a><span class="lineup__pos">C</span><span class="lineup__inj">OFS</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Likely To Play">
      <a title="Jabari Walker">Walker</a><span class="lineup__pos">F</span><span class="lineup__inj">Ques</span>
    </li>
    <li class="lineup__player has-injury-status" title="Very Unlikely To Play">
      <a title="Dominick Barlow">Barlow</a><span class="lineup__pos">C</span><span class="lineup__inj">Out</span>
    </li>
  </ul>
</div>
<div class="lineup is-nba">
  <a class="lineup__mteam is-visit">Trail Blazers</a>
  <a class="lineup__mteam is-home">Kings</a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Anfernee Simons">Simons</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Scoot Henderson">Henderson</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Jerami Grant">Grant</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Deandre Ayton">Ayton</a><span class="lineup__pos">C</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Shaedon Sharpe">Sharpe</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__player"><span class="lineup__pos">G</span></li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__no">None</li>
  </ul>
  <ul class="lineup__list is-home">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="De'Aaron Fox">Fox</a><span class="lineup__pos">PG</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Domantas Sabonis">Sabonis</a><span class="lineup__pos">C</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Keegan Murray">Murray</a><span class="lineup__pos">PF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="DeMar DeRozan">DeRozan</a><span class="lineup__pos">SF</span>
    </li>
    <li class="lineup__player" title="Very Likely To Play">
      <a title="Malik Monk">Monk</a><span class="lineup__pos">SG</span>
    </li>
    <li class="lineup__title is-middle">MAY NOT PLAY</li>
    <li class="lineup__player has-injury-status" title="Very Unlikely To Play">
      <a title="Kevin Huerter">Huerter</a><span class="lineup__pos">G</span><span class="lineup__inj">Out</span>
    </li>
  </ul>
</div>
<div class="lineup is-wnba">
  <a class="lineup__mteam is-visit white">Aces</a>
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="A'ja Wilson">Wilson</a><span class="lineup__pos">F</span>
    </li>
  </ul>
</div>
"""

WNBA_ONLY = """
<div class="lineup is-wnba">
  <ul class="lineup__list is-visit">
    <li class="lineup__player" title="Very Likely To Play">
      <a title="A'ja Wilson">Wilson</a>
    </li>
  </ul>
</div>
"""


def test_snapshot_reads_confirmed_projected_and_may_not_play() -> None:
    payload = NBADailyLineups(html=PAGE).lineup_snapshot(FETCHED_AT)

    assert payload["source"] == "rotowire"
    assert payload["url"] == NBA_LINEUPS_URL
    assert payload["fetched_at"] == FETCHED_AT
    assert len(payload["games"]) == 2

    away = payload["games"][0]["away"]
    assert away["team"] == "BOS"
    assert away["status"] == "confirmed"
    assert away["lineup"] == [
        {"name": "Jayson Tatum", "position": "SF"},
        {"name": "Jaylen Brown", "position": "SG"},
        {"name": "Derrick White", "position": "PG"},
        {"name": "Jrue Holiday", "position": None},
        {"name": "Al Horford", "position": "C"},
        {"name": "Payton Pritchard", "position": "PG"},
    ]
    assert away["may_not_play"] == [
        {"name": "Kristaps Porzingis", "position": "C", "badge": "Out"},
        {"name": "Neemias Queta", "position": "C", "badge": None},
    ]
    assert all(set(player) == {"name", "position"} for player in away["lineup"])

    home = payload["games"][0]["home"]
    assert home["team"] == "PHI"
    assert home["status"] == "projected"
    assert [player["name"] for player in home["lineup"]] == [
        "Tyrese Maxey",
        "VJ Edgecombe",
        "Jaylen Brown",
        "Dean Wade",
        "Joel Embiid",
    ]
    assert "badge" not in home["lineup"][-1]
    assert home["may_not_play"] == [
        {"name": "Joel Embiid", "position": "C", "badge": "Doubt"},
        {"name": "Ariel Hukporti", "position": "C", "badge": "OFS"},
        {"name": "Jabari Walker", "position": "F", "badge": "Ques"},
        {"name": "Dominick Barlow", "position": "C", "badge": "Out"},
    ]


def test_missing_status_chip_is_projected_and_nickname_maps_to_abbr() -> None:
    game = NBADailyLineups(html=PAGE).lineup_snapshot(FETCHED_AT)["games"][1]
    assert game["away"]["team"] == "POR"
    assert game["away"]["status"] == "projected"
    assert [player["name"] for player in game["away"]["lineup"]] == [
        "Anfernee Simons",
        "Scoot Henderson",
        "Jerami Grant",
        "Deandre Ayton",
        "Shaedon Sharpe",
    ]
    assert game["away"]["may_not_play"] == []
    assert game["home"]["team"] == "SAC"
    assert game["home"]["status"] == "projected"
    assert game["home"]["may_not_play"] == [
        {"name": "Kevin Huerter", "position": "G", "badge": "Out"}
    ]


def test_confirmed_text_without_the_class_is_confirmed() -> None:
    html = """
    <div class="lineup is-nba">
      <a class="lineup__team is-visit"><span class="lineup__abbr">MIA</span></a>
      <ul class="lineup__list is-visit">
        <li class="lineup__status">Confirmed Lineup</li>
        <li class="lineup__player"><a title="Bam Adebayo">Bam</a><span class="lineup__pos">C</span></li>
      </ul>
      <ul class="lineup__list is-home"></ul>
    </div>
    """
    game = NBADailyLineups(html=html).lineup_snapshot(FETCHED_AT)["games"][0]
    assert game["away"]["status"] == "confirmed"
    assert game["away"]["lineup"] == [{"name": "Bam Adebayo", "position": "C"}]
    assert game["away"]["may_not_play"] == []
    assert game["home"]["team"] is None
    assert game["home"]["status"] == "projected"
    assert game["home"]["lineup"] == []
    assert game["home"]["may_not_play"] == []


def test_wnba_cards_and_empty_pages_produce_no_games() -> None:
    names = []
    for game in NBADailyLineups(html=PAGE).lineup_snapshot(FETCHED_AT)["games"]:
        for side in ("away", "home"):
            names.extend(player["name"] for player in game[side]["lineup"])
            names.extend(player["name"] for player in game[side]["may_not_play"])
    assert "A'ja Wilson" not in names

    empty = NBADailyLineups(html="<div></div>").lineup_snapshot(FETCHED_AT)
    assert empty == {
        "source": "rotowire",
        "url": NBA_LINEUPS_URL,
        "fetched_at": FETCHED_AT,
        "games": [],
    }
    assert NBADailyLineups(html=WNBA_ONLY).lineup_snapshot(FETCHED_AT)["games"] == []


def test_filename_and_fetched_at_use_pacific_time() -> None:
    assert snapshot_filename(MOMENT) == "rotowire_nba_2026-10-24_223512.json"
    assert fetched_at_text(MOMENT) == "2026-10-24T22:35:12-07:00"
    utc = datetime(2026, 10, 25, 5, 35, 12, tzinfo=ZoneInfo("UTC"))
    assert snapshot_filename(utc) == "rotowire_nba_2026-10-24_223512.json"
    assert fetched_at_text(utc) == "2026-10-24T22:35:12-07:00"


def test_output_directory_is_the_props_folder() -> None:
    assert output_directory().parts[-3:] == ("data", "props", "rotowire_nba_lineup")


def test_write_snapshot_renames_temp_into_place(tmp_path) -> None:
    older = tmp_path / "rotowire_nba_2026-10-24_220000.json"
    older.write_text("{}\n", encoding="utf-8")
    payload = NBADailyLineups(html="<div></div>").lineup_snapshot(fetched_at_text(MOMENT))
    path = write_snapshot(payload, tmp_path, MOMENT)
    assert path == tmp_path / "rotowire_nba_2026-10-24_223512.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert older.read_text(encoding="utf-8") == "{}\n"
    assert set(tmp_path.iterdir()) == {older, path}


def test_write_snapshot_removes_temp_when_rename_fails(tmp_path, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "src.scrapers.nba.nba_rotowire_lineup.os.replace",
        boom,
    )
    payload = {"source": "rotowire", "url": NBA_LINEUPS_URL, "fetched_at": FETCHED_AT, "games": []}
    with pytest.raises(OSError, match="disk full"):
        write_snapshot(payload, tmp_path, MOMENT)
    assert list(tmp_path.iterdir()) == []


def test_run_fetches_once_and_writes_the_payload(tmp_path) -> None:
    calls = {"n": 0}

    class FakeLineups:
        def lineup_snapshot(self, fetched_at: str) -> dict:
            return {
                "source": "rotowire",
                "url": NBA_LINEUPS_URL,
                "fetched_at": fetched_at,
                "games": [],
            }

    def fetch():
        calls["n"] += 1
        return FakeLineups()

    assert run(MOMENT, tmp_path, fetch) == 0
    assert calls["n"] == 1
    written = json.loads(
        (tmp_path / "rotowire_nba_2026-10-24_223512.json").read_text(encoding="utf-8")
    )
    assert written["fetched_at"] == "2026-10-24T22:35:12-07:00"
    assert written["games"] == []
    assert not (tmp_path / ".rotowire_nba_2026-10-24_223512.json.tmp").exists()


def test_run_writes_nothing_when_the_fetch_fails(tmp_path, capsys) -> None:
    def fetch():
        raise RuntimeError("rotowire 503")

    assert run(MOMENT, tmp_path, fetch) == 1
    assert list(tmp_path.iterdir()) == []
    err = capsys.readouterr().err
    assert "rotowire 503" in err
    assert "Traceback" in err
