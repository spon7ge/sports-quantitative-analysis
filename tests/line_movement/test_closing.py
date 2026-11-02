from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from line_movement.closing import (
    MAX_GAP,
    ListedLine,
    Snapshot,
    closing_boards,
    standard_closing_points,
    write_closing_files,
)


def _line(player: str, stat: str, line: float, odds: str = "standard") -> ListedLine:
    return ListedLine(
        player=player,
        stat_type=stat,
        line_score=line,
        odds_type=odds,
        updated_at="2026-10-26T22:00:00Z",
        league="NBA",
    )


def _snap(stamp: str, lines: list[ListedLine]) -> Snapshot:
    return Snapshot(fetched_at=datetime.fromisoformat(stamp), lines=tuple(lines))


def test_closing_line_is_the_last_listing_before_the_player_drops() -> None:
    early = _snap(
        "2026-10-26T22:45:00+00:00",
        [_line("Jalen Brunson", "Points", 27.5)],
    )
    close = _snap(
        "2026-10-26T23:00:00+00:00",
        [
            _line("Jalen Brunson", "Rebounds", 3.5),
            _line("Jalen Brunson", "Points", 28.5),
            _line("Jayson Tatum", "Points", 30.5),
        ],
    )
    after = _snap(
        "2026-10-26T23:15:00+00:00",
        [_line("Jayson Tatum", "Points", 30.5)],
    )
    boards = closing_boards([after, early, close])
    players = {row.player: row for row in boards.by_date[date(2026, 10, 26)]}

    assert set(players) == {"Jalen Brunson"}
    brunson = players["Jalen Brunson"]
    assert brunson.closing_fetched_at == datetime(2026, 10, 26, 23, 0, tzinfo=UTC)
    assert brunson.next_fetched_at == datetime(2026, 10, 26, 23, 15, tzinfo=UTC)
    assert [(line.stat_type, line.line_score) for line in brunson.lines] == [
        ("Points", 28.5),
        ("Rebounds", 3.5),
    ]


def test_player_still_listed_on_the_last_snapshot_is_left_out() -> None:
    only = _snap("2026-10-26T20:00:00+00:00", [_line("A", "Points", 10)])
    boards = closing_boards([only])
    assert boards.by_date[date(2026, 10, 26)] == ()
    assert boards.left_out[date(2026, 10, 26)]["no_later_snapshot"] == 1


def test_drop_across_a_gap_longer_than_one_fetch_is_left_out() -> None:
    listed = _snap("2026-10-26T20:00:00+00:00", [_line("A", "Points", 10)])
    gone = _snap("2026-10-26T20:30:00+00:00", [])
    boards = closing_boards([listed, gone])
    assert boards.by_date[date(2026, 10, 26)] == ()
    assert boards.left_out[date(2026, 10, 26)]["next_fetch_too_late"] == 1


def test_same_day_relisting_keeps_the_later_line() -> None:
    first = _snap("2026-10-26T20:00:00+00:00", [_line("A", "Points", 10)])
    missing = _snap("2026-10-26T20:15:00+00:00", [])
    back = _snap("2026-10-26T20:30:00+00:00", [_line("A", "Points", 11)])
    gone = _snap("2026-10-26T20:45:00+00:00", [])
    boards = closing_boards([first, missing, back, gone])
    rows = boards.by_date[date(2026, 10, 26)]
    assert len(rows) == 1
    assert rows[0].closing_fetched_at == datetime(2026, 10, 26, 20, 30, tzinfo=UTC)
    assert rows[0].lines[0].line_score == 11


def test_next_day_relisting_keeps_the_prior_close() -> None:
    close = _snap("2026-10-26T23:00:00+00:00", [_line("A", "Points", 10)])
    gone = _snap("2026-10-26T23:15:00+00:00", [])
    nxt = _snap("2026-10-27T18:00:00+00:00", [_line("A", "Points", 12)])
    boards = closing_boards([close, gone, nxt])
    assert [row.lines[0].line_score for row in boards.by_date[date(2026, 10, 26)]] == [
        10
    ]
    assert boards.by_date[date(2026, 10, 27)] == ()
    assert boards.left_out[date(2026, 10, 27)]["no_later_snapshot"] == 1


def test_pacific_date_is_los_angeles() -> None:
    # 06:30Z is 23:30 the previous evening in Los Angeles during PDT.
    close = _snap("2026-10-27T06:30:00+00:00", [_line("A", "Points", 10)])
    gone = _snap("2026-10-27T06:45:00+00:00", [])
    boards = closing_boards([close, gone])
    assert date(2026, 10, 26) in boards.by_date
    assert boards.by_date[date(2026, 10, 26)][0].player == "A"
    assert date(2026, 10, 27) not in boards.by_date


def test_blank_player_names_are_counted_and_skipped() -> None:
    close = _snap(
        "2026-10-26T20:00:00+00:00",
        [_line("", "Points", 10), _line("A", "Points", 12)],
    )
    gone = _snap("2026-10-26T20:15:00+00:00", [])
    boards = closing_boards([close, gone])
    assert [row.player for row in boards.by_date[date(2026, 10, 26)]] == ["A"]
    assert boards.skipped_blank_lines == 1


def test_max_gap_default_is_one_fetch_plus_late_poll_slack() -> None:
    assert MAX_GAP == timedelta(minutes=22)


def test_naive_fetched_at_is_rejected() -> None:
    snap = Snapshot(
        fetched_at=datetime(2026, 10, 26, 20, 0),
        lines=(_line("A", "Points", 10),),
    )
    with pytest.raises(ValueError, match="timezone"):
        closing_boards([snap])


def test_write_closing_files_emits_one_json_per_date(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    _write_snapshot(
        source / "prizepicks_nba_2026-10-26_160000.json",
        "2026-10-26T23:00:00Z",
        [{"player": "Jalen Brunson", "stat_type": "Points", "line_score": 28.5,
          "odds_type": "standard", "updated_at": "2026-10-26T22:00:00Z", "league": "NBA"}],
    )
    _write_snapshot(
        source / "prizepicks_nba_2026-10-26_161500.json",
        "2026-10-26T23:15:00Z",
        [],
    )
    out = tmp_path / "out"
    written = write_closing_files(source, out)
    assert [path.name for path in written] == [
        "prizepicks_nba_closing_2026-10-26.json"
    ]
    payload = json.loads(written[0].read_text())
    assert payload["date"] == "2026-10-26"
    assert payload["player_count"] == 1
    assert payload["players"][0]["player"] == "Jalen Brunson"
    assert payload["players"][0]["lines"][0]["line_score"] == 28.5
    assert payload["left_out"]["next_fetch_too_late"] == 0


def _write_close(path: Path, day: str, players: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "source": "PrizePicks",
                "league": "NBA",
                "date": day,
                "player_count": len(players),
                "players": players,
            }
        )
    )


def test_standard_closing_points_keeps_one_players_standard_points(tmp_path: Path) -> None:
    _write_close(
        tmp_path / "prizepicks_nba_closing_2026-10-24.json",
        "2026-10-24",
        [
            {
                "player": "Julius Randle",
                "lines": [
                    {"stat_type": "Points", "line_score": 20.5, "odds_type": "standard"},
                    {"stat_type": "Points", "line_score": 25.5, "odds_type": "demon"},
                    {"stat_type": "Pts+Rebs", "line_score": 30.5, "odds_type": "standard"},
                ],
            },
            {
                "player": "Julius Randle + Scottie Barnes",
                "lines": [
                    {"stat_type": "Points", "line_score": 40.5, "odds_type": "standard"},
                ],
            },
        ],
    )
    _write_close(
        tmp_path / "prizepicks_nba_closing_2026-10-25.json",
        "2026-10-25",
        [
            {
                "player": "Julius Randle",
                "lines": [
                    {"stat_type": "Points", "line_score": 21.5, "odds_type": "standard"},
                ],
            }
        ],
    )
    assert standard_closing_points(tmp_path, "Julius Randle") == {
        "2026-10-24": 20.5,
        "2026-10-25": 21.5,
    }
    assert standard_closing_points(tmp_path, "Jayson Tatum") == {}


def test_standard_closing_points_rejects_two_standard_points_lines(tmp_path: Path) -> None:
    _write_close(
        tmp_path / "prizepicks_nba_closing_2026-10-24.json",
        "2026-10-24",
        [
            {
                "player": "Julius Randle",
                "lines": [
                    {"stat_type": "Points", "line_score": 20.5, "odds_type": "standard"},
                    {"stat_type": "Points", "line_score": 21.5, "odds_type": "standard"},
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="standard points"):
        standard_closing_points(tmp_path, "Julius Randle")


def test_underdog_closing_keeps_side_and_price(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    _write_picks(
        source / "underdog_nba_2026-10-27_161500.json",
        "2026-10-27T23:15:00Z",
        [
            {
                "full_name": "Jayson Tatum",
                "stat_name": "points",
                "stat_value": "25.5",
                "choice": "over",
                "american_price": "-120",
                "updated_at": "2026-10-27T23:00:00Z",
            },
            {
                "full_name": "Jayson Tatum",
                "stat_name": "points",
                "stat_value": "25.5",
                "choice": "under",
                "american_price": "-103",
                "updated_at": "2026-10-27T23:00:00Z",
            },
        ],
    )
    _write_picks(
        source / "underdog_nba_2026-10-27_163000.json",
        "2026-10-27T23:30:00Z",
        [],
    )
    written = write_closing_files(source, tmp_path / "out", book="underdog")
    payload = json.loads(written[0].read_text())
    assert written[0].name == "underdog_nba_closing_2026-10-27.json"
    assert payload["source"] == "Underdog"
    lines = payload["players"][0]["lines"]
    assert [(line["side"], line["line_score"], line["american_price"]) for line in lines] == [
        ("over", 25.5, -120),
        ("under", 25.5, -103),
    ]


def test_novig_closing_keeps_both_priced_sides(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    game = {
        "name": "Nets @ Celtics",
        "scheduled": "2026-10-27T23:35:00Z",
        "status": "not_started",
        "props": [
            {
                "player": "Jayson Tatum",
                "stat": "points",
                "line": 26.5,
                "is_main": True,
                "over": {"american": -125},
                "under": {"american": 102},
            }
        ],
    }
    _write_games(
        source / "novig_nba_2026-10-27_160000_props.json",
        "2026-10-27T23:00:00Z",
        [game],
    )
    _write_games(
        source / "novig_nba_2026-10-27_161500_props.json",
        "2026-10-27T23:15:00Z",
        [],
    )
    written = write_closing_files(source, tmp_path / "out", book="novig")
    payload = json.loads(written[0].read_text())
    assert payload["source"] == "Novig"
    lines = payload["players"][0]["lines"]
    assert [
        (line["stat_type"], line["side"], line["line_score"], line["american_price"])
        for line in lines
    ] == [
        ("points", "over", 26.5, -125),
        ("points", "under", 26.5, 102),
    ]


def test_pinnacle_closing_reads_flat_american_prices(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    _write_games(
        source / "pinnacle_nba_2026-10-27_160000_props.json",
        "2026-10-27T23:00:00Z",
        [
            {
                "label": "Nets vs Celtics",
                "start_time": "2026-10-27T23:40:00Z",
                "props": [
                    {
                        "player": "Jayson Tatum",
                        "stat": "points",
                        "line": 26.5,
                        "american_over": -118,
                        "american_under": -108,
                    }
                ],
            }
        ],
    )
    _write_games(
        source / "pinnacle_nba_2026-10-27_161500_props.json",
        "2026-10-27T23:15:00Z",
        [],
    )
    written = write_closing_files(source, tmp_path / "out", book="pinnacle")
    lines = json.loads(written[0].read_text())["players"][0]["lines"]
    assert [(line["side"], line["american_price"]) for line in lines] == [
        ("over", -118),
        ("under", -108),
    ]


def test_prophetx_closing_uses_the_games_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    _write_games(
        source / "prophetx_nba_2026-10-27_160000_props.json",
        "2026-10-27T23:00:00Z",
        [
            {
                "name": "Nets at Celtics",
                "scheduled": "2026-10-27T23:30:00Z",
                "props": [
                    {
                        "player": "Jayson Tatum",
                        "stat": "points",
                        "line": 26.5,
                        "over": {"american": -130, "stake": 10},
                        "under": {"american": 105, "stake": 20},
                    }
                ],
            }
        ],
    )
    _write_games(
        source / "prophetx_nba_2026-10-27_161500_props.json",
        "2026-10-27T23:15:00Z",
        [],
    )
    written = write_closing_files(source, tmp_path / "out", book="prophetx")
    assert written[0].name == "prophetx_nba_closing_2026-10-27.json"
    assert json.loads(written[0].read_text())["source"] == "ProphetX"


def _write_picks(path: Path, fetched_at: str, picks: list[dict]) -> None:
    path.write_text(
        json.dumps({"source": "Underdog", "fetched_at": fetched_at, "picks": picks})
    )


def _write_games(path: Path, fetched_at: str, games: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {"source": path.name.split("_", 1)[0], "fetched_at": fetched_at, "games": games}
        )
    )


def _write_snapshot(path: Path, fetched_at: str, projections: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "source": "PrizePicks",
                "league": "NBA",
                "fetched_at": fetched_at,
                "count": len(projections),
                "projections": projections,
            }
        )
    )
