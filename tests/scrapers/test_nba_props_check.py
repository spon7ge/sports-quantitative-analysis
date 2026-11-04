"""Open/close book tally for `python nba_props.py check`. No network."""

from src.scrapers.nba_props import check, format_check, tally_books


def _quote(
    odds=None,
    line=None,
    open_odds=None,
    open_line=None,
    open_spread=None,
    close_odds=None,
    close_line=None,
    close_spread=None,
):
    return {
        "odds": odds,
        "overUnder": line,
        "openOdds": open_odds,
        "openOverUnder": open_line,
        "openSpread": open_spread,
        "closeOdds": close_odds,
        "closeOverUnder": close_line,
        "closeSpread": close_spread,
    }


def _event(odds):
    return {"odds": odds}


def test_counts_open_and_close_separately():
    events = [
        _event(
            {
                "points-LEBRON_JAMES_1_NBA-game-ou-over": {
                    "byBookmaker": {
                        "pinnacle": _quote(
                            odds="-110",
                            line="25.5",
                            open_odds="-115",
                            open_line="24.5",
                            close_odds="-108",
                            close_line="25.5",
                        ),
                        "draftkings": _quote(
                            odds="-110",
                            line="25.5",
                            open_odds="-120",
                            open_line="24.5",
                        ),
                        "bet365": _quote(odds="-110", line="25.5"),
                        "fanduel": _quote(
                            odds="-110",
                            line="25.5",
                            close_odds="-105",
                            close_line="25.5",
                        ),
                    }
                },
                "points-LEBRON_JAMES_1_NBA-game-ou-under": {
                    "byBookmaker": {
                        "pinnacle": _quote(
                            odds="-110",
                            line="25.5",
                            open_odds="-105",
                            open_line="24.5",
                            close_odds="-112",
                            close_line="25.5",
                        ),
                    }
                },
            }
        )
    ]
    tally, markets, sample = tally_books(events)
    assert tally["pinnacle"] == {
        "with_open": 2,
        "no_open": 0,
        "with_close": 2,
        "no_close": 0,
    }
    assert tally["draftkings"] == {
        "with_open": 1,
        "no_open": 0,
        "with_close": 0,
        "no_close": 1,
    }
    assert tally["bet365"] == {
        "with_open": 0,
        "no_open": 1,
        "with_close": 0,
        "no_close": 1,
    }
    assert tally["fanduel"] == {
        "with_open": 0,
        "no_open": 1,
        "with_close": 1,
        "no_close": 0,
    }
    assert markets == {"points-*-game-ou-over", "points-*-game-ou-under"}
    assert sample[0] == "pinnacle"
    assert sample[1] == "points-LEBRON_JAMES_1_NBA-game-ou-over"


def test_spread_alone_counts_as_open_or_close():
    events = [
        _event(
            {
                "points-home-game-sp-home": {
                    "byBookmaker": {
                        "pinnacle": _quote(open_spread="-4.5", close_spread="-3.5"),
                    }
                }
            }
        )
    ]
    tally, markets, sample = tally_books(events)
    assert tally["pinnacle"]["with_open"] == 1
    assert tally["pinnacle"]["with_close"] == 1
    assert markets == {"points-home-game-sp-home"}
    assert sample[0] == "pinnacle"


def test_event_without_odds_adds_nothing():
    tally, markets, sample = tally_books([{}])
    assert tally == {}
    assert markets == set()
    assert sample is None


def test_format_flags_books_outside_the_scrape_list_and_names_missing_ones():
    tally = {
        "pinnacle": {"with_open": 2, "no_open": 0, "with_close": 2, "no_close": 0},
        "bet365": {"with_open": 0, "no_open": 3, "with_close": 1, "no_close": 2},
    }
    text = format_check(
        "2026-01-20",
        tally,
        {"points-*-game-ou-over"},
        (
            "bet365",
            "points-JAYSON_TATUM_1_NBA-game-ou-over",
            {"closeOdds": "-110"},
        ),
        books=["pinnacle", "underdog"],
    )
    assert "Open/close availability on 2026-01-20:" in text
    pinnacle = "pinnacle: with_open=2  no_open=0  with_close=2  no_close=0"
    bet365 = (
        "bet365: with_open=0  no_open=3  with_close=1  no_close=2"
        "   (not in your BOOKS list)"
    )
    assert pinnacle in text
    assert bet365 in text
    assert text.index(pinnacle) < text.index(bet365)
    assert "No data that day for: underdog" in text
    assert "oddIDs returned: points-*-game-ou-over" in text
    assert (
        "Sample byBookmaker entry (bet365, points-JAYSON_TATUM_1_NBA-game-ou-over):"
        in text
    )


def test_format_says_none_when_no_markets():
    text = format_check("2026-01-20", {}, set(), None, books=["pinnacle"])
    assert "oddIDs returned: (none)" in text
    assert "No data that day for: pinnacle" in text
    assert "Sample" not in text


def test_check_requests_the_next_day_with_no_book_filter(monkeypatch, capsys):
    seen = {}

    def fake(start, end, books=None):
        seen["start"] = start
        seen["end"] = end
        seen["books"] = books
        return []

    monkeypatch.setattr("src.scrapers.nba_props.fetch_events", fake)
    check("2026-01-20")
    assert seen == {"start": "2026-01-20", "end": "2026-01-21", "books": None}
    out = capsys.readouterr().out
    assert "Open/close availability on 2026-01-20:" in out
    assert "with_open=" not in out
