import numpy as np
import pandas as pd
import pytest

from models.shared.market import (
    american_to_decimal,
    choose_sides,
    decimal_to_american,
    ev_bucket_records,
    load_single_bookies,
    name_key,
    no_vig_over,
    over_given_no_push,
    score_market,
)


def test_price_round_trip():
    prices = np.array([-250, -110, 100, 150, 400])
    np.testing.assert_allclose(decimal_to_american(american_to_decimal(prices)), prices)


def test_no_vig_over_symmetric_prices_is_half():
    assert no_vig_over(-110, -110) == pytest.approx(0.5)
    assert no_vig_over(-150, 120) > 0.5


def test_name_key_strips_accents_suffixes_and_applies_aliases():
    assert name_key("Nikola Jokić") == "nikola jokic"
    assert name_key("Jaren Jackson Jr.") == "jaren jackson"
    assert name_key("Ron Holland II", {"ron holland": "ronald holland"}) == "ronald holland"


def test_over_given_no_push_drops_push_mass_on_whole_lines():
    pmf = np.array([[0.2, 0.3, 0.5]])
    assert over_given_no_push(pmf, [1.0])[0] == pytest.approx(0.5 / 0.7)
    assert over_given_no_push(pmf, [1.5])[0] == pytest.approx(0.5)


def _market_frame():
    return pd.DataFrame({
        "game_key": ["g1", "g1", "g2", "g3"],
        "pts": [20, 10, 15, 30],
        "line": [18.5, 12.5, 15.0, 25.5],
        "over_bet": [-110, -110, -110, 120],
        "under_bet": [-110, -110, -110, -140],
    })


def test_choose_sides_picks_higher_ev_side_and_settles_pushes():
    s = choose_sides(np.array([0.7, 0.2, 0.6, 0.5]), _market_frame())
    assert s["take_over"].tolist() == [True, False, True, True]
    assert s["won"].tolist() == [True, True, False, True]
    assert s["settled"].tolist() == [True, True, False, True]
    assert s.loc[2, "profit"] == 0.0
    assert s.loc[3, "profit"] == pytest.approx(1.2)


def test_score_market_returns_no_bets_below_threshold():
    m = _market_frame()
    market, bets = score_market(
        np.full(len(m), 0.5), m, np.full(len(m), 0.5), np.full(len(m), 0.5),
        edge_threshold=0.5, n_boot=50, seed=0,
    )
    assert market["n"] == 3
    assert bets is None


def test_ev_bucket_records_cover_every_line():
    m = _market_frame()
    records = ev_bucket_records(np.array([0.7, 0.2, 0.6, 0.5]), m, np.full(len(m), 0.5), n_boot=50, seed=0)
    assert sum(r["n"] for r in records) == len(m)


def test_load_single_bookies_consensus_and_book_quotes(tmp_path):
    rows = []
    for book, line, over, under in [
        ("fanduel", 20.5, -110, -110),
        ("draftkings", 20.5, -105, -115),
        ("betmgm", 21.5, -120, 100),
    ]:
        rows.append(("Nikola Jokić", "points", "over", book, line, over, "2025-01-11"))
        rows.append(("Nikola Jokić", "points", "under", book, line, under, "2025-01-11"))
    rows.append(("Nikola Jokić", "assists", "over", "fanduel", 9.5, -110, "2025-01-11"))
    odds = pd.DataFrame(rows, columns=["NAME", "CATEGORY", "SIDE", "BOOKMAKER", "LINE", "ODDS", "GAME_DATE"])
    path = tmp_path / "singleBookies.csv"
    odds.to_csv(path, index=False)
    # UTC GAME_DATE is the day after the local game date.
    logs = pd.DataFrame({
        "player_id": [203999],
        "player_name": ["Nikola Jokic"],
        "game_id": ["0022400555"],
        "game_date": [pd.Timestamp("2025-01-10")],
    })

    consensus, books = load_single_bookies(path, logs)

    assert consensus.attrs["match_rate"] == 1.0
    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["line"] == 20.5
    assert row["n_books"] == 2
    assert row["over_bet"] == pytest.approx(-105)
    assert row["under_bet"] == pytest.approx(-110)
    assert sorted(books["book"]) == ["betmgm", "draftkings", "fanduel"]
    assert books.set_index("book").loc["betmgm", "line"] == 21.5
