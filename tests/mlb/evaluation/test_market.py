"""Odds math, quote import, and as-of market comparison."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from src.mlb.config import REPO_ROOT
from src.mlb.evaluation.market import compare_market
from src.mlb.markets.odds import (
    american_to_implied,
    decimal_to_implied,
    expected_profit,
    implied_to_decimal,
    no_vig,
)
from src.mlb.markets.quotes import import_quotes
from src.mlb.schemas import MARKET_QUOTE_COLUMNS, PMF_COLUMNS


def test_american_to_implied() -> None:
    assert american_to_implied(100) == 0.5
    assert american_to_implied(-100) == 0.5
    assert abs(american_to_implied(150) - 100 / 250) < 1e-12
    assert abs(american_to_implied(-200) - 200 / 300) < 1e-12
    assert abs(american_to_implied(-110) - 110 / 210) < 1e-12


def test_decimal_and_inverse() -> None:
    assert abs(decimal_to_implied(2.0) - 0.5) < 1e-12
    assert abs(decimal_to_implied(1.5) - (2.0 / 3.0)) < 1e-12
    assert abs(implied_to_decimal(0.5) - 2.0) < 1e-12
    assert abs(implied_to_decimal(decimal_to_implied(1.91)) - 1.91) < 1e-12


def test_no_vig() -> None:
    q_over, q_under = no_vig(0.52, 0.52)
    assert abs(q_over - 0.5) < 1e-12
    assert abs(q_under - 0.5) < 1e-12
    raw = american_to_implied(-110)
    q_over, q_under = no_vig(raw, raw)
    assert abs(q_over + q_under - 1.0) < 1e-12
    assert abs(q_over - 0.5) < 1e-12


def test_expected_profit_with_pushes() -> None:
    assert abs(expected_profit(0.5, 0.5, 2.0) - 0.0) < 1e-12
    assert abs(expected_profit(0.4, 0.4, 2.0, p_push=0.2) - 0.0) < 1e-12
    assert abs(expected_profit(0.5, 0.3, 2.0, p_push=0.2) - 0.2) < 1e-12
    ev_no_push = expected_profit(0.4, 0.4, 1.91)
    ev_with_push = expected_profit(0.4, 0.4, 1.91, p_push=0.2)
    assert abs(ev_no_push - ev_with_push) < 1e-12


def test_import_quotes_from_fixture_csv(mlb_config) -> None:
    path = REPO_ROOT / "tests" / "mlb" / "fixtures" / "quotes.csv"
    frame = import_quotes(path, mlb_config)
    assert list(frame.columns) == list(MARKET_QUOTE_COLUMNS)
    assert "q-late-probe" in set(frame["quote_id"].astype(str))
    assert frame["price_format"].isin(["american"]).all()


def test_late_quote_probe_not_attached(fixture_tables, mlb_config) -> None:
    quotes = fixture_tables["market_quotes"]
    probe = quotes.loc[quotes["quote_id"] == "q-late-probe"].iloc[0]
    pregame = fixture_tables["pregame_snapshots"]
    snap = pregame.loc[
        (pregame["game_pk"] == probe["game_pk"])
        & (pregame["pitcher_id"] == probe["pitcher_id"])
    ].iloc[0]
    assert pd.Timestamp(probe["fetched_at_utc"]) >= pd.Timestamp(
        snap["prediction_cutoff_utc"]
    )
    preds = pd.DataFrame(
        {
            "pitcher_id": [int(probe["pitcher_id"])],
            "game_pk": [int(probe["game_pk"])],
            "prediction_cutoff_utc": [snap["prediction_cutoff_utc"]],
            "p_over_5_5": [0.45],
            "p_under_5_5": [0.55],
            **{name: [0.0] for name in PMF_COLUMNS},
        }
    )
    preds["pmf_05"] = 1.0
    compared = compare_market(preds, quotes, mlb_config)
    if "quote_id" in compared.columns:
        assert "q-late-probe" not in set(compared["quote_id"].astype(str))
    else:
        assert compared.empty


def test_late_quote_does_not_replace_timely(fixture_tables, mlb_config) -> None:
    quotes = fixture_tables["market_quotes"]
    probe = quotes.loc[quotes["quote_id"] == "q-late-probe"].iloc[0]
    pregame = fixture_tables["pregame_snapshots"]
    snap = pregame.loc[
        (pregame["game_pk"] == probe["game_pk"])
        & (pregame["pitcher_id"] == probe["pitcher_id"])
    ].iloc[0]
    timely = probe.copy()
    timely["quote_id"] = "q-timely"
    timely["fetched_at_utc"] = pd.Timestamp(
        snap["prediction_cutoff_utc"]
    ) - pd.Timedelta(minutes=20)
    timely["over_price"] = -115.0
    timely["under_price"] = -105.0
    timely["quote_status"] = "open"
    quotes2 = pd.concat([quotes, pd.DataFrame([timely])], ignore_index=True)
    preds = pd.DataFrame(
        {
            "pitcher_id": [int(probe["pitcher_id"])],
            "game_pk": [int(probe["game_pk"])],
            "prediction_cutoff_utc": [snap["prediction_cutoff_utc"]],
            "p_over_5_5": [0.45],
            "p_under_5_5": [0.55],
            "strikeouts": [6],
            **{name: [0.0] for name in PMF_COLUMNS},
        }
    )
    preds["pmf_06"] = 1.0
    compared = compare_market(preds, quotes2, mlb_config)
    assert len(compared) == 1
    assert compared.iloc[0]["quote_id"] == "q-timely"
    assert compared.iloc[0]["roi_type"] == "quoted_price_simulation"
    assert "realized" not in compared.columns.str.lower().to_list()
    raw_over = american_to_implied(-115.0)
    raw_under = american_to_implied(-105.0)
    q_over, q_under = no_vig(raw_over, raw_under)
    assert abs(compared.iloc[0]["market_p_over"] - q_over) < 1e-9
    assert abs(compared.iloc[0]["market_p_under"] - q_under) < 1e-9


def test_worst_price_during_latency(mlb_config) -> None:
    cfg = replace(mlb_config, quote_latency_seconds=60.0)
    cutoff = pd.Timestamp("2019-04-08 21:10:00", tz="UTC")
    preds = pd.DataFrame(
        {
            "pitcher_id": [111001],
            "game_pk": [500043],
            "prediction_cutoff_utc": [cutoff],
            "p_over_5_5": [0.5],
            "p_under_5_5": [0.5],
            **{name: [0.0] for name in PMF_COLUMNS},
        }
    )
    preds["pmf_05"] = 1.0
    quotes = pd.DataFrame(
        {
            "quote_id": ["q-a", "q-b"],
            "game_pk": [500043, 500043],
            "pitcher_id": [111001, 111001],
            "sportsbook": ["Book", "Book"],
            "jurisdiction": ["US", "US"],
            "line": [5.5, 5.5],
            "over_price": [-110.0, -150.0],
            "under_price": [-110.0, 130.0],
            "price_format": ["american", "american"],
            "fetched_at_utc": [
                cutoff - pd.Timedelta(seconds=40),
                cutoff - pd.Timedelta(seconds=10),
            ],
            "quote_status": ["open", "open"],
            "rule_version": ["k_standard_v1", "k_standard_v1"],
        }
    )
    compared = compare_market(preds, quotes, cfg)
    assert len(compared) == 1
    assert compared.iloc[0]["quote_id"] == "q-b"
    assert american_to_implied(-150.0) > american_to_implied(-110.0)
