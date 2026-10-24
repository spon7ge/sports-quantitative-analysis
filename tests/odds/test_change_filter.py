"""Odds change-filter: identity-scoped latest lookup and skip unchanged quotes."""

from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from src.odds.change_filter import (
    apply_change_filter,
    fetch_latest_quotes,
    filter_unchanged_quotes,
    identity_key_rows,
    values_equal,
)
from src.odds.quote_specs import QUOTE_SPECS, get_quote_spec


def test_values_equal_handles_nulls_and_floats() -> None:
    assert values_equal(None, None)
    assert values_equal(1.0, 1.0000000001)
    assert not values_equal(1.0, 1.1)
    assert not values_equal(None, 1)


def test_identity_key_rows_dedupes_complete_identity_including_odds_type() -> None:
    spec = get_quote_spec("nba_prizepicks")
    df = pd.DataFrame(
        [
            {
                "league": "nba",
                "player_name": "A",
                "stat_type": "Points",
                "odds_type": "standard",
                "line_score": 20.5,
            },
            {
                "league": "nba",
                "player_name": "A",
                "stat_type": "Points",
                "odds_type": "demon",
                "line_score": 22.5,
            },
            {
                "league": "nba",
                "player_name": "A",
                "stat_type": "Points",
                "odds_type": "standard",
                "line_score": 21.0,
            },
        ]
    )
    keys = identity_key_rows(df, spec.identity_cols)
    assert len(keys) == 2
    assert set(keys["odds_type"]) == {"standard", "demon"}


def test_filter_unchanged_skips_matching_compare_cols_same_et_day() -> None:
    spec = get_quote_spec("nba_fanduel")
    scraped = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)
    incoming = pd.DataFrame(
        [
            {
                "league": "nba",
                "player_name": "Jayson Tatum",
                "market_type": "player_points",
                "side": "over",
                "line_score": 27.5,
                "american_price": -110,
                "scraped_at": scraped,
            },
            {
                "league": "nba",
                "player_name": "Jaylen Brown",
                "market_type": "player_points",
                "side": "over",
                "line_score": 22.5,
                "american_price": -105,
                "scraped_at": scraped,
            },
        ]
    )
    latest = pd.DataFrame(
        [
            {
                "league": "nba",
                "player_name": "Jayson Tatum",
                "market_type": "player_points",
                "side": "over",
                "line_score": 27.5,
                "american_price": -110,
                "scraped_at": scraped,
            }
        ]
    )
    kept, skipped = filter_unchanged_quotes(incoming, latest=latest, spec=spec)
    assert skipped == 1
    assert len(kept) == 1
    assert kept.iloc[0]["player_name"] == "Jaylen Brown"


def test_fetch_latest_quotes_passes_only_incoming_identities(monkeypatch) -> None:
    """Lookup must not scan full league history — only identities in the scrape."""
    captured: dict = {}

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeEngine:
        def connect(self):
            return FakeConn()

    def fake_read_sql(sql, conn, params=None):
        captured["sql"] = str(sql)
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr("src.utils.db.get_engine", lambda: FakeEngine())
    monkeypatch.setattr(
        "src.utils.db.db_load_slot",
        __import__("contextlib").contextmanager(lambda: (yield)),
    )
    monkeypatch.setattr("src.odds.change_filter.pd.read_sql", fake_read_sql)

    spec = get_quote_spec("nba_prizepicks")
    incoming = pd.DataFrame(
        [
            {
                "league": "nba",
                "player_name": "A",
                "stat_type": "Points",
                "odds_type": "standard",
                "line_score": 20.5,
                "scraped_at": datetime(2026, 10, 23, tzinfo=timezone.utc),
            }
        ]
    )
    fetch_latest_quotes(
        "nba_prizepicks", league="nba", spec=spec, incoming=incoming
    )
    sql = captured["sql"].lower()
    assert "lateral" in sql
    assert "values" in sql
    assert "odds_type" in sql
    assert "id_0_" in str(captured["params"])
    assert captured["params"]["league"] == "nba"


def test_fetch_latest_quotes_empty_incoming_skips_db(monkeypatch) -> None:
    def boom():
        raise AssertionError("should not touch the database")

    monkeypatch.setattr("src.utils.db.get_engine", boom)
    out = fetch_latest_quotes(
        "nba_novig",
        league="nba",
        spec=get_quote_spec("nba_novig"),
        incoming=pd.DataFrame(),
    )
    assert out.empty


def test_apply_change_filter_forwards_incoming_to_fetch(monkeypatch) -> None:
    seen: dict = {}

    def fake_fetch(table, *, league, spec, incoming=None):
        seen["table"] = table
        seen["league"] = league
        seen["incoming_len"] = 0 if incoming is None else len(incoming)
        return pd.DataFrame()

    monkeypatch.setattr("src.odds.change_filter.fetch_latest_quotes", fake_fetch)
    monkeypatch.delenv("ODDS_SKIP_CHANGE_FILTER", raising=False)

    df = pd.DataFrame(
        [
            {
                "league": "nba",
                "player_name": "A",
                "market_type": "player_points",
                "side": "over",
                "line_score": 1.5,
                "american_price": -110,
                "scraped_at": datetime(2026, 10, 23, tzinfo=timezone.utc),
            }
        ]
    )
    apply_change_filter("nba_fanduel", df, league="nba")
    assert seen["table"] == "nba_fanduel"
    assert seen["incoming_len"] == 1


def test_quote_specs_cover_team_tables_for_index_migration() -> None:
    team_tables = [t for t in QUOTE_SPECS if t.endswith("_team")]
    assert "nba_novig_team" in team_tables
    assert "nba_prophetx_team" in team_tables
    assert "nba_pinnacle_team" in team_tables
