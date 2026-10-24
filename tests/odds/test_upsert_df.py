"""Bulk upsert helpers: conflict-key dedupe and batching."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.utils.db import _dedupe_rows_by_conflict, upsert_df


def test_dedupe_rows_by_conflict_keeps_last() -> None:
    cols = ["league", "player_name", "line_score"]
    rows = [
        ("nba", "A", 1.5),
        ("nba", "B", 2.5),
        ("nba", "A", 3.5),
    ]
    out_cols, out_rows = _dedupe_rows_by_conflict(cols, rows, ["league", "player_name"])
    assert out_cols == cols
    assert out_rows == [("nba", "A", 3.5), ("nba", "B", 2.5)]


def test_upsert_df_dedupes_and_batches(monkeypatch) -> None:
    executed: list[list] = []

    class FakeCursor:
        def execute(self, *a, **k):
            return None

        def close(self):
            return None

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    class FakeEngine:
        def raw_connection(self):
            return FakeConn()

    def fake_execute_values(cur, sql, batch, page_size=None):
        executed.append(list(batch))

    monkeypatch.setattr("src.utils.db.get_engine", lambda: FakeEngine())
    monkeypatch.setattr("src.utils.db.execute_values", fake_execute_values)
    monkeypatch.setattr(
        "src.utils.db._align_df_to_table",
        lambda df, schema, table: df,
    )
    monkeypatch.setattr("src.utils.db._table_columns", lambda *a, **k: frozenset())

    df = pd.DataFrame(
        [
            {"league": "nba", "player_name": "A", "line_score": 1.5},
            {"league": "nba", "player_name": "A", "line_score": 2.5},
            {"league": "nba", "player_name": "B", "line_score": 3.5},
            {"league": "nba", "player_name": "C", "line_score": 4.5},
        ]
    )
    upsert_df(
        "nba_fanduel",
        df,
        conflict_cols=["league", "player_name"],
        batch_size=2,
        lineage_col=None,
    )
    flat = [row for batch in executed for row in batch]
    # Last A wins; three unique conflict keys; batched by 2.
    assert len(flat) == 3
    assert len(executed) == 2
    by_player = {row[1]: row[2] for row in flat}
    assert by_player == {"A": 2.5, "B": 3.5, "C": 4.5}
