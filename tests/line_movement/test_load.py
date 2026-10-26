from __future__ import annotations

from pathlib import Path

import pandas as pd
from line_movement.load import BOOKS, load_all_books, snapshot_at
from line_movement.paths import repo_root


def test_books_has_seven_providers():
    assert len(BOOKS) == 7
    assert {b[0] for b in BOOKS} == {
        "fanduel",
        "draftkings",
        "pinnacle",
        "novig",
        "prophetx",
        "underdog",
        "prizepicks",
    }


def test_snapshot_at_is_pacific():
    path = Path("fanduel_nba_2026-10-24_120245.json")
    ts = snapshot_at(path)
    assert str(ts.tzinfo) == "America/Los_Angeles" or ts.tzname() in {"PDT", "PST"}
    assert ts.hour == 12 and ts.minute == 2


def test_load_all_books_smoke_2026_10_24():
    root = repo_root()
    # Skip if data not present in this checkout
    sample = root / "data/props/fanduel/nba"
    if not any(sample.glob("*2026-10-24*.json")):
        return
    frame = load_all_books("2026-10-24", root=root)
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["book"].unique()) == {b[0] for b in BOOKS}
    assert "snapshot_at" in frame.columns
    assert len(frame) > 0
