"""Historical tennis scraper writes under data/historical/."""

from __future__ import annotations

from src.scrapers.nba.paths import repo_root
from src.scrapers.tennis_props import CHECKPOINT, OUTPUT_CSV


def test_historical_tennis_paths_live_under_data_historical() -> None:
    root = repo_root()
    historical = root / "data" / "historical"
    assert OUTPUT_CSV == historical / "tennis_historical_odds.parquet"
    assert CHECKPOINT == historical / "done_windows_tennis.json"
