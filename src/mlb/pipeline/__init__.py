"""MLB data pipeline: ingest, parse, starts, features, and leakage checks."""

from __future__ import annotations

from src.mlb.pipeline.features import build_feature_rows, load_fixture_tables
from src.mlb.pipeline.ingest import (
    ingest_chadwick,
    ingest_lineups,
    ingest_schedule,
    ingest_statcast,
    snapshot_raw,
)
from src.mlb.pipeline.parse import parse_schedule, parse_statcast
from src.mlb.pipeline.quality import assert_no_leakage
from src.mlb.pipeline.starts import build_pitcher_starts

__all__ = [
    "ingest_statcast",
    "ingest_schedule",
    "ingest_lineups",
    "ingest_chadwick",
    "snapshot_raw",
    "parse_statcast",
    "parse_schedule",
    "build_pitcher_starts",
    "build_feature_rows",
    "assert_no_leakage",
    "load_fixture_tables",
]
