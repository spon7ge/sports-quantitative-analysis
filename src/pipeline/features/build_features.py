"""Constructor used by ``ContextFeatureEngineer``.

The notebook only calls ``enrich()`` on an already built training frame.
The full HoopVista ``FeatureEngineer`` also reads Supabase through
``src.pipeline.clean``, which this repo already uses for the silver
pipeline, so that loader is not vendored here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

LeagueKey = Literal["nba", "wnba"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SEASON = {"nba": "2025-26", "wnba": "2025"}


class FeatureEngineer:
    """Holds league and output paths for the context-feature layer."""

    def __init__(
        self,
        raw_data_dir: str | None = None,
        processed_data_dir: str | None = None,
        season: str | None = None,
        season_type: str = "Regular Season",
        league: LeagueKey = "nba",
    ):
        if league not in _DEFAULT_SEASON:
            raise ValueError(f"Unknown league: {league!r}")
        self.raw_data_dir = raw_data_dir or str(_REPO_ROOT / "data" / "raw")
        self.processed_data_dir = processed_data_dir or str(
            _REPO_ROOT / "data" / "processed"
        )
        self.league: LeagueKey = league
        self.season = season or _DEFAULT_SEASON[league]
        self.season_type = season_type
        os.makedirs(self.processed_data_dir, exist_ok=True)
