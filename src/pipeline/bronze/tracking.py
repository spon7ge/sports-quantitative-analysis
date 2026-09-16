"""Fetch player-tracking rows from a single game endpoint."""

from __future__ import annotations

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import boxscoreplayertrackv3

from .dataframes import normalize_columns
from .retry import call_with_retry

logger = logging.getLogger(__name__)

def fetch_game_tracking(
    game_id: object,
    delay: float,
) -> pd.DataFrame | None:
    normalized_game_id = str(game_id).zfill(10)
    time.sleep(delay)

    def request() -> pd.DataFrame:
        endpoint = boxscoreplayertrackv3.BoxScorePlayerTrackV3(
            game_id=normalized_game_id,
            timeout=60,
        )
        return endpoint.get_data_frames()[0]

    frame = call_with_retry(
        request,
        label=f"game {normalized_game_id}",
    )

    if frame is None or frame.empty:
        logger.warning(
            "Skipping game %s: no tracking data returned",
            normalized_game_id,
        )
        return None

    frame = normalize_columns(frame)
    missing = {"game_id", "player_id"} - set(frame.columns)

    if missing:
        logger.warning(
            "Skipping game %s: missing required columns %s",
            normalized_game_id,
            sorted(missing),
        )
        return None

    return frame