"""Game-log feature builder is lagged (no current-start leakage)."""

from __future__ import annotations

import pandas as pd
from src.mlb.pipeline.gamelog_features import build_gamelog_feature_rows
from src.mlb.pipeline.hf_tables import pregame_from_starts


def test_gamelog_k_rate_ignores_current_start(mlb_config) -> None:
    start_a = pd.Timestamp("2024-04-01 23:00:00", tz="UTC")
    start_b = pd.Timestamp("2024-04-08 23:00:00", tz="UTC")
    starts = pd.DataFrame(
        {
            "pitcher_id": [1, 1],
            "game_pk": [10, 11],
            "game_date": ["2024-04-01", "2024-04-08"],
            "season": [2024, 2024],
            "strikeouts": [0, 15],
            "batters_faced": [20, 20],
            "pitches": [80, 80],
            "outs": [15, 15],
            "role": ["starter", "starter"],
            "is_home": [1, 1],
            "opponent_team_id": [2, 2],
            "team_id": [3, 3],
            "venue_id": [1, 1],
            "pitcher_hand": ["R", "R"],
            "scheduled_start_utc": [start_a, start_b],
            "event_time_utc": [start_a, start_b],
            "ingested_at_utc": [start_a, start_b],
            "doubleheader": [0, 0],
        }
    )
    tables = {
        "pitcher_starts": starts,
        "pregame_snapshots": pregame_from_starts(starts, mlb_config),
    }
    rows = build_gamelog_feature_rows(tables, mlb_config)
    first = rows.loc[rows["game_pk"] == 10].iloc[0]
    second = rows.loc[rows["game_pk"] == 11].iloc[0]
    assert first["n_eff_k_bf_365"] == 0
    assert second["n_eff_k_bf_365"] == 20
    assert second["k_bf_shrunk_365"] < 0.2
