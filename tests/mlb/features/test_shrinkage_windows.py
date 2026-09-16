"""Rolling windows and empirical-Bayes shrinkage tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from src.mlb.pipeline.features import _shrink_rate, build_feature_rows
from src.mlb.schemas import (
    FEATURE_ROW_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PITCHER_START_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
    PREGAME_SNAPSHOT_COLUMNS,
    coerce_frame,
    empty_frame,
)


def test_rolling_windows_60_vs_365_differ_when_old_starts_exist(
    fixture_tables, mlb_config
) -> None:
    features = build_feature_rows(fixture_tables, mlb_config)
    later = features.loc[features["season"] == 2019]
    assert not later.empty
    diverged = later.loc[later["k_bf_shrunk_60"] != later["k_bf_shrunk_365"]]
    assert not diverged.empty
    assert (later["n_eff_k_bf_365"] >= later["n_eff_k_bf_60"]).all()


def test_empirical_bayes_zero_trials_returns_prior_mean() -> None:
    assert _shrink_rate(0, 0, 0.22, 175.0) == 0.22


def test_empirical_bayes_large_trials_near_raw_rate() -> None:
    prior = 0.20
    strength = 175.0
    raw = 0.40
    trials = 2000.0
    successes = raw * trials
    shrunk = _shrink_rate(successes, trials, prior, strength)
    assert abs(shrunk - raw) < abs(shrunk - prior)
    expected = (successes + prior * strength) / (trials + strength)
    assert abs(shrunk - expected) < 1e-12


def test_empirical_bayes_zero_trials_feature_row(mlb_config) -> None:
    cutoff = datetime(2018, 4, 7, 21, 10, tzinfo=UTC)
    ingested = datetime(2017, 1, 1, tzinfo=UTC)
    other_start = {
        "pitcher_id": 111002,
        "game_pk": 1,
        "game_date": "2017-06-01",
        "season": 2017,
        "strikeouts": 20,
        "batters_faced": 100,
        "pitches": 90,
        "outs": 18,
        "role": "starter",
        "is_home": 1,
        "opponent_team_id": 134,
        "team_id": 133,
        "venue_id": 1000,
        "pitcher_hand": "R",
        "scheduled_start_utc": datetime(2017, 6, 1, 23, tzinfo=UTC),
        "event_time_utc": datetime(2017, 6, 1, 23, tzinfo=UTC),
        "ingested_at_utc": ingested,
        "doubleheader": 0,
    }
    pregame = {
        "pregame_id": "zero-k",
        "game_pk": 9,
        "pitcher_id": 111001,
        "prediction_cutoff_utc": cutoff,
        "forecast_horizon_hours": 2.0,
        "scheduled_start_utc": datetime(2018, 4, 7, 23, 10, tzinfo=UTC),
        "game_date": "2018-04-07",
        "season": 2018,
        "starter_state": "probable",
        "lineup_state": "team_fallback",
        "roster_state": "active",
        "is_opener": 0,
        "is_il_return": 0,
        "is_restricted": 0,
        "is_home": 1,
        "team_id": 133,
        "opponent_team_id": 136,
        "venue_id": 1000,
        "pitcher_hand": "L",
        "expected_rhb_share": 0.5,
        "lineup_batter_ids_json": "[]",
        "source_snapshot_ids_json": "[]",
    }
    tables = {
        "pregame_snapshots": coerce_frame(
            pd.DataFrame([pregame]), PREGAME_SNAPSHOT_COLUMNS
        ),
        "pitcher_starts": coerce_frame(
            pd.DataFrame([other_start]), PITCHER_START_COLUMNS
        ),
        "pitch_events": empty_frame(PITCH_EVENT_COLUMNS),
        "plate_appearances": empty_frame(PLATE_APPEARANCE_COLUMNS),
        "game_versions": pd.DataFrame(),
        "id_map": pd.DataFrame(),
    }
    features = build_feature_rows(tables, mlb_config)
    prior = 20 / 100
    assert abs(features["k_bf_shrunk_365"].iloc[0] - prior) < 1e-12
    assert features["n_eff_k_bf_365"].iloc[0] == 0
    _ = FEATURE_ROW_COLUMNS
