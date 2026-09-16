"""Leakage guards for cutoff-strict feature rows."""

from __future__ import annotations

import pandas as pd

from src.mlb.pipeline.features import select_game_version


def assert_no_leakage(
    feature_rows: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    if feature_rows is None or feature_rows.empty:
        return
    cutoffs = pd.to_datetime(feature_rows["prediction_cutoff_utc"], utc=True)
    max_event = pd.to_datetime(feature_rows["max_input_event_time_utc"], utc=True)
    max_ingest = pd.to_datetime(feature_rows["max_source_ingestion_time_utc"], utc=True)
    if (max_event >= cutoffs).any():
        raise AssertionError(
            "max_input_event_time_utc is at or after prediction_cutoff_utc"
        )
    if (max_ingest >= cutoffs).any():
        raise AssertionError(
            "max_source_ingestion_time_utc is at or after prediction_cutoff_utc"
        )

    pitches = tables.get("pitch_events")
    has_times = (
        pitches is not None
        and not pitches.empty
        and "event_time_utc" in pitches.columns
    )
    if has_times:
        pitch_times = pd.to_datetime(pitches["event_time_utc"], utc=True)
        for idx, cutoff in cutoffs.items():
            used_until = max_event.loc[idx]
            leaked = (pitch_times >= cutoff) & (pitch_times <= used_until)
            if leaked.any():
                raise AssertionError("future pitch events used in feature rows")

    game_versions = tables.get("game_versions")
    if game_versions is None or game_versions.empty:
        return
    for idx, row in feature_rows.iterrows():
        cutoff = cutoffs.loc[idx]
        selected = select_game_version(game_versions, int(row["game_pk"]), cutoff)
        if selected is None:
            continue
        valid_from = pd.to_datetime(selected["valid_from_utc"], utc=True)
        if valid_from >= cutoff:
            raise AssertionError(
                "later game_versions probable pitcher used before valid_from"
            )
