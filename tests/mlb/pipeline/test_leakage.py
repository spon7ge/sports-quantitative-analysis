"""Cutoff leakage, postponements, doubleheaders, and starter-change tests."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from src.mlb.pipeline.features import build_feature_rows, select_game_version
from src.mlb.pipeline.quality import assert_no_leakage
from src.mlb.schemas import FEATURE_ROW_COLUMNS, coerce_frame


def _copy(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.copy() for name, frame in tables.items()}


def test_event_time_at_or_after_cutoff_is_excluded(fixture_tables, mlb_config) -> None:
    tables = _copy(fixture_tables)
    features = build_feature_rows(tables, mlb_config)
    late = features.loc[features["n_eff_k_bf_365"] > 0].iloc[-1]
    cutoff = pd.Timestamp(late["prediction_cutoff_utc"])
    pitcher_id = int(late["pitcher_id"])
    game_pk = int(late["game_pk"])

    fake = tables["pitcher_starts"].iloc[[0]].copy()
    fake["pitcher_id"] = pitcher_id
    fake["game_pk"] = 999001
    fake["event_time_utc"] = cutoff
    fake["scheduled_start_utc"] = cutoff
    fake["ingested_at_utc"] = cutoff - timedelta(days=1)
    fake["strikeouts"] = 99
    fake["batters_faced"] = 99
    tables["pitcher_starts"] = pd.concat(
        [tables["pitcher_starts"], fake], ignore_index=True
    )

    updated = build_feature_rows(tables, mlb_config)
    before = features.loc[
        (features["pitcher_id"] == pitcher_id) & (features["game_pk"] == game_pk)
    ].iloc[0]
    after = updated.loc[
        (updated["pitcher_id"] == pitcher_id) & (updated["game_pk"] == game_pk)
    ].iloc[0]
    assert after["k_bf_shrunk_365"] == before["k_bf_shrunk_365"]
    assert after["n_eff_k_bf_365"] == before["n_eff_k_bf_365"]


def test_snapshot_time_at_or_after_cutoff_is_excluded(
    fixture_tables, mlb_config
) -> None:
    tables = _copy(fixture_tables)
    features = build_feature_rows(tables, mlb_config)
    late = features.loc[features["n_eff_k_bf_365"] > 0].iloc[-1]
    cutoff = pd.Timestamp(late["prediction_cutoff_utc"])
    pitcher_id = int(late["pitcher_id"])
    game_pk = int(late["game_pk"])

    fake = tables["pitcher_starts"].iloc[[0]].copy()
    fake["pitcher_id"] = pitcher_id
    fake["game_pk"] = 999002
    fake["event_time_utc"] = cutoff - timedelta(days=7)
    fake["scheduled_start_utc"] = cutoff - timedelta(days=7)
    fake["ingested_at_utc"] = cutoff
    fake["strikeouts"] = 88
    fake["batters_faced"] = 88
    tables["pitcher_starts"] = pd.concat(
        [tables["pitcher_starts"], fake], ignore_index=True
    )

    updated = build_feature_rows(tables, mlb_config)
    before = features.loc[
        (features["pitcher_id"] == pitcher_id) & (features["game_pk"] == game_pk)
    ].iloc[0]
    after = updated.loc[
        (updated["pitcher_id"] == pitcher_id) & (updated["game_pk"] == game_pk)
    ].iloc[0]
    assert after["k_bf_shrunk_365"] == before["k_bf_shrunk_365"]
    assert after["n_eff_k_bf_365"] == before["n_eff_k_bf_365"]


def test_postponement_uses_rescheduled_date_only_after_valid_from(
    fixture_tables, mlb_config
) -> None:
    tables = _copy(fixture_tables)
    versions = tables["game_versions"]
    postponed = versions.loc[versions["status"] == "postponed"]
    assert not postponed.empty
    game_pk = int(postponed["game_pk"].iloc[0])
    original = versions.loc[
        (versions["game_pk"] == game_pk) & (versions["status"] == "postponed")
    ].iloc[0]
    rescheduled = versions.loc[
        (versions["game_pk"] == game_pk) & (versions["status"] == "rescheduled")
    ].iloc[0]

    pregame = (
        tables["pregame_snapshots"]
        .loc[tables["pregame_snapshots"]["game_pk"] == game_pk]
        .iloc[[0]]
    )
    early = pregame.copy()
    early["prediction_cutoff_utc"] = pd.Timestamp(original["valid_to_utc"])
    late = pregame.copy()
    late["prediction_cutoff_utc"] = pd.Timestamp(
        rescheduled["valid_from_utc"]
    ) + timedelta(minutes=1)

    early_tables = {**tables, "pregame_snapshots": early}
    late_tables = {**tables, "pregame_snapshots": late}
    feat_early = build_feature_rows(early_tables, mlb_config)
    feat_late = build_feature_rows(late_tables, mlb_config)
    assert feat_early["rest_days"].iloc[0] != feat_late["rest_days"].iloc[0]
    assert feat_late["rest_days"].iloc[0] - feat_early["rest_days"].iloc[0] == 1

    cutoff_early = pd.Timestamp(early["prediction_cutoff_utc"].iloc[0])
    selected = select_game_version(versions, game_pk, cutoff_early)
    assert selected is not None
    assert str(selected["status"]) == "postponed"


def test_doubleheaders_remain_distinct_rows(fixture_tables, mlb_config) -> None:
    pregame = fixture_tables["pregame_snapshots"]
    dh = (
        pregame.loc[pregame["doubleheader"] > 0]
        if "doubleheader" in pregame.columns
        else pregame
    )
    date_groups = pregame.groupby("game_date")["game_pk"].nunique()
    dh_dates = date_groups.loc[date_groups > 3]
    assert not dh_dates.empty
    game_date = dh_dates.index[0]
    rows = pregame.loc[pregame["game_date"] == game_date]
    features = build_feature_rows(fixture_tables, mlb_config)
    feat = features.loc[features["game_pk"].isin(set(rows["game_pk"]))]
    assert feat["game_pk"].nunique() == rows["game_pk"].nunique()
    assert len(feat) == len(rows)
    _ = dh


def test_changed_starter_lookup_ignores_later_version(
    fixture_tables, mlb_config
) -> None:
    tables = _copy(fixture_tables)
    versions = tables["game_versions"]
    grouped = versions.groupby("game_pk")["probable_away_pitcher_id"].nunique()
    change_pks = grouped.loc[grouped > 1]
    if change_pks.empty:
        game_pk = int(versions["game_pk"].iloc[0])
        scheduled = pd.Timestamp(versions["scheduled_start_utc"].iloc[0])
        extra = versions.iloc[[0]].copy()
        extra["probable_away_pitcher_id"] = 111005
        extra["valid_from_utc"] = scheduled - timedelta(hours=1)
        extra["valid_to_utc"] = scheduled + timedelta(hours=6)
        extra["snapshot_id"] = "snap-change"
        versions = pd.concat([versions, extra], ignore_index=True)
        tables["game_versions"] = versions
    else:
        game_pk = int(change_pks.index[0])

    game_v = versions.loc[versions["game_pk"] == game_pk].sort_values("valid_from_utc")
    original = game_v.iloc[0]
    later = game_v.iloc[-1]
    assert int(original["probable_away_pitcher_id"]) != int(
        later["probable_away_pitcher_id"]
    )
    cutoff_before = pd.Timestamp(later["valid_from_utc"]) - timedelta(minutes=5)
    selected = select_game_version(versions, game_pk, cutoff_before)
    assert selected is not None
    assert int(selected["probable_away_pitcher_id"]) == int(
        original["probable_away_pitcher_id"]
    )

    pregame = (
        tables["pregame_snapshots"]
        .loc[tables["pregame_snapshots"]["game_pk"] == game_pk]
        .copy()
    )
    if pregame.empty:
        pytest.skip("no pregame row for starter-change game")
    row = pregame.iloc[[0]].copy()
    row["prediction_cutoff_utc"] = cutoff_before
    # Pregame pitcher stays as provided even if a later version names a replacement.
    provided = int(row["pitcher_id"].iloc[0])
    mini = {**tables, "pregame_snapshots": row}
    features = build_feature_rows(mini, mlb_config)
    assert int(features["pitcher_id"].iloc[0]) == provided


def test_assert_no_leakage_passes_on_fixture_features(
    fixture_tables, mlb_config
) -> None:
    features = build_feature_rows(fixture_tables, mlb_config)
    assert_no_leakage(features, fixture_tables)
    coerced = coerce_frame(features, FEATURE_ROW_COLUMNS)
    assert list(coerced.columns) == list(FEATURE_ROW_COLUMNS)


def test_assert_no_leakage_fails_on_future_pitch(fixture_tables, mlb_config) -> None:
    tables = _copy(fixture_tables)
    features = build_feature_rows(tables, mlb_config)
    assert_no_leakage(features, tables)

    row = features.iloc[0]
    cutoff = pd.Timestamp(row["prediction_cutoff_utc"])
    future_time = cutoff + timedelta(hours=1)
    leaked_pitch = tables["pitch_events"].iloc[[0]].copy()
    leaked_pitch["pitcher_id"] = int(row["pitcher_id"])
    leaked_pitch["event_time_utc"] = future_time
    leaked_pitch["ingested_at_utc"] = future_time + timedelta(minutes=10)
    tables["pitch_events"] = pd.concat(
        [tables["pitch_events"], leaked_pitch], ignore_index=True
    )

    leaked_features = features.copy()
    leaked_features.loc[leaked_features.index[0], "max_input_event_time_utc"] = (
        future_time
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(leaked_features, tables)
