"""Feature-row contract tests."""

from __future__ import annotations

from src.mlb.pipeline.features import build_feature_rows, load_fixture_tables
from src.mlb.pipeline.starts import build_pitcher_starts
from src.mlb.schemas import FEATURE_ROW_COLUMNS, PITCHER_START_COLUMNS


def test_load_fixture_tables_wraps_shared_loader(fixture_dir) -> None:
    tables = load_fixture_tables(fixture_dir)
    assert "pregame_snapshots" in tables
    assert "pitch_events" in tables
    assert not tables["pregame_snapshots"].empty


def test_build_feature_rows_one_per_pregame(fixture_tables, mlb_config) -> None:
    features = build_feature_rows(fixture_tables, mlb_config)
    pregame = fixture_tables["pregame_snapshots"]
    assert len(features) == len(pregame)
    assert list(features.columns) == list(FEATURE_ROW_COLUMNS)
    assert features["expected_bf_oof"].isna().all()
    assert features["p_early_exit_oof"].isna().all()
    assert set(features["feature_set_version"].unique()) == {
        mlb_config.feature_set_version
    }


def test_build_pitcher_starts_has_schema(fixture_tables) -> None:
    starts = build_pitcher_starts(
        fixture_tables["plate_appearances"],
        fixture_tables["pitch_events"],
        fixture_tables["game_versions"],
    )
    for column in PITCHER_START_COLUMNS:
        assert column in starts.columns
    assert starts["game_pk"].nunique() >= 1
    assert (starts["batters_faced"] > 0).any()


def test_missing_flags_are_binary(fixture_tables, mlb_config) -> None:
    features = build_feature_rows(fixture_tables, mlb_config)
    flags = [
        "missing_pitcher_k",
        "missing_plate_discipline",
        "missing_workload",
        "missing_opponent",
        "missing_lineup",
        "missing_stuff",
        "missing_role",
    ]
    for name in flags:
        assert set(features[name].unique()).issubset({0, 1})
    early = features.loc[features["season"] == 2017]
    assert (early["missing_pitcher_k"] == 1).any()
