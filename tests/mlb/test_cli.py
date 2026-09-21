"""CLI help and optional fixture backtest entrypoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.mlb.cli import build_parser, main
from src.mlb.config import load_config
from src.mlb.schemas import GAME_VERSION_COLUMNS, ID_MAP_COLUMNS, coerce_frame
from src.mlb.storage import MlbStore


def test_help_lists_subcommands() -> None:
    help_text = build_parser().format_help()
    for name in (
        "ingest-statcast",
        "ingest-play-by-play",
        "ingest-lineup-slots",
        "snapshot-schedule",
        "snapshot-lineups",
        "build-features",
        "train-workload",
        "train-strikeouts",
        "backtest",
        "predict",
        "import-quotes",
        "compare-market",
        "daily-report",
        "ingest-hf-props",
        "ingest-gamelogs",
    ):
        assert name in help_text


def test_main_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_snapshot_lineups_uses_live_feed_lineup_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "mlb.yaml"
    yaml_path.write_text(
        f"data_dir: {tmp_path / 'data'}\nartifact_dir: {tmp_path / 'artifacts'}\n"
    )
    calls: list[dict[str, object]] = []

    def fake_load_symbol(name: str, _modules: tuple[str, ...]):
        assert name == "ingest_lineup_slots"

        def fake_ingest(_config, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame()

        return fake_ingest

    monkeypatch.setattr("src.mlb.cli._load_symbol", fake_load_symbol)

    code = main(
        [
            "--config",
            str(yaml_path),
            "--fixture",
            "snapshot-lineups",
            "--game-pk",
            "500043",
        ]
    )

    assert code == 0
    assert calls == [
        {"game_pks": [500043], "provenance": "live_feed", "http": None}
    ]


def test_season_ingest_with_no_stored_games_is_a_noop(tmp_path: Path) -> None:
    yaml_path = tmp_path / "mlb.yaml"
    yaml_path.write_text(
        f"data_dir: {tmp_path / 'data'}\nartifact_dir: {tmp_path / 'artifacts'}\n"
    )

    code = main(
        [
            "--config",
            str(yaml_path),
            "ingest-play-by-play",
            "--start-season",
            "2025",
            "--end-season",
            "2026",
        ]
    )

    assert code == 0


def test_cli_ingest_play_by_play_passes_stored_people(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "mlb.yaml"
    yaml_path.write_text(
        f"data_dir: {tmp_path / 'data'}\nartifact_dir: {tmp_path / 'artifacts'}\n"
    )
    config = load_config(yaml_path)
    store = MlbStore(config)
    store.write_table(
        "game_versions",
        coerce_frame(
            pd.DataFrame([{
                "game_pk": 500043,
                "scheduled_start_utc": "2025-07-01T19:00:00Z",
            }]),
            GAME_VERSION_COLUMNS,
        ),
    )
    store.write_table(
        "id_map",
        coerce_frame(
            pd.DataFrame([{"mlb_id": 123, "bats": "L"}]),
            ID_MAP_COLUMNS,
        ),
    )
    calls: list[dict[str, object]] = []

    def fake_load_symbol(name: str, _modules: tuple[str, ...]):
        assert name == "ingest_play_by_play"

        def fake_ingest(_config, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame()

        return fake_ingest

    monkeypatch.setattr("src.mlb.cli._load_symbol", fake_load_symbol)

    code = main([
        "--config",
        str(yaml_path),
        "--fixture",
        "ingest-play-by-play",
        "--start-season",
        "2025",
        "--end-season",
        "2025",
    ])

    assert code == 0
    assert calls[0]["game_pks"] == [500043]
    people = calls[0]["people"]
    assert isinstance(people, pd.DataFrame)
    assert people.loc[0, "bats"] == "L"


def test_cli_snapshot_lineups_fixture_returns_zero(tmp_path: Path) -> None:
    yaml_path = tmp_path / "mlb.yaml"
    yaml_path.write_text(
        f"data_dir: {tmp_path / 'data'}\nartifact_dir: {tmp_path / 'artifacts'}\n"
    )

    code = main(
        [
            "--config",
            str(yaml_path),
            "--fixture",
            "snapshot-lineups",
            "--game-pk",
            "500043",
        ]
    )

    assert code == 0


def test_cli_backtest_fixture(tmp_path: Path) -> None:
    yaml_path = tmp_path / "mlb.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"artifact_dir: {tmp_path / 'artifacts'}",
                "seed: 42",
            ]
        )
        + "\n"
    )
    code = main(["--config", str(yaml_path), "backtest", "--fixture"])
    assert code == 0
    assert (tmp_path / "artifacts" / "backtest_scores.json").exists()


def test_module_backtest_fixture_full_stack(tmp_path: Path) -> None:
    pipeline = pytest.importorskip("src.mlb.pipeline")
    if not callable(getattr(pipeline, "build_feature_rows", None)):
        pytest.skip("src.mlb.pipeline.build_feature_rows not available")
    models = pytest.importorskip("src.mlb.models")
    if not callable(getattr(models, "fit_strikeouts", None)):
        pytest.skip("src.mlb.models.fit_strikeouts not available")
    yaml_path = tmp_path / "mlb.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"artifact_dir: {tmp_path / 'artifacts'}",
                "seed: 42",
            ]
        )
        + "\n"
    )
    code = main(["--config", str(yaml_path), "backtest", "--fixture"])
    assert code == 0
