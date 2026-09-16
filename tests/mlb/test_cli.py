"""CLI help and optional fixture backtest entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.mlb.cli import build_parser, main


def test_help_lists_subcommands() -> None:
    help_text = build_parser().format_help()
    for name in (
        "ingest-statcast",
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
