"""YAML-backed configuration for the MLB strikeout MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.mlb import (
    FEATURE_SET_VERSION,
    MODEL_VERSION,
    PARSER_VERSION,
    WORKLOAD_MODEL_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "mlb.yaml"


def _as_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass(frozen=True)
class FoldWindow:
    name: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class MlbConfig:
    data_dir: Path
    artifact_dir: Path
    seed: int = 42
    parser_version: str = PARSER_VERSION
    feature_set_version: str = FEATURE_SET_VERSION
    model_version: str = MODEL_VERSION
    workload_model_version: str = WORKLOAD_MODEL_VERSION
    k_max: int = 15
    tail_mass_threshold: float = 0.001
    prediction_interval: float = 0.80
    lines: tuple[float, ...] = (4.5, 5.5, 6.5)
    early_exit_bf: int = 15
    pitcher_k_prior_strength: float = 175.0
    batter_k_prior_strength: float = 225.0
    rate_limit_seconds: float = 1.0
    user_agent: str = "nba-quant-mlb-research/1.0"
    posterior_draws: int = 64
    workload_min_train_starts: int = 24
    workload_l2: float = 1.0
    strikeout_l2: float = 2.0
    forecast_horizon_hours: float = 2.0
    quote_latency_seconds: float = 0.0
    folds: tuple[FoldWindow, ...] = field(
        default_factory=lambda: (
            FoldWindow("dev_2018", "2017-12-31", "2018-01-01", "2018-12-31"),
            FoldWindow("dev_2019", "2018-12-31", "2019-01-01", "2019-12-31"),
        )
    )
    frozen_eval_start: str = "2023-01-01"
    frozen_eval_end: str = "2025-12-31"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw_snapshots"

    @property
    def table_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def fixture_dir(self) -> Path:
        return REPO_ROOT / "tests" / "mlb" / "fixtures"


def load_config(path: str | Path | None = None) -> MlbConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {config_path} must be a mapping")
        raw = loaded
    repo_base = REPO_ROOT
    folds_raw = raw.get("folds") or [
        {
            "name": "dev_2018",
            "train_end": "2017-12-31",
            "test_start": "2018-01-01",
            "test_end": "2018-12-31",
        },
        {
            "name": "dev_2019",
            "train_end": "2018-12-31",
            "test_start": "2019-01-01",
            "test_end": "2019-12-31",
        },
    ]
    folds = tuple(
        FoldWindow(
            name=str(item["name"]),
            train_end=str(item["train_end"]),
            test_start=str(item["test_start"]),
            test_end=str(item["test_end"]),
        )
        for item in folds_raw
    )
    return MlbConfig(
        data_dir=_as_path(raw.get("data_dir", "data/mlb"), base=repo_base),
        artifact_dir=_as_path(
            raw.get("artifact_dir", "artifacts/mlb"),
            base=repo_base,
        ),
        seed=int(raw.get("seed", 42)),
        parser_version=str(raw.get("parser_version", PARSER_VERSION)),
        feature_set_version=str(
            raw.get("feature_set_version", FEATURE_SET_VERSION)
        ),
        model_version=str(raw.get("model_version", MODEL_VERSION)),
        workload_model_version=str(
            raw.get("workload_model_version", WORKLOAD_MODEL_VERSION)
        ),
        k_max=int(raw.get("k_max", 15)),
        tail_mass_threshold=float(raw.get("tail_mass_threshold", 0.001)),
        prediction_interval=float(raw.get("prediction_interval", 0.80)),
        lines=tuple(float(x) for x in raw.get("lines", (4.5, 5.5, 6.5))),
        early_exit_bf=int(raw.get("early_exit_bf", 15)),
        pitcher_k_prior_strength=float(
            raw.get("pitcher_k_prior_strength", 175.0)
        ),
        batter_k_prior_strength=float(
            raw.get("batter_k_prior_strength", 225.0)
        ),
        rate_limit_seconds=float(raw.get("rate_limit_seconds", 1.0)),
        user_agent=str(raw.get("user_agent", "nba-quant-mlb-research/1.0")),
        posterior_draws=int(raw.get("posterior_draws", 64)),
        workload_min_train_starts=int(
            raw.get("workload_min_train_starts", 24)
        ),
        workload_l2=float(raw.get("workload_l2", 1.0)),
        strikeout_l2=float(raw.get("strikeout_l2", 2.0)),
        forecast_horizon_hours=float(
            raw.get("forecast_horizon_hours", 2.0)
        ),
        quote_latency_seconds=float(raw.get("quote_latency_seconds", 0.0)),
        folds=folds,
        frozen_eval_start=str(raw.get("frozen_eval_start", "2023-01-01")),
        frozen_eval_end=str(raw.get("frozen_eval_end", "2025-12-31")),
    )
