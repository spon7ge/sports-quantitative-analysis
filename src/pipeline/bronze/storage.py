"""Parquet persistence for datasets and tracking checkpoints."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import LeagueConfig

class ParquetStore:
    def __init__(
        self,
        output_dir: str | Path,
        config: LeagueConfig,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config

    def path_for(self, dataset: str) -> Path:
        filename = self.config.parquet_name_by_dataset[dataset]
        return self.output_dir / filename

    def write(
        self,
        dataset: str,
        frame: pd.DataFrame,
    ) -> Path:
        path = self.path_for(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    @staticmethod
    def write_checkpoint(
        path: str | Path,
        frame: pd.DataFrame,
    ) -> Path:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(checkpoint, index=False)
        return checkpoint

    @staticmethod
    def read_checkpoint(
        path: str | Path,
    ) -> pd.DataFrame | None:
        checkpoint = Path(path)

        if not checkpoint.exists():
            return None

        return pd.read_parquet(checkpoint)