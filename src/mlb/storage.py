"""Partitioned Parquet plus DuckDB catalog for MLB tables."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.schemas import TABLE_SCHEMAS, coerce_frame, validate_frame


class MlbStore:
    def __init__(self, config: MlbConfig) -> None:
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.table_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)

    def table_path(self, name: str) -> Path:
        return self.config.table_dir / name

    def write_table(self, name: str, frame: pd.DataFrame) -> Path:
        if name not in TABLE_SCHEMAS:
            raise KeyError(f"Unknown table {name}")
        schema = TABLE_SCHEMAS[name]
        validate_frame(frame, schema, name=name)
        coerced = coerce_frame(frame, schema)
        path = self.table_path(name)
        path.mkdir(parents=True, exist_ok=True)
        single = path / "part-0.parquet"
        temporary = path / ".part-0.parquet.tmp"
        coerced.to_parquet(temporary, index=False)
        temporary.replace(single)
        for leftover in path.rglob("*.parquet"):
            if leftover.resolve() != single.resolve():
                leftover.unlink()
        return path

    def read_table(self, name: str) -> pd.DataFrame:
        path = self.table_path(name)
        single = path / "part-0.parquet"
        if single.exists():
            frame = pd.read_parquet(single)
        elif path.exists() and any(path.rglob("*.parquet")):
            frame = pd.read_parquet(path)
        else:
            from src.mlb.schemas import empty_frame

            return empty_frame(TABLE_SCHEMAS[name])
        return coerce_frame(frame, TABLE_SCHEMAS[name])

    def query(self, sql: str) -> pd.DataFrame:
        connection = duckdb.connect(str(self.config.data_dir / "catalog.duckdb"))
        try:
            for name in TABLE_SCHEMAS:
                table_path = self.table_path(name)
                if table_path.exists():
                    connection.execute(
                        "CREATE OR REPLACE VIEW "
                        f"{name} AS SELECT * FROM read_parquet(?)",
                        [str(table_path / "**" / "*.parquet")],
                    )
            return connection.execute(sql).df()
        finally:
            connection.close()

    def write_json(self, path: Path, payload: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
