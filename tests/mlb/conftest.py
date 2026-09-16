"""Shared pytest helpers for the MLB stack."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.mlb.config import load_config
from src.mlb.fixtures import load_fixture_tables, write_fixture_tables


@pytest.fixture(scope="session")
def mlb_config():
    return load_config()


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mlb_fixtures")
    write_fixture_tables(root)
    return root


@pytest.fixture(scope="session")
def fixture_tables(fixture_dir: Path):
    return load_fixture_tables(fixture_dir)
