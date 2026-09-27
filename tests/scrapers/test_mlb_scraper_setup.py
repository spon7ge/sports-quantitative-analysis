"""Repo root and MLB scraper defaults used by run_all_odds."""

from __future__ import annotations

import pytest

from src.scrapers.mlb.mlb_underdog import _DEFAULT_CONFIG
from src.scrapers.mlb.paths import repo_root


def test_repo_root_is_project_root_not_src() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "scrapers" / "mlb").is_dir()
    assert root.name != "src"


def test_underdog_default_url_is_v1_not_beta_v5() -> None:
    url = str(_DEFAULT_CONFIG["ud_pickem_url"])
    assert url == "https://api.underdogfantasy.com/v1/over_under_lines"
    assert "beta/v5" not in url


def test_odds_snapshot_loaders_importable() -> None:
    from src.odds.load_snapshots import (
        load_fanduel_snapshot,
        load_draftkings_snapshot,
        load_novig_props_snapshot,
        load_pinnacle_props_snapshot,
        load_prizepicks_snapshot,
        load_prophetx_props_snapshot,
        load_underdog_snapshot,
    )

    assert callable(load_novig_props_snapshot)
    assert callable(load_prophetx_props_snapshot)
    assert callable(load_underdog_snapshot)
    assert callable(load_pinnacle_props_snapshot)
    assert callable(load_prizepicks_snapshot)
    assert callable(load_fanduel_snapshot)
    assert callable(load_draftkings_snapshot)


@pytest.mark.parametrize(
    "name",
    [
        "mlb_novig",
        "mlb_underdog",
        "mlb_prophetx",
        "mlb_pinnacle",
        "mlb_prizepick",
        "mlb_fanduel",
        "mlb_draftking",
    ],
)
def test_mlb_scraper_imports_as_loose_script(name: str) -> None:
    """`python mlb_*.py` from src/scrapers/mlb must resolve paths.py."""
    import subprocess
    import sys
    from pathlib import Path

    mlb_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "mlb"
    probe = (
        "import runpy; "
        f"ns = runpy.run_path('{name}.py', run_name='not_main'); "
        "assert ns.get('_ROOT')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(mlb_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
