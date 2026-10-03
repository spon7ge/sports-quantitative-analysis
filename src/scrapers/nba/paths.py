"""Resolve the nba_quant repo root from nested scraper modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Walk parents until ``pyproject.toml`` is found."""
    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"could not find repo root from {here}")


def ensure_repo_on_path(start: str | os.PathLike[str] | None = None) -> Path:
    root = repo_root(start)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
