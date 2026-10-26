"""Locate the points notebook package and the repository root."""

from __future__ import annotations

import sys
from pathlib import Path


def package_src() -> Path:
    return Path(__file__).resolve().parents[1]


def points_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not find repo root (pyproject.toml)")


def ensure_repo_on_path() -> Path:
    root = repo_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
