from __future__ import annotations

from pathlib import Path


def package_src() -> Path:
    return Path(__file__).resolve().parents[1]


def analysis_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not find repo root (pyproject.toml)")
