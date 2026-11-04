"""The points notebooks must call the ml_insights name they actually import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    REPO / "notebooks/nba/points/00_points_xgboost.ipynb",
    REPO / "notebooks/nba/points/01_points_xgboost.ipynb",
]


def _cell_text(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_reliability_diagram_uses_imported_ml_insights_alias(path: Path):
    source = _cell_text(path)
    assert "import ml_insights as ml" in source
    assert "ml.plot_reliability_diagram(" in source
    assert "mli.plot_reliability_diagram(" not in source
