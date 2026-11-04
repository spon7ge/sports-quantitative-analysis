import ast
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
from test_predict_points import _GAMES, _history_for_pipeline, _row

live_points = importlib.import_module("src.live.01_predict_points")

_POINTS_NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks/nba/points/01_points_xgboost.ipynb"
)
_SLATE = "2026-10-28"
_LINEUPS = {"POR": ["Scoot Henderson", "Anfernee Simons"], "MEM": ["Ja Morant"]}


@pytest.fixture(autouse=True)
def _no_schedule_lookup(monkeypatch):
    monkeypatch.setattr(live_points, "findOpp", lambda *args, **kwargs: ("MEM", 0))


def _with_makes(history):
    """Makes at fixed rates so the smoothed shooting ratios are exact."""
    result = history.copy()
    result["fg3_m"] = 1.0
    result["ftm"] = 1.0
    result["fgm"] = result["fg3_m"] + 0.5 * (result["fga"] - result["fg3_a"])
    return result


def _notebook_features() -> list[str]:
    notebook = json.loads(_POINTS_NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for node in ast.parse(source).body if cell["cell_type"] == "code" else []:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "PTS_FEATURES"
                    for target in node.targets
                )
            ):
                return ast.literal_eval(node.value)
    raise AssertionError("PTS_FEATURES not found in the points notebook")


def test_feature_list_matches_the_points_notebook():
    assert live_points.FEATURES == _notebook_features()


def test_slate_row_uses_every_played_game_without_dropping_the_last():
    live = live_points.build_points_live(
        _with_makes(_history_for_pipeline()),
        ["Scoot Henderson", "Ja Morant", "Anfernee Simons"],
        _SLATE,
        games=_GAMES,
        lineups=_LINEUPS,
    )

    assert list(live.columns) == live_points.FEATURES
    assert not live[live_points.FEATURES].isna().any().any()
    scoot = live.loc["Scoot Henderson"]
    assert scoot["pts_lag_1"] == 30
    assert scoot["pts_mean_10"] == (10 + 20 + 30) / 3
    assert scoot["pts_std_10"] == pd.Series([10, 20, 30]).std()
    assert scoot["min_mean_10"] == (20 + 24 + 18) / 3
    assert scoot["two_pa_rolling_10"] == (5 + 9 + 7) / 3
    assert scoot["fg3a_rolling_10"] == 3
    assert scoot["fta_rolling_10"] == 2
    assert scoot["two_pct_smoothed"] == pytest.approx(0.5)
    assert scoot["three_pct_smoothed"] == pytest.approx(1 / 3)
    assert scoot["ft_pct_smoothed"] == pytest.approx(0.5)
    assert scoot["pts_rank_10"] == 1
    assert scoot["min_rank_10"] == 2
    assert scoot["starting"] == 1
    assert scoot["player_team_spread"] == -8.0


def test_games_on_the_slate_date_stay_out():
    history = pd.concat(
        [
            _history_for_pipeline(),
            pd.DataFrame(
                [
                    _row(
                        "Scoot Henderson", 1, "POR", 1, 2, 4, _SLATE,
                        100, 40, True, 40, 150,
                        pace=130, cfga=99, ufga=99, tchs=99, usg=0.9,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    row = live_points.points_pipeline(
        history, "Scoot Henderson", _SLATE, games=_GAMES, lineups=_LINEUPS
    )

    assert list(row.index) == live_points.FEATURES
    assert row["pts_mean_10"] == 20
    assert row["min_mean_10"] == (20 + 24 + 18) / 3
