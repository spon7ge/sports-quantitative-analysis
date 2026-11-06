import ast
import json
from pathlib import Path

import pandas as pd

from src.live.predict_points import (
    FEATURES,
    build_points_live,
    lineups_on,
    pinnacle_games_on,
    points_pipeline,
    unavailable_on,
)

_POINTS_NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks/nba/points/points_xgboost.ipynb"
)

_GAMES = [
    {
        "participants": ["Memphis Grizzlies", "Portland Trail Blazers"],
        "team_markets": {
            "spread": [
                {
                    "period": 0,
                    "is_alternate": True,
                    "lines": [
                        {"team": "Portland Trail Blazers", "points": -5.5},
                        {"team": "Memphis Grizzlies", "points": 5.5},
                    ],
                },
                {
                    "period": 0,
                    "lines": [
                        {"team": "Portland Trail Blazers", "points": -8.0, "side": "home"},
                        {"team": "Memphis Grizzlies", "points": 8.0, "side": "away"},
                    ],
                },
            ],
            "total": [
                {
                    "period": 0,
                    "lines": [
                        {"side": "over", "points": 234.5},
                        {"side": "under", "points": 234.5},
                    ],
                }
            ],
        }
    }
]


def _row(
    player,
    player_id,
    team,
    team_id,
    opponent_id,
    game_id,
    date,
    pts,
    fga,
    started,
    minutes,
    defense,
    *,
    pace,
    cfga,
    ufga,
    tchs,
    usg,
):
    return {
        "player_id": player_id,
        "player_name": player,
        "team_abbreviation": team,
        "team_id": team_id,
        "opp_team_id": opponent_id,
        "game_id": game_id,
        "game_date": date,
        "season_year": "2026-27",
        "pts": pts,
        "fga": fga,
        "fta": 2,
        "fg3_a": 3,
        "cfga": cfga,
        "ufga": ufga,
        "tchs": tchs,
        "usg_pct": usg,
        "minutes": minutes,
        "start_position": "G" if started else "",
        "team_pace": pace,
        "team_def_rating": defense,
        "team_off_rating": defense,
    }


_GAMES_SIDE_ONLY = [
    {
        "participants": ["Memphis Grizzlies", "Portland Trail Blazers"],
        "team_markets": {
            "spread": [
                {
                    "period": 0,
                    "lines": [
                        {"side": "home", "points": -8.0},
                        {"side": "away", "points": 8.0},
                    ],
                }
            ],
            "total": [
                {
                    "period": 0,
                    "lines": [
                        {"side": "over", "points": 234.5},
                        {"side": "under", "points": 234.5},
                    ],
                }
            ],
        },
    }
]


def _history_for_pipeline():
    games = [
        ("2026-10-22", 1, 10, 8, True, 20, 110, 4, 2, 40, 100),
        ("2026-10-24", 2, 20, 12, True, 24, 112, 6, 3, 50, 120),
        ("2026-10-26", 3, 30, 10, False, 18, 108, 5, 1, 30, 110),
    ]
    rows = []
    for date, game_id, pts, fga, started, minutes, defense, cfga, ufga, tchs, opp_def in games:
        rows.append(
            _row(
                "Scoot Henderson", 1, "POR", 1, 2, game_id, date,
                pts, fga, started, minutes, defense,
                pace=100, cfga=cfga, ufga=ufga, tchs=tchs, usg=0.20,
            )
        )
        rows.append(
            _row(
                "Anfernee Simons", 2, "POR", 1, 2, game_id, date,
                18, fga + 4, True, 30, defense,
                pace=100, cfga=cfga, ufga=ufga, tchs=tchs, usg=0.25,
            )
        )
        rows.append(
            _row(
                "Ja Morant", 3, "MEM", 2, 1, game_id, date,
                22, 16, True, 32, opp_def,
                pace=104, cfga=7, ufga=4, tchs=60, usg=0.30,
            )
        )
    return pd.DataFrame(rows)


def _notebook_features() -> list[str]:
    notebook = json.loads(_POINTS_NOTEBOOK.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "PTS_FEATURES = [" in "".join(cell.get("source", []))
    )
    return next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PTS_FEATURES"
            for target in node.targets
        )
    )


def test_feature_list_matches_the_points_notebook():
    assert FEATURES == _notebook_features()
    assert len(FEATURES) == 29


def test_pipeline_accepts_iso_history_dates_with_calendar_slate(monkeypatch):
    """Silver logs are ISO timestamps; live slate rows are YYYY-MM-DD."""
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )
    history = _history_for_pipeline().copy()
    history["game_date"] = (
        pd.to_datetime(history["game_date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    row = points_pipeline(
        history,
        "Scoot Henderson",
        "2026-10-28",
        games=_GAMES,
        lineups={"POR": ["Scoot Henderson"]},
    )
    assert list(row.index) == FEATURES
    assert row["starting"] == 1.0
    assert row["player_team_spread"] == -8.0


def test_pipeline_uses_recent_games_and_pinnacle_team_spread(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )

    row = points_pipeline(
        _history_for_pipeline(),
        "Scoot Henderson",
        "2026-10-28",
        games=_GAMES,
        lineups={"POR": ["Scoot Henderson"]},
    )

    assert list(row.index) == FEATURES
    assert row["min_lag_1"] == 18
    assert row["minutes_rolling_10"] == (20 + 24 + 18) / 3
    assert row["min_sd_10"] == pd.Series([20, 24, 18]).std()
    assert row["start_rate_10"] == 2 / 3
    assert row["starter_rank"] == 2
    assert row["fga_rank_10"] == 2
    assert row["usg_rank_10"] == 2
    assert row["usg_rolling_10"] == 0.20
    assert row["fta_rolling_20"] == 2
    assert row["fg3a_rolling_20"] == 3
    assert row["fg3a_rolling_5"] == 3
    assert row["cfga_rolling_10"] == 5
    assert row["cfga_rolling_5"] == 5
    assert row["cfga_mean_3_minus_10"] == 0
    assert row["ufga_rolling_5"] == 2
    assert row["tchs_rolling_5"] == 40
    assert row["tchs_max_10"] == 50
    assert row["opp_def_rating_mean_10"] == 110
    assert row["expected_pace"] == 102
    assert row["team_games_prev_5d"] == 2
    true_shooting = 60 / (2 * (30 + 0.44 * 6))
    assert row["ts_x_usg"] == pd.Series([0.20 * true_shooting]).iloc[0]
    assert row["player_team_spread"] == -8.0
    assert row["implied_team_total"] == 234.5 / 2 - (-8.0) / 2
    assert row["starting"] == 1


def test_pipeline_resolves_spread_from_side_when_team_missing(monkeypatch):
    """Older Pinnacle team snapshots omit team on spread lines (side only)."""
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )

    row = points_pipeline(
        _history_for_pipeline(),
        "Scoot Henderson",
        "2026-10-28",
        games=_GAMES_SIDE_ONLY,
        lineups={"POR": ["Scoot Henderson"]},
    )

    assert row["player_team_spread"] == -8.0
    assert row["implied_team_total"] == 234.5 / 2 - (-8.0) / 2


def test_pipeline_leaves_out_the_slate_date(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )
    history = _history_for_pipeline()
    leaked = _row(
        "Scoot Henderson", 1, "POR", 1, 2, 4, "2026-10-28",
        100, 40, True, 40, 150,
        pace=130, cfga=99, ufga=99, tchs=99, usg=0.9,
    )
    leaked_defense = _row(
        "Ja Morant", 3, "MEM", 2, 1, 4, "2026-10-28",
        50, 20, True, 36, 999,
        pace=130, cfga=99, ufga=99, tchs=99, usg=0.9,
    )
    history = pd.concat([history, pd.DataFrame([leaked, leaked_defense])], ignore_index=True)

    row = points_pipeline(
        history,
        "Scoot Henderson",
        "2026-10-28",
        games=_GAMES,
        lineups={"POR": ["Scoot Henderson"]},
    )

    assert row["min_lag_1"] == 18
    assert row["minutes_rolling_10"] == (20 + 24 + 18) / 3
    assert row["cfga_rolling_10"] == 5
    assert row["opp_def_rating_mean_10"] == 110


def test_missing_lineup_leaves_starting_missing(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )

    row = points_pipeline(
        _history_for_pipeline(),
        "Scoot Henderson",
        "2026-10-28",
        games=_GAMES,
        lineups={},
    )

    assert pd.isna(row["starting"])
    assert row["min_lag_1"] == 18


def test_history_without_player_id_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )
    history = _history_for_pipeline().drop(columns=["player_id"])

    try:
        points_pipeline(
            history,
            "Scoot Henderson",
            "2026-10-28",
            games=_GAMES,
            lineups={"POR": ["Scoot Henderson"]},
        )
    except KeyError as error:
        assert "player_id" in str(error)
    else:
        raise AssertionError("missing player_id was accepted")


def test_pinnacle_games_on_reads_newest_snapshot_first(tmp_path: Path):
    def payload(points: float) -> dict:
        return {
            "games": [
                {
                    "team_markets": {
                        "spread": [
                            {
                                "period": 0,
                                "lines": [
                                    {"team": "Portland Trail Blazers", "points": points},
                                ],
                            }
                        ]
                    }
                }
            ]
        }

    (tmp_path / "pinnacle_nba_2026-10-27_120000_team.json").write_text(
        json.dumps(payload(-3.0))
    )
    (tmp_path / "pinnacle_nba_2026-10-27_190000_team.json").write_text(
        json.dumps(payload(-8.0))
    )
    (tmp_path / "pinnacle_nba_2026-10-28_190000_team.json").write_text(
        json.dumps(payload(-20.0))
    )

    games = pinnacle_games_on("2026-10-27", directory=tmp_path)

    assert len(games) == 2
    assert games[0]["team_markets"]["spread"][0]["lines"][0]["points"] == -8.0


def test_build_points_live_batches_a_matchup_without_schedule_lookup(monkeypatch):
    """Live board: one slate pass, opponent from Pinnacle participants."""
    calls = {"findOpp": 0}

    def _boom(*args, **kwargs):
        calls["findOpp"] += 1
        raise AssertionError("live slate must not call findOpp when Pinnacle has the game")

    monkeypatch.setattr("src.live.predict_points.findOpp", _boom)

    live = build_points_live(
        _history_for_pipeline(),
        ["Scoot Henderson", "Ja Morant", "Anfernee Simons"],
        "2026-10-28",
        games=_GAMES,
        lineups={
            "POR": ["Scoot Henderson", "Anfernee Simons"],
            "MEM": ["Ja Morant"],
        },
    )

    assert list(live.index) == ["Scoot Henderson", "Ja Morant", "Anfernee Simons"]
    assert list(live.columns) == [*FEATURES, "pts_lag_1"]
    assert calls["findOpp"] == 0
    scoot = live.loc["Scoot Henderson"]
    ja = live.loc["Ja Morant"]
    assert scoot["starting"] == 1
    assert scoot["player_team_spread"] == -8.0
    assert scoot["implied_team_total"] == 234.5 / 2 - (-8.0) / 2
    assert scoot["opp_def_rating_mean_10"] == 110
    assert scoot["expected_pace"] == 102
    assert scoot["pts_lag_1"] == 30
    assert ja["starting"] == 1
    assert ja["player_team_spread"] == 8.0
    assert ja["opp_def_rating_mean_10"] == 110
    assert ja["pts_lag_1"] == 22


def test_build_points_live_leaves_out_the_slate_date(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )
    history = _history_for_pipeline()
    leaked = pd.DataFrame(
        [
            _row(
                "Scoot Henderson", 1, "POR", 1, 2, 4, "2026-10-28",
                100, 40, True, 40, 150,
                pace=130, cfga=99, ufga=99, tchs=99, usg=0.9,
            ),
            _row(
                "Ja Morant", 3, "MEM", 2, 1, 4, "2026-10-28",
                50, 20, True, 36, 999,
                pace=130, cfga=99, ufga=99, tchs=99, usg=0.9,
            ),
        ]
    )
    history = pd.concat([history, leaked], ignore_index=True)

    live = build_points_live(
        history,
        ["Scoot Henderson"],
        "2026-10-28",
        games=_GAMES,
        lineups={"POR": ["Scoot Henderson"]},
    )

    assert live.loc["Scoot Henderson", "min_lag_1"] == 18
    assert live.loc["Scoot Henderson", "cfga_rolling_10"] == 5
    assert live.loc["Scoot Henderson", "opp_def_rating_mean_10"] == 110


def test_build_points_live_keeps_starting_missing_without_a_posted_five(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )

    live = build_points_live(
        _history_for_pipeline(),
        ["Scoot Henderson"],
        "2026-10-28",
        games=_GAMES,
        lineups={},
    )

    assert pd.isna(live.loc["Scoot Henderson", "starting"])


def test_build_points_live_matches_board_names_without_suffix(monkeypatch):
    monkeypatch.setattr(
        "src.live.predict_points.findOpp",
        lambda *args, **kwargs: ("MEM", 0),
    )
    history = _history_for_pipeline()
    history.loc[history["player_name"].eq("Scoot Henderson"), "player_name"] = (
        "Scoot Henderson Jr."
    )

    live = build_points_live(
        history,
        ["Scoot Henderson", "Missing Player"],
        "2026-10-28",
        games=_GAMES,
        lineups={"POR": ["Scoot Henderson Jr."]},
    )

    assert list(live.index) == ["Scoot Henderson"]
    assert live.loc["Scoot Henderson", "min_lag_1"] == 18
    assert live.loc["Scoot Henderson", "starting"] == 1


def test_lineups_on_reads_latest_rotowire_file(tmp_path: Path):

    early = {
        "games": [
            {
                "home": {"team": "POR", "lineup": [{"name": "Old Name"}]},
                "away": {"team": "CHI", "lineup": [{"name": "Coby White"}]},
            }
        ]
    }
    late = {
        "games": [
            {
                "home": {
                    "team": "POR",
                    "lineup": [
                        {"name": "Scoot Henderson"},
                        {"name": "Anfernee Simons"},
                    ],
                },
                "away": {"team": "MEM", "lineup": [{"name": "Ja Morant"}]},
            }
        ]
    }
    (tmp_path / "rotowire_nba_2026-10-27_120000.json").write_text(json.dumps(early))
    (tmp_path / "rotowire_nba_2026-10-27_190000.json").write_text(json.dumps(late))

    lineups = lineups_on("2026-10-27", directory=tmp_path)

    assert lineups["POR"] == ["Scoot Henderson", "Anfernee Simons"]
    assert lineups["MEM"] == ["Ja Morant"]
    assert lineups["CHI"] == ["Coby White"]
    assert lineups_on("2026-10-22", directory=tmp_path) == {}


def test_unavailable_on_keeps_out_players_from_the_newest_file(tmp_path: Path):
    early = {
        "games": [
            {
                "home": {
                    "team": "POR",
                    "lineup": [{"name": "Scoot Henderson"}],
                    "may_not_play": [
                        {"name": "Kristaps Porzingis", "badge": "Out"},
                        {"name": "Jerami Grant", "badge": "Ques"},
                    ],
                },
                "away": {
                    "team": "CHI",
                    "lineup": [{"name": "Coby White"}],
                    "may_not_play": [{"name": "Lonzo Ball", "badge": "OFS"}],
                },
            }
        ]
    }
    late = {
        "games": [
            {
                "home": {
                    "team": "POR",
                    "lineup": [{"name": "Scoot Henderson"}],
                    "may_not_play": [{"name": "Jerami Grant", "badge": "Doubt"}],
                },
                "away": {"team": "MEM", "lineup": [{"name": "Ja Morant"}]},
            }
        ]
    }
    (tmp_path / "rotowire_nba_2026-10-27_120000.json").write_text(json.dumps(early))
    (tmp_path / "rotowire_nba_2026-10-27_190000.json").write_text(json.dumps(late))

    unavailable = unavailable_on("2026-10-27", directory=tmp_path)

    assert unavailable["POR"] == []
    assert unavailable["CHI"] == ["Lonzo Ball"]
    assert unavailable["MEM"] == []
    assert unavailable_on("2026-10-22", directory=tmp_path) == {}
