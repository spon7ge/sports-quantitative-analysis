from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from datetime import timedelta

from line_movement.viz import (
    assign_move_waves,
    continuity_line_path,
    current_differences,
    fixed_line_price_path,
    follower_rankings,
    line_move_log,
    plot_leader_follower_heatmap,
    plot_line_steps,
    plot_movement_timeline,
    plot_price_steps,
    summarize_timeline,
)


def _quote(book, snap, line, price, **kw):
    row = {
        "book": book,
        "snapshot_at": pd.Timestamp(snap, tz="America/Los_Angeles"),
        "player_norm": "test player",
        "event_key": "aaa__bbb",
        "prop": "points",
        "period": "game",
        "side": "over",
        "line": line,
        "american_price": price,
        "is_main": True,
        "line_updated_at": pd.NaT,
    }
    row.update(kw)
    return row


def test_continuity_line_path_is_one_line_per_book_snapshot():
    quotes = pd.DataFrame(
        [
            _quote("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _quote("fanduel", "2026-10-24 12:00:00", 25.5, -200, is_main=False),
            _quote("fanduel", "2026-10-24 12:15:00", 21.5, -110),
            _quote("draftkings", "2026-10-24 12:00:00", 20.5, -115),
            _quote("draftkings", "2026-10-24 12:15:00", 20.5, -115),
        ]
    )
    path = continuity_line_path(
        quotes,
        player_norm="test player",
        event_key="aaa__bbb",
        prop="points",
        period="game",
        side="over",
    )
    assert set(path.columns) >= {"book", "snapshot_at", "line", "american_price"}
    fd = path[path["book"] == "fanduel"].sort_values("snapshot_at")
    assert list(fd["line"]) == [20.5, 21.5]


def test_fixed_line_price_path_stays_on_one_line():
    quotes = pd.DataFrame(
        [
            _quote("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _quote("fanduel", "2026-10-24 12:15:00", 20.5, -120),
            _quote("fanduel", "2026-10-24 12:30:00", 21.5, -110),
            _quote("novig", "2026-10-24 12:00:00", 20.5, -105),
            _quote("novig", "2026-10-24 12:15:00", 20.5, -108),
        ]
    )
    path = fixed_line_price_path(
        quotes,
        line=20.5,
        player_norm="test player",
        event_key="aaa__bbb",
        prop="points",
        period="game",
        side="over",
    )
    assert set(path["line"].unique()) == {20.5}
    fd = path[path["book"] == "fanduel"].sort_values("snapshot_at")
    assert list(fd["american_price"]) == [-110, -120]


def test_line_move_log_flags_reversals():
    moves = pd.DataFrame(
        [
            {
                "book": "fanduel",
                "player_norm": "test player",
                "event_key": "aaa__bbb",
                "prop": "points",
                "period": "game",
                "side": "over",
                "change_type": "line",
                "line_up": True,
                "line_down": False,
                "prev_line": 20.5,
                "new_line": 21.5,
                "snapshot_at": pd.Timestamp("2026-10-24 12:15:00", tz="America/Los_Angeles"),
            },
            {
                "book": "fanduel",
                "player_norm": "test player",
                "event_key": "aaa__bbb",
                "prop": "points",
                "period": "game",
                "side": "over",
                "change_type": "line",
                "line_up": False,
                "line_down": True,
                "prev_line": 21.5,
                "new_line": 20.5,
                "snapshot_at": pd.Timestamp("2026-10-24 12:45:00", tz="America/Los_Angeles"),
            },
        ]
    )
    log = line_move_log(
        moves,
        player_norm="test player",
        event_key="aaa__bbb",
        prop="points",
        period="game",
        side="over",
    )
    assert list(log["direction"]) == ["up", "down"]
    assert list(log["reversed"]) == [False, True]


def test_plot_helpers_return_axes():
    path = pd.DataFrame(
        {
            "book": ["fanduel", "fanduel", "novig"],
            "snapshot_at": pd.to_datetime(
                [
                    "2026-10-24 12:00:00-07:00",
                    "2026-10-24 12:15:00-07:00",
                    "2026-10-24 12:00:00-07:00",
                ]
            ),
            "line": [20.5, 21.5, 20.5],
            "american_price": [-110, -110, -105],
        }
    )
    assert plot_line_steps(path) is not None
    assert plot_price_steps(path) is not None


def test_assign_move_waves_splits_on_gap():
    events = pd.DataFrame(
        {
            "book": ["fanduel", "novig", "draftkings", "pinnacle"],
            "snapshot_at": pd.to_datetime(
                [
                    "2026-10-24 12:00:00-07:00",
                    "2026-10-24 12:02:00-07:00",
                    "2026-10-24 12:30:00-07:00",
                    "2026-10-24 12:31:00-07:00",
                ]
            ),
            "direction": ["up", "up", "down", "down"],
        }
    )
    waved = assign_move_waves(events, gap=timedelta(minutes=5))
    assert list(waved["wave_id"]) == [0, 0, 1, 1]


def test_summarize_timeline_describes_clustering():
    events = pd.DataFrame(
        {
            "book": ["fanduel", "novig", "draftkings"],
            "snapshot_at": pd.to_datetime(
                [
                    "2026-10-24 12:00:00-07:00",
                    "2026-10-24 12:01:00-07:00",
                    "2026-10-24 13:00:00-07:00",
                ]
            ),
            "direction": ["up", "up", "down"],
            "wave_id": [0, 0, 1],
        }
    )
    summary = summarize_timeline(events)
    assert summary["n_waves"] == 2
    assert summary["multi_book_waves"] == 1
    assert summary["solo_waves"] == 1
    assert "wave" in summary["label"].lower() or "cluster" in summary["label"].lower()


def test_plot_movement_timeline_returns_axes():
    events = pd.DataFrame(
        {
            "book": ["fanduel", "novig", "fanduel"],
            "snapshot_at": pd.to_datetime(
                [
                    "2026-10-24 12:00:00-07:00",
                    "2026-10-24 12:01:00-07:00",
                    "2026-10-24 12:30:00-07:00",
                ]
            ),
            "direction": ["up", "up", "down"],
            "wave_id": [0, 0, 1],
        }
    )
    ax = plot_movement_timeline(events)
    assert ax is not None


def test_leader_follower_heatmap_and_follower_rankings():
    pairwise = pd.DataFrame(
        {
            "leader_book": ["fanduel", "fanduel", "novig"],
            "follower_book": ["novig", "draftkings", "draftkings"],
            "median_lag_windowed_sec": [60.0, 120.0, 30.0],
            "follow_count_windowed": [10, 4, 8],
            "follow_rate_windowed": [0.5, 0.2, 0.4],
        }
    )
    ax = plot_leader_follower_heatmap(pairwise)
    assert ax is not None
    ranks = follower_rankings(pairwise)
    assert list(ranks["follower_book"]) == ["draftkings", "novig"]
    assert ranks.loc[ranks["follower_book"].eq("draftkings"), "follow_count"].iloc[
        0
    ] == 12


def test_current_differences_reports_gap_and_persistence():
    quotes = pd.DataFrame(
        [
            _quote("fanduel", "2026-10-24 12:00:00", 20.5, -110),
            _quote("fanduel", "2026-10-24 12:15:00", 20.5, -110),
            _quote("fanduel", "2026-10-24 12:30:00", 21.5, -115),
            _quote("novig", "2026-10-24 12:00:00", 20.5, -108),
            _quote("novig", "2026-10-24 12:15:00", 20.5, -108),
            _quote("novig", "2026-10-24 12:30:00", 20.5, -105),
            _quote("pinnacle", "2026-10-24 12:00:00", 20.5, -109),
            _quote("pinnacle", "2026-10-24 12:15:00", 20.5, -109),
            _quote("pinnacle", "2026-10-24 12:30:00", 20.5, -109),
        ]
    )
    table = current_differences(
        quotes,
        player_norm="test player",
        event_key="aaa__bbb",
        prop="points",
        period="game",
        side="over",
    )
    assert set(table.columns) >= {
        "book",
        "current_line",
        "american_price",
        "consensus_line",
        "gap_from_consensus",
        "gap_persisted_sec",
        "quote_age_sec",
        "differs",
    }
    fd = table.set_index("book").loc["fanduel"]
    assert fd["current_line"] == 21.5
    assert fd["consensus_line"] == 20.5
    assert fd["gap_from_consensus"] == 1.0
    assert bool(fd["differs"]) is True
    assert fd["gap_persisted_sec"] == 0.0
    assert bool(table.set_index("book").loc["novig", "differs"]) is False
