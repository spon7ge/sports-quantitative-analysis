from __future__ import annotations

from datetime import timedelta

import line_movement
import pandas as pd
from line_movement.leaders import build_episodes, build_follows, pairwise_stats


def _move(book, snap, **flags):
    base = {
        "book": book,
        "player_norm": "test player",
        "event_key": "aaa__bbb",
        "prop": "points",
        "period": "game",
        "side": "over",
        "change_type": "price",
        "line_up": False,
        "line_down": False,
        "price_shortened": False,
        "price_lengthened": False,
        "snapshot_at": pd.Timestamp(snap, tz="America/Los_Angeles"),
        "prev_snapshot_at": pd.NaT,
        "lag_seconds": None,
        "prev_line": 20.5,
        "new_line": 20.5,
        "prev_american_price": -110,
        "new_american_price": -120,
        "series": "price",
        "line_choice": None,
    }
    base.update(flags)
    return base


def test_movement_after_window_starts_singleton_episode():
    moves = pd.DataFrame(
        [
            _move("fanduel", "2026-10-24 12:00:00", price_shortened=True),
            _move("draftkings", "2026-10-24 12:15:00", price_shortened=True),
            _move("pinnacle", "2026-10-24 13:00:00", price_shortened=True),
        ]
    )
    window = timedelta(minutes=30)

    episodes = build_episodes(moves, window=window)
    follows = build_follows(moves, episodes, window=window)

    assert list(episodes["leader_book"]) == ["fanduel", "pinnacle"]
    assert list(follows["follower_book"]) == ["draftkings"]
    assert follows.iloc[0]["within_window"]

    stats = pairwise_stats(follows, episodes)
    fd = stats[
        (stats["leader_book"] == "fanduel") & (stats["follower_book"] == "draftkings")
    ].iloc[0]
    assert fd["follow_count_windowed"] == 1
    assert fd["median_lag_windowed_sec"] == 15 * 60


def test_public_functions_are_exported():
    assert line_movement.build_episodes is build_episodes
    assert line_movement.build_follows is build_follows
    assert line_movement.pairwise_stats is pairwise_stats


def test_direction_prefers_line_and_ignores_incomplete_rows():
    moves = pd.DataFrame(
        [
            _move(
                "fanduel",
                "2026-10-24 12:00:00",
                change_type="line",
                line_up=True,
                price_shortened=True,
            ),
            _move(
                "draftkings",
                "2026-10-24 12:05:00",
                change_type="line",
                line_up=True,
            ),
            _move(
                "pinnacle",
                "2026-10-24 12:10:00",
                player_norm=None,
                line_up=True,
            ),
        ]
    )

    episodes = build_episodes(moves, window=timedelta(minutes=30))

    assert len(episodes) == 1
    assert episodes.iloc[0]["direction"] == "line_up"
    assert episodes.iloc[0]["change_type"] == "line"


def test_new_episode_starts_after_window_when_another_book_follows():
    moves = pd.DataFrame(
        [
            _move("fanduel", "2026-10-24 12:00:00", price_shortened=True),
            _move("draftkings", "2026-10-24 12:10:00", price_shortened=True),
            _move("pinnacle", "2026-10-24 13:00:00", price_shortened=True),
            _move("betrivers", "2026-10-24 13:10:00", price_shortened=True),
        ]
    )
    window = timedelta(minutes=30)

    episodes = build_episodes(moves, window=window)
    follows = build_follows(moves, episodes, window=window)

    assert list(episodes["leader_book"]) == ["fanduel", "pinnacle"]
    assert list(follows["follower_book"]) == ["draftkings", "betrivers"]


def test_only_first_follow_per_episode_and_follower_book_is_kept():
    moves = pd.DataFrame(
        [
            _move("fanduel", "2026-10-24 12:00:00", price_shortened=True),
            _move("draftkings", "2026-10-24 12:10:00", price_shortened=True),
            _move("draftkings", "2026-10-24 12:20:00", price_shortened=True),
        ]
    )
    window = timedelta(minutes=30)

    episodes = build_episodes(moves, window=window)
    follows = build_follows(moves, episodes, window=window)
    stats = pairwise_stats(follows, episodes)

    assert len(follows) == 1
    assert follows.iloc[0]["follower_snapshot_at"] == pd.Timestamp(
        "2026-10-24 12:10:00", tz="America/Los_Angeles"
    )
    row = stats.iloc[0]
    assert row["lead_count"] == 1
    assert row["follow_count_windowed"] == 1
    assert row["follow_rate_windowed"] == 1.0
    assert row["mean_lag_windowed_sec"] == 10 * 60
    assert row["median_lag_uncapped_sec"] == 10 * 60


def test_same_timestamp_other_book_is_not_a_follow():
    moves = pd.DataFrame(
        [
            _move("fanduel", "2026-10-24 12:00:00", price_shortened=True),
            _move("draftkings", "2026-10-24 12:00:00", price_shortened=True),
        ]
    )

    episodes = build_episodes(moves, window=timedelta(minutes=30))
    follows = build_follows(moves, episodes, window=timedelta(minutes=30))

    assert episodes.iloc[0]["leader_book"] == "draftkings"
    assert follows.empty


def test_zero_line_movement_does_not_create_episode():
    moves = pd.DataFrame(
        [
            _move(
                "fanduel",
                "2026-10-24 12:00:00",
                change_type="line",
                line_down=True,
                prev_line=20.5,
                new_line=0.0,
                series="line",
            )
        ]
    )

    assert build_episodes(moves).empty
