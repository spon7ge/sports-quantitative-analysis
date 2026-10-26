"""Build deterministic leader/follower episodes from movement rows."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

DEFAULT_FOLLOW_WINDOW = timedelta(minutes=30)
_EPSILON = 1e-9

_CONTINUITY_COLUMNS = ["player_norm", "event_key", "prop", "period", "side"]
_DIRECTION_FLAGS = [
    "line_up",
    "line_down",
    "price_shortened",
    "price_lengthened",
]
_EPISODE_COLUMNS = [
    "episode_id",
    *_CONTINUITY_COLUMNS,
    "direction",
    "leader_book",
    "leader_snapshot_at",
    "change_type",
]
_FOLLOW_COLUMNS = [
    "episode_id",
    "follower_book",
    "follower_snapshot_at",
    "lag_seconds",
    "within_window",
    "direction",
]
_STATS_COLUMNS = [
    "leader_book",
    "follower_book",
    "lead_count",
    "follow_count_windowed",
    "follow_rate_windowed",
    "median_lag_windowed_sec",
    "mean_lag_windowed_sec",
    "median_lag_uncapped_sec",
]
_REQUIRED_MOVEMENT_COLUMNS = {
    "book",
    "snapshot_at",
    "change_type",
    *_CONTINUITY_COLUMNS,
    *_DIRECTION_FLAGS,
}


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _directed_movements(movements: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_MOVEMENT_COLUMNS.difference(movements.columns))
    if missing:
        raise ValueError(f"movements missing required columns: {', '.join(missing)}")

    directed = movements.dropna(
        subset=[*_CONTINUITY_COLUMNS, "book", "snapshot_at"]
    ).copy()
    if {"prev_line", "new_line"}.issubset(directed.columns):
        prev_line = pd.to_numeric(directed["prev_line"], errors="coerce")
        new_line = pd.to_numeric(directed["new_line"], errors="coerce")
        zero_line = prev_line.abs().lt(_EPSILON) | new_line.abs().lt(_EPSILON)
        directed = directed.loc[~zero_line].copy()
    if directed.empty:
        return directed.assign(direction=pd.Series(dtype="object"))

    flags = directed[_DIRECTION_FLAGS].fillna(False).astype(bool)
    directed["direction"] = pd.NA
    for flag in _DIRECTION_FLAGS:
        unset = directed["direction"].isna()
        directed.loc[unset & flags[flag], "direction"] = flag
    directed = directed.dropna(subset=["direction"])
    return directed.sort_values(
        [*_CONTINUITY_COLUMNS, "direction", "snapshot_at", "book"],
        kind="stable",
    ).reset_index(drop=True)


def _candidate_episodes(
    directed: pd.DataFrame, window: timedelta
) -> list[pd.Series]:
    candidates: list[pd.Series] = []
    stream_columns = [*_CONTINUITY_COLUMNS, "direction"]
    for _, stream in directed.groupby(stream_columns, sort=False, dropna=False):
        leader: pd.Series | None = None
        for _, movement in stream.iterrows():
            if (
                leader is None
                or movement["snapshot_at"] - leader["snapshot_at"] > window
            ):
                if leader is not None:
                    candidates.append(leader)
                leader = movement
        if leader is not None:
            candidates.append(leader)
    return candidates


def build_episodes(
    movements: pd.DataFrame,
    *,
    window: timedelta = DEFAULT_FOLLOW_WINDOW,
) -> pd.DataFrame:
    """Split streams whenever a movement is beyond the leader-time window."""
    if window < timedelta(0):
        raise ValueError("window must be non-negative")
    directed = _directed_movements(movements)
    if directed.empty:
        return _empty(_EPISODE_COLUMNS)

    rows = []
    for episode_id, leader in enumerate(_candidate_episodes(directed, window)):
        rows.append(
            {
                "episode_id": episode_id,
                **{column: leader[column] for column in _CONTINUITY_COLUMNS},
                "direction": leader["direction"],
                "leader_book": leader["book"],
                "leader_snapshot_at": leader["snapshot_at"],
                "change_type": leader["change_type"],
            }
        )
    return pd.DataFrame(rows, columns=_EPISODE_COLUMNS)


def build_follows(
    movements: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    window: timedelta = DEFAULT_FOLLOW_WINDOW,
) -> pd.DataFrame:
    """Attach each book's first later movement to the latest episode leader."""
    if window < timedelta(0):
        raise ValueError("window must be non-negative")
    directed = _directed_movements(movements)
    if directed.empty or episodes.empty:
        return _empty(_FOLLOW_COLUMNS)

    rows: list[dict[str, object]] = []
    stream_columns = [*_CONTINUITY_COLUMNS, "direction"]
    ordered_episodes = episodes.sort_values(
        [*stream_columns, "leader_snapshot_at", "leader_book"], kind="stable"
    )
    for stream_key, stream_episodes in ordered_episodes.groupby(
        stream_columns, sort=False, dropna=False
    ):
        key_values = stream_key if isinstance(stream_key, tuple) else (stream_key,)
        mask = pd.Series(True, index=directed.index)
        for column, value in zip(stream_columns, key_values, strict=True):
            mask &= directed[column].eq(value)
        stream_movements = directed.loc[mask]
        episode_records = list(stream_episodes.to_dict("records"))
        for index, episode in enumerate(episode_records):
            leader_at = episode["leader_snapshot_at"]
            next_leader_at = (
                episode_records[index + 1]["leader_snapshot_at"]
                if index + 1 < len(episode_records)
                else None
            )
            follows = stream_movements[
                stream_movements["snapshot_at"].gt(leader_at)
                & stream_movements["book"].ne(episode["leader_book"])
            ]
            if next_leader_at is not None:
                follows = follows[follows["snapshot_at"].lt(next_leader_at)]
            follows = follows.drop_duplicates(subset=["book"], keep="first")
            for _, follower in follows.iterrows():
                lag_seconds = (follower["snapshot_at"] - leader_at).total_seconds()
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "follower_book": follower["book"],
                        "follower_snapshot_at": follower["snapshot_at"],
                        "lag_seconds": lag_seconds,
                        "within_window": lag_seconds <= window.total_seconds(),
                        "direction": episode["direction"],
                    }
                )
    return pd.DataFrame(rows, columns=_FOLLOW_COLUMNS)


def pairwise_stats(follows: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed leader/follower pairs and their lag statistics."""
    if follows.empty or episodes.empty:
        return _empty(_STATS_COLUMNS)

    joined = follows.merge(
        episodes[["episode_id", "leader_book"]],
        on="episode_id",
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        return _empty(_STATS_COLUMNS)

    lead_counts = episodes["leader_book"].value_counts()
    rows = []
    for (leader_book, follower_book), pair in joined.groupby(
        ["leader_book", "follower_book"], sort=True
    ):
        windowed = pair.loc[pair["within_window"].fillna(False)]
        lead_count = int(lead_counts.get(leader_book, 0))
        follow_count = len(windowed)
        rows.append(
            {
                "leader_book": leader_book,
                "follower_book": follower_book,
                "lead_count": lead_count,
                "follow_count_windowed": follow_count,
                "follow_rate_windowed": (
                    follow_count / lead_count if lead_count else float("nan")
                ),
                "median_lag_windowed_sec": windowed["lag_seconds"].median(),
                "mean_lag_windowed_sec": windowed["lag_seconds"].mean(),
                "median_lag_uncapped_sec": pair["lag_seconds"].median(),
            }
        )
    return pd.DataFrame(rows, columns=_STATS_COLUMNS)
