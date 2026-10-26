"""Step charts and movement timelines across books."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from line_movement.detect import _EPSILON, _choose_line

_DIRECTION_COLORS = {
    "up": "#2a9d8f",
    "down": "#e76f51",
    "shortened": "#457b9d",
    "lengthened": "#f4a261",
}
_DEFAULT_WAVE_GAP = timedelta(minutes=5)

# Continuity market without book (book is the series dimension on the chart).
_MARKET_COLS = ("player_norm", "event_key", "prop", "period", "side")


def _filter_market(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    missing = [name for name in filters if name not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns: {', '.join(sorted(missing))}")
    out = frame
    for name, value in filters.items():
        out = out.loc[out[name].eq(value)]
    return out.copy()


def _gap_persisted_seconds(
    book_path: pd.DataFrame,
    *,
    consensus_line: float,
    as_of: pd.Timestamp,
) -> float:
    """Seconds the book has continuously differed from consensus ending at as_of."""
    ordered = book_path.sort_values("snapshot_at", kind="stable")
    if ordered.empty:
        return float("nan")
    latest_line = float(ordered.iloc[-1]["line"])
    if abs(latest_line - consensus_line) <= _EPSILON:
        return 0.0

    lines = pd.to_numeric(ordered["line"], errors="coerce")
    matched = ordered.loc[(lines - consensus_line).abs() <= _EPSILON]
    if matched.empty:
        start = ordered.iloc[0]["snapshot_at"]
    else:
        last_match = matched.iloc[-1]["snapshot_at"]
        after = ordered.loc[ordered["snapshot_at"] > last_match]
        start = (
            after.iloc[0]["snapshot_at"]
            if not after.empty
            else ordered.iloc[-1]["snapshot_at"]
        )
    return float((as_of - pd.Timestamp(start)) / pd.Timedelta(seconds=1))


def current_differences(
    quotes: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    **filters: Any,
) -> pd.DataFrame:
    """Latest line/odds per book vs consensus, with gap age and quote freshness."""
    path = continuity_line_path(quotes, **filters)
    columns = [
        "book",
        "current_line",
        "american_price",
        "consensus_line",
        "gap_from_consensus",
        "price_gap_from_peers",
        "gap_persisted_sec",
        "quote_age_sec",
        "differs",
        "as_of",
        "snapshot_at",
    ]
    if path.empty:
        return pd.DataFrame(columns=columns)

    path = path.copy()
    path["snapshot_at"] = pd.to_datetime(path["snapshot_at"], errors="coerce")
    path = path.loc[path["snapshot_at"].notna()]
    if path.empty:
        return pd.DataFrame(columns=columns)

    clock = pd.Timestamp(as_of) if as_of is not None else path["snapshot_at"].max()
    latest = (
        path.sort_values("snapshot_at", kind="stable")
        .groupby("book", sort=False, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    consensus = float(pd.to_numeric(latest["line"], errors="coerce").median())
    peer_prices = pd.to_numeric(
        latest.loc[
            (pd.to_numeric(latest["line"], errors="coerce") - consensus).abs()
            <= _EPSILON,
            "american_price",
        ],
        errors="coerce",
    )
    peer_price = float(peer_prices.median()) if peer_prices.notna().any() else float("nan")

    rows: list[dict[str, Any]] = []
    for row in latest.itertuples(index=False):
        book = str(row.book)
        line = float(row.line)
        price = (
            float(row.american_price)
            if pd.notna(row.american_price)
            else float("nan")
        )
        gap = line - consensus
        differs = abs(gap) > _EPSILON
        book_path = path.loc[path["book"].astype(str).eq(book)]
        persisted = (
            _gap_persisted_seconds(
                book_path, consensus_line=consensus, as_of=clock
            )
            if differs
            else 0.0
        )
        age = float(
            (clock - pd.Timestamp(row.snapshot_at)) / pd.Timedelta(seconds=1)
        )
        price_gap = (
            price - peer_price
            if pd.notna(price) and pd.notna(peer_price)
            else float("nan")
        )
        rows.append(
            {
                "book": book,
                "current_line": line,
                "american_price": price,
                "consensus_line": consensus,
                "gap_from_consensus": gap,
                "price_gap_from_peers": price_gap,
                "gap_persisted_sec": persisted,
                "quote_age_sec": age,
                "differs": differs,
                "as_of": clock,
                "snapshot_at": row.snapshot_at,
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(
        ["differs", "gap_persisted_sec", "book"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def continuity_line_path(quotes: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    """One chosen line per book × snapshot for a continuity market."""
    required = [*_MARKET_COLS, "book", "snapshot_at", "line"]
    missing = [name for name in required if name not in quotes.columns]
    if missing:
        raise ValueError(f"quotes missing columns: {', '.join(sorted(missing))}")

    scoped = _filter_market(quotes, **filters)
    if scoped.empty:
        return pd.DataFrame(
            columns=["book", "snapshot_at", "line", "american_price", "line_choice"]
        )

    numeric = pd.to_numeric(scoped["line"], errors="coerce")
    scoped = scoped.loc[numeric.isna() | numeric.abs().gt(_EPSILON)]
    if scoped.empty:
        return pd.DataFrame(
            columns=["book", "snapshot_at", "line", "american_price", "line_choice"]
        )

    if "is_main" not in scoped.columns:
        scoped["is_main"] = pd.NA
    if "line_updated_at" not in scoped.columns:
        scoped["line_updated_at"] = pd.NaT
    if "american_price" not in scoped.columns:
        scoped["american_price"] = pd.NA

    group_cols = ["book", *_MARKET_COLS, "snapshot_at"]
    chosen = (
        scoped.groupby(group_cols, dropna=False, sort=False, group_keys=False)
        .apply(_choose_line, include_groups=False)
        .reset_index()
    )
    cols = ["book", "snapshot_at", "line", "american_price", "line_choice", *_MARKET_COLS]
    present = [col for col in cols if col in chosen.columns]
    return chosen.loc[:, present].sort_values(
        ["book", "snapshot_at"], kind="stable"
    ).reset_index(drop=True)


def fixed_line_price_path(
    quotes: pd.DataFrame, *, line: float, **filters: Any
) -> pd.DataFrame:
    """American prices over time for one fixed line on a continuity market."""
    required = [*_MARKET_COLS, "book", "snapshot_at", "line", "american_price"]
    missing = [name for name in required if name not in quotes.columns]
    if missing:
        raise ValueError(f"quotes missing columns: {', '.join(sorted(missing))}")

    scoped = _filter_market(quotes, **filters)
    lines = pd.to_numeric(scoped["line"], errors="coerce")
    scoped = scoped.loc[(lines - float(line)).abs().le(_EPSILON)]
    scoped = scoped.loc[scoped["american_price"].notna()]
    if scoped.empty:
        return pd.DataFrame(
            columns=["book", "snapshot_at", "line", "american_price", *_MARKET_COLS]
        )

    # One quote per book × snapshot (prefer main when present).
    if "is_main" in scoped.columns:
        scoped = scoped.sort_values("is_main", ascending=False, kind="stable")
    deduped = scoped.drop_duplicates(
        subset=["book", "snapshot_at"], keep="first"
    )
    cols = ["book", "snapshot_at", "line", "american_price", *_MARKET_COLS]
    return deduped.loc[:, cols].sort_values(
        ["book", "snapshot_at"], kind="stable"
    ).reset_index(drop=True)


def line_move_log(movements: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    """Per-book line moves with direction and reversal flags."""
    if movements.empty:
        return pd.DataFrame(
            columns=[
                "book",
                "snapshot_at",
                "prev_line",
                "new_line",
                "direction",
                "reversed",
            ]
        )

    scoped = movements.loc[movements["change_type"].eq("line")].copy()
    scoped = _filter_market(scoped, **filters) if filters else scoped
    if scoped.empty:
        return pd.DataFrame(
            columns=[
                "book",
                "snapshot_at",
                "prev_line",
                "new_line",
                "direction",
                "reversed",
            ]
        )

    scoped = scoped.sort_values(["book", "snapshot_at"], kind="stable")
    direction = pd.Series(pd.NA, index=scoped.index, dtype="string")
    direction = direction.mask(scoped["line_up"].fillna(False), "up")
    direction = direction.mask(scoped["line_down"].fillna(False), "down")
    scoped["direction"] = direction
    previous = scoped.groupby("book", sort=False)["direction"].shift()
    scoped["reversed"] = previous.notna() & direction.notna() & previous.ne(direction)
    cols = [
        "book",
        "snapshot_at",
        "prev_line",
        "new_line",
        "direction",
        "reversed",
        "prev_snapshot_at",
        "lag_seconds",
    ]
    present = [col for col in cols if col in scoped.columns]
    return scoped.loc[:, present].reset_index(drop=True)


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots(figsize=(10, 5))
    return created


def plot_line_steps(
    path: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    reversals: pd.DataFrame | None = None,
) -> Axes:
    """Step chart: time on X, line on Y, one series per book."""
    ax = _axes(ax)
    if path.empty:
        ax.text(0.5, 0.5, "No line path", ha="center", va="center")
        ax.set_axis_off()
        return ax

    for book, group in path.groupby("book", sort=True):
        ordered = group.sort_values("snapshot_at")
        ax.step(
            ordered["snapshot_at"],
            ordered["line"],
            where="post",
            label=str(book),
            linewidth=2,
        )
        ax.scatter(
            ordered["snapshot_at"],
            ordered["line"],
            s=18,
            zorder=3,
        )

    if reversals is not None and not reversals.empty:
        marks = reversals.loc[reversals["reversed"].fillna(False)]
        if not marks.empty:
            ax.scatter(
                marks["snapshot_at"],
                marks["new_line"],
                marker="x",
                s=80,
                c="black",
                linewidths=2,
                zorder=4,
                label="reversal",
            )

    ax.set_xlabel("Snapshot time")
    ax.set_ylabel("Line")
    ax.set_title(title or "Line path by book")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_price_steps(
    path: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Step chart: time on X, American price on Y, one series per book."""
    ax = _axes(ax)
    if path.empty:
        ax.text(0.5, 0.5, "No price path", ha="center", va="center")
        ax.set_axis_off()
        return ax

    for book, group in path.groupby("book", sort=True):
        ordered = group.sort_values("snapshot_at")
        ax.step(
            ordered["snapshot_at"],
            ordered["american_price"],
            where="post",
            label=str(book),
            linewidth=2,
        )
        ax.scatter(
            ordered["snapshot_at"],
            ordered["american_price"],
            s=18,
            zorder=3,
        )

    ax.set_xlabel("Snapshot time")
    ax.set_ylabel("American price")
    ax.set_title(title or "Price path at fixed line")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def assign_move_waves(
    events: pd.DataFrame,
    *,
    gap: timedelta = _DEFAULT_WAVE_GAP,
) -> pd.DataFrame:
    """Label moves that fall within ``gap`` of each other as the same wave."""
    if events.empty:
        out = events.copy()
        out["wave_id"] = pd.Series(dtype="int64")
        return out

    if "snapshot_at" not in events.columns:
        raise ValueError("events missing required column: snapshot_at")

    out = events.copy()
    out["snapshot_at"] = pd.to_datetime(out["snapshot_at"], errors="coerce")
    out = out.loc[out["snapshot_at"].notna()].sort_values(
        "snapshot_at", kind="stable"
    )
    if out.empty:
        out["wave_id"] = pd.Series(dtype="int64")
        return out

    times = out["snapshot_at"].to_numpy()
    wave_ids = [0]
    current = 0
    for index in range(1, len(times)):
        delta = pd.Timestamp(times[index]) - pd.Timestamp(times[index - 1])
        if delta > gap:
            current += 1
        wave_ids.append(current)
    out["wave_id"] = wave_ids
    return out.reset_index(drop=True)


def summarize_timeline(events: pd.DataFrame) -> dict[str, Any]:
    """Describe whether moves cluster, arrive in waves, or stay separate."""
    if events.empty or "wave_id" not in events.columns:
        return {
            "n_moves": 0,
            "n_waves": 0,
            "multi_book_waves": 0,
            "solo_waves": 0,
            "label": "No moves",
        }

    waved = events.dropna(subset=["wave_id"]).copy()
    if waved.empty:
        return {
            "n_moves": 0,
            "n_waves": 0,
            "multi_book_waves": 0,
            "solo_waves": 0,
            "label": "No moves",
        }

    per_wave = waved.groupby("wave_id", sort=True).agg(
        n_moves=("book", "size"),
        n_books=("book", "nunique"),
    )
    multi = int((per_wave["n_books"] >= 2).sum())
    solo = int((per_wave["n_books"] == 1).sum())
    n_waves = int(len(per_wave))

    if n_waves == 1 and multi >= 1:
        label = "Single cluster — books moved together"
    elif n_waves == 1 and solo >= 1:
        label = "Single book only — no cross-book cluster"
    elif multi >= 2:
        label = f"Waves — {multi} multi-book clusters, {solo} solo"
    elif multi == 1 and solo >= 1:
        label = f"Mixed — one multi-book cluster plus {solo} separate move(s)"
    else:
        label = f"Separate — {solo} solo move groups, no multi-book cluster"

    return {
        "n_moves": int(len(waved)),
        "n_waves": n_waves,
        "multi_book_waves": multi,
        "solo_waves": solo,
        "label": label,
    }


def plot_movement_timeline(
    events: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    gap: timedelta = _DEFAULT_WAVE_GAP,
) -> Axes:
    """One row per book; dots at change times, colored by direction; waves shaded."""
    ax = _axes(ax)
    if events.empty:
        ax.text(0.5, 0.5, "No movements", ha="center", va="center")
        ax.set_axis_off()
        return ax

    required = {"book", "snapshot_at", "direction"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"events missing columns: {', '.join(missing)}")

    waved = (
        events
        if "wave_id" in events.columns
        else assign_move_waves(events, gap=gap)
    )
    waved = waved.copy()
    waved["snapshot_at"] = pd.to_datetime(waved["snapshot_at"], errors="coerce")
    waved = waved.loc[waved["snapshot_at"].notna()]
    if waved.empty:
        ax.text(0.5, 0.5, "No movements", ha="center", va="center")
        ax.set_axis_off()
        return ax

    books = sorted(waved["book"].astype(str).unique())
    y_pos = {book: index for index, book in enumerate(books)}
    waved["y"] = waved["book"].astype(str).map(y_pos)

    # Shade each wave band.
    for wave_id, group in waved.groupby("wave_id", sort=True):
        start = group["snapshot_at"].min()
        end = group["snapshot_at"].max()
        pad = pd.Timedelta(minutes=1)
        ax.axvspan(
            start - pad,
            end + pad,
            color="#90a4ae",
            alpha=0.12 if group["book"].nunique() == 1 else 0.22,
            zorder=0,
        )
        ax.text(
            start,
            len(books) - 0.35,
            f"w{int(wave_id)}",
            fontsize=7,
            color="#546e7a",
            ha="left",
            va="top",
        )

    for direction, color in _DIRECTION_COLORS.items():
        subset = waved.loc[waved["direction"].astype(str).eq(direction)]
        if subset.empty:
            continue
        ax.scatter(
            subset["snapshot_at"],
            subset["y"],
            c=color,
            s=55,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
            label=direction,
        )

    # Unknown directions still plotted in gray.
    known = set(_DIRECTION_COLORS)
    other = waved.loc[~waved["direction"].astype(str).isin(known)]
    if not other.empty:
        ax.scatter(
            other["snapshot_at"],
            other["y"],
            c="#9e9e9e",
            s=55,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
            label="other",
        )

    ax.set_yticks(list(y_pos.values()), list(y_pos.keys()))
    ax.set_ylim(-0.75, len(books) - 0.25)
    ax.set_xlabel("Snapshot time")
    ax.set_ylabel("Book")
    summary = summarize_timeline(waved)
    ax.set_title(title or f"Movement timeline — {summary['label']}")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=name,
        )
        for name, color in _DIRECTION_COLORS.items()
        if name in set(waved["direction"].astype(str))
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_leader_follower_heatmap(
    pairwise: pd.DataFrame,
    *,
    ax: Axes | None = None,
    value_col: str = "median_lag_windowed_sec",
    title: str | None = None,
) -> Axes:
    """Rows = first mover, columns = responding book, color = median delay (sec)."""
    ax = _axes(ax)
    required = {"leader_book", "follower_book", value_col}
    if pairwise.empty or not required.issubset(pairwise.columns):
        ax.text(0.5, 0.5, "No pairwise lag data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    matrix = pairwise.pivot_table(
        index="leader_book",
        columns="follower_book",
        values=value_col,
        aggfunc="median",
    )
    if matrix.empty:
        ax.text(0.5, 0.5, "No pairwise lag data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    # Stable book order on both axes.
    books = sorted(set(matrix.index).union(matrix.columns))
    matrix = matrix.reindex(index=books, columns=books)

    heat = sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Median delay (sec)"},
        linewidths=0.4,
        linecolor="white",
    )
    heat.set(
        title=title
        or "Leader → follower median delay (who moves after whom)",
        xlabel="Responding book (follower)",
        ylabel="First mover (leader)",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return ax


def follower_rankings(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Books ranked by how often they respond after another book leads."""
    empty = pd.DataFrame(
        columns=[
            "follower_book",
            "follow_count",
            "median_lag_sec",
            "leaders_followed",
        ]
    )
    if pairwise.empty or "follower_book" not in pairwise.columns:
        return empty

    frame = pairwise.copy()
    count_col = (
        "follow_count_windowed"
        if "follow_count_windowed" in frame.columns
        else None
    )
    if count_col is None:
        frame["_count"] = 1
        count_col = "_count"

    grouped = frame.groupby("follower_book", as_index=False).agg(
        follow_count=(count_col, "sum"),
        leaders_followed=("leader_book", "nunique"),
    )
    if "median_lag_windowed_sec" in frame.columns:
        lags = (
            frame.groupby("follower_book", as_index=False)["median_lag_windowed_sec"]
            .median()
            .rename(columns={"median_lag_windowed_sec": "median_lag_sec"})
        )
        grouped = grouped.merge(lags, on="follower_book", how="left")
    else:
        grouped["median_lag_sec"] = pd.NA

    return grouped.sort_values(
        ["follow_count", "median_lag_sec"], ascending=[False, True]
    ).reset_index(drop=True)
