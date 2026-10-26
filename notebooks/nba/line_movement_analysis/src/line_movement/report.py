"""Persist daily line-movement tables, quality metadata, and summary plots."""

from __future__ import annotations

import base64
import html
import importlib.util
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from line_movement.paths import analysis_root
from line_movement.validate import QualityReport


def _resolve_analysis_root(root: Path | None) -> Path:
    return root if root is not None else analysis_root()


def report_dir(date: str, *, analysis_root: Path | None = None) -> Path:
    """Return the daily report directory, creating it when necessary."""
    root = _resolve_analysis_root(analysis_root)
    destination = root / "reports" / date
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots()
    return created


def plot_coverage(quotes: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the number of normalized quote rows available from each book."""
    ax = _axes(ax)
    if quotes.empty or "book" not in quotes:
        ax.text(0.5, 0.5, "No quote coverage", ha="center", va="center")
        ax.set_axis_off()
        return ax

    counts = quotes["book"].dropna().astype(str).value_counts().sort_index()
    if counts.empty:
        ax.text(0.5, 0.5, "No quote coverage", ha="center", va="center")
        ax.set_axis_off()
        return ax

    counts.plot.bar(ax=ax)
    ax.set(title="Quote coverage by book", xlabel="Book", ylabel="Quotes")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_movements_over_time(
    movements: pd.DataFrame, ax: Axes | None = None
) -> Axes:
    """Plot movement counts at each observed snapshot time."""
    ax = _axes(ax)
    if movements.empty or "snapshot_at" not in movements:
        ax.text(0.5, 0.5, "No movements", ha="center", va="center")
        ax.set_axis_off()
        return ax

    snapshots = pd.to_datetime(movements["snapshot_at"], utc=True, errors="coerce")
    counts = snapshots.dropna().value_counts().sort_index()
    if counts.empty:
        ax.text(0.5, 0.5, "No movements", ha="center", va="center")
        ax.set_axis_off()
        return ax

    ax.plot(counts.index, counts.values, marker="o")
    ax.set(title="Movements over time", xlabel="Snapshot time", ylabel="Movements")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_pairwise_lag(pairwise: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot median windowed lag for every observed leader/follower pair."""
    ax = _axes(ax)
    required = {
        "leader_book",
        "follower_book",
        "median_lag_windowed_sec",
    }
    if pairwise.empty or not required.issubset(pairwise.columns):
        ax.text(0.5, 0.5, "No pairwise lag data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    matrix = pairwise.pivot_table(
        index="leader_book",
        columns="follower_book",
        values="median_lag_windowed_sec",
        aggfunc="median",
    )
    if matrix.empty:
        ax.text(0.5, 0.5, "No pairwise lag data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="viridis", ax=ax)
    ax.set(
        title="Median pairwise lag (seconds)",
        xlabel="Follower book",
        ylabel="Leader book",
    )
    return ax


def _parquet_engine() -> str | None:
    for engine in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(engine) is not None:
            return engine
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _cell_for_parquet(value: Any) -> Any:
    """Make object cells pyarrow-safe (tuple keys with floats + None break Arrow)."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, tuple):
        return str(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _parquet_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy frame with object columns normalized for parquet writers."""
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        out[column] = out[column].map(_cell_for_parquet)
    return out


def _try_save_parquet(frame: pd.DataFrame, path: Path, *, engine: str) -> None:
    try:
        _parquet_ready(frame).to_parquet(path, index=False, engine=engine)
    except Exception as exc:  # noqa: BLE001 — parquet is optional; any dtype/engine failure skips one file
        warnings.warn(
            f"Skipping parquet {path.name}: {exc}",
            stacklevel=2,
        )


def _save_plot(path: Path, plotter, frame: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(8, 5))
    try:
        plotter(frame, ax=ax)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _png_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _frame_to_html(frame: pd.DataFrame, *, max_rows: int = 50) -> str:
    if frame.empty:
        return "<p class=\"empty\">No rows.</p>"
    view = frame.head(max_rows)
    return view.to_html(
        index=False,
        border=0,
        classes="table",
        float_format=lambda value: f"{value:g}",
        na_rep="",
    )


def _missing_rates_table(quality: QualityReport) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for book, rates in sorted(quality.missing_rates_by_book.items()):
        row: dict[str, Any] = {"book": book}
        row.update(rates)
        rows.append(row)
    return pd.DataFrame(rows)


def _leader_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty or "leader_book" not in episodes.columns:
        return pd.DataFrame(columns=["leader_book", "episodes"])
    return (
        episodes.groupby("leader_book", as_index=False)
        .size()
        .rename(columns={"size": "episodes"})
        .sort_values("episodes", ascending=False)
    )


def write_html_report(
    *,
    date: str,
    destination: Path,
    quotes: pd.DataFrame,
    movements: pd.DataFrame,
    episodes: pd.DataFrame,
    pairwise: pd.DataFrame,
    quality: QualityReport,
) -> Path:
    """Write a self-contained HTML report with embedded chart images."""
    image_specs = (
        ("coverage_by_book.png", "Quote coverage by book"),
        ("movements_over_time.png", "Movements over time"),
        ("pairwise_lag_heatmap.png", "Median pairwise lag (seconds)"),
    )
    figures: list[str] = []
    for filename, title in image_specs:
        path = destination / filename
        if not path.is_file():
            continue
        figures.append(
            "<section class=\"figure\">"
            f"<h2>{html.escape(title)}</h2>"
            f'<img src="{_png_data_uri(path)}" alt="{html.escape(title)}"/>'
            "</section>"
        )

    pairwise_view = pairwise
    if not pairwise.empty and "follow_count_windowed" in pairwise.columns:
        pairwise_view = pairwise.sort_values(
            "follow_count_windowed", ascending=False
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Line movement report — {html.escape(date)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #15202b;
      --muted: #5b6770;
      --line: #d7dde2;
      --bg: #f6f8fa;
      --card: #ffffff;
    }}
    body {{
      margin: 0;
      font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    h1, h2 {{ margin: 0 0 0.75rem; font-weight: 650; }}
    h1 {{ font-size: 1.75rem; }}
    h2 {{ font-size: 1.15rem; }}
    p.meta, p.note {{ color: var(--muted); }}
    p.note {{
      border-left: 3px solid var(--line);
      padding-left: 0.75rem;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      margin: 1.25rem 0 1.75rem;
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.85rem 1rem;
    }}
    .stat strong {{ display: block; font-size: 1.35rem; }}
    .stat span {{ color: var(--muted); font-size: 0.85rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1rem 1.1rem 1.2rem;
      margin-bottom: 1rem;
    }}
    section.figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
      background: #fff;
    }}
    table.table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    table.table th, table.table td {{
      text-align: left;
      padding: 0.45rem 0.55rem;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    table.table th {{ color: var(--muted); font-weight: 600; }}
    p.empty {{ color: var(--muted); margin: 0; }}
  </style>
</head>
<body>
  <main>
    <h1>Line movement report — {html.escape(date)}</h1>
    <p class="meta">
      Quotes {len(quotes):,} · Movements {len(movements):,} ·
      Episodes {len(episodes):,} · Pairwise rows {len(pairwise):,}
    </p>
    <p class="note">
      Scrapes land about every 15 minutes. First-observed in this report is not
      proof of true market leadership — only scrape-clock order.
    </p>
    <div class="stats">
      <div class="stat"><strong>{len(quotes):,}</strong><span>Normalized quotes</span></div>
      <div class="stat"><strong>{len(movements):,}</strong><span>Detected movements</span></div>
      <div class="stat"><strong>{len(episodes):,}</strong><span>Episodes</span></div>
      <div class="stat"><strong>{quality.dropped_by_schedule:,}</strong><span>Dropped by schedule</span></div>
      <div class="stat"><strong>{quality.duplicate_count:,}</strong><span>Duplicate quote keys</span></div>
    </div>
    {"".join(figures)}
    <section>
      <h2>Pairwise leader → follower stats</h2>
      {_frame_to_html(pairwise_view)}
    </section>
    <section>
      <h2>Episode leads by book</h2>
      {_frame_to_html(_leader_summary(episodes))}
    </section>
    <section>
      <h2>Missing-field rates by book</h2>
      {_frame_to_html(_missing_rates_table(quality))}
    </section>
  </main>
</body>
</html>
"""
    path = destination / "report.html"
    path.write_text(body, encoding="utf-8")
    return path


def save_daily_report(
    *,
    date: str,
    quotes: pd.DataFrame,
    groups: pd.DataFrame,
    movements: pd.DataFrame,
    episodes: pd.DataFrame,
    follows: pd.DataFrame,
    pairwise: pd.DataFrame,
    quality: QualityReport,
    analysis_root: Path | None = None,
) -> Path:
    """Write all daily analysis artifacts and return their directory."""
    destination = report_dir(date, analysis_root=analysis_root)
    tables = {
        "quotes_normalized": quotes,
        "market_groups": groups,
        "movements": movements,
        "episodes": episodes,
        "follows": follows,
        "pairwise_stats": pairwise,
    }
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)

    parquet_engine = _parquet_engine()
    if parquet_engine is not None:
        for name, frame in tables.items():
            _try_save_parquet(
                frame,
                destination / f"{name}.parquet",
                engine=parquet_engine,
            )

    with (destination / "quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(
            asdict(quality),
            file,
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    _save_plot(destination / "coverage_by_book.png", plot_coverage, quotes)
    _save_plot(
        destination / "movements_over_time.png",
        plot_movements_over_time,
        movements,
    )
    _save_plot(
        destination / "pairwise_lag_heatmap.png",
        plot_pairwise_lag,
        pairwise,
    )
    write_html_report(
        date=date,
        destination=destination,
        quotes=quotes,
        movements=movements,
        episodes=episodes,
        pairwise=pairwise,
        quality=quality,
    )
    return destination
