from __future__ import annotations

# The Agg backend must be selected before importing pyplot or line_movement.report.
# ruff: noqa: I001

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from line_movement import (
    plot_coverage,
    plot_movements_over_time,
    plot_pairwise_lag,
    report,
    save_daily_report,
)
from line_movement.match import attach_market_keys
from line_movement.validate import QualityReport


def test_save_daily_report_writes_all_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "analysis_root", lambda: tmp_path)
    quotes = pd.DataFrame(
        {
            "book": ["fanduel", "draftkings"],
            "snapshot_at": pd.to_datetime(
                ["2026-10-24T19:00:00Z", "2026-10-24T19:05:00Z"]
            ),
        }
    )
    movements = pd.DataFrame(
        {
            "book": ["fanduel"],
            "snapshot_at": pd.to_datetime(["2026-10-24T19:10:00Z"]),
            "change_type": ["line"],
        }
    )
    pairwise = pd.DataFrame(
        {
            "leader_book": ["fanduel"],
            "follower_book": ["draftkings"],
            "median_lag_windowed_sec": [120.0],
        }
    )
    quality = QualityReport(
        missing_rates_by_book={"fanduel": {"line": 0.0}},
        duplicate_count=1,
        schema_flags={"missing_start_time": 0},
        dropped_by_schedule=2,
        snapshot_span_by_book={
            "fanduel": {
                "min": pd.Timestamp("2026-10-24T19:00:00Z"),
                "max": pd.Timestamp("2026-10-24T19:10:00Z"),
            }
        },
    )

    output = save_daily_report(
        date="2026-10-24",
        quotes=quotes,
        groups=pd.DataFrame({"market_group_id": [1]}),
        movements=movements,
        episodes=pd.DataFrame({"episode_id": [0]}),
        follows=pd.DataFrame({"episode_id": [0], "lag_seconds": [120.0]}),
        pairwise=pairwise,
        quality=quality,
    )

    expected = {
        "quotes_normalized.csv",
        "market_groups.csv",
        "movements.csv",
        "episodes.csv",
        "follows.csv",
        "pairwise_stats.csv",
        "quality_report.json",
        "coverage_by_book.png",
        "movements_over_time.png",
        "pairwise_lag_heatmap.png",
        "report.html",
    }
    assert output == tmp_path / "reports" / "2026-10-24"
    assert expected.issubset(path.name for path in output.iterdir())
    parsed = json.loads((output / "quality_report.json").read_text())
    assert parsed["duplicate_count"] == 1
    assert parsed["snapshot_span_by_book"]["fanduel"]["min"] == (
        "2026-10-24T19:00:00+00:00"
    )
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "Line movement report — 2026-10-24" in html
    assert "data:image/png;base64," in html
    assert "fanduel" in html
    assert "draftkings" in html
    assert "Median pairwise lag" in html or "pairwise" in html.lower()


def test_save_daily_report_with_tuple_key_columns_writes_all_artifacts(
    tmp_path, monkeypatch
):
    """Tuple/NaN market keys must not abort CSV/JSON/PNG, and parquet must succeed."""
    monkeypatch.setattr(report, "analysis_root", lambda: tmp_path)
    # Real attach_market_keys output: tuples with float line + None for incomplete rows.
    quotes = attach_market_keys(
        pd.DataFrame(
            [
                {
                    "book": "fanduel",
                    "player_norm": "luka doncic",
                    "event_key": "dallas mavericks__san antonio spurs",
                    "prop": "points",
                    "period": "game",
                    "side": "over",
                    "line": 28.5,
                    "snapshot_at": pd.Timestamp("2026-10-24T19:00:00Z"),
                },
                {
                    "book": "prizepicks",
                    "player_norm": "luka doncic",
                    "event_key": None,
                    "prop": "points",
                    "period": "game",
                    "side": None,
                    "line": 28.5,
                    "snapshot_at": pd.Timestamp("2026-10-24T19:05:00Z"),
                },
            ]
        )
    )
    quality = QualityReport(
        missing_rates_by_book={},
        duplicate_count=0,
        schema_flags={},
        dropped_by_schedule=0,
        snapshot_span_by_book={},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = save_daily_report(
            date="2026-10-24",
            quotes=quotes,
            groups=pd.DataFrame({"market_group_id": [1]}),
            movements=pd.DataFrame(),
            episodes=pd.DataFrame(),
            follows=pd.DataFrame(),
            pairwise=pd.DataFrame(),
            quality=quality,
        )

    required = {
        "quotes_normalized.csv",
        "quotes_normalized.parquet",
        "market_groups.csv",
        "movements.csv",
        "episodes.csv",
        "follows.csv",
        "pairwise_stats.csv",
        "quality_report.json",
        "coverage_by_book.png",
        "movements_over_time.png",
        "pairwise_lag_heatmap.png",
    }
    assert required.issubset(path.name for path in output.iterdir())
    assert "market_key" in (output / "quotes_normalized.csv").read_text()
    parquet = pd.read_parquet(output / "quotes_normalized.parquet")
    assert len(parquet) == 2
    assert parquet["market_key"].notna().sum() == 1
    assert not any("Skipping parquet quotes_normalized" in str(w.message) for w in caught)


def test_plot_helpers_return_axes_and_accept_empty_frames():
    figure, axes = plt.subplots(1, 3)

    assert plot_coverage(pd.DataFrame(), ax=axes[0]) is axes[0]
    assert plot_movements_over_time(pd.DataFrame(), ax=axes[1]) is axes[1]
    assert plot_pairwise_lag(pd.DataFrame(), ax=axes[2]) is axes[2]

    plt.close(figure)
