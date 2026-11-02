"""Missingness, outliers, and season coverage for the selected features."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def feature_missingness(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    miss = pd.DataFrame(
        {
            "n_missing": frame[features].isna().sum(),
            "pct_missing": (frame[features].isna().mean() * 100).round(2),
        }
    ).sort_values(["pct_missing", "n_missing"], ascending=False)
    return miss


def missingness_summary(frame: pd.DataFrame, features: list[str]) -> str:
    miss = feature_missingness(frame, features)
    n_any = int(frame[features].isna().any(axis=1).sum())
    n_features = int((miss["n_missing"] > 0).sum())
    return "\n".join(
        [
            f"Appearance rows: {len(frame):,}",
            f"Features with a null: {n_features} / {len(features)}",
            f"Rows with any null feature: {n_any:,} ({n_any / len(frame):.1%})",
        ]
    )


def season_missingness(
    frame: pd.DataFrame,
    miss: pd.DataFrame,
) -> pd.DataFrame:
    sparse = miss.index[miss["n_missing"] > 0].tolist()
    if not sparse:
        return pd.DataFrame()
    return (
        frame[sparse]
        .isna()
        .groupby(frame["season_year"], sort=True)
        .mean()
        .mul(100)
        .round(1)
    )


def plot_missingness(miss: pd.DataFrame):
    sparse = miss.index[miss["n_missing"] > 0]
    if len(sparse) == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(sparse))))
    miss.loc[sparse, "pct_missing"].sort_values().plot.barh(ax=ax, color="C0")
    ax.set_xlabel("% missing")
    ax.set_title("Feature missingness on appearances")
    fig.tight_layout()
    return fig


def rate_outlier_summary(
    frame: pd.DataFrame,
    target_col: str,
    cap: float | None = None,
    *,
    unit: str = "points per minute",
) -> dict[str, float | int | str | None]:
    """Tukey fence for the label, plus a cap count when a cap is set."""
    rate = frame[target_col]
    q1, q3 = rate.quantile(0.25), rate.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_tukey = int(((rate < lo) | (rate > hi)).sum())
    return {
        "n": len(rate),
        "min": float(rate.min()),
        "p50": float(rate.median()),
        "max": float(rate.max()),
        "fence_low": float(lo),
        "fence_high": float(hi),
        "n_outside": n_tukey,
        "n_at_cap": 0 if cap is None else int((rate >= cap).sum()),
        "cap": None if cap is None else float(cap),
        "unit": unit,
    }


def format_rate_outliers(stats: dict[str, float | int | str | None]) -> str:
    n = int(stats["n"])
    outside = int(stats["n_outside"])
    lines = [
        (
            f"n={n:,}  min={stats['min']:.2f}  "
            f"p50={stats['p50']:.2f}  max={stats['max']:.2f}"
        ),
        (
            f"Tukey fence [{stats['fence_low']:.2f}, {stats['fence_high']:.2f}]  "
            f"outside={outside:,} ({outside / n:.1%})"
        ),
    ]
    if stats["cap"] is not None:
        unit = stats["unit"]
        if unit == "points per minute":
            unit = "points-per-minute"
        lines.append(
            f"Rows at the {float(stats['cap']):g} {unit} cap: "
            f"{int(stats['n_at_cap']):,}"
        )
    return "\n".join(lines)


def top_rate_rows(frame: pd.DataFrame, target_col: str, n: int = 8) -> pd.DataFrame:
    return (
        frame.nlargest(n, target_col)[
            ["season_year", "game_date", "player_name", target_col, "starting"]
        ].assign(game_date=lambda rows: pd.to_datetime(rows["game_date"]).dt.date)
    )


def plot_rate(
    frame: pd.DataFrame,
    target_col: str,
    upper_fence: float,
    *,
    unit: str = "points per minute",
):
    rate = frame[target_col]
    title = unit[:1].upper() + unit[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rate, bins=48, color="C0", edgecolor="white")
    axes[0].axvline(
        upper_fence,
        color="C3",
        ls="--",
        label=f"upper fence {upper_fence:.1f}",
    )
    axes[0].set_title(title)
    axes[0].set_xlabel(unit)
    axes[0].legend()
    sns.boxplot(x=rate, ax=axes[1], color="C0")
    axes[1].set_title(title)
    fig.tight_layout()
    return fig


def minutes_outlier_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    minutes = frame["minutes"]
    q1, q3 = minutes.quantile(0.25), minutes.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_tukey = int(((minutes < lo) | (minutes > hi)).sum())
    return {
        "n": len(minutes),
        "min": float(minutes.min()),
        "p50": float(minutes.median()),
        "max": float(minutes.max()),
        "fence_low": float(lo),
        "fence_high": float(hi),
        "n_outside": n_tukey,
        "n_above_48": int((minutes > 48).sum()),
    }


def format_minutes_outliers(stats: dict[str, float | int]) -> str:
    n = int(stats["n"])
    outside = int(stats["n_outside"])
    return "\n".join(
        [
            (
                f"n={n:,}  min={stats['min']:.2f}  "
                f"p50={stats['p50']:.2f}  max={stats['max']:.2f}"
            ),
            (
                f"Tukey fence [{stats['fence_low']:.2f}, {stats['fence_high']:.2f}]  "
                f"outside={outside:,} ({outside / n:.1%})"
            ),
            f"Player minutes above 48 (overtime range): {int(stats['n_above_48']):,}",
        ]
    )


def top_minute_rows(frame: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return (
        frame.nlargest(n, "minutes")[
            ["season_year", "game_date", "player_name", "minutes", "starting"]
        ]
        .assign(game_date=lambda rows: pd.to_datetime(rows["game_date"]).dt.date)
    )


def plot_minutes(frame: pd.DataFrame, upper_fence: float):
    minutes = frame["minutes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(minutes, bins=48, color="C0", edgecolor="white")
    axes[0].axvline(
        upper_fence,
        color="C3",
        ls="--",
        label=f"upper fence {upper_fence:.1f}",
    )
    axes[0].set_title("Minutes")
    axes[0].set_xlabel("minutes")
    axes[0].legend()
    sns.boxplot(x=minutes, ax=axes[1], color="C0")
    axes[1].set_title("Minutes")
    fig.tight_layout()
    return fig


def feature_outlier_table(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Tukey fences for non-binary selected features. Rows are not dropped."""
    rows = []
    for column in features:
        values = frame[column].dropna()
        if values.nunique(dropna=True) <= 2:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = float(q3 - q1)
        if spread <= 0:
            continue
        fence_lo = float(q1 - 1.5 * spread)
        fence_hi = float(q3 + 1.5 * spread)
        outside = (values < fence_lo) | (values > fence_hi)
        rows.append(
            {
                "feature": column,
                "n": int(values.shape[0]),
                "pct_outside": round(100 * float(outside.mean()), 2),
                "fence_low": round(fence_lo, 2),
                "fence_high": round(fence_hi, 2),
                "min": round(float(values.min()), 2),
                "p50": round(float(values.median()), 2),
                "max": round(float(values.max()), 2),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("pct_outside", ascending=False)
        .reset_index(drop=True)
    )


def season_coverage(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    complete = ~frame[features].isna().any(axis=1)
    coverage = frame.groupby("season_year", sort=True).agg(
        rows=("player_id", "size"),
        players=("player_id", "nunique"),
        games=("game_id", "nunique"),
        first_game=("game_date", "min"),
        last_game=("game_date", "max"),
    )
    coverage["first_game"] = pd.to_datetime(coverage["first_game"]).dt.date
    coverage["last_game"] = pd.to_datetime(coverage["last_game"]).dt.date
    coverage["pct_rows_complete"] = (
        complete.groupby(frame["season_year"], sort=True).mean().mul(100).round(1)
    )
    coverage["mean_missing_features"] = (
        frame[features]
        .isna()
        .sum(axis=1)
        .groupby(frame["season_year"], sort=True)
        .mean()
        .round(2)
    )
    return coverage


def plot_reliability(curves: Mapping[str, pd.DataFrame]):
    """Nominal tau against jittered coverage. The diagonal is perfect calibration.

    Each frame is a ``calibration_table`` result with ``tau`` and ``coverage``.
    The shaded band is tau 0.3–0.7, the range the line-gap check uses.
    """
    if not curves:
        raise ValueError("reliability curve needs at least one table")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.axvspan(0.3, 0.7, color="0.92", zorder=0)
    ax.plot([0, 1], [0, 1], color="0.45", ls="--", label="perfect calibration")
    for name, table in curves.items():
        tau, coverage = _reliability_xy(table)
        ax.plot(tau, coverage, marker="o", label=str(name))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _reliability_xy(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("reliability curve expects a calibration table")
    missing = [column for column in ("tau", "coverage") if column not in table.columns]
    if missing:
        raise ValueError("reliability table missing " + ", ".join(missing))
    if table.empty:
        raise ValueError("reliability table is empty")
    tau = table["tau"].to_numpy(dtype=float)
    coverage = table["coverage"].to_numpy(dtype=float)
    if not np.isfinite(tau).all() or not np.isfinite(coverage).all():
        raise ValueError("tau and coverage must be finite")
    if np.any((tau <= 0.0) | (tau >= 1.0)):
        raise ValueError("tau must be in (0, 1)")
    if np.any((coverage < 0.0) | (coverage > 1.0)):
        raise ValueError("coverage must be in [0, 1]")
    order = np.argsort(tau, kind="mergesort")
    return tau[order], coverage[order]
