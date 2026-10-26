"""Missingness, outliers, and season coverage for the selected features."""

from __future__ import annotations

import matplotlib.pyplot as plt
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
    cap: float,
) -> dict[str, float | int]:
    """Tukey fence and cap count for the points-per-minute label."""
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
        "n_at_cap": int((rate >= cap).sum()),
        "cap": float(cap),
    }


def format_rate_outliers(stats: dict[str, float | int]) -> str:
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
            (
                f"Rows at the {float(stats['cap']):g} points-per-minute cap: "
                f"{int(stats['n_at_cap']):,}"
            ),
        ]
    )


def top_rate_rows(frame: pd.DataFrame, target_col: str, n: int = 8) -> pd.DataFrame:
    return (
        frame.nlargest(n, target_col)[
            ["season_year", "game_date", "player_name", target_col, "starting"]
        ].assign(game_date=lambda rows: pd.to_datetime(rows["game_date"]).dt.date)
    )


def plot_rate(frame: pd.DataFrame, target_col: str, upper_fence: float):
    rate = frame[target_col]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rate, bins=48, color="C0", edgecolor="white")
    axes[0].axvline(upper_fence, color="C3", ls="--", label=f"upper fence {upper_fence:.1f}")
    axes[0].set_title("Points per minute")
    axes[0].set_xlabel("points per minute")
    axes[0].legend()
    sns.boxplot(x=rate, ax=axes[1], color="C0")
    axes[1].set_title("Points per minute")
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
