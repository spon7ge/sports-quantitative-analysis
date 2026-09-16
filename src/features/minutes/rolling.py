"""Shift-then-aggregate helpers for causal rolling features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    fallback: str | None = None,
) -> pd.Series:
    if column in frame:
        return pd.to_numeric(
            frame[column],
            errors="coerce",
        )
    if fallback is not None and fallback in frame:
        return pd.to_numeric(
            frame[fallback],
            errors="coerce",
        )
    return pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )


def group_keys(
    frame: pd.DataFrame,
    by: list[str],
) -> pd.Series:
    if len(by) == 1:
        return frame[by[0]]
    return pd.Series(
        list(zip(*(frame[column] for column in by))),
        index=frame.index,
    )


def prior_shift(
    frame: pd.DataFrame,
    column: str,
    by: list[str],
) -> pd.Series:
    return frame.groupby(by, sort=False)[column].shift(1)


def prior_roll(
    frame: pd.DataFrame,
    column: str,
    by: list[str],
    window: int,
    method: str,
    min_periods: int = 1,
) -> pd.Series:
    shifted = prior_shift(frame, column, by)
    rolled = getattr(
        shifted.groupby(
            group_keys(frame, by),
            sort=False,
        ).rolling(window, min_periods=min_periods),
        method,
    )()
    return _drop_group_level(rolled)


def prior_sum(
    frame: pd.DataFrame,
    column: str,
    by: list[str],
    window: int,
) -> pd.Series:
    return prior_roll(
        frame,
        column,
        by,
        window,
        "sum",
    )


def prior_ewm(
    frame: pd.DataFrame,
    column: str,
    by: list[str],
    *,
    halflife: float,
    min_periods: int = 1,
) -> pd.Series:
    shifted = prior_shift(frame, column, by)
    return shifted.groupby(
        group_keys(frame, by),
        sort=False,
    ).transform(
        lambda series: series.ewm(
            halflife=halflife,
            adjust=False,
            min_periods=min_periods,
        ).mean()
    )


def prior_expanding(
    frame: pd.DataFrame,
    column: str,
    by: list[str],
    method: str,
    min_periods: int = 1,
) -> pd.Series:
    shifted = prior_shift(frame, column, by)
    expanded = getattr(
        shifted.groupby(
            group_keys(frame, by),
            sort=False,
        ).expanding(min_periods=min_periods),
        method,
    )()
    return _drop_group_level(expanded)


def ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _drop_group_level(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.reset_index(
            level=0,
            drop=True,
        )
    return series
