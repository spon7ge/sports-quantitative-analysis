"""Expanding chronological backtest folds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig


def chronological_folds(
    dates,
    config: MlbConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, test_idx)`` pairs from ``config.folds``.

    ``dates`` is array-like of game_date strings (``YYYY-MM-DD``). Train rows
    satisfy ``date <= train_end``; test rows satisfy
    ``test_start <= date <= test_end``. Indices are not shuffled.
    """
    series = pd.Series(dates)
    stamps = pd.to_datetime(series, errors="coerce", utc=True)
    labels = stamps.dt.strftime("%Y-%m-%d")

    index = np.arange(len(labels), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for window in config.folds:
        train_mask = labels.astype("string") <= window.train_end
        test_mask = (labels.astype("string") >= window.test_start) & (
            labels.astype("string") <= window.test_end
        )
        train_idx = index[train_mask.fillna(False).to_numpy()]
        test_idx = index[test_mask.fillna(False).to_numpy()]
        folds.append((train_idx, test_idx))
    return folds
