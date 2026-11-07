"""Boundary checks for counts, probabilities, and identifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _invariant(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Invariant violated: {message}")


def _as_count_matrix(value: Any, n_sims: int, n_players: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer matrix") from exc
    if arr.shape != (n_sims, n_players):
        raise ValueError(f"{name} must have shape ({n_sims}, {n_players})")
    if (
        np.any(~np.isfinite(arr))
        or np.any(arr < 0)
        or np.any(arr != np.floor(arr))
        or np.any(arr > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must be a nonnegative integer matrix")
    return arr.astype(np.int64)


def _as_probability(value: Any, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1]") from exc
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _as_unit_interval_vector(value: Any, n_players: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain values in [0, 1]") from exc
    if (
        arr.shape != (n_players,)
        or np.any(~np.isfinite(arr))
        or np.any((arr < 0.0) | (arr > 1.0))
    ):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return arr


def _as_factor(value: Any, n_sims: int, n_players: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain nonnegative finite values") from exc
    if arr.shape not in ((n_players,), (n_sims, n_players)):
        raise ValueError(
            f"{name} must have shape ({n_players},) or ({n_sims}, {n_players})"
        )
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must contain nonnegative finite values")
    return np.broadcast_to(arr, (n_sims, n_players))


def _as_context(value: Any, n_players: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "context_factors must contain nonnegative finite values"
        ) from exc
    if arr.ndim != 1 or arr.shape != (n_players,):
        raise ValueError(
            "context_factors must be a one-dimensional player-aligned array"
        )
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("context_factors must contain nonnegative finite values")
    return arr


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _normalize_id(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("IDs cannot be booleans")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        raise ValueError(f"ID {value!r} is not an integer")
    return str(value).strip()


def _normalize_ids(values: Any, name: str) -> pd.Series:
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError(f"{name} has missing values")
    out = series.map(_normalize_id)
    if out.eq("").any():
        raise ValueError(f"{name} has blank values")
    return out
