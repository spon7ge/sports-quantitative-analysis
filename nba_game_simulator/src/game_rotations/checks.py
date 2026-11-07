"""Frame, date, and player-identifier checks."""
from __future__ import annotations

import warnings
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

def _player_key(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("player identifier cannot be boolean")

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("numeric player identifier must be a finite integer")
        return str(int(value))

    if not isinstance(value, str):
        raise ValueError("player identifier must be a string or integer")

    value = value.strip()

    if not value or value.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
        "inf",
        "+inf",
        "-inf",
    }:
        raise ValueError("missing or invalid player identifier")

    try:
        numeric = Decimal(value)
    except InvalidOperation:
        return value

    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError("numeric player identifier must be a finite integer")

    return str(int(numeric))


def _normalize_keys(frame, keys):
    _required(frame, keys)

    for key in keys:
        if key == "game_id":
            values = frame[key]

            if values.isna().any():
                raise ValueError("game_id cannot be missing")

            if not all(isinstance(v, str) for v in values):
                warnings.warn(
                    "non-string game IDs may already have lost leading zeros",
                    stacklevel=2,
                )

            frame[key] = [
                str(int(v))
                if isinstance(v, (float, np.floating))
                and np.isfinite(v)
                and float(v).is_integer()
                else str(v).strip()
                for v in values
            ]

            if frame[key].isin(["", "nan", "None", "inf", "-inf"]).any():
                raise ValueError("invalid game_id")
        else:
            frame[key] = [_player_key(v) for v in frame[key]]

def _number(value, name, *, positive=False, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if (
        not np.isfinite(result)
        or (positive and result <= 0)
        or (nonnegative and result < 0)
    ):
        raise ValueError(f"invalid {name}")

    return result


def _vector(value, n, name):
    result = np.asarray(value, dtype=float).reshape(-1)

    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {n} finite values")

    return result


def _binary(values, name):
    result = _as_float(pd.Series(values))

    if not np.isfinite(result).all() or ((result != 0) & (result != 1)).any():
        raise ValueError(f"{name} must contain only 0 or 1")

    return result == 1


def _as_float(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)


def _as_player_matrix(values, shape):
    values = np.asarray(values, dtype=float)

    if values.shape == (shape[1],) or values.shape == (1, shape[1]):
        return np.broadcast_to(values, shape).copy()

    if values.shape == shape:
        return values.copy()

    raise ValueError("player matrix has the wrong shape")


def _dates(values):
    dates = pd.to_datetime(values, errors="coerce")

    if dates.isna().any():
        raise ValueError("game dates must be valid")

    if dates.dt.tz is not None:
        raise ValueError("use timezone-free NBA game dates")

    return dates.dt.normalize()


def _one_game(frame, *, cutoff=None):
    """Supplied identifiers and dates must name one game."""
    if "game_date" in frame:
        dates = _dates(frame["game_date"])

        if dates.nunique() != 1:
            raise ValueError("one call must describe a single game date")

        if cutoff and dates.le(pd.Timestamp(cutoff)).any():
            raise ValueError("game_date must follow the model's training cutoff")

    for column in ("game_id", "gameId"):
        if column in frame and frame[column].nunique(dropna=False) != 1:
            raise ValueError("one call must describe a single game")


def _one_team(frame):
    if "team_id" not in frame:
        return

    teams = {_player_key(value) for value in frame["team_id"]}

    if len(teams) != 1:
        raise ValueError("a rotation must contain exactly one team")


def _frame(value, name):
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")


def _required(frame, columns):
    if not frame.columns.is_unique:
        raise ValueError("DataFrame column names must be unique")

    missing = [column for column in columns if column not in frame]

    if missing:
        raise ValueError(f"missing columns: {missing}")


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
