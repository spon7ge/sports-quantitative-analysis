"""Identifiers, frames, minutes, and availability checks."""

from __future__ import annotations

import re
import warnings
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from game_opportunities.constants import _ALIASES, _TOL


def as_nonneg_int_vector(
    value: Any,
    n_sims: int,
    name: str,
) -> np.ndarray:
    _positive_int(n_sims, "n_sims")

    arr = np.asarray(value)
    scalar = arr.ndim == 0
    flat = arr.reshape(-1)

    if not scalar and flat.shape != (n_sims,):
        raise ValueError(f"{name} must be scalar or length {n_sims}")

    limit = np.iinfo(np.int64).max
    integers = []

    for item in flat:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} cannot contain booleans")

        if isinstance(item, (int, np.integer)):
            number = int(item)

        elif isinstance(item, str):
            try:
                parsed = Decimal(item)
            except InvalidOperation as exc:
                raise ValueError(f"{name} must contain nonnegative integers") from exc

            if not parsed.is_finite() or parsed != parsed.to_integral_value():
                raise ValueError(f"{name} must contain nonnegative integers")

            number = int(parsed)

        else:
            numeric = _number(
                item,
                name,
                nonnegative=True,
            )

            if numeric != np.floor(numeric) or numeric >= 2**63:
                raise ValueError(f"{name} must contain int64 integers")

            number = int(numeric)

        if number < 0 or number > limit:
            raise ValueError(f"{name} is outside nonnegative int64 range")

        integers.append(number)

    result = np.array(integers, dtype=np.int64)

    if scalar:
        return np.full(
            n_sims,
            result[0],
            dtype=np.int64,
        )

    return result


def _normalize_rows(weights):
    values = np.asarray(weights, dtype=float)

    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("allocation weights must be finite nonnegative matrices")

    total = values.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        values,
        total,
        out=np.zeros_like(values),
        where=total > 0,
    )


def _integer_matrix(value, name):
    values = np.asarray(value)

    if values.ndim != 2 or not values.size:
        raise ValueError(f"{name} must be a nonempty matrix")

    return as_nonneg_int_vector(
        values.reshape(-1),
        values.size,
        name,
    ).reshape(values.shape)


def _check_minutes(
    minutes,
    available,
    game_minutes,
    *,
    strict_duration,
):
    if not np.isfinite(minutes).all() or (minutes < 0).any():
        raise ValueError("minutes must be finite and nonnegative")

    if (minutes[~available] != 0).any():
        raise ValueError("unavailable players must have zero minutes")

    if (minutes.sum(axis=1) <= 0).any():
        raise ValueError("every row needs positive minute exposure")

    if game_minutes is None and not strict_duration:
        return

    duration = (
        minutes.sum(axis=1) / 5
        if game_minutes is None
        else _float_vector(
            game_minutes,
            len(minutes),
            "game_minutes",
            nonnegative=True,
        )
    )

    if strict_duration:
        overtime = (duration - 48) / 5

        if (overtime < -_TOL).any() or (
            np.abs(overtime - np.rint(overtime)) > _TOL
        ).any():
            raise ValueError(
                "minute budget must describe 48 minutes plus whole overtime periods"
            )

    if (available.sum(axis=1) < 5).any() or (minutes > duration[:, None] + _TOL).any():
        raise ValueError("minutes exceed active roster capacity or game duration")

    if not np.allclose(
        minutes.sum(axis=1),
        5 * duration,
        rtol=0,
        atol=_TOL,
    ):
        raise ValueError("minute budget does not equal five times game duration")


def _availability(value, n, k):
    if value is None:
        return np.ones(
            (n, k),
            dtype=bool,
        )

    arr = np.asarray(value)

    if arr.shape == (k,):
        arr = np.broadcast_to(arr, (n, k))

    if arr.shape != (n, k):
        raise ValueError("availability must be (players,) or (simulations, players)")

    numeric = np.asarray(arr, dtype=float)

    if not np.isfinite(numeric).all() or ((numeric != 0) & (numeric != 1)).any():
        raise ValueError("availability must contain only boolean or 0/1 values")

    return numeric == 1


def _player_vector(
    value,
    k,
    name,
    *,
    probability=False,
):
    values = np.asarray(value, dtype=float)

    if values.shape != (k,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be a finite nonnegative player vector")

    if probability and (values > 1).any():
        raise ValueError(f"{name} must be in [0, 1]")

    return values


def _matrix(value, n, k, name):
    values = np.asarray(value, dtype=float)

    if values.shape == (k,):
        values = np.broadcast_to(
            values,
            (n, k),
        ).copy()

    if values.shape != (n, k):
        raise ValueError(f"{name} must be (players,) or (simulations, players)")

    return values


def _float_vector(
    value,
    n,
    name,
    *,
    nonnegative=False,
):
    values = np.asarray(value, dtype=float)

    if values.ndim == 0:
        values = np.full(n, float(values))

    if (
        values.shape != (n,)
        or not np.isfinite(values).all()
        or (nonnegative and (values < 0).any())
    ):
        raise ValueError(f"{name} must contain {n} finite valid values")

    return values


def _aliases(frame):
    out = frame.copy()

    if not out.columns.is_unique:
        raise ValueError("column names must be unique")

    for canonical, aliases in _ALIASES.items():
        if canonical in out:
            continue

        matches = [name for name in aliases if name in out]

        if len(matches) > 1:
            raise ValueError(
                f"multiple aliases for {canonical}; provide one canonical column"
            )

        if matches:
            out = out.rename(columns={matches[0]: canonical})

    return out


def _minute(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    text = str(value).strip()

    if ":" in text:
        parts = text.split(":")

        if len(parts) != 2:
            raise ValueError("minutes must be numeric or MM:SS")

        whole = float(parts[0])
        seconds = float(parts[1])

        if (
            not np.isfinite(whole)
            or not np.isfinite(seconds)
            or whole < 0
            or seconds < 0
            or seconds >= 60
            or whole != np.floor(whole)
        ):
            raise ValueError("invalid MM:SS minutes")

        return whole + seconds / 60

    if text.startswith("PT"):
        match = re.fullmatch(
            r"PT(?:(\d+(?:\.\d+)?)M)?"
            r"(?:(\d+(?:\.\d+)?)S)?",
            text,
        )

        if match is None or all(value is None for value in match.groups()):
            raise ValueError("invalid ISO minute duration")

        return float(match.group(1) or 0) + float(match.group(2) or 0) / 60

    return _number(
        value,
        "minutes",
        nonnegative=True,
    )


def _nonnegative(values, name):
    result = pd.to_numeric(
        values,
        errors="coerce",
    )

    present = values.notna() & values.astype("string").str.strip().ne("")

    if (
        (present & result.isna()).any()
        or np.isinf(
            result.to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        ).any()
        or (result < 0).fillna(False).any()
    ):
        raise ValueError(
            f"{name} must contain finite nonnegative numbers or missing values"
        )

    return result


def _key(value):
    if isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA:
        raise ValueError("invalid player/team identifier")

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        raise ValueError("missing player/team identifier")

    try:
        number = Decimal(text)
    except InvalidOperation:
        return text

    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("numeric identifier must be a finite integer")

    return str(int(number))


def _game_key(value):
    if isinstance(value, str):
        text = value.strip()

        if not text or not text.isdecimal():
            raise ValueError("game IDs must contain digits")

        return text

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("game ID cannot be boolean")

    numeric = _number(
        value,
        "game_id",
        nonnegative=True,
    )

    if numeric != np.floor(numeric):
        raise ValueError("game ID must be a whole number")

    warnings.warn(
        "numeric game IDs may already have lost leading zeros",
        stacklevel=2,
    )

    return str(int(numeric))


def _dates(values):
    dates = pd.to_datetime(
        values,
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError("game dates must be valid")

    if dates.dt.tz is not None:
        raise ValueError("use timezone-free NBA game dates")

    return dates.dt.normalize()


def _number(
    value,
    name,
    *,
    nonnegative=False,
):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not np.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"invalid {name}")

    return result


def _rng(value):
    if not isinstance(value, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _frame(value, name):
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")

    if not value.columns.is_unique:
        raise ValueError("column names must be unique")


def _required(frame, names):
    missing = [name for name in names if name not in frame]

    if missing:
        raise ValueError(f"missing columns: {missing}")
