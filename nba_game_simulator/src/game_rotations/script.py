"""Attached margin, overtime count, and game duration."""
from __future__ import annotations

import numpy as np
from role_context import BLOWOUT_MARGIN, OT_MIN, REG_MIN

from game_rotations.checks import _as_float, _number, _required, _vector
from game_rotations.constants import CONDITIONAL, _TOL

def _period_lengths(game_minutes, n_ot, regulation_minutes, ot_minutes):
    reg = _number(regulation_minutes, "regulation_minutes", positive=True)
    overtime = _number(ot_minutes, "ot_minutes", positive=True)
    count = _whole_ot(n_ot)
    duration = _number(game_minutes, "game_minutes", positive=True)

    if not np.isclose(duration, reg + count * overtime, rtol=0, atol=_TOL):
        raise ValueError("game minutes do not match overtime count")

    return [(i, reg / 4) for i in range(1, 5)] + [
        (5 + i, overtime)
        for i in range(count)
    ]


def _whole_ot(n_ot):
    value = _number(n_ot, "n_ot", nonnegative=True)

    if abs(value - round(value)) > _TOL:
        raise ValueError("n_ot must be a nonnegative whole number")

    return int(round(value))


def _script_arrays(
    n,
    n_ot,
    margin,
    team_total,
    cap,
    *,
    regulation_minutes=REG_MIN,
    ot_minutes=OT_MIN,
):
    if any(value is None for value in (margin, team_total, cap)):
        raise ValueError("supply attached margin, team budget and game duration")

    reg = _number(regulation_minutes, "regulation_minutes", positive=True)
    overtime = _number(ot_minutes, "ot_minutes", positive=True)

    margin, total, cap = (
        _vector(v, n, name)
        for v, name in (
            (margin, "margin"),
            (team_total, "team_total"),
            (cap, "cap"),
        )
    )

    counts = (
        (cap - reg) / overtime
        if n_ot is None
        else _vector(n_ot, n, "n_ot")
    )

    if (counts < 0).any() or (np.abs(counts - np.rint(counts)) > _TOL).any():
        raise ValueError("n_ot must contain nonnegative whole numbers")

    counts = np.rint(counts)

    if not np.allclose(cap, reg + counts * overtime, rtol=0, atol=_TOL):
        raise ValueError("game duration and overtime count disagree")

    if not np.allclose(total, 5 * cap, rtol=0, atol=_TOL):
        raise ValueError("team minute budget must equal five times game duration")

    return counts, counts > 0, margin, total, cap


def _attach_script(
    frame,
    *,
    validate_existing=False,
    regulation_minutes=REG_MIN,
    ot_minutes=OT_MIN,
):
    _required(frame, ["team_final_margin", "game_minutes", "n_ot"])
    margin = _as_float(frame["team_final_margin"])

    n_ot, ot, _, _, cap = _script_arrays(
        len(frame),
        _as_float(frame["n_ot"]),
        margin,
        5 * _as_float(frame["game_minutes"]),
        _as_float(frame["game_minutes"]),
        regulation_minutes=regulation_minutes,
        ot_minutes=ot_minutes,
    )

    derived = {
        "team_final_margin": margin,
        "abs_final_margin": np.abs(margin),
        "n_ot": n_ot,
        "went_ot": ot.astype(float),
        "game_minutes": cap,
        "blowout": (np.abs(margin) >= BLOWOUT_MARGIN).astype(float),
    }

    for name, value in derived.items():
        if validate_existing and name in frame:
            old = _as_float(frame[name])

            if not np.isfinite(old).all() or not np.allclose(
                old,
                value,
                rtol=0,
                atol=_TOL,
            ):
                raise ValueError(f"historical {name} conflicts with the game state")

        frame[name] = value

    _required(frame, CONDITIONAL)
