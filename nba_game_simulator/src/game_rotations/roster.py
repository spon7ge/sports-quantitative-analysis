"""Roster identity and who can play in a scenario."""
from __future__ import annotations

import numpy as np

from game_rotations.checks import _binary, _player_key, _required

def _roster_keys(role):
    _required(role, ["player_id"])
    keys = [_player_key(value) for value in role["player_id"]]

    if len(set(keys)) != len(keys):
        raise ValueError("roster lists the same player twice")

    if "is_starter" in role:
        if _binary(role["is_starter"], "is_starter").sum() > 5:
            raise ValueError("a roster cannot list more than five starters")

    flags = _available_flags(role)

    return set(keys), {
        key
        for key, flag in zip(keys, flags, strict=True)
        if flag
    }


def _available_flags(role):
    return (
        np.ones(len(role), dtype=bool)
        if "available" not in role
        else _binary(role["available"], "available")
    )


def _availability_matrix(role, availability, n):
    base = np.broadcast_to(_available_flags(role), (n, len(role)))

    if availability is None:
        return base.copy()

    values = np.asarray(availability)

    if values.dtype != np.bool_ or values.shape != (n, len(role)):
        raise ValueError("availability must be bool[n_simulations, n_players]")

    if np.any(values & ~base):
        raise ValueError("scenario marks a roster-unavailable player as available")

    return values.copy()


def _active_shape(values, shape):
    if values is None:
        return np.ones(shape, dtype=bool)

    values = np.asarray(values)

    if values.dtype != np.bool_ or values.shape != shape:
        raise ValueError("availability must be a boolean matrix matching player draws")

    return values
