"""Integer points PMF from a minutes draw and a rate draw.

One seeded RNG per player-game is handed to the copula. The uniforms
come back as a pair and each is pushed through that target's quantile
function. The samplers' own draw functions are not used: separate
draws are independent, and a shared hash draws the same uniforms twice.
"""

from __future__ import annotations

import math
from hashlib import sha256

import numpy as np
import pandas as pd

from models.shared.copula import minutes_tier
from models.shared.minutes_sampler import (
    QUANTILE_LEVELS,
    canonical_id,
    groups_for_frame as minutes_groups,
    ppf as minutes_ppf,
)
from models.shared.oos import RATE_CAP
from models.shared.ppm_sampler import (
    groups_for_frame as rate_groups,
    ppf as rate_ppf,
)


def points_pmf(
    rows: pd.DataFrame,
    copula,
    n_draws: int = 10_000,
    kmax: int = 90,
    *,
    minutes_tables,
    rate_tables,
    seed: int = 42,
):
    """PMF of ``k = clip(floor(minutes * rate + 0.5), 0, kmax)``.

    Rows in one predicted-minutes tier are mapped together. Returns
    ``(pmf, mean, median)`` aligned to ``rows``. The mean and the median
    are read from the PMF.
    """
    n_draws = _positive_draws(n_draws)
    kmax = _kmax(kmax)
    frame = rows.reset_index(drop=True)
    n_rows = len(frame)
    width = kmax + 1
    pmf = np.zeros((n_rows, width), dtype=float)
    if n_rows == 0:
        return pmf, np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    tiers = minutes_tier(frame["minutes_q_0.50"])
    minute_grids = _knot_grid(frame, "minutes")
    rate_grids = _knot_grid(frame, "rate")
    minute_lower, minute_upper = minutes_groups(frame["starting"], minutes_tables)
    rate_lower, rate_upper = rate_groups(frame["starting"], rate_tables)
    game_ids = frame["game_id"].to_numpy()
    player_ids = frame["player_id"].to_numpy()

    for tier in dict.fromkeys(tiers.tolist()):
        index = np.flatnonzero(tiers == tier)
        uniforms_m = np.empty((index.shape[0], n_draws), dtype=float)
        uniforms_r = np.empty((index.shape[0], n_draws), dtype=float)
        for local, row_index in enumerate(index):
            rng = _player_game_rng(game_ids[row_index], player_ids[row_index], seed)
            drawn = copula.sample(str(tier), n_draws, rng)
            uniforms_m[local] = _uniform_column(drawn, "u_m", n_draws)
            uniforms_r[local] = _uniform_column(drawn, "u_r", n_draws)
        minutes = minutes_ppf(
            uniforms_m,
            minute_grids[index],
            minute_lower[index],
            minute_upper[index],
            minutes_tables,
        )
        rate = np.minimum(
            rate_ppf(
                uniforms_r,
                rate_grids[index],
                rate_lower[index],
                rate_upper[index],
                rate_tables,
            ),
            RATE_CAP,
        )
        counts = np.clip(np.floor(minutes * rate + 0.5), 0, kmax).astype(np.int64)
        pmf[index] = _bincount_rows(counts, width) / float(n_draws)

    mean, median = _pmf_summaries(pmf)
    return pmf, mean, median


def line_probs(pmf, line: float):
    """``(over, push, under)`` for one points line.

    A half-point line has no push: over is the mass above ``floor(line)``.
    A whole-number line pushes on that integer.
    """
    mat = np.asarray(pmf, dtype=float)
    squeeze = mat.ndim == 1
    if squeeze:
        mat = mat.reshape(1, -1)
    if mat.ndim != 2:
        raise ValueError("pmf")
    level = float(line)
    if not math.isfinite(level):
        raise ValueError("line")
    width = mat.shape[1]
    if level.is_integer():
        over, push, under = _whole_line(mat, int(round(level)), width)
    else:
        over, push, under = _half_line(mat, int(math.floor(level)), width)
    if squeeze:
        return float(over[0]), float(push[0]), float(under[0])
    return over, push, under


def _player_game_rng(game_id, player_id, seed: int):
    material = (
        f"{int(seed)}|points|{canonical_id(game_id)}|{canonical_id(player_id)}"
    ).encode()
    digest = sha256(material).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _uniform_column(drawn, name: str, n_draws: int) -> np.ndarray:
    pairs = drawn.pairs
    values = np.asarray(pairs[name], dtype=float).reshape(-1)
    if values.shape != (n_draws,):
        raise ValueError("copula draw count")
    return values


def _knot_grid(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = [f"{prefix}_q_{level:.2f}" for level in QUANTILE_LEVELS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing {missing}")
    return frame.loc[:, columns].to_numpy(dtype=float)


def _bincount_rows(counts: np.ndarray, width: int) -> np.ndarray:
    n_rows = counts.shape[0]
    offsets = (np.arange(n_rows, dtype=np.int64) * width)[:, None]
    flat = np.bincount(
        (counts + offsets).ravel(),
        minlength=n_rows * width,
    )
    return flat.reshape(n_rows, width).astype(float)


def _pmf_summaries(pmf: np.ndarray):
    ks = np.arange(pmf.shape[1], dtype=float)
    mean = pmf @ ks
    cdf = np.cumsum(pmf, axis=1)
    median = np.argmax(cdf >= 0.5 - 1e-12, axis=1).astype(int)
    return mean, median


def _whole_line(mat: np.ndarray, point: int, width: int):
    n_rows = mat.shape[0]
    over = np.zeros(n_rows, dtype=float)
    push = np.zeros(n_rows, dtype=float)
    under = np.zeros(n_rows, dtype=float)
    if point < 0:
        over = mat.sum(axis=1)
    elif point >= width:
        under = mat.sum(axis=1)
    else:
        push = mat[:, point]
        if point + 1 < width:
            over = mat[:, point + 1 :].sum(axis=1)
        if point > 0:
            under = mat[:, :point].sum(axis=1)
    return over, push, under


def _half_line(mat: np.ndarray, floor_point: int, width: int):
    """Over is the mass strictly above ``floor_point``. Push is 0."""
    n_rows = mat.shape[0]
    push = np.zeros(n_rows, dtype=float)
    if floor_point < 0:
        over = mat.sum(axis=1)
        under = np.zeros(n_rows, dtype=float)
    elif floor_point >= width - 1:
        over = np.zeros(n_rows, dtype=float)
        under = mat.sum(axis=1)
    else:
        over = mat[:, floor_point + 1 :].sum(axis=1)
        under = mat[:, : floor_point + 1].sum(axis=1)
    return over, push, under


def _positive_draws(n_draws: int) -> int:
    count = int(n_draws)
    if count <= 0:
        raise ValueError("n_draws")
    return count


def _kmax(kmax: int) -> int:
    limit = int(kmax)
    if limit < 0:
        raise ValueError("kmax")
    return limit
