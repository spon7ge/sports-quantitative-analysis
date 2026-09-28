"""Out-of-sample minutes and rate PIT pairs, resampled by minutes tier.

Pairs are stored raw. Sampling draws those pairs with replacement. It does
not rank-transform them and it does not add jitter.
"""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.shared.metrics import MINUTE_Q50_TIERS
from models.shared.minutes_sampler import (
    QUANTILE_LEVELS,
    canonical_id,
    groups_for_frame as minutes_groups,
    randomized_pit as minutes_pit,
)
from models.shared.oos import capped_rate
from models.shared.ppm_sampler import (
    groups_for_frame as rate_groups,
    randomized_pit as rate_pit,
)

MIN_TIER_PAIRS = 3_000
INDEPENDENCE_DECILE_MEAN = 0.5
INDEPENDENCE_DECILE_SD = math.sqrt(1.0 / 12.0)
INDEPENDENCE_TAIL_RATIO = 1.0
IDEAL_TAIL_SHARE = 0.05
_TAIL_CELL = 0.01


class CopulaSample(NamedTuple):
    pairs: pd.DataFrame
    pooled: bool


def build_pit_pairs(
    oos_table: pd.DataFrame,
    *,
    minutes_tables,
    rate_tables,
    starting: pd.DataFrame | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomized PIT of realized minutes and of the capped rate, per row.

    ``u_m`` uses the minutes sampler CDF, tails included. ``u_r`` uses the
    rate sampler CDF on ``min(pts / minutes, 6)``. The tier is predicted
    minutes q50. Atoms are seeded per ``(game_id, player_id)``.
    """
    flags = _aligned_starting(oos_table, starting)
    minute_groups, minute_upper = minutes_groups(flags, minutes_tables)
    rate_lower, rate_upper = rate_groups(flags, rate_tables)
    minute_grids = _knot_grid(oos_table, "minutes")
    rate_grids = _knot_grid(oos_table, "rate")
    minutes = oos_table["minutes"].to_numpy(dtype=float)
    rate = capped_rate(oos_table["pts"], minutes)
    atom_minutes = _atom_uniform(
        oos_table["player_id"],
        oos_table["game_id"],
        seed=seed,
        label="pit_minutes",
    )
    atom_rate = _atom_uniform(
        oos_table["player_id"],
        oos_table["game_id"],
        seed=seed,
        label="pit_rate",
    )
    u_m = minutes_pit(
        minutes,
        atom_minutes,
        minute_grids,
        minute_groups,
        minute_upper,
        minutes_tables,
    )
    u_r = rate_pit(
        rate,
        atom_rate,
        rate_grids,
        rate_lower,
        rate_upper,
        rate_tables,
    )
    out = oos_table.loc[:, ["game_id", "player_id", "game_date"]].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["is_holdout"] = oos_table["is_holdout"].astype(bool).to_numpy()
    out["tier"] = minutes_tier(oos_table["minutes_q_0.50"])
    out["u_m"] = np.asarray(u_m, dtype=float)
    out["u_r"] = np.asarray(u_r, dtype=float)
    return out.reset_index(drop=True)


def minutes_tier(predicted_q50) -> np.ndarray:
    """Bucket predicted minutes q50 with the notebook cuts."""
    q50 = np.asarray(predicted_q50, dtype=float)
    labels = np.empty(q50.shape[0], dtype=object)
    seen = np.zeros(q50.shape[0], dtype=bool)
    for name, low, high in MINUTE_Q50_TIERS:
        mask = np.ones(q50.shape[0], dtype=bool)
        if low is not None:
            mask &= q50 >= low
        if high is not None:
            mask &= q50 < high
        labels[mask] = name
        seen |= mask
    if not seen.all() or not np.isfinite(q50).all():
        raise ValueError("tier")
    return labels


class EmpiricalCopula:
    """Resample stored PIT pairs with replacement.

    Pairs with ``game_date`` on or after ``before_date`` are dropped.
    Holdout pairs are dropped unless ``include_holdout`` is true. A tier
    with fewer than 3,000 pairs is drawn from every remaining tier, and
    the sample is flagged.
    """

    def __init__(self, pairs: pd.DataFrame, before_date, include_holdout: bool = False):
        self.before_date = pd.Timestamp(before_date)
        self.include_holdout = bool(include_holdout)
        self.pairs = _eligible_pairs(pairs, self.before_date, self.include_holdout)

    def sample(self, tier: str, n: int, rng) -> CopulaSample:
        n = _draw_count(n)
        tier_rows = self.pairs.loc[self.pairs["tier"].eq(tier)]
        pooled = len(tier_rows) < MIN_TIER_PAIRS
        pool = self.pairs if pooled else tier_rows
        if len(pool) == 0:
            raise ValueError("empty copula")
        if n == 0:
            return CopulaSample(pool.iloc[0:0].copy(), pooled)
        index = rng.integers(0, len(pool), size=n)
        drawn = pool.iloc[np.asarray(index)].reset_index(drop=True)
        return CopulaSample(drawn, pooled)


class IndependentCopula:
    """Independent ``U(0, 1)`` pairs. Same constructor and ``sample`` signature."""

    def __init__(self, pairs: pd.DataFrame, before_date, include_holdout: bool = False):
        self.before_date = pd.Timestamp(before_date)
        self.include_holdout = bool(include_holdout)
        self.pairs = _eligible_pairs(pairs, self.before_date, self.include_holdout)

    def sample(self, tier: str, n: int, rng) -> CopulaSample:
        n = _draw_count(n)
        drawn = pd.DataFrame(
            {
                "u_m": rng.random(n),
                "u_r": rng.random(n),
                "tier": tier,
            }
        )
        return CopulaSample(drawn, False)


def pair_dependence(pairs: pd.DataFrame) -> dict:
    """Spearman, decile profile, and tail shares of one PIT-pair set."""
    u_m = np.asarray(pairs["u_m"], dtype=float)
    u_r = np.asarray(pairs["u_r"], dtype=float)
    if u_m.shape != u_r.shape or u_m.ndim != 1:
        raise ValueError("pairs")
    n = int(u_m.shape[0])
    if n == 0:
        raise ValueError("empty pairs")
    return {
        "n": n,
        "spearman": float(spearmanr(u_m, u_r).statistic),
        "deciles": _u_r_by_u_m_decile(u_m, u_r),
        "lower_lower": _tail_ratio((u_m < 0.1) & (u_r < 0.1)),
        "lower_upper": _tail_ratio((u_m < 0.1) & (u_r > 0.9)),
        "share_u_m_below_05": float(np.mean(u_m < 0.05)),
        "share_u_m_above_95": float(np.mean(u_m > 0.95)),
        "share_u_r_below_05": float(np.mean(u_r < 0.05)),
        "share_u_r_above_95": float(np.mean(u_r > 0.95)),
    }


def render_dependence_report(pairs: pd.DataFrame) -> str:
    """Markdown for pre-holdout pairs, one block per predicted-minutes tier."""
    pre = pairs.loc[~pairs["is_holdout"].astype(bool)].copy()
    if pre.empty:
        raise ValueError("pre-holdout")
    lines = [
        "# Minutes and rate dependence",
        "",
        "These numbers describe the pre-holdout out-of-sample PIT pairs. "
        "They are not a calibration slice of the points PMF.",
        "",
        "`u_m` is the randomized PIT of realized minutes. `u_r` is the "
        "randomized PIT of `min(pts / minutes, 6)`. Both use the sampler "
        "CDFs, tails included. The tier is that row's predicted minutes q50.",
        "",
        "Under independence, Spearman correlation is 0. Inside each `u_m` "
        "decile, mean `u_r` is 0.500 and the SD of `u_r` is 0.289. "
        "`P(u_m < 0.1, u_r < 0.1) / 0.01` and "
        "`P(u_m < 0.1, u_r > 0.9) / 0.01` both equal 1.0. The share of "
        "either uniform below 0.05 or above 0.95 is 5%.",
        "",
        "Deciles of `u_m` are `[0.0, 0.1), …, [0.9, 1.0]`.",
        "",
        "## By tier",
        "",
        "| Tier | n | Spearman | P(u_m<0.1, u_r<0.1)/0.01 | "
        "P(u_m<0.1, u_r>0.9)/0.01 | u_m < 0.05 | u_m > 0.95 | "
        "u_r < 0.05 | u_r > 0.95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    blocks: list[tuple[str, dict]] = []
    for name, _low, _high in MINUTE_Q50_TIERS:
        block = pre.loc[pre["tier"].eq(name)]
        if block.empty:
            continue
        stats = pair_dependence(block)
        blocks.append((name, stats))
        lines.append(
            "| {tier} | {n} | {rho:.3f} | {ll:.3f} | {lu:.3f} | "
            "{m_lo} | {m_hi} | {r_lo} | {r_hi} |".format(
                tier=name,
                n=f"{stats['n']:,}",
                rho=stats["spearman"],
                ll=stats["lower_lower"],
                lu=stats["lower_upper"],
                m_lo=_pct(stats["share_u_m_below_05"]),
                m_hi=_pct(stats["share_u_m_above_95"]),
                r_lo=_pct(stats["share_u_r_below_05"]),
                r_hi=_pct(stats["share_u_r_above_95"]),
            )
        )
    lines.extend(
        [
            "",
            "## u_r inside each u_m decile",
            "",
            "| Tier | u_m decile | n | Mean u_r | SD u_r |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, stats in blocks:
        for decile in stats["deciles"].itertuples(index=False):
            lines.append(
                f"| {name} | {decile.u_m_lo:.1f}-{decile.u_m_hi:.1f} | "
                f"{decile.n:,} | {_num(decile.mean_u_r)} | {_num(decile.sd_u_r)} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_dependence_report(pairs: pd.DataFrame, path: str | Path) -> Path:
    """Write :func:`render_dependence_report` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dependence_report(pairs))
    return path


def _aligned_starting(oos: pd.DataFrame, starting: pd.DataFrame | None) -> pd.Series:
    if starting is None:
        if "starting" not in oos.columns:
            raise ValueError("starting")
        return oos["starting"]
    flags = starting.loc[:, ["game_id", "player_id", "starting"]].copy()
    flags["game_id"] = flags["game_id"].map(canonical_id)
    flags["player_id"] = flags["player_id"].map(canonical_id)
    if flags.duplicated(["game_id", "player_id"]).any():
        counts = flags.groupby(
            ["game_id", "player_id"], sort=False
        )["starting"].nunique()
        if bool((counts > 1).any()):
            raise ValueError("starting")
        flags = flags.drop_duplicates(["game_id", "player_id"])
    keys = pd.MultiIndex.from_arrays(
        [
            oos["game_id"].map(canonical_id),
            oos["player_id"].map(canonical_id),
        ]
    )
    values = flags.set_index(["game_id", "player_id"])["starting"].reindex(keys)
    if values.isna().any():
        raise ValueError("starting")
    return pd.Series(values.to_numpy(), index=oos.index)


def _knot_grid(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = [f"{prefix}_q_{level:.2f}" for level in QUANTILE_LEVELS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing {missing}")
    return frame.loc[:, columns].to_numpy(dtype=float)


def _atom_uniform(player_ids, game_ids, *, seed: int, label: str) -> np.ndarray:
    players = list(player_ids)
    games = list(game_ids)
    if len(players) != len(games):
        raise ValueError("id")
    out = np.empty(len(players), dtype=float)
    for index, (player_id, game_id) in enumerate(zip(players, games, strict=True)):
        material = (
            f"{seed}|{label}|{canonical_id(player_id)}|{canonical_id(game_id)}"
        ).encode()
        digest = sha256(material).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        out[index] = float(rng.random())
    return out


def _eligible_pairs(
    pairs: pd.DataFrame,
    before_date,
    include_holdout: bool,
) -> pd.DataFrame:
    dates = pd.to_datetime(pairs["game_date"])
    cutoff = pd.Timestamp(before_date)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    keep = np.asarray(dates < cutoff, dtype=bool)
    if not include_holdout:
        holdout = np.asarray(pairs["is_holdout"], dtype=bool)
        keep = keep & ~holdout
    return pairs.loc[keep].reset_index(drop=True)


def _draw_count(n: int) -> int:
    count = int(n)
    if count < 0:
        raise ValueError("n")
    return count


def _tail_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask) / _TAIL_CELL)


def _u_r_by_u_m_decile(u_m: np.ndarray, u_r: np.ndarray) -> pd.DataFrame:
    bins = np.clip(np.floor(u_m * 10.0).astype(int), 0, 9)
    rows = []
    for decile in range(10):
        values = u_r[bins == decile]
        count = int(values.shape[0])
        if count == 0:
            mean = sd = math.nan
        else:
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if count > 1 else math.nan
        rows.append(
            {
                "decile": decile + 1,
                "u_m_lo": decile / 10.0,
                "u_m_hi": (decile + 1) / 10.0,
                "n": count,
                "mean_u_r": mean,
                "sd_u_r": sd,
            }
        )
    return pd.DataFrame(rows)


def _pct(share: float) -> str:
    return f"{100.0 * share:.1f}%"


def _num(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.3f}"
