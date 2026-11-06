"""Layer 5 injury demo.

Illustrative rates, not fit from a sample. Made shots, misses, and
turnovers are held fixed so the table shows redistribution of the same
events. The spec's minute table does not list the starting point guard;
this demo uses 34 baseline minutes for that player.

Layer 4's ``simulate_game`` can supply the FGA, FGM, and TOV arrays.
Pass those into ``simulate_shared_events`` and use this layer's assists
and rebounds. Layer 4's own AST and OREB draws are the single-team
approximation and are not added on top of these.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from shared_events import simulate_shared_events


NAMES = np.array(
    ["Starter PG", "Backup PG", "Starting SG", "Center", "Bench guard"]
)
BASE_MIN = np.array([34.0, 22.0, 35.0, 31.0, 17.0])
INJURY_MIN = np.array([0.0, 32.0, 37.0, 33.0, 11.0])
CREATION = np.array([0.40, 0.25, 0.20, 0.08, 0.07])
REBOUND = np.array([0.05, 0.08, 0.10, 0.45, 0.12])


def _roster(n_sims: int, minutes: np.ndarray, available: np.ndarray) -> dict:
    fga = np.zeros((n_sims, 5), dtype=np.int64)
    fgm = np.zeros((n_sims, 5), dtype=np.int64)
    tov = np.zeros((n_sims, 5), dtype=np.int64)
    fga[:, 1:] = 8
    fgm[:, 1:] = 4
    tov[:, 1:] = 1
    return {
        "player_ids": NAMES,
        "fga": fga,
        "fgm": fgm,
        "tov": tov,
        "available": available,
        "projected_minutes": minutes,
        "reference_minutes": BASE_MIN,
        "oreb_rate": REBOUND,
        "dreb_rate": REBOUND,
        "creation_rate": CREATION,
        "assisted_rate": np.ones(5),
        "stl_rate": CREATION,
        "blk_rate": REBOUND,
    }


def _means(stats: dict[str, np.ndarray], stat: str) -> np.ndarray:
    return stats[stat].mean(axis=0)


def main() -> None:
    n_sims = 4_000
    opponent = {
        "player_ids": np.array(["Opp G", "Opp C"]),
        "fga": np.full((n_sims, 2), 15, dtype=np.int64),
        "fgm": np.full((n_sims, 2), 5, dtype=np.int64),
        "tov": np.full((n_sims, 2), 2, dtype=np.int64),
        "available": np.ones(2),
        "projected_minutes": np.array([34.0, 32.0]),
        "reference_minutes": np.array([34.0, 32.0]),
        "oreb_rate": np.array([0.2, 0.8]),
        "dreb_rate": np.array([0.3, 0.7]),
        "creation_rate": np.ones(2),
        "assisted_rate": np.full(2, 0.6),
        "stl_rate": np.ones(2),
        "blk_rate": np.array([0.2, 0.8]),
    }
    available = np.ones(5)
    injured = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    def run(minutes, active, seed):
        return simulate_shared_events(
            {"BOS": _roster(n_sims, minutes, active), "NYK": opponent},
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
            p_rebound=1.0,
            p_oreb=0.0,
            p_steal=1.0,
            p_block=1.0,
        )

    baseline = run(BASE_MIN, available, 1)
    injury = run(INJURY_MIN, injured, 2)
    base = baseline.teams["BOS"].stats
    inj = injury.teams["BOS"].stats

    # Every missed opponent shot is a defensive rebound, and every make is
    # assisted, so the team pools match the events in both scenarios.
    assert np.array_equal(base["AST"].sum(1), inj["AST"].sum(1))
    assert np.array_equal(base["DREB"].sum(1), np.full(n_sims, 20))
    assert np.array_equal(inj["DREB"].sum(1), np.full(n_sims, 20))
    assert np.all(inj["AST"][:, 0] == 0)
    assert _means(inj, "AST")[1] > _means(base, "AST")[1]
    assert _means(inj, "AST")[2] > _means(base, "AST")[2]
    assert _means(inj, "DREB")[3] > _means(base, "DREB")[3]
    assert _means(inj, "DREB")[4] < _means(base, "DREB")[4]

    print(f"{'player':<16}{'min0':>6}{'min1':>6}{'ast0':>8}{'ast1':>8}{'reb0':>8}{'reb1':>8}")
    ast0, ast1 = _means(base, "AST"), _means(inj, "AST")
    reb0, reb1 = _means(base, "REB"), _means(inj, "REB")
    for i, name in enumerate(NAMES):
        print(
            f"{name:<16}{BASE_MIN[i]:6.0f}{INJURY_MIN[i]:6.0f}"
            f"{ast0[i]:8.2f}{ast1[i]:8.2f}{reb0[i]:8.2f}{reb1[i]:8.2f}"
        )
    print(
        f"{'team':<16}{'':6}{'':6}"
        f"{ast0.sum():8.2f}{ast1.sum():8.2f}{reb0.sum():8.2f}{reb1.sum():8.2f}"
    )


if __name__ == "__main__":
    main()
