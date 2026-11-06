"""Standalone Layer 4 Boston demo. Generates team FGA outside the core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from possession_allocation import sample_negbin, simulate_game


def draw_team_fga(possessions, fga_per_poss, alpha, n_sims, rng, family="negbin"):
    mu = float(possessions) * float(fga_per_poss)
    if family == "poisson":
        return sample_negbin(mu, alpha=0.0, rng=rng, size=(n_sims,))
    return sample_negbin(mu, alpha=alpha, rng=rng, size=(n_sims,))


def summarize_player_stats(stats, player_ids):
    rows = []
    for j, pid in enumerate(player_ids):
        for name, arr in stats.items():
            col = arr[:, j]
            rows.append(
                {
                    "player": pid,
                    "stat": name,
                    "mean": float(col.mean()),
                    "sd": float(col.std(ddof=0)),
                    "p25": float(np.percentile(col, 25)),
                    "p50": float(np.percentile(col, 50)),
                    "p75": float(np.percentile(col, 75)),
                }
            )
    return pd.DataFrame(rows)


def main():
    n_sims = 10_000
    rng = np.random.default_rng(42)
    possessions = 101
    team_fga = draw_team_fga(
        possessions, fga_per_poss=0.88, alpha=0.03, n_sims=n_sims, rng=rng
    )
    players = {
        "player_ids": np.array(["Tatum", "Brown", "White", "Porzingis", "Other"]),
        "base_weights": np.array([0.31, 0.27, 0.19, 0.15, 0.08]),
        "available": np.ones(5),
        "projected_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "reference_minutes": np.array([36.0, 34.0, 32.0, 28.0, 18.0]),
        "p_3pa": np.array([0.45, 0.40, 0.55, 0.35, 0.40]),
        "p_3p": np.array([0.37, 0.35, 0.40, 0.37, 0.34]),
        "p_2p": np.array([0.55, 0.54, 0.52, 0.58, 0.50]),
        "fta_per_fga": np.array([0.28, 0.25, 0.15, 0.30, 0.18]),
        "ft_pct": np.array([0.83, 0.78, 0.88, 0.84, 0.75]),
        "tov_rate": np.array([0.12, 0.11, 0.09, 0.10, 0.10]),
        "assist_propensity": np.array([0.25, 0.20, 0.35, 0.10, 0.10]),
        "oreb_propensity": np.array([0.15, 0.15, 0.10, 0.40, 0.20]),
        "assist_rate": 0.60,
        "oreb_rate": 0.28,
        "dispersion": {"fta": 0.05, "ast": 0.05, "oreb": 0.05, "tov": 0.05},
    }
    result = simulate_game(
        team_inputs={"BOS": {"possessions": possessions, "team_fga": team_fga}},
        player_inputs={"BOS": players},
        n_sims=n_sims,
        rng=rng,
    )
    bos = result.teams["BOS"]
    assert np.array_equal(bos.stats["FGA"].sum(1), team_fga)
    assert np.all(bos.stats["AST"].sum(1) <= bos.stats["FGM"].sum(1))
    summary = summarize_player_stats(bos.stats, bos.player_ids)
    print(summary.pivot(index="player", columns="stat", values="mean"))
    print(
        summary[summary["stat"].isin(["PTS", "FGA", "3PM", "FTM", "AST", "TOV", "OREB"])]
        .sort_values(["player", "stat"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
