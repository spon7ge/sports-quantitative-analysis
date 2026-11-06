"""Layer 6 demo: layers 1–5 produce one box score and a joint prop.

The environment model is a synthetic league prior, not a fit. Player rates
are illustrative magnitudes. The printed probabilities are this run, not the
hypothetical 54/57/62 figures in the spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from box_score import Prop, format_outcome, prop_probability, simulate_box_scores
from game_enviroment import FEATURES


def _model() -> dict:
    def team(abbr: str) -> dict:
        return {
            "abbr": abbr,
            "num": 0.0,
            "den": 0.0,
            "last_date": "2024-04-10",
            "last_season": "2023-24",
        }

    return {
        "config": {"season_carry": 0.5, "prior_weight": 10.0},
        "state": {
            "league_poss": 100.0,
            "league_pts": 220.0,
            "teams": {"1": team("BOS"), "2": team("NYK")},
        },
        "model": {"name": "league_prior"},
        "features": [name for name in FEATURES if not str(name).startswith("lineup_")],
        "margin": {"a": 0.0, "b": 0.0, "sd": 12.0, "nu": 8.0, "scale": 12.0},
        "script": {"kappa": 0.0},
        "residual_t": {"nu": 30.0, "scale": 1.0},
        "ot": {
            "p_ot_overall": 0.0,
            "p_ot_by_spread_bucket": [0.0, 0.0, 0.0, 0.0],
            "p_more_given_ot": 0.0,
        },
        "split": {"values": [0], "probs": [1.0]},
    }


def _roster(ids: list[str], minutes: list[float]) -> dict:
    # Nine players so the 240-minute budget can keep a rotation, rather than
    # pinning five players to the 48-minute cap.
    players = len(ids)
    weights = np.array([0.22, 0.18, 0.14, 0.13, 0.12, 0.07, 0.06, 0.05, 0.03])
    return {
        "player_ids": np.array(ids),
        "expected_minutes": np.array(minutes, dtype=float),
        "minutes_sd": np.full(players, 1.5),
        "base_weights": weights,
        "available": np.ones(players),
        "reference_minutes": np.array(minutes, dtype=float),
        "p_3pa": np.array([0.42, 0.40, 0.38, 0.36, 0.22, 0.45, 0.12, 0.40, 0.05]),
        "p_3p": np.full(players, 0.36),
        "p_2p": np.full(players, 0.52),
        "fta_per_fga": np.full(players, 0.22),
        "ft_pct": np.full(players, 0.80),
        "tov_rate": np.full(players, 0.12),
        "assist_propensity": np.array(
            [0.22, 0.14, 0.12, 0.20, 0.08, 0.06, 0.05, 0.08, 0.05]
        ),
        "oreb_propensity": weights.copy(),
        "oreb_rate": np.array([0.06, 0.05, 0.04, 0.04, 0.16, 0.03, 0.14, 0.03, 0.20]),
        "dreb_rate": np.array([0.10, 0.08, 0.06, 0.07, 0.22, 0.04, 0.16, 0.04, 0.23]),
        "creation_rate": np.array(
            [0.24, 0.16, 0.12, 0.18, 0.08, 0.06, 0.05, 0.07, 0.04]
        ),
        "assisted_rate": np.array(
            [0.45, 0.55, 0.60, 0.40, 0.65, 0.75, 0.70, 0.72, 0.78]
        ),
        "stl_rate": np.array([0.14, 0.12, 0.14, 0.16, 0.08, 0.10, 0.08, 0.12, 0.06]),
        "blk_rate": np.array([0.06, 0.04, 0.04, 0.03, 0.22, 0.02, 0.16, 0.02, 0.28]),
    }


def main() -> None:
    box = simulate_box_scores(
        environment_model=_model(),
        home="BOS",
        away="NYK",
        date="2026-11-05",
        total=220.0,
        spread=0.0,
        rosters={
            "BOS": _roster(
                [
                    "Tatum",
                    "Brown",
                    "White",
                    "Holiday",
                    "Porzingis",
                    "Hauser",
                    "Horford",
                    "Pritchard",
                    "Kornet",
                ],
                [36, 34, 32, 32, 28, 22, 20, 18, 18],
            ),
            "NYK": _roster(
                [
                    "Brunson",
                    "Bridges",
                    "Anunoby",
                    "Hart",
                    "Randle",
                    "McBride",
                    "Achiuwa",
                    "Payne",
                    "Robinson",
                ],
                [36, 34, 34, 30, 32, 20, 18, 16, 20],
            ),
        },
        n_sims=2000,
        rng=np.random.default_rng(6),
    )
    game_minutes = box.environment["game_minutes"].to_numpy()
    bos_min = box.teams["BOS"].stats["MIN"]
    assert bos_min[:, 0].mean() > bos_min[:, -1].mean() + 10
    for team in ("BOS", "NYK"):
        minute_sum = box.teams[team].stats["MIN"].sum(axis=1)
        assert np.allclose(minute_sum, 5.0 * game_minutes, atol=0.1)
        assert np.all(
            box.teams[team].stats["AST"].sum(axis=1)
            <= box.teams[team].stats["FGM"].sum(axis=1)
        )

    print("simulation 0")
    for row in format_outcome(box, sim=0):
        print(
            f"  {row['team']:3} {row['player']:<12} "
            f"MIN {row['MIN']:5.1f}  PTS {row['PTS']:5.0f}  "
            f"REB {row['REB']:4.0f}  AST {row['AST']:4.0f}  "
            f"3PM {row['3PM']:3.0f}"
        )

    props = [
        Prop("BOS", "Tatum", "PTS", 22.5),
        Prop("BOS", "Brown", "PTS", 18.5),
    ]
    result = prop_probability(box, props)
    print(
        f"P(Tatum over 22.5 and Brown over 18.5) = {result.joint:.3f}; "
        f"product of marginals = {result.independent_product:.3f}"
    )
    print(f"marginals = {tuple(round(value, 3) for value in result.marginals)}")


if __name__ == "__main__":
    main()
