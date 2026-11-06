"""Illustrative Boston rotation segments on one attached script.

The schedule is not a coached rotation and not a reconstruction from
play-by-play. Play-by-play can show who entered, who left, the period,
and the clock (`actionType` Substitution, `description`, `period`,
`clock`, `personId`). Those reconstructed minutes are evidence. They
are not this script.

Layer 4 rates and the possession counts below are illustrative
magnitudes. Segments realize the Layer 3 minute budget. A later random
rotation would still be one draw per environment row, conditional on
the attached script and the role.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from possession_allocation import OpportunityModel, allocate_opportunities
from role_context import BLOWOUT_MARGIN
from rotations import (
    RotationSimulator,
    allocate_rotation,
    invalid_segment_count,
)

PLAYERS = (
    "Tatum",
    "Brown",
    "White",
    "Holiday",
    "Porzingis",
    "Hauser",
    "Horford",
    "Pritchard",
    "Kornet",
)
STARTERS = PLAYERS[:5]
MIX_A = ("Tatum", "Brown", "White", "Hauser", "Horford")
MIX_B = ("Brown", "White", "Holiday", "Pritchard", "Kornet")
SECOND = ("White", "Hauser", "Horford", "Pritchard", "Kornet")
BLOWOUT_CLOSE = ("Hauser", "Horford", "Pritchard", "Kornet", "White")

# Illustrative script. Regulation on these rows is 48 minutes. Each
# overtime period the script already drew is 5 minutes.
_REGULATION = 48.0
_OVERTIME = 5.0


def _segment(period: int, start: float, end: float, players: tuple[str, ...]) -> dict:
    return {
        "period": period,
        "start_min": start,
        "end_min": end,
        "players": list(players),
    }


def illustrative_schedule(n_ot: int, blowout: bool) -> list[dict]:
    """Posted starters, bench stints, and a closing five. Illustrative."""
    closing = BLOWOUT_CLOSE if blowout else STARTERS
    quarters = (
        (1, ((0.0, 6.0, STARTERS), (6.0, 9.0, MIX_A), (9.0, 12.0, MIX_B))),
        (2, ((0.0, 6.0, SECOND), (6.0, 9.0, MIX_B), (9.0, 12.0, STARTERS))),
        (3, ((0.0, 6.0, STARTERS), (6.0, 9.0, MIX_A), (9.0, 12.0, MIX_B))),
        (4, ((0.0, 6.0, SECOND), (6.0, 9.0, MIX_B), (9.0, 12.0, closing))),
    )
    segments = [
        _segment(period, start, end, players)
        for period, pieces in quarters
        for start, end, players in pieces
    ]
    overtime_five = BLOWOUT_CLOSE if blowout else STARTERS
    for extra in range(n_ot):
        segments.append(_segment(5 + extra, 0.0, _OVERTIME, overtime_five))
    return segments


def _role() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": list(PLAYERS),
            "team_id": ["BOS"] * len(PLAYERS),
            "is_starter": [1.0 if player in STARTERS else 0.0 for player in PLAYERS],
            "available": np.ones(len(PLAYERS)),
        }
    )


def _scripts() -> list[dict]:
    rows = (
        ("regulation", 4.0, 0),
        ("overtime", 2.0, 1),
        ("blowout", -24.0, 0),
    )
    scripts = []
    for label, margin, n_ot in rows:
        game_minutes = _REGULATION + _OVERTIME * n_ot
        # Same row as the minutes. Overtime lengthens the count. Pace is not raised.
        team_poss = 100.0 * game_minutes / _REGULATION
        scripts.append(
            {
                "label": label,
                "team_margin": margin,
                "n_ot": n_ot,
                "game_minutes": game_minutes,
                "team_poss": team_poss,
            }
        )
    return scripts


def _profile(player_ids: pd.Series) -> pd.DataFrame:
    """Illustrative attempts per minute. Not fit from a window."""
    rates = {
        "fga_per_min": [0.58, 0.52, 0.40, 0.42, 0.48, 0.36, 0.34, 0.44, 0.28],
        "p_3pa": [0.42, 0.40, 0.55, 0.48, 0.32, 0.62, 0.38, 0.50, 0.08],
        "p_3p": [0.37, 0.35, 0.38, 0.36, 0.35, 0.40, 0.36, 0.39, 0.30],
        "p_2p": [0.54, 0.55, 0.50, 0.52, 0.58, 0.48, 0.56, 0.46, 0.62],
        "fta_per_fga": [0.30, 0.24, 0.14, 0.16, 0.28, 0.10, 0.12, 0.12, 0.22],
        "ft_pct": [0.83, 0.78, 0.86, 0.84, 0.80, 0.88, 0.82, 0.86, 0.74],
        "tov_per_min": [0.08, 0.07, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.04],
    }
    frame = pd.DataFrame(rates)
    frame.insert(0, "player_id", list(player_ids))
    frame["fga_per_poss"] = 0.88
    frame["tov_per_poss"] = 0.14
    return frame


def _model() -> OpportunityModel:
    return OpportunityModel(
        fga_per_min=0.40,
        p_3pa=0.40,
        p_3p=0.36,
        p_2p=0.52,
        fta_per_fga=0.22,
        ft_pct=0.80,
        tov_per_min=0.05,
        fga_per_poss=0.88,
        tov_per_poss=0.14,
        alpha=0.03,
    )


def main() -> None:
    role = _role()
    profile = _profile(role["player_id"])
    model = _model()
    simulator = RotationSimulator(fallback_mu=24.0, fallback_sd=1.0)
    print("Illustrative Boston rotation. Not a coached rotation and not play-by-play.")
    print("Layer 4 rates are illustrative, not fit from history.")
    for script in _scripts():
        blowout = abs(script["team_margin"]) >= BLOWOUT_MARGIN
        segments = illustrative_schedule(script["n_ot"], blowout)
        invalid = invalid_segment_count(
            segments,
            role,
            game_minutes=script["game_minutes"],
            n_ot=script["n_ot"],
        )
        environment = pd.DataFrame(
            [
                {
                    "team_margin": script["team_margin"],
                    "n_ot": script["n_ot"],
                    "game_minutes": script["game_minutes"],
                }
            ]
        )
        minutes = allocate_rotation(
            simulator,
            role,
            environment,
            np.random.default_rng(3),
            method="segments",
            segments=segments,
        )
        game_minutes = float(script["game_minutes"])
        total = float(minutes.sum())
        if invalid != 0:
            raise RuntimeError(f"invalid segments: {invalid}")
        if abs(total - 5.0 * game_minutes) > 1e-6:
            raise RuntimeError("segment minutes missed the team minute budget")
        if float(minutes.max()) > game_minutes + 1e-6:
            raise RuntimeError("a player exceeded game minutes")

        print()
        print(
            f"{script['label']}: n_ot {script['n_ot']}  "
            f"team_margin {script['team_margin']:+.1f}  "
            f"cap {game_minutes:.1f}  "
            f"team total {total:.1f}  "
            f"invalid segments {invalid}"
        )
        for name, played in zip(PLAYERS, minutes[0], strict=True):
            print(f"  {name:<12} {played:5.1f}")

        allocated = allocate_opportunities(
            model,
            profile,
            minutes,
            np.array([script["team_poss"]], dtype=float),
            np.random.default_rng(3),
        )
        print(
            f"  illustrative team_poss {script['team_poss']:.1f}  "
            f"team FGA {int(allocated['team_fga'][0])}  "
            f"team TOV {int(allocated['team_tov'][0])}"
        )
        shares = "  ".join(
            f"{name} {share:.3f}"
            for name, share in zip(PLAYERS, allocated["shares"][0], strict=True)
        )
        print(f"  shot shares  {shares}")

    print()
    print(
        "Assumption: rotation segments realize the Layer 3 minute budget. "
        "A later random rotation would still be one draw per row, "
        "conditional on the attached script and the role."
    )


if __name__ == "__main__":
    main()
