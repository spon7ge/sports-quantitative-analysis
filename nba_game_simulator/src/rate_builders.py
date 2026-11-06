"""Optional PBP → player shooting rates for Layer 4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = (
    "teamTricode",
    "personId",
    "isFieldGoal",
    "shotValue",
    "shotResult",
)


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


def build_player_rates_from_pbp(
    pbp: pd.DataFrame,
    team_tricode: str,
    player_name_map: dict[int, str] | None = None,
) -> dict[str, np.ndarray | float]:
    missing = [c for c in _REQUIRED_COLUMNS if c not in pbp.columns]
    if missing:
        raise ValueError(f"pbp missing required columns: {missing}")

    team = pbp.loc[pbp["teamTricode"] == team_tricode]
    fga_rows = team.loc[team["isFieldGoal"].astype(int) == 1]
    if fga_rows.empty:
        raise ValueError(f"no field-goal attempts for team_tricode={team_tricode!r}")

    is_three = fga_rows["shotValue"].astype(int) == 3
    is_made = fga_rows["shotResult"] == "Made"
    work = fga_rows.assign(
        three_pa=is_three.astype(int),
        three_pm=(is_three & is_made).astype(int),
        two_pa=(~is_three).astype(int),
        two_pm=((~is_three) & is_made).astype(int),
    )

    stats = work.groupby("personId", sort=True).agg(
        fga=("personId", "size"),
        three_pa=("three_pa", "sum"),
        three_pm=("three_pm", "sum"),
        two_pa=("two_pa", "sum"),
        two_pm=("two_pm", "sum"),
    )

    player_ids = stats.index.to_numpy(dtype=np.int64)
    fga = stats["fga"].to_numpy(dtype=float)
    team_fga = float(fga.sum())

    result: dict[str, Any] = {
        "player_ids": player_ids,
        "base_weights": fga / team_fga,
        "p_3pa": _safe_ratio(stats["three_pa"].to_numpy(dtype=float), fga),
        "p_3p": _safe_ratio(
            stats["three_pm"].to_numpy(dtype=float),
            stats["three_pa"].to_numpy(dtype=float),
        ),
        "p_2p": _safe_ratio(
            stats["two_pm"].to_numpy(dtype=float),
            stats["two_pa"].to_numpy(dtype=float),
        ),
    }

    if player_name_map is not None:
        result["player_names"] = np.array(
            [player_name_map[int(pid)] for pid in player_ids],
            dtype=object,
        )

    return result
