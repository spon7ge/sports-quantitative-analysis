"""Joint props and one simulation's box score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_boxscore.checks import _normalize_id
from game_boxscore.constants import _COMBOS, _DOUBLE_STATS
from game_boxscore.types import (
    BoxScoreSet,
    OverPrice,
    PortfolioRisk,
    Prop,
    PropProbability,
)


def stat_values(box: BoxScoreSet, team: str, stat: str) -> np.ndarray:
    """``(n_sims, n_players)`` values of a box stat, combo, or DD/TD flag.

    Combos are ``PRA``, ``PR``, ``PA``, ``RA``, ``STOCKS``, or any
    ``+``-joined list such as ``PTS+REB``. ``DD`` and ``TD`` are 1 when at
    least two or three of PTS, REB, AST, STL, and BLK reach ten.
    """
    stats = box.teams[_resolve_team(box, team)].stats
    key = str(stat).upper().replace(" ", "")
    if key in stats:
        return stats[key]
    if key in ("DD", "TD"):
        tens = sum((stats[name] >= 10).astype(np.int64) for name in _DOUBLE_STATS)
        return (tens >= (2 if key == "DD" else 3)).astype(np.int64)
    parts = _COMBOS.get(key) or (tuple(key.split("+")) if "+" in key else None)
    if parts is None:
        raise ValueError(f"unknown stat {stat}")
    unknown = [part for part in parts if part not in stats]
    if unknown:
        raise ValueError(f"unknown stat {unknown} in {stat}")
    return sum(stats[part] for part in parts)


def prop_probability(box: BoxScoreSet, props: list[Prop]) -> PropProbability:
    """Fraction of simulations where every prop wins.

    An over on 27.5 is a value strictly above 27.5; an under is strictly
    below. A whole-number line can push, and a push is not a win. The
    independent product multiplies the marginals; it equals the joint only
    when the props do not move together inside a simulation.
    """
    if len(props) == 0:
        raise ValueError("props must not be empty")
    outcomes = [_prop_outcome(box, prop) for prop in props]
    wins = [win for win, _ in outcomes]
    together = np.logical_and.reduce(wins)
    joint = float(np.mean(together))
    marginals = tuple(float(np.mean(win)) for win in wins)
    return PropProbability(
        joint=joint,
        marginals=marginals,
        independent_product=float(np.prod(marginals)),
        n_sims=box.n_sims,
        pushes=tuple(float(np.mean(push)) for _, push in outcomes),
        joint_standard_error=float(np.sqrt(joint * (1.0 - joint) / box.n_sims)),
    )


def format_outcome(
    box: BoxScoreSet,
    sim: int,
    stats: tuple[str, ...] = ("MIN", "PTS", "REB", "AST", "3PM"),
) -> list[dict]:
    """One simulation, both teams, in roster order."""
    if isinstance(sim, bool) or not isinstance(sim, (int, np.integer)):
        raise ValueError("sim must be an integer simulation position")
    if sim < 0 or sim >= box.n_sims:
        raise ValueError(f"sim {sim} is outside 0..{box.n_sims - 1}")
    values = {
        side: {stat: stat_values(box, side, stat)[sim] for stat in stats}
        for side in box.teams
    }
    names = box.metadata.get("team_names", {})
    rows = []
    for side, team in box.teams.items():
        for index, player in enumerate(team.player_ids):
            row = {
                "sim_id": box.environment.index[sim],
                "team": names.get(side, side),
                "player": player,
            }
            for stat in stats:
                row[stat] = values[side][stat][index].item()
            rows.append(row)
    return rows


def player_summary(
    box: BoxScoreSet,
    stats: tuple[str, ...] = ("MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV"),
) -> pd.DataFrame:
    """Mean, spread, and quantiles of each player's simulated stats."""
    names = box.metadata.get("team_names", {})
    rows = []
    for side, team in box.teams.items():
        for stat in stats:
            values = np.asarray(stat_values(box, side, stat), dtype=float)
            q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9], axis=0)
            for index, player in enumerate(team.player_ids):
                column = values[:, index]
                rows.append(
                    {
                        "team": names.get(side, side),
                        "player_id": player,
                        "stat": stat,
                        "mean": float(column.mean()),
                        "sd": float(column.std()),
                        "p10": float(q10[index]),
                        "median": float(q50[index]),
                        "p90": float(q90[index]),
                        "p_zero": float(np.mean(column == 0)),
                    }
                )
    return pd.DataFrame(rows)


def price_over(box: BoxScoreSet, prop: Prop, american_odds: float) -> OverPrice:
    """Chance the stat clears the line, and the profit per dollar on that over.

    A push is not a win and not a loss. The stake is returned, so it does
    not enter the expected value.
    """
    wins, losses, pushes = _over_masks(box, prop)
    p_over = float(np.mean(wins))
    p_under = float(np.mean(losses))
    p_push = float(np.mean(pushes))
    return OverPrice(
        p_over=p_over,
        p_under=p_under,
        p_push=p_push,
        ev=p_over * _american_profit(american_odds) - p_under,
    )


def portfolio_risk(
    box: BoxScoreSet, props: list[Prop], american_odds: list[float]
) -> PortfolioRisk:
    """One dollar on every over. Risk is the spread of the combined profit.

    The joint chance uses the same simulation row for every leg. The
    independent product multiplies the separate chances.
    """
    if len(props) == 0:
        raise ValueError("portfolio needs at least one prop")
    prices = [float(price) for price in american_odds]
    if len(prices) != len(props):
        raise ValueError("each prop needs one American price")
    profits = []
    wins = []
    for prop, price in zip(props, prices, strict=True):
        leg_wins, leg_losses, _pushes = _over_masks(box, prop)
        profit = np.zeros(box.n_sims, dtype=float)
        profit[leg_wins] = _american_profit(price)
        profit[leg_losses] = -1.0
        profits.append(profit)
        wins.append(leg_wins)
    total = np.sum(np.vstack(profits), axis=0)
    together = np.logical_and.reduce(wins)
    marginals = [float(np.mean(win)) for win in wins]
    return PortfolioRisk(
        n=len(props),
        expected_profit=float(np.mean(total)),
        risk=float(np.std(total)),
        p_loss=float(np.mean(total < 0)),
        joint=float(np.mean(together)),
        independent_product=float(np.prod(marginals)),
    )


def _american_profit(american_odds: float) -> float:
    try:
        price = float(american_odds)
    except (TypeError, ValueError) as exc:
        raise ValueError("American odds must be a finite nonzero number") from exc
    if isinstance(american_odds, bool) or not np.isfinite(price) or price == 0:
        raise ValueError("American odds must be a finite nonzero number")
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def _over_masks(
    box: BoxScoreSet, prop: Prop
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prop.side != "over":
        raise ValueError("prop side must be over")
    wins, pushes = _prop_outcome(box, prop)
    return wins, ~wins & ~pushes, pushes


def _resolve_team(box: BoxScoreSet, team: str) -> str:
    if team in box.teams:
        return team
    for side, name in box.metadata.get("team_names", {}).items():
        if name == team:
            return side
    raise ValueError(f"unknown team {team}")


def _prop_outcome(box: BoxScoreSet, prop: Prop) -> tuple[np.ndarray, np.ndarray]:
    side = _resolve_team(box, prop.team)
    player = _normalize_id(prop.player_id)
    try:
        index = box.teams[side].player_ids.index(player)
    except ValueError as exc:
        raise ValueError(f"unknown player {prop.player_id}") from exc
    try:
        line = float(prop.line)
    except (TypeError, ValueError) as exc:
        raise ValueError("prop line must be a number") from exc
    if not np.isfinite(line):
        raise ValueError("prop line must be finite")
    if prop.side not in ("over", "under"):
        raise ValueError("prop side must be over or under")
    values = stat_values(box, side, prop.stat)[:, index]
    win = values > line if prop.side == "over" else values < line
    return win, values == line
