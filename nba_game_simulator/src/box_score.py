"""Layer 6: one box score from a single pass through layers 1–5.

Layer 1 draws the shared game. Layer 3 assigns minutes inside that game's
length. Layer 4 turns possessions and minutes into shots and turnovers.
Layer 5 turns the misses, makes, and turnovers into rebounds, assists,
steals, and blocks. ``assemble_box_score`` stacks those arrays. It does not
draw them again. The same simulation row is one outcome, so a joint prop is
the share of rows where every line clears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from game_enviroment import simulate_game as simulate_environment
from game_state import PRE
from possession_allocation import sample_negbin
from possession_allocation import simulate_game as simulate_shots
from rotations import scale_mu, sim_gaussian
from shared_events import simulate_shared_events

_SHOT_STATS = ("FGA", "FGM", "2PA", "2PM", "3PA", "3PM", "FTA", "FTM", "PTS", "TOV")
_EVENT_STATS = ("AST", "OREB", "DREB", "REB", "STL", "BLK")
_LAYER4_FIELDS = (
    "player_ids",
    "base_weights",
    "available",
    "reference_minutes",
    "p_3pa",
    "p_3p",
    "p_2p",
    "fta_per_fga",
    "ft_pct",
    "tov_rate",
    "assist_propensity",
    "oreb_propensity",
)
_LAYER5_FIELDS = (
    "player_ids",
    "available",
    "reference_minutes",
    "oreb_rate",
    "dreb_rate",
    "creation_rate",
    "assisted_rate",
    "stl_rate",
    "blk_rate",
)
_OPTIONAL_CONTEXT = (
    "role_factors",
    "lineup_factors",
    "opponent_shot_factors",
    "matchup_factors",
    "minutes_exponent",
)


@dataclass(frozen=True)
class Prop:
    team: str
    player_id: str
    stat: str
    line: float


@dataclass(frozen=True)
class PropProbability:
    joint: float
    marginals: tuple[float, ...]
    independent_product: float
    n_sims: int


@dataclass(frozen=True)
class TeamBoxScore:
    player_ids: tuple
    stats: dict[str, np.ndarray]


@dataclass(frozen=True)
class BoxScoreSet:
    teams: dict[str, TeamBoxScore]
    n_sims: int
    environment: pd.DataFrame
    metadata: dict


def simulate_box_scores(
    *,
    environment_model=None,
    home: str,
    away: str,
    date: str,
    rosters: dict[str, dict],
    n_sims: int,
    rng: np.random.Generator,
    total: float | None = None,
    spread: float | None = None,
    home_lineup: list | None = None,
    away_lineup: list | None = None,
    rotation_simulator=None,
    minute_method: str | None = None,
    fga_per_poss: float = 0.88,
    alpha_fga: float = 0.03,
    p_rebound: float = 0.90,
    p_oreb: float = 0.27,
    p_steal: float = 0.55,
    p_block: float = 0.06,
) -> BoxScoreSet:
    """Run layers 1–5 on one generator and keep one row per simulation."""
    if environment_model is None:
        raise ValueError("environment_model is required")
    if (
        isinstance(n_sims, bool)
        or not isinstance(n_sims, (int, np.integer))
        or int(n_sims) <= 0
    ):
        raise ValueError("n_sims must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    for team in (home, away):
        if team not in rosters:
            raise ValueError(f"rosters must include {team}")

    n_sims = int(n_sims)
    if minute_method is None:
        minute_method = "state" if rotation_simulator is not None else "gaussian"

    environment = simulate_environment(
        environment_model,
        home,
        away,
        date,
        total=total,
        spread=spread,
        n=n_sims,
        is_home=True,
        team_lineup=home_lineup,
        opponent_lineup=away_lineup,
        rng=rng,
    )
    minutes = {
        team: _simulate_minutes(
            team,
            rosters[team],
            environment,
            home=home,
            n_sims=n_sims,
            minute_method=minute_method,
            rotation_simulator=rotation_simulator,
            rng=rng,
        )
        for team in (home, away)
    }
    possessions = {
        home: _rounded_possessions(environment["team_poss"]),
        away: _rounded_possessions(environment["opp_poss"]),
    }
    team_inputs = {
        team: {
            "team_fga": sample_negbin(
                possessions[team] * float(fga_per_poss),
                float(alpha_fga),
                rng,
            ),
            "possessions": possessions[team],
        }
        for team in (home, away)
    }
    shots = simulate_shots(
        team_inputs,
        {
            team: _layer4_players(rosters[team], minutes[team])
            for team in (home, away)
        },
        n_sims,
        rng,
    )
    events = simulate_shared_events(
        {
            team: _layer5_team(rosters[team], minutes[team], shots.teams[team])
            for team in (home, away)
        },
        n_sims,
        rng,
        p_rebound=p_rebound,
        p_oreb=p_oreb,
        p_steal=p_steal,
        p_block=p_block,
    )
    teams = {
        team: _assemble(
            rosters[team],
            minutes[team],
            shots.teams[team],
            events.teams[team],
        )
        for team in (home, away)
    }
    return BoxScoreSet(
        teams=teams,
        n_sims=n_sims,
        environment=environment,
        metadata={
            "fga_per_poss": float(fga_per_poss),
            "alpha_fga": float(alpha_fga),
            "minute_method": minute_method,
            "possessions_rounded": True,
            "p_rebound": p_rebound,
            "p_oreb": p_oreb,
            "p_steal": p_steal,
            "p_block": p_block,
        },
    )


def assemble_box_score(
    environment: pd.DataFrame,
    players: dict[str, pd.DataFrame],
    minutes: dict[str, np.ndarray],
    shots: dict[str, dict[str, np.ndarray]],
    events: dict[str, dict[str, np.ndarray]],
) -> BoxScoreSet:
    """Stack one game's minutes, shots, and events. Row ``i`` is one outcome."""
    if not isinstance(environment, pd.DataFrame) or environment.empty:
        raise ValueError("environment must be a non-empty DataFrame")
    if (
        not isinstance(players, dict)
        or not isinstance(minutes, dict)
        or not isinstance(shots, dict)
        or not isinstance(events, dict)
        or set(players) != set(minutes)
        or set(players) != set(shots)
        or set(players) != set(events)
    ):
        raise ValueError(
            "players, minutes, shots, and events must name the same teams"
        )
    if len(players) != 2:
        raise ValueError("a box score requires two teams")
    first = np.asarray(minutes[next(iter(minutes))])
    if first.ndim != 2 or first.shape[0] == 0:
        raise ValueError("MIN must have shape (n_sims, n_players)")
    n_sims = int(first.shape[0])
    if len(environment) != n_sims:
        raise ValueError(f"environment must have one row per simulation ({n_sims})")

    teams: dict[str, TeamBoxScore] = {}
    for name, roster in players.items():
        if not isinstance(roster, pd.DataFrame) or roster.empty:
            raise ValueError(f"{name} players must be a non-empty DataFrame")
        if "player_id" not in roster.columns:
            raise ValueError(f"{name} players need player_id")
        player_ids = tuple(roster["player_id"].tolist())
        if len(set(player_ids)) != len(player_ids):
            raise ValueError(f"{name} lists the same player twice")
        played = _box_matrix(minutes[name], n_sims, len(player_ids), "MIN", whole=False)
        stat_rows = {"MIN": played}
        for stat in _SHOT_STATS:
            if stat not in shots[name]:
                raise ValueError(f"{name} shots need {stat}")
            stat_rows[stat] = _box_matrix(
                shots[name][stat], n_sims, len(player_ids), stat, whole=True
            )
        for stat in _EVENT_STATS:
            if stat not in events[name]:
                raise ValueError(f"{name} events need {stat}")
            stat_rows[stat] = _box_matrix(
                events[name][stat], n_sims, len(player_ids), stat, whole=True
            )
        teams[name] = TeamBoxScore(player_ids=player_ids, stats=stat_rows)
    return BoxScoreSet(
        teams=teams,
        n_sims=n_sims,
        environment=environment.reset_index(drop=True),
        metadata={"assembled": True},
    )


def _box_matrix(
    value,
    n_sims: int,
    n_players: int,
    name: str,
    *,
    whole: bool,
) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if arr.shape != (n_sims, n_players):
        raise ValueError(f"{name} must have shape ({n_sims}, {n_players})")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative and finite")
    if whole and np.any(arr != np.floor(arr)):
        raise ValueError(f"{name} must be a whole number")
    return arr.copy()


def prop_probability(box: BoxScoreSet, props: list[Prop]) -> PropProbability:
    """Fraction of simulations where every prop clears its line.

    An over on 27.5 is a value strictly above 27.5. The independent product
    multiplies the marginals; it equals the joint only when the props do not
    move together inside a simulation.
    """
    if len(props) == 0:
        raise ValueError("props must not be empty")
    hits = [_prop_hits(box, prop) for prop in props]
    together = np.logical_and.reduce(hits)
    marginals = tuple(float(np.mean(column)) for column in hits)
    return PropProbability(
        joint=float(np.mean(together)),
        marginals=marginals,
        independent_product=float(np.prod(marginals)),
        n_sims=box.n_sims,
    )


def format_outcome(
    box: BoxScoreSet,
    sim: int,
    stats: tuple[str, ...] = ("MIN", "PTS", "REB", "AST", "3PM"),
) -> list[dict]:
    """One simulation, both teams, in roster order."""
    if isinstance(sim, bool) or not isinstance(sim, (int, np.integer)):
        raise ValueError("sim must be an integer simulation index")
    if sim < 0 or sim >= box.n_sims:
        raise ValueError(f"sim {sim} is outside 0..{box.n_sims - 1}")
    rows = []
    for team_name, team in box.teams.items():
        for index, player in enumerate(team.player_ids):
            row = {"team": team_name, "player": player}
            for stat in stats:
                row[stat] = team.stats[stat][sim, index]
            rows.append(row)
    return rows


def _rounded_possessions(values: pd.Series) -> np.ndarray:
    rounded = np.rint(np.asarray(values, dtype=float))
    return np.maximum(rounded, 0.0).astype(np.int64)


def _simulate_minutes(
    team: str,
    roster: dict,
    environment: pd.DataFrame,
    *,
    home: str,
    n_sims: int,
    minute_method: str,
    rotation_simulator,
    rng: np.random.Generator,
) -> np.ndarray:
    available = np.asarray(roster["available"], dtype=float)
    expected = np.asarray(roster["expected_minutes"], dtype=float)
    active = available == 1.0
    if not np.any(active) or float(np.sum(expected[active])) <= 0.0:
        raise ValueError(f"{team} has no available minutes")
    game_minutes = environment["game_minutes"].to_numpy(dtype=float)
    team_total = 5.0 * game_minutes
    if rotation_simulator is None:
        drawn = sim_gaussian(
            scale_mu(expected[active], team_total),
            np.asarray(roster["minutes_sd"], dtype=float)[active],
            team_total,
            game_minutes,
            rng,
        )
    else:
        margin = environment["team_margin"].to_numpy(dtype=float)
        if team != home:
            margin = -margin
        drawn = rotation_simulator.simulate(
            _minute_frame(roster, active),
            method=minute_method,
            n=n_sims,
            n_ot=environment["n_ot"].to_numpy(),
            margin=margin,
            team_total=team_total,
            cap=game_minutes,
            rng=rng,
        )
    minutes = np.zeros((n_sims, len(expected)), dtype=float)
    minutes[:, active] = drawn
    return minutes


def _minute_frame(roster: dict, active: np.ndarray) -> pd.DataFrame:
    if roster.get("minute_frame") is not None:
        frame = roster["minute_frame"].reset_index(drop=True)
        if len(frame) != active.shape[0]:
            raise ValueError("minute_frame must have one row per roster player")
        frame = frame.loc[active].reset_index(drop=True)
    else:
        blank = pd.DataFrame({column: [np.nan] for column in PRE})
        frame = pd.concat([blank] * int(active.sum()), ignore_index=True)
        starter = np.asarray(
            roster.get("is_starter", np.zeros(active.shape[0])), dtype=float
        )
        frame["is_starter"] = starter[active]
    if "is_starter" not in frame.columns:
        frame["is_starter"] = 0
    return frame


def _layer4_players(roster: dict, minutes: np.ndarray) -> dict:
    players = {field: roster[field] for field in _LAYER4_FIELDS}
    players["projected_minutes"] = minutes
    for field in ("dispersion", "role_factors", "lineup_factors", "minutes_exponent"):
        if field in roster:
            players[field] = roster[field]
    return players


def _layer5_team(roster: dict, minutes: np.ndarray, shot) -> dict:
    team = {field: roster[field] for field in _LAYER5_FIELDS}
    team["projected_minutes"] = minutes
    team["fga"] = shot.stats["FGA"]
    team["fgm"] = shot.stats["FGM"]
    team["tov"] = shot.stats["TOV"]
    for field in _OPTIONAL_CONTEXT:
        if field in roster:
            team[field] = roster[field]
    return team


def _assemble(roster: dict, minutes: np.ndarray, shot, event) -> TeamBoxScore:
    stats = {"MIN": np.array(minutes, copy=True)}
    for name in _SHOT_STATS:
        stats[name] = np.array(shot.stats[name], copy=True)
    for name in _EVENT_STATS:
        stats[name] = np.array(event.stats[name], copy=True)
    return TeamBoxScore(player_ids=tuple(roster["player_ids"]), stats=stats)


def _prop_hits(box: BoxScoreSet, prop: Prop) -> np.ndarray:
    team = box.teams.get(prop.team)
    if team is None:
        raise ValueError(f"unknown team {prop.team}")
    try:
        index = team.player_ids.index(prop.player_id)
    except ValueError as exc:
        raise ValueError(f"unknown player {prop.player_id}") from exc
    if prop.stat not in team.stats:
        raise ValueError(f"unknown stat {prop.stat}")
    return team.stats[prop.stat][:, index] > prop.line
