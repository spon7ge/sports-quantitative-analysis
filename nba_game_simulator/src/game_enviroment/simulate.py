from __future__ import annotations

import json

import numpy as np
import pandas as pd
from game_enviroment.constants import ENVIRONMENT_COLUMNS
from game_enviroment.features import _finite, _line_feats, _team_snapshot
from game_enviroment.games import season_of
from game_enviroment.script import _draw
from game_enviroment.rotation import lineup_overrides
from lineup import LINEUP_GAME_FEATURES, lineup_game_features


def save_model(model, path):
    with open(path, "w") as file:
        json.dump(model, file, indent=2, allow_nan=False)
    print(f"Saved model -> {path}")


def load_model(path):
    with open(path) as file:
        model = json.load(file)
    if model.get("version") != 3:
        raise ValueError(
            "Retrain the model with this version of the script."
        )
    return model


def _resolve_team(model, key):
    teams = model["state"]["teams"]
    if str(key) in teams:
        return int(key)
    for tid, team in teams.items():
        if str(team["abbr"]).upper() == str(key).upper():
            return int(tid)
    raise ValueError(f"Unknown team {key!r}")


def simulate_game(
    model, team, opponent, date, total=None, spread=None,
    n=10000, seed=None, *,
    is_home=True,
    team_lineup=None,
    opponent_lineup=None,
    home_lineup=None,
    away_lineup=None,
    rng=None,
    rotation_features=None,
    rotation_context=None,
):
    """Spread belongs to `team`, negative=favored.

    Reuse row k for every teammate's simulation k.
    """
    if not isinstance(is_home, (bool, np.bool_)):
        raise ValueError("is_home must be true or false")
    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n <= 0
    ):
        raise ValueError("n must be a positive integer")

    for name, value in (("total", total), ("spread", spread)):
        if value is not None and not _finite(value):
            raise ValueError(f"{name} must be finite or None")
    if total is not None and float(total) <= 0:
        raise ValueError("total must be positive")
    if model.get("version") != 3:
        raise ValueError(
            "Retrain the model with this version of the script."
        )

    rng = np.random.default_rng(seed) if rng is None else rng
    st, cfg = model["state"], model["config"]
    hid = _resolve_team(model, team if is_home else opponent)
    aid = _resolve_team(model, opponent if is_home else team)
    if hid == aid:
        raise ValueError("Team and opponent must be different")

    date = pd.Timestamp(date).normalize()
    if pd.isna(date):
        raise ValueError("date must be valid")
    season = season_of(date)
    cutoff = max(
        pd.Timestamp(t["last_date"])
        for t in st["teams"].values()
    )
    if date <= cutoff:
        raise ValueError(
            "Simulation date must be after the training cutoff; "
            "use historical states for backtests."
        )

    ph, rh = _team_snapshot(
        st["teams"][str(hid)],
        date, season,
        cfg["season_carry"], cfg["prior_weight"],
    )
    pa, ra = _team_snapshot(
        st["teams"][str(aid)],
        date, season,
        cfg["season_carry"], cfg["prior_weight"],
    )

    if team_lineup is not None:
        if is_home:
            home_lineup = team_lineup
        else:
            away_lineup = team_lineup
    if opponent_lineup is not None:
        if is_home:
            away_lineup = opponent_lineup
        else:
            home_lineup = opponent_lineup

    for lineup in (home_lineup, away_lineup):
        if lineup is not None and (
            len(lineup) != 5 or len(set(lineup)) != 5
        ):
            raise ValueError(
                "A supplied lineup must have five distinct player IDs"
            )

    home_spread = (
        None if spread is None
        else float(spread) * (1 if is_home else -1)
    )
    row = _line_feats(
        total, home_spread, st["league_pts"]
    )
    row.update(
        home_id=hid,
        away_id=aid,
        lg_poss=st["league_poss"],
        total=np.nan if total is None else float(total),
        home_spread=np.nan if home_spread is None else home_spread,
        pace_h=ph,
        pace_a=pa,
        b2b_h=float(rh <= 1),
        b2b_a=float(ra <= 1),
        rested_h=float(rh >= 3),
        rested_a=float(ra >= 3),
    )

    if LINEUP_GAME_FEATURES:
        if "lineup" not in st:
            raise ValueError("Missing lineup state; retrain the model")
        row.update(lineup_game_features(
            st["lineup"], season, home_lineup, away_lineup
        ))
        for name in LINEUP_GAME_FEATURES:
            if not _finite(row.get(name)):
                row[name] = model["lineup_fill"].get(name, 0.0)

    multi = False
    frame = pd.DataFrame([row])
    if rotation_features is not None:
        if isinstance(rotation_features, dict):
            row.update(lineup_overrides(rotation_features))
            frame = pd.DataFrame([row])
        else:
            if len(rotation_features) != int(n):
                raise ValueError(
                    "rotation_features must be one mapping or one per draw"
                )
            multi = True
            frame = pd.DataFrame([
                {**row, **lineup_overrides(item)}
                for item in rotation_features
            ])
    samples = _draw(model, frame, 1 if multi else int(n), rng)
    index = (0, slice(None)) if multi else (slice(None), 0)
    own, other = (
        ("home", "away") if is_home else ("away", "home")
    )
    out = pd.DataFrame({
        "team_poss": samples[f"{own}_poss"][index],
        "opp_poss": samples[f"{other}_poss"][index],
        "pace": samples["pace"][index],
        "team_pts": samples[f"{own}_pts"][index],
        "opp_pts": samples[f"{other}_pts"][index],
        "game_total": samples["game_total"][index],
        "is_home": float(is_home),
        "n_ot": samples["n_ot"][index],
        "game_minutes": samples["game_minutes"][index],
        "team_margin": (
            samples["margin"][index] * (1 if is_home else -1)
        ),
    })[ENVIRONMENT_COLUMNS]

    out.attrs.update(
        team=st["teams"][str(hid if is_home else aid)]["abbr"],
        opponent=st["teams"][str(aid if is_home else hid)]["abbr"],
        rest_team=rh if is_home else ra,
        rest_opponent=ra if is_home else rh,
    )
    if rotation_context is not None:
        out.attrs["rotation"] = rotation_context
    return out


def print_sim_summary(sim, total, spread):
    a = sim.attrs
    location = (
        f"{a['opponent']} at {a['team']}"
        if bool(sim.is_home.iloc[0])
        else f"{a['team']} at {a['opponent']}"
    )
    print(
        f"\n{location} | total {total} | team spread {spread}"
    )

    columns = [
        "pace", "team_poss", "team_pts",
        "opp_pts", "game_total", "team_margin",
    ]
    summary = sim[columns].quantile(
        [0.05, 0.25, 0.5, 0.75, 0.95]
    ).T
    summary.columns = ["p05", "p25", "p50", "p75", "p95"]
    summary.insert(0, "sd", sim[columns].std())
    summary.insert(0, "mean", sim[columns].mean())
    print(summary.round(2).to_string())
    print(f"P(OT): {sim.n_ot.gt(0).mean():.1%}")
