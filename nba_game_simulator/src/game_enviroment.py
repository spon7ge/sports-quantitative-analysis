#!/usr/bin/env python3
"""
game_enviorment.py  --  Layer 1: simulate the game environment
===============================================================

Fits a game-level possession model on historical player game logs and draws
ONE shared game environment per simulation (possessions, margin, OT) that every
player in that simulation should reuse.  That shared draw is what creates the
natural correlation between player opportunities.

Pipeline
--------
1. Load player game logs (one or many parquet files; any number of seasons).
2. Collapse player rows -> team-game -> one row per game
   (home/away from `matchup`, OT detected from summed minutes).
3. Build strictly PRE-GAME features (no leakage):
     * EWMA team pace tendency (shrunk toward league average, carried across
       seasons with a decay factor)
     * rest / back-to-back flags
     * betting total (relative to the rolling league scoring level) and |spread|
     * rolling league pace prior (handles pace drift between seasons)
4. Fit candidate models on a time-ordered train split, score on a holdout:
     * league_prior   : no-skill baseline
     * ols            : OLS on the features
     * ridge_team     : OLS + shrunken team effects (ridge == random-intercept BLUP)
5. Calibrate a Student-t residual, a margin model, a pace<->game-script link,
   an OT model, and the home-away possession split.
6. Refit the chosen model on all data, save everything to JSON, and expose
   `simulate_game(...)` to draw the shared environment for an upcoming game.

Usage
-----
  # train (directory, glob, or single file)
  python game_enviorment.py --data "data/*.parquet" --model-out game_env_model.json

  # train and immediately simulate a game
  python game_enviorment.py --data data/ --simulate LAL BOS --date 2026-11-10 \
         --total 228.5 --spread -3.5 --n 20000

  # simulate from a saved model (no refit)
  python game_enviorment.py --model-in game_env_model.json \
         --simulate LAL BOS --date 2026-11-10 --total 228.5 --spread -3.5

Spread convention: pass the HOME team's spread in the same sign convention as
the `team_spread` column in your data. The script prints the convention it
detects (the slope of margin on spread: negative slope => negative = favorite).

Known simplifications
---------------------
* OT is drawn independently of the regulation margin (real OT games are close
  games by construction).  Fine for possessions; refine if you need exact
  margin/OT consistency.
* Lineup features are the starters in that game (start_position G/F/C). Their
  pace, offensive rating, and defensive rating come from earlier games only.
* Possessions come from the `team_poss` / `opp_poss` columns of your logs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats

from lineup import (
    LINEUP_GAME_FEATURES,
    attach_lineup_fill,
    build_lineup_features,
    fill_lineup_gaps,
    lineup_game_features,
    merge_lineup_features,
)

REG_MIN = 48.0
OT_MIN = 5.0
REST_CAP = 7

REQUIRED = ["season_year", "game_id", "game_date", "team_id", "team_abbreviation",
            "matchup", "min", "team_poss", "opp_poss", "team_pts", "opp_pts"]
        
OPTIONAL = ["season_type", "player_id", "start_position", "game_total", "team_spread",
            "player_team_spread", "pace", "off_rating", "def_rating",
            "team_pace", "team_off_rating", "team_def_rating"]

# Model features (all known at tip-off). Lineup columns use that game's starters
# and each starter's pace / off rating / def rating from earlier games only.
FEATURES = ["pace_h", "pace_a", "b2b_h", "b2b_a", "rested_h", "rested_a",
            "total_dev", "abs_spread", "total_missing", "spread_missing",
            *LINEUP_GAME_FEATURES]

SPREAD_EDGES = [0.0, 3.0, 6.0, 9.0, 1e9]


# --------------------------------------------------------------------------- #
# Loading and collapsing
# --------------------------------------------------------------------------- #
def _resolve_paths(spec: str) -> list[str]:
    if os.path.isdir(spec):
        paths = sorted(glob.glob(os.path.join(spec, "*.parquet")))
    else:
        paths = sorted(glob.glob(spec)) or ([spec] if os.path.exists(spec) else [])
    if not paths:
        sys.exit(f"No parquet files found for --data {spec!r}")
    return paths


def load_gamelogs(spec: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for p in _resolve_paths(spec):
        cols = set(pq.read_schema(p).names)
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            sys.exit(f"{p} is missing required columns: {missing}")
        use = [c for c in REQUIRED + OPTIONAL if c in cols]
        frames.append(pd.read_parquet(p, columns=use))
        print(f"  loaded {os.path.basename(p)}: {len(frames[-1]):,} rows")
    df = pd.concat(frames, ignore_index=True)
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan
    return df


def build_game_table(df: pd.DataFrame, include_playoffs: bool = False) -> pd.DataFrame:
    """Player rows -> one row per game (home vs away)."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["game_id"] = df["game_id"].astype(str)

    if df["season_type"].notna().any():
        allowed = {"Regular Season"} | ({"Playoffs", "PlayIn", "Play-In"} if include_playoffs else set())
        df = df[df["season_type"].fillna("Regular Season").isin(allowed)]

    df = df.dropna(subset=["team_id"])
    if df["player_id"].notna().any():
        df = df.drop_duplicates(["game_id", "player_id", "team_id"])
    df["spread_any"] = df["team_spread"].fillna(df["player_team_spread"])

    tg = df.groupby(["game_id", "team_id"], as_index=False).agg(
        season_year=("season_year", "first"),
        game_date=("game_date", "first"),
        abbr=("team_abbreviation", "first"),
        matchup=("matchup", "first"),
        poss=("team_poss", "first"),
        opp_poss=("opp_poss", "first"),
        pts=("team_pts", "first"),
        opp_pts=("opp_pts", "first"),
        total=("game_total", "first"),
        spread=("spread_any", "first"),
        team_min=("min", "sum"),
    )
    tg["is_home"] = tg["matchup"].str.contains(r"\bvs\.", regex=True)
    h = tg[tg["is_home"]].drop(columns=["matchup", "is_home"])
    a = tg[~tg["is_home"]].drop(columns=["matchup", "is_home"])
    m = h.merge(a, on="game_id", suffixes=("_h", "_a"))

    g = pd.DataFrame({
        "game_id": m["game_id"],
        "game_date": m["game_date_h"],
        "season_year": m["season_year_h"].astype(str),
        "home_id": m["team_id_h"].astype("int64"),
        "away_id": m["team_id_a"].astype("int64"),
        "home_abbr": m["abbr_h"],
        "away_abbr": m["abbr_a"],
        "home_poss": m["poss_h"].fillna(m["opp_poss_a"]),
        "away_poss": m["poss_a"].fillna(m["opp_poss_h"]),
        "home_pts": m["pts_h"],
        "away_pts": m["pts_a"],
        "game_minutes": (m["team_min_h"] + m["team_min_a"]) / 10.0,   # sum of player min / 5, averaged over teams
        "total": m["total_h"].fillna(m["total_a"]),
        "home_spread": m["spread_h"].fillna(-m["spread_a"]),
    })
    g["n_ot"] = np.clip(np.round((g["game_minutes"] - REG_MIN) / OT_MIN), 0, None).astype(int)
    g["nominal_min"] = REG_MIN + OT_MIN * g["n_ot"]
    g["poss"] = (g["home_poss"] + g["away_poss"]) / 2.0
    g["poss48"] = g["poss"] * REG_MIN / g["nominal_min"]          # possessions per 48 min
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["pts_total"] = g["home_pts"] + g["away_pts"]

    n0 = len(g)
    g = g.dropna(subset=["poss", "home_pts", "away_pts"])
    g = g[(g["game_minutes"] >= 44) & g["poss48"].between(70, 130)]
    if len(g) < n0:
        print(f"  dropped {n0 - len(g)} incomplete/odd games ({n0:,} -> {len(g):,})")
    if g.empty:
        sys.exit("No usable games after cleaning (check `min` and possession columns).")
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pre-game feature construction (streaming => no leakage)
# --------------------------------------------------------------------------- #
def season_of(date) -> str:
    d = pd.Timestamp(date)
    y = d.year if d.month >= 9 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _new_state() -> dict:
    return {"num": 0.0, "den": 0.0, "last_date": None, "last_season": None}


def _effective(st: dict, season: str, carry: float) -> tuple[float, float]:
    """Team pace accumulators, discounted once if a new season has started."""
    if st["last_season"] is not None and st["last_season"] != season:
        return st["num"] * carry, st["den"] * carry
    return st["num"], st["den"]


def _team_snapshot(st: dict, date, season: str, carry: float, prior_w: float) -> tuple[float, float]:
    num, den = _effective(st, season, carry)
    pace = num / (den + prior_w)                      # shrunk toward 0 (= league average)
    if st["last_date"] is None:
        rest = REST_CAP
    else:
        rest = int(np.clip((pd.Timestamp(date) - pd.Timestamp(st["last_date"])).days, 1, REST_CAP))
    return pace, float(rest)


def _line_feats(total, spread, lg_pts) -> dict:
    total = np.nan if total is None else float(total)
    spread = np.nan if spread is None else float(spread)
    tm, sm = not np.isfinite(total), not np.isfinite(spread)
    return {"total_dev": 0.0 if tm else total - lg_pts,
            "abs_spread": 0.0 if sm else abs(spread),
            "total_missing": float(tm), "spread_missing": float(sm)}


def build_features(g: pd.DataFrame, halflife: float, prior_w: float, carry: float,
                   window: int, min_league: int):
    """Return (game table with features, end-of-data state for live prediction)."""
    d = 0.5 ** (1.0 / halflife)
    hid, aid = g["home_id"].to_numpy(), g["away_id"].to_numpy()
    seasons = g["season_year"].to_numpy()
    poss48 = g["poss48"].to_numpy(float)
    ptot = g["pts_total"].to_numpy(float)
    total = g["total"].to_numpy(float)
    spread = g["home_spread"].to_numpy(float)

    # Lineup columns are already on the game table. FEATURES includes them,
    # and this loop would replace those values with the NaNs in `out`.
    cols = [c for c in FEATURES if c not in set(LINEUP_GAME_FEATURES)] + ["lg_poss", "lg_pts"]
    out = {c: np.full(len(g), np.nan) for c in cols}
    teams: dict[int, dict] = {}
    lp, lt = deque(maxlen=window), deque(maxlen=window)

    for date, idxs in g.groupby("game_date", sort=True).indices.items():
        have_lg = len(lp) >= min_league
        lprior = float(np.mean(lp)) if have_lg else np.nan
        tprior = float(np.mean(lt)) if have_lg else np.nan

        if have_lg:                                           # features for today's games
            for i in idxs:
                ph, rh = _team_snapshot(teams.setdefault(hid[i], _new_state()), date, seasons[i], carry, prior_w)
                pa, ra = _team_snapshot(teams.setdefault(aid[i], _new_state()), date, seasons[i], carry, prior_w)
                f = _line_feats(total[i], spread[i], tprior)
                f.update(pace_h=ph, pace_a=pa, b2b_h=float(rh <= 1), b2b_a=float(ra <= 1),
                         rested_h=float(rh >= 3), rested_a=float(ra >= 3))
                for k, v in f.items():
                    out[k][i] = v
                out["lg_poss"][i], out["lg_pts"][i] = lprior, tprior

        for i in idxs:                                        # update state AFTER the day's games
            for tid in (hid[i], aid[i]):
                st = teams.setdefault(tid, _new_state())
                num, den = _effective(st, seasons[i], carry)
                if have_lg:
                    num, den = num * d + (poss48[i] - lprior), den * d + 1.0
                st.update(num=num, den=den, last_date=date, last_season=seasons[i])
        for i in idxs:
            lp.append(poss48[i])
            lt.append(ptot[i])

    g = g.copy()
    for c in cols:
        g[c] = out[c]
    g["y_dev"] = g["poss48"] - g["lg_poss"]

    abbr = {}
    for side in ("home", "away"):
        abbr.update(dict(zip(g[f"{side}_id"], g[f"{side}_abbr"])))
    state = {
        "league_poss": float(np.mean(lp)), "league_pts": float(np.mean(lt)),
        "teams": {str(int(t)): {"abbr": abbr.get(t, str(t)), "num": s["num"], "den": s["den"],
                           "last_date": str(pd.Timestamp(s["last_date"]).date()),
                           "last_season": s["last_season"]} for t, s in teams.items()},
    }
    return g, state


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def design(df: pd.DataFrame, team_ids: list[int] | None, features: list[str] | None = None) -> np.ndarray:
    cols = FEATURES if features is None else list(features)
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    if team_ids:
        idx = {t: j for j, t in enumerate(team_ids)}
        T = np.zeros((len(df), len(team_ids)))
        for r, (h, a) in enumerate(zip(df["home_id"], df["away_id"])):
            for t in (h, a):
                if t in idx:
                    T[r, idx[t]] = 1.0
        X = np.column_stack([X, T])
    return X


def fit_model(name: str, df: pd.DataFrame, team_ids: list[int], lam: float = 0.0) -> dict:
    use_teams = name == "ridge_team"
    tids = team_ids if use_teams else None
    X, y = design(df, tids), df["y_dev"].to_numpy(float)
    pen = np.zeros(X.shape[1])
    if use_teams:
        pen[1 + len(FEATURES):] = lam                           # shrink only team effects
    # lstsq (min-norm) so constant columns, e.g. an always-0 'missing' flag, don't make this singular
    beta = np.linalg.lstsq(X.T @ X + np.diag(pen), X.T @ y, rcond=None)[0]
    return {"name": name, "lam": lam, "team_ids": tids or [], "beta": beta.tolist()}


def predict_dev(model: dict, df: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    if model["name"] == "league_prior":
        return np.zeros(len(df))
    cols = features if features is not None else model.get("features", FEATURES)
    return design(df, model["team_ids"] or None, cols) @ np.asarray(model["beta"])


def choose_lambda(df: pd.DataFrame, team_ids: list[int],
                  grid=(1, 3, 10, 30, 100, 300, 1000)) -> float:
    """Rolling-origin CV over chronological blocks."""
    df = df.sort_values("game_date")
    edges = np.linspace(0, len(df), 6).astype(int)
    best, best_s = grid[0], np.inf
    for lam in grid:
        errs = []
        for b in (2, 3, 4):
            tr, va = df.iloc[:edges[b]], df.iloc[edges[b]:edges[b + 1]]
            m = fit_model("ridge_team", tr, team_ids, lam)
            errs.append(np.mean((va["y_dev"].to_numpy() - predict_dev(m, va)) ** 2))
        s = np.sqrt(np.mean(errs))
        if s < best_s:
            best, best_s = lam, s
    return float(best)


def fit_t(x: np.ndarray) -> tuple[float, float]:
    nu, _, scale = stats.t.fit(x, floc=0.0)
    return float(np.clip(nu, 2.5, 100.0)), float(scale)


def folded_mean(mu, s):
    """E|X| for X ~ N(mu, s^2)."""
    mu = np.asarray(mu, float)
    s = max(float(s), 1e-9)
    return s * np.sqrt(2 / np.pi) * np.exp(-mu ** 2 / (2 * s ** 2)) + mu * (1 - 2 * stats.norm.cdf(-mu / s))


# --------------------------------------------------------------------------- #
# Game-script pieces: margin, pace<->margin link, OT, possession split
# --------------------------------------------------------------------------- #
def fit_margin(df: pd.DataFrame) -> dict:
    ok = df["home_spread"].notna().to_numpy()
    y = df["margin"].to_numpy(float)
    if ok.sum() >= 100:
        x = df.loc[ok, "home_spread"].to_numpy(float)
        b = np.cov(x, y[ok])[0, 1] / np.var(x, ddof=1)
        a = y[ok].mean() - b * x.mean()
        resid = y[ok] - (a + b * x)
    else:
        a, b = float(y.mean()), 0.0
        resid = y - a
    nu, scale = fit_t(resid)
    return {"a": float(a), "b": float(b), "sd": float(resid.std(ddof=1)), "nu": nu, "scale": scale}


def margin_mu(mm: dict, spread) -> float:
    s = np.nan if spread is None else float(spread)
    return mm["a"] + (mm["b"] * s if np.isfinite(s) else 0.0)


def fit_ot(df: pd.DataFrame) -> dict:
    sp = df["home_spread"].abs()
    ot = (df["n_ot"] >= 1).astype(float)
    buckets = []
    for lo, hi in zip(SPREAD_EDGES[:-1], SPREAD_EDGES[1:]):
        m = (sp >= lo) & (sp < hi)
        buckets.append(float(ot[m].mean()) if m.sum() >= 50 else float(ot.mean()))
    n1 = (df["n_ot"] >= 1).sum()
    return {"p_ot_overall": float(ot.mean()), "p_ot_by_spread_bucket": buckets,
            "p_more_given_ot": float((df["n_ot"] >= 2).sum() / n1) if n1 else 0.1}


def fit_split(df: pd.DataFrame) -> dict:
    diff = (df["home_poss"] - df["away_poss"]).round().astype(int)
    vc = diff.value_counts(normalize=True).sort_index()
    return {"values": [int(v) for v in vc.index], "probs": [float(p) for p in vc.values]}


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def _coverage(resid, pred_resid_scale, nu, qs=(0.8, 0.9)):
    return {q: float(np.mean(np.abs(resid) <= stats.t.ppf(0.5 + q / 2, nu) * pred_resid_scale)) for q in qs}


def train(g: pd.DataFrame, state: dict, args) -> dict:
    df = g.dropna(subset=["lg_poss"]).reset_index(drop=True)
    if len(df) < 300:
        print(f"  WARNING: only {len(df)} usable games; estimates will be noisy.")
    team_ids = sorted(set(df["home_id"]) | set(df["away_id"]))

    seasons = sorted(df["season_year"].unique())
    if len(seasons) > args.test_seasons:
        test_mask = df["season_year"].isin(seasons[-args.test_seasons:]).to_numpy()
        split_desc = f"holdout = {', '.join(seasons[-args.test_seasons:])}"
    else:
        cut = int(len(df) * 0.75)
        test_mask = np.arange(len(df)) >= cut
        split_desc = "holdout = last 25% of games by date (single season in data)"
    tr, te = df[~test_mask], df[test_mask]
    print(f"\nGames: {len(df):,} usable | train {len(tr):,} | test {len(te):,} ({split_desc})")
    print(f"Target (possessions/48): mean {df.poss48.mean():.1f}, SD {df.poss48.std():.2f}"
          f" | OT games {100 * (df.n_ot > 0).mean():.1f}%")
    print(f"Lines available for {100 * df.total.notna().mean():.0f}% of games")

    lam = choose_lambda(tr, team_ids) if len(tr) >= 300 else 30.0
    candidates = {
        "league_prior": {"name": "league_prior", "lam": 0.0, "team_ids": [], "beta": []},
        "ols": fit_model("ols", tr, team_ids),
        "ridge_team": fit_model("ridge_team", tr, team_ids, lam),
    }
    yte = te["y_dev"].to_numpy()
    print("\nHoldout performance (possessions per 48):")
    print(f"  {'model':<14}{'RMSE':>8}{'MAE':>8}{'R2 vs prior':>13}")
    rows = {}
    base_mse = np.mean(yte ** 2)
    for nm, m in candidates.items():
        e = yte - predict_dev(m, te)
        rows[nm] = float(np.sqrt(np.mean(e ** 2)))
        print(f"  {nm:<14}{rows[nm]:>8.3f}{np.mean(np.abs(e)):>8.3f}{1 - np.mean(e ** 2) / base_mse:>13.3f}")
    best_name = min(rows, key=lambda k: (rows[k], k != "league_prior"))
    if best_name == "ridge_team":
        print(f"  (ridge lambda chosen by rolling-origin CV: {lam:g})")
    print(f"  -> selected: {best_name}")

    # honest calibration check: t fit on TRAIN residuals, scored on holdout
    m_tr = candidates[best_name]
    e_tr = tr["y_dev"].to_numpy() - predict_dev(m_tr, tr)
    nu_chk, sc_chk = fit_t(e_tr)
    cov = _coverage(yte - predict_dev(m_tr, te), sc_chk, nu_chk)
    print(f"  Student-t residual (train fit): nu={nu_chk:.1f}, scale={sc_chk:.2f}"
          f" | holdout coverage 80%->{cov[0.8]:.1%}, 90%->{cov[0.9]:.1%}")

    # ---- refit on ALL data for the production artifact
    lam_all = choose_lambda(df, team_ids) if best_name == "ridge_team" and len(df) >= 300 else lam
    final = (candidates["league_prior"] if best_name == "league_prior"
             else fit_model(best_name, df, team_ids, lam_all))
    pred_dev = predict_dev(final, df)
    eps0 = df["y_dev"].to_numpy() - pred_dev

    mm = fit_margin(df)
    mu_m = np.array([margin_mu(mm, s) for s in df["home_spread"]])
    z = np.abs(df["margin"].to_numpy(float)) - folded_mean(mu_m, mm["sd"])
    kappa = float(np.cov(z, eps0)[0, 1] / np.var(z, ddof=1))
    eps = eps0 - kappa * z
    nu_p, sc_p = fit_t(eps)

    print("\nFinal model (refit on all data):")
    print(f"  residual SD of possessions/48: {eps0.std(ddof=1):.2f}  (kurtosis {stats.kurtosis(eps0):.2f})")
    print(f"  pace <-> game script: kappa = {kappa:+.4f} poss per point of |margin| above expectation"
          f" (corr {np.corrcoef(z, eps0)[0, 1]:+.3f})")
    print(f"  after removing script effect: t(nu={nu_p:.1f}, scale={sc_p:.2f})")
    print(f"  margin ~ spread: slope {mm['b']:+.3f}, intercept {mm['a']:+.2f}, resid SD {mm['sd']:.2f}"
          f"  => {_spread_convention(mm)}")
    if final["name"] != "league_prior":
        _print_coefs(final, df, eps0)
    _print_qq(eps, nu_p, sc_p)

    model = {
        "version": 1,
        "config": {"halflife": args.halflife, "prior_weight": args.prior_weight,
                   "season_carry": args.season_carry, "league_window": args.league_window},
        "features": FEATURES,
        "model": final,
        "residual_t": {"nu": nu_p, "scale": sc_p},
        "script": {"kappa": kappa},
        "margin": mm,
        "ot": fit_ot(df),
        "split": fit_split(df),
        "holdout_rmse": rows,
        "state": state,
        "trained_games": int(len(df)),
    }
    if args.plots:
        _plots(eps, nu_p, sc_p)
    return model


def _spread_convention(mm: dict) -> str:
    if mm["b"] == 0.0:
        return "not enough spread data to detect sign convention"
    return "NEGATIVE spread = home favorite" if mm["b"] < 0 else "POSITIVE spread = home favorite"


def _print_coefs(model: dict, df: pd.DataFrame, resid: np.ndarray):
    beta = np.asarray(model["beta"])
    names = ["intercept"] + FEATURES
    X = design(df, None)
    n, p = X.shape
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid @ resid) / (n - p))
    print("  coefficients (possessions/48; SEs are OLS-style approximations):")
    for i, nm in enumerate(names):
        print(f"    {nm:<24}{beta[i]:>9.3f}  (se {se[i]:.3f})")
    if model["team_ids"]:
        te = beta[len(names):]
        print(f"    team effects: SD {te.std():.2f}, range [{te.min():+.2f}, {te.max():+.2f}] (lambda={model['lam']:g})")


def _print_qq(eps, nu, scale):
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    emp = np.percentile(eps - np.median(eps), qs)
    fit = stats.t.ppf(np.array(qs) / 100, nu) * scale
    print("  tail check (empirical vs fitted-t quantiles of residual):")
    print("    pct  " + "".join(f"{q:>7}" for q in qs))
    print("    emp  " + "".join(f"{v:>7.1f}" for v in emp))
    print("    fit  " + "".join(f"{v:>7.1f}" for v in fit))


def _plots(eps, nu, scale, path="game_env_diagnostics.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plots)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(eps, bins=40, density=True, alpha=0.6)
    xs = np.linspace(eps.min(), eps.max(), 300)
    ax[0].plot(xs, stats.t.pdf(xs, nu, scale=scale), lw=2)
    ax[0].set_title("Possession residuals vs fitted t")
    stats.probplot(eps, dist=stats.t, sparams=(nu, 0, scale), plot=ax[1])
    ax[1].set_title("QQ vs fitted t")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_model(model: dict, path: str):
    with open(path, "w") as f:
        json.dump(model, f, indent=1, default=_json_default)
    print(f"\nSaved model -> {path}")


def load_model(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_team(model: dict, key) -> int:
    teams = model["state"]["teams"]
    if str(key) in teams:
        return int(key)
    for tid, s in teams.items():
        if s["abbr"].upper() == str(key).upper():
            return int(tid)
    sys.exit(f"Unknown team {key!r}. Known: {sorted(s['abbr'] for s in teams.values())}")


def simulate_game(model: dict, home, away, date, total=None, spread=None,
                  n: int = 10000, seed: int | None = None,
                  home_lineup: list | None = None, away_lineup: list | None = None) -> pd.DataFrame:
    """
    Draw `n` shared game environments for one upcoming game.
    Every player in simulation k must use row k of the returned frame.

    home_lineup and away_lineup are the player ids starting on `date`.
    Their ratings are priors stored on the model. Omit a side and that
    side uses the training-average five.

    Columns: poss48 (regulation-rate possessions), n_ot, game_minutes, game_poss,
             home_poss, away_poss, home_margin
    """
    rng = np.random.default_rng(seed)
    cfg, st = model["config"], model["state"]
    hid, aid = _resolve_team(model, home), _resolve_team(model, away)
    season = season_of(date)

    for t in (hid, aid):
        if pd.Timestamp(date) <= pd.Timestamp(st["teams"][str(t)]["last_date"]):
            print(f"  WARNING: {date} is not after {st['teams'][str(t)]['abbr']}'s last game in the "
                  f"training data ({st['teams'][str(t)]['last_date']}); rest/pace features will be off.")
    ph, rh = _team_snapshot(st["teams"][str(hid)], date, season, cfg["season_carry"], cfg["prior_weight"])
    pa, ra = _team_snapshot(st["teams"][str(aid)], date, season, cfg["season_carry"], cfg["prior_weight"])
    f = _line_feats(total, spread, st["league_pts"])
    f.update(pace_h=ph, pace_a=pa, b2b_h=float(rh <= 1), b2b_a=float(ra <= 1),
             rested_h=float(rh >= 3), rested_a=float(ra >= 3))
    feats = list(model.get("features") or FEATURES)
    if any(str(name).startswith("lineup_") for name in feats):
        lineup_state = st.get("lineup")
        if lineup_state is None:
            raise ValueError("model is missing lineup state; retrain before simulating")
        if home_lineup is None or away_lineup is None:
            print("  WARNING: starting lineup not supplied; lineup features use the training-average five.")
        f.update(lineup_game_features(lineup_state, season, home_lineup, away_lineup))
    row = pd.DataFrame([{**f, "home_id": hid, "away_id": aid}])
    mu_dev = float(predict_dev(model["model"], row, feats)[0])
    mu_poss = st["league_poss"] + mu_dev

    # margin first, then possessions conditional on the game script
    mm, kappa, rt = model["margin"], model["script"]["kappa"], model["residual_t"]
    mu_m = margin_mu(mm, spread)
    margin = mu_m + stats.t.rvs(mm["nu"], scale=mm["scale"], size=n, random_state=rng)
    script = np.abs(margin) - folded_mean(mu_m, mm["sd"])
    poss48 = mu_poss + kappa * script + stats.t.rvs(rt["nu"], scale=rt["scale"], size=n, random_state=rng)

    # overtime
    ot, sp = model["ot"], (abs(float(spread)) if spread is not None and np.isfinite(float(spread)) else None)
    p_ot = ot["p_ot_overall"] if sp is None else ot["p_ot_by_spread_bucket"][
        int(np.searchsorted(SPREAD_EDGES, sp, side="right") - 1)]
    n_ot = (rng.random(n) < p_ot).astype(int)
    for k in range(1, 5):
        m = n_ot == k
        n_ot[m] += (rng.random(m.sum()) < ot["p_more_given_ot"]).astype(int)
    minutes = REG_MIN + OT_MIN * n_ot
    game_poss = poss48 * minutes / REG_MIN

    # split shared game possessions between the teams
    sp_ = model["split"]
    diff = rng.choice(sp_["values"], size=n, p=np.asarray(sp_["probs"]) / np.sum(sp_["probs"]))
    out = pd.DataFrame({"poss48": poss48, "n_ot": n_ot, "game_minutes": minutes, "game_poss": game_poss,
                        "home_poss": game_poss + diff / 2.0, "away_poss": game_poss - diff / 2.0,
                        "home_margin": margin})
    out.attrs.update(mu_poss48=mu_poss, mu_margin=mu_m, home=st["teams"][str(hid)]["abbr"],
                     away=st["teams"][str(aid)]["abbr"], rest_home=rh, rest_away=ra)
    return out


def print_sim_summary(sim: pd.DataFrame, total, spread):
    a = sim.attrs
    print(f"\nGame environment: {a['away']} @ {a['home']}"
          f"  | total {total if total is not None else 'n/a'} | home spread {spread if spread is not None else 'n/a'}"
          f"  | rest H/A {a['rest_home']:.0f}/{a['rest_away']:.0f} days")
    print(f"  expected possessions/48: {a['mu_poss48']:.1f}   expected home margin: {a['mu_margin']:+.1f}")
    q = sim[["poss48", "game_poss", "home_margin"]].quantile([.05, .25, .5, .75, .95]).T
    q.columns = ["p05", "p25", "p50", "p75", "p95"]
    q.insert(0, "sd", sim[["poss48", "game_poss", "home_margin"]].std())
    q.insert(0, "mean", sim[["poss48", "game_poss", "home_margin"]].mean())
    print(q.round(2).to_string())
    print(f"  P(OT) = {(sim.n_ot > 0).mean():.1%}   corr(poss48, |margin|) = "
          f"{np.corrcoef(sim.poss48, sim.home_margin.abs())[0, 1]:+.3f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Layer 1 game-environment (possession) model + simulator")
    ap.add_argument("--data", help="parquet file, directory, or glob of player game logs")
    ap.add_argument("--model-out", default="game_env_model.json")
    ap.add_argument("--model-in", help="load a saved model instead of training")
    ap.add_argument("--test-seasons", type=int, default=1, help="# of most recent seasons held out for evaluation")
    ap.add_argument("--include-playoffs", action="store_true")
    ap.add_argument("--halflife", type=float, default=12.0, help="EWMA half-life in games")
    ap.add_argument("--prior-weight", type=float, default=4.0, help="shrinkage toward league pace (in games)")
    ap.add_argument("--season-carry", type=float, default=0.6, help="weight kept from prior seasons' pace")
    ap.add_argument("--league-window", type=int, default=300, help="# of past games in the league-pace prior")
    ap.add_argument("--min-league-games", type=int, default=30)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--simulate", nargs=2, metavar=("HOME", "AWAY"), help="team abbreviations (or ids)")
    ap.add_argument("--date", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--total", type=float)
    ap.add_argument("--spread", type=float, help="HOME team spread, same convention as your data")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sim-out", help="optionally write simulated draws to this parquet/csv path")
    args = ap.parse_args()

    if args.model_in:
        model = load_model(args.model_in)
    else:
        if not args.data:
            ap.error("--data is required unless --model-in is given")
        print("Loading data...")
        df = load_gamelogs(args.data)
        g = build_game_table(df, args.include_playoffs)
        print(f"  {len(g):,} games | {g.season_year.nunique()} season(s) | "
              f"{g.game_date.min().date()} -> {g.game_date.max().date()}")
        lineups, lineup_state = build_lineup_features(
            df, args.halflife, args.prior_weight, args.season_carry,
            args.league_window, args.include_playoffs,
        )
        g = merge_lineup_features(g, lineups)
        lineup_state = attach_lineup_fill(lineup_state, g)
        g = fill_lineup_gaps(g)
        g, state = build_features(g, args.halflife, args.prior_weight, args.season_carry,
                                  args.league_window, args.min_league_games)
        state["lineup"] = lineup_state
        model = train(g, state, args)
        save_model(model, args.model_out)

    if args.simulate:
        sim = simulate_game(model, args.simulate[0], args.simulate[1], args.date,
                            args.total, args.spread, args.n, args.seed)
        print_sim_summary(sim, args.total, args.spread)
        if args.sim_out:
            (sim.to_csv if args.sim_out.endswith(".csv") else sim.to_parquet)(args.sim_out, index=False)
            print(f"  wrote {args.sim_out}")


if __name__ == "__main__":
    main()