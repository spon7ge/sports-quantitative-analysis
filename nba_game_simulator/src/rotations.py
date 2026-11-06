"""
Layer 3: constrained team minutes simulator (simple v1).

Every simulated team-game must satisfy   sum(minutes) = 240 + 25 * (# overtimes),
and no player can exceed 48 (53 in OT) or go below 0.

Four simulators, compared on a time-ordered holdout:
  prototype  independent Normal, then rescale to 240            (your starter code)
  dirichlet  Dirichlet shares around expected minutes
  gaussian   independent Normal, then CONDITION on the team sum  (exact; keeps each
             player's variance instead of inflating/deflating it like rescaling)
  state      gaussian + simulated game state (blowout margin, OT, fouls) pushed
             through the Layer-2 conditional minutes model first

Backtest A: 2025-26 (minutes reconstructed from PBP), 55/20/25 time split
            train / calibrate / test.
Forward  B: 2026-27 player_gamelogs (official minutes, official starters, real
            spreads), using 2025-26 history for the rolling features.

Simplifications: roster = players who actually played (DNPs invisible), no lineup /
substitution transitions yet (that is Layer 3b), no coach effects yet.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import layer2_game_state as L2

OUT = "/mnt/user-data/outputs"
GAMELOGS = "/mnt/user-data/uploads/player_gamelogs.parquet"
PRE, STATE = L2.PRE, L2.STATE
CAP_REG, CAP_OT = 48.0, 53.0
MU_BINS = [0, 8, 16, 24, 30, 99]                      # for the sigma lookup table
GBM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


# --------------------------------------------------------------------------- #
# Constraint machinery
# --------------------------------------------------------------------------- #
def constrain(raw, w, total, cap, iters=6):
    """Project simulated minutes onto {sum = total, 0 <= x <= cap}.
    Step 1 is the exact Gaussian conditioning: spread the gap in proportion to each
    player's variance (w = sigma^2). Then clip and redistribute among players not at a bound."""
    m = raw + (total - raw.sum(1))[:, None] * (w / w.sum())[None, :]
    for _ in range(iters):
        m = np.clip(m, 0, cap[:, None])
        free = (m > 1e-9) & (m < cap[:, None] - 1e-9)
        wf = (w[None, :] * free)
        denom = wf.sum(1)
        gap = total - m.sum(1)
        m = m + np.where(denom > 0, gap / np.where(denom > 0, denom, 1), 0)[:, None] * wf
    return np.clip(m, 0, cap[:, None])


def scale_mu(mu, total):
    """Expected minutes rescaled so they sum to the (per-sim) team total."""
    mu = np.atleast_2d(mu)
    return mu * (total / mu.sum(1))[:, None]


def sim_prototype(mu, sd, total, cap, rng):
    n = len(total)
    raw = np.clip(rng.normal(mu, sd, (n, len(mu))), 0, None)
    return raw / raw.sum(1, keepdims=True) * total[:, None]


def sim_dirichlet(mu, alpha, total, cap, rng):
    shares = np.asarray(mu) / np.sum(mu)
    x = rng.dirichlet(alpha * shares, size=len(total)) * total[:, None]
    return constrain(x, np.ones(len(mu)), total, cap)


def team_shock(mean_mat, starter, tau, rng):
    """Shared game-flow shock: starters move together, bench takes the opposite side.
    Zero-sum, so the team total is untouched. tau is calibrated on held-out games."""
    ns = int(starter.sum())
    if tau <= 0 or ns == 0 or ns == len(starter):
        return mean_mat
    z = rng.normal(0, tau, len(mean_mat))
    bench_w = mean_mat[:, ~starter] / mean_mat[:, ~starter].sum(1, keepdims=True)
    out = mean_mat.copy()
    out[:, starter] += z[:, None]
    out[:, ~starter] -= (z * ns)[:, None] * bench_w
    return out


def sim_gaussian(mean_mat, sd, total, cap, rng):
    raw = mean_mat + rng.normal(0, 1, mean_mat.shape) * sd[None, :]
    return constrain(raw, sd ** 2, total, cap)


# --------------------------------------------------------------------------- #
# The simulator object (fit once, then call .simulate(roster))
# --------------------------------------------------------------------------- #
class RotationSimulator:
    def __init__(self, pre_model, cond_model, sigma_pre, sigma_cond, alpha,
                 margin_resid, p_ot, fallback_mu, fallback_sd):
        self.pre, self.cond = pre_model, cond_model
        self.sig_pre, self.sig_cond = sigma_pre, sigma_cond
        self.alpha, self.margin_resid, self.p_ot = alpha, margin_resid, p_ot
        self.fb_mu, self.fb_sd = fallback_mu, fallback_sd
        self.tau = {}                                  # team-level shock size per method (fit_tau)

    def _sigma(self, mu, table):
        return table[np.clip(np.digitize(mu, MU_BINS) - 1, 0, len(table) - 1)]

    def _draw_ot(self, n, rng):
        ot = rng.random(n) < self.p_ot
        total = 240 + 25 * ot
        return ot, total.astype(float), np.where(ot, CAP_OT, CAP_REG)

    def pregame_mu(self, roster):
        known = roster[PRE].notna().all(axis=1).values
        mu = np.full(len(roster), self.fb_mu)
        if known.any():
            mu[known] = self.pre.predict(roster.loc[known, PRE])
        return np.clip(mu, 0.5, 46), known

    def simulate(self, roster, method="state", n=2000, seed=0):
        """roster: one row per available player, with PRE feature columns
        (NaN features -> fallback bench-player assumption). Returns (n, k) minutes."""
        rng = np.random.default_rng(seed)
        mu, known = self.pregame_mu(roster)
        ot, total, cap = self._draw_ot(n, rng)
        sd_pre = np.where(known, self._sigma(mu, self.sig_pre), self.fb_sd)

        if method == "prototype":
            return sim_prototype(mu, sd_pre, total, cap, rng)
        if method == "dirichlet":
            return sim_dirichlet(mu, self.alpha, total, cap, rng)
        starter = roster["is_starter"].values == 1
        if method == "gaussian":
            m = team_shock(scale_mu(mu, total), starter, self.tau.get("gaussian", 0.0), rng)
            return sim_gaussian(m, sd_pre, total, cap, rng)

        # --- state-aware: draw margin / OT / fouls, get mu | state from the Layer-2 model
        base = roster.iloc[0]
        margin = base["team_exp_margin"] + rng.choice(self.margin_resid, n)
        margin = np.where(ot, rng.integers(-3, 4, n), margin)           # OT games are close
        k = int(known.sum())
        mu_state = np.tile(mu, (n, 1))
        if k:
            Xk = roster.loc[known, PRE].reset_index(drop=True)
            X = pd.concat([Xk] * n, ignore_index=True)
            sim_idx = np.repeat(np.arange(n), k)
            X["team_final_margin"] = margin[sim_idx]
            X["abs_final_margin"] = np.abs(margin)[sim_idx]
            X["went_ot"] = ot.astype(int)[sim_idx]
            X["pf"] = np.minimum(rng.poisson(np.tile(np.maximum(Xk["roll_pf"].values, 0.1), n)), 6)
            mu_state[:, known] = np.clip(self.cond.predict(X[PRE + STATE]).reshape(n, k), 0, 46)
        mu_state = team_shock(scale_mu(mu_state, total), starter, self.tau.get("state", 0.0), rng)
        sd = np.where(known, self._sigma(mu, self.sig_cond), self.fb_sd)
        return sim_gaussian(mu_state, sd, total, cap, rng)


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def load_inputs():
    games = pd.read_parquet(f"{OUT}/games.parquet")
    players = pd.read_parquet(f"{OUT}/players.parquet")
    d = pd.read_parquet(f"{OUT}/player_games.parquet")
    feat_cols = ["gameId", "personId"] + [c for c in PRE + STATE if c not in players.columns]
    r = players.merge(d[feat_cols], on=["gameId", "personId"], how="left")
    # team-game level context for rows without player history
    gi = games.set_index("gameId")
    sign = np.where(r.is_home == 1, 1, -1)
    r["team_exp_margin"] = r.gameId.map(gi.exp_margin) * sign
    r["is_starter"] = r.starter
    return games, r


def fit_simulator(r, games):
    """Time-ordered 55/20/25 split by game. Returns simulator + split frames."""
    gids = np.sort(games.gameId.unique())
    c1, c2 = gids[int(len(gids) * .55)], gids[int(len(gids) * .75)]
    has = r[PRE + STATE].notna().all(axis=1)
    tr, ca, te = r[has & (r.gameId < c1)], r[has & r.gameId.between(c1, c2, inclusive="left")], r[r.gameId >= c2]

    pre = lgb.LGBMRegressor(**GBM).fit(tr[PRE], tr.minutes)
    cond = lgb.LGBMRegressor(**GBM).fit(tr[PRE + STATE], tr.minutes)

    # calibrate noise on the calibration slice (out-of-sample residuals)
    def sigma_table(model, cols):
        mu = model.predict(ca[cols]); res = ca.minutes.values - mu
        b = np.clip(np.digitize(mu, MU_BINS) - 1, 0, len(MU_BINS) - 2)
        return np.array([res[b == i].std() if (b == i).sum() > 30 else res.std() for i in range(len(MU_BINS) - 1)])
    sig_pre, sig_cond = sigma_table(pre, PRE), sigma_table(cond, PRE + STATE)

    # Dirichlet concentration by method of moments on calibration team-games
    num = den = 0.0
    for _, g in ca.groupby(["gameId", "teamId"]):
        mu = pre.predict(g[PRE]); mu = mu / mu.sum() * 240
        s = mu / 240
        num += (240 ** 2 * s * (1 - s)).sum(); den += ((g.minutes.values - mu) ** 2).sum()
    alpha = num / den - 1

    gm = games[(games.home_gp >= 5) & (games.away_gp >= 5) & (games.gameId < c2)]
    margin_resid = (gm.margin - gm.exp_margin).values
    p_ot = (games[games.gameId < c2].n_periods > 4).mean()
    nofeat = r[~has & (r.gameId < c1)]
    sim = RotationSimulator(pre, cond, sig_pre, sig_cond, alpha, margin_resid, p_ot,
                            fallback_mu=nofeat.minutes.median(), fallback_sd=nofeat.minutes.std())
    fit_tau(sim, r[(r.gameId >= c1) & (r.gameId < c2)])
    print(f"  sigma by expected-minutes bin {MU_BINS[:-1]}: pregame {sig_pre.round(1)}, "
          f"given game state {sig_cond.round(1)}")
    print(f"  Dirichlet alpha {alpha:.0f} | P(OT) {p_ot:.1%} | fallback bench player: "
          f"{sim.fb_mu:.1f} +/- {sim.fb_sd:.1f} min")
    return sim, te


def fit_tau(sim, calib, methods=("gaussian", "state"), n=300, max_games=250):
    """Size of the shared starter/bench shock = whatever team-level variance in starter minutes
    the independent noise fails to produce, measured on the calibration slice."""
    groups = [(k, g) for k, g in calib.groupby(["gameId", "teamId"]) if 6 <= len(g) and g.is_starter.sum() > 0]
    groups = groups[::max(1, len(groups) // max_games)]
    for m in methods:
        sim.tau[m] = 0.0
        sq, var = [], []
        for (gid, tid), g in groups:
            g = g.reset_index(drop=True)
            st = g.is_starter.values == 1
            x = sim.simulate(g, m, n=n, seed=7)
            s_avg = x[:, st].mean(1)
            sq.append((g.minutes.values[st].mean() - s_avg.mean()) ** 2)
            var.append(s_avg.var())
        sim.tau[m] = float(np.sqrt(max(np.mean(sq) - np.mean(var), 0)))
    print(f"  team-level shock tau (min): {sim.tau}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def crps_samples(x, y):
    """x: (n_sims, k) samples, y: (k,) actuals -> per-player CRPS."""
    n = x.shape[0]
    xs = np.sort(x, axis=0)
    term2 = (2 / n ** 2) * ((2 * np.arange(1, n + 1) - n - 1)[:, None] * xs).sum(0)
    return np.abs(x - y).mean(0) - 0.5 * term2


def evaluate(sim, test, methods=("prototype", "dirichlet", "gaussian", "state"), n=800, label=""):
    rows, dev = [], {m: [] for m in methods}
    act_dev = []
    for (gid, tid), g in test.groupby(["gameId", "teamId"]):
        if len(g) < 6:
            continue
        g = g.reset_index(drop=True)
        y = g.minutes.values
        starter = g.is_starter.values == 1
        mu_pre, _ = sim.pregame_mu(g)
        mu_pre = mu_pre / mu_pre.sum() * y.sum()
        act_dev.append((y[starter].mean() - mu_pre[starter].mean(), y[~starter].mean() - mu_pre[~starter].mean()))
        for m in methods:
            x = sim.simulate(g, m, n=n, seed=hash((gid, tid)) % 2 ** 31)
            q10, q90 = np.percentile(x, [10, 90], axis=0)
            cr = crps_samples(x, y)
            for j in range(len(g)):
                rows.append((m, starter[j], cr[j], q10[j] <= y[j] <= q90[j], abs(np.median(x[:, j]) - y[j]),
                             (x[:, j] < y[j]).mean() + 0.5 * (x[:, j] == y[j]).mean()))
            dev[m].append((x[0][starter].mean() - mu_pre[starter].mean(), x[0][~starter].mean() - mu_pre[~starter].mean()))
    df = pd.DataFrame(rows, columns=["method", "starter", "crps", "cover80", "mae", "pit"])
    summ = df.groupby("method").agg(crps=("crps", "mean"), mae=("mae", "mean"), cover80=("cover80", "mean")).loc[list(methods)]
    for flag, name in ((True, "cover80_starters"), (False, "cover80_bench")):
        summ[name] = df[df.starter == flag].groupby("method").cover80.mean()
    # does the simulator reproduce how starters and bench trade minutes (blowouts etc.)?
    a = np.array(act_dev)
    summ["starter_avg_sd"] = [np.std(np.array(dev[m])[:, 0]) for m in methods]
    summ["corr_starter_bench"] = [np.corrcoef(np.array(dev[m]).T)[0, 1] for m in methods]
    print(f"\n{label}: {len(act_dev)} team-games, {len(df) // len(methods)} player-games")
    print(summ.round(3).to_string())
    print(f"  ACTUAL starter-avg sd {a[:, 0].std():.3f} | starter/bench deviation corr {np.corrcoef(a.T)[0, 1]:.3f}")
    return summ


# --------------------------------------------------------------------------- #
# Forward test on 2026-27 gamelogs
# --------------------------------------------------------------------------- #
def build_forward_rosters(players, d):
    gl = pd.read_parquet(GAMELOGS)
    gl = gl[gl.season_type == "Regular Season"].copy()
    gl = gl.drop(columns=["minutes"]).rename(columns={"player_id": "personId", "game_id": "gameId", "team_id": "teamId", "min": "minutes"})
    gl["personId"] = gl.personId.astype("int64"); gl["teamId"] = gl.teamId.astype("int64")
    gl["starter"] = (gl.start_position.fillna("") != "").astype(int)
    gl["is_home"] = gl.matchup.str.contains("vs.").astype(int)
    # history = 2025-26 (PBP-reconstructed) + 2026-27 (official logs), same definition as Layer 2
    hist = pd.concat([d[["personId", "gameId", "minutes", "starter", "pf"]],
                      gl[["personId", "gameId", "minutes", "starter", "pf"]]])
    hist = hist[hist.minutes >= 1].sort_values(["personId", "gameId"])
    grp = hist.groupby("personId")
    hist["roll_min"] = grp.minutes.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    hist["roll_min_sd"] = grp.minutes.transform(lambda s: s.shift(1).rolling(10, min_periods=4).std())
    hist["roll_start"] = grp.starter.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    hist["roll_pf"] = grp.pf.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    gl = gl.merge(hist[["personId", "gameId", "roll_min", "roll_min_sd", "roll_start", "roll_pf"]],
                  on=["personId", "gameId"], how="left")
    gl["is_starter"] = gl.starter
    gl["early_season"] = 1                                     # first two weeks of the season
    gl["team_exp_margin"] = -gl.player_team_spread             # spread < 0 = favourite (checked: corr w/ +/- is -0.45)
    gl = gl.dropna(subset=["team_exp_margin"])                 # keep games that have a real line
    return gl


# --------------------------------------------------------------------------- #
def injury_demo(sim, roster):
    """Remove the top-minutes player; show who absorbs the minutes."""
    base = sim.simulate(roster, "gaussian", n=4000, seed=1).mean(0)
    star = int(np.argmax(base))
    r2 = roster.drop(roster.index[star]).reset_index(drop=True)
    out = sim.simulate(r2, "gaussian", n=4000, seed=1).mean(0)
    tbl = pd.DataFrame({"player": roster.playerName.values, "with": base})
    tbl = tbl.merge(pd.DataFrame({"player": r2.playerName.values, "without": out}), on="player", how="left")
    tbl["delta"] = tbl["without"] - tbl["with"]
    return roster.playerName.iloc[star], tbl.dropna().sort_values("delta", ascending=False).head(5)


def main():
    print("Fitting on 2025-26 (time-ordered 55/20/25 split)...")
    games, r = load_inputs()
    sim, test = fit_simulator(r, games)

    # sanity: constraints always hold
    g0 = test[(test.gameId == test.gameId.iloc[0]) & (test.teamId == test.teamId.iloc[0])].reset_index(drop=True)
    x = sim.simulate(g0, "state", n=2000)
    print(f"  constraint check: team-sum in [{x.sum(1).min():.2f}, {x.sum(1).max():.2f}], "
          f"player min {x.min():.1f}, max {x.max():.1f}")

    evaluate(sim, test, label="A) 2025-26 holdout (last 25% of games)")

    print("\nB) Forward test: 2026-27 official logs, real spreads")
    d = pd.read_parquet(f"{OUT}/player_games.parquet")
    fwd = build_forward_rosters(r, d)
    print(f"  {fwd.gameId.nunique()} games with a line, {len(fwd)} player-games, "
          f"{fwd[PRE].notna().all(axis=1).mean():.0%} with rolling history")
    evaluate(sim, fwd, label="B) 2026-27 forward test")

    # injury-replacement demo on the first forward team-game with a full rotation
    g = fwd[(fwd.gameId == fwd.gameId.iloc[0]) & (fwd.teamId == fwd.teamId.iloc[0])].reset_index(drop=True)
    g["playerName"] = g.player_name
    star, tbl = injury_demo(sim, g)
    print(f"\nInjury demo ({g.team_abbreviation.iloc[0]}): remove {star} -> who absorbs the minutes")
    print(tbl.round(1).to_string(index=False))

    # prop-style output for one starter
    x = sim.simulate(g, "state", n=5000)
    j = int(np.argmax(sim.pregame_mu(g)[0]))
    p = g.player_name.iloc[j]
    print(f"\n{p}: mean {x[:, j].mean():.1f}, sd {x[:, j].std():.1f}, P(<32) {(x[:, j] < 32).mean():.1%}, "
          f"P(<28) {(x[:, j] < 28).mean():.1%}, 10-90%: {np.percentile(x[:, j], 10):.1f}-{np.percentile(x[:, j], 90):.1f}")
    return sim


if __name__ == "__main__":
    main()