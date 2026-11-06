"""
Layer 2: game state, blowout risk, and conditional minutes (simple v1).

Pipeline
  1. Replay every play-by-play game event by event -> game-state table
     (score diff, time remaining, possessions remaining, player/team fouls,
      lineups, timeouts used) + per-player minutes reconstructed from subs.
  2. Empirical "starters on floor" table by lead x time remaining
     (this is the Boston +3 vs +25 picture, straight from the data).
  3. Blowout model: P(|final margin| >= 20) from pre-game team strength.
  4. Minutes model: LightGBM, minutes | pre-game role + game state.
  5. Monte Carlo: simulate game state -> minutes distribution for a player.

Notes / simplifications (deliberate, see bottom of file for TODOs)
  * No spread in the PBP, so team strength = shrunken rolling net rating.
    If you add a `spread` column to the games table, the blowout model
    will use it automatically.
  * Regular season only (gameId prefix 002). Season is read from the gameId
    (002 + YY + game number), so any number of seasons can be passed in.
  * Multi-season: team strength carries over (shrunk), recent seasons get more
    weight, and the holdout is the latest full season (no leakage).
  * The PBP only contains players who actually did something, so true DNPs
    are invisible -> minutes are "conditional on playing".
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

PBP_GLOB = "/mnt/user-data/uploads/*pbp*.parquet"      # one file per season, or one combined file
OUT_DIR = "/mnt/user-data/outputs"
RECENCY_DECAY = 0.7        # sample weight = decay ** (seasons before the latest)
CARRY_GAMES = 15           # how many games' worth of last season's strength carries into a new season
REQUIRED_COLS = {"gameId", "actionId", "clock", "period", "teamId", "personId", "playerName",
                 "isFieldGoal", "scoreHome", "scoreAway", "location", "description",
                 "actionType", "subType"}
BLOWOUT_MARGIN = 20
REG_SECONDS = 2880

NONPLAYER_FOUL = ("Technical", "Defense 3 Second")          # don't count toward 6
NON_TEAM_FOUL = ("Offensive", "Offensive Charge") + NONPLAYER_FOUL
ON_COURT_TYPES = {"Made Shot", "Missed Shot", "Rebound", "Turnover",
                  "Free Throw", "Foul", ""}                  # "" = steal / block rows


# --------------------------------------------------------------------------- #
# 1. Load + replay
# --------------------------------------------------------------------------- #
def load_pbp(pattern=PBP_GLOB):
    import glob
    paths = sorted(glob.glob(pattern)) if isinstance(pattern, str) else list(pattern)
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    frames = []
    for p in paths:
        f = pd.read_parquet(p)
        missing = REQUIRED_COLS - set(f.columns)
        if missing:                                            # older seasons may differ: fail loudly
            raise ValueError(f"{p} is missing columns: {sorted(missing)}")
        frames.append(f)
        print(f"  loaded {p.split('/')[-1]}: {len(f):,} rows")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["gameId", "actionId"])
    df = df[df.gameId.str.startswith("002")].copy()
    df["season"] = 2000 + df.gameId.str[3:5].astype(int)       # 2025 = the 2025-26 season
    m = df.clock.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
    df["rem"] = m[0] * 60 + m[1]                              # sec left in period
    df = df.sort_values(["gameId", "actionId"]).reset_index(drop=True)
    return df


def period_bounds(p):
    """(start_elapsed, length) in seconds of game time."""
    if p <= 4:
        return 720 * (p - 1), 720
    return 2880 + 300 * (p - 5), 300


def sec_remaining_game(p, rem):
    return (4 - p) * 720 + rem if p <= 4 else rem


def infer_starters(rows, team, name2ids, prev_lineup):
    """Players on court at period start = acted (or got subbed out) before
    ever being subbed in."""
    seen, subbed_in, start = [], set(), []
    for r in rows:
        if r["type"] == "Substitution":
            if r["team"] != team:
                continue
            out = r["pid"]
            if out not in subbed_in and out not in start:
                start.append(out)
            cands = name2ids.get((team, r["in_name"]), [])
            cands = [c for c in cands if c not in start and c not in subbed_in] or cands
            if cands:
                subbed_in.add(cands[0])
        elif r["team"] == team and r["on_court_evt"]:
            if r["pid"] not in subbed_in and r["pid"] not in start:
                start.append(r["pid"])
    if len(start) < 5 and prev_lineup:                         # quiet starter: borrow
        for p in prev_lineup:
            if len(start) == 5:
                break
            if p not in start and p not in subbed_in:
                start.append(p)
    return start[:5] if len(start) >= 5 else start


def replay_game(g, avg_poss=100.0):
    """Walk one game event by event. Returns (state_rows, player_rows, game_row, issues)."""
    gid = g.gameId.iat[0]
    teams = g.loc[g.teamId > 0, "teamId"].unique()
    home = g.loc[(g.location == "h") & (g.teamId > 0), "teamId"].iat[0]
    away = [t for t in teams if t != home][0]

    # name -> ids per team (sub rows carry only last names for incoming player)
    name2ids, id2name = defaultdict(list), {}
    for t, pid, nm in g.loc[(g.teamId > 0) & (g.personId > 0), ["teamId", "personId", "playerName"]].drop_duplicates().itertuples(index=False):
        name2ids[(t, nm)].append(pid)
        id2name[pid] = nm

    pseudo = {}                                                # unresolvable bench guys: (pid -> (team, name))
    rows = []
    for r in g.itertuples(index=False):
        is_team_row = r.teamId == 0                           # timeouts / team rebounds
        team = r.personId if is_team_row else r.teamId
        in_name = None
        if r.actionType == "Substitution":
            mm = re.match(r"SUB: (.+) FOR (.+)", r.description)
            in_name = mm.group(1) if mm else None
        evt_ok = (r.actionType in ON_COURT_TYPES and not is_team_row and r.personId > 0
                  and not (r.actionType == "Foul" and any(k in r.subType for k in NONPLAYER_FOUL))
                  and not (r.actionType == "Free Throw" and "Technical" in r.subType))
        rows.append(dict(type=r.actionType, sub=r.subType, team=team, pid=r.personId,
                         period=r.period, rem=r.rem, desc=r.description, in_name=in_name,
                         sh=r.scoreHome, sa=r.scoreAway, is_team_row=is_team_row,
                         on_court_evt=evt_ok, fg=r.isFieldGoal, aid=r.actionId))

    by_period = defaultdict(list)
    for r in rows:
        by_period[r["period"]].append(r)

    state_rows, seconds = [], defaultdict(float)
    starters_g = {}                                            # period-1 lineups
    issues = defaultdict(int)
    score_h = score_a = 0
    pf = defaultdict(int)
    to_used = {home: 0, away: 0}
    poss = {home: 0, away: 0}
    last_miss_team = None
    prev_lineup = {home: None, away: None}

    for p in sorted(by_period):
        prow = by_period[p]
        p_start, p_len = period_bounds(p)
        lineup, on_since = {}, {}
        for t in (home, away):
            st = infer_starters(prow, t, name2ids, prev_lineup[t])
            if len(st) != 5:
                issues["bad_starters"] += 1
            lineup[t] = list(st)
            for pl in st:
                on_since[pl] = p_start
            if p == 1:
                starters_g[t] = tuple(sorted(st))
        team_fouls = {home: 0, away: 0}

        for r in prow:
            el = p_start + (p_len - r["rem"])
            t = r["team"]

            if r["type"] == "Substitution" and t in lineup:
                out = r["pid"]
                cands = name2ids.get((t, r["in_name"]), [])
                cands = [c for c in cands if c not in lineup[t]] or cands
                if cands:
                    pin = cands[0]
                else:                                           # never seen in any event: placeholder id
                    pin = -(1 + len(pseudo) + 1000 * (t % 1000))
                    pseudo[pin] = (t, r["in_name"])
                if out in lineup[t]:
                    lineup[t].remove(out)
                    seconds[out] += el - on_since.pop(out)
                    if pin not in lineup[t]:
                        lineup[t].append(pin)
                        on_since[pin] = el
                else:
                    issues["sub_out_not_on_court"] += 1
            elif r["type"] == "Foul" and r["pid"] > 0 and not any(k in r["sub"] for k in NONPLAYER_FOUL):
                pf[r["pid"]] += 1
                if not any(k in r["sub"] for k in NON_TEAM_FOUL) and t in team_fouls:
                    team_fouls[t] += 1
            elif r["type"] == "Timeout" and t in to_used:
                to_used[t] += 1

            # score (strings, only filled on scoring plays)
            if r["sh"] != "":
                score_h, score_a = int(r["sh"]), int(r["sa"])

            # possessions: FGA + 0.44 FTA + TOV - OREB
            if t in poss:
                if r["fg"] == 1:
                    poss[t] += 1
                elif r["type"] == "Free Throw":
                    poss[t] += 0.44
                elif r["type"] == "Turnover":
                    poss[t] += 1
                elif r["type"] == "Rebound" and t == last_miss_team:
                    poss[t] -= 1
            if r["type"] in ("Missed Shot", "Free Throw") and r["desc"].startswith("MISS"):
                last_miss_team = t
            elif r["type"] not in ("Rebound", "Substitution", "Foul", "Timeout", "Instant Replay", ""):
                last_miss_team = None

            sr = sec_remaining_game(p, r["rem"])
            state_rows.append((
                gid, r["aid"], p, el, sr, score_h - score_a,
                sr / REG_SECONDS * avg_poss if p <= 4 else sr / REG_SECONDS * avg_poss,
                tuple(sorted(lineup[home])), tuple(sorted(lineup[away])),
                tuple(pf[x] for x in sorted(lineup[home])), tuple(pf[x] for x in sorted(lineup[away])),
                team_fouls[home], team_fouls[away], to_used[home], to_used[away]))

        for t in (home, away):                                  # close the period
            for pl, s in on_since.items():
                if pl in lineup[t]:
                    seconds[pl] += p_start + p_len - s
            prev_lineup[t] = list(lineup[t])

    state = pd.DataFrame(state_rows, columns=[
        "gameId", "actionId", "period", "elapsed", "sec_rem", "score_diff", "poss_rem",
        "home_lineup", "away_lineup", "home_pf", "away_pf",
        "home_team_fouls", "away_team_fouls", "home_to_used", "away_to_used"])

    # player-game rows
    pl_rows = []
    pid_team = {pid: t for (t, nm), ids in name2ids.items() for pid in ids}
    for pid, (t, nm) in pseudo.items():
        pid_team[pid] = t; id2name[pid] = nm
    for pid, s in seconds.items():
        if pid not in pid_team:
            continue
        t = pid_team[pid]
        pl_rows.append(dict(gameId=gid, personId=pid, playerName=id2name.get(pid), teamId=t,
                            is_home=int(t == home), seconds=s, minutes=s / 60,
                            starter=int(pid in starters_g.get(t, ())), pf=pf[pid]))
    game = dict(gameId=gid, home=home, away=away, home_pts=score_h, away_pts=score_a,
                margin=score_h - score_a, n_periods=max(by_period),
                poss=(poss[home] + poss[away]) / 2, home_starters=starters_g.get(home),
                away_starters=starters_g.get(away), **dict(issues))
    return state, pd.DataFrame(pl_rows), game


def build_tables(df):
    # league-average possessions from a first pass is overkill; use a sensible default
    # and refine afterward (poss_rem is linear in it).
    states, players, games = [], [], []
    for _, g in df.groupby("gameId", sort=True):
        s, p, gm = replay_game(g)
        states.append(s); players.append(p); games.append(gm)
    states = pd.concat(states, ignore_index=True)
    players = pd.concat(players, ignore_index=True)
    games = pd.DataFrame(games).fillna({"bad_starters": 0, "sub_out_not_on_court": 0})

    # refine possessions-remaining with the actual league pace (reg-time possessions)
    games["season"] = 2000 + games.gameId.str[3:5].astype(int)
    players["season"] = 2000 + players.gameId.str[3:5].astype(int)
    league_poss = games[games.n_periods == 4].groupby("season").poss.mean()   # pace drifts by season
    states["poss_rem"] = states.sec_rem / REG_SECONDS * states.gameId.str[3:5].astype(int).add(2000).map(league_poss)
    return states, players, games, league_poss


# --------------------------------------------------------------------------- #
# 2. Starters-on-floor by game situation
# --------------------------------------------------------------------------- #
def starters_on_floor_table(states, games):
    st = states[states.period <= 4].copy()
    st = st.merge(games[["gameId", "home_starters", "away_starters"]], on="gameId")
    st["dt"] = (st.groupby(["gameId", "period"]).elapsed.shift(-1) - st.elapsed).clip(lower=0).fillna(0)

    def n_on(lineup, starters):
        return len(set(lineup) & set(starters)) if starters is not None else np.nan

    st["h_n"] = [n_on(a, b) for a, b in zip(st.home_lineup, st.home_starters)]
    st["a_n"] = [n_on(a, b) for a, b in zip(st.away_lineup, st.away_starters)]

    # stack both teams from the *team's own* perspective: margin = lead of that team
    h = st[["gameId", "sec_rem", "dt", "h_n"]].assign(lead=st.score_diff).rename(columns={"h_n": "n"})
    a = st[["gameId", "sec_rem", "dt", "a_n"]].assign(lead=-st.score_diff).rename(columns={"a_n": "n"})
    long = pd.concat([h, a]).dropna()
    long["lead_bin"] = pd.cut(long.lead, [-100, -20, -10, -5, 0, 5, 10, 20, 100],
                              labels=["<=-20", "-20..-10", "-10..-5", "-5..0", "0..5", "5..10", "10..20", ">20"])
    long["time_bin"] = pd.cut(long.sec_rem / 60, [0, 2, 5, 8, 12], labels=["0-2m", "2-5m", "5-8m", "8-12m"])
    long = long[long.sec_rem <= 12 * 60]                        # 4th quarter
    long["w"] = long.dt
    long["nw"] = long.n * long.dt
    g = long.groupby(["lead_bin", "time_bin"], observed=True)[["nw", "w"]].sum()
    return (g.nw / g.w).unstack("time_bin").round(2)


# --------------------------------------------------------------------------- #
# 3. Pre-game features + blowout model
# --------------------------------------------------------------------------- #
def add_pregame_features(games, shrink=10, home_adv=2.5):
    """Team strength = shrunken rolling average margin from *prior* games only."""
    games = games.sort_values("gameId").reset_index(drop=True)
    tot, n = defaultdict(float), defaultdict(int)      # strength accumulators (carried across seasons)
    gp = defaultdict(int)                              # games played THIS season
    cur_season, hs, as_, hn, an = None, [], [], [], []
    for r in games.itertuples():
        if r.season != cur_season:                     # new season: keep a shrunk prior, reset gp
            cur_season = r.season
            for t in list(n):
                if n[t] > 0:
                    avg = tot[t] / n[t]
                    n[t] = min(n[t], CARRY_GAMES)
                    tot[t] = avg * n[t]
            gp = defaultdict(int)
        hs.append(tot[r.home] / (n[r.home] + shrink)); hn.append(gp[r.home])
        as_.append(tot[r.away] / (n[r.away] + shrink)); an.append(gp[r.away])
        m = r.margin
        tot[r.home] += m; n[r.home] += 1; gp[r.home] += 1
        tot[r.away] -= m; n[r.away] += 1; gp[r.away] += 1
    games["home_strength"], games["away_strength"] = hs, as_
    games["home_gp"], games["away_gp"] = hn, an
    games["exp_margin"] = games.home_strength - games.away_strength + home_adv   # home perspective
    if "spread" in games.columns:                              # plug-in point
        games["exp_margin"] = -games["spread"]
    games["abs_exp_margin"] = games.exp_margin.abs()
    games["blowout"] = (games.margin.abs() >= BLOWOUT_MARGIN).astype(int)
    return games


def season_weights(seasons):
    return RECENCY_DECAY ** (seasons.max() - seasons)


def split_by_time(df, test_frac=0.25):
    """Multi-season: hold out the latest season. Single season: last test_frac of games."""
    if df.season.nunique() > 1:
        te = df.season == df.season.max()
    else:
        te = df.gameId >= np.sort(df.gameId.unique())[int(df.gameId.nunique() * (1 - test_frac))]
    return df[~te], df[te]


def fit_blowout_model(games, test_frac=0.25):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    g = games[(games.home_gp >= 5) & (games.away_gp >= 5)].sort_values("gameId")
    tr, te = split_by_time(g, test_frac)
    feats = ["abs_exp_margin"]
    clf = LogisticRegression().fit(tr[feats], tr.blowout, sample_weight=season_weights(tr.season))
    p = clf.predict_proba(te[feats])[:, 1]
    base = np.full(len(te), tr.blowout.mean())
    print(f"  blowout = |final margin| >= {BLOWOUT_MARGIN}; base rate {g.blowout.mean():.1%}")
    print(f"  holdout log-loss {log_loss(te.blowout, p):.4f}  (base-rate {log_loss(te.blowout, base):.4f}) | "
          f"Brier {brier_score_loss(te.blowout, p):.4f} (base {brier_score_loss(te.blowout, base):.4f})")
    clf = LogisticRegression().fit(g[feats], g.blowout, sample_weight=season_weights(g.season))
    return clf, feats


# --------------------------------------------------------------------------- #
# 4. Minutes model: minutes | role + game state
# --------------------------------------------------------------------------- #
PRE = ["roll_min", "roll_min_sd", "roll_start", "roll_pf", "is_starter", "is_home", "team_exp_margin",
       "early_season"]
STATE = ["team_final_margin", "abs_final_margin", "went_ot", "pf"]


def build_player_frame(players, games):
    d = players.drop(columns="season", errors="ignore").merge(games[["gameId", "season", "margin", "n_periods", "exp_margin", "home_gp", "away_gp"]], on="gameId")
    d = d[(d.minutes >= 1) & (d.personId > 0)].sort_values(["personId", "gameId"]).copy()
    sign = np.where(d.is_home == 1, 1, -1)
    d["team_exp_margin"] = d.exp_margin * sign
    d["team_final_margin"] = d.margin * sign
    d["abs_final_margin"] = d.margin.abs()
    d["went_ot"] = (d.n_periods > 4).astype(int)
    d["is_starter"] = d.starter
    grp = d.groupby("personId")
    d["roll_min"] = grp.minutes.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    d["roll_min_sd"] = grp.minutes.transform(lambda s: s.shift(1).rolling(10, min_periods=4).std())
    d["roll_start"] = grp.starter.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    d["roll_pf"] = grp.pf.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    # rolling stats run across seasons (continuity), but flag early-season rows where
    # last year's role may be stale (trades, injuries, new coach)
    d["gp_season"] = d.groupby(["personId", "season"]).cumcount()
    d["early_season"] = (d.gp_season < 10).astype(int)
    return d.dropna(subset=["roll_min"]).reset_index(drop=True)


def fit_minutes_model(d, test_frac=0.25):
    import lightgbm as lgb
    tr, te = split_by_time(d, test_frac)
    w = season_weights(tr.season)
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    pre_only = lgb.LGBMRegressor(**params).fit(tr[PRE], tr.minutes, sample_weight=w)
    full = lgb.LGBMRegressor(**params).fit(tr[PRE + STATE], tr.minutes, sample_weight=w)
    mae = lambda m, cols: np.abs(m.predict(te[cols]) - te.minutes).mean()
    print(f"  holdout MAE | rolling-avg baseline {np.abs(te.roll_min - te.minutes).mean():.2f} | "
          f"pre-game GBM {mae(pre_only, PRE):.2f} | + game state {mae(full, PRE + STATE):.2f}")
    # residuals of the conditional model (out-of-sample) -> noise for simulation
    resid = (te.minutes - full.predict(te[PRE + STATE])).values
    if d.season.nunique() > 1:                                  # does older data actually help?
        last = tr[tr.season == tr.season.max()]
        for label, sub, sw in [("latest season only", last, None),
                               ("all seasons, equal weight", tr, None),
                               ("all seasons, recency-weighted", tr, w)]:
            m = lgb.LGBMRegressor(**params).fit(sub[PRE + STATE], sub.minutes,
                                                sample_weight=None if sw is None else sw.loc[sub.index])
            print(f"    training window: {label:<30} holdout MAE {mae(m, PRE + STATE):.3f}")
    full_all = lgb.LGBMRegressor(**params).fit(d[PRE + STATE], d.minutes, sample_weight=season_weights(d.season))
    return full_all, resid, pre_only


# --------------------------------------------------------------------------- #
# 5. Monte Carlo: pre-game inputs -> minutes distribution
# --------------------------------------------------------------------------- #
def simulate_minutes(model, resid, margin_resid, player_row, n=5000, seed=0):
    """Draw game state, push through the conditional minutes model, add residual noise.
    margin_resid: empirical (actual - expected) home margins -> carries the blowout tail."""
    rng = np.random.default_rng(seed)
    team_margin = player_row["team_exp_margin"] + rng.choice(margin_resid, n)
    sim_pf = np.minimum(rng.poisson(max(player_row["roll_pf"], 0.1), n), 6)
    ot = (np.abs(team_margin) < 3) & (rng.random(n) < 0.35)    # crude OT proxy
    X = pd.DataFrame({c: player_row[c] for c in PRE}, index=range(n))
    X["team_final_margin"] = team_margin
    X["abs_final_margin"] = np.abs(team_margin)
    X["went_ot"] = ot.astype(int)
    X["pf"] = sim_pf
    mins = model.predict(X[PRE + STATE]) + rng.choice(resid, n)
    return np.clip(mins, 0, 48 if True else 60), team_margin


# --------------------------------------------------------------------------- #
def main():
    print("Loading + replaying games...")
    df = load_pbp()
    states, players, games, league_poss = build_tables(df)
    print(f"  {games.gameId.nunique()} games across seasons {sorted(games.season.unique())}, {len(states):,} state rows")
    print("  league possessions per team-game by season:", league_poss.round(1).to_dict())

    # --- validation of lineup reconstruction
    reg = games[games.n_periods == 4]
    team_min = players[players.gameId.isin(reg.gameId)].groupby(["season", "gameId", "teamId"]).minutes.sum()
    for season, tm in team_min.groupby("season"):               # check EACH season: formats can differ
        print(f"  VALIDATION {season}: team minutes mean {tm.mean():.1f} (target 240), "
              f"within +/-2: {((tm - 240).abs() <= 2).mean():.1%}")
    print(f"  games with a bad-starter period: {(games.bad_starters > 0).mean():.1%} | "
          f"sub-out-not-on-court events per game: {games.sub_out_not_on_court.mean():.2f}")

    print("\nAvg # of a team's starters on floor, 4th quarter (rows: team's own lead; cols: time left)")
    tbl = starters_on_floor_table(states, games)
    print(tbl.to_string())

    print("\nBlowout model")
    games = add_pregame_features(games)
    print("  blowout rate by season:", games.groupby("season").blowout.mean().round(3).to_dict())
    clf, bfeats = fit_blowout_model(games)
    games["p_blowout"] = clf.predict_proba(games[bfeats])[:, 1]
    for x in (0, 3, 6, 10):
        print(f"  |exp margin| = {x:>2} -> P(blowout) = {clf.predict_proba(pd.DataFrame({'abs_exp_margin': [x]}))[0, 1]:.1%}")

    print("\nMinutes model")
    d = build_player_frame(players, games)
    model, resid, _ = fit_minutes_model(d)
    margin_resid = (games.loc[(games.home_gp >= 5) & (games.away_gp >= 5), "margin"]
                    - games.loc[(games.home_gp >= 5) & (games.away_gp >= 5), "exp_margin"]).values
    print(f"  margin sd around expectation: {margin_resid.std():.1f}")

    # --- demo: a ~34-36 min starter on a big favorite vs. a pick'em
    last = d.sort_values("gameId").groupby("personId").tail(1)
    cand = last[(last.roll_min.between(33, 37)) & (last.is_starter == 1)]
    print("\nSimulated minutes distribution (5,000 sims), starters averaging ~35 min")
    print(f"{'player':<18}{'fav by':>7}{'mean':>7}{'P(<32)':>8}{'P(<30)':>8}{'p10':>6}{'p90':>6}")
    for _, r in cand.head(3).iterrows():
        for fav in (-8.0, 0.0, 10.0):
            row = r.copy(); row["team_exp_margin"] = fav
            m, _ = simulate_minutes(model, resid, margin_resid, row)
            print(f"{str(r.playerName):<18}{fav:>7.0f}{m.mean():>7.1f}{(m < 32).mean():>8.1%}"
                  f"{(m < 30).mean():>8.1%}{np.percentile(m, 10):>6.1f}{np.percentile(m, 90):>6.1f}")

    # --- save
    states.to_parquet(f"{OUT_DIR}/game_states.parquet")
    games.to_parquet(f"{OUT_DIR}/games.parquet")
    d.to_parquet(f"{OUT_DIR}/player_games.parquet")
    tbl.to_csv(f"{OUT_DIR}/starters_on_floor.csv")
    return states, players, games, d, model, resid, margin_resid


if __name__ == "__main__":
    main()