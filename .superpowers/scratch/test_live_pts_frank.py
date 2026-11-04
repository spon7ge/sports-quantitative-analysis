import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
nb = json.loads((ROOT / "notebooks/nba/points/live_pts.ipynb").read_text())
ns = {"__name__": "__main__"}
exec(compile("".join(nb["cells"][1]["source"]), "live_pts_cell1", "exec"), ns)

simulator = ns["load_simulator"]()
print("label", simulator["label"], "stale", simulator["stale"], "surprise", simulator["use_surprise"])

oos = pd.read_parquet(ROOT / "data/oos/nba/minutes_rate_oos.parquet")
rows = oos.loc[oos["window_id"].eq(5)].sample(300, random_state=0).reset_index(drop=True)
if "starting" not in rows.columns:
    rows["starting"] = (rows["minutes_q_0.50"] > 26).astype(int)
rows["team_name"] = "T"
rows["matchup"] = "A vs. B"
quoted = rows
lines = np.round(rows["minutes_q_0.50"] * rows["rate_q_0.50"]) + 0.5
matched = pd.DataFrame({
    "book": "test",
    "player_name": rows["player_id"].astype(str),
    "line_score": lines,
    "over_american": -110.0,
    "under_american": -110.0,
    "player_id": rows["player_id"],
    "game_id": rows["game_id"],
})
minutes_tables = ns["load_minutes_tails"](ns["MINUTES_TAILS_PATH"])
rate_tables = ns["load_rate_tails"](ns["RATE_TAILS_PATH"])
card, sim, pmf_table = ns["price_quotes"](quoted, matched, simulator, minutes_tables, rate_tables)
assert np.allclose([p.sum() for p in pmf_table.values()], 1.0)
print(card[["line_score", "minutes_q50", "rate_hat", "rate_cv", "pmf_mean", "pmf_median", "sim_over", "sim_push", "model_side", "copula"]].head())

rng = np.random.default_rng(1)
n = 20_000
bicop, model = simulator["bicop"], simulator["model"]
lower, upper = ns["minutes_groups"](sim["starting"], minutes_tables)
u = rng.random((len(sim), n))
w = rng.random((len(sim), n))
v = np.asarray(bicop.hinv1(np.column_stack([u.ravel(), w.ravel()])), dtype=float).reshape(u.shape)
minutes = np.maximum(ns["minutes_ppf"](u, sim[ns["MINUTES_KNOTS"]].to_numpy(float), lower, upper, minutes_tables), 1e-6)
mu = model.mean(minutes, sim["rate_hat"].to_numpy(float)[:, None])
alpha = model.alpha(minutes, sim["rate_cv"].to_numpy(float)[:, None])
points = ns["nb_ppf"](v, mu, alpha)
line_by_key = dict(zip(zip(matched["player_id"], matched["game_id"]), matched["line_score"]))
sim_line = np.array([line_by_key[(p, g)] for p, g in zip(sim["player_id"], sim["game_id"])])
mc_over = (points > sim_line[:, None]).mean(axis=1)
mc_mean = points.mean(axis=1)
quad_over = card.set_index(["player_id", "game_id"]).loc[list(zip(sim["player_id"], sim["game_id"])), "sim_over"].to_numpy()
se = np.sqrt(mc_over * (1 - mc_over) / n)
z = np.abs(quad_over - mc_over) / np.maximum(se, 1e-9)
print(f"P(over) quad vs MC: mean |diff| {np.abs(quad_over - mc_over).mean():.4f}, max {np.abs(quad_over - mc_over).max():.4f}, share > 3 SE {np.mean(z > 3):.3f}")
print(f"mean quad vs MC: mean |diff| {np.abs(sim['pmf_mean'] - mc_mean).mean():.3f}")

player = sim.iloc[0]
print(ns["illustration_draws"](player, simulator, minutes_tables, float(sim_line[0])))
