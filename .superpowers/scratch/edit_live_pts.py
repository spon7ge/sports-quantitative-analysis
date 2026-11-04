import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "notebooks/nba/points/live_pts.ipynb"
nb = json.loads(PATH.read_text())


def replace(cell: int, old: str, new: str) -> None:
    source = "".join(nb["cells"][cell]["source"])
    assert source.count(old) == 1, (cell, old[:80], source.count(old))
    nb["cells"][cell]["source"] = source.replace(old, new).splitlines(keepends=True)


def cut(cell: int, start: str, end: str, new: str) -> None:
    """Replace from `start` up to (not including) `end`."""
    source = "".join(nb["cells"][cell]["source"])
    i = source.index(start)
    j = source.index(end, i)
    assert source.count(start) == 1
    nb["cells"][cell]["source"] = (source[:i] + new + source[j:]).splitlines(keepends=True)


replace(0, "4. Resample stored `(minutes, rate)` rank pairs from before the quote date.\n"
           "5. Turn those draws into an integer points distribution and compare it with each book's price.",
        "4. Turn the rate knots into a mean rate and its coefficient of variation.\n"
        "5. Build an integer points distribution from the exported Frank copula and NB2 count model in `copula_spec.json`, and compare it with each book's price.")
replace(0, " The copula is the stored minutes-rate rank pairs, cut off before the quote date.",
        " The copula and count model are the ones fit in `minutes_ppm_copula.ipynb` on the walk-forward OOS pairs. They are fixed, not refit to the quote date.")

replace(1, "from pathlib import Path\nimport sys\n",
        "from hashlib import sha256\nimport json\nfrom pathlib import Path\nimport sys\nimport warnings\n")
replace(1, "import pandas as pd\nfrom IPython.display",
        "import pandas as pd\nimport pyvinecopulib as pv\nfrom IPython.display")
replace(1, "from sqlalchemy import text\n", "from scipy import stats\nfrom sqlalchemy import text\n")
replace(1, "from models.shared.copula import EmpiricalCopula, minutes_tier\n", "")
replace(1, "from models.shared.oos import RATE_CAP\n",
        "from models.shared.points_given_minutes import MINUTES_CENTER, CountModel, conditional_points_pmf\n")
replace(1, "from models.shared.simulate import line_probs, points_pmf\n",
        "from models.shared.simulate import line_probs\n")
replace(1, "SEED = 42\nN_DRAWS = 10_000\nKMAX = 90\n",
        "warnings.filterwarnings(\"ignore\", category=DeprecationWarning, module=\"pyvinecopulib\")\n\n"
        "SEED = 42\nKMAX = 90\nMINUTES_GRID = 50\nRATE_MEAN_GRID = 200\nV_CLIP = 1e-12\n")
replace(1, "OOS_PATH = ROOT / \"data/oos/nba/minutes_rate_oos.parquet\"\n"
           "RANK_PAIRS_PATH = ROOT / \"artifacts/nba/copula/rank_pairs.parquet\"\n",
        "SPEC_PATH = ROOT / \"artifacts/nba/copula/copula_spec.json\"\n")
replace(1, "MINUTE_FILL_COLUMNS = (\"minutes\", \"min\", \"min_sec\")\n",
        "MINUTE_FILL_COLUMNS = (\"minutes\", \"min\", \"min_sec\")\n"
        "MINUTES_KNOTS = [f\"minutes_q_{level:.2f}\" for level in QUANTILE_LEVELS]\n"
        "RATE_KNOTS = [f\"rate_q_{level:.2f}\" for level in QUANTILE_LEVELS]\n"
        "COUNT_FIELDS = (\"log_kappa\", \"beta\", \"log_alpha\", \"beta2\", \"alpha_minutes\", \"alpha_cv\", \"gamma\")\n"
        "LINEAGE_FILES = {\n"
        "    \"minutes_bundle\": MINUTES_MODEL_PATH,\n"
        "    \"ppm_bundle\": RATE_MODEL_PATH,\n"
        "    \"minutes_tails\": MINUTES_TAILS_PATH,\n"
        "    \"rate_tails\": RATE_TAILS_PATH,\n"
        "}\n")

cut(1, "def load_copula(before_date) -> EmpiricalCopula:", "def _american(value):", '''def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_simulator(path: Path = SPEC_PATH) -> dict:
    """Exported copula and NB2 count model from `copula_spec.json`."""
    spec = json.loads(path.read_text())
    block = spec["count_model"]
    bad = [name for name in COUNT_FIELDS if isinstance(block.get(name), str) or block.get(name) is None]
    if bad:
        raise ValueError(f"{path.name} has no numeric {bad}. Rerun the export cell in minutes_ppm_copula.ipynb.")
    if float(block["minutes_center"]) != MINUTES_CENTER:
        raise ValueError(f"spec minutes_center {block['minutes_center']} vs code {MINUTES_CENTER}")
    model = CountModel(**{name: float(block[name]) for name in COUNT_FIELDS})
    direct = not bool(spec["copula"]["used_in_simulation"])
    bicop = None if direct else pv.Bicop.from_json(spec["copula"]["pyvinecopulib_json"])
    stale = []
    for key, local in LINEAGE_FILES.items():
        stored = spec["lineage"]["files"].get(key, {})
        if stored.get("sha256") != file_sha256(local):
            stale.append(key)
    if direct:
        label = f"direct_nb2_gamma_{model.gamma:.3f}"
    else:
        label = f"{spec['copula']['family']}_{float(spec['copula']['parameters'][0]):.3f}_nb2"
    return {
        "spec": spec,
        "model": model,
        "bicop": bicop,
        "use_surprise": direct,
        "label": label,
        "stale": stale,
    }


def rate_moments(frame: pd.DataFrame, rate_tables) -> tuple[np.ndarray, np.ndarray]:
    """Mean rate and std / mean from a midpoint quadrature of the rate sampler."""
    grid_u = (np.arange(RATE_MEAN_GRID) + 0.5) / RATE_MEAN_GRID
    lower, upper = rate_groups(frame["starting"], rate_tables)
    draws = rate_ppf(
        np.tile(grid_u, (len(frame), 1)),
        frame[RATE_KNOTS].to_numpy(dtype=float),
        lower,
        upper,
        rate_tables,
    )
    mean = draws.mean(axis=1)
    return mean, draws.std(axis=1) / np.maximum(mean, 1e-9)


def points_pmfs(sim: pd.DataFrame, simulator: dict, minutes_tables) -> np.ndarray:
    """Points PMF per row, averaging the copula conditional over a minutes PIT grid."""
    u_grid = (np.arange(MINUTES_GRID) + 0.5) / MINUTES_GRID
    lower, upper = minutes_groups(sim["starting"], minutes_tables)
    minutes_at_u = minutes_ppf(
        np.tile(u_grid, (len(sim), 1)),
        sim[MINUTES_KNOTS].to_numpy(dtype=float),
        lower,
        upper,
        minutes_tables,
    )
    return conditional_points_pmf(
        minutes_at_u,
        u_grid,
        sim["rate_hat"].to_numpy(dtype=float),
        simulator["model"],
        kmax=KMAX,
        rate_cv=sim["rate_cv"].to_numpy(dtype=float),
        bicop=simulator["bicop"],
        use_surprise=simulator["use_surprise"],
    )


def nb_ppf(v, mu, alpha) -> np.ndarray:
    size = 1.0 / np.asarray(alpha, dtype=float)
    prob = size / (size + np.asarray(mu, dtype=float))
    q = np.clip(np.asarray(v, dtype=float), V_CLIP, 1.0 - V_CLIP)
    return stats.nbinom.ppf(q, size, prob)


''')

cut(1, "def price_quotes(", "    lookup = {(row.player_id", '''def price_quotes(quoted: pd.DataFrame, matched: pd.DataFrame, simulator: dict, minutes_tables, rate_tables):
    sim = quoted.drop_duplicates(["player_id", "game_id"]).reset_index(drop=True)
    sim["rate_hat"], sim["rate_cv"] = rate_moments(sim, rate_tables)
    pmf = points_pmfs(sim, simulator, minutes_tables)
    sim["pmf_mean"] = pmf @ np.arange(pmf.shape[1], dtype=float)
    sim["pmf_median"] = np.argmax(np.cumsum(pmf, axis=1) >= 0.5, axis=1)
''')
replace(1, '''                "rate_q50": float(player["rate_q_0.50"]),
''', '''                "rate_q50": float(player["rate_q_0.50"]),
                "rate_hat": float(player.rate_hat),
                "rate_cv": float(player.rate_cv),
''')
replace(1, '''                "n_draws": n_draws,
                "copula": "empirical_minutes_rate",
''', '''                "minutes_grid": MINUTES_GRID,
                "copula": simulator["label"],
''')

cut(1, "def illustration_draws(", "def run_slate(", '''def illustration_draws(player: pd.Series, simulator: dict, minutes_tables, line: float, n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    u = rng.random(n)
    w = rng.random(n)
    if simulator["bicop"] is None:
        v = w
    else:
        v = np.asarray(simulator["bicop"].hinv1(np.column_stack([u, w])), dtype=float)
    minute_grid = player[MINUTES_KNOTS].to_numpy(dtype=float).reshape(1, -1)
    lower, upper = minutes_groups(pd.Series([int(player["starting"])]), minutes_tables)
    minutes = minutes_ppf(u.reshape(1, -1), minute_grid, lower, upper, minutes_tables)[0]
    minutes = np.maximum(minutes, 1e-6)
    model = simulator["model"]
    surprise = stats.norm.ppf(u) if simulator["use_surprise"] else None
    mu = model.mean(minutes, float(player["rate_hat"]), surprise)
    alpha = model.alpha(minutes, float(player["rate_cv"]))
    points = nb_ppf(v, mu, alpha).astype(int)
    return pd.DataFrame(
        {
            "u_minutes": u,
            "v_points": v,
            "minutes": minutes,
            "nb2_mean": mu,
            "nb2_alpha": alpha,
            "points": points,
            "over": points > line,
        }
    )


''')

replace(1, "def run_slate(quotes: pd.DataFrame, markets: pd.DataFrame, *, n_draws: int = N_DRAWS) -> dict:",
        "def run_slate(quotes: pd.DataFrame, markets: pd.DataFrame) -> dict:")
cut(1, '    print(f"Sampling the minutes-rate copula', "    card, sim, pmf_table = price_quotes(", '''    simulator = load_simulator()
    spec = simulator["spec"]
    print(
        f"Pricing with {simulator['label']} from {SPEC_PATH.name}, status {spec.get('status')}, "
        f"{MINUTES_GRID}-point minutes grid."
    )
    if spec.get("status_note"):
        print(f"  spec note: {spec['status_note']}")
    if simulator["stale"]:
        print(f"  {', '.join(simulator['stale'])} changed since the copula was fit. Refit the copula on this bundle's OOS pairs.")
    minutes_tables = load_minutes_tails(MINUTES_TAILS_PATH)
    rate_tables = load_rate_tails(RATE_TAILS_PATH)
''')
replace(1, "price_quotes(quoted, matched, copula, minutes_tables, rate_tables, n_draws)",
        "price_quotes(quoted, matched, simulator, minutes_tables, rate_tables)")
replace(1, "illustration_draws(player, copula, minutes_tables, rate_tables, float(example_row.line_score))",
        "illustration_draws(player, simulator, minutes_tables, float(example_row.line_score))")

replace(4, "## Quantiles, copula, simulation", "## Quantiles, copula, count model")
replace(4, "Draws resample `(u_minutes, u_rate)` from rank pairs dated before the slate. Minutes and rate are the quantile functions at those uniforms. Points are `clip(floor(minutes x rate + 0.5), 0, 90)`. The price is the share of that distribution above the line. The median is a separate summary.",
        "The rate knots are summarized as `rate_hat`, the mean of a 200-point quadrature of the rate sampler, and `rate_cv`, its std / mean. Points given minutes are NB2 with mean `exp(log_kappa) x m^beta x exp(beta2 x log(m/24)^2) x rate_hat` and dispersion `exp(log_alpha) x (m/24)^alpha_minutes x rate_cv^alpha_cv`. Minutes PIT `U` and points PIT `V` are joined by the exported Frank copula. The PMF averages the copula conditional of `V` given `U` over a 50-point grid of minutes quantiles, with mass above 90 in the last bin. The price is the share of that PMF above the line, with a push on whole-number lines. The median is a separate summary.\n\n"
        "The copula and count model are read from `copula_spec.json`. The run prints the spec status and flags any bundle or tail file whose hash differs from the one the copula was fit on.")
replace(4, "The six draws under the card are one illustration.",
        "The six draws under the card are one illustration: `U` sets minutes, `V` comes from the copula given `U`, and points are the NB2 quantile at `V`.")

replace(5, "slate = run_slate(quotes, team_lines, n_draws=N_DRAWS)", "slate = run_slate(quotes, team_lines)")
replace(5, 'f"Minutes q50 {quote.minutes_q50:.2f}, rate q50 {quote.rate_q50:.3f}, "',
        'f"Minutes q50 {quote.minutes_q50:.2f}, rate mean {quote.rate_hat:.3f} (cv {quote.rate_cv:.2f}), "')
replace(5, 'c in {"minutes_q50", "rate_q50", "pmf_mean"}', 'c in {"minutes_q50", "rate_q50", "rate_hat", "rate_cv", "pmf_mean"}')

PATH.write_text(json.dumps(nb, indent=1 if PATH.read_text().startswith('{\n "') else 2, ensure_ascii=False) + "\n")
