"""Write the numeric NB2 log_alpha into copula_spec.json.

The spec export overwrote the fitted log_alpha with its formula string. This
reruns simulator_checks cells 2 and 5 (which refit the count model on the same
pairs and check every other coefficient against the spec to 1e-4) and stores
the refit value.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
nb = json.loads((ROOT / "notebooks/nba/copula/simulator_checks.ipynb").read_text())
src = []
for index in (2, 5):
    lines = "".join(nb["cells"][index]["source"]).splitlines()
    src.append("\n".join(
        line for line in lines
        if not line.startswith("%") and 'spec["count_model"]["log_alpha"]' not in line
    ))
ns = {"__name__": "__main__", "display": lambda *a, **k: None}
import os

os.chdir(ROOT)
for block in src:
    exec(compile(block, "simulator_checks", "exec"), ns)

count_all = ns["count_all"]
spec_path = ROOT / "artifacts/nba/copula/copula_spec.json"
spec = json.loads(spec_path.read_text())
block = spec["count_model"]
formula = block["log_alpha"]
if isinstance(formula, str):
    block["log_alpha_formula"] = formula
block["log_alpha"] = float(count_all.log_alpha)
spec["count_model"] = {
    **{k: block[k] for k in ("log_kappa", "beta", "log_alpha", "beta2", "alpha_minutes", "alpha_cv", "gamma", "minutes_center")},
    **{k: v for k, v in block.items() if k not in ("log_kappa", "beta", "log_alpha", "beta2", "alpha_minutes", "alpha_cv", "gamma", "minutes_center")},
}
spec_path.write_text(json.dumps(spec, indent=2))
print("log_alpha", count_all.log_alpha)
