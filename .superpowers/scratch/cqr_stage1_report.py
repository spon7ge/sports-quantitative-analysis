"""Run notebook step 8b on the saved walk-forward OOF."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".superpowers/scratch/mpl"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks/nba/minutes/src"))

import joblib
import numpy as np
import pandas as pd

nb = json.loads((ROOT / "notebooks/nba/minutes/02_minutes_xgboost.ipynb").read_text())
src = "".join(nb["cells"][23]["source"])
payload = joblib.load(ROOT / ".superpowers/scratch/cqr_wf.joblib")
wf_results = [
    {"train_mask": payload["train_masks"][fold_id]} for fold_id in range(1, 6)
]
namespace = {
    "wf": {"oof": payload["oof"]},
    "wf_results": wf_results,
    "ppm_df": payload["ppm"],
    "np": np,
    "pd": pd,
    "display": lambda obj: print(obj.to_string() if hasattr(obj, "to_string") else obj),
}
exec(compile(src, "cell_8b", "exec"), namespace)
print("REPORT_DONE")
