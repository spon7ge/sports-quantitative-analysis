"""Audit NBA_US player-points quotes as a pricing/settlement pipeline check.

Does not use realized ROI to select a model or threshold.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/alexgonzalez/Documents/nba_quant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xgboost_models.snapshot_audit import audit_nba_us_snapshot

OUT_DIR = ROOT / "artifacts" / "pricing"
FILES = (
    ROOT / "data" / "odds" / "NBA_US_20260210_144744.csv",
    ROOT / "data" / "odds" / "NBA_US_20260325_131922.csv",
)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {}
    for path in FILES:
        print(f"auditing {path.name}", flush=True)
        table, reports = audit_nba_us_snapshot(path)
        stem = path.stem.lower()
        parquet_path = OUT_DIR / f"{stem}_snapshot_audit.parquet"
        if table.empty:
            table.to_parquet(parquet_path, index=False)
        else:
            table.to_parquet(parquet_path, index=False)
        payload[reports["as_of"]] = _clean(reports)
        print(json.dumps(_clean(reports), indent=2)[:2000], flush=True)
        print(f"wrote {parquet_path} n={len(table)}", flush=True)

    out_json = OUT_DIR / "nba_us_snapshot_audit.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
