"""Run the five-variant joint coupling bake-off on preholdout folds.

Never loads 2025-26. Coupling estimates use earlier-fold data only.
MAE is recorded nowhere and cannot affect promotion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/alexgonzalez/Documents/nba_quant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xgboost_models.joint_calibration import (
    JOINT_CALIBRATION_ARTIFACT,
    JOINT_VARIANTS,
)
from src.models.xgboost_models.joint_variant_eval import (
    EVAL_SEED,
    N_DRAWS,
    OOF_PANEL_ARTIFACT,
    PROMOTION_CANDIDATE,
    VARIANT_METRICS_ARTIFACT,
    decide_promotion,
    load_preholdout_appearances,
    pool_metrics,
    prepare_oof_panel,
    score_variant_folds,
    write_promoted_artifact,
)
from src.models.xgboost_models.minutes import MINUTES_DISTRIBUTION_ARTIFACT

MINUTES_MEAN = (
    ROOT / "artifacts" / "models" / "minutes" / "xgboost_minutes.joblib"
)
POINTS_MEAN = (
    ROOT / "artifacts" / "models" / "points" / "xgboost_points.joblib"
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def main() -> None:
    appearances = load_preholdout_appearances(ROOT)
    print(
        f"Preholdout appearances: {len(appearances):,} "
        f"(2025-26 sealed, n_draws={N_DRAWS}, seed={EVAL_SEED})"
    )
    panel = prepare_oof_panel(
        appearances,
        minutes_mean_path=MINUTES_MEAN,
        minutes_dist_path=MINUTES_DISTRIBUTION_ARTIFACT,
        points_path=POINTS_MEAN,
        cache_path=OOF_PANEL_ARTIFACT,
        reuse_cache=True,
    )
    print(
        f"OOF panel rows: {len(panel):,}; "
        f"folds={sorted(panel['base_fold'].unique().tolist())}"
    )
    fold_metrics = score_variant_folds(
        panel,
        variants=JOINT_VARIANTS,
        n_draws=N_DRAWS,
        seed=EVAL_SEED,
        verbose=True,
    )
    pooled = {
        name: pool_metrics(rows)
        for name, rows in fold_metrics.items()
    }
    decision = decide_promotion(fold_metrics)
    payload = {
        "n_draws": N_DRAWS,
        "seed": EVAL_SEED,
        "candidate": PROMOTION_CANDIDATE,
        "selected": decision.selected,
        "passed": decision.passed,
        "reasons": decision.reasons,
        "pooled": pooled,
        "folds": fold_metrics,
    }
    VARIANT_METRICS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    VARIANT_METRICS_ARTIFACT.write_text(
        json.dumps(_jsonable(payload), indent=2)
    )
    print(json.dumps(_jsonable({
        "selected": decision.selected,
        "passed": decision.passed,
        "reasons": decision.reasons,
        "pooled": pooled,
    }), indent=2))
    print(f"Wrote {VARIANT_METRICS_ARTIFACT}")

    if decision.passed and decision.selected == PROMOTION_CANDIDATE:
        saved = write_promoted_artifact(
            panel,
            variant=PROMOTION_CANDIDATE,
            source_path=JOINT_CALIBRATION_ARTIFACT,
            dest_path=JOINT_CALIBRATION_ARTIFACT,
        )
        print(f"Promoted {PROMOTION_CANDIDATE}; saved {saved}")
    else:
        print("Retained current; joint_calibration.joblib not overwritten")


if __name__ == "__main__":
    main()
