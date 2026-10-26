"""Points-per-minute target, leak check, and the train-pool Tukey fence."""

from __future__ import annotations

import pandas as pd

from points_xgb.config import EXCLUDED_FEATURES, PTS_FEATURES, RATE_CAP, TARGET_COL
from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.oos import capped_rate  # noqa: E402
from models.shared.splits import tukey_fences  # noqa: E402
from src.features.minutes.rolling import prior_expanding, prior_shift  # noqa: E402


def assert_no_leak(features: list[str] | tuple[str, ...] = PTS_FEATURES) -> None:
    """Refuse game-N outcomes in the model feature list."""
    leaked = EXCLUDED_FEATURES.intersection(features)
    if leaked:
        raise AssertionError(f"Game-N outcomes leaked into features: {sorted(leaked)}")


def build_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the shared points features."""
    from src.features.points import add_points_features

    return add_points_features(frame)


def positive_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a positive parsed minute total."""
    return frame.loc[frame["minutes"] > 0].copy()


def assign_rate_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Cap points per minute and add the pregame naive columns.

    Rows are ordered by game date so the lag is the previous appearance.
    ``pts_per_min_ewm_hl_10`` is already built by the feature pipeline.
    """
    out = frame.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    out[TARGET_COL] = capped_rate(out["pts"], out["minutes"], cap=RATE_CAP)
    out["pts_per_min_lag_1"] = prior_shift(out, TARGET_COL, ["player_id"])
    out["season_pts_per_min_mean"] = prior_expanding(
        out, TARGET_COL, ["player_id", "season_year"], "mean"
    )
    return out


def apply_train_tukey(
    ppm_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, float, float, int]:
    """Drop train-pool rates outside the fence fit on that pool alone.

    Rates on the fence stay. The caller must not pass holdout rows.
    """
    low, high = tukey_fences(ppm_df[target_col])
    keep = ppm_df[target_col].between(low, high)
    n_dropped = int((~keep).sum())
    return ppm_df.loc[keep].reset_index(drop=True), low, high, n_dropped
