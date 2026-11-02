"""Points-per-minute target, leak check, and the train-pool Tukey fence."""

from __future__ import annotations

import pandas as pd

from points_xgb.config import EXCLUDED_FEATURES, RATE_CAP, TARGET_COL
from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.oos import capped_rate  # noqa: E402
from models.shared.splits import tukey_fences  # noqa: E402
from src.features.minutes.rolling import prior_expanding, prior_shift  # noqa: E402


def assert_no_leak(features: list[str] | tuple[str, ...]) -> None:
    """Refuse game-N outcomes in the model feature list."""
    leaked = EXCLUDED_FEATURES.intersection(features)
    if leaked:
        raise AssertionError(f"Game-N outcomes leaked into features: {sorted(leaked)}")


def build_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add starter rank, then the shared points features.

    ``starting`` is the recorded G/F/C flag from ``start_position`` when
    the loader has not already set it. Rank uses prior starts and prior
    minutes only, the same definition as the minutes model.
    ``starter_min_missing_teammates`` and ``total_stars_active`` use who
    played in this game, so they are known once inactives are confirmed.
    """
    from src.features.minutes import add_starter_features, add_total_stars_active
    from src.features.points import add_points_features

    prepared = _with_starting_flag(frame)
    featured = add_points_features(add_starter_features(prepared))
    return add_total_stars_active(featured)


def _with_starting_flag(frame: pd.DataFrame) -> pd.DataFrame:
    if "starting" in frame.columns:
        return frame
    out = frame.copy()
    start = (
        out["start_position"]
        if "start_position" in out.columns
        else pd.Series("", index=out.index)
    )
    out["starting"] = (
        start.astype("string").str.strip().isin(["G", "F", "C"]).astype(int)
    )
    return out


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
