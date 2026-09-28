from __future__ import annotations

import pandas as pd


def assign_walk_forward_blocks(
    starts: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    out = starts.copy()
    cal = calendar.copy()
    cal["season"] = cal["season"].astype(int)
    out = out.merge(cal, on="season", how="left")
    game_date = pd.to_datetime(out["game_date"])
    opened = pd.to_datetime(out["regular_season_open_date"])
    closed = pd.to_datetime(out["regular_season_close_date"])
    out["model_eligible"] = (
        opened.notna() & closed.notna() & game_date.ge(opened) & game_date.le(closed)
    )
    day_offset = (game_date - opened).dt.days
    block_index = (day_offset // 28).astype("Int64")
    label = out["season"].astype("Int64").astype(str) + "-" + block_index.astype(str)
    out["walk_forward_block"] = label.where(out["model_eligible"], pd.NA)
    return out
