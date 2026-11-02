"""Listed position as a fixed one-hot.

The five Basketball-Reference positions are a nominal code.
An integer code would force every split to respect
PG < SG < SF < PF < C. Missing and unrecognized values
stay missing on every column so a tree can send them
down its missing branch. The listing is a player
attribute, so the current row is used as-is.
"""

from __future__ import annotations

import pandas as pd

POSITION_LEVELS = ("PG", "SG", "SF", "PF", "C")


def add_position_encoding(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "pos" in result.columns:
        pos = result["pos"].astype("string").str.strip().str.upper()
    else:
        pos = pd.Series(pd.NA, index=result.index, dtype="string")
    known = pos.isin(list(POSITION_LEVELS))
    for level in POSITION_LEVELS:
        result[f"pos_{level.lower()}"] = (
            pos.eq(level).astype("float64").where(known)
        )
    return result
