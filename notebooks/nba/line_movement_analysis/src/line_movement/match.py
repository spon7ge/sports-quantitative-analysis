from __future__ import annotations

import pandas as pd

MARKET_KEY_COLS = (
    "player_norm",
    "event_key",
    "prop",
    "period",
    "side",
    "line",
)
_CONTINUITY_KEY_COLS = MARKET_KEY_COLS[:-1]
_GROUP_COLS = [*MARKET_KEY_COLS, "market_key"]


def _key_or_none(row: pd.Series, columns: tuple[str, ...]) -> tuple | None:
    values = tuple(row[column] for column in columns)
    return None if any(pd.isna(value) for value in values) else values


def attach_market_keys(quotes: pd.DataFrame) -> pd.DataFrame:
    """Attach exact-market and across-line continuity identifiers."""
    missing = sorted(set(MARKET_KEY_COLS).difference(quotes.columns))
    if missing:
        raise ValueError(f"quotes missing market key columns: {', '.join(missing)}")

    out = quotes.copy()
    out["market_key"] = out.apply(
        lambda row: _key_or_none(row, MARKET_KEY_COLS), axis=1
    )
    out["continuity_key"] = out.apply(
        lambda row: _key_or_none(row, _CONTINUITY_KEY_COLS), axis=1
    )
    return out


def _sorted_books(values: pd.Series) -> str:
    books = {str(value) for value in values if pd.notna(value)}
    return ",".join(sorted(books))


def build_market_groups(quotes: pd.DataFrame) -> pd.DataFrame:
    """Summarize quote coverage for each distinct market."""
    keyed = attach_market_keys(quotes)
    if "book" not in keyed.columns:
        raise ValueError("quotes missing required column: book")

    groups = (
        keyed.groupby(_GROUP_COLS, dropna=False, sort=False)
        .agg(
            n_books=("book", "nunique"),
            books=("book", _sorted_books),
            n_rows=("book", "size"),
        )
        .reset_index()
    )
    groups["matchable"] = groups["market_key"].notna() & groups["n_books"].ge(1)
    return groups[[*MARKET_KEY_COLS, "n_books", "books", "n_rows", "matchable"]]


def cross_book_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Return complete market groups quoted by at least two books."""
    required = {"n_books", "matchable"}
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"groups missing required columns: {', '.join(missing)}")
    return groups.loc[groups["matchable"] & groups["n_books"].ge(2)].reset_index(
        drop=True
    )
