"""Keep play-by-play features on games that match the box and predate the slate.

Substitution noise is already marked on the possession. A game whose shots,
free throws, turnovers, assists, or rebounds do not add up to the official
player totals is dropped, and the mismatch is returned as a flag. Heaves are
left out of the possession count, so the box may show a few more field-goal
attempts than the table.
"""

from __future__ import annotations

import pandas as pd
from game_events.links import EventLinks

# A game rarely has more than a couple of end-of-clock heaves, and those are
# not shot-mix attempts. A larger gap is a missed or double-counted trip.
HEAVE_ALLOWANCE = 2

_SHOT_STATS = ("fga", "fgm", "fta", "ftm", "tov")
_CREDIT_STATS = ("ast", "oreb", "dreb", "stl", "blk")


def rows_before(table: pd.DataFrame, slate) -> pd.DataFrame:
    """Rows whose ``game_date`` is strictly before ``slate``."""

    if not isinstance(table, pd.DataFrame):
        raise ValueError("play-by-play rows must be a DataFrame")
    if "game_date" not in table.columns:
        raise ValueError("play-by-play rows need game_date to enforce the slate date")

    day = pd.Timestamp(slate).normalize()
    dates = pd.to_datetime(table["game_date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("a play-by-play row has no game date")
    return table.loc[dates < day].reset_index(drop=True)


def possessions_for_slate(
    table: pd.DataFrame, slate, logs: pd.DataFrame
) -> pd.DataFrame:
    """Possessions before ``slate`` that match the official box."""

    dated = _with_dates(table, logs)
    day = pd.Timestamp(slate).normalize()
    early = dated.loc[dated["game_date"] < day].reset_index(drop=True)
    kept, _flags = reconcile_possessions(early, logs)
    return kept


def reconcile_possessions(
    table: pd.DataFrame, logs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop games whose shot, free-throw, or turnover totals miss the box.

    The flag frame has one row per failed stat, or one ``missing_box`` row
    when the game is not in the logs. Field-goal attempts may exceed the
    table by at most ``HEAVE_ALLOWANCE``.
    """

    if not isinstance(table, pd.DataFrame):
        raise ValueError("possessions must be a DataFrame")
    if table.empty:
        return table.copy(), _flags()

    totals = _possession_totals(table)
    official = _official_totals(logs, _SHOT_STATS)
    return _split(table, totals, official, _SHOT_STATS, _shot_mismatch)


def links_for_slate(links: EventLinks, slate, logs: pd.DataFrame) -> EventLinks:
    """Event links before ``slate`` whose credit totals match the box."""

    if not isinstance(links, EventLinks):
        raise ValueError("links must be EventLinks")

    early = _links_before(links, slate, logs)
    kept, _flags = reconcile_links(early, logs)
    return kept


def reconcile_links(
    links: EventLinks, logs: pd.DataFrame
) -> tuple[EventLinks, pd.DataFrame]:
    """Drop games whose assists, rebounds, steals, or blocks miss the box.

    A missing assist tag or a rebound credited to the wrong player changes
    the count, so that game is left out of the credit rates.
    """

    if not isinstance(links, EventLinks):
        raise ValueError("links must be EventLinks")

    totals = _link_totals(links)
    if totals.empty:
        return links, _flags()

    official = _official_totals(logs, _CREDIT_STATS)
    kept_ids, flags = _decide(totals, official, _CREDIT_STATS, _exact_mismatch)
    return _filter_links(links, kept_ids), flags


def _with_dates(table: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("possessions must be a DataFrame")
    if "game_id" not in table.columns:
        raise ValueError("possession rows need game_id")

    frame = table.copy()
    frame["game_id"] = frame["game_id"].map(_id)
    if "game_date" not in frame.columns:
        frame["game_date"] = frame["game_id"].map(_game_dates(logs))
    else:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    if frame["game_date"].isna().any():
        raise ValueError("a play-by-play row has no game date")
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    return frame


def _links_before(links: EventLinks, slate, logs: pd.DataFrame) -> EventLinks:
    found = _link_ids(links)
    if not found:
        return links

    dates = _game_dates(logs)
    missing = [game_id for game_id in found if game_id not in dates.index]
    if missing:
        raise ValueError("a play-by-play row has no game date")

    day = pd.Timestamp(slate).normalize()
    early = {game_id for game_id in found if dates.at[game_id] < day}
    return _filter_links(links, early)


def _split(
    table, totals, official, stats, mismatch
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_ids, flags = _decide(totals, official, stats, mismatch)
    keys = table["game_id"].map(_id)
    kept = table.loc[keys.isin(kept_ids)].reset_index(drop=True)
    return kept, flags


def _decide(totals, official, stats, mismatch) -> tuple[set[str], pd.DataFrame]:
    flags = []
    kept = set()
    for game_id, pbp in totals.iterrows():
        if game_id not in official.index:
            flags.append(_flag(game_id, "", None, None, "missing_box"))
            continue
        box = official.loc[game_id]
        failed = False
        for stat in stats:
            if stat not in box.index:
                continue
            pbp_value = int(pbp[stat])
            box_value = int(round(float(box[stat])))
            if mismatch(stat, pbp_value, box_value):
                failed = True
                flags.append(_flag(game_id, stat, pbp_value, box_value, "mismatch"))
        if not failed:
            kept.add(str(game_id))
    return kept, _flags(flags)


def _shot_mismatch(stat: str, pbp_value: int, box_value: int) -> bool:
    if stat == "fga":
        gap = box_value - pbp_value
        return gap < 0 or gap > HEAVE_ALLOWANCE
    return pbp_value != box_value


def _exact_mismatch(_stat: str, pbp_value: int, box_value: int) -> bool:
    return pbp_value != box_value


def _possession_totals(table: pd.DataFrame) -> pd.DataFrame:
    if "game_id" not in table.columns:
        raise ValueError("possession rows need game_id")
    if "shot_player_ids" not in table.columns or "shot_made" not in table.columns:
        raise ValueError("possession rows need shot_player_ids and shot_made")

    rows = []
    work = table.copy()
    work["game_id"] = work["game_id"].map(_id)
    for game_id, part in work.groupby("game_id", sort=False):
        fga = 0
        fgm = 0
        for shots, made in zip(part["shot_player_ids"], part["shot_made"], strict=True):
            fga += len(list(shots))
            fgm += sum(1 for flag in made if flag)
        fta = 0
        ftm = 0
        if "ft_fta" in part.columns and "ft_ftm" in part.columns:
            for attempts, makes in zip(part["ft_fta"], part["ft_ftm"], strict=True):
                fta += sum(int(value) for value in attempts)
                ftm += sum(int(value) for value in makes)
        if "tov" in part.columns:
            tov = int(pd.to_numeric(part["tov"], errors="coerce").fillna(0).sum())
        else:
            tov = 0
        rows.append(
            {
                "game_id": game_id,
                "fga": fga,
                "fgm": fgm,
                "fta": fta,
                "ftm": ftm,
                "tov": tov,
            }
        )
    return pd.DataFrame(rows).set_index("game_id")


def _link_totals(links: EventLinks) -> pd.DataFrame:
    ast = _sized(
        links.shots,
        lambda frame: frame["assisted"].astype(bool) & frame["passer_id"].notna()
        if {"assisted", "passer_id"} <= set(frame.columns)
        else pd.Series(False, index=frame.index),
    )
    oreb = _sized(
        links.rebounds,
        lambda frame: frame["outcome"].eq("oreb")
        if "outcome" in frame.columns
        else pd.Series(False, index=frame.index),
    )
    dreb = _sized(
        links.rebounds,
        lambda frame: frame["outcome"].eq("dreb")
        if "outcome" in frame.columns
        else pd.Series(False, index=frame.index),
    )
    stl = _sized(
        links.turnovers,
        lambda frame: frame["stealer_id"].notna()
        if "stealer_id" in frame.columns
        else pd.Series(False, index=frame.index),
    )
    blk = _sized(
        links.shots,
        lambda frame: frame["blocked"].astype(bool)
        if "blocked" in frame.columns
        else pd.Series(False, index=frame.index),
    )
    games = set().union(*[set(series.index) for series in (ast, oreb, dreb, stl, blk)])
    if not games:
        return pd.DataFrame(columns=_CREDIT_STATS)
    rows = []
    for game_id in games:
        rows.append(
            {
                "game_id": game_id,
                "ast": int(ast.get(game_id, 0)),
                "oreb": int(oreb.get(game_id, 0)),
                "dreb": int(dreb.get(game_id, 0)),
                "stl": int(stl.get(game_id, 0)),
                "blk": int(blk.get(game_id, 0)),
            }
        )
    return pd.DataFrame(rows).set_index("game_id")


def _sized(frame: pd.DataFrame, predicate) -> pd.Series:
    if frame is None or frame.empty or "game_id" not in frame.columns:
        return pd.Series(dtype="int64")
    work = frame.copy()
    work["game_id"] = work["game_id"].map(_id)
    chosen = work.loc[predicate(work)]
    if chosen.empty:
        return pd.Series(dtype="int64")
    return chosen.groupby("game_id", sort=False).size()


def _link_ids(links: EventLinks) -> list[str]:
    found = []
    for frame in (links.shots, links.rebounds, links.turnovers):
        if frame is None or frame.empty or "game_id" not in frame.columns:
            continue
        for game_id in frame["game_id"].map(_id):
            if game_id not in found:
                found.append(game_id)
    return found


def _filter_links(links: EventLinks, keep: set[str]) -> EventLinks:
    return EventLinks(
        _filter_frame(links.shots, keep),
        _filter_frame(links.rebounds, keep),
        _filter_frame(links.turnovers, keep),
    )


def _filter_frame(frame: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    if frame is None or frame.empty or "game_id" not in frame.columns:
        return frame.iloc[0:0].copy() if isinstance(frame, pd.DataFrame) else frame
    keys = frame["game_id"].map(_id)
    return frame.loc[keys.isin(keep)].reset_index(drop=True)


def _official_totals(logs: pd.DataFrame, stats: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(logs, pd.DataFrame) or "game_id" not in logs.columns:
        raise ValueError("logs need game_id")

    present = [stat for stat in stats if stat in logs.columns]
    missing = [stat for stat in stats if stat not in logs.columns]
    if missing:
        raise ValueError(f"logs need {missing} to reconcile play-by-play")

    frame = logs.copy()
    frame["game_id"] = frame["game_id"].map(_id)
    for stat in present:
        frame[stat] = pd.to_numeric(frame[stat], errors="coerce").fillna(0.0)
    return frame.groupby("game_id", sort=False)[present].sum()


def _game_dates(logs: pd.DataFrame) -> pd.Series:
    if (
        not isinstance(logs, pd.DataFrame)
        or "game_id" not in logs.columns
        or "game_date" not in logs.columns
    ):
        raise ValueError("logs need game_id and game_date")

    dates = pd.to_datetime(logs["game_date"], errors="coerce").dt.normalize()
    frame = pd.DataFrame(
        {
            "game_id": logs["game_id"].map(_id),
            "game_date": dates,
        }
    )
    if frame["game_date"].isna().any():
        raise ValueError("logs have a missing game date")
    if frame.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("a game ID has conflicting dates")
    return frame.drop_duplicates("game_id").set_index("game_id")["game_date"]


def _flags(rows: list[dict] | None = None) -> pd.DataFrame:
    columns = ["game_id", "stat", "pbp", "official", "reason"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _flag(game_id, stat, pbp, official, reason: str) -> dict:
    return {
        "game_id": game_id,
        "stat": stat,
        "pbp": pbp,
        "official": official,
        "reason": reason,
    }


def _id(value) -> str:
    if pd.isna(value):
        raise ValueError("game_id is missing")
    if isinstance(value, bool):
        raise ValueError("game_id is missing")
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text
