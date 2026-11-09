"""Grade simulator predictions against historical closing player props.

Predictions enter as simulation samples. Closing prices enter as odds.
Final stat lines enter only when a wager is settled. A book's missing
line stays missing. Moneylines, spreads, and totals are not bets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.models.odds import (
    american_to_decimal,
    american_to_implied_probability,
    expected_value,
)
from src.scrapers.arbitrage import canonical_team, player_key

PACIFIC = ZoneInfo("America/Los_Angeles")
STRATEGY_ORDER = ("top_5", "top_10", "top_20", "all")
_CORE_TOP_N = (5, 10, 20)
_GAME_MARKETS = frozenset({"moneyline", "spread", "total"})
_PRIZEPICKS_AMERICAN = -119
_PROP_MARKETS = (
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "threes",
    "pra",
    "pr",
    "pa",
    "ra",
    "stocks",
)
_BOOK_FOLDERS = {
    "draftkings": "draftkings",
    "fanduel": "fanduel",
    "novig": "novig",
    "pinnacle": "pinnacle",
    "prizepicks": "prizepicks",
    "prophetx": "prophetx",
    "underdog": "underdogs",
}
_STAT_ALIASES = {
    "points": "points",
    "pts": "points",
    "rebounds": "rebounds",
    "reb": "rebounds",
    "assists": "assists",
    "ast": "assists",
    "steals": "steals",
    "stl": "steals",
    "blocks": "blocks",
    "blk": "blocks",
    "threes": "threes",
    "3pm": "threes",
    "3-pointers": "threes",
    "pra": "pra",
    "p+r+a": "pra",
    "points+rebounds+assists": "pra",
    "points_rebounds_assists": "pra",
    "pts_rebs_asts": "pra",
    "pr": "pr",
    "points+rebounds": "pr",
    "points_rebounds": "pr",
    "pts_rebs": "pr",
    "pa": "pa",
    "points+assists": "pa",
    "points_assists": "pa",
    "pts_asts": "pa",
    "ra": "ra",
    "rebounds+assists": "ra",
    "rebounds_assists": "ra",
    "rebs_asts": "ra",
    "three_pointers_made": "threes",
    "three_points_made": "threes",
    "3-pt made": "threes",
    "pts+rebs": "pr",
    "pts+rebs+asts": "pra",
    "pts+asts": "pa",
    "rebs+asts": "ra",
    "blocked shots": "blocks",
    "stocks": "stocks",
    "blks_stls": "stocks",
}
_STAT_PARTS = {
    "points": ("PTS",),
    "rebounds": ("REB",),
    "assists": ("AST",),
    "steals": ("STL",),
    "blocks": ("BLK",),
    "threes": ("3PM",),
    "pra": ("PTS", "REB", "AST"),
    "pr": ("PTS", "REB"),
    "pa": ("PTS", "AST"),
    "ra": ("REB", "AST"),
    "stocks": ("STL", "BLK"),
}
_GAME_LEAKS = frozenset(
    {"home_pts", "away_pts", "actual_home_pts", "actual_away_pts", "result"}
)
_PLAYER_LEAKS = frozenset({"actual_pts", "actual_reb", "actual_ast", "result"})
_OFFER_COLUMNS = (
    "slate_date",
    "book",
    "game_key",
    "market",
    "selection",
    "side",
    "line",
    "american",
    "closing_fetched_at",
    "price_source",
    "player_name",
)
_LEDGER_COLUMNS = (
    "slate_date",
    "book",
    "game_id",
    "game",
    "market",
    "selection",
    "line",
    "model_probability",
    "push_probability",
    "implied_probability",
    "american",
    "decimal_odds",
    "ev",
    "rank",
    "stake",
    "actual_outcome",
    "result",
    "profit",
    "bankroll",
)
_COMPARISON_COLUMNS = (
    "strategy",
    "book",
    "total_bets",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "amount_wagered",
    "profit",
    "roi",
    "avg_ev",
    "avg_decimal_odds",
    "ending_bankroll",
    "skipped_unfunded",
)


@dataclass(frozen=True)
class BacktestConfig:
    """Controls that change the bet list without refitting the simulator."""

    start: str
    end: str
    bankroll: float
    stake: float
    stake_mode: str = "flat"
    top_n: int | None = 10
    books: tuple[str, ...] = ("pinnacle",)
    markets: tuple[str, ...] = ("points", "rebounds", "assists")
    n_sims: int = 500

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start).normalize()
        end = pd.Timestamp(self.end).normalize()
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError("backtest end must be on or after the start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if (
            isinstance(self.bankroll, bool)
            or not isinstance(self.bankroll, (int, float))
            or not math.isfinite(self.bankroll)
            or self.bankroll <= 0
        ):
            raise ValueError("bankroll must be a positive finite number")
        if self.stake_mode not in {"flat", "percent"}:
            raise ValueError("stake_mode must be flat or percent")
        if (
            isinstance(self.stake, bool)
            or not isinstance(self.stake, (int, float))
            or not math.isfinite(self.stake)
            or self.stake <= 0
        ):
            raise ValueError("stake must be a positive finite number")
        if self.stake_mode == "percent" and self.stake > 1:
            raise ValueError("percent stake must be a fraction of bankroll, at most 1")
        if self.top_n is not None and (
            isinstance(self.top_n, bool)
            or not isinstance(self.top_n, (int, np.integer))
            or self.top_n <= 0
        ):
            raise ValueError("top_n must be a positive integer or all picks")
        books = tuple(str(book) for book in self.books)
        if not books:
            raise ValueError("select at least one book")
        if "all" in books and books != ("all",):
            raise ValueError("all books cannot be combined with a named book")
        markets = tuple(str(market) for market in self.markets)
        if not markets:
            raise ValueError("select at least one market")
        game_market = next(
            (market for market in markets if market in _GAME_MARKETS), None
        )
        if game_market is not None:
            raise ValueError(f"{game_market} is not a player prop")
        unknown = next(
            (market for market in markets if market not in _PROP_MARKETS), None
        )
        if unknown is not None:
            raise ValueError(f"{unknown} is not a player prop")
        if (
            isinstance(self.n_sims, bool)
            or not isinstance(self.n_sims, (int, np.integer))
            or self.n_sims <= 0
        ):
            raise ValueError("n_sims must be a positive integer")
        object.__setattr__(self, "books", books)
        object.__setattr__(self, "markets", markets)


def unit_ev(
    win_probability: float,
    loss_probability: float,
    push_probability: float,
    american: float,
) -> float:
    """Expected profit per unit stake.

    A push returns the stake, so it adds nothing. With no push this is
    ``win_probability * decimal_odds - 1``.
    """
    _require_probability(win_probability, loss_probability, push_probability)
    return float(expected_value(win_probability, loss_probability, american))


def assert_history_before_slate(predictions: pd.DataFrame) -> None:
    """Every prediction cutoff is a calendar day before the game."""
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a DataFrame")
    missing = {"slate_date", "history_through"} - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions need {sorted(missing)}")
    slate = pd.to_datetime(predictions["slate_date"], errors="coerce").dt.normalize()
    history = pd.to_datetime(
        predictions["history_through"], errors="coerce"
    ).dt.normalize()
    if slate.isna().any() or history.isna().any():
        raise ValueError("prediction dates must be readable")
    leaked = history.ge(slate)
    if leaked.any():
        day = slate.loc[leaked].iloc[0].date()
        raise ValueError(f"history includes the slate date {day}")


def available_books(props_root: Path) -> tuple[str, ...]:
    """Books that have a player-prop closing directory."""
    root = Path(props_root)
    found = []
    for book in _BOOK_FOLDERS:
        base = _book_dir(root, book)
        if (base / "closing").is_dir():
            found.append(book)
    return tuple(found)


def load_closing_offers(
    props_root: Path,
    books: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Closing quotes only. An unpriced or wrong-day line is counted and dropped.

    ``opening_lines`` are never read. Team closing files are not read.
    A selected book with no prop file does not inherit another book's price.
    A PrizePicks standard line has no posted side or price, so both sides
    are filled at -119. Demon and goblin lines stay unpriced.
    """
    root = Path(props_root)
    selected = _selected_books(root, books)
    audit = {
        "files": 0,
        "unpriced": 0,
        "unsupported_market": 0,
        "unmatched_name": 0,
        "timestamp_mismatch": 0,
        "duplicate_offers_dropped": 0,
        "alternate_lines_dropped": 0,
    }
    rows: list[dict[str, object]] = []
    for book in selected:
        base = _book_dir(root, book)
        rows.extend(_read_player_files(base / "closing", book, audit))
    offers = pd.DataFrame(rows, columns=list(_OFFER_COLUMNS))
    if offers.empty:
        return offers, audit
    offers["slate_date"] = pd.to_datetime(offers["slate_date"]).dt.normalize()
    offers, dropped = _drop_duplicate_offers(offers)
    audit["duplicate_offers_dropped"] += dropped
    offers, alternates = _keep_main_lines(offers)
    audit["alternate_lines_dropped"] = alternates
    return offers.reset_index(drop=True), audit


def build_wagers(
    offers: pd.DataFrame,
    game_sims: pd.DataFrame,
    player_sims: pd.DataFrame,
    game_actuals: pd.DataFrame,
    player_actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach sample probabilities and settle them with final scores.

    Actual scores are not accepted on the simulation frames.
    """
    _reject_leaks(game_sims, _GAME_LEAKS, "game simulations")
    _reject_leaks(player_sims, _PLAYER_LEAKS, "player simulations")
    assert_history_before_slate(game_sims)
    games = _index_games(game_sims)
    players = _index_players(player_sims)
    _index_actual_games(game_actuals)
    actual_players = _index_actual_players(player_actuals)
    audit = {
        "graded": 0,
        "offers_missing_prediction": 0,
        "offers_missing_result": 0,
        "ambiguous": 0,
        "invalid_probability": 0,
        "players_missing_prop_odds": 0,
    }
    graded_rows: list[dict[str, object]] = []
    offered_players = set()
    if not offers.empty and "player_name" in offers.columns:
        named = offers.loc[offers["player_name"].notna(), "player_name"]
        offered_players = {
            (pd.Timestamp(day).normalize(), player_key(str(name)))
            for day, name in zip(
                offers.loc[named.index, "slate_date"], named, strict=False
            )
        }
    for record in players.itertuples(index=False):
        if (record.slate_date, record.player_key) not in offered_players:
            audit["players_missing_prop_odds"] += 1

    for offer in offers.itertuples(index=False):
        row = _grade_offer(offer, games, players, actual_players, audit)
        if row is not None:
            graded_rows.append(row)
    graded = pd.DataFrame(graded_rows)
    audit["graded"] = int(len(graded))
    if graded.empty:
        return graded, audit
    _assert_unique_bets(graded)
    return graded.reset_index(drop=True), audit


def run_strategies(
    graded: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Stake each book on its own. Top 5, 10, 20, and all start from one bankroll."""
    if not isinstance(config, BacktestConfig):
        raise ValueError("config must be a BacktestConfig")
    pool = _filter_graded(graded, config)
    books = _strategy_books(pool, config)
    names = _strategy_names(config)
    ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []
    for strategy in names:
        limit = None if strategy == "all" else int(strategy.split("_", 1)[1])
        for book in books:
            book_rows = pool.loc[pool["book"].eq(book)] if not pool.empty else pool
            ledger, skipped = _stake_book(book_rows, book, limit, config)
            ledgers[(strategy, book)] = ledger
            summaries.append(_summarize(strategy, book, ledger, skipped, config))
    comparison = pd.DataFrame(summaries, columns=list(_COMPARISON_COLUMNS))
    selected = "all" if config.top_n is None else f"top_{int(config.top_n)}"
    chosen = [
        ledgers[(selected, book)]
        for book in books
        if (selected, book) in ledgers and not ledgers[(selected, book)].empty
    ]
    recommendations = (
        pd.concat(chosen, ignore_index=True)
        if chosen
        else pd.DataFrame(columns=list(_LEDGER_COLUMNS))
    )
    return recommendations, comparison, ledgers


def _strategy_names(config: BacktestConfig) -> tuple[str, ...]:
    limits = list(_CORE_TOP_N)
    if config.top_n is not None and int(config.top_n) not in limits:
        limits.append(int(config.top_n))
    return tuple(f"top_{limit}" for limit in limits) + ("all",)


def _selected_books(root: Path, books: Sequence[str]) -> tuple[str, ...]:
    if isinstance(books, str) or not books:
        raise ValueError("books must be a non-empty list")
    chosen = tuple(str(book) for book in books)
    if chosen == ("all",):
        found = available_books(root)
        if not found:
            raise FileNotFoundError(f"no closing lines under {root}")
        return found
    unknown = [book for book in chosen if book not in _BOOK_FOLDERS]
    if unknown:
        raise ValueError(f"unknown book {unknown[0]}")
    for book in chosen:
        base = _book_dir(root, book)
        if not (base / "closing").is_dir():
            raise FileNotFoundError(f"no closing lines for {book} under {base}")
    return chosen


def _book_dir(root: Path, book: str) -> Path:
    return root / _BOOK_FOLDERS[book] / "nba"


def _read_player_files(folder: Path, book: str, audit: dict[str, int]) -> list[dict]:
    if not folder.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(folder.glob("*_nba_closing_*.json")):
        payload = _payload(path)
        slate = _slate_date(payload, path)
        audit["files"] += 1
        for player in _list(payload, "players", path):
            rows.extend(_player_quotes(player, book, slate, path, audit))
    return rows


def _player_quotes(
    player: object, book: str, slate, path: Path, audit: dict[str, int]
) -> list[dict]:
    if not isinstance(player, dict):
        raise ValueError(f"{path.name} player must be an object")
    name = str(player.get("player") or "").strip()
    if not name or player_key(name) == "":
        audit["unmatched_name"] += 1
        return []
    fetched = _close_stamp(player, slate, path, audit)
    if fetched is None:
        return []
    quotes = []
    for line in _list(player, "lines", path):
        quotes.extend(_player_quote(line, book, slate, name, fetched, path, audit))
    return quotes


def _player_quote(line, book, slate, name, fetched, path: Path, audit) -> list[dict]:
    if not isinstance(line, dict):
        raise ValueError(f"{path.name} line must be an object")
    market = _STAT_ALIASES.get(str(line.get("stat_type") or "").strip().lower())
    if market is None:
        audit["unsupported_market"] += 1
        return []
    number = _line(line.get("line_score"), path, required=True)
    if _prizepicks_standard_without_price(book, line):
        return [
            _quote(
                slate,
                book,
                "",
                market,
                f"{name} {side}",
                side,
                number,
                _PRIZEPICKS_AMERICAN,
                fetched,
                player_name=name,
            )
            for side in ("over", "under")
        ]
    american = _american(line.get("american_price"), path)
    if american is None:
        audit["unpriced"] += 1
        return []
    side = str(line.get("side") or "").strip().lower()
    if side not in {"over", "under"}:
        audit["unmatched_name"] += 1
        return []
    return [
        _quote(
            slate,
            book,
            "",
            market,
            f"{name} {side}",
            side,
            number,
            american,
            fetched,
            player_name=name,
        )
    ]


def _prizepicks_standard_without_price(book: str, line: dict) -> bool:
    """PrizePicks standard pick'em lines are posted at -119 on both sides."""
    if book != "prizepicks":
        return False
    odds_type = str(line.get("odds_type") or "").strip().lower()
    return odds_type == "standard" and line.get("american_price") is None


def _quote(
    slate,
    book,
    game_key,
    market,
    selection,
    side,
    line,
    american,
    fetched,
    player_name=None,
) -> dict:
    return {
        "slate_date": slate,
        "book": book,
        "game_key": game_key,
        "market": market,
        "selection": selection,
        "side": side,
        "line": np.nan if line is None else float(line),
        "american": float(american),
        "closing_fetched_at": fetched,
        "price_source": "closing",
        "player_name": player_name,
    }


def _payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _slate_date(payload: dict, path: Path) -> pd.Timestamp:
    if payload.get("timezone") != "America/Los_Angeles":
        raise ValueError(f"{path.name} closing date must be Pacific")
    slate = pd.Timestamp(payload.get("date")).normalize()
    if pd.isna(slate):
        raise ValueError(f"{path.name} has no slate date")
    return slate


def _list(payload: dict, key: str, path: Path) -> list:
    raw = payload.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path.name} {key} must be a list")
    return raw


def _close_stamp(
    record: dict, slate: pd.Timestamp, path: Path, audit: dict[str, int]
) -> str | None:
    raw = record.get("closing_fetched_at")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{path.name} is missing closing_fetched_at")
    fetched = _aware(raw, path)
    nxt = record.get("next_fetched_at")
    if nxt is not None:
        if not isinstance(nxt, str):
            raise ValueError(f"{path.name} next_fetched_at must be a timestamp")
        if _aware(nxt, path) <= fetched:
            audit["timestamp_mismatch"] += 1
            return None
    if fetched.astimezone(PACIFIC).date() != slate.date():
        audit["timestamp_mismatch"] += 1
        return None
    return fetched.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aware(value: str, path: Path) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{path.name} has an unreadable timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{path.name} timestamps must be timezone-aware")
    return parsed


def _line(value, path: Path, *, required: bool) -> float | None | bool:
    if value is None:
        if required:
            raise ValueError(f"{path.name} is missing a line")
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path.name} has a non-numeric line")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} has a non-numeric line") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path.name} has a non-numeric line")
    return number


def _american(value, path: Path) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path.name} has a non-numeric american price")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} has a non-numeric american price") from exc
    if not math.isfinite(number) or number == 0:
        raise ValueError(f"{path.name} has a non-numeric american price")
    return number


def _keep_main_lines(offers: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One two-way rung per market. The closest over/under pair is the close."""
    if offers.empty:
        return offers, 0
    kept = []
    dropped = 0
    keys = ["slate_date", "book", "game_key", "player_name", "market"]
    for _key, group in offers.groupby(keys, dropna=False, sort=False):
        chosen = _choose_main_rung(group)
        kept.append(chosen)
        dropped += int(len(group) - len(chosen))
    return pd.concat(kept, ignore_index=True), dropped


def _choose_main_rung(group: pd.DataFrame) -> pd.DataFrame:
    work = group.copy()
    work["_pair"] = [round(float(value), 4) for value in work["line"]]
    pairs = []
    for _pair, rows in work.groupby("_pair", sort=False):
        if rows["side"].nunique() >= 2:
            pairs.append(rows.drop_duplicates("side"))
    if pairs:
        best = min(pairs, key=_rung_gap)
        return best.drop(columns="_pair")
    work["_distance"] = [
        abs(american_to_implied_probability(float(price)) - 0.5)
        for price in work["american"]
    ]
    return (
        work.sort_values("_distance", kind="mergesort")
        .head(1)
        .drop(columns=["_pair", "_distance"])
    )


def _rung_gap(rows: pd.DataFrame) -> tuple[float, float]:
    prices = [float(price) for price in rows["american"]]
    implied = [american_to_implied_probability(price) for price in prices]
    gap = abs(implied[0] - implied[1]) if len(implied) >= 2 else 1.0
    return gap, abs(float(rows["line"].iloc[0]))


def _drop_duplicate_offers(offers: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = offers.copy()
    keys["_line"] = keys["line"].map(
        lambda value: None if pd.isna(value) else round(float(value), 4)
    )
    keys["_player"] = keys["player_name"].fillna("")
    duplicate = keys.duplicated(
        ["slate_date", "book", "game_key", "_player", "market", "selection", "_line"],
        keep="first",
    )
    return offers.loc[~duplicate].copy(), int(duplicate.sum())


def _reject_leaks(frame: pd.DataFrame, columns: frozenset[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame")
    found = columns.intersection(frame.columns)
    if found:
        raise ValueError(f"{label} include actual scores: {sorted(found)}")


def _index_games(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slate_date",
        "game_id",
        "home_abbr",
        "away_abbr",
        "history_through",
        "home_samples",
        "away_samples",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"game simulations need {sorted(missing)}")
    out = frame.copy()
    out["slate_date"] = pd.to_datetime(out["slate_date"]).dt.normalize()
    out["home_id"] = out["home_abbr"].map(
        lambda value: canonical_team(str(value), "nba")
    )
    out["away_id"] = out["away_abbr"].map(
        lambda value: canonical_team(str(value), "nba")
    )
    if out["home_id"].isna().any() or out["away_id"].isna().any():
        raise ValueError("game simulations have an unknown team")
    if (out["home_id"] == out["away_id"]).any():
        raise ValueError("game simulations repeat a team")
    out["game_key"] = [
        "|".join(sorted((home, away)))
        for home, away in zip(out["home_id"], out["away_id"], strict=True)
    ]
    if out.duplicated(["slate_date", "game_key"]).any():
        raise ValueError("two simulations share a slate and a matchup")
    return out.reset_index(drop=True)


def _index_players(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"slate_date", "game_id", "player_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"player simulations need {sorted(missing)}")
    out = frame.copy()
    out["slate_date"] = pd.to_datetime(out["slate_date"]).dt.normalize()
    out["player_key"] = out["player_name"].map(lambda value: player_key(str(value)))
    if out["player_key"].eq("").any():
        raise ValueError("player simulations have a blank name")
    if out.duplicated(["slate_date", "player_key"]).any():
        raise ValueError("two simulations share a player on one slate")
    return out.reset_index(drop=True)


def _index_actual_games(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slate_date",
        "game_id",
        "home_abbr",
        "away_abbr",
        "home_pts",
        "away_pts",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"game actuals need {sorted(missing)}")
    out = frame.copy()
    out["slate_date"] = pd.to_datetime(out["slate_date"]).dt.normalize()
    out["home_id"] = out["home_abbr"].map(
        lambda value: canonical_team(str(value), "nba")
    )
    out["away_id"] = out["away_abbr"].map(
        lambda value: canonical_team(str(value), "nba")
    )
    out["game_key"] = [
        "|".join(sorted((home, away)))
        for home, away in zip(out["home_id"], out["away_id"], strict=True)
    ]
    for column in ("home_pts", "away_pts"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["home_pts", "away_pts"]].isna().any().any():
        raise ValueError("game actuals must be numeric")
    return out.reset_index(drop=True)


def _index_actual_players(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"slate_date", "game_id", "player_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"player actuals need {sorted(missing)}")
    out = frame.copy()
    out["slate_date"] = pd.to_datetime(out["slate_date"]).dt.normalize()
    out["player_key"] = out["player_name"].map(lambda value: player_key(str(value)))
    return out.reset_index(drop=True)


def _grade_offer(offer, games, players, actual_players, audit) -> dict | None:
    slate = pd.Timestamp(offer.slate_date).normalize()
    if not pd.notna(offer.player_name):
        audit["unsupported_market"] += 1
        return None
    return _grade_player(offer, slate, players, actual_players, games, audit)


def _grade_player(offer, slate, players, actual_players, games, audit) -> dict | None:
    key = player_key(str(offer.player_name))
    matched = players.loc[
        players["slate_date"].eq(slate) & players["player_key"].eq(key)
    ]
    if len(matched) != 1:
        audit["offers_missing_prediction" if matched.empty else "ambiguous"] += 1
        return None
    player = matched.iloc[0]
    parts = _STAT_PARTS[offer.market]
    missing = [part for part in parts if part not in player.index]
    if missing:
        audit["offers_missing_prediction"] += 1
        return None
    samples = np.zeros(0, dtype=float)
    for part in parts:
        column = _samples(player[part], f"{offer.player_name} {part}")
        samples = column if samples.size == 0 else samples + column
    actual_row = actual_players.loc[
        actual_players["slate_date"].eq(slate) & actual_players["player_key"].eq(key)
    ]
    if len(actual_row) != 1:
        audit["offers_missing_result"] += 1
        return None
    actual = actual_row.iloc[0]
    if any(part not in actual.index for part in parts):
        audit["offers_missing_result"] += 1
        return None
    final = float(sum(float(actual[part]) for part in parts))
    game = games.loc[games["game_id"].astype(str).eq(str(player["game_id"]))]
    if len(game) != 1:
        audit["offers_missing_prediction"] += 1
        return None
    text = f"{offer.player_name} {offer.market} {_format_number(final)}"
    return _priced_row(
        offer,
        game.iloc[0],
        samples,
        float(offer.line),
        offer.side == "over",
        final,
        text,
    )


def _priced_row(offer, game, samples, line, higher_wins, final, outcome) -> dict:
    win, loss, push = _probabilities(samples, line, higher_wins)
    american = float(offer.american)
    result = _result(final, line, higher_wins)
    decimal = american_to_decimal(american)
    if not math.isclose(
        decimal, 1 / american_to_implied_probability(american), rel_tol=0, abs_tol=1e-9
    ):
        raise AssertionError("decimal odds do not match the implied probability")
    return {
        "slate_date": pd.Timestamp(offer.slate_date).normalize(),
        "book": offer.book,
        "game_id": str(game["game_id"]),
        "game": f"{game['away_abbr']} @ {game['home_abbr']}",
        "market": offer.market,
        "selection": offer.selection,
        "line": offer.line,
        "model_probability": win,
        "push_probability": push,
        "implied_probability": american_to_implied_probability(american),
        "american": american,
        "decimal_odds": decimal,
        "ev": unit_ev(win, loss, push, american),
        "result": result,
        "actual_outcome": outcome,
        "unit_profit": _unit_profit(result, american),
        "history_through": pd.Timestamp(game["history_through"]).normalize(),
    }


def _samples(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} samples must be a non-empty finite vector")
    return array


def _probabilities(
    samples: np.ndarray, line: float, higher_wins: bool
) -> tuple[float, float, float]:
    pushes = np.isclose(samples, line, rtol=0, atol=1e-6)
    wins = samples > line if higher_wins else samples < line
    wins = wins & ~pushes
    losses = ~wins & ~pushes
    win = float(wins.mean())
    loss = float(losses.mean())
    push = float(pushes.mean())
    _require_probability(win, loss, push)
    return win, loss, push


def _result(final: float, line: float, higher_wins: bool) -> str:
    if math.isclose(final, line, rel_tol=0, abs_tol=1e-6):
        return "push"
    won = final > line if higher_wins else final < line
    return "win" if won else "loss"


def _unit_profit(result: str, american: float) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -1.0
    if result == "win":
        return american_to_decimal(american) - 1
    raise ValueError(f"unknown bet result {result}")


def _require_probability(win: float, loss: float, push: float) -> None:
    parts = (win, loss, push)
    if any(not math.isfinite(part) or part < -1e-9 for part in parts):
        raise ValueError("probabilities must be finite and non-negative")
    if abs(sum(parts) - 1) > 1e-6:
        raise ValueError("win, loss, and push probabilities must sum to 1")


def _format_number(value: float) -> str:
    number = float(value)
    if math.isclose(number, round(number), abs_tol=1e-6):
        return str(int(round(number)))
    return f"{number:.1f}"


def _assert_unique_bets(graded: pd.DataFrame) -> None:
    keys = ["slate_date", "book", "game_id", "market", "selection", "line"]
    if graded.duplicated(keys).any():
        raise AssertionError("a settled bet was counted twice")
    assert_history_before_slate(graded)


def _filter_graded(graded: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if graded.empty:
        return graded
    required = {
        "slate_date",
        "book",
        "market",
        "selection",
        "ev",
        "result",
        "american",
        "decimal_odds",
        "unit_profit",
        "model_probability",
        "implied_probability",
        "actual_outcome",
        "game",
    }
    missing = required - set(graded.columns)
    if missing:
        raise ValueError(f"graded wagers need {sorted(missing)}")
    out = graded.copy()
    out["slate_date"] = pd.to_datetime(out["slate_date"]).dt.normalize()
    out = out.loc[
        out["slate_date"].ge(config.start)
        & out["slate_date"].le(config.end)
        & out["market"].isin(config.markets)
    ]
    if config.books != ("all",):
        out = out.loc[out["book"].isin(config.books)]
    if not out.empty and not np.isfinite(out["ev"].to_numpy(dtype=float)).all():
        raise ValueError("expected value must be finite")
    for row in out.itertuples(index=False):
        profit = _unit_profit(str(row.result), float(row.american))
        if not math.isclose(profit, float(row.unit_profit), rel_tol=0, abs_tol=1e-6):
            raise ValueError("unit profit does not match the settled result")
    return out.reset_index(drop=True)


def _strategy_books(pool: pd.DataFrame, config: BacktestConfig) -> tuple[str, ...]:
    if config.books != ("all",):
        return config.books
    if pool.empty:
        return ()
    return tuple(sorted(pool["book"].astype(str).unique()))


def _stake_book(
    rows: pd.DataFrame,
    book: str,
    limit: int | None,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, int]:
    if rows.empty:
        return pd.DataFrame(columns=list(_LEDGER_COLUMNS)), 0
    chosen = []
    for _day, day_rows in rows.groupby("slate_date", sort=True):
        ordered = day_rows.sort_values(
            ["ev", "game", "market", "selection"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        if limit is not None:
            ordered = ordered.head(limit)
        chosen.append(ordered)
    slate = pd.concat(chosen, ignore_index=True) if chosen else rows.iloc[0:0]
    bankroll = round(float(config.bankroll), 2)
    skipped = 0
    placed = []
    ranks: dict[pd.Timestamp, int] = {}
    for row in slate.itertuples(index=False):
        day = pd.Timestamp(row.slate_date)
        ranks[day] = ranks.get(day, 0) + 1
        stake = _stake_size(bankroll, config)
        if stake <= 0:
            skipped += 1
            continue
        profit = round(stake * float(row.unit_profit), 2)
        bankroll = round(bankroll + profit, 2)
        placed.append(
            {
                "slate_date": day,
                "book": book,
                "game_id": getattr(row, "game_id", ""),
                "game": row.game,
                "market": row.market,
                "selection": row.selection,
                "line": row.line,
                "model_probability": row.model_probability,
                "push_probability": getattr(row, "push_probability", 0.0),
                "implied_probability": row.implied_probability,
                "american": row.american,
                "decimal_odds": row.decimal_odds,
                "ev": row.ev,
                "rank": ranks[day],
                "stake": stake,
                "actual_outcome": row.actual_outcome,
                "result": row.result,
                "profit": profit,
                "bankroll": bankroll,
            }
        )
    ledger = pd.DataFrame(placed, columns=list(_LEDGER_COLUMNS))
    return ledger, skipped


def _stake_size(bankroll: float, config: BacktestConfig) -> float:
    if bankroll <= 0:
        return 0.0
    requested = (
        float(config.stake)
        if config.stake_mode == "flat"
        else float(config.stake) * bankroll
    )
    return round(min(requested, bankroll), 2)


def _summarize(
    strategy, book, ledger, skipped, config: BacktestConfig
) -> dict[str, object]:
    wins = int(ledger["result"].eq("win").sum()) if not ledger.empty else 0
    losses = int(ledger["result"].eq("loss").sum()) if not ledger.empty else 0
    pushes = int(ledger["result"].eq("push").sum()) if not ledger.empty else 0
    wagered = float(ledger["stake"].sum()) if not ledger.empty else 0.0
    profit = float(ledger["profit"].sum()) if not ledger.empty else 0.0
    decided = wins + losses
    return {
        "strategy": strategy,
        "book": book,
        "total_bets": int(len(ledger)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": np.nan if decided == 0 else wins / decided,
        "amount_wagered": wagered,
        "profit": profit,
        "roi": np.nan if wagered == 0 else profit / wagered,
        "avg_ev": np.nan if ledger.empty else float(ledger["ev"].mean()),
        "avg_decimal_odds": np.nan
        if ledger.empty
        else float(ledger["decimal_odds"].mean()),
        "ending_bankroll": round(float(config.bankroll) + profit, 2),
        "skipped_unfunded": int(skipped),
    }
