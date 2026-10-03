from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuoteSpec:
    identity_cols: tuple[str, ...]
    compare_cols: tuple[str, ...]


_PRIZEPICKS = QuoteSpec(
    identity_cols=("league", "player_name", "stat_type", "odds_type"),
    compare_cols=("line_score",),
)
_UNDERDOG = QuoteSpec(
    identity_cols=("league", "player_name", "stat_name", "side"),
    compare_cols=("line_score", "american_price", "payout_multiplier"),
)
_BOOK_PROPS = QuoteSpec(
    identity_cols=("league", "player_name", "market_type", "side"),
    compare_cols=("line_score", "american_price"),
)
_PARLAY_PROPS = QuoteSpec(
    identity_cols=("sportsbook", "league", "player_name", "market_type", "side"),
    compare_cols=("line_score", "american_price"),
)
_PINNACLE_TEAM = QuoteSpec(
    identity_cols=(
        "league",
        "away_team",
        "home_team",
        "market_type",
        "period",
        "is_alternate",
        "side",
    ),
    compare_cols=("points", "american_price", "decimal_price"),
)
_EXCHANGE_PROPS = QuoteSpec(
    identity_cols=("league", "event_id", "player_name", "stat_name", "side"),
    compare_cols=("line_score", "american_price", "stake"),
)
_EXCHANGE_TEAM = QuoteSpec(
    identity_cols=("league", "event_id", "market_type", "side"),
    compare_cols=("points", "american_price", "stake"),
)

QUOTE_SPECS: dict[str, QuoteSpec] = {
    "wnba_prizepicks": _PRIZEPICKS,
    "nba_prizepicks": _PRIZEPICKS,
    "mlb_prizepicks": _PRIZEPICKS,
    "wnba_underdogs": _UNDERDOG,
    "nba_underdogs": _UNDERDOG,
    "mlb_underdogs": _UNDERDOG,
    "wnba_pinnacle": _BOOK_PROPS,
    "nba_pinnacle": _BOOK_PROPS,
    "mlb_pinnacle": _BOOK_PROPS,
    "nfl_pinnacle": _BOOK_PROPS,
    "wnba_fanduel": _BOOK_PROPS,
    "nba_fanduel": _BOOK_PROPS,
    "mlb_fanduel": _BOOK_PROPS,
    "wnba_draftkings": _BOOK_PROPS,
    "nba_draftkings": _BOOK_PROPS,
    "mlb_draftkings": _BOOK_PROPS,
    "wnba_parlay_api_odds": _PARLAY_PROPS,
    "mlb_parlay_api_odds": _PARLAY_PROPS,
    "wnba_pinnacle_team": _PINNACLE_TEAM,
    "nba_pinnacle_team": _PINNACLE_TEAM,
    "mlb_pinnacle_team": _PINNACLE_TEAM,
    "nfl_pinnacle_team": _PINNACLE_TEAM,
    "mlb_prophetx": _EXCHANGE_PROPS,
    "nba_prophetx": _EXCHANGE_PROPS,
    "wnba_prophetx": _EXCHANGE_PROPS,
    "nfl_prophetx": _EXCHANGE_PROPS,
    "mlb_prophetx_team": _EXCHANGE_TEAM,
    "nba_prophetx_team": _EXCHANGE_TEAM,
    "wnba_prophetx_team": _EXCHANGE_TEAM,
    "nfl_prophetx_team": _EXCHANGE_TEAM,
    "mlb_novig": _EXCHANGE_PROPS,
    "nba_novig": _EXCHANGE_PROPS,
    "mlb_novig_team": _EXCHANGE_TEAM,
    "nba_novig_team": _EXCHANGE_TEAM,
    "wnba_novig": _EXCHANGE_PROPS,
    "wnba_novig_team": _EXCHANGE_TEAM,
    "nfl_novig": _EXCHANGE_PROPS,
    "nfl_novig_team": _EXCHANGE_TEAM,
}


def get_quote_spec(table: str) -> QuoteSpec:
    return QUOTE_SPECS[table]
