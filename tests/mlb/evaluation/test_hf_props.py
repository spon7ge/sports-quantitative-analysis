"""Hugging Face strikeout-prop import and 2026 table assembly (offline)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from src.mlb.evaluation.backtest import run_backtest
from src.mlb.markets.hf_props import (
    collapse_last_pre_start,
    pair_over_under,
    quotes_from_paired,
    select_asof_strikeout_quotes,
)
from src.mlb.pipeline.gamelogs import normalize_player_name, parse_game_log_splits
from src.mlb.pipeline.hf_tables import (
    HF_2026_FOLDS,
    assemble_hf_backtest_tables,
    map_quotes_to_starts,
)
from src.mlb.schemas import MARKET_QUOTE_COLUMNS


def test_collapse_keeps_last_tick_before_start() -> None:
    start = pd.Timestamp("2026-05-18 22:40:00", tz="UTC")
    ticks = pd.DataFrame(
        {
            "game_id": ["m~a", "m~a", "m~a"],
            "start_time": [start, start, start],
            "player": ["max meyer", "max meyer", "max meyer"],
            "line": [5.5, 5.5, 5.5],
            "side": ["over", "over", "over"],
            "book": ["pinnacle", "pinnacle", "pinnacle"],
            "ts": [
                start - pd.Timedelta(hours=4),
                start - pd.Timedelta(hours=2),
                start + pd.Timedelta(minutes=5),
            ],
            "odds": [1.95, 1.91, 1.50],
            "result": [6.0, 6.0, 6.0],
            "won": [True, True, False],
        }
    )
    collapsed = collapse_last_pre_start(ticks)
    assert len(collapsed) == 1
    assert abs(float(collapsed.iloc[0]["odds"]) - 1.95) < 1e-12


def test_pair_and_select_prefers_pinnacle_near_five_five() -> None:
    start = pd.Timestamp("2026-05-18 22:40:00", tz="UTC")
    ts = start - pd.Timedelta(hours=3)
    ticks = pd.DataFrame(
        {
            "game_id": ["m~a"] * 8,
            "start_time": [start] * 8,
            "player": ["Max Meyer"] * 8,
            "line": [4.5, 4.5, 5.5, 5.5, 5.5, 5.5, 6.5, 6.5],
            "side": ["over", "under", "over", "under", "over", "under", "over", "under"],
            "book": [
                "pinnacle",
                "pinnacle",
                "bet365",
                "bet365",
                "pinnacle",
                "pinnacle",
                "pinnacle",
                "pinnacle",
            ],
            "ts": [ts] * 8,
            "odds": [1.80, 2.05, 1.87, 1.95, 1.91, 1.91, 2.10, 1.75],
            "result": [6.0] * 8,
            "won": [True] * 8,
        }
    )
    paired = pair_over_under(collapse_last_pre_start(ticks))
    selected = select_asof_strikeout_quotes(paired)
    assert len(selected) == 1
    assert float(selected.iloc[0]["line"]) == 5.5
    assert selected.iloc[0]["book"] == "pinnacle"
    quotes = quotes_from_paired(
        selected,
        game_pk=pd.Series([824277]),
        pitcher_id=pd.Series([676974]),
    )
    assert list(quotes.columns) == list(MARKET_QUOTE_COLUMNS)
    assert quotes.iloc[0]["price_format"] == "decimal"


def test_normalize_player_name_strips_accents_and_suffix() -> None:
    assert normalize_player_name("José Ramírez Jr.") == "jose ramirez"
    assert normalize_player_name("Max Meyer") == "max meyer"


def test_parse_game_log_splits_keeps_starters_only() -> None:
    splits = [
        {
            "season": "2026",
            "date": "2026-05-18",
            "isHome": True,
            "stat": {
                "gamesStarted": 1,
                "strikeOuts": 6,
                "battersFaced": 23,
                "numberOfPitches": 90,
                "outs": 18,
                "inningsPitched": "6.0",
            },
            "team": {"id": 146},
            "opponent": {"id": 115},
            "game": {"gamePk": 824277, "gameNumber": 1},
        },
        {
            "season": "2026",
            "date": "2026-05-19",
            "isHome": False,
            "stat": {"gamesStarted": 0, "strikeOuts": 2, "battersFaced": 4},
            "team": {"id": 146},
            "opponent": {"id": 121},
            "game": {"gamePk": 824300, "gameNumber": 1},
        },
    ]
    rows = parse_game_log_splits(splits, pitcher_id=676974, pitcher_hand="R", season=2026)
    assert len(rows) == 1
    assert rows[0]["strikeouts"] == 6
    assert rows[0]["role"] == "starter"


def _synthetic_hf_bundle() -> dict[str, pd.DataFrame]:
    pitchers = (
        (676974, "Max Meyer", "R"),
        (671096, "Andrew Abbott", "L"),
    )
    start_rows = []
    quote_rows = []
    for season, n_starts, start_day in ((2025, 16, 10), (2026, 8, 1)):
        month = "04" if season == 2025 else "05"
        for p_i, (pitcher_id, name, hand) in enumerate(pitchers):
            for j in range(n_starts):
                day = start_day + j
                game_pk = season * 10000 + p_i * 100 + j
                date = f"{season}-{month}-{day:02d}"
                start = pd.Timestamp(f"{date} 23:10:00", tz="UTC")
                k = 4 + (j % 5)
                start_rows.append(
                    {
                        "pitcher_id": pitcher_id,
                        "game_pk": game_pk,
                        "game_date": date,
                        "season": season,
                        "strikeouts": k,
                        "batters_faced": 22 + j % 4,
                        "pitches": 85 + j,
                        "outs": 15 + j % 6,
                        "role": "starter",
                        "is_home": int(j % 2 == 0),
                        "opponent_team_id": 115,
                        "team_id": 146 if p_i == 0 else 113,
                        "venue_id": 1,
                        "pitcher_hand": hand,
                        "scheduled_start_utc": start,
                        "event_time_utc": start,
                        "ingested_at_utc": start,
                        "doubleheader": 0,
                        "player_key": normalize_player_name(name),
                    }
                )
                if season == 2026:
                    quote_rows.append(
                        {
                            "game_id": f"m~{game_pk}",
                            "player": name.lower(),
                            "book": "pinnacle",
                            "line": 5.5,
                            "over_price": 1.91,
                            "under_price": 1.91,
                            "start_time": start,
                            "fetched_at_utc": start - pd.Timedelta(hours=3),
                            "result": float(k),
                            "player_key": normalize_player_name(name),
                        }
                    )
    starts = pd.DataFrame(start_rows)
    quotes = pd.DataFrame(quote_rows)
    people = pd.DataFrame(
        {
            "mlb_id": [p[0] for p in pitchers],
            "key_mlbam": [p[0] for p in pitchers],
            "key_fangraphs": [pd.NA, pd.NA],
            "key_bbref": [pd.NA, pd.NA],
            "key_retro": [pd.NA, pd.NA],
            "name": [p[1] for p in pitchers],
            "bats": ["L", "L"],
            "throws": [p[2] for p in pitchers],
            "is_pitcher": [1, 1],
            "player_key": [normalize_player_name(p[1]) for p in pitchers],
        }
    )
    schedule = pd.DataFrame(
        {
            "game_pk": starts["game_pk"],
            "scheduled_start_utc": starts["scheduled_start_utc"],
            "status": "Final",
            "home_team_id": starts["team_id"],
            "away_team_id": starts["opponent_team_id"],
            "venue_id": 1,
            "doubleheader": 0,
            "probable_home_pitcher_id": starts["pitcher_id"],
            "probable_away_pitcher_id": pd.NA,
            "official_date": starts["game_date"],
        }
    )
    return {
        "starts": starts,
        "quotes": quotes,
        "people": people,
        "schedule": schedule,
    }


def test_map_quotes_to_unique_start(mlb_config) -> None:
    bundle = _synthetic_hf_bundle()
    mapped = map_quotes_to_starts(bundle["quotes"], bundle["starts"])
    assert not mapped.empty
    assert mapped["game_pk"].notna().all()
    assert mapped["pitcher_id"].isin({676974, 671096}).all()


def test_assemble_and_backtest_hf_synthetic(mlb_config, tmp_path) -> None:
    bundle = _synthetic_hf_bundle()
    cfg = replace(
        mlb_config,
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        folds=HF_2026_FOLDS,
        workload_min_train_starts=8,
    )
    tables = assemble_hf_backtest_tables(
        cfg,
        quotes=bundle["quotes"],
        starts=bundle["starts"],
        people=bundle["people"],
        schedule=bundle["schedule"],
        http=lambda url, params=None: b"{}",
    )
    assert not tables["market_quotes"].empty
    assert tables["market_quotes"]["price_format"].eq("decimal").all()
    result = run_backtest(tables, cfg)
    scores = result["scores"]
    assert not scores.empty
    assert "hf_may_2026" in set(scores["fold"])
    market = result["market"]
    assert not market.empty
    assert (market["roi_type"] == "quoted_price_simulation").all()
    assert "realized" not in market.columns.str.lower().to_list()
