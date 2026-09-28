import pandas as pd

from src.mlb.pipeline.walk_forward_features import feature_block


def _start(**kwargs):
    base = {
        "pitcher_id": 1,
        "game_pk": 1,
        "season": 2021,
        "game_date": "2021-04-01",
        "scheduled_start_utc": "2021-04-01T20:00:00Z",
        "is_home": 1,
        "opponent_team_id": 10,
        "strikeouts": 6,
        "batters_faced": 20,
        "outs": 15,
        "pitches": 80,
    }
    base.update(kwargs)
    return base


def test_same_day_doubleheader_does_not_enter_history():
    history = pd.DataFrame([_start(game_pk=1, game_date="2021-04-01", strikeouts=10, batters_faced=25)])
    block = pd.DataFrame(
        [
            _start(game_pk=2, game_date="2021-04-10", scheduled_start_utc="2021-04-10T17:00:00Z", strikeouts=4, batters_faced=18),
            _start(game_pk=3, game_date="2021-04-10", scheduled_start_utc="2021-04-10T23:00:00Z", strikeouts=9, batters_faced=22),
        ]
    )
    out = feature_block(history, block, m=50, kappa=3, rest_cap=30)
    assert out.loc[out["game_pk"] == 3, "pitcher_prior_regular_starts"].iloc[0] == 1
    assert out.loc[out["game_pk"] == 2, "pitcher_prior_regular_starts"].iloc[0] == 1


def test_earlier_date_in_the_block_updates_raw_sums_not_the_prior():
    history = pd.DataFrame(
        [_start(game_pk=1, game_date="2021-04-01", strikeouts=5, batters_faced=20)]
    )
    block = pd.DataFrame(
        [
            _start(game_pk=2, game_date="2021-04-06", strikeouts=1, batters_faced=10),
            _start(game_pk=3, game_date="2021-04-11", strikeouts=8, batters_faced=24),
        ]
    )
    out = feature_block(history, block, m=100, kappa=2, rest_cap=30)
    first_prior = out.loc[out["game_pk"] == 2, "league_k_per_bf_prior"].iloc[0]
    second_prior = out.loc[out["game_pk"] == 3, "league_k_per_bf_prior"].iloc[0]
    assert first_prior == second_prior == 5 / 20
    assert out.loc[out["game_pk"] == 3, "pitcher_prior_regular_starts"].iloc[0] == 2


def test_first_start_rest_is_the_median_of_capped_training_rest():
    history = pd.DataFrame(
        [
            _start(pitcher_id=7, game_pk=1, game_date="2021-04-01"),
            _start(pitcher_id=7, game_pk=2, game_date="2021-04-20", scheduled_start_utc="2021-04-20T20:00:00Z"),
        ]
    )
    block = pd.DataFrame([_start(pitcher_id=9, game_pk=3, game_date="2021-05-01")])
    out = feature_block(history, block, m=50, kappa=3, rest_cap=14)
    row = out.iloc[0]
    assert row["no_prior_regular_start"] == 1
    assert row["days_rest_capped"] == 14


def test_opponent_season_to_date_falls_back_to_cross_season_last10():
    history = pd.DataFrame(
        [
            _start(
                pitcher_id=7,
                game_pk=0,
                season=2021,
                game_date="2021-04-01",
                opponent_team_id=99,
                strikeouts=1,
                batters_faced=20,
            ),
            _start(
                pitcher_id=7,
                game_pk=99,
                season=2021,
                game_date="2021-04-15",
                scheduled_start_utc="2021-04-15T20:00:00Z",
                opponent_team_id=99,
                strikeouts=1,
                batters_faced=20,
            ),
            _start(
                pitcher_id=2,
                game_pk=1,
                season=2021,
                game_date="2021-09-01",
                opponent_team_id=10,
                strikeouts=2,
                batters_faced=20,
            ),
            _start(
                pitcher_id=3,
                game_pk=2,
                season=2021,
                game_date="2021-09-10",
                scheduled_start_utc="2021-09-10T20:00:00Z",
                opponent_team_id=10,
                strikeouts=8,
                batters_faced=20,
            ),
        ]
    )
    block = pd.DataFrame(
        [
            _start(
                pitcher_id=1,
                game_pk=100,
                season=2022,
                game_date="2022-04-01",
                opponent_team_id=10,
                strikeouts=5,
                batters_faced=20,
            )
        ]
    )
    out = feature_block(history, block, m=50, kappa=3, rest_cap=30)
    row = out.iloc[0]
    league = row["league_k_per_bf_prior"]
    last10 = row["opponent_k_rate_vs_starters_last10_smoothed"]
    season_to_date = row["opponent_k_rate_vs_starters_season_to_date_smoothed"]
    assert season_to_date == last10
    assert last10 != league
    assert season_to_date != league
