"""``add_starter_features`` must reproduce the minutes notebook's inline block."""

import numpy as np
import pandas as pd

from src.features.minutes import add_starter_features


def _notebook_block(df: pd.DataFrame) -> pd.DataFrame:
    # Frozen copy of the original minutes_xgboost cell.
    N = 10
    d = df.copy()
    d['game_date'] = pd.to_datetime(d['game_date'])
    d['mins'] = d['minutes']
    d['started'] = d['starting'].fillna(0).astype(int)
    d['played'] = (d['mins'] > 0).astype(int)
    sched = (d[['season_year', 'team_id', 'game_id', 'game_date']]
             .drop_duplicates(['team_id', 'game_id'])
             .sort_values(['team_id', 'game_date']))
    sched['tg_num'] = sched.groupby(['season_year', 'team_id']).cumcount()
    tenure = (d.groupby(['season_year', 'team_id', 'player_id'])['game_date']
               .agg(first='min', last='max').reset_index())
    grid = tenure.merge(sched, on=['season_year', 'team_id'])
    grid = grid[(grid['game_date'] >= grid['first']) & (grid['game_date'] <= grid['last'])]
    grid = grid.merge(d[['team_id', 'game_id', 'player_id', 'mins', 'started', 'played']],
                      on=['team_id', 'game_id', 'player_id'], how='left')
    grid[['mins', 'started', 'played']] = grid[['mins', 'started', 'played']].fillna(0)
    grid = grid.sort_values(['season_year', 'team_id', 'player_id', 'tg_num'])
    g = grid.groupby(['season_year', 'team_id', 'player_id'])
    grid['starts_lN'] = g['started'].transform(
        lambda s: s.shift(1).rolling(N, min_periods=1).sum())
    grid['mins_played'] = grid['mins'].where(grid['played'] == 1)
    grid['min_avg_lN'] = g['mins_played'].transform(
        lambda s: s.shift(1).rolling(N, min_periods=1).mean())
    grid = grid.sort_values(['team_id', 'game_id', 'starts_lN', 'min_avg_lN'],
                            ascending=[True, True, False, False])
    grid['starter_rank'] = grid.groupby(['team_id', 'game_id']).cumcount() + 1
    grid['is_main_starter'] = ((grid['starter_rank'] <= 5) & (grid['starts_lN'] > 0)).astype(int)
    main = grid[grid['is_main_starter'] == 1]
    team_feats = (main.assign(missing_min=main['min_avg_lN'].where(main['played'] == 0, 0))
                      .groupby(['team_id', 'game_id'])
                      .agg(main_starters_playing=('played', 'sum'),
                           main_starters_defined=('player_id', 'size'),
                           starter_min_missing=('missing_min', 'sum'))
                      .reset_index())
    df = (df.merge(team_feats, on=['team_id', 'game_id'], how='left')
            .merge(grid[['team_id', 'game_id', 'player_id', 'is_main_starter', 'min_avg_lN', 'starter_rank']],
                   on=['team_id', 'game_id', 'player_id'], how='left'))
    own_missing = np.where((df['is_main_starter'] == 1) & (df['minutes'].fillna(0) == 0),
                           df['min_avg_lN'].fillna(0), 0)
    df['starter_min_missing_teammates'] = df['starter_min_missing'] - own_missing
    df['main_starters_out'] = df['main_starters_defined'] - df['main_starters_playing']
    return df


def _synthetic_gamelogs(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2023-10-24", periods=30, freq="2D")
    for team in (1, 2):
        roster = [team * 100 + i for i in range(12)]
        for g, date in enumerate(dates):
            game_id = f"g{g:03d}"
            for rank, player in enumerate(roster):
                if rng.random() < 0.12:
                    continue
                starter = int(rank < 5 and rng.random() < 0.9)
                minutes = float(rng.uniform(24, 38) if starter else rng.uniform(0, 22))
                rows.append({
                    "season_year": "2023-24",
                    "team_id": team,
                    "game_id": game_id,
                    "game_date": date,
                    "player_id": player,
                    "minutes": minutes,
                    "starting": starter,
                })
    return pd.DataFrame(rows)


def test_shared_starter_features_match_notebook_block():
    frame = _synthetic_gamelogs()
    expected = _notebook_block(frame)
    actual = add_starter_features(frame)
    pd.testing.assert_frame_equal(actual, expected)
