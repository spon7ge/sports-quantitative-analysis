# Feature audit

Every column in `MIN_FEATURES` and in the rate-model list is safe. The frozen minutes bundle does not need a refit.

`starting` is tonight’s posted starting five. That five is known before tip, so the flag is pregame. The silver value is the tracking field `position` (`G` / `F` / `C` if he started, blank if he came off the bench). It records who took the opening tip. It does not read minutes, points, or anything that happens after tip. The tail sidecar groups on the same flag, which is the same pre-tip fact.

On 50 random appearances, each rolling feature recomputed from games with `game_date` strictly before that row matched the full-panel value. Maximum absolute error was 0. No column is leaky or `needs-final-injury-report`.

Read-only. No model, feature code, or silver file was changed.

## Contracts

Minutes: `MIN_FEATURES` in `notebooks/nba/minutes/min_nba_model.ipynb` (35 columns). Those names are the saved bundle’s `feature_names`, in the same order. The bundle’s train end is 2025-04-13 and its validation end is 2026-04-12.

Rate: `PTS_FEATURES` in `notebooks/nba/points/pts_nba_model.ipynb` (48 columns) = `starting` plus `DIRECT_POINTS_FEATURES` from `add_points_features`. The notebook still fits `pts`. `predicted_minutes_oof` is not in this list. There is no frozen V1 rate bundle.

`starting` is not built by either feature function. Both notebooks set it before the builder:

```python
df["starting"] = (
    df["start_position"].astype("string").str.strip().isin(["G", "F", "C"]).astype(int)
)
```

`start_position` is the tracking box-score field `position` (`src/pipeline/silver/columns.py`). On the full silver panel (176,738 player-games, 2019-20 through 2025-26) the values are only blank, `G`, `F`, and `C`, and they form a starting five every game: 33,156 `G`, 33,156 `F`, 16,578 `C`, 93,848 blank. `starting` equals that G/F/C flag on every row (mean 0.469).

`start_rate_10` is a different column. It is the prior-10 share of non-empty `start_position`, and it ignores the current game. On the audit sample, game `0022501092` (2026-03-30, player 1631248) has `starting` 1 and `start_rate_10` 0.2. Game `0022300952` (2024-03-13, player 1641764) has `starting` 0 and `start_rate_10` 0.9.

## How the recompute was done

Panel: `data/silver/nba/*/regular_season/player_gamelogs.parquet`, concatenated the same way as the notebooks, then `add_points_features` (which calls `add_minutes_features`).

Sample: 50 rows with `minutes > 0`, `numpy` Generator seed 26, without replacement. Dates run from 2019-10-27 to 2026-04-08, including 2025-26 rows. Prior-game counts on those rows run from 2 to 473. This check does not fit a model and does not edit a tail table.

For each sampled row, every rolling, season, stint, schedule, rate, and interaction column was recomputed using only that player’s games and both clubs’ team-games with `game_date` strictly earlier. Team-games were deduped the way `build_team_games` dedupes them (first row after sorting by `player_id`, `game_date`, `game_id`). Windows, EWM (`halflife=3`, `adjust=False`), sample standard deviations, season resets, adjacent-season shift, and stint resets follow `src/features/minutes/rolling.py`, `player.py`, `team.py`, `schedule.py`, and `src/features/points/`.

Season date ranges do not overlap (2019-20 ends 2020-08-14; 2020-21 starts 2020-12-22; the later seasons are the same shape). There are 0 player-date duplicates and 0 team-dates with two game ids, and player dates are monotonic, so a strict date cut matches the builder’s row shift.

`is_home`, `season_type_cat`, and the market columns were checked against the pregame fields on the same row (`matchup`, `season_type`, Rotowire total and signed spread). They also matched, error 0.

## Checks

| Check | Result |
|---|---|
| Rolling or season aggregates that include the current game or a later game | Clear for every aggregate in both lists. Prior-only recompute error was 0. |
| Starter flag or lineup taken from the same game’s box score | `starting` is the posted starting five, known before tip. `start_rate_10` and `season_start_rate` are prior windows of that same flag. |
| Teammate availability derived from who actually played tonight | No such column. `player_fga_share_10` is the player’s previous FGA over the team’s previous FGA. |
| Team or opponent stats that include the current game | Clear. `team_pace`, `team_net_rating`, `team_off_rating`, `team_def_rating`, and `team_fga` are one value per team-game (within-game nunique max is 1) and vary across the season (median 34 distinct paces and 70 distinct offensive ratings per team-season). The features shift one team-game before the window. |
| Same-game minutes, points, usage, fouls, or plus-minus | Those raw fields are not in either list. Lagged minutes, points, and usage matched the prior-only recompute. `pf`, `plus_minus`, `team_pf`, and `team_plus_minus` are on the silver rows and in neither list. |

## Safe columns

Fix for every row below: none.

Shared as-of mechanics:

- Player windows cross seasons. Order is `player_id`, `game_date`, `game_id`. `prior_shift` runs before the roll, EWM, or expanding aggregate.
- Season windows reset on `(player_id, season_year)` or `(team_id, season_year)`.
- A team stint resets when `team_id` changes. The current game’s team id assigns the stint; only earlier games in that stint enter the aggregate.
- Prior-season columns use the adjacent previous season (start year minus 1), and only when that season is the previous season present for the player. That season is over before the current one starts.
- Team and opponent windows are computed once on the team-game table, then joined on `game_id`. Opponent columns are the opponent’s own prior window.
- Per-minute rates are shift-then-sum of the counting stat over shift-then-sum of minutes. A zero minute sum is missing.
- Market inputs are Rotowire `Over_Under` and `Home_Line` (`data/bronze/rotowire/rotowire_nba_YYYY.csv`), joined on date and matchup and signed to the player’s team (`player_team_spread`). The file also has the final score; the features do not read it. There is no quote clock. The stored number is the line on the completed-game page, which is a pre-tip close and can already reflect late scratches.

### Minutes and rate

| Column | Source | As-of rule | Verdict |
|---|---|---|---|
| `starting` | Training: tracking `position` is `G` / `F` / `C`. Live: 1 when `player_name` is on that team’s Rotowire starting five, else 0 | Posted starting five, known before tip. Same flag the tail sidecar groups on | safe |
| `min_lag_1` | `target_minutes` (`min_sec`, else `min`, else `minutes`) | Previous appearance | safe |
| `min_ewm_hl_3` | same minutes | EWM halflife 3 on earlier appearances | safe |
| `min_std_10` | same minutes | Sample std of up to 10 earlier appearances (at least 2) | safe |
| `start_rate_10` | non-empty `start_position` on earlier games | Mean of up to 10 earlier games | safe |
| `season_min_mean` | same minutes | Expanding mean of earlier games in this `season_year` | safe |
| `current_team_min_mean` | same minutes | Expanding mean of earlier games in the current team stint | safe |
| `usg_wmean_10` | per-game `usg_pct` (0–1) times minutes | Sum of usage×minutes over sum of minutes, previous 10 | safe |
| `fga_per_min_10` | box `fga` and minutes | Sum of FGA over sum of minutes, previous 10 | safe |
| `team_pace_mean_10` | team advanced box `team_pace` for that team-game | Mean of up to 10 earlier team-games | safe |
| `opp_pace_mean_10` | opponent’s `team_pace` | Opponent’s prior-10 pace on this `game_id` | safe |
| `expected_possessions` | the two pace means | Average of `team_pace_mean_10` and `opp_pace_mean_10` | safe |
| `team_days_since_prev_game` | team schedule | Days since the previous team-game in this season. Season opener is missing | safe |
| `player_days_since_appearance` | dates with minutes > 0 | Days since the previous appearance in this season, capped at 30 | safe |
| `is_home` | `matchup` contains `vs.` | Known from the schedule | safe |
| `team_spread_canonical` | Rotowire home line, signed to the player’s team | Pregame close for this game | safe |
| `abs_spread` | absolute value of that spread | Same line | safe |
| `implied_team_total` | `game_total / 2 - spread / 2` | Same line and total | safe |

### Minutes only

| Column | Source | As-of rule | Verdict |
|---|---|---|---|
| `min_mean_20` | minutes | Mean of up to 20 earlier appearances | safe |
| `season_appearances_prior` | minutes > 0 | Count of earlier appearances in this season. Season opener is 0 | safe |
| `season_min_std` | minutes | Expanding sample std of earlier games in this season (at least 2) | safe |
| `season_start_rate` | non-empty `start_position` | Expanding mean of earlier games in this season | safe |
| `prior_season_min_mean` | minutes in the adjacent previous season | Full previous season, already complete | safe |
| `prior_season_appearances` | appearances in that season | Full previous season | safe |
| `current_team_rows_prior` | row count in the current team stint | Earlier games in the stint. First game with a team is 0 | safe |
| `assists_per_min_10` | tracking `assists` (matches box `ast`; correlation 0.999999 on all 176,738 rows) | Sum of assists over sum of minutes, previous 10 | safe |
| `team_net_rating_mean_10` | team advanced box `team_net_rating` | Mean of up to 10 earlier team-games | safe |
| `team_net_rating_season` | same net rating | Expanding mean of earlier team-games in this season | safe |
| `opp_net_rating_mean_10` | opponent’s `team_net_rating` | Opponent’s prior-10 net rating | safe |
| `team_games_prev_3d` | team schedule | Team-games in this season on `[game_date - 3 days, game_date)` | safe |
| `player_appearances_prev_7d` | appearance dates | Appearances in this season on `[game_date - 7 days, game_date)` | safe |
| `min_mean_3_minus_10` | `min_mean_3` minus `min_mean_10` | Both means are prior windows (3 and 10). They are not themselves in `MIN_FEATURES`; both recomputed with error 0 | safe |
| `min_mean_10_minus_season` | `min_mean_10` minus `season_min_mean` | Prior-10 mean minus same-season expanding mean | safe |
| `start_rate_x_abs_spread` | `start_rate_10` times `abs_spread` | Prior start rate times the pregame spread | safe |
| `min10_x_expected_possessions` | `min_mean_10` times `expected_possessions` | Prior-10 minutes times prior expected possessions | safe |

`current_team_min_mean` is in both contracts; the single definition is in the shared table.

### Rate only

| Column | Source | As-of rule | Verdict |
|---|---|---|---|
| `min_mean_3` | minutes | Mean of up to 3 earlier appearances | safe |
| `min_mean_10` | minutes | Mean of up to 10 earlier appearances | safe |
| `pts_lag_1` | box `pts` | Previous appearance | safe |
| `pts_mean_3` | box `pts` | Mean of up to 3 earlier appearances | safe |
| `pts_mean_10` | box `pts` | Mean of up to 10 earlier appearances | safe |
| `pts_mean_20` | box `pts` | Mean of up to 20 earlier appearances | safe |
| `pts_ewm_hl_3` | box `pts` | EWM halflife 3 on earlier appearances | safe |
| `pts_std_10` | box `pts` | Sample std of up to 10 earlier appearances (at least 2) | safe |
| `active_pts_mean_10` | box `pts` and minutes > 0 | Sum of points over count of appearances, previous 10 | safe |
| `season_pts_mean` | box `pts` | Expanding mean of earlier games in this season | safe |
| `pts_per_min_10` | points and minutes | Sum of points over sum of minutes, previous 10 | safe |
| `pts_per_min_20` | points and minutes | Same, previous 20 | safe |
| `season_pts_per_min` | points and minutes | Expanding sums within this season | safe |
| `current_team_pts_per_min` | points and minutes | Expanding sums within the current team stint | safe |
| `fg3a_per_min_10` | box `fg3_a` and minutes | Sum of three-point attempts over sum of minutes, previous 10 | safe |
| `fta_per_min_10` | box `fta` and minutes | Sum of free-throw attempts over sum of minutes, previous 10 | safe |
| `ts_agg_20` | points, `fga`, `fta` | `sum(pts) / (2 * (sum(fga) + 0.44 * sum(fta)))` over the previous 20 | safe |
| `three_attempt_rate_10` | `fg3_a` and `fga` | Sum of threes over sum of FGA, previous 10 | safe |
| `free_throw_rate_10` | `fta` and `fga` | Sum of FTA over sum of FGA, previous 10 | safe |
| `touches_per_min_10` | tracking `tchs` and minutes | Sum of touches over sum of minutes, previous 10 | safe |
| `player_fga_share_10` | player `fga` and team box `team_fga` on those same earlier games | Sum of player FGA over sum of team FGA, previous 10 | safe |
| `team_off_rating_mean_10` | team advanced box `team_off_rating` | Mean of up to 10 earlier team-games | safe |
| `opp_def_rating_mean_10` | opponent’s own `team_def_rating` | Opponent’s prior-10 defensive rating | safe |
| `game_total` | Rotowire `Over_Under` | Pregame close for this game | safe |
| `is_back_to_back` | `team_days_since_prev_game == 1` | 1 when the previous team-game in this season was yesterday, otherwise 0 (including the season opener) | safe |
| `season_type_cat` | `season_type` | Schedule label. All 176,738 silver rows are `Regular Season` (code 0) | safe |
| `pts_mean_3_minus_10` | `pts_mean_3` minus `pts_mean_10` | Both are prior point means | safe |
| `ppm_10_minus_season` | `pts_per_min_10` minus `season_pts_per_min` | Prior-10 rate minus same-season rate | safe |
| `fga_per_min_5_minus_season` | FGA per minute over 5 games, minus the same-season rate | Both pieces use earlier games only | safe |
| `usage_x_expected_possessions` | `usg_wmean_10` times `expected_possessions` | Prior usage times prior expected possessions | safe |

`expected_points_rate` and `expected_attempt_volume` are built from `predicted_minutes_oof` and are not in `PTS_FEATURES`.
