# Minutes discovery source map (01)

Date: 2026-09-10  
Approved: source map in notebook 01, unique-season silver concat, descriptive minutes slices.

## Goal

Notebook 01 inventories every information source that could explain playing minutes, then describes how minutes move across pregame (or explicitly tagged post-tip) slices. It does not promote features. Notebook 03 still owns ablation.

## Load

Glob `data/silver/nba/*/regular_season/player_gamelogs.parquet`. Prefer the folder whose name matches `season_year`. Drop duplicate `game_id`/`player_id` so five copies of 2021-22 collapse to one panel. After real fetches, multiple seasons remain.

## Source map

Each driver has: mechanism, candidate fields, feed, availability (`pregame_derived` / `needs_feed` / `post_tip`). Audit presence and coverage only.

## Slices (appearances with minutes > 0)

| Slice | Construction | Tag |
|---|---|---|
| Favorite / underdog / pick'em | `player_team_spread` sign | pregame |
| Heavy favorite / heavy underdog | spread ≤ −9 / ≥ +9 | pregame |
| Start vs not | `start_position` non-empty | post_tip; also prior-game starter |
| Opp defensive rating | opponent season-to-date `team_def_rating` entering the game | pregame derived |
| Pace | own team rolling-5 `team_pace` entering the game | pregame derived |
| Rest / B2B | days since player's last appearance; B2B = 1 | pregame derived |
| Heavy minutes last week | prior 7-day minutes sum; count of 32+ min games | pregame derived |
| Season phase | team game number tertiles start / mid / end | pregame derived |

Same-game `pace` and `opp_def_rating` are not used for these slices.

## Out of scope

Correlations for promotion, Rotowire re-scrape, fetching missing seasons.
