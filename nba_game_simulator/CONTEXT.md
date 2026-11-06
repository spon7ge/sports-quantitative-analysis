# Game environment

The shared draw underneath one NBA game, oriented to the team a prop is about. Layer 1 is settled in `game_enviroment.py`: one environment per game, the same rows for every player on both teams. Layer 2 is settled: one role per player from that player's side, his team and the opponent included, with that environment's script copied on. Layer 3 is the rotation, which turns those roles into minutes inside the script. Layer 4 turns those minutes and the possession counts into the shots and turnovers the teammates share. Layer 5 credits the assists, rebounds, steals, and blocks those shots and turnovers already produced. Layer 6 keeps that row as one box score. A joint prop is the share of rows where every line clears.

## Language

### One game

**Game environment**:
One draw of a game's length, possessions, and final score, shared by both teams. Columns are `team_poss`, `opp_poss`, `pace`, `team_pts`, `opp_pts`, `game_total`, `is_home`, `n_ot`, `game_minutes`, `team_margin`.
_Avoid_: a draw per player, a home-team-only frame

**Player's team**:
The team the prop belongs to. Every column is that team's view. The opponent's players use the same rows with the margin sign flipped.
_Avoid_: home team, when the player is away

**Home**:
Whether the player's team is at home (`is_home` 1 or 0). Training stays home/away; the environment is already translated.
_Avoid_: treating the first column as the home side

**Spread**:
The player's team's points spread on the environment call. Negative means that team is favored. Implied team points are half the total minus half the spread; the opponent's implied points are the rest of the total.
_Avoid_: home spread, the opponent's number, a plus line for a favorite

**Market weight**:
The share of each team's points taken from the implied number. A posted total and spread use one half. A missing line uses none, and points then equal possessions times efficiency.
_Avoid_: replacing the possession draw with the market total

**Pace**:
Regulation possessions per 48 minutes, averaged across the two teams. Overtime does not raise it.
_Avoid_: team possessions, the full-game count

**Team possessions**:
The player's team's full-game possession count. Overtime lengthens it by the same fraction the game is longer than 48 minutes.
_Avoid_: pace

**Opponent possessions**:
The opponent's full-game possession count, lengthened the same way. Possession allocation uses these two counts. Minutes do not.
_Avoid_: pace

**Team points**:
The player's team's final points, overtime included. Half the implied number and half possessions times efficiency when a line is posted. On a box-score row, the players' points sum to the nearest whole number of this total. Shot attempts still follow possessions. Makes and free throws move so the points land on this score.
_Avoid_: the market implied points alone, a second point total from the shot draw

**Team margin**:
Team points minus opponent points, after overtime. This is the final script, not the pre-game spread.
_Avoid_: spread, expected margin, home margin

**Overtime**:
How many extra periods the game drew, from the absolute regulation margin. Each period is five minutes. Possessions scale with the added time. Extra points land only on the share the market does not already fix, so a full market weight keeps the points on the implied number while the possession count still grows.
_Avoid_: an overtime coin flip, overtime drawn from the spread

**Game minutes**:
48 plus five for each overtime period. This is the per-player cap. Five times it is the team's minute budget.
_Avoid_: a flat 48, a flat 240

**Game script**:
The `team_margin`, `n_ot`, and `game_minutes` on one environment row. Layer 2 copies it onto the role. Layer 3 reads it as the final margin (sign flipped for the other team), `went_ot` when `n_ot > 0`, team minute budget `5 * game_minutes`, and cap `game_minutes`.
_Avoid_: a second margin draw, a second overtime draw

**Blowout**:
A game whose absolute final margin is at least 20. Whether this game is a blowout is read off `team_margin`.
_Avoid_: a separate blowout coin flip, `|expected margin|` as the simulated result

**Shared shock**:
The part of the possession-rate draw that hits both teams together. Each team also keeps a smaller shock of its own.
_Avoid_: independent pace draws, one environment per teammate

### Before tip-off

**As-of**:
Pace, rest, and starter ratings for a date come only from games played before that date.
_Avoid_: that night's box score, closing numbers as features

**Rest**:
Days since that team's previous game, from 1 to 7. One day or less is a back-to-back. Three days or more is rested.

**Starting five**:
The five player ids posted to start one side. A side with no posted names uses the training-average five.
_Avoid_: an empty list, projected minutes

**Slate**:
The player's team, opponent, home flag, spread, total, and the two starting fives for one game.
_Avoid_: the whole board

**Latest odds**:
The spread and total on the single newest Pinnacle team snapshot. A team absent from that file is off the board.
_Avoid_: an earlier file from the same day, a hardcoded date

**Latest lineups**:
Starting fives from the newest Rotowire file that day. A team that has left the page keeps the last five posted earlier that same day.
_Avoid_: a blank five for a team that tipped off and dropped off the later page

**Role**:
One player's pregame view of one game. It holds his role history, his team's form, the opponent's form, and what was posted for this game: home, start, and spread.
_Avoid_: a minutes draw, the final score

**Role history**:
The player's minutes, start rate, and fouls, and his team's and the opponent's pace, offense, defense, and rest, taken from games played before the date being scored.
_Avoid_: that game's box score, the opponent from the previous game when this game names a different team

**Training role**:
A role on a game that has already been played. The windows stop before that game. The script on the row is the result that was played, and the foul total is that game's fouls.
_Avoid_: counting that game's minutes in the window

**Live role**:
A role for a game that has not been played. Every completed game before that day stays in the window. The script is attached afterward from a Layer 1 draw.
_Avoid_: dropping the latest completed game, a row invented only so it can be discarded

**Posted start**:
Whether this player is in the starting five for the game being scored.
_Avoid_: his start rate, the previous game's lineup

**Foul total**:
Personal fouls in a game already played. On a training role it is the result, beside the script.
_Avoid_: the foul rate in the role history

**Attached script**:
A game script copied onto a role. Teammates share it. The other team's roles use the same draw with the margin sign flipped.
_Avoid_: a second draw, the pre-game spread

### What Layer 3 still owns

**Rotation**:
The minutes each player on one team plays in one attached script. They sum to the team minute budget, and no player exceeds game minutes.
_Avoid_: independent minutes that ignore his teammates, a budget of 240 when the script went to overtime

**Conditional minutes**:
One player's minutes given his role and that script, before the teammates are forced to sum to the budget.
_Avoid_: a margin or overtime drawn for the minutes alone, a 48-minute cap when the script went to overtime

**Game state**:
Score margin, time remaining, fouls, and who is on the floor while the game is being played. The rotation uses the final script. It does not need this path.
_Avoid_: re-drawing possessions, pace, or the final score

### What Layer 4 still owns

**Shot pool**:
The field-goal attempts on the player's team in one simulation. Its mean is that team's possession count times its recent attempts per possession. The count is the environment's.
_Avoid_: a new pace draw, the opponent's possession count

**Opportunity share**:
The fraction of the shot pool one player receives. It moves with his minutes and his recent attempts per minute. The shares on a team sum to one.
_Avoid_: an independent attempt draw, a share that ignores his teammates

**Shot profile**:
How a player turns an attempt into twos, threes, and free throws. On a training row the profile stops before that game. On a live row it keeps the latest completed game.
_Avoid_: that game's box score, a second profile for the opponent

**Turnover pool**:
The player's team's turnovers in one simulation. Its mean is that team's possession count times its recent turnovers per possession. Players share that pool.
_Avoid_: a turnover drawn apart from the possession count

### What Layer 5 still owns

**Assist**:
A made field goal credited to a teammate of the scorer. The make is the one already in the shot pool. The scorer does not assist himself.
_Avoid_: an assist drawn apart from a make, credit to the player who scored

**Rebound**:
A missed field goal that stays in play. The offense's recent rate splits those misses between the two teams. Teammates then share their team's rebounds.
_Avoid_: a rebound drawn for a player on his own, a second shot to create the miss

**Steal**:
A turnover by the opponent that the player's team takes. The turnover count is the opponent's.
_Avoid_: a steal drawn from the player's own turnovers

**Block**:
An opponent field-goal attempt that missed and was blocked. A made shot is not a block.
_Avoid_: a block of a shot that went in, a new attempt drawn for the block

### What Layer 6 still owns

**Box score**:
The minutes, shots, and events of one simulation, on one row for every player in the game.
_Avoid_: a fresh draw, a stat taken from a different simulation

**Prop**:
One player clearing one stat line. The line clears only when the stat is strictly above it.
_Avoid_: a push counted as a win, a line read from a different simulation

**Joint probability**:
The share of simulations in which every prop in a group clears. It is counted on the same row.
_Avoid_: the product of the separate chances, a probability from independent draws
