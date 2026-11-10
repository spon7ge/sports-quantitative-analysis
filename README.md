# Game-Level NBA Monte Carlo Simulation Engine

This project plays one NBA game many times. Each play is a full box score: the score, who was on the floor, and every player's points, rebounds, assists, and the rest.

"Monte Carlo" means we repeat the game with a little randomness, then count. If a player scores 25 or more in 6,200 of 10,000 plays, this model says that happens about 62% of the time. That 6,200 is an example of counting, not a result from a real slate.

Every play keeps one shared story. Play number 500 is the same imaginary game in every table. A fast game, a close score, and a player's big night stay tied together. Asking two questions about one game — a player over 25 points and a teammate over 8 rebounds — counts only the plays where both happened.

## The six steps

Each play walks through the same six steps, in order. A later step fills in detail. It does not throw away the score or the clock from an earlier step.

1. **The game script.** How fast the teams play, how many trips down the floor, whether it goes to overtime, and the final score. With play-by-play, the trip count and the pace follow the half and the margin, and overtime follows the late score.
2. **Minutes.** How long each player stays on the floor in that script. Fouls, a blowout, fatigue, a timeout, and a rare injury or ejection take him off the clock.
3. **Lineups.** The same clock seats five players at a time. Six players are never on the court together, and a player's minutes are the minutes he was in those fives. The closing five reads the last six minutes of the margin, not only the final score.
4. **Shots and points.** The trips down the floor become shots by zone, free throws, live-ball and dead-ball turnovers, and points. The points match the score from step 1.
5. **The other stats.** Steals come from live-ball turnovers. Blocks come from missed shots at the rim. Rebounds come from the misses, and an assist is tied to a made shot in a zone.
6. **The box score.** All of that is stacked into one finished line per player. Nothing is played a second time here. This step passes the shot arrays into step 5 and checks that the pieces describe one game.

```text
game script  ->  minutes  ->  five on the court  ->  shots and points
             ->  rebounds, steals, blocks  ->  one box score
```

## The code

The engine lives in `nba_game_simulator/src/`. Six folders, one per step.

```text
nba_game_simulator/src/game_enviroment/
    __init__.py      # public names
    __main__.py      # command line
    constants.py
    games.py         # logs to one row per game
    features.py      # pregame pace, rest, and lines
    pace.py          # possessions-per-48 model
    script.py        # overtime, possession gap, scoring, and the draw
    pbp_script.py    # counted trips, pace by half and margin, late-margin overtime
    train.py
    simulate.py
    slate.py         # Pinnacle lines and Rotowire fives
    rotation.py      # prior-only minutes and rotation ratings
    backtest.py      # date holdout of the shared draw

    python -m game_enviroment --backtest \
        --data "data/silver/nba/*/regular_season/player_gamelogs.parquet" \
        --out-dir backtest_out

nba_game_simulator/src/game_state/
    __init__.py      # public names
    __main__.py      # command line
    constants.py
    load.py          # player logs and play-by-play
    replay.py        # one game into lineups and minutes
    tables.py        # usable games and starters on the floor
    features.py      # pregame margin and rolling player features
    train.py
    simulate.py      # roster minutes inside a Layer 1 draw
    roster.py        # pregame player rows
    evaluate.py      # holdout against saved Layer 1 draws

    python -m game_state --pbp "..." --player-logs "..."

nba_game_simulator/src/game_rotations/
    __init__.py      # public names
    constants.py
    checks.py        # frame and identifier checks
    history.py       # training labels and live role rows
    participation.py # who plays in this script before the minute budget is split
    script.py        # attached margin, overtime, and duration
    roster.py        # who can play
    model.py         # ridge minutes and residual banks
    minutes.py       # team budget and bounded draws
    segments.py      # five-player schedules
    simulator.py     # RotationSimulator
    train.py         # fit on separate dates
    pbp.py           # historical templates, and player-minute rows for the process
    process.py       # foul, blowout, stint, and exit hazards; minutes and fives
    __main__.py      # train the historical templates from play-by-play

    python -m game_rotations \
        --pbp "..." --logs "..." --output outputs/pbp_rotation

nba_game_simulator/src/game_opportunities/
    __init__.py      # public names
    constants.py     # rate names and column aliases
    checks.py        # identifiers, minutes, availability
    model.py         # OpportunityModel
    counts.py        # pools, multinomials, conditional makes
    simulate.py      # caller-supplied team pools
    history.py       # logs and prior exposure rates
    train.py         # fit, training rows, live rows
    allocate.py      # one roster from possessions and minutes
    segments.py      # on-court intervals for both teams

    from game_opportunities import allocate_game_opportunities

nba_game_simulator/src/possession_table.py   # one row each time a team has the ball
nba_game_simulator/src/possession_rates.py   # usage, shot zones, and free-throw trips
nba_game_simulator/src/pbp_guard.py          # dates before the slate, and box-score checks

    from possession_table import possession_table
    from possession_rates import usage_rate

nba_game_simulator/src/game_events/
    __init__.py      # public names
    constants.py     # rate names, windows, and columns
    checks.py        # counts, probabilities, and identifiers
    draws.py         # beta-binomial pools and multinomial splits
    shares.py        # minute shares, joint assists, rebound pools
    simulate.py      # explicit probabilities for one game
    segments.py      # Layer 3 on-court credit
    model.py         # EventModel, save, and load
    logs.py          # official logs and a slate
    rates.py         # prior-only player and team rates
    credit.py        # shrunk assist, rebound, block, and steal rates from links
    links.py         # assist, rebound, steal, and block links in the play-by-play
    train.py         # league rates, windows, and dispersion
    allocate.py      # Layer 2 minutes, Layer 3 segments, Layer 4 shots
    evaluate.py      # one game against the box score

    from game_events import allocate_events

nba_game_simulator/src/game_boxscore/
    __init__.py      # public names
    constants.py     # sides, layer modules, and stat names
    types.py         # team inputs, models, props, and the box
    checks.py        # player identifiers
    simulate.py      # one pass through layers 1-5
    assemble.py      # stack minutes, shots, and events
    props.py         # joint props and one-row output

    from game_boxscore import assemble_box_score
```

The folder is spelled `game_enviroment`. That is the name on disk.

### 1. The game script — `game_enviroment`

This is the story both teams share.

Past games are turned into one row each. Before tip-off the model looks at how fast the teams usually play, how much rest they have had, and the point spread. From that it draws possessions, the gap between the teams, and the score. Tonight's inputs are the betting line and the expected starting five. A check on later dates asks whether the story still looks like real games the model was not trained on.

With play-by-play, the possession total is the counted trips, in place of the box estimate `FGA + 0.44·FTA − OREB + TOV`. A count is kept when the resulting pace sits between 70 and 130 possessions per 48 minutes. Anything outside that screen keeps the box estimate.

Pace depends on the half and the size of the margin. The first half uses the opening margin. The second half uses the simulated halftime margin. Both teams share that rate. A tied game stays at the pregame pace. A thin margin bucket is pulled toward the tied trips in that half, with 200 possessions of prior. The gap between the two teams' possession totals is still its own draw.

Overtime is one probability for the game, from the simulated margin with six minutes left, with two minutes left, and at the end of regulation, blended with the rate at that final regulation margin. It is shrunk toward the table rate with 40 games. A model trained without the play-by-play still uses the older guess from the pregame spread. Retrain to pick up the new tables. The saved file is model version 3 either way. The new file also carries `pace_state` and `ot_path`.

The same draw writes a margin path for the later steps: a tie at the opening tip, the halftime margin, the end of regulation, then overtime when there is any. The last minute of the path is the final margin of this step. A model without the half split holds the path flat at that final margin.

### 2. Minutes and 3. Lineups — one clock in `game_rotations`

These two steps are drawn together. A player's minutes are the time he stays on the floor, and the lineup is who shares that floor. Each player-minute is a sample. A copied whole-game rotation is one sample, so the hazards below replace it.

- **Foul trouble.** Fouls are simulated as the game goes. Six fouls is a disqualification at the buzzer, and a player ahead of that pace can be sat. His points accrue only while he is on the floor, so a foul-shortened night and a quiet scoring night land in the same play.
- **Blowouts.** Starters leave when the lead and the clock cross a threshold learned from the play-by-play. The fit also learns how often they come back.
- **Stints.** Time already spent in this stint, minutes already played in the game, and a timeout raise the chance of an ordinary substitution.
- **Injuries and ejections.** A sharp minute drop is a rare mixture on top of the usual substitution. An ejection is labeled when the feed records one. The injury rate stays a small prior, because the feed does not mark injuries as their own event.
- **Closing fives.** The last six minutes of the margin path decide the closing lineup. A game that was a blowout and then got close is a different close from a game that was close the whole way.

Only games before the slate enter the fit. The opponent's path is the same series with the sign flipped, so each team sees its own lead. A player listed as out is left off the roster and gets zero minutes.

`game_state` still holds the older minute model, and `game_rotations` can still pack a finished minute total into a historical template. The notebooks use the process above.

### 4. Shots and points — `game_opportunities`

Each trip down the floor has to become a shot, a free throw, a make, or a turnover.

The game draw still splits a team's shots by minutes. A player's share comes from how often he has shot per minute in past box scores, and from how many minutes he has in this play. When the lineup schedule is available, a shot is given only to someone who is on the court. The points that come out are lined up with the score from step 1. Lining up the score changes makes. It leaves the attempts where they were.

`possession_table.py` reads the play-by-play and writes one row every time a team has the ball. The row has the five on offense, the five on defense, the score margin, the period, the clock, and how the trip ended. A trip ends on a made shot, a defensive rebound, a turnover, a made last free throw, or the end of the period. An offensive rebound stays on the same row. Heaves are left out.

Every play-by-play feature uses games strictly before the slate, including with-or-without splits. The same calendar day is left out. `pbp_guard.py` drops a game whose play-by-play does not add up to the official box. Makes, free throws, turnovers, assists, rebounds, steals, and blocks have to match exactly. Field-goal attempts may be short by up to two heaves, because heaves are left out of the table. A missing box drops the game.

`possession_rates.py` turns the kept rows into the shares this step was built to use. Everything taken from those rows is shrunk.

- Usage is field-goal attempts divided by offensive possessions on the floor. A fast pace or a long stint no longer looks like more usage.
- Usage, including a with-or-without split, is pulled toward the average player's share of the shots. That prior is total field-goal attempts divided by five times the possessions, with 400 possessions of weight. It is the shots per player on the floor, so a full possession is five shares.
- Shots are rim (inside 4 feet), midrange, corner three, and above-the-break three. Each zone's make rate and its share of the shots are pulled toward the league rate in the same table. The prior is 80 attempts. And-one zone shares use the same prior.
- Free throws are two-shot fouls, three-shot fouls, and-ones, and bonus trips. Trip rates are pulled toward the league with 80 possessions. An and-one keeps the zone of the made shot.
- A possession is garbage when the lead is larger than 10 points plus 0.7 points for each minute left. Overtime uses only that period's clock. Garbage rows, and rows where the lineup had to be guessed, are left out of the rates.

The game draw reads those rates. `live_opportunities(..., possessions=table)` replaces the per-minute shot share with field goals per offensive possession, and the shot is drawn from the four zones. Free-throw attempts per possession come from the same rows. Playing time still decides how many of the team's trips a player is on the floor for. A player with no possessions in the table keeps his box-score rate.

For each play and each player the draw writes attempts and makes in the four zones, free throws, and turnovers split into live-ball and dead-ball. Live-ball is the share on the profile. When that share is missing, half the turnovers are treated as live-ball. That half is a prior, not a fitted rate. Team misses are kept by zone. Points are two for a rim or midrange make, three for either three, plus made free throws. After the score from step 1 moves the makes, the two-point makes are split back across rim and midrange, and the three-point makes across the corner and above the break, so the zone makes still add up to the field goals.

### 5. The other stats — `game_events`

The box score is more than points.

When step 4 wrote the zone makes, the live-ball turnovers, and the misses by zone, this step draws from those arrays.

- Steals are taken from the live-ball turnovers and credited to defenders who were on the floor.
- Blocks are taken from missed rim shots, inside 4 feet. A make at the rim is not a block chance, and a miss from farther out is not one either, once the zones are known.
- Offensive rebounds are drawn one zone at a time, because a rim miss and a three-point miss are kept at different rates. Defensive rebounds are the misses that stayed off the offensive glass. The two add up to the misses.
- An assist is tied to a made shot in a zone. The make is assisted with one probability, then a teammate on the floor is chosen as the passer. The passer cannot be the shooter.

Those rates are shrunk. The chance a make was assisted, for that shooter and zone, is pulled toward the zone's usual rate with 40 makes. Offensive-rebound, defensive-rebound, block, and steal chances are pulled toward the team rate with 40 chances. Passer strength is pulled toward a small share with 4 assists.

A credit table built from the play-by-play links fills the shooter-and-zone assist rates and the rebound rate in each zone. Without that table, every zone uses the team's assist rate and the team's offensive-rebound rate. The notebooks pass the shot arrays from step 4 into this step. A finished game can be compared with the real box score.

### 6. The box score — `game_boxscore`

This is the front door. One call runs steps 1 through 5, then stacks minutes, shots, and the other stats into a single box score per play.

It forwards the zone attempts, the zone makes, the live-ball and dead-ball turnovers, and the misses by zone into step 5. When those arrays are present, the assists and offensive rebounds sketched in step 4 are dropped, and step 5's draws are the ones on the box. It does not play the game again. It checks that the rows still describe one game: the same play index, minutes that fit the clock, and points that match the script. From those rows it can answer a prop, including a prop with more than one leg, by counting the plays where every leg won.

## On your computer

Python 3.12 or newer. From the repository root:

```bash
pip install -r requirements.txt
pytest
```

The three command-line entry points are listed in the tree above. Run them with `nba_game_simulator/src` on the Python path, because that is where the six folders live. This is a local research project. There is no website or server to deploy.
