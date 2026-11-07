# Game-Level NBA Monte Carlo Simulation Engine

This project plays one NBA game many times. Each play is a full box score: the score, who was on the floor, and every player's points, rebounds, assists, and the rest.

"Monte Carlo" means we repeat the game with a little randomness, then count. If a player scores 25 or more in 6,200 of 10,000 plays, this model says that happens about 62% of the time. That 6,200 is an example of counting, not a result from a real slate.

Every play keeps one shared story. Play number 500 is the same imaginary game in every table. A fast game, a close score, and a player's big night stay tied together. Asking two questions about one game — a player over 25 points and a teammate over 8 rebounds — counts only the plays where both happened.

## The six steps

Each play walks through the same six steps, in order. A later step fills in detail. It does not throw away the score or the clock from an earlier step.

1. **The game script.** How fast the teams play, how many trips down the floor, whether it goes to overtime, and the final score.
2. **Minutes.** How long each player is on the floor in that script. A blowout and a tight game hand out minutes differently.
3. **Lineups.** Those minutes are seated five players at a time. Six players are never on the court together, and a player's minutes still add up.
4. **Shots and points.** The trips down the floor become shots, free throws, turnovers, and points. The points match the score from step 1.
5. **The other stats.** Missed shots become rebounds. Turnovers and misses create chances for steals and blocks. An assist goes to a teammate who was actually on the court.
6. **The box score.** All of that is stacked into one finished line per player. Nothing is played a second time here. This step only checks that the pieces describe one game.

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
    pbp.py           # stint minutes and historical templates for Layer 2 minutes
    __main__.py      # train those templates from play-by-play

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

Past games are turned into one row each. Before tip-off the model looks at how fast the teams usually play, how much rest they have had, and the point spread. From that it draws possessions, the gap between the teams, the score, and a simple overtime guess. Overtime here is decided up front. It is not played possession by possession until the score is tied.

Tonight's inputs are the betting line and the expected starting five. A check on later dates asks whether the story still looks like real games the model was not trained on.

### 2. Minutes — `game_state`

This step answers one question: given that story, how many minutes does each player get?

It reads official player logs and the play-by-play, replays who was on the floor, and learns minutes from games like the one just drawn. The score, the margin, and overtime stay as step 1 wrote them.

### 3. Lineups — `game_rotations`

Minutes are not enough. Basketball is five players at a time.

This folder decides who is available, splits the team's minute budget, and builds a schedule of five-player groups. It can draw the minute totals itself. When step 2 already chose them, it keeps those totals and only changes who shares the floor. Real games are used as templates, so the substitutions follow patterns that have happened before. Training uses different dates from the games being checked.

### 4. Shots and points — `game_opportunities`

Each trip down the floor has to become a shot, a free throw, a make, or a turnover.

Players get a share of those chances from how often they have used them in the past, weighted by the minutes they have in this play. When the lineup schedule is available, a shot is given only to someone who is on the court. The points that come out are lined up with the score from step 1.

### 5. The other stats — `game_events`

The box score is more than points.

Misses are split into offensive and defensive rebounds. Turnovers and missed shots create steals and blocks. Assists are shared by teammates. If step 4 already assigned assists and offensive rebounds, this step keeps them. With a lineup schedule, credit goes only to players who were on the floor when the event happened. Without one, credit follows minutes.

A finished game can be compared with the real box score.

### 6. The box score — `game_boxscore`

This is the front door. One call runs steps 1 through 5, then stacks minutes, shots, and the other stats into a single box score per play.

It does not play the game again. It checks that the rows still describe one game: the same play index, minutes that fit the clock, and points that match the script. From those rows it can answer a prop, including a prop with more than one leg, by counting the plays where every leg won.

## On your computer

Python 3.12 or newer. From the repository root:

```bash
pip install -r requirements.txt
pytest
```

The three command-line entry points are listed in the tree above. Run them with `nba_game_simulator/src` on the Python path, because that is where the six folders live. This is a local research project. There is no website or server to deploy.
