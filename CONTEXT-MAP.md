# Context Map

## Contexts

- [Player props](./docs/context.md): full-game points pricing from minutes and points-per-minute quantiles
- [Game environment](./nba_game_simulator/CONTEXT.md): the shared game draw a player prop sits inside

## Relationships

- **Game environment → minutes → shots → shared events → box score**: Layer 2 reads one environment's script (`team_margin`, `n_ot`, `game_minutes`) and does not draw another game. Layer 4 reads that same draw's possession counts and the minutes. Layer 5 reads the shots and turnovers. Layer 6 reads that row as one box score. None of them draws another game.
- **Player props and the game environment**: separate stacks today. The quantile points model does not consume an environment row.
