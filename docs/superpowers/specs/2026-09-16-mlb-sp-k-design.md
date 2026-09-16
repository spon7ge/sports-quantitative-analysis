# MLB Starting-Pitcher Strikeout MVP

Hobby-level, leakage-safe, reproducible strikeout distributions for scheduled MLB starters. Lives beside the NBA stack; it does not share learners, features, or artifacts.

## Goal

For each scheduled starter at a prediction cutoff, emit E[K], P(K=k), 80% intervals, and over/under probabilities. Compare those probabilities to same-time de-vigged quotes when supplied. Do not claim profitability.

## Placement

- Package: `src/mlb/`
- Config: `config/mlb.yaml`
- Data: `data/mlb/` (gitignored except fixtures)
- Tests: `tests/mlb/`
- Docs: `docs/mlb/`
- CLI: `python -m src.mlb`

Follow existing repo conventions: Python 3.12, pandas, pyarrow, pytest/unittest, argparse, `src.*` imports, no Airflow.

## Architecture

Pitch-level Statcast and Stats API snapshots land in append-only `raw_snapshots`. Parsers emit `pitch_events`, `plate_appearances`, `pitcher_starts`, and versioned `game_versions`. Features are one pitcher-game-cutoff row using only events and ingestions strictly before the cutoff. A regularized workload NB produces rolling-origin OOF expected BF / uncertainty / early-exit probability. A penalized strikeout NB consumes those OOF features (never in-sample workload fits) and emits a coherent PMF. Calibration, backtest, and market comparison are chronological.

```
raw_snapshots → parsed tables → pregame_snapshots
        → feature_rows (cutoff-strict)
        → OOF workload → strikeout NB PMF
        → props / optional quote compare / daily report
```

## Leakage rules

- Event time and ingestion time must both precede the cutoff.
- Never backfill eventual starters, final lineups, realized weather, or later quotes.
- Workload predictions used as strikeout features are out-of-fold.
- Sportsbook lines are not baseball-model features.
- Random splits are forbidden.

## Out of MVP scope

Joint BF hazard simulation, batter-specific Poisson-binomial, boosting ensembles, umpire/weather/catcher effects, market-residual stacks.
