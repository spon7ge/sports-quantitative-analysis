# Layer 4: Player Possession / Opportunity Allocation Design

## Objective

Build a simple, functional player-level offensive opportunity allocation
model that integrates into the existing NBA Monte Carlo stack under
`nba_game_simulator/`. Layer 4 distributes a team's supplied field-goal
attempts across players, simulates shooting and selected secondary
statistics, and returns player box-score arrays suitable for 10,000-game
Monte Carlo summaries.

This is a statistical allocation layer, not a possession-by-possession
event engine.

## Boundaries

**In scope**

- Opportunity-share adjustment and normalization.
- Multinomial allocation of supplied team FGA to players.
- Binomial 2PA/3PA split and made-shot outcomes.
- Player-generated FTA/FTM from foul-drawing rates.
- Team-pool allocation for AST, OREB, and TOV (reconciled within Layer 4's
  modeled volumes).
- Config-driven player rates; optional PBP rate builder; Boston demo.
- Unit tests for contracts, reconciliation, and reproducibility.

**Out of scope**

- Generating team FGA inside the core simulator (demo/helper only).
- Dirichlet-multinomial, hierarchical Bayes, or complex correlation.
- Possession-by-possession play simulation.
- Full box-score accounting identity (e.g. possessions = FGA + TOV − OREB
  + …).
- Defensive stats, DRB, steals, blocks, or market pricing.
- Tight coupling to Layer 1–3 call sites (inputs are plain dicts/arrays so
  Layer 1 can supply FGA later without changing Layer 4).

## Architecture

Approach: split core vs helpers.

```
nba_game_simulator/
  src/
    __init__.py
    opportunity_shares.py
    possession_allocation.py   # single public simulate_game()
    rate_builders.py           # optional PBP → rates
  demos/
    bos_layer4_demo.py         # NegBin team FGA + Boston illustrative rates
tests/game_simulator/
  test_opportunity_shares.py
  test_possession_allocation.py
```

Naming note: the module is `possession_allocation.py` for continuity with
the Layer 4 docs, but the implementation allocates aggregate opportunities,
not discrete possessions.

### Statistical boundary

Two kinds of statistics:

| Kind | Stats | Rule |
| --- | --- | --- |
| Allocated team pools | FGA (required input), AST, OREB, TOV | Player totals sum to the team pool |
| Player-generated outcomes | FTA, FTM | Derived per player from FGA × rates; not forced to a team sum |

Team aggregates for allocated stats are derived as row sums of player
arrays, never independently simulated.

TOV: the team turnover pool must be supplied by the caller/demo, or
generated provisionally inside Layer 4 from
`possessions × Σ(p_i · tov_rate_i)` and documented as approximate.
Layer 4 does not claim the correct team turnover volume from first
principles.

AST and OREB team pools are outcome-dependent after shooting:
`team_fgm` and `team_missed_fga = team_fga − team_fgm`.

## Opportunity shares

Baseline weights `w_i` are **baseline share before the minutes
adjustment** (not usage conditional on minutes). Document this when
estimating from historical FGA%.

\[
\tilde{w}_i = w_i \cdot a_i \cdot m_i^{\gamma} \cdot r_i \cdot l_i
\qquad
p_i = \tilde{w}_i / \sum_j \tilde{w}_j
\]

\[
m_i = \frac{\text{projected minutes}_i}{\text{reference minutes}_i}
\]

| Component | Symbol | Default |
| --- | --- | --- |
| Baseline weight | \(w_i\) | required |
| Availability | \(a_i\) | 0 or 1 |
| Minutes exponent | \(\gamma\) | 1.0 |
| Role multiplier | \(r_i\) | 1.0 |
| Lineup multiplier | \(l_i\) | 1.0 |

If reference minutes are 0, treat the minute ratio as 0. If the adjusted
total is ≤ 0, raise `ValueError`. Return and retain adjusted shares for
diagnostics.

## Simulation sequence

Documented as a statistical sequence, not a causal possession simulator:

1. Validate inputs.
2. Compute opportunity shares \(p\).
3. Broadcast `team_fga` to shape `(n_sims,)`.
4. Allocate `FGA ~ Multinomial(team_fga, p)` so \(\sum_i FGA_i = FGA_{\text{team}}\).
5. `3PA ~ Binomial(FGA, p_{3PA})`, `2PA = FGA − 3PA`.
6. `3PM ~ Binomial(3PA, p_{3P})`, `2PM ~ Binomial(2PA, p_{2P})`,
   `FGM = 2PM + 3PM`.
7. `FTA ~ NB(FGA · fta_per_fga, α_{FTA})`, `FTM ~ Binomial(FTA, FT%)`.
8. Derive team FGM and missed FGA; draw AST/OREB team pools; multinomial
   allocate with propensity-weighted shares.
9. Allocate TOV from supplied or provisional team pool.
10. `PTS = 2·2PM + 3·3PM + FTM`.
11. Return arrays; summaries computed outside core from raw draws.

### Secondary volume models (v1)

| Stat | Generation | Exposure / pool |
| --- | --- | --- |
| FTA | NegBin (Poisson as baseline via dispersion) | \(\mu_i = FGA_i · fta\_per\_fga_i\) |
| FTM | Binomial | \(FTA_i\), \(FT\%_i\) |
| TOV | Multinomial over team TOV pool | Pool from input or provisional NB on poss × rates |
| AST | Multinomial over team AST pool | \(\mu_{\text{team}} = TeamFGM · assist\_rate\); cap pool at `TeamFGM` so \(\sum AST_i ≤ TeamFGM\). Allocation probabilities are renormalized `assist_propensity_i` (not raw opportunity shares). |
| OREB | Multinomial over team OREB pool | \(\mu_{\text{team}} = TeamMissedFGA · oreb\_rate\). Allocation probabilities are renormalized `oreb_propensity_i`. |
| TOV | Multinomial over team TOV pool | Allocation probabilities are renormalized `p_i · tov_rate_i`. |

NegBin dispersion parameters are configurable. Poisson is the special-case
baseline. Remaining box-score identities are approximate and labeled as
such until a possession engine exists.

## API

Single public entry point:

```python
simulate_game(
    team_inputs: dict[str, TeamInputs],
    player_inputs: dict[str, PlayerInputs],
    n_sims: int,
    rng: np.random.Generator,
) -> SimulationResult
```

### TeamInputs (per team)

- `possessions`: scalar or `(n_sims,)` — exposure for provisional TOV.
- `team_fga`: **required** scalar or `(n_sims,)` nonnegative integers.
- Optional `team_tov`: scalar or `(n_sims,)` nonnegative integers. If
  omitted, Layer 4 generates a provisional pool
  `NB(possessions × Σ(p_i · tov_rate_i), α_TOV)` and labels it approximate
  in the result metadata.

### PlayerInputs (per team, aligned arrays)

- `player_ids`
- `base_weights`, `available`, `projected_minutes`, `reference_minutes`
- Optional: `role_factors`, `lineup_factors`, `minutes_exponent`
- Rates: `p_3pa`, `p_3p`, `p_2p`, `fta_per_fga`, `ft_pct`, `tov_rate`,
  `assist_propensity`, `oreb_propensity`
- Optional team-level rates used for pool means: `assist_rate`,
  `oreb_rate`, and NegBin dispersion dict

### Validation (fail loudly; no silent correction)

- `team_fga` nonnegative integers; scalar or length `n_sims`.
- All teams share the same `n_sims`.
- At least one available player with positive adjusted weight.
- Opportunity shares sum to 1 within numerical tolerance after
  normalization.
- After allocation: each sim's player FGA sums exactly to that sim's
  `team_fga`.

Exception: AST pool is capped with `min(NB_draw, team_fgm)` — the only
documented soft constraint.

### SimulationResult

Per team, consistent player order via `player_ids`:

| Field | Shape |
| --- | --- |
| FGA, FGM, 2PA, 2PM, 3PA, 3PM, FTA, FTM, PTS, AST, TOV, OREB | `(n_sims, n_players)` |
| opportunity_shares | `(n_players,)` or `(n_sims, n_players)` if inputs vary by sim |
| player_ids | `(n_players,)` |

Team totals = sums over players for allocated stats. Reporting and
plotting stay out of `src/`.

## Rates and data

**Hybrid rates contract**

- Core simulator accepts a player-rate config only.
- `rate_builders.py` optionally builds that config from
  `data/PlayByPlay/*.parquet` (shot attempts/makes, turnovers, assists,
  offensive rebounds by player). Prefer PBP over silver for attempt and
  outcome rates; silver is lineup/roster oriented.
- Boston demo ships illustrative hardcoded rates and does not require
  parquet to run.

## Boston demo

`nba_game_simulator/demos/bos_layer4_demo.py`:

- BOS possessions = 101.
- Players and baseline weights: Tatum 31%, Brown 27%, White 19%,
  Porziņģis 15%, Other 8%.
- Demo sets projected ≈ reference minutes so \(m_i ≈ 1\) unless a
  sensitivity case changes availability.
- Generate `team_fga` **outside** core:

\[
\mu = possessions × fga\_per\_poss
\qquad
FGA \sim \mathrm{NegBin}(\mu, \alpha)
\]

  Poisson available as a baseline helper option.
- Run 10,000 sims with a seeded `np.random.Generator`.
- Print per-player mean, sd, p25/p50/p75 for key stats from raw arrays.
- Assert FGA reconciliation and AST ≤ FGM in the demo harness.

## Testing

Tests under `tests/game_simulator/`, written first:

- Shares normalize; unavailable players get 0; minutes ratio moves shares
  in the expected direction; proj = ref ⇒ shares ∝ \(w·a·r·l\).
- All unavailable ⇒ `ValueError`.
- `team_fga` scalar broadcast / vector length / reject non-integers and
  negatives.
- Multinomial FGA row sums equal `team_fga`.
- `2PA + 3PA = FGA`; makes ≤ attempts; FTM ≤ FTA.
- AST/OREB/TOV row sums equal their team pools; AST ≤ team FGM.
- Same seed ⇒ identical arrays.
- Summary helpers (if any, outside core) match `np.mean` /
  `np.std` / `np.percentile` on raw draws.

## Explicit assumptions

1. Layer 4 never decides how many team FGA occur.
2. `base_weights` are pre-minutes baseline shares; multiplying by
   \(m_i^\gamma\) is intentional and must match how weights are estimated.
3. FTA is not `Binomial(FGA, rate)`; it is NegBin volume with mean
   proportional to FGA.
4. AST/OREB/TOV are reconciled only within Layer 4's modeled pools;
   they are not full possession-consistent.
5. No cross-player shot-making correlation beyond the multinomial FGA
   allocation.
6. "Other players" may be a single residual bucket in the Boston demo.

## Success criteria

- Modular NumPy implementation with one public `simulate_game`.
- Hypothetical Boston 10k-sim example producing player means, sds, and
  25/50/75 percentiles.
- Basic validation tests green.
- Documented math and assumptions match the code.
- \(\sum_i FGA_i = FGA_{\text{team}}\) for every simulation.
