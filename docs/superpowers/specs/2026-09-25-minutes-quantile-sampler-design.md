# Minutes quantile sampler

Date: 2026-09-25
Approved: inverse-transform sampler for the quantile minutes model in `notebooks/nba/minutes/min_nba_model.ipynb`. Minutes only. The saved trees stay frozen. 2025-26 stays out of the tail tables.

## Goal

Turn one row's 11 predicted minute quantiles into a minute quantile function `Q`. The middle of `Q` is that row's own knots. The outer 5% on each side uses the shape of out-of-fold misses, separately by group. The notebook scores `Q` directly: tail-bin shares and CRPS. A minutes line is priced by inverting `Q`, not by counting draws. `Q` is piecewise linear, so `P(M < L) = inf{u : Q(u) >= L}`, which is exact on each linear piece. It is `0` below the bottom of `Q` and `1` above the top. Draws exist so minutes can later be combined with other random quantities. They are not a price.

This does not draw points and does not retune the quantile model. The old mean-plus-residual simulator is gone; this sampler is the minutes distribution.

## Pieces

| Piece | Path | Role |
|---|---|---|
| Sampler | `models/shared/minutes_sampler.py` | Grid prep, quantile map, tail builder, draw loop, sidecar load |
| Tests | `tests/models/test_minutes_sampler.py` | Synthetic grids and misses. No XGBoost fit |
| Sidecar | `models/saved_models/min_nba_tails_2026-04-12.joblib` | Pinned tail arrays, out-of-fold frame, rules |
| Notebook | `notebooks/nba/minutes/min_nba_model.ipynb` | Build the sidecar, fold-split tail report, one 2025-26 score |

The model joblib `models/saved_models/min_nba_model_2026-04-12.joblib` is not rewritten.

## Quantile levels

The code and the sidecar both store this list, in this order:

`0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95`

A grid built with `np.linspace(0.05, 0.95, 11)` is wrong. The loader raises if the sidecar's list differs from the code's.

## Grid prep

One function, used by the map and by the table builder. Out-of-fold rows go through it before `r` and `e` are computed.

1. Raise if any knot is NaN or infinite.
2. Monotonize once, by sorting the row.
3. Apply `max(q, 1e-3)` to every knot.

There is no second sort. `max` cannot reorder a sorted row. A knot already above `1e-3` is unchanged. A knot between 0 and `1e-3` is lifted by at most 0.06 seconds.

## Quantile map

One uniform `u` in `[0, 1]` becomes one minute total. `u` outside that interval raises.

Prepare the row's raw knots first. Call the prepared 0.05 and 0.95 knots `q05` and `q95`.

**Middle.** For `u` in `[0.05, 0.95]`, linearly interpolate the prepared knots at the 11 levels above. `Q(0.05) = q05` and `Q(0.95) = q95`. This branch does not read the tail tables.

**Lower tail.** For `u < 0.05`,

```
Q(u) = q05 * R^{-1}(u / 0.05)
```

`R` is the empirical distribution of `r = y / q05` for out-of-fold rows in that lower-tail group with `y < q05`, after grid prep, plus a pin at `r = 1`.

**Upper tail.** For `u > 0.95`,

```
Q(u) = q95 + E^{-1}((u - 0.95) / 0.05)
```

`E` is the empirical distribution of `e = y - q95` for out-of-fold rows in that upper-tail group with `y > q95`, after grid prep, plus a pin at `e = 0`.

The inverse of a sorted table, pin included, is

```python
np.interp(v, np.linspace(0, 1, len(table)), table)
```

Probability 0 returns the first entry. Probability 1 returns the last entry. After the sort, the lower pin `1` is the last entry, so probability 1 returns it. The upper pin `0` is the first entry, so probability 0 returns it and probability 1 returns the largest excess.

`quantile_minutes(u, grids, lower_groups, upper_groups, tables)` is the public map. `u` has shape `(rows, draws)`. The result has the same shape, in minutes, after the clip. A later teammate-dependence step will replace or shift `u` before this call. The map does not draw random numbers. An id with no array raises.

The finished draw is clipped to `[0, maximum_minutes("nba")]`, which is 63. The grid is not clipped. The clip at 0 does not bind: every prepared knot is at least `1e-3`, and every stored `r` is positive.

## Tail tables

Walk-forward validation rows from the pre-holdout pool only. The four validation windows do not overlap. They cover about 40% of train dates. The first half of train dates and the last 10% have no out-of-fold row. Holdout rows are never in the pool.

`run_walk_forward` keeps, on every validation fold, the raw 11 knots, fold id, game date, minutes, and `starting`. It still returns the last fold's predictions as it does today. Prepared quantiles are not stored.

### Early stopping

`run_walk_forward` takes `early_stop`, default `"validation"`.

| Setting | Early-stopping rows | Predicted rows |
|---|---|---|
| `"validation"` (default) | That fold's validation rows | Those same validation rows |
| `"train_tail"` | Last 10% of that fold's training dates | That fold's validation dates |

The default keeps today's walk-forward pinball and coverage reproducible. Those predictions are slightly optimistic, because each fold early-stops on the rows it then scores. The tail build uses `"train_tail"` only.

`fit_quantile_models` takes the early-stopping frame separately from the frame it predicts. Under `"train_tail"`, a whole `game_date` is on one side. The stopping slice and the fit slice each contain at least one date. Fewer than two training dates raises inside `fit_quantile_models` before any `XGBRegressor` is constructed. `build_tail_tables` raises unless the out-of-fold frame's stopping setting is `"train_tail"` at 10% of training dates.

### Who enters a table

The builder takes the list of fold ids to use. It raises if any requested id is absent. A frame that only contains fold 4 fails a request for folds 1–4 on those missing ids, before any miss rate is computed. Folds 1–2 alone are a valid in-memory build. They are not a valid sidecar.

For each selected row, after grid prep:

- Assert `y > 0`.
- Assert `game_date` is strictly before 2025-10-21.
- If `y < q05`, append `r = y / q05` to `(lower, group)`. Each ratio is strictly inside `(0, 1)`.
- If `y > q95`, append `e = y - q95` to `(upper, group)`. Each excess is strictly positive.

Before the pins are added, each requested fold must have a lower-tail miss rate and an upper-tail miss rate inside 4–6%. The rate is the share of that fold's rows, pooled across groups. A rate outside the band raises. That check catches predictions joined to the wrong minutes.

The builder allocates an array for every group the rule defines before it scans rows. Under `starting` those groups are `0` and `1`, on both tails. Then pin `1` onto each lower array and `0` onto each upper array, and sort. A defined group with no real miss, only the pin, raises at build time. The saved lower array ends at `1`. The saved upper array starts at `0`.

**Historical rows.** Both tails use group `starting`. On a played game, `starting` is `1` when `start_position` is G, F, or C, and `0` when `start_position` is blank. A blank in the box score means the player came off the bench. The builder sees that 0/1 column.

**Draw time.** The loader does not fill a missing `starting` with 0. Before lineups are confirmed, a blank means the lineup is unknown, and the loader raises. Only an explicit `0` or `1` selects a table. An unknown grouping rule raises in the builder and in the loader. A later upper tail can switch to q50 tiers by replacing its arrays and its grouping rule. The map stays the same, because the two tails already take different ids.

## Sidecar

`models/saved_models/min_nba_tails_2026-04-12.joblib` holds:

- The four pinned, sorted arrays, keyed by `(tail, group)`.
- The out-of-fold frame: raw 11 knots, fold id, game date, minutes, `starting`.
- Rules the map does not know:
  - floor `1e-3`
  - the 11 quantile levels
  - each fold's train-date range and validation-date range
  - stopping setting `"train_tail"` at 10% of training dates
  - holdout boundary 2025-10-21
  - each tail's grouping rule: `starting` for both tails today; tier edges when a tail switches
  - the fold ids the arrays were built from

The loader raises if the stored floor or the stored quantile levels differ from the code. It raises unless those fold ids are exactly 1, 2, 3, and 4. An unknown grouping rule raises. The model joblib stays untouched. A folds 1–2 build can exist in memory for the diagnostic; saving it and loading it fails this check.

## Draw loop

```python
sample_minutes(
    grids, lower_groups, upper_groups, player_ids, game_ids, tables,
    *, seed=42, draws=10_000,
)
```

`seed` and `draws` are explicit keyword arguments, defaulting to 42 and 10,000. The return has shape `(rows, draws)`. `sample_minutes` builds `u` and calls `quantile_minutes`. It does not implement `Q`.

`player_id` and `game_id` are required. A missing id raises. There is no fallback to row position. Tests use made-up ids.

Ids are canonicalized before they are hashed. Strip whitespace first. An integer, an integer-valued float, or a stripped string of digits then becomes the base-10 integer with no leading zeros (`0` stays `"0"`). `" 0021900001 "`, `"0021900001"`, `21900001`, and `"21900001"` are the same id. Any other stripped value is kept as that string. A missing id still raises before this step.

Draw `i` for a player-game is the `i`-th uniform from a NumPy generator seeded by SHA-256 of `seed | minutes | player_id | game_id`, using the first 8 bytes. The stream label `minutes` is part of the minutes sampler only. `JointPointsSimulator._rng` is unchanged, so a later points model can pair these draws with its own noise without sharing uniforms. The same canonical player-game returns the same minutes in any row order. Uniforms come from `Generator.random`, so they lie in `[0, 1)`. The map still accepts `1`.

The loader turns sidecar grouping rules into the two id arrays. The draw function only sees integer ids. A missing `starting` raises in the loader and is not coded as bench.

## Tests

All of these use synthetic grids and synthetic misses.

**Map**

- The pin test builds its tables with `build_tail_tables`. Both groups have a real miss on both tails, and the fold's pooled miss rates stay inside 4–6%. In the tested group, the ratio `0.5` is the only lower miss and the excess `10` is the only upper miss. That group's lower array ends at `1` and its upper array starts at `0`. For `ε = 1e-6`, the relative gap `|Q(0.05 - ε) - q05| / q05` is under `1e-3`. The upper gap `|Q(0.95 + ε) - q95|` is at most `1e-3` minutes. Dropping the pin from that group's built arrays drops the lower value to `0.5 * q05` and adds 10 minutes on the upper side. Exact `u = 0.05` and `u = 0.95` are the middle branch and are not this test.
- `Q(α)` equals the prepared knot at every stored level `α`, including the eight levels that `linspace(0.05, 0.95, 11)` gets wrong.
- `Q(u)` is non-decreasing on a grid of `u` from 0 to 1.
- A lower-tail `Q(u)` scales with that row's `q05`. An upper-tail `Q(u)` moves one-for-one with `q95`.
- `prepare_quantile_grid([-0.1, 0.0005, ...])` stays in level order and floors both knots to `1e-3`. A NaN knot raises.
- Group `0` and group `1` read different arrays. The same row can carry a different id on each tail.

**Draws**

- The same seed and the same player-game reproduce the same draw vector.
- Shuffling rows leaves each player-game's draws unchanged.
- Two different player-games, at a fixed seed and 10,000 draws, have Pearson correlation below 0.05 in absolute value, and the two vectors are not equal.
- `" 0021900001 "` and `21900001` produce the same draw vector.
- For the same seed and the same canonical ids, the minutes uniforms differ from the uniforms of the same SHA-256 construction with the `minutes` label removed (`seed | player_id | game_id`). A comparison with `JointPointsSimulator._rng` is not this test: that generator also hashes a bundle id, so its uniforms differ even when the minutes label was never used.
- Two rows whose prepared `q50` values differ by at least 15 minutes: each row's median draw is closer to its own `q50` than to the other row's `q50`.
- A raw draw past 63 comes back at 63.

**One test per hard failure**

- Non-finite knot.
- `y <= 0`.
- `game_date` on or after 2025-10-21.
- Stopping setting other than `"train_tail"` at 10%.
- A requested fold id absent from the frame.
- A requested fold whose lower-tail or upper-tail miss rate is outside 4–6%.
- A group with only its pin, including a group the rule defines that has no rows in the build.
- `starting` not in `{0, 1}` under the `starting` rule, including a missing value at draw time.
- An unknown grouping rule.
- Fewer than two training dates on the `train_tail` path. This raises before any `XGBRegressor` is constructed.
- Unknown group id at draw time.
- Knot count other than 11.
- `u` outside `[0, 1]`.
- Sidecar floor differs from `1e-3`.
- Sidecar quantile levels differ from the list above.
- Sidecar fold ids are not exactly 1, 2, 3, and 4.
- Missing `player_id` or `game_id`.

## Notebook check

Tail tables for the diagnostic are built from folds 1–2 only and scored on folds 3–4. Those tables stay in memory. They are not written to the sidecar path. Each tail is five bins of width 0.01.

Lower tail, among rows with `y < q05`:

| Bin | Condition |
|---|---|
| 1 | `y < Q(0.01)` |
| 2 | `Q(0.01) <= y < Q(0.02)` |
| 3 | `Q(0.02) <= y < Q(0.03)` |
| 4 | `Q(0.03) <= y < Q(0.04)` |
| 5 | `Q(0.04) <= y < Q(0.05)` |

Upper tail, among rows with `y > q95`:

| Bin | Condition |
|---|---|
| 1 | `q95 < y <= Q(0.96)` |
| 2 | `Q(0.96) < y <= Q(0.97)` |
| 3 | `Q(0.97) < y <= Q(0.98)` |
| 4 | `Q(0.98) < y <= Q(0.99)` |
| 5 | `y > Q(0.99)` |

For each bin the notebook reports the share of all rows in the slice, and the share of that tail's misses. The shape under test is the second share. Each bin should hold about 20% of its tail's misses. The raw share mixes in calibration at `q05` and `q95`; anything from 4–6% already passed the build.

The report is repeated on two splits, not on their cross:

- `starting` 0 and 1
- predicted-q50 tiers already in the notebook: `<15`, `15-24`, `24-31`, `31+`

Each share gets a 95% interval from 2,000 resamples clustered by `game_date`, using the same date-cluster resampling as `clustered_mae_bootstrap` in the minutes notebook.

After that report, score 2025-26 once. Knots come from `preds_ho`, the holdout predictions straight from the trees in `evaluate_holdout`. They do not come from `preds_ho_live`. Tables come from the sidecar built on folds 1–4. The record has two parts: the same bin report, with the same splits and intervals, and a CRPS for the whole of `Q`.

CRPS uses no draws. On `u = 0.001, 0.002, ..., 0.999`,

```
CRPS_i = 2 * mean_k pinball(y_i, Q_i(u_k), u_k)
```

`pinball` is the same residual formula as `pinball_loss` in `models.shared.metrics`, applied to one row. The reported score is the mean of `CRPS_i`. It is in minutes. With this pinball, WIS is `2 × mean pinball` over the 11 levels, which is the CRPS formula at 11 points instead of 999.

Before trusting the 999-point score, run the same function with `u` set to the 11 quantile levels. It should return 3.092. The only gap comes from the sort and the floor on rows whose knots changed. If the gap is larger than those rows can explain, the factor or the pinball sign is wrong. WIS scores the 11 knots. The 999-point CRPS scores `Q`, including the interpolated middle and the tails.

The record is not a reason to edit the tables, the grouping rules, the floor, or the stopping setting.

Building the sidecar requires a fresh `run_walk_forward(..., early_stop="train_tail")`. The predictions already stored in the notebook are the last fold only, and they early-stopped on the validation rows.
