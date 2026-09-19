# Assists feature builder (current12)

Date: 2026-09-19  
Revised: 2026-09-19 (review 2)  
Approved: leakage-safe 12-feature assists builder. Not a mean model. Not coupling. Not pricing.

## Goal

Add `src/features/assists/` so pregame assist features can be built the same causal way as points: Game N outcomes never enter a feature, scheduled-opponent history rather than a player’s previous `opp_*` values, and `predicted_minutes_oof` left empty for chronological stacking.

This slice ends when `add_assists_features` emits `CURRENT_ASSISTS_FEATURES` on synthetic frames and the leakage tests below pass.

## Parent architecture (later slices)

Keep the points workflow’s causal and operational structure. Diverge statistically:

- Reuse OOF minutes, chronological folds, cross-fitting, artifact checks, half-point pricing, and the preholdout promotion process.
- Treat `reg:squarederror` plus residual bootstrap as a baseline only.
- Prefer a direct discrete count distribution, starting with Poisson-style boosting and testing overdispersion.
- Do not copy \(g, \beta, \epsilon\) initially. Add minutes coupling only if cross-fitted minute shocks improve count NLL and discrete CRPS.
- Use randomized discrete PIT, not ordinary continuous PIT.
- `2025-26` stays sealed through selection.

Those rules constrain later specs. They do not expand this slice.

## Universe

- League: NBA regular-season silver player-game rows.
- Modeling target later: official box `ast` on appearances with canonical minutes \(> 0\).
- This slice uses synthetic frames only. It does not read silver, holdout, or artifacts.

### Output panel vs appearance lookup

These are different sets. Mixing them is how `days_rest` contradicted itself.

| Set | Rule |
|---|---|
| **Output panel** | Every input row. The current/candidate row always stays in the result, including when canonical minutes are NaN (pregame blanking). |
| **Appearance** | A **prior** row used as history: canonical minutes finite and \(> 0\). Last game’s minutes are pregame-known, so filtering the previous row on appearance is required. |

The current row is never required to be an appearance in order to receive features. Only the lookup of previous dates, previous ast, previous minutes, and previous start flags filters on appearance (and, for the rate, on a paired mask below).

## Files

| Path | Owns |
|---|---|
| `src/features/assists/columns.py` | `CURRENT_ASSISTS_FEATURES` (12, this order) and `ASSISTS_FEATURE_CHALLENGERS` (names only) |
| `src/features/assists/player.py` | Canonical `ast` / minutes, `start_rate_10`, trailing ast means and rate, `days_rest` |
| `src/features/assists/team.py` | Null-aware team-game table; own-team and scheduled-opponent snapshots |
| `src/features/assists/build.py` | `add_assists_features` |
| `src/features/assists/pregame.py` | `as_of` history cut, outcome blanking, `build_pregame_assists_features` |
| `src/features/assists/__init__.py` | Public exports |
| `tests/features/test_assists_features.py` | Shift-before-lookup, aliases, paired rate, team reduction, opponent asof, contract, `days_rest`, index stability |
| `tests/features/test_pregame_assists.py` | `game_date < as_of`, candidate isolation, one-sided slate |

Reuse from minutes, do not call `add_minutes_features`:

- `src/features/minutes/rolling.py` (`numeric_column`, `ratio`). Do not call `prior_sum` / `prior_roll` on two series independently for the rate — those skip NaNs per series. Player windows are last-K **qualifying prior rows**, not last-K panel rows with independent `skipna`.
- `is_home` rule: `matchup` contains `\bvs\.` (copy the minutes environment test, do not import `add_minutes_features`)

Do not modify `src/features/minutes/player.py`. That builder’s `assists_per_min_10` prefers tracking `assists` then box `ast`. The assists contract uses official `ast`. `add_assists_features` **overwrites** the 12 contract columns unconditionally, so a minutes-builder column of the same name cannot survive.

## Public API

```python
CURRENT_ASSISTS_FEATURES = [
    "predicted_minutes_oof",
    "start_rate_10",
    "ast_mean_10",
    "ast_mean_20",
    "assists_per_min_10",
    "team_ast_mean_10",
    "team_fgm_mean_10",
    "team_pace_mean_10",
    "opponent_team_ast_allowed_mean_10",
    "opponent_team_pace_mean_10",
    "days_rest",
    "is_home",
]

ASSISTS_FEATURE_CHALLENGERS = {
    "exposure_history": ["min_mean_10"],
    "usage_role": ["usg_wmean_10"],
    "position_role": ["position_guard_prior"],
    "assist_efficiency": ["ast_pct_wmean_10"],
    "passing_tracking": ["passes_per_min_10"],
    "opponent_make_environment": [
        "opponent_team_fgm_allowed_mean_10",
    ],
}

def add_assists_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Write the 12 contract columns. Leave predicted_minutes_oof as float64 NaN."""

def build_pregame_assists_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Feature candidates using only history rows with game_date < as_of."""
```

`add_assists_features`:

- Returns the input columns plus the 12 contract columns. Builder temps (`_row`, `_started_obs`, `_ast_obs`, `_minutes_obs`, and any other `_` column **the builder created**) are dropped. Input columns are kept.
- Overwrites the 12 contract columns if they already exist.
- Preserves the input **row order and index labels**, including duplicate labels. Mechanism, not goal: copy the frame; set `_row = np.arange(len(frame))`; sort/compute/join **on `_row` only**; `sort_values("_row")`; assign `frame.index` to the result wholesale. Never `reindex` / `join` / `align` on the original labels. Duplicate `RangeIndex` after a concat is the normal pregame case; label-based reindex raises or fans out rows.
- `build_pregame_assists_features` returns candidate rows in `candidates` order with `candidates` index labels (restored from `(game_id, player_id)`, not from concat labels).
- Missing **feature-source values** (NaN `ast`, NaN `team_pace`, …) become IEEE `NaN` features (`float64`), never imputed sentinels, never `pd.NA` / object.
- Missing **canonical minutes column**: if neither `minutes` nor `min` is in the frame, **raise** `ValueError`. That is a frame-construction error. It must not silently NaN `ast_mean_*`, `start_rate_10`, `assists_per_min_10`, and `days_rest`. A present column of all-NaN values is a data gap and does NaN those features without raising.

Challenger names are frozen for later one-at-a-time tests. This slice does not emit those columns. `touches_per_min_10` does not advance unless the passes challenger first wins.

## Canonical sources

| Quantity | Primary | Alias | Rule |
|---|---|---|---|
| Assists count | `ast` | `assists` | Use `ast` when the column exists. Use `assists` only if `ast` is absent from the frame. |
| Minutes | `minutes` | `min` | Use `minutes` when the column exists. Use `min` only if `minutes` is absent. If both columns are absent, raise `ValueError`. |

Leakage tests still NaN **both** aliases on Game N. A leftover tracking or clock value must not change that row’s features.

## Feature constructions

Copy the input. Assign `_row = np.arange(len(frame))`. Sort copies by (`player_id`, `game_date`, `game_id`) for player lookups and by (`team_id`, `game_date`, `game_id`) for team-game trailing means. Join computed columns onto `_row`. `sort_values("_row")`, drop builder temps, set `.index` to the original index object. Do not `reindex` on labels.

Only `days_rest` is season-scoped. Player `ast_mean_*`, `assists_per_min_10`, `start_rate_10`, and every team/opponent trailing-10 **cross the offseason**. That is deliberate: a mean has no 180-day-spike pathology, and prior-season form is informative. Do not store `season_year` on the team-game table; it is unused there.

For each output row \(i\), player windows use qualifying **prior** rows of that `player_id` with `(game_date, game_id)` strictly before \(i\). Window length K with `min_periods=1`: if 3 qualifying priors exist, the mean is over those 3. There is **no history-depth column**. A 1-game mean and a 10-game mean are indistinguishable to a later booster. That is deliberate in this slice (YAGNI); do not add `n_prior_*`.

| Feature | Construction |
|---|---|
| `predicted_minutes_oof` | `float64` column of IEEE `NaN` (`np.nan`), not `pd.NA`, not object. Trainers join expanding-window OOF minute means later. Actual Game N minutes are never a feature. |
| `start_rate_10` | Among prior **appearances** (minutes \(> 0\)), mean of the start indicator. Indicator = 1 iff stripped `start_position` is nonempty and not `""` / `"nan"` / `"<NA>"`. Absent, empty, or those sentinels count as a **non-start (0) and still occupy a window slot**. This is not analogous to dropping NaN `ast`. Partial `start_position` on a direct `add_assists_features` call biases the rate down. Pregame is safe: history is unblanked, and blanked candidates never become history via the per-date loop. DNPs / blanked-minute candidates do not enter the window and do not need to qualify to receive the feature. |
| `ast_mean_10` | Mean of canonical `ast` over the last 10 prior appearances that also have finite `ast`. Zero is valid; NaN `ast` does not fill a slot. |
| `ast_mean_20` | Same, K = 20. |
| `assists_per_min_10` | See paired mask below. |
| `team_ast_mean_10` | Mean of the last 10 **valid** prior team-games’ `team_ast` for `team_id`, with `game_date <` row \(i\)’s `game_date`. |
| `team_fgm_mean_10` | Same, `team_fgm`. |
| `team_pace_mean_10` | Same, `team_pace`. Computed once on the team-game table; own and opponent joins reuse that column on two keys. |
| `opponent_team_ast_allowed_mean_10` | Same trailing mean of `opp_ast` (assists allowed) for `opp_team_id`, with `game_date <` row \(i\)’s `game_date`. Not joined on tonight’s `game_id`. |
| `opponent_team_pace_mean_10` | The same `team_pace` trailing-10 column, looked up for `opp_team_id` with `game_date <` row \(i\)’s `game_date`. |
| `days_rest` | Season-scoped. See below. |
| `is_home` | 1.0 if `matchup` contains `\bvs\.`, else 0.0. Absent **column** and absent/NA **value** both yield NaN. The regex is **case-sensitive**, matching minutes. If silver ever uppercases `matchup`, `is_home` silently goes to 0. That risk is accepted; do not add `case=False` in this slice. |

No product term such as `assists_per_min_10 * predicted_minutes_oof`.

### Paired `assists_per_min_10`

Independent `prior_sum(ast, 10) / prior_sum(minutes, 10)` is forbidden. Pandas rolling sums skip NaNs per series, so 10 assist-games over 7 minute-games inflates the rate.

Qualifying prior row for the rate: canonical minutes finite and \(> 0\), **and** canonical `ast` finite. Mask both series to those rows before summing. Then

\[
\texttt{assists\_per\_min\_10}(i)
=
\frac{\sum \mathrm{ast}\text{ on the last 10 qualifying priors}}
{\sum \mathrm{minutes}\text{ on those same rows}}.
\]

Zero or empty denominator → NaN. Do not roll a per-game rate.

### `days_rest`

Reset by `player_id`, `season_year` so the offseason is not a 180-day rest spike.

- Minuend = row \(i\)’s `game_date` (always; \(i\) stays on the output panel even if minutes are blank).
- Previous date = `game_date` of the latest **prior appearance** in that player-season (canonical minutes finite and \(> 0\), `(game_date, game_id)` strictly before \(i\)).
- Value = `(minuend − previous date).days`
- No such prior appearance → NaN (not 0, not 180)
- No 30-day clip

Blanking Game N minutes does not change Game N’s `days_rest`: Game N is the minuend, not the previous-appearance lookup. Game N minutes are allowed to change whether Game N counts as an appearance **for a later row**; that is why the pregame per-date loop exists.

### Team-game table

Built from **all** input rows (not appearance-filtered). A DNP or blanked-minute player row may still carry team box columns.

One row per `(team_id, game_id)` via **null-aware reduction**, not `drop_duplicates(..., keep="first")`. Positional first after sort can pick a teammate whose `team_ast` / `team_fgm` / `team_pace` / `opp_ast` is NaN and poison ten windows.

For each `(team_id, game_id)`:

- `game_date`, `opp_team_id`: first non-null after sorting teammates by `player_id`
- `team_ast`, `team_fgm`, `team_pace`, `opp_ast`: first **finite** value after that sort; all-null → NaN
- Teammates are assumed to share team box stats. If they disagree, `player_id` order makes the pick deterministic.

A team-game with NaN `team_ast` does not occupy a slot in `team_ast_mean_10`; it may still occupy a slot in `team_pace_mean_10` if pace is finite. Stats are not pair-masked across columns.

### Opponent lookup (merge_asof, not same-`game_id` join)

A same-`game_id` join onto `prior_roll` only lands if the opponent’s team-game row exists for tonight, which requires opponent players in the frame. Full silver training has that; a partial candidate slate does not, and both opponent features would be silent all-NaN.

Do **not** join opponent features on `(game_id, opp_team_id)`.

After each completed team-game, store trailing-10 means **including that game** (valid games only, `min_periods=1`). Look up the latest snapshot with `game_date <` the player row’s `game_date`.

pandas `merge_asof` requires the `on` key to be **globally** monotonic in both frames, not merely within `by` groups. A `team_id`-major sort used for the trailing means will raise `ValueError: left keys must be sorted`. Immediately before each asof, stably sort **both** sides by `game_date` alone.

```python
pd.merge_asof(
    left.sort_values("game_date"),
    snapshots.sort_values("game_date"),
    on="game_date",
    direction="backward",
    allow_exact_matches=False,
    left_by="team_id",      # own-team join
    right_by="team_id",
)
pd.merge_asof(
    left.sort_values("game_date"),
    snapshots.sort_values("game_date"),
    on="game_date",
    direction="backward",
    allow_exact_matches=False,
    left_by="opp_team_id",  # opponent join
    right_by="team_id",
)
```

Do not pass a single `by=` (that requires matching names). Do not rename `opp_team_id` to `team_id` to force a `by=` — that silently matches own-team.

The asof left frame must carry `_row`. After each merge, attach snapshot columns by `_row`, not by the date-sorted index. Same-calendar-day different `game_id` (doubleheader) is out of scope; strict date inequality keeps Game N out.

Compute `team_pace` trailing-10 **once**. Join it as `team_pace_mean_10` on `team_id` and as `opponent_team_pace_mean_10` on `opp_team_id`.

Example: player faced Team A, then Team B. Tonight’s `opponent_team_ast_allowed_mean_10` is Team B’s trailing-10 `opp_ast` over B’s games with `game_date <` tonight, not Team A’s game and not the player’s trailing `opp_ast`.

Test 5’s frame still includes both teams on historical games (so B has a snapshot). A separate one-sided-slate test covers tonight without opponent players.

## Pregame / as-of

`build_pregame_assists_features`:

1. Keep history with `game_date < as_of` (normalized timestamp). Do not drop history rows on minutes.
2. Blank candidate outcomes (list below). Do not blank pregame-known keys.
3. For each candidate `game_date` group: `pd.concat([history, group], ignore_index=True)`, run `add_assists_features`, keep the group’s rows by `(game_id, player_id)` — never by concat index labels.
4. After all date groups, left-merge those featured rows onto `candidates[["game_id", "player_id"]]`, then assign `candidates.index` wholesale. Duplicate default `RangeIndex` on history and candidates is expected and unused. Candidates on later dates do not observe earlier candidates (each concat is history + that date’s group only).

The per-date loop is required for **window composition**, not only leakage. A blanked earlier candidate concatenated into a later candidate’s panel would sit in the timeline as a non-appearance (NaN minutes). If an implementation used last-K *panel* rows with independent `skipna` instead of last-K qualifying appearances, that hole would dilute the window. The loop makes “earlier candidate is not history” true even if a later optimizer changes the window code. Cost is \(O(\text{dates} \times \text{history})\), acceptable at synthetic scale.

### Blank list

Blank on candidates (and in leakage mutations):

- `ast`, `assists`
- `min`, `minutes`, `min_sec`
- `start_position`
- `team_ast`, `team_fgm`, `team_pace`
- `opp_ast`, `opp_pace`
- `pass`, `tchs`, `sast`, `ftast`

`start_position` blanks to `""`. Other listed columns blank to NaN. Do not share `src/features/points/pregame.py`.

### Do not blank (pregame-known)

`matchup`, `game_date`, `season_year`, `team_id`, `opp_team_id`, `player_id`, `game_id`. Blanking `matchup` would zero `is_home` silently.

### Challenger sources not blanked in this slice

This slice does not emit challenger columns, so it does not blank `usg_pct`, `ast_pct`, `opp_fgm`, or `pos`. That is a leak if those columns are added later without extending this list. The challenger slice must add them to the blank list **before** emitting `usg_wmean_10`, `ast_pct_wmean_10`, `opponent_team_fgm_allowed_mean_10`, or `position_guard_prior`. `pass` / `tchs` / `sast` / `ftast` are already blanked. `min` / `minutes` are already blanked (`min_mean_10`).

## Tests

Synthetic frames only. No network. No `2025-26` silver. Compare features on the original index (order-preserving return is part of the contract).

1. **Shift-before-lookup.** Mutating Game N `ast`, `minutes`, `team_ast`, `opp_ast`, or `start_position` does not change that row’s contract features, including `days_rest`.
2. **Blank both aliases.** NaN-ing both `ast` and `assists`, and both `min` and `minutes`, on Game N leaves that row’s features unchanged, including `days_rest`.
3. **Paired rate.** On a clean 10-appearance history, `assists_per_min_10` equals sum of prior official `ast` over sum of prior official minutes. **NaN-hole case:** insert a prior appearance with finite minutes and NaN `ast`. That hole must not enter numerator or denominator. An independent-skip implementation fails this.
4. **Null-aware team reduction.** Two teammates in one game, one with finite `team_ast` and one with NaN `team_ast` listed first after sort: both player rows get the finite value’s team snapshot. Game N `team_ast` does not enter the mean.
5. **Scheduled opponent.** History includes **both** clubs’ player rows. Player faces A then B. Tonight’s `opponent_team_ast_allowed_mean_10` equals B’s prior valid `opp_ast` mean, not A’s game. Give own team and B **distinct** `team_pace` histories and assert `team_pace_mean_10 != opponent_team_pace_mean_10` (same computed column, two keys; a swapped join would match).
6. **One-sided slate.** History has B’s games; tonight’s candidates are only the player’s team (no B players, no B row at tonight’s `game_id`). Opponent features are still B’s strict-prior snapshots, not all-NaN.
7. **Snapshot-includes-self.** From the synthetic team-games, compute B’s trailing-10 `opp_ast` over games with `game_date <` tonight **including B’s last pre-tonight game**. Tonight’s `opponent_team_ast_allowed_mean_10` equals that number. Make the last pre-tonight `opp_ast` an outlier so a double-shift (excluding that game) fails. No need to export the snapshot table.
8. **`game_date < as_of`.** A candidate dated on `as_of` cannot see same-day box scores. A later candidate cannot treat an earlier candidate as history. History and candidates both use a default `RangeIndex`.
9. **Contract.** `CURRENT_ASSISTS_FEATURES` is exactly the 12 names above, in that order. After `add_assists_features`, `predicted_minutes_oof` exists, dtype `float64`, all IEEE `NaN`.
10. **`days_rest` opener.** First player-season row is NaN. Second row (an appearance after an appearance) is the integer date gap. A candidate with blank minutes still receives `days_rest` from the prior appearance date.
11. **Index / order.** Input with a non-monotonic **and duplicate** index and unsorted dates comes back in the same order with the same index labels (duplicates included).
12. **Missing minutes column.** A frame with `ast` but neither `minutes` nor `min` raises `ValueError`. It does not return all-NaN assist windows.

## Important exclusions (do not emit)

- Raw Game N box, tracking, team, or opponent outcomes
- `ast_ratio`, `ast_to`
- Tracking `assists`, `sast`, `ftast` as features (they are blanked; they are not contract columns)
- Direct `pos`
- `game_total`, `team_spread`, `player_team_spread`
- Current lineup / injury context
- Challenger columns listed in `ASSISTS_FEATURE_CHALLENGERS`
- `touches_per_min_10`

## Out of scope

Poisson / NegBin booster, squared-error baseline, discrete CRPS, randomized PIT, OOF minute stacking beyond leaving the column NaN, \(g/\beta/\epsilon\), half-point AST pricing, feature-challenger bakeoffs, reading or scoring `2025-26`, changing minutes or points production artifacts, a history-depth column, `case=False` on `is_home`, same-day doubleheader opponent games.

## Success

`.venv/bin/python -m pytest tests/features/test_assists_features.py tests/features/test_pregame_assists.py -q` is green. Existing minutes and points feature tests still pass. No production joblib is written.
