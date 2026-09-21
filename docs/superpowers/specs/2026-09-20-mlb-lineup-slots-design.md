# Posted lineups and batter K/PA (vector, not mean)

Date: 2026-09-20  
Revised: 2026-09-20 (review 2)

Approved: persist cutoff-safe starting nines as a per-slot vector, plus Stats API play-by-play batter K/PA with hierarchical vs-hand shrinkage. Do not wire this into `nb_k_v1`, the BF mixture, Log5, or Poisson-binomial this slice.

Parent: `docs/superpowers/specs/2026-09-16-mlb-sp-k-design.md`. This slice is the lineup source that the mixture will consume later. It replaces the scalar `lineup_k_rate_shrunk` mean as the stored object; it does not delete that column from `k_mvp_v1`.

## Goal

For each team-game, store the ordered official or live 1–9 with per-batter shrunk K/PA vs the probable opposing starter’s hand, plus the hand-agnostic overall as a fallback. For each batter, store lagged, cutoff-strict, hand-split K/PA from play-by-play. The mixture can later walk slots and form a Poisson-binomial; this slice only puts the un-collapsed vector on disk.

A scalar mean cannot be un-collapsed. Do not store one.

## Non-goals (this slice)

- Do not change `STRIKEOUT_FEATURE_COLUMNS`, `fit_strikeouts`, or `nb_k_v1`.
- Do not change the game-log path’s `lineup_state = "team_fallback"` or `lineup_batter_ids_json = "[]"`.
- Do not implement Log5, the BF mixture, or Poisson-binomial convolution.
- Do not ingest Statcast.
- Do not retune `batter_k_prior_strength`. 225 stays in yaml as an inherited value, not a locked NLL choice. Ship a spread diagnostic (below), not a sweep.
- Do not scrape FanGraphs, Lahman, or Retrosheet.

## Why this source

Lineups post 2–4 hours before first pitch. `forecast_horizon_hours = 2`, so a live snapshot at cutoff is legitimate. Team K/PA cannot feed a BF-slot walk; the nine IDs can. The two compound after the mixture exists, which is why scoring stays off this slice.

## Verification (00 filter)

On 24 games / 48 team-games (2018–2025):

- `players.*.battingOrder` in `{100,200,…,900}` (last digit `0`) recovered a complete 1–9 on 48/48 sides.
- That vector matched the first nine unique PBP batters on 47/48 sides.
- `teams.*.battingOrder` matched PBP on 16/48 sides, and **0/32** once a substitution existed.

The one PBP mismatch is the definition, not a parser failure: 2019-05-15 Rays, opener Ryne Stanek listed ninth (`900`) and pinch-hit for (Andrew Velazquez) before he batted. PBP-first-nine includes the PH. The 00 vector keeps Stanek. Document this in `docs/mlb/data_dictionary.yaml` as the canonical reason PBP-first-nine is not the starting nine.

The 48-side sample justifies the parser rule. The real test at scale is the backfill’s per-side nine-slot coverage assertion (~39k sides), not this sample.

**Parser rule:** starting nine = 00-code `battingOrder` values, ordered by slot = first digit. Never `teams.*.battingOrder`.

## Placement

| Path | Owns |
|---|---|
| `src/mlb/schemas.py` | `BATTER_PA_COLUMNS`, `LINEUP_SLOT_COLUMNS`; register both in `TABLE_SCHEMAS` |
| `src/mlb/pipeline/pbp.py` | Fetch/parse PBP (prefer live feed); gzip cache; idempotent write of `batter_pas` |
| `src/mlb/pipeline/lineup_slots.py` | 00-filter parse; live vs historical as-of; freeze rates; coverage report |
| `src/mlb/models/batter_rates.py` | Two-level shrink, platoon odds-ratio, trailing league K/PA, `rate_version`, `is_strikeout` at read |
| `src/mlb/config.py` / `config/mlb.yaml` | `batter_hand_prior_strength: 400`; keep `batter_k_prior_strength: 225` unlabeled as locked |
| `src/mlb/cli.py` | `ingest-play-by-play`, `ingest-lineup-slots` |
| `src/mlb/pipeline/parse.py` | Change `parse_lineups` to 00-filter (or thin wrapper around lineup_slots parse) |
| `docs/mlb/data_dictionary.yaml` | New tables, Stanek note, Ohtani two-way note |
| `docs/mlb/CONTRACT.md` | New tables and signatures |
| `tests/mlb/pipeline/test_pbp.py` | `pa_id` idempotency, `event_type` retained, `is_strikeout` derived at read |
| `tests/mlb/pipeline/test_lineup_slots.py` | 00 vs array, Stanek, Ohtani, −24h stamp, DH skip, live still writes, coverage, `observed_before_cutoff` |
| `tests/mlb/models/test_batter_rates.py` | Hierarchy, leave-one-split-out, platoon offset, switch-hitter cells, `rate_version` hash |

Reuse `src/mlb/pipeline/http.py` (`HttpFn`), `src/mlb/pipeline/gamelogs.py` (`default_http`, people cache), and `MlbStore` concat-then-write. Do not reuse synthetic Statcast `plate_appearances` as the batter history.

Raw JSON snapshots are gzip-compressed on disk. Prefer one `GET https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live` per game (it carries `liveData.plays` and `liveData.boxscore`). Fall back to `v1/.../playByPlay` plus `v1/.../boxscore` only when the live payload lacks plays. That is ~19.4k calls, not ~39k.

## Public API

```python
LIVE_FEED_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
PBP_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
BOXSCORE_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

STRIKEOUT_EVENT_TYPES = frozenset({
    "strikeout",
    "strikeout_double_play",
    "strikeout_triple_play",
})

KPA_WINDOWS_DAYS = (60, 365)
KPA_PRIOR_SEASONS = 2
BOXSCORE_00_LEAD = pd.Timedelta(hours=24)
RATE_VERSION_PREFIX = "kpa_"
PITCHER_POSITIONS = frozenset({"P"})

def is_strikeout(event_type: str, *, events: frozenset[str] = STRIKEOUT_EVENT_TYPES) -> int:
    """Derived at read. Never stored on batter_pas."""

def parse_play_by_play(
    raw_payload: str | bytes,
    snapshot_id: str,
    ingested_at: datetime | pd.Timestamp,
) -> pd.DataFrame: ...

def ingest_play_by_play(
    config: MlbConfig,
    *,
    game_pks: list[int],
    http: HttpFn | None = None,
) -> pd.DataFrame: ...

def parse_starting_nine(raw_payload: str | bytes, game_pk: int | None = None) -> pd.DataFrame:
    """00-filter 1–9. Not teams.*.battingOrder."""

def freeze_lineup_slot_rates(
    slots: pd.DataFrame,
    batter_pas: pd.DataFrame,
    people: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame: ...

def ingest_lineup_slots(
    config: MlbConfig,
    *,
    game_pks: list[int],
    provenance: str,  # "live_feed" | "boxscore_00"
    http: HttpFn | None = None,
) -> pd.DataFrame: ...

def shrink_batter_k_pa(
    pas: pd.DataFrame,
    *,
    batter_id: int,
    opposing_pitcher_hand: str,
    bats: str,
    cutoff: pd.Timestamp,
    league: LeagueKPa,
    config: MlbConfig,
) -> dict[str, dict[str, float]]:
    """Keyed by window name '60' | '365' | 'prior2'.
    Each value has k_pa_vs_hand_shrunk, k_pa_overall_shrunk,
    pa_vs_hand, pa_all."""

def league_platoon_odds_ratio(
    *,
    bats: str,
    league_k_pa_cell: float,
    league_k_pa_bats: float,
) -> float:
    """1.0 only when bats is missing or empty.
    For L/R/S, callers pass cells computed inside that bats value."""

def rate_version(config: MlbConfig) -> str: ...

def lineup_identity_mismatch_rate(live: pd.DataFrame, official: pd.DataFrame) -> float:
    """Join on (game_pk, team_id, slot). Identities only. No K, no quotes."""

def write_lineup_coverage(slots: pd.DataFrame, skips: pd.DataFrame, *, season: int) -> dict: ...

def assert_lineup_coverage(coverage: dict, *, season: int) -> None:
    """Fail on zero announced rows or complete-nine rate below floor."""
```

CLI:

```text
python -m src.mlb ingest-play-by-play --start-season 2018 --end-season 2025
python -m src.mlb ingest-lineup-slots --start-season 2018 --end-season 2025
python -m src.mlb snapshot-lineups --game-pk INT
```

`snapshot-lineups` keeps its name. It writes `lineup_slots` with `provenance=live_feed` through `parse_starting_nine`. `--fixture` reads `tests/mlb/fixtures/raw/` and never uses the network.

## Tables

### `batter_pas`

One row per plate appearance from Stats API play-by-play. Not the Statcast `plate_appearances` table.

| Column | Type | Rule |
|---|---|---|
| `pa_id` | string | `f"{game_pk}_{atBatIndex}"`. Identity key. |
| `game_pk` | int64 | |
| `at_bat_index` | int64 | PBP `atBatIndex`, stable within a game |
| `batter_id` | int64 | |
| `pitcher_id` | int64 | |
| `pitcher_hand` | string | `L` / `R` from `matchup.pitcher.p_throws` / `pitchHand` |
| `batter_bats` | string | `L` / `R` / `S` from people / id_map, not inferred from this PA’s stand |
| `batter_stand` | string | `L` / `R` from this PA’s `matchup.batSide` / `stand` |
| `event_type` | string | Raw `result.eventType` (or `event`). Never drop this. |
| `event_time_utc` | datetime64[us, UTC] | `about.startTime` or `about.endTime`. See imputation below. |
| `event_time_imputed` | int64 | 1 when start/end were missing |
| `is_pitcher_in_game` | int64 | 1 iff this `batter_id` appears as `pitcher_id` on any PA in this `game_pk` |
| `ingested_at_utc` | datetime64[us, UTC] | **Wall clock of the write.** Dedupe recency only. Not an as-of. |
| `snapshot_id` | string | Raw snapshot of this game’s payload |

There is **no stored `is_strikeout`**. `is_strikeout(event_type)` is derived at read from `STRIKEOUT_EVENT_TYPES` (and therefore from `rate_version`). Test 8 flips the event set in the helper; stored rows do not change.

Denominator for K/PA is these rows (plate appearances), not at-bats. Walks, HBP, sac, and CI stay in the denominator. Revising `STRIKEOUT_EVENT_TYPES` is a re-read, not a 19k-call refetch.

**PBP cutoff join uses only `event_time_utc < cutoff`.** Do not also require `ingested_at_utc < cutoff` on `batter_pas`. A 2019 PA is a fact of the world; a 2026 fetch must not hide it from a 2019 cutoff. `ingested_at_utc` exists so re-ingest can keep the latest wall-clock row per `pa_id`.

Idempotent re-ingest: append, then keep the row with latest **wall-clock** `ingested_at_utc` per `pa_id`. Because that timestamp is the write time, a second pull is strictly later and the winner is determined.

#### Event-time imputation

Measure `about.startTime` / `endTime` missing rate on one fixture season before the 19k backfill. If it is 0, do not impute.

If times are missing: set `event_time_utc = original_scheduled_start + 4 hours` and `event_time_imputed = 1`. Cutoff-strict rate windows **exclude** `event_time_imputed == 1` rows. A traditional DH has game 2’s cutoff mid-afternoon; imputing a game-1 late PA at first pitch would leak those PAs into game 2. The +4h fallback is late (conservative for “already happened”) and the imputed flag drops them from windows anyway.

### `lineup_slots`

Long format: nine rows per snapshot. Append-only. Never update in place.

| Column | Type | Rule |
|---|---|---|
| `game_pk` | int64 | |
| `team_id` | int64 | Batting team |
| `side` | string | `home` / `away` |
| `slot` | int64 | 1–9 |
| `batter_id` | int64 | 00-filter ID for that slot |
| `slot_is_pitcher` | int64 | 1 iff that 00-slot’s boxscore `position.abbreviation` is `P` |
| `k_pa_vs_hand_shrunk_60` | float64 | Two-level vs-hand posterior, 60-day window. NaN if opposing hand unknown |
| `k_pa_vs_hand_shrunk_365` | float64 | Same, 365-day (mixture default) |
| `k_pa_vs_hand_shrunk_prior2` | float64 | Same, prior two seasons |
| `k_pa_overall_shrunk_60` | float64 | Leave-one-split-out overall, 60-day. Never NaN solely because hand is unknown |
| `k_pa_overall_shrunk_365` | float64 | Same, 365-day. Fallback when vs-hand is NaN so one slot does not null a Poisson-binomial |
| `k_pa_overall_shrunk_prior2` | float64 | Same, prior two seasons |
| `pa_vs_hand_60` | float64 | Raw PA count vs that hand, 60-day. Not named `n_eff_*` |
| `pa_vs_hand_365` | float64 | |
| `pa_vs_hand_prior2` | float64 | |
| `pa_all_60` | float64 | Raw PA count any hand, 60-day |
| `pa_all_365` | float64 | |
| `pa_all_prior2` | float64 | |
| `opposing_pitcher_hand` | string | Hand of the probable opposing starter **as of cutoff** (latest `game_versions` with `valid_from_utc < cutoff`). Not named `vs_hand`. |
| `vs_pitcher_id` | Int64 | That probable opposing starter’s MLB id. Null if unknown. A later scratch is detectable against `boxscore_00`. |
| `lineup_state` | string | `announced` / `projected` / `team_fallback`. Both honest nines are `announced`. `projected` unused this slice. |
| `observed_before_cutoff` | int64 | 1 iff `provenance == "live_feed"`. First-class leak bit. Invariant tested. |
| `provenance` | string | `live_feed` or `boxscore_00` |
| `rate_version` | string | `kpa_` + 12 hex chars of the hash below |
| `ingested_at_utc` | datetime64[us, UTC] | **As-of for the card.** See table below. Not write recency. |
| `snapshot_id` | string | |
| `season` | int64 | `gameData.game.season` from the payload. Do not derive from the UTC start. |

**Dedupe / read key:** `(game_pk, team_id, slot, rate_version)`. Filter to the requested `rate_version` (default: current config hash) **before** latest-wins. Then, among remaining rows with `ingested_at_utc < cutoff`, keep the latest `ingested_at_utc` per `(game_pk, team_id, slot, rate_version)`.

A re-freeze under a new `rate_version` is a new row, not a tie on a deterministic timestamp. A `live_feed` re-freeze **must copy the original snapshot wall clock** into `ingested_at_utc`; bumping it to now would land after cutoff and drop the row. `ingested_at_utc` is never the version discriminator.

## As-of and join rules

Cutoff remains `scheduled_start_utc - forecast_horizon_hours`.

- `lineup_slots`: `ingested_at_utc < cutoff` (as-of of the card).
- `batter_pas`: `event_time_utc < cutoff` only, and `event_time_imputed == 0` inside rate windows.

`scheduled_start_utc` is the **earliest** `game_versions.scheduled_start_utc` for that `game_pk` (original card). Never post-game `gameDate` / actual first pitch. Rain delay must not move a historical `boxscore_00` row later.

Probable opposing starter is the **opposite** rule: latest `game_versions` row with `valid_from_utc < cutoff` (and `valid_to_utc` null or `> cutoff`). Earliest would freeze a weeks-stale rotation. “A later scratch is detectable” only holds if `vs_pitcher_id` is as-of-cutoff.

| Provenance | `lineup_state` | `observed_before_cutoff` | `ingested_at_utc` |
|---|---|---|---|
| `live_feed` | `announced` | 1 | wall clock of the **original** snapshot (must already be `< cutoff` or the snapshot is dropped). Re-freezes copy this clock. |
| `boxscore_00` | `announced` | 0 | `original_scheduled_start - 24 hours` |

`boxscore_00` timestamps do **not** contain `forecast_horizon_hours`. `observed_before_cutoff=0` already marks the row retrospective. Stamping at `start − horizon − 1 minute` would drop every historical row if the yaml horizon moved to 3h — the same silent `team_fallback` failure the 1-minute margin was meant to prevent. `start − 24h` is `< cutoff` for any plausible horizon (2h, 3h, 12h).

Doubleheader game 2: if original scheduled start is missing or is a dummy (midnight UTC, equal to game 1, or NaT), write **zero** `boxscore_00` rows for that `game_pk`. Do not invent a timestamp. `live_feed` snapshots for that game still write when taken before cutoff.

## Coverage (not just zero-announced)

Zero announced rows is only one failure. DH2-with-dummy-start, postponed `game_pk`s with no boxscore, and sides without exactly nine distinct 00 codes all look like “no rows” and would still pass a per-season `announced > 0` check while dropping every DH2 for eight seasons.

`write_lineup_coverage` records, per season:

- `n_game_pks`
- `n_boxscore_ok`
- `n_sides_complete_nine` (exactly nine distinct 00 codes)
- `n_slots_written`
- `n_skip_dh2_dummy_start`
- `n_skip_no_boxscore`
- `n_skip_incomplete_nine`

`assert_lineup_coverage` raises if:

- `n_slots_written == 0`, or
- among games with a boxscore, `n_sides_complete_nine / (2 * n_boxscore_ok) < 0.95`

Skip reasons stay in the report even when the assertion passes.

## Shrinkage

Windows (shift-then-roll, cutoff-strict, non-imputed PAs): 60-day, 365-day, prior two seasons. **This slice stores all three** plus overall and vs-hand. A later mixture slice is a pure read of `lineup_slots`. `rate_version` includes the window definitions.

`shrink_batter_k_pa` returns a dict keyed by `'60' | '365' | 'prior2'`. It does not take a scalar `window_days` and “also” compute the others.

Two-level posterior for vs-hand, per window:

1. Leave-one-split-out overall:
   - `pa_rest = pa_all − pa_hand`, `k_rest = k_all − k_hand`
   - `overall = shrink_rate(k_rest, pa_rest, league_k_pa_trailing, batter_k_prior_strength)` with `batter_k_prior_strength = 225` (inherited, not locked).
   - Then `batter_hand_prior_strength = 400` means 400. Do not document a residual dependence.
2. `prior_mean` for the split is `overall` shifted by the league platoon odds ratio for `(bats, opposing_pitcher_hand)`:
   - `odds(p) = p / (1-p)` with `p` clipped to `(1e-6, 1-1e-6)`
   - `league_platoon_odds_ratio = odds(league_k_pa_cell) / odds(league_k_pa_bats)`
   - Callers compute both cells **inside the same `bats` value** (including `S`). Switch hitters get the S-specific vs-L / vs-R offset, not the RHB-vs-RHP offset and not a hardcoded `1.0`.
   - `1.0` only when `bats` is missing or empty.
   - `prior_odds = odds(overall) * league_platoon_odds_ratio`
   - `prior_mean = prior_odds / (1 + prior_odds)`
3. `k_pa_vs_hand_shrunk = shrink_rate(k_hand, pa_hand, prior_mean, batter_hand_prior_strength)` with `batter_hand_prior_strength = 400`.
4. If `pa_hand == 0`, `shrink_rate` already returns `prior_mean`. Tests must assert that, not NaN.

`league_platoon_odds_ratio` does not take `opposing_pitcher_hand`. Cell selection is upstream. The function only needs `bats` (for the missing/empty → 1.0 branch) and the two rates.

League K/PA series: trailing 365-day, cutoff-strict, **excluding pitcher-batter PAs**.

A PA is excluded from the league series when `is_pitcher_in_game == 1` **unless** that `batter_id` occupies a 00-slot in that `game_pk` whose position abbreviation is **not** `P` (two-way hitter batting as DH / OF / etc.). Relievers who bat in 2018–2021 NL games are in the PBP `pitcher_id` set and are not in the 00 nine, so they drop out automatically. The starting pitcher’s slot-9 PAs drop out because `is_pitcher_in_game == 1` and their 00-slot is `P`.

Ohtani 2022+ as a DH (position not `P`) while also pitching that day: `is_pitcher_in_game == 1`, but the 00-slot carve-out keeps those ~hitting PAs in the league series, and `slot_is_pitcher = 0` on his batting slot. Named test next to Stanek. Confirm the boxscore `position.abbreviation` on a real 2022+ Ohtani start in the fixture; if it is `P` on a hitting slot, treat that as a parser incident and do not exclude the hitting PAs — prefer the non-`P` position when a player has both.

`batter_hand_prior_strength = 400` is the intended regime: league K/PA ≈ 0.22, `p(1-p) ≈ 0.172`, true platoon-deviation sd ≈ 0.02 → MoM `k ≈ 430`. With ~175 vs-L PAs, weight on own split data is `175/(175+400) ≈ 30%`.

`batter_k_prior_strength = 225` is **not locked**. Static MoM on between-batter K% sd ≈ 0.06 is `k ≈ 48`. At k=225 a 100-PA part-timer gets ~31% weight on own data versus ~68% at k=48. Slots 7–9 are where lineup composition carries information. This slice does not sweep 225. It **does** ship a backfill diagnostic: spread of frozen `k_pa_vs_hand_shrunk_365` across slots 1–9 versus raw K/PA, by `pa_all_365` decile. If shrunk spread is near zero, the vector is not doing the job.

## `rate_version`

Canonical JSON, UTF-8, `json.dumps(..., sort_keys=True, separators=(",", ":"))`:

```json
{
  "exclude_pitcher_batters": "pbp_pitcher_ids_minus_two_way",
  "hand_prior": 400.0,
  "overall": "leave_one_split_out",
  "overall_prior": 225.0,
  "platoon": "odds_ratio",
  "prior_seasons": 2,
  "strikeout_events": ["strikeout", "strikeout_double_play", "strikeout_triple_play"],
  "windows_days": [60, 365]
}
```

`rate_version = "kpa_" + sha256(text).hexdigest()[:12]`. Changing either prior strength, any window, the event set, platoon method, overall construction, or pitcher exclusion **must** change the string.

## Historical vs live information content

Both honest cards are `lineup_state=announced` (content completeness: a 1–9 exists).

- `live_feed` + `observed_before_cutoff=1`: card as known at cutoff; late scratches unresolved.
- `boxscore_00` + `observed_before_cutoff=0`: official starting nine; scratches after cutoff are resolved. Strictly more informative than production. Backtest will look slightly better than live unless folds filter on `observed_before_cutoff`.

Invariant: `observed_before_cutoff == (provenance == "live_feed")`. Test it. The only way it drifts is a bug.

Once 2026 `live_feed` rows exist, `lineup_identity_mismatch_rate` joins them to `boxscore_00` on **`(game_pk, team_id, slot)`** and reports the fraction of differing `batter_id`s. Joining on `game_pk`+`slot` alone is a 2×2 per slot and reports ~50% mismatch on perfect data. Identities only — no quotes, no K. If ~1%, note it; if ~5%, retrospective rows need a haircut or exclusion from scoring folds. This slice ships the function; tests use a fixture pair.

## Ingest sequence

1. `ingest-play-by-play` for `game_pk`s in the season window. Prefer gzipped live-feed snapshots. Set `is_pitcher_in_game` from the PBP pitcher-id set. Rate-limit via the existing client.
2. `ingest-lineup-slots --provenance boxscore_00` for the same keys. Parse 00 nines per game (no season-wide identities-before-rates gate: reliever exclusion is already on `batter_pas`). Resolve original start from earliest `game_versions`; probable pitcher from latest-before-cutoff; skip DH2 with unusable start; stamp `ingested_at_utc = start − 24h`; freeze all three windows; write nine complete rows. Two-way carve-out uses that game’s 00-slot positions when building the league series for each cutoff. `write_lineup_coverage` + `assert_lineup_coverage` per season.
3. Live: `snapshot-lineups` writes `lineup_slots` with `provenance=live_feed` only if the 00-filter yields nine slots and wall-clock `ingested_at_utc < cutoff`.

If step 1 is interrupted, resume is safe because of `pa_id` + wall-clock recency. If step 2 is interrupted, skip `(game_pk, team_id, slot, rate_version, provenance)` already present.

## Existing code that must not be “half-wired”

- `src/mlb/pipeline/hf_tables.py` `pregame_from_starts` keeps `lineup_state="team_fallback"` and `lineup_batter_ids_json="[]"`.
- `src/mlb/pipeline/gamelog_features.py` keeps `missing_lineup = 1`.
- `src/mlb/schemas.py` `STRIKEOUT_FEATURE_COLUMNS` unchanged.
- Current `parse_lineups` using `teams.*.battingOrder` is **wrong** for completed games and is replaced by the 00 filter even for fixtures.

`lineup_k_rate_shrunk` in `k_mvp_v1` remains a mean for the unused Statcast feature builder. Do not store the new vector there.

## Tests (must exist)

Parser / slots:

1. 00-code nine ≠ end-of-game `battingOrder` when substitutions exist (the 0/32 case).
2. Stanek-style opener: slot 9 is the pitcher, `slot_is_pitcher=1`, PH is not in the vector.
3. Ohtani-style two-way: batting slot `slot_is_pitcher=0`; hitting PAs remain in the league series even if `is_pitcher_in_game=1`. Reliever pinch-hit PAs in an NL game are excluded from the league series.
4. `boxscore_00` `ingested_at_utc = start − 24h`; still `< cutoff` after raising `forecast_horizon_hours` to 3. A row stamped `start − 2h` is **not** the contract.
5. DH game 2 with missing original start writes zero `boxscore_00` rows.
6. DH game 2 still writes `live_feed` rows (the skip is provenance-specific).
7. `assert_lineup_coverage` passes on a healthy fixture; fails on zero written slots; fails when complete-nine rate is below 0.95 among boxscore games. Skip-reason counters are present for DH2 / no-box / incomplete nine.
8. `observed_before_cutoff == (provenance == "live_feed")` on every written row.
9. `lineup_identity_mismatch_rate` on two perfectly matching nines is 0, not ~0.5 (the `team_id` key).
10. Dedupe: two `rate_version`s for the same slot both survive; read path with version A does not return B. `live_feed` re-freeze keeps the original `ingested_at_utc`.
11. Probable pitcher is the latest `game_versions` before cutoff, not the earliest. Earliest still wins for `scheduled_start_utc`.

PBP:

12. Re-ingest of the same `game_pk` does not duplicate `pa_id`; latest **wall-clock** `ingested_at_utc` wins (second write is later).
13. `event_type` is stored; `is_strikeout` is not a column. Flipping `STRIKEOUT_EVENT_TYPES` in `is_strikeout()` changes the derived flag without a new payload.
14. Rate windows ignore `event_time_imputed == 1` rows. An imputed timestamp is `scheduled_start + 4h`, not `scheduled_start`.

Rates:

15. Hierarchical shrink: vs-hand posterior uses platoon-shifted **leave-one-split-out** overall as prior, not league and not overall-including-the-split.
16. Batter with zero vs-L PAs: result equals that prior mean; `pa_vs_hand_365 == 0`; not NaN. Also asserts unknown hand still fills `k_pa_overall_shrunk_*`.
17. `bats='S'` uses S-specific league cells (ratio need not be 1.0). Missing/empty `bats` gets ratio `1.0`.
18. `rate_version` changes when either prior strength or any window definition changes; unchanged when unrelated config changes (`strikeout_l2`).
19. Pitcher-batter PAs (starter slot 9 and reliever PH) are absent from trailing league K/PA; two-way hitting PAs are present.

Diagnostic (not a unit-test pass/fail gate): after the real backfill, print slot-1–9 spread of `k_pa_vs_hand_shrunk_365` vs raw K/PA by `pa_all_365` decile.

## Data dictionary notes (required lines)

- Stanek 2019-05-15: why PBP-first-nine is not the starting nine; 00-codes keep the opener.
- Ohtani 2022+: two-way carve-out so DH PAs stay in the league series; `slot_is_pitcher` follows batting-slot position, not “also pitched.”
- `observed_before_cutoff=0` rows are more informative than live; do not group by `lineup_state` alone when asking whether announced lineups pay.
- `batter_k_prior_strength=225` is inherited, not NLL-tuned. `pa_vs_hand_*` / `pa_all_*` are raw PA counts, not effective sample sizes.
- `batter_pas.ingested_at_utc` is write recency. Lineup as-of is `lineup_slots.ingested_at_utc`. PA as-of is `event_time_utc`.

## Out of scope reminders

Mixture: \(\sum_{bf} P(BF=bf)\cdot\text{PoissonBinomial}(k; p_1,\ldots,p_{bf})\) with Log5 odds-ratio \(p_i\) from pitcher rate, slot batter rate, and trailing league K/PA. That is the next slice after a BF mixture exists. This slice only makes the \(p_i\) vector reconstructable. Unknown-hand slots degrade to `k_pa_overall_shrunk_*` instead of nulling the walk.
