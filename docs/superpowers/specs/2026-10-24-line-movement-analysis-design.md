# Line movement analysis (daily notebook + local package)

Date: 2026-10-24  
Status: approved  
Approved decisions: full working notebook (option B); all player props including alternates; follower definition = windowed + uncapped (option C); package under notebook folder.

## Goal

Answer: which bookmakers move player-prop markets first, and which follow?

Deliver a dated exploratory notebook that loads seven providers’ scraped JSON for one slate day, validates and normalizes quotes, matches equivalent markets across books, detects line/price changes, measures leader/follower relationships, and writes a daily report for comparison with other dates.

Reusable logic lives next to the notebook so the notebook stays easy to follow. This slice does not write to Supabase and does not change `src/odds/` DB upsert paths.

## Scope

### In

- Date under analysis: `2026-10-24` (parameterized so later dates reuse the same code).
- Games in the notebook’s stated window: start times between 12:00 PM and 7:00 PM Pacific (final scheduled game that day per the notebook header).
- Scrapes every ~15 minutes from local JSON under `data/props/<book>/nba/`.
- Seven books: FanDuel, DraftKings, Pinnacle, Novig, ProphetX, Underdog, PrizePicks.
- All player props including alternates.
- Cross-book matching requires a matchable **player + event + prop + period + side + line**.
- Leader/follower: primary stats use a **30-minute** lag window; also report uncapped “next mover” diagnostics.
- Artifacts saved under `notebooks/nba/line_movement_analysis/reports/<date>/`.

### Out

- Team markets (moneylines, spreads, totals) — prop JSON only.
- DB upserts / change-filter integration with `src/odds/load_snapshots.py`.
- Live alerting or betting execution.
- Inferring PrizePicks sides when the payload has no over/under choice (those rows stay in load/validate + DQ; they drop out of match and leader/follower).

## Files

| Path | Owns |
|---|---|
| `notebooks/nba/line_movement_analysis/src/line_movement/__init__.py` | Public exports used by the notebook |
| `.../load.py` | Discover snapshot files; flatten book schemas; parse timestamps; return per-book and concatenated frames |
| `.../validate.py` | Timestamp checks, duplicates, missing fields, schema diffs; emit quality report + filtered frame |
| `.../normalize.py` | Canonical player names, prop types, periods, sides, event keys |
| `.../match.py` | Build market groups; validate cross-book coverage |
| `.../detect.py` | Per book×market previous-snapshot diffs → movement log |
| `.../leaders.py` | Episodes, first movers, windowed + uncapped followers, pairwise stats |
| `.../report.py` | Tables/chart helpers; write report artifacts for the date |
| `notebooks/nba/line_movement_analysis/line_movement_exploration_2026-10-24.ipynb` | Thin orchestration: config, section markdown 01–06, display, save |
| `notebooks/nba/line_movement_analysis/reports/<date>/` | Generated outputs (created at runtime; not hand-authored) |
| `tests/line_movement/` | Unit tests for normalize keys, movement detection, follower lag logic |

Top-level `src/odds/` remains the DB ingest path. Analysis may reuse concepts (identity vs compare columns) but must not import or modify upsert pipelines for this feature.

## Notebook shape

| Section | Title | Notebook does | Package does | Output object |
|---|---|---|---|---|
| 01 | Research question / scope | Keep existing header | — | — |
| 02 | Load and validate data | Call load + validate; show summary + DQ | Flatten seven schemas; quality checks | One normalized dataset (`quotes`) |
| 03 | Match equivalent markets | Show group coverage | Canonicalize + group | Comparable market groups |
| 04 | Detect line and price changes | Preview movement sample | Diff vs previous snapshot | Movement log |
| 05 | Leader/follower relationships | Show episode and pairwise tables | Match movements across books | Episodes + pairwise stats |
| 06 | Daily report | Render charts/tables; note DQ flags | Persist artifacts | Files under `reports/<date>/` |

Notebook cells stay thin: imports, `DATE` / path / window constants, function calls, `.head()` / plots. Business logic stays in the package.

Import approach: add `notebooks/nba/line_movement_analysis/src` to `sys.path` (or install-editable later). Prefer a single helper in the first code cell so path setup is obvious.

## Normalized quote schema

One row = one book × market side × snapshot.

| Column | Meaning |
|---|---|
| `book` | Provider id (`fanduel`, `draftkings`, `pinnacle`, `novig`, `prophetx`, `underdog`, `prizepicks`) |
| `snapshot_at` | Pacific timestamp from filename stamp (canonical scrape clock) |
| `fetched_at` | Payload fetch time (UTC-aware); not used as primary ordering |
| `source_file` | Basename of JSON file |
| `player` | Raw player string from payload |
| `player_norm` | Canonical player key for matching |
| `event_name` | Raw event / game label when present |
| `event_key` | Canonical event key for matching (normalized team pair or equivalent) |
| `prop` | Canonical prop / stat type |
| `period` | Canonical period; default `game` when absent |
| `side` | `over` / `under` (nullable when book has no side) |
| `line` | Numeric line |
| `american_price` | American odds when present |
| `decimal_price` | Optional |
| `implied_probability` | Optional |
| `is_main` | Main vs alternate when book exposes it; else null |
| `start_time` | Event start (UTC-aware) when present |
| `line_updated_at` | Book-reported update time when present |

Load preserves book-specific extras only if needed for DQ; analysis columns above are the contract for later steps.

## Matching rules (03)

**Market group key:**  
`(player_norm, event_key, prop, period, side, line)`

- Alternates are included: different lines are different groups.
- Rows missing any of `player_norm`, `event_key`, `prop`, `period`, `side`, or `line` are excluded from cross-book groups and counted in the DQ report.
- Groups with fewer than two books remain available for within-book movement detection but are excluded from pairwise leader/follower stats.
- Event matching uses normalized team names / event labels; ambiguous multi-matches are flagged in DQ rather than silently merged.

## Movement detection (04)

For each `(book, market_group_key)` sorted by `snapshot_at`:

- Compare current `line` and `american_price` to the previous available observation for that book and market.
- Emit a movement row when either value changes (null↔value counts as change; equal floats within absolute tolerance `1e-9` do not).
- Record: previous/new line and price, deltas, direction flags, `snapshot_at`, `prev_snapshot_at`, lag between snapshots, `change_type` ∈ {`line`, `price`, `both`}.

**Two series (both written into the movement log):**

1. **Price series** — key `(book, player_norm, event_key, prop, period, side, line)`. Detects american_price changes on a fixed line (alternates stay separate).
2. **Line continuity series** — key `(book, player_norm, event_key, prop, period, side)` (line excluded). When the quoted line for that side changes between snapshots, emit a line move `prev_line` → `new_line`. If multiple lines exist for the same continuity key in one snapshot (true alternates), use the book’s main line when `is_main` is available; otherwise use the line with the most recent `line_updated_at`, else the median offered line, and flag `line_choice=heuristic` in DQ.

**Direction flags (explicit):**

- `line_up` / `line_down`: new line − previous line > 0 / < 0.
- `price_shortened` / `price_lengthened`: american price moved toward / away from the bettor’s payout being worse for the same side — i.e. american_price decreased / increased (e.g. −110 → −120 is shortened; +100 → +110 is lengthened). Null price skips price direction.

## Leader / follower (05)

**Continuity key (cross-book):** `(player_norm, event_key, prop, period, side)`.

**Episode construction (deterministic):**

1. Collect all movements for a continuity key across books, sorted by `snapshot_at` then `book`.
2. Start a new episode when a movement occurs and no open episode for that key has a leader `snapshot_at` within the last N minutes (N = 30).
3. The earliest movement in that new cluster is the **leader** (`first_book`, `first_snapshot_at`, direction).
4. Later movements on the same continuity key with the **same direction** are **followers** of that episode if they belong to a different book.

**Same-direction** means: for line moves, same `line_up` or `line_down`; for price moves, same `price_shortened` or `price_lengthened`; mixed `both` must match on the components that both leader and follower share.

Two parallel views:

1. **Windowed (primary, N = 30 minutes):** follows with lag ≤ 30 minutes. Used for daily report headline tables.
2. **Uncapped diagnostic:** first later same-direction move by each other book, any lag. Reported separately so long lags are visible without polluting primary stats.

**Pairwise statistics** (ordered pair leader → follower):

- lead count, follow count (windowed)
- median / mean lag among windowed follows
- follow rate within window among opportunities
- uncapped next-mover median lag (diagnostic)

Default N = 30 minutes is a named constant; notebook may override for sensitivity checks without changing package defaults permanently.

## Daily report (06)

Artifacts under `notebooks/nba/line_movement_analysis/reports/<date>/`:

- `quotes_normalized.parquet` (or csv if parquet deps awkward in env)
- `movements.csv`
- `episodes.csv`
- `pairwise_stats.csv`
- `quality_report.json` (or csv summary)
- Figures: coverage by book, movement counts over time, pairwise lag heatmap or table image

Notebook section 06 renders the same tables/charts and lists DQ flags (missing side share by book, unmatched events, duplicate snapshot keys, books with snapshot gaps).

## Data quality checks (02)

- File discovery per book for `DATE`; empty book fails loudly.
- `snapshot_at` parseable from filename; monotonic coverage summary.
- Duplicate rows within `(book, snapshot_at, market identity)`.
- Missing rates for player, prop, side, line, american_price by book.
- Schema differences: which books lack side, event, is_main, prices.
- Start-time filter applied after parse; rows outside 12:00–19:00 Pacific game start are excluded from analysis frames (still counted in a “dropped by schedule filter” DQ metric).

## Testing

Minimum before claiming done:

- Normalize: player/stat/side canonicalization cases (including accents / Jr suffixes if implemented).
- Detect: price-only change, line change on continuity key, no false positive on identical consecutive snapshots.
- Leaders: follower inside 30 minutes counted; outside window excluded from primary but present in uncapped; pairwise lag correct on a tiny synthetic timeline.

Prefer pure functions over notebook cells for anything asserted in tests.

## Success criteria

1. Running the notebook for `2026-10-24` produces one normalized quotes frame and a non-empty movement log from real local JSON.
2. Market groups exist across at least the sportsbooks that expose side + line + player + event.
3. Pairwise windowed and uncapped tables are generated and saved under `reports/2026-10-24/`.
4. DQ report flags PrizePicks (and any other) side-missing rates instead of silently inventing sides.
5. Unit tests for normalize / detect / leaders pass.

## Non-goals / explicit non-decisions

- No claim that first-observed scrape implies true market leadership (15-minute resolution is a limit; report should state this).
- No production scheduler; daily report is notebook-driven.
- No merge of this package into top-level `src/` in this slice.
