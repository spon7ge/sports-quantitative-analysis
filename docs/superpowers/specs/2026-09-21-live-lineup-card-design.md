# Live lineup card notebook (NoVig JSON)

Approved: two-phase notebook `notebooks/mlb/live_lineup_card.ipynb`. Quotes from local `data/odds/*_mlb_*_props.json` (no Supabase). Pitcher strikeouts only. Snapshot today’s nines, ingest trailing-365 PBP once, freeze slot K/PA, sketch Log5 after a 100% `expected_bf` mixture placeholder, and score `nb_k_v1` as it exists (lineups not wired).

## Phase 1 (network, skip if cached)

- Schedule `startDate`/`endDate` = today−365d → today. `ingest_play_by_play` for `game_pk`s missing from `batter_pas` (gzip cache reused with `http=None`).
- Today’s slate: `ingest_lineup_slots(..., provenance="live_feed")` after PBP.
- `PBP_LIMIT` optional cap for a smoke run. Default `None` (full 365d).

## Phase 2 (local)

- Glob odds files. Keep `stat == "strikeouts"`. Book = JSON `source`.
- Match `"Away @ Home"` + date to Stats API team names; pitcher `player` via `normalize_player_name` / probable pitcher.
- `select_lineup_slots` at `start − forecast_horizon_hours`.
- Log5: \(p_i=\mathrm{Log5}(K/\mathrm{BF}_\text{pitcher}, K/\mathrm{PA}_{\text{slot }i}, \text{league }K/\mathrm{PA})\). Walk the opposing nine for `round(expected_bf)` BF (cycle after 9). Not Poisson-binomial. Not written into `STRIKEOUT_FEATURE_COLUMNS`.
- Predict with `artifacts/mlb/strikeouts/nb_k_v1.pkl` from gamelog features on synthetic today-rows + historical starter logs.

## Out of scope

Supabase. Wiring Log5 into `nb_k_v1`. Full-history 2018 backfill. Batter props (hits/SB/HR).
