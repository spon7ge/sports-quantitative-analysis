-- Latest-quote indexes for odds change-filter LATERAL lookups.
-- Shape: (league, <complete identity cols>, scraped_at DESC)
--
-- Run outside a transaction (CONCURRENTLY). On Supabase / Postgres:
--   psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f sql/migrations/20261023_odds_latest_quote_indexes.sql
--
-- Verify a plan with realistic params, e.g.:
--   EXPLAIN (ANALYZE, BUFFERS)
--   SELECT * FROM odds.nba_novig t
--   WHERE t.league = 'nba'
--     AND t.event_id IS NOT DISTINCT FROM '...'
--     AND t.player_name IS NOT DISTINCT FROM '...'
--     AND t.stat_name IS NOT DISTINCT FROM '...'
--     AND t.side IS NOT DISTINCT FROM 'over'
--   ORDER BY t.scraped_at DESC
--   LIMIT 1;

-- PrizePicks (identity includes odds_type)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_prizepicks_latest_quote
  ON odds.nba_prizepicks (league, player_name, stat_type, odds_type, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_prizepicks_latest_quote
  ON odds.mlb_prizepicks (league, player_name, stat_type, odds_type, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_prizepicks_latest_quote
  ON odds.wnba_prizepicks (league, player_name, stat_type, odds_type, scraped_at DESC);

-- Underdog
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_underdogs_latest_quote
  ON odds.nba_underdogs (league, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_underdogs_latest_quote
  ON odds.mlb_underdogs (league, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_underdogs_latest_quote
  ON odds.wnba_underdogs (league, player_name, stat_name, side, scraped_at DESC);

-- Sharp / book props (FanDuel, DraftKings, Pinnacle props)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_fanduel_latest_quote
  ON odds.nba_fanduel (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_fanduel_latest_quote
  ON odds.mlb_fanduel (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_fanduel_latest_quote
  ON odds.wnba_fanduel (league, player_name, market_type, side, scraped_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_draftkings_latest_quote
  ON odds.nba_draftkings (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_draftkings_latest_quote
  ON odds.mlb_draftkings (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_draftkings_latest_quote
  ON odds.wnba_draftkings (league, player_name, market_type, side, scraped_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_pinnacle_latest_quote
  ON odds.nba_pinnacle (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_pinnacle_latest_quote
  ON odds.mlb_pinnacle (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_pinnacle_latest_quote
  ON odds.wnba_pinnacle (league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_pinnacle_latest_quote
  ON odds.nfl_pinnacle (league, player_name, market_type, side, scraped_at DESC);

-- Parlay API odds
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_parlay_api_odds_latest_quote
  ON odds.wnba_parlay_api_odds (sportsbook, league, player_name, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_parlay_api_odds_latest_quote
  ON odds.mlb_parlay_api_odds (sportsbook, league, player_name, market_type, side, scraped_at DESC);

-- Pinnacle team
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_pinnacle_team_latest_quote
  ON odds.nba_pinnacle_team (
    league, away_team, home_team, market_type, period, is_alternate, side, scraped_at DESC
  );
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_pinnacle_team_latest_quote
  ON odds.mlb_pinnacle_team (
    league, away_team, home_team, market_type, period, is_alternate, side, scraped_at DESC
  );
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_pinnacle_team_latest_quote
  ON odds.wnba_pinnacle_team (
    league, away_team, home_team, market_type, period, is_alternate, side, scraped_at DESC
  );
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_pinnacle_team_latest_quote
  ON odds.nfl_pinnacle_team (
    league, away_team, home_team, market_type, period, is_alternate, side, scraped_at DESC
  );

-- ProphetX props + team
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_prophetx_latest_quote
  ON odds.nba_prophetx (league, event_id, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_prophetx_latest_quote
  ON odds.mlb_prophetx (league, event_id, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_prophetx_latest_quote
  ON odds.wnba_prophetx (league, event_id, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_prophetx_latest_quote
  ON odds.nfl_prophetx (league, event_id, player_name, stat_name, side, scraped_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_prophetx_team_latest_quote
  ON odds.nba_prophetx_team (league, event_id, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_prophetx_team_latest_quote
  ON odds.mlb_prophetx_team (league, event_id, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_prophetx_team_latest_quote
  ON odds.wnba_prophetx_team (league, event_id, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_prophetx_team_latest_quote
  ON odds.nfl_prophetx_team (league, event_id, market_type, side, scraped_at DESC);

-- Novig props + team
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_novig_latest_quote
  ON odds.nba_novig (league, event_id, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_novig_latest_quote
  ON odds.mlb_novig (league, event_id, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_novig_latest_quote
  ON odds.wnba_novig (league, event_id, player_name, stat_name, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_novig_latest_quote
  ON odds.nfl_novig (league, event_id, player_name, stat_name, side, scraped_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_novig_team_latest_quote
  ON odds.nba_novig_team (league, event_id, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_novig_team_latest_quote
  ON odds.mlb_novig_team (league, event_id, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wnba_novig_team_latest_quote
  ON odds.wnba_novig_team (league, event_id, market_type, side, scraped_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_novig_team_latest_quote
  ON odds.nfl_novig_team (league, event_id, market_type, side, scraped_at DESC);
