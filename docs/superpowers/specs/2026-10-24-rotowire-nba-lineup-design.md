# Rotowire NBA lineup snapshot

Date: 2026-10-24
Status: approved
Approved decisions: one `may_not_play` list with the raw badge (option C); one Rotowire fetch per odds run, written as its own props file (option 1).

## Goal

When the NBA odds scrapers run, also save the current RotoWire starting lineups. Each save is one JSON file under `data/props/rotowire_nba_lineup/`, named like the other props files. The file records, per team, whether the lineup is confirmed or projected, the posted lineup, and the MAY NOT PLAY list with the badge RotoWire printed.

The live starter flag in `src/scrapers/nba/nba_lineups.py` and `src/live/nba_lineup.py` stays as it is. This snapshot does not write to Supabase.

## Scope

### In

- One NBA job on `run_all_odds`, name `nba_rotowire_lineup`, module `src.scrapers.nba.nba_rotowire_lineup`.
- In `_league_jobs("nba")` the job is the last entry before `nba_prizepick`. It uses the HTTP pool. `resolve_jobs()` still moves both PrizePicks jobs to the end, so on a full run this job is the last non-PrizePicks name, then `mlb_prizepick`, then `nba_prizepick`.
- A full run and `--league nba` include it once. `--league mlb` does not. `--only` and `--exclude` treat it like any other job name.
- One GET of `https://www.rotowire.com/basketball/nba-lineups.php` per invocation. No book scraper fetches this page.
- One new file per successful run. Older files stay on disk. `data/props/` is gitignored; nothing in this feature commits a snapshot.
- NBA cards only. WNBA cards on the same page are ignored.

### Out

- Changing `expected_starters_by_abbr`, `posted_starting_flags`, or the minutes/live starter flag.
- A database upsert, a second output format, or scraping lineups from inside FanDuel, DraftKings, Pinnacle, Novig, ProphetX, Underdog, or PrizePicks.
- Game time, referees, odds, or projected minutes from the lineup card.
- Normalizing badge text (`Out` stays `Out`).

## File

Directory: `data/props/rotowire_nba_lineup/`

Name: `rotowire_nba_YYYY-MM-DD_HHMMSS.json`

Clock: `America/Los_Angeles`, same as the FanDuel props filenames. Example: `rotowire_nba_2026-10-24_223512.json`.

`fetched_at` inside the file is that same instant as an ISO-8601 string with the Pacific offset, so the name and the payload refer to one time.

Write the complete JSON to `.<filename>.tmp` in that same directory, then rename it to the final name. A failed request writes no file. The temp name for the example above is `.rotowire_nba_2026-10-24_223512.json.tmp`.

## Payload

```json
{
  "source": "rotowire",
  "url": "https://www.rotowire.com/basketball/nba-lineups.php",
  "fetched_at": "2026-10-24T22:35:12-07:00",
  "games": [
    {
      "away": {
        "team": "MIA",
        "status": "confirmed",
        "lineup": [
          {"name": "Donovan Mitchell", "position": "PG"}
        ],
        "may_not_play": []
      },
      "home": {
        "team": "CHA",
        "status": "confirmed",
        "lineup": [
          {"name": "Dennis Schroder", "position": "PG"}
        ],
        "may_not_play": [
          {"name": "Kon Knueppel", "position": "F", "badge": "Out"}
        ]
      }
    }
  ]
}
```

Games follow page order. Each side is `away` then `home`.

`team` is the abbreviation from `.lineup__abbr`. If that node is missing, use the nickname map already in `nba_lineups.py`. If both are missing, `team` is `null` and the side is still written.

`status` is `confirmed` when that side's status chip has class `is-confirmed` or its text is `Confirmed Lineup`. Every other chip, including `Expected Lineup`, and a missing chip, is `projected`.

`lineup` is every `li.lineup__player` node before the first `li.lineup__title` on that side, in that order. There is no cap at five and no filter on the play-likelihood title. On the current page that block is the posted five. Each entry is `name` and `position`. `name` is the anchor `title`, then the anchor text. `position` is the `.lineup__pos` text, or `null` when absent. Non-player nodes (Projected Minutes, On/Off Court Stats) are not players.

`may_not_play` is the `li.lineup__player` nodes after that title, in that order, including players marked `hide` / `is-ofs`. Each entry is `name`, `position`, and `badge`. `badge` is the `.lineup__inj` text exactly as printed (`Out`, `Doubt`, `Ques`, `OFS`, or whatever else is there). A missing badge is `null`. A side whose MAY NOT PLAY block is empty or says none (`li.lineup__no` only) has `may_not_play: []`.

A player listed in the five and again under MAY NOT PLAY stays in both lists. A badge on a starter row is not copied into `may_not_play` unless that player is also listed after the title. Lineup entries do not carry a badge field.

An NBA page with zero lineup cards still writes the file with `games: []`.

## Code

| Path | Owns |
|---|---|
| `src/scrapers/nba/nba_lineups.py` | Add `lineup_snapshot(fetched_at: str) -> dict` on `NBADailyLineups`. It reads `self.soup` only. No extra HTTP. Existing starter methods keep their current results. |
| `src/scrapers/nba/nba_rotowire_lineup.py` | CLI. Build one `NBADailyLineups()` (one GET), call `lineup_snapshot`, write the file, exit 0. Exit 1 on fetch or write failure, with no snapshot file left behind. |
| `src/scrapers/run_all_odds.py` | Register the NBA job. |
| `tests/scrapers/test_nba_rotowire_lineup.py` | Snapshot shape from fixture HTML, filename, atomic write, runner registration. |
| `tests/scrapers/test_run_all_odds.py` | Expect `nba_rotowire_lineup` on NBA runs and not on MLB runs. |
| `tests/scrapers/test_nba_lineups.py` | Existing starter-flag tests still pass unchanged. |

`lineup_snapshot` is the only new behavior on `NBADailyLineups`. The CLI does not parse HTML itself.

## Errors

- Non-200 or network failure: the CLI exits 1 and writes nothing. Other book jobs keep running. `--fail-fast` still cancels in-flight jobs after any non-zero exit, including this one. That is the existing runner rule.
- Rename failure after a successful fetch: exit 1 and remove the temporary file if it is still present.
- This job's failure does not change another book's exit code unless `--fail-fast` is set.

## Tests

Fixture HTML, no network:

- Confirmed chip → `confirmed`. Expected chip, and a missing chip, → `projected`.
- Lineup order and positions match the nodes before the title, including a sixth player when the fixture lists one there. A player listed only after the title is not in `lineup`.
- `may_not_play` keeps badge text `Out`, `Doubt`, `Ques`, and `OFS`, including a hidden OFS row.
- A player in the five and under MAY NOT PLAY appears in both. A starter-only badge does not create a `may_not_play` row.
- Empty MAY NOT PLAY → `[]`. Zero NBA cards → `games: []`.
- WNBA card is ignored.
- Filename is `rotowire_nba_YYYY-MM-DD_HHMMSS.json` under `data/props/rotowire_nba_lineup/` for a fixed Pacific timestamp. `fetched_at` matches that timestamp.
- Writer renames a temp file into place and does not leave the temp file on success.
- `resolve_jobs()` includes `nba_rotowire_lineup` once for the default run and for `--league nba`, and omits it for `--league mlb`. Module path is `src.scrapers.nba.nba_rotowire_lineup`. On the default run it is the last name before `mlb_prizepick` and `nba_prizepick`.
- Existing `test_nba_lineups.py` expectations stay the same.
