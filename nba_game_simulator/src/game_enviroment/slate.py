from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from game_enviroment.features import _finite


def _line_points(blocks, team_can, side, participants):
    from src.scrapers.arbitrage import canonical_team

    for block in blocks:
        if (
            block.get("period") not in (0, None)
            or block.get("is_alternate")
        ):
            continue

        for line in block.get("lines") or []:
            if side is not None and line.get("side") != side:
                continue

            label = line.get("team")
            if (
                not label
                and len(participants) >= 2
                and line.get("side") in ("away", "home")
            ):
                label = participants[int(line["side"] == "home")]

            if (
                team_can is not None
                and canonical_team(label, "nba") != team_can
            ):
                continue
            if _finite(line.get("points")):
                return float(line["points"])
    return None


def _roster_abbr(roster, team_can):
    from src.pipeline.silver.rotowire import normalize_team_abbreviation
    from src.scrapers.arbitrage import canonical_team

    for abbr in roster.team_abbreviation.dropna().unique():
        normalized = normalize_team_abbreviation(abbr)
        if canonical_team(normalized, "nba") == team_can:
            return normalized
    return None


def _match_roster_ids(names, abbr, roster):
    from src.pipeline.silver.positions import player_name_keys
    from src.pipeline.silver.rotowire import normalize_team_abbreviation

    rows = roster[
        roster.team_abbreviation.map(
            normalize_team_abbreviation
        ).eq(abbr)
    ]
    ids, missing = [], []

    for name in names:
        wanted = set(player_name_keys(name))
        matches = rows[
            rows.player_name.map(
                lambda x: bool(wanted & set(player_name_keys(x)))
            )
        ]
        candidates = matches.player_id.dropna().unique()

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous roster match for {name!r} on {abbr}"
            )
        if not len(candidates):
            missing.append(str(name))
        else:
            found = candidates[0]
            ids.append(
                int(found)
                if _finite(found) and float(found).is_integer()
                else found
            )

    return ids, missing


def _starter_ids(lineups, abbr, roster):
    names = list(lineups.get(abbr) or [])
    if not names:
        return None

    ids, missing = _match_roster_ids(names, abbr, roster)
    if missing:
        raise ValueError(f"Roster has no ID for {missing} on {abbr}")
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError(f"Expected five distinct starters for {abbr}")
    return ids


def player_slate(
    games, lineups, roster, team, unavailable=None, probable=None,
):
    from src.pipeline.silver.rotowire import normalize_team_abbreviation
    from src.scrapers.arbitrage import canonical_team

    missing = {
        "player_name", "player_id", "team_abbreviation"
    } - set(roster.columns)
    if missing:
        raise ValueError(f"Roster needs {sorted(missing)}")

    abbr = normalize_team_abbreviation(team)
    team_can = canonical_team(abbr, "nba")
    if team_can is None:
        raise ValueError(f"Unknown team {team!r}")

    for game in games:
        participants = game.get("participants") or []
        if len(participants) != 2:
            continue

        labels = [
            canonical_team(part, "nba")
            for part in participants
        ]
        if team_can not in labels:
            continue

        markets = game.get("team_markets") or {}
        spread = _line_points(
            markets.get("spread") or [],
            team_can, None, participants,
        )
        total = _line_points(
            markets.get("total") or [],
            None, "over", participants,
        )
        if spread is None or total is None:
            raise ValueError(
                f"Pinnacle spread or total missing for {abbr}"
            )

        opponent = _roster_abbr(
            roster,
            labels[1] if labels[0] == team_can else labels[0],
        )
        if opponent is None:
            raise ValueError(f"Roster has no opponent for {abbr}")

        outs = unavailable or {}
        probs = probable or {}
        own_out, own_missing = _match_roster_ids(
            outs.get(abbr) or [], abbr, roster
        )
        opp_out, opp_missing = _match_roster_ids(
            outs.get(opponent) or [], opponent, roster
        )
        own_probable, _ = _match_roster_ids(
            probs.get(abbr) or [], abbr, roster
        )
        opp_probable, _ = _match_roster_ids(
            probs.get(opponent) or [], opponent, roster
        )
        return {
            "team": abbr,
            "opponent": opponent,
            "is_home": labels[1] == team_can,
            "spread": spread,
            "total": total,
            "team_lineup": _starter_ids(lineups, abbr, roster),
            "opponent_lineup": _starter_ids(
                lineups, opponent, roster
            ),
            "team_out": own_out,
            "opponent_out": opp_out,
            "team_out_absent": own_missing,
            "opponent_out_absent": opp_missing,
            "team_probable": own_probable,
            "opponent_probable": opp_probable,
        }

    raise ValueError(f"No Pinnacle game for {abbr}")


def load_live_slate(
    team, date, roster, pinnacle_dir=None, lineup_dir=None,
):
    from src.live.predict_points import (
        lineups_on,
        pinnacle_games_on,
        probable_on,
        unavailable_on,
    )

    slate = player_slate(
        pinnacle_games_on(date, directory=pinnacle_dir),
        lineups_on(date, directory=lineup_dir),
        roster,
        team,
        unavailable_on(date, directory=lineup_dir),
        probable_on(date, directory=lineup_dir),
    )
    slate["date"] = str(pd.Timestamp(date).date())
    return slate


def _newest_snapshot(folder, pattern):
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern} in {folder}"
        )
    return files[-1]


def _snapshot_date(path):
    match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if match is None:
        raise ValueError(f"No slate date in {path.name}")
    return match.group(0)


def load_latest_slate(
    team, roster, pinnacle_dir=None, lineup_dir=None,
):
    from src.live.predict_points import (
        _LINEUP_DIR,
        _PINNACLE_DIR,
        lineups_on,
        probable_on,
        unavailable_on,
    )

    odds = _newest_snapshot(
        _PINNACLE_DIR if pinnacle_dir is None else pinnacle_dir,
        "pinnacle_nba_*_team.json",
    )
    fives = _newest_snapshot(
        _LINEUP_DIR if lineup_dir is None else lineup_dir,
        "rotowire_nba_*.json",
    )
    date = _snapshot_date(odds)
    if _snapshot_date(fives) != date:
        raise ValueError(
            "Latest odds and lineups are for different dates; "
            "use load_live_slate."
        )

    games = json.loads(odds.read_text()).get("games") or []
    slate = player_slate(
        games,
        lineups_on(date, directory=lineup_dir),
        roster,
        team,
        unavailable_on(date, directory=lineup_dir),
        probable_on(date, directory=lineup_dir),
    )
    slate.update(
        date=date,
        pinnacle_file=str(odds),
        lineup_file=str(fives),
    )
    return slate
