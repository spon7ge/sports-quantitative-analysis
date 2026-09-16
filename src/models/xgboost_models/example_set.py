"""Price NBA_US player_points quotes as of the snapshot pull date."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.points.pregame import (
    appearances_before,
    blank_outcomes,
    build_pregame_points_features,
)
from src.pipeline.silver.positions import player_name_keys
from src.models.odds import american_to_decimal
from src.models.xgboost_models.artifact_bundle import (
    load_joint_points_bundle,
)
from src.models.xgboost_models.joint_pricing import (
    CanonicalPlayerPointsMarket,
    price_joint_points_market,
)
from src.models.xgboost_models.joint_simulation import JointPointsSimulator

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SET_PATH = (
    ROOT / "data" / "odds" / "NBA_US_20260210_144744.csv"
)
AS_OF = pd.Timestamp("2026-02-10")
PLAYER_POINTS = "player_points"
SILVER_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)
SCHEDULE_COLUMNS = (
    "game_date",
    "game_id",
    "team_id",
    "team_abbreviation",
    "opp_team_id",
    "matchup",
    "season_type",
    "season_year",
    "game_total",
    "team_spread",
    "player_team_spread",
)


def load_player_points_quotes(
    path: str | Path = EXAMPLE_SET_PATH,
) -> pd.DataFrame:
    quotes = pd.read_csv(path)
    points = quotes.loc[
        quotes["CATEGORY"].astype("string").eq(PLAYER_POINTS)
    ].copy()
    if points.empty:
        raise ValueError("quote file has no player_points rows")
    return points


def snapshot_as_of(quotes: pd.DataFrame) -> pd.Timestamp:
    """Pull-date cutoff. Same-day and earlier box scores are hidden."""
    pulled = pd.to_datetime(quotes["DATA_PULLED_AT"].iloc[0])
    return pd.Timestamp(pulled.normalize().date())


def filter_commence_after_as_of(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    column: str | None = None,
) -> pd.DataFrame:
    """Keep rows whose game starts after the snapshot date.

    Does not display the pull date or earlier slates.
    """
    if frame.empty:
        return frame.copy()
    name = column
    if name is None:
        if "commence_time" in frame.columns:
            name = "commence_time"
        elif "COMMENCE_TIME" in frame.columns:
            name = "COMMENCE_TIME"
        else:
            raise KeyError("frame has no commence_time column")
    commence = pd.to_datetime(frame[name], errors="coerce").dt.normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    return frame.loc[commence > cutoff].copy()


def settle_priced_player_points(
    priced: pd.DataFrame,
    quotes: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Grade selected sides on local games strictly after ``as_of``.

    Same-day slates are not displayed, including quotes whose UTC
    commence date is the next day. DNPs void.
    """
    cutoff = pd.Timestamp(as_of).normalize()
    history = appearances_before(panel, cutoff)
    candidates = build_candidate_rows(
        quotes,
        history,
        schedule=panel,
        as_of=cutoff,
    )
    if candidates.empty or priced.empty:
        return priced.iloc[0:0].copy()
    candidates = candidates.copy()
    candidates["game_date"] = pd.to_datetime(
        candidates["game_date"], errors="coerce"
    ).dt.normalize()
    candidates["player_id"] = pd.to_numeric(
        candidates["player_id"], errors="coerce"
    )
    resolved = candidates.loc[
        candidates["game_date"].gt(cutoff)
        & ~candidates["game_id"].astype("string").str.startswith("pregame-")
    ].copy()
    if resolved.empty:
        return priced.iloc[0:0].copy()
    resolved["commence_date"] = pd.to_datetime(
        resolved["commence_time"], errors="coerce"
    ).dt.normalize()
    resolved = resolved.loc[
        :,
        ["player_id", "game_date", "game_id", "commence_date"],
    ].drop_duplicates(["player_id", "commence_date"])

    shown = priced.copy()
    shown["player_id"] = pd.to_numeric(shown["player_id"], errors="coerce")
    shown["book"] = shown["book"].astype("string")
    shown["player_name"] = shown["player_name"].astype("string")
    shown["selected_side"] = (
        shown["selected_side"].astype("string").str.strip().str.lower()
    )
    shown["line"] = pd.to_numeric(shown["line"], errors="coerce")
    shown["commence_date"] = pd.to_datetime(
        shown["commence_time"], errors="coerce"
    ).dt.normalize()
    shown = shown.merge(
        resolved,
        how="inner",
        on=["player_id", "commence_date"],
    )
    if shown.empty:
        return shown

    odds = _quote_side_odds(quotes)
    keyed = shown.merge(
        odds,
        how="left",
        left_on=["book", "player_name", "line", "commence_date", "selected_side"],
        right_on=["book", "player_name", "line", "commence_date", "side"],
    )
    actuals = panel.copy()
    actuals["game_date"] = pd.to_datetime(
        actuals["game_date"], errors="coerce"
    ).dt.normalize()
    actuals["player_id"] = pd.to_numeric(actuals["player_id"], errors="coerce")
    merged = keyed.merge(
        actuals.loc[:, ["player_id", "game_date", "pts", "minutes"]],
        how="left",
        on=["player_id", "game_date"],
    )
    merged = merged.loc[merged["game_date"].gt(cutoff)].copy()
    return _grade_rows(merged)


def _quote_side_odds(quotes: pd.DataFrame) -> pd.DataFrame:
    sides = quotes.copy()
    sides["side"] = (
        sides["OVER/UNDER"].astype("string").str.strip().str.lower()
    )
    sides["book"] = sides["BOOKMAKER"].astype("string")
    sides["player_name"] = sides["NAME"].astype("string")
    sides["line"] = pd.to_numeric(sides["LINE"], errors="coerce")
    sides["commence_date"] = pd.to_datetime(
        sides["COMMENCE_TIME"], errors="coerce"
    ).dt.normalize()
    sides["american_odds"] = pd.to_numeric(sides["ODDS"], errors="coerce")
    return sides.loc[
        :,
        [
            "book",
            "player_name",
            "line",
            "commence_date",
            "side",
            "american_odds",
        ],
    ]


def _grade_rows(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = pd.to_numeric(frame.get("minutes"), errors="coerce")
    pts = pd.to_numeric(frame.get("pts"), errors="coerce")
    line = pd.to_numeric(frame["line"], errors="coerce")
    side = frame["selected_side"].astype("string").str.strip().str.lower()
    odds = pd.to_numeric(frame.get("american_odds"), errors="coerce")
    played = minutes.gt(0) & pts.notna()
    has_side = side.isin(["over", "under"])
    over_hit = pts > line
    under_hit = pts < line
    hit = (side.eq("over") & over_hit) | (side.eq("under") & under_hit)
    miss = (side.eq("over") & under_hit) | (side.eq("under") & over_hit)
    grade = np.full(len(frame), "no_bet", dtype=object)
    grade = np.where(~played, "void", grade)
    grade = np.where(played & ~has_side, "no_bet", grade)
    grade = np.where(played & has_side & hit, "win", grade)
    grade = np.where(played & has_side & miss, "loss", grade)
    grade = np.where(
        played & has_side & ~hit & ~miss,
        "push",
        grade,
    )
    profit = np.full(len(frame), np.nan)
    for index, (result, american) in enumerate(zip(grade, odds)):
        if result in {"void", "push"}:
            profit[index] = 0.0
        elif result == "no_bet":
            profit[index] = np.nan
        elif result == "win" and np.isfinite(american):
            profit[index] = american_to_decimal(float(american)) - 1.0
        elif result == "loss":
            profit[index] = -1.0
    out = frame.copy()
    out["grade"] = grade
    out["profit"] = profit
    return out


def load_silver_panel(*, root: Path = ROOT) -> pd.DataFrame:
    frames = []
    for season in SILVER_SEASONS:
        path = (
            root
            / "data"
            / "silver"
            / "nba"
            / season
            / "regular_season"
            / "player_gamelogs.parquet"
        )
        frames.append(pd.read_parquet(path))
    panel = pd.concat(frames, ignore_index=True)
    panel["game_date"] = pd.to_datetime(
        panel["game_date"], errors="coerce"
    )
    return panel


def load_silver_history(
    *,
    as_of: str | pd.Timestamp = AS_OF,
    root: Path = ROOT,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = load_silver_panel(root=root) if panel is None else panel
    return appearances_before(source, as_of)


def map_quote_names(
    names: pd.Series,
    history: pd.DataFrame,
) -> dict[str, int]:
    latest = (
        history.sort_values("game_date")
        .drop_duplicates("player_id", keep="last")
    )
    key_to_ids: dict[str, set[int]] = defaultdict(set)
    for _, row in latest.iterrows():
        for key in player_name_keys(row["player_name"]):
            key_to_ids[key].add(int(row["player_id"]))

    mapping: dict[str, int] = {}
    for name in names.dropna().unique():
        matched: set[int] = set()
        for key in player_name_keys(name):
            matched |= key_to_ids.get(key, set())
        if len(matched) == 1:
            mapping[str(name)] = next(iter(matched))
    return mapping


def last_team_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "player_id",
            "player_name",
            "team_id",
            "team_abbreviation",
            "team_name",
        )
        if column in history.columns
    ]
    return (
        history.sort_values("game_date")
        .drop_duplicates("player_id", keep="last")
        [columns]
        .set_index("player_id")
    )


def team_schedule(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column for column in SCHEDULE_COLUMNS if column in panel.columns
    ]
    schedule = panel[columns].copy()
    schedule["game_date"] = pd.to_datetime(
        schedule["game_date"], errors="coerce"
    ).dt.normalize()
    return (
        schedule.drop_duplicates(["team_id", "game_date", "game_id"])
        .sort_values(["game_date", "team_id"])
        .reset_index(drop=True)
    )


def lookup_team_game(
    schedule: pd.DataFrame,
    *,
    team_id: object,
    game_date: pd.Timestamp,
    nearby_dates: list[pd.Timestamp],
) -> pd.Series | None:
    days = [pd.Timestamp(game_date).normalize(), *nearby_dates]
    seen: set[pd.Timestamp] = set()
    for day in days:
        day = pd.Timestamp(day).normalize()
        if day in seen:
            continue
        seen.add(day)
        hit = schedule.loc[
            schedule["team_id"].eq(team_id)
            & schedule["game_date"].eq(day)
        ]
        if not hit.empty:
            return hit.iloc[0]
    return None


def build_candidate_rows(
    quotes: pd.DataFrame,
    history: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp = AS_OF,
) -> pd.DataFrame:
    names = map_quote_names(quotes["NAME"], history)
    snapshot = last_team_snapshot(history)
    games = team_schedule(schedule if schedule is not None else history)
    unique = quotes.drop_duplicates(["NAME", "COMMENCE_TIME"]).copy()
    unique["player_id"] = unique["NAME"].map(names)
    unique = unique.loc[unique["player_id"].notna()].copy()
    unique["player_id"] = unique["player_id"].astype(int)
    unique["game_date"] = pd.to_datetime(unique["COMMENCE_TIME"])
    nearby = [
        pd.Timestamp(value).normalize()
        for value in unique["game_date"].dropna().unique()
    ]

    rows = []
    for record in unique.to_dict("records"):
        player_id = int(record["player_id"])
        if player_id not in snapshot.index:
            continue
        last = snapshot.loc[player_id]
        game_date = pd.Timestamp(record["game_date"]).normalize()
        context = lookup_team_game(
            games,
            team_id=last["team_id"],
            game_date=game_date,
            nearby_dates=nearby,
        )
        template = history.loc[
            history["player_id"].eq(player_id)
        ].sort_values("game_date").iloc[-1]
        candidate = blank_outcomes(template)
        candidate["player_id"] = player_id
        candidate["player_name"] = record["NAME"]
        candidate["commence_time"] = record["COMMENCE_TIME"]
        candidate["team_id"] = last["team_id"]
        candidate["team_abbreviation"] = last["team_abbreviation"]
        candidate["game_date"] = (
            pd.Timestamp(context["game_date"])
            if context is not None
            else game_date
        )
        candidate["season_year"] = "2025-26"
        candidate["season_type"] = "Regular Season"
        candidate["minutes"] = float("nan")
        if context is not None:
            candidate["game_id"] = context["game_id"]
            candidate["opp_team_id"] = context["opp_team_id"]
            candidate["matchup"] = context["matchup"]
            candidate["season_type"] = context.get(
                "season_type", "Regular Season"
            )
            if "season_year" in context.index and pd.notna(
                context["season_year"]
            ):
                candidate["season_year"] = context["season_year"]
            for column in (
                "game_total",
                "team_spread",
                "player_team_spread",
            ):
                if column in context.index:
                    candidate[column] = context[column]
        else:
            candidate["game_id"] = _pregame_game_id(
                game_date,
                str(last["team_abbreviation"]),
                None,
            )
            candidate["opp_team_id"] = pd.NA
            candidate["matchup"] = str(last["team_abbreviation"])
        rows.append(candidate)
    return pd.DataFrame(rows)


def pair_player_points_markets(
    quotes: pd.DataFrame,
    name_to_id: dict[str, int],
) -> list[tuple[CanonicalPlayerPointsMarket, str]]:
    points = quotes.copy()
    points["player_id"] = points["NAME"].map(name_to_id)
    grouped = points.groupby(
        ["BOOKMAKER", "NAME", "LINE", "COMMENCE_TIME", "DATA_PULLED_AT"],
        dropna=False,
    )
    markets: list[tuple[CanonicalPlayerPointsMarket, str]] = []
    for (book, name, line, commence, pulled), group in grouped:
        player_id = name_to_id.get(str(name))
        if player_id is None:
            continue
        sides = {
            str(row["OVER/UNDER"]).strip().lower(): row
            for _, row in group.iterrows()
        }
        if set(sides) != {"over", "under"}:
            continue
        game_id = f"pregame-{pd.Timestamp(commence).date()}-{player_id}"
        market = CanonicalPlayerPointsMarket(
            book=str(book),
            event=str(commence),
            player_id=player_id,
            game_id=game_id,
            stat="PTS",
            period="full_game",
            line=float(line),
            over_odds=float(sides["over"]["ODDS"]),
            under_odds=float(sides["under"]["ODDS"]),
            quote_ts=str(pulled),
        )
        markets.append((market, str(name)))
    return markets


def price_example_set_player_points(
    *,
    quotes_path: str | Path = EXAMPLE_SET_PATH,
    as_of: str | pd.Timestamp = AS_OF,
    n_draws: int = 4_000,
    root: Path = ROOT,
) -> pd.DataFrame:
    quotes = load_player_points_quotes(quotes_path)
    as_of = pd.Timestamp(as_of)
    quotes = filter_commence_after_as_of(quotes, as_of)
    panel = load_silver_panel(root=root)
    history = appearances_before(panel, as_of)
    names = map_quote_names(quotes["NAME"], history)
    candidates = build_candidate_rows(
        quotes,
        history,
        schedule=panel,
        as_of=as_of,
    )
    featured = build_pregame_points_features(
        history,
        candidates,
        as_of=as_of,
    )
    featured = featured.drop_duplicates(
        ["player_id", "game_date"], keep="first"
    )
    bundle = load_joint_points_bundle(
        minutes_mean_path=root
        / "artifacts"
        / "models"
        / "minutes"
        / "xgboost_minutes.joblib",
        minutes_dist_path=root
        / "artifacts"
        / "models"
        / "minutes"
        / "xgboost_minutes_distribution.joblib",
        points_path=root
        / "artifacts"
        / "models"
        / "points"
        / "xgboost_points.joblib",
        joint_calibration_path=root
        / "artifacts"
        / "models"
        / "points"
        / "joint_calibration.joblib",
    )
    simulator = JointPointsSimulator(bundle, n_draws=n_draws)
    clouds = {}
    for _, row in featured.iterrows():
        key = (
            int(row["player_id"]),
            pd.Timestamp(row["game_date"]).date(),
        )
        clouds[key] = simulator.simulate(row)

    rows = []
    for market, player_name in pair_player_points_markets(quotes, names):
        cloud = _cloud_for_market(clouds, market)
        if cloud is None:
            priced = {
                "status": "SKIP_NO_FEATURE",
                "player_name": player_name,
                "player_id": market.player_id,
                "book": market.book,
                "line": market.line,
                "commence_time": market.event,
                "reason": "no pregame feature row",
                "conditional_on_appearance": True,
            }
        else:
            priced = price_joint_points_market(cloud, market)
            priced.update(
                {
                    "player_name": player_name,
                    "player_id": market.player_id,
                    "book": market.book,
                    "commence_time": market.event,
                    "quote_ts": market.quote_ts,
                    "hat_m": cloud.hat_m,
                    "expected_minutes": cloud.expected_minutes,
                    "hat_p": cloud.hat_p,
                    "g": cloud.g,
                    "mu": cloud.mu,
                    "feature_as_of": str(pd.Timestamp(as_of).date()),
                    "season_type": "Regular Season",
                }
            )
        rows.append(_flatten_quote(priced))
    result = pd.DataFrame(rows)
    if not result.empty and "over_ev" in result:
        result = result.sort_values(
            "over_ev",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    return result


def _cloud_for_market(clouds: dict, market: CanonicalPlayerPointsMarket):
    player_id = int(market.player_id)
    commence = pd.Timestamp(market.event).date()
    direct = clouds.get((player_id, commence))
    if direct is not None:
        return direct
    for (pid, game_date), cloud in clouds.items():
        if pid == player_id and abs((game_date - commence).days) <= 1:
            return cloud
    return None


def _pregame_game_id(
    commence: object,
    team: str,
    opponent: str | None,
) -> str:
    date = pd.Timestamp(commence).strftime("%Y%m%d")
    sides = "-".join(sorted([team, opponent or "UNK"]))
    return f"pregame-{date}-{sides}"


def _flatten_quote(priced: dict) -> dict:
    flat = {
        key: value
        for key, value in priced.items()
        if key not in {"over", "under"}
    }
    for side in ("over", "under"):
        payload = priced.get(side) or {}
        if isinstance(payload, dict):
            for field, value in payload.items():
                flat[f"{side}_{field}"] = value
    return flat


def main() -> None:
    import sys

    quotes_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE_SET_PATH
    )
    if not quotes_path.is_absolute():
        quotes_path = ROOT / quotes_path
    quotes = load_player_points_quotes(quotes_path)
    pulled = pd.to_datetime(quotes["DATA_PULLED_AT"].iloc[0])
    as_of = pd.Timestamp(pulled.normalize().date())
    stamp = as_of.strftime("%Y%m%d")
    output = (
        ROOT
        / "artifacts"
        / "pricing"
        / f"nba_us_{stamp}_player_points.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    priced = price_example_set_player_points(
        quotes_path=quotes_path,
        as_of=as_of,
    )
    priced.to_csv(output, index=False)
    print(f"Wrote {output} ({len(priced)} player_points markets)")
    if "status" in priced:
        print(priced["status"].value_counts().to_string())
    if {"player_name", "book", "line", "selected_side", "status"}.issubset(
        priced.columns
    ):
        preview_cols = [
            column
            for column in (
                "player_name",
                "book",
                "line",
                "selected_side",
                "hat_p",
                "over_model_probability",
                "over_ev",
                "under_ev",
            )
            if column in priced.columns
        ]
        preview = priced.loc[
            priced["status"].eq("PRICED"),
            preview_cols,
        ]
        preview = filter_commence_after_as_of(preview, as_of)
        preview = preview.head(15)
        if not preview.empty:
            print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
