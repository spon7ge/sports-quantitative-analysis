"""Layer 3 turns one team's roles into minutes inside the attached script.

Training rows shift the window so the game being scored stays out.
Live rows do not shift, so the latest completed game stays in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from role_context import SCRIPT
from game_rotations import (
    CONDITIONAL,
    RotationSimulator,
    allocate_rotation,
    build_rotation_segments,
    constrain,
    draw_minutes,
    fit_rotation,
    invalid_segment_count,
    live_minutes,
    scale_mu,
    segments_from_minutes,
    training_minutes,
)
from sklearn.preprocessing import StandardScaler


def _appearance(**overrides) -> dict:
    row = {
        "season_year": "2023-24",
        "game_id": "g1",
        "game_date": "2024-01-01",
        "team_id": 10,
        "team_abbreviation": "HOM",
        "opp_team_id": 20,
        "matchup": "HOM vs. AWY",
        "player_id": 1,
        "start_position": "G",
        "min": 10.0,
        "pf": 2.0,
        "pace": 100.0,
        "off_rating": 110.0,
        "def_rating": 108.0,
        "team_pts": 110.0,
        "opp_pts": 100.0,
    }
    row.update(overrides)
    return row


def _game(game_id: str, game_date: str, minutes: float, *, opp_minutes: float = 20.0) -> list[dict]:
    return [
        _appearance(game_id=game_id, game_date=game_date, min=minutes, player_id=1),
        _appearance(
            game_id=game_id,
            game_date=game_date,
            team_id=20,
            team_abbreviation="AWY",
            opp_team_id=10,
            matchup="AWY @ HOM",
            player_id=2,
            min=opp_minutes,
        ),
    ]


def _slate(game_date: str = "2024-02-01") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": game_date,
                "team_id": 10,
                "opp_team_id": 20,
                "is_home": 1,
                "is_starter": 1,
                "team_spread": -3.5,
            }
        ]
    )


def _environment(margins, overs, lengths) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_margin": margins,
            "n_ot": overs,
            "game_minutes": lengths,
        }
    )


def _roster(n: int, *, starter_count: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": np.arange(n),
            "team_id": np.ones(n, dtype=int),
            "is_starter": (np.arange(n) < starter_count).astype(float),
            "min_mean_10": np.where(np.arange(n) < starter_count, 36.0, 12.0),
        }
    )


class _ScriptMinutes:
    """Starters give minutes back when this team is blown out."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        margin = frame["team_final_margin"].to_numpy(dtype=float)
        self.seen.append(margin.copy())
        starter = frame["is_starter"].to_numpy(dtype=float)
        loss = np.clip(-margin, 0, None)
        base = np.where(starter == 1.0, 36.0, 12.0)
        return base - starter * 0.5 * loss


def test_training_minutes_shift_excludes_the_game_being_scored():
    logs = pd.DataFrame(_game("g1", "2024-01-01", 10.0) + _game("g2", "2024-01-03", 30.0))

    trained = training_minutes(logs)
    second = trained.loc[trained["game_id"].eq("g2") & trained["player_id"].eq(1)].iloc[0]

    assert second["min_mean_10"] == 10.0
    assert second["minutes"] == 30.0


def test_live_minutes_keep_the_latest_completed_game():
    games = []
    for index in range(11):
        day = index + 1
        games.extend(_game(f"g{index}", f"2024-01-{day:02d}", 10.0))
    games.extend(_game("g11", "2024-01-12", 50.0))
    logs = pd.DataFrame(games)

    live = live_minutes(logs, _slate("2024-01-20")).iloc[0]
    trained = training_minutes(logs)
    latest = trained.loc[trained["game_id"].eq("g11") & trained["player_id"].eq(1)].iloc[0]

    # Last 10 completed games, latest included: nine 10s and one 50.
    assert live["min_mean_10"] == pytest.approx(14.0)
    assert latest["min_mean_10"] == pytest.approx(10.0)
    assert latest["minutes"] == 50.0
    assert live["is_starter"] == 1.0
    assert live["team_spread"] == -3.5
    assert "minutes" not in live.index
    assert "team_final_margin" not in live.index


def test_live_minutes_stop_before_the_slate_date():
    logs = pd.DataFrame(
        _game("g1", "2024-01-01", 10.0)
        + _game("g2", "2024-01-03", 30.0)
        + _game("g3", "2024-01-05", 100.0)
    )

    live = live_minutes(logs, _slate("2024-01-05")).iloc[0]
    trained = training_minutes(logs)
    same_day = trained.loc[trained["game_id"].eq("g3") & trained["player_id"].eq(1)].iloc[0]

    assert live["min_mean_10"] == pytest.approx(20.0)
    assert same_day["min_mean_10"] == pytest.approx(20.0)
    assert same_day["minutes"] == 100.0


def test_rotation_uses_the_script_budget_and_cap():
    roster = _roster(8)
    environment = _environment(
        [2.0, -1.0, 0.0],
        [0, 1, 2],
        [48.0, 53.0, 58.0],
    )
    simulator = RotationSimulator(p_ot=1.0, fallback_mu=30.0, fallback_sd=1.0)

    minutes = allocate_rotation(
        simulator,
        roster,
        environment,
        np.random.default_rng(0),
    )

    assert abs(minutes.sum(axis=1) - np.array([240.0, 265.0, 290.0])).max() < 1e-6
    assert minutes[0].max() <= 48.0 + 1e-6
    assert minutes[1].max() <= 53.0 + 1e-6
    assert minutes[2].max() <= 58.0 + 1e-6
    assert minutes.min() >= -1e-9


def test_short_roster_cannot_fill_the_budget():
    roster = _roster(4)
    environment = _environment([0.0], [0], [48.0])
    simulator = RotationSimulator(fallback_mu=30.0, fallback_sd=1.0)

    with pytest.raises(ValueError, match="budget"):
        allocate_rotation(simulator, roster, environment, np.random.default_rng(0))


def test_a_rotation_is_one_team():
    roster = _roster(8)
    roster.loc[5:, "team_id"] = 2
    environment = _environment([0.0], [0], [48.0])

    with pytest.raises(ValueError, match="one team"):
        allocate_rotation(
            RotationSimulator(fallback_mu=30.0, fallback_sd=1.0),
            roster,
            environment,
            np.random.default_rng(0),
        )


def test_allocate_rotation_rejects_more_than_one_game():
    roster = _roster(8)
    environment = _environment([0.0, 0.0], [0, 0], [48.0, 48.0])
    environment["game_id"] = ["g1", "g2"]

    with pytest.raises(ValueError, match="single game"):
        allocate_rotation(
            RotationSimulator(fallback_mu=30.0, fallback_sd=1.0),
            roster,
            environment,
            np.random.default_rng(0),
        )


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("game_id", ["g1", "g2"]),
        ("gameId", ["g1", "g2"]),
        ("game_date", ["2024-01-01", "2024-01-02"]),
    ],
)
def test_segments_from_minutes_rejects_more_than_one_game(column, values):
    role = _segment_role()
    environment = _environment([4.0, 4.0], [0, 0], [48.0, 48.0])
    environment[column] = values

    with pytest.raises(ValueError, match="single game"):
        segments_from_minutes(role, environment, np.zeros((2, len(role))))


def test_segments_from_minutes_accepts_repeated_rows_of_one_game():
    role = _segment_role()
    environment = _environment([4.0, -4.0], [0, 0], [48.0, 48.0])
    environment["game_id"] = ["g1", "g1"]
    environment["gameId"] = ["g1", "g1"]
    environment["game_date"] = ["2024-01-01", "2024-01-01"]
    minutes = np.tile(np.array([48.0] * 5 + [0.0] * 3), (2, 1))

    schedules = segments_from_minutes(role, environment, minutes)

    assert len(schedules) == 2
    assert schedules[0][0]["players"] == _five(0)
    covered = sum(segment["end_min"] - segment["start_min"] for segment in schedules[1])
    assert abs(covered - 48.0) < 1e-6


def test_the_direct_segment_builder_rejects_two_teams():
    role = _segment_role()
    role.loc[4:, "team_id"] = 2

    with pytest.raises(ValueError, match="one team"):
        build_rotation_segments(
            role,
            game_minutes=48.0,
            n_ot=0,
            team_margin=4.0,
        )


def test_blowout_loss_moves_minutes_from_starters_to_the_bench():
    roster = _roster(10)
    simulator = RotationSimulator(
        cond_model=_ScriptMinutes(),
        sigma_cond=np.full(5, 0.01),
        fallback_sd=0.01,
    )
    rng = np.random.default_rng(1)
    close = allocate_rotation(
        simulator,
        roster,
        _environment(np.full(30, 4.0), np.zeros(30), np.full(30, 48.0)),
        rng,
    )
    blowout_model = _ScriptMinutes()
    simulator.cond = blowout_model
    blowout = allocate_rotation(
        simulator,
        roster,
        _environment(np.full(30, -25.0), np.zeros(30), np.full(30, 48.0)),
        np.random.default_rng(1),
    )

    assert abs(close.sum(axis=1) - 240.0).max() < 1e-6
    assert abs(blowout.sum(axis=1) - 240.0).max() < 1e-6
    assert blowout[:, :5].mean() < close[:, :5].mean() - 2.0
    assert blowout[:, 5:].mean() > close[:, 5:].mean() + 1.0
    assert np.all(np.concatenate(blowout_model.seen) < 0)


def test_fit_rotation_keeps_the_players_recent_minutes():
    rows = []
    for game in range(6):
        for player_id, minutes in ((1, 32.0), (2, 14.0)):
            row = {column: 0.0 for column in CONDITIONAL}
            row.update(
                {
                    "player_id": player_id,
                    "game_date": f"2024-01-{game + 1:02d}",
                    "minutes": minutes,
                    "min_mean_10": minutes,
                    "is_starter": 1.0,
                    "game_minutes": 48.0,
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)

    simulator = fit_rotation(frame)
    mu, known = simulator.pregame_mu(frame.drop_duplicates("player_id"))

    assert known.all()
    assert mu[0] == pytest.approx(32.0)
    assert mu[1] == pytest.approx(14.0)
    assert set(SCRIPT).issubset(CONDITIONAL)


def test_fit_rotation_rejects_a_live_frame_with_no_target():
    logs = pd.DataFrame(_game("g1", "2024-01-01", 10.0) + _game("g2", "2024-01-03", 30.0))

    with pytest.raises(ValueError, match="minutes"):
        fit_rotation(live_minutes(logs, _slate()))


def test_missing_history_still_fills_the_script_budget():
    roster = pd.DataFrame({"player_id": np.arange(6), "team_id": 1, "is_starter": 0.0})
    environment = _environment([1.0], [0], [48.0])
    simulator = RotationSimulator(fallback_mu=20.0, fallback_sd=1.0)

    minutes = allocate_rotation(
        simulator,
        roster,
        environment,
        np.random.default_rng(2),
    )

    assert minutes.shape == (1, 6)
    assert abs(minutes.sum() - 240.0) < 1e-6
    assert minutes.max() <= 48.0 + 1e-6


def _blank(**overrides) -> dict:
    row = {column: 0.0 for column in CONDITIONAL}
    row.update(
        minutes=20.0,
        min_mean_10=20.0,
        is_starter=1.0,
        game_minutes=48.0,
        game_date="2024-01-01",
        player_id=1,
    )
    row.update(overrides)
    return row


class _RecordScript:
    def __init__(self) -> None:
        self.frame: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.frame = frame.copy()
        return np.full(len(frame), 20.0)


def test_constrain_meets_the_budget_when_players_sit_on_the_bounds():
    four = constrain(
        np.array([[48.0, 48.0, 0.0, 0.0], [10.0, 10.0, 10.0, 60.0]]),
        np.array([1.0, 1.0, 4.0, 1.0]),
        np.array([100.0, 80.0]),
        np.array([48.0, 48.0]),
    )
    five = constrain(
        np.array([[10.0, 20.0, 30.0, 40.0, 50.0]]),
        np.ones(5),
        np.array([240.0]),
        np.array([48.0]),
    )

    assert abs(four.sum(axis=1) - np.array([100.0, 80.0])).max() < 1e-6
    assert four.min() >= -1e-9
    assert np.all(four <= 48.0 + 1e-8)
    assert abs(five.sum() - 240.0) < 1e-6
    assert np.allclose(five, 48.0)


def test_low_minute_draw_keeps_its_mean_inside_the_cap():
    rng = np.random.default_rng(0)
    draw = draw_minutes(np.full(4000, 4.0), np.full(4000, 6.0), np.full(4000, 48.0), rng)

    assert abs(draw.mean() - 4.0) < 0.3
    assert draw.min() >= -1e-9
    assert draw.max() <= 48.0 + 1e-6


def test_script_moves_starters_and_the_bench_in_opposite_directions():
    rows = []
    for game in range(30):
        blowout = float(game % 2)
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=game)
        for starter, recent in ((1.0, 36.0), (0.0, 10.0)):
            delta = (-12.0 if starter else 8.0) * blowout
            rows.append(
                _blank(
                    player_id=int(starter) + 1,
                    game_date=date,
                    is_starter=starter,
                    min_mean_10=recent,
                    minutes=recent + delta,
                    blowout=blowout,
                    abs_final_margin=25.0 if blowout else 3.0,
                    team_final_margin=-25.0 if blowout else 3.0,
                )
            )
    simulator = fit_rotation(pd.DataFrame(rows))
    sample = pd.DataFrame(
        [
            _blank(is_starter=1.0, min_mean_10=36.0, blowout=0.0, abs_final_margin=3.0, team_final_margin=3.0),
            _blank(is_starter=1.0, min_mean_10=36.0, blowout=1.0, abs_final_margin=25.0, team_final_margin=-25.0),
            _blank(is_starter=0.0, min_mean_10=10.0, blowout=0.0, abs_final_margin=3.0, team_final_margin=3.0),
            _blank(is_starter=0.0, min_mean_10=10.0, blowout=1.0, abs_final_margin=25.0, team_final_margin=-25.0),
        ]
    )
    pred = simulator.cond.predict(sample)

    assert "min_mean_10" in simulator.cond.columns
    assert "min_mean_x_abs_margin" in simulator.cond.columns
    assert "starter_x_blowout" in simulator.cond.columns
    assert any(isinstance(step, StandardScaler) for _, step in simulator.cond.pipeline.steps)
    assert pred[1] < pred[0] - 6.0
    assert pred[3] > pred[2] + 4.0


def test_recent_games_are_in_the_fitted_script_model():
    rows = []
    for game in range(40):
        recent = game >= 30
        rows.append(
            _blank(
                game_date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=game),
                min_mean_10=32.0,
                minutes=14.0 if recent else 32.0,
                blowout=1.0 if recent else 0.0,
                abs_final_margin=28.0 if recent else 2.0,
                team_final_margin=-28.0 if recent else 2.0,
                is_starter=1.0,
            )
        )
    simulator = fit_rotation(pd.DataFrame(rows))
    sample = pd.DataFrame(
        [
            _blank(min_mean_10=32.0, blowout=0.0, abs_final_margin=2.0, team_final_margin=2.0),
            _blank(min_mean_10=32.0, blowout=1.0, abs_final_margin=28.0, team_final_margin=-28.0),
        ]
    )
    pred = simulator.cond.predict(sample)

    assert pred[1] < pred[0] - 8.0


def test_high_recent_minutes_revert_toward_the_league():
    rows = []
    means = np.tile(np.linspace(8.0, 36.0, 10), 4)
    for game, recent in enumerate(means):
        rows.append(
            _blank(
                game_date=pd.Timestamp("2024-03-01") + pd.Timedelta(days=int(game)),
                min_mean_10=float(recent),
                minutes=0.5 * float(recent) + 12.0,
            )
        )
    simulator = fit_rotation(pd.DataFrame(rows))
    sample = pd.DataFrame(
        [
            _blank(min_mean_10=36.0),
            _blank(min_mean_10=8.0),
        ]
    )
    high, low = simulator.cond.predict(sample)

    assert high < 36.0 - 3.0
    assert low > 8.0 + 3.0


def test_residuals_are_added_before_the_final_allocation():
    class _RawMinutes:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            starter = frame["is_starter"].to_numpy(dtype=float) == 1.0
            return np.where(starter, 36.0, 6.0)

    roster = _roster(8)
    # Bins are the pregame role, matching how the residual bank was built.
    roster["min_mean_10"] = [32.0] * 5 + [4.0] * 3
    for column in CONDITIONAL:
        if column not in roster:
            roster[column] = 0.0

    simulator = RotationSimulator(
        cond_model=_RawMinutes(),
        residual_cond={
            "groups": [
                np.array([24.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
            ]
        },
        fallback_sd=1.0,
    )
    minutes = allocate_rotation(
        simulator,
        roster,
        _environment([0.0], [0], [48.0]),
        np.random.default_rng(0),
    )
    raw = np.array([36.0] * 5 + [6.0] * 3)
    noise = np.array([0.0] * 5 + [24.0] * 3)
    expected = constrain(
        scale_mu(np.maximum(raw + noise, 0.0), [240.0]),
        np.ones(8),
        [240.0],
        [48.0],
    )

    assert minutes.shape == (1, 8)
    assert np.allclose(minutes, expected)
    assert minutes[0, :5] == pytest.approx(32.0)
    assert minutes[0, 5:] == pytest.approx(80.0 / 3.0)


def test_conditional_noise_is_tighter_than_the_pregame_residual():
    rows = []
    for game in range(40):
        blowout = float(game % 2)
        rows.append(
            _blank(
                game_date=pd.Timestamp("2024-04-01") + pd.Timedelta(days=game),
                min_mean_10=30.0,
                minutes=30.0 - 20.0 * blowout,
                blowout=blowout,
                abs_final_margin=25.0 if blowout else 2.0,
                team_final_margin=-25.0 if blowout else 2.0,
            )
        )
    simulator = fit_rotation(pd.DataFrame(rows))

    assert simulator.sig_cond.mean() < simulator.sig_pre.mean() - 1.0


def test_conditional_model_rejects_a_missing_script_column():
    frame = pd.DataFrame([_blank(game_date=f"2024-05-{day:02d}") for day in range(1, 9)])
    simulator = fit_rotation(frame)

    with pytest.raises(ValueError, match="blowout"):
        simulator.cond.predict(frame.drop(columns=["blowout"]))


def test_omitted_overtime_count_follows_the_game_length():
    recorder = _RecordScript()
    roster = _roster(6)
    RotationSimulator(
        cond_model=recorder,
        sigma_cond=np.full(5, 0.01),
        fallback_sd=0.01,
    ).simulate(
        roster,
        n=2,
        margin=[0.0, 0.0],
        team_total=[200.0, 225.0],
        cap=[40.0, 45.0],
        n_ot=None,
        regulation_minutes=40.0,
        ot_minutes=5.0,
        rng=np.random.default_rng(0),
    )

    first = recorder.frame.iloc[:6]
    second = recorder.frame.iloc[6:]
    assert first["n_ot"].eq(0.0).all()
    assert first["went_ot"].eq(0.0).all()
    assert second["n_ot"].eq(1.0).all()
    assert second["went_ot"].eq(1.0).all()


def test_training_window_excludes_the_target_game_and_keeps_the_posted_start():
    values = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 40]
    games = []
    for index, minutes in enumerate(values):
        games.extend(_game(f"g{index}", f"2024-01-{index + 1:02d}", float(minutes)))
    logs = pd.DataFrame(games)
    last = logs["game_id"].eq("g11") & logs["player_id"].eq(1)
    logs.loc[last, "start_position"] = ""

    trained = training_minutes(logs)
    player = trained.loc[trained["player_id"].eq(1)].sort_values("game_date")
    for index, row in enumerate(player.itertuples(index=False)):
        prior = values[max(0, index - 10) : index]
        if not prior:
            assert pd.isna(row.min_mean_10)
        else:
            assert row.min_mean_10 == pytest.approx(float(np.mean(prior)))
        assert row.minutes == values[index]
    assert player.iloc[-1].is_starter == 0.0
    assert player.iloc[-1].start_rate_10 == 1.0


def test_clock_minutes_count_in_the_target_and_the_next_window():
    logs = pd.DataFrame(
        _game("g1", "2024-01-01", 10.0)
        + _game("g2", "2024-01-03", 10.0)
        + _game("g3", "2024-01-05", 10.0)
    )
    mask = logs["player_id"].eq(1) & logs["game_id"].eq("g2")
    logs["min"] = logs["min"].astype(object)
    logs.loc[mask, "min"] = "34:12"
    trained = training_minutes(logs)
    player = trained.loc[trained["player_id"].eq(1)].set_index("game_id")

    assert player.loc["g2", "minutes"] == pytest.approx(34.2)
    assert player.loc["g3", "min_mean_10"] == pytest.approx((10.0 + 34.2) / 2.0)


def test_a_blank_minute_cell_falls_back_to_the_other_column():
    logs = pd.DataFrame(_game("g1", "2024-01-01", 10.0) + _game("g2", "2024-01-03", 30.0))
    logs["minutes"] = np.nan
    mask = logs["player_id"].eq(1) & logs["game_id"].eq("g2")
    logs.loc[mask, "min"] = np.nan
    logs.loc[mask, "minutes"] = 22.0

    trained = training_minutes(logs)
    second = trained.loc[trained["game_id"].eq("g2") & trained["player_id"].eq(1)].iloc[0]

    assert second["minutes"] == 22.0


def test_conflicting_minute_rows_are_rejected():
    logs = pd.DataFrame(_game("g1", "2024-01-01", 10.0) + _game("g1", "2024-01-01", 12.0))

    with pytest.raises(ValueError, match="conflict"):
        training_minutes(logs)


def _segment_role(n: int = 8, *, starters: int = 5, out: tuple[str, ...] = ()) -> pd.DataFrame:
    ids = [f"p{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "player_id": ids,
            "team_id": np.ones(n, dtype=int),
            "is_starter": [1.0 if index < starters else 0.0 for index in range(n)],
            "available": [0.0 if player in out else 1.0 for player in ids],
        }
    )


def _span(period: int, start: float, end: float, players: list[str]) -> dict:
    return {
        "period": period,
        "start_min": start,
        "end_min": end,
        "players": list(players),
    }


def _five(start: int = 0) -> list[str]:
    return [f"p{start + offset}" for offset in range(5)]


def _regulation(period_one: list[dict]) -> list[dict]:
    """One regulation game. Periods 2–4 stay with the posted five."""
    segments = list(period_one)
    for period in (2, 3, 4):
        segments.append(_span(period, 0.0, 12.0, _five()))
    return segments


def _segment_simulator() -> RotationSimulator:
    return RotationSimulator(fallback_mu=24.0, fallback_sd=1.0)


def _segment_minutes(role, environment, segments=None, *, seed: int = 0) -> np.ndarray:
    return allocate_rotation(
        _segment_simulator(),
        role,
        environment,
        np.random.default_rng(seed),
        method="segments",
        segments=segments,
    )


def test_a_segment_is_five_distinct_players_on_the_active_roster():
    role = _segment_role(10)
    environment = _environment([4.0], [0], [48.0])
    schedule = _regulation([_span(1, 0.0, 12.0, _five())])

    assert invalid_segment_count(schedule, role, game_minutes=48.0, n_ot=0) == 0
    minutes = _segment_minutes(role, environment, schedule)
    assert minutes.shape == (1, 10)

    with pytest.raises(ValueError, match="five"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 0.0, 12.0, _five()[:4])]),
        )
    with pytest.raises(ValueError, match="distinct"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 0.0, 12.0, ["p0", "p0", "p1", "p2", "p3"])]),
        )
    with pytest.raises(ValueError, match="roster"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 0.0, 12.0, ["p0", "p1", "p2", "p3", "ghost"])]),
        )


def test_an_unavailable_player_is_not_on_the_floor():
    role = _segment_role(10, out=("p0",))
    environment = _environment([4.0], [0], [48.0])

    with pytest.raises(ValueError, match="available"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 0.0, 12.0, _five())]),
        )

    short = _segment_role(6, out=("p0", "p1"))
    with pytest.raises(ValueError, match="budget"):
        _segment_minutes(short, environment)


def test_segments_cover_each_period_without_a_gap_or_overlap():
    role = _segment_role(10)
    environment = _environment([4.0], [0], [48.0])
    first, second = _five(0), _five(5)
    valid = _regulation(
        [
            _span(1, 0.0, 6.0, first),
            _span(1, 6.0, 12.0, second),
        ]
    )

    assert invalid_segment_count(valid, role, game_minutes=48.0, n_ot=0) == 0
    with pytest.raises(ValueError, match="positive"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 6.0, 6.0, first), _span(1, 6.0, 12.0, second)]),
        )
    with pytest.raises(ValueError, match="gap"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 0.0, 6.0, first), _span(1, 7.0, 12.0, second)]),
        )
    with pytest.raises(ValueError, match="overlap"):
        _segment_minutes(
            role,
            environment,
            _regulation([_span(1, 0.0, 7.0, first), _span(1, 6.0, 12.0, second)]),
        )


def test_segments_cover_the_script_game_minutes():
    role = _segment_role()
    for n_ot in (0, 1, 2):
        game_minutes = 48.0 + 5.0 * n_ot
        segments = build_rotation_segments(
            role,
            game_minutes=game_minutes,
            n_ot=n_ot,
            team_margin=3.0,
        )
        environment = _environment([3.0], [n_ot], [game_minutes])

        assert invalid_segment_count(
            segments, role, game_minutes=game_minutes, n_ot=n_ot
        ) == 0
        assert {segment["period"] for segment in segments} == set(range(1, 5 + n_ot))
        for period in range(1, 5):
            covered = sum(
                segment["end_min"] - segment["start_min"]
                for segment in segments
                if segment["period"] == period
            )
            assert covered == pytest.approx(12.0)
        for period in range(5, 5 + n_ot):
            covered = sum(
                segment["end_min"] - segment["start_min"]
                for segment in segments
                if segment["period"] == period
            )
            assert covered == pytest.approx(5.0)
        assert sum(segment["end_min"] - segment["start_min"] for segment in segments) == pytest.approx(
            game_minutes
        )
        minutes = _segment_minutes(role, environment)
        assert abs(minutes.sum() - 5.0 * game_minutes) < 1e-6
        assert minutes.max() <= game_minutes + 1e-6

    regulation = _regulation([_span(1, 0.0, 12.0, _five())])
    with pytest.raises(ValueError, match="period"):
        _segment_minutes(
            _segment_role(10),
            _environment([1.0], [0], [48.0]),
            regulation + [_span(5, 0.0, 5.0, _five())],
        )
    with pytest.raises(ValueError, match="period"):
        _segment_minutes(
            _segment_role(10),
            _environment([1.0], [1], [53.0]),
            regulation,
        )


def test_minutes_equal_accumulated_exposure():
    role = _segment_role(10)
    first, second = _five(0), _five(5)
    schedule = _regulation(
        [
            _span(1, 0.0, 6.0, first),
            _span(1, 6.0, 12.0, second),
        ]
    )
    environment = _environment([4.0], [0], [48.0])

    minutes = _segment_minutes(role, environment, schedule)

    # Period 1 is split 6 and 6. Periods 2–4 are 36 minutes for the first five.
    assert np.allclose(minutes[0], [42.0] * 5 + [6.0] * 5)
    assert abs(minutes.sum() - 5.0 * 48.0) < 1e-6
    assert minutes.max() <= 48.0 + 1e-6
    assert minutes.min() >= -1e-9


def test_overtime_adds_five_minutes_for_each_player_in_that_period():
    role = _segment_role(10)
    first = _five(0)
    regulation = _regulation(
        [
            _span(1, 0.0, 6.0, first),
            _span(1, 6.0, 12.0, _five(5)),
        ]
    )
    overtime = regulation + [_span(5, 0.0, 5.0, first)]
    base = _segment_minutes(role, _environment([2.0], [0], [48.0]), regulation)
    extra = _segment_minutes(role, _environment([2.0], [1], [53.0]), overtime)

    assert np.allclose(extra[0, :5] - base[0, :5], 5.0)
    assert np.allclose(extra[0, 5:], base[0, 5:])
    assert abs(extra.sum() - base.sum() - 5.0 * 5.0) < 1e-6
    assert abs(extra.sum() - 5.0 * 53.0) < 1e-6
    assert extra.max() <= 53.0 + 1e-6


def test_segment_minutes_are_not_rescaled(monkeypatch):
    def _rejected(*_args, **_kwargs):
        raise AssertionError("segment minutes must not be rescaled")

    monkeypatch.setattr("game_rotations.simulator.scale_mu", _rejected)
    monkeypatch.setattr("game_rotations.simulator.constrain", _rejected)
    monkeypatch.setattr("game_rotations.segments.scale_mu", _rejected)
    monkeypatch.setattr("game_rotations.segments.constrain", _rejected)
    role = _segment_role(10)
    schedule = _regulation(
        [
            _span(1, 0.0, 6.0, _five(0)),
            _span(1, 6.0, 12.0, _five(5)),
        ]
    )

    minutes = _segment_minutes(role, _environment([4.0], [0], [48.0]), schedule)

    assert np.allclose(minutes[0], [42.0] * 5 + [6.0] * 5)
    # The exposure is already the budget, so a later rescale would leave it
    # unchanged. The patches above fail if that rescale runs anyway.
    assert abs(minutes.sum() - 5.0 * 48.0) < 1e-6


def test_players_who_share_a_segment_move_together():
    role = _segment_role(10)
    environment = _environment([4.0], [0], [48.0])
    first, second = _five(0), _five(5)
    short = _segment_minutes(
        role,
        environment,
        _regulation([_span(1, 0.0, 6.0, first), _span(1, 6.0, 12.0, second)]),
    )
    longer = _segment_minutes(
        role,
        environment,
        _regulation([_span(1, 0.0, 8.0, first), _span(1, 8.0, 12.0, second)]),
    )

    delta = longer - short
    assert np.allclose(delta[0, :5], 2.0)
    assert np.allclose(delta[0, 5:], -2.0)
    assert abs(longer.sum() - 5.0 * 48.0) < 1e-6


def test_posted_start_changes_minutes_only_by_changing_segments():
    role = _segment_role(8)
    other = role.copy()
    other["is_starter"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    environment = _environment([4.0], [0], [48.0])
    opening = build_rotation_segments(
        role, game_minutes=48.0, n_ot=0, team_margin=4.0
    )
    replaced = build_rotation_segments(
        other, game_minutes=48.0, n_ot=0, team_margin=4.0
    )

    assert opening[0]["players"] == _five(0)
    assert replaced[0]["players"] == ["p3", "p4", "p5", "p6", "p7"]
    from_role = _segment_minutes(role, environment)
    from_other = _segment_minutes(other, environment)
    held = _segment_minutes(other, environment, opening)

    assert not np.allclose(from_role, from_other)
    assert np.allclose(held, from_role)


def test_removing_a_player_drops_him_and_keeps_the_budget():
    role = _segment_role(8, out=("p0",))
    environment = _environment([4.0], [0], [48.0])
    segments = build_rotation_segments(
        role, game_minutes=48.0, n_ot=0, team_margin=4.0
    )

    assert all("p0" not in segment["players"] for segment in segments)
    assert segments[0]["players"] == ["p1", "p2", "p3", "p4", "p5"]
    minutes = _segment_minutes(role, environment)
    assert minutes[0, 0] == 0.0
    assert abs(minutes.sum() - 5.0 * 48.0) < 1e-6
    assert minutes.max() <= 48.0 + 1e-6


def test_a_blowout_changes_the_closing_five():
    role = _segment_role(9)
    close = build_rotation_segments(
        role, game_minutes=48.0, n_ot=0, team_margin=4.0
    )
    blowout = build_rotation_segments(
        role, game_minutes=48.0, n_ot=0, team_margin=-24.0
    )
    overtime = build_rotation_segments(
        role, game_minutes=53.0, n_ot=1, team_margin=-24.0
    )

    def _closing(segments):
        return [segment for segment in segments if segment["period"] == 4][-1]["players"]

    assert _closing(close) == _five(0)
    assert _closing(blowout) == ["p5", "p6", "p7", "p8", "p0"]
    assert overtime[-1]["period"] == 5
    assert overtime[-1]["players"] == _closing(blowout)
    assert abs(overtime[-1]["end_min"] - overtime[-1]["start_min"] - 5.0) < 1e-9


def test_the_same_seed_repeats_segment_minutes():
    role = _segment_role()
    environment = _environment([4.0, -22.0], [0, 1], [48.0, 53.0])
    first = _segment_minutes(role, environment, seed=11)
    second = _segment_minutes(role, environment, seed=11)

    assert first.shape == (2, 8)
    assert np.array_equal(first, second)
    assert not np.allclose(first[0], first[1])
    direct = _segment_simulator().simulate(
        role,
        method="segments",
        n=2,
        n_ot=environment["n_ot"].to_numpy(),
        margin=environment["team_margin"].to_numpy(),
        team_total=5.0 * environment["game_minutes"].to_numpy(),
        cap=environment["game_minutes"].to_numpy(),
        rng=np.random.default_rng(11),
    )
    assert np.array_equal(direct, first)


def test_state_and_gaussian_stay_on_the_script_budget():
    roster = _roster(8)
    environment = _environment([4.0, -22.0], [0, 1], [48.0, 53.0])
    simulator = RotationSimulator(fallback_mu=30.0, fallback_sd=1.0)
    budget = 5.0 * environment["game_minutes"].to_numpy()
    cap = environment["game_minutes"].to_numpy()

    for method in ("state", "gaussian"):
        minutes = allocate_rotation(
            simulator,
            roster,
            environment,
            np.random.default_rng(0),
            method=method,
        )
        assert np.allclose(minutes.sum(axis=1), budget)
        assert np.all(minutes <= cap[:, None] + 1e-6)
        assert minutes.min() >= -1e-9
