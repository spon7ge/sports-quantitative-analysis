"""MLB and NBA odds scraper runner."""

from __future__ import annotations

import pytest
from src.scrapers.run_all_odds import KNOWN_NAMES, main, resolve_jobs

MLB_NAMES = [
    "mlb_novig",
    "mlb_prophetx",
    "mlb_underdog",
    "mlb_fanduel",
    "mlb_draftking",
    "mlb_pinnacle",
    "mlb_prizepick",
]
NBA_NAMES = [
    "nba_novig",
    "nba_prophetx",
    "nba_underdog",
    "nba_fanduel",
    "nba_draftking",
    "nba_pinnacle",
    "nba_rotowire_lineup",
    "nba_prizepick",
]


def test_resolve_jobs_default_runs_mlb_then_nba_with_prizepicks_last() -> None:
    jobs = resolve_jobs()
    assert [j.name for j in jobs] == (
        MLB_NAMES[:-1] + NBA_NAMES[:-1] + ["mlb_prizepick", "nba_prizepick"]
    )
    assert not any("wnba" in j.name or "wnba" in j.module for j in jobs)
    by_name = {j.name: j for j in jobs}
    assert by_name["mlb_novig"].module == "src.scrapers.mlb.mlb_novig"
    assert by_name["nba_novig"].module == "src.scrapers.nba.nba_novig"
    assert by_name["mlb_pinnacle"].env == {"PINNACLE_LEAGUES": "mlb"}
    assert by_name["nba_pinnacle"].env == {"PINNACLE_LEAGUES": "nba"}
    assert by_name["nba_fanduel"].module == "src.scrapers.nba.nba_fanduel"
    assert by_name["nba_draftking"].module == "src.scrapers.nba.nba_draftking"
    assert by_name["nba_rotowire_lineup"].module == (
        "src.scrapers.nba.nba_rotowire_lineup"
    )
    assert by_name["nba_rotowire_lineup"].league == "nba"
    assert by_name["nba_rotowire_lineup"].env is None
    names = [j.name for j in jobs]
    assert names[names.index("mlb_prizepick") - 1] == "nba_rotowire_lineup"
    assert "nba_rotowire_lineup" not in [j.name for j in resolve_jobs(league="mlb")]


def test_resolve_jobs_league_keeps_that_league_order() -> None:
    assert [j.name for j in resolve_jobs(league="mlb")] == MLB_NAMES
    assert [j.name for j in resolve_jobs(league="nba")] == NBA_NAMES
    assert all(j.league == "nba" for j in resolve_jobs(league="nba"))
    assert all(j.module.startswith("src.scrapers.nba.") for j in resolve_jobs(league="nba"))


def test_known_names_are_mlb_and_nba() -> None:
    assert KNOWN_NAMES == set(MLB_NAMES + NBA_NAMES)
    assert not any(name.startswith("wnba") for name in KNOWN_NAMES)


def test_resolve_jobs_only_subset_keeps_canonical_order() -> None:
    jobs = resolve_jobs(only=["nba_prizepick", "mlb_novig", "nba_novig"])
    assert [j.name for j in jobs] == ["mlb_novig", "nba_novig", "nba_prizepick"]


def test_resolve_jobs_only_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scraper"):
        resolve_jobs(only=["wnba_novig"])
    with pytest.raises(ValueError, match="Unknown scraper"):
        resolve_jobs(only=["not_a_scraper"])


def test_resolve_jobs_unknown_league_raises() -> None:
    with pytest.raises(ValueError, match="Unknown league"):
        resolve_jobs(league="wnba")


def test_resolve_jobs_exclude_prizepicks() -> None:
    names = [j.name for j in resolve_jobs(exclude=["mlb_prizepick", "nba_prizepick"])]
    assert names == MLB_NAMES[:-1] + NBA_NAMES[:-1]


def test_resolve_jobs_exclude_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scraper"):
        resolve_jobs(exclude=["wnba_prizepick"])


def test_cli_rejects_unknown_league() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--league", "wnba"])
    assert excinfo.value.code == 2


def test_cli_rejects_wnba_only() -> None:
    assert main(["--only", "wnba_novig"]) == 2


def test_cli_league_nba_runs_nba_jobs_only(monkeypatch) -> None:
    ran: list[str] = []

    def fake_run_job(job, **_kwargs):
        ran.append(job.name)
        return 0

    monkeypatch.setattr("src.scrapers.run_all_odds.run_job", fake_run_job)
    assert main(["--league", "nba"]) == 0
    assert set(ran) == set(NBA_NAMES)
    # PrizePicks HTTP attempt is in the HTTP pool; browser fallback is separate.
    assert "nba_prizepick" in ran


def test_cancel_running_terminates_procs() -> None:
    import threading

    from src.scrapers.run_all_odds import cancel_running

    class FakeProc:
        def __init__(self):
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    procs = [FakeProc(), FakeProc()]
    cancel_running(procs, threading.Lock())  # type: ignore[arg-type]
    assert all(p.terminated for p in procs)


def test_classify_job_pools() -> None:
    from src.scrapers.run_all_odds import classify_pool, resolve_jobs

    pools = {j.name: classify_pool(j) for j in resolve_jobs(league="nba")}
    assert pools["nba_novig"] == "http"
    assert pools["nba_draftking"] == "http"
    assert pools["nba_prizepick"] == "http"
    assert pools["nba_pinnacle"] == "pinnacle"
    assert pools["nba_rotowire_lineup"] == "http"


def test_prizepick_http_job_sets_no_browser() -> None:
    from src.scrapers.run_all_odds import resolve_jobs

    job = next(j for j in resolve_jobs(league="nba") if j.name == "nba_prizepick")
    assert job.env == {"PRIZEPICKS_NO_BROWSER": "1"}


def test_next_aligned_lands_on_quarter_hours() -> None:
    from datetime import datetime

    from src.scrapers.run_all_odds import next_aligned

    assert next_aligned(datetime(2026, 10, 24, 12, 7, 30), 15) == datetime(
        2026, 10, 24, 12, 15
    )
    assert next_aligned(datetime(2026, 10, 24, 12, 15), 15) == datetime(
        2026, 10, 24, 12, 15
    )
    assert next_aligned(datetime(2026, 10, 24, 12, 15, 1), 15) == datetime(
        2026, 10, 24, 12, 30
    )
    assert next_aligned(datetime(2026, 10, 24, 12, 59), 15) == datetime(
        2026, 10, 24, 13, 0
    )
    assert next_aligned(datetime(2026, 10, 24, 12, 0), 15) == datetime(
        2026, 10, 24, 12, 0
    )
    assert next_aligned(datetime(2026, 10, 24, 12, 45), 15) == datetime(
        2026, 10, 24, 12, 45
    )


def test_next_aligned_rejects_interval_that_does_not_divide_the_hour() -> None:
    from datetime import datetime

    from src.scrapers.run_all_odds import next_aligned

    with pytest.raises(ValueError, match="divisor of 60"):
        next_aligned(datetime(2026, 10, 24, 12, 7), 7)


def test_run_on_marks_waits_for_quarter_hours_and_skips_overrun_slots() -> None:
    from datetime import datetime, timedelta

    from src.scrapers.run_all_odds import run_on_marks

    clock = [datetime(2026, 10, 24, 12, 7, 30)]
    calls: list[datetime] = []

    def now() -> datetime:
        return clock[0]

    def sleep(seconds: float) -> None:
        clock[0] = clock[0] + timedelta(seconds=seconds)

    def run_once() -> int:
        calls.append(clock[0])
        clock[0] = clock[0] + timedelta(minutes=20)
        return 0

    assert (
        run_on_marks(
            interval_minutes=15,
            run_once=run_once,
            now=now,
            sleep=sleep,
            cycles=2,
        )
        == 0
    )
    assert calls == [
        datetime(2026, 10, 24, 12, 15),
        datetime(2026, 10, 24, 12, 45),
    ]


def test_run_all_prints_arbitrage_under_the_succeeded_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.scrapers.run_all_odds.run_job", lambda job, **_kwargs: 0)

    def fake_report(*, league, not_before, root=None):
        print("ARB_MARKER", flush=True)
        assert league == "nba"
        assert not_before is not None
        return 0

    monkeypatch.setattr("src.scrapers.run_all_odds.report_arbitrage", fake_report)
    assert main(["--league", "nba"]) == 0
    out = capsys.readouterr().out
    assert "8/8 succeeded" in out
    assert out.index("8/8 succeeded") < out.index("ARB_MARKER")


def test_arbitrage_error_does_not_fail_the_scrape(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.scrapers.run_all_odds.run_job", lambda job, **_kwargs: 0)

    def boom(*, league, not_before, root=None):
        raise RuntimeError("disk")

    monkeypatch.setattr("src.scrapers.run_all_odds.report_arbitrage", boom)
    assert main(["--league", "nba"]) == 0
    assert "Arbitrage check failed: disk" in capsys.readouterr().out


def test_cli_every_schedules_clock_aligned_runs(monkeypatch) -> None:
    seen: dict[str, int] = {}

    def fake_run_on_marks(**kwargs):
        seen["interval_minutes"] = kwargs["interval_minutes"]
        return 0

    monkeypatch.setattr("src.scrapers.run_all_odds.run_on_marks", fake_run_on_marks)
    assert main(["--league", "nba", "--every", "15"]) == 0
    assert seen["interval_minutes"] == 15


def test_cli_every_rejects_unaligned_interval() -> None:
    assert main(["--every", "7"]) == 2
