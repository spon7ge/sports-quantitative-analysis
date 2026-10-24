"""Stage report / Retry-After helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.scrapers.stage_report import StageReport, retry_after_seconds, stage_timer


def test_stage_timer_records_duration() -> None:
    report = StageReport(book="fanduel")
    with stage_timer(report, "download"):
        pass
    assert "download" in report.stages_s
    assert report.stages_s["download"] >= 0.0


def test_exit_code_requires_json_and_db() -> None:
    r = StageReport(book="novig", json_ok=True, db_ok=None)
    assert r.log_and_exit_code() == 0
    r.db_ok = False
    assert r.log_and_exit_code() == 1
    r2 = StageReport(book="novig", json_ok=False, db_ok=True)
    assert r2.log_and_exit_code() == 1


def test_retry_after_seconds() -> None:
    resp = SimpleNamespace(headers={"Retry-After": "2.5"})
    assert retry_after_seconds(resp, default=1.0) == 2.5
    assert retry_after_seconds(SimpleNamespace(headers={}), default=1.0) == 1.0
