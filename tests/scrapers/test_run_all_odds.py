"""MLB-only odds scraper runner."""

from __future__ import annotations

import pytest
from src.scrapers.run_all_odds import KNOWN_NAMES, main, resolve_jobs


def test_resolve_jobs_default_is_mlb_only() -> None:
    jobs = resolve_jobs()
    assert [j.name for j in jobs] == [
        "mlb_novig",
        "mlb_prophetx",
        "mlb_underdog",
        "mlb_fanduel",
        "mlb_draftking",
        "mlb_pinnacle",
        "mlb_prizepick",
    ]
    assert all(j.league == "mlb" for j in jobs)
    assert all(j.module.startswith("src.scrapers.mlb.") for j in jobs)
    by_name = {j.name: j for j in jobs}
    assert by_name["mlb_novig"].module == "src.scrapers.mlb.mlb_novig"
    assert by_name["mlb_fanduel"].module == "src.scrapers.mlb.mlb_fanduel"
    assert by_name["mlb_draftking"].module == "src.scrapers.mlb.mlb_draftking"
    assert by_name["mlb_pinnacle"].env == {"PINNACLE_LEAGUES": "mlb"}
    assert not any("wnba" in j.name or "wnba" in j.module for j in jobs)


def test_known_names_are_mlb_only() -> None:
    assert KNOWN_NAMES == {
        "mlb_novig",
        "mlb_prophetx",
        "mlb_underdog",
        "mlb_fanduel",
        "mlb_draftking",
        "mlb_pinnacle",
        "mlb_prizepick",
    }
    assert not any(name.startswith("wnba") for name in KNOWN_NAMES)


def test_resolve_jobs_only_subset_keeps_canonical_order() -> None:
    jobs = resolve_jobs(only=["mlb_prizepick", "mlb_novig"])
    assert [j.name for j in jobs] == ["mlb_novig", "mlb_prizepick"]


def test_resolve_jobs_only_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scraper"):
        resolve_jobs(only=["wnba_novig"])
    with pytest.raises(ValueError, match="Unknown scraper"):
        resolve_jobs(only=["not_a_scraper"])


def test_resolve_jobs_exclude_prizepicks() -> None:
    names = [j.name for j in resolve_jobs(exclude=["mlb_prizepick"])]
    assert names == [
        "mlb_novig",
        "mlb_prophetx",
        "mlb_underdog",
        "mlb_fanduel",
        "mlb_draftking",
        "mlb_pinnacle",
    ]


def test_resolve_jobs_exclude_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scraper"):
        resolve_jobs(exclude=["wnba_prizepick"])


def test_cli_rejects_league_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--league", "wnba"])
    assert excinfo.value.code == 2


def test_cli_rejects_wnba_only() -> None:
    assert main(["--only", "wnba_novig"]) == 2
