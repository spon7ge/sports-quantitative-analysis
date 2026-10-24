"""DraftKings explicit subcategory selection and exclusions."""

from __future__ import annotations

from src.scrapers.nba.nba_draftking import (
    _DEFAULT_CONFIG,
    subcategory_allowed,
    subcategories_from_nav,
)

_NAV_HTML = """
<html><script>
window.__LAYOUT__ = {"id":"NavRoot","title":"Nav Root","children":[
  {"id":"games","seoId":"games","title":"Games","parameters":{"leagueId":"42648"},"children":[
    {"id":"points","seoId":"player-points","title":"Player Points","parameters":{"sportId":"2","leagueId":"42648"},"children":[
      {"id":"pts","seoId":"points","title":"Points","tags":["PlayerProps"],
       "parameters":{"sportId":"2","leagueId":"42648","categoryId":"1215","subcategoryId":"12488"}},
      {"id":"1st","seoId":"1st-point","title":"1st Point","tags":["PlayerProps"],
       "parameters":{"sportId":"2","leagueId":"42648","categoryId":"1215","subcategoryId":"99901"}}
    ]},
    {"id":"live","seoId":"live-player-props","title":"Live Player Props","parameters":{"sportId":"2","leagueId":"42648"},"children":[
      {"id":"live-pts","seoId":"points","title":"Points","tags":["PlayerProps"],
       "parameters":{"sportId":"2","leagueId":"42648","categoryId":"9000","subcategoryId":"99902"}}
    ]},
    {"id":"combos","seoId":"player-combos","title":"Player Combos","parameters":{"sportId":"2","leagueId":"42648"},"children":[
      {"id":"pra","seoId":"pts-reb-ast","title":"Pts + Reb + Ast","tags":["PlayerProps"],
       "parameters":{"sportId":"2","leagueId":"42648","categoryId":"1216","subcategoryId":"12492"}}
    ]}
  ]}
]};
</script></html>
"""


def test_defaults_use_explicit_names_and_zero_delay() -> None:
    names = _DEFAULT_CONFIG.get("dk_subcategory_names") or []
    assert "Points" in names
    assert "Pts + Reb + Ast" in names
    assert float(_DEFAULT_CONFIG.get("dk_request_delay") or 0) == 0.0
    assert int(_DEFAULT_CONFIG.get("dk_concurrency") or 0) >= 6


def test_subcategory_allowed_excludes_live_and_first_event() -> None:
    assert subcategory_allowed(
        category_name="Player Points",
        name="Points",
        names=_DEFAULT_CONFIG["dk_subcategory_names"],
    )
    assert not subcategory_allowed(
        category_name="Live Player Props",
        name="Points",
        names=_DEFAULT_CONFIG["dk_subcategory_names"],
    )
    assert not subcategory_allowed(
        category_name="Player Points",
        name="1st Point",
        names=_DEFAULT_CONFIG["dk_subcategory_names"],
    )


def test_nav_keeps_explicit_pregame_tabs_only() -> None:
    subs = subcategories_from_nav(
        _NAV_HTML,
        names=_DEFAULT_CONFIG["dk_subcategory_names"],
    )
    assert {(s.name, s.subcategory_id) for s in subs} == {
        ("Points", "12488"),
        ("Pts + Reb + Ast", "12492"),
    }
