"""The SportsGameOdds key stays in the environment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRAPERS = (
    ROOT / "src" / "scrapers" / "nba_props.py",
    ROOT / "src" / "scrapers" / "tennis_props.py",
)


def test_sports_game_odds_key_is_read_from_the_environment() -> None:
    for path in SCRAPERS:
        text = path.read_text()
        assert 'API_KEY = "' not in text
        assert 'os.environ.get("SGO_API_KEY", "")' in text
