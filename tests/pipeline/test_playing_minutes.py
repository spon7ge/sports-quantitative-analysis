"""Silver boxscore minutes live in ``min``, not the tracking ``minutes`` column."""

from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline.silver.columns import assign_playing_minutes


class AssignPlayingMinutesTests(unittest.TestCase):
    def test_prefers_numeric_boxscore_min_over_empty_tracking_minutes(self) -> None:
        frame = pd.DataFrame(
            {
                "min": [39.95, 12.5],
                "minutes": [None, None],
            }
        )

        result = assign_playing_minutes(frame)

        pd.testing.assert_series_equal(
            result["minutes"],
            pd.Series([39.95, 12.5], name="minutes"),
        )

    def test_parses_clock_format_when_min_is_absent(self) -> None:
        frame = pd.DataFrame({"minutes": ["36:30", "8:05"]})

        result = assign_playing_minutes(frame)

        pd.testing.assert_series_equal(
            result["minutes"],
            pd.Series([36.5, 8 + 5 / 60], name="minutes"),
        )

    def test_raises_when_no_usable_minutes_exist(self) -> None:
        frame = pd.DataFrame({"minutes": [None, None]})

        with self.assertRaises(ValueError):
            assign_playing_minutes(frame)
