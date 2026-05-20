from __future__ import annotations

import unittest

import pandas as pd

from dashboard import workbook


class CalendarWeekUtilsTests(unittest.TestCase):
    def test_align_to_previous_sunday(self) -> None:
        self.assertEqual(
            workbook._align_to_previous_sunday(pd.Timestamp("2026-05-13")),
            pd.Timestamp("2026-05-10"),
        )
        self.assertEqual(
            workbook._align_to_previous_sunday(pd.Timestamp("2026-05-10")),
            pd.Timestamp("2026-05-10"),
        )

    def test_previous_completed_week_window(self) -> None:
        start, end = workbook._previous_completed_week_window(pd.Timestamp("2026-05-19"))
        self.assertEqual(start, pd.Timestamp("2026-05-10"))
        self.assertEqual(end, pd.Timestamp("2026-05-16"))

    def test_generate_calendar_week_labels_includes_partial_trailing_week(self) -> None:
        labels = workbook._generate_calendar_week_labels(
            pd.Timestamp("2026-05-10"),
            pd.Timestamp("2026-05-18"),
        )
        self.assertEqual(
            labels,
            [
                "2026-05-10 to 2026-05-16",
                "2026-05-17 to 2026-05-18",
            ],
        )


if __name__ == "__main__":
    unittest.main()
