from __future__ import annotations

import unittest

import pandas as pd

from dashboard.stringing import classify_stringing_missing_headers, _to_datetime_normalize, normalize_stringing_columns


class StringingNormalizationTests(unittest.TestCase):
    def test_normalization_collapses_duplicate_mapped_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "From AP": ["1/0", "2/0"],
                "To AP": ["1/1", "2/1"],
                "P/O Starting Date": ["2026-04-01", "2026-04-02"],
                "F/S/ Completion Date": ["2026-04-03", "2026-04-04"],
                "Length": [None, 50],
                "Length (m)": [123, 999],
            }
        )

        normalized, report = normalize_stringing_columns(frame)

        self.assertTrue(normalized.columns.is_unique)
        self.assertEqual(sum(1 for col in normalized.columns if col == "length_m"), 1)
        self.assertEqual(normalized.loc[0, "length_m"], 123)
        self.assertEqual(normalized.loc[1, "length_m"], 50)
        self.assertIn("Length", report.get("present", []))

    def test_section_column_derives_from_and_to_ap(self) -> None:
        frame = pd.DataFrame(
            {
                "Section": ["AP25-AP26", "98/0 to 99/0"],
                "P/O Starting Date": ["2026-06-17", "2026-06-14"],
                "F/S/ Completion Date": ["2026-06-24", "2026-06-15"],
                "Length": [1.028, 0.344],
            }
        )

        normalized, report = normalize_stringing_columns(frame)

        self.assertTrue(bool(classify_stringing_missing_headers(report)["is_critical_complete"]))
        self.assertEqual(normalized.loc[0, "from_ap"], "AP25")
        self.assertEqual(normalized.loc[0, "to_ap"], "AP26")
        self.assertEqual(normalized.loc[1, "from_ap"], "98/0")
        self.assertEqual(normalized.loc[1, "to_ap"], "99/0")

    def test_date_ranges_parse_start_and_completion_dates(self) -> None:
        value = "12-02-2025 to 16-02-2025"

        self.assertEqual(_to_datetime_normalize(value, prefer="first"), pd.Timestamp("2025-02-12"))
        self.assertEqual(_to_datetime_normalize(value, prefer="last"), pd.Timestamp("2025-02-16"))


if __name__ == "__main__":
    unittest.main()
