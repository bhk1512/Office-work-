from __future__ import annotations

import unittest

import pandas as pd

from dashboard.stringing import normalize_stringing_columns


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


if __name__ == "__main__":
    unittest.main()
