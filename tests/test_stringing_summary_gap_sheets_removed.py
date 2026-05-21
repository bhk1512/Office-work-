from __future__ import annotations

import unittest
from pathlib import Path


class StringingSummaryGapSheetsRemovedTests(unittest.TestCase):
    def test_legacy_gap_sheet_names_are_not_exported(self) -> None:
        source = Path("export_stringing_summary.py").read_text(encoding="utf-8")
        for sheet in (
            '"PO_FS_Gap"',
            '"PO_FS_Gap_Summary"',
            '"Erection_PO_Gap"',
            '"Erection_PO_Gap_Summary"',
        ):
            self.assertNotIn(sheet, source)

    def test_legacy_gap_builders_are_removed(self) -> None:
        source = Path("export_stringing_summary.py").read_text(encoding="utf-8")
        for symbol in (
            "def _build_gap_summary(",
            "def _build_po_fs_gap_table(",
            "def _build_erection_po_gap_table(",
            "def _build_erection_location_map(",
        ):
            self.assertNotIn(symbol, source)


if __name__ == "__main__":
    unittest.main()
