from __future__ import annotations

import unittest

import pandas as pd

from dashboard.stringing_analytics import _build_erection_po_gap_table, _normalize_stringing_compiled


class StringingAnalyticsEsGapTests(unittest.TestCase):
    def _compiled(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        return _normalize_stringing_compiled(pd.DataFrame(rows))

    def _erection(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_location_nos_ok_uses_location_mode(self) -> None:
        compiled = self._compiled(
            [
                {
                    "project_name": "TA 501",
                    "From AP": "65/0",
                    "To AP": "65/3",
                    "P/O Starting Date": "2026-05-10",
                    "location nos": "65/1,65/2",
                }
            ]
        )
        erection = self._erection(
            [
                {"project_key_norm": "ta501", "location_no": "65/0", "completion_date": "2026-05-01"},
                {"project_key_norm": "ta501", "location_no": "65/1", "completion_date": "2026-05-02"},
                {"project_key_norm": "ta501", "location_no": "65/2", "completion_date": "2026-05-03"},
                {"project_key_norm": "ta501", "location_no": "65/3", "completion_date": "2026-05-04"},
            ]
        )
        out = _build_erection_po_gap_table(compiled, erection)
        self.assertEqual(str(out.loc[0, "lag_inference_mode"]), "LOCATION_NOS")
        self.assertFalse(bool(out.loc[0, "lag_fallback_used"]))
        self.assertEqual(str(out.loc[0, "location_parse_status"]), "OK")
        self.assertEqual(int(out.loc[0, "unmatched_location_count"]), 0)
        self.assertEqual(int(out.loc[0, "gap_days"]), 6)

    def test_missing_location_nos_uses_alphabetic_fallback(self) -> None:
        compiled = self._compiled(
            [
                {
                    "project_name": "TA 501",
                    "From AP": "65/0",
                    "To AP": "65/2",
                    "P/O Starting Date": "2026-05-10",
                    "location nos": "",
                }
            ]
        )
        erection = self._erection(
            [
                {"project_key_norm": "ta501", "location_no": "65/0", "completion_date": "2026-05-01"},
                {"project_key_norm": "ta501", "location_no": "65/1", "completion_date": "2026-05-02"},
                {"project_key_norm": "ta501", "location_no": "65/2", "completion_date": "2026-05-03"},
            ]
        )
        out = _build_erection_po_gap_table(compiled, erection)
        self.assertEqual(str(out.loc[0, "lag_inference_mode"]), "ALPHABETIC_FALLBACK")
        self.assertTrue(bool(out.loc[0, "lag_fallback_used"]))
        self.assertEqual(str(out.loc[0, "location_parse_status"]), "EMPTY")
        self.assertEqual(int(out.loc[0, "gap_days"]), 7)

    def test_partial_parse_uses_hybrid_fallback(self) -> None:
        compiled = self._compiled(
            [
                {
                    "project_name": "TA 501",
                    "From AP": "65/0",
                    "To AP": "65/3",
                    "P/O Starting Date": "2026-05-10",
                    "location nos": "65/1,BAD-TOKEN,2",
                }
            ]
        )
        erection = self._erection(
            [
                {"project_key_norm": "ta501", "location_no": "65/0", "completion_date": "2026-05-01"},
                {"project_key_norm": "ta501", "location_no": "65/1", "completion_date": "2026-05-02"},
                {"project_key_norm": "ta501", "location_no": "65/2", "completion_date": "2026-05-03"},
                {"project_key_norm": "ta501", "location_no": "65/3", "completion_date": "2026-05-04"},
            ]
        )
        out = _build_erection_po_gap_table(compiled, erection)
        self.assertEqual(str(out.loc[0, "lag_inference_mode"]), "HYBRID_PARTIAL_FALLBACK")
        self.assertTrue(bool(out.loc[0, "lag_fallback_used"]))
        self.assertEqual(str(out.loc[0, "location_parse_status"]), "PARTIAL_PARSE")
        self.assertEqual(int(out.loc[0, "gap_days"]), 6)

    def test_gap_missing_can_still_mark_fallback(self) -> None:
        compiled = self._compiled(
            [
                {
                    "project_name": "TA 501",
                    "From AP": "",
                    "To AP": "",
                    "P/O Starting Date": "2026-05-10",
                    "location nos": "",
                }
            ]
        )
        erection = self._erection(
            [
                {"project_key_norm": "ta501", "location_no": "65/0", "completion_date": "2026-05-01"},
            ]
        )
        out = _build_erection_po_gap_table(compiled, erection)
        self.assertEqual(str(out.loc[0, "lag_inference_mode"]), "ALPHABETIC_FALLBACK")
        self.assertTrue(bool(out.loc[0, "lag_fallback_used"]))
        self.assertTrue(pd.isna(out.loc[0, "gap_days"]))


if __name__ == "__main__":
    unittest.main()
