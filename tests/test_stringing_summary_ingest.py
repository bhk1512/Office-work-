from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard import stringing_summary_ingest as summary_ingest


class StringingSummaryIngestTests(unittest.TestCase):
    def test_build_status_activity_fact_maps_groups(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 413",
                    "project_display": "TA 413",
                    "project_scope_key": "ta413",
                    "line_name": "",
                    "source_file": "TA 413 - DPR - 2026-04-14.xlsx",
                    "activity_raw": "Tower Erection (Nos)",
                    "activity_norm": "tower_erection",
                    "quantity_primary": 322,
                    "cumulative_progress": 184,
                    "balance_progress": 138,
                }
            ]
        )
        out = summary_ingest._build_status_activity_fact(raw)  # type: ignore[attr-defined]
        self.assertEqual(len(out.index), 1)
        self.assertEqual(str(out.loc[0, "activity_group"]), "Tower Erection")
        self.assertTrue(bool(out.loc[0, "core_activity"]))
        self.assertEqual(str(out.loc[0, "month"])[:7], "2026-04")

    def test_build_manpower_productivity_fact_uses_compiled_values(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "project": "TA 418",
                    "date": "2026-04-25",
                    "gang_name": "Alpha",
                    "from_ap": "1/0",
                    "to_ap": "2/0",
                    "method": "TSE",
                    "section_readiness": "Ready",
                    "daily_km": 0.8,
                    "po_km": 0.8,
                }
            ]
        )
        compiled = pd.DataFrame(
            [
                {
                    "Project Code": "TA 418",
                    "Project Name": "TA 418",
                    "From AP": "1/0",
                    "To AP": "2/0",
                    "Gang Name": "Alpha",
                    "Gang Strength": 12,
                    "No of Fitters": 5,
                }
            ]
        )
        audit = pd.DataFrame(
            [
                {
                    "project_code": "TA 418",
                    "signal_type": "PRESENT_WITH_VALUES",
                    "status": "PRESENT_WITH_VALUES",
                    "expected_manpower": "yes",
                }
            ]
        )

        out = summary_ingest._build_manpower_productivity_fact(daily, compiled, audit)  # type: ignore[attr-defined]
        self.assertEqual(len(out.index), 1)
        self.assertEqual(float(out.loc[0, "manpower_gang_strength"]), 12.0)
        self.assertEqual(float(out.loc[0, "manpower_fitters"]), 5.0)
        self.assertEqual(str(out.loc[0, "availability"]), "AVAILABLE")

    def test_compile_writes_expected_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Parquets"
            erection_root = root / "Erection"
            stringing_root = root / "Stringing"
            status_root = root / "ProgressStatus"
            stretch_root = root / "StretchReadiness"
            for folder in (erection_root, stringing_root, status_root, stretch_root):
                folder.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "project": "TA 418",
                        "date": "2026-04-25",
                        "gang_name": "Alpha",
                        "from_ap": "1/0",
                        "to_ap": "2/0",
                        "method": "TSE",
                        "section_readiness": "Ready",
                        "daily_km": 0.8,
                        "po_km": 0.8,
                    }
                ]
            ).to_parquet(stringing_root / "StringingDaily.parquet", index=False)

            pd.DataFrame(
                [
                    {
                        "project_code": "TA 418",
                        "project_display": "TA 418",
                        "project_scope_key": "ta418",
                        "line_name": "",
                        "source_file": "TA 418 - DPR - 2026-04-25.xlsx",
                        "activity_raw": "Tower Erection (Nos)",
                        "activity_norm": "tower_erection",
                        "quantity_primary": 100,
                        "cumulative_progress": 60,
                        "balance_progress": 40,
                    }
                ]
            ).to_parquet(status_root / "RawData.parquet", index=False)

            pd.DataFrame(
                [
                    {
                        "project_code": "TA 418",
                        "project_display": "TA 418",
                        "project_scope_key": "ta418",
                        "line_name": "",
                        "report_date": "2026-04-25",
                        "stretch_identifier": "AP-1 - AP-2",
                        "from_ap": "1/0",
                        "to_ap": "2/0",
                        "length_km": 0.8,
                        "readiness_state": "READY",
                    }
                ]
            ).to_parquet(stretch_root / "RawData.parquet", index=False)

            out_path = root / "StringingSummary" / "StringingSummary_Output.xlsx"
            compiled = summary_ingest.compile_stringing_summary_to_workbook(erection_root, out_path)
            self.assertTrue(compiled.exists())
            with pd.ExcelFile(compiled) as xl:
                for sheet in summary_ingest.STRINGING_SUMMARY_SHEETS:
                    self.assertIn(sheet, xl.sheet_names)


if __name__ == "__main__":
    unittest.main()
