from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard import stringing_ingest as ingest
from dashboard.stringing import add_length_units
from pipeline_runner import _apply_template_if_improves, _build_may_exclusion_rows, _coerce_excel_date_series


class StringingDprCaptureFixesTests(unittest.TestCase):
    def test_configured_stringing_falls_back_to_only_compiled_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook = Path(temp_dir) / "TB 408 - DPR - 2026-05-11.xlsx"
            frame = pd.DataFrame(
                [
                    ["From AP", "To AP", "Method", "P/O Starting Date", "F/S/ Completion Date", "Length"],
                    ["14A/0", "15/0", "TSE", "2026-04-25", "2026-05-04", 1.192],
                ]
            )
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="Stringing Compiled", header=False, index=False)

            result = ingest.load_stringing_sheet_frame(workbook, configured_sheet_name="Stringing")

            self.assertEqual(result.resolved_sheet, "Stringing Compiled")
            self.assertIn("Configured sheet 'Stringing' not found", result.fallback_note)
            self.assertEqual(len(result.frame.index), 1)

    def test_sheet_aware_template_selection_prefers_220kv(self) -> None:
        options = [
            ({0: "From AP"}, "TB 501 132kV Stringing"),
            ({0: "From AP", 1: "To AP"}, "TB 501 220kV Stringing"),
        ]

        _, selected_sheet = ingest.select_template_map_for_sheet(
            options,
            configured_sheet_name="Stringing-220kV",
            line_name="220kV",
        )

        self.assertEqual(selected_sheet, "TB 501 220kV Stringing")

    def test_template_mapping_skipped_when_headers_are_already_valid(self) -> None:
        frame = pd.DataFrame(
            {
                "From AP": ["14A/0"],
                "To AP": ["15/0"],
                "Method": ["TSE"],
                "Section Readiness": ["Ready"],
                "P/O Starting Date": ["2026-04-25"],
                "P/O Completion Date": ["2026-05-04"],
                "P/O": [1.192],
                "F/S Starting Date": ["2026-05-02"],
                "F/S/ Completion Date": ["2026-05-04"],
                "Length": [1.192],
                "Gang Name": ["Crew A"],
            }
        )
        bad_template = {0: "To AP", 1: "From AP"}

        output, normalized, report, _classification, changes, applied = _apply_template_if_improves(frame, bad_template)

        self.assertFalse(applied)
        self.assertEqual(changes, [])
        self.assertEqual(list(output.columns), list(frame.columns))
        self.assertTrue(bool(report["normalized_columns_ok"]))
        self.assertEqual(normalized.loc[0, "from_ap"], "14A/0")

    def test_length_unit_inference_ignores_total_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "from_ap": ["14A/0", pd.NA],
                "to_ap": ["15/0", "Total"],
                "length_m": [1.192, 191.391922],
                "status": ["Balance", pd.NA],
            }
        )

        out, metrics = add_length_units(frame)

        self.assertEqual(metrics["length_unit"], "km")
        self.assertAlmostEqual(float(out.loc[0, "length_km"]), 1.192, places=6)
        self.assertAlmostEqual(float(metrics["total_length_km"]), 1.192, places=6)

    def test_may_exclusions_keep_incomplete_rows_out_of_completed_daily(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "project_code": "TA 421",
                    "project_name": "TA 421",
                    "from_ap": "42/0",
                    "to_ap": "43/0",
                    "length_m": 3.949,
                    "length_km": 3.949,
                    "po_start_date": "2026-04-25",
                    "fs_complete_date": "2026-05-08",
                },
                {
                    "project_code": "TA 421",
                    "project_name": "TA 421",
                    "from_ap": "35/0",
                    "to_ap": "36/0",
                    "length_m": 1.858,
                    "length_km": 1.858,
                    "po_start_date": "2026-05-18",
                    "fs_starting_date": "2026-05-21",
                    "fs_complete_date": pd.NA,
                },
            ]
        )

        out = _build_may_exclusion_rows(
            frame,
            target_month=pd.Period("2026-05", freq="M"),
            workbook="TA 421 - DPR - 2026-05-26.xlsx",
            project="TA421",
            configured_sheet="Stringing Compiled",
            resolved_sheet="Stringing Compiled",
            line_name="",
        )

        self.assertEqual(len(out.index), 1)
        self.assertEqual(str(out.loc[0, "from_ap"]), "35/0")
        self.assertIn("Missing/invalid F/S completion date", str(out.loc[0, "exclusion_reason"]))

    def test_parquet_date_coercion_handles_mixed_dotted_dpr_dates(self) -> None:
        dates = pd.Series(["06.05.2024", "10.05.24", "25.12.25", "01.12.245", 45500])

        parsed = _coerce_excel_date_series(dates)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2024-05-06"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2024-05-10"))
        self.assertEqual(parsed.iloc[2], pd.Timestamp("2025-12-25"))
        self.assertTrue(pd.isna(parsed.iloc[3]))
        self.assertEqual(parsed.iloc[4], pd.Timestamp("2024-07-27"))


if __name__ == "__main__":
    unittest.main()
