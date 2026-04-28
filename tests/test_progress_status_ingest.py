from __future__ import annotations

import unittest

import pandas as pd
from openpyxl import Workbook

from dashboard import progress_status_ingest as status_ingest


class ProgressStatusIngestTests(unittest.TestCase):
    def test_extract_guardrails(self) -> None:
        wb = Workbook()
        ws = wb.active
        ws.title = "Template"
        ws["A1"] = "Guardrails"
        ws["A2"] = "block_anchor"
        ws["B2"] = "Progress Status"
        ws["A3"] = "header_rows"
        ws["B3"] = "2"
        ws["A4"] = "To Map"
        ws["A5"] = "activity_raw"

        parsed = status_ingest._extract_guardrails(ws)  # type: ignore[attr-defined]
        self.assertEqual(parsed.get("block_anchor"), "Progress Status")
        self.assertEqual(parsed.get("header_rows"), "2")

    def test_detect_header_single_two_three_row(self) -> None:
        single = pd.DataFrame(
            [
                ["Sr", "Activity", "Cumulative Progress", "Balance Progress"],
                ["1", "Foundation", "120", "10"],
            ]
        )
        hrow, hspan, _ = status_ingest._detect_header(  # type: ignore[attr-defined]
            single,
            start_row=0,
            end_row=len(single.index) - 1,
            required_tokens=["activity", "cumulative", "balance"],
            header_rows_options=[1],
        )
        self.assertEqual((hrow, hspan), (0, 1))

        two_row = pd.DataFrame(
            [
                ["", "Activity", "Quantity", "", "Cumulative till Last Month", "", "Plan for the Month", "Progress for the Month", "Cumulative Progress", "Balance Progress"],
                ["", "", "LOA", "Estimated", "L2", "Progress", "L2", "Till Date", "", ""],
                ["", "Foundation", "322", "330", "322", "228", "40", "4", "232", "98"],
            ]
        )
        hrow, hspan, _ = status_ingest._detect_header(  # type: ignore[attr-defined]
            two_row,
            start_row=0,
            end_row=len(two_row.index) - 1,
            required_tokens=["activity", "cumulative", "balance"],
            header_rows_options=[2],
        )
        self.assertEqual((hrow, hspan), (0, 2))

        three_row = pd.DataFrame(
            [
                ["", "PARTICULARS", "UOM", "Total", "Comp", "Balance", "April'26", "", "WIP", "Cumulative"],
                ["", "", "", "Qty", "Till", "as on", "Plan", "Actual", "", ""],
                ["", "", "", "", "Mar'26", "1 Apr'26", "", "", "", ""],
                ["", "Foundation", "Nos", "190", "137", "53", "6", "2", "5", "137"],
            ]
        )
        hrow, hspan, _ = status_ingest._detect_header(  # type: ignore[attr-defined]
            three_row,
            start_row=0,
            end_row=len(three_row.index) - 1,
            required_tokens=["particulars", "comp", "balance", "cumulative"],
            header_rows_options=[3],
        )
        self.assertEqual((hrow, hspan), (0, 3))

    def test_parse_with_section_split(self) -> None:
        df = pd.DataFrame(
            [
                ["Progress of 220kV Line", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["Sr. No.", "Activity", "UoM", "Total Quantity", "As per L2", "", "Cumm. Progress up to last Month", "", "Progress of Current Month (April'26)", "", "", "Cumulative till current month", "", "", "Resources Status", "", "", "Critical Issues"],
                ["", "", "", "", "Start Date", "End Date", "L2", "Actual", "Plan", "Today", "Progress in Month", "Target (L2)", "Completed", "Balance", "Targeted Gang", "Available Gang", "Shortfall", ""],
                ["1", "Foundation", "Nos.", "199", "", "", "199", "195", "4", "1", "2", "199", "197", "2", "1", "1", "0", ""],
                ["5", "Tower Erection", "Nos.", "199", "", "", "199", "195", "4", "0", "0", "199", "195", "4", "0", "0", "0", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["Progress of 132kV Line", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["Sr. No.", "Activity", "UoM", "Total Quantity", "As per L2", "", "Cumm. Progress up to last Month", "", "Progress of Current Month (April'26)", "", "", "Cumulative till current month", "", "", "Resources Status", "", "", "Critical Issues"],
                ["", "", "", "", "Start Date", "End Date", "L2", "Actual", "Plan", "Today", "Progress in Month", "Target (L2)", "Completed", "Balance", "Targeted Gang", "Available Gang", "Shortfall", ""],
                ["1", "Foundation", "Nos.", "39", "", "", "39", "31", "5", "1", "3", "39", "34", "5", "2", "2", "0", ""],
                ["7", "Stringing", "Kms", "9", "", "", "9", "5", "1.4", "0", "0.3", "9", "5.3", "3.7", "0", "0", "0", ""],
            ]
        )

        guardrails = {
            "section_start_contains": "Progress of",
            "section_label_regex": r"Progress of\\s*(.*?)\\s*$",
            "header_rows": "2",
            "required_tokens": "activity; cumm; progress; balance",
            "activity_allowlist": "foundation; tower erection; stringing",
        }
        template_map = {
            1: "activity_raw",
            3: "quantity_estimated_or_total",
            7: "cumulative_last_month",
            8: "plan_for_month",
            9: "today_progress",
            10: "progress_for_month",
            12: "cumulative_progress",
            13: "balance_progress",
            15: "gangs_working",
            17: "remarks",
        }
        result = status_ingest._parse_status_sheet_dataframe(  # type: ignore[attr-defined]
            df,
            guardrails=guardrails,
            template_map=template_map,
            project_code="TB 501",
            project_display="TB 501",
            project_scope_key="tb501",
            line_name="",
            line_name_source="",
            source_file="TB 501 - DPR.xlsx",
            source_sheet="Summary",
            configured_sheet="Summary",
            template_sheet="TB 501 Status",
        )
        self.assertEqual(result.parse_status, "OK")
        self.assertGreaterEqual(result.rows_emitted, 4)
        self.assertGreaterEqual(result.sections_detected, 2)


if __name__ == "__main__":
    unittest.main()
