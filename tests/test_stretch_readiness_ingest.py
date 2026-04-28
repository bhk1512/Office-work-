from __future__ import annotations

import unittest

import pandas as pd
from openpyxl import Workbook

from dashboard import stretch_readiness_ingest as stretch_ingest


class StretchReadinessIngestTests(unittest.TestCase):
    def test_extract_guardrails(self) -> None:
        wb = Workbook()
        ws = wb.active
        ws.title = "Template"
        ws["A1"] = "Guardrails"
        ws["A2"] = "required_tokens"
        ws["B2"] = "loc; final checking"
        ws["A3"] = "header_rows"
        ws["B3"] = "1"
        ws["A4"] = "To Map"
        ws["A5"] = "stretch_identifier"

        parsed = stretch_ingest._extract_guardrails_stretch(ws)  # type: ignore[attr-defined]
        self.assertEqual(parsed.get("required_tokens"), "loc; final checking")
        self.assertEqual(parsed.get("header_rows"), "1")

    def test_parse_tb507_both_required_rule(self) -> None:
        df = pd.DataFrame(
            [
                ["Client", "", "", ""],
                ["SL No", "Loc No", "Final Checking", "Tack Welding"],
                [1, "65/0", "DONE", "DONE"],
                [2, "65/1", "DONE", ""],
                [3, "66/0", "ROW", ""],
            ]
        )
        guardrails = {
            "required_tokens": "loc; final checking; tack welding",
            "header_rows": "1",
            "readiness_rule": "both_required",
        }
        template_map = {
            1: "stretch_identifier",
            2: "final_check_raw",
            3: "tack_welding_raw",
        }

        result = stretch_ingest._parse_stretch_sheet_dataframe(  # type: ignore[attr-defined]
            df,
            guardrails=guardrails,
            template_map=template_map,
            project_code="TB 507",
            project_display="TB 507 - MAIN",
            project_scope_key="tb507main",
            line_name="MAIN",
            line_name_source="config",
            source_file="TB 507 [MAIN] - DPR - 2026-04-24.xlsx",
            source_sheet="Final Check In And Tack Welding",
            configured_sheet="Final Check In And Tack Welding",
            template_sheet="TB 507 Stretch",
            report_date="2026-04-24",
        )
        self.assertEqual(result.parse_status, "OK")
        states = set(result.data["readiness_state"].astype(str).tolist())
        self.assertIn("READY", states)
        self.assertIn("PARTIAL", states)
        self.assertIn("NOT_READY", states)

    def test_detect_manpower_signal(self) -> None:
        frame = pd.DataFrame(
            {
                "From AP": ["1/0", "2/0"],
                "Gang Strength": [12, None],
                "Section Readiness": ["Ready", "Ready"],
            }
        )
        signal = stretch_ingest._detect_manpower_signal_from_frame(frame)  # type: ignore[attr-defined]
        self.assertEqual(signal.get("signal_type"), "PRESENT_WITH_VALUES")
        self.assertTrue(signal.get("readiness_present"))


if __name__ == "__main__":
    unittest.main()
