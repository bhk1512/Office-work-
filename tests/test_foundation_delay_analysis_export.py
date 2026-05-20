from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.foundation_delay_analysis import (
    build_foundation_delay_analysis_tables,
    write_foundation_delay_analysis_workbook,
)


class FoundationDelayAnalysisV2Tests(unittest.TestCase):
    def _build_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        foundation_completions = pd.DataFrame(
            [
                # TA 510
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "project_scope_key": "ta510::l1", "line_name": "L1", "source_type": "detail", "event_date": "2026-05-15", "location_no": "1/0"},
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "project_scope_key": "ta510::l1", "line_name": "L1", "source_type": "detail", "event_date": "2026-06-10", "location_no": "2/0"},
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "project_scope_key": "ta510::l1", "line_name": "L1", "source_type": "detail", "event_date": "2026-07-20", "location_no": "3/0"},
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "project_scope_key": "ta510::l1", "line_name": "L1", "source_type": "detail", "event_date": "2026-09-05", "location_no": "4/0"},
                # TB 501 split lines (should consolidate)
                {"project_code": "TB 501", "project_display": "TB 501 - 220kV", "project_scope_key": "tb501::220kv", "line_name": "220kV", "source_type": "detail", "event_date": "2026-04-01", "location_no": "220/1"},
                {"project_code": "TB 501", "project_display": "TB 501 - 132kV", "project_scope_key": "tb501::132kv", "line_name": "132kV", "source_type": "detail", "event_date": "2026-04-15", "location_no": "132/1"},
                # TA 602 (early lifecycle; large scope)
                {"project_code": "TA 602", "project_display": "TA 602", "project_scope_key": "ta602", "line_name": "", "source_type": "detail", "event_date": "2026-01-10", "location_no": "10/0"},
                {"project_code": "TA 602", "project_display": "TA 602", "project_scope_key": "ta602", "line_name": "", "source_type": "detail", "event_date": "2026-01-20", "location_no": "10/1"},
                # TA 700 (status-scope missing -> fallback)
                {"project_code": "TA 700", "project_display": "TA 700", "project_scope_key": "ta700", "line_name": "", "source_type": "detail", "event_date": "2026-03-03", "location_no": "70/0"},
                {"project_code": "TA 700", "project_display": "TA 700", "project_scope_key": "ta700", "line_name": "", "source_type": "detail", "event_date": "2026-03-07", "location_no": "70/1"},
            ]
        )
        foundation_coverage = pd.DataFrame(
            [
                {"project_code": "TA 510", "project_display": "TA 510", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed", "detail_completions": 4, "detail_rows": 4, "snapshot_rows": 0},
                {"project_code": "TB 501", "project_display": "TB 501", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed", "detail_completions": 2, "detail_rows": 2, "snapshot_rows": 0},
                {"project_code": "TA 602", "project_display": "TA 602", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed", "detail_completions": 2, "detail_rows": 2, "snapshot_rows": 0},
                {"project_code": "TA 700", "project_display": "TA 700", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed", "detail_completions": 2, "detail_rows": 2, "snapshot_rows": 0},
            ]
        )
        foundation_diagnostics = pd.DataFrame(
            [
                {"Project": "TA 510", "ParserMode": "template"},
                {"Project": "TB 501", "ParserMode": "header"},
                {"Project": "TA 602", "ParserMode": "template"},
                {"Project": "TA 700", "ParserMode": "template"},
            ]
        )
        source_daily = pd.DataFrame(
            [
                # TA 510
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "line_name": "L1", "location_no": "1/0", "start_date": "2026-05-20"},
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "line_name": "L1", "location_no": "2/0", "start_date": "2026-07-25"},
                {"project_code": "TA 510", "project_display": "TA 510 - L1", "line_name": "L1", "location_no": "4/0", "start_date": "2026-12-10"},
                # TB 501
                {"project_code": "TB 501", "project_display": "TB 501 - 220kV", "line_name": "220kV", "location_no": "220/1", "start_date": "2026-04-10"},
                {"project_code": "TB 501", "project_display": "TB 501 - 132kV", "line_name": "132kV", "location_no": "132/1", "start_date": "2026-08-20"},
                # TA 602 (one negative, one unmatched)
                {"project_code": "TA 602", "project_display": "TA 602", "line_name": "", "location_no": "10/0", "start_date": "2026-01-05"},
                # TA 700
                {"project_code": "TA 700", "project_display": "TA 700", "line_name": "", "location_no": "70/0", "start_date": "2026-03-20"},
                {"project_code": "TA 700", "project_display": "TA 700", "line_name": "", "location_no": "70/1", "start_date": "2026-04-05"},
            ]
        )
        progress_status_raw = pd.DataFrame(
            [
                # TA 510 latest snapshot duplicate rows: exact duplicates collapse to one
                {"project_code": "TA 510", "project_display": "TA 510", "project_scope_key": "ta510", "line_name": "", "activity_norm": "foundation", "activity_raw": "Foundation", "section_label": "Progress", "report_date": "2026-09-30", "quantity_primary": 10, "cumulative_progress": 4, "balance_progress": 6},
                {"project_code": "TA 510", "project_display": "TA 510", "project_scope_key": "ta510", "line_name": "", "activity_norm": "foundation", "activity_raw": "Foundation", "section_label": "Progress", "report_date": "2026-09-30", "quantity_primary": 10, "cumulative_progress": 4, "balance_progress": 6},
                # TB 501 latest snapshot distinct rows: both should sum
                {"project_code": "TB 501", "project_display": "TB 501 - 220kV", "project_scope_key": "tb501220kv", "line_name": "220kV", "activity_norm": "foundation", "activity_raw": "Foundation", "section_label": "220kV Line", "report_date": "2026-09-30", "quantity_primary": 40, "cumulative_progress": 20, "balance_progress": 20},
                {"project_code": "TB 501", "project_display": "TB 501 - 132kV", "project_scope_key": "tb501132kv", "line_name": "132kV", "activity_norm": "foundation", "activity_raw": "Foundation", "section_label": "132kV Line", "report_date": "2026-09-30", "quantity_primary": 20, "cumulative_progress": 8, "balance_progress": 12},
                # TA 602: large scope but small completions -> only early phase realized
                {"project_code": "TA 602", "project_display": "TA 602", "project_scope_key": "ta602", "line_name": "", "activity_norm": "foundation", "activity_raw": "Foundation", "section_label": "Progress", "report_date": "2026-09-30", "quantity_primary": 20, "cumulative_progress": 2, "balance_progress": 18},
            ]
        )
        return source_daily, foundation_completions, foundation_coverage, foundation_diagnostics, progress_status_raw

    def test_build_tables_scope_phase_buckets_and_rollups(self) -> None:
        inputs = self._build_inputs()
        tables = build_foundation_delay_analysis_tables(*inputs)

        expected_sheets = {
            "Delay Phase - Project",
            "Delay Phase - Series",
            "Delay Phase - Ownership",
            "Delay Buckets - Project",
            "Delay Buckets - Series",
            "Delay Buckets - Ownership",
            "Delay Coverage",
            "Delay Anomalies",
            "Scope Snapshot",
        }
        self.assertEqual(set(tables.keys()), expected_sheets)

        scope_snapshot = tables["Scope Snapshot"]
        ta510_scope = scope_snapshot[scope_snapshot["Project"].astype(str).str.strip().eq("TA 510")]
        self.assertFalse(ta510_scope.empty)
        self.assertEqual(float(ta510_scope.iloc[0]["Scope Total"]), 10.0)
        self.assertEqual(int(ta510_scope.iloc[0]["Duplicate Rows Dropped"]), 1)

        tb501_scope = scope_snapshot[scope_snapshot["Project"].astype(str).str.strip().eq("TB 501")]
        self.assertFalse(tb501_scope.empty)
        self.assertEqual(float(tb501_scope.iloc[0]["Scope Total"]), 60.0)

        ta700_scope = scope_snapshot[scope_snapshot["Project"].astype(str).str.strip().eq("TA 700")]
        self.assertFalse(ta700_scope.empty)
        self.assertEqual(str(ta700_scope.iloc[0]["Scope Source"]).strip(), "fallback_foundation_done_count")

        phase_project = tables["Delay Phase - Project"]
        ta602_phases = set(phase_project[phase_project["Project"].astype(str).str.strip().eq("TA 602")]["Phase"].astype(str))
        self.assertEqual(ta602_phases, {"0-20"})

        ta510_phase_rows = phase_project[phase_project["Project"].astype(str).str.strip().eq("TA 510")]
        self.assertTrue((ta510_phase_rows["Monsoon Overlap"].astype(str).str.strip() == "Yes").any())
        self.assertTrue((pd.to_numeric(ta510_phase_rows["Monsoon Foundation Count"], errors="coerce").fillna(0) > 0).any())

        bucket_project = tables["Delay Buckets - Project"]
        ta510_buckets = bucket_project[bucket_project["Project"].astype(str).str.strip().eq("TA 510")]
        self.assertFalse(ta510_buckets.empty)
        bucket_labels = set(ta510_buckets["Bucket"].astype(str))
        self.assertIn("0-30", bucket_labels)
        self.assertIn("31-60", bucket_labels)
        self.assertIn("91-120", bucket_labels)

        phase_series = tables["Delay Phase - Series"]
        self.assertTrue((phase_series["Series"].astype(str).str.strip() == "5xx").any())
        self.assertTrue((phase_series["Series"].astype(str).str.strip() == "6xx").any())

        phase_ownership = tables["Delay Phase - Ownership"]
        self.assertTrue((phase_ownership["Ownership"].astype(str).str.strip() == "Government").any())
        self.assertTrue((phase_ownership["Ownership"].astype(str).str.strip() == "Private").any())

        # Consolidated project-level rollup: TB 501 should appear as one project.
        tb501_project_rows = phase_project[phase_project["Project"].astype(str).str.strip().eq("TB 501")]
        self.assertFalse(tb501_project_rows.empty)
        self.assertFalse((phase_project["Project"].astype(str).str.contains("TB 501 -", regex=False)).any())

    def test_workbook_writer_creates_expected_tabs(self) -> None:
        inputs = self._build_inputs()
        tables = build_foundation_delay_analysis_tables(*inputs)
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "Foundation_Delay_Analysis.xlsx"
            write_foundation_delay_analysis_workbook(out, tables)
            self.assertTrue(out.exists())
            with pd.ExcelFile(out) as xl:
                self.assertEqual(
                    set(xl.sheet_names),
                    {
                        "Delay Phase - Project",
                        "Delay Phase - Series",
                        "Delay Phase - Ownership",
                        "Delay Buckets - Project",
                        "Delay Buckets - Series",
                        "Delay Buckets - Ownership",
                        "Delay Coverage",
                        "Delay Anomalies",
                        "Scope Snapshot",
                    },
                )


if __name__ == "__main__":
    unittest.main()

