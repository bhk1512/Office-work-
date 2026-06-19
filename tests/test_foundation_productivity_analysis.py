from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import export_foundation_productivity_analysis as cli
from dashboard.foundation_productivity_analysis import (
    SHEET_ORDER,
    build_foundation_productivity_tables,
    write_foundation_productivity_workbook,
)


def _sample_completions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "project_code": "TA 510",
                "project_display": "TA 510",
                "line_name": "",
                "location_no": "1/0",
                "gang_name": "Gang A",
                "start_date": pd.Timestamp("2026-05-01"),
                "event_date": pd.Timestamp("2026-05-05"),
                "source_type": "detail",
                "quality_flag": "detail_date",
                "source_file": "TA 510.xlsx",
                "source_sheet": "FDN",
            },
            {
                "project_code": "TA 510",
                "project_display": "TA 510",
                "line_name": "",
                "location_no": "1/1",
                "gang_name": "Gang A",
                "start_date": pd.Timestamp("2026-05-10"),
                "event_date": pd.Timestamp("2026-05-15"),
                "source_type": "detail",
                "quality_flag": "detail_date",
                "source_file": "TA 510.xlsx",
                "source_sheet": "FDN",
            },
            {
                "project_code": "TA 510",
                "project_display": "TA 510",
                "line_name": "",
                "location_no": "1/2",
                "gang_name": "",
                "start_date": pd.Timestamp("2026-05-20"),
                "event_date": pd.Timestamp("2026-05-20"),
                "source_type": "detail",
                "quality_flag": "detail_date",
                "source_file": "TA 510.xlsx",
                "source_sheet": "FDN",
            },
            {
                "project_code": "TA 510",
                "project_display": "TA 510",
                "line_name": "",
                "location_no": "1/3",
                "gang_name": "Gang B",
                "start_date": pd.Timestamp("2026-06-10"),
                "event_date": pd.Timestamp("2026-06-09"),
                "source_type": "detail",
                "quality_flag": "detail_date",
                "source_file": "TA 510.xlsx",
                "source_sheet": "FDN",
            },
            {
                "project_code": "TB 501",
                "project_display": "TB 501",
                "line_name": "220kV",
                "location_no": "2/0",
                "gang_name": "Gang A",
                "start_date": pd.Timestamp("2026-05-01"),
                "event_date": pd.Timestamp("2026-05-03"),
                "source_type": "detail",
                "quality_flag": "detail_date",
                "source_file": "TB 501.xlsx",
                "source_sheet": "Progress-220kV",
            },
            {
                "project_code": "TA 510",
                "project_display": "TA 510",
                "line_name": "",
                "location_no": "",
                "gang_name": "",
                "start_date": pd.NaT,
                "event_date": pd.Timestamp("2026-05-31"),
                "source_type": "snapshot_fallback",
                "quality_flag": "snapshot_cumulative",
                "source_file": "TA 510.xlsx",
                "source_sheet": "Status",
            },
        ]
    )


class FoundationProductivityAnalysisTests(unittest.TestCase):
    def test_tables_use_active_gang_months_and_exclude_snapshots(self) -> None:
        mapping = pd.DataFrame(
            [
                {"Project": "TA 510", "PCH": "PCH A"},
                {"Project": "TB 501", "PCH": "PCH B"},
            ]
        )
        coverage = pd.DataFrame(
            [{"project_code": "TB 408", "status": "NO_TARGET_SHEET", "reason": "Missing foundation sheet"}]
        )

        tables = build_foundation_productivity_tables(
            _sample_completions(),
            foundation_coverage=coverage,
            pch_mapping=mapping,
        )

        summary = dict(zip(tables["Portfolio Summary"]["Metric"], tables["Portfolio Summary"]["Value"]))
        self.assertEqual(int(summary["Total Foundations"]), 5)
        self.assertEqual(int(summary["Snapshot Rows Excluded"]), 1)
        self.assertEqual(int(summary["Active Gang-Months"]), 2)
        self.assertEqual(float(summary["Avg Foundations / Active Gang-Month"]), 2.0)
        self.assertEqual(float(summary["Gang Name Coverage %"]), 80.0)

        details = tables["Foundation Details"]
        loc_10 = details[details["Location No"].eq("1/0")].iloc[0]
        self.assertEqual(int(loc_10["Duration Days"]), 5)
        loc_13 = details[details["Location No"].eq("1/3")].iloc[0]
        self.assertTrue(pd.isna(loc_13["Duration Days"]))
        self.assertEqual(loc_13["Duration Status"], "Invalid Negative Duration")

        project_month = tables["Project Monthly Trend"]
        ta510_may = project_month[
            project_month["Project"].eq("TA 510") & project_month["Month"].eq("2026-05")
        ].iloc[0]
        self.assertEqual(int(ta510_may["Foundations Completed"]), 3)
        self.assertEqual(int(ta510_may["Unique Gangs"]), 1)
        self.assertEqual(float(ta510_may["Avg Foundations / Active Gang-Month"]), 2.0)
        self.assertFalse(tables["Gang Monthly Productivity"]["Gang"].astype(str).eq("Unassigned").any())
        self.assertFalse(tables["Foundation Insights"].empty)
        self.assertIn("Gang Productivity Benchmark", set(tables["Foundation Insights"]["Theme"].astype(str)))

        coverage_table = tables["Data Coverage"]
        self.assertTrue(coverage_table["Project"].astype(str).str.contains("TB 408").any())

    def test_month_filter_and_workbook_export(self) -> None:
        tables = build_foundation_productivity_tables(
            _sample_completions(),
            start_month="2026-06",
            end_month="2026-06",
        )
        summary = dict(zip(tables["Portfolio Summary"]["Metric"], tables["Portfolio Summary"]["Value"]))
        self.assertEqual(int(summary["Total Foundations"]), 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "foundation.xlsx"
            write_foundation_productivity_workbook(output, tables)
            self.assertTrue(output.exists())
            with pd.ExcelFile(output) as xl:
                self.assertEqual(set(SHEET_ORDER), set(xl.sheet_names))

    def test_cli_resolves_relative_output_under_productivity_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(cli, "PRODUCTIVITY_ROOT", root):
                resolved = cli._resolve_output_path(Path("Foundation_Productivity_Analysis.xlsx"))
            self.assertEqual(resolved, (root / "Foundation_Productivity_Analysis.xlsx").resolve())


if __name__ == "__main__":
    unittest.main()
