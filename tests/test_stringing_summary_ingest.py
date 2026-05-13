from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard import stringing_summary_ingest as summary_ingest


class StringingSummaryIngestTests(unittest.TestCase):
    def test_parse_report_date_uses_fallback_for_all_rows(self) -> None:
        fallback = pd.Series(
            [
                "TA 414 - DPR - 2026-03-26.xlsx",
                "TA 418 - DPR - 2026-04-29.xlsx",
                "TA 505 - DPR - 2026-04-17.xlsx",
            ]
        )
        parsed = summary_ingest._parse_report_date(None, fallback)  # type: ignore[attr-defined]
        self.assertEqual(int(parsed.notna().sum()), 3)
        self.assertEqual(parsed.dt.to_period("M").nunique(), 2)

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

    def test_final_sag_preferred_when_rough_and_final_present(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 421",
                    "project_display": "TA 421",
                    "project_scope_key": "ta421",
                    "line_name": "",
                    "section_label": "Progress Status",
                    "source_file": "TA 421 - DPR - 2026-05-10.xlsx",
                    "source_sheet": "Master Sheet",
                    "configured_sheet": "Master Sheet",
                    "template_sheet": "TA 421 Status",
                    "stringing_resolution_policy": "prefer_final_else_stringing",
                    "report_date": "2026-05-10",
                    "activity_raw": "Stringing -R/S",
                    "activity_norm": "stringing",
                    "cumulative_progress": 111.099,
                },
                {
                    "project_code": "TA 421",
                    "project_display": "TA 421",
                    "project_scope_key": "ta421",
                    "line_name": "",
                    "section_label": "Progress Status",
                    "source_file": "TA 421 - DPR - 2026-05-10.xlsx",
                    "source_sheet": "Master Sheet",
                    "configured_sheet": "Master Sheet",
                    "template_sheet": "TA 421 Status",
                    "stringing_resolution_policy": "prefer_final_else_stringing",
                    "report_date": "2026-05-10",
                    "activity_raw": "Stringing -F/S",
                    "activity_norm": "stringing",
                    "cumulative_progress": 111.099,
                },
            ]
        )
        status = summary_ingest._build_status_activity_fact(raw)  # type: ignore[attr-defined]
        self.assertEqual(int(len(status.index)), 1)
        self.assertEqual(str(status.loc[0, "activity_group"]), "Stringing")
        self.assertAlmostEqual(float(status.loc[0, "cumulative_progress"]), 111.099, places=3)

    def test_final_sag_row_used_for_stringing_snapshot(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 505",
                    "project_display": "TA 505",
                    "project_scope_key": "ta505",
                    "line_name": "",
                    "section_label": "Progress",
                    "source_file": "TA 505 - DPR - 2026-05-10.xlsx",
                    "source_sheet": "DPR-Summary",
                    "configured_sheet": "DPR-Summary",
                    "template_sheet": "TA 505 Status",
                    "stringing_resolution_policy": "prefer_final_else_stringing",
                    "report_date": "2026-05-10",
                    "activity_raw": "Stringing (Rough Sag)",
                    "activity_norm": "stringing",
                    "cumulative_progress": 14.973,
                },
                {
                    "project_code": "TA 505",
                    "project_display": "TA 505",
                    "project_scope_key": "ta505",
                    "line_name": "",
                    "section_label": "Progress",
                    "source_file": "TA 505 - DPR - 2026-05-10.xlsx",
                    "source_sheet": "DPR-Summary",
                    "configured_sheet": "DPR-Summary",
                    "template_sheet": "TA 505 Status",
                    "stringing_resolution_policy": "prefer_final_else_stringing",
                    "report_date": "2026-05-10",
                    "activity_raw": "Stringing (Final Sag)",
                    "activity_norm": "final_sag",
                    "cumulative_progress": 14.006,
                },
            ]
        )
        status = summary_ingest._build_status_activity_fact(raw)  # type: ignore[attr-defined]
        project, _ = summary_ingest._build_status_snapshots(status)  # type: ignore[attr-defined]
        self.assertEqual(int(len(project.index)), 1)
        self.assertAlmostEqual(float(project.loc[0, "stringing_cumulative_progress"]), 14.006, places=3)

    def test_final_sag_preference_not_applied_without_policy(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 505",
                    "project_display": "TA 505",
                    "project_scope_key": "ta505",
                    "line_name": "",
                    "section_label": "Progress",
                    "source_file": "TA 505 - DPR - 2026-05-10.xlsx",
                    "source_sheet": "DPR-Summary",
                    "configured_sheet": "DPR-Summary",
                    "template_sheet": "TA 505 Status",
                    "stringing_resolution_policy": "",
                    "report_date": "2026-05-10",
                    "activity_raw": "Stringing (Rough Sag)",
                    "activity_norm": "stringing",
                    "cumulative_progress": 14.973,
                },
                {
                    "project_code": "TA 505",
                    "project_display": "TA 505",
                    "project_scope_key": "ta505",
                    "line_name": "",
                    "section_label": "Progress",
                    "source_file": "TA 505 - DPR - 2026-05-10.xlsx",
                    "source_sheet": "DPR-Summary",
                    "configured_sheet": "DPR-Summary",
                    "template_sheet": "TA 505 Status",
                    "stringing_resolution_policy": "",
                    "report_date": "2026-05-10",
                    "activity_raw": "Stringing (Final Sag)",
                    "activity_norm": "final_sag",
                    "cumulative_progress": 14.006,
                },
            ]
        )
        status = summary_ingest._build_status_activity_fact(raw)  # type: ignore[attr-defined]
        project, _ = summary_ingest._build_status_snapshots(status)  # type: ignore[attr-defined]
        self.assertEqual(int(len(project.index)), 1)
        self.assertAlmostEqual(float(project.loc[0, "stringing_cumulative_progress"]), 14.973, places=3)

    def test_status_snapshot_supports_multi_month_from_filename_dates(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 419",
                    "project_display": "TA 419",
                    "project_scope_key": "ta419",
                    "line_name": "",
                    "source_file": "TA 419 - DPR - 2026-03-29.xlsx",
                    "activity_raw": "Foundation",
                    "activity_norm": "foundation",
                    "quantity_primary": 100,
                    "cumulative_progress": 50,
                    "plan_for_month": 10,
                    "progress_for_month": 5,
                },
                {
                    "project_code": "TA 419",
                    "project_display": "TA 419",
                    "project_scope_key": "ta419",
                    "line_name": "",
                    "source_file": "TA 419 - DPR - 2026-04-29.xlsx",
                    "activity_raw": "Foundation",
                    "activity_norm": "foundation",
                    "quantity_primary": 100,
                    "cumulative_progress": 60,
                    "plan_for_month": 12,
                    "progress_for_month": 8,
                },
            ]
        )
        status = summary_ingest._build_status_activity_fact(raw)  # type: ignore[attr-defined]
        status_project, status_overall = summary_ingest._build_status_snapshots(status)  # type: ignore[attr-defined]
        self.assertEqual(pd.to_datetime(status["month"], errors="coerce").dt.to_period("M").nunique(), 2)
        self.assertEqual(pd.to_datetime(status_project["month"], errors="coerce").dt.to_period("M").nunique(), 2)
        self.assertEqual(pd.to_datetime(status_overall["month"], errors="coerce").dt.to_period("M").nunique(), 2)

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

    def test_compile_excludes_completed_projects(self) -> None:
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
                    {"project": "TA 419", "date": "2026-04-25", "gang_name": "Alpha", "from_ap": "1/0", "to_ap": "2/0", "method": "TSE", "section_readiness": "Ready", "daily_km": 0.8, "po_km": 0.8},
                    {"project": "TA 505", "date": "2026-04-25", "gang_name": "Beta", "from_ap": "3/0", "to_ap": "4/0", "method": "TSE", "section_readiness": "Ready", "daily_km": 1.0, "po_km": 1.0},
                ]
            ).to_parquet(stringing_root / "StringingDaily.parquet", index=False)

            pd.DataFrame(
                [
                    {"project_code": "TA 419", "project_display": "TA 419", "project_scope_key": "ta419", "line_name": "", "source_file": "TA 419 - DPR - 2026-04-25.xlsx", "activity_raw": "Tower Erection (Nos)", "activity_norm": "tower_erection", "quantity_primary": 100, "cumulative_progress": 60, "balance_progress": 40},
                    {"project_code": "TA 505", "project_display": "TA 505", "project_scope_key": "ta505", "line_name": "", "source_file": "TA 505 - DPR - 2026-04-25.xlsx", "activity_raw": "Tower Erection (Nos)", "activity_norm": "tower_erection", "quantity_primary": 120, "cumulative_progress": 70, "balance_progress": 50},
                ]
            ).to_parquet(status_root / "RawData.parquet", index=False)

            pd.DataFrame(
                [
                    {"project_code": "TA 419", "project_display": "TA 419", "project_scope_key": "ta419", "line_name": "", "report_date": "2026-04-25", "stretch_identifier": "AP-1 - AP-2", "from_ap": "1/0", "to_ap": "2/0", "length_km": 0.8, "readiness_state": "READY"},
                    {"project_code": "TA 505", "project_display": "TA 505", "project_scope_key": "ta505", "line_name": "", "report_date": "2026-04-25", "stretch_identifier": "AP-3 - AP-4", "from_ap": "3/0", "to_ap": "4/0", "length_km": 1.0, "readiness_state": "READY"},
                ]
            ).to_parquet(stretch_root / "RawData.parquet", index=False)

            out_path = root / "StringingSummary" / "StringingSummary_Output.xlsx"
            compiled = summary_ingest.compile_stringing_summary_to_workbook(
                erection_root,
                out_path,
                completed_project_keys={"ta419"},
            )
            status_df = pd.read_excel(compiled, sheet_name="StatusActivityFact")
            self.assertFalse(status_df["project_code"].astype(str).str.contains("TA 419", case=False, na=False).any())
            self.assertTrue(status_df["project_code"].astype(str).str.contains("TA 505", case=False, na=False).any())


if __name__ == "__main__":
    unittest.main()
