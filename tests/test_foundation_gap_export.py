from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.config import AppConfig
from dashboard.workbook import export_erection_productivity_summary


class _DummyStore:
    def __init__(self, config: AppConfig) -> None:
        self._config = config


class FoundationGapExportTests(unittest.TestCase):
    def test_export_writes_foundation_gap_sheets_and_gap_math(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parquets_root = root / "Parquets"
            erection_root = parquets_root / "Erection"
            foundation_root = parquets_root / "Foundation"
            erection_root.mkdir(parents=True, exist_ok=True)
            foundation_root.mkdir(parents=True, exist_ok=True)

            foundation_completions = pd.DataFrame(
                [
                    {
                        "project_code": "TA 510",
                        "project_display": "TA 510",
                        "project_scope_key": "ta510::l1",
                        "line_name": "L1",
                        "line_name_source": "config",
                        "report_date": "2026-04-30",
                        "event_date": "2026-04-01",
                        "source_file": "TA 510 - DPR - 2026-04-30.xlsx",
                        "source_sheet": "FDN",
                        "configured_sheet": "FDN",
                        "source_type": "detail",
                        "quality_flag": "detail_date",
                        "location_no": "1/0",
                        "event_value": 1,
                        "cumulative_foundation": pd.NA,
                    },
                    {
                        "project_code": "TA 510",
                        "project_display": "TA 510",
                        "project_scope_key": "ta510::l2",
                        "line_name": "L2",
                        "line_name_source": "config",
                        "report_date": "2026-04-30",
                        "event_date": "2026-04-08",
                        "source_file": "TA 510 - DPR - 2026-04-30.xlsx",
                        "source_sheet": "FDN",
                        "configured_sheet": "FDN",
                        "source_type": "detail",
                        "quality_flag": "detail_date",
                        "location_no": "2/0",
                        "event_value": 1,
                        "cumulative_foundation": pd.NA,
                    },
                    {
                        "project_code": "TA 510",
                        "project_display": "TA 510",
                        "project_scope_key": "ta510::l2",
                        "line_name": "L2",
                        "line_name_source": "config",
                        "report_date": "2026-05-01",
                        "event_date": "2026-04-08",
                        "source_file": "TA 510 - DPR - 2026-05-01.xlsx",
                        "source_sheet": "FDN",
                        "configured_sheet": "FDN",
                        "source_type": "detail",
                        "quality_flag": "detail_date",
                        "location_no": "2/0",
                        "event_value": 1,
                        "cumulative_foundation": pd.NA,
                    },
                ]
            )
            foundation_coverage = pd.DataFrame(
                [
                    {
                        "project_code": "TA 510",
                        "project_display": "TA 510",
                        "status": "OK_DETAIL",
                        "reason_code": "OK_DETAIL",
                        "reason": "Detail rows parsed",
                        "source_used": "detail",
                        "snapshot_limited": "No",
                        "detail_rows": 2,
                        "detail_completions": 2,
                        "snapshot_rows": 0,
                    },
                    {
                        "project_code": "TB 408",
                        "project_display": "TB 408",
                        "status": "BLOCKED_NO_SOURCE",
                        "reason_code": "BLOCKED_NO_SOURCE",
                        "reason": "No usable foundation/status source",
                        "source_used": "missing",
                        "snapshot_limited": "No",
                        "detail_rows": 0,
                        "detail_completions": 0,
                        "snapshot_rows": 0,
                    },
                ]
            )
            foundation_diagnostics = pd.DataFrame(
                [
                    {
                        "Workbook": "TA 510 - DPR - 2026-04-30.xlsx",
                        "Project": "TA 510",
                        "Sheet": "FDN",
                        "ConfiguredSheet": "FDN",
                        "LineName": "L1",
                        "LineNameSource": "config",
                        "Rows": 10,
                        "DetailRows": 2,
                        "Completions": 2,
                        "ParserMode": "template",
                        "FallbackNote": "",
                        "Status": "OK_TEMPLATE",
                        "Reason": "",
                        "TemplateSheet": "TA 510 Foundation",
                        "TemplateApplied": True,
                        "TemplateChanges": "",
                    }
                ]
            )
            with pd.ExcelWriter(foundation_root / "FoundationCompiled_Output.xlsx", engine="openpyxl") as writer:
                foundation_completions.to_excel(writer, sheet_name="FoundationCompletions", index=False)
                foundation_coverage.to_excel(writer, sheet_name="Coverage", index=False)
                foundation_diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)

            daily_df = pd.DataFrame(
                [
                    {
                        "date": "2026-04-02",
                        "completion_date": "2026-04-02",
                        "start_date": "2026-04-01",
                        "location_no": "1/0",
                        "tower_type": "DA",
                        "tower_weight": 40.0,
                        "daily_prod_mt": 4.0,
                        "gang_name": "Alpha",
                        "status": "Done",
                        "project_name": "TA 510 Line 1",
                        "project_code": "TA 510",
                        "line_name": "L1",
                        "project_display": "TA 510 - L1",
                    },
                    {
                        "date": "2026-04-20",
                        "completion_date": "2026-04-20",
                        "start_date": "2026-04-19",
                        "location_no": "2/0",
                        "tower_type": "DB",
                        "tower_weight": 42.0,
                        "daily_prod_mt": 4.2,
                        "gang_name": "Beta",
                        "status": "Done",
                        "project_name": "TA 510 Line 2",
                        "project_code": "TA 510",
                        "line_name": "L2",
                        "project_display": "TA 510 - L2",
                    },
                    {
                        "date": "2026-05-10",
                        "completion_date": "2026-05-10",
                        "start_date": "2026-05-09",
                        "location_no": "3/0",
                        "tower_type": "DC",
                        "tower_weight": 44.0,
                        "daily_prod_mt": 4.4,
                        "gang_name": "Gamma",
                        "status": "Done",
                        "project_name": "TA 510 Line 2",
                        "project_code": "TA 510",
                        "line_name": "L2",
                        "project_display": "TA 510 - L2",
                    },
                ]
            )
            project_info = pd.DataFrame(
                [
                    {"project_code": "TA 510", "project_name": "TA 510", "pch": "PCH 1", "region": "North"},
                ]
            )

            output_path = root / "summary.xlsx"
            store = _DummyStore(AppConfig(data_path=erection_root))
            export_erection_productivity_summary(
                output_path=output_path,
                data_store=store,
                daily_df=daily_df,
                project_info=project_info,
                as_of_date="2026-05-31",
            )

            with pd.ExcelFile(output_path) as xl:
                self.assertIn("Erection Summary Weekly", xl.sheet_names)
                self.assertIn("Erection Summary Monthly", xl.sheet_names)
                self.assertNotIn("Foundation Gap Monthly", xl.sheet_names)
                self.assertNotIn("Foundation Gap Weekly", xl.sheet_names)
                self.assertNotIn("Foundation Gap Coverage", xl.sheet_names)
                self.assertNotIn("Foundation Delay Phases", xl.sheet_names)
                self.assertNotIn("Foundation Delay Monthly", xl.sheet_names)
                self.assertNotIn("Foundation Delay Coverage", xl.sheet_names)

            weekly_summary = pd.read_excel(output_path, sheet_name="Erection Summary Weekly", header=None)
            monthly_summary = pd.read_excel(output_path, sheet_name="Erection Summary Monthly", header=None)
            self.assertGreater(len(weekly_summary.index), 2)
            self.assertGreater(len(monthly_summary.index), 2)


if __name__ == "__main__":
    unittest.main()
