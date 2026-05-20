from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard import foundation_ingest
import pipeline_runner


def _write_config(
    path: Path,
    rows: list[dict[str, object]],
    *,
    template_sheets: dict[str, dict[int, str]] | None = None,
) -> None:
    columns = [
        "Project Code",
        "Erection Sheet Names",
        "Stringing Sheet Names",
        "Erection Template Check",
        "Stringing Template Check",
        "Erection Line Names",
        "Stringing Line Names",
        "Status Sheet Names",
        "Status Template Check",
        "Status Line Names",
        "Stretch Readiness Sheet Names",
        "Stretch Daily Stringing Sheet Names",
        "Stretch Template Check",
        "Stretch Line Names",
        "Stretch Manpower Expected",
        "Foundation Sheet Names",
        "Foundation Template Check",
        "Foundation Line Names",
        "Consolidation Rule",
        "Status File Identifier",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.reindex(columns=columns)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Sheet Names Check", index=False)
        for sheet_name, col_map in (template_sheets or {}).items():
            if not col_map:
                continue
            width = max(col_map.keys()) + 1
            rows_buf = [[""] * width for _ in range(2)]
            rows_buf[0][0] = "To Map"
            for idx, label in sorted(col_map.items()):
                if idx < width:
                    rows_buf[1][idx] = label
            pd.DataFrame(rows_buf).to_excel(
                writer,
                sheet_name=sheet_name[:31],
                header=False,
                index=False,
            )


def _write_dpr(path: Path, sheet_name: str, rows: list[list[object]]) -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)


class FoundationIngestTests(unittest.TestCase):
    def test_compile_parses_detail_dates_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TA 510",
                        "Foundation Sheet Names": "FDN",
                        "Foundation Line Names": "Main",
                    }
                ],
            )

            _write_dpr(
                dpr_root / "TA 510 - DPR - 2026-05-10.xlsx",
                "FDN",
                [
                    ["Location No", "Foundation Completion Date", "Status"],
                    ["1/0", "2026-05-09", "Completed"],
                    ["1/1", "2026-05-10", "Completed"],
                ],
            )
            _write_dpr(
                dpr_root / "TA 510 - DPR - 2026-05-11.xlsx",
                "FDN",
                [
                    ["Location No", "Foundation Completion Date", "Status"],
                    ["1/0", "2026-05-09", "Completed"],
                ],
            )

            out_path = root / "FoundationCompiled_Output.xlsx"
            compiled = foundation_ingest.compile_foundation_to_workbook(dpr_root, None, out_path)
            self.assertIsNotNone(compiled)
            self.assertTrue(out_path.exists())

            completion_df = pd.read_excel(out_path, sheet_name="FoundationCompletions")
            detail_df = completion_df[completion_df["source_type"].astype(str).str.lower() == "detail"]
            self.assertEqual(len(detail_df.index), 2)
            self.assertEqual(set(detail_df["location_no"].astype(str)), {"1/0", "1/1"})

    def test_compile_uses_template_mapping_for_wide_sheet_date_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TB 501",
                        "Foundation Sheet Names": "Progress-220kV",
                        "Foundation Template Check": "Yes",
                        "Foundation Line Names": "220kV",
                    }
                ],
                template_sheets={
                    "TB 501 Foundation": {
                        1: "Location No",
                        38: "Foundation Status",
                        44: "Foundation Completion Date",
                    }
                },
            )

            width = 50
            header = [""] * width
            header[1] = "LOC NO"
            header[38] = "Fnd Status"
            header[44] = "DOC"
            row_done = [""] * width
            row_done[0] = "1"
            row_done[1] = "1/0"
            row_done[38] = "C"
            row_done[44] = "2026-05-10"
            row_wip = [""] * width
            row_wip[0] = "2"
            row_wip[1] = "1/1"
            row_wip[38] = "WIP"

            _write_dpr(
                dpr_root / "TB 501 - DPR - 2026-05-10.xlsx",
                "Progress-220kV",
                [header, row_done, row_wip],
            )

            out_path = root / "FoundationCompiled_Output.xlsx"
            foundation_ingest.compile_foundation_to_workbook(
                dpr_root,
                None,
                out_path,
                status_raw=pd.DataFrame(),
            )

            completion_df = pd.read_excel(out_path, sheet_name="FoundationCompletions")
            coverage_df = pd.read_excel(out_path, sheet_name="Coverage")
            detail_df = completion_df[completion_df["source_type"].astype(str).str.lower() == "detail"]
            self.assertEqual(len(detail_df.index), 1)
            self.assertEqual(str(detail_df.iloc[0]["location_no"]), "1/0")
            tb501 = coverage_df[coverage_df["project_code"].astype(str).str.upper().eq("TB 501")]
            self.assertFalse(tb501.empty)
            self.assertEqual(str(tb501.iloc[0]["status"]).upper(), "OK_DETAIL")

    def test_compile_blank_foundation_config_explicitly_skips_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TA 504",
                        "Foundation Sheet Names": "",
                    }
                ],
            )
            _write_dpr(
                dpr_root / "TA 504 - DPR - 2026-05-18.xlsx",
                "DPR_TA-504",
                [
                    ["Random", "Sheet"],
                    ["No", "Foundation Detail"],
                ],
            )

            status_raw = pd.DataFrame(
                [
                    {
                        "project_code": "TA 504",
                        "activity_norm": "foundation",
                        "report_date": "2026-05-18",
                        "cumulative_progress": 141,
                        "source_sheet": "DPR_TA-504",
                        "configured_sheet": "DPR_TA-504",
                        "source_file": "TA 504 - DPR - 2026-05-18.xlsx",
                    }
                ]
            )

            out_path = root / "FoundationCompiled_Output.xlsx"
            foundation_ingest.compile_foundation_to_workbook(
                dpr_root,
                None,
                out_path,
                status_raw=status_raw,
            )
            completion_df = pd.read_excel(out_path, sheet_name="FoundationCompletions")
            coverage_df = pd.read_excel(out_path, sheet_name="Coverage")
            self.assertFalse(
                completion_df["source_type"].astype(str).str.lower().eq("snapshot_fallback").any()
            )
            skipped = coverage_df[
                coverage_df["project_code"].astype(str).str.upper().eq("TA 504")
                & coverage_df["status"].astype(str).str.upper().eq("SKIPPED_BLANK_CONFIG")
            ]
            self.assertFalse(skipped.empty)

    def test_compile_supports_duplicate_sheet_entries_with_file_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TB 507",
                        "Foundation Sheet Names": "Foundation; Foundation",
                        "Foundation Line Names": "MAIN; 765kV",
                        "Status File Identifier": "[MAIN] - DPR -; TB 507 - DPR -",
                    }
                ],
            )
            _write_dpr(
                dpr_root / "TB 507 [MAIN] - DPR - 2026-05-17.xlsx",
                "Foundation",
                [
                    ["Location Number", "Starting date", "Completion Date"],
                    ["83/1", "2025-01-08", "2025-01-17"],
                ],
            )
            _write_dpr(
                dpr_root / "TB 507 - DPR - 2026-05-17.xlsx",
                "Foundation",
                [
                    ["Location Number", "Starting date", "Completion Date"],
                    ["7/0", "2026-02-06", "2026-02-11"],
                ],
            )

            out_path = root / "FoundationCompiled_Output.xlsx"
            foundation_ingest.compile_foundation_to_workbook(
                dpr_root,
                None,
                out_path,
                status_raw=pd.DataFrame(),
            )
            completion_df = pd.read_excel(out_path, sheet_name="FoundationCompletions")
            detail = completion_df[completion_df["source_type"].astype(str).str.lower().eq("detail")].copy()
            self.assertEqual(set(detail["line_name"].astype(str).str.strip()), {"MAIN", "765kV"})
            self.assertEqual(set(detail["location_no"].astype(str).str.strip()), {"83/1", "7/0"})

    def test_compile_template_mapping_for_target_projects_avoids_rowwise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TA 505",
                        "Foundation Sheet Names": "Foundation",
                        "Foundation Template Check": "Yes",
                    },
                    {
                        "Project Code": "TA 512",
                        "Foundation Sheet Names": "Foundation",
                        "Foundation Template Check": "Yes",
                    },
                    {
                        "Project Code": "TA 602",
                        "Foundation Sheet Names": "FDN",
                        "Foundation Template Check": "Yes",
                    },
                    {
                        "Project Code": "TA 608",
                        "Foundation Sheet Names": "WIP-FDN",
                        "Foundation Template Check": "Yes",
                    },
                ],
                template_sheets={
                    "TA 505 Foundation": {2: "Location No", 21: "Foundation Start Date", 22: "Foundation Completion Date"},
                    "TA 512 Foundation": {3: "Loc. No.", 6: "Start", 7: "Completed"},
                    "TA 602 Foundation": {2: "Loc. No.", 6: "Start", 7: "Complete"},
                    "TA 608 Foundation": {1: "LOC. NO.", 6: "Start Date", 7: "Completion Date"},
                },
            )
            _write_dpr(
                dpr_root / "TA 505 - DPR - 2026-05-12.xlsx",
                "Foundation",
                [
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    [""] * 23,
                    ["", "", "Loc No.", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Cast Start", "Cast Complete"],
                    ["1", "row", "101/2", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Complete", "2024-09-11", "2024-09-26", "2024-10-02", "2024-10-06"],
                ],
            )
            _write_dpr(
                dpr_root / "TA 512 - DPR - 2026-05-18.xlsx",
                "Foundation",
                [
                    ["Sr. No.", "Month", "Month Sr. No.", "Loc. No.", "Type of Tower", "Type of Classification", "Start", "Completed"],
                    [1, "Feb'25", 1, "8/1", "DA-1.5", "Sandy", "2025-02-12", "2025-02-23"],
                ],
            )
            _write_dpr(
                dpr_root / "TA 602 - DPR - 2026-05-18.xlsx",
                "FDN",
                [
                    ["CUMM. SR.NO", "MONTH SR.NO.", "Loc. No.", "Type of Tower", "Type of Classification", "", "Start", "Complete"],
                    [1, 1, "125/2", "A+3", "DFR", "", "2025-09-02", "2025-09-15"],
                ],
            )
            _write_dpr(
                dpr_root / "TA 608 - DPR - 2026-04-05.xlsm",
                "WIP-FDN",
                [
                    ["S.NO.", "LOC. NO.", "Tower Type", "Classification", "GANG NAME", "Man power", "Start Date", "Completion Date"],
                    [1, "69/2", "A(00-02)+06", "SANDY PS", "Prabir Bala", 13, "2025-10-06", "2025-10-18"],
                ],
            )

            out_path = root / "FoundationCompiled_Output.xlsx"
            foundation_ingest.compile_foundation_to_workbook(
                dpr_root,
                None,
                out_path,
                status_raw=pd.DataFrame(),
            )
            completion_df = pd.read_excel(out_path, sheet_name="FoundationCompletions")
            diagnostics_df = pd.read_excel(out_path, sheet_name="Diagnostics")
            coverage_df = pd.read_excel(out_path, sheet_name="Coverage")
            target = diagnostics_df[
                diagnostics_df["Project"].astype(str).str.upper().isin({"TA 505", "TA 512", "TA 602", "TA 608"})
            ]
            self.assertFalse(target.empty)
            self.assertFalse(target["ParserMode"].astype(str).str.lower().eq("rowwise").any())
            for project in ("TA 505", "TA 512", "TA 602", "TA 608"):
                cov = coverage_df[coverage_df["project_code"].astype(str).str.upper().eq(project)]
                self.assertFalse(cov.empty)
                self.assertEqual(str(cov.iloc[0]["status"]).upper(), "OK_DETAIL")
            ta505 = completion_df[
                completion_df["project_code"].astype(str).str.upper().eq("TA 505")
                & completion_df["location_no"].astype(str).str.strip().eq("101/2")
            ]
            self.assertFalse(ta505.empty)
            self.assertEqual(
                pd.to_datetime(ta505.iloc[0]["event_date"]).strftime("%Y-%m-%d"),
                "2024-10-06",
            )

    def test_compile_marks_tb408_as_blocked_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TB 408",
                        "Foundation Sheet Names": "Foundation",
                    }
                ],
            )
            _write_dpr(
                dpr_root / "TB 408 - DPR - 2026-05-11.xlsx",
                "Erection Compiled",
                [
                    ["Location", "Complete Date"],
                    ["1/0", "2026-05-10"],
                ],
            )

            out_path = root / "FoundationCompiled_Output.xlsx"
            foundation_ingest.compile_foundation_to_workbook(
                dpr_root,
                None,
                out_path,
                status_raw=pd.DataFrame(),
            )
            coverage_df = pd.read_excel(out_path, sheet_name="Coverage")
            blocked = coverage_df[
                coverage_df["project_code"].astype(str).str.upper().eq("TB 408")
                & coverage_df["status"].astype(str).str.upper().eq("BLOCKED_NO_SOURCE")
            ]
            self.assertFalse(blocked.empty)

    def test_compile_captures_xml_scrub_fallback_note(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TA 513",
                        "Foundation Sheet Names": "FDN S-P",
                        "Foundation Line Names": "S-P",
                    }
                ],
            )
            file_path = dpr_root / "TA 513 - DPR - 2026-05-17.xlsx"
            _write_dpr(
                file_path,
                "FDN S-P",
                [
                    ["Location No", "Foundation Completion Date", "Status"],
                    ["2/0", "2026-05-16", "Completed"],
                ],
            )
            fake_df = pd.DataFrame(
                [
                    ["Location No", "Foundation Completion Date", "Status"],
                    ["2/0", "2026-05-16", "Completed"],
                ]
            )
            with patch.object(
                foundation_ingest,
                "load_sheet_with_csv_fallback",
                return_value=(fake_df, "FDN S-P", "XML scrub fallback used for sheet 'FDN S-P'"),
            ):
                out_path = root / "FoundationCompiled_Output.xlsx"
                foundation_ingest.compile_foundation_to_workbook(
                    dpr_root,
                    None,
                    out_path,
                    status_raw=pd.DataFrame(),
                )
            diagnostics = pd.read_excel(out_path, sheet_name="Diagnostics")
            self.assertTrue(
                diagnostics["FallbackNote"].astype(str).str.contains("XML scrub fallback used", case=False).any()
            )

    def test_pipeline_wrapper_and_parquet_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dpr_root = root / "DPRs"
            dpr_root.mkdir(parents=True, exist_ok=True)
            _write_config(
                root / "DPR_Config.xlsx",
                [
                    {
                        "Project Code": "TA 510",
                        "Foundation Sheet Names": "FDN",
                        "Foundation Line Names": "Main",
                    }
                ],
            )
            _write_dpr(
                dpr_root / "TA 510 - DPR - 2026-05-10.xlsx",
                "FDN",
                [
                    ["Location No", "Foundation Completion Date", "Status"],
                    ["1/0", "2026-05-09", "Completed"],
                ],
            )

            out_path = root / "Parquets" / "Foundation" / "FoundationCompiled_Output.xlsx"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            compiled = pipeline_runner.compile_foundation_to_workbook(
                dpr_root,
                None,
                out_path,
                completed_project_keys=None,
            )
            self.assertIsNotNone(compiled)
            self.assertTrue(out_path.exists())
            exported_dir = pipeline_runner.export_workbook_to_parquet(
                out_path,
                sheets=("FoundationRaw", "FoundationCompletions", "Coverage", "Diagnostics", "Issues"),
            )
            self.assertTrue((exported_dir / "FoundationRaw.parquet").exists())
            self.assertTrue((exported_dir / "FoundationCompletions.parquet").exists())


if __name__ == "__main__":
    unittest.main()
