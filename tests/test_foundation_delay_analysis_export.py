from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.foundation_delay_analysis import (
    MechanismConfig,
    build_legacy_erection_source_from_raw,
    build_complete_foundation_analysis_tables,
    build_foundation_delay_analysis_tables,
    build_v2_erection_source_from_raw,
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
        tables = build_foundation_delay_analysis_tables(*inputs, mechanism_config=MechanismConfig(min_foundation_count=1))

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
            "Mechanism Summary - Project",
            "Mechanism Summary - Series",
            "Mechanism Summary - Ownership",
            "Mechanism Matrix - Project",
            "Mechanism Matrix - Overall",
            "Mechanism Evidence Audit",
            "Mechanism Config",
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

        mechanism_project = tables["Mechanism Summary - Project"]
        ta510_mech = mechanism_project[
            mechanism_project["Project"].astype(str).str.strip().eq("TA 510")
            & mechanism_project["Cohort Window"].astype(str).str.strip().eq("Monsoon")
            & mechanism_project["Start Window"].astype(str).str.strip().eq("Post-Monsoon")
        ]
        self.assertFalse(ta510_mech.empty)
        ta510_row = ta510_mech.iloc[0]
        self.assertEqual(int(ta510_row["Foundations Total"]), 3)
        self.assertEqual(int(ta510_row["Matched Non-Negative"]), 2)
        self.assertEqual(int(ta510_row["Starts In Window"]), 0)
        self.assertEqual(float(ta510_row["Match Coverage %"]), 66.67)
        self.assertEqual(float(ta510_row["% Within Matched"]), 0.0)

        mechanism_audit = tables["Mechanism Evidence Audit"]
        self.assertIn("Foundation Source File", mechanism_audit.columns)
        self.assertIn("Erection Source File", mechanism_audit.columns)

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
                        "Mechanism Summary - Project",
                        "Mechanism Summary - Series",
                        "Mechanism Summary - Ownership",
                        "Mechanism Matrix - Project",
                        "Mechanism Matrix - Overall",
                        "Mechanism Evidence Audit",
                        "Mechanism Config",
                    },
                )

    def test_complete_bundle_contains_legacy_and_v2_tabs(self) -> None:
        source_daily, foundation_completions, foundation_coverage, foundation_diagnostics, progress_status_raw = self._build_inputs()
        tables = build_complete_foundation_analysis_tables(
            raw_erection_source=source_daily,
            foundation_completions=foundation_completions,
            foundation_coverage=foundation_coverage,
            foundation_diagnostics=foundation_diagnostics,
            progress_status_raw=progress_status_raw,
            daily_reference=source_daily,
        )
        expected = {
            "Foundation Gap Monthly",
            "Foundation Gap Weekly",
            "Foundation Gap Coverage",
            "Foundation Delay Phases",
            "Foundation Delay Monthly",
            "Foundation Delay Coverage",
            "Foundation Delay Anomalies",
            "Delay Phase - Project",
            "Delay Phase - Series",
            "Delay Phase - Ownership",
            "Delay Buckets - Project",
            "Delay Buckets - Series",
            "Delay Buckets - Ownership",
            "Delay Coverage",
            "Delay Anomalies",
            "Scope Snapshot",
            "Mechanism Summary - Project",
            "Mechanism Summary - Series",
            "Mechanism Summary - Ownership",
            "Mechanism Matrix - Project",
            "Mechanism Matrix - Overall",
            "Mechanism Evidence Audit",
            "Mechanism Config",
        }
        self.assertTrue(expected.issubset(set(tables.keys())))
        self.assertFalse(tables["Foundation Gap Monthly"].empty)
        self.assertFalse(tables["Foundation Delay Phases"].empty)
        self.assertFalse(tables["Delay Phase - Project"].empty)

    def test_legacy_and_v2_raw_source_eligibility_filters(self) -> None:
        raw = pd.DataFrame(
            [
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "1/0", "Start Date": "2026-01-01", "Complete Date": "2026-01-10"},
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "2/0", "Start Date": "2026-01-05", "Complete Date": pd.NA},
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "3/0", "Start Date": pd.NA, "Complete Date": "2026-01-20"},
            ]
        )
        legacy = build_legacy_erection_source_from_raw(raw)
        v2 = build_v2_erection_source_from_raw(raw)
        self.assertEqual(len(legacy.index), 1)
        self.assertEqual(set(legacy["location_no"].astype(str).tolist()), {"1/0"})
        self.assertEqual(len(v2.index), 2)
        self.assertEqual(set(v2["location_no"].astype(str).tolist()), {"1/0", "2/0"})

    def test_legacy_and_v2_matching_coherent_when_same_eligibility(self) -> None:
        foundation_completions = pd.DataFrame(
            [
                {"project_code": "TA 510", "project_display": "TA 510", "project_scope_key": "ta510", "line_name": "L1", "source_type": "detail", "event_date": "2026-01-01", "location_no": "1/0"},
                {"project_code": "TA 510", "project_display": "TA 510", "project_scope_key": "ta510", "line_name": "L1", "source_type": "detail", "event_date": "2026-01-02", "location_no": "2/0"},
            ]
        )
        foundation_coverage = pd.DataFrame(
            [
                {"project_code": "TA 510", "project_display": "TA 510", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed"},
            ]
        )
        foundation_diagnostics = pd.DataFrame([{"Project": "TA 510", "ParserMode": "template"}])
        raw = pd.DataFrame(
            [
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "1/0", "Start Date": "2026-01-03", "Complete Date": "2026-01-04"},
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "2/0", "Start Date": "2026-01-05", "Complete Date": "2026-01-06"},
            ]
        )
        tables = build_complete_foundation_analysis_tables(
            raw_erection_source=raw,
            foundation_completions=foundation_completions,
            foundation_coverage=foundation_coverage,
            foundation_diagnostics=foundation_diagnostics,
            progress_status_raw=pd.DataFrame(),
            daily_reference=pd.DataFrame(),
        )
        legacy_cov = tables["Foundation Delay Coverage"]
        v2_cov = tables["Delay Coverage"]
        legacy_row = legacy_cov[legacy_cov["Project"].astype(str).str.strip().eq("TA 510")].iloc[0]
        v2_row = v2_cov[v2_cov["Project"].astype(str).str.strip().eq("TA 510")].iloc[0]
        self.assertEqual(int(legacy_row["Matched Locations"]), int(v2_row["Matched Locations"]))
        self.assertEqual(int(legacy_row["Unmatched Locations"]), int(v2_row["Unmatched Locations"]))

    def test_legacy_tabs_exclude_start_only_rows(self) -> None:
        foundation_completions = pd.DataFrame(
            [
                {"project_code": "TA 510", "project_display": "TA 510", "project_scope_key": "ta510", "line_name": "L1", "source_type": "detail", "event_date": "2026-01-01", "location_no": "1/0"},
                {"project_code": "TA 510", "project_display": "TA 510", "project_scope_key": "ta510", "line_name": "L1", "source_type": "detail", "event_date": "2026-01-02", "location_no": "2/0"},
            ]
        )
        foundation_coverage = pd.DataFrame(
            [
                {"project_code": "TA 510", "project_display": "TA 510", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed", "snapshot_limited": "No", "detail_rows": 2, "detail_completions": 2, "snapshot_rows": 0},
            ]
        )
        foundation_diagnostics = pd.DataFrame([{"Project": "TA 510", "ParserMode": "template"}])
        raw = pd.DataFrame(
            [
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "1/0", "Start Date": "2026-01-03", "Complete Date": "2026-01-04"},
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "2/0", "Start Date": "2026-01-05", "Complete Date": pd.NA},
            ]
        )
        tables = build_complete_foundation_analysis_tables(
            raw_erection_source=raw,
            foundation_completions=foundation_completions,
            foundation_coverage=foundation_coverage,
            foundation_diagnostics=foundation_diagnostics,
            progress_status_raw=pd.DataFrame(),
            daily_reference=pd.DataFrame(),
        )
        legacy_cov = tables["Foundation Delay Coverage"]
        v2_cov = tables["Delay Coverage"]
        legacy_row = legacy_cov[legacy_cov["Project"].astype(str).str.strip().eq("TA 510")].iloc[0]
        v2_row = v2_cov[v2_cov["Project"].astype(str).str.strip().eq("TA 510")].iloc[0]
        self.assertEqual(int(legacy_row["Matched Locations"]), 1)
        self.assertEqual(int(v2_row["Matched Locations"]), 2)

    def test_raw_start_recovery_and_alias_metrics(self) -> None:
        foundation_completions = pd.DataFrame(
            [
                {
                    "project_code": "TA 413",
                    "project_display": "TA 413",
                    "project_scope_key": "ta413",
                    "line_name": "",
                    "source_type": "detail",
                    "event_date": "2026-03-05",
                    "location_no": "N'37A/0",
                }
            ]
        )
        foundation_coverage = pd.DataFrame(
            [
                {"project_code": "TA 413", "project_display": "TA 413", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed"},
            ]
        )
        foundation_diagnostics = pd.DataFrame([{"Project": "TA 413", "ParserMode": "template"}])
        # Raw erection source has start date but normalized location text is plain "37A/0".
        source_raw = pd.DataFrame(
            [
                {"Project Code": "TA 413", "Project Display": "TA 413", "Line Name": "", "Location No.": "37A/0", "Start Date": "2026-03-20"},
            ]
        )
        # Daily reference intentionally does not contain this location to mark as recovered.
        source_daily = pd.DataFrame(
            [
                {"project_code": "TA 413", "project_display": "TA 413", "line_name": "", "location_no": "20/0", "start_date": "2025-01-01"},
            ]
        )
        progress_status_raw = pd.DataFrame()

        tables = build_foundation_delay_analysis_tables(
            source_raw,
            foundation_completions,
            foundation_coverage,
            foundation_diagnostics,
            progress_status_raw,
            daily_reference=source_daily,
        )
        coverage = tables["Delay Coverage"]
        row = coverage[coverage["Project"].astype(str).str.strip().eq("TA 413")].iloc[0]
        self.assertEqual(int(row["Matched Locations"]), 1)
        self.assertEqual(int(row["RawData Start-Date Matches"]), 1)
        self.assertEqual(int(row["Alias Matches"]), 1)
        self.assertEqual(int(row["Dropped-by-Daily Recovered"]), 1)
        anomalies = tables["Delay Anomalies"]
        self.assertTrue(anomalies.empty)

    def test_multi_line_strict_line_to_line_matching(self) -> None:
        foundation_completions = pd.DataFrame(
            [
                {"project_code": "TA 513", "project_display": "TA 513", "project_scope_key": "ta513", "line_name": "S-F", "source_type": "detail", "event_date": "2026-05-04", "location_no": "30/0"},
                {"project_code": "TA 513", "project_display": "TA 513", "project_scope_key": "ta513", "line_name": "S-P", "source_type": "detail", "event_date": "2025-09-10", "location_no": "30/0"},
            ]
        )
        foundation_coverage = pd.DataFrame(
            [
                {"project_code": "TA 513", "project_display": "TA 513", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed"},
            ]
        )
        foundation_diagnostics = pd.DataFrame([{"Project": "TA 513", "ParserMode": "template"}])
        source_raw = pd.DataFrame(
            [
                {"Project Code": "TA 513", "Project Display": "TA 513", "Line Name": "Ere S-P", "Location No.": "30/0", "Start Date": "2025-12-08"},
            ]
        )
        progress_status_raw = pd.DataFrame()

        tables = build_foundation_delay_analysis_tables(
            source_raw,
            foundation_completions,
            foundation_coverage,
            foundation_diagnostics,
            progress_status_raw,
        )
        coverage = tables["Delay Coverage"]
        row = coverage[coverage["Project"].astype(str).str.strip().eq("TA 513")].iloc[0]
        self.assertEqual(int(row["Matched Locations"]), 0)
        self.assertEqual(int(row["Unmatched Locations"]), 2)
        anomalies = tables["Delay Anomalies"]
        self.assertEqual(int((anomalies["Issue"] == "UNMATCHED_LOCATION").sum()), 2)

    def test_mechanism_summary_threshold_and_window_config(self) -> None:
        inputs = self._build_inputs()
        default_tables = build_foundation_delay_analysis_tables(*inputs)
        self.assertTrue(default_tables["Mechanism Summary - Project"].empty)

        tuned = MechanismConfig(
            min_foundation_count=1,
            post_monsoon=(7,),
            post_monsoon_wide=(7, 12),
        )
        tuned_tables = build_foundation_delay_analysis_tables(*inputs, mechanism_config=tuned)
        summary = tuned_tables["Mechanism Summary - Project"]
        row = summary[
            summary["Project"].astype(str).str.strip().eq("TA 510")
            & summary["Cohort Window"].astype(str).str.strip().eq("Monsoon")
            & summary["Start Window"].astype(str).str.strip().eq("Post-Monsoon")
        ]
        self.assertFalse(row.empty)
        record = row.iloc[0]
        self.assertEqual(int(record["Foundations Total"]), 3)
        self.assertEqual(int(record["Matched Non-Negative"]), 2)
        self.assertEqual(int(record["Starts In Window"]), 1)
        self.assertEqual(float(record["% of All Foundations"]), 33.33)
        self.assertEqual(float(record["% Within Matched"]), 50.0)

    def test_mechanism_audit_blank_safe_without_source_columns(self) -> None:
        foundation_completions = pd.DataFrame(
            [
                {
                    "project_code": "TA 510",
                    "project_display": "TA 510",
                    "project_scope_key": "ta510",
                    "line_name": "L1",
                    "source_type": "detail",
                    "event_date": "2026-01-01",
                    "location_no": "1/0",
                }
            ]
        )
        foundation_coverage = pd.DataFrame(
            [
                {"project_code": "TA 510", "project_display": "TA 510", "status": "OK_DETAIL", "source_used": "detail", "reason": "Detail rows parsed"},
            ]
        )
        foundation_diagnostics = pd.DataFrame([{"Project": "TA 510", "ParserMode": "template"}])
        source_raw = pd.DataFrame(
            [
                {"Project Code": "TA 510", "Project Display": "TA 510", "Line Name": "L1", "Location No.": "1/0", "Start Date": "2026-01-02"},
            ]
        )
        tables = build_foundation_delay_analysis_tables(
            source_raw,
            foundation_completions,
            foundation_coverage,
            foundation_diagnostics,
            pd.DataFrame(),
            mechanism_config=MechanismConfig(min_foundation_count=1),
        )
        audit = tables["Mechanism Evidence Audit"]
        self.assertFalse(audit.empty)
        self.assertIn("Erection Source File", audit.columns)
        self.assertTrue((audit["Erection Source File"].fillna("").astype(str).str.strip() == "").all())
        self.assertTrue((audit["Foundation Source File"].fillna("").astype(str).str.strip() == "").all())

    def test_complete_bundle_adds_stringing_monitoring_numeric_and_audit(self) -> None:
        source_daily, foundation_completions, foundation_coverage, foundation_diagnostics, progress_status_raw = self._build_inputs()
        status_activity = pd.DataFrame(
            [
                {
                    "project_code": "TA 413",
                    "month": "2026-05-01",
                    "report_date": "2026-05-20",
                    "activity_raw": "Final Sag",
                    "activity_norm": "final_sag",
                    "activity_group": "Stringing",
                    "quantity_primary": 118.0,
                    "plan_for_month": 18.0,
                    "progress_for_month": 6.0,
                    "cumulative_progress": 74.0,
                    "balance_progress": 44.0,
                },
                {
                    "project_code": "TA 413",
                    "month": "2026-05-01",
                    "report_date": "2026-05-20",
                    "activity_raw": "Paying Out",
                    "activity_norm": "paying_out",
                    "activity_group": "Other",
                    "cumulative_progress": 80.0,
                    "balance_progress": 38.0,
                },
                {
                    "project_code": "TA 414",
                    "month": "2026-05-01",
                    "report_date": "2026-05-21",
                    "activity_raw": "Stringing (Km)",
                    "activity_norm": "stringing",
                    "activity_group": "Stringing",
                    "quantity_primary": 127.8,
                    "plan_for_month": pd.NA,
                    "progress_for_month": pd.NA,
                    "cumulative_progress": 10.4,
                    "balance_progress": 117.4,
                },
                {
                    "project_code": "TA 777",
                    "month": "2026-05-01",
                    "report_date": "2026-05-20",
                    "activity_raw": "Final Sag",
                    "activity_norm": "final_sag",
                    "activity_group": "Stringing",
                    "quantity_primary": 10.0,
                    "plan_for_month": 2.0,
                    "progress_for_month": 1.0,
                    "cumulative_progress": 5.0,
                    "balance_progress": 5.0,
                },
            ]
        )
        manpower_fact = pd.DataFrame(
            [
                {
                    "project_code": "TA 413",
                    "date": "2026-05-20",
                    "month": "2026-05-01",
                    "gang_name": "G1",
                    "daily_km": 1.2,
                    "manpower_gang_strength": 50,
                    "manpower_fitters": 10,
                },
                {
                    "project_code": "TA 413",
                    "date": "2026-05-20",
                    "month": "2026-05-01",
                    "gang_name": "G1",
                    "daily_km": 0.8,
                    "manpower_gang_strength": pd.NA,
                    "manpower_fitters": pd.NA,
                },
                {
                    "project_code": "TA 414",
                    "date": "2026-05-21",
                    "month": "2026-05-01",
                    "gang_name": "G2",
                    "daily_km": 0.4,
                    "manpower_gang_strength": pd.NA,
                    "manpower_fitters": pd.NA,
                },
            ]
        )
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 413",
                    "gang_name": "G1",
                    "from_ap": "1/0",
                    "to_ap": "1/1",
                    "location nos": "1/0,1/1",
                },
                {
                    "project_code": "TA 414",
                    "gang_name": "G2",
                    "from_ap": "2/0",
                    "to_ap": "2/1",
                    "location nos": "",
                },
            ]
        )
        stretch_summary = pd.DataFrame(
            [
                {"project_code": "TA 413", "ready_km": 50.0, "total_km": 100.0, "readiness_pct": 50.0},
            ]
        )
        stretch_manpower_audit = pd.DataFrame(
            [
                {"project_code": "TA 414", "status": "ABSENT"},
            ]
        )

        tables = build_complete_foundation_analysis_tables(
            raw_erection_source=source_daily,
            foundation_completions=foundation_completions,
            foundation_coverage=foundation_coverage,
            foundation_diagnostics=foundation_diagnostics,
            progress_status_raw=progress_status_raw,
            stringing_status_activity_fact=status_activity,
            stringing_manpower_fact=manpower_fact,
            stringing_compiled_raw=stringing_compiled_raw,
            stretch_readiness_summary=stretch_summary,
            stretch_readiness_manpower_audit=stretch_manpower_audit,
            daily_reference=source_daily,
        )

        self.assertIn("Stringing Monitoring Numeric", tables)
        self.assertIn("Stringing Monitoring Audit", tables)

        numeric = tables["Stringing Monitoring Numeric"]
        audit = tables["Stringing Monitoring Audit"]

        ta413_project = numeric[
            (numeric["project_code"].astype(str).str.strip() == "TA 413")
            & (numeric["row_type"].astype(str).str.strip() == "project")
        ]
        self.assertFalse(ta413_project.empty)
        self.assertTrue(pd.notna(ta413_project.iloc[0]["monthly_plan_km"]))
        self.assertTrue(pd.notna(ta413_project.iloc[0]["fs_achieved_month_km"]))

        ta414_project = numeric[
            (numeric["project_code"].astype(str).str.strip() == "TA 414")
            & (numeric["row_type"].astype(str).str.strip() == "project")
        ]
        self.assertFalse(ta414_project.empty)
        self.assertTrue(pd.isna(ta414_project.iloc[0]["monthly_plan_km"]))
        self.assertTrue(pd.isna(ta414_project.iloc[0]["fs_achieved_month_km"]))

        ta413_gang = numeric[
            (numeric["project_code"].astype(str).str.strip() == "TA 413")
            & (numeric["row_type"].astype(str).str.strip() == "gang")
            & (numeric["gang_name"].astype(str).str.strip() == "G1")
        ]
        self.assertFalse(ta413_gang.empty)

        # No project filter: TA 777 appears from stringing status data only.
        self.assertTrue((numeric["project_code"].astype(str).str.strip() == "TA 777").any())

        ta414_audit = audit[audit["project_code"].astype(str).str.strip() == "TA 414"]
        self.assertFalse(ta414_audit.empty)
        self.assertEqual(str(ta414_audit.iloc[0]["location_nos_available"]).strip(), "No")
        tags = str(ta414_audit.iloc[0]["missing_data_tags"])
        self.assertIn("MISSING_LOCATION_NOS", tags)
        self.assertIn("MISSING_STRETCH_READINESS", tags)


if __name__ == "__main__":
    unittest.main()
