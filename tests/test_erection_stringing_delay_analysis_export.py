from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.erection_stringing_delay_analysis import (
    build_erection_stringing_delay_tables,
    write_erection_stringing_delay_workbook,
)


class ErectionStringingDelayAnalysisTests(unittest.TestCase):
    def test_series_and_client_type_splits_present_in_es_and_pofs(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 510",
                    "project_name": "TA 510 - L1",
                    "project_scope_key": "ta510::l1",
                    "from_ap": "1/0",
                    "to_ap": "1/1",
                    "po_start_date": "2026-06-10",
                    "po_completion_date": "2026-06-12",
                    "fs_starting_date": "2026-06-15",
                    "method": "TSE",
                    "gang_name": "G1",
                    "location nos": "1/0,1/1",
                },
                {
                    "project_code": "TB 620",
                    "project_name": "TB 620 - L2",
                    "project_scope_key": "tb620::l2",
                    "from_ap": "2/0",
                    "to_ap": "2/1",
                    "po_start_date": "2026-06-20",
                    "po_completion_date": "2026-06-18",
                    "fs_starting_date": "2026-06-22",
                    "method": "Manual",
                    "gang_name": "G2",
                    "location nos": "2/0,2/1",
                },
            ]
        )
        erection_daily = pd.DataFrame(
            [
                {"project_code": "TA 510", "Project Scope Key": "ta510::l1", "location_no": "1/0", "completion_date": "2026-06-04"},
                {"project_code": "TA 510", "Project Scope Key": "ta510::l1", "location_no": "1/1", "completion_date": "2026-06-05"},
                {"project_code": "TB 620", "Project Scope Key": "tb620::l2", "location_no": "2/0", "completion_date": "2026-06-14"},
                {"project_code": "TB 620", "Project Scope Key": "tb620::l2", "location_no": "2/1", "completion_date": "2026-06-15"},
            ]
        )

        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=erection_daily,
        )

        es_detail = tables["ES Delay Detail"]
        self.assertIn("Series", es_detail.columns)
        self.assertIn("Client Type", es_detail.columns)
        self.assertEqual(set(es_detail["Series"].astype(str)), {"5xx", "6xx"})
        self.assertEqual(set(es_detail["Client Type"].astype(str)), {"TA", "TB"})

        es_summary = tables["ES Delay Summary"]
        es_series = es_summary[
            (es_summary["Scope Type"].astype(str) == "SERIES")
            & (es_summary["Method Split"].astype(str) == "All")
        ]
        es_clients = es_summary[
            (es_summary["Scope Type"].astype(str) == "CLIENT_TYPE")
            & (es_summary["Method Split"].astype(str) == "All")
        ]
        self.assertEqual(set(es_series["Scope Name"].astype(str)), {"5xx", "6xx"})
        self.assertEqual(set(es_clients["Scope Name"].astype(str)), {"TA", "TB"})
        self.assertEqual(int(es_series["Spans Considered"].sum()), 2)
        self.assertEqual(int(es_clients["Spans Considered"].sum()), 2)

        pofs_summary = tables["POFS Delay Summary"]
        pofs_series = pofs_summary[
            (pofs_summary["Scope Type"].astype(str) == "SERIES")
            & (pofs_summary["Method Split"].astype(str) == "All")
        ]
        pofs_clients = pofs_summary[
            (pofs_summary["Scope Type"].astype(str) == "CLIENT_TYPE")
            & (pofs_summary["Method Split"].astype(str) == "All")
        ]
        self.assertEqual(set(pofs_series["Scope Name"].astype(str)), {"5xx", "6xx"})
        self.assertEqual(set(pofs_clients["Scope Name"].astype(str)), {"TA", "TB"})
        self.assertEqual(int(pofs_series["Spans Considered"].sum()), 2)
        self.assertEqual(int(pofs_clients["Spans Considered"].sum()), 2)

    def test_scope_key_harmonization_recovers_cross_line_lag(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_name": "TB 501 - 220kV",
                    "project_scope_key": "tb501::220kv",
                    "from_ap": "220/1",
                    "to_ap": "220/2",
                    "po_start_date": "2026-06-10",
                    "method": "TSE",
                    "gang_name": "G1",
                    "location nos": "220/1,220/2",
                },
                {
                    "project_name": "TB 501 - 132kV",
                    "project_scope_key": "tb501::132kv",
                    "from_ap": "132/1",
                    "to_ap": "132/2",
                    "po_start_date": "2026-06-15",
                    "method": "Manual",
                    "gang_name": "G2",
                    "location nos": "132/1,132/2",
                },
            ]
        )
        erection_daily = pd.DataFrame(
            [
                {"project_code": "TB 501", "Project Scope Key": "tb501::220kv", "location_no": "220/1", "completion_date": "2026-06-05"},
                {"project_code": "TB 501", "Project Scope Key": "tb501::220kv", "location_no": "220/2", "completion_date": "2026-06-06"},
                {"project_code": "TB 501", "Project Scope Key": "tb501::132kv", "location_no": "132/1", "completion_date": "2026-06-11"},
                {"project_code": "TB 501", "Project Scope Key": "tb501::132kv", "location_no": "132/2", "completion_date": "2026-06-12"},
            ]
        )

        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=erection_daily,
        )

        detail = tables["ES Delay Detail"]
        self.assertEqual(int(len(detail.index)), 2)
        self.assertTrue(detail["Gap Days"].notna().all())
        self.assertEqual(set(detail["Join Project Key"].astype(str)), {"tb501220kv", "tb501132kv"})
        self.assertTrue(detail["Project Key In Erection Map"].fillna(False).astype(bool).all())

        summary = tables["ES Delay Summary"]
        overall_all = summary[
            (summary["Scope Type"].astype(str) == "OVERALL")
            & (summary["Scope Name"].astype(str) == "All Projects")
            & (summary["Method Split"].astype(str) == "All")
        ].iloc[0]
        self.assertEqual(int(overall_all["Spans Considered"]), 2)
        self.assertEqual(int(overall_all["Spans With Computable Lag"]), 2)
        self.assertEqual(float(overall_all["Lag Coverage %"]), 100.0)

        overall_tse = summary[
            (summary["Scope Type"].astype(str) == "OVERALL")
            & (summary["Scope Name"].astype(str) == "All Projects")
            & (summary["Method Split"].astype(str) == "TSE")
        ].iloc[0]
        overall_others = summary[
            (summary["Scope Type"].astype(str) == "OVERALL")
            & (summary["Scope Name"].astype(str) == "All Projects")
            & (summary["Method Split"].astype(str) == "Others")
        ].iloc[0]
        self.assertEqual(int(overall_tse["Spans Considered"]), 1)
        self.assertEqual(int(overall_others["Spans Considered"]), 1)

    def test_kpi_excludes_negative_lag_and_adds_negative_anomaly(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_name": "TA 510",
                    "project_scope_key": "ta510",
                    "from_ap": "10/1",
                    "to_ap": "10/2",
                    "po_start_date": "2026-06-10",
                    "method": "TSE",
                    "location nos": "10/1,10/2",
                },
                {
                    "project_name": "TA 510",
                    "project_scope_key": "ta510",
                    "from_ap": "10/3",
                    "to_ap": "10/4",
                    "po_start_date": "2026-06-01",
                    "method": "TSE",
                    "location nos": "10/3,10/4",
                },
            ]
        )
        erection_daily = pd.DataFrame(
            [
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "10/1", "completion_date": "2026-06-04"},
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "10/2", "completion_date": "2026-06-05"},
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "10/3", "completion_date": "2026-06-05"},
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "10/4", "completion_date": "2026-06-06"},
            ]
        )

        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=erection_daily,
        )

        summary = tables["ES Delay Summary"]
        overall_all = summary[
            (summary["Scope Type"].astype(str) == "OVERALL")
            & (summary["Scope Name"].astype(str) == "All Projects")
            & (summary["Method Split"].astype(str) == "All")
        ].iloc[0]
        self.assertEqual(float(overall_all["Average Lag Days"]), 5.0)
        self.assertEqual(float(overall_all["Median Lag Days"]), 5.0)
        self.assertEqual(int(overall_all["Negative Lag Excluded"]), 1)

        detail = tables["ES Delay Detail"]
        negative_detail = detail[pd.to_numeric(detail["Gap Days"], errors="coerce") < 0]
        self.assertEqual(int(len(negative_detail.index)), 1)
        self.assertTrue(pd.isna(negative_detail.iloc[0]["Gap Days Non-Negative"]))

        anomalies = tables["ES Delay Anomalies"]
        negative_rows = anomalies[anomalies["Issue"].astype(str) == "NEGATIVE_LAG_EXCLUDED"]
        self.assertEqual(int(len(negative_rows.index)), 1)

    def test_missing_project_match_creates_anomaly_and_coverage_drop(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_name": "TA 999",
                    "project_scope_key": "ta999",
                    "from_ap": "1/0",
                    "to_ap": "1/1",
                    "po_start_date": "2026-06-10",
                    "method": "TSE",
                    "location nos": "1/0,1/1",
                }
            ]
        )
        erection_daily = pd.DataFrame(
            [
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "1/0", "completion_date": "2026-06-05"},
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "1/1", "completion_date": "2026-06-06"},
            ]
        )

        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=erection_daily,
        )

        summary = tables["ES Delay Summary"]
        overall_all = summary[
            (summary["Scope Type"].astype(str) == "OVERALL")
            & (summary["Scope Name"].astype(str) == "All Projects")
            & (summary["Method Split"].astype(str) == "All")
        ].iloc[0]
        self.assertEqual(int(overall_all["Spans Considered"]), 1)
        self.assertEqual(int(overall_all["Spans With Computable Lag"]), 0)
        self.assertEqual(float(overall_all["Lag Coverage %"]), 0.0)

        anomalies = tables["ES Delay Anomalies"]
        self.assertEqual(int((anomalies["Issue"].astype(str) == "PROJECT_SCOPE_KEY_NO_ERECTION_MATCH").sum()), 1)

    def test_stringing_monitoring_tables_project_code_blanking_and_audit_tags(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_code": "TA 413",
                    "project_name": "TA 413 - L1",
                    "project_scope_key": "ta413::l1",
                    "gang_name": "G1",
                    "from_ap": "1/0",
                    "to_ap": "1/1",
                    "location nos": "1/0,1/1",
                    "method": "TSE",
                    "po_start_date": "2026-05-01",
                },
                {
                    "project_code": "TA 414",
                    "project_name": "TA 414 - L1",
                    "project_scope_key": "ta414::l1",
                    "gang_name": "G2",
                    "from_ap": "2/0",
                    "to_ap": "2/1",
                    "location nos": "",
                    "method": "TSE",
                    "po_start_date": "2026-05-01",
                },
            ]
        )
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

        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=pd.DataFrame(),
            stringing_status_activity_fact=status_activity,
            stringing_manpower_fact=manpower_fact,
            stretch_readiness_summary=stretch_summary,
            stretch_readiness_manpower_audit=stretch_manpower_audit,
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

        self.assertTrue((numeric["project_code"].astype(str).str.strip() == "TA 777").any())

        ta414_audit = audit[audit["project_code"].astype(str).str.strip() == "TA 414"]
        self.assertFalse(ta414_audit.empty)
        self.assertEqual(str(ta414_audit.iloc[0]["location_nos_available"]).strip(), "No")
        tags = str(ta414_audit.iloc[0]["missing_data_tags"])
        self.assertIn("MISSING_STRETCH_READINESS", tags)
        self.assertIn("MISSING_MANPOWER", tags)
        self.assertIn("MISSING_FITTER", tags)
        self.assertIn("MISSING_LOCATION_NOS", tags)

    def test_workbook_writer_persists_expected_sheets_and_columns(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_name": "TA 510",
                    "project_scope_key": "ta510",
                    "from_ap": "1/0",
                    "to_ap": "1/1",
                    "po_start_date": "2026-06-10",
                    "method": "TSE",
                    "location nos": "",
                }
            ]
        )
        erection_daily = pd.DataFrame(
            [
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "1/0", "completion_date": "2026-06-05"},
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "1/1", "completion_date": "2026-06-06"},
            ]
        )
        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=erection_daily,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "Erection_Stringing_Delay_Analysis.xlsx"
            write_erection_stringing_delay_workbook(out, tables)
            self.assertTrue(out.exists())
            with pd.ExcelFile(out) as xl:
                self.assertEqual(
                    xl.sheet_names,
                    [
                        "ES Delay Summary",
                        "ES Delay Coverage",
                        "ES Delay Anomalies",
                        "ES Delay Detail",
                        "POFS Delay Summary",
                        "POFS Delay Coverage",
                        "POFS Delay Anomalies",
                        "POFS Delay Detail",
                        "Stringing Monitoring Numeric",
                        "Stringing Monitoring Audit",
                    ],
                )
            for sheet_name, frame in tables.items():
                loaded = pd.read_excel(out, sheet_name=sheet_name, header=1)
                self.assertEqual(list(loaded.columns), list(frame.columns))

    def test_pofs_tables_kpi_and_anomaly_rules(self) -> None:
        stringing_compiled_raw = pd.DataFrame(
            [
                {
                    "project_name": "TA 510",
                    "project_scope_key": "ta510",
                    "from_ap": "1/0",
                    "to_ap": "1/1",
                    "po_start_date": "2026-06-01",
                    "po_completion_date": "2026-06-10",
                    "fs_starting_date": "2026-06-15",
                    "method": "TSE",
                },
                {
                    "project_name": "TA 510",
                    "project_scope_key": "ta510",
                    "from_ap": "1/2",
                    "to_ap": "1/3",
                    "po_start_date": "2026-06-02",
                    "po_completion_date": "2026-06-12",
                    "fs_starting_date": "2026-06-10",
                    "method": "TSE",
                },
                {
                    "project_name": "TA 510",
                    "project_scope_key": "ta510",
                    "from_ap": "1/4",
                    "to_ap": "1/5",
                    "po_start_date": "2026-06-03",
                    "po_completion_date": "2026-06-13",
                    "fs_starting_date": pd.NA,
                    "method": "Manual",
                },
            ]
        )
        erection_daily = pd.DataFrame(
            [
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "1/0", "completion_date": "2026-05-28"},
                {"project_code": "TA 510", "Project Scope Key": "ta510", "location_no": "1/1", "completion_date": "2026-05-29"},
            ]
        )

        tables = build_erection_stringing_delay_tables(
            stringing_compiled_raw=stringing_compiled_raw,
            erection_daily=erection_daily,
        )

        summary = tables["POFS Delay Summary"]
        overall_all = summary[
            (summary["Scope Type"].astype(str) == "OVERALL")
            & (summary["Scope Name"].astype(str) == "All Projects")
            & (summary["Method Split"].astype(str) == "All")
        ].iloc[0]
        self.assertEqual(int(overall_all["Spans Considered"]), 3)
        self.assertEqual(int(overall_all["Spans With Computable Lag"]), 2)
        self.assertEqual(float(overall_all["Lag Coverage %"]), 66.7)
        self.assertEqual(float(overall_all["Average Lag Days"]), 5.0)
        self.assertEqual(float(overall_all["Median Lag Days"]), 5.0)
        self.assertEqual(int(overall_all["Negative Lag Excluded"]), 1)

        coverage = tables["POFS Delay Coverage"]
        cov_all = coverage[
            (coverage["Scope Type"].astype(str) == "OVERALL")
            & (coverage["Scope Name"].astype(str) == "All Projects")
            & (coverage["Method Split"].astype(str) == "All")
        ].iloc[0]
        self.assertEqual(int(cov_all["Missing PO Completion Rows"]), 0)
        self.assertEqual(int(cov_all["Missing FS Start Rows"]), 1)
        self.assertEqual(int(cov_all["Negative Gap Flag Rows"]), 1)

        anomalies = tables["POFS Delay Anomalies"]
        self.assertEqual(int((anomalies["Issue"].astype(str) == "NEGATIVE_LAG_EXCLUDED").sum()), 1)
        self.assertEqual(int((anomalies["Issue"].astype(str) == "FS_STARTING_DATE_MISSING").sum()), 1)


if __name__ == "__main__":
    unittest.main()
