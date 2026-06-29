from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import prepare_daily_dpr_mail as mail


class PrepareDailyDprMailTests(unittest.TestCase):
    def test_scope_parser_accepts_combinations_and_aliases(self) -> None:
        self.assertEqual(mail._normalize_scope("erection,stringing"), ("erection", "stringing"))
        self.assertEqual(mail._normalize_scope("both"), ("erection", "stringing"))
        self.assertEqual(mail._normalize_scope("e,s"), ("erection", "stringing"))
        self.assertEqual(mail._normalize_scope("all"), ("erection", "stringing", "foundation"))
        self.assertEqual(mail._normalize_scope("stringing,foundation"), ("stringing", "foundation"))

    def test_scope_parser_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            mail._normalize_scope("painting")

    def test_cli_rejects_foundation_only_mail_scope(self) -> None:
        with self.assertRaises(SystemExit):
            mail._parse_args(["--scope", "foundation"])

    def test_refresh_outputs_forwards_scope_and_compile_only_to_pipeline(self) -> None:
        with patch.object(mail, "_run_step") as run_step:
            mail.refresh_outputs(skip_outlook_pull=True, scope=("erection",))

        run_step.assert_called_once()
        command = run_step.call_args.args[0]
        self.assertIn("--compile-only", command)
        self.assertIn("--scope", command)
        self.assertEqual(command[command.index("--scope") + 1], "erection")

    def test_prepare_mail_erection_scope_skips_stringing_section_and_builder(self) -> None:
        erection = pd.DataFrame(
            [
                {
                    "PCH": "PCH",
                    "Project": "TA 414",
                    "Plan (Nos.)": 2,
                    "Actual Towers (Nos.)": 1,
                    "Total MT": 10.0,
                    "Avg Tower Wt (MT)": 10.0,
                    "Productivity": 1.0,
                }
            ]
        )

        with TemporaryDirectory() as tmp:
            args = type(
                "Args",
                (),
                {
                    "scope": ("erection",),
                    "skip_refresh": True,
                    "skip_outlook_pull": False,
                    "month": "2026-06",
                    "as_of_date": "2026-06-09",
                    "output_html": Path(tmp) / "mail.html",
                    "no_draft": True,
                    "to": "",
                    "cc": "",
                },
            )()
            with (
                patch.object(mail, "_load_pch_mapping", return_value=pd.DataFrame(columns=["project_key", "PCH", "Project"])),
                patch.object(mail, "_build_erection_table", return_value=erection) as build_erection,
                patch.object(mail, "_build_stringing_table") as build_stringing,
            ):
                artifacts = mail.prepare_mail(args)

            build_erection.assert_called_once()
            build_stringing.assert_not_called()
            html = artifacts.html_path.read_text(encoding="utf-8")
            self.assertIn("Erection Productivity", html)
            self.assertNotIn("Stringing Productivity", html)
            self.assertIn("Daily DPR Erection Productivity Summary", artifacts.subject)

    def test_prepare_mail_stringing_scope_skips_erection_section_and_builder(self) -> None:
        stringing = pd.DataFrame(
            [
                {
                    "PCH": "PCH",
                    "Project": "TA 414",
                    "Plan (KM)": 2.0,
                    "Actual Achieved (KM)": 1.0,
                    "Productivity": 30.0,
                    "Scope (KM)": 10.0,
                    "Stringing Completed (KM)": 5.0,
                    "Stretch Ready (KM)": 3.0,
                }
            ]
        )

        with TemporaryDirectory() as tmp:
            args = type(
                "Args",
                (),
                {
                    "scope": ("stringing",),
                    "skip_refresh": True,
                    "skip_outlook_pull": False,
                    "month": "2026-06",
                    "as_of_date": "2026-06-09",
                    "output_html": Path(tmp) / "mail.html",
                    "no_draft": True,
                    "to": "",
                    "cc": "",
                },
            )()
            with (
                patch.object(mail, "_load_pch_mapping", return_value=pd.DataFrame(columns=["project_key", "PCH", "Project"])),
                patch.object(mail, "_build_erection_table") as build_erection,
                patch.object(mail, "_build_stringing_table", return_value=stringing) as build_stringing,
            ):
                artifacts = mail.prepare_mail(args)

            build_erection.assert_not_called()
            build_stringing.assert_called_once()
            html = artifacts.html_path.read_text(encoding="utf-8")
            self.assertNotIn("Erection Productivity", html)
            self.assertIn("Stringing Productivity", html)
            self.assertIn("Daily DPR Stringing Productivity Summary", artifacts.subject)

    def test_activity_status_carries_prior_positive_plan_when_latest_plan_is_zero(self) -> None:
        status = pd.DataFrame(
            [
                {
                    "project_code": "TB 605",
                    "month": pd.Timestamp("2026-06-01"),
                    "report_date": pd.Timestamp("2026-06-07"),
                    "activity_group": "Tower Erection",
                    "core_activity": True,
                    "plan_for_month": 80,
                    "progress_for_month": 9,
                    "quantity_primary": 523,
                    "cumulative_progress": 212,
                },
                {
                    "project_code": "TB 605",
                    "month": pd.Timestamp("2026-06-01"),
                    "report_date": pd.Timestamp("2026-06-09"),
                    "activity_group": "Tower Erection",
                    "core_activity": True,
                    "plan_for_month": 0,
                    "progress_for_month": 11,
                    "quantity_primary": 523,
                    "cumulative_progress": 214,
                },
            ]
        )

        with patch.object(mail, "_read_parquet", return_value=status):
            result = mail._activity_status(
                "Tower Erection",
                pd.Timestamp("2026-06-01"),
                pd.Timestamp("2026-06-30"),
                pd.Timestamp("2026-06-09"),
            )

        row = result.iloc[0]
        self.assertEqual(row["Plan"], 80)
        self.assertEqual(row["Actual"], 11)

    def test_erection_actual_ignores_rows_without_location_number(self) -> None:
        status = pd.DataFrame(
            [
                {
                    "project_code": "TB 608",
                    "month": pd.Timestamp("2026-06-01"),
                    "report_date": pd.Timestamp("2026-06-07"),
                    "activity_group": "Tower Erection",
                    "core_activity": True,
                    "plan_for_month": 20,
                    "progress_for_month": 1,
                    "quantity_primary": 318,
                    "cumulative_progress": 32,
                }
            ]
        )
        erection_raw = pd.DataFrame(
            [
                {
                    "Project Code": "TB 608",
                    "Complete Date": pd.Timestamp("2026-06-02"),
                    "Location No.": "103/0",
                    "Tower Weight": 27.485,
                },
                {
                    "Project Code": "TB 608",
                    "Complete Date": pd.Timestamp("2026-06-07"),
                    "Location No.": pd.NA,
                    "Tower Weight": 25.498,
                },
                {
                    "Project Code": "TB 608",
                    "Complete Date": pd.Timestamp("2026-06-07"),
                    "Location No.": "nan",
                    "Tower Weight": 25.498,
                },
            ]
        )
        mapping = pd.DataFrame(
            [{"project_key": "TB608", "PCH": "PCH", "Project": "TB 608"}]
        )

        def fake_read(path):
            path_text = str(path)
            if path_text.endswith("StringingSummary\\StatusActivityFact.parquet"):
                return status
            if path_text.endswith("Erection\\RawData.parquet"):
                return erection_raw
            return pd.DataFrame()

        with patch.object(mail, "_read_parquet", side_effect=fake_read):
            result = mail._build_erection_table(
                pd.Timestamp("2026-06-01"),
                pd.Timestamp("2026-06-30"),
                pd.Timestamp("2026-06-09"),
                mapping,
            )

        row = result.iloc[0]
        self.assertEqual(row["Plan (Nos.)"], 20)
        self.assertEqual(row["Actual Towers (Nos.)"], 1)
        self.assertAlmostEqual(row["Total MT"], 27.48)

    def test_erection_total_mt_uses_undated_completed_rows_when_status_has_line_actual(self) -> None:
        status = pd.DataFrame(
            [
                {
                    "project_code": "TB 501",
                    "line_name": "132kV",
                    "month": pd.Timestamp("2026-06-01"),
                    "report_date": pd.Timestamp("2026-06-15"),
                    "activity_group": "Tower Erection",
                    "core_activity": True,
                    "plan_for_month": 5,
                    "progress_for_month": 1,
                    "quantity_primary": 39,
                    "cumulative_progress": 32,
                },
                {
                    "project_code": "TB 501",
                    "line_name": "220kV",
                    "month": pd.Timestamp("2026-06-01"),
                    "report_date": pd.Timestamp("2026-06-15"),
                    "activity_group": "Tower Erection",
                    "core_activity": True,
                    "plan_for_month": 0,
                    "progress_for_month": 0,
                    "quantity_primary": 199,
                    "cumulative_progress": 199,
                },
            ]
        )
        erection_raw = pd.DataFrame(
            [
                {
                    "Project Code": "TB 501",
                    "Line Name": "132kV",
                    "Complete Date": pd.NaT,
                    "Location No.": "2/1",
                    "Tower Weight": 11.815,
                    "Status": "C",
                },
                {
                    "Project Code": "TB 501",
                    "Line Name": "220kV",
                    "Complete Date": pd.NaT,
                    "Location No.": "35/0",
                    "Tower Weight": 18.459,
                    "Status": "C",
                },
            ]
        )
        mapping = pd.DataFrame(
            [{"project_key": "TB501", "PCH": "PCH", "Project": "TB 501"}]
        )

        def fake_read(path):
            path_text = str(path)
            if path_text.endswith("StringingSummary\\StatusActivityFact.parquet"):
                return status
            if path_text.endswith("Erection\\RawData.parquet"):
                return erection_raw
            return pd.DataFrame()

        with patch.object(mail, "_read_parquet", side_effect=fake_read):
            result = mail._build_erection_table(
                pd.Timestamp("2026-06-01"),
                pd.Timestamp("2026-06-30"),
                pd.Timestamp("2026-06-17"),
                mapping,
            )

        row = result.iloc[0]
        self.assertEqual(row["Actual Towers (Nos.)"], 1)
        self.assertAlmostEqual(row["Total MT"], 11.82)
        self.assertAlmostEqual(row["Avg Tower Wt (MT)"], 11.82)


if __name__ == "__main__":
    unittest.main()
