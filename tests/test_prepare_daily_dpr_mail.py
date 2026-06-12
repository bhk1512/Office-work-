from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import prepare_daily_dpr_mail as mail


class PrepareDailyDprMailTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
