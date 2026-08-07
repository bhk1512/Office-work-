import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import outlook_dpr_watcher as watcher


class _Attachment:
    def __init__(self, filename: str) -> None:
        self.FileName = filename
        self.DisplayName = filename

    def SaveAsFile(self, target: str) -> None:
        Path(target).write_bytes(b"test workbook")


class _Attachments:
    def __init__(self, *attachments: _Attachment) -> None:
        self._attachments = attachments
        self.Count = len(attachments)

    def Item(self, index: int) -> _Attachment:
        return self._attachments[index - 1]


class _Mail:
    Class = 43
    SenderEmailAddress = "neogin@kecrpg.com"
    Subject = "TB-608 || Daily Progress Report 2026-05-31"
    ReceivedTime = dt.datetime(2026, 5, 31, 10, 0)

    def __init__(self, *attachments: _Attachment) -> None:
        self.Attachments = _Attachments(*attachments)


class OutlookDprWatcherTests(unittest.TestCase):
    def test_parse_date_from_attachment_name_accepts_two_digit_year(self) -> None:
        self.assertEqual(
            watcher.parse_date_from_text("DPR - TA418 Dt-28-06-26 (PGCIL).xlsx"),
            dt.date(2026, 6, 28),
        )

    def test_parse_date_from_project_prefixed_attachment_with_underscore_suffix(self) -> None:
        self.assertEqual(
            watcher.parse_date_from_text("TA-602-29-07-2026_.xlsx"),
            dt.date(2026, 7, 29),
        )

    def test_configured_tb608_xlsm_attachment_is_saved(self) -> None:
        mail = _Mail(_Attachment("DPR.xlsm"))
        email_config = {
            "neogin@kecrpg.com": [
                {
                    "project_code": "TB 608",
                    "attachment_rules": [
                        {
                            "match_key": "dpr",
                            "match_tokens": ["dpr"],
                            "line_name": "",
                        }
                    ],
                    "config_row_index": 0,
                }
            ]
        }

        with TemporaryDirectory() as tmp:
            with (
                patch.object(watcher, "DOWNLOAD_DIR", tmp),
                patch.object(watcher, "EMAIL_CONFIG", email_config),
                patch.object(watcher, "COMPLETED_PROJECT_KEYS", frozenset()),
            ):
                saved = watcher.save_latest_for_mail(mail)

            expected = Path(tmp) / "TB 608 - DPR - 2026-05-31.xlsm"
            self.assertEqual(saved, [str(expected)])
            self.assertTrue(expected.exists())

    def test_single_project_report_like_attachment_falls_back_to_sender_project(self) -> None:
        mail = _Mail(_Attachment("TA-602-29-07-2026_.xlsx"))
        mail.SenderEmailAddress = "shuklar03@kecrpg.com"
        mail.Subject = "TA-602 Daily Progress Report"
        mail.ReceivedTime = dt.datetime(2026, 7, 30, 10, 0)
        email_config = {
            "shuklar03@kecrpg.com": [
                {
                    "project_code": "TA 602",
                    "attachment_rules": [
                        {
                            "match_key": "dprta",
                            "match_tokens": ["dpr", "ta"],
                            "line_name": "",
                        }
                    ],
                    "config_row_index": 27,
                }
            ]
        }

        with TemporaryDirectory() as tmp:
            with (
                patch.object(watcher, "DOWNLOAD_DIR", tmp),
                patch.object(watcher, "EMAIL_CONFIG", email_config),
                patch.object(watcher, "COMPLETED_PROJECT_KEYS", frozenset()),
            ):
                saved = watcher.save_latest_for_mail(mail)

            expected = Path(tmp) / "TA 602 - DPR - 2026-07-29.xlsx"
            self.assertEqual(saved, [str(expected)])
            self.assertTrue(expected.exists())

    def test_single_project_report_text_attachment_falls_back_to_sender_project(self) -> None:
        mail = _Mail(_Attachment("TA-602 Daily Progress Report 30.07.26.xlsx"))
        mail.SenderEmailAddress = "shuklar03@kecrpg.com"
        mail.Subject = "TA-602 Daily Progress Report"
        mail.ReceivedTime = dt.datetime(2026, 7, 30, 10, 0)
        email_config = {
            "shuklar03@kecrpg.com": [
                {
                    "project_code": "TA 602",
                    "attachment_rules": [
                        {
                            "match_key": "dprta",
                            "match_tokens": ["dpr", "ta"],
                            "line_name": "",
                        }
                    ],
                    "config_row_index": 27,
                }
            ]
        }

        with TemporaryDirectory() as tmp:
            with (
                patch.object(watcher, "DOWNLOAD_DIR", tmp),
                patch.object(watcher, "EMAIL_CONFIG", email_config),
                patch.object(watcher, "COMPLETED_PROJECT_KEYS", frozenset()),
            ):
                saved = watcher.save_latest_for_mail(mail)

            expected = Path(tmp) / "TA 602 - DPR - 2026-07-30.xlsx"
            self.assertEqual(saved, [str(expected)])
            self.assertTrue(expected.exists())

    def test_single_project_non_report_attachment_is_not_fallback_saved(self) -> None:
        mail = _Mail(_Attachment("TA-602 Locations.xlsx"))
        mail.SenderEmailAddress = "shuklar03@kecrpg.com"
        mail.Subject = "TA-602 Daily Progress Report"
        mail.ReceivedTime = dt.datetime(2026, 7, 30, 10, 0)
        email_config = {
            "shuklar03@kecrpg.com": [
                {
                    "project_code": "TA 602",
                    "attachment_rules": [
                        {
                            "match_key": "dprta",
                            "match_tokens": ["dpr", "ta"],
                            "line_name": "",
                        }
                    ],
                    "config_row_index": 27,
                }
            ]
        }

        with TemporaryDirectory() as tmp:
            with (
                patch.object(watcher, "DOWNLOAD_DIR", tmp),
                patch.object(watcher, "EMAIL_CONFIG", email_config),
                patch.object(watcher, "COMPLETED_PROJECT_KEYS", frozenset()),
            ):
                saved = watcher.save_latest_for_mail(mail)

            self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
