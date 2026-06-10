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


if __name__ == "__main__":
    unittest.main()
