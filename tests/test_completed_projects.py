from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.completed_projects import (
    is_completed_project,
    load_completed_project_keys,
    normalize_project_code_key,
)


class CompletedProjectsTests(unittest.TestCase):
    def test_normalize_equivalence(self) -> None:
        self.assertEqual(normalize_project_code_key("TA 419"), normalize_project_code_key("TA-419"))
        self.assertEqual(normalize_project_code_key("TA 419"), normalize_project_code_key("ta419"))

    def test_load_missing_file_fail_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            keys = load_completed_project_keys(root, root)
            self.assertEqual(keys, set())

    def test_load_reads_project_code_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            path = raw_root / "Completed Projects.xlsx"
            pd.DataFrame({"Project Code": ["TA 419", "", None, "TB-507"]}).to_excel(path, index=False)
            keys = load_completed_project_keys(raw_root, None)
            self.assertIn(normalize_project_code_key("TA 419"), keys)
            self.assertIn(normalize_project_code_key("TB 507"), keys)
            self.assertTrue(is_completed_project("TA-419", keys))
            self.assertFalse(is_completed_project("TA 505", keys))


if __name__ == "__main__":
    unittest.main()

