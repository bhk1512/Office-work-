from __future__ import annotations

import unittest

import pipeline_runner


class PipelineRunnerScopeTests(unittest.TestCase):
    def test_scope_parser_accepts_combinations_and_aliases(self) -> None:
        self.assertEqual(pipeline_runner._normalize_scope("erection,stringing"), ("erection", "stringing"))
        self.assertEqual(pipeline_runner._normalize_scope("both"), ("erection", "stringing"))
        self.assertEqual(pipeline_runner._normalize_scope("e,s"), ("erection", "stringing"))
        self.assertEqual(pipeline_runner._normalize_scope("all"), ("erection", "stringing", "foundation"))
        self.assertEqual(pipeline_runner._normalize_scope("foundation"), ("foundation",))

    def test_scope_parser_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            pipeline_runner._normalize_scope("painting")

    def test_selected_stages_for_erection_scope_include_status_summary_dependencies(self) -> None:
        stages = pipeline_runner._selected_pipeline_stages(("erection",))
        self.assertTrue(stages["erection"])
        self.assertTrue(stages["progress_status"])
        self.assertTrue(stages["stringing_summary"])
        self.assertFalse(stages["stringing"])
        self.assertFalse(stages["foundation"])
        self.assertFalse(stages["stretch_readiness"])

    def test_selected_stages_for_stringing_scope_include_status_stretch_summary_dependencies(self) -> None:
        stages = pipeline_runner._selected_pipeline_stages(("stringing",))
        self.assertTrue(stages["stringing"])
        self.assertTrue(stages["progress_status"])
        self.assertTrue(stages["stretch_readiness"])
        self.assertTrue(stages["stringing_summary"])
        self.assertFalse(stages["erection"])
        self.assertFalse(stages["foundation"])

    def test_selected_stages_for_foundation_scope_are_foundation_only(self) -> None:
        stages = pipeline_runner._selected_pipeline_stages(("foundation",))
        self.assertTrue(stages["foundation"])
        self.assertFalse(stages["erection"])
        self.assertFalse(stages["stringing"])
        self.assertFalse(stages["progress_status"])
        self.assertFalse(stages["stretch_readiness"])
        self.assertFalse(stages["stringing_summary"])


if __name__ == "__main__":
    unittest.main()
