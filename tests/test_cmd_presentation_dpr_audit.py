import unittest

from export_cmd_presentation_dpr_audit import (
    classify_supply_field,
    classify_supply_item,
    classify_table_family,
    compare_values,
    normalize_project_code,
)


class CmdPresentationDprAuditTests(unittest.TestCase):
    def test_normalize_project_code_handles_deck_and_file_variants(self):
        cases = {
            "TA – 413": "TA 413",
            "TA-506": "TA 506",
            "TB 507 [MAIN] - DPR - 2026-05-21.xlsx": "TB 507",
            "400 KV D/C Twin Line Latehar- Patraru (A303/304)": "A-303/304",
            "A-722/723": "A-722/723",
            "A920": "A920",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_project_code(raw), expected)

    def test_classify_table_family_for_key_cmd_tables(self):
        self.assertEqual(
            classify_table_family(
                "project_status",
                "TL04 - Example",
                [["LOA DATE: 12/03/2024", "LOA Value in Cr.:- 632", "CONTRACTUAL COMPLETION: 11/12/25"]],
            ),
            "contract_header",
        )
        self.assertEqual(
            classify_table_family(
                "project_status",
                "TL04 - Example",
                [["Activity", "UoM", "As Per LOA (Qty)", "Completed"], ["Foundation", "No", "340", "304"]],
            ),
            "activity_status",
        )
        self.assertEqual(
            classify_table_family(
                "project_status",
                "TL04 - Example",
                [["No. of month from NOA", "", "UoM", "Qty"], ["", "Completion Plan", "Nos", "352"]],
            ),
            "monthly_plan",
        )

    def test_supply_item_and_field_classification(self):
        self.assertEqual(classify_supply_item("Stub Supply Actual"), "stub")
        self.assertEqual(classify_supply_item("160KN CLR"), "insulator_160")
        self.assertEqual(classify_supply_item("210KN CLR"), "insulator_210")
        self.assertEqual(classify_supply_item("HW Fittings"), "hardware")
        self.assertEqual(classify_supply_field("Actual Supplied"), "actual_supplied")
        self.assertEqual(classify_supply_field("Balance As on Date"), "balance")
        self.assertEqual(classify_supply_field("As per L2 (April'26)"), "l2_qty")

    def test_compare_values_handles_numeric_and_text_equivalence(self):
        self.assertEqual(compare_values("306", 306.0), "match")
        self.assertEqual(compare_values("306.00", "306"), "match")
        self.assertEqual(compare_values("Comp", "Completed"), "differs")
        self.assertEqual(compare_values("", "12"), "missing_in_presentation")
        self.assertEqual(compare_values("12", ""), "not_generated")


if __name__ == "__main__":
    unittest.main()
