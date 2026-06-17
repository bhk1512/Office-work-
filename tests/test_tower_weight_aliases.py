from __future__ import annotations

import pandas as pd

import erection_compiled_to_daily_new as erection
from dashboard import data_loader
from dashboard import workbook


def test_erection_header_detection_accepts_total_tower_weight() -> None:
    raw = pd.DataFrame(
        [
            ["Sl. No.", "LOC NO", "Tower Type", "Start Date", "Completion Date", "Gang Name", "Total Tower Weight"],
            [1, "1/0", "DD+0", "2026-06-01", "2026-06-02", "Gang 1", 9.261],
        ]
    )

    header_row, columns = erection.find_header_row(raw)

    assert header_row == 0
    assert columns is not None
    assert columns[6] == "tower weight"


def test_workbook_dpr_standardization_accepts_total_tower_weight() -> None:
    raw = pd.DataFrame(
        [
            ["Sl. No.", "Location No.", "Tower Type", "Starting Date", "Total Tower Weight", "Completion Date", "Gang Name", "Manpower"],
            [1, "1/0", "DD+0", "2026-06-01", 9.261, "2026-06-02", "Gang 1", 10],
        ]
    )

    header_row = workbook._find_dpr_header_row(raw)
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = raw.iloc[header_row].tolist()
    standardized = workbook._standardize_dpr_columns(data)

    assert header_row == 0
    assert "tower_weight" in standardized.columns
    assert float(standardized.iloc[0]["tower_weight"]) == 9.261


def test_daily_loader_reads_total_tower_weight_from_daily_sheet() -> None:
    source = pd.DataFrame(
        [
            {
                "Work Date": "2026-06-02",
                "Productivity": 4.6305,
                "Project Name": "TB 501 - 220kV",
                "Gang name": "Gang 1",
                "Total Tower Weight": 9.261,
            }
        ]
    )

    loaded = data_loader.load_daily_from_proddailyexpanded(source)

    assert "tower_weight" in loaded.columns
    assert float(loaded.iloc[0]["tower_weight"]) == 9.261


def test_rawdata_loader_carries_total_tower_weight_into_expanded_rows() -> None:
    source = pd.DataFrame(
        [
            {
                "Start Date": "2026-06-01",
                "Complete Date": "2026-06-02",
                "Productivity": 4.6305,
                "Project Name": "TB 501 - 220kV",
                "Gang name": "Gang 1",
                "Total Tower Weight": 9.261,
            }
        ]
    )

    loaded = data_loader.load_daily_from_rawdata(source)

    assert len(loaded.index) == 2
    assert "tower_weight" in loaded.columns
    assert set(loaded["tower_weight"].astype(float)) == {9.261}
