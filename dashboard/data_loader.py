"""Data loading utilities for the productivity dashboard."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


def _pick_column(df: pd.DataFrame, options: Iterable[str]) -> str:
    """Return the first matching column from *options*, raising if none are found."""

    mapping = {str(col).strip().lower(): col for col in df.columns}
    for option in options:
        key = option.strip().lower()
        if key in mapping:
            return mapping[key]
    for key, original in mapping.items():
        lowered = key.lower()
        if any(option.lower() in lowered for option in options):
            return original
    joined = ", ".join(options)
    raise KeyError(f"Column not found among {joined}")


def load_daily_from_dailyexpanded(xl: pd.ExcelFile, sheet: str = "DailyExpanded") -> pd.DataFrame:
    """Load daily productivity rows from the DailyExpanded sheet."""

    LOGGER.debug("Loading data from sheet '%s'", sheet)
    df = pd.read_excel(xl, sheet_name=sheet)
    col_date = _pick_column(df, ["Work Date", "date"])
    col_prod = _pick_column(df, ["Productivity", "daily_prod_mt", "avg_daily_prod_mt"])
    col_proj = _pick_column(df, ["Project Name", "project_name"])
    col_gang = _pick_column(df, ["Gang name", "gang_name"])
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(df[col_date], errors="coerce").dt.normalize(),
            "daily_prod_mt": pd.to_numeric(df[col_prod], errors="coerce"),
            "project_name": df[col_proj].astype(str).str.strip(),
            "gang_name": df[col_gang].astype(str).str.strip(),
        }
    ).dropna(subset=["date", "daily_prod_mt"])
    LOGGER.debug("Loaded %d daily rows from DailyExpanded", len(result))
    return result


def load_daily_from_rawdata(xl: pd.ExcelFile, sheet: str = "RawData") -> pd.DataFrame:
    """Load daily productivity rows from a RawData sheet by expanding date ranges."""

    LOGGER.debug("Loading data from sheet '%s'", sheet)
    df = pd.read_excel(xl, sheet_name=sheet)
    start_col = _pick_column(df, ["Start Date", "starting date"])
    end_col = _pick_column(df, ["Complete Date", "completion date"])
    prod_col = _pick_column(df, ["Productivity", "avg_daily_prod_mt", "daily_prod_mt"])
    project_col = _pick_column(df, ["Project Name", "project_name"])
    gang_col = _pick_column(df, ["Gang name", "gang_name"])

    base = pd.DataFrame(
        {
            "start": pd.to_datetime(df[start_col], errors="coerce"),
            "end": pd.to_datetime(df[end_col], errors="coerce"),
            "daily_prod_mt": pd.to_numeric(df[prod_col], errors="coerce"),
            "project_name": df[project_col].astype(str).str.strip(),
            "gang_name": df[gang_col].astype(str).str.strip(),
        }
    ).dropna(subset=["start", "end", "daily_prod_mt"])
    rows: list[dict[str, object]] = []
    for _, record in base.iterrows():
        for date in pd.date_range(record["start"], record["end"], freq="D"):
            rows.append(
                {
                    "date": date.normalize(),
                    "daily_prod_mt": record["daily_prod_mt"],
                    "project_name": record["project_name"],
                    "gang_name": record["gang_name"],
                }
            )
    LOGGER.debug("Expanded raw data into %d daily rows", len(rows))
    return pd.DataFrame(rows)


def load_daily(config_or_path: AppConfig | Path | str) -> pd.DataFrame:
    """Load daily productivity data from a config or explicit workbook path."""

    if isinstance(config_or_path, AppConfig):
        config = config_or_path
    else:
        workbook_path = Path(config_or_path)
        config = AppConfig(data_path=workbook_path)

    LOGGER.info("Loading workbook '%s'", config.data_path)
    workbook = pd.ExcelFile(config.data_path)
    if config.preferred_sheet in workbook.sheet_names:
        return load_daily_from_dailyexpanded(workbook, config.preferred_sheet)
    if "RawData" in workbook.sheet_names:
        return load_daily_from_rawdata(workbook, "RawData")
    raise FileNotFoundError("Neither 'DailyExpanded' nor 'RawData' found in workbook.")
