"""Data loading utilities for the productivity dashboard."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import re

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


def _pick_tol(df: pd.DataFrame, opts):
    m = {str(c).strip().lower(): c for c in df.columns}
    for o in opts:
        key = o.strip().lower()
        if key in m: return m[key]
    for k, c in m.items():
        if any(o.lower() in k for o in opts):
            return c
    raise KeyError(f"Column not found among {opts}: have {list(df.columns)}")



def load_project_details(path: Path, sheet: str = "ProjectDetails") -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names:
            return pd.DataFrame()
        df = pd.read_excel(xl, sheet_name=sheet)

        col_code   = _pick_tol(df, ["project_code"])
        col_name   = _pick_tol(df, ["project_name"])
        col_client = _pick_tol(df, ["client_name"])
        col_noa    = _pick_tol(df, ["noa_start"])
        col_loa    = _pick_tol(df, ["loa_end"])
        col_pe     = _pick_tol(df, ["planning_eng"])
        col_pch    = _pick_tol(df, ["pch"])
        col_rm     = _pick_tol(df, ["regional_mgr"])
        col_pm     = _pick_tol(df, ["project_mgr"])
        col_si     = _pick_tol(df, ["section_inch"])
        col_sup    = _pick_tol(df, ["supervisor"])

        out = pd.DataFrame({
            "project_code": df[col_code].astype(str).str.strip(),
            "project_name": df[col_name].astype(str).str.strip(),
            "client_name": df[col_client].astype(str).str.strip(),
            "noa_start":   pd.to_datetime(df[col_noa], errors="coerce"),
            "loa_end":     pd.to_datetime(df[col_loa], errors="coerce"),
            "planning_eng": df[col_pe].astype(str).str.strip(),
            "pch":          df[col_pch].astype(str).str.strip(),
            "regional_mgr": df[col_rm].astype(str).str.strip(),
            "project_mgr":  df[col_pm].astype(str).str.strip(),
            "section_inch": df[col_si].astype(str).str.strip(),
            "supervisor":   df[col_sup].astype(str).str.strip(),
        })
        out = out[(out["project_name"]!="nan") | (out["project_code"]!="nan")].copy()
        out["key_name"] = out["project_name"].str.lower().str.replace(r"\s+", " ", regex=True)
        if "Project Name" in df.columns:
            out["Project Name"] = df["Project Name"].astype(str).str.strip()
        return out
    except Exception:
        return pd.DataFrame()
