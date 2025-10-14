"""Data loading utilities for the productivity dashboard."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import AppConfig

LOGGER = logging.getLogger(__name__)

PROJECT_BASELINES_SHEET = "ProjectBaselines"
PROJECT_BASELINES_MONTHLY_SHEET = "ProjectBaselinesMonthly"

_PROJECT_BASELINE_OVERALL: dict[str, float] = {}
_PROJECT_BASELINE_MONTHLY: dict[str, dict[pd.Timestamp, float]] = {}
_PROJECT_BASELINE_SOURCE: Path | None = None


def _pick_column(df: pd.DataFrame, options: Iterable[str]) -> str:
    """Return the first matching column from *options*, raising if none are found."""

    mapping = {str(col).strip().lower(): col for col in df.columns}
    for option in options:
        key = option.strip().lower()
        if key in mapping:
            return mapping[key]
    for key, original in mapping.items():
        if any(option.lower() in key for option in options):
            return original
    joined = ", ".join(options)
    raise KeyError(f"Column not found among {joined}")


def _set_project_baseline_cache(
    overall: dict[str, float],
    monthly: dict[str, dict[pd.Timestamp, float]],
    source: Path | None,
) -> None:
    """Store project baseline maps for reuse across the app."""

    global _PROJECT_BASELINE_OVERALL, _PROJECT_BASELINE_MONTHLY, _PROJECT_BASELINE_SOURCE
    _PROJECT_BASELINE_OVERALL = dict(overall)
    _PROJECT_BASELINE_MONTHLY = {project: dict(month_map) for project, month_map in monthly.items()}
    _PROJECT_BASELINE_SOURCE = Path(source) if source else None


def get_project_baseline_maps() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Return cached project baseline maps (overall and monthly)."""

    return (
        dict(_PROJECT_BASELINE_OVERALL),
        {project: dict(month_map) for project, month_map in _PROJECT_BASELINE_MONTHLY.items()},
    )


def _compute_project_baseline_maps(
    data: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Compute overall and monthly productivity baselines for each project."""

    if data.empty or "project_name" not in data or "daily_prod_mt" not in data:
        return {}, {}

    working = data.copy()
    working["project_name"] = working["project_name"].astype(str).str.strip()
    working["daily_prod_mt"] = pd.to_numeric(working["daily_prod_mt"], errors="coerce")
    working = working.dropna(subset=["project_name", "daily_prod_mt"])
    if working.empty:
        return {}, {}

    month_series = None
    if "month" in working.columns:
        month_series = pd.to_datetime(working["month"], errors="coerce")
        if month_series.notna().any():
            month_series = month_series.dt.to_period("M").dt.to_timestamp()
        else:
            month_series = None
    if month_series is None:
        if "date" in working.columns:
            month_series = pd.to_datetime(working["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        else:
            month_series = pd.Series(pd.NaT, index=working.index)
    working["__baseline_month"] = month_series

    overall_series = working.groupby("project_name")["daily_prod_mt"].mean().dropna()
    overall = {str(project): float(value) for project, value in overall_series.items() if not pd.isna(value)}

    monthly: dict[str, dict[pd.Timestamp, float]] = {}
    monthly_series = (
        working.dropna(subset=["__baseline_month"])
        .groupby(["project_name", "__baseline_month"])["daily_prod_mt"]
        .mean()
        .dropna()
    )
    for (project, month), value in monthly_series.items():
        month_ts = pd.to_datetime(month)
        if pd.isna(month_ts):
            continue
        monthly.setdefault(str(project), {})[pd.Timestamp(month_ts)] = float(value)

    return overall, monthly


def _persist_project_baselines(
    workbook_path: Path | None,
    overall: dict[str, float],
    monthly: dict[str, dict[pd.Timestamp, float]],
) -> None:
    """Persist baseline tables into the compiled workbook for fast reuse."""

    if workbook_path is None:
        return
    path = Path(workbook_path)
    if not path.exists():
        LOGGER.warning(
            "Cannot write project baselines because workbook '%s' is missing.",
            path,
        )
        return

    overall_rows = [
        {"project_name": project, "baseline_mt_per_day": float(value)}
        for project, value in sorted(overall.items())
    ]
    overall_df = (
        pd.DataFrame(overall_rows)
        if overall_rows
        else pd.DataFrame(columns=["project_name", "baseline_mt_per_day"])
    )

    monthly_rows: list[dict[str, Any]] = []
    for project, month_map in monthly.items():
        for month, value in month_map.items():
            monthly_rows.append(
                {
                    "project_name": project,
                    "month": pd.to_datetime(month),
                    "baseline_mt_per_day": float(value),
                }
            )
    monthly_df = (
        pd.DataFrame(monthly_rows)
        if monthly_rows
        else pd.DataFrame(columns=["project_name", "month", "baseline_mt_per_day"])
    )
    if not monthly_df.empty:
        monthly_df["month"] = pd.to_datetime(monthly_df["month"], errors="coerce")
        monthly_df = monthly_df.dropna(subset=["month"]).sort_values(["project_name", "month"])

    try:
        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            overall_df.to_excel(writer, PROJECT_BASELINES_SHEET, index=False)
            monthly_df.to_excel(writer, PROJECT_BASELINES_MONTHLY_SHEET, index=False)
    except FileNotFoundError:
        LOGGER.warning(
            "Workbook '%s' not found when attempting to persist project baselines.",
            path,
        )
    except PermissionError:
        LOGGER.warning(
            "Permission denied while writing project baselines to '%s'.",
            path,
        )
    except Exception as exc:
        LOGGER.warning(
            "Failed to write project baselines to '%s': %s",
            path,
            exc,
        )


def _refresh_project_baselines(workbook_path: Path, data: pd.DataFrame) -> None:
    """Ensure project baseline sheets and caches reflect the current daily data."""

    if data.empty:
        load_project_baselines(workbook_path)
        return

    overall_map, monthly_map = _compute_project_baseline_maps(data)
    _set_project_baseline_cache(overall_map, monthly_map, workbook_path)
    _persist_project_baselines(workbook_path, overall_map, monthly_map)



def load_project_baselines(
    workbook_path: Path | str,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Load precomputed project baselines from the workbook, updating the cache."""

    path = Path(workbook_path)
    try:
        with pd.ExcelFile(path) as workbook:
            overall: dict[str, float] = {}
            monthly: dict[str, dict[pd.Timestamp, float]] = {}

            if PROJECT_BASELINES_SHEET in workbook.sheet_names:
                df_overall = pd.read_excel(workbook, sheet_name=PROJECT_BASELINES_SHEET)
                if not df_overall.empty:
                    try:
                        project_col = _pick_column(df_overall, ("project_name", "Project Name"))
                        baseline_col = _pick_column(
                            df_overall,
                            ("baseline_mt_per_day", "Baseline", "baseline"),
                        )
                    except KeyError:
                        pass
                    else:
                        cleaned = df_overall[[project_col, baseline_col]].copy()
                        cleaned[project_col] = cleaned[project_col].astype(str).str.strip()
                        cleaned[baseline_col] = pd.to_numeric(cleaned[baseline_col], errors="coerce")
                        cleaned = cleaned.dropna(subset=[project_col, baseline_col])
                        for _, row in cleaned.iterrows():
                            name = str(row[project_col]).strip()
                            value = float(row[baseline_col])
                            if name:
                                overall[name] = value

            if PROJECT_BASELINES_MONTHLY_SHEET in workbook.sheet_names:
                df_monthly = pd.read_excel(workbook, sheet_name=PROJECT_BASELINES_MONTHLY_SHEET)
                if not df_monthly.empty:
                    try:
                        project_col = _pick_column(df_monthly, ("project_name", "Project Name"))
                        month_col = _pick_column(df_monthly, ("month", "Month"))
                        baseline_col = _pick_column(
                            df_monthly,
                            ("baseline_mt_per_day", "Baseline", "baseline"),
                        )
                    except KeyError:
                        pass
                    else:
                        cleaned = df_monthly[[project_col, month_col, baseline_col]].copy()
                        cleaned[project_col] = cleaned[project_col].astype(str).str.strip()
                        cleaned[baseline_col] = pd.to_numeric(cleaned[baseline_col], errors="coerce")
                        cleaned[month_col] = pd.to_datetime(cleaned[month_col], errors="coerce")
                        cleaned = cleaned.dropna(subset=[project_col, month_col, baseline_col])
                        for _, row in cleaned.iterrows():
                            project = str(row[project_col]).strip()
                            month_ts = pd.to_datetime(row[month_col])
                            value = float(row[baseline_col])
                            if project and not pd.isna(month_ts):
                                monthly.setdefault(project, {})[pd.Timestamp(month_ts)] = value
    except FileNotFoundError:
        LOGGER.warning(
            "Workbook '%s' not found when loading project baselines.",
            path,
        )
        _set_project_baseline_cache({}, {}, path)
        return get_project_baseline_maps()
    except Exception as exc:
        LOGGER.warning(
            "Unable to load project baselines from '%s': %s",
            path,
            exc,
        )
        return get_project_baseline_maps()

    _set_project_baseline_cache(overall, monthly, path)
    return get_project_baseline_maps()



def load_daily_from_proddailyexpanded(
    xl: pd.ExcelFile, sheet: str = "ProdDailyExpanded"
) -> pd.DataFrame:
    """Load daily productivity rows from a ProdDailyExpanded-style sheet."""

    LOGGER.debug("Loading data from sheet '%s'", sheet)
    df = pd.read_excel(xl, sheet_name=sheet)
    col_date = _pick_column(df, ["Work Date", "date"])
    col_prod = _pick_column(df, ["Productivity", "daily_prod_mt", "avg_daily_prod_mt"])
    col_proj = _pick_column(df, ["Project Name", "project_name"])
    col_gang = _pick_column(df, ["Gang name", "gang_name"])

    def _pick_optional(frame: pd.DataFrame, options: tuple[str, ...]) -> str | None:
        try:
            return _pick_column(frame, options)
        except KeyError:
            return None

    def _normalize_text(value: object) -> str:
        text = str(value).replace("\u00a0", " ").strip()
        lowered = text.lower()
        if lowered in {"", "nan", "none", "null"}:
            return ""
        return text

    def _normalize_location(value: object) -> str:
        text = _normalize_text(value)
        if not text:
            return ""
        if text.endswith(".0") and text.replace(".", "", 1).isdigit():
            text = text.split(".", 1)[0]
        return text

    data: dict[str, Any] = {
        "date": pd.to_datetime(df[col_date], errors="coerce").dt.normalize(),
        "daily_prod_mt": pd.to_numeric(df[col_prod], errors="coerce"),
        "project_name": df[col_proj].astype(str).str.strip(),
        "gang_name": df[col_gang].astype(str).str.strip(),
    }

    col_location = _pick_optional(df, ("Location No.", "location no", "location number", "location"))
    if col_location:
        data["location_no"] = df[col_location].map(_normalize_location)

    col_tower = _pick_optional(df, ("Tower Weight", "tower weight", "tower_weight", "tower wt", "tower mt"))
    if col_tower:
        data["tower_weight"] = pd.to_numeric(df[col_tower], errors="coerce")

    col_start = _pick_optional(df, ("Start Date", "starting date"))
    if col_start:
        data["start_date"] = pd.to_datetime(df[col_start], errors="coerce")

    col_complete = _pick_optional(df, ("Complete Date", "completion date"))
    if col_complete:
        data["completion_date"] = pd.to_datetime(df[col_complete], errors="coerce")

    col_status = _pick_optional(df, ("Status",))
    if col_status:
        data["status"] = df[col_status].astype(str).str.strip()

    result = pd.DataFrame(data).dropna(subset=["date", "daily_prod_mt"])
    LOGGER.debug("Loaded %d daily rows from %s", len(result), sheet)
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

    result: pd.DataFrame | None = None
    with pd.ExcelFile(config.data_path) as workbook:
        candidates: list[str] = []
        if config.preferred_sheet:
            candidates.append(config.preferred_sheet)
        candidates.extend([
            "ProdDailyExpandedSingles",
        ])

        seen: set[str] = set()
        for sheet_name in candidates:
            if sheet_name and sheet_name not in seen and sheet_name in workbook.sheet_names:
                result = load_daily_from_proddailyexpanded(workbook, sheet_name)
                break
            seen.add(sheet_name)

        if result is None and "RawData" in workbook.sheet_names:
            result = load_daily_from_rawdata(workbook, "RawData")

    if result is None:
        raise FileNotFoundError("Neither 'ProdDailyExpandedSingles' nor fallback sheets found in workbook.")

    _refresh_project_baselines(config.data_path, result)
    return result


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
