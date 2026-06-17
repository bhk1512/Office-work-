"""Foundation productivity analysis workbook helpers."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


SHEET_ORDER = [
    "Portfolio Summary",
    "Portfolio Monthly Trend",
    "Project Monthly Trend",
    "Gang Monthly Productivity",
    "Gang Summary",
    "Duration Summary",
    "Foundation Details",
    "Data Coverage",
]


def _compact_project(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _display_project(value: object) -> str:
    compact = _compact_project(value)
    match = re.match(r"^([A-Z]+)(\d+.*)$", compact)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return "" if pd.isna(value) else str(value).strip()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).replace("\u00a0", " ").strip()
    return "" if text.lower() in {"nan", "none", "nat", "null"} else text


def _load_pch_mapping(mapping: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["project_key", "PCH", "Project"]
    if mapping is None or mapping.empty:
        return pd.DataFrame(columns=columns)
    if "Project" not in mapping.columns or "PCH" not in mapping.columns:
        return pd.DataFrame(columns=columns)
    out = mapping[["Project", "PCH"]].copy()
    out["project_key"] = out["Project"].map(_compact_project)
    out["Project"] = out["Project"].map(_display_project)
    out["PCH"] = out["PCH"].fillna("Unassigned").astype(str).str.strip()
    out = out[out["project_key"].astype(bool)]
    return out.drop_duplicates("project_key", keep="first").reset_index(drop=True)


def _month_start(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    return pd.Timestamp(f"{value}-01" if re.fullmatch(r"\d{4}-\d{2}", str(value)) else value).to_period("M").to_timestamp()


def _month_end(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    start = _month_start(value)
    if start is None:
        return None
    return (start + pd.offsets.MonthEnd(1)).normalize()


def _percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _avg_duration(values: pd.Series) -> float | pd.NA:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(valid.mean()), 2) if not valid.empty else pd.NA


def _prepare_detail_rows(
    foundation_completions: pd.DataFrame,
    pch_mapping: pd.DataFrame | None,
    *,
    start_month: str | pd.Timestamp | None = None,
    end_month: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    base_columns = [
        "PCH",
        "Project",
        "Project Key",
        "Line",
        "Location No",
        "Gang",
        "Start Date",
        "Completion Date",
        "Month",
        "Duration Days",
        "Duration Status",
        "Source Type",
        "Quality Flag",
        "Source File",
        "Source Sheet",
    ]
    if foundation_completions is None or foundation_completions.empty:
        return pd.DataFrame(columns=base_columns)

    mapping = _load_pch_mapping(pch_mapping)
    work = foundation_completions.copy()
    for column in ("project_code", "project_display", "line_name", "location_no", "gang_name", "source_type", "quality_flag", "source_file", "source_sheet"):
        if column not in work.columns:
            work[column] = ""
    if "start_date" not in work.columns:
        work["start_date"] = pd.NaT
    if "event_date" not in work.columns:
        work["event_date"] = pd.NaT

    work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce").dt.normalize()
    work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce").dt.normalize()
    start_bound = _month_start(start_month)
    end_bound = _month_end(end_month)
    if start_bound is not None:
        work = work[work["event_date"].ge(start_bound)]
    if end_bound is not None:
        work = work[work["event_date"].le(end_bound)]

    work["source_type_norm"] = work["source_type"].fillna("").astype(str).str.strip().str.lower()
    work["project_key"] = work["project_code"].map(_compact_project)
    work["Project"] = work["project_display"].where(work["project_display"].fillna("").astype(str).str.strip().ne(""), work["project_code"])
    work["Project"] = work["Project"].map(_display_project)
    work = work.merge(mapping[["project_key", "PCH"]], on="project_key", how="left")
    work["PCH"] = work["PCH"].fillna("Unassigned")
    work["Line"] = work["line_name"].map(_clean_text)
    work["Location No"] = work["location_no"].map(_clean_text)
    gang = work["gang_name"].map(_clean_text)
    work["Gang"] = gang.mask(gang.eq(""), "Unassigned")
    work["Month"] = work["event_date"].dt.to_period("M").astype(str)
    work["duration_raw"] = (work["event_date"] - work["start_date"]).dt.days + 1
    valid_duration = work["start_date"].notna() & work["event_date"].notna() & work["duration_raw"].ge(1)
    work["Duration Days"] = pd.to_numeric(work["duration_raw"].where(valid_duration), errors="coerce")
    work["Duration Status"] = "Valid"
    work.loc[work["start_date"].isna(), "Duration Status"] = "Missing Start Date"
    work.loc[work["event_date"].isna(), "Duration Status"] = "Missing Completion Date"
    work.loc[work["start_date"].notna() & work["event_date"].notna() & work["duration_raw"].lt(1), "Duration Status"] = "Invalid Negative Duration"

    detail = pd.DataFrame(
        {
            "PCH": work["PCH"],
            "Project": work["Project"],
            "Project Key": work["project_key"],
            "Line": work["Line"],
            "Location No": work["Location No"],
            "Gang": work["Gang"],
            "Start Date": work["start_date"],
            "Completion Date": work["event_date"],
            "Month": work["Month"],
            "Duration Days": work["Duration Days"],
            "Duration Status": work["Duration Status"],
            "Source Type": work["source_type"],
            "Quality Flag": work["quality_flag"],
            "Source File": work["source_file"],
            "Source Sheet": work["source_sheet"],
        }
    )
    return detail.reindex(columns=base_columns)


def _productivity_rows(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details.copy()
    return details[
        details["Source Type"].fillna("").astype(str).str.strip().str.lower().eq("detail")
        & pd.to_datetime(details["Completion Date"], errors="coerce").notna()
    ].copy()


def _assigned_rows(productivity: pd.DataFrame) -> pd.DataFrame:
    if productivity.empty:
        return productivity.copy()
    return productivity[productivity["Gang"].fillna("").astype(str).str.strip().ne("Unassigned")].copy()


def _portfolio_summary(productivity: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    assigned = _assigned_rows(productivity)
    gang_month = (
        assigned.groupby(["Gang", "Month"], dropna=False)
        .size()
        .rename("Foundations")
        .reset_index()
    )
    duration_valid = pd.to_numeric(productivity.get("Duration Days"), errors="coerce").dropna()
    total = int(len(productivity.index))
    assigned_count = int(len(assigned.index))
    rows = [
        ("Total Foundations", total),
        ("Projects", int(productivity["Project Key"].nunique()) if not productivity.empty else 0),
        ("Gangs With Data", int(assigned["Gang"].nunique()) if not assigned.empty else 0),
        ("Active Gang-Months", int(len(gang_month.index))),
        ("Avg Foundations / Active Gang-Month", round(assigned_count / len(gang_month.index), 2) if len(gang_month.index) else pd.NA),
        ("Median Foundations / Active Gang-Month", round(float(gang_month["Foundations"].median()), 2) if not gang_month.empty else pd.NA),
        ("Avg Duration Days", round(float(duration_valid.mean()), 2) if not duration_valid.empty else pd.NA),
        ("Duration Coverage %", _percent(len(duration_valid.index), total)),
        ("Gang Name Coverage %", _percent(assigned_count, total)),
        ("First Completion Date", pd.to_datetime(productivity["Completion Date"], errors="coerce").min() if total else pd.NaT),
        ("Last Completion Date", pd.to_datetime(productivity["Completion Date"], errors="coerce").max() if total else pd.NaT),
        ("Snapshot Rows Excluded", int(details["Source Type"].fillna("").astype(str).str.lower().eq("snapshot_fallback").sum()) if not details.empty else 0),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _portfolio_monthly(productivity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Month",
        "Foundations Completed",
        "Projects Active",
        "Unique Gangs",
        "Active Gang-Months",
        "Avg Foundations / Active Gang-Month",
        "Median Foundations / Active Gang-Month",
        "Avg Duration Days",
        "Duration Coverage %",
        "Gang Name Coverage %",
    ]
    if productivity.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for month, group in productivity.groupby("Month", dropna=False):
        assigned = _assigned_rows(group)
        gang_counts = assigned.groupby("Gang").size()
        duration_valid = pd.to_numeric(group["Duration Days"], errors="coerce").dropna()
        rows.append(
            {
                "Month": month,
                "Foundations Completed": int(len(group.index)),
                "Projects Active": int(group["Project Key"].nunique()),
                "Unique Gangs": int(assigned["Gang"].nunique()),
                "Active Gang-Months": int(len(gang_counts.index)),
                "Avg Foundations / Active Gang-Month": round(len(assigned.index) / len(gang_counts.index), 2) if len(gang_counts.index) else pd.NA,
                "Median Foundations / Active Gang-Month": round(float(gang_counts.median()), 2) if not gang_counts.empty else pd.NA,
                "Avg Duration Days": round(float(duration_valid.mean()), 2) if not duration_valid.empty else pd.NA,
                "Duration Coverage %": _percent(len(duration_valid.index), len(group.index)),
                "Gang Name Coverage %": _percent(len(assigned.index), len(group.index)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("Month")


def _project_monthly(productivity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "PCH",
        "Project",
        "Month",
        "Foundations Completed",
        "Unique Gangs",
        "Avg Foundations / Active Gang-Month",
        "Avg Duration Days",
        "Valid Duration Foundations",
        "Duration Coverage %",
    ]
    if productivity.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (pch, project, month), group in productivity.groupby(["PCH", "Project", "Month"], dropna=False):
        assigned = _assigned_rows(group)
        duration_valid = pd.to_numeric(group["Duration Days"], errors="coerce").dropna()
        unique_gangs = int(assigned["Gang"].nunique())
        rows.append(
            {
                "PCH": pch,
                "Project": project,
                "Month": month,
                "Foundations Completed": int(len(group.index)),
                "Unique Gangs": unique_gangs,
                "Avg Foundations / Active Gang-Month": round(len(assigned.index) / unique_gangs, 2) if unique_gangs else pd.NA,
                "Avg Duration Days": round(float(duration_valid.mean()), 2) if not duration_valid.empty else pd.NA,
                "Valid Duration Foundations": int(len(duration_valid.index)),
                "Duration Coverage %": _percent(len(duration_valid.index), len(group.index)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["PCH", "Project", "Month"])


def _gang_monthly(productivity: pd.DataFrame) -> pd.DataFrame:
    columns = ["PCH", "Project", "Month", "Gang", "Foundations Completed", "Avg Duration Days", "Valid Duration Foundations"]
    assigned = _assigned_rows(productivity)
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        assigned.groupby(["PCH", "Project", "Month", "Gang"], dropna=False)
        .agg(
            **{
                "Foundations Completed": ("Location No", "size"),
                "Avg Duration Days": ("Duration Days", _avg_duration),
                "Valid Duration Foundations": ("Duration Days", lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())),
            }
        )
        .reset_index()
    )
    return grouped.reindex(columns=columns).sort_values(["PCH", "Project", "Month", "Gang"])


def _gang_summary(productivity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Gang",
        "Total Foundations",
        "Active Months",
        "Avg Foundations / Active Month",
        "Median Monthly Foundations",
        "Best Month Foundations",
        "Projects Worked",
        "Project List",
        "Avg Duration Days",
        "Valid Duration Foundations",
    ]
    assigned = _assigned_rows(productivity)
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    month_counts = assigned.groupby(["Gang", "Month"]).size().rename("Monthly Foundations").reset_index()
    rows = []
    for gang, group in assigned.groupby("Gang", dropna=False):
        gang_months = month_counts[month_counts["Gang"].eq(gang)]["Monthly Foundations"]
        duration_valid = pd.to_numeric(group["Duration Days"], errors="coerce").dropna()
        project_list = sorted({str(value).strip() for value in group["Project"] if str(value).strip()})
        rows.append(
            {
                "Gang": gang,
                "Total Foundations": int(len(group.index)),
                "Active Months": int(group["Month"].nunique()),
                "Avg Foundations / Active Month": round(float(gang_months.mean()), 2) if not gang_months.empty else pd.NA,
                "Median Monthly Foundations": round(float(gang_months.median()), 2) if not gang_months.empty else pd.NA,
                "Best Month Foundations": int(gang_months.max()) if not gang_months.empty else 0,
                "Projects Worked": int(group["Project Key"].nunique()),
                "Project List": "; ".join(project_list),
                "Avg Duration Days": round(float(duration_valid.mean()), 2) if not duration_valid.empty else pd.NA,
                "Valid Duration Foundations": int(len(duration_valid.index)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["Total Foundations", "Gang"], ascending=[False, True])


def _duration_summary(productivity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "PCH",
        "Project",
        "Month",
        "Gang",
        "Foundations Completed",
        "Valid Duration Foundations",
        "Avg Duration Days",
        "Median Duration Days",
        "Min Duration Days",
        "Max Duration Days",
        "Missing Start Date Count",
        "Invalid Duration Count",
    ]
    if productivity.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in productivity.groupby(["PCH", "Project", "Month", "Gang"], dropna=False):
        valid = pd.to_numeric(group["Duration Days"], errors="coerce").dropna()
        rows.append(
            {
                "PCH": keys[0],
                "Project": keys[1],
                "Month": keys[2],
                "Gang": keys[3],
                "Foundations Completed": int(len(group.index)),
                "Valid Duration Foundations": int(len(valid.index)),
                "Avg Duration Days": round(float(valid.mean()), 2) if not valid.empty else pd.NA,
                "Median Duration Days": round(float(valid.median()), 2) if not valid.empty else pd.NA,
                "Min Duration Days": int(valid.min()) if not valid.empty else pd.NA,
                "Max Duration Days": int(valid.max()) if not valid.empty else pd.NA,
                "Missing Start Date Count": int(group["Duration Status"].eq("Missing Start Date").sum()),
                "Invalid Duration Count": int(group["Duration Status"].eq("Invalid Negative Duration").sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["PCH", "Project", "Month", "Gang"])


def _data_coverage(details: pd.DataFrame, raw: pd.DataFrame | None, coverage: pd.DataFrame | None) -> pd.DataFrame:
    productivity = _productivity_rows(details)
    rows = []

    def add(section: str, metric: str, value: object, project: str = "", status: str = "", reason: str = "") -> None:
        rows.append(
            {
                "Section": section,
                "Project": project,
                "Status": status,
                "Metric": metric,
                "Value": value,
                "Reason": reason,
            }
        )

    add("Input", "Total rows after month filter", int(len(details.index)))
    add("Input", "FoundationRaw rows available", int(len(raw.index)) if raw is not None else 0)
    add("Input", "Detail rows used for productivity", int(len(productivity.index)))
    add("Input", "Snapshot fallback rows excluded from productivity", int(details["Source Type"].fillna("").astype(str).str.lower().eq("snapshot_fallback").sum()) if not details.empty else 0)
    add("Data Quality", "Rows with missing gang", int(productivity["Gang"].eq("Unassigned").sum()) if not productivity.empty else 0)
    add("Data Quality", "Rows with missing start date", int(productivity["Duration Status"].eq("Missing Start Date").sum()) if not productivity.empty else 0)
    add("Data Quality", "Rows with invalid duration", int(productivity["Duration Status"].eq("Invalid Negative Duration").sum()) if not productivity.empty else 0)
    add("Data Quality", "Rows with valid duration", int(pd.to_numeric(productivity.get("Duration Days"), errors="coerce").notna().sum()) if not productivity.empty else 0)

    if coverage is not None and not coverage.empty:
        missing_status = {"MISSING", "NO_TARGET_SHEET", "BLOCKED_NO_SOURCE", "SKIPPED_NOT_IN_CONFIG", "SKIPPED_BLANK_CONFIG", "MAPPING_CONFIRMATION_REQUIRED"}
        work = coverage.copy()
        for col in ("project_code", "status", "reason"):
            if col not in work.columns:
                work[col] = ""
        for _, row in work[work["status"].fillna("").astype(str).str.upper().isin(missing_status)].iterrows():
            add(
                "Coverage",
                "Project with missing/limited foundation source",
                "",
                project=_display_project(row.get("project_code", "")),
                status=str(row.get("status", "")),
                reason=str(row.get("reason", "")),
            )
    return pd.DataFrame(rows, columns=["Section", "Project", "Status", "Metric", "Value", "Reason"])


def build_foundation_productivity_tables(
    foundation_completions: pd.DataFrame,
    foundation_raw: pd.DataFrame | None = None,
    foundation_coverage: pd.DataFrame | None = None,
    pch_mapping: pd.DataFrame | None = None,
    *,
    start_month: str | pd.Timestamp | None = None,
    end_month: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Build foundation productivity analysis tables."""
    details = _prepare_detail_rows(
        foundation_completions,
        pch_mapping,
        start_month=start_month,
        end_month=end_month,
    )
    productivity = _productivity_rows(details)
    tables = {
        "Portfolio Summary": _portfolio_summary(productivity, details),
        "Portfolio Monthly Trend": _portfolio_monthly(productivity),
        "Project Monthly Trend": _project_monthly(productivity),
        "Gang Monthly Productivity": _gang_monthly(productivity),
        "Gang Summary": _gang_summary(productivity),
        "Duration Summary": _duration_summary(productivity),
        "Foundation Details": details,
        "Data Coverage": _data_coverage(details, foundation_raw, foundation_coverage),
    }
    return tables


def write_foundation_productivity_workbook(output_path: str | Path, tables: dict[str, pd.DataFrame]) -> Path:
    """Write foundation productivity analysis tables to Excel."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in SHEET_ORDER:
            table = tables.get(sheet, pd.DataFrame())
            table.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                values = [str(cell.value) for cell in column_cells if cell.value is not None]
                width = min(max([len(value) for value in values] + [10]) + 2, 48)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return output
