"""Excel workbook export helpers."""
from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from math import ceil
from typing import Sequence, TYPE_CHECKING

import pandas as pd

from .config import AppConfig
from .metrics import (
    calc_idle_and_loss,
    compute_gang_baseline_maps,
    compute_idle_intervals_per_gang,
    compute_project_baseline_maps_for,
)
from .pch_normalizer import CANONICAL_PCH_PRIMARY, normalize_pch

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .state import AppDataStore

LOGGER = logging.getLogger(__name__)

AVG_PRODUCTIVITY_COLUMN = "Avg Productivity (MTD)"
TOTAL_MT_COLUMN = "Total MT (For this month) (MTD)"
TOTAL_COUNT_COLUMN = "Total No. of Erections (For this month) (MTD)"
_DEFAULT_SHEET_NAME = "Erection Summary"
_DEFAULT_KV_SHEET_NAME = "KV Productivity"
_PCH_SORT_ORDER = {name: idx for idx, name in enumerate(CANONICAL_PCH_PRIMARY)}

# TODO: Align the week bucket helper below with the dashboard's official week mapping
# once that logic is exposed outside the callbacks module.


def _generate_week_labels(month_start: pd.Timestamp, month_end: pd.Timestamp) -> list[str]:
    days = max(1, int((month_end - month_start).days) + 1)
    week_count = min(4, max(1, ceil(days / 7)))  # cap at 4 to avoid super-short trailing week
    return [f"Week {idx}" for idx in range(1, week_count + 1)]


def _sanitize_sheet_name(value: str) -> str:
    """Return a value safe for use as an Excel sheet name."""

    sanitized = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(value))
    return sanitized[:31]


def make_trace_workbook_bytes(
    scoped_data: pd.DataFrame,
    months_ts: Sequence[pd.Timestamp] | None,
    projects: Sequence[str] | None,
    gangs: Sequence[str] | None,
    bench: float,
    *,
    gang_for_sheet: str | None = None,
    config: AppConfig | None = None,
    project_info: pd.DataFrame | None = None,
    erections_completed: pd.DataFrame | None = None,
    erections_context: dict[str, object] | None = None,
) -> bytes:
    """Build the Excel export with summary, idle intervals, and daily detail."""

    active_config = config or AppConfig()
    LOGGER.info("Building trace workbook (rows=%d)", len(scoped_data))

    overall_baseline_map, monthly_baseline_map = compute_gang_baseline_maps(scoped_data)

    summary_rows: list[dict[str, object]] = []
    for gang_name, gang_df in scoped_data.groupby("gang_name"):
        overall_baseline = overall_baseline_map.get(gang_name)
        monthly_baseline = monthly_baseline_map.get(gang_name)
        idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
            gang_df,
            loss_max_gap_days=active_config.loss_max_gap_days,
            baseline_mt_per_day=overall_baseline,
            baseline_by_month=monthly_baseline,
        )
        summary_rows.append(
            {
                "gang_name": gang_name,
                "delivered_mt": delivered,
                "lost_mt": loss_mt,
                "potential_mt": potential,
                "baseline_mt_per_day": baseline,
                "idle_days_capped": idle,
                "first_date": gang_df["date"].min(),
                "last_date": gang_df["date"].max(),
                "active_days": gang_df["date"].nunique(),
            }
        )
    per_gang_summary = (
        pd.DataFrame(summary_rows).sort_values("potential_mt", ascending=False)
        if summary_rows
        else pd.DataFrame(
            columns=[
                "gang_name",
                "delivered_mt",
                "lost_mt",
                "potential_mt",
                "baseline_mt_per_day",
                "idle_days_capped",
                "first_date",
                "last_date",
                "active_days",
            ]
        )
    )

    idle_df = compute_idle_intervals_per_gang(
        scoped_data,
        loss_max_gap_days=active_config.loss_max_gap_days,
        baseline_month_lookup=monthly_baseline_map,
        baseline_fallback_map=overall_baseline_map,
    )
    daily_df = (
        scoped_data.sort_values(["gang_name", "date"])
        [["date", "gang_name", "project_name", "daily_prod_mt"]]
        .copy()
    )
    project_month = (
        scoped_data.groupby(["project_name", "month"])["daily_prod_mt"].mean().reset_index()
    )

    context_row = {
        "projects": ", ".join(projects or []) or "(all)",
        "gangs": ", ".join(gangs or []) or "(all)",
        "months": ", ".join(
            [timestamp.strftime("%Y-%m") for timestamp in (months_ts or [])]
        )
        or "(all / overall)",
        "benchmark": bench,
        "loss_cap_days": active_config.loss_max_gap_days,
    }
    if erections_context:
        def _format_context_date(value: object) -> str:
            if isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%d")
            if value:
                return str(value)
            return ""
        context_row["erections_range_start"] = _format_context_date(
            erections_context.get("range_start")
        )
        context_row["erections_range_end"] = _format_context_date(
            erections_context.get("range_end")
        )
        search_value = erections_context.get("search_text")
        context_row["erections_search"] = (search_value or "").strip()
    context_df = pd.DataFrame([context_row])

    assumptions = pd.DataFrame(
        {
            "Notes": [
                f"Loss cap per gap: {active_config.loss_max_gap_days} days.",
                "Efficiency = delivered / (delivered + lost). Lost = baseline * capped idle days.",
                "Idle interval = gaps between observed work dates; dates inferred from current filtered scope.",
                "All numbers reflect current dashboard filters (project, period, and gang if applied).",
            ]
        }
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        per_gang_summary.to_excel(writer, "PerGangSummary", index=False)
        idle_df.to_excel(writer, "IdleIntervals", index=False)
        daily_df.to_excel(writer, "DailyProductivity", index=False)
        project_month.to_excel(writer, "ProjectsMonthly", index=False)
        context_df.to_excel(writer, "SelectionContext", index=False)
        assumptions.to_excel(writer, "Assumptions", index=False)

        if erections_completed is not None:
            completions_sheet = erections_completed.copy()
            if "completion_date" in completions_sheet.columns:
                completions_sheet["completion_date"] = completions_sheet["completion_date"].apply(
                    lambda value: value.date() if isinstance(value, pd.Timestamp) else value
                )
            if "start_date" in completions_sheet.columns:
                completions_sheet["start_date"] = completions_sheet["start_date"].apply(
                    lambda value: value.date() if isinstance(value, pd.Timestamp) else (None if pd.isna(value) else value)
                )
            if "supervisor_name" in completions_sheet.columns:
                completions_sheet["supervisor_name"] = completions_sheet["supervisor_name"].fillna("")
            else:
                completions_sheet["supervisor_name"] = ""
            if "section_incharge_name" in completions_sheet.columns:
                completions_sheet["section_incharge_name"] = completions_sheet["section_incharge_name"].fillna("")
            else:
                completions_sheet["section_incharge_name"] = ""
            completions_sheet = completions_sheet.rename(
                columns={
                    "completion_date": "Completion Date",
                    "project_name": "Project",
                    "location_no": "Location",
                    "tower_weight_mt": "Tower Weight (MT)",
                    "daily_prod_mt": "Productivity (MT/day)",
                    "gang_name": "Gang",
                    "start_date": "Start Date",
                    "supervisor_name": "Supervisor",
                    "section_incharge_name": "Section Incharge",
                    "revenue_value": "Revenue",
                }
            )
            completions_sheet.to_excel(writer, "ErectionsCompleted", index=False)

        if gang_for_sheet:
            selected = scoped_data[scoped_data["gang_name"] == gang_for_sheet]
            if not selected.empty:
                single_idle = compute_idle_intervals_per_gang(
                    selected,
                    loss_max_gap_days=active_config.loss_max_gap_days,
                    baseline_month_lookup=monthly_baseline_map,
                    baseline_fallback_map=overall_baseline_map,
                )
                single_idle.to_excel(
                    writer,
                    _sanitize_sheet_name(f"Idle_{gang_for_sheet}"),
                    index=False,
                )
                (
                    selected.sort_values("date")[["date", "project_name", "daily_prod_mt"]]
                    .assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d"))
                    .to_excel(writer, _sanitize_sheet_name(f"Daily_{gang_for_sheet}"), index=False)
                )

                idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                    selected,
                    loss_max_gap_days=active_config.loss_max_gap_days,
                    baseline_mt_per_day=overall_baseline_map.get(gang_for_sheet),
                    baseline_by_month=monthly_baseline_map.get(gang_for_sheet),
                )
                efficiency = (delivered / potential * 100) if potential > 0 else 0.0
                pd.DataFrame(
                    [
                        {
                            "gang_name": gang_for_sheet,
                            "delivered_mt": delivered,
                            "lost_mt": loss_mt,
                            "potential_mt": potential,
                            "efficiency_%": efficiency,
                            "baseline_mt_per_day": baseline,
                            "idle_days_capped": idle,
                        }
                    ]
                ).to_excel(
                    writer,
                    _sanitize_sheet_name(f"Summary_{gang_for_sheet}"),
                    index=False,
                )
        
        
        # Optional: include ProjectDetails if exactly one project is selected
        if projects and len(projects) == 1 and project_info is not None and not project_info.empty:
            pname = str(projects[0]).strip()
            # project_name -> project_code from scoped_data (or fall back to global if needed)
            name_to_code = (
                scoped_data.dropna(subset=["project_name", "project_code"])
                           .drop_duplicates(subset=["project_name"])
                           .set_index("project_name")["project_code"]
                           .to_dict()
            )
            pcode = name_to_code.get(pname)
            if pcode:
                row = project_info[project_info["project_code"] == pcode]
                if not row.empty:
                    r = row.iloc[0]
                    pd.DataFrame([{
                        "Project Code": pcode,
                        "Project Name": pname,
                        "Client Name": r.get("client_name"),
                        "NOA Start Date": r.get("noa_start"),
                        "LOA End Date": r.get("loa_end"),
                        "Project Manager": r.get("project_mgr"),
                        "Regional Manager": r.get("regional_mgr"),
                        "Planning Engineer": r.get("planning_eng"),
                        "PCH": r.get("pch"),
                        "Section Incharge": r.get("section_inch"),
                        "Supervisor": r.get("supervisor"),
                    }]).to_excel(writer, "ProjectDetails_Selected", index=False)


    buffer.seek(0)
    return buffer.getvalue()



def _project_lookup_series(project_info: pd.DataFrame | None, column: str) -> pd.Series:
    if project_info is None or project_info.empty:
        return pd.Series(dtype="object")
    if "key_name" not in project_info.columns or column not in project_info.columns:
        return pd.Series(dtype="object")
    lookup = (
        project_info[["key_name", column]]
        .dropna(subset=["key_name"])
        .drop_duplicates("key_name")
        .set_index("key_name")[column]
    )
    return lookup


def _compact_project_key(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _prepare_project_voltage_lookup(project_kv: pd.DataFrame | None) -> dict[str, str]:
    if project_kv is None or project_kv.empty:
        return {}

    working = project_kv.copy()

    def _pick_col(options: Sequence[str]) -> str:
        lowered = {str(col).strip().lower(): col for col in working.columns}
        for option in options:
            normalized = option.strip().lower()
            if normalized in lowered:
                return lowered[normalized]
        for option in options:
            normalized = option.strip().lower()
            for candidate, original in lowered.items():
                if normalized and normalized in candidate:
                    return original
        raise KeyError

    try:
        project_col = _pick_col(["project", "project_code", "project name"])
        voltage_col = _pick_col(["voltage", "kv", "kv level"])
    except KeyError:
        LOGGER.warning("Projects_KV workbook missing required columns; voltage lookup disabled.")
        return {}

    lookup: dict[str, str] = {}
    for _, row in working.iterrows():
        project_value = row.get(project_col)
        voltage_value = row.get(voltage_col)
        key = _compact_project_key(project_value)
        if not key:
            continue
        if voltage_value is None or (isinstance(voltage_value, float) and pd.isna(voltage_value)):
            continue
        voltage_text = str(voltage_value).strip()
        if not voltage_text:
            continue
        lookup[key] = voltage_text
    return lookup


def _annotate_scope_with_voltage(scope: pd.DataFrame, voltage_lookup: dict[str, str]) -> pd.DataFrame:
    if scope is None or scope.empty:
        return pd.DataFrame()

    working = scope.copy()
    tower_series = working.get("tower_type")
    if tower_series is None:
        working["tower_type"] = "Unknown"
    else:
        normalized = tower_series.fillna(" ").astype(str).str.strip().str.upper()
        working["tower_type"] = normalized.where(normalized.astype(bool), "Unknown")
    family = working["tower_type"].str.extract(r"^(DA|DB|DC|DD)", expand=False)
    working["tower_family"] = family.fillna("Unknown")

    code_series = working.get("project_code")
    code_keys = code_series.fillna("" ).map(_compact_project_key) if code_series is not None else pd.Series("", index=working.index)
    name_series = working.get("project_name")
    fallback_keys = name_series.fillna("" ).map(_compact_project_key) if name_series is not None else pd.Series("", index=working.index)
    key_series = code_keys.where(code_keys.astype(bool), fallback_keys)
    working["voltage_label"] = key_series.map(voltage_lookup)
    working["voltage_label"] = working["voltage_label"].fillna("Unmapped")

    return working


def _build_project_meta_lookup(project_info: pd.DataFrame | None) -> dict[str, dict[str, str]]:
    if project_info is None or project_info.empty:
        return {}
    info = project_info.copy()
    info["project_code_value"] = (
        info.get("project_code", info.get("Project Code", ""))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    info["project_display_value"] = (
        info.get("Project Name", info.get("project_name", info["project_code_value"]))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    pch_col = None
    for candidate in ("pch", "PCH", "PCH Name", "pch_name"):
        if candidate in info.columns:
            pch_col = candidate
            break
    if pch_col is None:
        info["pch_value"] = ""
    else:
        info["pch_value"] = info[pch_col]

    lookup: dict[str, dict[str, str]] = {}
    for _, row in info.iterrows():
        entry = {
            "project_code": str(row.get("project_code_value", "")).strip(),
            "project_display": str(row.get("project_display_value", "")).strip(),
            "pch": normalize_pch(row.get("pch_value", "")),
        }
        keys = {
            _compact_project_key(row.get("project_code_value")),
            _compact_project_key(row.get("project_display_value")),
            _compact_project_key(row.get("key_name", "")),
        }
        for key in keys:
            if key and key not in lookup:
                lookup[key] = entry
    return lookup


def _assign_week_bucket(dates: pd.Series, month_start: pd.Timestamp, week_labels: Sequence[str]) -> pd.Series:
    offsets = (dates - month_start).dt.days
    buckets = ((offsets // 7) + 1).clip(lower=1, upper=len(week_labels))
    return buckets.fillna(0).astype(int)


def _prepare_month_scope(
    daily_df: pd.DataFrame | None,
    project_info: pd.DataFrame | None,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    week_labels: Sequence[str],
) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    working = daily_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"])
    scope = working[(working["date"] >= month_start) & (working["date"] <= month_end)].copy()
    if scope.empty:
        return scope

    scope["daily_prod_mt"] = pd.to_numeric(scope.get("daily_prod_mt"), errors="coerce").fillna(0.0)

    scope["project_name"] = scope.get("project_name", "").fillna("").astype(str).str.strip()
    meta_lookup = _build_project_meta_lookup(project_info)
    scope["project_code"] = scope.get("project_code", "").fillna("").astype(str).str.strip()
    scope["project_display"] = scope["project_code"].where(scope["project_code"].astype(bool), scope["project_name"])
    scope["project_display"] = scope["project_display"].fillna("").astype(str).str.strip()
    scope["project_display"] = scope["project_display"].where(scope["project_display"].astype(bool), scope["project_name"])
    scope["project_display"] = scope["project_display"].where(
        scope["project_display"].astype(bool), "(Unlabeled Project)"
    )
    scope["project_name"] = scope["project_name"].where(scope["project_name"].astype(bool), scope["project_display"])

    def _lookup_meta(value: object) -> dict[str, str] | None:
        if not meta_lookup:
            return None
        key = _compact_project_key(value)
        return meta_lookup.get(key)

    if meta_lookup:
        meta_from_code = scope["project_code"].map(_lookup_meta)
        meta_from_name = scope["project_name"].map(_lookup_meta)
        meta_series = meta_from_code.where(meta_from_code.notna(), meta_from_name)

        scope["project_code"] = scope["project_code"].where(
            scope["project_code"].astype(bool),
            meta_series.map(lambda rec: rec.get("project_code", "") if isinstance(rec, dict) else ""),
        )
        scope["project_display"] = scope["project_display"].where(
            scope["project_display"].astype(bool),
            meta_series.map(lambda rec: rec.get("project_display", "") if isinstance(rec, dict) else ""),
        )
        scope["project_display"] = scope["project_display"].fillna("").astype(str).str.strip()
        scope["project_display"] = scope["project_display"].where(scope["project_display"].astype(bool), scope["project_name"])
        scope["project_display"] = scope["project_display"].where(
            scope["project_display"].astype(bool), "(Unlabeled Project)"
        )
        scope["pch_display"] = meta_series.map(lambda rec: rec.get("pch", "") if isinstance(rec, dict) else "")
    else:
        scope["pch_display"] = ""

    scope["pch_display"] = scope["pch_display"].where(scope["pch_display"].astype(bool), scope.get("pch", ""))
    scope["pch_display"] = scope["pch_display"].fillna("").astype(str)
    scope["pch_display"] = scope["pch_display"].map(normalize_pch)
    scope["pch_display"] = scope["pch_display"].where(scope["pch_display"].astype(bool), "Unassigned")

    scope["location_no"] = scope.get("location_no", "").fillna("").astype(str).str.strip()
    scope["completion_date"] = pd.to_datetime(scope.get("completion_date"), errors="coerce").dt.normalize()
    scope["week_index"] = _assign_week_bucket(scope["date"], month_start, week_labels)

    return scope


def _prepare_completion_scope(
    scope: pd.DataFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    week_labels: Sequence[str],
) -> pd.DataFrame:
    if scope.empty:
        return pd.DataFrame()
    completion_mask = (
        scope["completion_date"].notna()
        & scope["date"].eq(scope["completion_date"])
        & (scope["completion_date"] >= month_start)
        & (scope["completion_date"] <= month_end)
    )
    completed = scope[completion_mask].copy()
    if completed.empty:
        return completed
    dedup_cols = [col for col in ("project_name_key", "location_no", "completion_date") if col in completed.columns]
    if dedup_cols:
        completed = completed.drop_duplicates(subset=dedup_cols)
    completed["completion_week_index"] = _assign_week_bucket(completed["completion_date"], month_start, week_labels)
    return completed


def _safe_average(values: list[object]) -> float:
    filtered = [float(v) for v in values if v is not None and not pd.isna(v)]
    return round(sum(filtered) / len(filtered), 2) if filtered else 0.0


def _compute_weekly_productivity_maps(scope: pd.DataFrame, week_labels: Sequence[str]) -> dict[str, dict[str, float]]:
    maps = {label: {} for label in week_labels}
    if scope.empty:
        return maps
    for week_idx, label in enumerate(week_labels, start=1):
        week_scope = scope[scope.get("week_index") == week_idx]
        if week_scope.empty:
            continue
        project_map, _ = compute_project_baseline_maps_for(week_scope, "daily_prod_mt")
        if project_map:
            maps[label] = project_map
    return maps


def _build_productivity_tables(
    scope: pd.DataFrame,
    completions: pd.DataFrame,
    week_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pch_columns = ["PCH", *week_labels, AVG_PRODUCTIVITY_COLUMN, TOTAL_MT_COLUMN, TOTAL_COUNT_COLUMN]
    project_columns = ["PCH", "Project", *week_labels, AVG_PRODUCTIVITY_COLUMN, TOTAL_MT_COLUMN, TOTAL_COUNT_COLUMN]

    if scope.empty:
        empty_pch = pd.DataFrame(columns=pch_columns)
        empty_project = pd.DataFrame(columns=project_columns)
        return empty_pch, empty_project

    project_meta = (
        scope[["pch_display", "project_name", "project_display"]]
        .dropna(subset=["project_name"])
        .drop_duplicates(subset=["pch_display", "project_name"])
        .reset_index(drop=True)
    )
    week_maps = _compute_weekly_productivity_maps(scope, week_labels)

    project_avg_map = (
        scope.groupby("project_name")["daily_prod_mt"].mean().to_dict()
        if not scope.empty
        else {}
    )
    pch_avg_map = (
        scope.groupby("pch_display")["daily_prod_mt"].mean().to_dict()
        if not scope.empty
        else {}
    )

    tower_weights = pd.to_numeric(completions.get("tower_weight"), errors="coerce") if isinstance(completions, pd.DataFrame) else pd.Series(dtype=float)
    if isinstance(completions, pd.DataFrame):
        completions = completions.assign(_tower_weight=tower_weights.fillna(0.0))
    else:
        completions = pd.DataFrame(columns=["pch_display", "project_name", "_tower_weight"])

    mt_totals_by_project = (
        completions.groupby(["pch_display", "project_name"])["_tower_weight"].sum().to_dict()
        if not completions.empty
        else {}
    )
    mt_totals_by_pch = (
        completions.groupby("pch_display")["_tower_weight"].sum().to_dict()
        if not completions.empty
        else {}
    )

    counts_by_project: dict[tuple[str, str], int] = {}
    counts_by_pch: dict[str, int] = {}
    if isinstance(completions, pd.DataFrame) and not completions.empty:
        counts_by_project = completions.groupby(["pch_display", "project_name"]).size().to_dict()
        counts_by_pch = completions.groupby("pch_display").size().to_dict()

    pch_rows: list[dict[str, object]] = []
    if not project_meta.empty:
        for pch_name in sorted(project_meta["pch_display"].unique(), key=lambda name: _pch_sort_components(name)):
            projects_in_pch = project_meta[project_meta["pch_display"] == pch_name]["project_name"].tolist()
            row = {"PCH": pch_name}
            for label in week_labels:
                project_values = [week_maps.get(label, {}).get(name) for name in projects_in_pch]
                row[label] = _safe_average(project_values)
            avg_value = pch_avg_map.get(pch_name)
            row[AVG_PRODUCTIVITY_COLUMN] = round(float(avg_value), 2) if avg_value is not None and not pd.isna(avg_value) else 0.0
            row[TOTAL_MT_COLUMN] = round(float(mt_totals_by_pch.get(pch_name, 0.0)), 2)
            row[TOTAL_COUNT_COLUMN] = int(counts_by_pch.get(pch_name, 0))
            pch_rows.append(row)

    if pch_rows:
        pch_summary = pd.DataFrame(pch_rows).reindex(columns=pch_columns).fillna(0.0)
        pch_summary = _sort_pch_frame(pch_summary, column="PCH")
    else:
        pch_summary = pd.DataFrame(columns=pch_columns)

    project_rows: list[dict[str, object]] = []
    for _, meta_row in project_meta.sort_values(["pch_display", "project_display"], key=lambda col: col.astype(str)).iterrows():
        pch_value = meta_row["pch_display"]
        project_name = meta_row["project_name"]
        display_name = meta_row["project_display"]
        row = {"PCH": pch_value, "Project": display_name}
        for label in week_labels:
            value = week_maps.get(label, {}).get(project_name)
            row[label] = round(float(value), 2) if value is not None and not pd.isna(value) else 0.0
        avg_value = project_avg_map.get(project_name)
        row[AVG_PRODUCTIVITY_COLUMN] = round(float(avg_value), 2) if avg_value is not None and not pd.isna(avg_value) else 0.0
        row[TOTAL_MT_COLUMN] = round(float(mt_totals_by_project.get((pch_value, project_name), 0.0)), 2)
        row[TOTAL_COUNT_COLUMN] = int(counts_by_project.get((pch_value, project_name), 0))
        project_rows.append(row)

    if project_rows:
        project_summary = pd.DataFrame(project_rows).reindex(columns=project_columns).fillna(0.0)
        order_components = project_summary["PCH"].map(_pch_sort_components)
        project_summary = project_summary.assign(
            _pch_order_bucket=order_components.map(lambda pair: pair[0]),
            _pch_order_value=order_components.map(lambda pair: pair[1]),
            _project_order=project_summary["Project"].astype(str).str.lower(),
        )
        project_summary = (
            project_summary.sort_values(by=["_pch_order_bucket", "_pch_order_value", "_project_order"])
            .drop(columns=["_pch_order_bucket", "_pch_order_value", "_project_order"])
            .reset_index(drop=True)
        )
    else:
        project_summary = pd.DataFrame(columns=project_columns)

    return pch_summary, project_summary


def _pch_sort_components(value: object) -> tuple[int, str]:
    label = str(value or "").strip()
    if label in _PCH_SORT_ORDER:
        return (0, f"{_PCH_SORT_ORDER[label]:03d}")
    lowered = label.lower()
    if not lowered or lowered == "unassigned":
        return (2, lowered)
    return (1, lowered)


def _sort_pch_frame(df: pd.DataFrame, column: str = "PCH") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    order_components = df[column].map(_pch_sort_components)
    df = df.assign(
        _pch_order_bucket=order_components.map(lambda pair: pair[0]),
        _pch_order_value=order_components.map(lambda pair: pair[1]),
    )
    df = df.sort_values(by=["_pch_order_bucket", "_pch_order_value"]).drop(columns=["_pch_order_bucket", "_pch_order_value"])
    return df.reset_index(drop=True)


def _build_group_productivity_table(
    scope: pd.DataFrame,
    completions: pd.DataFrame,
    week_labels: Sequence[str],
    *,
    group_columns: Sequence[str],
    column_labels: dict[str, str],
) -> pd.DataFrame:
    labeled_columns = [column_labels.get(col, col) for col in group_columns]
    ordered_columns = [*labeled_columns, *week_labels, AVG_PRODUCTIVITY_COLUMN, TOTAL_MT_COLUMN, TOTAL_COUNT_COLUMN]

    if scope is None or scope.empty:
        return pd.DataFrame(columns=ordered_columns)

    grouping = list(group_columns)
    summary = (
        scope.groupby(grouping, dropna=False)["daily_prod_mt"]
        .mean()
        .reset_index()
        .rename(columns={"daily_prod_mt": AVG_PRODUCTIVITY_COLUMN})
    )

    for idx, label in enumerate(week_labels, start=1):
        week_scope = scope[scope.get("week_index") == idx]
        if week_scope.empty:
            summary[label] = 0.0
            continue
        week_avg = (
            week_scope.groupby(grouping, dropna=False)["daily_prod_mt"]
            .mean()
            .reset_index()
            .rename(columns={"daily_prod_mt": label})
        )
        summary = summary.merge(week_avg, on=grouping, how="left")
        summary[label] = summary[label].fillna(0.0)

    weights = None
    counts = None
    if isinstance(completions, pd.DataFrame) and not completions.empty:
        tower_series = pd.to_numeric(completions.get("_tower_weight"), errors="coerce")
        weights = (
            completions.assign(_tower_weight=tower_series.fillna(0.0))
            .groupby(grouping, dropna=False)["_tower_weight"]
            .sum()
            .reset_index()
            .rename(columns={"_tower_weight": TOTAL_MT_COLUMN})
        )
        counts = (
            completions.groupby(grouping, dropna=False)
            .size()
            .reset_index(name=TOTAL_COUNT_COLUMN)
        )

    if weights is not None:
        summary = summary.merge(weights, on=grouping, how="left")
    else:
        summary[TOTAL_MT_COLUMN] = 0.0
    if counts is not None:
        summary = summary.merge(counts, on=grouping, how="left")
    else:
        summary[TOTAL_COUNT_COLUMN] = 0

    summary[TOTAL_MT_COLUMN] = summary[TOTAL_MT_COLUMN].fillna(0.0)
    summary[TOTAL_COUNT_COLUMN] = summary[TOTAL_COUNT_COLUMN].fillna(0).astype(int)

    for column in [AVG_PRODUCTIVITY_COLUMN, *week_labels, TOTAL_MT_COLUMN]:
        summary[column] = summary[column].fillna(0.0).round(2)

    summary = summary.rename(columns=column_labels)
    summary = summary.sort_values(labeled_columns).reset_index(drop=True)
    return summary.reindex(columns=ordered_columns)


def _suppress_repeated_pch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "PCH" not in df.columns:
        return df
    rows: list[pd.Series] = []
    current = None
    for _, row in df.iterrows():
        row_copy = row.copy()
        if current == row_copy["PCH"]:
            row_copy["PCH"] = ""
        else:
            current = row_copy["PCH"]
        rows.append(row_copy)
    return pd.DataFrame(rows, columns=df.columns)


def export_erection_productivity_summary(
    output_path: str | Path,
    *,
    data_store: AppDataStore | None = None,
    daily_df: pd.DataFrame | None = None,
    project_info: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | str | None = None,
    sheet_name: str = _DEFAULT_SHEET_NAME,
    blank_rows_between_tables: int = 3,
) -> Path:
    """
    Uses existing dashboard data to build and save the PCH and project-wise erection
    productivity summary Excel. Returns the path to the saved file.
    """

    source_daily = daily_df
    source_project_info = project_info
    candidate_date = pd.Timestamp(as_of_date) if as_of_date is not None else None

    if data_store is not None:
        if source_daily is None:
            source_daily = data_store.get_daily()
        if source_project_info is None:
            source_project_info = data_store.get_project_info()
        if candidate_date is None or pd.isna(candidate_date):
            candidate_date = data_store.metadata.last_data_date

    if source_daily is None or source_daily.empty:
        raise ValueError("No erection daily dataframe is available to export.")

    if candidate_date is None or pd.isna(candidate_date):
        date_series = pd.to_datetime(source_daily.get("date"), errors="coerce").dropna()
        if date_series.empty:
            raise ValueError("Unable to determine the current month for the erection summary export.")
        candidate_date = date_series.max()
    active_date = pd.Timestamp(candidate_date).normalize()
    month_start = active_date.to_period("M").to_timestamp()
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()

    week_labels = _generate_week_labels(month_start, month_end)
    offset_days = max(0, (active_date - month_start).days)
    week_count = len(week_labels) or 1
    current_week_idx = min(week_count, max(1, (offset_days // 7) + 1))
    current_week_label = week_labels[current_week_idx - 1]

    project_info_frame = source_project_info.copy() if isinstance(source_project_info, pd.DataFrame) else pd.DataFrame()

    scope = _prepare_month_scope(source_daily, project_info_frame, month_start, month_end, week_labels)
    completions = _prepare_completion_scope(scope, month_start, month_end, week_labels)

    pch_summary, project_summary = _build_productivity_tables(
        scope,
        completions,
        week_labels=week_labels,
    )

    sheet_label = _sanitize_sheet_name(sheet_name or _DEFAULT_SHEET_NAME) or _DEFAULT_SHEET_NAME
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Exporting erection productivity summary for %s (current week: %s) to %s",
        active_date.strftime("%Y-%m"),
        current_week_label,
        target_path,
    )

    gap_rows = max(2, int(blank_rows_between_tables))
    first_table_height = len(pch_summary) + 1  # header + rows
    second_table_start = first_table_height + gap_rows

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        pch_summary.to_excel(writer, sheet_name=sheet_label, index=False, startrow=0)
        project_summary.to_excel(writer, sheet_name=sheet_label, index=False, startrow=second_table_start)

    return target_path


def export_voltage_tower_productivity_summary(
    output_path: str | Path,
    *,
    data_store: AppDataStore | None = None,
    daily_df: pd.DataFrame | None = None,
    project_info: pd.DataFrame | None = None,
    project_voltage_df: pd.DataFrame | None = None,
    project_voltage_path: Path | None = None,
    as_of_date: pd.Timestamp | str | None = None,
    sheet_name: str = _DEFAULT_KV_SHEET_NAME,
    blank_rows_between_tables: int = 3,
) -> Path:
    """Export productivity grouped by KV level and tower type."""

    source_daily = daily_df
    source_project_info = project_info
    candidate_date = pd.Timestamp(as_of_date) if as_of_date is not None else None
    voltage_frame = project_voltage_df

    if project_voltage_df is None and project_voltage_path is not None:
        try:
            voltage_frame = pd.read_excel(project_voltage_path)
        except FileNotFoundError:
            LOGGER.warning("Projects_KV workbook not found at %s; voltage lookup disabled.", project_voltage_path)
            voltage_frame = pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Unable to read Projects_KV workbook '%s': %s", project_voltage_path, exc)
            voltage_frame = pd.DataFrame()

    if data_store is not None:
        if source_daily is None:
            source_daily = data_store.get_daily()
        if source_project_info is None:
            source_project_info = data_store.get_project_info()
        if candidate_date is None or pd.isna(candidate_date):
            candidate_date = data_store.metadata.last_data_date

    if source_daily is None or source_daily.empty:
        raise ValueError("No erection daily dataframe is available to export.")

    if candidate_date is None or pd.isna(candidate_date):
        date_series = pd.to_datetime(source_daily.get("date"), errors="coerce").dropna()
        if date_series.empty:
            raise ValueError("Unable to determine the current month for the KV productivity export.")
        candidate_date = date_series.max()

    active_date = pd.Timestamp(candidate_date).normalize()
    month_start = active_date.to_period("M").to_timestamp()
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()

    week_labels = _generate_week_labels(month_start, month_end)

    project_info_frame = source_project_info.copy() if isinstance(source_project_info, pd.DataFrame) else pd.DataFrame()
    voltage_lookup = _prepare_project_voltage_lookup(voltage_frame)

    scope = _prepare_month_scope(source_daily, project_info_frame, month_start, month_end, week_labels)
    if scope.empty:
        raise ValueError("No erection rows found for the requested month.")
    scope = _annotate_scope_with_voltage(scope, voltage_lookup)

    completions = _prepare_completion_scope(scope, month_start, month_end, week_labels)
    if isinstance(completions, pd.DataFrame) and not completions.empty:
        tower_weights = pd.to_numeric(completions.get("tower_weight"), errors="coerce")
        completions = completions.assign(_tower_weight=tower_weights.fillna(0.0))
    else:
        completions = pd.DataFrame(columns=list(scope.columns) + ["_tower_weight"])

    voltage_table = _build_group_productivity_table(
        scope,
        completions,
        week_labels,
        group_columns=["voltage_label"],
        column_labels={"voltage_label": "Voltage"},
    )
    tower_table = _build_group_productivity_table(
        scope,
        completions,
        week_labels,
        group_columns=["voltage_label", "tower_type"],
        column_labels={"voltage_label": "Voltage", "tower_type": "Tower Type"},
    )
    family_table = _build_group_productivity_table(
        scope,
        completions,
        week_labels,
        group_columns=["voltage_label", "tower_family"],
        column_labels={"voltage_label": "Voltage", "tower_family": "Tower Family"},
    )

    sheet_label = _sanitize_sheet_name(sheet_name or _DEFAULT_KV_SHEET_NAME) or _DEFAULT_KV_SHEET_NAME
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Exporting KV productivity summary for %s to %s",
        active_date.strftime("%Y-%m"),
        target_path,
    )

    gap_rows = max(2, int(blank_rows_between_tables))
    first_table_height = len(voltage_table) + 1
    second_table_start = first_table_height + gap_rows
    third_table_start = second_table_start + len(tower_table) + 1 + gap_rows

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        voltage_table.to_excel(writer, sheet_name=sheet_label, index=False, startrow=0)
        tower_table.to_excel(writer, sheet_name=sheet_label, index=False, startrow=second_table_start)
        family_table.to_excel(writer, sheet_name=sheet_label, index=False, startrow=third_table_start)

    return target_path


