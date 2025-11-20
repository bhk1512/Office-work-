"""Dash callbacks for the productivity dashboard."""

from __future__ import annotations

import hashlib
import logging
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import dash_bootstrap_components as dbc 
import pandas as pd
from io import BytesIO
import traceback
from typing import Any, Callable, Mapping, Sequence, TypeVar

import dash
import duckdb
import re
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.dependencies import MATCH, ALL
from datetime import datetime
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes
from uuid import uuid4
from threading import RLock

try:
    from dash import ctx as dash_ctx
except ImportError:  # Dash < 2.6
    dash_ctx = None

from .charts import (
    # create_monthly_line_chart,
    create_project_lines_chart,
    create_top_bottom_gangs_charts,
    build_responsibilities_chart,
    build_empty_responsibilities_figure,
)
from .config import AppConfig
from .data_loader import (
    load_stringing_compiled_raw as _load_stringing_compiled_raw,
    _resolve_stringing_microplan_root,
)
from .filters import apply_filters, resolve_months
from .metrics import (
    calc_idle_and_loss,
    calc_idle_and_loss_for_column,
    compute_idle_intervals_per_gang,
    compute_gang_baseline_maps,
    compute_project_baseline_maps,
    compute_project_baseline_maps_for,
)
from .workbook import make_trace_workbook_bytes
from .callback_utils import DataSelector, ResponsibilitiesAccessor, ResponsibilitiesPayload
from .plan_utils import (
    normalize_col_key as _normalize_col_key,
    normalize_location as _normalize_location,
    normalize_lower as _normalize_lower,
    normalize_text as _normalize_text,
    compact_project_key as _compact_project_key,
    infer_project_hint as _infer_project_hint,
    prepare_stringing_plan_frame as _prepare_stringing_plan_frame,
)
from .stringing import expand_stringing_to_daily_payout


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)
LOGGER.info("dashboard.callbacks module loaded")

BENCHMARK_MT_PER_DAY = 9.0
_STRINGING_KV_RANGE = {"400", "765"}
_STRINGING_FS_DATE_COLUMNS = ("final_sag_complete", "fs_complete_date", "fs_completed_date", "fs_completion_date")
_STRINGING_PO_DATE_COLUMNS = ("paying_out_complete", "po_completion_date", "po_completion")
BENCHMARK_KM_PER_MONTH = 5.0
_PROJECT_CODE_PATTERN = re.compile(r"(?i)\b([A-Z]{2,4})\s*[-_/ ]*(\d{2,5})\b")

# App-wide config instance for callback logic
config = AppConfig()

T = TypeVar("T")

_SCOPE_CACHE_TTL_SECONDS = 180.0
_SCOPE_CACHE_MAX_ITEMS = 8


@dataclass(slots=True)
class _ScopeCacheEntry:
    frame: pd.DataFrame
    stored_at: float
    aggregates: dict[str, Any] = field(default_factory=dict)


_SCOPE_CACHE: "OrderedDict[str, _ScopeCacheEntry]" = OrderedDict()


@dataclass(slots=True)
class _AggregateCacheEntry:
    value: Any
    stored_at: float


_AGGREGATE_CACHE_TTL_SECONDS = 180.0
_AGGREGATE_CACHE_MAX_ITEMS = 32
_AGGREGATE_CACHE: "OrderedDict[str, _AggregateCacheEntry]" = OrderedDict()
DATA_SELECTOR: DataSelector | None = None
_PROJECT_INFO_PROVIDER: Callable[[], pd.DataFrame] | None = None

_STRINGING_PLAN_CACHE_TTL_SECONDS = 600.0
_STRINGING_PLAN_CACHE: dict[str, Any] = {
    "frame": None,
    "completion": set(),
    "issues": [],
    "index": [],
    "stored_at": 0.0,
    "last_written": 0.0,
}
STRINGING_PLAN_ACCESSOR: ResponsibilitiesAccessor | None = None


def _extract_project_code(value: object) -> str:
    """
    Return the canonical project code from a label or identifier.

    The code is derived from the portion before a "CODE : Name" separator and
    then normalized via a regex that captures the alphanumeric code pattern.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if " : " in text:
        text = text.split(" : ", 1)[0].strip()
    match = _PROJECT_CODE_PATTERN.search(text) if _PROJECT_CODE_PATTERN else None
    if match:
        prefix, digits = match.groups()
        return f"{prefix.upper()} {digits}"
    return text


def _normalize_month_value(raw: Any) -> tuple[str | None, str | None]:
    """
    Normalize a month selector value (e.g., '2025-10', '2025-10-01') into a
    canonical YYYY-MM string plus a display label 'Oct 2025'.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None, None
    if isinstance(raw, (list, tuple)) and raw:
        # take the first element if a sequence sneaks in
        raw = raw[0]
    text = str(raw).strip()
    if not text:
        return None, None
    try:
        ts = pd.to_datetime(text, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(f"{text}-01", errors="coerce")
        if pd.isna(ts):
            return None, None
        ts = ts.to_period("M").to_timestamp()
        return ts.strftime("%Y-%m"), ts.strftime("%b %Y")
    except Exception:
        return None, None


def _resolve_triggered_id() -> Any:
    """
    Return the ID (string or dict) of the triggering input for the current callback,
    compatible with both legacy dash.callback_context and newer dash.ctx APIs.
    """
    if dash_ctx is not None:
        trig = getattr(dash_ctx, "triggered_id", None)
        if trig is not None:
            return trig
    ctx = dash.callback_context
    triggered = getattr(ctx, "triggered", None)
    if not triggered:
        return None
    raw = triggered[0]["prop_id"].split(".")[0]
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _filter_completion_rows(frame: pd.DataFrame, *, completion_column: str) -> pd.DataFrame:
    """Return only the rows whose normalized date matches the completion column."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    if completion_column not in frame.columns or "date" not in frame.columns:
        return pd.DataFrame()
    try:
        completion_norm = pd.to_datetime(frame[completion_column], errors="coerce").dt.normalize()
        date_norm = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    except Exception:
        return pd.DataFrame()
    mask = completion_norm.notna() & date_norm.notna() & (completion_norm == date_norm)
    if not mask.any():
        return pd.DataFrame()
    return frame.loc[mask].copy()


def _sum_completion_totals(
    frame: pd.DataFrame,
    *,
    value_column: str,
    completion_column: str,
    fallback_columns: Sequence[tuple[str, float]] | None = None,
) -> float | None:
    """Sum *value_column* for rows attributed to their completion dates."""
    completion_rows = _filter_completion_rows(frame, completion_column=completion_column)
    if completion_rows.empty:
        return None

    def _coerce_sum(column: str, scale: float = 1.0) -> float | None:
        if column not in completion_rows.columns:
            return None
        series = pd.to_numeric(completion_rows[column], errors="coerce").dropna()
        if series.empty:
            return None
        total_value = float(series.sum()) * scale
        return total_value if not pd.isna(total_value) else None

    total = _coerce_sum(value_column, 1.0)
    if total is not None:
        return total

    for column, scale in fallback_columns or ():
        total = _coerce_sum(column, scale)
        if total is not None:
            return total
    return None


_ERECTIONS_EXPORT_COLUMNS = [
    "completion_date",
    "project_name",
    "location_no",
    "tower_weight_mt",
    "daily_prod_mt",
    "gang_name",
    "start_date",
    "supervisor_name",
    "section_incharge_name",
    "revenue_value",
]


def _parse_completion_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _default_completion_date() -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.Timedelta(days=1)


def _project_filter_candidates(
    project_label: str | None,
    project_code: str | None = None,
) -> list[str]:
    """
    Build a resilient set of candidate project identifiers by combining the
    displayed label (which may include "CODE : Name" formatting) and the canonical
    project code. This helps modal scope filters match the underlying dataset
    regardless of how the project tile renders its title.
    """
    candidates: list[str] = []

    def _add(text: str | None) -> None:
        if not text:
            return
        normalized = text.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    sources = [project_code, project_label]
    for source in sources:
        code_value = _extract_project_code(source)
        if not code_value:
            continue
        _add(code_value)
        flattened = re.sub(r"[^A-Za-z0-9]", "", code_value)
        _add(flattened)
        _add(flattened.upper())
        _add(flattened.lower())

    return candidates


def _format_decimal(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")

def _infer_kv_from_text(name: object) -> str | None:
    """Return '765' or '400' if found in a project name, else None."""
    s = "" if name is None else str(name).lower()
    if "765" in s:
        return "765"
    if "400" in s:
        return "400"
    return None


# --- helper: average days across selected months (fallback 30) ---
def _avg_days_in_selected_months(months_ts) -> float:
    import pandas as pd
    days_factor = 30.0
    try:
        if months_ts:
            month_days = []
            for m in months_ts:
                try:
                    p = m if isinstance(m, pd.Period) else pd.Period(m, freq="M")
                    month_days.append(int(p.days_in_month))
                except Exception:
                    continue
            if month_days:
                days_factor = float(sum(month_days) / len(month_days))
    except Exception:
        pass
    return days_factor


def _prepare_erections_completed(
    scoped: pd.DataFrame,
    *,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    search_text: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scoped.empty or "completion_date" not in scoped.columns:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    working = scoped.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working["completion_date"] = pd.to_datetime(
        working["completion_date"], errors="coerce"
    ).dt.normalize()
    working = working[working["completion_date"].notna()]
    working = working[working["date"] == working["completion_date"]]
    working = working[
        (working["completion_date"] >= range_start)
        & (working["completion_date"] <= range_end)
    ]
    if working.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    working = working.drop_duplicates(
        subset=["project_name", "location_no", "completion_date", "gang_name"]
    ).copy()

    working["location_no_display"] = (
        working["location_no"].map(_normalize_location)
        if "location_no" in working.columns
        else ""
    )
    working["location_no_norm"] = working["location_no_display"]

    working["project_name_display"] = working["project_name"].astype(str).str.strip()
    working["project_name_norm"] = working["project_name_display"].map(_normalize_lower)
    working["gang_name_display"] = working["gang_name"].astype(str).str.strip()

    working["tower_weight_value"] = (
        pd.to_numeric(working["tower_weight"], errors="coerce")
        if "tower_weight" in working.columns
        else pd.Series(np.nan, index=working.index)
    )
    working["daily_prod_value"] = pd.to_numeric(working["daily_prod_mt"], errors="coerce")
    working["start_date_value"] = (
        pd.to_datetime(working["start_date"], errors="coerce").dt.normalize()
        if "start_date" in working.columns
        else pd.Series(pd.NaT, index=working.index)
    )

    supervisor_map: dict[tuple[str, str], str] = {}
    section_map: dict[tuple[str, str], str] = {}
    revenue_map: dict[tuple[str, str], float] = {}

    if responsibilities_provider is not None:
        try:
            resp_source = responsibilities_provider()
        except Exception as exc:
            LOGGER.warning(
                "Erections card: unable to access responsibilities data: %s",
                exc,
            )
            resp_source = pd.DataFrame()
        if resp_source is not None and not resp_source.empty:
            resp = resp_source.copy()
            resp["project_name_norm"] = (
                resp["project_name"].map(_normalize_lower)
                if "project_name" in resp.columns
                else ""
            )
            resp["location_no_norm"] = (
                resp["location_no"].map(_normalize_location)
                if "location_no" in resp.columns
                else ""
            )
            if "entity_name" not in resp.columns:
                resp["entity_name"] = ""
            entity_type_series = (
                resp["entity_type"]
                if "entity_type" in resp.columns
                else pd.Series(["" for _ in range(len(resp))], index=resp.index)
            )
            type_map = {
                "supervisor": "supervisor",
                "supervisors": "supervisor",
                "section incharge": "section incharge",
                "section-incharge": "section incharge",
                "section in-charge": "section incharge",
                "section inch": "section incharge",
            }
            resp["entity_type_norm"] = entity_type_series.map(
                lambda val: type_map.get(_normalize_lower(val), _normalize_lower(val))
            )
            resp["entity_name_norm"] = resp["entity_name"].map(_normalize_text)

            planned_series = (
                pd.to_numeric(resp["revenue_planned"], errors="coerce")
                if "revenue_planned" in resp.columns
                else pd.Series(np.nan, index=resp.index)
            )
            realised_series = (
                pd.to_numeric(resp["revenue_realised"], errors="coerce")
                if "revenue_realised" in resp.columns
                else pd.Series(np.nan, index=resp.index)
            )
            resp["revenue_value"] = realised_series.where(realised_series > 0).fillna(
                planned_series
            )

            def _collapse(series: pd.Series) -> str:
                names = [name for name in series if name]
                return ", ".join(dict.fromkeys(names))

            supervisor_series = (
                resp[resp["entity_type_norm"] == "supervisor"]
                .groupby(["project_name_norm", "location_no_norm"])["entity_name_norm"]
                .apply(_collapse)
            )
            supervisor_map = {
                key: value for key, value in supervisor_series.items() if value
            }

            section_series = (
                resp[resp["entity_type_norm"] == "section incharge"]
                .groupby(["project_name_norm", "location_no_norm"])["entity_name_norm"]
                .apply(_collapse)
            )
            section_map = {
                key: value for key, value in section_series.items() if value
            }

            revenue_series = (
                resp.groupby(["project_name_norm", "location_no_norm"])["revenue_value"].max()
            )
            revenue_map = {
                key: value for key, value in revenue_series.items() if pd.notna(value)
            }

    working["supervisor_name"] = [
        supervisor_map.get((proj, loc), "")
        for proj, loc in zip(
            working["project_name_norm"], working["location_no_norm"]
        )
    ]
    working["section_incharge_name"] = [
        section_map.get((proj, loc), "")
        for proj, loc in zip(
            working["project_name_norm"], working["location_no_norm"]
        )
    ]
    working["revenue_value"] = [
        revenue_map.get((proj, loc), np.nan)
        for proj, loc in zip(
            working["project_name_norm"], working["location_no_norm"]
        )
    ]

    export_df = pd.DataFrame(
        {
            "completion_date": working["completion_date"],
            "project_name": working["project_name_display"],
            "location_no": working["location_no_display"],
            "tower_weight_mt": working["tower_weight_value"],
            "daily_prod_mt": working["daily_prod_value"],
            "gang_name": working["gang_name_display"],
            "start_date": working["start_date_value"],
            "supervisor_name": working["supervisor_name"].fillna(""),
            "section_incharge_name": working["section_incharge_name"].fillna(""),
            "revenue_value": working["revenue_value"],
        }
    )

    if search_text:
        needle = search_text.strip().lower()
        if needle:
            mask = (
                export_df["project_name"].astype(str).str.lower().str.contains(needle, na=False)
                | export_df["location_no"].astype(str).str.lower().str.contains(needle, na=False)
                | export_df["gang_name"].astype(str).str.lower().str.contains(needle, na=False)
            )
            export_df = export_df[mask]

    if export_df.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    export_df = export_df.sort_values(
        ["completion_date", "project_name", "location_no"]
    ).reset_index(drop=True)

    display_df = pd.DataFrame(
        {
            "completion_date": export_df["completion_date"].dt.strftime("%d-%m-%Y").fillna(""),
            "project_name": export_df["project_name"],
            "location_no": export_df["location_no"],
            "tower_weight": export_df["tower_weight_mt"].map(_format_decimal),
            "daily_prod_mt": export_df["daily_prod_mt"].map(_format_decimal),
            "gang_name": export_df["gang_name"],
            "start_date": export_df["start_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else ""),
            "supervisor_name": export_df["supervisor_name"].fillna(""),
            "section_incharge_name": export_df["section_incharge_name"].fillna(""),
            "revenue": export_df["revenue_value"].map(_format_decimal),
        }
    )

    return export_df, display_df

# --- NEW ---
def _prepare_stringing_completed(
    scoped: pd.DataFrame,
    *,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    search_text: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build export/display data for the 'completed' table in STRINGING mode.
    Uses daily_km as the numeric measure (KM/day).
    """
    if scoped.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    working = scoped.copy()

    # date range gate
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"])
    in_range = (working["date"] >= range_start) & (working["date"] <= range_end)
    working = working.loc[in_range].copy()
    if working.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    # normalize display fields
    working["project_name_display"] = working.get("project_name", "").astype(str).str.strip()
    working["gang_name_display"] = working.get("gang_name", "").astype(str).str.strip()
    from_ap = working.get("from_ap", pd.Series([""] * len(working), index=working.index)).astype(str).str.strip()
    to_ap   = working.get("to_ap",   pd.Series([""] * len(working), index=working.index)).astype(str).str.strip()
    working["span_display"] = (from_ap + " \u2192 " + to_ap).str.strip(" \u2192 ")  # From→To

    # search filter (project/gang/from/to)
    if search_text:
        needle = _normalize_lower(search_text)
        mask = (
            working["project_name_display"].map(_normalize_lower).str.contains(needle, na=False)
            | working["gang_name_display"].map(_normalize_lower).str.contains(needle, na=False)
            | from_ap.map(_normalize_lower).str.contains(needle, na=False)
            | to_ap.map(_normalize_lower).str.contains(needle, na=False)
        )
        working = working.loc[mask]

    if working.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    # export frame (reuse erection schema so downstream stays happy)
    supervisor_series = working.get("supervisor_name")
    if supervisor_series is None:
        supervisor_series = pd.Series(["NA"] * len(working), index=working.index)
    else:
        supervisor_series = supervisor_series.astype(str).str.strip().replace("", "NA")
    section_series = working.get("section_incharge_name")
    if section_series is None:
        section_series = pd.Series(["NA"] * len(working), index=working.index)
    else:
        section_series = section_series.astype(str).str.strip().replace("", "NA")

    export_df = pd.DataFrame(
        {
            "completion_date": working["date"],
            "project_name": working["project_name_display"],
            "location_no": working["span_display"],  # show From→To in the 'Location' column
            "tower_weight_mt": pd.to_numeric(working.get("daily_km", np.nan), errors="coerce"),  # will display as KM
            "daily_prod_mt":  pd.to_numeric(working.get("daily_km", np.nan), errors="coerce"),  # KM/day
            "gang_name": working["gang_name_display"],
            "start_date": working["date"],           # fallback (F/S start may not be present in daily)
            "supervisor_name": supervisor_series,
            "section_incharge_name": section_series,
            "revenue_value": pd.Series([np.nan] * len(working), index=working.index),
        }
    ).sort_values(["completion_date", "project_name", "gang_name"])

    # display frame for DataTable
    display_df = pd.DataFrame(
        {
            "completion_date": export_df["completion_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else ""),
            "project_name": export_df["project_name"],
            "location_no": export_df["location_no"].fillna(""),
            "tower_weight": export_df["tower_weight_mt"].map(_format_decimal),  # shows numeric as text
            "daily_prod_mt": export_df["daily_prod_mt"].map(_format_decimal),
            "gang_name": export_df["gang_name"],
            "start_date": export_df["start_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else "NA"),
            "supervisor_name": export_df["supervisor_name"],
            "section_incharge_name": export_df["section_incharge_name"],
            "revenue": export_df["revenue_value"].map(lambda v: "-" if pd.isna(v) else _format_decimal(v)),
        }
    )

    return export_df, display_df



_slug = lambda s: re.sub(r"[^a-z0-9_-]+", "-", str(s).lower()).strip("-")

def _render_avp_row(
    gang,
    delivered,
    lost,
    total,
    pct,
    avg_prod=0.0,
    baseline=0.0,
    last_project=" ",
    last_date=" ",
    rate_label="MT/day",
    unit_total="MT",
    namespace: str = "avp",
):
    badge_cls = "good" if pct >= 80 else ("mid" if pct >= 65 else "low")
    delivered_pct = 0 if total == 0 else max(0, min(100, (delivered/total)*100))
    lost_pct = 0 if total == 0 else max(0, min(100, (lost/total)*100))

    safe_gang = _slug(gang)
    row_tip_id = f"{namespace}-tip-{safe_gang}"  # STRING id for tooltip target
    row_type = f"{namespace}-row"
    tip_type = f"{namespace}-tip"

    return html.Div(
        id={"type": row_type, "index": gang},   # <-- move pattern id to the row itself
        n_clicks=0,
        style={"cursor": "pointer"},             # (nice to have)
        className="avp-item",
        children=[
            html.Div(
                className="avp-head",
                children=[html.Div(gang, className="avp-name"),
                        html.Div(f"{int(round(pct))}%", className=f"avp-pct {badge_cls}")],
            ),
           
            html.Div(className="avp-track", children=[
                html.Div(className="avp-delivered", style={"width": f"{delivered_pct}%"}),
                html.Div(className="avp-lost", style={"left": f"{delivered_pct}%", "width": f"{lost_pct}%"}),
            ]),
            html.Div(className="avp-meta", children=[
                html.Span(f"{delivered:,.0f} {unit_total} vs {lost:,.0f} {unit_total} lost"),
                html.Div(f"{total:,.0f} {unit_total}", className="text-muted ..."),
            ]),


            html.Div(
                id={"type": tip_type, "index": gang},     # pattern id to allow row-wide click capture
                n_clicks=0,
                className="avp-tip-overlay",
                children=[
                    # string id to attach dbc.Tooltip (fills the overlay)
                    html.Span(id=row_tip_id, className="avp-tip-fill")
                ],
            ),

            dbc.Tooltip(
                [
                    html.Div(html.B(gang)),
                    html.Div(f"Project: {last_project}"),
                    html.Div(f"Last worked at: {last_date}"),
                    html.Div(f"Current {rate_label}: {avg_prod:.2f}"),
                    html.Div(f"Baseline {rate_label}: {baseline:.2f}"),
                ],
                target=row_tip_id,
                placement="right",
                delay={"show": 100, "hide": 100},
            ),
        ],
    )

def _ensure_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_mode(value: str | None) -> str:
    return (value or "erection").strip().lower()


def _normalize_str_list(values: Sequence[Any] | None, *, lower: bool = False) -> list[str]:
    result: list[str] = []
    if not values:
        return result
    for raw in values:
        text = "" if raw is None else str(raw).strip()
        if not text:
            continue
        result.append(text.lower() if lower else text)
    return result


def _completion_range_from_month_selection(
    months_value: Sequence[str] | None,
    quick_range: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Deduce the date window represented by the current month filters so modal
    completion pickers can mirror the home screen selection.
    """

    months_list = _normalize_str_list(_ensure_list(months_value))
    try:
        resolved_months = resolve_months(months_list, quick_range)
    except Exception:
        resolved_months = []

    periods: list[pd.Period] = []
    for raw in resolved_months:
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            continue
        periods.append(ts.to_period("M"))

    if not periods:
        default = _default_completion_date()
        return default, default

    first_period = min(periods)
    last_period = max(periods)
    start = first_period.to_timestamp(how="start").normalize()
    end = last_period.to_timestamp(how="end").normalize()
    return start, end


def _set_scope_cache_entry(key: str, frame: pd.DataFrame) -> None:
    if not key:
        return
    _SCOPE_CACHE[key] = _ScopeCacheEntry(frame.copy(deep=False), time.time())
    _SCOPE_CACHE.move_to_end(key)
    while len(_SCOPE_CACHE) > _SCOPE_CACHE_MAX_ITEMS:
        _SCOPE_CACHE.popitem(last=False)


def _remember_scope_frame(frame: pd.DataFrame) -> str:
    key = uuid4().hex
    _set_scope_cache_entry(key, frame)
    return key


def _get_scope_entry(key: str | None) -> _ScopeCacheEntry | None:
    if not key:
        return None
    entry = _SCOPE_CACHE.get(key)
    if not entry:
        return None
    if _SCOPE_CACHE_TTL_SECONDS > 0 and (time.time() - entry.stored_at) > _SCOPE_CACHE_TTL_SECONDS:
        _SCOPE_CACHE.pop(key, None)
        return None
    return entry


def _get_scope_cache_entry(key: str | None) -> pd.DataFrame | None:
    entry = _get_scope_entry(key)
    if entry is None:
        return None
    return entry.frame.copy(deep=False)


def _cached_scope_result(
    cache_key: str | None,
    token: str,
    producer: Callable[[], T],
    *,
    clone: Callable[[T], T] | None = None,
) -> T:
    entry = _get_scope_entry(cache_key)
    if entry is not None:
        existing = entry.aggregates.get(token)
        if existing is not None:
            return clone(existing) if clone else existing
    result = producer()
    if entry is not None:
        entry.aggregates[token] = result
    return clone(result) if clone else result


def _cached_global_result(
    token: str,
    producer: Callable[[], T],
    *,
    clone: Callable[[T], T] | None = None,
) -> T:
    entry = _AGGREGATE_CACHE.get(token)
    if entry is not None:
        if _AGGREGATE_CACHE_TTL_SECONDS <= 0 or (time.time() - entry.stored_at) <= _AGGREGATE_CACHE_TTL_SECONDS:
            return clone(entry.value) if clone else entry.value
        _AGGREGATE_CACHE.pop(token, None)
        entry = None
    result = producer()
    _AGGREGATE_CACHE[token] = _AggregateCacheEntry(result, time.time())
    _AGGREGATE_CACHE.move_to_end(token)
    while len(_AGGREGATE_CACHE) > _AGGREGATE_CACHE_MAX_ITEMS:
        _AGGREGATE_CACHE.popitem(last=False)
    return clone(result) if clone else result


def _clone_baseline_result(
    result: tuple[Mapping[str, float], Mapping[str, Mapping[pd.Timestamp, float]]],
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    overall, monthly = result
    overall_map = dict(overall or {})
    monthly_map = {gang: dict(month_map or {}) for gang, month_map in (monthly or {}).items()}
    return overall_map, monthly_map


def _clone_loss_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _clone_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def _hash_cache_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _attach_line_kv(work: pd.DataFrame) -> pd.DataFrame:
    """
    Adds __line_kv__ ('765'|'400'|NA) inferred from Project Details.
    Tries BOTH mappings: project name and project code.
    If mapping fails, falls back to the row's own project text.
    """
    try:
        provider = _PROJECT_INFO_PROVIDER
        if work is None or work.empty:
            return work

        proj_col = "project_name" if "project_name" in work.columns else ("project" if "project" in work.columns else None)
        if not proj_col:
            return work

        out = work.copy()
        out["__kv_source__"] = out[proj_col].astype(str).str.strip()

        dfpi = provider() if callable(provider) else None
        if dfpi is not None and not dfpi.empty:
            dpi = dfpi.copy()

            def _norm_key(x: object) -> str:
                return re.sub(r"\s+", " ", ("" if x is None else str(x)).strip().lower())

            # Build normalized keys on both name and code
            if "project_name" in dpi.columns:
                dpi["__name_key__"] = dpi["project_name"].astype(str).map(_norm_key)
            else:
                dpi["__name_key__"] = ""

            if "project_code" in dpi.columns:
                dpi["__code_key__"] = dpi["project_code"].astype(str).map(_norm_key)
            else:
                dpi["__code_key__"] = ""

            # Which column is the descriptive text? Prefer "Project Name"
            desc_col = "Project Name" if "Project Name" in dpi.columns else ("project_name" if "project_name" in dpi.columns else None)

            if desc_col:
                name_map = dict(zip(dpi["__name_key__"], dpi[desc_col].astype(str)))
                code_map = dict(zip(dpi["__code_key__"], dpi[desc_col].astype(str)))

                row_key = out[proj_col].astype(str).map(_norm_key)
                mapped_name = row_key.map(name_map)
                mapped_code = row_key.map(code_map)
                # Prefer name-map, fall back to code-map, then original project text
                desc_series = mapped_name.where(mapped_name.notna(), mapped_code)
                out["__kv_source__"] = desc_series.where(desc_series.notna(), out[proj_col].astype(str))

        src = out["__kv_source__"].astype(str).str.lower()
        out["__line_kv__"] = np.where(
            src.str.contains("765"),
            "765",
            np.where(src.str.contains("400"), "400", pd.NA),
        )
        return out
    except Exception:
        return work


def _stringing_scope(work: pd.DataFrame, kv_values, method_values) -> pd.DataFrame:
    work = work.copy()
    work = _attach_line_kv(work)

    kv_set = set(_normalize_str_list(kv_values))
    if kv_set and kv_set != {"400", "765"}:
        work = work[work["__line_kv__"].isin(kv_set)]

    method_set = {m.lower() for m in _normalize_str_list(method_values)}
    if method_set and method_set != {"manual", "tse"}:
        if "method" in work.columns:
            work = work[work["method"].astype(str).str.lower().isin(method_set)]
        else:
            work = work.iloc[0:0]
    return work


def _build_scope_frames(
    mode_value: str,
    *,
    project_list: Sequence[str],
    gang_list: Sequence[str],
    months_value: Sequence[str],
    quick_range: str | None,
    kv_values: Sequence[str] | None,
    method_values: Sequence[str] | None,
) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp], float]:
    selector = DATA_SELECTOR
    if selector is None:
        raise RuntimeError("Data selector not initialized.")
    eff_mode = _normalize_mode(mode_value)
    normalized_months = resolve_months(months_value, quick_range)
    kv_set = {value.strip() for value in _normalize_str_list(kv_values) if value and str(value).strip()}
    method_set = {value.strip().lower() for value in _normalize_str_list(method_values)}

    scoped_frames = selector.scopes_for(
        eff_mode,
        months=normalized_months,
        projects=project_list,
        gangs=gang_list,
        kv_filter=kv_set if eff_mode == "stringing" else None,
        method_filter=method_set if eff_mode == "stringing" else None,
    )

    if scoped_frames is None:
        df_day = selector.select(eff_mode)
        if not isinstance(df_day, pd.DataFrame) or df_day.empty:
            empty = pd.DataFrame()
            return (
                {"month": empty, "project": empty, "full": empty, "project_gang": empty},
                [],
                30.0,
            )

        work = df_day.copy()
        if eff_mode == "stringing":
            work = _stringing_scope(work, kv_values, method_values)

        month_scope = apply_filters(work, [], normalized_months, [])
        project_scope = apply_filters(work, project_list, normalized_months, [])
        full_scope = apply_filters(work, project_list, normalized_months, gang_list)
        project_gang_scope = apply_filters(work, project_list, [], gang_list)
    else:
        month_scope = scoped_frames.get("month", pd.DataFrame()).copy()
        project_scope = scoped_frames.get("project", pd.DataFrame()).copy()
        full_scope = scoped_frames.get("full", pd.DataFrame()).copy()
        project_gang_scope = scoped_frames.get("project_gang", pd.DataFrame()).copy()

    return (
        {
            "month": month_scope,
            "project": project_scope,
            "full": full_scope,
            "project_gang": project_gang_scope,
        },
        normalized_months,
        _avg_days_in_selected_months(normalized_months),
    )


def _build_scope_meta_payload(
    *,
    eff_mode: str,
    project_list: list[str],
    gang_list: list[str],
    months_list: list[str],
    quick_range: str | None,
    kv_values: Sequence[str] | None,
    method_values: Sequence[str] | None,
    kv_list: list[str],
    method_list: list[str],
) -> dict[str, Any]:
    frames, months_ts, days_factor = _build_scope_frames(
        eff_mode,
        project_list=project_list,
        gang_list=gang_list,
        months_value=months_list,
        quick_range=quick_range,
        kv_values=kv_values,
        method_values=method_values,
    )
    scope_keys = {name: _remember_scope_frame(frame) for name, frame in frames.items()}
    rows_meta = {name: int(len(frame.index)) for name, frame in frames.items()}
    signature_payload = {
        "mode": eff_mode,
        "projects": project_list,
        "gangs": gang_list,
        "months": months_list,
        "quick_range": quick_range,
        "kv": kv_list,
        "methods": method_list,
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()

    return {
        "mode": eff_mode,
        "signature": signature,
        "scopes": scope_keys,
        "rows": rows_meta,
        "days_factor": days_factor,
        "months_iso": [ts.isoformat() for ts in months_ts],
        "selected": {
            "projects": project_list,
            "gangs": gang_list,
            "months": months_list,
            "quick_range": quick_range,
            "kv": kv_list,
            "methods": method_list,
        },
    }


def _repopulate_scopes_from_meta(meta: dict[str, Any]) -> dict[str, pd.DataFrame]:
    selected = meta.get("selected") or {}
    scopes, _, _ = _build_scope_frames(
        _normalize_mode(meta.get("mode")),
        project_list=selected.get("projects", []),
        gang_list=selected.get("gangs", []),
        months_value=selected.get("months", []),
        quick_range=selected.get("quick_range"),
        kv_values=selected.get("kv", []),
        method_values=selected.get("methods", []),
    )
    for name, frame in scopes.items():
        cache_key = (meta.get("scopes") or {}).get(name)
        if cache_key:
            _set_scope_cache_entry(cache_key, frame)
    return scopes


def _scope_frame_from_store(meta: dict[str, Any] | None, scope_name: str) -> pd.DataFrame:
    if not isinstance(meta, dict):
        return pd.DataFrame()
    cache_key = (meta.get("scopes") or {}).get(scope_name)
    frame = _get_scope_cache_entry(cache_key)
    if frame is not None:
        return frame
    rebuilt = _repopulate_scopes_from_meta(meta)
    return rebuilt.get(scope_name, pd.DataFrame())


def _months_from_meta(meta: dict[str, Any] | None) -> list[pd.Timestamp]:
    raw_months = (meta or {}).get("months_iso") or []
    months: list[pd.Timestamp] = []
    for raw in raw_months:
        try:
            months.append(pd.to_datetime(raw))
        except Exception:
            continue
    return months


def _format_period_label(months: Sequence[pd.Timestamp]) -> str:
    if not months:
        return "(All periods)"
    periods = sorted({pd.Period(ts, "M") for ts in months})
    labels = [period.strftime("%b %Y") for period in periods]
    if len(periods) == 1:
        return f"({labels[0]})"
    if all(periods[i] + 1 == periods[i + 1] for i in range(len(periods) - 1)):
        return f"({labels[0]} - {labels[-1]})"
    if len(labels) <= 3:
        return "(" + ", ".join(labels) + ")"
    return "(" + ", ".join(labels[:3]) + ", ...)"


def register_callbacks(
    app: Dash,
    data_provider: Callable[[], pd.DataFrame],
    config: AppConfig,
    *,
    duckdb_connection: duckdb.DuckDBPyConnection | None = None,
    duckdb_lock: RLock | None = None,
    stringing_data_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_compiled_provider: Callable[[], pd.DataFrame] | None = None,
    project_info_provider: Callable[[], pd.DataFrame] | None = None,
    project_baseline_provider: Callable[[], tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]] | None = None,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    responsibilities_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    responsibilities_error_provider: Callable[[], str | None] | None = None,
    stringing_plan_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_plan_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    stringing_plan_error_provider: Callable[[], str | None] | None = None,
) -> None:

    LOGGER.debug("Registering callbacks")

    if config.enable_stringing and stringing_data_provider is None:
        raise RuntimeError("Stringing data provider must be supplied when stringing support is enabled.")

    data_selector = DataSelector(
        config=config,
        data_provider=data_provider,
        stringing_provider=stringing_data_provider,
        duckdb_connection=duckdb_connection,
        duckdb_lock=duckdb_lock,
        logger=LOGGER,
    )
    global DATA_SELECTOR, _PROJECT_INFO_PROVIDER
    DATA_SELECTOR = data_selector
    _PROJECT_INFO_PROVIDER = project_info_provider
    responsibilities_accessor = ResponsibilitiesAccessor(
        data_provider=responsibilities_provider,
        completion_provider=responsibilities_completion_provider,
        error_provider=responsibilities_error_provider,
        logger=LOGGER,
    )
    stringing_plan_accessor = ResponsibilitiesAccessor(
        data_provider=stringing_plan_provider,
        completion_provider=stringing_plan_completion_provider,
        error_provider=stringing_plan_error_provider,
        logger=LOGGER,
    )
    global STRINGING_PLAN_ACCESSOR
    STRINGING_PLAN_ACCESSOR = stringing_plan_accessor
    plan_accessors: dict[str, ResponsibilitiesAccessor] = {
        "erection": responsibilities_accessor,
        "stringing": stringing_plan_accessor,
    }
    has_plan_provider = {
        "erection": callable(responsibilities_provider),
        "stringing": callable(stringing_plan_provider),
    }

    def _resolve_monthly_plan_workbook_path(cfg: AppConfig, plan_mode: str) -> Path | None:
        normalized = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        root = Path(cfg.data_path).expanduser()
        candidates: list[Path] = []

        def _push(path: Path | str | None) -> None:
            if not path:
                return
            candidate = Path(path).expanduser()
            if candidate not in candidates:
                candidates.append(candidate)

        if root.is_file():
            _push(root)
        else:
            if normalized == "erection":
                _push(root / "ErectionCompiled_Output.xlsx")
                _push(root / "MicroPlanCompiled_Output.xlsx")
                _push(root / "ErectionCompiled.xlsx")
            else:
                _push(root / "StringingCompiled_Output.xlsx")
                _push(root / "StringingCompiled.xlsx")
                _push(root / "Stringing Compiled.xlsx")

        repo_root = Path(".").expanduser().resolve()
        if normalized == "erection":
            _push(repo_root / "Parquets" / "Erection" / "ErectionCompiled_Output.xlsx")
        else:
            # Sibling/specified dirs containing compiled stringing excel
            potential_roots: list[Path] = []
            potential_roots.append(root)
            potential_roots.append(root.parent / "Stringing")
            for rel in getattr(cfg, "stringing_parquet_dirs", ()):
                try:
                    rel_path = (root / Path(rel)).resolve()
                except Exception:
                    continue
                potential_roots.append(rel_path)
            potential_roots.append(repo_root / "Parquets" / "Stringing")
            for base in potential_roots:
                if not isinstance(base, Path):
                    continue
                _push(base / "StringingCompiled_Output.xlsx")
                _push(base / "StringingCompiled.xlsx")
                _push(base / "Stringing Compiled.xlsx")

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except PermissionError:
                continue
        return None

    def _fetch_monthly_plan(
        plan_mode: str = "erection",
        *,
        allow_workbook_fallback: bool = False,
    ) -> tuple[pd.DataFrame | None, set[tuple[str, str]], str | None, pd.ExcelFile | None]:
        mode_key = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        accessor = plan_accessors[mode_key]
        payload: ResponsibilitiesPayload = accessor.load()
        if payload.has_frame:
            frame = payload.frame.copy() if payload.frame is not None else None
            completion_keys = set(payload.completion_keys or set())
            return frame, completion_keys, payload.error, None

        completion_keys = set(payload.completion_keys or set())
        if mode_key == "stringing":
            plan_frame, _plan_keys, _plan_issues, _plan_index = _load_stringing_plan_snapshot(config)
            if isinstance(plan_frame, pd.DataFrame) and not plan_frame.empty:
                return plan_frame.copy(), completion_keys, payload.error, None
        load_error = payload.error
        if allow_workbook_fallback and not has_plan_provider.get(mode_key, False):
            cfg = config
            workbook_path = _resolve_monthly_plan_workbook_path(cfg, mode_key)
            plan_title = "Stringing plan" if mode_key == "stringing" else "Micro Plan"
            if workbook_path is None:
                LOGGER.warning(
                    "Monthly plan workbook for '%s' not found near data root '%s'.",
                    mode_key,
                    cfg.data_path,
                )
                return None, completion_keys, f"No {plan_title} data found in the compiled workbook.", None
            try:
                workbook = pd.ExcelFile(workbook_path)
            except FileNotFoundError:
                LOGGER.warning("Monthly plan workbook not found: %s", workbook_path)
                return None, completion_keys, "Compiled workbook not found.", None
            except Exception as exc:
                LOGGER.exception("Failed to open monthly plan workbook '%s': %s", workbook_path, exc)
                return None, completion_keys, "Unable to load monthly plan data.", None

            if mode_key == "stringing":
                preferred_sheet = getattr(cfg, "stringing_sheet_name", "") or "Stringing Compiled"
                sheet_name = next(
                    (
                        name
                        for name in workbook.sheet_names
                        if _normalize_col_key(name) == _normalize_col_key(preferred_sheet)
                    ),
                    None,
                )
                if not sheet_name:
                    LOGGER.warning("Stringing sheet missing in workbook '%s'; sheets=%s", workbook_path, workbook.sheet_names)
                    return (
                        None,
                        completion_keys,
                        "No Stringing plan data found in the compiled workbook.",
                        workbook,
                    )
            else:
                sheet_name = "MicroPlanResponsibilities"
                if sheet_name not in workbook.sheet_names:
                    LOGGER.warning("Sheet '%s' missing in workbook '%s'", sheet_name, workbook_path)
                    return (
                        None,
                        completion_keys,
                        "No Micro Plan data found in the compiled workbook.",
                        workbook,
                    )

            try:
                df_atomic = pd.read_excel(workbook, sheet_name=sheet_name)
            except Exception as exc:
                LOGGER.exception("Failed to load sheet '%s' for %s plan: %s", sheet_name, mode_key, exc)
                message = (
                    "Unable to load Stringing plan data."
                    if mode_key == "stringing"
                    else "Unable to load Micro Plan data."
                )
                return None, completion_keys, message, workbook
            try:
                setattr(workbook, "_plan_sheet_name", sheet_name)
                setattr(workbook, "_plan_workbook_path", Path(workbook_path))
            except Exception:
                pass
            return df_atomic, completion_keys, load_error, workbook

        return None, completion_keys, load_error, None

    def _build_tse_lookup_from_df(df: pd.DataFrame | None) -> tuple[dict[str, int], dict[str, str]]:
        canonical: dict[str, int] = {}
        aliases: dict[str, str] = {}
        if not isinstance(df, pd.DataFrame) or df.empty or "number_of_tse" not in df.columns:
            return canonical, aliases
        project_col = None
        for candidate in ("project_name", "project", "Project Name", "Project"):
            if candidate in df.columns:
                project_col = candidate
                break
        if project_col is None:
            return canonical, aliases
        work = df[[project_col, "number_of_tse"]].copy()
        work[project_col] = work[project_col].astype(str).str.strip()
        work["number_of_tse"] = pd.to_numeric(work["number_of_tse"], errors="coerce")
        work = work.dropna(subset=[project_col, "number_of_tse"])
        if work.empty:
            return canonical, aliases

        def _project_code_token(text: str) -> str | None:
            match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{3,4})\b", str(text).upper())
            if not match:
                return None
            return f"{match.group(1)}{match.group(2)}"

        grouped = work.groupby(work[project_col])["number_of_tse"].max()
        for project, raw_value in grouped.items():
            try:
                value = int(round(float(raw_value)))
            except (TypeError, ValueError):
                continue
            if pd.isna(value):
                continue
            canonical_key = _normalize_lower(project)
            if not canonical_key:
                continue
            if canonical_key not in canonical:
                canonical[canonical_key] = value
            compact_key = _compact_project_key(project)
            if compact_key:
                aliases.setdefault(compact_key, canonical_key)
            code_token = _project_code_token(project)
            if code_token:
                aliases.setdefault(_compact_project_key(code_token), canonical_key)
        return canonical, aliases

    def _get_stringing_tse_lookup() -> tuple[dict[str, int], dict[str, str]]:
        if not config.enable_stringing:
            return {}, {}

        def _producer() -> tuple[dict[str, int], dict[str, str]]:
            df_compiled = pd.DataFrame()
            if callable(stringing_compiled_provider):
                try:
                    df_compiled = stringing_compiled_provider()
                except Exception:
                    df_compiled = pd.DataFrame()
            if not isinstance(df_compiled, pd.DataFrame) or df_compiled.empty:
                try:
                    df_compiled = _load_stringing_compiled_raw(config)
                except Exception:
                    df_compiled = pd.DataFrame()
            return _build_tse_lookup_from_df(df_compiled)

        return _cached_global_result(
            "stringing:tse-lookup",
            _producer,
            clone=lambda payload: (dict(payload[0]), dict(payload[1])),
        )

    def _resolve_tse_value(
        norm_keys: Sequence[str],
        compact_keys: Sequence[str],
        canonical_map: Mapping[str, int],
        alias_map: Mapping[str, str],
    ) -> tuple[int | None, str | None]:
        if not canonical_map and not alias_map:
            return None, None
        for key in norm_keys:
            if key and key in canonical_map:
                return int(canonical_map[key]), key
        for key in compact_keys:
            if not key:
                continue
            canonical = alias_map.get(key)
            if canonical and canonical in canonical_map:
                return int(canonical_map[canonical]), canonical
        return None, None

    def _compute_planned_tower_layers(
        scoped_all: pd.DataFrame,
        months_ts: Sequence[pd.Timestamp],
    ) -> tuple[int, float]:
        active_months = sorted({ts for ts in months_ts if pd.notna(ts)})
        if not active_months or not has_plan_provider.get("erection"):
            return 0, 0.0
        try:
            resp_df, _, _, _ = _fetch_monthly_plan("erection")
            if not isinstance(resp_df, pd.DataFrame) or resp_df.empty:
                return 0, 0.0
            df_mp = resp_df.copy()
            if "plan_month" in df_mp.columns:
                df_mp["plan_month"] = pd.to_datetime(df_mp["plan_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                df_mp["completion_month"] = df_mp["plan_month"]
            elif "completion_date" in df_mp.columns:
                df_mp["completion_month"] = pd.to_datetime(df_mp["completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            else:
                df_mp["completion_month"] = pd.NaT
            for column in ("project_name", "project_key", "location_no"):
                if column not in df_mp.columns:
                    df_mp[column] = ""
                df_mp[column] = df_mp[column].map(_normalize_text)
            df_mp["project_name_lc"] = df_mp["project_name"].map(_normalize_lower)
            df_mp["project_key_lc"] = df_mp["project_key"].map(_normalize_lower)
            df_mp["location_no_norm"] = df_mp["location_no"].map(_normalize_location)
            if "project_name" in scoped_all.columns and not scoped_all.empty:
                sel_projects = set(
                    scoped_all["project_name"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
            else:
                sel_projects = set()
            if sel_projects:
                mask_project = df_mp["project_name_lc"].isin(sel_projects) | df_mp["project_key_lc"].isin(sel_projects)
                df_mp = df_mp.loc[mask_project].copy()
            df_mp = df_mp[df_mp["completion_month"].isin(active_months)].copy()
            if df_mp.empty:
                return 0, 0.0
            dedup_cols = ["project_name_lc", "location_no_norm"]
            valid_locations = df_mp.dropna(subset=dedup_cols).copy()
            if "tower_weight" in valid_locations.columns:
                valid_locations["tower_weight"] = pd.to_numeric(valid_locations["tower_weight"], errors="coerce").fillna(0.0)
            dedup_locations = (
                valid_locations.sort_values(dedup_cols)
                .drop_duplicates(subset=dedup_cols, keep="first")
            )
            planned_count = int(dedup_locations.shape[0])
            planned_mt = (
                float(dedup_locations.get("tower_weight", 0.0).sum())
                if "tower_weight" in dedup_locations.columns
                else 0.0
            )
            return planned_count, planned_mt
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to compute planned tower layers: %s", exc)
            return 0, 0.0

    def _derive_completion_window(
        loss_scope: pd.DataFrame,
        months_ts: Sequence[pd.Timestamp],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        today = pd.Timestamp.today().normalize()
        default_start = today.to_period("M").start_time.normalize()
        default_end = (today + pd.offsets.MonthEnd(0)).normalize()
        try:
            if months_ts:
                start = pd.Timestamp(min(months_ts)).normalize()
                end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
                return start, end
            if isinstance(loss_scope, pd.DataFrame) and not loss_scope.empty:
                comp_series = pd.to_datetime(loss_scope.get("completion_date"), errors="coerce").dropna()
                if len(comp_series):
                    return pd.Timestamp(comp_series.min()).normalize(), pd.Timestamp(comp_series.max()).normalize()
                date_series = pd.to_datetime(loss_scope.get("date"), errors="coerce").dropna()
                if len(date_series):
                    return pd.Timestamp(date_series.min()).normalize(), pd.Timestamp(date_series.max()).normalize()
        except Exception:
            pass
        return default_start, default_end

    def _count_completed_towers(
        loss_scope: pd.DataFrame,
        months_ts: Sequence[pd.Timestamp],
    ) -> int:
        if not isinstance(loss_scope, pd.DataFrame) or loss_scope.empty:
            return 0
        try:
            range_start, range_end = _derive_completion_window(loss_scope, months_ts)
            export_df, _ = _prepare_erections_completed(
                loss_scope,
                range_start=range_start,
                range_end=range_end,
                responsibilities_provider=None,
                search_text=None,
            )
            return int(len(export_df)) if isinstance(export_df, pd.DataFrame) else 0
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to count completed towers: %s", exc)
            return 0

    def _prepare_stringing_plan_frame(
        df_raw: pd.DataFrame,
        *,
        project_hint: str | None = None,
        source_path: Path | str | None = None,
        sheet_name: str | None = None,
    ) -> tuple[pd.DataFrame, set[tuple[str, str]], list[dict[str, str]]]:
        """
        Normalize the stringing monthly plan sheet into the generic responsibilities structure.
        Returns (frame, completion_keys) where completion keys capture completed spans.
        """

        if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            columns = [
                "project_name",
                "project_key",
                "entity_type",
                "entity_name",
                "location_no",
                "tower_weight",
                "revenue_planned",
                "revenue_realised",
                "stringing_span_completed",
                "span_from",
                "span_to",
                "method",
                "gang_strength",
                "paying_out_start",
                "final_sag_complete",
            ]
            return pd.DataFrame(columns=columns), set(), []

        col_lookup = {_normalize_col_key(col): col for col in df_raw.columns}
        column_aliases: dict[str, tuple[str, ...]] = {
            "serial": ("S. No.", "S no", "s no", "serial", "serial no", "jmc no", "span no"),
            "span_from": ("From AP", "from_ap", "from ap", "start tower", "from tower"),
            "span_to": ("To AP", "to_ap", "to ap", "end tower", "to tower"),
            "span_length": ("Span (m)", "span m", "span length", "length_m", "length (m)", "length"),
            "method": ("Method", "method"),
            "gang_strength": ("Gang Strength", "gang_strength"),
            "paying_out_start": ("Paying Out Start", "po_start_date", "p/o start", "po start", "paying_out_start"),
            "paying_out_complete": ("Paying Out Completed", "po_completion_date", "p/o completed", "po completion"),
            "final_sag_complete": ("Final Sag Complete", "fs_complete_date", "final sag", "fs complete date"),
            "gang_name": ("Gang Name", "gang_name"),
            "supervisor": ("Supervisor", "supervisor"),
            "section_incharge": ("Section Incharge", "section_incharge", "section incharge"),
        }

        def _resolve_series(key: str, default: Any = "") -> tuple[pd.Series, bool]:
            options = column_aliases.get(key, ())
            for candidate in options:
                norm = _normalize_col_key(candidate)
                if norm in col_lookup:
                    return df_raw[col_lookup[norm]], True
            norm = _normalize_col_key(key)
            if norm in col_lookup:
                return df_raw[col_lookup[norm]], True
            return pd.Series([default] * len(df_raw), index=df_raw.index), False

        def _optional_series(candidates: Sequence[str], default: Any = "") -> pd.Series:
            for candidate in candidates:
                key = _normalize_col_key(candidate)
                if key in col_lookup:
                    return df_raw[col_lookup[key]]
            return pd.Series([default] * len(df_raw), index=df_raw.index)

        # Log only if none of the alias options exist
        required_for_logging = {
            "S. No.": column_aliases["serial"],
            "From AP": column_aliases["span_from"],
            "To AP": column_aliases["span_to"],
            "Span (m)": column_aliases["span_length"],
            "Method": column_aliases["method"],
            "Gang Name": column_aliases["gang_name"],
            "Supervisor": column_aliases["supervisor"],
            "Section Incharge": column_aliases["section_incharge"],
        }
        issues: list[dict[str, str]] = []
        missing = [
            label
            for label, aliases in required_for_logging.items()
            if not any(_normalize_col_key(alias) in col_lookup for alias in aliases)
        ]
        if missing:
            LOGGER.warning("Monthly Plan (Stringing) missing columns: %s", ", ".join(missing))
            issues.append(
                {
                    "workbook": str(source_path or ""),
                    "sheet": sheet_name or "",
                    "issue": f"Missing columns: {', '.join(missing)}",
                }
            )

        project_names = _optional_series(("Project Name", "Project", "Project Title", "project_name")).map(_normalize_text)
        project_codes = _optional_series(("Project Code", "Project Key", "Project Id", "Project ID", "project")).map(
            _normalize_text
        )
        if project_hint:
            project_names = project_names.where(project_names.astype(bool), project_hint)
            project_codes = project_codes.where(project_codes.astype(bool), project_hint)
        serial_values, _ = _resolve_series("serial", default="")
        serial_values = serial_values.map(_normalize_text)
        span_from, _ = _resolve_series("span_from", default="")
        span_from = span_from.map(_normalize_text)
        span_to, _ = _resolve_series("span_to", default="")
        span_to = span_to.map(_normalize_text)
        span_length_series, _ = _resolve_series("span_length", default=0.0)
        span_length = pd.to_numeric(span_length_series, errors="coerce").fillna(0.0)
        method_values, _ = _resolve_series("method", default="")
        method_values = method_values.map(_normalize_text)
        gang_strength_series, gang_has_col = _resolve_series("gang_strength", default=pd.NA)
        gang_strength = pd.to_numeric(gang_strength_series, errors="coerce")
        paying_out_start_series, _ = _resolve_series("paying_out_start", default=pd.NaT)
        paying_out_complete_series, _ = _resolve_series("paying_out_complete", default=pd.NaT)
        paying_out_start = pd.to_datetime(paying_out_start_series, errors="coerce")
        paying_out_complete = pd.to_datetime(paying_out_complete_series, errors="coerce")
        final_sag_complete_series, _ = _resolve_series("final_sag_complete", default=pd.NaT)
        final_sag_complete = pd.to_datetime(final_sag_complete_series, errors="coerce")

        entity_sources: list[tuple[str, list[str]]] = []
        gang_series, has_gang = _resolve_series("gang_name", default="")
        if has_gang:
            entity_sources.append(("Gang", gang_series.map(_normalize_text).tolist()))
        supervisor_series, has_supervisor = _resolve_series("supervisor", default="")
        if has_supervisor:
            entity_sources.append(("Supervisor", supervisor_series.map(_normalize_text).tolist()))
        section_series, has_section = _resolve_series("section_incharge", default="")
        if has_section:
            entity_sources.append(("Section Incharge", section_series.map(_normalize_text).tolist()))

        normalized_rows: list[dict[str, Any]] = []
        completion_pairs: set[tuple[str, str]] = set()

        span_count = len(df_raw.index)
        span_done_mask = (paying_out_start.notna() & final_sag_complete.notna()).tolist()
        from_vals = span_from.tolist()
        to_vals = span_to.tolist()
        project_name_vals = project_names.tolist()
        project_code_vals = project_codes.tolist()
        serial_vals = serial_values.tolist()
        span_length_vals = span_length.tolist()
        method_vals = method_values.tolist()
        gang_strength_vals = gang_strength.tolist()
        paying_out_values = paying_out_start.tolist()
        paying_out_complete_values = paying_out_complete.tolist()
        final_sag_values = final_sag_complete.tolist()

        for idx in range(span_count):
            project_name = project_name_vals[idx]
            project_code = project_code_vals[idx] or project_name
            from_ap = from_vals[idx]
            to_ap = to_vals[idx]
            serial_label = serial_vals[idx]
            if from_ap and to_ap:
                span_label = f"{from_ap} \u2192 {to_ap}"
            else:
                span_label = from_ap or to_ap or serial_label or f"Span {idx + 1}"
            span_norm = _normalize_location(span_label)
            span_length_value = float(span_length_vals[idx]) if pd.notna(span_length_vals[idx]) else 0.0
            method_value = method_vals[idx]
            span_completed = bool(span_done_mask[idx])
            po_start_value = paying_out_values[idx]
            po_complete_value = paying_out_complete_values[idx]
            sag_complete_value = final_sag_values[idx]
            gang_strength_value = gang_strength_vals[idx]

            base_projects = [
                _normalize_lower(project_name),
                _normalize_lower(project_code),
            ]
            if span_completed and span_norm and any(base_projects):
                for candidate in base_projects:
                    if candidate:
                        completion_pairs.add((candidate, span_norm))

            for entity_label, entity_values in entity_sources:
                entity_name = entity_values[idx]
                if not entity_name:
                    continue
                normalized_rows.append(
                    {
                        "project_name": project_name,
                        "project_key": project_code or project_name,
                        "entity_type": entity_label,
                        "entity_name": entity_name,
                        "location_no": span_label,
                        "tower_weight": span_length_value,
                        "revenue_planned": 0.0,
                        "revenue_realised": 0.0,
                        "stringing_span_completed": span_completed,
                        "span_from": from_ap,
                        "span_to": to_ap,
                        "method": method_value,
                        "gang_strength": gang_strength_value,
                        "paying_out_start": po_start_value,
                        "paying_out_complete": po_complete_value,
                        "final_sag_complete": sag_complete_value,
                    }
                )

        normalized = pd.DataFrame(normalized_rows)
        required_payload_columns: list[tuple[str, Any]] = [
            ("project_name", ""),
            ("project_key", ""),
            ("entity_type", ""),
            ("entity_name", ""),
            ("location_no", ""),
            ("tower_weight", 0.0),
            ("revenue_planned", 0.0),
            ("revenue_realised", 0.0),
            ("stringing_span_completed", False),
            ("paying_out_complete", pd.NaT),
        ]
        for column, default in required_payload_columns:
            if column not in normalized.columns:
                normalized[column] = default
        if "completion_date" not in normalized.columns:
            normalized["completion_date"] = pd.NaT
        normalized["completion_date"] = pd.to_datetime(normalized["completion_date"], errors="coerce")

        def _fill_completion_from(column_name: str) -> None:
            if column_name not in normalized.columns:
                return
            fallback = pd.to_datetime(normalized[column_name], errors="coerce")
            if fallback is None:
                return
            normalized["completion_date"] = normalized["completion_date"].fillna(fallback)

        _fill_completion_from("final_sag_complete")
        _fill_completion_from("paying_out_complete")
        return normalized, completion_pairs, issues

    def _stringing_plan_output_path(cfg: AppConfig) -> Path:
        base = Path(cfg.data_path).expanduser()
        candidates: list[Path] = []

        def _push_directory(path_like: Path | str | None) -> None:
            if not path_like:
                return
            path_obj = Path(path_like).expanduser()
            if path_obj.suffix.lower() == ".xlsx":
                candidates.append(path_obj)
            else:
                candidates.append(path_obj / "StringingCompiled_Output.xlsx")

        if base.is_file():
            _push_directory(base.parent / "Stringing")
        else:
            _push_directory(base.parent / "Stringing")
            _push_directory(base.parent.parent / "Stringing")

        for rel in getattr(cfg, "stringing_parquet_dirs", ()):
            try:
                rel_path = (base / Path(rel)).resolve()
            except Exception:
                continue
            _push_directory(rel_path)

        _push_directory(Path("Parquets") / "Stringing")
        if base.is_file():
            _push_directory(base.with_name("StringingCompiled_Output.xlsx"))
        else:
            _push_directory(base / "Stringing")
        _push_directory(Path("Parquets"))
        _push_directory(Path("."))

        for candidate in candidates:
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                return candidate
            except Exception:
                continue
        fallback = Path("Parquets") / "Stringing" / "StringingCompiled_Output.xlsx"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback

    def _write_stringing_plan_snapshot(
        cfg: AppConfig,
        frame: pd.DataFrame,
        issues: list[dict[str, str]],
        index_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        output_path = _stringing_plan_output_path(cfg)
        try:
            mode = "a" if output_path.exists() else "w"
            with pd.ExcelWriter(
                output_path,
                engine="openpyxl",
                mode=mode,
                if_sheet_exists="replace",
            ) as writer:
                responsibilities_frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
                if responsibilities_frame is None or responsibilities_frame.empty:
                    responsibilities_frame = pd.DataFrame()
                for sheet in ("Stringing Plan", "MicroPlanResponsibilities"):
                    responsibilities_frame.to_excel(writer, sheet_name=sheet, index=False)
                index_frame = pd.DataFrame(index_rows or [])
                index_frame.to_excel(writer, sheet_name="MicroPlanIndex", index=False)
                issues_frame = (
                    pd.DataFrame(issues)
                    if issues
                    else pd.DataFrame([{"issue": "No issues detected"}])
                )
                for sheet in ("Stringing Plan Issues", "MicroPlanDataIssues"):
                    issues_frame.to_excel(writer, sheet_name=sheet, index=False)
        except Exception as exc:
            LOGGER.warning("Unable to write Stringing plan snapshot to '%s': %s", output_path, exc)

    def _maybe_write_stringing_plan_snapshot(
        cfg: AppConfig,
        frame: pd.DataFrame,
        issues: list[dict[str, str]],
        index_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        cache = _STRINGING_PLAN_CACHE
        ts_now = time.time()
        last_written = cache.get("last_written", 0.0)
        if ts_now - last_written < _STRINGING_PLAN_CACHE_TTL_SECONDS:
            return
        _write_stringing_plan_snapshot(cfg, frame, issues, index_rows or [])
        cache["last_written"] = ts_now

    def _load_stringing_plan_snapshot(
        cfg: AppConfig,
    ) -> tuple[pd.DataFrame | None, set[tuple[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
        cache = _STRINGING_PLAN_CACHE
        ts_now = time.time()
        cached_frame = cache["frame"]
        if (
            isinstance(cached_frame, pd.DataFrame)
            and not cached_frame.empty
            and (ts_now - cache["stored_at"] < _STRINGING_PLAN_CACHE_TTL_SECONDS)
        ):
            issues = cache["issues"]
            completion = cache["completion"]
            index_rows = cache.get("index", [])
            return cached_frame.copy(), set(completion), list(issues), list(index_rows)

        payload = stringing_plan_accessor.load()
        if payload.has_frame:
            frame = payload.frame.copy()
            completion_keys = set(payload.completion_keys or set())
            cache["frame"] = frame.copy()
            cache["completion"] = set(completion_keys)
            cache["issues"] = []
            cache["index"] = []
            cache["stored_at"] = ts_now
            cache["last_written"] = ts_now
            return frame, completion_keys, [], []

        root = _resolve_stringing_microplan_root(config.data_path)
        frames: list[pd.DataFrame] = []
        completion_keys: set[tuple[str, str]] = set()
        issues: list[dict[str, str]] = []
        index_rows: list[dict[str, Any]] = []

        def _append_index_entry(
            *,
            workbook: Path | str | None,
            sheet_name: str | None,
            project_code: str,
            project_label: str,
            rows_cleaned: int,
            status: str,
            error: str,
            issues_logged: int = 0,
            input_rows: int = 0,
            plan_month: str = "",
        ) -> None:
            index_rows.append(
                {
                    "file_path": str(workbook or ""),
                    "sheet_name": sheet_name or "",
                    "project_name": project_label or project_code or "",
                    "project_key": project_code or project_label or "",
                    "rows_cleaned": rows_cleaned,
                    "input_rows": input_rows,
                    "issues_logged": issues_logged,
                    "plan_month": plan_month,
                    "status": status,
                    "error": error or "",
                }
            )

        def _infer_plan_month_from_frame(frame: pd.DataFrame) -> str:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return ""
            candidate_columns = (
                "plan_month",
                "completion_date",
                "final_sag_complete",
                "paying_out_complete",
                "paying_out_start",
            )
            for column in candidate_columns:
                if column not in frame.columns:
                    continue
                series = pd.to_datetime(frame[column], errors="coerce")
                series = series.dropna()
                if series.empty:
                    continue
                ts = pd.Timestamp(series.iloc[0])
                return ts.to_period("M").to_timestamp().strftime("%Y-%m")
            return ""

        if not root.exists():
            message = f"Stringing micro plan root not found: {root}"
            LOGGER.warning(message)
            issues.append({"workbook": str(root), "sheet": "", "issue": "ROOT_NOT_FOUND"})
            _append_index_entry(
                workbook=root,
                sheet_name="",
                project_code="",
                project_label="",
                rows_cleaned=0,
                status="error",
                error="ROOT_NOT_FOUND",
            )
        else:
            preferred_sheet = getattr(cfg, "stringing_sheet_name", "")
            for workbook in sorted(root.rglob("*.xlsx")):
                if workbook.name.startswith("~$"):
                    continue
                project_code, project_label = _infer_project_hint(workbook)
                try:
                    xls = pd.ExcelFile(workbook)
                except Exception as exc:
                    issues.append({"workbook": str(workbook), "sheet": "", "issue": f"OPEN_FAILED: {exc}"})
                    _append_index_entry(
                        workbook=workbook,
                        sheet_name="",
                        project_code=project_code,
                        project_label=project_label,
                        rows_cleaned=0,
                        status="error",
                        error=f"OPEN_FAILED: {exc}",
                    )
                    continue
                sheet_name = next(
                    (
                        name
                        for name in xls.sheet_names
                        if preferred_sheet and _normalize_col_key(name) == _normalize_col_key(preferred_sheet)
                    ),
                    None,
                )
                if sheet_name is None:
                    sheet_name = next((name for name in xls.sheet_names if "string" in name.lower()), None)
                if sheet_name is None:
                    issues.append({"workbook": str(workbook), "sheet": "", "issue": "NO_STRINGING_SHEET"})
                    _append_index_entry(
                        workbook=workbook,
                        sheet_name="",
                        project_code=project_code,
                        project_label=project_label,
                        rows_cleaned=0,
                        status="error",
                        error="NO_STRINGING_SHEET",
                    )
                    continue
                try:
                    df_raw = pd.read_excel(workbook, sheet_name=sheet_name, header=1)
                except Exception as exc:
                    issues.append({"workbook": str(workbook), "sheet": sheet_name, "issue": f"READ_FAILED: {exc}"})
                    _append_index_entry(
                        workbook=workbook,
                        sheet_name=sheet_name,
                        project_code=project_code,
                        project_label=project_label,
                        rows_cleaned=0,
                        status="error",
                        error=f"READ_FAILED: {exc}",
                    )
                    continue
                normalized, _plan_keys, local_issues = _prepare_stringing_plan_frame(
                    df_raw,
                    project_hint=project_label or project_code,
                    source_path=workbook,
                    sheet_name=sheet_name,
                )
                if project_code:
                    normalized["project_key"] = normalized["project_key"].replace("", project_code)
                if project_label:
                    normalized["project_name"] = normalized["project_name"].replace("", project_label)
                frames.append(normalized)
                issues.extend(local_issues)
                plan_month_value = _infer_plan_month_from_frame(normalized)
                _append_index_entry(
                    workbook=workbook,
                    sheet_name=sheet_name,
                    project_code=project_code,
                    project_label=project_label,
                    rows_cleaned=int(len(normalized)),
                    input_rows=int(len(df_raw)),
                    status="ok",
                    error="",
                    issues_logged=len(local_issues),
                    plan_month=plan_month_value,
                )

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        _write_stringing_plan_snapshot(cfg, frame, issues, index_rows)
        cache["frame"] = frame.copy()
        cache["completion"] = set(completion_keys)
        cache["issues"] = list(issues)
        cache["index"] = list(index_rows)
        cache["stored_at"] = ts_now
        cache["last_written"] = ts_now
        return frame, completion_keys, issues, index_rows

    # --- shared: responsibilities figure + KPIs for a single project selection ---
    def _stringing_length_km_series(frame: pd.DataFrame) -> pd.Series:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.Series([], dtype=float)

        length_km_cols = [
            "length_km",
            "string_km",
            "planned_km",
            "po_km",
            "daily_km",
        ]
        meter_cols = [
            "length_m",
            "length (m)",
            "span_m",
            "span (m)",
            "span length",
            "length",
            "p/o",
        ]

        for col in length_km_cols:
            if col in frame.columns:
                series = pd.to_numeric(frame[col], errors="coerce")
                return series.fillna(0.0)

        for col in meter_cols:
            if col in frame.columns:
                series = pd.to_numeric(frame[col], errors="coerce")
                return (series / 1000.0).fillna(0.0)

        if "tower_weight" in frame.columns:
            series = pd.to_numeric(frame["tower_weight"], errors="coerce")
            return (series / 1000.0).fillna(0.0)

        return pd.Series(0.0, index=frame.index, dtype=float)

    def _build_monthly_plan_for_project(
        project_value: str | Sequence[str] | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
        *,
        plan_mode: str = "erection",
    ):
        plan_key = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        plan_title = "Monthly Plan (Stringing)" if plan_key == "stringing" else "Monthly Plan (Erection)"
        plan_noun = "Stringing plan" if plan_key == "stringing" else "Monthly Plan"

        def _empty_response(message: str):
            empty_fig = build_empty_responsibilities_figure(message)
            return empty_fig, "\u2014", "\u2014", "\u2014"

        candidates_raw: list[Any] = []
        if isinstance(project_value, dict):
            candidates_raw.extend([
                project_value.get("name"),
                project_value.get("code"),
            ])
            candidates_raw.extend(project_value.values())
        elif isinstance(project_value, Sequence) and not isinstance(project_value, (str, bytes)):
            candidates_raw.extend(project_value)
        elif project_value is not None:
            candidates_raw.append(project_value)

        project_candidates: list[str] = []
        seen_candidates: set[str] = set()
        for candidate in candidates_raw:
            text = "" if candidate is None else str(candidate).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            project_candidates.append(text)

        if not project_candidates:
            return _empty_response("Select a single project to view its details.")

        entity = (entity_value or "Supervisor").strip()
        metric = (metric_value or "tower_weight").strip()
        metric = metric if metric in {"revenue", "tower_weight"} else "tower_weight"
        stringing_length_label = plan_key == "stringing" and metric == "tower_weight"
        stringing_length_label = plan_key == "stringing" and metric == "tower_weight"

        df_atomic, completed_keys, load_error_msg, workbook = _fetch_monthly_plan(
            plan_key,
            allow_workbook_fallback=True,
        )
        if df_atomic is None or df_atomic.empty:
            message = load_error_msg or f"No {plan_title} data found in the compiled workbook."
            return _empty_response(message)

        completion_keys = set(completed_keys or set())
        df_atomic = df_atomic.copy()
        plan_source_path = None
        plan_sheet_name = None
        if workbook is not None:
            plan_source_path = getattr(workbook, "_plan_workbook_path", None)
            plan_sheet_name = getattr(workbook, "_plan_sheet_name", None)
            if plan_source_path is None:
                plan_source_path = getattr(workbook, "io", None)
        plan_issues: list[dict[str, str]] = []
        if plan_key == "stringing":
            if workbook is not None and "stringing_span_completed" not in df_atomic.columns:
                df_atomic, _plan_completion_keys, plan_issues = _prepare_stringing_plan_frame(
                    df_atomic,
                    source_path=plan_source_path,
                    sheet_name=plan_sheet_name,
                )
            elif "stringing_span_completed" in df_atomic.columns:
                df_atomic["stringing_span_completed"] = df_atomic["stringing_span_completed"].fillna(False)
            if workbook is not None:
                _maybe_write_stringing_plan_snapshot(config, df_atomic, plan_issues, [])

        month_list = _ensure_list(months_value)
        months_ts = resolve_months(month_list, quick_range_value)
        active_months = sorted({ts for ts in months_ts if pd.notna(ts)})

        if "plan_month" in df_atomic.columns:
            df_atomic["plan_month"] = pd.to_datetime(
                df_atomic["plan_month"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            df_atomic["completion_month"] = df_atomic["plan_month"]
        elif "completion_date" in df_atomic.columns:
            df_atomic["completion_month"] = pd.to_datetime(
                df_atomic["completion_date"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        else:
            df_atomic["completion_month"] = pd.NaT

        # text normalizers (copy from local scope to avoid shadowing)
        def _norm_text(v: object) -> str:
            s = str(v).replace("\u00a0", " ").strip()
            low = s.lower()
            if low in {"", "nan", "none", "null"}:
                return ""
            return s

        def _norm_lc(v: object) -> str:
            return _norm_text(v).lower()

        def _norm_loc(v: object) -> str:
            t = _norm_text(v)
            if not t:
                return ""
            if t.endswith(".0") and t.replace(".", "", 1).isdigit():
                t = t.split(".", 1)[0]
            return t

        for c in ("project_key", "project_name", "entity_type", "entity_name", "location_no"):
            if c not in df_atomic.columns:
                df_atomic[c] = ""
        df_atomic["project_name_lc"] = df_atomic["project_name"].map(_norm_lc)
        df_atomic["project_key_lc"] = df_atomic["project_key"].astype(str).map(_norm_lc)
        df_atomic["location_no_norm"] = df_atomic["location_no"].map(_norm_loc)

        # Filter to selected months
        if active_months:
            df_atomic = df_atomic[df_atomic["completion_month"].isin(active_months)].copy()

        # Filter by project (supports name or code; robust compact match)
        df_entity = pd.DataFrame()
        for candidate in project_candidates:
            sel = _norm_lc(candidate)
            mask_name_or_key = (
                (df_atomic["project_name_lc"] == sel) | (df_atomic["project_key_lc"] == sel)
            )
            if not mask_name_or_key.any():
                import re as _re
                sel_compact = _re.sub(r"[^a-z0-9]", "", sel)
                project_name_compact = df_atomic["project_name_lc"].str.replace(r"[^a-z0-9]", "", regex=True)
                project_key_compact = df_atomic["project_key_lc"].str.replace(r"[^a-z0-9]", "", regex=True)
                mask_name_or_key = (
                    (project_name_compact == sel_compact) | (project_key_compact == sel_compact)
                )
            if mask_name_or_key.any():
                df_entity = df_atomic[mask_name_or_key].copy()
                break

        if df_entity.empty:
            return _empty_response("No plan entries found for the selected project.")

        if stringing_length_label:
            df_entity["__stringing_length_km"] = _stringing_length_km_series(df_entity)

        # Entity filter (Supervisor / Section Incharge / Gang)
        ent_map = {
            "supervisor": "supervisor",
            "supervisors": "supervisor",
            "section incharge": "section incharge",
            "section-incharge": "section incharge",
            "section in-charge": "section incharge",
            "gang": "gang",
            "gangs": "gang",
        }
        entity_norm = ent_map.get(entity.lower(), entity.lower())
        df_entity["entity_type_lc"] = df_entity["entity_type"].map(_norm_lc)
        df_entity = df_entity[df_entity["entity_type_lc"] == entity_norm].copy()

        if df_entity.empty:
            return _empty_response("No plan entries found for the selected filters.")

        df_entity["is_completed"] = [
            (proj, loc) in completed_keys
            for proj, loc in zip(df_entity["project_name_lc"], df_entity["location_no_norm"])
        ]

        df_entity["revenue_planned"] = pd.to_numeric(df_entity.get("revenue_planned", 0.0), errors="coerce").fillna(0.0)
        df_entity["revenue_realised"] = pd.to_numeric(df_entity.get("revenue_realised", 0.0), errors="coerce").fillna(0.0)
        df_entity["tower_weight"] = pd.to_numeric(df_entity.get("tower_weight", 0.0), errors="coerce").fillna(0.0)
        if stringing_length_label:
            length_series = pd.to_numeric(
                df_entity.get("__stringing_length_km"), errors="coerce"
            ).fillna(0.0)
            df_entity["tower_weight"] = length_series

        df_entity["delivered_revenue"] = np.where(
            df_entity["revenue_realised"] > 0,
            df_entity["revenue_realised"],
            np.where(df_entity["is_completed"], df_entity["revenue_planned"], 0.0),
        )
        df_entity["delivered_tower_weight"] = np.where(
            df_entity["is_completed"], df_entity["tower_weight"], 0.0
        )

        df_entity = df_entity[df_entity.get("entity_name", "").astype(bool)].copy()
        if df_entity.empty:
            return _empty_response("No plan entries found for the selected filters.")

        aggregated = (
            df_entity.groupby("entity_name", as_index=False)[
                [
                    "revenue_planned",
                    "delivered_revenue",
                    "tower_weight",
                    "delivered_tower_weight",
                    "location_no",
                ]
            ].agg({
                "revenue_planned": "sum",
                "delivered_revenue": "sum",
                "tower_weight": "sum",
                "delivered_tower_weight": "sum",
                "location_no": lambda s: [str(v).strip() for v in s if str(v).strip()],
            })
        )

        target_metric_col = "revenue_planned" if metric == "revenue" else "tower_weight"
        delivered_metric_col = ("delivered_revenue" if metric == "revenue" else "delivered_tower_weight")

        # Derive location lists
        filtered_target = df_entity[df_entity[target_metric_col] > 0]
        if filtered_target.empty:
            filtered_target = df_entity
        target_locations = (
            filtered_target.groupby("entity_name")["location_no"].apply(list).rename("target_locations")
        )
        filtered_delivered = df_entity[df_entity[delivered_metric_col] > 0]
        delivered_locations = (
            filtered_delivered.groupby("entity_name")["location_no"].apply(list).rename("delivered_locations")
        )
        aggregated = aggregated.merge(target_locations, on="entity_name", how="left")
        aggregated = aggregated.merge(delivered_locations, on="entity_name", how="left")

        aggregated["delivered_value"] = np.where(
            metric == "revenue",
            aggregated["delivered_revenue"],
            aggregated["delivered_tower_weight"],
        )

        axis_override = "Length (KM)" if stringing_length_label else None
        unit_override = "KM" if stringing_length_label else None

        # Ensure chart builder has the target column by metric name
        if "revenue_planned" in aggregated.columns and "revenue" not in aggregated.columns:
            aggregated["revenue"] = aggregated["revenue_planned"]

        if aggregated.empty:
            return _empty_response("No responsibilities found for the selected filters.")

        fig = build_responsibilities_chart(
            aggregated,
            entity_label=entity,
            metric=metric,
            axis_title_override=axis_override,
            unit_label_override=unit_override,
            title=None,
            top_n=20,
        )

        if metric == "revenue":
            total_target = float(aggregated["revenue_planned"].sum())
            total_delivered = float(aggregated["delivered_revenue"].sum())
        else:
            total_target = float(aggregated["tower_weight"].sum())
            total_delivered = float(aggregated["delivered_tower_weight"].sum())

        achievement = 0.0 if total_target == 0 else (total_delivered / total_target) * 100.0

        def fmt_num(value: float) -> str:
            if metric == "revenue":
                return f"\u20b9{value:,.0f}"
            unit = "KM" if stringing_length_label else "MT"
            precision = 1 if stringing_length_label else 0
            return f"{value:,.{precision}f} {unit}"

        kpi_target_txt = fmt_num(total_target)
        kpi_deliv_txt = fmt_num(total_delivered)
        kpi_ach_txt = f"{achievement:.0f}%"

        return fig, kpi_target_txt, kpi_deliv_txt, kpi_ach_txt
    
    def _get_project_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
        if project_baseline_provider is None:
            return {}, {}
        try:
            overall_map, monthly_map = project_baseline_provider()
        except Exception as exc:
            LOGGER.warning("Failed to retrieve project baselines: %s", exc)
            return {}, {}
        return overall_map or {}, monthly_map or {}

    def _format_summary_value(value: float | None, unit_label: str | None = None, *, precision: int = 1) -> str:
        if value is None or pd.isna(value):
            return "-"
        try:
            numeric = float(value)
        except Exception:
            return "-"
        formatted = f"{numeric:,.{precision}f}"
        return f"{formatted} {unit_label}" if unit_label else formatted

    def _avg_metric_value(df: pd.DataFrame, metric_col: str, is_stringing: bool) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty or metric_col not in df.columns:
            return 0.0
        if is_stringing:
            required_cols = {"gang_name", "month"}
            if not required_cols.issubset(df.columns):
                return 0.0
            monthly_totals = (
                df.groupby(["gang_name", "month"], dropna=True)[metric_col]
                .sum()
                .reset_index(name="monthly_value")
            )
            if monthly_totals.empty:
                return 0.0
            return float(monthly_totals["monthly_value"].mean())
        return float(pd.to_numeric(df[metric_col], errors="coerce").dropna().mean())

    def _empty_summary_payload(is_stringing: bool) -> dict[str, str]:
        payload = {
            "projects": "-",
            "totals": "-",
            "gangs": "-",
            "productivity": "-",
            "lost_units": "-",
        }
        if is_stringing:
            payload["po_completion"] = "-"
            payload["tse"] = "-"
        return payload

    def _compute_mode_summary_components(
        *,
        scoped_full: pd.DataFrame,
        scoped_all: pd.DataFrame,
        months_ts: list[pd.Timestamp],
        days_factor: float,
        metric_col: str,
        is_stringing: bool,
        meta_signature: str,
        cache_key: str | None,
        scope_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "has_rows": not (scoped_full.empty and scoped_all.empty),
            "loss_scope": pd.DataFrame(),
            "history_scope": pd.DataFrame(),
            "projects_count": 0,
            "gangs_count": 0,
            "total_delivered": 0.0,
            "total_lost": 0.0,
            "total_potential": 0.0,
            "balance_value": 0.0,
        }
        if not result["has_rows"]:
            return result

        has_selected_months = bool(months_ts)
        earliest_month = None
        if has_selected_months and not scoped_all.empty and "month" in scoped_all.columns:
            month_values = sorted({ts for ts in months_ts if pd.notna(ts)})
            period_mask = scoped_all["month"].isin(month_values)
            loss_scope = scoped_all.loc[period_mask].copy()
            earliest_month = month_values[0] if month_values else None
            history_scope = scoped_all.loc[scoped_all["month"] < (earliest_month or pd.Timestamp.max)].copy()
        else:
            loss_scope = scoped_all.copy()
            history_scope = scoped_all.copy()

        if not loss_scope.empty:
            if "gang_name" in loss_scope.columns:
                loss_scope = loss_scope.dropna(subset=["gang_name"])
                loss_scope["gang_name"] = loss_scope["gang_name"].astype(str).str.strip()
            if "project_name" in loss_scope.columns:
                loss_scope["project_name"] = loss_scope["project_name"].astype(str).str.strip()

        result["loss_scope"] = loss_scope
        result["history_scope"] = history_scope

        def _maybe_cached(token: str, producer: Callable[[], T], *, clone: Callable[[T], T] | None = None) -> T:
            if cache_key:
                return _cached_scope_result(cache_key, token, producer, clone=clone)
            value = producer()
            return clone(value) if clone else value

        precomputed_overall, precomputed_monthly = _get_project_baselines()
        use_precomputed = (not is_stringing) and bool(precomputed_overall)
        proj_overall_all: dict[str, float] = {}
        proj_monthly: dict[str, dict[pd.Timestamp, float]] = {}

        if use_precomputed:
            if "project_name" in scoped_all.columns:
                available_projects = (
                    scoped_all["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                available_projects = []
            if available_projects:
                proj_overall_all = {
                    project: precomputed_overall.get(project)
                    for project in available_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in available_projects
                }
            else:
                proj_overall_all = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if has_selected_months and earliest_month is not None:
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < earliest_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < earliest_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token_all = f"project-baseline::{metric_col}::{meta_signature}"

            def _compute_baseline_all() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if scoped_all.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(scoped_all, metric_col)
                return compute_project_baseline_maps(scoped_all)

            proj_overall_all, proj_monthly_all = _maybe_cached(
                baseline_token_all,
                _compute_baseline_all,
                clone=_clone_baseline_result,
            )

            if has_selected_months and earliest_month is not None:
                proj_overall_all = {
                    project: value
                    for project, value in proj_overall_all.items()
                    if value is not None
                }
                filtered: dict[str, dict[pd.Timestamp, float]] = {}
                for project, month_map in proj_monthly_all.items():
                    subset = {month: value for month, value in month_map.items() if month < earliest_month}
                    if subset:
                        filtered[project] = subset
                proj_monthly = filtered
            else:
                proj_monthly = proj_monthly_all

        gang_to_project = (
            scoped_all[["gang_name", "project_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("gang_name")["project_name"]
            .astype(str)
            .to_dict()
        )

        baseline_overall_map = {g: proj_overall_all.get(p) for g, p in gang_to_project.items()}
        baseline_monthly_map = {g: proj_monthly.get(p, {}) for g, p in gang_to_project.items()}

        loss_token = f"loss::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}"

        def _compute_loss_rows() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            if loss_scope.empty:
                return rows
            for gang_name, gang_df in loss_scope.groupby("gang_name"):
                if gang_df.empty:
                    continue
                overall_baseline = baseline_overall_map.get(gang_name)
                if is_stringing:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss_for_column(
                        gang_df,
                        metric_column=metric_col,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                    )
                else:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                        gang_df,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_mt_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                    )
                rows.append(
                    {
                        "gang_name": gang_name,
                        "delivered": delivered,
                        "lost": loss_mt,
                        "potential": potential,
                        "avg_prod": (gang_df[metric_col].mean() if metric_col in gang_df.columns else 0.0),
                        "baseline": baseline,
                    }
                )
            return rows

        loss_rows = _maybe_cached(
            loss_token,
            _compute_loss_rows,
            clone=_clone_loss_rows,
        )

        loss_df = pd.DataFrame(loss_rows)
        if is_stringing and not loss_df.empty:
            loss_df["avg_prod"] = loss_df["avg_prod"].astype(float) * days_factor
            loss_df["baseline"] = loss_df["baseline"].astype(float) * days_factor

        metric_scope = loss_scope if not loss_scope.empty else scoped_full
        total_metric = (
            float(metric_scope[metric_col].sum())
            if not metric_scope.empty and metric_col in metric_scope.columns
            else 0.0
        )
        total_delivered = float(loss_df["delivered"].sum()) if not loss_df.empty else 0.0
        total_lost = float(loss_df["lost"].sum()) if not loss_df.empty else 0.0
        total_potential = total_delivered + total_lost if (total_delivered or total_lost) else total_metric
        balance_value = max(total_potential - total_delivered, 0.0)

        completion_override: float | None = None
        if is_stringing:
            completion_override = _sum_completion_totals(
                scoped_full,
                value_column="length_km",
                completion_column="fs_complete_date",
                fallback_columns=[
                    ("span (m)", 0.001),
                    ("span_m", 0.001),
                    ("length", 0.001),
                    ("length_m", 0.001),
                    ("tower_weight", 0.001),
                ],
            )
        else:
            completion_override = _sum_completion_totals(
                scoped_full,
                value_column="tower_weight",
                completion_column="completion_date",
            )
        if completion_override is not None:
            total_delivered = completion_override
            total_metric = completion_override if not total_lost else completion_override + total_lost
            total_potential = total_delivered + total_lost if total_lost else total_delivered
            balance_value = max(total_potential - total_delivered, 0.0)

        def _nunique(frame: pd.DataFrame, column: str) -> int:
            if column not in frame.columns or frame.empty:
                return 0
            return (
                frame[column]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )

        projects_count = _nunique(loss_scope, "project_name") or _nunique(scoped_full, "project_name")
        if is_stringing:
            try:
                plan_df, _ = _stringing_plan_totals_by_project(
                    months_ts,
                    current_month=months_ts[-1] if months_ts else None,
                )
            except Exception:
                plan_df = pd.DataFrame()
            if isinstance(plan_df, pd.DataFrame) and not plan_df.empty:
                plan_keys = set(plan_df.index.astype(str))
                selected = (scope_meta or {}).get("selected") or {}
                project_filters = _normalize_str_list(selected.get("projects"))
                if project_filters:
                    filter_keys = {
                        _compact_project_key(value) or _normalize_lower(value)
                        for value in project_filters
                        if str(value).strip()
                    }
                    filter_keys = {key for key in filter_keys if key}
                    if filter_keys:
                        plan_keys = {key for key in plan_keys if key in filter_keys}
                delivered_keys: set[str] = set()
                for frame in (loss_scope, scoped_full):
                    if not isinstance(frame, pd.DataFrame) or frame.empty or "project_name" not in frame.columns:
                        continue
                    delivered_keys.update(
                        frame["project_name"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .map(lambda value: _compact_project_key(value) or _normalize_lower(value))
                    )
                delivered_keys = {key for key in delivered_keys if key}
                total_project_keys = plan_keys | delivered_keys
                total_project_keys = {key for key in total_project_keys if key}
                if total_project_keys:
                    projects_count = len(total_project_keys)
        gangs_count = _nunique(loss_scope, "gang_name") or _nunique(scoped_full, "gang_name")

        result.update(
            {
                "projects_count": projects_count,
                "gangs_count": gangs_count,
                "total_delivered": total_delivered,
                "total_lost": total_lost,
                "total_potential": total_potential,
                "balance_value": balance_value,
            }
        )
        return result

    def _get_stringing_po_daily_frame() -> pd.DataFrame:
        if not config.enable_stringing:
            return pd.DataFrame()

        def _producer() -> pd.DataFrame:
            df_compiled = pd.DataFrame()
            if callable(stringing_compiled_provider):
                try:
                    df_compiled = stringing_compiled_provider()
                except Exception:
                    df_compiled = pd.DataFrame()
            if not isinstance(df_compiled, pd.DataFrame) or df_compiled.empty:
                try:
                    df_compiled = _load_stringing_compiled_raw(config)
                except Exception:
                    df_compiled = pd.DataFrame()
            if not isinstance(df_compiled, pd.DataFrame) or df_compiled.empty:
                return pd.DataFrame()
            payout_source = df_compiled.copy()
            po_col = None
            for cand in ("paying_out_complete", "po_completion_date", "po_completion"):
                if cand in payout_source.columns:
                    po_col = cand
                    break
            if po_col is None:
                return pd.DataFrame()
            payout_source["po_completion_date"] = pd.to_datetime(payout_source[po_col], errors="coerce")
            payout_source = payout_source.dropna(subset=["po_completion_date"])
            if payout_source.empty:
                return pd.DataFrame()
            payout_source["po_start_date"] = payout_source["po_completion_date"]
            try:
                payout = expand_stringing_to_daily_payout(payout_source)
            except Exception:
                LOGGER.exception("Failed to expand P/O completion daily rows")
                return pd.DataFrame()
            if payout.empty:
                return payout
            if "project_name" not in payout.columns and "project" in payout.columns:
                payout["project_name"] = payout["project"]
            if "project" not in payout.columns and "project_name" in payout.columns:
                payout["project"] = payout["project_name"]
            if "month" not in payout.columns and "date" in payout.columns:
                payout["date"] = pd.to_datetime(payout["date"], errors="coerce")
                payout = payout.dropna(subset=["date"])
                payout["month"] = payout["date"].dt.to_period("M").to_timestamp()
            return payout

        return _cached_global_result(
            "stringing:po_completion_daily",
            _producer,
            clone=_clone_dataframe,
        )

    def _compute_po_completion_totals(
        scope_meta: dict[str, Any] | None,
        months_ts: list[pd.Timestamp],
        days_factor: float,
    ) -> str:
        plan_total = _stringing_planned_total_for_dates(
            scope_meta,
            months_ts,
            date_columns=_STRINGING_PO_DATE_COLUMNS,
        )
        has_plan_scope = _stringing_scope_has_plan(
            scope_meta,
            months_ts,
            date_columns=_STRINGING_PO_DATE_COLUMNS,
        )
        frame = _get_stringing_po_daily_frame()
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            if plan_total <= 0 and not has_plan_scope:
                return "-"
            total_txt = f"{plan_total:.1f} KM" if plan_total > 0 else "0.0 KM"
            done_txt = _format_summary_value(0.0, "KM")
            balance_txt = _format_summary_value(plan_total, "KM") if plan_total > 0 else "\u2014"
            return f"{total_txt} / {done_txt} / {balance_txt}"

        selected = (scope_meta or {}).get("selected") or {}
        projects = selected.get("projects") or []
        gangs = selected.get("gangs") or []
        kv_values = selected.get("kv") or []
        method_values = selected.get("methods") or []

        scoped_base = _stringing_scope(frame, kv_values, method_values)
        scoped_full = apply_filters(scoped_base, projects, months_ts, gangs)

        done_total = _sum_completion_totals(
            scoped_full,
            value_column="po_km",
            completion_column="po_completion_date",
            fallback_columns=[("po", 0.001)],
        ) or 0.0
        if done_total == 0.0 and isinstance(scoped_full, pd.DataFrame) and not scoped_full.empty and "daily_km" in scoped_full.columns:
            done_total = float(pd.to_numeric(scoped_full["daily_km"], errors="coerce").dropna().sum())

        done_txt = _format_summary_value(done_total, "KM")
        if plan_total > 0:
            total_txt = f"{plan_total:.1f} KM"
            balance_txt = _format_summary_value(max(plan_total - done_total, 0.0), "KM")
        elif has_plan_scope:
            total_txt = "0.0 KM"
            balance_txt = "\u2014"
        else:
            total_txt = "No Plan"
            balance_txt = "\u2014"
        return f"{total_txt} / {done_txt} / {balance_txt}"

    def _summarize_scope_for_cards(scope_meta: dict[str, Any] | None) -> dict[str, str]:
        mode = _normalize_mode((scope_meta or {}).get("mode"))
        summary = _empty_summary_payload(mode == "stringing")
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            return summary
        try:
            months_ts = _months_from_meta(scope_meta)
            days_factor = float(scope_meta.get("days_factor") or 30.0)
            is_stringing = mode == "stringing"
            metric_col = "daily_km" if is_stringing else "daily_prod_mt"
            unit_short = "KM" if is_stringing else "MT"
            prod_unit = "KM/month" if is_stringing else "MT/day"
            meta_signature = scope_meta.get("signature") or "nosig"

            scoped_full = _scope_frame_from_store(scope_meta, "full").copy()
            scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()
            if scoped_full.empty and scoped_all.empty:
                return summary

            scope_keys = scope_meta.get("scopes") or {}
            project_gang_key = scope_keys.get("project_gang")
            components = _compute_mode_summary_components(
                scoped_full=scoped_full,
                scoped_all=scoped_all,
                months_ts=months_ts,
                days_factor=days_factor,
                metric_col=metric_col,
                is_stringing=is_stringing,
                meta_signature=meta_signature,
                cache_key=project_gang_key,
                scope_meta=scope_meta,
            )

            loss_scope = components["loss_scope"]
            history_scope = components["history_scope"]
            projects_count = components["projects_count"]
            gangs_count = components["gangs_count"]
            total_delivered = components["total_delivered"]
            total_lost = components["total_lost"]
            total_potential = components["total_potential"]
            balance_value = components["balance_value"]

            def _collect_project_labels(frame: pd.DataFrame | None) -> list[str]:
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    return []
                labels: list[str] = []
                seen: set[str] = set()
                for column in (
                    "project_name",
                    "project",
                    "project_name_display",
                    "Project Name",
                    "project_code",
                ):
                    if column not in frame.columns:
                        continue
                    values = (
                        frame[column]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .replace("", pd.NA)
                        .dropna()
                    )
                    for value in values:
                        if value not in seen:
                            seen.add(value)
                            labels.append(value)
                return labels

            def _project_label_keys(text: object) -> tuple[list[str], list[str]]:
                base = str(text or "").strip()
                if not base:
                    return [], []
                parts = [base]
                if " : " in base:
                    left, right = base.split(" : ", 1)
                    parts.extend([left.strip(), right.strip()])
                norm_keys: list[str] = []
                compact_keys: list[str] = []
                for part in parts:
                    norm = _normalize_lower(part)
                    if norm and norm not in norm_keys:
                        norm_keys.append(norm)
                    compact = _compact_project_key(part)
                    if compact and compact not in compact_keys:
                        compact_keys.append(compact)
                match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{3,4})\b", base.upper())
                if match:
                    compact = _compact_project_key(f"{match.group(1)}{match.group(2)}")
                    if compact and compact not in compact_keys:
                        compact_keys.append(compact)
                return norm_keys, compact_keys

            tse_count: int | None = None
            if is_stringing:
                tse_norm_map, tse_alias_map = _get_stringing_tse_lookup()
                label_source = loss_scope if not loss_scope.empty else scoped_full
                labels = _collect_project_labels(label_source)
                matched_total = 0
                matched_any = False
                used_projects: set[str] = set()
                if labels and (tse_norm_map or tse_alias_map):
                    for label in labels:
                        norm_keys, compact_keys = _project_label_keys(label)
                        value, canonical_id = _resolve_tse_value(norm_keys, compact_keys, tse_norm_map, tse_alias_map)
                        if canonical_id and canonical_id not in used_projects and value is not None:
                            used_projects.add(canonical_id)
                            matched_total += int(value)
                            matched_any = True
                if matched_any:
                    tse_count = matched_total
                else:
                    fallback_scope = loss_scope if not loss_scope.empty else scoped_full
                    tse_count = 0
                    if {"method", "gang_name"}.issubset(fallback_scope.columns) and not fallback_scope.empty:
                        tse_mask = fallback_scope["method"].astype(str).str.strip().str.lower() == "tse"
                        tse_count = int(
                            fallback_scope.loc[tse_mask, "gang_name"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .nunique()
                        )

            prod_current = _avg_metric_value(scoped_full, metric_col, is_stringing)
            prod_history = _avg_metric_value(history_scope, metric_col, is_stringing)
            if prod_history == 0.0:
                prod_history = prod_current

            summary["projects"] = f"{projects_count:,}" if projects_count else "-"
            summary["gangs"] = f"{gangs_count:,}" if gangs_count else "-"
            if is_stringing:
                planned_total_value = _stringing_planned_total_for_dates(
                    scope_meta,
                    months_ts,
                    date_columns=_STRINGING_FS_DATE_COLUMNS,
                )
                has_plan_scope = _stringing_scope_has_plan(
                    scope_meta,
                    months_ts,
                    date_columns=_STRINGING_FS_DATE_COLUMNS,
                )
                done_txt = _format_summary_value(total_delivered, unit_short)
                if planned_total_value > 0:
                    total_txt = f"{planned_total_value:.1f} {unit_short}"
                    balance_txt = _format_summary_value(max(planned_total_value - total_delivered, 0.0), unit_short)
                elif has_plan_scope:
                    total_txt = "0.0 KM"
                    balance_txt = "\u2014"
                else:
                    total_txt = "No Plan"
                    balance_txt = "\u2014"
            else:
                tower_done = _count_completed_towers(loss_scope, months_ts)
                planned_towers, _ = _compute_planned_tower_layers(scoped_all, months_ts)
                total_towers = planned_towers or tower_done
                balance_towers = max(total_towers - tower_done, 0)
                total_txt = f"{total_towers:,}"
                done_txt = f"{tower_done:,}"
                balance_txt = f"{balance_towers:,}"
            summary["totals"] = f"{total_txt} / {done_txt} / {balance_txt}"
            prod_txt = _format_summary_value(prod_current, prod_unit, precision=2)
            hist_txt = _format_summary_value(prod_history, prod_unit, precision=2)
            summary["productivity"] = f"{prod_txt} / {hist_txt}"
            summary["lost_units"] = _format_summary_value(total_lost, unit_short)
            if is_stringing:
                summary["po_completion"] = _compute_po_completion_totals(scope_meta, months_ts, days_factor)
                summary["tse"] = "-" if tse_count is None else f"{tse_count:,}"
            return summary
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to compute %s summary card: %s", mode, exc)
            return summary

    @app.callback(
        Output("erection-card-projects", "children"),
        Output("erection-card-totals", "children"),
        Output("erection-card-gangs", "children"),
        Output("erection-card-productivity", "children"),
        Output("erection-card-loss", "children"),
        Output("stringing-card-projects", "children"),
        Output("stringing-card-totals", "children"),
        Output("stringing-card-gangs", "children"),
        Output("stringing-card-productivity", "children"),
        Output("stringing-card-loss", "children"),
        Output("stringing-card-tse", "children"),
        Input("store-filtered-scope", "data"),
    )
    def _update_mode_summary_cards(scope_meta: dict[str, Any] | None) -> tuple[str, ...]:
        selected = (scope_meta or {}).get("selected") or {}

        def _extract_list(key: str, *, lower: bool = False) -> list[str]:
            values = selected.get(key) or []
            return _normalize_str_list(values, lower=lower)

        projects = _extract_list("projects")
        months = _extract_list("months")
        gangs = _extract_list("gangs")
        kv_values = _extract_list("kv")
        method_values = _extract_list("methods", lower=True)
        quick_range = selected.get("quick_range")

        current_mode = _normalize_mode((scope_meta or {}).get("mode"))

        def _meta_for(mode_name: str) -> dict[str, Any] | None:
            if current_mode == mode_name and isinstance(scope_meta, dict):
                return scope_meta
            try:
                return _build_scope_meta_payload(
                    eff_mode=mode_name,
                    project_list=projects,
                    gang_list=gangs,
                    months_list=months,
                    quick_range=quick_range,
                    kv_values=kv_values,
                    method_values=method_values,
                    kv_list=kv_values,
                    method_list=method_values,
                )
            except Exception:
                LOGGER.exception("Unable to build %s scope for summary cards", mode_name)
                return None

        erection_summary = _summarize_scope_for_cards(_meta_for("erection"))
        if config.enable_stringing:
            stringing_summary = _summarize_scope_for_cards(_meta_for("stringing"))
        else:
            stringing_summary = _empty_summary_payload(True)

        return (
            erection_summary["projects"],
            erection_summary["totals"],
            erection_summary["gangs"],
            erection_summary["productivity"],
            erection_summary["lost_units"],
            stringing_summary["projects"],
            stringing_summary["totals"],
            stringing_summary["gangs"],
            stringing_summary["productivity"],
            stringing_summary["lost_units"],
            stringing_summary.get("tse", "-"),
        )


    # Charts OR AVP rows -> store-click-meta (robust & single source of truth)
    app.clientside_callback(
        """
        // charts OR AVP (row or overlay) -> store-click-meta
        function(lossClick, topClick, bottomClick, rowTs, tipTs) {
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;

        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];

        // --- AVP surfaces (row or overlay) only accept real, timestamped clicks
        try {
            const pid = JSON.parse(idPart);
            if (pid && (pid.type === "avp-row" || pid.type === "avp-tip")) {
            if (!prop.endsWith(".n_clicks_timestamp")) return NO; // ignore re-renders
            const ts = ctx.triggered[0].value;
            if (!ts || ts <= 0) return NO;                        // must be a real click
            const gang = pid.index;
            if (!gang) return NO;
            return { source: "g-actual-vs-bench", gang: String(gang), ts: Date.now() };
            }
        } catch(e) { /* not a pattern id; continue */ }

        // --- charts path
        let cd = null;
        if (idPart === "g-actual-vs-bench") cd = lossClick;
        else if (idPart === "g-top5")       cd = topClick;
        else if (idPart === "g-bottom5")    cd = bottomClick;
        else return NO;

        if (!cd || !cd.points || !cd.points.length) return NO;
        const pt = cd.points[0];

        // Extract gang robustly (y for horiz bars, x for vertical)
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;
        else if (typeof pt.x === "string") gang = pt.x;
        else if (pt.customdata){
            if (typeof pt.customdata === "string")      gang = pt.customdata;
            else if (Array.isArray(pt.customdata))      gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object") gang = pt.customdata.gang || pt.customdata.name || null;
        }
        if (!gang) return NO;

        return { source: idPart, gang: String(gang), ts: Date.now() };
        }
        """,
        Output("store-click-meta", "data"),
        [
        Input("g-actual-vs-bench", "clickData"),
        Input("g-top5", "clickData"),
        Input("g-bottom5", "clickData"),
        Input({"type":"avp-row","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        Input({"type":"avp-tip","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        ],
        prevent_initial_call=True,
    )


    # Keep trace gang selection in sync with the last click (chart or AVP)
    app.clientside_callback(
        """
        function(meta){
        if (!meta || !meta.gang) return window.dash_clientside.no_update;
        return meta.gang;
        }
        """,
        Output("store-selected-gang", "data"),
        Input("store-click-meta", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, rowTs, tipTs){
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;
        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];
        try {
            const pid = JSON.parse(idPart);
            if (pid && (pid.type === "project-modal-avp-row" || pid.type === "project-modal-avp-tip")) {
                if (!prop.endsWith(".n_clicks_timestamp")) return NO;
                const ts = ctx.triggered[0].value;
                if (!ts || ts <= 0) return NO;
                const gang = pid.index;
                if (!gang) return NO;
                return { source: "project-modal-actual-vs-bench", gang: String(gang), ts: Date.now() };
            }
        } catch(e) { /* ignore */ }

        let cd = null;
        if (idPart === "project-modal-actual-vs-bench") cd = lossClick;
        else if (idPart === "project-modal-top5")       cd = topClick;
        else if (idPart === "project-modal-bottom5")    cd = bottomClick;
        else return NO;

        if (!cd || !cd.points || !cd.points.length) return NO;
        const pt = cd.points[0];
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;
        else if (typeof pt.x === "string") gang = pt.x;
        else if (pt.customdata){
            if (typeof pt.customdata === "string")      gang = pt.customdata;
            else if (Array.isArray(pt.customdata))       gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object") gang = pt.customdata.gang || pt.customdata.name || null;
        }
        if (!gang) return NO;
        return { source: idPart, gang: String(gang), ts: Date.now() };
        }
        """,
        Output("store-project-modal-click-meta", "data"),
        [
            Input("project-modal-actual-vs-bench", "clickData"),
            Input("project-modal-top5", "clickData"),
            Input("project-modal-bottom5", "clickData"),
            Input({"type":"project-modal-avp-row","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
            Input({"type":"project-modal-avp-tip","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        ],
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta){
        if (!meta || !meta.gang) return window.dash_clientside.no_update;
        return meta.gang;
        }
        """,
        Output("project-modal-selected-gang", "data"),
        Input("store-project-modal-click-meta", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta){
        if(!meta || !meta.source || !meta.gang) return "";
        const CHART_SOURCES = new Set(["g-actual-vs-bench","g-top5","g-bottom5"]);
        if (!CHART_SOURCES.has(meta.source)) return "";

        // retry briefly so the anchor exists before we scroll
        let tries = 0;
        function go(){
            const anchor = document.getElementById("trace-anchor") || document.getElementById("tables-anchor");
            if (!anchor) { if (tries++ < 25) setTimeout(go, 60); return; }
            anchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        setTimeout(go, 0);
        return String(Date.now());
        }
        """,
        Output("scroll-wire", "children"),
        Input("store-click-meta", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta){
        if(!meta || !meta.source || !meta.gang) return "";
        const CHART_SOURCES = new Set(["project-modal-actual-vs-bench","project-modal-top5","project-modal-bottom5"]);
        if (!CHART_SOURCES.has(meta.source)) return "";

        let tries = 0;
        function go(){
            const anchor = document.getElementById("project-modal-trace-anchor");
            if (!anchor) { if (tries++ < 25) setTimeout(go, 60); return; }
            anchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        setTimeout(go, 0);
        return String(Date.now());
        }
        """,
        Output("project-modal-scroll-wire", "children"),
        Input("store-project-modal-click-meta", "data"),
        prevent_initial_call=True,
    )



    @app.callback(
        Output("f-project", "value"),
        Output("f-month", "value"),
        Output("f-gang", "value"),
        Output("f-quick-range", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("link-clear-quick-range", "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_filter_reset(
        reset_clicks: int | None,
        clear_quick_clicks: int | None,
    ) -> tuple[Any, Any, Any, Any]:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        # Clear only quick range if that link was clicked
        if trigger_id == "link-clear-quick-range":
            return dash.no_update, dash.no_update, dash.no_update, None
        # On mode toggle or Reset click: reset all filters to defaults
        if trigger_id == "btn-reset-filters":
            # Compute default month from the latest data date in the active mode's dataset
            try:
                df = data_selector.select("erection")
                latest_date = None
                if isinstance(df, pd.DataFrame) and not df.empty and "date" in df.columns:
                    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
                    if not dates.empty:
                        latest_date = dates.max()
                default_month = (
                    pd.Timestamp(latest_date).strftime("%Y-%m") if latest_date is not None else datetime.today().strftime("%Y-%m")
                )
            except Exception:
                default_month = datetime.today().strftime("%Y-%m")
            return None, [default_month], None, None
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    @app.callback(
        Output("store-filtered-scope", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
        prevent_initial_call=False,
    )
    def _sync_filtered_scope_store(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        kv_values: Sequence[str] | None,
        method_values: Sequence[str] | None,
    ) -> dict[str, Any]:
        eff_mode = "erection"
        project_list = _normalize_str_list(_ensure_list(projects))
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        kv_list = _normalize_str_list(kv_values)
        method_list = _normalize_str_list(method_values, lower=True)

        frames, months_ts, days_factor = _build_scope_frames(
            eff_mode,
            project_list=project_list,
            gang_list=gang_list,
            months_value=months_list,
            quick_range=quick_range,
            kv_values=kv_values,
            method_values=method_values,
        )

        scope_keys = {name: _remember_scope_frame(frame) for name, frame in frames.items()}
        rows_meta = {name: int(len(frame.index)) for name, frame in frames.items()}
        signature_payload = {
            "mode": eff_mode,
            "projects": project_list,
            "gangs": gang_list,
            "months": months_list,
            "quick_range": quick_range,
            "kv": kv_list,
            "methods": method_list,
        }
        signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return {
            "mode": eff_mode,
            "signature": signature,
            "scopes": scope_keys,
            "rows": rows_meta,
            "days_factor": days_factor,
            "months_iso": [ts.isoformat() for ts in months_ts],
            "selected": {
                "projects": project_list,
                "gangs": gang_list,
                "months": months_list,
                "quick_range": quick_range,
                "kv": kv_list,
                "methods": method_list,
            },
        }


    @app.callback(
        Output("project-modal-erections-range", "start_date"),
        Output("project-modal-erections-range", "end_date"),
        Output("project-modal-stringing-range", "start_date"),
        Output("project-modal-stringing-range", "end_date"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def _sync_modal_completion_ranges(months, quick_range):
        start, end = _completion_range_from_month_selection(months, quick_range)
        start_iso = start.date().isoformat()
        end_iso = end.date().isoformat()
        return start_iso, end_iso, start_iso, end_iso


    @app.callback(
        Output("f-project", "options"),
        Input("store-filtered-scope", "data"),
    )
    def update_project_options(scope_meta: dict[str, Any] | None) -> list[dict[str, str]]:
        try:
            scope = _scope_frame_from_store(scope_meta, "month")
            if scope.empty:
                return []
            proj_col = "project_name" if "project_name" in scope.columns else ("project" if "project" in scope.columns else None)
            if not proj_col:
                return []
            projects = (
                scope[proj_col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            projects = sorted({p for p in projects if p})
            return [{"label": project, "value": project} for project in projects]
        except Exception as exc:
            LOGGER.exception("Failed to build project options: %s", exc)
            return []

    @app.callback(
        Output("f-gang", "options"),
        Input("store-filtered-scope", "data"),
    )
    def update_gang_options(scope_meta: dict[str, Any] | None) -> list[dict[str, str]]:
        try:
            scope = _scope_frame_from_store(scope_meta, "project")
            if scope.empty or "gang_name" not in scope.columns:
                return []
            gangs = (
                scope["gang_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            gangs = sorted({g for g in gangs if g})
            return [{"label": gang, "value": gang} for gang in gangs]
        except Exception as exc:
            LOGGER.exception("Failed to build gang options: %s", exc)
            return []

    @app.callback(
        Output("f-month", "options"),
        Input("store-filtered-scope", "data"),
    )
    def update_month_options(scope_meta: dict[str, Any] | None) -> list[dict[str, str]]:
        try:
            scope = _scope_frame_from_store(scope_meta, "project_gang")
            if scope.empty or "month" not in scope.columns:
                return []
            months = sorted(pd.to_datetime(scope["month"].dropna().unique()))
            quick_range = ((scope_meta or {}).get("selected") or {}).get("quick_range")
            if quick_range:
                allowed = set(resolve_months(None, quick_range))
                months = [m for m in months if m in allowed]
            if not months:
                months = sorted(pd.to_datetime(scope["month"].dropna().unique()))
            return [{"label": m.strftime("%b %Y"), "value": m.strftime("%Y-%m")} for m in months]
        except Exception as exc:
            LOGGER.exception("Failed to build month options: %s", exc)
            return []

    @app.callback(
        Output("f-month", "value", allow_duplicate=True),
        Input("f-month", "options"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        prevent_initial_call='initial_duplicate',
    )
    def ensure_default_month(options, current_value, quick_range):
        try:
            # If user cleared all months (empty list) keep it blank instead of forcing latest.
            if isinstance(current_value, list) and len(current_value) == 0:
                return dash.no_update
            # When quick-range is active its callback sets months=None; do not override it here.
            if quick_range:
                return dash.no_update
            # If a month already selected and appears in options, keep it
            if current_value:
                selected = set(current_value if isinstance(current_value, (list, tuple)) else [current_value])
                opt_values = {opt.get("value") for opt in (options or [])}
                if selected & opt_values:
                    return dash.no_update
            # Pick the latest available month option (based on data), not today's month
            opt_values = [opt.get("value") for opt in (options or []) if isinstance(opt, dict)]
            if not opt_values:
                return dash.no_update
            # Parse values like YYYY-MM and choose the max
            def _parse(val: str):
                try:
                    y, m = val.split("-")
                    return int(y) * 100 + int(m)
                except Exception:
                    return -1
            latest = max(opt_values, key=_parse)
            return [latest]
        except Exception:
            return dash.no_update

    @app.callback(
        Output("f-month", "value", allow_duplicate=True),
        Input("f-quick-range", "value"),
        prevent_initial_call=True,
    )
    def _clear_month_value_on_quick_change(qr):
        # When a quick-range is chosen, let code derive months from it; drop stale manual months.
        if qr:
            return None
        return dash.no_update

    @app.callback(
        Output("f-quick-range", "value", allow_duplicate=True),
        Input("f-month", "value"),
        State("f-quick-range", "value"),
        prevent_initial_call=True,
    )
    def _clear_quick_range_on_month_change(months, quick_range_value):
        # Reset quick-range when manual months are selected so filters stay mutually exclusive.
        month_list = _ensure_list(months)
        if month_list and quick_range_value:
            return None
        return dash.no_update


    @app.callback(
        Output("label-resp-period", "children"),
        Output("label-stringing-plan-period", "children"),
        Output("label-perf-period", "children"),
        Output("label-gang-period", "children"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def update_period_labels(
        months: Sequence[str] | None,
        quick_range: str | None,
    ) -> tuple[str, str, str, str]:
        month_list = _ensure_list(months)
        months_ts = resolve_months(month_list, quick_range)
        label = _format_period_label(months_ts)
        return label, label, label, label

    # ---- Project Overview (dynamic body) -----------------------------------------
    @app.callback(
        Output("pd-title", "children"),
        Output("project-details", "children"),
        Input("f-project", "value"),
        prevent_initial_call=False,
    )
    def show_project_details(selected_project):
        """Render the project overview grid or an informative message."""

        default_title = "Project Overview"

        def _clean_text(raw: Any) -> str:
            if raw is None:
                return ""
            text = str(raw).strip()
            if not text:
                return ""
            if any(marker in text for marker in ("Ãƒ", "Ã‚")):
                try:
                    text = text.encode("latin-1").decode("utf-8").strip()
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            return text

        def _normalize_for_match(raw: Any) -> str:
            text = _clean_text(raw)
            return " ".join(text.lower().split())

        if isinstance(selected_project, (list, tuple)):
            cleaned = [_clean_text(value) for value in selected_project if _clean_text(value)]
            if len(cleaned) != 1:
                return (
                    default_title,
                    html.Div("Select a single project to view its details.", className="project-empty"),
                )
            selected_project = cleaned[0]
        else:
            selected_project = _clean_text(selected_project)

        if not selected_project:
            return (
                default_title,
                html.Div("Select a single project to view its details.", className="project-empty"),
            )

        if not project_info_provider:
            return (
                default_title,
                html.Div("No 'Project Details' source configured.", className="project-empty"),
            )

        df_info = project_info_provider()
        if df_info is None or df_info.empty:
            return (
                default_title,
                html.Div("No 'Project Details' sheet found in the source workbook.", className="project-empty"),
            )

        df_info = df_info.copy()
        target_norm = _normalize_for_match(selected_project)
        # Accept multiple identifier variants for robust matching
        candidate_columns = [
            col
            for col in (
                "Project Name",
                "project_name",
                "project_code",
                "Project Code",
                "key_name",
            )
            if col in df_info.columns
        ]

        row = pd.DataFrame()
        # 1) strict normalized equality against known identifier columns
        for col in candidate_columns:
            try:
                mask = df_info[col].apply(_normalize_for_match) == target_norm
            except Exception:
                mask = pd.Series(False, index=df_info.index)
            if mask.any():
                row = df_info.loc[mask]
                break

        # 2) relaxed contains on human name fields (normalized)
        if row.empty:
            for human_col in ("Project Name", "project_name"):
                if human_col in df_info.columns:
                    series = df_info[human_col].astype(str).apply(_normalize_for_match)
                    mask = series.str.contains(target_norm, case=False, na=False)
                    if mask.any():
                        row = df_info.loc[mask]
                        break

        # 3) compact code match (remove non-alphanumerics) for project codes (any case variant)
        if row.empty:
            import re as _re

            def _compact(s: str) -> str:
                return _re.sub(r"[^a-z0-9]", "", (s or "").lower())

            target_comp = _compact(selected_project)
            for code_col in ("project_code", "Project Code"):
                if code_col in df_info.columns:
                    try:
                        comp_series = (
                            df_info[code_col]
                            .astype(str)
                            .map(_clean_text)
                            .map(_compact)
                        )
                        mask = comp_series == target_comp
                        if mask.any():
                            row = df_info.loc[mask]
                            break
                    except Exception:
                        continue

        if row.empty:
            return (
                default_title,
                html.Div(f"No project details found for {selected_project}.", className="project-empty"),
            )

        record = row.iloc[0]

        def fmt_txt(key: str) -> str:
            value = record.get(key, "")
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            return _clean_text(value)

        def fmt_date(key: str) -> str:
            value = record.get(key, None)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            try:
                return pd.to_datetime(value).strftime("%d-%m-%Y")
            except Exception:
                return _clean_text(value)

        display_name = fmt_txt("Project Name") or fmt_txt("project_name") or selected_project

        # Normalize PCH display using centralized normalizer if present
        try:
            from .pch_normalizer import normalize_pch as _norm_pch_display
        except Exception:
            def _norm_pch_display(v):
                return str(v or "").strip()

        body = html.Div(
            [
                html.Div(
                    [
                        html.P("PROJECT NAME", className="project-label"),
                        html.H6(fmt_txt("project_name") or fmt_txt("Project Name"), className="project-value"),
                        html.P("CLIENT", className="project-label"),
                        html.H6(fmt_txt("client_name"), className="project-value"),
                        html.P("NOA START", className="project-label"),
                        html.H6(fmt_date("noa_start"), className="project-value"),
                        html.P("LOA END", className="project-label"),
                        html.H6(fmt_date("loa_end"), className="project-value"),
                    ],
                    className="project-col",
                ),
                html.Div(
                    [
                        html.P("PCH", className="project-label"),
                        html.H6(_norm_pch_display(record.get("pch")), className="project-value"),
                        html.P("REGIONAL MANAGER", className="project-label"),
                        html.H6(fmt_txt("regional_mgr"), className="project-value"),
                        html.P("PROJECT MANAGER", className="project-label"),
                        html.H6(fmt_txt("project_mgr"), className="project-value"),
                        html.P("PLANNING ENGINEER", className="project-label"),
                        html.H6(fmt_txt("planning_eng"), className="project-value"),
                    ],
                    className="project-col",
                ),
                html.Div(
                    [
                        html.P("SECTION INCHARGE", className="project-label"),
                        html.H6(fmt_txt("section_inch"), className="project-value"),
                        html.P("SUPERVISORS", className="project-label"),
                        html.H6(fmt_txt("supervisor"), className="project-value"),
                    ],
                    className="project-col",
                ),
            ],
            className="project-grid",
        )

        title = f"Project Overview"
        return title, body

    # --- NEW: responsibilities chart callback ---
    # --- NEW: responsibilities chart callback ---

        # Responsibilities: grouped bars + three KPIs
    def _render_monthly_plan_card(
        *,
        plan_mode: str,
        project_value: str | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
    ):
        plan_key = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        plan_title = "Monthly Plan (Stringing)" if plan_key == "stringing" else "Monthly Plan (Erection)"
        plan_noun = "Stringing plan" if plan_key == "stringing" else "Monthly Plan"

        def _empty_response(message: str):
            empty_fig = build_empty_responsibilities_figure(message)
            return empty_fig, "\u2014", "\u2014", "\u2014"

        if not project_value:
            return _empty_response("Select a single project to view its details.")

        if isinstance(project_value, (list, tuple)):
            cleaned_projects = [str(p).strip() for p in project_value if p]
            if len(cleaned_projects) != 1:
                return _empty_response("Select a single project to view its details.")
            project_value = cleaned_projects[0]

        entity_value = (entity_value or "Supervisor").strip()
        metric_value = (metric_value or "tower_weight").strip()
        metric_value = metric_value if metric_value in {"revenue", "tower_weight"} else "tower_weight"
        stringing_length_label = plan_key == "stringing" and metric_value == "tower_weight"

        df_atomic, completed_keys, load_error_msg, workbook = _fetch_monthly_plan(
            plan_key,
            allow_workbook_fallback=True,
        )
        if df_atomic is None or df_atomic.empty:
            message = load_error_msg or f"No {plan_title} data found in the compiled workbook."
            return _empty_response(message)

        df_atomic = df_atomic.copy()

        month_list = _ensure_list(months_value)
        months_ts = resolve_months(month_list, quick_range_value)
        active_months = sorted({ts for ts in months_ts if pd.notna(ts)})

        if "plan_month" in df_atomic.columns:
            df_atomic["plan_month"] = pd.to_datetime(
                df_atomic["plan_month"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            df_atomic["completion_month"] = df_atomic["plan_month"]
        elif "completion_date" in df_atomic.columns:
            df_atomic["completion_month"] = pd.to_datetime(
                df_atomic["completion_date"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        else:
            df_atomic["completion_month"] = pd.NaT

        def _normalize_text(value: object) -> str:
            text = str(value).replace("\u00a0", " ").strip()
            lowered = text.lower()
            if lowered in {"", "nan", "none", "null"}:
                return ""
            return text

        def _normalize_lower(value: object) -> str:
            return _normalize_text(value).lower()

        def _normalize_location(value: object) -> str:
            txt = _normalize_text(value)
            if not txt or txt.lower() in {"nan", "none"}:
                return ""
            if txt.endswith(".0") and txt.replace(".", "", 1).isdigit():
                txt = txt.split(".", 1)[0]
            return txt

        text_columns = ("project_key", "project_name", "entity_type", "entity_name", "location_no")
        for col in text_columns:
            if col not in df_atomic.columns:
                df_atomic[col] = ""
            df_atomic[col] = df_atomic[col].map(_normalize_text)

        standard_entity_labels = {
            "gangs": "Gang",
            "gang": "Gang",
            "section incharges": "Section Incharge",
            "section incharge": "Section Incharge",
            "section in-charge": "Section Incharge",
            "supervisors": "Supervisor",
            "supervisor": "Supervisor",
        }
        df_atomic["entity_type"] = df_atomic["entity_type"].map(
            lambda val: standard_entity_labels.get(val.lower(), val) if val else val
        )

        numeric_columns = {
            "revenue_planned": 0.0,
            "revenue_realised": 0.0,
            "tower_weight": 0.0,
        }
        for col, default in numeric_columns.items():
            if col not in df_atomic.columns:
                df_atomic[col] = default
            df_atomic[col] = pd.to_numeric(df_atomic[col], errors="coerce").fillna(default)

        df_atomic["project_key_lc"] = df_atomic["project_key"].str.lower()
        df_atomic["project_name_lc"] = df_atomic["project_name"].str.lower()
        df_atomic["entity_type_lc"] = df_atomic["entity_type"].str.lower()
        df_atomic["location_no_norm"] = df_atomic["location_no"].map(_normalize_location)

        if (not has_plan_provider.get(plan_key)) and workbook is not None:
            daily_sheet = None
            for candidate in ("ProdDailyExpandedSingles",):
                if candidate in workbook.sheet_names:
                    daily_sheet = candidate
                    break

            if daily_sheet:
                try:
                    df_daily = pd.read_excel(workbook, sheet_name=daily_sheet, usecols=None)
                except Exception as exc:
                    LOGGER.warning("Failed to load daily sheet '%s': %s", daily_sheet, exc)
                else:
                    def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
                        mapping = {str(col).strip().lower(): col for col in frame.columns}
                        for candidate in candidates:
                            key = candidate.strip().lower()
                            if key in mapping:
                                return mapping[key]
                        for key, original in mapping.items():
                            if any(cand.lower() in key for cand in candidates):
                                return original
                        raise KeyError(candidates)

                    try:
                        col_proj = _pick_column(df_daily, ("project_name", "project"))
                        col_loc = _pick_column(
                            df_daily, ("location_no", "location number", "location")
                        )
                    except KeyError:
                        LOGGER.warning(
                            "Daily sheet missing project/location columns; delivered will rely on realised values only."
                        )
                    else:
                        cleaned_projects = df_daily[col_proj].map(_normalize_lower)
                        cleaned_locations = df_daily[col_loc].map(_normalize_location)
                        completed_keys = {
                            (p, loc) for p, loc in zip(cleaned_projects, cleaned_locations) if p and loc
                        }

        df_project = df_atomic[df_atomic["project_key_lc"] == project_value.lower()].copy()
        if df_project.empty:
            df_project = df_atomic[df_atomic["project_name_lc"] == project_value.lower()].copy()
        if df_project.empty:
            return _empty_response(f"Selected project not found in {plan_noun} data.")

        if stringing_length_label:
            df_project["__stringing_length_km"] = _stringing_length_km_series(df_project)

        if not active_months:
            return _empty_response(f"Select a month to view the {plan_noun.lower()}.")

        if active_months:
            month_mask = df_project["completion_month"].isin(active_months)
            if not month_mask.any():
                label = _format_period_label(active_months)
                label_clean = label.strip("()") if label else "selected month"
                return _empty_response(f"{plan_noun} for {label_clean} is not available.")
            df_project = df_project.loc[month_mask].copy()

        entity_lc = entity_value.lower()
        df_entity = df_project[df_project["entity_type_lc"] == entity_lc].copy()

        if df_entity.empty:
            return _empty_response("No plan entries found for the selected filters.")

        df_entity["is_completed"] = [
            (proj, loc) in completed_keys
            for proj, loc in zip(df_entity["project_name_lc"], df_entity["location_no_norm"])
        ]

        df_entity["delivered_revenue"] = np.where(
            df_entity["revenue_realised"] > 0,
            df_entity["revenue_realised"],
            np.where(df_entity["is_completed"], df_entity["revenue_planned"], 0.0),
        )
        if stringing_length_label:
            length_series = pd.to_numeric(
                df_entity.get("__stringing_length_km"), errors="coerce"
            ).fillna(0.0)
            df_entity["tower_weight"] = length_series
            df_entity["delivered_tower_weight"] = np.where(
                df_entity["is_completed"], length_series, 0.0
            )
        else:
            df_entity["delivered_tower_weight"] = np.where(
                df_entity["is_completed"], df_entity["tower_weight"], 0.0
            )
        if stringing_length_label:
            df_entity["tower_weight"] = (
                pd.to_numeric(df_entity["tower_weight"], errors="coerce").fillna(0.0) / 1000.0
            )
            df_entity["delivered_tower_weight"] = (
                pd.to_numeric(df_entity["delivered_tower_weight"], errors="coerce").fillna(0.0) / 1000.0
            )
        if stringing_length_label:
            df_entity["tower_weight"] = (
                pd.to_numeric(df_entity["tower_weight"], errors="coerce").fillna(0.0) / 1000.0
            )
            df_entity["delivered_tower_weight"] = (
                pd.to_numeric(df_entity["delivered_tower_weight"], errors="coerce").fillna(0.0) / 1000.0
            )
        df_entity = df_entity[df_entity["entity_name"].astype(bool)].copy()
        if df_entity.empty:
            return _empty_response("No plan entries found for the selected filters.")

        aggregated = (
            df_entity.groupby("entity_name", as_index=False)[
                [
                    "revenue_planned",
                    "delivered_revenue",
                    "tower_weight",
                    "delivered_tower_weight",
                ]
            ]
            .sum()
        )
        aggregated = aggregated.rename(columns={"revenue_planned": "revenue"})
        target_metric_col = "revenue_planned" if metric_value == "revenue" else "tower_weight"
        delivered_metric_col = "delivered_revenue" if metric_value == "revenue" else "delivered_tower_weight"

        def _collect_locations(values: pd.Series) -> list[str]:
            seen: set[str] = set()
            ordered: list[str] = []
            for raw in values:
                if pd.isna(raw):
                    continue
                text = str(raw).strip()
                if not text or text.lower() in {"nan", "none"}:
                    continue
                if text not in seen:
                    seen.add(text)
                    ordered.append(text)
            return ordered

        def _ensure_location_list(value: object) -> list[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, (tuple, set)):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(",") if part.strip()]
                return parts
            return []

        filtered_target = df_entity[df_entity[target_metric_col] > 0]
        if filtered_target.empty:
            filtered_target = df_entity

        target_locations = (
            filtered_target.groupby("entity_name")["location_no"]
            .agg(_collect_locations)
        )
        filtered_delivered = df_entity[df_entity[delivered_metric_col] > 0]

        delivered_locations = (
            filtered_delivered.groupby("entity_name")["location_no"]
            .agg(_collect_locations)
        )

        aggregated = aggregated.merge(
            target_locations.rename("target_locations"),
            on="entity_name",
            how="left",
        )
        aggregated = aggregated.merge(
            delivered_locations.rename("delivered_locations"),
            on="entity_name",
            how="left",
        )

        aggregated["target_locations"] = aggregated["target_locations"].apply(_ensure_location_list)
        aggregated["delivered_locations"] = aggregated["delivered_locations"].apply(_ensure_location_list)

        if aggregated.empty:
            return _empty_response("No plan entries found for the selected filters.")

        aggregated["delivered_value"] = np.where(
            metric_value == "revenue",
            aggregated["delivered_revenue"],
            aggregated["delivered_tower_weight"],
        )

        axis_override = "Length (KM)" if stringing_length_label else None
        unit_override = "KM" if stringing_length_label else None

        fig = build_responsibilities_chart(
            aggregated,
            entity_label=entity_value,
            metric=metric_value,
            axis_title_override=axis_override,
            unit_label_override=unit_override,
            title=None,
            top_n=20,
        )

        if metric_value == "revenue":
            total_target = float(aggregated["revenue"].sum())
            total_delivered = float(aggregated["delivered_revenue"].sum())
        else:
            total_target = float(aggregated["tower_weight"].sum())
            total_delivered = float(aggregated["delivered_tower_weight"].sum())

        achievement = 0.0 if total_target == 0 else (total_delivered / total_target) * 100.0

        def fmt_num(value: float) -> str:
            if metric_value == "revenue":
                return f"\u20b9{value:,.0f}"
            unit = "KM" if stringing_length_label else "MT"
            precision = 1 if stringing_length_label else 0
            return f"{value:,.{precision}f} {unit}"

        kpi_target_txt = fmt_num(total_target)
        kpi_deliv_txt = fmt_num(total_delivered)
        kpi_ach_txt = f"{achievement:.0f}%"

        return fig, kpi_target_txt, kpi_deliv_txt, kpi_ach_txt
    @app.callback(
        Output("g-responsibilities", "figure"),
        Output("kpi-resp-target-value", "children"),
        Output("kpi-resp-delivered-value", "children"),
        Output("kpi-resp-ach-value", "children"),
        Input("f-project", "value"),
        Input("f-resp-entity", "value"),
        Input("f-resp-metric", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def update_monthly_plan_erection(
        project_value: str | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
    ):
        return _render_monthly_plan_card(
            plan_mode="erection",
            project_value=project_value,
            entity_value=entity_value,
            metric_value=metric_value,
            months_value=months_value,
            quick_range_value=quick_range_value,
        )

    @app.callback(
        Output("g-stringing-plan", "figure"),
        Output("kpi-stringing-plan-target", "children"),
        Output("kpi-stringing-plan-delivered", "children"),
        Output("kpi-stringing-plan-ach", "children"),
        Input("f-project", "value"),
        Input("f-stringing-plan-entity", "value"),
        Input("f-stringing-plan-metric", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def update_monthly_plan_stringing(
        project_value: str | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
    ):
        return _render_monthly_plan_card(
            plan_mode="stringing",
            project_value=project_value,
            entity_value=entity_value,
            metric_value=metric_value,
            months_value=months_value,
            quick_range_value=quick_range_value,
        )



    def _compute_dashboard_outputs(
        scope_meta: dict[str, Any] | None,
        topbot_metric: str | None,
        *,
        avp_namespace: str = "avp",
    ) -> tuple:
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        selected = scope_meta.get("selected") or {}
        project_list = selected.get("projects", [])
        gang_list = selected.get("gangs", [])
        months_ts = _months_from_meta(scope_meta)
        days_factor = float(scope_meta.get("days_factor") or 30.0)
        eff_mode = _normalize_mode(scope_meta.get("mode"))
        meta_signature = scope_meta.get("signature") or "nosig"

        scoped = _scope_frame_from_store(scope_meta, "full").copy()
        scoped_top_bottom = _scope_frame_from_store(scope_meta, "project").copy()
        scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()
        scope_keys = scope_meta.get("scopes") or {}
        project_gang_key = scope_keys.get("project_gang")
        empty_loss_columns = [
            "gang_name",
            "delivered",
            "lost",
            "potential",
            "avg_prod",
            "baseline",
            "efficiency_pct",
            "total_mt",
        ]
        loss_df = pd.DataFrame(columns=empty_loss_columns)

        is_stringing = eff_mode == "stringing"
        metric_col = "daily_km" if is_stringing else "daily_prod_mt"
        unit_short = "KM" if is_stringing else "MT"

        if is_stringing:
            benchmark = BENCHMARK_KM_PER_MONTH
            if not scoped.empty and metric_col in scoped.columns:
                monthly_totals = (
                    scoped.groupby(["gang_name", "month"], dropna=True)[metric_col]
                    .sum()
                    .reset_index(name="monthly_value")
                )
                avg_prod = float(monthly_totals["monthly_value"].mean()) if not monthly_totals.empty else 0.0
            else:
                avg_prod = 0.0
            delta_pct = (avg_prod - benchmark) / benchmark * 100 if benchmark else None
            kpi_avg = f"{avg_prod:.2f} KM/month"
            kpi_delta = "(n/a)" if delta_pct is None else f"({delta_pct:+.0f}% vs {benchmark:.1f} KM/month)"
            project_bench = BENCHMARK_KM_PER_MONTH
            avg_line_for_project = (
                scoped[metric_col].mean() if len(scoped) and (metric_col in scoped.columns) else 0.0
            )
        else:
            benchmark = BENCHMARK_MT_PER_DAY
            avg_prod = scoped[metric_col].mean() if len(scoped) and (metric_col in scoped.columns) else 0.0
            delta_pct = (avg_prod - benchmark) / benchmark * 100 if benchmark else None
            kpi_avg = f"{avg_prod:.2f} {unit_short}"
            kpi_delta = "(n/a)" if delta_pct is None else f"({delta_pct:+.0f}% vs {benchmark:.1f} {unit_short})"
            project_bench = benchmark
            avg_line_for_project = avg_prod

        has_selected_months = bool(months_ts)
        baseline_map: dict[str, float] = {}
        baseline_monthly_map: dict[str, dict[pd.Timestamp, float]] = {}

        if "gang_name" not in scoped_all.columns:
            scoped_all["gang_name"] = pd.Series(dtype=str)
        if "project_name" not in scoped_all.columns:
            scoped_all["project_name"] = pd.Series(dtype=str)

        earliest_month = None
        if has_selected_months and not scoped_all.empty and "month" in scoped_all.columns:
            month_values = sorted(set(months_ts))
            period_mask = scoped_all["month"].isin(month_values)
            loss_scope = scoped_all.loc[period_mask].copy()
            earliest_month = month_values[0] if month_values else None
            history_scope = scoped_all.loc[scoped_all["month"] < (earliest_month or pd.Timestamp.max)].copy()
        else:
            loss_scope = scoped_all.copy()
            history_scope = scoped_all.copy()
        loss_scope = loss_scope.copy()
        if not loss_scope.empty:
            if "gang_name" in loss_scope.columns:
                loss_scope = loss_scope.dropna(subset=["gang_name"])
                loss_scope["gang_name"] = loss_scope["gang_name"].astype(str).str.strip()
            if "project_name" in loss_scope.columns:
                loss_scope["project_name"] = loss_scope["project_name"].astype(str).str.strip()

        # --- PROJECT-level baselines, then map them onto gangs ---
        precomputed_overall, precomputed_monthly = _get_project_baselines()
        use_precomputed = (not is_stringing) and bool(precomputed_overall)
        proj_overall_all: dict[str, float] = {}
        proj_monthly: dict[str, dict[pd.Timestamp, float]] = {}

        if use_precomputed:
            if "project_name" in scoped_all.columns:
                available_projects = (
                    scoped_all["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                available_projects = []
            if available_projects:
                proj_overall_all = {
                    project: precomputed_overall.get(project)
                    for project in available_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in available_projects
                }
            else:
                proj_overall_all = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if has_selected_months and earliest_month is not None:
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < earliest_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < earliest_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token_all = f"project-baseline::{metric_col}::{meta_signature}"

            def _compute_baseline_all() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if scoped_all.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(scoped_all, metric_col)
                return compute_project_baseline_maps(scoped_all)

            proj_overall_all, proj_monthly_all = _cached_scope_result(
                project_gang_key,
                baseline_token_all,
                _compute_baseline_all,
                clone=_clone_baseline_result,
            )

            history_key = (
                f"{baseline_token_all}::history::{earliest_month.isoformat()}"
                if earliest_month is not None
                else f"{baseline_token_all}::history::all"
            )

            def _compute_baseline_hist() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if history_scope.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(history_scope, metric_col)
                return compute_project_baseline_maps(history_scope)

            proj_overall_hist, proj_monthly_hist = _cached_scope_result(
                project_gang_key,
                history_key,
                _compute_baseline_hist,
                clone=_clone_baseline_result,
            )
            if proj_overall_hist:
                proj_overall_all.update(proj_overall_hist)
            proj_monthly = proj_monthly_hist

        # Gang <-> Project bridge
        gang_to_project = (
            scoped_all[["gang_name", "project_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("gang_name")["project_name"]
            .astype(str)
            .to_dict()
        )

        baseline_overall_map = {g: proj_overall_all.get(p) for g, p in gang_to_project.items()}
        baseline_monthly_map = {g: proj_monthly.get(p, {}) for g, p in gang_to_project.items()}
        baseline_map = baseline_overall_map or {}

        loss_token = f"loss::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}"

        def _compute_loss_rows() -> list[dict[str, float]]:
            rows: list[dict[str, float]] = []
            for gang_name, gang_df in loss_scope.groupby("gang_name"):
                if gang_df.empty:
                    continue
                overall_baseline = baseline_overall_map.get(gang_name)
                if is_stringing:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss_for_column(
                        gang_df,
                        metric_column=metric_col,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                    )
                else:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                        gang_df,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_mt_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                    )
                rows.append(
                    {
                        "gang_name": gang_name,
                        "delivered": delivered,
                        "lost": loss_mt,
                        "potential": potential,
                        "avg_prod": (gang_df[metric_col].mean() if metric_col in gang_df.columns else 0.0),
                        "baseline": baseline,
                    }
                )
            return rows

        loss_rows = _cached_scope_result(
            project_gang_key,
            loss_token,
            _compute_loss_rows,
            clone=_clone_loss_rows,
        )

        if loss_rows:
            loss_df = pd.DataFrame(loss_rows)
            if is_stringing and not loss_df.empty:
                loss_df["avg_prod"] = loss_df["avg_prod"].astype(float) * days_factor
                loss_df["baseline"] = loss_df["baseline"].astype(float) * days_factor

            deliv = loss_df["delivered"].astype(float)
            lost = loss_df["lost"].astype(float)
            potential = loss_df["potential"].astype(float)
            sum_series = deliv.add(lost)
            use_sum = deliv.notna() & lost.notna()
            potential_fallback = potential.where(potential.notna(), sum_series)
            total_series = pd.Series(
                np.where(use_sum, sum_series, potential_fallback),
                index=loss_df.index,
            ).fillna(0.0)
            efficiency_series = np.where(
                total_series > 0.0,
                (deliv.fillna(0.0) / total_series) * 100.0,
                0.0,
            )
            loss_df = (
                loss_df.assign(
                    efficiency_pct=efficiency_series,
                    total_mt=total_series,
                )
                .sort_values("efficiency_pct", ascending=True)
                .reset_index(drop=True)
            )
        else:
            loss_df = loss_df.copy()
        # --- meta for hover: last project & last worked date per gang (within current filters)
        meta_ready = {"gang_name", "project_name", "date"}.issubset(scoped_all.columns) and not scoped_all.empty

        if meta_ready:
            idx_last = (
                scoped_all.sort_values("date")
                .groupby("gang_name", observed=True)["date"]
                .idxmax()
            )
            base_cols = ["gang_name", "project_name", "date"]
            meta = (
                scoped_all.loc[idx_last, base_cols]
                .rename(columns={"project_name": "last_project", "date": "last_date"})
            )
            # Attach stringing meta if present at the last row per gang
            extra_cols = ["from_ap", "to_ap", "method", "po_id", "status"]
            present_extras = [c for c in extra_cols if c in scoped_all.columns]
            if present_extras:
                extras = scoped_all.loc[idx_last, ["gang_name", *present_extras]].copy()
                meta = meta.merge(extras, on="gang_name", how="left")
            loss_df = loss_df.merge(meta, on="gang_name", how="left")
        else:
            # guarantee columns exist even when we couldn't compute meta
            loss_df = loss_df.assign(last_project=np.nan, last_date=pd.NaT)
        

        # pretty, null-safe strings for hover (NO KeyError even if meta missing)
        last_date_series = pd.to_datetime(loss_df.get("last_date"), errors="coerce")
        loss_df["last_date_str"] = last_date_series.dt.strftime("%d-%b-%Y").fillna("")
        loss_df["last_project"]  = loss_df.get("last_project").fillna("")

        # Build left-card HTML list from loss_df (now that meta is attached)

        avp_children = []

        if not loss_df.empty:

            for _, r in loss_df.iterrows():

                total = float(r.get("total_mt", 0.0))

                if total == 0.0:

                    base_total = (

                        r["delivered"] + r["lost"]

                        if pd.notna(r["delivered"]) and pd.notna(r["lost"])

                        else r["potential"]

                    )

                    total = float(base_total) if pd.notna(base_total) else 0.0

                pct = r.get("efficiency_pct", np.nan)

                pct = float(pct) if pd.notna(pct) else 0.0

                if total > 0.0 and pct == 0.0:

                    pct = (100.0 * float(r["delivered"]) / total)
                rate_label = "KM/month" if is_stringing else f"{unit_short}/day"
                unit_total = "KM" if is_stringing else unit_short
                avp_children.append(

                    _render_avp_row(

                        r["gang_name"], float(r["delivered"]), float(r["lost"]),

                        total, pct,

                        avg_prod=float(r.get("avg_prod", 0.0)),

                        baseline=float(r.get("baseline", 0.0)),

                        last_project=str(r.get("last_project", "\uFFFD")),

                        last_date=str(r.get("last_date_str", "\uFFFD")),
                        rate_label=rate_label,
                        unit_total=unit_total,
                        namespace=avp_namespace,
                    )

                )
        else:
            avp_children.append(
                html.Div(
                    "No gang performance data for the current filters.",
                    className="text-muted small px-2 py-3",
                )
            )

        row_px = 56
        topbot_margin = 120
        fig_height = int(row_px * max(1, len(loss_df)) + topbot_margin)

        active_gangs = loss_scope["gang_name"].nunique()
        # totals and units by mode
        total_metric = float(loss_scope[metric_col].sum()) if (not loss_scope.empty and metric_col in loss_scope.columns) else 0.0
        total_delivered = float(loss_df["delivered"].sum()) if not loss_df.empty else 0.0
        total_lost = float(loss_df["lost"].sum()) if not loss_df.empty else 0.0
        total_potential = total_delivered + total_lost
        lost_pct = (total_lost / total_potential * 100) if total_potential > 0 else 0.0

        kpi_active = f"{active_gangs}"
        kpi_total = f"{total_metric:.1f} {unit_short}"
        # Secondary planned layer from Micro Plan (erection mode only)
        kpi_total_planned = ""
        kpi_total_nos_planned = ""
        if not is_stringing:
            try:
                active_months = sorted({ts for ts in months_ts if pd.notna(ts)})
                if active_months and has_plan_provider.get("erection"):
                    resp_df, _, _, _ = _fetch_monthly_plan("erection")
                    if isinstance(resp_df, pd.DataFrame) and not resp_df.empty:
                        df_mp = resp_df.copy()
                        # completion month (use folder-derived plan_month when available)
                        if "plan_month" in df_mp.columns:
                            df_mp["plan_month"] = pd.to_datetime(
                                df_mp["plan_month"], errors="coerce"
                            ).dt.to_period("M").dt.to_timestamp()
                            df_mp["completion_month"] = df_mp["plan_month"]
                        elif 'completion_date' in df_mp.columns:
                            df_mp['completion_month'] = pd.to_datetime(df_mp['completion_date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
                        else:
                            df_mp['completion_month'] = pd.NaT
                        # normalize project + location
                        def _norm_txt(x):
                            s = "" if x is None else str(x).replace("\u00a0", " ").strip()
                            return "" if s.lower() in {"", "nan", "none", "null"} else s
                        def _norm_lc(x):
                            return _norm_txt(x).lower()
                        def _norm_loc(x):
                            t = _norm_txt(x)
                            if not t:
                                return ""
                            if t.endswith('.0') and t.replace('.', '', 1).isdigit():
                                t = t.split('.', 1)[0]
                            return t
                        for c in ("project_name", "project_key", "location_no"):
                            if c not in df_mp.columns:
                                df_mp[c] = ""
                            df_mp[c] = df_mp[c].map(_norm_txt)
                        df_mp['project_name_lc'] = df_mp['project_name'].map(_norm_lc)
                        df_mp['project_key_lc'] = df_mp['project_key'].map(_norm_lc)
                        df_mp['location_no_norm'] = df_mp['location_no'].map(_norm_loc)
                        # filter by selected projects present in current scope
                        if "project_name" in scoped_all.columns and not scoped_all.empty:
                            sel_projects = set(scoped_all["project_name"].dropna().astype(str).str.strip().str.lower())
                        else:
                            sel_projects = set()
                        if sel_projects:
                            mask_project = df_mp['project_name_lc'].isin(sel_projects) | df_mp['project_key_lc'].isin(sel_projects)
                            df_mp = df_mp.loc[mask_project].copy()
                        # filter by selected months
                        df_mp = df_mp[df_mp['completion_month'].isin(active_months)].copy()
                        if not df_mp.empty:
                            dedup_cols = ["project_name_lc", "location_no_norm"]
                            valid_locations = df_mp.dropna(subset=dedup_cols).copy()
                            if "tower_weight" in valid_locations.columns:
                                valid_locations["tower_weight"] = (
                                    pd.to_numeric(valid_locations["tower_weight"], errors="coerce").fillna(0.0)
                                )
                            dedup_locations = (
                                valid_locations.sort_values(dedup_cols)
                                .drop_duplicates(subset=dedup_cols, keep="first")
                            )
                            if "tower_weight" in df_mp.columns:
                                planned_mt = float(dedup_locations.get("tower_weight", 0.0).sum())
                            else:
                                planned_mt = 0.0
                            planned_tower_count = int(dedup_locations.shape[0])
                            kpi_total_planned = f"Planned: {planned_mt:.1f} MT"
                            kpi_total_nos_planned = f"Planned: {planned_tower_count}"
            except Exception:
                # leave planned KPIs blank on any failure
                kpi_total_planned = ""
                kpi_total_nos_planned = ""
        # Compute number of towers erected matching the current scope
        # If no months are selected, include the full available range in scope
        try:
            if not is_stringing:
                if months_ts:
                    range_start = pd.Timestamp(min(months_ts)).normalize()
                    range_end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
                else:
                    # Derive an all-time window from the scoped data
                    if isinstance(loss_scope, pd.DataFrame) and not loss_scope.empty:
                        comp_series = pd.to_datetime(loss_scope.get("completion_date"), errors="coerce")
                        comp_series = comp_series[comp_series.notna()]
                        if len(comp_series):
                            range_start = pd.Timestamp(comp_series.min()).normalize()
                            range_end = pd.Timestamp(comp_series.max()).normalize()
                        else:
                            date_series = pd.to_datetime(loss_scope.get("date"), errors="coerce")
                            date_series = date_series[date_series.notna()]
                            if len(date_series):
                                range_start = pd.Timestamp(date_series.min()).normalize()
                                range_end = pd.Timestamp(date_series.max()).normalize()
                            else:
                                # Fallback to current month if no dates are available
                                today = pd.Timestamp.today().normalize()
                                range_start = today.to_period("M").start_time.normalize()
                                range_end = (today + pd.offsets.MonthEnd(0)).normalize()
                    else:
                        today = pd.Timestamp.today().normalize()
                        range_start = today.to_period("M").start_time.normalize()
                        range_end = (today + pd.offsets.MonthEnd(0)).normalize()

                export_df, _ = _prepare_erections_completed(
                    loss_scope,
                    range_start=range_start,
                    range_end=range_end,
                    responsibilities_provider=None,
                    search_text=None,
                )
                tower_count = int(len(export_df)) if isinstance(export_df, pd.DataFrame) else 0
                kpi_total_nos = f"{tower_count}"
            else:
                kpi_total_nos = ""
        except Exception:
            kpi_total_nos = ""
        kpi_loss = f"{total_lost:.1f} {unit_short}"
        kpi_loss_delta = f"{lost_pct:.1f}%"




        fig_loss = go.Figure()
        if not loss_df.empty:
            # Determine if stringing extra fields are available for hover
            has_stringing_meta = all(c in loss_df.columns for c in ["from_ap", "to_ap", "method", "po_id", "status"]) if is_stringing else False
            hover_extra = (
                "From AP: %{customdata[4]}<br>"
                "To AP: %{customdata[5]}<br>"
                "Method: %{customdata[6]}<br>"
                "PO: %{customdata[7]}<br>"
                "Status: %{customdata[8]}<br>"
            ) if has_stringing_meta else ""
            # --- Delivered bar (replace the existing fig_loss.add_bar block) ---
            fig_loss.add_bar(
                x=loss_df["delivered"],
                y=loss_df["gang_name"],
                orientation="h",
                marker_color="green",
                text=loss_df["delivered"].round(1),
                textposition="inside",
                name="Delivered",
                width=0.95,
                # match Top/Bottom customdata shape: [last_project, last_date_str, current_metric, baseline_metric]
                customdata=(
                    np.stack(
                        [
                            loss_df["last_project"].fillna(" "),
                            loss_df["last_date_str"].fillna(" "),
                            loss_df["avg_prod"].fillna(0.0),      # current metric
                            loss_df["baseline"].fillna(0.0),      # baseline
                            *(
                                [
                                    loss_df.get("from_ap", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("to_ap", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("method", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("po_id", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("status", pd.Series([" "] * len(loss_df))).fillna(" "),
                                ] if has_stringing_meta else []
                            ),
                        ],
                        axis=-1,
                    )
                ),
                hovertemplate=(
                    "%{y}<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked at: %{customdata[1]}<br>"
                    f"Current {('KM/month' if is_stringing else unit_short + '/day')}: %{{customdata[2]:.2f}}<br>"
                    f"Baseline {('KM/month' if is_stringing else unit_short + '/day')}: %{{customdata[3]:.2f}}<br>"

                    + hover_extra + "<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.95)",
                    font=dict(color="#111827", size=13),
                    bordercolor="rgba(17,24,39,0.15)",
                    align="left",
                    namelength=0,
                ),
            )

            # --- Loss bar (replace the existing fig_loss.add_bar block) ---
            fig_loss.add_bar(
                x=loss_df["lost"],
                y=loss_df["gang_name"],
                orientation="h",
                marker_color="red",
                text=loss_df["lost"].round(1),
                textposition="inside",
                name="Loss",
                base=loss_df["delivered"],
                width=0.95,
                customdata=(
                    np.stack(
                        [
                            loss_df["last_project"].fillna(" "),
                            loss_df["last_date_str"].fillna(" "),
                            loss_df["avg_prod"].fillna(0.0),
                            loss_df["baseline"].fillna(0.0),
                            *(
                                [
                                    loss_df.get("from_ap", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("to_ap", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("method", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("po_id", pd.Series([" "] * len(loss_df))).fillna(" "),
                                    loss_df.get("status", pd.Series([" "] * len(loss_df))).fillna(" "),
                                ] if has_stringing_meta else []
                            ),
                        ],
                        axis=-1,
                    )
                ),
                hovertemplate=(
                    "%{y}<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked at: %{customdata[1]}<br>"
                    f"Current {('KM/month' if is_stringing else unit_short + '/day')}: %{{customdata[2]:.2f}}<br>"
                    f"Baseline {('KM/month' if is_stringing else unit_short + '/day')}: %{{customdata[3]:.2f}}<br>"

                    + hover_extra + "<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.95)",
                    font=dict(color="#111827", size=13),
                    bordercolor="rgba(17,24,39,0.15)",
                    align="left",
                    namelength=0,
                ),
            )

            for _, row in loss_df.iterrows():
                fig_loss.add_annotation(
                    x=row["potential"],
                    y=row["gang_name"],
                    text=(
                        f"{row['avg_prod']:.2f} "
                        f"{'KM/month' if is_stringing else unit_short + '/day'} "
                        f"(Baseline: {row['baseline']:.2f} "
                        f"{'KM/month' if is_stringing else unit_short + '/day'})"
                    ),
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10, color="black"),
                )
        fig_loss.update_layout(
            barmode="stack",
            bargap=0.02,
            height=fig_height,
            margin=dict(l=140, r=120, t=30, b=30),
            xaxis_title=f"Potential ({unit_short})",
            yaxis_title="Gang",
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
            dragmode=False,
        )
        fig_loss.update_layout(hovermode="closest", clickmode="event+select")
        fig_loss.update_xaxes(showspikes=False, fixedrange=True)
        fig_loss.update_yaxes(showspikes=False, fixedrange=True, type="category")
        
        # fig_monthly = create_monthly_line_chart(scoped, bench=benchmark)
        charts_scope = scoped_top_bottom.copy()
        if not charts_scope.empty and "gang_name" in charts_scope.columns:
            charts_scope = charts_scope.dropna(subset=["gang_name"])
            charts_scope["gang_name"] = charts_scope["gang_name"].astype(str).str.strip()
        projects_scope = data_selector.select(eff_mode)
        if isinstance(projects_scope, pd.DataFrame):
            projects_scope = projects_scope.copy()
            if is_stringing:
                projects_scope = _stringing_scope(
                    projects_scope,
                    selected.get("kv"),
                    selected.get("methods"),
                )
        else:
            projects_scope = pd.DataFrame()
        if is_stringing:
            if "daily_km" in charts_scope.columns:
                charts_scope["daily_prod_mt"] = charts_scope["daily_km"]
            if "daily_km" in projects_scope.columns:
                projects_scope["daily_prod_mt"] = projects_scope["daily_km"]

            # Convert Top/Bottom input to per-month values (KM/month)
            try:
                monthly_cur = (
                    charts_scope.groupby(["gang_name", "month"], dropna=True)["daily_prod_mt"].sum().reset_index()
                )
                monthly_cur = monthly_cur.groupby("gang_name")["daily_prod_mt"].mean().reset_index()
                monthly_cur = monthly_cur.rename(columns={"daily_prod_mt": "monthly_value"})
                charts_scope = monthly_cur.rename(columns={"monthly_value": "daily_prod_mt"})
            except Exception:
                pass

            # Scale baseline map to KM/month using average days in selected months (fallback 30)
            if baseline_map:
                baseline_map = {g: (float(v) * days_factor if v is not None and not pd.isna(v) else 0.0) for g, v in baseline_map.items()}

        fig_top5, fig_bottom5 = create_top_bottom_gangs_charts(
            charts_scope, metric=(topbot_metric or "prod"), baseline_map=baseline_map
        )
        fig_project = create_project_lines_chart(
            projects_scope,
            selected_projects=project_list or None,
            bench=project_bench,
            avg_line=avg_line_for_project,   # Average line remains per-day for project chart
        )

        # If in stringing mode, adapt figure labels/annotations to KM/month units
        if is_stringing:
            # Top/Bottom charts: replace MT/day -> KM/month and MT -> KM in hover and y-axis title
            # Build extras map from last activity rows per gang if available
            extras_map: dict[str, tuple[str, str, str, str, str]] = {}
            try:
                if meta_ready:
                    # meta_ready computed earlier for loss chart on scoped_all
                    # reuse scoped_all last-row index and columns if present
                    idx_last_tb = charts_scope.sort_values("date").groupby("gang_name")["date"].idxmax()
                    needed = ["gang_name", "from_ap", "to_ap", "method", "po_id", "status"]
                    if all(c in charts_scope.columns for c in needed):
                        subset = charts_scope.loc[idx_last_tb, needed].copy()
                        for _, row in subset.iterrows():
                            extras_map[str(row["gang_name"])]=(
                                str(row.get("from_ap", " ") or " "),
                                str(row.get("to_ap", " ") or " "),
                                str(row.get("method", " ") or " "),
                                str(row.get("po_id", " ") or " "),
                                str(row.get("status", " ") or " "),
                            )
            except Exception:
                extras_map = {}

            for fig in (fig_top5, fig_bottom5):
                try:
                    ytitle = fig.layout.yaxis.title.text or ""
                    if ytitle:
                        fig.update_yaxes(title_text=ytitle.replace("MT/day", "KM/month").replace("MT", "KM"))
                except Exception:
                    pass
                try:
                    for tr in fig.data:
                        # augment hovertemplate and customdata with extras if available
                        if extras_map and hasattr(tr, "customdata") and isinstance(tr.customdata, (list, tuple, np.ndarray)):
                            xcats = list(tr.x) if hasattr(tr, "x") else []
                            if xcats:
                                new_cd = []
                                for i, gname in enumerate(xcats):
                                    base = list(tr.customdata[i]) if isinstance(tr.customdata, (list, tuple, np.ndarray)) else []
                                    extra = list(extras_map.get(str(gname), (" ", " ", " ", " ", " ")))
                                    new_cd.append(base + extra)
                                tr.customdata = np.array(new_cd)
                            # extend hovertemplate
                        if hasattr(tr, "hovertemplate") and isinstance(tr.hovertemplate, str):
                            extra_ht = ("<br>From AP: %{customdata[4]}<br>To AP: %{customdata[5]}<br>Method: %{customdata[6]}<br>PO: %{customdata[7]}<br>Status: %{customdata[8]}")
                            if extra_ht not in tr.hovertemplate:
                                tr.hovertemplate = tr.hovertemplate.replace("<extra>", f"{extra_ht}<extra>")
                        if hasattr(tr, "hovertemplate") and isinstance(tr.hovertemplate, str):
                            tr.hovertemplate = tr.hovertemplate.replace(" MT/day", " KM/month").replace(" MT", " KM")
                except Exception:
                    pass
            # Projects-over-months chart: replace MT labels in axis + annotations
            try:
                ytitle = fig_project.layout.yaxis.title.text or ""
                if ytitle:
                    fig_project.update_yaxes(title_text=ytitle.replace("(MT)", "(KM)"))
                annots = list(getattr(fig_project.layout, "annotations", []) or [])
                if annots:
                    for a in annots:
                        if hasattr(a, "text") and isinstance(a.text, str):
                            a.text = a.text.replace(" MT/day", " KM/month")
                    fig_project.update_layout(annotations=annots)
            except Exception:
                pass

        return (
            kpi_avg,
            kpi_delta,
            kpi_active,
            kpi_total,
            kpi_total_planned,
            kpi_total_nos,
            kpi_total_nos_planned,
            kpi_loss,
            kpi_loss_delta,
            avp_children,
            fig_loss,   # kept but hidden in layout to preserve clickData wiring
            # fig_monthly,
            fig_top5,
            fig_bottom5,
            fig_project,
        )

    def _resolve_focus_mode(
        focus_payload: dict[str, Any] | None,
        fallback_mode: str | None,
    ) -> str:
        """
        Prefer the project tile's mode (erection/stringing) when building modal scopes,
        but gracefully fall back to the global mode store when the tile payload lacks it.
        """
        if isinstance(focus_payload, dict):
            focus_mode = focus_payload.get("mode")
            if isinstance(focus_mode, str) and focus_mode.strip():
                return _normalize_mode(focus_mode)
        return _normalize_mode(fallback_mode)

    def _modal_mode_from_store(payload: Any, fallback: str | None = None) -> str:
        raw: str | None
        if isinstance(payload, dict):
            raw = payload.get("mode")
        else:
            raw = payload
        text = ""
        if isinstance(raw, str):
            text = raw.strip()
        elif raw is not None:
            text = str(raw).strip()
        if "|" in text:
            text = text.split("|", 1)[0]
        if text:
            return _normalize_mode(text)
        return _normalize_mode(fallback)

    def _compose_modal_mode_payload(mode_value: str) -> str:
        mode_text = _normalize_mode(mode_value)
        millis = int(time.time() * 1000)
        return f"{mode_text}|{millis}"

    def _build_project_scope_meta(
        project_name: str,
        project_code: str | None,
        mode_value: str,
        months,
        quick_range,
        gangs,
        kv_values,
        method_values,
    ) -> dict[str, Any]:
        project_candidates = _project_filter_candidates(project_name, project_code)
        project_list = _normalize_str_list(project_candidates)
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        kv_list = _ensure_list(kv_values)
        method_list = _ensure_list(method_values)
        eff_mode = _normalize_mode(mode_value)
        return _build_scope_meta_payload(
            eff_mode=eff_mode,
            project_list=project_list,
            gang_list=gang_list,
            months_list=months_list,
            quick_range=quick_range,
            kv_values=kv_list,
            method_values=method_list,
            kv_list=_normalize_str_list(kv_list),
            method_list=_normalize_str_list(method_list, lower=True),
        )

    @app.callback(
        Output("kpi-avg", "children"),
        Output("kpi-delta", "children"),
        Output("kpi-active", "children"),
        Output("kpi-total", "children"),
        Output("kpi-total-planned", "children"),
        Output("kpi-total-nos", "children"),
        Output("kpi-total-nos-planned", "children"),
        Output("kpi-loss", "children"),
        Output("kpi-loss-delta", "children"),
        Output("avp-list", "children"),
        Output("g-actual-vs-bench", "figure"),
        Output("g-top5", "figure"),
        Output("g-bottom5", "figure"),
        Output("g-projects-over-months", "figure"),
        Input("store-filtered-scope", "data"),
        Input("f-topbot-metric", "value"),
    )
    def update_dashboard(
        scope_meta: dict[str, Any] | None,
        topbot_metric: str | None,
    ) -> tuple:
        return _compute_dashboard_outputs(scope_meta, topbot_metric)
        
    CHART_SOURCES = {"g-actual-vs-bench", "g-top5", "g-bottom5"}

    def _compute_trace_table_payload(
        scope_meta: dict[str, Any], gang_focus: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        selected = scope_meta.get("selected") or {}
        months_ts = _months_from_meta(scope_meta)
        meta_signature = scope_meta.get("signature") or "nosig"
        scope_keys = scope_meta.get("scopes") or {}
        project_gang_key = scope_keys.get("project_gang")

        base_scope = _scope_frame_from_store(scope_meta, "project").copy()
        scoped = _scope_frame_from_store(scope_meta, "full").copy()
        scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()

        is_stringing = _normalize_mode(scope_meta.get("mode")) == "stringing"
        metric_col = "daily_km" if is_stringing else "daily_prod_mt"

        def pick_gang_scope(target_gang: str | None) -> pd.DataFrame:
            if not target_gang:
                return pd.DataFrame()
            subset = base_scope[base_scope["gang_name"] == target_gang]
            if not subset.empty:
                return subset
            fb = scoped_all[scoped_all["gang_name"] == target_gang].copy()
            if months_ts and "month" in fb.columns:
                fb = fb[fb["month"].isin(months_ts)]
            return fb

        baseline_source = scoped_all.copy()
        precomputed_overall, precomputed_monthly = _get_project_baselines()
        use_precomputed = (not is_stringing) and bool(precomputed_overall)
        if use_precomputed:
            if "project_name" in baseline_source.columns:
                candidate_projects = (
                    baseline_source["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                candidate_projects = []
            if candidate_projects:
                proj_overall = {
                    project: precomputed_overall.get(project)
                    for project in candidate_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in candidate_projects
                }
            else:
                proj_overall = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if months_ts:
                cutoff_month = min(months_ts)
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < cutoff_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < cutoff_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token = f"trace-project-baseline::{metric_col}::{is_stringing}::{meta_signature}"

            def _compute_trace_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if baseline_source.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(baseline_source, metric_col)
                return compute_project_baseline_maps(baseline_source)

            proj_overall, proj_monthly = _cached_scope_result(
                project_gang_key,
                baseline_token,
                _compute_trace_baselines,
                clone=_clone_baseline_result,
            )

        if use_precomputed:
            project_overall = proj_overall
            project_monthly = proj_monthly
        else:
            project_overall = proj_overall
            project_monthly = proj_monthly

        if {"gang_name", "project_name"}.issubset(baseline_source.columns):
            g2p = (
                baseline_source[["gang_name", "project_name"]]
                .dropna()
                .drop_duplicates()
                .set_index("gang_name")["project_name"]
                .astype(str)
                .to_dict()
            )
        else:
            g2p = {}

        overall_baseline_map = {g: project_overall.get(p) for g, p in g2p.items()}
        monthly_baseline_map = {g: project_monthly.get(p, {}) for g, p in g2p.items()}

        idle_source = pick_gang_scope(gang_focus)
        if idle_source.empty:
            idle_source = scoped if not scoped.empty else base_scope

        idle_token = f"idle::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}::{gang_focus or '*'}"

        def _compute_idle_df() -> pd.DataFrame:
            if idle_source.empty:
                return pd.DataFrame()
            return compute_idle_intervals_per_gang(
                idle_source,
                loss_max_gap_days=config.loss_max_gap_days,
                baseline_month_lookup=monthly_baseline_map,
                baseline_fallback_map=overall_baseline_map,
            )

        idle_df = _cached_scope_result(
            project_gang_key,
            idle_token,
            _compute_idle_df,
            clone=_clone_dataframe,
        )
        if not idle_df.empty:
            idle_df["interval_loss_mt"] = (
                idle_df["baseline"].astype(float)
                * idle_df["idle_days_capped"].astype(float)
            )
            idle_df["cumulative_loss"] = idle_df.groupby("gang_name")[
                "interval_loss_mt"
            ].cumsum()

            def _fmt_metric(value):
                if pd.isna(value):
                    return ""
                formatted = f"{value:.2f}"
                return formatted.rstrip("0").rstrip(".")

            idle_df = (
                idle_df.assign(
                    interval_start=idle_df["interval_start"].dt.strftime("%d-%m-%Y"),
                    interval_end=idle_df["interval_end"].dt.strftime("%d-%m-%Y"),
                    baseline=idle_df["baseline"].apply(_fmt_metric),
                    cumulative_loss=idle_df["cumulative_loss"].apply(_fmt_metric),
                )
                .drop(columns=["interval_loss_mt"])
            )
        idle_data = idle_df.to_dict("records")

        daily_source = pick_gang_scope(gang_focus)
        if daily_source.empty:
            daily_source = scoped if not scoped.empty else base_scope
        sort_cols = ["gang_name", "date"]
        daily_source = daily_source.sort_values(sort_cols)
        _cols = ["date", "gang_name", metric_col]
        if "project_name" in daily_source.columns:
            _cols.insert(2, "project_name")
        daily_source = daily_source[_cols]
        if not daily_source.empty:
            daily_source = daily_source.assign(
                date=daily_source["date"].dt.strftime("%d-%m-%Y"),
                daily_prod_mt=(
                    pd.to_numeric(daily_source[metric_col], errors="coerce").round(2).map(
                        lambda v: "" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip(".")
                    )
                ),
            )
            if metric_col != "daily_prod_mt":
                daily_source = daily_source.drop(columns=[metric_col])
        daily_data = daily_source.to_dict("records")
        return idle_data, daily_data

    @app.callback(
        Output("tbl-idle-intervals", "data"),
        Output("tbl-daily-prod", "data"),
        Output("modal-tbl-idle-intervals", "data"),
        Output("modal-tbl-daily-prod", "data"),
        Input("store-click-meta", "data"),
        Input("trace-gang", "value"),
        Input("modal-trace-gang", "value"),
        Input("store-filtered-scope", "data"),
        prevent_initial_call=True,
    )
    def update_trace_tables(
        meta,
        trace_gang_value,
        modal_trace_gang_value,
        scope_meta: dict[str, Any] | None,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        dropdown_selection = trace_gang_value or modal_trace_gang_value
        meta_source = meta.get("source") if isinstance(meta, dict) else None
        meta_gang = meta.get("gang") if isinstance(meta, dict) else None
        meta_is_chart = meta_source in CHART_SOURCES and bool(meta_gang)

        if triggered_id == "store-click-meta":
            gang_focus = meta_gang if meta_is_chart else dropdown_selection
        else:
            gang_focus = dropdown_selection or (meta_gang if meta_is_chart else None)

        if not gang_focus or not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        selected = scope_meta.get("selected") or {}
        project_list = selected.get("projects", [])
        gang_list = selected.get("gangs", [])
        months_ts = _months_from_meta(scope_meta)
        meta_signature = scope_meta.get("signature") or "nosig"
        scope_keys = scope_meta.get("scopes") or {}
        project_gang_key = scope_keys.get("project_gang")

        base_scope = _scope_frame_from_store(scope_meta, "project").copy()
        scoped = _scope_frame_from_store(scope_meta, "full").copy()
        scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()

        is_stringing = _normalize_mode(scope_meta.get("mode")) == "stringing"
        metric_col = "daily_km" if is_stringing else "daily_prod_mt"

        def pick_gang_scope(target_gang: str | None) -> pd.DataFrame:
            if not target_gang:
                return pd.DataFrame()
            subset = base_scope[base_scope["gang_name"] == target_gang]
            if not subset.empty:
                return subset
            fb = scoped_all[scoped_all["gang_name"] == target_gang].copy()
            if months_ts and "month" in fb.columns:
                fb = fb[fb["month"].isin(months_ts)]
            return fb

        baseline_source = scoped_all.copy()
        precomputed_overall, precomputed_monthly = _get_project_baselines()
        use_precomputed = (not is_stringing) and bool(precomputed_overall)
        if use_precomputed:
            if "project_name" in baseline_source.columns:
                candidate_projects = (
                    baseline_source["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                candidate_projects = []
            if candidate_projects:
                proj_overall = {
                    project: precomputed_overall.get(project)
                    for project in candidate_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in candidate_projects
                }
            else:
                proj_overall = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if months_ts:
                cutoff_month = min(months_ts)
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < cutoff_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < cutoff_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token = f"trace-project-baseline::{metric_col}::{is_stringing}::{meta_signature}"

            def _compute_trace_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if baseline_source.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(baseline_source, metric_col)
                return compute_project_baseline_maps(baseline_source)

            proj_overall, proj_monthly = _cached_scope_result(
                project_gang_key,
                baseline_token,
                _compute_trace_baselines,
                clone=_clone_baseline_result,
            )

        project_overall = proj_overall
        project_monthly = proj_monthly

        if {"gang_name", "project_name"}.issubset(baseline_source.columns):
            g2p = (
                baseline_source[["gang_name", "project_name"]]
                .dropna()
                .drop_duplicates()
                .set_index("gang_name")["project_name"]
                .astype(str)
                .to_dict()
            )
        else:
            g2p = {}

        overall_baseline_map = {g: project_overall.get(p) for g, p in g2p.items()}
        monthly_baseline_map = {g: project_monthly.get(p, {}) for g, p in g2p.items()}

        idle_source = pick_gang_scope(gang_focus)
        if idle_source.empty:
            idle_source = scoped if not scoped.empty else base_scope

        idle_token = f"idle::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}::{gang_focus or '*'}"

        def _compute_idle_df() -> pd.DataFrame:
            if idle_source.empty:
                return pd.DataFrame()
            return compute_idle_intervals_per_gang(
                idle_source,
                loss_max_gap_days=config.loss_max_gap_days,
                baseline_month_lookup=monthly_baseline_map,
                baseline_fallback_map=overall_baseline_map,
            )

        idle_df = _cached_scope_result(
            project_gang_key,
            idle_token,
            _compute_idle_df,
            clone=_clone_dataframe,
        )
        if not idle_df.empty:
            idle_df["interval_loss_mt"] = (
                idle_df["baseline"].astype(float)
                * idle_df["idle_days_capped"].astype(float)
            )
            idle_df["cumulative_loss"] = idle_df.groupby("gang_name")[
                "interval_loss_mt"
            ].cumsum()

            def _fmt_metric(value):
                if pd.isna(value):
                    return ""
                formatted = f"{value:.2f}"
                return formatted.rstrip("0").rstrip(".")

            idle_df = (
                idle_df.assign(
                    interval_start=idle_df["interval_start"].dt.strftime("%d-%m-%Y"),
                    interval_end=idle_df["interval_end"].dt.strftime("%d-%m-%Y"),
                    baseline=idle_df["baseline"].apply(_fmt_metric),
                    cumulative_loss=idle_df["cumulative_loss"].apply(_fmt_metric),
                )
                .drop(columns=["interval_loss_mt"])
            )
        idle_data = idle_df.to_dict("records")

        daily_source = pick_gang_scope(gang_focus)
        if daily_source.empty:
            daily_source = scoped if not scoped.empty else base_scope
        sort_cols = ["gang_name", "date"]
        daily_source = daily_source.sort_values(sort_cols)
        _cols = ["date", "gang_name", metric_col]
        if "project_name" in daily_source.columns:
            _cols.insert(2, "project_name")
        daily_source = daily_source[_cols]
        if not daily_source.empty:
            daily_source = daily_source.assign(
                date=daily_source["date"].dt.strftime("%d-%m-%Y"),
                daily_prod_mt=(
                    pd.to_numeric(daily_source[metric_col], errors="coerce").round(2).map(
                        lambda v: "" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip(".")
                    )
                ),
            )
            if metric_col != "daily_prod_mt":
                daily_source = daily_source.drop(columns=[metric_col])
        daily_data = daily_source.to_dict("records")
        return idle_data, daily_data, idle_data, daily_data

    @app.callback(
        Output("project-modal-trace-gang", "options"),
        Output("project-modal-trace-gang", "value"),
        Input("store-project-tile-focus", "data"),
        Input("project-modal-selected-gang", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
        Input("store-project-modal-performance-mode", "data"),
        prevent_initial_call=True,
    )
    def _project_modal_trace_dropdown(
        focus_data: dict[str, Any] | None,
        selected_gang: str | None,
        months,
        quick_range,
        gangs,
        kv_values,
        method_values,
        performance_mode,
    ):
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return [], None
        project_code = (focus_data or {}).get("code")
        eff_mode = _modal_mode_from_store(performance_mode, "erection")
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        scope_meta = _build_project_scope_meta(
            project_name,
            project_code,
            eff_mode,
            months,
            quick_range,
            gangs,
            kv_values,
            method_values,
        )
        base_scope = _scope_frame_from_store(scope_meta, "project")
        if base_scope.empty or "gang_name" not in base_scope.columns:
            return [], None
        gangs_series = (
            base_scope["gang_name"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
        )
        gangs_series = sorted({g for g in gangs_series if g})
        options = [{"label": g, "value": g} for g in gangs_series]
        if not options:
            return [], None
        if selected_gang and selected_gang in gangs_series:
            value = selected_gang
        else:
            value = gangs_series[0]
        return options, value

    @app.callback(
        Output("project-modal-tbl-idle-intervals", "data"),
        Output("project-modal-tbl-daily-prod", "data"),
        Input("store-project-modal-click-meta", "data"),
        Input("project-modal-trace-gang", "value"),
        Input("project-modal-selected-gang", "data"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
        Input("store-project-modal-performance-mode", "data"),
        prevent_initial_call=True,
    )
    def _project_modal_trace_tables(
        modal_meta,
        dropdown_value,
        selected_store_gang,
        focus_data: dict[str, Any] | None,
        months,
        quick_range,
        gangs,
        kv_values,
        method_values,
        performance_mode,
    ):
        project_name = (focus_data or {}).get("project")
        if not project_name:
            raise PreventUpdate
        project_code = (focus_data or {}).get("code")
        eff_mode = _modal_mode_from_store(performance_mode, "erection")
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        ctx = dash.callback_context
        triggered_id = None
        if ctx.triggered:
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        dropdown_selection = dropdown_value or selected_store_gang
        if not isinstance(modal_meta, dict):
            modal_meta = {}
        meta_source = modal_meta.get("source")
        meta_gang = modal_meta.get("gang")
        modal_sources = {
            "project-modal-actual-vs-bench",
            "project-modal-top5",
            "project-modal-bottom5",
        }
        meta_is_chart = meta_source in modal_sources and bool(meta_gang)
        if triggered_id == "store-project-modal-click-meta":
            gang_focus = meta_gang if meta_is_chart else dropdown_selection
        else:
            gang_focus = dropdown_selection or (meta_gang if meta_is_chart else None)
        if not gang_focus:
            raise PreventUpdate
        scope_meta = _build_project_scope_meta(
            project_name,
            project_code,
            eff_mode,
            months,
            quick_range,
            gangs,
            kv_values,
            method_values,
        )
        idle_data, daily_data = _compute_trace_table_payload(scope_meta, gang_focus)
        return idle_data, daily_data

    @app.callback(
        Output("project-modal-avp-list", "children"),
        Output("project-modal-actual-vs-bench", "figure"),
        Output("project-modal-top5", "figure"),
        Output("project-modal-bottom5", "figure"),
        Input("store-project-tile-focus", "data"),
        Input("project-modal-topbot-metric", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
        Input("store-project-modal-performance-mode", "data"),
    )
    def _update_project_modal_performance(
        focus_data: dict[str, Any] | None,
        topbot_metric: str | None,
        months,
        quick_range,
        gangs,
        kv_values,
        method_values,
        performance_mode,
    ):
        empty_fig = go.Figure()
        if not focus_data or not focus_data.get("project"):
            return (
                html.Div("Select a project tile to view gang performance.", className="text-muted"),
                empty_fig,
                empty_fig,
                empty_fig,
            )

        project_name = focus_data.get("project")
        project_code = focus_data.get("code")
        eff_mode = _modal_mode_from_store(performance_mode, "erection")
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"

        scope_meta = _build_project_scope_meta(
            project_name,
            project_code,
            eff_mode,
            months,
            quick_range,
            gangs,
            kv_values,
            method_values,
        )

        try:
            (
                *_kpi,
                avp_children,
                fig_loss,
                fig_top5,
                fig_bottom5,
                _fig_project,
            ) = _compute_dashboard_outputs(
                scope_meta,
                topbot_metric,
                avp_namespace="project-modal-avp",
            )
        except PreventUpdate:
            return (
                html.Div("No data available for the selected project.", className="text-muted"),
                empty_fig,
                empty_fig,
                empty_fig,
            )

        avp_children = avp_children or html.Div("No gangs available for this selection.", className="text-muted")
        return avp_children, fig_loss, fig_top5, fig_bottom5


    @app.callback(
        Output("tbl-erections-completed", "columns"),
        Output("tbl-erections-completed", "data"),
        Input("erections-completion-range", "start_date"),
        Input("erections-completion-range", "end_date"),
        Input("erections-search", "value"),
        Input("store-filtered-scope", "data"),
    )
    def update_erections_completed(
        start_date,
        end_date,
        search_text,
        scope_meta: dict[str, Any] | None,
    ) -> list[dict[str, object]]:
        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        eff_mode = _normalize_mode(scope_meta.get("mode"))
        scoped = _scope_frame_from_store(scope_meta, "project_gang").copy()

        if eff_mode == "stringing":
            export_df, display_df = _prepare_stringing_completed(
                scoped,
                range_start=range_start,
                range_end=range_end,
                search_text=search_text,
            )
            columns = [
                {"name": "Completion Date", "id": "completion_date"},
                {"name": "Project", "id": "project_name"},
                {"name": "Span (From-To)", "id": "location_no"},
                {"name": "Length (KM)", "id": "tower_weight"},
                {"name": "Productivity (KM/day)", "id": "daily_prod_mt"},
                {"name": "Gang", "id": "gang_name"},
                {"name": "F/S Start Date", "id": "start_date"},
                {"name": "Supervisor", "id": "supervisor_name"},
                {"name": "Section Incharge", "id": "section_incharge_name"},
                {"name": "Revenue", "id": "revenue"},
            ]
        else:
            export_df, display_df = _prepare_erections_completed(
                scoped,
                range_start=range_start,
                range_end=range_end,
                responsibilities_provider=responsibilities_provider,
                search_text=search_text,
            )
            columns = [
                {"name": "Completion Date", "id": "completion_date"},
                {"name": "Project", "id": "project_name"},
                {"name": "Location", "id": "location_no"},
                {"name": "Tower Weight (MT)", "id": "tower_weight"},
                {"name": "Productivity (MT/day)", "id": "daily_prod_mt"},
                {"name": "Gang", "id": "gang_name"},
                {"name": "Start Date", "id": "start_date"},
                {"name": "Supervisor", "id": "supervisor_name"},
                {"name": "Section Incharge", "id": "section_incharge_name"},
                {"name": "Revenue", "id": "revenue"},
            ]

        if display_df.empty:
            return columns, []
        return columns, display_df.to_dict("records")

    @app.callback(
        Output("project-modal-erections-search", "value"),
        Input("project-modal-erections-search-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_modal_erections_search(n):
        if not n:
            raise PreventUpdate
        return ""

    @app.callback(
        Output("project-modal-stringing-search", "value"),
        Input("project-modal-stringing-search-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_modal_stringing_search(n):
        if not n:
            raise PreventUpdate
        return ""

    @app.callback(
        Output("project-modal-erections-table", "data"),
        Input("project-modal-erections-range", "start_date"),
        Input("project-modal-erections-range", "end_date"),
        Input("project-modal-erections-search", "value"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
    )
    def _update_modal_erections_table(
        start_date,
        end_date,
        search_text,
        focus_data: dict[str, Any] | None,
        months,
        quick_range,
        gangs,
    ):
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return []
        project_code = (focus_data or {}).get("code")

        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        candidate_ids = _project_filter_candidates(project_name, project_code)
        if not candidate_ids:
            if project_name:
                candidate_ids = [str(project_name)]
            elif project_code:
                candidate_ids = [str(project_code)]
        project_list = _normalize_str_list(candidate_ids)
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))

        frames, _, _ = _build_scope_frames(
            "erection",
            project_list=project_list,
            gang_list=gang_list,
            months_value=months_list,
            quick_range=quick_range,
            kv_values=[],
            method_values=[],
        )
        scoped = frames.get("project_gang", pd.DataFrame()).copy()
        export_df, display_df = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=responsibilities_provider,
            search_text=search_text,
        )
        return display_df.to_dict("records") if not display_df.empty else []

    @app.callback(
        Output("project-modal-stringing-table", "data"),
        Input("project-modal-stringing-range", "start_date"),
        Input("project-modal-stringing-range", "end_date"),
        Input("project-modal-stringing-search", "value"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
    )
    def _update_modal_stringing_table(
        start_date,
        end_date,
        search_text,
        focus_data: dict[str, Any] | None,
        months,
        quick_range,
        gangs,
        kv_values,
        method_values,
    ):
        if not config.enable_stringing:
            return []
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return []
        project_code = (focus_data or {}).get("code")

        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        candidate_ids = _project_filter_candidates(project_name, project_code)
        if not candidate_ids:
            if project_name:
                candidate_ids = [str(project_name)]
            elif project_code:
                candidate_ids = [str(project_code)]
        project_list = _normalize_str_list(candidate_ids)
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        kv_list = _ensure_list(kv_values)
        method_list = _ensure_list(method_values)

        months_ts = resolve_months(months_list, quick_range)

        frames, _, _ = _build_scope_frames(
            "stringing",
            project_list=project_list,
            gang_list=gang_list,
            months_value=months_list,
            quick_range=quick_range,
            kv_values=kv_list,
            method_values=method_list,
        )
        scoped = frames.get("project_gang", pd.DataFrame()).copy()
        if "date" not in scoped.columns or scoped.empty:
            selector = DATA_SELECTOR
            df_fallback = selector.select("stringing") if selector is not None else pd.DataFrame()
            if isinstance(df_fallback, pd.DataFrame) and not df_fallback.empty:
                scoped = apply_filters(
                    _stringing_scope(df_fallback, kv_list, method_list),
                    project_list,
                    months_ts,
                    gang_list,
                )
        export_df, display_df = _prepare_stringing_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            search_text=search_text,
        )
        return display_df.to_dict("records") if not display_df.empty else []

    @app.callback(
        Output("erections-completion-range", "start_date"),
        Output("erections-completion-range", "end_date"),
        Output("erections-search", "value"),
        Input("btn-reset-erections", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_erections_controls(reset_clicks: int | None) -> tuple[str, str, str]:
        if not reset_clicks:
            raise PreventUpdate

        default = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date().isoformat()
        return default, default, ""

    @app.callback(
        Output("download-trace-xlsx", "data"),
        Output("project-modal-download-trace", "data"),
        Input("btn-export-trace", "n_clicks"),
        Input("modal-btn-export-trace", "n_clicks"),
        Input("project-modal-btn-export-trace", "n_clicks"),
        State("f-project", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
        State("trace-gang", "value"),
        State("store-selected-gang", "data"),
        State("erections-completion-range", "start_date"),
        State("erections-completion-range", "end_date"),
        State("erections-search", "value"),
        State("store-project-tile-focus", "data"),
        State("project-modal-trace-gang", "value"),
        State("project-modal-erections-range", "start_date"),
        State("project-modal-erections-range", "end_date"),
        State("project-modal-erections-search", "value"),
        prevent_initial_call=True,
    )
    def export_trace(
        main_clicks: int | None,
        modal_clicks: int | None,
        project_modal_clicks: int | None,
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        trace_gang_value: str | None,
        selected_gang: str | None,
        erections_start: str | None,
        erections_end: str | None,
        erections_search: str | None,
        focus_data: dict[str, Any] | None,
        project_modal_trace_gang: str | None,
        modal_start: str | None,
        modal_end: str | None,
        modal_search: str | None,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "project-modal-btn-export-trace":
            project_name = (focus_data or {}).get("project")
            if not project_name:
                raise PreventUpdate
            project_list = [project_name]
            month_list = _ensure_list(months)
            gang_list = _ensure_list(gangs)
            df_day = data_selector.select("erection")
            months_ts = resolve_months(month_list, quick_range)
            scoped = apply_filters(df_day, project_list, months_ts, gang_list)
            gang_for_sheet = project_modal_trace_gang or selected_gang
            benchmark_value = BENCHMARK_MT_PER_DAY
            project_info_df = project_info_provider() if project_info_provider else None

            range_start = _parse_completion_date(modal_start) or _default_completion_date()
            range_end = _parse_completion_date(modal_end) or range_start
            if range_start > range_end:
                range_start, range_end = range_end, range_start
            erections_export_df, _ = _prepare_erections_completed(
                scoped,
                range_start=range_start,
                range_end=range_end,
                responsibilities_provider=responsibilities_provider,
                search_text=modal_search,
            )
            erections_context = {
                "range_start": range_start,
                "range_end": range_end,
                "search_text": (modal_search or ""),
            }

            def _writer(buffer: BytesIO) -> None:
                buffer.write(
                    make_trace_workbook_bytes(
                        scoped,
                        months_ts,
                        project_list,
                        gang_list,
                        benchmark_value,
                        gang_for_sheet=gang_for_sheet,
                        config=config,
                        project_info=project_info_df,
                        erections_completed=erections_export_df,
                        erections_context=erections_context,
                    )
                )

            return dash.no_update, send_bytes(_writer, "Trace_Calcs.xlsx")

        if not (main_clicks or modal_clicks):
            raise PreventUpdate

        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        gang_list = _ensure_list(gangs)

        df_day = data_selector.select("erection")
        months_ts = resolve_months(month_list, quick_range)
        scoped = apply_filters(df_day, project_list, months_ts, gang_list)
        gang_for_sheet = trace_gang_value or selected_gang
        benchmark_value = BENCHMARK_MT_PER_DAY
        project_info_df = project_info_provider() if project_info_provider else None

        range_start = _parse_completion_date(erections_start) or _default_completion_date()
        range_end = _parse_completion_date(erections_end) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        erections_export_df, _ = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=responsibilities_provider,
            search_text=erections_search,
        )

        erections_context = {
            "range_start": range_start,
            "range_end": range_end,
            "search_text": (erections_search or ""),
        }

        def _writer(buffer: BytesIO) -> None:
            buffer.write(
                make_trace_workbook_bytes(
                    scoped,
                    months_ts,
                    project_list,
                    gang_list,
                    benchmark_value,
                    gang_for_sheet=gang_for_sheet,
                    config=config,
                    project_info=project_info_df,
                    erections_completed=erections_export_df,
                    erections_context=erections_context,
                )
            )

        return send_bytes(_writer, "Trace_Calcs.xlsx"), dash.no_update
    @app.callback(
        Output("trace-gang", "options"),
        Output("trace-gang", "value"),
        Output("modal-trace-gang", "options"),
        Output("modal-trace-gang", "value"),
        Input("store-filtered-scope", "data"),
        Input("store-selected-gang", "data"),
        State("trace-gang", "value"),
    )
    def update_trace_gang_options(
        scope_meta: dict[str, Any] | None,
        clicked_gang: str | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, str]], str | None, list[dict[str, str]], str | None]:
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        base = _scope_frame_from_store(scope_meta, "project")
        if base.empty or "gang_name" not in base.columns:
            options: list[dict[str, str]] = []
            return options, None, options, None

        gangs = (
            base["gang_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        gangs = sorted({g for g in gangs if g})
        options = [{"label": gang, "value": gang} for gang in gangs]

        if clicked_gang and clicked_gang in gangs:
            value = clicked_gang
        elif current_value and current_value in gangs:
            value = current_value
        else:
            value = None
        return options, value, options, value

    # --- KPI Details drilldown: populate inline accordion ---
    def _filter_pch_header_pills(
        pill_components: list[tuple[str, Any]], pill_focus: str | None
    ) -> list[Any]:
        if not pill_focus:
            return [component for _, component in pill_components]
        focus = str(pill_focus).strip().lower()
        if not focus or focus in {"all", "default"}:
            return [component for _, component in pill_components]
        filtered = [component for key, component in pill_components if key == focus]
        return filtered or [component for _, component in pill_components]


    def _build_tile_metric_rows(
        *,
        mode: str,
        focus_metric: str | None,
        prod_current_value: float | None,
        prod_overall_value: float | None,
        total_current_value: float | None,
        total_planned_value: float | None,
        plan_available: bool = True,
        gangs_value: int | None = None,
        loss_value: float | None = None,
        tse_value: int | None = None,
        po_planned_value: float | None = None,
        po_done_value: float | None = None,
    ) -> list[html.Div]:
        """
        Build the dynamic KPI rows rendered inside each project tile.

        When *focus_metric* is provided (i.e., a summary pill opened the modal),
        the tile shows only the requested metric. Otherwise the default trio of
        production + totals rows is displayed.
        """

        is_stringing = mode == "stringing"
        prod_unit = "KM/day" if is_stringing else "MT/day"
        total_unit = "KM" if is_stringing else "Towers"
        loss_unit = "KM" if is_stringing else "MT"

        def _fmt_prod(value: float | None) -> str:
            if value is None or pd.isna(value):
                return "\u2014"
            return f"{float(value):.2f} {prod_unit}"

        def _fmt_prod_compact(value: float | None) -> str:
            if value is None or pd.isna(value):
                return "\u2014"
            return f"{float(value):.2f}"

        def _ensure_number(value: float | int | None, *, default: float = 0.0) -> float:
            if value is None or pd.isna(value):
                return default
            try:
                return float(value)
            except Exception:
                return default

        def _optional_number(value: float | int | None) -> float | None:
            if value is None or pd.isna(value):
                return None
            try:
                return float(value)
            except Exception:
                return None

        focus = (focus_metric or "").strip().lower()
        if focus not in _PCH_PILL_LABELS or focus == "projects":
            focus = None

        total_current = _ensure_number(total_current_value)
        plan_value_available = plan_available and total_planned_value is not None and not pd.isna(total_planned_value)
        total_planned = _ensure_number(total_planned_value) if plan_value_available else 0.0
        balance_value = max(total_planned - total_current, 0.0) if plan_value_available else None

        if is_stringing:
            delivered_label = f"{total_current:.1f} {total_unit}"
            planned_label = f"{total_planned:.1f} {total_unit}" if plan_value_available else "No Plan"
            balance_label = f"{balance_value:.1f} {total_unit}" if plan_value_available else "\u2014"
        else:
            delivered_label = f"{int(round(total_current))} {total_unit}"
            planned_label = f"{int(round(total_planned))} {total_unit}" if plan_value_available else "No Plan"
            balance_label = f"{int(round(balance_value))} {total_unit}" if plan_value_available else "\u2014"

        if plan_value_available:
            totals_display = f"{delivered_label} delivered / {planned_label} planned ({balance_label} balance)"
            totals_focus_display = f"{planned_label} / {delivered_label} / {balance_label}"
        else:
            totals_display = f"{delivered_label} delivered / No Plan available"
            totals_focus_display = f"No Plan / {delivered_label} / \u2014"

        def _row(label: str, value: str) -> html.Div:
            return html.Div(
                [
                    html.Span(f"{label} : ", className="me-2"),
                    dbc.Badge(value, color="dark", className="me-2", style={"fontSize": "1.05rem"}),
                ],
                className="mb-2",
            )

        if focus == "totals":
            return [_row("F/S Total Planned / Done / Balance", totals_focus_display)]
        if focus == "gangs":
            value = "\u2014" if gangs_value is None else f"{int(gangs_value):,}"
            return [_row("Gangs", value)]
        if focus == "productivity":
            prod_pair = f"{_fmt_prod_compact(prod_current_value)} / {_fmt_prod_compact(prod_overall_value)} {prod_unit}"
            return [_row("Productivity / Historical Avg", prod_pair)]
        if focus == "loss":
            if loss_value is None:
                value = "\u2014"
            else:
                value = f"{float(loss_value):.1f} {loss_unit}"
            return [_row("Lost Units", value)]
        if focus == "tse" and is_stringing:
            value = "\u2014" if tse_value is None else f"{int(tse_value):,}"
            return [_row("No. of TSE", value)]

        rows = [
            _row("Prod This Month", _fmt_prod(prod_current_value)),
            _row("Historical Avg", _fmt_prod(prod_overall_value)),
            _row(f"{total_unit} This Month", totals_display),
        ]

        po_plan = _optional_number(po_planned_value)
        po_done = _optional_number(po_done_value)
        if is_stringing and focus is None and (po_plan is not None or po_done is not None):
            plan_label = f"{po_plan:.1f} KM" if po_plan is not None else "No Plan"
            done_label = f"{po_done:.1f} KM" if po_done is not None else "\u2014"
            if po_plan is not None:
                balance_value = max(po_plan - (po_done or 0.0), 0.0)
                balance_label = f"{balance_value:.1f} KM"
            else:
                balance_label = "\u2014"
            rows.append(
                _row(
                    "P/O Total Planned / Done / Balance",
                    f"{plan_label} / {done_label} / {balance_label}",
                )
            )

        return rows

    def _populate_kpi_pch(
        projects,
        months,
        quick_range,
        mode_value: str | None,
        kv_filter,
        method_filter,
        *,
        use_modal_ids: bool = False,
        pill_focus: str | None = None,
    ):
        mode = (mode_value or "erection").strip().lower()
        try:
            import re as _re_slug
        except Exception:  # pragma: no cover - regex should be available, but keep fallback
            _re_slug = None

        focus_metric = None
        if pill_focus:
            focus_raw = str(pill_focus).strip().lower()
            if focus_raw in _PCH_PILL_LABELS and focus_raw != "projects":
                focus_metric = focus_raw

        tile_context = "modal" if use_modal_ids else "inline"
        tile_metadata: dict[str, dict[str, Any]] = {}

        def _slugify_pch(value: Any) -> str:
            text = str(value or "").strip().lower()
            if not text:
                return "unknown"
            if _re_slug is not None:
                text = _re_slug.sub(r"[^a-z0-9]+", "-", text)
            else:
                text = text.replace(" ", "-")
            text = text.strip("-")
            return text or "unknown"

        def _empty_pch_items(message: str) -> list[dbc.AccordionItem]:
            return [
                dbc.AccordionItem(
                    title="No PCH data",
                    children=html.Div(message, className="text-muted"),
                    item_id="pch-empty",
                    className="pch-section mb-2",
                )
            ]

        def _detect_project_column(frame: pd.DataFrame | None) -> str | None:
            """
            Return the first available project column in *frame* that can be used
            to associate gang rows back to a project label.
            """
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return None
            for candidate in ("project_name", "project", "Project Name", "project_name_display"):
                if candidate in frame.columns:
                    return candidate
            return None

        def _compact_project_token(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())

        def _project_code_key_sets(entry: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
            norm_keys: list[str] = []
            compact_keys: list[str] = []
            if not isinstance(entry, Mapping):
                return norm_keys, compact_keys
            seen: set[str] = set()
            for field in ("project_code", "code", "project", "project_name", "project_name_display"):
                code_value = _extract_project_code(entry.get(field))
                if not code_value:
                    continue
                canonical = code_value.lower()
                if canonical in seen:
                    continue
                seen.add(canonical)
                norm_key = _normalize_lower(code_value)
                if norm_key and norm_key not in norm_keys:
                    norm_keys.append(norm_key)
                compact_key = _compact_project_token(code_value)
                if compact_key and compact_key not in compact_keys:
                    compact_keys.append(compact_key)
            return norm_keys, compact_keys

        def _resolve_project_code(entry: Mapping[str, Any] | None) -> str:
            if not isinstance(entry, Mapping):
                return ""
            for field in ("project_code", "code", "project", "project_name", "project_name_display"):
                code_value = _extract_project_code(entry.get(field))
                if code_value:
                    return code_value
            return ""

        def _component_id_token(prefix: str, value: Any) -> str:
            base = _compact_project_token(value)
            if base:
                return base
            text = str(value or "").strip()
            if not text:
                text = prefix or "component"
            try:
                raw = text.encode("utf-8")
            except Exception:
                raw = text.encode("utf-8", errors="ignore")
            digest = hashlib.sha1(raw).hexdigest()[:8]
            return f"{prefix}-{digest}"

        def _filter_scope_by_projects(
            scope: pd.DataFrame | None, rows: list[dict[str, Any]]
        ) -> pd.DataFrame:
            if not isinstance(scope, pd.DataFrame) or scope.empty or not rows:
                return pd.DataFrame()
            match_codes: set[str] = set()
            for entry in rows:
                _, compact_keys = _project_code_key_sets(entry)
                match_codes.update(compact_keys)
            if not match_codes:
                return pd.DataFrame()

            mask = pd.Series(False, index=scope.index)
            for column in ("project_code", "project", "project_name", "project_name_display"):
                if column in scope.columns:
                    mask |= scope[column].astype(str).map(_compact_project_token).isin(match_codes)

            if not mask.any():
                return scope.iloc[0:0].copy()
            return scope.loc[mask].copy()

        def _project_scope_for_row(
            row: dict[str, Any],
            *,
            primary_scope: pd.DataFrame | None,
            fallback_scope: pd.DataFrame | None,
        ) -> pd.DataFrame:
            subset = _filter_scope_by_projects(primary_scope, [row])
            if subset.empty:
                subset = _filter_scope_by_projects(fallback_scope, [row])
            return subset

        def _count_gangs(frame: pd.DataFrame | None) -> int | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty or "gang_name" not in frame.columns:
                return None
            series = (
                frame["gang_name"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            series = series[series != ""]
            if series.empty:
                return 0
            return int(series.nunique())

        def _count_tse(frame: pd.DataFrame | None) -> int | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty or "method" not in frame.columns:
                return None
            mask = frame["method"].astype(str).str.strip().str.lower() == "tse"
            if not mask.any():
                return 0
            if "gang_name" in frame.columns:
                gangs = (
                    frame.loc[mask, "gang_name"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                gangs = gangs[gangs != ""]
                if not gangs.empty:
                    return int(gangs.nunique())
            return int(mask.sum())

        def _prepare_scope_for_loss(frame: pd.DataFrame | None) -> pd.DataFrame:
            if not isinstance(frame, pd.DataFrame):
                return pd.DataFrame()
            if frame.empty:
                return frame
            if "project_name" not in frame.columns:
                for alt in ("project", "project_name_display"):
                    if alt in frame.columns:
                        work = frame.copy()
                        work["project_name"] = work[alt].astype(str)
                        return work
            return frame

        def _compute_project_loss_value(frame: pd.DataFrame | None, *, mode: str) -> float | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return None
            metric_col = "daily_km" if mode == "stringing" else "daily_prod_mt"
            if metric_col not in frame.columns:
                return None
            work = _prepare_scope_for_loss(frame)
            if work.empty or "project_name" not in work.columns or "gang_name" not in work.columns:
                return None
            work = work.dropna(subset=["gang_name"]).copy()
            work["gang_name"] = work["gang_name"].astype(str).str.strip()
            work = work[work["gang_name"] != ""]
            if work.empty:
                return None
            try:
                if mode == "stringing":
                    overall_map, monthly_map = compute_project_baseline_maps_for(work, metric_col)
                else:
                    overall_map, monthly_map = compute_project_baseline_maps(work)
            except Exception:
                overall_map, monthly_map = {}, {}
            proj_col = _detect_project_column(work) or ("project_name" if "project_name" in work.columns else None)
            if proj_col and proj_col not in work.columns:
                proj_col = "project_name"
            if proj_col:
                gang_project_map = (
                    work[["gang_name", proj_col]]
                    .dropna(subset=["gang_name"])
                    .astype(str)
                    .rename(columns={proj_col: "project_name"})
                    .drop_duplicates(subset=["gang_name"], keep="last")
                    .set_index("gang_name")["project_name"]
                    .to_dict()
                )
            else:
                gang_project_map = {}
            default_project = str(work["project_name"].iloc[0]) if "project_name" in work.columns and not work.empty else None
            total_loss = 0.0
            for gang, gdf in work.groupby("gang_name"):
                if gdf.empty:
                    continue
                project_label = gang_project_map.get(str(gang), default_project)
                baseline_value = overall_map.get(project_label)
                monthly_lookup = monthly_map.get(project_label, {})
                try:
                    if mode == "stringing":
                        _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss_for_column(
                            gdf,
                            metric_column=metric_col,
                            loss_max_gap_days=config.loss_max_gap_days,
                            baseline_per_day=baseline_value,
                            baseline_by_month=monthly_lookup,
                        )
                    else:
                        _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss(
                            gdf,
                            loss_max_gap_days=config.loss_max_gap_days,
                            baseline_mt_per_day=baseline_value,
                            baseline_by_month=monthly_lookup,
                        )
                except Exception:
                    continue
                if pd.notna(loss_val):
                    total_loss += float(loss_val)
            return total_loss

        kv_list = _ensure_list(kv_filter)
        method_list = _normalize_str_list(method_filter, lower=True)
        pch_sections: list[dbc.AccordionItem] = []
        # Erection mode (existing flow)
        stringing_plan_month_map: dict[str, set[pd.Timestamp]] = (
            _load_stringing_plan_month_map(date_columns=_STRINGING_FS_DATE_COLUMNS) if mode == "stringing" else {}
        )

        def _project_has_plan(name: str, code: str) -> bool:
            return _stringing_project_has_plan(stringing_plan_month_map, name, code)

        def _project_plan_months(name: str, code: str) -> list[pd.Timestamp]:
            return _stringing_plan_months_for_project(stringing_plan_month_map, name, code)

        if mode == "stringing":
            # Stringing mode: build PCH-wise planned vs delivered (KM)
            project_list = _ensure_list(projects)
            month_list = _ensure_list(months)
            months_ts = resolve_months(month_list, quick_range)

            # Month range for display/derivations
            if months_ts:
                range_start = pd.Timestamp(min(months_ts)).normalize()
                range_end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
            else:
                today = pd.Timestamp.today().normalize()
                range_start = today.to_period("M").start_time.normalize()
                range_end = (today + pd.offsets.MonthEnd(0)).normalize()
            current_month_ts = range_end.to_period("M").to_timestamp()

            # Delivered KM from per-day stringing dataset
            df_day = data_selector.select("stringing")
            scoped = apply_filters(df_day, project_list, months_ts, [])
            try:
                scope_frames, _, _ = _build_scope_frames(
                    "stringing",
                    project_list=project_list,
                    gang_list=[],
                    months_value=month_list,
                    quick_range=quick_range,
                    kv_values=kv_list,
                    method_values=method_list,
                )
                scope_full = scope_frames.get("full", pd.DataFrame()).copy()
            except Exception:
                scope_full = pd.DataFrame()
            delivered_km_current_series = pd.Series(dtype=float)
            if isinstance(scoped, pd.DataFrame) and not scoped.empty:
                proj_col = "project_name" if "project_name" in scoped.columns else ("project" if "project" in scoped.columns else None)
                if proj_col is None:
                    return _empty_pch_items("Missing project information in the dataset."), None
                scoped_norm = scoped.copy()
                if proj_col != "project_name":
                    scoped_norm = scoped_norm.rename(columns={proj_col: "project_name"})
                scoped_norm["project_name"] = scoped_norm["project_name"].astype(str).str.strip()
                completion_rows = _filter_completion_rows(scoped_norm, completion_column="fs_complete_date")
                if not completion_rows.empty and "length_km" in completion_rows.columns:
                    delivered_km_by_project = (
                        completion_rows.groupby("project_name")["length_km"]
                        .sum()
                        .rename("delivered_km")
                        .to_frame()
                    )
                    completion_rows = completion_rows.copy()
                    completion_rows["month"] = pd.to_datetime(completion_rows["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                    current_rows = completion_rows[completion_rows["month"] == current_month_ts]
                    if not current_rows.empty:
                        delivered_km_current_series = current_rows.groupby("project_name")["length_km"].sum()
                elif "daily_km" in scoped_norm.columns:
                    delivered_km_by_project = (
                        scoped_norm.groupby("project_name")["daily_km"].sum().rename("delivered_km").to_frame()
                    )
                    scoped_month = scoped_norm.copy()
                    if "month" not in scoped_month.columns and "date" in scoped_month.columns:
                        scoped_month["date"] = pd.to_datetime(scoped_month["date"], errors="coerce")
                        scoped_month = scoped_month.dropna(subset=["date"])
                        scoped_month["month"] = scoped_month["date"].dt.to_period("M").dt.to_timestamp()
                    elif "month" in scoped_month.columns:
                        scoped_month["month"] = pd.to_datetime(scoped_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                    if "month" in scoped_month.columns:
                        scoped_current = scoped_month[scoped_month["month"] == current_month_ts]
                        if not scoped_current.empty:
                            delivered_km_current_series = (
                                scoped_current.groupby("project_name")["daily_km"].sum()
                            )
                else:
                    delivered_km_by_project = pd.DataFrame(columns=["delivered_km"])
            else:
                delivered_km_by_project = pd.DataFrame(columns=["delivered_km"])
                delivered_km_current_series = pd.Series(dtype=float)

            plan_union, plan_current_lookup = _stringing_plan_totals_by_project(
                months_ts,
                current_month=current_month_ts,
            )
            planned_km_current_series = plan_current_lookup.copy()

            try:
                tse_norm_lookup, tse_alias_lookup = _get_stringing_tse_lookup()
            except Exception:
                LOGGER.exception("Failed to build TSE lookup")
                tse_norm_lookup, tse_alias_lookup = {}, {}

            # Merge planned and delivered into a projects table
            delivered_union = delivered_km_by_project.copy()
            if not delivered_union.empty:
                delivered_union = delivered_union.copy()
                delivered_union["project_name_delivered"] = delivered_union.index.astype(str)
                delivered_union["project_key_norm"] = delivered_union["project_name_delivered"].map(
                    lambda value: _compact_project_key(value) or _normalize_lower(value)
                )
                delivered_union = delivered_union[delivered_union["project_key_norm"].astype(str).str.strip() != ""]
                if not delivered_union.empty:
                    delivered_union = (
                        delivered_union.set_index("project_key_norm")[["delivered_km", "project_name_delivered"]]
                    )
            else:
                delivered_union = pd.DataFrame(columns=["delivered_km", "project_name_delivered"]).set_index(
                    pd.Index([], name="project_key_norm")
                )

            if not plan_union.empty:
                plan_union = plan_union.copy()
            else:
                plan_union = pd.DataFrame(columns=["planned_km", "project_name_plan"]).set_index(
                    pd.Index([], name="project_key_norm")
                )

            projects_df = (
                plan_union.join(delivered_union, how="outer")
                .fillna({"planned_km": 0.0, "delivered_km": 0.0})
                .reset_index()
                .rename(columns={"index": "project_key_norm"})
            )
            delivered_names = projects_df["project_name_delivered"].fillna("").astype(str)
            plan_names = projects_df["project_name_plan"].fillna("").astype(str)
            projects_df["project_name"] = delivered_names.where(
                delivered_names.str.strip() != "",
                plan_names,
            )
            projects_df["project_name"] = projects_df["project_name"].fillna("")
            projects_df["project_name"] = projects_df["project_name"].where(
                projects_df["project_name"].str.strip() != "",
                projects_df["project_key_norm"],
            )
            projects_df["project_name"] = projects_df["project_name"].astype(str).map(_format_stringing_project_label)
            project_label_by_key = (
                projects_df.set_index("project_key_norm")["project_name"].to_dict()
                if not projects_df.empty
                else {}
            )

            if not planned_km_current_series.empty:
                planned_km_current_series = planned_km_current_series.rename(
                    index=lambda key: project_label_by_key.get(str(key), key)
                )
            else:
                planned_km_current_series = pd.Series(dtype=float)
            projects_df = projects_df.drop(columns=["project_name_plan", "project_name_delivered"], errors="ignore")
            projects_df["project_name_display"] = projects_df["project_name"].astype(str)

            # Project meta (PCH, managers) from Project Details
            try:
                info_df = project_info_provider() if callable(project_info_provider) else None
            except Exception:
                info_df = None
            pch_col = "pch"
            info_key_map: dict[str, int] = {}
            if isinstance(info_df, pd.DataFrame) and not info_df.empty:
                info = info_df.copy()
                info["project_name_display"] = info.get("Project Name", info.get("project_name", "")).astype(str)
                info["project_name_norm"] = info["project_name_display"].map(_normalize_lower)
                pch_col = None
                for cand in ("PCH", "pch", "PCH Name", "PCHName", "pch_name"):
                    if cand in info.columns:
                        pch_col = cand
                        break
                if pch_col is None:
                    info["pch"] = ""
                    pch_col = "pch"
                try:
                    import re as _re_key

                    def _compact_code(value: str) -> str:
                        return _re_key.sub(r"[^a-z0-9]", "", (value or "").lower())
                except Exception:

                    def _compact_code(value: str) -> str:
                        return str(value or "").strip().lower().replace(" ", "")

                name_keys = info["project_name_display"].astype(str).map(_compact_code)
                for idx, key in zip(info.index, name_keys):
                    if key and key not in info_key_map:
                        info_key_map[key] = idx
                for code_col in ("project_code", "Project Code"):
                    if code_col in info.columns:
                        code_keys = info[code_col].astype(str).map(_compact_code)
                        for idx, key in zip(info.index, code_keys):
                            if key and key not in info_key_map:
                                info_key_map[key] = idx
            else:
                info = pd.DataFrame(columns=["project_name_display", "project_name_norm", "pch", "regional_mgr", "project_mgr", "planning_eng"])
                def _compact_code(value: str) -> str:
                    return str(value or "").strip().lower().replace(" ", "")

            proj_info_pch = {}
            proj_info_pch_norm = {}
            if not info.empty and "pch" in info.columns:
                proj_info_pch = dict(zip(info["project_name_display"], info["pch"].astype(str)))
                proj_info_pch_norm = { _normalize_lower(k): str(v) for k, v in proj_info_pch.items() if str(k).strip() }

            try:
                from .pch_normalizer import normalize_pch as _normalize_pch, CANONICAL_PCH_PRIMARY as _PCH_ORDER
            except Exception:
                def _normalize_pch(v):
                    return str(v or "").strip()
                _PCH_ORDER = ()

            # Build structure: PCH -> list of project tiles
            projects_rows: dict[str, list[dict[str, Any]]] = {}
            for _, row in projects_df.iterrows():
                proj = str(row.get("project_name_display", "")).strip()
                if not proj:
                    continue
                planned_km = float(row.get("planned_km", 0.0) or 0.0)
                delivered_km = float(row.get("delivered_km", 0.0) or 0.0)
                # Meta join
                meta = info[info.get("project_name_norm", "").astype(str) == _normalize_lower(proj)].iloc[:1] if not info.empty else pd.DataFrame()
                if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                    target_key = _compact_code(proj)
                    if target_key and target_key in info_key_map:
                        meta = info.loc[[info_key_map[target_key]]]
                raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
                if (not raw_pch) and proj:
                    raw_pch = proj_info_pch.get(proj, "") or proj_info_pch_norm.get(_normalize_lower(proj), "")
                # Use normalized PCH if known; otherwise keep the original as-is (no 'Unassigned')
                pch_label = _normalize_pch(raw_pch) or str(raw_pch or "").strip()
                try:
                    proj_code = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                        meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                    )) if isinstance(meta, pd.DataFrame) else ""
                except Exception:
                    proj_code = ""
                if isinstance(meta, pd.DataFrame) and not meta.empty:
                    proj_display_series = meta.get("project_name_display", meta.get("project_name", pd.Series([proj], index=meta.index)))
                    proj_display_name = str(proj_display_series.iloc[0]) if not proj_display_series.empty else str(proj)
                else:
                    proj_display_name = str(proj)
                proj_display = f"{proj_code} : {proj_display_name}".strip(" :") if proj_code else proj_display_name
                plan_months = _project_plan_months(proj_display_name, proj_code or proj_display_name)
                resolved_code = (
                    _extract_project_code(proj_code)
                    or _extract_project_code(proj_display)
                    or _extract_project_code(proj)
                )
                if not resolved_code:
                    resolved_code = proj_code or proj
                rec = {
                    "project_name": proj_display,
                    "project_code": resolved_code,
                    "code": resolved_code,
                    "regional_mgr": (meta.get("regional_mgr", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns) else ""),
                    "project_mgr": (meta.get("project_mgr", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns) else ""),
                    "planning_eng": (meta.get("planning_eng", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns) else ""),
                    # store KM in MT fields to reuse downstream structure but change labels to KM
                    "planned_mt": round(planned_km, 1),
                    "delivered_mt": round(delivered_km, 1),
                    # counts not applicable for stringing
                    "planned_nos": 0,
                    "delivered_nos": 0,
                    "_stringing_plan_months": plan_months,
                    "_stringing_plan_available": bool(plan_months),
                }
                projects_rows.setdefault(pch_label, []).append(rec)

            def _pch_sort_key(name: str) -> tuple[int, str]:
                # Put empty/blank PCH last; then canonical order; then alphabetical
                if not str(name or "").strip():
                    return (2, "")
                try:
                    idx = list(_PCH_ORDER).index(name)
                    return (0, f"{idx:03d}")
                except ValueError:
                    return (1, str(name))

            delivered_current_norm_map: dict[str, float] = {}
            delivered_current_compact_map: dict[str, float] = {}
            planned_current_norm_map: dict[str, float] = {}
            planned_current_compact_map: dict[str, float] = {}
            prod_current_norm_map: dict[str, float] = {}
            prod_current_compact_map: dict[str, float] = {}
            prod_overall_norm_map: dict[str, float] = {}
            prod_overall_compact_map: dict[str, float] = {}

            def _build_lookup_maps(source: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
                norm_map: dict[str, float] = {}
                compact_map: dict[str, float] = {}
                if not source:
                    return norm_map, compact_map
                for key, raw_val in source.items():
                    text = str(key or "").strip()
                    if not text:
                        continue
                    try:
                        value = float(raw_val)
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(value):
                        continue
                    norm_key = _normalize_lower(text)
                    if norm_key and norm_key not in norm_map:
                        norm_map[norm_key] = value
                    compact_key = _compact_code(text)
                    if compact_key and compact_key not in compact_map:
                        compact_map[compact_key] = value
                return norm_map, compact_map

            if isinstance(delivered_km_current_series, pd.Series) and not delivered_km_current_series.empty:
                delivered_current_norm_map, delivered_current_compact_map = _build_lookup_maps(delivered_km_current_series.to_dict())
            if isinstance(planned_km_current_series, pd.Series) and not planned_km_current_series.empty:
                planned_current_norm_map, planned_current_compact_map = _build_lookup_maps(planned_km_current_series.to_dict())

            po_plan_lookup_raw: dict[str, float] = {}
            po_plan_norm_map: dict[str, float] = {}
            po_plan_compact_map: dict[str, float] = {}
            try:
                po_plan_union, _ = _stringing_plan_totals_by_project(
                    months_ts,
                    current_month=None,
                    date_columns=_STRINGING_PO_DATE_COLUMNS,
                )
            except Exception:
                LOGGER.exception("Failed to load P/O plan totals for tiles")
                po_plan_union = pd.DataFrame()
            if isinstance(po_plan_union, pd.DataFrame) and not po_plan_union.empty:
                po_plan_reset = po_plan_union.reset_index().rename(columns={"index": "project_key_norm"})
                for _, entry in po_plan_reset.iterrows():
                    plan_value = entry.get("planned_km")
                    if pd.isna(plan_value):
                        continue
                    try:
                        numeric_value = float(plan_value)
                    except (TypeError, ValueError):
                        continue
                    label_candidates = [
                        entry.get("project_name_plan"),
                        project_label_by_key.get(str(entry.get("project_key_norm")), ""),
                        entry.get("project_key_norm"),
                    ]
                    for candidate in label_candidates:
                        text = str(candidate or "").strip()
                        if not text:
                            continue
                        po_plan_lookup_raw[text] = numeric_value
                if po_plan_lookup_raw:
                    po_plan_norm_map, po_plan_compact_map = _build_lookup_maps(po_plan_lookup_raw)

            po_done_lookup_raw: dict[str, float] = {}
            po_done_norm_map: dict[str, float] = {}
            po_done_compact_map: dict[str, float] = {}
            po_frame = _get_stringing_po_daily_frame()
            if isinstance(po_frame, pd.DataFrame) and not po_frame.empty:
                po_scoped = _stringing_scope(po_frame, kv_list, method_list)
                po_filtered = apply_filters(po_scoped, project_list, months_ts, [])
                if isinstance(po_filtered, pd.DataFrame) and not po_filtered.empty:
                    proj_col = "project_name" if "project_name" in po_filtered.columns else (
                        "project" if "project" in po_filtered.columns else None
                    )
                    if proj_col:
                        work = po_filtered.copy()
                        work[proj_col] = work[proj_col].astype(str).str.strip()
                        work = work.rename(columns={proj_col: "project_name"})
                        work = work[work["project_name"] != ""]
                        completion_work = _filter_completion_rows(work, completion_column="po_completion_date")
                        if not completion_work.empty and "length_km" in completion_work.columns:
                            po_grouped = completion_work.groupby("project_name")["length_km"].sum()
                        elif "daily_km" in work.columns:
                            work = work.dropna(subset=["project_name", "daily_km"])
                            po_grouped = work.groupby("project_name")["daily_km"].sum(min_count=1)
                        else:
                            po_grouped = pd.Series(dtype=float)
                        if not po_grouped.empty:
                            for label, raw_val in po_grouped.items():
                                if pd.isna(raw_val):
                                    continue
                                try:
                                    numeric_value = float(raw_val)
                                except (TypeError, ValueError):
                                    continue
                                display_label = _format_stringing_project_label(str(label))
                                norm_key = _compact_project_key(label) or _normalize_lower(label)
                                for name in (display_label, project_label_by_key.get(norm_key), norm_key):
                                    if not name:
                                        continue
                                    text = str(name).strip()
                                    if text:
                                        po_done_lookup_raw[text] = numeric_value
            if po_done_lookup_raw:
                po_done_norm_map, po_done_compact_map = _build_lookup_maps(po_done_lookup_raw)

            if isinstance(df_day, pd.DataFrame) and not df_day.empty and "daily_km" in df_day.columns:
                day_filtered = df_day.copy()
                if project_list and "project_name" in day_filtered.columns:
                    project_filter_values = [str(p).strip() for p in project_list if str(p).strip()]
                    if project_filter_values:
                        day_filtered = day_filtered[day_filtered["project_name"].astype(str).str.strip().isin(project_filter_values)]
                project_name_col = "project_name" if "project_name" in day_filtered.columns else ("project" if "project" in day_filtered.columns else None)
                if project_name_col:
                    day_filtered[project_name_col] = day_filtered[project_name_col].astype(str).str.strip()
                    day_filtered = day_filtered.rename(columns={project_name_col: "project_name"})
                if "month" not in day_filtered.columns and "date" in day_filtered.columns:
                    day_filtered["date"] = pd.to_datetime(day_filtered["date"], errors="coerce")
                    day_filtered["month"] = day_filtered["date"].dt.to_period("M").dt.to_timestamp()
                elif "month" in day_filtered.columns:
                    day_filtered["month"] = pd.to_datetime(day_filtered["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                day_filtered = day_filtered.dropna(subset=["project_name", "daily_km"])
                baseline_payload = {
                    "kind": "stringing-day-baseline",
                    "projects": sorted(project_list),
                    "months": [ts.isoformat() for ts in months_ts],
                    "rows": int(len(day_filtered.index)),
                    "version": int(df_day.attrs.get("_appdata_version", 0)),
                }
                baseline_token = f"stringing-baseline::{_hash_cache_payload(baseline_payload)}"

                def _compute_day_baseline() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                    if day_filtered.empty:
                        return {}, {}
                    return compute_project_baseline_maps_for(day_filtered, "daily_km")

                overall_map, _ = _cached_global_result(
                    baseline_token,
                    _compute_day_baseline,
                    clone=_clone_baseline_result,
                )
                prod_overall_norm_map, prod_overall_compact_map = _build_lookup_maps(overall_map)
                current_scope = day_filtered[day_filtered["month"] == current_month_ts] if "month" in day_filtered.columns else pd.DataFrame()
                if not current_scope.empty:
                    current_payload = dict(baseline_payload)
                    current_payload.update(
                        {
                            "window": "current",
                            "month": current_month_ts.isoformat(),
                            "rows": int(len(current_scope.index)),
                        }
                    )
                    current_token = f"stringing-baseline::{_hash_cache_payload(current_payload)}"

                    def _compute_current_baseline() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                        return compute_project_baseline_maps_for(current_scope, "daily_km")

                    current_map, _ = _cached_global_result(
                        current_token,
                        _compute_current_baseline,
                        clone=_clone_baseline_result,
                    )
                    prod_current_norm_map, prod_current_compact_map = _build_lookup_maps(current_map)
                if not prod_current_norm_map:
                    prod_current_norm_map, prod_current_compact_map = prod_overall_norm_map.copy(), prod_overall_compact_map.copy()

            pch_sections = []
            for pch in sorted(projects_rows.keys(), key=_pch_sort_key):
                rows = projects_rows[pch]
                project_count = len(rows)
                project_codes: list[str] = []
                prod_current_values: list[float] = []
                prod_overall_values: list[float] = []
                delivered_month_total = 0.0
                planned_month_total = 0.0
                planned_value_count = 0
                gangs_total = 0
                lost_units_total = 0.0

                def _project_lookup_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
                    return _project_code_key_sets(row)

                def _lookup_with_keys(norm_keys: list[str], compact_keys: list[str], norm_map: Mapping[str, float], compact_map: Mapping[str, float]) -> float | None:
                    for key in norm_keys:
                        if key in norm_map:
                            return norm_map[key]
                    for key in compact_keys:
                        if key in compact_map:
                            return compact_map[key]
                    return None

                for r in sorted(rows, key=lambda x: str(x["project_name"])):
                    norm_keys, compact_keys = _project_lookup_keys(r)
                    code_value = _resolve_project_code(r)
                    if code_value and code_value not in project_codes:
                        project_codes.append(code_value)
                    has_plan = bool(r.get("_stringing_plan_available"))

                    prod_current_val = _lookup_with_keys(norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map)
                    if prod_current_val is not None:
                        prod_current_values.append(prod_current_val)

                    prod_overall_val = _lookup_with_keys(norm_keys, compact_keys, prod_overall_norm_map, prod_overall_compact_map)
                    if prod_overall_val is not None:
                        prod_overall_values.append(prod_overall_val)

                    delivered_month_val = _lookup_with_keys(norm_keys, compact_keys, delivered_current_norm_map, delivered_current_compact_map)
                    if delivered_month_val is not None:
                        delivered_month_total += float(delivered_month_val)

                    planned_month_val = _lookup_with_keys(norm_keys, compact_keys, planned_current_norm_map, planned_current_compact_map)
                    if has_plan and planned_month_val is not None:
                        planned_month_total += float(planned_month_val)
                        planned_value_count += 1

                # Derive Gangs and Lost Units per PCH (stringing)
                try:
                    unit_short = "KM"
                    metric_col = "daily_km"
                    sub = _filter_scope_by_projects(scope_full, rows)
                    if sub.empty:
                        sub = _filter_scope_by_projects(scoped, rows)
                    if not sub.empty:
                        if "gang_name" in sub.columns:
                            gangs_total = (
                                sub["gang_name"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                            )
                        try:
                            overall_map, monthly_map = compute_project_baseline_maps_for(sub, metric_col)
                        except Exception:
                            overall_map, monthly_map = {}, {}
                        proj_col_for_gang = _detect_project_column(sub)
                        if proj_col_for_gang:
                            gang_project_map = (
                                sub[["gang_name", proj_col_for_gang]]
                                .dropna()
                                .astype(str)
                                .rename(columns={proj_col_for_gang: "project_name"})
                                .drop_duplicates(subset=["gang_name", "project_name"], keep="last")
                                .set_index("gang_name")["project_name"]
                                .to_dict()
                            )
                        else:
                            gang_project_map = {}
                        for gang, gdf in sub.groupby("gang_name"):
                            if gdf.empty:
                                continue
                            proj_for_gang = gang_project_map.get(str(gang))
                            ov = overall_map.get(proj_for_gang)
                            mm = monthly_map.get(proj_for_gang, {})
                            try:
                                _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss_for_column(
                                    gdf,
                                    metric_column=metric_col,
                                    loss_max_gap_days=config.loss_max_gap_days,
                                    baseline_per_day=ov,
                                    baseline_by_month=mm,
                                )
                                if pd.notna(loss_val):
                                    lost_units_total += float(loss_val)
                            except Exception:
                                pass
                except Exception:
                    gangs_total = gangs_total or 0
                    lost_units_total = lost_units_total or 0.0

                prod_current_avg = float(sum(prod_current_values) / len(prod_current_values)) if prod_current_values else None
                prod_overall_avg = float(sum(prod_overall_values) / len(prod_overall_values)) if prod_overall_values else None

                fmt_prod_current = f"{prod_current_avg:.2f}" if prod_current_avg is not None else "\u2014"
                fmt_prod_overall = f"{prod_overall_avg:.2f}" if prod_overall_avg is not None else "\u2014"
               
                projects_label = f"Projects: {', '.join(project_codes)}" if project_codes else f"Projects: {project_count}"
                km_delivered_label = round(delivered_month_total, 1)
                if planned_value_count:
                    km_planned_value = round(planned_month_total, 1)
                    km_balance_value = round(max(km_planned_value - km_delivered_label, 0.0), 1)
                    planned_display = f"{km_planned_value:.1f}"
                    balance_display = f"{km_balance_value:.1f}"
                else:
                    planned_display = "No Plan"
                    balance_display = "\u2014"
                pill_components = [
                                    ("projects", dbc.Button(
                                        projects_label,
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "projects"},
                                        className="pch-pill pch-pill-projects mb-1", color="link"
                                    )),
                                    ("productivity", dbc.Button(
                                        f"Productivity (Month/Overall): {fmt_prod_current} / {fmt_prod_overall} KM/day",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "productivity"},
                                        className="pch-pill pch-pill-prod-month mb-1", color="link"
                                    )),
                                    ("totals", dbc.Button(
                                        f"F/S Total Planned / Done / Balance: {planned_display} / {km_delivered_label:.1f} / {balance_display} KM",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "totals"},
                                        className="pch-pill pch-pill-towers mb-1", color="link"
                                    )),
                                    ("gangs", dbc.Button(
                                        f"Gangs: {int(gangs_total):,}",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "gangs"},
                                        className="pch-pill pch-pill-gangs mb-1", color="link"
                                    )),
                                    ("loss", dbc.Button(
                                        f"Lost Units: {lost_units_total:.1f} KM",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "loss"},
                                        className="pch-pill pch-pill-loss mb-1", color="link"
                                    )),
                                ]

                title_component = html.Div(
                    [
                        html.Span(str(pch or "Unassigned"), className="fw-semibold"),
                        html.Div(
                            _filter_pch_header_pills(pill_components, pill_focus),
                            className="pch-pill-group ms-auto d-none d-md-flex",
                        ),
                    ],
                    className="d-flex align-items-center justify-content-between w-100",
                )

                tile_cols = []
                for r in sorted(rows, key=lambda x: str(x["project_name"])):
                    proj_name = str(r["project_name"]).strip()
                    display_code = _resolve_project_code(r)
                    if display_code and _compact_project_token(display_code) != _compact_project_token(proj_name):
                        proj_title = f"{display_code} : {proj_name}"
                    else:
                        proj_title = proj_name

                    raw_code_for_keys = display_code or r.get("project_code") or r.get("project_key") or proj_name
                    proj_code = _compact_project_token(str(raw_code_for_keys or proj_name))

                    tile_summary_children = [
                        html.Div(html.Strong(proj_title), className="mb-2"),
                        html.Div([
                            html.Span("Regional Manager : ", className="text-muted me-1"),
                            dbc.Badge(r.get("regional_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold"),
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Project Manager : ", className="text-muted me-1"),
                            dbc.Badge(r.get("project_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold"),
                        ], className="mb-2"),
                    ]

                    norm_keys, compact_keys = _project_lookup_keys(r)

                    project_scope_cache: pd.DataFrame | None = None

                    def _lazy_scope() -> pd.DataFrame:
                        nonlocal project_scope_cache
                        if project_scope_cache is None:
                            project_scope_cache = _project_scope_for_row(
                                r,
                                primary_scope=scope_full,
                                fallback_scope=scoped,
                            )
                        return project_scope_cache

                    gangs_metric = None
                    loss_metric = None
                    tse_metric = None
                    if focus_metric in {"gangs", "loss", "tse"}:
                        scope_subset = _lazy_scope()
                        if focus_metric == "gangs":
                            gangs_metric = _count_gangs(scope_subset)
                        elif focus_metric == "loss":
                            loss_metric = _compute_project_loss_value(scope_subset, mode="stringing")
                        elif focus_metric == "tse":
                            tse_value, _canon = _resolve_tse_value(
                                norm_keys,
                                compact_keys,
                                tse_norm_lookup,
                                tse_alias_lookup,
                            )
                            if tse_value is not None:
                                tse_metric = int(tse_value)
                            else:
                                tse_metric = _count_tse(scope_subset)
                    prod_current_value = _lookup_with_keys(
                        norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map
                    )
                    prod_overall_value = _lookup_with_keys(
                        norm_keys, compact_keys, prod_overall_norm_map, prod_overall_compact_map
                    )
                    delivered_current_value = _lookup_with_keys(
                        norm_keys, compact_keys, delivered_current_norm_map, delivered_current_compact_map
                    )
                    planned_current_value = _lookup_with_keys(
                        norm_keys, compact_keys, planned_current_norm_map, planned_current_compact_map
                    )
                    po_plan_value = _lookup_with_keys(
                        norm_keys, compact_keys, po_plan_norm_map, po_plan_compact_map
                    )
                    po_done_value = _lookup_with_keys(
                        norm_keys, compact_keys, po_done_norm_map, po_done_compact_map
                    )

                    plan_available = bool(r.get("_stringing_plan_available"))
                    plan_total_value = float(r.get("planned_km", 0.0) or 0.0)
                    plan_has_data = plan_available or plan_total_value > 0
                    if planned_current_value is None and plan_has_data:
                        planned_effective = plan_total_value
                    else:
                        planned_effective = planned_current_value
                    tile_summary_children.extend(
                        _build_tile_metric_rows(
                            mode="stringing",
                            focus_metric=focus_metric,
                            prod_current_value=prod_current_value,
                            prod_overall_value=prod_overall_value,
                            total_current_value=(
                                delivered_current_value if delivered_current_value is not None else r.get("delivered_mt", 0.0)
                            ),
                            total_planned_value=planned_effective if plan_has_data else None,
                            plan_available=plan_has_data,
                            gangs_value=gangs_metric,
                            loss_value=loss_metric,
                            tse_value=tse_metric,
                            po_planned_value=po_plan_value,
                            po_done_value=po_done_value,
                        )
                    )

                    project_token = _component_id_token("proj", proj_name)
                    card_id = {
                        "type": "project-tile-trigger",
                        "mode": "stringing",
                        "project": project_token,
                        "context": tile_context,
                    }
                    tile_metadata[project_token] = {
                        "project": proj_name,
                        "code": display_code or proj_name,
                        "display": proj_title,
                        "mode": "stringing",
                        "pch": str(pch),
                    }

                    tile_body_children = [
                        html.Div(tile_summary_children, className="project-tile-summary"),
                    ]
                    stringing_month_buttons: list[Any] = []
                    available_months = list(r.get("_stringing_plan_months") or [])
                    if available_months:
                        stringing_month_buttons.append(html.Span("Monthly Plan (Stringing) : ", className="me-2"))
                        available_months = available_months[-2:]
                        for ts in available_months:
                            label = ts.strftime("%b %Y")
                            value = ts.strftime("%Y-%m")
                            key_payload = "||".join([
                                "stringing",
                                proj_code or "",
                                value or "",
                                proj_name,
                            ])
                            if use_modal_ids:
                                key_payload = f"{key_payload}||__modal__"
                            stringing_month_buttons.append(
                                dbc.Button(
                                    label,
                                    id={"type": "proj-resp-open", "key": key_payload},
                                    color="link",
                                    className="p-0 me-1",
                                )
                            )
                        tile_body_children.append(html.Div(stringing_month_buttons, className="mb-2"))
                    else:
                        tile_body_children.append(html.Div("Micro Plan not available.", className="text-muted mb-2"))

                    tile_card = dbc.Card(dbc.CardBody(tile_body_children), className="h-100 shadow-sm")

                    tile_cols.append(
                        dbc.Col(
                            html.Div(
                                tile_card,
                                id=card_id,
                                n_clicks=0,
                                className="project-tile-card",
                                role="button",
                                tabIndex=0,
                            ),
                            xs=12,
                            sm=12,
                            md=6,
                            lg=4,
                            className="mb-3",
                        )
                    )
                body_children = (
                    [dbc.Row(tile_cols, className="g-3")]
                    if tile_cols
                    else [html.Div("No projects available.", className="text-muted")]
                )
                pch_sections.append(
                    dbc.AccordionItem(
                        title=title_component,
                        children=body_children,
                        item_id=f"pch-{_slugify_pch(pch)}",
                        className="pch-section mb-2",
                    )
                )
            if not pch_sections:
                pch_sections = _empty_pch_items("No projects match the current filters.")
            return pch_sections, None, tile_metadata

        # --- Erection mode below ---

        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        months_ts = resolve_months(month_list, quick_range)

        if months_ts:
            range_start = pd.Timestamp(min(months_ts)).normalize()
            range_end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
        else:
            today = pd.Timestamp.today().normalize()
            range_start = today.to_period("M").start_time.normalize()
            range_end = (today + pd.offsets.MonthEnd(0)).normalize()

        df_day = data_selector.select("erection")
        scoped = apply_filters(df_day, project_list, months_ts, [])
        try:
            scope_frames, _, _ = _build_scope_frames(
                "erection",
                project_list=project_list,
                gang_list=[],
                months_value=month_list,
                quick_range=quick_range,
                kv_values=None,
                method_values=None,
            )
            scope_full = scope_frames.get("full", pd.DataFrame()).copy()
        except Exception:
            scope_full = pd.DataFrame()
        export_df, _ = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=None,
            search_text=None,
        )
        if not isinstance(export_df, pd.DataFrame):
            export_df = pd.DataFrame(columns=["project_name", "location_no", "tower_weight_mt", "daily_prod_mt", "gang_name", "supervisor_name", "section_incharge_name"])

        df_mp = None
        if has_plan_provider.get("erection"):
            df_mp_frame, _, _, _ = _fetch_monthly_plan("erection")
            if isinstance(df_mp_frame, pd.DataFrame):
                df_mp = df_mp_frame
        # Keep an unfiltered copy to test project-level availability (any month)
        mp_all = df_mp.copy() if isinstance(df_mp, pd.DataFrame) else None
        if isinstance(mp_all, pd.DataFrame):
            if "plan_month" in mp_all.columns:
                mp_all["plan_month"] = pd.to_datetime(
                    mp_all["plan_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
                mp_all["completion_month"] = mp_all["plan_month"]
            elif "completion_month" in mp_all.columns:
                mp_all["completion_month"] = pd.to_datetime(
                    mp_all["completion_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            elif "completion_date" in mp_all.columns:
                mp_all["completion_month"] = pd.to_datetime(
                    mp_all["completion_date"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            else:
                mp_all["completion_month"] = pd.NaT
        # Do not block the modal if Micro Plan is unavailable; proceed with empty frame
        if df_mp is None:
            mp = pd.DataFrame(columns=[
                "project_name", "project_key", "location_no", "entity_type", "entity_name",
                "tower_weight", "pch", "plan_month", "completion_month"
            ])
        else:
            mp = df_mp.copy()

        if "plan_month" in mp.columns:
            mp["plan_month"] = pd.to_datetime(
                mp["plan_month"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            mp["completion_month"] = mp["plan_month"]
        elif "completion_month" in mp.columns:
            mp["completion_month"] = pd.to_datetime(
                mp["completion_month"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        elif "completion_date" in mp.columns:
            mp["completion_month"] = pd.to_datetime(
                mp["completion_date"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        else:
            mp["completion_month"] = pd.NaT

        if months_ts:
            mp = mp[mp["completion_month"].isin(months_ts)].copy()
        if project_list:
            import re as _re
            proj_names_lc = set(str(p).strip().lower() for p in project_list)
            proj_names_compact = set(_re.sub(r"[^a-z0-9]", "", s) for s in proj_names_lc)
            name_lc = (mp.get("project_name", pd.Series([""] * len(mp), index=mp.index)).astype(str).str.strip().str.lower())
            key_lc = (mp.get("project_key", pd.Series([""] * len(mp), index=mp.index)).astype(str).str.strip().str.lower())
            name_compact = name_lc.str.replace(r"[^a-z0-9]", "", regex=True)
            key_compact = key_lc.str.replace(r"[^a-z0-9]", "", regex=True)
            mask_mp = (
                name_lc.isin(proj_names_lc) | key_lc.isin(proj_names_lc) |
                name_compact.isin(proj_names_compact) | key_compact.isin(proj_names_compact)
            )
            mp = mp[mask_mp].copy()
            # Apply same project filter to unfiltered copy used for availability checks
            if isinstance(mp_all, pd.DataFrame) and not mp_all.empty:
                name_lc_all = (mp_all.get("project_name", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                key_lc_all = (mp_all.get("project_key", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                name_compact_all = name_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                key_compact_all = key_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                mask_all = (
                    name_lc_all.isin(proj_names_lc) | key_lc_all.isin(proj_names_lc) |
                    name_compact_all.isin(proj_names_compact) | key_compact_all.isin(proj_names_compact)
                )
                mp_all = mp_all[mask_all].copy()

        # Normalized helper columns
        mp["location_no_norm"] = (mp.get("location_no", pd.Series([""] * len(mp), index=mp.index)).map(_normalize_location))
        mp["project_name_display"] = (mp.get("project_name", pd.Series([""] * len(mp), index=mp.index)).astype(str))
        mp["pch_display"] = (mp.get("pch", pd.Series([""] * len(mp), index=mp.index)).astype(str))
        mp["_tw_"] = pd.to_numeric(mp.get("tower_weight", 0.0), errors="coerce").fillna(0.0)

        # Project-level planned
        planned_mt = mp.groupby(["pch_display", "project_name_display"], dropna=False)["_tw_"].sum()
        planned_nos = (
            mp.dropna(subset=["location_no_norm"])\
              .drop_duplicates(["pch_display", "project_name_display", "location_no_norm"])\
              .groupby(["pch_display", "project_name_display"]).size()
        )

        # Delivered aggregates and meta (by project and location)
        ed = export_df.copy()
        ed["project_name_display"] = ed.get("project_name", "").astype(str)
        ed["location_no_norm"] = ed.get("location_no", "").map(_normalize_location)
        delivered_mt = ed.groupby(["project_name_display"]) ["tower_weight_mt"].sum()
        delivered_nos = (
            ed.dropna(subset=["location_no_norm"])\
              .drop_duplicates(["project_name_display", "location_no_norm"])\
              .groupby(["project_name_display"]).size()
        )
        meta_cols = ["daily_prod_mt", "gang_name", "supervisor_name", "section_incharge_name", "start_date"]
        loc_meta = (
            ed.sort_values("completion_date").drop_duplicates("location_no_norm", keep="last")[
                ["location_no_norm", *[c for c in meta_cols if c in ed.columns]]
            ] if not ed.empty else pd.DataFrame(columns=["location_no_norm", *meta_cols])
        ).set_index("location_no_norm") if not ed.empty else pd.DataFrame()

        # Project meta (regional/project manager, planning engineer) + PCH mapping from Project Details
        try:
            info_df = project_info_provider() if callable(project_info_provider) else None
        except Exception:
            info_df = None
        # Ensure PCH normalizer default exists before we start building lookup maps.
        if "_normalize_pch" not in locals():
            def _normalize_pch(v):
                return str(v or "").strip()
            _PCH_ORDER = ()

        info_name_to_pch: dict[str, str] = {}
        info_code_to_pch: dict[str, str] = {}
        if isinstance(info_df, pd.DataFrame) and not info_df.empty:
            info = info_df.copy()
            info["project_name_display"] = info.get("Project Name", info.get("project_name", "")).astype(str)
            # Prepare normalized key for robust matching (case-insensitive, trimmed)
            info["project_name_norm"] = info["project_name_display"].map(_normalize_lower)
            # Find a PCH column in a forgiving way
            pch_col = None
            for cand in ("PCH", "pch", "PCH Name", "PCHName", "pch_name"):
                if cand in info.columns:
                    pch_col = cand
                    break
            if pch_col is None:
                info["pch"] = ""
                pch_col = "pch"
            # Build compact key map for robust project lookup across datasets (e.g., 'TA418' vs 'TA 418')
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                info_key_map: dict[str, int] = {}
                name_keys = info["project_name_display"].astype(str).map(_compact_code)
                for idx, key in zip(info.index, name_keys):
                    if key and key not in info_key_map:
                        info_key_map[key] = idx
                for code_col in ("project_code", "Project Code"):
                    if code_col in info.columns:
                        code_keys = info[code_col].astype(str).map(_compact_code)
                        for idx, key in zip(info.index, code_keys):
                            if key and key not in info_key_map:
                                info_key_map[key] = idx
            except Exception:
                info_key_map = {}

            if pch_col in info.columns:
                info_name_to_pch = {
                    _normalize_lower(str(row.get("project_name_display", ""))): _normalize_pch(row.get(pch_col, ""))
                    for _, row in info.iterrows()
                    if str(row.get("project_name_display", "")).strip()
                }
                if "project_code" in info.columns:
                    try:
                        info_code_to_pch = {
                            re.sub(r"[^a-z0-9]", "", str(row.get("project_code", "")).strip().lower()): _normalize_pch(row.get(pch_col, ""))
                            for _, row in info.iterrows()
                            if str(row.get("project_code", "")).strip()
                        }
                    except Exception:
                        info_code_to_pch = {}
        else:
            info = pd.DataFrame(columns=["project_name_display", "project_name_norm", "pch", "regional_mgr", "project_mgr", "planning_eng"])
            info_key_map = {}

        # Build hierarchy: PCH -> Projects -> Locations
        # Ensure we always have a PCH value; fall back to project-info mapping if blank
        proj_info_pch = {}
        if not info.empty and "pch" in info.columns:
            proj_info_pch = dict(zip(info["project_name_display"], info["pch"].astype(str)))

        # Import PCH normalizer to canonicalize labels for grouping and display
        try:
            from .pch_normalizer import normalize_pch as _normalize_pch, CANONICAL_PCH_PRIMARY as _PCH_ORDER
        except Exception:
            def _normalize_pch(v):
                return str(v or "").strip()
            _PCH_ORDER = ()

        # Aggregate per (normalized PCH, project) to avoid duplicates when Micro Plan has variant PCHs
        aggregated = {}
        aggregated_by_proj_key: dict[str, tuple[str, str]] = {}
        for (mp_pch, proj), mt in planned_mt.items():
            nos_planned = int(planned_nos.get((mp_pch, proj), 0)) if hasattr(planned_nos, 'get') else 0
            # Robust lookup of Project Details row using normalized name; if not found, try compact code match
            proj_norm = _normalize_lower(proj)
            # Use normalized-name match, then compact-key map fallback
            meta = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
            if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                try:
                    import re as _re
                    def _compact_code(s: str) -> str:
                        return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                    target_key = _compact_code(str(proj))
                    if target_key and target_key in info_key_map:
                        meta = info.loc[[info_key_map[target_key]]]
                except Exception:
                    pass
            raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
            # Determine PCH solely from Project Details; if unrecognized, keep original
            pch_label = (_normalize_pch(raw_pch) or str(raw_pch or "").strip())
            # Derive display heading and identity key using code when available
            try:
                proj_code = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                    meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                )) if isinstance(meta, pd.DataFrame) else ""
            except Exception:
                proj_code = ""
            proj_display_name = str(meta.get("project_name_display", pd.Series([proj])).iloc[0]) if isinstance(meta, pd.DataFrame) and not meta.empty else str(proj)
            proj_display = f"{proj_code} : {proj_display_name}".strip(" :") if proj_code else proj_display_name
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                proj_key = _compact_code(proj_code) or _compact_code(proj_display_name) or _compact_code(proj)
            except Exception:
                proj_key = proj_norm
            key = (pch_label, proj_display)
            if key not in aggregated:
                mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
                nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
                aggregated[key] = {
                    "pch": pch_label,
                    "project_name": proj_display,
                    "project_code": proj_code,
                    "planned_mt": 0.0,
                    "delivered_mt": mt_del,
                    "planned_nos": 0,
                    "delivered_nos": nos_del,
                    "regional_mgr": (meta["regional_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns else ""),
                    "project_mgr": (meta["project_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns else ""),
                    "planning_eng": (meta["planning_eng"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns else ""),
                }
            aggregated[key]["planned_mt"] += float(mt)
            aggregated[key]["planned_nos"] += nos_planned
            # Track by compact project key to prevent duplicates across sources
            try:
                aggregated_by_proj_key[proj_key] = key
            except Exception:
                pass

        # Also include projects that only have delivered data (no Micro Plan rows)
        try:
            delivered_projects = list(getattr(delivered_mt, 'index', []))
        except Exception:
            delivered_projects = []
        for proj in map(lambda x: str(x), delivered_projects):
            if not proj or not str(proj).strip():
                continue
            proj_norm = _normalize_lower(proj)
            # If this project already exists from the planned aggregation, update delivered values instead of duplicating
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                # Prefer code if we can resolve it from info
                meta_lookup = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
                proj_code_lookup = (meta_lookup.get("project_code").iloc[0] if (isinstance(meta_lookup, pd.DataFrame) and not meta_lookup.empty and "project_code" in meta_lookup.columns) else (
                    meta_lookup.get("Project Code").iloc[0] if (isinstance(meta_lookup, pd.DataFrame) and not meta_lookup.empty and "Project Code" in meta_lookup.columns) else ""
                )) if isinstance(meta_lookup, pd.DataFrame) else ""
                proj_key = _compact_code(proj_code_lookup) or _compact_code(proj)
            except Exception:
                proj_key = proj_norm
            if proj_key in aggregated_by_proj_key:
                try:
                    existing_key = aggregated_by_proj_key[proj_key]
                    mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
                    nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
                    aggregated[existing_key]["delivered_mt"] = mt_del
                    aggregated[existing_key]["delivered_nos"] = nos_del
                    continue
                except Exception:
                    # fallback to normal add path
                    pass
            meta = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
            if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                # Try compact-key lookup
                if proj_key and proj_key in info_key_map:
                    meta = info.loc[[info_key_map[proj_key]]]
            raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
            pch_label = (
                _normalize_pch(raw_pch)
                or str(raw_pch or "").strip()
            )
            # Build display and project code for heading
            try:
                proj_code2 = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                    meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                )) if isinstance(meta, pd.DataFrame) else ""
            except Exception:
                proj_code2 = ""
            proj_display_name2 = str(meta.get("project_name_display", pd.Series([proj])).iloc[0]) if isinstance(meta, pd.DataFrame) and not meta.empty else str(proj)
            resolved_code2 = (
                _extract_project_code(proj_code2)
                or _extract_project_code(proj_display_name2)
                or _extract_project_code(proj)
            )
            if not resolved_code2:
                resolved_code2 = proj_code2
            proj_display2 = f"{resolved_code2} : {proj_display_name2}".strip(" :") if resolved_code2 else proj_display_name2
            key = (pch_label, proj_display2)
            if key in aggregated:
                continue
            mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
            nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
            aggregated[key] = {
                "pch": pch_label,
                "project_name": proj_display2,
                "project_code": resolved_code2 or proj_code2,
                "code": resolved_code2 or proj_code2,
                "planned_mt": 0.0,
                "delivered_mt": mt_del,
                "planned_nos": 0,
                "delivered_nos": nos_del,
                "regional_mgr": (meta["regional_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns else ""),
                "project_mgr": (meta["project_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns else ""),
                "planning_eng": (meta["planning_eng"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns else ""),
            }
            try:
                aggregated_by_proj_key[proj_key] = key
            except Exception:
                pass

        # Finally, include any projects present only in Project Details (no delivered and no MP)
        try:
            # Use selected projects filter if provided; otherwise consider all info rows
            info_iter = info.copy()
            if project_list:
                pl = set(str(p).strip().lower() for p in project_list)
                info_iter = info_iter[info_iter["project_name_display"].astype(str).str.strip().str.lower().isin(pl)]
        except Exception:
            info_iter = info
        for _, meta_row in info_iter.iterrows():
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                proj = str(meta_row.get("project_name_display", "")).strip()
                code_raw = str(meta_row.get("project_code", meta_row.get("Project Code", "")))
                resolved_code = _extract_project_code(code_raw) or _extract_project_code(proj)
                proj_key = _compact_code(resolved_code or code_raw) or _compact_code(proj)
                if not proj or proj_key in aggregated_by_proj_key:
                    continue
                raw_pch = str(meta_row.get(pch_col, "")) if pch_col in meta_row.index else ""
                pch_label = (_normalize_pch(raw_pch) or str(raw_pch or "").strip())
                proj_display = f"{resolved_code} : {proj}".strip(" :") if resolved_code else proj
                key = (pch_label, proj_display)
                aggregated[key] = {
                    "pch": pch_label,
                    "project_name": proj_display,
                    "project_code": resolved_code or code_raw,
                    "code": resolved_code or code_raw,
                    "planned_mt": 0.0,
                    "delivered_mt": 0.0,
                    "planned_nos": 0,
                    "delivered_nos": 0,
                    "regional_mgr": str(meta_row.get("regional_mgr", "")),
                    "project_mgr": str(meta_row.get("project_mgr", "")),
                    "planning_eng": str(meta_row.get("planning_eng", "")),
                }
                aggregated_by_proj_key[proj_key] = key
            except Exception:
                continue

        # Build rows grouped by normalized PCH
        projects_rows: dict[str, list[dict]] = {}
        for (_pch_label, _proj), rec in aggregated.items():
            rec["planned_mt"] = round(float(rec["planned_mt"]), 1)
            rec["delivered_mt"] = round(float(rec["delivered_mt"]), 1)
            projects_rows.setdefault(_pch_label, []).append(rec)

        current_month_ts = range_end.to_period("M").to_timestamp()

        def _compact_key(value: str) -> str:
            text = str(value or "").strip().lower()
            if not text:
                return ""
            return re.sub(r"[^a-z0-9]", "", text)

        def _build_metric_lookup(source: dict[str, float] | None) -> tuple[dict[str, float], dict[str, float]]:
            norm_map: dict[str, float] = {}
            compact_map: dict[str, float] = {}
            if not source:
                return norm_map, compact_map
            for proj_name, raw_value in source.items():
                text = str(proj_name or "").strip()
                if not text:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if pd.isna(value):
                    continue
                norm_key = _normalize_lower(text)
                if norm_key and norm_key not in norm_map:
                    norm_map[norm_key] = value
                compact_key = _compact_key(text)
                if compact_key and compact_key not in compact_map:
                    compact_map[compact_key] = value
            return norm_map, compact_map

        prod_current_norm_map: dict[str, float] = {}
        prod_current_compact_map: dict[str, float] = {}
        prod_history_norm_map: dict[str, float] = {}
        prod_history_compact_map: dict[str, float] = {}
        towers_current_norm_map: dict[str, int] = {}
        towers_current_compact_map: dict[str, int] = {}
        towers_planned_norm_map: dict[str, int] = {}
        towers_planned_compact_map: dict[str, int] = {}

        if isinstance(df_day, pd.DataFrame) and not df_day.empty:
            day_filtered = df_day.copy()
            if project_list and "project_name" in day_filtered.columns:
                project_filter_values = [str(p).strip() for p in project_list if str(p).strip()]
                if project_filter_values:
                    day_filtered = day_filtered[
                        day_filtered["project_name"].astype(str).str.strip().isin(project_filter_values)
                    ]
            if not day_filtered.empty and {"month", "daily_prod_mt", "project_name"}.issubset(day_filtered.columns):
                day_filtered = day_filtered.copy()
                day_filtered["month"] = pd.to_datetime(day_filtered["month"], errors="coerce")
                day_filtered["project_name"] = day_filtered["project_name"].astype(str).str.strip()
                day_filtered = day_filtered.dropna(subset=["month", "daily_prod_mt", "project_name"])

                current_scope = day_filtered[day_filtered["month"] == current_month_ts]
                if not current_scope.empty:
                    current_payload = {
                        "kind": "erection-current-baseline",
                        "projects": sorted(project_list),
                        "months": [ts.isoformat() for ts in months_ts],
                        "month": current_month_ts.isoformat(),
                        "rows": int(len(current_scope.index)),
                        "version": int(df_day.attrs.get("_appdata_version", 0)),
                    }
                    current_token = f"erection-baseline::{_hash_cache_payload(current_payload)}"

                    def _compute_current_prod() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                        return compute_project_baseline_maps_for(current_scope, "daily_prod_mt")

                    prod_current_raw, _ = _cached_global_result(
                        current_token,
                        _compute_current_prod,
                        clone=_clone_baseline_result,
                    )
                    prod_current_norm_map, prod_current_compact_map = _build_metric_lookup(prod_current_raw)

                history_scope = day_filtered[day_filtered["month"] < current_month_ts]
                if not history_scope.empty:
                    history_payload = {
                        "kind": "erection-history-baseline",
                        "projects": sorted(project_list),
                        "months": [ts.isoformat() for ts in months_ts],
                        "month": current_month_ts.isoformat(),
                        "rows": int(len(history_scope.index)),
                        "version": int(df_day.attrs.get("_appdata_version", 0)),
                    }
                    history_token = f"erection-baseline::{_hash_cache_payload(history_payload)}"

                    def _compute_history_prod() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                        return compute_project_baseline_maps_for(history_scope, "daily_prod_mt")

                    prod_history_raw, _ = _cached_global_result(
                        history_token,
                        _compute_history_prod,
                        clone=_clone_baseline_result,
                    )
                    prod_history_norm_map, prod_history_compact_map = _build_metric_lookup(prod_history_raw)

        if isinstance(ed, pd.DataFrame) and not ed.empty and "completion_date" in ed.columns:
            ed_for_towers = ed.copy()
            ed_for_towers["completion_month"] = pd.to_datetime(ed_for_towers["completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            ed_for_towers = ed_for_towers.dropna(subset=["completion_month"])
            ed_for_towers["project_name_display"] = ed_for_towers["project_name_display"].astype(str).str.strip()
            ed_current = ed_for_towers[ed_for_towers["completion_month"] == current_month_ts]
            if not ed_current.empty and "location_no_norm" in ed_current.columns:
                towers_series = (
                    ed_current.dropna(subset=["location_no_norm"])
                              .drop_duplicates(["project_name_display", "location_no_norm"])
                              .groupby("project_name_display")
                              .size()
                )
                for proj_name, count in towers_series.items():
                    text = str(proj_name or "").strip()
                    if not text:
                        continue
                    norm_key = _normalize_lower(text)
                    if norm_key and norm_key not in towers_current_norm_map:
                        towers_current_norm_map[norm_key] = int(count)
                    compact_key = _compact_key(text)
                    if compact_key and compact_key not in towers_current_compact_map:
                        towers_current_compact_map[compact_key] = int(count)

        if isinstance(mp, pd.DataFrame) and not mp.empty and "completion_month" in mp.columns:
            mp_current = mp[mp["completion_month"] == current_month_ts].copy()
            if not mp_current.empty and "location_no_norm" in mp_current.columns:
                planned_series = (
                    mp_current.dropna(subset=["location_no_norm"])
                              .drop_duplicates(["project_name_display", "location_no_norm"])
                              .groupby("project_name_display")
                              .size()
                )
                for proj_name, count in planned_series.items():
                    text = str(proj_name or "").strip()
                    if not text:
                        continue
                    norm_key = _normalize_lower(text)
                    if norm_key and norm_key not in towers_planned_norm_map:
                        towers_planned_norm_map[norm_key] = int(count)
                    compact_key = _compact_key(text)
                    if compact_key and compact_key not in towers_planned_compact_map:
                        towers_planned_compact_map[compact_key] = int(count)

        def _project_lookup_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
            return _project_code_key_sets(row)

        def _lookup_with_key(
            norm_keys: list[str],
            compact_keys: list[str],
            norm_map: dict[str, float],
            compact_map: dict[str, float],
        ) -> tuple[float | None, str | None]:
            for key in norm_keys:
                if key in norm_map:
                    return norm_map[key], key
            for key in compact_keys:
                if key in compact_map:
                    return compact_map[key], key
            return None, None

        def _project_locations(project_name: str) -> list[dict]:
            # Planned per location (tower weight) for the project
            # project_name in the aggregated rows may be "CODE : NAME"; match using NAME part
            pname = str(project_name)
            base_name = pname.split(" : ", 1)[1] if " : " in pname else pname
            mp_proj = mp[mp["project_name_display"].astype(str) == str(base_name)].copy()
            planned_loc = mp_proj.groupby("location_no_norm")["_tw_"].sum().rename("planned_mt") if not mp_proj.empty else pd.Series(dtype=float)
            ed_proj = ed[ed["project_name_display"].astype(str) == str(base_name)].copy()
            delivered_loc = ed_proj.groupby("location_no_norm")["tower_weight_mt"].sum().rename("delivered_mt") if not ed_proj.empty else pd.Series(dtype=float)
            keys = set(planned_loc.index.tolist()) | set(delivered_loc.index.tolist())
            out = []
            for loc in sorted(k for k in keys if k):
                meta = loc_meta.loc[loc] if (isinstance(loc_meta, pd.DataFrame) and (loc in getattr(loc_meta, 'index', []))) else None
                out.append({
                    "location_no": loc,
                    "planned_mt": round(float(planned_loc.get(loc, 0.0) or 0.0), 1),
                    "delivered_mt": round(float(delivered_loc.get(loc, 0.0) or 0.0), 1),
                    "daily_prod_mt": (float(meta["daily_prod_mt"]) if (isinstance(meta, pd.Series) and "daily_prod_mt" in meta) else None),
                    "gang_name": (str(meta["gang_name"]) if (isinstance(meta, pd.Series) and "gang_name" in meta) else ""),
                    "supervisor_name": (str(meta["supervisor_name"]) if (isinstance(meta, pd.Series) and "supervisor_name" in meta) else ""),
                    "section_incharge_name": (str(meta["section_incharge_name"]) if (isinstance(meta, pd.Series) and "section_incharge_name" in meta) else ""),
                })
            return out

        # Order PCH sections: canonical order first, then alphabetical; keep empty last
        def _pch_sort_key(name: str) -> tuple[int, str]:
            if not str(name or "").strip():
                return (2, "")
            try:
                idx = list(_PCH_ORDER).index(name)
                return (0, f"{idx:03d}")
            except ValueError:
                return (1, str(name))

        pch_sections = []
        for pch in sorted(projects_rows.keys(), key=_pch_sort_key):
            rows = projects_rows[pch]

            project_count = len(rows)
            project_codes: list[str] = []
            prod_current_values: list[float] = []
            prod_history_values: list[float] = []
            towers_delivered_total = 0
            towers_planned_total = 0
            towers_delivered_keys: set[str] = set()
            towers_planned_keys: set[str] = set()

            for r in sorted(rows, key=lambda x: str(x["project_name"])):
                norm_keys, compact_keys = _project_lookup_keys(r)
                code_value = _resolve_project_code(r)
                if code_value and code_value not in project_codes:
                    project_codes.append(code_value)

                prod_current_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map
                )
                if prod_current_value is not None:
                    prod_current_values.append(prod_current_value)

                prod_history_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_history_norm_map, prod_history_compact_map
                )
                if prod_history_value is not None:
                    prod_history_values.append(prod_history_value)

                tower_value, tower_key = _lookup_with_key(
                    norm_keys, compact_keys, towers_current_norm_map, towers_current_compact_map
                )
                if tower_value is not None and tower_key and tower_key not in towers_delivered_keys:
                    towers_delivered_keys.add(tower_key)
                    towers_delivered_total += int(tower_value)

                tower_plan_value, tower_plan_key = _lookup_with_key(
                    norm_keys, compact_keys, towers_planned_norm_map, towers_planned_compact_map
                )
                if tower_plan_value is not None and tower_plan_key and tower_plan_key not in towers_planned_keys:
                    towers_planned_keys.add(tower_plan_key)
                    towers_planned_total += int(tower_plan_value)

            prod_current_avg = (
                float(sum(prod_current_values) / len(prod_current_values)) if prod_current_values else None
            )
            prod_history_avg = (
                float(sum(prod_history_values) / len(prod_history_values)) if prod_history_values else None
            )
            fmt_prod_current = f"{prod_current_avg:.2f}" if prod_current_avg is not None else "\u2014"
            fmt_prod_history = f"{prod_history_avg:.2f}" if prod_history_avg is not None else "\u2014"
            projects_label = (
                f"Projects: {', '.join(project_codes)}" if project_codes else f"Projects: {project_count}"
            )
            towers_delivered_label = int(towers_delivered_total)
            towers_planned_label = int(towers_planned_total)
            # Derive gangs and lost units per PCH (erection)
            gangs_total = 0
            lost_units_total = 0.0
            try:
                unit_short = "MT"
                metric_col = "daily_prod_mt"
                sub = _filter_scope_by_projects(scope_full, rows)
                if sub.empty:
                    sub = _filter_scope_by_projects(scoped, rows)

                if not sub.empty:
                    if "gang_name" in sub.columns:
                        gangs_total = (
                            sub["gang_name"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                        )
                    try:
                        overall_map, monthly_map = compute_project_baseline_maps(sub)
                    except Exception:
                        overall_map, monthly_map = {}, {}
                    proj_col_for_gang = _detect_project_column(sub)
                    if proj_col_for_gang:
                        gang_project_map = (
                            sub[["gang_name", proj_col_for_gang]]
                            .dropna()
                            .astype(str)
                            .rename(columns={proj_col_for_gang: "project_name"})
                            .drop_duplicates(subset=["gang_name", "project_name"], keep="last")
                            .set_index("gang_name")["project_name"]
                            .to_dict()
                        )
                    else:
                        gang_project_map = {}
                    for gang, gdf in sub.groupby("gang_name"):
                        if gdf.empty:
                            continue
                        proj_for_gang = gang_project_map.get(str(gang))
                        ov = overall_map.get(proj_for_gang)
                        mm = monthly_map.get(proj_for_gang, {})
                        try:
                            _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss(
                                gdf,
                                loss_max_gap_days=config.loss_max_gap_days,
                                baseline_mt_per_day=ov,
                                baseline_by_month=mm,
                            )
                            if pd.notna(loss_val):
                                lost_units_total += float(loss_val)
                        except Exception:
                            pass
            except Exception:
                gangs_total = gangs_total or 0
                lost_units_total = lost_units_total or 0.0

            towers_balance_label = max(towers_planned_label - towers_delivered_label, 0)
            pill_components = [
                                ("projects", dbc.Button(
                                    projects_label,
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "projects"},
                                    className="pch-pill pch-pill-projects me-2 mb-1", color="link"
                                )),
                                ("productivity", dbc.Button(
                                    f"Productivity (Month/Overall): {fmt_prod_current} / {fmt_prod_history} MT/day",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "productivity"},
                                    className="pch-pill pch-pill-prod-month me-2 mb-1", color="link"
                                )),
                                ("totals", dbc.Button(
                                    f"F/S Total Planned / Done / Balance: {towers_planned_label} / {towers_delivered_label} / {towers_balance_label}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "totals"},
                                    className="pch-pill pch-pill-towers me-2 mb-1", color="link"
                                )),
                                ("gangs", dbc.Button(
                                    f"Gangs: {int(gangs_total):,}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "gangs"},
                                    className="pch-pill pch-pill-prod-overall me-2 mb-1", color="link", n_clicks=0, 
                                )),
                                ("loss", dbc.Button(
                                    f"Lost Units: {lost_units_total:.1f} MT",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "loss"},
                                    className="pch-pill pch-pill-loss me-2 mb-1", color="link", n_clicks=0, 
                                )),
                            ]

            header_pills = _filter_pch_header_pills(pill_components, pill_focus)

            header = dbc.Row(
                [
                    dbc.Col(html.H6(str(pch), className="mb-0"), md=3),
                    dbc.Col(
                        html.Div(header_pills, className="pch-pill-group justify-content-md-end"),
                        md=9,
                    ),
                ],
                className="pch-header align-items-center py-2",
            )

            # Nested tiles for projects within this PCH
            project_items = []  # legacy; no longer used
            tile_cols = []
            for r in sorted(rows, key=lambda x: str(x["project_name"])):
                proj_name = str(r["project_name"]).strip()
                display_code = _resolve_project_code(r)
                # (legacy header removed)
                # Detect if Micro Plan rows exist for this project using robust name/code matching
                import re as _re
                # proj_name may be in format "CODE : NAME"; split for robust matching
                full_name = str(proj_name).strip()
                if " : " in full_name:
                    code_part, name_part = full_name.split(" : ", 1)
                else:
                    code_part, name_part = "", full_name
                sel_name = name_part.strip().lower()
                sel_code = _re.sub(r"[^a-z0-9]", "", code_part.strip().lower()) if code_part else ""
                sel_compact = _re.sub(r"[^a-z0-9]", "", sel_name)

                def _has_project(frame: pd.DataFrame | None) -> bool:
                    try:
                        if not isinstance(frame, pd.DataFrame) or frame.empty:
                            return False
                        name_lc = (frame.get("project_name", pd.Series([""] * len(frame), index=frame.index))
                                   .astype(str).str.strip().str.lower())
                        key_lc = (frame.get("project_key", pd.Series([""] * len(frame), index=frame.index))
                                  .astype(str).str.strip().str.lower())
                        name_compact = name_lc.str.replace(r"[^a-z0-9]", "", regex=True)
                        key_compact = key_lc.str.replace(r"[^a-z0-9]", "", regex=True)
                        mask = (
                            (name_lc == sel_name) | (key_lc == sel_name) |
                            (name_compact == sel_compact) | (key_compact == sel_compact) |
                            ((sel_code != "") & (key_compact == sel_code))
                        )
                        return bool(mask.any())
                    except Exception:
                        return False

                has_mp = _has_project(mp)
                has_mp_any = _has_project(mp_all)

                # Build Section Incharge -> Supervisor summary (planned/delivered) using Micro Plan responsibilities first
                def _section_supervisor_summary(project: str):
                    mp_proj = mp[mp["project_name_display"].astype(str) == str(project)].copy()
                    # normalize labels and fields
                    std_map = {
                        "gangs": "Gang", "gang": "Gang",
                        "section incharges": "Section Incharge", "section incharge": "Section Incharge", "section in-charge": "Section Incharge",
                        "supervisors": "Supervisor", "supervisor": "Supervisor",
                    }
                    et = mp_proj.get("entity_type", "").astype(str).str.lower()
                    mp_proj["entity_type_std"] = et.map(lambda v: std_map.get(v, v.title()))
                    mp_proj["location_no_norm"] = mp_proj.get("location_no", "").map(_normalize_location)
                    mp_proj["tower_weight_val"] = pd.to_numeric(mp_proj.get("tower_weight", 0.0), errors="coerce").fillna(0.0)

                    # completion markers
                    completed: set[tuple[str, str]] = set()
                    try:
                        if callable(responsibilities_completion_provider):
                            completed = set(responsibilities_completion_provider())
                    except Exception:
                        completed = set()
                    proj_lc = str(project).strip().lower()

                    # collapse to location granularity to avoid double-counting
                    # section map per location
                    sec_map = (
                        mp_proj[mp_proj["entity_type_std"] == "Section Incharge"][
                            ["location_no_norm", "entity_name"]
                        ]
                        .dropna()
                        .drop_duplicates("location_no_norm", keep="last")
                        .rename(columns={"entity_name": "section"})
                    )
                    sup_map = (
                        mp_proj[mp_proj["entity_type_std"] == "Supervisor"][
                            ["location_no_norm", "entity_name"]
                        ]
                        .dropna()
                        .drop_duplicates("location_no_norm", keep="last")
                        .rename(columns={"entity_name": "supervisor"})
                    )

                    loc = (
                        mp_proj.groupby("location_no_norm", as_index=False)["tower_weight_val"].max()
                    )
                    loc["is_completed"] = [
                        (proj_lc, _normalize_lower(loc_id)) in completed for loc_id in loc["location_no_norm"]
                    ]
                    loc = loc.merge(sec_map, on="location_no_norm", how="left").merge(sup_map, on="location_no_norm", how="left")
                    loc["section"] = loc["section"].fillna("Unassigned").astype(str)
                    loc["supervisor"] = loc["supervisor"].fillna("Unassigned").astype(str)
                    loc["delivered_mt_val"] = np.where(loc["is_completed"], loc["tower_weight_val"], 0.0)
                    loc["delivered_n"] = np.where(loc["is_completed"], 1, 0)

                    # section aggregates
                    sec_g = loc.groupby("section", as_index=False).agg(
                        planned_nos=("location_no_norm", "nunique"),
                        planned_mt=("tower_weight_val", "sum"),
                        delivered_nos=("delivered_n", "sum"),
                        delivered_mt=("delivered_mt_val", "sum"),
                    )
                    result = {
                        str(row["section"]): {
                            "planned_nos": int(row["planned_nos"]),
                            "planned_mt": float(row["planned_mt"]),
                            "delivered_nos": int(row["delivered_nos"]),
                            "delivered_mt": float(row["delivered_mt"]),
                            "supervisors": [],
                        }
                        for _, row in sec_g.iterrows()
                    }

                    # supervisor aggregates within each section
                    for sec_name, sub in loc.groupby("section"):
                        sup_g = sub.groupby("supervisor", as_index=False).agg(
                            planned_nos=("location_no_norm", "nunique"),
                            planned_mt=("tower_weight_val", "sum"),
                            delivered_nos=("delivered_n", "sum"),
                            delivered_mt=("delivered_mt_val", "sum"),
                        )
                        for _, row in sup_g.iterrows():
                            result.setdefault(str(sec_name), {"planned_nos":0,"planned_mt":0.0,"delivered_nos":0,"delivered_mt":0.0,"supervisors":[]})
                            result[str(sec_name)]["supervisors"].append({
                                "name": str(row["supervisor"]),
                                "planned_nos": int(row["planned_nos"]),
                                "planned_mt": float(row["planned_mt"]),
                                "delivered_nos": int(row["delivered_nos"]),
                                "delivered_mt": float(row["delivered_mt"]),
                            })
                    return result

                sections_children = []
                if has_mp:
                    summary = _section_supervisor_summary(r["project_name"])
                    for sec_name in sorted(summary.keys()):
                        sec_data = summary[sec_name]
                        sup_children = []
                        for sup_item in sorted(sec_data["supervisors"], key=lambda x: x["name"]):
                            sup_children.append(html.Div([
                                html.Span(sup_item["name"], className="me-2 fw-semibold"),
                                dbc.Badge(f"Nos {sup_item['delivered_nos']}/{sup_item['planned_nos']}", color="primary", className="me-2"),
                                dbc.Badge(f"MT {sup_item['delivered_mt']:.1f}/{sup_item['planned_mt']:.1f}", color="dark"),
                            ], className="mb-1"))
                        sections_children.append(html.Div([
                            html.Div([
                                html.Span("Section Incharge: ", className="text-muted"), html.Strong(sec_name),
                                html.Span(" ", className="me-1"),
                                dbc.Badge(f"Nos {sec_data['delivered_nos']}/{sec_data['planned_nos']}", color="primary", className="ms-2 me-2"),
                                dbc.Badge(f"MT {sec_data['delivered_mt']:.1f}/{sec_data['planned_mt']:.1f}", color="dark"),
                            ], className="mb-2"),
                            *sup_children
                        ], className="mb-3"))

                # (legacy accordion meta/details removed)

                # Build the grid tile representation used for the new layout
                key = f"{pch}::{proj_name}"
                raw_code = display_code or r.get("project_code") or r.get("project_key") or proj_name
                proj_code = _compact_project_token(str(raw_code))
                tile_summary_children = [
                    html.Div(html.Strong(proj_name), className="mb-2"),
                    html.Div([
                        html.Span("Regional Manager : ", className="text-muted me-1"),
                        dbc.Badge(r.get("regional_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Project Manager : ", className="text-muted me-1"),
                        dbc.Badge(r.get("project_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold")
                    ], className="mb-2"),
                ]
                # Build month-aware responsibilities openers
                def _month_buttons_for_project() -> list:
                    # derive available months from mp_all for this project
                    months_vals: list = []
                    try:
                        if isinstance(mp_all, pd.DataFrame) and not mp_all.empty:
                            # Filter mp_all to this project using same robust matcher
                            name_lc_all = (mp_all.get("project_name", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                            key_lc_all = (mp_all.get("project_key", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                            name_compact_all = name_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                            key_compact_all = key_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                            mask_all = (
                                (name_lc_all == sel_name) | (key_lc_all == sel_name) |
                                (name_compact_all == sel_compact) | (key_compact_all == sel_compact) |
                                ((sel_code != "") & (key_compact_all == sel_code))
                            )
                            sub = mp_all.loc[mask_all].copy()
                            if "completion_month" not in sub.columns:
                                if "plan_month" in sub.columns:
                                    try:
                                        sub["completion_month"] = pd.to_datetime(sub["plan_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                                    except Exception:
                                        sub["completion_month"] = pd.NaT
                                elif "completion_date" in sub.columns:
                                    try:
                                        sub["completion_month"] = pd.to_datetime(sub["completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                                    except Exception:
                                        sub["completion_month"] = pd.NaT
                                else:
                                    sub["completion_month"] = pd.NaT
                            months_vals = sorted({pd.Timestamp(v) for v in sub["completion_month"] if pd.notna(v)})
                    except Exception:
                        months_vals = []
                    # render buttons
                    out: list = []
                    if months_vals:
                        out.append(html.Span("Monthly Plan (Erection) : ", className="me-2"))
                        # cap to last 2 months only
                        months_vals = sorted(months_vals)
                        months_vals = months_vals[-2:]
                        for ts in months_vals:
                            label = ts.strftime("%b %Y")
                            value = ts.strftime("%Y-%m")
                            key_payload = "||".join([
                                "erection",
                                proj_code or "",
                                value or "",
                                name_part.strip(),
                            ])
                            if use_modal_ids:
                                key_payload = f"{key_payload}||__modal__"
                            out.append(
                                dbc.Button(
                                    label,
                                    id={"type": "proj-resp-open", "key": key_payload},
                                    color="link",
                                    className="p-0 me-1",
                                )
                            )
                    return out

                # Project-level KPI highlights aligned with the PCH header metrics
                norm_keys, compact_keys = _project_lookup_keys(r)
                prod_current_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map
                )
                prod_overall_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_history_norm_map, prod_history_compact_map
                )
                towers_current_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, towers_current_norm_map, towers_current_compact_map
                )
                towers_planned_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, towers_planned_norm_map, towers_planned_compact_map
                )

                project_scope_cache: pd.DataFrame | None = None

                def _lazy_scope() -> pd.DataFrame:
                    nonlocal project_scope_cache
                    if project_scope_cache is None:
                        project_scope_cache = _project_scope_for_row(
                            r,
                            primary_scope=scope_full,
                            fallback_scope=scoped,
                        )
                    return project_scope_cache

                gangs_metric = None
                loss_metric = None
                if focus_metric in {"gangs", "loss"}:
                    scope_subset = _lazy_scope()
                    if focus_metric == "gangs":
                        gangs_metric = _count_gangs(scope_subset)
                    elif focus_metric == "loss":
                        loss_metric = _compute_project_loss_value(scope_subset, mode="erection")

                tile_summary_children.extend(
                    _build_tile_metric_rows(
                        mode="erection",
                        focus_metric=focus_metric,
                        prod_current_value=prod_current_value,
                        prod_overall_value=prod_overall_value,
                        total_current_value=towers_current_value,
                        total_planned_value=towers_planned_value,
                        gangs_value=gangs_metric,
                        loss_value=loss_metric,
                    )
                )
                project_token = _component_id_token("proj", proj_name)
                card_id = {
                    "type": "project-tile-trigger",
                    "mode": "erection",
                    "project": project_token,
                    "context": tile_context,
                }
                tile_metadata[project_token] = {
                    "project": proj_name,
                    "code": display_code or proj_name,
                    "display": proj_name,
                    "mode": "erection",
                    "pch": str(pch),
                }

                tile_body_children = [
                    html.Div(tile_summary_children, className="project-tile-summary"),
                ]
                month_buttons = _month_buttons_for_project()
                if month_buttons:
                    tile_body_children.append(html.Div(month_buttons, className="mb-1"))
                else:
                    tile_body_children.append(html.Div("Micro Plan not available.", className="text-muted"))

                tile_card = dbc.Card(dbc.CardBody(tile_body_children), className="h-100 shadow-sm")

                tile_cols.append(
                    dbc.Col(
                        html.Div(
                            tile_card,
                            id=card_id,
                            n_clicks=0,
                            className="project-tile-card",
                            role="button",
                            tabIndex=0,
                        ),
                        xs=12,
                        sm=12,
                        md=6,
                        lg=4,
                        className="mb-3"
                    )
                )

            body_children = (
                [dbc.Row(tile_cols, className="g-3")]
                if tile_cols
                else [html.Div("No projects available.", className="text-muted")]
            )
            towers_balance_label = max(towers_planned_label - towers_delivered_label, 0)
            pill_components = [
                                ("projects", dbc.Button(
                                    projects_label,
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "projects"},
                                    className="pch-pill pch-pill-projects me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("productivity", dbc.Button(
                                    f"Productivity / Historical Avg: {fmt_prod_current} / {fmt_prod_history} MT/day",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "productivity"},
                                    className="pch-pill pch-pill-prod-month me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("totals", dbc.Button(
                                    f"F/S Total Planned / Done / Balance: {towers_planned_label} / {towers_delivered_label} / {towers_balance_label}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "totals"},
                                    className="pch-pill pch-pill-towers me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("gangs", dbc.Button(
                                    f"Gangs: {int(gangs_total):,}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "gangs"},
                                    className="pch-pill pch-pill-gangs me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("loss", dbc.Button(
                                    f"Lost Units: {lost_units_total:.1f} MT",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "loss"},
                                    className="pch-pill pch-pill-loss me-2 mb-1", color="link", n_clicks=0
                                )),
                            ]

            title_component = html.Div(
                [
                    html.Span(str(pch or "Unassigned"), className="fw-semibold"),
                    html.Div(
                        _filter_pch_header_pills(pill_components, pill_focus),
                        className="pch-pill-group ms-auto d-none d-md-flex",
                    ),
                ],
                className="d-flex align-items-center justify-content-between w-100",
            )
            pch_sections.append(
                dbc.AccordionItem(
                    title=title_component,
                    children=body_children,
                    item_id=f"pch-{_slugify_pch(pch)}",
                    className="pch-section mb-2",
                )
            )

        if not pch_sections:
            pch_sections = _empty_pch_items("No projects match the current filters.")

        return pch_sections, None, tile_metadata

    _PCH_PILL_LABELS = {
        "projects": "Projects Covered",
        "totals": "F/S Total Planned / Done / Balance",
        "gangs": "Gangs",
        "productivity": "Productivity / Historical Avg",
        "loss": "Lost Units",
        "tse": "No. of TSE",
    }

    @app.callback(
        Output("store-pch-modal-focus", "data"),
        Output("kpi-pch-modal", "is_open"),
        Input({"type": "summary-pill-trigger", "mode": ALL, "metric": ALL}, "n_clicks"),
        Input("kpi-pch-modal-close", "n_clicks"),
        Input("f-project", "value"),
        State("kpi-pch-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_pch_modal(pill_clicks, close_clicks, project_values, is_open):
        trigger = _resolve_triggered_id()
        ctx = dash.callback_context
        triggered_entries = getattr(ctx, "triggered", None)
        trigger_value = None
        if triggered_entries:
            try:
                trigger_value = triggered_entries[0].get("value")
            except Exception:
                trigger_value = None

        if trigger == "kpi-pch-modal-close":
            return dash.no_update, False
        if trigger == "f-project":
            selected_projects = _normalize_str_list(_ensure_list(project_values))
            if not selected_projects:
                return dash.no_update, False
            payload = {"metric": "projects", "mode": "erection"}
            return payload, True
        if isinstance(trigger, dict) and trigger.get("type") == "summary-pill-trigger":
            if not trigger_value:
                raise PreventUpdate
            metric = str(trigger.get("metric") or "").strip().lower()
            mode = str(trigger.get("mode") or "").strip().lower() or "erection"
            if metric not in _PCH_PILL_LABELS:
                raise PreventUpdate
            if mode == "stringing" and not config.enable_stringing:
                raise PreventUpdate
            if mode not in {"erection", "stringing"}:
                mode = "erection"
            payload = {"metric": metric, "mode": mode}
            return payload, True
        raise PreventUpdate

    @app.callback(
        Output("kpi-pch-modal-accordion", "children"),
        Output("kpi-pch-modal-accordion", "active_item"),
        Output("kpi-pch-modal-title", "children"),
        Output("store-project-tile-meta", "data"),
        Input("store-pch-modal-focus", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
        prevent_initial_call=True,
    )
    def _render_pch_modal(focus_data, projects, months, quick_range, kv_filter, method_filter):
        if not isinstance(focus_data, dict) or not focus_data.get("metric"):
            raise PreventUpdate
        metric = str(focus_data.get("metric") or "").strip().lower()
        if metric not in _PCH_PILL_LABELS:
            raise PreventUpdate
        mode_value = str(focus_data.get("mode") or "").strip().lower()
        if mode_value not in {"erection", "stringing"}:
            mode_value = "erection"
        if mode_value == "stringing" and not config.enable_stringing:
            raise PreventUpdate
        sections, active_item, tile_meta = _populate_kpi_pch(
            projects,
            months,
            quick_range,
            mode_value,
            kv_filter,
            method_filter,
            use_modal_ids=True,
            pill_focus=metric,
        )
        mode_label = "Stringing" if mode_value == "stringing" else "Erection"
        title = f"PCH-wise { _PCH_PILL_LABELS[metric] } ({mode_label})"
        return sections, active_item, title, tile_meta

    @app.callback(
        Output("store-project-tile-focus", "data"),
        Output("project-detail-modal", "is_open"),
        Input({"type": "project-tile-trigger", "project": ALL, "mode": ALL, "context": ALL}, "n_clicks"),
        Input("project-modal-close", "n_clicks"),
        Input({"type": "proj-resp-open", "key": ALL}, "n_clicks"),
        State("project-detail-modal", "is_open"),
        State("store-project-tile-meta", "data"),
        prevent_initial_call=True,
    )
    def _toggle_project_tile_modal(tile_clicks, close_clicks, _resp_open_clicks, is_open, tile_meta_data):
        ctx = dash.callback_context
        triggered_entries = getattr(ctx, "triggered", None)
        trigger = _resolve_triggered_id()
        LOGGER.info(
            "project-modal-toggle trigger=%s entries=%s tile_clicks=%s close=%s open=%s",
            trigger,
            triggered_entries,
            tile_clicks,
            close_clicks,
            is_open,
        )
        if trigger == "project-modal-close":
            return dash.no_update, False
        if isinstance(trigger, dict) and trigger.get("type") == "proj-resp-open":
            raise PreventUpdate
        if isinstance(trigger, dict) and trigger.get("type") == "project-tile-trigger":
            if trigger.get("context") == "placeholder":
                raise PreventUpdate
            first_entry_value = None
            if triggered_entries:
                try:
                    first_entry_value = triggered_entries[0].get("value")
                except Exception:
                    first_entry_value = None
            if not first_entry_value:
                raise PreventUpdate
            project_key = trigger.get("project")
            tile_meta = tile_meta_data or {}
            meta = tile_meta.get(project_key or "")
            if not meta:
                LOGGER.warning("No tile metadata found for key %s", project_key)
                raise PreventUpdate
            payload = {
                "project": meta.get("project") or meta.get("display"),
                "code": meta.get("code"),
                "display": meta.get("display") or meta.get("project"),
                "mode": meta.get("mode") or trigger.get("mode"),
                "pch": meta.get("pch"),
                "ts": time.time(),
            }
            return payload, True
        raise PreventUpdate

    @app.callback(
        Output("project-modal-summary", "children"),
        Output("project-modal-title", "children"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),
        Input("f-method", "value"),
        prevent_initial_call=True,
    )
    def _render_project_modal_summary(
        focus_data,
        months,
        quick_range,
        gangs,
        kv_values,
        method_values,
    ):
        base_message = "Select a project tile to view its detailed view."
        base_title = "Project Deep Dive"
        if not isinstance(focus_data, dict):
            return base_message, base_title

        project_name = (focus_data.get("project") or "").strip()
        project_display = (focus_data.get("display") or project_name).strip()
        project_code = (focus_data.get("code") or "").strip()
        title_label = project_display or project_name
        lookup_label = project_name or project_display

        candidate_ids = _project_filter_candidates(lookup_label, project_code)
        if not candidate_ids:
            fallback = [value for value in (project_name, project_display, project_code) if value]
            candidate_ids = fallback[:1]
        project_list = _normalize_str_list(candidate_ids)
        if not project_list:
            return base_message, base_title

        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        kv_list = _normalize_str_list(_ensure_list(kv_values))
        method_list = _normalize_str_list(_ensure_list(method_values), lower=True)

        def _project_summary_for_mode(mode_value: str, *, is_stringing: bool) -> dict[str, str]:
            kv_payload = kv_list if is_stringing else []
            method_payload = method_list if is_stringing else []
            try:
                scope_meta = _build_scope_meta_payload(
                    eff_mode=mode_value,
                    project_list=project_list,
                    gang_list=gang_list,
                    months_list=months_list,
                    quick_range=quick_range,
                    kv_values=kv_payload,
                    method_values=method_payload,
                    kv_list=kv_payload,
                    method_list=method_payload,
                )
            except Exception:
                LOGGER.exception(
                    "Unable to build %s summary for project %s", mode_value, title_label or "unknown"
                )
                return _empty_summary_payload(is_stringing)
            return _summarize_scope_for_cards(scope_meta)

        erection_summary = _project_summary_for_mode("erection", is_stringing=False)
        stringing_summary = (
            _project_summary_for_mode("stringing", is_stringing=True) if config.enable_stringing else None
        )

        def _summary_card(
            title: str,
            summary_payload: dict[str, str],
            *,
            include_tse: bool = False,
            include_po: bool = False,
        ) -> dbc.Col:
            rows = [
                ("Projects Covered", summary_payload.get("projects", "-")),
                ("F/S Total Planned / Done / Balance", summary_payload.get("totals", "-")),
                ("Gangs", summary_payload.get("gangs", "-")),
                ("Productivity / Historical Avg", summary_payload.get("productivity", "-")),
                ("Lost Units", summary_payload.get("lost_units", "-")),
            ]
            if include_po:
                rows.append(("P/O Total Planned / Done / Balance", summary_payload.get("po_completion", "-")))
            if include_tse:
                rows.append(("No. of TSE", summary_payload.get("tse", "-")))
            pills = [
                html.Div(
                    [
                        html.Span(label, className="summary-pill__label"),
                        html.Span(value, className="summary-pill__value"),
                    ],
                    className="summary-pill",
                )
                for label, value in rows
            ]
            return dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [html.Div(title, className="fw-semibold mb-3"), *pills],
                        className="d-flex flex-column gap-2",
                    ),
                    className="shadow-sm h-100",
                ),
                xs=12,
                md=6,
            )

        cards: list[dbc.Col] = [
            _summary_card("Erection Snapshot", erection_summary or _empty_summary_payload(False)),
        ]
        if config.enable_stringing:
            cards.append(
                _summary_card(
                    "Stringing Snapshot",
                    stringing_summary or _empty_summary_payload(True),
                    include_tse=True,
                    include_po=True,
                )
            )

        summary_layout = dbc.Row(cards, className="g-3 project-modal-summary-table")
        title = f"{base_title} · {title_label}" if title_label else base_title
        return summary_layout, title

    @app.callback(
        Output("store-project-modal-section", "data"),
        Output("store-project-modal-performance-mode", "data"),
        Input("project-modal-btn-erections", "n_clicks"),
        Input("project-modal-btn-stringing", "n_clicks"),
        Input("project-modal-btn-performance-erection", "n_clicks"),
        Input("project-modal-btn-performance-stringing", "n_clicks"),
        Input("store-project-tile-focus", "data"),
        State("store-project-modal-section", "data"),
        State("store-project-modal-performance-mode", "data"),
        prevent_initial_call=True,
    )
    def _set_modal_section(
        btn_e,
        btn_s,
        btn_perf_e,
        btn_perf_s,
        focus_data,
        current_section,
        current_mode,
    ):
        trigger = _resolve_triggered_id()
        perf_mode = _modal_mode_from_store(current_mode, "erection")

        def _payload(mode_value: str) -> str:
            return _compose_modal_mode_payload(mode_value)

        if trigger == "store-project-tile-focus":
            perf_mode = _resolve_focus_mode(focus_data, perf_mode)
            if perf_mode == "stringing" and not config.enable_stringing:
                perf_mode = "erection"
            return "erections", _payload(perf_mode)
        if trigger == "project-modal-btn-erections":
            return "erections", _payload(perf_mode)
        if trigger == "project-modal-btn-stringing":
            if not config.enable_stringing:
                raise PreventUpdate
            return "stringing", _payload(perf_mode)
        if trigger == "project-modal-btn-performance-erection":
            return "performance", _payload("erection")
        if trigger == "project-modal-btn-performance-stringing":
            target_mode = "stringing" if config.enable_stringing else "erection"
            return "performance", _payload(target_mode)
        return (current_section or "erections"), _payload(perf_mode)

    @app.callback(
        Output("project-modal-section-erections", "is_open"),
        Output("project-modal-section-stringing", "is_open"),
        Output("project-modal-section-performance", "is_open"),
        Output("project-modal-btn-erections", "className"),
        Output("project-modal-btn-stringing", "className"),
        Output("project-modal-btn-performance-erection", "className"),
        Output("project-modal-btn-performance-stringing", "className"),
        Input("store-project-modal-section", "data"),
        Input("store-project-modal-performance-mode", "data"),
    )
    def _sync_modal_sections(active_section: str | None, performance_mode: Any):
        active = (active_section or "erections").strip().lower()
        perf_mode = _modal_mode_from_store(performance_mode, "erection")

        def _is_open(key: str) -> bool:
            return active == key

        def _class(key: str) -> str:
            base = "modal-section-btn"
            return f"{base} active" if active == key else base

        def _perf_class(target: str) -> str:
            base = "modal-section-btn"
            is_active = active == "performance" and perf_mode == target
            return f"{base} active" if is_active else base

        return (
            _is_open("erections"),
            _is_open("stringing"),
            _is_open("performance"),
            _class("erections"),
            _class("stringing"),
            _perf_class("erection"),
            _perf_class("stringing"),
        )

    # Toggle responsibilities visibility inside each project tile (pattern-matching IDs)
    @app.callback(
        Output({"type": "proj-resp-collapse", "key": MATCH}, "is_open"),
        Input({"type": "proj-resp-toggle", "key": MATCH}, "n_clicks"),
        State({"type": "proj-resp-collapse", "key": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_tile_resp(n, is_open):
        if not n:
            raise PreventUpdate
        return not bool(is_open)

    # --- Project Monthly Plan mini-modal: open/close and set project code ---
    @app.callback(
        Output("proj-resp-modal", "is_open"),
        Output("proj-resp-modal-title", "children"),
        Output("store-proj-resp-code", "data"),
        Output("store-proj-resp-month", "data"),
        Output("store-proj-resp-plan", "data"),
        Input({"type": "proj-resp-open", "key": ALL}, "n_clicks"),
        Input("proj-resp-modal-close", "n_clicks"),
        State("proj-resp-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_proj_resp_modal(open_clicks, close_clicks, is_open):
        trigger_id = _resolve_triggered_id()
        if trigger_id is None:
            raise PreventUpdate
        if trigger_id == "proj-resp-modal-close":
            return False, dash.no_update, dash.no_update, None, None
        ctx = dash.callback_context
        triggered_entries = getattr(ctx, "triggered", None)
        if not triggered_entries:
            raise PreventUpdate
        trigger_value = triggered_entries[0].get("value")
        if not trigger_value:
            # Ignore initial invocation where n_clicks is zero/None
            raise PreventUpdate
        key_str = None
        if isinstance(trigger_id, dict):
            id_obj = trigger_id
            key_str = id_obj.get("key")
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if id_obj.get("type") != "proj-resp-open":
            raise PreventUpdate

        code = name = month_raw = None
        plan_mode = "erection"
        if isinstance(key_str, str):
            parts = key_str.split("||")
            if parts and parts[-1] == "__modal__":
                parts = parts[:-1]
            if parts:
                candidate_mode = (parts[0] or "").strip().lower()
                if candidate_mode in {"erection", "stringing"}:
                    plan_mode = candidate_mode
                    parts = parts[1:]
            if parts:
                code = parts[0] or None
            if len(parts) > 1:
                month_raw = parts[1] or None
            if len(parts) > 2:
                name = parts[2] or None

        month_value, month_label = _normalize_month_value(month_raw)
        display_title = name or code
        plan_title = "Monthly Plan (Stringing)" if plan_mode == "stringing" else "Monthly Plan (Erection)"
        if display_title:
            title = f"{plan_title} \u2014 {display_title}"
        else:
            title = plan_title
        if month_label:
            title = f"{title} ({month_label})"
        payload = {"code": code, "name": name}
        return True, title, payload, month_value, plan_mode

    # --- Render responsibilities inside the project mini-modal ---
    @app.callback(
        Output("proj-resp-graph", "figure"),
        Output("proj-resp-kpi-target", "children"),
        Output("proj-resp-kpi-delivered", "children"),
        Output("proj-resp-kpi-ach", "children"),
        Input("store-proj-resp-code", "data"),
        Input("store-proj-resp-month", "data"),
        Input("store-proj-resp-plan", "data"),
        Input("proj-resp-entity", "value"),
        Input("proj-resp-metric", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("proj-resp-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _render_proj_resp(code_value, month_value, plan_mode_value, entity_value, metric_value, months_value, quick_value, is_open):
        if not code_value:
            raise PreventUpdate
        project_identifiers: list[str] = []
        seen: set[str] = set()

        def _append_identifier(candidate: Any) -> None:
            text = "" if candidate is None else str(candidate).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                project_identifiers.append(text)

        if isinstance(code_value, dict):
            for candidate in (code_value.get("name"), code_value.get("code")):
                _append_identifier(candidate)
        elif isinstance(code_value, Sequence) and not isinstance(code_value, (str, bytes)):
            for candidate in code_value:
                _append_identifier(candidate)
        else:
            _append_identifier(code_value)
        if not project_identifiers:
            raise PreventUpdate
        # If a dedicated month is chosen from the tile, override the global filters
        normalized_month, _month_label = _normalize_month_value(month_value)
        if normalized_month:
            months_value = [normalized_month]
            quick_value = None
        plan_key = "stringing" if str(plan_mode_value).strip().lower() == "stringing" else "erection"
        return _build_monthly_plan_for_project(
            project_value=project_identifiers,
            entity_value=entity_value,
            metric_value=metric_value,
            months_value=months_value,
            quick_range_value=quick_value,
            plan_mode=plan_key,
        )

    @app.callback(
        Output("trace-modal", "is_open"),
        Output("trace-modal-title", "children"),
        # Output("store-selected-gang", "data"),
        Input("store-dblclick", "data"),
        Input("trace-modal-close", "n_clicks"),
        State("trace-modal", "is_open"),
        State("store-selected-gang", "data"),
        prevent_initial_call=True,
    )
    def toggle_trace_modal(
        dbl_click: dict[str, Any] | None,
        close_clicks: int | None,
        is_open: bool,
        current_selection: str | None,
    ) -> tuple[bool, Any, str | None]:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "trace-modal-close":
            return False, dash.no_update
        if trigger == "store-dblclick":
            if not dbl_click or not dbl_click.get("gang"):
                raise PreventUpdate
            gang_value = dbl_click["gang"]
            title = f"Traceability - {gang_value}"
            return True, title
        raise PreventUpdate


def _stringing_plan_month_series(frame: pd.DataFrame, date_columns: Sequence[str]) -> pd.Series:
    """Return a normalized month-series derived from the provided date columns."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series([], dtype="datetime64[ns]")

    accumulator: pd.Series | None = None
    for column in date_columns:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        accumulator = parsed.copy() if accumulator is None else accumulator.fillna(parsed)

    if accumulator is None and "completion_date" in frame.columns:
        accumulator = pd.to_datetime(frame["completion_date"], errors="coerce")

    if accumulator is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    return accumulator.dt.to_period("M").dt.to_timestamp()


def _load_stringing_plan_month_map(
    *,
    date_columns: Sequence[str] = _STRINGING_FS_DATE_COLUMNS,
) -> dict[str, set[pd.Timestamp]]:
    try:
        plan_snapshot, _, _, _ = _load_stringing_plan_snapshot(config)
    except Exception:
        plan_snapshot = None
    plan_map: dict[str, set[pd.Timestamp]] = {}
    if not isinstance(plan_snapshot, pd.DataFrame) or plan_snapshot.empty:
        return plan_map
    plan = plan_snapshot.copy()
    plan_months = _stringing_plan_month_series(plan, date_columns)
    plan["__plan_month"] = plan_months
    plan = plan.dropna(subset=["__plan_month"])
    if plan.empty:
        return plan_map
    plan["__name_norm"] = plan.get("project_name", pd.Series([], dtype=str)).astype(str).map(_normalize_lower)
    plan["__code_norm"] = plan.get("project_key", pd.Series([], dtype=str)).astype(str).map(_normalize_lower)
    plan["__name_compact"] = plan.get("project_name", pd.Series([], dtype=str)).astype(str).map(_compact_project_key)
    plan["__code_compact"] = plan.get("project_key", pd.Series([], dtype=str)).astype(str).map(_compact_project_key)
    for _, row in plan.iterrows():
        month_value = row.get("__plan_month")
        if pd.isna(month_value):
            continue
        ts = pd.Timestamp(month_value)
        for key in (
            row.get("__name_norm"),
            row.get("__code_norm"),
            row.get("__name_compact"),
            row.get("__code_compact"),
        ):
            if key:
                plan_map.setdefault(key, set()).add(ts)
    return plan_map

def _stringing_plan_keys(name: str | None, code: str | None) -> list[str]:
    keys: set[str] = set()

    def _add_value(raw: str) -> None:
        if not raw:
            return
        norm = _normalize_lower(raw)
        if norm:
            keys.add(norm)
        compact = _compact_project_key(raw)
        if compact:
            keys.add(compact)

    for value in (name, code):
        if not value:
            continue
        text = str(value)
        _add_value(text)
        if " : " in text:
            left, right = text.split(" : ", 1)
            _add_value(left)
            _add_value(right)
    return [key for key in keys if key]


def _stringing_project_has_plan(plan_map: Mapping[str, set[pd.Timestamp]], name: str | None, code: str | None) -> bool:
    for key in _stringing_plan_keys(name, code):
        if key in plan_map:
            return True
    return False


def _stringing_plan_months_for_project(plan_map: Mapping[str, set[pd.Timestamp]], name: str, code: str) -> list[pd.Timestamp]:
    months: set[pd.Timestamp] = set()
    for key in _stringing_plan_keys(name, code):
        months.update(plan_map.get(key, ()))
    return sorted(months)

_STRINGING_PROJECT_CODE_PATTERN = re.compile(r"^(TA|TB)\s*-?\s*(\d{2,4})$", re.IGNORECASE)


def _format_stringing_project_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = _STRINGING_PROJECT_CODE_PATTERN.match(text)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"
    return text


def _stringing_plan_span_series(
    frame: pd.DataFrame,
    *,
    fallback_norms: Sequence[str] | None = None,
) -> pd.Series | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    candidates = {
        "span_m",
        "span m",
        "span",
        "span length",
        "span (m)",
        "span(km)",
        "span_km",
        "span(k m)",
        "span(m)",
        "span (km)",
        "length_m",
        "length m",
        "length",
        "length (m)",
        "length(km)",
        "length_km",
    }
    series_acc: pd.Series | None = None
    for column in frame.columns:
        norm = _normalize_col_key(column)
        if norm not in candidates:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        norm_key = norm.replace(" ", "")
        if "km" in norm_key:
            km_values = numeric
        elif norm in {"length_m", "length", "p/o", "span_m", "span (m)", "span", "tower_weight", "po_km"}:
            km_values = numeric / 1000.0
        else:
            km_values = numeric / 1000.0
        if series_acc is None:
            series_acc = km_values
        else:
            series_acc = series_acc.fillna(km_values)
    if series_acc is not None:
        return series_acc

    fallback_norms = tuple(fallback_norms or ())
    if fallback_norms:
        for column in frame.columns:
            norm = _normalize_col_key(column)
            if norm not in fallback_norms:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            norm_key = norm.replace(" ", "")
            if "km" in norm_key:
                km_values = numeric
            elif norm in {"length_m", "length", "p/o", "span_m", "span (m)", "span", "tower_weight", "po_km"}:
                km_values = numeric / 1000.0
            else:
                km_values = numeric
            if series_acc is None:
                series_acc = km_values
            else:
                series_acc = series_acc.fillna(km_values)
        if series_acc is not None:
            return series_acc

    return None


def _stringing_plan_totals_by_project(
    months_ts: Sequence[pd.Timestamp],
    *,
    current_month: pd.Timestamp | None = None,
    date_columns: Sequence[str] = _STRINGING_FS_DATE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-project planned KM (deduplicated) from Micro Plan responsibilities."""

    try:
        plan_snapshot, _, _, _ = _load_stringing_plan_snapshot(config)
    except Exception:
        plan_snapshot = None

    empty_df = pd.DataFrame(columns=["planned_km", "project_name_plan"])
    empty_series = pd.Series(dtype=float)
    if not isinstance(plan_snapshot, pd.DataFrame) or plan_snapshot.empty:
        return empty_df, empty_series

    plan = plan_snapshot.copy()
    plan["__plan_month"] = _stringing_plan_month_series(plan, date_columns)
    plan = plan.dropna(subset=["__plan_month"])
    if plan.empty:
        return empty_df, empty_series

    fallback_norms: tuple[str, ...]
    if tuple(date_columns) == _STRINGING_PO_DATE_COLUMNS:
        fallback_norms = ("p/o",)
    else:
        fallback_norms = ("span (m)", "span_m", "length", "length_m", "tower_weight")
    span_series = _stringing_plan_span_series(plan, fallback_norms=fallback_norms)
    if span_series is None:
        return empty_df, empty_series
    plan["__plan_km"] = span_series.fillna(0.0)

    plan["__project_name"] = plan.get("project_name", pd.Series([""] * len(plan), index=plan.index)).astype(str)
    plan["__project_key"] = plan.get("project_key", pd.Series([""] * len(plan), index=plan.index)).astype(str)
    plan["__project_display"] = plan["__project_name"].where(
        plan["__project_name"].str.strip() != "", plan["__project_key"]
    )
    plan["__project_display"] = plan["__project_display"].map(_format_stringing_project_label)
    plan["__project_norm"] = plan["__project_key"].map(_compact_project_key)
    missing_norm = plan["__project_norm"].astype(str).str.strip() == ""
    if missing_norm.any():
        plan.loc[missing_norm, "__project_norm"] = plan.loc[missing_norm, "__project_display"].map(
            lambda value: _compact_project_key(value) or _normalize_lower(value)
        )
    plan["__project_norm"] = plan["__project_norm"].fillna("")
    plan = plan[plan["__project_norm"] != ""].copy()
    if plan.empty:
        return empty_df, empty_series

    location_series = plan.get("location_no")
    if location_series is not None:
        plan["__location_norm"] = location_series.map(_normalize_location)
        plan = plan.drop_duplicates(
            subset=["__project_norm", "__location_norm", "__plan_month"],
            keep="last",
        )

    if months_ts:
        month_set = {pd.Timestamp(ts) for ts in months_ts if pd.notna(ts)}
        if month_set:
            plan = plan[plan["__plan_month"].isin(month_set)].copy()
    if plan.empty:
        return empty_df, empty_series

    totals = (
        plan.groupby("__project_norm")
        .agg(
            planned_km=("__plan_km", "sum"),
            project_name_plan=("__project_display", "last"),
        )
        .sort_index()
    )
    totals.index.name = "project_key_norm"

    plan_current = pd.Series(dtype=float)
    if current_month is not None and not pd.isna(current_month):
        current_month = pd.Timestamp(current_month)
        plan_current = plan[plan["__plan_month"] == current_month]
        if not plan_current.empty:
            plan_current = plan_current.groupby("__project_norm")["__plan_km"].sum()
        else:
            plan_current = pd.Series(dtype=float)

    return totals, plan_current


def _stringing_scope_has_plan(
    scope_meta: dict[str, Any] | None,
    months_ts: list[pd.Timestamp],
    *,
    date_columns: Sequence[str] = _STRINGING_FS_DATE_COLUMNS,
) -> bool:
    plan_map = _load_stringing_plan_month_map(date_columns=date_columns)
    if not plan_map:
        return False
    selected = (scope_meta or {}).get("selected") or {}
    project_filters = _normalize_str_list(selected.get("projects"))
    target_months = {ts for ts in months_ts if pd.notna(ts)}

    def _matches_months(months: set[pd.Timestamp]) -> bool:
        if not months:
            return False
        if not target_months:
            return True
        return bool(target_months & months)

    if project_filters:
        for project in project_filters:
            keys = _stringing_plan_keys(project, project)
            for key in keys:
                months = plan_map.get(key)
                if months and _matches_months(months):
                    return True
    else:
        for months in plan_map.values():
            if _matches_months(months):
                return True
    return False


def _stringing_planned_total_for_dates(
    scope_meta: dict[str, Any] | None,
    months_ts: list[pd.Timestamp],
    *,
    date_columns: Sequence[str],
) -> float:
    try:
        plan_df, _ = _stringing_plan_totals_by_project(
            months_ts,
            current_month=None,
            date_columns=date_columns,
        )
    except Exception:
        LOGGER.exception("Failed to compute stringing planned totals from compiled data.")
        plan_df = pd.DataFrame()

    if not isinstance(plan_df, pd.DataFrame) or plan_df.empty:
        return 0.0

    selected = (scope_meta or {}).get("selected") or {}
    project_filters = _normalize_str_list(selected.get("projects"))
    if project_filters:
        filter_keys = {
            _compact_project_key(value) or _normalize_lower(value)
            for value in project_filters
            if str(value).strip()
        }
        filter_keys = {key for key in filter_keys if key}
        if filter_keys:
            plan_df = plan_df[plan_df.index.isin(filter_keys)].copy()
            if plan_df.empty:
                return 0.0

    return float(plan_df["planned_km"].sum())


def _load_stringing_plan_snapshot(
    cfg: AppConfig,
) -> tuple[pd.DataFrame | None, set[tuple[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    cache = _STRINGING_PLAN_CACHE
    ts_now = time.time()
    cached_frame = cache["frame"]
    if (
        isinstance(cached_frame, pd.DataFrame)
        and not cached_frame.empty
        and (ts_now - cache["stored_at"] < _STRINGING_PLAN_CACHE_TTL_SECONDS)
    ):
        issues = cache["issues"]
        completion = cache["completion"]
        index_rows = cache.get("index", [])
        return cached_frame.copy(), set(completion), list(issues), list(index_rows)

    accessor = STRINGING_PLAN_ACCESSOR
    if accessor is None:
        return None, set(), [], []

    payload = accessor.load()
    if payload.has_frame:
        frame = payload.frame.copy()
        completion_keys = set(payload.completion_keys or set())
        cache["frame"] = frame.copy()
        cache["completion"] = set(completion_keys)
        cache["issues"] = []
        cache["index"] = []
        cache["stored_at"] = ts_now
        cache["last_written"] = ts_now
        return frame, completion_keys, [], []

    return None, set(), [], []

