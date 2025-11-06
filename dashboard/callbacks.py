"""Dash callbacks for the productivity dashboard."""

from __future__ import annotations

import logging
import json
import dash_bootstrap_components as dbc 
import pandas as pd
from io import BytesIO
import traceback
from typing import Any, Callable, Mapping, Sequence

import dash
import re
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.dependencies import MATCH, ALL
from datetime import datetime
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes

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
from .data_loader import load_stringing_compiled_raw as _load_stringing_compiled_raw
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


LOGGER = logging.getLogger(__name__)

BENCHMARK_MT_PER_DAY = 9.0
BENCHMARK_KM_PER_MONTH = 5.0

# App-wide config instance for callback logic
config = AppConfig()


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
    if not txt:
        return ""
    if txt.endswith(".0") and txt.replace(".", "", 1).isdigit():
        txt = txt.split(".", 1)[0]
    return txt


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
    export_df = pd.DataFrame(
        {
            "completion_date": working["date"],
            "project_name": working["project_name_display"],
            "location_no": working["span_display"],  # show From→To in the 'Location' column
            "tower_weight_mt": pd.to_numeric(working.get("daily_km", np.nan), errors="coerce"),  # will display as KM
            "daily_prod_mt":  pd.to_numeric(working.get("daily_km", np.nan), errors="coerce"),  # KM/day
            "gang_name": working["gang_name_display"],
            "start_date": working["date"],           # fallback (F/S start may not be present in daily)
            "supervisor_name": "",
            "section_incharge_name": "",
            "revenue_value": np.nan,
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
            "start_date": export_df["start_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else ""),
            "supervisor_name": export_df["supervisor_name"],
            "section_incharge_name": export_df["section_incharge_name"],
            "revenue": export_df["revenue_value"].map(_format_decimal),
        }
    )

    return export_df, display_df



_slug = lambda s: re.sub(r"[^a-z0-9_-]+", "-", str(s).lower()).strip("-")

def _render_avp_row(gang, delivered, lost, total, pct, avg_prod=0.0, baseline=0.0, last_project=" ", last_date=" ", rate_label="MT/day", unit_total="MT"):
    badge_cls = "good" if pct >= 80 else ("mid" if pct >= 65 else "low")
    delivered_pct = 0 if total == 0 else max(0, min(100, (delivered/total)*100))
    lost_pct = 0 if total == 0 else max(0, min(100, (lost/total)*100))

    row_tip_id = f"avp-tip-{gang}"  # STRING id for tooltip target

    return html.Div(
        id={"type": "avp-row", "index": gang},   # <-- move pattern id to the row itself
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
                id={"type": "avp-tip", "index": gang},     # pattern id to allow row-wide click capture
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
    stringing_data_provider: Callable[[], pd.DataFrame] | None = None,
    project_info_provider: Callable[[], pd.DataFrame] | None = None,
    project_baseline_provider: Callable[[], tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]] | None = None,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    responsibilities_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    responsibilities_error_provider: Callable[[], str | None] | None = None,
) -> None:

    LOGGER.debug("Registering callbacks")

    data_selector = DataSelector(
        config=config,
        data_provider=data_provider,
        stringing_provider=stringing_data_provider,
        logger=LOGGER,
    )
    responsibilities_accessor = ResponsibilitiesAccessor(
        data_provider=responsibilities_provider,
        completion_provider=responsibilities_completion_provider,
        error_provider=responsibilities_error_provider,
        logger=LOGGER,
    )
    has_responsibilities_provider = callable(responsibilities_provider)

    def _fetch_responsibilities(
        allow_workbook_fallback: bool = False,
    ) -> tuple[pd.DataFrame | None, set[tuple[str, str]], str | None, pd.ExcelFile | None]:
        payload: ResponsibilitiesPayload = responsibilities_accessor.load()
        if payload.has_frame:
            frame = payload.frame.copy() if payload.frame is not None else None
            completion_keys = set(payload.completion_keys or set())
            return frame, completion_keys, payload.error, None

        completion_keys = set(payload.completion_keys or set())
        load_error = payload.error
        if allow_workbook_fallback and not has_responsibilities_provider:
            cfg = config
            try:
                workbook = pd.ExcelFile(cfg.data_path)
            except FileNotFoundError:
                LOGGER.warning("Responsibilities workbook not found: %s", cfg.data_path)
                return None, completion_keys, "Compiled workbook not found.", None
            except Exception as exc:
                LOGGER.exception("Failed to open responsibilities workbook: %s", exc)
                return None, completion_keys, "Unable to load Micro Plan data.", None

            atomic_sheet = "MicroPlanResponsibilities"
            if atomic_sheet not in workbook.sheet_names:
                LOGGER.warning("Sheet '%s' missing in workbook", atomic_sheet)
                return None, completion_keys, "No Micro Plan data found in the compiled workbook.", workbook

            df_atomic = pd.read_excel(workbook, sheet_name=atomic_sheet)
            return df_atomic, completion_keys, load_error, workbook

        return None, completion_keys, load_error, None

    # --- shared: responsibilities figure + KPIs for a single project selection ---
    def _build_responsibilities_for_project(
        project_value: str | Sequence[str] | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
    ):
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

        df_atomic, completed_keys, load_error_msg, _ = _fetch_responsibilities(
            allow_workbook_fallback=True
        )
        if df_atomic is None or df_atomic.empty:
            message = load_error_msg or "No Micro Plan data found in the compiled workbook."
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
            return _empty_response("No responsibilities found for the selected project.")

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
            return _empty_response("No responsibilities found for the selected filters.")

        df_entity["is_completed"] = [
            (proj, loc) in completed_keys
            for proj, loc in zip(df_entity["project_name_lc"], df_entity["location_no_norm"])
        ]

        df_entity["revenue_planned"] = pd.to_numeric(df_entity.get("revenue_planned", 0.0), errors="coerce").fillna(0.0)
        df_entity["revenue_realised"] = pd.to_numeric(df_entity.get("revenue_realised", 0.0), errors="coerce").fillna(0.0)
        df_entity["tower_weight"] = pd.to_numeric(df_entity.get("tower_weight", 0.0), errors="coerce").fillna(0.0)

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
            return _empty_response("No responsibilities found for the selected filters.")

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

        # Ensure chart builder has the target column by metric name
        if "revenue_planned" in aggregated.columns and "revenue" not in aggregated.columns:
            aggregated["revenue"] = aggregated["revenue_planned"]

        if aggregated.empty:
            return _empty_response("No responsibilities found for the selected filters.")

        fig = build_responsibilities_chart(
            aggregated,
            entity_label=entity,
            metric=metric,
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
            return f"{value:,.0f} MT"

        kpi_target_txt = fmt_num(total_target)
        kpi_deliv_txt = fmt_num(total_delivered)
        kpi_ach_txt = f"{achievement:.0f}%"

        return fig, kpi_target_txt, kpi_deliv_txt, kpi_ach_txt
    
        # --- helper: attach __line_kv__ by looking up Project Details "Project Name" ---
    def _attach_line_kv(work: pd.DataFrame) -> pd.DataFrame:
        """
        Adds __line_kv__ ('765'|'400'|NA) inferred from Project Details.
        Tries BOTH mappings: project name and project code.
        If mapping fails, falls back to the row's own project text.
        """
        try:
            if work is None or work.empty:
                return work

            proj_col = "project_name" if "project_name" in work.columns else ("project" if "project" in work.columns else None)
            if not proj_col:
                return work

            out = work.copy()
            out["__kv_source__"] = out[proj_col].astype(str).str.strip()

            dfpi = project_info_provider() if project_info_provider is not None else None
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



    # --- Mode toggle -> store + banner text ---
    @app.callback(
        Output("store-mode", "data"),
        Output("mode-banner", "children"),
        Input("mode-toggle", "value"),
        prevent_initial_call=True,
    )
    def _sync_mode_store_and_banner(mode_value: str | None):
        mode = (mode_value or "erection").strip().lower()
        banner = "Erection mode" if mode == "erection" else "Stringing mode"
        return mode, banner
    def _get_project_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
        if project_baseline_provider is None:
            return {}, {}
        try:
            overall_map, monthly_map = project_baseline_provider()
        except Exception as exc:
            LOGGER.warning("Failed to retrieve project baselines: %s", exc)
            return {}, {}
        return overall_map or {}, monthly_map or {}


    # Charts OR AVP rows -> store-click-meta (robust & single source of truth)
    app.clientside_callback(
        """
        // charts OR AVP (row or overlay) -> store-click-meta
        function(lossClick, topClick, bottomClick, rowTs, tipTs) {
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;

        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];

        // --- AVP surfaces (row or overlay) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â only accept real, timestamped clicks
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


    @app.callback(
        Output("lbl-erections-title", "children"),
        Input("mode-toggle", "value"),
        Input("store-mode", "data"),
    )
    def _title_for_completed(toggle_value, mode_value):
        eff_mode = (toggle_value or mode_value or "erection").strip().lower()
        return "Stringing Completed" if eff_mode == "stringing" else "Erections Completed"

    @app.callback(
        Output("stringing-filters-wrap", "style"),
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def _toggle_stringing_filters(mode_value, toggle_value):
        mode = (toggle_value or mode_value or "erection").strip().lower()
        return {"display": "block"} if mode == "stringing" else {"display": "none"}

    # Make KPI row span full width in Stringing (4 cards) vs 5 in Erection
    @app.callback(
        Output("kpi-row", "className"),
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def _kpi_row_class(mode_value, toggle_value):
        base = "g-3 align-items-stretch row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-lg-4 "
        mode = (toggle_value or mode_value or "erection").strip().lower()
        # Stringing shows 4 KPI tiles — use 4 cols on xl as well
        return base + ("row-cols-xl-4" if mode == "stringing" else "row-cols-xl-5")
    
    @app.callback(
        Output("card-total-nos", "style"),
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def _toggle_total_nos_card(mode_value, toggle_value):
        mode = (toggle_value or mode_value or "erection").strip().lower()
        return {} if mode == "erection" else {"display": "none"}
    
    @app.callback(
        Output("f-kv", "value"),
        Output("f-method", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("mode-toggle", "value"),
        prevent_initial_call=True,
    )
    def _reset_stringing_filters(_n, _mode_value):
        # Default = overall view (both selected)
        return ["400", "765"], ["manual", "tse"]


    

    @app.callback(
        Output("f-project", "value"),
        Output("f-month", "value"),
        Output("f-gang", "value"),
        Output("f-quick-range", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("link-clear-quick-range", "n_clicks"),
        Input("mode-toggle", "value"),
        prevent_initial_call=True,
    )
    def handle_filter_reset(
        reset_clicks: int | None,
        clear_quick_clicks: int | None,
        mode_value: str | None,
    ) -> tuple[Any, Any, Any, Any]:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        # Clear only quick range if that link was clicked
        if trigger_id == "link-clear-quick-range":
            return dash.no_update, dash.no_update, dash.no_update, None
        # On mode toggle or Reset click: reset all filters to defaults
        if trigger_id in {"mode-toggle", "btn-reset-filters"}:
            # Compute default month from the latest data date in the active mode's dataset
            try:
                eff_mode = (mode_value or "erection").strip().lower()
                df = data_selector.select(eff_mode)
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
        Output("f-project", "options"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),         # NEW
        Input("f-method", "value"),     # NEW
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_project_options(
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        kv_values: Sequence[str] | None,       # NEW
        method_values: Sequence[str] | None,   # NEW
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, str]]:
        try:
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = data_selector.select(eff_mode)
            if df_day is None or df_day.empty:
                return []
            # Derive month if missing (belt-and-braces)
            if "month" not in df_day.columns and "date" in df_day.columns:
                work = df_day.copy()
                work["date"] = pd.to_datetime(work["date"], errors="coerce")
                work = work.dropna(subset=["date"])  # keep valid only
                work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
            else:
                work = df_day.copy()

                        # --- Stringing-only filters: Line kV + Method ---
            eff_mode = (toggle_value or mode_value or "erection").strip().lower()
            if eff_mode == "stringing":
                work = _attach_line_kv(work)

                # --- kV chips: apply filter only if the selection is a proper subset ---
                kv_set = set(kv_values or [])
                if kv_set and kv_set != {"400", "765"}:
                    work = work[work["__line_kv__"].isin(kv_set)]

                # --- Method chips: same subset rule ---
                mset = set((m or "").lower() for m in (method_values or []))
                if mset and mset != {"manual", "tse"} and "method" in work.columns:
                    work = work[work["method"].astype(str).str.lower().isin(mset)]



            months_ts = resolve_months(_ensure_list(months), quick_range)
            days_factor = _avg_days_in_selected_months(months_ts)

            filtered = work
            if months_ts and "month" in filtered.columns:
                filtered = filtered[filtered["month"].isin(months_ts)]
            gang_list = _ensure_list(gangs)
            if gang_list and "gang_name" in filtered.columns:
                filtered = filtered[filtered["gang_name"].isin(gang_list)]

            # Project column may be aliased
            proj_col = "project_name" if "project_name" in filtered.columns else ("project" if "project" in filtered.columns else None)
            if not proj_col:
                return []
            projects = sorted(pd.Series(filtered[proj_col]).dropna().astype(str).str.strip().unique())
            if not projects:
                projects = sorted(pd.Series(work[proj_col]).dropna().astype(str).str.strip().unique())
            return [{"label": p, "value": p} for p in projects]
        except Exception as exc:
            LOGGER.exception("Failed to build project options: %s", exc)
            # Return an empty list instead of 500 to keep UI responsive
            return []

    @app.callback(
        Output("f-gang", "options"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-kv", "value"),         # NEW
        Input("f-method", "value"),     # NEW
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_gang_options(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        kv_values: Sequence[str] | None,       # NEW
        method_values: Sequence[str] | None,   # NEW
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, str]]:
        try:
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = data_selector.select(eff_mode)
            if df_day is None or df_day.empty:
                return []
            work = df_day.copy()
            eff_mode = (toggle_value or mode_value or "erection").strip().lower()
            if eff_mode == "stringing":
                work = _attach_line_kv(work)

                # --- kV chips: apply filter only if the selection is a proper subset ---
                kv_set = set(kv_values or [])
                if kv_set and kv_set != {"400", "765"}:
                    work = work[work["__line_kv__"].isin(kv_set)]

                # --- Method chips: same subset rule ---
                mset = set((m or "").lower() for m in (method_values or []))
                if mset and mset != {"manual", "tse"} and "method" in work.columns:
                    work = work[work["method"].astype(str).str.lower().isin(mset)]



            if "month" not in work.columns and "date" in work.columns:
                work["date"] = pd.to_datetime(work["date"], errors="coerce")
                work = work.dropna(subset=["date"])
                work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()

            filtered = work
            project_list = _ensure_list(projects)
            proj_col = "project_name" if "project_name" in filtered.columns else ("project" if "project" in filtered.columns else None)
            if project_list and proj_col:
                filtered = filtered[filtered[proj_col].isin(project_list)]
            months_ts = resolve_months(_ensure_list(months), quick_range)
            if months_ts and "month" in filtered.columns:
                filtered = filtered[filtered["month"].isin(months_ts)]

            if "gang_name" not in filtered.columns:
                return []
            gangs = sorted(filtered["gang_name"].dropna().astype(str).str.strip().unique())
            if not gangs:
                gangs = sorted(work.get("gang_name", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique())
            return [{"label": g, "value": g} for g in gangs]
        except Exception as exc:
            LOGGER.exception("Failed to build gang options: %s", exc)
            return []

    @app.callback(
        Output("f-month", "options"),
        Input("f-project", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),         # NEW
        Input("f-method", "value"),     # NEW
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_month_options(
        projects: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        kv_values: Sequence[str] | None,       # NEW
        method_values: Sequence[str] | None,   # NEW
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, str]]:
        try:
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = data_selector.select(eff_mode)
            if df_day is None or df_day.empty:
                return []
            work = df_day.copy()
            eff_mode = (toggle_value or mode_value or "erection").strip().lower()
            if eff_mode == "stringing":
                work = _attach_line_kv(work)

                # --- kV chips: apply filter only if the selection is a proper subset ---
                kv_set = set(kv_values or [])
                if kv_set and kv_set != {"400", "765"}:
                    work = work[work["__line_kv__"].isin(kv_set)]

                # --- Method chips: same subset rule ---
                mset = set((m or "").lower() for m in (method_values or []))
                if mset and mset != {"manual", "tse"} and "method" in work.columns:
                    work = work[work["method"].astype(str).str.lower().isin(mset)]



            # Ensure month exists from date
            if "month" not in work.columns and "date" in work.columns:
                work["date"] = pd.to_datetime(work["date"], errors="coerce")
                work = work.dropna(subset=["date"]).copy()
                work["month"] = work["date"].to_period("M").dt.to_timestamp()

            filtered = work
            project_list = _ensure_list(projects)
            proj_col = "project_name" if "project_name" in filtered.columns else ("project" if "project" in filtered.columns else None)
            if project_list and proj_col:
                filtered = filtered[filtered[proj_col].isin(project_list)]
            gang_list = _ensure_list(gangs)
            if gang_list and "gang_name" in filtered.columns:
                filtered = filtered[filtered["gang_name"].isin(gang_list)]

            if "month" not in filtered.columns:
                return []
            months = sorted(pd.to_datetime(filtered["month"].dropna().unique()))
            if quick_range:
                months_range = resolve_months(None, quick_range)
                months = [m for m in months if m in months_range]
            if not months and "month" in work.columns:
                months = sorted(pd.to_datetime(work["month"].dropna().unique()))
            return [{"label": m.strftime("%b %Y"), "value": m.strftime("%Y-%m")} for m in months]
        except Exception as exc:
            LOGGER.exception("Failed to build month options: %s", exc)
            return []

    @app.callback(
        Output("f-month", "value", allow_duplicate=True),
        Input("f-month", "options"),
        Input("store-mode", "data"),
        State("f-month", "value"),
        prevent_initial_call='initial_duplicate',
    )
    def ensure_default_month(options, mode_value, current_value):
        try:
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
        Output("label-perf-period", "children"),
        Output("label-gang-period", "children"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def update_period_labels(
        months: Sequence[str] | None,
        quick_range: str | None,
    ) -> tuple[str, str, str]:
        month_list = _ensure_list(months)
        months_ts = resolve_months(month_list, quick_range)
        label = _format_period_label(months_ts)
        return label, label, label

    # ---- Project Overview (dynamic body) -----------------------------------------
    @app.callback(
        Output("pd-title", "children"),
        Output("project-details", "children"),
        Input("f-project", "value"),
        Input("store-mode", "data"),    # NEW: re-render when mode changes
        Input("mode-toggle", "value"),  # NEW: also react to the visible toggle
        prevent_initial_call=False,
    )
    def show_project_details(selected_project, _mode_value, _toggle_value):
        """Render the project overview grid or an informative message."""

        default_title = "Project Overview"

        def _clean_text(raw: Any) -> str:
            if raw is None:
                return ""
            text = str(raw).strip()
            if not text:
                return ""
            if any(marker in text for marker in ("Ã", "Â")):
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
    def update_responsibilities(
        project_value: str | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
    ): 
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

        df_atomic, completed_keys, load_error_msg, workbook = _fetch_responsibilities(
            allow_workbook_fallback=True
        )
        if df_atomic is None or df_atomic.empty:
            message = load_error_msg or "No Micro Plan data found in the compiled workbook."
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

        if (not has_responsibilities_provider) and workbook is not None:
            daily_sheet = None
            for candidate in ("ProdDailyExpandedSingles"):
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
                        cleaned = pd.DataFrame(
                            {
                                "project_name_lc": df_daily[col_proj].map(_normalize_lower),
                                "location_no_norm": df_daily[col_loc].map(_normalize_location),
                            }
                        )
                        completed_keys = {
                            (p, loc)
                            for p, loc in zip(cleaned["project_name_lc"], cleaned["location_no_norm"])
                            if p and loc
                        }
            else:
                LOGGER.info(
                    "Daily expanded sheets not found; delivered values fall back to realised revenue only."
                )

        sel_norm = _normalize_text(project_value)
        sel_lc = sel_norm.lower()
        sel_compact = re.sub(r"[^a-z0-9]", "", sel_lc)

        df_project = df_atomic[
            (df_atomic["project_name_lc"] == sel_lc) | (df_atomic["project_key_lc"] == sel_lc)
        ]

        if df_project.empty and sel_compact:
            project_name_compact = df_atomic["project_name_lc"].str.replace(r"[^a-z0-9]", "", regex=True)
            project_key_compact = df_atomic["project_key_lc"].str.replace(r"[^a-z0-9]", "", regex=True)
            df_project = df_atomic[
                (project_name_compact == sel_compact) | (project_key_compact == sel_compact)
            ]

        if df_project.empty:
            return _empty_response("Selected project not found in Micro Plan data.")

        if not active_months:
            return _empty_response("Select a month to view the Micro Plan.")

        if active_months:
            month_mask = df_project['completion_month'].isin(active_months)
            if not month_mask.any():
                label = _format_period_label(active_months)
                label_clean = label.strip('()') if label else 'selected month'
                return _empty_response(f"Micro Plan for {label_clean} is not available.")
            df_project = df_project.loc[month_mask].copy()

        entity_lc = entity_value.lower()
        df_entity = df_project[df_project["entity_type_lc"] == entity_lc].copy()

        if df_entity.empty:
            return _empty_response("No responsibilities found for the selected filters.")

        df_entity["is_completed"] = [
            (proj, loc) in completed_keys
            for proj, loc in zip(df_entity["project_name_lc"], df_entity["location_no_norm"])
        ]

        df_entity["delivered_revenue"] = np.where(
            df_entity["revenue_realised"] > 0,
            df_entity["revenue_realised"],
            np.where(df_entity["is_completed"], df_entity["revenue_planned"], 0.0),
        )
        df_entity["delivered_tower_weight"] = np.where(
            df_entity["is_completed"], df_entity["tower_weight"], 0.0
        )

        df_entity = df_entity[df_entity["entity_name"].astype(bool)].copy()
        if df_entity.empty:
            return _empty_response("No responsibilities found for the selected filters.")

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
        delivered_metric_col = ("delivered_revenue" if metric_value == "revenue" else "delivered_tower_weight")

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
                parts = [part.strip() for part in value.split(',') if part.strip()]
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
            return _empty_response("No responsibilities found for the selected filters.")

        aggregated["delivered_value"] = np.where(
            metric_value == "revenue",
            aggregated["delivered_revenue"],
            aggregated["delivered_tower_weight"],
        )

        fig = build_responsibilities_chart(
            aggregated,
            entity_label=entity_value,
            metric=metric_value,
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
            return f"{value:,.0f} MT"

        kpi_target_txt = fmt_num(total_target)
        kpi_deliv_txt = fmt_num(total_delivered)
        kpi_ach_txt = f"{achievement:.0f}%"

        return fig, kpi_target_txt, kpi_deliv_txt, kpi_ach_txt


    @app.callback(
        Output("kpi-avg", "children"),
        Output("kpi-delta", "children"),
        Output("kpi-active", "children"),
        Output("kpi-total", "children"),
        Output("kpi-total-planned", "children"),
        Output("kpi-total-nos", "children"),
        Output("kpi-total-nos-planned", "children"),
        Output("kpi-loss", "children"),
        Output("kpi-loss-delta", "children"),   # <--- add this line
        Output("avp-list", "children"),            # <-- NEW (HTML children)
        Output("g-actual-vs-bench", "figure"),
        # Output("g-monthly", "figure"),
        Output("g-top5", "figure"),
        Output("g-bottom5", "figure"),
        Output("g-projects-over-months", "figure"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("f-kv", "value"),         # NEW
        Input("f-method", "value"),     # NEW
        Input("f-topbot-metric", "value"),
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_dashboard(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        kv_values: Sequence[str] | None,       # NEW
        method_values: Sequence[str] | None,   # NEW
        topbot_metric: str | None,         
        mode_value: str | None,
        toggle_value: str | None,
    ) -> tuple:
        if True:
            project_list = _ensure_list(projects)
            month_list = _ensure_list(months)
            gang_list = _ensure_list(gangs)
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = data_selector.select(eff_mode)
                        # --- Stringing-only filters: Line kV + Method ---
            if eff_mode == "stringing" and isinstance(df_day, pd.DataFrame) and not df_day.empty:
                df_day = _attach_line_kv(df_day)

                kv_set = set(kv_values or [])
                if kv_set and kv_set != {"400", "765"}:
                    df_day = df_day[df_day["__line_kv__"].isin(kv_set)]

                mset = set((m or "").lower() for m in (method_values or []))
                if mset and mset != {"manual", "tse"} and "method" in df_day.columns:
                    df_day = df_day[df_day["method"].astype(str).str.lower().isin(mset)]



            months_ts = resolve_months(month_list, quick_range)
            days_factor = _avg_days_in_selected_months(months_ts)

            scoped = apply_filters(df_day, project_list, months_ts, gang_list)
            scoped_top_bottom = apply_filters(df_day, project_list, months_ts, [])

            is_stringing = (str(eff_mode or "erection").strip().lower() == "stringing")
            metric_col = "daily_km" if is_stringing else "daily_prod_mt"
            unit_short = "KM" if is_stringing else "MT"

            if is_stringing:
                # Monthly KPI and benchmark for stringing mode
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
                kpi_delta = (
                    "(n/a)" if delta_pct is None else f"({delta_pct:+.0f}% vs {benchmark:.1f} KM/month)"
                )
                # Keep project chart lines in per-day units for readability
                project_bench = BENCHMARK_KM_PER_MONTH
                avg_line_for_project = (scoped[metric_col].mean() if len(scoped) and (metric_col in scoped.columns) else 0.0)
            else:
                benchmark = BENCHMARK_MT_PER_DAY
                avg_prod = scoped[metric_col].mean() if len(scoped) and (metric_col in scoped.columns) else 0.0
                delta_pct = (avg_prod - benchmark) / benchmark * 100 if benchmark else None
                kpi_avg = f"{avg_prod:.2f} {unit_short}"
                kpi_delta = (
                    "(n/a)" if delta_pct is None else f"({delta_pct:+.0f}% vs {benchmark:.1f} {unit_short})"
                )
                project_bench = benchmark
                avg_line_for_project = avg_prod

            has_selected_months = bool(months_ts)

            scope_mask = pd.Series(True, index=df_day.index)
            if project_list and ("project_name" in df_day.columns):
                scope_mask &= df_day["project_name"].isin(project_list)
            if gang_list and ("gang_name" in df_day.columns):
                scope_mask &= df_day["gang_name"].isin(gang_list)
            scoped_all = df_day.loc[scope_mask].copy()
            # Ensure expected keys exist to avoid KeyError on selection
            if "gang_name" not in scoped_all.columns:
                scoped_all["gang_name"] = pd.Series(dtype=str)
            if "project_name" not in scoped_all.columns:
                scoped_all["project_name"] = pd.Series(dtype=str)

            if has_selected_months and not scoped_all.empty and "month" in scoped_all:
                month_values = sorted(set(months_ts))
                period_mask = scoped_all["month"].isin(month_values)
                loss_scope = scoped_all.loc[period_mask].copy()
                earliest_month = month_values[0]
                history_scope = scoped_all.loc[scoped_all["month"] < earliest_month].copy()
            else:
                loss_scope = scoped_all.copy()
                history_scope = scoped_all.copy()

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
                if is_stringing:
                    proj_overall_all, proj_monthly_all = compute_project_baseline_maps_for(scoped_all, metric_col)
                    proj_overall_hist, proj_monthly_hist = compute_project_baseline_maps_for(history_scope, metric_col)
                else:
                    proj_overall_all, proj_monthly_all = compute_project_baseline_maps(scoped_all)
                    proj_overall_hist, proj_monthly_hist = compute_project_baseline_maps(history_scope)
                if proj_overall_hist:
                    proj_overall_all.update(proj_overall_hist)
                proj_monthly = proj_monthly_hist
        # Gang → Project bridge
            gang_to_project = (
                scoped_all[["gang_name", "project_name"]]
                .dropna()
                .drop_duplicates()
                .set_index("gang_name")["project_name"]
                .astype(str)
                .to_dict()
            )

            # Build the same names your downstream code already expects:
            baseline_overall_map = {g: proj_overall_all.get(p) for g, p in gang_to_project.items()}
            baseline_monthly_map = {g: proj_monthly.get(p, {}) for g, p in gang_to_project.items()}

            baseline_map = baseline_overall_map


            baseline_map = baseline_overall_map
            loss_rows: list[dict[str, float]] = []
            for gang_name, gang_df in loss_scope.groupby("gang_name"):
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
                loss_rows.append(
                    {
                        "gang_name": gang_name,
                        "delivered": delivered,
                        "lost": loss_mt,
                        "potential": potential,
                        "avg_prod": (gang_df[metric_col].mean() if metric_col in gang_df.columns else 0.0),
                        "baseline": baseline,
                    }
                )
            if loss_rows:

                loss_df = pd.DataFrame(loss_rows)
                # --- convert per-day metrics to per-month for stringing ---
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

                loss_df = pd.DataFrame(

                    columns=[

                        "gang_name",

                        "delivered",

                        "lost",

                        "potential",

                        "avg_prod",

                        "baseline",

                        "efficiency_pct",

                        "total_mt",

                    ]

                )







        

        # --- meta for hover: last project & last worked date per gang (within current filters)
        meta_ready = {"gang_name", "project_name", "date"}.issubset(scoped_all.columns) and not scoped_all.empty

        if meta_ready:
                idx_last = scoped_all.sort_values("date").groupby("gang_name")["date"].idxmax()
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
        loss_df["last_date_str"] = last_date_series.dt.strftime("%d-%b-%Y").fillna("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â")
        loss_df["last_project"]  = loss_df.get("last_project").fillna("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â")

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
                if active_months and has_responsibilities_provider:
                    resp_df, _, _, _ = _fetch_responsibilities()
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
                            # planned MT is sum of tower_weight
                            if 'tower_weight' in df_mp.columns:
                                planned_mt = pd.to_numeric(df_mp['tower_weight'], errors='coerce').fillna(0.0).sum()
                            else:
                                planned_mt = 0.0
                            # planned towers = unique locations per project
                            planned_tower_count = int(df_mp[['project_name_lc', 'location_no_norm']].dropna().drop_duplicates().shape[0])
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
        projects_scope = df_day.copy()
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
                days_factor = 30.0
                try:
                    if months_ts:
                        month_days = [pd.to_datetime(m).to_period("M").days_in_month if not isinstance(m, pd.Period) else m.days_in_month for m in months_ts]
                        if month_days:
                            days_factor = float(sum(month_days) / len(month_days))
                except Exception:
                    days_factor = 30.0
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
        
    CHART_SOURCES = {"g-actual-vs-bench", "g-top5", "g-bottom5"}

    @app.callback(
        Output("tbl-idle-intervals", "data"),
        Output("tbl-daily-prod", "data"),
        Output("modal-tbl-idle-intervals", "data"),
        Output("modal-tbl-daily-prod", "data"),
        Input("store-click-meta", "data"),
        Input("trace-gang", "value"),
        Input("modal-trace-gang", "value"),
        State("f-project", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
        State("store-mode", "data"),
        prevent_initial_call=True,
    )
    def update_trace_tables(
        meta,
        trace_gang_value,
        modal_trace_gang_value,
        projects,
        months,
        quick_range,
        gangs,
        mode_value: str | None,
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
            if meta_is_chart:
                gang_focus = meta_gang
            else:
                gang_focus = dropdown_selection
        else:
            gang_focus = dropdown_selection or (meta_gang if meta_is_chart else None)

        if not gang_focus:
            raise PreventUpdate


        project_list = _ensure_list(projects)
        month_list   = _ensure_list(months)
        gang_list    = _ensure_list(gangs)

        df_day = data_selector.select(mode_value)
        months_ts = resolve_months(month_list, quick_range)

        base_scope = apply_filters(df_day, project_list, months_ts, []).copy()
        scoped     = apply_filters(df_day, project_list, months_ts, gang_list).copy()

        is_stringing = (str(mode_value or "erection").strip().lower() == "stringing")
        metric_col = "daily_km" if is_stringing else "daily_prod_mt"

        def pick_gang_scope(target_gang: str | None) -> pd.DataFrame:
            if not target_gang:
                return pd.DataFrame()
            subset = base_scope[base_scope["gang_name"] == target_gang]
            if not subset.empty:
                return subset
            fb = df_day[df_day["gang_name"] == target_gang].copy()
            if project_list:
                fb = fb[fb["project_name"].isin(project_list)]
            if months_ts:
                fb = fb[fb["month"].isin(months_ts)]
            return fb

        baseline_source = df_day.copy()
        if project_list:
            baseline_source = baseline_source[baseline_source["project_name"].isin(project_list)]
        if gang_list:
            baseline_source = baseline_source[baseline_source["gang_name"].isin(gang_list)]
        # PROJECT-level baselines for trace/idle view, then map to gang keys
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
            if is_stringing:
                proj_overall, proj_monthly = compute_project_baseline_maps_for(baseline_source, metric_col)
            else:
                proj_overall, proj_monthly = compute_project_baseline_maps(baseline_source)
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

        overall_baseline_map = {g: proj_overall.get(p) for g, p in g2p.items()}
        monthly_baseline_map = {g: proj_monthly.get(p, {}) for g, p in g2p.items()}


        # Idle intervals
        idle_source = pick_gang_scope(gang_focus)
        if idle_source.empty:
            idle_source = scoped if not scoped.empty else base_scope

        idle_df = compute_idle_intervals_per_gang(
            idle_source,
            loss_max_gap_days=config.loss_max_gap_days,
            baseline_month_lookup=monthly_baseline_map,
            baseline_fallback_map=overall_baseline_map,
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

        # Daily prod
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
            # align to expected column id for the table
            daily_source = daily_source.drop(columns=[metric_col])
        daily_data = daily_source.to_dict("records")

        # mirror into modal tables
        return idle_data, daily_data, idle_data, daily_data

    @app.callback(
        Output("tbl-erections-completed", "columns"),
        Output("tbl-erections-completed", "data"),
        Input("erections-completion-range", "start_date"),
        Input("erections-completion-range", "end_date"),
        Input("f-project", "value"),
        Input("f-gang", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("erections-search", "value"),
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_erections_completed(
        start_date,
        end_date,
        projects,
        gangs,
        months,
        quick_range,
        search_text,
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, object]]:
        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        project_list = _ensure_list(projects)
        gang_list = _ensure_list(gangs)
        _unused_months = _ensure_list(months)
        _unused_quick_range = quick_range

        eff_mode = (toggle_value or mode_value or "erection")
        df_day = data_selector.select(eff_mode)
        scoped = apply_filters(df_day, project_list, [], gang_list).copy()

        if str(eff_mode).strip().lower() == "stringing":
                export_df, display_df = _prepare_stringing_completed(
                    scoped,
                    range_start=range_start,
                    range_end=range_end,
                    search_text=search_text,
                )
                columns = [
                    {"name": "Completion Date",           "id": "completion_date"},
                    {"name": "Project",                   "id": "project_name"},
                    {"name": "Span (From→To)",            "id": "location_no"},   # we render From→To here
                    {"name": "Length (KM)",               "id": "tower_weight"},  # was Tower Weight (MT)
                    {"name": "Productivity (KM/day)",     "id": "daily_prod_mt"}, # was MT/day
                    {"name": "Gang",                      "id": "gang_name"},
                    {"name": "F/S Start Date",            "id": "start_date"},
                    {"name": "Supervisor",                "id": "supervisor_name"},
                    {"name": "Section Incharge",          "id": "section_incharge_name"},
                    {"name": "Revenue",                   "id": "revenue"},
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
                    {"name": "Completion Date",           "id": "completion_date"},
                    {"name": "Project",                   "id": "project_name"},
                    {"name": "Location",                  "id": "location_no"},
                    {"name": "Tower Weight (MT)",         "id": "tower_weight"},
                    {"name": "Productivity (MT/day)",     "id": "daily_prod_mt"},
                    {"name": "Gang",                      "id": "gang_name"},
                    {"name": "Start Date",                "id": "start_date"},
                    {"name": "Supervisor",                "id": "supervisor_name"},
                    {"name": "Section Incharge",          "id": "section_incharge_name"},
                    {"name": "Revenue",                   "id": "revenue"},
                ]

            # return columns + rows (empty list if nothing)
        if display_df.empty:
            return columns, []
        return columns, display_df.to_dict("records")

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
        Input("btn-export-trace", "n_clicks"),
        Input("modal-btn-export-trace", "n_clicks"),
        State("f-project", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
        State("trace-gang", "value"),
        State("store-selected-gang", "data"),
        State("erections-completion-range", "start_date"),
        State("erections-completion-range", "end_date"),
        State("erections-search", "value"),
        State("store-mode", "data"),
        prevent_initial_call=True,
    )
    def export_trace(
        main_clicks: int | None,
        modal_clicks: int | None,
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        trace_gang_value: str | None,
        selected_gang: str | None,
        erections_start: str | None,
        erections_end: str | None,
        erections_search: str | None,
        mode_value: str | None,
    ):
        if not (main_clicks or modal_clicks):
            raise PreventUpdate

        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        gang_list = _ensure_list(gangs)

        df_day = data_selector.select(mode_value)
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

        return send_bytes(_writer, "Trace_Calcs.xlsx")
    @app.callback(
        Output("trace-gang", "options"),
        Output("trace-gang", "value"),
        Output("modal-trace-gang", "options"),
        Output("modal-trace-gang", "value"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("store-selected-gang", "data"),
        Input("store-mode", "data"),
        State("trace-gang", "value"),
    )
    def update_trace_gang_options(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        clicked_gang: str | None,
        mode_value: str | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, str]], str | None, list[dict[str, str]], str | None]:
        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)

        df_day = data_selector.select(mode_value)
        months_ts = resolve_months(month_list, quick_range)
        base = apply_filters(df_day, project_list, months_ts, [])
        # Be defensive when switching to Stringing with no data configured
        if not isinstance(base, pd.DataFrame) or base.empty or "gang_name" not in base.columns:
            options: list[dict[str, str]] = []
            value = None
            return options, value, options, value
        gangs = (
            base["gang_name"].dropna().astype(str).str.strip().unique().tolist()
        )
        gangs = sorted([g for g in gangs if g])
        options = [{"label": gang, "value": gang} for gang in gangs]

        if clicked_gang and clicked_gang in gangs:
            value = clicked_gang
        elif current_value and current_value in gangs:
            value = current_value
        else:
            value = None
        return options, value, options, value

    # Hidden debug text to confirm source switching
    @app.callback(
        Output("mode-data-debug", "children"),
        Input("store-mode", "data"),
        prevent_initial_call=False,
    )
    def _debug_mode_data(mode_value: str | None) -> str:
        try:
            df = data_selector.select(mode_value)
            rows = int(len(df.index)) if hasattr(df, "index") else 0
            mode = (mode_value or "erection").strip().lower()
            return f"mode={mode}; rows={rows}"
        except Exception as exc:  # pragma: no cover
            mode = (mode_value or "erection").strip().lower()
            return f"mode={mode}; error={type(exc).__name__}"


    # --- KPI Details drilldown: populate inline accordion ---
    @app.callback(
        Output("kpi-pch-accordion", "children"),
        Output("kpi-pch-accordion", "active_item"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("store-mode", "data"),
    )
    def _populate_kpi_pch(projects, months, quick_range, mode_value: str | None):
        mode = (mode_value or "erection").strip().lower()
        try:
            import re as _re_slug
        except Exception:  # pragma: no cover - regex should be available, but keep fallback
            _re_slug = None

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
        # Erection mode (existing flow)
        if mode != "stringing":
            # fall through to original erection implementation below
            pass
        else:
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
            delivered_km_current_series = pd.Series(dtype=float)
            if isinstance(scoped, pd.DataFrame) and not scoped.empty:
                proj_col = "project_name" if "project_name" in scoped.columns else ("project" if "project" in scoped.columns else None)
                if proj_col is None:
                    return _empty_pch_items("Missing project information in the dataset."), None
                delivered_km_by_project = (
                    scoped.groupby(scoped[proj_col].astype(str))
                          .agg({"daily_km": "sum"})
                          .rename(columns={"daily_km": "delivered_km"})
                ) if "daily_km" in scoped.columns else pd.DataFrame(columns=["delivered_km"])
                if "daily_km" in scoped.columns:
                    scoped_month = scoped.copy()
                    if "month" not in scoped_month.columns and "date" in scoped_month.columns:
                        scoped_month["date"] = pd.to_datetime(scoped_month["date"], errors="coerce")
                        scoped_month = scoped_month.dropna(subset=["date"])
                        scoped_month["month"] = scoped_month["date"].dt.to_period("M").dt.to_timestamp()
                    elif "month" in scoped_month.columns:
                        scoped_month["month"] = pd.to_datetime(scoped_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                    if "month" in scoped_month.columns:
                        scoped_current = scoped_month[scoped_month["month"] == current_month_ts]
                        if not scoped_current.empty:
                            delivered_km_current_series = scoped_current.groupby(scoped_current[proj_col].astype(str))["daily_km"].sum()
            else:
                delivered_km_by_project = pd.DataFrame(columns=["delivered_km"])
                delivered_km_current_series = pd.Series(dtype=float)

            # Planned KM from compiled stringing dataset (section-level total length)
            try:
                df_compiled = _load_stringing_compiled_raw(config)
            except Exception:
                df_compiled = pd.DataFrame()
            planned_km_by_project = pd.DataFrame(columns=["planned_km"])
            planned_km_current_series = pd.Series(dtype=float)
            if isinstance(df_compiled, pd.DataFrame) and not df_compiled.empty:
                comp = df_compiled.copy()
                # Detect project/name column robustly
                comp_proj_col = None
                for cand in ("project_name", "project", "Project Name", "Project"):
                    if cand in comp.columns:
                        comp_proj_col = cand
                        break
                if comp_proj_col is not None:
                    # Try to filter by month using FS completion or PO start month if present
                    date_col = None
                    for dc in ("fs_complete_date", "fs_completed_date", "fs_date", "po_start_date", "date"):
                        if dc in comp.columns:
                            date_col = dc
                            break
                    if date_col is not None and months_ts:
                        comp[date_col] = pd.to_datetime(comp[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
                        comp = comp[comp[date_col].isin(months_ts)].copy()
                    # Ensure length_km exists
                    if "length_km" not in comp.columns and "length_m" in comp.columns:
                        comp["length_km"] = pd.to_numeric(comp["length_m"], errors="coerce") / 1000.0
                    comp["length_km"] = pd.to_numeric(comp.get("length_km", np.nan), errors="coerce")
                    planned_km_by_project = (
                        comp.groupby(comp[comp_proj_col].astype(str))["length_km"].sum().to_frame("planned_km")
                    )
                    if date_col is not None and date_col in comp.columns:
                        comp_current = comp[comp[date_col] == current_month_ts].copy()
                        if not comp_current.empty:
                            planned_km_current_series = comp_current.groupby(comp_current[comp_proj_col].astype(str))["length_km"].sum()

            # Merge planned and delivered into a projects table
            projects_df = (
                planned_km_by_project
                .join(delivered_km_by_project, how="outer")
                .fillna(0.0)
                .reset_index()
                .rename(columns={"index": "project_name"})
            )
            if "project_name" not in projects_df.columns:
                # If group key preserved original name
                for cand in ("project_name", "project", "Project Name"):
                    if cand in projects_df.columns:
                        projects_df = projects_df.rename(columns={cand: "project_name"})
                        break
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
                rec = {
                    "project_name": proj,
                    "project_code": (meta.get("project_code", pd.Series([proj])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else proj),
                    "regional_mgr": (meta.get("regional_mgr", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns) else ""),
                    "project_mgr": (meta.get("project_mgr", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns) else ""),
                    "planning_eng": (meta.get("planning_eng", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns) else ""),
                    # store KM in MT fields to reuse downstream structure but change labels to KM
                    "planned_mt": round(planned_km, 1),
                    "delivered_mt": round(delivered_km, 1),
                    # counts not applicable for stringing
                    "planned_nos": 0,
                    "delivered_nos": 0,
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
                overall_map, _ = compute_project_baseline_maps_for(day_filtered, "daily_km")
                prod_overall_norm_map, prod_overall_compact_map = _build_lookup_maps(overall_map)
                current_scope = day_filtered[day_filtered["month"] == current_month_ts] if "month" in day_filtered.columns else pd.DataFrame()
                if not current_scope.empty:
                    current_map, _ = compute_project_baseline_maps_for(current_scope, "daily_km")
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

                def _project_lookup_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
                    display = str(row.get("project_name", "")).strip()
                    base = display.split(" : ", 1)[1] if " : " in display else display
                    code = str(row.get("project_code", "")).strip()
                    texts = [display, base, code]
                    norm_keys: list[str] = []
                    compact_keys: list[str] = []
                    for text in texts:
                        if not text or text.lower() == "nan":
                            continue
                        norm_key = _normalize_lower(text)
                        if norm_key and norm_key not in norm_keys:
                            norm_keys.append(norm_key)
                        compact_key = _compact_code(text)
                        if compact_key and compact_key not in compact_keys:
                            compact_keys.append(compact_key)
                    return norm_keys, compact_keys

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
                    code_value = str(r.get("project_code", "")).strip()
                    if code_value and code_value.lower() not in ("", "nan") and code_value not in project_codes:
                        project_codes.append(code_value)

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
                    if planned_month_val is not None:
                        planned_month_total += float(planned_month_val)

                prod_current_avg = float(sum(prod_current_values) / len(prod_current_values)) if prod_current_values else None
                prod_overall_avg = float(sum(prod_overall_values) / len(prod_overall_values)) if prod_overall_values else None

                fmt_prod_current = f"{prod_current_avg:.2f}" if prod_current_avg is not None else "\u2014"
                fmt_prod_overall = f"{prod_overall_avg:.2f}" if prod_overall_avg is not None else "\u2014"
                projects_label = f"Projects: {', '.join(project_codes)}" if project_codes else f"Projects: {project_count}"
                km_delivered_label = round(delivered_month_total, 1)
                km_planned_label = round(planned_month_total, 1)

                title_component = html.Div(
                    [
                        html.Span(str(pch or "Unassigned"), className="fw-semibold"),
                        html.Div(
                            [
                                html.Span(projects_label, className="pch-pill pch-pill-projects mb-1"),
                                html.Span(f"Prod This Month: {fmt_prod_current} KM/day", className="pch-pill pch-pill-prod-month mb-1"),
                                html.Span(f"Prod Overall: {fmt_prod_overall} KM/day", className="pch-pill pch-pill-prod-overall mb-1"),
                                html.Span(
                                    f"KM This Month: {km_delivered_label:.1f} delivered / {km_planned_label:.1f} planned",
                                    className="pch-pill pch-pill-towers mb-1",
                                ),
                            ],
                            className="pch-pill-group ms-auto d-none d-md-flex",
                        ),
                    ],
                    className="d-flex align-items-center justify-content-between w-100",
                )

                tile_cols = []
                for r in sorted(rows, key=lambda x: str(x["project_name"])):
                    proj_name = str(r["project_name"]).strip()
                    # Build robust compact code for modal open
                    try:
                        import re as _re
                        def _compact_code_text(s: str) -> str:
                            return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                    except Exception:
                        def _compact_code_text(s: str) -> str:
                            return (s or "").strip().lower().replace(" ", "")
                    raw_code = r.get("project_code") or r.get("project_key") or proj_name
                    proj_code = _compact_code_text(str(raw_code))
                    current_month_value = pd.Timestamp.today().strftime("%Y-%m")
                    current_month_label = pd.Timestamp.today().strftime("%b %Y")
                    current_key_payload = "||".join([
                        proj_code or "",
                        current_month_value,
                        proj_name,
                    ])
                    tile_body = html.Div([
                        html.Div(html.Strong(proj_name), className="mb-2"),
                        html.Div([
                            html.Span("Regional Manager : ", className="text-muted me-1"),
                            dbc.Badge(r.get("regional_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold")
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Project Manager : ", className="text-muted me-1"),
                            dbc.Badge(r.get("project_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Length Stringed : ", className="me-2"),
                            dbc.Badge(f"{r['delivered_mt']:.1f} / {r['planned_mt']:.1f} KM", color="dark", className="me-2", style={"fontSize": "1.05rem"}),
                        ], className="mb-2"),
                        html.Div([
                            html.Span("View Responsibilities : ", className="me-2"),
                            # For stringing, responsibilities still open the erection responsibilities modal filtered by project
                            dbc.Button(
                                current_month_label,
                                id={"type": "proj-resp-open", "key": current_key_payload},
                                color="link",
                                className="p-0 me-1",
                            ),
                        ], className="mb-2"),
                    ])
                    tile_cols.append(
                        dbc.Col(
                            dbc.Card(dbc.CardBody(tile_body), className="h-100 shadow-sm"),
                            xs=12, sm=12, md=6, lg=4, className="mb-3"
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

            return pch_sections, None

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
        if has_responsibilities_provider:
            df_mp_frame, _, _, _ = _fetch_responsibilities()
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
            proj_display2 = f"{proj_code2} : {proj_display_name2}".strip(" :") if proj_code2 else proj_display_name2
            key = (pch_label, proj_display2)
            if key in aggregated:
                continue
            mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
            nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
            aggregated[key] = {
                "pch": pch_label,
                "project_name": proj_display2,
                "project_code": proj_code2,
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
                code = str(meta_row.get("project_code", meta_row.get("Project Code", "")))
                proj_key = _compact_code(code) or _compact_code(proj)
                if not proj or proj_key in aggregated_by_proj_key:
                    continue
                raw_pch = str(meta_row.get(pch_col, "")) if pch_col in meta_row.index else ""
                pch_label = (_normalize_pch(raw_pch) or str(raw_pch or "").strip())
                proj_display = f"{code} : {proj}".strip(" :") if code else proj
                key = (pch_label, proj_display)
                aggregated[key] = {
                    "pch": pch_label,
                    "project_name": proj_display,
                    "project_code": (str(meta_row.get("project_code", meta_row.get("Project Code", ""))) if isinstance(meta_row, pd.Series) else ""),
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
                    prod_current_raw, _ = compute_project_baseline_maps_for(current_scope, "daily_prod_mt")
                    prod_current_norm_map, prod_current_compact_map = _build_metric_lookup(prod_current_raw)

                history_scope = day_filtered[day_filtered["month"] < current_month_ts]
                if not history_scope.empty:
                    prod_history_raw, _ = compute_project_baseline_maps_for(history_scope, "daily_prod_mt")
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
            display = str(row.get("project_name", "")).strip()
            base = display.split(" : ", 1)[1] if " : " in display else display
            code = str(row.get("project_code", "")).strip()
            texts = [display, base, code]
            norm_keys: list[str] = []
            compact_keys: list[str] = []
            for text in texts:
                if not text:
                    continue
                norm_key = _normalize_lower(text)
                if norm_key and norm_key not in norm_keys:
                    norm_keys.append(norm_key)
                compact_key = _compact_key(text)
                if compact_key and compact_key not in compact_keys:
                    compact_keys.append(compact_key)
            return norm_keys, compact_keys

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
                code_value = str(r.get("project_code", "")).strip()
                if not code_value:
                    project_label = str(r.get("project_name", "")).strip()
                    if " : " in project_label:
                        code_value = project_label.split(" : ", 1)[0].strip()
                if code_value and code_value.lower() != "nan" and code_value not in project_codes:
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

            header_pills = [
                html.Span(projects_label, className="pch-pill pch-pill-projects me-2 mb-1"),
                html.Span(f"Prod This Month: {fmt_prod_current} MT/day", className="pch-pill pch-pill-prod-month me-2 mb-1"),
                html.Span(f"Prod Overall: {fmt_prod_history} MT/day", className="pch-pill pch-pill-prod-overall me-2 mb-1"),
                html.Span(
                    f"Towers This Month: {towers_delivered_label} delivered / {towers_planned_label} planned",
                    className="pch-pill pch-pill-towers me-2 mb-1",
                ),
            ]

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
                # Build compact code for modal IDs (match against project_key)
                try:
                    import re as _re
                    def _compact_code_text(s: str) -> str:
                        return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                except Exception:
                    def _compact_code_text(s: str) -> str:
                        return (s or "").strip().lower().replace(" ", "")
                raw_code = r.get("project_code") or r.get("project_key") or proj_name
                proj_code = _compact_code_text(str(raw_code))
                tile_body_children = [
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
                        out.append(html.Span("View Responsibilities : ", className="me-2"))
                        # cap to last 2 months only
                        months_vals = sorted(months_vals)
                        months_vals = months_vals[-2:]
                        for ts in months_vals:
                            label = ts.strftime("%b %Y")
                            value = ts.strftime("%Y-%m")
                            key_payload = "||".join([
                                proj_code or "",
                                value or "",
                                name_part.strip(),
                            ])
                            out.append(
                                dbc.Button(
                                    label,
                                    id={"type": "proj-resp-open", "key": key_payload},
                                    color="link",
                                    className="p-0 me-1",
                                )
                            )
                    return out

                # Common stats section
                tile_body_children.extend([
                    html.Div([
                        html.Span("Towers Erected : ", className="me-2"),
                        (dbc.Badge(f"{r['delivered_nos']} / {r['planned_nos']}", color="primary", className="me-2", style={"fontSize": "1.05rem"}) if has_mp else None),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Volume Erected : ", className="me-2"),
                        (dbc.Badge(f"{r['delivered_mt']:.1f} / {r['planned_mt']:.1f} MT", color="dark", className="me-2", style={"fontSize": "1.05rem"}) if has_mp else None),
                    ], className="mb-2"),
                ])
                month_buttons = _month_buttons_for_project()
                if month_buttons:
                    tile_body_children.append(html.Div(month_buttons, className="mb-1"))
                else:
                    tile_body_children.append(html.Div("Micro Plan not available.", className="text-muted"))

                tile_body = html.Div(tile_body_children)

                tile_cols.append(
                    dbc.Col(
                        dbc.Card(dbc.CardBody(tile_body), className="h-100 shadow-sm"),
                        xs=12, sm=12, md=6, lg=4, className="mb-3"
                    )
                )

            body_children = (
                [dbc.Row(tile_cols, className="g-3")]
                if tile_cols
                else [html.Div("No projects available.", className="text-muted")]
            )
            title_component = html.Div(
                [
                    html.Span(str(pch or "Unassigned"), className="fw-semibold"),
                    html.Div(
                        [
                            html.Span(projects_label, className="pch-pill pch-pill-projects mb-1"),
                            html.Span(f"Prod This Month: {fmt_prod_current} MT/day", className="pch-pill pch-pill-prod-month mb-1"),
                            html.Span(f"Prod Overall: {fmt_prod_history} MT/day", className="pch-pill pch-pill-prod-overall mb-1"),
                            html.Span(
                                f"Towers This Month: {towers_delivered_label} delivered / {towers_planned_label} planned",
                                className="pch-pill pch-pill-towers mb-1",
                            ),
                        ],
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

        return pch_sections, None

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

    # --- Project Responsibilities mini-modal: open/close and set project code ---
    @app.callback(
        Output("proj-resp-modal", "is_open"),
        Output("proj-resp-modal-title", "children"),
        Output("store-proj-resp-code", "data"),
        Output("store-proj-resp-month", "data"),
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
            return False, dash.no_update, dash.no_update, None
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
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if id_obj.get("type") != "proj-resp-open":
            raise PreventUpdate

        code = name = month_raw = None
        if isinstance(key_str, str):
            parts = key_str.split("||")
            if parts:
                code = parts[0] or None
            if len(parts) > 1:
                month_raw = parts[1] or None
            if len(parts) > 2:
                name = parts[2] or None

        month_value, month_label = _normalize_month_value(month_raw)
        display_title = name or code
        if display_title:
            title = f"Responsibilities \u2014 {display_title}"
        else:
            title = "Responsibilities"
        if month_label:
            title = f"{title} ({month_label})"
        payload = {"code": code, "name": name}
        return True, title, payload, month_value

    # --- Render responsibilities inside the project mini-modal ---
    @app.callback(
        Output("proj-resp-graph", "figure"),
        Output("proj-resp-kpi-target", "children"),
        Output("proj-resp-kpi-delivered", "children"),
        Output("proj-resp-kpi-ach", "children"),
        Input("store-proj-resp-code", "data"),
        Input("store-proj-resp-month", "data"),
        Input("proj-resp-entity", "value"),
        Input("proj-resp-metric", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("proj-resp-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _render_proj_resp(code_value, month_value, entity_value, metric_value, months_value, quick_value, is_open):
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
        return _build_responsibilities_for_project(
            project_value=project_identifiers,
            entity_value=entity_value,
            metric_value=metric_value,
            months_value=months_value,
            quick_range_value=quick_value,
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


    # Mode-aware labels and table column headers
    @app.callback(
        Output("label-avg", "children"),
        Output("label-total", "children"),
        Output("label-lost", "children"),
        Output("tbl-idle-intervals", "columns"),
        Output("modal-tbl-idle-intervals", "columns"),
        Output("tbl-daily-prod", "columns"),
        Output("modal-tbl-daily-prod", "columns"),
        Input("store-mode", "data"),
        prevent_initial_call=False,
    )
    def _mode_labels_and_tables(mode_value: str | None):
        mode = (mode_value or "erection").strip().lower()
        is_stringing = mode == "stringing"
        unit_short = "KM" if is_stringing else "MT"
        avg_label = "Avg Output / Gang / Month" if is_stringing else "Avg Output / Gang / Day"
        total_label = "Delivered (KM)" if is_stringing else "Volume Erected"
        lost_label = "Lost (KM)" if is_stringing else "Lost Units"

        idle_cols = [
            {"name": "Gang", "id": "gang_name"},
            {"name": "Interval Start", "id": "interval_start"},
            {"name": "Interval End", "id": "interval_end"},
            {"name": "Raw Gap (days)", "id": "raw_gap_days"},
            {"name": "Idle Counted (days)", "id": "idle_days_capped"},
            {"name": f"Baseline ({unit_short}/day)", "id": "baseline"},
            {"name": f"Cumulative Loss ({unit_short})", "id": "cumulative_loss"},
        ]
        daily_cols = [
            {"name": "Gang", "id": "gang_name"},
            {"name": "Project", "id": "project_name"},
            {"name": "Date", "id": "date"},
            {"name": f"{unit_short}/day", "id": "daily_prod_mt"},
        ]
        return avg_label, total_label, lost_label, idle_cols, idle_cols, daily_cols, daily_cols

