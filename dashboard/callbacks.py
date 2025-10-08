"""Dash callbacks for the productivity dashboard."""

from __future__ import annotations

import logging
import dash_bootstrap_components as dbc 
import pandas as pd
from io import BytesIO
from typing import Any, Callable, Sequence

import dash
import re
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes

from .charts import (
    # create_monthly_line_chart,
    create_project_lines_chart,
    create_top_bottom_gangs_charts,
    build_responsibilities_chart,
    build_empty_responsibilities_figure,
)
from .config import AppConfig
from .filters import apply_filters, resolve_months
from .metrics import calc_idle_and_loss, compute_idle_intervals_per_gang
from .workbook import make_trace_workbook_bytes


LOGGER = logging.getLogger(__name__)

BENCHMARK_MT_PER_DAY = 9.0



_slug = lambda s: re.sub(r"[^a-z0-9_-]+", "-", str(s).lower()).strip("-")

def _render_avp_row(gang, delivered, lost, total, pct, avg_prod=0.0, baseline=0.0, last_project="—", last_date="—"):
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
                html.Span([html.Span(f"{delivered:.0f} MT", className="del"), " vs ", html.Span(f"{lost:.0f} MT lost", className="los")]),
                html.Span(f"{total:.0f} MT", className="tot"),
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
                    html.Div(f"Current MT/day: {avg_prod:.2f}"),
                    html.Div(f"Baseline MT/day: {baseline:.2f}"),
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
    project_info_provider: Callable[[], pd.DataFrame] | None = None,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    responsibilities_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    responsibilities_error_provider: Callable[[], str | None] | None = None,
) -> None:
    """Register all Dash callbacks on *app*."""

    LOGGER.debug("Registering callbacks")

    # Charts OR AVP rows -> store-click-meta (robust & single source of truth)
    app.clientside_callback(
        """
        // charts OR AVP (row or overlay) -> store-click-meta
        function(lossClick, topClick, bottomClick, rowTs, tipTs) {
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;

        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];

        // --- AVP surfaces (row or overlay) — only accept real, timestamped clicks
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




    # # ONE unified clientside callback for BOTH graph clicks and AVP row clicks
    # app.clientside_callback(
    #     """
    #     function(lossClick, topClick, bottomClick, rowClicks, rowIds, prev) {
    #     const C = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
    #     if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;
    #     const trg = ctx.triggered[0].prop_id || "";

    #     // ---- robust: parse the pattern id JSON instead of relying on key order
    #     try {
    #         const idPart = trg.split(".")[0];
    #         const pid = JSON.parse(idPart);
    #         if (pid && pid.type === "avp-row") {
    #         if (!rowClicks || !rowClicks.length) return NO;
    #         let last = -1;
    #         for (let i = 0; i < rowClicks.length; i++) { if (rowClicks[i]) last = i; }
    #         if (last < 0) return NO;
    #         const gang = rowIds && rowIds[last] && rowIds[last].index;
    #         return gang || NO;
    #         }
    #     } catch (e) { /* ignore and fall through to chart clicks */ }

    #     // ---- charts (unchanged)
    #     let cd = null;
    #     if (trg.startsWith("g-actual-vs-bench.")) cd = lossClick;
    #     else if (trg.startsWith("g-top5."))        cd = topClick;
    #     else if (trg.startsWith("g-bottom5."))     cd = bottomClick;
    #     else return NO;

    #     if (!cd || !cd.points || !cd.points.length) return NO;
    #     const pt = cd.points[0];
    #     let gang = null;
    #     if (typeof pt.y === "string")      gang = pt.y;
    #     else if (typeof pt.x === "string") gang = pt.x;
    #     else if (pt.customdata) {
    #         if (typeof pt.customdata === "string")       gang = pt.customdata;
    #         else if (Array.isArray(pt.customdata))       gang = pt.customdata.find(v => typeof v === "string") || null;
    #         else if (typeof pt.customdata === "object")  gang = pt.customdata.gang || pt.customdata.name || null;
    #     }
    #     return gang || NO;
    #     }
    #     """,
    #     Output("store-selected-gang", "data"),
    #     [
    #     Input("g-actual-vs-bench", "clickData"),
    #     Input("g-top5", "clickData"),
    #     Input("g-bottom5", "clickData"),
    #     Input({"type":"avp-row","index": dash.dependencies.ALL}, "n_clicks"),
    #     ],
    #     [
    #     State({"type":"avp-row","index": dash.dependencies.ALL}, "id"),
    #     State("store-selected-gang", "data"),
    #     ],
    # )
    

    

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
        if trigger_id == "link-clear-quick-range":
            return dash.no_update, dash.no_update, dash.no_update, None
        return None, None, None, None


    @app.callback(
        Output("f-project", "options"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
    )
    def update_project_options(
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
    ) -> list[dict[str, str]]:
        df_day = data_provider()
        months_ts = resolve_months(_ensure_list(months), quick_range)

        filtered = df_day.copy()
        if months_ts:
            filtered = filtered[filtered["month"].isin(months_ts)]
        gang_list = _ensure_list(gangs)
        if gang_list:
            filtered = filtered[filtered["gang_name"].isin(gang_list)]

        projects = sorted(filtered["project_name"].dropna().unique())
        if not projects:
            projects = sorted(df_day["project_name"].dropna().unique())
        return [{"label": project, "value": project} for project in projects]

    @app.callback(
        Output("f-gang", "options"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def update_gang_options(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
    ) -> list[dict[str, str]]:
        df_day = data_provider()
        filtered = df_day.copy()
        project_list = _ensure_list(projects)
        if project_list:
            filtered = filtered[filtered["project_name"].isin(project_list)]
        months_ts = resolve_months(_ensure_list(months), quick_range)
        if months_ts:
            filtered = filtered[filtered["month"].isin(months_ts)]

        gangs = sorted(filtered["gang_name"].dropna().unique())
        if not gangs:
            gangs = sorted(df_day["gang_name"].dropna().unique())
        return [{"label": gang, "value": gang} for gang in gangs]

    @app.callback(
        Output("f-month", "options"),
        Input("f-project", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
    )
    def update_month_options(
        projects: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
    ) -> list[dict[str, str]]:
        df_day = data_provider()
        filtered = df_day.copy()
        project_list = _ensure_list(projects)
        if project_list:
            filtered = filtered[filtered["project_name"].isin(project_list)]
        gang_list = _ensure_list(gangs)
        if gang_list:
            filtered = filtered[filtered["gang_name"].isin(gang_list)]

        months = sorted(filtered["month"].dropna().unique())
        if quick_range:
            months_range = resolve_months(None, quick_range)
            months = [month for month in months if month in months_range]
        if not months:
            months = sorted(df_day["month"].dropna().unique())
        return [
            {"label": month.strftime("%b %Y"), "value": month.strftime("%Y-%m")}
            for month in months
        ]

    @app.callback(
        Output("f-month", "value", allow_duplicate=True),
        Input("f-quick-range", "value"),
        prevent_initial_call=True,
    )
    def _clear_month_value_on_quick_change(qr):
        # When a quick-range is chosen, let code derive months from it; drop stale manual months.
        return None


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
        Output("pd-title", "children"),          # header text
        Output("project-details", "children"),   # body (message OR 3-col grid)
        Input("f-project", "value"),
        prevent_initial_call=False,
    )
    def show_project_details(selected_project):
        """
        If no selection -> show friendly message.
        If selection found -> render the 3-column grid with real values.
        Uses project_info_provider() as in your existing code.
        """
        default_title = "Project Overview"

        # Normalize list vs string from a multi=True dropdown
        if isinstance(selected_project, (list, tuple)):
            if len(selected_project) != 1:
                return (
                    default_title,
                    html.Div("Select a single project to view its details.", className="project-empty"),
                )
            selected_project = selected_project[0]

        # 1) No selection -> message
        if not selected_project:
            return (
                default_title,
                html.Div("Select a single project to view its details.", className="project-empty"),
            )

        # 2) Source existence checks (same expectations as your current code)
        if not project_info_provider:
            return (
                default_title,
                html.Div("No 'Project Details' source configured.", className="project-empty"),
            )

        df_info = project_info_provider()  # must return a DataFrame
        if df_info is None or df_info.empty:
            return (
                default_title,
                html.Div("No 'Project Details' sheet found in the source workbook.", className="project-empty"),
            )

        pname = str(selected_project).strip()

        # Try exact match first, then normalized match on 'Project Name'
        row = df_info[df_info.get("Project Name") == pname]
        if "Project Name" in df_info.columns:
            row = df_info[df_info["Project Name"].astype(str).str.strip() == pname]
        
        if row.empty and "Project Name" in df_info.columns:
            norm = lambda s: " ".join(str(s).strip().lower().split())
            row = df_info[df_info["Project Name"].apply(norm) == norm(pname)]

        if row.empty:
            return (
                default_title,
                html.Div(f"No project details found for “{pname}”.", className="project-empty"),
            )

        r = row.iloc[0]

        # Helpers (same as you used)
        def fmt_txt(key: str) -> str:
            v = r.get(key, "")
            return "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v).strip()

        def fmt_date(key: str) -> str:
            v = r.get(key, None)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            try:
                return pd.to_datetime(v).strftime("%d-%m-%Y")
            except Exception:
                return str(v)

        # 3) Build the 3-column body (matches your new CSS skin)
        body = html.Div(
            [
                # Column 1 (Left)
                html.Div(
                    [
                        html.P("PROJECT CODE", className="project-label"),
                        html.H6(fmt_txt("project_code"), className="project-value"),

                        html.P("CLIENT", className="project-label"),
                        html.H6(fmt_txt("client_name"), className="project-value"),

                        html.P("NOA START", className="project-label"),
                        html.H6(fmt_date("noa_start"), className="project-value"),

                        html.P("LOA END", className="project-label"),
                        html.H6(fmt_date("loa_end"), className="project-value"),
                    ],
                    className="project-col",
                ),

                # Column 2 (Middle)
                html.Div(
                    [
                        html.P("PCH", className="project-label"),
                        html.H6(fmt_txt("pch"), className="project-value"),

                        html.P("REGIONAL MANAGER", className="project-label"),
                        html.H6(fmt_txt("regional_mgr"), className="project-value"),

                        html.P("PROJECT MANAGER", className="project-label"),
                        html.H6(fmt_txt("project_mgr"), className="project-value"),

                        html.P("PLANNING ENGINEER", className="project-label"),
                        html.H6(fmt_txt("planning_eng"), className="project-value"),
                    ],
                    className="project-col",
                ),

                # Column 3 (Right)
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

        title = f"{pname} – Project Overview"
        return title, body



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
    )
    def update_responsibilities(
        project_value: str | None,
        entity_value: str | None,
        metric_value: str | None,
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

        load_error_msg: str | None = None
        completed_keys: set[tuple[str, str]] = set()
        df_atomic: pd.DataFrame | None = None
        workbook: pd.ExcelFile | None = None

        if responsibilities_provider is not None:
            try:
                df_atomic = responsibilities_provider()
            except RuntimeError as exc:
                LOGGER.warning("Responsibilities data unavailable: %s", exc)
                return _empty_response(str(exc))
            except Exception as exc:
                LOGGER.exception("Failed to access responsibilities data: %s", exc)
                return _empty_response("Unable to load Micro Plan data.")
            else:
                df_atomic = df_atomic.copy()
                if responsibilities_completion_provider is not None:
                    try:
                        completed_keys = set(responsibilities_completion_provider())
                    except Exception as exc:
                        LOGGER.warning("Failed to resolve responsibilities completion keys: %s", exc)
                        completed_keys = set()
                if responsibilities_error_provider is not None:
                    try:
                        load_error_msg = responsibilities_error_provider()
                    except Exception:
                        load_error_msg = None
        else:
            cfg = config
            try:
                workbook = pd.ExcelFile(cfg.data_path)
            except FileNotFoundError:
                LOGGER.warning("Responsibilities workbook not found: %s", cfg.data_path)
                return _empty_response("Compiled workbook not found.")
            except Exception as exc:
                LOGGER.exception("Failed to open responsibilities workbook: %s", exc)
                return _empty_response("Unable to load Micro Plan data.")

            atomic_sheet = "MicroPlanResponsibilities"
            if atomic_sheet not in workbook.sheet_names:
                LOGGER.warning("Sheet '%s' missing in workbook", atomic_sheet)
                return _empty_response("No Micro Plan data found in the compiled workbook.")

            df_atomic = pd.read_excel(workbook, sheet_name=atomic_sheet)

        if df_atomic is None or df_atomic.empty:
            message = load_error_msg or "No Micro Plan data found in the compiled workbook."
            return _empty_response(message)

        df_atomic = df_atomic.copy()

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

        if responsibilities_provider is None and workbook is not None:
            daily_sheet = None
            for candidate in ("Daily Expanded", "DailyExpanded"):
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
                    "Daily Expanded sheet not found; delivered values fall back to realised revenue only."
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
        Input("f-topbot-metric", "value"),
    )
    def update_dashboard(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        topbot_metric: str | None,         
    ) -> tuple:
        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        gang_list = _ensure_list(gangs)

        df_day = data_provider()
        months_ts = resolve_months(month_list, quick_range)

        scoped = apply_filters(df_day, project_list, months_ts, gang_list)
        scoped_top_bottom = apply_filters(df_day, project_list, months_ts, [])

        benchmark = BENCHMARK_MT_PER_DAY
        avg_prod = scoped["daily_prod_mt"].mean() if len(scoped) else 0.0
        delta_pct = (avg_prod - benchmark) / benchmark * 100 if benchmark else None
        kpi_avg = f"{avg_prod:.2f} MT"
        kpi_delta = (
            "(n/a)"
            if delta_pct is None
            else f"({delta_pct:+.0f}% vs {benchmark:.1f} MT)"
        )

        has_selected_months = bool(months_ts)

        scope_mask = pd.Series(True, index=df_day.index)
        if project_list:
            scope_mask &= df_day["project_name"].isin(project_list)
        if gang_list:
            scope_mask &= df_day["gang_name"].isin(gang_list)
        scoped_all = df_day.loc[scope_mask].copy()

        if has_selected_months and not scoped_all.empty and "month" in scoped_all:
            month_values = sorted(set(months_ts))
            period_mask = scoped_all["month"].isin(month_values)
            loss_scope = scoped_all.loc[period_mask].copy()
            earliest_month = month_values[0]
            history_scope = scoped_all.loc[scoped_all["month"] < earliest_month].copy()
        else:
            loss_scope = scoped_all.copy()
            history_scope = scoped_all.copy()

        baseline_map = {}
        if not history_scope.empty and "gang_name" in history_scope:
            baseline_map = (
                history_scope.groupby("gang_name")["daily_prod_mt"].mean().dropna().to_dict()
            )

        loss_rows: list[dict[str, float]] = []
        for gang_name, gang_df in loss_scope.groupby("gang_name"):
            override_baseline = baseline_map.get(gang_name)
            idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                gang_df,
                loss_max_gap_days=config.loss_max_gap_days,
                baseline_mt_per_day=override_baseline,
            )
            loss_rows.append(
                {
                    "gang_name": gang_name,
                    "delivered": delivered,
                    "lost": loss_mt,
                    "potential": potential,
                    "avg_prod": gang_df["daily_prod_mt"].mean(),
                    "baseline": baseline,
                }
            )
        if loss_rows:

            loss_df = pd.DataFrame(loss_rows)

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
            meta = (
                scoped_all.loc[idx_last, ["gang_name", "project_name", "date"]]
                .rename(columns={"project_name": "last_project", "date": "last_date"})
            )
            loss_df = loss_df.merge(meta, on="gang_name", how="left")
        else:
            # guarantee columns exist even when we couldn't compute meta
            loss_df = loss_df.assign(last_project=np.nan, last_date=pd.NaT)

        # pretty, null-safe strings for hover (NO KeyError even if meta missing)
        last_date_series = pd.to_datetime(loss_df.get("last_date"), errors="coerce")
        loss_df["last_date_str"] = last_date_series.dt.strftime("%d-%b-%Y").fillna("—")
        loss_df["last_project"]  = loss_df.get("last_project").fillna("—")

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

                avp_children.append(

                    _render_avp_row(

                        r["gang_name"], float(r["delivered"]), float(r["lost"]),

                        total, pct,

                        avg_prod=float(r.get("avg_prod", 0.0)),

                        baseline=float(r.get("baseline", 0.0)),

                        last_project=str(r.get("last_project", "\uFFFD")),

                        last_date=str(r.get("last_date_str", "\uFFFD")),

                    )

                )











        row_px = 56
        topbot_margin = 120
        fig_height = int(row_px * max(1, len(loss_df)) + topbot_margin)

        active_gangs = loss_scope["gang_name"].nunique()
        total_period_mt = float(loss_scope["daily_prod_mt"].sum()) if not loss_scope.empty else 0.0
        total_delivered = float(loss_df["delivered"].sum()) if not loss_df.empty else 0.0
        total_lost = float(loss_df["lost"].sum()) if not loss_df.empty else 0.0
        total_potential = total_delivered + total_lost
        lost_pct = (total_lost / total_potential * 100) if total_potential > 0 else 0.0

        kpi_active = f"{active_gangs}"
        kpi_total = f"{total_period_mt:.1f} MT"
        kpi_loss = f"{total_lost:.1f} MT"
        kpi_loss_delta = f"{lost_pct:.1f}%"




        fig_loss = go.Figure()
        if not loss_df.empty:
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
                customdata=np.stack(
                    [
                        loss_df["last_project"].fillna("—"),
                        loss_df["last_date_str"].fillna("—"),
                        loss_df["avg_prod"].fillna(0.0),      # current metric (MT/day)
                        loss_df["baseline"].fillna(0.0),      # baseline (MT/day)
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{y}<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked at: %{customdata[1]}<br>"
                    "Current MT/day: %{customdata[2]:.2f}<br>"
                    "Baseline MT/day: %{customdata[3]:.2f}<extra></extra>"
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
                customdata=np.stack(
                    [
                        loss_df["last_project"].fillna("—"),
                        loss_df["last_date_str"].fillna("—"),
                        loss_df["avg_prod"].fillna(0.0),
                        loss_df["baseline"].fillna(0.0),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{y}<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked at: %{customdata[1]}<br>"
                    "Current MT/day: %{customdata[2]:.2f}<br>"
                    "Baseline MT/day: %{customdata[3]:.2f}<extra></extra>"
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
                    text=f"{row['avg_prod']:.2f} MT/day (Baseline: {row['baseline']:.2f} MT/day)",
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
            xaxis_title="Potential (MT)",
            yaxis_title="Gang",
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
            dragmode=False,
        )
        fig_loss.update_layout(hovermode="closest", clickmode="event+select")
        fig_loss.update_xaxes(showspikes=False, fixedrange=True)
        fig_loss.update_yaxes(showspikes=False, fixedrange=True, type="category")
        
        # fig_monthly = create_monthly_line_chart(scoped, bench=benchmark)
        fig_top5, fig_bottom5 = create_top_bottom_gangs_charts(scoped_top_bottom, metric=(topbot_metric or "prod"), baseline_map=baseline_map)
        fig_project = create_project_lines_chart(
            df_day,
            selected_projects=project_list or None,
            bench=benchmark,
            avg_line=avg_prod,   # NEW: show Average (Avg Output / Gang / Day) line
        )

        return (
            kpi_avg,
            kpi_delta,
            kpi_active,
            kpi_total,
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

        df_day = data_provider()
        months_ts = resolve_months(month_list, quick_range)

        base_scope = apply_filters(df_day, project_list, months_ts, []).copy()
        scoped     = apply_filters(df_day, project_list, months_ts, gang_list).copy()

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

        # Idle intervals
        idle_source = pick_gang_scope(gang_focus)
        if idle_source.empty:
            idle_source = scoped if not scoped.empty else base_scope

        idle_df = compute_idle_intervals_per_gang(
            idle_source, loss_max_gap_days=config.loss_max_gap_days
        )
        baseline_lookup: dict[str, float] = {}
        if not idle_source.empty:
            baseline_lookup = {
                gang: calc_idle_and_loss(
                    gang_df,
                    loss_max_gap_days=config.loss_max_gap_days,
                )[1]
                for gang, gang_df in idle_source.groupby("gang_name")
            }
        if not idle_df.empty:
            idle_df["baseline"] = idle_df["gang_name"].map(baseline_lookup).fillna(0.0)
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
        daily_source = daily_source.sort_values(["gang_name", "date"])[
            ["date", "gang_name", "project_name", "daily_prod_mt"]
        ]
        if not daily_source.empty:
            daily_source = daily_source.assign(
                date=daily_source["date"].dt.strftime("%d-%m-%Y"),
                daily_prod_mt=(
                    daily_source["daily_prod_mt"].round(2).map(
                        lambda v: "" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip(".")
                    )
                ),
            )
        daily_data = daily_source.to_dict("records")

        # mirror into modal tables
        return idle_data, daily_data, idle_data, daily_data

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
    ):
        if not (main_clicks or modal_clicks):
            raise PreventUpdate

        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        gang_list = _ensure_list(gangs)

        df_day = data_provider()
        months_ts = resolve_months(month_list, quick_range)
        scoped = apply_filters(df_day, project_list, months_ts, gang_list)
        gang_for_sheet = trace_gang_value or selected_gang
        benchmark_value = BENCHMARK_MT_PER_DAY
        project_info_df = project_info_provider() if project_info_provider else None

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
        State("trace-gang", "value"),
    )
    def update_trace_gang_options(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        clicked_gang: str | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, str]], str | None, list[dict[str, str]], str | None]:
        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)

        df_day = data_provider()
        months_ts = resolve_months(month_list, quick_range)
        base = apply_filters(df_day, project_list, months_ts, [])
        gangs = sorted(base["gang_name"].dropna().unique().tolist())
        options = [{"label": gang, "value": gang} for gang in gangs]

        if clicked_gang and clicked_gang in gangs:
            value = clicked_gang
        elif current_value and current_value in gangs:
            value = current_value
        else:
            value = None
        return options, value, options, value



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












