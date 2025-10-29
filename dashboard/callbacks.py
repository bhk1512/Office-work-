"""Dash callbacks for the productivity dashboard."""

from __future__ import annotations

import logging
import dash_bootstrap_components as dbc 
import pandas as pd
from io import BytesIO
import traceback
from typing import Any, Callable, Sequence

import dash
import re
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from datetime import datetime
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
from .data_loader import load_stringing_daily as _load_stringing_daily
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


LOGGER = logging.getLogger(__name__)

BENCHMARK_MT_PER_DAY = 9.0

# App-wide config instance for callback logic
config = AppConfig()


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


_slug = lambda s: re.sub(r"[^a-z0-9_-]+", "-", str(s).lower()).strip("-")

def _render_avp_row(gang, delivered, lost, total, pct, avg_prod=0.0, baseline=0.0, last_project=" ", last_date=" "):
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
    project_baseline_provider: Callable[[], tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]] | None = None,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    responsibilities_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    responsibilities_error_provider: Callable[[], str | None] | None = None,
) -> None:

    LOGGER.debug("Registering callbacks")

    # Tiny selector to swap daily dataset based on mode
    def _select_daily(mode_value: str | None) -> pd.DataFrame:
        mode = (mode_value or "erection").strip().lower()
        if mode == "stringing":
            df = _load_stringing_daily(config)
        else:
            df = data_provider()
        # Normalize required columns and provide consistent aliases across modes
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # Ensure date and month fields exist and are typed
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).copy()
            if "month" not in df.columns:
                df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        # Provide a unified project column for filters/options
        if "project_name" not in df.columns:
            if "project" in df.columns:
                df["project_name"] = df["project"].astype(str).str.strip()
        # Provide a unified daily metric alias when using stringing data
        if mode == "stringing" and "daily_km" in df.columns and "daily_prod_mt" not in df.columns:
            # For downstream components that expect daily_prod_mt (charts/helpers)
            df["daily_prod_mt"] = df["daily_km"]
        return df

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
            default_month = datetime.today().strftime("%Y-%m")
            return None, [default_month], None, None
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    @app.callback(
        Output("f-project", "options"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_project_options(
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, str]]:
        try:
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = _select_daily(eff_mode)
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

            months_ts = resolve_months(_ensure_list(months), quick_range)

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
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_gang_options(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, str]]:
        try:
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = _select_daily(eff_mode)
            if df_day is None or df_day.empty:
                return []
            work = df_day.copy()
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
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_month_options(
        projects: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        mode_value: str | None,
        toggle_value: str | None,
    ) -> list[dict[str, str]]:
        try:
            # Prefer the live toggle for immediate switching; fall back to stored mode
            eff_mode = (toggle_value or mode_value or "erection")
            df_day = _select_daily(eff_mode)
            if df_day is None or df_day.empty:
                return []
            work = df_day.copy()
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
            # Prefer current month if available, else pick latest option
            opt_values = [opt.get("value") for opt in (options or []) if isinstance(opt, dict)]
            if not opt_values:
                return dash.no_update
            today_val = datetime.today().strftime("%Y-%m")
            if today_val in opt_values:
                return [today_val]
            # Parse to pick latest
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
        candidate_columns = [col for col in ("Project Name", "project_name", "project_code") if col in df_info.columns]

        row = pd.DataFrame()
        for col in candidate_columns:
            mask = df_info[col].apply(_normalize_for_match) == target_norm
            if mask.any():
                row = df_info.loc[mask]
                break

        if row.empty and "Project Name" in df_info.columns:
            series = df_info["Project Name"].astype(str).apply(_clean_text)
            mask = series.str.contains(selected_project, case=False, na=False)
            if mask.any():
                row = df_info.loc[mask]

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

        display_name = fmt_txt("Project Name") or selected_project

        body = html.Div(
            [
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

        title = f"{display_name} Project Overview"
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

        month_list = _ensure_list(months_value)
        months_ts = resolve_months(month_list, quick_range_value)
        active_months = sorted({ts for ts in months_ts if pd.notna(ts)})

        if 'completion_date' in df_atomic.columns:
            df_atomic['completion_month'] = pd.to_datetime(
                df_atomic['completion_date'], errors='coerce'
            ).dt.to_period('M').dt.to_timestamp()
        else:
            df_atomic['completion_month'] = pd.NaT

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
        Input("store-mode", "data"),
        Input("mode-toggle", "value"),
    )
    def update_dashboard(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
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
            df_day = _select_daily(eff_mode)
            months_ts = resolve_months(month_list, quick_range)

            scoped = apply_filters(df_day, project_list, months_ts, gang_list)
            scoped_top_bottom = apply_filters(df_day, project_list, months_ts, [])

            is_stringing = (str(eff_mode or "erection").strip().lower() == "stringing")
            metric_col = "daily_km" if is_stringing else "daily_prod_mt"
            unit_short = "KM" if is_stringing else "MT"
            benchmark = BENCHMARK_MT_PER_DAY
            avg_prod = scoped[metric_col].mean() if len(scoped) and (metric_col in scoped.columns) else 0.0
            delta_pct = (avg_prod - benchmark) / benchmark * 100 if benchmark else None
            kpi_avg = f"{avg_prod:.2f} {unit_short}"
            kpi_delta = (
                "(n/a)"
                if delta_pct is None
                else f"({delta_pct:+.0f}% vs {benchmark:.1f} {unit_short})"
            )

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
        # totals and units by mode
        total_metric = float(loss_scope[metric_col].sum()) if (not loss_scope.empty and metric_col in loss_scope.columns) else 0.0
        total_delivered = float(loss_df["delivered"].sum()) if not loss_df.empty else 0.0
        total_lost = float(loss_df["lost"].sum()) if not loss_df.empty else 0.0
        total_potential = total_delivered + total_lost
        lost_pct = (total_lost / total_potential * 100) if total_potential > 0 else 0.0

        kpi_active = f"{active_gangs}"
        kpi_total = f"{total_metric:.1f} {unit_short}"
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
                    f"Current {unit_short}/day: %{{customdata[2]:.2f}}<br>"
                    f"Baseline {unit_short}/day: %{{customdata[3]:.2f}}<br>"
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
                    f"Current {unit_short}/day: %{{customdata[2]:.2f}}<br>"
                    f"Baseline {unit_short}/day: %{{customdata[3]:.2f}}<br>"
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
                    text=f"{row['avg_prod']:.2f} {unit_short}/day (Baseline: {row['baseline']:.2f} {unit_short}/day)",
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
        fig_top5, fig_bottom5 = create_top_bottom_gangs_charts(
            charts_scope, metric=(topbot_metric or "prod"), baseline_map=baseline_map
        )
        fig_project = create_project_lines_chart(
            projects_scope,
            selected_projects=project_list or None,
            bench=benchmark,
            avg_line=avg_prod,   # show Average (Avg Output / Gang / Day) line
        )

        # If in stringing mode, adapt figure labels/annotations to KM-based units
        if is_stringing:
            # Top/Bottom charts: replace MT -> KM in hover and y-axis title
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
                        fig.update_yaxes(title_text=ytitle.replace("MT/day", "KM/day").replace("MT", "KM"))
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
                            tr.hovertemplate = tr.hovertemplate.replace(" MT/day", " KM/day").replace(" MT", " KM")
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
                            a.text = a.text.replace(" MT/day", " KM/day")
                    fig_project.update_layout(annotations=annots)
            except Exception:
                pass

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

        df_day = _select_daily(mode_value)
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
        df_day = _select_daily(eff_mode)
        scoped = apply_filters(df_day, project_list, [], gang_list).copy()

        export_df, display_df = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=responsibilities_provider,
            search_text=search_text,
        )

        if display_df.empty:
            return []

        return display_df.to_dict("records")

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

        df_day = _select_daily(mode_value)
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

        df_day = _select_daily(mode_value)
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
            df = _select_daily(mode_value)
            rows = int(len(df.index)) if hasattr(df, "index") else 0
            mode = (mode_value or "erection").strip().lower()
            return f"mode={mode}; rows={rows}"
        except Exception as exc:  # pragma: no cover
            mode = (mode_value or "erection").strip().lower()
            return f"mode={mode}; error={type(exc).__name__}"



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

        total_label = "Delivered (KM)" if is_stringing else "Total Erection"
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
        return total_label, lost_label, idle_cols, idle_cols, daily_cols, daily_cols
