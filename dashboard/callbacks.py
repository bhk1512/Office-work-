"""Dash callbacks for the productivity dashboard."""

from __future__ import annotations

import logging
import dash_bootstrap_components as dbc 
import pandas as pd
from io import BytesIO
from typing import Any, Callable, Sequence

import dash
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.dcc import send_bytes
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ALL

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
from .data_loader import load_microplan_responsibilities


LOGGER = logging.getLogger(__name__)

BENCHMARK_MT_PER_DAY = 9.0



_slug = lambda s: re.sub(r"[^a-z0-9_-]+", "-", str(s).lower()).strip("-")

def _render_avp_row(gang, delivered, lost, total, pct, avg_prod=0.0, baseline=0.0, last_project="—", last_date="—"):
    badge_cls = "good" if pct >= 80 else ("mid" if pct >= 65 else "low")
    delivered_pct = 0 if total == 0 else max(0, min(100, (delivered/total)*100))
    lost_pct = 0 if total == 0 else max(0, min(100, (lost/total)*100))

    row_id = f"avp-row-{_slug(gang)}"  # unique string id for tooltip target

    return html.Div(
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
            # 1) CLICK OVERLAY (pattern ID) — this is what Dash listens to for n_clicks
            html.Div(
                id={"type": "avp-row", "index": gang},  # pattern-matching ID for clicks
                n_clicks=0,
                className="avp-hit",
            ),

            # 2) TOOLTIP ANCHOR (string ID) — tiny element used ONLY to anchor the tooltip
            html.Span(id=row_id, className="avp-tip-anchor"),

            # 3) TOOLTIP (bootstrap) — target is the string id above
            dbc.Tooltip(
                [
                    html.Div(html.B(gang)),
                    html.Div(f"Project: {last_project}"),
                    html.Div(f"Last worked at: {last_date}"),
                    html.Div(f"Current MT/day: {avg_prod:.2f}"),
                    html.Div(f"Baseline MT/day: {baseline:.2f}"),
                ],
                target=row_id,                 # <-- IMPORTANT: target must be a STRING id
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

# project_list = _ensure_list(projects)
# month_list = _ensure_list(months)


def register_callbacks(
    app: Dash,
    data_provider: Callable[[], pd.DataFrame],
    config: AppConfig,
    project_info_provider: Callable[[], pd.DataFrame] | None = None,
) -> None:
    """Register all Dash callbacks on *app*."""

    LOGGER.debug("Registering callbacks")

    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, prevMeta) {
        const C  = window.dash_clientside;
        const NO = C.no_update;
        const ctx = C.callback_context;
        const trg = (ctx && ctx.triggered && ctx.triggered[0] && ctx.triggered[0].prop_id) || "";

        let cd = null, src = null;
        if (trg.startsWith("g-actual-vs-bench.")) { cd = lossClick;  src = "g-actual-vs-bench"; }
        else if (trg.startsWith("g-top5."))        { cd = topClick;   src = "g-top5"; }
        else if (trg.startsWith("g-bottom5."))     { cd = bottomClick; src = "g-bottom5"; }
        else return [NO, NO];

        if (!cd || !cd.points || !cd.points.length) return [NO, NO];

        const pt  = cd.points[0];
        const now = Date.now();

        // ✅ Prefer axis category for gang (robust to customdata being [project, date])
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;   // horiz bars -> gang on y
        else if (typeof pt.x === "string") gang = pt.x;   // vertical bars -> gang on x
        else if (pt.customdata) {
            if (typeof pt.customdata === "string") gang = pt.customdata;
            else if (Array.isArray(pt.customdata)) gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object") gang = pt.customdata.gang || pt.customdata.name || null;
        }

        if (!gang) return [NO, NO];

        const newMeta = { source: src, gang: gang, ts: now };
        return [newMeta, NO];
        }
        """,
        [Output("store-click-meta", "data"), Output("store-dblclick", "data")],
        [Input("g-actual-vs-bench", "clickData"),
        Input("g-top5", "clickData"),
        Input("g-bottom5", "clickData")],
        [State("store-click-meta", "data")]
    )





    # Auto-scroll to Traceability when any chart is clicked
    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, rowClicks, selectedGang) {
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;
        const trg = ctx.triggered[0].prop_id || "";

        let shouldScroll = false;

        // A) Any graph clickData fired with a valid point
        if (trg.endsWith(".clickData")) {
            let cd = null;
            if (trg.startsWith("g-actual-vs-bench.")) cd = lossClick;
            else if (trg.startsWith("g-top5."))        cd = topClick;
            else if (trg.startsWith("g-bottom5."))     cd = bottomClick;
            if (cd && cd.points && cd.points.length) shouldScroll = true;
        }

        // B) Any AVP row click (pattern ID), robust to key order
        try {
            const idPart = trg.split(".")[0];
            const pid = JSON.parse(idPart);
            if (pid && pid.type === "avp-row") {
            // we don't need to inspect which one; any click should scroll
            shouldScroll = true;
            }
        } catch(e) {}

        // C) Fallback: when store-selected-gang updates (covers programmatic updates)
        if (trg === "store-selected-gang.data" && selectedGang) {
            shouldScroll = true;
        }

        if (!shouldScroll) return NO;

        const anchor = document.getElementById("trace-anchor");
        if (anchor) {
            const OFFSET = 80; // adjust if you have a taller header
            const y = anchor.getBoundingClientRect().top + window.pageYOffset - OFFSET;
            window.scrollTo({ top: y, behavior: "smooth" });
        }
        // Return a changing token so Dash marks this callback as updated
        return String(Date.now());
        }
        """,
        Output("scroll-wire", "children"),
        [
            Input("g-actual-vs-bench", "clickData"),
            Input("g-top5", "clickData"),
            Input("g-bottom5", "clickData"),
            Input({"type":"avp-row","index": ALL}, "n_clicks"),   # NEW
            Input("store-selected-gang", "data"),                 # NEW
        ]
    )


    # ONE unified clientside callback for BOTH graph clicks and AVP row clicks
    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, rowClicks, rowIds, prev) {
        const C = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;
        const trg = ctx.triggered[0].prop_id || "";

        // ---- robust: parse the pattern id JSON instead of relying on key order
        try {
            const idPart = trg.split(".")[0];
            const pid = JSON.parse(idPart);
            if (pid && pid.type === "avp-row") {
            if (!rowClicks || !rowClicks.length) return NO;
            let last = -1;
            for (let i = 0; i < rowClicks.length; i++) { if (rowClicks[i]) last = i; }
            if (last < 0) return NO;
            const gang = rowIds && rowIds[last] && rowIds[last].index;
            return gang || NO;
            }
        } catch (e) { /* ignore and fall through to chart clicks */ }

        // ---- charts (unchanged)
        let cd = null;
        if (trg.startsWith("g-actual-vs-bench.")) cd = lossClick;
        else if (trg.startsWith("g-top5."))        cd = topClick;
        else if (trg.startsWith("g-bottom5."))     cd = bottomClick;
        else return NO;

        if (!cd || !cd.points || !cd.points.length) return NO;
        const pt = cd.points[0];
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;
        else if (typeof pt.x === "string") gang = pt.x;
        else if (pt.customdata) {
            if (typeof pt.customdata === "string")       gang = pt.customdata;
            else if (Array.isArray(pt.customdata))       gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object")  gang = pt.customdata.gang || pt.customdata.name || null;
        }
        return gang || NO;
        }
        """,
        Output("store-selected-gang", "data"),
        [
        Input("g-actual-vs-bench", "clickData"),
        Input("g-top5", "clickData"),
        Input("g-bottom5", "clickData"),
        Input({"type":"avp-row","index": dash.dependencies.ALL}, "n_clicks"),
        ],
        [
        State({"type":"avp-row","index": dash.dependencies.ALL}, "id"),
        State("store-selected-gang", "data"),
        ],
    )
    

    

    @app.callback(
        Output("f-project", "value"),
        Output("f-month", "value"),
        Output("f-gang", "value"),
        Output("f-quick-range", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("btn-clear-quick-range", "n_clicks"),
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
        if trigger_id == "btn-clear-quick-range":
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
        Output("project-details", "children"),
        Input("f-project", "value"),
        prevent_initial_call=False,
    )
    def show_project_details(selected_projects):
        
        
        if not selected_projects:
            return "Select a single project to view its details."       
        
        if not project_info_provider:
            return "No 'Project Details' source configured."  
        df_info = project_info_provider()
        if df_info is None or df_info.empty:
            return "No 'Project Details' sheet found in the source workbook."

        pname = str(selected_projects).strip() 
        # Match by Project Name (exact match first, then tolerant fallback)
        row = df_info[df_info["Project Name"] == pname]
        if row.empty and "Project Name" in df_info.columns:
            # fallback to the normalized helper built during load
            norm = lambda s: " ".join(str(s).strip().lower().split())
            row = df_info[df_info["Project Name"].apply(norm) == norm(pname)]
        if row.empty:
            return f"No project details found for “{pname}”."


        r = row.iloc[0]

        def fmt_date(x):
            return "" if pd.isna(x) else pd.to_datetime(x).strftime("%d-%m-%Y")

        return html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div([html.Span("Project Code: ", className="text-muted"), html.B(r.get("project_code",""))]),
                    html.Div([html.Span("Client: ", className="text-muted"), html.B(r.get("client_name",""))]),
                ]), md=6),
                dbc.Col(html.Div([
                    html.Div([html.Span("NOA Start: ", className="text-muted"), html.B(fmt_date(r.get("noa_start")))]),
                    html.Div([html.Span("LOA End: ", className="text-muted"), html.B(fmt_date(r.get("loa_end")))]),
                ]), md=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div([html.Span("Project Manager: ", className="text-muted"), html.B(r.get("project_mgr",""))]),
                    html.Div([html.Span("Regional Manager: ", className="text-muted"), html.B(r.get("regional_mgr",""))]),
                    html.Div([html.Span("Planning Engineer: ", className="text-muted"), html.B(r.get("planning_eng",""))]),
                ]), md=6),
                dbc.Col(html.Div([
                    html.Div([html.Span("PCH: ", className="text-muted"), html.B(r.get("pch",""))]),
                    html.Div([html.Span("Section Incharge: ", className="text-muted"), html.B(r.get("section_inch",""))]),
                    html.Div([html.Span("Supervisor: ", className="text-muted"), html.B(r.get("supervisor",""))]),
                ]), md=6),
            ]),
        ])

    # --- NEW: responsibilities chart callback ---

    @app.callback(
        Output("g-responsibilities", "figure"),
        Input("f-project", "value"),
        Input("f-resp-entity", "value"),
        Input("f-resp-metric", "value"),
    )
    def update_responsibilities_chart(project_value: str | None, entity_value: str | None, metric_value: str | None):
        # If no project selected, show friendly message (do NOT PreventUpdate, or it stays blank)
        if not project_value:
            return build_empty_responsibilities_figure("Select a single project to view its details.")

        entity_value = (entity_value or "Supervisor").strip()
        metric_value = (metric_value or "tower_weight").strip()

        cfg = AppConfig()
        df = load_microplan_responsibilities(cfg.data_path)

        # If sheet missing/empty → explain
        if df.empty:
            return build_empty_responsibilities_figure("No Micro Plan data found in the compiled workbook.")

        # Normalize project value (your dropdown uses 'Project Name' text)
        sel_norm = str(project_value).replace("\u00a0", " ").strip()
        sel_lc = sel_norm.lower()

        # STRICT match first on project_name, then on key, then fallback to contains
        dfp = df[(df["project_name_lc"] == sel_lc) | (df["project_key_lc"] == sel_lc)]
        if dfp.empty:
            dfp = df[df["project_name_lc"].str.contains(sel_lc, na=False)]

        # Only keep the selected entity type
        dfp = dfp[dfp["entity_type"].astype(str).str.strip().str.lower() == entity_value.lower()]

        if dfp.empty:
            return build_empty_responsibilities_figure("No responsibilities found for the selected filters.")

        # Build figure
        title = None  # keep clean; or set to f"{entity_value} — {metric_value.replace('_', ' ').title()}"
        return build_responsibilities_chart(dfp, entity_label=entity_value, metric=metric_value, title=title, top_n=20)


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
        Output("g-project-lines", "figure"),
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

        selected_months = months_ts or []
        if selected_months:
            month_start = max(selected_months)
        else:
            candidate = scoped if not scoped.empty else df_day
            last_date = pd.to_datetime(candidate["date"].max())
            if pd.isna(last_date):
                month_start = pd.Timestamp.today().to_period("M").to_timestamp()
            else:
                month_start = last_date.to_period("M").to_timestamp()
        month_end = month_start + pd.offsets.MonthBegin(1)

        scope_mask = pd.Series(True, index=df_day.index)
        if project_list:
            scope_mask &= df_day["project_name"].isin(project_list)
        if gang_list:
            scope_mask &= df_day["gang_name"].isin(gang_list)
        scoped_all = df_day.loc[scope_mask].copy()

        loss_scope = scoped_all[
            (scoped_all["date"] >= month_start) & (scoped_all["date"] < month_end)
        ].copy()
        history_scope = scoped_all[scoped_all["date"] < month_start]

        baseline_map = {}
        if not history_scope.empty:
            baseline_map = (
                history_scope.groupby("gang_name")["daily_prod_mt"].mean().fillna(0).to_dict()
            )

        loss_rows: list[dict[str, float]] = []
        for gang_name, gang_df in loss_scope.groupby("gang_name"):
            override_baseline = baseline_map.get(gang_name, 0.0)
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
        loss_df = (
            pd.DataFrame(loss_rows).sort_values("potential", ascending=True)
            if loss_rows
            else pd.DataFrame(
                columns=["gang_name", "delivered", "lost", "potential", "avg_prod", "baseline"]
            )
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
                total = (r["delivered"] + r["lost"]) if pd.notna(r["delivered"]) and pd.notna(r["lost"]) else r["potential"]
                total = float(total) if pd.notna(total) else 0.0
                pct = 0.0 if total == 0 else (100.0 * float(r["delivered"]) / total)
                avp_children.append(
                    _render_avp_row(
                        r["gang_name"], float(r["delivered"]), float(r["lost"]),
                        total, pct,
                        avg_prod=float(r.get("avg_prod", 0.0)),
                        baseline=float(r.get("baseline", 0.0)),
                        last_project=str(r.get("last_project", "—")),
                        last_date=str(r.get("last_date_str", "—")),
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
            fig_loss.add_bar(
                x=loss_df["delivered"],
                y=loss_df["gang_name"],
                orientation="h",
                marker_color="green",
                text=loss_df["delivered"].round(1),
                textposition="inside",
                name="Delivered",
                width=0.95,
                customdata=np.stack([loss_df["last_project"], loss_df["last_date_str"]], axis=-1),
                hovertemplate=(
                    "%{y}<br>"
                    "Delivered: %{x:.1f} MT<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked: %{customdata[1]}<extra></extra>"
                ),
            )

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
                customdata=np.stack([loss_df["last_project"], loss_df["last_date_str"]], axis=-1),
                hovertemplate=(
                    "%{y}<br>"
                    "Loss: %{x:.1f} MT<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked: %{customdata[1]}<extra></extra>"
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
        fig_loss.update_yaxes(showspikes=False, fixedrange=True)
        
        # fig_monthly = create_monthly_line_chart(scoped, bench=benchmark)
        fig_top5, fig_bottom5 = create_top_bottom_gangs_charts(scoped_top_bottom, metric=(topbot_metric or "prod"), baseline_map=baseline_map)
        fig_project = create_project_lines_chart(
            df_day,
            selected_projects=project_list or None,
            bench=benchmark,
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



    
    @app.callback(
        Output("tbl-idle-intervals", "data"),
        Output("tbl-daily-prod", "data"),
        Output("modal-tbl-idle-intervals", "data"),
        Output("modal-tbl-daily-prod", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("trace-gang", "value"),
        Input("store-selected-gang", "data"),
    )
    def update_trace_tables(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        trace_gang_value: str | None,
        selected_gang: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        gang_list = _ensure_list(gangs)

        df_day = data_provider()
        months_ts = resolve_months(month_list, quick_range)

        base_scope = apply_filters(df_day, project_list, months_ts, []).copy()
        scoped = apply_filters(df_day, project_list, months_ts, gang_list).copy()

        def pick_gang_scope(target_gang: str | None) -> pd.DataFrame:
            if not target_gang:
                return pd.DataFrame()
            subset = base_scope[base_scope["gang_name"] == target_gang]
            if not subset.empty:
                return subset
            fallback = df_day[df_day["gang_name"] == target_gang].copy()
            if project_list:
                fallback = fallback[fallback["project_name"].isin(project_list)]
            if months_ts:
                fallback = fallback[fallback["month"].isin(months_ts)]
            return fallback

        gang_focus = trace_gang_value or selected_gang

        if gang_focus:
            idle_source = pick_gang_scope(gang_focus)
        elif gang_list:
            idle_source = base_scope[base_scope["gang_name"].isin(gang_list)]
        else:
            idle_source = base_scope
        if idle_source.empty:
            idle_source = scoped if not scoped.empty else base_scope

        idle_df = compute_idle_intervals_per_gang(
            idle_source, loss_max_gap_days=config.loss_max_gap_days
        )
        if not idle_df.empty:
            idle_df = idle_df.assign(
                interval_start=idle_df["interval_start"].dt.strftime("%d-%m-%Y"),
                interval_end=idle_df["interval_end"].dt.strftime("%d-%m-%Y"),
            )
        idle_data = idle_df.to_dict("records")

        if gang_focus:
            daily_source = pick_gang_scope(gang_focus)
        elif gang_list:
            daily_source = base_scope[base_scope["gang_name"].isin(gang_list)]
        else:
            daily_source = base_scope
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
                        lambda value: ""
                        if pd.isna(value)
                        else f"{value:.2f}".rstrip("0").rstrip(".")
                    )
                ),
            )
        daily_data = daily_source.to_dict("records")

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
