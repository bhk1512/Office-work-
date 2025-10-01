"""Dash callbacks for the productivity dashboard."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Callable, Sequence

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.dcc import send_bytes

from .charts import (
    create_monthly_line_chart,
    create_project_lines_chart,
    create_top_bottom_gangs_charts,
)
from .config import AppConfig
from .filters import apply_filters, resolve_months
from .metrics import calc_idle_and_loss, compute_idle_intervals_per_gang
from .workbook import make_trace_workbook_bytes

LOGGER = logging.getLogger(__name__)

BENCHMARK_MT_PER_DAY = 9.0



def register_callbacks(
    app: Dash,
    data_provider: Callable[[], pd.DataFrame],
    config: AppConfig,
) -> None:
    """Register all Dash callbacks on *app*."""

    LOGGER.debug("Registering callbacks")

    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, prevMeta) {
            function getGang(cd, src){
                if(!cd || !cd.points || !cd.points.length){ return null; }
                return src === 'g-actual-vs-bench' ? cd.points[0].y : cd.points[0].x;
            }
            var src = null, cd = null;
            if(lossClick && lossClick.points){ src = 'g-actual-vs-bench'; cd = lossClick; }
            else if(topClick && topClick.points){ src = 'g-top5'; cd = topClick; }
            else if(bottomClick && bottomClick.points){ src = 'g-bottom5'; cd = bottomClick; }
            if(!src){
                return [prevMeta, null];
            }
            var gang = getGang(cd, src);
            var now  = Date.now();
            var last = prevMeta || {};
            var isSameSrc  = last.source === src;
            var isSameGang = last.gang === gang;
            var isDbl = isSameSrc && isSameGang && (now - (last.ts || 0) <= 700);

            var newMeta = {source: src, gang: gang, ts: now};
            var dblData = isDbl ? {source: src, gang: gang, ts: now} : null;

            return [newMeta, dblData];
        }
        """,
        [Output("store-click-meta", "data"), Output("store-dblclick", "data")],
        [
            Input("g-actual-vs-bench", "clickData"),
            Input("g-top5", "clickData"),
            Input("g-bottom5", "clickData"),
        ],
        [State("store-click-meta", "data")],
    )

    

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
        months_ts = resolve_months(months, quick_range)

        filtered = df_day.copy()
        if months_ts:
            filtered = filtered[filtered["month"].isin(months_ts)]
        if gangs:
            filtered = filtered[filtered["gang_name"].isin(gangs)]

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
        if projects:
            filtered = filtered[filtered["project_name"].isin(projects)]
        months_ts = resolve_months(months, quick_range)
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
        if projects:
            filtered = filtered[filtered["project_name"].isin(projects)]
        if gangs:
            filtered = filtered[filtered["gang_name"].isin(gangs)]

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
        Output("kpi-avg", "children"),
        Output("kpi-delta", "children"),
        Output("kpi-active", "children"),
        Output("kpi-total", "children"),
        Output("kpi-loss", "children"),
        Output("g-actual-vs-bench", "figure"),
        Output("g-monthly", "figure"),
        Output("g-top5", "figure"),
        Output("g-bottom5", "figure"),
        Output("g-project-lines", "figure"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
    )
    def update_dashboard(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
    ) -> tuple:
        df_day = data_provider()
        months_ts = resolve_months(months, quick_range)

        scoped = apply_filters(
            df_day,
            projects or [],
            months_ts,
            gangs or [],
        )
        scoped_top_bottom = apply_filters(
            df_day,
            projects or [],
            months_ts,
            [],
        )

        benchmark = BENCHMARK_MT_PER_DAY
        avg_prod = scoped["daily_prod_mt"].mean() if len(scoped) else 0.0
        delta_pct = (avg_prod - benchmark) / benchmark * 100 if benchmark else None
        kpi_avg = f"{avg_prod:.2f} MT"
        kpi_delta = "(n/a)" if delta_pct is None else f"{delta_pct:+.1f}%"
        
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
        if projects:
            scope_mask &= df_day["project_name"].isin(projects)
        if gangs:
            scope_mask &= df_day["gang_name"].isin(gangs)
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
        kpi_loss = f"{lost_pct:.1f}%"

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
        )

        fig_monthly = create_monthly_line_chart(scoped, bench=benchmark)
        fig_top5, fig_bottom5 = create_top_bottom_gangs_charts(scoped_top_bottom)
        fig_project = create_project_lines_chart(
            df_day,
            selected_projects=projects,
            bench=benchmark,
        )

        return (
            kpi_avg,
            kpi_delta,
            kpi_active,
            kpi_total,
            kpi_loss,
            fig_loss,
            fig_monthly,
            fig_top5,
            fig_bottom5,
            fig_project,
        )

    @app.callback(
        Output("loss-modal", "is_open"),
        Output("modal-title", "children"),
        Output("modal-body", "children"),
        Input("store-dblclick", "data"),
        Input("modal-close", "n_clicks"),
        State("loss-modal", "is_open"),
        State("f-project", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
    )
    def show_loss_on_double_click(
        dbl_click: dict[str, Any] | None,
        close_clicks: int | None,
        is_open: bool,
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
    ):
        context = dash.callback_context
        if not context.triggered:
            return is_open, dash.no_update, dash.no_update
        if context.triggered[0]["prop_id"].startswith("modal-close"):
            return False, dash.no_update, dash.no_update
        if not dbl_click or not dbl_click.get("gang"):
            return is_open, dash.no_update, dash.no_update

        df_day = data_provider()
        gang_clicked = dbl_click["gang"]
        months_ts = resolve_months(months, quick_range)

        scope_mask = pd.Series(True, index=df_day.index)
        if projects:
            scope_mask &= df_day["project_name"].isin(projects)
        scoped_all = df_day.loc[scope_mask].copy()

        if months_ts:
            month_start = max(months_ts)
        else:
            last_date = pd.to_datetime(df_day["date"].max())
            if pd.isna(last_date):
                month_start = pd.Timestamp.today().to_period("M").to_timestamp()
            else:
                month_start = last_date.to_period("M").to_timestamp()
        month_end = month_start + pd.offsets.MonthBegin(1)

        selection = scoped_all[
            (scoped_all["date"] >= month_start)
            & (scoped_all["date"] < month_end)
            & (scoped_all["gang_name"] == gang_clicked)
        ]
        if selection.empty:
            return True, "Gang Efficiency & Loss", "No data in current selection."

        history = scoped_all[
            (scoped_all["date"] < month_start)
            & (scoped_all["gang_name"] == gang_clicked)
        ]
        baseline_override = history["daily_prod_mt"].mean() if not history.empty else 0.0
        if pd.isna(baseline_override):
            baseline_override = 0.0

        idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
            selection,
            loss_max_gap_days=config.loss_max_gap_days,
            baseline_mt_per_day=baseline_override,
        )
        efficiency = (delivered / potential * 100) if potential > 0 else 0.0
        lost_pct = (loss_mt / potential * 100) if potential > 0 else 0.0

        body = html.Div(
            [
                html.Div(f"Gang: {gang_clicked}", className="fw-bold mb-2"),
                html.Div(f"Erected (Delivered): {delivered:.2f} MT"),
                html.Div(
                    f"Loss: {loss_mt:.2f} MT ({lost_pct:.1f}%)",
                    className="text-danger",
                ),
                html.Div(f"Efficiency: {efficiency:.1f}%", className="mt-1"),
                html.Hr(),
                html.Div(f"Baseline MT/day: {baseline:.2f}"),
                html.Div(f"Idle days (cap {config.loss_max_gap_days}): {idle}"),
                html.Div(f"Potential MT: {potential:.2f}"),
            ]
        )
        return True, "Gang Efficiency & Loss", body

    @app.callback(
        Output("f-month", "value"),
        Input("f-quick-range", "value"),
        prevent_initial_call=True,
    )
    def clear_months_when_quick_range_selected(quick: str | None):
        if quick:
            return []
        return dash.no_update

    @app.callback(
        Output("f-quick-range", "value"),
        Input("f-month", "value"),
        prevent_initial_call=True,
    )
    def clear_quick_when_months_selected(month_values: Sequence[str] | None):
        if month_values:
            return None
        return dash.no_update

    @app.callback(
        Output("store-selected-gang", "data"),
        Input("g-actual-vs-bench", "clickData"),
        Input("g-top5", "clickData"),
        Input("g-bottom5", "clickData"),
        prevent_initial_call=True,
    )
    def set_selected_gang(loss_click, top_click, bottom_click):
        context = dash.callback_context
        if not context.triggered:
            return dash.no_update
        source = context.triggered[0]["prop_id"].split(".")[0]
        if source == "g-actual-vs-bench":
            return (loss_click or {}).get("points", [{}])[0].get("y")
        if source == "g-top5":
            return (top_click or {}).get("points", [{}])[0].get("x")
        if source == "g-bottom5":
            return (bottom_click or {}).get("points", [{}])[0].get("x")
        return dash.no_update

    @app.callback(
        Output("tbl-idle-intervals", "data"),
        Output("tbl-daily-prod", "data"),
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        df_day = data_provider()
        months_ts = resolve_months(months, quick_range)

        scoped = apply_filters(
            df_day,
            projects or [],
            months_ts,
            gangs or [],
        )
        base_scope = apply_filters(
            df_day,
            projects or [],
            months_ts,
            [],
        )

        if trace_gang_value:
            idle_source = base_scope[base_scope["gang_name"] == trace_gang_value]
        elif selected_gang:
            if selected_gang in scoped["gang_name"].unique():
                idle_source = scoped[scoped["gang_name"] == selected_gang]
            else:
                idle_source = base_scope[base_scope["gang_name"] == selected_gang]
        elif gangs:
            idle_source = scoped
        else:
            idle_source = scoped

        idle_df = compute_idle_intervals_per_gang(
            idle_source, loss_max_gap_days=config.loss_max_gap_days
        )
        if not idle_df.empty:
            idle_df = idle_df.assign(
                interval_start=idle_df["interval_start"].dt.strftime("%d-%m-%Y"),
                interval_end=idle_df["interval_end"].dt.strftime("%d-%m-%Y"),
            )
        idle_data = idle_df.to_dict("records")

        if trace_gang_value:
            daily = base_scope[base_scope["gang_name"] == trace_gang_value]
        elif selected_gang:
            if selected_gang in scoped["gang_name"].unique():
                daily = scoped[scoped["gang_name"] == selected_gang]
            else:
                daily = base_scope[base_scope["gang_name"] == selected_gang]
        elif gangs:
            daily = scoped[scoped["gang_name"].isin(gangs)]
        else:
            daily = scoped

        daily = daily.sort_values(["gang_name", "date"])[
            ["date", "gang_name", "project_name", "daily_prod_mt"]
        ]
        if not daily.empty:
            daily = daily.assign(
                date=daily["date"].dt.strftime("%d-%m-%Y"),
                daily_prod_mt=(
                    daily["daily_prod_mt"].round(2).map(
                        lambda value: ""
                        if pd.isna(value)
                        else f"{value:.2f}".rstrip("0").rstrip(".")
                    )
                ),
            )
        return idle_data, daily.to_dict("records")

    @app.callback(
        Output("download-trace-xlsx", "data"),
        Input("btn-export-trace", "n_clicks"),
        State("f-project", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
        State("trace-gang", "value"),
        State("store-selected-gang", "data"),
        prevent_initial_call=True,
    )
    def export_trace(
        _n_clicks: int | None,
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        trace_gang_value: str | None,
        selected_gang: str | None,
    ):
        df_day = data_provider()
        months_ts = resolve_months(months, quick_range)
        scoped = apply_filters(
            df_day,
            projects or [],
            months_ts,
            gangs or [],
        )
        gang_for_sheet = trace_gang_value or selected_gang
        benchmark_value = BENCHMARK_MT_PER_DAY

        def _writer(buffer: BytesIO) -> None:
            buffer.write(
                make_trace_workbook_bytes(
                    scoped,
                    months_ts,
                    projects or [],
                    gangs or [],
                    benchmark_value,
                    gang_for_sheet=gang_for_sheet,
                    config=config,
                )
            )

        return send_bytes(_writer, "Trace_Calcs.xlsx")

    @app.callback(
        Output("trace-gang", "options"),
        Output("trace-gang", "value"),
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
    ) -> tuple[list[dict[str, str]], str | None]:
        df_day = data_provider()
        months_ts = resolve_months(months, quick_range)
        base = apply_filters(
            df_day,
            projects or [],
            months_ts,
            [],
        )
        gangs = sorted(base["gang_name"].dropna().unique().tolist())
        options = [{"label": gang, "value": gang} for gang in gangs]

        if clicked_gang and clicked_gang in gangs:
            value = clicked_gang
        elif current_value and current_value in gangs:
            value = current_value
        else:
            value = None
        return options, value



















