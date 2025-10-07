"""Plotly chart builders."""
from __future__ import annotations

import logging
from typing import Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import numpy as np

from .config import AppConfig

LOGGER = logging.getLogger(__name__)
DEFAULT_BENCHMARK = AppConfig().default_benchmark


def _empty_figure(height: int = 300) -> go.Figure:
    """Return an empty figure placeholder."""

    return go.Figure().update_layout(height=height, margin=dict(l=40, r=20, t=30, b=50))


# def create_monthly_line_chart(data: pd.DataFrame, bench: float = DEFAULT_BENCHMARK) -> go.Figure:
#     """Build the average productivity line chart."""

#     if data.empty:
#         LOGGER.debug("Monthly line chart requested with empty dataset")
#         return _empty_figure()

#     monthly = data.groupby("month")["daily_prod_mt"].mean().reset_index()
#     figure = go.Figure()
#     figure.add_trace(
#         go.Scatter(
#             x=monthly["month"],
#             y=monthly["daily_prod_mt"],
#             mode="lines+markers",
#             name="Avg Productivity",
#             line=dict(color="#0074D9"),
#         )
#     )
#     figure.add_hline(
#         y=bench,
#         line_dash="dot",
#         line_color="red",
#         annotation_text=f"Benchmark {bench} MT/day",
#         annotation_position="top left",
#     )
#     figure.update_layout(
#         height=300,
#         margin=dict(l=40, r=20, t=30, b=50),
#         xaxis_title="Month",
#         yaxis_title="Avg Productivity (MT)",
#         plot_bgcolor="#fafafa",
#         paper_bgcolor="#ffffff",
#     )
#     LOGGER.debug("Monthly line chart built for %d months", len(monthly))
#     return figure


def create_top_bottom_gangs_charts(
    data: pd.DataFrame,
    metric: str = "prod",
    baseline_map: dict[str, float] | None = None,
) -> Tuple[go.Figure, go.Figure]:
    """Build top and bottom five gang bar charts."""

    if data.empty:
        LOGGER.debug("Top/Bottom charts requested with empty dataset")
        empty = _empty_figure(height=280)
        return empty, empty

    # pick aggregator and labels
    if metric == "erection":
        # total MT in scope (sum of daily MT)
        per_gang = (data.groupby("gang_name", as_index=False)["daily_prod_mt"].sum())
        ytitle = "MT"
        textfmt = lambda s: s.round(1)
        hoverline = "Total: %{y:.2f} MT"
    else:
        # average daily productivity (default)
        per_gang = (data.groupby("gang_name", as_index=False)["daily_prod_mt"].mean())
        ytitle = "MT/day"
        textfmt = lambda s: s.round(2)
        hoverline = "Avg: %{y:.2f} MT/day"

    per_gang = per_gang.sort_values("daily_prod_mt", ascending=False)
    top5 = per_gang.head(5)
    bottom5 = per_gang.tail(5)


    # ---------- NEW: add meta = project at last activity + last date ----------
    if {"gang_name", "project_name", "date"}.issubset(data.columns):
        # find the last activity row per gang
        idx_last = data.sort_values("date").groupby("gang_name")["date"].idxmax()
        meta = (
            data.loc[idx_last, ["gang_name", "project_name", "date"]]
                .rename(columns={"project_name": "last_project", "date": "last_date"})
        )
        top5 = top5.merge(meta, on="gang_name", how="left")
        bottom5 = bottom5.merge(meta, on="gang_name", how="left")

        # pretty dates for hover
        top5["last_date_str"] = pd.to_datetime(top5["last_date"], errors="coerce").dt.strftime("%d-%b-%Y")
        bottom5["last_date_str"] = pd.to_datetime(bottom5["last_date"], errors="coerce").dt.strftime("%d-%b-%Y")
    else:
        top5["last_project"] = ""
        top5["last_date_str"] = ""
        bottom5["last_project"] = ""
        bottom5["last_date_str"] = ""
    # -------------------------------------------------------------------------
        # Add current & baseline values for hover
    if baseline_map is None:
        baseline_map = {}
    top5["baseline_metric"] = top5["gang_name"].map(baseline_map).fillna(0.0)
    bottom5["baseline_metric"] = bottom5["gang_name"].map(baseline_map).fillna(0.0)

    # Current metric is what we plotted (y value)
    top5["current_metric"] = top5["daily_prod_mt"]
    bottom5["current_metric"] = bottom5["daily_prod_mt"]

    EXEC_GREEN = "#2E7D32"
    EXEC_RED   = "#C62828"
    GRID_GRAY  = "#e9ecef"

    top_chart = go.Figure(
        data=[
            go.Bar(
                x=top5["gang_name"],
                y=top5["daily_prod_mt"],
                marker_color=EXEC_GREEN,
                marker_line_width=0,
                text=textfmt(top5["daily_prod_mt"]),
                textposition="outside",
                textfont=dict(size=12, color="#111"),
                name="Top 5",
                customdata=np.stack(
                    [top5["last_project"].fillna("—"),
                    top5["last_date_str"].fillna("—"),
                    top5["current_metric"],
                    top5["baseline_metric"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{x}<br>" + hoverline + "<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked at: %{customdata[1]}<br>"
                    "Current " + ytitle + ": %{customdata[2]:.2f}<br>"
                    "Baseline " + ytitle + ": %{customdata[3]:.2f}<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.95)",
                    font=dict(color="#111827", size=13),
                    bordercolor="rgba(17,24,39,0.15)",
                    align="left",
                    namelength=0,
                ),
            )
        ]
    )

    ymax_top = float(top5["daily_prod_mt"].max()) if not top5.empty else 1.0
    top_chart.update_yaxes(range=[0, ymax_top * 1.15], gridcolor=GRID_GRAY, zeroline=False, showspikes=False)
    top_chart.update_xaxes(tickangle=-10, showspikes=False)
    top_chart.update_layout(
        yaxis_title=ytitle,
        height=200,
        margin=dict(l=36, r=16, t=10, b=10),
        bargap=0.25,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        dragmode=False,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        hovermode="closest",
    )
    

    bottom_chart = go.Figure(
        data=[
            go.Bar(
                x=bottom5["gang_name"],
                y=bottom5["daily_prod_mt"],
                marker_color=EXEC_RED,
                marker_line_width=0,
                text=textfmt(bottom5["daily_prod_mt"]),
                textposition="outside",
                textfont=dict(size=12, color="#111"),
                name="Bottom 5",
                customdata=np.stack(
                    [bottom5["last_project"].fillna("—"),
                    bottom5["last_date_str"].fillna("—"),
                    bottom5["current_metric"],
                    bottom5["baseline_metric"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{x}<br>" + hoverline + "<br>"
                    "Project: %{customdata[0]}<br>"
                    "Last worked at: %{customdata[1]}<br>"
                    "Current " + ytitle + ": %{customdata[2]:.2f}<br>"
                    "Baseline " + ytitle + ": %{customdata[3]:.2f}<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.95)",
                    font=dict(color="#111827", size=13),
                    bordercolor="rgba(17,24,39,0.15)",
                    align="left",
                    namelength=0,
                ),
            )
        ]
    )
    
    ymax_bot = float(bottom5["daily_prod_mt"].max()) if not bottom5.empty else 1.0
    bottom_chart.update_yaxes(range=[0, ymax_bot * 1.15], gridcolor=GRID_GRAY, zeroline=False, showspikes=False)
    bottom_chart.update_xaxes(tickangle=-10, showspikes=False)
    bottom_chart.update_layout(
        yaxis_title=ytitle,
        height=200,
        margin=dict(l=36, r=16, t=10, b=10),
        bargap=0.25,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        dragmode=False,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        hovermode="closest",
    )
    
    
    LOGGER.debug("Top/Bottom charts built with %d gangs", len(per_gang))

    return top_chart, bottom_chart


def create_project_lines_chart(
    data: pd.DataFrame,
    selected_projects: Sequence[str] | None = None,
    bench: float = DEFAULT_BENCHMARK,
    avg_line: float | None = None,    # NEW: draw "Average (Avg Output / Gang / Day)" line when provided
) -> go.Figure:
    """Per-project monthly productivity lines.

    - Selected projects are emphasized (thicker, full opacity).
    - Non-selected projects are faded (lower opacity).
    - Legend sits BELOW the chart.
    - Shows Benchmark (red dashed) and optional Average (cyan dotted) lines.
    """

    if data.empty:
        LOGGER.debug("Project lines chart requested with empty dataset")
        return _empty_figure()

    monthly = (
        data.groupby(["month", "project_name"])["daily_prod_mt"].mean().reset_index()
    )

    highlight_projects = set(selected_projects or [])
    fig = go.Figure()

    for project in monthly["project_name"].unique():
        d = monthly[monthly["project_name"] == project]
        is_highlight = project in highlight_projects

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(d["month"]),
                y=d["daily_prod_mt"],
                mode="lines+markers" if is_highlight else "lines",
                name=project,
                opacity=1.0 if is_highlight or not highlight_projects else 0.28,
                line=dict(width=3.2 if is_highlight else 1.8, shape="spline"),
                marker=dict(size=6 if is_highlight else 0),
                hovertemplate="<b>%{x|%b %Y}</b><br>%{meta}: %{y:.2f} MT<extra></extra>",
                meta=project,
            )
        )

    # Benchmark horizontal line (kept)
    if bench is not None:
        fig.add_hline(
            y=bench,
            line_dash="dash",
            line_color="#EF4444",
            line_width=2,
            annotation_text=f"Benchmark {bench:.1f} MT/day",
            annotation_position="top left",
            annotation_font_color="#374151",
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )

    # NEW: Average horizontal line (from KPI)
    if avg_line is not None:
        fig.add_hline(
            y=avg_line,
            line_dash="dot",
            line_color="#0EA5E9",
            line_width=2,
            annotation_text=f"Average {avg_line:.1f} MT/day",
            annotation_position="top left",
            annotation_yshift=-18,
            annotation_font_color="#0369A1",
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=10, b=90),  # extra bottom space for legend
        xaxis=dict(
            title="Month",
            tickformat="%b %Y",
            showgrid=True,
            gridcolor="#eef0f3",
            zeroline=False,
        ),
        yaxis=dict(
            title="Avg Productivity (MT)",
            rangemode="tozero",
            showgrid=True,
            gridcolor="#eef0f3",
            zeroline=False,
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,          # BELOW the chart
            xanchor="left",
            x=0.0,
            font=dict(size=12),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        transition=dict(duration=200),
    )

    LOGGER.debug("Project lines chart built for %d projects", monthly['project_name'].nunique())
    return fig



# --- NEW: responsibilities chart builder ---


def build_empty_responsibilities_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13)
    )
    return fig


def build_responsibilities_chart(
    df: pd.DataFrame,
    entity_label: str,                  # "Gang" | "Section Incharge" | "Supervisor"
    metric: str = "tower_weight",       # "revenue" | "tower_weight"
    title: str | None = None,
    top_n: int = 20
) -> go.Figure:
    if df.empty:
        return build_empty_responsibilities_figure("No Micro Plan data for the selected project.")

    metric = metric if metric in ("revenue", "tower_weight") else "tower_weight"
    axis_title = "Revenue" if metric == "revenue" else "Tower Weight (MT)"

    g = (
        df.groupby("entity_name", as_index=False)[metric]
          .sum()
          .sort_values(metric, ascending=False)
          .head(top_n)
    )

    # Delivered:
    # If your data already has a delivered column per entity, use it.
    delivered_col = None
    for cand in ("delivered", "actual_delivered", "achieved", "delivered_mt", "delivered_value"):
        if cand in df.columns:
            delivered_col = cand
            break

    if delivered_col:
        d = (
            df.groupby("entity_name", as_index=False)[delivered_col]
              .sum()
        )
        g = g.merge(d, on="entity_name", how="left")
        g["delivered_plot"] = g[delivered_col].fillna(0)
    else:
        # Fallback to your earlier convention (10% proxy)
        g["delivered_plot"] = (0.10 * g[metric]).round(2)

    fig = go.Figure()

    # Responsibility Target (blue bars)
    fig.add_bar(
        x=g["entity_name"],
        y=g[metric],
        name="Responsibility Target",
        width=0.55,
        marker_color="#3B82F6",
        hovertemplate="<b>%{x}</b><br>Target: %{y:,.0f}<extra></extra>",
    )

    # Actual Delivered (red bars)
    fig.add_bar(
        x=g["entity_name"],
        y=g["delivered_plot"],
        name="Actual Delivered",
        width=0.55,
        marker_color="#EF4444",
        hovertemplate="<b>%{x}</b><br>Delivered: %{y:,.0f}<extra></extra>",
    )


    fig.update_layout(
        barmode="group",
        bargap=0.30,        # space between groups
        bargroupgap=0.22,   # space between the two bars in a group
        height=360,
        margin=dict(l=40, r=20, t=10, b=60),
        xaxis_title=entity_label,
        yaxis_title=axis_title,
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0.0,
            font=dict(size=11)
        ),
    )

    fig.update_xaxes(tickangle=-20, automargin=True, showspikes=False)
    fig.update_yaxes(gridcolor="#e9ecef", zeroline=False, rangemode="tozero", showspikes=False)

    return fig

