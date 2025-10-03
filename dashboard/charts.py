"""Plotly chart builders."""
from __future__ import annotations

import logging
from typing import Iterable, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

from .config import AppConfig
from collections import defaultdict

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


def create_top_bottom_gangs_charts(data: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Build top and bottom five gang bar charts."""

    if data.empty:
        LOGGER.debug("Top/Bottom charts requested with empty dataset")
        empty = _empty_figure(height=280)
        return empty, empty

    per_gang = data.groupby("gang_name")["daily_prod_mt"].mean().reset_index()
    per_gang = per_gang.sort_values("daily_prod_mt", ascending=False)
    top5 = per_gang.head(5)
    bottom5 = per_gang.tail(5)

    top_chart = go.Figure(
        go.Bar(
            x=top5["gang_name"],
            y=top5["daily_prod_mt"],
            marker_color="green",
            text=top5["daily_prod_mt"].round(2),
            textposition="outside",
            name="Top 5",
        )
    )
    top_chart.update_layout(
        # title="Top 5 Gangs (Avg Productivity)",
        yaxis_title="MT/day",
        height=280,
        margin=dict(l=40, r=20, t=30, b=50),
        dragmode=False,
    )
    top_chart.update_xaxes(fixedrange=True)
    top_chart.update_yaxes(fixedrange=True)


    bottom_chart = go.Figure(
        go.Bar(
            x=bottom5["gang_name"],
            y=bottom5["daily_prod_mt"],
            marker_color="red",
            text=bottom5["daily_prod_mt"].round(2),
            textposition="outside",
            name="Bottom 5",
        )
    )
    bottom_chart.update_layout(
        # title="Bottom 5 Gangs (Avg Productivity)",
        yaxis_title="MT/day",
        height=280,
        margin=dict(l=40, r=20, t=30, b=50),
        dragmode=False,
    )
    bottom_chart.update_xaxes(fixedrange=True)
    bottom_chart.update_yaxes(fixedrange=True)
    LOGGER.debug("Top/Bottom charts built with %d gangs", len(per_gang))

    return top_chart, bottom_chart


def create_project_lines_chart(
    data: pd.DataFrame,
    selected_projects: Sequence[str] | None = None,
    bench: float = DEFAULT_BENCHMARK,
) -> go.Figure:
    """Build the per-project monthly productivity lines chart.

    If `selected_projects` is provided, those projects are HIGHLIGHTED
    (thicker, full opacity) and others are deemphasized (lower opacity).
    """

    if data.empty:
        LOGGER.debug("Project lines chart requested with empty dataset")
        return _empty_figure()

    monthly = (
        data.groupby(["month", "project_name"])["daily_prod_mt"].mean().reset_index()
    )

    # NEW: treat incoming list as "highlight list" (not a filter)
    highlight_projects = set(selected_projects or [])

    figure = go.Figure()
    all_projects = monthly["project_name"].unique()

    for project in all_projects:
        project_data = monthly[monthly["project_name"] == project]
        is_highlight = project in highlight_projects

        figure.add_trace(
            go.Scatter(
                x=project_data["month"],
                y=project_data["daily_prod_mt"],
                mode="lines+markers",
                name=project,
                # --- emphasis vs background
                opacity=1.0 if is_highlight or not highlight_projects else 0.25,
                line=dict(width=3 if is_highlight else 1.5),
                marker=dict(size=6 if is_highlight else 4),
            )
        )

    figure.add_hline(
        y=bench,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Benchmark {bench} MT/day",
        annotation_position="top left",
    )
    figure.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=50),
        xaxis_title="Month",
        yaxis_title="Avg Productivity (MT)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
    )
    LOGGER.debug("Project lines chart built for %d projects", len(all_projects))
    return figure



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
    metric: str = "tower_weight",       # "revenue" | "tower_weight"  (default tower_weight)
    title: str | None = None,
    top_n: int = 20
) -> go.Figure:
    if df.empty:
        return build_empty_responsibilities_figure("No Micro Plan data for the selected project.")

    metric = metric if metric in ("revenue", "tower_weight") else "tower_weight"
    axis_title = "Revenue" if metric == "revenue" else "Tower Weight (MT)"

    g = (df.groupby("entity_name", as_index=False)[metric]
           .sum()
           .sort_values(metric, ascending=False)
           .head(top_n))
    g["delivered"] = (0.10 * g[metric]).round(2)

    fig = go.Figure()
    fig.add_bar(y=g["entity_name"], x=g[metric], orientation="h", name="Responsibility", width=0.9)
    fig.add_bar(y=g["entity_name"], x=g["delivered"], orientation="h", name="Delivered (10%)", width=0.9)

    fig.update_layout(
        barmode="group",
        height=max(320, 22 * len(g) + 120),
        margin=dict(l=200, r=40, t=30, b=50),
        xaxis_title=axis_title,
        yaxis_title=entity_label,
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        title=title or None
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    return fig
