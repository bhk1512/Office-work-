"""Dash layout composition."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dcc import Download
import urllib.parse

from .analytics_layout import build_analytics_layout
from .stringing_analytics_layout import build_stringing_analytics_layout

CLICK_GRAPH_CONFIG = {
    "displayModeBar": False,
    "doubleClick": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
}

TODAY_DATE = datetime.today().date()
DEFAULT_COMPLETION_DATE = TODAY_DATE - timedelta(days=1)

# Inline SVG fragments for simple icons (white strokes)
_ICON_SHAPES = {
    "trend_down":  '<polyline points="4,8 10,14 13,11 20,18" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "trend_up":    '<polyline points="4,16 10,10 13,13 20,6"  stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "users":       ('<circle cx="9" cy="9" r="3" stroke="white" stroke-width="2" fill="none"/>'
                    '<circle cx="15" cy="9" r="3" stroke="white" stroke-width="2" fill="none"/>'
                    '<path d="M3 20c1-3 4-5 9-5s8 2 9 5" stroke="white" stroke-width="2" fill="none" stroke-linecap="round"/>'),
    "activity":    '<polyline points="3,12 7,12 10,3 14,21 17,12 21,12" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
}

def _icon(name: str, size: int = 18) -> html.Img:
    """Return a small white SVG icon as <img src='data:image/svg+xml;utf8,...'>"""
    inner = _ICON_SHAPES[name]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24">{inner}</svg>'
    )
    uri = "data:image/svg+xml;utf8," + urllib.parse.quote(svg)
    return html.Img(src=uri, style={"width": f"{size}px", "height": f"{size}px"})

def _build_trace_contents(
    trace_dropdown_id: str,
    export_button_id: str,
    idle_table_id: str,
    daily_table_id: str,
) -> list:
    """Create the shared traceability body layout."""

    return [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Label(
                                "Pick a gang (overrides Gang filter)",
                                className="fw-semibold mb-1",
                            ),
                            dcc.Dropdown(
                                id=trace_dropdown_id,
                                options=[],
                                value=None,
                                placeholder="Start typing a gang...",
                                clearable=True,
                                persistence=True,
                                persistence_type="session",
                                className="filter-select",
                            ),
                        ],
                        className="filter-field",
                    ),
                    md=6,
                ),
            ],
            className="mb-3 filter-card",
        ),
        
        dbc.Row(
            [
                dbc.Col(html.H5("Traceability"), md=8),
                dbc.Col(
                    dbc.Button(
                        "Export Trace Excel",
                        id=export_button_id,
                        color="primary",
                    ),
                    md=4,
                    className="text-end",
                ),
            ],
            className="align-items-center mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            "Idle Intervals (per gang)",
                            className="fw-bold mb-2",
                        ),
                        dash_table.DataTable(
                            id=idle_table_id,
                            columns=[
                                {"name": "Gang", "id": "gang_name"},
                                {"name": "Interval Start", "id": "interval_start"},
                                {"name": "Interval End", "id": "interval_end"},
                                {"name": "Raw Gap (days)", "id": "raw_gap_days"},
                                {"name": "Idle Counted (days)", "id": "idle_days_capped"},
                                {"name": "Baseline (MT/day)", "id": "baseline"},
                                {"name": "Cumulative Loss (MT)", "id": "cumulative_loss"},
                            ],
                            data=[],
                            page_size=10,
                            fixed_rows={"headers": True},
                            style_table={"overflowX": "auto", "maxHeight": "380px"},
                            style_cell={
                                "fontFamily": "Inter, system-ui",
                                "fontSize": 13,
                            },
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.Div(
                            "Daily Productivity (selected scope)",
                            className="fw-bold mb-2",
                        ),
                        dash_table.DataTable(
                            id=daily_table_id,
                            columns=[
                                {"name": "Gang", "id": "gang_name"},
                                {"name": "Project", "id": "project_name"},
                                {"name": "Date", "id": "date"},
                                {"name": "MT/day", "id": "daily_prod_mt"},
                            ],
                            data=[],
                            page_size=10,
                            fixed_rows={"headers": True},
                            style_table={"overflowX": "auto", "maxHeight": "380px"},
                            style_cell={
                                "fontFamily": "Inter, system-ui",
                                "fontSize": 13,
                                "border": "1px solid var(--border, #e6e9f0)",
                            },
                            style_header={"border": "1px solid var(--border, #e6e9f0)"},
                        ),
                    ],
                    md=6,
                ),
            ]
        ),
    ]


def build_controls() -> html.Div:
    """Inline filter controls for the header row."""
    return html.Div(
        [
            dcc.Dropdown(
                id="f-project",
                multi=True,
                placeholder="Select project(s)",
                className="filter-select filter-select--compact",
                style={"width": "260px"},
            ),
            dcc.Dropdown(
                id="f-month",
                multi=True,
                value=[],
                placeholder="Select month(s)",
                className="filter-select filter-select--compact",
                style={"width": "220px"},
            ),
            dbc.RadioItems(
                id="f-quick-range",
                options=[
                    {"label": "Last 3M", "value": "3M"},
                    {"label": "Last Qtr", "value": "Q"},
                    {"label": "Last 6M", "value": "6M"},
                    {"label": "YTD", "value": "YTD"},
                ],
                value=None,
                inline=True,
                class_name="segment segment--compact",
                label_class_name="segment-label",
                label_checked_class_name="segment-label--active",
                input_class_name="segment-input",
            ),
            html.Div(
                dcc.Dropdown(
                    id="f-line",
                    multi=True,
                    placeholder="Select line(s)",
                    className="filter-select filter-select--compact",
                    style={"width": "220px"},
                ),
                style={"display": "none"},
            ),
            html.Div(
                dbc.RadioItems(
                    id="f-time-lens",
                    options=[
                        {"label": "Monthly", "value": "monthly"},
                        {"label": "Weekly", "value": "weekly"},
                    ],
                    value="monthly",
                    inline=True,
                    class_name="segment segment--compact",
                    label_class_name="segment-label",
                    label_checked_class_name="segment-label--active",
                    input_class_name="segment-input",
                ),
                style={"display": "none"},
            ),
        ],
        className="topbar__filters",
    )


def _build_executive_kpi_card(
    title: str,
    value_id: str,
    sub_id: str,
) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(title, className="exec-kpi-title"),
                    html.Div(id=value_id, className="exec-kpi-value"),
                    html.Div(id=sub_id, className="exec-kpi-sub"),
                ]
            ),
            className="exec-kpi-card h-100",
        ),
        md=2,
        sm=6,
        xs=12,
    )


def build_executive_overview_layout() -> html.Div:
    return html.Div(
        [
            html.Div(id="exec-compliance-ticker", className="compliance-ticker", style={"display": "none"}),
            dcc.Interval(id="executive-refresh-interval", interval=5 * 60 * 1000, n_intervals=0),
            dcc.Store(id="executive-overview-payload", data={}),
            dcc.Store(id="exec-pch-expanded", data=[]),
            dbc.Row(
                [
                    _build_executive_kpi_card(
                        "Portfolio Completion",
                        "exec-kpi-portfolio-completion",
                        "exec-kpi-portfolio-completion-sub",
                    ),
                    _build_executive_kpi_card(
                        "Plan Attainment",
                        "exec-kpi-plan-attainment",
                        "exec-kpi-plan-attainment-sub",
                    ),
                    _build_executive_kpi_card(
                        "Stretch Readiness",
                        "exec-kpi-readiness",
                        "exec-kpi-readiness-sub",
                    ),
                    _build_executive_kpi_card(
                        "Erection-Stringing Gap",
                        "exec-kpi-gap",
                        "exec-kpi-gap-sub",
                    ),
                    _build_executive_kpi_card(
                        "Erection Productivity",
                        "exec-kpi-manpower",
                        "exec-kpi-manpower-sub",
                    ),
                    _build_executive_kpi_card(
                        "Stringing Productivity",
                        "exec-kpi-atrisk",
                        "exec-kpi-atrisk-sub",
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Div("Portfolio Status", className="section-title"),
                                    html.Div(
                                        [
                                            dbc.RadioItems(
                                                id="exec-portfolio-view",
                                                options=[
                                                    {"label": "Cumulative", "value": "cum"},
                                                    {"label": "This Month", "value": "month"},
                                                ],
                                                value="cum",
                                                inline=True,
                                                class_name="segment segment--compact",
                                                label_class_name="segment-label",
                                                label_checked_class_name="segment-label--active",
                                                input_class_name="segment-input",
                                            ),
                                        ]
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            ),
                            dbc.CardBody(
                                html.Div(id="exec-portfolio-table-container", style={"overflowX": "auto"}),
                                className="p-0",
                            ),
                        ],
                        className="viz-card shadow-soft",
                    ),
                    md=12,
                ),
                className="g-3 mb-3",
            ),
            html.Div(
                [
                    dash_table.DataTable(
                        id="exec-project-ranking-table",
                        columns=[],
                        data=[],
                        page_size=10,
                    ),
                    dcc.Graph(
                        id="exec-risk-driver-graph",
                        config=CLICK_GRAPH_CONFIG,
                        style={"height": "1px"},
                    ),
                ],
                style={"display": "none"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Leadership Callouts", className="section-title"),
                                        html.Div("Top 5 data-driven highlights for current scope", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    html.Ul(
                                        id="exec-callouts-list",
                                        className="exec-callouts-list",
                                        children=[html.Li("No data for selected scope.")],
                                    )
                                ),
                            ],
                            className="viz-card shadow-soft h-100",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Coverage Summary", className="section-title"),
                                        html.Div("Source availability by project and category", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="exec-coverage-summary-table",
                                        columns=[],
                                        data=[],
                                        page_size=8,
                                        fixed_rows={"headers": True},
                                        style_table={"overflowX": "auto", "maxHeight": "280px"},
                                        style_cell={
                                            "fontFamily": "Inter, system-ui",
                                            "fontSize": 12,
                                            "border": "1px solid var(--border, #e6e9f0)",
                                            "padding": "6px 8px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "fontWeight": "600",
                                            "backgroundColor": "#f6f8fc",
                                            "border": "1px solid var(--border, #e6e9f0)",
                                        },
                                    )
                                ),
                            ],
                            className="viz-card shadow-soft h-100",
                        ),
                        md=4,
                    ),
                ],
                className="g-3",
            ),
        ],
        className="dashboard-tab-content",
    )


def build_project_overview_layout() -> html.Div:
    def _completed_controls(range_id: str, search_id: str, title: str, subtitle: str) -> dbc.Row:
        return dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div(title, className="section-title mb-1"),
                            html.Div(subtitle, className="section-sub mb-2"),
                            dcc.DatePickerRange(
                                id=range_id,
                                min_date_allowed=datetime(2021, 1, 1),
                                max_date_allowed=TODAY_DATE,
                                start_date=DEFAULT_COMPLETION_DATE,
                                end_date=DEFAULT_COMPLETION_DATE,
                                display_format="DD-MM-YYYY",
                                minimum_nights=0,
                                persistence=True,
                                persistence_type="session",
                                className="filter-date",
                            ),
                        ],
                        className="filter-field",
                    ),
                    md=6,
                    lg=4,
                ),
                dbc.Col(
                    html.Div(
                        dbc.Input(
                            id=search_id,
                            placeholder="Filter by project, gang, or location",
                            type="text",
                            value="",
                            className="filter-input",
                        ),
                        className="filter-field",
                    ),
                    md=4,
                    lg=4,
                ),
                dbc.Col(
                    dbc.Button(
                        "Clear",
                        id=f"{search_id}-reset",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                    md=2,
                    lg=2,
                ),
            ],
            className="g-3 align-items-end mb-3",
        )

    def _completed_table(table_id: str, columns: list[dict[str, str]]) -> dash_table.DataTable:
        return dash_table.DataTable(
            id=table_id,
            columns=columns,
            data=[],
            page_size=15,
            sort_action="native",
            filter_action="native",
            fixed_rows={"headers": True},
            style_table={"overflowX": "auto", "maxHeight": "480px"},
            style_cell={
                "fontFamily": "Inter, system-ui",
                "fontSize": 13,
                "border": "1px solid var(--border, #e6e9f0)",
            },
            style_header={"border": "1px solid var(--border, #e6e9f0)"},
        )

    raw_data_accordion = dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    html.Div(
                        [
                            _completed_controls(
                                "proj-overview-erections-range",
                                "proj-overview-erections-search",
                                "Erections Completed",
                                "Completion date (defaults to yesterday)",
                            ),
                            _completed_table(
                                "proj-overview-erections-table",
                                [
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
                                ],
                            ),
                        ],
                        id="proj-overview-anchor-erections",
                    ),
                    html.Hr(className="my-3"),
                    html.Div(
                        [
                            _completed_controls(
                                "proj-overview-stringing-range",
                                "proj-overview-stringing-search",
                                "Stringing Completed",
                                "Filter by completion span",
                            ),
                            _completed_table(
                                "proj-overview-stringing-table",
                                [
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
                                ],
                            ),
                        ],
                        id="proj-overview-anchor-stringing",
                    ),
                ],
                title="Raw Data",
                item_id="raw-data",
            ),
        ],
        start_collapsed=True,
        always_open=False,
        className="exec-accordion",
    )

    perf_panel = dbc.Collapse(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Div("Erection Gangs", className="section-title")),
                                dbc.CardBody(
                                    [
                                        html.Div(id="proj-overview-avp-list", className="avp-wrap"),
                                        dcc.Graph(
                                            id="proj-overview-actual-vs-bench",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"display": "none"},
                                        ),
                                    ]
                                ),
                            ],
                            className="viz-card",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Div("Top / Bottom Gangs", className="section-title")),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="proj-overview-top5",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"height": "140px"},
                                        ),
                                        dcc.Graph(
                                            id="proj-overview-bottom5",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"height": "140px"},
                                        ),
                                    ]
                                ),
                            ],
                            className="viz-card",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        id="proj-overview-perf-collapse",
        is_open=False,
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Select Project",
                                className="fw-700 text-navy",
                                style={"fontSize": "0.8rem", "marginBottom": "4px"},
                            ),
                            dcc.Dropdown(
                                id="proj-overview-project-select",
                                options=[],
                                value=None,
                                placeholder="Choose a project...",
                                clearable=False,
                                className="filter-select filter-select--light",
                                style={"background": "white", "color": "var(--text)"},
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(id="proj-overview-rag-badge-container"),
                        md=2,
                        className="d-flex align-items-end",
                    ),
                    dbc.Col(
                        html.Div(
                            id="proj-overview-last-updated",
                            className="text-muted",
                            style={"fontSize": "0.78rem", "paddingBottom": "6px"},
                        ),
                        md=6,
                        className="d-flex align-items-end justify-content-end",
                    ),
                ],
                className="mb-3 align-items-end",
            ),
            dcc.Interval(id="project-overview-refresh-interval", interval=5 * 60 * 1000, n_intervals=0),
            dcc.Store(id="project-overview-payload", data={}),
            dcc.Store(id="store-project-overview-scope", data=None),
            dcc.Store(id="store-project-overview-performance-mode", data="erection|0"),
            html.Div(id="proj-dpr-strip", className="dpr-strip"),
            dbc.Row(
                [
                    _build_executive_kpi_card(
                        "Project Completion",
                        "proj-kpi-completion",
                        "proj-kpi-completion-sub",
                    ),
                    _build_executive_kpi_card(
                        "Plan Attainment",
                        "proj-kpi-plan",
                        "proj-kpi-plan-sub",
                    ),
                    _build_executive_kpi_card(
                        "Stretch Readiness",
                        "proj-kpi-readiness",
                        "proj-kpi-readiness-sub",
                    ),
                    _build_executive_kpi_card(
                        "Erection-Stringing Gap",
                        "proj-kpi-gap",
                        "proj-kpi-gap-sub",
                    ),
                    _build_executive_kpi_card(
                        "Erection Productivity",
                        "proj-kpi-manpower",
                        "proj-kpi-manpower-sub",
                    ),
                    _build_executive_kpi_card(
                        "Stringing Productivity",
                        "proj-kpi-stringing-productivity",
                        "proj-kpi-stringing-productivity-sub",
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Div("Activity Progress", className="section-title"),
                            html.Div("Cumulative actual vs scope", className="section-sub"),
                        ]
                    ),
                    dbc.CardBody(html.Div(id="proj-activity-breakdown", className="p-0")),
                ],
                className="viz-card mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Stretch Readiness", className="section-title"),
                                        html.Div(id="proj-stretch-summary-chip", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="proj-stretch-state-graph",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"height": "280px"},
                                        ),
                                        html.Div(id="proj-stretch-blocked-summary"),
                                    ]
                                ),
                            ],
                            className="viz-card h-100",
                        ),
                        md=5,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Stretch Readiness Gap Distribution", className="section-title"),
                                        html.Div("Distribution of E→S readiness gap buckets for the selected project", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="proj-es-lag-chart",
                                        config=CLICK_GRAPH_CONFIG,
                                        style={"height": "260px"},
                                    )
                                ),
                            ],
                            className="viz-card h-100",
                        ),
                        md=7,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Erection Productivity", className="section-title"),
                                        html.Div("MT/day — gang performance vs baseline", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="proj-erection-prod-chart",
                                        config=CLICK_GRAPH_CONFIG,
                                        style={"height": "240px"},
                                    )
                                ),
                            ],
                            className="viz-card",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Stringing Productivity", className="section-title"),
                                        html.Div("km/day — gang output vs baseline", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="proj-stringing-prod-chart",
                                        config=CLICK_GRAPH_CONFIG,
                                        style={"height": "240px"},
                                    )
                                ),
                            ],
                            className="viz-card",
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-3",
            ),
            perf_panel,
            dbc.Button(
                "Show Gang Performance",
                id="proj-overview-perf-toggle",
                color="link",
                size="sm",
                className="text-muted mt-2 mb-3",
                n_clicks=0,
            ),
            html.Div(
                [
                    dbc.RadioItems(
                        id="proj-overview-stringing-scope",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Manual", "value": "manual"},
                            {"label": "TSE", "value": "tse"},
                            {"label": "Hotline", "value": "hotline"},
                        ],
                        value="all",
                    ),
                    dbc.RadioItems(
                        id="proj-overview-topbot-metric",
                        options=[
                            {"label": "Productivity", "value": "prod"},
                            {"label": html.Span("Erection", id="proj-overview-topbot-mode-label"), "value": "erection"},
                        ],
                        value="prod",
                    ),
                ],
                style={"display": "none"},
            ),
            raw_data_accordion,
        ],
        className="dashboard-tab-content",
    )


def build_mode_summary_cards(
    *,
    show_global_buttons: bool = True,
    show_stringing_scope_control: bool = True,
) -> dbc.Row:
    """Twin overview cards for Erection and Stringing metrics."""

    def _card(title: str, rows: list[tuple[str, str, str]], mode_key: str) -> dbc.Col:
        header_controls = None
        if mode_key == "stringing" and show_stringing_scope_control:
            header_controls = html.Div(
                [
                    html.Div("Deployment", className="filter-label mb-1 me-2"),
                    dbc.RadioItems(
                        id="f-stringing-scope",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Manual", "value": "manual"},
                            {"label": "TSE", "value": "tse"},
                            {"label": "Hotline", "value": "hotline"},
                        ],
                        value="all",
                        inline=True,
                        class_name="segment segment--compact",
                        label_class_name="segment-label",
                        label_checked_class_name="segment-label--active",
                        input_class_name="segment-input",
                    ),
                ],
                className="stringing-scope-control d-flex flex-wrap align-items-center gap-2",
            )

        button = (
            dbc.Button(
                "Show Overall Gang Performance",
                id=f"btn-open-global-performance-{mode_key}",
                color="primary",
                size="sm",
                className="summary-card__cta",
            )
            if show_global_buttons
            else None
        )

        def _header_block() -> html.Div:
            left_children: list[Any] = [html.Div(title, className="fw-semibold")]
            if header_controls:
                left_children.append(header_controls)
            return html.Div(
                left_children,
                className="d-flex flex-wrap align-items-center gap-2",
            )

        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [_header_block(), button] if button is not None else [_header_block()],
                            className="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3",
                        ),
                        *[
                            html.Div(
                                [
                                    html.Span(label, className="summary-pill__label"),
                                    html.Span(id=value_id, children="-", className="summary-pill__value"),
                                ],
                                className="summary-pill",
                                role="button",
                                tabIndex=0,
                                id={"type": "summary-pill-trigger", "mode": mode_key, "metric": metric_key},
                                n_clicks=0,
                            )
                            for label, value_id, metric_key in rows
                        ],
                    ],
                    className="d-flex flex-column gap-2",
                ),
                className="shadow-sm h-100",
            ),
            md=6,
        )

    erection_rows = [
        ("Projects Covered", "erection-card-projects", "projects"),
        ("Total Planned / Done / Balance", "erection-card-totals", "totals"),
        ("Gangs", "erection-card-gangs", "gangs"),
        ("Productivity / Historical Avg", "erection-card-productivity", "productivity"),
        ("Lost Units", "erection-card-loss", "loss"),
    ]
    stringing_rows = [
        ("Projects Covered", "stringing-card-projects", "projects"),
        ("F/S Total Planned / Done / Balance", "stringing-card-totals", "totals"),
        ("Gangs", "stringing-card-gangs", "gangs"),
        ("Productivity / Historical Avg", "stringing-card-productivity", "productivity"),
        ("Lost Units", "stringing-card-loss", "loss"),
        ("No. of TSE", "stringing-card-tse", "tse"),
    ]

    return dbc.Row(
        [
            _card("Erection", erection_rows, "erection"),
            _card("Stringing", stringing_rows, "stringing"),
        ],
        className="g-3 mb-3",
    )


# Lucide paths (top-right icons)
_LUCIDE_TREND_DOWN = "7 7 17 17M17 7h0v10H7"
_LUCIDE_USERS      = "17 21v-2a4 4 0 0 0-4-4H11a4 4 0 0 0-4 4v2M7 7a4 4 0 1 0 8 0 4 4 0 0 0-8 0"
_LUCIDE_TREND_UP   = "7 17 17 7M7 7h10v10"
_LUCIDE_ACTIVITY   = "22 12h-4l-3 9-6-18-3 9H2"

def build_kpi_cards() -> dbc.Row:
    # Keep nodes present for callbacks but hide them from the cleaned-up home view.
    return dbc.Row(
        [
            # 1) Avg Output / Gang / Day  (blue)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(id="label-avg", children="Avg Output / Gang / Day", className="kpi-label"),
                            html.Div(
                                [
                                    html.Span(id="kpi-avg", className="kpi-value"),
                                    html.Span(id="kpi-delta", className="kpi-delta"),
                                ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    className="kpi kpi--blue",
                ),           
            ),

            # 2) Active Projects (purple)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Active Gangs", className="kpi-label"),
                            html.Div(
                                [ html.Span(id="kpi-active", className="kpi-value") ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    className="kpi kpi--purple",
                ),
           ),

            # 3) Towers Erected - visible only in erection mode
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(id="label-total-nos", children="Towers Erected", className="kpi-label"),
                            html.Div(
                                [
                                    html.Span(id="kpi-total-nos", className="kpi-value"),
                                    html.Span(id="kpi-total-nos-planned", className="kpi-delta"),
                                ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    id="kpi-card-total-nos",
                    className="kpi kpi--green",
                ),
                id="card-total-nos",
            ),

            # 3b) Total Erection (MT) – visible only in erection mode
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(id="label-total", children="Volume Erected", className="kpi-label"),
                            html.Div(
                                [
                                    html.Span(id="kpi-total", className="kpi-value"),
                                    html.Span(id="kpi-total-planned", className="kpi-delta"),
                                ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    id="kpi-card-total-mt",
                    className="kpi kpi--green",
                ),
            ),

            # 4) Lost Units (red)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(id="label-lost", children="Lost Units", className="kpi-label"),
                            html.Div(
                                [
                                    html.Span(id="kpi-loss", className="kpi-value"),
                                    html.Span(id="kpi-loss-delta", className="kpi-delta"),
                                ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    className="kpi kpi--red",
                ),
            ),
        ],
        id="kpi-row",
        className="g-3 align-items-stretch row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-lg-4 row-cols-xl-5",
        style={"display": "none"},
    )


def build_project_details_card() -> dbc.Card:
    """Project Overview card: the body is dynamic (message OR 3-col grid)."""
    # Keep element for callbacks but hide from the streamlined home view.
    return dbc.Card(
        dbc.CardBody(
            [
                # Header (title is filled by callback)
                html.Div(
                    [html.Div(id="pd-title", className="project-card__title", children="Project Overview")],
                    className="project-card__head",
                ),

                # Body (callback will inject either message OR the 3-column grid)
                html.Div(
                    id="project-details",
                    className="project-details__body",
                    children=html.Div("Select a single project to view its details.", className="project-empty"),
                ),
            ],
            className="project-card",        # blue surface on CardBody
        ),
        className="mb-3 project-card-wrap",  # neutral wrapper
        style={"display": "none"},
    )


ROW_PX = 55
VISIBLE_ROWS = 10
TOPBOT_MARGIN = 120
CONTAINER_HEIGHT = ROW_PX * VISIBLE_ROWS + TOPBOT_MARGIN


def build_trace_block() -> dbc.Card:
    """Return the traceability card with tables and export controls."""

    contents = [
    html.Div(id="trace-anchor"),  # <-- anchor lives ONLY in the main page, not the modal
    ]
    contents += _build_trace_contents(
        "trace-gang",
        "btn-export-trace",
        "tbl-idle-intervals",
        "tbl-daily-prod",
    )
    contents.extend([
        Download(id="download-trace-xlsx"),
    ])
    # Keep block rendered (callbacks expect ids) but hide it on the simplified home screen.
    return dbc.Card(
        dbc.CardBody(contents),
        className="mt-4 shadow-sm",
        style={"display": "none"},
    )



def build_erections_card() -> dbc.Card:
    """Standalone card that lists completed erections for the selected filters."""

    controls = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Div("Erections Completed", className="section-title mb-2", id="lbl-erections-title"),
                        # html.Div(
                        #     "Completion date (defaults to yesterday)",
                        #     className="fw-semibold mb-1",
                        # ),
                        dcc.DatePickerRange(
                            id="erections-completion-range",
                            min_date_allowed=datetime(2021, 1, 1),
                            max_date_allowed=TODAY_DATE,
                            start_date=DEFAULT_COMPLETION_DATE,
                            end_date=DEFAULT_COMPLETION_DATE,
                            display_format="DD-MM-YYYY",
                            minimum_nights=0,
                            persistence=True,
                            persistence_type="session",
                            className="filter-date",
                        ),
                    ],
                    className="filter-field",
                ),
                md=6,
                lg=4,
            ),
            dbc.Col(
                html.Div(
                    dbc.Input(
                        id="erections-search",
                        placeholder="Filter by project, gang, or location",
                        type="text",
                        value="",
                        className="filter-input",
                    ),
                    className="filter-field",
                ),
                md=4,
                lg=4,
            ),
            dbc.Col(
                dbc.Button(
                    "Reset",
                    id="btn-reset-erections",
                    color="secondary",
                    outline=True,
                    className="w-100",
                ),
                md=2,
                lg=2,
            ),
        ],
        className="g-3 align-items-end mb-3 filter-card",
    )

    table = dash_table.DataTable(
        id="tbl-erections-completed",
        columns=[
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
        ],
        data=[],
        page_size=15,
        sort_action="native",
        filter_action="native",
        fixed_rows={"headers": True},
        style_table={"overflowX": "auto", "maxHeight": "500px"},
        style_cell={
            "fontFamily": "Inter, system-ui",
            "fontSize": 13,
            "border": "1px solid var(--border, #e6e9f0)",
        },
        style_header={"border": "1px solid var(--border, #e6e9f0)"},
    )

    body = [controls, table]

    return dbc.Card(
        dbc.CardBody(body),
        className="mt-4 shadow-sm",
        style={"display": "none"},
    )


def build_trace_modal() -> dbc.Modal:
    """Return the modal that mirrors the traceability section."""

    modal_contents = _build_trace_contents(
        "modal-trace-gang",
        "modal-btn-export-trace",
        "modal-tbl-idle-intervals",
        "modal-tbl-daily-prod",
    )
    modal_card = dbc.Card(
        dbc.CardBody(modal_contents),
        className="shadow-sm",
    )
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="trace-modal-title")),
            dbc.ModalBody(modal_card),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="trace-modal-close",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        id="trace-modal",
        is_open=False,
        size="xl",
        scrollable=True,
    )


def build_project_tile_modal() -> dbc.Modal:
    """Large project modal with status-first layout and compact raw-data access."""

    def _completed_controls(range_id: str, search_id: str, title: str, subtitle: str) -> dbc.Row:
        return dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div(title, className="section-title mb-1"),
                            html.Div(subtitle, className="section-sub mb-2"),
                            dcc.DatePickerRange(
                                id=range_id,
                                min_date_allowed=datetime(2021, 1, 1),
                                max_date_allowed=TODAY_DATE,
                                start_date=DEFAULT_COMPLETION_DATE,
                                end_date=DEFAULT_COMPLETION_DATE,
                                display_format="DD-MM-YYYY",
                                minimum_nights=0,
                                persistence=True,
                                persistence_type="session",
                                className="filter-date",
                            ),
                        ],
                        className="filter-field",
                    ),
                    md=6,
                    lg=4,
                ),
                dbc.Col(
                    html.Div(
                        dbc.Input(
                            id=search_id,
                            placeholder="Filter by project, gang, or location",
                            type="text",
                            value="",
                            className="filter-input",
                        ),
                        className="filter-field",
                    ),
                    md=4,
                    lg=4,
                ),
                dbc.Col(
                    dbc.Button(
                        "Clear",
                        id=f"{search_id}-reset",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                    md=2,
                    lg=2,
                ),
            ],
            className="g-3 align-items-end mb-3",
        )

    def _completed_table(table_id: str, columns: list[dict[str, str]]) -> dash_table.DataTable:
        return dash_table.DataTable(
            id=table_id,
            columns=columns,
            data=[],
            page_size=15,
            sort_action="native",
            filter_action="native",
            fixed_rows={"headers": True},
            style_table={"overflowX": "auto", "maxHeight": "480px"},
            style_cell={
                "fontFamily": "Inter, system-ui",
                "fontSize": 13,
                "border": "1px solid var(--border, #e6e9f0)",
            },
            style_header={"border": "1px solid var(--border, #e6e9f0)"},
        )

    raw_data_accordion = dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    html.Div(
                        [
                            _completed_controls(
                                "project-modal-erections-range",
                                "project-modal-erections-search",
                                "Erections Completed",
                                "Completion date (defaults to yesterday)",
                            ),
                            _completed_table(
                                "project-modal-erections-table",
                                [
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
                                ],
                            ),
                        ],
                        id="project-modal-anchor-erections",
                    ),
                    html.Hr(className="my-3"),
                    html.Div(
                        [
                            _completed_controls(
                                "project-modal-stringing-range",
                                "project-modal-stringing-search",
                                "Stringing Completed",
                                "Filter by completion span",
                            ),
                            _completed_table(
                                "project-modal-stringing-table",
                                [
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
                                ],
                            ),
                        ],
                        id="project-modal-anchor-stringing",
                    ),
                ],
                title="Raw Data",
                item_id="raw-data",
            ),
        ],
        start_collapsed=True,
        always_open=False,
        className="exec-accordion",
    )

    perf_panel = dbc.Collapse(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Div("Erection Gangs", className="section-title")),
                                dbc.CardBody(
                                    [
                                        html.Div(id="project-modal-avp-list", className="avp-wrap"),
                                        dcc.Graph(
                                            id="project-modal-actual-vs-bench",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"display": "none"},
                                        ),
                                    ]
                                ),
                            ],
                            className="viz-card",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Div("Top / Bottom Gangs", className="section-title")),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="project-modal-top5",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"height": "140px"},
                                        ),
                                        dcc.Graph(
                                            id="project-modal-bottom5",
                                            config=CLICK_GRAPH_CONFIG,
                                            style={"height": "140px"},
                                        ),
                                    ]
                                ),
                            ],
                            className="viz-card",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        id="project-modal-perf-collapse",
        is_open=False,
    )

    compatibility_stubs = html.Div(
        [
            html.Div(id="project-modal-summary"),
            dbc.Collapse(html.Div(), id="project-modal-section-erections", is_open=False),
            dbc.Collapse(html.Div(), id="project-modal-section-stringing", is_open=False),
            dbc.Collapse(html.Div(), id="project-modal-section-performance", is_open=False),
            html.Div(id="project-modal-anchor-performance"),
            dbc.Button(id="project-modal-btn-erections", style={"display": "none"}, n_clicks=0),
            dbc.Button(id="project-modal-btn-stringing", style={"display": "none"}, n_clicks=0),
            dbc.Button(id="project-modal-btn-performance-erection", style={"display": "none"}, n_clicks=0),
            dbc.Button(id="project-modal-btn-performance-stringing", style={"display": "none"}, n_clicks=0),
            html.Div(
                [
                    dbc.RadioItems(
                        id="project-modal-stringing-scope",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Manual", "value": "manual"},
                            {"label": "TSE", "value": "tse"},
                            {"label": "Hotline", "value": "hotline"},
                        ],
                        value="all",
                    ),
                    dbc.RadioItems(
                        id="project-modal-topbot-metric",
                        options=[
                            {"label": "Productivity", "value": "prod"},
                            {"label": html.Span("Erection", id="project-modal-topbot-mode-label"), "value": "erection"},
                        ],
                        value="prod",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(id="project-modal-trace-anchor"),
                    *_build_trace_contents(
                        "project-modal-trace-gang",
                        "project-modal-btn-export-trace",
                        "project-modal-tbl-idle-intervals",
                        "project-modal-tbl-daily-prod",
                    ),
                    Download(id="project-modal-download-trace"),
                ]
            ),
        ],
        style={"display": "none"},
    )

    body_sections = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Select Project",
                            className="fw-700 text-navy",
                            style={"fontSize": "0.8rem", "marginBottom": "4px"},
                        ),
                        dcc.Dropdown(
                            id="project-modal-project-select",
                            options=[],
                            value=None,
                            placeholder="Choose a project...",
                            clearable=False,
                            className="filter-select filter-select--light",
                            style={"background": "white", "color": "var(--text)"},
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    html.Div(id="project-modal-rag-badge-container"),
                    md=2,
                    className="d-flex align-items-end",
                ),
                dbc.Col(
                    html.Div(
                        id="project-modal-last-updated",
                        className="text-muted",
                        style={"fontSize": "0.78rem", "paddingBottom": "6px"},
                    ),
                    md=6,
                    className="d-flex align-items-end justify-content-end",
                ),
            ],
            className="mb-3 align-items-end",
        ),
        html.Div(
            [
                html.Div(id="project-modal-dpr-strip", className="dpr-strip"),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                _build_executive_kpi_card(
                    "Project Completion",
                    "project-modal-kpi-completion",
                    "project-modal-kpi-completion-sub",
                ),
                _build_executive_kpi_card(
                    "Plan Attainment",
                    "project-modal-kpi-plan",
                    "project-modal-kpi-plan-sub",
                ),
                _build_executive_kpi_card(
                    "Stretch Readiness",
                    "project-modal-kpi-readiness",
                    "project-modal-kpi-readiness-sub",
                ),
                _build_executive_kpi_card(
                    "Erection-Stringing Gap",
                    "project-modal-kpi-gap",
                    "project-modal-kpi-gap-sub",
                ),
                _build_executive_kpi_card(
                    "Erection Productivity",
                    "project-modal-kpi-manpower",
                    "project-modal-kpi-manpower-sub",
                ),
                _build_executive_kpi_card(
                    "Stringing Productivity",
                    "project-modal-kpi-stringing-productivity",
                    "project-modal-kpi-stringing-productivity-sub",
                ),
            ],
            className="g-3 mb-3",
        ),
        dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.Div("Activity Progress", className="section-title"),
                        html.Div("Cumulative actual vs scope", className="section-sub"),
                    ]
                ),
                dbc.CardBody(html.Div(id="project-modal-activity-rows")),
            ],
            className="viz-card mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Div("Stretch Readiness", className="section-title"),
                                    html.Div(id="project-modal-stretch-summary-chip", className="section-sub"),
                                ]
                            ),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="project-modal-stretch-pie",
                                        config=CLICK_GRAPH_CONFIG,
                                        style={"height": "280px"},
                                    ),
                                    html.Div(id="project-modal-stretch-blocked-summary"),
                                ]
                            ),
                        ],
                        className="viz-card h-100",
                    ),
                    md=5,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Div("Stretch Readiness Gap Distribution", className="section-title"),
                                    html.Div("Distribution of E->S readiness gap buckets for the selected project", className="section-sub"),
                                ]
                            ),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="project-modal-es-lag",
                                    config=CLICK_GRAPH_CONFIG,
                                    style={"height": "260px"},
                                )
                            ),
                        ],
                        className="viz-card h-100",
                    ),
                    md=7,
                ),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Div("Erection Productivity", className="section-title"),
                                    html.Div("MT/day - gang performance vs baseline", className="section-sub"),
                                ]
                            ),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="project-modal-erection-prod-chart",
                                    config=CLICK_GRAPH_CONFIG,
                                    style={"height": "240px"},
                                )
                            ),
                        ],
                        className="viz-card",
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Div("Stringing Productivity", className="section-title"),
                                    html.Div("km/day - gang output vs baseline", className="section-sub"),
                                ]
                            ),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="project-modal-stringing-prod-chart",
                                    config=CLICK_GRAPH_CONFIG,
                                    style={"height": "240px"},
                                )
                            ),
                        ],
                        className="viz-card",
                    ),
                    md=6,
                ),
            ],
            className="g-3 mb-3",
        ),
        perf_panel,
        dbc.Button(
            "Show Gang Performance",
            id="project-modal-perf-toggle",
            color="link",
            size="sm",
            className="text-muted mt-2 mb-3",
            n_clicks=0,
        ),
        raw_data_accordion,
        compatibility_stubs,
        html.Div(id="project-modal-scroll-wire", style={"display": "none"}),
    ]
    return dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    dbc.ModalTitle(id="project-modal-title", children="Project Deep Dive"),
                    html.Button(
                        type="button",
                        className="btn-close project-modal-close-top",
                        id="project-modal-close-top",
                        n_clicks=0,
                        **{"aria-label": "Close project details", "title": "Close"},
                    ),
                ],
                close_button=False,
                className="project-modal-header",
            ),
            dbc.ModalBody(body_sections, className="project-modal-body"),
        ],
        id="project-detail-modal",
        is_open=False,
        size="xl",
        fullscreen=True,
        scrollable=True,
        backdrop="static",
    )


def build_global_performance_modal() -> dbc.Modal:
    """Modal that surfaces the global gang performance views with dedicated filters."""

    filter_controls = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Label("Project(s)", className="fw-semibold mb-1"),
                                            dcc.Dropdown(
                                                id="global-performance-projects",
                                                multi=True,
                                                placeholder="Select project(s)",
                                                className="filter-select",
                                                persistence=True,
                                                persistence_type="session",
                                            ),
                                        ],
                                        className="filter-field",
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Label("Month(s)", className="fw-semibold mb-1"),
                                            dcc.Dropdown(
                                                id="global-performance-months",
                                                multi=True,
                                                placeholder="Select month(s)",
                                                className="filter-select",
                                                persistence=True,
                                                persistence_type="session",
                                            ),
                                        ],
                                        className="filter-field",
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Label(
                                                "Benchmark (MT/day)",
                                                className="fw-semibold mb-1",
                                                id="global-performance-benchmark-label",
                                            ),
                                            dcc.Input(
                                                id="global-performance-benchmark",
                                                type="number",
                                                placeholder="Enter benchmark",
                                                min=0,
                                                step=0.1,
                                                debounce=True,
                                                className="filter-input",
                                            ),
                                            html.Div(
                                                "Only gangs beating this benchmark will be listed below.",
                                                className="form-text text-muted small mt-1",
                                            ),
                                        ],
                                        className="filter-field",
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Label(
                                                "Min Erections",
                                                className="fw-semibold mb-1",
                                                id="global-performance-erections-threshold-label",
                                            ),
                                            dcc.Input(
                                                id="global-performance-min-erections",
                                                type="number",
                                                placeholder="Enter minimum",
                                                min=0,
                                                step=1,
                                                debounce=True,
                                                className="filter-input",
                                            ),
                                            html.Div(
                                                "Only gangs with more than this number of erections are considered.",
                                                className="form-text text-muted small mt-1",
                                            ),
                                        ],
                                        className="filter-field",
                                    ),
                                    md=3,
                                ),
                            ],
                            className="g-3",
                        ),
                    ]
                )
            ]
        ),
        className="shadow-sm mb-4",
    )

    benchmark_table = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div("Benchmark Highlights", className="section-title"),
                        html.Div(
                            "Gangs exceeding the entered benchmark.",
                            className="section-sub",
                        ),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    id="global-performance-benchmark-status",
                    className="text-muted small mb-2",
                    children="Enter a benchmark to view the leading gangs.",
                ),
                dash_table.DataTable(
                    id="global-performance-benchmark-table",
                    columns=[
                        {"name": "Gang", "id": "name"},
                        {"name": "Project", "id": "project"},
                        {"name": "Last Worked At", "id": "last_worked_at"},
                        {"name": "Erections", "id": "erections"},
                        {"name": "Current MT/day", "id": "current_rate"},
                        {"name": "Baseline MT/day", "id": "baseline_rate"},
                    ],
                    data=[],
                    page_size=8,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontFamily": "Inter, system-ui",
                        "fontSize": 13,
                        "border": "1px solid var(--border, #e6e9f0)",
                        "padding": "6px 8px",
                    },
                    style_header={"backgroundColor": "#f8fafc", "fontWeight": "600"},
                ),
            ]
        ),
        className="shadow-sm mb-4",
    )

    performance_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                className="section-header",
                                children=[
                                    html.Div(
                                        [
                                            html.Div("Gang Performance", className="section-title"),
                                            html.Div(
                                                "Delivered vs Lost (selected scope)",
                                                className="section-sub",
                                            ),
                                        ],
                                        className="d-flex flex-column gap-1",
                                    ),
                                    html.Div(
                                        className="legend",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Span(className="legend__dot dot--delivered"),
                                                    "Delivered Output",
                                                ],
                                                className="legend__item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(className="legend__dot dot--lost"),
                                                    "Lost Potential",
                                                ],
                                                className="legend__item",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Hr(style={"borderColor": "var(--border)", "margin": "8px 0 10px"}),
                            html.Div(id="global-performance-avp-list", className="avp-wrap"),
                            dcc.Graph(
                                id="global-performance-actual-vs-bench",
                                config=CLICK_GRAPH_CONFIG,
                                style={"display": "none"},
                            ),
                        ]
                    ),
                    className="viz-card shadow-sm",
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Div("Performance Rankings", className="section-title"),
                                                html.Div(
                                                    "Top and bottom gangs (selected scope)",
                                                    className="section-sub",
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.Col(
                                        dbc.RadioItems(
                                            id="global-performance-topbot-metric",
                                            options=[
                                                {"label": "Productivity", "value": "prod"},
                                                {
                                                    "label": html.Span(
                                                        "Erection",
                                                        id="global-performance-topbot-mode-label",
                                                    ),
                                                    "value": "erection",
                                                },
                                            ],
                                            value="prod",
                                            inline=True,
                                            class_name="segment",
                                            label_class_name="segment-label",
                                            label_checked_class_name="segment-label--active",
                                            input_class_name="segment-input",
                                        ),
                                        width="auto",
                                        align="center",
                                    ),
                                ],
                                justify="between",
                                align="center",
                            )
                        ),
                        dbc.CardBody(
                            [
                                html.Div("Top 5 Performers", className="text-success fw-semibold mb-2"),
                                dcc.Graph(
                                    id="global-performance-top5",
                                    config=CLICK_GRAPH_CONFIG,
                                    style={"cursor": "pointer"},
                                ),
                                html.Hr(className="my-3"),
                                html.Div("Bottom 5 Performers", className="text-danger fw-semibold mb-2"),
                                dcc.Graph(
                                    id="global-performance-bottom5",
                                    config=CLICK_GRAPH_CONFIG,
                                    style={"cursor": "pointer"},
                                ),
                            ]
                        ),
                    ],
                    className="viz-card shadow-sm",
                ),
                md=6,
            ),
        ],
        className="mb-4",
    )

    trace_contents = [
        html.Div(id="global-performance-trace-anchor"),
        *_build_trace_contents(
            "global-performance-trace-gang",
            "global-performance-btn-export-trace",
            "global-performance-tbl-idle-intervals",
            "global-performance-tbl-daily-prod",
        ),
    ]
    trace_block = dbc.Card(
        dbc.CardBody(trace_contents + [Download(id="global-performance-download-trace")]),
        className="shadow-sm",
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Gang Performance (All Projects)")),
            dbc.ModalBody([filter_controls, benchmark_table, performance_cards, trace_block]),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="global-performance-modal-close",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        id="global-performance-modal",
        is_open=False,
        size="xl",
        fullscreen=True,
        scrollable=True,
        backdrop="static",
    )
def build_kpi_pch_modal() -> dbc.Modal:
    """Modal variant of the PCH-wise drilldown used when summary pills are clicked."""

    body = dbc.Card(
        dbc.CardBody(
            [
                dbc.Accordion(
                    id="kpi-pch-modal-accordion",
                    start_collapsed=True,
                    always_open=False,
                    flush=True,
                    active_item=None,
                    className="pch-accordion",
                )
            ]
        ),
        className="shadow-sm",
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(id="kpi-pch-modal-title", children="PCH-wise Planned vs Delivered")
            ),
            dbc.ModalBody(body),
            dbc.ModalFooter(
                dbc.Button("Close", id="kpi-pch-modal-close", className="ms-auto", n_clicks=0)
            ),
        ],
        id="kpi-pch-modal",
        is_open=False,
        size="xl",
        scrollable=True,
    )

def build_project_responsibilities_modal() -> dbc.Modal:
    """Nested mini-modal to show Monthly Plan chart for a selected project."""
    body = dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(
                    id="proj-resp-graph",
                    config={"displayModeBar": False},
                    responsive=True,
                    style={"height": "360px", "minHeight": "280px", "width": "100%"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(id="proj-resp-kpi-target", className="kpi-value"),
                                        html.Div("Total Target", className="kpi-sub"),
                                    ]
                                ),
                                className="kpi kpi-blue",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(id="proj-resp-kpi-delivered", className="kpi-value"),
                                        html.Div("Delivered", className="kpi-sub"),
                                    ]
                                ),
                                className="kpi kpi-green",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(id="proj-resp-kpi-ach", className="kpi-value"),
                                        html.Div("Achievement", className="kpi-sub"),
                                    ]
                                ),
                                className="kpi kpi-red",
                            ),
                            md=4,
                        ),
                    ],
                    className="kpi-row-compact",
                ),
            ]
        ),
        className="shadow-sm responsibilities-modal-card",
    )
    # Header with title on left and local filter pills on right (mirrors main card)
    header = dbc.ModalHeader(
        dbc.Row(
            [
                dbc.Col(
                    dbc.ModalTitle(id="proj-resp-modal-title"),
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.RadioItems(
                                id="proj-resp-entity",
                                options=[
                                    {"label": "Gangs", "value": "Gang"},
                                    {"label": "Section Incharges", "value": "Section Incharge"},
                                    {"label": "Supervisors", "value": "Supervisor"},
                                ],
                                value="Supervisor",
                                inline=True,
                                class_name="segment segment-xxs",
                                label_class_name="segment-label",
                                label_checked_class_name="segment-label--active",
                                input_class_name="segment-input",
                            ),
                            dbc.RadioItems(
                                id="proj-resp-metric",
                                options=[
                                    {"label": "Tower Weight", "value": "tower_weight"},
                                    {"label": "Revenue", "value": "revenue"},
                                ],
                                value="tower_weight",
                                inline=True,
                                class_name="segment segment-xxs",
                                label_class_name="segment-label",
                                label_checked_class_name="segment-label--active",
                                input_class_name="segment-input",
                            ),
                        ],
                        className="header-pills d-flex flex-row align-items-center justify-content-end",
                    ),
                    width="auto",
                ),
            ],
            className="align-items-center  justify-content-between g-2",
        ),
        close_button=True,
    )

    return dbc.Modal(
        [
            header,
            dbc.ModalBody(body),
            dbc.ModalFooter(
                dbc.Button("Close", id="proj-resp-modal-close", className="ms-auto")
            ),
        ],
        id="proj-resp-modal",
        is_open=False,
        size="xl",
        scrollable=True,
        backdrop=True,
        keyboard=True,
        content_class_name="responsibilities-modal",
    )

def build_header(title: str, last_updated_text: str) -> html.Div:
    """Top section: title, controls, and actions in a single top bar row."""

    # Build small inline SVGs as IMG data URIs (Dash-safe across versions)
    cube_svg_str = '''
<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
      xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M12 2L20 6.5V17.5L12 22L4 17.5V6.5L12 2Z" stroke="white" stroke-width="1.6"/>
  <path d="M12 2V12L20 17.5" stroke="white" stroke-width="1.6"/>
  <path d="M12 12L4 17.5" stroke="white" stroke-width="1.6"/>
</svg>
'''.strip()
    cube_img = html.Img(
        src="data:image/svg+xml;utf8," + urllib.parse.quote(cube_svg_str),
        style={"width": "18px", "height": "18px"},
    )

    controls = build_controls()

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(cube_img, className="brand-badge"),
                            html.Div(title, className="topbar__title"),
                        ],
                        className="topbar__left",
                    ),
                    controls,
                    html.Div(
                        [
                            html.Div(
                                f"Last updated: {last_updated_text}",
                                className="topbar__updated",
                            ),
                            dbc.Button(
                                "Export Executive PDF",
                                id="btn-export-executive-pdf",
                                color="primary",
                                size="sm",
                                className="topbar__reset",
                            ),
                            dbc.Button(
                                "Reset filters",
                                id="btn-reset-filters",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="topbar__reset",
                            ),
                            html.A(
                                "Clear quick range",
                                id="link-clear-quick-range",
                                n_clicks=0,
                                className="topbar__clear-link",
                                style={"display": "none"},
                            ),
                        ],
                        className="topbar__right topbar__actions",
                    ),
                ],
                className="topbar__row",
            ),
            html.Div(
                dcc.Dropdown(
                    id="f-gang",
                    multi=True,
                    placeholder="Select gang(s)",
                    className="filter-select",
                ),
                style={"display": "none"},
            ),
        ],
        className="topbar",
    )



def build_layout(last_updated_text: str) -> dbc.Container:
    """Assemble the full Dash layout."""

    trace_modal = build_trace_modal()
    pch_modal = build_kpi_pch_modal()
    project_modal = build_project_tile_modal()
    global_performance_modal = build_global_performance_modal()
    analytics_layout = build_analytics_layout()
    stringing_analytics_layout = build_stringing_analytics_layout()
    executive_layout = build_executive_overview_layout()
    project_overview_layout = build_project_overview_layout()

    home_content = html.Div(
        [
            build_mode_summary_cards(
                show_global_buttons=False,
                show_stringing_scope_control=False,
            ),
            build_kpi_cards(),
            build_project_details_card(),
            # Hide historical graph + plan cards to keep home screen minimal while callbacks remain intact.
            # Hide gang performance + ranking cards on the home screen while keeping callbacks wired.
            dbc.Row(
                [
                    # LEFT: Projects over Months (only)
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div("Projects over Months", className="section-title"),
                                        html.Div("Monthly output trends for selected projects", className="section-sub"),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="g-projects-over-months",
                                            config={"displayModeBar": False},
                                            style={"marginBottom": "6px", "height": "360px"},
                                        ),
                                    ],
                                    className="d-flex flex-column",
                                ),
                            ],
                            className="viz-card shadow-soft section-gap-top flex-fill w-100",
                        ),
                        md=6,
                        className="d-flex",
                    ),
                    # RIGHT: Monthly Plan cards
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Row(
                                                [
                                                    # Left: Title + subtitle
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Div("Monthly Plan (Erection)", className="section-title"),
                                                                html.Div(
                                                                    [
                                                                        "Target vs Delivered ",
                                                                        html.Span(
                                                                            "(All periods)",
                                                                            id="label-resp-period",
                                                                        ),
                                                                    ],
                                                                    className="section-sub",
                                                                ),
                                                            ]
                                                        ),
                                                        className="d-flex flex-column justify-content-center",
                                                        lg=7, md=7, sm=12,
                                                    ),
                                                    # Right: filter pills
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                dbc.RadioItems(
                                                                    id="f-resp-entity",
                                                                    options=[
                                                                        {"label": "Gangs", "value": "Gang"},
                                                                        {"label": "Section Incharges", "value": "Section Incharge"},
                                                                        {"label": "Supervisors", "value": "Supervisor"},
                                                                    ],
                                                                    value="Supervisor",
                                                                    inline=True,
                                                                    class_name="segment segment-xxs",
                                                                    label_class_name="segment-label",
                                                                    label_checked_class_name="segment-label--active",
                                                                    input_class_name="segment-input",
                                                                ),
                                                                dbc.RadioItems(
                                                                    id="f-resp-metric",
                                                                    options=[
                                                                        {"label": "Tower Weight", "value": "tower_weight"},
                                                                        {"label": "Revenue", "value": "revenue"},
                                                                    ],
                                                                    value="tower_weight",
                                                                    inline=True,
                                                                    class_name="segment segment-xxs",
                                                                    label_class_name="segment-label",
                                                                    label_checked_class_name="segment-label--active",
                                                                    input_class_name="segment-input",
                                                                ),
                                                            ],
                                                            className="header-pills d-flex flex-row align-items-center justify-content-end",
                                                        ),
                                                        width="auto",
                                                    ),
                                                ],
                                                className="align-items-center  justify-content-between g-2",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="g-responsibilities",
                                                    config={"displayModeBar": False},
                                                    responsive=True,
                                                    style={"height": "360px", "minHeight": "300px", "width": "100%"},
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            dbc.Card(
                                                                dbc.CardBody(
                                                                    [
                                                                        html.Div(id="kpi-resp-target-value", className="kpi-value"),
                                                                        html.Div("Total Target", className="kpi-sub"),
                                                                    ]
                                                                ),
                                                                className="kpi kpi-blue",
                                                            ),
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            dbc.Card(
                                                                dbc.CardBody(
                                                                    [
                                                                        html.Div(id="kpi-resp-delivered-value", className="kpi-value"),
                                                                        html.Div("Total Delivered", className="kpi-sub"),
                                                                    ]
                                                                ),
                                                                className="kpi kpi-red",
                                                            ),
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            dbc.Card(
                                                                dbc.CardBody(
                                                                    [
                                                                        html.Div(id="kpi-resp-ach-value", className="kpi-value"),
                                                                        html.Div("Overall Achievement", className="kpi-sub"),
                                                                    ]
                                                                ),
                                                                className="kpi kpi-green",
                                                            ),
                                                            md=4,
                                                        ),
                                                    ],
                                                    className="g-2 mt-1 kpi-row-compact",
                                                ),
                                            ],
                                            className="d-flex flex-column",
                                        ),
                                    ],
                                    className="viz-card shadow-soft section-gap-top flex-fill w-100 responsibilities-card",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Div("Monthly Plan (Stringing)", className="section-title"),
                                                                html.Div(
                                                                    [
                                                                        "Target vs Delivered ",
                                                                        html.Span(
                                                                            "(All periods)",
                                                                            id="label-stringing-plan-period",
                                                                        ),
                                                                    ],
                                                                    className="section-sub",
                                                                ),
                                                            ]
                                                        ),
                                                        className="d-flex flex-column justify-content-center",
                                                        lg=7, md=7, sm=12,
                                                    ),
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                dbc.RadioItems(
                                                                    id="f-stringing-plan-entity",
                                                                    options=[
                                                                        {"label": "Gangs", "value": "Gang"},
                                                                        {"label": "Section Incharges", "value": "Section Incharge"},
                                                                        {"label": "Supervisors", "value": "Supervisor"},
                                                                    ],
                                                                    value="Supervisor",
                                                                    inline=True,
                                                                    class_name="segment segment-xxs",
                                                                    label_class_name="segment-label",
                                                                    label_checked_class_name="segment-label--active",
                                                                    input_class_name="segment-input",
                                                                ),
                                                                dbc.RadioItems(
                                                                    id="f-stringing-plan-metric",
                                                                    options=[
                                                                        {"label": "Tower Weight", "value": "tower_weight"},
                                                                        {"label": "Revenue", "value": "revenue"},
                                                                    ],
                                                                    value="tower_weight",
                                                                    inline=True,
                                                                    class_name="segment segment-xxs",
                                                                    label_class_name="segment-label",
                                                                    label_checked_class_name="segment-label--active",
                                                                    input_class_name="segment-input",
                                                                ),
                                                            ],
                                                            className="header-pills d-flex flex-row align-items-center justify-content-end",
                                                        ),
                                                        width="auto",
                                                    ),
                                                ],
                                                className="align-items-center  justify-content-between g-2",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="g-stringing-plan",
                                                    config={"displayModeBar": False},
                                                    responsive=True,
                                                    style={"height": "360px", "minHeight": "300px", "width": "100%"},
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            dbc.Card(
                                                                dbc.CardBody(
                                                                    [
                                                                        html.Div(id="kpi-stringing-plan-target", className="kpi-value"),
                                                                        html.Div("Total Target", className="kpi-sub"),
                                                                    ]
                                                                ),
                                                                className="kpi kpi-blue",
                                                            ),
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            dbc.Card(
                                                                dbc.CardBody(
                                                                    [
                                                                        html.Div(id="kpi-stringing-plan-delivered", className="kpi-value"),
                                                                        html.Div("Total Delivered", className="kpi-sub"),
                                                                    ]
                                                                ),
                                                                className="kpi kpi-red",
                                                            ),
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            dbc.Card(
                                                                dbc.CardBody(
                                                                    [
                                                                        html.Div(id="kpi-stringing-plan-ach", className="kpi-value"),
                                                                        html.Div("Overall Achievement", className="kpi-sub"),
                                                                    ]
                                                                ),
                                                                className="kpi kpi-green",
                                                            ),
                                                            md=4,
                                                        ),
                                                    ],
                                                    className="g-2 mt-1 kpi-row-compact",
                                                ),
                                            ],
                                            className="d-flex flex-column",
                                        ),
                                    ],
                                    className="viz-card shadow-soft section-gap-top flex-fill w-100 responsibilities-card",
                                ),
                            ],
                            className="d-flex flex-column gap-3 w-100",
                        ),
                        md=6,
                        className="d-flex",
                    ),
                ],
                className="mb-4",
                align="stretch",
                style={"display": "none"},
            ),
            dbc.Row(
                [
                    # LEFT: Actual vs Potential (Figma-style list with scroll)
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        className="section-header",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Gang Performance",
                                                        className="section-title",
                                                    ),
                                                    html.Div(
                                                        [
                                                            "Delivered vs Lost ",
                                                            html.Span(
                                                                "(All periods)",
                                                                id="label-gang-period",
                                                            ),
                                                        ],
                                                        className="section-sub",
                                                    ),
                                                ],
                                                className="d-flex flex-column gap-1",
                                            ),
                                            html.Div(
                                                className="legend",
                                                children=[
                                                    html.Div([html.Span(className="legend__dot dot--delivered"), "Delivered Output"], className="legend__item"),
                                                    html.Div([html.Span(className="legend__dot dot--lost"), "Lost Potential"], className="legend__item"),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Hr(style={"borderColor": "var(--border)", "margin": "8px 0 10px"}),
                                    html.Div(id="avp-list", className="avp-wrap"),
                                    dcc.Graph(
                                        id="g-actual-vs-bench",
                                        config=CLICK_GRAPH_CONFIG,
                                        style={"display": "none"},
                                    ),
                                ]
                            ),
                            className="same-h viz-card shadow-sm",
                        ),
                        md=6,
                    ),


                    # RIGHT: merged Top/Bottom inside one card
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div(
                                                    children=[
                                                        html.Div(
                                                            "Performance Rankings",
                                                            className="section-title",
                                                        ),
                                                        html.Div(
                                                            [
                                                                "Top and bottom performing gangs ",
                                                                html.Span(
                                                                    "(All periods)",
                                                                    id="label-perf-period",
                                                                ),
                                                            ],
                                                            className="section-sub",
                                                        ),
                                                    ]
                                                ),
                                                align="center",
                                            ),
                                            dbc.Col(
                                                dbc.RadioItems(
                                                    id="f-topbot-metric",              # keep same id
                                                    options=[
                                                        {"label": "Productivity", "value": "prod"},
                                                        {"label": "Erection", "value": "erection"},
                                                    ],
                                                    value="prod",
                                                    inline=True,
                                                    class_name="segment",
                                                    label_class_name="segment-label",
                                                    label_checked_class_name="segment-label--active",
                                                    input_class_name="segment-input",
                                                ),
                                                width="auto",
                                                align="center",
                                            ),
                                        ],
                                        justify="between",
                                        align="center",
                                    )
                                ),

                                dbc.CardBody(
                                    [
                                        html.Div("Top 5 Performers", className="text-success fw-semibold mb-2"),
                                        dcc.Graph(id="g-top5", config=CLICK_GRAPH_CONFIG, style={"cursor":"pointer"}),
                                        html.Hr(className="my-3"),
                                        html.Div("Bottom 5 Performers", className="text-danger fw-semibold mb-2"),
                                        dcc.Graph(id="g-bottom5", config=CLICK_GRAPH_CONFIG, style={"cursor":"pointer"}),
                                    ]
                                ),
                            ],
                            className="same-h viz-card shadow-sm",
                        ),
                        md=6,
                    ),
                ],
                className="mb-4",
                style={"display": "none"},
            ),
            build_trace_block(),
            build_erections_card(),
            dcc.Store(id="store-click-meta", data=None),
            dcc.Store(id="store-dblclick", data=None),
            dcc.Store(id="store-selected-gang", data=None),
            dcc.Store(id="store-filtered-scope", data=None),
            dcc.Store(id="store-stringing-scope", data="all"),
            dcc.Store(id="store-global-performance-mode", data="erection"),
            dcc.Store(id="store-global-performance-scope", data=None),
            dcc.Store(id="store-global-performance-click-meta", data=None),
            dcc.Store(id="global-performance-selected-gang", data=None),
            dcc.Store(id="store-pch-modal-focus", data=None),
            html.Div(id="scroll-wire", style={"display": "none"}),   # <- add this
            trace_modal,
            global_performance_modal,
            build_project_responsibilities_modal(),
            pch_modal,
            project_modal,
            dcc.Store(id="store-kpi-selected-project", data=None),
            dcc.Store(id="store-proj-resp-code", data=None),
            dcc.Store(id="store-proj-resp-month", data=None),
            dcc.Store(id="store-proj-resp-plan", data=None),
            dcc.Store(id="store-project-tile-focus", data=None),
            dcc.Store(id="store-project-surface-focus", data=None),
            dcc.Store(id="store-project-modal-focus-cache", data=None),
            dcc.Store(id="store-project-modal-section", data="erections"),
            dcc.Store(id="store-project-modal-click-meta", data=None),
            dcc.Store(id="store-project-modal-scope", data=None),
            dcc.Store(id="project-modal-selected-gang", data=None),
            dcc.Store(id="store-project-modal-performance-mode", data="erection|0"),
            dcc.Store(id="project-modal-scroll-target", data=None),
            dcc.Store(id="store-project-tile-meta", data={}),
            dcc.Store(id="store-project-modal-history", data=None),
            html.Div(id="project-modal-history-wire", style={"display": "none"}),
            html.Button(
                id={
                    "type": "project-tile-trigger",
                    "mode": "placeholder",
                    "project": "__placeholder__",
                    "context": "placeholder",
                },
                n_clicks=0,
                type="button",
                style={"display": "none"},
            ),
        ],
        className="dashboard-tab-content",
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(executive_layout, label="Executive Overview", tab_id="executive-overview"),
            dbc.Tab(project_overview_layout, label="Project Overview", tab_id="project-overview"),
            dbc.Tab(analytics_layout, label="Tower Erection Analytics", tab_id="analytics"),
            dbc.Tab(stringing_analytics_layout, label="Stringing Analytics", tab_id="stringing-analytics"),
        ],
        id="main-tabs",
        active_tab="executive-overview",
        className="main-tabs",
    )

    layout = dbc.Container(
        [
            dcc.Location(id="project-modal-location", refresh=False),
            build_header("Productivity Dashboard", last_updated_text),
            dcc.Download(id="download-executive-pdf"),
            tabs,
            html.Div(
                home_content.children,
                id="legacy-dashboard-mount",
                className="dashboard-tab-content",
                style={"display": "none"},
            ),
        ],
        fluid=True,
    )
    return layout







