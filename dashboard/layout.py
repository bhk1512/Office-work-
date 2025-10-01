"""Dash layout composition."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dcc import Download


def build_controls(default_benchmark: float) -> dbc.Card:
    """Return the filter controls card."""

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Filters", className="fw-bold mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="f-project",
                                multi=True,
                                placeholder="Select project(s)",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="f-month",
                                multi=True,
                                placeholder="Select month(s)",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="f-gang",
                                multi=True,
                                placeholder="Select gang(s)",
                            ),
                            md=4,
                        ),
                    ],
                    className="g-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
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
                                className="mt-2",
                            ),
                            md=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                options=[
                                    {
                                        "label": " Overall months (ignore month filter)",
                                        "value": "all_months",
                                    }
                                ],
                                value=[],
                                id="f-overall-months",
                                switch=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                options=[
                                    {
                                        "label": " Overall gangs (ignore gang filter)",
                                        "value": "all_gangs",
                                    }
                                ],
                                value=[],
                                id="f-overall-gangs",
                                switch=True,
                            ),
                            md=4,
                        ),
                    ],
                    className="mt-2",
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    "Benchmark (MT / gang / day)",
                                    className="small text-muted",
                                ),
                                dcc.Input(
                                    id="bench",
                                    type="number",
                                    value=default_benchmark,
                                    step=0.1,
                                ),
                            ],
                            md=3,
                        ),
                    ]
                ),
            ]
        ),
        className="mb-3 shadow-sm",
    )


def build_kpi_cards() -> dbc.Row:
    """Return the KPI cards row."""

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Avg Output / Gang / Day",
                                className="fw-bold text-white-50",
                            ),
                            html.H2(id="kpi-avg", className="mb-0 text-white"),
                            html.Div(id="kpi-delta", className="small text-white-50"),
                        ]
                    ),
                    className="shadow-sm",
                    style={
                        "backgroundColor": "#4f9cff",
                        "border": "0",
                        "borderRadius": "12px",
                    },
                ),
                xs=12,
                sm=6,
                md=6,
                lg=3,
                xl=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Benchmark Target",
                                className="fw-bold text-white-50",
                            ),
                            html.H2(id="kpi-bench", className="mb-0 text-white"),
                        ]
                    ),
                    className="shadow-sm",
                    style={
                        "backgroundColor": "#9FE2BF",
                        "border": "0",
                        "borderRadius": "12px",
                    },
                ),
                xs=12,
                sm=6,
                md=6,
                lg=3,
                xl=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Active Gangs",
                                className="fw-bold text-white-50",
                            ),
                            html.H2(id="kpi-active", className="mb-0 text-white"),
                        ]
                    ),
                    className="shadow-sm",
                    style={
                        "backgroundColor": "#d6b4fc",
                        "border": "0",
                        "borderRadius": "12px",
                    },
                ),
                xs=12,
                sm=6,
                md=6,
                lg=3,
                xl=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Total Erection (This Period)",
                                className="fw-bold text-white-50",
                            ),
                            html.H2(id="kpi-total", className="mb-0 text-white"),
                        ]
                    ),
                    className="shadow-sm",
                    style={
                        "backgroundColor": "#ffbb66",
                        "border": "0",
                        "borderRadius": "12px",
                    },
                ),
                xs=12,
                sm=6,
                md=6,
                lg=3,
                xl=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Lost Units",
                                className="fw-bold text-white-50",
                            ),
                            html.H2(id="kpi-loss", className="mb-0 text-white"),
                        ]
                    ),
                    className="shadow-sm",
                    style={
                        "backgroundColor": "#ff7b7b",
                        "border": "0",
                        "borderRadius": "12px",
                    },
                ),
                xs=12,
                sm=6,
                md=6,
                lg=3,
                xl=2,
            ),
        ],
        className="g-3 align-items-stretch",
    )


ROW_PX = 56
VISIBLE_ROWS = 15
TOPBOT_MARGIN = 120
CONTAINER_HEIGHT = ROW_PX * VISIBLE_ROWS + TOPBOT_MARGIN


def build_trace_block() -> dbc.Card:
    """Return the traceability card with tables and export controls."""

    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Pick a gang (overrides Gang filter)",
                                    className="fw-semibold mb-1",
                                ),
                                dcc.Dropdown(
                                    id="trace-gang",
                                    options=[],
                                    value=None,
                                    placeholder="Start typing a gang...",
                                    clearable=True,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.H5("Traceability"), md=8),
                        dbc.Col(
                            dbc.Button(
                                "Export Trace Excel",
                                id="btn-export-trace",
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
                                    id="tbl-idle-intervals",
                                    columns=[
                                        {"name": "Gang", "id": "gang_name"},
                                        {
                                            "name": "Interval Start",
                                            "id": "interval_start",
                                        },
                                        {
                                            "name": "Interval End",
                                            "id": "interval_end",
                                        },
                                        {
                                            "name": "Raw Gap (days)",
                                            "id": "raw_gap_days",
                                        },
                                        {
                                            "name": "Idle Counted (days)",
                                            "id": "idle_days_capped",
                                        },
                                    ],
                                    data=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
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
                                    id="tbl-daily-prod",
                                    columns=[
                                        {"name": "Date", "id": "date"},
                                        {"name": "Gang", "id": "gang_name"},
                                        {
                                            "name": "Project",
                                            "id": "project_name",
                                        },
                                        {"name": "MT/day", "id": "daily_prod_mt"},
                                    ],
                                    data=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "Inter, system-ui",
                                        "fontSize": 13,
                                    },
                                ),
                            ],
                            md=6,
                        ),
                    ]
                ),
                Download(id="download-trace-xlsx"),
                dcc.Store(id="store-selected-gang"),
            ]
        ),
        className="mt-4 shadow-sm",
    )


def build_layout(last_updated_text: str, default_benchmark: float) -> dbc.Container:
    """Assemble the full Dash layout."""

    controls = build_controls(default_benchmark)
    row_height = f"{CONTAINER_HEIGHT}px"
    gang_bar = html.Div(
        dcc.Graph(
            id="g-actual-vs-bench",
            config={"displayModeBar": False, "doubleClick": "false"},
        ),
        style={"height": row_height, "overflowY": "auto"},
    )
    modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
            dbc.ModalBody(id="modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="modal-close", className="ms-auto", n_clicks=0)
            ),
        ],
        id="loss-modal",
        is_open=False,
    )
    layout = dbc.Container(
        [
            html.Div(
                f"Last Updated On: {last_updated_text}",
                className="text-muted fw-semibold",
            ),
            html.H2(
                "Measure Output. Expose Lost Units.",
                className="mt-3 fw-bold",
            ),
            html.Div("Tag causes; assign fixes.", className="text-muted mb-3"),
            controls,
            build_kpi_cards(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Projects over Months"),
                            dcc.Graph(
                                id="g-project-lines",
                                config={"displayModeBar": False},
                            ),
                            html.H5("Average Productivity per Month"),
                            dcc.Graph(
                                id="g-monthly",
                                config={"displayModeBar": False},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H5(
                                "Actual vs Potential Performance (All Gangs)"
                            ),
                            gang_bar,
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Top 5 Gangs"),
                            dcc.Graph(
                                id="g-top5",
                                config={
                                    "displayModeBar": False,
                                    "doubleClick": "false",
                                },
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Bottom 5 Gangs"),
                            dcc.Graph(
                                id="g-bottom5",
                                config={
                                    "displayModeBar": False,
                                    "doubleClick": "false",
                                },
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            build_trace_block(),
            dcc.Store(id="store-click-meta", data=None),
            dcc.Store(id="store-dblclick", data=None),
            modal,
        ],
        fluid=True,
    )
    return layout
