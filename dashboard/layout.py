"""Dash layout composition."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dcc import Download
import urllib.parse
from dash import html

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
                            id=daily_table_id,
                            columns=[
                                {"name": "Date", "id": "date"},
                                {"name": "Gang", "id": "gang_name"},
                                {"name": "Project", "id": "project_name"},
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
    ]


def build_controls() -> dbc.Card:
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
                                multi=False,
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
                            # Left: radios + Clear link on one line
                            dbc.Col(
                                html.Div(
                                    [
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
                                            className="mb-0",        # tighter
                                        ),
                                        dbc.Button(
                                            "Clear Quick Range",
                                            id="btn-clear-quick-range",
                                            color="link",
                                            size="sm",
                                            className="p-0 ms-2",    # looks like a link, small gap
                                        ),
                                    ],
                                    className="d-flex align-items-center flex-wrap mt-2 gap-2",
                                ),
                                md=10,
                            ),

                            # Right: Reset button, top-aligned
                            dbc.Col(
                                dbc.Button(
                                    "Reset Filters",
                                    id="btn-reset-filters",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="",               # no extra top margin
                                ),
                                md=2,
                                className="d-flex justify-content-end align-items-start mt-2",
                            ),
                        ],
                        className="g-2",
                    ),
            ]
        ),
        className="mb-3 shadow-sm",
    )


def _svg(icon_path: str, size: int = 18):
    """Small inline SVG helper (white icons)."""
    return html.Span(
        dangerously_allow_html=True,
        children=f'''
<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="{icon_path}" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
''',
    )

# Lucide paths (top-right icons)
_LUCIDE_TREND_DOWN = "7 7 17 17M17 7h0v10H7"
_LUCIDE_USERS      = "17 21v-2a4 4 0 0 0-4-4H11a4 4 0 0 0-4 4v2M7 7a4 4 0 1 0 8 0 4 4 0 0 0-8 0"
_LUCIDE_TREND_UP   = "7 17 17 7M7 7h10v10"
_LUCIDE_ACTIVITY   = "22 12h-4l-3 9-6-18-3 9H2"

def build_kpi_cards() -> dbc.Row:
    return dbc.Row(
        [
            # 1) Avg Output / Gang / Day  (blue)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(_icon("trend_down"), className="kpi__icon"),
                            html.Div("Avg Output / Gang / Day", className="kpi-label"),
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
                xs=12, sm=6, md=6, lg=3, xl=3,
            ),

            # 2) Active Projects (purple)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(_icon("users"), className="kpi__icon"),
                            html.Div("Active Projects", className="kpi-label"),
                            html.Div(
                                [ html.Span(id="kpi-active", className="kpi-value") ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    className="kpi kpi--purple",
                ),
                xs=12, sm=6, md=6, lg=3, xl=3,
            ),

            # 3) Total Erection (This Period) (orange for Figma look)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(_icon("trend_up"), className="kpi__icon"),
                            html.Div("Total Erection (This Period)", className="kpi-label"),
                            html.Div(
                                [ html.Span(id="kpi-total", className="kpi-value") ],
                                className="kpi-row",
                            ),
                        ]
                    ),
                    className="kpi kpi--green",
                ),
                xs=12, sm=6, md=6, lg=3, xl=3,
            ),

            # 4) Lost Units (red)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(_icon("trend_down"), className="kpi__icon"),
                            html.Div("Lost Units", className="kpi-label"),
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
                xs=12, sm=6, md=6, lg=3, xl=3,
            ),
        ],
        className="g-3 align-items-stretch",
    )


def build_project_details_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                [
                    html.Div("Project Details", className="project-details__title"),
                    # small neutral icon (purely visual; no functionality)
                    html.Span("ℹ️", style={"opacity": 0.6, "fontSize": "0.95rem"}),
                ],
                className="project-details mb-2",
            ),
            html.Hr(className="my-2", style={"borderColor": "var(--border)"}),
            html.Div(
                id="project-details",
                className="project-details__body",
                children="Select a single project to view its details.",
            ),
        ]),
        className="mb-3 card--elev",
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
    return dbc.Card(
        dbc.CardBody(contents),
        className="mt-4 shadow-sm",
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


def build_layout(last_updated_text: str) -> dbc.Container:
    """Assemble the full Dash layout."""

    controls = build_controls()
    row_height = f"{CONTAINER_HEIGHT}px"
    gang_bar = html.Div(
        dcc.Graph(
            id="g-actual-vs-bench",
            config=CLICK_GRAPH_CONFIG,
        ),
        style={"height": row_height, "overflowY": "auto"},
    )
    trace_modal = build_trace_modal()
    layout = dbc.Container(
        [
            html.H2(
                f"Last Updated On: {last_updated_text}",
                className="text-muted fw-semibold",
            ),
            
            controls,
            build_project_details_card(),     # <-- INSERT HERE
            build_kpi_cards(),
            dbc.Row(
                [
                    # LEFT: Projects over Months (only)
                    dbc.Col(
                        [
                            html.H5("Projects over Months"),
                            dcc.Graph(
                                id="g-project-lines",
                                config={"displayModeBar": False},
                            ),
                        ],
                        md=6,
                    ),
                    # RIGHT: Responsibilities (moved here)
                    dbc.Col(
                        [
                            html.H5("Responsibilities"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.RadioItems(
                                            id="f-resp-entity",
                                            options=[
                                                {"label": "Gangs", "value": "Gang"},
                                                {"label": "Section Incharges", "value": "Section Incharge"},
                                                {"label": "Supervisors", "value": "Supervisor"},
                                            ],
                                            value="Supervisor",
                                            inputStyle={"marginRight": "6px"},
                                            labelStyle={"marginRight": "18px"},
                                            inline=True,
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dcc.RadioItems(
                                            id="f-resp-metric",
                                            options=[
                                                {"label": "Tower Weight", "value": "tower_weight"},
                                                {"label": "Revenue", "value": "revenue"},
                                            ],
                                            value="tower_weight",
                                            inputStyle={"marginRight": "6px"},
                                            labelStyle={"marginRight": "18px"},
                                            inline=True,
                                        ),
                                        width="auto",
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dcc.Graph(id="g-responsibilities", config={"displayModeBar": False}),
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    # LEFT: Actual vs Potential (moved down here)
                    dbc.Col(
                        [
                            html.H5("Actual vs Potential Performance (All Gangs)"),
                            html.Div(
                                dcc.Graph(id="g-actual-vs-bench", config=CLICK_GRAPH_CONFIG),
                                style={"height": row_height, "overflowY": "auto"},
                            ),
                        ],
                        md=6,
                    ),

                    # RIGHT: merged Top/Bottom inside one card
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            dbc.Col(html.Span("Performance Rankings"), align="center"),
                                            dbc.Col(
                                                dcc.RadioItems(
                                                    id="f-topbot-metric",
                                                    options=[
                                                        {"label": "Productivity (MT/day)", "value": "prod"},
                                                        {"label": "Erection (MT)", "value": "erection"},
                                                    ],
                                                    value="prod",
                                                    inline=True,
                                                    inputStyle={"marginRight": "6px"},
                                                    labelStyle={"marginRight": "18px"},
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        justify="between",
                                        align="center",
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div("Top 5 Performers", className="text-success fw-semibold mb-2"),
                                        dcc.Graph(id="g-top5", config=CLICK_GRAPH_CONFIG),
                                        html.Hr(className="my-3"),
                                        html.Div("Bottom 5 Performers", className="text-danger fw-semibold mb-2"),
                                        dcc.Graph(id="g-bottom5", config=CLICK_GRAPH_CONFIG),
                                    ]
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            build_trace_block(),
            dcc.Store(id="store-click-meta", data=None),
            dcc.Store(id="store-dblclick", data=None),
            dcc.Store(id="store-selected-gang", data=None),   
            html.Div(id="scroll-wire", style={"display": "none"}),   # <- add this
            trace_modal,
        ],
        fluid=True,
    )
    return layout



