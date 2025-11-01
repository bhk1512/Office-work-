"""Dash layout composition."""
from __future__ import annotations

from datetime import datetime, timedelta

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dcc import Download
import urllib.parse

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

CURRENT_MONTH_VALUE = datetime.today().strftime("%Y-%m")
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
                                {"name": "Gang", "id": "gang_name"},
                                {"name": "Project", "id": "project_name"},
                                {"name": "Date", "id": "date"},
                                {"name": "MT/day", "id": "daily_prod_mt"},
                            ],
                            data=[],
                            page_size=10,
                            style_table={"overflowX": "auto"},
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


def build_controls() -> dbc.Card:
    """Filter controls: Projects → Gangs → Months; quick range under Months; Reset left."""
    return dbc.Card(
        dbc.CardBody(
            [
                # Row 1: Projects, Gangs, Months
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="f-project",
                                        multi=True,
                                        placeholder="Select project(s)",
                                        className="filter-select",
                                    ),
                                ],
                                className="filter-field",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="f-gang",
                                        multi=True,
                                        placeholder="Select gang(s)",
                                        className="filter-select",
                                    ),
                                ],
                                className="filter-field",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="f-month",
                                        multi=True,
                                        placeholder="Select month(s)",
                                        className="filter-select",
                                        value=[CURRENT_MONTH_VALUE],
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                className="filter-field",
                            ),
                            md=4,
                        ),
                    ],
                    className="g-3",
                ),

                # Row 2: Reset (left) + Quick range under Months (right)
                dbc.Row(
                    [
                        # Left: Reset button
                        dbc.Col(
                            dbc.Button(
                                "Reset Filters",
                                id="btn-reset-filters",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="filter-reset-btn",
                            ),
                            md=4,
                            className="d-flex align-items-center",
                        ),

                        # Middle spacer (keeps radios visually under the Months column)
                        dbc.Col(md=4),

                        # Right: Quick-range radios + Clear link, under Months
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
                                        className="filter-quick-items",
                                    ),
                                    html.A(
                                        "Clear Quick Range",
                                        id="link-clear-quick-range",
                                        n_clicks=0,
                                        className="filter-clear-link",
                                        style={"cursor": "pointer"},
                                    ),
                                ],
                                className="filter-quick-under-months",
                            ),
                            md=4,
                        ),
                    ],
                    className="g-2 mt-1",
                ),
                                # Row 3: Stringing-only filters (Line kV + Method)
                html.Div(
                    [
                        dbc.Row(
                            [
                                # Left: Line kV chips
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div("Line (kV)", className="filter-label mb-1"),
                                            dbc.Checklist(
                                                id="f-kv",
                                                options=[
                                                    {"label": "400 kV", "value": "400"},
                                                    {"label": "765 kV", "value": "765"},
                                                ],
                                                value=["400", "765"],  # default: overall (both)
                                                inline=True,
                                            ),
                                        ]
                                    ),
                                    md=6,
                                ),
                                # Right: Method chips
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div("Method", className="filter-label mb-1"),
                                            dbc.Checklist(
                                                id="f-method",
                                                options=[
                                                    {"label": "Manual", "value": "manual"},
                                                    {"label": "TSE", "value": "tse"},
                                                ],
                                                value=["manual", "tse"],  # default: overall (both)
                                                inline=True,
                                            ),
                                        ]
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-2",
                        )
                    ],
                    id="stringing-filters-wrap",
                    style={"display": "none"},  # shown only in stringing mode by callback
                ),
            ]
        ),
        className="mb-3 shadow-sm filter-card",
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
                xs=12, sm=6, md=6, lg=3, xl=3,
            ),

            # 2) Active Projects (purple)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(_icon("users"), className="kpi__icon"),
                            html.Div("Active Gangs", className="kpi-label"),
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

            # 3) Total Erection (orange for Figma look)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(_icon("trend_up"), className="kpi__icon"),
                            html.Div(id="label-total", children="Total Erection", className="kpi-label"),
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
                xs=12, sm=6, md=6, lg=3, xl=3,
            ),
        ],
        className="g-3 align-items-stretch",
    )


def build_project_details_card() -> dbc.Card:
    """Project Overview card: the body is dynamic (message OR 3-col grid)."""
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



def build_erections_card() -> dbc.Card:
    """Standalone card that lists completed erections for the selected filters."""

    controls = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Div("Erections Completed", className="section-title mb-2", id="lbl-erections-title"),
                        html.Div(
                            "Completion date (defaults to yesterday)",
                            className="fw-semibold mb-1",
                        ),
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
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
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


def build_header(title: str, last_updated_text: str) -> html.Div:
    """Top section: icon + big title, and 'Last Updated On' line under it."""

    # Build small inline SVGs as IMG data URIs (Dash-safe across versions)
    cube_svg_str = '''
<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M12 2L20 6.5V17.5L12 22L4 17.5V6.5L12 2Z" stroke="white" stroke-width="1.6"/>
  <path d="M12 2V12L20 17.5" stroke="white" stroke-width="1.6"/>
  <path d="M12 12L4 17.5" stroke="white" stroke-width="1.6"/>
</svg>
'''.strip()
    cube_img = html.Img(
        src="data:image/svg+xml;utf8," + urllib.parse.quote(cube_svg_str),
        style={"width": "22px", "height": "22px"},
    )

    calendar_svg_str = '''
<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="3" y="4" width="18" height="17" rx="2" stroke="#64748B" stroke-width="1.5"/>
  <path d="M8 2V6M16 2V6" stroke="#64748B" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M3 9H21" stroke="#64748B" stroke-width="1.5"/>
</svg>
'''.strip()
    calendar_img = html.Img(
        src="data:image/svg+xml;utf8," + urllib.parse.quote(calendar_svg_str),
        style={"width": "16px", "height": "16px", "marginRight": "8px"},
    )

    # Mode banner + toggle (top-right)
    mode_controls = html.Div(
        [
            html.Span(
                "Erection mode",
                id="mode-banner",
                style={
                    "fontSize": "12px",
                    "color": "#64748B",
                    "background": "#F1F5F9",
                    "padding": "4px 8px",
                    "borderRadius": "8px",
                },
            ),
            dcc.RadioItems(
                id="mode-toggle",
                options=[
                    {"label": "Erection", "value": "erection"},
                    {"label": "Stringing", "value": "stringing"},
                ],
                value="erection",
                inline=True,
                style={"fontSize": "12px"},
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "inline-flex", "gap": "0", "marginRight": "10px"},
            ),
        ],
        className="topbar__mode",
        style={"marginLeft": "auto", "display": "flex", "gap": "10px", "alignItems": "center"},
    )

    return html.Div(
        [
            html.Div(  # left icon badge
                html.Div(cube_img, className="brand-badge"),
                className="topbar__icon"
            ),
            html.Div(  # right text block
                [
                    html.Div(title, className="topbar__title"),
                    html.Div(
                        [calendar_img, html.Span(f"Last Updated On: {last_updated_text}")],
                        className="topbar__meta",
                    ),
                ],
                className="topbar__text",
            ),
            mode_controls,
        ],
        className="topbar",
    )



def build_layout(last_updated_text: str) -> dbc.Container:
    """Assemble the full Dash layout."""

    controls = build_controls()
    
    trace_modal = build_trace_modal()
    layout = dbc.Container(
        [
            build_header("Productivity Dashboard", last_updated_text),
            
            controls,
            build_project_details_card(),     # <-- INSERT HERE
            build_kpi_cards(),
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
                    # RIGHT: Responsibilities (Figma-styled card with KPIs)
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            # Left: Title + subtitle
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div("Responsibilities", className="section-title"),
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
                                                lg=7, md=7, sm=12,  # reserve space so pills fit on the right
                                            ),

                                            # Right: BOTH pill groups on same row as the title
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            className="segment segment-xxs",
                                                            children=[
                                                                dcc.RadioItems(
                                                                    id="f-resp-entity",
                                                                    options=[
                                                                        {"label": "Gangs", "value": "Gang"},
                                                                        {"label": "Section Incharges", "value": "Section Incharge"},
                                                                        {"label": "Supervisors", "value": "Supervisor"},
                                                                    ],
                                                                    value="Supervisor",
                                                                    labelStyle={"display": "inline-flex", "gap": "0"},
                                                                    inputStyle={"marginRight": "6px"},
                                                                )
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="segment segment-xxs",
                                                            children=[
                                                                dcc.RadioItems(
                                                                    id="f-resp-metric",
                                                                    options=[
                                                                        {"label": "Tower Weight", "value": "tower_weight"},
                                                                        {"label": "Revenue", "value": "revenue"},
                                                                    ],
                                                                    value="tower_weight",
                                                                    labelStyle={"display": "inline-flex", "gap": "0"},
                                                                    inputStyle={"marginRight": "6px"},
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                    className="header-pills d-flex flex-row align-items-center justify-content-end"
                                                ),
                                                 width="auto",
                                            ),
                                        ],
                                        # KEY: don't allow wrapping at desktop widths
                                        className="align-items-center  justify-content-between g-2",
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="g-responsibilities",
                                            config={"displayModeBar": False},
                                            style={"height": "360px"},
                                        ),

                                        # KPI row
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
                                                    md=4
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
                                                    md=4
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
                                                    md=4
                                                ),
                                            ],
                                            className="g-2 mt-1 kpi-row-compact",
                                        ),
                                    ],
                                    className="d-flex flex-column",
                                ),
                            ],
                            className="viz-card shadow-soft section-gap-top flex-fill w-100",  # matches other cards
                        ),
                        md=6,
                        className="d-flex",
                    ),
                ],
                className="mb-4",
                align="stretch",
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
                                    # Keep the original figure hidden so existing clientside callbacks keep working
                                    # dcc.Graph(id="g-actual-vs-bench", config=CLICK_GRAPH_CONFIG, style={"display": "none"}),
                                    dcc.Graph(
                                        id="g-actual-vs-bench",
                                        config=CLICK_GRAPH_CONFIG,
                                        style={
                                            "display": "none",
                                        },
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
                                                html.Div(
                                                    className="segment",
                                                    children=[
                                                        dcc.RadioItems(
                                                            id="f-topbot-metric",              # keep same id
                                                            options=[
                                                                {"label": "Productivity", "value": "prod"},
                                                                {"label": "Erection", "value": "erection"},
                                                            ],
                                                            value="prod",
                                                            labelStyle={"display": "inline-flex", "gap": "0"},
                                                            inputStyle={"marginRight": "8px"},
                                                        )
                                                    ],
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
            ),
            build_trace_block(),
            build_erections_card(),
            dcc.Store(id="store-click-meta", data=None),
            dcc.Store(id="store-dblclick", data=None),
            dcc.Store(id="store-selected-gang", data=None),   
            dcc.Store(id="store-mode", data="erection"),
            html.Div(id="mode-data-debug", style={"display": "none"}),
            html.Div(id="scroll-wire", style={"display": "none"}),   # <- add this
            trace_modal,
        ],
        fluid=True,
    )
    return layout




