"""Layout for the Analytics tab."""
from __future__ import annotations

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


GRAPH_CONFIG = {
    "displayModeBar": False,
    "doubleClick": False,
    "scrollZoom": False,
}


def _audit_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="analytics-audit-table",
        columns=[],
        data=[],
        page_size=12,
        fixed_rows={"headers": True},
        style_table={"overflowX": "auto", "maxHeight": "440px"},
        style_cell={
            "fontFamily": "Inter, system-ui",
            "fontSize": 13,
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


def build_analytics_layout() -> html.Div:
    """Return the Analytics tab content."""
    audit_drawer = dbc.Offcanvas(
        [
            dbc.Tabs(
                [
                    dbc.Tab(_audit_table(), tab_id="audit-table", label="Audit Table"),
                    dbc.Tab(
                        html.Div(
                            id="analytics-audit-definition",
                            className="analytics-audit-definition",
                        ),
                        tab_id="audit-definition",
                        label="Definition",
                    ),
                    dbc.Tab(
                        html.Div(
                            [
                                html.Div(
                                    "Export the current audit table to Excel.",
                                    className="text-muted mb-3",
                                ),
                                dbc.Button(
                                    "Export Audit Excel",
                                    id="analytics-audit-export-btn",
                                    color="primary",
                                    className="analytics-export-btn",
                                ),
                            ],
                            className="analytics-audit-export",
                        ),
                        tab_id="audit-export",
                        label="Export",
                    ),
                ],
                id="analytics-audit-tabs",
                active_tab="audit-table",
            ),
            dbc.Button(
                "Close",
                id="analytics-audit-close",
                outline=True,
                color="secondary",
                size="sm",
                className="mt-3",
            ),
        ],
        id="analytics-audit-drawer",
        title=html.Div(id="analytics-audit-title"),
        placement="end",
        is_open=False,
        scrollable=True,
        className="analytics-audit-drawer",
    )

    scope_strip = html.Div(
        [
            html.Span(id="analytics-scope-range", className="analytics-scope-chip"),
            html.Span(id="analytics-scope-projects", className="analytics-scope-chip"),
            html.Span(id="analytics-scope-gangs", className="analytics-scope-chip"),
            html.Span(id="analytics-scope-activedays", className="analytics-scope-chip"),
            html.Span("Idle cap: 15 days/window", className="analytics-scope-chip"),
        ],
        className="analytics-scope-strip analytics-scope-strip--inline",
    )

    kpi_cards = dbc.Row(
        [
            dbc.Col(
                html.Button(
                    [
                        html.Div(
                            "Low-productivity active-day share (0-4 MT/day bucket)",
                            className="analytics-kpi-title",
                        ),
                        html.Div(id="analytics-kpi-low-output-value", className="analytics-kpi-value"),
                        html.Div(id="analytics-kpi-low-output-sub", className="analytics-kpi-sub"),
                    ],
                    id="analytics-kpi-low-output",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=3,
            ),
            dbc.Col(
                html.Button(
                    [
                        html.Div(
                            "Idle windows per gang (High vs Low)",
                            className="analytics-kpi-title",
                        ),
                        html.Div(id="analytics-kpi-idle-value", className="analytics-kpi-value"),
                        html.Div(id="analytics-kpi-idle-sub", className="analytics-kpi-sub"),
                    ],
                    id="analytics-kpi-idle-windows",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=3,
            ),
            dbc.Col(
                html.Button(
                    [
                        html.Div("Hotspot by project", className="analytics-kpi-title"),
                        html.Div(id="analytics-kpi-hotspot-value", className="analytics-kpi-value"),
                        html.Div(id="analytics-kpi-hotspot-sub", className="analytics-kpi-sub"),
                    ],
                    id="analytics-kpi-hotspot",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=3,
            ),
            dbc.Col(
                html.Button(
                    [
                        html.Div("Recoverable output (estimate)", className="analytics-kpi-title"),
                        html.Div(id="analytics-kpi-recovery-value", className="analytics-kpi-value"),
                        html.Div(id="analytics-kpi-recovery-sub", className="analytics-kpi-sub"),
                    ],
                    id="analytics-kpi-recovery",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=3,
            ),
        ],
        className="g-3 mb-4",
    )

    whatif_card = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        "What-if: Improve bucket productivity",
                        className="section-title",
                    ),
                    html.Div(
                        "Assumption: output constant; uplift applies only to selected bucket",
                        className="section-sub",
                    ),
                ]
            ),
            dbc.CardBody(
                [
                dcc.Dropdown(
                    id="analytics-whatif-bucket",
                    options=[],
                    value=None,
                    clearable=False,
                    className="analytics-whatif-select",
                ),
                dcc.Slider(
                    id="analytics-whatif-slider",
                    min=2,
                    max=20,
                    step=0.5,
                    value=4,
                    marks={value: str(value) for value in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    id="analytics-whatif-reduction",
                                    className="analytics-whatif-value",
                                ),
                                html.Div(
                                    "Additional output if bucket reaches target",
                                    className="analytics-whatif-label",
                                ),
                            ],
                            className="analytics-whatif-metric",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    id="analytics-whatif-saved",
                                    className="analytics-whatif-value",
                                ),
                                html.Div(
                                    "Deployment slots freed for reallocation",
                                    className="analytics-whatif-label",
                                ),
                            ],
                            className="analytics-whatif-metric",
                        ),
                    ],
                    className="analytics-whatif-metrics",
                ),
                html.Div(
                    dcc.Graph(
                        id="analytics-whatif-chart",
                        config=GRAPH_CONFIG,
                        style={"height": "100%", "minHeight": "180px"},
                    ),
                    className="analytics-whatif-chart-wrap",
                ),
                html.Div(
                    [
                        html.A(
                            "Reset",
                            id="analytics-whatif-reset",
                            n_clicks=0,
                            className="analytics-whatif-reset",
                        ),
                        html.Span(
                            id="analytics-whatif-status",
                            className="analytics-whatif-status",
                        ),
                    ],
                    className="analytics-whatif-footer",
                ),
                ],
                className="analytics-whatif-body",
            ),
        ],
        className="viz-card shadow-soft analytics-whatif-card h-100 w-100",
    )

    row2 = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Div("Active-days vs Output Share", className="section-title"),
                                html.Div("Bucketed by MT/day", className="section-sub"),
                            ]
                        ),
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="analytics-bucket-chart",
                                    config=GRAPH_CONFIG,
                                    style={"height": "240px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Low-productivity active-day share",
                                            className="analytics-lowshare-title",
                                        ),
                                        html.Div(
                                            "% of active days in <4 MT/day bucket",
                                            className="analytics-lowshare-subtitle",
                                        ),
                                        html.Div(
                                            id="analytics-lowshare-scope",
                                            className="analytics-lowshare-scope",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    id="analytics-lowshare-value",
                                                    className="analytics-lowshare-value",
                                                ),
                                                html.Div(
                                                    id="analytics-lowshare-delta",
                                                    className="analytics-lowshare-delta",
                                                ),
                                            ],
                                            className="analytics-lowshare-metrics",
                                        ),
                                        dcc.Graph(
                                            id="analytics-lowshare-chart",
                                            config=GRAPH_CONFIG,
                                            style={"height": "140px"},
                                        ),
                                    ],
                                    className="analytics-lowshare-inline",
                                ),
                            ],
                            className="analytics-bucket-body",
                        ),
                    ],
                    className="viz-card shadow-soft h-100 w-100",
                ),
                md=6,
                className="d-flex",
            ),
            dbc.Col(whatif_card, md=6, className="d-flex"),
        ],
        className="g-3 mb-4 align-items-stretch analytics-row analytics-row-2",
    )

    hotspot_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Div("Idle frequency and magnitude by productivity tier", className="section-title"),
                                html.Div("Low (<4), Mid (4-6), High (>6)", className="section-sub"),
                            ]
                        ),
                        dbc.CardBody(
                            dcc.Graph(
                                id="analytics-tier-chart",
                                config=GRAPH_CONFIG,
                                style={"height": "320px"},
                            )
                        ),
                    ],
                    className="viz-card shadow-soft h-100 w-100",
                ),
                md=6,
                className="d-flex",
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Div("Hotspot Ranking", className="section-title"),
                                html.Div("Top 10 projects by idle-days/100 towers", className="section-sub"),
                            ]
                        ),
                        dbc.CardBody(
                            dcc.Graph(
                                id="analytics-hotspot-chart",
                                config=GRAPH_CONFIG,
                                style={"height": "320px"},
                            )
                        ),
                    ],
                    className="viz-card shadow-soft h-100 w-100",
                ),
                md=6,
                className="d-flex",
            ),
        ],
        className="g-3 mb-4 align-items-stretch analytics-row analytics-row-3",
    )

    histogram = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Gang Productivity Distribution", className="section-title"),
                    html.Div("MT/day bins (0-13)", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="analytics-hist-median",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("Median Productivity", className="analytics-mini-label"),
                                        ]
                                    ),
                                    className="analytics-mini-card",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="analytics-hist-pct-low",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("% <4 MT/day", className="analytics-mini-label"),
                                        ]
                                    ),
                                    className="analytics-mini-card",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="analytics-hist-pct-high",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("% >6 MT/day", className="analytics-mini-label"),
                                        ]
                                    ),
                                    className="analytics-mini-card",
                                ),
                                md=4,
                            ),
                        ],
                        className="g-3 mb-3",
                    ),
                    dcc.Graph(
                        id="analytics-hist-chart",
                        config=GRAPH_CONFIG,
                        style={"height": "320px"},
                    ),
                ]
            ),
        ],
        className="viz-card shadow-soft w-100",
    )

    histogram_row = dbc.Row(
        [dbc.Col(histogram, md=12, className="d-flex")],
        className="g-3 mb-4 align-items-stretch analytics-row analytics-row-4",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Tower Erection Analytics", className="section-title"),
                            scope_strip,
                        ]
                    ),
                    dbc.Button(
                        "Show Overall Gang Performance",
                        id="btn-open-global-performance-erection",
                        color="primary",
                        size="sm",
                        className="summary-card__cta",
                    ),
                ],
                className="analytics-header d-flex flex-wrap justify-content-between align-items-start gap-2",
            ),
            kpi_cards,
            row2,
            hotspot_row,
            histogram_row,
            dcc.Store(id="analytics-payload", data=None),
            dcc.Store(id="analytics-audit-selection", data=None),
            dcc.Interval(id="analytics-refresh-interval", interval=30 * 60 * 1000, n_intervals=0),
            dcc.Download(id="analytics-audit-download"),
            audit_drawer,
        ],
        className="analytics-page",
    )
