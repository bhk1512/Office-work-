"""Layout for the Stringing Analytics tab."""
from __future__ import annotations

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


GRAPH_CONFIG = {
    "displayModeBar": False,
    "doubleClick": False,
    "scrollZoom": False,
}


def _audit_table_block() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Section", className="fw-semibold mb-1"),
                    dcc.Dropdown(
                        id="stringing-analytics-section-filter",
                        options=[],
                        value=None,
                        clearable=True,
                        placeholder="All sections",
                        className="filter-select filter-select--compact",
                    ),
                ],
                className="filter-field mb-2",
            ),
            dash_table.DataTable(
                id="stringing-analytics-audit-table",
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
            ),
        ]
    )


def build_stringing_analytics_layout() -> html.Div:
    audit_drawer = dbc.Offcanvas(
        [
            dbc.Tabs(
                [
                    dbc.Tab(_audit_table_block(), tab_id="audit-table", label="Audit Table"),
                    dbc.Tab(
                        html.Div(
                            id="stringing-analytics-audit-definition",
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
                                    id="stringing-analytics-audit-export-btn",
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
                id="stringing-analytics-audit-tabs",
                active_tab="audit-table",
            ),
            dbc.Button(
                "Close",
                id="stringing-analytics-audit-close",
                outline=True,
                color="secondary",
                size="sm",
                className="mt-3",
            ),
        ],
        id="stringing-analytics-audit-drawer",
        title=html.Div(id="stringing-analytics-audit-title"),
        placement="end",
        is_open=False,
        scrollable=True,
        className="analytics-audit-drawer",
    )

    scope_strip = html.Div(
        [
            html.Span(id="stringing-analytics-scope-range", className="analytics-scope-chip"),
            html.Span(id="stringing-analytics-scope-projects", className="analytics-scope-chip"),
            html.Span(id="stringing-analytics-scope-gangs", className="analytics-scope-chip"),
            html.Span(id="stringing-analytics-scope-spans", className="analytics-scope-chip"),
            html.Span(id="stringing-analytics-scope-totalkm", className="analytics-scope-chip"),
            html.Span("Manual excluded", className="analytics-scope-chip"),
        ],
        className="analytics-scope-strip analytics-scope-strip--inline",
    )

    kpi_cards = dbc.Row(
        [
            dbc.Col(
                html.Button(
                    [
                        html.Div("Output (km): total km strung (TSE)", className="analytics-kpi-title"),
                        html.Div(id="stringing-analytics-kpi-output", className="analytics-kpi-value"),
                        html.Div(id="stringing-analytics-kpi-output-sub", className="analytics-kpi-sub"),
                    ],
                    id="stringing-analytics-kpi-output-card",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=4,
            ),
            dbc.Col(
                html.Button(
                    [
                        html.Div("Readiness delay (median days)", className="analytics-kpi-title"),
                        html.Div(id="stringing-analytics-kpi-readiness", className="analytics-kpi-value"),
                        html.Div(id="stringing-analytics-kpi-readiness-sub", className="analytics-kpi-sub"),
                    ],
                    id="stringing-analytics-kpi-readiness-card",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=4,
            ),
            dbc.Col(
                html.Button(
                    [
                        html.Div("Flow delay (median days)", className="analytics-kpi-title"),
                        html.Div(id="stringing-analytics-kpi-flow", className="analytics-kpi-value"),
                        html.Div(id="stringing-analytics-kpi-flow-sub", className="analytics-kpi-sub"),
                    ],
                    id="stringing-analytics-kpi-flow-card",
                    className="analytics-kpi-card",
                    n_clicks=0,
                    type="button",
                ),
                md=4,
            ),
        ],
        className="g-3 mb-4",
    )

    readiness_hist = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Stretch Readiness Gap Distribution", className="section-title"),
                    html.Div(
                        "Erection complete -> P/O start",
                        className="section-sub",
                    ),
                    html.Div(
                        "Span = tower-to-tower section used for stringing",
                        className="analytics-note",
                    ),
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
                                                id="stringing-analytics-readiness-pct-15",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("% spans >15 days", className="analytics-mini-label"),
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
                                                id="stringing-analytics-readiness-pct-60",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("% spans >60 days", className="analytics-mini-label"),
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
                                                id="stringing-analytics-readiness-median",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("Median gap (days)", className="analytics-mini-label"),
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
                        id="stringing-analytics-readiness-hist",
                        config=GRAPH_CONFIG,
                        style={"height": "300px"},
                    ),
                ]
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    readiness_hotspot = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Project Hotspots (Readiness)", className="section-title"),
                    html.Div("Top 10 by median E->P/O gap", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="stringing-analytics-readiness-hotspot",
                    config=GRAPH_CONFIG,
                    style={"height": "320px"},
                )
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    readiness_funnel = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Readiness Funnel", className="section-title"),
                    html.Div("Where spans stack up", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="stringing-analytics-readiness-funnel",
                    config=GRAPH_CONFIG,
                    style={"height": "260px"},
                )
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    productivity_hist = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Gang Productivity Distribution", className="section-title"),
                    html.Div("KM/month bins", className="section-sub"),
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
                                                id="stringing-analytics-prod-median",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("Median productivity", className="analytics-mini-label"),
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
                                                id="stringing-analytics-prod-pct-low",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("% <3 km/month", className="analytics-mini-label"),
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
                                                id="stringing-analytics-prod-pct-high",
                                                className="analytics-mini-value",
                                            ),
                                            html.Div("% >=6 km/month", className="analytics-mini-label"),
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
                        id="stringing-analytics-prod-hist",
                        config=GRAPH_CONFIG,
                        style={"height": "300px"},
                    ),
                ]
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    productivity_share = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Gang Share vs Output Share", className="section-title"),
                    html.Div("Buckets by km/month", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                [
                    dcc.Graph(
                        id="stringing-analytics-share-chart",
                        config=GRAPH_CONFIG,
                        style={"height": "260px"},
                    ),
                    html.Div(
                        "Gang share = % of TSE gangs in bucket - Output share = % of total km executed by those gangs",
                        className="analytics-note",
                    ),
                ]
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    whatif_card = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("What-if: Improve bucket productivity", className="section-title"),
                    html.Div("Output constant; illustrative only", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                [
                    dcc.Dropdown(
                        id="stringing-analytics-whatif-bucket",
                        options=[],
                        value=None,
                        clearable=False,
                        className="analytics-whatif-select",
                    ),
                    dcc.Slider(
                        id="stringing-analytics-whatif-slider",
                        min=1,
                        max=10,
                        step=0.5,
                        value=4,
                        marks={value: str(value) for value in [1, 2, 3, 4, 6, 8, 10]},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        id="stringing-analytics-whatif-saved",
                                        className="analytics-whatif-value",
                                    ),
                                    html.Div(
                                        "Equivalent gang-months saved",
                                        className="analytics-whatif-label",
                                    ),
                                ],
                                className="analytics-whatif-metric",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id="stringing-analytics-whatif-unlocked",
                                        className="analytics-whatif-value",
                                    ),
                                    html.Div(
                                        "Equivalent km unlocked (illustrative)",
                                        className="analytics-whatif-label",
                                    ),
                                ],
                                className="analytics-whatif-metric",
                            ),
                        ],
                        className="analytics-whatif-metrics",
                    ),
                    dcc.Graph(
                        id="stringing-analytics-whatif-chart",
                        config=GRAPH_CONFIG,
                        style={"height": "180px"},
                    ),
                    html.Div(
                        [
                            html.A(
                                "Reset",
                                id="stringing-analytics-whatif-reset",
                                n_clicks=0,
                                className="analytics-whatif-reset",
                            ),
                            html.Span(
                                id="stringing-analytics-whatif-status",
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

    flow_hist = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("P/O -> Sag Lag Distribution", className="section-title"),
                    html.Div("P/O complete -> Final sag start", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="stringing-analytics-flow-hist",
                    config=GRAPH_CONFIG,
                    style={"height": "280px"},
                )
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    cycle_chart = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("End-to-end Span Cycle Time", className="section-title"),
                    html.Div("Erection complete -> Sag complete/start", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="stringing-analytics-cycle-chart",
                    config=GRAPH_CONFIG,
                    style={"height": "280px"},
                )
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    ageing_table = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Ageing List", className="section-title"),
                    html.Div("Top 20 spans by ageing days", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                dash_table.DataTable(
                    id="stringing-analytics-ageing-table",
                    columns=[],
                    data=[],
                    page_size=10,
                    fixed_rows={"headers": True},
                    style_table={"overflowX": "auto", "maxHeight": "360px"},
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
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    readiness_vs_prod = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Readiness vs Productivity", className="section-title"),
                    html.Div("Avg productivity by readiness bucket", className="section-sub"),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="stringing-analytics-relationship-chart",
                    config=GRAPH_CONFIG,
                    style={"height": "280px"},
                )
            ),
        ],
        className="viz-card shadow-soft h-100 w-100",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Stringing Analytics", className="section-title"),
                    scope_strip,
                ],
                className="analytics-header",
            ),
            kpi_cards,
            dbc.Row(
                [
                    dbc.Col(readiness_hist, md=7, className="d-flex"),
                    dbc.Col(readiness_hotspot, md=5, className="d-flex"),
                ],
                className="g-3 mb-4 align-items-stretch",
            ),
            dbc.Row(
                [dbc.Col(readiness_funnel, md=12, className="d-flex")],
                className="g-3 mb-4 align-items-stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(productivity_hist, md=6, className="d-flex"),
                    dbc.Col(productivity_share, md=6, className="d-flex"),
                ],
                className="g-3 mb-4 align-items-stretch",
            ),
            dbc.Row(
                [dbc.Col(whatif_card, md=12, className="d-flex")],
                className="g-3 mb-4 align-items-stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(flow_hist, md=6, className="d-flex"),
                    dbc.Col(cycle_chart, md=6, className="d-flex"),
                ],
                className="g-3 mb-4 align-items-stretch",
            ),
            dbc.Row(
                [dbc.Col(ageing_table, md=12, className="d-flex")],
                className="g-3 mb-4 align-items-stretch",
            ),
            dbc.Row(
                [dbc.Col(readiness_vs_prod, md=12, className="d-flex")],
                className="g-3 mb-4 align-items-stretch",
            ),
            dcc.Store(id="stringing-analytics-payload", data=None),
            dcc.Store(id="stringing-analytics-audit-selection", data=None),
            dcc.Interval(id="stringing-analytics-refresh-interval", interval=30 * 60 * 1000, n_intervals=0),
            dcc.Download(id="stringing-analytics-audit-download"),
            audit_drawer,
        ],
        className="analytics-page",
    )
