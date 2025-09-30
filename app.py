import numpy as np
import pandas as pd
from pathlib import Path
import datetime as dt

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots

from io import BytesIO
from dash import dash_table
from dash.dcc import Download, send_bytes

import time

# ------------------ CONFIG ------------------
DATA_PATH = Path("ErectionCompiled_Output.xlsx")
PREFERRED_SHEET = "DailyExpanded"
DEFAULT_BENCHMARK = 9.0
LOSS_MAX_GAP_DAYS = 15

# ------------------ LOAD DATA ------------------
def load_daily_from_dailyexpanded(xl, sheet="DailyExpanded"):
    df = pd.read_excel(xl, sheet_name=sheet)
    # column picking with light tolerance
    def pick(df, opts):
        m = {str(c).strip().lower(): c for c in df.columns}
        for o in opts:
            key = o.strip().lower()
            if key in m: return m[key]
        for k, c in m.items():
            if any(o.lower() in k for o in opts): return c
        raise KeyError(f"Column not found among {opts}")
    col_date = pick(df, ["Work Date","date"])
    col_prod = pick(df, ["Productivity","daily_prod_mt","avg_daily_prod_mt"])
    col_proj = pick(df, ["Project Name","project_name"])
    col_gang = pick(df, ["Gang name","gang_name"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df[col_date], errors="coerce").dt.normalize(),
        "daily_prod_mt": pd.to_numeric(df[col_prod], errors="coerce"),
        "project_name": df[col_proj].astype(str).str.strip(),
        "gang_name": df[col_gang].astype(str).str.strip()
    }).dropna(subset=["date","daily_prod_mt"])
    return out

def load_daily_from_rawdata(xl, sheet="RawData"):
    df = pd.read_excel(xl, sheet_name=sheet)
    def pick(df, opts):
        m = {str(c).strip().lower(): c for c in df.columns}
        for o in opts:
            key = o.strip().lower()
            if key in m: return m[key]
        for k, c in m.items():
            if any(o.lower() in k for o in opts): return c
        raise KeyError(f"Column not found among {opts}")
    s = pick(df, ["Start Date","starting date"])
    e = pick(df, ["Complete Date","completion date"])
    p = pick(df, ["Productivity","avg_daily_prod_mt","daily_prod_mt"])
    pr= pick(df, ["Project Name","project_name"])
    g = pick(df, ["Gang name","gang_name"])
    base = pd.DataFrame({
        "start": pd.to_datetime(df[s], errors="coerce"),
        "end":   pd.to_datetime(df[e], errors="coerce"),
        "daily_prod_mt": pd.to_numeric(df[p], errors="coerce"),
        "project_name": df[pr].astype(str).str.strip(),
        "gang_name": df[g].astype(str).str.strip()
    }).dropna(subset=["start","end","daily_prod_mt"])
    rows = []
    for _, r in base.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append({
                "date": d.normalize(),
                "daily_prod_mt": r["daily_prod_mt"],
                "project_name": r["project_name"],
                "gang_name": r["gang_name"]
            })
    return pd.DataFrame(rows)

def load_daily(path: Path):
    xl = pd.ExcelFile(path)
    if PREFERRED_SHEET in xl.sheet_names:
        return load_daily_from_dailyexpanded(xl, PREFERRED_SHEET)
    elif "RawData" in xl.sheet_names:
        return load_daily_from_rawdata(xl, "RawData")
    else:
        raise FileNotFoundError("Neither 'DailyExpanded' nor 'RawData' found in workbook.")

df_day = load_daily(DATA_PATH)
df_day["month"] = df_day["date"].dt.to_period("M").dt.to_timestamp()

# ------------------ METRICS & CHART HELPERS ------------------
def calc_idle_and_loss(group_df: pd.DataFrame, loss_max_gap_days=LOSS_MAX_GAP_DAYS, baseline_mt_per_day=None):
    # Idle days between non-consecutive work dates (cap each gap by loss_max_gap_days)
    dts = group_df["date"].dropna().drop_duplicates().sort_values()
    if len(dts) > 1:
        diff_days = dts.diff().dt.days.fillna(0).astype(int) - 1
        idle = int(np.minimum(diff_days[diff_days >= 1], loss_max_gap_days).sum())
    else:
        idle = 0
    if baseline_mt_per_day is not None and not pd.isna(baseline_mt_per_day):
        baseline = float(baseline_mt_per_day)
    elif len(group_df):
        baseline = float(group_df["daily_prod_mt"].mean())
    else:
        baseline = 0.0
    loss_mt = baseline * idle
    delivered_mt = float(group_df["daily_prod_mt"].sum())
    potential_mt = delivered_mt + loss_mt
    return idle, baseline, loss_mt, delivered_mt, potential_mt

def apply_filters(df, projects, months, gangs, overall_months=False, overall_gangs=False):
    d = df.copy()
    if projects:
        d = d[d["project_name"].isin(projects)]
    if not overall_months and months:
        d = d[d["month"].isin(months)]
    if not overall_gangs and gangs:
        d = d[d["gang_name"].isin(gangs)]
    return d

def create_monthly_line_chart(d, bench=DEFAULT_BENCHMARK):
    if d.empty:
        return go.Figure().update_layout(height=300, margin=dict(l=40,r=20,t=30,b=50))
    df_monthly = d.groupby("month")["daily_prod_mt"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_monthly["month"], y=df_monthly["daily_prod_mt"],
        mode="lines+markers", name="Avg Productivity", line=dict(color="#0074D9")
    ))
    fig.add_hline(
        y=bench, line_dash="dot", line_color="red",
        annotation_text=f"Benchmark {bench} MT/day",
        annotation_position="top left"
    )
    fig.update_layout(
        height=300, margin=dict(l=40,r=20,t=30,b=50),
        xaxis_title="Month", yaxis_title="Avg Productivity (MT)",
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff"
    )
    return fig

def create_top_bottom_gangs_charts(d):
    if d.empty:
        empty = go.Figure().update_layout(height=280, margin=dict(l=40,r=20,t=30,b=50))
        return empty, empty
    df_gang = d.groupby("gang_name")["daily_prod_mt"].mean().reset_index()
    df_gang = df_gang.sort_values("daily_prod_mt", ascending=False)
    top5 = df_gang.head(5); bottom5 = df_gang.tail(5)

    fig_top = go.Figure(go.Bar(
        x=top5["gang_name"], y=top5["daily_prod_mt"],
        marker_color="green", text=top5["daily_prod_mt"].round(2),
        textposition="outside", name="Top 5"
    ))
    fig_top.update_layout(title="Top 5 Gangs (Avg Productivity)", yaxis_title="MT/day",
                          height=280, margin=dict(l=40,r=20,t=30,b=50))

    fig_bottom = go.Figure(go.Bar(
        x=bottom5["gang_name"], y=bottom5["daily_prod_mt"],
        marker_color="red", text=bottom5["daily_prod_mt"].round(2),
        textposition="outside", name="Bottom 5"
    ))
    fig_bottom.update_layout(title="Bottom 5 Gangs (Avg Productivity)", yaxis_title="MT/day",
                             height=280, margin=dict(l=40,r=20,t=30,b=50))
    return fig_top, fig_bottom

def create_project_lines_chart(df_all, selected_projects=None, bench=DEFAULT_BENCHMARK):
    if df_all.empty:
        return go.Figure().update_layout(height=300, margin=dict(l=40,r=20,t=30,b=50))
    df_monthly = df_all.groupby(["month","project_name"])["daily_prod_mt"].mean().reset_index()
    fig = go.Figure()
    if selected_projects:
        projects_to_plot = selected_projects
    else:
        projects_to_plot = df_monthly["project_name"].unique()
    for proj in projects_to_plot:
        proj_df = df_monthly[df_monthly["project_name"] == proj]
        fig.add_trace(go.Scatter(x=proj_df["month"], y=proj_df["daily_prod_mt"],
                                 mode="lines+markers", name=proj))
    fig.add_hline(y=bench, line_dash="dot", line_color="red",
                  annotation_text=f"Benchmark {bench} MT/day",
                  annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=40,r=20,t=30,b=50),
                      xaxis_title="Month", yaxis_title="Avg Productivity (MT)",
                      plot_bgcolor="#fafafa", paper_bgcolor="#ffffff")
    return fig


def compute_idle_intervals_per_gang(d: pd.DataFrame, loss_max_gap_days=LOSS_MAX_GAP_DAYS) -> pd.DataFrame:
    """
    Returns one row per idle interval per gang:
      gang_name, interval_start, interval_end, raw_gap_days, idle_days_capped
    """
    rows = []
    for gname, gdf in d.groupby("gang_name"):
        dts = gdf["date"].dropna().drop_duplicates().sort_values().to_list()
        if len(dts) < 2:
            continue
        for i in range(1, len(dts)):
            gap = (dts[i] - dts[i-1]).days - 1
            if gap >= 1:
                interval_start = (dts[i-1] + pd.Timedelta(days=1)).normalize()
                interval_end   = (dts[i]   - pd.Timedelta(days=1)).normalize()
                rows.append({
                    "gang_name": gname,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "raw_gap_days": gap,
                    "idle_days_capped": int(min(gap, loss_max_gap_days)),
                })
    return pd.DataFrame(rows)

from io import BytesIO

def make_trace_workbook_bytes(
    d_scoped: pd.DataFrame,
    months_ts,
    projects,
    gangs,
    bench,
    gang_for_sheet: str | None = None,
) -> bytes:
    """
    Build an Excel (bytes) containing:
      - PerGangSummary
      - IdleIntervals
      - DailyProductivity
      - ProjectsMonthly
      - SelectionContext
      - Assumptions
      - (optional) Summary_<gang>, Idle_<gang>, Daily_<gang>
    """
    def _sanitize_sheet_name(s: str) -> str:
        # Excel sheet name: max 31 chars, no []:*?/\
        import re
        s = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(s))
        return s[:31]

    # --- Per-gang summary using calc_idle_and_loss ---
    summ_rows = []
    for gname, gdf in d_scoped.groupby("gang_name"):
        idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(gdf)
        summ_rows.append({
            "gang_name": gname,
            "delivered_mt": delivered,
            "lost_mt": loss_mt,
            "potential_mt": potential,
            "baseline_mt_per_day": baseline,
            "idle_days_capped": idle,
            "first_date": gdf["date"].min(),
            "last_date": gdf["date"].max(),
            "active_days": gdf["date"].nunique(),
        })
    per_gang_summary = (
        pd.DataFrame(summ_rows)
        .sort_values("potential_mt", ascending=False)
        if summ_rows else pd.DataFrame(columns=[
            "gang_name","delivered_mt","lost_mt","potential_mt",
            "baseline_mt_per_day","idle_days_capped","first_date","last_date","active_days"
        ])
    )

    idle_df = compute_idle_intervals_per_gang(d_scoped)
    daily_df = d_scoped.sort_values(["gang_name","date"])[
        ["date","gang_name","project_name","daily_prod_mt"]
    ].copy()
    proj_month = d_scoped.groupby(["project_name","month"])["daily_prod_mt"].mean().reset_index()

    ctx = {
        "projects": ", ".join(projects or []) or "(all)",
        "gangs": ", ".join(gangs or []) or "(all)",
        "months": ", ".join([m.strftime("%Y-%m") for m in (months_ts or [])]) or "(all / overall)",
        "benchmark": bench,
        "loss_cap_days": LOSS_MAX_GAP_DAYS,
    }
    ctx_df = pd.DataFrame([ctx])

    assumptions = pd.DataFrame({
        "Notes": [
            f"Loss cap per gap: {LOSS_MAX_GAP_DAYS} days.",
            "Efficiency = delivered / (delivered + lost). Lost = baseline * capped idle days.",
            "Idle interval = gaps between observed work dates; dates inferred from current filtered scope.",
            "All numbers reflect current dashboard filters (project, period, and gang if applied).",
        ]
    })

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        per_gang_summary.to_excel(w, "PerGangSummary", index=False)
        idle_df.to_excel(w, "IdleIntervals", index=False)
        daily_df.to_excel(w, "DailyProductivity", index=False)
        proj_month.to_excel(w, "ProjectsMonthly", index=False)
        ctx_df.to_excel(w, "SelectionContext", index=False)
        assumptions.to_excel(w, "Assumptions", index=False)

        # Optional dedicated sheets for a selected/picked gang
        if gang_for_sheet:
            gname = str(gang_for_sheet)
            gsel = d_scoped[d_scoped["gang_name"] == gname]
            if not gsel.empty:
                idle_one = compute_idle_intervals_per_gang(gsel)
                idle_one.to_excel(w, _sanitize_sheet_name(f"Idle_{gname}"), index=False)
                gsel.sort_values("date")[["date","project_name","daily_prod_mt"]].assign(
                    date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
                ).to_excel(w, _sanitize_sheet_name(f"Daily_{gname}"), index=False)

                idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(gsel)
                eff = (delivered / potential * 100) if potential > 0 else 0.0
                pd.DataFrame([{
                    "gang_name": gname,
                    "delivered_mt": delivered,
                    "lost_mt": loss_mt,
                    "potential_mt": potential,
                    "efficiency_%": eff,
                    "baseline_mt_per_day": baseline,
                    "idle_days_capped": idle
                }]).to_excel(w, _sanitize_sheet_name(f"Summary_{gname}"), index=False)

    return bio.getvalue()




# ------------------ APP ------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "KEC Productivity"


from dash.dependencies import Input, Output, State

app.clientside_callback(
    """
    function(lossClick, topClick, bottomClick, prevMeta) {
        function getGang(cd, src){
            if(!cd || !cd.points || !cd.points.length){ return null; }
            return src === 'g-actual-vs-bench' ? cd.points[0].y : cd.points[0].x;
        }
        // Determine which graph fired
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
        var isDbl = isSameSrc && isSameGang && (now - (last.ts || 0) <= 700);  // 700ms window

        var newMeta = {source: src, gang: gang, ts: now};
        var dblData = isDbl ? {source: src, gang: gang, ts: now} : null;

        return [newMeta, dblData];
    }
    """,
    [Output("store-click-meta", "data"), Output("store-dblclick", "data")],
    [Input("g-actual-vs-bench", "clickData"),
     Input("g-top5", "clickData"),
     Input("g-bottom5", "clickData")],
    [State("store-click-meta", "data")]
)









# Options
project_options = sorted(df_day["project_name"].dropna().unique())
month_options = sorted(df_day["month"].dropna().unique())
month_option_items = [
    {"label": m.strftime("%b %Y"), "value": m.strftime("%Y-%m")}
    for m in month_options
]
month_labels = [m.strftime("%b %Y") for m in month_options]  # Month names
gang_options = sorted(df_day["gang_name"].dropna().unique())

# ---- FILTER BAR ----
def get_quick_date_options():
    today = pd.Timestamp.today().normalize()
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    last_3_months = today - pd.DateOffset(months=3)
    last_quarter_start = (today.to_period('Q') - 1).start_time
    last_6_months = today - pd.DateOffset(months=6)

    return {
        "3M": (last_3_months, today),
        "Q": (last_quarter_start, today),
        "6M": (last_6_months, today),
        "YTD": (start_of_year, today)
    }

controls = dbc.Card(
    dbc.CardBody([
        html.Div("Filters", className="fw-bold mb-2"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(project_options, multi=True, placeholder="Select project(s)", id="f-project"), md=4),
            dbc.Col(dcc.Dropdown(month_option_items, multi=True, placeholder="Select month(s)", id="f-month"), md=4),
            dbc.Col(dcc.Dropdown(gang_options, multi=True, placeholder="Select gang(s)", id="f-gang"), md=4),
        ], className="g-2"),

        # Quick time ranges
        dbc.Row([
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
                    className="mt-2"
                ),
                md=12
            ),
        ]),

        dbc.Row([
            dbc.Col(dbc.Checklist(
                options=[{"label": " Overall months (ignore month filter)", "value": "all_months"}],
                value=[], id="f-overall-months", switch=True
            ), md=4),
            dbc.Col(dbc.Checklist(
                options=[{"label": " Overall gangs (ignore gang filter)", "value": "all_gangs"}],
                value=[], id="f-overall-gangs", switch=True
            ), md=4),
        ], className="mt-2"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div("Benchmark (MT / gang / day)", className="small text-muted"),
                dcc.Input(id="bench", type="number", value=DEFAULT_BENCHMARK, step=0.1)
            ], md=3),
        ])
    ]),
    className="mb-3 shadow-sm"
)


# ---- KPI CARDS ----
def kpi_card(title, id_value, color, subtitle=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="fw-bold"),
            html.H2(id=id_value, className="mb-0"),
            html.Div(subtitle, className="small text-muted") if subtitle else None
        ]),
        className=f"text-white shadow-sm", style={"backgroundColor": color}
    )

kpi_cards = dbc.Row([
    # Avg Output (with delta line)
    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div("Avg Output / Gang / Day", className="fw-bold text-white-50"),
                html.H2(id="kpi-avg", className="mb-0 text-white"),
                html.Div(id="kpi-delta", className="small text-white-50")
            ]),
            className="shadow-sm", style={"backgroundColor": "#4f9cff", "border": "0", "borderRadius": "12px"}
        ),
        xs=12, sm=6, md=6, lg=3, xl=2
    ),

    # Benchmark
    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div("Benchmark Target", className="fw-bold text-white-50"),
                html.H2(id="kpi-bench", className="mb-0 text-white")
            ]),
            className="shadow-sm", style={"backgroundColor": "#9FE2BF", "border": "0", "borderRadius": "12px"}
        ),
        xs=12, sm=6, md=6, lg=3, xl=2
    ),

    # Active Gangs
    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div("Active Gangs", className="fw-bold text-white-50"),
                html.H2(id="kpi-active", className="mb-0 text-white")
            ]),
            className="shadow-sm", style={"backgroundColor": "#d6b4fc", "border": "0", "borderRadius": "12px"}
        ),
        xs=12, sm=6, md=6, lg=3, xl=2
    ),

    # Avg Efficiency
    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div("Avg Efficiency", className="fw-bold text-white-50"),
                html.H2(id="kpi-eff", className="mb-0 text-white")
            ]),
            className="shadow-sm", style={"backgroundColor": "#ffbb66", "border": "0", "borderRadius": "12px"}
        ),
        xs=12, sm=6, md=6, lg=3, xl=2
    ),

    # Lost Units
    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div("Lost Units", className="fw-bold text-white-50"),
                html.H2(id="kpi-loss", className="mb-0 text-white")
            ]),
            className="shadow-sm", style={"backgroundColor": "#ff7b7b", "border": "0", "borderRadius": "12px"}
        ),
        xs=12, sm=6, md=6, lg=3, xl=2
    ),
], className="g-3 align-items-stretch")


# at the top (near other UI constants), mirror the same numbers
ROW_PX = 56
VISIBLE_ROWS = 15
TOPBOT_MARGIN = 120
CONTAINER_HEIGHT = ROW_PX * VISIBLE_ROWS + TOPBOT_MARGIN  # â‰ˆ 960px

# Graph wrapper
gang_bar = html.Div(
    dcc.Graph(id="g-actual-vs-bench", config={"displayModeBar": False, "doubleClick": "false"}),
    style={"height": f"{CONTAINER_HEIGHT}px", "overflowY": "auto"}  # shows ~15, scrolls for more
)



# Define the modal once (e.g., above app.layout)
modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
    dbc.ModalBody(id="modal-body"),
    dbc.ModalFooter(dbc.Button("Close", id="modal-close", className="ms-auto", n_clicks=0)),
], id="loss-modal", is_open=False)


# Store the last clicked gang (optional, for drill)
store_selected = dcc.Store(id="store-selected-gang")
store_click_meta = dcc.Store(id="store-click-meta", data=None)   # remembers last single click
store_dblclick   = dcc.Store(id="store-dblclick", data=None)     # set only on real double-click

trace_block = dbc.Card(
    dbc.CardBody([
	# directly inside dbc.CardBody([ ... ]) in trace_block, BEFORE the two table columns:
	dbc.Row([
            dbc.Col([
            html.Label("Pick a gang (overrides Gang filter)", className="fw-semibold mb-1"),
            dcc.Dropdown(
                id="trace-gang",
                options=[],            # will be filled by a callback
                value=None,
                placeholder="Start typing a gang...",
                clearable=True,
                persistence=True,
                persistence_type="session"
            )
        ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(html.H5("Traceability"), md=8),
            dbc.Col(dbc.Button("Export Trace Excel", id="btn-export-trace", color="primary"),
                    md=4, className="text-end"),
        ], className="align-items-center mb-3"),

        dbc.Row([
            dbc.Col([
                html.Div("Idle Intervals (per gang)", className="fw-bold mb-2"),
                dash_table.DataTable(
                    id="tbl-idle-intervals",
                    columns=[
                        {"name": "Gang", "id": "gang_name"},
                        {"name": "Interval Start", "id": "interval_start"},
                        {"name": "Interval End", "id": "interval_end"},
                        {"name": "Raw Gap (days)", "id": "raw_gap_days"},
                        {"name": "Idle Counted (days)", "id": "idle_days_capped"},
                    ],
                    data=[], page_size=10, style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Inter, system-ui", "fontSize": 13},
                )
            ], md=6),
            dbc.Col([
                html.Div("Daily Productivity (selected scope)", className="fw-bold mb-2"),
                dash_table.DataTable(
                    id="tbl-daily-prod",
                    columns=[
                        {"name": "Date", "id": "date"},
                        {"name": "Gang", "id": "gang_name"},
                        {"name": "Project", "id": "project_name"},
                        {"name": "MT/day", "id": "daily_prod_mt"},
                    ],
                    data=[], page_size=10, style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Inter, system-ui", "fontSize": 13},
                )
            ], md=6),
        ]),
        Download(id="download-trace-xlsx"),
        store_selected,
    ]),
    className="mt-4 shadow-sm"
)



app.layout = dbc.Container([
    html.H2("Measure Output. Expose Lost Units.", className="mt-3 fw-bold"),
    html.Div("Tag causes; assign fixes.", className="text-muted mb-3"),
    controls,
    kpi_cards,
    dbc.Row([
        dbc.Col([
    	    html.H5("Projects over Months"),
    	    dcc.Graph(id="g-project-lines", config={"displayModeBar": False}),
    	    html.H5("Average Productivity per Month"),
    	    dcc.Graph(id="g-monthly", config={"displayModeBar": False}),
	    ], md=6),

        dbc.Col([
            html.H5("Actual vs Potential Performance (All Gangs)"),
            gang_bar
        ], md=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([html.H5("Top 5 Gangs"), dcc.Graph(id="g-top5", config={"displayModeBar": False, "doubleClick": "false"})], md=6),
        dbc.Col([html.H5("Bottom 5 Gangs"), dcc.Graph(id="g-bottom5", config={"displayModeBar": False, "doubleClick": "false"})], md=6),
    ], className="g-3"),
    trace_block,
    store_click_meta,
    store_dblclick, 
modal,
], fluid=True)


# -------------- CALLBACKS --------------

@app.callback(
    Output("kpi-avg", "children"),
    Output("kpi-delta", "children"),
    Output("kpi-bench", "children"),
    Output("kpi-active", "children"),   # NEW
    Output("kpi-eff", "children"),      # NEW
    Output("kpi-loss", "children"),     # NEW
    Output("g-actual-vs-bench", "figure"),
    Output("g-monthly", "figure"),
    Output("g-top5", "figure"),
    Output("g-bottom5", "figure"),
    Output("g-project-lines", "figure"),
    Input("f-project", "value"),
    Input("f-month", "value"),
    Input("f-quick-range", "value"),    # keep if you added quick ranges
    Input("f-gang", "value"),
    Input("f-overall-months", "value"),
    Input("f-overall-gangs", "value"),
    Input("bench", "value"),
)
def update_dashboard(projects, months, quick_range, gangs, overall_months_val, overall_gangs_val, bench):
    # --- existing selection parsing (keep yours) ---
    overall_months = "all_months" in (overall_months_val or [])
    overall_gangs = "all_gangs" in (overall_gangs_val or [])

    months_ts = []
    if quick_range:
        start_dt, end_dt = get_quick_date_options()[quick_range]
        months_ts = pd.period_range(start=start_dt, end=end_dt, freq="M").to_timestamp().tolist()
    elif months:
        months_ts = [pd.Period(m, "M").to_timestamp() for m in months]

    d = apply_filters(df_day, projects or [], months_ts or [], gangs or [],
                      overall_months=overall_months, overall_gangs=overall_gangs)
    
    # inside update_dashboard(...) after you build months_ts / overall_* flags
    d_scoped = apply_filters(
        df_day, projects or [], months_ts or [], gangs or [],
        overall_months=overall_months, overall_gangs=overall_gangs
    )

    # ignore the current gang filter for top/bottom
    d_topbot = apply_filters(
        df_day, projects or [], months_ts or [], [],               # gangs=[]
        overall_months=overall_months, overall_gangs=True          # ignore gang filter
    )


    # ---- KPI: Avg Output vs Bench (existing) ----
    avg_prod = d["daily_prod_mt"].mean() if len(d) else 0.0
    delta_pct = None if (bench is None or bench == 0) else (avg_prod - bench) / bench * 100
    kpi_avg = f"{avg_prod:.2f} MT"
    kpi_delta = "(n/a)" if delta_pct is None else f"{delta_pct:+.1f}%"
    kpi_bench = f"{bench:.2f} MT"

    # ---- Determine month scope for loss metrics ----
    selected_months = months_ts or []
    if selected_months:
        loss_month_start = max(selected_months)
    else:
        loss_month_start = pd.Timestamp.today().to_period("M").to_timestamp()
    loss_month_end = loss_month_start + pd.offsets.MonthBegin(1)

    scope_mask = pd.Series(True, index=df_day.index)
    if projects:
        scope_mask &= df_day["project_name"].isin(projects)
    if not overall_gangs and gangs:
        scope_mask &= df_day["gang_name"].isin(gangs)
    scoped_all = df_day.loc[scope_mask].copy()

    d_loss_scope = scoped_all[
        (scoped_all["date"] >= loss_month_start) & (scoped_all["date"] < loss_month_end)
    ].copy()

    history_scope = scoped_all[scoped_all["date"] < loss_month_start]
    baseline_map = {}
    if not history_scope.empty:
        baseline_map = (
            history_scope.groupby("gang_name")["daily_prod_mt"]
            .mean()
            .fillna(0)
            .to_dict()
        )

    # ---- Build loss dataframe (latest month only) ----
    loss_rows = []
    for gname, gdf in d_loss_scope.groupby("gang_name"):
        override_baseline = baseline_map.get(gname, 0.0)
        idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
            gdf, baseline_mt_per_day=override_baseline
        )
        loss_rows.append({
            "gang_name": gname,
            "delivered": delivered,
            "lost": loss_mt,
            "potential": potential,
            "avg_prod": gdf["daily_prod_mt"].mean(),
            "baseline": baseline
        })
    df_loss = (
        pd.DataFrame(loss_rows).sort_values("potential", ascending=True)
        if loss_rows else pd.DataFrame(columns=["gang_name","delivered","lost","potential","avg_prod","baseline"])
    )
    
    # --- thickness per gang row (track height) ---
    ROW_PX = 56            # try 56–64 for chunky rows
    TOPBOT_MARGIN = 120    # top + bottom margins

    fig_height = int(ROW_PX * max(1, len(df_loss)) + TOPBOT_MARGIN)


    # ---- NEW KPIs: Active Gangs, Avg Efficiency, Lost Units ----
    active_gangs = d_loss_scope["gang_name"].nunique()
    tot_delivered = float(df_loss["delivered"].sum()) if not df_loss.empty else 0.0
    tot_lost = float(df_loss["lost"].sum()) if not df_loss.empty else 0.0
    tot_potential = tot_delivered + tot_lost
    eff_pct = (tot_delivered / tot_potential * 100) if tot_potential > 0 else 0.0
    lost_pct = (tot_lost / tot_potential * 100) if tot_potential > 0 else 0.0

    kpi_active = f"{active_gangs}"
    kpi_eff = f"{eff_pct:.1f}%"
    kpi_loss = f"{lost_pct:.1f}%"

    fig_loss = go.Figure()
    if not df_loss.empty:
        fig_loss.add_bar(
            x=df_loss["delivered"], y=df_loss["gang_name"],
            orientation="h", marker_color="green",
            text=df_loss["delivered"].round(1), textposition="inside",
            name="Delivered", width=0.95
        )
        fig_loss.add_bar(
            x=df_loss["lost"], y=df_loss["gang_name"],
            orientation="h", marker_color="red",
            text=df_loss["lost"].round(1), textposition="inside",
            name="Loss", base=df_loss["delivered"], width=0.95
        )
        for _, row in df_loss.iterrows():
            fig_loss.add_annotation(
                x=row["potential"], y=row["gang_name"],
                text=f"{row['avg_prod']:.2f} MT/day (Baseline: {row['baseline']:.2f} MT/day)",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=10, color="black")
            )
    fig_loss.update_layout(
        barmode="stack", bargap=0.02,
        height=fig_height,
        margin=dict(l=140, r=120, t=30, b=30),
        xaxis_title="Potential (MT)", yaxis_title="Gang",
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff"
    )

    # ---- Other charts (unchanged builders) ----
    fig_line = create_monthly_line_chart(d, bench=bench)
    fig_top5, fig_bottom5 = create_top_bottom_gangs_charts(d_topbot)
    fig_project_lines = create_project_lines_chart(df_day, selected_projects=projects, bench=bench)

    return (kpi_avg, kpi_delta, kpi_bench,
            kpi_active, kpi_eff, kpi_loss,
            fig_loss, fig_line, fig_top5, fig_bottom5, fig_project_lines)



# Loss modal: click a bar (gang) to show loss breakdown for current filter
@app.callback(
    Output("loss-modal", "is_open"),
    Output("modal-title", "children"),
    Output("modal-body", "children"),
    Input("store-dblclick", "data"),   # <â€” open only on true double-click
    Input("modal-close", "n_clicks"),
    State("loss-modal", "is_open"),
    State("f-project", "value"),
    State("f-month", "value"),
    State("f-quick-range", "value"),
    State("f-gang", "value"),
    State("f-overall-months", "value"),
    State("f-overall-gangs", "value"),
)
def show_loss_on_double_click(dbl, close_clicks, is_open,
                              projects, months, quick_range, gangs,
                              overall_months_val, overall_gangs_val):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, dash.no_update, dash.no_update

    # Close button
    if ctx.triggered[0]["prop_id"].startswith("modal-close"):
        return False, dash.no_update, dash.no_update

    # No double-click data -> do nothing
    if not dbl or not dbl.get("gang"):
        return is_open, dash.no_update, dash.no_update

    gang_clicked = dbl["gang"]
    overall_months = "all_months" in (overall_months_val or [])

    # Resolve months
    months_ts = []
    if quick_range:
        start_dt, end_dt = get_quick_date_options()[quick_range]
        months_ts = pd.period_range(start=start_dt, end=end_dt, freq="M").to_timestamp().tolist()
    elif months:
        months_ts = [pd.Period(m, "M").to_timestamp() for m in months]

    # Project scope only; month + baseline mirror main loss logic
    scope_mask = pd.Series(True, index=df_day.index)
    if projects:
        scope_mask &= df_day["project_name"].isin(projects)
    scoped_all = df_day.loc[scope_mask].copy()

    loss_month_candidates = months_ts or []
    if loss_month_candidates:
        loss_month_start = max(loss_month_candidates)
    else:
        loss_month_start = pd.Timestamp.today().to_period("M").to_timestamp()
    loss_month_end = loss_month_start + pd.offsets.MonthBegin(1)

    d_sel = scoped_all[
        (scoped_all["date"] >= loss_month_start)
        & (scoped_all["date"] < loss_month_end)
        & (scoped_all["gang_name"] == gang_clicked)
    ]

    if d_sel.empty:
        return True, "Gang Efficiency & Loss", "No data in current selection."

    history_sel = scoped_all[
        (scoped_all["date"] < loss_month_start)
        & (scoped_all["gang_name"] == gang_clicked)
    ]
    baseline_override = history_sel["daily_prod_mt"].mean() if not history_sel.empty else 0.0
    if pd.isna(baseline_override):
        baseline_override = 0.0

    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
        d_sel, baseline_mt_per_day=baseline_override
    )
    eff = (delivered / potential * 100) if potential > 0 else 0.0
    lost_pct = (loss_mt / potential * 100) if potential > 0 else 0.0


    body = html.Div([
        html.Div(f"Gang: {gang_clicked}", className="fw-bold mb-2"),
        html.Div(f"Erected (Delivered): {delivered:.2f} MT"),
        html.Div(f"Loss: {loss_mt:.2f} MT ({lost_pct:.1f}%)", className="text-danger"),
        html.Div(f"Efficiency: {eff:.1f}%", className="mt-1"),
        html.Hr(),
        html.Div(f"Baseline MT/day: {baseline:.2f}"),
        html.Div(f"Idle days (cap {LOSS_MAX_GAP_DAYS}): {idle}"),
        html.Div(f"Potential MT: {potential:.2f}"),
    ])
    return True, "Gang Efficiency & Loss", body


# If a quick range is chosen, clear the manual Month dropdown

@app.callback(
    Output("f-month", "value"),
    Input("f-quick-range", "value"),
    prevent_initial_call=True
)
def _clear_months_when_quickrange_selected(quick):
    if quick:
        return []  # clear months
    return dash.no_update

# If manual months are picked, clear the quick range
@app.callback(
    Output("f-quick-range", "value"),
    Input("f-month", "value"),
    prevent_initial_call=True
)
def _clear_quick_when_months_selected(months):
    if months:
        return None
    return dash.no_update


@app.callback(
    Output("store-selected-gang", "data"),
    Input("g-actual-vs-bench", "clickData"),
    Input("g-top5", "clickData"),
    Input("g-bottom5", "clickData"),
    prevent_initial_call=True
)
def set_selected_gang(loss_click, top_click, bottom_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    source = ctx.triggered[0]["prop_id"].split(".")[0]
    if source == "g-actual-vs-bench":
        return (loss_click or {}).get("points", [{}])[0].get("y")  # y holds gang for horizontal
    if source == "g-top5":
        return (top_click or {}).get("points", [{}])[0].get("x")   # x holds gang for vertical
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
    Input("f-overall-months", "value"),
    Input("f-overall-gangs", "value"),
    Input("trace-gang", "value"),          # NEW: dropdown value (highest priority)
    Input("store-selected-gang", "data"),   # clicked gang
)
def update_trace_tables(projects, months, quick_range, gangs, overall_months_val, overall_gangs_val,
                        trace_gang_value, selected_gang):
    overall_months = "all_months" in (overall_months_val or [])
    overall_gangs  = "all_gangs"  in (overall_gangs_val  or [])

    # months â†’ timestamps
    months_ts = []
    if quick_range:
        start_dt, end_dt = get_quick_date_options()[quick_range]
        months_ts = pd.period_range(start=start_dt, end=end_dt, freq="M").to_timestamp().tolist()
    elif months:
        months_ts = [pd.Period(m, "M").to_timestamp() for m in months]

    # scoped = respects all filters
    d_scoped = apply_filters(df_day, projects or [], months_ts or [], gangs or [],
                             overall_months=overall_months, overall_gangs=overall_gangs)
    # base = project+period only (ignore gang filter)
    d_base = apply_filters(df_day, projects or [], months_ts or [], [],
                           overall_months=overall_months, overall_gangs=True)

    # ----- Idle Intervals: dropdown > clicked gang > gang filter > all in scope
    if trace_gang_value:
        d_for_idle = d_base[d_base["gang_name"] == trace_gang_value]
    elif selected_gang:
        if selected_gang in d_scoped["gang_name"].unique():
            d_for_idle = d_scoped[d_scoped["gang_name"] == selected_gang]
        else:
            d_for_idle = d_base[d_base["gang_name"] == selected_gang]
    elif gangs:
        d_for_idle = d_scoped
    else:
        d_for_idle = d_scoped

    idle_df = compute_idle_intervals_per_gang(d_for_idle)
    if not idle_df.empty:
        idle_df = idle_df.assign(
            interval_start=idle_df["interval_start"].dt.strftime("%d-%m-%Y"),
            interval_end=idle_df["interval_end"].dt.strftime("%d-%m-%Y"),
        )
    idle_data = idle_df.to_dict("records")

    # ----- Daily table: same priority
    if trace_gang_value:
        daily = d_base[d_base["gang_name"] == trace_gang_value]
    elif selected_gang:
        if selected_gang in d_scoped["gang_name"].unique():
            daily = d_scoped[d_scoped["gang_name"] == selected_gang]
        else:
            daily = d_base[d_base["gang_name"] == selected_gang]
    elif gangs:
        daily = d_scoped[d_scoped["gang_name"].isin(gangs)]
    else:
        daily = d_scoped

    daily = daily.sort_values(["gang_name", "date"])[["date","gang_name","project_name","daily_prod_mt"]]
    if not daily.empty:
        daily = daily.assign(
            date=daily["date"].dt.strftime("%d-%m-%Y"),
            daily_prod_mt=(
                daily["daily_prod_mt"]
                .round(2)
                .map(lambda v: "" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip("."))
            )
        )
    return idle_data, daily.to_dict("records")




from io import BytesIO
from dash.dcc import send_bytes

@app.callback(
    Output("download-trace-xlsx", "data"),
    Input("btn-export-trace", "n_clicks"),
    State("f-project", "value"),
    State("f-month", "value"),
    State("f-quick-range", "value"),
    State("f-gang", "value"),
    State("f-overall-months", "value"),
    State("f-overall-gangs", "value"),
    State("bench", "value"),
    State("trace-gang", "value"),         # NEW
    State("store-selected-gang", "data"), # clicked gang
    prevent_initial_call=True
)
def export_trace(n, projects, months, quick_range, gangs, overall_months_val, overall_gangs_val,
                 bench, trace_gang_value, selected_gang):
    overall_months = "all_months" in (overall_months_val or [])
    overall_gangs  = "all_gangs"  in (overall_gangs_val  or [])

    months_ts = []
    if quick_range:
        start_dt, end_dt = get_quick_date_options()[quick_range]
        months_ts = pd.period_range(start=start_dt, end=end_dt, freq="M").to_timestamp().tolist()
    elif months:
        months_ts = [pd.Period(m, "M").to_timestamp() for m in months]

    d_scoped = apply_filters(df_day, projects or [], months_ts or [], gangs or [],
                             overall_months=overall_months, overall_gangs=overall_gangs)

    # choose gang for dedicated sheet
    gang_for_sheet = trace_gang_value or selected_gang

    def _writer(bio: BytesIO):
        bio.write(make_trace_workbook_bytes(d_scoped, months_ts, projects, gangs, bench,
                                            gang_for_sheet=gang_for_sheet))

    return send_bytes(_writer, "Trace_Calcs.xlsx")



@app.callback(
    Output("trace-gang", "options"),
    Output("trace-gang", "value"),
    Input("f-project", "value"),
    Input("f-month", "value"),
    Input("f-quick-range", "value"),
    Input("f-overall-months", "value"),
    Input("f-overall-gangs", "value"),
    Input("store-selected-gang", "data"),  # clicked gang (if any)
    State("trace-gang", "value"),          # current dropdown value
)
def update_trace_gang_options(projects, months, quick_range,
                              overall_months_val, overall_gangs_val,
                              clicked_gang, current_value):
    overall_months = "all_months" in (overall_months_val or [])

    # resolve months â†’ timestamps
    months_ts = []
    if quick_range:
        start_dt, end_dt = get_quick_date_options()[quick_range]
        months_ts = pd.period_range(start=start_dt, end=end_dt, freq="M").to_timestamp().tolist()
    elif months:
        months_ts = [pd.Period(m, "M").to_timestamp() for m in months]

    # build options from project+period (ignore gang filter)
    d_base = apply_filters(df_day, projects or [], months_ts or [], [],
                           overall_months=overall_months, overall_gangs=True)

    gangs = sorted(d_base["gang_name"].dropna().unique().tolist())
    options = [{"label": g, "value": g} for g in gangs]

    # value priority: clicked gang â†’ keep current if still valid â†’ None
    if clicked_gang and clicked_gang in gangs:
        value = clicked_gang
    elif current_value and current_value in gangs:
        value = current_value
    else:
        value = None

    return options, value





if __name__ == "__main__":
    # For LAN sharing: set host="0.0.0.0"
    app.run_server(host="0.0.0.0", port=8050, debug=False)





