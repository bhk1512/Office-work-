"""Dash callbacks for the productivity dashboard."""

from __future__ import annotations

import hashlib
import logging
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import urllib.parse
import dash_bootstrap_components as dbc 
import pandas as pd
from io import BytesIO
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar, TYPE_CHECKING
from weakref import WeakSet

import dash
import duckdb
import re
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.dependencies import MATCH, ALL
from datetime import datetime
from dash.exceptions import PreventUpdate
from dash.dcc import send_bytes
from uuid import uuid4
from threading import RLock

try:
    from dash import ctx as dash_ctx
except ImportError:  # Dash < 2.6
    dash_ctx = None

try:
    from diskcache import Cache
except Exception:  # pragma: no cover - optional fallback
    Cache = None  # type: ignore[assignment]

from .charts import (
    # create_monthly_line_chart,
    create_project_lines_chart,
    create_top_bottom_gangs_charts,
    build_responsibilities_chart,
    build_empty_responsibilities_figure,
)
from .config import AppConfig, resolve_log_level
from .data_loader import (
    load_stringing_compiled_raw as _load_stringing_compiled_raw,
    _resolve_stringing_microplan_root,
)
from .filters import apply_filters, resolve_months
from .metrics import (
    calc_idle_and_loss,
    calc_idle_and_loss_for_column,
    compute_idle_intervals_per_gang,
    compute_gang_baseline_maps,
    compute_project_baseline_maps,
    compute_project_baseline_maps_for,
)
from .workbook import make_trace_workbook_bytes
from .analytics import (
    build_analytics_payload,
    IDLE_CAP_DAYS,
    MIN_ERECTIONS_FOR_TIERS,
    PRODUCTIVITY_TIER_HIGH,
    PRODUCTIVITY_TIER_LOW,
)
from .stringing_analytics import build_stringing_analytics_payload
from .callback_utils import DataSelector, ResponsibilitiesAccessor, ResponsibilitiesPayload
from .plan_utils import (
    normalize_col_key as _normalize_col_key,
    normalize_location as _normalize_location,
    normalize_lower as _normalize_lower,
    normalize_text as _normalize_text,
    compact_project_key as _compact_project_key,
    infer_project_hint as _infer_project_hint,
    prepare_stringing_plan_frame as _prepare_stringing_plan_frame,
)
from .stringing import expand_stringing_to_daily_payout, build_tse_lookup_from_df


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(resolve_log_level(os.getenv("LOG_LEVEL")))
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.propagate = False
LOGGER.setLevel(resolve_log_level(os.getenv("LOG_LEVEL")))
LOGGER.info("dashboard.callbacks module loaded")

BENCHMARK_MT_PER_DAY = 9.0
_STRINGING_FS_DATE_COLUMNS = ("final_sag_complete", "fs_complete_date", "fs_completed_date", "fs_completion_date")
_STRINGING_PO_DATE_COLUMNS = ("paying_out_complete", "po_completion_date", "po_completion")
_STRINGING_METHODS = ("manual", "tse", "hotline")
BENCHMARK_KM_PER_MONTH = 5.0
_PROJECT_CODE_PATTERN = re.compile(r"(?i)\b([A-Z]{2,4})\s*[-_/ ]*(\d{2,5})\b")
_PROJECT_MODAL_QUERY_PARAM = "projectModal"
CHART_SOURCES = {"g-actual-vs-bench", "g-top5", "g-bottom5"}
GLOBAL_MODAL_CHART_SOURCES = {
    "global-performance-actual-vs-bench",
    "global-performance-top5",
    "global-performance-bottom5",
}
PROJECT_MODAL_CHART_SOURCES = {
    "project-modal-actual-vs-bench",
    "project-modal-top5",
    "project-modal-bottom5",
}

# App-wide config instance for callback logic
config = AppConfig()

T = TypeVar("T")

_SCOPE_CACHE_TTL_SECONDS = 180.0
_SCOPE_CACHE_MAX_ITEMS = 8


@dataclass(slots=True)
class _ScopeCacheEntry:
    frame: pd.DataFrame
    stored_at: float
    aggregates: dict[str, Any] = field(default_factory=dict)


_SCOPE_CACHE: "OrderedDict[str, _ScopeCacheEntry]" = OrderedDict()


@dataclass(slots=True)
class _AggregateCacheEntry:
    value: Any
    stored_at: float


_AGGREGATE_CACHE_TTL_SECONDS = 180.0
_AGGREGATE_CACHE_MAX_ITEMS = 32
_AGGREGATE_CACHE: "OrderedDict[str, _AggregateCacheEntry]" = OrderedDict()
DATA_SELECTOR: DataSelector | None = None
_PROJECT_INFO_PROVIDER: Callable[[], pd.DataFrame] | None = None
_REGISTERED_DASH_APPS: "WeakSet[Dash]" = WeakSet()

_ANALYTICS_CACHE_TTL_SECONDS = 12 * 60 * 60
_ANALYTICS_CACHE_DIR = Path(".") / ".analytics_cache"
_ANALYTICS_CACHE = Cache(str(_ANALYTICS_CACHE_DIR)) if Cache else None
_ANALYTICS_STAMP_CACHE: dict[str, object] = {"stamp": "", "checked_at": 0.0}

_IDLE_INTERVAL_CACHE_TTL_SECONDS = 300.0
_IDLE_INTERVAL_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_IDLE_INTERVAL_PROVIDER: Callable[[], pd.DataFrame] | None = None
_STRINGING_IDLE_INTERVAL_PROVIDER: Callable[[], pd.DataFrame] | None = None

_STRINGING_PLAN_SUMMARY_CACHE_TTL_SECONDS = 300.0
_STRINGING_PLAN_SUMMARY_CACHE: dict[str, Any] = {"frame": None, "stored_at": 0.0}
_STRINGING_PLAN_SUMMARY_PROVIDER: Callable[[], pd.DataFrame] | None = None

_STRINGING_PLAN_CACHE_TTL_SECONDS = 600.0
_STRINGING_PLAN_CACHE: dict[str, Any] = {
    "frame": None,
    "completion": set(),
    "issues": [],
    "index": [],
    "stored_at": 0.0,
    "last_written": 0.0,
}
STRINGING_PLAN_ACCESSOR: ResponsibilitiesAccessor | None = None


if TYPE_CHECKING:
    _get_stringing_tse_lookup: Callable[[], tuple[dict[str, int], dict[str, str]]]


def _get_stringing_tse_lookup() -> tuple[dict[str, int], dict[str, str]]:
    """Fallback stub replaced inside register_callbacks."""
    return {}, {}


def _default_stringing_method_values() -> list[str]:
    return list(_STRINGING_METHODS)


def _method_filters_for_scope(selection: str | None) -> list[str]:
    normalized = _normalize_deployment_filter(selection)
    if normalized == "tse":
        return ["tse"]
    if normalized == "manual":
        return ["manual"]
    if normalized == "hotline":
        return ["hotline"]
    return list(_STRINGING_METHODS)
def _extract_project_code(value: object) -> str:
    """
    Return the canonical project code from a label or identifier.

    The code is derived from the portion before a "CODE : Name" separator and
    then normalized via a regex that captures the alphanumeric code pattern.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if " : " in text:
        text = text.split(" : ", 1)[0].strip()
    match = _PROJECT_CODE_PATTERN.search(text) if _PROJECT_CODE_PATTERN else None
    if match:
        prefix, digits = match.groups()
        return f"{prefix.upper()} {digits}"
    return text


def _idle_table_for_mode(mode: str) -> pd.DataFrame:
    """Return cached idle interval table for the requested mode."""

    normalized = "stringing" if str(mode).strip().lower() == "stringing" else "erection"
    cache_entry = _IDLE_INTERVAL_CACHE.get(normalized)
    ts_now = time.time()
    if cache_entry is not None:
        frame, stored_at = cache_entry
        if ts_now - stored_at < _IDLE_INTERVAL_CACHE_TTL_SECONDS:
            return frame.copy()

    provider = _STRINGING_IDLE_INTERVAL_PROVIDER if normalized == "stringing" else _IDLE_INTERVAL_PROVIDER
    if provider is None:
        return pd.DataFrame()
    try:
        table = provider()
    except Exception:
        LOGGER.warning("Failed to load idle intervals for mode %s", normalized, exc_info=True)
        return pd.DataFrame()
    if not isinstance(table, pd.DataFrame):
        return pd.DataFrame()
    _IDLE_INTERVAL_CACHE[normalized] = (table.copy(), ts_now)
    return table.copy()


def _get_stringing_plan_summary_frame() -> pd.DataFrame:
    """Return the cached stringing plan summary frame supplied by the data store."""

    cache = _STRINGING_PLAN_SUMMARY_CACHE
    ts_now = time.time()
    frame = cache.get("frame")
    stored_at = cache.get("stored_at", 0.0)
    if isinstance(frame, pd.DataFrame) and (ts_now - stored_at < _STRINGING_PLAN_SUMMARY_CACHE_TTL_SECONDS):
        return frame.copy()

    provider = _STRINGING_PLAN_SUMMARY_PROVIDER
    if provider is None:
        return pd.DataFrame()
    try:
        summary = provider()
    except Exception:
        LOGGER.warning("Failed to load stringing plan summary frame", exc_info=True)
        summary = pd.DataFrame()

    if not isinstance(summary, pd.DataFrame):
        cache["frame"] = pd.DataFrame()
        cache["stored_at"] = ts_now
        return pd.DataFrame()

    cache["frame"] = summary.copy()
    cache["stored_at"] = ts_now
    return summary.copy()


def _normalize_month_value(raw: Any) -> tuple[str | None, str | None]:
    """
    Normalize a month selector value (e.g., '2025-10', '2025-10-01') into a
    canonical YYYY-MM string plus a display label 'Oct 2025'.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None, None
    if isinstance(raw, (list, tuple)) and raw:
        # take the first element if a sequence sneaks in
        raw = raw[0]
    text = str(raw).strip()
    if not text:
        return None, None
    try:
        ts = pd.to_datetime(text, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(f"{text}-01", errors="coerce")
        if pd.isna(ts):
            return None, None
        ts = ts.to_period("M").to_timestamp()
        return ts.strftime("%Y-%m"), ts.strftime("%b %Y")
    except Exception:
        return None, None


def _resolve_triggered_id() -> Any:
    """
    Return the ID (string or dict) of the triggering input for the current callback,
    compatible with both legacy dash.callback_context and newer dash.ctx APIs.
    """
    if dash_ctx is not None:
        trig = getattr(dash_ctx, "triggered_id", None)
        if trig is not None:
            return trig
    ctx = dash.callback_context
    triggered = getattr(ctx, "triggered", None)
    if not triggered:
        return None
    raw = triggered[0]["prop_id"].split(".")[0]
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _href_has_project_modal_flag(href: str | None) -> bool:
    """Return True when the provided href contains the modal query flag."""
    if not href:
        return False
    try:
        parsed = urllib.parse.urlsplit(str(href))
    except Exception:
        return False
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    values = params.get(_PROJECT_MODAL_QUERY_PARAM)
    if not values:
        return False
    for value in values:
        normalized = str(value).strip().lower()
        if normalized and normalized not in {"0", "false", "none"}:
            return True
    return False


def _filter_completion_rows(frame: pd.DataFrame, *, completion_column: str) -> pd.DataFrame:
    """Return only the rows whose normalized date matches the completion column."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    if completion_column not in frame.columns or "date" not in frame.columns:
        return pd.DataFrame()
    try:
        completion_norm = pd.to_datetime(frame[completion_column], errors="coerce").dt.normalize()
        date_norm = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    except Exception:
        return pd.DataFrame()
    mask = completion_norm.notna() & date_norm.notna() & (completion_norm == date_norm)
    if not mask.any():
        return pd.DataFrame()
    return frame.loc[mask].copy()


def _sum_completion_totals(
    frame: pd.DataFrame,
    *,
    value_column: str,
    completion_column: str,
    fallback_columns: Sequence[tuple[str, float]] | None = None,
) -> float | None:
    """Sum *value_column* for rows attributed to their completion dates."""
    completion_rows = _filter_completion_rows(frame, completion_column=completion_column)
    if completion_rows.empty:
        return None

    def _coerce_sum(column: str, scale: float = 1.0) -> float | None:
        if column not in completion_rows.columns:
            return None
        series = pd.to_numeric(completion_rows[column], errors="coerce").dropna()
        if series.empty:
            return None
        total_value = float(series.sum()) * scale
        return total_value if not pd.isna(total_value) else None

    total = _coerce_sum(value_column, 1.0)
    if total is not None:
        return total

    for column, scale in fallback_columns or ():
        total = _coerce_sum(column, scale)
        if total is not None:
            return total
    return None


_ERECTIONS_EXPORT_COLUMNS = [
    "completion_date",
    "project_name",
    "location_no",
    "tower_weight_mt",
    "daily_prod_mt",
    "gang_name",
    "start_date",
    "supervisor_name",
    "section_incharge_name",
    "revenue_value",
]


def _parse_completion_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _default_completion_date() -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.Timedelta(days=1)


def _project_filter_candidates(
    project_label: str | None,
    project_code: str | None = None,
) -> list[str]:
    """
    Build a resilient set of candidate project identifiers by combining the
    displayed label (which may include "CODE : Name" formatting) and the canonical
    project code. This helps modal scope filters match the underlying dataset
    regardless of how the project tile renders its title.
    """
    candidates: list[str] = []

    def _add(text: str | None) -> None:
        if not text:
            return
        normalized = text.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    sources = [project_code, project_label]
    for source in sources:
        code_value = _extract_project_code(source)
        if not code_value:
            continue
        _add(code_value)
        flattened = re.sub(r"[^A-Za-z0-9]", "", code_value)
        _add(flattened)
        _add(flattened.upper())
        _add(flattened.lower())

    return candidates


def _match_tile_meta_entry(
    project_label: str | None,
    tile_meta: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Locate the stored tile metadata entry that best matches *project_label*.

    The helper normalizes both the requested label/code and the cached tile
    metadata entries so we can safely resolve selections from the free-form
    project filter back to the same payload used by tile clicks.
    """

    if not project_label or not tile_meta:
        return None

    target_tokens = set(_normalize_str_list(_project_filter_candidates(project_label, project_label), lower=True))
    if not target_tokens:
        return None

    for meta in tile_meta.values():
        if not isinstance(meta, Mapping):
            continue
        meta_label = meta.get("project") or meta.get("display")
        meta_code = meta.get("code")
        meta_tokens = set(_normalize_str_list(_project_filter_candidates(meta_label, meta_code), lower=True))
        if meta_tokens & target_tokens:
            return dict(meta)
    return None


def _format_decimal(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _analytics_data_stamp(data_path: Path) -> str:
    now = time.time()
    cached_stamp = _ANALYTICS_STAMP_CACHE.get("stamp")
    cached_at = _ANALYTICS_STAMP_CACHE.get("checked_at", 0.0)
    if cached_stamp and isinstance(cached_at, (int, float)) and (now - cached_at) < 300:
        return str(cached_stamp)

    stamp = 0.0
    try:
        path = Path(data_path)
        if path.is_file():
            stamp = path.stat().st_mtime
        else:
            candidates: list[Path] = []
            for pattern in ("*.parquet", "*.parq", "*.pq", "*.xlsx", "*.xlsm", "*.xlsb", "*.xls"):
                candidates.extend(path.rglob(pattern))
            if candidates:
                stamp = max(candidate.stat().st_mtime for candidate in candidates)
            elif path.exists():
                stamp = path.stat().st_mtime
    except Exception:
        stamp = 0.0

    _ANALYTICS_STAMP_CACHE["stamp"] = stamp
    _ANALYTICS_STAMP_CACHE["checked_at"] = now
    return str(stamp)


def _analytics_cache_key(
    projects: Sequence[str],
    months: Sequence[pd.Timestamp],
    gangs: Sequence[str],
    data_stamp: str,
) -> str:
    payload = {
        "mode": "erection",
        "bucket_version": "mt_day_v1",
        "projects": sorted({str(value).strip() for value in projects if str(value).strip()}),
        "months": [ts.strftime("%Y-%m") for ts in months if isinstance(ts, pd.Timestamp)],
        "gangs": sorted({str(value).strip() for value in gangs if str(value).strip()}),
        "data_stamp": data_stamp,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _stringing_analytics_cache_key(
    projects: Sequence[str],
    months: Sequence[pd.Timestamp],
    gangs: Sequence[str],
    data_stamp: str,
    compiled_rows: int | None = None,
) -> str:
    payload = {
        "mode": "stringing",
        "projects": sorted({str(value).strip() for value in projects if str(value).strip()}),
        "months": [ts.strftime("%Y-%m") for ts in months if isinstance(ts, pd.Timestamp)],
        "gangs": sorted({str(value).strip() for value in gangs if str(value).strip()}),
        "data_stamp": data_stamp,
        "compiled_rows": int(compiled_rows or 0),
        "version": "v2",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

# --- helper: average days across selected months (fallback 30) ---
def _avg_days_in_selected_months(months_ts) -> float:
    import pandas as pd
    days_factor = 30.0
    try:
        if months_ts:
            month_days = []
            for m in months_ts:
                try:
                    p = m if isinstance(m, pd.Period) else pd.Period(m, freq="M")
                    month_days.append(int(p.days_in_month))
                except Exception:
                    continue
            if month_days:
                days_factor = float(sum(month_days) / len(month_days))
    except Exception:
        pass
    return days_factor


def _prepare_erections_completed(
    scoped: pd.DataFrame,
    *,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    search_text: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scoped.empty or "completion_date" not in scoped.columns:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    working = scoped.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working["completion_date"] = pd.to_datetime(
        working["completion_date"], errors="coerce"
    ).dt.normalize()
    working = working[working["completion_date"].notna()]
    working = working[working["date"] == working["completion_date"]]
    working = working[
        (working["completion_date"] >= range_start)
        & (working["completion_date"] <= range_end)
    ]
    if working.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    working = working.drop_duplicates(
        subset=["project_name", "location_no", "completion_date", "gang_name"]
    ).copy()

    working["location_no_display"] = (
        working["location_no"].map(_normalize_location)
        if "location_no" in working.columns
        else ""
    )
    working["location_no_norm"] = working["location_no_display"]

    working["project_name_display"] = working["project_name"].astype(str).str.strip()
    working["project_name_norm"] = working["project_name_display"].map(_normalize_lower)
    working["gang_name_display"] = working["gang_name"].astype(str).str.strip()

    working["tower_weight_value"] = (
        pd.to_numeric(working["tower_weight"], errors="coerce")
        if "tower_weight" in working.columns
        else pd.Series(np.nan, index=working.index)
    )
    working["daily_prod_value"] = pd.to_numeric(working["daily_prod_mt"], errors="coerce")
    working["start_date_value"] = (
        pd.to_datetime(working["start_date"], errors="coerce").dt.normalize()
        if "start_date" in working.columns
        else pd.Series(pd.NaT, index=working.index)
    )

    supervisor_map: dict[tuple[str, str], str] = {}
    section_map: dict[tuple[str, str], str] = {}
    revenue_map: dict[tuple[str, str], float] = {}

    if responsibilities_provider is not None:
        try:
            resp_source = responsibilities_provider()
        except Exception as exc:
            LOGGER.warning(
                "Erections card: unable to access responsibilities data: %s",
                exc,
            )
            resp_source = pd.DataFrame()
        if resp_source is not None and not resp_source.empty:
            resp = resp_source.copy()
            resp["project_name_norm"] = (
                resp["project_name"].map(_normalize_lower)
                if "project_name" in resp.columns
                else ""
            )
            resp["location_no_norm"] = (
                resp["location_no"].map(_normalize_location)
                if "location_no" in resp.columns
                else ""
            )
            if "entity_name" not in resp.columns:
                resp["entity_name"] = ""
            entity_type_series = (
                resp["entity_type"]
                if "entity_type" in resp.columns
                else pd.Series(["" for _ in range(len(resp))], index=resp.index)
            )
            type_map = {
                "supervisor": "supervisor",
                "supervisors": "supervisor",
                "section incharge": "section incharge",
                "section-incharge": "section incharge",
                "section in-charge": "section incharge",
                "section inch": "section incharge",
            }
            resp["entity_type_norm"] = entity_type_series.map(
                lambda val: type_map.get(_normalize_lower(val), _normalize_lower(val))
            )
            resp["entity_name_norm"] = resp["entity_name"].map(_normalize_text)

            planned_series = (
                pd.to_numeric(resp["revenue_planned"], errors="coerce")
                if "revenue_planned" in resp.columns
                else pd.Series(np.nan, index=resp.index)
            )
            realised_series = (
                pd.to_numeric(resp["revenue_realised"], errors="coerce")
                if "revenue_realised" in resp.columns
                else pd.Series(np.nan, index=resp.index)
            )
            resp["revenue_value"] = realised_series.where(realised_series > 0).fillna(
                planned_series
            )

            def _collapse(series: pd.Series) -> str:
                names = [name for name in series if name]
                return ", ".join(dict.fromkeys(names))

            supervisor_series = (
                resp[resp["entity_type_norm"] == "supervisor"]
                .groupby(["project_name_norm", "location_no_norm"])["entity_name_norm"]
                .apply(_collapse)
            )
            supervisor_map = {
                key: value for key, value in supervisor_series.items() if value
            }

            section_series = (
                resp[resp["entity_type_norm"] == "section incharge"]
                .groupby(["project_name_norm", "location_no_norm"])["entity_name_norm"]
                .apply(_collapse)
            )
            section_map = {
                key: value for key, value in section_series.items() if value
            }

            revenue_series = (
                resp.groupby(["project_name_norm", "location_no_norm"])["revenue_value"].max()
            )
            revenue_map = {
                key: value for key, value in revenue_series.items() if pd.notna(value)
            }

    working["supervisor_name"] = [
        supervisor_map.get((proj, loc), "")
        for proj, loc in zip(
            working["project_name_norm"], working["location_no_norm"]
        )
    ]
    working["section_incharge_name"] = [
        section_map.get((proj, loc), "")
        for proj, loc in zip(
            working["project_name_norm"], working["location_no_norm"]
        )
    ]
    working["revenue_value"] = [
        revenue_map.get((proj, loc), np.nan)
        for proj, loc in zip(
            working["project_name_norm"], working["location_no_norm"]
        )
    ]

    export_df = pd.DataFrame(
        {
            "completion_date": working["completion_date"],
            "project_name": working["project_name_display"],
            "location_no": working["location_no_display"],
            "tower_weight_mt": working["tower_weight_value"],
            "daily_prod_mt": working["daily_prod_value"],
            "gang_name": working["gang_name_display"],
            "start_date": working["start_date_value"],
            "supervisor_name": working["supervisor_name"].fillna(""),
            "section_incharge_name": working["section_incharge_name"].fillna(""),
            "revenue_value": working["revenue_value"],
        }
    )

    if search_text:
        needle = search_text.strip().lower()
        if needle:
            mask = (
                export_df["project_name"].astype(str).str.lower().str.contains(needle, na=False)
                | export_df["location_no"].astype(str).str.lower().str.contains(needle, na=False)
                | export_df["gang_name"].astype(str).str.lower().str.contains(needle, na=False)
            )
            export_df = export_df[mask]

    if export_df.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    export_df = export_df.sort_values(
        ["completion_date", "project_name", "location_no"]
    ).reset_index(drop=True)

    display_df = pd.DataFrame(
        {
            "completion_date": export_df["completion_date"].dt.strftime("%d-%m-%Y").fillna(""),
            "project_name": export_df["project_name"],
            "location_no": export_df["location_no"],
            "tower_weight": export_df["tower_weight_mt"].map(_format_decimal),
            "daily_prod_mt": export_df["daily_prod_mt"].map(_format_decimal),
            "gang_name": export_df["gang_name"],
            "start_date": export_df["start_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else ""),
            "supervisor_name": export_df["supervisor_name"].fillna(""),
            "section_incharge_name": export_df["section_incharge_name"].fillna(""),
            "revenue": export_df["revenue_value"].map(_format_decimal),
        }
    )

    return export_df, display_df

# --- NEW ---
def _prepare_stringing_completed(
    scoped: pd.DataFrame,
    *,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    search_text: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build export/display data for the 'completed' table in STRINGING mode.
    Uses daily_km as the numeric measure (KM/day).
    """
    if scoped.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    working = scoped.copy()

    # date range gate
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"])
    in_range = (working["date"] >= range_start) & (working["date"] <= range_end)
    working = working.loc[in_range].copy()
    if working.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    # normalize display fields
    working["project_name_display"] = working.get("project_name", "").astype(str).str.strip()
    working["gang_name_display"] = working.get("gang_name", "").astype(str).str.strip()
    from_ap = working.get("from_ap", pd.Series([""] * len(working), index=working.index)).astype(str).str.strip()
    to_ap   = working.get("to_ap",   pd.Series([""] * len(working), index=working.index)).astype(str).str.strip()
    working["span_display"] = (from_ap + " \u2192 " + to_ap).str.strip(" \u2192 ")  # From→To

    # search filter (project/gang/from/to)
    if search_text:
        needle = _normalize_lower(search_text)
        mask = (
            working["project_name_display"].map(_normalize_lower).str.contains(needle, na=False)
            | working["gang_name_display"].map(_normalize_lower).str.contains(needle, na=False)
            | from_ap.map(_normalize_lower).str.contains(needle, na=False)
            | to_ap.map(_normalize_lower).str.contains(needle, na=False)
        )
        working = working.loc[mask]

    if working.empty:
        empty = pd.DataFrame(columns=_ERECTIONS_EXPORT_COLUMNS)
        return empty, empty

    # export frame (reuse erection schema so downstream stays happy)
    supervisor_series = working.get("supervisor_name")
    if supervisor_series is None:
        supervisor_series = pd.Series(["NA"] * len(working), index=working.index)
    else:
        supervisor_series = supervisor_series.astype(str).str.strip().replace("", "NA")
    section_series = working.get("section_incharge_name")
    if section_series is None:
        section_series = pd.Series(["NA"] * len(working), index=working.index)
    else:
        section_series = section_series.astype(str).str.strip().replace("", "NA")

    export_df = pd.DataFrame(
        {
            "completion_date": working["date"],
            "project_name": working["project_name_display"],
            "location_no": working["span_display"],  # show From→To in the 'Location' column
            "tower_weight_mt": pd.to_numeric(working.get("daily_km", np.nan), errors="coerce"),  # will display as KM
            "daily_prod_mt":  pd.to_numeric(working.get("daily_km", np.nan), errors="coerce"),  # KM/day
            "gang_name": working["gang_name_display"],
            "start_date": working["date"],           # fallback (F/S start may not be present in daily)
            "supervisor_name": supervisor_series,
            "section_incharge_name": section_series,
            "revenue_value": pd.Series([np.nan] * len(working), index=working.index),
        }
    ).sort_values(["completion_date", "project_name", "gang_name"])

    # display frame for DataTable
    display_df = pd.DataFrame(
        {
            "completion_date": export_df["completion_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else ""),
            "project_name": export_df["project_name"],
            "location_no": export_df["location_no"].fillna(""),
            "tower_weight": export_df["tower_weight_mt"].map(_format_decimal),  # shows numeric as text
            "daily_prod_mt": export_df["daily_prod_mt"].map(_format_decimal),
            "gang_name": export_df["gang_name"],
            "start_date": export_df["start_date"].apply(lambda dt: dt.strftime("%d-%m-%Y") if pd.notna(dt) else "NA"),
            "supervisor_name": export_df["supervisor_name"],
            "section_incharge_name": export_df["section_incharge_name"],
            "revenue": export_df["revenue_value"].map(lambda v: "-" if pd.isna(v) else _format_decimal(v)),
        }
    )

    return export_df, display_df



_slug = lambda s: re.sub(r"[^a-z0-9_-]+", "-", str(s).lower()).strip("-")

def _render_avp_row(
    gang,
    delivered,
    lost,
    total,
    pct,
    avg_prod=0.0,
    baseline=0.0,
    last_project=" ",
    last_date=" ",
    rate_label="MT/day",
    unit_total="MT",
    namespace: str = "avp",
):
    badge_cls = "good" if pct >= 80 else ("mid" if pct >= 65 else "low")
    delivered_pct = 0 if total == 0 else max(0, min(100, (delivered/total)*100))
    lost_pct = 0 if total == 0 else max(0, min(100, (lost/total)*100))

    safe_gang = _slug(gang)
    row_tip_id = f"{namespace}-tip-{safe_gang}"  # STRING id for tooltip target
    row_type = f"{namespace}-row"
    tip_type = f"{namespace}-tip"

    return html.Div(
        id={"type": row_type, "index": gang},   # <-- move pattern id to the row itself
        n_clicks=0,
        style={"cursor": "pointer"},             # (nice to have)
        className="avp-item",
        children=[
            html.Div(
                className="avp-head",
                children=[html.Div(gang, className="avp-name"),
                        html.Div(f"{int(round(pct))}%", className=f"avp-pct {badge_cls}")],
            ),
           
            html.Div(className="avp-track", children=[
                html.Div(className="avp-delivered", style={"width": f"{delivered_pct}%"}),
                html.Div(className="avp-lost", style={"left": f"{delivered_pct}%", "width": f"{lost_pct}%"}),
            ]),
            html.Div(className="avp-meta", children=[
                html.Span(f"{delivered:,.0f} {unit_total} vs {lost:,.0f} {unit_total} lost"),
                html.Div(f"{total:,.0f} {unit_total}", className="text-muted ..."),
            ]),


            html.Div(
                id={"type": tip_type, "index": gang},     # pattern id to allow row-wide click capture
                n_clicks=0,
                className="avp-tip-overlay",
                children=[
                    # string id to attach dbc.Tooltip (fills the overlay)
                    html.Span(id=row_tip_id, className="avp-tip-fill")
                ],
            ),

            dbc.Tooltip(
                [
                    html.Div(html.B(gang)),
                    html.Div(f"Project: {last_project}"),
                    html.Div(f"Last worked at: {last_date}"),
                    html.Div(f"Current {rate_label}: {avg_prod:.2f}"),
                    html.Div(f"Baseline {rate_label}: {baseline:.2f}"),
                ],
                target=row_tip_id,
                placement="right",
                delay={"show": 100, "hide": 100},
            ),
        ],
    )

def _ensure_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_mode(value: str | None) -> str:
    return (value or "erection").strip().lower()


def _compose_modal_mode_payload(mode: str, *, ts: float | None = None) -> dict[str, Any]:
    """Build a consistent payload for the performance mode dcc.Store."""

    normalized = _normalize_mode(mode)
    return {
        "mode": normalized,
        "ts": float(ts if ts is not None else time.time()),
    }


def _modal_mode_from_store(value: Any, default: str = "erection") -> str:
    """Extract the current performance mode from a store payload."""

    fallback = _normalize_mode(default)

    def _coerce(candidate: Any) -> str | None:
        if candidate is None:
            return None
        try:
            normalized = _normalize_mode(str(candidate))
        except Exception:
            return None
        return normalized

    if isinstance(value, Mapping):
        extracted = _coerce(value.get("mode"))
        if extracted:
            return extracted

    if isinstance(value, str):
        extracted = _coerce(value)
        if extracted:
            return extracted
        try:
            decoded = json.loads(value)
        except Exception:
            decoded = None
        if isinstance(decoded, Mapping):
            extracted = _coerce(decoded.get("mode"))
            if extracted:
                return extracted

    return fallback


def _resolve_focus_mode(focus_payload: Any, fallback: str = "erection") -> str:
    """Determine the effective modal mode from the current tile focus payload."""

    normalized_fallback = _normalize_mode(fallback)
    if isinstance(focus_payload, Mapping):
        raw_mode = focus_payload.get("mode") or focus_payload.get("project_mode")
        normalized = _normalize_mode(str(raw_mode)) if raw_mode else None
        if normalized in {"erection", "stringing"}:
            if normalized == "stringing" and not config.enable_stringing:
                return "erection"
            return normalized
    return normalized_fallback


def _normalize_deployment_filter(value: str | None) -> str:
    """Normalize deployment radio selection into the canonical scope key."""
    text = (value or "").strip().lower()
    if text in _STRINGING_METHODS:
        return text
    return "all"


def _normalize_str_list(values: Sequence[Any] | None, *, lower: bool = False) -> list[str]:
    result: list[str] = []
    if not values:
        return result
    for raw in values:
        text = "" if raw is None else str(raw).strip()
        if not text:
            continue
        result.append(text.lower() if lower else text)
    return result


def _completion_range_from_month_selection(
    months_value: Sequence[str] | None,
    quick_range: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Deduce the date window represented by the current month filters so modal
    completion pickers can mirror the home screen selection.
    """

    months_list = _normalize_str_list(_ensure_list(months_value))
    try:
        resolved_months = resolve_months(months_list, quick_range)
    except Exception:
        resolved_months = []

    periods: list[pd.Period] = []
    for raw in resolved_months:
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            continue
        periods.append(ts.to_period("M"))

    if not periods:
        default = _default_completion_date()
        return default, default

    first_period = min(periods)
    last_period = max(periods)
    start = first_period.to_timestamp(how="start").normalize()
    end = last_period.to_timestamp(how="end").normalize()
    return start, end


def _set_scope_cache_entry(key: str, frame: pd.DataFrame) -> None:
    if not key:
        return
    _SCOPE_CACHE[key] = _ScopeCacheEntry(frame.copy(deep=False), time.time())
    _SCOPE_CACHE.move_to_end(key)
    while len(_SCOPE_CACHE) > _SCOPE_CACHE_MAX_ITEMS:
        _SCOPE_CACHE.popitem(last=False)


def _remember_scope_frame(frame: pd.DataFrame) -> str:
    key = uuid4().hex
    _set_scope_cache_entry(key, frame)
    return key


def _get_scope_entry(key: str | None) -> _ScopeCacheEntry | None:
    if not key:
        return None
    entry = _SCOPE_CACHE.get(key)
    if not entry:
        return None
    if _SCOPE_CACHE_TTL_SECONDS > 0 and (time.time() - entry.stored_at) > _SCOPE_CACHE_TTL_SECONDS:
        _SCOPE_CACHE.pop(key, None)
        return None
    return entry


def _get_scope_cache_entry(key: str | None) -> pd.DataFrame | None:
    entry = _get_scope_entry(key)
    if entry is None:
        return None
    return entry.frame.copy(deep=False)


def _cached_scope_result(
    cache_key: str | None,
    token: str,
    producer: Callable[[], T],
    *,
    clone: Callable[[T], T] | None = None,
) -> T:
    entry = _get_scope_entry(cache_key)
    if entry is not None:
        existing = entry.aggregates.get(token)
        if existing is not None:
            return clone(existing) if clone else existing
    result = producer()
    if entry is not None:
        entry.aggregates[token] = result
    return clone(result) if clone else result


def _cached_global_result(
    token: str,
    producer: Callable[[], T],
    *,
    clone: Callable[[T], T] | None = None,
) -> T:
    entry = _AGGREGATE_CACHE.get(token)
    if entry is not None:
        if _AGGREGATE_CACHE_TTL_SECONDS <= 0 or (time.time() - entry.stored_at) <= _AGGREGATE_CACHE_TTL_SECONDS:
            return clone(entry.value) if clone else entry.value
        _AGGREGATE_CACHE.pop(token, None)
        entry = None
    result = producer()
    _AGGREGATE_CACHE[token] = _AggregateCacheEntry(result, time.time())
    _AGGREGATE_CACHE.move_to_end(token)
    while len(_AGGREGATE_CACHE) > _AGGREGATE_CACHE_MAX_ITEMS:
        _AGGREGATE_CACHE.popitem(last=False)
    return clone(result) if clone else result


def _clone_baseline_result(
    result: tuple[Mapping[str, float], Mapping[str, Mapping[pd.Timestamp, float]]],
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    overall, monthly = result
    overall_map = dict(overall or {})
    monthly_map = {gang: dict(month_map or {}) for gang, month_map in (monthly or {}).items()}
    return overall_map, monthly_map


def _clone_loss_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _clone_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def _hash_cache_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _stringing_scope(work: pd.DataFrame, method_values) -> pd.DataFrame:
    work = work.copy()
    method_series = None
    if "method" in work.columns:
        method_series = work["method"].astype(str).str.strip().str.lower()
    elif "method_norm" in work.columns:
        method_series = work["method_norm"].astype(str).str.strip().str.lower()

    method_set = {m.lower() for m in _normalize_str_list(method_values)}
    method_universe = {m.lower() for m in _STRINGING_METHODS}
    if method_series is not None and method_set:
        if method_set == {"tse"}:
            work = work[method_series == "tse"]
        elif method_set == {"hotline"}:
            work = work[method_series == "hotline"]
        elif method_set == {"manual"}:
            work = work[method_series == "manual"]
        elif method_set == method_universe:
            pass
        else:
            work = work[method_series.isin(method_set)]
    elif method_set and not method_universe.issuperset(method_set):
        work = work.iloc[0:0]
    return work


def _reference_frame_for_deployment(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame | None:
    for key in ("full", "project", "month", "project_gang"):
        frame = frames.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return None


def _build_deployment_metadata(reference_frame: pd.DataFrame | None, scope_value: str | None) -> dict[str, Any]:
    selection = _normalize_deployment_filter(scope_value)
    metadata = {"selection": selection, "tse_norm": [], "tse_compact": []}
    if selection == "all":
        return metadata
    tse_norm_map, tse_alias_map = _get_stringing_tse_lookup()
    norm_keys = {key for key, value in tse_norm_map.items() if value}
    compact_keys = {alias for alias, canonical in tse_alias_map.items() if canonical in norm_keys}
    if isinstance(reference_frame, pd.DataFrame) and not reference_frame.empty:
        method_column = None
        for candidate in ("method", "Method"):
            if candidate in reference_frame.columns:
                method_column = candidate
                break
        if method_column:
            mask = reference_frame[method_column].astype(str).str.strip().str.lower() == "tse"
        else:
            mask = pd.Series(False, index=reference_frame.index)
        if mask.any():
            for project_column in (
                "project_name",
                "project",
                "project_name_display",
                "Project Name",
                "project_code",
                "project_key",
            ):
                if project_column not in reference_frame.columns:
                    continue
                values = (
                    reference_frame.loc[mask, project_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                )
                for value in values:
                    norm_key = _normalize_lower(value)
                    if norm_key:
                        norm_keys.add(norm_key)
                    compact_key = _compact_project_key(value)
                    if compact_key:
                        compact_keys.add(compact_key)
    metadata["tse_norm"] = sorted(norm_keys)
    metadata["tse_compact"] = sorted(compact_keys)
    return metadata


def _project_match_mask(
    frame: pd.DataFrame,
    *,
    norm_keys: set[str],
    compact_keys: set[str],
) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(False, index=getattr(frame, "index", pd.Index([])))
    mask = pd.Series(False, index=frame.index)
    if not norm_keys and not compact_keys:
        return mask
    candidate_columns = (
        "project_name",
        "project",
        "project_name_display",
        "Project Name",
        "project_code",
        "project_key",
    )
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.strip()
        normalized = values.map(_normalize_lower)
        if norm_keys:
            mask = mask | normalized.isin(norm_keys)
        if compact_keys:
            compact_series = values.map(_compact_project_key)
            mask = mask | compact_series.isin(compact_keys)
    return mask.fillna(False)


def _filter_frame_with_metadata(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        return frame
    selection = _normalize_deployment_filter((metadata or {}).get("selection"))
    if selection == "all":
        return frame
    norm_keys = set((metadata or {}).get("tse_norm") or [])
    compact_keys = set((metadata or {}).get("tse_compact") or [])
    if not norm_keys and not compact_keys:
        return frame.iloc[0:0].copy() if selection == "tse" else frame.copy()
    mask = _project_match_mask(frame, norm_keys=norm_keys, compact_keys=compact_keys)
    return frame.loc[mask].copy() if selection == "tse" else frame.loc[~mask].copy()


def _filter_frame_for_deployment(frame: pd.DataFrame, selection: str | None) -> pd.DataFrame:
    selection_value = _normalize_deployment_filter(selection)
    if selection_value == "all" or not isinstance(frame, pd.DataFrame):
        return frame
    if "deployment_tse_flag" in frame.columns:
        tse_mask = frame["deployment_tse_flag"].fillna(False).astype(bool)
        filtered = frame.loc[tse_mask].copy() if selection_value == "tse" else frame.loc[~tse_mask].copy()
    else:
        metadata = _build_deployment_metadata(frame, selection_value)
        filtered = _filter_frame_with_metadata(frame, metadata)
    return _filter_frame_by_method(filtered, selection_value)


def _filter_frame_by_method(frame: pd.DataFrame, selection: str | None) -> pd.DataFrame:
    selection_value = _normalize_deployment_filter(selection)
    if selection_value == "all" or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    if "method" in frame.columns:
        method_series = frame["method"].astype(str).str.strip().str.lower()
    elif "method_norm" in frame.columns:
        method_series = frame["method_norm"].astype(str).str.strip().str.lower()
    else:
        return frame
    if selection_value == "tse":
        mask = method_series == "tse"
    elif selection_value == "hotline":
        mask = method_series == "hotline"
    elif selection_value == "manual":
        mask = method_series == "manual"
    else:
        return frame
    if not mask.any():
        return frame.iloc[0:0].copy()
    return frame.loc[mask].copy()


def _build_scope_frames(
    mode_value: str,
    *,
    project_list: Sequence[str],
    gang_list: Sequence[str],
    months_value: Sequence[str],
    quick_range: str | None,
    method_values: Sequence[str] | None,
    deployment_filter: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp], float]:
    selector = DATA_SELECTOR
    if selector is None:
        raise RuntimeError("Data selector not initialized.")
    eff_mode = _normalize_mode(mode_value)
    normalized_months = resolve_months(months_value, quick_range)
    method_set = {value.strip().lower() for value in _normalize_str_list(method_values)}

    scoped_frames = selector.scopes_for(
        eff_mode,
        months=normalized_months,
        projects=project_list,
        gangs=gang_list,
        method_filter=method_set if eff_mode == "stringing" else None,
    )

    if scoped_frames is None:
        df_day = selector.select(eff_mode)
        if not isinstance(df_day, pd.DataFrame) or df_day.empty:
            empty = pd.DataFrame()
            return (
                {"month": empty, "project": empty, "full": empty, "project_gang": empty},
                [],
                30.0,
            )

        work = df_day.copy()
        if eff_mode == "stringing":
            work = _stringing_scope(work, method_values)

        month_scope = apply_filters(work, [], normalized_months, [])
        project_scope = apply_filters(work, project_list, normalized_months, [])
        full_scope = apply_filters(work, project_list, normalized_months, gang_list)
        project_gang_scope = apply_filters(work, project_list, [], gang_list)
    else:
        month_scope = scoped_frames.get("month", pd.DataFrame()).copy()
        project_scope = scoped_frames.get("project", pd.DataFrame()).copy()
        full_scope = scoped_frames.get("full", pd.DataFrame()).copy()
        project_gang_scope = scoped_frames.get("project_gang", pd.DataFrame()).copy()

    frames = {
        "month": month_scope,
        "project": project_scope,
        "full": full_scope,
        "project_gang": project_gang_scope,
    }
    if eff_mode == "stringing":
        frames = {
            name: _filter_frame_for_deployment(frame, deployment_filter)
            for name, frame in frames.items()
        }

    return (
        frames,
        normalized_months,
        _avg_days_in_selected_months(normalized_months),
    )


def _build_scope_meta_payload(
    *,
    eff_mode: str,
    project_list: list[str],
    gang_list: list[str],
    months_list: list[str],
    quick_range: str | None,
    method_values: Sequence[str] | None,
    method_list: list[str],
    deployment_filter: str | None = None,
) -> dict[str, Any]:
    normalized_scope = _normalize_deployment_filter(deployment_filter)
    frames, months_ts, days_factor = _build_scope_frames(
        eff_mode,
        project_list=project_list,
        gang_list=gang_list,
        months_value=months_list,
        quick_range=quick_range,
        method_values=method_values,
        deployment_filter=normalized_scope,
    )
    scope_keys = {name: _remember_scope_frame(frame) for name, frame in frames.items()}
    rows_meta = {name: int(len(frame.index)) for name, frame in frames.items()}
    signature_payload = {
        "mode": eff_mode,
        "projects": project_list,
        "gangs": gang_list,
        "months": months_list,
        "quick_range": quick_range,
        "methods": method_list,
        "stringing_scope": normalized_scope,
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()

    return {
        "mode": eff_mode,
        "signature": signature,
        "scopes": scope_keys,
        "rows": rows_meta,
        "days_factor": days_factor,
        "months_iso": [ts.isoformat() for ts in months_ts],
        "selected": {
            "projects": project_list,
            "gangs": gang_list,
            "months": months_list,
            "quick_range": quick_range,
            "methods": method_list,
            "stringing_scope": normalized_scope,
        },
    }


def _normalize_min_erections(value: Any) -> int | None:
    """
    Convert the UI value for the minimum erections filter into a non-negative integer.
    Returns None when the filter should be ignored.
    """

    if isinstance(value, (int, float)) and not pd.isna(value):
        candidate = int(value)
    elif isinstance(value, str):
        candidate_str = value.strip()
        if not candidate_str:
            return None
        try:
            candidate = int(float(candidate_str))
        except Exception:
            return None
    else:
        return None
    if candidate < 0:
        return None
    return candidate


def _min_erections_from_meta(meta: Mapping[str, Any] | None) -> int | None:
    """Extract the normalized minimum erections filter value from the scope metadata."""

    if not isinstance(meta, Mapping):
        return None
    if "min_erections" in meta:
        return _normalize_min_erections(meta.get("min_erections"))
    selected = meta.get("selected")
    if isinstance(selected, Mapping) and "min_erections" in selected:
        return _normalize_min_erections(selected.get("min_erections"))
    return None


def _filter_frame_for_min_erections(frame: pd.DataFrame, min_erections: Any) -> pd.DataFrame:
    """
    Limit the provided frame to gangs whose completed erections exceed the threshold.

    The helper gracefully returns the original frame when the filter is inactive or the
    frame lacks gang level information.
    """

    threshold = _normalize_min_erections(min_erections)
    if threshold is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    if "gang_name" not in frame.columns:
        return frame
    work = frame.copy()
    if "location_no" in work.columns:
        counts = (
            work.groupby("gang_name")["location_no"]
            .nunique(dropna=True)
            .rename("erections")
        )
    else:
        counts = work.groupby("gang_name").size()
    keep_gangs = counts[counts > threshold].index
    if len(keep_gangs) == 0:
        return work.iloc[0:0].copy()
    return work[work["gang_name"].isin(keep_gangs)].copy()


def _scope_meta_with_min_erections(
    scope_meta: dict[str, Any] | None,
    min_erections: Any,
) -> dict[str, Any]:
    """
    Return a shallow copy of *scope_meta* whose scope frames honor the min erections filter.
    """

    threshold = _normalize_min_erections(min_erections)
    if threshold is None or not isinstance(scope_meta, dict):
        return scope_meta
    scopes = scope_meta.get("scopes")
    if not isinstance(scopes, Mapping):
        return scope_meta
    filtered_scope_keys: dict[str, str] = {}
    filtered_rows = dict(scope_meta.get("rows", {}))
    for name in scopes:
        frame = _scope_frame_from_store(scope_meta, name).copy()
        filtered_frame = _filter_frame_for_min_erections(frame, threshold)
        filtered_scope_keys[name] = _remember_scope_frame(filtered_frame)
        filtered_rows[name] = int(len(filtered_frame.index))
    meta_copy = dict(scope_meta)
    meta_copy["scopes"] = filtered_scope_keys
    meta_copy["rows"] = filtered_rows
    meta_copy["min_erections"] = threshold
    selected = dict(meta_copy.get("selected") or {})
    selected["min_erections"] = threshold
    meta_copy["selected"] = selected
    return meta_copy


def _repopulate_scopes_from_meta(meta: dict[str, Any]) -> dict[str, pd.DataFrame]:
    selected = meta.get("selected") or {}
    normalized_scope = _normalize_deployment_filter(selected.get("stringing_scope"))
    scopes, _, _ = _build_scope_frames(
        _normalize_mode(meta.get("mode")),
        project_list=selected.get("projects", []),
        gang_list=selected.get("gangs", []),
        months_value=selected.get("months", []),
        quick_range=selected.get("quick_range"),
        method_values=selected.get("methods", []),
        deployment_filter=normalized_scope,
    )
    for name, frame in scopes.items():
        cache_key = (meta.get("scopes") or {}).get(name)
        if cache_key:
            _set_scope_cache_entry(cache_key, frame)
    return scopes


def _build_project_scope_meta(
    project_label: str | None,
    project_code: str | None,
    eff_mode: str,
    months_value,
    quick_range,
    gang_values,
    method_values: Sequence[str] | None,
    deployment_filter: str | None,
) -> dict[str, Any]:
    """
    Build the cached scope payload used by the project modal. The helper mirrors
    `_build_scope_meta_payload` but always narrows the project list to variations
    of the selected tile so callbacks can consistently rehydrate scope frames.
    """

    mode_normalized = _normalize_mode(eff_mode)
    project_filters = _project_filter_candidates(project_label, project_code)
    if not project_filters:
        fallback = project_label or project_code
        if fallback:
            project_filters = [fallback]

    gang_list = _normalize_str_list(_ensure_list(gang_values))
    months_list = _normalize_str_list(_ensure_list(months_value))
    method_list = _normalize_str_list(_ensure_list(method_values), lower=True)

    return _build_scope_meta_payload(
        eff_mode=mode_normalized,
        project_list=project_filters,
        gang_list=gang_list,
        months_list=months_list,
        quick_range=quick_range,
        method_values=method_values,
        method_list=method_list,
        deployment_filter=deployment_filter,
    )


def _build_avp_rows(
    scope_df: pd.DataFrame,
    *,
    namespace: str,
    metric: str,
    loss_rows: Sequence[Mapping[str, Any]] | None = None,
    unit_label: str | None = None,
    rate_label: str = "MT/day",
) -> list[Any]:
    """
    Build the gang-level AVP rows rendered next to the charts so clicks/hover
    interactions continue to work. Each row exposes pattern-matching IDs used by
    the existing clientside callbacks.
    """

    if scope_df.empty or "gang_name" not in scope_df.columns:
        scope_df = pd.DataFrame(columns=["gang_name"])

    # Build lookup for last project/date per gang to populate tooltips.
    last_meta: dict[str, dict[str, str]] = {}
    if not scope_df.empty and {"gang_name", "project_name"}.issubset(scope_df.columns):
        meta_cols = ["gang_name", "project_name"]
        if "date" in scope_df.columns:
            meta_cols.append("date")
        elif "month" in scope_df.columns:
            meta_cols.append("month")
        meta = scope_df[meta_cols].copy()
        time_col = None
        if "date" in meta.columns:
            time_col = "__ts"
            meta[time_col] = pd.to_datetime(meta["date"], errors="coerce")
        elif "month" in meta.columns:
            time_col = "__ts"
            meta[time_col] = pd.to_datetime(meta["month"], errors="coerce")
        if time_col is None:
            meta["__ts"] = pd.Timestamp.now()
            time_col = "__ts"
        last_rows = (
            meta.dropna(subset=["gang_name"])
            .assign(gang_name=lambda df: df["gang_name"].astype(str).str.strip())
        )
        last_rows = last_rows[last_rows["gang_name"] != ""]
        if not last_rows.empty:
            last_rows = (
                last_rows.sort_values(time_col)
                .groupby("gang_name")
                .tail(1)[["gang_name", "project_name", time_col]]
            )
            for _, record in last_rows.iterrows():
                gang = record["gang_name"]
                last_project = str(record.get("project_name") or "-")
                ts_val = record.get(time_col)
                if pd.notna(ts_val):
                    last_date = pd.to_datetime(ts_val).strftime("%d-%b-%Y")
                else:
                    last_date = "-"
                last_meta[gang] = {"project": last_project, "date": last_date}

    unit_total = unit_label or ("KM" if metric == "stringing" else "MT")

    rows: list[Any] = []
    if loss_rows:
        df_loss = pd.DataFrame(list(loss_rows))
        if not df_loss.empty and "gang_name" in df_loss.columns:
            df_loss = df_loss.dropna(subset=["gang_name"])
            df_loss["gang_name"] = df_loss["gang_name"].astype(str).str.strip()
            df_loss = df_loss[df_loss["gang_name"] != ""]
            for column in ("delivered", "lost", "potential", "baseline", "avg_prod"):
                if column in df_loss.columns:
                    df_loss[column] = pd.to_numeric(df_loss[column], errors="coerce").fillna(0.0)
            if "potential" not in df_loss.columns:
                df_loss["potential"] = df_loss["delivered"] + df_loss["lost"]
            df_loss = df_loss.sort_values("potential", ascending=False).head(15)
            for _, record in df_loss.iterrows():
                gang_name = record["gang_name"]
                delivered = float(record.get("delivered") or 0.0)
                lost = float(record.get("lost") or 0.0)
                total = float(record.get("potential") or (delivered + lost))
                avg_prod = float(record.get("avg_prod") or 0.0)
                baseline = float(record.get("baseline") or 0.0)
                pct = 0.0 if total <= 0 else (delivered / total) * 100.0
                meta_info = last_meta.get(gang_name, {})
                last_project = meta_info.get("project", "-")
                last_date = meta_info.get("date", "-")
                rows.append(
                    _render_avp_row(
                        gang_name,
                        delivered,
                        lost,
                        total,
                        pct,
                        avg_prod=avg_prod,
                        baseline=baseline,
                        last_project=last_project,
                        last_date=last_date,
                        rate_label=rate_label,
                        unit_total=unit_total,
                        namespace=namespace,
                    )
                )
    if rows:
        return rows

    work = scope_df.copy()
    work = work.dropna(subset=["gang_name"])
    work["gang_name"] = work["gang_name"].astype(str).str.strip()
    work = work[work["gang_name"] != ""]
    if work.empty:
        return [html.Div("No gangs available for this selection.", className="text-muted")]

    if metric == "erection":
        grouped = (
            work.groupby("gang_name", as_index=False)["daily_prod_mt"]
            .sum()
            .rename(columns={"daily_prod_mt": "metric_value"})
        )
        metric_label = "Total MT"
    else:
        grouped = (
            work.groupby("gang_name", as_index=False)["daily_prod_mt"]
            .mean()
            .rename(columns={"daily_prod_mt": "metric_value"})
        )
        metric_label = "Avg MT/day"

    if "date" in work.columns:
        work["__date"] = pd.to_datetime(work["date"], errors="coerce")
        last_activity = (
            work.sort_values("__date")
            .groupby("gang_name")
            .tail(1)[["gang_name", "project_name", "__date"]]
            .rename(columns={"project_name": "last_project", "__date": "last_date"})
        )
        grouped = grouped.merge(last_activity, on="gang_name", how="left")
    else:
        grouped["last_project"] = ""
        grouped["last_date"] = pd.NaT

    grouped = grouped.sort_values("metric_value", ascending=False).head(10)

    rows = []
    for _, row in grouped.iterrows():
        gang_name = row["gang_name"]
        project_name = row.get("last_project") or "-"
        last_date = row.get("last_date")
        if pd.notna(last_date):
            last_date_label = pd.to_datetime(last_date).strftime("%d-%b-%Y")
        else:
            last_date_label = "-"
        metric_value = row["metric_value"]
        row_id = {"type": f"{namespace}-row", "index": gang_name}
        tip_id = {"type": f"{namespace}-tip", "index": gang_name}
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(gang_name, className="avp-gang"),
                            html.Div(f"{metric_value:.2f} {metric_label}", className="avp-metric"),
                            html.Div(f"Last project: {project_name}", className="avp-meta"),
                            html.Div(f"Last worked: {last_date_label}", className="avp-meta"),
                        ],
                        className="avp-row-body",
                    ),
                    html.Div(
                        html.Span(className="avp-tip-fill"),
                        id=tip_id,
                        className="avp-tip-overlay",
                        n_clicks=0,
                        n_clicks_timestamp=0,
                    ),
                ],
                id=row_id,
                className="avp-row",
                n_clicks=0,
                n_clicks_timestamp=0,
            )
        )

    return rows


def _build_loss_vs_potential_figure(
    loss_rows: Sequence[Mapping[str, Any]] | None,
    *,
    unit_label: str,
) -> go.Figure:
    """Return the delivered vs lost stacked bar chart for gang performance."""

    base_layout = dict(
        height=260,
        margin=dict(l=36, r=16, t=10, b=50),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
    )

    df = pd.DataFrame(loss_rows or [])
    required_cols = {"gang_name", "delivered", "lost"}
    if df.empty or not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(**base_layout)
        fig.add_annotation(
            text="No gang performance data for this selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#6b7280"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    df = df.copy()
    df["gang_name"] = df["gang_name"].astype(str).str.strip()
    df = df[df["gang_name"] != ""]
    for column in ("delivered", "lost", "potential", "baseline", "avg_prod"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "potential" not in df.columns:
        df["potential"] = df["delivered"] + df["lost"]

    df = df.sort_values("potential", ascending=False).head(10)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout)
        fig.add_annotation(
            text="No gang performance data for this selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#6b7280"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    hover_fields = [
        df.get("potential", pd.Series([0.0] * len(df), index=df.index)),
        df.get("baseline", pd.Series([0.0] * len(df), index=df.index)),
        df.get("avg_prod", pd.Series([0.0] * len(df), index=df.index)),
    ]
    customdata = np.stack([series.to_numpy(dtype=float) for series in hover_fields], axis=-1)

    delivered_color = "#22C55E"
    lost_color = "#EF4444"
    unit_text = unit_label or "MT"

    fig = go.Figure()
    fig.add_bar(
        x=df["gang_name"],
        y=df["delivered"],
        name="Delivered Output",
        marker_color=delivered_color,
        customdata=customdata,
        hovertemplate=(
            "<b>%{x}</b><br>"
            + f"Delivered: %{{y:,.1f}} {unit_text}<br>"
            + f"Lost: %{{customdata[0]-y:,.1f}} {unit_text}<br>"
            + "Baseline: %{customdata[1]:,.2f}<br>"
            + "Avg Output: %{customdata[2]:,.2f}<extra></extra>"
        ),
    )
    fig.add_bar(
        x=df["gang_name"],
        y=df["lost"],
        name="Lost Potential",
        marker_color=lost_color,
        customdata=customdata,
        hovertemplate=(
            "<b>%{x}</b><br>"
            + f"Lost: %{{y:,.1f}} {unit_text}<br>"
            + f"Delivered: %{{customdata[0]-y:,.1f}} {unit_text}<extra></extra>"
        ),
    )

    fig.update_layout(
        **base_layout,
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(tickangle=-15, automargin=True, showspikes=False)
    fig.update_yaxes(title=f"{unit_text}", gridcolor="#e5e7eb", zeroline=False, showspikes=False)
    return fig


def _compute_dashboard_outputs(
    scope_meta: dict[str, Any] | None,
    metric: str | None,
    *,
    avp_namespace: str = "avp-row",
    summarizer: Callable[[dict[str, Any] | None], dict[str, str] | None] | None = None,
    summary_factory: Callable[[bool], dict[str, str]] | None = None,
    min_erections: int | None = None,
) -> tuple[Any, ...]:
    """
    Compute the tuple consumed by the project modal performance block.
    The helper returns KPI text plus the AVP rows and three Plotly figures.
    """

    if not isinstance(scope_meta, dict):
        raise PreventUpdate
    min_erections_value = _normalize_min_erections(min_erections)
    if min_erections_value is not None:
        scope_meta = _scope_meta_with_min_erections(scope_meta, min_erections_value)

    def _fallback_summary_payload(is_stringing: bool) -> dict[str, str]:
        payload = {
            "projects": "-",
            "totals": "-",
            "gangs": "-",
            "productivity": "-",
            "lost_units": "-",
        }
        if is_stringing:
            payload["po_completion"] = "-"
            payload["tse"] = "-"
        payload["_meta"] = {
            "loss_rows": [],
            "unit_label": "KM" if is_stringing else "MT",
            "mode": "stringing" if is_stringing else "erection",
            "is_stringing": is_stringing,
        }
        return payload

    summary_fn = summarizer or globals().get("_summarize_scope_for_cards")

    if not callable(summary_fn):
        def _empty_summarizer(_: dict[str, Any] | None) -> dict[str, str] | None:
            return None

        summary_fn = _empty_summarizer

    summary_factory_fn = summary_factory or globals().get("_empty_summary_payload") or _fallback_summary_payload
    if not callable(summary_factory_fn):
        summary_factory_fn = _fallback_summary_payload

    mode_normalized = _normalize_mode((scope_meta or {}).get("mode"))
    summary = summary_fn(scope_meta) or summary_factory_fn(mode_normalized == "stringing")
    summary_meta = summary.get("_meta", {}) if isinstance(summary, dict) else {}
    loss_rows = summary_meta.get("loss_rows") or []
    unit_label = summary_meta.get("unit_label") or ("KM" if mode_normalized == "stringing" else "MT")
    is_stringing_mode = bool(summary_meta.get("is_stringing", mode_normalized == "stringing"))
    rate_label = "KM/month" if is_stringing_mode else "MT/day"
    fig_loss = _build_loss_vs_potential_figure(loss_rows, unit_label=unit_label)
    project_scope = _scope_frame_from_store(scope_meta, "project").copy()
    if project_scope.empty or "gang_name" not in project_scope.columns:
        raise PreventUpdate

    if "daily_prod_mt" not in project_scope.columns:
        alt_series = None
        if is_stringing_mode and "daily_km" in project_scope.columns:
            alt_series = project_scope["daily_km"]
        elif "daily_prod_value" in project_scope.columns:
            alt_series = project_scope["daily_prod_value"]
        if alt_series is not None:
            project_scope["daily_prod_mt"] = pd.to_numeric(alt_series, errors="coerce")
        else:
            project_scope["daily_prod_mt"] = np.nan
    else:
        project_scope["daily_prod_mt"] = pd.to_numeric(project_scope["daily_prod_mt"], errors="coerce")

    metric_key = (metric or "prod").strip().lower()
    if metric_key not in {"prod", "erection"}:
        metric_key = "prod"

    # Ensure months are datetime for project lines chart
    if "month" in project_scope.columns:
        project_scope["month"] = pd.to_datetime(project_scope["month"], errors="coerce")

    fig_top, fig_bottom = create_top_bottom_gangs_charts(
        project_scope,
        metric=metric_key,
        baseline_map=None,
        is_stringing=is_stringing_mode,
    )

    selected_projects = (scope_meta.get("selected") or {}).get("projects") or []
    monthly_scope = project_scope[["month", "project_name", "daily_prod_mt"]].dropna(subset=["month", "project_name"])
    avg_line = None
    if not monthly_scope.empty:
        avg_line = float(pd.to_numeric(monthly_scope["daily_prod_mt"], errors="coerce").dropna().mean())
    fig_lines = create_project_lines_chart(
        monthly_scope,
        selected_projects=selected_projects,
        avg_line=avg_line,
    )

    avp_children = _build_avp_rows(
        project_scope,
        namespace=avp_namespace,
        metric=metric_key,
        loss_rows=loss_rows,
        unit_label=unit_label,
        rate_label=rate_label,
    )

    # Returning KPIs keeps tuple compatibility even though modal path ignores them.
    return (
        summary.get("projects", "-"),
        summary.get("totals", "-"),
        summary.get("gangs", "-"),
        summary.get("productivity", "-"),
        avp_children,
        fig_loss,
        fig_top,
        fig_bottom,
        fig_lines,
    )


def _scope_frame_from_store(meta: dict[str, Any] | None, scope_name: str) -> pd.DataFrame:
    if not isinstance(meta, dict):
        return pd.DataFrame()
    cache_key = (meta.get("scopes") or {}).get(scope_name)
    frame = _get_scope_cache_entry(cache_key)
    if frame is not None:
        return frame
    rebuilt = _repopulate_scopes_from_meta(meta)
    return rebuilt.get(scope_name, pd.DataFrame())


def _months_from_meta(meta: dict[str, Any] | None) -> list[pd.Timestamp]:
    raw_months = (meta or {}).get("months_iso") or []
    months: list[pd.Timestamp] = []
    for raw in raw_months:
        try:
            months.append(pd.to_datetime(raw))
        except Exception:
            continue
    return months


def _format_period_label(months: Sequence[pd.Timestamp]) -> str:
    if not months:
        return "(All periods)"
    periods = sorted({pd.Period(ts, "M") for ts in months})
    labels = [period.strftime("%b %Y") for period in periods]
    if len(periods) == 1:
        return f"({labels[0]})"
    if all(periods[i] + 1 == periods[i + 1] for i in range(len(periods) - 1)):
        return f"({labels[0]} - {labels[-1]})"
    if len(labels) <= 3:
        return "(" + ", ".join(labels) + ")"
    return "(" + ", ".join(labels[:3]) + ", ...)"


def register_callbacks(
    app: Dash,
    data_provider: Callable[[], pd.DataFrame],
    config: AppConfig,
    *,
    duckdb_connection: duckdb.DuckDBPyConnection | None = None,
    duckdb_lock: RLock | None = None,
    stringing_data_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_compiled_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_tse_lookup_provider: Callable[[], tuple[dict[str, int], dict[str, str]]] | None = None,
    idle_interval_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_idle_interval_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_plan_summary_provider: Callable[[], pd.DataFrame] | None = None,
    project_info_provider: Callable[[], pd.DataFrame] | None = None,
    project_baseline_provider: Callable[[], tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]] | None = None,
    responsibilities_provider: Callable[[], pd.DataFrame] | None = None,
    responsibilities_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    responsibilities_error_provider: Callable[[], str | None] | None = None,
    stringing_plan_provider: Callable[[], pd.DataFrame] | None = None,
    stringing_plan_completion_provider: Callable[[], set[tuple[str, str]]] | None = None,
    stringing_plan_error_provider: Callable[[], str | None] | None = None,
    responsibility_slice_provider: Callable[[str, Sequence[str], str], pd.DataFrame] | None = None,
    responsibility_completion_lookup_provider: Callable[[str], set[tuple[str, str]]] | None = None,
) -> None:

    LOGGER.debug("Registering callbacks")

    if config.enable_stringing and stringing_data_provider is None:
        raise RuntimeError("Stringing data provider must be supplied when stringing support is enabled.")

    if app in _REGISTERED_DASH_APPS:
        LOGGER.warning("register_callbacks invoked multiple times for the same Dash app; skipping duplicate registration.")
        return
    _REGISTERED_DASH_APPS.add(app)

    data_selector = DataSelector(
        config=config,
        data_provider=data_provider,
        stringing_provider=stringing_data_provider,
        duckdb_connection=duckdb_connection,
        duckdb_lock=duckdb_lock,
        logger=LOGGER,
    )
    global DATA_SELECTOR, _PROJECT_INFO_PROVIDER
    global _IDLE_INTERVAL_PROVIDER, _STRINGING_IDLE_INTERVAL_PROVIDER, _STRINGING_PLAN_SUMMARY_PROVIDER
    DATA_SELECTOR = data_selector
    _PROJECT_INFO_PROVIDER = project_info_provider
    _IDLE_INTERVAL_PROVIDER = idle_interval_provider
    _STRINGING_IDLE_INTERVAL_PROVIDER = stringing_idle_interval_provider
    _STRINGING_PLAN_SUMMARY_PROVIDER = stringing_plan_summary_provider
    _IDLE_INTERVAL_CACHE.clear()
    _STRINGING_PLAN_SUMMARY_CACHE["frame"] = None
    _STRINGING_PLAN_SUMMARY_CACHE["stored_at"] = 0.0
    responsibilities_accessor = ResponsibilitiesAccessor(
        data_provider=responsibilities_provider,
        completion_provider=responsibilities_completion_provider,
        error_provider=responsibilities_error_provider,
        logger=LOGGER,
    )
    stringing_plan_accessor = ResponsibilitiesAccessor(
        data_provider=stringing_plan_provider,
        completion_provider=stringing_plan_completion_provider,
        error_provider=stringing_plan_error_provider,
        logger=LOGGER,
    )
    global STRINGING_PLAN_ACCESSOR
    STRINGING_PLAN_ACCESSOR = stringing_plan_accessor
    plan_accessors: dict[str, ResponsibilitiesAccessor] = {
        "erection": responsibilities_accessor,
        "stringing": stringing_plan_accessor,
    }
    has_plan_provider = {
        "erection": callable(responsibilities_provider),
        "stringing": callable(stringing_plan_provider),
    }

    def _resolve_monthly_plan_workbook_path(cfg: AppConfig, plan_mode: str) -> Path | None:
        normalized = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        root = Path(cfg.data_path).expanduser()
        candidates: list[Path] = []

        def _push(path: Path | str | None) -> None:
            if not path:
                return
            candidate = Path(path).expanduser()
            if candidate not in candidates:
                candidates.append(candidate)

        if root.is_file():
            _push(root)
        else:
            if normalized == "erection":
                _push(root / "ErectionCompiled_Output.xlsx")
                _push(root / "MicroPlanCompiled_Output.xlsx")
                _push(root / "ErectionCompiled.xlsx")
            else:
                _push(root / "StringingCompiled_Output.xlsx")
                _push(root / "StringingCompiled.xlsx")
                _push(root / "Stringing Compiled.xlsx")

        repo_root = Path(".").expanduser().resolve()
        if normalized == "erection":
            _push(repo_root / "Parquets" / "Erection" / "ErectionCompiled_Output.xlsx")
        else:
            # Sibling/specified dirs containing compiled stringing excel
            potential_roots: list[Path] = []
            potential_roots.append(root)
            potential_roots.append(root.parent / "Stringing")
            for rel in getattr(cfg, "stringing_parquet_dirs", ()):
                try:
                    rel_path = (root / Path(rel)).resolve()
                except Exception:
                    continue
                potential_roots.append(rel_path)
            potential_roots.append(repo_root / "Parquets" / "Stringing")
            for base in potential_roots:
                if not isinstance(base, Path):
                    continue
                _push(base / "StringingCompiled_Output.xlsx")
                _push(base / "StringingCompiled.xlsx")
                _push(base / "Stringing Compiled.xlsx")

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except PermissionError:
                continue
        return None

    def _fetch_monthly_plan(
        plan_mode: str = "erection",
        *,
        allow_workbook_fallback: bool = False,
    ) -> tuple[pd.DataFrame | None, set[tuple[str, str]], str | None, pd.ExcelFile | None]:
        mode_key = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        accessor = plan_accessors[mode_key]
        payload: ResponsibilitiesPayload = accessor.load()
        if payload.has_frame:
            frame = payload.frame.copy() if payload.frame is not None else None
            completion_keys = set(payload.completion_keys or set())
            return frame, completion_keys, payload.error, None

        completion_keys = set(payload.completion_keys or set())
        if mode_key == "stringing":
            plan_frame, _plan_keys, _plan_issues, _plan_index = _load_stringing_plan_snapshot(config)
            if isinstance(plan_frame, pd.DataFrame) and not plan_frame.empty:
                return plan_frame.copy(), completion_keys, payload.error, None
        load_error = payload.error
        if allow_workbook_fallback and not has_plan_provider.get(mode_key, False):
            cfg = config
            workbook_path = _resolve_monthly_plan_workbook_path(cfg, mode_key)
            plan_title = "Stringing plan" if mode_key == "stringing" else "Micro Plan"
            if workbook_path is None:
                LOGGER.warning(
                    "Monthly plan workbook for '%s' not found near data root '%s'.",
                    mode_key,
                    cfg.data_path,
                )
                return None, completion_keys, f"No {plan_title} data found in the compiled workbook.", None
            try:
                workbook = pd.ExcelFile(workbook_path)
            except FileNotFoundError:
                LOGGER.warning("Monthly plan workbook not found: %s", workbook_path)
                return None, completion_keys, "Compiled workbook not found.", None
            except Exception as exc:
                LOGGER.exception("Failed to open monthly plan workbook '%s': %s", workbook_path, exc)
                return None, completion_keys, "Unable to load monthly plan data.", None

            if mode_key == "stringing":
                preferred_sheet = getattr(cfg, "stringing_sheet_name", "") or "Stringing Compiled"
                sheet_name = next(
                    (
                        name
                        for name in workbook.sheet_names
                        if _normalize_col_key(name) == _normalize_col_key(preferred_sheet)
                    ),
                    None,
                )
                if not sheet_name:
                    LOGGER.warning("Stringing sheet missing in workbook '%s'; sheets=%s", workbook_path, workbook.sheet_names)
                    return (
                        None,
                        completion_keys,
                        "No Stringing plan data found in the compiled workbook.",
                        workbook,
                    )
            else:
                sheet_name = "MicroPlanResponsibilities"
                if sheet_name not in workbook.sheet_names:
                    LOGGER.warning("Sheet '%s' missing in workbook '%s'", sheet_name, workbook_path)
                    return (
                        None,
                        completion_keys,
                        "No Micro Plan data found in the compiled workbook.",
                        workbook,
                    )

            try:
                df_atomic = pd.read_excel(workbook, sheet_name=sheet_name)
            except Exception as exc:
                LOGGER.exception("Failed to load sheet '%s' for %s plan: %s", sheet_name, mode_key, exc)
                message = (
                    "Unable to load Stringing plan data."
                    if mode_key == "stringing"
                    else "Unable to load Micro Plan data."
                )
                return None, completion_keys, message, workbook
            try:
                setattr(workbook, "_plan_sheet_name", sheet_name)
                setattr(workbook, "_plan_workbook_path", Path(workbook_path))
            except Exception:
                pass
            return df_atomic, completion_keys, load_error, workbook

        return None, completion_keys, load_error, None

    def _load_responsibility_slice(
        plan_mode: str,
        project_candidates: Sequence[str],
        entity_norm: str,
    ) -> pd.DataFrame:
        if not callable(responsibility_slice_provider):
            return pd.DataFrame()
        try:
            return responsibility_slice_provider(plan_mode, project_candidates, entity_norm)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Unable to access responsibility slice for mode '%s': %s", plan_mode, exc)
            return pd.DataFrame()

    def _load_responsibility_completion_lookup(plan_mode: str) -> set[tuple[str, str]]:
        if not callable(responsibility_completion_lookup_provider):
            return set()
        try:
            lookup = responsibility_completion_lookup_provider(plan_mode)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Unable to access responsibility completion lookup for mode '%s': %s",
                plan_mode,
                exc,
            )
            return set()
        return set(lookup or set())

    global _get_stringing_tse_lookup

    def _get_stringing_tse_lookup() -> tuple[dict[str, int], dict[str, str]]:
        if not config.enable_stringing:
            return {}, {}

        def _producer() -> tuple[dict[str, int], dict[str, str]]:
            if callable(stringing_tse_lookup_provider):
                try:
                    lookup = stringing_tse_lookup_provider()
                except Exception:
                    lookup = None
                else:
                    if isinstance(lookup, tuple) and lookup:
                        canonical_map = dict(lookup[0] or {})
                        alias_map = dict(lookup[1] or {})
                        if canonical_map or alias_map:
                            return canonical_map, alias_map
            df_compiled = pd.DataFrame()
            if callable(stringing_compiled_provider):
                try:
                    df_compiled = stringing_compiled_provider()
                except Exception:
                    df_compiled = pd.DataFrame()
            if not isinstance(df_compiled, pd.DataFrame) or df_compiled.empty:
                try:
                    df_compiled = _load_stringing_compiled_raw(config)
                except Exception:
                    df_compiled = pd.DataFrame()
            return build_tse_lookup_from_df(df_compiled)

        return _cached_global_result(
            "stringing:tse-lookup",
            _producer,
            clone=lambda payload: (dict(payload[0]), dict(payload[1])),
        )

    def _resolve_tse_value(
        norm_keys: Sequence[str],
        compact_keys: Sequence[str],
        canonical_map: Mapping[str, int],
        alias_map: Mapping[str, str],
    ) -> tuple[int | None, str | None]:
        if not canonical_map and not alias_map:
            return None, None
        for key in norm_keys:
            if key and key in canonical_map:
                return int(canonical_map[key]), key
        for key in compact_keys:
            if not key:
                continue
            canonical = alias_map.get(key)
            if canonical and canonical in canonical_map:
                return int(canonical_map[canonical]), canonical
        return None, None

    def _compute_planned_tower_layers(
        scoped_all: pd.DataFrame,
        months_ts: Sequence[pd.Timestamp],
    ) -> tuple[int, float]:
        active_months = sorted({ts for ts in months_ts if pd.notna(ts)})
        if not active_months or not has_plan_provider.get("erection"):
            return 0, 0.0
        try:
            resp_df, _, _, _ = _fetch_monthly_plan("erection")
            if not isinstance(resp_df, pd.DataFrame) or resp_df.empty:
                return 0, 0.0
            df_mp = resp_df.copy()
            if "plan_month" in df_mp.columns:
                df_mp["plan_month"] = pd.to_datetime(df_mp["plan_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                df_mp["completion_month"] = df_mp["plan_month"]
            elif "completion_date" in df_mp.columns:
                df_mp["completion_month"] = pd.to_datetime(df_mp["completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            else:
                df_mp["completion_month"] = pd.NaT
            for column in ("project_name", "project_key", "location_no"):
                if column not in df_mp.columns:
                    df_mp[column] = ""
                df_mp[column] = df_mp[column].map(_normalize_text)
            df_mp["project_name_lc"] = df_mp["project_name"].map(_normalize_lower)
            df_mp["project_key_lc"] = df_mp["project_key"].map(_normalize_lower)
            df_mp["location_no_norm"] = df_mp["location_no"].map(_normalize_location)
            if "project_name" in scoped_all.columns and not scoped_all.empty:
                sel_projects = set(
                    scoped_all["project_name"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
            else:
                sel_projects = set()
            if sel_projects:
                mask_project = df_mp["project_name_lc"].isin(sel_projects) | df_mp["project_key_lc"].isin(sel_projects)
                df_mp = df_mp.loc[mask_project].copy()
            df_mp = df_mp[df_mp["completion_month"].isin(active_months)].copy()
            if df_mp.empty:
                return 0, 0.0
            dedup_cols = ["project_name_lc", "location_no_norm"]
            valid_locations = df_mp.dropna(subset=dedup_cols).copy()
            if "tower_weight" in valid_locations.columns:
                valid_locations["tower_weight"] = pd.to_numeric(valid_locations["tower_weight"], errors="coerce").fillna(0.0)
            dedup_locations = (
                valid_locations.sort_values(dedup_cols)
                .drop_duplicates(subset=dedup_cols, keep="first")
            )
            planned_count = int(dedup_locations.shape[0])
            planned_mt = (
                float(dedup_locations.get("tower_weight", 0.0).sum())
                if "tower_weight" in dedup_locations.columns
                else 0.0
            )
            return planned_count, planned_mt
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to compute planned tower layers: %s", exc)
            return 0, 0.0

    def _derive_completion_window(
        loss_scope: pd.DataFrame,
        months_ts: Sequence[pd.Timestamp],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        today = pd.Timestamp.today().normalize()
        default_start = today.to_period("M").start_time.normalize()
        default_end = (today + pd.offsets.MonthEnd(0)).normalize()
        try:
            if months_ts:
                start = pd.Timestamp(min(months_ts)).normalize()
                end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
                return start, end
            if isinstance(loss_scope, pd.DataFrame) and not loss_scope.empty:
                comp_series = pd.to_datetime(loss_scope.get("completion_date"), errors="coerce").dropna()
                if len(comp_series):
                    return pd.Timestamp(comp_series.min()).normalize(), pd.Timestamp(comp_series.max()).normalize()
                date_series = pd.to_datetime(loss_scope.get("date"), errors="coerce").dropna()
                if len(date_series):
                    return pd.Timestamp(date_series.min()).normalize(), pd.Timestamp(date_series.max()).normalize()
        except Exception:
            pass
        return default_start, default_end

    def _count_completed_towers(
        loss_scope: pd.DataFrame,
        months_ts: Sequence[pd.Timestamp],
    ) -> int:
        if not isinstance(loss_scope, pd.DataFrame) or loss_scope.empty:
            return 0
        try:
            range_start, range_end = _derive_completion_window(loss_scope, months_ts)
            export_df, _ = _prepare_erections_completed(
                loss_scope,
                range_start=range_start,
                range_end=range_end,
                responsibilities_provider=None,
                search_text=None,
            )
            return int(len(export_df)) if isinstance(export_df, pd.DataFrame) else 0
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to count completed towers: %s", exc)
            return 0

    def _prepare_stringing_plan_frame(
        df_raw: pd.DataFrame,
        *,
        project_hint: str | None = None,
        source_path: Path | str | None = None,
        sheet_name: str | None = None,
    ) -> tuple[pd.DataFrame, set[tuple[str, str]], list[dict[str, str]]]:
        """
        Normalize the stringing monthly plan sheet into the generic responsibilities structure.
        Returns (frame, completion_keys) where completion keys capture completed spans.
        """

        if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            columns = [
                "project_name",
                "project_key",
                "entity_type",
                "entity_name",
                "location_no",
                "tower_weight",
                "revenue_planned",
                "revenue_realised",
                "stringing_span_completed",
                "span_from",
                "span_to",
                "method",
                "gang_strength",
                "paying_out_start",
                "final_sag_complete",
            ]
            return pd.DataFrame(columns=columns), set(), []

        col_lookup = {_normalize_col_key(col): col for col in df_raw.columns}
        column_aliases: dict[str, tuple[str, ...]] = {
            "serial": ("S. No.", "S no", "s no", "serial", "serial no", "jmc no", "span no"),
            "span_from": ("From AP", "from_ap", "from ap", "start tower", "from tower"),
            "span_to": ("To AP", "to_ap", "to ap", "end tower", "to tower"),
            "span_length": ("Span (m)", "span m", "span length", "length_m", "length (m)", "length"),
            "method": ("Method", "method"),
            "gang_strength": ("Gang Strength", "gang_strength"),
            "paying_out_start": ("Paying Out Start", "po_start_date", "p/o start", "po start", "paying_out_start"),
            "paying_out_complete": ("Paying Out Completed", "po_completion_date", "p/o completed", "po completion"),
            "final_sag_complete": ("Final Sag Complete", "fs_complete_date", "final sag", "fs complete date"),
            "gang_name": ("Gang Name", "gang_name"),
            "supervisor": ("Supervisor", "supervisor"),
            "section_incharge": ("Section Incharge", "section_incharge", "section incharge"),
        }

        def _resolve_series(key: str, default: Any = "") -> tuple[pd.Series, bool]:
            options = column_aliases.get(key, ())
            for candidate in options:
                norm = _normalize_col_key(candidate)
                if norm in col_lookup:
                    return df_raw[col_lookup[norm]], True
            norm = _normalize_col_key(key)
            if norm in col_lookup:
                return df_raw[col_lookup[norm]], True
            return pd.Series([default] * len(df_raw), index=df_raw.index), False

        def _optional_series(candidates: Sequence[str], default: Any = "") -> pd.Series:
            for candidate in candidates:
                key = _normalize_col_key(candidate)
                if key in col_lookup:
                    return df_raw[col_lookup[key]]
            return pd.Series([default] * len(df_raw), index=df_raw.index)

        # Log only if none of the alias options exist
        required_for_logging = {
            "S. No.": column_aliases["serial"],
            "From AP": column_aliases["span_from"],
            "To AP": column_aliases["span_to"],
            "Span (m)": column_aliases["span_length"],
            "Method": column_aliases["method"],
            "Gang Name": column_aliases["gang_name"],
            "Supervisor": column_aliases["supervisor"],
            "Section Incharge": column_aliases["section_incharge"],
        }
        issues: list[dict[str, str]] = []
        missing = [
            label
            for label, aliases in required_for_logging.items()
            if not any(_normalize_col_key(alias) in col_lookup for alias in aliases)
        ]
        if missing:
            LOGGER.warning("Monthly Plan (Stringing) missing columns: %s", ", ".join(missing))
            issues.append(
                {
                    "workbook": str(source_path or ""),
                    "sheet": sheet_name or "",
                    "issue": f"Missing columns: {', '.join(missing)}",
                }
            )

        project_names = _optional_series(("Project Name", "Project", "Project Title", "project_name")).map(_normalize_text)
        project_codes = _optional_series(("Project Code", "Project Key", "Project Id", "Project ID", "project")).map(
            _normalize_text
        )
        if project_hint:
            project_names = project_names.where(project_names.astype(bool), project_hint)
            project_codes = project_codes.where(project_codes.astype(bool), project_hint)
        serial_values, _ = _resolve_series("serial", default="")
        serial_values = serial_values.map(_normalize_text)
        span_from, _ = _resolve_series("span_from", default="")
        span_from = span_from.map(_normalize_text)
        span_to, _ = _resolve_series("span_to", default="")
        span_to = span_to.map(_normalize_text)
        span_length_series, _ = _resolve_series("span_length", default=0.0)
        span_length = pd.to_numeric(span_length_series, errors="coerce").fillna(0.0)
        method_values, _ = _resolve_series("method", default="")
        method_values = method_values.map(_normalize_text)
        gang_strength_series, gang_has_col = _resolve_series("gang_strength", default=pd.NA)
        gang_strength = pd.to_numeric(gang_strength_series, errors="coerce")
        paying_out_start_series, _ = _resolve_series("paying_out_start", default=pd.NaT)
        paying_out_complete_series, _ = _resolve_series("paying_out_complete", default=pd.NaT)
        paying_out_start = pd.to_datetime(paying_out_start_series, errors="coerce")
        paying_out_complete = pd.to_datetime(paying_out_complete_series, errors="coerce")
        final_sag_complete_series, _ = _resolve_series("final_sag_complete", default=pd.NaT)
        final_sag_complete = pd.to_datetime(final_sag_complete_series, errors="coerce")

        entity_sources: list[tuple[str, list[str]]] = []
        gang_series, has_gang = _resolve_series("gang_name", default="")
        if has_gang:
            entity_sources.append(("Gang", gang_series.map(_normalize_text).tolist()))
        supervisor_series, has_supervisor = _resolve_series("supervisor", default="")
        if has_supervisor:
            entity_sources.append(("Supervisor", supervisor_series.map(_normalize_text).tolist()))
        section_series, has_section = _resolve_series("section_incharge", default="")
        if has_section:
            entity_sources.append(("Section Incharge", section_series.map(_normalize_text).tolist()))

        normalized_rows: list[dict[str, Any]] = []
        completion_pairs: set[tuple[str, str]] = set()

        span_count = len(df_raw.index)
        span_done_mask = (paying_out_start.notna() & final_sag_complete.notna()).tolist()
        from_vals = span_from.tolist()
        to_vals = span_to.tolist()
        project_name_vals = project_names.tolist()
        project_code_vals = project_codes.tolist()
        serial_vals = serial_values.tolist()
        span_length_vals = span_length.tolist()
        method_vals = method_values.tolist()
        gang_strength_vals = gang_strength.tolist()
        paying_out_values = paying_out_start.tolist()
        paying_out_complete_values = paying_out_complete.tolist()
        final_sag_values = final_sag_complete.tolist()

        for idx in range(span_count):
            project_name = project_name_vals[idx]
            project_code = project_code_vals[idx] or project_name
            from_ap = from_vals[idx]
            to_ap = to_vals[idx]
            serial_label = serial_vals[idx]
            if from_ap and to_ap:
                span_label = f"{from_ap} \u2192 {to_ap}"
            else:
                span_label = from_ap or to_ap or serial_label or f"Span {idx + 1}"
            span_norm = _normalize_location(span_label)
            span_length_value = float(span_length_vals[idx]) if pd.notna(span_length_vals[idx]) else 0.0
            method_value = method_vals[idx]
            span_completed = bool(span_done_mask[idx])
            po_start_value = paying_out_values[idx]
            po_complete_value = paying_out_complete_values[idx]
            sag_complete_value = final_sag_values[idx]
            gang_strength_value = gang_strength_vals[idx]

            base_projects = [
                _normalize_lower(project_name),
                _normalize_lower(project_code),
            ]
            if span_completed and span_norm and any(base_projects):
                for candidate in base_projects:
                    if candidate:
                        completion_pairs.add((candidate, span_norm))

            for entity_label, entity_values in entity_sources:
                entity_name = entity_values[idx]
                if not entity_name:
                    continue
                normalized_rows.append(
                    {
                        "project_name": project_name,
                        "project_key": project_code or project_name,
                        "entity_type": entity_label,
                        "entity_name": entity_name,
                        "location_no": span_label,
                        "tower_weight": span_length_value,
                        "revenue_planned": 0.0,
                        "revenue_realised": 0.0,
                        "stringing_span_completed": span_completed,
                        "span_from": from_ap,
                        "span_to": to_ap,
                        "method": method_value,
                        "gang_strength": gang_strength_value,
                        "paying_out_start": po_start_value,
                        "paying_out_complete": po_complete_value,
                        "final_sag_complete": sag_complete_value,
                    }
                )

        normalized = pd.DataFrame(normalized_rows)
        required_payload_columns: list[tuple[str, Any]] = [
            ("project_name", ""),
            ("project_key", ""),
            ("entity_type", ""),
            ("entity_name", ""),
            ("location_no", ""),
            ("tower_weight", 0.0),
            ("revenue_planned", 0.0),
            ("revenue_realised", 0.0),
            ("stringing_span_completed", False),
            ("paying_out_complete", pd.NaT),
        ]
        for column, default in required_payload_columns:
            if column not in normalized.columns:
                normalized[column] = default
        if "completion_date" not in normalized.columns:
            normalized["completion_date"] = pd.NaT
        normalized["completion_date"] = pd.to_datetime(normalized["completion_date"], errors="coerce")

        def _fill_completion_from(column_name: str) -> None:
            if column_name not in normalized.columns:
                return
            fallback = pd.to_datetime(normalized[column_name], errors="coerce")
            if fallback is None:
                return
            normalized["completion_date"] = normalized["completion_date"].fillna(fallback)

        _fill_completion_from("final_sag_complete")
        _fill_completion_from("paying_out_complete")
        return normalized, completion_pairs, issues

    def _stringing_plan_output_path(cfg: AppConfig) -> Path:
        base = Path(cfg.data_path).expanduser()
        candidates: list[Path] = []

        def _push_directory(path_like: Path | str | None) -> None:
            if not path_like:
                return
            path_obj = Path(path_like).expanduser()
            if path_obj.suffix.lower() == ".xlsx":
                candidates.append(path_obj)
            else:
                candidates.append(path_obj / "StringingCompiled_Output.xlsx")

        if base.is_file():
            _push_directory(base.parent / "Stringing")
        else:
            _push_directory(base.parent / "Stringing")
            _push_directory(base.parent.parent / "Stringing")

        for rel in getattr(cfg, "stringing_parquet_dirs", ()):
            try:
                rel_path = (base / Path(rel)).resolve()
            except Exception:
                continue
            _push_directory(rel_path)

        _push_directory(Path("Parquets") / "Stringing")
        if base.is_file():
            _push_directory(base.with_name("StringingCompiled_Output.xlsx"))
        else:
            _push_directory(base / "Stringing")
        _push_directory(Path("Parquets"))
        _push_directory(Path("."))

        for candidate in candidates:
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                return candidate
            except Exception:
                continue
        fallback = Path("Parquets") / "Stringing" / "StringingCompiled_Output.xlsx"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback

    def _write_stringing_plan_snapshot(
        cfg: AppConfig,
        frame: pd.DataFrame,
        issues: list[dict[str, str]],
        index_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        output_path = _stringing_plan_output_path(cfg)
        try:
            mode = "a" if output_path.exists() else "w"
            with pd.ExcelWriter(
                output_path,
                engine="openpyxl",
                mode=mode,
                if_sheet_exists="replace",
            ) as writer:
                responsibilities_frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
                if responsibilities_frame is None or responsibilities_frame.empty:
                    responsibilities_frame = pd.DataFrame()
                for sheet in ("Stringing Plan", "MicroPlanResponsibilities"):
                    responsibilities_frame.to_excel(writer, sheet_name=sheet, index=False)
                index_frame = pd.DataFrame(index_rows or [])
                index_frame.to_excel(writer, sheet_name="MicroPlanIndex", index=False)
                issues_frame = (
                    pd.DataFrame(issues)
                    if issues
                    else pd.DataFrame([{"issue": "No issues detected"}])
                )
                for sheet in ("Stringing Plan Issues", "MicroPlanDataIssues"):
                    issues_frame.to_excel(writer, sheet_name=sheet, index=False)
        except Exception as exc:
            LOGGER.warning("Unable to write Stringing plan snapshot to '%s': %s", output_path, exc)

    def _maybe_write_stringing_plan_snapshot(
        cfg: AppConfig,
        frame: pd.DataFrame,
        issues: list[dict[str, str]],
        index_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        cache = _STRINGING_PLAN_CACHE
        ts_now = time.time()
        last_written = cache.get("last_written", 0.0)
        if ts_now - last_written < _STRINGING_PLAN_CACHE_TTL_SECONDS:
            return
        _write_stringing_plan_snapshot(cfg, frame, issues, index_rows or [])
        cache["last_written"] = ts_now

    def _load_stringing_plan_snapshot(
        cfg: AppConfig,
    ) -> tuple[pd.DataFrame | None, set[tuple[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
        cache = _STRINGING_PLAN_CACHE
        ts_now = time.time()
        cached_frame = cache["frame"]
        if (
            isinstance(cached_frame, pd.DataFrame)
            and not cached_frame.empty
            and (ts_now - cache["stored_at"] < _STRINGING_PLAN_CACHE_TTL_SECONDS)
        ):
            issues = cache["issues"]
            completion = cache["completion"]
            index_rows = cache.get("index", [])
            return cached_frame.copy(), set(completion), list(issues), list(index_rows)

        payload = stringing_plan_accessor.load()
        if payload.has_frame:
            frame = payload.frame.copy()
            completion_keys = set(payload.completion_keys or set())
            cache["frame"] = frame.copy()
            cache["completion"] = set(completion_keys)
            cache["issues"] = []
            cache["index"] = []
            cache["stored_at"] = ts_now
            cache["last_written"] = ts_now
            return frame, completion_keys, [], []

        root = _resolve_stringing_microplan_root(config.data_path)
        frames: list[pd.DataFrame] = []
        completion_keys: set[tuple[str, str]] = set()
        issues: list[dict[str, str]] = []
        index_rows: list[dict[str, Any]] = []

        def _append_index_entry(
            *,
            workbook: Path | str | None,
            sheet_name: str | None,
            project_code: str,
            project_label: str,
            rows_cleaned: int,
            status: str,
            error: str,
            issues_logged: int = 0,
            input_rows: int = 0,
            plan_month: str = "",
        ) -> None:
            index_rows.append(
                {
                    "file_path": str(workbook or ""),
                    "sheet_name": sheet_name or "",
                    "project_name": project_label or project_code or "",
                    "project_key": project_code or project_label or "",
                    "rows_cleaned": rows_cleaned,
                    "input_rows": input_rows,
                    "issues_logged": issues_logged,
                    "plan_month": plan_month,
                    "status": status,
                    "error": error or "",
                }
            )

        def _infer_plan_month_from_frame(frame: pd.DataFrame) -> str:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return ""
            candidate_columns = (
                "plan_month",
                "completion_date",
                "final_sag_complete",
                "paying_out_complete",
                "paying_out_start",
            )
            for column in candidate_columns:
                if column not in frame.columns:
                    continue
                series = pd.to_datetime(frame[column], errors="coerce")
                series = series.dropna()
                if series.empty:
                    continue
                ts = pd.Timestamp(series.iloc[0])
                return ts.to_period("M").to_timestamp().strftime("%Y-%m")
            return ""

        if not root.exists():
            message = f"Stringing micro plan root not found: {root}"
            LOGGER.warning(message)
            issues.append({"workbook": str(root), "sheet": "", "issue": "ROOT_NOT_FOUND"})
            _append_index_entry(
                workbook=root,
                sheet_name="",
                project_code="",
                project_label="",
                rows_cleaned=0,
                status="error",
                error="ROOT_NOT_FOUND",
            )
        else:
            preferred_sheet = getattr(cfg, "stringing_sheet_name", "")
            for workbook in sorted(root.rglob("*.xlsx")):
                if workbook.name.startswith("~$"):
                    continue
                project_code, project_label = _infer_project_hint(workbook)
                try:
                    xls = pd.ExcelFile(workbook)
                except Exception as exc:
                    issues.append({"workbook": str(workbook), "sheet": "", "issue": f"OPEN_FAILED: {exc}"})
                    _append_index_entry(
                        workbook=workbook,
                        sheet_name="",
                        project_code=project_code,
                        project_label=project_label,
                        rows_cleaned=0,
                        status="error",
                        error=f"OPEN_FAILED: {exc}",
                    )
                    continue
                sheet_name = next(
                    (
                        name
                        for name in xls.sheet_names
                        if preferred_sheet and _normalize_col_key(name) == _normalize_col_key(preferred_sheet)
                    ),
                    None,
                )
                if sheet_name is None:
                    sheet_name = next((name for name in xls.sheet_names if "string" in name.lower()), None)
                if sheet_name is None:
                    issues.append({"workbook": str(workbook), "sheet": "", "issue": "NO_STRINGING_SHEET"})
                    _append_index_entry(
                        workbook=workbook,
                        sheet_name="",
                        project_code=project_code,
                        project_label=project_label,
                        rows_cleaned=0,
                        status="error",
                        error="NO_STRINGING_SHEET",
                    )
                    continue
                try:
                    df_raw = pd.read_excel(workbook, sheet_name=sheet_name, header=1)
                except Exception as exc:
                    issues.append({"workbook": str(workbook), "sheet": sheet_name, "issue": f"READ_FAILED: {exc}"})
                    _append_index_entry(
                        workbook=workbook,
                        sheet_name=sheet_name,
                        project_code=project_code,
                        project_label=project_label,
                        rows_cleaned=0,
                        status="error",
                        error=f"READ_FAILED: {exc}",
                    )
                    continue
                normalized, _plan_keys, local_issues = _prepare_stringing_plan_frame(
                    df_raw,
                    project_hint=project_label or project_code,
                    source_path=workbook,
                    sheet_name=sheet_name,
                )
                if project_code:
                    normalized["project_key"] = normalized["project_key"].replace("", project_code)
                if project_label:
                    normalized["project_name"] = normalized["project_name"].replace("", project_label)
                frames.append(normalized)
                issues.extend(local_issues)
                plan_month_value = _infer_plan_month_from_frame(normalized)
                _append_index_entry(
                    workbook=workbook,
                    sheet_name=sheet_name,
                    project_code=project_code,
                    project_label=project_label,
                    rows_cleaned=int(len(normalized)),
                    input_rows=int(len(df_raw)),
                    status="ok",
                    error="",
                    issues_logged=len(local_issues),
                    plan_month=plan_month_value,
                )

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        _write_stringing_plan_snapshot(cfg, frame, issues, index_rows)
        cache["frame"] = frame.copy()
        cache["completion"] = set(completion_keys)
        cache["issues"] = list(issues)
        cache["index"] = list(index_rows)
        cache["stored_at"] = ts_now
        cache["last_written"] = ts_now
        return frame, completion_keys, issues, index_rows

    # --- shared: responsibilities figure + KPIs for a single project selection ---
    def _stringing_length_km_series(frame: pd.DataFrame) -> pd.Series:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.Series([], dtype=float)

        length_km_cols = [
            "length_km",
            "string_km",
            "planned_km",
            "po_km",
            "daily_km",
        ]
        meter_cols = [
            "length_m",
            "length (m)",
            "span_m",
            "span (m)",
            "span length",
            "length",
            "p/o",
        ]

        for col in length_km_cols:
            if col in frame.columns:
                series = pd.to_numeric(frame[col], errors="coerce")
                return series.fillna(0.0)

        for col in meter_cols:
            if col in frame.columns:
                series = pd.to_numeric(frame[col], errors="coerce")
                return (series / 1000.0).fillna(0.0)

        if "tower_weight" in frame.columns:
            series = pd.to_numeric(frame["tower_weight"], errors="coerce")
            return (series / 1000.0).fillna(0.0)

        return pd.Series(0.0, index=frame.index, dtype=float)

    def _build_monthly_plan_for_project(
        project_value: str | Sequence[str] | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
        *,
        plan_mode: str = "erection",
    ):
        plan_key = "stringing" if str(plan_mode).strip().lower() == "stringing" else "erection"
        plan_title = "Monthly Plan (Stringing)" if plan_key == "stringing" else "Monthly Plan (Erection)"
        plan_noun = "Stringing plan" if plan_key == "stringing" else "Monthly Plan"

        def _empty_response(message: str):
            empty_fig = build_empty_responsibilities_figure(message)
            return empty_fig, "\u2014", "\u2014", "\u2014"

        candidates_raw: list[Any] = []
        if isinstance(project_value, dict):
            candidates_raw.extend([
                project_value.get("name"),
                project_value.get("code"),
            ])
            candidates_raw.extend(project_value.values())
        elif isinstance(project_value, Sequence) and not isinstance(project_value, (str, bytes)):
            candidates_raw.extend(project_value)
        elif project_value is not None:
            candidates_raw.append(project_value)

        project_candidates: list[str] = []
        seen_candidates: set[str] = set()
        for candidate in candidates_raw:
            text = "" if candidate is None else str(candidate).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            project_candidates.append(text)

        if not project_candidates:
            return _empty_response("Select a single project to view its details.")

        entity = (entity_value or "Supervisor").strip()
        metric = (metric_value or "tower_weight").strip()
        metric = metric if metric in {"revenue", "tower_weight"} else "tower_weight"
        stringing_length_label = plan_key == "stringing" and metric == "tower_weight"
        stringing_length_label = plan_key == "stringing" and metric == "tower_weight"

        ent_map = {
            "supervisor": "supervisor",
            "supervisors": "supervisor",
            "section incharge": "section incharge",
            "section-incharge": "section incharge",
            "section in-charge": "section incharge",
            "gang": "gang",
            "gangs": "gang",
        }
        entity_norm = ent_map.get(entity.lower(), entity.lower())

        month_list = _ensure_list(months_value)
        months_ts = resolve_months(month_list, quick_range_value)
        active_months = sorted({ts for ts in months_ts if pd.notna(ts)})

        completion_keys: set[tuple[str, str]] = set()
        df_entity = pd.DataFrame()
        slice_candidates = project_candidates or []
        if slice_candidates:
            slice_frame = _load_responsibility_slice(plan_key, slice_candidates, entity_norm)
            if not slice_frame.empty:
                df_entity = slice_frame.copy()
                completion_keys = _load_responsibility_completion_lookup(plan_key)
                if active_months and "completion_month" in df_entity.columns:
                    df_entity = df_entity[df_entity["completion_month"].isin(active_months)].copy()

        if df_entity.empty:
            df_atomic, completed_keys, load_error_msg, workbook = _fetch_monthly_plan(
                plan_key,
                allow_workbook_fallback=True,
            )
            if df_atomic is None or df_atomic.empty:
                message = load_error_msg or f"No {plan_title} data found in the compiled workbook."
                return _empty_response(message)

            completion_keys = {
                (_compact_project_key(project), _normalize_location(location))
                for project, location in (completed_keys or set())
            }
            completion_keys = {(proj, loc) for proj, loc in completion_keys if proj and loc}
            df_atomic = df_atomic.copy()
            plan_source_path = None
            plan_sheet_name = None
            if workbook is not None:
                plan_source_path = getattr(workbook, "_plan_workbook_path", None)
                plan_sheet_name = getattr(workbook, "_plan_sheet_name", None)
                if plan_source_path is None:
                    plan_source_path = getattr(workbook, "io", None)
            plan_issues: list[dict[str, str]] = []
            if plan_key == "stringing":
                if workbook is not None and "stringing_span_completed" not in df_atomic.columns:
                    df_atomic, _plan_completion_keys, plan_issues = _prepare_stringing_plan_frame(
                        df_atomic,
                        source_path=plan_source_path,
                        sheet_name=plan_sheet_name,
                    )
                    completion_keys.update(
                        {
                            (_compact_project_key(project), _normalize_location(location))
                            for project, location in _plan_completion_keys
                        }
                    )
                    completion_keys = {(proj, loc) for proj, loc in completion_keys if proj and loc}
                elif "stringing_span_completed" in df_atomic.columns:
                    df_atomic["stringing_span_completed"] = df_atomic["stringing_span_completed"].fillna(False)
                if workbook is not None:
                    _maybe_write_stringing_plan_snapshot(config, df_atomic, plan_issues, [])

            if "plan_month" in df_atomic.columns:
                df_atomic["plan_month"] = pd.to_datetime(
                    df_atomic["plan_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
                df_atomic["completion_month"] = df_atomic["plan_month"]
            elif "completion_date" in df_atomic.columns:
                df_atomic["completion_month"] = pd.to_datetime(
                    df_atomic["completion_date"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            else:
                df_atomic["completion_month"] = pd.NaT

            def _norm_text(v: object) -> str:
                s = str(v).replace("\u00a0", " ").strip()
                low = s.lower()
                if low in {"", "nan", "none", "null"}:
                    return ""
                return s

            def _norm_lc(v: object) -> str:
                return _norm_text(v).lower()

            def _norm_loc(v: object) -> str:
                t = _norm_text(v)
                if not t:
                    return ""
                if t.endswith(".0") and t.replace(".", "", 1).isdigit():
                    t = t.split(".", 1)[0]
                return t

            for c in ("project_key", "project_name", "entity_type", "entity_name", "location_no"):
                if c not in df_atomic.columns:
                    df_atomic[c] = ""
            df_atomic["project_name_lc"] = df_atomic["project_name"].map(_norm_lc)
            df_atomic["project_key_lc"] = df_atomic["project_key"].astype(str).map(_norm_lc)
            df_atomic["project_key_norm"] = df_atomic["project_key_lc"].map(_compact_project_key)
            df_atomic["project_key_norm"] = df_atomic["project_key_norm"].where(
                df_atomic["project_key_norm"].astype(bool),
                df_atomic["project_name_lc"].map(_compact_project_key),
            )
            df_atomic["location_no_norm"] = df_atomic["location_no"].map(_norm_loc)

            if active_months:
                df_atomic = df_atomic[df_atomic["completion_month"].isin(active_months)].copy()

            for candidate in project_candidates:
                sel = _norm_lc(candidate)
                mask_name_or_key = (
                    (df_atomic["project_name_lc"] == sel) | (df_atomic["project_key_lc"] == sel)
                )
                if not mask_name_or_key.any():
                    import re as _re

                    sel_compact = _re.sub(r"[^a-z0-9]", "", sel)
                    project_name_compact = df_atomic["project_name_lc"].str.replace(r"[^a-z0-9]", "", regex=True)
                    project_key_compact = df_atomic["project_key_lc"].str.replace(r"[^a-z0-9]", "", regex=True)

                    mask_name_or_key = (
                        (project_name_compact == sel_compact) | (project_key_compact == sel_compact)
                    )
                if mask_name_or_key.any():
                    df_entity = df_atomic[mask_name_or_key].copy()
                    break

            if df_entity.empty:
                return _empty_response("No plan entries found for the selected project.")

            if stringing_length_label:
                df_entity["__stringing_length_km"] = _stringing_length_km_series(df_entity)

            df_entity["entity_type_lc"] = df_entity["entity_type"].map(_norm_lc)
            df_entity = df_entity[df_entity["entity_type_lc"] == entity_norm].copy()

            if df_entity.empty:
                return _empty_response("No plan entries found for the selected filters.")

        if stringing_length_label and "__stringing_length_km" not in df_entity.columns:
            df_entity["__stringing_length_km"] = _stringing_length_km_series(df_entity)

        df_entity["revenue_planned"] = pd.to_numeric(df_entity.get("revenue_planned", 0.0), errors="coerce").fillna(0.0)
        df_entity["revenue_realised"] = pd.to_numeric(df_entity.get("revenue_realised", 0.0), errors="coerce").fillna(0.0)
        df_entity["tower_weight"] = pd.to_numeric(df_entity.get("tower_weight", 0.0), errors="coerce").fillna(0.0)
        if stringing_length_label:
            length_series = pd.to_numeric(
                df_entity.get("__stringing_length_km"), errors="coerce"
            ).fillna(0.0)
            df_entity["tower_weight"] = length_series

        if "is_completed" not in df_entity.columns:
            def _project_norm_series(frame: pd.DataFrame) -> pd.Series:
                if "project_key_norm" in frame.columns:
                    base = frame["project_key_norm"]
                else:
                    base = None
                    for column in (
                        "project_key",
                        "project_code",
                        "project",
                        "project_name",
                        "project_name_display",
                    ):
                        if column in frame.columns:
                            base = frame[column]
                            break
                    if base is None:
                        base = pd.Series([""] * len(frame), index=frame.index)
                return base.astype(str).map(lambda value: _compact_project_key(value) or _normalize_lower(value))

            def _location_norm_series(frame: pd.DataFrame) -> pd.Series:
                if "location_no_norm" in frame.columns:
                    base = frame["location_no_norm"]
                else:
                    base = None
                    for column in ("location_no", "location", "location_name"):
                        if column in frame.columns:
                            base = frame[column]
                            break
                    if base is None:
                        base = pd.Series([""] * len(frame), index=frame.index)
                return base.astype(str).map(_normalize_location)

            project_norm = _project_norm_series(df_entity)
            location_norm = _location_norm_series(df_entity)
            if completion_keys:
                completion_mask = [
                    (proj, loc) in completion_keys for proj, loc in zip(project_norm.tolist(), location_norm.tolist())
                ]
            else:
                completion_mask = [False] * len(df_entity)
            df_entity["is_completed"] = completion_mask

        df_entity["delivered_revenue"] = np.where(
            df_entity["revenue_realised"] > 0,
            df_entity["revenue_realised"],
            np.where(df_entity["is_completed"], df_entity["revenue_planned"], 0.0),
        )
        df_entity["delivered_tower_weight"] = np.where(
            df_entity["is_completed"], df_entity["tower_weight"], 0.0
        )

        df_entity = df_entity[df_entity.get("entity_name", "").astype(bool)].copy()
        if df_entity.empty:
            return _empty_response("No plan entries found for the selected filters.")

        aggregated = (
            df_entity.groupby("entity_name", as_index=False)[
                [
                    "revenue_planned",
                    "delivered_revenue",
                    "tower_weight",
                    "delivered_tower_weight",
                    "location_no",
                ]
            ].agg({
                "revenue_planned": "sum",
                "delivered_revenue": "sum",
                "tower_weight": "sum",
                "delivered_tower_weight": "sum",
                "location_no": lambda s: [str(v).strip() for v in s if str(v).strip()],
            })
        )

        target_metric_col = "revenue_planned" if metric == "revenue" else "tower_weight"
        delivered_metric_col = ("delivered_revenue" if metric == "revenue" else "delivered_tower_weight")

        # Derive location lists
        filtered_target = df_entity[df_entity[target_metric_col] > 0]
        if filtered_target.empty:
            filtered_target = df_entity
        target_locations = (
            filtered_target.groupby("entity_name")["location_no"].apply(list).rename("target_locations")
        )
        filtered_delivered = df_entity[df_entity[delivered_metric_col] > 0]
        delivered_locations = (
            filtered_delivered.groupby("entity_name")["location_no"].apply(list).rename("delivered_locations")
        )
        aggregated = aggregated.merge(target_locations, on="entity_name", how="left")
        aggregated = aggregated.merge(delivered_locations, on="entity_name", how="left")

        aggregated["delivered_value"] = np.where(
            metric == "revenue",
            aggregated["delivered_revenue"],
            aggregated["delivered_tower_weight"],
        )

        axis_override = "Length (KM)" if stringing_length_label else None
        unit_override = "KM" if stringing_length_label else None

        # Ensure chart builder has the target column by metric name
        if "revenue_planned" in aggregated.columns and "revenue" not in aggregated.columns:
            aggregated["revenue"] = aggregated["revenue_planned"]

        if aggregated.empty:
            return _empty_response("No responsibilities found for the selected filters.")

        fig = build_responsibilities_chart(
            aggregated,
            entity_label=entity,
            metric=metric,
            axis_title_override=axis_override,
            unit_label_override=unit_override,
            title=None,
            top_n=20,
        )

        if metric == "revenue":
            total_target = float(aggregated["revenue_planned"].sum())
            total_delivered = float(aggregated["delivered_revenue"].sum())
        else:
            total_target = float(aggregated["tower_weight"].sum())
            total_delivered = float(aggregated["delivered_tower_weight"].sum())

        achievement = 0.0 if total_target == 0 else (total_delivered / total_target) * 100.0

        def fmt_num(value: float) -> str:
            if metric == "revenue":
                return f"\u20b9{value:,.0f}"
            unit = "KM" if stringing_length_label else "MT"
            precision = 1 if stringing_length_label else 0
            return f"{value:,.{precision}f} {unit}"

        kpi_target_txt = fmt_num(total_target)
        kpi_deliv_txt = fmt_num(total_delivered)
        kpi_ach_txt = f"{achievement:.0f}%"

        return fig, kpi_target_txt, kpi_deliv_txt, kpi_ach_txt
    
    def _get_project_baselines(
        mode: str = "erection",
    ) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
        if project_baseline_provider is None:
            return {}, {}
        try:
            overall_map, monthly_map = project_baseline_provider(mode)  # type: ignore[misc]
        except TypeError:
            overall_map, monthly_map = project_baseline_provider()  # type: ignore[call-arg]
        except Exception as exc:
            LOGGER.warning("Failed to retrieve project baselines: %s", exc)
            return {}, {}
        return overall_map or {}, monthly_map or {}

    def _format_summary_value(value: float | None, unit_label: str | None = None, *, precision: int = 1) -> str:
        if value is None or pd.isna(value):
            return "-"
        try:
            numeric = float(value)
        except Exception:
            return "-"
        formatted = f"{numeric:,.{precision}f}"
        return f"{formatted} {unit_label}" if unit_label else formatted

    def _avg_metric_value(df: pd.DataFrame, metric_col: str, is_stringing: bool) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty or metric_col not in df.columns:
            return 0.0
        if is_stringing:
            required_cols = {"gang_name", "month"}
            if not required_cols.issubset(df.columns):
                return 0.0
            monthly_totals = (
                df.groupby(["gang_name", "month"], dropna=True)[metric_col]
                .sum()
                .reset_index(name="monthly_value")
            )
            if monthly_totals.empty:
                return 0.0
            return float(monthly_totals["monthly_value"].mean())
        return float(pd.to_numeric(df[metric_col], errors="coerce").dropna().mean())

    def _empty_summary_payload(is_stringing: bool) -> dict[str, str]:
        payload = {
            "projects": "-",
            "totals": "-",
            "gangs": "-",
            "productivity": "-",
            "lost_units": "-",
        }
        if is_stringing:
            payload["po_completion"] = "-"
            payload["tse"] = "-"
        payload["_meta"] = {
            "loss_rows": [],
            "unit_label": "KM" if is_stringing else "MT",
            "mode": "stringing" if is_stringing else "erection",
            "is_stringing": is_stringing,
        }
        return payload

    def _compute_mode_summary_components(
        *,
        scoped_full: pd.DataFrame,
        scoped_all: pd.DataFrame,
        months_ts: list[pd.Timestamp],
        days_factor: float,
        metric_col: str,
        is_stringing: bool,
        meta_signature: str,
        cache_key: str | None,
        scope_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "has_rows": not (scoped_full.empty and scoped_all.empty),
            "loss_scope": pd.DataFrame(),
            "history_scope": pd.DataFrame(),
            "projects_count": 0,
            "gangs_count": 0,
            "total_delivered": 0.0,
            "total_lost": 0.0,
            "total_potential": 0.0,
            "balance_value": 0.0,
            "loss_rows": [],
        }
        selected = (scope_meta or {}).get("selected") or {}
        project_filters = _normalize_str_list(selected.get("projects"))

        _project_label_columns = (
            "project_name",
            "project",
            "project_name_display",
            "Project Name",
            "project_key",
            "project_code",
            "project_key_norm",
        )
        _project_file_pattern = re.compile(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{2,4})\b", re.IGNORECASE)

        def _collect_project_key_tokens(frame: pd.DataFrame) -> set[str]:
            tokens: set[str] = set()
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return tokens
            for column in _project_label_columns:
                if column not in frame.columns:
                    continue
                values = (
                    frame[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                )
                for value in values:
                    compact = _compact_project_key(value)
                    normalized = _normalize_lower(value)
                    if compact:
                        tokens.add(compact)
                    elif normalized:
                        tokens.add(normalized)
            if "source_file" in frame.columns:
                file_values = (
                    frame["source_file"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                )
                for value in file_values:
                    match = _project_file_pattern.search(value)
                    if not match:
                        continue
                    compact = _compact_project_key(f"{match.group(1)}{match.group(2)}")
                    if compact:
                        tokens.add(compact)
            return tokens

        def _normalize_project_token_set(values: Iterable[object] | None) -> set[str]:
            tokens: set[str] = set()
            if values is None:
                return tokens
            for value in values:
                compact = _compact_project_key(value)
                normalized = _normalize_lower(value)
                if compact:
                    tokens.add(compact)
                elif normalized:
                    tokens.add(normalized)
            return {token for token in tokens if token}

        if not result["has_rows"]:
            return result

        has_selected_months = bool(months_ts)
        earliest_month = None
        if has_selected_months and not scoped_all.empty and "month" in scoped_all.columns:
            month_values = sorted({ts for ts in months_ts if pd.notna(ts)})
            period_mask = scoped_all["month"].isin(month_values)
            loss_scope = scoped_all.loc[period_mask].copy()
            earliest_month = month_values[0] if month_values else None
            history_scope = scoped_all.loc[scoped_all["month"] < (earliest_month or pd.Timestamp.max)].copy()
        else:
            loss_scope = scoped_all.copy()
            history_scope = scoped_all.copy()

        allowed_months = set(month_values) if has_selected_months and month_values else None
        idle_table = _idle_table_for_mode("stringing" if is_stringing else "erection")

        if not loss_scope.empty:
            if "gang_name" in loss_scope.columns:
                loss_scope = loss_scope.dropna(subset=["gang_name"])
                loss_scope["gang_name"] = loss_scope["gang_name"].astype(str).str.strip()
            if "project_name" in loss_scope.columns:
                loss_scope["project_name"] = loss_scope["project_name"].astype(str).str.strip()

        result["loss_scope"] = loss_scope
        result["history_scope"] = history_scope

        def _maybe_cached(token: str, producer: Callable[[], T], *, clone: Callable[[T], T] | None = None) -> T:
            if cache_key:
                return _cached_scope_result(cache_key, token, producer, clone=clone)
            value = producer()
            return clone(value) if clone else value

        precomputed_overall, precomputed_monthly = _get_project_baselines(
            "stringing" if is_stringing else "erection"
        )
        use_precomputed = bool(precomputed_overall)
        proj_overall_all: dict[str, float] = {}
        proj_monthly: dict[str, dict[pd.Timestamp, float]] = {}

        if use_precomputed:
            if "project_name" in scoped_all.columns:
                available_projects = (
                    scoped_all["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                available_projects = []
            if available_projects:
                proj_overall_all = {
                    project: precomputed_overall.get(project)
                    for project in available_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in available_projects
                }
            else:
                proj_overall_all = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if has_selected_months and earliest_month is not None:
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < earliest_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < earliest_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token_all = f"project-baseline::{metric_col}::{meta_signature}"

            def _compute_baseline_all() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if scoped_all.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(scoped_all, metric_col)
                return compute_project_baseline_maps(scoped_all)

            proj_overall_all, proj_monthly_all = _maybe_cached(
                baseline_token_all,
                _compute_baseline_all,
                clone=_clone_baseline_result,
            )

            if has_selected_months and earliest_month is not None:
                proj_overall_all = {
                    project: value
                    for project, value in proj_overall_all.items()
                    if value is not None
                }
                filtered: dict[str, dict[pd.Timestamp, float]] = {}
                for project, month_map in proj_monthly_all.items():
                    subset = {month: value for month, value in month_map.items() if month < earliest_month}
                    if subset:
                        filtered[project] = subset
                proj_monthly = filtered
            else:
                proj_monthly = proj_monthly_all

        gang_to_project = (
            scoped_all[["gang_name", "project_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("gang_name")["project_name"]
            .astype(str)
            .to_dict()
        )

        baseline_overall_map = {g: proj_overall_all.get(p) for g, p in gang_to_project.items()}
        baseline_monthly_map = {g: proj_monthly.get(p, {}) for g, p in gang_to_project.items()}

        loss_token = f"loss::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}"

        def _compute_loss_rows() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            if loss_scope.empty:
                return rows
            for gang_name, gang_df in loss_scope.groupby("gang_name"):
                if gang_df.empty:
                    continue
                overall_baseline = baseline_overall_map.get(gang_name)
                if is_stringing:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss_for_column(
                        gang_df,
                        metric_column=metric_col,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                        idle_intervals=idle_table,
                        allowed_months=allowed_months,
                    )
                else:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                        gang_df,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_mt_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                        idle_intervals=idle_table,
                        allowed_months=allowed_months,
                    )
                rows.append(
                    {
                        "gang_name": gang_name,
                        "delivered": delivered,
                        "lost": loss_mt,
                        "potential": potential,
                        "avg_prod": (gang_df[metric_col].mean() if metric_col in gang_df.columns else 0.0),
                        "baseline": baseline,
                    }
                )
            return rows

        loss_rows = _maybe_cached(
            loss_token,
            _compute_loss_rows,
            clone=_clone_loss_rows,
        )
        result["loss_rows"] = loss_rows

        loss_df = pd.DataFrame(loss_rows)
        if is_stringing and not loss_df.empty:
            loss_df["avg_prod"] = loss_df["avg_prod"].astype(float) * days_factor
            loss_df["baseline"] = loss_df["baseline"].astype(float) * days_factor

        metric_scope = loss_scope if not loss_scope.empty else scoped_full
        total_metric = (
            float(metric_scope[metric_col].sum())
            if not metric_scope.empty and metric_col in metric_scope.columns
            else 0.0
        )
        total_delivered = float(loss_df["delivered"].sum()) if not loss_df.empty else 0.0
        total_lost = float(loss_df["lost"].sum()) if not loss_df.empty else 0.0
        total_potential = total_delivered + total_lost if (total_delivered or total_lost) else total_metric
        balance_value = max(total_potential - total_delivered, 0.0)

        completion_override: float | None = None
        if is_stringing:
            completion_override = _sum_completion_totals(
                scoped_full,
                value_column="length_km",
                completion_column="fs_complete_date",
                fallback_columns=[
                    ("span (m)", 0.001),
                    ("span_m", 0.001),
                    ("length", 0.001),
                    ("length_m", 0.001),
                    ("tower_weight", 0.001),
                ],
            )
        else:
            completion_override = _sum_completion_totals(
                scoped_full,
                value_column="tower_weight",
                completion_column="completion_date",
            )
        if completion_override is not None:
            total_delivered = completion_override
            total_metric = completion_override if not total_lost else completion_override + total_lost
            total_potential = total_delivered + total_lost if total_lost else total_delivered
            balance_value = max(total_potential - total_delivered, 0.0)

        def _nunique(frame: pd.DataFrame, column: str) -> int:
            if column not in frame.columns or frame.empty:
                return 0
            return (
                frame[column]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )

        project_keys = set()
        for frame in (loss_scope, scoped_full):
            project_keys.update(_collect_project_key_tokens(frame))
        if is_stringing and callable(stringing_compiled_provider):
            try:
                compiled_df = stringing_compiled_provider() or pd.DataFrame()
            except Exception:
                compiled_df = pd.DataFrame()
            if isinstance(compiled_df, pd.DataFrame) and not compiled_df.empty:
                filtered_compiled = compiled_df
                if allowed_months:
                    mask = pd.Series(False, index=filtered_compiled.index)
                    for column in (
                        "month",
                        "date",
                        "fs_complete_date",
                        "fs_start_date",
                        "fs_starting_date",
                        "po_completion_date",
                        "po_completion",
                        "paying_out_complete",
                        "paying_out_start",
                        "completion date",
                        "starting date",
                        "completion month",
                    ):
                        if column not in filtered_compiled.columns:
                            continue
                        series = pd.to_datetime(filtered_compiled[column], errors="coerce")
                        if series.isna().all():
                            continue
                        normalized = series.dt.to_period("M").dt.to_timestamp()
                        mask = mask | normalized.isin(allowed_months)
                    if mask.any():
                        filtered_compiled = filtered_compiled.loc[mask].copy()
                compiled_tokens = _collect_project_key_tokens(filtered_compiled)
                if project_filters:
                    filter_keys = _normalize_project_token_set(project_filters)
                    if filter_keys:
                        compiled_tokens = {token for token in compiled_tokens if token in filter_keys}
                project_keys.update(compiled_tokens)
        projects_count = len(project_keys)
        if is_stringing:
            try:
                plan_df, _ = _stringing_plan_totals_by_project(
                    months_ts,
                    current_month=months_ts[-1] if months_ts else None,
                )
            except Exception:
                plan_df = pd.DataFrame()
            if isinstance(plan_df, pd.DataFrame) and not plan_df.empty:
                plan_keys = _normalize_project_token_set(plan_df.index.astype(str).tolist())
                project_filters = _normalize_str_list(selected.get("projects"))
                if project_filters:
                    filter_keys = _normalize_project_token_set(project_filters)
                    if filter_keys:
                        plan_keys = {key for key in plan_keys if key in filter_keys}
                combined_keys = {key for key in project_keys | plan_keys if key}
                if combined_keys:
                    projects_count = len(combined_keys)
        gangs_count = _nunique(loss_scope, "gang_name") or _nunique(scoped_full, "gang_name")

        result.update(
            {
                "projects_count": projects_count,
                "gangs_count": gangs_count,
                "total_delivered": total_delivered,
                "total_lost": total_lost,
                "total_potential": total_potential,
                "balance_value": balance_value,
            }
        )
        return result

    def _get_stringing_po_daily_frame() -> pd.DataFrame:
        if not config.enable_stringing:
            return pd.DataFrame()

        def _producer() -> pd.DataFrame:
            df_compiled = pd.DataFrame()
            if callable(stringing_compiled_provider):
                try:
                    df_compiled = stringing_compiled_provider()
                except Exception:
                    df_compiled = pd.DataFrame()
            if not isinstance(df_compiled, pd.DataFrame) or df_compiled.empty:
                return pd.DataFrame()
            payout_source = df_compiled.copy()
            po_col = None
            for cand in ("paying_out_complete", "po_completion_date", "po_completion"):
                if cand in payout_source.columns:
                    po_col = cand
                    break
            if po_col is None:
                return pd.DataFrame()
            payout_source["po_completion_date"] = pd.to_datetime(payout_source[po_col], errors="coerce")
            payout_source = payout_source.dropna(subset=["po_completion_date"])
            if payout_source.empty:
                return pd.DataFrame()
            payout_source["po_start_date"] = payout_source["po_completion_date"]
            try:
                payout = expand_stringing_to_daily_payout(payout_source)
            except Exception:
                LOGGER.exception("Failed to expand P/O completion daily rows")
                return pd.DataFrame()
            if payout.empty:
                return payout
            if "project_name" not in payout.columns and "project" in payout.columns:
                payout["project_name"] = payout["project"]
            if "project" not in payout.columns and "project_name" in payout.columns:
                payout["project"] = payout["project_name"]
            if "month" not in payout.columns and "date" in payout.columns:
                payout["date"] = pd.to_datetime(payout["date"], errors="coerce")
                payout = payout.dropna(subset=["date"])
                payout["month"] = payout["date"].dt.to_period("M").to_timestamp()
            return payout

        return _cached_global_result(
            "stringing:po_completion_daily",
            _producer,
            clone=_clone_dataframe,
        )

    def _compute_po_completion_totals(
        scope_meta: dict[str, Any] | None,
        months_ts: list[pd.Timestamp],
        days_factor: float,
    ) -> str:
        plan_total = _stringing_planned_total_for_dates(
            scope_meta,
            months_ts,
            date_columns=_STRINGING_PO_DATE_COLUMNS,
        )
        has_plan_scope = _stringing_scope_has_plan(
            scope_meta,
            months_ts,
            date_columns=_STRINGING_PO_DATE_COLUMNS,
        )
        frame = _get_stringing_po_daily_frame()
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            if plan_total <= 0 and not has_plan_scope:
                return "-"
            total_txt = f"{plan_total:.1f} KM" if plan_total > 0 else "0.0 KM"
            done_txt = _format_summary_value(0.0, "KM")
            balance_txt = _format_summary_value(plan_total, "KM") if plan_total > 0 else "\u2014"
            return f"{total_txt} / {done_txt} / {balance_txt}"

        selected = (scope_meta or {}).get("selected") or {}
        projects = selected.get("projects") or []
        gangs = selected.get("gangs") or []
        method_values = selected.get("methods") or []

        scoped_base = _stringing_scope(frame, method_values)
        scoped_full = apply_filters(scoped_base, projects, months_ts, gangs)

        done_total = _sum_completion_totals(
            scoped_full,
            value_column="po_km",
            completion_column="po_completion_date",
            fallback_columns=[("po", 0.001)],
        ) or 0.0
        if done_total == 0.0 and isinstance(scoped_full, pd.DataFrame) and not scoped_full.empty and "daily_km" in scoped_full.columns:
            done_total = float(pd.to_numeric(scoped_full["daily_km"], errors="coerce").dropna().sum())

        done_txt = _format_summary_value(done_total, "KM")
        if plan_total > 0:
            total_txt = f"{plan_total:.1f} KM"
            balance_txt = _format_summary_value(max(plan_total - done_total, 0.0), "KM")
        elif has_plan_scope:
            total_txt = "0.0 KM"
            balance_txt = "\u2014"
        else:
            total_txt = "No Plan"
            balance_txt = "\u2014"
        return f"{total_txt} / {done_txt} / {balance_txt}"

    def _summarize_scope_for_cards(scope_meta: dict[str, Any] | None) -> dict[str, str]:
        mode = _normalize_mode((scope_meta or {}).get("mode"))
        summary = _empty_summary_payload(mode == "stringing")
        meta_info = summary.setdefault("_meta", {})
        meta_info.setdefault("loss_rows", [])
        meta_info.setdefault("mode", mode)
        meta_info.setdefault("is_stringing", mode == "stringing")
        meta_info.setdefault("unit_label", "KM" if mode == "stringing" else "MT")
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            return summary
        try:
            months_ts = _months_from_meta(scope_meta)
            days_factor = float(scope_meta.get("days_factor") or 30.0)
            is_stringing = mode == "stringing"
            metric_col = "daily_km" if is_stringing else "daily_prod_mt"
            unit_short = "KM" if is_stringing else "MT"
            prod_unit = "KM/month" if is_stringing else "MT/day"
            meta_signature = scope_meta.get("signature") or "nosig"
            selected = scope_meta.get("selected") or {}
            meta_info["unit_label"] = unit_short
            meta_info["is_stringing"] = is_stringing

            scoped_full = _scope_frame_from_store(scope_meta, "full").copy()
            scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()
            if scoped_full.empty and scoped_all.empty:
                return summary

            scope_keys = scope_meta.get("scopes") or {}
            project_gang_key = scope_keys.get("project_gang")
            components = _compute_mode_summary_components(
                scoped_full=scoped_full,
                scoped_all=scoped_all,
                months_ts=months_ts,
                days_factor=days_factor,
                metric_col=metric_col,
                is_stringing=is_stringing,
                meta_signature=meta_signature,
                cache_key=project_gang_key,
                scope_meta=scope_meta,
            )

            loss_scope = components["loss_scope"]
            history_scope = components["history_scope"]
            projects_count = components["projects_count"]
            gangs_count = components["gangs_count"]
            total_delivered = components["total_delivered"]
            total_lost = components["total_lost"]
            total_potential = components["total_potential"]
            balance_value = components["balance_value"]
            meta_info["loss_rows"] = components.get("loss_rows") or []
            balance_reference_delivered = total_delivered
            if is_stringing:
                deployment_scope = _normalize_deployment_filter(selected.get("stringing_scope"))
                if deployment_scope != "all":
                    try:
                        frames_all, _, _ = _build_scope_frames(
                            "stringing",
                            project_list=selected.get("projects", []),
                            gang_list=selected.get("gangs", []),
                            months_value=selected.get("months", []),
                            quick_range=selected.get("quick_range"),
                            method_values=selected.get("methods", []),
                            deployment_filter="all",
                        )
                        scoped_full_all = frames_all.get("full", pd.DataFrame()).copy()
                        scoped_all_all = frames_all.get("project_gang", pd.DataFrame()).copy()
                    except Exception:
                        scoped_full_all = pd.DataFrame()
                        scoped_all_all = pd.DataFrame()
                    if not (scoped_full_all.empty and scoped_all_all.empty):
                        scope_meta_all = dict(scope_meta)
                        scope_meta_all["selected"] = dict(selected)
                        scope_meta_all["selected"]["stringing_scope"] = "all"
                        all_components = _compute_mode_summary_components(
                            scoped_full=scoped_full_all,
                            scoped_all=scoped_all_all,
                            months_ts=months_ts,
                            days_factor=days_factor,
                            metric_col=metric_col,
                            is_stringing=True,
                            meta_signature=f"{meta_signature}::all-deployments",
                            cache_key=None,
                            scope_meta=scope_meta_all,
                        )
                        balance_reference_delivered = all_components["total_delivered"]

            def _collect_project_labels(frame: pd.DataFrame | None) -> list[str]:
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    return []
                labels: list[str] = []
                seen: set[str] = set()
                for column in (
                    "project_name",
                    "project",
                    "project_name_display",
                    "Project Name",
                    "project_code",
                ):
                    if column not in frame.columns:
                        continue
                    values = (
                        frame[column]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .replace("", pd.NA)
                        .dropna()
                    )
                    for value in values:
                        if value not in seen:
                            seen.add(value)
                            labels.append(value)
                return labels

            def _project_label_keys(text: object) -> tuple[list[str], list[str]]:
                base = str(text or "").strip()
                if not base:
                    return [], []
                parts = [base]
                if " : " in base:
                    left, right = base.split(" : ", 1)
                    parts.extend([left.strip(), right.strip()])
                norm_keys: list[str] = []
                compact_keys: list[str] = []
                for part in parts:
                    norm = _normalize_lower(part)
                    if norm and norm not in norm_keys:
                        norm_keys.append(norm)
                    compact = _compact_project_key(part)
                    if compact and compact not in compact_keys:
                        compact_keys.append(compact)
                match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{3,4})\b", base.upper())
                if match:
                    compact = _compact_project_key(f"{match.group(1)}{match.group(2)}")
                    if compact and compact not in compact_keys:
                        compact_keys.append(compact)
                return norm_keys, compact_keys

            tse_count: int | None = None
            if is_stringing:
                tse_norm_map, tse_alias_map = _get_stringing_tse_lookup()
                label_source = loss_scope if not loss_scope.empty else scoped_full
                labels = _collect_project_labels(label_source)
                matched_total = 0
                matched_any = False
                used_projects: set[str] = set()
                if labels and (tse_norm_map or tse_alias_map):
                    for label in labels:
                        norm_keys, compact_keys = _project_label_keys(label)
                        value, canonical_id = _resolve_tse_value(norm_keys, compact_keys, tse_norm_map, tse_alias_map)
                        if canonical_id and canonical_id not in used_projects and value is not None:
                            used_projects.add(canonical_id)
                            matched_total += int(value)
                            matched_any = True
                if matched_any:
                    tse_count = matched_total
                else:
                    fallback_scope = loss_scope if not loss_scope.empty else scoped_full
                    tse_count = 0
                    if {"method", "gang_name"}.issubset(fallback_scope.columns) and not fallback_scope.empty:
                        tse_mask = fallback_scope["method"].astype(str).str.strip().str.lower() == "tse"
                        tse_count = int(
                            fallback_scope.loc[tse_mask, "gang_name"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .nunique()
                        )

            prod_current = _avg_metric_value(scoped_full, metric_col, is_stringing)
            prod_history = _avg_metric_value(history_scope, metric_col, is_stringing)
            if prod_history == 0.0:
                prod_history = prod_current

            summary["projects"] = f"{projects_count:,}" if projects_count else "-"
            summary["gangs"] = f"{gangs_count:,}" if gangs_count else "-"
            if is_stringing:
                planned_total_value = _stringing_planned_total_for_dates(
                    scope_meta,
                    months_ts,
                    date_columns=_STRINGING_FS_DATE_COLUMNS,
                )
                has_plan_scope = _stringing_scope_has_plan(
                    scope_meta,
                    months_ts,
                    date_columns=_STRINGING_FS_DATE_COLUMNS,
                )
                done_txt = _format_summary_value(total_delivered, unit_short)
                if planned_total_value > 0:
                    total_txt = f"{planned_total_value:.1f} {unit_short}"
                    balance_txt = _format_summary_value(
                        max(planned_total_value - balance_reference_delivered, 0.0),
                        unit_short,
                    )
                elif has_plan_scope:
                    total_txt = "0.0 KM"
                    balance_txt = "\u2014"
                else:
                    total_txt = "No Plan"
                    balance_txt = "\u2014"
            else:
                tower_done = _count_completed_towers(loss_scope, months_ts)
                planned_towers, _ = _compute_planned_tower_layers(scoped_all, months_ts)
                total_towers = planned_towers or tower_done
                balance_towers = max(total_towers - tower_done, 0)
                total_txt = f"{total_towers:,}"
                done_txt = f"{tower_done:,}"
                balance_txt = f"{balance_towers:,}"
            summary["totals"] = f"{total_txt} / {done_txt} / {balance_txt}"
            prod_txt = _format_summary_value(prod_current, prod_unit, precision=2)
            hist_txt = _format_summary_value(prod_history, prod_unit, precision=2)
            summary["productivity"] = f"{prod_txt} / {hist_txt}"
            summary["lost_units"] = _format_summary_value(total_lost, unit_short)
            if is_stringing:
                summary["po_completion"] = _compute_po_completion_totals(scope_meta, months_ts, days_factor)
                summary["tse"] = "-" if tse_count is None else f"{tse_count:,}"
            return summary
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to compute %s summary card: %s", mode, exc)
            return summary

    # Charts OR AVP rows -> store-click-meta (robust & single source of truth)
    app.clientside_callback(
        """
        // charts OR AVP (row or overlay) -> store-click-meta
        function(lossClick, topClick, bottomClick, rowTs, tipTs) {
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;

        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];

        // --- AVP surfaces (row or overlay) only accept real, timestamped clicks
        try {
            const pid = JSON.parse(idPart);
            if (pid && (pid.type === "avp-row" || pid.type === "avp-tip")) {
            if (!prop.endsWith(".n_clicks_timestamp")) return NO; // ignore re-renders
            const ts = ctx.triggered[0].value;
            if (!ts || ts <= 0) return NO;                        // must be a real click
            const gang = pid.index;
            if (!gang) return NO;
            return { source: "g-actual-vs-bench", gang: String(gang), ts: Date.now() };
            }
        } catch(e) { /* not a pattern id; continue */ }

        // --- charts path
        let cd = null;
        if (idPart === "g-actual-vs-bench") cd = lossClick;
        else if (idPart === "g-top5")       cd = topClick;
        else if (idPart === "g-bottom5")    cd = bottomClick;
        else return NO;

        if (!cd || !cd.points || !cd.points.length) return NO;
        const pt = cd.points[0];

        // Extract gang robustly (y for horiz bars, x for vertical)
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;
        else if (typeof pt.x === "string") gang = pt.x;
        else if (pt.customdata){
            if (typeof pt.customdata === "string")      gang = pt.customdata;
            else if (Array.isArray(pt.customdata))      gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object") gang = pt.customdata.gang || pt.customdata.name || null;
        }
        if (!gang) return NO;

        return { source: idPart, gang: String(gang), ts: Date.now() };
        }
        """,
        Output("store-click-meta", "data"),
        [
        Input("g-actual-vs-bench", "clickData"),
        Input("g-top5", "clickData"),
        Input("g-bottom5", "clickData"),
        Input({"type":"avp-row","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        Input({"type":"avp-tip","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        ],
        prevent_initial_call=True,
    )


    # Keep trace gang selection in sync with the last click (chart or AVP)
    app.clientside_callback(
        """
        function(meta){
        if (!meta || !meta.gang) return window.dash_clientside.no_update;
        return meta.gang;
        }
        """,
        Output("store-selected-gang", "data"),
        Input("store-click-meta", "data"),
        prevent_initial_call=True,
    )

    # Global modal: charts or AVP rows drive the shared click store
    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, rowTs, tipTs){
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;
        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];
        try {
            const pid = JSON.parse(idPart);
            if (pid && (pid.type === "global-performance-avp-row" || pid.type === "global-performance-avp-tip")) {
                if (!prop.endsWith(".n_clicks_timestamp")) return NO;
                const ts = ctx.triggered[0].value;
                if (!ts || ts <= 0) return NO;
                const gang = pid.index;
                if (!gang) return NO;
                return { source: "global-performance-actual-vs-bench", gang: String(gang), ts: Date.now() };
            }
        } catch(e) { /* ignore pattern parse */ }

        let cd = null;
        if (idPart === "global-performance-actual-vs-bench") cd = lossClick;
        else if (idPart === "global-performance-top5")       cd = topClick;
        else if (idPart === "global-performance-bottom5")    cd = bottomClick;
        else return NO;

        if (!cd || !cd.points || !cd.points.length) return NO;
        const pt = cd.points[0];
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;
        else if (typeof pt.x === "string") gang = pt.x;
        else if (pt.customdata){
            if (typeof pt.customdata === "string")      gang = pt.customdata;
            else if (Array.isArray(pt.customdata))       gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object")  gang = pt.customdata.gang || pt.customdata.name || null;
        }
        if (!gang) return NO;
        return { source: idPart, gang: String(gang), ts: Date.now() };
        }
        """,
        Output("store-global-performance-click-meta", "data"),
        [
            Input("global-performance-actual-vs-bench", "clickData"),
            Input("global-performance-top5", "clickData"),
            Input("global-performance-bottom5", "clickData"),
            Input({"type": "global-performance-avp-row", "index": dash.dependencies.ALL}, "n_clicks_timestamp"),
            Input({"type": "global-performance-avp-tip", "index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        ],
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta){
        if (!meta || !meta.gang) return window.dash_clientside.no_update;
        return meta.gang;
        }
        """,
        Output("global-performance-selected-gang", "data"),
        Input("store-global-performance-click-meta", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(lossClick, topClick, bottomClick, rowTs, tipTs){
        const C  = window.dash_clientside, NO = C.no_update, ctx = C.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return NO;
        const prop   = ctx.triggered[0].prop_id || "";
        const idPart = prop.split(".")[0];
        try {
            const pid = JSON.parse(idPart);
            if (pid && (pid.type === "project-modal-avp-row" || pid.type === "project-modal-avp-tip")) {
                if (!prop.endsWith(".n_clicks_timestamp")) return NO;
                const ts = ctx.triggered[0].value;
                if (!ts || ts <= 0) return NO;
                const gang = pid.index;
                if (!gang) return NO;
                return { source: "project-modal-actual-vs-bench", gang: String(gang), ts: Date.now() };
            }
        } catch(e) { /* ignore */ }

        let cd = null;
        if (idPart === "project-modal-actual-vs-bench") cd = lossClick;
        else if (idPart === "project-modal-top5")       cd = topClick;
        else if (idPart === "project-modal-bottom5")    cd = bottomClick;
        else return NO;

        if (!cd || !cd.points || !cd.points.length) return NO;
        const pt = cd.points[0];
        let gang = null;
        if (typeof pt.y === "string")      gang = pt.y;
        else if (typeof pt.x === "string") gang = pt.x;
        else if (pt.customdata){
            if (typeof pt.customdata === "string")      gang = pt.customdata;
            else if (Array.isArray(pt.customdata))       gang = pt.customdata.find(v => typeof v === "string") || null;
            else if (typeof pt.customdata === "object") gang = pt.customdata.gang || pt.customdata.name || null;
        }
        if (!gang) return NO;
        return { source: idPart, gang: String(gang), ts: Date.now() };
        }
        """,
        Output("store-project-modal-click-meta", "data"),
        [
            Input("project-modal-actual-vs-bench", "clickData"),
            Input("project-modal-top5", "clickData"),
            Input("project-modal-bottom5", "clickData"),
            Input({"type":"project-modal-avp-row","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
            Input({"type":"project-modal-avp-tip","index": dash.dependencies.ALL}, "n_clicks_timestamp"),
        ],
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta){
        if (!meta || !meta.gang) return window.dash_clientside.no_update;
        return meta.gang;
        }
        """,
        Output("project-modal-selected-gang", "data"),
        Input("store-project-modal-click-meta", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta){
        if(!meta || !meta.source || !meta.gang) return "";
        const CHART_SOURCES = new Set(["g-actual-vs-bench","g-top5","g-bottom5"]);
        if (!CHART_SOURCES.has(meta.source)) return "";

        // retry briefly so the anchor exists before we scroll
        let tries = 0;
        function go(){
            const anchor = document.getElementById("trace-anchor") || document.getElementById("tables-anchor");
            if (!anchor) { if (tries++ < 25) setTimeout(go, 60); return; }
            anchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        setTimeout(go, 0);
        return String(Date.now());
        }
        """,
        Output("scroll-wire", "children"),
        Input("store-click-meta", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(meta, scrollTarget){
        const ctx = window.dash_clientside && window.dash_clientside.callback_context;
        const trig = ctx && ctx.triggered && ctx.triggered.length ? ctx.triggered[0] : null;
        const prop = trig && trig.prop_id ? trig.prop_id : "";
        const NO = window.dash_clientside.no_update;

        if (prop === "store-project-modal-click-meta.data") {
            if(!meta || !meta.source || !meta.gang) return "";
            const CHART_SOURCES = new Set(["project-modal-actual-vs-bench","project-modal-top5","project-modal-bottom5"]);
            if (!CHART_SOURCES.has(meta.source)) return "";

            let tries = 0;
            function go(){
                const anchor = document.getElementById("project-modal-trace-anchor");
                if (!anchor) { if (tries++ < 25) setTimeout(go, 60); return; }
                anchor.scrollIntoView({ behavior: "smooth", block: "start" });
            }
            setTimeout(go, 0);
            return String(Date.now());
        }

        if (prop === "project-modal-scroll-target.data") {
            if (!scrollTarget || !scrollTarget.anchor) return NO;
            const anchorId = scrollTarget.anchor;
            let tries = 0;
            function go(){
                const target = document.getElementById(anchorId);
                if (!target) {
                    if (tries++ < 40) setTimeout(go, 75);
                    return;
                }
                const rect = target.getBoundingClientRect();
                const hidden = rect.height === 0;
                if (hidden && tries++ < 40) {
                    setTimeout(go, 75);
                    return;
                }
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
            setTimeout(go, 0);
            return String(Date.now());
        }

        return NO;
        }
        """,
        Output("project-modal-scroll-wire", "children"),
        Input("store-project-modal-click-meta", "data"),
        Input("project-modal-scroll-target", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(historyState){
        const NO = window.dash_clientside.no_update;
        if (!historyState || !historyState.action) return NO;
        const ACTION = historyState.action;
        const PARAM = "projectModal";
        try {
            const url = new URL(window.location.href);
            if (ACTION === "open") {
                if (url.searchParams.get(PARAM) === "1") return NO;
                url.searchParams.set(PARAM, "1");
                window.history.pushState({ projectModal: true }, "", url.toString());
            } else if (ACTION === "close") {
                if (!url.searchParams.has(PARAM)) return NO;
                window.history.back();
            }
        } catch (err) {
            console.warn("project-modal history update failed", err);
        }
        return Date.now();
        }
        """,
        Output("project-modal-history-wire", "children"),
        Input("store-project-modal-history", "data"),
        prevent_initial_call=True,
    )



    @app.callback(
        Output("f-project", "value"),
        Output("f-gang", "value"),
        Output("f-stringing-scope", "value"),
        Input("btn-reset-filters", "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_filter_reset(
        reset_clicks: int | None,
    ) -> tuple[Any, Any, str]:
        if not reset_clicks:
            raise PreventUpdate
        return None, None, "all"
    @app.callback(
        Output("store-filtered-scope", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("store-stringing-scope", "data"),
        prevent_initial_call=False,
    )
    def _sync_filtered_scope_store(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        stringing_scope: str | None,
    ) -> dict[str, Any]:
        eff_mode = "erection"
        project_list = _normalize_str_list(_ensure_list(projects))
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        method_values = _method_filters_for_scope(stringing_scope)
        if not method_values:
            method_values = list(_STRINGING_METHODS)
        method_list = _normalize_str_list(method_values, lower=True)

        frames, months_ts, days_factor = _build_scope_frames(
            eff_mode,
            project_list=project_list,
            gang_list=gang_list,
            months_value=months_list,
            quick_range=quick_range,
            method_values=method_values,
            deployment_filter=stringing_scope,
        )

        scope_keys = {name: _remember_scope_frame(frame) for name, frame in frames.items()}
        rows_meta = {name: int(len(frame.index)) for name, frame in frames.items()}
        signature_payload = {
            "mode": eff_mode,
            "projects": project_list,
            "gangs": gang_list,
            "months": months_list,
            "quick_range": quick_range,
            "methods": method_list,
            "stringing_scope": _normalize_deployment_filter(stringing_scope),
        }
        signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return {
            "mode": eff_mode,
            "signature": signature,
            "scopes": scope_keys,
            "rows": rows_meta,
            "days_factor": days_factor,
            "months_iso": [ts.isoformat() for ts in months_ts],
            "selected": {
            "projects": project_list,
            "gangs": gang_list,
            "months": months_list,
            "quick_range": quick_range,
            "methods": method_list,
            "stringing_scope": _normalize_deployment_filter(stringing_scope),
        },
        }


    @app.callback(
        Output("project-modal-erections-range", "start_date"),
        Output("project-modal-erections-range", "end_date"),
        Output("project-modal-stringing-range", "start_date"),
        Output("project-modal-stringing-range", "end_date"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def _sync_modal_completion_ranges(months, quick_range):
        start, end = _completion_range_from_month_selection(months, quick_range)
        start_iso = start.date().isoformat()
        end_iso = end.date().isoformat()
        return start_iso, end_iso, start_iso, end_iso


    @app.callback(
        Output("store-stringing-scope", "data"),
        Input("f-stringing-scope", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("project-modal-stringing-scope", "value"),
        State("store-stringing-scope", "data"),
        prevent_initial_call=False,
    )
    def _sync_stringing_scope_store(
        home_value: str | None,
        reset_clicks: int | None,
        modal_value: str | None,
        current_value: str | None,
    ) -> Any:
        ctx = dash.callback_context
        current_normalized = _normalize_deployment_filter(current_value)
        next_value = current_normalized or "all"
        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == "btn-reset-filters":
                next_value = "all"
            elif trigger == "f-stringing-scope" and home_value:
                next_value = home_value
            elif trigger == "project-modal-stringing-scope" and modal_value:
                next_value = modal_value
        normalized = _normalize_deployment_filter(next_value)
        if normalized == current_normalized:
            return dash.no_update
        return normalized


    @app.callback(
        Output("project-modal-stringing-scope", "value"),
        Input("store-stringing-scope", "data"),
        State("project-modal-stringing-scope", "value"),
        prevent_initial_call=False,
    )
    def _sync_stringing_scope_modal(
        scope_value: str | None,
        modal_value: str | None,
    ) -> Any:
        normalized = _normalize_deployment_filter(scope_value)
        modal_norm = _normalize_deployment_filter(modal_value)
        if normalized == modal_norm:
            return dash.no_update
        return normalized


    @app.callback(
        Output("f-project", "options"),
        Input("store-filtered-scope", "data"),
    )
    def update_project_options(scope_meta: dict[str, Any] | None) -> list[dict[str, str]]:
        try:
            scope = _scope_frame_from_store(scope_meta, "month")
            if scope.empty:
                return []
            proj_col = "project_name" if "project_name" in scope.columns else ("project" if "project" in scope.columns else None)
            if not proj_col:
                return []
            projects = (
                scope[proj_col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            projects = sorted({p for p in projects if p})
            return [{"label": project, "value": project} for project in projects]
        except Exception as exc:
            LOGGER.exception("Failed to build project options: %s", exc)
            return []

    @app.callback(
        Output("f-gang", "options"),
        Input("store-filtered-scope", "data"),
    )
    def update_gang_options(scope_meta: dict[str, Any] | None) -> list[dict[str, str]]:
        try:
            scope = _scope_frame_from_store(scope_meta, "project")
            if scope.empty or "gang_name" not in scope.columns:
                return []
            gangs = (
                scope["gang_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            gangs = sorted({g for g in gangs if g})
            return [{"label": gang, "value": gang} for gang in gangs]
        except Exception as exc:
            LOGGER.exception("Failed to build gang options: %s", exc)
            return []

    @app.callback(
        Output("f-month", "options"),
        Input("f-project", "value"),
        Input("f-gang", "value"),
        Input("f-quick-range", "value"),
        Input("store-stringing-scope", "data"),
    )
    def update_month_options(
        projects: Sequence[str] | None,
        gangs: Sequence[str] | None,
        quick_range: str | None,
        stringing_scope: str | None,
    ) -> list[dict[str, str]]:
        project_list = _normalize_str_list(_ensure_list(projects))
        gang_list = _normalize_str_list(_ensure_list(gangs))
        method_values = _method_filters_for_scope(stringing_scope)

        try:
            frames, _, _ = _build_scope_frames(
                "erection",
                project_list=project_list,
                gang_list=gang_list,
                months_value=[],
                quick_range=quick_range,
                method_values=method_values,
                deployment_filter=stringing_scope,
            )
            scope = frames.get("project_gang", pd.DataFrame())
            if scope.empty or "month" not in scope.columns:
                return []
            months_series = scope["month"].dropna().unique()
            base_months = pd.to_datetime(months_series, errors="coerce").dropna().tolist()
            months = sorted(base_months)
            if quick_range:
                allowed = set(resolve_months(None, quick_range))
                months = [m for m in months if m in allowed]
            if not months:
                months = sorted(base_months)
            if not months:
                return []
            return [{"label": m.strftime("%b %Y"), "value": m.strftime("%Y-%m")} for m in months]
        except Exception as exc:
            LOGGER.exception("Failed to build month options: %s", exc)
            return []

    @app.callback(
        Output("global-performance-projects", "options"),
        Input("store-filtered-scope", "data"),
    )
    def _populate_global_performance_project_options(_: dict[str, Any] | None) -> list[dict[str, str]]:
        method_values = _default_stringing_method_values()
        try:
            frames, _, _ = _build_scope_frames(
                "erection",
                project_list=[],
                gang_list=[],
                months_value=[],
                quick_range=None,
                method_values=method_values,
                deployment_filter="all",
            )
            scope = frames.get("project_gang", pd.DataFrame())
            if scope.empty:
                return []
            proj_col = "project_name" if "project_name" in scope.columns else ("project" if "project" in scope.columns else None)
            if not proj_col:
                return []
            projects = (
                scope[proj_col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            projects = sorted({p for p in projects if p})
            return [{"label": project, "value": project} for project in projects]
        except Exception as exc:
            LOGGER.exception("Failed to populate global performance project options: %s", exc)
            return []

    @app.callback(
        Output("global-performance-months", "options"),
        Input("global-performance-projects", "value"),
    )
    def _populate_global_performance_month_options(
        projects: Sequence[str] | None,
    ) -> list[dict[str, str]]:
        project_list = _normalize_str_list(_ensure_list(projects))
        method_values = _default_stringing_method_values()
        try:
            frames, _, _ = _build_scope_frames(
                "erection",
                project_list=project_list,
                gang_list=[],
                months_value=[],
                quick_range=None,
                method_values=method_values,
                deployment_filter="all",
            )
            scope = frames.get("project_gang", pd.DataFrame())
            if scope.empty or "month" not in scope.columns:
                return []
            months_series = pd.to_datetime(scope["month"], errors="coerce").dropna()
            months = sorted({pd.Period(ts, "M").to_timestamp() for ts in months_series})
            return [{"label": ts.strftime("%b %Y"), "value": ts.strftime("%Y-%m")} for ts in months]
        except Exception as exc:
            LOGGER.exception("Failed to populate global performance month options: %s", exc)
            return []

    @app.callback(
        Output("store-global-performance-scope", "data"),
        Input("global-performance-projects", "value"),
        Input("global-performance-months", "value"),
        Input("global-performance-min-erections", "value"),
        Input("store-global-performance-mode", "data"),
        Input("store-stringing-scope", "data"),
    )
    def _sync_global_performance_scope_store(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        min_erections_value: Any,
        mode_value: Any,
        stringing_scope: str | None,
    ) -> dict[str, Any]:
        project_list = _normalize_str_list(_ensure_list(projects))
        months_list = _normalize_str_list(_ensure_list(months))
        eff_mode = _normalize_mode(mode_value) or "erection"
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        deployment_filter = stringing_scope if eff_mode == "stringing" else None
        method_values = (
            _method_filters_for_scope(deployment_filter)
            if eff_mode == "stringing"
            else _default_stringing_method_values()
        )
        try:
            payload = _build_scope_meta_payload(
                eff_mode=eff_mode,
                project_list=project_list,
                gang_list=[],
                months_list=months_list,
                quick_range=None,
                method_values=method_values,
                method_list=_normalize_str_list(method_values, lower=True),
                deployment_filter=deployment_filter,
            )
            payload["min_erections"] = _normalize_min_erections(min_erections_value)
            if isinstance(payload.get("selected"), dict):
                payload["selected"]["min_erections"] = payload["min_erections"]
            return payload
        except Exception as exc:
            LOGGER.exception("Failed to build global performance scope: %s", exc)
            return dash.no_update

    @app.callback(
        Output("global-performance-modal", "is_open"),
        Output("store-global-performance-mode", "data"),
        Input("btn-open-global-performance-erection", "n_clicks"),
        Input("btn-open-global-performance-stringing", "n_clicks"),
        Input("global-performance-modal-close", "n_clicks"),
        State("global-performance-modal", "is_open"),
        State("store-global-performance-mode", "data"),
    )
    def _toggle_global_performance_modal(
        open_erection: int | None,
        open_stringing: int | None,
        close_clicks: int | None,
        is_open: bool | None,
        mode_value: Any,
    ) -> tuple[bool, str]:
        ctx = dash.callback_context
        current_mode = _normalize_mode(mode_value) or "erection"
        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        else:
            trigger = None

        if trigger == "global-performance-modal-close":
            return False, current_mode
        if trigger == "btn-open-global-performance-erection":
            return True, "erection"
        if trigger == "btn-open-global-performance-stringing":
            target_mode = "stringing" if config.enable_stringing else "erection"
            return True, target_mode
        return bool(is_open), current_mode

    @app.callback(
        Output("global-performance-topbot-mode-label", "children"),
        Output("global-performance-tbl-idle-intervals", "columns"),
        Output("global-performance-tbl-daily-prod", "columns"),
        Output("global-performance-benchmark-table", "columns"),
        Input("store-global-performance-mode", "data"),
    )
    def _sync_global_performance_ui(mode_value):
        eff_mode = _normalize_mode(mode_value) or "erection"
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        is_stringing = eff_mode == "stringing"
        idle_columns = [
            {"name": "Gang", "id": "gang_name"},
            {"name": "Interval Start", "id": "interval_start"},
            {"name": "Interval End", "id": "interval_end"},
            {"name": "Raw Gap (days)", "id": "raw_gap_days"},
            {"name": "Idle Counted (days)", "id": "idle_days_capped"},
            {
                "name": "Baseline (KM/Month)" if is_stringing else "Baseline (MT/day)",
                "id": "baseline",
            },
            {
                "name": "Cumulative Loss (KM)" if is_stringing else "Cumulative Loss (MT)",
                "id": "cumulative_loss",
            },
        ]
        daily_columns = [
            {"name": "Gang", "id": "gang_name"},
            {"name": "Project", "id": "project_name"},
            {"name": "Date", "id": "date"},
            {
                "name": "KM/Month" if is_stringing else "MT/day",
                "id": "daily_prod_mt",
            },
        ]
        label = "Stringing" if is_stringing else "Erection"
        benchmark_columns = [
            {"name": "Gang", "id": "name"},
            {"name": "Project", "id": "project"},
            {"name": "Last Worked At", "id": "last_worked_at"},
            {
                "name": "Spans Completed" if is_stringing else "Erections",
                "id": "erections",
            },
            {
                "name": "Current KM/month" if is_stringing else "Current MT/day",
                "id": "current_rate",
            },
            {
                "name": "Baseline KM/month" if is_stringing else "Baseline MT/day",
                "id": "baseline_rate",
            },
        ]
        return label, idle_columns, daily_columns, benchmark_columns

    @app.callback(
        Output("global-performance-benchmark-label", "children"),
        Input("store-global-performance-mode", "data"),
    )
    def _sync_global_performance_benchmark_label(mode_value):
        eff_mode = _normalize_mode(mode_value) or "erection"
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        unit = "KM/month" if eff_mode == "stringing" else "MT/day"
        return f"Benchmark ({unit})"

    @app.callback(
        Output("global-performance-erections-threshold-label", "children"),
        Input("store-global-performance-mode", "data"),
    )
    def _sync_global_performance_min_erections_label(mode_value):
        eff_mode = _normalize_mode(mode_value) or "erection"
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        return "Min Spans Completed" if eff_mode == "stringing" else "Min Erections"

    @app.callback(
        Output("global-performance-benchmark-table", "data"),
        Output("global-performance-benchmark-status", "children"),
        Input("store-global-performance-scope", "data"),
        Input("global-performance-benchmark", "value"),
    )
    def _populate_global_performance_benchmark_table(
        scope_meta: dict[str, Any] | None,
        benchmark_value: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        eff_mode = _normalize_mode((scope_meta or {}).get("mode"))
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        is_stringing = eff_mode == "stringing"
        unit_label = "KM/month" if is_stringing else "MT/day"
        min_erections = _min_erections_from_meta(scope_meta)
        if not isinstance(scope_meta, dict):
            return [], "Select filters to populate the benchmark view."

        benchmark: float | None = None
        if isinstance(benchmark_value, (int, float)):
            benchmark = float(benchmark_value)
        elif isinstance(benchmark_value, str) and benchmark_value.strip():
            try:
                benchmark = float(benchmark_value.strip())
            except ValueError:
                benchmark = None
        if benchmark is None:
            return [], f"Enter a benchmark in {unit_label} to view the leading gangs."

        project_df = _scope_frame_from_store(scope_meta, "project").copy()
        if project_df.empty or "gang_name" not in project_df.columns:
            return [], "No gang activity found for the selected scope."
        project_df = _filter_frame_for_min_erections(project_df, min_erections)
        if project_df.empty:
            if min_erections is not None:
                noun = "spans" if is_stringing else "erections"
                return [], f"No gangs have more than {min_erections} completed {noun} for this selection."
            return [], "No gang activity found for the selected scope."

        if "daily_prod_mt" not in project_df.columns:
            alt_series = None
            if is_stringing and "daily_km" in project_df.columns:
                alt_series = project_df["daily_km"]
            elif "daily_prod_value" in project_df.columns:
                alt_series = project_df["daily_prod_value"]
            if alt_series is not None:
                project_df["daily_prod_mt"] = pd.to_numeric(alt_series, errors="coerce")
            else:
                project_df["daily_prod_mt"] = np.nan
        else:
            project_df["daily_prod_mt"] = pd.to_numeric(project_df["daily_prod_mt"], errors="coerce")

        metric_series = project_df["daily_prod_mt"]
        project_df = project_df.assign(
            gang_name=project_df["gang_name"].astype(str).str.strip(),
            daily_prod_mt=metric_series,
        ).dropna(subset=["gang_name", "daily_prod_mt"])
        if project_df.empty:
            return [], "No valid productivity records available for this scope."

        date_column = None
        for candidate in ("date", "completion_date", "interval_end"):
            if candidate in project_df.columns:
                date_column = candidate
                break
        if date_column:
            project_df["_gp_date"] = pd.to_datetime(project_df[date_column], errors="coerce")
        else:
            project_df["_gp_date"] = pd.NaT

        avg_series = (
            project_df.groupby("gang_name")["daily_prod_mt"]
            .mean()
            .rename("current_metric")
        )
        if "location_no" in project_df.columns:
            count_series = (
                project_df.groupby("gang_name")["location_no"]
                .nunique(dropna=True)
                .rename("erections")
            )
        else:
            count_series = project_df.groupby("gang_name").size().rename("erections")

        latest_rows = (
            project_df.sort_values("_gp_date")
            .groupby("gang_name", as_index=False)
            .tail(1)
        )
        if "project_name" in latest_rows.columns:
            project_map = latest_rows.set_index("gang_name")["project_name"]
        else:
            project_map = pd.Series(dtype=str)
        date_map = latest_rows.set_index("gang_name")["_gp_date"]

        baseline_mode = "stringing" if eff_mode == "stringing" else "erection"
        project_baselines, _ = _get_project_baselines(baseline_mode)

        summary = (
            avg_series.to_frame()
            .join(count_series, how="left")
            .reset_index()
            .rename(columns={"gang_name": "name"})
        )
        summary["project"] = summary["name"].map(project_map).fillna("\u2014")
        summary["last_worked_at"] = summary["name"].map(date_map)
        summary["baseline_metric"] = summary["project"].map(project_baselines or {})
        summary["erections"] = summary["erections"].fillna(0).astype(int)

        qualifying = summary[summary["current_metric"] >= benchmark].copy()
        if qualifying.empty:
            return [], f"No gangs exceed {benchmark:.2f} {unit_label} for this selection."

        def _fmt_rate(value: Any) -> str:
            if value is None or pd.isna(value):
                return "\u2014"
            try:
                return f"{float(value):.2f}"
            except Exception:
                return "\u2014"

        def _fmt_date(value: Any) -> str:
            if isinstance(value, str):
                try:
                    parsed = pd.to_datetime(value)
                except Exception:
                    return value or "\u2014"
                return parsed.strftime("%d-%b-%Y") if not pd.isna(parsed) else "\u2014"
            if pd.isna(value):
                return "\u2014"
            try:
                return pd.to_datetime(value).strftime("%d-%b-%Y")
            except Exception:
                return "\u2014"

        qualifying = qualifying.sort_values("current_metric", ascending=False)
        qualifying["last_worked_at"] = qualifying["last_worked_at"].map(_fmt_date)
        qualifying["current_rate"] = qualifying["current_metric"].map(_fmt_rate)
        qualifying["baseline_rate"] = qualifying["baseline_metric"].map(_fmt_rate)
        qualifying["erections"] = qualifying["erections"].astype(int)

        data = qualifying[
            ["name", "project", "last_worked_at", "erections", "current_rate", "baseline_rate"]
        ].to_dict("records")
        status = f"{len(qualifying)} gang(s) above {benchmark:.2f} {unit_label}."
        return data, status

    @app.callback(
        Output("global-performance-avp-list", "children"),
        Output("global-performance-actual-vs-bench", "figure"),
        Output("global-performance-top5", "figure"),
        Output("global-performance-bottom5", "figure"),
        Input("store-global-performance-scope", "data"),
        Input("global-performance-topbot-metric", "value"),
    )
    def _update_global_performance_modal(
        scope_meta: dict[str, Any] | None,
        topbot_metric: str | None,
    ):
        empty_fig = go.Figure()
        if not isinstance(scope_meta, dict):
            return (
                html.Div("Select at least one project/month to view gang performance.", className="text-muted"),
                empty_fig,
                empty_fig,
                empty_fig,
            )
        min_erections_filter = _min_erections_from_meta(scope_meta)
        eff_mode = _normalize_mode(scope_meta.get("mode"))
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        is_stringing_mode = eff_mode == "stringing"
        try:
            (
                *_kpi,
                avp_children,
                fig_loss,
                fig_top,
                fig_bottom,
                _fig_project,
            ) = _compute_dashboard_outputs(
                scope_meta,
                topbot_metric,
                avp_namespace="global-performance-avp",
                summarizer=_summarize_scope_for_cards,
                summary_factory=_empty_summary_payload,
                min_erections=min_erections_filter,
            )
        except PreventUpdate:
            if min_erections_filter is not None:
                noun = "spans" if is_stringing_mode else "erections"
                message = f"No gangs have more than {min_erections_filter} completed {noun} for this selection."
            else:
                message = "No data available for the selected scope."
            return (
                html.Div(message, className="text-muted"),
                empty_fig,
                empty_fig,
                empty_fig,
            )

        avp_children = avp_children or html.Div("No gangs available for this selection.", className="text-muted")
        return avp_children, fig_loss, fig_top, fig_bottom

    def _resolve_reset_month_value() -> str:
        try:
            df = data_selector.select("erection")
            latest_date = None
            if isinstance(df, pd.DataFrame) and not df.empty and "date" in df.columns:
                dates = pd.to_datetime(df["date"], errors="coerce").dropna()
                if not dates.empty:
                    latest_date = dates.max()
            if latest_date is None:
                return datetime.today().strftime("%Y-%m")
            return pd.Timestamp(latest_date).strftime("%Y-%m")
        except Exception:
            return datetime.today().strftime("%Y-%m")

    def _pick_latest_month_value(
        options: Sequence[dict[str, Any]] | None,
        current_value: Sequence[str] | str | None,
    ) -> Any:
        if isinstance(current_value, list) and len(current_value) == 0:
            return dash.no_update
        if current_value:
            selected = set(current_value if isinstance(current_value, (list, tuple)) else [current_value])
            opt_values = {opt.get("value") for opt in (options or []) if isinstance(opt, dict)}
            if selected & opt_values:
                return dash.no_update
        opt_values = [opt.get("value") for opt in (options or []) if isinstance(opt, dict)]
        if not opt_values:
            return dash.no_update

        def _parse(val: str) -> int:
            try:
                y, m = val.split("-")
                return int(y) * 100 + int(m)
            except Exception:
                return -1

        latest = max(opt_values, key=_parse)
        return [latest]

    @app.callback(
        Output("f-month", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("f-month", "options"),
        Input("f-quick-range", "value"),
        State("f-month", "value"),
        prevent_initial_call=True,
    )
    def sync_month_filter_value(
        reset_clicks: int | None,
        options: Sequence[dict[str, Any]] | None,
        quick_range_value: str | None,
        current_value: Sequence[str] | str | None,
    ) -> Any:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "btn-reset-filters":
            return [_resolve_reset_month_value()]
        if trigger_id == "f-quick-range" and quick_range_value:
            return None
        if quick_range_value:
            return dash.no_update
        return _pick_latest_month_value(options, current_value)

    @app.callback(
        Output("f-quick-range", "value"),
        Input("btn-reset-filters", "n_clicks"),
        Input("link-clear-quick-range", "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_quick_range_value(
        reset_clicks: int | None,
        clear_quick_clicks: int | None,
    ) -> Any:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id in {"btn-reset-filters", "link-clear-quick-range"}:
            return None
        return dash.no_update

    @app.callback(
        Output("link-clear-quick-range", "style"),
        Input("f-quick-range", "value"),
        prevent_initial_call=False,
    )
    def toggle_clear_quick_range_link(quick_range: str | None) -> dict:
        if quick_range:
            return {}
        return {"display": "none"}


    @app.callback(
        Output("label-resp-period", "children"),
        Output("label-stringing-plan-period", "children"),
        Output("label-perf-period", "children"),
        Output("label-gang-period", "children"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
    )
    def update_period_labels(
        months: Sequence[str] | None,
        quick_range: str | None,
    ) -> tuple[str, str, str, str]:
        month_list = _ensure_list(months)
        months_ts = resolve_months(month_list, quick_range)
        label = _format_period_label(months_ts)
        return label, label, label, label

    # ---- Project Overview (dynamic body) -----------------------------------------
    @app.callback(
        Output("pd-title", "children"),
        Output("project-details", "children"),
        Input("f-project", "value"),
        prevent_initial_call=False,
    )
    def show_project_details(selected_project):
        """Render the project overview grid or an informative message."""

        default_title = "Project Overview"

        def _clean_text(raw: Any) -> str:
            if raw is None:
                return ""
            text = str(raw).strip()
            if not text:
                return ""
            if any(marker in text for marker in ("Ãƒ", "Ã‚")):
                try:
                    text = text.encode("latin-1").decode("utf-8").strip()
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            return text

        def _normalize_for_match(raw: Any) -> str:
            text = _clean_text(raw)
            return " ".join(text.lower().split())

        if isinstance(selected_project, (list, tuple)):
            cleaned = [_clean_text(value) for value in selected_project if _clean_text(value)]
            if len(cleaned) != 1:
                return (
                    default_title,
                    html.Div("Select a single project to view its details.", className="project-empty"),
                )
            selected_project = cleaned[0]
        else:
            selected_project = _clean_text(selected_project)

        if not selected_project:
            return (
                default_title,
                html.Div("Select a single project to view its details.", className="project-empty"),
            )

        if not project_info_provider:
            return (
                default_title,
                html.Div("No 'Project Details' source configured.", className="project-empty"),
            )

        df_info = project_info_provider()
        if df_info is None or df_info.empty:
            return (
                default_title,
                html.Div("No 'Project Details' sheet found in the source workbook.", className="project-empty"),
            )

        df_info = df_info.copy()
        target_norm = _normalize_for_match(selected_project)
        # Accept multiple identifier variants for robust matching
        candidate_columns = [
            col
            for col in (
                "Project Name",
                "project_name",
                "project_code",
                "Project Code",
                "key_name",
            )
            if col in df_info.columns
        ]

        row = pd.DataFrame()
        # 1) strict normalized equality against known identifier columns
        for col in candidate_columns:
            try:
                mask = df_info[col].apply(_normalize_for_match) == target_norm
            except Exception:
                mask = pd.Series(False, index=df_info.index)
            if mask.any():
                row = df_info.loc[mask]
                break

        # 2) relaxed contains on human name fields (normalized)
        if row.empty:
            for human_col in ("Project Name", "project_name"):
                if human_col in df_info.columns:
                    series = df_info[human_col].astype(str).apply(_normalize_for_match)
                    mask = series.str.contains(target_norm, case=False, na=False)
                    if mask.any():
                        row = df_info.loc[mask]
                        break

        # 3) compact code match (remove non-alphanumerics) for project codes (any case variant)
        if row.empty:
            import re as _re

            def _compact(s: str) -> str:
                return _re.sub(r"[^a-z0-9]", "", (s or "").lower())

            target_comp = _compact(selected_project)
            for code_col in ("project_code", "Project Code"):
                if code_col in df_info.columns:
                    try:
                        comp_series = (
                            df_info[code_col]
                            .astype(str)
                            .map(_clean_text)
                            .map(_compact)
                        )
                        mask = comp_series == target_comp
                        if mask.any():
                            row = df_info.loc[mask]
                            break
                    except Exception:
                        continue

        if row.empty:
            return (
                default_title,
                html.Div(f"No project details found for {selected_project}.", className="project-empty"),
            )

        record = row.iloc[0]

        def fmt_txt(key: str) -> str:
            value = record.get(key, "")
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            return _clean_text(value)

        def fmt_date(key: str) -> str:
            value = record.get(key, None)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            try:
                return pd.to_datetime(value).strftime("%d-%m-%Y")
            except Exception:
                return _clean_text(value)

        display_name = fmt_txt("Project Name") or fmt_txt("project_name") or selected_project

        # Normalize PCH display using centralized normalizer if present
        try:
            from .pch_normalizer import normalize_pch as _norm_pch_display
        except Exception:
            def _norm_pch_display(v):
                return str(v or "").strip()

        body = html.Div(
            [
                html.Div(
                    [
                        html.P("PROJECT NAME", className="project-label"),
                        html.H6(fmt_txt("project_name") or fmt_txt("Project Name"), className="project-value"),
                        html.P("CLIENT", className="project-label"),
                        html.H6(fmt_txt("client_name"), className="project-value"),
                        html.P("NOA START", className="project-label"),
                        html.H6(fmt_date("noa_start"), className="project-value"),
                        html.P("LOA END", className="project-label"),
                        html.H6(fmt_date("loa_end"), className="project-value"),
                    ],
                    className="project-col",
                ),
                html.Div(
                    [
                        html.P("PCH", className="project-label"),
                        html.H6(_norm_pch_display(record.get("pch")), className="project-value"),
                        html.P("REGIONAL MANAGER", className="project-label"),
                        html.H6(fmt_txt("regional_mgr"), className="project-value"),
                        html.P("PROJECT MANAGER", className="project-label"),
                        html.H6(fmt_txt("project_mgr"), className="project-value"),
                        html.P("PLANNING ENGINEER", className="project-label"),
                        html.H6(fmt_txt("planning_eng"), className="project-value"),
                    ],
                    className="project-col",
                ),
                html.Div(
                    [
                        html.P("SECTION INCHARGE", className="project-label"),
                        html.H6(fmt_txt("section_inch"), className="project-value"),
                        html.P("SUPERVISORS", className="project-label"),
                        html.H6(fmt_txt("supervisor"), className="project-value"),
                    ],
                    className="project-col",
                ),
            ],
            className="project-grid",
        )

        title = f"Project Overview"
        return title, body

    # --- NEW: responsibilities chart callback ---
    # --- NEW: responsibilities chart callback ---

    # Responsibilities: grouped bars + three KPIs
    def _render_monthly_plan_card(
        *,
        plan_mode: str,
        project_value: str | None,
        entity_value: str | None,
        metric_value: str | None,
        months_value: Sequence[str] | None,
        quick_range_value: str | None,
    ):
        return _build_monthly_plan_for_project(
            plan_mode=plan_mode,
            project_value=project_value,
            entity_value=entity_value,
            metric_value=metric_value,
            months_value=months_value,
            quick_range_value=quick_range_value,
        )

    
    def _get_project_baselines(
        mode: str = "erection",
    ) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
        if project_baseline_provider is None:
            return {}, {}
        try:
            overall_map, monthly_map = project_baseline_provider(mode)  # type: ignore[misc]
        except TypeError:
            overall_map, monthly_map = project_baseline_provider()  # type: ignore[call-arg]
        except Exception as exc:
            LOGGER.warning("Failed to retrieve project baselines: %s", exc)
            return {}, {}
        return overall_map or {}, monthly_map or {}

    def _format_summary_value(value: float | None, unit_label: str | None = None, *, precision: int = 1) -> str:
        if value is None or pd.isna(value):
            return "-"
        try:
            numeric = float(value)
        except Exception:
            return "-"
        formatted = f"{numeric:,.{precision}f}"
        return f"{formatted} {unit_label}" if unit_label else formatted

    def _avg_metric_value(df: pd.DataFrame, metric_col: str, is_stringing: bool) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty or metric_col not in df.columns:
            return 0.0
        if is_stringing:
            required_cols = {"gang_name", "month"}
            if not required_cols.issubset(df.columns):
                return 0.0
            monthly_totals = (
                df.groupby(["gang_name", "month"], dropna=True)[metric_col]
                .sum()
                .reset_index(name="monthly_value")
            )
            if monthly_totals.empty:
                return 0.0
            return float(monthly_totals["monthly_value"].mean())
        return float(pd.to_numeric(df[metric_col], errors="coerce").dropna().mean())

    def _empty_summary_payload(is_stringing: bool) -> dict[str, str]:
        payload = {
            "projects": "-",
            "totals": "-",
            "gangs": "-",
            "productivity": "-",
            "lost_units": "-",
        }
        if is_stringing:
            payload["po_completion"] = "-"
            payload["tse"] = "-"
        payload["_meta"] = {
            "loss_rows": [],
            "unit_label": "KM" if is_stringing else "MT",
            "mode": "stringing" if is_stringing else "erection",
            "is_stringing": is_stringing,
        }
        return payload

    def _compute_mode_summary_components(
        *,
        scoped_full: pd.DataFrame,
        scoped_all: pd.DataFrame,
        months_ts: list[pd.Timestamp],
        days_factor: float,
        metric_col: str,
        is_stringing: bool,
        meta_signature: str,
        cache_key: str | None,
        scope_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "has_rows": not (scoped_full.empty and scoped_all.empty),
            "loss_scope": pd.DataFrame(),
            "history_scope": pd.DataFrame(),
            "projects_count": 0,
            "gangs_count": 0,
            "total_delivered": 0.0,
            "total_lost": 0.0,
            "total_potential": 0.0,
            "balance_value": 0.0,
            "loss_rows": [],
        }
        if not result["has_rows"]:
            return result

        has_selected_months = bool(months_ts)
        earliest_month = None
        if has_selected_months and not scoped_all.empty and "month" in scoped_all.columns:
            month_values = sorted({ts for ts in months_ts if pd.notna(ts)})
            period_mask = scoped_all["month"].isin(month_values)
            loss_scope = scoped_all.loc[period_mask].copy()
            earliest_month = month_values[0] if month_values else None
            history_scope = scoped_all.loc[scoped_all["month"] < (earliest_month or pd.Timestamp.max)].copy()
        else:
            loss_scope = scoped_all.copy()
            history_scope = scoped_all.copy()

        allowed_months = set(month_values) if has_selected_months and month_values else None
        idle_table = _idle_table_for_mode("stringing" if is_stringing else "erection")

        if not loss_scope.empty:
            if "gang_name" in loss_scope.columns:
                loss_scope = loss_scope.dropna(subset=["gang_name"])
                loss_scope["gang_name"] = loss_scope["gang_name"].astype(str).str.strip()
            if "project_name" in loss_scope.columns:
                loss_scope["project_name"] = loss_scope["project_name"].astype(str).str.strip()

        result["loss_scope"] = loss_scope
        result["history_scope"] = history_scope

        def _maybe_cached(token: str, producer: Callable[[], T], *, clone: Callable[[T], T] | None = None) -> T:
            if cache_key:
                return _cached_scope_result(cache_key, token, producer, clone=clone)
            value = producer()
            return clone(value) if clone else value

        precomputed_overall, precomputed_monthly = _get_project_baselines(
            "stringing" if is_stringing else "erection"
        )
        use_precomputed = bool(precomputed_overall)
        proj_overall_all: dict[str, float] = {}
        proj_monthly: dict[str, dict[pd.Timestamp, float]] = {}

        if use_precomputed:
            if "project_name" in scoped_all.columns:
                available_projects = (
                    scoped_all["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                available_projects = []
            if available_projects:
                proj_overall_all = {
                    project: precomputed_overall.get(project)
                    for project in available_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in available_projects
                }
            else:
                proj_overall_all = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if has_selected_months and earliest_month is not None:
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < earliest_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < earliest_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token_all = f"project-baseline::{metric_col}::{meta_signature}"

            def _compute_baseline_all() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if scoped_all.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(scoped_all, metric_col)
                return compute_project_baseline_maps(scoped_all)

            proj_overall_all, proj_monthly_all = _maybe_cached(
                baseline_token_all,
                _compute_baseline_all,
                clone=_clone_baseline_result,
            )

            if has_selected_months and earliest_month is not None:
                proj_overall_all = {
                    project: value
                    for project, value in proj_overall_all.items()
                    if value is not None
                }
                filtered: dict[str, dict[pd.Timestamp, float]] = {}
                for project, month_map in proj_monthly_all.items():
                    subset = {month: value for month, value in month_map.items() if month < earliest_month}
                    if subset:
                        filtered[project] = subset
                proj_monthly = filtered
            else:
                proj_monthly = proj_monthly_all

        gang_to_project = (
            scoped_all[["gang_name", "project_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("gang_name")["project_name"]
            .astype(str)
            .to_dict()
        )

        baseline_overall_map = {g: proj_overall_all.get(p) for g, p in gang_to_project.items()}
        baseline_monthly_map = {g: proj_monthly.get(p, {}) for g, p in gang_to_project.items()}

        loss_token = f"loss::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}"

        def _compute_loss_rows() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            if loss_scope.empty:
                return rows
            for gang_name, gang_df in loss_scope.groupby("gang_name"):
                if gang_df.empty:
                    continue
                overall_baseline = baseline_overall_map.get(gang_name)
                if is_stringing:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss_for_column(
                        gang_df,
                        metric_column=metric_col,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                        idle_intervals=idle_table,
                        allowed_months=allowed_months,
                    )
                else:
                    idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                        gang_df,
                        loss_max_gap_days=config.loss_max_gap_days,
                        baseline_mt_per_day=overall_baseline,
                        baseline_by_month=baseline_monthly_map.get(gang_name),
                        idle_intervals=idle_table,
                        allowed_months=allowed_months,
                    )
                rows.append(
                    {
                        "gang_name": gang_name,
                        "delivered": delivered,
                        "lost": loss_mt,
                        "potential": potential,
                        "avg_prod": (gang_df[metric_col].mean() if metric_col in gang_df.columns else 0.0),
                        "baseline": baseline,
                    }
                )
            return rows

        loss_rows = _maybe_cached(
            loss_token,
            _compute_loss_rows,
            clone=_clone_loss_rows,
        )
        result["loss_rows"] = loss_rows

        loss_df = pd.DataFrame(loss_rows)
        if is_stringing and not loss_df.empty:
            loss_df["avg_prod"] = loss_df["avg_prod"].astype(float) * days_factor
            loss_df["baseline"] = loss_df["baseline"].astype(float) * days_factor

        metric_scope = loss_scope if not loss_scope.empty else scoped_full
        total_metric = (
            float(metric_scope[metric_col].sum())
            if not metric_scope.empty and metric_col in metric_scope.columns
            else 0.0
        )
        total_delivered = float(loss_df["delivered"].sum()) if not loss_df.empty else 0.0
        total_lost = float(loss_df["lost"].sum()) if not loss_df.empty else 0.0
        total_potential = total_delivered + total_lost if (total_delivered or total_lost) else total_metric
        balance_value = max(total_potential - total_delivered, 0.0)

        completion_override: float | None = None
        if is_stringing:
            completion_override = _sum_completion_totals(
                scoped_full,
                value_column="length_km",
                completion_column="fs_complete_date",
                fallback_columns=[
                    ("span (m)", 0.001),
                    ("span_m", 0.001),
                    ("length", 0.001),
                    ("length_m", 0.001),
                    ("tower_weight", 0.001),
                ],
            )
        else:
            completion_override = _sum_completion_totals(
                scoped_full,
                value_column="tower_weight",
                completion_column="completion_date",
            )
        if completion_override is not None:
            total_delivered = completion_override
            total_metric = completion_override if not total_lost else completion_override + total_lost
            total_potential = total_delivered + total_lost if total_lost else total_delivered
            balance_value = max(total_potential - total_delivered, 0.0)

        def _nunique(frame: pd.DataFrame, column: str) -> int:
            if column not in frame.columns or frame.empty:
                return 0
            return (
                frame[column]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )

        projects_count = _nunique(loss_scope, "project_name") or _nunique(scoped_full, "project_name")
        if is_stringing:
            try:
                plan_df, _ = _stringing_plan_totals_by_project(
                    months_ts,
                    current_month=months_ts[-1] if months_ts else None,
                )
            except Exception:
                plan_df = pd.DataFrame()
            if isinstance(plan_df, pd.DataFrame) and not plan_df.empty:
                plan_keys = set(plan_df.index.astype(str))
                selected = (scope_meta or {}).get("selected") or {}
                project_filters = _normalize_str_list(selected.get("projects"))
                if project_filters:
                    filter_keys = {
                        _compact_project_key(value) or _normalize_lower(value)
                        for value in project_filters
                        if str(value).strip()
                    }
                    filter_keys = {key for key in filter_keys if key}
                    if filter_keys:
                        plan_keys = {key for key in plan_keys if key in filter_keys}
                delivered_keys: set[str] = set()
                for frame in (loss_scope, scoped_full):
                    if not isinstance(frame, pd.DataFrame) or frame.empty or "project_name" not in frame.columns:
                        continue
                    delivered_keys.update(
                        frame["project_name"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .map(lambda value: _compact_project_key(value) or _normalize_lower(value))
                    )
                delivered_keys = {key for key in delivered_keys if key}
                po_keys: set[str] = set()
                try:
                    po_frame = _get_stringing_po_daily_frame()
                except Exception:
                    po_frame = pd.DataFrame()
                if isinstance(po_frame, pd.DataFrame) and not po_frame.empty:
                    po_scoped = _stringing_scope(po_frame, selected.get("methods") or [])
                    po_filtered = apply_filters(po_scoped, project_filters, months_ts, [])
                    po_filtered = _filter_frame_for_deployment(po_filtered, selected.get("stringing_scope"))
                    if not po_filtered.empty:
                        po_col = "project_name" if "project_name" in po_filtered.columns else (
                            "project" if "project" in po_filtered.columns else None
                        )
                        if po_col:
                            po_values = (
                                po_filtered[po_col]
                                .dropna()
                                .astype(str)
                                .str.strip()
                            )
                            po_keys = {
                                _compact_project_key(value) or _normalize_lower(value)
                                for value in po_values
                                if str(value).strip()
                            }
                            po_keys = {key for key in po_keys if key}
                if po_keys:
                    delivered_keys.update(po_keys)
                total_project_keys = plan_keys | delivered_keys
                total_project_keys = {key for key in total_project_keys if key}
                if total_project_keys:
                    projects_count = len(total_project_keys)
        gangs_count = _nunique(loss_scope, "gang_name") or _nunique(scoped_full, "gang_name")

        result.update(
            {
                "projects_count": projects_count,
                "gangs_count": gangs_count,
                "total_delivered": total_delivered,
                "total_lost": total_lost,
                "total_potential": total_potential,
                "balance_value": balance_value,
            }
        )
        return result

    def _get_stringing_po_daily_frame() -> pd.DataFrame:
        if not config.enable_stringing:
            return pd.DataFrame()

        def _producer() -> pd.DataFrame:
            df_compiled = pd.DataFrame()
            if callable(stringing_compiled_provider):
                try:
                    df_compiled = stringing_compiled_provider()
                except Exception:
                    df_compiled = pd.DataFrame()
            if not isinstance(df_compiled, pd.DataFrame) or df_compiled.empty:
                return pd.DataFrame()
            payout_source = df_compiled.copy()
            po_col = None
            for cand in ("paying_out_complete", "po_completion_date", "po_completion"):
                if cand in payout_source.columns:
                    po_col = cand
                    break
            if po_col is None:
                return pd.DataFrame()
            payout_source["po_completion_date"] = pd.to_datetime(payout_source[po_col], errors="coerce")
            payout_source = payout_source.dropna(subset=["po_completion_date"])
            if payout_source.empty:
                return pd.DataFrame()
            payout_source["po_start_date"] = payout_source["po_completion_date"]
            try:
                payout = expand_stringing_to_daily_payout(payout_source)
            except Exception:
                LOGGER.exception("Failed to expand P/O completion daily rows")
                return pd.DataFrame()
            if payout.empty:
                return payout
            if "project_name" not in payout.columns and "project" in payout.columns:
                payout["project_name"] = payout["project"]
            if "project" not in payout.columns and "project_name" in payout.columns:
                payout["project"] = payout["project_name"]
            if "month" not in payout.columns and "date" in payout.columns:
                payout["date"] = pd.to_datetime(payout["date"], errors="coerce")
                payout = payout.dropna(subset=["date"])
                payout["month"] = payout["date"].dt.to_period("M").to_timestamp()
            return payout

        return _cached_global_result(
            "stringing:po_completion_daily",
            _producer,
            clone=_clone_dataframe,
        )

    def _compute_po_completion_totals(
        scope_meta: dict[str, Any] | None,
        months_ts: list[pd.Timestamp],
        days_factor: float,
    ) -> str:
        plan_total = _stringing_planned_total_for_dates(
            scope_meta,
            months_ts,
            date_columns=_STRINGING_PO_DATE_COLUMNS,
        )
        has_plan_scope = _stringing_scope_has_plan(
            scope_meta,
            months_ts,
            date_columns=_STRINGING_PO_DATE_COLUMNS,
        )
        frame = _get_stringing_po_daily_frame()
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            if plan_total <= 0 and not has_plan_scope:
                return "-"
            total_txt = f"{plan_total:.1f} KM" if plan_total > 0 else "0.0 KM"
            done_txt = _format_summary_value(0.0, "KM")
            balance_txt = _format_summary_value(plan_total, "KM") if plan_total > 0 else "\u2014"
            return f"{total_txt} / {done_txt} / {balance_txt}"

        selected = (scope_meta or {}).get("selected") or {}
        projects = selected.get("projects") or []
        gangs = selected.get("gangs") or []
        method_values = selected.get("methods") or []

        scoped_base = _stringing_scope(frame, method_values)
        scoped_full = apply_filters(scoped_base, projects, months_ts, gangs)

        done_total = _sum_completion_totals(
            scoped_full,
            value_column="po_km",
            completion_column="po_completion_date",
            fallback_columns=[("po", 0.001)],
        ) or 0.0
        if done_total == 0.0 and isinstance(scoped_full, pd.DataFrame) and not scoped_full.empty and "daily_km" in scoped_full.columns:
            done_total = float(pd.to_numeric(scoped_full["daily_km"], errors="coerce").dropna().sum())

        done_txt = _format_summary_value(done_total, "KM")
        if plan_total > 0:
            total_txt = f"{plan_total:.1f} KM"
            balance_txt = _format_summary_value(max(plan_total - done_total, 0.0), "KM")
        elif has_plan_scope:
            total_txt = "0.0 KM"
            balance_txt = "\u2014"
        else:
            total_txt = "No Plan"
            balance_txt = "\u2014"
        return f"{total_txt} / {done_txt} / {balance_txt}"

    def _summarize_scope_for_cards(scope_meta: dict[str, Any] | None) -> dict[str, str]:
        mode = _normalize_mode((scope_meta or {}).get("mode"))
        summary = _empty_summary_payload(mode == "stringing")
        meta_info = summary.setdefault("_meta", {})
        meta_info.setdefault("loss_rows", [])
        meta_info.setdefault("mode", mode)
        meta_info.setdefault("is_stringing", mode == "stringing")
        meta_info.setdefault("unit_label", "KM" if mode == "stringing" else "MT")
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            return summary
        try:
            months_ts = _months_from_meta(scope_meta)
            days_factor = float(scope_meta.get("days_factor") or 30.0)
            is_stringing = mode == "stringing"
            metric_col = "daily_km" if is_stringing else "daily_prod_mt"
            unit_short = "KM" if is_stringing else "MT"
            prod_unit = "KM/month" if is_stringing else "MT/day"
            meta_signature = scope_meta.get("signature") or "nosig"
            selected = scope_meta.get("selected") or {}
            meta_info["unit_label"] = unit_short
            meta_info["is_stringing"] = is_stringing

            scoped_full = _scope_frame_from_store(scope_meta, "full").copy()
            scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()
            if scoped_full.empty and scoped_all.empty:
                return summary

            scope_keys = scope_meta.get("scopes") or {}
            project_gang_key = scope_keys.get("project_gang")
            components = _compute_mode_summary_components(
                scoped_full=scoped_full,
                scoped_all=scoped_all,
                months_ts=months_ts,
                days_factor=days_factor,
                metric_col=metric_col,
                is_stringing=is_stringing,
                meta_signature=meta_signature,
                cache_key=project_gang_key,
                scope_meta=scope_meta,
            )

            loss_scope = components["loss_scope"]
            history_scope = components["history_scope"]
            projects_count = components["projects_count"]
            gangs_count = components["gangs_count"]
            total_delivered = components["total_delivered"]
            total_lost = components["total_lost"]
            total_potential = components["total_potential"]
            balance_value = components["balance_value"]
            meta_info["loss_rows"] = components.get("loss_rows") or []
            balance_reference_delivered = total_delivered
            if is_stringing:
                deployment_scope = _normalize_deployment_filter(selected.get("stringing_scope"))
                if deployment_scope != "all":
                    try:
                        frames_all, _, _ = _build_scope_frames(
                            "stringing",
                            project_list=selected.get("projects", []),
                            gang_list=selected.get("gangs", []),
                            months_value=selected.get("months", []),
                            quick_range=selected.get("quick_range"),
                            method_values=selected.get("methods", []),
                            deployment_filter="all",
                        )
                        scoped_full_all = frames_all.get("full", pd.DataFrame()).copy()
                        scoped_all_all = frames_all.get("project_gang", pd.DataFrame()).copy()
                    except Exception:
                        scoped_full_all = pd.DataFrame()
                        scoped_all_all = pd.DataFrame()
                    if not (scoped_full_all.empty and scoped_all_all.empty):
                        scope_meta_all = dict(scope_meta)
                        scope_meta_all["selected"] = dict(selected)
                        scope_meta_all["selected"]["stringing_scope"] = "all"
                        all_components = _compute_mode_summary_components(
                            scoped_full=scoped_full_all,
                            scoped_all=scoped_all_all,
                            months_ts=months_ts,
                            days_factor=days_factor,
                            metric_col=metric_col,
                            is_stringing=True,
                            meta_signature=f"{meta_signature}::all-deployments",
                            cache_key=None,
                            scope_meta=scope_meta_all,
                        )
                        balance_reference_delivered = all_components["total_delivered"]

            def _collect_project_labels(frame: pd.DataFrame | None) -> list[str]:
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    return []
                labels: list[str] = []
                seen: set[str] = set()
                for column in (
                    "project_name",
                    "project",
                    "project_name_display",
                    "Project Name",
                    "project_code",
                ):
                    if column not in frame.columns:
                        continue
                    values = (
                        frame[column]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .replace("", pd.NA)
                        .dropna()
                    )
                    for value in values:
                        if value not in seen:
                            seen.add(value)
                            labels.append(value)
                return labels

            def _project_label_keys(text: object) -> tuple[list[str], list[str]]:
                base = str(text or "").strip()
                if not base:
                    return [], []
                parts = [base]
                if " : " in base:
                    left, right = base.split(" : ", 1)
                    parts.extend([left.strip(), right.strip()])
                norm_keys: list[str] = []
                compact_keys: list[str] = []
                for part in parts:
                    norm = _normalize_lower(part)
                    if norm and norm not in norm_keys:
                        norm_keys.append(norm)
                    compact = _compact_project_key(part)
                    if compact and compact not in compact_keys:
                        compact_keys.append(compact)
                match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{3,4})\b", base.upper())
                if match:
                    compact = _compact_project_key(f"{match.group(1)}{match.group(2)}")
                    if compact and compact not in compact_keys:
                        compact_keys.append(compact)
                return norm_keys, compact_keys

            tse_count: int | None = None
            if is_stringing:
                tse_norm_map, tse_alias_map = _get_stringing_tse_lookup()
                label_source = loss_scope if not loss_scope.empty else scoped_full
                labels = _collect_project_labels(label_source)
                matched_total = 0
                matched_any = False
                used_projects: set[str] = set()
                if labels and (tse_norm_map or tse_alias_map):
                    for label in labels:
                        norm_keys, compact_keys = _project_label_keys(label)
                        value, canonical_id = _resolve_tse_value(norm_keys, compact_keys, tse_norm_map, tse_alias_map)
                        if canonical_id and canonical_id not in used_projects and value is not None:
                            used_projects.add(canonical_id)
                            matched_total += int(value)
                            matched_any = True
                if matched_any:
                    tse_count = matched_total
                else:
                    fallback_scope = loss_scope if not loss_scope.empty else scoped_full
                    tse_count = 0
                    if {"method", "gang_name"}.issubset(fallback_scope.columns) and not fallback_scope.empty:
                        tse_mask = fallback_scope["method"].astype(str).str.strip().str.lower() == "tse"
                        tse_count = int(
                            fallback_scope.loc[tse_mask, "gang_name"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .nunique()
                        )

            prod_current = _avg_metric_value(scoped_full, metric_col, is_stringing)
            prod_history = _avg_metric_value(history_scope, metric_col, is_stringing)
            if prod_history == 0.0:
                prod_history = prod_current

            summary["projects"] = f"{projects_count:,}" if projects_count else "-"
            summary["gangs"] = f"{gangs_count:,}" if gangs_count else "-"
            if is_stringing:
                planned_total_value = _stringing_planned_total_for_dates(
                    scope_meta,
                    months_ts,
                    date_columns=_STRINGING_FS_DATE_COLUMNS,
                )
                has_plan_scope = _stringing_scope_has_plan(
                    scope_meta,
                    months_ts,
                    date_columns=_STRINGING_FS_DATE_COLUMNS,
                )
                done_txt = _format_summary_value(total_delivered, unit_short)
                if planned_total_value > 0:
                    total_txt = f"{planned_total_value:.1f} {unit_short}"
                    balance_txt = _format_summary_value(
                        max(planned_total_value - balance_reference_delivered, 0.0),
                        unit_short,
                    )
                elif has_plan_scope:
                    total_txt = "0.0 KM"
                    balance_txt = "\u2014"
                else:
                    total_txt = "No Plan"
                    balance_txt = "\u2014"
            else:
                tower_done = _count_completed_towers(loss_scope, months_ts)
                planned_towers, _ = _compute_planned_tower_layers(scoped_all, months_ts)
                total_towers = planned_towers or tower_done
                balance_towers = max(total_towers - tower_done, 0)
                total_txt = f"{total_towers:,}"
                done_txt = f"{tower_done:,}"
                balance_txt = f"{balance_towers:,}"
            summary["totals"] = f"{total_txt} / {done_txt} / {balance_txt}"
            prod_txt = _format_summary_value(prod_current, prod_unit, precision=2)
            hist_txt = _format_summary_value(prod_history, prod_unit, precision=2)
            summary["productivity"] = f"{prod_txt} / {hist_txt}"
            summary["lost_units"] = _format_summary_value(total_lost, unit_short)
            if is_stringing:
                summary["po_completion"] = _compute_po_completion_totals(scope_meta, months_ts, days_factor)
                summary["tse"] = "-" if tse_count is None else f"{tse_count:,}"
            return summary
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to compute %s summary card: %s", mode, exc)
            return summary

    @app.callback(
        Output("erection-card-projects", "children"),
        Output("erection-card-totals", "children"),
        Output("erection-card-gangs", "children"),
        Output("erection-card-productivity", "children"),
        Output("erection-card-loss", "children"),
        Output("stringing-card-projects", "children"),
        Output("stringing-card-totals", "children"),
        Output("stringing-card-gangs", "children"),
        Output("stringing-card-productivity", "children"),
        Output("stringing-card-loss", "children"),
        Output("stringing-card-tse", "children"),
        Input("store-filtered-scope", "data"),
    )
    def _update_mode_summary_cards(scope_meta: dict[str, Any] | None) -> tuple[str, ...]:
        selected = (scope_meta or {}).get("selected") or {}

        def _extract_list(key: str, *, lower: bool = False) -> list[str]:
            values = selected.get(key) or []
            return _normalize_str_list(values, lower=lower)

        projects = _extract_list("projects")
        months = _extract_list("months")
        gangs = _extract_list("gangs")
        method_values = _extract_list("methods", lower=True)
        quick_range = selected.get("quick_range")
        stringing_scope = selected.get("stringing_scope")

        current_mode = _normalize_mode((scope_meta or {}).get("mode"))

        def _meta_for(mode_name: str) -> dict[str, Any] | None:
            if current_mode == mode_name and isinstance(scope_meta, dict):
                return scope_meta
            try:
                return _build_scope_meta_payload(
                    eff_mode=mode_name,
                    project_list=projects,
                    gang_list=gangs,
                    months_list=months,
                    quick_range=quick_range,
                    method_values=method_values,
                    method_list=method_values,
                    deployment_filter=stringing_scope if mode_name == "stringing" else "all",
                )
            except Exception:
                LOGGER.exception("Unable to build %s scope for summary cards", mode_name)
                return None

        erection_summary = _summarize_scope_for_cards(_meta_for("erection"))
        if config.enable_stringing:
            stringing_summary = _summarize_scope_for_cards(_meta_for("stringing"))
        else:
            stringing_summary = _empty_summary_payload(True)

        return (
            erection_summary["projects"],
            erection_summary["totals"],
            erection_summary["gangs"],
            erection_summary["productivity"],
            erection_summary["lost_units"],
            stringing_summary["projects"],
            stringing_summary["totals"],
            stringing_summary["gangs"],
            stringing_summary["productivity"],
            stringing_summary["lost_units"],
            stringing_summary.get("tse", "-"),
        )


        CHART_SOURCES = {"g-actual-vs-bench", "g-top5", "g-bottom5"}
    GLOBAL_MODAL_CHART_SOURCES = {
        "global-performance-actual-vs-bench",
        "global-performance-top5",
        "global-performance-bottom5",
    }

    def _prepare_trace_dataframes(
        scope_meta: dict[str, Any],
        gang_focus: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (idle_df, daily_df) for the requested gang, formatted for display/export."""

        if not isinstance(scope_meta, dict):
            return pd.DataFrame(), pd.DataFrame()

        selected = scope_meta.get("selected") or {}
        months_ts = _months_from_meta(scope_meta)
        meta_signature = scope_meta.get("signature") or "nosig"
        scope_keys = scope_meta.get("scopes") or {}
        project_gang_key = scope_keys.get("project_gang")

        base_scope = _scope_frame_from_store(scope_meta, "project").copy()
        scoped = _scope_frame_from_store(scope_meta, "full").copy()
        scoped_all = _scope_frame_from_store(scope_meta, "project_gang").copy()

        is_stringing = _normalize_mode(scope_meta.get("mode")) == "stringing"
        metric_col = "daily_km" if is_stringing else "daily_prod_mt"

        def pick_gang_scope(target_gang: str | None) -> pd.DataFrame:
            if not target_gang:
                return pd.DataFrame()
            subset = base_scope[base_scope["gang_name"] == target_gang]
            if not subset.empty:
                return subset
            fb = scoped_all[scoped_all["gang_name"] == target_gang].copy()
            if months_ts and "month" in fb.columns:
                fb = fb[fb["month"].isin(months_ts)]
            return fb

        baseline_source = scoped_all.copy()
        precomputed_overall, precomputed_monthly = _get_project_baselines(
            "stringing" if is_stringing else "erection"
        )
        use_precomputed = bool(precomputed_overall)
        if use_precomputed:
            if "project_name" in baseline_source.columns:
                candidate_projects = (
                    baseline_source["project_name"].dropna().astype(str).str.strip().unique().tolist()
                )
            else:
                candidate_projects = []
            if candidate_projects:
                proj_overall = {
                    project: precomputed_overall.get(project)
                    for project in candidate_projects
                    if precomputed_overall.get(project) is not None
                }
                monthly_candidates = {
                    project: precomputed_monthly.get(project, {})
                    for project in candidate_projects
                }
            else:
                proj_overall = dict(precomputed_overall)
                monthly_candidates = dict(precomputed_monthly)
            if months_ts:
                cutoff_month = min(months_ts)
                proj_monthly = {
                    project: {
                        month: value
                        for month, value in month_map.items()
                        if month < cutoff_month
                    }
                    for project, month_map in monthly_candidates.items()
                    if any(month < cutoff_month for month in month_map)
                }
            else:
                proj_monthly = monthly_candidates
        else:
            baseline_token = f"trace-project-baseline::{metric_col}::{is_stringing}::{meta_signature}"

            def _compute_trace_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                if baseline_source.empty:
                    return {}, {}
                if is_stringing:
                    return compute_project_baseline_maps_for(baseline_source, metric_col)
                return compute_project_baseline_maps(baseline_source)

            proj_overall, proj_monthly = _cached_scope_result(
                project_gang_key,
                baseline_token,
                _compute_trace_baselines,
                clone=_clone_baseline_result,
            )

        project_overall = proj_overall
        project_monthly = proj_monthly

        if {"gang_name", "project_name"}.issubset(baseline_source.columns):
            g2p = (
                baseline_source[["gang_name", "project_name"]]
                .dropna()
                .drop_duplicates()
                .set_index("gang_name")["project_name"]
                .astype(str)
                .to_dict()
            )
        else:
            g2p = {}

        overall_baseline_map = {g: project_overall.get(p) for g, p in g2p.items()}
        monthly_baseline_map = {g: project_monthly.get(p, {}) for g, p in g2p.items()}

        idle_source = pick_gang_scope(gang_focus)
        if idle_source.empty:
            idle_source = scoped if not scoped.empty else base_scope

        idle_token = f"idle::{metric_col}::{config.loss_max_gap_days}::{is_stringing}::{meta_signature}::{gang_focus or '*'}"

        def _compute_idle_df() -> pd.DataFrame:
            if idle_source.empty:
                return pd.DataFrame()
            return compute_idle_intervals_per_gang(
                idle_source,
                loss_max_gap_days=config.loss_max_gap_days,
                baseline_month_lookup=monthly_baseline_map,
                baseline_fallback_map=overall_baseline_map,
            )

        idle_df = _cached_scope_result(
            project_gang_key,
            idle_token,
            _compute_idle_df,
            clone=_clone_dataframe,
        )
        if not idle_df.empty:
            idle_df["interval_loss_mt"] = (
                idle_df["baseline"].astype(float)
                * idle_df["idle_days_capped"].astype(float)
            )
            idle_df["cumulative_loss"] = idle_df.groupby("gang_name")[
                "interval_loss_mt"
            ].cumsum()

            def _fmt_metric(value):
                if pd.isna(value):
                    return ""
                formatted = f"{value:.2f}"
                return formatted.rstrip("0").rstrip(".")

            idle_df = (
                idle_df.assign(
                    interval_start=idle_df["interval_start"].dt.strftime("%d-%m-%Y"),
                    interval_end=idle_df["interval_end"].dt.strftime("%d-%m-%Y"),
                    baseline=idle_df["baseline"].apply(_fmt_metric),
                    cumulative_loss=idle_df["cumulative_loss"].apply(_fmt_metric),
                )
                .drop(columns=["interval_loss_mt"])
            )

        daily_source = pick_gang_scope(gang_focus)
        if daily_source.empty:
            daily_source = scoped if not scoped.empty else base_scope
        sort_cols = [ "gang_name", "date"]
        daily_source = daily_source.sort_values(sort_cols)
        columns = ["date", "gang_name", metric_col]
        if "project_name" in daily_source.columns:
            columns.insert(2, "project_name")
        daily_source = daily_source[columns]
        if not daily_source.empty:
            daily_source = daily_source.assign(
                date=daily_source["date"].dt.strftime("%d-%m-%Y"),
                daily_prod_mt=(
                    pd.to_numeric(daily_source[metric_col], errors="coerce").round(2).map(
                        lambda v: "" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip(".")
                    )
                ),
            )
            if metric_col != "daily_prod_mt":
                daily_source = daily_source.drop(columns=[metric_col])

        return idle_df, daily_source

    def _compute_trace_table_payload(
        scope_meta: dict[str, Any], gang_focus: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        idle_df, daily_df = _prepare_trace_dataframes(scope_meta, gang_focus)
        return idle_df.to_dict("records"), daily_df.to_dict("records")

    @app.callback(
        Output("tbl-idle-intervals", "data"),
        Output("tbl-daily-prod", "data"),
        Output("modal-tbl-idle-intervals", "data"),
        Output("modal-tbl-daily-prod", "data"),
        Input("store-click-meta", "data"),
        Input("trace-gang", "value"),
        Input("modal-trace-gang", "value"),
        Input("store-filtered-scope", "data"),
        prevent_initial_call=True,
    )
    def update_trace_tables(
        meta,
        trace_gang_value,
        modal_trace_gang_value,
        scope_meta: dict[str, Any] | None,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        dropdown_selection = trace_gang_value or modal_trace_gang_value
        meta_source = meta.get("source") if isinstance(meta, dict) else None
        meta_gang = meta.get("gang") if isinstance(meta, dict) else None
        meta_is_chart = meta_source in CHART_SOURCES and bool(meta_gang)

        if triggered_id == "store-click-meta":
            gang_focus = meta_gang if meta_is_chart else dropdown_selection
        else:
            gang_focus = dropdown_selection or (meta_gang if meta_is_chart else None)

        if not gang_focus or not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        idle_df, daily_df = _prepare_trace_dataframes(scope_meta, gang_focus)
        idle_records = idle_df.to_dict("records")
        daily_records = daily_df.to_dict("records")
        return idle_records, daily_records, idle_records, daily_records

    @app.callback(
        Output("store-project-modal-scope", "data"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("project-modal-stringing-scope", "value"),
        Input("store-project-modal-performance-mode", "data"),
        prevent_initial_call=False,
    )
    def _sync_project_modal_scope_store(
        focus_data: dict[str, Any] | None,
        months,
        quick_range,
        gangs,
        stringing_scope,
        performance_mode,
    ):
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return None
        project_code = (focus_data or {}).get("code")
        eff_mode = _modal_mode_from_store(performance_mode, "erection")
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        method_values = _method_filters_for_scope(stringing_scope) if eff_mode == "stringing" else []
        return _build_project_scope_meta(
            project_name,
            project_code,
            eff_mode,
            months,
            quick_range,
            gangs,
            method_values,
            stringing_scope,
        )

    @app.callback(
        Output("project-modal-trace-gang", "options"),
        Output("project-modal-trace-gang", "value"),
        Input("store-project-modal-scope", "data"),
        Input("store-project-tile-focus", "data"),
        Input("project-modal-selected-gang", "data"),
        prevent_initial_call=True,
    )
    def _project_modal_trace_dropdown(
        scope_meta: dict[str, Any] | None,
        focus_data: dict[str, Any] | None,
        selected_gang: str | None,
    ):
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return [], None
        if not isinstance(scope_meta, dict):
            return [], None
        base_scope = _scope_frame_from_store(scope_meta, "project")
        if base_scope.empty or "gang_name" not in base_scope.columns:
            return [], None
        gangs_series = (
            base_scope["gang_name"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
        )
        gangs_series = sorted({g for g in gangs_series if g})
        options = [{"label": g, "value": g} for g in gangs_series]
        if not options:
            return [], None
        if selected_gang and selected_gang in gangs_series:
            value = selected_gang
        else:
            value = gangs_series[0]
        return options, value

    @app.callback(
        Output("project-modal-tbl-idle-intervals", "data"),
        Output("project-modal-tbl-daily-prod", "data"),
        Input("store-project-modal-scope", "data"),
        Input("store-project-modal-click-meta", "data"),
        Input("project-modal-trace-gang", "value"),
        Input("project-modal-selected-gang", "data"),
        Input("store-project-tile-focus", "data"),
        prevent_initial_call=True,
    )
    def _project_modal_trace_tables(
        scope_meta,
        modal_meta,
        dropdown_value,
        selected_store_gang,
        focus_data: dict[str, Any] | None,
    ):
        if not isinstance(scope_meta, dict):
            raise PreventUpdate
        project_name = (focus_data or {}).get("project")
        if not project_name:
            raise PreventUpdate
        ctx = dash.callback_context
        triggered_id = None
        if ctx.triggered:
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        dropdown_selection = dropdown_value or selected_store_gang
        if not isinstance(modal_meta, dict):
            modal_meta = {}
        meta_source = modal_meta.get("source")
        meta_gang = modal_meta.get("gang")
        modal_sources = {
            "project-modal-actual-vs-bench",
            "project-modal-top5",
            "project-modal-bottom5",
        }
        meta_is_chart = meta_source in modal_sources and bool(meta_gang)
        if triggered_id == "store-project-modal-click-meta":
            gang_focus = meta_gang if meta_is_chart else dropdown_selection
        else:
            gang_focus = dropdown_selection or (meta_gang if meta_is_chart else None)
        if not gang_focus:
            raise PreventUpdate
        idle_data, daily_data = _compute_trace_table_payload(scope_meta, gang_focus)
        return idle_data, daily_data

    @app.callback(
        Output("project-modal-topbot-mode-label", "children"),
        Output("project-modal-tbl-idle-intervals", "columns"),
        Output("project-modal-tbl-daily-prod", "columns"),
        Input("store-project-modal-performance-mode", "data"),
    )
    def _sync_project_modal_trace_ui(performance_mode):
        eff_mode = _modal_mode_from_store(performance_mode, "erection")
        if eff_mode == "stringing" and not config.enable_stringing:
            eff_mode = "erection"
        is_stringing = eff_mode == "stringing"

        idle_columns = [
            {"name": "Gang", "id": "gang_name"},
            {"name": "Interval Start", "id": "interval_start"},
            {"name": "Interval End", "id": "interval_end"},
            {"name": "Raw Gap (days)", "id": "raw_gap_days"},
            {"name": "Idle Counted (days)", "id": "idle_days_capped"},
            {
                "name": "Baseline (KM/Month)" if is_stringing else "Baseline (MT/day)",
                "id": "baseline",
            },
            {
                "name": "Cumulative Loss (KM)" if is_stringing else "Cumulative Loss (MT)",
                "id": "cumulative_loss",
            },
        ]
        daily_columns = [
            {"name": "Gang", "id": "gang_name"},
            {"name": "Project", "id": "project_name"},
            {"name": "Date", "id": "date"},
            {
                "name": "KM/Month" if is_stringing else "MT/day",
                "id": "daily_prod_mt",
            },
        ]

        label = "Stringing" if is_stringing else "Erection"
        return label, idle_columns, daily_columns

    @app.callback(
        Output("project-modal-avp-list", "children"),
        Output("project-modal-actual-vs-bench", "figure"),
        Output("project-modal-top5", "figure"),
        Output("project-modal-bottom5", "figure"),
        Input("store-project-modal-scope", "data"),
        Input("store-project-tile-focus", "data"),
        Input("project-modal-topbot-metric", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
        State("project-modal-stringing-scope", "value"),
        State("store-project-modal-performance-mode", "data"),
    )
    def _update_project_modal_performance(
        scope_store: dict[str, Any] | None,
        focus_data: dict[str, Any] | None,
        topbot_metric: str | None,
        months,
        quick_range,
        gangs,
        stringing_scope,
        performance_mode,
    ):
        empty_fig = go.Figure()
        if not focus_data or not focus_data.get("project"):
            return (
                html.Div("Select a project tile to view gang performance.", className="text-muted"),
                empty_fig,
                empty_fig,
                empty_fig,
            )

        project_name = focus_data.get("project")
        project_code = focus_data.get("code")
        scope_meta = None
        if isinstance(scope_store, dict):
            selected = (scope_store.get("selected") or {}).get("projects") or []
            if project_name in selected:
                scope_meta = scope_store
        if scope_meta is None:
            eff_mode = _modal_mode_from_store(performance_mode, "erection")
            if eff_mode == "stringing" and not config.enable_stringing:
                eff_mode = "erection"
            method_values = _method_filters_for_scope(stringing_scope) if eff_mode == "stringing" else []
            scope_meta = _build_project_scope_meta(
                project_name,
                project_code,
                eff_mode,
                months,
                quick_range,
                gangs,
                method_values,
                stringing_scope,
            )

        try:
            (
                *_kpi,
                avp_children,
                fig_loss,
                fig_top5,
                fig_bottom5,
                _fig_project,
            ) = _compute_dashboard_outputs(
                scope_meta,
                topbot_metric,
                avp_namespace="project-modal-avp",
                summarizer=_summarize_scope_for_cards,
                summary_factory=_empty_summary_payload,
            )
        except PreventUpdate:
            return (
                html.Div("No data available for the selected project.", className="text-muted"),
                empty_fig,
                empty_fig,
                empty_fig,
            )

        avp_children = avp_children or html.Div("No gangs available for this selection.", className="text-muted")
        return avp_children, fig_loss, fig_top5, fig_bottom5


    @app.callback(
        Output("tbl-erections-completed", "columns"),
        Output("tbl-erections-completed", "data"),
        Input("erections-completion-range", "start_date"),
        Input("erections-completion-range", "end_date"),
        Input("erections-search", "value"),
        Input("store-filtered-scope", "data"),
    )
    def update_erections_completed(
        start_date,
        end_date,
        search_text,
        scope_meta: dict[str, Any] | None,
    ) -> list[dict[str, object]]:
        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        eff_mode = _normalize_mode(scope_meta.get("mode"))
        scoped = _scope_frame_from_store(scope_meta, "project_gang").copy()

        if eff_mode == "stringing":
            export_df, display_df = _prepare_stringing_completed(
                scoped,
                range_start=range_start,
                range_end=range_end,
                search_text=search_text,
            )
            columns = [
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
            ]
        else:
            export_df, display_df = _prepare_erections_completed(
                scoped,
                range_start=range_start,
                range_end=range_end,
                responsibilities_provider=responsibilities_provider,
                search_text=search_text,
            )
            columns = [
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
            ]

        if display_df.empty:
            return columns, []
        return columns, display_df.to_dict("records")

    @app.callback(
        Output("project-modal-erections-search", "value"),
        Input("project-modal-erections-search-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_modal_erections_search(n):
        if not n:
            raise PreventUpdate
        return ""

    @app.callback(
        Output("project-modal-stringing-search", "value"),
        Input("project-modal-stringing-search-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_modal_stringing_search(n):
        if not n:
            raise PreventUpdate
        return ""

    @app.callback(
        Output("project-modal-erections-table", "data"),
        Input("project-modal-erections-range", "start_date"),
        Input("project-modal-erections-range", "end_date"),
        Input("project-modal-erections-search", "value"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
    )
    def _update_modal_erections_table(
        start_date,
        end_date,
        search_text,
        focus_data: dict[str, Any] | None,
        months,
        quick_range,
        gangs,
    ):
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return []
        project_code = (focus_data or {}).get("code")

        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        candidate_ids = _project_filter_candidates(project_name, project_code)
        if not candidate_ids:
            if project_name:
                candidate_ids = [str(project_name)]
            elif project_code:
                candidate_ids = [str(project_code)]
        project_list = _normalize_str_list(candidate_ids)
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))

        frames, _, _ = _build_scope_frames(
            "erection",
            project_list=project_list,
            gang_list=gang_list,
            months_value=months_list,
            quick_range=quick_range,
            method_values=[],
        )
        scoped = frames.get("project_gang", pd.DataFrame()).copy()
        export_df, display_df = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=responsibilities_provider,
            search_text=search_text,
        )
        return display_df.to_dict("records") if not display_df.empty else []

    @app.callback(
        Output("project-modal-stringing-table", "data"),
        Input("project-modal-stringing-range", "start_date"),
        Input("project-modal-stringing-range", "end_date"),
        Input("project-modal-stringing-search", "value"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("project-modal-stringing-scope", "value"),
    )
    def _update_modal_stringing_table(
        start_date,
        end_date,
        search_text,
        focus_data: dict[str, Any] | None,
        months,
        quick_range,
        gangs,
        stringing_scope,
    ):
        if not config.enable_stringing:
            return []
        project_name = (focus_data or {}).get("project")
        if not project_name:
            return []
        project_code = (focus_data or {}).get("code")

        range_start = _parse_completion_date(start_date) or _default_completion_date()
        range_end = _parse_completion_date(end_date) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        candidate_ids = _project_filter_candidates(project_name, project_code)
        if not candidate_ids:
            if project_name:
                candidate_ids = [str(project_name)]
            elif project_code:
                candidate_ids = [str(project_code)]
        project_list = _normalize_str_list(candidate_ids)
        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        method_list = _method_filters_for_scope(stringing_scope)

        months_ts = resolve_months(months_list, quick_range)

        frames, _, _ = _build_scope_frames(
            "stringing",
            project_list=project_list,
            gang_list=gang_list,
            months_value=months_list,
            quick_range=quick_range,
            method_values=method_list,
            deployment_filter=stringing_scope,
        )
        scoped = frames.get("project_gang", pd.DataFrame()).copy()
        if "date" not in scoped.columns or scoped.empty:
            selector = DATA_SELECTOR
            df_fallback = selector.select("stringing") if selector is not None else pd.DataFrame()
            if isinstance(df_fallback, pd.DataFrame) and not df_fallback.empty:
                scoped = apply_filters(
                    _stringing_scope(df_fallback, method_list),
                    project_list,
                    months_ts,
                    gang_list,
                )
        export_df, display_df = _prepare_stringing_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            search_text=search_text,
        )
        return display_df.to_dict("records") if not display_df.empty else []

    @app.callback(
        Output("erections-completion-range", "start_date"),
        Output("erections-completion-range", "end_date"),
        Output("erections-search", "value"),
        Input("btn-reset-erections", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_erections_controls(reset_clicks: int | None) -> tuple[str, str, str]:
        if not reset_clicks:
            raise PreventUpdate

        default = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date().isoformat()
        return default, default, ""

    @app.callback(
        Output("download-trace-xlsx", "data"),
        Output("project-modal-download-trace", "data"),
        Input("btn-export-trace", "n_clicks"),
        Input("modal-btn-export-trace", "n_clicks"),
        Input("project-modal-btn-export-trace", "n_clicks"),
        State("f-project", "value"),
        State("f-month", "value"),
        State("f-quick-range", "value"),
        State("f-gang", "value"),
        State("trace-gang", "value"),
        State("store-selected-gang", "data"),
        State("erections-completion-range", "start_date"),
        State("erections-completion-range", "end_date"),
        State("erections-search", "value"),
        State("store-project-tile-focus", "data"),
        State("project-modal-trace-gang", "value"),
        State("project-modal-erections-range", "start_date"),
        State("project-modal-erections-range", "end_date"),
        State("project-modal-erections-search", "value"),
        prevent_initial_call=True,
    )
    def export_trace(
        main_clicks: int | None,
        modal_clicks: int | None,
        project_modal_clicks: int | None,
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        trace_gang_value: str | None,
        selected_gang: str | None,
        erections_start: str | None,
        erections_end: str | None,
        erections_search: str | None,
        focus_data: dict[str, Any] | None,
        project_modal_trace_gang: str | None,
        modal_start: str | None,
        modal_end: str | None,
        modal_search: str | None,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "project-modal-btn-export-trace":
            project_name = (focus_data or {}).get("project")
            if not project_name:
                raise PreventUpdate
            project_list = [project_name]
            month_list = _ensure_list(months)
            gang_list = _ensure_list(gangs)
            df_day = data_selector.select("erection")
            months_ts = resolve_months(month_list, quick_range)
            scoped = apply_filters(df_day, project_list, months_ts, gang_list)
            gang_for_sheet = project_modal_trace_gang or selected_gang
            benchmark_value = BENCHMARK_MT_PER_DAY
            project_info_df = project_info_provider() if project_info_provider else None

            range_start = _parse_completion_date(modal_start) or _default_completion_date()
            range_end = _parse_completion_date(modal_end) or range_start
            if range_start > range_end:
                range_start, range_end = range_end, range_start
            erections_export_df, _ = _prepare_erections_completed(
                scoped,
                range_start=range_start,
                range_end=range_end,
                responsibilities_provider=responsibilities_provider,
                search_text=modal_search,
            )
            erections_context = {
                "range_start": range_start,
                "range_end": range_end,
                "search_text": (modal_search or ""),
            }

            def _writer(buffer: BytesIO) -> None:
                buffer.write(
                    make_trace_workbook_bytes(
                        scoped,
                        months_ts,
                        project_list,
                        gang_list,
                        benchmark_value,
                        gang_for_sheet=gang_for_sheet,
                        config=config,
                        project_info=project_info_df,
                        erections_completed=erections_export_df,
                        erections_context=erections_context,
                    )
                )

            return dash.no_update, send_bytes(_writer, "Trace_Calcs.xlsx")

        if not (main_clicks or modal_clicks):
            raise PreventUpdate

        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        gang_list = _ensure_list(gangs)

        df_day = data_selector.select("erection")
        months_ts = resolve_months(month_list, quick_range)
        scoped = apply_filters(df_day, project_list, months_ts, gang_list)
        gang_for_sheet = trace_gang_value or selected_gang
        benchmark_value = BENCHMARK_MT_PER_DAY
        project_info_df = project_info_provider() if project_info_provider else None

        range_start = _parse_completion_date(erections_start) or _default_completion_date()
        range_end = _parse_completion_date(erections_end) or range_start
        if range_start > range_end:
            range_start, range_end = range_end, range_start

        erections_export_df, _ = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=responsibilities_provider,
            search_text=erections_search,
        )

        erections_context = {
            "range_start": range_start,
            "range_end": range_end,
            "search_text": (erections_search or ""),
        }

        def _writer(buffer: BytesIO) -> None:
            buffer.write(
                make_trace_workbook_bytes(
                    scoped,
                    months_ts,
                    project_list,
                    gang_list,
                    benchmark_value,
                    gang_for_sheet=gang_for_sheet,
                    config=config,
                    project_info=project_info_df,
                    erections_completed=erections_export_df,
                    erections_context=erections_context,
                )
            )

        return send_bytes(_writer, "Trace_Calcs.xlsx"), dash.no_update

    @app.callback(
        Output("global-performance-download-trace", "data"),
        Input("global-performance-btn-export-trace", "n_clicks"),
        State("store-global-performance-scope", "data"),
        State("global-performance-trace-gang", "value"),
        State("global-performance-selected-gang", "data"),
        prevent_initial_call=True,
    )
    def _export_global_performance_trace(
        export_clicks: int | None,
        scope_meta: dict[str, Any] | None,
        dropdown_value: str | None,
        selected_store_gang: str | None,
    ):
        if not export_clicks or not isinstance(scope_meta, dict):
            raise PreventUpdate
        min_erections_filter = _min_erections_from_meta(scope_meta)
        scoped = _scope_frame_from_store(scope_meta, "project_gang").copy()
        scoped = _filter_frame_for_min_erections(scoped, min_erections_filter)
        if scoped.empty:
            raise PreventUpdate
        selected = scope_meta.get("selected") or {}
        project_list = selected.get("projects", [])
        gang_list = selected.get("gangs", [])
        months_ts = _months_from_meta(scope_meta)
        gang_for_sheet = dropdown_value or selected_store_gang
        project_info_df = project_info_provider() if project_info_provider else None

        def _writer(buffer: BytesIO) -> None:
            buffer.write(
                make_trace_workbook_bytes(
                    scoped,
                    months_ts,
                    project_list,
                    gang_list,
                    BENCHMARK_MT_PER_DAY,
                    gang_for_sheet=gang_for_sheet,
                    config=config,
                    project_info=project_info_df,
                )
            )

        return send_bytes(_writer, "Trace_Calcs.xlsx")

    @app.callback(
        Output("global-performance-tbl-idle-intervals", "data"),
        Output("global-performance-tbl-daily-prod", "data"),
        Input("store-global-performance-scope", "data"),
        Input("store-global-performance-click-meta", "data"),
        Input("global-performance-trace-gang", "value"),
        Input("global-performance-selected-gang", "data"),
        prevent_initial_call=True,
    )
    def _update_global_performance_trace_tables(
        scope_meta: dict[str, Any] | None,
        meta: dict[str, Any] | None,
        dropdown_value: str | None,
        selected_store_gang: str | None,
    ):
        if not isinstance(scope_meta, dict):
            raise PreventUpdate
        min_erections_filter = _min_erections_from_meta(scope_meta)
        scope_for_tables = (
            scope_meta
            if min_erections_filter is None
            else _scope_meta_with_min_erections(scope_meta, min_erections_filter)
        )
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        meta_source = meta.get("source") if isinstance(meta, dict) else None
        meta_gang = meta.get("gang") if isinstance(meta, dict) else None
        meta_is_chart = meta_source in GLOBAL_MODAL_CHART_SOURCES and bool(meta_gang)
        if triggered_id == "store-global-performance-click-meta":
            gang_focus = meta_gang if meta_is_chart else (dropdown_value or selected_store_gang)
        else:
            gang_focus = dropdown_value or (meta_gang if meta_is_chart else selected_store_gang)
        if not gang_focus:
            raise PreventUpdate
        idle_df, daily_df = _prepare_trace_dataframes(scope_for_tables, gang_focus)
        return idle_df.to_dict("records"), daily_df.to_dict("records")

    @app.callback(
        Output("trace-gang", "options"),
        Output("trace-gang", "value"),
        Output("modal-trace-gang", "options"),
        Output("modal-trace-gang", "value"),
        Input("store-filtered-scope", "data"),
        Input("store-selected-gang", "data"),
        State("trace-gang", "value"),
    )
    def update_trace_gang_options(
        scope_meta: dict[str, Any] | None,
        clicked_gang: str | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, str]], str | None, list[dict[str, str]], str | None]:
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            raise PreventUpdate

        base = _scope_frame_from_store(scope_meta, "project")
        if base.empty or "gang_name" not in base.columns:
            options: list[dict[str, str]] = []
            return options, None, options, None

        gangs = (
            base["gang_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        gangs = sorted({g for g in gangs if g})
        options = [{"label": gang, "value": gang} for gang in gangs]

        if clicked_gang and clicked_gang in gangs:
            value = clicked_gang
        elif current_value and current_value in gangs:
            value = current_value
        else:
            value = None
        return options, value, options, value

    @app.callback(
        Output("global-performance-trace-gang", "options"),
        Output("global-performance-trace-gang", "value"),
        Input("store-global-performance-scope", "data"),
        Input("global-performance-selected-gang", "data"),
        State("global-performance-trace-gang", "value"),
    )
    def _sync_global_performance_trace_dropdown(
        scope_meta: dict[str, Any] | None,
        selected_gang: str | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        if not isinstance(scope_meta, dict) or "scopes" not in scope_meta:
            return [], None
        min_erections_filter = _min_erections_from_meta(scope_meta)
        base = _scope_frame_from_store(scope_meta, "project")
        base = _filter_frame_for_min_erections(base, min_erections_filter)
        if base.empty or "gang_name" not in base.columns:
            return [], None
        gangs = (
            base["gang_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        gangs = sorted({g for g in gangs if g})
        options = [{"label": gang, "value": gang} for gang in gangs]
        if not options:
            return [], None
        if selected_gang and selected_gang in gangs:
            value = selected_gang
        elif current_value and current_value in gangs:
            value = current_value
        else:
            value = gangs[0]
        return options, value

    # --- KPI Details drilldown: populate inline accordion ---
    def _filter_pch_header_pills(
        pill_components: list[tuple[str, Any]], pill_focus: str | None
    ) -> list[Any]:
        if not pill_focus:
            return [component for _, component in pill_components]
        focus = str(pill_focus).strip().lower()
        if not focus or focus in {"all", "default"}:
            return [component for _, component in pill_components]
        filtered = [component for key, component in pill_components if key == focus]
        return filtered or [component for _, component in pill_components]


    def _build_tile_metric_rows(
        *,
        mode: str,
        focus_metric: str | None,
        prod_current_value: float | None,
        prod_overall_value: float | None,
        total_current_value: float | None,
        total_planned_value: float | None,
        plan_available: bool = True,
        gangs_value: int | None = None,
        loss_value: float | None = None,
        tse_value: int | None = None,
        po_planned_value: float | None = None,
        po_done_value: float | None = None,
    ) -> list[html.Div]:
        """
        Build the dynamic KPI rows rendered inside each project tile.

        When *focus_metric* is provided (i.e., a summary pill opened the modal),
        the tile shows only the requested metric. Otherwise the default trio of
        production + totals rows is displayed.
        """

        is_stringing = mode == "stringing"
        prod_unit = "KM/day" if is_stringing else "MT/day"
        total_unit = "KM" if is_stringing else "Towers"
        loss_unit = "KM" if is_stringing else "MT"

        def _fmt_prod(value: float | None) -> str:
            if value is None or pd.isna(value):
                return "\u2014"
            return f"{float(value):.2f} {prod_unit}"

        def _fmt_prod_compact(value: float | None) -> str:
            if value is None or pd.isna(value):
                return "\u2014"
            return f"{float(value):.2f}"

        def _ensure_number(value: float | int | None, *, default: float = 0.0) -> float:
            if value is None or pd.isna(value):
                return default
            try:
                return float(value)
            except Exception:
                return default

        def _optional_number(value: float | int | None) -> float | None:
            if value is None or pd.isna(value):
                return None
            try:
                return float(value)
            except Exception:
                return None

        focus = (focus_metric or "").strip().lower()
        if focus not in _PCH_PILL_LABELS or focus == "projects":
            focus = None

        total_current = _ensure_number(total_current_value)
        plan_value_available = plan_available and total_planned_value is not None and not pd.isna(total_planned_value)
        total_planned = _ensure_number(total_planned_value) if plan_value_available else 0.0
        balance_value = max(total_planned - total_current, 0.0) if plan_value_available else None

        if is_stringing:
            delivered_label = f"{total_current:.1f} {total_unit}"
            planned_label = f"{total_planned:.1f} {total_unit}" if plan_value_available else "No Plan"
            balance_label = f"{balance_value:.1f} {total_unit}" if plan_value_available else "\u2014"
        else:
            delivered_label = f"{int(round(total_current))} {total_unit}"
            planned_label = f"{int(round(total_planned))} {total_unit}" if plan_value_available else "No Plan"
            balance_label = f"{int(round(balance_value))} {total_unit}" if plan_value_available else "\u2014"

        if plan_value_available:
            totals_display = f"{delivered_label} delivered / {planned_label} planned ({balance_label} balance)"
            totals_focus_display = f"{planned_label} / {delivered_label} / {balance_label}"
        else:
            totals_display = f"{delivered_label} delivered / No Plan available"
            totals_focus_display = f"No Plan / {delivered_label} / \u2014"

        def _row(label: str, value: str) -> html.Div:
            return html.Div(
                [
                    html.Span(f"{label} : ", className="me-2"),
                    dbc.Badge(value, color="dark", className="me-2", style={"fontSize": "1.05rem"}),
                ],
                className="mb-2",
            )

        if focus == "totals":
            return [_row("F/S Total Planned / Done / Balance", totals_focus_display)]
        if focus == "gangs":
            value = "\u2014" if gangs_value is None else f"{int(gangs_value):,}"
            return [_row("Gangs", value)]
        if focus == "productivity":
            prod_pair = f"{_fmt_prod_compact(prod_current_value)} / {_fmt_prod_compact(prod_overall_value)} {prod_unit}"
            return [_row("Productivity / Historical Avg", prod_pair)]
        if focus == "loss":
            if loss_value is None:
                value = "\u2014"
            else:
                value = f"{float(loss_value):.1f} {loss_unit}"
            return [_row("Lost Units", value)]
        if focus == "tse" and is_stringing:
            value = "\u2014" if tse_value is None else f"{int(tse_value):,}"
            return [_row("No. of TSE", value)]

        totals_row_label = f"{total_unit} This Month"
        totals_row_value = totals_display
        if is_stringing:
            totals_row_label = "F/S Total Planned / Done / Balance"
            totals_row_value = totals_focus_display
        else:
            totals_row_label = "Towers Delivered / Planned"
            delivered_count = int(round(total_current))
            if plan_value_available:
                totals_row_value = f"{delivered_count} / {int(round(total_planned))}"
            else:
                totals_row_value = f"{delivered_count} / No Plan Available"

        rows = [
            _row("Prod This Month", _fmt_prod(prod_current_value)),
            _row("Historical Avg", _fmt_prod(prod_overall_value)),
            _row(totals_row_label, totals_row_value),
        ]

        po_plan = _optional_number(po_planned_value)
        po_done = _optional_number(po_done_value)
        if is_stringing and focus is None and (po_plan is not None or po_done is not None):
            plan_label = f"{po_plan:.1f} KM" if po_plan is not None else "No Plan"
            done_label = f"{po_done:.1f} KM" if po_done is not None else "\u2014"
            if po_plan is not None:
                balance_value = max(po_plan - (po_done or 0.0), 0.0)
                balance_label = f"{balance_value:.1f} KM"
            else:
                balance_label = "\u2014"
            rows.append(
                _row(
                    "P/O Total Planned / Done / Balance",
                    f"{plan_label} / {done_label} / {balance_label}",
                )
            )

        return rows

    def _populate_kpi_pch(
        projects,
        months,
        quick_range,
        mode_value: str | None,
        method_filter,
        stringing_scope,
        *,
        use_modal_ids: bool = False,
        pill_focus: str | None = None,
    ):
        mode = (mode_value or "erection").strip().lower()
        try:
            import re as _re_slug
        except Exception:  # pragma: no cover - regex should be available, but keep fallback
            _re_slug = None

        focus_metric = None
        if pill_focus:
            focus_raw = str(pill_focus).strip().lower()
            if focus_raw in _PCH_PILL_LABELS and focus_raw != "projects":
                focus_metric = focus_raw

        idle_table_stringing = _idle_table_for_mode("stringing")
        idle_table_erection = _idle_table_for_mode("erection")

        tile_context = "modal" if use_modal_ids else "inline"
        tile_metadata: dict[str, dict[str, Any]] = {}

        def _slugify_pch(value: Any) -> str:
            text = str(value or "").strip().lower()
            if not text:
                return "unknown"
            if _re_slug is not None:
                text = _re_slug.sub(r"[^a-z0-9]+", "-", text)
            else:
                text = text.replace(" ", "-")
            text = text.strip("-")
            return text or "unknown"

        def _empty_pch_items(message: str) -> list[dbc.AccordionItem]:
            return [
                dbc.AccordionItem(
                    title="No PCH data",
                    children=html.Div(message, className="text-muted"),
                    item_id="pch-empty",
                    className="pch-section mb-2",
                )
            ]

        def _detect_project_column(frame: pd.DataFrame | None) -> str | None:
            """
            Return the first available project column in *frame* that can be used
            to associate gang rows back to a project label.
            """
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return None
            for candidate in ("project_name", "project", "Project Name", "project_name_display"):
                if candidate in frame.columns:
                    return candidate
            return None

        def _compact_project_token(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())

        def _project_code_key_sets(entry: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
            norm_keys: list[str] = []
            compact_keys: list[str] = []
            if not isinstance(entry, Mapping):
                return norm_keys, compact_keys
            seen: set[str] = set()
            for field in ("project_code", "code", "project", "project_name", "project_name_display"):
                code_value = _extract_project_code(entry.get(field))
                if not code_value:
                    continue
                canonical = code_value.lower()
                if canonical in seen:
                    continue
                seen.add(canonical)
                norm_key = _normalize_lower(code_value)
                if norm_key and norm_key not in norm_keys:
                    norm_keys.append(norm_key)
                compact_key = _compact_project_token(code_value)
                if compact_key and compact_key not in compact_keys:
                    compact_keys.append(compact_key)
            return norm_keys, compact_keys

        def _resolve_project_code(entry: Mapping[str, Any] | None) -> str:
            if not isinstance(entry, Mapping):
                return ""
            for field in ("project_code", "code", "project", "project_name", "project_name_display"):
                code_value = _extract_project_code(entry.get(field))
                if code_value:
                    return code_value
            return ""

        def _component_id_token(prefix: str, value: Any) -> str:
            base = _compact_project_token(value)
            if base:
                return base
            text = str(value or "").strip()
            if not text:
                text = prefix or "component"
            try:
                raw = text.encode("utf-8")
            except Exception:
                raw = text.encode("utf-8", errors="ignore")
            digest = hashlib.sha1(raw).hexdigest()[:8]
            return f"{prefix}-{digest}"

        def _filter_scope_by_projects(
            scope: pd.DataFrame | None, rows: list[dict[str, Any]]
        ) -> pd.DataFrame:
            if not isinstance(scope, pd.DataFrame) or scope.empty or not rows:
                return pd.DataFrame()
            match_codes: set[str] = set()
            for entry in rows:
                _, compact_keys = _project_code_key_sets(entry)
                match_codes.update(compact_keys)
            if not match_codes:
                return pd.DataFrame()

            mask = pd.Series(False, index=scope.index)
            for column in ("project_code", "project", "project_name", "project_name_display"):
                if column in scope.columns:
                    mask |= scope[column].astype(str).map(_compact_project_token).isin(match_codes)

            if not mask.any():
                return scope.iloc[0:0].copy()
            return scope.loc[mask].copy()

        def _project_scope_for_row(
            row: dict[str, Any],
            *,
            primary_scope: pd.DataFrame | None,
            fallback_scope: pd.DataFrame | None,
        ) -> pd.DataFrame:
            subset = _filter_scope_by_projects(primary_scope, [row])
            if subset.empty:
                subset = _filter_scope_by_projects(fallback_scope, [row])
            return subset

        def _count_gangs(frame: pd.DataFrame | None) -> int | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty or "gang_name" not in frame.columns:
                return None
            series = (
                frame["gang_name"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            series = series[series != ""]
            if series.empty:
                return 0
            return int(series.nunique())

        def _count_tse(frame: pd.DataFrame | None) -> int | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty or "method" not in frame.columns:
                return None
            mask = frame["method"].astype(str).str.strip().str.lower() == "tse"
            if not mask.any():
                return 0
            if "gang_name" in frame.columns:
                gangs = (
                    frame.loc[mask, "gang_name"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                gangs = gangs[gangs != ""]
                if not gangs.empty:
                    return int(gangs.nunique())
            return int(mask.sum())

        def _prepare_scope_for_loss(frame: pd.DataFrame | None) -> pd.DataFrame:
            if not isinstance(frame, pd.DataFrame):
                return pd.DataFrame()
            if frame.empty:
                return frame
            if "project_name" not in frame.columns:
                for alt in ("project", "project_name_display"):
                    if alt in frame.columns:
                        work = frame.copy()
                        work["project_name"] = work[alt].astype(str)
                        return work
            return frame

        def _compute_project_loss_value(frame: pd.DataFrame | None, *, mode: str) -> float | None:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return None
            metric_col = "daily_km" if mode == "stringing" else "daily_prod_mt"
            if metric_col not in frame.columns:
                return None
            work = _prepare_scope_for_loss(frame)
            if work.empty or "project_name" not in work.columns or "gang_name" not in work.columns:
                return None
            work = work.dropna(subset=["gang_name"]).copy()
            work["gang_name"] = work["gang_name"].astype(str).str.strip()
            work = work[work["gang_name"] != ""]
            if work.empty:
                return None
            try:
                if mode == "stringing":
                    overall_map, monthly_map = compute_project_baseline_maps_for(work, metric_col)
                else:
                    overall_map, monthly_map = compute_project_baseline_maps(work)
            except Exception:
                overall_map, monthly_map = {}, {}
            idle_table_local = _idle_table_for_mode(mode)
            month_filter_local: set[pd.Timestamp] | None = None
            if "month" in work.columns:
                months = pd.to_datetime(work["month"], errors="coerce").dropna()
                if not months.empty:
                    month_filter_local = set(months.unique().tolist())
            proj_col = _detect_project_column(work) or ("project_name" if "project_name" in work.columns else None)
            if proj_col and proj_col not in work.columns:
                proj_col = "project_name"
            if proj_col:
                gang_project_map = (
                    work[["gang_name", proj_col]]
                    .dropna(subset=["gang_name"])
                    .astype(str)
                    .rename(columns={proj_col: "project_name"})
                    .drop_duplicates(subset=["gang_name"], keep="last")
                    .set_index("gang_name")["project_name"]
                    .to_dict()
                )
            else:
                gang_project_map = {}
            default_project = str(work["project_name"].iloc[0]) if "project_name" in work.columns and not work.empty else None
            total_loss = 0.0
            for gang, gdf in work.groupby("gang_name"):
                if gdf.empty:
                    continue
                project_label = gang_project_map.get(str(gang), default_project)
                baseline_value = overall_map.get(project_label)
                monthly_lookup = monthly_map.get(project_label, {})
                try:
                    if mode == "stringing":
                        _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss_for_column(
                            gdf,
                            metric_column=metric_col,
                            loss_max_gap_days=config.loss_max_gap_days,
                            baseline_per_day=baseline_value,
                            baseline_by_month=monthly_lookup,
                            idle_intervals=idle_table_local,
                            allowed_months=month_filter_local,
                        )
                    else:
                        _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss(
                            gdf,
                            loss_max_gap_days=config.loss_max_gap_days,
                            baseline_mt_per_day=baseline_value,
                            baseline_by_month=monthly_lookup,
                            idle_intervals=idle_table_local,
                            allowed_months=month_filter_local,
                        )
                except Exception:
                    continue
                if pd.notna(loss_val):
                    total_loss += float(loss_val)
            return total_loss

        method_list = _normalize_str_list(method_filter, lower=True)
        deployment_scope = _normalize_deployment_filter(stringing_scope)
        pch_sections: list[dbc.AccordionItem] = []
        # Erection mode (existing flow)
        stringing_plan_month_map: dict[str, set[pd.Timestamp]] = (
            _load_stringing_plan_month_map(date_columns=_STRINGING_FS_DATE_COLUMNS) if mode == "stringing" else {}
        )

        def _project_has_plan(name: str, code: str) -> bool:
            return _stringing_project_has_plan(stringing_plan_month_map, name, code)

        def _project_plan_months(name: str, code: str) -> list[pd.Timestamp]:
            return _stringing_plan_months_for_project(stringing_plan_month_map, name, code)

        if mode == "stringing":
            # Stringing mode: build PCH-wise planned vs delivered (KM)
            project_list = _ensure_list(projects)
            month_list = _ensure_list(months)
            months_ts = resolve_months(month_list, quick_range)

            # Month range for display/derivations
            if months_ts:
                range_start = pd.Timestamp(min(months_ts)).normalize()
                range_end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
            else:
                today = pd.Timestamp.today().normalize()
                range_start = today.to_period("M").start_time.normalize()
                range_end = (today + pd.offsets.MonthEnd(0)).normalize()
            current_month_ts = range_end.to_period("M").to_timestamp()

            def _stringing_delivery_stats(
                source: pd.DataFrame,
            ) -> tuple[pd.DataFrame, pd.Series, bool]:
                delivered_series = pd.Series(dtype=float)
                if not isinstance(source, pd.DataFrame) or source.empty:
                    return pd.DataFrame(columns=["delivered_km"]), delivered_series, False
                proj_col = "project_name" if "project_name" in source.columns else (
                    "project" if "project" in source.columns else None
                )
                if proj_col is None:
                    return pd.DataFrame(columns=["delivered_km"]), delivered_series, True
                scoped_norm = source.copy()
                if proj_col != "project_name":
                    scoped_norm = scoped_norm.rename(columns={proj_col: "project_name"})
                scoped_norm["project_name"] = scoped_norm["project_name"].astype(str).str.strip()
                completion_rows = _filter_completion_rows(scoped_norm, completion_column="fs_complete_date")
                delivered_df = pd.DataFrame(columns=["delivered_km"])
                if not completion_rows.empty and "length_km" in completion_rows.columns:
                    delivered_df = (
                        completion_rows.groupby("project_name")["length_km"]
                        .sum()
                        .rename("delivered_km")
                        .to_frame()
                    )
                    completion_rows = completion_rows.copy()
                    completion_rows["month"] = pd.to_datetime(completion_rows["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                    current_rows = completion_rows[completion_rows["month"] == current_month_ts]
                    if not current_rows.empty:
                        delivered_series = current_rows.groupby("project_name")["length_km"].sum()
                elif "daily_km" in scoped_norm.columns:
                    delivered_df = (
                        scoped_norm.groupby("project_name")["daily_km"].sum().rename("delivered_km").to_frame()
                    )
                    scoped_month = scoped_norm.copy()
                    if "month" not in scoped_month.columns and "date" in scoped_month.columns:
                        scoped_month["date"] = pd.to_datetime(scoped_month["date"], errors="coerce")
                        scoped_month = scoped_month.dropna(subset=["date"])
                        scoped_month["month"] = scoped_month["date"].dt.to_period("M").dt.to_timestamp()
                    elif "month" in scoped_month.columns:
                        scoped_month["month"] = pd.to_datetime(scoped_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                    if "month" in scoped_month.columns:
                        scoped_current = scoped_month[scoped_month["month"] == current_month_ts]
                        if not scoped_current.empty:
                            delivered_series = (
                                scoped_current.groupby("project_name")["daily_km"].sum()
                            )
                else:
                    delivered_df = pd.DataFrame(columns=["delivered_km"])
                return delivered_df, delivered_series, False

            # Delivered KM from per-day stringing dataset
            df_day = data_selector.select("stringing")
            scoped_base = apply_filters(df_day, project_list, months_ts, [])
            scoped = _filter_frame_for_deployment(scoped_base, deployment_scope)
            try:
                scope_frames, _, _ = _build_scope_frames(
                    "stringing",
                    project_list=project_list,
                    gang_list=[],
                    months_value=month_list,
                    quick_range=quick_range,
                    method_values=method_list,
                    deployment_filter=deployment_scope,
                )
                scope_full = scope_frames.get("full", pd.DataFrame()).copy()
            except Exception:
                scope_full = pd.DataFrame()
            delivered_km_by_project, delivered_km_current_series, missing_project_info = _stringing_delivery_stats(scoped)
            if missing_project_info:
                return _empty_pch_items("Missing project information in the dataset."), None
            _, delivered_km_current_series_all, _ = _stringing_delivery_stats(scoped_base)

            plan_union, plan_current_lookup = _stringing_plan_totals_by_project(
                months_ts,
                current_month=current_month_ts,
            )
            planned_km_current_series = plan_current_lookup.copy()

            try:
                tse_norm_lookup, tse_alias_lookup = _get_stringing_tse_lookup()
            except Exception:
                LOGGER.exception("Failed to build TSE lookup")
                tse_norm_lookup, tse_alias_lookup = {}, {}

            # Merge planned and delivered into a projects table
            delivered_union = delivered_km_by_project.copy()
            if not delivered_union.empty:
                delivered_union = delivered_union.copy()
                delivered_union["project_name_delivered"] = delivered_union.index.astype(str)
                delivered_union["project_key_norm"] = delivered_union["project_name_delivered"].map(
                    lambda value: _compact_project_key(value) or _normalize_lower(value)
                )
                delivered_union = delivered_union[delivered_union["project_key_norm"].astype(str).str.strip() != ""]
                if not delivered_union.empty:
                    delivered_union = (
                        delivered_union.set_index("project_key_norm")[["delivered_km", "project_name_delivered"]]
                    )
            else:
                delivered_union = pd.DataFrame(columns=["delivered_km", "project_name_delivered"]).set_index(
                    pd.Index([], name="project_key_norm")
                )

            if not plan_union.empty:
                plan_union = plan_union.copy()
            else:
                plan_union = pd.DataFrame(columns=["planned_km", "project_name_plan"]).set_index(
                    pd.Index([], name="project_key_norm")
                )

            po_union = pd.DataFrame(columns=["project_name_po"]).set_index(
                pd.Index([], name="project_key_norm")
            )
            try:
                po_frame = _get_stringing_po_daily_frame()
            except Exception:
                po_frame = pd.DataFrame()
            if isinstance(po_frame, pd.DataFrame) and not po_frame.empty:
                po_scoped = _stringing_scope(po_frame, method_list)
                po_filtered = apply_filters(po_scoped, project_list, months_ts, [])
                po_filtered = _filter_frame_for_deployment(po_filtered, deployment_scope)
                if not po_filtered.empty:
                    po_proj_col = "project_name" if "project_name" in po_filtered.columns else (
                        "project" if "project" in po_filtered.columns else None
                    )
                    if po_proj_col:
                        po_work = po_filtered.copy()
                        if po_proj_col != "project_name":
                            po_work = po_work.rename(columns={po_proj_col: "project_name"})
                        po_work["project_name"] = po_work["project_name"].astype(str).str.strip()
                        po_work["project_key_norm"] = po_work["project_name"].map(
                            lambda value: _compact_project_key(value) or _normalize_lower(value)
                        )
                        po_work = po_work[po_work["project_key_norm"].astype(str).str.strip() != ""]
                        if not po_work.empty:
                            po_union = (
                                po_work.drop_duplicates(subset=["project_key_norm"])
                                .set_index("project_key_norm")[["project_name"]]
                                .rename(columns={"project_name": "project_name_po"})
                            )

            combined_projects = (
                plan_union.join(delivered_union, how="outer").join(po_union, how="outer")
                .fillna({"planned_km": 0.0, "delivered_km": 0.0})
            )
            combined_projects.index.name = "__project_key_norm_idx__"
            projects_df = (
                combined_projects.reset_index()
                .rename(columns={"__project_key_norm_idx__": "project_key_norm"})
            )
            delivered_names = projects_df.pop("project_name_delivered").fillna("").astype(str)
            plan_names = projects_df.pop("project_name_plan").fillna("").astype(str)
            po_names = (
                projects_df.pop("project_name_po").fillna("").astype(str)
                if "project_name_po" in projects_df.columns
                else pd.Series([""] * len(projects_df), index=projects_df.index)
            )
            base_name = delivered_names.where(delivered_names.str.strip() != "", plan_names)
            base_name = base_name.where(base_name.str.strip() != "", po_names)
            base_name = base_name.fillna("")
            fallback_mask = base_name.str.strip() == ""
            if fallback_mask.any():
                base_name = base_name.mask(fallback_mask, projects_df.loc[fallback_mask, "project_key_norm"])
            projects_df["project_name"] = base_name.astype(str).map(_format_stringing_project_label)
            project_label_by_key = (
                projects_df.set_index("project_key_norm")["project_name"].to_dict()
                if not projects_df.empty
                else {}
            )

            if not planned_km_current_series.empty:
                planned_km_current_series = planned_km_current_series.rename(
                    index=lambda key: project_label_by_key.get(str(key), key)
                )
            else:
                planned_km_current_series = pd.Series(dtype=float)
            projects_df = projects_df.drop(columns=["project_name_plan", "project_name_delivered"], errors="ignore")
            projects_df["project_name_display"] = projects_df["project_name"].astype(str)

            # Project meta (PCH, managers) from Project Details
            try:
                info_df = project_info_provider() if callable(project_info_provider) else None
            except Exception:
                info_df = None
            pch_col = "pch"
            info_key_map: dict[str, int] = {}
            if isinstance(info_df, pd.DataFrame) and not info_df.empty:
                info = info_df.copy()
                info["project_name_display"] = info.get("Project Name", info.get("project_name", "")).astype(str)
                info["project_name_norm"] = info["project_name_display"].map(_normalize_lower)
                pch_col = None
                for cand in ("PCH", "pch", "PCH Name", "PCHName", "pch_name"):
                    if cand in info.columns:
                        pch_col = cand
                        break
                if pch_col is None:
                    info["pch"] = ""
                    pch_col = "pch"
                try:
                    import re as _re_key

                    def _compact_code(value: str) -> str:
                        return _re_key.sub(r"[^a-z0-9]", "", (value or "").lower())
                except Exception:

                    def _compact_code(value: str) -> str:
                        return str(value or "").strip().lower().replace(" ", "")

                name_keys = info["project_name_display"].astype(str).map(_compact_code)
                for idx, key in zip(info.index, name_keys):
                    if key and key not in info_key_map:
                        info_key_map[key] = idx
                for code_col in ("project_code", "Project Code"):
                    if code_col in info.columns:
                        code_keys = info[code_col].astype(str).map(_compact_code)
                        for idx, key in zip(info.index, code_keys):
                            if key and key not in info_key_map:
                                info_key_map[key] = idx
            else:
                info = pd.DataFrame(columns=["project_name_display", "project_name_norm", "pch", "regional_mgr", "project_mgr", "planning_eng"])
                def _compact_code(value: str) -> str:
                    return str(value or "").strip().lower().replace(" ", "")

            proj_info_pch = {}
            proj_info_pch_norm = {}
            if not info.empty and "pch" in info.columns:
                proj_info_pch = dict(zip(info["project_name_display"], info["pch"].astype(str)))
                proj_info_pch_norm = { _normalize_lower(k): str(v) for k, v in proj_info_pch.items() if str(k).strip() }

            try:
                from .pch_normalizer import normalize_pch as _normalize_pch, CANONICAL_PCH_PRIMARY as _PCH_ORDER
            except Exception:
                def _normalize_pch(v):
                    return str(v or "").strip()
                _PCH_ORDER = ()

            # Build structure: PCH -> list of project tiles
            projects_rows: dict[str, list[dict[str, Any]]] = {}
            for _, row in projects_df.iterrows():
                proj = str(row.get("project_name_display", "")).strip()
                if not proj:
                    continue
                planned_km = float(row.get("planned_km", 0.0) or 0.0)
                delivered_km = float(row.get("delivered_km", 0.0) or 0.0)
                # Meta join
                meta = info[info.get("project_name_norm", "").astype(str) == _normalize_lower(proj)].iloc[:1] if not info.empty else pd.DataFrame()
                if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                    target_key = _compact_code(proj)
                    if target_key and target_key in info_key_map:
                        meta = info.loc[[info_key_map[target_key]]]
                raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
                if (not raw_pch) and proj:
                    raw_pch = proj_info_pch.get(proj, "") or proj_info_pch_norm.get(_normalize_lower(proj), "")
                # Use normalized PCH if known; otherwise keep the original as-is (no 'Unassigned')
                pch_label = _normalize_pch(raw_pch) or str(raw_pch or "").strip()
                try:
                    proj_code = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                        meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                    )) if isinstance(meta, pd.DataFrame) else ""
                except Exception:
                    proj_code = ""
                if isinstance(meta, pd.DataFrame) and not meta.empty:
                    proj_display_series = meta.get("project_name_display", meta.get("project_name", pd.Series([proj], index=meta.index)))
                    proj_display_name = str(proj_display_series.iloc[0]) if not proj_display_series.empty else str(proj)
                else:
                    proj_display_name = str(proj)
                proj_display = f"{proj_code} : {proj_display_name}".strip(" :") if proj_code else proj_display_name
                plan_months = _project_plan_months(proj_display_name, proj_code or proj_display_name)
                resolved_code = (
                    _extract_project_code(proj_code)
                    or _extract_project_code(proj_display)
                    or _extract_project_code(proj)
                )
                if not resolved_code:
                    resolved_code = proj_code or proj
                rec = {
                    "project_name": proj_display,
                    "project_code": resolved_code,
                    "code": resolved_code,
                    "regional_mgr": (meta.get("regional_mgr", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns) else ""),
                    "project_mgr": (meta.get("project_mgr", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns) else ""),
                    "planning_eng": (meta.get("planning_eng", pd.Series([""])).iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns) else ""),
                    # store KM in MT fields to reuse downstream structure but change labels to KM
                    "planned_mt": round(planned_km, 1),
                    "delivered_mt": round(delivered_km, 1),
                    # counts not applicable for stringing
                    "planned_nos": 0,
                    "delivered_nos": 0,
                    "_stringing_plan_months": plan_months,
                    "_stringing_plan_available": bool(plan_months),
                }
                projects_rows.setdefault(pch_label, []).append(rec)

            def _pch_sort_key(name: str) -> tuple[int, str]:
                # Put empty/blank PCH last; then canonical order; then alphabetical
                if not str(name or "").strip():
                    return (2, "")
                try:
                    idx = list(_PCH_ORDER).index(name)
                    return (0, f"{idx:03d}")
                except ValueError:
                    return (1, str(name))

            delivered_current_norm_map: dict[str, float] = {}
            delivered_current_compact_map: dict[str, float] = {}
            delivered_current_norm_map_all: dict[str, float] = {}
            delivered_current_compact_map_all: dict[str, float] = {}
            planned_current_norm_map: dict[str, float] = {}
            planned_current_compact_map: dict[str, float] = {}
            prod_current_norm_map: dict[str, float] = {}
            prod_current_compact_map: dict[str, float] = {}
            prod_overall_norm_map: dict[str, float] = {}
            prod_overall_compact_map: dict[str, float] = {}

            def _build_lookup_maps(source: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
                norm_map: dict[str, float] = {}
                compact_map: dict[str, float] = {}
                if not source:
                    return norm_map, compact_map
                for key, raw_val in source.items():
                    text = str(key or "").strip()
                    if not text:
                        continue
                    try:
                        value = float(raw_val)
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(value):
                        continue
                    norm_key = _normalize_lower(text)
                    if norm_key and norm_key not in norm_map:
                        norm_map[norm_key] = value
                    compact_key = _compact_code(text)
                    if compact_key and compact_key not in compact_map:
                        compact_map[compact_key] = value
                return norm_map, compact_map

            if isinstance(delivered_km_current_series, pd.Series) and not delivered_km_current_series.empty:
                delivered_current_norm_map, delivered_current_compact_map = _build_lookup_maps(delivered_km_current_series.to_dict())
            if isinstance(delivered_km_current_series_all, pd.Series) and not delivered_km_current_series_all.empty:
                delivered_current_norm_map_all, delivered_current_compact_map_all = _build_lookup_maps(
                    delivered_km_current_series_all.to_dict()
                )
            if isinstance(planned_km_current_series, pd.Series) and not planned_km_current_series.empty:
                planned_current_norm_map, planned_current_compact_map = _build_lookup_maps(planned_km_current_series.to_dict())

            po_plan_lookup_raw: dict[str, float] = {}
            po_plan_norm_map: dict[str, float] = {}
            po_plan_compact_map: dict[str, float] = {}
            try:
                po_plan_union, _ = _stringing_plan_totals_by_project(
                    months_ts,
                    current_month=None,
                    date_columns=_STRINGING_PO_DATE_COLUMNS,
                )
            except Exception:
                LOGGER.exception("Failed to load P/O plan totals for tiles")
                po_plan_union = pd.DataFrame()
            if isinstance(po_plan_union, pd.DataFrame) and not po_plan_union.empty:
                po_plan_reset = po_plan_union.reset_index().rename(columns={"index": "project_key_norm"})
                for _, entry in po_plan_reset.iterrows():
                    plan_value = entry.get("planned_km")
                    if pd.isna(plan_value):
                        continue
                    try:
                        numeric_value = float(plan_value)
                    except (TypeError, ValueError):
                        continue
                    label_candidates = [
                        entry.get("project_name_plan"),
                        project_label_by_key.get(str(entry.get("project_key_norm")), ""),
                        entry.get("project_key_norm"),
                    ]
                    for candidate in label_candidates:
                        text = str(candidate or "").strip()
                        if not text:
                            continue
                        po_plan_lookup_raw[text] = numeric_value
                if po_plan_lookup_raw:
                    po_plan_norm_map, po_plan_compact_map = _build_lookup_maps(po_plan_lookup_raw)

            po_done_lookup_raw: dict[str, float] = {}
            po_done_norm_map: dict[str, float] = {}
            po_done_compact_map: dict[str, float] = {}
            po_frame = _get_stringing_po_daily_frame()
            if isinstance(po_frame, pd.DataFrame) and not po_frame.empty:
                po_scoped = _stringing_scope(po_frame, method_list)
                po_filtered = apply_filters(po_scoped, project_list, months_ts, [])
                po_filtered = _filter_frame_for_deployment(po_filtered, deployment_scope)
                if isinstance(po_filtered, pd.DataFrame) and not po_filtered.empty:
                    proj_col = "project_name" if "project_name" in po_filtered.columns else (
                        "project" if "project" in po_filtered.columns else None
                    )
                    if proj_col:
                        work = po_filtered.copy()
                        work[proj_col] = work[proj_col].astype(str).str.strip()
                        work = work.rename(columns={proj_col: "project_name"})
                        work = work[work["project_name"] != ""]
                        completion_work = _filter_completion_rows(work, completion_column="po_completion_date")
                        if not completion_work.empty and "length_km" in completion_work.columns:
                            po_grouped = completion_work.groupby("project_name")["length_km"].sum()
                        elif "daily_km" in work.columns:
                            work = work.dropna(subset=["project_name", "daily_km"])
                            po_grouped = work.groupby("project_name")["daily_km"].sum(min_count=1)
                        else:
                            po_grouped = pd.Series(dtype=float)
                        if not po_grouped.empty:
                            for label, raw_val in po_grouped.items():
                                if pd.isna(raw_val):
                                    continue
                                try:
                                    numeric_value = float(raw_val)
                                except (TypeError, ValueError):
                                    continue
                                display_label = _format_stringing_project_label(str(label))
                                norm_key = _compact_project_key(label) or _normalize_lower(label)
                                for name in (display_label, project_label_by_key.get(norm_key), norm_key):
                                    if not name:
                                        continue
                                    text = str(name).strip()
                                    if text:
                                        po_done_lookup_raw[text] = numeric_value
            if po_done_lookup_raw:
                po_done_norm_map, po_done_compact_map = _build_lookup_maps(po_done_lookup_raw)

            if isinstance(df_day, pd.DataFrame) and not df_day.empty and "daily_km" in df_day.columns:
                day_filtered = df_day.copy()
                if project_list and "project_name" in day_filtered.columns:
                    project_filter_values = [str(p).strip() for p in project_list if str(p).strip()]
                    if project_filter_values:
                        day_filtered = day_filtered[day_filtered["project_name"].astype(str).str.strip().isin(project_filter_values)]
                project_name_col = "project_name" if "project_name" in day_filtered.columns else ("project" if "project" in day_filtered.columns else None)
                if project_name_col:
                    day_filtered[project_name_col] = day_filtered[project_name_col].astype(str).str.strip()
                    day_filtered = day_filtered.rename(columns={project_name_col: "project_name"})
                if "month" not in day_filtered.columns and "date" in day_filtered.columns:
                    day_filtered["date"] = pd.to_datetime(day_filtered["date"], errors="coerce")
                    day_filtered["month"] = day_filtered["date"].dt.to_period("M").dt.to_timestamp()
                elif "month" in day_filtered.columns:
                    day_filtered["month"] = pd.to_datetime(day_filtered["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                day_filtered = day_filtered.dropna(subset=["project_name", "daily_km"])
                baseline_payload = {
                    "kind": "stringing-day-baseline",
                    "projects": sorted(project_list),
                    "months": [ts.isoformat() for ts in months_ts],
                    "rows": int(len(day_filtered.index)),
                    "version": int(df_day.attrs.get("_appdata_version", 0)),
                }
                baseline_token = f"stringing-baseline::{_hash_cache_payload(baseline_payload)}"

                def _compute_day_baseline() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                    if day_filtered.empty:
                        return {}, {}
                    return compute_project_baseline_maps_for(day_filtered, "daily_km")

                overall_map, _ = _cached_global_result(
                    baseline_token,
                    _compute_day_baseline,
                    clone=_clone_baseline_result,
                )
                prod_overall_norm_map, prod_overall_compact_map = _build_lookup_maps(overall_map)
                current_scope = day_filtered[day_filtered["month"] == current_month_ts] if "month" in day_filtered.columns else pd.DataFrame()
                if not current_scope.empty:
                    current_payload = dict(baseline_payload)
                    current_payload.update(
                        {
                            "window": "current",
                            "month": current_month_ts.isoformat(),
                            "rows": int(len(current_scope.index)),
                        }
                    )
                    current_token = f"stringing-baseline::{_hash_cache_payload(current_payload)}"

                    def _compute_current_baseline() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                        return compute_project_baseline_maps_for(current_scope, "daily_km")

                    current_map, _ = _cached_global_result(
                        current_token,
                        _compute_current_baseline,
                        clone=_clone_baseline_result,
                    )
                    prod_current_norm_map, prod_current_compact_map = _build_lookup_maps(current_map)
                if not prod_current_norm_map:
                    prod_current_norm_map, prod_current_compact_map = prod_overall_norm_map.copy(), prod_overall_compact_map.copy()

            pch_sections = []
            for pch in sorted(projects_rows.keys(), key=_pch_sort_key):
                rows = projects_rows[pch]
                project_count = len(rows)
                project_codes: list[str] = []
                prod_current_values: list[float] = []
                prod_overall_values: list[float] = []
                delivered_month_total = 0.0
                delivered_month_total_all = 0.0
                planned_month_total = 0.0
                planned_value_count = 0
                gangs_total = 0
                lost_units_total = 0.0

                def _project_lookup_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
                    return _project_code_key_sets(row)

                def _lookup_with_keys(norm_keys: list[str], compact_keys: list[str], norm_map: Mapping[str, float], compact_map: Mapping[str, float]) -> float | None:
                    for key in norm_keys:
                        if key in norm_map:
                            return norm_map[key]
                    for key in compact_keys:
                        if key in compact_map:
                            return compact_map[key]
                    return None

                for r in sorted(rows, key=lambda x: str(x["project_name"])):
                    norm_keys, compact_keys = _project_lookup_keys(r)
                    code_value = _resolve_project_code(r)
                    if code_value and code_value not in project_codes:
                        project_codes.append(code_value)
                    has_plan = bool(r.get("_stringing_plan_available"))

                    prod_current_val = _lookup_with_keys(norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map)
                    if prod_current_val is not None:
                        prod_current_values.append(prod_current_val)

                    prod_overall_val = _lookup_with_keys(norm_keys, compact_keys, prod_overall_norm_map, prod_overall_compact_map)
                    if prod_overall_val is not None:
                        prod_overall_values.append(prod_overall_val)

                    delivered_month_val = _lookup_with_keys(norm_keys, compact_keys, delivered_current_norm_map, delivered_current_compact_map)
                    if delivered_month_val is not None:
                        delivered_month_total += float(delivered_month_val)
                    delivered_month_val_all = _lookup_with_keys(
                        norm_keys,
                        compact_keys,
                        delivered_current_norm_map_all,
                        delivered_current_compact_map_all,
                    )
                    if delivered_month_val_all is not None:
                        delivered_month_total_all += float(delivered_month_val_all)

                    planned_month_val = _lookup_with_keys(norm_keys, compact_keys, planned_current_norm_map, planned_current_compact_map)
                    if has_plan and planned_month_val is not None:
                        planned_month_total += float(planned_month_val)
                        planned_value_count += 1

                # Derive Gangs and Lost Units per PCH (stringing)
                try:
                    unit_short = "KM"
                    metric_col = "daily_km"
                    sub = _filter_scope_by_projects(scope_full, rows)
                    if sub.empty:
                        sub = _filter_scope_by_projects(scoped, rows)
                    if not sub.empty:
                        if "gang_name" in sub.columns:
                            gangs_total = (
                                sub["gang_name"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                            )
                        try:
                            overall_map, monthly_map = compute_project_baseline_maps_for(sub, metric_col)
                        except Exception:
                            overall_map, monthly_map = {}, {}
                        proj_col_for_gang = _detect_project_column(sub)
                        if proj_col_for_gang:
                            gang_project_map = (
                                sub[["gang_name", proj_col_for_gang]]
                                .dropna()
                                .astype(str)
                                .rename(columns={proj_col_for_gang: "project_name"})
                                .drop_duplicates(subset=["gang_name", "project_name"], keep="last")
                                .set_index("gang_name")["project_name"]
                                .to_dict()
                            )
                        else:
                            gang_project_map = {}
                        month_filter = _month_filter_for_frame(sub)
                        for gang, gdf in sub.groupby("gang_name"):
                            if gdf.empty:
                                continue
                            proj_for_gang = gang_project_map.get(str(gang))
                            ov = overall_map.get(proj_for_gang)
                            mm = monthly_map.get(proj_for_gang, {})
                            try:
                                _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss_for_column(
                                    gdf,
                                    metric_column=metric_col,
                                    loss_max_gap_days=config.loss_max_gap_days,
                                    baseline_per_day=ov,
                                    baseline_by_month=mm,
                                    idle_intervals=idle_table_stringing,
                                    allowed_months=month_filter,
                                )
                                if pd.notna(loss_val):
                                    lost_units_total += float(loss_val)
                            except Exception:
                                pass
                except Exception:
                    gangs_total = gangs_total or 0
                    lost_units_total = lost_units_total or 0.0

                prod_current_avg = float(sum(prod_current_values) / len(prod_current_values)) if prod_current_values else None
                prod_overall_avg = float(sum(prod_overall_values) / len(prod_overall_values)) if prod_overall_values else None

                fmt_prod_current = f"{prod_current_avg:.2f}" if prod_current_avg is not None else "\u2014"
                fmt_prod_overall = f"{prod_overall_avg:.2f}" if prod_overall_avg is not None else "\u2014"
               
                projects_label = f"Projects: {', '.join(project_codes)}" if project_codes else f"Projects: {project_count}"
                km_delivered_label = round(delivered_month_total, 1)
                km_delivered_all_label = round(delivered_month_total_all, 1)
                if planned_value_count:
                    km_planned_value = round(planned_month_total, 1)
                    km_balance_value = round(max(km_planned_value - km_delivered_all_label, 0.0), 1)
                    planned_display = f"{km_planned_value:.1f}"
                    balance_display = f"{km_balance_value:.1f}"
                else:
                    planned_display = "No Plan"
                    balance_display = "\u2014"
                pill_components = [
                                    ("projects", dbc.Button(
                                        projects_label,
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "projects"},
                                        className="pch-pill pch-pill-projects mb-1", color="link"
                                    )),
                                    ("productivity", dbc.Button(
                                        f"Productivity (Month/Overall): {fmt_prod_current} / {fmt_prod_overall} KM/day",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "productivity"},
                                        className="pch-pill pch-pill-prod-month mb-1", color="link"
                                    )),
                                    ("totals", dbc.Button(
                                        f"F/S Total Planned / Done / Balance: {planned_display} / {km_delivered_label:.1f} / {balance_display} KM",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "totals"},
                                        className="pch-pill pch-pill-towers mb-1", color="link"
                                    )),
                                    ("gangs", dbc.Button(
                                        f"Gangs: {int(gangs_total):,}",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "gangs"},
                                        className="pch-pill pch-pill-gangs mb-1", color="link"
                                    )),
                                    ("loss", dbc.Button(
                                        f"Lost Units: {lost_units_total:.1f} KM",
                                        id={"type": "summary-pill-trigger", "mode": "stringing", "metric": "loss"},
                                        className="pch-pill pch-pill-loss mb-1", color="link"
                                    )),
                                ]

                title_component = html.Div(
                    [
                        html.Span(str(pch or "Unassigned"), className="fw-semibold"),
                        html.Div(
                            _filter_pch_header_pills(pill_components, pill_focus),
                            className="pch-pill-group ms-auto d-none d-md-flex",
                        ),
                    ],
                    className="d-flex align-items-center justify-content-between w-100",
                )

                tile_cols = []
                for r in sorted(rows, key=lambda x: str(x["project_name"])):
                    proj_name = str(r["project_name"]).strip()
                    display_code = _resolve_project_code(r)
                    if display_code and _compact_project_token(display_code) != _compact_project_token(proj_name):
                        proj_title = f"{display_code} : {proj_name}"
                    else:
                        proj_title = proj_name

                    raw_code_for_keys = display_code or r.get("project_code") or r.get("project_key") or proj_name
                    proj_code = _compact_project_token(str(raw_code_for_keys or proj_name))

                    tile_summary_children = [
                        html.Div(html.Strong(proj_title), className="mb-2"),
                        html.Div([
                            html.Span("Regional Manager : ", className="text-muted me-1"),
                            dbc.Badge(r.get("regional_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold"),
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Project Manager : ", className="text-muted me-1"),
                            dbc.Badge(r.get("project_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold"),
                        ], className="mb-2"),
                    ]

                    norm_keys, compact_keys = _project_lookup_keys(r)

                    project_scope_cache: pd.DataFrame | None = None

                    def _lazy_scope() -> pd.DataFrame:
                        nonlocal project_scope_cache
                        if project_scope_cache is None:
                            project_scope_cache = _project_scope_for_row(
                                r,
                                primary_scope=scope_full,
                                fallback_scope=scoped,
                            )
                        return project_scope_cache

                    gangs_metric = None
                    loss_metric = None
                    tse_metric = None
                    if focus_metric in {"gangs", "loss", "tse"}:
                        scope_subset = _lazy_scope()
                        if focus_metric == "gangs":
                            gangs_metric = _count_gangs(scope_subset)
                        elif focus_metric == "loss":
                            loss_metric = _compute_project_loss_value(scope_subset, mode="stringing")
                        elif focus_metric == "tse":
                            tse_value, _canon = _resolve_tse_value(
                                norm_keys,
                                compact_keys,
                                tse_norm_lookup,
                                tse_alias_lookup,
                            )
                            if tse_value is not None:
                                tse_metric = int(tse_value)
                            else:
                                tse_metric = _count_tse(scope_subset)
                    prod_current_value = _lookup_with_keys(
                        norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map
                    )
                    prod_overall_value = _lookup_with_keys(
                        norm_keys, compact_keys, prod_overall_norm_map, prod_overall_compact_map
                    )
                    delivered_current_value = _lookup_with_keys(
                        norm_keys, compact_keys, delivered_current_norm_map, delivered_current_compact_map
                    )
                    planned_current_value = _lookup_with_keys(
                        norm_keys, compact_keys, planned_current_norm_map, planned_current_compact_map
                    )
                    po_plan_value = _lookup_with_keys(
                        norm_keys, compact_keys, po_plan_norm_map, po_plan_compact_map
                    )
                    po_done_value = _lookup_with_keys(
                        norm_keys, compact_keys, po_done_norm_map, po_done_compact_map
                    )

                    plan_available = bool(r.get("_stringing_plan_available"))
                    plan_total_value = float(r.get("planned_km", 0.0) or 0.0)
                    plan_has_data = plan_available or plan_total_value > 0
                    if planned_current_value is None and plan_has_data:
                        planned_effective = plan_total_value
                    else:
                        planned_effective = planned_current_value
                    tile_summary_children.extend(
                        _build_tile_metric_rows(
                            mode="stringing",
                            focus_metric=focus_metric,
                            prod_current_value=prod_current_value,
                            prod_overall_value=prod_overall_value,
                            total_current_value=(
                                delivered_current_value if delivered_current_value is not None else r.get("delivered_mt", 0.0)
                            ),
                            total_planned_value=planned_effective if plan_has_data else None,
                            plan_available=plan_has_data,
                            gangs_value=gangs_metric,
                            loss_value=loss_metric,
                            tse_value=tse_metric,
                            po_planned_value=po_plan_value,
                            po_done_value=po_done_value,
                        )
                    )

                    project_token = _component_id_token("proj", proj_name)
                    card_id = {
                        "type": "project-tile-trigger",
                        "mode": "stringing",
                        "project": project_token,
                        "context": tile_context,
                    }
                    tile_metadata[project_token] = {
                        "project": proj_name,
                        "code": display_code or proj_name,
                        "display": proj_title,
                        "mode": "stringing",
                        "pch": str(pch),
                    }

                    tile_body_children = [
                        html.Div(tile_summary_children, className="project-tile-summary"),
                    ]
                    stringing_month_buttons: list[Any] = []
                    available_months = list(r.get("_stringing_plan_months") or [])
                    if available_months:
                        stringing_month_buttons.append(html.Span("Monthly Plan (Stringing) : ", className="me-2"))
                        available_months = available_months[-2:]
                        for ts in available_months:
                            label = ts.strftime("%b %Y")
                            value = ts.strftime("%Y-%m")
                            key_payload = "||".join([
                                "stringing",
                                proj_code or "",
                                value or "",
                                proj_name,
                            ])
                            if use_modal_ids:
                                key_payload = f"{key_payload}||__modal__"
                            stringing_month_buttons.append(
                                dbc.Button(
                                    label,
                                    id={"type": "proj-resp-open", "key": key_payload},
                                    color="link",
                                    className="p-0 me-1",
                                )
                            )
                        tile_body_children.append(html.Div(stringing_month_buttons, className="mb-2"))
                    else:
                        tile_body_children.append(html.Div("Micro Plan not available.", className="text-muted mb-2"))

                    tile_card = dbc.Card(dbc.CardBody(tile_body_children), className="h-100 shadow-sm")

                    tile_cols.append(
                        dbc.Col(
                            html.Div(
                                tile_card,
                                id=card_id,
                                n_clicks=0,
                                className="project-tile-card",
                                role="button",
                                tabIndex=0,
                            ),
                            xs=12,
                            sm=12,
                            md=6,
                            lg=4,
                            className="mb-3",
                        )
                    )
                body_children = (
                    [dbc.Row(tile_cols, className="g-3")]
                    if tile_cols
                    else [html.Div("No projects available.", className="text-muted")]
                )
                pch_sections.append(
                    dbc.AccordionItem(
                        title=title_component,
                        children=body_children,
                        item_id=f"pch-{_slugify_pch(pch)}",
                        className="pch-section mb-2",
                    )
                )
            if not pch_sections:
                pch_sections = _empty_pch_items("No projects match the current filters.")
            return pch_sections, None, tile_metadata

        # --- Erection mode below ---

        project_list = _ensure_list(projects)
        month_list = _ensure_list(months)
        months_ts = resolve_months(month_list, quick_range)

        if months_ts:
            range_start = pd.Timestamp(min(months_ts)).normalize()
            range_end = (pd.Timestamp(max(months_ts)) + pd.offsets.MonthEnd(0)).normalize()
        else:
            today = pd.Timestamp.today().normalize()
            range_start = today.to_period("M").start_time.normalize()
            range_end = (today + pd.offsets.MonthEnd(0)).normalize()

        df_day = data_selector.select("erection")
        scoped = apply_filters(df_day, project_list, months_ts, [])
        try:
            scope_frames, _, _ = _build_scope_frames(
                "erection",
                project_list=project_list,
                gang_list=[],
                months_value=month_list,
                quick_range=quick_range,
                method_values=None,
            )
            scope_full = scope_frames.get("full", pd.DataFrame()).copy()
        except Exception:
            scope_full = pd.DataFrame()
        export_df, _ = _prepare_erections_completed(
            scoped,
            range_start=range_start,
            range_end=range_end,
            responsibilities_provider=None,
            search_text=None,
        )
        if not isinstance(export_df, pd.DataFrame):
            export_df = pd.DataFrame(columns=["project_name", "location_no", "tower_weight_mt", "daily_prod_mt", "gang_name", "supervisor_name", "section_incharge_name"])

        df_mp = None
        if has_plan_provider.get("erection"):
            df_mp_frame, _, _, _ = _fetch_monthly_plan("erection")
            if isinstance(df_mp_frame, pd.DataFrame):
                df_mp = df_mp_frame
        # Keep an unfiltered copy to test project-level availability (any month)
        mp_all = df_mp.copy() if isinstance(df_mp, pd.DataFrame) else None
        if isinstance(mp_all, pd.DataFrame):
            if "plan_month" in mp_all.columns:
                mp_all["plan_month"] = pd.to_datetime(
                    mp_all["plan_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
                mp_all["completion_month"] = mp_all["plan_month"]
            elif "completion_month" in mp_all.columns:
                mp_all["completion_month"] = pd.to_datetime(
                    mp_all["completion_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            elif "completion_date" in mp_all.columns:
                mp_all["completion_month"] = pd.to_datetime(
                    mp_all["completion_date"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            else:
                mp_all["completion_month"] = pd.NaT
        # Do not block the modal if Micro Plan is unavailable; proceed with empty frame
        if df_mp is None:
            mp = pd.DataFrame(columns=[
                "project_name", "project_key", "location_no", "entity_type", "entity_name",
                "tower_weight", "pch", "plan_month", "completion_month"
            ])
        else:
            mp = df_mp.copy()

        if "plan_month" in mp.columns:
            mp["plan_month"] = pd.to_datetime(
                mp["plan_month"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            mp["completion_month"] = mp["plan_month"]
        elif "completion_month" in mp.columns:
            mp["completion_month"] = pd.to_datetime(
                mp["completion_month"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        elif "completion_date" in mp.columns:
            mp["completion_month"] = pd.to_datetime(
                mp["completion_date"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        else:
            mp["completion_month"] = pd.NaT

        if months_ts:
            mp = mp[mp["completion_month"].isin(months_ts)].copy()
        if project_list:
            import re as _re
            proj_names_lc = set(str(p).strip().lower() for p in project_list)
            proj_names_compact = set(_re.sub(r"[^a-z0-9]", "", s) for s in proj_names_lc)
            name_lc = (mp.get("project_name", pd.Series([""] * len(mp), index=mp.index)).astype(str).str.strip().str.lower())
            key_lc = (mp.get("project_key", pd.Series([""] * len(mp), index=mp.index)).astype(str).str.strip().str.lower())
            name_compact = name_lc.str.replace(r"[^a-z0-9]", "", regex=True)
            key_compact = key_lc.str.replace(r"[^a-z0-9]", "", regex=True)
            mask_mp = (
                name_lc.isin(proj_names_lc) | key_lc.isin(proj_names_lc) |
                name_compact.isin(proj_names_compact) | key_compact.isin(proj_names_compact)
            )
            mp = mp[mask_mp].copy()
            # Apply same project filter to unfiltered copy used for availability checks
            if isinstance(mp_all, pd.DataFrame) and not mp_all.empty:
                name_lc_all = (mp_all.get("project_name", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                key_lc_all = (mp_all.get("project_key", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                name_compact_all = name_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                key_compact_all = key_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                mask_all = (
                    name_lc_all.isin(proj_names_lc) | key_lc_all.isin(proj_names_lc) |
                    name_compact_all.isin(proj_names_compact) | key_compact_all.isin(proj_names_compact)
                )
                mp_all = mp_all[mask_all].copy()

        # Normalized helper columns
        mp["location_no_norm"] = (mp.get("location_no", pd.Series([""] * len(mp), index=mp.index)).map(_normalize_location))
        mp["project_name_display"] = (mp.get("project_name", pd.Series([""] * len(mp), index=mp.index)).astype(str))
        mp["pch_display"] = (mp.get("pch", pd.Series([""] * len(mp), index=mp.index)).astype(str))
        mp["_tw_"] = pd.to_numeric(mp.get("tower_weight", 0.0), errors="coerce").fillna(0.0)

        # Project-level planned
        planned_mt = mp.groupby(["pch_display", "project_name_display"], dropna=False)["_tw_"].sum()
        planned_nos = (
            mp.dropna(subset=["location_no_norm"])\
              .drop_duplicates(["pch_display", "project_name_display", "location_no_norm"])\
              .groupby(["pch_display", "project_name_display"]).size()
        )

        # Delivered aggregates and meta (by project and location)
        ed = export_df.copy()
        ed["project_name_display"] = ed.get("project_name", "").astype(str)
        ed["location_no_norm"] = ed.get("location_no", "").map(_normalize_location)
        delivered_mt = ed.groupby(["project_name_display"]) ["tower_weight_mt"].sum()
        delivered_nos = (
            ed.dropna(subset=["location_no_norm"])\
              .drop_duplicates(["project_name_display", "location_no_norm"])\
              .groupby(["project_name_display"]).size()
        )
        meta_cols = ["daily_prod_mt", "gang_name", "supervisor_name", "section_incharge_name", "start_date"]
        loc_meta = (
            ed.sort_values("completion_date").drop_duplicates("location_no_norm", keep="last")[
                ["location_no_norm", *[c for c in meta_cols if c in ed.columns]]
            ] if not ed.empty else pd.DataFrame(columns=["location_no_norm", *meta_cols])
        ).set_index("location_no_norm") if not ed.empty else pd.DataFrame()

        # Project meta (regional/project manager, planning engineer) + PCH mapping from Project Details
        try:
            info_df = project_info_provider() if callable(project_info_provider) else None
        except Exception:
            info_df = None
        # Ensure PCH normalizer default exists before we start building lookup maps.
        if "_normalize_pch" not in locals():
            def _normalize_pch(v):
                return str(v or "").strip()
            _PCH_ORDER = ()

        info_name_to_pch: dict[str, str] = {}
        info_code_to_pch: dict[str, str] = {}
        if isinstance(info_df, pd.DataFrame) and not info_df.empty:
            info = info_df.copy()
            info["project_name_display"] = info.get("Project Name", info.get("project_name", "")).astype(str)
            # Prepare normalized key for robust matching (case-insensitive, trimmed)
            info["project_name_norm"] = info["project_name_display"].map(_normalize_lower)
            # Find a PCH column in a forgiving way
            pch_col = None
            for cand in ("PCH", "pch", "PCH Name", "PCHName", "pch_name"):
                if cand in info.columns:
                    pch_col = cand
                    break
            if pch_col is None:
                info["pch"] = ""
                pch_col = "pch"
            # Build compact key map for robust project lookup across datasets (e.g., 'TA418' vs 'TA 418')
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                info_key_map: dict[str, int] = {}
                name_keys = info["project_name_display"].astype(str).map(_compact_code)
                for idx, key in zip(info.index, name_keys):
                    if key and key not in info_key_map:
                        info_key_map[key] = idx
                for code_col in ("project_code", "Project Code"):
                    if code_col in info.columns:
                        code_keys = info[code_col].astype(str).map(_compact_code)
                        for idx, key in zip(info.index, code_keys):
                            if key and key not in info_key_map:
                                info_key_map[key] = idx
            except Exception:
                info_key_map = {}

            if pch_col in info.columns:
                info_name_to_pch = {
                    _normalize_lower(str(row.get("project_name_display", ""))): _normalize_pch(row.get(pch_col, ""))
                    for _, row in info.iterrows()
                    if str(row.get("project_name_display", "")).strip()
                }
                if "project_code" in info.columns:
                    try:
                        info_code_to_pch = {
                            re.sub(r"[^a-z0-9]", "", str(row.get("project_code", "")).strip().lower()): _normalize_pch(row.get(pch_col, ""))
                            for _, row in info.iterrows()
                            if str(row.get("project_code", "")).strip()
                        }
                    except Exception:
                        info_code_to_pch = {}
        else:
            info = pd.DataFrame(columns=["project_name_display", "project_name_norm", "pch", "regional_mgr", "project_mgr", "planning_eng"])
            info_key_map = {}

        # Build hierarchy: PCH -> Projects -> Locations
        # Ensure we always have a PCH value; fall back to project-info mapping if blank
        proj_info_pch = {}
        if not info.empty and "pch" in info.columns:
            proj_info_pch = dict(zip(info["project_name_display"], info["pch"].astype(str)))

        # Import PCH normalizer to canonicalize labels for grouping and display
        try:
            from .pch_normalizer import normalize_pch as _normalize_pch, CANONICAL_PCH_PRIMARY as _PCH_ORDER
        except Exception:
            def _normalize_pch(v):
                return str(v or "").strip()
            _PCH_ORDER = ()

        # Aggregate per (normalized PCH, project) to avoid duplicates when Micro Plan has variant PCHs
        aggregated = {}
        aggregated_by_proj_key: dict[str, tuple[str, str]] = {}
        for (mp_pch, proj), mt in planned_mt.items():
            nos_planned = int(planned_nos.get((mp_pch, proj), 0)) if hasattr(planned_nos, 'get') else 0
            # Robust lookup of Project Details row using normalized name; if not found, try compact code match
            proj_norm = _normalize_lower(proj)
            # Use normalized-name match, then compact-key map fallback
            meta = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
            if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                try:
                    import re as _re
                    def _compact_code(s: str) -> str:
                        return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                    target_key = _compact_code(str(proj))
                    if target_key and target_key in info_key_map:
                        meta = info.loc[[info_key_map[target_key]]]
                except Exception:
                    pass
            raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
            # Determine PCH solely from Project Details; if unrecognized, keep original
            pch_label = (_normalize_pch(raw_pch) or str(raw_pch or "").strip())
            # Derive display heading and identity key using code when available
            try:
                proj_code = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                    meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                )) if isinstance(meta, pd.DataFrame) else ""
            except Exception:
                proj_code = ""
            proj_display_name = str(meta.get("project_name_display", pd.Series([proj])).iloc[0]) if isinstance(meta, pd.DataFrame) and not meta.empty else str(proj)
            proj_display = f"{proj_code} : {proj_display_name}".strip(" :") if proj_code else proj_display_name
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                proj_key = _compact_code(proj_code) or _compact_code(proj_display_name) or _compact_code(proj)
            except Exception:
                proj_key = proj_norm
            key = (pch_label, proj_display)
            if key not in aggregated:
                mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
                nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
                aggregated[key] = {
                    "pch": pch_label,
                    "project_name": proj_display,
                    "project_code": proj_code,
                    "planned_mt": 0.0,
                    "delivered_mt": mt_del,
                    "planned_nos": 0,
                    "delivered_nos": nos_del,
                    "regional_mgr": (meta["regional_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns else ""),
                    "project_mgr": (meta["project_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns else ""),
                    "planning_eng": (meta["planning_eng"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns else ""),
                }
            aggregated[key]["planned_mt"] += float(mt)
            aggregated[key]["planned_nos"] += nos_planned
            # Track by compact project key to prevent duplicates across sources
            try:
                aggregated_by_proj_key[proj_key] = key
            except Exception:
                pass

        # Also include projects that only have delivered data (no Micro Plan rows)
        try:
            delivered_projects = list(getattr(delivered_mt, 'index', []))
        except Exception:
            delivered_projects = []
        for proj in map(lambda x: str(x), delivered_projects):
            if not proj or not str(proj).strip():
                continue
            proj_norm = _normalize_lower(proj)
            # If this project already exists from the planned aggregation, update delivered values instead of duplicating
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                # Prefer code if we can resolve it from info
                meta_lookup = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
                proj_code_lookup = (meta_lookup.get("project_code").iloc[0] if (isinstance(meta_lookup, pd.DataFrame) and not meta_lookup.empty and "project_code" in meta_lookup.columns) else (
                    meta_lookup.get("Project Code").iloc[0] if (isinstance(meta_lookup, pd.DataFrame) and not meta_lookup.empty and "Project Code" in meta_lookup.columns) else ""
                )) if isinstance(meta_lookup, pd.DataFrame) else ""
                proj_key = _compact_code(proj_code_lookup) or _compact_code(proj)
            except Exception:
                proj_key = proj_norm
            if proj_key in aggregated_by_proj_key:
                try:
                    existing_key = aggregated_by_proj_key[proj_key]
                    mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
                    nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
                    aggregated[existing_key]["delivered_mt"] = mt_del
                    aggregated[existing_key]["delivered_nos"] = nos_del
                    continue
                except Exception:
                    # fallback to normal add path
                    pass
            meta = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
            if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                # Try compact-key lookup
                if proj_key and proj_key in info_key_map:
                    meta = info.loc[[info_key_map[proj_key]]]
            raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
            pch_label = (
                _normalize_pch(raw_pch)
                or str(raw_pch or "").strip()
            )
            # Build display and project code for heading
            try:
                proj_code2 = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                    meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                )) if isinstance(meta, pd.DataFrame) else ""
            except Exception:
                proj_code2 = ""
            proj_display_name2 = str(meta.get("project_name_display", pd.Series([proj])).iloc[0]) if isinstance(meta, pd.DataFrame) and not meta.empty else str(proj)
            resolved_code2 = (
                _extract_project_code(proj_code2)
                or _extract_project_code(proj_display_name2)
                or _extract_project_code(proj)
            )
            if not resolved_code2:
                resolved_code2 = proj_code2
            proj_display2 = f"{resolved_code2} : {proj_display_name2}".strip(" :") if resolved_code2 else proj_display_name2
            key = (pch_label, proj_display2)
            if key in aggregated:
                continue
            mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, 'get') else 0.0
            nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, 'get') else 0
            aggregated[key] = {
                "pch": pch_label,
                "project_name": proj_display2,
                "project_code": resolved_code2 or proj_code2,
                "code": resolved_code2 or proj_code2,
                "planned_mt": 0.0,
                "delivered_mt": mt_del,
                "planned_nos": 0,
                "delivered_nos": nos_del,
                "regional_mgr": (meta["regional_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns else ""),
                "project_mgr": (meta["project_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns else ""),
                "planning_eng": (meta["planning_eng"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns else ""),
            }
            try:
                aggregated_by_proj_key[proj_key] = key
            except Exception:
                pass

        # Finally, include any projects present only in Project Details (no delivered and no MP)
        try:
            # Use selected projects filter if provided; otherwise consider all info rows
            info_iter = info.copy()
            if project_list:
                pl = set(str(p).strip().lower() for p in project_list)
                info_iter = info_iter[info_iter["project_name_display"].astype(str).str.strip().str.lower().isin(pl)]
        except Exception:
            info_iter = info
        for _, meta_row in info_iter.iterrows():
            try:
                import re as _re
                def _compact_code(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
                proj = str(meta_row.get("project_name_display", "")).strip()
                code_raw = str(meta_row.get("project_code", meta_row.get("Project Code", "")))
                resolved_code = _extract_project_code(code_raw) or _extract_project_code(proj)
                proj_key = _compact_code(resolved_code or code_raw) or _compact_code(proj)
                if not proj or proj_key in aggregated_by_proj_key:
                    continue
                raw_pch = str(meta_row.get(pch_col, "")) if pch_col in meta_row.index else ""
                pch_label = (_normalize_pch(raw_pch) or str(raw_pch or "").strip())
                proj_display = f"{resolved_code} : {proj}".strip(" :") if resolved_code else proj
                key = (pch_label, proj_display)
                aggregated[key] = {
                    "pch": pch_label,
                    "project_name": proj_display,
                    "project_code": resolved_code or code_raw,
                    "code": resolved_code or code_raw,
                    "planned_mt": 0.0,
                    "delivered_mt": 0.0,
                    "planned_nos": 0,
                    "delivered_nos": 0,
                    "regional_mgr": str(meta_row.get("regional_mgr", "")),
                    "project_mgr": str(meta_row.get("project_mgr", "")),
                    "planning_eng": str(meta_row.get("planning_eng", "")),
                }
                aggregated_by_proj_key[proj_key] = key
            except Exception:
                continue

        # Also include projects seen in the daily scope even if they have no completion rows/plan data
        try:
            scoped_projects = []
            if isinstance(scoped, pd.DataFrame) and not scoped.empty:
                proj_col = "project_name" if "project_name" in scoped.columns else (
                    "project" if "project" in scoped.columns else None
                )
                if proj_col:
                    scoped_projects = (
                        scoped[proj_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .tolist()
                    )
        except Exception:
            scoped_projects = []

        if scoped_projects:
            try:
                import re as _re

                def _compact_code_fallback(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())
            except Exception:

                def _compact_code_fallback(s: str) -> str:
                    return str(s or "").strip().lower().replace(" ", "")

            for proj in sorted({p for p in scoped_projects if p}):
                proj_norm = _normalize_lower(proj)
                proj_key = _compact_code_fallback(proj)
                if not proj_key or proj_key in aggregated_by_proj_key:
                    continue
                meta = info[info.get("project_name_norm", "").astype(str) == proj_norm].iloc[:1] if not info.empty else pd.DataFrame()
                if (not isinstance(meta, pd.DataFrame)) or meta.empty:
                    if proj_key and proj_key in info_key_map:
                        meta = info.loc[[info_key_map[proj_key]]]
                raw_pch = (meta[pch_col].iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and pch_col in meta.columns) else "")
                pch_label = (_normalize_pch(raw_pch) or str(raw_pch or "").strip())
                try:
                    proj_code = (meta.get("project_code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "project_code" in meta.columns) else (
                        meta.get("Project Code").iloc[0] if (isinstance(meta, pd.DataFrame) and not meta.empty and "Project Code" in meta.columns) else ""
                    )) if isinstance(meta, pd.DataFrame) else ""
                except Exception:
                    proj_code = ""
                proj_display_name = str(meta.get("project_name_display", pd.Series([proj])).iloc[0]) if isinstance(meta, pd.DataFrame) and not meta.empty else str(proj)
                resolved_code = _extract_project_code(proj_code) or _extract_project_code(proj_display_name) or _extract_project_code(proj)
                if not resolved_code:
                    resolved_code = proj_code or proj
                proj_display = f"{resolved_code} : {proj_display_name}".strip(" :") if resolved_code else proj_display_name
                key = (pch_label, proj_display)
                if key in aggregated:
                    continue
                mt_del = float(delivered_mt.get(proj, 0.0)) if hasattr(delivered_mt, "get") else 0.0
                nos_del = int(delivered_nos.get(proj, 0)) if hasattr(delivered_nos, "get") else 0
                aggregated[key] = {
                    "pch": pch_label,
                    "project_name": proj_display,
                    "project_code": resolved_code or proj_code,
                    "code": resolved_code or proj_code,
                    "planned_mt": 0.0,
                    "delivered_mt": mt_del,
                    "planned_nos": 0,
                    "delivered_nos": nos_del,
                    "regional_mgr": (meta["regional_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "regional_mgr" in meta.columns else ""),
                    "project_mgr": (meta["project_mgr"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "project_mgr" in meta.columns else ""),
                    "planning_eng": (meta["planning_eng"].iloc[0] if isinstance(meta, pd.DataFrame) and not meta.empty and "planning_eng" in meta.columns else ""),
                }
                aggregated_by_proj_key[proj_key] = key

        # Build rows grouped by normalized PCH
        projects_rows: dict[str, list[dict]] = {}
        for (_pch_label, _proj), rec in aggregated.items():
            rec["planned_mt"] = round(float(rec["planned_mt"]), 1)
            rec["delivered_mt"] = round(float(rec["delivered_mt"]), 1)
            projects_rows.setdefault(_pch_label, []).append(rec)

        current_month_ts = range_end.to_period("M").to_timestamp()

        def _compact_key(value: str) -> str:
            text = str(value or "").strip().lower()
            if not text:
                return ""
            return re.sub(r"[^a-z0-9]", "", text)

        def _build_metric_lookup(source: dict[str, float] | None) -> tuple[dict[str, float], dict[str, float]]:
            norm_map: dict[str, float] = {}
            compact_map: dict[str, float] = {}
            if not source:
                return norm_map, compact_map
            for proj_name, raw_value in source.items():
                text = str(proj_name or "").strip()
                if not text:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if pd.isna(value):
                    continue
                norm_key = _normalize_lower(text)
                if norm_key and norm_key not in norm_map:
                    norm_map[norm_key] = value
                compact_key = _compact_key(text)
                if compact_key and compact_key not in compact_map:
                    compact_map[compact_key] = value
            return norm_map, compact_map

        prod_current_norm_map: dict[str, float] = {}
        prod_current_compact_map: dict[str, float] = {}
        prod_history_norm_map: dict[str, float] = {}
        prod_history_compact_map: dict[str, float] = {}
        towers_current_norm_map: dict[str, int] = {}
        towers_current_compact_map: dict[str, int] = {}
        towers_planned_norm_map: dict[str, int] = {}
        towers_planned_compact_map: dict[str, int] = {}

        if isinstance(df_day, pd.DataFrame) and not df_day.empty:
            day_filtered = df_day.copy()
            if project_list and "project_name" in day_filtered.columns:
                project_filter_values = [str(p).strip() for p in project_list if str(p).strip()]
                if project_filter_values:
                    day_filtered = day_filtered[
                        day_filtered["project_name"].astype(str).str.strip().isin(project_filter_values)
                    ]
            if not day_filtered.empty and {"month", "daily_prod_mt", "project_name"}.issubset(day_filtered.columns):
                day_filtered = day_filtered.copy()
                day_filtered["month"] = pd.to_datetime(day_filtered["month"], errors="coerce")
                day_filtered["project_name"] = day_filtered["project_name"].astype(str).str.strip()
                day_filtered = day_filtered.dropna(subset=["month", "daily_prod_mt", "project_name"])

                current_scope = day_filtered[day_filtered["month"] == current_month_ts]
                if not current_scope.empty:
                    current_payload = {
                        "kind": "erection-current-baseline",
                        "projects": sorted(project_list),
                        "months": [ts.isoformat() for ts in months_ts],
                        "month": current_month_ts.isoformat(),
                        "rows": int(len(current_scope.index)),
                        "version": int(df_day.attrs.get("_appdata_version", 0)),
                    }
                    current_token = f"erection-baseline::{_hash_cache_payload(current_payload)}"

                    def _compute_current_prod() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                        return compute_project_baseline_maps_for(current_scope, "daily_prod_mt")

                    prod_current_raw, _ = _cached_global_result(
                        current_token,
                        _compute_current_prod,
                        clone=_clone_baseline_result,
                    )
                    prod_current_norm_map, prod_current_compact_map = _build_metric_lookup(prod_current_raw)

                history_scope = day_filtered[day_filtered["month"] < current_month_ts]
                if not history_scope.empty:
                    history_payload = {
                        "kind": "erection-history-baseline",
                        "projects": sorted(project_list),
                        "months": [ts.isoformat() for ts in months_ts],
                        "month": current_month_ts.isoformat(),
                        "rows": int(len(history_scope.index)),
                        "version": int(df_day.attrs.get("_appdata_version", 0)),
                    }
                    history_token = f"erection-baseline::{_hash_cache_payload(history_payload)}"

                    def _compute_history_prod() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
                        return compute_project_baseline_maps_for(history_scope, "daily_prod_mt")

                    prod_history_raw, _ = _cached_global_result(
                        history_token,
                        _compute_history_prod,
                        clone=_clone_baseline_result,
                    )
                    prod_history_norm_map, prod_history_compact_map = _build_metric_lookup(prod_history_raw)

        if isinstance(ed, pd.DataFrame) and not ed.empty and "completion_date" in ed.columns:
            ed_for_towers = ed.copy()
            ed_for_towers["completion_month"] = pd.to_datetime(ed_for_towers["completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            ed_for_towers = ed_for_towers.dropna(subset=["completion_month"])
            ed_for_towers["project_name_display"] = ed_for_towers["project_name_display"].astype(str).str.strip()
            ed_current = ed_for_towers[ed_for_towers["completion_month"] == current_month_ts]
            if not ed_current.empty and "location_no_norm" in ed_current.columns:
                towers_series = (
                    ed_current.dropna(subset=["location_no_norm"])
                              .drop_duplicates(["project_name_display", "location_no_norm"])
                              .groupby("project_name_display")
                              .size()
                )
                for proj_name, count in towers_series.items():
                    text = str(proj_name or "").strip()
                    if not text:
                        continue
                    norm_key = _normalize_lower(text)
                    if norm_key and norm_key not in towers_current_norm_map:
                        towers_current_norm_map[norm_key] = int(count)
                    compact_key = _compact_key(text)
                    if compact_key and compact_key not in towers_current_compact_map:
                        towers_current_compact_map[compact_key] = int(count)

        if isinstance(mp, pd.DataFrame) and not mp.empty and "completion_month" in mp.columns:
            mp_current = mp[mp["completion_month"] == current_month_ts].copy()
            if not mp_current.empty and "location_no_norm" in mp_current.columns:
                planned_series = (
                    mp_current.dropna(subset=["location_no_norm"])
                              .drop_duplicates(["project_name_display", "location_no_norm"])
                              .groupby("project_name_display")
                              .size()
                )
                for proj_name, count in planned_series.items():
                    text = str(proj_name or "").strip()
                    if not text:
                        continue
                    norm_key = _normalize_lower(text)
                    if norm_key and norm_key not in towers_planned_norm_map:
                        towers_planned_norm_map[norm_key] = int(count)
                    compact_key = _compact_key(text)
                    if compact_key and compact_key not in towers_planned_compact_map:
                        towers_planned_compact_map[compact_key] = int(count)

        def _project_lookup_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
            return _project_code_key_sets(row)

        def _lookup_with_key(
            norm_keys: list[str],
            compact_keys: list[str],
            norm_map: dict[str, float],
            compact_map: dict[str, float],
        ) -> tuple[float | None, str | None]:
            for key in norm_keys:
                if key in norm_map:
                    return norm_map[key], key
            for key in compact_keys:
                if key in compact_map:
                    return compact_map[key], key
            return None, None

        def _project_locations(project_name: str) -> list[dict]:
            # Planned per location (tower weight) for the project
            # project_name in the aggregated rows may be "CODE : NAME"; match using NAME part
            pname = str(project_name)
            base_name = pname.split(" : ", 1)[1] if " : " in pname else pname
            mp_proj = mp[mp["project_name_display"].astype(str) == str(base_name)].copy()
            planned_loc = mp_proj.groupby("location_no_norm")["_tw_"].sum().rename("planned_mt") if not mp_proj.empty else pd.Series(dtype=float)
            ed_proj = ed[ed["project_name_display"].astype(str) == str(base_name)].copy()
            delivered_loc = ed_proj.groupby("location_no_norm")["tower_weight_mt"].sum().rename("delivered_mt") if not ed_proj.empty else pd.Series(dtype=float)
            keys = set(planned_loc.index.tolist()) | set(delivered_loc.index.tolist())
            out = []
            for loc in sorted(k for k in keys if k):
                meta = loc_meta.loc[loc] if (isinstance(loc_meta, pd.DataFrame) and (loc in getattr(loc_meta, 'index', []))) else None
                out.append({
                    "location_no": loc,
                    "planned_mt": round(float(planned_loc.get(loc, 0.0) or 0.0), 1),
                    "delivered_mt": round(float(delivered_loc.get(loc, 0.0) or 0.0), 1),
                    "daily_prod_mt": (float(meta["daily_prod_mt"]) if (isinstance(meta, pd.Series) and "daily_prod_mt" in meta) else None),
                    "gang_name": (str(meta["gang_name"]) if (isinstance(meta, pd.Series) and "gang_name" in meta) else ""),
                    "supervisor_name": (str(meta["supervisor_name"]) if (isinstance(meta, pd.Series) and "supervisor_name" in meta) else ""),
                    "section_incharge_name": (str(meta["section_incharge_name"]) if (isinstance(meta, pd.Series) and "section_incharge_name" in meta) else ""),
                })
            return out

        # Order PCH sections: canonical order first, then alphabetical; keep empty last
        def _pch_sort_key(name: str) -> tuple[int, str]:
            if not str(name or "").strip():
                return (2, "")
            try:
                idx = list(_PCH_ORDER).index(name)
                return (0, f"{idx:03d}")
            except ValueError:
                return (1, str(name))

        pch_sections = []
        for pch in sorted(projects_rows.keys(), key=_pch_sort_key):
            rows = projects_rows[pch]

            project_count = len(rows)
            project_codes: list[str] = []
            prod_current_values: list[float] = []
            prod_history_values: list[float] = []
            towers_delivered_total = 0
            towers_planned_total = 0
            towers_delivered_keys: set[str] = set()
            towers_planned_keys: set[str] = set()

            for r in sorted(rows, key=lambda x: str(x["project_name"])):
                norm_keys, compact_keys = _project_lookup_keys(r)
                code_value = _resolve_project_code(r)
                if code_value and code_value not in project_codes:
                    project_codes.append(code_value)

                prod_current_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map
                )
                if prod_current_value is not None:
                    prod_current_values.append(prod_current_value)

                prod_history_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_history_norm_map, prod_history_compact_map
                )
                if prod_history_value is not None:
                    prod_history_values.append(prod_history_value)

                tower_value, tower_key = _lookup_with_key(
                    norm_keys, compact_keys, towers_current_norm_map, towers_current_compact_map
                )
                if tower_value is not None and tower_key and tower_key not in towers_delivered_keys:
                    towers_delivered_keys.add(tower_key)
                    towers_delivered_total += int(tower_value)

                tower_plan_value, tower_plan_key = _lookup_with_key(
                    norm_keys, compact_keys, towers_planned_norm_map, towers_planned_compact_map
                )
                if tower_plan_value is not None and tower_plan_key and tower_plan_key not in towers_planned_keys:
                    towers_planned_keys.add(tower_plan_key)
                    towers_planned_total += int(tower_plan_value)

            prod_current_avg = (
                float(sum(prod_current_values) / len(prod_current_values)) if prod_current_values else None
            )
            prod_history_avg = (
                float(sum(prod_history_values) / len(prod_history_values)) if prod_history_values else None
            )
            fmt_prod_current = f"{prod_current_avg:.2f}" if prod_current_avg is not None else "\u2014"
            fmt_prod_history = f"{prod_history_avg:.2f}" if prod_history_avg is not None else "\u2014"
            projects_label = (
                f"Projects: {', '.join(project_codes)}" if project_codes else f"Projects: {project_count}"
            )
            towers_delivered_label = int(towers_delivered_total)
            towers_planned_label = int(towers_planned_total)
            # Derive gangs and lost units per PCH (erection)
            gangs_total = 0
            lost_units_total = 0.0
            try:
                unit_short = "MT"
                metric_col = "daily_prod_mt"
                sub = _filter_scope_by_projects(scope_full, rows)
                if sub.empty:
                    sub = _filter_scope_by_projects(scoped, rows)

                if not sub.empty:
                    if "gang_name" in sub.columns:
                        gangs_total = (
                            sub["gang_name"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                        )
                    try:
                        overall_map, monthly_map = compute_project_baseline_maps(sub)
                    except Exception:
                        overall_map, monthly_map = {}, {}
                    proj_col_for_gang = _detect_project_column(sub)
                    if proj_col_for_gang:
                        gang_project_map = (
                            sub[["gang_name", proj_col_for_gang]]
                            .dropna()
                            .astype(str)
                            .rename(columns={proj_col_for_gang: "project_name"})
                            .drop_duplicates(subset=["gang_name", "project_name"], keep="last")
                            .set_index("gang_name")["project_name"]
                            .to_dict()
                        )
                    else:
                        gang_project_map = {}
                    month_filter = _month_filter_for_frame(sub)
                    for gang, gdf in sub.groupby("gang_name"):
                        if gdf.empty:
                            continue
                        proj_for_gang = gang_project_map.get(str(gang))
                        ov = overall_map.get(proj_for_gang)
                        mm = monthly_map.get(proj_for_gang, {})
                        try:
                            _idle, _baseline, loss_val, _deliv, _pot = calc_idle_and_loss(
                                gdf,
                                loss_max_gap_days=config.loss_max_gap_days,
                                baseline_mt_per_day=ov,
                                baseline_by_month=mm,
                                idle_intervals=idle_table_erection,
                                allowed_months=month_filter,
                            )
                            if pd.notna(loss_val):
                                lost_units_total += float(loss_val)
                        except Exception:
                            pass
            except Exception:
                gangs_total = gangs_total or 0
                lost_units_total = lost_units_total or 0.0

            towers_balance_label = max(towers_planned_label - towers_delivered_label, 0)
            pill_components = [
                                ("projects", dbc.Button(
                                    projects_label,
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "projects"},
                                    className="pch-pill pch-pill-projects me-2 mb-1", color="link"
                                )),
                                ("productivity", dbc.Button(
                                    f"Productivity (Month/Overall): {fmt_prod_current} / {fmt_prod_history} MT/day",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "productivity"},
                                    className="pch-pill pch-pill-prod-month me-2 mb-1", color="link"
                                )),
                                ("totals", dbc.Button(
                                    f"F/S Total Planned / Done / Balance: {towers_planned_label} / {towers_delivered_label} / {towers_balance_label}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "totals"},
                                    className="pch-pill pch-pill-towers me-2 mb-1", color="link"
                                )),
                                ("gangs", dbc.Button(
                                    f"Gangs: {int(gangs_total):,}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "gangs"},
                                    className="pch-pill pch-pill-prod-overall me-2 mb-1", color="link", n_clicks=0, 
                                )),
                                ("loss", dbc.Button(
                                    f"Lost Units: {lost_units_total:.1f} MT",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "loss"},
                                    className="pch-pill pch-pill-loss me-2 mb-1", color="link", n_clicks=0, 
                                )),
                            ]

            header_pills = _filter_pch_header_pills(pill_components, pill_focus)

            header = dbc.Row(
                [
                    dbc.Col(html.H6(str(pch), className="mb-0"), md=3),
                    dbc.Col(
                        html.Div(header_pills, className="pch-pill-group justify-content-md-end"),
                        md=9,
                    ),
                ],
                className="pch-header align-items-center py-2",
            )

            # Nested tiles for projects within this PCH
            project_items = []  # legacy; no longer used
            tile_cols = []
            for r in sorted(rows, key=lambda x: str(x["project_name"])):
                proj_name = str(r["project_name"]).strip()
                display_code = _resolve_project_code(r)
                # (legacy header removed)
                # Detect if Micro Plan rows exist for this project using robust name/code matching
                import re as _re
                # proj_name may be in format "CODE : NAME"; split for robust matching
                full_name = str(proj_name).strip()
                if " : " in full_name:
                    code_part, name_part = full_name.split(" : ", 1)
                else:
                    code_part, name_part = "", full_name
                sel_name = name_part.strip().lower()
                sel_code = _re.sub(r"[^a-z0-9]", "", code_part.strip().lower()) if code_part else ""
                sel_compact = _re.sub(r"[^a-z0-9]", "", sel_name)

                def _has_project(frame: pd.DataFrame | None) -> bool:
                    try:
                        if not isinstance(frame, pd.DataFrame) or frame.empty:
                            return False
                        name_lc = (frame.get("project_name", pd.Series([""] * len(frame), index=frame.index))
                                   .astype(str).str.strip().str.lower())
                        key_lc = (frame.get("project_key", pd.Series([""] * len(frame), index=frame.index))
                                  .astype(str).str.strip().str.lower())
                        name_compact = name_lc.str.replace(r"[^a-z0-9]", "", regex=True)
                        key_compact = key_lc.str.replace(r"[^a-z0-9]", "", regex=True)
                        mask = (
                            (name_lc == sel_name) | (key_lc == sel_name) |
                            (name_compact == sel_compact) | (key_compact == sel_compact) |
                            ((sel_code != "") & (key_compact == sel_code))
                        )
                        return bool(mask.any())
                    except Exception:
                        return False

                has_mp = _has_project(mp)
                has_mp_any = _has_project(mp_all)

                # Build Section Incharge -> Supervisor summary (planned/delivered) using Micro Plan responsibilities first
                def _section_supervisor_summary(project: str):
                    mp_proj = mp[mp["project_name_display"].astype(str) == str(project)].copy()
                    # normalize labels and fields
                    std_map = {
                        "gangs": "Gang", "gang": "Gang",
                        "section incharges": "Section Incharge", "section incharge": "Section Incharge", "section in-charge": "Section Incharge",
                        "supervisors": "Supervisor", "supervisor": "Supervisor",
                    }
                    et = mp_proj.get("entity_type", "").astype(str).str.lower()
                    mp_proj["entity_type_std"] = et.map(lambda v: std_map.get(v, v.title()))
                    mp_proj["location_no_norm"] = mp_proj.get("location_no", "").map(_normalize_location)
                    mp_proj["tower_weight_val"] = pd.to_numeric(mp_proj.get("tower_weight", 0.0), errors="coerce").fillna(0.0)

                    # completion markers
                    completed: set[tuple[str, str]] = set()
                    try:
                        if callable(responsibilities_completion_provider):
                            completed = set(responsibilities_completion_provider())
                    except Exception:
                        completed = set()
                    proj_lc = str(project).strip().lower()

                    # collapse to location granularity to avoid double-counting
                    # section map per location
                    sec_map = (
                        mp_proj[mp_proj["entity_type_std"] == "Section Incharge"][
                            ["location_no_norm", "entity_name"]
                        ]
                        .dropna()
                        .drop_duplicates("location_no_norm", keep="last")
                        .rename(columns={"entity_name": "section"})
                    )
                    sup_map = (
                        mp_proj[mp_proj["entity_type_std"] == "Supervisor"][
                            ["location_no_norm", "entity_name"]
                        ]
                        .dropna()
                        .drop_duplicates("location_no_norm", keep="last")
                        .rename(columns={"entity_name": "supervisor"})
                    )

                    loc = (
                        mp_proj.groupby("location_no_norm", as_index=False)["tower_weight_val"].max()
                    )
                    loc["is_completed"] = [
                        (proj_lc, _normalize_lower(loc_id)) in completed for loc_id in loc["location_no_norm"]
                    ]
                    loc = loc.merge(sec_map, on="location_no_norm", how="left").merge(sup_map, on="location_no_norm", how="left")
                    loc["section"] = loc["section"].fillna("Unassigned").astype(str)
                    loc["supervisor"] = loc["supervisor"].fillna("Unassigned").astype(str)
                    loc["delivered_mt_val"] = np.where(loc["is_completed"], loc["tower_weight_val"], 0.0)
                    loc["delivered_n"] = np.where(loc["is_completed"], 1, 0)

                    # section aggregates
                    sec_g = loc.groupby("section", as_index=False).agg(
                        planned_nos=("location_no_norm", "nunique"),
                        planned_mt=("tower_weight_val", "sum"),
                        delivered_nos=("delivered_n", "sum"),
                        delivered_mt=("delivered_mt_val", "sum"),
                    )
                    result = {
                        str(row["section"]): {
                            "planned_nos": int(row["planned_nos"]),
                            "planned_mt": float(row["planned_mt"]),
                            "delivered_nos": int(row["delivered_nos"]),
                            "delivered_mt": float(row["delivered_mt"]),
                            "supervisors": [],
                        }
                        for _, row in sec_g.iterrows()
                    }

                    # supervisor aggregates within each section
                    for sec_name, sub in loc.groupby("section"):
                        sup_g = sub.groupby("supervisor", as_index=False).agg(
                            planned_nos=("location_no_norm", "nunique"),
                            planned_mt=("tower_weight_val", "sum"),
                            delivered_nos=("delivered_n", "sum"),
                            delivered_mt=("delivered_mt_val", "sum"),
                        )
                        for _, row in sup_g.iterrows():
                            result.setdefault(str(sec_name), {"planned_nos":0,"planned_mt":0.0,"delivered_nos":0,"delivered_mt":0.0,"supervisors":[]})
                            result[str(sec_name)]["supervisors"].append({
                                "name": str(row["supervisor"]),
                                "planned_nos": int(row["planned_nos"]),
                                "planned_mt": float(row["planned_mt"]),
                                "delivered_nos": int(row["delivered_nos"]),
                                "delivered_mt": float(row["delivered_mt"]),
                            })
                    return result

                sections_children = []
                if has_mp:
                    summary = _section_supervisor_summary(r["project_name"])
                    for sec_name in sorted(summary.keys()):
                        sec_data = summary[sec_name]
                        sup_children = []
                        for sup_item in sorted(sec_data["supervisors"], key=lambda x: x["name"]):
                            sup_children.append(html.Div([
                                html.Span(sup_item["name"], className="me-2 fw-semibold"),
                                dbc.Badge(f"Nos {sup_item['delivered_nos']}/{sup_item['planned_nos']}", color="primary", className="me-2"),
                                dbc.Badge(f"MT {sup_item['delivered_mt']:.1f}/{sup_item['planned_mt']:.1f}", color="dark"),
                            ], className="mb-1"))
                        sections_children.append(html.Div([
                            html.Div([
                                html.Span("Section Incharge: ", className="text-muted"), html.Strong(sec_name),
                                html.Span(" ", className="me-1"),
                                dbc.Badge(f"Nos {sec_data['delivered_nos']}/{sec_data['planned_nos']}", color="primary", className="ms-2 me-2"),
                                dbc.Badge(f"MT {sec_data['delivered_mt']:.1f}/{sec_data['planned_mt']:.1f}", color="dark"),
                            ], className="mb-2"),
                            *sup_children
                        ], className="mb-3"))

                # (legacy accordion meta/details removed)

                # Build the grid tile representation used for the new layout
                key = f"{pch}::{proj_name}"
                raw_code = display_code or r.get("project_code") or r.get("project_key") or proj_name
                proj_code = _compact_project_token(str(raw_code))
                tile_summary_children = [
                    html.Div(html.Strong(proj_name), className="mb-2"),
                    html.Div([
                        html.Span("Regional Manager : ", className="text-muted me-1"),
                        dbc.Badge(r.get("regional_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Project Manager : ", className="text-muted me-1"),
                        dbc.Badge(r.get("project_mgr", "-") or "-", color="light", text_color="dark", className="fw-semibold")
                    ], className="mb-2"),
                ]
                # Build month-aware responsibilities openers
                def _month_buttons_for_project() -> list:
                    # derive available months from mp_all for this project
                    months_vals: list = []
                    try:
                        if isinstance(mp_all, pd.DataFrame) and not mp_all.empty:
                            # Filter mp_all to this project using same robust matcher
                            name_lc_all = (mp_all.get("project_name", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                            key_lc_all = (mp_all.get("project_key", pd.Series([""] * len(mp_all), index=mp_all.index)).astype(str).str.strip().str.lower())
                            name_compact_all = name_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                            key_compact_all = key_lc_all.str.replace(r"[^a-z0-9]", "", regex=True)
                            mask_all = (
                                (name_lc_all == sel_name) | (key_lc_all == sel_name) |
                                (name_compact_all == sel_compact) | (key_compact_all == sel_compact) |
                                ((sel_code != "") & (key_compact_all == sel_code))
                            )
                            sub = mp_all.loc[mask_all].copy()
                            if "completion_month" not in sub.columns:
                                if "plan_month" in sub.columns:
                                    try:
                                        sub["completion_month"] = pd.to_datetime(sub["plan_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                                    except Exception:
                                        sub["completion_month"] = pd.NaT
                                elif "completion_date" in sub.columns:
                                    try:
                                        sub["completion_month"] = pd.to_datetime(sub["completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                                    except Exception:
                                        sub["completion_month"] = pd.NaT
                                else:
                                    sub["completion_month"] = pd.NaT
                            months_vals = sorted({pd.Timestamp(v) for v in sub["completion_month"] if pd.notna(v)})
                    except Exception:
                        months_vals = []
                    # render buttons
                    out: list = []
                    if months_vals:
                        out.append(html.Span("Monthly Plan (Erection) : ", className="me-2"))
                        # cap to last 2 months only
                        months_vals = sorted(months_vals)
                        months_vals = months_vals[-2:]
                        for ts in months_vals:
                            label = ts.strftime("%b %Y")
                            value = ts.strftime("%Y-%m")
                            key_payload = "||".join([
                                "erection",
                                proj_code or "",
                                value or "",
                                name_part.strip(),
                            ])
                            if use_modal_ids:
                                key_payload = f"{key_payload}||__modal__"
                            out.append(
                                dbc.Button(
                                    label,
                                    id={"type": "proj-resp-open", "key": key_payload},
                                    color="link",
                                    className="p-0 me-1",
                                )
                            )
                    return out

                # Project-level KPI highlights aligned with the PCH header metrics
                norm_keys, compact_keys = _project_lookup_keys(r)
                prod_current_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_current_norm_map, prod_current_compact_map
                )
                prod_overall_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, prod_history_norm_map, prod_history_compact_map
                )
                towers_current_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, towers_current_norm_map, towers_current_compact_map
                )
                towers_planned_value, _ = _lookup_with_key(
                    norm_keys, compact_keys, towers_planned_norm_map, towers_planned_compact_map
                )

                project_scope_cache: pd.DataFrame | None = None

                def _lazy_scope() -> pd.DataFrame:
                    nonlocal project_scope_cache
                    if project_scope_cache is None:
                        project_scope_cache = _project_scope_for_row(
                            r,
                            primary_scope=scope_full,
                            fallback_scope=scoped,
                        )
                    return project_scope_cache

                gangs_metric = None
                loss_metric = None
                if focus_metric in {"gangs", "loss"}:
                    scope_subset = _lazy_scope()
                    if focus_metric == "gangs":
                        gangs_metric = _count_gangs(scope_subset)
                    elif focus_metric == "loss":
                        loss_metric = _compute_project_loss_value(scope_subset, mode="erection")

                tile_summary_children.extend(
                    _build_tile_metric_rows(
                        mode="erection",
                        focus_metric=focus_metric,
                        prod_current_value=prod_current_value,
                        prod_overall_value=prod_overall_value,
                        total_current_value=towers_current_value,
                        total_planned_value=towers_planned_value,
                        gangs_value=gangs_metric,
                        loss_value=loss_metric,
                    )
                )
                project_token = _component_id_token("proj", proj_name)
                card_id = {
                    "type": "project-tile-trigger",
                    "mode": "erection",
                    "project": project_token,
                    "context": tile_context,
                }
                tile_metadata[project_token] = {
                    "project": proj_name,
                    "code": display_code or proj_name,
                    "display": proj_name,
                    "mode": "erection",
                    "pch": str(pch),
                }

                tile_body_children = [
                    html.Div(tile_summary_children, className="project-tile-summary"),
                ]
                month_buttons = _month_buttons_for_project()
                if month_buttons:
                    tile_body_children.append(html.Div(month_buttons, className="mb-1"))
                else:
                    tile_body_children.append(html.Div("Micro Plan not available.", className="text-muted"))

                tile_card = dbc.Card(dbc.CardBody(tile_body_children), className="h-100 shadow-sm")

                tile_cols.append(
                    dbc.Col(
                        html.Div(
                            tile_card,
                            id=card_id,
                            n_clicks=0,
                            className="project-tile-card",
                            role="button",
                            tabIndex=0,
                        ),
                        xs=12,
                        sm=12,
                        md=6,
                        lg=4,
                        className="mb-3"
                    )
                )

            body_children = (
                [dbc.Row(tile_cols, className="g-3")]
                if tile_cols
                else [html.Div("No projects available.", className="text-muted")]
            )
            towers_balance_label = max(towers_planned_label - towers_delivered_label, 0)
            pill_components = [
                                ("projects", dbc.Button(
                                    projects_label,
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "projects"},
                                    className="pch-pill pch-pill-projects me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("productivity", dbc.Button(
                                    f"Productivity / Historical Avg: {fmt_prod_current} / {fmt_prod_history} MT/day",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "productivity"},
                                    className="pch-pill pch-pill-prod-month me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("totals", dbc.Button(
                                    f"F/S Total Planned / Done / Balance: {towers_planned_label} / {towers_delivered_label} / {towers_balance_label}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "totals"},
                                    className="pch-pill pch-pill-towers me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("gangs", dbc.Button(
                                    f"Gangs: {int(gangs_total):,}",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "gangs"},
                                    className="pch-pill pch-pill-gangs me-2 mb-1", color="link", n_clicks=0
                                )),
                                ("loss", dbc.Button(
                                    f"Lost Units: {lost_units_total:.1f} MT",
                                    id={"type": "summary-pill-trigger", "mode": "erection", "metric": "loss"},
                                    className="pch-pill pch-pill-loss me-2 mb-1", color="link", n_clicks=0
                                )),
                            ]

            title_component = html.Div(
                [
                    html.Span(str(pch or "Unassigned"), className="fw-semibold"),
                    html.Div(
                        _filter_pch_header_pills(pill_components, pill_focus),
                        className="pch-pill-group ms-auto d-none d-md-flex",
                    ),
                ],
                className="d-flex align-items-center justify-content-between w-100",
            )
            pch_sections.append(
                dbc.AccordionItem(
                    title=title_component,
                    children=body_children,
                    item_id=f"pch-{_slugify_pch(pch)}",
                    className="pch-section mb-2",
                )
            )

        if not pch_sections:
            pch_sections = _empty_pch_items("No projects match the current filters.")

        return pch_sections, None, tile_metadata

    _PCH_PILL_LABELS = {
        "projects": "Projects Covered",
        "totals": "F/S Total Planned / Done / Balance",
        "gangs": "Gangs",
        "productivity": "Productivity / Historical Avg",
        "loss": "Lost Units",
        "tse": "No. of TSE",
    }

    @app.callback(
        Output("store-pch-modal-focus", "data"),
        Output("kpi-pch-modal", "is_open"),
        Input({"type": "summary-pill-trigger", "mode": ALL, "metric": ALL}, "n_clicks"),
        Input("kpi-pch-modal-close", "n_clicks"),
        State("kpi-pch-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_pch_modal(pill_clicks, close_clicks, is_open):
        trigger = _resolve_triggered_id()
        ctx = dash.callback_context
        triggered_entries = getattr(ctx, "triggered", None)
        trigger_value = None
        if triggered_entries:
            try:
                trigger_value = triggered_entries[0].get("value")
            except Exception:
                trigger_value = None

        if trigger == "kpi-pch-modal-close":
            return dash.no_update, False
        if isinstance(trigger, dict) and trigger.get("type") == "summary-pill-trigger":
            if not trigger_value:
                raise PreventUpdate
            metric = str(trigger.get("metric") or "").strip().lower()
            mode = str(trigger.get("mode") or "").strip().lower() or "erection"
            if metric not in _PCH_PILL_LABELS:
                raise PreventUpdate
            if mode == "stringing" and not config.enable_stringing:
                raise PreventUpdate
            if mode not in {"erection", "stringing"}:
                mode = "erection"
            payload = {"metric": metric, "mode": mode}
            return payload, True
        raise PreventUpdate

    @app.callback(
        Output("kpi-pch-modal-accordion", "children"),
        Output("kpi-pch-modal-accordion", "active_item"),
        Output("kpi-pch-modal-title", "children"),
        Output("store-project-tile-meta", "data"),
        Input("store-pch-modal-focus", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("store-stringing-scope", "data"),
        prevent_initial_call=True,
    )
    def _render_pch_modal(focus_data, projects, months, quick_range, stringing_scope):
        if not isinstance(focus_data, dict) or not focus_data.get("metric"):
            raise PreventUpdate
        metric = str(focus_data.get("metric") or "").strip().lower()
        if metric not in _PCH_PILL_LABELS:
            raise PreventUpdate
        mode_value = str(focus_data.get("mode") or "").strip().lower()
        if mode_value not in {"erection", "stringing"}:
            mode_value = "erection"
        if mode_value == "stringing" and not config.enable_stringing:
            raise PreventUpdate
        method_filter = _method_filters_for_scope(stringing_scope) if mode_value == "stringing" else []
        sections, active_item, tile_meta = _populate_kpi_pch(
            projects,
            months,
            quick_range,
            mode_value,
            method_filter,
            stringing_scope,
            use_modal_ids=True,
            pill_focus=metric,
        )
        mode_label = "Stringing" if mode_value == "stringing" else "Erection"
        title = f"PCH-wise { _PCH_PILL_LABELS[metric] } ({mode_label})"
        return sections, active_item, title, tile_meta

    @app.callback(
        Output("store-project-modal-focus-cache", "data"),
        Input("store-project-tile-focus", "data"),
        prevent_initial_call=False,
    )
    def _cache_project_modal_focus(focus_payload):
        """Mirror the latest project focus payload for history-driven modal opens."""
        if isinstance(focus_payload, Mapping):
            return focus_payload
        return None

    @app.callback(
        Output("store-project-tile-focus", "data"),
        Output("project-detail-modal", "is_open"),
        Output("store-project-modal-history", "data"),
        Input({"type": "project-tile-trigger", "project": ALL, "mode": ALL, "context": ALL}, "n_clicks"),
        Input("project-modal-close-top", "n_clicks"),
        Input({"type": "proj-resp-open", "key": ALL}, "n_clicks"),
        Input("f-project", "value"),
        Input("project-modal-location", "href"),
        State("project-detail-modal", "is_open"),
        State("store-project-tile-meta", "data"),
        State("store-project-modal-focus-cache", "data"),
        prevent_initial_call=True,
    )
    def _toggle_project_tile_modal(
        tile_clicks,
        close_icon_clicks,
        _resp_open_clicks,
        project_values,
        location_href,
        is_open,
        tile_meta_data,
        focus_cache,
    ):
        ctx = dash.callback_context
        triggered_entries = getattr(ctx, "triggered", None)
        trigger = _resolve_triggered_id()
        wants_modal_flag = _href_has_project_modal_flag(location_href)
        LOGGER.info(
            "project-modal-toggle trigger=%s entries=%s tile_clicks=%s close_top=%s open=%s flag=%s",
            trigger,
            triggered_entries,
            tile_clicks,
            close_icon_clicks,
            is_open,
            wants_modal_flag,
        )

        history_payload = dash.no_update

        def _history_action(action: str) -> dict[str, Any]:
            return {"action": action, "ts": time.time()}

        if trigger == "project-modal-close-top":
            if not is_open:
                raise PreventUpdate
            return dash.no_update, False, _history_action("close")

        if trigger == "project-modal-location":
            if wants_modal_flag:
                if not focus_cache:
                    # Ignore stray modal query params until a project has been selected.
                    raise PreventUpdate
                if not is_open:
                    return dash.no_update, True, dash.no_update
            if not wants_modal_flag and is_open:
                return dash.no_update, False, dash.no_update
            raise PreventUpdate

        if isinstance(trigger, dict) and trigger.get("type") == "proj-resp-open":
            raise PreventUpdate
        if trigger == "f-project":
            normalized_projects = _normalize_str_list(_ensure_list(project_values))
            if not normalized_projects:
                if is_open:
                    history_payload = _history_action("close")
                return None, False, history_payload
            target_label = normalized_projects[-1]
            matched_meta = _match_tile_meta_entry(target_label, tile_meta_data if isinstance(tile_meta_data, Mapping) else None)
            project_code = _extract_project_code(target_label)
            payload = {
                "project": target_label,
                "code": project_code or target_label,
                "display": target_label,
                "mode": "erection",
                "pch": None,
                "ts": time.time(),
            }
            if matched_meta:
                payload.update(
                    {
                        "project": matched_meta.get("project") or payload["project"],
                        "display": matched_meta.get("display") or payload["display"],
                        "code": matched_meta.get("code") or payload["code"],
                        "mode": matched_meta.get("mode") or payload["mode"],
                        "pch": matched_meta.get("pch"),
                    }
                )
            if not is_open:
                history_payload = _history_action("open")
            return payload, True, history_payload
        if isinstance(trigger, dict) and trigger.get("type") == "project-tile-trigger":
            if trigger.get("context") == "placeholder":
                raise PreventUpdate
            first_entry_value = None
            if triggered_entries:
                try:
                    first_entry_value = triggered_entries[0].get("value")
                except Exception:
                    first_entry_value = None
            if not first_entry_value:
                raise PreventUpdate
            project_key = trigger.get("project")
            tile_meta = tile_meta_data or {}
            meta = tile_meta.get(project_key or "")
            if not meta:
                LOGGER.warning("No tile metadata found for key %s", project_key)
                raise PreventUpdate
            payload = {
                "project": meta.get("project") or meta.get("display"),
                "code": meta.get("code"),
                "display": meta.get("display") or meta.get("project"),
                "mode": meta.get("mode") or trigger.get("mode"),
                "pch": meta.get("pch"),
                "ts": time.time(),
            }
            if not is_open:
                history_payload = _history_action("open")
            return payload, True, history_payload
        raise PreventUpdate

    def _project_modal_summary_placeholder() -> html.Div:
        return html.Div(
            [
                html.Div("Select a project tile to view its detailed view.", className="mb-2"),
                html.Div(
                    [
                        dbc.Button(
                            "Show Completed Towers",
                            id="project-modal-btn-erections",
                            color="primary",
                            size="lg",
                            className="modal-section-btn",
                            n_clicks=0,
                        ),
                        dbc.Button(
                            "Show Gang Performance",
                            id="project-modal-btn-performance-erection",
                            color="primary",
                            size="lg",
                            className="modal-section-btn",
                            n_clicks=0,
                        ),
                    ],
                    className="d-flex gap-2 flex-wrap",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Show Completed Stringing",
                            id="project-modal-btn-stringing",
                            color="primary",
                            size="lg",
                            className="modal-section-btn",
                            n_clicks=0,
                        ),
                        dbc.Button(
                            "Show Gang Performance",
                            id="project-modal-btn-performance-stringing",
                            color="primary",
                            size="lg",
                            className="modal-section-btn",
                            n_clicks=0,
                        ),
                    ],
                    className="d-flex gap-2 flex-wrap",
                    style={"display": "none"},
                ),
                html.Div(
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
                    style={"display": "none"},
                ),
            ],
            className="project-empty",
        )

    @app.callback(
        Output("project-modal-summary", "children"),
        Output("project-modal-title", "children"),
        Input("store-project-tile-focus", "data"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("project-modal-stringing-scope", "value"),
        prevent_initial_call=True,
    )
    def _render_project_modal_summary(
        focus_data,
        months,
        quick_range,
        gangs,
        stringing_scope,
    ):
        base_title = "Project Deep Dive"
        if not isinstance(focus_data, dict):
            return _project_modal_summary_placeholder(), base_title

        project_name = (focus_data.get("project") or "").strip()
        project_display = (focus_data.get("display") or project_name).strip()
        project_code = (focus_data.get("code") or "").strip()
        title_label = project_display or project_name
        lookup_label = project_name or project_display

        candidate_ids = _project_filter_candidates(lookup_label, project_code)
        if not candidate_ids:
            fallback = [value for value in (project_name, project_display, project_code) if value]
            candidate_ids = fallback[:1]
        project_list = _normalize_str_list(candidate_ids)
        if not project_list:
            return _project_modal_summary_placeholder(), base_title

        gang_list = _normalize_str_list(_ensure_list(gangs))
        months_list = _normalize_str_list(_ensure_list(months))
        deployment_scope = _normalize_deployment_filter(stringing_scope)
        method_list = _normalize_str_list(_method_filters_for_scope(deployment_scope), lower=True)

        def _project_summary_for_mode(mode_value: str, *, is_stringing: bool) -> dict[str, str]:
            method_payload = method_list if is_stringing else []
            try:
                scope_meta = _build_scope_meta_payload(
                    eff_mode=mode_value,
                    project_list=project_list,
                    gang_list=gang_list,
                    months_list=months_list,
                    quick_range=quick_range,
                    method_values=method_payload,
                    method_list=method_payload,
                    deployment_filter=deployment_scope if is_stringing else "all",
                )
            except Exception:
                LOGGER.exception(
                    "Unable to build %s summary for project %s", mode_value, title_label or "unknown"
                )
                return _empty_summary_payload(is_stringing)
            return _summarize_scope_for_cards(scope_meta)

        erection_summary = _project_summary_for_mode("erection", is_stringing=False)
        stringing_summary = (
            _project_summary_for_mode("stringing", is_stringing=True) if config.enable_stringing else None
        )

        import re as _re

        display_label = project_display or project_name or lookup_label

        def _compact_project_value(value: Any) -> str:
            text = "" if value is None else str(value).strip().lower()
            if not text:
                return ""
            return _re.sub(r"[^a-z0-9]", "", text)

        match_normals: set[str] = set()
        match_compacts: set[str] = set()

        def _register_match_value(value: Any) -> None:
            text = "" if value is None else str(value).strip()
            if not text:
                return
            lowered = text.lower()
            match_normals.add(lowered)
            compact = _re.sub(r"[^a-z0-9]", "", lowered)
            if compact:
                match_compacts.add(compact)

        for candidate in project_list:
            _register_match_value(candidate)
        for candidate in (project_code, project_display, project_name, lookup_label):
            _register_match_value(candidate)

        project_code_token = _compact_project_value(project_code or display_label)

        def _collect_plan_months(plan_mode: str) -> list[pd.Timestamp]:
            try:
                frame, _, _, _ = _fetch_monthly_plan(plan_mode)
            except Exception:
                LOGGER.exception("Unable to load %s monthly plan for modal summary", plan_mode)
                return []
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return []
            work = frame.copy()
            if "completion_month" in work.columns:
                work["completion_month"] = pd.to_datetime(
                    work["completion_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            elif "plan_month" in work.columns:
                work["completion_month"] = pd.to_datetime(
                    work["plan_month"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            elif "completion_date" in work.columns:
                work["completion_month"] = pd.to_datetime(
                    work["completion_date"], errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            else:
                work["completion_month"] = pd.NaT
            if "completion_month" not in work.columns:
                return []
            match_columns = [
                column
                for column in (
                    "project_name",
                    "project",
                    "project_key",
                    "project_name_display",
                    "Project Name",
                    "Project Code",
                )
                if column in work.columns
            ]
            if not match_columns or (not match_normals and not match_compacts):
                return []
            mask = pd.Series(False, index=work.index)
            for column in match_columns:
                series = work[column].astype(str).str.strip().str.lower()
                col_mask = pd.Series(False, index=series.index)
                if match_normals:
                    col_mask |= series.isin(match_normals)
                if match_compacts:
                    col_mask |= series.str.replace(r"[^a-z0-9]", "", regex=True).isin(match_compacts)
                mask |= col_mask
            if not mask.any():
                return []
            subset = work.loc[mask, "completion_month"].dropna()
            if subset.empty:
                return []
            months = sorted({pd.Timestamp(value) for value in subset})
            if not months:
                return []
            return months[-2:]

        def _monthly_plan_block(plan_mode: str) -> html.Div | None:
            label = "Monthly Plan (Stringing)" if plan_mode == "stringing" else "Monthly Plan (Erection)"
            months = _collect_plan_months(plan_mode)
            if not months:
                return html.Div("Micro Plan not available.", className="summary-plan-tile summary-plan-empty text-muted")
            mode_token = "stringing" if plan_mode == "stringing" else "erection"
            link_buttons: list[Any] = []
            for ts in months:
                label_text = ts.strftime("%b %Y")
                month_value = ts.strftime("%Y-%m")
                payload = "||".join(
                    [
                        mode_token,
                        project_code_token or "",
                        month_value or "",
                        display_label or project_name or "",
                    ]
                )
                payload = f"{payload}||__modal__"
                link_buttons.append(
                    dbc.Button(
                        label_text,
                        id={"type": "proj-resp-open", "key": payload},
                        color="link",
                        className="summary-plan-link",
                    )
                )
            return html.Div(
                [
                    html.Div(label, className="summary-plan-tile__title"),
                    html.Div(
                        link_buttons,
                        className="summary-plan-tile__links d-flex flex-wrap gap-2",
                    ),
                ],
                className="summary-plan-tile",
            )

        def _summary_card(
            title: str,
            summary_payload: dict[str, str],
            *,
            include_tse: bool = False,
            include_po: bool = False,
            actions: list[Any] | None = None,
            controls: Any | None = None,
        ) -> dbc.Col:
            rows = [
                ("F/S Total Planned / Done / Balance", summary_payload.get("totals", "-")),
                ("Gangs", summary_payload.get("gangs", "-")),
                ("Productivity / Historical Avg", summary_payload.get("productivity", "-")),
                ("Lost Units", summary_payload.get("lost_units", "-")),
            ]
            if include_po:
                rows.append(("P/O Total Planned / Done / Balance", summary_payload.get("po_completion", "-")))
            if include_tse:
                rows.append(("No. of TSE", summary_payload.get("tse", "-")))
            pills = [
                html.Div(
                    [
                        html.Span(label, className="summary-pill__label"),
                        html.Span(value, className="summary-pill__value"),
                    ],
                    className="summary-pill",
                )
                for label, value in rows
            ]
            if controls is not None:
                header = html.Div(
                    [
                        html.Div(title, className="snapshot-card__title fw-semibold"),
                        html.Div(
                            controls,
                            className="stringing-scope-control d-flex flex-wrap align-items-center gap-2",
                        ),
                    ],
                    className="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-2",
                )
            else:
                header = html.Div(title, className="snapshot-card__title fw-semibold mb-2")
            children = [header, *pills]
            if actions:
                children.append(
                    html.Div(actions, className="summary-card-actions d-flex flex-column gap-2 mt-2")
                )
            return dbc.Col(
                dbc.Card(
                    dbc.CardBody(children, className="d-flex flex-column gap-1 summary-card-body"),
                    className="shadow-sm h-100 snapshot-card",
                ),
                xs=12,
                md=6,
            )

        erection_actions: list[Any] = []
        plan_block = _monthly_plan_block("erection")
        if plan_block is not None:
            erection_actions.append(plan_block)
        erection_actions.append(
            html.Div(
                [
                    dbc.Button(
                        "Show Completed Towers",
                        id="project-modal-btn-erections",
                        color="primary",
                        size="lg",
                        className="modal-section-btn",
                    ),
                    dbc.Button(
                        "Show Gang Performance",
                        id="project-modal-btn-performance-erection",
                        color="primary",
                        size="lg",
                        className="modal-section-btn",
                    ),
                ],
                className="d-flex flex-wrap gap-2",
            )
        )

        stringing_actions: list[Any] | None = None
        stringing_scope_control: Any | None = None
        if config.enable_stringing:
            stringing_actions = []
            plan_block_stringing = _monthly_plan_block("stringing")
            if plan_block_stringing is not None:
                stringing_actions.append(plan_block_stringing)
            stringing_actions.append(
                html.Div(
                    [
                        dbc.Button(
                            "Show Completed Stringing",
                            id="project-modal-btn-stringing",
                            color="primary",
                            size="lg",
                            className="modal-section-btn",
                        ),
                        dbc.Button(
                            "Show Gang Performance",
                            id="project-modal-btn-performance-stringing",
                            color="primary",
                            size="lg",
                            className="modal-section-btn",
                        ),
                    ],
                    className="d-flex flex-wrap gap-2",
                )
            )
            stringing_scope_control = [
                html.Div("Deployment", className="filter-label mb-1 me-2"),
                dbc.RadioItems(
                    id="project-modal-stringing-scope",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Manual", "value": "manual"},
                        {"label": "TSE", "value": "tse"},
                        {"label": "Hotline", "value": "hotline"},
                    ],
                    value=deployment_scope,
                    class_name="segment",
                    label_class_name="segment-label",
                    label_checked_class_name="segment-label--active",
                    input_class_name="segment-input",
                ),
            ]
        else:
            hidden_controls = html.Div(
                [
                    html.Div(
                        [
                            dbc.Button(
                                "Show Completed Stringing",
                                id="project-modal-btn-stringing",
                                color="primary",
                                size="lg",
                                className="modal-section-btn",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "Show Gang Performance",
                                id="project-modal-btn-performance-stringing",
                                color="primary",
                                size="lg",
                                className="modal-section-btn",
                                n_clicks=0,
                            ),
                        ],
                        className="d-flex flex-wrap gap-2",
                    ),
                    dbc.RadioItems(
                        id="project-modal-stringing-scope",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Manual", "value": "manual"},
                            {"label": "TSE", "value": "tse"},
                            {"label": "Hotline", "value": "hotline"},
                        ],
                        value=deployment_scope,
                    ),
                ],
                style={"display": "none"},
            )
            cards.append(hidden_controls)

        cards: list[dbc.Col] = [
            _summary_card(
                "Erection Snapshot",
                erection_summary or _empty_summary_payload(False),
                actions=erection_actions,
            ),
        ]
        if config.enable_stringing:
            cards.append(
                _summary_card(
                    "Stringing Snapshot",
                    stringing_summary or _empty_summary_payload(True),
                    include_tse=True,
                    include_po=True,
                    actions=stringing_actions,
                    controls=stringing_scope_control,
                )
            )

        summary_layout = dbc.Row(cards, className="g-2 project-modal-summary-table")
        title = f"{base_title} · {title_label}" if title_label else base_title
        return summary_layout, title

    @app.callback(
        Output("store-project-modal-section", "data"),
        Output("store-project-modal-performance-mode", "data"),
        Output("project-modal-scroll-target", "data"),
        Input("project-modal-btn-erections", "n_clicks"),
        Input("project-modal-btn-stringing", "n_clicks"),
        Input("project-modal-btn-performance-erection", "n_clicks"),
        Input("project-modal-btn-performance-stringing", "n_clicks"),
        Input("store-project-tile-focus", "data"),
        State("store-project-modal-section", "data"),
        State("store-project-modal-performance-mode", "data"),
        prevent_initial_call=True,
    )
    def _set_modal_section(
        btn_e,
        btn_s,
        btn_perf_e,
        btn_perf_s,
        focus_data,
        current_section,
        current_mode,
    ):
        trigger = _resolve_triggered_id()
        perf_mode = _modal_mode_from_store(current_mode, "erection")

        def _payload(mode_value: str) -> str:
            return _compose_modal_mode_payload(mode_value)

        def _scroll_payload(anchor_id: str) -> dict[str, float]:
            return {"anchor": anchor_id, "ts": time.time()}

        if trigger == "store-project-tile-focus":
            perf_mode = _resolve_focus_mode(focus_data, perf_mode)
            if perf_mode == "stringing" and not config.enable_stringing:
                perf_mode = "erection"
            return "erections", _payload(perf_mode), dash.no_update
        if trigger == "project-modal-btn-erections":
            if not btn_e:
                raise PreventUpdate
            return (
                "erections",
                _payload(perf_mode),
                _scroll_payload("project-modal-anchor-erections"),
            )
        if trigger == "project-modal-btn-stringing":
            if not config.enable_stringing:
                raise PreventUpdate
            if not btn_s:
                raise PreventUpdate
            return (
                "stringing",
                _payload(perf_mode),
                _scroll_payload("project-modal-anchor-stringing"),
            )
        if trigger == "project-modal-btn-performance-erection":
            if not btn_perf_e:
                raise PreventUpdate
            return (
                "performance",
                _payload("erection"),
                _scroll_payload("project-modal-anchor-performance"),
            )
        if trigger == "project-modal-btn-performance-stringing":
            if not btn_perf_s:
                raise PreventUpdate
            target_mode = "stringing" if config.enable_stringing else "erection"
            return (
                "performance",
                _payload(target_mode),
                _scroll_payload("project-modal-anchor-performance"),
            )
        return (current_section or "erections"), _payload(perf_mode), dash.no_update

    @app.callback(
        Output("project-modal-section-erections", "is_open"),
        Output("project-modal-section-stringing", "is_open"),
        Output("project-modal-section-performance", "is_open"),
        Output("project-modal-btn-erections", "className"),
        Output("project-modal-btn-stringing", "className"),
        Output("project-modal-btn-performance-erection", "className"),
        Output("project-modal-btn-performance-stringing", "className"),
        Input("store-project-modal-section", "data"),
        Input("store-project-modal-performance-mode", "data"),
    )
    def _sync_modal_sections(active_section: str | None, performance_mode: Any):
        active = (active_section or "erections").strip().lower()
        perf_mode = _modal_mode_from_store(performance_mode, "erection")

        def _is_open(key: str) -> bool:
            return active == key

        def _class(key: str) -> str:
            base = "modal-section-btn"
            return f"{base} active" if active == key else base

        def _perf_class(target: str) -> str:
            base = "modal-section-btn"
            is_active = active == "performance" and perf_mode == target
            return f"{base} active" if is_active else base

        return (
            _is_open("erections"),
            _is_open("stringing"),
            _is_open("performance"),
            _class("erections"),
            _class("stringing"),
            _perf_class("erection"),
            _perf_class("stringing"),
        )

    # Toggle responsibilities visibility inside each project tile (pattern-matching IDs)
    @app.callback(
        Output({"type": "proj-resp-collapse", "key": MATCH}, "is_open"),
        Input({"type": "proj-resp-toggle", "key": MATCH}, "n_clicks"),
        State({"type": "proj-resp-collapse", "key": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_tile_resp(n, is_open):
        if not n:
            raise PreventUpdate
        return not bool(is_open)

    # --- Project Monthly Plan mini-modal: open/close and set project code ---
    @app.callback(
        Output("proj-resp-modal", "is_open"),
        Output("proj-resp-modal-title", "children"),
        Output("store-proj-resp-code", "data"),
        Output("store-proj-resp-month", "data"),
        Output("store-proj-resp-plan", "data"),
        Input({"type": "proj-resp-open", "key": ALL}, "n_clicks"),
        Input("proj-resp-modal-close", "n_clicks"),
        State("proj-resp-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_proj_resp_modal(open_clicks, close_clicks, is_open):
        trigger_id = _resolve_triggered_id()
        if trigger_id is None:
            raise PreventUpdate
        if trigger_id == "proj-resp-modal-close":
            return False, dash.no_update, dash.no_update, None, None
        ctx = dash.callback_context
        triggered_entries = getattr(ctx, "triggered", None)
        if not triggered_entries:
            raise PreventUpdate
        trigger_value = triggered_entries[0].get("value")
        if not trigger_value:
            # Ignore initial invocation where n_clicks is zero/None
            raise PreventUpdate
        key_str = None
        if isinstance(trigger_id, dict):
            id_obj = trigger_id
            key_str = id_obj.get("key")
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if id_obj.get("type") != "proj-resp-open":
            raise PreventUpdate

        code = name = month_raw = None
        plan_mode = "erection"
        if isinstance(key_str, str):
            parts = key_str.split("||")
            if parts and parts[-1] == "__modal__":
                parts = parts[:-1]
            if parts:
                candidate_mode = (parts[0] or "").strip().lower()
                if candidate_mode in {"erection", "stringing"}:
                    plan_mode = candidate_mode
                    parts = parts[1:]
            if parts:
                code = parts[0] or None
            if len(parts) > 1:
                month_raw = parts[1] or None
            if len(parts) > 2:
                name = parts[2] or None

        month_value, month_label = _normalize_month_value(month_raw)
        display_title = name or code
        plan_title = "Monthly Plan (Stringing)" if plan_mode == "stringing" else "Monthly Plan (Erection)"
        if display_title:
            title = f"{plan_title} \u2014 {display_title}"
        else:
            title = plan_title
        if month_label:
            title = f"{title} ({month_label})"
        payload = {"code": code, "name": name}
        return True, title, payload, month_value, plan_mode

    # --- Render responsibilities inside the project mini-modal ---
    @app.callback(
        Output("proj-resp-graph", "figure"),
        Output("proj-resp-kpi-target", "children"),
        Output("proj-resp-kpi-delivered", "children"),
        Output("proj-resp-kpi-ach", "children"),
        Input("store-proj-resp-code", "data"),
        Input("store-proj-resp-month", "data"),
        Input("store-proj-resp-plan", "data"),
        Input("proj-resp-entity", "value"),
        Input("proj-resp-metric", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("proj-resp-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _render_proj_resp(code_value, month_value, plan_mode_value, entity_value, metric_value, months_value, quick_value, is_open):
        if not code_value:
            raise PreventUpdate
        project_identifiers: list[str] = []
        seen: set[str] = set()

        def _append_identifier(candidate: Any) -> None:
            text = "" if candidate is None else str(candidate).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                project_identifiers.append(text)

        if isinstance(code_value, dict):
            for candidate in (code_value.get("name"), code_value.get("code")):
                _append_identifier(candidate)
        elif isinstance(code_value, Sequence) and not isinstance(code_value, (str, bytes)):
            for candidate in code_value:
                _append_identifier(candidate)
        else:
            _append_identifier(code_value)
        if not project_identifiers:
            raise PreventUpdate
        # If a dedicated month is chosen from the tile, override the global filters
        normalized_month, _month_label = _normalize_month_value(month_value)
        if normalized_month:
            months_value = [normalized_month]
            quick_value = None
        plan_key = "stringing" if str(plan_mode_value).strip().lower() == "stringing" else "erection"
        return _build_monthly_plan_for_project(
            project_value=project_identifiers,
            entity_value=entity_value,
            metric_value=metric_value,
            months_value=months_value,
            quick_range_value=quick_value,
            plan_mode=plan_key,
        )

    @app.callback(
        Output("trace-modal", "is_open"),
        Output("trace-modal-title", "children"),
        # Output("store-selected-gang", "data"),
        Input("store-dblclick", "data"),
        Input("trace-modal-close", "n_clicks"),
        State("trace-modal", "is_open"),
        State("store-selected-gang", "data"),
        prevent_initial_call=True,
    )
    def toggle_trace_modal(
        dbl_click: dict[str, Any] | None,
        close_clicks: int | None,
        is_open: bool,
        current_selection: str | None,
    ) -> tuple[bool, Any, str | None]:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "trace-modal-close":
            return False, dash.no_update
        if trigger == "store-dblclick":
            if not dbl_click or not dbl_click.get("gang"):
                raise PreventUpdate
            gang_value = dbl_click["gang"]
            title = f"Traceability - {gang_value}"
            return True, title
        raise PreventUpdate

    def _analytics_empty_fig(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 13, "color": "#64748b"},
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 30, "r": 20, "t": 20, "b": 30},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    def _safe_float(value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if np.isfinite(number) else 0.0

    def _analytics_chart_layout(title: str, yaxis_title: str) -> dict[str, Any]:
        return {
            "template": "plotly_white",
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "margin": {"l": 40, "r": 20, "t": 24, "b": 40},
            "font": {"family": "Inter, system-ui", "size": 12, "color": "#0f172a"},
            "legend": {"orientation": "h", "y": 1.15, "x": 0},
            "xaxis": {"title": title, "tickangle": -25},
            "yaxis": {"title": yaxis_title, "gridcolor": "#e6e9f0", "zerolinecolor": "#e6e9f0"},
        }

    def _analytics_sparkline(
        x_values: list[str],
        y_values: list[float],
        *,
        color: str,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 5},
                hovertemplate="%{y:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    def _analytics_table_from_selection(
        payload: dict[str, Any] | None,
        selection: dict[str, Any] | None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], str]:
        if not payload or not selection:
            return [], [], ""
        kind = selection.get("kind")
        definition = ""

        if kind == "bucket":
            bucket = selection.get("bucket")
            rows = [
                row
                for row in (payload.get("bucket", {}).get("gang_months") or [])
                if row.get("bucket_label") == bucket
            ]
            df = pd.DataFrame(rows)
            columns = [
                {"name": "Gang", "id": "gang_name"},
                {"name": "Month", "id": "month"},
                {"name": "MT (month)", "id": "total_mt"},
                {"name": "Active days", "id": "active_days"},
                {"name": "Avg MT/day", "id": "avg_mt_day"},
                {"name": "Bucket", "id": "bucket_label"},
                {"name": "Projects", "id": "projects"},
            ]
            if not df.empty:
                df["total_mt"] = pd.to_numeric(df["total_mt"], errors="coerce").round(2)
                df["active_days"] = pd.to_numeric(df["active_days"], errors="coerce").round(0)
                df["avg_mt_day"] = pd.to_numeric(df["avg_mt_day"], errors="coerce").round(2)
            definition = (
                "Buckets use average MT/day per gang per calendar month. "
                "Shares compare the count of deployments with their MT contribution. "
                "What-if assumes the selected bucket reaches the target MT/day while output and active days stay constant."
            )
            return columns, df.to_dict("records"), definition

        if kind == "tier":
            tier = selection.get("tier")
            rows = [
                row
                for row in (payload.get("tiers", {}).get("gangs") or [])
                if row.get("tier") == tier
            ]
            df = pd.DataFrame(rows)
            if "erections_completed" in df.columns:
                df = df[pd.to_numeric(df["erections_completed"], errors="coerce") >= MIN_ERECTIONS_FOR_TIERS]
            columns = [
                {"name": "Gang", "id": "gang_name"},
                {"name": "Tier", "id": "tier"},
                {"name": "Avg MT/day", "id": "avg_prod_mt_day"},
                {"name": "Idle Windows", "id": "idle_windows"},
                {"name": "Idle Days", "id": "idle_days_capped"},
                {"name": "Completions", "id": "erections_completed"},
                {"name": "Towers", "id": "towers"},
                {"name": "Projects", "id": "projects"},
            ]
            if not df.empty:
                df["avg_prod_mt_day"] = pd.to_numeric(df["avg_prod_mt_day"], errors="coerce").round(2)
                df["idle_days_capped"] = pd.to_numeric(df["idle_days_capped"], errors="coerce").round(1)
            definition = (
                "Tiers use average MT/day per gang over the selected period. "
                "Idle windows count gaps between work dates; idle days are capped at 15."
            )
            return columns, df.to_dict("records"), definition

        if kind == "hist_bin":
            bin_label = selection.get("bin")
            rows = [
                row
                for row in (payload.get("tiers", {}).get("gangs") or [])
                if row.get("hist_bin") == bin_label
            ]
            df = pd.DataFrame(rows)
            columns = [
                {"name": "Gang", "id": "gang_name"},
                {"name": "Bin", "id": "hist_bin"},
                {"name": "Avg MT/day", "id": "avg_prod_mt_day"},
                {"name": "Tier", "id": "tier"},
                {"name": "Idle Windows", "id": "idle_windows"},
                {"name": "Idle Days", "id": "idle_days_capped"},
                {"name": "Projects", "id": "projects"},
            ]
            if not df.empty:
                df["avg_prod_mt_day"] = pd.to_numeric(df["avg_prod_mt_day"], errors="coerce").round(2)
                df["idle_days_capped"] = pd.to_numeric(df["idle_days_capped"], errors="coerce").round(1)
            definition = (
                "Histogram bins use per-gang average MT/day across the selected period."
            )
            return columns, df.to_dict("records"), definition

        if kind == "idle_windows":
            tiers = set(selection.get("tiers") or [])
            gang_rows = payload.get("tiers", {}).get("gangs") or []
            eligible_gangs = {
                row.get("gang_name")
                for row in gang_rows
                if float(row.get("erections_completed", 0) or 0) >= MIN_ERECTIONS_FOR_TIERS
            }
            tier_map = {
                row.get("gang_name"): row.get("tier")
                for row in gang_rows
                if row.get("gang_name") in eligible_gangs
            }
            interval_rows = payload.get("tiers", {}).get("idle_intervals") or []
            rows = []
            for row in interval_rows:
                if row.get("gang_name") not in eligible_gangs:
                    continue
                tier = tier_map.get(row.get("gang_name"))
                if tiers and tier not in tiers:
                    continue
                merged = dict(row)
                merged["tier"] = tier or ""
                rows.append(merged)
            df = pd.DataFrame(rows)
            columns = [
                {"name": "Gang", "id": "gang_name"},
                {"name": "Tier", "id": "tier"},
                {"name": "Interval Start", "id": "interval_start"},
                {"name": "Interval End", "id": "interval_end"},
                {"name": "Raw Gap (days)", "id": "raw_gap_days"},
                {"name": "Idle Counted (days)", "id": "idle_days_capped"},
            ]
            definition = (
                "Idle window = gap between consecutive work dates. "
                "Idle days per window are capped at 15."
            )
            return columns, df.to_dict("records"), definition

        if kind == "hotspot":
            project_name = selection.get("project")
            rows = [
                row
                for row in (payload.get("hotspot", {}).get("gangs") or [])
                if not project_name or row.get("project_name") == project_name
            ]
            if not rows:
                rows = payload.get("hotspot", {}).get("projects") or []
            df = pd.DataFrame(rows)
            if "project_name" in df.columns and "gang_name" in df.columns:
                columns = [
                    {"name": "Project", "id": "project_name"},
                    {"name": "Gang", "id": "gang_name"},
                    {"name": "Avg MT/day", "id": "avg_prod_mt_day"},
                    {"name": "Idle Windows", "id": "idle_windows"},
                    {"name": "Idle Days", "id": "idle_days_capped"},
                    {"name": "Towers", "id": "towers"},
                ]
                if not df.empty:
                    df["avg_prod_mt_day"] = pd.to_numeric(df["avg_prod_mt_day"], errors="coerce").round(2)
                    df["idle_days_capped"] = pd.to_numeric(df["idle_days_capped"], errors="coerce").round(1)
            else:
                columns = [
                    {"name": "Project", "id": "project_name"},
                    {"name": "Gangs", "id": "gangs"},
                    {"name": "Towers", "id": "towers"},
                    {"name": "Idle Days", "id": "idle_days"},
                    {"name": "Idle Days / 100 Towers", "id": "idle_days_per_100"},
                ]
                if not df.empty and "idle_days_per_100" in df.columns:
                    df["idle_days_per_100"] = pd.to_numeric(df["idle_days_per_100"], errors="coerce").round(1)
            definition = (
                "Idle days per 100 towers = total idle days / towers * 100. "
                "Towers are counted from completion-date rows."
            )
            return columns, df.to_dict("records"), definition

        if kind == "pareto":
            pareto = payload.get("pareto") or {}
            df = pd.DataFrame(
                [
                    {
                        "Metric": "Top 20% deployment output share",
                        "Output Share (%)": float(pareto.get("top20_share", 0.0)),
                    },
                    {
                        "Metric": "Top 10% deployment output share",
                        "Output Share (%)": float(pareto.get("top10_share", 0.0)),
                    },
                ]
            )
            columns = [
                {"name": "Metric", "id": "Metric"},
                {"name": "Output Share (%)", "id": "Output Share (%)"},
            ]
            definition = "Pareto share uses deployment output sorted descending within the selected window."
            df["Output Share (%)"] = pd.to_numeric(df["Output Share (%)"], errors="coerce").round(1)
            return columns, df.to_dict("records"), definition

        return [], [], ""

    @app.callback(
        Output("analytics-payload", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("analytics-refresh-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def _compute_analytics_payload(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        _tick: int | None,
    ) -> dict[str, Any]:
        data_selector_local = DATA_SELECTOR
        if data_selector_local is None:
            return build_analytics_payload(pd.DataFrame())
        project_list = _normalize_str_list(_ensure_list(projects))
        gang_list = _normalize_str_list(_ensure_list(gangs))
        month_list = _normalize_str_list(_ensure_list(months))
        months_ts = resolve_months(month_list, quick_range)
        df_day = data_selector_local.select("erection")
        scoped = apply_filters(df_day, project_list, months_ts, gang_list)
        data_stamp = _analytics_data_stamp(config.data_path)
        cache_key = _analytics_cache_key(project_list, months_ts, gang_list, data_stamp)
        if _ANALYTICS_CACHE is not None:
            cached = _ANALYTICS_CACHE.get(cache_key)
            if cached is not None:
                return cached
        payload = build_analytics_payload(
            scoped,
            idle_cap_days=IDLE_CAP_DAYS,
            min_erections=MIN_ERECTIONS_FOR_TIERS,
        )
        scope_start = pd.to_datetime(scoped.get("date"), errors="coerce").min() if "date" in scoped.columns else pd.NaT
        scope_end = pd.to_datetime(scoped.get("date"), errors="coerce").max() if "date" in scoped.columns else pd.NaT
        scope_projects = int(scoped["project_name"].nunique()) if "project_name" in scoped.columns else 0
        scope_gangs = int(scoped["gang_name"].nunique()) if "gang_name" in scoped.columns else 0
        scope_gang_months = int((payload.get("whatif_base_inputs") or {}).get("total_gang_months", 0))
        payload["scope"] = {
            "start": scope_start.strftime("%Y-%m-%d") if pd.notna(scope_start) else "",
            "end": scope_end.strftime("%Y-%m-%d") if pd.notna(scope_end) else "",
            "projects": scope_projects,
            "gangs": scope_gangs,
            "gang_months": scope_gang_months,
        }
        payload["meta"] = {
            "projects": project_list,
            "gangs": gang_list,
            "months": [ts.strftime("%Y-%m") for ts in months_ts],
            "quick_range": quick_range or "",
            "data_stamp": data_stamp,
        }
        if _ANALYTICS_CACHE is not None:
            _ANALYTICS_CACHE.set(cache_key, payload, expire=_ANALYTICS_CACHE_TTL_SECONDS)
        return payload

    @app.callback(
        Output("analytics-scope-range", "children"),
        Output("analytics-scope-projects", "children"),
        Output("analytics-scope-gangs", "children"),
        Output("analytics-scope-gangmonths", "children"),
        Output("analytics-lowshare-scope", "children"),
        Input("analytics-payload", "data"),
    )
    def _render_scope_chips(payload: dict[str, Any] | None):
        scope = (payload or {}).get("scope") or {}
        start_raw = scope.get("start") or ""
        end_raw = scope.get("end") or ""
        start_ts = pd.to_datetime(start_raw, errors="coerce")
        end_ts = pd.to_datetime(end_raw, errors="coerce")

        if pd.notna(start_ts) and pd.notna(end_ts):
            range_label = f"Scope: {start_ts.strftime('%d %b %Y')} – {end_ts.strftime('%d %b %Y')}"
            scope_short = "Scope: {}–{}".format(
                start_ts.strftime("%b'%y"),
                end_ts.strftime("%b'%y"),
            )
        else:
            range_label = "Scope: N/A"
            scope_short = "Scope: N/A"

        projects = int(scope.get("projects", 0) or 0)
        gangs = int(scope.get("gangs", 0) or 0)
        gang_months = int(scope.get("gang_months", 0) or 0)
        return (
            range_label,
            f"Projects: {projects}",
            f"Gangs: {gangs}",
            f"Gang periods: {gang_months}",
            scope_short,
        )

    @app.callback(
        Output("analytics-kpi-low-output-value", "children"),
        Output("analytics-kpi-low-output-sub", "children"),
        Output("analytics-kpi-idle-value", "children"),
        Output("analytics-kpi-idle-sub", "children"),
        Output("analytics-kpi-hotspot-value", "children"),
        Output("analytics-kpi-hotspot-sub", "children"),
        Input("analytics-payload", "data"),
    )
    def _render_analytics_kpis(payload: dict[str, Any] | None):
        if not payload or not payload.get("kpis"):
            return "N/A", "", "N/A", "", "N/A", ""
        kpis = payload["kpis"]
        low_share = float(kpis.get("low_output_resources_share", 0.0)) * 100.0
        low_out = float(kpis.get("low_output_output_share", 0.0)) * 100.0
        low_value = f"{low_share:.0f}% resources -> {low_out:.0f}% output"
        low_sub = "Share of deployments vs MT output"

        high_idle = float(kpis.get("idle_windows_high", 0.0))
        low_idle = float(kpis.get("idle_windows_low", 0.0))
        idle_value = f"{high_idle:.1f} vs {low_idle:.1f} idle windows/gang"
        idle_sub = f"High (>{PRODUCTIVITY_TIER_HIGH:g} MT/day) vs Low (<{PRODUCTIVITY_TIER_LOW:g})"

        top_project = str(kpis.get("top_hotspot_project") or "")
        top_value = float(kpis.get("top_hotspot_value", 0.0))
        next_value = float(kpis.get("next_hotspot_value", 0.0))
        if top_project:
            hotspot_value = f"{top_project}: {top_value:.0f} idle-days/100 towers"
            hotspot_sub = f"Next highest (>=10 gangs): {next_value:.0f}"
        else:
            hotspot_value = "No hotspot found"
            hotspot_sub = ""
        return low_value, low_sub, idle_value, idle_sub, hotspot_value, hotspot_sub

    @app.callback(
        Output("analytics-lowshare-chart", "figure"),
        Output("analytics-lowshare-value", "children"),
        Output("analytics-lowshare-delta", "children"),
        Input("analytics-payload", "data"),
    )
    def _render_lowshare_card(payload: dict[str, Any] | None):
        trends = (payload or {}).get("trends") or {}
        low_rows = trends.get("low_bucket") or []
        if len(low_rows) < 3:
            fig = _analytics_empty_fig("Insufficient history")
            return fig, "N/A", ""

        low_months = [row.get("month") for row in low_rows]
        low_values = [float(row.get("pct_low_bucket", 0.0)) for row in low_rows]
        low_last = low_values[-1]
        low_prev = low_values[-2] if len(low_values) > 1 else low_last
        delta = low_last - low_prev
        delta_text = f"{delta:+.0f} pp"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=low_months,
                y=low_values,
                mode="lines+markers",
                line={"color": "#2563eb", "width": 3},
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.12)",
                marker={"size": 6},
                hovertemplate="%{y:.1f}%<extra></extra>",
            )
        )
        fig.add_annotation(
            x=low_months[-1],
            y=low_values[-1],
            text=f"{low_last:.0f}%",
            showarrow=True,
            arrowhead=2,
            ax=12,
            ay=-18,
            font={"size": 11, "color": "#1e3a8a"},
        )
        mid_index = len(low_months) // 2
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            xaxis={
                "tickvals": [low_months[0], low_months[mid_index], low_months[-1]],
                "ticktext": [
                    pd.to_datetime(low_months[0]).strftime("%b'%y"),
                    pd.to_datetime(low_months[mid_index]).strftime("%b'%y"),
                    pd.to_datetime(low_months[-1]).strftime("%b'%y"),
                ],
                "tickfont": {"size": 10},
            },
            yaxis={"gridcolor": "#e6e9f0", "tickfont": {"size": 10}},
            showlegend=False,
        )

        return fig, f"{low_last:.0f}%", delta_text

    @app.callback(
        Output("analytics-hotspot-chart", "figure"),
        Input("analytics-payload", "data"),
    )
    def _render_hotspot_ranking(payload: dict[str, Any] | None):
        rows = (payload or {}).get("hotspot", {}).get("top10") or []
        if not rows:
            return _analytics_empty_fig("No hotspot data")

        df = pd.DataFrame(rows)
        df["idle_days_per_100"] = pd.to_numeric(df["idle_days_per_100"], errors="coerce").fillna(0.0)
        df = df.sort_values("idle_days_per_100", ascending=True)
        fig = go.Figure()
        fig.add_bar(
            x=df["idle_days_per_100"],
            y=df["project_name"],
            orientation="h",
            marker_color="#ef4444",
            hovertemplate="%{x:.1f}<extra></extra>",
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 120, "r": 20, "t": 20, "b": 30},
            xaxis={"title": "Idle Days / 100 Towers", "gridcolor": "#e6e9f0"},
            yaxis={"title": "", "tickfont": {"size": 11}},
        )
        return fig

    @app.callback(
        Output("analytics-whatif-bucket", "options"),
        Output("analytics-whatif-bucket", "value"),
        Output("analytics-whatif-slider", "value"),
        Output("analytics-whatif-slider", "disabled"),
        Output("analytics-whatif-status", "children"),
        Input("analytics-payload", "data"),
        Input("analytics-whatif-bucket", "value"),
        Input("analytics-whatif-reset", "n_clicks"),
    )
    def _sync_whatif_controls(
        payload: dict[str, Any] | None,
        current_bucket: str | None,
        reset_clicks: int | None,
    ):
        summary_rows = (payload or {}).get("bucket", {}).get("summary") or []
        if not summary_rows:
            return [], None, 4, True, "No bucket data in selected window"

        options = [{"label": row.get("bucket_label"), "value": row.get("bucket_label")} for row in summary_rows]
        labels = [row.get("bucket_label") for row in summary_rows]
        counts_map = {row.get("bucket_label"): float(row.get("gang_months", 0.0)) for row in summary_rows}
        default_bucket = None
        if "0-4" in labels and counts_map.get("0-4", 0.0) > 0:
            default_bucket = "0-4"
        if default_bucket is None:
            for label in labels:
                if counts_map.get(label, 0.0) > 0:
                    default_bucket = label
                    break
        if default_bucket is None:
            default_bucket = "0-4" if "0-4" in labels else (labels[0] if labels else None)
        bucket_value = current_bucket if current_bucket in labels else default_bucket

        bucket_map = {row.get("bucket_label"): row for row in summary_rows}
        bucket_row = bucket_map.get(bucket_value) if bucket_value else None
        gang_months = float(bucket_row.get("gang_months", 0.0)) if bucket_row else 0.0
        avg_mt_day = float(bucket_row.get("avg_mt_day", 0.0)) if bucket_row else 0.0

        if gang_months <= 0:
            return options, bucket_value, 4, True, "No bucket data in selected window"

        def _round_to_half(value: float) -> float:
            return round(value * 2) / 2.0

        slider_value = _round_to_half(avg_mt_day)
        slider_value = max(2.0, min(20.0, slider_value))

        total_gang_months = float((payload or {}).get("whatif_base_inputs", {}).get("total_gang_months", 0.0))
        share_text = ""
        if total_gang_months > 0:
            share = gang_months / total_gang_months * 100.0
            share_text = f"Selected bucket share: {share:.0f}% of gang periods"

        return options, bucket_value, slider_value, False, share_text

    @app.callback(
        Output("analytics-whatif-reduction", "children"),
        Output("analytics-whatif-saved", "children"),
        Output("analytics-whatif-chart", "figure"),
        Input("analytics-payload", "data"),
        Input("analytics-whatif-bucket", "value"),
        Input("analytics-whatif-slider", "value"),
    )
    def _render_whatif_outputs(
        payload: dict[str, Any] | None,
        bucket_value: str | None,
        target_value: int | float | None,
    ):
        summary_rows = (payload or {}).get("bucket", {}).get("summary") or []
        bucket_map = {row.get("bucket_label"): row for row in summary_rows}
        bucket_row = bucket_map.get(bucket_value)
        if not bucket_row:
            return "0%", "0", _analytics_empty_fig("No data")

        total_inputs = (payload or {}).get("whatif_base_inputs", {})
        total_output = float(total_inputs.get("total_output", 0.0))
        total_gm = float(total_inputs.get("total_gang_months", 0.0))

        n_bucket = float(bucket_row.get("gang_months", 0.0))
        total_bucket = float(bucket_row.get("mt_total", 0.0))
        current_avg = float(bucket_row.get("avg_mt_day", 0.0))
        avg_active_days = float(bucket_row.get("avg_active_days", 0.0))
        target_avg = float(target_value or 0.0)

        if n_bucket <= 0 or total_gm <= 0 or total_output <= 0 or target_avg <= 0 or avg_active_days <= 0:
            return "0%", "0", _analytics_empty_fig("No data")

        n_bucket_new = total_bucket / (target_avg * avg_active_days) if target_avg > 0 else n_bucket
        n_new = (total_gm - n_bucket) + n_bucket_new
        saved = total_gm - n_new
        reduction_pct = (saved / total_gm * 100.0) if total_gm else 0.0

        reduction_text = f"{reduction_pct:.0f}%"
        saved_text = f"{saved:.0f}"

        fig = go.Figure()
        fig.add_bar(
            x=["Current Avg", "Target Avg"],
            y=[current_avg, target_avg],
            marker_color=["#94a3b8", "#22c55e"],
            text=[f"{current_avg:.1f} MT/day", f"{target_avg:.1f} MT/day"],
            textposition="outside",
            hovertemplate="%{y:.1f} MT/day<extra></extra>",
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 20, "r": 10, "t": 10, "b": 20},
            xaxis={"title": "", "tickangle": 0},
            yaxis={"title": "MT/day"},
        )
        return reduction_text, saved_text, fig

    @app.callback(
        Output("analytics-hist-median", "children"),
        Output("analytics-hist-pct-low", "children"),
        Output("analytics-hist-pct-high", "children"),
        Input("analytics-payload", "data"),
    )
    def _render_hist_tiles(payload: dict[str, Any] | None):
        histogram = (payload or {}).get("histogram", {})
        median = float(histogram.get("median_prod", 0.0))
        pct_low = float(histogram.get("pct_below_low", 0.0))
        pct_high = float(histogram.get("pct_above_high", 0.0))
        return f"{median:.2f}", f"{pct_low:.0f}%", f"{pct_high:.0f}%"

    @app.callback(
        Output("analytics-bucket-chart", "figure"),
        Output("analytics-tier-chart", "figure"),
        Output("analytics-hist-chart", "figure"),
        Input("analytics-payload", "data"),
    )
    def _render_analytics_charts(payload: dict[str, Any] | None):
        if not payload:
            empty = _analytics_empty_fig("No data")
            return empty, empty, empty

        bucket_rows = payload.get("bucket", {}).get("summary") or []
        if not bucket_rows:
            bucket_fig = _analytics_empty_fig("No bucket data")
        else:
            bucket_order = ["0-4", "4-6", "6-8", "8-10", "10-12", "12+"]
            bucket_map = {str(row.get("bucket_label") or ""): row for row in bucket_rows}
            labels = [label for label in bucket_order if label in bucket_map]
            if not labels:
                labels = [str(row.get("bucket_label") or "") for row in bucket_rows]
            gang_share = [_safe_float(bucket_map.get(label, {}).get("gang_month_share", 0.0)) * 100.0 for label in labels]
            mt_share = [_safe_float(bucket_map.get(label, {}).get("mt_share", 0.0)) * 100.0 for label in labels]
            if sum(gang_share) <= 0.0 and sum(mt_share) <= 0.0:
                bucket_fig = _analytics_empty_fig("No bucket data")
            else:
                bucket_fig = go.Figure()
                bucket_fig.add_bar(
                    x=labels,
                    y=gang_share,
                    name="Deployment Share",
                    marker_color="#2563eb",
                    hovertemplate="%{y:.1f}%<extra></extra>",
                )
                bucket_fig.add_bar(
                    x=labels,
                    y=mt_share,
                    name="MT Output Share",
                    marker_color="#22c55e",
                    hovertemplate="%{y:.1f}%<extra></extra>",
                )
            bucket_layout = _analytics_chart_layout("Bucket", "Share (%)")
            bucket_layout["xaxis"] = {**bucket_layout.get("xaxis", {}), "type": "category"}
            bucket_fig.update_layout(
                barmode="group",
                **bucket_layout,
            )

        tier_rows = payload.get("tiers", {}).get("summary") or []
        if not tier_rows:
            tier_fig = _analytics_empty_fig("No tier data")
        else:
            tier_order = [
                f"Low (<{PRODUCTIVITY_TIER_LOW:g})",
                f"Mid ({PRODUCTIVITY_TIER_LOW:g}-{PRODUCTIVITY_TIER_HIGH:g})",
                f"High (>{PRODUCTIVITY_TIER_HIGH:g})",
            ]
            tier_map = {str(row.get("tier") or ""): row for row in tier_rows}
            tier_labels = [label for label in tier_order if label in tier_map]
            if not tier_labels:
                tier_labels = [str(row.get("tier") or "") for row in tier_rows]
            tier_windows = [_safe_float(tier_map.get(label, {}).get("avg_idle_windows", 0.0)) for label in tier_labels]
            tier_days = [_safe_float(tier_map.get(label, {}).get("avg_idle_days", 0.0)) for label in tier_labels]
            if sum(tier_windows) <= 0.0 and sum(tier_days) <= 0.0:
                tier_fig = _analytics_empty_fig("No tier data")
            else:
                tier_fig = go.Figure()
                tier_fig.add_bar(
                    x=tier_labels,
                    y=tier_windows,
                    name="Avg Idle Windows",
                    marker_color="#f97316",
                    text=[f"{value:.1f}" for value in tier_windows],
                    textposition="outside",
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
                tier_fig.add_bar(
                    x=tier_labels,
                    y=tier_days,
                    name="Avg Idle Days",
                    marker_color="#38bdf8",
                    text=[f"{value:.1f}" for value in tier_days],
                    textposition="outside",
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
                tier_fig.update_layout(
                    barmode="group",
                    **_analytics_chart_layout("Tier", "Avg Idle Windows / Days"),
                )

        hist_rows = payload.get("histogram", {}).get("bins") or []
        if not hist_rows:
            hist_fig = _analytics_empty_fig("No histogram data")
        else:
            hist_labels = [str(row.get("bin_label") or "") for row in hist_rows]
            hist_values = [_safe_float(row.get("count", 0.0)) for row in hist_rows]
            if sum(hist_values) <= 0.0:
                hist_fig = _analytics_empty_fig("No histogram data")
            else:
                hist_fig = go.Figure()
                hist_fig.add_bar(
                    x=hist_labels,
                    y=hist_values,
                    marker_color="#0f766e",
                    hovertemplate="%{y} gangs<extra></extra>",
                )
                hist_fig.update_layout(**_analytics_chart_layout("MT/day Bin", "Gangs"))
                hist_fig.update_xaxes(type="category")

        return bucket_fig, tier_fig, hist_fig

    @app.callback(
        Output("analytics-audit-drawer", "is_open"),
        Output("analytics-audit-selection", "data"),
        Output("analytics-audit-title", "children"),
        Input("analytics-kpi-low-output", "n_clicks"),
        Input("analytics-kpi-idle-windows", "n_clicks"),
        Input("analytics-kpi-hotspot", "n_clicks"),
        Input("analytics-bucket-chart", "clickData"),
        Input("analytics-tier-chart", "clickData"),
        Input("analytics-hist-chart", "clickData"),
        Input("analytics-hotspot-chart", "clickData"),
        Input("analytics-audit-close", "n_clicks"),
        State("analytics-payload", "data"),
        State("analytics-audit-drawer", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_analytics_audit(
        low_clicks,
        idle_clicks,
        hotspot_clicks,
        bucket_click,
        tier_click,
        hist_click,
        hotspot_chart_click,
        close_clicks,
        payload,
        is_open,
    ):
        trigger = _resolve_triggered_id()
        if trigger == "analytics-audit-close":
            return False, dash.no_update, dash.no_update
        if payload is None:
            raise PreventUpdate

        selection: dict[str, Any] | None = None
        title = "Audit"
        if trigger == "analytics-kpi-low-output":
            selection = {"kind": "bucket", "bucket": "0-4"}
            title = "Low-output Deployment"
        elif trigger == "analytics-kpi-idle-windows":
            selection = {
                "kind": "idle_windows",
                "tiers": [
                    f"High (>{PRODUCTIVITY_TIER_HIGH:g})",
                    f"Low (<{PRODUCTIVITY_TIER_LOW:g})",
                ],
            }
            title = "Idle Windows: High vs Low"
        elif trigger == "analytics-kpi-hotspot":
            top_project = (payload.get("kpis") or {}).get("top_hotspot_project") or ""
            selection = {"kind": "hotspot", "project": top_project or None}
            title = "Project Hotspot"
        elif trigger == "analytics-bucket-chart":
            bucket_label = (bucket_click or {}).get("points", [{}])[0].get("x")
            if not bucket_label:
                raise PreventUpdate
            selection = {"kind": "bucket", "bucket": bucket_label}
            title = f"Bucket: {bucket_label}"
        elif trigger == "analytics-tier-chart":
            tier_label = (tier_click or {}).get("points", [{}])[0].get("x")
            if not tier_label:
                raise PreventUpdate
            selection = {"kind": "tier", "tier": tier_label}
            title = f"Tier: {tier_label}"
        elif trigger == "analytics-hist-chart":
            bin_label = (hist_click or {}).get("points", [{}])[0].get("x")
            if not bin_label:
                raise PreventUpdate
            selection = {"kind": "hist_bin", "bin": bin_label}
            title = f"Histogram Bin: {bin_label}"
        elif trigger == "analytics-hotspot-chart":
            project_label = (hotspot_chart_click or {}).get("points", [{}])[0].get("y")
            if not project_label:
                raise PreventUpdate
            selection = {"kind": "hotspot", "project": project_label}
            title = f"Project Hotspot: {project_label}"
        else:
            raise PreventUpdate

        return True, selection, title

    @app.callback(
        Output("analytics-audit-table", "columns"),
        Output("analytics-audit-table", "data"),
        Output("analytics-audit-definition", "children"),
        Input("analytics-payload", "data"),
        Input("analytics-audit-selection", "data"),
    )
    def _render_analytics_audit_table(payload: dict[str, Any] | None, selection: dict[str, Any] | None):
        columns, data, definition = _analytics_table_from_selection(payload, selection)
        return columns, data, definition

    @app.callback(
        Output("analytics-audit-download", "data"),
        Input("analytics-audit-export-btn", "n_clicks"),
        State("analytics-payload", "data"),
        State("analytics-audit-selection", "data"),
        prevent_initial_call=True,
    )
    def _export_analytics_audit(
        export_clicks: int | None,
        payload: dict[str, Any] | None,
        selection: dict[str, Any] | None,
    ):
        if not export_clicks:
            raise PreventUpdate
        columns, data, _definition = _analytics_table_from_selection(payload, selection)
        if not columns:
            raise PreventUpdate
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame(columns=[col["name"] for col in columns])

        def _writer(buffer: BytesIO) -> None:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Audit", index=False)

        return send_bytes(_writer, "Analytics_Audit.xlsx")

    # --- Stringing analytics ---
    def _stringing_analytics_table_from_selection(
        payload: dict[str, Any] | None,
        selection: dict[str, Any] | None,
        section_filter: str | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], str]:
        if not payload or not selection:
            return [], [], ""
        kind = selection.get("kind")

        def _format_dates(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
            for col in columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            return df

        def _readiness_bucket_label(value: float | int | None) -> str:
            if value is None or pd.isna(value):
                return ""
            if value <= 15:
                return "0-15"
            if value <= 30:
                return "16-30"
            if value <= 60:
                return "31-60"
            if value <= 90:
                return "61-90"
            return ">90"

        def _flow_bucket_label(value: float | int | None) -> str:
            if value is None or pd.isna(value):
                return ""
            if value <= 3:
                return "0-3"
            if value <= 7:
                return "4-7"
            if value <= 14:
                return "8-14"
            return ">14"

        def _cycle_bucket_label(value: float | int | None) -> str:
            if value is None or pd.isna(value):
                return ""
            if value <= 30:
                return "0-30"
            if value <= 60:
                return "31-60"
            if value <= 90:
                return "61-90"
            return ">90"

        if kind in {"readiness_gap", "readiness_bucket", "readiness_project"}:
            rows = (payload.get("readiness") or {}).get("gaps") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return [], [], ""
            if kind == "readiness_bucket":
                bucket = selection.get("bucket")
                df["bucket"] = pd.to_numeric(df.get("gap_days"), errors="coerce").map(_readiness_bucket_label)
                df = df[df["bucket"] == bucket]
            if kind == "readiness_project":
                project = selection.get("project")
                if project:
                    df = df[df.get("project_name") == project]
            if section_filter:
                df = df[df.get("section") == section_filter]
            df = _format_dates(df, ["po_start_date", "last_erection_completion_date"])
            columns = [
                {"name": "Project", "id": "project_name"},
                {"name": "Section", "id": "section"},
                {"name": "Span", "id": "span"},
                {"name": "Gang", "id": "gang_name"},
                {"name": "Erection complete", "id": "last_erection_completion_date"},
                {"name": "P/O start", "id": "po_start_date"},
                {"name": "Gap (days)", "id": "gap_days"},
            ]
            definition = (
                "Readiness delay = P/O start date minus last erection completion within the span. "
                "TSE-only spans; manual excluded. Negative gaps are retained and should be reviewed."
            )
            return columns, df.to_dict("records"), definition

        if kind in {"flow_gap", "flow_bucket"}:
            rows = (payload.get("flow") or {}).get("gaps") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return [], [], ""
            if kind == "flow_bucket":
                bucket = selection.get("bucket")
                df["bucket"] = pd.to_numeric(df.get("gap_days"), errors="coerce").map(_flow_bucket_label)
                df = df[df["bucket"] == bucket]
            if section_filter:
                df = df[df.get("section") == section_filter]
            df = _format_dates(df, ["po_completion_date", "fs_starting_date"])
            columns = [
                {"name": "Project", "id": "project_name"},
                {"name": "Section", "id": "section"},
                {"name": "Span", "id": "span"},
                {"name": "Gang", "id": "gang_name"},
                {"name": "P/O complete", "id": "po_completion_date"},
                {"name": "Sag start", "id": "fs_starting_date"},
                {"name": "Gap (days)", "id": "gap_days"},
            ]
            definition = (
                "Flow delay = sag start date minus P/O completion date. "
                "TSE-only spans; manual excluded."
            )
            return columns, df.to_dict("records"), definition

        if kind in {"cycle_gap", "cycle_bucket"}:
            rows = (payload.get("cycle") or {}).get("gaps") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return [], [], ""
            if kind == "cycle_bucket":
                bucket = selection.get("bucket")
                df["bucket"] = pd.to_numeric(df.get("cycle_days"), errors="coerce").map(_cycle_bucket_label)
                df = df[df["bucket"] == bucket]
            if section_filter:
                df = df[df.get("section") == section_filter]
            df = _format_dates(df, ["last_erection_completion_date", "sag_end_date"])
            columns = [
                {"name": "Project", "id": "project_name"},
                {"name": "Section", "id": "section"},
                {"name": "Span", "id": "span"},
                {"name": "Erection complete", "id": "last_erection_completion_date"},
                {"name": "Sag end", "id": "sag_end_date"},
                {"name": "Cycle days", "id": "cycle_days"},
            ]
            definition = (
                "End-to-end cycle time = sag end (complete or start) minus last erection completion within span. "
                "TSE-only spans; manual excluded."
            )
            return columns, df.to_dict("records"), definition

        if kind == "productivity_bucket":
            rows = (payload.get("productivity") or {}).get("gangs") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return [], [], ""
            bucket = selection.get("bucket")
            if bucket:
                df = df[df.get("bucket") == bucket]
            columns = [
                {"name": "Gang", "id": "gang_name"},
                {"name": "Avg km/month", "id": "avg_km_month"},
                {"name": "Total km", "id": "total_km"},
                {"name": "Active days", "id": "active_days"},
                {"name": "Spans", "id": "spans"},
                {"name": "Projects", "id": "projects"},
            ]
            definition = (
                "Productivity uses average daily km per gang across the selected period, scaled to km/month. "
                "TSE-only gangs; manual excluded."
            )
            return columns, df.to_dict("records"), definition

        if kind == "whatif":
            rows = (payload.get("productivity") or {}).get("gang_months", {}).get("rows") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return [], [], ""
            bucket = selection.get("bucket")
            if bucket:
                df = df[df.get("bucket") == bucket]
            df["month"] = pd.to_datetime(df.get("month"), errors="coerce").dt.strftime("%Y-%m")
            columns = [
                {"name": "Gang", "id": "gang_name"},
                {"name": "Month", "id": "month"},
                {"name": "KM (month)", "id": "total_km"},
                {"name": "Bucket", "id": "bucket"},
            ]
            definition = (
                "What-if uses gang-month totals (sum of daily km per gang per calendar month). "
                "Illustrative only; assumes output constant."
            )
            return columns, df.to_dict("records"), definition

        if kind == "ageing":
            rows = (payload.get("cycle") or {}).get("ageing") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return [], [], ""
            columns = [
                {"name": "Project", "id": "project_name"},
                {"name": "Section", "id": "section"},
                {"name": "Span", "id": "span"},
                {"name": "Stage", "id": "current_stage"},
                {"name": "Ageing days", "id": "ageing_days"},
            ]
            definition = (
                "Ageing shows top spans without sag completion, measured from the latest completed stage. "
                "TSE-only spans; manual excluded."
            )
            return columns, df.to_dict("records"), definition

        if kind == "relationship":
            rows = (payload.get("relationship") or {}).get("summary") or []
            df = pd.DataFrame(rows)
            columns = [
                {"name": "Readiness bucket", "id": "bucket"},
                {"name": "Avg productivity (km/month)", "id": "avg_km_month"},
                {"name": "Spans", "id": "spans"},
            ]
            definition = (
                "Readiness vs productivity groups spans by readiness bucket and averages gang productivity. "
                "TSE-only spans; manual excluded."
            )
            return columns, df.to_dict("records"), definition

        return [], [], ""

    @app.callback(
        Output("stringing-analytics-payload", "data"),
        Input("f-project", "value"),
        Input("f-month", "value"),
        Input("f-quick-range", "value"),
        Input("f-gang", "value"),
        Input("stringing-analytics-refresh-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def _compute_stringing_analytics_payload(
        projects: Sequence[str] | None,
        months: Sequence[str] | None,
        quick_range: str | None,
        gangs: Sequence[str] | None,
        _tick: int | None,
    ) -> dict[str, Any]:
        data_selector_local = DATA_SELECTOR
        if data_selector_local is None:
            return build_stringing_analytics_payload(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        project_list = _normalize_str_list(_ensure_list(projects))
        gang_list = _normalize_str_list(_ensure_list(gangs))
        month_list = _normalize_str_list(_ensure_list(months))
        months_ts = resolve_months(month_list, quick_range)
        daily = data_selector_local.select("stringing")
        erection_daily = data_selector_local.select("erection")
        compiled = pd.DataFrame()
        if callable(stringing_compiled_provider):
            try:
                compiled = stringing_compiled_provider() or pd.DataFrame()
            except Exception:
                compiled = pd.DataFrame()
        if compiled.empty:
            try:
                compiled = _load_stringing_compiled_raw(config.stringing_data_path)
            except Exception:
                compiled = pd.DataFrame()
        data_stamp = _analytics_data_stamp(config.stringing_data_path)
        cache_key = _stringing_analytics_cache_key(
            project_list,
            months_ts,
            gang_list,
            data_stamp,
            compiled_rows=len(compiled.index) if isinstance(compiled, pd.DataFrame) else 0,
        )
        if _ANALYTICS_CACHE is not None:
            cached = _ANALYTICS_CACHE.get(cache_key)
            if cached is not None:
                return cached
        payload = build_stringing_analytics_payload(
            daily,
            compiled,
            erection_daily,
            projects=project_list,
            months=months_ts,
            gangs=gang_list,
            method_filter="tse",
        )
        payload["meta"] = {
            "projects": project_list,
            "gangs": gang_list,
            "months": [ts.strftime("%Y-%m") for ts in months_ts],
            "quick_range": quick_range or "",
            "data_stamp": data_stamp,
        }
        if _ANALYTICS_CACHE is not None:
            _ANALYTICS_CACHE.set(cache_key, payload, expire=_ANALYTICS_CACHE_TTL_SECONDS)
        return payload

    @app.callback(
        Output("stringing-analytics-scope-range", "children"),
        Output("stringing-analytics-scope-projects", "children"),
        Output("stringing-analytics-scope-gangs", "children"),
        Output("stringing-analytics-scope-spans", "children"),
        Output("stringing-analytics-scope-totalkm", "children"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_stringing_scope_chips(payload: dict[str, Any] | None):
        scope = (payload or {}).get("scope") or {}
        start_raw = scope.get("start") or ""
        end_raw = scope.get("end") or ""
        start_ts = pd.to_datetime(start_raw, errors="coerce")
        end_ts = pd.to_datetime(end_raw, errors="coerce")
        if pd.notna(start_ts) and pd.notna(end_ts):
            range_label = f"Scope: {start_ts.strftime('%d %b %Y')} - {end_ts.strftime('%d %b %Y')}"
        else:
            range_label = "Scope: (empty)"
        projects = scope.get("projects", 0)
        gangs = scope.get("gangs", 0)
        spans = scope.get("spans", 0)
        total_km = scope.get("total_km", 0.0)
        return (
            range_label,
            f"Projects: {projects}",
            f"Gangs: {gangs}",
            f"Spans: {spans}",
            f"Total km: {total_km:.1f}",
        )

    @app.callback(
        Output("stringing-analytics-kpi-output", "children"),
        Output("stringing-analytics-kpi-output-sub", "children"),
        Output("stringing-analytics-kpi-readiness", "children"),
        Output("stringing-analytics-kpi-readiness-sub", "children"),
        Output("stringing-analytics-kpi-flow", "children"),
        Output("stringing-analytics-kpi-flow-sub", "children"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_stringing_kpis(payload: dict[str, Any] | None):
        kpis = (payload or {}).get("kpis") or {}
        output_km = float(kpis.get("output_km", 0.0))
        output_n = int(kpis.get("output_n", 0))
        readiness = float(kpis.get("readiness_median", 0.0))
        readiness_n = int(kpis.get("readiness_n", 0))
        flow = float(kpis.get("flow_median", 0.0))
        flow_n = int(kpis.get("flow_n", 0))
        return (
            f"{output_km:.1f}",
            f"n={output_n}",
            f"{readiness:.1f}",
            f"n={readiness_n}",
            f"{flow:.1f}",
            f"n={flow_n}",
        )

    @app.callback(
        Output("stringing-analytics-readiness-hist", "figure"),
        Output("stringing-analytics-readiness-pct-15", "children"),
        Output("stringing-analytics-readiness-pct-60", "children"),
        Output("stringing-analytics-readiness-median", "children"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_readiness_hist(payload: dict[str, Any] | None):
        readiness = (payload or {}).get("readiness") or {}
        stats = readiness.get("stats") or {}
        hist = readiness.get("histogram") or []
        if not hist:
            fig = _analytics_empty_fig("No readiness data")
        else:
            labels = [row.get("bucket") for row in hist]
            values = [int(row.get("count", 0)) for row in hist]
            if sum(values) <= 0:
                fig = _analytics_empty_fig("No readiness data")
            else:
                fig = go.Figure()
                fig.add_bar(
                    x=labels,
                    y=values,
                    marker_color="#0ea5e9",
                    hovertemplate="%{y} spans<extra></extra>",
                )
                fig.update_layout(**_analytics_chart_layout("Gap bucket (days)", "Spans"))
        return (
            fig,
            f"{float(stats.get('pct_over_15', 0.0)):.0f}%",
            f"{float(stats.get('pct_over_60', 0.0)):.0f}%",
            f"{float(stats.get('median', 0.0)):.1f}",
        )

    @app.callback(
        Output("stringing-analytics-readiness-hotspot", "figure"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_readiness_hotspot(payload: dict[str, Any] | None):
        rows = (payload or {}).get("readiness", {}).get("hotspots") or []
        if not rows:
            return _analytics_empty_fig("No hotspot data")
        df = pd.DataFrame(rows).sort_values("median_gap", ascending=True)
        fig = go.Figure()
        fig.add_bar(
            x=df["median_gap"],
            y=df["project_name"],
            orientation="h",
            marker_color="#ef4444",
            hovertemplate="%{x:.1f} days<extra></extra>",
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 120, "r": 20, "t": 20, "b": 30},
            xaxis={"title": "Median E->P/O gap (days)", "gridcolor": "#e6e9f0"},
            yaxis={"title": "", "tickfont": {"size": 11}},
        )
        return fig

    @app.callback(
        Output("stringing-analytics-readiness-funnel", "figure"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_readiness_funnel(payload: dict[str, Any] | None):
        rows = (payload or {}).get("readiness", {}).get("funnel") or []
        if not rows:
            return _analytics_empty_fig("No funnel data")
        labels = [row.get("stage") for row in rows]
        values = [int(row.get("count", 0)) for row in rows]
        fig = go.Figure(go.Funnel(y=labels, x=values, textinfo="value+percent initial"))
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 40, "r": 20, "t": 20, "b": 30},
        )
        return fig

    @app.callback(
        Output("stringing-analytics-prod-hist", "figure"),
        Output("stringing-analytics-prod-median", "children"),
        Output("stringing-analytics-prod-pct-low", "children"),
        Output("stringing-analytics-prod-pct-high", "children"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_productivity_hist(payload: dict[str, Any] | None):
        productivity = (payload or {}).get("productivity") or {}
        summary = productivity.get("summary") or {}
        hist = productivity.get("histogram") or []
        if not hist:
            fig = _analytics_empty_fig("No productivity data")
        else:
            labels = [row.get("bucket") for row in hist]
            values = [int(row.get("count", 0)) for row in hist]
            fig = go.Figure()
            fig.add_bar(
                x=labels,
                y=values,
                marker_color="#14b8a6",
                hovertemplate="%{y} gangs<extra></extra>",
            )
            fig.update_layout(**_analytics_chart_layout("KM/month bucket", "Gangs"))
        return (
            fig,
            f"{float(summary.get('median', 0.0)):.2f}",
            f"{float(summary.get('pct_below_3', 0.0)):.0f}%",
            f"{float(summary.get('pct_above_6', 0.0)):.0f}%",
        )

    @app.callback(
        Output("stringing-analytics-share-chart", "figure"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_productivity_share(payload: dict[str, Any] | None):
        rows = (payload or {}).get("productivity", {}).get("share") or []
        if not rows:
            return _analytics_empty_fig("No share data")
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(
            x=df["bucket"],
            y=df["gang_share"],
            name="Gang share",
            marker_color="#2563eb",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
        fig.add_bar(
            x=df["bucket"],
            y=df["km_share"],
            name="Output share",
            marker_color="#22c55e",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
        fig.update_layout(barmode="group", **_analytics_chart_layout("Bucket", "Share (%)"))
        return fig

    @app.callback(
        Output("stringing-analytics-whatif-bucket", "options"),
        Output("stringing-analytics-whatif-bucket", "value"),
        Output("stringing-analytics-whatif-slider", "value"),
        Output("stringing-analytics-whatif-slider", "disabled"),
        Output("stringing-analytics-whatif-status", "children"),
        Input("stringing-analytics-payload", "data"),
        Input("stringing-analytics-whatif-bucket", "value"),
        Input("stringing-analytics-whatif-reset", "n_clicks"),
    )
    def _sync_stringing_whatif_controls(
        payload: dict[str, Any] | None,
        current_bucket: str | None,
        reset_clicks: int | None,
    ):
        summary_rows = (payload or {}).get("productivity", {}).get("gang_months", {}).get("summary") or []
        if not summary_rows:
            return [], None, 4, True, "No bucket data"
        options = [{"label": row.get("bucket"), "value": row.get("bucket")} for row in summary_rows]
        labels = [row.get("bucket") for row in summary_rows]
        counts_map = {row.get("bucket"): float(row.get("gang_months", 0.0)) for row in summary_rows}
        default_bucket = None
        if "0-2" in labels and counts_map.get("0-2", 0.0) > 0:
            default_bucket = "0-2"
        if default_bucket is None:
            for label in labels:
                if counts_map.get(label, 0.0) > 0:
                    default_bucket = label
                    break
        bucket_value = current_bucket if current_bucket in labels else default_bucket
        bucket_map = {row.get("bucket"): row for row in summary_rows}
        bucket_row = bucket_map.get(bucket_value) if bucket_value else None
        avg_km = float(bucket_row.get("avg_km", 0.0)) if bucket_row else 0.0
        disabled = float(bucket_row.get("gang_months", 0.0)) <= 0 if bucket_row else True
        status = ""
        total_gm = float((payload or {}).get("productivity", {}).get("gang_months", {}).get("total_gang_months", 0.0))
        if total_gm > 0 and bucket_row:
            share = float(bucket_row.get("gang_months", 0.0)) / total_gm * 100.0
            status = f"Selected bucket share: {share:.0f}% of gang periods"
        return options, bucket_value, round(avg_km or 4, 1), disabled, status

    @app.callback(
        Output("stringing-analytics-whatif-saved", "children"),
        Output("stringing-analytics-whatif-unlocked", "children"),
        Output("stringing-analytics-whatif-chart", "figure"),
        Input("stringing-analytics-payload", "data"),
        Input("stringing-analytics-whatif-bucket", "value"),
        Input("stringing-analytics-whatif-slider", "value"),
    )
    def _render_stringing_whatif_outputs(
        payload: dict[str, Any] | None,
        bucket_value: str | None,
        target_value: int | float | None,
    ):
        gang_months = (payload or {}).get("productivity", {}).get("gang_months", {}) or {}
        summary_rows = gang_months.get("summary") or []
        bucket_map = {row.get("bucket"): row for row in summary_rows}
        bucket_row = bucket_map.get(bucket_value)
        if not bucket_row:
            return "0", "0", _analytics_empty_fig("No data")
        total_gm = float(gang_months.get("total_gang_months", 0.0))
        n_bucket = float(bucket_row.get("gang_months", 0.0))
        total_bucket = float(bucket_row.get("km_total", 0.0))
        current_avg = float(bucket_row.get("avg_km", 0.0))
        target_avg = float(target_value or 0.0)
        if n_bucket <= 0 or total_gm <= 0 or target_avg <= 0:
            return "0", "0", _analytics_empty_fig("No data")
        n_bucket_new = total_bucket / target_avg if target_avg > 0 else n_bucket
        n_new = (total_gm - n_bucket) + n_bucket_new
        saved = total_gm - n_new
        unlocked = max(target_avg - current_avg, 0.0) * n_bucket
        fig = go.Figure()
        fig.add_bar(
            x=["Current Avg", "Target Avg"],
            y=[current_avg, target_avg],
            marker_color=["#94a3b8", "#22c55e"],
            text=[f"{current_avg:.1f}", f"{target_avg:.1f}"],
            textposition="outside",
            hovertemplate="%{y:.1f} km/mo<extra></extra>",
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 20, "r": 10, "t": 10, "b": 20},
            xaxis={"title": ""},
            yaxis={"title": "KM/month"},
        )
        return f"{saved:.1f}", f"{unlocked:.1f}", fig

    @app.callback(
        Output("stringing-analytics-flow-hist", "figure"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_flow_hist(payload: dict[str, Any] | None):
        flow = (payload or {}).get("flow") or {}
        hist = flow.get("histogram") or []
        if not hist:
            return _analytics_empty_fig("No flow data")
        labels = [row.get("bucket") for row in hist]
        values = [int(row.get("count", 0)) for row in hist]
        if sum(values) <= 0:
            return _analytics_empty_fig("No flow data")
        fig = go.Figure()
        fig.add_bar(
            x=labels,
            y=values,
            marker_color="#f97316",
            hovertemplate="%{y} spans<extra></extra>",
        )
        fig.update_layout(**_analytics_chart_layout("Gap bucket (days)", "Spans"))
        return fig

    @app.callback(
        Output("stringing-analytics-cycle-chart", "figure"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_cycle_chart(payload: dict[str, Any] | None):
        cycle = (payload or {}).get("cycle") or {}
        hist = cycle.get("histogram") or []
        if not hist:
            return _analytics_empty_fig("No cycle data")
        labels = [row.get("bucket") for row in hist]
        values = [int(row.get("count", 0)) for row in hist]
        if sum(values) <= 0:
            return _analytics_empty_fig("No cycle data")
        fig = go.Figure()
        fig.add_bar(
            x=labels,
            y=values,
            marker_color="#6366f1",
            hovertemplate="%{y} spans<extra></extra>",
        )
        fig.update_layout(**_analytics_chart_layout("Cycle bucket (days)", "Spans"))
        return fig

    @app.callback(
        Output("stringing-analytics-ageing-table", "columns"),
        Output("stringing-analytics-ageing-table", "data"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_ageing_table(payload: dict[str, Any] | None):
        rows = (payload or {}).get("cycle", {}).get("ageing") or []
        df = pd.DataFrame(rows)
        columns = [
            {"name": "Project", "id": "project_name"},
            {"name": "Section", "id": "section"},
            {"name": "Span", "id": "span"},
            {"name": "Stage", "id": "current_stage"},
            {"name": "Ageing days", "id": "ageing_days"},
        ]
        return columns, df.to_dict("records")

    @app.callback(
        Output("stringing-analytics-relationship-chart", "figure"),
        Input("stringing-analytics-payload", "data"),
    )
    def _render_relationship_chart(payload: dict[str, Any] | None):
        rows = (payload or {}).get("relationship", {}).get("summary") or []
        if not rows:
            return _analytics_empty_fig("No relationship data")
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(
            x=df["bucket"],
            y=df["avg_km_month"],
            marker_color="#0f766e",
            hovertemplate="%{y:.2f} km/month<extra></extra>",
        )
        fig.update_layout(**_analytics_chart_layout("Readiness bucket", "Avg km/month"))
        return fig

    @app.callback(
        Output("stringing-analytics-audit-drawer", "is_open"),
        Output("stringing-analytics-audit-selection", "data"),
        Output("stringing-analytics-audit-title", "children"),
        Input("stringing-analytics-kpi-output-card", "n_clicks"),
        Input("stringing-analytics-kpi-readiness-card", "n_clicks"),
        Input("stringing-analytics-kpi-flow-card", "n_clicks"),
        Input("stringing-analytics-readiness-hist", "clickData"),
        Input("stringing-analytics-readiness-hotspot", "clickData"),
        Input("stringing-analytics-readiness-funnel", "clickData"),
        Input("stringing-analytics-prod-hist", "clickData"),
        Input("stringing-analytics-share-chart", "clickData"),
        Input("stringing-analytics-flow-hist", "clickData"),
        Input("stringing-analytics-cycle-chart", "clickData"),
        Input("stringing-analytics-relationship-chart", "clickData"),
        Input("stringing-analytics-audit-close", "n_clicks"),
        State("stringing-analytics-payload", "data"),
        State("stringing-analytics-audit-drawer", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_stringing_analytics_audit(
        output_click,
        readiness_click,
        flow_click,
        readiness_hist_click,
        readiness_hotspot_click,
        readiness_funnel_click,
        prod_hist_click,
        share_click,
        flow_hist_click,
        cycle_click,
        relationship_click,
        close_click,
        payload,
        is_open,
    ):
        trigger = _resolve_triggered_id()
        if trigger == "stringing-analytics-audit-close":
            return False, dash.no_update, dash.no_update
        if payload is None:
            raise PreventUpdate
        selection: dict[str, Any] | None = None
        title = "Audit"
        if trigger == "stringing-analytics-kpi-output-card":
            selection = {"kind": "productivity_bucket"}
            title = "Output (KM) by Gang"
        elif trigger == "stringing-analytics-kpi-readiness-card":
            selection = {"kind": "readiness_gap"}
            title = "Readiness Gap"
        elif trigger == "stringing-analytics-kpi-flow-card":
            selection = {"kind": "flow_gap"}
            title = "Flow Gap"
        elif trigger == "stringing-analytics-readiness-hist":
            bucket = (readiness_hist_click or {}).get("points", [{}])[0].get("x")
            selection = {"kind": "readiness_bucket", "bucket": bucket}
            title = f"Readiness Bucket: {bucket}"
        elif trigger == "stringing-analytics-readiness-hotspot":
            project = (readiness_hotspot_click or {}).get("points", [{}])[0].get("y")
            selection = {"kind": "readiness_project", "project": project}
            title = f"Readiness Hotspot: {project}"
        elif trigger == "stringing-analytics-readiness-funnel":
            selection = {"kind": "readiness_gap"}
            title = "Readiness Funnel"
        elif trigger == "stringing-analytics-prod-hist":
            bucket = (prod_hist_click or {}).get("points", [{}])[0].get("x")
            selection = {"kind": "productivity_bucket", "bucket": bucket}
            title = f"Productivity Bucket: {bucket}"
        elif trigger == "stringing-analytics-share-chart":
            bucket = (share_click or {}).get("points", [{}])[0].get("x")
            selection = {"kind": "productivity_bucket", "bucket": bucket}
            title = f"Share Bucket: {bucket}"
        elif trigger == "stringing-analytics-flow-hist":
            bucket = (flow_hist_click or {}).get("points", [{}])[0].get("x")
            selection = {"kind": "flow_bucket", "bucket": bucket}
            title = f"Flow Bucket: {bucket}"
        elif trigger == "stringing-analytics-cycle-chart":
            bucket = (cycle_click or {}).get("points", [{}])[0].get("x")
            selection = {"kind": "cycle_bucket", "bucket": bucket}
            title = f"Cycle Bucket: {bucket}"
        elif trigger == "stringing-analytics-relationship-chart":
            selection = {"kind": "relationship"}
            title = "Readiness vs Productivity"
        else:
            raise PreventUpdate
        return True, selection, title

    @app.callback(
        Output("stringing-analytics-section-filter", "options"),
        Output("stringing-analytics-section-filter", "value"),
        Input("stringing-analytics-audit-selection", "data"),
        State("stringing-analytics-payload", "data"),
    )
    def _sync_stringing_section_filter(selection: dict[str, Any] | None, payload: dict[str, Any] | None):
        if not selection or not payload:
            return [], None
        if selection.get("kind") != "readiness_project":
            return [], None
        project = selection.get("project")
        rows = (payload.get("readiness") or {}).get("gaps") or []
        df = pd.DataFrame(rows)
        if project:
            df = df[df.get("project_name") == project]
        if df.empty or "section" not in df.columns:
            return [], None
        sections = sorted({str(val) for val in df["section"].dropna().astype(str).str.strip() if val})
        options = [{"label": value, "value": value} for value in sections]
        return options, None

    @app.callback(
        Output("stringing-analytics-audit-table", "columns"),
        Output("stringing-analytics-audit-table", "data"),
        Output("stringing-analytics-audit-definition", "children"),
        Input("stringing-analytics-payload", "data"),
        Input("stringing-analytics-audit-selection", "data"),
        Input("stringing-analytics-section-filter", "value"),
    )
    def _render_stringing_audit_table(
        payload: dict[str, Any] | None,
        selection: dict[str, Any] | None,
        section_value: str | None,
    ):
        columns, data, definition = _stringing_analytics_table_from_selection(
            payload, selection, section_value
        )
        return columns, data, definition

    @app.callback(
        Output("stringing-analytics-audit-download", "data"),
        Input("stringing-analytics-audit-export-btn", "n_clicks"),
        State("stringing-analytics-payload", "data"),
        State("stringing-analytics-audit-selection", "data"),
        State("stringing-analytics-section-filter", "value"),
        prevent_initial_call=True,
    )
    def _export_stringing_audit(
        export_clicks: int | None,
        payload: dict[str, Any] | None,
        selection: dict[str, Any] | None,
        section_value: str | None,
    ):
        if not export_clicks:
            raise PreventUpdate
        columns, data, _definition = _stringing_analytics_table_from_selection(
            payload, selection, section_value
        )
        if not columns:
            raise PreventUpdate
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame(columns=[col["name"] for col in columns])

        def _writer(buffer: BytesIO) -> None:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Audit", index=False)

        return send_bytes(_writer, "Stringing_Analytics_Audit.xlsx")

    if config.enable_stringing:
        try:
            _load_stringing_plan_snapshot(config)
        except Exception:
            LOGGER.warning("Stringing plan cache warm-up failed during init.", exc_info=True)
        try:
            _get_stringing_tse_lookup()
        except Exception:
            LOGGER.warning("Stringing TSE lookup warm-up failed during init.", exc_info=True)


def _stringing_plan_month_series(frame: pd.DataFrame, date_columns: Sequence[str]) -> pd.Series:
    """Return a normalized month-series derived from the provided date columns."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series([], dtype="datetime64[ns]")

    accumulator: pd.Series | None = None
    for column in date_columns:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        accumulator = parsed.copy() if accumulator is None else accumulator.fillna(parsed)

    if accumulator is None and "completion_date" in frame.columns:
        accumulator = pd.to_datetime(frame["completion_date"], errors="coerce")

    if accumulator is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    return accumulator.dt.to_period("M").dt.to_timestamp()


def _load_stringing_plan_month_map(
    *,
    date_columns: Sequence[str] = _STRINGING_FS_DATE_COLUMNS,
) -> dict[str, set[pd.Timestamp]]:
    summary = _get_stringing_plan_summary_frame()
    plan_map: dict[str, set[pd.Timestamp]] = {}
    if summary.empty:
        return plan_map
    plan = summary.copy()
    plan["plan_month"] = pd.to_datetime(plan["plan_month"], errors="coerce")
    plan = plan.dropna(subset=["plan_month"])
    if plan.empty:
        return plan_map
    name_series = plan.get("project_name_display", plan["project_key_norm"].astype(str))
    code_series = plan.get("project_key", plan["project_key_norm"].astype(str))
    plan["__name_norm"] = name_series.astype(str).map(_normalize_lower)
    plan["__code_norm"] = code_series.astype(str).map(_normalize_lower)
    plan["__name_compact"] = name_series.astype(str).map(_compact_project_key)
    plan["__code_compact"] = code_series.astype(str).map(_compact_project_key)
    for _, row in plan.iterrows():
        month_value = row.get("plan_month")
        if pd.isna(month_value):
            continue
        ts = pd.Timestamp(month_value)
        for key in (
            row.get("__name_norm"),
            row.get("__code_norm"),
            row.get("__name_compact"),
            row.get("__code_compact"),
            row.get("project_key_norm"),
        ):
            if key:
                plan_map.setdefault(str(key), set()).add(ts)
    return plan_map

def _stringing_plan_keys(name: str | None, code: str | None) -> list[str]:
    keys: set[str] = set()

    def _add_value(raw: str) -> None:
        if not raw:
            return
        norm = _normalize_lower(raw)
        if norm:
            keys.add(norm)
        compact = _compact_project_key(raw)
        if compact:
            keys.add(compact)

    for value in (name, code):
        if not value:
            continue
        text = str(value)
        _add_value(text)
        if " : " in text:
            left, right = text.split(" : ", 1)
            _add_value(left)
            _add_value(right)
    return [key for key in keys if key]


def _stringing_project_has_plan(plan_map: Mapping[str, set[pd.Timestamp]], name: str | None, code: str | None) -> bool:
    for key in _stringing_plan_keys(name, code):
        if key in plan_map:
            return True
    return False


def _stringing_plan_months_for_project(plan_map: Mapping[str, set[pd.Timestamp]], name: str, code: str) -> list[pd.Timestamp]:
    months: set[pd.Timestamp] = set()
    for key in _stringing_plan_keys(name, code):
        months.update(plan_map.get(key, ()))
    return sorted(months)

_STRINGING_PROJECT_CODE_PATTERN = re.compile(r"^(TA|TB)\s*-?\s*(\d{2,4})$", re.IGNORECASE)


def _format_stringing_project_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = _STRINGING_PROJECT_CODE_PATTERN.match(text)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"
    return text


def _stringing_plan_span_series(
    frame: pd.DataFrame,
    *,
    fallback_norms: Sequence[str] | None = None,
) -> pd.Series | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    candidates = {
        "span_m",
        "span m",
        "span",
        "span length",
        "span (m)",
        "span(km)",
        "span_km",
        "span(k m)",
        "span(m)",
        "span (km)",
        "length_m",
        "length m",
        "length",
        "length (m)",
        "length(km)",
        "length_km",
    }
    series_acc: pd.Series | None = None
    for column in frame.columns:
        norm = _normalize_col_key(column)
        if norm not in candidates:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        norm_key = norm.replace(" ", "")
        if "km" in norm_key:
            km_values = numeric
        elif norm in {"length_m", "length", "p/o", "span_m", "span (m)", "span", "tower_weight", "po_km"}:
            km_values = numeric / 1000.0
        else:
            km_values = numeric / 1000.0
        if series_acc is None:
            series_acc = km_values
        else:
            series_acc = series_acc.fillna(km_values)
    if series_acc is not None:
        return series_acc

    fallback_norms = tuple(fallback_norms or ())
    if fallback_norms:
        for column in frame.columns:
            norm = _normalize_col_key(column)
            if norm not in fallback_norms:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            norm_key = norm.replace(" ", "")
            if "km" in norm_key:
                km_values = numeric
            elif norm in {"length_m", "length", "p/o", "span_m", "span (m)", "span", "tower_weight", "po_km"}:
                km_values = numeric / 1000.0
            else:
                km_values = numeric
            if series_acc is None:
                series_acc = km_values
            else:
                series_acc = series_acc.fillna(km_values)
        if series_acc is not None:
            return series_acc

    return None


def _stringing_plan_totals_by_project(
    months_ts: Sequence[pd.Timestamp],
    *,
    current_month: pd.Timestamp | None = None,
    date_columns: Sequence[str] = _STRINGING_FS_DATE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-project planned KM (deduplicated) from Micro Plan responsibilities."""

    try:
        plan_snapshot, _, _, _ = _load_stringing_plan_snapshot(config)
    except Exception:
        plan_snapshot = None

    empty_df = pd.DataFrame(columns=["planned_km", "project_name_plan"])
    empty_series = pd.Series(dtype=float)
    if not isinstance(plan_snapshot, pd.DataFrame) or plan_snapshot.empty:
        return empty_df, empty_series

    plan = plan_snapshot.copy()
    plan["__plan_month"] = _stringing_plan_month_series(plan, date_columns)
    plan = plan.dropna(subset=["__plan_month"])
    if plan.empty:
        return empty_df, empty_series

    fallback_norms: tuple[str, ...]
    if tuple(date_columns) == _STRINGING_PO_DATE_COLUMNS:
        fallback_norms = ("p/o",)
    else:
        fallback_norms = ("span (m)", "span_m", "length", "length_m", "tower_weight")
    span_series = _stringing_plan_span_series(plan, fallback_norms=fallback_norms)
    if span_series is None:
        return empty_df, empty_series
    plan["__plan_km"] = span_series.fillna(0.0)

    plan["__project_name"] = plan.get("project_name", pd.Series([""] * len(plan), index=plan.index)).astype(str)
    plan["__project_key"] = plan.get("project_key", pd.Series([""] * len(plan), index=plan.index)).astype(str)
    plan["__project_display"] = plan["__project_name"].where(
        plan["__project_name"].str.strip() != "", plan["__project_key"]
    )
    plan["__project_display"] = plan["__project_display"].map(_format_stringing_project_label)
    plan["__project_norm"] = plan["__project_key"].map(_compact_project_key)
    missing_norm = plan["__project_norm"].astype(str).str.strip() == ""
    if missing_norm.any():
        plan.loc[missing_norm, "__project_norm"] = plan.loc[missing_norm, "__project_display"].map(
            lambda value: _compact_project_key(value) or _normalize_lower(value)
        )
    plan["__project_norm"] = plan["__project_norm"].fillna("")
    plan = plan[plan["__project_norm"] != ""].copy()
    if plan.empty:
        return empty_df, empty_series

    location_series = plan.get("location_no")
    if location_series is not None:
        plan["__location_norm"] = location_series.map(_normalize_location)
        plan = plan.drop_duplicates(
            subset=["__project_norm", "__location_norm", "__plan_month"],
            keep="last",
        )

    if months_ts:
        month_set = {pd.Timestamp(ts) for ts in months_ts if pd.notna(ts)}
        if month_set:
            plan = plan[plan["__plan_month"].isin(month_set)].copy()
    if plan.empty:
        return empty_df, empty_series

    totals = (
        plan.groupby("__project_norm")
        .agg(
            planned_km=("__plan_km", "sum"),
            project_name_plan=("__project_display", "last"),
        )
        .sort_index()
    )
    totals.index.name = "project_key_norm"

    plan_current = pd.Series(dtype=float)
    if current_month is not None and not pd.isna(current_month):
        current_month = pd.Timestamp(current_month)
        plan_current = plan[plan["__plan_month"] == current_month]
        if not plan_current.empty:
            plan_current = plan_current.groupby("__project_norm")["__plan_km"].sum()
        else:
            plan_current = pd.Series(dtype=float)

    return totals, plan_current


def _stringing_scope_has_plan(
    scope_meta: dict[str, Any] | None,
    months_ts: list[pd.Timestamp],
    *,
    date_columns: Sequence[str] = _STRINGING_FS_DATE_COLUMNS,
) -> bool:
    plan_map = _load_stringing_plan_month_map(date_columns=date_columns)
    if not plan_map:
        return False
    selected = (scope_meta or {}).get("selected") or {}
    project_filters = _normalize_str_list(selected.get("projects"))
    target_months = {ts for ts in months_ts if pd.notna(ts)}

    def _matches_months(months: set[pd.Timestamp]) -> bool:
        if not months:
            return False
        if not target_months:
            return True
        return bool(target_months & months)

    if project_filters:
        for project in project_filters:
            keys = _stringing_plan_keys(project, project)
            for key in keys:
                months = plan_map.get(key)
                if months and _matches_months(months):
                    return True
    else:
        for months in plan_map.values():
            if _matches_months(months):
                return True
    return False


def _stringing_planned_total_for_dates(
    scope_meta: dict[str, Any] | None,
    months_ts: list[pd.Timestamp],
    *,
    date_columns: Sequence[str],
) -> float:
    try:
        plan_df, _ = _stringing_plan_totals_by_project(
            months_ts,
            current_month=None,
            date_columns=date_columns,
        )
    except Exception:
        LOGGER.exception("Failed to compute stringing planned totals from compiled data.")
        plan_df = pd.DataFrame()

    if not isinstance(plan_df, pd.DataFrame) or plan_df.empty:
        return 0.0

    selected = (scope_meta or {}).get("selected") or {}
    project_filters = _normalize_str_list(selected.get("projects"))
    if project_filters:
        filter_keys = {
            _compact_project_key(value) or _normalize_lower(value)
            for value in project_filters
            if str(value).strip()
        }
        filter_keys = {key for key in filter_keys if key}
        if filter_keys:
            plan_df = plan_df[plan_df.index.isin(filter_keys)].copy()
            if plan_df.empty:
                return 0.0

    return float(plan_df["planned_km"].sum())


def _load_stringing_plan_snapshot(
    cfg: AppConfig,
) -> tuple[pd.DataFrame | None, set[tuple[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    cache = _STRINGING_PLAN_CACHE
    ts_now = time.time()
    cached_frame = cache["frame"]
    if (
        isinstance(cached_frame, pd.DataFrame)
        and not cached_frame.empty
        and (ts_now - cache["stored_at"] < _STRINGING_PLAN_CACHE_TTL_SECONDS)
    ):
        issues = cache["issues"]
        completion = cache["completion"]
        index_rows = cache.get("index", [])
        return cached_frame.copy(), set(completion), list(issues), list(index_rows)

    accessor = STRINGING_PLAN_ACCESSOR
    if accessor is None:
        return None, set(), [], []

    payload = accessor.load()
    if payload.has_frame:
        frame = payload.frame.copy()
        completion_keys = set(payload.completion_keys or set())
        cache["frame"] = frame.copy()
        cache["completion"] = set(completion_keys)
        cache["issues"] = []
        cache["index"] = []
        cache["stored_at"] = ts_now
        cache["last_written"] = ts_now
        return frame, completion_keys, [], []

    return None, set(), [], []



