"""Stringing analytics computations for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import re

import numpy as np
import pandas as pd

from .plan_utils import compact_project_key, normalize_location
from .stringing import add_length_units, normalize_stringing_columns, _to_datetime_normalize

_MONTH_PRODUCTIVITY_FACTOR = 30.0

_READINESS_BUCKETS = [
    ("0-15", 0, 15),
    ("16-30", 16, 30),
    ("31-60", 31, 60),
    ("61-90", 61, 90),
    (">90", 91, None),
]

_PO_FS_BUCKETS = [
    ("0-3", 0, 3),
    ("4-7", 4, 7),
    ("8-14", 8, 14),
    (">14", 15, None),
]

_PRODUCTIVITY_BUCKETS = [
    ("0-2", 0.0, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-6", 4.0, 6.0),
    (">=6", 6.0, None),
]

_CYCLE_BUCKETS = [
    ("0-30", 0, 30),
    ("31-60", 31, 60),
    ("61-90", 61, 90),
    (">90", 91, None),
]

_LOCATION_RE = re.compile(r"^\s*(\d+)([A-Za-z]+)?(?:\s*/\s*(\d+))?\s*$")
_AP_PREFIX_RE = re.compile(r"^\s*ap[\s\-_/]*", flags=re.IGNORECASE)
_GANTRY_RE = re.compile(r"\b(gantry|gty)\b", flags=re.IGNORECASE)
_LOCATION_TOKEN_SPLIT_RE = re.compile(r"\s*,\s*")
_LOCATION_FULL_TOKEN_RE = re.compile(r"^\d+[A-Z]*/\d+[A-Z]*$", flags=re.IGNORECASE)
_LOCATION_SHORTHAND_RE = re.compile(r"^[A-Z0-9]+$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class StringingAnalyticsPayload:
    scope: dict
    kpis: dict
    readiness: dict
    productivity: dict
    flow: dict
    cycle: dict
    relationship: dict

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "kpis": self.kpis,
            "readiness": self.readiness,
            "productivity": self.productivity,
            "flow": self.flow,
            "cycle": self.cycle,
            "relationship": self.relationship,
        }


def build_stringing_analytics_payload(
    daily_df: pd.DataFrame,
    compiled_df: pd.DataFrame,
    erection_daily: pd.DataFrame,
    *,
    projects: Sequence[str] | None = None,
    months: Sequence[pd.Timestamp] | None = None,
    gangs: Sequence[str] | None = None,
    method_filter: str = "tse",
    status_activity_fact: pd.DataFrame | None = None,
    status_snapshot_project: pd.DataFrame | None = None,
    status_snapshot_overall: pd.DataFrame | None = None,
    stretch_section_fact: pd.DataFrame | None = None,
    manpower_productivity_fact: pd.DataFrame | None = None,
) -> dict:
    """Return a serializable payload for stringing analytics."""
    projects = list(projects or [])
    months = list(months or [])
    gangs = list(gangs or [])

    daily = _prepare_stringing_daily(daily_df)
    daily = _filter_method(daily, method_filter)
    daily = _apply_filters_safe(daily, projects, months, gangs)

    compiled = _normalize_stringing_compiled(compiled_df)
    compiled = _filter_method(compiled, method_filter)

    compiled_po_start = _filter_compiled_by_date(compiled, "po_start_date", projects, months, gangs)

    scope_start = pd.to_datetime(daily.get("date"), errors="coerce").min() if "date" in daily.columns else pd.NaT
    scope_end = pd.to_datetime(daily.get("date"), errors="coerce").max() if "date" in daily.columns else pd.NaT
    scope_projects = _nunique(compiled_po_start, "project_name")
    scope_gangs = _nunique(daily, "gang_name") or _nunique(compiled_po_start, "gang_name")
    scope_spans = _nunique(compiled_po_start, "span_key")
    scope_total_km = float(pd.to_numeric(daily.get("daily_km"), errors="coerce").sum()) if not daily.empty else 0.0

    readiness_table = _build_erection_po_gap_table(compiled_po_start, erection_daily)
    readiness_stats = _gap_stats(readiness_table, "gap_days")
    readiness_hist = _bucket_distribution(readiness_table, "gap_days", _READINESS_BUCKETS)
    readiness_project = _project_hotspots(readiness_table, metric="median_gap")
    readiness_funnel = _readiness_funnel(compiled_po_start, readiness_table)

    flow_table = _build_po_fs_gap_table(
        _filter_compiled_by_date(compiled, "po_completion_date", projects, months, gangs)
    )
    flow_stats = _gap_stats(flow_table, "gap_days")
    flow_hist = _bucket_distribution(flow_table, "gap_days", _PO_FS_BUCKETS)

    cycle_table = _build_cycle_time_table(
        compiled,
        readiness_table,
        projects=projects,
        months=months,
        gangs=gangs,
    )
    cycle_hist = _bucket_distribution(cycle_table, "cycle_days", _CYCLE_BUCKETS)

    ageing_table, ageing_as_of = _build_ageing_table(
        compiled,
        readiness_table,
        projects=projects,
        months=months,
        gangs=gangs,
    )

    productivity = _build_gang_productivity(daily)
    productivity_hist = _bucket_distribution(productivity, "avg_km_month", _PRODUCTIVITY_BUCKETS)
    productivity_summary = _productivity_summary(productivity)
    productivity_share = _productivity_bucket_share(productivity)

    gang_months = _build_gang_month_buckets(daily)

    relationship = _build_readiness_productivity_relationship(readiness_table, productivity)

    kpis = {
        "output_km": round(scope_total_km, 2),
        "output_n": scope_spans,
        "readiness_median": readiness_stats.get("median", 0.0),
        "readiness_n": readiness_stats.get("n", 0),
        "flow_median": flow_stats.get("median", 0.0),
        "flow_n": flow_stats.get("n", 0),
    }

    payload = StringingAnalyticsPayload(
        scope={
            "start": scope_start.strftime("%Y-%m-%d") if pd.notna(scope_start) else "",
            "end": scope_end.strftime("%Y-%m-%d") if pd.notna(scope_end) else "",
            "projects": scope_projects,
            "gangs": scope_gangs,
            "spans": scope_spans,
            "total_km": round(scope_total_km, 2),
        },
        kpis=kpis,
        readiness={
            "gaps": _serialize_frame(readiness_table),
            "histogram": readiness_hist,
            "stats": readiness_stats,
            "hotspots": readiness_project,
            "funnel": readiness_funnel,
        },
        productivity={
            "gangs": _serialize_frame(productivity),
            "histogram": productivity_hist,
            "summary": productivity_summary,
            "share": productivity_share,
            "gang_months": gang_months,
        },
        flow={
            "gaps": _serialize_frame(flow_table),
            "histogram": flow_hist,
            "stats": flow_stats,
        },
        cycle={
            "gaps": _serialize_frame(cycle_table),
            "histogram": cycle_hist,
            "ageing": _serialize_frame(ageing_table),
            "ageing_as_of": ageing_as_of,
        },
        relationship=relationship,
    ).to_dict()

    payload["status_overview"] = _build_status_overview_contract(
        status_activity_fact if isinstance(status_activity_fact, pd.DataFrame) else pd.DataFrame(),
        status_snapshot_project if isinstance(status_snapshot_project, pd.DataFrame) else pd.DataFrame(),
        status_snapshot_overall if isinstance(status_snapshot_overall, pd.DataFrame) else pd.DataFrame(),
        projects=projects,
        months=months,
    )
    payload["stretch_readiness"] = _build_stretch_readiness_contract(
        stretch_section_fact if isinstance(stretch_section_fact, pd.DataFrame) else pd.DataFrame(),
        projects=projects,
        months=months,
    )
    payload["manpower_productivity"] = _build_manpower_productivity_contract(
        manpower_productivity_fact if isinstance(manpower_productivity_fact, pd.DataFrame) else pd.DataFrame(),
        projects=projects,
        months=months,
        gangs=gangs,
    )

    return payload

def _parse_date_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series([], dtype="datetime64[ns]")
    parsed = series.map(_to_datetime_normalize)
    parsed = pd.to_datetime(parsed, errors="coerce")
    return parsed.dt.normalize()


def _ensure_project_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "project_name" not in work.columns:
        work["project_name"] = work.get("project", "")
    if "project" not in work.columns:
        work["project"] = work.get("project_name", "")
    work["project_name"] = work["project_name"].fillna("").astype(str).str.strip()
    if "project_key_norm" not in work.columns:
        work["project_key_norm"] = work["project_name"].map(compact_project_key)
    work["project_key_norm"] = work["project_key_norm"].fillna("").astype(str)
    return work


def _prepare_stringing_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = _ensure_project_fields(df)
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
        work = work.dropna(subset=["date"])
    if "month" not in work.columns and "date" in work.columns:
        work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    if "daily_km" in work.columns:
        work["daily_km"] = pd.to_numeric(work["daily_km"], errors="coerce")
    if "gang_name" not in work.columns:
        work["gang_name"] = ""
    work["gang_name"] = work["gang_name"].fillna("").astype(str).str.strip()
    work = _add_span_key(work)
    return work


def _normalize_stringing_compiled(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized, _ = normalize_stringing_columns(df)
    normalized = _ensure_project_fields(normalized)
    normalized, _ = add_length_units(normalized)
    for col in ("from_ap", "to_ap", "gang_name", "method", "source_file", "section_readiness"):
        if col not in normalized.columns:
            normalized[col] = ""
    normalized["from_ap"] = normalized["from_ap"].fillna("").astype(str).str.strip()
    normalized["to_ap"] = normalized["to_ap"].fillna("").astype(str).str.strip()
    normalized["gang_name"] = normalized["gang_name"].fillna("").astype(str).str.strip()
    normalized["method"] = normalized["method"].fillna("").astype(str).str.strip()
    normalized["source_file"] = normalized["source_file"].fillna("").astype(str).str.strip()

    normalized["po_start_date"] = _parse_date_series(normalized.get("po_start_date"))
    normalized["po_completion_date"] = _parse_date_series(normalized.get("po_completion_date"))
    normalized["fs_starting_date"] = _parse_date_series(normalized.get("fs_starting_date"))
    normalized["fs_complete_date"] = _parse_date_series(normalized.get("fs_complete_date"))

    normalized = _add_span_key(normalized)
    normalized["section_label"] = _derive_section_label(normalized)
    normalized["span_label"] = normalized.apply(_span_label, axis=1)
    return normalized


def _derive_section_label(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype=object)
    for column in (
        "section",
        "section_name",
        "section_readiness",
        "section_incharge",
        "section_incharge_name",
    ):
        if column in df.columns:
            series = df[column].fillna("").astype(str).str.strip()
            if series.astype(bool).any():
                return series.where(series.astype(bool), "Unassigned")
    return pd.Series(["Unassigned"] * len(df), index=df.index, dtype=object)

def _normalize_method(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def _method_group(value: str) -> str:
    if "tse" in value:
        return "TSE"
    if "manual" in value:
        return "Manual"
    return "Other"


def _filter_method(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "method" not in df.columns and "method_norm" not in df.columns:
        return df.copy()
    work = df.copy()
    method_series = work.get("method", work.get("method_norm", pd.Series("", index=work.index)))
    method_norm = method_series.map(_normalize_method)
    work["method_group"] = method_norm.map(_method_group)
    method_present = method_norm.replace("", pd.NA).notna().any()
    if mode == "tse":
        if not method_present:
            return work
        return work[work["method_group"] == "TSE"].copy()
    if mode == "exclude_manual":
        if not method_present:
            return work
        return work[work["method_group"] != "Manual"].copy()
    return work


def _normalize_location_token(value: object) -> str:
    text = normalize_location(value)
    if not text:
        return ""
    text = re.sub(r"^\s*AP[\s\-_./]*", "", text, flags=re.IGNORECASE)
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(\d)\.0\b", r"\1", text)
    return text


def _pick_location_nos_column(work: pd.DataFrame) -> str | None:
    if work is None or work.empty:
        return None
    col_keys = {re.sub(r"[^a-z0-9]+", "", str(col).strip().lower()): col for col in work.columns}
    for name in ("locationnos", "locationno_s", "locationsnos", "locationnumbers", "locationlist"):
        if name in col_keys:
            return col_keys[name]
    for key, col in col_keys.items():
        if "location" in key and ("nos" in key or "numbers" in key):
            return col
    return None


def _normalize_required_location_sequence(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_location_token(value)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _parse_location_nos_extras(location_nos_raw: object) -> tuple[list[str], str, str]:
    raw_text = str(location_nos_raw or "").strip()
    if not raw_text:
        return [], "EMPTY", ""
    tokens = [token.strip() for token in _LOCATION_TOKEN_SPLIT_RE.split(raw_text) if token.strip()]
    if not tokens:
        return [], "EMPTY", ""

    extras: list[str] = []
    current_anchor_prefix = ""
    saw_full_anchor = False
    unparsed_tokens: list[str] = []
    for token in tokens:
        normalized = _normalize_location_token(token)
        if not normalized:
            continue
        if _LOCATION_FULL_TOKEN_RE.fullmatch(normalized):
            saw_full_anchor = True
            current_anchor_prefix = normalized.split("/", 1)[0]
            extras.append(normalized)
            continue
        if _LOCATION_SHORTHAND_RE.fullmatch(normalized):
            if saw_full_anchor and current_anchor_prefix:
                extras.append(f"{current_anchor_prefix}/{normalized}")
            else:
                unparsed_tokens.append(normalized)
            continue
        unparsed_tokens.append(normalized)

    extras = _normalize_required_location_sequence(extras)
    if unparsed_tokens and not saw_full_anchor:
        return [], "SHORTHAND_NO_ANCHOR", "Location Nos has shorthand tokens without an explicit anchor token."
    if unparsed_tokens:
        return extras, "PARTIAL_PARSE", f"Unparsed tokens: {', '.join(unparsed_tokens[:5])}"
    return extras, "OK", ""


def _build_required_locations(from_ap: object, to_ap: object, location_nos_raw: object) -> tuple[list[str], str, str]:
    endpoints = _normalize_required_location_sequence((str(from_ap or ""), str(to_ap or "")))
    extras, parse_status, parse_issue = _parse_location_nos_extras(location_nos_raw)
    required = _normalize_required_location_sequence([*endpoints, *extras])
    if parse_status == "SHORTHAND_NO_ANCHOR":
        required = endpoints
    if not endpoints and not required:
        return [], parse_status, parse_issue
    return required, parse_status, parse_issue


def _letter_rank(value: str) -> int:
    rank = 0
    for ch in value.upper():
        if "A" <= ch <= "Z":
            rank = (rank * 26) + (ord(ch) - ord("A") + 1)
    return rank


def _strip_ap_prefix(value: object) -> str:
    text = normalize_location(value)
    if not text:
        return ""
    return _AP_PREFIX_RE.sub("", text).strip()


def _is_gantry_label(value: object) -> bool:
    text = normalize_location(value)
    if not text:
        return False
    cleaned = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    return bool(_GANTRY_RE.search(cleaned))


def _location_order_key(value: object) -> int | None:
    text = _strip_ap_prefix(value)
    if not text:
        return None
    match = _LOCATION_RE.match(text)
    if not match:
        return None
    main = int(match.group(1))
    letters = match.group(2) or ""
    sub = int(match.group(3) or 0)
    letter_rank = _letter_rank(letters) if letters else 0
    return (main * 1_000_000) + (letter_rank * 1_000) + sub


def _add_span_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = _ensure_project_fields(df)
    for col in ("from_ap", "to_ap"):
        if col not in work.columns:
            work[col] = ""
    work["from_ap_norm"] = work["from_ap"].map(_strip_ap_prefix)
    work["to_ap_norm"] = work["to_ap"].map(_strip_ap_prefix)
    span_base = (
        work["project_key_norm"].fillna("")
        + "|"
        + work["from_ap_norm"].fillna("")
        + "|"
        + work["to_ap_norm"].fillna("")
    )
    fallback = work.get("row_id", pd.Series("", index=work.index))
    work["span_key"] = span_base.where(
        (work["from_ap_norm"].astype(bool) | work["to_ap_norm"].astype(bool)),
        fallback.astype(str),
    )
    return work


def _span_label(row: pd.Series) -> str:
    from_ap = str(row.get("from_ap_norm") or row.get("from_ap") or "").strip()
    to_ap = str(row.get("to_ap_norm") or row.get("to_ap") or "").strip()
    if from_ap or to_ap:
        return f"{from_ap}-{to_ap}".strip("-")
    return str(row.get("span_key") or "").strip()


def _resolve_project_key_norm(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("project_code")
    if source is None:
        source = frame.get("Project Code")
    if source is None:
        source = frame.get("project_name")
    if source is None:
        source = frame.get("Project Name")
    if source is None:
        source = frame.get("project")
    if source is None:
        source = frame.get("Project")
    if source is None:
        source = pd.Series("", index=frame.index)
    source = source.fillna("").astype(str).str.strip()
    fallback = frame.get(
        "project_name",
        frame.get(
            "Project Name",
            frame.get("project", frame.get("Project", pd.Series("", index=frame.index))),
        ),
    )
    fallback = fallback.fillna("").astype(str).str.strip()
    resolved = source.where(source.astype(bool), fallback)
    return resolved.map(compact_project_key)


def _apply_filters_safe(df: pd.DataFrame, projects, months, gangs) -> pd.DataFrame:
    from .filters import apply_filters
    if df is None or df.empty:
        return pd.DataFrame()
    return apply_filters(df, projects, months, gangs)


def _filter_compiled_by_date(
    df: pd.DataFrame,
    date_column: str,
    projects: Sequence[str],
    months: Sequence[pd.Timestamp],
    gangs: Sequence[str],
) -> pd.DataFrame:
    if df is None or df.empty or date_column not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["date"] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=["date"])
    work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    return _apply_filters_safe(work, projects, months, gangs)


def _serialize_frame(df: pd.DataFrame) -> list[dict[str, object]]:
    if df is None or df.empty:
        return []
    return df.replace({np.nan: None}).to_dict("records")


def _nunique(df: pd.DataFrame, column: str) -> int:
    if df is None or df.empty or column not in df.columns:
        return 0
    return (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )

def _build_erection_location_map(erection_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if erection_daily is None or erection_daily.empty:
        return {}
    work = erection_daily.copy()
    completion = None
    for col in ("completion_date", "Complete Date", "complete date", "date", "Work Date", "work date"):
        if col in work.columns:
            completion = work[col]
            break
    if completion is None:
        return {}
    work["completion_date"] = pd.to_datetime(completion, errors="coerce").dt.normalize()
    location_col = ""
    for col in ("location_no", "Location No.", "location no.", "Location No", "location no", "location"):
        if col in work.columns:
            location_col = col
            break
    if not location_col:
        return {}
    work["location_no"] = work[location_col].fillna("").astype(str).str.strip()
    project_key_series = work.get("project_key_norm")
    if project_key_series is None:
        project_key_series = _resolve_project_key_norm(work)
    else:
        project_key_series = project_key_series.fillna("").astype(str)
        missing_mask = ~project_key_series.astype(bool)
        if bool(missing_mask.any()):
            project_key_series = project_key_series.where(~missing_mask, _resolve_project_key_norm(work))
    work["project_key_norm"] = project_key_series.fillna("").astype(str)
    work = work.dropna(subset=["completion_date"])
    work = work[work["location_no"].astype(bool) & work["project_key_norm"].astype(bool)]
    if work.empty:
        return {}
    work["location_no_norm"] = work["location_no"].map(normalize_location)
    work["location_token_norm"] = work["location_no"].map(_normalize_location_token)
    work["loc_order"] = work["location_no_norm"].map(_location_order_key)
    work = work.dropna(subset=["loc_order"])
    if work.empty:
        return {}
    work = (
        work.sort_values("completion_date")
        .drop_duplicates(subset=["project_key_norm", "location_no_norm", "loc_order"], keep="last")
    )
    project_map: dict[str, pd.DataFrame] = {}
    for project_key, group in work.groupby("project_key_norm"):
        project_map[str(project_key)] = group[
            ["loc_order", "completion_date", "location_no_norm", "location_token_norm", "location_no"]
        ].copy()
    return project_map


def _build_erection_po_gap_table(
    compiled: pd.DataFrame,
    erection_daily: pd.DataFrame,
) -> pd.DataFrame:
    if compiled is None or compiled.empty:
        return pd.DataFrame()
    work = compiled.copy()
    if "po_start_date" not in work.columns:
        return pd.DataFrame()
    work = work.dropna(subset=["po_start_date"])
    if work.empty:
        return pd.DataFrame()

    work["from_order"] = work["from_ap"].map(_location_order_key)
    work["to_order"] = work["to_ap"].map(_location_order_key)
    work["from_is_gantry"] = work["from_ap"].map(_is_gantry_label)
    work["to_is_gantry"] = work["to_ap"].map(_is_gantry_label)
    location_nos_col = _pick_location_nos_column(work)
    if location_nos_col:
        work["__location_nos_raw"] = work[location_nos_col]
    else:
        work["__location_nos_raw"] = ""
    built = work.apply(
        lambda row: _build_required_locations(row.get("from_ap"), row.get("to_ap"), row.get("__location_nos_raw")),
        axis=1,
    )
    work["required_locations"] = built.map(lambda item: item[0])
    work["location_parse_status"] = built.map(lambda item: item[1])
    work["location_parse_issue"] = built.map(lambda item: item[2])

    erection_map = _build_erection_location_map(erection_daily)

    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        project_key = row.get("project_key_norm", "")
        po_start = row.get("po_start_date")
        from_order = row.get("from_order")
        to_order = row.get("to_order")
        from_ap = row.get("from_ap", "")
        to_ap = row.get("to_ap", "")
        from_is_gantry = bool(row.get("from_is_gantry", False))
        to_is_gantry = bool(row.get("to_is_gantry", False))
        parse_status = str(row.get("location_parse_status") or "EMPTY").strip().upper() or "EMPTY"
        parse_issue = str(row.get("location_parse_issue") or "").strip()
        required_locations = list(row.get("required_locations") or [])
        location_nos_raw = row.get("__location_nos_raw", "")

        base = {
            "project_name": row.get("project_name", ""),
            "section": row.get("section_label", ""),
            "span": row.get("span_label", ""),
            "from_ap": from_ap,
            "to_ap": to_ap,
            "gang_name": row.get("gang_name", ""),
            "po_start_date": po_start,
            "span_key": row.get("span_key", ""),
            "lag_inference_mode": "ALPHABETIC_FALLBACK",
            "lag_fallback_used": True,
            "location_parse_status": parse_status,
            "location_parse_issue": parse_issue,
            "required_location_count": int(len(required_locations)),
            "matched_location_count": 0,
            "unmatched_location_count": int(len(required_locations)),
            "required_locations": ", ".join(required_locations),
            "matched_locations": "",
            "unmatched_locations": ", ".join(required_locations),
            "location_nos_raw": str(location_nos_raw or ""),
        }

        if pd.isna(po_start):
            rows.append({**base, "last_erection_completion_date": pd.NaT, "gap_days": pd.NA})
            continue

        project_df = erection_map.get(str(project_key))
        if project_df is None or project_df.empty:
            rows.append({**base, "last_erection_completion_date": pd.NaT, "gap_days": pd.NA})
            continue

        alpha_last_completion = pd.NaT
        if not (from_is_gantry and to_is_gantry):
            if (not from_is_gantry and pd.notna(from_order)) or (not to_is_gantry and pd.notna(to_order)):
                try:
                    if to_is_gantry and not from_is_gantry and pd.notna(from_order):
                        lo = int(from_order)
                        hi = int(project_df["loc_order"].max())
                    elif from_is_gantry and not to_is_gantry and pd.notna(to_order):
                        lo = int(project_df["loc_order"].min())
                        hi = int(to_order)
                    elif pd.notna(from_order) and pd.notna(to_order):
                        lo = min(int(from_order), int(to_order))
                        hi = max(int(from_order), int(to_order))
                    else:
                        lo = None
                        hi = None
                    if lo is not None and hi is not None:
                        span_df = project_df[(project_df["loc_order"] >= lo) & (project_df["loc_order"] <= hi)]
                        if not span_df.empty:
                            alpha_last_completion = span_df["completion_date"].max()
                except Exception:
                    alpha_last_completion = pd.NaT

        token_series = project_df.get("location_token_norm", pd.Series("", index=project_df.index)).fillna("").astype(str)
        token_frame = project_df.assign(__token=token_series)
        token_frame = token_frame[token_frame["__token"].astype(bool)].copy()
        token_map: dict[str, pd.Timestamp] = (
            token_frame.set_index("__token")["completion_date"].to_dict() if not token_frame.empty else {}
        )
        matched_locations: list[str] = []
        unmatched_locations: list[str] = []
        matched_dates: list[pd.Timestamp] = []
        for token in required_locations:
            hit = token_map.get(token)
            if pd.notna(hit):
                matched_locations.append(token)
                matched_dates.append(hit)
            else:
                unmatched_locations.append(token)
        loc_last_completion = max(matched_dates) if matched_dates else pd.NaT

        has_location_signal = bool(str(location_nos_raw or "").strip()) and parse_status != "EMPTY"
        mode = "ALPHABETIC_FALLBACK"
        fallback_used = True
        if parse_status == "OK" and required_locations and not unmatched_locations and pd.notna(loc_last_completion):
            mode = "LOCATION_NOS"
            fallback_used = False
            last_completion = loc_last_completion
        else:
            if has_location_signal or parse_status in {"PARTIAL_PARSE", "SHORTHAND_NO_ANCHOR"}:
                mode = "HYBRID_PARTIAL_FALLBACK"
            candidates = [ts for ts in (loc_last_completion, alpha_last_completion) if pd.notna(ts)]
            last_completion = max(candidates) if candidates else pd.NaT

        gap_days = (po_start - last_completion).days if pd.notna(last_completion) else pd.NA
        rows.append(
            {
                **base,
                "lag_inference_mode": mode,
                "lag_fallback_used": bool(fallback_used),
                "matched_location_count": int(len(matched_locations)),
                "unmatched_location_count": int(len(unmatched_locations)),
                "matched_locations": ", ".join(matched_locations),
                "unmatched_locations": ", ".join(unmatched_locations),
                "last_erection_completion_date": last_completion,
                "gap_days": int(gap_days) if gap_days is not pd.NA else pd.NA,
            }
        )

    return pd.DataFrame(rows)


def _build_po_fs_gap_table(compiled: pd.DataFrame) -> pd.DataFrame:
    if compiled is None or compiled.empty:
        return pd.DataFrame()
    work = compiled.copy()
    if "po_completion_date" not in work.columns or "fs_starting_date" not in work.columns:
        return pd.DataFrame()
    work = work.dropna(subset=["po_completion_date", "fs_starting_date"])
    if work.empty:
        return pd.DataFrame()

    gap = (work["fs_starting_date"] - work["po_completion_date"]).dt.days
    work = work.assign(
        gap_days=gap,
        negative_gap_flag=gap.fillna(0).lt(0),
    )
    return work[
        [
            "project_name",
            "section_label",
            "span_label",
            "from_ap",
            "to_ap",
            "gang_name",
            "po_completion_date",
            "fs_starting_date",
            "gap_days",
            "negative_gap_flag",
            "span_key",
        ]
    ].rename(columns={"section_label": "section", "span_label": "span"})


def _gap_stats(df: pd.DataFrame, column: str) -> dict[str, float | int]:
    if df is None or df.empty or column not in df.columns:
        return {"n": 0, "median": 0.0, "pct_over_15": 0.0, "pct_over_60": 0.0}
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"n": 0, "median": 0.0, "pct_over_15": 0.0, "pct_over_60": 0.0}
    series = series.clip(lower=0)
    n = int(series.size)
    median = float(series.median())
    pct_over_15 = float((series > 15).mean() * 100.0)
    pct_over_60 = float((series > 60).mean() * 100.0)
    return {
        "n": n,
        "median": round(median, 1),
        "pct_over_15": round(pct_over_15, 1),
        "pct_over_60": round(pct_over_60, 1),
    }


def _bucket_distribution(df: pd.DataFrame, column: str, buckets: list[tuple[str, float | int, float | int | None]]) -> list[dict[str, object]]:
    if df is None or df.empty or column not in df.columns:
        return [{"bucket": label, "count": 0, "pct": 0.0} for label, _, _ in buckets]
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return [{"bucket": label, "count": 0, "pct": 0.0} for label, _, _ in buckets]
    series = series.clip(lower=0)
    total = float(series.size)
    rows: list[dict[str, object]] = []
    for label, lower, upper in buckets:
        if upper is None:
            mask = series >= float(lower)
        else:
            mask = (series >= float(lower)) & (series <= float(upper))
        count = int(mask.sum())
        pct = round((count / total) * 100.0, 1) if total else 0.0
        rows.append({"bucket": label, "count": count, "pct": pct})
    return rows


def _project_hotspots(df: pd.DataFrame, *, metric: str = "median_gap") -> list[dict[str, object]]:
    if df is None or df.empty or "project_name" not in df.columns:
        return []
    work = df.copy()
    work["gap_days"] = pd.to_numeric(work.get("gap_days"), errors="coerce")
    work = work.dropna(subset=["gap_days"])
    if work.empty:
        return []
    grouped = work.groupby("project_name").agg(
        median_gap=("gap_days", "median"),
        pct_over_60=("gap_days", lambda s: float((s > 60).mean() * 100.0)),
        spans=("span_key", "nunique"),
    )
    grouped = grouped.reset_index()
    grouped["median_gap"] = grouped["median_gap"].round(1)
    grouped["pct_over_60"] = grouped["pct_over_60"].round(1)
    grouped = grouped.sort_values("median_gap", ascending=False).head(10)
    return grouped.to_dict("records")


def _readiness_funnel(compiled: pd.DataFrame, readiness_table: pd.DataFrame) -> list[dict[str, object]]:
    if compiled is None or compiled.empty:
        return []
    base = compiled.drop_duplicates(subset=["span_key"]).copy()
    last_map = {}
    if readiness_table is not None and not readiness_table.empty:
        last_map = readiness_table.set_index("span_key")["last_erection_completion_date"].to_dict()
    base["last_erection_completion_date"] = base["span_key"].map(last_map)

    return [
        {"stage": "Erection complete", "count": int(base["last_erection_completion_date"].notna().sum())},
        {"stage": "P/O started", "count": int(base["po_start_date"].notna().sum())},
        {"stage": "P/O completed", "count": int(base["po_completion_date"].notna().sum())},
        {"stage": "Sag completed", "count": int(base["fs_complete_date"].notna().sum())},
    ]


def _build_cycle_time_table(
    compiled: pd.DataFrame,
    readiness_table: pd.DataFrame,
    *,
    projects: Sequence[str],
    months: Sequence[pd.Timestamp],
    gangs: Sequence[str],
) -> pd.DataFrame:
    if compiled is None or compiled.empty:
        return pd.DataFrame()
    base = compiled.copy()
    base["sag_end_date"] = base["fs_complete_date"].where(
        base["fs_complete_date"].notna(),
        base["fs_starting_date"],
    )
    base = base.dropna(subset=["sag_end_date"])
    if base.empty:
        return pd.DataFrame()
    base["date"] = base["sag_end_date"]
    base["month"] = base["date"].dt.to_period("M").dt.to_timestamp()
    base = _apply_filters_safe(base, projects, months, gangs)
    if base.empty:
        return pd.DataFrame()

    last_map = {}
    if readiness_table is not None and not readiness_table.empty:
        last_map = readiness_table.set_index("span_key")["last_erection_completion_date"].to_dict()
    base["last_erection_completion_date"] = base["span_key"].map(last_map)
    base = base.dropna(subset=["last_erection_completion_date"])
    if base.empty:
        return pd.DataFrame()

    base["cycle_days"] = (base["sag_end_date"] - base["last_erection_completion_date"]).dt.days
    return base[
        [
            "project_name",
            "section_label",
            "span_label",
            "from_ap",
            "to_ap",
            "gang_name",
            "last_erection_completion_date",
            "sag_end_date",
            "cycle_days",
            "span_key",
        ]
    ].rename(columns={"section_label": "section", "span_label": "span"})


def _build_ageing_table(
    compiled: pd.DataFrame,
    readiness_table: pd.DataFrame,
    *,
    projects: Sequence[str],
    months: Sequence[pd.Timestamp],
    gangs: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    if compiled is None or compiled.empty:
        return pd.DataFrame(), ""
    base = compiled.copy()
    base["date"] = base["po_start_date"]
    base["month"] = base["date"].dt.to_period("M").dt.to_timestamp()
    base = _apply_filters_safe(base, projects, months, gangs)
    if base.empty:
        return pd.DataFrame(), ""

    last_map = {}
    if readiness_table is not None and not readiness_table.empty:
        last_map = readiness_table.set_index("span_key")["last_erection_completion_date"].to_dict()
    base["last_erection_completion_date"] = base["span_key"].map(last_map)

    as_of_candidates = []
    for col in ("fs_complete_date", "fs_starting_date", "po_completion_date", "po_start_date"):
        if col in base.columns:
            as_of_candidates.append(base[col].max())
    as_of_candidates = [val for val in as_of_candidates if pd.notna(val)]
    as_of = max(as_of_candidates) if as_of_candidates else pd.Timestamp.today().normalize()

    pending = base[base["fs_complete_date"].isna()].copy()
    if pending.empty:
        return pd.DataFrame(), as_of.strftime("%Y-%m-%d")

    def _stage(row: pd.Series) -> tuple[str, pd.Timestamp | None]:
        if pd.notna(row.get("po_completion_date")):
            return "P/O done", row.get("po_completion_date")
        if pd.notna(row.get("po_start_date")):
            return "P/O started", row.get("po_start_date")
        if pd.notna(row.get("last_erection_completion_date")):
            return "E-done", row.get("last_erection_completion_date")
        return "Unknown", None

    stages = pending.apply(_stage, axis=1, result_type="expand")
    pending["current_stage"] = stages[0]
    pending["stage_date"] = stages[1]
    pending["ageing_days"] = (as_of - pending["stage_date"]).dt.days
    pending["ageing_days"] = pd.to_numeric(pending["ageing_days"], errors="coerce")
    pending = pending.dropna(subset=["ageing_days"])

    pending = pending.sort_values("ageing_days", ascending=False).head(20)
    return (
        pending[
            [
                "project_name",
                "section_label",
                "span_label",
                "current_stage",
                "ageing_days",
            ]
        ].rename(columns={"section_label": "section", "span_label": "span"}),
        as_of.strftime("%Y-%m-%d"),
    )

def _build_gang_productivity(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["daily_km"] = pd.to_numeric(work.get("daily_km"), errors="coerce")
    work = work.dropna(subset=["daily_km"])
    work["gang_name"] = work.get("gang_name", "").fillna("").astype(str).str.strip()
    work = work[work["gang_name"].astype(bool)]
    if work.empty:
        return pd.DataFrame()
    grouped = (
        work.groupby("gang_name", dropna=False)
        .agg(
            avg_daily=("daily_km", "mean"),
            total_km=("daily_km", "sum"),
            active_days=("date", lambda s: s.dropna().nunique()),
            spans=("span_key", "nunique"),
            projects=("project_key_norm", "nunique"),
        )
        .reset_index()
    )
    grouped["avg_km_month"] = (grouped["avg_daily"].fillna(0.0) * _MONTH_PRODUCTIVITY_FACTOR).round(2)
    grouped["total_km"] = grouped["total_km"].fillna(0.0).round(2)
    grouped["active_days"] = grouped["active_days"].fillna(0).astype(int)
    grouped["spans"] = grouped["spans"].fillna(0).astype(int)
    grouped["projects"] = grouped["projects"].fillna(0).astype(int)
    grouped["bucket"] = grouped["avg_km_month"].map(_productivity_bucket)
    return grouped.sort_values("avg_km_month", ascending=False).reset_index(drop=True)


def _productivity_bucket(value: float) -> str:
    if value < 2:
        return "0-2"
    if value < 4:
        return "2-4"
    if value < 6:
        return "4-6"
    return ">=6"


def _productivity_summary(productivity: pd.DataFrame) -> dict[str, float | int]:
    if productivity is None or productivity.empty or "avg_km_month" not in productivity.columns:
        return {"median": 0.0, "pct_below_3": 0.0, "pct_above_6": 0.0, "n": 0}
    series = pd.to_numeric(productivity["avg_km_month"], errors="coerce").dropna()
    if series.empty:
        return {"median": 0.0, "pct_below_3": 0.0, "pct_above_6": 0.0, "n": 0}
    n = int(series.size)
    median = float(series.median())
    pct_below = float((series < 3).mean() * 100.0)
    pct_above = float((series >= 6).mean() * 100.0)
    return {
        "median": round(median, 2),
        "pct_below_3": round(pct_below, 1),
        "pct_above_6": round(pct_above, 1),
        "n": n,
    }


def _productivity_bucket_share(productivity: pd.DataFrame) -> list[dict[str, object]]:
    if productivity is None or productivity.empty:
        return []
    work = productivity.copy()
    total_gangs = float(len(work.index))
    total_km = float(pd.to_numeric(work.get("total_km"), errors="coerce").sum())
    rows: list[dict[str, object]] = []
    for label, _, _ in _PRODUCTIVITY_BUCKETS:
        bucket_df = work[work["bucket"] == label]
        gangs = int(len(bucket_df.index))
        km = float(pd.to_numeric(bucket_df.get("total_km"), errors="coerce").sum())
        rows.append(
            {
                "bucket": label,
                "gangs": gangs,
                "gang_share": (gangs / total_gangs * 100.0) if total_gangs else 0.0,
                "km": round(km, 2),
                "km_share": (km / total_km * 100.0) if total_km else 0.0,
            }
        )
    return rows


def _build_gang_month_buckets(daily: pd.DataFrame) -> dict[str, object]:
    if daily is None or daily.empty:
        return {
            "summary": [],
            "rows": [],
            "total_output": 0.0,
            "total_gang_months": 0,
            "total_active_days": 0.0,
        }
    work = daily.copy()
    if "month" not in work.columns:
        if "date" in work.columns:
            work["month"] = pd.to_datetime(work["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        else:
            return {
                "summary": [],
                "rows": [],
                "total_output": 0.0,
                "total_gang_months": 0,
                "total_active_days": 0.0,
            }
    work["daily_km"] = pd.to_numeric(work.get("daily_km"), errors="coerce")
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    else:
        work["date"] = pd.NaT
    work = work.dropna(subset=["daily_km", "gang_name", "month"])
    if work.empty:
        return {
            "summary": [],
            "rows": [],
            "total_output": 0.0,
            "total_gang_months": 0,
            "total_active_days": 0.0,
        }

    grouped = (
        work.groupby(["gang_name", "month"], dropna=False)
        .agg(
            total_km=("daily_km", "sum"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    grouped["bucket"] = grouped["total_km"].map(_productivity_bucket)

    total_output = float(grouped["total_km"].sum())
    total_gm = float(len(grouped.index))
    total_active_days = float(grouped["active_days"].sum()) if total_gm else 0.0

    summary = (
        grouped.groupby("bucket")
        .agg(
            gang_months=("gang_name", "size"),
            km_total=("total_km", "sum"),
            active_days_total=("active_days", "sum"),
        )
        .reset_index()
    )
    summary = summary.set_index("bucket").reindex([label for label, _, _ in _PRODUCTIVITY_BUCKETS], fill_value=0)
    summary = summary.reset_index()
    summary["active_days_share"] = (
        summary["active_days_total"].astype(float) / total_active_days if total_active_days else 0.0
    )
    summary["km_share"] = summary["km_total"].astype(float) / total_output if total_output else 0.0
    summary["avg_km"] = (
        summary["km_total"].astype(float) / summary["gang_months"].replace(0, np.nan)
    ).fillna(0.0)

    return {
        "summary": summary.to_dict("records"),
        "rows": grouped.to_dict("records"),
        "total_output": round(total_output, 2),
        "total_gang_months": int(total_gm),
        "total_active_days": float(total_active_days),
    }


def _build_readiness_productivity_relationship(
    readiness_table: pd.DataFrame,
    productivity: pd.DataFrame,
) -> dict[str, object]:
    if readiness_table is None or readiness_table.empty or productivity is None or productivity.empty:
        return {"summary": []}
    prod_map = productivity.set_index("gang_name")["avg_km_month"].to_dict()
    work = readiness_table.copy()
    work["avg_km_month"] = work["gang_name"].map(prod_map)
    work = work.dropna(subset=["gap_days", "avg_km_month"])
    if work.empty:
        return {"summary": []}
    work["bucket"] = work["gap_days"].map(lambda v: _bucket_label(v, _READINESS_BUCKETS))
    grouped = (
        work.groupby("bucket")
        .agg(avg_km_month=("avg_km_month", "mean"), spans=("span_key", "nunique"))
        .reset_index()
    )
    grouped["avg_km_month"] = grouped["avg_km_month"].round(2)
    return {"summary": grouped.to_dict("records")}


def _bucket_label(value: float | int | None, buckets: list[tuple[str, float | int, float | int | None]]) -> str:
    if value is None or pd.isna(value):
        return ""
    for label, lower, upper in buckets:
        if upper is None and value >= float(lower):
            return label
        if upper is not None and float(lower) <= float(value) <= float(upper):
            return label
    return buckets[-1][0] if buckets else ""


def _derive_scope_project_key(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="string")
    if "project_scope_key" in frame.columns:
        source = frame["project_scope_key"]
    elif "project_code" in frame.columns:
        source = frame["project_code"]
    elif "project_name" in frame.columns:
        source = frame["project_name"]
    elif "project_display" in frame.columns:
        source = frame["project_display"]
    else:
        source = pd.Series("", index=frame.index)
    return source.fillna("").astype(str).map(compact_project_key)


def _filter_scope_frame(
    frame: pd.DataFrame,
    *,
    projects: Sequence[str] | None,
    months: Sequence[pd.Timestamp] | None,
    gangs: Sequence[str] | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    project_keys = {compact_project_key(value) for value in (projects or []) if str(value).strip()}
    if project_keys:
        work = work[_derive_scope_project_key(work).isin(project_keys)]
    if months:
        month_values = {pd.Timestamp(ts).to_period("M").to_timestamp() for ts in months if isinstance(ts, pd.Timestamp)}
        if month_values:
            month_col = None
            for candidate in ("month", "report_date", "date"):
                if candidate in work.columns:
                    month_col = candidate
                    break
            if month_col is not None:
                month_series = pd.to_datetime(work[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
                work = work[month_series.isin(month_values)]
    if gangs:
        gang_values = {str(value).strip().lower() for value in gangs if str(value).strip()}
        if gang_values and "gang_name" in work.columns:
            work = work[work["gang_name"].fillna("").astype(str).str.strip().str.lower().isin(gang_values)]
    return work.reset_index(drop=True)


def _build_status_overview_contract(
    status_activity: pd.DataFrame,
    status_project: pd.DataFrame,
    status_overall: pd.DataFrame,
    *,
    projects: Sequence[str] | None,
    months: Sequence[pd.Timestamp] | None,
) -> dict[str, object]:
    project_scope = _filter_scope_frame(status_project, projects=projects, months=months)
    if project_scope.empty and isinstance(status_activity, pd.DataFrame) and not status_activity.empty:
        derived = status_activity.copy()
        for col in ("quantity_primary", "cumulative_progress", "balance_progress", "plan_for_month", "progress_for_month", "today_progress"):
            if col in derived.columns:
                derived[col] = pd.to_numeric(derived[col], errors="coerce")
        if "month" not in derived.columns:
            derived["month"] = pd.to_datetime(derived.get("report_date"), errors="coerce").dt.to_period("M").dt.to_timestamp()
        project_cols = ["project_code", "project_display", "project_scope_key", "line_name", "month"]
        project_scope = (
            derived.groupby(project_cols, dropna=False)
            .agg(
                quantity_primary_sum=("quantity_primary", "sum"),
                cumulative_progress_sum=("cumulative_progress", "sum"),
                balance_progress_sum=("balance_progress", "sum"),
                plan_for_month_sum=("plan_for_month", "sum"),
                progress_for_month_sum=("progress_for_month", "sum"),
                today_progress_sum=("today_progress", "sum"),
            )
            .reset_index()
        )
        denom = project_scope["quantity_primary_sum"].where(project_scope["quantity_primary_sum"] > 0)
        project_scope["completion_pct"] = (project_scope["cumulative_progress_sum"] / denom * 100.0).fillna(0.0)

    overall_scope = _filter_scope_frame(status_overall, projects=projects, months=months)
    if overall_scope.empty and not project_scope.empty:
        overall_scope = (
            project_scope.groupby("month", dropna=False)
            .agg(
                projects_total=("project_scope_key", lambda s: int(s.fillna("").astype(str).str.strip().astype(bool).sum())),
                quantity_primary_sum=("quantity_primary_sum", "sum"),
                cumulative_progress_sum=("cumulative_progress_sum", "sum"),
                balance_progress_sum=("balance_progress_sum", "sum"),
                plan_for_month_sum=("plan_for_month_sum", "sum"),
                progress_for_month_sum=("progress_for_month_sum", "sum"),
                today_progress_sum=("today_progress_sum", "sum"),
            )
            .reset_index()
        )
        denom = overall_scope["quantity_primary_sum"].where(overall_scope["quantity_primary_sum"] > 0)
        overall_scope["completion_pct"] = (overall_scope["cumulative_progress_sum"] / denom * 100.0).fillna(0.0)

    highlights: list[dict[str, object]] = []
    activity_scope = _filter_scope_frame(status_activity, projects=projects, months=months)
    if not activity_scope.empty:
        if "activity_group" not in activity_scope.columns and "activity_norm" in activity_scope.columns:
            activity_scope["activity_group"] = activity_scope["activity_norm"].fillna("").astype(str)
        for col in ("cumulative_progress", "balance_progress"):
            if col in activity_scope.columns:
                activity_scope[col] = pd.to_numeric(activity_scope[col], errors="coerce")
        highlights = (
            activity_scope.groupby(["month", "project_code", "project_display", "activity_group"], dropna=False)
            .agg(
                cumulative_progress=("cumulative_progress", "sum"),
                balance_progress=("balance_progress", "sum"),
            )
            .reset_index()
            .sort_values(["month", "project_code", "activity_group"])
            .to_dict("records")
        )

    return {
        "project_trend": project_scope.sort_values(["month", "project_code", "line_name"]).to_dict("records") if not project_scope.empty else [],
        "overall_trend": overall_scope.sort_values(["month"]).to_dict("records") if not overall_scope.empty else [],
        "activity_highlights": highlights,
    }


def _build_stretch_readiness_contract(
    stretch_section_fact: pd.DataFrame,
    *,
    projects: Sequence[str] | None,
    months: Sequence[pd.Timestamp] | None,
) -> dict[str, object]:
    scope = _filter_scope_frame(stretch_section_fact, projects=projects, months=months)
    if scope.empty:
        return {"project_line_trend": [], "overall_trend": [], "section_sample": []}

    if "readiness_state" not in scope.columns:
        scope["readiness_state"] = "UNKNOWN"
    readiness = scope["readiness_state"].fillna("").astype(str).str.upper()
    scope["is_ready"] = readiness.eq("READY")
    scope["is_partial"] = readiness.eq("PARTIAL")
    scope["is_not_ready"] = readiness.eq("NOT_READY")
    scope["is_unknown"] = readiness.eq("UNKNOWN")
    length_source = scope["length_km"] if "length_km" in scope.columns else pd.Series(np.nan, index=scope.index)
    scope["length_km"] = pd.to_numeric(length_source, errors="coerce")
    scope["ready_length_km"] = scope["length_km"].where(scope["is_ready"])

    project_line = (
        scope.groupby(["month", "project_code", "project_display", "line_name"], dropna=False)
        .agg(
            sections_total=("section_id", lambda s: int(s.fillna("").astype(str).str.strip().astype(bool).sum())),
            sections_ready=("is_ready", "sum"),
            sections_partial=("is_partial", "sum"),
            sections_not_ready=("is_not_ready", "sum"),
            sections_unknown=("is_unknown", "sum"),
            ready_km=("ready_length_km", "sum"),
            total_km=("length_km", "sum"),
        )
        .reset_index()
    )
    project_line["readiness_pct"] = (
        project_line["sections_ready"] / project_line["sections_total"].replace(0, np.nan) * 100.0
    ).fillna(0.0)

    overall = (
        project_line.groupby("month", dropna=False)
        .agg(
            sections_total=("sections_total", "sum"),
            sections_ready=("sections_ready", "sum"),
            sections_partial=("sections_partial", "sum"),
            sections_not_ready=("sections_not_ready", "sum"),
            sections_unknown=("sections_unknown", "sum"),
            ready_km=("ready_km", "sum"),
            total_km=("total_km", "sum"),
        )
        .reset_index()
    )
    overall["readiness_pct"] = (
        overall["sections_ready"] / overall["sections_total"].replace(0, np.nan) * 100.0
    ).fillna(0.0)

    sample_cols = [col for col in ("project_code", "project_display", "line_name", "section_id", "readiness_state", "length_km", "month") if col in scope.columns]
    sample = scope[sample_cols].head(500).to_dict("records") if sample_cols else []
    return {
        "project_line_trend": project_line.sort_values(["month", "project_code", "line_name"]).to_dict("records"),
        "overall_trend": overall.sort_values(["month"]).to_dict("records"),
        "section_sample": sample,
    }


def _build_manpower_productivity_contract(
    manpower_fact: pd.DataFrame,
    *,
    projects: Sequence[str] | None,
    months: Sequence[pd.Timestamp] | None,
    gangs: Sequence[str] | None,
) -> dict[str, object]:
    scope = _filter_scope_frame(manpower_fact, projects=projects, months=months, gangs=gangs)
    if scope.empty:
        return {"pairings": [], "project_line_day": [], "availability_summary": []}

    for col in ("daily_km", "manpower_gang_strength", "manpower_fitters"):
        if col in scope.columns:
            scope[col] = pd.to_numeric(scope[col], errors="coerce")
    if "date" in scope.columns:
        scope["date"] = pd.to_datetime(scope["date"], errors="coerce").dt.normalize()

    pairings = (
        scope.groupby(["project_code", "project_display", "line_name", "date", "gang_name"], dropna=False)
        .agg(
            daily_km=("daily_km", "sum"),
            manpower_gang_strength=("manpower_gang_strength", "mean"),
            manpower_fitters=("manpower_fitters", "mean"),
            rows=("span_key", "count"),
            availability=("availability", "first"),
        )
        .reset_index()
    )

    project_line_day = (
        scope.groupby(["project_code", "project_display", "line_name", "date"], dropna=False)
        .agg(
            output_km=("daily_km", "sum"),
            manpower_gang_strength=("manpower_gang_strength", "mean"),
            manpower_fitters=("manpower_fitters", "mean"),
            active_gangs=("gang_name", lambda s: int(s.fillna("").astype(str).str.strip().astype(bool).nunique())),
            rows=("span_key", "count"),
        )
        .reset_index()
    )

    availability_summary = (
        scope.groupby(["project_code", "project_display", "availability"], dropna=False)
        .agg(rows=("span_key", "count"))
        .reset_index()
        .to_dict("records")
    )

    return {
        "pairings": pairings.sort_values(["date", "project_code", "line_name", "gang_name"]).head(2000).to_dict("records"),
        "project_line_day": project_line_day.sort_values(["date", "project_code", "line_name"]).head(2000).to_dict("records"),
        "availability_summary": availability_summary,
    }


__all__ = ["build_stringing_analytics_payload"]
