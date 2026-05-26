"""Standalone stretch readiness analysis based on erection tower tightening."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import stringing_ingest as ingest
from .project_identity import build_project_display, build_project_scope_key, normalize_line_name
from .stretch_readiness_ingest import (
    _build_required_locations,
    _is_valid_date_value,
    _normalize_location_token,
    _pick_column,
    _pick_location_nos_column,
    _report_timestamp_with_fallback,
    _scope_key_for_match,
)
from .stringing import add_length_units, normalize_stringing_columns


EXECUTIVE_SUMMARY_COLUMNS = ["Metric", "Value"]
PROJECT_SUMMARY_COLUMNS = [
    "project_code",
    "project_display",
    "project_scope_keys",
    "erection_rows",
    "unique_towers",
    "erected_towers",
    "tightened_towers",
    "erected_not_tightened_towers",
    "usable_stringing_spans",
    "all_erected_spans",
    "all_tightened_spans",
    "already_strung_spans",
    "ready_to_string_spans",
    "ready_to_string_km",
    "data_bucket",
    "coverage_reason",
]
SPAN_READINESS_COLUMNS = [
    "project_code",
    "project_display",
    "project_scope_key",
    "line_name",
    "source_file",
    "source_sheet",
    "source_row_number",
    "from_ap",
    "to_ap",
    "stretch_identifier",
    "length_km",
    "required_location_count",
    "erected_location_count",
    "tightened_location_count",
    "missing_erection_count",
    "missing_tightening_count",
    "required_locations",
    "erected_locations",
    "tightened_locations",
    "missing_erection_locations",
    "missing_tightening_locations",
    "location_parse_status",
    "location_parse_issue",
    "already_strung",
    "all_erected",
    "all_tightened",
    "ready_to_string",
    "readiness_reason",
]
TOWER_GAP_COLUMNS = [
    "project_code",
    "project_display",
    "project_scope_key",
    "line_name",
    "location_no",
    "complete_date",
    "status",
    "tower_tightening_raw",
    "tower_tightening",
    "source_file",
    "source_sheet",
]
COVERAGE_COLUMNS = [
    "project_code",
    "project_display",
    "data_bucket",
    "has_erection_rows",
    "has_tightening_values",
    "has_usable_stringing_spans",
    "erected_towers",
    "tightened_towers",
    "usable_stringing_spans",
    "ready_to_string_spans",
    "reason",
]
ASSUMPTION_COLUMNS = ["Topic", "Assumption"]

READY_TIGHTENING_TOKENS = "dates; yes; y; true; ok; done; completed; complete; c"
BLOCKED_TIGHTENING_TOKENS = "no; n; false; pending; wip; balance; row; hold; blocked"


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _series_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series("", index=frame.index, dtype="object")


def _completion_signal(value: object, *, parsed_date: object = None) -> bool:
    if parsed_date is not None:
        try:
            if pd.notna(pd.to_datetime(parsed_date, errors="coerce")):
                return True
        except Exception:
            pass
    return _is_valid_date_value(value)


def _is_erection_complete(complete_date: object, status: object) -> bool:
    try:
        if pd.notna(pd.to_datetime(complete_date, errors="coerce")):
            return True
    except Exception:
        pass
    return _is_valid_date_value(status)


def _is_already_strung(row: pd.Series) -> bool:
    for column in ("po_completion_date", "fs_complete_date", "status"):
        if column in row.index and _is_valid_date_value(row.get(column)):
            return True
    return False


def _derive_stretch_identifier(row: pd.Series, from_ap: str, to_ap: str) -> str:
    for column in ("stretch_identifier", "section", "section_name", "section_id", "section_label", "Sec", "sec"):
        text = _safe_text(row.get(column))
        if text:
            return text
    if from_ap and to_ap:
        return f"{from_ap} - {to_ap}"
    return from_ap or to_ap


def _join_locations(values: Iterable[str]) -> str:
    return ", ".join([value for value in values if value])


def _normalize_erection_towers(erection_raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(erection_raw, pd.DataFrame) or erection_raw.empty:
        return pd.DataFrame()

    work = erection_raw.copy()
    location_col = _pick_column(work, ("location no", "location no.", "location_no", "location number", "location"))
    if not location_col:
        return pd.DataFrame()

    project_col = _pick_column(work, ("project_code", "project code", "project"))
    scope_col = _pick_column(work, ("project_scope_key", "project scope key"))
    line_col = _pick_column(work, ("line_name", "line name"))
    display_col = _pick_column(work, ("project_display", "project display", "project_name", "project name"))
    source_file_col = _pick_column(work, ("source_file", "source file"))
    source_sheet_col = _pick_column(work, ("source_sheet", "source sheet"))
    complete_col = _pick_column(work, ("complete date", "completion date", "end date", "doc"))
    status_col = _pick_column(work, ("status", "erection status"))
    tightening_raw_col = _pick_column(work, ("tower tightening raw", "tower tightening", "tower_tightening", "tower tightening date", "tightening date"))
    tightening_date_col = _pick_column(work, ("tower tightening", "tower tightening date", "tightening date"))

    out = pd.DataFrame(index=work.index)
    out["project_code"] = _series_or_empty(work, project_col or "").map(_safe_text) if project_col else ""
    out["line_name"] = _series_or_empty(work, line_col or "").map(normalize_line_name) if line_col else ""
    out["project_display"] = _series_or_empty(work, display_col or "").map(_safe_text) if display_col else ""
    out["project_scope_key"] = _series_or_empty(work, scope_col or "").map(_safe_text) if scope_col else ""
    out["location_no"] = work[location_col].map(_safe_text)
    out["loc_norm"] = work[location_col].map(_normalize_location_token)
    out["complete_date"] = _series_or_empty(work, complete_col or "") if complete_col else pd.NaT
    out["status"] = _series_or_empty(work, status_col or "").map(_safe_text) if status_col else ""
    out["tower_tightening_raw"] = _series_or_empty(work, tightening_raw_col or "").map(_safe_text) if tightening_raw_col else ""
    out["tower_tightening"] = _series_or_empty(work, tightening_date_col or "") if tightening_date_col else pd.NaT
    out["source_file"] = _series_or_empty(work, source_file_col or "").map(_safe_text) if source_file_col else ""
    out["source_sheet"] = _series_or_empty(work, source_sheet_col or "").map(_safe_text) if source_sheet_col else ""

    missing_display = ~out["project_display"].astype(bool)
    if missing_display.any():
        out.loc[missing_display, "project_display"] = [
            build_project_display(code, line, code) or code
            for code, line in zip(out.loc[missing_display, "project_code"], out.loc[missing_display, "line_name"])
        ]
    missing_scope = ~out["project_scope_key"].astype(bool)
    if missing_scope.any():
        out.loc[missing_scope, "project_scope_key"] = [
            build_project_scope_key(code, line, display)
            for code, line, display in zip(
                out.loc[missing_scope, "project_code"],
                out.loc[missing_scope, "line_name"],
                out.loc[missing_scope, "project_display"],
            )
        ]

    out["project_key"] = out["project_code"].map(ingest.normalize_project_code_key)
    out["scope_norm"] = [
        _scope_key_for_match(scope, code, line)
        for scope, code, line in zip(out["project_scope_key"], out["project_code"], out["line_name"])
    ]
    out["erected"] = [
        _is_erection_complete(complete, status)
        for complete, status in zip(out["complete_date"], out["status"])
    ]
    out["tightened"] = [
        _completion_signal(raw, parsed_date=parsed)
        for raw, parsed in zip(out["tower_tightening_raw"], out["tower_tightening"])
    ]
    out["report_ts"] = [
        _report_timestamp_with_fallback("", source_file)
        for source_file in out["source_file"]
    ]
    out["_seq"] = range(len(out.index))
    out = out[out["project_key"].astype(bool) & out["loc_norm"].astype(bool)].copy()
    if out.empty:
        return out
    out = out.sort_values(["scope_norm", "project_key", "loc_norm", "report_ts", "_seq"])
    out = out.drop_duplicates(subset=["scope_norm", "project_key", "loc_norm"], keep="last")
    return out.reset_index(drop=True)


def _normalize_stringing_spans(stringing_compiled_raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(stringing_compiled_raw, pd.DataFrame) or stringing_compiled_raw.empty:
        return pd.DataFrame()
    compiled, _ = normalize_stringing_columns(stringing_compiled_raw)
    compiled, _ = add_length_units(compiled)
    work = compiled.copy()
    for column in (
        "project_code",
        "project_name",
        "project_display",
        "project",
        "line_name",
        "project_scope_key",
        "from_ap",
        "to_ap",
        "source_file",
        "_source_file",
        "source_sheet",
        "length_km",
        "po_completion_date",
        "fs_complete_date",
        "status",
    ):
        if column not in work.columns:
            work[column] = ""
    location_nos_col = _pick_location_nos_column(work)
    work["__location_nos_raw"] = work[location_nos_col] if location_nos_col else ""
    return work


def _build_lookup_maps(towers: pd.DataFrame) -> tuple[dict[tuple[str, str], pd.Series], dict[tuple[str, str], pd.Series]]:
    by_scope: dict[tuple[str, str], pd.Series] = {}
    by_project: dict[tuple[str, str], pd.Series] = {}
    if towers.empty:
        return by_scope, by_project
    for _, row in towers.iterrows():
        loc = _safe_text(row.get("loc_norm"))
        if not loc:
            continue
        scope = _safe_text(row.get("scope_norm"))
        if scope:
            by_scope[(scope, loc)] = row
        project_key = _safe_text(row.get("project_key"))
        if project_key:
            by_project[(project_key, loc)] = row
    return by_scope, by_project


def _build_span_readiness(towers: pd.DataFrame, stringing_spans: pd.DataFrame) -> pd.DataFrame:
    if stringing_spans.empty:
        return pd.DataFrame(columns=SPAN_READINESS_COLUMNS)
    by_scope, by_project = _build_lookup_maps(towers)
    rows: list[dict[str, object]] = []

    for idx, row in stringing_spans.iterrows():
        project_code = _safe_text(row.get("project_code")) or _safe_text(row.get("project")) or _safe_text(row.get("project_name"))
        project_key = ingest.normalize_project_code_key(project_code)
        if not project_key:
            continue
        line_name = normalize_line_name(row.get("line_name"))
        project_display = _safe_text(row.get("project_display")) or _safe_text(row.get("project_name")) or project_code
        if not project_display:
            project_display = build_project_display(project_code, line_name, project_code) or project_code
        project_scope_key = _safe_text(row.get("project_scope_key")) or build_project_scope_key(project_code, line_name, project_display)
        scope_norm = _scope_key_for_match(project_scope_key, project_code, line_name)
        from_ap = _safe_text(row.get("from_ap") or row.get("from"))
        to_ap = _safe_text(row.get("to_ap") or row.get("to"))
        required, parse_status, parse_issue = _build_required_locations(from_ap, to_ap, row.get("__location_nos_raw"))
        if not required:
            continue

        erected_locations: list[str] = []
        tightened_locations: list[str] = []
        missing_erection: list[str] = []
        missing_tightening: list[str] = []
        for loc in required:
            tower = by_scope.get((scope_norm, loc))
            if tower is None:
                tower = by_project.get((project_key, loc))
            erected = bool(tower is not None and tower.get("erected", False))
            tightened = bool(tower is not None and tower.get("tightened", False))
            if erected:
                erected_locations.append(loc)
            else:
                missing_erection.append(loc)
            if tightened:
                tightened_locations.append(loc)
            else:
                missing_tightening.append(loc)

        already_strung = _is_already_strung(row)
        all_erected = bool(required and not missing_erection)
        all_tightened = bool(required and not missing_tightening)
        ready = bool(all_erected and all_tightened and not already_strung)
        if ready:
            reason = "READY_TO_STRING"
        elif already_strung:
            reason = "ALREADY_STRUNG"
        elif not all_erected:
            reason = "ERECTION_NOT_COMPLETE"
        elif not all_tightened:
            reason = "TIGHTENING_PENDING"
        else:
            reason = "NOT_READY"
        length_km = pd.to_numeric(pd.Series([row.get("length_km")]), errors="coerce").iloc[0]
        rows.append(
            {
                "project_code": project_code,
                "project_display": project_display,
                "project_scope_key": project_scope_key,
                "line_name": line_name,
                "source_file": _safe_text(row.get("source_file")) or _safe_text(row.get("_source_file")),
                "source_sheet": _safe_text(row.get("source_sheet")),
                "source_row_number": int(pd.to_numeric(pd.Series([row.get("source_row_number", idx + 1)]), errors="coerce").fillna(idx + 1).iloc[0]),
                "from_ap": from_ap,
                "to_ap": to_ap,
                "stretch_identifier": _derive_stretch_identifier(row, from_ap, to_ap),
                "length_km": float(length_km) if pd.notna(length_km) else None,
                "required_location_count": int(len(required)),
                "erected_location_count": int(len(erected_locations)),
                "tightened_location_count": int(len(tightened_locations)),
                "missing_erection_count": int(len(missing_erection)),
                "missing_tightening_count": int(len(missing_tightening)),
                "required_locations": _join_locations(required),
                "erected_locations": _join_locations(erected_locations),
                "tightened_locations": _join_locations(tightened_locations),
                "missing_erection_locations": _join_locations(missing_erection),
                "missing_tightening_locations": _join_locations(missing_tightening),
                "location_parse_status": parse_status,
                "location_parse_issue": parse_issue,
                "already_strung": already_strung,
                "all_erected": all_erected,
                "all_tightened": all_tightened,
                "ready_to_string": ready,
                "readiness_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=SPAN_READINESS_COLUMNS)


def _build_tower_gap(towers: pd.DataFrame) -> pd.DataFrame:
    if towers.empty:
        return pd.DataFrame(columns=TOWER_GAP_COLUMNS)
    gap = towers[towers["erected"].astype(bool) & ~towers["tightened"].astype(bool)].copy()
    if gap.empty:
        return pd.DataFrame(columns=TOWER_GAP_COLUMNS)
    return gap.reindex(columns=TOWER_GAP_COLUMNS).reset_index(drop=True)


def _coverage_reason(bucket: str, tightened: int, usable_spans: int) -> str:
    if bucket == "INCLUDED":
        return "Erection tower tightening and usable stringing spans are available."
    if tightened > 0 and usable_spans <= 0:
        return "Erection tower tightening is available, but no usable stringing spans were found."
    if usable_spans > 0 and tightened <= 0:
        return "Usable stringing spans are available, but no erection-sheet tower tightening values were found."
    return "No erection-sheet tower tightening values and no usable stringing span link were found."


def _build_project_summary(towers: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    tower_rows: list[dict[str, object]] = []
    if not towers.empty:
        for project_key, group in towers.groupby("project_key", dropna=False):
            project_code = _safe_text(group["project_code"].iloc[0])
            tower_rows.append(
                {
                    "project_key": project_key,
                    "project_code": project_code,
                    "project_display": _safe_text(group["project_display"].iloc[0]) or project_code,
                    "project_scope_keys": "; ".join(sorted(set(group["project_scope_key"].dropna().astype(str).str.strip()))),
                    "erection_rows": int(len(group.index)),
                    "unique_towers": int(group["loc_norm"].nunique()),
                    "erected_towers": int(group["erected"].sum()),
                    "tightened_towers": int(group["tightened"].sum()),
                    "erected_not_tightened_towers": int((group["erected"].astype(bool) & ~group["tightened"].astype(bool)).sum()),
                }
            )
    tower_summary = pd.DataFrame(tower_rows)
    if tower_summary.empty:
        tower_summary = pd.DataFrame(columns=["project_key"])

    span_rows: list[dict[str, object]] = []
    if not spans.empty:
        work = spans.copy()
        work["project_key"] = work["project_code"].map(ingest.normalize_project_code_key)
        for project_key, group in work.groupby("project_key", dropna=False):
            ready_mask = group["ready_to_string"].astype(bool)
            span_rows.append(
                {
                    "project_key": project_key,
                    "project_code_span": _safe_text(group["project_code"].iloc[0]),
                    "project_display_span": _safe_text(group["project_display"].iloc[0]),
                    "usable_stringing_spans": int(len(group.index)),
                    "all_erected_spans": int(group["all_erected"].sum()),
                    "all_tightened_spans": int(group["all_tightened"].sum()),
                    "already_strung_spans": int(group["already_strung"].sum()),
                    "ready_to_string_spans": int(ready_mask.sum()),
                    "ready_to_string_km": float(pd.to_numeric(group.loc[ready_mask, "length_km"], errors="coerce").fillna(0.0).sum()),
                }
            )
    span_summary = pd.DataFrame(span_rows)
    if span_summary.empty:
        span_summary = pd.DataFrame(columns=["project_key"])

    if tower_summary.empty and span_summary.empty:
        return pd.DataFrame(columns=PROJECT_SUMMARY_COLUMNS)
    merged = pd.merge(tower_summary, span_summary, on="project_key", how="outer")
    if "project_code" not in merged.columns:
        merged["project_code"] = ""
    if "project_code_span" not in merged.columns:
        merged["project_code_span"] = ""
    if "project_display" not in merged.columns:
        merged["project_display"] = ""
    if "project_display_span" not in merged.columns:
        merged["project_display_span"] = ""
    merged["project_code"] = merged["project_code"].fillna("").astype(str).str.strip()
    merged.loc[~merged["project_code"].astype(bool), "project_code"] = (
        merged.loc[~merged["project_code"].astype(bool), "project_code_span"].fillna("").astype(str).str.strip()
    )
    merged["project_display"] = merged["project_display"].fillna("").astype(str).str.strip()
    merged.loc[~merged["project_display"].astype(bool), "project_display"] = (
        merged.loc[~merged["project_display"].astype(bool), "project_display_span"].fillna("").astype(str).str.strip()
    )
    for column in (
        "erection_rows",
        "unique_towers",
        "erected_towers",
        "tightened_towers",
        "erected_not_tightened_towers",
        "usable_stringing_spans",
        "all_erected_spans",
        "all_tightened_spans",
        "already_strung_spans",
        "ready_to_string_spans",
    ):
        if column not in merged.columns:
            merged[column] = 0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype(int)
    if "ready_to_string_km" not in merged.columns:
        merged["ready_to_string_km"] = 0.0
    merged["ready_to_string_km"] = pd.to_numeric(merged["ready_to_string_km"], errors="coerce").fillna(0.0)
    if "project_scope_keys" not in merged.columns:
        merged["project_scope_keys"] = ""
    merged["project_scope_keys"] = merged["project_scope_keys"].fillna("").astype(str)

    buckets: list[str] = []
    reasons: list[str] = []
    for _, row in merged.iterrows():
        tightened = int(row.get("tightened_towers", 0))
        usable_spans = int(row.get("usable_stringing_spans", 0))
        if tightened > 0 and usable_spans > 0:
            bucket = "INCLUDED"
        elif tightened > 0 or usable_spans > 0:
            bucket = "PARTIAL"
        else:
            bucket = "NO_TIGHTENING_OR_LINK"
        buckets.append(bucket)
        reasons.append(_coverage_reason(bucket, tightened, usable_spans))
    merged["data_bucket"] = buckets
    merged["coverage_reason"] = reasons
    return merged.reindex(columns=PROJECT_SUMMARY_COLUMNS).sort_values("project_code").reset_index(drop=True)


def _build_coverage(project_summary: pd.DataFrame) -> pd.DataFrame:
    if project_summary.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    coverage = project_summary.copy()
    coverage["has_erection_rows"] = pd.to_numeric(coverage["erection_rows"], errors="coerce").fillna(0).gt(0)
    coverage["has_tightening_values"] = pd.to_numeric(coverage["tightened_towers"], errors="coerce").fillna(0).gt(0)
    coverage["has_usable_stringing_spans"] = pd.to_numeric(coverage["usable_stringing_spans"], errors="coerce").fillna(0).gt(0)
    coverage["reason"] = coverage["coverage_reason"]
    return coverage.reindex(columns=COVERAGE_COLUMNS)


def _build_executive_summary(project_summary: pd.DataFrame) -> pd.DataFrame:
    if project_summary.empty:
        rows = [
            ("Projects total", 0),
            ("Projects included", 0),
            ("Projects partial", 0),
            ("Projects no tightening/link", 0),
        ]
        return pd.DataFrame(rows, columns=EXECUTIVE_SUMMARY_COLUMNS)
    bucket = project_summary["data_bucket"].fillna("").astype(str)
    rows = [
        ("Projects total", int(len(project_summary.index))),
        ("Projects included", int(bucket.eq("INCLUDED").sum())),
        ("Projects partial", int(bucket.eq("PARTIAL").sum())),
        ("Projects no tightening/link", int(bucket.eq("NO_TIGHTENING_OR_LINK").sum())),
        ("Erected towers", int(pd.to_numeric(project_summary["erected_towers"], errors="coerce").fillna(0).sum())),
        ("Tightened towers", int(pd.to_numeric(project_summary["tightened_towers"], errors="coerce").fillna(0).sum())),
        ("Erected not tightened towers", int(pd.to_numeric(project_summary["erected_not_tightened_towers"], errors="coerce").fillna(0).sum())),
        ("Usable stringing spans", int(pd.to_numeric(project_summary["usable_stringing_spans"], errors="coerce").fillna(0).sum())),
        ("Ready to string spans", int(pd.to_numeric(project_summary["ready_to_string_spans"], errors="coerce").fillna(0).sum())),
        ("Ready to string km", round(float(pd.to_numeric(project_summary["ready_to_string_km"], errors="coerce").fillna(0.0).sum()), 3)),
    ]
    return pd.DataFrame(rows, columns=EXECUTIVE_SUMMARY_COLUMNS)


def _build_assumptions() -> pd.DataFrame:
    rows = [
        ("Source policy", "Only Tower Tightening Raw / Tower Tightening fields from erection compiled raw data are used."),
        ("Tightening complete tokens", READY_TIGHTENING_TOKENS),
        ("Tightening blocker tokens", BLOCKED_TIGHTENING_TOKENS),
        ("Ready-to-string rule", "All required locations erected and tightened, and no paying-out/final-sag/status completion recorded."),
        ("Location matching", "Stringing endpoints plus Location Nos are matched to normalized erection location numbers by project scope, then project fallback."),
    ]
    return pd.DataFrame(rows, columns=ASSUMPTION_COLUMNS)


def build_stretch_tightening_readiness_tables(
    *,
    erection_raw: pd.DataFrame,
    stringing_compiled_raw: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build standalone stretch tightening readiness analysis tables."""
    towers = _normalize_erection_towers(erection_raw)
    stringing_spans = _normalize_stringing_spans(stringing_compiled_raw)
    span_readiness = _build_span_readiness(towers, stringing_spans)
    tower_gap = _build_tower_gap(towers)
    project_summary = _build_project_summary(towers, span_readiness)
    coverage = _build_coverage(project_summary)
    executive_summary = _build_executive_summary(project_summary)
    assumptions = _build_assumptions()
    return {
        "Executive Summary": executive_summary,
        "Project Summary": project_summary,
        "Span Readiness": span_readiness,
        "Tower Gap": tower_gap,
        "Coverage": coverage,
        "Assumptions": assumptions,
    }


def write_stretch_tightening_readiness_workbook(
    output_path: str | Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    """Write the stretch tightening readiness analysis workbook."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = [
        "Executive Summary",
        "Project Summary",
        "Span Readiness",
        "Tower Gap",
        "Coverage",
        "Assumptions",
    ]
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in ordered:
            table = tables.get(sheet, pd.DataFrame())
            pd.DataFrame([[sheet]]).to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=0)
            table.to_excel(writer, sheet_name=sheet, index=False, startrow=1)
    return output
