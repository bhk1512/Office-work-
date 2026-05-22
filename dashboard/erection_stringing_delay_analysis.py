"""Erection->Stringing and PO->FS delay analysis builders and workbook writer."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .plan_utils import compact_project_key
from .project_identity import extract_base_project_code
from .stringing_analytics import (
    _build_erection_po_gap_table,
    _filter_method,
    _normalize_stringing_compiled,
    _resolve_project_key_norm,
)


_SCOPE_PREFIX_CODE_RE = re.compile(r"^(T[A-Z])(\d{3,4})", flags=re.IGNORECASE)


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    return text


def _coerce_date(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _method_split(value: object) -> str:
    text = _safe_text(value).lower()
    if "tse" in text:
        return "TSE"
    return "Others"


def _preferred_project_key(
    frame: pd.DataFrame,
    *,
    scope_candidates: tuple[str, ...],
) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="object")

    fallback = frame.get("project_key_norm")
    if fallback is None:
        fallback = _resolve_project_key_norm(frame)
    else:
        fallback = fallback.fillna("").astype(str)
        missing = ~fallback.astype(bool)
        if bool(missing.any()):
            fallback = fallback.where(~missing, _resolve_project_key_norm(frame))
    fallback = fallback.fillna("").astype(str).map(compact_project_key)

    scope_series = pd.Series("", index=frame.index, dtype="object")
    for column in scope_candidates:
        if column not in frame.columns:
            continue
        candidate = frame[column].fillna("").astype(str).str.strip().map(compact_project_key)
        scope_series = scope_series.where(scope_series.astype(bool), candidate)
    return scope_series.where(scope_series.astype(bool), fallback)


def _extract_base_code_from_compact_scope(value: object) -> str:
    compact = compact_project_key(value)
    if not compact:
        return ""
    match = _SCOPE_PREFIX_CODE_RE.match(compact.upper())
    if not match:
        return ""
    prefix = match.group(1).upper()
    return f"{prefix} {int(match.group(2))}"


def _derive_base_project_code(
    *,
    project_code: object,
    project_display: object,
    project_name: object,
    project_scope_key: object,
    join_project_key: object,
) -> str:
    for value in (project_code, project_display, project_name, project_scope_key):
        code = extract_base_project_code(value)
        if code:
            return code
    for value in (project_scope_key, join_project_key):
        code = _extract_base_code_from_compact_scope(value)
        if code:
            return code
    return ""


def _series_label(base_code: object) -> str:
    text = _safe_text(base_code)
    match = re.search(r"\d{3,4}", text)
    if not match:
        return "Other"
    lead = match.group(0)[0]
    if lead in {"4", "5", "6"}:
        return f"{lead}xx"
    return "Other"


def _client_type_label(base_code: object) -> str:
    text = _safe_text(base_code).upper()
    if text.startswith("TA"):
        return "TA"
    if text.startswith("TB"):
        return "TB"
    return "Other"


def _ordered_group_values(series: pd.Series, preferred: tuple[str, ...]) -> list[str]:
    values = series.fillna("").astype(str).str.strip()
    present = {value for value in values.tolist() if value}
    ordered = [value for value in preferred if value in present]
    remaining = sorted(present - set(preferred))
    return ordered + remaining


def _empty_es_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Scope Type",
            "Scope Name",
            "Method Split",
            "Spans Considered",
            "Spans With Computable Lag",
            "Lag Coverage %",
            "Average Lag Days",
            "Median Lag Days",
            "% Lag >60 Days",
            "Negative Lag Excluded",
            "Fallback Rows",
            "Fallback %",
        ]
    )


def _empty_es_coverage_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Scope Type",
            "Scope Name",
            "Method Split",
            "Spans Considered",
            "Spans With Computable Lag",
            "Lag Coverage %",
            "Fallback Rows",
            "Fallback %",
            "LOCATION_NOS Rows",
            "HYBRID_PARTIAL_FALLBACK Rows",
            "ALPHABETIC_FALLBACK Rows",
            "Parse EMPTY Rows",
            "Parse OK Rows",
            "Parse PARTIAL_PARSE Rows",
            "Parse SHORTHAND_NO_ANCHOR Rows",
            "Negative Lag Excluded",
            "Average Lag Days",
            "Median Lag Days",
        ]
    )


def _empty_es_anomaly_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Project",
            "Project Scope Key",
            "Series",
            "Client Type",
            "Method Split",
            "Span Key",
            "From AP",
            "To AP",
            "PO Start Date",
            "Last Erection Completion Date",
            "Gap Days",
            "Issue",
            "Lag Inference Mode",
            "Location Parse Status",
            "Required Location Count",
            "Matched Location Count",
            "Unmatched Location Count",
            "Fallback Used",
        ]
    )


def _empty_es_detail_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Project",
            "Project Code",
            "Project Display",
            "Project Scope Key",
            "Join Project Key",
            "Base Project Code",
            "Series",
            "Client Type",
            "Method Split",
            "Section",
            "Span",
            "Span Key",
            "Gang Name",
            "From AP",
            "To AP",
            "PO Start Date",
            "Last Erection Completion Date",
            "Gap Days",
            "Gap Days Non-Negative",
            "Lag Inference Mode",
            "Fallback Used",
            "Location Parse Status",
            "Location Parse Issue",
            "Required Location Count",
            "Matched Location Count",
            "Unmatched Location Count",
            "Required Locations",
            "Matched Locations",
            "Unmatched Locations",
            "Location Nos Raw",
            "Project Key In Erection Map",
        ]
    )


def _empty_pofs_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Scope Type",
            "Scope Name",
            "Method Split",
            "Spans Considered",
            "Spans With Computable Lag",
            "Lag Coverage %",
            "Average Lag Days",
            "Median Lag Days",
            "% Lag >15 Days",
            "Negative Lag Excluded",
        ]
    )


def _empty_pofs_coverage_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Scope Type",
            "Scope Name",
            "Method Split",
            "Spans Considered",
            "Spans With Computable Lag",
            "Lag Coverage %",
            "Negative Lag Excluded",
            "Negative Gap Flag Rows",
            "Missing PO Completion Rows",
            "Missing FS Start Rows",
            "Average Lag Days",
            "Median Lag Days",
        ]
    )


def _empty_pofs_anomaly_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Project",
            "Project Scope Key",
            "Series",
            "Client Type",
            "Method Split",
            "Span Key",
            "From AP",
            "To AP",
            "PO Completion Date",
            "FS Starting Date",
            "Gap Days",
            "Issue",
            "Negative Gap Flag",
        ]
    )


def _empty_pofs_detail_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Project",
            "Project Code",
            "Project Display",
            "Project Scope Key",
            "Join Project Key",
            "Base Project Code",
            "Series",
            "Client Type",
            "Method Split",
            "Section",
            "Span",
            "Span Key",
            "Gang Name",
            "From AP",
            "To AP",
            "PO Completion Date",
            "FS Starting Date",
            "Gap Days",
            "Gap Days Non-Negative",
            "Negative Gap Flag",
            "Has PO Completion Date",
            "Has FS Starting Date",
        ]
    )


def _aggregate_es_row(scope: pd.DataFrame, *, scope_type: str, scope_name: str, method_split: str) -> dict[str, object]:
    total = int(len(scope.index))
    if total == 0:
        return {
            "Scope Type": scope_type,
            "Scope Name": scope_name,
            "Method Split": method_split,
            "Spans Considered": 0,
            "Spans With Computable Lag": 0,
            "Lag Coverage %": 0.0,
            "Average Lag Days": pd.NA,
            "Median Lag Days": pd.NA,
            "% Lag >60 Days": pd.NA,
            "Negative Lag Excluded": 0,
            "Fallback Rows": 0,
            "Fallback %": 0.0,
        }

    gaps = pd.to_numeric(scope.get("Gap Days"), errors="coerce")
    computable = gaps.notna()
    non_negative = gaps[computable & gaps.ge(0)]
    spans_with_lag = int(computable.sum())
    coverage_pct = round((spans_with_lag / total) * 100.0, 1) if total else 0.0
    fallback_rows = int(scope.get("Fallback Used", pd.Series(False, index=scope.index)).fillna(False).astype(bool).sum())
    fallback_pct = round((fallback_rows / total) * 100.0, 1) if total else 0.0
    negative_count = int(gaps.lt(0).sum())

    avg_lag = float(non_negative.mean()) if not non_negative.empty else pd.NA
    median_lag = float(non_negative.median()) if not non_negative.empty else pd.NA
    pct_over_60 = float((non_negative > 60).mean() * 100.0) if not non_negative.empty else pd.NA

    return {
        "Scope Type": scope_type,
        "Scope Name": scope_name,
        "Method Split": method_split,
        "Spans Considered": total,
        "Spans With Computable Lag": spans_with_lag,
        "Lag Coverage %": coverage_pct,
        "Average Lag Days": round(avg_lag, 1) if pd.notna(avg_lag) else pd.NA,
        "Median Lag Days": round(median_lag, 1) if pd.notna(median_lag) else pd.NA,
        "% Lag >60 Days": round(pct_over_60, 1) if pd.notna(pct_over_60) else pd.NA,
        "Negative Lag Excluded": negative_count,
        "Fallback Rows": fallback_rows,
        "Fallback %": fallback_pct,
    }


def _coverage_es_row(scope: pd.DataFrame, *, scope_type: str, scope_name: str, method_split: str) -> dict[str, object]:
    summary = _aggregate_es_row(scope, scope_type=scope_type, scope_name=scope_name, method_split=method_split)
    mode = scope.get("Lag Inference Mode", pd.Series("", index=scope.index)).fillna("").astype(str)
    parse = scope.get("Location Parse Status", pd.Series("", index=scope.index)).fillna("").astype(str)
    return {
        "Scope Type": scope_type,
        "Scope Name": scope_name,
        "Method Split": method_split,
        "Spans Considered": summary["Spans Considered"],
        "Spans With Computable Lag": summary["Spans With Computable Lag"],
        "Lag Coverage %": summary["Lag Coverage %"],
        "Fallback Rows": summary["Fallback Rows"],
        "Fallback %": summary["Fallback %"],
        "LOCATION_NOS Rows": int(mode.eq("LOCATION_NOS").sum()),
        "HYBRID_PARTIAL_FALLBACK Rows": int(mode.eq("HYBRID_PARTIAL_FALLBACK").sum()),
        "ALPHABETIC_FALLBACK Rows": int(mode.eq("ALPHABETIC_FALLBACK").sum()),
        "Parse EMPTY Rows": int(parse.eq("EMPTY").sum()),
        "Parse OK Rows": int(parse.eq("OK").sum()),
        "Parse PARTIAL_PARSE Rows": int(parse.eq("PARTIAL_PARSE").sum()),
        "Parse SHORTHAND_NO_ANCHOR Rows": int(parse.eq("SHORTHAND_NO_ANCHOR").sum()),
        "Negative Lag Excluded": summary["Negative Lag Excluded"],
        "Average Lag Days": summary["Average Lag Days"],
        "Median Lag Days": summary["Median Lag Days"],
    }


def _aggregate_pofs_row(scope: pd.DataFrame, *, scope_type: str, scope_name: str, method_split: str) -> dict[str, object]:
    total = int(len(scope.index))
    if total == 0:
        return {
            "Scope Type": scope_type,
            "Scope Name": scope_name,
            "Method Split": method_split,
            "Spans Considered": 0,
            "Spans With Computable Lag": 0,
            "Lag Coverage %": 0.0,
            "Average Lag Days": pd.NA,
            "Median Lag Days": pd.NA,
            "% Lag >15 Days": pd.NA,
            "Negative Lag Excluded": 0,
        }

    gaps = pd.to_numeric(scope.get("Gap Days"), errors="coerce")
    computable = gaps.notna()
    non_negative = gaps[computable & gaps.ge(0)]
    spans_with_lag = int(computable.sum())
    coverage_pct = round((spans_with_lag / total) * 100.0, 1) if total else 0.0
    negative_count = int(gaps.lt(0).sum())

    avg_lag = float(non_negative.mean()) if not non_negative.empty else pd.NA
    median_lag = float(non_negative.median()) if not non_negative.empty else pd.NA
    pct_over_15 = float((non_negative > 15).mean() * 100.0) if not non_negative.empty else pd.NA

    return {
        "Scope Type": scope_type,
        "Scope Name": scope_name,
        "Method Split": method_split,
        "Spans Considered": total,
        "Spans With Computable Lag": spans_with_lag,
        "Lag Coverage %": coverage_pct,
        "Average Lag Days": round(avg_lag, 1) if pd.notna(avg_lag) else pd.NA,
        "Median Lag Days": round(median_lag, 1) if pd.notna(median_lag) else pd.NA,
        "% Lag >15 Days": round(pct_over_15, 1) if pd.notna(pct_over_15) else pd.NA,
        "Negative Lag Excluded": negative_count,
    }


def _coverage_pofs_row(scope: pd.DataFrame, *, scope_type: str, scope_name: str, method_split: str) -> dict[str, object]:
    summary = _aggregate_pofs_row(scope, scope_type=scope_type, scope_name=scope_name, method_split=method_split)
    return {
        "Scope Type": scope_type,
        "Scope Name": scope_name,
        "Method Split": method_split,
        "Spans Considered": summary["Spans Considered"],
        "Spans With Computable Lag": summary["Spans With Computable Lag"],
        "Lag Coverage %": summary["Lag Coverage %"],
        "Negative Lag Excluded": summary["Negative Lag Excluded"],
        "Negative Gap Flag Rows": int(scope.get("Negative Gap Flag", pd.Series(False, index=scope.index)).fillna(False).astype(bool).sum()),
        "Missing PO Completion Rows": int((~scope.get("Has PO Completion Date", pd.Series(False, index=scope.index)).fillna(False).astype(bool)).sum()),
        "Missing FS Start Rows": int((~scope.get("Has FS Starting Date", pd.Series(False, index=scope.index)).fillna(False).astype(bool)).sum()),
        "Average Lag Days": summary["Average Lag Days"],
        "Median Lag Days": summary["Median Lag Days"],
    }


def _build_es_anomaly_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return _empty_es_anomaly_table()

    rows: list[dict[str, object]] = []
    for _, row in detail.iterrows():
        gap_value = pd.to_numeric(pd.Series([row.get("Gap Days")]), errors="coerce").iloc[0]
        if pd.notna(gap_value) and float(gap_value) >= 0:
            continue

        if pd.notna(gap_value) and float(gap_value) < 0:
            issue = "NEGATIVE_LAG_EXCLUDED"
        elif not bool(row.get("Project Key In Erection Map", False)):
            issue = "PROJECT_SCOPE_KEY_NO_ERECTION_MATCH"
        elif int(pd.to_numeric(pd.Series([row.get("Required Location Count", 0)]), errors="coerce").fillna(0).iloc[0]) <= 0:
            issue = "INSUFFICIENT_REQUIRED_LOCATIONS"
        elif int(pd.to_numeric(pd.Series([row.get("Matched Location Count", 0)]), errors="coerce").fillna(0).iloc[0]) <= 0:
            issue = "UNMATCHED_REQUIRED_LOCATIONS"
        else:
            issue = "GAP_NOT_COMPUTABLE"

        rows.append(
            {
                "Project": row.get("Project", ""),
                "Project Scope Key": row.get("Project Scope Key", ""),
                "Series": row.get("Series", ""),
                "Client Type": row.get("Client Type", ""),
                "Method Split": row.get("Method Split", ""),
                "Span Key": row.get("Span Key", ""),
                "From AP": row.get("From AP", ""),
                "To AP": row.get("To AP", ""),
                "PO Start Date": row.get("PO Start Date", pd.NaT),
                "Last Erection Completion Date": row.get("Last Erection Completion Date", pd.NaT),
                "Gap Days": row.get("Gap Days", pd.NA),
                "Issue": issue,
                "Lag Inference Mode": row.get("Lag Inference Mode", ""),
                "Location Parse Status": row.get("Location Parse Status", ""),
                "Required Location Count": row.get("Required Location Count", 0),
                "Matched Location Count": row.get("Matched Location Count", 0),
                "Unmatched Location Count": row.get("Unmatched Location Count", 0),
                "Fallback Used": bool(row.get("Fallback Used", False)),
            }
        )
    return pd.DataFrame(rows, columns=_empty_es_anomaly_table().columns)


def _build_pofs_anomaly_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return _empty_pofs_anomaly_table()

    rows: list[dict[str, object]] = []
    for _, row in detail.iterrows():
        gap_value = pd.to_numeric(pd.Series([row.get("Gap Days")]), errors="coerce").iloc[0]
        has_po = bool(row.get("Has PO Completion Date", False))
        has_fs = bool(row.get("Has FS Starting Date", False))

        if pd.notna(gap_value) and float(gap_value) >= 0:
            continue

        if pd.notna(gap_value) and float(gap_value) < 0:
            issue = "NEGATIVE_LAG_EXCLUDED"
        elif not has_po:
            issue = "PO_COMPLETION_DATE_MISSING"
        elif not has_fs:
            issue = "FS_STARTING_DATE_MISSING"
        else:
            issue = "GAP_NOT_COMPUTABLE"

        rows.append(
            {
                "Project": row.get("Project", ""),
                "Project Scope Key": row.get("Project Scope Key", ""),
                "Series": row.get("Series", ""),
                "Client Type": row.get("Client Type", ""),
                "Method Split": row.get("Method Split", ""),
                "Span Key": row.get("Span Key", ""),
                "From AP": row.get("From AP", ""),
                "To AP": row.get("To AP", ""),
                "PO Completion Date": row.get("PO Completion Date", pd.NaT),
                "FS Starting Date": row.get("FS Starting Date", pd.NaT),
                "Gap Days": row.get("Gap Days", pd.NA),
                "Issue": issue,
                "Negative Gap Flag": bool(row.get("Negative Gap Flag", False)),
            }
        )
    return pd.DataFrame(rows, columns=_empty_pofs_anomaly_table().columns)


def _build_es_tables(
    *,
    compiled_scope: pd.DataFrame,
    erection_work: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summary_empty = _empty_es_summary_table()
    coverage_empty = _empty_es_coverage_table()
    anomaly_empty = _empty_es_anomaly_table()
    detail_empty = _empty_es_detail_table()

    if compiled_scope is None or compiled_scope.empty:
        return {
            "ES Delay Summary": summary_empty,
            "ES Delay Coverage": coverage_empty,
            "ES Delay Anomalies": anomaly_empty,
            "ES Delay Detail": detail_empty,
        }

    erection_key_set = set(
        erection_work.get("project_key_norm", pd.Series([], dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s.astype(bool)]
        .tolist()
    )

    gap = _build_erection_po_gap_table(compiled_scope, erection_work).reset_index(drop=True)
    gap["__analysis_row_id"] = pd.Series(range(len(gap.index)), index=gap.index, dtype="int64")

    meta = pd.DataFrame(
        {
            "__analysis_row_id": compiled_scope["__analysis_row_id"],
            "Project Code": compiled_scope.get("project_code", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Project Display": compiled_scope.get("project_display", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Project Scope Key": compiled_scope.get("project_scope_key", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Join Project Key": compiled_scope.get("project_key_norm", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Method Split": compiled_scope.get("method", pd.Series("", index=compiled_scope.index)).map(_method_split),
        }
    )

    detail = gap.merge(meta, on="__analysis_row_id", how="left")
    detail["Project"] = detail.get("project_name", pd.Series("", index=detail.index)).fillna("").astype(str).str.strip()
    detail["Project"] = detail["Project"].where(
        detail["Project"].astype(bool),
        detail["Project Display"].where(detail["Project Display"].astype(bool), detail["Project Code"]),
    )
    detail["Project Key In Erection Map"] = detail["Join Project Key"].fillna("").astype(str).map(lambda key: key in erection_key_set)
    detail["Gap Days"] = pd.to_numeric(detail.get("gap_days"), errors="coerce")
    detail["Gap Days Non-Negative"] = detail["Gap Days"].where(detail["Gap Days"].ge(0))
    detail["PO Start Date"] = pd.to_datetime(detail.get("po_start_date"), errors="coerce").dt.normalize()
    detail["Last Erection Completion Date"] = pd.to_datetime(detail.get("last_erection_completion_date"), errors="coerce").dt.normalize()
    detail["Base Project Code"] = [
        _derive_base_project_code(
            project_code=row.get("Project Code", ""),
            project_display=row.get("Project Display", ""),
            project_name=row.get("Project", ""),
            project_scope_key=row.get("Project Scope Key", ""),
            join_project_key=row.get("Join Project Key", ""),
        )
        for _, row in detail.iterrows()
    ]
    detail["Series"] = detail["Base Project Code"].map(_series_label)
    detail["Client Type"] = detail["Base Project Code"].map(_client_type_label)

    detail = detail.rename(
        columns={
            "section": "Section",
            "span": "Span",
            "span_key": "Span Key",
            "gang_name": "Gang Name",
            "from_ap": "From AP",
            "to_ap": "To AP",
            "lag_inference_mode": "Lag Inference Mode",
            "lag_fallback_used": "Fallback Used",
            "location_parse_status": "Location Parse Status",
            "location_parse_issue": "Location Parse Issue",
            "required_location_count": "Required Location Count",
            "matched_location_count": "Matched Location Count",
            "unmatched_location_count": "Unmatched Location Count",
            "required_locations": "Required Locations",
            "matched_locations": "Matched Locations",
            "unmatched_locations": "Unmatched Locations",
            "location_nos_raw": "Location Nos Raw",
        }
    )
    detail = detail[detail_empty.columns].copy()

    summary_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    overall = detail.copy()
    summary_rows.append(_aggregate_es_row(overall, scope_type="OVERALL", scope_name="All Projects", method_split="All"))
    coverage_rows.append(_coverage_es_row(overall, scope_type="OVERALL", scope_name="All Projects", method_split="All"))
    for split_name in ("TSE", "Others"):
        split_scope = detail[detail["Method Split"] == split_name].copy()
        summary_rows.append(_aggregate_es_row(split_scope, scope_type="OVERALL", scope_name="All Projects", method_split=split_name))
        coverage_rows.append(_coverage_es_row(split_scope, scope_type="OVERALL", scope_name="All Projects", method_split=split_name))

    for project_name, project_scope in detail.groupby("Project", dropna=False):
        project_text = _safe_text(project_name) or "Unknown"
        summary_rows.append(_aggregate_es_row(project_scope, scope_type="PROJECT", scope_name=project_text, method_split="All"))
        coverage_rows.append(_coverage_es_row(project_scope, scope_type="PROJECT", scope_name=project_text, method_split="All"))
        for split_name in ("TSE", "Others"):
            split_scope = project_scope[project_scope["Method Split"] == split_name].copy()
            coverage_rows.append(_coverage_es_row(split_scope, scope_type="PROJECT_METHOD", scope_name=project_text, method_split=split_name))

    for series_name in _ordered_group_values(detail.get("Series", pd.Series(dtype="object")), ("4xx", "5xx", "6xx", "Other")):
        series_scope = detail[detail["Series"] == series_name].copy()
        summary_rows.append(_aggregate_es_row(series_scope, scope_type="SERIES", scope_name=series_name, method_split="All"))
        coverage_rows.append(_coverage_es_row(series_scope, scope_type="SERIES", scope_name=series_name, method_split="All"))

    for client_name in _ordered_group_values(detail.get("Client Type", pd.Series(dtype="object")), ("TA", "TB", "Other")):
        client_scope = detail[detail["Client Type"] == client_name].copy()
        summary_rows.append(_aggregate_es_row(client_scope, scope_type="CLIENT_TYPE", scope_name=client_name, method_split="All"))
        coverage_rows.append(_coverage_es_row(client_scope, scope_type="CLIENT_TYPE", scope_name=client_name, method_split="All"))

    return {
        "ES Delay Summary": pd.DataFrame(summary_rows, columns=summary_empty.columns),
        "ES Delay Coverage": pd.DataFrame(coverage_rows, columns=coverage_empty.columns),
        "ES Delay Anomalies": _build_es_anomaly_table(detail),
        "ES Delay Detail": detail,
    }


def _build_pofs_tables(
    *,
    compiled_scope: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summary_empty = _empty_pofs_summary_table()
    coverage_empty = _empty_pofs_coverage_table()
    anomaly_empty = _empty_pofs_anomaly_table()
    detail_empty = _empty_pofs_detail_table()

    if compiled_scope is None or compiled_scope.empty:
        return {
            "POFS Delay Summary": summary_empty,
            "POFS Delay Coverage": coverage_empty,
            "POFS Delay Anomalies": anomaly_empty,
            "POFS Delay Detail": detail_empty,
        }

    detail = pd.DataFrame(
        {
            "Project": compiled_scope.get("project_name", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Project Code": compiled_scope.get("project_code", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Project Display": compiled_scope.get("project_display", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Project Scope Key": compiled_scope.get("project_scope_key", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Join Project Key": compiled_scope.get("project_key_norm", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Method Split": compiled_scope.get("method", pd.Series("", index=compiled_scope.index)).map(_method_split),
            "Section": compiled_scope.get("section_label", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Span": compiled_scope.get("span_label", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Span Key": compiled_scope.get("span_key", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "Gang Name": compiled_scope.get("gang_name", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "From AP": compiled_scope.get("from_ap", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "To AP": compiled_scope.get("to_ap", pd.Series("", index=compiled_scope.index)).fillna("").astype(str).str.strip(),
            "PO Completion Date": pd.to_datetime(compiled_scope.get("po_completion_date"), errors="coerce").dt.normalize(),
            "FS Starting Date": pd.to_datetime(compiled_scope.get("fs_starting_date"), errors="coerce").dt.normalize(),
        }
    )
    detail["Project"] = detail["Project"].where(
        detail["Project"].astype(bool),
        detail["Project Display"].where(detail["Project Display"].astype(bool), detail["Project Code"]),
    )
    detail["Has PO Completion Date"] = detail["PO Completion Date"].notna()
    detail["Has FS Starting Date"] = detail["FS Starting Date"].notna()
    can_compute = detail["Has PO Completion Date"] & detail["Has FS Starting Date"]
    detail["Gap Days"] = pd.NA
    detail.loc[can_compute, "Gap Days"] = (
        detail.loc[can_compute, "FS Starting Date"] - detail.loc[can_compute, "PO Completion Date"]
    ).dt.days.astype("Int64")
    detail["Gap Days"] = pd.to_numeric(detail["Gap Days"], errors="coerce")
    detail["Gap Days Non-Negative"] = detail["Gap Days"].where(detail["Gap Days"].ge(0))
    detail["Negative Gap Flag"] = detail["Gap Days"].lt(0).fillna(False)
    detail["Base Project Code"] = [
        _derive_base_project_code(
            project_code=row.get("Project Code", ""),
            project_display=row.get("Project Display", ""),
            project_name=row.get("Project", ""),
            project_scope_key=row.get("Project Scope Key", ""),
            join_project_key=row.get("Join Project Key", ""),
        )
        for _, row in detail.iterrows()
    ]
    detail["Series"] = detail["Base Project Code"].map(_series_label)
    detail["Client Type"] = detail["Base Project Code"].map(_client_type_label)
    detail = detail[detail_empty.columns].copy()

    summary_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    overall = detail.copy()
    summary_rows.append(_aggregate_pofs_row(overall, scope_type="OVERALL", scope_name="All Projects", method_split="All"))
    coverage_rows.append(_coverage_pofs_row(overall, scope_type="OVERALL", scope_name="All Projects", method_split="All"))
    for split_name in ("TSE", "Others"):
        split_scope = detail[detail["Method Split"] == split_name].copy()
        summary_rows.append(_aggregate_pofs_row(split_scope, scope_type="OVERALL", scope_name="All Projects", method_split=split_name))
        coverage_rows.append(_coverage_pofs_row(split_scope, scope_type="OVERALL", scope_name="All Projects", method_split=split_name))

    for project_name, project_scope in detail.groupby("Project", dropna=False):
        project_text = _safe_text(project_name) or "Unknown"
        summary_rows.append(_aggregate_pofs_row(project_scope, scope_type="PROJECT", scope_name=project_text, method_split="All"))
        coverage_rows.append(_coverage_pofs_row(project_scope, scope_type="PROJECT", scope_name=project_text, method_split="All"))
        for split_name in ("TSE", "Others"):
            split_scope = project_scope[project_scope["Method Split"] == split_name].copy()
            coverage_rows.append(_coverage_pofs_row(split_scope, scope_type="PROJECT_METHOD", scope_name=project_text, method_split=split_name))

    for series_name in _ordered_group_values(detail.get("Series", pd.Series(dtype="object")), ("4xx", "5xx", "6xx", "Other")):
        series_scope = detail[detail["Series"] == series_name].copy()
        summary_rows.append(_aggregate_pofs_row(series_scope, scope_type="SERIES", scope_name=series_name, method_split="All"))
        coverage_rows.append(_coverage_pofs_row(series_scope, scope_type="SERIES", scope_name=series_name, method_split="All"))

    for client_name in _ordered_group_values(detail.get("Client Type", pd.Series(dtype="object")), ("TA", "TB", "Other")):
        client_scope = detail[detail["Client Type"] == client_name].copy()
        summary_rows.append(_aggregate_pofs_row(client_scope, scope_type="CLIENT_TYPE", scope_name=client_name, method_split="All"))
        coverage_rows.append(_coverage_pofs_row(client_scope, scope_type="CLIENT_TYPE", scope_name=client_name, method_split="All"))

    return {
        "POFS Delay Summary": pd.DataFrame(summary_rows, columns=summary_empty.columns),
        "POFS Delay Coverage": pd.DataFrame(coverage_rows, columns=coverage_empty.columns),
        "POFS Delay Anomalies": _build_pofs_anomaly_table(detail),
        "POFS Delay Detail": detail,
    }


def _normalize_monitoring_project_code(value: object) -> str:
    code = extract_base_project_code(value)
    if code:
        return code
    text = _safe_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).upper()


def _normalize_monitoring_activity_text(value: object) -> str:
    return _safe_text(value).lower()


def _as_monitoring_numeric(series: pd.Series | object) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series([series]), errors="coerce")


def _sum_number_strict(series: pd.Series) -> float | None:
    numeric = _as_monitoring_numeric(series)
    if numeric.empty or not numeric.notna().any():
        return None
    if numeric.isna().any():
        return None
    return float(numeric.sum())


def _sum_number(series: pd.Series) -> float | None:
    numeric = _as_monitoring_numeric(series).dropna()
    if numeric.empty:
        return None
    return float(numeric.sum())


def _first_valid_number(series: pd.Series) -> float | None:
    numeric = _as_monitoring_numeric(series).dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _build_stringing_monitoring_tables(
    *,
    status_activity_fact: pd.DataFrame | None,
    manpower_fact: pd.DataFrame | None,
    stringing_compiled_raw: pd.DataFrame | None,
    stretch_summary: pd.DataFrame | None,
    stretch_manpower_audit: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        "project_code",
        "row_type",
        "gang_name",
        "report_month",
        "project_total_scope_km",
        "monthly_plan_km",
        "fs_achieved_month_km",
        "%_achieved",
        "wip_km",
        "rough_sag_completed_km",
        "rough_sag_balance_km",
        "final_sag_completed_km",
        "final_sag_balance_km",
        "stretch_readiness_pct",
        "existing_manpower",
        "fitter_count",
    ]
    audit_columns = [
        "project_code",
        "status_activity_available",
        "status_final_sag_available",
        "status_paying_out_available",
        "stretch_readiness_available",
        "manpower_available",
        "fitter_available",
        "location_nos_available",
        "status_activity_fill_pct",
        "status_final_sag_fill_pct",
        "status_paying_out_fill_pct",
        "stretch_readiness_fill_pct",
        "manpower_fill_pct",
        "fitter_fill_pct",
        "location_nos_fill_pct",
        "missing_data_tags",
    ]

    status = status_activity_fact.copy() if isinstance(status_activity_fact, pd.DataFrame) else pd.DataFrame()
    manpower = manpower_fact.copy() if isinstance(manpower_fact, pd.DataFrame) else pd.DataFrame()
    compiled = stringing_compiled_raw.copy() if isinstance(stringing_compiled_raw, pd.DataFrame) else pd.DataFrame()
    stretch = stretch_summary.copy() if isinstance(stretch_summary, pd.DataFrame) else pd.DataFrame()
    stretch_audit = stretch_manpower_audit.copy() if isinstance(stretch_manpower_audit, pd.DataFrame) else pd.DataFrame()

    if not status.empty:
        status["project_code"] = status.get(
            "project_code",
            pd.Series("", index=status.index, dtype="object"),
        ).map(_normalize_monitoring_project_code)
        status = status[status["project_code"].astype(bool)].copy()
        status["month"] = pd.to_datetime(status.get("month"), errors="coerce").dt.to_period("M").dt.to_timestamp()
        status["report_date"] = pd.to_datetime(status.get("report_date"), errors="coerce")
        status["activity_norm_text"] = status.get(
            "activity_norm",
            pd.Series("", index=status.index, dtype="object"),
        ).map(_normalize_monitoring_activity_text)
        status["activity_raw_text"] = status.get(
            "activity_raw",
            pd.Series("", index=status.index, dtype="object"),
        ).map(_normalize_monitoring_activity_text)
        status["activity_group_text"] = status.get(
            "activity_group",
            pd.Series("", index=status.index, dtype="object"),
        ).map(_normalize_monitoring_activity_text)

    if not manpower.empty:
        manpower["project_code"] = manpower.get(
            "project_code",
            pd.Series("", index=manpower.index, dtype="object"),
        ).map(_normalize_monitoring_project_code)
        manpower = manpower[manpower["project_code"].astype(bool)].copy()
        manpower["date"] = pd.to_datetime(manpower.get("date"), errors="coerce")
        manpower["month"] = pd.to_datetime(manpower.get("month"), errors="coerce").dt.to_period("M").dt.to_timestamp()
        manpower["gang_name"] = manpower.get(
            "gang_name",
            pd.Series("", index=manpower.index, dtype="object"),
        ).fillna("").astype(str).str.strip()
        manpower["manpower_gang_strength"] = _as_monitoring_numeric(
            manpower.get("manpower_gang_strength", pd.Series(dtype="object"))
        )
        manpower["manpower_fitters"] = _as_monitoring_numeric(
            manpower.get("manpower_fitters", pd.Series(dtype="object"))
        )
        manpower["daily_km"] = _as_monitoring_numeric(manpower.get("daily_km", pd.Series(dtype="object")))

    if not compiled.empty:
        compiled["project_code"] = compiled.get(
            "project_code",
            pd.Series("", index=compiled.index, dtype="object"),
        ).map(_normalize_monitoring_project_code)
        compiled = compiled[compiled["project_code"].astype(bool)].copy()
        compiled["gang_name"] = compiled.get(
            "gang_name",
            pd.Series("", index=compiled.index, dtype="object"),
        ).fillna("").astype(str).str.strip()
        location_series = compiled.get("location nos", pd.Series("", index=compiled.index)).fillna("").astype(str).str.strip()
        from_series = compiled.get("from_ap", pd.Series("", index=compiled.index)).fillna("").astype(str).str.strip()
        to_series = compiled.get("to_ap", pd.Series("", index=compiled.index)).fillna("").astype(str).str.strip()
        compiled["__is_span_row"] = location_series.astype(bool) | from_series.astype(bool) | to_series.astype(bool)
        compiled["__has_location_nos"] = location_series.astype(bool)

    if not stretch.empty:
        stretch["project_code"] = stretch.get(
            "project_code",
            pd.Series("", index=stretch.index, dtype="object"),
        ).map(_normalize_monitoring_project_code)
        stretch = stretch[stretch["project_code"].astype(bool)].copy()
        stretch["ready_km"] = _as_monitoring_numeric(stretch.get("ready_km", pd.Series(dtype="object")))
        stretch["total_km"] = _as_monitoring_numeric(stretch.get("total_km", pd.Series(dtype="object")))
        stretch["readiness_pct"] = _as_monitoring_numeric(stretch.get("readiness_pct", pd.Series(dtype="object")))

    if not stretch_audit.empty:
        stretch_audit["project_code"] = stretch_audit.get(
            "project_code",
            pd.Series("", index=stretch_audit.index, dtype="object"),
        ).map(_normalize_monitoring_project_code)
        stretch_audit = stretch_audit[stretch_audit["project_code"].astype(bool)].copy()
        stretch_audit["status_text"] = stretch_audit.get(
            "status",
            pd.Series("", index=stretch_audit.index, dtype="object"),
        ).map(_normalize_monitoring_activity_text)

    project_codes: set[str] = set()
    for frame in (status, manpower, compiled, stretch, stretch_audit):
        if frame.empty or "project_code" not in frame.columns:
            continue
        project_codes.update(frame["project_code"].fillna("").astype(str).str.strip().tolist())
    project_codes = {code for code in project_codes if code}

    if not project_codes:
        return pd.DataFrame(columns=numeric_columns), pd.DataFrame(columns=audit_columns)

    numeric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for project_code in sorted(project_codes):
        status_project = status[status["project_code"] == project_code].copy() if not status.empty else pd.DataFrame()
        manpower_project = manpower[manpower["project_code"] == project_code].copy() if not manpower.empty else pd.DataFrame()
        compiled_project = compiled[compiled["project_code"] == project_code].copy() if not compiled.empty else pd.DataFrame()
        stretch_project = stretch[stretch["project_code"] == project_code].copy() if not stretch.empty else pd.DataFrame()
        stretch_audit_project = (
            stretch_audit[stretch_audit["project_code"] == project_code].copy() if not stretch_audit.empty else pd.DataFrame()
        )

        latest_month = status_project["month"].dropna().max() if not status_project.empty else pd.NaT
        if pd.isna(latest_month) and not manpower_project.empty:
            latest_month = manpower_project["month"].dropna().max()

        if not status_project.empty and pd.notna(latest_month):
            status_scope = status_project[status_project["month"] == latest_month].copy()
        else:
            status_scope = status_project.copy()

        final_mask = (
            status_scope["activity_raw_text"].str.contains("final sag", regex=False)
            | status_scope["activity_norm_text"].str.contains("final_sag", regex=False)
        ) if not status_scope.empty else pd.Series(dtype="bool")
        stringing_mask = (
            status_scope["activity_group_text"].str.contains("stringing", regex=False)
            | status_scope["activity_norm_text"].str.contains("stringing", regex=False)
        ) if not status_scope.empty else pd.Series(dtype="bool")
        paying_mask = (
            status_scope["activity_raw_text"].str.contains("paying out", regex=False)
            | status_scope["activity_norm_text"].str.contains("paying_out", regex=False)
        ) if not status_scope.empty else pd.Series(dtype="bool")

        final_rows = status_scope[final_mask].copy() if not status_scope.empty else pd.DataFrame()
        string_rows = status_scope[stringing_mask].copy() if not status_scope.empty else pd.DataFrame()
        paying_rows = status_scope[paying_mask].copy() if not status_scope.empty else pd.DataFrame()
        final_or_string_rows = final_rows if not final_rows.empty else string_rows

        project_total_scope_km = _sum_number_strict(final_or_string_rows.get("quantity_primary", pd.Series(dtype="object")))
        monthly_plan_km = _sum_number_strict(final_or_string_rows.get("plan_for_month", pd.Series(dtype="object")))
        fs_achieved_month_km = _sum_number_strict(final_or_string_rows.get("progress_for_month", pd.Series(dtype="object")))
        final_sag_completed_km = _sum_number_strict(final_or_string_rows.get("cumulative_progress", pd.Series(dtype="object")))
        final_sag_balance_km = _sum_number_strict(final_or_string_rows.get("balance_progress", pd.Series(dtype="object")))
        rough_sag_completed_km = _sum_number_strict(paying_rows.get("cumulative_progress", pd.Series(dtype="object")))
        rough_sag_balance_km = _sum_number_strict(paying_rows.get("balance_progress", pd.Series(dtype="object")))

        pct_achieved = None
        if monthly_plan_km is not None and fs_achieved_month_km is not None and monthly_plan_km > 0:
            pct_achieved = float(fs_achieved_month_km / monthly_plan_km)

        wip_km = None
        if rough_sag_completed_km is not None and final_sag_completed_km is not None:
            delta = rough_sag_completed_km - final_sag_completed_km
            if delta >= 0:
                wip_km = float(delta)

        stretch_readiness_pct = None
        if not stretch_project.empty:
            total_series = _as_monitoring_numeric(stretch_project.get("total_km", pd.Series(dtype="object")))
            ready_series = _as_monitoring_numeric(stretch_project.get("ready_km", pd.Series(dtype="object")))
            if total_series.notna().any() and not total_series.isna().any() and ready_series.notna().any() and not ready_series.isna().any():
                total_km = float(total_series.sum())
                if total_km > 0:
                    stretch_readiness_pct = float((ready_series.sum() / total_km) * 100.0)
            if stretch_readiness_pct is None:
                readiness_series = _as_monitoring_numeric(stretch_project.get("readiness_pct", pd.Series(dtype="object")))
                if readiness_series.notna().any() and not readiness_series.isna().any():
                    stretch_readiness_pct = _first_valid_number(readiness_series)

        manpower_latest_scope = pd.DataFrame()
        if not manpower_project.empty:
            latest_date = manpower_project["date"].dropna().max()
            if pd.notna(latest_date):
                manpower_latest_scope = manpower_project[manpower_project["date"] == latest_date].copy()
            else:
                manpower_latest_scope = manpower_project.copy()

        existing_manpower = None
        fitter_count = None
        if not manpower_latest_scope.empty:
            grouped = manpower_latest_scope.copy()
            grouped["__gang_key"] = grouped["gang_name"].astype(str).str.strip()
            grouped["__gang_key"] = grouped["__gang_key"].where(grouped["__gang_key"].astype(bool), "__UNKNOWN__")
            manpower_by_gang = grouped.groupby("__gang_key", dropna=False)["manpower_gang_strength"].max()
            fitters_by_gang = grouped.groupby("__gang_key", dropna=False)["manpower_fitters"].max()
            if manpower_by_gang.notna().any() and not manpower_by_gang.isna().any():
                existing_manpower = float(manpower_by_gang.sum())
            if fitters_by_gang.notna().any() and not fitters_by_gang.isna().any():
                fitter_count = float(fitters_by_gang.sum())

        report_month_text = pd.Timestamp(latest_month).strftime("%Y-%m") if pd.notna(latest_month) else ""

        numeric_rows.append(
            {
                "project_code": project_code,
                "row_type": "project",
                "gang_name": "",
                "report_month": report_month_text,
                "project_total_scope_km": project_total_scope_km,
                "monthly_plan_km": monthly_plan_km,
                "fs_achieved_month_km": fs_achieved_month_km,
                "%_achieved": pct_achieved,
                "wip_km": wip_km,
                "rough_sag_completed_km": rough_sag_completed_km,
                "rough_sag_balance_km": rough_sag_balance_km,
                "final_sag_completed_km": final_sag_completed_km,
                "final_sag_balance_km": final_sag_balance_km,
                "stretch_readiness_pct": stretch_readiness_pct,
                "existing_manpower": existing_manpower,
                "fitter_count": fitter_count,
            }
        )

        gang_names = []
        if not compiled_project.empty:
            gang_names = (
                compiled_project["gang_name"]
                .fillna("")
                .astype(str)
                .str.strip()
                .loc[lambda x: x.astype(bool)]
                .drop_duplicates()
                .tolist()
            )
        for gang_name in gang_names:
            gang_scope = (
                manpower_project[manpower_project["gang_name"] == gang_name].copy() if not manpower_project.empty else pd.DataFrame()
            )
            gang_existing_manpower = None
            gang_fitter_count = None
            gang_fs_achieved = None
            if not gang_scope.empty:
                latest_gang_date = gang_scope["date"].dropna().max()
                if pd.notna(latest_gang_date):
                    gang_day = gang_scope[gang_scope["date"] == latest_gang_date].copy()
                else:
                    gang_day = gang_scope.copy()
                gang_existing_manpower = _sum_number(gang_day["manpower_gang_strength"])
                gang_fitter_count = _sum_number(gang_day["manpower_fitters"])
                if pd.notna(latest_month):
                    gang_month = gang_scope[gang_scope["month"] == latest_month].copy()
                else:
                    gang_month = gang_scope.copy()
                gang_fs_achieved = _sum_number_strict(gang_month["daily_km"])

            numeric_rows.append(
                {
                    "project_code": project_code,
                    "row_type": "gang",
                    "gang_name": gang_name,
                    "report_month": report_month_text,
                    "project_total_scope_km": pd.NA,
                    "monthly_plan_km": pd.NA,
                    "fs_achieved_month_km": gang_fs_achieved,
                    "%_achieved": pd.NA,
                    "wip_km": pd.NA,
                    "rough_sag_completed_km": pd.NA,
                    "rough_sag_balance_km": pd.NA,
                    "final_sag_completed_km": pd.NA,
                    "final_sag_balance_km": pd.NA,
                    "stretch_readiness_pct": pd.NA,
                    "existing_manpower": gang_existing_manpower,
                    "fitter_count": gang_fitter_count,
                }
            )

        status_activity_available = not status_scope.empty
        status_final_sag_available = not final_or_string_rows.empty
        status_paying_out_available = not paying_rows.empty
        stretch_readiness_available = not stretch_project.empty and stretch_readiness_pct is not None

        manpower_total_rows = int(len(manpower_project.index))
        manpower_available_count = int(manpower_project["manpower_gang_strength"].notna().sum()) if manpower_total_rows else 0
        fitter_available_count = int(manpower_project["manpower_fitters"].notna().sum()) if manpower_total_rows else 0
        manpower_fill_pct = (manpower_available_count / manpower_total_rows * 100.0) if manpower_total_rows else 0.0
        fitter_fill_pct = (fitter_available_count / manpower_total_rows * 100.0) if manpower_total_rows else 0.0
        manpower_available = manpower_available_count > 0
        fitter_available = fitter_available_count > 0

        location_total = 0
        location_filled = 0
        if not compiled_project.empty:
            span_scope = compiled_project[compiled_project["__is_span_row"]].copy()
            location_total = int(len(span_scope.index))
            location_filled = int(span_scope["__has_location_nos"].sum())
        location_fill_pct = (location_filled / location_total * 100.0) if location_total else 0.0
        location_nos_available = location_filled > 0

        stretch_fill_pct = 0.0
        if not stretch_project.empty:
            readiness_series = _as_monitoring_numeric(stretch_project.get("readiness_pct", pd.Series(dtype="object")))
            if stretch_readiness_pct is not None:
                stretch_fill_pct = 100.0
            elif len(readiness_series.index):
                stretch_fill_pct = float((readiness_series.notna().sum() / len(readiness_series.index)) * 100.0)

        status_activity_fill = 0.0
        if not status_scope.empty:
            status_activity_fill = float(
                (
                    status_scope[["quantity_primary", "plan_for_month", "progress_for_month", "cumulative_progress", "balance_progress"]]
                    .notna()
                    .sum()
                    .sum()
                    / (len(status_scope.index) * 5)
                )
                * 100.0
            )

        status_final_fill = 0.0
        if not final_or_string_rows.empty:
            final_vals = final_or_string_rows[
                ["quantity_primary", "plan_for_month", "progress_for_month", "cumulative_progress", "balance_progress"]
            ]
            status_final_fill = float((final_vals.notna().sum().sum() / (len(final_vals.index) * len(final_vals.columns))) * 100.0)

        status_paying_fill = 0.0
        if not paying_rows.empty:
            paying_vals = paying_rows[["cumulative_progress", "balance_progress"]]
            status_paying_fill = float((paying_vals.notna().sum().sum() / (len(paying_vals.index) * len(paying_vals.columns))) * 100.0)

        missing_tags: list[str] = []
        if not status_activity_available:
            missing_tags.append("MISSING_STATUS_ACTIVITY")
        if not status_final_sag_available:
            missing_tags.append("MISSING_STATUS_FINAL_SAG")
        if not status_paying_out_available:
            missing_tags.append("MISSING_STATUS_PAYING_OUT")
        if not stretch_readiness_available:
            missing_tags.append("MISSING_STRETCH_READINESS")
        if not manpower_available:
            missing_tags.append("MISSING_MANPOWER")
        if not fitter_available:
            missing_tags.append("MISSING_FITTER")
        if not location_nos_available:
            missing_tags.append("MISSING_LOCATION_NOS")
        if not stretch_audit_project.empty and stretch_audit_project["status_text"].str.contains("absent", regex=False).any():
            missing_tags.append("MISSING_STRETCH_MANPOWER_SIGNAL")

        audit_rows.append(
            {
                "project_code": project_code,
                "status_activity_available": "Yes" if status_activity_available else "No",
                "status_final_sag_available": "Yes" if status_final_sag_available else "No",
                "status_paying_out_available": "Yes" if status_paying_out_available else "No",
                "stretch_readiness_available": "Yes" if stretch_readiness_available else "No",
                "manpower_available": "Yes" if manpower_available else "No",
                "fitter_available": "Yes" if fitter_available else "No",
                "location_nos_available": "Yes" if location_nos_available else "No",
                "status_activity_fill_pct": round(status_activity_fill, 1),
                "status_final_sag_fill_pct": round(status_final_fill, 1),
                "status_paying_out_fill_pct": round(status_paying_fill, 1),
                "stretch_readiness_fill_pct": round(stretch_fill_pct, 1),
                "manpower_fill_pct": round(manpower_fill_pct, 1),
                "fitter_fill_pct": round(fitter_fill_pct, 1),
                "location_nos_fill_pct": round(location_fill_pct, 1),
                "missing_data_tags": ";".join(dict.fromkeys(missing_tags)),
            }
        )

    numeric_df = pd.DataFrame(numeric_rows, columns=numeric_columns)
    audit_df = pd.DataFrame(audit_rows, columns=audit_columns)
    return numeric_df, audit_df


def build_erection_stringing_delay_tables(
    *,
    stringing_compiled_raw: pd.DataFrame,
    erection_daily: pd.DataFrame,
    stringing_status_activity_fact: pd.DataFrame | None = None,
    stringing_manpower_fact: pd.DataFrame | None = None,
    stretch_readiness_summary: pd.DataFrame | None = None,
    stretch_readiness_manpower_audit: pd.DataFrame | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    method_scope: str = "all",
) -> dict[str, pd.DataFrame]:
    """Build E->S and PO->FS delay analysis tables from compiled stringing and erection daily inputs."""
    es_empty = {
        "ES Delay Summary": _empty_es_summary_table(),
        "ES Delay Coverage": _empty_es_coverage_table(),
        "ES Delay Anomalies": _empty_es_anomaly_table(),
        "ES Delay Detail": _empty_es_detail_table(),
    }
    pofs_empty = {
        "POFS Delay Summary": _empty_pofs_summary_table(),
        "POFS Delay Coverage": _empty_pofs_coverage_table(),
        "POFS Delay Anomalies": _empty_pofs_anomaly_table(),
        "POFS Delay Detail": _empty_pofs_detail_table(),
    }
    monitoring_numeric, monitoring_audit = _build_stringing_monitoring_tables(
        status_activity_fact=stringing_status_activity_fact,
        manpower_fact=stringing_manpower_fact,
        stringing_compiled_raw=stringing_compiled_raw,
        stretch_summary=stretch_readiness_summary,
        stretch_manpower_audit=stretch_readiness_manpower_audit,
    )

    compiled = _normalize_stringing_compiled(stringing_compiled_raw)
    if compiled.empty:
        return {
            **es_empty,
            **pofs_empty,
            "Stringing Monitoring Numeric": monitoring_numeric,
            "Stringing Monitoring Audit": monitoring_audit,
        }

    mode = _safe_text(method_scope).lower() or "all"
    if mode not in {"all", "tse", "exclude_manual"}:
        mode = "all"
    compiled = _filter_method(compiled, mode)
    if compiled.empty:
        return {
            **es_empty,
            **pofs_empty,
            "Stringing Monitoring Numeric": monitoring_numeric,
            "Stringing Monitoring Audit": monitoring_audit,
        }

    compiled = compiled.reset_index(drop=True).copy()
    compiled["po_start_date"] = pd.to_datetime(compiled.get("po_start_date"), errors="coerce").dt.normalize()
    compiled["po_completion_date"] = pd.to_datetime(compiled.get("po_completion_date"), errors="coerce").dt.normalize()
    compiled["fs_starting_date"] = pd.to_datetime(compiled.get("fs_starting_date"), errors="coerce").dt.normalize()
    compiled["__analysis_row_id"] = pd.Series(range(len(compiled.index)), index=compiled.index, dtype="int64")
    compiled["project_key_norm"] = _preferred_project_key(
        compiled,
        scope_candidates=("project_scope_key", "Project Scope Key"),
    )

    start_ts = _coerce_date(start_date)
    end_ts = _coerce_date(end_date)

    compiled_es = compiled[compiled["po_start_date"].notna()].copy()
    if start_ts is not None:
        compiled_es = compiled_es[compiled_es["po_start_date"] >= start_ts].copy()
    if end_ts is not None:
        compiled_es = compiled_es[compiled_es["po_start_date"] <= end_ts].copy()

    compiled_pofs = compiled.copy()
    if start_ts is not None:
        compiled_pofs = compiled_pofs[compiled_pofs["po_completion_date"].notna()].copy()
        compiled_pofs = compiled_pofs[compiled_pofs["po_completion_date"] >= start_ts].copy()
    if end_ts is not None:
        compiled_pofs = compiled_pofs[compiled_pofs["po_completion_date"].notna()].copy()
        compiled_pofs = compiled_pofs[compiled_pofs["po_completion_date"] <= end_ts].copy()

    erection_work = erection_daily.copy() if isinstance(erection_daily, pd.DataFrame) else pd.DataFrame()
    if not erection_work.empty:
        erection_work["project_key_norm"] = _preferred_project_key(
            erection_work,
            scope_candidates=("Project Scope Key", "project_scope_key"),
        )

    es_tables = _build_es_tables(compiled_scope=compiled_es, erection_work=erection_work)
    pofs_tables = _build_pofs_tables(compiled_scope=compiled_pofs)
    return {
        **es_tables,
        **pofs_tables,
        "Stringing Monitoring Numeric": monitoring_numeric,
        "Stringing Monitoring Audit": monitoring_audit,
    }


def write_erection_stringing_delay_workbook(
    output_path: str | Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    """Write E->S and PO->FS delay analysis workbook."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = [
        "ES Delay Summary",
        "ES Delay Coverage",
        "ES Delay Anomalies",
        "ES Delay Detail",
        "POFS Delay Summary",
        "POFS Delay Coverage",
        "POFS Delay Anomalies",
        "POFS Delay Detail",
        "Stringing Monitoring Numeric",
        "Stringing Monitoring Audit",
    ]
    seen = set(ordered)
    for key in tables.keys():
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in ordered:
            if sheet not in tables:
                continue
            table = tables.get(sheet, pd.DataFrame())
            pd.DataFrame([[sheet]]).to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=0)
            table.to_excel(writer, sheet_name=sheet, index=False, startrow=1)
    return output
