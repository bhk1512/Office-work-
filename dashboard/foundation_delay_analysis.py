"""Foundation delay analysis helpers.

This module contains:
1) Legacy delay-trend table builder used by workbook export (compatibility).
2) Foundation Delay V2 builders for standalone, scope-aware analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .project_identity import build_project_rollup_identity, extract_base_project_code

MONSOON_MONTHS = {6, 7, 8, 9}
PHASE_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 20, "0-20"),
    (20, 40, "20-40"),
    (40, 60, "40-60"),
    (60, 80, "60-80"),
    (80, 10_000, "80-100+"),
)
PHASE_ORDER = {label: idx for idx, (_, _, label) in enumerate(PHASE_BANDS, start=1)}


@dataclass(frozen=True)
class MechanismConfig:
    pre_monsoon: tuple[int, ...] = (4, 5)
    monsoon: tuple[int, ...] = (6, 7, 8, 9)
    post_monsoon: tuple[int, ...] = (10, 11)
    post_monsoon_wide: tuple[int, ...] = (10, 11, 12)
    dry: tuple[int, ...] = (12, 1, 2, 3)
    min_foundation_count: int = 5


def _normalize_month_tuple(values: object, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return fallback
    normalized: list[int] = []
    seen: set[int] = set()
    for raw in values if isinstance(values, (list, tuple, set)) else [values]:
        numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            continue
        month = int(numeric)
        if month < 1 or month > 12 or month in seen:
            continue
        seen.add(month)
        normalized.append(month)
    return tuple(normalized) if normalized else fallback


def _resolve_mechanism_config(config: MechanismConfig | None) -> MechanismConfig:
    if config is None:
        return MechanismConfig()
    min_count_numeric = pd.to_numeric(pd.Series([getattr(config, "min_foundation_count", 5)]), errors="coerce").iloc[0]
    min_count = int(min_count_numeric) if pd.notna(min_count_numeric) else 5
    if min_count < 0:
        min_count = 0
    return MechanismConfig(
        pre_monsoon=_normalize_month_tuple(getattr(config, "pre_monsoon", (4, 5)), (4, 5)),
        monsoon=_normalize_month_tuple(getattr(config, "monsoon", (6, 7, 8, 9)), (6, 7, 8, 9)),
        post_monsoon=_normalize_month_tuple(getattr(config, "post_monsoon", (10, 11)), (10, 11)),
        post_monsoon_wide=_normalize_month_tuple(getattr(config, "post_monsoon_wide", (10, 11, 12)), (10, 11, 12)),
        dry=_normalize_month_tuple(getattr(config, "dry", (12, 1, 2, 3)), (12, 1, 2, 3)),
        min_foundation_count=min_count,
    )


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    return text


def _compact_project_key(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _pick_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _normalize_location_alias_text(value: object) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    # Conservative cleanup rules for known location notations only.
    text = re.sub(r"^N['`’]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+[A-Za-z](?:\s*,\s*[A-Za-z])+$", "", text)
    text = re.sub(r"\s+[A-Za-z]$", "", text)
    return text.strip()


def _normalize_erection_source_columns(source: pd.DataFrame) -> pd.DataFrame:
    if source is None or source.empty:
        return pd.DataFrame()
    work = source.copy()
    mappings: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("project_code", ("project_code", "Project Code")),
        ("project_display", ("project_display", "Project Display")),
        ("project_name", ("project_name", "Project Name", "project")),
        ("line_name", ("line_name", "Line Name")),
        ("location_no", ("location_no", "Location No.", "Location No", "location no")),
        ("start_date", ("start_date", "Start Date", "starting date", "Starting Date")),
        ("source_file", ("source_file", "Source File", "_source_file")),
        ("source_sheet", ("source_sheet", "Source Sheet", "sheet", "configured_sheet")),
    )
    for target, candidates in mappings:
        if target in work.columns:
            continue
        series = _pick_series(work, candidates)
        work[target] = series
    return work


def build_legacy_erection_source_from_raw(raw_source: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw erection rows and keep only records with both start and completion dates."""
    source = _normalize_erection_source_columns(raw_source)
    if source.empty:
        return pd.DataFrame()
    work = source.copy()
    work["start_date"] = _coerce_mixed_excel_dates_series(work.get("start_date", pd.Series(dtype="object")))
    completion_series = _pick_series(
        work,
        ("completion_date", "Complete Date", "complete date", "completion date"),
    )
    work["completion_date"] = _coerce_mixed_excel_dates_series(completion_series)
    work = work[work["start_date"].notna() & work["completion_date"].notna()].copy()
    return work


def build_v2_erection_source_from_raw(raw_source: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw erection rows and keep records with valid start dates."""
    source = _normalize_erection_source_columns(raw_source)
    if source.empty:
        return pd.DataFrame()
    work = source.copy()
    work["start_date"] = _coerce_mixed_excel_dates_series(work.get("start_date", pd.Series(dtype="object")))
    completion_series = _pick_series(
        work,
        ("completion_date", "Complete Date", "complete date", "completion date"),
    )
    work["completion_date"] = _coerce_mixed_excel_dates_series(completion_series)
    work = work[work["start_date"].notna()].copy()
    return work


def _coerce_mixed_excel_dates_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    excel_mask = parsed.isna() & numeric.notna() & numeric.between(20000, 80000)
    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime(
            numeric.loc[excel_mask],
            errors="coerce",
            unit="D",
            origin="1899-12-30",
        )
    return pd.to_datetime(parsed, errors="coerce").dt.normalize()


def _series_stats(delay_series: pd.Series) -> tuple[object, object, object]:
    values = pd.to_numeric(delay_series, errors="coerce").dropna()
    if values.empty:
        return pd.NA, pd.NA, pd.NA
    return (
        float(values.median()),
        float(values.mean()),
        float(values.quantile(0.9)),
    )


def _apply_project_rollup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    working = df.copy()
    project_code = working.get("project_code", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()
    project_display = working.get(
        "project_display",
        working.get("project_name", pd.Series("", index=working.index)),
    ).fillna("").astype(str).str.strip()
    project_name = working.get(
        "project_name",
        working.get("project", pd.Series("", index=working.index)),
    ).fillna("").astype(str).str.strip()
    identities = [
        build_project_rollup_identity(code, display, name)
        for code, display, name in zip(project_code, project_display, project_name)
    ]
    working["project_rollup_display"] = pd.Series(
        [identity.get("project_rollup_display", "") for identity in identities],
        index=working.index,
        dtype="object",
    ).fillna("").astype(str).str.strip()
    working["project_rollup_display"] = working["project_rollup_display"].where(
        working["project_rollup_display"].astype(bool),
        project_display.where(project_display.astype(bool), project_name),
    )
    working["project_rollup_key"] = pd.Series(
        [identity.get("project_rollup_key", "") for identity in identities],
        index=working.index,
        dtype="object",
    ).fillna("").astype(str).str.strip()
    fallback_key = working["project_rollup_display"].map(_compact_project_key)
    working["project_rollup_key"] = working["project_rollup_key"].where(
        working["project_rollup_key"].astype(bool),
        fallback_key,
    )
    working["project_base_code"] = pd.Series(
        [identity.get("project_base_code", "") for identity in identities],
        index=working.index,
        dtype="object",
    ).fillna("").astype(str).str.strip()
    fallback_code = working["project_rollup_display"].map(extract_base_project_code)
    working["project_base_code"] = working["project_base_code"].where(
        working["project_base_code"].astype(bool),
        fallback_code,
    )
    return working


def _build_erection_start_events(source_daily: pd.DataFrame) -> pd.DataFrame:
    if source_daily is None or source_daily.empty:
        return pd.DataFrame()
    source = _normalize_erection_source_columns(source_daily)
    if source.empty:
        return pd.DataFrame()
    work = _apply_project_rollup_columns(source)
    work["start_date"] = _coerce_mixed_excel_dates_series(work.get("start_date", pd.Series(dtype="object")))
    work = work[work["start_date"].notna()].copy()
    if work.empty:
        return work
    work = work[work["start_date"].dt.year >= 1980].copy()
    work["location_no"] = work.get("location_no", "").fillna("").astype(str).str.strip()
    work["line_name"] = work.get("line_name", "").fillna("").astype(str).str.strip()
    work["source_file"] = work.get("source_file", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    work["source_sheet"] = work.get("source_sheet", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    work["line_name_norm"] = work["line_name"].map(_compact_project_key)
    work["location_no_norm"] = work["location_no"].map(_compact_project_key)
    work["location_no_alias_norm"] = work["location_no"].map(_normalize_location_alias_text).map(_compact_project_key)
    work = work[
        work["project_rollup_key"].astype(bool)
        & work["location_no_norm"].astype(bool)
    ].copy()
    if work.empty:
        return work
    return (
        work.sort_values(["project_rollup_key", "start_date", "line_name", "location_no"])
        .groupby(["project_rollup_key", "line_name_norm", "location_no_norm"], as_index=False)
        .agg(
            erection_start=("start_date", "min"),
            erection_line=("line_name", "first"),
            erection_source_file=("source_file", "first"),
            erection_source_sheet=("source_sheet", "first"),
        )
    )


def _build_foundation_detail_events(foundation_completions: pd.DataFrame) -> pd.DataFrame:
    if foundation_completions is None or foundation_completions.empty:
        return pd.DataFrame()
    foundation = _apply_project_rollup_columns(foundation_completions)
    foundation["event_date"] = _coerce_mixed_excel_dates_series(foundation.get("event_date", pd.Series(dtype="object")))
    foundation["source_type"] = foundation.get("source_type", "").fillna("").astype(str).str.strip().str.lower()
    foundation = foundation[(foundation["source_type"] == "detail") & foundation["event_date"].notna()].copy()
    if foundation.empty:
        return foundation
    foundation = foundation[foundation["event_date"].dt.year >= 1980].copy()
    foundation["line_name"] = foundation.get("line_name", "").fillna("").astype(str).str.strip()
    foundation["location_no"] = foundation.get("location_no", "").fillna("").astype(str).str.strip()
    foundation["source_file"] = foundation.get("source_file", pd.Series("", index=foundation.index)).fillna("").astype(str).str.strip()
    foundation["source_sheet"] = foundation.get(
        "source_sheet",
        foundation.get("configured_sheet", pd.Series("", index=foundation.index)),
    ).fillna("").astype(str).str.strip()
    foundation["line_name_norm"] = foundation["line_name"].map(_compact_project_key)
    foundation["location_no_norm"] = foundation["location_no"].map(_compact_project_key)
    foundation["location_no_alias_norm"] = foundation["location_no"].map(_normalize_location_alias_text).map(_compact_project_key)
    foundation = foundation[
        foundation["project_rollup_key"].astype(bool)
        & foundation["location_no_norm"].astype(bool)
    ].copy()
    if foundation.empty:
        return foundation
    return (
        foundation.sort_values(["project_rollup_key", "event_date", "line_name", "location_no"])
        .groupby(
            ["project_rollup_key", "project_rollup_display", "project_base_code", "line_name_norm", "location_no_norm"],
            as_index=False,
        )
        .agg(
            foundation_complete=("event_date", "min"),
            foundation_line=("line_name", "first"),
            location_no=("location_no", "first"),
            location_no_alias_norm=("location_no_alias_norm", "first"),
            foundation_source_file=("source_file", "first"),
            foundation_source_sheet=("source_sheet", "first"),
        )
    )


def _build_merged_delay_fact(source_daily: pd.DataFrame, foundation_completions: pd.DataFrame) -> pd.DataFrame:
    foundation_loc = _build_foundation_detail_events(foundation_completions)
    if foundation_loc.empty:
        return pd.DataFrame()
    erection_loc = _build_erection_start_events(source_daily)
    if erection_loc.empty:
        erection_loc = pd.DataFrame(
            columns=[
                "project_rollup_key",
                "line_name_norm",
                "location_no_norm",
                "erection_start",
                "erection_line",
                "erection_source_file",
                "erection_source_sheet",
            ]
        )
    merged = foundation_loc.merge(
        erection_loc.rename(
            columns={
                "erection_start": "erection_start_exact",
                "erection_line": "erection_line_exact",
                "erection_source_file": "erection_source_file_exact",
                "erection_source_sheet": "erection_source_sheet_exact",
            }
        ),
        on=["project_rollup_key", "line_name_norm", "location_no_norm"],
        how="left",
    )

    alias_from_foundation = pd.DataFrame()
    if not erection_loc.empty:
        alias_from_foundation = erection_loc.rename(
            columns={
                "location_no_norm": "location_no_alias_norm",
                "erection_start": "erection_start_alias",
                "erection_line": "erection_line_alias",
                "erection_source_file": "erection_source_file_alias",
                "erection_source_sheet": "erection_source_sheet_alias",
            }
        )[
            [
                "project_rollup_key",
                "line_name_norm",
                "location_no_alias_norm",
                "erection_start_alias",
                "erection_line_alias",
                "erection_source_file_alias",
                "erection_source_sheet_alias",
            ]
        ]
    if alias_from_foundation.empty:
        alias_from_foundation = pd.DataFrame(
            columns=[
                "project_rollup_key",
                "line_name_norm",
                "location_no_alias_norm",
                "erection_start_alias",
                "erection_line_alias",
                "erection_source_file_alias",
                "erection_source_sheet_alias",
            ]
        )
    merged = merged.merge(
        alias_from_foundation,
        on=["project_rollup_key", "line_name_norm", "location_no_alias_norm"],
        how="left",
    )

    merged["erection_start"] = merged["erection_start_exact"].where(
        merged["erection_start_exact"].notna(),
        merged["erection_start_alias"],
    )
    merged["erection_line"] = merged["erection_line_exact"].where(
        merged["erection_line_exact"].astype(bool),
        merged["erection_line_alias"],
    )
    merged["erection_source_file"] = merged["erection_source_file_exact"].where(
        merged["erection_source_file_exact"].astype(bool),
        merged["erection_source_file_alias"],
    )
    merged["erection_source_sheet"] = merged["erection_source_sheet_exact"].where(
        merged["erection_source_sheet_exact"].astype(bool),
        merged["erection_source_sheet_alias"],
    )
    merged["match_basis"] = ""
    exact_mask = merged["erection_start_exact"].notna()
    alias_mask = (~exact_mask) & merged["erection_start_alias"].notna()
    merged.loc[exact_mask, "match_basis"] = "exact"
    merged.loc[alias_mask, "match_basis"] = "alias"

    if not erection_loc.empty:
        ere_loc_counts = (
            erection_loc.sort_values(["project_rollup_key", "location_no_norm", "erection_start", "line_name_norm"])
            .groupby(["project_rollup_key", "location_no_norm"], as_index=False)
            .agg(
                ere_start_fallback=("erection_start", "min"),
                ere_line_count=("line_name_norm", "nunique"),
                ere_source_file_fallback=("erection_source_file", "first"),
                ere_source_sheet_fallback=("erection_source_sheet", "first"),
            )
        )
    else:
        ere_loc_counts = pd.DataFrame(
            columns=[
                "project_rollup_key",
                "location_no_norm",
                "ere_start_fallback",
                "ere_line_count",
                "ere_source_file_fallback",
                "ere_source_sheet_fallback",
            ]
        )
    fdn_loc_counts = (
        foundation_loc.groupby(["project_rollup_key", "location_no_norm"], as_index=False)
        .agg(fdn_line_count=("line_name_norm", "nunique"))
    )
    fallback = fdn_loc_counts.merge(
        ere_loc_counts,
        on=["project_rollup_key", "location_no_norm"],
        how="left",
    )
    fallback = fallback[(fallback["fdn_line_count"] == 1) & (fallback["ere_line_count"] == 1)].copy()
    fallback = fallback[
        [
            "project_rollup_key",
            "location_no_norm",
            "ere_start_fallback",
            "ere_source_file_fallback",
            "ere_source_sheet_fallback",
        ]
    ]
    merged = merged.merge(
        fallback,
        on=["project_rollup_key", "location_no_norm"],
        how="left",
    )
    merged["erection_start_final"] = merged["erection_start"].where(
        merged["erection_start"].notna(),
        merged["ere_start_fallback"],
    )
    merged["erection_source_file"] = merged["erection_source_file"].where(
        merged["erection_source_file"].astype(bool),
        merged["ere_source_file_fallback"],
    )
    merged["erection_source_sheet"] = merged["erection_source_sheet"].where(
        merged["erection_source_sheet"].astype(bool),
        merged["ere_source_sheet_fallback"],
    )
    fallback_mask = merged["match_basis"].eq("") & merged["ere_start_fallback"].notna()
    merged.loc[fallback_mask, "match_basis"] = "fallback_single_line"
    merged["matched"] = merged["erection_start_final"].notna()
    merged["delay_days"] = (
        pd.to_datetime(merged["erection_start_final"], errors="coerce")
        - pd.to_datetime(merged["foundation_complete"], errors="coerce")
    ).dt.days
    merged["negative_delay"] = merged["matched"] & merged["delay_days"].lt(0)
    merged["delay_for_stats"] = merged["delay_days"].where(merged["matched"] & ~merged["negative_delay"])
    foundation_month_ts = pd.to_datetime(merged["foundation_complete"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    merged["foundation_month"] = foundation_month_ts.dt.strftime("%Y-%m").where(foundation_month_ts.notna(), "")
    merged["foundation_month_num"] = foundation_month_ts.dt.month
    erection_month_ts = pd.to_datetime(merged["erection_start_final"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    merged["erection_start_month"] = erection_month_ts.dt.strftime("%Y-%m").where(erection_month_ts.notna(), "")
    merged["erection_start_month_num"] = erection_month_ts.dt.month
    return merged


def _phase_bucket_legacy(total: int) -> list[int]:
    if total <= 0:
        return []
    return [((idx * 5) // total) + 1 for idx in range(total)]


def build_foundation_delay_trend_tables_legacy(
    source_daily: pd.DataFrame,
    foundation_completions: pd.DataFrame,
    foundation_coverage: pd.DataFrame,
    foundation_diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy builder used by existing productivity workbook.

    This keeps old columns and semantics intact.
    """
    phase_columns = [
        "Project",
        "Phase",
        "Foundation Bucket Start %",
        "Foundation Bucket End %",
        "Foundation Count",
        "Matched Count",
        "Match %",
        "Median Delay Days",
        "Average Delay Days",
        "P90 Delay Days",
        "Negative Excluded",
        "Unmatched",
        "Source Type",
        "Parser Modes",
        "Coverage Status",
    ]
    monthly_columns = [
        "Project",
        "Month",
        "Foundation Count",
        "Matched Count",
        "Match %",
        "Median Delay Days",
        "Average Delay Days",
        "P90 Delay Days",
        "Negative Excluded",
        "Unmatched",
        "Source Type",
        "Parser Modes",
        "Coverage Status",
    ]
    coverage_columns = [
        "Project",
        "Eligible",
        "Coverage Status",
        "Source Type",
        "Parser Modes",
        "Reason",
        "Foundation Locations",
        "Matched Locations",
        "Negative Excluded",
        "Unmatched Locations",
        "Notes",
    ]
    anomaly_columns = [
        "Project",
        "Location",
        "Foundation Date",
        "Erection Start",
        "Delay Days",
        "Issue",
        "Foundation Line",
        "Erection Line",
    ]

    coverage = _apply_project_rollup_columns(foundation_coverage) if isinstance(foundation_coverage, pd.DataFrame) else pd.DataFrame()
    diagnostics = _apply_project_rollup_columns(foundation_diagnostics) if isinstance(foundation_diagnostics, pd.DataFrame) else pd.DataFrame()
    merged = _build_merged_delay_fact(source_daily, foundation_completions)

    if coverage.empty and diagnostics.empty and merged.empty:
        return (
            pd.DataFrame(columns=phase_columns),
            pd.DataFrame(columns=monthly_columns),
            pd.DataFrame(columns=coverage_columns),
            pd.DataFrame(columns=anomaly_columns),
        )

    coverage_by_key: dict[str, dict[str, str]] = {}
    if not coverage.empty:
        coverage["status"] = coverage.get("status", "").fillna("").astype(str).str.strip()
        coverage["source_used"] = coverage.get("source_used", "").fillna("").astype(str).str.strip().str.lower()
        coverage["reason"] = coverage.get("reason", "").fillna("").astype(str).str.strip()
        for project_key, group in coverage.groupby("project_rollup_key", dropna=False):
            key = str(project_key or "").strip()
            if not key:
                continue
            first = group.iloc[0]
            coverage_by_key[key] = {
                "project": str(first.get("project_rollup_display", "")).strip() or str(first.get("project_display", "")).strip(),
                "status": "; ".join(sorted({str(v).strip() for v in group["status"] if str(v).strip()})),
                "source_used": "; ".join(sorted({str(v).strip() for v in group["source_used"] if str(v).strip()})),
                "reason": " | ".join(sorted({str(v).strip() for v in group["reason"] if str(v).strip()})),
            }

    parser_modes_by_key: dict[str, str] = {}
    if not diagnostics.empty:
        if "Project" in diagnostics.columns and "project_code" not in diagnostics.columns:
            diagnostics["project_code"] = diagnostics["Project"]
        diagnostics["parser_mode"] = diagnostics.get("ParserMode", "").fillna("").astype(str).str.strip().str.lower()
        for project_key, group in diagnostics.groupby("project_rollup_key", dropna=False):
            key = str(project_key or "").strip()
            if not key:
                continue
            parser_modes = sorted({str(v).strip() for v in group["parser_mode"] if str(v).strip()})
            parser_modes_by_key[key] = ", ".join(parser_modes)

    if merged.empty:
        coverage_rows = []
        for project_key in sorted(coverage_by_key.keys()):
            cov = coverage_by_key.get(project_key, {})
            status = str(cov.get("status", "")).strip() or "MISSING"
            source_used = str(cov.get("source_used", "")).strip() or "missing"
            parser_modes = parser_modes_by_key.get(project_key, "")
            reason = str(cov.get("reason", "")).strip() or "No detail foundation completion events available."
            coverage_rows.append(
                {
                    "Project": str(cov.get("project", "")).strip() or project_key.upper(),
                    "Eligible": "No",
                    "Coverage Status": status,
                    "Source Type": source_used,
                    "Parser Modes": parser_modes,
                    "Reason": reason,
                    "Foundation Locations": 0,
                    "Matched Locations": 0,
                    "Negative Excluded": 0,
                    "Unmatched Locations": 0,
                    "Notes": "",
                }
            )
        return (
            pd.DataFrame(columns=phase_columns),
            pd.DataFrame(columns=monthly_columns),
            pd.DataFrame(coverage_rows, columns=coverage_columns),
            pd.DataFrame(columns=anomaly_columns),
        )

    merged["month"] = pd.to_datetime(merged["foundation_complete"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    all_project_keys = sorted(
        set(merged["project_rollup_key"].dropna().astype(str).str.strip().tolist())
        | set(coverage_by_key.keys())
    )
    phase_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    anomaly_rows: list[dict[str, object]] = []

    for project_key in all_project_keys:
        if not project_key:
            continue
        cov = coverage_by_key.get(project_key, {})
        project_rows = merged[merged["project_rollup_key"] == project_key].copy()
        project_display = str(cov.get("project", "")).strip()
        if not project_display and not project_rows.empty:
            project_display = str(project_rows.iloc[0].get("project_rollup_display", "")).strip()
        if not project_display:
            project_display = project_key.upper()

        status = str(cov.get("status", "")).strip() or "MISSING"
        source_used = str(cov.get("source_used", "")).strip() or "missing"
        parser_modes = parser_modes_by_key.get(project_key, "")
        status_upper = status.upper()
        parser_upper = parser_modes.upper()

        foundation_count = int(len(project_rows.index))
        matched_count = int(project_rows["matched"].sum()) if not project_rows.empty else 0
        negative_excluded = int(project_rows["negative_delay"].sum()) if not project_rows.empty else 0
        unmatched = max(foundation_count - matched_count, 0)

        eligible = True
        reason = ""
        if "SKIPPED_BLANK_CONFIG" in status_upper:
            eligible = False
            reason = "Foundation mapping intentionally blank in DPR_Config."
        elif "ROWWISE" in parser_upper:
            eligible = False
            reason = "Rowwise foundation parser mode is excluded for delay trend."
        elif "detail" not in source_used.lower():
            eligible = False
            reason = "Only snapshot/missing foundation source is available."
        elif foundation_count == 0:
            eligible = False
            reason = "No detail foundation completion events available."

        if not reason:
            reason = str(cov.get("reason", "")).strip()

        if eligible and foundation_count > 0:
            ordered = project_rows.sort_values(["foundation_complete", "location_no_norm"]).reset_index(drop=True)
            ordered["phase_bucket"] = _phase_bucket_legacy(len(ordered.index))

            for phase in range(1, 6):
                phase_scope = ordered[ordered["phase_bucket"] == phase].copy()
                f_count = int(len(phase_scope.index))
                m_count = int(phase_scope["matched"].sum()) if f_count else 0
                n_excl = int(phase_scope["negative_delay"].sum()) if f_count else 0
                u_count = max(f_count - m_count, 0)
                m_pct = (m_count / f_count * 100.0) if f_count else 0.0
                median_delay, avg_delay, p90_delay = _series_stats(phase_scope["delay_for_stats"] if f_count else pd.Series(dtype="float64"))
                phase_rows.append(
                    {
                        "Project": project_display,
                        "Phase": f"Phase {phase}",
                        "Foundation Bucket Start %": (phase - 1) * 20,
                        "Foundation Bucket End %": phase * 20,
                        "Foundation Count": f_count,
                        "Matched Count": m_count,
                        "Match %": round(m_pct, 2),
                        "Median Delay Days": median_delay,
                        "Average Delay Days": avg_delay,
                        "P90 Delay Days": p90_delay,
                        "Negative Excluded": n_excl,
                        "Unmatched": u_count,
                        "Source Type": source_used or "detail",
                        "Parser Modes": parser_modes,
                        "Coverage Status": status,
                    }
                )

            for month_key, month_scope in ordered.groupby("month", dropna=False):
                if pd.isna(month_key):
                    continue
                f_count = int(len(month_scope.index))
                m_count = int(month_scope["matched"].sum()) if f_count else 0
                n_excl = int(month_scope["negative_delay"].sum()) if f_count else 0
                u_count = max(f_count - m_count, 0)
                m_pct = (m_count / f_count * 100.0) if f_count else 0.0
                median_delay, avg_delay, p90_delay = _series_stats(month_scope["delay_for_stats"])
                monthly_rows.append(
                    {
                        "Project": project_display,
                        "Month": pd.Timestamp(month_key).strftime("%Y-%m"),
                        "Foundation Count": f_count,
                        "Matched Count": m_count,
                        "Match %": round(m_pct, 2),
                        "Median Delay Days": median_delay,
                        "Average Delay Days": avg_delay,
                        "P90 Delay Days": p90_delay,
                        "Negative Excluded": n_excl,
                        "Unmatched": u_count,
                        "Source Type": source_used or "detail",
                        "Parser Modes": parser_modes,
                        "Coverage Status": status,
                    }
                )

            negatives = ordered[ordered["negative_delay"]].copy()
            for _, row in negatives.iterrows():
                anomaly_rows.append(
                    {
                        "Project": project_display,
                        "Location": str(row.get("location_no", "")).strip(),
                        "Foundation Date": pd.Timestamp(row.get("foundation_complete")).strftime("%Y-%m-%d"),
                        "Erection Start": pd.Timestamp(row.get("erection_start_final")).strftime("%Y-%m-%d"),
                        "Delay Days": float(row.get("delay_days")),
                        "Issue": "NEGATIVE_DELAY_EXCLUDED",
                        "Foundation Line": str(row.get("foundation_line", "")).strip(),
                        "Erection Line": str(row.get("erection_line", "")).strip(),
                    }
                )
            unresolved = ordered[~ordered["matched"]].copy()
            for _, row in unresolved.iterrows():
                anomaly_rows.append(
                    {
                        "Project": project_display,
                        "Location": str(row.get("location_no", "")).strip(),
                        "Foundation Date": pd.Timestamp(row.get("foundation_complete")).strftime("%Y-%m-%d"),
                        "Erection Start": "",
                        "Delay Days": pd.NA,
                        "Issue": "UNMATCHED_LOCATION",
                        "Foundation Line": str(row.get("foundation_line", "")).strip(),
                        "Erection Line": "",
                    }
                )

        coverage_rows.append(
            {
                "Project": project_display,
                "Eligible": "Yes" if eligible else "No",
                "Coverage Status": status,
                "Source Type": source_used,
                "Parser Modes": parser_modes,
                "Reason": reason,
                "Foundation Locations": foundation_count,
                "Matched Locations": matched_count,
                "Negative Excluded": negative_excluded,
                "Unmatched Locations": unmatched,
                "Notes": "Negative delays are excluded from trend statistics.",
            }
        )

    return (
        pd.DataFrame(phase_rows, columns=phase_columns),
        pd.DataFrame(monthly_rows, columns=monthly_columns),
        pd.DataFrame(coverage_rows, columns=coverage_columns),
        pd.DataFrame(anomaly_rows, columns=anomaly_columns),
    )


def _build_erection_completion_events_for_gap(source_daily: pd.DataFrame) -> pd.DataFrame:
    if source_daily is None or source_daily.empty:
        return pd.DataFrame()
    source = _normalize_erection_source_columns(source_daily)
    if source.empty:
        return pd.DataFrame()
    work = _apply_project_rollup_columns(source)
    completion_series = _pick_series(
        source,
        ("completion_date", "Complete Date", "complete date", "completion date"),
    )
    work["completion_date"] = _coerce_mixed_excel_dates_series(completion_series)
    work = work[work["completion_date"].notna()].copy()
    if work.empty:
        return work
    work = work[work["completion_date"].dt.year >= 1980].copy()
    if work.empty:
        return work
    work["location_no"] = work.get("location_no", "").fillna("").astype(str).str.strip()
    work["line_name"] = work.get("line_name", "").fillna("").astype(str).str.strip()
    work["project_code"] = work.get("project_code", "").fillna("").astype(str).str.strip()
    work["project_code_norm"] = work["project_code"].map(_compact_project_key)
    work["line_name_norm"] = work["line_name"].map(_compact_project_key)
    work["location_no_norm"] = work["location_no"].map(_compact_project_key)
    dedupe_cols = ["project_code_norm", "line_name_norm", "location_no_norm", "completion_date"]
    work = work.drop_duplicates(subset=dedupe_cols, keep="last")
    return work


def build_foundation_vs_erection_gap_tables_legacy(
    source_daily: pd.DataFrame,
    foundation_completions: pd.DataFrame,
    foundation_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_columns = [
        "Project",
        "Source Type",
        "Snapshot Limited",
        "Month",
        "Foundations Cumulative",
        "Erections Cumulative",
        "Gap Cumulative",
    ]
    weekly_columns = [
        "Project",
        "Source Type",
        "Snapshot Limited",
        "Week Start",
        "Week End",
        "Week",
        "Foundations Cumulative",
        "Erections Cumulative",
        "Gap Cumulative",
    ]
    coverage_columns = [
        "Project",
        "Foundation Source Used",
        "Coverage Status",
        "Coverage Reason",
        "Snapshot Limited",
        "First Month Available",
        "Last Month Available",
        "First Week Available",
        "Last Week Available",
        "Detail Rows",
        "Detail Completions",
        "Snapshot Rows",
        "Erection First Month",
        "Erection Last Month",
        "Notes",
    ]

    erections = _build_erection_completion_events_for_gap(source_daily)
    foundation = _apply_project_rollup_columns(foundation_completions) if isinstance(foundation_completions, pd.DataFrame) else pd.DataFrame()
    coverage = _apply_project_rollup_columns(foundation_coverage) if isinstance(foundation_coverage, pd.DataFrame) else pd.DataFrame()

    if foundation.empty and erections.empty and coverage.empty:
        return (
            pd.DataFrame(columns=monthly_columns),
            pd.DataFrame(columns=weekly_columns),
            pd.DataFrame(columns=coverage_columns),
        )

    if not foundation.empty:
        foundation["event_date"] = _coerce_mixed_excel_dates_series(foundation.get("event_date", pd.Series(dtype="object")))
        foundation["location_no"] = foundation.get("location_no", "").fillna("").astype(str).str.strip()
        foundation["line_name"] = foundation.get("line_name", "").fillna("").astype(str).str.strip()
        foundation["source_type"] = foundation.get("source_type", "").fillna("").astype(str).str.strip().str.lower()
        foundation["project_code"] = foundation.get("project_code", "").fillna("").astype(str).str.strip()
        foundation["project_code_norm"] = foundation["project_code"].map(_compact_project_key)
        foundation["line_name_norm"] = foundation["line_name"].map(_compact_project_key)
        foundation["location_no_norm"] = foundation["location_no"].map(_compact_project_key)
        foundation["cumulative_foundation"] = pd.to_numeric(foundation.get("cumulative_foundation"), errors="coerce")
    else:
        foundation = pd.DataFrame()

    coverage_by_key: dict[str, dict[str, object]] = {}
    excluded_gap_projects: set[str] = set()
    if not coverage.empty:
        coverage["status"] = coverage.get("status", "").fillna("").astype(str).str.strip()
        coverage["reason"] = coverage.get("reason", "").fillna("").astype(str).str.strip()
        coverage["source_used"] = coverage.get("source_used", "").fillna("").astype(str).str.strip()
        coverage["snapshot_limited"] = coverage.get("snapshot_limited", pd.Series("", index=coverage.index)).fillna("").astype(str).str.strip()
        coverage["detail_rows"] = pd.to_numeric(coverage.get("detail_rows", pd.Series(0, index=coverage.index)), errors="coerce").fillna(0).astype(int)
        coverage["detail_completions"] = pd.to_numeric(coverage.get("detail_completions", pd.Series(0, index=coverage.index)), errors="coerce").fillna(0).astype(int)
        coverage["snapshot_rows"] = pd.to_numeric(coverage.get("snapshot_rows", pd.Series(0, index=coverage.index)), errors="coerce").fillna(0).astype(int)
        for project_key, group in coverage.groupby("project_rollup_key", dropna=False):
            key = _safe_text(project_key)
            if not key:
                continue
            first = group.iloc[0]
            coverage_by_key[key] = {
                "project": _safe_text(first.get("project_rollup_display", "")) or _safe_text(first.get("project_display", "")),
                "status": "; ".join(sorted({_safe_text(v) for v in group["status"] if _safe_text(v)})),
                "reason": " | ".join(sorted({_safe_text(v) for v in group["reason"] if _safe_text(v)})),
                "source_used": "; ".join(sorted({_safe_text(v) for v in group["source_used"] if _safe_text(v)})),
                "snapshot_limited": "Yes" if any(_safe_text(v).lower() == "yes" for v in group["snapshot_limited"]) else "No",
                "detail_rows": int(group["detail_rows"].sum()),
                "detail_completions": int(group["detail_completions"].sum()),
                "snapshot_rows": int(group["snapshot_rows"].sum()),
            }
            status_text = str(coverage_by_key[key].get("status", "")).upper()
            if "SKIPPED_BLANK_CONFIG" in status_text:
                excluded_gap_projects.add(key)

    detail = pd.DataFrame()
    snapshot = pd.DataFrame()
    if not foundation.empty:
        detail = foundation[(foundation["source_type"] == "detail") & foundation["event_date"].notna()].copy()
        if not detail.empty:
            detail = detail.drop_duplicates(
                subset=["project_code_norm", "line_name_norm", "location_no_norm", "event_date"],
                keep="last",
            )
        snapshot = foundation[(foundation["source_type"] != "detail") & foundation["event_date"].notna()].copy()
        if not snapshot.empty:
            snapshot = (
                snapshot.sort_values("event_date")
                .groupby(["project_rollup_key", "project_rollup_display", "event_date"], as_index=False)
                .agg(cumulative_foundation=("cumulative_foundation", "max"))
            )

    projects: dict[str, str] = {}
    for frame in (erections, detail, snapshot, coverage):
        if frame is None or frame.empty:
            continue
        key_series = frame.get("project_rollup_key")
        display_series = frame.get("project_rollup_display")
        if key_series is None or display_series is None:
            continue
        for key, display in zip(key_series, display_series):
            key_text = _safe_text(key)
            display_text = _safe_text(display)
            if key_text and display_text and key_text not in projects:
                projects[key_text] = display_text
    for excluded_key in excluded_gap_projects:
        projects.pop(excluded_key, None)

    monthly_rows: list[dict[str, object]] = []
    weekly_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for project_key, project_name in sorted(projects.items(), key=lambda item: item[1]):
        project_erections = erections[erections["project_rollup_key"] == project_key].copy() if not erections.empty else pd.DataFrame()
        project_detail = detail[detail["project_rollup_key"] == project_key].copy() if not detail.empty else pd.DataFrame()
        project_snapshot = snapshot[snapshot["project_rollup_key"] == project_key].copy() if not snapshot.empty else pd.DataFrame()
        coverage_rec = coverage_by_key.get(project_key, {})

        if not project_detail.empty:
            source_type = "detail"
            snapshot_limited = "No"
        elif not project_snapshot.empty:
            source_type = "snapshot_fallback"
            snapshot_limited = "Yes"
        else:
            source_type = _safe_text(coverage_rec.get("source_used", "")) or "missing"
            snapshot_limited = _safe_text(coverage_rec.get("snapshot_limited", "No")) or "No"

        erection_month = (
            project_erections.assign(period=project_erections["completion_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("period")
            .size()
            if not project_erections.empty
            else pd.Series(dtype="int64")
        )
        erection_week = (
            project_erections.assign(period=project_erections["completion_date"] - pd.to_timedelta((project_erections["completion_date"].dt.weekday + 1) % 7, unit="D"))
            .groupby("period")
            .size()
            if not project_erections.empty
            else pd.Series(dtype="int64")
        )
        erection_month_cum = erection_month.sort_index().cumsum() if not erection_month.empty else erection_month
        erection_week_cum = erection_week.sort_index().cumsum() if not erection_week.empty else erection_week

        if source_type == "detail":
            foundation_month_count = (
                project_detail.assign(period=project_detail["event_date"].dt.to_period("M").dt.to_timestamp())
                .groupby("period")
                .size()
            )
            foundation_week_count = (
                project_detail.assign(period=project_detail["event_date"] - pd.to_timedelta((project_detail["event_date"].dt.weekday + 1) % 7, unit="D"))
                .groupby("period")
                .size()
            )
            foundation_month_cum = foundation_month_count.sort_index().cumsum()
            foundation_week_cum = foundation_week_count.sort_index().cumsum()
        elif source_type == "snapshot_fallback":
            foundation_month_cum = (
                project_snapshot.assign(period=project_snapshot["event_date"].dt.to_period("M").dt.to_timestamp())
                .groupby("period")["cumulative_foundation"]
                .max()
                .sort_index()
            )
            foundation_week_cum = (
                project_snapshot.assign(period=project_snapshot["event_date"] - pd.to_timedelta((project_snapshot["event_date"].dt.weekday + 1) % 7, unit="D"))
                .groupby("period")["cumulative_foundation"]
                .max()
                .sort_index()
            )
        else:
            foundation_month_cum = pd.Series(dtype="float64")
            foundation_week_cum = pd.Series(dtype="float64")

        month_index = sorted(set(erection_month_cum.index.tolist()) | set(foundation_month_cum.index.tolist()))
        week_index = sorted(set(erection_week_cum.index.tolist()) | set(foundation_week_cum.index.tolist()))

        if source_type == "snapshot_fallback":
            fm = foundation_month_cum.reindex(month_index).ffill()
            fw = foundation_week_cum.reindex(week_index).ffill()
        elif source_type == "missing":
            fm = pd.Series([float("nan")] * len(month_index), index=month_index, dtype="float64")
            fw = pd.Series([float("nan")] * len(week_index), index=week_index, dtype="float64")
        else:
            fm = foundation_month_cum.reindex(month_index).ffill().fillna(0.0) if month_index else pd.Series(dtype="float64")
            fw = foundation_week_cum.reindex(week_index).ffill().fillna(0.0) if week_index else pd.Series(dtype="float64")

        em = erection_month_cum.reindex(month_index).ffill().fillna(0.0) if month_index else pd.Series(dtype="float64")
        ew = erection_week_cum.reindex(week_index).ffill().fillna(0.0) if week_index else pd.Series(dtype="float64")
        if not em.empty:
            em = pd.to_numeric(em, errors="coerce").fillna(0.0)
        if not ew.empty:
            ew = pd.to_numeric(ew, errors="coerce").fillna(0.0)
        if source_type == "detail":
            if not fm.empty:
                fm = pd.to_numeric(fm, errors="coerce").fillna(0.0)
            if not fw.empty:
                fw = pd.to_numeric(fw, errors="coerce").fillna(0.0)

        for period in month_index:
            foundation_value = fm.loc[period] if period in fm.index else pd.NA
            erection_value = em.loc[period] if period in em.index else 0.0
            gap_value = pd.NA if pd.isna(foundation_value) else float(foundation_value) - float(erection_value)
            monthly_rows.append(
                {
                    "Project": project_name,
                    "Source Type": source_type,
                    "Snapshot Limited": snapshot_limited,
                    "Month": pd.Timestamp(period).strftime("%Y-%m"),
                    "Foundations Cumulative": foundation_value,
                    "Erections Cumulative": float(erection_value),
                    "Gap Cumulative": gap_value,
                }
            )

        for period in week_index:
            week_start = pd.Timestamp(period).normalize()
            week_end = week_start + pd.Timedelta(days=6)
            foundation_value = fw.loc[period] if period in fw.index else pd.NA
            erection_value = ew.loc[period] if period in ew.index else 0.0
            gap_value = pd.NA if pd.isna(foundation_value) else float(foundation_value) - float(erection_value)
            weekly_rows.append(
                {
                    "Project": project_name,
                    "Source Type": source_type,
                    "Snapshot Limited": snapshot_limited,
                    "Week Start": week_start.strftime("%Y-%m-%d"),
                    "Week End": week_end.strftime("%Y-%m-%d"),
                    "Week": f"{week_start:%Y-%m-%d} to {week_end:%Y-%m-%d}",
                    "Foundations Cumulative": foundation_value,
                    "Erections Cumulative": float(erection_value),
                    "Gap Cumulative": gap_value,
                }
            )

        if not project_erections.empty:
            ere_first_month = project_erections["completion_date"].min().strftime("%Y-%m")
            ere_last_month = project_erections["completion_date"].max().strftime("%Y-%m")
        else:
            ere_first_month = ""
            ere_last_month = ""

        month_available = sorted({row["Month"] for row in monthly_rows if row["Project"] == project_name and pd.notna(row["Foundations Cumulative"])})
        week_available = sorted({row["Week"] for row in weekly_rows if row["Project"] == project_name and pd.notna(row["Foundations Cumulative"])})

        coverage_rows.append(
            {
                "Project": project_name,
                "Foundation Source Used": source_type,
                "Coverage Status": _safe_text(coverage_rec.get("status", "")) or ("OK_DETAIL" if source_type == "detail" else "MISSING"),
                "Coverage Reason": _safe_text(coverage_rec.get("reason", "")),
                "Snapshot Limited": snapshot_limited,
                "First Month Available": month_available[0] if month_available else "",
                "Last Month Available": month_available[-1] if month_available else "",
                "First Week Available": week_available[0] if week_available else "",
                "Last Week Available": week_available[-1] if week_available else "",
                "Detail Rows": int(coverage_rec.get("detail_rows", 0)),
                "Detail Completions": int(coverage_rec.get("detail_completions", 0)),
                "Snapshot Rows": int(coverage_rec.get("snapshot_rows", 0)),
                "Erection First Month": ere_first_month,
                "Erection Last Month": ere_last_month,
                "Notes": "Snapshot values are carry-forwarded between reporting dates." if source_type == "snapshot_fallback" else "",
            }
        )

    monthly_df = pd.DataFrame(monthly_rows, columns=monthly_columns)
    weekly_df = pd.DataFrame(weekly_rows, columns=weekly_columns)
    coverage_df = pd.DataFrame(coverage_rows, columns=coverage_columns)
    return monthly_df, weekly_df, coverage_df


def _normalize_activity_for_scope(frame: pd.DataFrame) -> pd.Series:
    if "activity_norm" in frame.columns:
        activity = frame["activity_norm"].fillna("").astype(str).str.strip().str.lower()
    else:
        activity = frame.get("activity_raw", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip().str.lower()
    return activity.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")


def _numeric_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: "NA" if pd.isna(value) else f"{float(value):.6f}")


def _build_scope_snapshot(progress_status_raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "project_rollup_key",
        "Project",
        "project_base_code",
        "scope_total",
        "scope_source",
        "scope_report_date",
        "status_rows_considered",
        "status_rows_used",
        "duplicate_rows_dropped",
        "quantity_primary_sum",
        "cumulative_progress_sum",
        "balance_progress_sum",
        "note",
    ]
    if progress_status_raw is None or progress_status_raw.empty:
        return pd.DataFrame(columns=columns)

    work = _apply_project_rollup_columns(progress_status_raw)
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["activity_scope_key"] = _normalize_activity_for_scope(work)
    work = work[work["activity_scope_key"].eq("foundation")].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["report_date"] = _coerce_mixed_excel_dates_series(work.get("report_date", pd.Series(dtype="object")))
    work = work[work["report_date"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["section_key"] = work.get("section_label", "").fillna("").astype(str).str.strip().str.lower()
    work["activity_key"] = work.get("activity_raw", "").fillna("").astype(str).str.strip().str.lower()
    work["q_key"] = _numeric_key(work.get("quantity_primary", pd.Series(dtype="object")))
    work["c_key"] = _numeric_key(work.get("cumulative_progress", pd.Series(dtype="object")))
    work["b_key"] = _numeric_key(work.get("balance_progress", pd.Series(dtype="object")))

    work["quantity_primary_num"] = pd.to_numeric(work.get("quantity_primary"), errors="coerce")
    work["cumulative_progress_num"] = pd.to_numeric(work.get("cumulative_progress"), errors="coerce")
    work["balance_progress_num"] = pd.to_numeric(work.get("balance_progress"), errors="coerce")

    rows: list[dict[str, object]] = []
    for project_key, project_rows in work.groupby("project_rollup_key", dropna=False):
        key = _safe_text(project_key)
        if not key:
            continue
        latest_date = pd.to_datetime(project_rows["report_date"], errors="coerce").max()
        if pd.isna(latest_date):
            continue
        latest_rows = project_rows[project_rows["report_date"] == latest_date].copy()
        considered = int(len(latest_rows.index))
        dedup = latest_rows.drop_duplicates(
            subset=[
                "project_rollup_key",
                "report_date",
                "section_key",
                "activity_key",
                "q_key",
                "c_key",
                "b_key",
            ],
            keep="last",
        )
        used = int(len(dedup.index))
        dropped = max(considered - used, 0)

        qty_sum = float(pd.to_numeric(dedup["quantity_primary_num"], errors="coerce").sum(skipna=True))
        cum_sum = float(pd.to_numeric(dedup["cumulative_progress_num"], errors="coerce").sum(skipna=True))
        bal_sum = float(pd.to_numeric(dedup["balance_progress_num"], errors="coerce").sum(skipna=True))

        scope_total: float | pd._libs.missing.NAType
        scope_source: str
        note = ""
        if qty_sum > 0:
            scope_total = qty_sum
            scope_source = "status_quantity_primary_latest"
        elif (cum_sum + bal_sum) > 0:
            scope_total = float(cum_sum + bal_sum)
            scope_source = "status_cumulative_plus_balance_latest"
            note = "Quantity primary was non-positive; fallback used cumulative+balance."
        else:
            scope_total = pd.NA
            scope_source = "status_scope_missing"
            note = "No positive scope value found in latest foundation status rows."

        first = dedup.iloc[0] if not dedup.empty else latest_rows.iloc[0]
        project_display = _safe_text(first.get("project_rollup_display", "")) or key.upper()
        base_code = _safe_text(first.get("project_base_code", "")) or extract_base_project_code(project_display)

        rows.append(
            {
                "project_rollup_key": key,
                "Project": project_display,
                "project_base_code": base_code,
                "scope_total": scope_total,
                "scope_source": scope_source,
                "scope_report_date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
                "status_rows_considered": considered,
                "status_rows_used": used,
                "duplicate_rows_dropped": dropped,
                "quantity_primary_sum": qty_sum,
                "cumulative_progress_sum": cum_sum,
                "balance_progress_sum": bal_sum,
                "note": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _series_label(base_code: str) -> str:
    digits = re.findall(r"\d{3,4}", _safe_text(base_code))
    if not digits:
        return "Other"
    lead = digits[0][0]
    if lead in {"4", "5", "6"}:
        return f"{lead}xx"
    return "Other"


def _ownership_label(base_code: str) -> str:
    text = _safe_text(base_code).upper()
    if text.startswith("TA"):
        return "Government"
    if text.startswith("TB"):
        return "Private"
    return "Unknown"


def _assign_phase_label(progress_pct: float) -> tuple[str, int, int]:
    for start, end, label in PHASE_BANDS:
        if progress_pct <= end:
            return label, start, end if end < 1000 else 100
    return "80-100+", 80, 100


def _monsoon_overlap_days(start: pd.Timestamp | pd.NaT, end: pd.Timestamp | pd.NaT) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e < s:
        return 0
    days = pd.date_range(s, e, freq="D")
    return int(sum(1 for day in days if int(day.month) in MONSOON_MONTHS))


def _build_phase_rows(
    facts: pd.DataFrame,
    *,
    group_column: str,
    group_label: str,
) -> pd.DataFrame:
    columns = [
        group_label,
        "Phase",
        "Foundation Bucket Start %",
        "Foundation Bucket End %",
        "Foundation Count",
        "Matched Count",
        "Match %",
        "Median Delay Days",
        "Average Delay Days",
        "P90 Delay Days",
        "Negative Excluded",
        "Unmatched",
        "Scope Total",
        "Phase Foundations % of Scope",
        "Phase Start Date",
        "Phase End Date",
        "Phase Duration Days",
        "Monsoon Overlap",
        "Monsoon Overlap Days",
        "Monsoon Foundation Count",
    ]
    if facts.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group_value, group_scope in facts.groupby(group_column, dropna=False):
        group_value_text = _safe_text(group_value)
        if not group_value_text:
            continue
        project_scopes = (
            group_scope[["project_rollup_key", "scope_total"]]
            .drop_duplicates(subset=["project_rollup_key"])
        )
        group_scope_total = float(pd.to_numeric(project_scopes["scope_total"], errors="coerce").fillna(0.0).sum())

        for phase, phase_scope in sorted(
            group_scope.groupby("phase_label", dropna=False),
            key=lambda item: PHASE_ORDER.get(_safe_text(item[0]), 999),
        ):
            phase_label = _safe_text(phase)
            if not phase_label:
                continue
            foundation_count = int(len(phase_scope.index))
            matched_count = int(phase_scope["matched"].sum()) if foundation_count else 0
            negative_excluded = int(phase_scope["negative_delay"].sum()) if foundation_count else 0
            unmatched = max(foundation_count - matched_count, 0)
            match_pct = (matched_count / foundation_count * 100.0) if foundation_count else 0.0
            median_delay, avg_delay, p90_delay = _series_stats(phase_scope["delay_for_stats"])
            phase_start = pd.to_datetime(phase_scope["foundation_complete"], errors="coerce").min()
            phase_end = pd.to_datetime(phase_scope["foundation_complete"], errors="coerce").max()
            phase_duration = int((phase_end - phase_start).days + 1) if pd.notna(phase_start) and pd.notna(phase_end) else 0
            monsoon_days = _monsoon_overlap_days(phase_start, phase_end)
            monsoon_foundation_count = int(
                phase_scope["foundation_complete"].map(lambda value: int(pd.Timestamp(value).month) in MONSOON_MONTHS if pd.notna(value) else False).sum()
            )
            start_pct = int(pd.to_numeric(phase_scope["phase_start_pct"], errors="coerce").dropna().iloc[0]) if phase_scope["phase_start_pct"].notna().any() else pd.NA
            end_pct = int(pd.to_numeric(phase_scope["phase_end_pct"], errors="coerce").dropna().iloc[0]) if phase_scope["phase_end_pct"].notna().any() else pd.NA
            phase_scope_pct = (foundation_count / group_scope_total * 100.0) if group_scope_total > 0 else pd.NA
            rows.append(
                {
                    group_label: group_value_text,
                    "Phase": phase_label,
                    "Foundation Bucket Start %": start_pct,
                    "Foundation Bucket End %": end_pct,
                    "Foundation Count": foundation_count,
                    "Matched Count": matched_count,
                    "Match %": round(match_pct, 2),
                    "Median Delay Days": median_delay,
                    "Average Delay Days": avg_delay,
                    "P90 Delay Days": p90_delay,
                    "Negative Excluded": negative_excluded,
                    "Unmatched": unmatched,
                    "Scope Total": group_scope_total,
                    "Phase Foundations % of Scope": round(float(phase_scope_pct), 2) if pd.notna(phase_scope_pct) else pd.NA,
                    "Phase Start Date": pd.Timestamp(phase_start).strftime("%Y-%m-%d") if pd.notna(phase_start) else "",
                    "Phase End Date": pd.Timestamp(phase_end).strftime("%Y-%m-%d") if pd.notna(phase_end) else "",
                    "Phase Duration Days": phase_duration,
                    "Monsoon Overlap": "Yes" if monsoon_days > 0 else "No",
                    "Monsoon Overlap Days": monsoon_days,
                    "Monsoon Foundation Count": monsoon_foundation_count,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _build_dynamic_range_buckets(max_delay: int, step: int = 30) -> list[tuple[int, int, str]]:
    if max_delay < 0:
        max_delay = 0
    rounded_max = max(step, ((max_delay + step - 1) // step) * step)
    buckets: list[tuple[int, int, str]] = []
    for idx, end in enumerate(range(step, rounded_max + 1, step)):
        if idx == 0:
            start = 0
        else:
            start = ((idx - 1) * step) + step + 1
        buckets.append((start, end, f"{start}-{end}"))
    return buckets


def _build_bucket_rows(
    facts: pd.DataFrame,
    *,
    group_column: str,
    group_label: str,
) -> pd.DataFrame:
    columns = [
        group_label,
        "Bucket",
        "Bucket Start Days",
        "Bucket End Days",
        "Foundations In Bucket",
        "% of Foundations Done Till Date",
        "% of Scope",
        "Foundations Done Till Date",
        "Scope Total",
        "Matched Non-Negative",
        "Unmatched",
        "Negative Excluded",
    ]
    if facts.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group_value, group_scope in facts.groupby(group_column, dropna=False):
        group_value_text = _safe_text(group_value)
        if not group_value_text:
            continue
        foundations_done = int(len(group_scope.index))
        unmatched = int((~group_scope["matched"]).sum())
        negative_excluded = int(group_scope["negative_delay"].sum())
        project_scopes = (
            group_scope[["project_rollup_key", "scope_total"]]
            .drop_duplicates(subset=["project_rollup_key"])
        )
        scope_total = float(pd.to_numeric(project_scopes["scope_total"], errors="coerce").fillna(0.0).sum())

        eligible = group_scope[
            group_scope["matched"] & ~group_scope["negative_delay"] & pd.to_numeric(group_scope["delay_days"], errors="coerce").ge(0)
        ].copy()
        delays = pd.to_numeric(eligible["delay_days"], errors="coerce").dropna().astype(int)
        matched_non_negative = int(len(delays.index))
        if delays.empty:
            rows.append(
                {
                    group_label: group_value_text,
                    "Bucket": "No Matched Delays",
                    "Bucket Start Days": pd.NA,
                    "Bucket End Days": pd.NA,
                    "Foundations In Bucket": 0,
                    "% of Foundations Done Till Date": 0.0,
                    "% of Scope": 0.0 if scope_total > 0 else pd.NA,
                    "Foundations Done Till Date": foundations_done,
                    "Scope Total": scope_total,
                    "Matched Non-Negative": matched_non_negative,
                    "Unmatched": unmatched,
                    "Negative Excluded": negative_excluded,
                }
            )
            continue

        buckets = _build_dynamic_range_buckets(int(delays.max()), step=30)
        for start, end, label in buckets:
            bucket_count = int(((delays >= start) & (delays <= end)).sum())
            pct_done = (bucket_count / foundations_done * 100.0) if foundations_done else 0.0
            pct_scope = (bucket_count / scope_total * 100.0) if scope_total > 0 else pd.NA
            rows.append(
                {
                    group_label: group_value_text,
                    "Bucket": label,
                    "Bucket Start Days": start,
                    "Bucket End Days": end,
                    "Foundations In Bucket": bucket_count,
                    "% of Foundations Done Till Date": round(pct_done, 2),
                    "% of Scope": round(float(pct_scope), 2) if pd.notna(pct_scope) else pd.NA,
                    "Foundations Done Till Date": foundations_done,
                    "Scope Total": scope_total,
                    "Matched Non-Negative": matched_non_negative,
                    "Unmatched": unmatched,
                    "Negative Excluded": negative_excluded,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _month_values_text(months: tuple[int, ...]) -> str:
    return ",".join(str(int(month)) for month in months)


def _mechanism_windows(config: MechanismConfig) -> tuple[list[tuple[str, tuple[int, ...]]], list[tuple[str, tuple[int, ...]]]]:
    cohort_windows = [
        ("All Foundations", tuple(range(1, 13))),
        ("Pre-Monsoon", config.pre_monsoon),
        ("Monsoon", config.monsoon),
        ("Dry Season", config.dry),
    ]
    start_windows = [
        ("Post-Monsoon", config.post_monsoon),
        ("Post-Monsoon Wide", config.post_monsoon_wide),
    ]
    return cohort_windows, start_windows


def _build_mechanism_summary_rows(
    facts: pd.DataFrame,
    *,
    group_column: str,
    group_label: str,
    config: MechanismConfig,
) -> pd.DataFrame:
    columns = [
        group_label,
        "Cohort Window",
        "Cohort Months",
        "Start Window",
        "Start Months",
        "Foundations Total",
        "Matched Non-Negative",
        "Starts In Window",
        "% of All Foundations",
        "% Within Matched",
        "Match Coverage %",
        "Min Foundations Threshold",
        "Threshold Applied",
    ]
    if facts.empty:
        return pd.DataFrame(columns=columns)
    cohort_windows, start_windows = _mechanism_windows(config)
    rows: list[dict[str, object]] = []
    for group_value, group_scope in facts.groupby(group_column, dropna=False):
        group_value_text = _safe_text(group_value)
        if not group_value_text:
            continue
        month_series = pd.to_numeric(group_scope.get("foundation_month_num"), errors="coerce")
        start_month_series = pd.to_numeric(group_scope.get("erection_start_month_num"), errors="coerce")
        for cohort_name, cohort_months in cohort_windows:
            cohort_scope = group_scope[month_series.isin(list(cohort_months))].copy()
            foundation_total = int(len(cohort_scope.index))
            if foundation_total < int(config.min_foundation_count):
                continue
            matched_scope = cohort_scope[
                cohort_scope["matched"]
                & ~cohort_scope["negative_delay"]
                & pd.to_numeric(cohort_scope.get("erection_start_month_num"), errors="coerce").notna()
            ].copy()
            matched_non_negative = int(len(matched_scope.index))
            for start_name, start_months in start_windows:
                starts_in_window = int(start_month_series.loc[matched_scope.index].isin(list(start_months)).sum()) if matched_non_negative else 0
                pct_all = (starts_in_window / foundation_total * 100.0) if foundation_total else pd.NA
                pct_matched = (starts_in_window / matched_non_negative * 100.0) if matched_non_negative else pd.NA
                match_coverage = (matched_non_negative / foundation_total * 100.0) if foundation_total else pd.NA
                rows.append(
                    {
                        group_label: group_value_text,
                        "Cohort Window": cohort_name,
                        "Cohort Months": _month_values_text(cohort_months),
                        "Start Window": start_name,
                        "Start Months": _month_values_text(start_months),
                        "Foundations Total": foundation_total,
                        "Matched Non-Negative": matched_non_negative,
                        "Starts In Window": starts_in_window,
                        "% of All Foundations": round(float(pct_all), 2) if pd.notna(pct_all) else pd.NA,
                        "% Within Matched": round(float(pct_matched), 2) if pd.notna(pct_matched) else pd.NA,
                        "Match Coverage %": round(float(match_coverage), 2) if pd.notna(match_coverage) else pd.NA,
                        "Min Foundations Threshold": int(config.min_foundation_count),
                        "Threshold Applied": "Yes",
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _build_mechanism_matrix_project(facts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Project",
        "Foundation Month",
        "Erection Start Month",
        "Foundations In Pair",
        "Foundations Total (Month)",
        "Matched Non-Negative (Month)",
        "% of All Foundations In Month",
        "% of Matched Non-Negative In Month",
    ]
    if facts.empty:
        return pd.DataFrame(columns=columns)

    work = facts.copy()
    work["foundation_month_period"] = pd.to_datetime(work.get("foundation_complete"), errors="coerce").dt.to_period("M")
    work["erection_month_period"] = pd.to_datetime(work.get("erection_start_final"), errors="coerce").dt.to_period("M")
    rows: list[dict[str, object]] = []
    for project, project_scope in work.groupby("Project", dropna=False):
        project_name = _safe_text(project)
        if not project_name:
            continue
        month_total = project_scope.groupby("foundation_month_period", dropna=False).size()
        matched_scope = project_scope[
            project_scope["matched"]
            & ~project_scope["negative_delay"]
            & project_scope["erection_month_period"].notna()
            & project_scope["foundation_month_period"].notna()
        ].copy()
        matched_month_total = matched_scope.groupby("foundation_month_period", dropna=False).size()
        pair_counts = matched_scope.groupby(["foundation_month_period", "erection_month_period"], dropna=False).size()
        for (foundation_period, erection_period), count in pair_counts.items():
            if pd.isna(foundation_period) or pd.isna(erection_period):
                continue
            fdn_total = int(month_total.get(foundation_period, 0))
            matched_total = int(matched_month_total.get(foundation_period, 0))
            pct_all = (int(count) / fdn_total * 100.0) if fdn_total else pd.NA
            pct_matched = (int(count) / matched_total * 100.0) if matched_total else pd.NA
            rows.append(
                {
                    "Project": project_name,
                    "Foundation Month": str(foundation_period),
                    "Erection Start Month": str(erection_period),
                    "Foundations In Pair": int(count),
                    "Foundations Total (Month)": fdn_total,
                    "Matched Non-Negative (Month)": matched_total,
                    "% of All Foundations In Month": round(float(pct_all), 2) if pd.notna(pct_all) else pd.NA,
                    "% of Matched Non-Negative In Month": round(float(pct_matched), 2) if pd.notna(pct_matched) else pd.NA,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _build_mechanism_matrix_overall(facts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Foundation Month",
        "Erection Start Month",
        "Foundations In Pair",
        "Foundations Total (Month)",
        "Matched Non-Negative (Month)",
        "% of All Foundations In Month",
        "% of Matched Non-Negative In Month",
    ]
    if facts.empty:
        return pd.DataFrame(columns=columns)
    work = facts.copy()
    work["foundation_month_period"] = pd.to_datetime(work.get("foundation_complete"), errors="coerce").dt.to_period("M")
    work["erection_month_period"] = pd.to_datetime(work.get("erection_start_final"), errors="coerce").dt.to_period("M")
    month_total = work.groupby("foundation_month_period", dropna=False).size()
    matched_scope = work[
        work["matched"]
        & ~work["negative_delay"]
        & work["foundation_month_period"].notna()
        & work["erection_month_period"].notna()
    ].copy()
    matched_month_total = matched_scope.groupby("foundation_month_period", dropna=False).size()
    pair_counts = matched_scope.groupby(["foundation_month_period", "erection_month_period"], dropna=False).size()
    rows: list[dict[str, object]] = []
    for (foundation_period, erection_period), count in pair_counts.items():
        if pd.isna(foundation_period) or pd.isna(erection_period):
            continue
        fdn_total = int(month_total.get(foundation_period, 0))
        matched_total = int(matched_month_total.get(foundation_period, 0))
        pct_all = (int(count) / fdn_total * 100.0) if fdn_total else pd.NA
        pct_matched = (int(count) / matched_total * 100.0) if matched_total else pd.NA
        rows.append(
            {
                "Foundation Month": str(foundation_period),
                "Erection Start Month": str(erection_period),
                "Foundations In Pair": int(count),
                "Foundations Total (Month)": fdn_total,
                "Matched Non-Negative (Month)": matched_total,
                "% of All Foundations In Month": round(float(pct_all), 2) if pd.notna(pct_all) else pd.NA,
                "% of Matched Non-Negative In Month": round(float(pct_matched), 2) if pd.notna(pct_matched) else pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_mechanism_evidence_audit(facts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Project",
        "Series",
        "Ownership",
        "Project Rollup Key",
        "Location",
        "Foundation Date",
        "Foundation Month",
        "Foundation Month Num",
        "Erection Start",
        "Erection Start Month",
        "Erection Start Month Num",
        "Delay Days",
        "Matched",
        "Negative Excluded",
        "Match Basis",
        "Foundation Line",
        "Erection Line",
        "Foundation Source File",
        "Foundation Source Sheet",
        "Erection Source File",
        "Erection Source Sheet",
    ]
    if facts.empty:
        return pd.DataFrame(columns=columns)
    work = facts.copy()
    work["Foundation Date"] = pd.to_datetime(work.get("foundation_complete"), errors="coerce").dt.strftime("%Y-%m-%d")
    work["Foundation Date"] = work["Foundation Date"].where(work["Foundation Date"].notna(), "")
    work["Erection Start"] = pd.to_datetime(work.get("erection_start_final"), errors="coerce").dt.strftime("%Y-%m-%d")
    work["Erection Start"] = work["Erection Start"].where(work["Erection Start"].notna(), "")
    audit = pd.DataFrame(
        {
            "Project": work.get("Project", "").fillna("").astype(str).str.strip(),
            "Series": work.get("Series", "").fillna("").astype(str).str.strip(),
            "Ownership": work.get("Ownership", "").fillna("").astype(str).str.strip(),
            "Project Rollup Key": work.get("project_rollup_key", "").fillna("").astype(str).str.strip(),
            "Location": work.get("location_no", "").fillna("").astype(str).str.strip(),
            "Foundation Date": work["Foundation Date"],
            "Foundation Month": work.get("foundation_month", "").fillna("").astype(str).str.strip(),
            "Foundation Month Num": pd.to_numeric(work.get("foundation_month_num"), errors="coerce"),
            "Erection Start": work["Erection Start"],
            "Erection Start Month": work.get("erection_start_month", "").fillna("").astype(str).str.strip(),
            "Erection Start Month Num": pd.to_numeric(work.get("erection_start_month_num"), errors="coerce"),
            "Delay Days": pd.to_numeric(work.get("delay_days"), errors="coerce"),
            "Matched": work.get("matched", False).fillna(False).map(lambda value: "Yes" if bool(value) else "No"),
            "Negative Excluded": work.get("negative_delay", False).fillna(False).map(lambda value: "Yes" if bool(value) else "No"),
            "Match Basis": work.get("match_basis", "").fillna("").astype(str).str.strip(),
            "Foundation Line": work.get("foundation_line", "").fillna("").astype(str).str.strip(),
            "Erection Line": work.get("erection_line", "").fillna("").astype(str).str.strip(),
            "Foundation Source File": work.get("foundation_source_file", "").fillna("").astype(str).str.strip(),
            "Foundation Source Sheet": work.get("foundation_source_sheet", "").fillna("").astype(str).str.strip(),
            "Erection Source File": work.get("erection_source_file", "").fillna("").astype(str).str.strip(),
            "Erection Source Sheet": work.get("erection_source_sheet", "").fillna("").astype(str).str.strip(),
        }
    )
    return audit.reindex(columns=columns)


def _build_mechanism_config_table(config: MechanismConfig) -> pd.DataFrame:
    rows = [
        {"Parameter": "pre_monsoon", "Value": _month_values_text(config.pre_monsoon)},
        {"Parameter": "monsoon", "Value": _month_values_text(config.monsoon)},
        {"Parameter": "post_monsoon", "Value": _month_values_text(config.post_monsoon)},
        {"Parameter": "post_monsoon_wide", "Value": _month_values_text(config.post_monsoon_wide)},
        {"Parameter": "dry", "Value": _month_values_text(config.dry)},
        {"Parameter": "min_foundation_count", "Value": str(int(config.min_foundation_count))},
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def build_foundation_delay_analysis_tables(
    source_daily: pd.DataFrame,
    foundation_completions: pd.DataFrame,
    foundation_coverage: pd.DataFrame,
    foundation_diagnostics: pd.DataFrame,
    progress_status_raw: pd.DataFrame,
    daily_reference: pd.DataFrame | None = None,
    mechanism_config: MechanismConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build Foundation Delay Analysis V2 tables."""
    resolved_mechanism_config = _resolve_mechanism_config(mechanism_config)
    merged = _build_merged_delay_fact(source_daily, foundation_completions)
    daily_merged = (
        _build_merged_delay_fact(daily_reference, foundation_completions)
        if isinstance(daily_reference, pd.DataFrame) and not daily_reference.empty
        else pd.DataFrame()
    )
    coverage = _apply_project_rollup_columns(foundation_coverage) if isinstance(foundation_coverage, pd.DataFrame) else pd.DataFrame()
    diagnostics = _apply_project_rollup_columns(foundation_diagnostics) if isinstance(foundation_diagnostics, pd.DataFrame) else pd.DataFrame()
    scope_snapshot = _build_scope_snapshot(progress_status_raw if isinstance(progress_status_raw, pd.DataFrame) else pd.DataFrame())

    coverage_by_key: dict[str, dict[str, str]] = {}
    if not coverage.empty:
        coverage["status"] = coverage.get("status", "").fillna("").astype(str).str.strip()
        coverage["source_used"] = coverage.get("source_used", "").fillna("").astype(str).str.strip()
        coverage["reason"] = coverage.get("reason", "").fillna("").astype(str).str.strip()
        coverage["project_base_code"] = coverage.get("project_base_code", "").fillna("").astype(str).str.strip()
        for project_key, group in coverage.groupby("project_rollup_key", dropna=False):
            key = _safe_text(project_key)
            if not key:
                continue
            first = group.iloc[0]
            coverage_by_key[key] = {
                "Project": _safe_text(first.get("project_rollup_display", "")) or _safe_text(first.get("project_display", "")) or key.upper(),
                "status": "; ".join(sorted({_safe_text(v) for v in group["status"] if _safe_text(v)})),
                "source_used": "; ".join(sorted({_safe_text(v) for v in group["source_used"] if _safe_text(v)})),
                "reason": " | ".join(sorted({_safe_text(v) for v in group["reason"] if _safe_text(v)})),
                "project_base_code": _safe_text(first.get("project_base_code", "")),
            }

    parser_modes_by_key: dict[str, str] = {}
    if not diagnostics.empty:
        diagnostics["parser_mode"] = diagnostics.get("ParserMode", "").fillna("").astype(str).str.strip().str.lower()
        for project_key, group in diagnostics.groupby("project_rollup_key", dropna=False):
            key = _safe_text(project_key)
            if not key:
                continue
            parser_modes = sorted({_safe_text(v) for v in group["parser_mode"] if _safe_text(v)})
            parser_modes_by_key[key] = ", ".join(parser_modes)

    scope_by_key = {
        _safe_text(row.get("project_rollup_key", "")): row
        for _, row in scope_snapshot.iterrows()
        if _safe_text(row.get("project_rollup_key", ""))
    }

    all_project_keys = sorted(
        set(merged["project_rollup_key"].dropna().astype(str).str.strip().tolist()) if not merged.empty else set()
        | set(coverage_by_key.keys())
        | set(scope_by_key.keys())
    )

    fact_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    scope_rows: list[dict[str, Any]] = []

    for project_key in all_project_keys:
        if not project_key:
            continue
        project_scope = merged[merged["project_rollup_key"] == project_key].copy() if not merged.empty else pd.DataFrame()
        coverage_rec = coverage_by_key.get(project_key, {})
        scope_rec = scope_by_key.get(project_key, {})

        project_name = _safe_text(coverage_rec.get("Project", "")) or (
            _safe_text(project_scope.iloc[0].get("project_rollup_display", "")) if not project_scope.empty else _safe_text(scope_rec.get("Project", "")) or project_key.upper()
        )
        base_code = _safe_text(coverage_rec.get("project_base_code", ""))
        if not base_code and not project_scope.empty:
            base_code = _safe_text(project_scope.iloc[0].get("project_base_code", ""))
        if not base_code:
            base_code = _safe_text(scope_rec.get("project_base_code", ""))
        if not base_code:
            base_code = extract_base_project_code(project_name)

        status = _safe_text(coverage_rec.get("status", "")) or "MISSING"
        source_used = _safe_text(coverage_rec.get("source_used", "")) or "missing"
        reason = _safe_text(coverage_rec.get("reason", ""))
        parser_modes = _safe_text(parser_modes_by_key.get(project_key, ""))

        foundation_count = int(len(project_scope.index))
        matched_count = int(project_scope["matched"].sum()) if foundation_count else 0
        raw_start_matches = int(project_scope["matched"].sum()) if foundation_count else 0
        alias_matches = int(project_scope["match_basis"].astype(str).str.strip().eq("alias").sum()) if foundation_count else 0
        negative_excluded = int(project_scope["negative_delay"].sum()) if foundation_count else 0
        unmatched = max(foundation_count - matched_count, 0)
        dropped_recovered = 0
        if not daily_merged.empty and foundation_count:
            daily_scope = daily_merged[daily_merged["project_rollup_key"] == project_key].copy()
            if not daily_scope.empty:
                daily_key = daily_scope[["line_name_norm", "location_no_norm", "foundation_complete", "matched"]].copy()
                daily_key["daily_matched"] = daily_key["matched"].fillna(False)
                daily_key = daily_key.drop(columns=["matched"])
                probe = project_scope.merge(
                    daily_key,
                    on=["line_name_norm", "location_no_norm", "foundation_complete"],
                    how="left",
                )
                dropped_recovered = int((probe["matched"] & ~probe["daily_matched"].fillna(False)).sum())

        scope_total_value = pd.to_numeric(pd.Series([scope_rec.get("scope_total", pd.NA)]), errors="coerce").iloc[0]
        if pd.notna(scope_total_value) and float(scope_total_value) > 0:
            scope_total = float(scope_total_value)
            scope_source = _safe_text(scope_rec.get("scope_source", "")) or "status_quantity_primary_latest"
            scope_report_date = _safe_text(scope_rec.get("scope_report_date", ""))
            scope_note = _safe_text(scope_rec.get("note", ""))
            status_rows_considered = int(pd.to_numeric(pd.Series([scope_rec.get("status_rows_considered", 0)]), errors="coerce").fillna(0).iloc[0])
            status_rows_used = int(pd.to_numeric(pd.Series([scope_rec.get("status_rows_used", 0)]), errors="coerce").fillna(0).iloc[0])
            duplicate_rows_dropped = int(pd.to_numeric(pd.Series([scope_rec.get("duplicate_rows_dropped", 0)]), errors="coerce").fillna(0).iloc[0])
        else:
            scope_total = float(foundation_count)
            scope_source = "fallback_foundation_done_count"
            scope_report_date = ""
            scope_note = "Status scope missing/non-positive; fallback to foundations done count."
            status_rows_considered = 0
            status_rows_used = 0
            duplicate_rows_dropped = 0

        scope_rows.append(
            {
                "Project": project_name,
                "Project Rollup Key": project_key,
                "Project Base Code": base_code,
                "Series": _series_label(base_code),
                "Ownership": _ownership_label(base_code),
                "Scope Total": scope_total,
                "Scope Source": scope_source,
                "Scope Report Date": scope_report_date,
                "Status Rows Considered": status_rows_considered,
                "Status Rows Used": status_rows_used,
                "Duplicate Rows Dropped": duplicate_rows_dropped,
                "Quantity Primary Sum": scope_rec.get("quantity_primary_sum", pd.NA),
                "Cumulative Progress Sum": scope_rec.get("cumulative_progress_sum", pd.NA),
                "Balance Progress Sum": scope_rec.get("balance_progress_sum", pd.NA),
                "Fallback Foundation Done Count": foundation_count,
                "Coverage Status": status,
                "Coverage Reason": reason,
                "Note": scope_note,
            }
        )

        if not project_scope.empty:
            ordered = project_scope.sort_values(["foundation_complete", "location_no_norm"]).reset_index(drop=True)
            ordered["running_foundations"] = range(1, len(ordered.index) + 1)
            ordered["running_pct"] = ordered["running_foundations"] / max(scope_total, 1.0) * 100.0
            phase_parts = ordered["running_pct"].map(_assign_phase_label)
            ordered["phase_label"] = phase_parts.map(lambda item: item[0])
            ordered["phase_start_pct"] = phase_parts.map(lambda item: item[1])
            ordered["phase_end_pct"] = phase_parts.map(lambda item: item[2])
            ordered["Project"] = project_name
            ordered["project_base_code"] = base_code
            ordered["Series"] = _series_label(base_code)
            ordered["Ownership"] = _ownership_label(base_code)
            ordered["scope_total"] = scope_total
            ordered["scope_source"] = scope_source
            ordered["scope_report_date"] = scope_report_date
            ordered["source_used"] = source_used
            ordered["coverage_status"] = status
            ordered["parser_modes"] = parser_modes
            fact_rows.extend(ordered.to_dict(orient="records"))

            negatives = ordered[ordered["negative_delay"]].copy()
            for _, row in negatives.iterrows():
                anomaly_rows.append(
                    {
                        "Project": project_name,
                        "Series": _series_label(base_code),
                        "Ownership": _ownership_label(base_code),
                        "Phase": _safe_text(row.get("phase_label", "")),
                        "Location": _safe_text(row.get("location_no", "")),
                        "Foundation Date": pd.Timestamp(row.get("foundation_complete")).strftime("%Y-%m-%d"),
                        "Erection Start": pd.Timestamp(row.get("erection_start_final")).strftime("%Y-%m-%d"),
                        "Delay Days": float(row.get("delay_days")),
                        "Issue": "NEGATIVE_DELAY_EXCLUDED",
                        "Foundation Line": _safe_text(row.get("foundation_line", "")),
                        "Erection Line": _safe_text(row.get("erection_line", "")),
                    }
                )
            unresolved = ordered[~ordered["matched"]].copy()
            for _, row in unresolved.iterrows():
                anomaly_rows.append(
                    {
                        "Project": project_name,
                        "Series": _series_label(base_code),
                        "Ownership": _ownership_label(base_code),
                        "Phase": _safe_text(row.get("phase_label", "")),
                        "Location": _safe_text(row.get("location_no", "")),
                        "Foundation Date": pd.Timestamp(row.get("foundation_complete")).strftime("%Y-%m-%d"),
                        "Erection Start": "",
                        "Delay Days": pd.NA,
                        "Issue": "UNMATCHED_LOCATION",
                        "Foundation Line": _safe_text(row.get("foundation_line", "")),
                        "Erection Line": "",
                    }
                )

        coverage_rows.append(
            {
                "Project": project_name,
                "Series": _series_label(base_code),
                "Ownership": _ownership_label(base_code),
                "Eligible": "Yes" if foundation_count > 0 else "No",
                "Coverage Status": status,
                "Source Type": source_used,
                "Parser Modes": parser_modes,
                "Reason": reason or ("No detail foundation completion events available." if foundation_count == 0 else ""),
                "Foundation Locations": foundation_count,
                "Matched Locations": matched_count,
                "RawData Start-Date Matches": raw_start_matches,
                "Dropped-by-Daily Recovered": dropped_recovered,
                "Alias Matches": alias_matches,
                "Negative Excluded": negative_excluded,
                "Unmatched Locations": unmatched,
                "Scope Total": scope_total,
                "Scope Source": scope_source,
                "Scope Report Date": scope_report_date,
                "Status Rows Considered": status_rows_considered,
                "Status Rows Used": status_rows_used,
                "Duplicate Rows Dropped": duplicate_rows_dropped,
                "Notes": "Negative delays are excluded from trend stats; dynamic buckets use non-negative matched delays only.",
            }
        )

    facts = pd.DataFrame(fact_rows)
    scope_snapshot_df = pd.DataFrame(scope_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    anomalies_df = pd.DataFrame(anomaly_rows)

    if facts.empty:
        project_phase = _build_phase_rows(pd.DataFrame(), group_column="Project", group_label="Project")
        series_phase = _build_phase_rows(pd.DataFrame(), group_column="Series", group_label="Series")
        ownership_phase = _build_phase_rows(pd.DataFrame(), group_column="Ownership", group_label="Ownership")
        project_buckets = _build_bucket_rows(pd.DataFrame(), group_column="Project", group_label="Project")
        series_buckets = _build_bucket_rows(pd.DataFrame(), group_column="Series", group_label="Series")
        ownership_buckets = _build_bucket_rows(pd.DataFrame(), group_column="Ownership", group_label="Ownership")
    else:
        project_phase = _build_phase_rows(facts, group_column="Project", group_label="Project")
        series_phase = _build_phase_rows(facts, group_column="Series", group_label="Series")
        ownership_phase = _build_phase_rows(facts, group_column="Ownership", group_label="Ownership")
        project_buckets = _build_bucket_rows(facts, group_column="Project", group_label="Project")
        series_buckets = _build_bucket_rows(facts, group_column="Series", group_label="Series")
        ownership_buckets = _build_bucket_rows(facts, group_column="Ownership", group_label="Ownership")

    if not project_phase.empty:
        meta = coverage_df[["Project", "Source Type", "Parser Modes", "Coverage Status", "Scope Source", "Scope Report Date"]].drop_duplicates(subset=["Project"])
        project_phase = project_phase.merge(meta, on="Project", how="left")

    mechanism_summary_project = _build_mechanism_summary_rows(
        facts,
        group_column="Project",
        group_label="Project",
        config=resolved_mechanism_config,
    )
    mechanism_summary_series = _build_mechanism_summary_rows(
        facts,
        group_column="Series",
        group_label="Series",
        config=resolved_mechanism_config,
    )
    mechanism_summary_ownership = _build_mechanism_summary_rows(
        facts,
        group_column="Ownership",
        group_label="Ownership",
        config=resolved_mechanism_config,
    )
    mechanism_matrix_project = _build_mechanism_matrix_project(facts)
    mechanism_matrix_overall = _build_mechanism_matrix_overall(facts)
    mechanism_audit = _build_mechanism_evidence_audit(facts)
    mechanism_cfg = _build_mechanism_config_table(resolved_mechanism_config)

    return {
        "Delay Phase - Project": project_phase,
        "Delay Phase - Series": series_phase,
        "Delay Phase - Ownership": ownership_phase,
        "Delay Buckets - Project": project_buckets,
        "Delay Buckets - Series": series_buckets,
        "Delay Buckets - Ownership": ownership_buckets,
        "Delay Coverage": coverage_df,
        "Delay Anomalies": anomalies_df,
        "Scope Snapshot": scope_snapshot_df,
        "Mechanism Summary - Project": mechanism_summary_project,
        "Mechanism Summary - Series": mechanism_summary_series,
        "Mechanism Summary - Ownership": mechanism_summary_ownership,
        "Mechanism Matrix - Project": mechanism_matrix_project,
        "Mechanism Matrix - Overall": mechanism_matrix_overall,
        "Mechanism Evidence Audit": mechanism_audit,
        "Mechanism Config": mechanism_cfg,
    }


def _normalize_monitoring_project_code(value: object) -> str:
    code = extract_base_project_code(value)
    if code:
        return code
    text = _safe_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).upper()


def _normalize_activity_text(value: object) -> str:
    text = _safe_text(value)
    return text.lower()


def _as_numeric(series: pd.Series | object) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series([series]), errors="coerce")


def _first_valid_number(series: pd.Series) -> float | None:
    numeric = _as_numeric(series).dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _sum_number(series: pd.Series) -> float | None:
    numeric = _as_numeric(series).dropna()
    if numeric.empty:
        return None
    return float(numeric.sum())


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
        "status_final_sag_available",
        "status_paying_out_available",
        "stretch_readiness_available",
        "manpower_available",
        "fitter_available",
        "location_nos_available",
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
    stretch_audit = (
        stretch_manpower_audit.copy() if isinstance(stretch_manpower_audit, pd.DataFrame) else pd.DataFrame()
    )

    if not status.empty:
        status["project_code"] = status.get("project_code", "").map(_normalize_monitoring_project_code)
        status = status[status["project_code"].astype(bool)].copy()
        status["month"] = pd.to_datetime(status.get("month"), errors="coerce").dt.to_period("M").dt.to_timestamp()
        status["report_date"] = pd.to_datetime(status.get("report_date"), errors="coerce")
        status["activity_norm_text"] = status.get("activity_norm", "").map(_normalize_activity_text)
        status["activity_raw_text"] = status.get("activity_raw", "").map(_normalize_activity_text)
        status["activity_group_text"] = status.get("activity_group", "").map(_normalize_activity_text)

    if not manpower.empty:
        manpower["project_code"] = manpower.get("project_code", "").map(_normalize_monitoring_project_code)
        manpower = manpower[manpower["project_code"].astype(bool)].copy()
        manpower["date"] = pd.to_datetime(manpower.get("date"), errors="coerce")
        manpower["month"] = pd.to_datetime(manpower.get("month"), errors="coerce").dt.to_period("M").dt.to_timestamp()
        manpower["gang_name"] = manpower.get("gang_name", "").fillna("").astype(str).str.strip()
        manpower["manpower_gang_strength"] = _as_numeric(manpower.get("manpower_gang_strength", pd.Series(dtype="object")))
        manpower["manpower_fitters"] = _as_numeric(manpower.get("manpower_fitters", pd.Series(dtype="object")))
        manpower["daily_km"] = _as_numeric(manpower.get("daily_km", pd.Series(dtype="object")))

    if not compiled.empty:
        compiled["project_code"] = compiled.get("project_code", "").map(_normalize_monitoring_project_code)
        compiled = compiled[compiled["project_code"].astype(bool)].copy()
        compiled["gang_name"] = compiled.get("gang_name", "").fillna("").astype(str).str.strip()
        location_series = compiled.get("location nos", pd.Series("", index=compiled.index)).fillna("").astype(str).str.strip()
        from_series = compiled.get("from_ap", pd.Series("", index=compiled.index)).fillna("").astype(str).str.strip()
        to_series = compiled.get("to_ap", pd.Series("", index=compiled.index)).fillna("").astype(str).str.strip()
        compiled["__is_span_row"] = (
            location_series.astype(bool)
            | from_series.astype(bool)
            | to_series.astype(bool)
        )
        compiled["__has_location_nos"] = location_series.astype(bool)

    if not stretch.empty:
        stretch["project_code"] = stretch.get("project_code", "").map(_normalize_monitoring_project_code)
        stretch = stretch[stretch["project_code"].astype(bool)].copy()
        stretch["ready_km"] = _as_numeric(stretch.get("ready_km", pd.Series(dtype="object")))
        stretch["total_km"] = _as_numeric(stretch.get("total_km", pd.Series(dtype="object")))
        stretch["readiness_pct"] = _as_numeric(stretch.get("readiness_pct", pd.Series(dtype="object")))

    if not stretch_audit.empty:
        stretch_audit["project_code"] = stretch_audit.get("project_code", "").map(_normalize_monitoring_project_code)
        stretch_audit = stretch_audit[stretch_audit["project_code"].astype(bool)].copy()
        stretch_audit["status_text"] = stretch_audit.get("status", "").map(_normalize_activity_text)

    project_codes: set[str] = set()
    for frame in (status, manpower, compiled, stretch, stretch_audit):
        if not frame.empty and "project_code" in frame.columns:
            project_codes.update(
                frame["project_code"].fillna("").astype(str).str.strip().tolist()
            )
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
            stretch_audit[stretch_audit["project_code"] == project_code].copy()
            if not stretch_audit.empty
            else pd.DataFrame()
        )

        latest_month = (
            status_project["month"].dropna().max() if not status_project.empty else pd.NaT
        )
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

        project_total_scope_km = _sum_number(final_or_string_rows.get("quantity_primary", pd.Series(dtype="object")))
        monthly_plan_km = _sum_number(final_or_string_rows.get("plan_for_month", pd.Series(dtype="object")))
        fs_achieved_month_km = _sum_number(final_or_string_rows.get("progress_for_month", pd.Series(dtype="object")))
        final_sag_completed_km = _sum_number(final_or_string_rows.get("cumulative_progress", pd.Series(dtype="object")))
        final_sag_balance_km = _sum_number(final_or_string_rows.get("balance_progress", pd.Series(dtype="object")))
        rough_sag_completed_km = _sum_number(paying_rows.get("cumulative_progress", pd.Series(dtype="object")))
        rough_sag_balance_km = _sum_number(paying_rows.get("balance_progress", pd.Series(dtype="object")))

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
            total_km = _as_numeric(stretch_project.get("total_km", pd.Series(dtype="object"))).fillna(0).sum()
            ready_km = _as_numeric(stretch_project.get("ready_km", pd.Series(dtype="object"))).fillna(0).sum()
            if total_km > 0:
                stretch_readiness_pct = float((ready_km / total_km) * 100.0)
            else:
                stretch_readiness_pct = _first_valid_number(
                    _as_numeric(stretch_project.get("readiness_pct", pd.Series(dtype="object")))
                )

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
            if manpower_by_gang.notna().any():
                existing_manpower = float(manpower_by_gang.dropna().sum())
            if fitters_by_gang.notna().any():
                fitter_count = float(fitters_by_gang.dropna().sum())

        report_month_text = ""
        if pd.notna(latest_month):
            report_month_text = pd.Timestamp(latest_month).strftime("%Y-%m")

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
            gang_scope = manpower_project[manpower_project["gang_name"] == gang_name].copy() if not manpower_project.empty else pd.DataFrame()
            gang_existing_manpower = None
            gang_fitter_count = None
            gang_fs_achieved = None
            if not gang_scope.empty:
                latest_gang_date = gang_scope["date"].dropna().max()
                if pd.notna(latest_gang_date):
                    gang_day = gang_scope[gang_scope["date"] == latest_gang_date].copy()
                else:
                    gang_day = gang_scope.copy()
                gang_existing_manpower = _first_valid_number(gang_day["manpower_gang_strength"])
                gang_fitter_count = _first_valid_number(gang_day["manpower_fitters"])
                if pd.notna(latest_month):
                    gang_month = gang_scope[gang_scope["month"] == latest_month].copy()
                else:
                    gang_month = gang_scope.copy()
                gang_fs_achieved = _sum_number(gang_month["daily_km"])
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

        status_final_sag_available = not final_or_string_rows.empty
        status_paying_out_available = not paying_rows.empty
        stretch_readiness_available = not stretch_project.empty

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
            if stretch_readiness_pct is not None:
                stretch_fill_pct = 100.0
            else:
                stretch_fill_pct = float((stretch_project["readiness_pct"].notna().sum() / len(stretch_project)) * 100.0)

        status_final_fill = 0.0
        if not final_or_string_rows.empty:
            final_vals = final_or_string_rows[["quantity_primary", "plan_for_month", "progress_for_month", "cumulative_progress", "balance_progress"]]
            status_final_fill = float((final_vals.notna().sum().sum() / (len(final_vals.index) * len(final_vals.columns))) * 100.0) if len(final_vals.index) else 0.0

        status_paying_fill = 0.0
        if not paying_rows.empty:
            paying_vals = paying_rows[["cumulative_progress", "balance_progress"]]
            status_paying_fill = float((paying_vals.notna().sum().sum() / (len(paying_vals.index) * len(paying_vals.columns))) * 100.0) if len(paying_vals.index) else 0.0

        missing_tags: list[str] = []
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
        if not stretch_audit_project.empty:
            has_absent = stretch_audit_project["status_text"].str.contains("absent", regex=False).any()
            if has_absent and "MISSING_STRETCH_MANPOWER_SIGNAL" not in missing_tags:
                missing_tags.append("MISSING_STRETCH_MANPOWER_SIGNAL")

        audit_rows.append(
            {
                "project_code": project_code,
                "status_final_sag_available": "Yes" if status_final_sag_available else "No",
                "status_paying_out_available": "Yes" if status_paying_out_available else "No",
                "stretch_readiness_available": "Yes" if stretch_readiness_available else "No",
                "manpower_available": "Yes" if manpower_available else "No",
                "fitter_available": "Yes" if fitter_available else "No",
                "location_nos_available": "Yes" if location_nos_available else "No",
                "status_final_sag_fill_pct": round(status_final_fill, 1),
                "status_paying_out_fill_pct": round(status_paying_fill, 1),
                "stretch_readiness_fill_pct": round(stretch_fill_pct, 1),
                "manpower_fill_pct": round(manpower_fill_pct, 1),
                "fitter_fill_pct": round(fitter_fill_pct, 1),
                "location_nos_fill_pct": round(location_fill_pct, 1),
                "missing_data_tags": ";".join(missing_tags),
            }
        )

    numeric_df = pd.DataFrame(numeric_rows, columns=numeric_columns)
    audit_df = pd.DataFrame(audit_rows, columns=audit_columns)
    return numeric_df, audit_df


def build_complete_foundation_analysis_tables(
    *,
    raw_erection_source: pd.DataFrame,
    foundation_completions: pd.DataFrame,
    foundation_coverage: pd.DataFrame,
    foundation_diagnostics: pd.DataFrame,
    progress_status_raw: pd.DataFrame,
    stringing_status_activity_fact: pd.DataFrame | None = None,
    stringing_manpower_fact: pd.DataFrame | None = None,
    stringing_compiled_raw: pd.DataFrame | None = None,
    stretch_readiness_summary: pd.DataFrame | None = None,
    stretch_readiness_manpower_audit: pd.DataFrame | None = None,
    daily_reference: pd.DataFrame | None = None,
    mechanism_config: MechanismConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build complete foundation analysis bundle (legacy + V2)."""
    legacy_source = build_legacy_erection_source_from_raw(raw_erection_source)
    v2_source = build_v2_erection_source_from_raw(raw_erection_source)
    gap_monthly, gap_weekly, gap_coverage = build_foundation_vs_erection_gap_tables_legacy(
        source_daily=legacy_source,
        foundation_completions=foundation_completions,
        foundation_coverage=foundation_coverage,
    )
    delay_phase, delay_monthly, delay_coverage, delay_anomalies = build_foundation_delay_trend_tables_legacy(
        source_daily=legacy_source,
        foundation_completions=foundation_completions,
        foundation_coverage=foundation_coverage,
        foundation_diagnostics=foundation_diagnostics,
    )
    v2_tables = build_foundation_delay_analysis_tables(
        source_daily=v2_source,
        foundation_completions=foundation_completions,
        foundation_coverage=foundation_coverage,
        foundation_diagnostics=foundation_diagnostics,
        progress_status_raw=progress_status_raw,
        daily_reference=daily_reference,
        mechanism_config=mechanism_config,
    )
    tables = {
        "Foundation Gap Monthly": gap_monthly,
        "Foundation Gap Weekly": gap_weekly,
        "Foundation Gap Coverage": gap_coverage,
        "Foundation Delay Phases": delay_phase,
        "Foundation Delay Monthly": delay_monthly,
        "Foundation Delay Coverage": delay_coverage,
        "Foundation Delay Anomalies": delay_anomalies,
    }
    tables.update(v2_tables)
    monitoring_numeric, monitoring_audit = _build_stringing_monitoring_tables(
        status_activity_fact=stringing_status_activity_fact,
        manpower_fact=stringing_manpower_fact,
        stringing_compiled_raw=stringing_compiled_raw,
        stretch_summary=stretch_readiness_summary,
        stretch_manpower_audit=stretch_readiness_manpower_audit,
    )
    tables["Stringing Monitoring Numeric"] = monitoring_numeric
    tables["Stringing Monitoring Audit"] = monitoring_audit
    return tables


def write_foundation_delay_analysis_workbook(
    output_path: str | Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    """Write Foundation Delay Analysis workbook (legacy + V2 tabs if provided)."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_sheets = [
        "Foundation Gap Monthly",
        "Foundation Gap Weekly",
        "Foundation Gap Coverage",
        "Foundation Delay Phases",
        "Foundation Delay Monthly",
        "Foundation Delay Coverage",
        "Foundation Delay Anomalies",
        "Delay Phase - Project",
        "Delay Phase - Series",
        "Delay Phase - Ownership",
        "Delay Buckets - Project",
        "Delay Buckets - Series",
        "Delay Buckets - Ownership",
        "Delay Coverage",
        "Delay Anomalies",
        "Scope Snapshot",
        "Mechanism Summary - Project",
        "Mechanism Summary - Series",
        "Mechanism Summary - Ownership",
        "Mechanism Matrix - Project",
        "Mechanism Matrix - Overall",
        "Mechanism Evidence Audit",
        "Mechanism Config",
        "Stringing Monitoring Numeric",
        "Stringing Monitoring Audit",
    ]
    seen = set(ordered_sheets)
    for sheet_name in tables.keys():
        if sheet_name not in seen:
            ordered_sheets.append(sheet_name)
            seen.add(sheet_name)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in ordered_sheets:
            if sheet not in tables:
                continue
            table = tables.get(sheet, pd.DataFrame())
            pd.DataFrame([[sheet]]).to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                header=False,
                startrow=0,
            )
            table.to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=1,
            )
    return output
