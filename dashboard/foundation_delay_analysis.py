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
    work = _apply_project_rollup_columns(source_daily)
    work["start_date"] = _coerce_mixed_excel_dates_series(work.get("start_date", pd.Series(dtype="object")))
    work = work[work["start_date"].notna()].copy()
    if work.empty:
        return work
    work = work[work["start_date"].dt.year >= 1980].copy()
    work["location_no"] = work.get("location_no", "").fillna("").astype(str).str.strip()
    work["line_name"] = work.get("line_name", "").fillna("").astype(str).str.strip()
    work["line_name_norm"] = work["line_name"].map(_compact_project_key)
    work["location_no_norm"] = work["location_no"].map(_compact_project_key)
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
    foundation["line_name_norm"] = foundation["line_name"].map(_compact_project_key)
    foundation["location_no_norm"] = foundation["location_no"].map(_compact_project_key)
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
        )
    )


def _build_merged_delay_fact(source_daily: pd.DataFrame, foundation_completions: pd.DataFrame) -> pd.DataFrame:
    foundation_loc = _build_foundation_detail_events(foundation_completions)
    if foundation_loc.empty:
        return pd.DataFrame()
    erection_loc = _build_erection_start_events(source_daily)
    if erection_loc.empty:
        erection_loc = pd.DataFrame(
            columns=["project_rollup_key", "line_name_norm", "location_no_norm", "erection_start", "erection_line"]
        )
    merged = foundation_loc.merge(
        erection_loc,
        on=["project_rollup_key", "line_name_norm", "location_no_norm"],
        how="left",
    )

    if not erection_loc.empty:
        ere_loc_counts = (
            erection_loc.groupby(["project_rollup_key", "location_no_norm"], as_index=False)
            .agg(
                ere_start_fallback=("erection_start", "min"),
                ere_line_count=("line_name_norm", "nunique"),
            )
        )
    else:
        ere_loc_counts = pd.DataFrame(columns=["project_rollup_key", "location_no_norm", "ere_start_fallback", "ere_line_count"])
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
    fallback = fallback[["project_rollup_key", "location_no_norm", "ere_start_fallback"]]
    merged = merged.merge(
        fallback,
        on=["project_rollup_key", "location_no_norm"],
        how="left",
    )
    merged["erection_start_final"] = merged["erection_start"].where(
        merged["erection_start"].notna(),
        merged["ere_start_fallback"],
    )
    merged["matched"] = merged["erection_start_final"].notna()
    merged["delay_days"] = (
        pd.to_datetime(merged["erection_start_final"], errors="coerce")
        - pd.to_datetime(merged["foundation_complete"], errors="coerce")
    ).dt.days
    merged["negative_delay"] = merged["matched"] & merged["delay_days"].lt(0)
    merged["delay_for_stats"] = merged["delay_days"].where(merged["matched"] & ~merged["negative_delay"])
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


def build_foundation_delay_analysis_tables(
    source_daily: pd.DataFrame,
    foundation_completions: pd.DataFrame,
    foundation_coverage: pd.DataFrame,
    foundation_diagnostics: pd.DataFrame,
    progress_status_raw: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build Foundation Delay Analysis V2 tables."""
    merged = _build_merged_delay_fact(source_daily, foundation_completions)
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
        negative_excluded = int(project_scope["negative_delay"].sum()) if foundation_count else 0
        unmatched = max(foundation_count - matched_count, 0)

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
    }


def write_foundation_delay_analysis_workbook(
    output_path: str | Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    """Write Foundation Delay Analysis workbook with one label row per sheet."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_sheets = [
        "Delay Phase - Project",
        "Delay Phase - Series",
        "Delay Phase - Ownership",
        "Delay Buckets - Project",
        "Delay Buckets - Series",
        "Delay Buckets - Ownership",
        "Delay Coverage",
        "Delay Anomalies",
        "Scope Snapshot",
    ]
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in ordered_sheets:
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

