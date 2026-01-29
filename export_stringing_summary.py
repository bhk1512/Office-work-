#!/usr/bin/env python3
"""Standalone CLI to export the stringing productivity summary workbook."""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dashboard.config import AppConfig, configure_logging
from dashboard.plan_utils import compact_project_key, normalize_location
from dashboard.state import AppDataStore
from dashboard.stringing import (
    expand_stringing_to_daily_payout,
    expand_stringing_to_daily_fs,
    normalize_stringing_columns,
)

LOGGER = logging.getLogger("stringing_export")

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Stringing_Productivity_Summary.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the stringing productivity summary Excel workbook."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional override for the erection dataset root (Parquets/Erection).",
    )
    parser.add_argument(
        "--stringing-data-path",
        type=Path,
        help="Optional override for the stringing dataset root (Parquets/Stringing).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Optional YYYY-MM-DD start date to filter daily rows.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Optional YYYY-MM-DD end date to filter daily rows.",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = PRODUCTIVITY_ROOT / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:
        raise SystemExit(f"Invalid date '{value}': {exc}") from exc


def _filter_by_date(df: pd.DataFrame, column: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working[column] = pd.to_datetime(working[column], errors="coerce").dt.normalize()
    working = working.dropna(subset=[column])
    if start is not None:
        working = working[working[column] >= start]
    if end is not None:
        working = working[working[column] <= end]
    return working


_LOCATION_RE = re.compile(r"^\s*(\d+)([A-Za-z]+)?(?:\s*/\s*(\d+))?\s*$")


def _letter_rank(value: str) -> int:
    rank = 0
    for ch in value.upper():
        if "A" <= ch <= "Z":
            rank = (rank * 26) + (ord(ch) - ord("A") + 1)
    return rank


def _location_order_key(value: object) -> int | None:
    text = normalize_location(value)
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


def _ensure_project_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    working = df.copy()
    if "project_name" not in working.columns:
        working["project_name"] = working.get("project", "")
    if "project" not in working.columns:
        working["project"] = working.get("project_name", "")
    working["project_name"] = working["project_name"].fillna("").astype(str).str.strip()
    if "project_key_norm" not in working.columns:
        working["project_key_norm"] = working["project_name"].map(compact_project_key)
    working["project_key_norm"] = working["project_key_norm"].fillna("").astype(str)
    return working


def _add_span_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    working = _ensure_project_fields(df)
    for col in ("from_ap", "to_ap"):
        if col not in working.columns:
            working[col] = ""
    working["from_ap_norm"] = working["from_ap"].map(normalize_location)
    working["to_ap_norm"] = working["to_ap"].map(normalize_location)
    span_base = (
        working["project_key_norm"].fillna("")
        + "|"
        + working["from_ap_norm"].fillna("")
        + "|"
        + working["to_ap_norm"].fillna("")
    )
    fallback = working.get("row_id", pd.Series("", index=working.index))
    working["span_key"] = span_base.where(
        (working["from_ap_norm"].astype(bool) | working["to_ap_norm"].astype(bool)),
        fallback.astype(str),
    )
    return working


def _build_stage_summary(df: pd.DataFrame, stage_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            [
                {
                    "stage": stage_label,
                    "avg_productivity_km_day": 0.0,
                    "total_km": 0.0,
                    "active_days": 0,
                    "spans": 0,
                    "gangs": 0,
                    "projects": 0,
                    "start_date": pd.NaT,
                    "end_date": pd.NaT,
                }
            ]
        )
    working = _add_span_key(df)
    working["daily_km"] = pd.to_numeric(working.get("daily_km"), errors="coerce").fillna(0.0)
    dates = pd.to_datetime(working.get("date"), errors="coerce").dropna()
    avg_prod = round(float(working["daily_km"].mean()), 4) if len(working) else 0.0
    return pd.DataFrame(
        [
            {
                "stage": stage_label,
                "avg_productivity_km_day": avg_prod,
                "total_km": round(float(working["daily_km"].sum()), 4),
                "active_days": int(dates.nunique()) if not dates.empty else 0,
                "spans": int(working["span_key"].nunique()),
                "gangs": int(working.get("gang_name", pd.Series([], dtype=object)).nunique()),
                "projects": int(working["project_key_norm"].nunique()),
                "start_date": dates.min() if not dates.empty else pd.NaT,
                "end_date": dates.max() if not dates.empty else pd.NaT,
            }
        ]
    )


def _build_gang_productivity(df: pd.DataFrame, stage_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "stage",
                "gang_name",
                "avg_productivity_km_day",
                "total_km",
                "active_days",
                "spans",
                "projects",
            ]
        )
    working = _add_span_key(df)
    working["daily_km"] = pd.to_numeric(working.get("daily_km"), errors="coerce").fillna(0.0)
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
    working["gang_name"] = working.get("gang_name", "").fillna("").astype(str).str.strip()
    working = working[working["gang_name"].astype(bool)]
    if working.empty:
        return pd.DataFrame(
            columns=[
                "stage",
                "gang_name",
                "avg_productivity_km_day",
                "total_km",
                "active_days",
                "spans",
                "projects",
            ]
        )
    grouped = (
        working.groupby("gang_name", dropna=False)
        .agg(
            avg_productivity_km_day=("daily_km", "mean"),
            total_km=("daily_km", "sum"),
            active_days=("date", lambda s: s.dropna().nunique()),
            spans=("span_key", "nunique"),
            projects=("project_key_norm", "nunique"),
        )
        .reset_index()
    )
    grouped["avg_productivity_km_day"] = grouped["avg_productivity_km_day"].fillna(0.0).round(4)
    grouped["total_km"] = grouped["total_km"].fillna(0.0).round(4)
    grouped["active_days"] = grouped["active_days"].fillna(0).astype(int)
    grouped["spans"] = grouped["spans"].fillna(0).astype(int)
    grouped["projects"] = grouped["projects"].fillna(0).astype(int)
    grouped.insert(0, "stage", stage_label)
    return grouped.sort_values(["stage", "gang_name"]).reset_index(drop=True)


def _normalize_stringing_compiled(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized, _ = normalize_stringing_columns(df)
    normalized = _ensure_project_fields(normalized)
    for col in ("from_ap", "to_ap", "gang_name", "source_file"):
        if col not in normalized.columns:
            normalized[col] = ""
    normalized["from_ap"] = normalized["from_ap"].fillna("").astype(str).str.strip()
    normalized["to_ap"] = normalized["to_ap"].fillna("").astype(str).str.strip()
    normalized["gang_name"] = normalized["gang_name"].fillna("").astype(str).str.strip()
    normalized["source_file"] = normalized["source_file"].fillna("").astype(str).str.strip()
    return normalized


def _build_gap_summary(df: pd.DataFrame, gap_column: str) -> pd.DataFrame:
    if df is None or df.empty or gap_column not in df.columns:
        return pd.DataFrame(
            [
                {
                    "count": 0,
                    "avg_gap_days": 0.0,
                    "median_gap_days": 0.0,
                    "min_gap_days": 0,
                    "max_gap_days": 0,
                }
            ]
        )
    series = pd.to_numeric(df[gap_column], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame(
            [
                {
                    "count": 0,
                    "avg_gap_days": 0.0,
                    "median_gap_days": 0.0,
                    "min_gap_days": 0,
                    "max_gap_days": 0,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "count": int(series.size),
                "avg_gap_days": round(float(series.mean()), 2),
                "median_gap_days": round(float(series.median()), 2),
                "min_gap_days": int(series.min()),
                "max_gap_days": int(series.max()),
            }
        ]
    )


def _build_po_fs_gap_table(
    compiled: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    work = _normalize_stringing_compiled(compiled)
    if work.empty:
        return pd.DataFrame()
    work["po_completion_date"] = pd.to_datetime(work.get("po_completion_date"), errors="coerce").dt.normalize()
    work["fs_starting_date"] = pd.to_datetime(work.get("fs_starting_date"), errors="coerce").dt.normalize()

    scoped = work.copy()
    if start_date is not None:
        scoped = scoped[scoped["po_completion_date"] >= start_date]
    if end_date is not None:
        scoped = scoped[scoped["po_completion_date"] <= end_date]
    if scoped.empty:
        return pd.DataFrame()

    gap = (scoped["fs_starting_date"] - scoped["po_completion_date"]).dt.days
    negative_mask = gap < 0
    if negative_mask.any():
        for _, row in scoped.loc[negative_mask].iterrows():
            issues.append(
                {
                    "issue_type": "PO_FS_NEGATIVE_GAP",
                    "project": row.get("project_name", ""),
                    "from_ap": row.get("from_ap", ""),
                    "to_ap": row.get("to_ap", ""),
                    "po_completion_date": row.get("po_completion_date", pd.NaT),
                    "fs_starting_date": row.get("fs_starting_date", pd.NaT),
                    "details": "FS start is before PO completion",
                }
            )
    scoped["gap_days"] = gap.where(gap >= 0)
    scoped["gap_days"] = scoped["gap_days"].fillna(0).astype(int)
    scoped["negative_gap_flag"] = negative_mask.fillna(False)

    return scoped[
        [
            "project_name",
            "from_ap",
            "to_ap",
            "gang_name",
            "po_completion_date",
            "fs_starting_date",
            "gap_days",
            "negative_gap_flag",
            "source_file",
        ]
    ].copy()


def _build_erection_location_map(erection_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if erection_daily is None or erection_daily.empty:
        return {}
    working = erection_daily.copy()
    working["completion_date"] = pd.to_datetime(working.get("completion_date"), errors="coerce").dt.normalize()
    working["location_no"] = working.get("location_no", "").fillna("").astype(str).str.strip()
    working["project_key_norm"] = working.get("project_key_norm", "").fillna("").astype(str)
    working = working.dropna(subset=["completion_date"])
    working = working[working["location_no"].astype(bool) & working["project_key_norm"].astype(bool)]
    if working.empty:
        return {}
    working["location_no_norm"] = working["location_no"].map(normalize_location)
    working["loc_order"] = working["location_no_norm"].map(_location_order_key)
    working = working.dropna(subset=["loc_order"])
    if working.empty:
        return {}
    working = (
        working.sort_values("completion_date")
        .drop_duplicates(subset=["project_key_norm", "location_no_norm", "loc_order"], keep="last")
    )
    project_map: dict[str, pd.DataFrame] = {}
    for project_key, group in working.groupby("project_key_norm"):
        project_map[str(project_key)] = group[
            ["loc_order", "completion_date", "location_no_norm", "location_no"]
        ].copy()
    return project_map


def _build_erection_po_gap_table(
    compiled: pd.DataFrame,
    erection_daily: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    work = _normalize_stringing_compiled(compiled)
    if work.empty:
        return pd.DataFrame()
    work["po_start_date"] = pd.to_datetime(work.get("po_start_date"), errors="coerce").dt.normalize()
    if start_date is not None:
        work = work[work["po_start_date"] >= start_date]
    if end_date is not None:
        work = work[work["po_start_date"] <= end_date]
    if work.empty:
        return pd.DataFrame()

    work["from_ap_norm"] = work["from_ap"].map(normalize_location)
    work["to_ap_norm"] = work["to_ap"].map(normalize_location)
    work["from_order"] = work["from_ap_norm"].map(_location_order_key)
    work["to_order"] = work["to_ap_norm"].map(_location_order_key)

    erection_map = _build_erection_location_map(erection_daily)

    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        project_key = row.get("project_key_norm", "")
        po_start = row.get("po_start_date")
        from_order = row.get("from_order")
        to_order = row.get("to_order")
        from_ap = row.get("from_ap", "")
        to_ap = row.get("to_ap", "")
        base = {
            "project_name": row.get("project_name", ""),
            "from_ap": from_ap,
            "to_ap": to_ap,
            "gang_name": row.get("gang_name", ""),
            "po_start_date": po_start,
            "source_file": row.get("source_file", ""),
        }

        if pd.isna(po_start):
            issues.append(
                {
                    "issue_type": "PO_START_MISSING",
                    "project": row.get("project_name", ""),
                    "from_ap": from_ap,
                    "to_ap": to_ap,
                    "details": "Missing PO start date",
                }
            )
            rows.append({**base, "last_erection_completion_date": pd.NaT, "gap_days": pd.NA, "erection_locations": 0})
            continue

        if pd.isna(from_order) or pd.isna(to_order):
            issues.append(
                {
                    "issue_type": "LOCATION_PARSE_FAILED",
                    "project": row.get("project_name", ""),
                    "from_ap": from_ap,
                    "to_ap": to_ap,
                    "details": "Unable to parse span location ordering",
                }
            )
            rows.append({**base, "last_erection_completion_date": pd.NaT, "gap_days": pd.NA, "erection_locations": 0})
            continue

        project_df = erection_map.get(str(project_key))
        if project_df is None or project_df.empty:
            issues.append(
                {
                    "issue_type": "ERECTION_PROJECT_NOT_FOUND",
                    "project": row.get("project_name", ""),
                    "from_ap": from_ap,
                    "to_ap": to_ap,
                    "details": "No erection data for project",
                }
            )
            rows.append({**base, "last_erection_completion_date": pd.NaT, "gap_days": pd.NA, "erection_locations": 0})
            continue

        lo = min(int(from_order), int(to_order))
        hi = max(int(from_order), int(to_order))
        span_df = project_df[(project_df["loc_order"] >= lo) & (project_df["loc_order"] <= hi)]
        if span_df.empty:
            issues.append(
                {
                    "issue_type": "ERECTION_SPAN_EMPTY",
                    "project": row.get("project_name", ""),
                    "from_ap": from_ap,
                    "to_ap": to_ap,
                    "details": "No erection locations found in span range",
                }
            )
            rows.append({**base, "last_erection_completion_date": pd.NaT, "gap_days": pd.NA, "erection_locations": 0})
            continue

        last_completion = span_df["completion_date"].max()
        gap_days = (po_start - last_completion).days if pd.notna(last_completion) else pd.NA
        if gap_days is not pd.NA and gap_days < 0:
            issues.append(
                {
                    "issue_type": "ERECTION_PO_NEGATIVE_GAP",
                    "project": row.get("project_name", ""),
                    "from_ap": from_ap,
                    "to_ap": to_ap,
                    "details": "PO start is before last erection completion in span",
                }
            )
            gap_days = 0

        rows.append(
            {
                **base,
                "last_erection_completion_date": last_completion,
                "gap_days": int(gap_days) if gap_days is not pd.NA else pd.NA,
                "erection_locations": int(span_df["location_no_norm"].nunique()),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = _parse_args()
    configure_logging()

    config = AppConfig()
    if args.data_path:
        config = replace(config, data_path=Path(args.data_path).expanduser())
    if args.stringing_data_path:
        config = replace(config, stringing_data_path=Path(args.stringing_data_path).expanduser())
    config.validate()

    store = AppDataStore(config)
    LOGGER.info("Bootstrapping datasets from %s", config.data_path)
    store.bootstrap(config)

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date must be before or equal to --end-date.")

    stringing_daily = store.get_stringing()
    stringing_compiled = store.get_stringing_compiled()
    erection_daily = store.get_daily()

    if stringing_compiled is None or stringing_compiled.empty:
        LOGGER.warning("Stringing compiled dataset not available; some analyses will be empty.")

    # Overall stringing daily (PO start -> FS complete)
    overall_daily = _filter_by_date(stringing_daily, "date", start_date, end_date)

    # PO-only daily (PO start -> PO completion)
    po_daily = expand_stringing_to_daily_payout(stringing_compiled) if not stringing_compiled.empty else pd.DataFrame()
    po_daily = _filter_by_date(po_daily, "date", start_date, end_date)

    # FS-only daily (FS start -> FS complete)
    fs_daily = expand_stringing_to_daily_fs(stringing_compiled) if not stringing_compiled.empty else pd.DataFrame()
    fs_daily = _filter_by_date(fs_daily, "date", start_date, end_date)

    summary_rows = pd.concat(
        [
            _build_stage_summary(overall_daily, "Overall"),
            _build_stage_summary(po_daily, "P/O"),
            _build_stage_summary(fs_daily, "F/S"),
        ],
        ignore_index=True,
    )

    gang_rows = pd.concat(
        [
            _build_gang_productivity(overall_daily, "Overall"),
            _build_gang_productivity(po_daily, "P/O"),
            _build_gang_productivity(fs_daily, "F/S"),
        ],
        ignore_index=True,
    )

    issues: list[dict[str, object]] = []

    po_fs_gap = _build_po_fs_gap_table(
        stringing_compiled,
        start_date=start_date,
        end_date=end_date,
        issues=issues,
    )
    po_fs_gap_summary = _build_gap_summary(po_fs_gap, "gap_days")

    erection_po_gap = _build_erection_po_gap_table(
        stringing_compiled,
        erection_daily,
        start_date=start_date,
        end_date=end_date,
        issues=issues,
    )
    erection_po_gap_summary = _build_gap_summary(erection_po_gap, "gap_days")

    issues_df = pd.DataFrame(issues)

    output_path = _resolve_output_path(args.output)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_rows.to_excel(writer, sheet_name="Summary", index=False)
        gang_rows.to_excel(writer, sheet_name="Gang_Productivity", index=False)
        po_fs_gap.to_excel(writer, sheet_name="PO_FS_Gap", index=False)
        po_fs_gap_summary.to_excel(writer, sheet_name="PO_FS_Gap_Summary", index=False)
        erection_po_gap.to_excel(writer, sheet_name="Erection_PO_Gap", index=False)
        erection_po_gap_summary.to_excel(writer, sheet_name="Erection_PO_Gap_Summary", index=False)
        if not issues_df.empty:
            issues_df.to_excel(writer, sheet_name="Data_Issues", index=False)

    LOGGER.info("Stringing summary exported to %s", output_path)
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
