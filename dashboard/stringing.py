"""Stringing dataset utilities: column normalization (map only).

This module provides a light-weight mapping from the exact spreadsheet
headers to normalized snake_case field names. It does not perform any
date expansion or type coercion — only column renaming plus a presence
report for required inputs.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re
import hashlib

import pandas as pd
import numpy as np


# Exact headers expected from the source sheet mapped to snake_case names
_STRINGING_COLUMN_MAP: Dict[str, str] = {
    "From AP": "from_ap",
    "To AP": "to_ap",
    "Method": "method",
    "Section Readiness": "section_readiness",
    "P/O Starting Date": "po_start_date",
    "P/O Completion Date": "po_completion_date",
    "P/O": "po",
    "F/S Starting Date": "fs_starting_date",
    "F/S/ Completion Date": "fs_complete_date",
    "Length": "length_m",
    "Status": "status",
    "Gang Name": "gang_name",
}


# --- Header detection utilities (tolerant like erection) ---
def _nrm_header(text: object) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _canon_key(text: object) -> str:
    return re.sub(r"\s+", "", _nrm_header(text))


_EXPECTED_HEADERS = list(_STRINGING_COLUMN_MAP.keys())
_EXPECTED_KEYS = {exp: _canon_key(exp) for exp in _EXPECTED_HEADERS}


def find_stringing_header_row(df_raw: pd.DataFrame, search_rows: int = 40) -> Tuple[Optional[int], Optional[List[str]]]:
    """Locate the header row within the top `search_rows` by matching expected headers.

    Returns (row_index, normalized_header_labels) or (None, None) if not found.
    """
    best: Optional[Tuple[int, List[str], float]] = None
    nrows = min(search_rows, df_raw.shape[0])

    for r in range(nrows):
        row_vals = [_nrm_header(x) for x in list(df_raw.iloc[r, :].values)]
        row_keys = [_canon_key(v) for v in row_vals]
        score = 0.0
        mapping: Dict[int, str] = {}
        used_expected: set[str] = set()
        for i, key in enumerate(row_keys):
            if not key:
                continue
            for exp, exp_key in _EXPECTED_KEYS.items():
                if exp in used_expected:
                    continue
                if key == exp_key:
                    mapping[i] = exp
                    score += 1.0
                    used_expected.add(exp)
                    break
        non_empty = sum(1 for v in row_vals if v)
        score += max(0, non_empty - 3) * 0.02
        if best is None or score > best[2]:
            cols = [mapping.get(i, row_vals[i]) for i in range(len(row_vals))]
            best = (r, cols, score)

    if best is None:
        return None, None
    return best[0], best[1]


def read_stringing_sheet_robust(path: str | bytes | "pathlike", sheet_name: str) -> pd.DataFrame:
    """Read a stringing sheet by inferring the header row if needed.

    - Reads with header=None, scans the first rows to find the header, renames columns,
      and returns rows under the detected header.
    - If detection fails, falls back to default header=0 read.
    """
    try:
        with pd.ExcelFile(path) as xl:
            df_raw = xl.parse(sheet_name=sheet_name, header=None)
    except Exception:
        # Fall back to a simple read
        return pd.read_excel(path, sheet_name=sheet_name)

    header_row, labels = find_stringing_header_row(df_raw)
    if header_row is None or labels is None:
        return pd.read_excel(path, sheet_name=sheet_name)

    # Slice rows below header_row and set columns to the detected labels
    data = df_raw.iloc[header_row + 1 :].copy()
    labels_series = pd.Series(labels)
    # Drop trailing completely empty columns
    last_non_empty = labels_series.replace("", pd.NA).last_valid_index()
    if last_non_empty is not None:
        data = data.iloc[:, : last_non_empty + 1]
        labels_series = labels_series.iloc[: last_non_empty + 1]
    data.columns = [str(c).strip() for c in labels_series.values]
    data = data.reset_index(drop=True)
    return data


def parse_project_code_from_filename(name: str) -> str:
    m = re.search(r"\b(TA|TB)\s*[- ]?\s*(\d{3,4})\b", name.upper())
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return re.sub(r"\s+", " ", str(name)).strip()


def normalize_stringing_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Rename known stringing columns to snake_case and report presence.

    Parameters
    - df: Input DataFrame containing raw stringing columns.

    Returns
    - (normalized_df, report):
        - normalized_df: a shallow copy of df with columns renamed where
          exact matches were found; other columns are preserved as-is.
        - report: dict with keys:
            - normalized_columns_ok: bool (True if all required found)
            - present: list[str] of headers found from the required set
            - missing: list[str] of required headers not found
            - applied_map: dict[str, str] of column renames actually applied
    """

    required_headers: List[str] = list(_STRINGING_COLUMN_MAP.keys())
    # Exact-match presence
    present: List[str] = [name for name in required_headers if name in df.columns]
    missing: List[str] = [name for name in required_headers if name not in df.columns]

    # Build tolerant rename map using canonical keys as fallback
    applied_map: Dict[str, str] = {}
    # First, capture exact matches
    for name in present:
        applied_map[name] = _STRINGING_COLUMN_MAP[name]
    # Then, try canonical-key matches for the rest
    if missing:
        expected_by_key = {_canon_key(k): v for k, v in _STRINGING_COLUMN_MAP.items()}
        for col in df.columns:
            if col in applied_map:
                continue
            key = _canon_key(col)
            if key in expected_by_key:
                applied_map[col] = expected_by_key[key]
        # recompute presence/missing after tolerant mapping
        present = [orig for orig in required_headers if orig in df.columns or _canon_key(orig) in {_canon_key(c): c for c in df.columns}]
        missing = [name for name in required_headers if name not in present]

    normalized = df.rename(columns=applied_map).copy()

    report: Dict[str, object] = {
        "normalized_columns_ok": len(missing) == 0,
        "present": present,
        "missing": missing,
        "applied_map": applied_map,
    }
    return normalized, report


def _to_datetime_normalize(value: object) -> pd.Timestamp | None:
    """Parse a single value to a normalized Timestamp or None if invalid.

    Mirrors erection start/end parsing semantics: pandas to_datetime with
    errors='coerce' and normalization to midnight; returns None on failure.
    """
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _is_filled(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null"}


def summarize_date_parsing(df: pd.DataFrame) -> Dict[str, object]:
    """Compute date parsing metrics for Stringing without expanding rows.

    Uses the same parse semantics as erection start/end.

    Returns a report dict including:
    - po_start_date_parsed_count
    - fs_complete_date_parsed_count
    - invalid_date_rows: rows with any filled date value that failed to parse
    - date_columns_present: mapping of column->bool
    """
    normalized, _ = normalize_stringing_columns(df)

    po_col = "po_start_date"
    fs_col = "fs_complete_date"

    present_po = po_col in normalized.columns
    present_fs = fs_col in normalized.columns

    po_series = normalized[po_col] if present_po else pd.Series([], dtype=object)
    fs_series = normalized[fs_col] if present_fs else pd.Series([], dtype=object)

    # Determine which entries are filled (user-provided) vs. empty
    po_filled = po_series.map(_is_filled) if present_po else pd.Series([], dtype=bool)
    fs_filled = fs_series.map(_is_filled) if present_fs else pd.Series([], dtype=bool)

    # Parse using the same logic (coerce + normalize)
    po_parsed = po_series.map(_to_datetime_normalize) if present_po else pd.Series([], dtype="datetime64[ns]")
    fs_parsed = fs_series.map(_to_datetime_normalize) if present_fs else pd.Series([], dtype="datetime64[ns]")

    po_ok = po_parsed.notna() if present_po else pd.Series([], dtype=bool)
    fs_ok = fs_parsed.notna() if present_fs else pd.Series([], dtype=bool)

    po_count = int(po_ok.sum()) if present_po else 0
    fs_count = int(fs_ok.sum()) if present_fs else 0

    # Invalid rows: had a value but failed to parse for any of the tracked columns
    po_invalid = (po_filled & ~po_ok) if present_po else pd.Series([], dtype=bool)
    fs_invalid = (fs_filled & ~fs_ok) if present_fs else pd.Series([], dtype=bool)
    # Align indices if both present; if only one present, use that
    if present_po and present_fs:
        invalid_any = po_invalid.reindex(normalized.index, fill_value=False) | fs_invalid.reindex(normalized.index, fill_value=False)
    elif present_po:
        invalid_any = po_invalid
    elif present_fs:
        invalid_any = fs_invalid
    else:
        invalid_any = pd.Series([], dtype=bool)

    report: Dict[str, object] = {
        "po_start_date_parsed_count": po_count,
        "fs_complete_date_parsed_count": fs_count,
        "invalid_date_rows": int(invalid_any.sum()) if len(invalid_any) else 0,
        "date_columns_present": {po_col: bool(present_po), fs_col: bool(present_fs)},
    }
    return report


def _pick_project_column(df: pd.DataFrame) -> str | None:
    """Return a column name to use for project if available.

    Prefers common variants like 'Project Name' or 'project_name'. If not found,
    also accepts project code style columns (e.g. 'Project Code', 'project_code').
    As a last resort, chooses any column whose normalized name contains
    'project' and either 'name' or 'code'. Returns None if nothing suitable.
    """
    # Exact/common candidates first
    candidates = [
        "Project Name",
        "project_name",
        "project",
        "Project",
        "projectName",
        "ProjectName",
        # Code-oriented variants
        "Project Code",
        "project_code",
        "ProjectCode",
        "projectcode",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lowered = {str(c).strip().lower(): c for c in df.columns}
    if "project name" in lowered:
        return lowered["project name"]
    if "project code" in lowered:
        return lowered["project code"]

    # Fuzzy: any header containing both tokens: project + (name|code)
    for key, original in lowered.items():
        if "project" in key and ("name" in key or "code" in key):
            return original
    return None


def expand_stringing_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Expand stringing records to per-day rows using PO start to F/S complete.

    Rules align with erection expansion for date range inclusivity:
    - Inclusive of both endpoints: [po_start_date, fs_complete_date]
    - Rows with missing start/end or non-positive durations are skipped.

    Output columns (if available in inputs):
    - project, gang_name, date, month, from_ap, to_ap, method,
      section_readiness, po_id, fs_start_date, fs_complete_date,
      status, length_km, row_id.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "project",
                "gang_name",
                "date",
                "month",
                "from_ap",
                "to_ap",
                "method",
                "section_readiness",
                "po_id",
                "fs_start_date",
                "fs_complete_date",
                "status",
                "length_km",
                "daily_km",
                "row_id",
            ]
        )

    # Normalize column names first (map-only)
    normalized, _ = normalize_stringing_columns(df)
    # Add length units (meters -> km) while preserving meters
    normalized, _length_metrics = add_length_units(normalized)

    # Determine project column (optional)
    project_col = _pick_project_column(df) or _pick_project_column(normalized)

    # Parse dates with the same helper semantics used elsewhere
    po_col = "po_start_date"
    end_col = "fs_complete_date"
    # Build working view to avoid pandas chained assignment traps
    work = normalized.copy()

    if po_col not in work.columns or end_col not in work.columns:
        # Required dates missing; return empty with schema
        return pd.DataFrame(
            columns=[
                "project",
                "gang_name",
                "date",
                "month",
                "from_ap",
                "to_ap",
                "method",
                "section_readiness",
                "po_id",
                "fs_start_date",
                "fs_complete_date",
                "status",
                "length_km",
                "daily_km",
                "row_id",
            ]
        )

    work[po_col] = work[po_col].map(_to_datetime_normalize)
    work[end_col] = work[end_col].map(_to_datetime_normalize)

    # Validity: both dates present, duration positive (inclusive range >= 1 day)
    missing_dt = work[po_col].isna() | work[end_col].isna()
    duration_days = (work[end_col] - work[po_col]).dt.days + 1
    non_positive = (~missing_dt) & (duration_days <= 0)
    valid_mask = (~missing_dt) & (~non_positive)
    valid = work.loc[valid_mask].copy()

    # Evenly distribute total section length across the active days
    # Guard against divide-by-zero (already filtered non_positive above)
    if "length_km" in valid.columns:
        valid["_duration_days"] = duration_days.loc[valid.index].astype(float)
        valid["daily_km"] = (
            pd.to_numeric(valid["length_km"], errors="coerce")
            .astype(float)
            .div(valid["_duration_days"].where(valid["_duration_days"] > 0, np.nan))
        )
    else:
        valid["daily_km"] = np.nan

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "project",
                "gang_name",
                "date",
                "month",
                "from_ap",
                "to_ap",
                "method",
                "section_readiness",
                "po_id",
                "fs_start_date",
                "fs_complete_date",
                "status",
                "length_km",
                "row_id",
            ]
        )

    # Ensure expected columns exist to avoid KeyErrors on selection
    for col in [
        "gang_name",
        "from_ap",
        "to_ap",
        "method",
        "section_readiness",
        "po",
        "status",
        "length_km",
    ]:
        if col not in valid.columns:
            valid[col] = pd.NA

    rows: List[Dict[str, object]] = []
    for _, r in valid.iterrows():
        start: pd.Timestamp = r[po_col]
        end: pd.Timestamp = r[end_col]
        # Daily inclusive range
        for d in pd.date_range(start, end, freq="D"):
            project_val = r[project_col] if project_col and project_col in valid.columns else pd.NA
            date_norm = d.normalize()
            month_ts = date_norm.to_period("M").to_timestamp()
            rows.append(
                {
                    "project": project_val,
                    "gang_name": r["gang_name"],
                    "date": date_norm,
                    "month": month_ts,
                    "from_ap": r["from_ap"],
                    "to_ap": r["to_ap"],
                    "method": r["method"],
                    "section_readiness": r["section_readiness"],
                    "po_id": r["po"],
                    "fs_start_date": r.get("fs_starting_date", pd.NA),
                    "fs_complete_date": r[end_col],
                    "status": r["status"],
                    "length_km": r["length_km"],
                    "daily_km": r.get("daily_km", np.nan),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "project",
                "gang_name",
                "date",
                "month",
                "from_ap",
                "to_ap",
                "method",
                "section_readiness",
                "po_id",
                "fs_start_date",
                "fs_complete_date",
                "status",
                "length_km",
                "row_id",
            ]
        )

    out = pd.DataFrame(rows)

    # Stable unique row id per (project, gang, date, from/to, po)
    def _mk_row_id(row: pd.Series) -> str:
        parts = [
            str(row.get("project", "")),
            str(row.get("gang_name", "")),
            str(pd.Timestamp(row.get("date")).date()),
            str(row.get("from_ap", "")),
            str(row.get("to_ap", "")),
            str(row.get("po_id", "")),
        ]
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return f"stringing:{digest[:16]}"

    out["row_id"] = out.apply(_mk_row_id, axis=1)

    # Order columns to match expected schema
    wanted = [
        "project",
        "gang_name",
        "date",
        "month",
        "from_ap",
        "to_ap",
        "method",
        "section_readiness",
        "po_id",
        "fs_start_date",
        "fs_complete_date",
        "status",
        "length_km",
        "daily_km",
        "row_id",
    ]
    # Include only those that exist, then add missing as NA for predictability
    for c in wanted:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[wanted]
    return out


def add_length_units(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Add ``length_km`` derived from meters and compute sanity metrics.

    Expects the input DataFrame to have the normalized column ``length_m``.
    If present, attempts to coerce it to numeric meters, derive kilometers,
    and compute simple sanity metrics for health logging.

    Returns
    - (df_out, metrics):
        - df_out: copy of df with numeric ``length_m`` (if present) and
          a new ``length_km`` column.
        - metrics: dict with keys
            - total_length_km: float
            - min_length_km: float
            - max_length_km: float
    """
    if df is None or df.empty:
        return (df.copy() if df is not None else pd.DataFrame()), {
            "total_length_km": 0.0,
            "min_length_km": 0.0,
            "max_length_km": 0.0,
        }

    out = df.copy()
    if "length_m" not in out.columns:
        return out, {
            "total_length_km": 0.0,
            "min_length_km": 0.0,
            "max_length_km": 0.0,
        }

    meters = pd.to_numeric(out["length_m"], errors="coerce")
    out["length_m"] = meters
    out["length_km"] = meters / 1000.0

    km = out["length_km"].dropna()
    total_km = float(km.sum()) if len(km) else 0.0
    min_km = float(km.min()) if len(km) else 0.0
    max_km = float(km.max()) if len(km) else 0.0

    metrics: Dict[str, float] = {
        "total_length_km": total_km,
        "min_length_km": min_km,
        "max_length_km": max_km,
    }
    return out, metrics
