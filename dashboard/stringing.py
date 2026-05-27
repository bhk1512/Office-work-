"""Stringing dataset utilities: column normalization (map only).

This module provides a light-weight mapping from the exact spreadsheet
headers to normalized snake_case field names. It does not perform any
date expansion or type coercion — only column renaming plus a presence
report for required inputs.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Mapping
import re
import hashlib

import pandas as pd
import numpy as np
from .plan_utils import normalize_lower, compact_project_key
from .project_identity import parse_project_identity_from_filename as _parse_project_identity


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
_STRINGING_OPTIONAL_HEADERS: set[str] = {"Status"}
_STRINGING_HEADER_ALIASES: Dict[str, str] = {
    "from": "From AP",
    "to": "To AP",
    "typeofsections": "Method",
    "sectiontype": "Method",
    "payingoutrsag": "P/O",
    "payingout": "P/O",
    "nameofgang": "Gang Name",
    "gang": "Gang Name",
    "sectionlength": "Length",
    "sectionlengthm": "Length",
    "spanm": "Length",
    "span": "Length",
    "lengthm": "Length",
}
_STRINGING_CRITICAL_HEADERS: tuple[str, ...] = (
    "From AP",
    "To AP",
    "P/O Starting Date",
    "F/S/ Completion Date",
)
_STRINGING_LENGTH_SOURCE_HEADERS: tuple[str, ...] = ("Length", "P/O")


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


def _coerce_numeric_cell(value: object) -> Optional[int]:
    """Best-effort coercion of a spreadsheet cell to an integer."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(round(value))
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return int(round(float(match.group())))
    except (TypeError, ValueError):
        return None


def extract_stringing_number_of_tse(path: str | bytes | "pathlike", sheet_name: str, search_rows: int = 20) -> Optional[int]:
    """
    Locate the \"Number of TSE\" value from the top rows of a Stringing Compiled sheet.

    The workbook typically places this label near the first row with the value in the
    adjacent column. We scan a handful of rows before the main header and return the
    first numeric value found next to the label.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=search_rows)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    exact_targets = {"numberoftse", "numberoftses", "nooftse", "nooftses", "numoftse", "numoftses"}
    arr = df.to_numpy(object)
    for row in arr:
        for idx, cell in enumerate(row):
            key = _canon_key(cell)
            if not key:
                continue
            looks_like_tse_label = key in exact_targets or (
                "tse" in key and any(token in key for token in ("number", "num", "noof", "count"))
            )
            if not looks_like_tse_label:
                continue

            # Most files keep value in adjacent cell, but some embed it in the same cell
            # (e.g., "Number of TSE: 4") or place it on the left.
            candidates = []
            if idx + 1 < len(row):
                candidates.append(row[idx + 1])
            if idx - 1 >= 0:
                candidates.append(row[idx - 1])
            candidates.append(cell)

            for candidate in candidates:
                coerced = _coerce_numeric_cell(candidate)
                if coerced is not None:
                    return coerced
    return None


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
    identity = _parse_project_identity(str(name))
    project_code = identity.get("project_code", "")
    if project_code:
        return compact_project_key(project_code).upper() if re.search(r"\bT[A-Z]\b", project_code) else project_code
    return re.sub(r"\s+", " ", str(name)).strip()


def build_tse_lookup_from_df(df: pd.DataFrame | None) -> tuple[dict[str, int], dict[str, str]]:
    """Return canonical + alias maps for TSE project detection."""

    canonical: dict[str, int] = {}
    aliases: dict[str, str] = {}
    if not isinstance(df, pd.DataFrame) or df.empty or "number_of_tse" not in df.columns:
        return canonical, aliases

    project_col = None
    for candidate in ("project_name", "project", "Project Name", "Project"):
        if candidate in df.columns:
            project_col = candidate
            break
    if project_col is None:
        return canonical, aliases

    work = df[[project_col, "number_of_tse"]].copy()
    work[project_col] = work[project_col].astype(str).str.strip()
    work["number_of_tse"] = pd.to_numeric(work["number_of_tse"], errors="coerce")
    work = work.dropna(subset=[project_col, "number_of_tse"])
    if work.empty:
        return canonical, aliases

    def _project_code_token(text: str) -> str | None:
        match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{3,4})\b", str(text).upper())
        if not match:
            return None
        return f"{match.group(1)}{match.group(2)}"

    grouped = work.groupby(work[project_col])["number_of_tse"].max()
    for project, raw_value in grouped.items():
        try:
            value = int(round(float(raw_value)))
        except (TypeError, ValueError):
            continue
        if pd.isna(value):
            continue
        canonical_key = normalize_lower(project)
        if not canonical_key:
            continue
        if canonical_key not in canonical:
            canonical[canonical_key] = value
        compact_key = compact_project_key(project)
        if compact_key:
            aliases.setdefault(compact_key, canonical_key)
        code_token = _project_code_token(project)
        if code_token:
            aliases.setdefault(compact_project_key(code_token), canonical_key)
    return canonical, aliases


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

    tracked_headers: List[str] = list(_STRINGING_COLUMN_MAP.keys())
    required_headers: List[str] = [
        name for name in tracked_headers if name not in _STRINGING_OPTIONAL_HEADERS
    ]

    expected_by_key = {_canon_key(k): v for k, v in _STRINGING_COLUMN_MAP.items()}
    alias_by_key = {_canon_key(k): v for k, v in _STRINGING_HEADER_ALIASES.items()}
    recognized_keys: set[str] = set()
    applied_map: Dict[str, str] = {}

    for col in df.columns:
        if col in _STRINGING_COLUMN_MAP:
            applied_map[col] = _STRINGING_COLUMN_MAP[col]
            recognized_keys.add(_canon_key(col))
            continue
        key = _canon_key(col)
        if key in expected_by_key and col not in applied_map:
            applied_map[col] = expected_by_key[key]
            recognized_keys.add(key)
            continue
        alias_target = alias_by_key.get(key)
        if alias_target and col not in applied_map:
            mapped = _STRINGING_COLUMN_MAP.get(alias_target)
            if mapped:
                applied_map[col] = mapped
                recognized_keys.add(_canon_key(alias_target))

    present: List[str] = [
        header for header in tracked_headers if _canon_key(header) in recognized_keys
    ]
    missing: List[str] = [
        header for header in required_headers if _canon_key(header) not in recognized_keys
    ]

    normalized = df.rename(columns=applied_map).copy()
    # Template/header aliases can map multiple source columns to one canonical
    # name (for example, two variants of length). Collapse duplicates so
    # downstream concatenation and reindexing remain stable.
    normalized = _collapse_duplicate_columns(normalized)

    report: Dict[str, object] = {
        "normalized_columns_ok": len(missing) == 0,
        "present": present,
        "missing": missing,
        "applied_map": applied_map,
    }
    return normalized, report


def classify_stringing_missing_headers(report: Mapping[str, object] | None) -> Dict[str, object]:
    """Split missing headers into critical vs. non-critical groups."""

    missing = [str(v).strip() for v in list((report or {}).get("missing", []) or []) if str(v).strip()]
    present = {str(v).strip() for v in list((report or {}).get("present", []) or []) if str(v).strip()}
    missing_set = set(missing)

    critical_missing: list[str] = [header for header in _STRINGING_CRITICAL_HEADERS if header in missing_set]
    has_length_source = any(header in present for header in _STRINGING_LENGTH_SOURCE_HEADERS)
    if not has_length_source:
        critical_missing.append("Length")

    non_critical_missing: list[str] = [
        header
        for header in missing
        if header not in set(_STRINGING_CRITICAL_HEADERS) and header not in set(_STRINGING_LENGTH_SOURCE_HEADERS)
    ]
    return {
        "critical_missing": critical_missing,
        "non_critical_missing": non_critical_missing,
        "has_length_source": has_length_source,
        "is_critical_complete": len(critical_missing) == 0,
    }


def _to_datetime_normalize(value: object) -> pd.Timestamp | None:
    """Parse a single value to a normalized Timestamp or None if invalid.

    Mirrors erection start/end parsing semantics: pandas to_datetime with
    errors='coerce' and normalization to midnight; returns None on failure.
    """
    if value is None:
        return None

    parsed = None
    # Excel serials often arrive as numbers (or numeric strings). Convert those
    # via the Excel epoch; fall back to pandas' default otherwise.
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            parsed = pd.NaT
        else:
            parsed = pd.to_datetime(value, errors="coerce", unit="D", origin="1899-12-30")
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            parsed = pd.NaT
        else:
            if re.match(r"^\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*$", text):
                parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
            else:
                parsed = pd.to_datetime(text, errors="coerce")
            if pd.isna(parsed):
                numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
                if pd.notna(numeric):
                    parsed = pd.to_datetime(numeric, errors="coerce", unit="D", origin="1899-12-30")
    else:
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


def _empty_stage_frame() -> pd.DataFrame:
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


def _is_blank_like(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def _blank_like_mask(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series([], dtype=bool)
    mask = series.isna()
    try:
        text = series.astype("string").fillna("").str.strip().str.lower()
        mask = mask | text.isin({"", "nan", "none", "null"})
    except Exception:
        pass
    return mask.fillna(True)


def _coalesce_duplicate_named_columns(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Merge duplicate columns with the same name by taking first non-blank value."""
    if df is None or df.empty:
        return df
    out = df.copy()
    matches = [idx for idx, col in enumerate(out.columns) if col == column_name]
    if len(matches) <= 1:
        return out

    merged = pd.Series(pd.NA, index=out.index, dtype="object")
    for idx in matches:
        series = out.iloc[:, idx]
        candidate = series.mask(_blank_like_mask(series), pd.NA)
        merged = merged.where(~_blank_like_mask(merged), candidate)

    first_idx = matches[0]
    keep_mask = [idx == first_idx or col != column_name for idx, col in enumerate(out.columns)]
    out = out.iloc[:, keep_mask].copy()
    out[column_name] = merged.astype("object").to_numpy()
    return out


def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse known duplicate mapped columns and then keep first of any remaining duplicates."""
    if df is None or df.empty or df.columns.is_unique:
        return df
    out = df.copy()
    for name in (
        "length_m",
        "po_start_date",
        "po_completion_date",
        "fs_starting_date",
        "fs_complete_date",
        "po",
        "method",
        "gang_name",
    ):
        out = _coalesce_duplicate_named_columns(out, name)
    if not out.columns.is_unique:
        out = out.loc[:, ~out.columns.duplicated(keep="first")].copy()
    return out


_LOCATION_RE = re.compile(r"^\s*(\d+)([A-Za-z]+)?(?:\s*/\s*(\d+))?\s*$")
_AP_PREFIX_RE = re.compile(r"^\s*ap[\s\-_/]*", flags=re.IGNORECASE)
_GANTRY_RE = re.compile(r"\b(gantry|gty)\b", flags=re.IGNORECASE)


def _normalize_location_text(value: object) -> str:
    text = "" if value is None else str(value).replace("\u00a0", " ").strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    return text


def _strip_ap_prefix(value: object) -> str:
    return _AP_PREFIX_RE.sub("", _normalize_location_text(value)).strip()


def _is_gantry_label(value: object) -> bool:
    text = _normalize_location_text(value)
    if not text:
        return False
    cleaned = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    return bool(_GANTRY_RE.search(cleaned))


def _letter_rank(value: str) -> int:
    rank = 0
    for ch in value.upper():
        if "A" <= ch <= "Z":
            rank = (rank * 26) + (ord(ch) - ord("A") + 1)
    return rank


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
    return (main * 1_000_000) + (_letter_rank(letters) * 1_000) + sub


def _resolve_project_key_norm(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("project_code")
    if source is None:
        source = frame.get("project_name")
    if source is None:
        source = frame.get("project")
    if source is None:
        source = pd.Series("", index=frame.index)
    source = source.fillna("").astype(str).str.strip()
    fallback = frame.get("project_name", frame.get("project", pd.Series("", index=frame.index)))
    fallback = fallback.fillna("").astype(str).str.strip()
    resolved = source.where(source.astype(bool), fallback)
    return resolved.map(compact_project_key)


def _build_erection_location_map(erection_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not isinstance(erection_daily, pd.DataFrame) or erection_daily.empty:
        return {}
    work = erection_daily.copy()
    if "location_no" not in work.columns:
        return {}
    completion = work.get("completion_date")
    if completion is None:
        completion = work.get("date")
    if completion is None:
        return {}
    work["completion_date"] = pd.to_datetime(completion, errors="coerce").dt.normalize()
    work["location_no_norm"] = work["location_no"].map(_normalize_location_text)
    work["loc_order"] = work["location_no_norm"].map(_location_order_key)
    work["project_key_norm"] = _resolve_project_key_norm(work)
    work = work[
        work["project_key_norm"].astype(bool)
        & work["location_no_norm"].astype(bool)
        & work["loc_order"].notna()
        & work["completion_date"].notna()
    ]
    if work.empty:
        return {}

    work = (
        work.sort_values(["project_key_norm", "loc_order", "completion_date"])
        .drop_duplicates(subset=["project_key_norm", "location_no_norm", "loc_order"], keep="last")
    )
    location_map: dict[str, pd.DataFrame] = {}
    for project_key, group in work.groupby("project_key_norm"):
        location_map[str(project_key)] = group[["location_no_norm", "loc_order", "completion_date"]].copy()
    return location_map


def infer_missing_methods_from_erection(
    compiled_df: pd.DataFrame,
    erection_daily: pd.DataFrame | None,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Fill missing method values using erection span depth assumptions."""

    if compiled_df is None or compiled_df.empty:
        return compiled_df.copy() if isinstance(compiled_df, pd.DataFrame) else pd.DataFrame(), {
            "method_inferred_rows": 0,
            "method_inferred_manual_rows": 0,
            "method_inferred_tse_rows": 0,
            "method_inferred_unresolved_rows": 0,
        }

    work = compiled_df.copy()
    for col in ("from_ap", "to_ap", "method"):
        if col not in work.columns:
            work[col] = ""
    work["method"] = work["method"].fillna("").astype(str).str.strip()
    missing_method = work["method"].str.lower().isin({"", "nan", "none", "null"})
    work["method_inferred"] = False
    work["method_inference_reason"] = ""
    work["erection_locations_for_method"] = pd.NA
    if not missing_method.any():
        return work, {
            "method_inferred_rows": 0,
            "method_inferred_manual_rows": 0,
            "method_inferred_tse_rows": 0,
            "method_inferred_unresolved_rows": 0,
        }

    location_map = _build_erection_location_map(erection_daily if isinstance(erection_daily, pd.DataFrame) else pd.DataFrame())
    project_key_norm = _resolve_project_key_norm(work)

    inferred_rows = 0
    manual_rows = 0
    tse_rows = 0
    unresolved_rows = 0

    for idx in work.index[missing_method]:
        project_key = str(project_key_norm.loc[idx] or "").strip()
        from_ap = work.at[idx, "from_ap"]
        to_ap = work.at[idx, "to_ap"]
        project_df = location_map.get(project_key)
        method_value = "tse"
        reason = "SPAN_UNRESOLVED_TSE_FALLBACK"
        loc_count = 0

        if project_df is not None and not project_df.empty:
            from_is_gantry = _is_gantry_label(from_ap)
            to_is_gantry = _is_gantry_label(to_ap)
            from_order = _location_order_key(from_ap)
            to_order = _location_order_key(to_ap)
            try:
                if from_is_gantry and to_is_gantry:
                    span_df = pd.DataFrame()
                elif to_is_gantry and from_order is not None:
                    lo = int(from_order)
                    hi = int(project_df["loc_order"].max())
                    span_df = project_df[(project_df["loc_order"] >= lo) & (project_df["loc_order"] <= hi)]
                elif from_is_gantry and to_order is not None:
                    lo = int(project_df["loc_order"].min())
                    hi = int(to_order)
                    span_df = project_df[(project_df["loc_order"] >= lo) & (project_df["loc_order"] <= hi)]
                elif from_order is not None and to_order is not None:
                    lo = min(int(from_order), int(to_order))
                    hi = max(int(from_order), int(to_order))
                    span_df = project_df[(project_df["loc_order"] >= lo) & (project_df["loc_order"] <= hi)]
                else:
                    span_df = pd.DataFrame()
            except Exception:
                span_df = pd.DataFrame()

            if not span_df.empty:
                loc_count = int(span_df["location_no_norm"].nunique())
                if loc_count <= 2:
                    method_value = "manual"
                    reason = "NO_INTERMEDIATE_TOWERS_MANUAL"
                    manual_rows += 1
                else:
                    method_value = "tse"
                    reason = "INTERMEDIATE_TOWERS_TSE"
                    tse_rows += 1
            else:
                unresolved_rows += 1
                tse_rows += 1
        else:
            unresolved_rows += 1
            tse_rows += 1

        work.at[idx, "method"] = method_value
        work.at[idx, "method_inferred"] = True
        work.at[idx, "method_inference_reason"] = reason
        work.at[idx, "erection_locations_for_method"] = int(loc_count)
        inferred_rows += 1

    return work, {
        "method_inferred_rows": int(inferred_rows),
        "method_inferred_manual_rows": int(manual_rows),
        "method_inferred_tse_rows": int(tse_rows),
        "method_inferred_unresolved_rows": int(unresolved_rows),
    }


def _expand_stringing_stage_to_daily(
    df: pd.DataFrame,
    *,
    start_column: str,
    end_column: str,
    output_end_column: str,
    value_column: str = "length_km",
) -> pd.DataFrame:
    """Internal helper to expand rows between a start column and a completion column."""
    if df is None or df.empty:
        return _empty_stage_frame()

    normalized, _ = normalize_stringing_columns(df)
    normalized = _collapse_duplicate_columns(normalized)
    normalized, _length_metrics = add_length_units(normalized)
    project_col = _pick_project_column(df) or _pick_project_column(normalized)

    start_col = start_column
    end_col = end_column
    work = normalized.copy()

    if start_col not in work.columns or end_col not in work.columns:
        return _empty_stage_frame()

    work[start_col] = work[start_col].map(_to_datetime_normalize)
    work[end_col] = work[end_col].map(_to_datetime_normalize)
    if "po" in work.columns and "po_km" not in work.columns:
        po_values = pd.to_numeric(work["po"], errors="coerce")
        unit_series = work.get("length_unit")
        if isinstance(unit_series, pd.Series):
            is_km = unit_series.astype(str).str.lower().eq("km")
            work["po_km"] = po_values.where(is_km, po_values / 1000.0)
        else:
            work["po_km"] = po_values / 1000.0

    missing_dt = work[start_col].isna() | work[end_col].isna()
    duration_days = (work[end_col] - work[start_col]).dt.days + 1
    non_positive = (~missing_dt) & (duration_days <= 0)
    valid_mask = (~missing_dt) & (~non_positive)
    valid = work.loc[valid_mask].copy()

    metric_column = value_column if value_column in valid.columns else None
    if metric_column is None and "length_km" in valid.columns:
        metric_column = "length_km"

    valid["_duration_days"] = duration_days.loc[valid.index].astype(float)
    if metric_column:
        metric_values = pd.to_numeric(valid[metric_column], errors="coerce")
    else:
        metric_values = pd.Series(np.nan, index=valid.index)

    # Prefer the requested value_column, then fall back:
    # - to length_km when the primary metric is not already length-based
    # - to po_km when the primary metric is not already PO-based
    if value_column != "length_km":
        metric_values = metric_values.fillna(pd.to_numeric(valid.get("length_km"), errors="coerce"))
    if value_column != "po_km":
        metric_values = metric_values.fillna(pd.to_numeric(valid.get("po_km"), errors="coerce"))

    valid["_metric_km"] = metric_values
    valid["daily_km"] = valid["_metric_km"].div(valid["_duration_days"].where(valid["_duration_days"] > 0, np.nan))

    if valid.empty:
        return _empty_stage_frame()

    for col in [
        "gang_name",
        "from_ap",
        "to_ap",
        "method",
        "method_inferred",
        "method_inference_reason",
        "section_readiness",
        "po",
        "status",
        "length_km",
    ]:
        if col not in valid.columns:
            valid[col] = pd.NA

    rows: List[Dict[str, object]] = []
    for _, r in valid.iterrows():
        start: pd.Timestamp = r[start_col]
        end: pd.Timestamp = r[end_col]
        method_inferred_value = r.get("method_inferred", False)
        method_inferred = False if _is_blank_like(method_inferred_value) else bool(method_inferred_value)
        for d in pd.date_range(start, end, freq="D"):
            project_val = r[project_col] if project_col and project_col in valid.columns else pd.NA
            date_norm = d.normalize()
            month_ts = date_norm.to_period("M").to_timestamp()
            row = {
                "project": project_val,
                "gang_name": r["gang_name"],
                "date": date_norm,
                "month": month_ts,
                "from_ap": r["from_ap"],
                "to_ap": r["to_ap"],
                "method": r["method"],
                "method_inferred": method_inferred,
                "method_inference_reason": r.get("method_inference_reason", ""),
                "section_readiness": r["section_readiness"],
                "po_id": r["po"],
                "fs_start_date": r.get("fs_starting_date", r.get(start_col, pd.NA)),
                "fs_complete_date": r[end_col],
                "status": r["status"],
                "length_km": r["length_km"],
                "po_km": r.get("po_km", pd.NA),
                "daily_km": r.get("daily_km", np.nan),
            }
            if output_end_column and output_end_column != "fs_complete_date":
                row[output_end_column] = r[end_col]
            rows.append(row)

    if not rows:
        return _empty_stage_frame()

    out = pd.DataFrame(rows)

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

    wanted = [
        "project",
        "gang_name",
        "date",
        "month",
        "from_ap",
        "to_ap",
        "method",
        "method_inferred",
        "method_inference_reason",
        "section_readiness",
        "po_id",
        "fs_start_date",
        "fs_complete_date",
        "status",
        "length_km",
        "po_km",
        "daily_km",
        "row_id",
    ]
    if output_end_column and output_end_column not in wanted:
        insert_at = wanted.index("fs_complete_date") + 1
        wanted.insert(insert_at, output_end_column)
    for c in wanted:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[wanted]
    return out


def expand_stringing_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Expand stringing records to per-day rows using P/O start to F/S complete."""
    return _expand_stringing_stage_to_daily(
        df,
        start_column="po_start_date",
        end_column="fs_complete_date",
        output_end_column="fs_complete_date",
        value_column="length_km",
    )


def expand_stringing_to_daily_payout(df: pd.DataFrame) -> pd.DataFrame:
    """Expand stringing records between PO start and P/O completion dates."""
    return _expand_stringing_stage_to_daily(
        df,
        start_column="po_start_date",
        end_column="po_completion_date",
        output_end_column="po_completion_date",
        value_column="po_km",
    )


def expand_stringing_to_daily_fs(df: pd.DataFrame) -> pd.DataFrame:
    """Expand stringing records between F/S start and F/S completion dates."""
    return _expand_stringing_stage_to_daily(
        df,
        start_column="fs_starting_date",
        end_column="fs_complete_date",
        output_end_column="fs_complete_date",
        value_column="length_km",
    )



def _infer_length_unit(values: pd.Series) -> str:
    """Infer whether length values are in meters or kilometers.

    Heuristic:
    - If typical values are small (p90 <= 50 and max <= 100), assume KM.
    - Otherwise assume meters.
    """
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    if cleaned.empty:
        return "m"
    max_val = float(cleaned.max())
    p90 = float(cleaned.quantile(0.9))
    if max_val <= 100 and p90 <= 50:
        return "km"
    return "m"


def _summary_or_header_marker(value: object) -> bool:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return compact in {
        "from",
        "to",
        "fromap",
        "toap",
        "total",
        "grandtotal",
        "subtotal",
        "section",
        "length",
        "lengthinkm",
        "lengthinkms",
        "lengthinm",
        "lengthinmeters",
    }


def _row_level_length_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype="bool")
    if "from_ap" in frame.columns and "to_ap" in frame.columns:
        from_blank = _blank_like_mask(frame["from_ap"])
        to_blank = _blank_like_mask(frame["to_ap"])
        from_marker = frame["from_ap"].map(_summary_or_header_marker)
        to_marker = frame["to_ap"].map(_summary_or_header_marker)
        mask = ~(from_blank | to_blank | from_marker | to_marker)
    elif "to_ap" in frame.columns:
        to_blank = _blank_like_mask(frame["to_ap"])
        to_marker = frame["to_ap"].map(_summary_or_header_marker)
        mask = ~(to_blank | to_marker)
    elif "from_ap" in frame.columns:
        from_blank = _blank_like_mask(frame["from_ap"])
        from_marker = frame["from_ap"].map(_summary_or_header_marker)
        mask = ~(from_blank | from_marker)

    if "status" in frame.columns:
        status_marker = frame["status"].map(_summary_or_header_marker)
        mask = mask & ~status_marker
    return mask


def add_length_units(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Add ``length_km`` derived from length values and compute sanity metrics.

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
    out = _collapse_duplicate_columns(out)
    if "length_m" not in out.columns:
        return out, {
            "total_length_km": 0.0,
            "min_length_km": 0.0,
            "max_length_km": 0.0,
        }

    length_values = out["length_m"]
    if isinstance(length_values, pd.DataFrame):
        merged = pd.Series(pd.NA, index=out.index, dtype="object")
        for _, series in length_values.items():
            candidate = series.mask(_blank_like_mask(series), pd.NA)
            merged = merged.where(~_blank_like_mask(merged), candidate)
        length_values = merged
    raw_values = pd.to_numeric(length_values, errors="coerce")
    unit_series = pd.Series(index=out.index, dtype="object")

    group_key = None
    for candidate in ("source_file", "project_name", "project"):
        if candidate in out.columns:
            group_key = candidate
            break

    def _inference_sample(frame: pd.DataFrame) -> pd.Series:
        series_values = frame["length_m"]
        if isinstance(series_values, pd.DataFrame):
            merged_values = pd.Series(pd.NA, index=frame.index, dtype="object")
            for _, series in series_values.items():
                candidate = series.mask(_blank_like_mask(series), pd.NA)
                merged_values = merged_values.where(~_blank_like_mask(merged_values), candidate)
            series_values = merged_values
        numeric = pd.to_numeric(series_values, errors="coerce")
        row_level = _row_level_length_mask(frame)
        if bool(row_level.any()):
            filtered = numeric.where(row_level)
            if filtered.notna().any():
                return filtered
        return numeric

    if group_key:
        for key, group in out.groupby(group_key, dropna=False):
            unit = _infer_length_unit(_inference_sample(group))
            unit_series.loc[group.index] = unit
    else:
        unit = _infer_length_unit(_inference_sample(out))
        unit_series = pd.Series(unit, index=out.index, dtype="object")

    unit_series = unit_series.fillna("m")
    is_km = unit_series.eq("km")
    out["length_km"] = raw_values.where(is_km, raw_values / 1000.0)
    out["length_m"] = raw_values.where(~is_km, raw_values * 1000.0)
    out["length_unit"] = unit_series

    km = out["length_km"].where(_row_level_length_mask(out)).dropna()
    total_km = float(km.sum()) if len(km) else 0.0
    min_km = float(km.min()) if len(km) else 0.0
    max_km = float(km.max()) if len(km) else 0.0

    unit_label = "mixed"
    if unit_series.nunique(dropna=True) == 1:
        unit_label = str(unit_series.dropna().iloc[0])

    metrics: Dict[str, float] = {
        "total_length_km": total_km,
        "min_length_km": min_km,
        "max_length_km": max_km,
        "length_unit": unit_label,
    }
    return out, metrics
