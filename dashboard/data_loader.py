"""Data loading utilities for the productivity dashboard."""
from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

import re
import numpy as np
import json

from pathlib import Path

from .config import AppConfig
from .stringing import (
    expand_stringing_to_daily,
    normalize_stringing_columns,
    summarize_date_parsing,
    add_length_units,
    read_stringing_sheet_robust,
    find_stringing_header_row,
    parse_project_code_from_filename,
    extract_stringing_number_of_tse,
)
from .plan_utils import infer_project_hint, prepare_stringing_plan_frame
from .project_identity import (
    build_project_display,
    build_project_scope_key,
    parse_sheet_line_entries,
    normalize_line_name,
    parse_project_identity_from_filename,
)

CONFIG = AppConfig()

CACHE_TTL_SECONDS = CONFIG.cache_ttl_seconds
CACHE_MAXSIZE = CONFIG.cache_maxsize

LOGGER = logging.getLogger(__name__)


_OPENPYXL_PRINT_TITLES_PATCHED = False


def _patch_openpyxl_invalid_print_titles() -> None:
    """Work around workbooks whose Print Titles defined names evaluate to #N/A.

    Such files cause openpyxl to raise ValueError during load. We intercept
    the parser once and fall back to an empty PrintTitles instance instead
    of failing the entire read.
    """
    global _OPENPYXL_PRINT_TITLES_PATCHED
    if _OPENPYXL_PRINT_TITLES_PATCHED:
        return
    try:
        from openpyxl.worksheet import print_settings as _ps
    except Exception:
        return
    descriptor = _ps.PrintTitles.__dict__.get("from_string")
    if descriptor is None:
        return
    original = descriptor.__func__ if hasattr(descriptor, "__func__") else descriptor
    if not callable(original):
        return

    def _safe_from_string(cls, value):
        try:
            return original(cls, value)
        except ValueError:
            LOGGER.warning(
                "Openpyxl: ignoring invalid Print Titles definition: %s",
                value,
            )
            return cls()

    _ps.PrintTitles.from_string = classmethod(_safe_from_string)
    _OPENPYXL_PRINT_TITLES_PATCHED = True


_patch_openpyxl_invalid_print_titles()

PROJECT_BASELINES_SHEET = "ProjectBaselines"
PROJECT_BASELINES_MONTHLY_SHEET = "ProjectBaselinesMonthly"

PARQUET_SUFFIXES: tuple[str, ...] = (".parquet", ".parq", ".pq")

_PROJECT_BASELINE_OVERALL: dict[str, float] = {}
_PROJECT_BASELINE_MONTHLY: dict[str, dict[pd.Timestamp, float]] = {}
_PROJECT_BASELINE_SOURCE: Path | None = None
_PROJECT_RE = re.compile(r'\b(TA|TB)\s*[-_ ]?\s*(\d{3,4})\b', re.I)

# ===== STRINGING: single-folder outputs, scan-all-each-run ====================
def _repo_root_from(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "pipeline_config.json").exists():
            return p
    # Fallback: if not found, use two levels up from this file
    return cur

def _load_pipeline_cfg(repo_root: Path) -> dict:
    cfg_path = repo_root / "pipeline_config.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _resolve_stringing_raw_root(data_path_hint: Path | str) -> Path:
    """
    Find RAW DPRs folder. Priority:
    1) pipeline_config.json -> "input_directory" (relative to repo root)
    2) <repo>/Raw Data/DPRs
    3) <hint>/Raw Data/DPRs
    """
    repo_root = _repo_root_from(Path(__file__))
    cfg = _load_pipeline_cfg(repo_root)
    input_dir = cfg.get("input_directory")  # e.g., "Raw Data/DPRs"
    if input_dir:
        raw_root = (repo_root / input_dir).resolve()
        if raw_root.exists():
            print(f"[Stringing] RAW root from pipeline_config.json: {raw_root}")
            return raw_root
    cand1 = (repo_root / "Raw Data" / "DPRs").resolve()
    if cand1.exists():
        print(f"[Stringing] RAW root fallback <repo>/Raw Data/DPRs: {cand1}")
        return cand1
    cand2 = (Path(data_path_hint) / "Raw Data" / "DPRs").resolve()
    print(f"[Stringing] RAW root ultimate fallback: {cand2}")
    return cand2


def _resolve_stringing_microplan_root(data_path_hint: Path | str | None = None) -> Path:
    """
    Locate the Stringing Micro Plan root directory. Priority:
    1) pipeline_config.json -> "stringing_microplan_directory" (or "microplan_directory")
    2) <repo>/Raw Data/Micro Plans
    3) <hint>/Raw Data/Micro Plans
    """
    repo_root = _repo_root_from(Path(__file__))
    cfg = _load_pipeline_cfg(repo_root)

    def _resolve_candidate(path_like: str | Path | None) -> Path | None:
        if not path_like:
            return None
        candidate = Path(path_like)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        return candidate

    configured = cfg.get("stringing_microplan_directory") or cfg.get("microplan_directory")
    repo_default = (repo_root / "Raw Data" / "Micro Plans").resolve()
    hint_base = Path(data_path_hint) if data_path_hint else Path(".")
    hint_default = (hint_base / "Raw Data" / "Micro Plans").resolve()

    for label, candidate in (
        ("pipeline_config", _resolve_candidate(configured)),
        ("<repo>/Raw Data/Micro Plans", repo_default),
        ("hint/Raw Data/Micro Plans", hint_default),
    ):
        if candidate and candidate.exists():
            print(f"[Stringing] Micro plan root ({label}): {candidate}")
            return candidate

    print(f"[Stringing] Micro plan fallback: {hint_default}")
    return hint_default


def _norm_sheet(s: str) -> str:
    """
    Normalize a sheet name by:
      - lowercasing
      - removing ALL non [a-z0-9] characters (spaces, dashes, underscores, dots, etc.)
    Examples:
      "Stringing Compiled"   -> "stringingcompiled"
      "STRINGING-COMPILED"   -> "stringingcompiled"
      "stringing_compiled."  -> "stringingcompiled"
    """
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _match_sheet_name(sheet_names: Iterable[str], desired_sheet: str) -> str | None:
    """
    Return the actual sheet name from the workbook whose normalized form
    exactly equals the normalized desired_sheet. No aliases.
    """
    want = _norm_sheet(desired_sheet)
    if not want:
        return None
    for s in sheet_names:
        if _norm_sheet(s) == want:
            return s  # return the exact name as present in the file
    return None

def _project_from_filename(name: str) -> str | None:
    """Parse TA/TB + digits from a filename -> 'TA 415' / 'TB 408'."""
    if not name:
        return None
    identity = parse_project_identity_from_filename(Path(name).name)
    project_code = identity.get("project_code", "")
    return project_code or None


def _apply_project_identity_columns(
    df: pd.DataFrame,
    source_path: Path | str | None,
    *,
    project_code: str = "",
    fallback_name: str = "",
    line_name_override: str = "",
    source_sheet: str = "",
    project_name_column: str = "project_name",
    project_key_column: str = "project_key",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    working = df.copy()
    source_name = Path(source_path).name if source_path else ""
    identity = parse_project_identity_from_filename(source_name) if source_name else {}
    forced_line_name = normalize_line_name(line_name_override)
    force_line_identity = bool(forced_line_name)
    line_name = forced_line_name or normalize_line_name(identity.get("line_name", ""))
    base_code = str(project_code or identity.get("project_code", "") or "").strip()
    visible_name = build_project_display(base_code, line_name, fallback_name or identity.get("project_display", ""))
    scope_key = build_project_scope_key(base_code, line_name, fallback_name or identity.get("project_display", ""))

    if "line_name" not in working.columns or force_line_identity:
        working["line_name"] = line_name
    else:
        series = working["line_name"].fillna("").astype(str).map(normalize_line_name)
        working["line_name"] = series.where(series.astype(bool), line_name)

    if "project_code" not in working.columns:
        working["project_code"] = base_code
    else:
        series = working["project_code"].fillna("").astype(str).str.strip()
        working["project_code"] = series.where(series.astype(bool), base_code)

    if project_name_column not in working.columns or force_line_identity:
        working[project_name_column] = visible_name
    else:
        series = working[project_name_column].fillna("").astype(str).str.strip()
        working[project_name_column] = series.where(series.astype(bool), visible_name)
        if visible_name and line_name:
            working[project_name_column] = visible_name

    if "project_display" not in working.columns or force_line_identity:
        working["project_display"] = visible_name
    else:
        series = working["project_display"].fillna("").astype(str).str.strip()
        working["project_display"] = series.where(series.astype(bool), visible_name)
        if visible_name and line_name:
            working["project_display"] = visible_name

    if "project_scope_key" not in working.columns or force_line_identity:
        working["project_scope_key"] = scope_key
    else:
        series = working["project_scope_key"].fillna("").astype(str).str.strip()
        working["project_scope_key"] = series.where(series.astype(bool), scope_key)

    if project_key_column:
        if project_key_column not in working.columns:
            working[project_key_column] = base_code or visible_name
        else:
            series = working[project_key_column].fillna("").astype(str).str.strip()
            working[project_key_column] = series.where(series.astype(bool), base_code or visible_name)
    if source_sheet:
        if "source_sheet" not in working.columns or force_line_identity:
            working["source_sheet"] = source_sheet
        else:
            series = working["source_sheet"].fillna("").astype(str).str.strip()
            working["source_sheet"] = series.where(series.astype(bool), source_sheet)
    return working


def _normalize_project_code_key(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _normalize_space_only(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _resolve_dpr_config_path(raw_root: Path) -> Path | None:
    candidate = raw_root.parent / "DPR_Config.xlsx"
    if candidate.exists():
        return candidate
    repo_root = _repo_root_from(Path(__file__))
    fallback = repo_root / "Raw Data" / "DPR_Config.xlsx"
    if fallback.exists():
        return fallback
    return None


def _load_stringing_sheet_config(raw_root: Path) -> dict[str, list[dict[str, str]]]:
    config_path = _resolve_dpr_config_path(raw_root)
    if config_path is None:
        return {}

    try:
        df = pd.read_excel(config_path, sheet_name="Sheet Names Check", dtype=object)
    except Exception as exc:
        LOGGER.warning("Stringing: failed to read DPR config %s: %s", config_path, exc)
        return {}

    if df is None or df.empty:
        return {}

    headers = {_normalize_space_only(col): col for col in df.columns}
    project_col = headers.get("project code")
    stringing_col = headers.get("stringing sheet names")
    line_col = headers.get("stringing line names")
    if project_col is None or stringing_col is None:
        LOGGER.warning(
            "Stringing: DPR config %s is missing columns 'Project Code' or 'Stringing Sheet Names'.",
            config_path,
        )
        return {}

    mapping: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        project_val = row.get(project_col)
        if project_val is None or pd.isna(project_val) or str(project_val).strip() == "":
            continue
        project_key = _normalize_project_code_key(project_val)
        raw_stringing = row.get(stringing_col)
        if raw_stringing is None or pd.isna(raw_stringing) or str(raw_stringing).strip() == "":
            mapping[project_key] = []
            continue
        raw_line_names = row.get(line_col) if line_col else None
        sheet_count = len([chunk for chunk in str(raw_stringing).split(",") if str(chunk).strip()])
        line_chunks = [str(chunk).strip() for chunk in str(raw_line_names).split(",")] if raw_line_names not in (None, "") and not pd.isna(raw_line_names) else []
        if line_chunks and len(line_chunks) != sheet_count:
            LOGGER.warning(
                "Stringing: project '%s' has mismatched 'Stringing Line Names'; falling back to sheet-name inference.",
                str(project_val).strip(),
            )
            raw_line_names = ""

        entries = parse_sheet_line_entries(raw_stringing, raw_line_names, "stringing")
        deduped_entries: list[dict[str, str]] = []
        seen_sheet_keys: set[str] = set()
        for entry in entries:
            key = _normalize_space_only(entry.get("sheet_name"))
            if not key or key in seen_sheet_keys:
                continue
            seen_sheet_keys.add(key)
            deduped_entries.append(entry)
        mapping[project_key] = deduped_entries
    return mapping


def _resolve_project_sheet_name(sheet_names: Iterable[str], project_candidates: list[str]) -> str | None:
    by_space_key: dict[str, str] = {}
    for name in sheet_names:
        key = _normalize_space_only(name)
        if key and key not in by_space_key:
            by_space_key[key] = str(name)
    for candidate in project_candidates:
        hit = by_space_key.get(_normalize_space_only(candidate))
        if hit:
            return hit
    return None

def _stringing_root(base: Path) -> Path:
    """
    Resolve the single Stringing artifacts folder: <repo>/Parquets/Stringing.
    Also migrates legacy nested daily file once (if found).
    """
    def _find_parquets_anchor(start: Path) -> Path:
        cur = start.resolve()
        if cur.is_file():
            cur = cur.parent
        for p in [cur, *cur.parents]:
            cand = p / "Parquets"
            if cand.exists() and cand.is_dir():
                return cand
        here = Path(__file__).resolve()
        for p in [here.parent, *here.parents]:
            cand = p / "Parquets"
            if cand.exists() and cand.is_dir():
                return cand
        return (start / "Parquets").resolve()

    anchor = _find_parquets_anchor(base)
    root = (anchor / "Stringing").resolve()
    root.mkdir(parents=True, exist_ok=True)

    # one-time migration of legacy nested path
    legacy = root / "StringingDaily" / "stringing_daily.parquet"
    flat   = root / "StringingDaily.parquet"
    if legacy.exists() and not flat.exists():
        try:
            legacy.rename(flat)
            try:
                legacy.parent.rmdir()
            except Exception:
                pass
        except Exception:
            pass
    return root

def _iter_excel_candidates(raw_root: Path) -> list[Path]:
    """Recursively find all Excel files under RAW DPR root (matches runner)."""
    raw_root = raw_root.resolve()
    return sorted([p for p in raw_root.rglob("*.xls*") if p.is_file()])

def _list_excel_sheet_names(xlsx_path: Path | str) -> tuple[list[str], str | None]:
    """
    Return all sheet names from *xlsx_path*.

    Provides the names plus an optional error string if the workbook could not be opened.
    """
    try:
        with pd.ExcelFile(xlsx_path) as xl:
            return list(xl.sheet_names), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _resolve_excel_sheet_name(
    xlsx_path: Path,
    desired_sheet: str,
    *,
    contains_keyword: str | None = None,
    sheet_names: Iterable[str] | None = None,
) -> str | None:
    """
    Return the actual sheet name matching *desired_sheet* (case/spacing agnostic).

    When *contains_keyword* is provided, fall back to the first sheet whose title
    contains that keyword (case-insensitive) if the normalized match fails.
    """
    names: list[str]
    try:
        if sheet_names is not None:
            names = [str(name) for name in sheet_names]
        else:
            with pd.ExcelFile(xlsx_path) as xl:
                names = list(xl.sheet_names)
    except Exception:
        return None

    actual = _match_sheet_name(names, desired_sheet)
    if actual:
        return actual
    if desired_sheet in names:
        return desired_sheet
    if contains_keyword:
        keyword = contains_keyword.strip().lower()
        if keyword:
            for sheet_name in names:
                if keyword in sheet_name.lower():
                    return sheet_name
    return None



def _concat_union(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate with union of columns, preserving the first DF's order.

    Some workbooks duplicate headers (e.g., unnamed blank columns). Pandas
    refuses to reindex when duplicate column labels are present, so we drop
    subsequent duplicates before aligning the frames.
    """

    if not dfs:
        return pd.DataFrame()

    def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.columns.is_unique:
            return df
        return df.loc[:, ~df.columns.duplicated()].copy()

    normalized = [_dedup_columns(df) for df in dfs]
    cols = list(normalized[0].columns)
    for df in normalized[1:]:
        for c in df.columns:
            if c not in cols:
                cols.append(c)
    aligned = [df.reindex(columns=cols) for df in normalized]
    return pd.concat(aligned, ignore_index=True)


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
        series = pd.to_datetime(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        ts = pd.Timestamp(series.iloc[0])
        return ts.to_period("M").to_timestamp().strftime("%Y-%m")
    return ""


def _load_precompiled_stringing_microplan(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Reuse micro plan sheets written by the standalone microplan compiler if they exist.
    Falls back to None when the workbook is absent or unreadable.
    """
    workbook = out_root / "StringingCompiled_Output.xlsx"
    if not workbook.exists():
        return None

    try:
        sheets = pd.read_excel(
            workbook,
            sheet_name=[
                "MicroPlanResponsibilities",
                "MicroPlanIndex",
                "MicroPlanDataIssues",
            ],
        )
    except ValueError:
        # Some sheets missing; let caller rebuild from raw inputs.
        return None
    except Exception as exc:
        LOGGER.warning("Stringing: failed to read micro plan sheets from '%s': %s", workbook, exc)
        return None

    responsibilities = sheets.get("MicroPlanResponsibilities", pd.DataFrame())
    index_df = sheets.get("MicroPlanIndex", pd.DataFrame())
    issues_df = sheets.get("MicroPlanDataIssues", pd.DataFrame())
    return responsibilities, index_df, issues_df

def build_stringing_artifacts_every_run(raw_root: Path, sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    1) Resolve RAW DPRs root from pipeline_config.json (like erection flow)
    2) Recursively scan all Excels; log every file in Diagnostics
    3) For files that have the target sheet, read -> normalize -> expand daily
    4) Write ONE master compiled parquet, ONE master daily parquet, and ONE combined Excel
    Returns: (compiled_all, daily_all, out_root)
    """
    # Ensure we scan the correct RAW path (not the dataset path)
    raw_root = _resolve_stringing_raw_root(raw_root)
    plan_root = _resolve_stringing_microplan_root(raw_root)

    # Stringing output root (flat structure under <repo>/Parquets/Stringing)
    out_root = _stringing_root(raw_root)

    candidates = _iter_excel_candidates(raw_root)
    stringing_sheet_config = _load_stringing_sheet_config(raw_root)
    has_stringing_config = bool(stringing_sheet_config)
    skipped_no_stringing = 0
    skipped_not_in_config = 0
    plan_candidates = _iter_excel_candidates(plan_root)
    compiled_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    diag_rows: list[dict] = []
    data_issue_rows: list[pd.DataFrame] = []
    plan_frames: list[pd.DataFrame] = []
    plan_index_rows: list[dict[str, Any]] = []
    plan_issue_rows: list[dict[str, str]] = []
    dpr_issue_rows: list[dict[str, Any]] = []

    def _append_plan_index_entry(
        *,
        workbook: Path | str | None,
        sheet_name: str | None,
        project_code: str,
        project_label: str,
        rows_cleaned: int,
        input_rows: int,
        status: str,
        error: str,
        issues_logged: int = 0,
        plan_month: str = "",
        available_sheets: Iterable[str] | None = None,
    ) -> None:
        plan_index_rows.append(
            {
                "file_path": str(workbook or ""),
                "sheet_name": sheet_name or "",
                "project_name": project_label or project_code or "",
                "project_key": project_code or project_label or "",
                "rows_cleaned": int(rows_cleaned),
                "input_rows": int(input_rows),
                "issues_logged": int(issues_logged),
                "plan_month": plan_month or "",
                "status": status,
                "error": error or "",
                "available_sheets": "; ".join(available_sheets) if available_sheets else "",
            }
        )

    if not candidates:
        LOGGER.warning("Stringing: no Excel files found under '%s'", raw_root)

    def _log_dpr_diag(
        workbook: Path | str,
        project: str,
        rows: int,
        daily_rows: int,
        status: str,
        *,
        sheet_name: str | None = None,
        configured_sheet: str | None = None,
        line_name: str | None = None,
        line_name_source: str | None = None,
        header_row: int | None = None,
        columns_detected: list[str] | None = None,
        normalized_columns_ok: bool | None = None,
        present_columns: list[str] | None = None,
        missing_columns: list[str] | None = None,
        applied_map: dict[str, str] | None = None,
    ) -> None:
        diag_rows.append(
            {
                "Workbook": Path(workbook).name if workbook else "",
                "Project": project,
                "Sheet": sheet_name or "",
                "ConfiguredSheet": configured_sheet or "",
                "LineName": line_name or "",
                "LineNameSource": line_name_source or "",
                "DetectedHeaderRow": "" if header_row is None else int(header_row),
                "ColumnsDetected": ", ".join(columns_detected or []),
                "NormalizedColumnsOk": bool(normalized_columns_ok) if normalized_columns_ok is not None else "",
                "PresentColumns": ", ".join(present_columns or []),
                "MissingColumns": ", ".join(missing_columns or []),
                "AppliedMap": ", ".join(
                    [f"{key}->{value}" for key, value in (applied_map or {}).items()]
                ),
                "Rows": int(rows),
                "DailyRows": int(daily_rows),
                "Status": status,
            }
        )
        if status != "OK":
            dpr_issue_rows.append(
                {
                    "Workbook": Path(workbook).name if workbook else "",
                    "Project": project,
                    "Sheet": sheet_name or "",
                    "ConfiguredSheet": configured_sheet or "",
                    "LineName": line_name or "",
                    "LineNameSource": line_name_source or "",
                    "Issue": status,
                    "MissingColumns": ", ".join(missing_columns or []),
                    "Rows": int(rows),
                    "DailyRows": int(daily_rows),
                }
            )

    def _append_dpr_issue(
        workbook: Path | str,
        project: str,
        *,
        sheet_name: str | None,
        configured_sheet: str | None = None,
        line_name: str | None = None,
        line_name_source: str | None = None,
        issue: str,
        missing_columns: list[str] | None = None,
        rows: int = 0,
        daily_rows: int = 0,
    ) -> None:
        dpr_issue_rows.append(
            {
                "Workbook": Path(workbook).name if workbook else "",
                "Project": project,
                "Sheet": sheet_name or "",
                "ConfiguredSheet": configured_sheet or "",
                "LineName": line_name or "",
                "LineNameSource": line_name_source or "",
                "Issue": issue,
                "MissingColumns": ", ".join(missing_columns or []),
                "Rows": int(rows),
                "DailyRows": int(daily_rows),
            }
        )

    for wb in candidates:
        proj = _project_from_filename(wb.name) or "UNKNOWN"
        project_key = _normalize_project_code_key(proj)
        project_sheet_candidates = stringing_sheet_config.get(project_key)
        if has_stringing_config and project_sheet_candidates is None:
            skipped_not_in_config += 1
            continue
        if project_sheet_candidates is not None and not project_sheet_candidates:
            skipped_no_stringing += 1
            continue
        today = pd.Timestamp.today().normalize()
        workbook_sheet_names, _ = _list_excel_sheet_names(wb)
        if project_sheet_candidates:
            sheet_requests: list[tuple[dict[str, str], str | None]] = []
            for sheet_entry in project_sheet_candidates:
                configured_name = str(sheet_entry.get("sheet_name", "")).strip()
                actual_sheet = _resolve_project_sheet_name(workbook_sheet_names, [configured_name])
                sheet_requests.append((sheet_entry, actual_sheet))
        else:
            actual_sheet = _resolve_excel_sheet_name(
                wb,
                sheet_name,
                contains_keyword="stringing",
                sheet_names=workbook_sheet_names,
            )
            sheet_requests = [({"sheet_name": actual_sheet or "", "line_name": "", "line_name_source": ""}, actual_sheet)]

        for sheet_entry, actual_sheet in sheet_requests:
            configured_name = str(sheet_entry.get("sheet_name", "")).strip()
            line_name_override = normalize_line_name(sheet_entry.get("line_name", ""))
            line_name_source = str(sheet_entry.get("line_name_source", "")).strip()

            if not actual_sheet:
                _log_dpr_diag(
                    wb,
                    proj,
                    0,
                    0,
                    "NO_TARGET_SHEET",
                    sheet_name="",
                    configured_sheet=configured_name,
                    line_name=line_name_override,
                    line_name_source=line_name_source,
                )
                continue

            header_row = None
            header_labels: list[str] = []
            try:
                with pd.ExcelFile(wb) as xl:
                    df_probe = xl.parse(sheet_name=actual_sheet, header=None)
                header_row, header_labels = find_stringing_header_row(df_probe)
                header_labels = [str(label).strip() for label in (header_labels or [])]
            except Exception:
                header_row, header_labels = None, []

            try:
                df_raw = read_stringing_sheet_robust(str(wb), actual_sheet)
            except Exception as exc:
                LOGGER.warning(
                    "Stringing: failed reading '%s' [desired='%s', actual='%s']: %s",
                    wb,
                    configured_name or sheet_name,
                    actual_sheet,
                    exc,
                )
                _log_dpr_diag(
                    wb,
                    proj,
                    0,
                    0,
                    f"READ_FAIL: {type(exc).__name__}",
                    sheet_name=actual_sheet,
                    configured_sheet=configured_name,
                    line_name=line_name_override,
                    line_name_source=line_name_source,
                    header_row=header_row,
                    columns_detected=header_labels,
                )
                continue

            if df_raw is None or df_raw.empty:
                _log_dpr_diag(
                    wb,
                    proj,
                    0,
                    0,
                    "EMPTY_SHEET",
                    sheet_name=actual_sheet,
                    configured_sheet=configured_name,
                    line_name=line_name_override,
                    line_name_source=line_name_source,
                    header_row=header_row,
                    columns_detected=header_labels,
                )
                continue

            tse_value = extract_stringing_number_of_tse(str(wb), actual_sheet)

            try:
                compiled_norm, norm_report = normalize_stringing_columns(df_raw)
            except Exception:
                compiled_norm = df_raw.copy()
                norm_report = {"missing": []}

            missing_headers = list(norm_report.get("missing", []))
            if missing_headers:
                LOGGER.warning(
                    "Stringing: workbook '%s' sheet '%s' missing required headers: %s",
                    wb.name,
                    actual_sheet,
                    ", ".join(missing_headers),
                )
                _append_dpr_issue(
                    wb,
                    proj,
                    sheet_name=actual_sheet,
                    configured_sheet=configured_name,
                    line_name=line_name_override,
                    line_name_source=line_name_source,
                    issue="MISSING_REQUIRED_COLUMNS",
                    missing_columns=missing_headers,
                    rows=int(len(df_raw.index)),
                    daily_rows=0,
                )

            duplicate_headers = df_raw.columns[df_raw.columns.duplicated()].tolist()
            if duplicate_headers:
                deduped = list(dict.fromkeys(duplicate_headers))
                LOGGER.warning(
                    "Stringing: workbook '%s' sheet '%s' has duplicate headers that will be dropped: %s",
                    wb.name,
                    actual_sheet,
                    ", ".join(deduped),
                )

            if "source_file" not in compiled_norm.columns:
                compiled_norm["source_file"] = wb.name
            compiled_norm = _apply_project_identity_columns(
                compiled_norm,
                wb,
                project_code=proj,
                fallback_name=proj,
                line_name_override=line_name_override,
                source_sheet=actual_sheet,
                project_name_column="project_name",
                project_key_column="project_key",
            )
            if "project" not in compiled_norm.columns:
                compiled_norm["project"] = compiled_norm["project_name"]

            if "number_of_tse" not in compiled_norm.columns:
                compiled_norm["number_of_tse"] = pd.NA
            if tse_value is not None:
                compiled_norm["number_of_tse"] = int(tse_value)

            effective_line_name = str(compiled_norm.get("line_name", pd.Series([""])).iloc[0])

            try:
                work = compiled_norm.copy()
                has_po_start = "po_start_date" in work.columns
                has_po_complete = "po_completion_date" in work.columns
                has_fs_start = "fs_starting_date" in work.columns
                has_fs_complete = "fs_complete_date" in work.columns
                for col in ("po_start_date", "po_completion_date", "fs_starting_date", "fs_complete_date"):
                    if col in work.columns:
                        work[col] = pd.to_datetime(work[col], errors="coerce").dt.normalize()
                    else:
                        work[col] = pd.NaT

                work["length_km"] = pd.to_numeric(work.get("length_km", pd.Series(pd.NA, index=work.index)), errors="coerce")
                work["po_km"] = pd.to_numeric(work.get("po_km", pd.Series(pd.NA, index=work.index)), errors="coerce")

                def _filled(series: pd.Series) -> pd.Series:
                    return series.notna() & series.astype(str).str.strip().ne("")

                po_start_filled = _filled(work.get("po_start_date", pd.Series(pd.NaT, index=work.index))) if has_po_start else pd.Series(False, index=work.index)
                po_complete_filled = _filled(work.get("po_completion_date", pd.Series(pd.NaT, index=work.index))) if has_po_complete else pd.Series(False, index=work.index)
                fs_start_filled = _filled(work.get("fs_starting_date", pd.Series(pd.NaT, index=work.index))) if has_fs_start else pd.Series(False, index=work.index)
                fs_complete_filled = _filled(work.get("fs_complete_date", pd.Series(pd.NaT, index=work.index))) if has_fs_complete else pd.Series(False, index=work.index)

                po_start_invalid = po_start_filled & work["po_start_date"].isna() if has_po_start else pd.Series(False, index=work.index)
                po_complete_invalid = po_complete_filled & work["po_completion_date"].isna() if has_po_complete else pd.Series(False, index=work.index)
                fs_start_invalid = fs_start_filled & work["fs_starting_date"].isna() if has_fs_start else pd.Series(False, index=work.index)
                fs_complete_invalid = fs_complete_filled & work["fs_complete_date"].isna() if has_fs_complete else pd.Series(False, index=work.index)

                po_missing = (
                    (~po_start_filled if has_po_start else pd.Series(False, index=work.index))
                    | (~po_complete_filled if has_po_complete else pd.Series(False, index=work.index))
                )
                fs_missing = (
                    (~fs_start_filled if has_fs_start else pd.Series(False, index=work.index))
                    | (~fs_complete_filled if has_fs_complete else pd.Series(False, index=work.index))
                )

                po_non_positive = (
                    work["po_start_date"].notna()
                    & work["po_completion_date"].notna()
                    & ((work["po_completion_date"] - work["po_start_date"]).dt.days + 1 <= 0)
                ) if has_po_start and has_po_complete else pd.Series(False, index=work.index)
                fs_non_positive = (
                    work["fs_starting_date"].notna()
                    & work["fs_complete_date"].notna()
                    & ((work["fs_complete_date"] - work["fs_starting_date"]).dt.days + 1 <= 0)
                ) if has_fs_start and has_fs_complete else pd.Series(False, index=work.index)

                po_future = (work["po_completion_date"].notna() & (work["po_completion_date"] >= today)) if has_po_complete else pd.Series(False, index=work.index)
                fs_future = (work["fs_complete_date"].notna() & (work["fs_complete_date"] >= today)) if has_fs_complete else pd.Series(False, index=work.index)

                any_issue = (
                    po_start_invalid
                    | po_complete_invalid
                    | fs_start_invalid
                    | fs_complete_invalid
                    | po_missing
                    | fs_missing
                    | po_non_positive
                    | fs_non_positive
                    | po_future
                    | fs_future
                )

                if any_issue.any():
                    columns_for_issues = [
                        "project_name",
                        "from_ap",
                        "to_ap",
                        "gang_name",
                        "method",
                        "status",
                        "po_start_date",
                        "po_completion_date",
                        "fs_starting_date",
                        "fs_complete_date",
                        "length_m",
                        "length_km",
                        "po_km",
                        "source_file",
                        "source_sheet",
                    ]
                    for col in columns_for_issues:
                        if col not in work.columns:
                            work[col] = pd.NA

                    def _mk_issue(row: pd.Series) -> str:
                        messages: list[str] = []
                        if has_po_start and pd.isna(row.get("po_start_date")):
                            messages.append("Missing/Invalid PO Start Date")
                        if has_po_complete and pd.isna(row.get("po_completion_date")):
                            messages.append("Missing/Invalid PO Complete Date")
                        if has_fs_start and pd.isna(row.get("fs_starting_date")):
                            messages.append("Missing/Invalid F/S Start Date")
                        if has_fs_complete and pd.isna(row.get("fs_complete_date")):
                            messages.append("Missing/Invalid F/S Complete Date")
                        if has_po_start and has_po_complete and row.get("po_start_date") is not pd.NaT and row.get("po_completion_date") is not pd.NaT:
                            try:
                                if (row["po_completion_date"] - row["po_start_date"]).days + 1 <= 0:
                                    messages.append("PO Start > PO Complete (non-positive duration)")
                            except Exception:
                                pass
                        if has_fs_start and has_fs_complete and row.get("fs_starting_date") is not pd.NaT and row.get("fs_complete_date") is not pd.NaT:
                            try:
                                if (row["fs_complete_date"] - row["fs_starting_date"]).days + 1 <= 0:
                                    messages.append("F/S Start > F/S Complete (non-positive duration)")
                            except Exception:
                                pass
                        if has_po_complete and pd.notna(row.get("po_completion_date")) and row["po_completion_date"] >= today:
                            messages.append("PO Completion >= today (future)")
                        if has_fs_complete and pd.notna(row.get("fs_complete_date")) and row["fs_complete_date"] >= today:
                            messages.append("F/S Completion >= today (future)")
                        return "; ".join(messages)

                    issue_rows = work.loc[any_issue, columns_for_issues].copy()
                    issue_rows["Issues"] = issue_rows.apply(_mk_issue, axis=1)
                    data_issue_rows.append(issue_rows)
            except Exception:
                pass

            try:
                daily = expand_stringing_to_daily(compiled_norm)
            except Exception:
                daily = pd.DataFrame()

            compiled_frames.append(compiled_norm)
            if not daily.empty:
                if "source_file" not in daily.columns:
                    daily["source_file"] = wb.name
                daily = _apply_project_identity_columns(
                    daily,
                    wb,
                    project_code=str(compiled_norm.get("project_code", pd.Series([proj])).iloc[0]),
                    fallback_name=str(compiled_norm.get("project_name", pd.Series([proj])).iloc[0]),
                    line_name_override=line_name_override,
                    source_sheet=actual_sheet,
                    project_name_column="project_name",
                    project_key_column="project_key",
                )
                if "project" not in daily.columns:
                    daily["project"] = daily["project_name"]
                daily_frames.append(daily)
                _log_dpr_diag(
                    wb,
                    proj,
                    int(len(compiled_norm)),
                    int(len(daily)),
                    "OK",
                    sheet_name=actual_sheet,
                    configured_sheet=configured_name,
                    line_name=effective_line_name,
                    line_name_source=line_name_source,
                    header_row=header_row,
                    columns_detected=header_labels,
                    normalized_columns_ok=bool(norm_report.get("normalized_columns_ok", False)),
                    present_columns=list(norm_report.get("present", [])),
                    missing_columns=list(norm_report.get("missing", [])),
                    applied_map=dict(norm_report.get("applied_map", {})),
                )
            else:
                _log_dpr_diag(
                    wb,
                    proj,
                    int(len(compiled_norm)),
                    0,
                    "NO_DAILY",
                    sheet_name=actual_sheet,
                    configured_sheet=configured_name,
                    line_name=effective_line_name,
                    line_name_source=line_name_source,
                    header_row=header_row,
                    columns_detected=header_labels,
                    normalized_columns_ok=bool(norm_report.get("normalized_columns_ok", False)),
                    present_columns=list(norm_report.get("present", [])),
                    missing_columns=list(norm_report.get("missing", [])),
                    applied_map=dict(norm_report.get("applied_map", {})),
                )

    if skipped_no_stringing:
        LOGGER.info(
            "Stringing: skipped %s workbook(s) per DPR_Config (no stringing sheet configured).",
            skipped_no_stringing,
        )
    if skipped_not_in_config:
        LOGGER.info(
            "Stringing: skipped %s workbook(s) not listed in DPR_Config.",
            skipped_not_in_config,
        )

    precompiled_plan = _load_precompiled_stringing_microplan(out_root)
    plan_responsibilities: pd.DataFrame | None = None
    plan_index_df: pd.DataFrame | None = None
    plan_issue_sheet: pd.DataFrame | None = None

    if precompiled_plan is None:
        if not plan_candidates:
            LOGGER.warning("Stringing: no Micro Plan files found under '%s'", plan_root)

        for wb in plan_candidates:
            sheet_names, sheet_names_error = _list_excel_sheet_names(wb)
            available_sheets = list(sheet_names)
            if sheet_names_error:
                available_sheets = [f"[error: {sheet_names_error}]"]
            project_code, project_label = infer_project_hint(wb)
            names_for_matching: Iterable[str] | None = sheet_names if sheet_names_error is None else None
            actual_sheet = _resolve_excel_sheet_name(
                wb,
                sheet_name,
                contains_keyword="stringing",
                sheet_names=names_for_matching,
            )
            if not actual_sheet:
                missing_issue = "NO_STRINGING_SHEET"
                if sheet_names_error:
                    missing_issue = f"SHEET_LIST_FAILED: {sheet_names_error}"
                plan_issue_rows.append({"workbook": str(wb), "sheet": "", "issue": missing_issue})
                _append_plan_index_entry(
                    workbook=wb,
                    sheet_name="",
                    project_code=project_code,
                    project_label=project_label,
                    rows_cleaned=0,
                    input_rows=0,
                    status="error",
                    error=missing_issue,
                    available_sheets=available_sheets,
                )
                continue
            try:
                df_raw = pd.read_excel(wb, sheet_name=actual_sheet, header=1)
            except Exception as exc:
                plan_issue_rows.append({"workbook": str(wb), "sheet": actual_sheet, "issue": f"READ_FAILED: {exc}"})
                _append_plan_index_entry(
                    workbook=wb,
                    sheet_name=actual_sheet,
                    project_code=project_code,
                    project_label=project_label,
                    rows_cleaned=0,
                    input_rows=0,
                    status="error",
                    error=f"READ_FAILED: {exc}",
                    available_sheets=available_sheets,
                )
                continue

            normalized, _plan_completion, local_issues = prepare_stringing_plan_frame(
                df_raw,
                project_hint=project_label or project_code,
                source_path=wb,
                sheet_name=actual_sheet,
            )
            if project_code:
                normalized["project_key"] = normalized["project_key"].replace("", project_code)
            if project_label:
                normalized["project_name"] = normalized["project_name"].replace("", project_label)
            normalized = _apply_project_identity_columns(
                normalized,
                wb,
                project_code=project_code,
                fallback_name=project_label or project_code,
                project_name_column="project_name",
                project_key_column="project_key",
            )
            plan_frames.append(normalized)
            plan_issue_rows.extend(local_issues)
            plan_month_value = _infer_plan_month_from_frame(normalized)
            _append_plan_index_entry(
                workbook=wb,
                sheet_name=actual_sheet,
                project_code=project_code,
                project_label=project_label,
                rows_cleaned=int(len(normalized)),
                input_rows=int(len(df_raw)),
                status="ok",
                error="",
                issues_logged=len(local_issues),
                plan_month=plan_month_value,
                available_sheets=available_sheets,
            )

        compiled_all = _concat_union(compiled_frames) if compiled_frames else pd.DataFrame()
        daily_all    = _concat_union(daily_frames) if daily_frames else pd.DataFrame()
        if plan_frames:
            plan_responsibilities = pd.concat(plan_frames, ignore_index=True)
        else:
            plan_responsibilities, _, _ = prepare_stringing_plan_frame(pd.DataFrame())
        plan_index_df = pd.DataFrame(plan_index_rows)
        if plan_index_df.empty:
            plan_index_df = pd.DataFrame(
                columns=[
                    "file_path",
                    "sheet_name",
                    "project_name",
                    "project_key",
                    "rows_cleaned",
                    "input_rows",
                    "issues_logged",
                    "plan_month",
                    "status",
                    "error",
                    "available_sheets",
                ]
            )
        plan_issue_df = pd.DataFrame(plan_issue_rows)
        if plan_issue_df.empty:
            plan_issue_sheet = pd.DataFrame(columns=["workbook", "sheet", "issue"])
        else:
            plan_issue_sheet = plan_issue_df
    else:
        plan_responsibilities, plan_index_df, plan_issue_sheet = precompiled_plan
        compiled_all = _concat_union(compiled_frames) if compiled_frames else pd.DataFrame()
        daily_all    = _concat_union(daily_frames) if daily_frames else pd.DataFrame()
        if plan_index_df.empty:
            plan_index_df = pd.DataFrame(
                columns=[
                    "file_path",
                    "sheet_name",
                    "project_name",
                    "project_key",
                    "rows_cleaned",
                    "input_rows",
                    "issues_logged",
                    "plan_month",
                    "status",
                    "error",
                    "available_sheets",
                ]
            )
        if plan_issue_sheet.empty:
            plan_issue_sheet = pd.DataFrame(columns=["workbook", "sheet", "issue"])

    # Write masters (overwrite each run) — keep a single directory (no nested daily folder)
    master_compiled = out_root / "StringingCompiled.parquet"
    master_daily    = out_root / "StringingDaily.parquet"
    try:
        _write_parquet(compiled_all, master_compiled)
    except Exception as exc:
        LOGGER.warning("Stringing: failed writing master compiled parquet: %s", exc)
    try:
        _write_parquet(daily_all, master_daily)
    except Exception as exc:
        LOGGER.warning("Stringing: failed writing master daily parquet: %s", exc)

    diag_df = pd.DataFrame(diag_rows)
    dpr_issues_df = pd.DataFrame(dpr_issue_rows)
    data_issues_df = pd.concat(data_issue_rows, ignore_index=True) if data_issue_rows else pd.DataFrame()
    if dpr_issues_df.empty:
        dpr_issues_df = pd.DataFrame(
            columns=["Workbook", "Project", "Sheet", "Issue", "MissingColumns", "Rows", "DailyRows"]
        )
    if data_issues_df.empty:
        data_issues_df = pd.DataFrame(
            columns=[
                "project_name",
                "from_ap",
                "to_ap",
                "gang_name",
                "method",
                "status",
                "po_start_date",
                "po_completion_date",
                "fs_starting_date",
                "fs_complete_date",
                "length_m",
                "length_km",
                "po_km",
                "source_file",
                "Issues",
            ]
        )

    # Combined Excel (fallback/log) — always write; Diagnostics includes *all* files scanned
    try:
        with pd.ExcelWriter(out_root / "StringingCompiled_Output.xlsx", engine="openpyxl", mode="w") as xw:
            (compiled_all if not compiled_all.empty else pd.DataFrame()).to_excel(
                xw, sheet_name=sheet_name, index=False
            )
            if not daily_all.empty:
                daily_all.to_excel(xw, sheet_name="Daily", index=False)
            # Erection-style sheets for consistency
            (daily_all if not daily_all.empty else pd.DataFrame()).to_excel(
                xw, sheet_name="ProdDailyExpanded", index=False
            )
            (compiled_all if not compiled_all.empty else pd.DataFrame()).to_excel(
                xw, sheet_name="RawData", index=False
            )
            data_issues_df.to_excel(xw, sheet_name="Data Issues", index=False)
            dpr_issues_df.to_excel(xw, sheet_name="Issues", index=False)
            diag_df.to_excel(xw, sheet_name="Diagnostics", index=False)
            plan_responsibilities.to_excel(xw, sheet_name="MicroPlanResponsibilities", index=False)
            plan_index_df.to_excel(xw, sheet_name="MicroPlanIndex", index=False)
            plan_issue_sheet.to_excel(xw, sheet_name="MicroPlanDataIssues", index=False)
    except Exception as exc:
        LOGGER.warning("Stringing: failed writing combined Excel: %s", exc)

    print(f"[Stringing] RAW Root       : {raw_root}")
    print(f"[Stringing] Out Root       : {out_root}")
    print(f"[Stringing] Master Compiled: {master_compiled}")
    print(f"[Stringing] Master Daily   : {master_daily}")

    return compiled_all, daily_all, out_root



# --- Cached artifact readers -------------------------------------------------
def _read_prebuilt_stringing_artifacts(
    raw_root: Path,
    sheet_name: str,
    probe_dirs: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, Path] | None:
    """
    Attempt to reuse cached Parquets/Stringing outputs before rebuilding from RAW inputs.

    Returns (compiled_df, daily_df, artifact_root) or None if no cached dataset exists.
    """

    def _candidate_roots() -> list[Path]:
        roots: list[Path] = []
        canonical = _stringing_root(raw_root)
        roots.append(canonical)
        search_root = _resolve_search_root(raw_root)
        for rel in probe_dirs:
            if not rel:
                continue
            try:
                candidate = (search_root / Path(rel)).resolve()
            except Exception:
                continue
            if candidate not in roots:
                roots.append(candidate)
        return roots

    for root in _candidate_roots():
        workbook_path = root / "StringingCompiled_Output.xlsx"
        compiled_df: pd.DataFrame | None = None
        daily_df: pd.DataFrame | None = None

        compiled_source = _find_parquet_source(root, "StringingCompiled") or _find_parquet_source(root, sheet_name)
        if compiled_source:
            try:
                compiled_df = _read_parquet(compiled_source)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Stringing: failed reading compiled parquet '%s': %s", compiled_source, exc)

        if compiled_df is None and workbook_path.exists():
            compiled_df = _try_read_excel_sheet(workbook_path, sheet_name)

        daily_source = _find_parquet_source(root, "StringingDaily")
        if daily_source:
            try:
                daily_df = _read_parquet(daily_source)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Stringing: failed reading daily parquet '%s': %s", daily_source, exc)

        if daily_df is None and workbook_path.exists():
            try:
                daily_df = pd.read_excel(workbook_path, sheet_name="Daily")
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Stringing: failed reading 'Daily' sheet from '%s': %s", workbook_path, exc)
                daily_df = pd.DataFrame()

        if compiled_df is not None and daily_df is not None:
            missing_parts: list[str] = []
            if isinstance(compiled_df, pd.DataFrame) and not compiled_df.empty:
                if not _stringing_frame_has_project_metadata(compiled_df):
                    missing_parts.append("compiled")
            if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
                if not _stringing_frame_has_project_metadata(daily_df):
                    missing_parts.append("daily")

            if missing_parts:
                LOGGER.warning(
                    "Stringing: cached dataset in %s missing project metadata (%s); forcing rebuild",
                    root,
                    ", ".join(missing_parts),
                )
                continue

            LOGGER.info("Stringing: loaded cached artifacts from %s", root)
            return compiled_df, daily_df, root
    return None


# --- NEW: ensure project_name is present on a stringing frame ---
def _ensure_stringing_project_name(df: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """
    If 'project_name' is missing/blank in df, fill it by parsing the project
    code from the Excel/parquet file name (e.g., TA 415 / TB 408).
    Also sets a 'project' mirror column and 'source_file' for auditability.
    """
    working = df.copy()
    if "source_file" not in working.columns:
        working["source_file"] = base_path.name
    working = _apply_project_identity_columns(working, base_path, project_name_column="project_name", project_key_column="")
    if "project" not in working.columns:
        working["project"] = working["project_name"]
    else:
        s = working["project"].astype(str).str.strip()
        working["project"] = s.mask(s.eq("") | s.eq("nan") | s.eq("None"), working["project_name"])
    return working


def _stringing_frame_has_project_metadata(df: pd.DataFrame) -> bool:
    """Return True when a stringing frame has at least one non-empty project field."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return False

    def _has_values(column: str) -> bool:
        if column not in df.columns:
            return False
        series = df[column]
        if series.empty:
            return False
        try:
            normalized = series.astype("string").fillna("").str.strip().str.lower()
        except Exception:
            normalized = series.astype(str).str.strip().str.lower()
        mask = ~normalized.isin({"", "nan", "none", "null"})
        return bool(mask.any())

    return _has_values("project_name") or _has_values("project") or _has_values("project_display")


def _ttl_lru_cache(maxsize: int, ttl_seconds: int):
    """Return an LRU cache decorator with simple time-based invalidation."""

    def decorator(func):
        cached = functools.lru_cache(maxsize=maxsize)(func)
        expiry = {"value": 0.0}

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            if ttl_seconds > 0 and now >= expiry["value"]:
                cached.cache_clear()
                expiry["value"] = now + ttl_seconds
            return cached(*args, **kwargs)

        wrapped.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        return wrapped

    return decorator


def _parquet_dataset_available(path: Path) -> bool:
    """Return True if *path* references a Parquet dataset (file or directory)."""

    path = Path(path)
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in PARQUET_SUFFIXES
    for suffix in PARQUET_SUFFIXES:
        iterator = path.rglob(f"*{suffix}")
        if next(iterator, None) is not None:
            return True
    return False


def _candidate_stems(name: str) -> list[str]:
    cleaned = name.strip()
    if not cleaned:
        return []
    variants = {
        cleaned,
        cleaned.replace(" ", ""),
        cleaned.replace("_", ""),
        cleaned.replace("-", ""),
        cleaned.lower(),
        cleaned.upper(),
    }
    return [variant for variant in variants if variant]


def _resolve_search_root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    return path


def _find_parquet_source(path: Path, table: str | None) -> str | None:
    """Return a DuckDB-compatible parquet path or glob for *table* relative to *path*."""

    if not table:
        return None
    table = table.strip()
    if not table:
        return None

    path = Path(path)
    lower_table = table.lower()

    if path.is_file() and path.suffix.lower() in PARQUET_SUFFIXES and path.stem.lower() == lower_table:
        return str(path)

    root = _resolve_search_root(path)
    stems = _candidate_stems(table)

    for stem in stems:
        for suffix in PARQUET_SUFFIXES:
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                return str(candidate)

    for stem in stems:
        directory = root / stem
        if directory.is_dir():
            for suffix in PARQUET_SUFFIXES:
                if any(directory.glob(f"*{suffix}")):
                    return str(directory / f"*{suffix}")

    if root.is_dir():
        for suffix in PARQUET_SUFFIXES:
            match = next(
                (
                    candidate
                    for candidate in root.glob(f"**/*{suffix}")
                    if candidate.stem.lower() == lower_table
                ),
                None,
            )
            if match:
                return str(match)
    return None


def _read_parquet(source: str) -> pd.DataFrame:
    """Read *source* (file or glob) into a DataFrame via DuckDB."""

    LOGGER.debug("Reading parquet via DuckDB: %s", source)
    with duckdb.connect(database=":memory:") as con:
        return con.execute("SELECT * FROM read_parquet(?)", [source]).df()

def is_parquet_dataset(path: Path | str) -> bool:
    """Return True if *path* references a parquet-backed dataset."""

    return _parquet_dataset_available(Path(path))


def find_parquet_source(path: Path | str, table: str) -> str | None:
    """Resolve the parquet file/glob for *table* relative to *path*."""

    return _find_parquet_source(Path(path), table)


def read_parquet_table(source: str) -> pd.DataFrame:
    """Load *source* into a DataFrame using DuckDB."""

    return _read_parquet(source)


# -----------------------------
# Stringing compiled (stub)
# -----------------------------
def _try_read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read the target sheet from an Excel file using normalization that removes all non-alphanumerics.
    If the normalized names don’t match, returns an empty DataFrame.
    """
    try:
        with pd.ExcelFile(path) as xl:
            actual = _match_sheet_name(xl.sheet_names, sheet_name)
            if not actual:
                LOGGER.warning(
                    "Stringing: sheet matching '%s' (norm='%s') not found in '%s'. Available: %s",
                    sheet_name, _norm_sheet(sheet_name), path, xl.sheet_names
                )
                return pd.DataFrame()
        # Use your robust reader with the resolved actual sheet name
        return read_stringing_sheet_robust(str(path), actual)
    except FileNotFoundError:
        LOGGER.warning("Stringing: workbook not found: '%s'", path)
        return pd.DataFrame()
    except Exception as exc:
        LOGGER.warning("Stringing: failed reading sheet '%s' from '%s': %s", sheet_name, path, exc)
        return pd.DataFrame()


def _find_stringing_parquet_source(root: Path, sheet_name: str, probe_dirs: tuple[str, ...]) -> str | None:
    """Return a parquet file/glob for the stringing dataset, if any.

    Strategy:
    - Prefer an exact table match via the given sheet name (normalized stems).
    - Else probe configured directory names that may contain parquet files.
    Returns a DuckDB-compatible source (file path or directory glob) or None.
    """
    # 1) Try direct table/filename match using existing helper
    source = _find_parquet_source(root, sheet_name)
    if source:
        return source

    # 2) Probe configured directory names (e.g., 'StringingCompiled_Output_parquet')
    search_root = _resolve_search_root(root)
    for dirname in probe_dirs:
        candidate_dir = search_root / dirname
        if candidate_dir.is_dir():
            for suffix in PARQUET_SUFFIXES:
                if any(candidate_dir.glob(f"*{suffix}")):
                    return str(candidate_dir / f"*{suffix}")

    # 3) Fallback: search recursively for a matching stem based on sheet name variants
    #    This mirrors _find_parquet_source behavior but broadens search if needed.
    stems = _candidate_stems(sheet_name)
    if search_root.is_dir():
        for suffix in PARQUET_SUFFIXES:
            for candidate in search_root.glob(f"**/*{suffix}"):
                lowered = candidate.stem.lower()
                if any(stem.lower() == lowered for stem in stems):
                    return str(candidate)
    return None


def _stringing_output_paths(base: Path) -> tuple[Path, Path]:
    """
    Returns (workbook_path, parquet_dir) for STRINGING artifacts.

    Target layout (single folder):
      <repo>/Parquets/Stringing/
        - StringingCompiled.parquet
        - StringingDaily.parquet
        - StringingCompiled_Output.xlsx
    """
    def _find_parquets_anchor(start: Path) -> Path | None:
        cur = start.resolve()
        if cur.is_file():
            cur = cur.parent
        # walk up from start
        for parent in [cur, *cur.parents]:
            cand = parent / "Parquets"
            if cand.exists() and cand.is_dir():
                return cand
        # also try from this module's location
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            cand = parent / "Parquets"
            if cand.exists() and cand.is_dir():
                return cand
        return None

    anchor = _find_parquets_anchor(base) or (base.resolve() / "Parquets")
    root = (anchor / "Stringing").resolve()
    root.mkdir(parents=True, exist_ok=True)
    workbook_path = root / "StringingCompiled_Output.xlsx"
    parquet_dir = root
    return workbook_path, parquet_dir



def _export_stringing_compiled_artifacts(base: Path, sheet_name: str, df_raw: pd.DataFrame) -> None:
    """Write a compiled stringing workbook and a simple parquet dataset.

    - Creates `StringingCompiled_Output.xlsx` with sheets:
        - the original `sheet_name` (raw or lightly normalized)
        - `Diagnostics` with presence/health info
        - `Issues` listing rows with invalid/missing critical dates
        - `README_Assumptions` noting basic rules
    - Writes parquet files directly under the `Parquets/Stringing` folder
      for the raw compiled table (faster subsequent loads).

    This mirrors the erection flow at a lightweight level and is idempotent
    (overwrites workbook; refreshes parquet files on each call without using
    the legacy *_parquet directories).
    """
    # --- ensure project column exists on the artifact we’re about to write ---
    try:
        df_raw = _ensure_stringing_project_name(df_raw, Path(base))
    except Exception:
        pass

    if df_raw is None or df_raw.empty:
        return
    workbook_path, parquet_dir = _stringing_output_paths(base)
    
    print(f"[Stringing] workbook path: {workbook_path}")
    print(f"[Stringing] parquet dir : {parquet_dir}")
    # Build diagnostics and issues
    try:
        normalized, norm_report = normalize_stringing_columns(df_raw)
    except Exception:
        normalized, norm_report = df_raw.copy(), {"normalized_columns_ok": False, "present": [], "missing": [], "applied_map": {}}

    # Keep line-aware project metadata in both raw and normalized artifacts.
    source_name = str(Path(base).name)
    proj = parse_project_code_from_filename(source_name)
    df_raw = _apply_project_identity_columns(
        df_raw,
        Path(base),
        project_code=proj,
        fallback_name=proj,
        project_name_column="project_name",
        project_key_column="project_key",
    )
    normalized = _apply_project_identity_columns(
        normalized,
        Path(base),
        project_code=proj,
        fallback_name=proj,
        project_name_column="project_name",
        project_key_column="project_key",
    )
    if "project" not in df_raw.columns:
        df_raw["project"] = df_raw["project_name"]
    if "project" not in normalized.columns:
        normalized["project"] = normalized["project_name"]

    # write Excel workbook (always)
    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="w") as xw:
        normalized.to_excel(xw, sheet_name=sheet_name, index=False)
    
    # write compiled parquet in the SAME folder
    compiled_parquet = parquet_dir / "StringingCompiled.parquet"
    _write_parquet(df_raw, compiled_parquet)
    
    try:
        date_metrics = summarize_date_parsing(df_raw)
    except Exception:
        date_metrics = {"po_start_date_parsed_count": 0, "fs_complete_date_parsed_count": 0, "invalid_date_rows": 0}

    try:
        _, length_metrics = add_length_units(normalized)
    except Exception:
        length_metrics = {"total_length_km": 0.0, "min_length_km": 0.0, "max_length_km": 0.0}

    # Prepare issues table
    issues_df = pd.DataFrame()
    try:
        work = normalized.copy()
        po_col = "po_start_date"
        end_col = "fs_complete_date"
        if po_col in work.columns and end_col in work.columns:
            po_val = work[po_col]
            end_val = work[end_col]
            po_parsed = pd.to_datetime(po_val, errors="coerce").dt.normalize()
            end_parsed = pd.to_datetime(end_val, errors="coerce").dt.normalize()
            po_filled = po_val.astype(str).str.strip().ne("") & po_val.notna()
            end_filled = end_val.astype(str).str.strip().ne("") & end_val.notna()
            po_invalid = po_filled & po_parsed.isna()
            end_invalid = end_filled & end_parsed.isna()
            missing_po = ~po_filled
            missing_end = ~end_filled
            any_issue = po_invalid | end_invalid | missing_po | missing_end
            if any_issue.any():
                tmp = work.loc[any_issue].copy()
                def _mk_issue(row):
                    msgs = []
                    if pd.isna(pd.to_datetime(row.get(po_col), errors="coerce")) and str(row.get(po_col, "")).strip():
                        msgs.append("Invalid PO Start Date")
                    if pd.isna(pd.to_datetime(row.get(end_col), errors="coerce")) and str(row.get(end_col, "")).strip():
                        msgs.append("Invalid F/S Complete Date")
                    if not str(row.get(po_col, "")).strip():
                        msgs.append("Missing PO Start Date")
                    if not str(row.get(end_col, "")).strip():
                        msgs.append("Missing F/S Complete Date")
                    return "; ".join(msgs)
                tmp["Issues"] = tmp.apply(_mk_issue, axis=1)
                issues_df = tmp
    except Exception:
        issues_df = pd.DataFrame()

    source_name = str(Path(base).name)
    project_guess = parse_project_code_from_filename(source_name)
    diagnostics_rows = [{
        "sheet": sheet_name,
        "rows": int(len(df_raw.index)),
        "source": source_name,
        "project_code_guess": project_guess,
        "normalized_columns_ok": bool(norm_report.get("normalized_columns_ok", False)),
        "present_columns": ", ".join(norm_report.get("present", [])),
        "missing_columns": ", ".join(norm_report.get("missing", [])),
        "po_start_date_parsed_count": int(date_metrics.get("po_start_date_parsed_count", 0)),
        "fs_complete_date_parsed_count": int(date_metrics.get("fs_complete_date_parsed_count", 0)),
        "invalid_date_rows": int(date_metrics.get("invalid_date_rows", 0)),
        "total_length_km": float(length_metrics.get("total_length_km", 0.0)),
        "min_length_km": float(length_metrics.get("min_length_km", 0.0)),
        "max_length_km": float(length_metrics.get("max_length_km", 0.0)),
    }]
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    readme_df = pd.DataFrame([
        {
            "Note": "Stringing compiled workbook generated by dashboard loader.",
            "Rules": "Dates parsed with pandas to_datetime (coerce); configured stringing sheets are processed independently with per-sheet line identity before concatenation; PO start to F/S complete inclusive; basic column normalization applied.",
        }
    ])

    try:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            df_raw.to_excel(writer, sheet_name=sheet_name[:31] or "Stringing", index=False)
            diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)
            if not issues_df.empty:
                issues_df.to_excel(writer, sheet_name="Issues", index=False)
            readme_df.to_excel(writer, sheet_name="README_Assumptions", index=False)
        LOGGER.info("Wrote stringing compiled workbook to %s", workbook_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to write stringing workbook '%s': %s", workbook_path, exc)

    try:
        if parquet_dir.exists():
            for p in parquet_dir.glob("**/*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
        parquet_dir.mkdir(parents=True, exist_ok=True)
        compiled_parquet = parquet_dir / "StringingCompiled.parquet"
        _write_parquet(df_raw, compiled_parquet)
        LOGGER.info("Wrote stringing compiled parquet to %s", compiled_parquet)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to write stringing compiled parquet near '%s': %s", parquet_dir, exc)

@_ttl_lru_cache(maxsize=CACHE_MAXSIZE, ttl_seconds=CACHE_TTL_SECONDS)
def _load_stringing_artifacts_cached(
    data_path: str,
    sheet_name: str,
    probe_dirs: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Shared cached builder so compiled + daily reads reuse the same scan."""

    raw_root = _resolve_stringing_raw_root(Path(data_path))
    prebuilt = _read_prebuilt_stringing_artifacts(raw_root, sheet_name, probe_dirs)
    if prebuilt is not None:
        compiled_all, daily_all, out_root = prebuilt
    else:
        LOGGER.info("Stringing: cached artifacts not found; rebuilding from RAW under '%s'", raw_root)
        compiled_all, daily_all, out_root = build_stringing_artifacts_every_run(raw_root, sheet_name)
    return compiled_all, daily_all, out_root


def _load_stringing_compiled_raw_cached(data_path: str, sheet_name: str, probe_dirs: tuple[str, ...]) -> pd.DataFrame:
    compiled_all, _, _ = _load_stringing_artifacts_cached(data_path, sheet_name, probe_dirs)
    return compiled_all


def load_stringing_compiled_raw(config_or_path: AppConfig | Path | str) -> pd.DataFrame:
    """Safe, non-expanding reader for the 'Stringing Compiled' dataset.

    - Detects presence via sheet name for Excel or via parquet files/dirs.
    - Returns a DataFrame with raw columns as-is.
    - Returns an empty DataFrame (and logs a warning) if not found.
    - Caches results for a short TTL to avoid expensive health probes.
    """
    if isinstance(config_or_path, AppConfig):
        config = config_or_path
    else:
        config = AppConfig(data_path=Path(config_or_path))

    resolved = str(Path(config.data_path).resolve())
    df = _load_stringing_compiled_raw_cached(
        resolved,
        config.stringing_sheet_name,
        tuple(getattr(config, "stringing_parquet_dirs", ())) or tuple(),
    )
    return df.copy()

load_stringing_compiled_raw.cache_clear = _load_stringing_artifacts_cached.cache_clear  # type: ignore[attr-defined]

# -----------------------------
# Stringing daily (expanded)
# -----------------------------
def _guarded_write_stringing_daily(root: Path, table: str, df: pd.DataFrame) -> None:
    """Persist daily df to a directory named like a parquet table under root.

    Mirrors erection flow: a subdirectory with the table name contains parquet file(s).
    If parquet files already exist, do not overwrite.
    """
    search_root = _resolve_search_root(root)
    target_dir = search_root / table
    target_dir.mkdir(parents=True, exist_ok=True)
    has_parquet = any(target_dir.rglob("*.parquet")) or any(target_dir.rglob("*.parq")) or any(target_dir.rglob("*.pq"))
    if has_parquet:
        return
    # Write a single file for simplicity
    target_file = target_dir / "stringing_daily.parquet"
    try:
        _write_parquet(df, target_file)
        LOGGER.info("Wrote stringing daily parquet to %s", target_file)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to write stringing daily parquet to '%s': %s", target_file, exc)


def _load_stringing_daily_cached(data_path: str, sheet_name: str, probe_dirs: tuple[str, ...], daily_table: str = "StringingDaily") -> pd.DataFrame:
    _, daily_all, _ = _load_stringing_artifacts_cached(data_path, sheet_name, probe_dirs)
    return daily_all


def load_stringing_daily(config_or_path: AppConfig | Path | str) -> pd.DataFrame:
    """Public loader for expanded per-day stringing rows.

    Parquet-first; Excel fallback via compiled raw + expansion. Caches via TTL.
    """
    if isinstance(config_or_path, AppConfig):
        config = config_or_path
    else:
        config = AppConfig(data_path=Path(config_or_path))

    resolved = str(Path(config.data_path).resolve())
    df = _load_stringing_daily_cached(
        resolved,
        config.stringing_sheet_name,
        tuple(getattr(config, "stringing_parquet_dirs", ())) or tuple(),
        getattr(config, "stringing_daily_table", "StringingDaily"),
    )
    return df.copy()

load_stringing_daily.cache_clear = _load_stringing_artifacts_cached.cache_clear  # type: ignore[attr-defined]


def _write_parquet(df: pd.DataFrame, destination: Path) -> None:
    """Persist *df* to *destination* using DuckDB for consistent parquet writes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        with duckdb.connect(database=":memory:") as con:
            con.register("df_to_write", df)
            con.execute(
                "COPY df_to_write TO ? (FORMAT 'parquet', COMPRESSION 'zstd')",
                [str(destination)],
            )
            return
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("DuckDB parquet write failed (%s); falling back to pandas writer.", exc)
    try:
        df.to_parquet(destination, compression="zstd", index=False)
        return
    except Exception as exc:  # pragma: no cover - attempt with string-cast
        LOGGER.warning("Pandas parquet write failed (%s); retrying with string columns.", exc)
        safe = df.copy()
        object_columns = safe.select_dtypes(include="object").columns
        for column in object_columns:
            safe[column] = safe[column].astype(str)
        try:
            safe.to_parquet(destination, compression="zstd", index=False)
            return
        except Exception as exc2:  # pragma: no cover - last-resort fallback
            LOGGER.warning("String-coerced parquet write failed (%s); writing CSV instead.", exc2)
            csv_path = destination.with_suffix(".csv")
            safe.to_csv(csv_path, index=False)


def _pick_column(df: pd.DataFrame, options: Iterable[str]) -> str:
    """Return the first matching column from *options*, raising if none are found."""

    mapping = {str(col).strip().lower(): col for col in df.columns}
    for option in options:
        key = option.strip().lower()
        if key in mapping:
            return mapping[key]
    for key, original in mapping.items():
        if any(option.lower() in key for option in options):
            return original
    joined = ", ".join(options)
    raise KeyError(f"Column not found among {joined}")


def _set_project_baseline_cache(
    overall: dict[str, float],
    monthly: dict[str, dict[pd.Timestamp, float]],
    source: Path | None,
) -> None:
    """Store project baseline maps for reuse across the app."""

    global _PROJECT_BASELINE_OVERALL, _PROJECT_BASELINE_MONTHLY, _PROJECT_BASELINE_SOURCE
    _PROJECT_BASELINE_OVERALL = dict(overall)
    _PROJECT_BASELINE_MONTHLY = {project: dict(month_map) for project, month_map in monthly.items()}
    _PROJECT_BASELINE_SOURCE = Path(source) if source else None


def get_project_baseline_maps() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Return cached project baseline maps (overall and monthly)."""

    return (
        dict(_PROJECT_BASELINE_OVERALL),
        {project: dict(month_map) for project, month_map in _PROJECT_BASELINE_MONTHLY.items()},
    )


def _compute_project_baseline_maps(
    data: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Compute overall and monthly productivity baselines for each project."""

    if data.empty or "project_name" not in data or "daily_prod_mt" not in data:
        return {}, {}

    working = data.copy()
    working["project_name"] = working["project_name"].astype(str).str.strip()
    working["daily_prod_mt"] = pd.to_numeric(working["daily_prod_mt"], errors="coerce")
    working = working.dropna(subset=["project_name", "daily_prod_mt"])
    if working.empty:
        return {}, {}

    month_series = None
    if "month" in working.columns:
        month_series = pd.to_datetime(working["month"], errors="coerce")
        if month_series.notna().any():
            month_series = month_series.dt.to_period("M").dt.to_timestamp()
        else:
            month_series = None
    if month_series is None:
        if "date" in working.columns:
            month_series = pd.to_datetime(working["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        else:
            month_series = pd.Series(pd.NaT, index=working.index)
    working["__baseline_month"] = month_series

    overall_series = working.groupby("project_name")["daily_prod_mt"].mean().dropna()
    overall = {str(project): float(value) for project, value in overall_series.items() if not pd.isna(value)}

    monthly: dict[str, dict[pd.Timestamp, float]] = {}
    monthly_series = (
        working.dropna(subset=["__baseline_month"])
        .groupby(["project_name", "__baseline_month"])["daily_prod_mt"]
        .mean()
        .dropna()
    )
    for (project, month), value in monthly_series.items():
        month_ts = pd.to_datetime(month)
        if pd.isna(month_ts):
            continue
        monthly.setdefault(str(project), {})[pd.Timestamp(month_ts)] = float(value)

    return overall, monthly


def _parse_project_baseline_frames(
    df_overall: pd.DataFrame | None,
    df_monthly: pd.DataFrame | None,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Convert baseline dataframes into cached mapping structures."""

    overall: dict[str, float] = {}
    monthly: dict[str, dict[pd.Timestamp, float]] = {}

    if df_overall is not None and not df_overall.empty:
        try:
            project_col = _pick_column(df_overall, ("project_name", "Project Name"))
            baseline_col = _pick_column(df_overall, ("baseline_mt_per_day", "Baseline", "baseline"))
        except KeyError:
            pass
        else:
            cleaned = df_overall[[project_col, baseline_col]].copy()
            cleaned[project_col] = cleaned[project_col].astype(str).str.strip()
            cleaned[baseline_col] = pd.to_numeric(cleaned[baseline_col], errors="coerce")
            cleaned = cleaned.dropna(subset=[project_col, baseline_col])
            for _, row in cleaned.iterrows():
                name = str(row[project_col]).strip()
                value = float(row[baseline_col])
                if name:
                    overall[name] = value

    if df_monthly is not None and not df_monthly.empty:
        try:
            project_col = _pick_column(df_monthly, ("project_name", "Project Name"))
            month_col = _pick_column(df_monthly, ("month", "Month"))
            baseline_col = _pick_column(df_monthly, ("baseline_mt_per_day", "Baseline", "baseline"))
        except KeyError:
            pass
        else:
            cleaned = df_monthly[[project_col, month_col, baseline_col]].copy()
            cleaned[project_col] = cleaned[project_col].astype(str).str.strip()
            cleaned[baseline_col] = pd.to_numeric(cleaned[baseline_col], errors="coerce")
            cleaned[month_col] = pd.to_datetime(cleaned[month_col], errors="coerce")
            cleaned = cleaned.dropna(subset=[project_col, month_col, baseline_col])
            for _, row in cleaned.iterrows():
                project = str(row[project_col]).strip()
                month_ts = pd.to_datetime(row[month_col])
                value = float(row[baseline_col])
                if project and not pd.isna(month_ts):
                    monthly.setdefault(project, {})[pd.Timestamp(month_ts)] = value

    return overall, monthly


def _baseline_parquet_destination(data_path: Path, sheet_name: str) -> Path:
    root = data_path if data_path.is_dir() else data_path.parent
    return root / f"{sheet_name}.parquet"


def _persist_project_baselines(
    workbook_path: Path | None,
    overall: dict[str, float],
    monthly: dict[str, dict[pd.Timestamp, float]],
) -> None:
    """Persist baseline tables into the compiled workbook for fast reuse."""

    if workbook_path is None:
        return
    path = Path(workbook_path)

    overall_rows = [
        {"project_name": project, "baseline_mt_per_day": float(value)}
        for project, value in sorted(overall.items())
    ]
    overall_df = (
        pd.DataFrame(overall_rows)
        if overall_rows
        else pd.DataFrame(columns=["project_name", "baseline_mt_per_day"])
    )

    monthly_rows: list[dict[str, Any]] = []
    for project, month_map in monthly.items():
        for month, value in month_map.items():
            monthly_rows.append(
                {
                    "project_name": project,
                    "month": pd.to_datetime(month),
                    "baseline_mt_per_day": float(value),
                }
            )
    monthly_df = (
        pd.DataFrame(monthly_rows)
        if monthly_rows
        else pd.DataFrame(columns=["project_name", "month", "baseline_mt_per_day"])
    )
    if not monthly_df.empty:
        monthly_df["month"] = pd.to_datetime(monthly_df["month"], errors="coerce")
        monthly_df = monthly_df.dropna(subset=["month"]).sort_values(["project_name", "month"])

    if _parquet_dataset_available(path):
        try:
            _write_parquet(overall_df, _baseline_parquet_destination(path, PROJECT_BASELINES_SHEET))
            _write_parquet(
                monthly_df,
                _baseline_parquet_destination(path, PROJECT_BASELINES_MONTHLY_SHEET),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Failed to write project baselines to '%s' (parquet): %s",
                path,
                exc,
            )
        return

    if not path.exists():
        LOGGER.warning(
            "Cannot write project baselines because workbook '%s' is missing.",
            path,
        )
        return

    try:
        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            overall_df.to_excel(writer, PROJECT_BASELINES_SHEET, index=False)
            monthly_df.to_excel(writer, PROJECT_BASELINES_MONTHLY_SHEET, index=False)
    except FileNotFoundError:
        LOGGER.warning(
            "Workbook '%s' not found when attempting to persist project baselines.",
            path,
        )
    except PermissionError:
        LOGGER.warning(
            "Permission denied while writing project baselines to '%s'.",
            path,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "Failed to write project baselines to '%s': %s",
            path,
            exc,
        )


def _refresh_project_baselines(workbook_path: Path, data: pd.DataFrame) -> None:
    """Ensure project baseline sheets and caches reflect the current daily data."""

    if data.empty:
        load_project_baselines(workbook_path)
        return

    overall_map, monthly_map = _compute_project_baseline_maps(data)
    _set_project_baseline_cache(overall_map, monthly_map, workbook_path)
    _persist_project_baselines(workbook_path, overall_map, monthly_map)


def load_project_baselines(
    workbook_path: Path | str,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Load precomputed project baselines from storage, updating the cache."""

    path = Path(workbook_path)

    if _parquet_dataset_available(path):
        try:
            overall_source = _find_parquet_source(path, PROJECT_BASELINES_SHEET)
            monthly_source = _find_parquet_source(path, PROJECT_BASELINES_MONTHLY_SHEET)
            if not overall_source and not monthly_source:
                raise FileNotFoundError(f"No baseline parquet files found near '{path}'.")
            df_overall = _read_parquet(overall_source) if overall_source else None
            df_monthly = _read_parquet(monthly_source) if monthly_source else None
        except FileNotFoundError:
            LOGGER.warning(
                "Baseline parquet files not found near '%s'.",
                path,
            )
            _set_project_baseline_cache({}, {}, path)
            return get_project_baseline_maps()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Unable to load project baselines from '%s': %s",
                path,
                exc,
            )
            return get_project_baseline_maps()
        else:
            overall, monthly = _parse_project_baseline_frames(df_overall, df_monthly)
            _set_project_baseline_cache(overall, monthly, path)
            return get_project_baseline_maps()

    try:
        with pd.ExcelFile(path) as workbook:
            df_overall = (
                pd.read_excel(workbook, sheet_name=PROJECT_BASELINES_SHEET)
                if PROJECT_BASELINES_SHEET in workbook.sheet_names
                else None
            )
            df_monthly = (
                pd.read_excel(workbook, sheet_name=PROJECT_BASELINES_MONTHLY_SHEET)
                if PROJECT_BASELINES_MONTHLY_SHEET in workbook.sheet_names
                else None
            )
    except FileNotFoundError:
        LOGGER.warning(
            "Workbook '%s' not found when loading project baselines.",
            path,
        )
        _set_project_baseline_cache({}, {}, path)
        return get_project_baseline_maps()
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "Unable to load project baselines from '%s': %s",
            path,
            exc,
        )
        return get_project_baseline_maps()

    overall, monthly = _parse_project_baseline_frames(df_overall, df_monthly)
    _set_project_baseline_cache(overall, monthly, path)
    return get_project_baseline_maps()


def _coerce_mixed_excel_dates(values: Any) -> pd.Series:
    """Return datetime series parsed from Excel date serials or textual dates."""

    series = pd.Series(values)
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if isinstance(numeric, pd.Series):
        excel_mask = numeric.notna()
        if excel_mask.any():
            excel_mask &= numeric.between(20000, 80000)
            if excel_mask.any():
                excel_converted = pd.to_datetime(
                    numeric,
                    errors="coerce",
                    unit="D",
                    origin="1899-12-30",
                )
                parsed = parsed.where(~excel_mask, excel_converted)
    return parsed


def load_daily_from_proddailyexpanded(
    source: pd.DataFrame | pd.ExcelFile,
    sheet: str = "ProdDailyExpanded",
) -> pd.DataFrame:
    """Load daily productivity rows from a ProdDailyExpanded-style sheet."""

    LOGGER.debug("Loading data from sheet '%s'", sheet)
    if isinstance(source, pd.ExcelFile):
        df = pd.read_excel(source, sheet_name=sheet)
    else:
        df = source.copy()
    col_date = _pick_column(df, ["Work Date", "date"])
    col_prod = _pick_column(df, ["Productivity", "daily_prod_mt", "avg_daily_prod_mt"])
    col_proj = _pick_column(df, ["Project Name", "project_name"])
    col_gang = _pick_column(df, ["Gang name", "gang_name"])

    def _pick_optional(frame: pd.DataFrame, options: tuple[str, ...]) -> str | None:
        try:
            return _pick_column(frame, options)
        except KeyError:
            return None

    def _normalize_text(value: object) -> str:
        text = str(value).replace("\u00a0", " ").strip()
        lowered = text.lower()
        if lowered in {"", "nan", "none", "null"}:
            return ""
        return text

    data: dict[str, Any] = {
        "date": _coerce_mixed_excel_dates(df[col_date]).dt.normalize(),
        "daily_prod_mt": pd.to_numeric(df[col_prod], errors="coerce"),
        "project_name": df[col_proj].astype(str).str.strip(),
        "gang_name": df[col_gang].astype(str).str.strip(),
    }

    col_location = _pick_optional(df, ("Location No.", "location no", "location number", "location"))
    if col_location:
        data["location_no"] = df[col_location].map(_normalize_location)

    col_tower = _pick_optional(df, ("Tower Weight", "tower weight", "tower_weight", "tower wt", "tower mt"))
    if col_tower:
        data["tower_weight"] = pd.to_numeric(df[col_tower], errors="coerce")

    col_tower_type = _pick_optional(df, ("Tower Type", "Type of Tower", "tower type", "type of tower", "type"))
    if col_tower_type:
        data["tower_type"] = df[col_tower_type].map(_normalize_tower_type)

    col_start = _pick_optional(df, ("Start Date", "starting date"))
    if col_start:
        data["start_date"] = _coerce_mixed_excel_dates(df[col_start])

    col_complete = _pick_optional(df, ("Complete Date", "completion date"))
    if col_complete:
        data["completion_date"] = _coerce_mixed_excel_dates(df[col_complete])

    col_status = _pick_optional(df, ("Status",))
    if col_status:
        data["status"] = df[col_status].astype(str).str.strip()
    col_project_code = _pick_optional(df, ("Project Code", "project_code"))
    if col_project_code:
        data["project_code"] = df[col_project_code].astype(str).str.strip()
    col_line_name = _pick_optional(df, ("Line Name", "line_name"))
    if col_line_name:
        data["line_name"] = df[col_line_name].astype(str).str.strip()
    col_project_display = _pick_optional(df, ("Project Display", "project_display"))
    if col_project_display:
        data["project_display"] = df[col_project_display].astype(str).str.strip()
    col_scope_key = _pick_optional(df, ("Project Scope Key", "project_scope_key"))
    if col_scope_key:
        data["project_scope_key"] = df[col_scope_key].astype(str).str.strip()

    result = pd.DataFrame(data).dropna(subset=["date", "daily_prod_mt"])
    LOGGER.debug("Loaded %d daily rows from %s", len(result), sheet)
    return result


def load_daily_from_rawdata(source: pd.DataFrame | pd.ExcelFile, sheet: str = "RawData") -> pd.DataFrame:
    """Load daily productivity rows from a RawData sheet by expanding date ranges."""

    LOGGER.debug("Loading data from sheet '%s'", sheet)
    if isinstance(source, pd.ExcelFile):
        df = pd.read_excel(source, sheet_name=sheet)
    else:
        df = source.copy()
    start_col = _pick_column(df, ["Start Date", "starting date"])
    end_col = _pick_column(df, ["Complete Date", "completion date"])
    prod_col = _pick_column(df, ["Productivity", "avg_daily_prod_mt", "daily_prod_mt"])
    project_col = _pick_column(df, ["Project Name", "project_name"])
    gang_col = _pick_column(df, ["Gang name", "gang_name"])

    base = pd.DataFrame(
        {
            "start": _coerce_mixed_excel_dates(df[start_col]).dt.normalize(),
            "end": _coerce_mixed_excel_dates(df[end_col]).dt.normalize(),
            "daily_prod_mt": pd.to_numeric(df[prod_col], errors="coerce"),
            "project_name": df[project_col].astype(str).str.strip(),
            "gang_name": df[gang_col].astype(str).str.strip(),
        }
    ).dropna(subset=["start", "end", "daily_prod_mt"])

    tower_type_col = None
    for candidate in ("Tower Type", "Type of Tower", "tower type", "type of tower", "type"):
        if candidate in df.columns:
            tower_type_col = candidate
            break
    if tower_type_col:
        base["tower_type"] = df[tower_type_col].map(_normalize_tower_type)
    else:
        base["tower_type"] = ""
    for output_col, candidates in (
        ("project_code", ("Project Code", "project_code")),
        ("line_name", ("Line Name", "line_name")),
        ("project_display", ("Project Display", "project_display")),
        ("project_scope_key", ("Project Scope Key", "project_scope_key")),
    ):
        selected = None
        for candidate in candidates:
            if candidate in df.columns:
                selected = candidate
                break
        if selected:
            base[output_col] = df[selected].astype(str).str.strip()
        else:
            base[output_col] = ""
    rows: list[dict[str, object]] = []
    for _, record in base.iterrows():
        for date in pd.date_range(record["start"], record["end"], freq="D"):
                rows.append(
                    {
                        "date": date.normalize(),
                        "daily_prod_mt": record["daily_prod_mt"],
                        "project_name": record["project_name"],
                        "project_code": record.get("project_code", ""),
                        "line_name": record.get("line_name", ""),
                        "project_display": record.get("project_display", ""),
                        "project_scope_key": record.get("project_scope_key", ""),
                        "gang_name": record["gang_name"],
                        "tower_type": record.get("tower_type", ""),
                    }
                )
    LOGGER.debug("Expanded raw data into %d daily rows", len(rows))
    return pd.DataFrame(rows)


def _load_daily_via_duckdb(data_path: Path, preferred_sheet: str | None) -> pd.DataFrame | None:
    if not _parquet_dataset_available(data_path):
        return None

    candidates: list[str] = []
    if preferred_sheet:
        candidates.append(preferred_sheet)
    candidates.extend(["ProdDailyExpandedSingles", "ProdDailyExpanded"])

    for sheet_name in candidates:
        source = _find_parquet_source(data_path, sheet_name)
        if source:
            df = _read_parquet(source)
            LOGGER.debug("Loaded daily data via DuckDB from '%s' (%s)", data_path, sheet_name)
            return load_daily_from_proddailyexpanded(df, sheet_name)

    raw_source = _find_parquet_source(data_path, "RawData")
    if raw_source:
        df_raw = _read_parquet(raw_source)
        LOGGER.debug("Loaded raw daily data via DuckDB from '%s' (RawData)", data_path)
        return load_daily_from_rawdata(df_raw, sheet="RawData")

    return None


def _load_daily_via_excel(data_path: Path, preferred_sheet: str | None) -> pd.DataFrame:
    target = data_path
    if data_path.is_dir():
        excel_candidates = sorted(data_path.glob("*.xls*"))
        if not excel_candidates:
            raise FileNotFoundError(f"No Excel workbooks found in '{data_path}'.")
        target = excel_candidates[0]

    with pd.ExcelFile(target) as workbook:
        candidates: list[str] = []
        if preferred_sheet:
            candidates.append(preferred_sheet)
        candidates.extend(["ProdDailyExpandedSingles"])

        result: pd.DataFrame | None = None
        seen: set[str] = set()
        for sheet_name in candidates:
            if sheet_name and sheet_name not in seen and sheet_name in workbook.sheet_names:
                LOGGER.debug("Loaded daily data from Excel sheet '%s' in '%s'", sheet_name, target)
                result = load_daily_from_proddailyexpanded(workbook, sheet_name)
                break
            seen.add(sheet_name)

        if result is None and "RawData" in workbook.sheet_names:
            LOGGER.debug("Falling back to RawData sheet in '%s'", target)
            result = load_daily_from_rawdata(workbook, "RawData")

    if result is None:
        raise FileNotFoundError("Neither 'ProdDailyExpandedSingles' nor fallback sheets found in workbook.")
    return result


@_ttl_lru_cache(maxsize=CACHE_MAXSIZE, ttl_seconds=CACHE_TTL_SECONDS)
def _load_daily_cached(data_path: str, preferred_sheet: str) -> pd.DataFrame:
    path = Path(data_path)
    sheet = preferred_sheet or None

    duckdb_df = _load_daily_via_duckdb(path, sheet)
    if duckdb_df is not None:
        return duckdb_df

    if _parquet_dataset_available(path):
        raise FileNotFoundError(f"Parquet dataset for daily productivity not found near '{path}'.")
    LOGGER.debug("Parquet dataset not available for '%s'; using Excel fallback.", path)
    return _load_daily_via_excel(path, sheet)


def load_daily(config_or_path: AppConfig | Path | str) -> pd.DataFrame:
    """Load daily productivity data from a config or explicit path."""

    if isinstance(config_or_path, AppConfig):
        config = config_or_path
    else:
        workbook_path = Path(config_or_path)
        config = AppConfig(data_path=workbook_path)

    LOGGER.info("Loading dataset '%s'", config.data_path)

    resolved = str(Path(config.data_path).resolve())
    try:
        cached_df = _load_daily_cached(resolved, config.preferred_sheet or "")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Unable to locate productivity dataset at '{config.data_path}'."
        ) from exc

    result = cached_df.copy()
    return result


load_daily.cache_clear = _load_daily_cached.cache_clear  # type: ignore[attr-defined]


def _pick_tol(df: pd.DataFrame, opts):
    m = {str(c).strip().lower(): c for c in df.columns}
    for o in opts:
        key = o.strip().lower()
        if key in m:
            return m[key]
    for k, c in m.items():
        if any(o.lower() in k for o in opts):
            return c
    raise KeyError(f"Column not found among {opts}: have {list(df.columns)}")


def _prepare_project_details(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    try:
        # Robust column resolver: treat spaces/underscores/dashes equally
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

        # Build lookup maps once
        cols = list(df.columns)
        map_exact = {str(c).strip().lower(): c for c in cols}
        map_norm = {_norm(c): c for c in cols}

        def pick_any(options: list[str]) -> str:
            for o in options:
                key = str(o).strip().lower()
                if key in map_exact:
                    return map_exact[key]
            for o in options:
                nkey = _norm(o)
                if nkey in map_norm:
                    return map_norm[nkey]
            # last resort: contains on normalized keys
            for o in options:
                nkey = _norm(o)
                for k_norm, orig in map_norm.items():
                    if nkey and nkey in k_norm:
                        return orig
            raise KeyError(options)

        # Accept both snake_case and human labels from the DPR sheet
        col_code   = pick_any(["project_code", "Project Code", "ProjectCode"])  # often present in compiled
        col_name   = pick_any(["project_name", "Project Name"])                  # DPR sheet uses 'Project Name'
        col_client = pick_any(["client_name", "Client Name", "Client"])         
        col_noa    = pick_any(["noa_start", "NOA Start", "NOA Date", "NOA"])
        col_loa    = pick_any(["loa_end", "LOA End", "LOA Date", "LOA"])
        col_pe     = pick_any(["planning_eng", "Planning Engineer"])            
        col_pch    = pick_any(["pch", "PCH"])                                   
        col_rm     = pick_any(["regional_mgr", "Regional Manager"])             
        col_pm     = pick_any(["project_mgr", "Project Manager"])               
        col_si     = pick_any(["section_inch", "Section Incharge", "Section Incharge/Engineer", "Section Incharge/Engg"])
        col_sup    = pick_any(["supervisor", "Supervisors", "Supervisor"])      

        out = pd.DataFrame({
            "project_code": df[col_code].astype(str).str.strip(),
            "project_name": df[col_name].astype(str).str.strip(),
            "client_name": df[col_client].astype(str).str.strip(),
            "noa_start": pd.to_datetime(df[col_noa], errors="coerce"),
            "loa_end": pd.to_datetime(df[col_loa], errors="coerce"),
            "planning_eng": df[col_pe].astype(str).str.strip(),
            "pch": df[col_pch].astype(str).str.strip(),
            "regional_mgr": df[col_rm].astype(str).str.strip(),
            "project_mgr": df[col_pm].astype(str).str.strip(),
            "section_inch": df[col_si].astype(str).str.strip(),
            "supervisor": df[col_sup].astype(str).str.strip(),
        })
        out = out[(out["project_name"] != "nan") | (out["project_code"] != "nan")].copy()
        out["key_name"] = out["project_name"].str.lower().str.replace(r"\s+", " ", regex=True)
        # Preserve original DPR label if present for clarity
        # Recover actual source column for 'Project Name' if present under any spacing/case variant
        try:
            src_col = next(c for c in df.columns if _norm(c) == _norm("Project Name"))
            out["Project Name"] = df[src_col].astype(str).str.strip()
        except StopIteration:
            pass
        return out
    except Exception:
        return pd.DataFrame()


def _resolve_project_pch_path() -> Path | None:
    repo_root = _repo_root_from(Path(__file__))
    candidate = repo_root / "Raw Data" / "Projects and PCH.xlsx"
    if candidate.exists():
        return candidate
    return None


def _load_project_pch_mapping(mapping_path: Path | None) -> pd.DataFrame:
    if mapping_path is None:
        return pd.DataFrame()

    try:
        frame = pd.read_excel(mapping_path)
    except FileNotFoundError:
        LOGGER.warning("Projects/PCH workbook was not found at %s; skipping fallback.", mapping_path)
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected formats
        LOGGER.warning("Unable to read Projects/PCH workbook '%s': %s", mapping_path, exc)
        return pd.DataFrame()

    if frame.empty:
        LOGGER.warning("Projects/PCH workbook at %s is empty; skipping fallback.", mapping_path)
        return pd.DataFrame()

    def _match_column(keywords: tuple[str, ...]) -> str:
        for column in frame.columns:
            label = str(column).strip().lower()
            if any(keyword in label for keyword in keywords):
                return column
        raise KeyError

    try:
        project_col = _match_column(("project",))
        pch_col = _match_column(("pch",))
    except KeyError:
        LOGGER.warning(
            "Projects/PCH workbook '%s' is missing required columns (need both project and PCH); skipping fallback.",
            mapping_path,
        )
        return pd.DataFrame()

    name_col = None
    for column in frame.columns:
        label = str(column).strip().lower()
        if "project name" in label:
            name_col = column
            break

    cols = [project_col, pch_col]
    if name_col:
        cols.append(name_col)
    working = frame[cols].copy()
    rename = {project_col: "project_code", pch_col: "pch"}
    if name_col:
        rename[name_col] = "project_name"
    working = working.rename(columns=rename)

    def _clean_text(value: object) -> str:
        text = "" if value is None else str(value).strip()
        lowered = text.lower()
        if lowered in {"", "nan", "none", "null"}:
            return ""
        return text

    working["project_code"] = working["project_code"].map(_clean_text)
    working["pch"] = working["pch"].map(_clean_text)
    if "project_name" in working.columns:
        working["project_name"] = working["project_name"].map(_clean_text)
    else:
        working["project_name"] = ""

    working["project_name"] = working["project_name"].where(
        working["project_name"].astype(bool),
        working["project_code"],
    )

    working = working[(working["project_code"].astype(bool)) & (working["pch"].astype(bool))]
    if working.empty:
        LOGGER.warning(
            "Projects/PCH workbook '%s' has no usable rows after cleaning; skipping fallback.",
            mapping_path,
        )
        return pd.DataFrame()

    working["key_name"] = working["project_name"].str.lower().str.replace(r"\s+", " ", regex=True)
    LOGGER.info("Loaded %d project-to-PCH mappings from %s", len(working), mapping_path)
    return working.reset_index(drop=True)


def _augment_project_details_with_pch(details: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if details is None or details.empty:
        base_cols = [
            "project_code",
            "project_name",
            "client_name",
            "noa_start",
            "loa_end",
            "planning_eng",
            "pch",
            "regional_mgr",
            "project_mgr",
            "section_inch",
            "supervisor",
            "key_name",
        ]
        details = pd.DataFrame(columns=base_cols)

    if mapping is None or mapping.empty:
        return details.copy()

    work = details.copy()

    def _clean_text(value: object) -> str:
        text = "" if value is None else str(value).strip()
        lowered = text.lower()
        if lowered in {"", "nan", "none", "null"}:
            return ""
        return text

    def _compact(value: object) -> str:
        return re.sub(r"[^a-z0-9]", "", _clean_text(value).lower())

    for column in ("project_code", "project_name", "pch"):
        if column in work.columns:
            work[column] = work[column].map(_clean_text)
        else:
            work[column] = ""

    work["project_code_norm"] = work["project_code"].map(_compact)
    work["project_name_norm"] = work["project_name"].map(_compact)

    mapping = mapping.copy()
    mapping["project_code_norm"] = mapping["project_code"].map(_compact)
    mapping["project_name_norm"] = mapping["project_name"].map(_compact)

    code_to_pch = (
        mapping.dropna(subset=["project_code_norm", "pch"])
        .drop_duplicates(subset=["project_code_norm"])
        .set_index("project_code_norm")["pch"]
        .to_dict()
    )
    name_to_pch = (
        mapping.dropna(subset=["project_name_norm", "pch"])
        .drop_duplicates(subset=["project_name_norm"])
        .set_index("project_name_norm")["pch"]
        .to_dict()
    )

    missing_mask = work["pch"].map(_clean_text).eq("")
    if missing_mask.any():
        filled = work.loc[missing_mask, "project_code_norm"].map(code_to_pch).fillna("")
        work.loc[missing_mask, "pch"] = filled
        missing_mask = work["pch"].map(_clean_text).eq("")
        if missing_mask.any():
            filled = work.loc[missing_mask, "project_name_norm"].map(name_to_pch).fillna("")
            work.loc[missing_mask, "pch"] = filled

    existing_codes = set(work["project_code_norm"].dropna())
    existing_names = set(work["project_name_norm"].dropna())
    extra_rows = []
    for _, row in mapping.iterrows():
        code_norm = row.get("project_code_norm", "")
        name_norm = row.get("project_name_norm", "")
        if code_norm and code_norm in existing_codes:
            continue
        if name_norm and name_norm in existing_names:
            continue
        project_code = row.get("project_code", "")
        project_name = row.get("project_name", "") or project_code
        new_row = {col: "" for col in work.columns}
        new_row["project_code"] = project_code
        new_row["project_name"] = project_name
        new_row["pch"] = row.get("pch", "")
        if "key_name" in work.columns:
            new_row["key_name"] = str(project_name).lower().replace("  ", " ").strip()
        if "Project Name" in work.columns:
            new_row["Project Name"] = project_name
        if "noa_start" in work.columns:
            new_row["noa_start"] = pd.NaT
        if "loa_end" in work.columns:
            new_row["loa_end"] = pd.NaT
        extra_rows.append(new_row)

    if extra_rows:
        work = pd.concat([work, pd.DataFrame(extra_rows)], ignore_index=True)

    work = work.drop(columns=["project_code_norm", "project_name_norm"], errors="ignore")
    return work


@_ttl_lru_cache(maxsize=CACHE_MAXSIZE, ttl_seconds=CACHE_TTL_SECONDS)
def _load_project_details_cached(data_path: str, sheet: str) -> pd.DataFrame:
    path = Path(data_path)
    if _parquet_dataset_available(path):
        source = _find_parquet_source(path, sheet)
        if not source:
            return pd.DataFrame()
        df = _read_parquet(source)
    else:
        try:
            with pd.ExcelFile(path) as xl:
                if sheet not in xl.sheet_names:
                    return pd.DataFrame()
                df = pd.read_excel(xl, sheet_name=sheet)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.debug("Failed to load project details from '%s': %s", path, exc)
            return pd.DataFrame()
    return _prepare_project_details(df)


def load_project_details(path: Path, sheet: str = "ProjectDetails") -> pd.DataFrame:
    cached = _load_project_details_cached(str(Path(path).resolve()), sheet)
    mapping_path = _resolve_project_pch_path()
    mapping = _load_project_pch_mapping(mapping_path)
    return _augment_project_details_with_pch(cached, mapping).copy()


load_project_details.cache_clear = _load_project_details_cached.cache_clear  # type: ignore[attr-defined]


def _normalize_location(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ").strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    return text


def _normalize_tower_type(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper().replace("\u00a0", " ")
    if not text or text in {"NAN", "NA", "NONE"}:
        return ""
    compact = re.sub(r"\s+", "", text)
    base_match = re.search(r"(DA|DB|DC|DD)", compact)
    if base_match:
        compact = compact[base_match.start():]
    match = re.match(r"^(DA|DB|DC|DD)(?:\+?(\d+))?$", compact)
    if match:
        base = match.group(1)
        ext = match.group(2)
        ext_value = str(int(ext)) if ext is not None else "0"
        return f"{base}+{ext_value}"
    return compact
