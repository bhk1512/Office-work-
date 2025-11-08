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
    parse_project_code_from_filename,
)

CONFIG = AppConfig()

CACHE_TTL_SECONDS = CONFIG.cache_ttl_seconds
CACHE_MAXSIZE = CONFIG.cache_maxsize

LOGGER = logging.getLogger(__name__)

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
    # Fallbacks
    cand1 = (repo_root / "Raw Data" / "DPRs").resolve()
    if cand1.exists():
        print(f"[Stringing] RAW root fallback <repo>/Raw Data/DPRs: {cand1}")
        return cand1
    cand2 = (Path(data_path_hint) / "Raw Data" / "DPRs").resolve()
    print(f"[Stringing] RAW root ultimate fallback: {cand2}")
    return cand2


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

def _match_sheet_name(xl: "pd.ExcelFile", desired_sheet: str) -> str | None:
    """
    Return the actual sheet name from the workbook whose normalized form
    exactly equals the normalized desired_sheet. No aliases.
    """
    want = _norm_sheet(desired_sheet)
    if not want:
        return None
    for s in xl.sheet_names:
        if _norm_sheet(s) == want:
            return s  # return the exact name as present in the file
    return None

def _project_from_filename(name: str) -> str | None:
    """Parse TA/TB + digits from a filename -> 'TA 415' / 'TB 408'."""
    if not name:
        return None
    m = _PROJECT_RE.search(Path(name).name.upper())
    return f"{m.group(1)} {m.group(2)}" if m else None

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

def _excel_has_sheet(xlsx_path: Path, desired_sheet: str) -> bool:
    try:
        with pd.ExcelFile(xlsx_path) as xl:
            return desired_sheet in xl.sheet_names
    except Exception:
        return False



def _concat_union(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate with union of columns, preserving the first DF's order."""
    if not dfs:
        return pd.DataFrame()
    cols = list(dfs[0].columns)
    for df in dfs[1:]:
        for c in df.columns:
            if c not in cols:
                cols.append(c)
    aligned = [df.reindex(columns=cols) for df in dfs]
    return pd.concat(aligned, ignore_index=True)

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

    # Stringing output root (flat structure under <repo>/Parquets/Stringing)
    out_root = _stringing_root(raw_root)

    candidates = _iter_excel_candidates(raw_root)
    compiled_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    diag_rows: list[dict] = []

    if not candidates:
        LOGGER.warning("Stringing: no Excel files found under '%s'", raw_root)

    for wb in candidates:
        proj = _project_from_filename(wb.name) or "UNKNOWN"

        # If the sheet does not exist, log and continue (so Diagnostics shows it)
        if not _excel_has_sheet(wb, sheet_name):
            diag_rows.append({
                "Workbook": wb.name,
                "Project": proj,
                "Rows": 0,
                "DailyRows": 0,
                "Status": "NO_TARGET_SHEET"
            })
            continue

        # Try robust parse of the desired sheet
        try:
            df_raw = read_stringing_sheet_robust(str(wb), sheet_name)
        except Exception as exc:
            LOGGER.warning("Stringing: failed reading '%s' [%s]: %s", wb, sheet_name, exc)
            diag_rows.append({
                "Workbook": wb.name,
                "Project": proj,
                "Rows": 0,
                "DailyRows": 0,
                "Status": f"READ_FAIL: {type(exc).__name__}"
            })
            continue

        if df_raw is None or df_raw.empty:
            diag_rows.append({
                "Workbook": wb.name,
                "Project": proj,
                "Rows": 0,
                "DailyRows": 0,
                "Status": "EMPTY_SHEET"
            })
            continue

        # Normalize columns
        try:
            compiled_norm, _ = normalize_stringing_columns(df_raw)
        except Exception:
            compiled_norm = df_raw.copy()

        # Inject project/source for traceability (fill blank project_name)
        if "project_name" not in compiled_norm.columns:
            compiled_norm["project_name"] = proj
        else:
            s = compiled_norm["project_name"].astype(str).str.strip()
            compiled_norm["project_name"] = s.mask(s.eq("") | s.eq("nan") | s.eq("None"), proj)
        if "project" not in compiled_norm.columns:
            compiled_norm["project"] = compiled_norm["project_name"]
        if "source_file" not in compiled_norm.columns:
            compiled_norm["source_file"] = wb.name

        # Expand to daily
        try:
            daily = expand_stringing_to_daily(compiled_norm)
        except Exception:
            daily = pd.DataFrame()

        compiled_frames.append(compiled_norm)
        if not daily.empty:
            # Keep project/source on daily rows as well
            if "project_name" not in daily.columns:
                daily["project_name"] = compiled_norm["project_name"].iloc[0]
            if "project" not in daily.columns:
                daily["project"] = daily["project_name"]
            if "source_file" not in daily.columns:
                daily["source_file"] = wb.name
            daily_frames.append(daily)
            diag_rows.append({
                "Workbook": wb.name,
                "Project": proj,
                "Rows": int(len(compiled_norm)),
                "DailyRows": int(len(daily)),
                "Status": "OK"
            })
        else:
            diag_rows.append({
                "Workbook": wb.name,
                "Project": proj,
                "Rows": int(len(compiled_norm)),
                "DailyRows": 0,
                "Status": "NO_DAILY"
            })

    compiled_all = _concat_union(compiled_frames) if compiled_frames else pd.DataFrame()
    daily_all    = _concat_union(daily_frames) if daily_frames else pd.DataFrame()

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

    # Combined Excel (fallback/log) — always write; Diagnostics includes *all* files scanned
    try:
        with pd.ExcelWriter(out_root / "StringingCompiled_Output.xlsx", engine="openpyxl", mode="w") as xw:
            (compiled_all if not compiled_all.empty else pd.DataFrame()).to_excel(
                xw, sheet_name=sheet_name, index=False
            )
            if not daily_all.empty:
                daily_all.to_excel(xw, sheet_name="Daily", index=False)
            pd.DataFrame(diag_rows).to_excel(xw, sheet_name="Diagnostics", index=False)
    except Exception as exc:
        LOGGER.warning("Stringing: failed writing combined Excel: %s", exc)

    print(f"[Stringing] RAW Root       : {raw_root}")
    print(f"[Stringing] Out Root       : {out_root}")
    print(f"[Stringing] Master Compiled: {master_compiled}")
    print(f"[Stringing] Master Daily   : {master_daily}")

    return compiled_all, daily_all, out_root



# --- NEW: ensure project_name is present on a stringing frame ---
def _ensure_stringing_project_name(df: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """
    If 'project_name' is missing/blank in df, fill it by parsing the project
    code from the Excel/parquet file name (e.g., TA 415 / TB 408).
    Also sets a 'project' mirror column and 'source_file' for auditability.
    """
    try:
        from .stringing import parse_project_code_from_filename
    except Exception:
        # Fallback to the local regex helper if needed
        def parse_project_code_from_filename(name: str) -> str | None:
            m = _PROJECT_RE.search(str(name).upper())
            return f"{m.group(1)} {m.group(2)}" if m else None

    df = df.copy()
    # Where to parse from: prefer the file's name; if it's a directory, still try its name.
    guess = parse_project_code_from_filename(base_path.name)
    if not guess:
        # One more try with the full path string in case the code is in a parent dir
        guess = parse_project_code_from_filename(str(base_path))

    # Add the source file for traceability (safe no-op if already present)
    if "source_file" not in df.columns:
        df["source_file"] = base_path.name

    if not guess:
        return df

    # Normalize and fill project_name
    if "project_name" not in df.columns:
        df["project_name"] = guess
    else:
        s = df["project_name"].astype(str).str.strip()
        df["project_name"] = s.mask(s.eq("") | s.eq("nan") | s.eq("None"), guess)

    # Keep a 'project' mirror (some parts of the app look for this)
    if "project" not in df.columns:
        df["project"] = df["project_name"]
    else:
        s = df["project"].astype(str).str.strip()
        df["project"] = s.mask(s.eq("") | s.eq("nan") | s.eq("None"), df["project_name"])

    return df


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
            actual = _match_sheet_name(xl, sheet_name)
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

    # --- NEW: inject project name into both raw + normalized from file name ---
    source_name = str(Path(base).name)
    proj = parse_project_code_from_filename(source_name)

    if proj:
        for fr in (df_raw, normalized):
            if "project_name" not in fr.columns:
                fr["project_name"] = proj
            else:
                s = fr["project_name"].astype(str).str.strip()
                fr["project_name"] = s.mask(s.eq("") | s.eq("nan") | s.eq("None"), proj)
            if "project" not in fr.columns:
                fr["project"] = fr["project_name"]

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
            "Rules": "Dates parsed with pandas to_datetime (coerce); PO start to F/S complete inclusive; basic column normalization applied.",
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
def _load_stringing_compiled_raw_cached(data_path: str, sheet_name: str, probe_dirs: tuple[str, ...]) -> pd.DataFrame:
    """
    New behavior: scan RAW DPRs each run and rebuild the master compiled parquet.
    """
    raw_root = _resolve_stringing_raw_root(Path(data_path))
    _, daily_all, _ = build_stringing_artifacts_every_run(raw_root, sheet_name)
    compiled_all, _, _ = build_stringing_artifacts_every_run(raw_root, sheet_name)
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


@_ttl_lru_cache(maxsize=CACHE_MAXSIZE, ttl_seconds=CACHE_TTL_SECONDS)
def _load_stringing_daily_cached(data_path: str, sheet_name: str, probe_dirs: tuple[str, ...], daily_table: str = "StringingDaily") -> pd.DataFrame:
    """
    New behavior: scan RAW DPRs each run and rebuild the master daily parquet.
    """
    raw_root = _resolve_stringing_raw_root(Path(data_path))
    _, daily_all, _ = build_stringing_artifacts_every_run(raw_root, sheet_name)
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


def _write_parquet(df: pd.DataFrame, destination: Path) -> None:
    """Persist *df* to *destination* using DuckDB for consistent parquet writes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with duckdb.connect(database=":memory:") as con:
        con.register("df_to_write", df)
        con.execute(
            "COPY df_to_write TO ? (FORMAT 'parquet', COMPRESSION 'zstd')",
            [str(destination)],
        )


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

    def _normalize_location(value: object) -> str:
        text = _normalize_text(value)
        if not text:
            return ""
        if text.endswith(".0") and text.replace(".", "", 1).isdigit():
            text = text.split(".", 1)[0]
        return text

    data: dict[str, Any] = {
        "date": pd.to_datetime(df[col_date], errors="coerce").dt.normalize(),
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

    col_start = _pick_optional(df, ("Start Date", "starting date"))
    if col_start:
        data["start_date"] = pd.to_datetime(df[col_start], errors="coerce")

    col_complete = _pick_optional(df, ("Complete Date", "completion date"))
    if col_complete:
        data["completion_date"] = pd.to_datetime(df[col_complete], errors="coerce")

    col_status = _pick_optional(df, ("Status",))
    if col_status:
        data["status"] = df[col_status].astype(str).str.strip()

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
            "start": pd.to_datetime(df[start_col], errors="coerce"),
            "end": pd.to_datetime(df[end_col], errors="coerce"),
            "daily_prod_mt": pd.to_numeric(df[prod_col], errors="coerce"),
            "project_name": df[project_col].astype(str).str.strip(),
            "gang_name": df[gang_col].astype(str).str.strip(),
        }
    ).dropna(subset=["start", "end", "daily_prod_mt"])
    rows: list[dict[str, object]] = []
    for _, record in base.iterrows():
        for date in pd.date_range(record["start"], record["end"], freq="D"):
            rows.append(
                {
                    "date": date.normalize(),
                    "daily_prod_mt": record["daily_prod_mt"],
                    "project_name": record["project_name"],
                    "gang_name": record["gang_name"],
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
    _refresh_project_baselines(Path(config.data_path), result)
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
    return cached.copy()


load_project_details.cache_clear = _load_project_details_cached.cache_clear  # type: ignore[attr-defined]
