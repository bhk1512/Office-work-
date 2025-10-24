# microplan_compile.py
# ------------------------------------------------------------
# Build a tidy, analysis-ready "Micro Plan" responsibilities table
# and write it into the compiled workbook.
#
# What it does:
#   - Discovers files whose name contains "Micro Plan" (case-insensitive)
#   - Auto-detects the header row within the first 50 rows
#   - Canonicalizes columns to a stable schema (easy to tweak below)
#   - Aggregates responsibilities (sum of Revenue, Tower Weight) for:
#       Gang / Section Incharge / Supervisor
#   - Writes a single tidy sheet "MicroPlanResponsibilities"
#   - Optionally writes per-project cleaned micro plan sheets
#   - Writes an index sheet for traceability ("MicroPlanIndex")
#
# How to use (CLI):
#   python microplan_compile.py --input-dir "<folder with project files>" \
#                               --output-xlsx "ErectionCompiled_Output.xlsx"
#
# How to use (import):
#   from microplan_compile import compile_microplans_to_workbook
#   compile_microplans_to_workbook(input_dir="...", output_path="...")
#
# Dependencies:
#   pip install pandas openpyxl
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import re
from glob import glob
from typing import Optional

import pandas as pd
from openpyxl import load_workbook

# =========================
# === Config (Editable) ===
# =========================

# --- Micro Plan: schema config (edit here when your header standard changes) ---
MICROPLAN_SCHEMA_V1 = {
    "required": {
        "gang_name":        ["gang name"],
        "section_incharge": ["section incharge"],
        "supervisor":       ["supervisor"],
        "revenue_planned":   ["revenue planned", "revenue"],
        "revenue_realised":  ["revenue realised", "revenue realized", "revenue realization"],
        "location_no":       ["location no.", "loc no", "loc no.", "location no"],

    },
    "optional": {
        "tower_weight": ["tower weight"],
        "tower_type": ["tower type"],
        "manpower":            ["manpower"],
        "power_tools_issued":  ["power tools issued (yes/no)", "power tools issued"],
        "material_feeding":    ["material feeding", "matl feeding"],
        "starting_date":       ["starting date", "start date"],
        "completion_date":     ["completion date", "end date"],
        "tack_welding":        ["tack-welding", "tack welding"],
        "final_checking":      ["final checking", "final check"],
    }
}
CURRENT_MICROPLAN_SCHEMA = MICROPLAN_SCHEMA_V1


# Output sheet names
MICROPLAN_AGG_SHEET_NAME = "MicroPlanResponsibilities"
MICROPLAN_INDEX_SHEET    = "MicroPlanIndex"

# Search pattern for "Micro Plan" files
MICROPLAN_GLOB_PATTERN = "**/*micro*plan*.xls*"

# Header search window
HEADER_SCAN_MAX_ROWS = 50
MICROPLAN_LEFT_WIDTH = 30

# ==========================
# === Helper / Utilities ===
# ==========================

def _norm_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"[\s\-_./()]+", " ", x)
    x = re.sub(r"[^a-z0-9 %]+", "", x)
    return " ".join(x.split())

_DETECTION_TOKENS_RAW = {
    # core entity/metrics
    "gang name", "section incharge", "supervisor",
    "tower weight", "revenue planned", "revenue realised", "revenue realized",
    "location no", "loc no",
    # helpful supporting headers (boosts scoring)
    # "sl", "loc no.", "tower type", "manpower",
    # "power tools issued (yes/no)", "matl feeding",
    # "starting date", "completion date", "tack-welding", "final checking",
}

_DETECTION_TOKENS = {_norm_text(s) for s in _DETECTION_TOKENS_RAW}

def _pick_erection_sheet(path: str) -> str:
    """
    Return the sheet whose name contains 'erection' (case-insensitive).
    If none found, fall back to the first sheet.
    """
    wb = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    
    for ws in wb.worksheets:
        if "erection" in ws.title.strip().lower():
            return ws.title
    # fallback: first visible
    return wb.worksheets[0].title


def _detect_header_row(path: str, sheet: str, max_rows: int = HEADER_SCAN_MAX_ROWS) -> int:
    """
    Detect the header row by scanning the first `max_rows` rows without any
    index-based column limits. If you want to ignore far-right noise, we slice
    *after* reading instead of using `usecols`, which can go out of bounds on
    narrow sheets.
    """
    prev = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        nrows=max_rows,
        engine="openpyxl",
        dtype=object,
    )

    best_row, best_score = None, -1

    # Evaluate each candidate row; only consider the first MICROPLAN_LEFT_WIDTH cells
    width = MICROPLAN_LEFT_WIDTH if isinstance(MICROPLAN_LEFT_WIDTH, int) and MICROPLAN_LEFT_WIDTH > 0 else prev.shape[1]

    for r in range(len(prev)):
        # slice row to left window, coerce to strings safely
        row_vals = prev.iloc[r].tolist()[:width]
        toks = {_norm_text(v) for v in row_vals if v is not None and str(v).strip() != ""}
        score = sum(1 for t in _DETECTION_TOKENS if t in toks)
        if score > best_score:
            best_row, best_score = r, score

    # Require a minimal confidence to avoid false positives
    if best_row is None or best_score < 3:
        raise ValueError(f"Header row not found on sheet '{sheet}' (best_score={best_score}).")
    return best_row







def _alias_to_canonical(colname: str, schema: dict) -> Optional[str]:
    """Return canonical key (gang_name, supervisor, etc.) if colname matches an alias."""
    n = _norm_text(colname)
    for canon, aliases in {**schema["required"], **schema.get("optional", {})}.items():
        if any(n == _norm_text(a) for a in aliases):
            return canon
    return None




def infer_project_name_from_filename(path: str) -> str:
    """
    Extract project code from filenames like:
    'Micro Plan - TA-416 Sep'25.xlsx' → 'TA416'
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    # Strip the "micro plan" words
    s = re.sub(r"(?i)\bmicro\s*plan\b", " ", stem)

    # Remove month/year tokens (Sep'25, Oct 2025, etc.)
    s = re.sub(r"(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)['\-\s]*\d{2,4}", " ", s)
    s = re.sub(r"\b20\d{2}\b", " ", s)
    s = re.sub(r"\b\d{2}\b", " ", s)

    s = " ".join(s.split())

    # Extract first letter+3 digit code, remove dashes/spaces
    m = re.search(r"(?i)\b([A-Z]{1,3})[\s\-_]*([0-9]{3})\b", s)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"

    return stem



def normalize_key(s: str) -> str:
    """Stable snake_key → 'TA416' → 'ta416'"""
    return re.sub(r"[^a-z0-9]+", "", _norm_text(s))


def _safe_read_excel_by_header(path: str, sheet_name: str, header_row: int) -> pd.DataFrame:
    """
    Read a sheet using the detected header row. Avoid index-based `usecols`;
    we'll trim by names (and optionally left width) after reading.
    """
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=header_row,
        dtype=object,
        engine="openpyxl",
    )
    # standardize NA and strip column label whitespace
    df = df.where(df.notna(), None)
    df.columns = [str(c).strip() if c is not None else "" for c in df.columns]
    return df

def read_microplan_file(path: str,
                        schema: dict,
                        header_scan_rows: int = HEADER_SCAN_MAX_ROWS) -> pd.DataFrame:
    sheet = _pick_erection_sheet(path)
    header_row = _detect_header_row(path,sheet, max_rows=header_scan_rows)

    df = _safe_read_excel_by_header(path, sheet, header_row)

     # --- NEW TRUNCATION LOGIC ---
    # Find first completely empty row after header
    mask_empty = df.isna().all(axis=1)
    if mask_empty.any():
        first_empty_idx = mask_empty.idxmax()  # index of first True
        # truncate everything after this row
        df = df.loc[:first_empty_idx-1]

    rename_map = {}
    for col in df.columns:
        canon = _alias_to_canonical(col, schema)
        if canon:
            rename_map[col] = canon
    df = df.rename(columns=rename_map)

    needed = ["gang_name", "section_incharge", "supervisor", "tower_weight", "revenue_planned", "revenue_realised", "location_no"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}. Columns read: {list(df.columns)}")


    for col in ["gang_name", "section_incharge", "supervisor"]:
        if col in df.columns:
            df[col] = df[col].ffill().astype(str).str.strip()
    
    # location as stable string id (not numeric), trimmed
    df["location_no"] = df["location_no"].astype(object).map(
        lambda x: str(x).strip() if x is not None and str(x).strip() != "" else None
    )

    for c in ("tower_weight", "revenue_planned", "revenue_realised"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    mask = (df["revenue_planned"].fillna(0) == 0) & (df["revenue_realised"].fillna(0) == 0) & (df["tower_weight"].fillna(0) == 0)
    df = df.loc[~mask].reset_index(drop=True)

    return df.reset_index(drop=True)







def build_responsibilities_atomic(df_clean: pd.DataFrame,
                                  project_name: str,
                                  project_key: str) -> pd.DataFrame:
    """
    Return atomic rows: one row per (project, entity_type, entity_name, location_no)
    with the *per-row* revenue/tower_weight values from the microplan (no pre-aggregation).
    """
    # Ensure required columns exist (reader enforces schema earlier)
    for c in ("revenue_planned", "revenue_realised", "tower_weight"):
        if c not in df_clean.columns:
            df_clean[c] = 0.0

    # normalize location as stable string (already done earlier, but keep safe)
    loc = df_clean["location_no"].astype(object).map(
        lambda x: str(x).strip() if x is not None and str(x).strip() != "" else None
    )

    frames = []
    pairs = [
        ("gang_name",        "Gang"),
        ("section_incharge", "Section Incharge"),
        ("supervisor",       "Supervisor"),
    ]

    for col, etype in pairs:
        if col not in df_clean.columns:
            continue
        # atomic slice for this entity type
        cols_to_take = [c for c in [
            col, "location_no",
            "revenue_planned","revenue_realised","tower_weight",
            "tower_type","manpower","power_tools_issued","material_feeding",
            "starting_date","completion_date","tack_welding","final_checking"
        ] if c in df_clean.columns]
        part = df_clean[cols_to_take].copy()
        part.insert(0, "project_key",  project_key)
        part.insert(1, "project_name", project_name)
        part.insert(2, "entity_type",  etype)
        part = part.rename(columns={col: "entity_name"})
        # Drop rows with missing entity_name or missing/blank location
        part["entity_name"] = part["entity_name"].astype(str).str.strip()
        part["location_no"] = loc  # normalized above
        part = part[(part["entity_name"] != "") & part["location_no"].notna()]
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=[
            "project_key","project_name","entity_type","entity_name",
            "location_no","revenue_planned","revenue_realised","tower_weight"
        ])

    out = pd.concat(frames, ignore_index=True)
    # Column order for consistency
    cols = [
        "project_key","project_name","entity_type","entity_name","location_no",
        "revenue_planned","revenue_realised","tower_weight",
        "tower_type","manpower","power_tools_issued","material_feeding",
        "starting_date","completion_date","tack_welding","final_checking"
    ]
    return out[cols]




def _safe_write_df(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str, index: bool = False) -> None:
    """
    Overwrite a sheet if it exists. Uses writer.book provided by pandas'
    openpyxl engine; no manual assignment to writer.book.
    """
    wb = writer.book  # read-only property provided by pandas
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        wb.remove(ws)
        # Also clean pandas' internal cache to avoid stale references
        if hasattr(writer, "sheets") and sheet_name in writer.sheets:
            del writer.sheets[sheet_name]
    df.to_excel(writer, sheet_name=sheet_name, index=index)


def _open_writer_overwriting_sheets(output_path: str) -> pd.ExcelWriter:
    """
    Open an ExcelWriter. If the file exists, open in append mode and allow
    us to overwrite specific sheets later. Do NOT assign writer.book.
    """
    if os.path.exists(output_path):
        return pd.ExcelWriter(
            output_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="overlay",  # we'll delete/replace targeted sheets ourselves
        )
    else:
        return pd.ExcelWriter(output_path, engine="openpyxl", mode="w")


# ======================
# === Main Compile  ===
# ======================

def compile_microplans_to_workbook(
    input_dir: str,
    output_path: str,
    *,
    schema: dict = CURRENT_MICROPLAN_SCHEMA,
    glob_pattern: str = MICROPLAN_GLOB_PATTERN
) -> None:
    """
    - Finds files with 'micro plan' in the name (case-insensitive).
    - For each file: detect header, clean, aggregate responsibilities.
    - Writes one combined sheet: MICROPLAN_AGG_SHEET_NAME
    - Optionally writes per-project cleaned sheet(s).
    - Also writes an index (file → project) for traceability.
    """
    paths = sorted(glob(os.path.join(input_dir, glob_pattern), recursive=True))

    atomic_all = []
    index_rows = []
    
    for p in paths:
        proj_name = infer_project_name_from_filename(p)
        proj_key  = normalize_key(proj_name)
        
        # if write_raw_per_project else None
        try:
            df_clean = read_microplan_file(p, schema)
                        # ensure numeric responsibility columns exist
            for c in ("revenue_planned", "tower_weight"):
                if c not in df_clean.columns:
                    df_clean[c] = 0.0

            # build and collect per-project responsibilities
            resp_long = build_responsibilities_atomic(df_clean, proj_name, proj_key)
            if not resp_long.empty:
                atomic_all.append(resp_long)         

            # ...
            index_rows.append({
                "file_path": p,
                "project_name": proj_name,
                "project_key": proj_key,
                "rows_cleaned": len(df_clean),
                "status": "ok",
                "error": ""
            })
        except Exception as e:
            index_rows.append({
                "file_path": p,
                "project_name": proj_name,   # keep for debugging
                "project_key": proj_key,     # keep for debugging
                "rows_cleaned": 0,
                "status": "error",
                "error": str(e),
            })


    # Write combined outputs in one go
    # --- Write combined outputs (atomic only) ---
    with _open_writer_overwriting_sheets(output_path) as writer:
        # Atomic responsibilities sheet
        if atomic_all:
            responsibilities = pd.concat(atomic_all, ignore_index=True)
        else:
            responsibilities = pd.DataFrame(columns=[
                "project_key","project_name","entity_type","entity_name",
                "location_no","revenue_planned","revenue_realised","tower_weight",
                "tower_type","manpower","power_tools_issued","material_feeding",
                "starting_date","completion_date","tack_welding","final_checking"
            ])

        _safe_write_df(writer, responsibilities, MICROPLAN_AGG_SHEET_NAME, index=False)

        # Index sheet for traceability
        idx_df = pd.DataFrame(
            index_rows,
            columns=["file_path","project_name","project_key","rows_cleaned","status","error"]
        )
        _safe_write_df(writer, idx_df, MICROPLAN_INDEX_SHEET, index=False)




# ======================
# === CLI Entrypoint ===
# ======================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compile 'Micro Plan' responsibilities into the compiled workbook.")
    p.add_argument("--input-dir", required=True, help="Root folder containing project Excel files (where 'Micro Plan' files reside).")
    p.add_argument("--output-xlsx", required=True, help="Path to the compiled workbook (e.g., ErectionCompiled_Output.xlsx).")
    # p.add_argument("--write-raw", action="store_true",
                #    help="Also write per-project cleaned Micro Plan sheets (MicroPlan_<project_key>).")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    compile_microplans_to_workbook(
        input_dir=args.input_dir,
        output_path=args.output_xlsx,
    )


if __name__ == "__main__":
    main()
