#!/usr/bin/env python3
"""
Parse 'Erection Compiled' sheets from one or many Excel files, compute productivity,
expand to daily rows, and write a consolidated workbook:

Sheets written:
  - ProdDailyExpandedSingles : per-day rows including single-occurrence gangs
  - RawData           : per-erection (unexpanded) rows (the 6 fields)
  - Data Issues       : row-level problems (requested columns + 'Issues')
  - Issues            : file-level problems (missing sheet/headers, no valid rows, etc.)
  - Diagnostics       : which sheet used, detected header row, and normalized column names
  - README_Assumptions: assumptions and cleaning notes

Usage examples (Windows CMD):
  python erection_compiled_to_daily.py ^
    --input "C:\\path\\to\\DPR_Files" ^
    --output "C:\\path\\ErectionCompiled_Output.xlsx"

  python erection_compiled_to_daily.py ^
    --files "C:\\path\\TA 408.xlsx" "C:\\path\\TA413-PBNTL-20.08.2025.xlsx" ^
           "C:\\path\\TA325 ANTL KEC DPR_18-08-2025.xlsx" "C:\\path\\DPR TA-416,14-09-25.xlsx" ^
    --output "C:\\path\\ErectionCompiled_Output.xlsx"
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------- Config ----------
EXPECTED_HEADERS = [
    "location no",
    "type of tower",
    "starting date",
    "completion date",
    "gang name",
    "tower weight",
    "status",
]

# Accepts: "Erection Compiled", "Erection-Compiled", "Erection Compiled v2", etc.
TARGET_SHEET_REGEX = re.compile(r"^\s*erection\s*.*\s*compiled\s*$", flags=re.I)

# Business rules (centralized here)
START_CUTOFF = pd.Timestamp("2021-01-01")
TODAY = pd.Timestamp.today().normalize()
TOWER_MIN_MT = 10.0
TOWER_MAX_MT = 200.0

# Column order for per-day expanded output
PER_DAY_COLUMNS = [
    "Work Date",
    "Start Date",
    "Complete Date",
    "Gang name",
    "Tower Weight",
    "Productivity",
    "Project Name",
    "Location No.",
    "Status",
]

# ---------- Helpers ----------
def nrm_header(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s.strip())
    s = s.strip().strip(".")
    s = s.lower()
    return s


def find_header_row(df_raw: pd.DataFrame, search_rows: int = 30) -> Tuple[Optional[int], Optional[list]]:
    best = None
    best_score = -1
    nrows = min(search_rows, df_raw.shape[0])

    for r in range(nrows):
        row_vals = [nrm_header(x) for x in list(df_raw.iloc[r, :].values)]

        score = 0
        mapping = {}
        for i, val in enumerate(row_vals):
            if not val:
                continue
            for exp in EXPECTED_HEADERS:
                if exp == val or exp.replace(" ", "") in val.replace(" ", ""):
                    mapping[i] = exp
                    score += 1
                    break

        non_empty = sum(1 for v in row_vals if v)
        score += max(0, non_empty - 3) * 0.02

        if score > best_score:
            cols = [mapping.get(i, row_vals[i]) for i in range(len(row_vals))]
            best = (r, cols)
            best_score = score

    if best and best_score >= 3:
        return best
    return None, None


def find_target_sheet(xl: pd.ExcelFile) -> Optional[str]:
    for s in xl.sheet_names:
        if s.strip().lower() == "erection compiled":
            return s
    for s in xl.sheet_names:
        if TARGET_SHEET_REGEX.search(s):
            return s
    return None


def to_number_mt(x):
    """Parse weight values like '5.5 MT', '7 t', '3,200' â†’ float (MT)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(",", "")
    s = re.sub(r"(mt|tons?|t)\b", "", s)
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def to_date(x):
    """Parse text dates (DD/MM/YYYY, etc.) and Excel serials."""
    d = pd.to_datetime(x, errors="coerce", dayfirst=True)
    if (hasattr(d, "all") and pd.isna(d).all()) or (not hasattr(d, "all") and pd.isna(d)):
        try:
            v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
            if pd.notna(v):
                return pd.to_datetime("1899-12-30") + pd.to_timedelta(v, unit="D")
        except Exception:
            pass
    return d


def parse_project_from_filename(name: str) -> str:
    m = re.search(r"\b(TA|TB)\s*[- ]?\s*(\d{3,4})\b", name.upper())
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return Path(name).stem


def normalize_gang_name(name: str) -> str:
    """
    - Strip special chars (keep letters, digits, spaces)
    - Each word Title Case
    - If ends with digits stuck to a letter (e.g., 'Xyz4'), insert a space â†’ 'Xyz 4'
    Examples:
      'sobha devi' -> 'Sobha Devi'
      'sobha-devi' -> 'Sobha Devi'
      'xyz4' -> 'Xyz 4'
      'xyz-4' -> 'Xyz 4'
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    # keep letters, digits, spaces â†’ replace other runs with a space
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Title case
    s = s.title()
    # Insert a space before trailing digits if jammed to a letter
    s = re.sub(r"([A-Za-z])(\d+)$", r"\1 \2", s)
    return s.strip()

# --- NEW: tolerant loader for a single source file's "Project Details" ---
def load_project_details_from_source(xl: pd.ExcelFile, source_file: Path) -> pd.DataFrame:
    # only proceed if a "Project Details" sheet exists
    if "Project Details" not in xl.sheet_names:
        return pd.DataFrame()

    dfp = pd.read_excel(xl, sheet_name="Project Details")

    # tolerant column picks (allow minor header variations)
    def pick(df, *opts):
        cols = {str(c).strip().lower(): c for c in df.columns}
        for o in opts:
            k = o.strip().lower()
            if k in cols:
                return cols[k]
        # contains fallback
        for k, c in cols.items():
            if any(o.lower() in k for o in opts):
                return c
        raise KeyError(f"Missing one of {opts} in {list(df.columns)}")

    try:
        c_code = pick(dfp, "Project Code", "project_code", "code")
        c_name = pick(dfp, "Project Name", "project_name", "name")
        c_client = pick(dfp, "Client Name", "client", "client_name")
        c_noa = pick(dfp, "NOA Start Date", "noa start", "start date")
        c_loa = pick(dfp, "LOA End Date", "loa end", "end date")
        c_pm = pick(dfp, "Project Manager", "project manger", "pm")
        c_rm = pick(dfp, "Regional Manager", "regional_manager")
        c_pe = pick(dfp, "Planning Engineer", "planning_engineer")
        c_pch = pick(dfp, "PCH")
        c_si = pick(dfp, "Section Incharge", "section_incharge")
        c_sup = pick(dfp, "Supervisor", "supervisor")
    except KeyError:
        # If the sheet is weirdly formatted, skip gracefully
        return pd.DataFrame()

        # Build a narrow frame with the columns we care about
    meta = pd.DataFrame({
        "project_code": dfp[c_code],
        "project_name": dfp[c_name],
        "client_name":  dfp[c_client],
        "noa_start":    pd.to_datetime(dfp[c_noa], errors="coerce"),
        "loa_end":      pd.to_datetime(dfp[c_loa], errors="coerce"),
        "project_mgr":  dfp[c_pm],
        "regional_mgr": dfp[c_rm],
        "planning_eng": dfp[c_pe],
        "pch":          dfp[c_pch],
        "section_inch": dfp[c_si],
        "supervisor":   dfp[c_sup],
    }).astype(object)

    # Forward-fill project metadata so blank rows (extra names) inherit the project
    meta[["project_code","project_name","client_name","noa_start","loa_end",
          "project_mgr","regional_mgr","planning_eng","pch"]] = \
        meta[["project_code","project_name","client_name","noa_start","loa_end",
              "project_mgr","regional_mgr","planning_eng","pch"]].ffill()

    # Helper: join unique non-empty values, preserving order
    def uniq_join(series):
        vals = [str(x).strip() for x in series if pd.notna(x) and str(x).strip() != ""]
        seen = set(); out = []
        for v in vals:
            if v not in seen:
                seen.add(v); out.append(v)
        return ", ".join(out)

    # Aggregate to one row per project_code (collect multiples for the two fields)
    out = (meta
           .groupby("project_code", dropna=False)
           .agg({
               "project_name": "first",
               "client_name":  "first",
               "noa_start":    "first",
               "loa_end":      "first",
               "project_mgr":  "first",
               "regional_mgr": "first",
               "planning_eng": "first",
               "pch":          "first",
               "section_inch": uniq_join,
               "supervisor":   uniq_join,
           })
           .reset_index()
    )

    # Uppercase/trim code; drop empties
    out["project_code"] = out["project_code"].astype(str).str.strip().str.upper()
    out = out[out["project_code"].ne("")]

    # File name for traceability + pass-through literal "Project Name" label if present
    out["_source_file"] = source_file.name
    if "Project Name" in dfp.columns:
        # keep the human label from sheet; itâ€™s already ffilled via meta
        out["Project Name"] = meta.groupby("project_code")["project_name"].first().astype(str).str.strip()

    return out


# ---------- Core (per file) ----------
def process_file(path: Path):
    """
    Process a single workbook; return:
      per_day         : per-day expanded rows (with Work Date)
      per_day_with_singles : per-day rows including single-occurrence gangs
      per_erection    : per-erection (unexpanded) rows (the 6 fields)
      diagnostics     : dict (file, project, sheet, detected header row, normalized columns)
      issues          : list[dict] (file-level)
      data_issues_df  : DataFrame with requested columns + 'Issues' (row-level)
    """
    issues = []
    data_issues_rows = []  # row-level issues here
    empty_df = pd.DataFrame()
    diag = {"file": path.name, "project": parse_project_from_filename(path.name)}

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        issues.append({"file": path.name, "issue": f"Excel open error: {e}"})
        return empty_df, empty_df, empty_df, diag, issues, empty_df

    target = find_target_sheet(xl)
    if not target:
        issues.append({"file": path.name, "issue": "Sheet 'Erection Compiled' not found", "sheets": xl.sheet_names})
        return empty_df, empty_df, empty_df, diag, issues, empty_df

    try:
        df_raw = pd.read_excel(xl, sheet_name=target, header=None)
    except Exception as e:
        issues.append({"file": path.name, "issue": f"Read error: {e}", "sheet": target})
        return empty_df, empty_df, empty_df, diag, issues, empty_df

    hdr_row, cols = find_header_row(df_raw, search_rows=30)
    if hdr_row is None:
        issues.append({"file": path.name, "issue": "Could not detect header row automatically", "sheet": target})
        return empty_df, empty_df, empty_df, diag, issues, empty_df

    df = df_raw.iloc[hdr_row + 1:].copy()
    df.columns = cols

    diag.update({
        "sheet": target,
        "detected_header_row": hdr_row,
        "columns_detected": ", ".join(cols[:20])
    })

    # Only the fields we need for computation
    needed = ["starting date", "completion date", "gang name", "tower weight", "location no","status"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        issues.append({
            "file": path.name,
            "issue": f"Missing required columns after header-detect: {missing}",
            "columns": list(df.columns)
        })
        return empty_df, empty_df, empty_df, diag, issues, empty_df

    work = df[needed].copy()

    # Parse + clean (do not drop yet; we want to capture issues first)
    work["Start Date"] = work["starting date"].apply(to_date)
    work["Complete Date"] = work["completion date"].apply(to_date)
    work["Gang name"] = work["gang name"].apply(normalize_gang_name)
    work["Tower Weight"] = work["tower weight"].apply(to_number_mt)
    work["Project Name"] = parse_project_from_filename(path.name)
    # NEW: normalize required passthrough fields
    work["Location No."] = (
        work["location no"].astype(object).map(lambda x: str(x).strip() if pd.notna(x) else pd.NA)
    )

    work["Status"] = (
        work["status"].astype(object).map(lambda x: str(x).strip() if pd.notna(x) else pd.NA)
    )

    # Precompute validity flags
    missing_dt_mask = work["Start Date"].isna() | work["Complete Date"].isna()
    # compute days where both dates exist
    days = (work["Complete Date"] - work["Start Date"]).dt.days + 1
    non_positive_days_mask = (~missing_dt_mask) & (days <= 0)
    old_start_mask = (~work["Start Date"].isna()) & (work["Start Date"] < START_CUTOFF)
    future_completion_mask = (~work["Complete Date"].isna()) & (work["Complete Date"] >= TODAY)
    tw_out_of_range_mask = (~work["Tower Weight"].isna()) & (
        (work["Tower Weight"] < TOWER_MIN_MT) | (work["Tower Weight"] > TOWER_MAX_MT)
    )

    # Productivity (only where days valid)
    prod = pd.Series(np.nan, index=work.index, dtype="float")
    valid_for_prod = (~missing_dt_mask) & (days > 0)
    prod[valid_for_prod] = work.loc[valid_for_prod, "Tower Weight"] / days[valid_for_prod]
    work["Productivity"] = prod

    # Helper to push rows into Data Issues with a reason
    def push_data_issue(mask, reason: str):
        if mask.any():
            sub = work.loc[mask, ["Start Date", "Complete Date", "Gang name", "Tower Weight",
                                  "Productivity", "Project Name", "Location No.", "Status"]].copy()
            sub["Issues"] = reason
            data_issues_rows.append(sub)

    # Collect row-level issues (not expanded)
    push_data_issue(missing_dt_mask, "Missing start/end date (not expanded)")
    push_data_issue(non_positive_days_mask, "Non-positive duration (Start > End or 0) (not expanded)")
    push_data_issue(old_start_mask, f"Start before {START_CUTOFF.date()} (not expanded)")
    push_data_issue(future_completion_mask, f"Completion on/after {TODAY.date()} (not expanded)")
    push_data_issue(tw_out_of_range_mask, f"Tower Weight out of range (<{TOWER_MIN_MT} or >{TOWER_MAX_MT}) (not expanded)")

    # Exclude all above issues from expansion consideration
    invalid_mask = missing_dt_mask | non_positive_days_mask | old_start_mask | future_completion_mask | tw_out_of_range_mask
    work_valid = work.loc[~invalid_mask].copy()

    # Identify single-occurrence gangs (per Project + Gang), add to Data Issues, exclude from expansion
    if not work_valid.empty:
        occ = work_valid.groupby(["Project Name", "Gang name"]).size().rename("n").reset_index()
        singles = occ[occ["n"] == 1][["Project Name", "Gang name"]]
        if not singles.empty:
            singles_key = set(map(tuple, singles.to_records(index=False)))
            single_mask = work_valid.apply(
                lambda r: (r["Project Name"], r["Gang name"]) in singles_key, axis=1
            )  # <-- mask aligned to work_valid

            # --- align mask to 'work' before sending to push_data_issue ---
            single_idx = work_valid.index[single_mask]
            single_mask_work = work.index.isin(single_idx)
            push_data_issue(single_mask_work, "Single-occurrence gang (not expanded)")
            # --------------------------------------------------------------

            work_to_expand = work_valid.loc[~single_mask].copy()
        else:
            work_to_expand = work_valid.copy()
    else:
        work_to_expand = work_valid.copy()


    # ---- Per-erection (UNEXPANDED) ----
    per_erection = work[[
        "Start Date", "Complete Date", "Gang name", "Tower Weight", "Productivity", "Project Name", "Location No.", "Status"
    ]].copy()

    # ---- Per-day (EXPANDED) ----
    def expand_per_day(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty:
            return pd.DataFrame(columns=PER_DAY_COLUMNS)

        rows = []
        for _, r in source.iterrows():
            for d in pd.date_range(r["Start Date"], r["Complete Date"], freq="D"):
                rows.append({
                    "Work Date": d.normalize(),
                    "Start Date": r["Start Date"].normalize(),
                    "Complete Date": r["Complete Date"].normalize(),
                    "Gang name": r["Gang name"],
                    "Tower Weight": r["Tower Weight"],
                    "Productivity": r["Productivity"],
                    "Project Name": r["Project Name"],
                    "Location No.": r["Location No."],
                    "Status": r["Status"],
                })

        result = pd.DataFrame(rows)
        if result.empty:
            return result.reindex(columns=PER_DAY_COLUMNS)

        result = result.sort_values(
            ["Project Name", "Work Date", "Gang name", "Start Date"],
            ignore_index=True
        )
        return result.reindex(columns=PER_DAY_COLUMNS)

    # Only build the expanded per-day rows including single-occurrence gangs
    per_day_with_singles = expand_per_day(work_valid)

    data_issues_df = pd.concat(data_issues_rows, ignore_index=True) if data_issues_rows else pd.DataFrame()

    return per_day_with_singles, per_erection, diag, issues, data_issues_df


# ---------- Styling ----------
def style_sheet(ws, tab_color="99CCFF"):
    """Minimalist styling: colored tab + bold header with fill + borders + freeze header + friendly widths."""
    try:
        from openpyxl.styles import PatternFill, Font, Border, Side, Alignment
    except Exception:
        return  # styling optional; skip if openpyxl is limited

    # Tab color
    ws.sheet_properties.tabColor = tab_color

    if ws.max_row < 1 or ws.max_column < 1:
        return

    # Freeze header
    ws.freeze_panes = "A2"

    # Header styling
    header_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    bold_font = Font(bold=True)
    thin = Side(style="thin", color="DDDDDD")
    border = Border(top=thin, left=thin, right=thin, bottom=thin)

    for c in ws[1]:
        c.fill = header_fill
        c.font = bold_font
        c.border = border
        c.alignment = Alignment(vertical="center")

    # Light borders on data cells + basic width
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.border = border

    # Column widths (simple heuristic)
    for col_cells in ws.columns:
        max_len = 10
        for cell in col_cells[: min(200, ws.max_row)]:  # don't scan entire huge sheets
            try:
                max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
            except Exception:
                pass
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 45)


# ---------- Main ----------
# def main():
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Parse 'Erection Compiled', compute productivity, expand to daily rows."
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", help="Folder containing .xlsx/.xlsm files")
    grp.add_argument("--files", nargs="+", help="Explicit list of .xlsx/.xlsm files")
    ap.add_argument("--output", required=True, help="Output Excel path")
    # args = ap.parse_args()
    args = ap.parse_args(argv)

    # Resolve input files
    paths: List[Path] = []
    if args.input:
        folder = Path(args.input)
        if not folder.exists():
            raise SystemExit(f"Input folder not found: {folder}")
        for fp in folder.iterdir():
            if fp.is_file() and fp.suffix.lower() in (".xlsx", ".xlsm"):
                name_lower = fp.name.lower()
                if any(k in name_lower for k in ("consolidated", "output", "compiled")) and "erection" not in name_lower:
                    continue
                paths.append(fp)
    else:
        paths = [Path(p) for p in args.files]

    all_per_day_with_singles, all_per_erection = [], []
    all_issues, all_diag = [], []
    all_data_issues = []
    all_proj_details = []

    for p in paths:
        if not p.exists():
            all_issues.append({"file": p.name, "issue": "missing"})
            continue

        per_day_with_singles, per_erection, diag, issues, data_issues_df = process_file(p)

        # --- NEW: attempt to read "Project Details" from this source ---
        try:
            xl_src = pd.ExcelFile(p, engine="openpyxl")
            dfp = load_project_details_from_source(xl_src, p)
            if not dfp.empty:
                fn_project_name = parse_project_from_filename(p.name)  # you already use this elsewhere
                dfp["Project Name"] = fn_project_name                 # <-- NEW (title-case col)
                all_proj_details.append(dfp)
        except Exception as e:
            all_issues.append({"file": p.name, "issue": f"Project Details read error: {e}"})

        if not per_day_with_singles.empty:
            all_per_day_with_singles.append(per_day_with_singles.assign(_source_file=p.name))
        if not per_erection.empty:
            all_per_erection.append(per_erection.assign(_source_file=p.name))
        if not data_issues_df.empty:
            all_data_issues.append(data_issues_df.assign(_source_file=p.name))

        if diag:
            all_diag.append(diag)
        all_issues.extend(issues)


    # Consolidate across all inputs
    per_day_with_singles_consol = pd.concat(all_per_day_with_singles, ignore_index=True) if all_per_day_with_singles else pd.DataFrame()
    per_erection_consol = pd.concat(all_per_erection, ignore_index=True) if all_per_erection else pd.DataFrame()
    data_issues_consol = pd.concat(all_data_issues, ignore_index=True) if all_data_issues else pd.DataFrame()
    issues_df = pd.DataFrame(all_issues) if all_issues else pd.DataFrame()
    diag_df = pd.DataFrame(all_diag) if all_diag else pd.DataFrame()
    projdetails_df = pd.DataFrame()
    projdetails_out = pd.DataFrame()
        # --- NEW: consolidate Project Details across inputs ---
        # --- consolidate Project Details across inputs (NEW) ---
    if all_proj_details:
        projdetails_df = pd.concat(all_proj_details, ignore_index=True)

        # Deduplicate by project_code; latest file wins
        projdetails_df = (
            projdetails_df.sort_values("_source_file")
                        .drop_duplicates(subset=["project_code"], keep="last")
        )

        # Order & friendly headers (align with other sheets' style)
        projdetails_out = projdetails_df.rename(columns={
            "project_code": "Project Code",
            "client_name": "Client Name",
            "noa_start": "NOA Start Date",
            "loa_end": "LOA End Date",
            "project_mgr": "Project Manager",
            "regional_mgr": "Regional Manager",
            "planning_eng": "Planning Engineer",
            "pch": "PCH",
            "section_inch": "Section Incharge",
            "supervisor": "Supervisor",
        })[
            [
                "Project Code",
                "Project Name",
                "project_name",           # <-- ensure this is present
                "Client Name",
                "NOA Start Date",
                "LOA End Date",
                "Project Manager",
                "Regional Manager",
                "Planning Engineer",
                "PCH",
                "Section Incharge",
                "Supervisor",
            ]
        ]
    else:
        projdetails_out = pd.DataFrame()



    # README / Assumptions
    readme_lines = [
        "Assumptions & Cleaning Rules:",
        f"- Start date cutoff: rows with Start Date before {START_CUTOFF.date()} are not expanded and logged in 'Data Issues'.",
        f"- Completion date must be before {TODAY.date()} (future completions go to 'Data Issues').",
        f"- Tower Weight range: only [{TOWER_MIN_MT:.0f}, {TOWER_MAX_MT:.0f}] MT is considered valid for expansion; out-of-range rows go to 'Data Issues'.",
        "- Missing or non-positive durations (Start/End missing or Start > End or 0) are logged to 'Data Issues' and not expanded.",
        "- Gangs that performed erection only once (per Project+Gang) are not expanded and are logged in 'Data Issues'.",
        "- Gang name normalization: remove special characters (digits kept), Title Case words, and insert a space before trailing digits (e.g., 'xyz4' â†’ 'Xyz 4').",
        "- Sheets:",
        "    â€¢ ProdDailyExpanded  : per-day expanded rows used by the dashboard",
        "    â€¢ ProdDailyExpandedSingles : per-day expanded rows including single-occurrence gangs",
        "    â€¢ RawData        : per-erection rows (unexpanded), for traceability",
        "    â€¢ Data Issues    : row-level data problems (reason in 'Issues' column)",
        "    â€¢ Issues         : file-level problems (missing sheet/headers, read/open errors, etc.)",
        "    â€¢ Diagnostics    : which sheet used, detected header row, and normalized header text",
        "- Dashboard note: a 15-day productivity loss cap is used in downstream analytics.",
    ]
    readme_df = pd.DataFrame({"Notes": readme_lines})

    # Write output workbook (+ minimal styling)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if not per_day_with_singles_consol.empty:
            per_day_with_singles_consol.drop(columns=["_source_file"], errors="ignore").to_excel(w, sheet_name="ProdDailyExpandedSingles", index=False)
        if not per_erection_consol.empty:
            per_erection_consol.drop(columns=["_source_file"], errors="ignore").to_excel(w, sheet_name="RawData", index=False)
        if not data_issues_consol.empty:
            data_issues_consol.drop(columns=["_source_file"], errors="ignore").to_excel(w, sheet_name="Data Issues", index=False)
        if not issues_df.empty:
            issues_df.to_excel(w, sheet_name="Issues", index=False)
        if not diag_df.empty:
            diag_df.to_excel(w, sheet_name="Diagnostics", index=False)
        # --- NEW: write consolidated Project Details ---
        if not projdetails_df.empty:
            projdetails_df.to_excel(w, sheet_name="ProjectDetails", index=False)
        readme_df.to_excel(w, sheet_name="README_Assumptions", index=False)

        # Apply styling
        wb = w.book
        for sheet_name, color in [
            ("ProdDailyExpandedSingles", "9CC3E6"),  # blue variant including singles
            ("RawData", "C6E0B4"),         # green
            ("Data Issues", "F8CBAD"),     # light red
            ("Issues", "D9D2E9"),          # purple
            ("Diagnostics", "FFE699"),     # yellow
            ("README_Assumptions", "99CCFF"),
            ("ProjectDetails", "99E6E6"),  # --- NEW ---
        ]:
            if sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                style_sheet(ws, tab_color=color)

    print(f"Done. Wrote {out_path}")


def run_pipeline(input_path=None, output_path=None, files=None, extra_args=None):
    """Convenience wrapper to run the CLI pipeline programmatically."""
    cli_args = []
    if input_path:
        cli_args.extend(["--input", str(input_path)])
    elif files:
        cli_args.append("--files")
        cli_args.extend(str(p) for p in files)
    else:
        raise ValueError("Either input_path or files must be provided.")

    if output_path:
        cli_args.extend(["--output", str(output_path)])
    else:
        raise ValueError("An output_path is required to write the compiled workbook.")

    if extra_args:
        cli_args.extend(list(extra_args))

    main(cli_args)


if __name__ == "__main__":
    main()

