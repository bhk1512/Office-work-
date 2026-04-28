import argparse
import json
import os
import re
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd

from erection_compiled_to_daily_new import run_pipeline
from dashboard.config import AppConfig
from dashboard import stringing_ingest as ingest
from dashboard import progress_status_ingest
from dashboard.stringing import (
    expand_stringing_to_daily,
    normalize_stringing_columns,
    classify_stringing_missing_headers,
    summarize_date_parsing,
    add_length_units,
    parse_project_code_from_filename,
    find_stringing_header_row,
    infer_missing_methods_from_erection,
)
from microplan_compile import (
    compile_microplans_to_workbook,
    compile_stringing_microplans_to_workbook,
)
from dashboard.project_identity import (
    build_project_display,
    build_project_scope_key,
    normalize_line_name,
    parse_project_identity_from_filename,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Dict[str, Any] = {
    "input_directory": "Raw Data/DPRs",
    "microplan_directory": "Raw Data/Micro Plans",
    "output_file": "Parquets/Erection/ErectionCompiled_Output.xlsx",
    "pipeline_extra_args": [],
    "dash_host": "0.0.0.0",
    "dash_port": 8050,
    "dash_debug": False,
}


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Configuration file {path} must contain a JSON object.")
    return data


def _resolve_path(value: Optional[str], base: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _normalise_files(raw: Optional[Iterable[str]], base: Path) -> Optional[List[Path]]:
    if raw is None:
        return None
    files: List[Path] = []
    for item in raw:
        resolved = _resolve_path(str(item), base)
        if resolved is None:
            continue
        files.append(resolved)
    return files if files else None


PARQUET_SHEETS: tuple[str, ...] = (
    "ProdDailyExpandedSingles",
    "ProdDailyExpanded",
    "RawData",
    "ProjectBaselines",
    "ProjectBaselinesMonthly",
    "ProjectDetails",
    "MicroPlanResponsibilities",
    "MicroPlanIndex",
)


def _write_parquet(df: pd.DataFrame, destination: Path) -> None:
    """Persist *df* atomically to *destination* using DuckDB/pandas fallbacks."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(df, pd.DataFrame):
        print(f"[pipeline] Warning: skipped parquet write for {destination} (input is not a dataframe).")
        return
    if len(df.columns) == 0:
        print(f"[pipeline] Warning: skipped parquet write for {destination} (no columns).")
        return

    temp_path = destination.with_name(f"{destination.name}.tmp")
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass

    def _validate(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            if path.stat().st_size < 12:
                return False
            with path.open("rb") as fh:
                header = fh.read(4)
                fh.seek(-4, 2)
                trailer = fh.read(4)
            if header != b"PAR1" or trailer != b"PAR1":
                return False
            with duckdb.connect(database=":memory:") as con:
                con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)])
            return True
        except Exception:
            return False

    write_ok = False
    try:
        with duckdb.connect(database=":memory:") as con:
            con.register("df_to_write", df)
            con.execute(
                "COPY df_to_write TO ? (FORMAT 'parquet', COMPRESSION 'zstd')",
                [str(temp_path)],
            )
        write_ok = _validate(temp_path)
    except Exception as exc:
        print(f"[pipeline] Warning: DuckDB parquet write failed for {destination}: {exc}")

    if not write_ok:
        try:
            df.to_parquet(temp_path, compression="zstd", index=False)
            write_ok = _validate(temp_path)
        except Exception as exc:
            print(f"[pipeline] Warning: pandas parquet write failed for {destination}: {exc}")
            safe = df.copy()
            for column in safe.select_dtypes(include="object").columns:
                safe[column] = safe[column].astype(str)
            try:
                safe.to_parquet(temp_path, compression="zstd", index=False)
                write_ok = _validate(temp_path)
            except Exception as exc2:
                print(f"[pipeline] Warning: string-coerced pandas parquet write failed for {destination}: {exc2}")
                try:
                    all_text = df.astype(str)
                    all_text.to_parquet(temp_path, compression="zstd", index=False)
                    write_ok = _validate(temp_path)
                except Exception as exc3:
                    print(f"[pipeline] Warning: full-string pandas parquet write failed for {destination}: {exc3}")
                    write_ok = False

    if not write_ok:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        print(f"[pipeline] Warning: skipped replacing parquet {destination}; temp artifact validation failed.")
        return

    temp_path.replace(destination)


def export_workbook_to_parquet(workbook_path: Path, sheets: Iterable[str] | None = None) -> Path:
    """Export selected workbook sheets to parquet files under the dataset folder.

    New behavior: writes directly into the workbook's parent directory (e.g.,
    Parquets/Erection) instead of creating a sibling "*_parquet" directory.
    """

    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook '{workbook_path}' does not exist.")

    target_dir = workbook_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    sheet_list = list(sheets) if sheets is not None else list(PARQUET_SHEETS)
    exported: list[str] = []
    with pd.ExcelFile(workbook_path) as workbook:
        available = set(workbook.sheet_names)
        for sheet in sheet_list:
            if sheet not in available:
                continue
            df = workbook.parse(sheet_name=sheet)
            if df is None:
                continue
            destination = target_dir / f"{sheet}.parquet"
            _write_parquet(df, destination)
            exported.append(sheet)

    if not exported:
        print(f"[pipeline] No matching sheets were exported from {workbook_path}.")
    else:
        print(f"[pipeline] Exported sheets to parquet: {', '.join(exported)}")

    return target_dir


def _coerce_excel_date_series(series: pd.Series) -> pd.Series:
    """Convert Excel serial dates or textual dates into pandas timestamps."""

    if series is None:
        return pd.Series([], dtype="datetime64[ns]")
    work = pd.Series(series)
    parsed = pd.to_datetime(work, errors="coerce")
    numeric = pd.to_numeric(work, errors="coerce")
    use_excel = numeric.notna() & parsed.isna()
    if use_excel.any():
        excel = pd.to_datetime(
            numeric[use_excel],
            errors="coerce",
            unit="D",
            origin="1899-12-30",
        )
        parsed.loc[use_excel] = excel
    return parsed.dt.normalize()


def _normalize_stringing_dates_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *frame* with obvious date/month columns coerced to timestamps."""

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        label = str(column).strip().lower()
        if "date" in label or "month" in label:
            out[column] = _coerce_excel_date_series(out[column])
    return out


def _stringing_candidates(input_dir: Optional[Path], files: Optional[List[Path]]) -> List[Path]:
    if files:
        return [
            p
            for p in files
            if p.suffix.lower() in (".xlsx", ".xlsm", ".xls") and p.exists() and not p.name.startswith("~$")
        ]
    if input_dir and input_dir.exists():
        return sorted(
            [
                p
                for p in input_dir.rglob("*.xls*")
                if p.is_file() and not p.name.startswith("~$")
            ]
        )
    return []


def _load_erection_daily_reference() -> pd.DataFrame:
    candidates = [
        (BASE_DIR / "Parquets" / "Erection" / "ProdDailyExpanded.parquet").resolve(),
        (BASE_DIR / "Parquets" / "Erection" / "ProdDailyExpandedSingles.parquet").resolve(),
    ]
    for parquet_path in candidates:
        if not parquet_path.exists():
            continue
        try:
            with duckdb.connect(database=":memory:") as con:
                return con.execute("SELECT * FROM read_parquet(?)", [str(parquet_path)]).df()
        except Exception:
            continue
    workbook = (BASE_DIR / "Parquets" / "Erection" / "ErectionCompiled_Output.xlsx").resolve()
    if workbook.exists():
        try:
            with pd.ExcelFile(workbook) as xl:
                for sheet in ("ProdDailyExpanded", "ProdDailyExpandedSingles"):
                    if sheet in xl.sheet_names:
                        frame = xl.parse(sheet_name=sheet)
                        if isinstance(frame, pd.DataFrame) and not frame.empty:
                            return frame
        except Exception:
            pass
    return pd.DataFrame()


def _normalize_project_code_key(value: object) -> str:
    return ingest.normalize_project_code_key(value)


def _normalize_space_only(value: object) -> str:
    return ingest.normalize_space_only(value)


def _resolve_dpr_config_path(input_dir: Optional[Path]) -> Optional[Path]:
    raw_root = input_dir if input_dir is not None else (BASE_DIR / "Raw Data" / "DPRs")
    return ingest.resolve_dpr_config_path(raw_root, repo_root=BASE_DIR)


def _load_stringing_sheet_config(input_dir: Optional[Path]) -> Dict[str, List[Dict[str, str]]]:
    raw_root = input_dir if input_dir is not None else (BASE_DIR / "Raw Data" / "DPRs")
    return ingest.load_stringing_sheet_config(raw_root, repo_root=BASE_DIR)


def _resolve_named_template_sheet(wb, expected_name: str) -> Optional[str]:
    expected_key = _normalize_space_only(expected_name)
    for name in wb.sheetnames:
        if _normalize_space_only(name) == expected_key:
            return name
    return None


def _resolve_project_template_sheets(wb, project_name: object, discipline: str) -> List[str]:
    project_text = str(project_name or "").strip()
    if not project_text:
        return []

    resolved: List[str] = []
    seen: set[str] = set()

    for expected in (
        f"{project_text} {discipline}",
        f"{project_text} {discipline} Template Check",
    ):
        hit = _resolve_named_template_sheet(wb, expected)
        if hit and hit not in seen:
            seen.add(hit)
            resolved.append(hit)

    project_key = _normalize_space_only(project_text)
    discipline_key = _normalize_space_only(discipline)
    for name in wb.sheetnames:
        key = _normalize_space_only(name)
        if not key:
            continue
        if key.startswith(project_key) and key.endswith(discipline_key) and name not in seen:
            seen.add(name)
            resolved.append(name)

    return resolved


def _extract_template_column_map(ws) -> Dict[int, str]:
    to_map_row = None
    for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        for cell in row:
            if _normalize_space_only(cell) == "to map":
                to_map_row = row_idx
                break
        if to_map_row is not None:
            break

    if to_map_row is None:
        return {}

    labels_row = to_map_row + 1
    row_values = next(ws.iter_rows(min_row=labels_row, max_row=labels_row, values_only=True), ())
    mapping: Dict[int, str] = {}
    for col_idx, value in enumerate(row_values):
        label = str(value).strip() if value is not None else ""
        if not label:
            continue
        mapping[col_idx] = label
    return mapping


def _load_stringing_template_mapping_config(
    input_dir: Optional[Path],
    *,
    include_unchecked: bool = False,
) -> Tuple[Dict[str, Tuple[Dict[int, str], str]], Dict[str, str]]:
    raw_root = input_dir if input_dir is not None else (BASE_DIR / "Raw Data" / "DPRs")
    return ingest.load_stringing_template_mapping_config(
        raw_root,
        repo_root=BASE_DIR,
        include_unchecked=include_unchecked,
    )


def _apply_template_column_mapping(
    df: pd.DataFrame,
    template_map: Dict[int, str],
) -> Tuple[pd.DataFrame, List[str]]:
    return ingest.apply_template_column_mapping(df, template_map)


def _clean_stringing_header_label(label: object) -> str:
    if label is None:
        return ""
    try:
        if pd.isna(label):
            return ""
    except Exception:
        pass
    text = str(label).strip()
    if text.lower() == "nan":
        return ""
    return text


def _make_unique_headers(labels: List[str]) -> List[str]:
    unique: List[str] = []
    seen: Dict[str, int] = {}
    for idx, label in enumerate(labels, start=1):
        base = label if label else f"unnamed_col_{idx}"
        key = _normalize_space_only(base) or base.lower()
        count = seen.get(key, 0) + 1
        seen[key] = count
        unique.append(base if count == 1 else f"{base}__{count}")
    return unique


def _sanitize_stringing_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cleaned = [_clean_stringing_header_label(col) for col in list(df.columns)]
    unique = _make_unique_headers(cleaned)
    out = df.copy()
    out.columns = unique
    return out


def _materialize_stringing_data(
    df_raw: pd.DataFrame, header_row: int, header_labels: List[object]
) -> Tuple[pd.DataFrame, List[str]]:
    labels = [_clean_stringing_header_label(label) for label in header_labels]
    data = df_raw.iloc[header_row + 1 :].copy()
    labels_series = pd.Series(labels)
    last_non_empty = labels_series.replace("", pd.NA).last_valid_index()
    if last_non_empty is not None:
        data = data.iloc[:, : last_non_empty + 1]
        labels_series = labels_series.iloc[: last_non_empty + 1]
    clean_labels = [_clean_stringing_header_label(c) for c in labels_series.values]
    clean_labels = _make_unique_headers(clean_labels)
    data.columns = clean_labels
    return data.reset_index(drop=True), clean_labels


def _extract_stringing_data_with_detected_header(
    df_raw: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[int], List[str]]:
    header_row, header_labels = find_stringing_header_row(df_raw)
    if header_row is None or header_labels is None:
        return None, None, []
    data, labels = _materialize_stringing_data(df_raw, int(header_row), list(header_labels))
    return data, int(header_row), labels


def _extract_stringing_data_with_first_row_header(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[int], List[str]]:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), None, []
    labels = list(df_raw.iloc[0, :].values)
    data, clean_labels = _materialize_stringing_data(df_raw, 0, labels)
    return data, 0, clean_labels


def _normalize_sheet_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    lowered = str(label).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", lowered)


def _find_stringing_sheet_name_from_list(
    names: list[str],
    preferred: Optional[str],
    project_candidates: Optional[List[str]] = None,
) -> Optional[str]:
    if not names:
        return None
    if project_candidates:
        by_space_key: Dict[str, str] = {}
        for name in names:
            key = _normalize_space_only(name)
            if key and key not in by_space_key:
                by_space_key[key] = name
        for candidate in project_candidates:
            hit = by_space_key.get(_normalize_space_only(candidate))
            if hit:
                return hit
        return None
    if preferred:
        target = preferred.strip().lower()
        target_normalized = _normalize_sheet_label(preferred)
        for n in names:
            if n.strip().lower() == target:
                return n
        if target_normalized:
            for n in names:
                if _normalize_sheet_label(n) == target_normalized:
                    return n
    # tolerant fallback: case-insensitive contains "stringing" and "compiled"
    lowered = [(n, n.strip().lower()) for n in names]
    for original, low in lowered:
        if "stringing" in low and "compiled" in low:
            return original
    return None


def _find_stringing_sheet_name(
    xl: pd.ExcelFile,
    preferred: Optional[str],
    project_candidates: Optional[List[str]] = None,
) -> Optional[str]:
    return _find_stringing_sheet_name_from_list(list(xl.sheet_names), preferred, project_candidates)


def _write_stringing_artifacts(
    output_path: Path,
    raw_df: pd.DataFrame,
    sheet_name: str,
    source_files: Optional[List[Path]] = None,
    *,
    diagnostics_df: Optional[pd.DataFrame] = None,
    issues_df: Optional[pd.DataFrame] = None,
    data_issues_df: Optional[pd.DataFrame] = None,
) -> Path:
    # Output workbook + parquet dirs
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write parquet files directly under Parquets/Stringing (no legacy *_parquet dir)
    parquet_dir = output_path.parent

    if diagnostics_df is None or issues_df is None or data_issues_df is None:
        # Diagnostics and issues fallback (summary)
        try:
            normalized, norm_report = normalize_stringing_columns(raw_df)
        except Exception:
            normalized, norm_report = raw_df.copy(), {"normalized_columns_ok": False, "present": [], "missing": [], "applied_map": {}}
        try:
            date_metrics = summarize_date_parsing(raw_df)
        except Exception:
            date_metrics = {"po_start_date_parsed_count": 0, "fs_complete_date_parsed_count": 0, "invalid_date_rows": 0}
        try:
            _, length_metrics = add_length_units(normalized)
        except Exception:
            length_metrics = {"total_length_km": 0.0, "min_length_km": 0.0, "max_length_km": 0.0}

        if issues_df is None:
            issues_df = pd.DataFrame()
        if data_issues_df is None:
            data_issues_df = pd.DataFrame()

        # Project codes from source file names (best-effort)
        projects: List[str] = []
        if source_files:
            seen = set()
            for p in source_files:
                code = parse_project_code_from_filename(p.name)
                if code and code not in seen:
                    seen.add(code)
                    projects.append(code)

        if diagnostics_df is None:
            diagnostics_df = pd.DataFrame([
                {
                    "sheet": sheet_name,
                    "rows": int(len(raw_df.index)),
                    "source_file_count": int(len(source_files) if source_files else 0),
                    "projects_detected": ", ".join(projects) if projects else "",
                    "normalized_columns_ok": bool(norm_report.get("normalized_columns_ok", False)),
                    "present_columns": ", ".join(norm_report.get("present", [])),
                    "missing_columns": ", ".join(norm_report.get("missing", [])),
                    "po_start_date_parsed_count": int(date_metrics.get("po_start_date_parsed_count", 0)),
                    "fs_complete_date_parsed_count": int(date_metrics.get("fs_complete_date_parsed_count", 0)),
                    "invalid_date_rows": int(date_metrics.get("invalid_date_rows", 0)),
                    "total_length_km": float(length_metrics.get("total_length_km", 0.0)),
                    "min_length_km": float(length_metrics.get("min_length_km", 0.0)),
                    "max_length_km": float(length_metrics.get("max_length_km", 0.0)),
                }
            ])

    method_inferred_rows = 0
    if isinstance(diagnostics_df, pd.DataFrame) and "MethodInferenceRows" in diagnostics_df.columns:
        method_inferred_rows = int(
            pd.to_numeric(diagnostics_df["MethodInferenceRows"], errors="coerce").fillna(0).sum()
        )
    readme_df = pd.DataFrame(
        [
            {
                "Note": "Compiled from DPR files: stringing sheet consolidation.",
                "Rules": "Preserve raw columns; diagnostics include column presence and date parsing; configured stringing sheets are processed independently with per-sheet line identity before concatenation; PO start to F/S complete inclusive for daily expansion.",
            },
            {
                "Note": "Method fallback assumption",
                "Rules": (
                    "If Method is missing, infer using erection span depth: "
                    "erection_locations<=2 => manual, >2 => tse, unresolved => tse fallback. "
                    f"Current inferred rows: {method_inferred_rows}."
                ),
            },
        ]
    )

    # Write workbook atomically and keep last known-good output on failure.
    temp_output = output_path.with_name(f"{output_path.name}.tmp")
    if temp_output.exists():
        try:
            temp_output.unlink()
        except Exception:
            pass
    try:
        with pd.ExcelWriter(temp_output, engine="openpyxl") as writer:
            raw_df.to_excel(writer, sheet_name=sheet_name[:31] or "Stringing", index=False)
            diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)
            if data_issues_df is not None and not data_issues_df.empty:
                data_issues_df.to_excel(writer, sheet_name="Data Issues", index=False)
            if issues_df is not None and not issues_df.empty:
                issues_df.to_excel(writer, sheet_name="Issues", index=False)
            readme_df.to_excel(writer, sheet_name="README_Assumptions", index=False)
        with pd.ExcelFile(temp_output) as probe:
            _ = probe.sheet_names
        temp_output.replace(output_path)
        print(f"[pipeline] Stringing: wrote workbook {output_path}")
    except Exception as exc:
        if temp_output.exists():
            try:
                temp_output.unlink()
            except Exception:
                pass
        print(f"[pipeline] Warning: failed to write stringing workbook {output_path}: {exc}")

    # Refresh parquet artifacts using atomic parquet writes.
    parquet_dir.mkdir(parents=True, exist_ok=True)
    compiled_parquet = parquet_dir / "StringingCompiled.parquet"
    compiled_ready = _normalize_stringing_dates_for_parquet(raw_df)
    _write_parquet(compiled_ready, compiled_parquet)
    print(f"[pipeline] Stringing: wrote compiled parquet {compiled_parquet}")

    # Precompute daily parquet to mirror dashboard artifact layout.
    try:
        daily = expand_stringing_to_daily(raw_df)
        if daily is not None and not daily.empty:
            daily_ready = _normalize_stringing_dates_for_parquet(daily)
            daily_parquet = output_path.parent / "StringingDaily.parquet"
            _write_parquet(daily_ready, daily_parquet)
            print(f"[pipeline] Stringing: wrote daily parquet {daily_parquet}")
    except Exception as exc:
        print(f"[pipeline] Warning: failed to write stringing daily parquet: {exc}")

    return parquet_dir


def compile_stringing_to_workbook(
    input_dir: Optional[Path],
    files: Optional[List[Path]],
    output_path: Path,
    sheet_name: Optional[str] = None,
) -> Optional[Path]:
    candidates = _stringing_candidates(input_dir, files)
    if not candidates:
        print("[pipeline] Stringing: no candidate files found; skipping.")
        return None

    stringing_sheet_config = _load_stringing_sheet_config(input_dir)
    stringing_template_config, stringing_template_errors = _load_stringing_template_mapping_config(input_dir)
    stringing_template_all_config, _ = _load_stringing_template_mapping_config(input_dir, include_unchecked=True)
    erection_daily_reference = _load_erection_daily_reference()
    selected_candidates: List[
        Tuple[
            Path,
            str,
            Optional[Dict[str, str]],
            Optional[Dict[int, str]],
            Optional[str],
            Optional[str],
        ]
    ] = []
    skipped_no_stringing = 0
    skipped_not_in_config = 0
    has_stringing_config = bool(stringing_sheet_config)
    for candidate in candidates:
        project = parse_project_code_from_filename(candidate.name) or "UNKNOWN"
        project_key = _normalize_project_code_key(project)
        configured_sheets = stringing_sheet_config.get(project_key)
        template_pair = stringing_template_config.get(project_key)
        template_map = template_pair[0] if template_pair else None
        template_sheet_name = template_pair[1] if template_pair else None
        template_error = stringing_template_errors.get(project_key)
        if has_stringing_config and configured_sheets is None:
            skipped_not_in_config += 1
            continue
        if configured_sheets is not None:
            if not configured_sheets:
                skipped_no_stringing += 1
                continue
            for configured_sheet in configured_sheets:
                selected_candidates.append(
                    (candidate, project, configured_sheet, template_map, template_sheet_name, template_error)
                )
        else:
            selected_candidates.append((candidate, project, None, template_map, template_sheet_name, template_error))

    if skipped_no_stringing:
        print(f"[pipeline] Stringing: skipped {skipped_no_stringing} workbook(s) per DPR_Config (no stringing sheet configured).")
    if skipped_not_in_config:
        print(f"[pipeline] Stringing: skipped {skipped_not_in_config} workbook(s) not listed in DPR_Config.")
    if not selected_candidates:
        print("[pipeline] Stringing: no workbooks require stringing compilation after DPR_Config filtering.")
        return None

    compiled: List[pd.DataFrame] = []
    missing: List[str] = []
    used_name: Optional[str] = None
    preferred = (sheet_name or AppConfig().stringing_sheet_name or "").strip()
    diag_rows: List[Dict[str, Any]] = []
    issue_rows: List[Dict[str, Any]] = []
    data_issue_rows: List[pd.DataFrame] = []
    today = pd.Timestamp.today().normalize()
    template_error_logged: set[str] = set()

    for f, project, configured_sheet_entry, template_map, template_sheet_name, template_error in selected_candidates:
        found = None
        header_row = None
        header_labels: List[str] = []
        fallback_note = None
        df = None
        template_changes: List[str] = []
        configured_sheet_name = str(configured_sheet_entry.get("sheet_name", "")).strip() if configured_sheet_entry else ""
        line_name_override = normalize_line_name(configured_sheet_entry.get("line_name", "")) if configured_sheet_entry else ""
        line_name_source = str(configured_sheet_entry.get("line_name_source", "")).strip() if configured_sheet_entry else ""
        project_key = _normalize_project_code_key(project)
        fallback_template_pair = stringing_template_all_config.get(project_key)
        fallback_template_map = fallback_template_pair[0] if fallback_template_pair else None
        fallback_template_sheet = fallback_template_pair[1] if fallback_template_pair else ""
        min_columns = (max((template_map or fallback_template_map).keys()) + 1) if (template_map or fallback_template_map) else None
        template_applied = False
        template_fallback_used = False
        template_sheet_used = template_sheet_name or ""

        if template_error:
            if f.name in template_error_logged:
                continue
            template_error_logged.add(f.name)
            issue_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": "",
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name_override,
                    "LineNameSource": line_name_source,
                    "Issue": f"TEMPLATE_CONFIG_ERROR: {template_error}",
                    "MissingColumns": "",
                    "Rows": 0,
                    "DailyRows": 0,
                }
            )
            diag_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": "",
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name_override,
                    "LineNameSource": line_name_source,
                    "DetectedHeaderRow": "",
                    "ColumnsDetected": "",
                    "NormalizedColumnsOk": "",
                    "PresentColumns": "",
                    "MissingColumns": "",
                    "AppliedMap": "",
                    "Rows": 0,
                    "DailyRows": 0,
                    "Status": "TEMPLATE_CONFIG_ERROR",
                    "FallbackNote": "",
                    "TemplateSheet": "",
                    "TemplateApplied": "",
                    "TemplateChanges": "",
                }
            )
            continue

        try:
            load_result = ingest.load_stringing_sheet_frame(
                f,
                configured_sheet_name=configured_sheet_name,
                preferred_sheet_name=preferred,
                min_columns=min_columns,
            )
            found = load_result.resolved_sheet
            fallback_note = load_result.fallback_note
            header_row = load_result.header_row
            header_labels = list(load_result.header_labels)
            df = load_result.frame
            if found is None:
                raise ValueError("NO_TARGET_SHEET")
            used_name = used_name or found
        except Exception as read_exc:
            is_no_target = isinstance(read_exc, ValueError) and str(read_exc) == "NO_TARGET_SHEET"
            if is_no_target:
                missing.append(f.name)
            issue_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found or "",
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name_override,
                    "LineNameSource": line_name_source,
                    "Issue": "NO_TARGET_SHEET" if is_no_target else f"READ_FAIL: {type(read_exc).__name__}",
                    "MissingColumns": "",
                    "Rows": 0,
                    "DailyRows": 0,
                }
            )
            diag_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found or "",
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name_override,
                    "LineNameSource": line_name_source,
                    "DetectedHeaderRow": "" if header_row is None else int(header_row),
                    "ColumnsDetected": ", ".join(header_labels),
                    "NormalizedColumnsOk": "",
                    "PresentColumns": "",
                    "MissingColumns": "",
                    "AppliedMap": "",
                    "Rows": 0,
                    "DailyRows": 0,
                    "Status": "NO_TARGET_SHEET" if is_no_target else "READ_FAIL",
                    "FallbackNote": "",
                    "TemplateSheet": template_sheet_name or "",
                    "TemplateApplied": "",
                    "TemplateChanges": "",
                }
            )
            continue

        if df is None or df.empty:
            issue_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found or "",
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name_override,
                    "LineNameSource": line_name_source,
                    "Issue": "EMPTY_SHEET",
                    "MissingColumns": "",
                    "Rows": 0,
                    "DailyRows": 0,
                }
            )
            diag_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found or "",
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name_override,
                    "LineNameSource": line_name_source,
                    "DetectedHeaderRow": "" if header_row is None else int(header_row),
                    "ColumnsDetected": ", ".join(header_labels),
                    "NormalizedColumnsOk": "",
                    "PresentColumns": "",
                    "MissingColumns": "",
                    "AppliedMap": "",
                    "Rows": 0,
                    "DailyRows": 0,
                    "Status": "EMPTY_SHEET",
                    "FallbackNote": fallback_note or "",
                    "TemplateSheet": template_sheet_name or "",
                    "TemplateApplied": "",
                    "TemplateChanges": "",
                }
            )
            continue

        if template_map:
            df, template_changes = _apply_template_column_mapping(df, template_map)
            header_labels = [str(col) for col in df.columns]
            template_applied = True
            template_sheet_used = template_sheet_name or template_sheet_used

        df = _sanitize_stringing_columns(df)
        header_labels = [str(col) for col in df.columns]
        df = df.copy()
        identity = parse_project_identity_from_filename(f.name)
        file_line_name = normalize_line_name(identity.get("line_name", ""))
        line_name = line_name_override or file_line_name
        project_code_display = str(identity.get("project_code", "")).strip() or str(project or "").strip()
        project_display = build_project_display(project_code_display, line_name, project or project_code_display)
        project_scope_key = build_project_scope_key(project_code_display, line_name, project_display)
        if "project_code" not in df.columns:
            df["project_code"] = project_code_display
        if "line_name" not in df.columns:
            df["line_name"] = line_name
        elif line_name:
            df["line_name"] = line_name
        if "project_name" not in df.columns:
            df["project_name"] = project_display
        elif line_name:
            df["project_name"] = project_display
        if "project_display" not in df.columns:
            df["project_display"] = project_display
        elif line_name:
            df["project_display"] = project_display
        if "project_scope_key" not in df.columns:
            df["project_scope_key"] = project_scope_key
        elif line_name:
            df["project_scope_key"] = project_scope_key
        if "project" not in df.columns:
            df["project"] = df["project_name"]
        df["_source_file"] = f.name
        df["source_sheet"] = found or ""

        try:
            compiled_norm, norm_report = normalize_stringing_columns(df)
        except Exception:
            compiled_norm = df.copy()
            norm_report = {"normalized_columns_ok": False, "present": [], "missing": [], "applied_map": {}}
        classification = classify_stringing_missing_headers(norm_report)
        critical_missing = list(classification.get("critical_missing", []))
        non_critical_missing = list(classification.get("non_critical_missing", []))
        if critical_missing and fallback_template_map and not template_applied:
            fallback_df, fallback_changes = _apply_template_column_mapping(df, fallback_template_map)
            try:
                fallback_norm, fallback_report = normalize_stringing_columns(fallback_df)
            except Exception:
                fallback_norm = fallback_df.copy()
                fallback_report = {"normalized_columns_ok": False, "present": [], "missing": [], "applied_map": {}}
            fallback_classification = classify_stringing_missing_headers(fallback_report)
            fallback_critical = list(fallback_classification.get("critical_missing", []))
            if len(fallback_critical) < len(critical_missing):
                compiled_norm = fallback_norm
                norm_report = fallback_report
                classification = fallback_classification
                critical_missing = fallback_critical
                non_critical_missing = list(fallback_classification.get("non_critical_missing", []))
                template_fallback_used = True
                template_changes = fallback_changes
                template_sheet_used = fallback_template_sheet or template_sheet_used

        missing_headers = list(norm_report.get("missing", []))
        if critical_missing:
            issue_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found,
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name,
                    "LineNameSource": line_name_source,
                    "Issue": "MISSING_REQUIRED_COLUMNS",
                    "MissingColumns": ", ".join(critical_missing),
                    "Rows": int(len(df.index)),
                    "DailyRows": 0,
                }
            )
        elif non_critical_missing:
            issue_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found,
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name,
                    "LineNameSource": line_name_source,
                    "Issue": "MISSING_NONCRITICAL_COLUMNS",
                    "MissingColumns": ", ".join(non_critical_missing),
                    "Rows": int(len(df.index)),
                    "DailyRows": 0,
                }
            )

        compiled_norm, method_inference_summary = infer_missing_methods_from_erection(
            compiled_norm,
            erection_daily_reference,
        )
        method_inference_rows = int(method_inference_summary.get("method_inferred_rows", 0) or 0)
        compiled.append(compiled_norm)

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

            def _filled(series: pd.Series) -> pd.Series:
                return series.notna() & series.astype(str).str.strip().ne("")

            po_start_filled = _filled(work["po_start_date"]) if has_po_start else pd.Series(False, index=work.index)
            po_complete_filled = _filled(work["po_completion_date"]) if has_po_complete else pd.Series(False, index=work.index)
            fs_start_filled = _filled(work["fs_starting_date"]) if has_fs_start else pd.Series(False, index=work.index)
            fs_complete_filled = _filled(work["fs_complete_date"]) if has_fs_complete else pd.Series(False, index=work.index)

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
                    "_source_file",
                    "source_sheet",
                ]
                for col in columns_for_issues:
                    if col not in work.columns:
                        work[col] = pd.NA

                def _mk_issue(row: pd.Series) -> str:
                    messages: List[str] = []
                    if has_po_start and pd.isna(row.get("po_start_date")):
                        messages.append("Missing/Invalid PO Start Date")
                    if has_po_complete and pd.isna(row.get("po_completion_date")):
                        messages.append("Missing/Invalid PO Complete Date")
                    if has_fs_start and pd.isna(row.get("fs_starting_date")):
                        messages.append("Missing/Invalid F/S Start Date")
                    if has_fs_complete and pd.isna(row.get("fs_complete_date")):
                        messages.append("Missing/Invalid F/S Complete Date")
                    if has_po_start and has_po_complete:
                        try:
                            if (row["po_completion_date"] - row["po_start_date"]).days + 1 <= 0:
                                messages.append("PO Start > PO Complete (non-positive duration)")
                        except Exception:
                            pass
                    if has_fs_start and has_fs_complete:
                        try:
                            if (row["fs_complete_date"] - row["fs_starting_date"]).dt.days + 1 <= 0:
                                messages.append("F/S Start > F/S Complete (non-positive duration)")
                        except Exception:
                            pass
                    if has_po_complete and pd.notna(row.get("po_completion_date")) and row["po_completion_date"] >= today:
                        messages.append("PO Completion >= today (future)")
                    if has_fs_complete and pd.notna(row.get("fs_complete_date")) and row["fs_complete_date"] >= today:
                        messages.append("F/S Completion >= today (future)")
                    return "; ".join(messages)

                issue_df = work.loc[any_issue, columns_for_issues].copy()
                issue_df = issue_df.rename(columns={"_source_file": "source_file"})
                issue_df["Issues"] = issue_df.apply(_mk_issue, axis=1)
                data_issue_rows.append(issue_df)

            if has_po_start and has_fs_complete:
                stage_days = (work["fs_complete_date"] - work["po_start_date"]).dt.days + 1
                valid_stage = (
                    work["po_start_date"].notna()
                    & work["fs_complete_date"].notna()
                    & (stage_days > 0)
                )
                daily_rows = int(stage_days.loc[valid_stage].sum()) if valid_stage.any() else 0
            else:
                daily_rows = 0
        except Exception:
            daily_rows = 0

        if daily_rows == 0 and not critical_missing:
            issue_rows.append(
                {
                    "Workbook": f.name,
                    "Project": project,
                    "Sheet": found,
                    "ConfiguredSheet": configured_sheet_name,
                    "LineName": line_name,
                    "LineNameSource": line_name_source,
                    "Issue": "NO_DAILY",
                    "MissingColumns": "",
                    "Rows": int(len(df.index)),
                    "DailyRows": 0,
                }
            )

        if critical_missing:
            status = "MISSING_REQUIRED_COLUMNS"
        elif daily_rows == 0:
            status = "NO_DAILY"
        else:
            status = "FALLBACK" if (fallback_note or template_fallback_used) else "OK"
        diag_rows.append(
            {
                "Workbook": f.name,
                "Project": project,
                "Sheet": found,
                "ConfiguredSheet": configured_sheet_name,
                "LineName": line_name,
                "LineNameSource": line_name_source,
                "DetectedHeaderRow": "" if header_row is None else int(header_row),
                "ColumnsDetected": ", ".join(header_labels),
                "NormalizedColumnsOk": bool(norm_report.get("normalized_columns_ok", False)),
                "PresentColumns": ", ".join(norm_report.get("present", [])),
                "MissingColumns": ", ".join(norm_report.get("missing", [])),
                "AppliedMap": ", ".join(
                    [f"{key}->{value}" for key, value in norm_report.get("applied_map", {}).items()]
                ),
                "Rows": int(len(compiled_norm.index)),
                "DailyRows": daily_rows,
                "Status": status,
                "FallbackNote": fallback_note or "",
                "TemplateSheet": template_sheet_used or "",
                "TemplateApplied": bool(template_applied or template_fallback_used),
                "TemplateFallbackUsed": bool(template_fallback_used),
                "TemplateChanges": "; ".join(template_changes),
                "MethodInferenceRows": int(method_inference_rows),
            }
        )

    if not compiled and missing:
        empty_df = pd.DataFrame()
        _ = _write_stringing_artifacts(
            output_path,
            empty_df,
            used_name or preferred or "Stringing Compiled",
            diagnostics_df=pd.DataFrame(diag_rows),
            issues_df=pd.DataFrame(issue_rows),
            data_issues_df=pd.concat(data_issue_rows, ignore_index=True) if data_issue_rows else pd.DataFrame(),
        )
        print("[pipeline] Stringing: no sheets found; wrote empty diagnostics workbook.")
        return None

    if not compiled:
        print("[pipeline] Stringing: nothing to compile.")
        return None

    all_df = pd.concat(compiled, ignore_index=True, copy=False)
    parquet_dir = _write_stringing_artifacts(
        output_path,
        all_df,
        used_name or preferred or "Stringing Compiled",
        source_files=list(dict.fromkeys(path for path, _, _, _, _, _ in selected_candidates)),
        diagnostics_df=pd.DataFrame(diag_rows),
        issues_df=pd.DataFrame(issue_rows),
        data_issues_df=pd.concat(data_issue_rows, ignore_index=True) if data_issue_rows else pd.DataFrame(),
    )
    return parquet_dir


def compile_progress_status_to_workbook(
    input_dir: Optional[Path],
    files: Optional[List[Path]],
    output_path: Path,
) -> Optional[Path]:
    try:
        result = progress_status_ingest.compile_progress_status_to_workbook(
            input_dir,
            files,
            output_path,
            repo_root=BASE_DIR,
        )
    except Exception as exc:
        print(f"[pipeline] ProgressStatus: failed to compile from DPRs: {exc}")
        return None
    return result

def _reload_dashboard_data(dashboard_module: Any, workbook_path: Path) -> None:
    """Reload the dashboard dataframe and recompute derived fields."""
    df = dashboard_module.load_daily(workbook_path)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    if hasattr(dashboard_module, "set_df_day"):
        dashboard_module.set_df_day(df)
    else:
        dashboard_module.df_day = df


def _clear_stringing_artifacts(base_dir: Path) -> None:
    if base_dir.name == "Erection" and base_dir.parent.name == "Parquets":
        stringing_base = base_dir.parent / "Stringing"
    else:
        stringing_base = base_dir / "Stringing" if base_dir.name != "Stringing" else base_dir

    targets = [
        stringing_base / "StringingCompiled_Output.xlsx",
        stringing_base / "StringingCompiled.parquet",
        stringing_base / "StringingDaily.parquet",
        stringing_base / "StringingDaily" / "stringing_daily.parquet",
        stringing_base / "StringingDaily",
    ]

    for target in targets:
        try:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        except Exception as exc:
            print(f"[pipeline] Warning: failed to remove stringing artifact {target}: {exc}")




def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the erection compiled pipeline then launch the dashboard."
    )
    parser.add_argument("--input", help="Folder containing the source Excel files.")
    parser.add_argument("--files", nargs="+", help="Explicit list of Excel files (overrides --input).")
    parser.add_argument("--output", help="Destination Excel workbook path.")
    parser.add_argument("--skip-compile", action="store_true", help="Launch dashboard without re-running the pipeline.")
    parser.add_argument(
        "--force-stringing-rebuild",
        action="store_true",
        help="Delete cached Stringing outputs before compiling (forces DPR re-read).",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.json",
        help="Configuration file relative to the project root (default: pipeline_config.json)."
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Do not start the Dash dev server after pipeline; just prepare app module.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        help="Additional CLI arguments to forward to the pipeline script."
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = _resolve_path(args.config, BASE_DIR)
    if config_path is None:
        raise SystemExit("Unable to resolve configuration path.")

    config = DEFAULT_CONFIG.copy()
    config.update(_load_config(config_path))

    cli_files = args.files
    config_files = config.get("files") if isinstance(config.get("files"), list) else None
    files = cli_files or config_files

    cli_input = args.input
    config_input = config.get("input_directory")
    config_microplan_input = config.get("microplan_directory")

    if files and cli_input:
        parser.error("Provide either --files or --input, not both.")
    if files and config_input and cli_files is None:
        parser.error("Configuration specifies both files and input_directory; please keep only one.")

    env_input = os.getenv("PIPELINE_INPUT_DIR")
    env_output = os.getenv("PIPELINE_OUTPUT_FILE")
    env_microplan_input = os.getenv("MICROPLAN_INPUT_DIR")

    input_path = cli_input or env_input or config_input
    output_path = args.output or env_output or config.get("output_file")
    microplan_input = env_microplan_input or config_microplan_input

    if input_path is None and not files:
        parser.error("An input directory or explicit file list is required for the pipeline.")
    if output_path is None:
        parser.error("An output path is required for the pipeline.")

    resolved_files = _normalise_files(files, BASE_DIR)
    resolved_input = _resolve_path(input_path, BASE_DIR) if input_path else None
    resolved_output = _resolve_path(output_path, BASE_DIR)
    resolved_microplan_input = _resolve_path(microplan_input, BASE_DIR) if microplan_input else None

    extra_args: List[str] = []
    if isinstance(config.get("pipeline_extra_args"), list):
        extra_args.extend(str(v) for v in config["pipeline_extra_args"])
    if args.extra_args:
        extra_args.extend(args.extra_args)



    stringing_out_path: Path | None = None
    progress_status_out_path: Path | None = None

    if not args.skip_compile:
        # Ensure fresh outputs on every compile run
        # 1) Remove any existing compiled workbook
        # 2) Remove any existing parquet dataset directory
        if resolved_output:
            try:
                if resolved_output.exists() and resolved_output.is_file():
                    print(f"[pipeline] Removing existing workbook: {resolved_output}")
                    resolved_output.unlink()
            except Exception as exc:
                print(f"[pipeline] Warning: failed to remove workbook {resolved_output}: {exc}")

            try:
                parquet_dir = resolved_output.parent / f"{resolved_output.stem}_parquet"
                if parquet_dir.exists() and parquet_dir.is_dir():
                    print(f"[pipeline] Removing existing parquet dataset: {parquet_dir}")
                    shutil.rmtree(parquet_dir)
            except Exception as exc:
                print(f"[pipeline] Warning: failed to remove parquet dir {parquet_dir}: {exc}")

        if args.force_stringing_rebuild:
            _clear_stringing_artifacts(resolved_output.parent if resolved_output else BASE_DIR)

        if resolved_input:
            print(f"[pipeline] Compiling from folder: {resolved_input}")
        if resolved_files:
            print("[pipeline] Compiling from files\n  - " + "\n  - ".join(str(p) for p in resolved_files))
        print(f"[pipeline] Writing output to: {resolved_output}")
        run_pipeline(
            input_path=resolved_input,
            files=[str(p) for p in resolved_files] if resolved_files else None,
            output_path=str(resolved_output) if resolved_output else None,
            extra_args=extra_args,
        )
                # --- NEW: Compile Micro Plan responsibilities into the same workbook ---
        # Prefer the input folder; if the user passed explicit files, derive a common parent
        if resolved_microplan_input:
            micro_input_dir = str(resolved_microplan_input)
        elif resolved_files:
            common_parent = os.path.commonpath([os.path.dirname(str(p)) for p in resolved_files])
            micro_input_dir = common_parent
        elif resolved_input:
            micro_input_dir = str(resolved_input)
        else:
            micro_input_dir = None

        if micro_input_dir:
            print(f"[pipeline] MicroPlan: scanning '{micro_input_dir}' and writing to '{resolved_output}'")
            compile_microplans_to_workbook(
                input_dir=micro_input_dir,
                output_path=str(resolved_output),
            )
        else:
            print("[pipeline] MicroPlan: no input directory configured; skipping.")

        if resolved_output:
            try:
                parquet_dir = export_workbook_to_parquet(resolved_output)
            except Exception as exc:
                print(f"[pipeline] Failed to export parquet dataset: {exc}")
                parquet_dir = None
        else:
            parquet_dir = None

        # --- Compile Stringing from the DPR sources ---
        # Always attempt to refresh micro plan outputs even if DPR compilation fails.
        # Many sites currently have malformed DPRs, but we still want the micro plans.
        base_dir = resolved_output.parent if resolved_output else BASE_DIR
        if base_dir.name == "Erection" and base_dir.parent.name == "Parquets":
            stringing_base = base_dir.parent / "Stringing"
        else:
            # Fallback: put stringing next to the current base
            stringing_base = base_dir / "Stringing" if base_dir.name != "Stringing" else base_dir
        stringing_base.mkdir(parents=True, exist_ok=True)
        stringing_out = stringing_base / "StringingCompiled_Output.xlsx"
        stringing_out_path = stringing_out

        stringing_input = resolved_input
        stringing_files = resolved_files

        try:
            print(f"[pipeline] Stringing: compiling to {stringing_out}")
            stringing_parquet_dir = compile_stringing_to_workbook(
                stringing_input,
                stringing_files,
                stringing_out,
                sheet_name=AppConfig().stringing_sheet_name,
            )
            if stringing_parquet_dir:
                print(f"[pipeline] Stringing: compiled parquet at {stringing_parquet_dir}")
        except Exception as exc:
            print(f"[pipeline] Stringing: failed to compile from DPRs: {exc}")

        if micro_input_dir:
            try:
                print(f"[pipeline] Stringing MicroPlan: scanning '{micro_input_dir}' and writing to '{stringing_out}'")
                compile_stringing_microplans_to_workbook(
                    input_dir=micro_input_dir,
                    output_path=str(stringing_out),
                )
            except Exception as exc:
                print(f"[pipeline] Stringing MicroPlan: failed to compile micro plans: {exc}")
        else:
            print("[pipeline] Stringing MicroPlan: no input directory configured; skipping.")

        if stringing_out.exists():
            try:
                export_workbook_to_parquet(
                    stringing_out,
                    sheets=("MicroPlanResponsibilities", "MicroPlanIndex", "MicroPlanDataIssues"),
                )
            except Exception as exc:
                print(f"[pipeline] Stringing: failed to export Micro Plan sheets to parquet: {exc}")

        # --- Compile Progress Status from the DPR sources ---
        if base_dir.name == "Erection" and base_dir.parent.name == "Parquets":
            progress_status_base = base_dir.parent / "ProgressStatus"
        else:
            progress_status_base = base_dir / "ProgressStatus" if base_dir.name != "ProgressStatus" else base_dir
        progress_status_base.mkdir(parents=True, exist_ok=True)
        progress_status_out = progress_status_base / "ProgressStatus_Output.xlsx"
        progress_status_out_path = progress_status_out
        print(f"[pipeline] ProgressStatus: compiling to {progress_status_out}")
        compiled_status = compile_progress_status_to_workbook(
            stringing_input,
            stringing_files,
            progress_status_out,
        )
        if compiled_status and compiled_status.exists():
            try:
                export_workbook_to_parquet(
                    compiled_status,
                    sheets=("RawData", "Diagnostics", "Issues", "Coverage"),
                )
            except Exception as exc:
                print(f"[pipeline] ProgressStatus: failed to export parquet: {exc}")
    else:
        print("[pipeline] Skipping compilation step as requested.")
        parquet_dir = None

    dataset_path: Path | None = resolved_output
    if parquet_dir:
        dataset_path = Path(parquet_dir)
    elif resolved_output:
        candidate_dir = resolved_output.parent / f"{resolved_output.stem}_parquet"
        if candidate_dir.exists():
            dataset_path = candidate_dir
    if dataset_path:
        print(f"[pipeline] Using dataset path: {dataset_path}")
    else:
        print("[pipeline] Dataset path unresolved; using dashboard defaults.")

    dash_host = os.getenv("DASH_HOST", config.get("dash_host", "0.0.0.0"))
    dash_port = int(os.getenv("DASH_PORT", config.get("dash_port", 8050)))
    dash_debug = os.getenv("DASH_DEBUG", str(config.get("dash_debug", False))).lower() in ("1", "true", "yes")


    print("[dashboard] Loading Dash app...")
    dashboard = import_module("app")

    if dataset_path is not None:
        dataset_target = Path(dataset_path)
    else:
        dataset_target = Path(dashboard.DATA_PATH)

    if stringing_out_path and stringing_out_path.exists():
        stringing_target = stringing_out_path
    else:
        stringing_target = Path(dashboard.CONFIG.stringing_data_path)

    new_config = dashboard.AppConfig(data_path=dataset_target, stringing_data_path=stringing_target)
    dashboard.CONFIG = new_config
    dashboard.DATA_PATH = dataset_target
    dashboard.app = dashboard.create_app(new_config)
    dashboard.server = dashboard.app.server

    _reload_dashboard_data(dashboard, dashboard.DATA_PATH)
    print(f"[dashboard] Dataset path configured: {dashboard.DATA_PATH}")

    if not args.no_serve:
        print(f"[dashboard] Starting server on http://{dash_host}:{dash_port}")
        dashboard.app.run_server(host=dash_host, port=dash_port, debug=dash_debug)
    else:
        print("[dashboard] Skipping dev server (--no-serve). Data reloaded and app module ready.")


if __name__ == "__main__":
    main()


