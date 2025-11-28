"""Production-grade ETL to extract Visual Chart progress summaries from DPR workbooks.

Run from shell or scheduler:
    export DPR_FOLDER=/path/to/dpr/files
    export OUTPUT_ROOT=/path/to/output
    python dpr_visual_chart_summary.py

Outputs a consolidated "Progress Summary.xlsx" that the dashboard can consume.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_ACTIVITIES = (
    "Foundation",
    "Earthing",
    "Erection",
    "Tack Welding",
    "Stringing (Rough Sag)",
    "Stringing (Final Sag)",
)

SUMMARY_COLUMNS = [
    "project_code",
    "project_name",
    "activity",
    "total",
    "completed",
    "balance",
]

_PUNCTUATION_RE = re.compile(r"[.,;:\-_/\\()\[\]]")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DPR_FOLDER = BASE_DIR / "Raw Data" / "DPRs"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "Raw Data"


def normalize_text(value: Any) -> str:
    """Normalize arbitrary text before matching."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value)
    text = text.lower().strip()
    if not text:
        return ""
    text = text.replace("qty", "quantity")
    text = _PUNCTUATION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_activity(raw: Any) -> str | None:
    """Map raw activity text to one of the canonical activities."""
    normalized = normalize_text(raw)
    if not normalized:
        return None
    if "foundation" in normalized:
        return "Foundation"
    if "earthing" in normalized:
        return "Earthing"
    if "erection" in normalized:
        return "Erection"
    if "tack" in normalized and "weld" in normalized:
        return "Tack Welding"
    if ("stringing" in normalized and "rough" in normalized) or "rough sag" in normalized:
        return "Stringing (Rough Sag)"
    if ("stringing" in normalized and "final" in normalized) or "final sag" in normalized:
        return "Stringing (Final Sag)"
    return None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _clean_str(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return str(value).strip()


def _normalized_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(normalize_text)


def _is_name_header(text: str) -> bool:
    if not text:
        return False
    return any(token in text for token in ("activity", "description", "description of item"))


def _is_total_header(text: str) -> bool:
    if not text:
        return False
    if "completed" in text or "balance" in text:
        return False
    return "total" in text or "scope" in text


def _is_completed_header(text: str) -> bool:
    return bool(text and "completed" in text)


def _is_balance_header(text: str) -> bool:
    return bool(text and "balance" in text)


def detect_header_row(normalized_df: pd.DataFrame) -> int | None:
    """Locate the single-row header within the top 20 rows."""
    max_rows = min(20, normalized_df.shape[0])
    for row_idx in range(max_rows):
        row = normalized_df.iloc[row_idx]
        if (
            any(_is_name_header(cell) for cell in row)
            and any(_is_total_header(cell) for cell in row)
            and any(_is_completed_header(cell) for cell in row)
            and any(_is_balance_header(cell) for cell in row)
        ):
            return row_idx
    return None


def get_header_columns(normalized_df: pd.DataFrame, header_row: int) -> dict[str, int]:
    mapping: dict[str, int] = {}
    row = normalized_df.iloc[header_row]
    for col_idx, text in row.items():
        text = text or ""
        if "name" not in mapping and _is_name_header(text):
            mapping["name"] = col_idx
            continue
        if "total" not in mapping and _is_total_header(text):
            mapping["total"] = col_idx
            continue
        if "completed" not in mapping and _is_completed_header(text):
            mapping["completed"] = col_idx
            continue
        if "balance" not in mapping and _is_balance_header(text):
            mapping["balance"] = col_idx
    return mapping if len(mapping) == 4 else {}


def detect_two_row_header(normalized_df: pd.DataFrame) -> tuple[int, dict[str, int]] | None:
    if normalized_df.shape[0] < 2:
        return None
    mapping: dict[str, int] = {}
    for col_idx in normalized_df.columns:
        top = normalized_df.iat[0, col_idx] or ""
        bottom = normalized_df.iat[1, col_idx] or ""
        combined = f"{top} {bottom}".strip()
        if "name" not in mapping and (_is_name_header(combined) or _is_name_header(top) or _is_name_header(bottom)):
            mapping["name"] = col_idx
        if "total" not in mapping and "scope" in combined and "total" in combined:
            mapping["total"] = col_idx
        if "completed" not in mapping and "completed" in combined and "total" in combined:
            mapping["completed"] = col_idx
        if "balance" not in mapping and "balance" in combined and "total" in combined:
            mapping["balance"] = col_idx
    if len(mapping) == 4:
        return 1, mapping
    return None


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        value = cleaned.replace(",", "")
    try:
        num = pd.to_numeric([value], errors="coerce")[0]
    except Exception:
        return None
    if pd.isna(num):
        return None
    return float(num)


def find_visual_chart_sheet(workbook: pd.ExcelFile) -> str | None:
    for sheet in workbook.sheet_names:
        if "visual chart" in normalize_text(sheet):
            return sheet
    return None


def get_project_meta(workbook: pd.ExcelFile, workbook_path: Path) -> dict[str, str | None]:
    fallback = workbook_path.stem
    meta = {"project_code": fallback, "project_name": fallback}
    for sheet in workbook.sheet_names:
        if "project details" not in normalize_text(sheet):
            continue
        try:
            df = pd.read_excel(workbook, sheet_name=sheet, engine="openpyxl")
        except Exception as exc:
            print(f"[WARN] Failed to read '{sheet}' for project metadata in '{workbook_path.name}': {exc}")
            return meta
        if df.empty:
            return meta
        data_row = None
        for idx in range(len(df)):
            row = df.iloc[idx]
            if any(not _is_blank(row[col]) for col in df.columns):
                data_row = row
                break
        if data_row is None:
            return meta
        for col in df.columns:
            header = normalize_text(col)
            value = _clean_str(data_row[col])
            if not value:
                continue
            if "project" in header and "code" in header:
                meta["project_code"] = value
            if "project" in header and "name" in header:
                meta["project_name"] = value
        return meta
    return meta


def extract_progress_from_sheet(
    df: pd.DataFrame,
    project_meta: dict[str, str | None],
    *,
    source: str | None = None,
) -> list[dict[str, Any]]:
    normalized_df = _normalized_view(df)
    header_row = detect_header_row(normalized_df)
    header_map: dict[str, int] = {}
    if header_row is not None:
        header_map = get_header_columns(normalized_df, header_row)
    if not header_map:
        two_row = detect_two_row_header(normalized_df)
        if two_row:
            header_row, header_map = two_row
    if not header_map:
        print(f"[WARN] Could not locate Visual Chart header in '{source or 'sheet'}'.")
        return []

    start_row = header_row + 1
    max_rows = min(df.shape[0], 200)
    records: list[dict[str, Any]] = []
    seen_activities: set[str] = set()
    for row_idx in range(start_row, max_rows):
        raw_name = df.iat[row_idx, header_map["name"]]
        normalized_name = normalize_text(raw_name)
        if not normalized_name:
            if records:
                break
            continue
        activity = normalize_activity(raw_name)
        if activity is None:
            continue
        if activity in seen_activities:
            continue
        record = {
            "project_code": project_meta.get("project_code"),
            "project_name": project_meta.get("project_name"),
            "activity": activity,
            "total": _coerce_numeric(df.iat[row_idx, header_map["total"]]),
            "completed": _coerce_numeric(df.iat[row_idx, header_map["completed"]]),
            "balance": _coerce_numeric(df.iat[row_idx, header_map["balance"]]),
        }
        records.append(record)
        seen_activities.add(activity)
        if len(seen_activities) == len(CANONICAL_ACTIVITIES):
            break
    if not records:
        print(f"[INFO] No matching activities found in '{source or 'sheet'}'.")
    return records


def process_dpr_file(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.name.startswith("~$"):
        return []
    print(f"[INFO] Processing {path}")
    try:
        workbook = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        print(f"[WARN] Failed to open '{path}': {exc}")
        return []
    try:
        sheet = find_visual_chart_sheet(workbook)
        if not sheet:
            print(f"[WARN] Visual Chart sheet not found in '{path.name}'. Skipping.")
            return []
        project_meta = get_project_meta(workbook, path)
        try:
            df = pd.read_excel(workbook, sheet_name=sheet, header=None, engine="openpyxl")
        except Exception as exc:
            print(f"[WARN] Failed to read Visual Chart sheet '{sheet}' in '{path.name}': {exc}")
            return []
        return extract_progress_from_sheet(df, project_meta, source=str(path))
    finally:
        workbook.close()


def build_progress_summary(dpr_folder: Path, output_path: Path) -> Path:
    dpr_folder = Path(dpr_folder)
    if not dpr_folder.is_dir():
        raise FileNotFoundError(f"DPR folder '{dpr_folder}' does not exist or is not a directory.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_resolved = output_path.resolve()

    records: list[dict[str, Any]] = []
    for workbook_path in sorted(dpr_folder.rglob("*.xlsx")):
        if workbook_path.name.startswith("~$"):
            continue
        try:
            if workbook_path.resolve() == target_resolved:
                continue
        except Exception:
            pass
        records.extend(process_dpr_file(workbook_path))

    if records:
        df = pd.DataFrame(records)[SUMMARY_COLUMNS]
        df = df.drop_duplicates()
    else:
        df = pd.DataFrame(columns=SUMMARY_COLUMNS)

    df.to_excel(output_path, index=False, sheet_name="Progress Summary")
    print(f"[INFO] Wrote {len(df)} rows to '{output_path}'.")
    return output_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Progress Summary from DPR workbooks.")
    parser.add_argument("--dpr-folder", type=Path, default=None, help="Folder containing DPR Excel files.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where the Progress Summary.xlsx file will be written.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit path to the Progress Summary workbook.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    dpr_folder = args.dpr_folder or os.getenv("DPR_FOLDER") or DEFAULT_DPR_FOLDER
    dpr_folder_path = Path(dpr_folder)

    output_file = args.output_file
    if output_file is None:
        output_root = args.output_root or os.getenv("OUTPUT_ROOT") or DEFAULT_OUTPUT_ROOT
        output_root_path = Path(output_root)
        output_file = output_root_path / "Progress Summary.xlsx"

    build_progress_summary(dpr_folder_path, output_file)


if __name__ == "__main__":
    main()
