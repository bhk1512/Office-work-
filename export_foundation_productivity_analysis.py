#!/usr/bin/env python3
"""Standalone CLI to export foundation productivity analysis workbook."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dashboard.foundation_productivity_analysis import (
    build_foundation_productivity_tables,
    write_foundation_productivity_workbook,
)


BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
PARQUET_ROOT = BASE_DIR / "Parquets"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Foundation_Productivity_Analysis.xlsx"
DEFAULT_PROJECT_PCH_PATH = BASE_DIR / "Raw Data" / "Projects and PCH.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone foundation productivity analysis workbook."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        help="Optional YYYY-MM start month for completion-date filtering.",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        help="Optional YYYY-MM end month for completion-date filtering.",
    )
    parser.add_argument(
        "--project-pch-path",
        type=Path,
        default=DEFAULT_PROJECT_PCH_PATH,
        help="Project/PCH mapping workbook path (default: %(default)s).",
    )
    parser.add_argument(
        "--foundation-dir",
        type=Path,
        default=PARQUET_ROOT / "Foundation",
        help="Foundation parquet/workbook output directory (default: %(default)s).",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = PRODUCTIVITY_ROOT / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_foundation_table(foundation_dir: Path, parquet_name: str, workbook_sheet: str) -> pd.DataFrame:
    parquet_path = foundation_dir / parquet_name
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    workbook_path = foundation_dir / "FoundationCompiled_Output.xlsx"
    if workbook_path.exists():
        return pd.read_excel(workbook_path, sheet_name=workbook_sheet)
    return pd.DataFrame()


def _read_mapping(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()


def main() -> int:
    args = _parse_args()
    if bool(args.start_month) ^ bool(args.end_month):
        raise SystemExit("Both --start-month and --end-month are required when filtering.")

    foundation_dir = args.foundation_dir.expanduser().resolve()
    completions = _read_foundation_table(foundation_dir, "FoundationCompletions.parquet", "FoundationCompletions")
    raw = _read_foundation_table(foundation_dir, "FoundationRaw.parquet", "FoundationRaw")
    coverage = _read_foundation_table(foundation_dir, "Coverage.parquet", "Coverage")
    mapping = _read_mapping(args.project_pch_path.expanduser())

    if completions.empty:
        raise SystemExit(f"No foundation completions found under {foundation_dir}")

    tables = build_foundation_productivity_tables(
        completions,
        raw,
        coverage,
        mapping,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    output = write_foundation_productivity_workbook(_resolve_output_path(args.output), tables)
    print(f"[export] Wrote '{output}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
