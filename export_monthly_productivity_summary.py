#!/usr/bin/env python3
"""CLI utility to build a monthly productivity workbook using the compiled dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from dashboard.config import configure_logging
from dashboard.workbook import export_monthly_productivity_summary

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
MONTHLY_SUMMARY_DIR = PRODUCTIVITY_ROOT / "Monthly"
DEFAULT_MONTHLY_OUTPUT = MONTHLY_SUMMARY_DIR / "Project_Monthly_Productivity.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a month-wise erection productivity workbook using only "
            "ErectionCompiled_Output.xlsx."
        )
    )
    parser.add_argument("project_code", help="Project code to analyze (e.g., TA414).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_MONTHLY_OUTPUT,
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--compiled-path",
        type=Path,
        default=Path("Parquets") / "Erection" / "ErectionCompiled_Output.xlsx",
        help="Path to ErectionCompiled_Output.xlsx (default: %(default)s).",
    )
    parser.add_argument(
        "--sheet-summary",
        type=str,
        default="MonthlySummary",
        help="Name of the summary sheet in the workbook (default: %(default)s).",
    )
    parser.add_argument(
        "--sheet-details",
        type=str,
        default="MonthlyDetails",
        help="Name of the detail sheet in the workbook (default: %(default)s).",
    )
    parser.add_argument(
        "--sheet-context",
        type=str,
        default="SelectionContext",
        help="Name of the context sheet in the workbook (default: %(default)s).",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    """
    Route relative monthly export paths into Productivity Summaries/Monthly automatically.
    """

    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = MONTHLY_SUMMARY_DIR / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    args = _parse_args()
    configure_logging()

    output_path = _resolve_output_path(args.output)
    compiled_path = args.compiled_path.expanduser().resolve()
    export_monthly_productivity_summary(
        project_code=args.project_code,
        output_path=output_path,
        compiled_path=compiled_path,
        sheet_summary=args.sheet_summary,
        sheet_details=args.sheet_details,
        sheet_context=args.sheet_context,
    )
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
