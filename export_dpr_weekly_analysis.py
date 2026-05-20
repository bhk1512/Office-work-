#!/usr/bin/env python3
"""CLI utility to build a weekly DPR analysis workbook."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dashboard.config import configure_logging
from dashboard.workbook import export_weekly_dpr_analysis

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
WEEKLY_SUMMARY_DIR = PRODUCTIVITY_ROOT / "Weekly"
DEFAULT_WEEKLY_OUTPUT = WEEKLY_SUMMARY_DIR / "DPR_Weekly_Analysis.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a weekly DPR analysis workbook showing completions, "
            "manpower, and tower details for a specific project."
        )
    )
    parser.add_argument("project_code", help="Project code to analyze (e.g., TA414).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_WEEKLY_OUTPUT,
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--dpr-path",
        type=Path,
        action="append",
        help="Explicit DPR workbook path (can be provided multiple times).",
    )
    parser.add_argument(
        "--dpr-folder",
        type=Path,
        help="Folder to search for DPR files when --dpr-path is not provided (default: Raw Data/DPRs).",
    )
    parser.add_argument(
        "--daily-path",
        type=Path,
        default=Path("Parquets") / "Erection" / "ProdDailyExpandedSingles.parquet",
        help="Compiled erection daily dataset (parquet/xlsx) to prefer over DPR files (default: %(default)s).",
    )
    parser.add_argument(
        "--dpr-only",
        action="store_true",
        help="Ignore the compiled daily dataset and read DPR workbooks only.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Optional YYYY-MM-DD to anchor the month (default: latest completion date in the DPR).",
    )
    parser.add_argument(
        "--week-mode",
        type=str,
        default="legacy",
        choices=["legacy", "calendar_sun_sat"],
        help="Week bucketing mode (default: %(default)s).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Optional YYYY-MM-DD weekly scope start date (used in calendar_sun_sat mode).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Optional YYYY-MM-DD weekly scope end date (requires --start-date).",
    )
    parser.add_argument(
        "--previous-week",
        action="store_true",
        help="Use the previous completed Sunday-Saturday week (calendar_sun_sat mode).",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    """
    Normalize relative output paths so weekly summaries always land under Productivity Summaries/Weekly.
    """

    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = WEEKLY_SUMMARY_DIR / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    args = _parse_args()
    configure_logging()

    output_path = _resolve_output_path(args.output)
    dpr_paths = [path.expanduser().resolve() for path in args.dpr_path] if args.dpr_path else None
    dpr_folder = args.dpr_folder.expanduser().resolve() if args.dpr_folder else None
    as_of = pd.Timestamp(args.as_of_date) if args.as_of_date else None
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None
    if end_date is not None and start_date is None:
        raise SystemExit("--end-date requires --start-date.")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("--start-date must be before or equal to --end-date.")
    daily_path = None
    if not args.dpr_only and args.daily_path:
        daily_path = args.daily_path.expanduser().resolve()

    week_mode = args.week_mode
    if week_mode == "legacy" and (args.previous_week or args.start_date or args.end_date):
        week_mode = "calendar_sun_sat"

    export_weekly_dpr_analysis(
        project_code=args.project_code,
        output_path=output_path,
        dpr_paths=dpr_paths,
        dpr_folder=dpr_folder,
        as_of_date=as_of,
        daily_path=daily_path,
        week_mode=week_mode,
        week_start_date=start_date,
        week_end_date=end_date,
        previous_week=bool(args.previous_week),
    )
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
