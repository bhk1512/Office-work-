#!/usr/bin/env python3
"""Standalone CLI to export Erection->Stringing delay analysis workbook."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dashboard.config import AppConfig, configure_logging
from dashboard.data_loader import (
    load_daily,
    load_stringing_compiled_raw,
    load_stringing_summary_table,
    load_stretch_readiness_manpower_audit,
    load_stretch_readiness_summary,
)
from dashboard.erection_stringing_delay_analysis import (
    build_erection_stringing_delay_tables,
    write_erection_stringing_delay_workbook,
)

LOGGER = logging.getLogger("erection_stringing_delay_export")

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Erection_Stringing_Delay_Analysis.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone Erection->Stringing delay analysis workbook."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional override for erection dataset root (typically Parquets/Erection).",
    )
    parser.add_argument(
        "--stringing-data-path",
        type=Path,
        help="Optional override for stringing dataset root.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Optional YYYY-MM-DD start date to filter by PO Start Date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Optional YYYY-MM-DD end date to filter by PO Start Date.",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = PRODUCTIVITY_ROOT / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_date(value: str | None, label: str) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:
        raise SystemExit(f"Invalid {label} '{value}': {exc}") from exc


def main() -> int:
    args = _parse_args()
    configure_logging()

    config = AppConfig()
    if args.data_path:
        config = replace(config, data_path=Path(args.data_path).expanduser())
    if args.stringing_data_path:
        config = replace(config, stringing_data_path=Path(args.stringing_data_path).expanduser())
    config.validate()

    start_date = _parse_date(args.start_date, "start-date")
    end_date = _parse_date(args.end_date, "end-date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("Invalid date range: --start-date cannot be after --end-date.")

    LOGGER.info("Loading erection daily dataset from %s", config.data_path)
    erection_daily = load_daily(config)
    LOGGER.info("Loading stringing compiled dataset from %s", config.stringing_data_path)
    stringing_compiled_raw = load_stringing_compiled_raw(config)
    stringing_status_activity = load_stringing_summary_table(config, "StatusActivityFact")
    stringing_manpower_fact = load_stringing_summary_table(config, "ManpowerProductivityFact")
    stretch_readiness_summary = load_stretch_readiness_summary(config)
    stretch_manpower_audit = load_stretch_readiness_manpower_audit(config)

    tables = build_erection_stringing_delay_tables(
        stringing_compiled_raw=stringing_compiled_raw,
        erection_daily=erection_daily,
        stringing_status_activity_fact=stringing_status_activity,
        stringing_manpower_fact=stringing_manpower_fact,
        stretch_readiness_summary=stretch_readiness_summary,
        stretch_readiness_manpower_audit=stretch_manpower_audit,
        start_date=start_date,
        end_date=end_date,
        method_scope="all",
    )

    output = _resolve_output_path(args.output)
    written = write_erection_stringing_delay_workbook(output, tables)
    LOGGER.info("Erection->Stringing delay analysis exported to %s", written)
    print(f"[export] Wrote '{written}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
