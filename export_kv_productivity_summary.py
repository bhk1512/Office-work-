#!/usr/bin/env python3
"""Standalone CLI to export productivity grouped by KV and tower type."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dashboard.config import AppConfig, configure_logging
from dashboard.state import AppDataStore
from dashboard.workbook import export_voltage_tower_productivity_summary

LOGGER = logging.getLogger("kv_export")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the KV and tower-type productivity summary workbook.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("KV_Productivity_Summary.xlsx"),
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional override for the erection dataset root (Parquets/Erection).",
    )
    parser.add_argument(
        "--projects-kv",
        type=Path,
        default=Path("Raw Data") / "Projects_KV.xlsx",
        help="Path to the Projects_KV workbook (default: %(default)s).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Optional YYYY-MM-DD date to treat as 'today' for the export.",
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Optional YYYY-MM month to summarise (overrides --as-of-date).",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        help="Optional YYYY-MM start month for a continuous range (requires --end-month).",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        help="Optional YYYY-MM end month for a continuous range (requires --start-month).",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="KV Productivity",
        help="Sheet name for the output workbook (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-stringing",
        action="store_true",
        help="Skip loading stringing datasets to speed up the bootstrap.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging()

    config = AppConfig()
    if args.data_path:
        config = replace(config, data_path=Path(args.data_path).expanduser())
    if args.skip_stringing and config.enable_stringing:
        config = replace(config, enable_stringing=False)
    config.validate()

    store = AppDataStore(config)
    LOGGER.info("Bootstrapping datasets from %s", config.data_path)
    store.bootstrap(config)

    if args.month and (args.start_month or args.end_month):
        raise SystemExit("Choose either --month or the --start-month/--end-month range, not both.")
    if (args.start_month and not args.end_month) or (args.end_month and not args.start_month):
        raise SystemExit("--start-month and --end-month must be provided together.")
    if args.as_of_date and (args.month or args.start_month or args.end_month):
        raise SystemExit("--as-of-date cannot be combined with --month or --start-month/--end-month.")

    as_of: pd.Timestamp | None = None
    range_start: pd.Timestamp | None = None
    range_end: pd.Timestamp | None = None
    if args.month:
        try:
            as_of = pd.Timestamp(f"{args.month}-01")
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise SystemExit(f"Invalid --month value '{args.month}': {exc}") from exc
    elif args.start_month and args.end_month:
        try:
            range_start = pd.Timestamp(f"{args.start_month}-01")
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise SystemExit(f"Invalid --start-month value '{args.start_month}': {exc}") from exc
        try:
            range_end = pd.Timestamp(f"{args.end_month}-01")
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise SystemExit(f"Invalid --end-month value '{args.end_month}': {exc}") from exc
    elif args.as_of_date:
        try:
            as_of = pd.Timestamp(args.as_of_date)
        except Exception as exc:
            raise SystemExit(f"Invalid --as-of-date value '{args.as_of_date}': {exc}") from exc

    output_path = Path(args.output).expanduser().resolve()
    export_voltage_tower_productivity_summary(
        output_path=output_path,
        data_store=store,
        as_of_date=as_of,
        range_start=range_start,
        range_end=range_end,
        sheet_name=args.sheet_name,
        project_voltage_path=Path(args.projects_kv).expanduser(),
    )
    LOGGER.info("KV productivity summary exported to %s", output_path)
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
