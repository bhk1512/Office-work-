#!/usr/bin/env python3
"""Standalone CLI to export the erection productivity summary workbook."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dashboard.config import AppConfig, configure_logging
from dashboard.state import AppDataStore
from dashboard import workbook as workbook_module
from dashboard.workbook import export_erection_productivity_summary

LOGGER = logging.getLogger("erection_export")

# Backward-compatible shim: the workbook helper expects keyword-only arg but
# older code paths sometimes pass it positionally.
if hasattr(workbook_module, "_build_weekly_summary"):
    _orig_build_weekly_summary = workbook_module._build_weekly_summary

    def _shim_build_weekly_summary(scope, completions, group_cols, current_week_label, *args, **kwargs):
        return _orig_build_weekly_summary(
            scope,
            completions,
            group_cols,
            current_week_label=current_week_label,
            *args,
            **kwargs,
        )

    workbook_module._build_weekly_summary = _shim_build_weekly_summary  # type: ignore[attr-defined]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the PCH and Project-level erection productivity summary Excel "
            "without running the Dash server."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("Erection_Productivity_Summary.xlsx"),
        help="Target Excel file path (default: %(default)s).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional override for the erection dataset root (Parquets/Erection).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Optional YYYY-MM-DD date to treat as 'today' for the export.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="Erection Summary",
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

    as_of = pd.Timestamp(args.as_of_date) if args.as_of_date else None
    output_path = Path(args.output).expanduser().resolve()
    export_erection_productivity_summary(
        output_path=output_path,
        data_store=store,
        as_of_date=as_of,
        sheet_name=args.sheet_name,
    )
    LOGGER.info("Erection summary exported to %s", output_path)
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
