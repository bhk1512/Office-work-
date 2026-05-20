#!/usr/bin/env python3
"""Standalone CLI to export Foundation Delay Analysis V2 workbook."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from dashboard.config import AppConfig, configure_logging
from dashboard.data_loader import (
    load_daily,
    load_foundation_completions,
    load_foundation_coverage,
    load_foundation_diagnostics,
    load_progress_status_raw,
)
from dashboard.foundation_delay_analysis import (
    build_foundation_delay_analysis_tables,
    write_foundation_delay_analysis_workbook,
)

LOGGER = logging.getLogger("foundation_delay_export")

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Foundation_Delay_Analysis.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone Foundation Delay Analysis workbook."
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
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = PRODUCTIVITY_ROOT / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    args = _parse_args()
    configure_logging()

    config = AppConfig()
    if args.data_path:
        config = replace(config, data_path=Path(args.data_path).expanduser())
    config.validate()

    LOGGER.info("Loading source datasets from %s", config.data_path)
    daily_df = load_daily(config)
    foundation_completions = load_foundation_completions(config)
    foundation_coverage = load_foundation_coverage(config)
    foundation_diagnostics = load_foundation_diagnostics(config)
    progress_status_raw = load_progress_status_raw(config)

    tables = build_foundation_delay_analysis_tables(
        source_daily=daily_df,
        foundation_completions=foundation_completions,
        foundation_coverage=foundation_coverage,
        foundation_diagnostics=foundation_diagnostics,
        progress_status_raw=progress_status_raw,
    )
    output = _resolve_output_path(args.output)
    written = write_foundation_delay_analysis_workbook(output, tables)
    LOGGER.info("Foundation delay analysis exported to %s", written)
    print(f"[export] Wrote '{written}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

