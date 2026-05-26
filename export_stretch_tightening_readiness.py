#!/usr/bin/env python3
"""Standalone CLI to export stretch readiness from erection tower tightening."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from dashboard.config import AppConfig, configure_logging
from dashboard.data_loader import load_erection_raw, load_stringing_compiled_raw
from dashboard.stretch_tightening_readiness import (
    build_stretch_tightening_readiness_tables,
    write_stretch_tightening_readiness_workbook,
)

LOGGER = logging.getLogger("stretch_tightening_readiness_export")

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Stretch_Tightening_Readiness_Analysis.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone stretch readiness analysis from erection tower tightening."
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
    if args.stringing_data_path:
        config = replace(config, stringing_data_path=Path(args.stringing_data_path).expanduser())
    config.validate()

    LOGGER.info("Loading erection raw dataset from %s", config.data_path)
    erection_raw = load_erection_raw(config)
    LOGGER.info("Loading stringing compiled dataset from %s", config.stringing_data_path)
    stringing_compiled_raw = load_stringing_compiled_raw(config)

    tables = build_stretch_tightening_readiness_tables(
        erection_raw=erection_raw,
        stringing_compiled_raw=stringing_compiled_raw,
    )

    output = _resolve_output_path(args.output)
    written = write_stretch_tightening_readiness_workbook(output, tables)
    LOGGER.info("Stretch tightening readiness analysis exported to %s", written)
    print(f"[export] Wrote '{written}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
