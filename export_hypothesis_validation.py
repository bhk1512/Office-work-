#!/usr/bin/env python3
"""Standalone CLI to export hypothesis validation tables to Excel."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.analytics import build_analytics_payload
from dashboard.config import AppConfig, configure_logging
from dashboard.state import AppDataStore

LOGGER = logging.getLogger("hypothesis_export")

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Hypothesis_Validation.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an Excel workbook with hypothesis validation tables from the "
            "erection analytics payload."
        )
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
        help="Optional override for the erection dataset root (default from AppConfig).",
    )
    parser.add_argument(
        "--include-stringing",
        action="store_true",
        help="Opt in to preloading stringing datasets during bootstrap.",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = PRODUCTIVITY_ROOT / path
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _table_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    return pd.DataFrame()


def _write_sheet(writer: pd.ExcelWriter, sheet_name: str, table: Any) -> None:
    frame = _table_frame(table)
    frame.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> int:
    args = _parse_args()
    configure_logging()

    config = AppConfig()
    if args.data_path:
        config = replace(config, data_path=Path(args.data_path).expanduser())
    if config.enable_stringing and not args.include_stringing:
        config = replace(config, enable_stringing=False)
    config.validate()

    output_path = _resolve_output_path(args.output)

    store = AppDataStore(config)
    LOGGER.info("Bootstrapping datasets from %s", config.data_path)
    store.bootstrap(config)

    daily_df = store.get_daily()
    payload = build_analytics_payload(daily_df)
    hypothesis = payload.get("hypothesis", {}) if isinstance(payload, dict) else {}

    h1 = hypothesis.get("h1_crosswalk", {}) if isinstance(hypothesis, dict) else {}
    h2 = hypothesis.get("h2_idle_underutilization", {}) if isinstance(hypothesis, dict) else {}
    h3_diag = hypothesis.get("h3_stint_diagnostics", {}) if isinstance(hypothesis, dict) else {}
    h3_scenario = hypothesis.get("h3_consolidation_scenario", {}) if isinstance(hypothesis, dict) else {}
    row_proxy = hypothesis.get("row_cooccurrence_proxy", {}) if isinstance(hypothesis, dict) else {}

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        _write_sheet(writer, "H1_Crosswalk_Gang", h1.get("by_gang_crosswalk", []))
        _write_sheet(writer, "H1_Crosswalk_Defs", h1.get("definition_summary", []))
        _write_sheet(writer, "H1_Bucket_Imbalance", h1.get("bucket_imbalance", []))

        _write_sheet(writer, "H2_Tiers", h2.get("tiers", []))
        _write_sheet(writer, "H2_HighVsLow", h2.get("delta_high_vs_low", {}))

        _write_sheet(writer, "H3_Stint_Diagnostics", h3_diag)
        _write_sheet(writer, "H3_Consolidation_Stint", h3_scenario.get("per_stint_scenario", []))
        _write_sheet(writer, "H3_Consolidation_Sum", h3_scenario.get("scenario_summary", {}))

        _write_sheet(writer, "ROW_ProjectMonth", row_proxy.get("project_month_summary", []))
        _write_sheet(writer, "ROW_Proxy_Summary", row_proxy.get("proxy_summary", {}))

    LOGGER.info("Hypothesis workbook exported to %s", output_path)
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
