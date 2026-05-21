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
    load_erection_raw,
    load_foundation_completions,
    load_foundation_coverage,
    load_foundation_diagnostics,
    load_progress_status_raw,
    load_stringing_compiled_raw,
    load_stringing_summary_table,
    load_stretch_readiness_manpower_audit,
    load_stretch_readiness_summary,
)
from dashboard.foundation_delay_analysis import (
    MechanismConfig,
    build_complete_foundation_analysis_tables,
    write_foundation_delay_analysis_workbook,
)

LOGGER = logging.getLogger("foundation_delay_export")

BASE_DIR = Path(__file__).resolve().parent
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_OUTPUT = PRODUCTIVITY_ROOT / "Foundation_Delay_Analysis.xlsx"


def _parse_month_list(raw: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return fallback
    values: list[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        token_text = token.strip()
        if not token_text:
            continue
        try:
            month = int(token_text)
        except ValueError:
            continue
        if month < 1 or month > 12 or month in seen:
            continue
        seen.add(month)
        values.append(month)
    return tuple(values) if values else fallback


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
    parser.add_argument(
        "--pre-monsoon-months",
        type=str,
        help="Comma-separated months for pre-monsoon cohort window (default: 4,5).",
    )
    parser.add_argument(
        "--monsoon-months",
        type=str,
        help="Comma-separated months for monsoon cohort window (default: 6,7,8,9).",
    )
    parser.add_argument(
        "--post-monsoon-months",
        type=str,
        help="Comma-separated months for primary post-monsoon start window (default: 10,11).",
    )
    parser.add_argument(
        "--post-monsoon-wide-months",
        type=str,
        help="Comma-separated months for wide post-monsoon start window (default: 10,11,12).",
    )
    parser.add_argument(
        "--dry-months",
        type=str,
        help="Comma-separated months for dry-season cohort window (default: 12,1,2,3).",
    )
    parser.add_argument(
        "--min-foundation-count",
        type=int,
        help="Minimum foundations required per cohort window for mechanism summary rows (default: 5).",
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
    raw_erection_df = load_erection_raw(config)
    daily_df = load_daily(config)
    foundation_completions = load_foundation_completions(config)
    foundation_coverage = load_foundation_coverage(config)
    foundation_diagnostics = load_foundation_diagnostics(config)
    progress_status_raw = load_progress_status_raw(config)
    stringing_status_activity = load_stringing_summary_table(config, "StatusActivityFact")
    stringing_manpower_fact = load_stringing_summary_table(config, "ManpowerProductivityFact")
    stringing_compiled_raw = load_stringing_compiled_raw(config)
    stretch_readiness_summary = load_stretch_readiness_summary(config)
    stretch_manpower_audit = load_stretch_readiness_manpower_audit(config)
    mechanism_config = MechanismConfig(
        pre_monsoon=_parse_month_list(args.pre_monsoon_months, (4, 5)),
        monsoon=_parse_month_list(args.monsoon_months, (6, 7, 8, 9)),
        post_monsoon=_parse_month_list(args.post_monsoon_months, (10, 11)),
        post_monsoon_wide=_parse_month_list(args.post_monsoon_wide_months, (10, 11, 12)),
        dry=_parse_month_list(args.dry_months, (12, 1, 2, 3)),
        min_foundation_count=max(int(args.min_foundation_count), 0) if args.min_foundation_count is not None else 5,
    )

    tables = build_complete_foundation_analysis_tables(
        raw_erection_source=raw_erection_df,
        foundation_completions=foundation_completions,
        foundation_coverage=foundation_coverage,
        foundation_diagnostics=foundation_diagnostics,
        progress_status_raw=progress_status_raw,
        stringing_status_activity_fact=stringing_status_activity,
        stringing_manpower_fact=stringing_manpower_fact,
        stringing_compiled_raw=stringing_compiled_raw,
        stretch_readiness_summary=stretch_readiness_summary,
        stretch_readiness_manpower_audit=stretch_manpower_audit,
        daily_reference=daily_df,
        mechanism_config=mechanism_config,
    )
    output = _resolve_output_path(args.output)
    written = write_foundation_delay_analysis_workbook(output, tables)
    LOGGER.info("Foundation delay analysis exported to %s", written)
    print(f"[export] Wrote '{written}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
