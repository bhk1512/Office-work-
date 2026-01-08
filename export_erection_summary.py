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


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_PCH_PATH = BASE_DIR / "Raw Data" / "Projects and PCH.xlsx"
PRODUCTIVITY_ROOT = BASE_DIR / "Productivity Summaries"
DEFAULT_PRODUCTIVITY_OUTPUT = PRODUCTIVITY_ROOT / "Erection_Productivity_Summary.xlsx"


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
        default=DEFAULT_PRODUCTIVITY_OUTPUT,
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
        "--start-month",
        type=str,
        help="Optional YYYY-MM start month for a continuous range (used with --end-month).",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        help="Optional YYYY-MM end month for a continuous range (used with --start-month).",
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
    parser.add_argument(
        "--project-pch-path",
        type=Path,
        help=(
            "Optional override for the Projects/PCH mapping workbook "
            "(default: Raw Data/Projects and PCH.xlsx)."
        ),
    )
    parser.add_argument(
        "--gang-min-erections",
        type=int,
        default=4,
        help="Minimum erections for a gang to appear in project rankings (default: %(default)s).",
    )
    return parser.parse_args()


def _resolve_output_path(candidate: Path) -> Path:
    """
    Ensure erection productivity exports always land under Productivity Summaries unless
    the caller explicitly provides an absolute target elsewhere.
    """

    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        resolved = PRODUCTIVITY_ROOT / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_project_pch_mapping(mapping_path: Path | None) -> pd.DataFrame:
    if mapping_path is None:
        return pd.DataFrame()

    candidate = Path(mapping_path).expanduser()
    try:
        frame = pd.read_excel(candidate)
    except FileNotFoundError:
        LOGGER.warning("Projects/PCH workbook was not found at %s; falling back to Project Details data.", candidate)
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected formats
        LOGGER.warning("Unable to read Projects/PCH workbook '%s': %s", candidate, exc)
        return pd.DataFrame()

    if frame.empty:
        LOGGER.warning("Projects/PCH workbook at %s is empty; falling back to Project Details data.", candidate)
        return pd.DataFrame()

    def _match_column(keywords: tuple[str, ...]) -> str:
        for column in frame.columns:
            label = str(column).strip().lower()
            if any(keyword in label for keyword in keywords):
                return column
        raise KeyError

    try:
        project_col = _match_column(("project",))
        pch_col = _match_column(("pch",))
    except KeyError:
        LOGGER.warning(
            "Projects/PCH workbook '%s' is missing required columns (need both project and PCH); fallback enabled.",
            candidate,
        )
        return pd.DataFrame()

    working = (
        frame[[project_col, pch_col]]
        .rename(columns={project_col: "project_code", pch_col: "pch"})
        .copy()
    )
    working["project_code"] = working["project_code"].fillna("").astype(str).str.strip()
    working["pch"] = working["pch"].fillna("").astype(str).str.strip()
    working = working[(working["project_code"].astype(bool)) & (working["pch"].astype(bool))]
    if working.empty:
        LOGGER.warning(
            "Projects/PCH workbook '%s' has no usable rows after cleaning; falling back to Project Details data.",
            candidate,
        )
        return pd.DataFrame()

    working["project_name"] = working["project_code"]
    working["key_name"] = working["project_code"]

    LOGGER.info("Loaded %d project-to-PCH mappings from %s", len(working), candidate)
    return working.reset_index(drop=True)


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

    mapping_path = args.project_pch_path or DEFAULT_PROJECT_PCH_PATH
    project_pch_df = _load_project_pch_mapping(mapping_path)

    as_of = pd.Timestamp(args.as_of_date) if args.as_of_date else None
    range_start = None
    range_end = None
    if args.start_month or args.end_month:
        if not (args.start_month and args.end_month):
            raise SystemExit("Both --start-month and --end-month are required to set a month range.")
        try:
            range_start = pd.Timestamp(f"{args.start_month}-01")
            range_end = pd.Timestamp(f"{args.end_month}-01") + pd.offsets.MonthEnd(1)
        except Exception as exc:
            raise SystemExit(f"Invalid month range values: {exc}") from exc
        if range_start > range_end:
            raise SystemExit("--start-month must be before or equal to --end-month.")
    output_path = _resolve_output_path(args.output)
    export_erection_productivity_summary(
        output_path=output_path,
        data_store=store,
        as_of_date=as_of,
        sheet_name=args.sheet_name,
        project_info=project_pch_df if not project_pch_df.empty else None,
        range_start=range_start,
        range_end=range_end,
        gang_min_erections=args.gang_min_erections,
    )
    LOGGER.info("Erection summary exported to %s", output_path)
    print(f"[export] Wrote '{output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
