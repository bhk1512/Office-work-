"""Application configuration and logging utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WORKBOOK = Path("ErectionCompiled_Output_testRun.xlsx")


def _resolve_default_data_path() -> Path:
    parquet_dir = DEFAULT_WORKBOOK.parent / f"{DEFAULT_WORKBOOK.stem}_parquet"
    if parquet_dir.exists():
        return parquet_dir
    return DEFAULT_WORKBOOK


_DEFAULT_DATA_PATH = _resolve_default_data_path()


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration for the productivity dashboard."""

    data_path: Path = _DEFAULT_DATA_PATH
    preferred_sheet: str = "ProdDailyExpandedSingles"
    default_benchmark: float = 9.0
    loss_max_gap_days: int = 15


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the application."""

    if logging.getLogger().handlers:
        logging.getLogger(__name__).debug("Logging already configured; skipping reconfiguration.")
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
