"""Application configuration and logging utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration for the productivity dashboard."""

    data_path: Path = Path("ErectionCompiled_Output.xlsx")
    preferred_sheet: str = "DailyExpanded"
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
