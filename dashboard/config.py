"""Application configuration and logging utilities."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WORKBOOK = Path("ErectionCompiled_Output.xlsx")


def _resolve_default_data_path() -> Path:
    parquet_dir = DEFAULT_WORKBOOK.parent / f"{DEFAULT_WORKBOOK.stem}_parquet"
    if parquet_dir.exists():
        return parquet_dir
    return DEFAULT_WORKBOOK


_DEFAULT_DATA_PATH = _resolve_default_data_path()


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration sourced from environment variables or defaults."""

    # Security and auth
    secret_key: str = os.getenv("SECRET_KEY", "change-me")
    oidc_issuer: str | None = os.getenv("OIDC_ISSUER") or None
    oidc_client_id: str | None = os.getenv("OIDC_CLIENT_ID") or None
    oidc_client_secret: str | None = os.getenv("OIDC_CLIENT_SECRET") or None
    allowed_groups_admin: str = os.getenv("ADMIN_GROUP", "dash-admins")
    allowed_groups_view: str = os.getenv("VIEW_GROUP", "dash-viewers")

    # Caching and processing
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))
    cache_maxsize: int = int(os.getenv("CACHE_MAXSIZE", "16"))
    loss_max_gap_days: int = 15

    # Runtime environment
    app_env: str = os.getenv("APP_ENV", "development")
    enable_https: bool = os.getenv("ENABLE_HTTPS", "0") == "1"
    behind_proxy: bool = os.getenv("BEHIND_PROXY", "0") == "1"

    # Data selection and defaults
    preferred_sheet: str = "ProdDailyExpandedSingles"
    default_benchmark: float = 9.0
    data_path: Path = _DEFAULT_DATA_PATH
    allowed_data_root: Path = Path(os.getenv("ALLOWED_DATA_ROOT", ".")).resolve()

    def validate(self) -> None:
        """Ensure configured paths stay within the permitted root."""

        resolved_root = Path(self.allowed_data_root).expanduser().resolve()
        resolved_data = Path(self.data_path).expanduser().resolve()

        if resolved_root != resolved_data and resolved_root not in resolved_data.parents:
            raise ValueError(
                f"DATA_PATH '{resolved_data}' must reside inside ALLOWED_DATA_ROOT '{resolved_root}'."
            )


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the application."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.debug("Logging already configured; skipping reconfiguration.")
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
