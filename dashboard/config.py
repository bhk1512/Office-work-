"""Application configuration and logging utilities."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


# Default workbook path used as a legacy fallback. The application now
# prefers the Parquets/Erection layout and writes/reads from there.
DEFAULT_WORKBOOK = Path("Parquets") / "Erection" / "ErectionCompiled_Output.xlsx"


def _parse_csv_env(name: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of values."""

    raw = os.getenv(name, "")
    if not raw:
        return ()
    parts: list[str] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if value:
            parts.append(value)
    return tuple(parts)


def _resolve_default_data_path() -> Path:
    """Resolve the default dataset root.

    Priority:
    1) New standard layout: Parquets/Erection (preferred root for erection data)
    2) Legacy layout: ErectionCompiled_Output_parquet directory next to workbook
    3) Legacy workbook path: ErectionCompiled_Output.xlsx
    """
    # 1) Preferred: Parquets/Erection
    parquets_root = Path("Parquets")
    erection_root = parquets_root / "Erection"
    try:
        if erection_root.exists():
            # Use it if it already contains parquet files (common in builds)
            if any(erection_root.rglob("*.parquet")) or any(erection_root.rglob("*.parq")) or any(erection_root.rglob("*.pq")):
                return erection_root
            # Even if empty, prefer the directory so writers will populate it
            return erection_root
    except Exception:
        # Fall through to legacy paths if any FS error occurs
        pass

    # 2) Legacy compiled parquet directory
    parquet_dir = DEFAULT_WORKBOOK.parent / f"{DEFAULT_WORKBOOK.stem}_parquet"
    if parquet_dir.exists():
        return parquet_dir

    # 3) Legacy workbook file
    return DEFAULT_WORKBOOK


_DEFAULT_DATA_PATH = _resolve_default_data_path()


def _resolve_default_stringing_data_path() -> Path:
    """Resolve the default dataset root for stringing artifacts."""

    override = os.getenv("STRINGING_DATA_PATH")
    if override:
        return Path(override)
    return Path("Parquets") / "Stringing" / "StringingCompiled_Output.xlsx"


_DEFAULT_STRINGING_DATA_PATH = _resolve_default_stringing_data_path()


def _resolve_default_stringing_summary_data_path() -> Path:
    """Resolve the default dataset root for unified stringing summary artifacts."""

    override = os.getenv("STRINGING_SUMMARY_DATA_PATH")
    if override:
        return Path(override)
    return Path("Parquets") / "StringingSummary" / "StringingSummary_Output.xlsx"


_DEFAULT_STRINGING_SUMMARY_DATA_PATH = _resolve_default_stringing_summary_data_path()
_DEFAULT_CSP_SCRIPT_SRC = ("https://unpkg.com", "https://cdn.plot.ly", "https://cdn.jsdelivr.net")
_DEFAULT_CSP_STYLE_SRC = (
    "https://fonts.googleapis.com",
    "https://cdn.jsdelivr.net",
    "https://unpkg.com",
    "https://cdn.plot.ly",
)
_DEFAULT_CSP_FONT_SRC = ("https://fonts.gstatic.com",)

# --- Idle Normalization ---
IDLE_NORM_DAYS_PER_MONTH: float = 30.44        # avg calendar days per month
IDLE_MAX_GAP_DAYS: int = 15                    # cap per idle window (was hardcoded in both engines)
IDLE_OFF_SYSTEM_GAP_DAYS: int = 45             # gaps longer than this = gang off-system, skip entirely
# (previously loss_max_gap_days in metrics.py - centralize here)
IDLE_MIN_COMPLETIONS_FOR_TIER: int = 3         # min completions to include gang in tier analysis
IDLE_BASELINE_ERECTION_FALLBACK: float = 5.0   # MT/day fallback
IDLE_BASELINE_GENERIC_FALLBACK: float = 1.0    # generic metric fallback
loss_max_gap_days = IDLE_OFF_SYSTEM_GAP_DAYS   # legacy alias, deprecate later


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
    idle_max_gap_days: int = IDLE_MAX_GAP_DAYS
    idle_off_system_gap_days: int = IDLE_OFF_SYSTEM_GAP_DAYS

    # Runtime environment
    app_env: str = os.getenv("APP_ENV", "development")
    enable_https: bool = os.getenv("ENABLE_HTTPS", "0") == "1"
    behind_proxy: bool = os.getenv("BEHIND_PROXY", "0") == "1"

    # Data selection and defaults
    preferred_sheet: str = "ProdDailyExpandedSingles"
    default_benchmark: float = 9.0
    exec_plan_green_pct: float = float(os.getenv("EXEC_PLAN_GREEN_PCT", "95"))
    exec_plan_amber_low_pct: float = float(os.getenv("EXEC_PLAN_AMBER_LOW_PCT", "80"))
    exec_readiness_green_pct: float = float(os.getenv("EXEC_READINESS_GREEN_PCT", "85"))
    exec_readiness_amber_low_pct: float = float(os.getenv("EXEC_READINESS_AMBER_LOW_PCT", "65"))
    exec_manpower_green_pct: float = float(os.getenv("EXEC_MANPOWER_GREEN_PCT", "70"))
    exec_manpower_amber_low_pct: float = float(os.getenv("EXEC_MANPOWER_AMBER_LOW_PCT", "40"))
    exec_gap_green_pct: float = float(os.getenv("EXEC_GAP_GREEN_PCT", "10"))
    exec_gap_amber_high_pct: float = float(os.getenv("EXEC_GAP_AMBER_HIGH_PCT", "20"))
    data_path: Path = _DEFAULT_DATA_PATH
    stringing_data_path: Path = _DEFAULT_STRINGING_DATA_PATH
    stringing_summary_data_path: Path = _DEFAULT_STRINGING_SUMMARY_DATA_PATH
    allowed_data_root: Path = Path(os.getenv("ALLOWED_DATA_ROOT", ".")).resolve()

    # Stringing (uses sibling folder under Parquets by default)
    enable_stringing: bool = os.getenv("ENABLE_STRINGING", "1") in {"1", "true", "True"}
    stringing_sheet_name: str = os.getenv("STRINGING_SHEET_NAME", "Stringing Compiled")
    # Comma-separated list of parquet directory names to probe when stringing is enabled
    stringing_parquet_dirs: tuple[str, ...] = tuple(
        d.strip()
        for d in os.getenv(
            "STRINGING_PARQUET_DIRS",
            # Prefer Parquets/Stringing (as sibling of Parquets/Erection). Fallbacks retained.
            "../Stringing,StringingCompiled_Output_parquet,Stringing_Output_parquet",
        ).split(",")
        if d.strip()
    )
    # Parquet table name for expanded per-day stringing rows (mirrors erection style)
    # A directory with this name is created under the dataset root, with parquet files inside.
    # Default to a sibling under Parquets/Stringing so reads/writes land there
    stringing_daily_table: str = os.getenv("STRINGING_DAILY_TABLE", "../Stringing/StringingDaily")

    # Display + CSP customisation
    display_timezone: str | None = os.getenv("DISPLAY_TIMEZONE", "Asia/Kolkata") or None
    csp_script_src: tuple[str, ...] = _DEFAULT_CSP_SCRIPT_SRC + _parse_csv_env("CSP_SCRIPT_SRC")
    csp_style_src: tuple[str, ...] = _DEFAULT_CSP_STYLE_SRC + _parse_csv_env("CSP_STYLE_SRC")
    csp_font_src: tuple[str, ...] = _DEFAULT_CSP_FONT_SRC + _parse_csv_env("CSP_FONT_SRC")
    csp_connect_src: tuple[str, ...] = _parse_csv_env("CSP_CONNECT_SRC")
    csp_img_src: tuple[str, ...] = _parse_csv_env("CSP_IMG_SRC")

    def __getattr__(self, name: str) -> object:
        legacy_key = "loss" + "_max_gap_days"
        if name == legacy_key:
            return IDLE_OFF_SYSTEM_GAP_DAYS
        raise AttributeError(name)

    def validate(self) -> None:
        """Ensure configured paths stay within the permitted root."""

        resolved_root = Path(self.allowed_data_root).expanduser().resolve()
        resolved_data = Path(self.data_path).expanduser().resolve()
        resolved_stringing = Path(self.stringing_data_path).expanduser().resolve()
        resolved_summary = Path(self.stringing_summary_data_path).expanduser().resolve()

        if resolved_root != resolved_data and resolved_root not in resolved_data.parents:
            raise ValueError(
                f"DATA_PATH '{resolved_data}' must reside inside ALLOWED_DATA_ROOT '{resolved_root}'."
            )

        if resolved_root != resolved_stringing and resolved_root not in resolved_stringing.parents:
            raise ValueError(
                f"STRINGING_DATA_PATH '{resolved_stringing}' must reside inside ALLOWED_DATA_ROOT '{resolved_root}'."
            )

        if resolved_root != resolved_summary and resolved_root not in resolved_summary.parents:
            raise ValueError(
                f"STRINGING_SUMMARY_DATA_PATH '{resolved_summary}' must reside inside ALLOWED_DATA_ROOT '{resolved_root}'."
            )


def resolve_log_level(value: str | None) -> int:
    if not value:
        return logging.WARNING
    text = value.strip().upper()
    if text.isdigit():
        return int(text)
    return logging._nameToLevel.get(text, logging.WARNING)


def configure_logging(level: int | None = None) -> None:
    """Configure root logging once for the application."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.debug("Logging already configured; skipping reconfiguration.")
        return

    effective_level = level if level is not None else resolve_log_level(os.getenv("LOG_LEVEL"))

    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

