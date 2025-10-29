"""Dash application entry point."""
from __future__ import annotations

import json
import logging
import os
from pickle import NONE
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import psutil
from dash import Dash
from flask import request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from pythonjsonlogger import jsonlogger
from werkzeug.middleware.proxy_fix import ProxyFix

from dashboard.callbacks import register_callbacks
from dashboard.config import AppConfig, configure_logging
from dashboard.data_loader import (
    load_daily as _load_daily,
    load_project_details,
    get_project_baseline_maps,
    find_parquet_source,
    is_parquet_dataset,
    read_parquet_table,
    load_stringing_compiled_raw as _load_stringing_compiled_raw,
    load_stringing_daily as _load_stringing_daily,
)
from dashboard.layout import build_layout
from dashboard.stringing import (
    normalize_stringing_columns,
    summarize_date_parsing,
    add_length_units,
)


LOGGER = logging.getLogger(__name__)


def _ensure_json_logging() -> None:
    """Attach a JSON formatter to the root logger if not already present."""

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_is_json_handler", False):
            return

    json_handler = logging.StreamHandler()
    json_handler.setFormatter(jsonlogger.JsonFormatter())
    json_handler._is_json_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(json_handler)


CONFIG = AppConfig()
CONFIG.validate()
DATA_PATH: Path = CONFIG.data_path

df_day: pd.DataFrame | None = None
LAST_UPDATED_DATE: pd.Timestamp | None = None
LAST_UPDATED_TEXT: str = "N/A"

df_projinfo: pd.DataFrame | None = None
def get_df_projinfo() -> pd.DataFrame:
    return df_projinfo if df_projinfo is not None else pd.DataFrame()
def get_project_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Return cached project productivity baselines."""
    return get_project_baseline_maps()


def _normalize_text(value: object) -> str:
    text = str(value).replace("\u00a0", " ").strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    return text


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


def _normalize_location(value: object) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    return text



df_responsibilities: pd.DataFrame | None = None
RESP_COMPLETED_KEYS: set[tuple[str, str]] = set()
RESP_LOAD_ERROR: str | None = None


def _set_responsibilities_data(
    df: pd.DataFrame | None,
    completion_keys: set[tuple[str, str]],
    error_message: str | None,
) -> None:
    global df_responsibilities, RESP_COMPLETED_KEYS, RESP_LOAD_ERROR
    df_responsibilities = None if df is None else df.copy()
    RESP_COMPLETED_KEYS = set(completion_keys)
    RESP_LOAD_ERROR = error_message


def get_responsibilities_df() -> pd.DataFrame:
    if df_responsibilities is None:
        raise RuntimeError(RESP_LOAD_ERROR or "Responsibilities data not loaded.")
    return df_responsibilities.copy()


def get_responsibilities_completion_keys() -> set[tuple[str, str]]:
    return set(RESP_COMPLETED_KEYS)


def get_responsibilities_error() -> str | None:
    return RESP_LOAD_ERROR


def _load_responsibilities_data(
    config: AppConfig,
) -> tuple[pd.DataFrame | None, set[tuple[str, str]], str | None]:
    path = Path(config.data_path)

    df_atomic: pd.DataFrame | None = None
    df_daily: pd.DataFrame | None = None

    if is_parquet_dataset(path):
        try:
            resp_source = find_parquet_source(path, "MicroPlanResponsibilities")
        except Exception:
            resp_source = None

        if not resp_source:
            LOGGER.warning("Sheet '%s' missing in parquet dataset", "MicroPlanResponsibilities")
            return pd.DataFrame(), set(), "No Micro Plan data found in the compiled dataset."

        try:
            df_atomic = read_parquet_table(resp_source)
        except FileNotFoundError:
            LOGGER.warning("Responsibilities parquet not found near: %s", path)
            return pd.DataFrame(), set(), "No Micro Plan data found in the compiled dataset."
        except Exception as exc:
            LOGGER.exception("Failed to load responsibilities parquet: %s", exc)
            return None, set(), "Unable to load Micro Plan data."

        candidates = [config.preferred_sheet, "ProdDailyExpandedSingles", "ProdDailyExpanded"]
        for candidate in [c for c in candidates if c]:
            source = find_parquet_source(path, candidate)
            if not source:
                continue
            try:
                df_daily = read_parquet_table(source)
                break
            except Exception as exc:
                LOGGER.warning("Failed to load daily dataset '%s': %s", candidate, exc)
    else:
        try:
            workbook = pd.ExcelFile(path)
        except FileNotFoundError:
            LOGGER.warning("Responsibilities workbook not found: %s", config.data_path)
            return None, set(), "Compiled workbook not found."
        except Exception as exc:
            LOGGER.exception("Failed to open responsibilities workbook: %s", exc)
            return None, set(), "Unable to load Micro Plan data."

        sheet_name = "MicroPlanResponsibilities"
        if sheet_name not in workbook.sheet_names:
            LOGGER.warning("Sheet '%s' missing in workbook", sheet_name)
            return pd.DataFrame(), set(), "No Micro Plan data found in the compiled workbook."

        df_atomic = pd.read_excel(workbook, sheet_name=sheet_name)

        candidates = [config.preferred_sheet, "ProdDailyExpandedSingles", "ProdDailyExpanded"]
        daily_sheet = next(
            (
                candidate
                for candidate in candidates
                if candidate and candidate in workbook.sheet_names
            ),
            None,
        )
        if daily_sheet:
            try:
                df_daily = pd.read_excel(workbook, sheet_name=daily_sheet, usecols=None)
            except Exception as exc:
                LOGGER.warning("Failed to load daily sheet '%s': %s", daily_sheet, exc)

    completion_keys: set[tuple[str, str]] = set()

    if df_daily is not None and not df_daily.empty:

        def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
            mapping = {str(col).strip().lower(): col for col in frame.columns}
            for candidate in candidates:
                key = candidate.strip().lower()
                if key in mapping:
                    return mapping[key]
            for key, original in mapping.items():
                if any(cand.lower() in key for cand in candidates):
                    return original
            raise KeyError(candidates)

        try:
            col_proj = _pick_column(df_daily, ("project_name", "project"))
            col_loc = _pick_column(
                df_daily, ("location_no", "location number", "location")
            )
        except KeyError:
            LOGGER.warning(
                "Daily dataset missing project/location columns; delivered will rely on realised values only."
            )
        else:
            cleaned_projects = df_daily[col_proj].map(_normalize_lower)
            cleaned_locations = df_daily[col_loc].map(_normalize_location)
            completion_keys = {
                (p, loc) for p, loc in zip(cleaned_projects, cleaned_locations) if p and loc
            }
    else:
        LOGGER.info(
            "Daily expanded data not found; delivered values fall back to realised revenue only."
        )

    return df_atomic, completion_keys, None



def _update_last_updated_metadata(df: pd.DataFrame) -> None:
    """Update module-level metadata derived from *df*."""

    global LAST_UPDATED_DATE, LAST_UPDATED_TEXT
    last_date = df["date"].max()
    LAST_UPDATED_DATE = pd.to_datetime(last_date) if pd.notna(last_date) else None
    LAST_UPDATED_TEXT = (
        LAST_UPDATED_DATE.strftime("%d-%m-%Y") if LAST_UPDATED_DATE is not None else "N/A"
    )


def set_df_day(df: pd.DataFrame) -> None:
    """Set the global daily dataframe and refresh derived metadata."""

    global df_day
    df_day = df
    _update_last_updated_metadata(df)


def get_df_day() -> pd.DataFrame:
    """Return the current daily dataframe."""

    if df_day is None:
        raise RuntimeError("Daily dataframe not loaded.")
    return df_day


def get_last_updated_text() -> str:
    """Return the last updated text for health probes."""

    return LAST_UPDATED_TEXT


def load_daily(config_or_path) -> pd.DataFrame:  # type: ignore[override]
    """Compatibility wrapper around the refactored data loader."""

    return _load_daily(config_or_path)


def initialise_data(config: AppConfig) -> Tuple[pd.DataFrame, str]:
    """Load the productivity dataset and compute display metadata."""

    df = _load_daily(config)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    set_df_day(df)
    LOGGER.info("Loaded %d daily rows", len(df))
    global df_projinfo
    df_projinfo = load_project_details(config.data_path)
    resp_df, resp_keys, resp_error = _load_responsibilities_data(config)
    _set_responsibilities_data(resp_df, resp_keys, resp_error)


    # --- add project_code into df_day by joining to ProjectDetails on project_name (normalized) ---
    if df_projinfo is not None and not df_projinfo.empty and "project_name" in df.columns:
        df_norm = df.copy()
        df_norm["__key_name__"] = (
            df_norm["project_name"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
        )
        map_df = df_projinfo[["key_name", "project_code"]].dropna().drop_duplicates("key_name")
        df_norm = df_norm.merge(map_df, left_on="__key_name__", right_on="key_name", how="left")
        df_norm = df_norm.drop(columns=["__key_name__", "key_name"])
        set_df_day(df_norm)
        df = df_norm
    
    logging.getLogger(__name__).info(
        "Loaded workbook: %s | df_day rows=%s, cols=%s | projinfo rows=%s",
        config.data_path, len(df), list(df.columns), 0 if df_projinfo is None else len(df_projinfo)
    )

    return df, LAST_UPDATED_TEXT


def create_app(config: AppConfig | None = None) -> Dash:
    """Create and configure the Dash application instance."""

    configure_logging()
    _ensure_json_logging()

    active_config = config or AppConfig()
    active_config.validate()
    _, last_updated_text = initialise_data(active_config)

    app_instance = Dash(__name__)
    app_instance.title = "KEC Productivity"
    app_instance.layout = build_layout(last_updated_text)

    server = app_instance.server
    server.config["SECRET_KEY"] = active_config.secret_key
    server.config["SESSION_COOKIE_SECURE"] = True
    server.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    server.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

    if active_config.app_env == "production":
        if active_config.behind_proxy:
            server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1)

        # Strict CSP for prod; leave it OFF in dev to avoid blocking Dash inline JS.
        Talisman(
            server,
            force_https=True,
            strict_transport_security=True,
            frame_options="DENY",
            content_security_policy={
                "default-src": "'self'",
                "img-src": "'self' data:",
                "style-src": "'self' 'unsafe-inline'",
                "script-src": "'self' 'unsafe-inline' 'unsafe-eval'",
                "connect-src": "'self'",
                "font-src": "'self'",
            },
        )

        Limiter(get_remote_address, app=server, default_limits=["120/minute"])
    else:
        # DEV: no CSP, no limiter, no proxy fix (keeps things simple)
        pass
    
    @server.before_request
    def _capture_request_start() -> None:
        request.environ["request_start_time"] = time.perf_counter()

    @server.after_request
    def _log_request(response):  # type: ignore[override]
        start_time = request.environ.get("request_start_time")
        duration_ms = 0.0
        if start_time is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000

        content_length = response.calculate_content_length() or 0
        log_data = {
            "event": "http_request",
            "path": request.path,
            "method": request.method,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "content_length": content_length,
        }
        LOGGER.info("request", extra=log_data)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @server.get("/__/health")
    def healthcheck():  # type: ignore[override]
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        # Probe stringing dataset safely (non-expanding, cached)
        stringing_df = pd.DataFrame()
        try:
            stringing_df = _load_stringing_compiled_raw(CONFIG)
        except Exception:
            # Keep health lightweight and resilient; treat as not found on errors
            stringing_df = pd.DataFrame()
        stringing_found = not stringing_df.empty
        stringing_rows = int(len(stringing_df.index)) if stringing_found else 0

        # Column normalization probe for Stringing dataset (map-only)
        normalized_ok = False
        normalized_missing: list[str] = []
        date_parse_metrics = {
            "po_start_date_parsed_count": 0,
            "fs_complete_date_parsed_count": 0,
            "invalid_date_rows": 0,
        }
        length_metrics = {
            "total_length_km": 0.0,
            "min_length_km": 0.0,
            "max_length_km": 0.0,
        }
        stringing_daily_rows = 0
        if stringing_found:
            normalized_df = stringing_df
            try:
                normalized_df, norm_report = normalize_stringing_columns(stringing_df)
                normalized_ok = bool(norm_report.get("normalized_columns_ok", False))
                normalized_missing = list(norm_report.get("missing", []))
            except Exception:
                normalized_ok = False
                normalized_missing = []
            # Date parse metrics (no expansion) using the same semantics as erection dates
            try:
                date_parse_metrics = summarize_date_parsing(stringing_df)
            except Exception:
                date_parse_metrics = {
                    "po_start_date_parsed_count": 0,
                    "fs_complete_date_parsed_count": 0,
                    "invalid_date_rows": 0,
                }
            # Length sanity in km (units-only normalization)
            try:
                _, length_metrics = add_length_units(normalized_df)
            except Exception:
                length_metrics = {
                    "total_length_km": 0.0,
                    "min_length_km": 0.0,
                    "max_length_km": 0.0,
                }
            # Expanded per-day stringing row count (parquet-first, Excel fallback)
            try:
                daily_df = _load_stringing_daily(CONFIG)
                stringing_daily_rows = int(len(daily_df.index))
            except Exception:
                stringing_daily_rows = 0
        payload = {
            "status": "ok",
            "rss_mb": round(rss_mb, 2),
            "last_updated": get_last_updated_text(),
            # Stringing placeholders (no reads yet)
            "stringing_enabled": bool(CONFIG.enable_stringing),
            "stringing_sheet": CONFIG.stringing_sheet_name,
            "stringing_parquet_dirs": list(getattr(CONFIG, "stringing_parquet_dirs", ())),
            # Stub reader probe results
            "stringing_sheet_found": bool(stringing_found),
            "stringing_row_count": stringing_rows,
            # Normalization report
            "normalized_columns_ok": bool(normalized_ok),
            "normalized_missing": normalized_missing,
            # Date parsing metrics
            "po_start_date_parsed_count": int(date_parse_metrics.get("po_start_date_parsed_count", 0)),
            "fs_complete_date_parsed_count": int(date_parse_metrics.get("fs_complete_date_parsed_count", 0)),
            "invalid_date_rows": int(date_parse_metrics.get("invalid_date_rows", 0)),
            # Length metrics (km)
            "total_length_km": float(length_metrics.get("total_length_km", 0.0)),
            "min_length_km": float(length_metrics.get("min_length_km", 0.0)),
            "max_length_km": float(length_metrics.get("max_length_km", 0.0)),
            # Stringing daily expanded probe
            "stringing_daily_rows": int(stringing_daily_rows),
        }

        # Emit a log line summarizing normalization and date parsing health
        try:
            LOGGER.info(
                "stringing_health",
                extra={
                    "event": "stringing_health",
                    "found": bool(stringing_found),
                    "rows": stringing_rows,
                    "normalized_columns_ok": bool(normalized_ok),
                    "normalized_missing": normalized_missing,
                    "po_start_date_parsed_count": payload["po_start_date_parsed_count"],
                    "fs_complete_date_parsed_count": payload["fs_complete_date_parsed_count"],
                    "invalid_date_rows": payload["invalid_date_rows"],
                    "total_length_km": payload["total_length_km"],
                    "min_length_km": payload["min_length_km"],
                    "max_length_km": payload["max_length_km"],
                    "stringing_daily_rows": payload["stringing_daily_rows"],
                },
            )
        except Exception:
            pass
        return server.response_class(
            response=json.dumps(payload),
            status=200,
            mimetype="application/json",
        )

    @server.get("/__/ready")
    def readiness():  # type: ignore[override]
        status = 200
        rows = 0
        try:
            daily_df = get_df_day()
        except RuntimeError:
            status = 503
        else:
            rows = len(daily_df.index)
            if daily_df.empty:
                status = 503

        payload = {
            "status": "ok" if status == 200 else "unavailable",
            "rows": rows,
        }
        return server.response_class(
            response=json.dumps(payload),
            status=status,
            mimetype="application/json",
        )
    
    @server.errorhandler(403)
    def _forbidden(e): return {"error":"forbidden"}, 403

    @server.errorhandler(500)
    def _ise(e): return {"error":"internal server error"}, 500
    
    register_callbacks(
        app_instance,
        get_df_day,
        active_config,
        project_info_provider=get_df_projinfo,
        project_baseline_provider=get_project_baselines,
        responsibilities_provider=get_responsibilities_df,
        responsibilities_completion_provider=get_responsibilities_completion_keys,
        responsibilities_error_provider=get_responsibilities_error,
    )
    return app_instance


def main() -> None:
    """Run the Dash development server."""

    app.run_server(host="0.0.0.0", port=8050, debug=False)


app = create_app(CONFIG)
server = app.server


if __name__ == "__main__":
    main()
