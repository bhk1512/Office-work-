"""Dash application entry point."""
from __future__ import annotations

import json
import logging
import os
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
    get_project_baseline_maps,
    load_stringing_compiled_raw as _load_stringing_compiled_raw,
    load_stringing_daily as _load_stringing_daily,
)
from dashboard.layout import build_layout
from dashboard.stringing import (
    normalize_stringing_columns,
    summarize_date_parsing,
    add_length_units,
)
from dashboard.state import AppDataStore


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
DATA_STORE = AppDataStore(CONFIG)


def get_df_projinfo() -> pd.DataFrame:
    return DATA_STORE.get_project_info()


def get_project_baselines() -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Return cached project productivity baselines."""
    return get_project_baseline_maps()


def get_responsibilities_df() -> pd.DataFrame:
    return DATA_STORE.get_responsibilities_frame()


def get_responsibilities_completion_keys() -> set[tuple[str, str]]:
    return DATA_STORE.get_responsibilities_completion_keys()


def get_responsibilities_error() -> str | None:
    return DATA_STORE.get_responsibilities_error()


def set_df_day(df: pd.DataFrame) -> None:
    """Compatibility shim retained for the pipeline runner."""

    DATA_STORE.set_daily(df)


def get_df_day() -> pd.DataFrame:
    """Return the current daily dataframe."""

    return DATA_STORE.get_daily()


def get_last_updated_text() -> str:
    """Return the app data load timestamp text (for header/health)."""

    return DATA_STORE.metadata.last_loaded_text


# --- Stringing dataset accessors (mirrors erection flow) ---
def set_df_stringing_day(df: pd.DataFrame) -> None:
    DATA_STORE.set_stringing(df)


def get_df_stringing_day() -> pd.DataFrame:
    return DATA_STORE.get_stringing()


def set_df_stringing_compiled(df: pd.DataFrame) -> None:
    DATA_STORE.set_stringing_compiled(df)


def get_df_stringing_compiled() -> pd.DataFrame:
    return DATA_STORE.get_stringing_compiled()


def load_daily(config_or_path) -> pd.DataFrame:  # type: ignore[override]
    """Compatibility wrapper around the refactored data loader."""

    return _load_daily(config_or_path)


def initialise_data(config: AppConfig) -> Tuple[pd.DataFrame, str]:
    """Load the productivity dataset and compute display metadata."""

    return DATA_STORE.bootstrap(config)


def create_app(config: AppConfig | None = None) -> Dash:
    """Create and configure the Dash application instance."""

    configure_logging()
    _ensure_json_logging()

    global CONFIG, DATA_PATH, DATA_STORE
    active_config = config or CONFIG
    active_config.validate()
    CONFIG = active_config
    DATA_PATH = active_config.data_path
    if config is not None:
        DATA_STORE = AppDataStore(active_config)

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
        def _merge_csp_values(*groups: tuple[str, ...] | list[str]) -> list[str]:
            merged: list[str] = []
            seen: set[str] = set()
            for group in groups:
                for value in group:
                    if value not in seen:
                        seen.add(value)
                        merged.append(value)
            return merged

        csp = {
            "default-src": ["'self'"],
            "img-src": ["'self'", "data:"],
            "style-src": ["'self'", "'unsafe-inline'"],
            "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
            "connect-src": ["'self'"],
            "font-src": ["'self'", "data:"],
        }

        csp["script-src"] = _merge_csp_values(csp["script-src"], active_config.csp_script_src)
        csp["style-src"] = _merge_csp_values(csp["style-src"], active_config.csp_style_src)
        csp["font-src"] = _merge_csp_values(csp["font-src"], active_config.csp_font_src)
        csp["connect-src"] = _merge_csp_values(csp["connect-src"], active_config.csp_connect_src)
        csp["img-src"] = _merge_csp_values(csp["img-src"], active_config.csp_img_src)

        csp_directives = {key: " ".join(values) for key, values in csp.items()}

        Talisman(
            server,
            force_https=True,
            strict_transport_security=True,
            frame_options="DENY",
            content_security_policy=csp_directives,
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
            stringing_df = get_df_stringing_compiled()
        except Exception:
            stringing_df = pd.DataFrame()
        if stringing_df.empty:
            try:
                stringing_df = _load_stringing_compiled_raw(CONFIG)
                if not stringing_df.empty:
                    set_df_stringing_compiled(stringing_df)
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
        # Provide a preloaded stringing data provider for instant switching
        duckdb_connection=DATA_STORE.get_duckdb_connection(),
        stringing_data_provider=get_df_stringing_day,
        stringing_compiled_provider=get_df_stringing_compiled,
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
