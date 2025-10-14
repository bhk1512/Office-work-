"""Dash application entry point."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from dash import Dash
import dash_bootstrap_components as dbc

from dashboard.callbacks import register_callbacks
from dashboard.config import AppConfig, configure_logging
from dashboard.data_loader import (
    load_daily as _load_daily,
    load_project_details,
    get_project_baseline_maps,
    find_parquet_source,
    is_parquet_dataset,
    read_parquet_table,
)
from dashboard.layout import build_layout


LOGGER = logging.getLogger(__name__)


CONFIG = AppConfig()
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
    active_config = config or AppConfig()
    _, last_updated_text = initialise_data(active_config)

    app_instance = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app_instance.title = "KEC Productivity"
    app_instance.layout = build_layout(last_updated_text)

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
app = create_app(CONFIG)
server = app.server


if __name__ == "__main__":
    main()







