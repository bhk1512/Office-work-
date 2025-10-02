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
from dashboard.data_loader import load_daily as _load_daily, load_project_details
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

    # --- add project_code into df_day by joining to ProjectDetails on project_name (normalized) ---
    if df_projinfo is not None and not df_projinfo.empty and "project name" in df.columns:
        df_norm = df.copy()
        df_norm["__key_name__"] = (
            df_norm["project name"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
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

    register_callbacks(app_instance, get_df_day, active_config, project_info_provider=get_df_projinfo)
    return app_instance


def main() -> None:
    """Run the Dash development server."""

    app.run_server(host="0.0.0.0", port=8050, debug=False)


app = create_app(CONFIG)


if __name__ == "__main__":
    main()

