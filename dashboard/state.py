"""Centralized runtime data store for the Dash server."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Tuple

import pandas as pd

from .config import AppConfig
from .data_loader import (
    load_daily as _load_daily,
    load_project_details,
    load_stringing_daily as _load_stringing_daily,
)
from .services.responsibilities import (
    ResponsibilitiesSnapshot,
    load_responsibilities_snapshot,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Human-readable metadata for the currently loaded dataset."""

    last_data_date: pd.Timestamp | None = None
    last_data_date_text: str = "N/A"
    last_loaded_text: str = "N/A"

    def update_from_df(self, df: pd.DataFrame) -> None:
        date_col = pd.to_datetime(df.get("date"), errors="coerce") if "date" in df.columns else None
        last_date = date_col.max() if date_col is not None else None
        self.last_data_date = last_date if pd.notna(last_date) else None
        self.last_data_date_text = (
            self.last_data_date.strftime("%d-%m-%Y") if self.last_data_date is not None else "N/A"
        )
        try:
            self.last_loaded_text = pd.Timestamp.now().strftime("%d-%m-%Y")
        except Exception:
            self.last_loaded_text = "N/A"


class AppDataStore:
    """Application-wide mutable state guarded by a re-entrant lock."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._lock = RLock()
        self._daily: pd.DataFrame | None = None
        self._stringing_daily: pd.DataFrame | None = None
        self._project_info: pd.DataFrame | None = None
        self._responsibilities: ResponsibilitiesSnapshot = ResponsibilitiesSnapshot(None, set(), None)
        self.metadata = DatasetMetadata()

    def bootstrap(self, config: AppConfig | None = None) -> Tuple[pd.DataFrame, str]:
        """Hydrate the cache from disk and return (daily_df, last_loaded_text)."""

        cfg = config or self._config
        daily = _load_daily(cfg)
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"]).copy()
        daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

        project_info = load_project_details(cfg.data_path)
        daily = self._attach_project_codes(daily, project_info)

        self.set_daily(daily)
        with self._lock:
            self._project_info = project_info.copy()
            self._responsibilities = load_responsibilities_snapshot(cfg)

        self._maybe_preload_stringing(cfg)

        LOGGER.info(
            "Loaded workbook: %s | rows=%s | cols=%s | project_info_rows=%s",
            cfg.data_path,
            len(daily),
            list(daily.columns),
            0 if project_info is None else len(project_info),
        )
        return daily, self.metadata.last_loaded_text

    def set_daily(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._daily = df.copy()
            self.metadata.update_from_df(self._daily)

    def get_daily(self) -> pd.DataFrame:
        with self._lock:
            if self._daily is None:
                raise RuntimeError("Daily dataframe not loaded.")
            return self._daily

    def set_stringing(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stringing_daily = df.copy()

    def get_stringing(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stringing_daily.copy()
                if isinstance(self._stringing_daily, pd.DataFrame)
                else pd.DataFrame()
            )

    def get_project_info(self) -> pd.DataFrame:
        with self._lock:
            return self._project_info.copy() if isinstance(self._project_info, pd.DataFrame) else pd.DataFrame()

    def get_responsibilities_frame(self) -> pd.DataFrame:
        with self._lock:
            return self._responsibilities.require_frame()

    def get_responsibilities_completion_keys(self) -> set[tuple[str, str]]:
        with self._lock:
            return set(self._responsibilities.completion_keys)

    def get_responsibilities_error(self) -> str | None:
        with self._lock:
            return self._responsibilities.error

    def _maybe_preload_stringing(self, config: AppConfig) -> None:
        if not config.enable_stringing:
            return
        try:
            stringing_df = _load_stringing_daily(config)
        except Exception as exc:
            LOGGER.warning("Stringing preload failed: %s", exc)
            return
        if stringing_df.empty:
            self.set_stringing(stringing_df)
            LOGGER.info("Preloaded stringing daily rows: 0")
            return

        stringing_df = stringing_df.copy()
        if "date" in stringing_df.columns:
            stringing_df["date"] = pd.to_datetime(stringing_df["date"], errors="coerce")
            stringing_df = stringing_df.dropna(subset=["date"]).copy()
            if "month" not in stringing_df.columns:
                stringing_df["month"] = stringing_df["date"].dt.to_period("M").dt.to_timestamp()
        self.set_stringing(stringing_df)
        LOGGER.info("Preloaded stringing daily rows: %d", len(stringing_df))

    def _attach_project_codes(self, df: pd.DataFrame, project_info: pd.DataFrame) -> pd.DataFrame:
        if project_info is None or project_info.empty or "project_name" not in df.columns:
            return df

        required = {"key_name", "project_code"}
        if not required.issubset(set(project_info.columns)):
            return df

        working = df.copy()
        working["__key_name__"] = (
            working["project_name"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
        )
        map_df = (
            project_info[["key_name", "project_code"]]
            .dropna()
            .drop_duplicates("key_name")
        )
        enriched = working.merge(map_df, left_on="__key_name__", right_on="key_name", how="left")
        enriched = enriched.drop(columns=["__key_name__", "key_name"])
        return enriched


__all__ = ["AppDataStore", "DatasetMetadata"]
