"""Centralized runtime data store for the Dash server."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from threading import RLock
from typing import Mapping, Tuple

import duckdb
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for Py<3.9 (not expected)
    ZoneInfo = None  # type: ignore[misc,assignment]

from .config import AppConfig
from .data_loader import (
    load_daily as _load_daily,
    load_project_details,
    load_stringing_daily as _load_stringing_daily,
    load_stringing_compiled_raw as _load_stringing_compiled_raw,
)
from .plan_utils import (
    compact_project_key,
    compute_stringing_completion_pairs,
    normalize_lower,
)
from .services.responsibilities import (
    ResponsibilitiesSnapshot,
    load_responsibilities_snapshot,
)
from .stringing import build_tse_lookup_from_df

LOGGER = logging.getLogger(__name__)

DUCKDB_TABLE_ERECTION = "appdata_erection_daily"
DUCKDB_TABLE_STRINGING = "appdata_stringing_daily"


@dataclass
class DatasetMetadata:
    """Human-readable metadata for the currently loaded dataset."""

    display_timezone: str | None = None
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

        tzinfo = None
        if self.display_timezone:
            if ZoneInfo is None:
                LOGGER.warning(
                    "zoneinfo unavailable; cannot apply display timezone '%s'.", self.display_timezone
                )
            else:
                try:
                    tzinfo = ZoneInfo(self.display_timezone)
                except Exception:  # pragma: no cover - invalid timezone string
                    LOGGER.warning(
                        "Invalid DISPLAY_TIMEZONE '%s'; using server local time.",
                        self.display_timezone,
                    )
                    tzinfo = None

        try:
            now_display = pd.Timestamp.now(tz=tzinfo) if tzinfo is not None else pd.Timestamp.now()
            if tzinfo is not None:
                now_display = now_display.tz_localize(None)
            self.last_loaded_text = now_display.strftime("%d-%m-%Y")
        except Exception:
            self.last_loaded_text = "N/A"


class AppDataStore:
    """Application-wide mutable state guarded by a re-entrant lock."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._lock = RLock()
        self._daily: pd.DataFrame | None = None
        self._stringing_daily: pd.DataFrame | None = None
        self._stringing_compiled: pd.DataFrame | None = None
        self._project_info: pd.DataFrame | None = None
        self._responsibilities: ResponsibilitiesSnapshot = ResponsibilitiesSnapshot(None, set(), None)
        self._stringing_responsibilities: ResponsibilitiesSnapshot = ResponsibilitiesSnapshot(None, set(), None)
        self._stringing_completion_keys: set[tuple[str, str]] = set()
        self.metadata = DatasetMetadata(display_timezone=config.display_timezone)
        self._daily_version = 0
        self._stringing_version = 0
        self._duckdb_conn = self._create_duckdb_connection()
        self._duckdb_lock = RLock()
        self._stringing_tse_canonical: dict[str, int] = {}
        self._stringing_tse_alias: dict[str, str] = {}

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
            stringing_cfg = replace(cfg, data_path=cfg.stringing_data_path)
            self._stringing_responsibilities = load_responsibilities_snapshot(stringing_cfg)

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
            self._daily_version += 1
            working = df.copy()
            working.attrs["_appdata_mode"] = "erection"
            working.attrs["_appdata_version"] = self._daily_version
            self._daily = working
            self.metadata.update_from_df(self._daily)
            self._register_duckdb_table(DUCKDB_TABLE_ERECTION, self._daily)

    def get_daily(self) -> pd.DataFrame:
        with self._lock:
            if self._daily is None:
                raise RuntimeError("Daily dataframe not loaded.")
            return self._daily

    def set_stringing(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stringing_version += 1
            working = df.copy()
            working.attrs["_appdata_mode"] = "stringing"
            working.attrs["_appdata_version"] = self._stringing_version
            self._augment_stringing_frame(working)
            self._stringing_daily = working
            self._stringing_compiled = None
            self._stringing_completion_keys = compute_stringing_completion_pairs(working)
            self._register_duckdb_table(DUCKDB_TABLE_STRINGING, self._stringing_daily)

    def get_stringing(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stringing_daily.copy()
                if isinstance(self._stringing_daily, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_stringing_compiled(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stringing_compiled = df.copy()

    def get_stringing_compiled(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stringing_compiled.copy()
                if isinstance(self._stringing_compiled, pd.DataFrame)
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

    def get_stringing_responsibilities_frame(self) -> pd.DataFrame:
        with self._lock:
            return self._stringing_responsibilities.require_frame()

    def get_stringing_responsibilities_completion_keys(self) -> set[tuple[str, str]]:
        with self._lock:
            completion = set(self._stringing_responsibilities.completion_keys)
            completion.update(self._stringing_completion_keys)
            return completion

    def get_stringing_responsibilities_error(self) -> str | None:
        with self._lock:
            return self._stringing_responsibilities.error

    def get_stringing_tse_lookup(self) -> tuple[dict[str, int], dict[str, str]]:
        with self._lock:
            return (
                dict(self._stringing_tse_canonical),
                dict(self._stringing_tse_alias),
            )

    def get_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return self._duckdb_conn

    def get_duckdb_lock(self) -> RLock:
        """Return the re-entrant lock guarding DuckDB access."""

        return self._duckdb_lock

    def _maybe_preload_stringing(self, config: AppConfig) -> None:
        if not config.enable_stringing:
            return
        compiled_df: pd.DataFrame | None = None
        try:
            compiled_df = _load_stringing_compiled_raw(config)
        except Exception as exc:
            LOGGER.warning("Stringing compiled preload failed: %s", exc)
            compiled_df = None
        canonical_map: dict[str, int] = {}
        alias_map: dict[str, str] = {}
        if isinstance(compiled_df, pd.DataFrame):
            canonical_map, alias_map = build_tse_lookup_from_df(compiled_df)
        self._update_stringing_tse_lookup(canonical_map, alias_map)
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

        if isinstance(compiled_df, pd.DataFrame):
            self.set_stringing_compiled(compiled_df)
            LOGGER.info("Preloaded stringing compiled rows: %d", len(compiled_df))

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

    def _create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(database=":memory:", read_only=False)
        conn.execute("PRAGMA enable_object_cache")
        return conn

    def _register_duckdb_table(self, table_name: str, frame: pd.DataFrame | None) -> None:
        conn = self._duckdb_conn
        if conn is None:
            return
        with self._duckdb_lock:
            if frame is None or frame.empty:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                return
            temp_name = f"__df_{table_name}"
            conn.register(temp_name, frame)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {temp_name}")
            conn.unregister(temp_name)
            self._create_duckdb_indexes(table_name, frame.columns)

    def _create_duckdb_indexes(self, table_name: str, columns: pd.Index) -> None:
        conn = self._duckdb_conn
        if conn is None:
            return
        candidates = [col for col in ("month", "project_name", "gang_name") if col in columns]
        for col in candidates:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name}({col})")

    def _augment_stringing_frame(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            frame["line_kv"] = pd.Series(dtype="string")
            frame["method_norm"] = pd.Series(dtype="string")
            frame["deployment_tse_flag"] = pd.Series(dtype=bool)
            return
        source = None
        if "project_name" in frame.columns:
            source = frame["project_name"].astype(str)
        elif "project" in frame.columns:
            source = frame["project"].astype(str)
        else:
            source = pd.Series("", index=frame.index)
        norm = source.str.lower()
        line_kv = np.where(
            norm.str.contains("765", na=False),
            "765",
            np.where(norm.str.contains("400", na=False), "400", pd.NA),
        )
        frame["line_kv"] = pd.Series(line_kv, index=frame.index).astype("string")
        if "method" in frame.columns:
            method_norm = frame["method"].astype(str).str.strip().str.lower()
            frame["method_norm"] = method_norm.mask(method_norm.isin({"", "nan", "none"})).astype("string")
        else:
            frame["method_norm"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["deployment_tse_flag"] = self._compute_tse_mask(frame)

    def _compute_tse_mask(self, frame: pd.DataFrame) -> pd.Series:
        canonical = self._stringing_tse_canonical
        alias_map = self._stringing_tse_alias
        if not canonical and not alias_map:
            mask = pd.Series(False, index=frame.index, dtype=bool)
        else:
            norm_keys = set((canonical or {}).keys())
            alias_keys = set((alias_map or {}).keys())
            mask = pd.Series(False, index=frame.index, dtype=bool)
            candidate_columns = (
                "project_name",
                "project",
                "project_name_display",
                "Project Name",
                "project_code",
                "project_key",
            )

            def _evaluate(series: pd.Series) -> pd.Series:
                base = series.astype(str).str.strip()
                result = pd.Series(False, index=series.index, dtype=bool)
                if norm_keys:
                    result = result | base.map(normalize_lower).isin(norm_keys)
                if alias_keys:
                    result = result | base.map(compact_project_key).isin(alias_keys)
                if base.str.contains(" : ").any():
                    parts = base.str.split(" : ", n=1, expand=True)
                    for idx in range(min(2, parts.shape[1])):
                        split_series = parts[idx].str.strip()
                        if norm_keys:
                            result = result | split_series.map(normalize_lower).isin(norm_keys)
                        if alias_keys:
                            result = result | split_series.map(compact_project_key).isin(alias_keys)
                return result

            for column in candidate_columns:
                if column not in frame.columns:
                    continue
                mask = mask | _evaluate(frame[column])

        method_series = None
        if "method_norm" in frame.columns:
            method_series = frame["method_norm"].astype(str).str.strip().str.lower()
        elif "method" in frame.columns:
            method_series = frame["method"].astype(str).str.strip().str.lower()
        if method_series is not None:
            mask = mask | method_series.eq("tse")

        return mask.fillna(False)

    def _update_stringing_tse_lookup(
        self,
        canonical: Mapping[str, int] | None,
        aliases: Mapping[str, str] | None,
    ) -> None:
        with self._lock:
            self._stringing_tse_canonical = dict(canonical or {})
            self._stringing_tse_alias = dict(aliases or {})


__all__ = ["AppDataStore", "DatasetMetadata", "DUCKDB_TABLE_ERECTION", "DUCKDB_TABLE_STRINGING"]
