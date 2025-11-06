"""Shared helpers for Dash callback registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Iterable, Sequence

import duckdb
import pandas as pd

from .config import AppConfig
from .data_loader import load_stringing_daily as _load_stringing_daily
from .filters import apply_filters
from .state import DUCKDB_TABLE_ERECTION, DUCKDB_TABLE_STRINGING


@dataclass(frozen=True)
class ResponsibilitiesPayload:
    """Bundle of Micro Plan responsibilities data."""

    frame: pd.DataFrame | None
    completion_keys: set[tuple[str, str]]
    error: str | None = None

    @property
    def has_frame(self) -> bool:
        return isinstance(self.frame, pd.DataFrame) and not self.frame.empty


class ResponsibilitiesAccessor:
    """Centralized, defensive accessor around disparate provider callables."""

    def __init__(
        self,
        *,
        data_provider: Callable[[], pd.DataFrame] | None,
        completion_provider: Callable[[], Iterable[tuple[str, str]]] | None,
        error_provider: Callable[[], str | None] | None,
        logger: logging.Logger,
    ) -> None:
        self._data_provider = data_provider
        self._completion_provider = completion_provider
        self._error_provider = error_provider
        self._logger = logger

    def load(self) -> ResponsibilitiesPayload:
        frame: pd.DataFrame | None = None
        completion_keys: set[tuple[str, str]] = set()
        error_message: str | None = None

        if callable(self._data_provider):
            try:
                frame = self._data_provider()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Unable to access responsibilities data: %s", exc)
                frame = None
                error_message = str(exc)
        else:
            frame = None

        if callable(self._completion_provider):
            try:
                completion_keys = set(self._completion_provider())
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Unable to access responsibilities completion keys: %s", exc)

        if callable(self._error_provider):
            try:
                error_message = self._error_provider()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Unable to access responsibilities error message: %s", exc)

        return ResponsibilitiesPayload(frame=frame, completion_keys=completion_keys, error=error_message)


@dataclass(slots=True)
class _FrameCacheEntry:
    """Single-mode cache entry storing a normalized dataframe and its source signature."""

    signature: tuple[object, ...]
    frame: pd.DataFrame

    def view(self) -> pd.DataFrame:
        """Return a lightweight view of the cached dataframe."""
        cat_cols = self.frame.select_dtypes(include="category").columns.tolist()
        if cat_cols:
            safe = self.frame.copy()
            for col in cat_cols:
                safe[col] = safe[col].astype(str)
            self.frame = safe
        return self.frame.copy(deep=False)


class DataSelector:
    """Mode-aware daily dataframe resolver used across callbacks."""

    _DEFAULT_MODE = "erection"

    def __init__(
        self,
        *,
        config: AppConfig,
        data_provider: Callable[[], pd.DataFrame],
        stringing_provider: Callable[[], pd.DataFrame] | None,
        duckdb_connection: duckdb.DuckDBPyConnection | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._data_provider = data_provider
        self._stringing_provider = stringing_provider
        self._duckdb_connection = duckdb_connection
        self._logger = logger or logging.getLogger(__name__)
        self._lock = RLock()
        self._cache: dict[str, _FrameCacheEntry] = {}
        self._stringing_fallback: pd.DataFrame | None = None
        self._duckdb_tables = {
            "erection": DUCKDB_TABLE_ERECTION,
            "stringing": DUCKDB_TABLE_STRINGING,
        }

    def select(self, mode_value: str | None) -> pd.DataFrame:
        mode = (mode_value or self._DEFAULT_MODE).strip().lower()
        if not mode:
            mode = self._DEFAULT_MODE
        try:
            source_df = self._resolve_source(mode)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Data provider for mode '%s' failed: %s", mode, exc)
            self._invalidate_cache(mode)
            return pd.DataFrame()
        if not isinstance(source_df, pd.DataFrame) or source_df.empty:
            self._invalidate_cache(mode)
            return pd.DataFrame()
        entry = self._get_or_build_cache(mode, source_df)
        if entry is None:
            return pd.DataFrame()
        return entry.view()

    def scopes_for(
        self,
        mode_value: str,
        *,
        months: Sequence[pd.Timestamp],
        projects: Sequence[str],
        gangs: Sequence[str],
        kv_filter: set[str] | None = None,
        method_filter: set[str] | None = None,
    ) -> dict[str, pd.DataFrame] | None:
        mode = (mode_value or self._DEFAULT_MODE).strip().lower() or self._DEFAULT_MODE
        months_list = list(months or [])
        project_list = list(projects or [])
        gang_list = list(gangs or [])
        kv_set = set(kv_filter or set())
        method_set = {m.strip().lower() for m in (method_filter or set()) if m}

        if mode != "stringing":
            kv_set.clear()
            method_set.clear()

        conn = self._duckdb_connection
        if conn is None:
            return self._scopes_via_pandas(
                mode,
                months_list,
                project_list,
                gang_list,
                kv_set,
                method_set,
            )

        try:
            scopes = self._scopes_via_duckdb(
                mode,
                months_list,
                project_list,
                gang_list,
                kv_set,
                method_set,
                connection=conn,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("DuckDB scope query failed for mode '%s': %s", mode, exc)
            scopes = None

        if scopes is None:
            return self._scopes_via_pandas(
                mode,
                months_list,
                project_list,
                gang_list,
                kv_set,
                method_set,
            )
        return scopes

    def _resolve_source(self, mode: str) -> pd.DataFrame:
        if mode == "stringing":
            return self._load_stringing()
        return self._data_provider()

    def _get_or_build_cache(self, mode: str, source_df: pd.DataFrame) -> _FrameCacheEntry | None:
        signature = self._frame_signature(source_df)
        with self._lock:
            cached = self._cache.get(mode)
            if cached and cached.signature == signature:
                return cached
        normalized = self._normalize_dataframe(source_df, mode)
        entry = _FrameCacheEntry(signature=signature, frame=normalized)
        with self._lock:
            self._cache[mode] = entry
        return entry

    def _invalidate_cache(self, mode: str) -> None:
        with self._lock:
            self._cache.pop(mode, None)

    def _load_stringing(self) -> pd.DataFrame:
        if callable(self._stringing_provider):
            try:
                return self._stringing_provider()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Stringing provider failed; falling back to loader: %s", exc)
        with self._lock:
            fallback = self._stringing_fallback
        if isinstance(fallback, pd.DataFrame):
            return fallback
        try:
            fallback = _load_stringing_daily(self._config)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Stringing loader failed: %s", exc)
            return pd.DataFrame()
        with self._lock:
            self._stringing_fallback = fallback
        return fallback

    def _scopes_via_pandas(
        self,
        mode: str,
        months: Sequence[pd.Timestamp],
        projects: Sequence[str],
        gangs: Sequence[str],
        kv_filter: set[str],
        method_filter: set[str],
    ) -> dict[str, pd.DataFrame]:
        df_day = self.select(mode)
        if not isinstance(df_day, pd.DataFrame) or df_day.empty:
            empty = pd.DataFrame()
            return {"month": empty, "project": empty, "full": empty, "project_gang": empty}

        working = df_day
        if mode == "stringing":
            working = self._apply_stringing_filters(working, kv_filter, method_filter)

        return {
            "month": apply_filters(working, [], months, []),
            "project": apply_filters(working, projects, months, []),
            "full": apply_filters(working, projects, months, gangs),
            "project_gang": apply_filters(working, projects, [], gangs),
        }

    def _scopes_via_duckdb(
        self,
        mode: str,
        months: Sequence[pd.Timestamp],
        projects: Sequence[str],
        gangs: Sequence[str],
        kv_filter: set[str],
        method_filter: set[str],
        *,
        connection: duckdb.DuckDBPyConnection,
    ) -> dict[str, pd.DataFrame] | None:
        table_name = self._duckdb_tables.get(mode)
        if not table_name:
            return None

        kv_norm = self._normalize_kv_filter(kv_filter)
        method_norm = method_filter

        month_scope = self._query_duckdb_scope(
            table_name,
            connection,
            mode=mode,
            months=months,
            projects=[],
            gangs=[],
            kv_filter=kv_norm,
            method_filter=method_norm,
        )
        project_scope = self._query_duckdb_scope(
            table_name,
            connection,
            mode=mode,
            months=months,
            projects=projects,
            gangs=[],
            kv_filter=kv_norm,
            method_filter=method_norm,
        )
        full_scope = self._query_duckdb_scope(
            table_name,
            connection,
            mode=mode,
            months=months,
            projects=projects,
            gangs=gangs,
            kv_filter=kv_norm,
            method_filter=method_norm,
        )
        project_gang_scope = self._query_duckdb_scope(
            table_name,
            connection,
            mode=mode,
            months=[],
            projects=projects,
            gangs=gangs,
            kv_filter=kv_norm,
            method_filter=method_norm,
        )

        if month_scope is None:
            return None

        def _ensure(frame: pd.DataFrame | None) -> pd.DataFrame:
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()

        return {
            "month": _ensure(month_scope),
            "project": _ensure(project_scope),
            "full": _ensure(full_scope),
            "project_gang": _ensure(project_gang_scope),
        }

    def _query_duckdb_scope(
        self,
        table_name: str,
        connection: duckdb.DuckDBPyConnection,
        *,
        mode: str,
        months: Sequence[pd.Timestamp],
        projects: Sequence[str],
        gangs: Sequence[str],
        kv_filter: set[str],
        method_filter: set[str],
    ) -> pd.DataFrame | None:
        columns = self._duckdb_columns(connection, table_name)
        where_clauses: list[str] = []
        params: list[object] = []

        if months and "month" in columns:
            where_clauses.append("month IN (SELECT * FROM UNNEST(?))")
            params.append([self._timestamp_to_py(ts) for ts in months])
        if projects and "project_name" in columns:
            where_clauses.append("project_name IN (SELECT * FROM UNNEST(?))")
            params.append(list(projects))
        if gangs and "gang_name" in columns:
            where_clauses.append("gang_name IN (SELECT * FROM UNNEST(?))")
            params.append(list(gangs))

        leftover_kv: set[str] = set()
        leftover_method: set[str] = set()

        if mode == "stringing":
            if kv_filter and kv_filter != {"400", "765"}:
                if "line_kv" in columns:
                    where_clauses.append("COALESCE(line_kv, '') IN (SELECT * FROM UNNEST(?))")
                    params.append(sorted(kv_filter))
                else:
                    leftover_kv = set(kv_filter)
            if method_filter and method_filter != {"manual", "tse"}:
                if "method_norm" in columns:
                    where_clauses.append("COALESCE(method_norm, '') IN (SELECT * FROM UNNEST(?))")
                    params.append(sorted(method_filter))
                elif "method" in columns:
                    where_clauses.append("lower(COALESCE(method, '')) IN (SELECT * FROM UNNEST(?))")
                    params.append(sorted(method_filter))
                else:
                    leftover_method = set(method_filter)

        query = f"SELECT * FROM {table_name}"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        result = connection.execute(query, params).df()

        if mode == "stringing" and (leftover_kv or leftover_method):
            result = self._apply_stringing_filters(result, leftover_kv, leftover_method)

        return result

    def _duckdb_columns(self, connection: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
        info = connection.execute(f"PRAGMA table_info('{table_name}')").df()
        return set(info["name"].astype(str)) if not info.empty else set()

    def _timestamp_to_py(self, value: pd.Timestamp) -> object:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        return pd.Timestamp(value).to_pydatetime()

    def _normalize_kv_filter(self, values: set[str]) -> set[str]:
        normalized: set[str] = set()
        for raw in values:
            if not raw:
                continue
            text = str(raw).strip().lower()
            digits = "".join(ch for ch in text if ch.isdigit())
            normalized.add(digits or text.upper())
        return {val for val in normalized if val}

    def _apply_stringing_filters(
        self,
        frame: pd.DataFrame,
        kv_filter: set[str],
        method_filter: set[str],
    ) -> pd.DataFrame:
        working = frame
        if working.empty:
            return working
        result = working

        if kv_filter and kv_filter != {"400", "765"}:
            if "line_kv" in result.columns:
                mask = result["line_kv"].astype(str).str.lower().isin({v.lower() for v in kv_filter})
                result = result.loc[mask]
        if method_filter and method_filter != {"manual", "tse"}:
            if "method_norm" in result.columns:
                mask = result["method_norm"].astype(str).str.lower().isin(method_filter)
                result = result.loc[mask]
            elif "method" in result.columns:
                mask = result["method"].astype(str).str.lower().isin(method_filter)
                result = result.loc[mask]
        return result

    def _normalize_dataframe(self, raw_df: pd.DataFrame, mode: str) -> pd.DataFrame:
        working = raw_df
        owns_data = False

        def ensure_materialized(*, deep: bool = False) -> pd.DataFrame:
            nonlocal working, owns_data
            if not owns_data:
                working = working.copy(deep=deep)
                owns_data = True
            return working

        if "date" in working.columns:
            date_series = working["date"]
            if not pd.api.types.is_datetime64_any_dtype(date_series):
                converted = pd.to_datetime(date_series, errors="coerce")
                if not converted.equals(date_series):
                    frame = ensure_materialized()
                    frame["date"] = converted
                    working = frame
                    date_series = frame["date"]
                else:
                    date_series = converted
            mask = date_series.notna()
            if not mask.all():
                # materialize before filtering to avoid chained assignment issues
                frame = ensure_materialized(deep=False)
                working = frame.loc[mask].copy()
                owns_data = True
                date_series = working["date"]
            if "month" not in working.columns:
                frame = ensure_materialized()
                frame["month"] = date_series.dt.to_period("M").dt.to_timestamp()
                working = frame
        elif "month" in working.columns and not pd.api.types.is_datetime64_any_dtype(working["month"]):
            month_series = pd.to_datetime(working["month"], errors="coerce")
            frame = ensure_materialized()
            frame["month"] = month_series
            working = frame

        if "project_name" not in working.columns and "project" in working.columns:
            frame = ensure_materialized()
            frame["project_name"] = frame["project"].astype(str).str.strip()
            working = frame

        if mode == "stringing" and "daily_km" in working.columns and "daily_prod_mt" not in working.columns:
            frame = ensure_materialized()
            frame["daily_prod_mt"] = frame["daily_km"]
            working = frame

        categorical_cols = working.select_dtypes(include="category").columns.tolist()
        if categorical_cols:
            frame = ensure_materialized()
            for col in categorical_cols:
                frame[col] = frame[col].astype(str)
            working = frame

        # Ensure the cached frame is detached from the provider dataframe reference.
        if owns_data:
            return working
        return working.copy(deep=False)

    def _frame_signature(self, df: pd.DataFrame) -> tuple[object, ...]:
        version = None
        mode = None
        attrs = getattr(df, "attrs", None)
        if isinstance(attrs, dict):
            version = attrs.get("_appdata_version")
            mode = attrs.get("_appdata_mode")
        column_hash = hash(tuple(df.columns))
        row_count = len(df.index)
        if version is not None:
            return ("version", mode, version, row_count, column_hash)
        return ("object", id(df), row_count, column_hash)


__all__ = ["ResponsibilitiesAccessor", "ResponsibilitiesPayload", "DataSelector"]
