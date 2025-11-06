"""Shared helpers for Dash callback registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from .config import AppConfig
from .data_loader import load_stringing_daily as _load_stringing_daily


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


class DataSelector:
    """Mode-aware daily dataframe resolver used across callbacks."""

    def __init__(
        self,
        *,
        config: AppConfig,
        data_provider: Callable[[], pd.DataFrame],
        stringing_provider: Callable[[], pd.DataFrame] | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._data_provider = data_provider
        self._stringing_provider = stringing_provider
        self._logger = logger or logging.getLogger(__name__)

    def select(self, mode_value: str | None) -> pd.DataFrame:
        mode = (mode_value or "erection").strip().lower()
        if mode == "stringing":
            df = self._load_stringing()
        else:
            df = self._data_provider()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).copy()
            if "month" not in df.columns:
                df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        if "project_name" not in df.columns and "project" in df.columns:
            df["project_name"] = df["project"].astype(str).str.strip()
        if mode == "stringing" and "daily_km" in df.columns and "daily_prod_mt" not in df.columns:
            df["daily_prod_mt"] = df["daily_km"]
        return df

    def _load_stringing(self) -> pd.DataFrame:
        if callable(self._stringing_provider):
            try:
                return self._stringing_provider()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Stringing provider failed; falling back to loader: %s", exc)
        try:
            return _load_stringing_daily(self._config)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Stringing loader failed: %s", exc)
            return pd.DataFrame()


__all__ = ["ResponsibilitiesAccessor", "ResponsibilitiesPayload", "DataSelector"]

