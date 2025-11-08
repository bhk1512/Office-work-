"""Loading and normalization helpers for Micro Plan responsibilities data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import AppConfig
from ..data_loader import find_parquet_source, is_parquet_dataset, read_parquet_table


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponsibilitiesSnapshot:
    """Immutable payload containing responsibilities data and metadata."""

    frame: pd.DataFrame | None
    completion_keys: set[tuple[str, str]]
    error: str | None = None

    @classmethod
    def empty(cls, message: str | None = None) -> "ResponsibilitiesSnapshot":
        return cls(pd.DataFrame(), set(), message)

    def require_frame(self) -> pd.DataFrame:
        """Return the underlying dataframe or raise if unavailable."""

        if self.frame is None:
            raise RuntimeError(self.error or "Responsibilities data not loaded.")
        return self.frame.copy()

    def copy(self) -> "ResponsibilitiesSnapshot":
        """Return a shallow copy to avoid accidental shared mutation."""

        frame_copy = None if self.frame is None else self.frame.copy()
        return ResponsibilitiesSnapshot(frame_copy, set(self.completion_keys), self.error)


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


def _pick_column(frame: pd.DataFrame, choices: Iterable[str]) -> str:
    mapping = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in choices:
        key = candidate.strip().lower()
        if key in mapping:
            return mapping[key]
    for key, original in mapping.items():
        lowered = key.lower()
        if any(choice.lower() in lowered for choice in choices):
            return original
    raise KeyError(tuple(choices))


def load_responsibilities_snapshot(config: AppConfig) -> ResponsibilitiesSnapshot:
    """Load Micro Plan responsibilities atomic sheet plus completion metadata."""

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
            return ResponsibilitiesSnapshot.empty("No Micro Plan data found in the compiled dataset.")

        try:
            df_atomic = read_parquet_table(resp_source)
        except FileNotFoundError:
            LOGGER.warning("Responsibilities parquet not found near: %s", path)
            return ResponsibilitiesSnapshot.empty("No Micro Plan data found in the compiled dataset.")
        except Exception as exc:
            LOGGER.exception("Failed to load responsibilities parquet: %s", exc)
            return ResponsibilitiesSnapshot(None, set(), "Unable to load Micro Plan data.")

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
            with pd.ExcelFile(path) as workbook:
                sheet_names = set(workbook.sheet_names)

                sheet_name = "MicroPlanResponsibilities"
                if sheet_name not in sheet_names:
                    LOGGER.warning("Sheet '%s' missing in workbook", sheet_name)
                    return ResponsibilitiesSnapshot.empty("No Micro Plan data found in the compiled workbook.")

                df_atomic = pd.read_excel(workbook, sheet_name=sheet_name)

                candidates = [config.preferred_sheet, "ProdDailyExpandedSingles", "ProdDailyExpanded"]
                daily_sheet = next((candidate for candidate in candidates if candidate and candidate in sheet_names), None)
                if daily_sheet:
                    try:
                        df_daily = pd.read_excel(workbook, sheet_name=daily_sheet, usecols=None)
                    except Exception as exc:
                        LOGGER.warning("Failed to load daily sheet '%s': %s", daily_sheet, exc)
        except FileNotFoundError:
            LOGGER.warning("Responsibilities workbook not found: %s", config.data_path)
            return ResponsibilitiesSnapshot(None, set(), "Compiled workbook not found.")
        except Exception as exc:
            LOGGER.exception("Failed to open responsibilities workbook: %s", exc)
            return ResponsibilitiesSnapshot(None, set(), "Unable to load Micro Plan data.")

    completion_keys: set[tuple[str, str]] = set()
    if df_daily is not None and not df_daily.empty:
        try:
            col_proj = _pick_column(df_daily, ("project_name", "project"))
            col_loc = _pick_column(df_daily, ("location_no", "location number", "location"))
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
        LOGGER.info("Daily expanded data not found; delivered values fall back to realised revenue only.")

    return ResponsibilitiesSnapshot(df_atomic, completion_keys, None)


__all__ = ["ResponsibilitiesSnapshot", "load_responsibilities_snapshot"]
