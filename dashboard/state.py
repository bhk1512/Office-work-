"""Centralized runtime data store for the Dash server."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from threading import RLock
from typing import Mapping, Sequence, Tuple

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
    load_stringing_coverage as _load_stringing_coverage,
    load_progress_status_raw as _load_progress_status_raw,
    load_progress_status_coverage as _load_progress_status_coverage,
    load_stretch_readiness_raw as _load_stretch_readiness_raw,
    load_stretch_readiness_summary as _load_stretch_readiness_summary,
    load_stretch_readiness_manpower_audit as _load_stretch_readiness_manpower_audit,
    load_stretch_readiness_coverage as _load_stretch_readiness_coverage,
    load_stringing_summary_table as _load_stringing_summary_table,
)
from .metrics import (
    calc_idle_and_loss,
    calc_idle_and_loss_for_column,
    compute_gang_baseline_maps,
    compute_idle_intervals_per_gang,
    compute_project_baseline_maps,
    compute_project_baseline_maps_for,
)
from .plan_utils import (
    compact_project_key,
    compute_stringing_completion_pairs,
    normalize_lower,
    normalize_location,
    normalize_text,
)
from .services.responsibilities import (
    ResponsibilitiesSnapshot,
    load_responsibilities_snapshot,
)
from .project_identity import build_project_display, build_project_scope_key, normalize_line_name
from .stringing import build_tse_lookup_from_df

import re as _re

LOGGER = logging.getLogger(__name__)


def _build_erection_scope_key_bridge(config_path: "Path | None") -> "dict[str, str]":
    """Build a mapping from original erection scope keys to kV-normalized ones.

    For split projects where the DPR file uses a non-kV line name (e.g. "MAIN") but
    the stringing-summary pipeline normalizes it to a kV value (e.g. "400kV"), the
    erection data carries the original scope key ("tb507main") while the rest of the
    dashboard uses the kV-normalized one ("tb507400kv").  This function reads
    Sheet Names Check in DPR_Config.xlsx and produces the mapping so erection rows
    can be found under the normalized key.
    """
    from pathlib import Path as _Path

    if config_path is None:
        return {}
    path = _Path(config_path)
    if not path.exists():
        return {}
    try:
        df = pd.read_excel(path, sheet_name="Sheet Names Check")
        df.columns = [str(c).strip().lower() for c in df.columns]
    except Exception:
        return {}

    bridge: dict[str, str] = {}
    for _, row in df.iterrows():
        proj_code = str(row.get("project code") or "").strip()
        rule = str(row.get("consolidation rule") or "").strip().lower()
        if not proj_code or rule != "split":
            continue
        raw_lines = str(row.get("status line names") or "").strip()
        raw_sheets = str(row.get("status sheet names") or "").strip()
        if not raw_lines or not raw_sheets:
            continue
        line_parts = [p.strip() for p in raw_lines.split(";") if p.strip()]
        sheet_parts = [p.strip() for p in raw_sheets.split(";") if p.strip()]
        raw_file_ids_raw = row.get("status file identifier")
        file_id_parts = (
            [p.strip() for p in str(raw_file_ids_raw).split(";")]
            if raw_file_ids_raw and str(raw_file_ids_raw).strip() not in ("", "nan", "None")
            else []
        )
        proj_compact = compact_project_key(proj_code)
        bare_scope_key = build_project_scope_key(proj_code, "")
        for i, (line_name, sheet_name) in enumerate(zip(line_parts, sheet_parts)):
            m = _re.search(r"\b(\d{2,4})\s*k\s*v\b", sheet_name, _re.IGNORECASE)
            if not m:
                continue
            kv_value = f"{m.group(1)}kV"
            kv_compact = _re.sub(r"[^a-z0-9]", "", kv_value.lower())
            normalized_key = build_project_scope_key(proj_code, kv_value)
            line_compact = compact_project_key(line_name)
            if line_compact != kv_compact:
                original_key = build_project_scope_key(proj_code, line_name)
                if original_key and normalized_key and original_key != normalized_key:
                    bridge[original_key] = normalized_key
            file_id = file_id_parts[i] if i < len(file_id_parts) else ""
            if not file_id.strip() and bare_scope_key and normalized_key and bare_scope_key != normalized_key:
                bridge.setdefault(bare_scope_key, normalized_key)
    return bridge

DUCKDB_TABLE_ERECTION = "appdata_erection_daily"
DUCKDB_TABLE_STRINGING = "appdata_stringing_daily"
STRINGING_SUMMARY_TABLES: tuple[str, ...] = (
    "StatusActivityFact",
    "StatusSnapshotProject",
    "StatusSnapshotOverall",
    "StretchSectionFact",
    "ManpowerProductivityFact",
    "Coverage",
    "Diagnostics",
    "Issues",
)


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
        self.metadata = DatasetMetadata(display_timezone=config.display_timezone)

        self._duckdb_conn = self._create_duckdb_connection()
        self._duckdb_lock = RLock()

        # Raw frames
        self._daily: pd.DataFrame | None = None
        self._stringing_daily: pd.DataFrame | None = None
        self._stringing_compiled: pd.DataFrame | None = None
        self._stringing_coverage: pd.DataFrame | None = None
        self._progress_status_raw: pd.DataFrame | None = None
        self._progress_status_coverage: pd.DataFrame | None = None
        self._stretch_readiness_raw: pd.DataFrame | None = None
        self._stretch_readiness_summary: pd.DataFrame | None = None
        self._stretch_readiness_manpower_audit: pd.DataFrame | None = None
        self._stretch_readiness_coverage: pd.DataFrame | None = None
        self._stringing_summary_tables: dict[str, pd.DataFrame] = {
            table: pd.DataFrame() for table in STRINGING_SUMMARY_TABLES
        }
        self._project_info: pd.DataFrame | None = None

        # Responsibilities
        self._responsibilities: ResponsibilitiesSnapshot = ResponsibilitiesSnapshot(None, set(), None)
        self._stringing_responsibilities: ResponsibilitiesSnapshot = ResponsibilitiesSnapshot(None, set(), None)
        self._responsibility_frame: pd.DataFrame = pd.DataFrame()
        self._stringing_responsibility_frame: pd.DataFrame = pd.DataFrame()
        self._responsibility_index: dict[str, dict[str, pd.DataFrame]] = {}
        self._stringing_responsibility_index: dict[str, dict[str, pd.DataFrame]] = {}
        self._responsibility_alias_map: dict[str, str] = {}
        self._stringing_responsibility_alias_map: dict[str, str] = {}
        self._responsibility_completion_lookup: set[tuple[str, str]] = set()
        self._stringing_completion_lookup: set[tuple[str, str]] = set()
        self._stringing_completion_keys: set[tuple[str, str]] = set()

        # Baselines and summaries
        self._erection_baseline_overall: dict[str, float] = {}
        self._erection_baseline_monthly: dict[str, dict[pd.Timestamp, float]] = {}
        self._stringing_baseline_overall: dict[str, float] = {}
        self._stringing_baseline_monthly: dict[str, dict[pd.Timestamp, float]] = {}
        self._erection_gang_summary: pd.DataFrame = pd.DataFrame()
        self._stringing_gang_summary: pd.DataFrame = pd.DataFrame()
        self._erection_idle_intervals: pd.DataFrame = pd.DataFrame()
        self._stringing_idle_intervals: pd.DataFrame = pd.DataFrame()

        # PCH + plan blocks
        self._pch_summary: pd.DataFrame = pd.DataFrame()
        self._pch_tiles: dict[str, pd.DataFrame] = {}
        self._stringing_scope_frames: dict[str, pd.DataFrame] = {}
        self._stringing_plan_summary: pd.DataFrame = pd.DataFrame()
        self._stringing_plan_planned: dict[str, dict[pd.Timestamp, float]] = {}
        self._stringing_plan_delivered: dict[str, dict[pd.Timestamp, float]] = {}

        # Traceability
        self._traceability_tables: dict[str, pd.DataFrame] = {"erection": pd.DataFrame(), "stringing": pd.DataFrame()}

        # Misc stringing helpers
        self._stringing_tse_canonical: dict[str, int] = {}
        self._stringing_tse_alias: dict[str, str] = {}

        self._daily_version = 0
        self._stringing_version = 0

    # ------------------------------------------------------------------
    # Bootstrap + loaders
    # ------------------------------------------------------------------
    def bootstrap(self, config: AppConfig | None = None) -> Tuple[pd.DataFrame, str]:
        """Hydrate the cache from disk and return (daily_df, last_loaded_text)."""

        cfg = config or self._config
        LOGGER.info("Bootstrapping AppDataStore (data_path=%s)", cfg.data_path)

        from pathlib import Path as _Path
        _data_root = _Path(cfg.data_path).resolve()
        _repo_root = _data_root.parent.parent if _data_root.name.lower() == "erection" else _data_root.parent
        _dpr_config_path = _repo_root / "Raw Data" / "DPR_Config.xlsx"
        _scope_key_bridge = _build_erection_scope_key_bridge(_dpr_config_path if _dpr_config_path.exists() else None)

        erection_daily = self._prepare_daily_frame(_load_daily(cfg), mode="erection", scope_key_bridge=_scope_key_bridge)
        project_info = load_project_details(cfg.data_path)
        erection_daily = self._attach_project_codes(erection_daily, project_info)

        self.set_daily(erection_daily)
        with self._lock:
            self._project_info = project_info.copy()
            self._responsibilities = load_responsibilities_snapshot(cfg)

        stringing_cfg = replace(cfg, data_path=cfg.stringing_data_path)
        with self._lock:
            self._stringing_responsibilities = load_responsibilities_snapshot(stringing_cfg)

        self._precompute_erection_structures(cfg)
        self._maybe_preload_stringing(cfg)
        self._maybe_preload_status_stretch_summary(cfg)
        self._precompute_stringing_structures(cfg)
        self._build_pch_tiles()
        self._build_traceability_tables(cfg)

        LOGGER.info(
            "Bootstrap complete: erection rows=%s | stringing rows=%s",
            0 if self._daily is None else len(self._daily),
            0 if self._stringing_daily is None else len(self._stringing_daily),
        )
        return self.get_daily(), self.metadata.last_loaded_text

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
            return self._daily.copy()

    def set_stringing(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stringing_version += 1
            working = df.copy()
            working.attrs["_appdata_mode"] = "stringing"
            working.attrs["_appdata_version"] = self._stringing_version
            self._augment_stringing_frame(working)
            self._stringing_daily = working
            self._stringing_compiled = None
            raw_pairs = compute_stringing_completion_pairs(working)
            normalized_pairs = {
                (compact_project_key(project), normalize_location(location))
                for project, location in raw_pairs
            }
            self._stringing_completion_keys = {
                (project, location)
                for project, location in normalized_pairs
                if project and location
            }
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

    def set_stringing_coverage(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stringing_coverage = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_stringing_coverage(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stringing_coverage.copy()
                if isinstance(self._stringing_coverage, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_progress_status_raw(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._progress_status_raw = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_progress_status_raw(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._progress_status_raw.copy()
                if isinstance(self._progress_status_raw, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_progress_status_coverage(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._progress_status_coverage = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_progress_status_coverage(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._progress_status_coverage.copy()
                if isinstance(self._progress_status_coverage, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_stretch_readiness_raw(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stretch_readiness_raw = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_stretch_readiness_raw(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stretch_readiness_raw.copy()
                if isinstance(self._stretch_readiness_raw, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_stretch_readiness_summary(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stretch_readiness_summary = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_stretch_readiness_summary(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stretch_readiness_summary.copy()
                if isinstance(self._stretch_readiness_summary, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_stretch_readiness_manpower_audit(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stretch_readiness_manpower_audit = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_stretch_readiness_manpower_audit(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stretch_readiness_manpower_audit.copy()
                if isinstance(self._stretch_readiness_manpower_audit, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_stretch_readiness_coverage(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._stretch_readiness_coverage = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_stretch_readiness_coverage(self) -> pd.DataFrame:
        with self._lock:
            return (
                self._stretch_readiness_coverage.copy()
                if isinstance(self._stretch_readiness_coverage, pd.DataFrame)
                else pd.DataFrame()
            )

    def set_stringing_summary_table(self, table_name: str, df: pd.DataFrame) -> None:
        key = str(table_name or "").strip()
        if not key:
            return
        with self._lock:
            self._stringing_summary_tables[key] = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def get_stringing_summary_table(self, table_name: str) -> pd.DataFrame:
        key = str(table_name or "").strip()
        if not key:
            return pd.DataFrame()
        with self._lock:
            frame = self._stringing_summary_tables.get(key)
            return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    # ------------------------------------------------------------------
    # Responsibilities (erection + stringing)
    # ------------------------------------------------------------------
    def get_responsibilities_frame(self) -> pd.DataFrame:
        with self._lock:
            return self._responsibility_frame.copy()

    def get_responsibilities_completion_keys(self) -> set[tuple[str, str]]:
        return self.get_responsibility_completion_lookup("erection")

    def get_responsibilities_error(self) -> str | None:
        with self._lock:
            return self._responsibilities.error

    def get_stringing_responsibilities_frame(self) -> pd.DataFrame:
        with self._lock:
            return self._stringing_responsibility_frame.copy()

    def get_stringing_responsibilities_completion_keys(self) -> set[tuple[str, str]]:
        return self.get_responsibility_completion_lookup("stringing")

    def get_stringing_responsibilities_error(self) -> str | None:
        with self._lock:
            return self._stringing_responsibilities.error

    def get_responsibility_completion_lookup(self, mode: str = "erection") -> set[tuple[str, str]]:
        with self._lock:
            if mode == "stringing":
                lookup = set(self._stringing_completion_lookup)
                lookup.update(self._stringing_completion_keys)
                return lookup
            return set(self._responsibility_completion_lookup)

    def get_responsibility_index(self, mode: str = "erection") -> dict[str, dict[str, pd.DataFrame]]:
        with self._lock:
            if mode == "stringing":
                return {k: {ek: df.copy() for ek, df in v.items()} for k, v in self._stringing_responsibility_index.items()}
            return {k: {ek: df.copy() for ek, df in v.items()} for k, v in self._responsibility_index.items()}

    def get_responsibilities_slice(
        self,
        *,
        mode: str,
        project_candidates: Sequence[str],
        entity_type: str | None = None,
    ) -> pd.DataFrame:
        with self._lock:
            if mode == "stringing":
                index = self._stringing_responsibility_index
                alias_map = self._stringing_responsibility_alias_map
            else:
                index = self._responsibility_index
                alias_map = self._responsibility_alias_map
        keys = self._resolve_responsibility_project_candidates(project_candidates, alias_map)
        if not keys:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        entity_norm = normalize_lower(entity_type or "")
        for key in keys:
            project_map = index.get(key)
            if not project_map:
                continue
            if entity_norm:
                entity_df = project_map.get(entity_norm)
                if isinstance(entity_df, pd.DataFrame):
                    frames.append(entity_df.copy())
            else:
                frames.extend(df.copy() for df in project_map.values())
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def is_responsibility_location_complete(
        self,
        project_value: str,
        location_value: str,
        *,
        mode: str = "erection",
    ) -> bool:
        with self._lock:
            alias_map = self._stringing_responsibility_alias_map if mode == "stringing" else self._responsibility_alias_map
        canonical = self._resolve_responsibility_project_key(project_value, alias_map)
        location_norm = normalize_location(location_value)
        if not canonical or not location_norm:
            return False
        lookup = self.get_responsibility_completion_lookup(mode)
        return (canonical, location_norm) in lookup

    # ------------------------------------------------------------------
    # Baselines, summaries, PCH tiles, scopes, traceability
    # ------------------------------------------------------------------
    def get_project_baselines(
        self, mode: str = "erection"
    ) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
        with self._lock:
            if mode == "stringing":
                return dict(self._stringing_baseline_overall), {
                    project: dict(month_map)
                    for project, month_map in self._stringing_baseline_monthly.items()
                }
            return dict(self._erection_baseline_overall), {
                project: dict(month_map)
                for project, month_map in self._erection_baseline_monthly.items()
            }

    def get_gang_summary(self, mode: str = "erection") -> pd.DataFrame:
        with self._lock:
            if mode == "stringing":
                return self._stringing_gang_summary.copy()
            return self._erection_gang_summary.copy()

    def get_idle_intervals(self, mode: str = "erection") -> pd.DataFrame:
        with self._lock:
            table = self._stringing_idle_intervals if mode == "stringing" else self._erection_idle_intervals
            return table.copy()

    def get_pch_summary(self) -> pd.DataFrame:
        with self._lock:
            return self._pch_summary.copy()

    def get_pch_tiles(self) -> dict[str, pd.DataFrame]:
        with self._lock:
            return {key: df.copy() for key, df in self._pch_tiles.items()}

    def get_stringing_scope_frames(self) -> dict[str, pd.DataFrame]:
        with self._lock:
            return {key: frame.copy() for key, frame in self._stringing_scope_frames.items()}

    def get_stringing_plan_summary(self) -> pd.DataFrame:
        with self._lock:
            return self._stringing_plan_summary.copy()

    def get_stringing_plan_monthly_totals(
        self,
    ) -> tuple[dict[str, dict[pd.Timestamp, float]], dict[str, dict[pd.Timestamp, float]]]:
        with self._lock:
            planned = {project: dict(month_map) for project, month_map in self._stringing_plan_planned.items()}
            delivered = {project: dict(month_map) for project, month_map in self._stringing_plan_delivered.items()}
            return planned, delivered

    def get_traceability_table(self, mode: str = "erection") -> pd.DataFrame:
        with self._lock:
            return self._traceability_tables.get(mode, pd.DataFrame()).copy()

    def get_project_info(self) -> pd.DataFrame:
        with self._lock:
            return self._project_info.copy() if isinstance(self._project_info, pd.DataFrame) else pd.DataFrame()

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _precompute_erection_structures(self, config: AppConfig) -> None:
        LOGGER.info("Precomputing erection aggregates")
        daily = self.get_daily()
        if daily.empty:
            self._erection_baseline_overall = {}
            self._erection_baseline_monthly = {}
            self._erection_gang_summary = pd.DataFrame()
        else:
            overall, monthly = compute_project_baseline_maps(daily)
            self._erection_baseline_overall = overall
            self._erection_baseline_monthly = monthly
            self._erection_gang_summary = self._build_gang_summary(
                daily,
                metric_column="daily_prod_mt",
                loss_max_gap=config.loss_max_gap_days,
                use_metric_function=False,
            )
        self._prepare_responsibility_store(self._responsibilities, mode="erection")

    def _precompute_stringing_structures(self, config: AppConfig) -> None:
        LOGGER.info("Precomputing stringing aggregates")
        stringing_df = self.get_stringing()
        if stringing_df.empty:
            self._stringing_baseline_overall = {}
            self._stringing_baseline_monthly = {}
            self._stringing_gang_summary = pd.DataFrame()
            self._stringing_scope_frames = {}
            self._stringing_plan_summary = pd.DataFrame()
            self._stringing_plan_planned = {}
            self._stringing_plan_delivered = {}
        else:
            overall, monthly = compute_project_baseline_maps_for(stringing_df, "daily_km")
            self._stringing_baseline_overall = overall
            self._stringing_baseline_monthly = monthly
            self._stringing_gang_summary = self._build_gang_summary(
                stringing_df,
                metric_column="daily_km",
                loss_max_gap=config.loss_max_gap_days,
                use_metric_function=True,
            )
            self._build_stringing_scopes(stringing_df)
        self._prepare_responsibility_store(self._stringing_responsibilities, mode="stringing")
        self._build_stringing_plan_summary(stringing_df)

    def _prepare_daily_frame(self, df: pd.DataFrame, *, mode: str, scope_key_bridge: "dict[str, str] | None" = None) -> pd.DataFrame:
        if df.empty:
            return df
        working = df.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.dropna(subset=["date"]).copy()
        working["month"] = working["date"].dt.to_period("M").dt.to_timestamp()
        if "project_name" not in working.columns:
            working["project_name"] = ""
        working["project_name"] = working["project_name"].astype(str).str.strip()
        if "line_name" in working.columns:
            working["line_name"] = working["line_name"].map(normalize_line_name)
        else:
            working["line_name"] = ""
        if "project_code" not in working.columns:
            working["project_code"] = ""
        working["project_code"] = working["project_code"].fillna("").astype(str).str.strip()
        computed_display = pd.Series(
            [
                build_project_display(code, line, fallback)
                for code, line, fallback in zip(
                    working["project_code"], working["line_name"], working["project_name"]
                )
            ],
            index=working.index,
        )
        if "project_display" in working.columns:
            existing_display = working["project_display"].fillna("").astype(str).str.strip()
            working["project_display"] = existing_display.where(existing_display.astype(bool), computed_display)
            line_mask = working["line_name"].astype(bool)
            if line_mask.any():
                working.loc[line_mask, "project_display"] = computed_display.loc[line_mask]
        else:
            working["project_display"] = computed_display
        computed_scope = pd.Series(
            [
                build_project_scope_key(code, line, fallback)
                for code, line, fallback in zip(
                    working["project_code"], working["line_name"], working["project_name"]
                )
            ],
            index=working.index,
        )
        if "project_scope_key" in working.columns:
            existing_scope = working["project_scope_key"].fillna("").astype(str).str.strip()
            working["project_scope_key"] = existing_scope.where(existing_scope.astype(bool), computed_scope)
        else:
            working["project_scope_key"] = computed_scope
        if "project" not in working.columns:
            working["project"] = ""
        working["project"] = working["project"].fillna("").astype(str).str.strip()
        working["project_name_key"] = working["project_name"].str.lower().str.replace(r"\s+", " ", regex=True)
        project_key = working.get("project_code", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()
        project_key = project_key.where(project_key.astype(bool), working["project_name"])
        project_key = project_key.where(project_key.astype(bool), working["project"])
        project_key = project_key.where(project_key.astype(bool), working["project_display"])
        project_key = project_key.where(project_key.astype(bool), working["project_scope_key"])
        working["project_key"] = project_key
        working["project_key_norm"] = working["project_key"].map(compact_project_key)
        working["project_scope_key_norm"] = working["project_scope_key"].map(compact_project_key)
        if "gang_name" not in working.columns:
            working["gang_name"] = ""
        working["gang_name"] = working["gang_name"].astype(str).str.strip()
        working["gang_key"] = working["gang_name"].str.lower().str.replace(r"\s+", " ", regex=True)
        if mode == "stringing" and "daily_km" in working.columns:
            working["daily_km"] = pd.to_numeric(working["daily_km"], errors="coerce")
        if mode == "erection" and "daily_prod_mt" in working.columns:
            working["daily_prod_mt"] = pd.to_numeric(working["daily_prod_mt"], errors="coerce")
        if mode == "erection" and scope_key_bridge and "project_scope_key" in working.columns:
            working["project_scope_key"] = working["project_scope_key"].map(
                lambda k: scope_key_bridge.get(k, k)
            )
            working["project_scope_key_norm"] = working["project_scope_key"].map(compact_project_key)
        return working

    def _build_gang_summary(
        self,
        data: pd.DataFrame,
        *,
        metric_column: str,
        loss_max_gap: int,
        use_metric_function: bool,
    ) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        gang_overall, gang_monthly = compute_gang_baseline_maps(
            data.assign(daily_prod_mt=data[metric_column]) if metric_column != "daily_prod_mt" else data
        )

        for gang_name, gang_df in data.groupby("gang_name"):
            if use_metric_function:
                idle, baseline, loss, delivered, potential = calc_idle_and_loss_for_column(
                    gang_df,
                    metric_column=metric_column,
                    loss_max_gap_days=loss_max_gap,
                    baseline_per_day=gang_overall.get(gang_name),
                    baseline_by_month=gang_monthly.get(gang_name),
                )
            else:
                idle, baseline, loss, delivered, potential = calc_idle_and_loss(
                    gang_df,
                    loss_max_gap_days=loss_max_gap,
                    baseline_mt_per_day=gang_overall.get(gang_name),
                    baseline_by_month=gang_monthly.get(gang_name),
                )
            rows.append(
                {
                    "gang_name": gang_name,
                    "projects": gang_df["project_name"].nunique(),
                    "idle_days": idle,
                    "baseline": baseline,
                    "lost": loss,
                    "delivered": delivered,
                    "potential": potential,
                    "first_date": gang_df["date"].min(),
                    "last_date": gang_df["date"].max(),
                }
            )
        summary = pd.DataFrame(rows)
        return summary.set_index("gang_name") if not summary.empty else summary

    def _prepare_responsibility_store(self, snapshot: ResponsibilitiesSnapshot, *, mode: str) -> None:
        normalized = self._normalize_responsibility_frame(snapshot)
        index, alias_map = self._build_responsibility_index_from_frame(normalized)
        completion_lookup = self._normalize_completion_pairs(
            set(snapshot.completion_keys or set()),
            alias_map,
        )
        with self._lock:
            if mode == "stringing":
                self._stringing_responsibility_frame = normalized
                self._stringing_responsibility_index = index
                self._stringing_responsibility_alias_map = alias_map
                self._stringing_completion_lookup = completion_lookup
            else:
                self._responsibility_frame = normalized
                self._responsibility_index = index
                self._responsibility_alias_map = alias_map
                self._responsibility_completion_lookup = completion_lookup

    def _normalize_responsibility_frame(self, snapshot: ResponsibilitiesSnapshot) -> pd.DataFrame:
        try:
            frame = snapshot.require_frame()
        except RuntimeError:
            return pd.DataFrame()
        if frame is None or frame.empty:
            return pd.DataFrame()

        working = frame.copy()
        working["project_name"] = working.get("project_name", working.get("project", "")).fillna("").astype(str)
        working["project_name"] = working["project_name"].map(normalize_text)
        working["project_key"] = working.get("project_key", working["project_name"]).fillna("").astype(str)
        working["project_key"] = working["project_key"].map(normalize_text)
        working["project_name_lc"] = working["project_name"].map(normalize_lower)
        working["project_key_lc"] = working["project_key"].map(normalize_lower)
        working["project_key_norm"] = working["project_key"].map(compact_project_key)
        fallback_compact = working["project_name"].map(compact_project_key)
        working["project_key_norm"] = working["project_key_norm"].where(
            working["project_key_norm"].astype(bool), fallback_compact
        )
        working["project_compact"] = fallback_compact

        working["entity_type"] = working.get("entity_type", "").fillna("").astype(str).str.strip()
        working["entity_type_norm"] = working["entity_type"].str.lower()
        working["entity_name"] = working.get("entity_name", "").fillna("").astype(str).str.strip()
        working["location_no"] = working.get("location_no", "").fillna("").astype(str).str.strip()
        working["location_no_norm"] = working["location_no"].map(normalize_location)

        numeric_columns = ("tower_weight", "revenue_planned", "revenue_realised")
        for column in numeric_columns:
            working[column] = pd.to_numeric(working.get(column, 0.0), errors="coerce").fillna(0.0)

        bool_columns = ("stringing_span_completed",)
        for column in bool_columns:
            working[column] = working.get(column, False)
            if column in working.columns:
                working[column] = working[column].fillna(False).astype(bool)
            else:
                working[column] = False

        date_candidates = (
            "plan_month",
            "completion_date",
            "paying_out_start",
            "paying_out_complete",
            "final_sag_complete",
        )
        parsed_dates: dict[str, pd.Series] = {}
        for column in date_candidates:
            if column in working.columns:
                parsed_dates[column] = pd.to_datetime(working[column], errors="coerce")
                working[column] = parsed_dates[column]
        plan_month_series = parsed_dates.get("plan_month")
        if plan_month_series is not None:
            working["plan_month"] = plan_month_series.dt.to_period("M").dt.to_timestamp()
        else:
            working["plan_month"] = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns]")

        completion_series = working["plan_month"].copy()
        if completion_series.isna().all():
            completion_series = pd.Series(pd.NaT, index=working.index)
        for column in ("completion_date", "paying_out_start", "paying_out_complete", "final_sag_complete"):
            series = parsed_dates.get(column)
            if series is not None:
                completion_series = completion_series.where(completion_series.notna(), series)
        working["completion_month"] = completion_series.dt.to_period("M").dt.to_timestamp()
        working["completion_month"] = working["completion_month"].fillna(pd.NaT)

        return working

    def _build_responsibility_index_from_frame(
        self, frame: pd.DataFrame
    ) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, str]]:
        index: dict[str, dict[str, pd.DataFrame]] = {}
        alias_map: dict[str, str] = {}
        if frame.empty or "project_key_norm" not in frame.columns:
            return index, alias_map

        for project_key, project_df in frame.groupby("project_key_norm"):
            key = str(project_key)
            if not key:
                continue
            entity_map: dict[str, pd.DataFrame] = {}
            for entity, entity_df in project_df.groupby("entity_type_norm"):
                entity_norm = str(entity).strip()
                if not entity_norm:
                    continue
                entity_map[entity_norm] = entity_df.reset_index(drop=True)
            if not entity_map:
                continue
            index[key] = entity_map
            alias_candidates = set(project_df["project_key_norm"].dropna().astype(str))
            alias_candidates.update(project_df["project_key_lc"].dropna().astype(str))
            alias_candidates.update(project_df["project_name_lc"].dropna().astype(str))
            alias_candidates.update(project_df["project_compact"].dropna().astype(str))
            alias_candidates.add(key)
            for alias in alias_candidates:
                if alias:
                    alias_map.setdefault(alias, key)
        return index, alias_map

    def _normalize_completion_pairs(
        self,
        completion_pairs: set[tuple[str, str]],
        alias_map: Mapping[str, str],
    ) -> set[tuple[str, str]]:
        normalized: set[tuple[str, str]] = set()
        for project_key, location in completion_pairs:
            canonical = self._resolve_responsibility_project_key(project_key, alias_map)
            location_norm = normalize_location(location)
            if canonical and location_norm:
                normalized.add((canonical, location_norm))
        return normalized

    def _resolve_responsibility_project_candidates(
        self,
        project_candidates: Sequence[str],
        alias_map: Mapping[str, str],
    ) -> list[str]:
        seen: set[str] = set()
        resolved: list[str] = []
        for candidate in project_candidates:
            canonical = self._resolve_responsibility_project_key(candidate, alias_map)
            if canonical and canonical not in seen:
                seen.add(canonical)
                resolved.append(canonical)
        return resolved

    def _resolve_responsibility_project_key(
        self,
        candidate: object,
        alias_map: Mapping[str, str],
    ) -> str | None:
        text = normalize_text(candidate)
        if not text:
            return None
        lookup_keys = [
            compact_project_key(text),
            normalize_lower(text),
            text.strip().lower(),
        ]
        for key in lookup_keys:
            if key and key in alias_map:
                return alias_map[key]
        fallback = compact_project_key(text)
        return fallback if fallback else None

    def _build_pch_tiles(self) -> None:
        LOGGER.info("Building PCH tiles")
        project_info = self.get_project_info()
        daily = self.get_daily()
        if project_info.empty or daily.empty:
            self._pch_tiles = {}
            self._pch_summary = pd.DataFrame()
            return

        lookup = project_info[["key_name", "pch", "project_name", "project_code"]].dropna(subset=["key_name"]).copy()
        lookup["project_code_norm"] = lookup["project_code"].map(compact_project_key)
        lookup["pch"] = lookup["pch"].fillna("Unknown")
        merged = daily.merge(
            lookup,
            left_on="project_key_norm",
            right_on="project_code_norm",
            how="left",
        )
        if "project_name" not in merged.columns:
            if "project_name_x" in merged.columns or "project_name_y" in merged.columns:
                project_name_x = merged.get("project_name_x")
                if project_name_x is None:
                    project_name_x = pd.Series("", index=merged.index)
                project_name_y = merged.get("project_name_y")
                if project_name_y is None:
                    project_name_y = pd.Series("", index=merged.index)
                merged["project_name"] = project_name_x.where(
                    project_name_x.fillna("").astype(str).astype(bool),
                    project_name_y,
                )
                merged["project_name"] = merged["project_name"].fillna("").astype(str)
            else:
                merged["project_name"] = merged.get("project_key", "")
        merged["pch"] = merged["pch"].fillna("Unknown")
        for column in ("project_name_x", "project_name_y"):
            if column in merged.columns:
                merged = merged.drop(columns=[column])
        summary = (
            merged.groupby("pch")
            .agg(
                projects=("project_name", "nunique"),
                gang_count=("gang_name", "nunique"),
                delivered=("daily_prod_mt", "sum"),
            )
            .reset_index()
        )

        gang_map = (
            daily[["gang_name", "project_key_norm"]]
            .drop_duplicates("gang_name")
            .merge(lookup, left_on="project_key_norm", right_on="project_code_norm", how="left")
        )
        if not self._erection_gang_summary.empty:
            gang_summary = self._erection_gang_summary.reset_index()
            gang_summary = gang_summary.merge(gang_map[["gang_name", "pch"]], on="gang_name", how="left")
            loss_by_pch = gang_summary.groupby(gang_summary["pch"].fillna("Unknown"))["lost"].sum()
            summary = summary.merge(
                loss_by_pch.rename("lost_units"),
                left_on="pch",
                right_index=True,
                how="left",
            )
        else:
            summary["lost_units"] = 0.0
        summary["lost_units"] = summary["lost_units"].fillna(0.0)
        self._pch_summary = summary

        per_project = (
            merged.groupby(["pch", "project_name"])
            .agg(
                delivered=("daily_prod_mt", "sum"),
                gang_count=("gang_name", "nunique"),
            )
            .reset_index()
        )
        tiles: dict[str, pd.DataFrame] = {}
        for pch_name, group in per_project.groupby("pch"):
            tiles[str(pch_name)] = group.sort_values("delivered", ascending=False).reset_index(drop=True)
        self._pch_tiles = tiles

    def _build_stringing_scopes(self, frame: pd.DataFrame) -> None:
        scopes = {
            "all": frame.copy(),
            "tse": frame[frame.get("deployment_tse_flag", False)].copy(),
            "manual": frame[~frame.get("deployment_tse_flag", False)].copy(),
            "hotline": frame[frame.get("method_norm", "").eq("hotline")].copy(),
        }
        self._stringing_scope_frames = scopes

    def _build_stringing_plan_summary(self, stringing_daily: pd.DataFrame) -> None:
        frame = pd.DataFrame()
        try:
            frame = self._stringing_responsibilities.require_frame()
        except RuntimeError:
            self._stringing_plan_summary = pd.DataFrame()
            self._stringing_plan_planned = {}
            self._stringing_plan_delivered = {}
            return

        if frame.empty:
            self._stringing_plan_summary = pd.DataFrame()
            self._stringing_plan_planned = {}
            self._stringing_plan_delivered = {}
            return

        working = frame.copy()
        if "project_key_norm" in working.columns:
            working["project_key_norm"] = working["project_key_norm"].fillna("").astype(str)
        else:
            fallback_project = working.get("project_key", working.get("project_name", ""))
            if fallback_project is None:
                fallback_project = pd.Series("", index=working.index)
            working["project_key_norm"] = fallback_project.map(compact_project_key)
        if "paying_out_start" in working:
            paying_out_start_series = pd.to_datetime(working["paying_out_start"], errors="coerce")
        else:
            paying_out_start_series = pd.Series(pd.NaT, index=working.index)
        if "final_sag_complete" in working:
            final_sag_series = pd.to_datetime(working["final_sag_complete"], errors="coerce")
        else:
            final_sag_series = pd.Series(pd.NaT, index=working.index)
        plan_month = pd.to_datetime(paying_out_start_series, errors="coerce")
        plan_month = plan_month.fillna(pd.to_datetime(final_sag_series, errors="coerce"))
        working["plan_month"] = plan_month.dt.to_period("M").dt.to_timestamp()
        working["plan_month"] = pd.to_datetime(working["plan_month"], errors="coerce")
        working["project_name_display"] = working.get("project_name", working.get("project_key", "")).astype(str)
        working["project_key_raw"] = working.get("project_key", working.get("project_name", "")).astype(str)
        location_series = working.get("location_no")
        if location_series is not None:
            working["location_no_norm"] = location_series.map(normalize_location)
        else:
            working["location_no_norm"] = ""
        working = working.sort_values(["project_key_norm", "plan_month"]).drop_duplicates(
            subset=["project_key_norm", "plan_month", "location_no_norm"],
            keep="last",
        )
        if "stringing_span_completed" not in working.columns:
            working["stringing_span_completed"] = False
        else:
            working["stringing_span_completed"] = working["stringing_span_completed"].fillna(False).astype(bool)

        grouped = (
            working.groupby(["project_key_norm", "plan_month"])
            .agg(
                planned_km=("tower_weight", "sum"),
                span_count=("location_no_norm", "nunique"),
                completed_spans=("stringing_span_completed", "sum"),
                project_name_display=("project_name_display", "last"),
                project_key=("project_key_raw", "last"),
            )
            .reset_index()
        )

        delivered = pd.DataFrame(columns=["project_key_norm", "plan_month", "delivered_km"])
        if not stringing_daily.empty:
            if "project_key_norm" not in stringing_daily.columns:
                project_source = stringing_daily.get("project_key")
                if project_source is None:
                    project_source = stringing_daily.get("project_name")
                if project_source is None:
                    project_source = pd.Series("", index=stringing_daily.index)
                stringing_daily["project_key_norm"] = project_source.map(compact_project_key)
            if "month" in stringing_daily.columns:
                month_series = pd.to_datetime(stringing_daily["month"], errors="coerce")
            elif "date" in stringing_daily.columns:
                month_series = pd.to_datetime(stringing_daily["date"], errors="coerce")
            else:
                month_series = pd.Series(pd.NaT, index=stringing_daily.index)
            stringing_daily["month"] = month_series.dt.to_period("M").dt.to_timestamp()
            stringing_daily["month"] = pd.to_datetime(stringing_daily["month"], errors="coerce")
            delivered = (
                stringing_daily.groupby(["project_key_norm", "month"])
                .agg(delivered_km=("daily_km", "sum"))
                .reset_index()
                .rename(columns={"month": "plan_month"})
            )

        delivered["plan_month"] = pd.to_datetime(delivered["plan_month"], errors="coerce")
        plan_summary = grouped.merge(delivered, on=["project_key_norm", "plan_month"], how="left")
        plan_summary["delivered_km"] = plan_summary["delivered_km"].fillna(0.0)
        self._stringing_plan_summary = plan_summary

        planned_map: dict[str, dict[pd.Timestamp, float]] = defaultdict(dict)
        delivered_map: dict[str, dict[pd.Timestamp, float]] = defaultdict(dict)
        for _, row in plan_summary.iterrows():
            project_key = str(row["project_key_norm"])
            plan_month = row["plan_month"]
            planned_map[project_key][plan_month] = float(row.get("planned_km", 0.0) or 0.0)
            delivered_map[project_key][plan_month] = float(row.get("delivered_km", 0.0) or 0.0)
        self._stringing_plan_planned = dict(planned_map)
        self._stringing_plan_delivered = dict(delivered_map)

    def _build_traceability_tables(self, config: AppConfig) -> None:
        daily = self.get_daily()
        if daily.empty:
            self._erection_idle_intervals = pd.DataFrame()
        else:
            self._erection_idle_intervals = compute_idle_intervals_per_gang(
                daily,
                loss_max_gap_days=config.loss_max_gap_days,
            )

        stringing = self.get_stringing()
        if stringing.empty:
            self._stringing_idle_intervals = pd.DataFrame()
        else:
            working = stringing.copy()
            if "daily_prod_mt" not in working.columns and "daily_km" in working.columns:
                working["daily_prod_mt"] = working["daily_km"]
            self._stringing_idle_intervals = compute_idle_intervals_per_gang(
                working,
                loss_max_gap_days=config.loss_max_gap_days,
            )

        self._traceability_tables = {
            "erection": self._erection_idle_intervals.copy(),
            "stringing": self._stringing_idle_intervals.copy(),
        }

    def _maybe_preload_stringing(self, config: AppConfig) -> None:
        if not config.enable_stringing:
            LOGGER.info("Stringing disabled; skipping preload")
            return

        try:
            coverage_df = _load_stringing_coverage(config)
        except Exception as exc:
            LOGGER.warning("Stringing coverage preload failed: %s", exc)
            coverage_df = pd.DataFrame()
        self.set_stringing_coverage(coverage_df)

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
            stringing_df = self._prepare_daily_frame(_load_stringing_daily(config), mode="stringing")
        except Exception as exc:
            LOGGER.warning("Stringing preload failed: %s", exc)
            return
        if stringing_df.empty:
            self.set_stringing(stringing_df)
            LOGGER.info("Preloaded stringing daily rows: 0")
            return

        if "month" not in stringing_df.columns:
            stringing_df["month"] = stringing_df["date"].dt.to_period("M").dt.to_timestamp()
        project_key_series = stringing_df.get("project_key", pd.Series("", index=stringing_df.index)).fillna("").astype(str)
        if "project" in stringing_df.columns:
            project_key_series = project_key_series.where(project_key_series.str.strip().astype(bool), stringing_df["project"].fillna("").astype(str))
        stringing_df["project_key_norm"] = project_key_series.map(compact_project_key)
        self.set_stringing(stringing_df)
        LOGGER.info("Preloaded stringing daily rows: %d", len(stringing_df))

        if isinstance(compiled_df, pd.DataFrame):
            self.set_stringing_compiled(compiled_df)
            LOGGER.info("Preloaded stringing compiled rows: %d", len(compiled_df))

    def _maybe_preload_status_stretch_summary(self, config: AppConfig) -> None:
        try:
            self.set_progress_status_raw(_load_progress_status_raw(config))
            self.set_progress_status_coverage(_load_progress_status_coverage(config))
        except Exception as exc:
            LOGGER.warning("Progress status preload failed: %s", exc)
            self.set_progress_status_raw(pd.DataFrame())
            self.set_progress_status_coverage(pd.DataFrame())

        try:
            self.set_stretch_readiness_raw(_load_stretch_readiness_raw(config))
            self.set_stretch_readiness_summary(_load_stretch_readiness_summary(config))
            self.set_stretch_readiness_manpower_audit(_load_stretch_readiness_manpower_audit(config))
            self.set_stretch_readiness_coverage(_load_stretch_readiness_coverage(config))
        except Exception as exc:
            LOGGER.warning("Stretch readiness preload failed: %s", exc)
            self.set_stretch_readiness_raw(pd.DataFrame())
            self.set_stretch_readiness_summary(pd.DataFrame())
            self.set_stretch_readiness_manpower_audit(pd.DataFrame())
            self.set_stretch_readiness_coverage(pd.DataFrame())

        for table_name in STRINGING_SUMMARY_TABLES:
            try:
                frame = _load_stringing_summary_table(config, table_name)
            except Exception as exc:
                LOGGER.warning("Stringing summary preload failed for %s: %s", table_name, exc)
                frame = pd.DataFrame()
            self.set_stringing_summary_table(table_name, frame)

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
        project_code_series = working.get("project_code")
        if isinstance(project_code_series, pd.Series):
            working["__project_code_norm__"] = project_code_series.fillna("").astype(str).map(compact_project_key)
        else:
            working["__project_code_norm__"] = ""
        map_df = project_info[["key_name", "project_code"]].dropna().copy()
        map_df["__project_code_norm__"] = map_df["project_code"].map(compact_project_key)
        map_df = map_df.drop_duplicates("key_name")
        enriched = working.merge(map_df, left_on="__key_name__", right_on="key_name", how="left")
        if "project_code_y" in enriched.columns:
            base_code = enriched.get("project_code_x", pd.Series("", index=enriched.index)).fillna("").astype(str)
            lookup_code = enriched.get("project_code_y", pd.Series("", index=enriched.index)).fillna("").astype(str)
            enriched["project_code"] = base_code.where(base_code.astype(bool), lookup_code)
            code_missing = ~enriched["project_code"].astype(bool)
            if code_missing.any():
                code_map = (
                    project_info[["project_code"]]
                    .dropna()
                    .assign(__project_code_norm__=project_info["project_code"].map(compact_project_key))
                    .drop_duplicates("__project_code_norm__")
                    .set_index("__project_code_norm__")["project_code"]
                )
                enriched.loc[code_missing, "project_code"] = (
                    enriched.loc[code_missing, "__project_code_norm__"].map(code_map).fillna("")
                )
        enriched = enriched.drop(columns=["__key_name__", "__project_code_norm__", "key_name"], errors="ignore")
        enriched = enriched.drop(columns=["project_code_x", "project_code_y"], errors="ignore")
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
            frame["method_norm"] = pd.Series(dtype="string")
            frame["deployment_tse_flag"] = pd.Series(dtype=bool)
            return
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
