"""Computation helpers for productivity metrics."""
from __future__ import annotations

import logging
from datetime import date
from typing import Mapping, Optional, Tuple

import pandas as pd

from .config import (
    IDLE_BASELINE_ERECTION_FALLBACK,
    IDLE_BASELINE_GENERIC_FALLBACK,
    IDLE_MAX_GAP_DAYS,
)
from .idle_utils import (
    compute_intervals_for_dates,
    split_interval_by_month,
    summarize_gang_intervals,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_IDLE_MAX_GAP_DAYS = IDLE_MAX_GAP_DAYS


def _month_floor(value: pd.Timestamp) -> pd.Timestamp:
    """Return the first day of the month for *value*."""
    return value.to_period("M").to_timestamp()


def _split_interval_by_month(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split an interval into month-aligned segments (inclusive)."""
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current <= end:
        period = current.to_period("M")
        next_month_start = (period + 1).to_timestamp()
        segment_end = min(end, next_month_start - pd.Timedelta(days=1))
        segments.append((current, segment_end))
        current = segment_end + pd.Timedelta(days=1)
    return segments


def _iter_idle_segments(
    gang_name: str,
    gang_df: pd.DataFrame,
    idle_max_gap_days: int,
) -> list[dict[str, object]]:
    """Return idle interval segments for *gang_df* split by month."""
    date_series = pd.to_datetime(gang_df.get("date"), errors="coerce").dropna().dt.date
    dates = date_series.drop_duplicates().sort_values().to_list()
    segments: list[dict[str, object]] = []
    if len(dates) < 2:
        return segments

    intervals = compute_intervals_for_dates(dates, skip_off_system=True)
    for interval in intervals:
        if interval.get("skipped"):
            continue
        interval_start = pd.Timestamp(interval["interval_start"]).normalize()
        interval_end = pd.Timestamp(interval["interval_end"]).normalize()
        capped_remaining = int(min(int(interval["capped_gap_days"]), int(idle_max_gap_days)))
        for seg_start, seg_end in _split_interval_by_month(interval_start, interval_end):
            seg_days = (seg_end - seg_start).days + 1
            capped_days = int(min(seg_days, max(capped_remaining, 0)))
            if capped_days <= 0:
                continue
            segments.append(
                {
                    "gang_name": gang_name,
                    "interval_start": seg_start,
                    "interval_end": seg_end,
                    "interval_month": _month_floor(seg_start),
                    "raw_gap_days": seg_days,
                    "idle_days_capped": capped_days,
                }
            )
            capped_remaining = max(0, capped_remaining - capped_days)

    return segments


def _segments_for_gang(
    gang_name: str,
    gang_df: pd.DataFrame,
    *,
    idle_max_gap_days: int,
    idle_intervals: pd.DataFrame | None,
    allowed_months: set[pd.Timestamp] | None,
) -> list[dict[str, object]]:
    """Return idle segments sourced from precomputed intervals when available."""

    if idle_intervals is None or idle_intervals.empty:
        return _iter_idle_segments(gang_name, gang_df, idle_max_gap_days)

    segments_df = idle_intervals
    if "gang_name" in segments_df.columns and gang_name:
        segments_df = segments_df[segments_df["gang_name"].astype(str) == gang_name]
    if segments_df.empty:
        return []

    if allowed_months and "interval_month" in segments_df.columns:
        segments_df = segments_df[segments_df["interval_month"].isin(allowed_months)]
    if segments_df.empty:
        return []

    if not gang_df.empty and "date" in gang_df.columns:
        date_series = pd.to_datetime(gang_df["date"], errors="coerce").dropna()
        if not date_series.empty and {"interval_start", "interval_end"}.issubset(segments_df.columns):
            start_date = date_series.min()
            end_date = date_series.max()
            segments_df = segments_df[
                (pd.to_datetime(segments_df["interval_start"], errors="coerce") >= start_date)
                & (pd.to_datetime(segments_df["interval_end"], errors="coerce") <= end_date)
            ]
    return segments_df.to_dict("records")


def _resolve_month_series(df: pd.DataFrame) -> pd.Series:
    """Return a Timestamp series representing the month for each row."""
    if "month" in df.columns:
        month_series = pd.to_datetime(df["month"], errors="coerce")
        if month_series.notna().any():
            return month_series
    return pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()


def compute_gang_baseline_maps(
    data: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """Return (overall, per-month) baseline mappings for each gang."""
    if data.empty:
        return {}, {}

    month_series = _resolve_month_series(data)
    working = data.assign(__baseline_month=month_series)

    overall_series = (
        working.groupby("gang_name")["daily_prod_mt"].mean().dropna()
    )
    overall = {gang: float(value) for gang, value in overall_series.items()}

    monthly: dict[str, dict[pd.Timestamp, float]] = {}
    grouped = (
        working.dropna(subset=["__baseline_month"])
        .groupby(["gang_name", "__baseline_month"])["daily_prod_mt"]
        .mean()
        .dropna()
    )
    for (gang, month), value in grouped.items():
        monthly.setdefault(gang, {})[month] = float(value)

    return overall, monthly


def compute_project_baseline_maps(
    data: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[pd.Timestamp, float]]]:
    """
    Return (overall_project_map, monthly_project_map) where keys are project_name.
    Baseline = mean of 'daily_prod_mt' over gang-day rows, i.e., total MT / total gang-days.
    Requires columns: ['project_name', 'date', 'daily_prod_mt'] (or your equivalent).
    """
    if data.empty:
        return {}, {}

    # Reuse your existing month resolver so month logic is consistent
    month_series = _resolve_month_series(data)
    working = data.assign(__baseline_month=month_series)

    # Overall (weighted by gang-days naturally, since it's row-level mean)
    overall_series = (
        working.dropna(subset=["project_name", "daily_prod_mt"])
               .groupby("project_name")["daily_prod_mt"]
               .mean()
               .dropna()
    )
    overall_map = {str(k): float(v) for k, v in overall_series.items()}

    # Monthly (again mean over gang-day rows inside each month)
    month_series = (
        working.dropna(subset=["project_name", "daily_prod_mt", "__baseline_month"])
               .groupby(["project_name", "__baseline_month"])["daily_prod_mt"]
               .mean()
               .dropna()
    )

    monthly_map: dict[str, dict[pd.Timestamp, float]] = {}
    for (project, month_ts), val in month_series.items():
        monthly_map.setdefault(str(project), {})[pd.to_datetime(month_ts)] = float(val)

    return overall_map, monthly_map


def _resolve_idle_gap_days(
    idle_max_gap_days: int,
    legacy_kwargs: Mapping[str, object] | None = None,
) -> int:
    """Accept legacy kwargs while standardizing on idle_max_gap_days."""
    resolved = int(idle_max_gap_days)
    if not legacy_kwargs:
        return resolved
    for key, value in legacy_kwargs.items():
        key_text = str(key)
        if key_text.endswith("gap_days"):
            try:
                resolved = int(value)  # type: ignore[arg-type]
                break
            except (TypeError, ValueError):
                continue
    return resolved


def calc_idle_and_loss(
    group_df: pd.DataFrame,
    idle_max_gap_days: int = DEFAULT_IDLE_MAX_GAP_DAYS,
    baseline_mt_per_day: float | None = None,
    baseline_by_month: Mapping[pd.Timestamp, float] | None = None,
    *,
    idle_intervals: pd.DataFrame | None = None,
    allowed_months: set[pd.Timestamp] | None = None,
    **legacy_kwargs: object,
) -> Tuple[int, float, float, float, float]:
    """Return idle days, baseline, lost MT, delivered MT, and potential MT for a gang."""
    idle_max_gap_days = _resolve_idle_gap_days(idle_max_gap_days, legacy_kwargs)

    gang_name = ""
    if "gang_name" in group_df.columns:
        non_null = group_df["gang_name"].dropna()
        if not non_null.empty:
            gang_name = str(non_null.iloc[0])

    segments = _segments_for_gang(
        gang_name,
        group_df,
        idle_max_gap_days=idle_max_gap_days,
        idle_intervals=idle_intervals,
        allowed_months=allowed_months,
    )
    idle_days_total = float(sum(float(seg.get("idle_days_capped", 0) or 0) for seg in segments))
    idle_days = int(round(idle_days_total))

    fallback_baseline: float | None = None
    if baseline_mt_per_day is not None and not pd.isna(baseline_mt_per_day):
        fallback_baseline = float(baseline_mt_per_day)
    elif len(group_df):
        mean_val = group_df["daily_prod_mt"].mean()
        if not pd.isna(mean_val):
            fallback_baseline = float(mean_val)

    if fallback_baseline is None or fallback_baseline <= 0.0:
        fallback_baseline = IDLE_BASELINE_ERECTION_FALLBACK

    loss_mt = 0.0
    weighted_total = 0.0
    weighted_days = 0.0

    for seg in segments:
        month_key = seg["interval_month"]
        baseline_value = None
        if baseline_by_month:
            baseline_value = baseline_by_month.get(month_key)
        if baseline_value is None:
            baseline_value = fallback_baseline

        idle_capped = float(seg["idle_days_capped"])
        if idle_capped > 0:
            weighted_total += baseline_value * idle_capped
            weighted_days += idle_capped
        loss_mt += baseline_value * idle_capped

    if not segments:
        loss_mt = fallback_baseline * idle_days

    if weighted_days > 0:
        baseline = weighted_total / weighted_days
    else:
        baseline = fallback_baseline

    delivered_mt = float(group_df["daily_prod_mt"].sum())
    potential_mt = delivered_mt + loss_mt
    LOGGER.debug(
        "Computed idle/loss metrics: idle=%s, baseline=%.2f, loss=%.2f, delivered=%.2f",
        idle_days,
        baseline,
        loss_mt,
        delivered_mt,
    )
    return idle_days, baseline, loss_mt, delivered_mt, potential_mt


def calc_idle_and_loss_for_column(
    group_df: pd.DataFrame,
    *,
    metric_column: str,
    idle_max_gap_days: int = DEFAULT_IDLE_MAX_GAP_DAYS,
    baseline_per_day: float | None = None,
    baseline_by_month: Mapping[pd.Timestamp, float] | None = None,
    idle_intervals: pd.DataFrame | None = None,
    allowed_months: set[pd.Timestamp] | None = None,
    **legacy_kwargs: object,
) -> Tuple[int, float, float, float, float]:
    """Generic variant of idle/loss computation for an arbitrary metric column.

    Returns (idle_days_capped, baseline_per_day, loss_value, delivered_value, potential_value).

    The idle day detection logic mirrors `calc_idle_and_loss`; only the metric
    aggregation (delivered, baseline units) is parameterized via ``metric_column``.
    """
    if group_df is None or group_df.empty or metric_column not in group_df.columns:
        return 0, float(baseline_per_day or 0.0), 0.0, 0.0, 0.0
    idle_max_gap_days = _resolve_idle_gap_days(idle_max_gap_days, legacy_kwargs)

    gang_name = ""
    if "gang_name" in group_df.columns:
        non_null = group_df["gang_name"].dropna()
        if not non_null.empty:
            gang_name = str(non_null.iloc[0])

    segments = _segments_for_gang(
        gang_name,
        group_df,
        idle_max_gap_days=idle_max_gap_days,
        idle_intervals=idle_intervals,
        allowed_months=allowed_months,
    )
    idle_days_total = float(sum(float(seg.get("idle_days_capped", 0) or 0) for seg in segments))
    idle_days = int(round(idle_days_total))

    metric_series = pd.to_numeric(group_df[metric_column], errors="coerce")

    # Fallback baseline from provided arg -> mean of metric -> small positive default
    fallback_baseline: float | None = None
    if baseline_per_day is not None and not pd.isna(baseline_per_day):
        fallback_baseline = float(baseline_per_day)
    else:
        mean_val = metric_series.mean()
        if not pd.isna(mean_val):
            fallback_baseline = float(mean_val)
    if fallback_baseline is None or fallback_baseline <= 0.0:
        fallback_baseline = IDLE_BASELINE_GENERIC_FALLBACK

    loss_value = 0.0
    weighted_total = 0.0
    weighted_days = 0.0

    for seg in segments:
        month_key = seg["interval_month"]
        baseline_value = None
        if baseline_by_month:
            baseline_value = baseline_by_month.get(month_key)
        if baseline_value is None:
            baseline_value = fallback_baseline

        idle_capped = float(seg["idle_days_capped"])
        if idle_capped > 0:
            weighted_total += baseline_value * idle_capped
            weighted_days += idle_capped
        loss_value += baseline_value * idle_capped

    if not segments:
        loss_value = fallback_baseline * idle_days

    baseline_value = (weighted_total / weighted_days) if weighted_days > 0 else fallback_baseline

    delivered_value = float(metric_series.sum())
    potential_value = delivered_value + loss_value
    return idle_days, baseline_value, loss_value, delivered_value, potential_value


def compute_project_baseline_maps_for(
    data: pd.DataFrame,
    metric_column: str,
) -> tuple[dict[str, float], dict[pd.Timestamp, dict[pd.Timestamp, float]]]:
    """Generic project baseline maps for an arbitrary metric column.

    Baseline is the mean of ``metric_column`` over gang-day rows.
    Returns (overall_project_map, monthly_project_map).
    """
    if data is None or data.empty or metric_column not in data.columns or "project_name" not in data.columns:
        return {}, {}

    working = data.copy()
    working["project_name"] = working["project_name"].astype(str).str.strip()
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=["project_name", metric_column])
    if working.empty:
        return {}, {}

    month_series = _resolve_month_series(working)
    working["__baseline_month"] = month_series

    overall_series = working.groupby("project_name")[metric_column].mean().dropna()
    overall = {str(project): float(value) for project, value in overall_series.items() if not pd.isna(value)}

    monthly: dict[str, dict[pd.Timestamp, float]] = {}
    monthly_series = (
        working.dropna(subset=["__baseline_month"])
        .groupby(["project_name", "__baseline_month"])[metric_column]
        .mean()
        .dropna()
    )
    for (project, month), value in monthly_series.items():
        month_ts = pd.to_datetime(month)
        if pd.isna(month_ts):
            continue
        monthly.setdefault(str(project), {})[pd.Timestamp(month_ts)] = float(value)

    return overall, monthly



def compute_idle_intervals_per_gang(
    data: pd.DataFrame,
    scope_start: date | None = None,
    scope_end: date | None = None,
    baseline_map: Optional[dict[str, float]] = None,
    metric_path: str = "erection",
    skip_off_system: bool = True,
    idle_max_gap_days: int = DEFAULT_IDLE_MAX_GAP_DAYS,
    baseline_month_lookup: Mapping[str, Mapping[pd.Timestamp, float]] | None = None,
    baseline_fallback_map: Mapping[str, float] | None = None,
    **legacy_kwargs: object,
) -> pd.DataFrame:
    """
    Return a DataFrame describing idle intervals per gang.

    Existing interval-level columns are preserved; normalized monthly metrics
    are added via shared idle_utils.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    resolved_idle_max_gap_days = _resolve_idle_gap_days(idle_max_gap_days, legacy_kwargs)

    rows: list[dict[str, object]] = []
    month_lookup = baseline_month_lookup or {}
    fallback_lookup = dict(baseline_fallback_map or {})
    fallback_lookup.update(dict(baseline_map or {}))
    fallback = (
        IDLE_BASELINE_ERECTION_FALLBACK if metric_path == "erection"
        else IDLE_BASELINE_GENERIC_FALLBACK
    )
    gang_col = "gang_name" if "gang_name" in data.columns else "gang_id"

    for gang_name, gang_df in data.groupby(gang_col):
        dates = (
            pd.to_datetime(gang_df.get("date"), errors="coerce")
            .dropna()
            .dt.date
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if len(dates) < 2:
            continue
        gang_scope_start = dates[0]
        gang_scope_end = dates[-1]
        if scope_start is not None:
            gang_scope_start = max(gang_scope_start, scope_start)
        if scope_end is not None:
            gang_scope_end = min(gang_scope_end, scope_end)
        if gang_scope_end < gang_scope_start:
            continue

        baseline_default = fallback_lookup.get(gang_name)
        if baseline_default is None or pd.isna(baseline_default) or baseline_default <= 0.0:
            metric_series = None
            if "metric" in gang_df.columns:
                metric_series = pd.to_numeric(gang_df["metric"], errors="coerce")
            elif "daily_prod_mt" in gang_df.columns:
                metric_series = pd.to_numeric(gang_df["daily_prod_mt"], errors="coerce")
            elif "daily_km" in gang_df.columns:
                metric_series = pd.to_numeric(gang_df["daily_km"], errors="coerce")
            if isinstance(metric_series, pd.Series):
                mean_val = metric_series.mean()
                if not pd.isna(mean_val):
                    baseline_default = float(mean_val)
        if baseline_default is None or baseline_default <= 0.0:
            baseline_default = fallback

        intervals = compute_intervals_for_dates(dates, skip_off_system=skip_off_system)
        for interval in intervals:
            if interval.get("skipped"):
                continue
            capped = min(int(interval["raw_gap_days"]), int(resolved_idle_max_gap_days))
            interval["capped_gap_days"] = capped

        summary = summarize_gang_intervals(
            intervals=intervals,
            scope_start=gang_scope_start,
            scope_end=gang_scope_end,
            gang_id=str(gang_name),
            baseline_mt_per_day=float(baseline_default),
            all_work_dates=dates,
        )
        summary["gang_name"] = str(gang_name)

        monthly_map = month_lookup.get(gang_name, {})
        monthly_rows: list[dict[str, object]] = []
        gang_rows: list[dict[str, object]] = []

        for interval in intervals:
            if interval.get("skipped"):
                continue
            splits = split_interval_by_month(
                interval_start=interval["interval_start"],
                interval_end=interval["interval_end"],
                capped_gap_days=int(interval["capped_gap_days"]),
                raw_gap_days=int(interval["raw_gap_days"]),
            )
            for split in splits:
                month_key = pd.Timestamp(year=int(split["year"]), month=int(split["month"]), day=1)
                baseline_value = monthly_map.get(month_key, baseline_default)
                allocated_days = float(split["allocated_capped_days"])
                monthly_rows.append(
                    {
                        "gang_name": str(gang_name),
                        "year": int(split["year"]),
                        "month": int(split["month"]),
                        "allocated_capped_days": allocated_days,
                        "lost_mt": round(allocated_days * float(baseline_value), 2),
                    }
                )
                gang_rows.append(
                    {
                        "gang_name": str(gang_name),
                        "interval_start": pd.Timestamp(interval["interval_start"]),
                        "interval_end": pd.Timestamp(interval["interval_end"]),
                        "interval_month": month_key,
                        "raw_gap_days": int(interval["raw_gap_days"]),
                        "idle_days_capped": allocated_days,
                        "baseline": float(baseline_value),
                        "scope_start": summary["scope_start"],
                        "scope_end": summary["scope_end"],
                        "scope_days": summary["scope_days"],
                        "scope_months": summary["scope_months"],
                        "active_months": summary["active_months"],
                        "idle_windows": summary["idle_windows"],
                        "idle_days_raw": summary["idle_days_raw"],
                        "avg_raw_gap_days": summary["avg_raw_gap_days"],
                        "idle_windows_per_month": summary["idle_windows_per_month"],
                        "idle_days_per_month": summary["idle_days_per_month"],
                        "idle_windows_per_active_month": summary["idle_windows_per_active_month"],
                        "idle_days_per_active_month": summary["idle_days_per_active_month"],
                        "lost_mt": round(allocated_days * float(baseline_value), 2),
                        "baseline_mt_per_day": summary["baseline_mt_per_day"],
                    }
                )

        if not gang_rows:
            gang_rows.append(
                {
                    "gang_name": str(gang_name),
                    "interval_start": pd.NaT,
                    "interval_end": pd.NaT,
                    "interval_month": pd.NaT,
                    "raw_gap_days": 0,
                    "idle_days_capped": 0.0,
                    "baseline": float(baseline_default),
                    "scope_start": summary["scope_start"],
                    "scope_end": summary["scope_end"],
                    "scope_days": summary["scope_days"],
                    "scope_months": summary["scope_months"],
                    "active_months": summary["active_months"],
                    "idle_windows": summary["idle_windows"],
                    "idle_days_raw": summary["idle_days_raw"],
                    "avg_raw_gap_days": summary["avg_raw_gap_days"],
                    "idle_windows_per_month": summary["idle_windows_per_month"],
                    "idle_days_per_month": summary["idle_days_per_month"],
                    "idle_windows_per_active_month": summary["idle_windows_per_active_month"],
                    "idle_days_per_active_month": summary["idle_days_per_active_month"],
                    "lost_mt": summary["lost_mt"],
                    "baseline_mt_per_day": summary["baseline_mt_per_day"],
                }
            )

        for row in gang_rows:
            row["monthly_attribution"] = monthly_rows
            rows.append(row)

    LOGGER.debug("Identified %d idle interval segments", len(rows))
    return pd.DataFrame(rows)





def compute_stringing_metrics(
    data: pd.DataFrame,
    *,
    idle_max_gap_days: int = DEFAULT_IDLE_MAX_GAP_DAYS,
    baseline_overall_map: Mapping[str, float] | None = None,
    baseline_monthly_map: Mapping[str, Mapping[pd.Timestamp, float]] | None = None,
    **legacy_kwargs: object,
) -> pd.DataFrame:
    """Compute delivered/lost/potential for stringing in KM.

    Mirrors the erection compute but uses ``daily_km`` as the delivered metric
    and baseline units in KM/day. If overall/monthly baselines are not provided,
    falls back to per-gang mean of ``daily_km`` over the given rows.
    """
    if data is None or data.empty:
        return pd.DataFrame(
            columns=[
                "gang_name",
                "delivered_km",
                "lost_km",
                "potential_km",
                "baseline_km_per_day",
                "idle_days_capped",
                "first_date",
                "last_date",
                "active_days",
            ]
        )

    idle_max_gap_days = _resolve_idle_gap_days(idle_max_gap_days, legacy_kwargs)
    overall = dict(baseline_overall_map or {})
    monthly = {k: dict(v) for k, v in (baseline_monthly_map or {}).items()}

    rows: list[dict[str, object]] = []
    for gang_name, gang_df in data.groupby("gang_name"):
        idle, baseline, loss, delivered, potential = calc_idle_and_loss_for_column(
            gang_df,
            metric_column="daily_km",
            idle_max_gap_days=idle_max_gap_days,
            baseline_per_day=overall.get(gang_name),
            baseline_by_month=monthly.get(gang_name),
        )
        rows.append(
            {
                "gang_name": gang_name,
                "delivered_km": delivered,
                "lost_km": loss,
                "potential_km": potential,
                "baseline_km_per_day": baseline,
                "idle_days_capped": idle,
                "first_date": gang_df["date"].min(),
                "last_date": gang_df["date"].max(),
                "active_days": gang_df["date"].nunique(),
            }
        )

    return pd.DataFrame(rows).sort_values("potential_km", ascending=False)
