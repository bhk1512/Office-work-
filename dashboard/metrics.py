"""Computation helpers for productivity metrics."""
from __future__ import annotations

import logging
from typing import Mapping, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig

LOGGER = logging.getLogger(__name__)
DEFAULT_LOSS_MAX_GAP_DAYS = AppConfig().loss_max_gap_days


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
    loss_max_gap_days: int,
) -> list[dict[str, object]]:
    """Return idle interval segments for *gang_df* split by month."""
    dates = gang_df["date"].dropna().drop_duplicates().sort_values().to_list()
    segments: list[dict[str, object]] = []
    if len(dates) < 2:
        return segments

    for index in range(1, len(dates)):
        gap = (dates[index] - dates[index - 1]).days - 1
        if gap < 1:
            continue
        if gap > loss_max_gap_days:
            # Treat long gaps as gang off-system; skip loss accounting entirely.
            continue

        interval_start = (dates[index - 1] + pd.Timedelta(days=1)).normalize()
        interval_end = (dates[index] - pd.Timedelta(days=1)).normalize()

        capped_remaining = loss_max_gap_days
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

def calc_idle_and_loss(
    group_df: pd.DataFrame,
    loss_max_gap_days: int = DEFAULT_LOSS_MAX_GAP_DAYS,
    baseline_mt_per_day: float | None = None,
    baseline_by_month: Mapping[pd.Timestamp, float] | None = None,
) -> Tuple[int, float, float, float, float]:
    """Return idle days, baseline, lost MT, delivered MT, and potential MT for a gang."""

    gang_name = ""
    if "gang_name" in group_df.columns:
        non_null = group_df["gang_name"].dropna()
        if not non_null.empty:
            gang_name = str(non_null.iloc[0])

    segments = _iter_idle_segments(gang_name, group_df, loss_max_gap_days)
    idle_days = int(sum(seg["idle_days_capped"] for seg in segments))

    fallback_baseline: float | None = None
    if baseline_mt_per_day is not None and not pd.isna(baseline_mt_per_day):
        fallback_baseline = float(baseline_mt_per_day)
    elif len(group_df):
        mean_val = group_df["daily_prod_mt"].mean()
        if not pd.isna(mean_val):
            fallback_baseline = float(mean_val)

    if fallback_baseline is None or fallback_baseline <= 0.0:
        fallback_baseline = 5.0

    loss_mt = 0.0
    weighted_total = 0.0
    weighted_days = 0

    for seg in segments:
        month_key = seg["interval_month"]
        baseline_value = None
        if baseline_by_month:
            baseline_value = baseline_by_month.get(month_key)
        if baseline_value is None:
            baseline_value = fallback_baseline

        idle_capped = int(seg["idle_days_capped"])
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
    loss_max_gap_days: int = DEFAULT_LOSS_MAX_GAP_DAYS,
    baseline_per_day: float | None = None,
    baseline_by_month: Mapping[pd.Timestamp, float] | None = None,
) -> Tuple[int, float, float, float, float]:
    """Generic variant of idle/loss computation for an arbitrary metric column.

    Returns (idle_days_capped, baseline_per_day, loss_value, delivered_value, potential_value).

    The idle day detection logic mirrors `calc_idle_and_loss`; only the metric
    aggregation (delivered, baseline units) is parameterized via ``metric_column``.
    """
    if group_df is None or group_df.empty or metric_column not in group_df.columns:
        return 0, float(baseline_per_day or 0.0), 0.0, 0.0, 0.0

    gang_name = ""
    if "gang_name" in group_df.columns:
        non_null = group_df["gang_name"].dropna()
        if not non_null.empty:
            gang_name = str(non_null.iloc[0])

    segments = _iter_idle_segments(gang_name, group_df, loss_max_gap_days)
    idle_days = int(sum(seg["idle_days_capped"] for seg in segments))

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
        fallback_baseline = 1.0

    loss_value = 0.0
    weighted_total = 0.0
    weighted_days = 0

    for seg in segments:
        month_key = seg["interval_month"]
        baseline_value = None
        if baseline_by_month:
            baseline_value = baseline_by_month.get(month_key)
        if baseline_value is None:
            baseline_value = fallback_baseline

        idle_capped = int(seg["idle_days_capped"])
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
    loss_max_gap_days: int = DEFAULT_LOSS_MAX_GAP_DAYS,
    baseline_month_lookup: Mapping[str, Mapping[pd.Timestamp, float]] | None = None,
    baseline_fallback_map: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame describing idle intervals per gang."""

    rows: list[dict[str, object]] = []
    month_lookup = baseline_month_lookup or {}
    fallback_lookup = baseline_fallback_map or {}

    for gang_name, gang_df in data.groupby("gang_name"):
        segments = _iter_idle_segments(gang_name, gang_df, loss_max_gap_days)
        if not segments:
            continue

        baseline_default = fallback_lookup.get(gang_name)
        if baseline_default is None or pd.isna(baseline_default) or baseline_default <= 0.0:
            mean_val = gang_df["daily_prod_mt"].mean()
            if not pd.isna(mean_val):
                baseline_default = float(mean_val)
        if baseline_default is None or baseline_default <= 0.0:
            baseline_default = 5.0

        monthly_map = month_lookup.get(gang_name, {})

        for seg in segments:
            month_key = seg["interval_month"]
            baseline_value = monthly_map.get(month_key, baseline_default)
            row = dict(seg)
            row["baseline"] = float(baseline_value)
            row.pop("interval_month", None)
            rows.append(row)

    LOGGER.debug("Identified %d idle interval segments", len(rows))
    return pd.DataFrame(rows)





def compute_stringing_metrics(
    data: pd.DataFrame,
    *,
    loss_max_gap_days: int = DEFAULT_LOSS_MAX_GAP_DAYS,
    baseline_overall_map: Mapping[str, float] | None = None,
    baseline_monthly_map: Mapping[str, Mapping[pd.Timestamp, float]] | None = None,
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

    overall = dict(baseline_overall_map or {})
    monthly = {k: dict(v) for k, v in (baseline_monthly_map or {}).items()}

    rows: list[dict[str, object]] = []
    for gang_name, gang_df in data.groupby("gang_name"):
        idle, baseline, loss, delivered, potential = calc_idle_and_loss_for_column(
            gang_df,
            metric_column="daily_km",
            loss_max_gap_days=loss_max_gap_days,
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
