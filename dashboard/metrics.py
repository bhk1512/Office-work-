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

        interval_start = (dates[index - 1] + pd.Timedelta(days=1)).normalize()
        interval_end = (dates[index] - pd.Timedelta(days=1)).normalize()

        capped_remaining = loss_max_gap_days
        for seg_start, seg_end in _split_interval_by_month(interval_start, interval_end):
            seg_days = (seg_end - seg_start).days + 1
            capped_days = int(min(seg_days, max(capped_remaining, 0)))
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

