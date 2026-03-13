"""
idle_utils.py
Shared idle interval computation used by both analytics.py and metrics.py.
All per-month normalization lives here.
"""

from __future__ import annotations

import pandas as pd
from datetime import date, timedelta
from .config import (
    IDLE_MAX_GAP_DAYS,
    IDLE_OFF_SYSTEM_GAP_DAYS,
    IDLE_NORM_DAYS_PER_MONTH,
)


# ---------------------------------------------------------------------------
# Core: compute raw idle intervals for a single gang's sorted work dates
# ---------------------------------------------------------------------------

def compute_intervals_for_dates(
    sorted_dates: list[date],
    skip_off_system: bool = True,
) -> list[dict]:
    """
    Given a sorted list of unique work dates for a gang, return a list of
    idle interval dicts.

    Each dict contains:
        interval_start  : date  (first idle day)
        interval_end    : date  (last idle day)
        raw_gap_days    : int   (calendar days in gap)
        capped_gap_days : int   (min(raw_gap_days, IDLE_MAX_GAP_DAYS))
        skipped         : bool  (True if gap > IDLE_OFF_SYSTEM_GAP_DAYS and skip_off_system=True)

    Args:
        sorted_dates:      Deduplicated, sorted work dates.
        skip_off_system:   If True, gaps > IDLE_OFF_SYSTEM_GAP_DAYS are skipped
                           (gang treated as off-system / demobilized).
                           Set False only for analytics summary view.
    """
    intervals = []
    for i in range(1, len(sorted_dates)):
        d_prev = sorted_dates[i - 1]
        d_curr = sorted_dates[i]
        gap_days = (d_curr - d_prev).days - 1

        if gap_days < 1:
            continue  # consecutive days, no idle

        skipped = skip_off_system and (gap_days > IDLE_OFF_SYSTEM_GAP_DAYS)
        capped = min(gap_days, IDLE_MAX_GAP_DAYS)

        intervals.append({
            "interval_start": d_prev + timedelta(days=1),
            "interval_end":   d_curr - timedelta(days=1),
            "raw_gap_days":   gap_days,
            "capped_gap_days": 0 if skipped else capped,
            "skipped": skipped,
        })

    return intervals


# ---------------------------------------------------------------------------
# Active month helper
# ---------------------------------------------------------------------------

def compute_active_months(dates: list[date]) -> float:
    """
    Count distinct calendar months in which a gang recorded work.
    """
    if not dates:
        return 0.0
    return float(len({(d.year, d.month) for d in dates}))


# ---------------------------------------------------------------------------
# Deployment window helpers
# ---------------------------------------------------------------------------

def compute_deployment_days(
    dates: list[date],
    off_system_threshold: int = IDLE_OFF_SYSTEM_GAP_DAYS,
    min_days: int = 1,
) -> float:
    """
    Compute in-play deployment days by removing confirmed off-system gaps.

    Deployment days = full first-to-last span minus each internal gap strictly
    greater than `off_system_threshold`. Result is clamped to at least
    `min_days` to keep downstream denominators stable.
    """
    if not dates:
        return float(max(1, int(min_days)))

    unique_sorted = sorted(set(dates))
    if len(unique_sorted) == 1:
        return float(max(1, int(min_days)))

    total_days = (unique_sorted[-1] - unique_sorted[0]).days + 1
    for i in range(1, len(unique_sorted)):
        gap_days = (unique_sorted[i] - unique_sorted[i - 1]).days - 1
        if gap_days > int(off_system_threshold):
            total_days -= gap_days

    return float(max(int(total_days), int(min_days), 1))


def compute_deployment_months(
    dates: list[date],
    off_system_threshold: int = IDLE_OFF_SYSTEM_GAP_DAYS,
    min_days: int = 1,
) -> float:
    """Deployment days converted to months via IDLE_NORM_DAYS_PER_MONTH."""
    deployment_days = compute_deployment_days(
        dates=dates,
        off_system_threshold=off_system_threshold,
        min_days=min_days,
    )
    return deployment_days / IDLE_NORM_DAYS_PER_MONTH


# ---------------------------------------------------------------------------
# Gang-level summary from intervals
# ---------------------------------------------------------------------------

def summarize_gang_intervals(
    intervals: list[dict],
    scope_start: date,
    scope_end: date,
    gang_id: str,
    baseline_mt_per_day: float,
    all_work_dates: list[date] | None = None,
) -> dict:
    """
    Aggregate interval list into a per-gang summary dict.

    Returned keys:
        gang_id
        scope_start, scope_end
        scope_days              : int
        scope_months            : float   (scope_days / IDLE_NORM_DAYS_PER_MONTH)
        active_months           : float   (distinct months with work)
        deployment_days         : float   (scope days minus confirmed off-system gaps)
        deployment_months       : float   (deployment_days / IDLE_NORM_DAYS_PER_MONTH)
        idle_windows            : int     (count of non-skipped intervals)
        idle_days_capped        : int     (sum of capped_gap_days)
        idle_days_raw           : int     (sum of raw_gap_days for non-skipped)
        avg_raw_gap_days        : float   (idle_days_raw / idle_windows, 0 if none)

        --- NORMALIZED (per month) ---
        idle_windows_per_month  : float
        idle_days_per_month     : float
        idle_windows_per_active_month : float
        idle_days_per_active_month    : float
        idle_windows_per_deployment_month : float
        idle_days_per_deployment_month    : float

        --- LOSS ESTIMATION ---
        lost_mt                 : float   (baseline x idle_days_capped)
        potential_mt            : float   (delivered + lost - caller must add delivered)
        baseline_mt_per_day     : float
    """
    scope_days = (scope_end - scope_start).days + 1
    scope_months = scope_days / IDLE_NORM_DAYS_PER_MONTH
    work_dates = all_work_dates or []
    active_months = compute_active_months(work_dates)
    deployment_days = compute_deployment_days(work_dates)
    deployment_months = deployment_days / IDLE_NORM_DAYS_PER_MONTH

    valid = [iv for iv in intervals if not iv["skipped"]]

    idle_windows     = len(valid)
    idle_days_capped = sum(iv["capped_gap_days"] for iv in valid)
    idle_days_raw    = sum(iv["raw_gap_days"]    for iv in valid)
    avg_raw_gap_days = idle_days_raw / idle_windows if idle_windows > 0 else 0.0

    idle_windows_per_month = idle_windows     / scope_months if scope_months > 0 else 0.0
    idle_days_per_month    = idle_days_capped / scope_months if scope_months > 0 else 0.0
    idle_windows_per_active_month = idle_windows / active_months if active_months > 0 else 0.0
    idle_days_per_active_month = idle_days_capped / active_months if active_months > 0 else 0.0
    idle_windows_per_deployment_month = (
        idle_windows / deployment_months if deployment_months > 0 else 0.0
    )
    idle_days_per_deployment_month = (
        idle_days_capped / deployment_months if deployment_months > 0 else 0.0
    )

    lost_mt = baseline_mt_per_day * idle_days_capped

    return {
        "gang_id":               gang_id,
        "scope_start":           scope_start,
        "scope_end":             scope_end,
        "scope_days":            scope_days,
        "scope_months":          round(scope_months, 3),
        "active_months":         round(active_months, 3),
        "deployment_days":       round(deployment_days, 3),
        "deployment_months":     round(deployment_months, 3),
        "idle_windows":          idle_windows,
        "idle_days_capped":      idle_days_capped,
        "idle_days_raw":         idle_days_raw,
        "avg_raw_gap_days":      round(avg_raw_gap_days, 2),
        "idle_windows_per_month": round(idle_windows_per_month, 3),
        "idle_days_per_month":   round(idle_days_per_month, 3),
        "idle_windows_per_active_month": round(idle_windows_per_active_month, 3),
        "idle_days_per_active_month": round(idle_days_per_active_month, 3),
        "idle_windows_per_deployment_month": round(idle_windows_per_deployment_month, 3),
        "idle_days_per_deployment_month": round(idle_days_per_deployment_month, 3),
        "lost_mt":               round(lost_mt, 2),
        "baseline_mt_per_day":   baseline_mt_per_day,
    }


# ---------------------------------------------------------------------------
# Monthly split helper (used by metrics.py for monthly attribution)
# ---------------------------------------------------------------------------

def split_interval_by_month(
    interval_start: date,
    interval_end: date,
    capped_gap_days: int,
    raw_gap_days: int,
) -> list[dict]:
    """
    Split a single idle interval across calendar months it spans.
    Proportional allocation: month_share = (days_in_month / raw_gap_days) x capped_gap_days.

    Returns list of dicts with keys:
        year, month, days_in_month, allocated_capped_days
    """
    if raw_gap_days == 0:
        return []

    result = []
    cursor = interval_start

    while cursor <= interval_end:
        month_end = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        chunk_end = min(interval_end, month_end)
        days_in_month = (chunk_end - cursor).days + 1
        allocated = round((days_in_month / raw_gap_days) * capped_gap_days, 4)

        result.append({
            "year":                  cursor.year,
            "month":                 cursor.month,
            "days_in_month":         days_in_month,
            "allocated_capped_days": allocated,
        })
        cursor = chunk_end + timedelta(days=1)

    return result


# ---------------------------------------------------------------------------
# Scope utilities
# ---------------------------------------------------------------------------

def derive_scope_bounds(df: pd.DataFrame, date_col: str = "date") -> tuple[date, date]:
    """
    Derive scope start/end from a filtered dataframe's date column.
    Use this when no explicit scope dates are passed from filters.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    return dates.min().date(), dates.max().date()


def scope_months(scope_start: date, scope_end: date) -> float:
    """Return scope length in months."""
    return ((scope_end - scope_start).days + 1) / IDLE_NORM_DAYS_PER_MONTH


def recovery_mt_estimate(
    gangs_summary: list[dict],
    reduction_days_per_month: float = 0.5,
) -> dict:
    """
    Leadership-facing: estimate MT recoverable if each gang reduces idle
    by `reduction_days_per_month` idle days per month.

    Args:
        gangs_summary:             List of per-gang summary dicts (from summarize_gang_intervals).
        reduction_days_per_month:  Target reduction in idle days/month (default 0.5).

    Returns dict:
        total_recovery_mt         : float
        per_gang_avg_recovery_mt  : float
        gang_count                : int
        reduction_days_per_month  : float
        assumptions               : list[str]   (audit trail for leadership deck)
    """
    total = 0.0
    for g in gangs_summary:
        recoverable_days = reduction_days_per_month * g["scope_months"]
        total += g["baseline_mt_per_day"] * recoverable_days

    return {
        "total_recovery_mt":        round(total, 1),
        "per_gang_avg_recovery_mt": round(total / len(gangs_summary), 2) if gangs_summary else 0.0,
        "gang_count":               len(gangs_summary),
        "reduction_days_per_month": reduction_days_per_month,
        "assumptions": [
            f"Idle reduction target: {reduction_days_per_month} days/month/gang",
            f"Scope months vary per gang (from scope_start to scope_end)",
            "Baseline MT/day is gang-specific (falls back to config default if missing)",
            f"IDLE_MAX_GAP_DAYS cap = {IDLE_MAX_GAP_DAYS}",
            f"Off-system gaps (>{IDLE_OFF_SYSTEM_GAP_DAYS} days) excluded",
        ],
    }
