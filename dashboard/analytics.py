"""Analytics computations for the erection dashboard."""
from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from .config import (
    IDLE_BASELINE_ERECTION_FALLBACK,
    IDLE_BASELINE_GENERIC_FALLBACK,
    IDLE_MAX_GAP_DAYS,
    IDLE_MIN_COMPLETIONS_FOR_TIER,
    IDLE_NORM_DAYS_PER_MONTH,
)
from .idle_utils import compute_intervals_for_dates, derive_scope_bounds, summarize_gang_intervals


# Analytics tuning notes:
# - PRODUCTIVITY_TIER_LOW / PRODUCTIVITY_TIER_HIGH control tier thresholds (<4 / 4-6 / >6).
# - IDLE_CAP_DAYS caps idle days per gap at IDLE_MAX_GAP_DAYS.
# - MIN_ERECTIONS_FOR_TIERS excludes gangs with fewer than 3 completions from tier stats.
PRODUCTIVITY_TIER_LOW = 4.0
PRODUCTIVITY_TIER_HIGH = 6.0
IDLE_CAP_DAYS = IDLE_MAX_GAP_DAYS
MIN_ERECTIONS_FOR_TIERS = IDLE_MIN_COMPLETIONS_FOR_TIER
HISTOGRAM_MAX_BIN = 13


@dataclass(frozen=True)
class AnalyticsPayload:
    kpis: dict
    bucket: dict
    tiers: dict
    histogram: dict
    hotspot: dict
    trends: dict
    pareto: dict
    whatif: dict

    def to_dict(self) -> dict:
        return {
            "kpis": self.kpis,
            "bucket": self.bucket,
            "tiers": self.tiers,
            "histogram": self.histogram,
            "hotspot": self.hotspot,
            "trends": self.trends,
            "pareto": self.pareto,
            "whatif": self.whatif,
        }


def compute_analytics_idle_summary(
    df: pd.DataFrame,
    scope_start: Optional[date] = None,
    scope_end: Optional[date] = None,
    baseline_map: Optional[dict] = None,
    metric_path: str = "erection",
) -> pd.DataFrame:
    """
    Analytics tab idle summary.
    Returns one row per gang with tier, hotspot, and normalized idle metrics.
    """
    frame = _prepare_daily_frame(df)
    if frame.empty:
        return pd.DataFrame()

    metric_col = "daily_prod_mt" if metric_path == "erection" else "metric"
    if metric_col not in frame.columns:
        metric_col = "daily_prod_mt" if "daily_prod_mt" in frame.columns else metric_col
    fallback = (
        IDLE_BASELINE_ERECTION_FALLBACK if metric_path == "erection"
        else IDLE_BASELINE_GENERIC_FALLBACK
    )

    records: list[dict[str, object]] = []
    baseline_lookup = baseline_map or {}
    for gang_name, group in frame.groupby("gang_name"):
        completions = (
            pd.to_datetime(group.get("completion_date"), errors="coerce").notna().sum()
            if "completion_date" in group.columns
            else 0
        )
        tier_eligible = completions >= IDLE_MIN_COMPLETIONS_FOR_TIER

        dates = (
            pd.to_datetime(group.get("date"), errors="coerce")
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

        baseline = baseline_lookup.get(gang_name)
        metric_series = pd.to_numeric(group.get(metric_col), errors="coerce")
        if baseline is None:
            metric_mean = metric_series.mean()
            baseline = float(metric_mean) if not pd.isna(metric_mean) and metric_mean > 0 else fallback

        intervals = compute_intervals_for_dates(dates, skip_off_system=False)
        summary = summarize_gang_intervals(
            intervals=intervals,
            scope_start=gang_scope_start,
            scope_end=gang_scope_end,
            gang_id=str(gang_name),
            baseline_mt_per_day=float(baseline),
            all_work_dates=dates,
        )

        avg_productivity = float(metric_series.mean()) if metric_series.notna().any() else 0.0
        if avg_productivity < PRODUCTIVITY_TIER_LOW:
            tier = "Low"
        elif avg_productivity <= PRODUCTIVITY_TIER_HIGH:
            tier = "Mid"
        else:
            tier = "High"

        if "completion_date" in group.columns:
            completion_dates = pd.to_datetime(group["completion_date"], errors="coerce").dt.normalize()
            work_dates = pd.to_datetime(group["date"], errors="coerce").dt.normalize()
            tower_count = int((completion_dates.notna() & work_dates.notna() & completion_dates.eq(work_dates)).sum())
        else:
            tower_count = 0
        idle_days_per_100 = (
            (summary["idle_days_capped"] / tower_count * 100) if tower_count > 0 else None
        )

        records.append(
            {
                **summary,
                "gang_name": str(gang_name),
                "avg_productivity": round(avg_productivity, 2),
                "tier": tier,
                "tier_eligible": bool(tier_eligible),
                "tower_count": int(tower_count),
                "idle_days_per_100": round(idle_days_per_100, 2) if idle_days_per_100 is not None else None,
            }
        )

    return pd.DataFrame(records)


def compute_tier_summary(analytics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-gang analytics into tier-level summary.
    Includes normalized columns avg_idle_windows_per_month, avg_idle_days_per_month.
    """
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(
            columns=[
                "tier",
                "gang_count",
                "avg_idle_windows",
                "avg_idle_days",
                "avg_idle_windows_per_month",
                "avg_idle_days_per_month",
                "avg_idle_windows_per_active_month",
                "avg_idle_days_per_active_month",
                "avg_raw_gap_days",
                "avg_productivity",
            ]
        )

    if "tier_eligible" in analytics_df.columns:
        eligible = analytics_df[analytics_df["tier_eligible"]].copy()
    else:
        eligible = analytics_df.copy()
    for column in (
        "idle_windows",
        "idle_days_capped",
        "idle_windows_per_month",
        "idle_days_per_month",
        "idle_windows_per_active_month",
        "idle_days_per_active_month",
    ):
        if column not in eligible.columns:
            eligible[column] = 0.0
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "tier",
                "gang_count",
                "avg_idle_windows",
                "avg_idle_days",
                "avg_idle_windows_per_month",
                "avg_idle_days_per_month",
                "avg_idle_windows_per_active_month",
                "avg_idle_days_per_active_month",
                "avg_raw_gap_days",
                "avg_productivity",
            ]
        )

    return (
        eligible.groupby("tier")
        .agg(
            gang_count=("gang_name", "count"),
            avg_idle_windows=("idle_windows", "mean"),
            avg_idle_days=("idle_days_capped", "mean"),
            avg_idle_windows_per_month=("idle_windows_per_month", "mean"),
            avg_idle_days_per_month=("idle_days_per_month", "mean"),
            avg_idle_windows_per_active_month=("idle_windows_per_active_month", "mean"),
            avg_idle_days_per_active_month=("idle_days_per_active_month", "mean"),
            avg_raw_gap_days=("avg_raw_gap_days", "mean"),
            avg_productivity=("avg_productivity", "mean"),
        )
        .round(2)
        .reset_index()
    )


def build_analytics_payload(
    daily_df: pd.DataFrame,
    *,
    idle_cap_days: int = IDLE_CAP_DAYS,
    min_erections: int = MIN_ERECTIONS_FOR_TIERS,
) -> dict:
    """Return a serializable analytics payload from a filtered daily dataframe."""
    frame = _prepare_daily_frame(daily_df)
    if frame.empty:
        return _empty_payload()

    completions = _completion_rows(frame)
    gang_month_summary, gang_month_rows = _compute_gang_month_buckets(frame)
    idle_intervals = _compute_idle_intervals(frame, idle_cap_days=idle_cap_days)
    gang_metrics = _compute_gang_metrics(frame, completions, idle_intervals)
    gang_metrics["tier"] = gang_metrics["avg_prod_mt_day"].map(_assign_tier)
    gang_metrics["hist_bin"] = _assign_hist_bins(gang_metrics["avg_prod_mt_day"])

    tier_source = gang_metrics.copy()
    if "erections_completed" in tier_source.columns:
        tier_source = tier_source[tier_source["erections_completed"] >= min_erections]

    tier_summary = _compute_tier_summary(tier_source)
    histogram_summary = _compute_histogram(gang_metrics)
    project_summary, project_gang_summary = _compute_hotspot_summary(
        frame,
        completions,
        idle_cap_days=idle_cap_days,
    )
    hotspot_top10 = _filter_hotspot_top10(project_summary)
    trends = _compute_trends(
        frame,
        idle_cap_days=idle_cap_days,
        min_erections=min_erections,
    )
    pareto = _compute_pareto_metrics(gang_month_rows)
    whatif_inputs = _compute_whatif_inputs(gang_month_rows)

    kpis = _build_kpis(
        gang_month_summary,
        tier_summary,
        project_summary,
    )

    payload = AnalyticsPayload(
        kpis=kpis,
        bucket={
            "summary": _serialize_frame(gang_month_summary, month_cols=()),
            "gang_months": _serialize_frame(gang_month_rows, month_cols=("month",)),
        },
        tiers={
            "summary": _serialize_frame(tier_summary, month_cols=()),
            "gangs": _serialize_frame(gang_metrics, month_cols=()),
            "idle_intervals": _serialize_frame(
                idle_intervals,
                month_cols=(),
                date_cols=("interval_start", "interval_end"),
            ),
        },
        histogram=histogram_summary,
        hotspot={
            "projects": _serialize_frame(project_summary, month_cols=()),
            "top10": _serialize_frame(hotspot_top10, month_cols=()),
            "gangs": _serialize_frame(project_gang_summary, month_cols=()),
        },
        trends={
            "low_bucket": _serialize_frame(trends.get("low_bucket", pd.DataFrame()), month_cols=("month",)),
            "idle_windows": _serialize_frame(trends.get("idle_windows", pd.DataFrame()), month_cols=("month",)),
        },
        pareto=pareto,
        whatif=whatif_inputs,
    )
    result = payload.to_dict()
    result["trend_df_low_bucket"] = result["trends"]["low_bucket"]
    result["trend_df_idle_windows"] = result["trends"]["idle_windows"]
    result["hotspot_top10_df"] = result["hotspot"]["top10"]
    result["pareto_metrics"] = result["pareto"]
    result["whatif_base_inputs"] = result["whatif"]
    return result


def _prepare_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    if "date" not in working.columns:
        return pd.DataFrame()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"]).copy()
    if "month" not in working.columns:
        working["month"] = working["date"].dt.to_period("M").dt.to_timestamp()
    if "project_name" not in working.columns:
        working["project_name"] = working.get("project", "")
    working["project_name"] = working["project_name"].fillna("").astype(str).str.strip()
    if "gang_name" not in working.columns:
        working["gang_name"] = ""
    working["gang_name"] = working["gang_name"].fillna("").astype(str).str.strip()
    if "location_no" not in working.columns:
        working["location_no"] = ""
    if "daily_prod_mt" not in working.columns:
        return pd.DataFrame()
    working["daily_prod_mt"] = pd.to_numeric(working["daily_prod_mt"], errors="coerce")
    working = working.dropna(subset=["daily_prod_mt"]).copy()
    if "completion_date" in working.columns:
        working["completion_date"] = pd.to_datetime(
            working["completion_date"],
            errors="coerce",
        ).dt.normalize()
    return working


def _completion_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "completion_date" not in frame.columns:
        return pd.DataFrame()
    completion_date = pd.to_datetime(frame["completion_date"], errors="coerce")
    mask = completion_date.notna() & (frame["date"] == completion_date)
    if not mask.any():
        return pd.DataFrame()
    return frame.loc[mask].copy()


def _join_unique(values: Iterable[object]) -> str:
    cleaned = sorted({str(value).strip() for value in values if str(value).strip()})
    return ", ".join(cleaned)


def _compute_gang_month_buckets(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return _empty_bucket_frames()
    grouped = (
        frame.groupby(["gang_name", "month"])
        .agg(
            total_mt=("daily_prod_mt", "sum"),
            active_days=("date", "nunique"),
            projects=("project_name", _join_unique),
        )
        .reset_index()
    )
    if grouped.empty:
        return _empty_bucket_frames()

    grouped["avg_mt_day"] = (
        grouped["total_mt"].astype(float) / grouped["active_days"].replace(0, np.nan).astype(float)
    ).fillna(0.0)

    bins, labels = _bucket_bins_and_labels(grouped["avg_mt_day"].max())
    grouped["bucket_label"] = pd.cut(
        grouped["avg_mt_day"].clip(lower=0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)

    total_gang_months = float(len(grouped.index))
    total_mt = float(grouped["total_mt"].sum()) if total_gang_months else 0.0

    summary = (
        grouped.groupby("bucket_label")
        .agg(
            gang_months=("gang_name", "size"),
            mt_total=("total_mt", "sum"),
            active_days_total=("active_days", "sum"),
        )
        .reset_index()
    )
    summary = summary.set_index("bucket_label").reindex(labels, fill_value=0).reset_index()
    summary["gang_month_share"] = (
        summary["gang_months"].astype(float) / total_gang_months if total_gang_months else 0.0
    )
    summary["mt_share"] = summary["mt_total"].astype(float) / total_mt if total_mt else 0.0
    summary["avg_mt"] = (
        summary["mt_total"].astype(float) / summary["gang_months"].replace(0, np.nan)
    ).fillna(0.0)
    summary["avg_mt_day"] = (
        summary["mt_total"].astype(float) / summary["active_days_total"].replace(0, np.nan)
    ).fillna(0.0)
    summary["avg_active_days"] = (
        summary["active_days_total"].astype(float) / summary["gang_months"].replace(0, np.nan)
    ).fillna(0.0)
    return summary, grouped


def _bucket_bins_and_labels(max_mt: float | int | None) -> tuple[list[float], list[str]]:
    edges = [0, 4, 6, 8, 10, 12, float("inf")]
    labels = ["0-4", "4-6", "6-8", "8-10", "10-12", "12+"]
    return edges, labels


def _compute_idle_intervals(frame: pd.DataFrame, *, idle_cap_days: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(columns=["gang_name", "interval_start", "interval_end", "raw_gap_days", "idle_days_capped"])

    for gang_name, gang_df in frame.groupby("gang_name"):
        dates = (
            pd.to_datetime(gang_df["date"], errors="coerce")
            .dropna()
            .dt.date
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if len(dates) < 2:
            continue
        intervals = compute_intervals_for_dates(dates, skip_off_system=False)
        for interval in intervals:
            raw_gap = int(interval["raw_gap_days"])
            if raw_gap <= 0:
                continue
            rows.append(
                {
                    "gang_name": gang_name,
                    "interval_start": pd.Timestamp(interval["interval_start"]).normalize(),
                    "interval_end": pd.Timestamp(interval["interval_end"]).normalize(),
                    "raw_gap_days": raw_gap,
                    "idle_days_capped": int(min(raw_gap, idle_cap_days)),
                }
            )
    return pd.DataFrame(rows)


def _compute_gang_metrics(
    frame: pd.DataFrame,
    completions: pd.DataFrame,
    idle_intervals: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    productivity = frame.groupby("gang_name")["daily_prod_mt"].mean().rename("avg_prod_mt_day")
    project_mix = frame.groupby("gang_name")["project_name"].agg(_join_unique).rename("projects")
    project_count = frame.groupby("gang_name")["project_name"].nunique().rename("project_count")

    if completions.empty:
        towers = pd.Series(0, index=productivity.index, name="towers")
        erections = pd.Series(0, index=productivity.index, name="erections_completed")
    else:
        if "location_no" in completions.columns:
            towers = completions.groupby("gang_name")["location_no"].nunique().rename("towers")
        else:
            towers = completions.groupby("gang_name").size().rename("towers")
        erections = towers.rename("erections_completed")

    if idle_intervals.empty:
        idle_windows = pd.Series(0, index=productivity.index, name="idle_windows")
        idle_days = pd.Series(0, index=productivity.index, name="idle_days_capped")
        avg_gap = pd.Series(0.0, index=productivity.index, name="avg_raw_gap_days")
    else:
        idle_windows = idle_intervals.groupby("gang_name").size().rename("idle_windows")
        idle_days = idle_intervals.groupby("gang_name")["idle_days_capped"].sum().rename("idle_days_capped")
        avg_gap = idle_intervals.groupby("gang_name")["raw_gap_days"].mean().rename("avg_raw_gap_days")

    gang_metrics = pd.concat(
        [
            productivity,
            idle_windows,
            idle_days,
            avg_gap,
            towers,
            erections,
            project_mix,
            project_count,
        ],
        axis=1,
    ).reset_index()
    gang_metrics = gang_metrics.fillna(
        {
            "idle_windows": 0,
            "idle_days_capped": 0,
            "avg_raw_gap_days": 0.0,
            "towers": 0,
            "erections_completed": 0,
            "projects": "",
            "project_count": 0,
        }
    )
    if "date" in frame.columns and not frame.empty:
        date_work = frame[["gang_name", "date"]].copy()
        date_work["__date"] = pd.to_datetime(date_work["date"], errors="coerce").dt.normalize()
        date_work = date_work.dropna(subset=["__date"])
    else:
        date_work = pd.DataFrame(columns=["gang_name", "__date"])

    if date_work.empty:
        scope_months_by_gang = pd.Series(0.0, index=gang_metrics["gang_name"], dtype="float64")
        active_months_by_gang = pd.Series(0.0, index=gang_metrics["gang_name"], dtype="float64")
    else:
        scope_bounds = date_work.groupby("gang_name")["__date"].agg(["min", "max"])
        scope_months_by_gang = (
            ((scope_bounds["max"] - scope_bounds["min"]).dt.days + 1).astype(float) / IDLE_NORM_DAYS_PER_MONTH
        )
        active_months_by_gang = (
            date_work.groupby("gang_name")["__date"]
            .apply(lambda values: values.dt.to_period("M").nunique())
            .astype(float)
        )

    gang_metrics["scope_months"] = (
        gang_metrics["gang_name"].map(scope_months_by_gang).fillna(0.0).round(3)
    )
    gang_metrics["active_months"] = (
        gang_metrics["gang_name"].map(active_months_by_gang).fillna(0.0).round(3)
    )

    scope_den = gang_metrics["scope_months"].replace(0, np.nan)
    active_den = gang_metrics["active_months"].replace(0, np.nan)
    gang_metrics["idle_windows_per_month"] = (
        gang_metrics["idle_windows"].astype(float) / scope_den
    )
    gang_metrics["idle_days_per_month"] = (
        gang_metrics["idle_days_capped"].astype(float) / scope_den
    )
    gang_metrics["idle_windows_per_active_month"] = (
        gang_metrics["idle_windows"].astype(float) / active_den
    )
    gang_metrics["idle_days_per_active_month"] = (
        gang_metrics["idle_days_capped"].astype(float) / active_den
    )
    gang_metrics["idle_windows_per_month"] = pd.to_numeric(
        gang_metrics["idle_windows_per_month"], errors="coerce"
    ).fillna(0.0).round(3)
    gang_metrics["idle_days_per_month"] = pd.to_numeric(
        gang_metrics["idle_days_per_month"], errors="coerce"
    ).fillna(0.0).round(3)
    gang_metrics["idle_windows_per_active_month"] = pd.to_numeric(
        gang_metrics["idle_windows_per_active_month"], errors="coerce"
    ).fillna(0.0).round(3)
    gang_metrics["idle_days_per_active_month"] = pd.to_numeric(
        gang_metrics["idle_days_per_active_month"], errors="coerce"
    ).fillna(0.0).round(3)
    return gang_metrics


def _assign_tier(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    metric = float(value)
    if metric < PRODUCTIVITY_TIER_LOW:
        return f"Low (<{PRODUCTIVITY_TIER_LOW:g})"
    if metric <= PRODUCTIVITY_TIER_HIGH:
        return f"Mid ({PRODUCTIVITY_TIER_LOW:g}-{PRODUCTIVITY_TIER_HIGH:g})"
    return f"High (>{PRODUCTIVITY_TIER_HIGH:g})"


def _assign_hist_bins(values: pd.Series) -> pd.Series:
    if values is None or values.empty:
        return pd.Series(dtype="string")
    clipped = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0, upper=HISTOGRAM_MAX_BIN)
    edges = list(range(0, HISTOGRAM_MAX_BIN + 1))
    labels = _histogram_labels()
    return pd.cut(
        clipped,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(str)


def _compute_tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    tiers = [
        f"Low (<{PRODUCTIVITY_TIER_LOW:g})",
        f"Mid ({PRODUCTIVITY_TIER_LOW:g}-{PRODUCTIVITY_TIER_HIGH:g})",
        f"High (>{PRODUCTIVITY_TIER_HIGH:g})",
    ]
    if frame.empty or "tier" not in frame.columns:
        return pd.DataFrame(
            {
                "tier": tiers,
                "avg_idle_windows": [0.0, 0.0, 0.0],
                "avg_idle_days": [0.0, 0.0, 0.0],
                "avg_idle_windows_per_month": [0.0, 0.0, 0.0],
                "avg_idle_days_per_month": [0.0, 0.0, 0.0],
                "avg_idle_windows_per_active_month": [0.0, 0.0, 0.0],
                "avg_idle_days_per_active_month": [0.0, 0.0, 0.0],
                "gangs": [0, 0, 0],
            }
        )
    for column in (
        "idle_windows",
        "idle_days_capped",
        "idle_windows_per_month",
        "idle_days_per_month",
        "idle_windows_per_active_month",
        "idle_days_per_active_month",
    ):
        if column not in frame.columns:
            frame[column] = 0.0
    summary = (
        frame.groupby("tier")
        .agg(
            avg_idle_windows=("idle_windows", "mean"),
            avg_idle_days=("idle_days_capped", "mean"),
            avg_idle_windows_per_month=("idle_windows_per_month", "mean"),
            avg_idle_days_per_month=("idle_days_per_month", "mean"),
            avg_idle_windows_per_active_month=("idle_windows_per_active_month", "mean"),
            avg_idle_days_per_active_month=("idle_days_per_active_month", "mean"),
            gangs=("gang_name", "nunique"),
        )
        .reset_index()
    )
    summary = summary.set_index("tier").reindex(tiers, fill_value=0.0).reset_index()
    summary["avg_idle_windows"] = summary["avg_idle_windows"].astype(float)
    summary["avg_idle_days"] = summary["avg_idle_days"].astype(float)
    summary["avg_idle_windows_per_month"] = summary["avg_idle_windows_per_month"].astype(float)
    summary["avg_idle_days_per_month"] = summary["avg_idle_days_per_month"].astype(float)
    summary["avg_idle_windows_per_active_month"] = summary["avg_idle_windows_per_active_month"].astype(float)
    summary["avg_idle_days_per_active_month"] = summary["avg_idle_days_per_active_month"].astype(float)
    summary["gangs"] = summary["gangs"].astype(int)
    return summary


def _compute_histogram(frame: pd.DataFrame) -> dict:
    if frame.empty or "avg_prod_mt_day" not in frame.columns:
        return {
            "bins": [],
            "median_prod": 0.0,
            "pct_below_low": 0.0,
            "pct_above_high": 0.0,
        }
    series = pd.to_numeric(frame["avg_prod_mt_day"], errors="coerce").dropna()
    if series.empty:
        return {
            "bins": [],
            "median_prod": 0.0,
            "pct_below_low": 0.0,
            "pct_above_high": 0.0,
        }
    hist_bins = _assign_hist_bins(series)
    labels = _histogram_labels()
    counts = hist_bins.value_counts().reindex(labels, fill_value=0).reset_index()
    counts.columns = ["bin_label", "count"]
    total = float(len(series))
    pct_below = float((series < PRODUCTIVITY_TIER_LOW).sum()) / total * 100.0
    pct_above = float((series > PRODUCTIVITY_TIER_HIGH).sum()) / total * 100.0
    return {
        "bins": _serialize_frame(counts, month_cols=()),
        "median_prod": float(series.median()),
        "pct_below_low": pct_below,
        "pct_above_high": pct_above,
    }


def _histogram_labels() -> list[str]:
    return [
        ("[0,1]" if idx == 0 else f"({idx},{idx + 1}]")
        for idx in range(0, HISTOGRAM_MAX_BIN)
    ]


def _compute_hotspot_summary(
    frame: pd.DataFrame,
    completions: pd.DataFrame,
    *,
    idle_cap_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return (
            pd.DataFrame(
                columns=[
                    "project_name",
                    "gangs",
                    "towers",
                    "idle_days",
                    "idle_days_per_100",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "project_name",
                    "gang_name",
                    "avg_prod_mt_day",
                    "idle_windows",
                    "idle_days_capped",
                    "towers",
                ]
            ),
        )

    project_rows: list[dict[str, object]] = []
    gang_rows: list[dict[str, object]] = []
    project_names = frame["project_name"].fillna("").astype(str)
    for project_name, project_df in frame.groupby(project_names):
        if not project_name:
            continue
        project_idle = _compute_idle_intervals(project_df, idle_cap_days=idle_cap_days)
        idle_days = float(project_idle["idle_days_capped"].sum()) if not project_idle.empty else 0.0
        if completions.empty:
            project_completions = pd.DataFrame()
        else:
            project_completions = completions[completions["project_name"] == project_name]
        if project_completions.empty:
            towers = 0
        elif "location_no" in project_completions.columns:
            towers = int(project_completions["location_no"].nunique())
        else:
            towers = int(len(project_completions.index))
        gangs = int(project_df["gang_name"].nunique())
        idle_per_100 = (idle_days / towers * 100.0) if towers else 0.0
        project_rows.append(
            {
                "project_name": project_name,
                "gangs": gangs,
                "towers": towers,
                "idle_days": idle_days,
                "idle_days_per_100": idle_per_100,
            }
        )

        productivity = project_df.groupby("gang_name")["daily_prod_mt"].mean()
        if project_idle.empty:
            idle_windows = pd.Series(0, index=productivity.index)
            idle_days_g = pd.Series(0.0, index=productivity.index)
        else:
            idle_windows = project_idle.groupby("gang_name").size()
            idle_days_g = project_idle.groupby("gang_name")["idle_days_capped"].sum()
        if project_completions.empty:
            towers_g = pd.Series(0, index=productivity.index)
        elif "location_no" in project_completions.columns:
            towers_g = project_completions.groupby("gang_name")["location_no"].nunique()
        else:
            towers_g = project_completions.groupby("gang_name").size()
        for gang_name, avg_prod in productivity.items():
            gang_rows.append(
                {
                    "project_name": project_name,
                    "gang_name": gang_name,
                    "avg_prod_mt_day": float(avg_prod) if not pd.isna(avg_prod) else 0.0,
                    "idle_windows": int(idle_windows.get(gang_name, 0)),
                    "idle_days_capped": float(idle_days_g.get(gang_name, 0.0)),
                    "towers": int(towers_g.get(gang_name, 0)),
                }
            )

    project_df = pd.DataFrame(project_rows)
    if not project_df.empty:
        project_df = project_df.sort_values("idle_days_per_100", ascending=False)
    gang_df = pd.DataFrame(gang_rows)
    return project_df, gang_df


def _filter_hotspot_top10(project_summary: pd.DataFrame) -> pd.DataFrame:
    if project_summary is None or project_summary.empty:
        return pd.DataFrame(columns=["project_name", "gangs", "towers", "idle_days", "idle_days_per_100"])
    eligible = project_summary[project_summary["gangs"] >= 10].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["project_name", "gangs", "towers", "idle_days", "idle_days_per_100"])
    return eligible.sort_values("idle_days_per_100", ascending=False).head(10)


def _compute_trends(
    frame: pd.DataFrame,
    *,
    idle_cap_days: int,
    min_erections: int,
) -> dict[str, pd.DataFrame]:
    if frame.empty or "month" not in frame.columns:
        return {
            "low_bucket": pd.DataFrame(columns=["month", "pct_low_bucket"]),
            "idle_windows": pd.DataFrame(columns=["month", "avg_idle_windows"]),
        }

    month_series = pd.to_datetime(frame["month"], errors="coerce")
    months = sorted({ts for ts in month_series.dropna().tolist()})
    low_rows: list[dict[str, object]] = []
    idle_rows: list[dict[str, object]] = []

    for month_value in months:
        month_df = frame[frame["month"] == month_value]
        if month_df.empty:
            continue
        month_buckets, month_rows = _compute_gang_month_buckets(month_df)
        if not month_rows.empty:
            low_count = int((month_rows["bucket_label"] == "0-4").sum())
            total_count = int(len(month_rows.index))
            pct_low = (low_count / total_count * 100.0) if total_count else 0.0
        else:
            pct_low = 0.0
        low_rows.append({"month": month_value, "pct_low_bucket": pct_low})

        eligible_gangs = None
        month_completion = _completion_rows(month_df)
        if not month_completion.empty:
            month_completion["month"] = month_completion["month"].astype("datetime64[ns]")
            counts = month_completion.groupby("gang_name").size()
            eligible_gangs = set(counts[counts >= min_erections].index)
            if not eligible_gangs:
                eligible_gangs = None

        intervals = _compute_idle_intervals(month_df, idle_cap_days=idle_cap_days)
        if intervals.empty:
            avg_idle = 0.0
        else:
            interval_counts = intervals.groupby("gang_name").size()
            if eligible_gangs is not None:
                interval_counts = interval_counts[interval_counts.index.isin(eligible_gangs)]
            avg_idle = float(interval_counts.mean()) if not interval_counts.empty else 0.0
        idle_rows.append({"month": month_value, "avg_idle_windows": avg_idle})

    return {
        "low_bucket": pd.DataFrame(low_rows),
        "idle_windows": pd.DataFrame(idle_rows),
    }


def _compute_pareto_metrics(gang_month_rows: pd.DataFrame) -> dict[str, float]:
    if gang_month_rows is None or gang_month_rows.empty or "total_mt" not in gang_month_rows.columns:
        return {"top20_share": 0.0, "top10_share": 0.0}
    working = gang_month_rows.copy()
    working["total_mt"] = pd.to_numeric(working["total_mt"], errors="coerce").fillna(0.0)
    working = working.sort_values("total_mt", ascending=False)
    total = float(working["total_mt"].sum())
    if total <= 0.0:
        return {"top20_share": 0.0, "top10_share": 0.0}
    total_gm = len(working.index)
    top20_count = max(1, int(np.ceil(0.2 * total_gm))) if total_gm else 0
    top10_count = max(1, int(np.ceil(0.1 * total_gm))) if total_gm else 0
    top20_share = float(working.head(top20_count)["total_mt"].sum()) / total * 100.0 if top20_count else 0.0
    top10_share = float(working.head(top10_count)["total_mt"].sum()) / total * 100.0 if top10_count else 0.0
    return {"top20_share": top20_share, "top10_share": top10_share}


def _compute_whatif_inputs(gang_month_rows: pd.DataFrame) -> dict[str, float]:
    if gang_month_rows is None or gang_month_rows.empty:
        return {
            "low_bucket_count": 0.0,
            "low_bucket_output": 0.0,
            "total_gang_months": 0.0,
            "total_output": 0.0,
            "low_bucket_avg": 0.0,
        }
    working = gang_month_rows.copy()
    working["total_mt"] = pd.to_numeric(working["total_mt"], errors="coerce").fillna(0.0)
    low_mask = working["bucket_label"] == "0-4"
    low_count = float(low_mask.sum())
    low_output = float(working.loc[low_mask, "total_mt"].sum())
    low_active_days = float(pd.to_numeric(working.loc[low_mask, "active_days"], errors="coerce").fillna(0).sum())
    total_gm = float(len(working.index))
    total_output = float(working["total_mt"].sum())
    low_avg = (low_output / low_active_days) if low_active_days else 0.0
    return {
        "low_bucket_count": low_count,
        "low_bucket_output": low_output,
        "total_gang_months": total_gm,
        "total_output": total_output,
        "low_bucket_avg": low_avg,
    }


def _build_kpis(
    bucket_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    project_summary: pd.DataFrame,
) -> dict:
    low_bucket = bucket_summary[bucket_summary["bucket_label"] == "0-4"]
    low_share = float(low_bucket["gang_month_share"].iloc[0]) if not low_bucket.empty else 0.0
    low_mt_share = float(low_bucket["mt_share"].iloc[0]) if not low_bucket.empty else 0.0

    def _tier_value(label: str) -> float:
        if tier_summary.empty:
            return 0.0
        match = tier_summary[tier_summary["tier"] == label]
        if match.empty:
            return 0.0
        return float(match["avg_idle_windows"].iloc[0])

    low_label = f"Low (<{PRODUCTIVITY_TIER_LOW:g})"
    high_label = f"High (>{PRODUCTIVITY_TIER_HIGH:g})"
    low_idle = _tier_value(low_label)
    high_idle = _tier_value(high_label)
    delta_pct = ((high_idle - low_idle) / low_idle * 100.0) if low_idle else 0.0

    top_project = ""
    top_value = 0.0
    next_value = 0.0
    if project_summary is not None and not project_summary.empty:
        eligible = project_summary[project_summary["gangs"] >= 10]
        ranked = eligible if not eligible.empty else project_summary
        ranked = ranked.sort_values("idle_days_per_100", ascending=False)
        top = ranked.iloc[0]
        top_project = str(top.get("project_name", ""))
        top_value = float(top.get("idle_days_per_100", 0.0) or 0.0)
        if len(ranked) > 1:
            next_value = float(ranked.iloc[1].get("idle_days_per_100", 0.0) or 0.0)

    return {
        "low_output_resources_share": low_share,
        "low_output_output_share": low_mt_share,
        "idle_windows_high": high_idle,
        "idle_windows_low": low_idle,
        "delta_pct": delta_pct,
        "top_hotspot_project": top_project,
        "top_hotspot_value": top_value,
        "next_hotspot_value": next_value,
    }


def _serialize_frame(
    frame: pd.DataFrame,
    *,
    month_cols: tuple[str, ...],
    date_cols: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    for column in month_cols:
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce").dt.strftime("%Y-%m")
    for column in date_cols or ():
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in working.select_dtypes(include=[np.number]).columns:
        working[column] = working[column].apply(lambda value: float(value) if not pd.isna(value) else 0.0)
    return working.to_dict("records")


def _empty_bucket_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            columns=[
                "bucket_label",
                "gang_months",
                "mt_total",
                "active_days_total",
                "gang_month_share",
                "mt_share",
                "avg_mt",
                "avg_mt_day",
                "avg_active_days",
            ]
        ),
        pd.DataFrame(
            columns=[
                "gang_name",
                "month",
                "total_mt",
                "active_days",
                "avg_mt_day",
                "projects",
                "bucket_label",
            ]
        ),
    )


def _empty_payload() -> dict:
    payload = AnalyticsPayload(
        kpis={
            "low_output_resources_share": 0.0,
            "low_output_output_share": 0.0,
            "idle_windows_high": 0.0,
            "idle_windows_low": 0.0,
            "delta_pct": 0.0,
            "top_hotspot_project": "",
            "top_hotspot_value": 0.0,
            "next_hotspot_value": 0.0,
        },
        bucket={"summary": [], "gang_months": []},
        tiers={"summary": [], "gangs": [], "idle_intervals": []},
        histogram={"bins": [], "median_prod": 0.0, "pct_below_low": 0.0, "pct_above_high": 0.0},
        hotspot={"projects": [], "top10": [], "gangs": []},
        trends={"low_bucket": [], "idle_windows": []},
        pareto={"top20_share": 0.0, "top10_share": 0.0},
        whatif={
            "low_bucket_count": 0.0,
            "low_bucket_output": 0.0,
            "total_gang_months": 0.0,
            "total_output": 0.0,
            "low_bucket_avg": 0.0,
        },
    )
    result = payload.to_dict()
    result["trend_df_low_bucket"] = []
    result["trend_df_idle_windows"] = []
    result["hotspot_top10_df"] = []
    result["pareto_metrics"] = result["pareto"]
    result["whatif_base_inputs"] = result["whatif"]
    return result
