"""Computation helpers for productivity metrics."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .config import AppConfig

LOGGER = logging.getLogger(__name__)
DEFAULT_LOSS_MAX_GAP_DAYS = AppConfig().loss_max_gap_days


def calc_idle_and_loss(
    group_df: pd.DataFrame,
    loss_max_gap_days: int = DEFAULT_LOSS_MAX_GAP_DAYS,
    baseline_mt_per_day: float | None = None,
) -> Tuple[int, float, float, float, float]:
    """Return idle days, baseline, lost MT, delivered MT, and potential MT for a gang."""

    dates = group_df["date"].dropna().drop_duplicates().sort_values()
    if len(dates) > 1:
        diff_days = dates.diff().dt.days.fillna(0).astype(int) - 1
        idle_days = int(np.minimum(diff_days[diff_days >= 1], loss_max_gap_days).sum())
    else:
        idle_days = 0

    baseline: float | None = None
    if baseline_mt_per_day is not None and not pd.isna(baseline_mt_per_day):
        baseline = float(baseline_mt_per_day)
    elif len(group_df):
        mean_val = group_df["daily_prod_mt"].mean()
        if not pd.isna(mean_val):
            baseline = float(mean_val)
    if baseline is None or baseline <= 0.0:
        baseline = 5.0

    loss_mt = baseline * idle_days
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
    data: pd.DataFrame, loss_max_gap_days: int = DEFAULT_LOSS_MAX_GAP_DAYS
) -> pd.DataFrame:
    """Return a DataFrame describing idle intervals per gang."""

    rows: list[dict[str, object]] = []
    for gang_name, gang_df in data.groupby("gang_name"):
        dates = gang_df["date"].dropna().drop_duplicates().sort_values().to_list()
        if len(dates) < 2:
            continue
        for index in range(1, len(dates)):
            gap = (dates[index] - dates[index - 1]).days - 1
            if gap >= 1:
                interval_start = (dates[index - 1] + pd.Timedelta(days=1)).normalize()
                interval_end = (dates[index] - pd.Timedelta(days=1)).normalize()
                rows.append(
                    {
                        "gang_name": gang_name,
                        "interval_start": interval_start,
                        "interval_end": interval_end,
                        "raw_gap_days": gap,
                        "idle_days_capped": int(min(gap, loss_max_gap_days)),
                    }
                )
    LOGGER.debug("Identified %d idle intervals", len(rows))
    return pd.DataFrame(rows)
