"""Filtering helpers and date utilities."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


DateRange = Tuple[pd.Timestamp, pd.Timestamp]


def apply_filters(
    data: pd.DataFrame,
    projects: Sequence[str],
    months: Sequence[pd.Timestamp],
    gangs: Sequence[str],
) -> pd.DataFrame:
    """Filter *data* by the provided selections."""

    filtered = data.copy()
    if projects:
        filtered = filtered[filtered["project_name"].isin(projects)]
    if months:
        filtered = filtered[filtered["month"].isin(months)]
    if gangs:
        filtered = filtered[filtered["gang_name"].isin(gangs)]
    LOGGER.debug(
        "Filtered data down to %d rows (projects=%s, months=%s, gangs=%s)",
        len(filtered),
        list(projects),
        [month.strftime("%Y-%m") for month in months],
        list(gangs),
    )
    return filtered


def get_quick_date_options() -> Dict[str, DateRange]:
    """Return quick date range options keyed by code."""

    today = pd.Timestamp.today().normalize()
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    last_three_months = today - pd.DateOffset(months=3)
    last_quarter_start = (today.to_period("Q") - 1).start_time
    last_six_months = today - pd.DateOffset(months=6)

    options = {
        "3M": (last_three_months, today),
        "Q": (last_quarter_start, today),
        "6M": (last_six_months, today),
        "YTD": (start_of_year, today),
    }
    LOGGER.debug("Quick date options computed: %s", options)
    return options


def resolve_months(months: Sequence[str] | None, quick_range: str | None) -> List[pd.Timestamp]:
    """Resolve month strings or quick range code into Timestamp values."""

    if quick_range:
        start, end = get_quick_date_options()[quick_range]
        return pd.period_range(start=start, end=end, freq="M").to_timestamp().tolist()
    if months:
        return [pd.Period(month, "M").to_timestamp() for month in months]
    return []
