"""Excel workbook export helpers."""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Sequence

import pandas as pd

from .config import AppConfig
from .metrics import calc_idle_and_loss, compute_idle_intervals_per_gang

LOGGER = logging.getLogger(__name__)


def _sanitize_sheet_name(value: str) -> str:
    """Return a value safe for use as an Excel sheet name."""

    sanitized = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(value))
    return sanitized[:31]


def make_trace_workbook_bytes(
    scoped_data: pd.DataFrame,
    months_ts: Sequence[pd.Timestamp] | None,
    projects: Sequence[str] | None,
    gangs: Sequence[str] | None,
    bench: float,
    *,
    gang_for_sheet: str | None = None,
    config: AppConfig | None = None,
    project_info: pd.DataFrame | None = None,
) -> bytes:
    """Build the Excel export with summary, idle intervals, and daily detail."""

    active_config = config or AppConfig()
    LOGGER.info("Building trace workbook (rows=%d)", len(scoped_data))

    summary_rows: list[dict[str, object]] = []
    for gang_name, gang_df in scoped_data.groupby("gang_name"):
        idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
            gang_df, loss_max_gap_days=active_config.loss_max_gap_days
        )
        summary_rows.append(
            {
                "gang_name": gang_name,
                "delivered_mt": delivered,
                "lost_mt": loss_mt,
                "potential_mt": potential,
                "baseline_mt_per_day": baseline,
                "idle_days_capped": idle,
                "first_date": gang_df["date"].min(),
                "last_date": gang_df["date"].max(),
                "active_days": gang_df["date"].nunique(),
            }
        )
    per_gang_summary = (
        pd.DataFrame(summary_rows).sort_values("potential_mt", ascending=False)
        if summary_rows
        else pd.DataFrame(
            columns=[
                "gang_name",
                "delivered_mt",
                "lost_mt",
                "potential_mt",
                "baseline_mt_per_day",
                "idle_days_capped",
                "first_date",
                "last_date",
                "active_days",
            ]
        )
    )

    idle_df = compute_idle_intervals_per_gang(
        scoped_data, loss_max_gap_days=active_config.loss_max_gap_days
    )
    daily_df = (
        scoped_data.sort_values(["gang_name", "date"])
        [["date", "gang_name", "project_name", "daily_prod_mt"]]
        .copy()
    )
    project_month = (
        scoped_data.groupby(["project_name", "month"])["daily_prod_mt"].mean().reset_index()
    )

    context_df = pd.DataFrame(
        [
            {
                "projects": ", ".join(projects or []) or "(all)",
                "gangs": ", ".join(gangs or []) or "(all)",
                "months": ", ".join(
                    [timestamp.strftime("%Y-%m") for timestamp in (months_ts or [])]
                )
                or "(all / overall)",
                "benchmark": bench,
                "loss_cap_days": active_config.loss_max_gap_days,
            }
        ]
    )

    assumptions = pd.DataFrame(
        {
            "Notes": [
                f"Loss cap per gap: {active_config.loss_max_gap_days} days.",
                "Efficiency = delivered / (delivered + lost). Lost = baseline * capped idle days.",
                "Idle interval = gaps between observed work dates; dates inferred from current filtered scope.",
                "All numbers reflect current dashboard filters (project, period, and gang if applied).",
            ]
        }
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        per_gang_summary.to_excel(writer, "PerGangSummary", index=False)
        idle_df.to_excel(writer, "IdleIntervals", index=False)
        daily_df.to_excel(writer, "DailyProductivity", index=False)
        project_month.to_excel(writer, "ProjectsMonthly", index=False)
        context_df.to_excel(writer, "SelectionContext", index=False)
        assumptions.to_excel(writer, "Assumptions", index=False)

        if gang_for_sheet:
            selected = scoped_data[scoped_data["gang_name"] == gang_for_sheet]
            if not selected.empty:
                single_idle = compute_idle_intervals_per_gang(
                    selected, loss_max_gap_days=active_config.loss_max_gap_days
                )
                single_idle.to_excel(
                    writer,
                    _sanitize_sheet_name(f"Idle_{gang_for_sheet}"),
                    index=False,
                )
                (
                    selected.sort_values("date")[["date", "project_name", "daily_prod_mt"]]
                    .assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d"))
                    .to_excel(writer, _sanitize_sheet_name(f"Daily_{gang_for_sheet}"), index=False)
                )

                idle, baseline, loss_mt, delivered, potential = calc_idle_and_loss(
                    selected, loss_max_gap_days=active_config.loss_max_gap_days
                )
                efficiency = (delivered / potential * 100) if potential > 0 else 0.0
                pd.DataFrame(
                    [
                        {
                            "gang_name": gang_for_sheet,
                            "delivered_mt": delivered,
                            "lost_mt": loss_mt,
                            "potential_mt": potential,
                            "efficiency_%": efficiency,
                            "baseline_mt_per_day": baseline,
                            "idle_days_capped": idle,
                        }
                    ]
                ).to_excel(
                    writer,
                    _sanitize_sheet_name(f"Summary_{gang_for_sheet}"),
                    index=False,
                )
        
        
        # Optional: include ProjectDetails if exactly one project is selected
        if projects and len(projects) == 1 and project_info is not None and not project_info.empty:
            pname = str(projects[0]).strip()
            # project_name -> project_code from scoped_data (or fall back to global if needed)
            name_to_code = (
                scoped_data.dropna(subset=["project_name", "project_code"])
                           .drop_duplicates(subset=["project_name"])
                           .set_index("project_name")["project_code"]
                           .to_dict()
            )
            pcode = name_to_code.get(pname)
            if pcode:
                row = project_info[project_info["project_code"] == pcode]
                if not row.empty:
                    r = row.iloc[0]
                    pd.DataFrame([{
                        "Project Code": pcode,
                        "Project Name": pname,
                        "Client Name": r.get("client_name"),
                        "NOA Start Date": r.get("noa_start"),
                        "LOA End Date": r.get("loa_end"),
                        "Project Manager": r.get("project_mgr"),
                        "Regional Manager": r.get("regional_mgr"),
                        "Planning Engineer": r.get("planning_eng"),
                        "PCH": r.get("pch"),
                        "Section Incharge": r.get("section_inch"),
                        "Supervisor": r.get("supervisor"),
                    }]).to_excel(writer, "ProjectDetails_Selected", index=False)


    buffer.seek(0)
    return buffer.getvalue()
