#!/usr/bin/env python3
"""Prepare the daily DPR productivity mail draft from refreshed DPR outputs."""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "Raw Data"
PARQUET_DIR = BASE_DIR / "Parquets"
PRODUCTIVITY_DIR = BASE_DIR / "Productivity Summaries"

PCH_ORDER = {
    "Mr. Arun Felbin": 1,
    "Mr. NK Gupta": 2,
    "Mr. Nabajit Baruah": 3,
}


@dataclass(frozen=True)
class MailArtifacts:
    html_path: Path
    subject: str
    as_of_date: pd.Timestamp
    erection: pd.DataFrame
    stringing: pd.DataFrame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh DPR data and create the daily DPR productivity Outlook draft."
    )
    parser.add_argument(
        "--month",
        help="Target month in YYYY-MM format. Defaults to the current calendar month.",
    )
    parser.add_argument(
        "--as-of-date",
        help="Optional YYYY-MM-DD cutoff. Defaults to latest DPR status report date in the target month.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Use existing parquet/workbook outputs without pulling Outlook DPRs or rerunning the pipeline.",
    )
    parser.add_argument(
        "--skip-outlook-pull",
        action="store_true",
        help="Rerun the pipeline from existing Raw Data/DPRs files without pulling Outlook DPR attachments.",
    )
    parser.add_argument(
        "--no-draft",
        action="store_true",
        help="Write the HTML mail body but do not create an Outlook draft.",
    )
    parser.add_argument(
        "--to",
        default="",
        help="Optional semicolon-separated To recipients for the draft.",
    )
    parser.add_argument(
        "--cc",
        default="",
        help="Optional semicolon-separated CC recipients for the draft.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Optional output path for the generated HTML body.",
    )
    return parser.parse_args()


def _run_step(command: list[str]) -> None:
    print("[daily-mail] running:", " ".join(command))
    subprocess.run(command, cwd=BASE_DIR, check=True)


def refresh_outputs(*, skip_outlook_pull: bool = False) -> None:
    if not skip_outlook_pull:
        _run_step([sys.executable, "outlook_dpr_watcher.py"])
    _run_step([sys.executable, "pipeline_runner.py", "--config", "pipeline_config.json", "--no-serve"])


def _compact_project(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _display_project(value: object) -> str:
    compact = _compact_project(value)
    match = re.match(r"^([A-Z]+)(\d+.*)$", compact)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return str(value or "").strip()


def _month_window(month: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if month:
        start = pd.Timestamp(f"{month}-01").normalize()
    else:
        start = pd.Timestamp.today().normalize().to_period("M").to_timestamp()
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    return start, end


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_pch_mapping() -> pd.DataFrame:
    path = RAW_DIR / "Projects and PCH.xlsx"
    if not path.exists():
        return pd.DataFrame(columns=["project_key", "PCH", "Project"])

    frame = pd.read_excel(path)
    if frame.empty or "Project" not in frame.columns or "PCH" not in frame.columns:
        return pd.DataFrame(columns=["project_key", "PCH", "Project"])

    out = frame[["Project", "PCH"]].copy()
    out["project_key"] = out["Project"].map(_compact_project)
    out["Project"] = out["Project"].map(_display_project)
    out["PCH"] = out["PCH"].fillna("Unassigned").astype(str).str.strip()
    out = out[out["project_key"].astype(bool)]
    return out.drop_duplicates("project_key", keep="first").reset_index(drop=True)


def _target_as_of_date(month_start: pd.Timestamp, month_end: pd.Timestamp, override: str | None) -> pd.Timestamp:
    if override:
        return min(pd.Timestamp(override).normalize(), month_end)

    candidates: list[pd.Timestamp] = []
    status = _read_parquet(PARQUET_DIR / "StringingSummary" / "StatusActivityFact.parquet")
    if not status.empty and "report_date" in status.columns:
        dates = pd.to_datetime(status["report_date"], errors="coerce").dropna().dt.normalize()
        scoped = dates[(dates >= month_start) & (dates <= month_end)]
        if not scoped.empty:
            candidates.append(scoped.max())

    raw = _read_parquet(PARQUET_DIR / "Erection" / "RawData.parquet")
    if not raw.empty and "Complete Date" in raw.columns:
        dates = pd.to_datetime(raw["Complete Date"], errors="coerce").dropna().dt.normalize()
        scoped = dates[(dates >= month_start) & (dates <= month_end)]
        if not scoped.empty:
            candidates.append(scoped.max())

    if candidates:
        return min(max(candidates), month_end)
    return min(pd.Timestamp.today().normalize(), month_end)


def _activity_status(
    activity_group: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    status = _read_parquet(PARQUET_DIR / "StringingSummary" / "StatusActivityFact.parquet")
    columns = [
        "project_key",
        "Project",
        "Plan",
        "Actual",
        "Scope",
        "Completed",
        "Report Date",
    ]
    if status.empty:
        return pd.DataFrame(columns=columns)

    work = status.copy()
    work["month"] = pd.to_datetime(work.get("month"), errors="coerce").dt.normalize()
    work["report_date"] = pd.to_datetime(work.get("report_date"), errors="coerce").dt.normalize()
    work["project_key"] = work.get("project_code", "").map(_compact_project)
    work["Project"] = work.get("project_code", "").map(_display_project)
    work["activity_group"] = work.get("activity_group", "").fillna("").astype(str).str.strip()
    if "core_activity" in work.columns:
        work = work[work["core_activity"].fillna(False).astype(bool)]
    work = work[
        (work["activity_group"].str.casefold() == activity_group.casefold())
        & (work["month"] == month_start)
        & (work["report_date"].notna())
        & (work["report_date"] <= as_of_date)
        & work["project_key"].astype(bool)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    latest_dates = work.groupby("project_key")["report_date"].transform("max")
    work = work[work["report_date"] == latest_dates].copy()

    for col in ("plan_for_month", "progress_for_month", "quantity_primary", "cumulative_progress"):
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    grouped = (
        work.groupby("project_key", dropna=False)
        .agg(
            Project=("Project", "first"),
            Plan=("plan_for_month", lambda values: values.sum(min_count=1)),
            Actual=("progress_for_month", lambda values: values.sum(min_count=1)),
            Scope=("quantity_primary", lambda values: values.sum(min_count=1)),
            Completed=("cumulative_progress", lambda values: values.sum(min_count=1)),
            **{"Report Date": ("report_date", "max")},
        )
        .reset_index()
    )
    return grouped.reindex(columns=columns)


def _merge_identity(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame = pd.DataFrame(columns=["project_key", "Project"])
    out = frame.merge(mapping[["project_key", "PCH"]], on="project_key", how="left")
    out["PCH"] = out["PCH"].fillna("Unassigned")
    out["Project"] = out["Project"].fillna(out["project_key"].map(_display_project))
    return out


def _sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["_pch_order"] = out["PCH"].map(PCH_ORDER).fillna(99)
    out["_project_order"] = out["Project"].astype(str).map(_compact_project)
    return out.sort_values(["_pch_order", "PCH", "_project_order"]).drop(
        columns=["_pch_order", "_project_order"]
    )


def _valid_location_mask(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.notna() & text.ne("") & ~text.str.casefold().isin({"nan", "none", "nat"})


def _build_erection_table(
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    as_of_date: pd.Timestamp,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    raw = _read_parquet(PARQUET_DIR / "Erection" / "RawData.parquet")
    daily = _read_parquet(PARQUET_DIR / "Erection" / "ProdDailyExpandedSingles.parquet")
    status = _activity_status("Tower Erection", month_start, month_end, as_of_date)

    actual = pd.DataFrame(columns=["project_key", "Total MT", "Towers"])
    if not raw.empty:
        work = raw.copy()
        work["complete_date"] = pd.to_datetime(work.get("Complete Date"), errors="coerce").dt.normalize()
        work["project_key"] = work.get("Project Code", "").map(_compact_project)
        work["tower_weight"] = pd.to_numeric(work.get("Tower Weight"), errors="coerce")
        if "Location No." in work.columns:
            valid_location = _valid_location_mask(work["Location No."])
        else:
            valid_location = pd.Series(True, index=work.index)
        work = work[
            (work["complete_date"] >= month_start)
            & (work["complete_date"] <= min(as_of_date, month_end))
            & valid_location
            & work["project_key"].astype(bool)
        ]
        if not work.empty:
            actual = (
                work.groupby("project_key", dropna=False)
                .agg(**{"Total MT": ("tower_weight", "sum"), "Towers": ("tower_weight", "size")})
                .reset_index()
            )

    productivity = pd.DataFrame(columns=["project_key", "Productivity", "total_km"])
    if not daily.empty:
        work = daily.copy()
        work["work_date"] = pd.to_datetime(work.get("Work Date"), errors="coerce").dt.normalize()
        work["project_key"] = work.get("Project Code", "").map(_compact_project)
        work["Productivity"] = pd.to_numeric(work.get("Productivity"), errors="coerce")
        work = work[
            (work["work_date"] >= month_start)
            & (work["work_date"] <= min(as_of_date, month_end))
            & work["project_key"].astype(bool)
        ]
        if not work.empty:
            productivity = (
                work.groupby("project_key", dropna=False)["Productivity"]
                .mean()
                .reset_index()
            )

    projects = pd.DataFrame({"project_key": sorted(set(status["project_key"]) | set(actual["project_key"]))})
    table = projects.merge(status[["project_key", "Project", "Plan", "Actual"]], on="project_key", how="left")
    table = table.merge(actual, on="project_key", how="left")
    table = table.merge(productivity, on="project_key", how="left")
    table = _merge_identity(table, mapping)
    raw_towers = pd.to_numeric(table.get("Towers"), errors="coerce")
    status_towers = pd.to_numeric(table.get("Actual"), errors="coerce")
    table["Towers"] = raw_towers.where(raw_towers.gt(0), status_towers)
    planned = pd.to_numeric(table.get("Plan"), errors="coerce").notna()
    table.loc[planned & pd.to_numeric(table["Towers"], errors="coerce").isna(), "Towers"] = 0
    table["Avg Tower Wt (MT)"] = pd.to_numeric(table.get("Total MT"), errors="coerce") / raw_towers.where(raw_towers.gt(0))
    table = table.rename(columns={"Plan": "Plan (Nos.)", "Towers": "Actual Towers (Nos.)"})
    ordered = [
        "PCH",
        "Project",
        "Plan (Nos.)",
        "Actual Towers (Nos.)",
        "Total MT",
        "Avg Tower Wt (MT)",
        "Productivity",
    ]
    table = table.reindex(columns=ordered)
    keep = (
        pd.to_numeric(table["Plan (Nos.)"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(table["Actual Towers (Nos.)"], errors="coerce").fillna(0).gt(0)
    )
    return _round_numeric(_sort_rows(table[keep]))


def _build_stringing_table(
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    as_of_date: pd.Timestamp,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    status = _activity_status("Stringing", month_start, month_end, as_of_date)
    daily = _read_parquet(PARQUET_DIR / "Stringing" / "StringingDaily.parquet")
    readiness = _read_parquet(PARQUET_DIR / "StretchReadiness" / "Summary.parquet")

    productivity = pd.DataFrame(columns=["project_key", "Productivity"])
    if not daily.empty:
        work = daily.copy()
        work["date"] = pd.to_datetime(work.get("date"), errors="coerce").dt.normalize()
        work["project_key"] = work.get("project", "").map(_compact_project)
        work["daily_km"] = pd.to_numeric(work.get("daily_km"), errors="coerce").fillna(0.0)
        work = work[
            (work["date"] >= month_start)
            & (work["date"] <= min(as_of_date, month_end))
            & work["project_key"].astype(bool)
        ]
        if not work.empty:
            grouped = work.groupby("project_key", dropna=False).agg(
                total_km=("daily_km", "sum"),
                active_days=("date", "nunique"),
            )
            grouped["Productivity"] = grouped["total_km"] / grouped["active_days"].replace({0: pd.NA}) * 30.0
            productivity = grouped.reset_index()[["project_key", "Productivity", "total_km"]]

    ready = pd.DataFrame(columns=["project_key", "Stretch Ready (KM)"])
    if not readiness.empty:
        work = readiness.copy()
        work["report_date"] = pd.to_datetime(work.get("report_date"), errors="coerce").dt.normalize()
        work["project_key"] = work.get("project_code", "").map(_compact_project)
        work["ready_km"] = pd.to_numeric(work.get("ready_km"), errors="coerce")
        work = work[
            (work["report_date"] >= month_start)
            & (work["report_date"] <= min(as_of_date, month_end))
            & work["project_key"].astype(bool)
        ]
        if not work.empty:
            latest_dates = work.groupby("project_key")["report_date"].transform("max")
            work = work[work["report_date"] == latest_dates]
            ready = (
                work.groupby("project_key", dropna=False)["ready_km"]
                .sum()
                .reset_index()
                .rename(columns={"ready_km": "Stretch Ready (KM)"})
            )

    projects = pd.DataFrame({"project_key": sorted(set(status["project_key"]) | set(productivity["project_key"]))})
    table = projects.merge(
        status[["project_key", "Project", "Plan", "Actual", "Scope", "Completed"]],
        on="project_key",
        how="left",
    )
    table = table.merge(productivity, on="project_key", how="left")
    table = table.merge(ready, on="project_key", how="left")
    table = _merge_identity(table, mapping)
    table = table.rename(
        columns={
            "Plan": "Plan (KM)",
            "Actual": "Actual Achieved (KM)",
            "Scope": "Scope (KM)",
            "Completed": "Stringing Completed (KM)",
        }
    )
    if "total_km" in table.columns:
        status_actual = pd.to_numeric(table.get("Actual Achieved (KM)"), errors="coerce")
        daily_actual = pd.to_numeric(table.get("total_km"), errors="coerce")
        table["Actual Achieved (KM)"] = status_actual.where(status_actual.notna(), daily_actual)
    planned = pd.to_numeric(table.get("Plan (KM)"), errors="coerce").notna()
    table.loc[
        planned & pd.to_numeric(table["Actual Achieved (KM)"], errors="coerce").isna(),
        "Actual Achieved (KM)",
    ] = 0.0
    ordered = [
        "PCH",
        "Project",
        "Plan (KM)",
        "Actual Achieved (KM)",
        "Productivity",
        "Scope (KM)",
        "Stringing Completed (KM)",
        "Stretch Ready (KM)",
    ]
    table = table.reindex(columns=ordered)
    keep = (
        pd.to_numeric(table["Plan (KM)"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(table["Actual Achieved (KM)"], errors="coerce").fillna(0).abs().gt(0.005)
    )
    return _round_numeric(_sort_rows(table[keep]))


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column == "Actual Towers (Nos.)":
            out[column] = pd.to_numeric(out[column], errors="coerce").round(0).astype("Int64")
        elif column not in {"PCH", "Project"}:
            values = pd.to_numeric(out[column], errors="coerce")
            values = values.mask(values.abs().lt(0.005), 0.0)
            out[column] = values.round(2)
    return out


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int,)) or str(type(value)).endswith("Integer'>"):
        return f"{int(value):,}"
    try:
        number = float(value)
    except Exception:
        return html.escape(str(value))
    if abs(number - round(number)) < 0.005:
        return f"{number:,.0f}"
    return f"{number:,.2f}"


def _totals_row(frame: pd.DataFrame, kind: str) -> dict[str, object]:
    row = {column: "" for column in frame.columns}
    row["PCH"] = "Portfolio Total"
    row["Project"] = "-"
    if kind == "erection":
        total_mt = pd.to_numeric(frame.get("Total MT"), errors="coerce").sum()
        towers = pd.to_numeric(frame.get("Actual Towers (Nos.)"), errors="coerce").sum()
        plan = pd.to_numeric(frame.get("Plan (Nos.)"), errors="coerce").sum()
        row["Plan (Nos.)"] = plan
        row["Actual Towers (Nos.)"] = towers
        row["Total MT"] = total_mt
        row["Avg Tower Wt (MT)"] = total_mt / towers if towers else pd.NA
        prod = pd.to_numeric(frame.get("Productivity"), errors="coerce").dropna()
        row["Productivity"] = prod.mean() if not prod.empty else pd.NA
    else:
        for column in ["Plan (KM)", "Actual Achieved (KM)", "Scope (KM)", "Stringing Completed (KM)", "Stretch Ready (KM)"]:
            row[column] = pd.to_numeric(frame.get(column), errors="coerce").sum()
        prod = pd.to_numeric(frame.get("Productivity"), errors="coerce").dropna()
        row["Productivity"] = prod.mean() if not prod.empty else pd.NA
    return row


def _html_table(frame: pd.DataFrame, kind: str) -> str:
    if frame.empty:
        return "<p>No rows found for this month.</p>"
    rows = frame.to_dict("records") + [_totals_row(frame, kind)]
    columns = list(frame.columns)
    parts = [
        "<table>",
        "<thead><tr>",
        *[f"<th>{html.escape(column)}</th>" for column in columns],
        "</tr></thead>",
        "<tbody>",
    ]
    last_pch = None
    for row in rows:
        is_total = row.get("PCH") == "Portfolio Total"
        parts.append("<tr class='total'>" if is_total else "<tr>")
        for column in columns:
            value = row.get(column)
            if column == "PCH" and value == last_pch and not is_total:
                text = ""
            elif column in {"PCH", "Project"}:
                text = html.escape(str(value if not pd.isna(value) else ""))
            else:
                text = _format_number(value)
            parts.append(f"<td>{text}</td>")
        if row.get("PCH") not in {"", "Portfolio Total"}:
            last_pch = row.get("PCH")
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "\n".join(parts)


def _build_html(
    erection: pd.DataFrame,
    stringing: pd.DataFrame,
    *,
    month_start: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> str:
    period_label = month_start.strftime("%B %Y")
    as_of_label = as_of_date.strftime("%d-%b-%Y")
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #1f1f1f; }}
p {{ margin: 0 0 10px 0; }}
h3 {{ margin: 16px 0 8px 0; font-size: 13pt; }}
table {{ border-collapse: collapse; margin: 6px 0 16px 0; font-size: 10pt; }}
th, td {{ border: 1px solid #808080; padding: 4px 6px; text-align: right; white-space: nowrap; }}
th {{ background: #d9eaf7; font-weight: 700; text-align: center; }}
td:first-child, td:nth-child(2) {{ text-align: left; }}
tr.total td {{ font-weight: 700; background: #f2f2f2; }}
</style>
</head>
<body>
<p>Respected Sirs,</p>
<p>Please find below the monthly productivity summary for ongoing erection and stringing works for {period_label} month-to-date, as on {as_of_label}.</p>
<h3>Erection Productivity</h3>
{_html_table(erection, "erection")}
<h3>Stringing Productivity</h3>
{_html_table(stringing, "stringing")}
<p>Regards,</p>
</body>
</html>
"""


def _create_outlook_draft(subject: str, html_body: str, *, to: str = "", cc: str = "") -> None:
    try:
        import win32com.client  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on Windows Outlook installation
        raise RuntimeError("pywin32/Outlook COM is not available in this environment.") from exc

    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)
    mail.Subject = subject
    mail.HTMLBody = html_body
    if to:
        mail.To = to
    if cc:
        mail.CC = cc
    mail.Save()
    mail.Display()


def prepare_mail(args: argparse.Namespace) -> MailArtifacts:
    if not args.skip_refresh:
        refresh_outputs(skip_outlook_pull=args.skip_outlook_pull)

    month_start, month_end = _month_window(args.month)
    as_of_date = _target_as_of_date(month_start, month_end, args.as_of_date)
    mapping = _load_pch_mapping()

    erection = _build_erection_table(month_start, month_end, as_of_date, mapping)
    stringing = _build_stringing_table(month_start, month_end, as_of_date, mapping)

    subject = f"Daily DPR Productivity Summary - {month_start:%B %Y} MTD as on {as_of_date:%d-%b-%Y}"
    html_body = _build_html(erection, stringing, month_start=month_start, as_of_date=as_of_date)
    output_path = args.output_html or PRODUCTIVITY_DIR / f"Daily_DPR_Mail_{as_of_date:%Y-%m-%d}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_body, encoding="utf-8")

    if not args.no_draft:
        _create_outlook_draft(subject, html_body, to=args.to, cc=args.cc)

    return MailArtifacts(
        html_path=output_path,
        subject=subject,
        as_of_date=as_of_date,
        erection=erection,
        stringing=stringing,
    )


def main() -> int:
    args = _parse_args()
    artifacts = prepare_mail(args)
    print(f"[daily-mail] Subject: {artifacts.subject}")
    print(f"[daily-mail] HTML body: {artifacts.html_path}")
    print(f"[daily-mail] Erection rows: {len(artifacts.erection)}")
    print(f"[daily-mail] Stringing rows: {len(artifacts.stringing)}")
    if args.no_draft:
        print("[daily-mail] Outlook draft creation skipped.")
    else:
        print("[daily-mail] Outlook draft created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
