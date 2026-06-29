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

SCOPE_ORDER = ("erection", "stringing", "foundation")
MAIL_SCOPE_ORDER = ("erection", "stringing")
DEFAULT_SCOPE = SCOPE_ORDER
SCOPE_ALIASES = {
    "all": SCOPE_ORDER,
    "both": MAIL_SCOPE_ORDER,
    "e": ("erection",),
    "erection": ("erection",),
    "s": ("stringing",),
    "stringing": ("stringing",),
    "f": ("foundation",),
    "foundation": ("foundation",),
}


@dataclass(frozen=True)
class MailArtifacts:
    html_path: Path
    subject: str
    as_of_date: pd.Timestamp
    erection: pd.DataFrame
    stringing: pd.DataFrame


def _normalize_scope(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_SCOPE

    selected: set[str] = set()
    for token in str(raw).split(","):
        key = token.strip().casefold()
        if not key:
            continue
        values = SCOPE_ALIASES.get(key)
        if values is None:
            valid = ", ".join(sorted(SCOPE_ALIASES))
            raise ValueError(f"Invalid scope '{token.strip()}'. Expected one or more of: {valid}.")
        selected.update(values)

    if not selected:
        raise ValueError("Scope must include at least one work type.")
    return tuple(scope for scope in SCOPE_ORDER if scope in selected)


def _scope_arg(raw: str) -> tuple[str, ...]:
    try:
        return _normalize_scope(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _scope_for_cli(scope: tuple[str, ...]) -> str:
    if scope == DEFAULT_SCOPE:
        return "all"
    if scope == MAIL_SCOPE_ORDER:
        return "both"
    return ",".join(scope)


def _mail_sections(scope: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in MAIL_SCOPE_ORDER if item in scope)


def _validate_mail_scope(scope: tuple[str, ...]) -> None:
    if not _mail_sections(scope):
        raise ValueError("--scope foundation is refresh-only; include erection or stringing to create a mail.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh DPR data and create the daily DPR productivity Outlook draft."
    )
    parser.add_argument(
        "--scope",
        type=_scope_arg,
        default=DEFAULT_SCOPE,
        help=(
            "Comma-separated work types to refresh/build: all, both, erection, stringing, foundation "
            "(aliases: e, s, f). Defaults to all."
        ),
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
    args = parser.parse_args(argv)
    try:
        _validate_mail_scope(args.scope)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _run_step(command: list[str]) -> None:
    print("[daily-mail] running:", " ".join(command))
    subprocess.run(command, cwd=BASE_DIR, check=True)


def refresh_outputs(*, skip_outlook_pull: bool = False, scope: tuple[str, ...] = DEFAULT_SCOPE) -> None:
    if not skip_outlook_pull:
        _run_step([sys.executable, "outlook_dpr_watcher.py"])
    _run_step(
        [
            sys.executable,
            "pipeline_runner.py",
            "--config",
            "pipeline_config.json",
            "--no-serve",
            "--compile-only",
            "--scope",
            _scope_for_cli(scope),
        ]
    )


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

    for col in ("plan_for_month", "progress_for_month", "quantity_primary", "cumulative_progress"):
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    snapshots = (
        work.groupby(["project_key", "report_date"], dropna=False)
        .agg(
            SnapshotPlan=("plan_for_month", lambda values: values.sum(min_count=1)),
            SnapshotActual=("progress_for_month", lambda values: values.sum(min_count=1)),
        )
        .reset_index()
    )
    prior_positive_plan = (
        snapshots[pd.to_numeric(snapshots["SnapshotPlan"], errors="coerce").gt(0)]
        .sort_values(["project_key", "report_date"])
        .drop_duplicates("project_key", keep="last")
        [["project_key", "SnapshotPlan"]]
        .rename(columns={"SnapshotPlan": "PriorPositivePlan"})
    )

    latest_dates = work.groupby("project_key")["report_date"].transform("max")
    work = work[work["report_date"] == latest_dates].copy()

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
    grouped = grouped.merge(prior_positive_plan, on="project_key", how="left")
    latest_plan = pd.to_numeric(grouped["Plan"], errors="coerce")
    latest_actual = pd.to_numeric(grouped["Actual"], errors="coerce")
    carry_forward = latest_plan.fillna(0).le(0) & latest_actual.fillna(0).gt(0)
    grouped.loc[carry_forward, "Plan"] = grouped.loc[carry_forward, "PriorPositivePlan"]
    grouped = grouped.drop(columns="PriorPositivePlan")
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


def _line_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _completed_status_mask(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.casefold()
    return text.eq("c") | text.str.contains(r"\b(?:complete|completed|done)\b", regex=True, na=False)


def _optional_series(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    series = frame.get(column)
    if isinstance(series, pd.Series):
        return series
    return pd.Series(default, index=frame.index)


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
    undated_completed = pd.DataFrame(columns=["project_key", "line_key", "tower_weight"])
    dated_line_counts = pd.DataFrame(columns=["project_key", "line_key", "dated_towers"])
    if not raw.empty:
        work = raw.copy()
        work["complete_date"] = pd.to_datetime(work.get("Complete Date"), errors="coerce").dt.normalize()
        work["project_key"] = work.get("Project Code", "").map(_compact_project)
        work["line_key"] = _optional_series(work, "Line Name").map(_line_key)
        work["tower_weight"] = pd.to_numeric(work.get("Tower Weight"), errors="coerce")
        if "Location No." in work.columns:
            valid_location = _valid_location_mask(work["Location No."])
        else:
            valid_location = pd.Series(True, index=work.index)
        valid_project_location = valid_location & work["project_key"].astype(bool)
        work = work[
            (work["complete_date"] >= month_start)
            & (work["complete_date"] <= min(as_of_date, month_end))
            & valid_project_location
        ]
        if not work.empty:
            actual = (
                work.groupby("project_key", dropna=False)
                .agg(**{"Total MT": ("tower_weight", "sum"), "Towers": ("tower_weight", "size")})
                .reset_index()
            )
            dated_line_counts = (
                work.groupby(["project_key", "line_key"], dropna=False)
                .size()
                .reset_index(name="dated_towers")
            )

        raw_work = raw.copy()
        raw_work["complete_date"] = pd.to_datetime(raw_work.get("Complete Date"), errors="coerce").dt.normalize()
        raw_work["project_key"] = raw_work.get("Project Code", "").map(_compact_project)
        raw_work["line_key"] = _optional_series(raw_work, "Line Name").map(_line_key)
        raw_work["tower_weight"] = pd.to_numeric(raw_work.get("Tower Weight"), errors="coerce")
        if "Location No." in raw_work.columns:
            raw_valid_location = _valid_location_mask(raw_work["Location No."])
        else:
            raw_valid_location = pd.Series(True, index=raw_work.index)
        raw_status = raw_work["Status"] if "Status" in raw_work.columns else pd.Series("", index=raw_work.index)
        undated_completed = raw_work[
            raw_work["complete_date"].isna()
            & raw_valid_location
            & raw_work["project_key"].astype(bool)
            & _completed_status_mask(raw_status)
            & raw_work["tower_weight"].notna()
        ][["project_key", "line_key", "tower_weight"]].copy()

    if not undated_completed.empty and not status.empty:
        status_raw = _read_parquet(PARQUET_DIR / "StringingSummary" / "StatusActivityFact.parquet")
        if not status_raw.empty:
            status_line = status_raw.copy()
            status_line["report_date"] = pd.to_datetime(status_line.get("report_date"), errors="coerce").dt.normalize()
            status_line["month"] = pd.to_datetime(status_line.get("month"), errors="coerce").dt.normalize()
            status_line["project_key"] = status_line.get("project_code", "").map(_compact_project)
            status_line["line_key"] = status_line.get("line_name", "").map(_line_key)
            status_line["progress_for_month"] = pd.to_numeric(status_line.get("progress_for_month"), errors="coerce")
            status_line = status_line[
                status_line["activity_group"].astype(str).str.casefold().eq("tower erection")
                & status_line["core_activity"].fillna(False).astype(bool)
                & status_line["project_key"].astype(bool)
                & status_line["line_key"].astype(bool)
                & status_line["month"].eq(month_start)
                & status_line["report_date"].le(min(as_of_date, month_end))
            ].copy()
            if not status_line.empty:
                latest_report = status_line.groupby(["project_key", "line_key"], dropna=False)["report_date"].transform("max")
                status_line = status_line[status_line["report_date"].eq(latest_report)].copy()
                status_line_actual = (
                    status_line.groupby(["project_key", "line_key"], dropna=False)["progress_for_month"]
                    .sum(min_count=1)
                    .reset_index(name="status_towers")
                )
                line_gap = status_line_actual.merge(
                    dated_line_counts,
                    on=["project_key", "line_key"],
                    how="left",
                )
                line_gap["dated_towers"] = pd.to_numeric(line_gap.get("dated_towers"), errors="coerce").fillna(0)
                line_gap["missing_towers"] = (
                    pd.to_numeric(line_gap["status_towers"], errors="coerce").fillna(0) - line_gap["dated_towers"]
                ).clip(lower=0).round().astype(int)

                fallback_rows: list[dict[str, object]] = []
                for _, gap_row in line_gap[line_gap["missing_towers"].gt(0)].iterrows():
                    candidates = undated_completed[
                        undated_completed["project_key"].eq(gap_row["project_key"])
                        & undated_completed["line_key"].eq(gap_row["line_key"])
                    ].head(int(gap_row["missing_towers"]))
                    if candidates.empty:
                        continue
                    fallback_rows.append(
                        {
                            "project_key": gap_row["project_key"],
                            "Total MT": float(candidates["tower_weight"].sum()),
                            "Towers": int(len(candidates.index)),
                        }
                    )
                if fallback_rows:
                    fallback_actual = pd.DataFrame(fallback_rows).groupby("project_key", as_index=False).sum()
                    if actual.empty:
                        actual = fallback_actual.copy()
                    else:
                        actual = (
                            pd.concat([actual, fallback_actual], ignore_index=True)
                            .groupby("project_key", as_index=False)
                            .agg(**{"Total MT": ("Total MT", "sum"), "Towers": ("Towers", "sum")})
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
    sections: tuple[str, ...] = MAIL_SCOPE_ORDER,
) -> str:
    period_label = month_start.strftime("%B %Y")
    as_of_label = as_of_date.strftime("%d-%b-%Y")
    section_labels = {"erection": "erection", "stringing": "stringing"}
    selected_labels = [section_labels[section] for section in sections]
    if len(selected_labels) == 1:
        work_label = selected_labels[0]
    else:
        work_label = " and ".join(selected_labels)

    body_parts = [
        "<p>Respected Sirs,</p>",
        (
            "<p>Please find below the monthly productivity summary for ongoing "
            f"{work_label} works for {period_label} month-to-date, as on {as_of_label}.</p>"
        ),
    ]
    if "erection" in sections:
        body_parts.extend(["<h3>Erection Productivity</h3>", _html_table(erection, "erection")])
    if "stringing" in sections:
        body_parts.extend(["<h3>Stringing Productivity</h3>", _html_table(stringing, "stringing")])
    body_parts.extend(["<p>Regards,</p>"])

    body_html = "\n".join(body_parts)
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
{body_html}
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
    scope = tuple(getattr(args, "scope", DEFAULT_SCOPE))
    _validate_mail_scope(scope)
    sections = _mail_sections(scope)

    if not args.skip_refresh:
        refresh_outputs(skip_outlook_pull=args.skip_outlook_pull, scope=scope)

    month_start, month_end = _month_window(args.month)
    as_of_date = _target_as_of_date(month_start, month_end, args.as_of_date)
    mapping = _load_pch_mapping()

    erection = (
        _build_erection_table(month_start, month_end, as_of_date, mapping)
        if "erection" in sections
        else pd.DataFrame()
    )
    stringing = (
        _build_stringing_table(month_start, month_end, as_of_date, mapping)
        if "stringing" in sections
        else pd.DataFrame()
    )

    subject_prefix = (
        "Daily DPR Productivity Summary"
        if sections == MAIL_SCOPE_ORDER
        else f"Daily DPR {sections[0].title()} Productivity Summary"
    )
    subject = f"{subject_prefix} - {month_start:%B %Y} MTD as on {as_of_date:%d-%b-%Y}"
    html_body = _build_html(
        erection,
        stringing,
        month_start=month_start,
        as_of_date=as_of_date,
        sections=sections,
    )
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
