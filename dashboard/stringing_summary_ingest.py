"""Unified Stringing Summary compiler for Status, Stretch Readiness, and Manpower contracts."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import duckdb
import pandas as pd

from .project_identity import build_project_display, build_project_scope_key, normalize_line_name
from .stringing import normalize_stringing_columns

PARQUET_SUFFIXES: tuple[str, ...] = (".parquet", ".parq", ".pq")

STRINGING_SUMMARY_SHEETS: tuple[str, ...] = (
    "StatusActivityFact",
    "StatusSnapshotProject",
    "StatusSnapshotOverall",
    "StretchSectionFact",
    "ManpowerProductivityFact",
    "Coverage",
    "Diagnostics",
    "Issues",
)

_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_PROJECT_RE = re.compile(r"\b(TA|TB)\s*[-_ ]?\s*(\d{3,4})\b", flags=re.IGNORECASE)


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null", "nat"}:
        return ""
    return text


def _norm_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _safe_text(value).lower())


def _normalize_project_code(value: object) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    match = _PROJECT_RE.search(text)
    if not match:
        return text
    return f"{match.group(1).upper()} {match.group(2)}"


def _normalize_project_display(code: str, display: str, line_name: str) -> str:
    visible = _safe_text(display)
    code_text = _safe_text(code)
    if not visible:
        visible = code_text
    return build_project_display(code_text, normalize_line_name(line_name), visible)


def _extract_date_from_text(value: object) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    match = _DATE_RE.search(text)
    return match.group(1) if match else ""


def _parse_report_date(series: pd.Series | None, fallback: pd.Series | None = None) -> pd.Series:
    if series is None:
        parsed = pd.Series(pd.NaT, dtype="datetime64[ns]")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    if fallback is not None:
        fill = fallback.fillna("").astype(str).map(_extract_date_from_text)
        fallback_parsed = pd.to_datetime(fill, errors="coerce")
        parsed = parsed.where(parsed.notna(), fallback_parsed)
    return pd.to_datetime(parsed, errors="coerce").dt.normalize()


def _resolve_sibling_root(base_dir: Path, sibling_name: str) -> Path:
    root = base_dir.resolve()
    if root.is_file():
        root = root.parent
    sibling_key = sibling_name.strip().lower()
    if root.name.lower() == sibling_key:
        return root
    if root.name.lower() == "parquets":
        return root / sibling_name
    if root.name.lower() == "erection" and root.parent.name.lower() == "parquets":
        return root.parent / sibling_name
    if root.parent.name.lower() == "parquets":
        return root.parent / sibling_name
    return root / sibling_name


def _is_probably_parquet_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in PARQUET_SUFFIXES:
        return False
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as fh:
            header = fh.read(4)
            fh.seek(-4, 2)
            trailer = fh.read(4)
        return header == b"PAR1" and trailer == b"PAR1"
    except Exception:
        return False


def _find_parquet_source(root: Path, table: str) -> str | None:
    root = root.resolve()
    if root.is_file() and root.suffix.lower() in PARQUET_SUFFIXES and root.stem.lower() == table.lower():
        return str(root)
    if not root.exists():
        return None
    candidates = [table, table.lower(), table.upper(), table.replace(" ", ""), table.replace("_", "")]
    if root.is_file():
        root = root.parent
    for stem in candidates:
        for suffix in PARQUET_SUFFIXES:
            candidate = root / f"{stem}{suffix}"
            if candidate.exists() and _is_probably_parquet_file(candidate):
                return str(candidate)
    for stem in candidates:
        directory = root / stem
        if directory.is_dir():
            for suffix in PARQUET_SUFFIXES:
                files = list(directory.glob(f"*{suffix}"))
                if files and all(_is_probably_parquet_file(file) for file in files):
                    return str(directory / f"*{suffix}")
    for suffix in PARQUET_SUFFIXES:
        match = next((p for p in root.glob(f"**/*{suffix}") if p.stem.lower() == table.lower() and _is_probably_parquet_file(p)), None)
        if match is not None:
            return str(match)
    return None


def _read_parquet(source: str) -> pd.DataFrame:
    with duckdb.connect(database=":memory:") as con:
        return con.execute("SELECT * FROM read_parquet(?)", [source]).df()


def _load_table(root: Path, workbook_name: str, sheet_name: str) -> pd.DataFrame:
    parquet = _find_parquet_source(root, sheet_name)
    if parquet:
        try:
            return _read_parquet(parquet)
        except Exception:
            pass
    workbook_path = root / workbook_name
    if workbook_path.exists():
        try:
            with pd.ExcelFile(workbook_path) as xl:
                if sheet_name in xl.sheet_names:
                    return xl.parse(sheet_name=sheet_name)
        except Exception:
            pass
    return pd.DataFrame()


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    for column in columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _build_status_activity_fact(progress_raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(progress_raw, pd.DataFrame) or progress_raw.empty:
        return pd.DataFrame(
            columns=[
                "project_code",
                "project_display",
                "project_scope_key",
                "line_name",
                "report_date",
                "month",
                "section_label",
                "activity_raw",
                "activity_norm",
                "activity_group",
                "core_activity",
                "quantity_primary",
                "cumulative_last_month",
                "plan_for_month",
                "progress_for_month",
                "today_progress",
                "cumulative_progress",
                "balance_progress",
                "gangs_working",
                "remarks",
                "source_file",
                "source_sheet",
                "configured_sheet",
                "template_sheet",
            ]
        )

    work = progress_raw.copy()
    for column in (
        "project_code",
        "project_display",
        "project_scope_key",
        "line_name",
        "section_label",
        "activity_raw",
        "activity_norm",
        "source_file",
        "source_sheet",
        "configured_sheet",
        "template_sheet",
    ):
        if column not in work.columns:
            work[column] = ""

    work["project_code"] = work["project_code"].map(_normalize_project_code)
    work["line_name"] = work["line_name"].map(normalize_line_name)
    work["project_display"] = [
        _normalize_project_display(code, display, line)
        for code, display, line in zip(work["project_code"], work["project_display"], work["line_name"])
    ]
    work["project_scope_key"] = [
        _safe_text(scope)
        or build_project_scope_key(code, line, display)
        for scope, code, line, display in zip(
            work["project_scope_key"],
            work["project_code"],
            work["line_name"],
            work["project_display"],
        )
    ]

    report_date = _parse_report_date(work.get("report_date"), work.get("source_file"))
    work["report_date"] = report_date
    work["month"] = report_date.dt.to_period("M").dt.to_timestamp()

    work = _coerce_numeric(
        work,
        (
            "quantity_primary",
            "cumulative_last_month",
            "plan_for_month",
            "progress_for_month",
            "today_progress",
            "cumulative_progress",
            "balance_progress",
        ),
    )

    if "activity_norm" in work.columns:
        activity_norm = work["activity_norm"].fillna("").astype(str).str.strip().str.lower()
    else:
        activity_norm = work["activity_raw"].fillna("").astype(str).str.strip().str.lower()
    activity_norm = activity_norm.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    work["activity_norm"] = activity_norm

    def _activity_group(norm_value: str) -> str:
        norm = _safe_text(norm_value).lower()
        if "foundation" in norm:
            return "Foundation"
        if "tower" in norm and "erection" in norm:
            return "Tower Erection"
        if "opgw" in norm and "string" in norm:
            return "OPGW Stringing"
        if "string" in norm:
            return "Stringing"
        return "Other"

    work["activity_group"] = work["activity_norm"].map(_activity_group)
    work["core_activity"] = work["activity_group"].isin({"Foundation", "Tower Erection", "Stringing", "OPGW Stringing"})

    output_cols = [
        "project_code",
        "project_display",
        "project_scope_key",
        "line_name",
        "report_date",
        "month",
        "section_label",
        "activity_raw",
        "activity_norm",
        "activity_group",
        "core_activity",
        "quantity_primary",
        "cumulative_last_month",
        "plan_for_month",
        "progress_for_month",
        "today_progress",
        "cumulative_progress",
        "balance_progress",
        "gangs_working",
        "remarks",
        "source_file",
        "source_sheet",
        "configured_sheet",
        "template_sheet",
    ]
    for col in output_cols:
        if col not in work.columns:
            work[col] = ""
    return work[output_cols].reset_index(drop=True)


def _build_status_snapshots(status_fact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(status_fact, pd.DataFrame) or status_fact.empty:
        project = pd.DataFrame(
            columns=[
                "project_code",
                "project_display",
                "project_scope_key",
                "line_name",
                "month",
                "report_date_max",
                "activities_total",
                "quantity_primary_sum",
                "cumulative_last_month_sum",
                "plan_for_month_sum",
                "progress_for_month_sum",
                "today_progress_sum",
                "cumulative_progress_sum",
                "balance_progress_sum",
                "completion_pct",
                "foundation_cumulative_progress",
                "tower_erection_cumulative_progress",
                "stringing_cumulative_progress",
                "opgw_stringing_cumulative_progress",
            ]
        )
        overall = pd.DataFrame(
            columns=[
                "month",
                "projects_total",
                "activities_total",
                "quantity_primary_sum",
                "cumulative_last_month_sum",
                "plan_for_month_sum",
                "progress_for_month_sum",
                "today_progress_sum",
                "cumulative_progress_sum",
                "balance_progress_sum",
                "completion_pct",
                "foundation_cumulative_progress",
                "tower_erection_cumulative_progress",
                "stringing_cumulative_progress",
                "opgw_stringing_cumulative_progress",
            ]
        )
        return project, overall

    work = status_fact.copy()
    numeric_cols = [
        "quantity_primary",
        "cumulative_last_month",
        "plan_for_month",
        "progress_for_month",
        "today_progress",
        "cumulative_progress",
        "balance_progress",
    ]
    work = _coerce_numeric(work, numeric_cols)

    group_cols = ["project_code", "project_display", "project_scope_key", "line_name", "month"]
    agg_map = {
        "report_date": "max",
        "activity_norm": "nunique",
        "quantity_primary": "sum",
        "cumulative_last_month": "sum",
        "plan_for_month": "sum",
        "progress_for_month": "sum",
        "today_progress": "sum",
        "cumulative_progress": "sum",
        "balance_progress": "sum",
    }
    project = work.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    project = project.rename(
        columns={
            "report_date": "report_date_max",
            "activity_norm": "activities_total",
            "quantity_primary": "quantity_primary_sum",
            "cumulative_last_month": "cumulative_last_month_sum",
            "plan_for_month": "plan_for_month_sum",
            "progress_for_month": "progress_for_month_sum",
            "today_progress": "today_progress_sum",
            "cumulative_progress": "cumulative_progress_sum",
            "balance_progress": "balance_progress_sum",
        }
    )
    denom = project["quantity_primary_sum"].where(project["quantity_primary_sum"] > 0)
    project["completion_pct"] = (project["cumulative_progress_sum"] / denom * 100.0).fillna(0.0)

    for key, label in (
        ("Foundation", "foundation_cumulative_progress"),
        ("Tower Erection", "tower_erection_cumulative_progress"),
        ("Stringing", "stringing_cumulative_progress"),
        ("OPGW Stringing", "opgw_stringing_cumulative_progress"),
    ):
        subset = (
            work[work["activity_group"] == key]
            .groupby(group_cols, dropna=False)["cumulative_progress"]
            .sum()
            .reset_index()
            .rename(columns={"cumulative_progress": label})
        )
        project = project.merge(subset, on=group_cols, how="left")

    for label in (
        "foundation_cumulative_progress",
        "tower_erection_cumulative_progress",
        "stringing_cumulative_progress",
        "opgw_stringing_cumulative_progress",
    ):
        if label in project.columns:
            project[label] = pd.to_numeric(project[label], errors="coerce").fillna(0.0)

    overall = (
        project.groupby("month", dropna=False)
        .agg(
            projects_total=("project_scope_key", lambda s: int(s.fillna("").astype(str).str.strip().astype(bool).sum())),
            activities_total=("activities_total", "sum"),
            quantity_primary_sum=("quantity_primary_sum", "sum"),
            cumulative_last_month_sum=("cumulative_last_month_sum", "sum"),
            plan_for_month_sum=("plan_for_month_sum", "sum"),
            progress_for_month_sum=("progress_for_month_sum", "sum"),
            today_progress_sum=("today_progress_sum", "sum"),
            cumulative_progress_sum=("cumulative_progress_sum", "sum"),
            balance_progress_sum=("balance_progress_sum", "sum"),
            foundation_cumulative_progress=("foundation_cumulative_progress", "sum"),
            tower_erection_cumulative_progress=("tower_erection_cumulative_progress", "sum"),
            stringing_cumulative_progress=("stringing_cumulative_progress", "sum"),
            opgw_stringing_cumulative_progress=("opgw_stringing_cumulative_progress", "sum"),
        )
        .reset_index()
    )
    overall_denom = overall["quantity_primary_sum"].where(overall["quantity_primary_sum"] > 0)
    overall["completion_pct"] = (overall["cumulative_progress_sum"] / overall_denom * 100.0).fillna(0.0)
    return project.reset_index(drop=True), overall.reset_index(drop=True)


def _normalize_section_id(stretch_identifier: object, from_ap: object, to_ap: object, section_label: object) -> str:
    identifier = _safe_text(stretch_identifier)
    if identifier:
        return identifier
    from_loc = _safe_text(from_ap)
    to_loc = _safe_text(to_ap)
    if from_loc or to_loc:
        return f"{from_loc}-{to_loc}".strip("-")
    return _safe_text(section_label)


def _build_stretch_section_fact(stretch_raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(stretch_raw, pd.DataFrame) or stretch_raw.empty:
        return pd.DataFrame(
            columns=[
                "project_code",
                "project_display",
                "project_scope_key",
                "line_name",
                "report_date",
                "month",
                "section_label",
                "section_id",
                "from_ap",
                "to_ap",
                "length_km",
                "readiness_state",
                "is_ready",
                "is_partial",
                "is_not_ready",
                "is_unknown",
                "balance_towers",
                "remarks",
                "source_file",
                "source_sheet",
                "configured_sheet",
                "template_sheet",
            ]
        )

    work = stretch_raw.copy()
    for column in (
        "project_code",
        "project_display",
        "project_scope_key",
        "line_name",
        "stretch_identifier",
        "from_ap",
        "to_ap",
        "section_label",
        "readiness_state",
        "source_file",
        "source_sheet",
        "configured_sheet",
        "template_sheet",
        "remarks",
        "balance_towers",
    ):
        if column not in work.columns:
            work[column] = ""

    work["project_code"] = work["project_code"].map(_normalize_project_code)
    work["line_name"] = work["line_name"].map(normalize_line_name)
    work["project_display"] = [
        _normalize_project_display(code, display, line)
        for code, display, line in zip(work["project_code"], work["project_display"], work["line_name"])
    ]
    work["project_scope_key"] = [
        _safe_text(scope)
        or build_project_scope_key(code, line, display)
        for scope, code, line, display in zip(
            work["project_scope_key"],
            work["project_code"],
            work["line_name"],
            work["project_display"],
        )
    ]

    report_date = _parse_report_date(work.get("report_date"), work.get("source_file"))
    work["report_date"] = report_date
    work["month"] = report_date.dt.to_period("M").dt.to_timestamp()

    work["section_id"] = [
        _normalize_section_id(stretch_identifier, from_ap, to_ap, section_label)
        for stretch_identifier, from_ap, to_ap, section_label in zip(
            work["stretch_identifier"], work["from_ap"], work["to_ap"], work["section_label"]
        )
    ]

    work = _coerce_numeric(work, ("length_km", "balance_towers"))
    readiness = work["readiness_state"].fillna("").astype(str).str.strip().str.upper()
    allowed = {"READY", "PARTIAL", "NOT_READY", "UNKNOWN"}
    readiness = readiness.where(readiness.isin(allowed), "UNKNOWN")
    work["readiness_state"] = readiness
    work["is_ready"] = readiness.eq("READY")
    work["is_partial"] = readiness.eq("PARTIAL")
    work["is_not_ready"] = readiness.eq("NOT_READY")
    work["is_unknown"] = readiness.eq("UNKNOWN")

    output_cols = [
        "project_code",
        "project_display",
        "project_scope_key",
        "line_name",
        "report_date",
        "month",
        "section_label",
        "section_id",
        "from_ap",
        "to_ap",
        "length_km",
        "readiness_state",
        "is_ready",
        "is_partial",
        "is_not_ready",
        "is_unknown",
        "balance_towers",
        "remarks",
        "source_file",
        "source_sheet",
        "configured_sheet",
        "template_sheet",
    ]
    return work[output_cols].reset_index(drop=True)


def _normalize_loc(value: object) -> str:
    text = _safe_text(value).upper()
    text = re.sub(r"^AP[\s\-_/]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    return text


def _span_key(from_ap: object, to_ap: object) -> str:
    start = _normalize_loc(from_ap)
    end = _normalize_loc(to_ap)
    if not start and not end:
        return ""
    pair = sorted([start, end])
    return f"{pair[0]}|{pair[1]}"


def _select_column_by_aliases(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normalized_map = {_norm_key(column): column for column in df.columns}
    for alias in aliases:
        alias_key = _norm_key(alias)
        if alias_key in normalized_map:
            return normalized_map[alias_key]
    for alias in aliases:
        alias_key = _norm_key(alias)
        for key, original in normalized_map.items():
            if alias_key and alias_key in key:
                return original
    return None


def _build_compiled_manpower_map(compiled_raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(compiled_raw, pd.DataFrame) or compiled_raw.empty:
        return pd.DataFrame(
            columns=[
                "project_code",
                "line_name",
                "project_scope_key",
                "span_key",
                "gang_name_key",
                "manpower_gang_strength",
                "manpower_fitters",
            ]
        )

    normalized, _ = normalize_stringing_columns(compiled_raw)
    work = normalized.copy()
    for column in ("project_code", "project_name", "project", "line_name", "from_ap", "to_ap", "gang_name"):
        if column not in work.columns:
            work[column] = ""
    if not work["project_code"].fillna("").astype(str).str.strip().astype(bool).any():
        for candidate in ("Project Code", "project code", "PROJECT CODE"):
            if candidate in compiled_raw.columns:
                work["project_code"] = compiled_raw[candidate]
                break
    if not work["project_name"].fillna("").astype(str).str.strip().astype(bool).any():
        for candidate in ("Project Name", "project name", "PROJECT NAME"):
            if candidate in compiled_raw.columns:
                work["project_name"] = compiled_raw[candidate]
                break
    if not work["project"].fillna("").astype(str).str.strip().astype(bool).any():
        for candidate in ("Project", "project"):
            if candidate in compiled_raw.columns:
                work["project"] = compiled_raw[candidate]
                break
    if not work["line_name"].fillna("").astype(str).str.strip().astype(bool).any():
        for candidate in ("Line Name", "line name", "LINE NAME"):
            if candidate in compiled_raw.columns:
                work["line_name"] = compiled_raw[candidate]
                break

    project_display_raw = work["project_name"].fillna("").astype(str)
    project_display_raw = project_display_raw.where(project_display_raw.str.strip().astype(bool), work["project"].fillna("").astype(str))
    project_codes = work["project_code"].map(_normalize_project_code)
    project_codes = project_codes.where(project_codes.astype(bool), project_display_raw.map(_normalize_project_code))

    work["project_code"] = project_codes
    work["line_name"] = work["line_name"].map(normalize_line_name)
    work["project_display"] = [
        _normalize_project_display(code, display, line)
        for code, display, line in zip(work["project_code"], project_display_raw, work["line_name"])
    ]
    work["project_scope_key"] = [
        build_project_scope_key(code, line, display)
        for code, line, display in zip(work["project_code"], work["line_name"], work["project_display"])
    ]
    work["span_key"] = [_span_key(a, b) for a, b in zip(work["from_ap"], work["to_ap"])]
    work["gang_name_key"] = work["gang_name"].fillna("").astype(str).str.strip().str.lower()

    gang_strength_col = _select_column_by_aliases(work, ("gang strength", "gang_strength", "gangsize", "gang size", "strength"))
    fitters_col = _select_column_by_aliases(work, ("number of fitters", "no of fitters", "fitters", "fitter"))

    work["manpower_gang_strength"] = pd.to_numeric(work[gang_strength_col], errors="coerce") if gang_strength_col else pd.NA
    work["manpower_fitters"] = pd.to_numeric(work[fitters_col], errors="coerce") if fitters_col else pd.NA

    keep_cols = [
        "project_code",
        "line_name",
        "project_scope_key",
        "span_key",
        "gang_name_key",
        "manpower_gang_strength",
        "manpower_fitters",
    ]
    subset = work[keep_cols].copy()
    subset = subset[subset["project_code"].fillna("").astype(str).str.strip().astype(bool)]
    subset = subset[subset["span_key"].fillna("").astype(str).str.strip().astype(bool)]
    subset = subset.drop_duplicates(subset=["project_code", "line_name", "span_key", "gang_name_key"], keep="last")
    return subset.reset_index(drop=True)


def _audit_priority(signal: str) -> int:
    key = _safe_text(signal).upper()
    if key == "PRESENT_WITH_VALUES":
        return 4
    if key == "HEADER_ONLY":
        return 3
    if key == "ABSENT":
        return 2
    if key.startswith("MISSING"):
        return 1
    return 0


def _prepare_manpower_audit_map(manpower_audit: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(manpower_audit, pd.DataFrame) or manpower_audit.empty:
        return pd.DataFrame(columns=["project_code", "audit_signal_type", "audit_status", "expected_manpower", "expected_match", "audit_reason"])

    work = manpower_audit.copy()
    for column in ("project_code", "signal_type", "status", "expected_manpower", "expected_match", "reason"):
        if column not in work.columns:
            work[column] = ""
    work["project_code"] = work["project_code"].map(_normalize_project_code)
    work["signal_type"] = work["signal_type"].fillna("").astype(str).str.strip().str.upper()
    work["_priority"] = work["signal_type"].map(_audit_priority)
    work = work.sort_values(["project_code", "_priority"], ascending=[True, False])
    grouped = work.groupby("project_code", dropna=False).first().reset_index()
    grouped = grouped.rename(
        columns={
            "signal_type": "audit_signal_type",
            "status": "audit_status",
            "reason": "audit_reason",
        }
    )
    return grouped[["project_code", "audit_signal_type", "audit_status", "expected_manpower", "expected_match", "audit_reason"]]


def _build_manpower_productivity_fact(
    stringing_daily: pd.DataFrame,
    stringing_compiled: pd.DataFrame,
    manpower_audit: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "project_code",
        "project_display",
        "project_scope_key",
        "line_name",
        "date",
        "month",
        "gang_name",
        "from_ap",
        "to_ap",
        "span_key",
        "method",
        "section_readiness",
        "daily_km",
        "po_km",
        "manpower_gang_strength",
        "manpower_fitters",
        "manpower_signal_type",
        "manpower_status",
        "expected_manpower",
        "expected_match",
        "availability",
        "availability_reason",
    ]
    if not isinstance(stringing_daily, pd.DataFrame) or stringing_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = stringing_daily.copy()
    for col in ("project", "project_code", "project_name", "line_name", "date", "gang_name", "from_ap", "to_ap", "method", "section_readiness", "daily_km", "po_km"):
        if col not in daily.columns:
            daily[col] = ""

    display_series = daily["project"].fillna("").astype(str)
    display_series = display_series.where(display_series.str.strip().astype(bool), daily["project_name"].fillna("").astype(str))
    project_codes = daily["project_code"].map(_normalize_project_code)
    project_codes = project_codes.where(project_codes.astype(bool), display_series.map(_normalize_project_code))

    daily["project_code"] = project_codes
    daily["line_name"] = daily["line_name"].map(normalize_line_name)
    daily["project_display"] = [
        _normalize_project_display(code, display, line)
        for code, display, line in zip(daily["project_code"], display_series, daily["line_name"])
    ]
    daily["project_scope_key"] = [
        build_project_scope_key(code, line, display)
        for code, line, display in zip(daily["project_code"], daily["line_name"], daily["project_display"])
    ]
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    daily["daily_km"] = pd.to_numeric(daily["daily_km"], errors="coerce")
    daily["po_km"] = pd.to_numeric(daily.get("po_km"), errors="coerce")
    daily["gang_name"] = daily["gang_name"].fillna("").astype(str).str.strip()
    daily["gang_name_key"] = daily["gang_name"].str.lower()
    daily["span_key"] = [_span_key(a, b) for a, b in zip(daily["from_ap"], daily["to_ap"])]

    manpower_map = _build_compiled_manpower_map(stringing_compiled)
    if not manpower_map.empty:
        exact = daily.merge(
            manpower_map,
            how="left",
            on=["project_code", "line_name", "project_scope_key", "span_key", "gang_name_key"],
        )
        fallback_gang = (
            manpower_map.groupby(["project_code", "span_key", "gang_name_key"], dropna=False)[["manpower_gang_strength", "manpower_fitters"]]
            .first()
            .reset_index()
            .rename(
                columns={
                    "manpower_gang_strength": "manpower_gang_strength_gang_fallback",
                    "manpower_fitters": "manpower_fitters_gang_fallback",
                }
            )
        )
        exact = exact.merge(fallback_gang, how="left", on=["project_code", "span_key", "gang_name_key"])
        fallback_span = (
            manpower_map.groupby(["project_code", "span_key"], dropna=False)[["manpower_gang_strength", "manpower_fitters"]]
            .first()
            .reset_index()
            .rename(
                columns={
                    "manpower_gang_strength": "manpower_gang_strength_span_fallback",
                    "manpower_fitters": "manpower_fitters_span_fallback",
                }
            )
        )
        exact = exact.merge(fallback_span, how="left", on=["project_code", "span_key"])
        exact["manpower_gang_strength"] = pd.to_numeric(exact.get("manpower_gang_strength"), errors="coerce")
        exact["manpower_fitters"] = pd.to_numeric(exact.get("manpower_fitters"), errors="coerce")
        exact["manpower_gang_strength"] = exact["manpower_gang_strength"].where(
            exact["manpower_gang_strength"].notna(),
            pd.to_numeric(exact.get("manpower_gang_strength_gang_fallback"), errors="coerce"),
        )
        exact["manpower_fitters"] = exact["manpower_fitters"].where(
            exact["manpower_fitters"].notna(),
            pd.to_numeric(exact.get("manpower_fitters_gang_fallback"), errors="coerce"),
        )
        exact["manpower_gang_strength"] = exact["manpower_gang_strength"].where(
            exact["manpower_gang_strength"].notna(),
            pd.to_numeric(exact.get("manpower_gang_strength_span_fallback"), errors="coerce"),
        )
        exact["manpower_fitters"] = exact["manpower_fitters"].where(
            exact["manpower_fitters"].notna(),
            pd.to_numeric(exact.get("manpower_fitters_span_fallback"), errors="coerce"),
        )
        daily = exact
    else:
        daily["manpower_gang_strength"] = pd.NA
        daily["manpower_fitters"] = pd.NA

    audit_map = _prepare_manpower_audit_map(manpower_audit)
    if not audit_map.empty:
        daily = daily.merge(audit_map, how="left", on="project_code")
    else:
        daily["audit_signal_type"] = ""
        daily["audit_status"] = ""
        daily["expected_manpower"] = ""
        daily["expected_match"] = pd.NA
        daily["audit_reason"] = ""

    has_values = pd.to_numeric(daily.get("manpower_gang_strength"), errors="coerce").notna() | pd.to_numeric(
        daily.get("manpower_fitters"), errors="coerce"
    ).notna()
    audit_signal = daily.get("audit_signal_type", pd.Series("", index=daily.index)).fillna("").astype(str).str.upper()
    daily["manpower_signal_type"] = audit_signal.where(audit_signal.astype(bool), "UNKNOWN")
    daily.loc[has_values, "manpower_signal_type"] = "PRESENT_WITH_VALUES"

    availability = pd.Series("NO_DATA", index=daily.index, dtype="object")
    availability = availability.where(~has_values, "AVAILABLE")
    availability = availability.where(~(~has_values & audit_signal.eq("HEADER_ONLY")), "HEADER_ONLY")
    availability = availability.where(~(~has_values & audit_signal.eq("ABSENT")), "NO_DATA")
    daily["availability"] = availability

    reason = pd.Series("No manpower values found for this span/day.", index=daily.index, dtype="object")
    reason = reason.where(~has_values, "Manpower values mapped from stringing compiled sheet.")
    reason = reason.where(~(~has_values & audit_signal.eq("HEADER_ONLY")), "Manpower header exists but values are blank in source DPR.")
    reason = reason.where(~(~has_values & audit_signal.str.startswith("MISSING")), "Manpower source sheet missing in DPR configuration/source.")
    daily["availability_reason"] = reason
    daily["manpower_status"] = daily.get("audit_status", "")

    output = daily[
        [
            "project_code",
            "project_display",
            "project_scope_key",
            "line_name",
            "date",
            "month",
            "gang_name",
            "from_ap",
            "to_ap",
            "span_key",
            "method",
            "section_readiness",
            "daily_km",
            "po_km",
            "manpower_gang_strength",
            "manpower_fitters",
            "manpower_signal_type",
            "manpower_status",
            "expected_manpower",
            "expected_match",
            "availability",
            "availability_reason",
        ]
    ].copy()
    return output.sort_values(["project_code", "line_name", "date", "gang_name"], na_position="last").reset_index(drop=True)


def _build_coverage(
    stringing_coverage: pd.DataFrame,
    status_coverage: pd.DataFrame,
    stretch_coverage: pd.DataFrame,
    manpower_fact: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _emit_row(
        *,
        project_code: object,
        project_display: object,
        category: str,
        status: object,
        reason_code: object,
        reason: object,
        workbook: object,
        configured_sheet: object,
        resolved_sheet: object,
        row_count: object,
    ) -> None:
        code = _normalize_project_code(project_code)
        display = _safe_text(project_display) or code
        numeric_rows = pd.to_numeric(row_count, errors="coerce")
        rows.append(
            {
                "project_code": code,
                "project_display": display,
                "project_scope_key": build_project_scope_key(code, "", display),
                "category": category,
                "status": _safe_text(status),
                "reason_code": _safe_text(reason_code),
                "reason": _safe_text(reason),
                "workbook": _safe_text(workbook),
                "configured_sheet": _safe_text(configured_sheet),
                "resolved_sheet": _safe_text(resolved_sheet),
                "rows": int(numeric_rows) if pd.notna(numeric_rows) else 0,
            }
        )

    if isinstance(stringing_coverage, pd.DataFrame) and not stringing_coverage.empty:
        for record in stringing_coverage.to_dict("records"):
            _emit_row(
                project_code=record.get("project_code"),
                project_display=record.get("project_display") or record.get("project_code"),
                category="stringing",
                status=record.get("status"),
                reason_code=record.get("reason_code") or record.get("status"),
                reason=record.get("reason"),
                workbook=record.get("workbook"),
                configured_sheet=record.get("configured_sheet"),
                resolved_sheet=record.get("resolved_sheet"),
                row_count=record.get("compiled_rows") or record.get("daily_rows") or 0,
            )

    if isinstance(status_coverage, pd.DataFrame) and not status_coverage.empty:
        for record in status_coverage.to_dict("records"):
            _emit_row(
                project_code=record.get("project_code"),
                project_display=record.get("project_display") or record.get("project_code"),
                category="status",
                status=record.get("status"),
                reason_code=record.get("reason_code") or record.get("status"),
                reason=record.get("reason"),
                workbook=record.get("workbook"),
                configured_sheet=record.get("configured_sheet"),
                resolved_sheet=record.get("resolved_sheet"),
                row_count=record.get("rows") or 0,
            )

    if isinstance(stretch_coverage, pd.DataFrame) and not stretch_coverage.empty:
        for record in stretch_coverage.to_dict("records"):
            _emit_row(
                project_code=record.get("project_code"),
                project_display=record.get("project_display") or record.get("project_code"),
                category=_safe_text(record.get("category") or "stretch"),
                status=record.get("status"),
                reason_code=record.get("reason_code") or record.get("status"),
                reason=record.get("reason"),
                workbook=record.get("workbook"),
                configured_sheet=record.get("configured_sheet"),
                resolved_sheet=record.get("resolved_sheet"),
                row_count=record.get("rows") or 0,
            )

    if isinstance(manpower_fact, pd.DataFrame) and not manpower_fact.empty:
        grouped = (
            manpower_fact.groupby(["project_code", "project_display"], dropna=False)
            .agg(
                rows=("date", "count"),
                available_rows=("availability", lambda s: int(s.fillna("").astype(str).eq("AVAILABLE").sum())),
                header_only_rows=("availability", lambda s: int(s.fillna("").astype(str).eq("HEADER_ONLY").sum())),
            )
            .reset_index()
        )
        for record in grouped.to_dict("records"):
            status = "AVAILABLE" if int(record.get("available_rows", 0)) > 0 else (
                "HEADER_ONLY" if int(record.get("header_only_rows", 0)) > 0 else "NO_DATA"
            )
            reason = (
                "Manpower mapped to productivity rows." if status == "AVAILABLE" else (
                    "Manpower headers present without values." if status == "HEADER_ONLY" else "No manpower values were mapped."
                )
            )
            _emit_row(
                project_code=record.get("project_code"),
                project_display=record.get("project_display") or record.get("project_code"),
                category="manpower_productivity",
                status=status,
                reason_code=status,
                reason=reason,
                workbook="",
                configured_sheet="",
                resolved_sheet="",
                row_count=record.get("rows") or 0,
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "project_code",
                "project_display",
                "project_scope_key",
                "category",
                "status",
                "reason_code",
                "reason",
                "workbook",
                "configured_sheet",
                "resolved_sheet",
                "rows",
            ]
        )

    coverage = pd.DataFrame(rows)
    return coverage.sort_values(["project_code", "category", "status"]).reset_index(drop=True)


def _build_diagnostics(
    stringing_daily: pd.DataFrame,
    status_raw: pd.DataFrame,
    stretch_raw: pd.DataFrame,
    manpower_audit: pd.DataFrame,
    status_fact: pd.DataFrame,
    stretch_fact: pd.DataFrame,
    manpower_fact: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _row(component: str, source_df: pd.DataFrame) -> dict[str, object]:
        present = isinstance(source_df, pd.DataFrame) and not source_df.empty
        return {
            "component": component,
            "status": "AVAILABLE" if present else "NO_DATA",
            "rows": int(len(source_df.index)) if isinstance(source_df, pd.DataFrame) else 0,
        }

    rows.append(_row("StringingDailySource", stringing_daily))
    rows.append(_row("ProgressStatusRawSource", status_raw))
    rows.append(_row("StretchReadinessRawSource", stretch_raw))
    rows.append(_row("StretchManpowerAuditSource", manpower_audit))
    rows.append(_row("StatusActivityFact", status_fact))
    rows.append(_row("StretchSectionFact", stretch_fact))
    rows.append(_row("ManpowerProductivityFact", manpower_fact))

    return pd.DataFrame(rows)


def _build_issues(issues: list[dict[str, object]]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["severity", "component", "code", "message"])
    frame = pd.DataFrame(issues)
    for col in ("severity", "component", "code", "message"):
        if col not in frame.columns:
            frame[col] = ""
    return frame[["severity", "component", "code", "message"]]


def compile_stringing_summary_to_workbook(base_dir: Path, output_path: Path) -> Path:
    """Compile StringingSummary workbook from existing Stringing/Status/Stretch artifacts."""

    base_dir = Path(base_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stringing_root = _resolve_sibling_root(base_dir, "Stringing")
    status_root = _resolve_sibling_root(base_dir, "ProgressStatus")
    stretch_root = _resolve_sibling_root(base_dir, "StretchReadiness")
    issues: list[dict[str, object]] = []

    def _load(root: Path, workbook: str, sheet: str, component: str) -> pd.DataFrame:
        table = _load_table(root, workbook, sheet)
        if table.empty:
            issues.append(
                {
                    "severity": "INFO",
                    "component": component,
                    "code": "NO_DATA",
                    "message": f"{component} table '{sheet}' unavailable under {root}.",
                }
            )
        return table

    stringing_daily = _load(stringing_root, "StringingCompiled_Output.xlsx", "StringingDaily", "Stringing")
    stringing_compiled = _load(stringing_root, "StringingCompiled_Output.xlsx", "StringingCompiled", "Stringing")
    stringing_coverage = _load(stringing_root, "StringingCompiled_Output.xlsx", "StringingCoverage", "Stringing")

    status_raw = _load(status_root, "ProgressStatus_Output.xlsx", "RawData", "ProgressStatus")
    status_coverage = _load(status_root, "ProgressStatus_Output.xlsx", "Coverage", "ProgressStatus")

    stretch_raw = _load(stretch_root, "StretchReadiness_Output.xlsx", "RawData", "StretchReadiness")
    stretch_coverage = _load(stretch_root, "StretchReadiness_Output.xlsx", "Coverage", "StretchReadiness")
    manpower_audit = _load(stretch_root, "StretchReadiness_Output.xlsx", "ManpowerAudit", "StretchReadiness")

    status_fact = _build_status_activity_fact(status_raw)
    status_project, status_overall = _build_status_snapshots(status_fact)
    stretch_fact = _build_stretch_section_fact(stretch_raw)
    manpower_fact = _build_manpower_productivity_fact(stringing_daily, stringing_compiled, manpower_audit)
    coverage = _build_coverage(stringing_coverage, status_coverage, stretch_coverage, manpower_fact)
    diagnostics = _build_diagnostics(
        stringing_daily,
        status_raw,
        stretch_raw,
        manpower_audit,
        status_fact,
        stretch_fact,
        manpower_fact,
    )
    issues_df = _build_issues(issues)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        status_fact.to_excel(writer, sheet_name="StatusActivityFact", index=False)
        status_project.to_excel(writer, sheet_name="StatusSnapshotProject", index=False)
        status_overall.to_excel(writer, sheet_name="StatusSnapshotOverall", index=False)
        stretch_fact.to_excel(writer, sheet_name="StretchSectionFact", index=False)
        manpower_fact.to_excel(writer, sheet_name="ManpowerProductivityFact", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
        issues_df.to_excel(writer, sheet_name="Issues", index=False)

    return output_path


__all__ = ["compile_stringing_summary_to_workbook", "STRINGING_SUMMARY_SHEETS"]
