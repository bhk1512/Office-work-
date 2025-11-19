"""Shared helpers for plan normalization and stringing plan transformations."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)


def normalize_text(value: object) -> str:
    text = str(value).replace("\u00a0", " ").strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    return text


def normalize_lower(value: object) -> str:
    return normalize_text(value).lower()


def compact_project_key(value: object) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_col_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_location(value: object) -> str:
    txt = normalize_text(value)
    if not txt:
        return ""
    if txt.endswith(".0") and txt.replace(".", "", 1).isdigit():
        txt = txt.split(".", 1)[0]
    return txt


def infer_project_hint(path: Path | str | None) -> tuple[str, str]:
    if path is None:
        return "", ""
    path_obj = Path(path)
    name = path_obj.stem
    match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{2,4})\b", name, re.IGNORECASE)
    code = ""
    if match:
        code = f"{match.group(1).upper()}-{match.group(2)}"
    label = re.sub(r"(?i)(micro\s*plan\s*-\s*)", "", name).strip(" _-")
    label = label or code
    return code, label


def prepare_stringing_plan_frame(
    df_raw: pd.DataFrame,
    *,
    project_hint: str | None = None,
    source_path: Path | str | None = None,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, set[tuple[str, str]], list[dict[str, str]]]:
    """
    Normalize the stringing monthly plan sheet into the generic responsibilities structure.
    Returns (frame, completion_keys, issues) where completion keys capture completed spans.
    """

    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        columns = [
            "project_name",
            "project_key",
            "entity_type",
            "entity_name",
            "location_no",
            "tower_weight",
            "revenue_planned",
            "revenue_realised",
            "stringing_span_completed",
            "span_from",
            "span_to",
            "method",
            "gang_strength",
            "paying_out_start",
            "final_sag_complete",
        ]
        return pd.DataFrame(columns=columns), set(), []

    col_lookup = {normalize_col_key(col): col for col in df_raw.columns}
    column_aliases: dict[str, tuple[str, ...]] = {
        "serial": ("S. No.", "S no", "s no", "serial", "serial no", "jmc no", "span no"),
        "span_from": ("From AP", "from_ap", "from ap", "start tower", "from tower"),
        "span_to": ("To AP", "to_ap", "to ap", "end tower", "to tower"),
        "span_length": ("Span (m)", "span m", "span length", "length_m", "length (m)", "length"),
        "method": ("Method", "method"),
        "gang_strength": ("Gang Strength", "gang_strength"),
        "paying_out_start": ("Paying Out Start", "po_start_date", "p/o start", "po start", "paying_out_start"),
        "paying_out_complete": ("Paying Out Completed", "po_completion_date", "p/o completed", "po completion"),
        "final_sag_complete": ("Final Sag Complete", "fs_complete_date", "final sag", "fs complete date"),
        "gang_name": ("Gang Name", "gang_name"),
        "supervisor": ("Supervisor", "supervisor"),
        "section_incharge": ("Section Incharge", "section_incharge", "section incharge"),
        "po_length": ("P/O", "po", "p/o length", "po length", "p/o"),
    }

    def _resolve_series(key: str, default: Any = "") -> tuple[pd.Series, bool]:
        options = column_aliases.get(key, ())
        for candidate in options:
            norm = normalize_col_key(candidate)
            if norm in col_lookup:
                return df_raw[col_lookup[norm]], True
        norm = normalize_col_key(key)
        if norm in col_lookup:
            return df_raw[col_lookup[norm]], True
        return pd.Series([default] * len(df_raw), index=df_raw.index), False

    def _optional_series(candidates: Sequence[str], default: Any = "") -> pd.Series:
        for candidate in candidates:
            key = normalize_col_key(candidate)
            if key in col_lookup:
                return df_raw[col_lookup[key]]
        return pd.Series([default] * len(df_raw), index=df_raw.index)

    issues: list[dict[str, str]] = []
    required_for_logging = {
        "S. No.": column_aliases["serial"],
        "From AP": column_aliases["span_from"],
        "To AP": column_aliases["span_to"],
        "Span (m)": column_aliases["span_length"],
        "Method": column_aliases["method"],
        "Gang Name": column_aliases["gang_name"],
        "Supervisor": column_aliases["supervisor"],
        "Section Incharge": column_aliases["section_incharge"],
    }
    missing = [
        label
        for label, aliases in required_for_logging.items()
        if not any(normalize_col_key(alias) in col_lookup for alias in aliases)
    ]
    if missing:
        LOGGER.warning("Monthly Plan (Stringing) missing columns: %s", ", ".join(missing))
        issues.append(
            {
                "workbook": str(source_path or ""),
                "sheet": sheet_name or "",
                "issue": f"Missing columns: {', '.join(missing)}",
            }
        )

    project_names = _optional_series(("Project Name", "Project", "Project Title", "project_name")).map(normalize_text)
    project_codes = _optional_series(("Project Code", "Project Key", "Project Id", "Project ID", "project")).map(
        normalize_text
    )
    if project_hint:
        project_names = project_names.where(project_names.astype(bool), project_hint)
        project_codes = project_codes.where(project_codes.astype(bool), project_hint)

    def _coerce_project_code(value: str | None) -> str:
        text = normalize_text(value or "")
        match = re.search(r"\b(TA|TB)\s*[-_/ ]?\s*(\d{2,4})\b", text, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()}{match.group(2)}"
        return ""

    def _derive_project_identity(name_raw: str, code_raw: str, hint: str | None) -> tuple[str, str]:
        code = _coerce_project_code(code_raw) or _coerce_project_code(name_raw) or _coerce_project_code(hint)
        if code:
            return code, code.lower()
        fallback = normalize_text(name_raw) or normalize_text(code_raw) or normalize_text(hint)
        if fallback:
            clean_code = _coerce_project_code(fallback)
            if clean_code:
                return clean_code, clean_code.lower()
            return fallback, fallback.lower()
        return "", ""
    serial_values, _ = _resolve_series("serial", default="")
    serial_values = serial_values.map(normalize_text)
    span_from, _ = _resolve_series("span_from", default="")
    span_from = span_from.map(normalize_text)
    span_to, _ = _resolve_series("span_to", default="")
    span_to = span_to.map(normalize_text)
    span_length_series, _ = _resolve_series("span_length", default=0.0)
    span_length = pd.to_numeric(span_length_series, errors="coerce").fillna(0.0)
    po_length_series, _ = _resolve_series("po_length", default=0.0)
    po_length = pd.to_numeric(po_length_series, errors="coerce").fillna(0.0)
    method_values, _ = _resolve_series("method", default="")
    method_values = method_values.map(normalize_text)
    gang_strength_series, gang_has_col = _resolve_series("gang_strength", default=pd.NA)
    gang_strength = pd.to_numeric(gang_strength_series, errors="coerce")
    paying_out_start_series, _ = _resolve_series("paying_out_start", default=pd.NaT)
    paying_out_complete_series, _ = _resolve_series("paying_out_complete", default=pd.NaT)
    paying_out_start = pd.to_datetime(paying_out_start_series, errors="coerce")
    paying_out_complete = pd.to_datetime(paying_out_complete_series, errors="coerce")
    final_sag_complete_series, _ = _resolve_series("final_sag_complete", default=pd.NaT)
    final_sag_complete = pd.to_datetime(final_sag_complete_series, errors="coerce")

    entity_sources: list[tuple[str, list[str]]] = []
    gang_series, has_gang = _resolve_series("gang_name", default="")
    if has_gang:
        entity_sources.append(("Gang", gang_series.map(normalize_text).tolist()))
    supervisor_series, has_supervisor = _resolve_series("supervisor", default="")
    if has_supervisor:
        entity_sources.append(("Supervisor", supervisor_series.map(normalize_text).tolist()))
    section_series, has_section = _resolve_series("section_incharge", default="")
    if has_section:
        entity_sources.append(("Section Incharge", section_series.map(normalize_text).tolist()))

    normalized_rows: list[dict[str, Any]] = []
    completion_pairs: set[tuple[str, str]] = set()

    span_count = len(df_raw.index)
    span_done_mask = (paying_out_start.notna() & final_sag_complete.notna()).tolist()
    from_vals = span_from.tolist()
    to_vals = span_to.tolist()
    project_name_vals = project_names.tolist()
    project_code_vals = project_codes.tolist()
    serial_vals = serial_values.tolist()
    span_length_vals = span_length.tolist()
    method_vals = method_values.tolist()
    gang_strength_vals = gang_strength.tolist()
    paying_out_values = paying_out_start.tolist()
    paying_out_complete_values = paying_out_complete.tolist()
    final_sag_values = final_sag_complete.tolist()

    for idx in range(span_count):
        project_name_raw = project_name_vals[idx]
        project_code_raw = project_code_vals[idx]
        project_name, project_code = _derive_project_identity(
            project_name_raw,
            project_code_raw,
            project_hint,
        )
        from_ap = from_vals[idx]
        to_ap = to_vals[idx]
        serial_label = serial_vals[idx]
        if from_ap and to_ap:
            span_label = f"{from_ap} \u2192 {to_ap}"
        else:
            span_label = from_ap or to_ap or serial_label or f"Span {idx + 1}"
        span_norm = normalize_location(span_label)
        span_length_value = float(span_length_vals[idx]) if pd.notna(span_length_vals[idx]) else 0.0
        method_value = method_vals[idx]
        span_completed = bool(span_done_mask[idx])
        po_start_value = paying_out_values[idx]
        po_complete_value = paying_out_complete_values[idx]
        sag_complete_value = final_sag_values[idx]
        gang_strength_value = gang_strength_vals[idx]

        base_projects = [
            normalize_lower(project_name),
            normalize_lower(project_code),
        ]
        if span_completed and span_norm and any(base_projects):
            for candidate in base_projects:
                if candidate:
                    completion_pairs.add((candidate, span_norm))

        for entity_label, entity_values in entity_sources:
            entity_name = entity_values[idx]
            if not entity_name:
                continue
            normalized_rows.append(
                {
                    "project_name": project_name,
                    "project_key": project_code or project_name,
                    "entity_type": entity_label,
                    "entity_name": entity_name,
                    "location_no": span_label,
                    "tower_weight": span_length_value,
                    "p/o": float(po_length[idx]) if idx < len(po_length) and pd.notna(po_length[idx]) else 0.0,
                    "revenue_planned": 0.0,
                    "revenue_realised": 0.0,
                    "stringing_span_completed": span_completed,
                    "span_from": from_ap,
                    "span_to": to_ap,
                    "method": method_value,
                    "gang_strength": gang_strength_value,
                    "paying_out_start": po_start_value,
                    "paying_out_complete": po_complete_value,
                    "final_sag_complete": sag_complete_value,
                }
            )

    normalized = pd.DataFrame(normalized_rows)
    required_payload_columns: list[tuple[str, Any]] = [
        ("project_name", ""),
        ("project_key", ""),
        ("entity_type", ""),
        ("entity_name", ""),
        ("location_no", ""),
        ("tower_weight", 0.0),
        ("p/o", 0.0),
        ("revenue_planned", 0.0),
        ("revenue_realised", 0.0),
        ("stringing_span_completed", False),
        ("paying_out_complete", pd.NaT),
    ]
    for column, default in required_payload_columns:
        if column not in normalized.columns:
            normalized[column] = default
    if "completion_date" not in normalized.columns:
        normalized["completion_date"] = pd.NaT
    normalized["completion_date"] = pd.to_datetime(normalized["completion_date"], errors="coerce")

    if "tower_weight" in normalized.columns:
        base_span = pd.to_numeric(normalized["tower_weight"], errors="coerce")
        for alias in ("span (m)", "span_m", "length", "length_m"):
            if alias not in normalized.columns:
                normalized[alias] = base_span

    def _fill_completion_from(column_name: str) -> None:
        if column_name not in normalized.columns:
            return
        fallback = pd.to_datetime(normalized[column_name], errors="coerce")
        if fallback is None:
            return
        normalized["completion_date"] = normalized["completion_date"].fillna(fallback)

    _fill_completion_from("final_sag_complete")
    _fill_completion_from("paying_out_complete")
    return normalized, completion_pairs, issues


def compute_stringing_completion_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """
    Derive completed (project, span) pairs from the DPR-derived daily dataframe
    by looking for rows that have a valid Final Sag / F/S Completion date.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return set()

    col_lookup = {normalize_col_key(col): col for col in df.columns}

    def _resolve_column(candidates: Sequence[str]) -> str | None:
        for candidate in candidates:
            key = normalize_col_key(candidate)
            if key in col_lookup:
                return col_lookup[key]
        return None

    fs_col = _resolve_column(
        (
            "fs_complete_date",
            "final_sag_complete",
            "final sag complete",
            "f/s/ completion date",
            "f/s completion date",
        )
    )
    project_col = _resolve_column(("project_name", "project", "project title", "project code"))
    from_col = _resolve_column(("from_ap", "from ap", "start tower"))
    to_col = _resolve_column(("to_ap", "to ap", "end tower"))
    location_col = _resolve_column(("location_no", "location number", "span label", "span"))

    if fs_col is None or project_col is None:
        return set()

    relevant_cols = {project_col, fs_col}
    if from_col:
        relevant_cols.add(from_col)
    if to_col:
        relevant_cols.add(to_col)
    if location_col:
        relevant_cols.add(location_col)

    work = df[list(relevant_cols)].copy()
    work[fs_col] = pd.to_datetime(work[fs_col], errors="coerce")
    work = work.dropna(subset=[fs_col])
    if work.empty:
        return set()

    arrow = " \u2192 "
    completion_pairs: set[tuple[str, str]] = set()
    for idx in work.index:
        project_raw = work.at[idx, project_col]
        key_options = {
            normalize_lower(project_raw),
            compact_project_key(project_raw),
        }
        key_options = {key for key in key_options if key}
        if not key_options:
            continue

        start_label = normalize_text(work.at[idx, from_col]) if from_col else ""
        end_label = normalize_text(work.at[idx, to_col]) if to_col else ""
        derived_location = ""
        if start_label and end_label:
            derived_location = normalize_location(f"{start_label}{arrow}{end_label}")
        elif start_label or end_label:
            derived_location = normalize_location(start_label or end_label)
        elif location_col:
            derived_location = normalize_location(work.at[idx, location_col])
        if not derived_location:
            continue

        for key in key_options:
            completion_pairs.add((key, derived_location))
    return completion_pairs


__all__ = [
    "normalize_text",
    "normalize_lower",
    "normalize_col_key",
    "normalize_location",
    "compact_project_key",
    "infer_project_hint",
    "prepare_stringing_plan_frame",
    "compute_stringing_completion_pairs",
]
