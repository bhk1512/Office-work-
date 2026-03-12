from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path


_PROJECT_CODE_RE = re.compile(r"\b(T[A-Z])\s*[-_. ]?\s*(\d{3,4})\b", re.IGNORECASE)
_CANONICAL_SUFFIX_RE = re.compile(
    r"\s*-\s*dpr\s*-\s*(20\d{2}-\d{2}-\d{2})(?P<ext>\.[^.]+)?\s*$",
    re.IGNORECASE,
)
_LINE_TOKEN_RE = re.compile(r"\[(?P<line>[^\[\]]+)\]")
_GENERIC_LINE_TOKENS = {
    "compiled",
    "compilled",
    "compliled",
    "compile",
    "stringing",
    "erection",
}


def _compact(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_line_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _split_csv_tokens(value: object, *, keep_empty: bool) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = [str(part).strip() for part in re.split(r"[;,]", text)]
    if keep_empty:
        return parts
    return [part for part in parts if part]


def infer_line_name_from_sheet_name(sheet_name: object, mode: str = "") -> str:
    raw = normalize_line_name(sheet_name)
    if not raw:
        return ""

    text = raw
    mode_key = str(mode or "").strip().lower()
    tokens: list[str]
    if mode_key in {"erection", "stringing"}:
        tokens = [f"{mode_key} compiled", mode_key]
    else:
        tokens = ["erection compiled", "erection", "stringing compiled", "stringing"]

    for token in tokens:
        escaped = re.escape(token)
        text = re.sub(rf"^\s*{escaped}\s*[-_:/]*\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"\s*[-_:/]*\s*{escaped}\s*$", "", text, flags=re.IGNORECASE)

    text = re.sub(r"^[\s\-_:./]+", "", text)
    text = re.sub(r"[\s\-_:./]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if compact in {"compiled", "compilled", "compliled", "compile"}:
        return ""
    if normalize_line_name(text).lower() in _GENERIC_LINE_TOKENS:
        return ""
    return text


def parse_sheet_line_entries(
    raw_sheet_names: object,
    raw_line_names: object,
    mode: str,
    *,
    infer_from_sheet_name: bool = True,
) -> list[dict[str, str]]:
    sheet_names = _split_csv_tokens(raw_sheet_names, keep_empty=False)
    if not sheet_names:
        return []

    line_overrides = _split_csv_tokens(raw_line_names, keep_empty=True)
    entries: list[dict[str, str]] = []
    for idx, sheet_name in enumerate(sheet_names):
        override = line_overrides[idx] if idx < len(line_overrides) else ""
        line_name = normalize_line_name(override)
        source = "config" if line_name else "inferred"
        if not line_name and infer_from_sheet_name:
            line_name = infer_line_name_from_sheet_name(sheet_name, mode=mode)
            source = "inferred" if line_name else ""
        elif not line_name:
            source = ""
        entries.append(
            {
                "sheet_name": sheet_name,
                "line_name": line_name,
                "line_name_source": source,
            }
        )
    return entries


def _normalize_project_code(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = _PROJECT_CODE_RE.search(text)
    if match:
        return f"{match.group(1).upper()} {int(match.group(2))}"
    return re.sub(r"\s+", " ", text)


def extract_base_project_code(value: object) -> str:
    """Return canonical base project code (e.g., 'TB 501') when present."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = _PROJECT_CODE_RE.search(text)
    if not match:
        return ""
    return f"{match.group(1).upper()} {int(match.group(2))}"


def build_project_rollup_identity(
    project_code: object,
    project_display: object = "",
    project_name: object = "",
) -> dict[str, str]:
    """Return consolidated project identity for rollup-level reporting."""
    code = extract_base_project_code(project_code)
    if not code:
        code = extract_base_project_code(project_display)
    if not code:
        code = extract_base_project_code(project_name)

    display = re.sub(r"\s+", " ", str(project_display or "").strip())
    name = re.sub(r"\s+", " ", str(project_name or "").strip())
    fallback = display or name or _normalize_project_code(project_code)
    rollup_display = code or fallback
    variant_display = display or name or code or fallback
    return {
        "project_rollup_display": rollup_display,
        "project_rollup_key": _compact(rollup_display),
        "project_variant_display": variant_display,
        "project_base_code": code,
    }


def _sanitize_line_for_filename(value: object) -> str:
    text = normalize_line_name(value)
    if not text:
        return ""
    text = re.sub(r'[<>:"/\\|?*]+', " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_project_display(
    project_code: object,
    line_name: object,
    fallback_name: object = "",
) -> str:
    code = _normalize_project_code(project_code)
    fallback = re.sub(r"\s+", " ", str(fallback_name or "").strip())
    line = normalize_line_name(line_name)
    base = code or fallback
    if not base:
        return ""
    if line:
        return f"{base} - {line}"
    return base


def build_project_scope_key(
    project_code: object,
    line_name: object,
    fallback_name: object = "",
) -> str:
    code = _normalize_project_code(project_code)
    fallback = re.sub(r"\s+", " ", str(fallback_name or "").strip())
    line = normalize_line_name(line_name)
    base = code or fallback
    if not base:
        return ""
    if line:
        return _compact(f"{base}::{line}")
    return _compact(base)


def canonical_dpr_filename(
    project_code: str,
    report_date: dt.date,
    ext: str,
    line_name: str = "",
) -> str:
    code = _normalize_project_code(project_code)
    safe_line = _sanitize_line_for_filename(line_name)
    prefix = code
    if safe_line:
        prefix = f"{prefix} [{safe_line}]"
    return f"{prefix} - DPR - {report_date:%Y-%m-%d}{ext}"


def parse_project_identity_from_filename(name: str) -> dict[str, str]:
    raw = str(name or "")
    stem = Path(raw).name
    stem_no_ext, ext = os.path.splitext(stem)
    base_text = stem_no_ext
    suffix_match = _CANONICAL_SUFFIX_RE.search(stem)
    if suffix_match:
        # Strip the canonical suffix from the original name without extension drift.
        base_text = stem[: suffix_match.start()].strip()
    line_match = _LINE_TOKEN_RE.search(base_text)
    line_name = normalize_line_name(line_match.group("line")) if line_match else ""
    cleaned = _LINE_TOKEN_RE.sub(" ", base_text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    project_code = _normalize_project_code(cleaned)
    fallback = re.sub(r"\s+", " ", cleaned).strip()
    if not project_code:
        fallback = re.sub(r"\s+", " ", Path(raw).stem).strip()
    project_display = build_project_display(project_code, line_name, fallback)
    return {
        "project_code": project_code or fallback,
        "line_name": line_name,
        "project_display": project_display or fallback,
        "project_scope_key": build_project_scope_key(project_code, line_name, fallback),
        "extension": ext.lower(),
    }
