from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

_DEFAULT_FILE_NAME = "Completed Projects.xlsx"
_DEFAULT_COLUMN_NAME = "Project Code"
_PROJECT_RE = re.compile(r"\b(TA|TB)\s*[-_ ]?\s*(\d{2,4})\b", flags=re.IGNORECASE)


def normalize_project_code_key(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    match = _PROJECT_RE.search(text)
    if match:
        return f"{match.group(1).lower()}{match.group(2)}"
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _resolve_completed_projects_path(raw_root: Path | None, repo_root: Path | None) -> Path | None:
    if raw_root is not None:
        candidate = Path(raw_root) / _DEFAULT_FILE_NAME
        if candidate.exists():
            return candidate
    if repo_root is not None:
        fallback = Path(repo_root) / "Raw Data" / _DEFAULT_FILE_NAME
        if fallback.exists():
            return fallback
    return None


def load_completed_project_keys(raw_root: Path | None, repo_root: Path | None) -> set[str]:
    path = _resolve_completed_projects_path(raw_root, repo_root)
    if path is None:
        LOGGER.warning(
            "Completed project exclusion list not found (expected '%s' under raw data). Continuing without exclusions.",
            _DEFAULT_FILE_NAME,
        )
        return set()

    try:
        df = pd.read_excel(path)
    except Exception as exc:
        LOGGER.warning(
            "Completed project exclusion list unreadable at '%s': %s. Continuing without exclusions.",
            path,
            exc,
        )
        return set()

    column_name = None
    normalized_columns = {
        re.sub(r"[^a-z0-9]+", "", str(col).strip().lower()): col
        for col in df.columns
    }
    for candidate in ("projectcode", "projectcodes"):
        if candidate in normalized_columns:
            column_name = normalized_columns[candidate]
            break

    if column_name is None:
        LOGGER.warning(
            "Completed project exclusion list at '%s' is missing '%s'. Continuing without exclusions.",
            path,
            _DEFAULT_COLUMN_NAME,
        )
        return set()

    keys: set[str] = set()
    for value in df[column_name].tolist():
        key = normalize_project_code_key(value)
        if key:
            keys.add(key)
    return keys


def is_completed_project(project_code: object, completed_keys: set[str]) -> bool:
    if not completed_keys:
        return False
    return normalize_project_code_key(project_code) in completed_keys
