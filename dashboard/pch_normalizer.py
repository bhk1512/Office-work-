"""Utilities to normalize PCH names to canonical forms.

This centralizes the mapping so all views use consistent labels.
"""
from __future__ import annotations

import re
from typing import Dict


# Canonical PCH display names
CANONICAL_PCH_PRIMARY = (
    "Mr. Sanjay Kumar Gupta",
    "Mr. Arun Felbin",
    "Mr. Dhiraj Vashisth",
    "Mr. Nikhilesh Kumar Gupta",
)


def _compact_key(value: object) -> str:
    """Lowercase, remove honorifics and non-alphanumerics, then compact.

    Examples:
    - "Mr. S K Gupta" -> "skgupta"
    - "Sanjay K. Gupta" -> "sanjaykgupta"
    - "Nikhilesh Kumar Gupta" -> "nikhileshkumargupta"
    """
    if value is None:
        return ""
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return ""
    # Remove common honorifics/titles
    for hon in ("mr", "mr.", "shri", "sri", "er", "er."):
        s = re.sub(rf"\b{re.escape(hon)}\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)  # keep only a-z0-9
    return s


# Build alias map using compact keys for robust matching
_ALIASES: Dict[str, str] = {}

def _add_alias(canonical: str, *aliases: str) -> None:
    key = _compact_key(canonical)
    _ALIASES[key] = canonical
    for a in aliases:
        _ALIASES[_compact_key(a)] = canonical


# Mr. Sanjay Kumar Gupta and common variants
_add_alias(
    "Mr. Sanjay Kumar Gupta",
    "Sanjay Kumar Gupta",
    "Sanjay K Gupta",
    "Sanjay Gupta",
    "S. K. Gupta",
    "S K Gupta",
    "SK Gupta",
    "Sanjay Kr Gupta"
)

# Mr. Arun Felbin (no known variants beyond honorific/casing)
_add_alias(
    "Mr. Arun Felbin",
    "Arun Felbin",
    "A Felbin"
)

# Mr. Dhiraj Vashisth (no known variants beyond honorific/casing)
_add_alias(
    "Mr. Dhiraj Vashisth",
    "Dhiraj Vashisth",
    "D Vashisth"
)

# Mr. Nikhilesh Kumar Gupta and common variants
_add_alias(
    "Mr. Nikhilesh Kumar Gupta",
    "Nikhilesh Kumar Gupta",
    "Nikhilesh K Gupta",
    "Nikhilesh Gupta",
    "N. K. Gupta",
    "N K Gupta",
    "NK Gupta",
    "Nikhilesh Kr Gupta"
)


def normalize_pch(value: object) -> str:
    """Return canonical PCH name if known, else cleaned original.

    - Empty/null-ish inputs return an empty string.
    - Known variants map to their canonical form.
    - Unknown inputs are returned stripped (original casing preserved if possible).
    """
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return ""
    key = _compact_key(raw)
    if not key:
        return ""
    canonical = _ALIASES.get(key)
    return canonical or raw


__all__ = ["normalize_pch", "CANONICAL_PCH_PRIMARY"]

