#!/usr/bin/env python3
"""Verification checks for stringing project coverage + artifact integrity."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from dashboard import stringing_ingest as ingest
from dashboard.config import AppConfig, configure_logging
from dashboard.state import AppDataStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify stringing coverage completeness and artifact health."
    )
    parser.add_argument("--data-path", type=Path, help="Override erection dataset root.")
    parser.add_argument("--stringing-data-path", type=Path, help="Override stringing dataset root.")
    return parser.parse_args()


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pipeline_config.json").exists():
            return candidate
    return here


def _resolve_raw_root(repo_root: Path) -> Path:
    cfg_path = repo_root / "pipeline_config.json"
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                input_dir = payload.get("input_directory")
                if input_dir:
                    resolved = (repo_root / str(input_dir)).resolve()
                    if resolved.exists():
                        return resolved
        except Exception:
            pass
    fallback = (repo_root / "Raw Data" / "DPRs").resolve()
    return fallback


def _normalize_project_keys(values: Iterable[object]) -> set[str]:
    keys: set[str] = set()
    for value in values:
        key = ingest.normalize_project_code_key(value)
        if key:
            keys.add(key)
    return keys


def _validate_parquet(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        if path.stat().st_size < 12:
            return False, "too small"
        with duckdb.connect(database=":memory:") as con:
            con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()
        return True, "ok"
    except Exception as exc:
        return False, f"unreadable ({type(exc).__name__}: {exc})"


def main() -> int:
    args = _parse_args()
    configure_logging()

    config = AppConfig()
    if args.data_path:
        config = replace(config, data_path=Path(args.data_path).expanduser())
    if args.stringing_data_path:
        config = replace(config, stringing_data_path=Path(args.stringing_data_path).expanduser())
    config.validate()

    store = AppDataStore(config)
    store.bootstrap(config)

    repo_root = _repo_root()
    raw_root = _resolve_raw_root(repo_root)
    sheet_config = ingest.load_stringing_sheet_config(raw_root, repo_root=repo_root)
    configured_keys = set(sheet_config.keys())

    coverage_df = store.get_stringing_coverage()
    coverage_code_keys = _normalize_project_keys(coverage_df.get("project_code", pd.Series(dtype=object)))
    coverage_name_keys = _normalize_project_keys(coverage_df.get("project_display", pd.Series(dtype=object)))
    coverage_keys = coverage_code_keys | coverage_name_keys

    missing_from_coverage = sorted(configured_keys - coverage_keys)

    daily_df = store.get_stringing()
    scope_keys = set()
    if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
        if "project_code" in daily_df.columns:
            scope_keys |= _normalize_project_keys(daily_df["project_code"])
        if "project_name" in daily_df.columns:
            scope_keys |= _normalize_project_keys(daily_df["project_name"])
        if "project_display" in daily_df.columns:
            scope_keys |= _normalize_project_keys(daily_df["project_display"])
    scope_keys |= coverage_keys
    missing_from_scope = sorted(configured_keys - scope_keys)

    stringing_root = Path(config.stringing_data_path).expanduser().resolve()
    artifact_root = stringing_root.parent if stringing_root.is_file() else stringing_root
    parquet_checks = {
        "StringingCompiled.parquet": _validate_parquet(artifact_root / "StringingCompiled.parquet"),
        "StringingDaily.parquet": _validate_parquet(artifact_root / "StringingDaily.parquet"),
        "StringingCoverage.parquet": _validate_parquet(artifact_root / "StringingCoverage.parquet"),
    }

    workbook_path = artifact_root / "StringingCompiled_Output.xlsx"
    workbook_ok = False
    workbook_reason = "missing"
    if workbook_path.exists():
        try:
            with pd.ExcelFile(workbook_path) as xl:
                workbook_ok = "StringingCoverage" in xl.sheet_names
                workbook_reason = "ok" if workbook_ok else "missing StringingCoverage sheet"
        except Exception as exc:
            workbook_reason = f"unreadable ({type(exc).__name__}: {exc})"

    failures: list[str] = []
    if missing_from_coverage:
        failures.append(
            "Configured projects missing from StringingCoverage: "
            + ", ".join(sorted(missing_from_coverage))
        )
    if missing_from_scope:
        failures.append(
            "Configured projects missing from dashboard/export scope union: "
            + ", ".join(sorted(missing_from_scope))
        )
    for artifact_name, (ok, reason) in parquet_checks.items():
        if not ok:
            failures.append(f"{artifact_name} check failed: {reason}")
    if not workbook_ok:
        failures.append(f"StringingCompiled_Output.xlsx check failed: {workbook_reason}")

    print(f"[verify] repo_root: {repo_root}")
    print(f"[verify] raw_root: {raw_root}")
    print(f"[verify] configured projects: {len(configured_keys)}")
    print(f"[verify] coverage rows: {0 if coverage_df is None else len(coverage_df)}")
    for artifact_name, (_, reason) in parquet_checks.items():
        print(f"[verify] {artifact_name}: {reason}")
    print(f"[verify] workbook: {workbook_reason}")

    if failures:
        print("[verify] FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[verify] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
