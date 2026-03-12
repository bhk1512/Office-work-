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
from export_stringing_summary import _prepare_stringing_scope


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


def _compact(value: object) -> str:
    return ingest.normalize_project_code_key(value)


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
    if not isinstance(coverage_df, pd.DataFrame):
        coverage_df = pd.DataFrame()
    coverage_df = coverage_df.copy()
    if "reason_code" not in coverage_df.columns:
        coverage_df["reason_code"] = coverage_df.get("status", "")
    if "method_inference_rows" not in coverage_df.columns:
        coverage_df["method_inference_rows"] = 0
    if "assumption_notes" not in coverage_df.columns:
        coverage_df["assumption_notes"] = ""
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

    duplicate_identity_failures: list[str] = []
    if not coverage_df.empty:
        coverage_df["project_code_key"] = coverage_df.get("project_code", "").map(_compact)
        coverage_df["project_display_key"] = coverage_df.get("project_display", "").map(_compact)
        for code_key, group in coverage_df.groupby("project_code_key", dropna=False):
            if not code_key:
                continue
            display_keys = sorted({str(v).strip() for v in group["project_display_key"].tolist() if str(v).strip()})
            if len(display_keys) > 1:
                duplicate_identity_failures.append(
                    f"{code_key}: {', '.join(display_keys)}"
                )

    method_inferred_total_compiled = 0
    compiled_df = store.get_stringing_compiled()
    if isinstance(compiled_df, pd.DataFrame) and not compiled_df.empty and "method_inferred" in compiled_df.columns:
        method_inferred_total_compiled = int(
            pd.Series(compiled_df["method_inferred"]).fillna(False).astype(bool).sum()
        )
    method_inferred_total_coverage = int(
        pd.to_numeric(coverage_df.get("method_inference_rows", pd.Series(0)), errors="coerce").fillna(0).sum()
    ) if not coverage_df.empty else 0
    missing_inference_notes = int(
        coverage_df[
            (pd.to_numeric(coverage_df.get("method_inference_rows", 0), errors="coerce").fillna(0) > 0)
            & (~coverage_df.get("assumption_notes", "").fillna("").astype(str).str.strip().astype(bool))
        ].shape[0]
    ) if not coverage_df.empty else 0

    pch_unassigned_failures: list[str] = []
    project_info = store.get_project_info()
    if isinstance(project_info, pd.DataFrame) and not project_info.empty and isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
        try:
            scope = _prepare_stringing_scope(daily_df, project_info)
            if not scope.empty:
                scope["project_code_key"] = scope.get("project_code", "").map(_compact)
                pch_missing = scope[
                    scope["project_code_key"].astype(bool)
                    & scope.get("pch_display", "").fillna("").astype(str).str.strip().str.lower().eq("unassigned")
                ]
                if not pch_missing.empty:
                    known_pch = (
                        project_info[["project_code", "pch"]]
                        .dropna(subset=["project_code"])
                        .assign(project_code_key=lambda df: df["project_code"].map(_compact))
                    )
                    known_pch = known_pch[known_pch["pch"].fillna("").astype(str).str.strip().astype(bool)]
                    known_keys = set(known_pch["project_code_key"].tolist())
                    offenders = sorted({str(v).strip() for v in pch_missing["project_code_key"].tolist() if str(v).strip() and str(v).strip() in known_keys})
                    if offenders:
                        pch_unassigned_failures = offenders
        except Exception:
            pch_unassigned_failures = []

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
    if duplicate_identity_failures:
        failures.append(
            "Canonical project identity duplicates found in coverage: "
            + " | ".join(duplicate_identity_failures)
        )
    if method_inferred_total_compiled > 0 and method_inferred_total_coverage <= 0:
        failures.append(
            "Method inference applied in compiled data but coverage method_inference_rows is zero."
        )
    if missing_inference_notes > 0:
        failures.append(
            f"Coverage rows with inferred methods but empty assumption_notes: {missing_inference_notes}."
        )
    if pch_unassigned_failures:
        failures.append(
            "Projects with known PCH still resolve as Unassigned in prepared scope: "
            + ", ".join(pch_unassigned_failures)
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
    print(f"[verify] method_inferred_rows(compiled): {method_inferred_total_compiled}")
    print(f"[verify] method_inference_rows(coverage): {method_inferred_total_coverage}")
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
