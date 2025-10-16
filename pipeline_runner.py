import argparse
import json
import os
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
import pandas as pd

from erection_compiled_to_daily_new import run_pipeline
from microplan_compile import compile_microplans_to_workbook

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Dict[str, Any] = {
    "input_directory": "Raw Data/DPRs",
    "microplan_directory": "Raw Data/Micro Plans",
    "output_file": "ErectionCompiled_Output.xlsx",
    "pipeline_extra_args": [],
    "dash_host": "0.0.0.0",
    "dash_port": 8050,
    "dash_debug": False,
}


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Configuration file {path} must contain a JSON object.")
    return data


def _resolve_path(value: Optional[str], base: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _normalise_files(raw: Optional[Iterable[str]], base: Path) -> Optional[List[Path]]:
    if raw is None:
        return None
    files: List[Path] = []
    for item in raw:
        resolved = _resolve_path(str(item), base)
        if resolved is None:
            continue
        files.append(resolved)
    return files if files else None


PARQUET_SHEETS: tuple[str, ...] = (
    "ProdDailyExpandedSingles",
    "ProdDailyExpanded",
    "RawData",
    "ProjectBaselines",
    "ProjectBaselinesMonthly",
    "ProjectDetails",
    "MicroPlanResponsibilities",
    "MicroPlanIndex",
)


def _write_parquet(df: pd.DataFrame, destination: Path) -> None:
    """Persist *df* to *destination* using DuckDB for consistent parquet output."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with duckdb.connect(database=":memory:") as con:
        con.register("df_to_write", df)
        con.execute(
            "COPY df_to_write TO ? (FORMAT 'parquet', COMPRESSION 'zstd')",
            [str(destination)],
        )


def export_workbook_to_parquet(workbook_path: Path, sheets: Iterable[str] | None = None) -> Path:
    """Export selected workbook sheets to parquet files alongside the workbook."""

    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook '{workbook_path}' does not exist.")

    target_dir = workbook_path.parent / f"{workbook_path.stem}_parquet"
    # Recreate the parquet directory to ensure a fresh dataset (no stale files)
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
        except Exception as exc:
            print(f"[pipeline] Warning: failed to clear parquet dir {target_dir}: {exc}")
    target_dir.mkdir(parents=True, exist_ok=True)

    sheet_list = list(sheets) if sheets is not None else list(PARQUET_SHEETS)
    exported: list[str] = []
    with pd.ExcelFile(workbook_path) as workbook:
        available = set(workbook.sheet_names)
        for sheet in sheet_list:
            if sheet not in available:
                continue
            df = workbook.parse(sheet_name=sheet)
            if df is None:
                continue
            destination = target_dir / f"{sheet}.parquet"
            _write_parquet(df, destination)
            exported.append(sheet)

    if not exported:
        print(f"[pipeline] No matching sheets were exported from {workbook_path}.")
    else:
        print(f"[pipeline] Exported sheets to parquet: {', '.join(exported)}")

    return target_dir


def _reload_dashboard_data(dashboard_module: Any, workbook_path: Path) -> None:
    """Reload the dashboard dataframe and recompute derived fields."""
    df = dashboard_module.load_daily(workbook_path)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    if hasattr(dashboard_module, "set_df_day"):
        dashboard_module.set_df_day(df)
    else:
        dashboard_module.df_day = df




def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the erection compiled pipeline then launch the dashboard."
    )
    parser.add_argument("--input", help="Folder containing the source Excel files.")
    parser.add_argument("--files", nargs="+", help="Explicit list of Excel files (overrides --input).")
    parser.add_argument("--output", help="Destination Excel workbook path.")
    parser.add_argument("--skip-compile", action="store_true", help="Launch dashboard without re-running the pipeline.")
    parser.add_argument(
        "--config",
        default="pipeline_config.json",
        help="Configuration file relative to the project root (default: pipeline_config.json)."
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Do not start the Dash dev server after pipeline; just prepare app module.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        help="Additional CLI arguments to forward to the pipeline script."
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = _resolve_path(args.config, BASE_DIR)
    if config_path is None:
        raise SystemExit("Unable to resolve configuration path.")

    config = DEFAULT_CONFIG.copy()
    config.update(_load_config(config_path))

    cli_files = args.files
    config_files = config.get("files") if isinstance(config.get("files"), list) else None
    files = cli_files or config_files

    cli_input = args.input
    config_input = config.get("input_directory")
    config_microplan_input = config.get("microplan_directory")

    if files and cli_input:
        parser.error("Provide either --files or --input, not both.")
    if files and config_input and cli_files is None:
        parser.error("Configuration specifies both files and input_directory; please keep only one.")

    env_input = os.getenv("PIPELINE_INPUT_DIR")
    env_output = os.getenv("PIPELINE_OUTPUT_FILE")
    env_microplan_input = os.getenv("MICROPLAN_INPUT_DIR")

    input_path = cli_input or env_input or config_input
    output_path = args.output or env_output or config.get("output_file")
    microplan_input = env_microplan_input or config_microplan_input

    if input_path is None and not files:
        parser.error("An input directory or explicit file list is required for the pipeline.")
    if output_path is None:
        parser.error("An output path is required for the pipeline.")

    resolved_files = _normalise_files(files, BASE_DIR)
    resolved_input = _resolve_path(input_path, BASE_DIR) if input_path else None
    resolved_output = _resolve_path(output_path, BASE_DIR)
    resolved_microplan_input = _resolve_path(microplan_input, BASE_DIR) if microplan_input else None

    extra_args: List[str] = []
    if isinstance(config.get("pipeline_extra_args"), list):
        extra_args.extend(str(v) for v in config["pipeline_extra_args"])
    if args.extra_args:
        extra_args.extend(args.extra_args)



    if not args.skip_compile:
        # Ensure fresh outputs on every compile run
        # 1) Remove any existing compiled workbook
        # 2) Remove any existing parquet dataset directory
        if resolved_output:
            try:
                if resolved_output.exists() and resolved_output.is_file():
                    print(f"[pipeline] Removing existing workbook: {resolved_output}")
                    resolved_output.unlink()
            except Exception as exc:
                print(f"[pipeline] Warning: failed to remove workbook {resolved_output}: {exc}")

            try:
                parquet_dir = resolved_output.parent / f"{resolved_output.stem}_parquet"
                if parquet_dir.exists() and parquet_dir.is_dir():
                    print(f"[pipeline] Removing existing parquet dataset: {parquet_dir}")
                    shutil.rmtree(parquet_dir)
            except Exception as exc:
                print(f"[pipeline] Warning: failed to remove parquet dir {parquet_dir}: {exc}")

        if resolved_input:
            print(f"[pipeline] Compiling from folder: {resolved_input}")
        if resolved_files:
            print("[pipeline] Compiling from files\n  - " + "\n  - ".join(str(p) for p in resolved_files))
        print(f"[pipeline] Writing output to: {resolved_output}")
        run_pipeline(
            input_path=resolved_input,
            files=[str(p) for p in resolved_files] if resolved_files else None,
            output_path=str(resolved_output) if resolved_output else None,
            extra_args=extra_args,
        )
                # --- NEW: Compile Micro Plan responsibilities into the same workbook ---
        # Prefer the input folder; if the user passed explicit files, derive a common parent
        if resolved_microplan_input:
            micro_input_dir = str(resolved_microplan_input)
        elif resolved_files:
            common_parent = os.path.commonpath([os.path.dirname(str(p)) for p in resolved_files])
            micro_input_dir = common_parent
        elif resolved_input:
            micro_input_dir = str(resolved_input)
        else:
            micro_input_dir = None

        if micro_input_dir:
            print(f"[pipeline] MicroPlan: scanning '{micro_input_dir}' and writing to '{resolved_output}'")
            compile_microplans_to_workbook(
                input_dir=micro_input_dir,
                output_path=str(resolved_output),
            )
        else:
            print("[pipeline] MicroPlan: no input directory configured; skipping.")

        if resolved_output:
            try:
                parquet_dir = export_workbook_to_parquet(resolved_output)
            except Exception as exc:
                print(f"[pipeline] Failed to export parquet dataset: {exc}")
                parquet_dir = None
        else:
            parquet_dir = None
    else:
        print("[pipeline] Skipping compilation step as requested.")
        parquet_dir = None

    dataset_path: Path | None = resolved_output
    if parquet_dir:
        dataset_path = Path(parquet_dir)
    elif resolved_output:
        candidate_dir = resolved_output.parent / f"{resolved_output.stem}_parquet"
        if candidate_dir.exists():
            dataset_path = candidate_dir
    if dataset_path:
        print(f"[pipeline] Using dataset path: {dataset_path}")
    else:
        print("[pipeline] Dataset path unresolved; using dashboard defaults.")

    dash_host = os.getenv("DASH_HOST", config.get("dash_host", "0.0.0.0"))
    dash_port = int(os.getenv("DASH_PORT", config.get("dash_port", 8050)))
    dash_debug = os.getenv("DASH_DEBUG", str(config.get("dash_debug", False))).lower() in ("1", "true", "yes")


    print("[dashboard] Loading Dash app...")
    dashboard = import_module("app")

    if dataset_path is not None:
        dataset_target = Path(dataset_path)
    else:
        dataset_target = Path(dashboard.DATA_PATH)

    new_config = dashboard.AppConfig(data_path=dataset_target)
    dashboard.CONFIG = new_config
    dashboard.DATA_PATH = dataset_target
    dashboard.app = dashboard.create_app(new_config)
    dashboard.server = dashboard.app.server

    _reload_dashboard_data(dashboard, dashboard.DATA_PATH)
    print(f"[dashboard] Dataset path configured: {dashboard.DATA_PATH}")

    if not args.no_serve:
        print(f"[dashboard] Starting server on http://{dash_host}:{dash_port}")
        dashboard.app.run_server(host=dash_host, port=dash_port, debug=dash_debug)
    else:
        print("[dashboard] Skipping dev server (--no-serve). Data reloaded and app module ready.")


if __name__ == "__main__":
    main()






