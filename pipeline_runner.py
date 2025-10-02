import argparse
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from erection_compiled_to_daily_new import run_pipeline

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Dict[str, Any] = {
    "input_directory": "Test",
    "output_file": "ErectionCompiled_Output_testRun.xlsx",
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

    if files and cli_input:
        parser.error("Provide either --files or --input, not both.")
    if files and config_input and cli_files is None:
        parser.error("Configuration specifies both files and input_directory; please keep only one.")

    env_input = os.getenv("PIPELINE_INPUT_DIR")
    env_output = os.getenv("PIPELINE_OUTPUT_FILE")

    input_path = cli_input or env_input or config_input
    output_path = args.output or env_output or config.get("output_file")

    if input_path is None and not files:
        parser.error("An input directory or explicit file list is required for the pipeline.")
    if output_path is None:
        parser.error("An output path is required for the pipeline.")

    resolved_files = _normalise_files(files, BASE_DIR)
    resolved_input = _resolve_path(input_path, BASE_DIR) if input_path else None
    resolved_output = _resolve_path(output_path, BASE_DIR)

    extra_args: List[str] = []
    if isinstance(config.get("pipeline_extra_args"), list):
        extra_args.extend(str(v) for v in config["pipeline_extra_args"])
    if args.extra_args:
        extra_args.extend(args.extra_args)

    if not args.skip_compile:
        if resolved_input:
            print(f"[pipeline] Compiling from folder: {resolved_input}")
        if resolved_files:
            print("[pipeline] Compiling from files:\n  - " + "\n  - ".join(str(p) for p in resolved_files))
        print(f"[pipeline] Writing output to: {resolved_output}")
        run_pipeline(
            input_path=resolved_input,
            files=[str(p) for p in resolved_files] if resolved_files else None,
            output_path=str(resolved_output) if resolved_output else None,
            extra_args=extra_args,
        )
    else:
        print("[pipeline] Skipping compilation step as requested.")

    dash_host = os.getenv("DASH_HOST", config.get("dash_host", "0.0.0.0"))
    dash_port = int(os.getenv("DASH_PORT", config.get("dash_port", 8050)))
    dash_debug = os.getenv("DASH_DEBUG", str(config.get("dash_debug", False))).lower() in ("1", "true", "yes")

    print("[dashboard] Loading Dash app...")
    dashboard = import_module("app")

    if resolved_output:
        dashboard.DATA_PATH = Path(resolved_output)
        # dashboard.df_day = dashboard.load_daily(dashboard.DATA_PATH)
        _reload_dashboard_data(dashboard, dashboard.DATA_PATH)
    else:
        # dashboard.df_day = dashboard.load_daily(dashboard.DATA_PATH)
        _reload_dashboard_data(dashboard, dashboard.DATA_PATH)

    print(f"[dashboard] Starting server on http://{dash_host}:{dash_port}")
    dashboard.app.run_server(host=dash_host, port=dash_port, debug=dash_debug)


if __name__ == "__main__":
    main()
