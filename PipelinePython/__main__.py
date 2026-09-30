"""Operator CLI for the canonical local biosignal pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "data.yaml"
TRAINING_CHANNELS = ("ppg_green", "ppg_ir", "ppg_red")
FLOW_RUNNERS = {
    "data": PACKAGE_ROOT / "01_data_preprocess" / "run.py",
    "training-gold": PACKAGE_ROOT / "01_data_preprocess" / "training_gold.py",
    "peaks": PACKAGE_ROOT / "02_peak_to_peak" / "run.py",
    "stress-onnx": PACKAGE_ROOT / "03_stress_onnx" / "run.py",
    "all": PACKAGE_ROOT / "03_stress_onnx" / "run.py",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m PipelinePython",
        description="Canonical local biosignal data, peak, and stress-ONNX workflows.",
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="raw audit through Silver/Gold preparation")
    _add_run_options(data)
    data.add_argument("--participant", help="optional participant-scoped data run")

    training_gold = commands.add_parser(
        "training-gold",
        help="derive model windows from one immutable canonical Silver run",
    )
    _add_run_options(training_gold)
    training_gold.add_argument("--source-run-id", required=True)

    peaks = commands.add_parser("peaks", help="ECG-PPG peak-to-peak validation")
    _add_run_options(peaks)

    stress = commands.add_parser(
        "stress-onnx", help="participant-safe training and gated ONNX export"
    )
    _add_run_options(stress)
    stress.add_argument("--sensor", choices=TRAINING_CHANNELS)
    stress.add_argument(
        "--export-mode",
        choices=("production", "research", "internal"),
        default="production",
        help="production is legacy; research is development-only; internal defers the one-time internal test",
    )

    internal_publish = commands.add_parser(
        "publish-internal",
        help="evaluate frozen ID-29..ID-32 once and export internal Android ONNX",
    )
    _add_run_options(internal_publish, require_run_id=True)
    internal_publish.add_argument("--sensor", choices=TRAINING_CHANNELS)

    readiness = commands.add_parser(
        "android-readiness",
        help="verify current-run research Android parity without rerunning the pipeline",
    )
    readiness.add_argument("--run-id", required=True)
    readiness.add_argument("--apk", type=Path, required=True)
    readiness.add_argument("--receipt", action="append", type=Path, default=[])
    readiness.add_argument(
        "--device-status",
        choices=("READY", "OFFLINE", "INSTRUMENTATION_FAILED"),
        required=True,
    )

    internal_readiness = commands.add_parser(
        "verify-internal-android",
        help="verify a run-scoped internal Android parity receipt without rerunning data stages",
    )
    internal_readiness.add_argument("--run-id", required=True)
    internal_readiness.add_argument("--apk", type=Path, required=True)
    internal_readiness.add_argument("--receipt", action="append", type=Path, default=[])
    internal_readiness.add_argument(
        "--device-status",
        choices=("READY", "OFFLINE", "INSTRUMENTATION_FAILED"),
        required=True,
    )

    live_capture = commands.add_parser(
        "verify-live-ppg-capture",
        help="compare an opt-in real Watch capture with the frozen Android run contract",
    )
    live_capture.add_argument("--run-id", required=True)
    live_capture.add_argument("--capture-dir", type=Path, required=True)
    live_capture.add_argument(
        "--receipt",
        type=Path,
        help="optional new run-scoped receipt path; written only after every eligible window passes",
    )

    complete = commands.add_parser("all", help="run data, peaks, then stress-onnx")
    _add_run_options(complete)
    complete.add_argument(
        "--export-mode",
        choices=("production", "research", "internal"),
        default="research",
        help="export mode for the final stress model stage",
    )
    complete.add_argument(
        "--smoke",
        action="store_true",
        help="run complete flow only through validation; never train or export",
    )
    return parser


def _add_run_options(
    parser: argparse.ArgumentParser, *, require_run_id: bool = False
) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=require_run_id)
    parser.add_argument("--resume", action="store_true")


def _load_runner(flow: str) -> Callable:
    path = FLOW_RUNNERS[flow]
    spec = importlib.util.spec_from_file_location(f"_pipeline_flow_{flow}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(module, ModuleType) or not hasattr(module, "run"):
        raise TypeError(f"Workflow runner has no run() function: {path}")
    return module.run


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if arguments.command == "android-readiness":
        from .common.android_readiness import write_android_readiness_report

        report, artifact_path, report_path = write_android_readiness_report(
            PROJECT_ROOT,
            arguments.run_id,
            apk_path=arguments.apk,
            receipt_paths=arguments.receipt,
            device_connection_status=arguments.device_status,
        )
        report["artifact_path"] = str(artifact_path)
        report["report_path"] = str(report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "verify-internal-android":
        from .common.android_readiness import write_internal_android_readiness_report

        report, artifact_path, report_path = write_internal_android_readiness_report(
            PROJECT_ROOT,
            arguments.run_id,
            apk_path=arguments.apk,
            receipt_paths=arguments.receipt,
            device_connection_status=arguments.device_status,
        )
        report["artifact_path"] = str(artifact_path)
        report["report_path"] = str(report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "verify-live-ppg-capture":
        from .common.live_ppg_capture_verifier import verify_live_ppg_capture

        receipt_path = arguments.receipt or (
            PROJECT_ROOT
            / "data"
            / "artifacts"
            / "runs"
            / arguments.run_id
            / "export"
            / "internal"
            / "device_parity"
            / "sensor_path"
            / "live_ppg_sensor_parity_receipt.json"
        )
        report = verify_live_ppg_capture(
            PROJECT_ROOT,
            arguments.run_id,
            arguments.capture_dir,
            receipt_path=receipt_path,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") == "GO" else 2
    if arguments.command == "publish-internal":
        from .common.cli import _with_isolated_run
        from .common.settings import load_pipeline_config
        from .common.workflows.stress_onnx import publish_internal_models

        config_path = arguments.config
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = _with_isolated_run(
            load_pipeline_config(config_path.resolve()), arguments.run_id
        )
        result = publish_internal_models(PROJECT_ROOT, config, sensor=arguments.sensor)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    flow = "all" if arguments.command == "all" else arguments.command
    if arguments.command != "all":
        arguments.smoke = False
    if arguments.command not in {"stress-onnx", "all"}:
        arguments.sensor = None
        arguments.export_mode = "production"
    if arguments.command != "data":
        arguments.participant = None
    result = _load_runner(flow)(arguments, project_root=PROJECT_ROOT)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
