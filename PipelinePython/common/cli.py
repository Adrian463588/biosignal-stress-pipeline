"""Local v4 command-line interface based on PRD and AGENTS.md."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .local_pipeline import LocalBiosignalPipeline
from .settings import TRAINING_CHANNELS, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PPG_SENSORS = TRAINING_CHANNELS


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if "--phase" in resolved_argv:
        return _parse_phase_args(resolved_argv)

    parser = argparse.ArgumentParser(description="Local biosignal pipeline v4")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    # ingest
    cmd_ingest = commands.add_parser("ingest")
    cmd_ingest.add_argument("--source")
    cmd_ingest.add_argument("--output")
    cmd_ingest.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # sqa
    cmd_sqa = commands.add_parser("sqa")
    cmd_sqa.add_argument("--input")
    cmd_sqa.add_argument("--output")
    cmd_sqa.add_argument("--visualize", action="store_true")
    cmd_sqa.add_argument("--config", default="PipelinePython/configs/data.yaml")

    for stage_name in ("sync-raw", "segment", "preprocess", "extract-features"):
        stage = commands.add_parser(stage_name)
        stage.add_argument("--participant")
        stage.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # eda
    cmd_eda = commands.add_parser("eda")
    cmd_eda.add_argument("--input")
    cmd_eda.add_argument("--output")
    cmd_eda.add_argument("--visualize", action="store_true")
    cmd_eda.add_argument("--stage", choices=["raw", "silver", "gold"], default="silver")
    cmd_eda.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # peaks
    cmd_peaks = commands.add_parser("peaks")
    cmd_peaks.add_argument("--input")
    cmd_peaks.add_argument("--output")
    cmd_peaks.add_argument("--visualize", action="store_true")
    cmd_peaks.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # features
    cmd_features = commands.add_parser("features")
    cmd_features.add_argument("--input")
    cmd_features.add_argument("--output")
    cmd_features.add_argument("--version")
    cmd_features.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # train
    cmd_training_gold = commands.add_parser(
        "prepare-training-gold",
        help="Derive training window variants from an immutable clean Silver run",
    )
    cmd_training_gold.add_argument("--source-run-id", default="20260809T044902Z")
    cmd_training_gold.add_argument("--run-id")
    cmd_training_gold.add_argument(
        "--config", default="PipelinePython/configs/data.yaml"
    )

    cmd_train = commands.add_parser("train")
    cmd_train.add_argument("--input")
    cmd_train.add_argument("--output")
    cmd_train.add_argument("--nested-cv", action="store_true")
    cmd_train.add_argument("--outer-folds", type=int, default=28)
    cmd_train.add_argument("--holdout", nargs="+")
    cmd_train.add_argument("--augmentation")
    cmd_train.add_argument("--sensor", choices=list(TRAINING_CHANNELS))
    cmd_train.add_argument("--run-id")
    cmd_train.add_argument(
        "--gold-input-root",
        help="Immutable Gold run directory used as the local-ML input",
    )
    cmd_train.add_argument("--smoke", action="store_true")
    cmd_train.add_argument("--resume", action="store_true")
    cmd_train.add_argument("--config", default="PipelinePython/configs/data.yaml")

    cmd_colab_handoff = commands.add_parser(
        "prepare-colab-dl",
        help="Create a checksum-bound manual Colab DL handoff after local ML",
    )
    cmd_colab_handoff.add_argument("--run-id")
    cmd_colab_handoff.add_argument("--gold-input-root")
    cmd_colab_handoff.add_argument(
        "--config", default="PipelinePython/configs/data.yaml"
    )

    cmd_csv = commands.add_parser(
        "export-csv",
        help="Export CSV views of Silver, Gold, and peak-table artifacts",
    )
    cmd_csv.add_argument("--run-root", required=True)
    cmd_csv.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # evaluate
    cmd_eval = commands.add_parser("evaluate")
    cmd_eval.add_argument("--model")
    cmd_eval.add_argument("--output")
    cmd_eval.add_argument("--config", default="PipelinePython/configs/data.yaml")
    cmd_eval.add_argument("--sensor", choices=list(TRAINING_CHANNELS))

    cmd_validate = commands.add_parser("validate")
    cmd_validate.add_argument("--sensor", choices=list(TRAINING_CHANNELS))
    cmd_validate.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # export-onnx
    cmd_export = commands.add_parser("export-onnx")
    cmd_export.add_argument("--model")
    cmd_export.add_argument("--output")
    cmd_export.add_argument("--include-probability", action="store_true")
    cmd_export.add_argument("--verify-outputs", action="store_true")
    cmd_export.add_argument("--config", default="PipelinePython/configs/data.yaml")
    cmd_export.add_argument("--sensor", choices=list(TRAINING_CHANNELS))
    cmd_export.add_argument("--run-id")
    cmd_export.add_argument("--gold-input-root")

    cmd_preflight = commands.add_parser(
        "onnx-preflight",
        help="Fail-closed converter preflight on a development-approved candidate",
    )
    cmd_preflight.add_argument("--config", default="PipelinePython/configs/data.yaml")
    cmd_preflight.add_argument(
        "--sensor", required=True, choices=list(TRAINING_CHANNELS)
    )
    cmd_preflight.add_argument("--run-id")
    cmd_preflight.add_argument("--gold-input-root")

    # android-prep
    cmd_android = commands.add_parser("android-prep")
    cmd_android.add_argument("--onnx")
    cmd_android.add_argument("--output")
    cmd_android.add_argument("--config", default="PipelinePython/configs/data.yaml")
    cmd_android.add_argument("--sensor", choices=list(TRAINING_CHANNELS))
    cmd_android.add_argument("--run-id")
    cmd_android.add_argument("--gold-input-root")

    cmd_promote_android = commands.add_parser(
        "promote-android-model",
        help="Verify a physical DeviceParityReceipt and atomically publish one channel",
    )
    cmd_promote_android.add_argument(
        "--receipt",
        action="append",
        required=True,
        help="DeviceParityReceipt path; repeat to atomically promote Green and IR.",
    )
    cmd_promote_android.add_argument("--apk", required=True)
    cmd_promote_android.add_argument(
        "--config", default="PipelinePython/configs/data.yaml"
    )
    cmd_promote_android.add_argument("--run-id")
    cmd_promote_android.add_argument("--gold-input-root")

    # run-pipeline
    cmd_run = commands.add_parser("run-pipeline")
    cmd_run.add_argument("--source")
    cmd_run.add_argument("--output")
    cmd_run.add_argument("--reports")
    cmd_run.add_argument("--visualize", action="store_true")
    cmd_run.add_argument("--audit", action="store_true")
    cmd_run.add_argument("--sensor", choices=list(TRAINING_CHANNELS))
    cmd_run.add_argument("--smoke", action="store_true")
    cmd_run.add_argument("--resume", action="store_true")
    cmd_run.add_argument("--run-id")
    cmd_run.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # canonical workflow composition root
    cmd_pipeline = commands.add_parser(
        "pipeline",
        help="Run one canonical data, peak-to-peak, stress-ONNX, or complete flow",
    )
    cmd_pipeline.add_argument(
        "--flow", choices=["data", "peaks", "stress-onnx", "all"], default="all"
    )
    cmd_pipeline.add_argument("--participant")
    cmd_pipeline.add_argument("--sensor", choices=list(TRAINING_CHANNELS))
    cmd_pipeline.add_argument("--smoke", action="store_true")
    cmd_pipeline.add_argument("--resume", action="store_true")
    cmd_pipeline.add_argument("--run-id")
    cmd_pipeline.add_argument("--config", default="PipelinePython/configs/data.yaml")

    # run-scoped Android readiness; this command only verifies existing
    # artifacts and device receipts. It never reruns biosignal stages.
    cmd_android_readiness = commands.add_parser(
        "android-readiness",
        help="Verify research Android parity and report production readiness",
    )
    cmd_android_readiness.add_argument("--run-id", required=True)
    cmd_android_readiness.add_argument("--apk", required=True)
    cmd_android_readiness.add_argument("--receipt", action="append", default=[])
    cmd_android_readiness.add_argument(
        "--device-status",
        choices=["READY", "OFFLINE", "INSTRUMENTATION_FAILED"],
        required=True,
    )
    cmd_android_readiness.add_argument(
        "--config", default="PipelinePython/configs/data.yaml"
    )

    # generate-eda-report
    cmd_eda_report = commands.add_parser("generate-eda-report")
    cmd_eda_report.add_argument("--input")
    cmd_eda_report.add_argument("--output")
    cmd_eda_report.add_argument("--per-participant", action="store_true")
    cmd_eda_report.add_argument("--per-channel", action="store_true")
    cmd_eda_report.add_argument(
        "--stage", choices=["raw", "silver", "gold"], default="silver"
    )
    cmd_eda_report.add_argument("--config", default="PipelinePython/configs/data.yaml")

    cmd_peak_report = commands.add_parser("validate-ppg-ecg")
    cmd_peak_report.add_argument("--config", default="PipelinePython/configs/data.yaml")
    cmd_peak_csv = commands.add_parser(
        "visualize-ppg-ecg-csv",
        help="Build CSV-first per-session ECG-PPG peak visual audits",
    )
    cmd_peak_csv.add_argument("--run-root", required=True)
    cmd_peak_csv.add_argument("--config", default="PipelinePython/configs/data.yaml")

    cmd_canonical_run = commands.add_parser(
        "run-ppg-ecg",
        help="Run isolated Raw-to-Gold, 240-second validation, and PPG-only baseline ML",
    )
    cmd_canonical_run.add_argument(
        "--config", default="PipelinePython/configs/data.yaml"
    )
    cmd_canonical_run.add_argument("--run-id")

    cmd_rerun_report = commands.add_parser("report-preprocessing-peak")
    cmd_rerun_report.add_argument(
        "--config", default="PipelinePython/configs/data.yaml"
    )

    cmd_rerun_report = commands.add_parser("rerun-report")
    cmd_rerun_report.add_argument("--artifacts-root", required=True)
    cmd_rerun_report.add_argument("--reports-root", required=True)
    cmd_rerun_report.add_argument("--output", required=True)

    # audit
    cmd_audit = commands.add_parser("audit")
    cmd_audit.add_argument("--input")
    cmd_audit.add_argument("--output")
    cmd_audit.add_argument("--format")
    cmd_audit.add_argument("--config", default="PipelinePython/configs/data.yaml")

    return parser.parse_args(resolved_argv)


def _parse_phase_args(argv: list[str]) -> argparse.Namespace:
    """Parse the stable PRD phase interface without removing legacy subcommands."""

    parser = argparse.ArgumentParser(description="Canonical local biosignal pipeline")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("ingest", "eda", "validate", "train", "export", "all"),
    )
    parser.add_argument("--participant")
    parser.add_argument("--config", default="PipelinePython/configs/data.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv, namespace=argparse.Namespace(command="phase"))


def _selected_sensor(requested: str | None = None) -> str:
    # Kept existing selection logic
    return requested or "ppg_green"


def _with_isolated_run(config, run_id: str | None, gold_input_root: str | None = None):
    """Return a config whose mutable artifacts belong to one UTC-labelled run."""

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"\d{8}T\d{6}Z", resolved_run_id):
        raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")
    artifact_root = f"data/artifacts/runs/{resolved_run_id}"
    report_root = f"reports/runs/{resolved_run_id}"
    artifacts = {
        **config.artifacts,
        "run_id": resolved_run_id,
        "artifacts_root": artifact_root,
        "bronze_dir": f"{artifact_root}/bronze",
        "sync_dir": f"{artifact_root}/sync",
        "segmented_dir": f"{artifact_root}/segmented",
        "silver_dir": f"{artifact_root}/silver",
        "gold_dir": f"{artifact_root}/gold",
        "gold_features": f"{artifact_root}/gold/gold_features_v3.parquet",
        "gold_ibi_sequences": f"{artifact_root}/gold/gold_ibi_sequences_v3.parquet",
        "manifest": f"{artifact_root}/gold/manifest_v3.json",
        "eda_dir": f"{artifact_root}/eda",
        "validation_dir": f"{artifact_root}/validation",
        "models_dir": f"{artifact_root}/models",
        "export_dir": f"{artifact_root}/export",
        "reports_dir": report_root,
    }
    if gold_input_root:
        input_root = Path(gold_input_root).resolve()
        artifacts.update(
            {
                "gold_input_root": str(input_root),
                "gold_input_features": str(input_root / "gold_features_v3.parquet"),
                "gold_input_ibi_sequences": str(
                    input_root / "gold_ibi_sequences_v3.parquet"
                ),
                "gold_input_readiness": str(input_root / "data_readiness.json"),
            }
        )
    model_version = f"{config.model_version}-{resolved_run_id}"
    return replace(config, model_version=model_version, artifacts=artifacts)


def _with_model_artifact_run(config, run_id: str | None, gold_input_root: str | None):
    """Bind model work to one run while keeping an explicit Gold input override."""

    isolated = _with_isolated_run(config, run_id, gold_input_root)
    artifact_root = Path(isolated.artifacts["artifacts_root"]).resolve()
    artifacts = {
        **isolated.artifacts,
        "gold_dir": str(artifact_root / "gold"),
        "gold_features": str(artifact_root / "gold" / "gold_features_v3.parquet"),
        "export_dir": str(artifact_root / "export"),
        "models_dir": str(artifact_root / "models"),
    }
    if gold_input_root:
        input_root = Path(gold_input_root).resolve()
        artifacts.update(
            {
                "gold_input_root": str(input_root),
                "gold_input_features": str(input_root / "gold_features_v3.parquet"),
                "gold_input_ibi_sequences": str(
                    input_root / "gold_ibi_sequences_v3.parquet"
                ),
                "gold_input_readiness": str(input_root / "data_readiness.json"),
            }
        )
    return replace(isolated, artifacts=artifacts)


def _train_with_escalation(
    project_root: Path,
    config,
    sensor: str | None,
) -> dict[str, dict]:
    """Run local classical ML only; Colab is prepared explicitly afterwards."""

    from .settings import pipeline_paths_from_config
    from .training_canonical import CanonicalClassicalTrainer

    paths = pipeline_paths_from_config(project_root, config)
    trainer = CanonicalClassicalTrainer(project_root, config, paths)
    sensors = (sensor,) if sensor else tuple(config.training.training_channels)
    unexpected = set(sensors) - set(config.training.training_channels)
    if unexpected:
        raise ValueError(f"Training sensor is not allowed: {sorted(unexpected)}")
    return {channel: trainer.run(channel) for channel in sensors}


def _component_config_path(config_path: str) -> Path:
    """Resolve the active training config used to bind a Colab resume run."""

    selected = (PROJECT_ROOT / config_path).resolve()
    companion = selected.parent / "training.yaml"
    return (
        companion
        if companion.is_file()
        else PROJECT_ROOT / "PipelinePython" / "configs" / "training.yaml"
    )


def _prepare_colab_handoff(
    project_root: Path,
    config,
    reports: dict[str, dict],
    *,
    config_path: str,
) -> Path | None:
    """Write one fail-closed, manually transferred Colab manifest if DL is needed."""

    from .colab_handoff import build_handoff, sha256_file, write_handoff_manifest
    from .settings import pipeline_paths_from_config

    local_ml_handoff = {
        channel: {
            "requires_dl": bool(report.get("requires_dl")),
            "trigger_reason": report.get("trigger_reason"),
            "predictor_columns": report.get(
                "predictor_columns", report.get("feature_order", [])
            ),
            "best_classical_model": report.get("model_name"),
            "best_classical_metrics": report.get("development_metrics"),
            "holdout_evaluated": bool(report.get("holdout_evaluated")),
            "channel_data_fingerprint": report.get(
                "channel_data_fingerprint", report.get("data_fingerprint")
            ),
        }
        for channel, report in reports.items()
    }
    if not any(result["requires_dl"] for result in local_ml_handoff.values()):
        return None

    current_gold_root = Path(config.artifacts.get("gold_dir", "data/gold"))
    current_gold_root = (
        current_gold_root
        if current_gold_root.is_absolute()
        else project_root / current_gold_root
    )
    if (current_gold_root / "manifest_v3.json").is_file():
        from .training_dl_canonical import build_waveform_handoff

        return build_waveform_handoff(
            project_root,
            config,
            reports,
            config_path=Path(config_path),
        )

    paths = pipeline_paths_from_config(project_root, config)
    gold_ibi_config = Path(
        config.artifacts.get(
            "gold_input_ibi_sequences",
            config.artifacts.get(
                "gold_ibi_sequences", "data/gold/gold_ibi_sequences_v3.parquet"
            ),
        )
    )
    source_gold_ibi_path = (
        gold_ibi_config
        if gold_ibi_config.is_absolute()
        else project_root / gold_ibi_config
    )
    run_id = str(config.artifacts.get("run_id") or paths.artifacts_root.name)
    dl_channels = {
        channel for channel, result in local_ml_handoff.items() if result["requires_dl"]
    }
    if not source_gold_ibi_path.is_file():
        raise FileNotFoundError(
            f"Canonical Gold IBI artifact is missing: {source_gold_ibi_path}"
        )
    sequences = pd.read_parquet(source_gold_ibi_path)
    split_column = "split"
    if split_column not in sequences.columns:
        development = {f"ID-{index:02d}" for index in range(1, 29)}
        holdout = {f"ID-{index:02d}" for index in range(29, 33)}
        participants = sequences["participant_id"].astype(str)
        unexpected = sorted(set(participants) - development - holdout)
        if unexpected:
            raise ValueError(
                "Gold IBI handoff contains participants outside ID-01..ID-32: "
                f"{unexpected}"
            )
        sequences = sequences.copy()
        sequences[split_column] = participants.map(
            lambda participant: (
                "development" if participant in development else "holdout"
            )
        )

    required_columns = {
        "participant_id",
        "session_id",
        "phase",
        "sensor_channel",
        "window_id",
        "label",
        "split",
        "quality_score",
        "quality_tier",
        "source_file",
        "source_checksum",
        "preprocessing_version",
        "feature_version",
        "ibi_values",
        "ibi_mask",
    }
    missing = sorted(required_columns - set(sequences.columns))
    if missing:
        raise ValueError(f"Gold IBI handoff is missing columns: {missing}")
    filtered = sequences[sequences["sensor_channel"].astype(str).isin(dl_channels)].loc[
        :, sorted(required_columns)
    ]
    if filtered.empty:
        raise ValueError("Conditional DL handoff contains no eligible channel rows")
    if set(filtered["sensor_channel"].astype(str)) != dl_channels:
        raise ValueError("Conditional DL handoff channel coverage is incomplete")
    reports_root = Path(config.artifacts.get("reports_dir", "reports"))
    reports_root = (
        reports_root if reports_root.is_absolute() else project_root / reports_root
    )
    handoff_root = reports_root / "model_eval" / "ml" / "colab_input"
    handoff_root.mkdir(parents=True, exist_ok=True)
    gold_ibi_path = handoff_root / "gold_ibi_sequences_v3.parquet"
    temporary_gold = gold_ibi_path.with_suffix(".parquet.tmp")
    filtered.to_parquet(temporary_gold, index=False)
    temporary_gold.replace(gold_ibi_path)
    handoff = build_handoff(
        run_id=run_id,
        gold_ibi_path=gold_ibi_path,
        training_config_sha256=sha256_file(_component_config_path(config_path)),
        local_ml_handoff=local_ml_handoff,
    )
    return write_handoff_manifest(
        handoff,
        reports_root / "model_eval" / "ml" / "colab_dl_handoff_manifest.json",
    )


def _load_local_ml_reports(project_root: Path, config) -> dict[str, dict]:
    """Read previously written local ML evidence without retraining or tuning."""

    reports_root = Path(config.artifacts.get("reports_dir", "reports"))
    reports_root = (
        reports_root if reports_root.is_absolute() else project_root / reports_root
    )
    base = reports_root / "model_eval" / config.model_version
    reports: dict[str, dict] = {}
    for channel in config.training.training_channels:
        report_path = base / f"{channel}_full.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Local ML report is required before Colab handoff: {report_path}"
            )
        reports[channel] = json.loads(report_path.read_text(encoding="utf-8"))
    return reports


def _export_selected_model(project_root: Path, config, sensor: str) -> dict:
    """Dispatch to the exporter matching the actual accepted training report."""
    if sensor not in TRAINING_CHANNELS:
        raise ValueError(f"ONNX export is restricted to {TRAINING_CHANNELS}: {sensor}")

    reports_root = Path(config.artifacts.get("reports_dir", "reports"))
    reports_root = (
        reports_root if reports_root.is_absolute() else project_root / reports_root
    )
    report_path = (
        reports_root / "model_eval" / config.model_version / f"{sensor}_full.json"
    )
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("selected_arch"):
            raise RuntimeError(
                "legacy_dl_export_quarantined: use the waveform-only Colab contract"
            )
    from .onnx_v3 import OnnxBundleExporter

    return OnnxBundleExporter(project_root, config).run(sensor)


def _latest_compatible_run_id(phase: str) -> str:
    """Reject the retired implicit run resolver.

    A phase may consume only the explicitly selected run.  Keeping this
    adapter as a hard failure prevents old callers from silently reintroducing
    cross-run Gold or model provenance.
    """

    raise FileNotFoundError(
        f"Implicit run selection is disabled for --phase {phase}; provide --run-id"
    )


def _resolved_phase_run_id(phase: str, requested: str | None) -> str:
    if requested:
        if not re.fullmatch(r"\d{8}T\d{6}Z", requested):
            raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")
        return requested
    if phase in {"ingest", "all"}:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raise ValueError("canonical cohort phase requires an explicit --run-id")


def _run_phase_pipeline(args: argparse.Namespace, base_config) -> None:
    """Compose concrete services at the CLI boundary and run abstract stages."""

    from .orchestration import (
        CallablePipelineStage,
        PipelineOrchestrator,
        PipelineRunContext,
        StageResult,
    )
    from .peak_detector_audit import PeakDetectorAudit
    from .peak_validation import generate_canonical_peak_validation_reports
    from .settings import pipeline_paths_from_config
    from .wrangling_eda import generate_eda_reports

    if args.participant:
        if not re.fullmatch(r"ID-(0[1-9]|[12][0-9]|3[0-2])", args.participant):
            raise ValueError("participant must be ID-01..ID-32")
        if args.phase in {"validate", "train", "export", "all"}:
            raise ValueError(
                f"--participant is not allowed for cohort phase {args.phase}"
            )

    run_id = _resolved_phase_run_id(args.phase, args.run_id)
    config = _with_isolated_run(base_config, run_id)
    context = PipelineRunContext(
        project_root=PROJECT_ROOT,
        config=config,
        config_path=(PROJECT_ROOT / args.config).resolve(),
        run_id=run_id,
        participant=args.participant,
    )
    data = LocalBiosignalPipeline(PROJECT_ROOT, config)
    paths = pipeline_paths_from_config(PROJECT_ROOT, config)
    reports_root = Path(config.artifacts["reports_dir"])
    reports_root = (
        reports_root if reports_root.is_absolute() else PROJECT_ROOT / reports_root
    )

    def validate_inputs(_context: PipelineRunContext) -> StageResult:
        frame = data.validate_inputs()
        output = reports_root / "data_quality" / "raw_input_index_v3.parquet"
        return StageResult((output,), {"row_count": len(frame)})

    def ingest(_context: PipelineRunContext) -> StageResult:
        frame = data.ingest(args.participant)
        return StageResult(
            (paths.bronze_dir / "ingestion_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def sync(_context: PipelineRunContext) -> StageResult:
        frame = data.sync_raw(args.participant)
        return StageResult(
            (data.sync_dir / "synchronization_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def segment(_context: PipelineRunContext) -> StageResult:
        frame = data.segment(args.participant)
        return StageResult(
            (data.segmented_dir / "segmentation_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def preprocess(_context: PipelineRunContext) -> StageResult:
        frame = data.preprocess(args.participant)
        return StageResult(
            (paths.silver_dir / "preprocessing_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def extract(_context: PipelineRunContext) -> StageResult:
        frame = data.extract_features(args.participant)
        return StageResult(
            (
                paths.gold_dir / "gold_features_v3.parquet",
                paths.gold_dir / "gold_ibi_sequences_v3.parquet",
                paths.gold_dir / "data_readiness.json",
            ),
            {"window_count": len(frame)},
        )

    def audit_peak_detector(_context: PipelineRunContext) -> StageResult:
        payload = PeakDetectorAudit(PROJECT_ROOT, config, paths).run()
        output = paths.validation_dir / "detector_selection" / "detector_selection.json"
        return StageResult(
            (output,),
            {
                "frozen_config_verified": payload["frozen_config_verified"],
                "development_participant_count": len(
                    payload["development_participants"]
                ),
            },
        )

    def eda(stage: str):
        def action(_context: PipelineRunContext) -> StageResult:
            generate_eda_reports(PROJECT_ROOT, config, stage=stage)
            return StageResult(
                (
                    paths.eda_dir / f"eda_report_{stage}.html",
                    paths.eda_dir / f"eda_report_{stage}.md",
                    paths.eda_dir / f"ydata_profiling_status_{stage}.json",
                ),
                {"eda_stage": stage},
            )

        return action

    def validate(_context: PipelineRunContext) -> StageResult:
        outputs = generate_canonical_peak_validation_reports(PROJECT_ROOT, config)
        return StageResult(
            (
                outputs.metrics_path,
                outputs.output_dir / "validation_summary.json",
                paths.validation_dir
                / "hrv_agreement"
                / "physiological_validation.json",
            ),
            {"row_count": outputs.row_count, "primary_gate": "hrv_prv_agreement"},
        )

    def train(_context: PipelineRunContext) -> StageResult:
        reports = _train_with_escalation(PROJECT_ROOT, config, None)
        handoff = _prepare_colab_handoff(
            PROJECT_ROOT,
            config,
            reports,
            config_path=args.config,
        )
        outputs = [reports_root / "model_eval" / config.model_version]
        if handoff is not None:
            outputs.append(handoff)
        return StageResult(
            tuple(outputs),
            {
                "release_status": {
                    channel: report.get("release_status")
                    for channel, report in reports.items()
                },
                "colab_handoff": str(handoff) if handoff else None,
            },
        )

    def export(_context: PipelineRunContext) -> StageResult:
        reports = _load_local_ml_reports(PROJECT_ROOT, config)
        candidates = [
            channel
            for channel, report in reports.items()
            if report.get("release_status") == "release_candidate"
        ]
        if not candidates:
            raise RuntimeError(
                "ONNX export blocked: no channel has a frozen release-candidate contract"
            )
        for channel in candidates:
            _export_selected_model(PROJECT_ROOT, config, channel)
        return StageResult(
            (paths.export_dir,), {"release_candidate_channels": candidates}
        )

    detector_selection_path = (
        paths.validation_dir / "detector_selection" / "detector_selection.json"
    )
    validation_recommendation_path = (
        paths.validation_dir / "hrv_agreement" / "physiological_validation.json"
    )
    stages = (
        CallablePipelineStage("validate_inputs", validate_inputs),
        CallablePipelineStage("ingest", ingest),
        CallablePipelineStage(
            "eda_raw",
            eda("raw"),
            (paths.bronze_dir / "ingestion_manifest_v3.parquet",),
        ),
        CallablePipelineStage(
            "sync",
            sync,
            (paths.bronze_dir / "ingestion_manifest_v3.parquet",),
        ),
        CallablePipelineStage(
            "segment",
            segment,
            (
                paths.bronze_dir / "ingestion_manifest_v3.parquet",
                data.sync_dir / "synchronization_manifest_v3.parquet",
            ),
        ),
        CallablePipelineStage(
            "preprocess",
            preprocess,
            (data.segmented_dir / "segmentation_manifest_v3.parquet",),
        ),
        CallablePipelineStage(
            "eda_silver",
            eda("silver"),
            (paths.silver_dir / "preprocessing_manifest_v3.parquet",),
        ),
        CallablePipelineStage(
            "audit_peak_detector",
            audit_peak_detector,
            (
                paths.silver_dir / "preprocessing_manifest_v3.parquet",
                data.sync_dir / "synchronization_manifest_v3.parquet",
            ),
        ),
        CallablePipelineStage(
            "extract",
            extract,
            (
                paths.silver_dir / "preprocessing_manifest_v3.parquet",
                detector_selection_path,
            ),
        ),
        CallablePipelineStage(
            "eda_gold",
            eda("gold"),
            (
                paths.gold_dir / "gold_features_v3.parquet",
                paths.gold_dir / "data_readiness.json",
            ),
        ),
        CallablePipelineStage(
            "validate",
            validate,
            (
                paths.silver_dir / "preprocessing_manifest_v3.parquet",
                data.sync_dir / "synchronization_manifest_v3.parquet",
                detector_selection_path,
            ),
        ),
        CallablePipelineStage(
            "train",
            train,
            (
                paths.gold_dir / "gold_features_v3.parquet",
                paths.gold_dir / "gold_ibi_sequences_v3.parquet",
                paths.gold_dir / "data_readiness.json",
                validation_recommendation_path,
            ),
        ),
        CallablePipelineStage(
            "export",
            export,
            (
                reports_root / "model_eval" / config.model_version,
                paths.gold_dir / "data_readiness.json",
                validation_recommendation_path,
            ),
        ),
    )
    phase_stages = {
        "ingest": ["validate_inputs", "ingest"],
        "validate": ["audit_peak_detector", "validate"],
        "train": ["train"],
        "export": ["export"],
        "all": [
            "validate_inputs",
            "ingest",
            "eda_raw",
            "sync",
            "segment",
            "preprocess",
            "eda_silver",
            "audit_peak_detector",
            "extract",
            "eda_gold",
            "validate",
            "train",
            "export",
        ],
    }
    if args.phase == "eda":
        selected = ["eda_raw"]
        if (paths.silver_dir / "preprocessing_manifest_v3.parquet").is_file():
            selected.append("eda_silver")
        if (paths.gold_dir / "gold_features_v3.parquet").is_file():
            selected.append("eda_gold")
    else:
        selected = phase_stages[args.phase]
    if args.smoke:
        if args.phase != "all":
            raise ValueError("--smoke is supported only with --phase all")
        selected = selected[: selected.index("train")]
    state = PipelineOrchestrator(context, stages).run(selected, resume=args.resume)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "phase": args.phase,
                "state_path": str(context.state_path),
                "raw_integrity_verified": state.get("raw_integrity_verified", False),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    config = load_pipeline_config(args.config)
    if args.command == "android-readiness":
        from .android_readiness import write_android_readiness_report

        report, artifact_path, report_path = write_android_readiness_report(
            PROJECT_ROOT,
            args.run_id,
            apk_path=Path(args.apk).resolve(),
            receipt_paths=[Path(receipt).resolve() for receipt in args.receipt],
            device_connection_status=args.device_status,
        )
        report["artifact_path"] = str(artifact_path)
        report["report_path"] = str(report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "pipeline":
        if args.participant and args.flow != "data":
            raise ValueError("--participant is supported only with --flow data")
        if args.sensor and args.flow != "stress-onnx":
            raise ValueError("--sensor is supported only with --flow stress-onnx")
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config = _with_isolated_run(config, run_id)
        from .workflows.runner import CanonicalWorkflowRunner, WorkflowOptions

        state = CanonicalWorkflowRunner(
            WorkflowOptions(
                project_root=PROJECT_ROOT,
                config=config,
                config_path=(PROJECT_ROOT / args.config).resolve(),
                run_id=run_id,
                participant=args.participant,
                sensor=args.sensor,
            )
        ).run(args.flow, resume=args.resume, smoke=args.smoke)
        artifact_root = Path(config.artifacts["artifacts_root"])
        if not artifact_root.is_absolute():
            artifact_root = PROJECT_ROOT / artifact_root
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "flow": args.flow,
                    "state_path": str(artifact_root / "run_state.json"),
                    "raw_integrity_verified": state.get(
                        "raw_integrity_verified", False
                    ),
                },
                indent=2,
            )
        )
        return
    legacy_evidence_commands = {
        "phase",
        "prepare-training-gold",
        "ingest",
        "sqa",
        "sync-raw",
        "segment",
        "preprocess",
        "extract-features",
        "eda",
        "generate-eda-report",
        "peaks",
        "validate-ppg-ecg",
        "visualize-ppg-ecg-csv",
        "report-preprocessing-peak",
        "run-ppg-ecg",
        "rerun-report",
        "features",
        "train",
        "prepare-colab-dl",
        "export-csv",
        "evaluate",
        "validate",
        "export-onnx",
        "onnx-preflight",
        "android-prep",
        "promote-android-model",
        "run-pipeline",
        "audit",
    }
    if args.command in legacy_evidence_commands:
        raise RuntimeError(
            f"legacy_cli_quarantined:{args.command}: "
            "use `python -m PipelinePython pipeline --flow all --run-id <new-run-id>`"
        )
    if args.command == "phase":
        _run_phase_pipeline(args, config)
        return
    if args.command in {
        "run-ppg-ecg",
        "run-pipeline",
        "train",
        "prepare-colab-dl",
    }:
        config = _with_isolated_run(
            config,
            args.run_id,
            getattr(args, "gold_input_root", None),
        )
    elif args.command in {
        "export-onnx",
        "onnx-preflight",
        "android-prep",
        "promote-android-model",
    }:
        config = _with_model_artifact_run(
            config,
            args.run_id,
            getattr(args, "gold_input_root", None),
        )
    if args.command == "prepare-training-gold":
        from .training_gold import TrainingGoldBuilder

        output_run_id = args.run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        if not re.fullmatch(r"\d{8}T\d{6}Z", output_run_id):
            raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")
        source_run = PROJECT_ROOT / "data" / "artifacts" / "runs" / args.source_run_id
        output_run = PROJECT_ROOT / "data" / "artifacts" / "runs" / output_run_id
        result = TrainingGoldBuilder(
            PROJECT_ROOT, config, source_run, output_run
        ).build()
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    data = LocalBiosignalPipeline(PROJECT_ROOT, config)

    if args.command == "ingest":
        data.validate_inputs()
        data.ingest()
        print("Ingestion complete.")
    elif args.command == "sqa":
        data.sync_raw()
        data.segment()
        data.preprocess()
        print("SQA and MADC complete.")
    elif args.command == "sync-raw":
        data.sync_raw(args.participant)
        print("Raw ECG clock synchronization complete.")
    elif args.command == "segment":
        data.segment(args.participant)
        print("Anchor segmentation complete.")
    elif args.command == "preprocess":
        data.preprocess(args.participant)
        print("MADC and SQA preprocessing complete.")
    elif args.command == "extract-features":
        data.extract_features(args.participant)
        print("Gold feature extraction complete.")
    elif args.command == "eda" or args.command == "generate-eda-report":
        from .wrangling_eda import generate_eda_reports

        generate_eda_reports(PROJECT_ROOT, config, stage=args.stage)
        print("EDA generation complete.")
    elif args.command == "peaks" or args.command == "validate-ppg-ecg":
        from .peak_validation import generate_canonical_peak_validation_reports

        generate_canonical_peak_validation_reports(PROJECT_ROOT, config)
        print("Peak validation complete.")
    elif args.command == "visualize-ppg-ecg-csv":
        from .peak_csv_visualization import build_csv_peak_visual_report

        outputs = build_csv_peak_visual_report(Path(args.run_root), PROJECT_ROOT)
        print(
            json.dumps(
                {
                    "output_dir": str(outputs.output_dir),
                    "manifest_path": str(outputs.manifest_path),
                    "row_count": outputs.row_count,
                    "valid_count": outputs.valid_count,
                },
                indent=2,
            )
        )
    elif args.command == "report-preprocessing-peak":
        from .rerun_report import generate_preprocessing_peak_report

        report = generate_preprocessing_peak_report(PROJECT_ROOT, config)
        print(f"Preprocessing/peak report generated: {report}")
    elif args.command == "run-ppg-ecg":
        from .peak_validation import generate_canonical_peak_validation_reports
        from .rerun_report import generate_preprocessing_peak_report
        from .training_canonical import run_baseline_ml
        from .wrangling_eda import generate_eda_reports
        from .settings import pipeline_paths_from_config

        print(
            "Running canonical PPG-only Gold and 240-second ECG validation pipeline..."
        )
        data.validate_inputs()
        data.ingest()
        generate_eda_reports(PROJECT_ROOT, config, stage="raw")
        data.sync_raw()
        data.segment()
        data.preprocess()
        generate_eda_reports(PROJECT_ROOT, config, stage="silver")
        data.extract_features()
        generate_eda_reports(PROJECT_ROOT, config, stage="gold")
        generate_canonical_peak_validation_reports(PROJECT_ROOT, config)
        from .peak_csv_visualization import build_csv_peak_visual_report

        build_csv_peak_visual_report(
            pipeline_paths_from_config(PROJECT_ROOT, config).artifacts_root,
            PROJECT_ROOT,
        )
        reports = run_baseline_ml(
            PROJECT_ROOT, config, pipeline_paths_from_config(PROJECT_ROOT, config)
        )
        handoff = _prepare_colab_handoff(
            PROJECT_ROOT, config, reports, config_path=args.config
        )
        report = generate_preprocessing_peak_report(PROJECT_ROOT, config)
        print(f"Baseline ML reports: {json.dumps(reports, sort_keys=True)}")
        if handoff:
            print(f"Conditional Colab DL handoff: {handoff}")
        print(f"Canonical report: {report}")
    elif args.command == "rerun-report":
        from .rerun_report import generate_rerun_report

        generate_rerun_report(
            project_root=PROJECT_ROOT,
            artifacts_root=Path(args.artifacts_root),
            reports_root=Path(args.reports_root),
            output_path=Path(args.output),
        )
        print("Read-only rerun report complete.")
    elif args.command == "features":
        data.extract_features()
        print("Feature extraction complete.")
    elif args.command == "train":
        reports = _train_with_escalation(PROJECT_ROOT, config, args.sensor)
        handoff = _prepare_colab_handoff(
            PROJECT_ROOT, config, reports, config_path=args.config
        )
        if any(
            report.get("release_status") != "candidate_ready"
            for report in reports.values()
        ):
            logging.getLogger(__name__).warning(
                "At least one selected sensor did not pass the locked development/holdout gates"
            )
        if handoff:
            print(f"Conditional Colab DL handoff: {handoff}")
        print("Training complete.")
    elif args.command == "prepare-colab-dl":
        handoff = _prepare_colab_handoff(
            PROJECT_ROOT,
            config,
            _load_local_ml_reports(PROJECT_ROOT, config),
            config_path=args.config,
        )
        if handoff is None:
            print("No channel requires conditional Colab DL.")
        else:
            print(f"Conditional Colab DL handoff: {handoff}")
    elif args.command == "export-csv":
        from .csv_artifacts import export_run_csv

        result = export_run_csv(Path(args.run_root))
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate":
        from .validation_v3 import V3ReleaseValidator

        sensor = _selected_sensor(args.sensor)
        V3ReleaseValidator(PROJECT_ROOT).run(sensor)
        print("Evaluation complete.")
    elif args.command == "validate":
        from .validation_v3 import V3ReleaseValidator

        result = V3ReleaseValidator(PROJECT_ROOT).run(args.sensor)
        if result.get("release_status") not in {"candidate", "accepted"}:
            logging.getLogger(__name__).warning("Release validation remains blocked")
        print("Validation complete.")
    elif args.command == "export-onnx":
        if args.sensor:
            exported = {
                args.sensor: _export_selected_model(PROJECT_ROOT, config, args.sensor)
            }
        else:
            from .onnx_v3 import OnnxBundleExporter

            exported = OnnxBundleExporter(PROJECT_ROOT, config).run()
        print(json.dumps(exported, indent=2, sort_keys=True))
    elif args.command == "onnx-preflight":
        from .onnx_v3 import OnnxBundleExporter

        result = OnnxBundleExporter(PROJECT_ROOT, config).preflight_sensor(args.sensor)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "android-prep":
        from .android_vectors import AndroidVectorGenerator
        from .settings import pipeline_paths_from_config

        sensors = (args.sensor,) if args.sensor else config.android.deployable_sensors
        export_dir = pipeline_paths_from_config(PROJECT_ROOT, config).export_dir
        generated: dict[str, dict[str, object]] = {}
        for sensor in sensors:
            payload = AndroidVectorGenerator(PROJECT_ROOT, config).generate(sensor)
            generated[sensor] = {
                "model_version": payload["model_version"],
                "sensor_channel": payload["sensor_channel"],
                "python_vectors_passed": payload["python_vectors_passed"],
                "python_vector_status": payload["python_vector_status"],
                "android_instrumentation_status": payload[
                    "android_instrumentation_status"
                ],
                "android_ready": payload["android_ready"],
                "candidate_vectors_path": str(
                    export_dir / config.model_version / sensor / "test_vectors.json"
                ),
            }
        print(json.dumps(generated, indent=2, sort_keys=True))
    elif args.command == "promote-android-model":
        from .android_vectors import DeviceParityPromoter

        result = DeviceParityPromoter(PROJECT_ROOT, config).promote_many(
            [Path(receipt).resolve() for receipt in args.receipt],
            Path(args.apk).resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run-pipeline":
        from .peak_validation import generate_canonical_peak_validation_reports
        from .wrangling_eda import generate_eda_reports

        # End to End Run
        print("Running full pipeline...")
        data.validate_inputs()
        data.ingest()
        data.sync_raw()
        data.segment()
        data.preprocess()
        generate_eda_reports(PROJECT_ROOT, config)
        generate_canonical_peak_validation_reports(PROJECT_ROOT, config)
        from .peak_csv_visualization import build_csv_peak_visual_report

        build_csv_peak_visual_report(
            pipeline_paths_from_config(PROJECT_ROOT, config).artifacts_root,
            PROJECT_ROOT,
        )
        data.extract_features()
        reports = _train_with_escalation(PROJECT_ROOT, config, args.sensor)
        handoff = _prepare_colab_handoff(
            PROJECT_ROOT, config, reports, config_path=args.config
        )
        for sensor, report in reports.items():
            if report.get("release_status") == "candidate_ready":
                _export_selected_model(PROJECT_ROOT, config, sensor)
                AndroidVectorGenerator(PROJECT_ROOT, config).generate(sensor)
        if handoff:
            print(f"Conditional Colab DL handoff: {handoff}")
        print("Pipeline execution complete.")
    elif args.command == "audit":
        data.validate_inputs()
        print("Audit report generated.")
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
