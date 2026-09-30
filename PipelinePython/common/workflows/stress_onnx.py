"""Training, conditional Colab handoff, and gated ONNX promotion."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from ..onnx_v3 import OnnxBundleExporter, OnnxExportMode
from ..internal_publish import publish_internal_tests
from ..settings import TRAINING_CHANNELS, PipelinePaths, pipeline_paths_from_config
from ..stress_comparison import write_stress_run_comparison
from ..training_dl_canonical import build_waveform_handoff
from ..training_canonical import run_baseline_ml
from ..orchestration import CallablePipelineStage, PipelineRunContext, StageResult


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a status document without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_stress_selection_contract(
    project_root: Path,
    config: Any,
    paths: PipelinePaths,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    allow_full_cohort_physiology: bool = True,
) -> Path:
    """Freeze channel-local model decisions before full-cohort physiology access."""

    run_contract_path = paths.artifacts_root / "run_contract_fingerprint.json"
    if not run_contract_path.is_file():
        raise FileNotFoundError("run_contract_fingerprint_missing")
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    expected_channels = tuple(config.training.training_channels)
    reports_root = _configured_path(project_root, config, "reports_dir", "reports")
    report_root = reports_root / "model_eval" / config.model_version
    report_checksums: dict[str, str] = {}
    channel_decisions: dict[str, dict[str, Any]] = {}
    for channel in expected_channels:
        report_path = report_root / f"{channel}_full.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if str(report.get("run_id", "")) != str(run_contract.get("run_id", "")):
            raise RuntimeError(f"stress_report_run_mismatch:{channel}")
        runtime_report = reports.get(channel, {})
        gold_validation = runtime_report.get("gold_input_validation", {})
        admission_status = str(
            runtime_report.get(
                "training_admission_status",
                gold_validation.get("training_status", "blocked"),
            )
        )
        release_status = str(report.get("release_status", ""))
        requires_dl = bool(report.get("requires_dl"))
        selected_model = report.get("model_name")
        model_fingerprint = report.get("model_contract_fingerprint")
        quality_cohort = report.get("selected_quality_cohort")
        window_seconds = report.get("selected_window_seconds")
        window_stride_seconds = report.get("selected_stride_seconds")
        decision_threshold = report.get("threshold")
        predictor_columns = report.get("predictor_columns", [])
        frozen_classical_contract_complete = bool(
            selected_model
            and re.fullmatch(r"[0-9a-f]{64}", str(model_fingerprint or ""))
            and quality_cohort
            and isinstance(window_seconds, int)
            and window_seconds > 0
            and isinstance(window_stride_seconds, int)
            and window_stride_seconds > 0
            and isinstance(decision_threshold, (int, float))
            and math.isfinite(float(decision_threshold))
            and predictor_columns
        )
        if release_status in {
            "blocked_hrv_prv_admission",
            "blocked_insufficient_ppg_coverage",
        }:
            terminal_status = "COVERAGE_BLOCKED"
        elif admission_status != "ready":
            terminal_status = "COVERAGE_BLOCKED"
        elif requires_dl:
            terminal_status = "DL_PENDING"
        elif frozen_classical_contract_complete:
            terminal_status = "FROZEN_CLASSICAL"
        elif selected_model:
            terminal_status = "CONTRACT_INVALID"
        elif release_status.startswith("blocked_"):
            terminal_status = "TERMINAL_NO_GO"
        else:
            terminal_status = "CONTRACT_INVALID"
        report_checksums[channel] = _sha256(report_path)
        channel_decisions[channel] = {
            "terminal_status": terminal_status,
            "training_admission_status": admission_status,
            "selected_model": selected_model,
            "release_status": release_status,
            "requires_dl": requires_dl,
            "holdout_evaluated": bool(report.get("holdout_evaluated")),
            "quality_cohort": quality_cohort,
            "window_seconds": window_seconds,
            "window_stride_seconds": window_stride_seconds,
            "decision_threshold": decision_threshold,
            "predictor_columns": predictor_columns,
            "model_contract_fingerprint": model_fingerprint,
            "gold_features_sha256": gold_validation.get("features_sha256"),
            "hrv_training_admission": runtime_report.get("hrv_training_admission"),
            "report_path": str(report_path.resolve()),
            "report_sha256": report_checksums[channel],
        }
    terminal_statuses = {
        "COVERAGE_BLOCKED",
        "FROZEN_CLASSICAL",
        "TERMINAL_NO_GO",
    }
    frozen = bool(
        set(channel_decisions) == set(expected_channels)
        and all(
            row["terminal_status"] in terminal_statuses
            for row in channel_decisions.values()
        )
    )
    payload = {
        "schema_version": "stress-selection-contract-v1",
        "selection_contract_frozen": frozen,
        "selection_scope": "ID-01..ID-28",
        "selection_participants": list(config.training.development_participant_ids),
        "model_internal_test_scope": "ID-29..ID-32",
        "full_cohort_physiology_may_open": (frozen and allow_full_cohort_physiology),
        "channels": channel_decisions,
        "report_checksums": report_checksums,
        "run_id": run_contract.get("run_id"),
        "source_tree_sha256": run_contract.get("source_tree_sha256"),
        "config_fingerprint": run_contract.get("config_fingerprint"),
        "run_contract_sha256": _sha256(run_contract_path),
    }
    payload["selection_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target = paths.models_dir / "stress_selection_contract.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def train_classical_models(
    project_root: Path,
    config: Any,
    *,
    sensor: str | None = None,
    defer_reused_internal_test: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train only real Gold rows with the locked participant-safe trainer."""

    paths = pipeline_paths_from_config(project_root, config)
    if sensor is None:
        return run_baseline_ml(
            project_root,
            config,
            paths,
            defer_reused_internal_test=defer_reused_internal_test,
        )
    if sensor not in config.training.training_channels:
        raise ValueError(f"Training sensor is not allowed: {sensor}")
    return run_baseline_ml(
        project_root,
        config,
        paths,
        sensor=sensor,
        defer_reused_internal_test=defer_reused_internal_test,
    )


def load_local_ml_reports(
    project_root: Path,
    config: Any,
    *,
    sensor: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load frozen local-ML evidence without retraining or tuning."""

    reports_root = _configured_path(project_root, config, "reports_dir", "reports")
    base = reports_root / "model_eval" / config.model_version
    channels = (sensor,) if sensor else tuple(config.training.training_channels)
    reports: dict[str, dict[str, Any]] = {}
    for channel in channels:
        report_path = base / f"{channel}_full.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Local ML report is required before ONNX export: {report_path}"
            )
        reports[channel] = json.loads(report_path.read_text(encoding="utf-8"))
    return reports


def prepare_colab_handoff(
    project_root: Path,
    config: Any,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    config_path: Path,
) -> Path | None:
    """Write a checksum-bound manual Colab handoff only for failed channels."""

    return build_waveform_handoff(
        project_root,
        config,
        reports,
        config_path=config_path,
    )


def export_selected_model(
    project_root: Path,
    config: Any,
    sensor: str,
    *,
    mode: OnnxExportMode | str = OnnxExportMode.PRODUCTION,
) -> dict[str, Any]:
    """Export one frozen candidate through its actual model family."""

    if sensor not in TRAINING_CHANNELS:
        raise ValueError(f"ONNX export is restricted to {TRAINING_CHANNELS}: {sensor}")
    return OnnxBundleExporter(project_root, config, mode=mode).run(sensor)


def _published_internal_channels(exported: Mapping[str, Any]) -> list[str]:
    """Return only v2 artifacts that were actually exported successfully."""

    return sorted(
        channel
        for channel, artifact in exported.items()
        if channel in TRAINING_CHANNELS
        and isinstance(artifact, Mapping)
        and artifact.get("onnx_export_status") == "GO"
        and artifact.get("internal_android_status") == "PENDING_DEVICE_PARITY"
    )


def publish_internal_models(
    project_root: Path,
    config: Any,
    *,
    sensor: str | None = None,
) -> dict[str, Any]:
    """Open the frozen ID-29..ID-32 internal test exactly once, then export."""

    paths = pipeline_paths_from_config(project_root, config)
    contract_path = paths.models_dir / "stress_selection_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError("stress_selection_contract_missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("selection_contract_frozen") is not True:
        raise RuntimeError("stress_selection_contract_not_frozen")
    if contract.get("run_id") != config.artifacts.get("run_id"):
        raise RuntimeError("stress_selection_contract_run_mismatch")
    internal_test = publish_internal_tests(project_root, config, paths, sensor=sensor)
    eligible = [
        channel
        for channel, evidence in internal_test.items()
        if evidence.get("internal_test_status") == "GO"
    ]
    exporter = OnnxBundleExporter(project_root, config, mode=OnnxExportMode.INTERNAL)
    if sensor is not None:
        exported = (
            exporter.run(sensor)
            if sensor in eligible
            else {
                sensor: {
                    "sensor_channel": sensor,
                    "status": "blocked",
                    "reason": internal_test.get(sensor, {}).get(
                        "reason", "internal_test_gate_failed"
                    ),
                }
            }
        )
    else:
        exported = exporter.run()
    published_channels = _published_internal_channels(exported)
    export_root = paths.export_dir / "internal"
    status_path = export_root / "internal_export_status.json"
    status = {
        "schema_version": "internal-android-v2",
        "run_id": config.artifacts.get("run_id"),
        "validation_scope": "internal_32_participants",
        "claims_policy": "decision_support_only",
        "internal_test": internal_test,
        "onnx_export": exported,
        "internal_test_eligible_channels": eligible,
        "eligible_channels": published_channels,
        "internal_android_status": (
            "PENDING_DEVICE_PARITY" if published_channels else "NO_GO"
        ),
    }
    _atomic_write_json(status_path, status)
    return {
        "internal_test": internal_test,
        "onnx_export": exported,
        "status_path": str(status_path),
    }


class StressOnnxWorkflow:
    """Compose local ML, conditional DL handoff, and gated ONNX export."""

    def __init__(
        self,
        project_root: Path,
        config: Any,
        paths: PipelinePaths,
        *,
        config_path: Path,
        sensor: str | None = None,
        export_mode: OnnxExportMode | str = OnnxExportMode.PRODUCTION,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.paths = paths
        self.config_path = config_path
        self.sensor = sensor
        self.export_mode = OnnxExportMode(export_mode)

    @property
    def reports_root(self) -> Path:
        return _configured_path(
            self.project_root, self.config, "reports_dir", "reports"
        )

    def stages(self) -> tuple[CallablePipelineStage, ...]:
        model_reports = self.reports_root / "model_eval" / self.config.model_version
        return (
            CallablePipelineStage(
                "train",
                self.train,
                (
                    self.paths.gold_dir / "gold_features_v3.parquet",
                    self.paths.gold_dir / "data_readiness.json",
                    self.paths.gold_dir / "manifest_v3.json",
                    self.paths.validation_dir
                    / "hrv_agreement"
                    / "selection"
                    / "hrv_training_admission.json",
                ),
            ),
            CallablePipelineStage(
                "export",
                self.export,
                (
                    model_reports,
                    self.paths.gold_dir / "data_readiness.json",
                    self.paths.gold_dir / "manifest_v3.json",
                ),
            ),
        )

    def train(self, _: PipelineRunContext) -> StageResult:
        reports = train_classical_models(
            self.project_root,
            self.config,
            sensor=self.sensor,
            defer_reused_internal_test=self.export_mode is OnnxExportMode.INTERNAL,
        )
        handoff = prepare_colab_handoff(
            self.project_root,
            self.config,
            reports,
            config_path=self.config_path,
        )
        selection_contract = freeze_stress_selection_contract(
            self.project_root,
            self.config,
            self.paths,
            reports,
            allow_full_cohort_physiology=self.export_mode
            is not OnnxExportMode.INTERNAL,
        )
        model_reports = self.reports_root / "model_eval" / self.config.model_version
        outputs = [model_reports / f"{channel}_full.json" for channel in reports]
        summary = model_reports / "local_ml_summary.json"
        if summary.is_file():
            outputs.append(summary)
        if handoff is not None:
            outputs.append(handoff)
        outputs.append(selection_contract)
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

    def export(self, _: PipelineRunContext) -> StageResult:
        reports = load_local_ml_reports(
            self.project_root, self.config, sensor=self.sensor
        )
        if self.export_mode is OnnxExportMode.INTERNAL:
            # The all/stress command freezes candidates only.  The explicit
            # publish command is the sole authority allowed to read ID-29..32.
            internal_root = self.paths.export_dir / "internal"
            status_path = internal_root / "internal_export_status.json"
            status = {
                "schema_version": "internal-android-v2",
                "run_id": self.config.artifacts.get("run_id"),
                "validation_scope": "internal_32_participants",
                "claims_policy": "decision_support_only",
                "internal_android_status": "PENDING_INTERNAL_TEST",
                "channels": {
                    channel: {
                        "development_model_status": (
                            "GO"
                            if report.get("development_candidate_bundle")
                            else "NO_GO"
                        ),
                        "internal_test_status": "PENDING",
                        "reason": report.get("release_status"),
                    }
                    for channel, report in reports.items()
                },
            }
            _atomic_write_json(status_path, status)
            return StageResult(
                (status_path,),
                {
                    "export_mode": self.export_mode.value,
                    "candidate_channels": [
                        channel
                        for channel, report in reports.items()
                        if report.get("development_candidate_bundle")
                    ],
                },
            )
        if self.export_mode is OnnxExportMode.RESEARCH:
            candidates = [
                channel
                for channel, report in reports.items()
                if report.get("development_candidate_bundle")
                and report.get("holdout_evaluated") is False
                and "holdout_metrics" not in report
            ]
            empty_reason = "no_frozen_development_candidate"
        else:
            candidates = [
                channel
                for channel, report in reports.items()
                if report.get("release_status") == "release_candidate"
            ]
            empty_reason = "no_frozen_release_candidate"
        if not candidates:
            self.paths.export_dir.mkdir(parents=True, exist_ok=True)
            status_path = self.paths.export_dir / "export_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "NO_GO",
                        "reason": empty_reason,
                        "export_mode": self.export_mode.value,
                        "channels": {
                            channel: report.get("release_status", "NO_GO")
                            for channel, report in reports.items()
                        },
                        "onnx_published": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return StageResult(
                (status_path,),
                {"release_candidate_channels": [], "onnx_published": False},
            )
        exported: dict[str, dict[str, Any]] = {}
        for channel in candidates:
            result = export_selected_model(
                self.project_root,
                self.config,
                channel,
                mode=self.export_mode,
            )
            exported[channel] = dict(result[channel])
        status_path = self.paths.export_dir / "export_status.json"
        status = {
            "status": (
                "RESEARCH_READY"
                if self.export_mode is OnnxExportMode.RESEARCH
                else "RELEASE_CANDIDATE"
            ),
            "export_mode": self.export_mode.value,
            "channels": {
                channel: {
                    "artifact_status": manifest.get("artifact_status"),
                    "release_status": manifest.get("release_status"),
                    "android_ready": manifest.get("android_ready"),
                    "onnx_sha256": manifest.get("onnx_sha256"),
                    "parity": manifest.get("onnx_parity"),
                }
                for channel, manifest in exported.items()
            },
            "blocked_channels": {
                channel: report.get("release_status")
                for channel, report in reports.items()
                if channel not in exported
            },
            "onnx_published": self.export_mode is OnnxExportMode.PRODUCTION,
            "research_onnx_available": self.export_mode is OnnxExportMode.RESEARCH,
        }
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs: list[Path] = [self.paths.export_dir, status_path]
        if self.export_mode is OnnxExportMode.RESEARCH:
            research_root = self.paths.export_dir / "research"
            checklist_path = research_root / "onnx_research_checklist.json"
            checklist = {
                "schema_version": "onnx-research-checklist-v1",
                "run_id": self.config.artifacts.get("run_id"),
                "checks": {
                    "development_only": all(
                        item.get("evaluation_scope") == "development_only"
                        for item in exported.values()
                    ),
                    "no_holdout_metrics": all(
                        "holdout_metrics" not in item for item in exported.values()
                    ),
                    "minimum_100_real_vectors": all(
                        int(item.get("parity_vector_count", 0)) >= 100
                        for item in exported.values()
                    ),
                    "onnx_checker_passed": all(
                        item.get("onnx_checker_passed") is True
                        for item in exported.values()
                    ),
                    "parity_below_1e_5": all(
                        float(item.get("onnx_parity", {}).get("max_abs_diff", 1.0))
                        < 1.0e-5
                        for item in exported.values()
                    ),
                    "android_not_production_ready": all(
                        item.get("android_ready") is False
                        and item.get("deployment_scope") == "research"
                        for item in exported.values()
                    ),
                },
                "channels": sorted(exported),
                "blocked_channels": status["blocked_channels"],
            }
            checklist_path.write_text(
                json.dumps(checklist, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            markdown_path = research_root / "onnx_research_checklist.md"
            markdown_path.write_text(
                "# ONNX Research Checklist\n\n"
                + "\n".join(
                    f"- [{'x' if passed else ' '}] {name}"
                    for name, passed in checklist["checks"].items()
                )
                + "\n",
                encoding="utf-8",
            )
            comparison_paths = write_stress_run_comparison(
                self.project_root,
                str(self.config.artifacts.get("run_id")),
            )
            outputs.extend((checklist_path, markdown_path, *comparison_paths))
        return StageResult(
            tuple(outputs),
            {
                "export_mode": self.export_mode.value,
                "candidate_channels": candidates,
            },
        )


def _configured_path(project_root: Path, config: Any, key: str, default: str) -> Path:
    value = Path(config.artifacts.get(key, default))
    return value if value.is_absolute() else project_root / value


def _component_config_path(project_root: Path, config_path: Path) -> Path:
    selected = config_path if config_path.is_absolute() else project_root / config_path
    companion = selected.parent / "training.yaml"
    return (
        companion
        if companion.is_file()
        else project_root / "PipelinePython" / "configs" / "training.yaml"
    )
