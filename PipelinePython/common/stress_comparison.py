"""Run-scoped comparison for stress-model and ONNX evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


COMPARISON_BASELINE_RUNS = ("20260830T065109Z", "20260810T031618Z")
CHANNELS = ("ppg_green", "ppg_ir", "ppg_red")


def write_stress_run_comparison(
    project_root: Path,
    current_run_id: str,
) -> tuple[Path, Path, Path]:
    """Write a read-only comparison using only run-scoped artifacts."""

    root = Path(project_root)
    run_ids = (*COMPARISON_BASELINE_RUNS, current_run_id)
    summaries = {run_id: _summarize_run(root, run_id) for run_id in run_ids}
    payload = {
        "schema_version": "stress-onnx-run-comparison-v2",
        "current_run_id": current_run_id,
        "baseline_run_ids": list(COMPARISON_BASELINE_RUNS),
        "historical_reference_status": "historical_source_unverifiable",
        "runs": summaries,
        "comparison_policy": {
            "run_scoped_artifacts_only": True,
            "global_model_index_used": False,
            "holdout_metrics_used_for_selection": False,
            "current_run_used_as_training_input": False,
        },
    }

    export_root = root / "data" / "artifacts" / "runs" / current_run_id / "export"
    research_root = export_root / "research"
    json_path = research_root / "run_comparison.json"
    markdown_path = research_root / "run_comparison.md"
    report_path = (
        root / "reports" / "runs" / current_run_id / "stress_onnx_comparison.md"
    )
    _write_json(json_path, payload)
    markdown = _to_markdown(payload)
    _write_text(markdown_path, markdown)
    _write_text(report_path, markdown)
    return json_path, markdown_path, report_path


def _summarize_run(root: Path, run_id: str) -> dict[str, Any]:
    run_root = root / "data" / "artifacts" / "runs" / run_id
    export_status = _read_json(run_root / "export" / "export_status.json")
    readiness = _read_json(run_root / "gold" / "data_readiness.json")
    research_index = _read_json(
        run_root / "export" / "research" / "research_model_index.json"
    )
    models = _models_from_index(research_index)
    if not models:
        models = _models_from_candidates(run_root)

    export_channels = export_status.get("channels", {})
    channels: dict[str, Any] = {}
    for channel in CHANNELS:
        channel_status = export_channels.get(channel, "UNAVAILABLE")
        if not isinstance(channel_status, Mapping):
            channel_status = {"status": str(channel_status)}
        channels[channel] = {
            "export_status": dict(channel_status),
            "model": models.get(channel, {"status": "UNAVAILABLE"}),
        }

    artifact_exists = bool(export_status or readiness or research_index or models)
    return {
        "run_id": run_id,
        "artifact_status": "AVAILABLE" if artifact_exists else "UNAVAILABLE",
        "export_status": export_status.get("status", "UNAVAILABLE"),
        "export_mode": export_status.get("export_mode"),
        "onnx_published": export_status.get("onnx_published"),
        "research_onnx_available": export_status.get("research_onnx_available"),
        "run_scoped_research_index": bool(research_index),
        "hrv_validation_decision": readiness.get("hrv_validation_decision"),
        "training_status": readiness.get("training_status"),
        "model_release_status": readiness.get("model_release_status"),
        "channels": channels,
    }


def _models_from_index(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = index.get("models", [])
    if not isinstance(rows, list):
        return {}
    models: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        channel = row.get("sensor_channel")
        if channel not in CHANNELS:
            continue
        files = row.get("files", {})
        model_file = files.get("model", {}) if isinstance(files, Mapping) else {}
        models[str(channel)] = {
            "status": "AVAILABLE",
            "model_version": row.get("model_version"),
            "performance_tier": row.get("performance_tier"),
            "balanced_accuracy_development": row.get("balanced_accuracy_development"),
            "f1_development": row.get("f1_development"),
            "roc_auc_development": row.get("roc_auc_development"),
            "window_seconds": row.get("window_seconds"),
            "stride_seconds": row.get("stride_seconds"),
            "artifact_status": row.get("artifact_status"),
            "release_status": row.get("release_status"),
            "android_ready": row.get("android_ready"),
            "evaluation_scope": row.get("evaluation_scope"),
            "onnx_sha256": (
                model_file.get("sha256") if isinstance(model_file, Mapping) else None
            ),
        }
    return models


def _models_from_candidates(run_root: Path) -> dict[str, dict[str, Any]]:
    model_root = run_root / "models" / "stress"
    models: dict[str, dict[str, Any]] = {}
    for path in sorted(model_root.rglob("candidate_manifest.json")):
        if path.parent.name != "development_candidate":
            continue
        channel = path.relative_to(model_root).parts[0]
        if channel not in CHANNELS or channel in models:
            continue
        payload = _read_json(path)
        metrics = payload.get("development_metrics", {})
        models[channel] = {
            "status": "AVAILABLE",
            "model_version": payload.get("model_version"),
            "performance_tier": payload.get("performance_tier"),
            "balanced_accuracy_development": metrics.get("balanced_accuracy"),
            "f1_development": metrics.get("f1"),
            "roc_auc_development": metrics.get("roc_auc"),
            "window_seconds": payload.get("window_seconds"),
            "stride_seconds": payload.get("stride_seconds"),
            "artifact_status": payload.get("artifact_status"),
            "release_status": payload.get("release_status"),
            "android_ready": payload.get("android_ready"),
            "evaluation_scope": payload.get("evaluation_scope"),
            "onnx_sha256": None,
        }
    return models


def _to_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Stress ONNX run comparison",
        "",
        f"Current run: `{payload['current_run_id']}`",
        "",
        "| Run | Export | Run-scoped research index | Green | IR | Red |",
        "|---|---|---|---|---|---|",
    ]
    runs = payload.get("runs", {})
    for run_id in (*payload["baseline_run_ids"], payload["current_run_id"]):
        run = runs.get(run_id, {})
        channels = run.get("channels", {})
        cells = [_model_cell(channels.get(channel, {})) for channel in CHANNELS]
        lines.append(
            f"| {run_id} | {run.get('export_status', 'UNAVAILABLE')} | "
            f"{run.get('run_scoped_research_index', False)} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Metrics are development-only when present; no holdout result is used for selection.",
            "The comparison reads run-scoped artifacts and never falls back to the global model index.",
            "Production remains fail-closed unless an independently bound release candidate exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def _model_cell(channel: Mapping[str, Any]) -> str:
    model = channel.get("model", {})
    if not isinstance(model, Mapping) or model.get("status") != "AVAILABLE":
        return str(model.get("status", "UNAVAILABLE"))
    ba = _format_number(model.get("balanced_accuracy_development"))
    f1 = _format_number(model.get("f1_development"))
    auc = _format_number(model.get("roc_auc_development"))
    window = model.get("window_seconds")
    stride = model.get("stride_seconds")
    return f"{ba}/{f1}/{auc} ({window}/{stride}s)"


def _format_number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
