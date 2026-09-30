"""Evidence-only comparison for selective ECG--PPG pipeline restoration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


LOCKED_RAW_CORPUS_SHA256 = (
    "8d156cc4932826c3e022a6867552b5b720685ef336ce22c049a8036bc50d3561"
)
HISTORICAL_RUN_IDS = (
    "20260809T193919Z",
    "20260827T084037Z",
    "20260828T220000Z",
)
REQUIRED_METRICS = ("beat_f1", "ibi_mae_ms", "hr_mae_bpm")


@dataclass(frozen=True)
class RestoreComparisonOutputs:
    """Durable comparison evidence written for one canonical run."""

    json_path: Path
    markdown_path: Path
    report_path: Path


def build_restore_comparison(
    project_root: Path,
    current_run_id: str,
    current_metrics_path: Path,
    *,
    historical_run_ids: Iterable[str] = HISTORICAL_RUN_IDS,
) -> RestoreComparisonOutputs:
    """Compare immutable run artifacts without reusing them as pipeline inputs."""

    root = Path(project_root)
    current_metrics_path = Path(current_metrics_path)
    if not current_metrics_path.is_file():
        raise FileNotFoundError(current_metrics_path)

    run_ids = [*dict.fromkeys((*historical_run_ids, current_run_id))]
    runs: list[dict[str, Any]] = []
    for run_id in run_ids:
        metrics_path = (
            current_metrics_path
            if run_id == current_run_id
            else root
            / "data"
            / "artifacts"
            / "runs"
            / run_id
            / "validation"
            / "ppg_ecg_240s"
            / "ppg_ecg_metrics_240s.csv"
        )
        runs.append(_summarize_run(root, run_id, metrics_path))

    current = next(item for item in runs if item["run_id"] == current_run_id)
    baselines = {
        item["run_id"]: item for item in runs if item["run_id"] != current_run_id
    }
    payload = {
        "schema_version": "selective-restore-comparison-v1",
        "current_run_id": current_run_id,
        "historical_reference_status": "historical_source_unverifiable",
        "historical_reference_commits": {
            "primary": "c1245c5f20acdc492b72aa7abbdd5de1fc10990c",
            "parent": "58624c02886cc0d422b94b11d181d8e0e1b47e33",
        },
        "locked_raw_corpus_sha256": LOCKED_RAW_CORPUS_SHA256,
        "raw_corpus_match": all(
            item.get("raw_corpus_sha256") == LOCKED_RAW_CORPUS_SHA256
            for item in runs
            if item["artifact_status"] == "available"
        ),
        "runs": runs,
        "deltas": {
            run_id: _delta(current, baseline)
            for run_id, baseline in baselines.items()
            if baseline["artifact_status"] == "available"
        },
        "interpretation": {
            "augmentation_used_for_peak_validation": False,
            "coverage_target_is_mandatory": False,
            "detector_gate_unchanged": True,
            "no_synthetic_peak_or_waveform": True,
        },
    }

    output_dir = current_metrics_path.parent
    json_path = output_dir / "restore_comparison.json"
    markdown_path = output_dir / "restore_comparison.md"
    report_path = root / "reports" / "runs" / current_run_id / "restore_comparison.md"
    _write_json(json_path, payload)
    markdown = _markdown(payload)
    _write_text(markdown_path, markdown)
    _write_text(report_path, markdown)
    return RestoreComparisonOutputs(json_path, markdown_path, report_path)


def _summarize_run(
    project_root: Path, run_id: str, metrics_path: Path
) -> dict[str, Any]:
    identity = _run_identity(project_root, run_id)
    base: dict[str, Any] = {
        "run_id": run_id,
        "metrics_path": _relative(project_root, metrics_path),
        **identity,
    }
    if not metrics_path.is_file():
        return {**base, "artifact_status": "missing"}

    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 192:
        return {
            **base,
            "artifact_status": "invalid_row_count",
            "row_count": int(len(metrics)),
        }

    valid = _valid_mask(metrics)
    computable = _metric_computable_mask(metrics)
    strict = _strict_quality_mask(metrics)
    if strict is None and run_id != "20260809T193919Z":
        # Run 27/28 used morphology as a hard validation gate, so their valid
        # rows are the only defensible strict-quality approximation. Run 09
        # predates that gate and remains explicitly unavailable.
        strict = valid
    channels = {
        str(channel): _channel_summary(
            group,
            infer_strict_from_valid=run_id != "20260809T193919Z",
        )
        for channel, group in metrics.groupby("sensor_channel", sort=True)
    }
    return {
        **base,
        "artifact_status": "available",
        "row_count": int(len(metrics)),
        "analysis_valid_count": int(valid.sum()),
        "excluded_count": int((~valid).sum()),
        "metric_computable_count": int(computable.sum()),
        "strict_quality_count": int(strict.sum()) if strict is not None else None,
        "ecg_interpolated_sample_count": _integer_sum(
            metrics, "ecg_interpolated_sample_count"
        ),
        "ppg_interpolated_sample_count": _integer_sum(
            metrics, "ppg_interpolated_sample_count"
        ),
        "detector_status": _detector_status(project_root, run_id),
        "channels": channels,
        "exclusions": _exclusions(metrics, valid),
    }


def _run_identity(project_root: Path, run_id: str) -> dict[str, Any]:
    run_root = project_root / "data" / "artifacts" / "runs" / run_id
    state = _read_json(run_root / "run_state.json")
    contract = _read_json(run_root / "run_contract_fingerprint.json")
    source_tree = contract.get("source_tree_sha256") or state.get("source_tree_sha256")
    return {
        "raw_corpus_sha256": state.get("raw_corpus_sha256"),
        "source_tree_sha256": source_tree,
        "config_sha256": state.get("config_sha256"),
        "source_status": (
            "source_fingerprinted" if source_tree else "historical_source_unverifiable"
        ),
    }


def _channel_summary(
    frame: pd.DataFrame, *, infer_strict_from_valid: bool
) -> dict[str, Any]:
    valid = _valid_mask(frame)
    computable = _metric_computable_mask(frame)
    strict = _strict_quality_mask(frame)
    if strict is None and infer_strict_from_valid:
        strict = valid
    selected = frame.loc[valid & computable]
    return {
        "expected_combinations": int(len(frame)),
        "analysis_valid_count": int(valid.sum()),
        "metric_computable_count": int(computable.sum()),
        "strict_quality_count": int(strict.sum()) if strict is not None else None,
        "median_beat_f1": _median(selected, "beat_f1"),
        "median_ibi_mae_ms": _median(selected, "ibi_mae_ms"),
        "median_hr_mae_bpm": _median(selected, "hr_mae_bpm"),
    }


def _delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis_valid_count": _subtract(current, baseline, "analysis_valid_count"),
        "metric_computable_count": _subtract(
            current, baseline, "metric_computable_count"
        ),
        "channels": {
            channel: {
                metric: _subtract(
                    current["channels"][channel], baseline["channels"][channel], metric
                )
                for metric in (
                    "analysis_valid_count",
                    "median_beat_f1",
                    "median_ibi_mae_ms",
                    "median_hr_mae_bpm",
                )
            }
            for channel in current.get("channels", {})
            if channel in baseline.get("channels", {})
        },
    }


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "validation_status" in frame:
        return frame["validation_status"].astype(str).eq("valid")
    return (
        frame.get("status", pd.Series("excluded", index=frame.index))
        .astype(str)
        .eq("ok")
    )


def _metric_computable_mask(frame: pd.DataFrame) -> pd.Series:
    if "metric_status" in frame:
        return frame["metric_status"].astype(str).isin({"computed", "reported"})
    finite = pd.Series(True, index=frame.index)
    for column in REQUIRED_METRICS:
        if column not in frame:
            return pd.Series(False, index=frame.index)
        finite &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    return finite


def _strict_quality_mask(frame: pd.DataFrame) -> pd.Series | None:
    if "strict_quality_eligible" not in frame:
        return None
    values = frame["strict_quality_eligible"]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _exclusions(frame: pd.DataFrame, valid: pd.Series) -> dict[str, int]:
    excluded = frame.loc[~valid]
    if excluded.empty or "exclusion_reason" not in excluded:
        return {}
    reasons = (
        excluded["exclusion_reason"]
        .fillna("unspecified_exclusion")
        .astype(str)
        .replace("", "unspecified_exclusion")
        .value_counts()
    )
    return {str(reason): int(count) for reason, count in reasons.items()}


def _detector_status(project_root: Path, run_id: str) -> str:
    path = (
        project_root
        / "data"
        / "artifacts"
        / "runs"
        / run_id
        / "validation"
        / "detector_selection"
        / "detector_selection.json"
    )
    payload = _read_json(path)
    return str(
        payload.get("detector_status")
        or payload.get("selection_status")
        or payload.get("overall_status")
        or "unavailable"
    )


def _integer_sum(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int(values.sum())


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _subtract(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> Any:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selective peak-pipeline restoration comparison",
        "",
        f"Current run: `{payload['current_run_id']}`",
        "",
        "| Run | Source | Analysis valid | Metric computable | Strict quality | ECG interp | PPG interp | Detector |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in payload["runs"]:
        if run["artifact_status"] != "available":
            lines.append(
                f"| {run['run_id']} | {run['source_status']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |"
            )
            continue
        strict = run["strict_quality_count"]
        lines.append(
            f"| {run['run_id']} | {run['source_status']} | "
            f"{run['analysis_valid_count']}/192 | {run['metric_computable_count']}/192 | "
            f"{strict if strict is not None else 'unavailable'} | "
            f"{run['ecg_interpolated_sample_count']} | "
            f"{run['ppg_interpolated_sample_count']} | {run['detector_status']} |"
        )
    lines.extend(
        [
            "",
            "Coverage is reported separately from strict-quality eligibility.",
            "No augmentation contributes to peak-validation metrics.",
            "Historical run source remains unverifiable because it lacks a source-tree fingerprint.",
        ]
    )
    return "\n".join(lines) + "\n"


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, allow_nan=False))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
