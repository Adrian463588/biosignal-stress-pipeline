"""Waveform-only conditional DL contract for the canonical local pipeline.

The local Python pipeline prepares and validates this handoff.  Training remains
an explicit Google Colab T4 activity; this module never uploads data, creates
samples, pads a waveform, or treats an IBI/tabular artifact as a DL input.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .orchestration import source_tree_sha256
from .settings import PipelineConfig, pipeline_paths_from_config
from .training_protocol import outer_participant_folds

DEVELOPMENT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(1, 29))
HOLDOUT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(29, 33))
DL_ARCHITECTURES = ("1d_cnn", "tcn", "cnn_bilstm_or_gru")
HANDOFF_SCHEMA_VERSION = "colab-dl-waveform-v1"
FIXED_SEQUENCE_CONTRACT = {
    "window_seconds": 180,
    "stride_seconds": 10,
    "shape": "[batch, observed_samples, 1]",
    "mask_shape": "[batch, observed_samples]",
    "normalization": "per_window_zscore_observed_only",
}


class WaveformHandoffError(ValueError):
    """Raised when a waveform handoff or resume contract is unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise WaveformHandoffError(f"artifact_collision:{path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def _require_string_array(bundle: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in bundle:
        raise WaveformHandoffError(f"waveform_artifact_missing:{name}")
    values = np.asarray(bundle[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise WaveformHandoffError(f"waveform_artifact_invalid_string_array:{name}")
    return values.astype(str)


def validate_waveform_artifact(
    path: Path,
    *,
    expected_channel: str,
    expected_participant: str,
    expected_phase: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one real Gold waveform bundle without allowing object arrays."""

    artifact = path.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"waveform_artifact_missing:{artifact}")
    actual_sha256 = sha256_file(artifact)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise WaveformHandoffError(f"waveform_checksum_mismatch:{artifact}")
    try:
        loaded = np.load(artifact, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise WaveformHandoffError(f"waveform_artifact_unreadable:{artifact}") from exc
    with loaded as arrays:
        names = set(arrays.files)
        forbidden = sorted(
            name
            for name in names
            if any(
                token in name.lower()
                for token in ("ibi", "tabular", "ecg", "pat", "motion")
            )
        )
        if forbidden:
            raise WaveformHandoffError(
                f"waveform_only_contract_violation:{','.join(forbidden)}"
            )
        required = {
            "waveforms",
            "observed_mask",
            "interpolated_mask",
            "normalization_mean",
            "normalization_std",
            "window_ids",
            "participant_ids",
            "session_ids",
            "phases",
            "sensor_channels",
            "quality_tiers",
            "source_files",
            "source_checksums",
            "sampling_rate_hz",
            "source_sampling_rate_hz",
            "target_sampling_rate_hz",
            "window_start_utc",
            "window_end_utc",
        }
        missing = sorted(required - names)
        if missing:
            raise WaveformHandoffError(f"waveform_artifact_missing:{','.join(missing)}")
        waveforms = np.asarray(arrays["waveforms"])
        observed_mask = np.asarray(arrays["observed_mask"])
        interpolated_mask = np.asarray(arrays["interpolated_mask"])
        if waveforms.ndim != 3 or waveforms.shape[-1] != 1:
            raise WaveformHandoffError("waveform_shape_must_be_batch_samples_one")
        if observed_mask.shape != waveforms.shape[:2]:
            raise WaveformHandoffError("observed_mask_shape_mismatch")
        if interpolated_mask.shape != waveforms.shape[:2]:
            raise WaveformHandoffError("interpolated_mask_shape_mismatch")
        if waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise WaveformHandoffError("waveform_artifact_empty")
        if waveforms.dtype.kind not in {"f", "i", "u"}:
            raise WaveformHandoffError("waveform_dtype_must_be_numeric")
        if not np.isfinite(waveforms.astype(np.float64)).all():
            raise WaveformHandoffError("waveform_artifact_non_finite")
        if not np.isin(observed_mask, [0, 1]).all():
            raise WaveformHandoffError("observed_mask_must_be_binary")
        if not np.isin(interpolated_mask, [0, 1]).all():
            raise WaveformHandoffError("interpolated_mask_must_be_binary")
        normalization_mean = np.asarray(arrays["normalization_mean"])
        normalization_std = np.asarray(arrays["normalization_std"])
        if normalization_mean.shape != (waveforms.shape[0], 1):
            raise WaveformHandoffError("normalization_mean_shape_mismatch")
        if normalization_std.shape != (waveforms.shape[0], 1):
            raise WaveformHandoffError("normalization_std_shape_mismatch")
        if (
            not np.isfinite(normalization_mean).all()
            or not np.isfinite(normalization_std).all()
        ):
            raise WaveformHandoffError("normalization_parameters_non_finite")
        if np.any(normalization_std <= 0.0):
            raise WaveformHandoffError("normalization_std_must_be_positive")
        for rate_name in (
            "sampling_rate_hz",
            "source_sampling_rate_hz",
            "target_sampling_rate_hz",
        ):
            rates = np.asarray(arrays[rate_name], dtype=np.float64)
            if rates.shape != (waveforms.shape[0],) or not np.isfinite(rates).all():
                raise WaveformHandoffError(f"{rate_name}_invalid")
            if np.any(rates <= 0.0):
                raise WaveformHandoffError(f"{rate_name}_must_be_positive")
        if not np.all(observed_mask.sum(axis=1) > 0):
            raise WaveformHandoffError("waveform_row_has_no_observed_samples")
        participants = _require_string_array(arrays, "participant_ids")
        phases = _require_string_array(arrays, "phases")
        channels = _require_string_array(arrays, "sensor_channels")
        if len(participants) != waveforms.shape[0] or len(phases) != waveforms.shape[0]:
            raise WaveformHandoffError("waveform_lineage_row_count_mismatch")
        if len(channels) != waveforms.shape[0]:
            raise WaveformHandoffError("waveform_channel_row_count_mismatch")
        if set(participants) != {expected_participant}:
            raise WaveformHandoffError("waveform_participant_lineage_mismatch")
        if set(phases) != {expected_phase}:
            raise WaveformHandoffError("waveform_phase_lineage_mismatch")
        if set(channels) != {expected_channel}:
            raise WaveformHandoffError("waveform_channel_lineage_mismatch")
        return {
            "path": str(artifact),
            "sha256": actual_sha256,
            "shape": [int(value) for value in waveforms.shape],
            "window_count": int(waveforms.shape[0]),
            "sample_count": int(waveforms.shape[1]),
            "participant_id": expected_participant,
            "phase": expected_phase,
            "sensor_channel": expected_channel,
            "interpolated_sample_count": int(interpolated_mask.sum()),
            "sampling_rate_hz_median": float(
                np.median(np.asarray(arrays["sampling_rate_hz"], dtype=np.float64))
            ),
            "target_sampling_rate_hz_median": float(
                np.median(
                    np.asarray(arrays["target_sampling_rate_hz"], dtype=np.float64)
                )
            ),
        }


@dataclass(frozen=True)
class WaveformDlHandoff:
    """Checksum-bound development waveform handoff for one or more channels."""

    run_id: str
    gold_manifest_path: str
    gold_manifest_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    config_fingerprint: str
    waveform_artifacts: tuple[Mapping[str, Any], ...]
    dl_channels: tuple[str, ...]
    outer_folds: tuple[tuple[str, ...], ...]
    run_contract_sha256: str = ""
    inner_group_splits: int = 5
    schema_version: str = HANDOFF_SCHEMA_VERSION
    t4_required: bool = True
    input_contract: Mapping[str, Any] = field(
        default_factory=lambda: dict(FIXED_SEQUENCE_CONTRACT)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "gold_manifest_path": self.gold_manifest_path,
            "gold_manifest_sha256": self.gold_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "config_fingerprint": self.config_fingerprint,
            "run_contract_sha256": self.run_contract_sha256,
            "waveform_artifacts": [dict(item) for item in self.waveform_artifacts],
            "dl_channels": list(self.dl_channels),
            "outer_folds": [list(fold) for fold in self.outer_folds],
            "inner_group_splits": self.inner_group_splits,
            "t4_required": self.t4_required,
            "input_contract": dict(self.input_contract),
        }

    @property
    def fold_contract_sha256(self) -> str:
        return _json_sha256(
            {
                "development": list(DEVELOPMENT_PARTICIPANTS),
                "holdout": list(HOLDOUT_PARTICIPANTS),
                "outer_folds": [list(fold) for fold in self.outer_folds],
                "inner_group_splits": self.inner_group_splits,
            }
        )

    def validate(self) -> None:
        if self.schema_version != HANDOFF_SCHEMA_VERSION:
            raise WaveformHandoffError("unsupported_waveform_handoff_schema")
        if not self.run_id:
            raise WaveformHandoffError("waveform_handoff_run_id_required")
        for name in (
            "gold_manifest_sha256",
            "training_config_sha256",
            "source_tree_sha256",
            "config_fingerprint",
            "run_contract_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise WaveformHandoffError(f"invalid_fingerprint:{name}")
        if not self.t4_required:
            raise WaveformHandoffError("conditional_dl_requires_colab_t4")
        if self.inner_group_splits != 5:
            raise WaveformHandoffError("inner_group_kfold_must_be_five")
        expected_folds = outer_participant_folds(DEVELOPMENT_PARTICIPANTS)
        if self.outer_folds != expected_folds:
            raise WaveformHandoffError("outer_fold_contract_mismatch")
        if not self.dl_channels or any(
            channel not in {"ppg_green", "ppg_ir", "ppg_red"}
            for channel in self.dl_channels
        ):
            raise WaveformHandoffError("invalid_dl_channel_set")
        if self.input_contract != FIXED_SEQUENCE_CONTRACT:
            raise WaveformHandoffError("waveform_input_contract_mismatch")
        groups = {
            (
                str(item.get("participant_id")),
                str(item.get("phase")),
                str(item.get("sensor_channel")),
            )
            for item in self.waveform_artifacts
        }
        expected_groups = {
            (participant, phase, channel)
            for participant in DEVELOPMENT_PARTICIPANTS
            for phase in ("relax", "task")
            for channel in self.dl_channels
        }
        if (
            len(self.waveform_artifacts) != len(expected_groups)
            or groups != expected_groups
        ):
            raise WaveformHandoffError(
                "waveform_handoff_requires_complete_development_groups"
            )
        by_channel: dict[str, set[int]] = {
            channel: set() for channel in self.dl_channels
        }
        for item in self.waveform_artifacts:
            channel = str(item.get("sensor_channel"))
            participant = str(item.get("participant_id"))
            phase = str(item.get("phase"))
            if not str(item.get("relative_path", "")):
                raise WaveformHandoffError("waveform_contract_relative_path_missing")
            colab_path = str(item.get("colab_relative_path", ""))
            expected_colab_path = f"gold_waveforms/{channel}/{participant}/{phase}.npz"
            if colab_path != expected_colab_path:
                raise WaveformHandoffError("waveform_contract_colab_path_mismatch")
            shape = item.get("shape")
            if not isinstance(shape, list) or len(shape) != 3 or int(shape[-1]) != 1:
                raise WaveformHandoffError("waveform_contract_shape_invalid")
            by_channel[channel].add(int(shape[1]))
        if any(len(lengths) != 1 for lengths in by_channel.values()):
            raise WaveformHandoffError("waveform_channel_shape_not_fixed")


def _config_path(project_root: Path, config_path: Path) -> Path:
    selected = config_path if config_path.is_absolute() else project_root / config_path
    companion = selected.parent / "training.yaml"
    return (
        companion
        if companion.is_file()
        else project_root / "PipelinePython" / "configs" / "training.yaml"
    )


def build_waveform_handoff(
    project_root: Path,
    config: PipelineConfig,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    config_path: Path,
) -> Path | None:
    """Build a development-only handoff from current-run Gold waveforms."""

    required_channels = tuple(
        channel
        for channel, report in reports.items()
        if bool(report.get("requires_dl"))
    )
    if not required_channels:
        return None
    paths = pipeline_paths_from_config(project_root, config)
    run_id = str(config.artifacts.get("run_id", ""))
    if not run_id:
        raise WaveformHandoffError("canonical_waveform_handoff_requires_run_id")
    manifest_path = paths.gold_dir / "manifest_v3.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"current_run_gold_manifest_missing:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("run_id", "")) != run_id:
        raise WaveformHandoffError("gold_manifest_run_id_mismatch")
    contract_path = paths.artifacts_root / "run_contract_fingerprint.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"run_contract_fingerprint_missing:{contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if str(contract.get("run_id", "")) != run_id:
        raise WaveformHandoffError("run_contract_run_id_mismatch")
    if str(contract.get("source_tree_sha256", "")).startswith("UNAVAILABLE:"):
        raise WaveformHandoffError("run_contract_source_tree_unavailable")
    current_source_tree = source_tree_sha256(project_root)
    if str(contract.get("source_tree_sha256")) != current_source_tree:
        raise WaveformHandoffError("run_contract_source_tree_mismatch")
    config_fingerprint = str(contract.get("config_fingerprint", ""))
    if len(config_fingerprint) != 64:
        raise WaveformHandoffError("run_contract_config_fingerprint_missing")
    run_contract_sha256 = sha256_file(contract_path)
    gold_root = paths.gold_dir.resolve()
    artifacts: list[dict[str, Any]] = []
    for item in manifest.get("waveform_contracts", ()):
        if str(item.get("status", "")).upper() != "AVAILABLE":
            continue
        channel = str(item.get("sensor_channel", ""))
        participant = str(item.get("participant_id", ""))
        phase = str(item.get("phase", ""))
        if (
            channel not in required_channels
            or participant not in DEVELOPMENT_PARTICIPANTS
        ):
            continue
        relative_path = str(item.get("path", ""))
        if not relative_path:
            raise WaveformHandoffError("waveform_contract_path_missing")
        artifact = (project_root / relative_path).resolve()
        try:
            artifact.relative_to(gold_root)
        except ValueError as exc:
            raise WaveformHandoffError(
                "waveform_artifact_outside_current_gold"
            ) from exc
        evidence = validate_waveform_artifact(
            artifact,
            expected_channel=channel,
            expected_participant=participant,
            expected_phase=phase,
            expected_sha256=str(item.get("sha256", "")) or None,
        )
        evidence["relative_path"] = str(artifact.relative_to(project_root)).replace(
            "\\", "/"
        )
        evidence["colab_relative_path"] = (
            f"gold_waveforms/{channel}/{participant}/{phase}.npz"
        )
        artifacts.append(evidence)
    handoff = WaveformDlHandoff(
        run_id=run_id,
        gold_manifest_path=str(manifest_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        gold_manifest_sha256=sha256_file(manifest_path),
        training_config_sha256=sha256_file(_config_path(project_root, config_path)),
        source_tree_sha256=current_source_tree,
        config_fingerprint=config_fingerprint,
        waveform_artifacts=tuple(
            sorted(artifacts, key=lambda item: str(item["relative_path"]))
        ),
        dl_channels=tuple(sorted(required_channels)),
        outer_folds=outer_participant_folds(DEVELOPMENT_PARTICIPANTS),
        run_contract_sha256=run_contract_sha256,
    )
    handoff.validate()
    reports_root = Path(config.artifacts.get("reports_dir", "reports"))
    reports_root = (
        reports_root if reports_root.is_absolute() else project_root / reports_root
    )
    destination = (
        reports_root
        / "model_eval"
        / "ml"
        / "colab_input"
        / "colab_dl_waveform_manifest.json"
    )
    return _atomic_json(destination, handoff.to_dict())


def load_waveform_handoff(path: Path) -> WaveformDlHandoff:
    payload = json.loads(path.read_text(encoding="utf-8"))
    handoff = WaveformDlHandoff(
        run_id=str(payload.get("run_id", "")),
        gold_manifest_path=str(payload.get("gold_manifest_path", "")),
        gold_manifest_sha256=str(payload.get("gold_manifest_sha256", "")),
        training_config_sha256=str(payload.get("training_config_sha256", "")),
        source_tree_sha256=str(payload.get("source_tree_sha256", "")),
        config_fingerprint=str(payload.get("config_fingerprint", "")),
        run_contract_sha256=str(payload.get("run_contract_sha256", "")),
        waveform_artifacts=tuple(payload.get("waveform_artifacts", ())),
        dl_channels=tuple(payload.get("dl_channels", ())),
        outer_folds=tuple(tuple(fold) for fold in payload.get("outer_folds", ())),
        inner_group_splits=int(payload.get("inner_group_splits", 0)),
        schema_version=str(payload.get("schema_version", "")),
        t4_required=bool(payload.get("t4_required", False)),
        input_contract=payload.get("input_contract", {}),
    )
    handoff.validate()
    return handoff


def require_colab_t4(gpu_name: str | None) -> None:
    """Fail explicitly unless the conditional DL runtime is an NVIDIA T4."""

    normalized = str(gpu_name or "").upper()
    if "T4" not in normalized:
        raise WaveformHandoffError("conditional_dl_requires_nvidia_t4")


def validate_dl_resume_manifest(
    resume_manifest: Mapping[str, Any],
    handoff: WaveformDlHandoff,
    *,
    channel: str,
    architecture: str,
    checkpoint_path: Path | None = None,
) -> None:
    """Reject a checkpoint created from a different Gold/config/fold contract."""

    handoff.validate()
    if channel not in handoff.dl_channels:
        raise WaveformHandoffError("resume_channel_not_handed_off")
    if architecture not in DL_ARCHITECTURES:
        raise WaveformHandoffError("resume_architecture_not_supported")
    expected = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "run_id": handoff.run_id,
        "gold_manifest_sha256": handoff.gold_manifest_sha256,
        "training_config_sha256": handoff.training_config_sha256,
        "source_tree_sha256": handoff.source_tree_sha256,
        "config_fingerprint": handoff.config_fingerprint,
        "run_contract_sha256": handoff.run_contract_sha256,
        "fold_contract_sha256": handoff.fold_contract_sha256,
        "channel": channel,
        "architecture": architecture,
    }
    for key, value in expected.items():
        if resume_manifest.get(key) != value:
            raise WaveformHandoffError(f"resume_manifest_mismatch:{key}")
    if checkpoint_path is not None:
        expected_checksum = str(resume_manifest.get("checkpoint_sha256", ""))
        if expected_checksum != sha256_file(checkpoint_path):
            raise WaveformHandoffError("checkpoint_checksum_mismatch")
    completed = tuple(resume_manifest.get("completed_outer_folds", ()))
    if any(fold not in range(7) for fold in completed):
        raise WaveformHandoffError("resume_contains_invalid_outer_fold")
