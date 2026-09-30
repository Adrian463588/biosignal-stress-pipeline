"""Idempotent local pipeline with sensor-specific immutable raw sources."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .anchors import AnchorMapper, AnchorWindowCoverageError, AnchorWindowSlicer
from .contracts import MODEL_PHASES, SignalFrameContract, SynchronizationResult
from .feature_policy import validate_feature_order
from .features_v4 import extract_ppg_features
from .ingestion import (
    DataLoader,
    ParticipantSession,
    get_sampling_rate_hz,
    load_signal_file,
)
from .config import (
    CONDITIONS,
    filter_config_for_channel,
    load_madc_config,
    load_sqa_config,
)
from .preprocessing import (
    PpgPreprocessor,
    SignalFilter,
    MADC,
)
from .raw_sync import ClockCorrection, validate_ecg_against_ppg
from .peak_detection import PpgPeakDetector
from .settings import PipelineConfig
from .sqi import PostPeakSQA, PrePeakSQA, RawSignalSQA
from .training_protocol import channel_group_readiness

logger = logging.getLogger(__name__)

_SENSOR_ATTRS: tuple[tuple[str, str, float], ...] = (
    ("ecg", "ecg", 130.0),
    ("ppg_ir", "ppg_ir", 100.0),
    ("ppg_red", "ppg_red", 100.0),
    ("ppg_green", "ppg_green", 25.0),
)
_FOLDER_PHASE = {"Relaks": "relax", "Stress": "task"}
_LABELS = {"relax": 0, "task": 1}
_QUALITY_POLICY_VERSION = "ppg_lightweight_v2_acceptable_training"
_STRICT_QUALITY_TIERS = frozenset({"excellent", "good"})


def _sync_exclusion(
    *,
    participant_id: str,
    recording_id: str,
    reason: str,
    ecg_sample_count: int,
    reference_channel: str | None,
) -> tuple[ClockCorrection, SynchronizationResult]:
    """Keep one malformed session traceable without aborting the batch."""

    correction = ClockCorrection(
        corrected_unix_ms=np.full(ecg_sample_count, np.nan, dtype=np.float64),
        alpha=1.0,
        beta_seconds=0.0,
        excluded=True,
        exclusion_reason=reason,
        timebase_source="unavailable",
        observed_sample_count=ecg_sample_count,
        clock_selection_basis="metadata_only",
    )
    result = SynchronizationResult(
        participant_id=participant_id,
        recording_id=recording_id,
        alpha=1.0,
        beta_seconds=0.0,
        method="unavailable",
        confidence=0.0,
        overlap_seconds=0.0,
        ecg_peak_count=0,
        ppg_peak_count=0,
        matched_peak_count=0,
        median_pat_ms=None,
        excluded=True,
        exclusion_reason=reason,
        clock_selection_basis="metadata_only",
        observed_sample_count=ecg_sample_count,
        reference_signal_role="diagnostic_only",
        ppg_diagnostic_reason=(
            f"reference_channel={reference_channel or 'unavailable'}"
        ),
        clock_excluded=True,
        clock_exclusion_reason=reason,
        diagnostic_reference_channel=reference_channel,
        timebase_source="unavailable",
    )
    return correction, result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_mirror(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Write a human-readable CSV mirror beside a typed parquet artifact."""

    csv_path = parquet_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return csv_path


def _actual_sampling_rate(
    timestamps_ms: NDArray[np.floating], fallback: float
) -> float:
    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    finite = timestamps[np.isfinite(timestamps)]
    differences = np.diff(np.sort(finite))
    positive = differences[differences > 0.0]
    if positive.size == 0:
        return float(fallback)
    value = 1000.0 / float(np.median(positive))
    return value if np.isfinite(value) and value > 0.0 else float(fallback)


def _epoch_milliseconds_to_utc(values: NDArray[np.floating]) -> pd.DatetimeIndex:
    """Convert a finite epoch-millisecond timeline without float overflow."""

    epoch_ms = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(epoch_ms).all():
        raise ValueError("corrected_ecg_timestamp_nonfinite")
    max_epoch_ms = np.iinfo(np.int64).max / 1_000_000.0
    min_epoch_ms = np.iinfo(np.int64).min / 1_000_000.0
    if np.any(epoch_ms < min_epoch_ms) or np.any(epoch_ms > max_epoch_ms):
        raise ValueError("corrected_ecg_timestamp_out_of_datetime_range")
    return pd.to_datetime(
        np.rint(epoch_ms).astype(np.int64),
        unit="ms",
        utc=True,
    )


def _metadata_float(metadata: Mapping[str, str], key: str) -> float | None:
    raw = str(metadata.get(key, "")).lower().replace("seconds", "").replace("sec", "")
    try:
        return float(raw.strip())
    except ValueError:
        return None


def _normalize_observed_waveform_batch(
    waveforms: NDArray[np.floating],
    interpolated_mask: NDArray[np.integer],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize finite Silver waveforms using observed samples only.

    The returned mask preserves the distinction between samples observed by
    the sensor and samples filled by the bounded Silver interpolation policy.
    No value is created here and no missing value is replaced with a numeric
    sentinel.
    """

    values = np.asarray(waveforms, dtype=np.float32)
    interpolated = np.asarray(interpolated_mask)
    if values.ndim != 2 or interpolated.shape != values.shape:
        raise ValueError("waveform_batch_shape_mismatch")
    if not np.isin(interpolated, [0, 1]).all():
        raise ValueError("interpolated_mask_must_be_binary")
    if not np.isfinite(values).all():
        raise ValueError("waveform_batch_must_be_finite")

    observed_mask = (interpolated == 0).astype(np.uint8)
    observed_counts = observed_mask.sum(axis=1, keepdims=True)
    if np.any(observed_counts == 0):
        raise ValueError("waveform_row_has_no_observed_samples")

    observed = observed_mask.astype(bool)
    observed_values = np.where(observed, values, 0.0)
    means = observed_values.sum(axis=1, keepdims=True) / observed_counts
    centered = np.where(observed, values - means, 0.0)
    stds = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / observed_counts)
    stds = np.maximum(stds, np.finfo(np.float32).eps)
    normalized = (values - means) / stds
    return normalized.astype(np.float32), observed_mask, means, stds


class LocalBiosignalPipeline:
    """Local stage runner with explicit physical boundaries and contracts."""

    def __init__(self, project_root: Path, config: PipelineConfig) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.raw_root = config.raw_source.renamed_dir
        self.trimmed_root = config.raw_source.trimmed_dir
        self.bronze_dir = self._configured_path("bronze_dir", "data/bronze")
        self.sync_dir = self._configured_path("sync_dir", "data/sync")
        self.segmented_dir = self._configured_path("segmented_dir", "data/segmented")
        self.silver_dir = self._configured_path("silver_dir", "data/silver")
        self.gold_dir = self._configured_path("gold_dir", "data/gold")
        self.report_dir = self._configured_path("reports_dir", "reports")
        sqa_config = load_sqa_config()
        self._sqa_config = sqa_config
        self._sqa = RawSignalSQA(
            diagnostic_neurokit=bool(
                sqa_config.get("neurokit_quality_diagnostic", False)
            )
        )
        self._pre_peak_sqa = PrePeakSQA(self._sqa)
        self._post_peak_sqa = PostPeakSQA(
            self._sqa,
            minimum_peak_count=self.config.peak_detection.minimum_peak_count,
            minimum_physiological_ibi_ratio=(
                self.config.peak_detection.minimum_physiological_ibi_ratio
            ),
            physiological_ibi_min_ms=(
                self.config.peak_detection.physiological_ibi_min_ms
            ),
            physiological_ibi_max_ms=(
                self.config.peak_detection.physiological_ibi_max_ms
            ),
        )
        self._ppg_peak_detector = PpgPeakDetector(self.config.peak_detection)
        self._ppg_preprocessor = PpgPreprocessor()
        self._continuous_rate_contract = (
            self.config.preprocessing_version == "v3.1.0_continuous25_ir100"
        )
        self._madc = MADC(load_madc_config())
        self._window_slicer = AnchorWindowSlicer(
            source_phase_duration_seconds=self.config.window.source_phase_duration_seconds,
            edge_trim_start_seconds=self.config.window.edge_trim_start_seconds,
            edge_trim_end_seconds=self.config.window.edge_trim_end_seconds,
            analysis_phase_duration_seconds=self.config.window.analysis_phase_duration_seconds,
        )

    def _configured_path(self, key: str, default: str) -> Path:
        value = Path(self.config.artifacts.get(key, default))
        return value if value.is_absolute() else self.root / value

    def _participant_ids(self, participant: str | None = None) -> tuple[str, ...]:
        if participant is not None:
            assert (self.raw_root / participant).is_dir(), (
                f"unknown renamed participant: {participant}"
            )
            assert (self.trimmed_root / participant).is_dir(), (
                f"unknown trimmed participant: {participant}"
            )
            return (participant,)
        return tuple(f"ID-{index:02d}" for index in range(1, 33))

    def _sessions(self, participant: str | None = None) -> Iterator[ParticipantSession]:
        run_id = str(self.config.artifacts.get("run_id", ""))
        renamed_loader = DataLoader(self.raw_root, run_id=run_id)
        trimmed_loader = DataLoader(self.trimmed_root, run_id=run_id)
        for participant_id in self._participant_ids(participant):
            renamed = {
                session.condition: session
                for session in renamed_loader.load_participant(participant_id)
            }
            trimmed = {
                session.condition: session
                for session in trimmed_loader.load_participant(participant_id)
            }
            # Keep a canonical row for every configured phase even when one
            # source tree is incomplete.  Raising here used to abort the whole
            # batch and hid the missing-source evidence from downstream audit.
            for condition in sorted(_FOLDER_PHASE):
                complete = renamed.get(condition)
                phase_trimmed = trimmed.get(condition)
                session_errors: list[str] = []
                if complete is None:
                    session_errors.append(f"renamed_phase_missing:{condition}")
                if phase_trimmed is None:
                    session_errors.append(f"trimmed_phase_missing:{condition}")
                yield ParticipantSession(
                    participant=participant_id,
                    condition=condition,
                    label=(
                        complete.label
                        if complete is not None
                        else phase_trimmed.label
                        if phase_trimmed is not None
                        else CONDITIONS[condition]
                    ),
                    ecg=complete.ecg if complete is not None else None,
                    ppg_ir=phase_trimmed.ppg_ir if phase_trimmed is not None else None,
                    ppg_red=phase_trimmed.ppg_red
                    if phase_trimmed is not None
                    else None,
                    ppg_green=phase_trimmed.ppg_green
                    if phase_trimmed is not None
                    else None,
                    signal_meta={
                        **(complete.signal_meta if complete is not None else {}),
                        **(
                            phase_trimmed.signal_meta
                            if phase_trimmed is not None
                            else {}
                        ),
                    },
                    errors=[
                        *session_errors,
                        *(complete.errors if complete is not None else []),
                        *(phase_trimmed.errors if phase_trimmed is not None else []),
                    ],
                    lineage_ambiguous_channels=(
                        (complete.lineage_ambiguous_channels if complete else set())
                        | (
                            phase_trimmed.lineage_ambiguous_channels
                            if phase_trimmed
                            else set()
                        )
                    ),
                )

    @staticmethod
    def _frame_for_sensor(
        session: ParticipantSession,
        sensor_channel: str,
    ) -> pd.DataFrame | None:
        return getattr(session, sensor_channel)

    def validate_inputs(self) -> pd.DataFrame:
        """Index immutable raw files and validate locked coverage."""

        assert self.raw_root.is_dir(), f"renamed raw source missing: {self.raw_root}"
        assert self.trimmed_root.is_dir(), (
            f"trimmed raw source missing: {self.trimmed_root}"
        )
        records: list[dict[str, Any]] = []
        selection_records: list[dict[str, Any]] = []
        audit_records: list[dict[str, Any]] = []
        for participant_id in self._participant_ids():
            for folder, phase in _FOLDER_PHASE.items():
                for sensor, _, _ in _SENSOR_ATTRS:
                    source_root = (
                        self.raw_root if sensor == "ecg" else self.trimmed_root
                    )
                    recording_dir = source_root / participant_id / folder
                    patterns = {
                        "ecg": "*_ECG_*.csv",
                        "ppg_ir": "*PPG IR*.csv",
                        "ppg_red": "*PPG Red*.csv",
                        "ppg_green": "*PPG Green*.csv",
                    }
                    matches = (
                        sorted(recording_dir.glob(patterns[sensor]))
                        if recording_dir.is_dir()
                        else []
                    )
                    recording_id = f"{participant_id}_{phase}_raw"
                    selection_status = (
                        "SELECTED"
                        if len(matches) == 1
                        else "LINEAGE_AMBIGUOUS"
                        if len(matches) > 1
                        else "MISSING"
                    )
                    source_files = [
                        str(path.relative_to(self.root)).replace("\\", "/")
                        for path in matches
                    ]
                    source_checksums = [_sha256(path) for path in matches]
                    selection_records.append(
                        {
                            "participant_id": participant_id,
                            "phase": phase,
                            "recording_id": recording_id,
                            "sensor_channel": sensor,
                            "source_files": json.dumps(source_files),
                            "source_checksums": json.dumps(source_checksums),
                            "file_order": json.dumps(source_files),
                            "source_file": source_files[0] if len(matches) == 1 else "",
                            "source_checksum": (
                                source_checksums[0] if len(matches) == 1 else ""
                            ),
                            "overlap": False,
                            "status": selection_status,
                            "exclusion_reason": (
                                None
                                if selection_status == "SELECTED"
                                else selection_status
                            ),
                        }
                    )
                    records.append(
                        {
                            "participant_id": participant_id,
                            "phase": phase,
                            "recording_id": recording_id,
                            "sensor_channel": sensor,
                            "source_role": "renamed_complete_ecg"
                            if sensor == "ecg"
                            else "trimmed_phase_ppg",
                            "file_count": len(matches),
                            "source_file": (
                                str(matches[0].relative_to(self.root)).replace(
                                    "\\", "/"
                                )
                                if len(matches) == 1
                                else ""
                            ),
                            "sha256": _sha256(matches[0]) if len(matches) == 1 else "",
                            "status": "ok"
                            if len(matches) == 1
                            else "missing_or_duplicate",
                            "selection_status": selection_status,
                        }
                    )
                trimmed_ecg = self.trimmed_root / participant_id / folder
                audit_matches = (
                    sorted(trimmed_ecg.glob("*_ECG_*.csv"))
                    if trimmed_ecg.is_dir()
                    else []
                )
                audit_records.append(
                    {
                        "participant_id": participant_id,
                        "phase": phase,
                        "sensor_channel": "ecg",
                        "source_role": "trimmed_ecg_audit_only",
                        "file_count": len(audit_matches),
                        "source_file": str(
                            audit_matches[0].relative_to(self.root)
                        ).replace("\\", "/")
                        if len(audit_matches) == 1
                        else "",
                        "status": "audit_only"
                        if len(audit_matches) == 1
                        else "missing_or_duplicate",
                    }
                )
        index = pd.DataFrame(records)
        assert len(index) == 32 * 2 * 4
        assert "routine" not in set(index["phase"])
        output = self.report_dir / "data_quality" / "raw_input_index_v3.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(output, index=False)
        _write_json(
            output.with_suffix(".json"),
            {
                "rows": len(index),
                "participants": int(index["participant_id"].nunique()),
                "missing_or_duplicate": int((index["status"] != "ok").sum()),
                "ecg_source": str(self.raw_root.relative_to(self.root)).replace(
                    "\\", "/"
                ),
                "ppg_source": str(self.trimmed_root.relative_to(self.root)).replace(
                    "\\", "/"
                ),
                "trimmed_ecg_role": "audit_only",
            },
        )
        pd.DataFrame(audit_records).to_parquet(
            output.with_name("trimmed_ecg_audit_index_v3.parquet"), index=False
        )
        selection_path = output.with_name("recording_selection_v1.parquet")
        selection_frame = pd.DataFrame(selection_records)
        selection_frame.to_parquet(selection_path, index=False)
        _write_json(
            selection_path.with_suffix(".json"),
            {
                "schema_version": "recording-selection-v1",
                "rows": len(selection_frame),
                "selected": int((selection_frame["status"] == "SELECTED").sum()),
                "lineage_ambiguous": int(
                    (selection_frame["status"] == "LINEAGE_AMBIGUOUS").sum()
                ),
                "missing": int((selection_frame["status"] == "MISSING").sum()),
            },
        )
        return index

    def _ingest_legacy_compat(self, participant: str | None = None) -> pd.DataFrame:
        """Compatibility implementation retained outside the canonical workflow."""

        selection_path = (
            self.report_dir / "data_quality" / "recording_selection_v1.parquet"
        )
        if not selection_path.is_file():
            self.validate_inputs()
        selections = pd.read_parquet(selection_path)
        selection_lookup = {
            (
                str(row.participant_id),
                str(row.phase),
                str(row.sensor_channel),
            ): str(row.status)
            for row in selections.itertuples(index=False)
        }
        manifest_rows: list[dict[str, Any]] = []
        issue_rows: list[dict[str, Any]] = []
        for source_root in (self.raw_root, self.trimmed_root):
            for participant_id in self._participant_ids(participant):
                participant_root = source_root / participant_id
                if not participant_root.is_dir():
                    continue
                for phase_dir in sorted(participant_root.iterdir()):
                    if not phase_dir.is_dir() or phase_dir.name in _FOLDER_PHASE:
                        continue
                    files = sorted(phase_dir.glob("*.csv"))
                    if not files:
                        files = [phase_dir]
                    for source in files:
                        source_file = str(source.relative_to(self.root)).replace(
                            "\\", "/"
                        )
                        issue_rows.append(
                            {
                                "participant_id": participant_id,
                                "session_id": participant_id,
                                "recording_id": (
                                    f"{participant_id}_{phase_dir.name}_unknown"
                                ),
                                "phase": phase_dir.name,
                                "sensor_channel": self._audit_sensor_from_name(
                                    source.name
                                ),
                                "source_file": source_file,
                                "source_checksum": (
                                    _sha256(source) if source.is_file() else ""
                                ),
                                "row_index": None,
                                "raw_sample_index": None,
                                "issue_type": "unknown_phase",
                                "column": "",
                                "issues": (
                                    f"Unknown phase directory: {phase_dir.name}"
                                ),
                            }
                        )
        for session in self._sessions(participant):
            participant_id = session.participant
            folder_phase = _FOLDER_PHASE[session.condition]
            recording_id = f"{participant_id}_{folder_phase}_raw"
            for sensor_channel, attribute, default_fs in _SENSOR_ATTRS:
                raw = self._frame_for_sensor(session, attribute)
                selection_status = selection_lookup.get(
                    (participant_id, folder_phase, sensor_channel), "MISSING"
                )
                if selection_status != "SELECTED":
                    manifest_rows.append(
                        {
                            "participant_id": participant_id,
                            "recording_id": recording_id,
                            "source_phase_hint": folder_phase,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "sensor_channel": sensor_channel,
                            "status": selection_status,
                            "row_count": 0,
                            "exclusion_reason": selection_status,
                        }
                    )
                    continue
                if sensor_channel in session.lineage_ambiguous_channels:
                    manifest_rows.append(
                        {
                            "participant_id": participant_id,
                            "recording_id": recording_id,
                            "source_phase_hint": folder_phase,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "sensor_channel": sensor_channel,
                            "status": "LINEAGE_AMBIGUOUS",
                            "row_count": 0,
                            "exclusion_reason": "LINEAGE_AMBIGUOUS",
                        }
                    )
                    continue
                if raw is None or raw.empty:
                    manifest_rows.append(
                        {
                            "participant_id": participant_id,
                            "recording_id": recording_id,
                            "source_phase_hint": folder_phase,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "sensor_channel": sensor_channel,
                            "status": "missing",
                            "exclusion_reason": "raw_file_missing",
                        }
                    )
                    continue
                value_column = "ecg_mv" if sensor_channel == "ecg" else "ppg_raw"
                source = Path(str(raw.attrs.get("source_file", ""))).resolve()
                source_relative = str(source.relative_to(self.root)).replace("\\", "/")
                source_checksum = _sha256(source)
                raw_timestamps = pd.to_numeric(raw["unix_ms"], errors="coerce")
                values = pd.to_numeric(raw[value_column], errors="coerce")
                for parser_issue in raw.attrs.get("ingestion_issues", []):
                    issue_rows.append(
                        {
                            "participant_id": participant_id,
                            "session_id": participant_id,
                            "recording_id": recording_id,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "phase_canonical": folder_phase,
                            "sensor_channel": sensor_channel,
                            "source_file": source_relative,
                            "source_checksum": source_checksum,
                            **parser_issue,
                        }
                    )
                raw_timestamp_array = raw_timestamps.to_numpy(dtype=float)
                duplicate_mask = (
                    raw_timestamps.duplicated(keep=False) & raw_timestamps.notna()
                )
                for raw_index in np.flatnonzero(duplicate_mask.to_numpy()):
                    issue_rows.append(
                        {
                            "participant_id": participant_id,
                            "session_id": participant_id,
                            "recording_id": recording_id,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "phase_canonical": folder_phase,
                            "sensor_channel": sensor_channel,
                            "source_file": source_relative,
                            "source_checksum": source_checksum,
                            "row_index": int(raw_index + 1),
                            "raw_sample_index": int(raw_index),
                            "issue_type": "duplicate_timestamp",
                            "column": "unix_ms",
                            "issues": "Duplicate timestamp in source order",
                        }
                    )
                finite_timestamp_for_edges = np.isfinite(raw_timestamp_array)
                for raw_index in np.flatnonzero(
                    finite_timestamp_for_edges[:-1]
                    & finite_timestamp_for_edges[1:]
                    & (np.diff(raw_timestamp_array) <= 0.0)
                ):
                    issue_rows.append(
                        {
                            "participant_id": participant_id,
                            "session_id": participant_id,
                            "recording_id": recording_id,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "phase_canonical": folder_phase,
                            "sensor_channel": sensor_channel,
                            "source_file": source_relative,
                            "source_checksum": source_checksum,
                            "row_index": int(raw_index + 2),
                            "raw_sample_index": int(raw_index + 1),
                            "issue_type": "non_monotonic_timestamp",
                            "column": "unix_ms",
                            "issues": "Timestamp is not increasing in source order",
                        }
                    )
                raw_sample_index = pd.to_numeric(
                    raw.get(
                        "raw_sample_index",
                        pd.Series(np.arange(len(raw)), index=raw.index),
                    ),
                    errors="coerce",
                )
                timestamp_valid = pd.Series(
                    np.isfinite(raw_timestamps.to_numpy(dtype=float)),
                    index=raw_timestamps.index,
                )
                signal_valid = np.isfinite(values.to_numpy(dtype=float))
                if not timestamp_valid.any() or not signal_valid.any():
                    manifest_rows.append(
                        {
                            "participant_id": participant_id,
                            "recording_id": recording_id,
                            "source_phase_hint": folder_phase,
                            "phase": folder_phase,
                            "phase_source": session.condition,
                            "sensor_channel": sensor_channel,
                            "status": "invalid",
                            "row_count": int(len(raw)),
                            "valid_timestamp_count": int(timestamp_valid.sum()),
                            "finite_signal_count": int(signal_valid.sum()),
                            "exclusion_reason": (
                                "no_valid_timestamp"
                                if not timestamp_valid.any()
                                else "no_finite_signal_value"
                            ),
                        }
                    )
                    continue

                # Keep source order. Metadata clock correction depends on the
                # original sample sequence; sorting is a separate diagnostic
                # view, never an ingestion mutation.
                timestamp_mask = timestamp_valid.to_numpy(dtype=bool)
                raw_timestamps = raw_timestamps[timestamp_mask].reset_index(drop=True)
                values = values[timestamp_mask].reset_index(drop=True)
                raw_sample_index = raw_sample_index[timestamp_mask].reset_index(
                    drop=True
                )
                acc_x = pd.to_numeric(
                    raw["acc_x"]
                    if "acc_x" in raw
                    else pd.Series(np.nan, index=raw.index),
                    errors="coerce",
                )[timestamp_mask].reset_index(drop=True)
                acc_y = pd.to_numeric(
                    raw["acc_y"]
                    if "acc_y" in raw
                    else pd.Series(np.nan, index=raw.index),
                    errors="coerce",
                )[timestamp_mask].reset_index(drop=True)
                acc_z = pd.to_numeric(
                    raw["acc_z"]
                    if "acc_z" in raw
                    else pd.Series(np.nan, index=raw.index),
                    errors="coerce",
                )[timestamp_mask].reset_index(drop=True)
                metadata = dict(raw.attrs.get("metadata", {}))
                declared_fs = get_sampling_rate_hz(raw, default_fs)
                actual_fs = _actual_sampling_rate(
                    raw_timestamps.to_numpy(), declared_fs
                )
                timestamp_utc = pd.to_datetime(raw_timestamps, unit="ms", utc=True)
                frame = pd.DataFrame(
                    {
                        "participant_id": participant_id,
                        "session_id": participant_id,
                        "recording_id": recording_id,
                        "phase": "unknown",
                        "raw_sample_index": raw_sample_index.to_numpy(dtype=np.int64),
                        "sensor_channel": sensor_channel,
                        "timestamp_raw_utc": timestamp_utc,
                        "timestamp_corrected_utc": timestamp_utc,
                        "timestamp_local": timestamp_utc.dt.tz_convert(
                            self.config.timezone_local
                        ),
                        "elapsed_seconds": (
                            raw_timestamps.to_numpy(dtype=np.float64)
                            - float(raw_timestamps.iloc[0])
                        )
                        / 1000.0,
                        "sampling_rate_hz": actual_fs,
                        "raw_value": values.to_numpy(dtype=np.float64),
                        "clean_value": values.to_numpy(dtype=np.float64),
                        "acc_x": acc_x.to_numpy(dtype=np.float64),
                        "acc_y": acc_y.to_numpy(dtype=np.float64),
                        "acc_z": acc_z.to_numpy(dtype=np.float64),
                        "quality_score": np.nan,
                        "quality_tier": "not_assessed",
                        "quality_flags": "not_assessed",
                        "source_file": source_relative,
                        "source_checksum": source_checksum,
                        "exclusion_reason": "",
                        "source_role": (
                            "renamed_complete_ecg"
                            if sensor_channel == "ecg"
                            else "trimmed_phase_ppg"
                        ),
                        "preprocessing_version": self.config.preprocessing_version,
                        "sync_alpha": 1.0,
                        "sync_beta_seconds": 0.0,
                        "sync_method": "not_applied",
                        "sync_confidence": 0.0,
                        "ecg_excluded": False,
                    }
                )
                # A raw frame may have non-monotonic source timestamps. Do
                # not reorder it merely to satisfy the cleaned-frame contract;
                # sync-raw creates the monotonic corrected ECG view later.
                if not timestamp_utc.is_monotonic_increasing:
                    frame["quality_flags"] = "raw_timestamp_non_monotonic"
                destination = self.bronze_dir / participant_id / recording_id
                destination.mkdir(parents=True, exist_ok=True)
                output = destination / f"{sensor_channel}.parquet"
                frame.to_parquet(output, index=False)
                _write_csv_mirror(frame, output)
                declared_samples = _metadata_float(metadata, "samplescount")
                declared_duration = _metadata_float(metadata, "duration")
                duration = float(
                    (raw_timestamps.iloc[-1] - raw_timestamps.iloc[0]) / 1000.0
                )
                monotonic = bool(np.all(np.diff(raw_timestamps.to_numpy()) > 0))
                manifest_rows.append(
                    {
                        "participant_id": participant_id,
                        "recording_id": recording_id,
                        "source_phase_hint": folder_phase,
                        "phase": folder_phase,
                        "phase_source": session.condition,
                        "sensor_channel": sensor_channel,
                        "source_file": source_relative,
                        "source_sha256": source_checksum,
                        "source_checksum": source_checksum,
                        "row_count": len(frame),
                        "raw_row_count": int(len(raw)),
                        "valid_timestamp_count": int(timestamp_valid.sum()),
                        "invalid_timestamp_count": int((~timestamp_valid).sum()),
                        "finite_signal_count": int(signal_valid[timestamp_mask].sum()),
                        "nonfinite_signal_count": int(
                            (~signal_valid[timestamp_mask]).sum()
                        ),
                        "declared_sample_count": declared_samples,
                        "declared_duration_seconds": declared_duration,
                        "observed_duration_seconds": duration,
                        "declared_sampling_rate_hz": declared_fs,
                        "actual_sampling_rate_hz": actual_fs,
                        "timestamp_monotonic": monotonic,
                        "nan_ratio": float(
                            1.0
                            - (
                                timestamp_valid.to_numpy(dtype=bool) & signal_valid
                            ).mean()
                        ),
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                        "bronze_path": str(output.relative_to(self.root)).replace(
                            "\\", "/"
                        ),
                        "status": "ok" if monotonic else "invalid",
                        "exclusion_reason": ""
                        if monotonic
                        else "timestamp_non_monotonic",
                    }
                )
        manifest = pd.DataFrame(manifest_rows)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.bronze_dir / "ingestion_manifest_v3.parquet"
        manifest.to_parquet(manifest_path, index=False)
        _write_csv_mirror(manifest, manifest_path)
        issue_frame = pd.DataFrame(
            issue_rows,
            columns=[
                "run_id",
                "participant_id",
                "session_id",
                "phase_source",
                "recording_id",
                "phase",
                "phase_canonical",
                "sensor_channel",
                "source_file",
                "source_checksum",
                "row_index",
                "raw_sample_index",
                "issue_type",
                "issue_code",
                "severity",
                "column",
                "issues",
            ],
        )
        issue_frame["run_id"] = str(self.config.artifacts.get("run_id", ""))
        issue_frame["phase_source"] = issue_frame["phase_source"].where(
            issue_frame["phase_source"].notna()
            & issue_frame["phase_source"].astype(str).ne(""),
            issue_frame["phase"],
        )
        phase_fallback = (
            issue_frame["phase"]
            .map(
                {"Relaks": "relax", "Stress": "task", "relax": "relax", "task": "task"}
            )
            .fillna("")
        )
        issue_frame["phase_canonical"] = issue_frame["phase_canonical"].where(
            issue_frame["phase_canonical"].notna()
            & issue_frame["phase_canonical"].astype(str).ne(""),
            phase_fallback,
        )
        issue_frame["issue_code"] = issue_frame["issue_code"].where(
            issue_frame["issue_code"].notna()
            & issue_frame["issue_code"].astype(str).ne(""),
            issue_frame["issue_type"],
        )
        issue_frame["severity"] = issue_frame["issue_type"].map(
            lambda value: (
                "error"
                if value
                in {
                    "parse_error",
                    "malformed_row",
                    "invalid_value",
                    "invalid_timestamp",
                    "lineage_ambiguous",
                    "unknown_phase",
                    "unknown_participant",
                    "duplicate_timestamp",
                    "non_monotonic_timestamp",
                    "load_error",
                }
                else "warning"
            )
        )
        issue_path = self.bronze_dir / "ingestion_issues_v3.parquet"
        issue_frame.to_parquet(issue_path, index=False)
        _write_csv_mirror(issue_frame, issue_path)
        return manifest

    def _ingest_loaded_signal(
        self,
        *,
        raw: pd.DataFrame,
        participant_id: str,
        recording_id: str,
        phase: str,
        phase_source: str,
        sensor_channel: str,
        source_relative: str,
        source_checksum: str,
        default_fs: float,
        manifest_rows: list[dict[str, Any]],
        issue_rows: list[dict[str, Any]],
    ) -> None:
        """Persist one loaded source file and its audit rows, then release it."""

        value_column = "ecg_mv" if sensor_channel == "ecg" else "ppg_raw"
        raw_timestamps = pd.to_numeric(raw["unix_ms"], errors="coerce")
        values = pd.to_numeric(raw[value_column], errors="coerce")
        issue_context = {
            "participant_id": participant_id,
            "session_id": participant_id,
            "recording_id": recording_id,
            "phase": phase,
            "phase_source": phase_source,
            "phase_canonical": phase,
            "sensor_channel": sensor_channel,
            "source_file": source_relative,
            "source_checksum": source_checksum,
        }
        for parser_issue in raw.attrs.get("ingestion_issues", []):
            issue_rows.append({**issue_context, **parser_issue})

        raw_timestamp_array = raw_timestamps.to_numpy(dtype=float)
        finite_timestamp_for_edges = np.isfinite(raw_timestamp_array)
        non_monotonic_indices = np.flatnonzero(
            finite_timestamp_for_edges[:-1]
            & finite_timestamp_for_edges[1:]
            & (np.diff(raw_timestamp_array) <= 0.0)
        )
        for raw_index in non_monotonic_indices:
            issue_rows.append(
                {
                    **issue_context,
                    "row_index": int(raw_index + 2),
                    "raw_sample_index": int(raw_index + 1),
                    "issue_type": "non_monotonic_timestamp",
                    "column": "unix_ms",
                    "issues": "Timestamp is not increasing in source order",
                }
            )

        raw_sample_index = pd.to_numeric(
            raw.get(
                "raw_sample_index",
                pd.Series(np.arange(len(raw)), index=raw.index),
            ),
            errors="coerce",
        )
        timestamp_valid = pd.Series(
            np.isfinite(raw_timestamps.to_numpy(dtype=float)),
            index=raw_timestamps.index,
        )
        signal_valid = np.isfinite(values.to_numpy(dtype=float))
        base_manifest = {
            "participant_id": participant_id,
            "recording_id": recording_id,
            "source_phase_hint": phase_source,
            "phase": phase,
            "phase_source": phase_source,
            "sensor_channel": sensor_channel,
            "source_file": source_relative,
            "source_sha256": source_checksum,
            "source_checksum": source_checksum,
            "raw_row_count": int(len(raw)),
        }
        if not timestamp_valid.any() or not signal_valid.any():
            manifest_rows.append(
                {
                    **base_manifest,
                    "status": "invalid",
                    "row_count": int(len(raw)),
                    "valid_timestamp_count": int(timestamp_valid.sum()),
                    "finite_signal_count": int(signal_valid.sum()),
                    "exclusion_reason": (
                        "no_valid_timestamp"
                        if not timestamp_valid.any()
                        else "no_finite_signal_value"
                    ),
                }
            )
            return

        timestamp_mask = timestamp_valid.to_numpy(dtype=bool)
        raw_timestamps = raw_timestamps[timestamp_mask].reset_index(drop=True)
        values = values[timestamp_mask].reset_index(drop=True)
        raw_sample_index = raw_sample_index[timestamp_mask].reset_index(drop=True)
        if not np.isfinite(raw_sample_index.to_numpy(dtype=float)).all():
            issue_rows.append(
                {
                    **issue_context,
                    "row_index": None,
                    "raw_sample_index": None,
                    "issue_type": "invalid_value",
                    "column": "raw_sample_index",
                    "issues": "raw_sample_index is not finite",
                }
            )
            manifest_rows.append(
                {
                    **base_manifest,
                    "status": "invalid",
                    "row_count": int(len(raw)),
                    "valid_timestamp_count": int(timestamp_valid.sum()),
                    "finite_signal_count": int(signal_valid[timestamp_mask].sum()),
                    "exclusion_reason": "invalid_raw_sample_index",
                }
            )
            return

        def _optional_numeric_column(column: str) -> np.ndarray:
            if column not in raw:
                return np.full(len(raw), np.nan, dtype=np.float64)[timestamp_mask]
            return pd.to_numeric(raw[column], errors="coerce")[timestamp_mask].to_numpy(
                dtype=np.float64
            )

        acc_x = _optional_numeric_column("acc_x")
        acc_y = _optional_numeric_column("acc_y")
        acc_z = _optional_numeric_column("acc_z")
        ppg_aux_1 = _optional_numeric_column("ppg_aux_1")
        ppg_aux_2 = _optional_numeric_column("ppg_aux_2")
        ppg_aux_3 = _optional_numeric_column("ppg_aux_3")
        accelerometer_provenance_status = str(
            raw.attrs.get(
                "accelerometer_provenance_status",
                "NOT_APPLICABLE" if sensor_channel == "ecg" else "UNAVAILABLE",
            )
        )
        metadata = dict(raw.attrs.get("metadata", {}))
        declared_fs = get_sampling_rate_hz(raw, default_fs)
        actual_fs = _actual_sampling_rate(raw_timestamps.to_numpy(), declared_fs)
        timestamp_utc = pd.to_datetime(raw_timestamps, unit="ms", utc=True)
        frame = pd.DataFrame(
            {
                "participant_id": participant_id,
                "session_id": participant_id,
                "recording_id": recording_id,
                "phase": phase,
                "phase_source": phase_source,
                "raw_sample_index": raw_sample_index.to_numpy(dtype=np.int64),
                "sensor_channel": sensor_channel,
                "timestamp_raw_utc": timestamp_utc,
                "timestamp_corrected_utc": timestamp_utc,
                "timestamp_local": timestamp_utc.dt.tz_convert(
                    self.config.timezone_local
                ),
                "elapsed_seconds": (
                    raw_timestamps.to_numpy(dtype=np.float64)
                    - float(raw_timestamps.iloc[0])
                )
                / 1000.0,
                "sampling_rate_hz": actual_fs,
                "raw_value": values.to_numpy(dtype=np.float64),
                "clean_value": values.to_numpy(dtype=np.float64),
                "acc_x": acc_x,
                "acc_y": acc_y,
                "acc_z": acc_z,
                "ppg_aux_1": ppg_aux_1,
                "ppg_aux_2": ppg_aux_2,
                "ppg_aux_3": ppg_aux_3,
                "accelerometer_provenance_status": (accelerometer_provenance_status),
                "quality_score": np.nan,
                "quality_tier": "not_assessed",
                "quality_flags": "not_assessed",
                "source_file": source_relative,
                "source_checksum": source_checksum,
                "exclusion_reason": "",
                "source_role": (
                    "renamed_complete_ecg"
                    if sensor_channel == "ecg"
                    else "trimmed_phase_ppg"
                ),
                "preprocessing_version": self.config.preprocessing_version,
                "sync_alpha": 1.0,
                "sync_beta_seconds": 0.0,
                "sync_method": "not_applied",
                "sync_confidence": 0.0,
                "ecg_excluded": False,
            }
        )
        if not timestamp_utc.is_monotonic_increasing:
            frame["quality_flags"] = "raw_timestamp_non_monotonic"

        destination = self.bronze_dir / participant_id / recording_id
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f"{sensor_channel}.parquet"
        frame.to_parquet(output, index=False)
        _write_csv_mirror(frame, output)
        declared_samples = _metadata_float(metadata, "samplescount")
        declared_duration = _metadata_float(metadata, "duration")
        duration = float((raw_timestamps.iloc[-1] - raw_timestamps.iloc[0]) / 1000.0)
        monotonic = bool(np.all(np.diff(raw_timestamps.to_numpy()) > 0))
        manifest_rows.append(
            {
                **base_manifest,
                "row_count": len(frame),
                "valid_timestamp_count": int(timestamp_valid.sum()),
                "invalid_timestamp_count": int((~timestamp_valid).sum()),
                "finite_signal_count": int(signal_valid[timestamp_mask].sum()),
                "nonfinite_signal_count": int((~signal_valid[timestamp_mask]).sum()),
                "declared_sample_count": declared_samples,
                "declared_duration_seconds": declared_duration,
                "observed_duration_seconds": duration,
                "declared_sampling_rate_hz": declared_fs,
                "actual_sampling_rate_hz": actual_fs,
                "timestamp_monotonic": monotonic,
                "nan_ratio": float(
                    1.0 - (timestamp_valid.to_numpy(dtype=bool) & signal_valid).mean()
                ),
                "metadata_json": json.dumps(metadata, sort_keys=True),
                "bronze_path": str(output.relative_to(self.root)).replace("\\", "/"),
                "status": "ok" if monotonic else "invalid",
                "exclusion_reason": "" if monotonic else "timestamp_non_monotonic",
            }
        )

    def ingest(self, participant: str | None = None) -> pd.DataFrame:
        """Ingest one selected source file at a time with complete provenance."""

        selection_path = (
            self.report_dir / "data_quality" / "recording_selection_v1.parquet"
        )
        if not selection_path.is_file():
            self.validate_inputs()
        selections = pd.read_parquet(selection_path)
        if participant is not None:
            selections = selections[selections["participant_id"] == participant].copy()

        manifest_rows: list[dict[str, Any]] = []
        issue_rows: list[dict[str, Any]] = []
        allowed_participants = set(self._participant_ids())

        for source_root in (self.raw_root, self.trimmed_root):
            if not source_root.is_dir():
                continue
            for participant_root in sorted(source_root.iterdir()):
                if (
                    not participant_root.is_dir()
                    or participant_root.name in allowed_participants
                ):
                    continue
                files = sorted(
                    path for path in participant_root.rglob("*") if path.is_file()
                )
                if not files:
                    files = [participant_root]
                for source in files:
                    issue_rows.append(
                        {
                            "participant_id": participant_root.name,
                            "session_id": participant_root.name,
                            "recording_id": f"{participant_root.name}_unknown_raw",
                            "phase": source.parent.name
                            if source.is_file()
                            else "unknown",
                            "phase_source": source.parent.name
                            if source.is_file()
                            else "unknown",
                            "phase_canonical": "",
                            "sensor_channel": self._audit_sensor_from_name(source.name),
                            "source_file": str(source.resolve()),
                            "source_checksum": _sha256(source)
                            if source.is_file()
                            else "",
                            "row_index": None,
                            "raw_sample_index": None,
                            "issue_type": "unknown_participant",
                            "issue_code": "unknown_participant",
                            "column": "",
                            "issues": (
                                "Participant directory is outside ID-01..ID-32: "
                                f"{participant_root.name}"
                            ),
                        }
                    )

        for source_root in (self.raw_root, self.trimmed_root):
            for participant_id in selections["participant_id"].astype(str).unique():
                participant_root = source_root / participant_id
                if not participant_root.is_dir():
                    continue
                for phase_dir in sorted(participant_root.iterdir()):
                    if phase_dir.name in _FOLDER_PHASE or not phase_dir.is_dir():
                        continue
                    files = sorted(phase_dir.glob("*.csv"))
                    if not files:
                        files = [phase_dir]
                    for source in files:
                        issue_rows.append(
                            {
                                "participant_id": participant_id,
                                "session_id": participant_id,
                                "recording_id": f"{participant_id}_{phase_dir.name}_unknown",
                                "phase": phase_dir.name,
                                "phase_source": phase_dir.name,
                                "phase_canonical": "",
                                "sensor_channel": self._audit_sensor_from_name(
                                    source.name
                                ),
                                "source_file": str(source.resolve()),
                                "source_checksum": _sha256(source)
                                if source.is_file()
                                else "",
                                "row_index": None,
                                "raw_sample_index": None,
                                "issue_type": "unknown_phase",
                                "issue_code": "unknown_phase",
                                "column": "",
                                "issues": f"Unknown phase directory: {phase_dir.name}",
                            }
                        )

        default_rates = dict(
            (sensor, default_fs) for sensor, _, default_fs in _SENSOR_ATTRS
        )
        for row in selections.sort_values(
            ["participant_id", "phase", "sensor_channel"]
        ).itertuples(index=False):
            participant_id = str(row.participant_id)
            phase = str(row.phase)
            phase_source = str(
                getattr(row, "phase_source", "")
                or {"relax": "Relaks", "task": "Stress"}.get(phase, phase)
            )
            sensor_channel = str(row.sensor_channel)
            recording_id = str(row.recording_id)
            selection_status = str(row.status)
            source_relative = str(getattr(row, "source_file", "") or "")
            source_checksum = str(
                getattr(row, "sha256", "") or getattr(row, "source_checksum", "") or ""
            )
            base_manifest = {
                "participant_id": participant_id,
                "recording_id": recording_id,
                "source_phase_hint": phase_source,
                "phase": phase,
                "phase_source": phase_source,
                "sensor_channel": sensor_channel,
                "source_file": source_relative,
                "source_sha256": source_checksum,
                "source_checksum": source_checksum,
            }
            if selection_status != "SELECTED":
                manifest_rows.append(
                    {
                        **base_manifest,
                        "status": selection_status,
                        "row_count": 0,
                        "exclusion_reason": selection_status,
                    }
                )
                continue
            source = (self.root / source_relative).resolve()
            try:
                raw = load_signal_file(
                    source,
                    sensor_channel,
                    include_provenance=True,
                )
            except (OSError, ValueError, KeyError) as exc:
                reason = f"load_error:{type(exc).__name__}:{exc}"
                issue_rows.append(
                    {
                        "participant_id": participant_id,
                        "session_id": participant_id,
                        "recording_id": recording_id,
                        "phase": phase,
                        "phase_source": phase_source,
                        "phase_canonical": phase,
                        "sensor_channel": sensor_channel,
                        "source_file": source_relative,
                        "source_checksum": source_checksum,
                        "row_index": None,
                        "raw_sample_index": None,
                        "issue_type": "load_error",
                        "issue_code": "load_error",
                        "column": "",
                        "issues": reason,
                    }
                )
                manifest_rows.append(
                    {
                        **base_manifest,
                        "status": "INVALID",
                        "row_count": 0,
                        "exclusion_reason": reason,
                    }
                )
                continue
            raw.attrs["source_file"] = str(source)
            raw.attrs["source_checksum"] = source_checksum
            self._ingest_loaded_signal(
                raw=raw,
                participant_id=participant_id,
                recording_id=recording_id,
                phase=phase,
                phase_source=phase_source,
                sensor_channel=sensor_channel,
                source_relative=source_relative,
                source_checksum=source_checksum,
                default_fs=default_rates[sensor_channel],
                manifest_rows=manifest_rows,
                issue_rows=issue_rows,
            )

        manifest = pd.DataFrame(manifest_rows)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.bronze_dir / "ingestion_manifest_v3.parquet"
        manifest.to_parquet(manifest_path, index=False)
        _write_csv_mirror(manifest, manifest_path)
        issue_frame = pd.DataFrame(
            issue_rows,
            columns=[
                "run_id",
                "participant_id",
                "session_id",
                "phase_source",
                "recording_id",
                "phase",
                "phase_canonical",
                "sensor_channel",
                "source_file",
                "source_checksum",
                "row_index",
                "raw_sample_index",
                "issue_type",
                "issue_code",
                "severity",
                "column",
                "issues",
            ],
        )
        issue_frame["run_id"] = str(self.config.artifacts.get("run_id", ""))
        issue_frame["phase_source"] = issue_frame["phase_source"].where(
            issue_frame["phase_source"].notna()
            & issue_frame["phase_source"].astype(str).ne(""),
            issue_frame["phase"],
        )
        phase_fallback = (
            issue_frame["phase"]
            .map(
                {"Relaks": "relax", "Stress": "task", "relax": "relax", "task": "task"}
            )
            .fillna("")
        )
        issue_frame["phase_canonical"] = issue_frame["phase_canonical"].where(
            issue_frame["phase_canonical"].notna()
            & issue_frame["phase_canonical"].astype(str).ne(""),
            phase_fallback,
        )
        issue_frame["issue_code"] = issue_frame["issue_code"].where(
            issue_frame["issue_code"].notna()
            & issue_frame["issue_code"].astype(str).ne(""),
            issue_frame["issue_type"],
        )
        error_types = {
            "parse_error",
            "malformed_row",
            "invalid_value",
            "invalid_timestamp",
            "lineage_ambiguous",
            "unknown_phase",
            "unknown_participant",
            "duplicate_timestamp",
            "non_monotonic_timestamp",
            "load_error",
        }
        issue_frame["severity"] = issue_frame["issue_type"].map(
            lambda value: "error" if value in error_types else "warning"
        )
        issue_path = self.bronze_dir / "ingestion_issues_v3.parquet"
        issue_frame.to_parquet(issue_path, index=False)
        _write_csv_mirror(issue_frame, issue_path)
        return manifest

    @staticmethod
    def _audit_sensor_from_name(filename: str) -> str:
        lowered = filename.lower()
        if "_ecg_" in lowered:
            return "ecg"
        if "ppg green" in lowered:
            return "ppg_green"
        if "ppg ir" in lowered:
            return "ppg_ir"
        if "ppg red" in lowered:
            return "ppg_red"
        return "unknown"

    def _load_ingestion_manifest(self) -> pd.DataFrame:
        path = self.bronze_dir / "ingestion_manifest_v3.parquet"
        assert path.exists(), "run ingest before sync-raw"
        return pd.read_parquet(path)

    def _read_validation_summary(self) -> dict[str, Any] | None:
        validation_dir = self._configured_path("validation_dir", "data/validation")
        for directory in ("ppg_ecg_240s", "ppg_ecg"):
            for filename in (
                "validation_summary_240s.json",
                "validation_summary_5min.json",
                "validation_summary.json",
            ):
                path = validation_dir / directory / filename
                if path.is_file():
                    return json.loads(path.read_text(encoding="utf-8"))
        return None

    def sync_raw(self, participant: str | None = None) -> pd.DataFrame:
        """Correct full-record ECG clocks, then validate against unchanged PPG."""

        manifest = self._load_ingestion_manifest()
        anchors = self._anchors()
        if participant:
            manifest = manifest[manifest["participant_id"] == participant]
            anchors = anchors[anchors["participant_id"] == participant]
        results: list[tuple[SynchronizationResult, str]] = []
        for (participant_id, recording_id), group in manifest.groupby(
            ["participant_id", "recording_id"], sort=True
        ):
            # ``phase`` is the canonical key used by the anchor table.  Keep
            # ``source_phase_hint`` only as lineage; aliases such as Relaks or
            # Stress must not break anchor lookup.
            phase = str(group["phase"].iloc[0]).strip().lower()
            anchor = anchors[
                (anchors["participant_id"] == participant_id)
                & (anchors["phase"] == phase)
            ]
            sensor_paths = {
                str(row.sensor_channel): self.root / str(row.bronze_path)
                for row in group.itertuples()
                if getattr(row, "status", "") == "ok"
            }
            frames = {
                sensor: pd.read_parquet(path) for sensor, path in sensor_paths.items()
            }
            ecg = frames.get("ecg")
            reference_sensor = next(
                (
                    sensor
                    for sensor in ("ppg_ir", "ppg_green", "ppg_red")
                    if sensor in frames
                ),
                None,
            )
            correction: ClockCorrection | None = None
            if ecg is None or reference_sensor is None:
                result = SynchronizationResult(
                    participant_id=str(participant_id),
                    recording_id=str(recording_id),
                    alpha=1.0,
                    beta_seconds=0.0,
                    method="unavailable",
                    confidence=0.0,
                    overlap_seconds=0.0,
                    ecg_peak_count=0,
                    ppg_peak_count=0,
                    matched_peak_count=0,
                    median_pat_ms=None,
                    excluded=True,
                    exclusion_reason="missing_ecg_or_ppg_reference",
                    clock_excluded=True,
                    clock_exclusion_reason="missing_ecg_or_ppg_reference",
                )
            else:
                reference = frames[reference_sensor]
                ecg_row = group[group["sensor_channel"] == "ecg"].iloc[0]
                try:
                    metadata = json.loads(str(ecg_row["metadata_json"]))
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata_json_not_object")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    correction, result = _sync_exclusion(
                        participant_id=str(participant_id),
                        recording_id=str(recording_id),
                        reason=f"invalid_ecg_metadata:{type(exc).__name__}:{exc}",
                        ecg_sample_count=len(ecg),
                        reference_channel=reference_sensor,
                    )
                else:
                    if len(anchor) != 1:
                        correction, result = _sync_exclusion(
                            participant_id=str(participant_id),
                            recording_id=str(recording_id),
                            reason=f"anchor_cardinality:{len(anchor)}",
                            ecg_sample_count=len(ecg),
                            reference_channel=reference_sensor,
                        )
                    else:
                        anchor_start_ms = float(
                            pd.Timestamp(anchor.iloc[0]["start_utc"]).timestamp()
                            * 1000.0
                        )
                        phase_start_ms = anchor_start_ms + (
                            self.config.window.edge_trim_start_seconds * 1000.0
                        )
                        phase_end_ms = phase_start_ms + (
                            self.config.window.validation_window_seconds * 1000.0
                        )
                        correction, result = validate_ecg_against_ppg(
                            participant_id=str(participant_id),
                            recording_id=str(recording_id),
                            ecg_timestamps_ms=pd.to_datetime(
                                ecg["timestamp_raw_utc"], utc=True
                            )
                            .astype("int64")
                            .to_numpy(dtype=np.float64)
                            / 1_000_000.0,
                            ecg_values=ecg["raw_value"].to_numpy(dtype=np.float64),
                            ppg_timestamps_ms=pd.to_datetime(
                                reference["timestamp_raw_utc"], utc=True
                            )
                            .astype("int64")
                            .to_numpy(dtype=np.float64)
                            / 1_000_000.0,
                            ppg_values=reference["raw_value"].to_numpy(
                                dtype=np.float64
                            ),
                            metadata=metadata,
                            config=self.config.synchronization,
                            phase_start_ms=phase_start_ms,
                            phase_end_ms=phase_end_ms,
                            diagnostic_reference_channel=reference_sensor,
                        )
            results.append((result, phase))
            for sensor, frame in frames.items():
                synced = frame.copy()
                if sensor == "ecg" and correction is not None:
                    try:
                        corrected = _epoch_milliseconds_to_utc(
                            correction.corrected_unix_ms
                        )
                    except ValueError:
                        if not (result.clock_excluded or result.excluded):
                            raise
                    else:
                        synced["timestamp_corrected_utc"] = corrected
                        synced["timestamp_local"] = corrected.tz_convert(
                            self.config.timezone_local
                        )
                        synced["elapsed_seconds"] = (
                            np.asarray(correction.corrected_unix_ms, dtype=np.float64)
                            - float(correction.corrected_unix_ms[0])
                        ) / 1000.0
                synced["sync_alpha"] = result.alpha if sensor == "ecg" else 1.0
                synced["sync_beta_seconds"] = (
                    result.beta_seconds if sensor == "ecg" else 0.0
                )
                synced["sync_method"] = (
                    result.method if sensor == "ecg" else "ppg_reference_unchanged"
                )
                synced["sync_confidence"] = (
                    result.confidence if sensor == "ecg" else 1.0
                )
                clock_excluded = (
                    result.clock_excluded
                    if result.clock_excluded is not None
                    else result.excluded
                )
                synced["ecg_excluded"] = clock_excluded if sensor == "ecg" else False
                synced["exclusion_reason"] = (
                    (result.clock_exclusion_reason or result.exclusion_reason or "")
                    if sensor == "ecg"
                    else ""
                )
                if sensor != "ecg":
                    pd.testing.assert_series_equal(
                        synced["timestamp_raw_utc"],
                        frame["timestamp_raw_utc"],
                        check_names=False,
                    )
                    pd.testing.assert_series_equal(
                        synced["raw_value"], frame["raw_value"], check_names=False
                    )
                if sensor == "ecg" and result.clock_excluded:
                    # Preserve the excluded raw ECG for audit lineage.  It is
                    # never presented as a corrected reference or used for
                    # canonical validation.
                    synced["timestamp_corrected_utc"] = synced["timestamp_raw_utc"]
                    synced["timestamp_local"] = synced["timestamp_local"]
                    synced["elapsed_seconds"] = (
                        pd.to_datetime(
                            synced["timestamp_raw_utc"], utc=True, errors="coerce"
                        )
                        - pd.to_datetime(
                            synced["timestamp_raw_utc"], utc=True, errors="coerce"
                        ).iloc[0]
                    ).dt.total_seconds()
                    synced["quality_flags"] = "ecg_clock_excluded"
                else:
                    SignalFrameContract().assert_valid(synced)
                output_dir = self.sync_dir / str(participant_id) / str(recording_id)
                output_dir.mkdir(parents=True, exist_ok=True)
                synced.to_parquet(output_dir / f"{sensor}.parquet", index=False)
                _write_csv_mirror(synced, output_dir / f"{sensor}.parquet")

        result_frame = pd.DataFrame(
            {**asdict(result), "phase": phase} for result, phase in results
        )
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        sync_manifest_path = self.sync_dir / "synchronization_manifest_v3.parquet"
        result_frame.to_parquet(sync_manifest_path, index=False)
        _write_csv_mirror(result_frame, sync_manifest_path)
        validation_dir = self.report_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(validation_dir / "ecg_synchronization_v3.csv", index=False)
        return result_frame

    def _anchors(self) -> pd.DataFrame:
        mapper = AnchorMapper(
            self.config.raw_source.anchor_csv,
            self.config.timezone_local,
            mapping_csv_path=self.raw_root / "mapping.csv",
        )
        anchors = mapper.load()
        anchors["phase"] = anchors["phase"].astype(str).str.strip().str.lower()
        anchors = anchors[anchors["phase"].isin(MODEL_PHASES)].copy()
        anchors = anchors.rename(columns={"participant_code": "participant_id"})
        anchors["start_utc"] = anchors["start_timestamp"]
        anchors["end_utc"] = anchors["end_timestamp"]
        assert "routine" not in set(anchors["phase"])
        expected = {
            (f"ID-{index:02d}", phase)
            for index in range(1, 33)
            for phase in MODEL_PHASES
        }
        observed = set(zip(anchors["participant_id"], anchors["phase"], strict=True))
        if observed != expected or len(anchors) != len(expected):
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                "anchor_cardinality_invalid:"
                f"missing={missing};extra={extra};rows={len(anchors)}"
            )
        if anchors.duplicated(["participant_id", "phase"]).any():
            raise ValueError("anchor_duplicate_participant_phase")
        return anchors

    def segment(
        self,
        participant: str | None = None,
        *,
        mode: Literal["validation", "gold"] = "gold",
    ) -> pd.DataFrame:
        """Slice corrected signals at the full anchor boundary.

        This stage preserves the nominal 300-second source phase.  The
        canonical 240-second core is selected only after the one configured
        sensor filter has run, so filter transients stay outside the analysis
        window.  No rows are synthesized here.
        """

        sync_manifest_path = self.sync_dir / "synchronization_manifest_v3.parquet"
        assert sync_manifest_path.exists(), "run sync-raw before segment"
        sync_manifest = pd.read_parquet(sync_manifest_path)
        anchors = self._anchors()
        if participant:
            anchors = anchors[anchors["participant_id"] == participant]
        manifest_rows: list[dict[str, Any]] = []
        for anchor in anchors.itertuples():
            start, end = self._window_slicer.bounds(anchor.start_utc, mode="source")
            core_start, core_end = self._window_slicer.bounds(
                anchor.start_utc, mode=mode
            )
            recording_id = f"{anchor.participant_id}_{anchor.phase}_raw"
            sync_rows = sync_manifest[
                (sync_manifest["participant_id"] == anchor.participant_id)
                & (sync_manifest["phase"] == anchor.phase)
                & (sync_manifest["recording_id"] == recording_id)
            ]
            for sensor in ("ecg", "ppg_ir", "ppg_red", "ppg_green"):
                if len(sync_rows) != 1:
                    manifest_rows.append(
                        {
                            "participant_id": anchor.participant_id,
                            "phase": anchor.phase,
                            "sensor_channel": sensor,
                            "row_count": 0,
                            "status": "excluded",
                            "exclusion_reason": (
                                "sync_recording_cardinality:"
                                f"recording_id={recording_id};rows={len(sync_rows)}"
                            ),
                        }
                    )
                    continue
                source = (
                    self.sync_dir
                    / anchor.participant_id
                    / recording_id
                    / f"{sensor}.parquet"
                )
                if not source.is_file():
                    manifest_rows.append(
                        {
                            "participant_id": anchor.participant_id,
                            "phase": anchor.phase,
                            "sensor_channel": sensor,
                            "row_count": 0,
                            "status": "excluded",
                            "exclusion_reason": f"missing_sync_source:{source.name}",
                        }
                    )
                    continue
                frame = pd.read_parquet(source)
                timestamps = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
                segment = frame[(timestamps >= start) & (timestamps < end)].copy()
                if segment.empty:
                    manifest_rows.append(
                        {
                            "participant_id": anchor.participant_id,
                            "phase": anchor.phase,
                            "sensor_channel": sensor,
                            "row_count": 0,
                            "status": "excluded",
                            "exclusion_reason": "no_anchor_overlap",
                        }
                    )
                    continue
                segment = segment.sort_values(
                    "timestamp_corrected_utc", kind="stable"
                ).drop_duplicates("timestamp_corrected_utc", keep="first")
                segment["phase"] = anchor.phase
                segment["recording_id"] = recording_id
                segment["anchor_start_utc"] = anchor.start_utc
                segment["anchor_end_utc"] = anchor.end_utc
                segment["segment_start_utc"] = start
                segment["segment_end_utc"] = end
                segment["core_start_utc"] = core_start
                segment["core_end_utc"] = core_end
                segment["slice_mode"] = "source_300s"
                SignalFrameContract().assert_valid(segment)
                output_dir = self.segmented_dir / anchor.participant_id / anchor.phase
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"{sensor}.parquet"
                segment.to_parquet(output, index=False)
                _write_csv_mirror(segment, output)
                manifest_rows.append(
                    {
                        "participant_id": anchor.participant_id,
                        "phase": anchor.phase,
                        "sensor_channel": sensor,
                        "row_count": len(segment),
                        "start_utc": segment["timestamp_corrected_utc"].iloc[0],
                        "end_utc": segment["timestamp_corrected_utc"].iloc[-1],
                        "status": "ok",
                        "exclusion_reason": (
                            str(segment["exclusion_reason"].iloc[0])
                            if sensor == "ecg"
                            else ""
                        ),
                        "path": str(output.relative_to(self.root)).replace("\\", "/"),
                    }
                )
        result = pd.DataFrame(manifest_rows)
        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        segment_manifest_path = self.segmented_dir / "segmentation_manifest_v3.parquet"
        result.to_parquet(segment_manifest_path, index=False)
        _write_csv_mirror(result, segment_manifest_path)
        return result

    def _crop_to_core(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select the observed 240-second core after source-level filtering."""

        if frame.empty:
            return frame.copy()
        if "core_start_utc" in frame.columns and "core_end_utc" in frame.columns:
            core_start = pd.Timestamp(frame["core_start_utc"].iloc[0]).tz_convert("UTC")
            core_end = pd.Timestamp(frame["core_end_utc"].iloc[0]).tz_convert("UTC")
        elif "anchor_start_utc" in frame.columns:
            core_start, core_end = self._window_slicer.bounds(
                pd.Timestamp(frame["anchor_start_utc"].iloc[0]), mode="core"
            )
        else:
            raise ValueError("missing_core_window_lineage")
        timestamps = pd.to_datetime(
            frame["timestamp_corrected_utc"], utc=True, errors="coerce"
        )
        selected = frame.loc[
            (timestamps >= core_start) & (timestamps < core_end)
        ].copy()
        if selected.empty:
            raise ValueError("core_window_has_no_observed_samples")
        selected["core_start_utc"] = core_start
        selected["core_end_utc"] = core_end
        selected["slice_mode"] = "core_240s"
        return selected

    def _continuous_ppg_frame(
        self,
        frame: pd.DataFrame,
        channel: str,
        source_fs: float,
    ) -> tuple[pd.DataFrame, float, float]:
        """Keep a non-resampling compatibility adapter outside the canonical path.

        The retired continuous-rate contract must not manufacture a 25 Hz to
        100 Hz waveform.  A caller that already supplies an observed 100 Hz
        frame may still receive that unchanged frame for legacy unit callers;
        any other rate is rejected.  Current configuration rejects this route
        before it can produce Silver or validation evidence.
        """
        if channel not in {"ppg_green", "ppg_ir"}:
            raise RuntimeError("deprecated_continuous_rate_contract_unavailable")
        timestamps = pd.to_datetime(
            frame.get("timestamp_corrected_utc"), utc=True, errors="coerce"
        )
        if timestamps.isna().any() or len(timestamps) < 2:
            raise RuntimeError("deprecated_continuous_rate_contract_unavailable")
        elapsed_ms = timestamps.astype("int64").to_numpy(dtype=np.float64) / 1.0e6
        intervals_ms = np.diff(elapsed_ms)
        if np.any(intervals_ms <= 0.0):
            raise RuntimeError("deprecated_continuous_rate_contract_unavailable")
        observed_fs = 1_000.0 / float(np.median(intervals_ms))
        model_fs = 100.0
        if not np.isclose(observed_fs, model_fs, rtol=0.02, atol=0.05):
            raise RuntimeError("deprecated_continuous_rate_contract_unavailable")
        canonical = frame.copy()
        canonical["sampling_rate_hz"] = model_fs
        canonical["source_sampling_rate_hz"] = observed_fs
        return canonical, 25.0, model_fs

    def preprocess(self, participant: str | None = None) -> pd.DataFrame:
        """Create Silver using one canonical PPG filter and motion annotations."""

        manifest_path = self.segmented_dir / "segmentation_manifest_v3.parquet"
        assert manifest_path.exists(), "run segment before preprocess"
        manifest = pd.read_parquet(manifest_path)
        manifest = manifest[manifest["status"] == "ok"]
        if participant:
            manifest = manifest[manifest["participant_id"] == participant]
        outputs: list[dict[str, Any]] = []
        for row in manifest.itertuples():
            source = self.root / str(row.path)
            frame = pd.read_parquet(source)
            try:
                # Lead-in/out are acquisition context, not analysis data.
                # Crop first so a dropout outside [+30,+270) cannot suppress
                # filtering or reject an otherwise valid canonical core.
                frame = self._crop_to_core(frame)
            except (AnchorWindowCoverageError, ValueError) as exc:
                source_file = str(frame["source_file"].iloc[0])
                source_checksum = str(frame["source_checksum"].iloc[0])
                outputs.append(
                    {
                        "participant_id": row.participant_id,
                        "phase": row.phase,
                        "sensor_channel": row.sensor_channel,
                        "row_count": 0,
                        "quality_score": 0.0,
                        "quality_tier": "invalid",
                        "quality_flags": "core_window_excluded",
                        "motion_score": np.nan,
                        "motion_annotation": "motion_unavailable",
                        "accelerometer_scale_status": "unavailable",
                        "interpolated_sample_count": 0,
                        "source_file": source_file,
                        "source_checksum": source_checksum,
                        "exclusion_reason": f"core_window:{exc}",
                        "preprocessing_version": self.config.preprocessing_version,
                        "path": "",
                    }
                )
                continue
            timestamps = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
            values = pd.to_numeric(frame["raw_value"], errors="coerce")
            fs = float(frame["sampling_rate_hz"].median())
            interpolation_mask = np.zeros(len(frame), dtype=bool)
            interpolation_provenance = ""
            preprocessing_reason = ""
            if row.sensor_channel.startswith("ppg_"):
                source_fs = fs
                if self._continuous_rate_contract and row.sensor_channel in {
                    "ppg_green",
                    "ppg_ir",
                }:
                    frame, source_fs, fs = self._continuous_ppg_frame(
                        frame,
                        str(row.sensor_channel),
                        source_fs,
                    )
                    timestamps = pd.to_datetime(
                        frame["timestamp_corrected_utc"], utc=True
                    )
                    values = pd.to_numeric(frame["raw_value"], errors="coerce")
                cleaned_ppg = self._ppg_preprocessor.preprocess(
                    frame,
                    row.sensor_channel,
                    fs,
                )
                frame = cleaned_ppg.frame
                corrected = cleaned_ppg.clean_value
                interpolation_mask = cleaned_ppg.interpolated_mask
                timestamps = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
                values = pd.to_numeric(frame["raw_value"], errors="coerce")
                interpolation_provenance = str(
                    cleaned_ppg.frame["interpolation_provenance"].iloc[0]
                    if not cleaned_ppg.frame.empty
                    else ""
                )
                preprocessing_reason = cleaned_ppg.exclusion_reason or ""
            else:
                corrected = values.to_numpy(dtype=np.float64)
                if np.isfinite(corrected).all() and corrected.size > max(
                    30, int(fs * 3)
                ):
                    filter_spec = filter_config_for_channel("ecg")
                    highcut = min(float(filter_spec["highcut_hz"]), fs * 0.45)
                    corrected = SignalFilter.bandpass(
                        corrected,
                        fs=fs,
                        lowcut=float(filter_spec["lowcut_hz"]),
                        highcut=highcut,
                        order=int(filter_spec.get("order", 4)),
                    )
            # The waveform was cropped before interpolation/filtering.  The
            # only changes are bounded interpolation and one configured filter;
            # motion remains annotation-only.
            frame["clean_value"] = corrected
            frame["interpolated_mask"] = interpolation_mask
            timestamps = pd.to_datetime(
                frame["timestamp_corrected_utc"], utc=True, errors="coerce"
            )
            values = pd.to_numeric(frame["raw_value"], errors="coerce")
            corrected = pd.to_numeric(frame["clean_value"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            interpolation_mask = frame["interpolated_mask"].to_numpy(dtype=bool)
            if len(frame) != len(corrected):
                raise ValueError("clean_value_length_mismatch")
            madc_result = self._madc.process_window(frame, sampling_rate_hz=fs)
            motion_annotation = str(madc_result.get("motion_annotation", ""))
            score, tier, flags = self._sqa.score(
                values,
                timestamps,
                cleaned_values=corrected,
                channel=str(row.sensor_channel),
                sampling_rate_hz=fs,
                interpolation_mask=interpolation_mask,
            )
            if motion_annotation:
                flags.append(f"motion_annotation:{motion_annotation}")
            if not bool(madc_result.get("motion_available", False)):
                flags.append("motion_unavailable")
            if preprocessing_reason:
                flags.append(preprocessing_reason)
            if np.any(interpolation_mask) and "short_gap_interpolated" not in flags:
                flags.append("short_gap_interpolated")
            if madc_result.get("madc_action") == "MOTION_HEAVY_ANNOTATED":
                if bool(self._sqa_config.get("motion_annotation_only", True)):
                    flags.append("motion_heavy_masked")
                else:
                    flags.append("motion_heavy_exclusion")
                    score = 0.0
                    tier = "invalid"
                    preprocessing_reason = "motion_heavy_exclusion"

            motion_score = madc_result.get("motion_score", np.nan)
            motion_score_array = np.asarray(
                madc_result.get("motion_score_array", np.full(len(frame), np.nan)),
                dtype=float,
            )
            motion_artifact_mask = np.asarray(
                madc_result.get("artifact_mask", np.zeros(len(frame), dtype=bool)),
                dtype=bool,
            )
            motion_heavy_mask = np.asarray(
                madc_result.get("motion_heavy_mask", np.zeros(len(frame), dtype=bool)),
                dtype=bool,
            )
            if not (
                len(motion_score_array)
                == len(motion_artifact_mask)
                == len(motion_heavy_mask)
                == len(frame)
            ):
                raise ValueError("madc_output_length_mismatch")
            frame["motion_score"] = motion_score_array
            frame["motion_artifact_mask"] = motion_artifact_mask
            frame["motion_heavy_mask"] = motion_heavy_mask
            frame["heavy_motion_ratio"] = madc_result.get("heavy_motion_ratio", np.nan)
            frame["madc_window_seconds"] = madc_result.get(
                "madc_window_seconds", np.nan
            )
            frame["madc_window_samples"] = madc_result.get(
                "madc_window_samples", np.nan
            )
            frame["motion_annotation"] = motion_annotation
            frame["accelerometer_scale_status"] = str(
                madc_result.get("accelerometer_scale_status", "unavailable")
            )
            frame["motion_method"] = str(madc_result.get("motion_method", ""))
            frame["madc_action"] = str(madc_result.get("madc_action", ""))
            frame["madc_signal_modified"] = bool(
                madc_result.get("provenance", {}).get("signal_modified", False)
            )
            frame["interpolated_sample_count"] = int(interpolation_mask.sum())
            frame["interpolation_provenance"] = interpolation_provenance
            frame["quality_score"] = score
            frame["quality_tier"] = tier
            frame["quality_flags"] = "|".join(flags) if flags else "PASS"
            frame["exclusion_reason"] = preprocessing_reason
            frame["preprocessing_version"] = self.config.preprocessing_version
            source_rate_for_manifest = (
                float(frame["source_sampling_rate_hz"].iloc[0])
                if "source_sampling_rate_hz" in frame.columns
                else fs
            )
            SignalFrameContract().assert_valid(frame)
            output_dir = self.silver_dir / row.participant_id / row.phase
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"{row.sensor_channel}.parquet"
            frame.to_parquet(output, index=False)
            _write_csv_mirror(frame, output)
            outputs.append(
                {
                    "participant_id": row.participant_id,
                    "phase": row.phase,
                    "sensor_channel": row.sensor_channel,
                    "row_count": len(frame),
                    "quality_score": score,
                    "quality_tier": tier,
                    "quality_flags": "|".join(flags) if flags else "PASS",
                    "motion_score": (
                        float(motion_score) if motion_score is not None else np.nan
                    ),
                    "motion_annotation": motion_annotation,
                    "heavy_motion_ratio": madc_result.get("heavy_motion_ratio", np.nan),
                    "madc_window_seconds": madc_result.get(
                        "madc_window_seconds", np.nan
                    ),
                    "madc_window_samples": madc_result.get(
                        "madc_window_samples", np.nan
                    ),
                    "accelerometer_scale_status": str(
                        madc_result.get("accelerometer_scale_status", "unavailable")
                    ),
                    "interpolated_sample_count": int(interpolation_mask.sum()),
                    "source_file": str(frame["source_file"].iloc[0]),
                    "source_checksum": str(frame["source_checksum"].iloc[0]),
                    "exclusion_reason": preprocessing_reason,
                    "preprocessing_version": self.config.preprocessing_version,
                    "source_rate_hz": source_rate_for_manifest,
                    "model_rate_hz": fs,
                    "path": str(output.relative_to(self.root)).replace("\\", "/"),
                }
            )
            continue
        result = pd.DataFrame(outputs)
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        silver_manifest_path = self.silver_dir / "preprocessing_manifest_v3.parquet"
        result.to_parquet(silver_manifest_path, index=False)
        _write_csv_mirror(result, silver_manifest_path)
        return result

    @staticmethod
    def _uniform_window(
        timestamps: pd.Series,
        values: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        sampling_rate_hz: float,
        source_interpolated_mask: pd.Series | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        time_ms = (
            pd.to_datetime(timestamps, utc=True)
            .astype("int64")
            .to_numpy(dtype=np.float64)
            / 1_000_000.0
        )
        signal = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        keep = np.isfinite(time_ms)
        time_ms, signal = time_ms[keep], signal[keep]
        assert time_ms.size > 2
        finite = np.isfinite(signal)
        if not finite.any():
            raise ValueError("window_all_nan")
        intervals = np.diff(time_ms)
        expected_interval = 1000.0 / sampling_rate_hz
        max_gap_ms = 500.0
        gap_excess = intervals - expected_interval
        if np.any(gap_excess > max_gap_ms):
            raise ValueError("long_dropout_rejected")
        if not finite.all():
            missing_mask = ~finite
            padded = np.concatenate(([False], missing_mask, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
            if any(
                (end - start + 1) * expected_interval > max_gap_ms
                for start, end in zip(starts, ends)
            ):
                raise ValueError("long_dropout_rejected")
            if starts.size == 0 or starts[0] == 0 or ends[-1] == len(signal) - 1:
                raise ValueError("unbounded_dropout_rejected")
            interpolated_intervals = [
                (time_ms[gap_start - 1], time_ms[gap_end + 1])
                for gap_start, gap_end in zip(starts, ends)
            ]
            missing_indices = np.flatnonzero(missing_mask)
            signal[missing_indices] = np.interp(
                missing_indices,
                np.flatnonzero(finite),
                signal[finite],
            )
        else:
            interpolated_intervals = []
        timestamp_gap_intervals = [
            (time_ms[index], time_ms[index + 1])
            for index, excess in enumerate(gap_excess)
            if excess > max(1.0, expected_interval * 0.5)
        ]
        grid = np.arange(
            start.timestamp() * 1000.0,
            end.timestamp() * 1000.0,
            1000.0 / sampling_rate_hz,
        )
        assert grid.size >= int(179 * sampling_rate_hz)
        interpolated_mask = np.zeros(grid.shape, dtype=bool)
        for interval_start, interval_end in timestamp_gap_intervals:
            interpolated_mask |= (grid > interval_start) & (grid < interval_end)
        for interval_start, interval_end in interpolated_intervals:
            interpolated_mask |= (grid >= interval_start) & (grid <= interval_end)
        if source_interpolated_mask is not None:
            source_mask = pd.Series(source_interpolated_mask).to_numpy(dtype=bool)
            if source_mask.size == len(time_ms) and source_mask.any():
                source_times = time_ms[source_mask]
                half_interval = expected_interval / 2.0
                for sample_time in source_times:
                    interpolated_mask |= np.abs(grid - sample_time) <= half_interval
        return grid, np.interp(grid, time_ms, signal), interpolated_mask

    @staticmethod
    def _observed_window(
        timestamps: pd.Series,
        values: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        source_interpolated_mask: pd.Series | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], float]:
        """Return one finite observed window without resampling or padding."""

        time_ms = (
            pd.to_datetime(timestamps, utc=True, errors="coerce")
            .astype("int64")
            .to_numpy(dtype=np.float64)
            / 1_000_000.0
        )
        signal = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        source_mask = (
            pd.Series(source_interpolated_mask, index=values.index)
            .fillna(False)
            .to_numpy(dtype=bool)
            if source_interpolated_mask is not None
            else np.zeros(len(signal), dtype=bool)
        )
        if len(time_ms) != len(signal) or len(source_mask) != len(signal):
            raise ValueError("observed_window_length_mismatch")
        finite = np.isfinite(time_ms) & np.isfinite(signal)
        if finite.sum() < 3 or not finite.all():
            raise ValueError("non_finite_observed_window")
        if not np.all(np.diff(time_ms) > 0.0):
            raise ValueError("observed_window_timestamp_not_monotonic")
        intervals = np.diff(time_ms)
        median_interval = float(np.median(intervals))
        if not np.isfinite(median_interval) or median_interval <= 0.0:
            raise ValueError("observed_window_sampling_rate_unavailable")
        if np.any(intervals - median_interval > max(500.0, 3.0 * median_interval)):
            raise ValueError("long_dropout_rejected")
        start_ms = start.timestamp() * 1000.0
        end_ms = end.timestamp() * 1000.0
        selected = (time_ms >= start_ms) & (time_ms < end_ms)
        if selected.sum() < 3:
            raise ValueError("observed_window_has_too_few_samples")
        selected_times = time_ms[selected]
        coverage_tolerance_ms = max(500.0, 2.0 * median_interval)
        if (
            selected_times[0] > start_ms + coverage_tolerance_ms
            or selected_times[-1] < end_ms - coverage_tolerance_ms
        ):
            raise ValueError("insufficient_time_coverage")
        observed_duration_ms = float(selected_times[-1] - selected_times[0])
        if not np.isfinite(observed_duration_ms) or observed_duration_ms <= 0.0:
            raise ValueError("observed_window_sampling_rate_unavailable")
        sampling_rate_hz = (
            float(selected_times.size - 1) * 1000.0 / observed_duration_ms
        )
        if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
            raise ValueError("observed_window_sampling_rate_unavailable")
        return (
            selected_times,
            signal[selected],
            source_mask[selected],
            float(sampling_rate_hz),
        )

    def extract_features(self, participant: str | None = None) -> pd.DataFrame:
        """Create hierarchical Gold v3 features and fixed masked IBI sequences."""

        manifest_path = self.silver_dir / "preprocessing_manifest_v3.parquet"
        assert manifest_path.exists(), "run preprocess before extract-features"
        manifest = pd.read_parquet(manifest_path)
        manifest = manifest[manifest["sensor_channel"].str.startswith("ppg_")]
        if participant:
            manifest = manifest[manifest["participant_id"] == participant]
        feature_rows: list[dict[str, Any]] = []
        feature_maps: list[Mapping[str, float]] = []
        sequence_rows: list[dict[str, Any]] = []
        waveform_rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        waveform_contracts: list[dict[str, Any]] = []
        parity_pool = set(self.config.training.parity_pool_participant_ids)
        window_seconds = self.config.window.size_seconds
        step_seconds = self.config.window.step_seconds

        for row in manifest.itertuples():
            source = self.root / str(row.path)
            frame = pd.read_parquet(source)
            source_checksum = str(frame["source_checksum"].iloc[0])
            raw_source_file = str(frame["source_file"].iloc[0])
            session_id = str(
                frame.get("session_id", pd.Series([row.participant_id])).iloc[0]
            )
            recording_id = str(
                frame.get(
                    "recording_id", pd.Series([f"{row.participant_id}_{row.phase}_raw"])
                ).iloc[0]
            )
            model_rate_hz = float(frame["sampling_rate_hz"].median())
            source_rate_hz = (
                float(frame["source_sampling_rate_hz"].median())
                if "source_sampling_rate_hz" in frame.columns
                else model_rate_hz
            )
            lineage = {
                "participant_id": row.participant_id,
                "session_id": session_id,
                "recording_id": recording_id,
                "phase": row.phase,
                "sensor_channel": row.sensor_channel,
                "source_file": raw_source_file,
                "source_checksum": source_checksum,
                "quality_flags": "",
                "preprocessing_version": self.config.preprocessing_version,
                "source_rate_hz": source_rate_hz,
                "model_rate_hz": model_rate_hz,
                "peak_detection_method": "",
                "peak_detection_fallback_reason": "",
                "peak_detection_status": "UNAVAILABLE",
                "peak_detection_status_reason": "not_evaluated",
                "peak_count": 0,
                "physiological_ibi_ratio": 0.0,
            }
            timestamps = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
            first = (
                pd.Timestamp(frame["core_start_utc"].iloc[0]).tz_convert("UTC")
                if "core_start_utc" in frame.columns
                else timestamps.min()
            )
            last = (
                pd.Timestamp(frame["core_end_utc"].iloc[0]).tz_convert("UTC")
                if "core_end_utc" in frame.columns
                else timestamps.max()
            )
            window_index = 0
            start = first
            while start + pd.Timedelta(seconds=window_seconds) <= last:
                window_lineage = {
                    **lineage,
                    "peak_detection_method": "",
                    "peak_detection_fallback_reason": "",
                    "peak_detection_status": "UNAVAILABLE",
                    "peak_detection_status_reason": "not_evaluated",
                    "peak_count": 0,
                    "physiological_ibi_ratio": 0.0,
                }
                end = start + pd.Timedelta(seconds=window_seconds)
                selected = (timestamps >= start) & (timestamps < end)
                original = frame[selected]
                window_id = f"{row.participant_id}_{row.phase}_{row.sensor_channel}_{window_index:03d}"
                if (
                    original.empty
                    or (
                        pd.to_datetime(
                            original["timestamp_corrected_utc"], utc=True
                        ).iloc[-1]
                        - pd.to_datetime(
                            original["timestamp_corrected_utc"], utc=True
                        ).iloc[0]
                    ).total_seconds()
                    < window_seconds - 2.0
                ):
                    exclusions.append(
                        {
                            **window_lineage,
                            "window_id": window_id,
                            "quality_score": 0.0,
                            "quality_tier": "invalid",
                            "training_eligible": False,
                            "quality_policy_version": _QUALITY_POLICY_VERSION,
                            "exclusion_reason": "insufficient_time_coverage",
                        }
                    )
                    start += pd.Timedelta(seconds=step_seconds)
                    window_index += 1
                    continue
                fs = float(original["sampling_rate_hz"].median())
                score = 0.0
                tier = "invalid"
                try:
                    observed_ms, observed, interpolated_mask, fs = (
                        self._observed_window(
                            original["timestamp_corrected_utc"],
                            original["clean_value"],
                            start,
                            end,
                            original.get("interpolated_mask"),
                        )
                    )
                    observed_timestamps = pd.to_datetime(
                        observed_ms, unit="ms", utc=True
                    )
                    pre_quality = self._pre_peak_sqa.assess(
                        observed,
                        timestamps_utc=observed_timestamps,
                        channel=str(row.sensor_channel),
                        sampling_rate_hz=fs,
                        interpolation_mask=interpolated_mask,
                    )
                    detection = self._ppg_peak_detector.detect(
                        observed,
                        fs,
                        str(row.sensor_channel),
                    )
                    peaks = detection.peaks
                    window_lineage.update(
                        {
                            "peak_detection_method": detection.method,
                            "peak_detection_fallback_reason": detection.fallback_reason,
                            "peak_detection_status": detection.status,
                            "peak_detection_status_reason": detection.status_reason,
                            "peak_count": int(len(detection.peaks)),
                            "physiological_ibi_ratio": (
                                detection.physiological_ibi_ratio
                            ),
                        }
                    )
                    post_quality = self._post_peak_sqa.assess(
                        observed,
                        peaks,
                        sampling_rate_hz=fs,
                        timestamps_utc=observed_timestamps,
                        interpolation_mask=interpolated_mask,
                        pre_quality=pre_quality,
                    )
                    score = post_quality.score
                    tier = post_quality.tier
                    flags = list(post_quality.flags)
                    if tier not in self.config.training.allowed_quality_tiers:
                        raise ValueError(f"quality_tier:{tier}")
                    features, ibi_values, ibi_mask = extract_ppg_features(
                        observed,
                        fs,
                        channel=str(row.sensor_channel),
                        detector=self._ppg_peak_detector,
                        detection=detection,
                        window_seconds=window_seconds,
                        experimental_features=self.config.features.promoted_experimental,
                    )
                except (AssertionError, ValueError, FloatingPointError) as exc:
                    exclusions.append(
                        {
                            **window_lineage,
                            "window_id": window_id,
                            "quality_score": score,
                            "quality_tier": tier,
                            "training_eligible": False,
                            "quality_policy_version": _QUALITY_POLICY_VERSION,
                            "exclusion_reason": str(exc),
                        }
                    )
                    start += pd.Timedelta(seconds=step_seconds)
                    window_index += 1
                    continue
                feature_maps.append(features)
                interpolated_count = int(interpolated_mask.sum())
                quality_flags = list(flags) if flags else ["PASS"]
                if interpolated_count:
                    quality_flags.append("short_gap_interpolated")
                base = {
                    "participant_id": row.participant_id,
                    "session_id": session_id,
                    "recording_id": recording_id,
                    "phase": row.phase,
                    "sensor_channel": row.sensor_channel,
                    "window_id": window_id,
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": end.isoformat(),
                    "sampling_rate_hz": fs,
                    "source_rate_hz": source_rate_hz,
                    "model_rate_hz": model_rate_hz,
                    "label": _LABELS[row.phase],
                    "label_name": "Normal" if row.phase == "relax" else "Stres",
                    "quality_score": score,
                    "quality_tier": tier,
                    "training_eligible": bool(
                        tier in self.config.training.allowed_quality_tiers
                        and detection.status == "AVAILABLE"
                        and "motion_heavy_exclusion" not in quality_flags
                    ),
                    "quality_policy_version": _QUALITY_POLICY_VERSION,
                    "quality_flags": "|".join(quality_flags),
                    "interpolated_short_gap": bool(interpolated_count),
                    "interpolated_sample_count": interpolated_count,
                    "interpolation_provenance": (
                        "silver_bounded_linear_waveform" if interpolated_count else ""
                    ),
                    "exclusion_reason": "",
                    "feature_version": self.config.feature_version,
                    "preprocessing_version": self.config.preprocessing_version,
                    "split": "parity_pool"
                    if row.participant_id in parity_pool
                    else "train",
                    "source_file": raw_source_file,
                    "source_checksum": source_checksum,
                    "motion_score": float(row.motion_score)
                    if hasattr(row, "motion_score")
                    else np.nan,
                    "motion_annotation": str(row.motion_annotation)
                    if hasattr(row, "motion_annotation")
                    else "",
                    "accelerometer_scale_status": str(row.accelerometer_scale_status)
                    if hasattr(row, "accelerometer_scale_status")
                    else "unavailable",
                    "motion_method": str(row.motion_method)
                    if hasattr(row, "motion_method")
                    else "",
                    "interpolated_fraction": float(interpolated_count)
                    / max(1, len(observed)),
                    "peak_detection_method": detection.method,
                    "peak_detection_fallback_reason": detection.fallback_reason,
                    "peak_detection_status": detection.status,
                    "peak_detection_status_reason": detection.status_reason,
                    "peak_count": int(len(detection.peaks)),
                    "physiological_ibi_ratio": detection.physiological_ibi_ratio,
                }
                feature_rows.append({**base, **features})
                sequence_rows.append(
                    {
                        **base,
                        "ibi_values": ibi_values.tolist(),
                        "ibi_mask": ibi_mask.tolist(),
                        "waveform_interpolated_mask": interpolated_mask.tolist(),
                    }
                )
                waveform_rows.append(
                    {
                        **base,
                        "waveform": observed.astype(np.float32),
                        "interpolated_mask": interpolated_mask.tolist(),
                    }
                )
                start += pd.Timedelta(seconds=step_seconds)
                window_index += 1

        gold = pd.DataFrame(feature_rows)
        sequences = pd.DataFrame(sequence_rows)
        if gold.empty:
            gold = pd.DataFrame(
                columns=[
                    "participant_id",
                    "session_id",
                    "recording_id",
                    "phase",
                    "sensor_channel",
                    "window_id",
                    "window_start_utc",
                    "window_end_utc",
                    "sampling_rate_hz",
                    "label",
                    "quality_score",
                    "quality_tier",
                    "training_eligible",
                    "quality_policy_version",
                    "label_name",
                    "quality_flags",
                    "feature_version",
                    "preprocessing_version",
                    "split",
                    "source_file",
                    "source_checksum",
                ]
            )
        if sequences.empty:
            sequences = pd.DataFrame(
                columns=[
                    "participant_id",
                    "session_id",
                    "phase",
                    "sensor_channel",
                    "window_id",
                ]
            )
        available_feature_columns = (
            tuple(sorted(feature_maps[0])) if feature_maps else ()
        )
        predictors = tuple(self.config.features.predictor_default)
        if not gold.empty:
            validate_feature_order(
                predictors,
                (),
            )
            missing_predictors = sorted(set(predictors) - set(gold.columns))
            if missing_predictors:
                raise ValueError(
                    f"canonical_predictors_missing:{','.join(missing_predictors)}"
                )
            assert "routine" not in set(gold["phase"])

        waveforms = pd.DataFrame(waveform_rows)
        waveform_dir = self.gold_dir / "waveforms"
        for (participant_id, phase, sensor), group in gold.groupby(
            ["participant_id", "phase", "sensor_channel"], sort=True
        ):
            output_dir = self.gold_dir / "features" / "stress" / participant_id / phase
            output_dir.mkdir(parents=True, exist_ok=True)
            group.to_parquet(output_dir / f"{sensor}.parquet", index=False)
            _write_csv_mirror(group, output_dir / f"{sensor}.parquet")

            # Save raw waveforms
            w_group = waveforms[
                (waveforms["participant_id"] == participant_id)
                & (waveforms["phase"] == phase)
                & (waveforms["sensor_channel"] == sensor)
            ]
            if not w_group.empty:
                w_dir = waveform_dir / participant_id / phase
                w_dir.mkdir(parents=True, exist_ok=True)
                lengths = {len(np.asarray(value)) for value in w_group["waveform"]}
                if len(lengths) != 1:
                    waveform_contracts.append(
                        {
                            "status": "UNAVAILABLE",
                            "reason": "observed_window_lengths_are_not_fixed",
                            "participant_id": participant_id,
                            "phase": phase,
                        }
                    )
                else:
                    w_array = np.stack(w_group["waveform"].values).astype(
                        np.float32, copy=False
                    )
                    masks = np.stack(
                        [
                            np.asarray(value, dtype=np.uint8)
                            for value in w_group["interpolated_mask"]
                        ]
                    )
                    try:
                        normalized, observed_mask, w_means, w_stds = (
                            _normalize_observed_waveform_batch(w_array, masks)
                        )
                    except ValueError as exc:
                        waveform_contracts.append(
                            {
                                "status": "UNAVAILABLE",
                                "reason": str(exc),
                                "participant_id": participant_id,
                                "phase": phase,
                                "sensor_channel": sensor,
                            }
                        )
                        continue
                    waveform_path = w_dir / f"{sensor}.npz"
                    np.savez_compressed(
                        waveform_path,
                        waveforms=np.expand_dims(normalized, axis=-1),
                        observed_mask=observed_mask,
                        interpolated_mask=masks,
                        normalization_mean=w_means.astype(np.float32),
                        normalization_std=w_stds.astype(np.float32),
                        window_ids=np.asarray(
                            w_group["window_id"].astype(str), dtype="U"
                        ),
                        participant_ids=np.asarray(
                            w_group["participant_id"].astype(str), dtype="U"
                        ),
                        session_ids=np.asarray(
                            w_group["session_id"].astype(str), dtype="U"
                        ),
                        phases=np.asarray(w_group["phase"].astype(str), dtype="U"),
                        sensor_channels=np.asarray(
                            w_group["sensor_channel"].astype(str), dtype="U"
                        ),
                        quality_tiers=np.asarray(
                            w_group["quality_tier"].astype(str), dtype="U"
                        ),
                        source_files=np.asarray(
                            w_group["source_file"].astype(str), dtype="U"
                        ),
                        source_checksums=np.asarray(
                            w_group["source_checksum"].astype(str), dtype="U"
                        ),
                        sampling_rate_hz=w_group["sampling_rate_hz"].to_numpy(
                            dtype=np.float32
                        ),
                        source_sampling_rate_hz=w_group["source_rate_hz"].to_numpy(
                            dtype=np.float32
                        ),
                        target_sampling_rate_hz=w_group["model_rate_hz"].to_numpy(
                            dtype=np.float32
                        ),
                        window_start_utc=np.asarray(
                            w_group["window_start_utc"].astype(str), dtype="U"
                        ),
                        window_end_utc=np.asarray(
                            w_group["window_end_utc"].astype(str), dtype="U"
                        ),
                    )
                    w_group.drop(columns=["waveform"]).to_parquet(
                        w_dir / f"{sensor}_lineage.parquet", index=False
                    )
                    waveform_contracts.append(
                        {
                            "status": "AVAILABLE",
                            "path": str(waveform_path.relative_to(self.root)).replace(
                                "\\", "/"
                            ),
                            "sha256": _sha256(waveform_path),
                            "shape": [
                                int(value)
                                for value in np.expand_dims(normalized, -1).shape
                            ],
                            "normalization": "per_window_zscore_observed_only",
                            "observed_mask": True,
                            "interpolated_mask": True,
                            "participant_id": participant_id,
                            "phase": phase,
                            "sensor_channel": sensor,
                        }
                    )

        self.gold_dir.mkdir(parents=True, exist_ok=True)
        gold_path = self.gold_dir / "gold_features_v3.parquet"
        sequence_path = self.gold_dir / "gold_ibi_sequences_v3.parquet"
        exclusion_path = self.gold_dir / "gold_exclusions_v3.parquet"
        gold.to_parquet(gold_path, index=False)
        sequences.to_parquet(sequence_path, index=False)
        exclusions_frame = pd.DataFrame(exclusions)
        exclusions_frame.to_parquet(exclusion_path, index=False)
        _write_csv_mirror(gold, gold_path)
        _write_csv_mirror(sequences, sequence_path)
        _write_csv_mirror(exclusions_frame, exclusion_path)
        run_contract_path = (
            self._configured_path("artifacts_root", "data/artifacts")
            / "run_contract_fingerprint.json"
        )
        run_contract = (
            json.loads(run_contract_path.read_text(encoding="utf-8"))
            if run_contract_path.is_file()
            else {}
        )
        silver_files = sorted(self.silver_dir.glob("ID-*/*/*.parquet"))
        _write_json(
            self.gold_dir / "manifest_v3.json",
            {
                "preprocessing_version": self.config.preprocessing_version,
                "feature_version": self.config.feature_version,
                "feature_order": predictors,
                "feature_count": len(predictors),
                "available_feature_columns": available_feature_columns,
                "window_count": len(gold),
                "sequence_length": self.config.window.ibi_sequence_length,
                "source_phase_duration_seconds": self.config.window.source_phase_duration_seconds,
                "edge_trim_start_seconds": self.config.window.edge_trim_start_seconds,
                "edge_trim_end_seconds": self.config.window.edge_trim_end_seconds,
                "analysis_phase_duration_seconds": self.config.window.analysis_phase_duration_seconds,
                "gold_window_seconds": self.config.window.gold_window_seconds,
                "gold_stride_seconds": self.config.window.gold_stride_seconds,
                "training_channels": list(self.config.training.training_channels),
                "run_id": str(self.config.artifacts.get("run_id", "")),
                "run_contract_sha256": (
                    _sha256(run_contract_path) if run_contract_path.is_file() else None
                ),
                "source_tree_sha256": run_contract.get("source_tree_sha256"),
                "config_fingerprint": run_contract.get("config_fingerprint"),
                "source_checksums": run_contract.get("source_checksums", {}),
                "source_silver_manifest": str(
                    (self.silver_dir / "preprocessing_manifest_v3.parquet").relative_to(
                        self.root
                    )
                ).replace("\\", "/"),
                "source_silver_manifest_sha256": _sha256(
                    self.silver_dir / "preprocessing_manifest_v3.parquet"
                ),
                "waveform_contracts": waveform_contracts,
                "split_contract": {
                    "development": list(
                        self.config.training.development_participant_ids
                    ),
                    "holdout": list(self.config.training.parity_pool_participant_ids),
                    "holdout_status": self.config.training.reused_test_cohort_status,
                },
                "development_participants": [
                    f"ID-{index:02d}" for index in range(1, 29)
                ],
                "parity_pool_participants": list(
                    self.config.training.parity_pool_participant_ids
                ),
                "input_hashes": {
                    str(path.relative_to(self.root)).replace("\\", "/"): _sha256(path)
                    for path in silver_files
                },
                "artifact_checksums": {
                    "gold_features_v3.parquet": _sha256(gold_path),
                    "gold_ibi_sequences_v3.parquet": _sha256(sequence_path),
                    "gold_exclusions_v3.parquet": _sha256(exclusion_path),
                },
                "exclusions": exclusions,
            },
        )
        expected_groups = {
            (participant_id, phase, sensor)
            for participant_id in (f"ID-{index:02d}" for index in range(1, 33))
            for phase in MODEL_PHASES
            for sensor in ("ppg_ir", "ppg_red", "ppg_green")
        }
        observed_groups = (
            set(
                gold[["participant_id", "phase", "sensor_channel"]].itertuples(
                    index=False, name=None
                )
            )
            if not gold.empty
            else set()
        )
        missing_groups = sorted(expected_groups - observed_groups)
        observed_group_count = len(observed_groups)
        expected_group_count = len(expected_groups)
        training_coverage_threshold = float(
            self.config.training.minimum_training_development_group_coverage
        )
        group_coverage_fraction = observed_group_count / max(expected_group_count, 1)
        ppg_gold_status = (
            "ready"
            if group_coverage_fraction >= training_coverage_threshold
            else "blocked"
        )
        training_channels = tuple(self.config.training.training_channels)
        expected_training_groups = {
            (participant_id, phase, sensor)
            for participant_id in (f"ID-{index:02d}" for index in range(1, 33))
            for phase in MODEL_PHASES
            for sensor in training_channels
        }
        observed_training_groups = {
            group for group in observed_groups if group[2] in training_channels
        }
        channel_readiness = channel_group_readiness(
            observed_training_groups,
            minimum_training_development_group_coverage=(training_coverage_threshold),
            minimum_development_groups_per_phase=(
                self.config.training.minimum_development_groups_per_phase
            ),
            minimum_development_participants=(
                self.config.training.minimum_development_participants
            ),
        )
        missing_training_groups = sorted(
            expected_training_groups - observed_training_groups
        )
        training_status = str(channel_readiness["training_status"])
        strict_window_count = (
            int(gold["quality_tier"].isin(_STRICT_QUALITY_TIERS).sum())
            if not gold.empty
            else 0
        )
        expanded_window_count = int(len(gold))
        expected_windows_per_group = (
            (
                self.config.window.analysis_phase_duration_seconds
                - self.config.window.gold_window_seconds
            )
            // self.config.window.gold_stride_seconds
        ) + 1
        expected_window_count = expected_group_count * expected_windows_per_group
        window_coverage_fraction = expanded_window_count / max(expected_window_count, 1)
        window_counts_by_channel = (
            gold.groupby("sensor_channel").size().astype(int).to_dict()
            if not gold.empty
            else {}
        )
        expected_windows_by_channel = {
            channel: 64 * expected_windows_per_group for channel in training_channels
        }
        window_coverage_by_channel = {
            channel: window_counts_by_channel.get(channel, 0)
            / max(expected_windows_by_channel[channel], 1)
            for channel in training_channels
        }
        strict_group_count = (
            len(
                set(
                    gold.loc[
                        gold["quality_tier"].isin(_STRICT_QUALITY_TIERS),
                        ["participant_id", "phase", "sensor_channel"],
                    ].itertuples(index=False, name=None)
                )
            )
            if not gold.empty
            else 0
        )
        validation_summary = self._read_validation_summary()
        if validation_summary is None:
            ecg_validation_status = "blocked"
            ecg_validation_reason = "validation_not_run"
            ecg_validation_combinations = 0
            beat_diagnostic_status = "not_evaluated"
            hrv_validation_decision = "INSUFFICIENT_EVIDENCE"
        else:
            ecg_validation_combinations = int(
                validation_summary.get("total_combinations", 0)
            )
            beat_diagnostic_status = str(
                validation_summary.get("beat_diagnostic_status", "reported")
            )
            hrv_validation_decision = str(
                validation_summary.get(
                    "hrv_validation_decision", "INSUFFICIENT_EVIDENCE"
                )
            )
            ecg_validation_status = (
                "ready" if hrv_validation_decision == "GO" else "blocked"
            )
            ecg_validation_reason = (
                "hrv_prv_primary_gate_passed"
                if ecg_validation_status == "ready"
                else "hrv_prv_primary_gate_not_passed"
            )
        readiness_status = (
            "ready"
            if ppg_gold_status == "ready" and training_status == "ready"
            else "blocked"
        )
        _write_json(
            self.gold_dir / "data_readiness.json",
            {
                "run_id": str(self.config.artifacts.get("run_id", "")),
                "run_contract_sha256": (
                    _sha256(run_contract_path) if run_contract_path.is_file() else None
                ),
                "source_tree_sha256": run_contract.get("source_tree_sha256"),
                "config_fingerprint": run_contract.get("config_fingerprint"),
                "status": readiness_status,
                "ppg_gold_status": ppg_gold_status,
                "gold_completeness_status": (
                    "complete" if not missing_groups else "partial"
                ),
                "gold_group_coverage_fraction": group_coverage_fraction,
                "gold_group_coverage_threshold": training_coverage_threshold,
                "gold_group_coverage_status": ppg_gold_status,
                "gold_group_coverage_unit": "participant_phase_channel_group",
                "training_group_coverage_by_channel": channel_readiness[
                    "coverage_fraction"
                ],
                "training_group_coverage_status_by_channel": channel_readiness[
                    "coverage_status"
                ],
                "training_participant_coverage_by_channel": channel_readiness[
                    "participant_coverage_fraction"
                ],
                "training_participant_coverage_status_by_channel": channel_readiness[
                    "participant_coverage_status"
                ],
                "beat_diagnostic_status": beat_diagnostic_status,
                "hrv_validation_decision": hrv_validation_decision,
                "ecg_validation_channel_status": {},
                "hrv_validation_channel_statuses": {},
                "spectral_validation_channel_statuses": {},
                "ecg_validation_status": ecg_validation_status,
                "ecg_validation_reason": ecg_validation_reason,
                "training_status": training_status,
                "model_performance_status": "not_evaluated",
                "model_release_status": "blocked",
                "model_release_status_by_channel": {
                    channel: "not_evaluated" for channel in training_channels
                },
                "model_release_reason": "model_evaluation_not_run_for_core_240s",
                "source_phase_duration_seconds": self.config.window.source_phase_duration_seconds,
                "edge_trim_start_seconds": self.config.window.edge_trim_start_seconds,
                "edge_trim_end_seconds": self.config.window.edge_trim_end_seconds,
                "analysis_phase_duration_seconds": self.config.window.analysis_phase_duration_seconds,
                "validation_window_seconds": self.config.window.validation_window_seconds,
                "gold_window_seconds": self.config.window.gold_window_seconds,
                "gold_stride_seconds": self.config.window.gold_stride_seconds,
                "participant_ids": sorted(gold["participant_id"].unique().tolist())
                if not gold.empty
                else [],
                "phase_values": sorted(gold["phase"].unique().tolist())
                if not gold.empty
                else [],
                "sensor_channels": sorted(gold["sensor_channel"].unique().tolist())
                if not gold.empty
                else [],
                "ml_ready_window_count": int(len(gold)),
                "expanded_training_window_count": expanded_window_count,
                "strict_training_window_count": strict_window_count,
                "expected_window_count": expected_window_count,
                "window_coverage_fraction": window_coverage_fraction,
                "window_coverage_threshold": training_coverage_threshold,
                "window_coverage_status": (
                    "ready"
                    if window_coverage_fraction >= training_coverage_threshold
                    else "blocked"
                ),
                "expected_windows_per_group": expected_windows_per_group,
                "expected_windows_by_channel": expected_windows_by_channel,
                "window_coverage_by_channel": window_coverage_by_channel,
                "excluded_window_count": int(len(exclusions)),
                "expected_group_count": len(expected_groups),
                "observed_group_count": observed_group_count,
                "expected_ppg_groups": len(expected_groups),
                "observed_ppg_groups": observed_group_count,
                "training_channels": list(training_channels),
                "expected_training_groups": len(expected_training_groups),
                "observed_training_groups": len(observed_training_groups),
                "gold_channel_completeness_status": channel_readiness["channel_status"],
                "training_channel_status": channel_readiness["training_channel_status"],
                "training_development_coverage": channel_readiness[
                    "development_coverage"
                ],
                "expected_training_groups_by_channel": channel_readiness[
                    "expected_groups"
                ],
                "observed_training_groups_by_channel": channel_readiness[
                    "observed_groups"
                ],
                "missing_training_groups_by_channel": channel_readiness[
                    "missing_groups"
                ],
                "missing_training_groups": [
                    list(group) for group in missing_training_groups
                ],
                "strict_observed_group_count": strict_group_count,
                "expanded_observed_group_count": observed_group_count,
                "strict_quality_tiers": sorted(_STRICT_QUALITY_TIERS),
                "expanded_quality_tiers": list(
                    self.config.training.allowed_quality_tiers
                ),
                "quality_policy_version": _QUALITY_POLICY_VERSION,
                "ecg_validation_combinations": ecg_validation_combinations,
                "missing_groups": [list(group) for group in missing_groups],
                "allowed_quality_tiers": list(
                    self.config.training.allowed_quality_tiers
                ),
                "gold_artifact_checksums": {
                    "gold_features_v3.parquet": _sha256(gold_path),
                    "gold_ibi_sequences_v3.parquet": _sha256(sequence_path),
                    "gold_exclusions_v3.parquet": _sha256(exclusion_path),
                },
                "holdout_participants": list(
                    self.config.training.parity_pool_participant_ids
                ),
                "holdout_used_for_fitting": False,
                "lineage_columns": [
                    "participant_id",
                    "session_id",
                    "phase",
                    "sensor_channel",
                    "window_id",
                    "source_file",
                    "source_checksum",
                    "quality_score",
                    "quality_tier",
                    "quality_flags",
                    "exclusion_reason",
                    "label_name",
                    "preprocessing_version",
                ],
            },
        )
        return gold

    def run_data_stages(self, participant: str | None = None) -> pd.DataFrame:
        self.validate_inputs()
        self.ingest(participant)
        self.sync_raw(participant)
        self.segment(participant)
        self.preprocess(participant)
        return self.extract_features(participant)
