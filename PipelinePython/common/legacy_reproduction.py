"""Isolated reproduction of the historical PipelineSignal HRV estimand.

This module intentionally reproduces transformations that are unsuitable for
canonical physiological validation.  Its artifacts are diagnostic-only and
cannot be consumed by Gold, stress training, or ONNX publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import pearsonr, spearmanr

PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(1, 33))
PHASES = ("relax", "task")
CHANNELS = ("ppg_green", "ppg_ir", "ppg_red")
FEATURES = (
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_pct",
    "lf_power_ms2",
    "hf_power_ms2",
    "lf_hf_ratio",
)
LEGACY_REPORT_FEATURES = {
    "RMSSD (ms)": "rmssd_ms",
    "SDNN (ms)": "sdnn_ms",
    "pNN50 (%)": "pnn50_pct",
    "LF (ms²)": "lf_power_ms2",
    "HF (ms²)": "hf_power_ms2",
    "LF/HF Ratio": "lf_hf_ratio",
}


@dataclass(frozen=True)
class LegacyReproductionConfig:
    segment_seconds: float = 30.0
    target_rate_hz: float = 100.0
    lowcut_hz: float = 0.5
    highcut_hz: float = 4.0
    filter_order: int = 4
    minimum_segment_coverage: float = 0.95
    minimum_peaks_per_segment: int = 4
    peak_distance_seconds: float = 1.0 / 3.0
    peak_prominence_std: float = 0.25
    peak_height_std: float = 0.20
    edge_margin_seconds: float = 0.5
    ibi_min_ms: float = 333.0
    ibi_max_ms: float = 1500.0
    minimum_intervals: int = 10
    tachogram_rate_hz: float = 4.0
    max_lag_seconds: float = 30.0
    lag_coarse_step_seconds: float = 0.5
    lag_fine_step_seconds: float = 0.05


@dataclass(frozen=True)
class SegmentDecision:
    segment_index: int
    start_seconds: float
    end_seconds: float
    sample_count: int
    expected_sample_count: int
    quality_label: str
    quality_score: int
    accepted: bool
    peak_count: int
    derived_sample_count: int
    reason: str


def cubic_upsample(
    times_seconds: Sequence[float],
    values: Sequence[float],
    target_rate_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce natural-cubic waveform upsampling without leaving support."""

    times = np.asarray(times_seconds, dtype=float)
    signal = np.asarray(values, dtype=float)
    if times.ndim != 1 or signal.ndim != 1 or len(times) != len(signal):
        raise ValueError("upsampling_requires_equal_one_dimensional_inputs")
    if len(times) < 4 or not np.isfinite(times).all() or not np.isfinite(signal).all():
        raise ValueError("upsampling_requires_four_finite_samples")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("upsampling_requires_strictly_increasing_time")
    grid = np.arange(times[0], times[-1], 1.0 / target_rate_hz)
    spline = CubicSpline(times, signal, bc_type="natural", extrapolate=True)
    return grid, np.asarray(spline(grid), dtype=float)


def compressed_peak_intervals(
    peak_times_seconds: Sequence[float],
    peak_segment_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce legacy adjacency and count rejected-segment bridges."""

    times = np.asarray(peak_times_seconds, dtype=float)
    segment_ids = np.asarray(peak_segment_ids, dtype=int)
    if times.size != segment_ids.size:
        raise ValueError("peak_times_and_segment_ids_must_match")
    if times.size < 2:
        return np.empty(0), np.empty(0), 0
    order = np.argsort(times, kind="stable")
    times = times[order]
    segment_ids = segment_ids[order]
    intervals = np.diff(times) * 1000.0
    midpoints = (times[:-1] + times[1:]) / 2.0
    bridged = int(np.sum(np.diff(segment_ids) > 1))
    return intervals, midpoints, bridged


def legacy_hrv_features(
    peak_times_seconds: Sequence[float],
    peak_segment_ids: Sequence[int],
    config: LegacyReproductionConfig,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute exact legacy-style compressed HRV/PRV features."""

    rr_ms, rr_mid, bridged = compressed_peak_intervals(
        peak_times_seconds, peak_segment_ids
    )
    original_count = len(rr_ms)
    rr_ms, rr_mid = _filter_rr_outliers(rr_ms, rr_mid, config)
    rejected_count = original_count - len(rr_ms)
    empty = {feature: float("nan") for feature in FEATURES}
    audit = {
        "original_interval_count": original_count,
        "clean_interval_count": len(rr_ms),
        "rejected_interval_count": rejected_count,
        "bridged_rejected_segment_count": bridged,
        "compressed_successive_difference_count": max(len(rr_ms) - 1, 0),
        "tachogram_extrapolation_enabled": 1,
        # The historical grid lies between the first and final interval time.
        # ``fill_value='extrapolate'`` is enabled but normally evaluates zero
        # points outside support; hidden internal gaps are the actual concern.
        "tachogram_extrapolated_point_count": 0,
    }
    if len(rr_ms) < config.minimum_intervals:
        return empty, audit

    differences = np.diff(rr_ms)
    features = {
        "rmssd_ms": float(np.sqrt(np.mean(np.square(differences)))),
        "sdnn_ms": float(np.std(rr_ms, ddof=1)),
        "pnn50_pct": float(np.mean(np.abs(differences) > 50.0) * 100.0),
        **_frequency_features(rr_ms, rr_mid, config),
    }
    return features, audit


def optimal_ihr_lag(
    ecg_peak_times: Sequence[float],
    ppg_peak_times: Sequence[float],
    config: LegacyReproductionConfig,
) -> tuple[float, float, float]:
    """Reproduce same-recording absolute-IHR-correlation lag selection."""

    ecg_times, ecg_ihr = _instantaneous_hr(ecg_peak_times)
    ppg_times, ppg_ihr = _instantaneous_hr(ppg_peak_times)
    if len(ecg_times) < 5 or len(ppg_times) < 5:
        return 0.0, float("nan"), float("nan")

    before = _ihr_correlation(ecg_times, ecg_ihr, ppg_times, ppg_ihr, 0.0)
    best_lag = 0.0
    best_correlation = 0.0
    coarse = np.arange(
        -config.max_lag_seconds,
        config.max_lag_seconds + config.lag_coarse_step_seconds,
        config.lag_coarse_step_seconds,
    )
    for lag in coarse:
        correlation = _ihr_correlation(
            ecg_times, ecg_ihr, ppg_times, ppg_ihr, float(lag)
        )
        if np.isfinite(correlation) and abs(correlation) > abs(best_correlation):
            best_lag = float(lag)
            best_correlation = float(correlation)
    fine = np.arange(
        best_lag - config.lag_coarse_step_seconds,
        best_lag + config.lag_coarse_step_seconds + config.lag_fine_step_seconds,
        config.lag_fine_step_seconds,
    )
    for lag in fine:
        correlation = _ihr_correlation(
            ecg_times, ecg_ihr, ppg_times, ppg_ihr, float(lag)
        )
        if np.isfinite(correlation) and abs(correlation) > abs(best_correlation):
            best_lag = float(lag)
            best_correlation = float(correlation)
    return best_lag, before, best_correlation


class LegacyReproductionRunner:
    """Run the legacy estimand against one immutable canonical Silver run."""

    def __init__(
        self,
        project_root: Path,
        source_run_id: str,
        run_id: str,
        config: LegacyReproductionConfig | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.source_run_id = source_run_id
        self.run_id = run_id
        self.config = config or LegacyReproductionConfig()
        self.source_root = (
            self.project_root / "data" / "artifacts" / "runs" / source_run_id
        )
        self.output_root = (
            self.project_root
            / "data"
            / "artifacts"
            / "diagnostics"
            / "pipeline_signal_reproduction"
            / run_id
        )

    def run(self) -> dict[str, Any]:
        manifest_path = (
            self.source_root / "silver" / "preprocessing_manifest_v3.parquet"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"source_silver_manifest_missing:{manifest_path}")
        if self.output_root.exists():
            raise FileExistsError(
                f"legacy_reproduction_output_collision:{self.output_root}"
            )

        manifest = pd.read_parquet(manifest_path)
        required = {"participant_id", "phase", "sensor_channel", "path"}
        if not required.issubset(manifest.columns):
            raise ValueError("source_silver_manifest_contract_mismatch")
        rows: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        for participant_id in PARTICIPANTS:
            for phase in PHASES:
                ecg_selection = self._select_manifest_row(
                    manifest, participant_id, phase, "ecg"
                )
                for channel in CHANNELS:
                    ppg_selection = self._select_manifest_row(
                        manifest, participant_id, phase, channel
                    )
                    session_row, segment_rows = self._session(
                        participant_id,
                        phase,
                        channel,
                        ecg_selection,
                        ppg_selection,
                    )
                    rows.append(session_row)
                    segments.extend(segment_rows)

        sessions = pd.DataFrame(rows)
        if len(sessions) != 192:
            raise AssertionError(
                f"legacy_reproduction_expected_192_rows:{len(sessions)}"
            )
        segment_frame = pd.DataFrame(segments)
        metrics = _agreement_metrics(sessions)
        comparison = self._comparison(sessions, metrics, manifest_path)
        self._publish(sessions, segment_frame, metrics, comparison, manifest_path)
        return {
            "run_id": self.run_id,
            "source_run_id": self.source_run_id,
            "status": "LEGACY_REPRODUCTION_ONLY",
            "canonical_status_affected": False,
            "model_eligible": False,
            "session_row_count": len(sessions),
            "computed_session_count": int(sessions["status"].eq("COMPUTED").sum()),
            "output_dir": str(self.output_root),
        }

    def _session(
        self,
        participant_id: str,
        phase: str,
        channel: str,
        ecg_selection: pd.Series | None,
        ppg_selection: pd.Series | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        base = self._base_row(participant_id, phase, channel)
        if ecg_selection is None or ppg_selection is None:
            missing = "ecg" if ecg_selection is None else channel
            return {**base, "status": "EXCLUDED", "reason": f"missing_{missing}"}, []
        try:
            ecg = self._read_frame(ecg_selection)
            ppg = self._read_frame(ppg_selection)
            ppg_times, ppg_values, ppg_rate, silver_interpolated = _signal_arrays(ppg)
            ecg_times, ecg_values, ecg_rate, _ = _signal_arrays(ecg)
            ppg_peaks, ppg_segments, segment_rows, derived_count = self._ppg_peaks(
                participant_id, phase, channel, ppg_times, ppg_values, ppg_rate
            )
            ecg_peaks = _ecg_peaks(ecg_times, ecg_values, ecg_rate)
            if not ppg_segments or len(ppg_peaks) < 2 or len(ecg_peaks) < 2:
                return (
                    {
                        **base,
                        **self._lineage(ecg_selection, ppg_selection),
                        "status": "EXCLUDED",
                        "reason": "insufficient_accepted_peaks",
                        "accepted_segment_count": len(ppg_segments),
                        "ppg_peak_count": len(ppg_peaks),
                        "ecg_peak_count": len(ecg_peaks),
                        "derived_waveform_sample_count": derived_count,
                    },
                    segment_rows,
                )

            lag, correlation_before, correlation_after = optimal_ihr_lag(
                ecg_peaks, ppg_peaks[:, 0], self.config
            )
            shifted_ecg = ecg_peaks + lag
            ecg_selected, ecg_segment_ids = _select_peaks_to_segments(
                shifted_ecg, ppg_segments
            )
            ppg_selected = ppg_peaks[:, 0]
            ppg_segment_ids = ppg_peaks[:, 1].astype(int)
            ecg_features, ecg_audit = legacy_hrv_features(
                ecg_selected, ecg_segment_ids, self.config
            )
            ppg_features, ppg_audit = legacy_hrv_features(
                ppg_selected, ppg_segment_ids, self.config
            )
            beat_diagnostics = _beat_diagnostics(ecg_selected, ppg_selected)
            feature_values = [*ecg_features.values(), *ppg_features.values()]
            status = "COMPUTED" if np.isfinite(feature_values).all() else "PARTIAL"
            row = {
                **base,
                **self._lineage(ecg_selection, ppg_selection),
                "status": status,
                "reason": "" if status == "COMPUTED" else "nonfinite_hrv_feature",
                "source_sampling_rate_hz": ppg_rate,
                "ecg_sampling_rate_hz": ecg_rate,
                "waveform_upsampling_method": (
                    "natural_cubic_25_to_100" if ppg_rate < 50.0 else "not_required"
                ),
                "observed_waveform_sample_count": len(ppg_values),
                "derived_waveform_sample_count": derived_count,
                "silver_interpolated_input_count": silver_interpolated,
                "total_segment_count": len(segment_rows),
                "accepted_segment_count": len(ppg_segments),
                "rejected_segment_count": len(segment_rows) - len(ppg_segments),
                "ecg_ihr_selected_lag_seconds": lag,
                "ecg_ihr_correlation_before": correlation_before,
                "ecg_ihr_correlation_after": correlation_after,
                "ecg_peak_count": len(ecg_selected),
                "ppg_peak_count": len(ppg_selected),
                **beat_diagnostics,
                **{f"ecg_{key}": value for key, value in ecg_features.items()},
                **{f"ppg_{key}": value for key, value in ppg_features.items()},
                **{f"ecg_{key}": value for key, value in ecg_audit.items()},
                **{f"ppg_{key}": value for key, value in ppg_audit.items()},
            }
            return row, segment_rows
        except Exception as exc:
            return (
                {
                    **base,
                    **self._lineage(ecg_selection, ppg_selection),
                    "status": "EXCLUDED",
                    "reason": f"{type(exc).__name__}:{exc}",
                },
                [],
            )

    def _ppg_peaks(
        self,
        participant_id: str,
        phase: str,
        channel: str,
        times: np.ndarray,
        values: np.ndarray,
        source_rate_hz: float,
    ) -> tuple[np.ndarray, list[tuple[int, float, float]], list[dict[str, Any]], int]:
        start = float(times[0])
        stop = float(times[-1])
        segment_count = int(np.floor((stop - start) / self.config.segment_seconds))
        peaks: list[tuple[float, int]] = []
        accepted: list[tuple[int, float, float]] = []
        decisions: list[dict[str, Any]] = []
        derived_total = 0
        expected = int(round(self.config.segment_seconds * source_rate_hz))
        for index in range(segment_count):
            segment_start = start + index * self.config.segment_seconds
            segment_end = segment_start + self.config.segment_seconds
            mask = (times >= segment_start) & (times < segment_end)
            segment_times = times[mask]
            segment_values = values[mask]
            reason = ""
            quality_label = "Poor"
            quality_score = 0
            segment_peaks = np.empty(0)
            derived_count = 0
            coverage = len(segment_values) / expected if expected else 0.0
            if coverage < self.config.minimum_segment_coverage:
                reason = "insufficient_segment_coverage"
            elif not np.isfinite(segment_values).all():
                reason = "nonfinite_segment"
            elif np.max(np.diff(segment_times)) > 2.5 / source_rate_hz:
                reason = "segment_gap"
            else:
                filtered = _bandpass(segment_values, source_rate_hz, self.config)
                quality_label, quality_score = _legacy_quality(
                    filtered, source_rate_hz, self.config
                )
                if quality_label in {"Good", "Moderate"}:
                    if source_rate_hz < 50.0:
                        detection_times, detection_values = cubic_upsample(
                            segment_times, filtered, self.config.target_rate_hz
                        )
                        detection_rate = self.config.target_rate_hz
                        derived_count = len(detection_values) - len(filtered)
                    else:
                        detection_times = segment_times
                        detection_values = filtered
                        detection_rate = source_rate_hz
                    local_indices = _legacy_detect_peaks(
                        detection_values, detection_rate, self.config
                    )
                    segment_peaks = detection_times[local_indices]
                    accepted.append((index, segment_start, segment_end))
                    peaks.extend((float(value), index) for value in segment_peaks)
                else:
                    reason = "legacy_sqa_poor"
            derived_total += derived_count
            decision = SegmentDecision(
                segment_index=index,
                start_seconds=segment_start,
                end_seconds=segment_end,
                sample_count=len(segment_values),
                expected_sample_count=expected,
                quality_label=quality_label,
                quality_score=quality_score,
                accepted=quality_label in {"Good", "Moderate"} and not reason,
                peak_count=len(segment_peaks),
                derived_sample_count=derived_count,
                reason=reason,
            )
            decisions.append(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "sensor_channel": channel,
                    **asdict(decision),
                    "status": "LEGACY_REPRODUCTION_ONLY",
                }
            )
        return (
            np.asarray(peaks, dtype=float).reshape(-1, 2),
            accepted,
            decisions,
            derived_total,
        )

    def _comparison(
        self,
        sessions: pd.DataFrame,
        metrics: pd.DataFrame,
        manifest_path: Path,
    ) -> dict[str, Any]:
        canonical_path = (
            self.source_root
            / "validation"
            / "hrv_agreement"
            / "hrv_agreement_metrics.csv"
        )
        canonical = (
            pd.read_csv(canonical_path) if canonical_path.is_file() else pd.DataFrame()
        )
        report_path = (
            self.project_root
            / "DATA PPG-20260609T013317Z-3-001"
            / "DATA PPG"
            / "PipelineSignal"
            / "HRV Validation"
            / "results"
            / "interpretasi_hasil.txt"
        )
        historical_pearson = _parse_historical_pearson(report_path)
        return {
            "schema_version": "legacy-estimand-reproduction-v1",
            "run_id": self.run_id,
            "source_run_id": self.source_run_id,
            "status": "LEGACY_REPRODUCTION_ONLY",
            "decision_role": "DIAGNOSTIC_ONLY",
            "canonical_status_affected": False,
            "detector_selection_eligible": False,
            "holdout_selection_eligible": False,
            "training_eligible": False,
            "onnx_eligible": False,
            "source_manifest": str(manifest_path.relative_to(self.project_root)),
            "source_manifest_sha256": _sha256(manifest_path),
            "transformations": {
                "cubic_waveform_upsampling_25_to_100": True,
                "accepted_30_second_segment_pooling": True,
                "compressed_adjacency_after_rejection": True,
                "tachogram_cubic_interpolation_extrapolation_enabled": True,
                "ecg_shift_selected_by_same_recording_ihr_correlation": True,
            },
            "transformation_audit": _transformation_audit(sessions),
            "diagnostic_beat_summary": _diagnostic_beat_summary(sessions),
            "cohorts": _cohort_summary(sessions, metrics),
            "canonical": _canonical_summary(canonical),
            "green_historical_parity": _green_historical_parity(
                metrics, canonical, historical_pearson
            ),
            "historical_report": {
                "status": "historical_unverifiable",
                "path": str(report_path.relative_to(self.project_root)),
                "sha256": _sha256(report_path) if report_path.is_file() else None,
                "parsed_pearson": historical_pearson,
            },
        }

    def _publish(
        self,
        sessions: pd.DataFrame,
        segments: pd.DataFrame,
        metrics: pd.DataFrame,
        comparison: Mapping[str, Any],
        manifest_path: Path,
    ) -> None:
        staging = self.output_root.with_name(f"{self.output_root.name}.staging")
        if staging.exists():
            raise FileExistsError(f"legacy_reproduction_staging_collision:{staging}")
        staging.mkdir(parents=True)
        sessions.to_csv(staging / "legacy_hrv_feature_pairs_240s.csv", index=False)
        sessions.to_parquet(
            staging / "legacy_hrv_feature_pairs_240s.parquet", index=False
        )
        segments.to_csv(staging / "legacy_segment_decisions_30s.csv", index=False)
        metrics.to_csv(staging / "legacy_hrv_agreement_metrics.csv", index=False)
        _write_json(staging / "legacy_run_comparison.json", comparison)
        checklist = _implementation_checklist(sessions, segments, comparison)
        _write_json(staging / "legacy_implementation_checklist.json", checklist)
        (staging / "legacy_implementation_checklist.md").write_text(
            _checklist_markdown(checklist), encoding="utf-8"
        )
        contract = {
            "schema_version": "legacy-estimand-reproduction-contract-v1",
            "run_id": self.run_id,
            "source_run_id": self.source_run_id,
            "status": "LEGACY_REPRODUCTION_ONLY",
            "decision_role": "DIAGNOSTIC_ONLY",
            "canonical_status_affected": False,
            "detector_selection_eligible": False,
            "holdout_selection_eligible": False,
            "training_eligible": False,
            "onnx_eligible": False,
            "source_manifest_sha256": _sha256(manifest_path),
            "config": asdict(self.config),
            "code_sha256": _sha256(Path(__file__)),
        }
        _write_json(staging / "legacy_reproduction_contract.json", contract)
        (staging / "legacy_reproduction_report.md").write_text(
            _markdown_report(comparison), encoding="utf-8"
        )
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, self.output_root)

    def _read_frame(self, row: pd.Series) -> pd.DataFrame:
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = self.project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"silver_frame_missing:{path}")
        return pd.read_parquet(path)

    @staticmethod
    def _select_manifest_row(
        manifest: pd.DataFrame,
        participant_id: str,
        phase: str,
        channel: str,
    ) -> pd.Series | None:
        selected = manifest[
            manifest["participant_id"].eq(participant_id)
            & manifest["phase"].eq(phase)
            & manifest["sensor_channel"].eq(channel)
        ]
        if len(selected) > 1:
            raise ValueError(
                f"lineage_ambiguous:{participant_id}:{phase}:{channel}:{len(selected)}"
            )
        return None if selected.empty else selected.iloc[0]

    def _base_row(
        self, participant_id: str, phase: str, channel: str
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_run_id": self.source_run_id,
            "participant_id": participant_id,
            "phase": phase,
            "sensor_channel": channel,
            "estimand": "quality_selected_compressed_legacy",
            "evidence_status": "LEGACY_REPRODUCTION_ONLY",
            "canonical_status_affected": False,
            "model_eligible": False,
        }

    @staticmethod
    def _lineage(ecg: pd.Series, ppg: pd.Series) -> dict[str, Any]:
        return {
            "recording_id": str(ppg.get("recording_id", "")),
            "ecg_source_file": str(ecg.get("source_file", "")),
            "ecg_source_checksum": str(ecg.get("source_checksum", "")),
            "ppg_source_file": str(ppg.get("source_file", "")),
            "ppg_source_checksum": str(ppg.get("source_checksum", "")),
        }


def _signal_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, int]:
    required = {"elapsed_seconds", "clean_value", "sampling_rate_hz"}
    if not required.issubset(frame.columns):
        raise ValueError("silver_frame_contract_mismatch")
    ordered = frame.sort_values("elapsed_seconds", kind="stable").drop_duplicates(
        "elapsed_seconds", keep="first"
    )
    times = pd.to_numeric(ordered["elapsed_seconds"], errors="coerce").to_numpy(float)
    values = pd.to_numeric(ordered["clean_value"], errors="coerce").to_numpy(float)
    finite_time = np.isfinite(times)
    times = times[finite_time]
    values = values[finite_time]
    rate_values = pd.to_numeric(ordered["sampling_rate_hz"], errors="coerce")
    rate = float(rate_values[np.isfinite(rate_values)].median())
    if not np.isfinite(rate) or rate <= 0.0:
        deltas = np.diff(times)
        rate = float(1.0 / np.median(deltas[deltas > 0.0]))
    interpolated = 0
    if "interpolated_mask" in ordered.columns:
        interpolated = int(
            ordered["interpolated_mask"].fillna(False).astype(bool).sum()
        )
    return times, values, rate, interpolated


def _bandpass(
    values: np.ndarray,
    sampling_rate_hz: float,
    config: LegacyReproductionConfig,
) -> np.ndarray:
    highcut = min(config.highcut_hz, sampling_rate_hz * 0.49)
    if highcut <= config.lowcut_hz:
        raise ValueError("legacy_bandpass_invalid_for_sampling_rate")
    sos = butter(
        config.filter_order,
        [config.lowcut_hz, highcut],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    return np.asarray(sosfiltfilt(sos, values), dtype=float)


def _legacy_detect_peaks(
    values: np.ndarray,
    sampling_rate_hz: float,
    config: LegacyReproductionConfig,
) -> np.ndarray:
    centered = values - np.median(values)
    standard_deviation = float(np.std(centered))
    if standard_deviation <= np.finfo(float).eps:
        return np.empty(0, dtype=int)
    distance = max(1, int(round(config.peak_distance_seconds * sampling_rate_hz)))
    prominence = max(
        np.finfo(float).eps, config.peak_prominence_std * standard_deviation
    )
    height = config.peak_height_std * standard_deviation
    peaks, _ = find_peaks(
        centered,
        distance=distance,
        prominence=prominence,
        height=height,
    )
    if not len(peaks):
        peaks, _ = find_peaks(centered, distance=distance, prominence=prominence * 0.3)
    margin = int(round(config.edge_margin_seconds * sampling_rate_hz))
    return peaks[(peaks >= margin) & (peaks < len(values) - margin)]


def _legacy_quality(
    values: np.ndarray,
    sampling_rate_hz: float,
    config: LegacyReproductionConfig,
) -> tuple[str, int]:
    centered = values - np.median(values)
    prominence = 0.3 * float(np.std(centered))
    peaks, _ = find_peaks(
        centered,
        distance=max(1, int(round(0.32 * sampling_rate_hz))),
        prominence=max(prominence, np.finfo(float).eps),
    )
    if len(peaks) < config.minimum_peaks_per_segment:
        return "Poor", 0
    intervals = np.diff(peaks) / sampling_rate_hz
    valid_ratio = float(np.mean((intervals >= 0.33) & (intervals <= 1.50)))
    valid = intervals[(intervals >= 0.33) & (intervals <= 1.50)]
    ibi_cv = float(np.std(valid, ddof=1) / np.mean(valid)) if len(valid) > 1 else np.inf
    amplitudes = values[peaks]
    amplitude_mean = float(np.mean(amplitudes))
    amplitude_cv = (
        float(np.std(amplitudes, ddof=1) / amplitude_mean)
        if amplitude_mean > 1e-10
        else np.inf
    )
    median_interval = float(np.median(intervals))
    outlier_ratio = float(
        np.mean(
            (intervals > 1.8 * median_interval) | (intervals < 0.6 * median_interval)
        )
    )
    template = _template_consistency(values, peaks, sampling_rate_hz)
    score = sum(
        (
            _score_high(valid_ratio, 0.90, 0.70),
            _score_high(template, 0.80, 0.60),
            _score_low(ibi_cv, 0.15, 0.30),
            _score_low(amplitude_cv, 0.30, 0.50),
            _score_low(outlier_ratio, 0.10, 0.25),
        )
    )
    return ("Good" if score >= 9 else "Moderate" if score >= 5 else "Poor"), score


def _template_consistency(
    values: np.ndarray, peaks: np.ndarray, sampling_rate_hz: float
) -> float:
    before = int(round(0.24 * sampling_rate_hz))
    after = int(round(0.48 * sampling_rate_hz))
    snippets: list[np.ndarray] = []
    for peak in peaks:
        if peak - before < 0 or peak + after + 1 > len(values):
            continue
        snippet = values[peak - before : peak + after + 1].astype(float, copy=True)
        snippet -= np.mean(snippet)
        norm = np.linalg.norm(snippet)
        if norm > 1e-10:
            snippets.append(snippet / norm)
    if len(snippets) < 2:
        return float("nan")
    template = np.mean(snippets, axis=0)
    norm = np.linalg.norm(template)
    if norm <= 1e-10:
        return float("nan")
    template /= norm
    return float(np.median([np.dot(snippet, template) for snippet in snippets]))


def _score_high(value: float, good: float, moderate: float) -> int:
    if not np.isfinite(value):
        return 0
    return 2 if value >= good else 1 if value >= moderate else 0


def _score_low(value: float, good: float, moderate: float) -> int:
    if not np.isfinite(value):
        return 0
    return 2 if value < good else 1 if value <= moderate else 0


def _finite_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [
        run
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if len(run)
    ]


def _ecg_peaks(
    times: np.ndarray, values: np.ndarray, sampling_rate_hz: float
) -> np.ndarray:
    peaks: list[float] = []
    for run in _finite_runs(np.isfinite(values)):
        if len(run) < int(round(5.0 * sampling_rate_hz)):
            continue
        _, info = nk.ecg_peaks(
            values[run], sampling_rate=sampling_rate_hz, method="neurokit"
        )
        local = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        local = local[(local >= 0) & (local < len(run))]
        peaks.extend(times[run[local]].tolist())
    return np.unique(np.asarray(peaks, dtype=float))


def _select_peaks_to_segments(
    peaks: Sequence[float], segments: Sequence[tuple[int, float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[float] = []
    segment_ids: list[int] = []
    for segment_id, start, end in segments:
        for peak in np.asarray(peaks, dtype=float):
            if start <= peak < end:
                selected.append(float(peak))
                segment_ids.append(segment_id)
    return np.asarray(selected), np.asarray(segment_ids, dtype=int)


def _beat_diagnostics(
    ecg_peaks: Sequence[float], ppg_peaks: Sequence[float]
) -> dict[str, float | int]:
    """Fixed-PAT one-to-one diagnostic; never used by the legacy HRV score."""

    ecg = np.sort(np.asarray(ecg_peaks, dtype=float))
    ppg = np.sort(np.asarray(ppg_peaks, dtype=float))
    ecg_index = 0
    ppg_index = 0
    matches: list[tuple[int, int]] = []
    while ecg_index < len(ecg) and ppg_index < len(ppg):
        delay = ppg[ppg_index] - ecg[ecg_index]
        if delay < 0.1:
            ppg_index += 1
        elif delay > 0.4:
            ecg_index += 1
        else:
            matches.append((ecg_index, ppg_index))
            ecg_index += 1
            ppg_index += 1
    precision = len(matches) / len(ppg) if len(ppg) else 0.0
    recall = len(matches) / len(ecg) if len(ecg) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    ibi_errors: list[float] = []
    for left, right in zip(matches[:-1], matches[1:], strict=True):
        if right[0] - left[0] == 1 and right[1] - left[1] == 1:
            ecg_interval = (ecg[right[0]] - ecg[left[0]]) * 1000.0
            ppg_interval = (ppg[right[1]] - ppg[left[1]]) * 1000.0
            ibi_errors.append(abs(ppg_interval - ecg_interval))
    return {
        "diagnostic_matched_peak_count": len(matches),
        "diagnostic_beat_precision": precision,
        "diagnostic_beat_recall": recall,
        "diagnostic_beat_f1": f1,
        "diagnostic_adjacent_ibi_pair_count": len(ibi_errors),
        "diagnostic_ibi_mae_ms": float(np.mean(ibi_errors)) if ibi_errors else np.nan,
    }


def _filter_rr_outliers(
    rr_ms: np.ndarray,
    midpoints: np.ndarray,
    config: LegacyReproductionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (rr_ms >= config.ibi_min_ms) & (rr_ms <= config.ibi_max_ms)
    rr = rr_ms[valid]
    times = midpoints[valid]
    if not len(rr):
        return rr, times
    moving = np.ones(len(rr), dtype=bool)
    for index in range(len(rr)):
        local = rr[max(0, index - 2) : min(len(rr), index + 3)]
        median = float(np.median(local))
        moving[index] = median > 0.0 and abs(rr[index] - median) / median <= 0.20
    rr = rr[moving]
    times = times[moving]
    if not len(rr):
        return rr, times
    median = float(np.median(rr))
    mad = float(np.median(np.abs(rr - median)) * 1.4826)
    keep = np.abs(rr - median) <= 3.0 * mad if mad > 0.0 else np.ones(len(rr), bool)
    return rr[keep], times[keep]


def _frequency_features(
    rr_ms: np.ndarray,
    midpoints: np.ndarray,
    config: LegacyReproductionConfig,
) -> dict[str, float]:
    unavailable = {
        "lf_power_ms2": float("nan"),
        "hf_power_ms2": float("nan"),
        "lf_hf_ratio": float("nan"),
    }
    if len(rr_ms) < 4 or midpoints[-1] - midpoints[0] < 5.0:
        return unavailable
    count = int((midpoints[-1] - midpoints[0]) * config.tachogram_rate_hz)
    if count < 16:
        return unavailable
    grid = np.linspace(midpoints[0], midpoints[-1], count)
    kind = "cubic" if len(rr_ms) >= 4 else "linear"
    interpolator = interp1d(
        midpoints,
        rr_ms,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )
    tachogram = np.asarray(interpolator(grid), dtype=float)
    nperseg = max(16, min(256, len(tachogram) // 2))
    frequencies, spectrum = welch(
        tachogram,
        fs=config.tachogram_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=min(128, nperseg - 1),
    )
    lf = _band_power(frequencies, spectrum, 0.04, 0.15)
    hf = _band_power(frequencies, spectrum, 0.15, 0.40)
    return {
        "lf_power_ms2": lf,
        "hf_power_ms2": hf,
        "lf_hf_ratio": lf / hf if hf > 0.0 else float("nan"),
    }


def _band_power(
    frequencies: np.ndarray, spectrum: np.ndarray, low: float, high: float
) -> float:
    selected = (frequencies >= low) & (frequencies <= high)
    if selected.sum() < 2:
        return 0.0
    return float(np.trapezoid(spectrum[selected], frequencies[selected]))


def _instantaneous_hr(peak_times: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.sort(np.asarray(peak_times, dtype=float))
    intervals = np.diff(peaks)
    times = (peaks[:-1] + peaks[1:]) / 2.0
    heart_rate = 60.0 / intervals
    valid = np.isfinite(heart_rate) & (heart_rate > 40.0) & (heart_rate < 200.0)
    times = times[valid]
    heart_rate = heart_rate[valid]
    if len(heart_rate) < 3:
        return times, heart_rate
    median = float(np.median(heart_rate))
    mad = float(np.median(np.abs(heart_rate - median)) * 1.4826)
    keep = (
        np.abs(heart_rate - median) < 3.0 * mad
        if mad > 0.0
        else np.ones(len(heart_rate), bool)
    )
    return times[keep], heart_rate[keep]


def _ihr_correlation(
    ecg_times: np.ndarray,
    ecg_ihr: np.ndarray,
    ppg_times: np.ndarray,
    ppg_ihr: np.ndarray,
    lag_seconds: float,
) -> float:
    shifted = ecg_times + lag_seconds
    start = max(float(shifted[0]), float(ppg_times[0]))
    end = min(float(shifted[-1]), float(ppg_times[-1]))
    if end - start < 20.0:
        return float("nan")
    grid = np.arange(start, end, 0.5)
    if len(grid) < 10:
        return float("nan")
    ecg = np.interp(grid, shifted, ecg_ihr)
    ppg = np.interp(grid, ppg_times, ppg_ihr)
    if np.std(ecg) <= np.finfo(float).eps or np.std(ppg) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(ecg, ppg)[0, 1])


def _icc_a1(left: np.ndarray, right: np.ndarray) -> float:
    values = np.column_stack([left, right])
    subject_count, rater_count = values.shape
    if subject_count < 3:
        return float("nan")
    grand = float(values.mean())
    subject_mean = values.mean(axis=1)
    rater_mean = values.mean(axis=0)
    ms_subject = float(
        rater_count * np.square(subject_mean - grand).sum() / (subject_count - 1)
    )
    ms_rater = float(
        subject_count * np.square(rater_mean - grand).sum() / (rater_count - 1)
    )
    residual = values - subject_mean[:, None] - rater_mean[None, :] + grand
    ms_error = float(
        np.square(residual).sum() / ((subject_count - 1) * (rater_count - 1))
    )
    denominator = (
        ms_subject
        + (rater_count - 1) * ms_error
        + rater_count * (ms_rater - ms_error) / subject_count
    )
    return float((ms_subject - ms_error) / denominator) if denominator else float("nan")


def _agreement_metrics(sessions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, cohort_frame in (
        ("legacy_31", sessions[~sessions["participant_id"].eq("ID-32")]),
        ("extended_32", sessions),
    ):
        for channel in CHANNELS:
            channel_frame = cohort_frame[cohort_frame["sensor_channel"].eq(channel)]
            for scope in ("pooled", *PHASES):
                scoped = (
                    channel_frame
                    if scope == "pooled"
                    else channel_frame[channel_frame["phase"].eq(scope)]
                )
                for feature in FEATURES:
                    selected = scoped[
                        np.isfinite(
                            pd.to_numeric(scoped[f"ecg_{feature}"], errors="coerce")
                        )
                        & np.isfinite(
                            pd.to_numeric(scoped[f"ppg_{feature}"], errors="coerce")
                        )
                    ]
                    rows.append(
                        _agreement_row(selected, cohort, channel, scope, feature)
                    )
    return pd.DataFrame(rows)


def _agreement_row(
    frame: pd.DataFrame, cohort: str, channel: str, scope: str, feature: str
) -> dict[str, Any]:
    base = {
        "cohort": cohort,
        "sensor_channel": channel,
        "scope": scope,
        "feature": feature,
        "sample_count": len(frame),
        "evidence_status": "LEGACY_REPRODUCTION_ONLY",
        "canonical_gate_eligible": False,
    }
    if len(frame) < 3:
        return {**base, "status": "INSUFFICIENT_EVIDENCE"}
    left = frame[f"ecg_{feature}"].to_numpy(float)
    right = frame[f"ppg_{feature}"].to_numpy(float)
    ecg_range = float(np.ptp(left))
    if ecg_range <= 0.0 or np.ptp(right) <= 0.0:
        return {**base, "status": "INSUFFICIENT_EVIDENCE"}
    pearson = float(pearsonr(left, right).statistic)
    spearman = float(spearmanr(left, right).statistic)
    icc = _icc_a1(left, right)
    mae = float(np.mean(np.abs(right - left)))
    accuracy_component = float(1.0 - np.clip(mae / ecg_range, 0.0, 1.0))
    components = np.asarray([pearson, spearman, icc, accuracy_component])
    return {
        **base,
        "status": "COMPUTED"
        if np.isfinite(components).all()
        else "INSUFFICIENT_EVIDENCE",
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "icc_a1": icc,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(np.square(right - left)))),
        "bias": float(np.mean(right - left)),
        "ecg_range": ecg_range,
        "accuracy_component": accuracy_component,
        "composite_score": float(np.mean(components))
        if np.isfinite(components).all()
        else np.nan,
    }


def _cohort_summary(sessions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, selected in (
        ("legacy_31", sessions[~sessions["participant_id"].eq("ID-32")]),
        ("extended_32", sessions),
    ):
        channel_summary: dict[str, Any] = {}
        for channel in CHANNELS:
            channel_sessions = selected[selected["sensor_channel"].eq(channel)]
            pooled = metrics[
                metrics["cohort"].eq(cohort)
                & metrics["sensor_channel"].eq(channel)
                & metrics["scope"].eq("pooled")
            ]
            channel_summary[channel] = {
                "expected_session_count": len(channel_sessions),
                "computed_session_count": int(
                    channel_sessions["status"].eq("COMPUTED").sum()
                ),
                "partial_session_count": int(
                    channel_sessions["status"].eq("PARTIAL").sum()
                ),
                "excluded_session_count": int(
                    channel_sessions["status"].eq("EXCLUDED").sum()
                ),
                "pooled_metrics": pooled.to_dict("records"),
            }
        result[cohort] = channel_summary
    return result


def _canonical_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"status": "UNAVAILABLE"}
    columns = [
        column
        for column in (
            "sensor_channel",
            "scope",
            "feature",
            "sample_count",
            "status",
            "pearson_r",
            "spearman_rho",
            "icc_a1",
            "mae",
            "composite_score",
        )
        if column in frame.columns
    ]
    return {"status": "AVAILABLE", "metrics": frame[columns].to_dict("records")}


def _green_historical_parity(
    metrics: pd.DataFrame,
    canonical: pd.DataFrame,
    historical: Mapping[str, float],
) -> list[dict[str, Any]]:
    legacy_31 = metrics[
        metrics["cohort"].eq("legacy_31")
        & metrics["sensor_channel"].eq("ppg_green")
        & metrics["scope"].eq("pooled")
    ]
    extended_32 = metrics[
        metrics["cohort"].eq("extended_32")
        & metrics["sensor_channel"].eq("ppg_green")
        & metrics["scope"].eq("pooled")
    ]
    canonical_green = canonical[
        canonical.get("sensor_channel", pd.Series(dtype=str)).eq("ppg_green")
        & canonical.get("scope", pd.Series(dtype=str)).eq("pooled")
    ]
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        canonical_feature = "pnn50" if feature == "pnn50_pct" else feature
        historic_value = historical.get(feature)
        legacy_row = legacy_31[legacy_31["feature"].eq(feature)]
        extended_row = extended_32[extended_32["feature"].eq(feature)]
        canonical_row = canonical_green[
            canonical_green.get("feature", pd.Series(dtype=str)).eq(canonical_feature)
        ]
        legacy_value = _first_number(legacy_row, "pearson_r")
        rows.append(
            {
                "feature": feature,
                "historical_report_pearson": historic_value,
                "reproduction_31_pearson": legacy_value,
                "reproduction_31_delta": (
                    legacy_value - historic_value
                    if legacy_value is not None and historic_value is not None
                    else None
                ),
                "extension_32_pearson": _first_number(extended_row, "pearson_r"),
                "canonical_pearson": _first_number(canonical_row, "pearson_r"),
                "historical_status": "historical_unverifiable",
                "reproduction_status": "LEGACY_REPRODUCTION_ONLY",
            }
        )
    return rows


def _first_number(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = pd.to_numeric(frame.iloc[0][column], errors="coerce")
    return float(value) if np.isfinite(value) else None


def _transformation_audit(sessions: pd.DataFrame) -> dict[str, Any]:
    computed = sessions[sessions["status"].isin(["COMPUTED", "PARTIAL"])]
    return {
        "session_rows": len(sessions),
        "computed_or_partial_rows": len(computed),
        "derived_waveform_sample_count": int(
            pd.to_numeric(
                sessions.get("derived_waveform_sample_count", 0), errors="coerce"
            )
            .fillna(0)
            .sum()
        ),
        "accepted_segment_count": int(
            pd.to_numeric(sessions.get("accepted_segment_count", 0), errors="coerce")
            .fillna(0)
            .sum()
        ),
        "rejected_segment_count": int(
            pd.to_numeric(sessions.get("rejected_segment_count", 0), errors="coerce")
            .fillna(0)
            .sum()
        ),
        "ecg_bridged_rejected_segment_count": int(
            pd.to_numeric(
                sessions.get("ecg_bridged_rejected_segment_count", 0), errors="coerce"
            )
            .fillna(0)
            .sum()
        ),
        "ppg_bridged_rejected_segment_count": int(
            pd.to_numeric(
                sessions.get("ppg_bridged_rejected_segment_count", 0), errors="coerce"
            )
            .fillna(0)
            .sum()
        ),
        "tachogram_extrapolation_enabled": True,
        "actual_extrapolated_tachogram_point_count": 0,
        "median_ihr_correlation_before": _series_median(
            sessions, "ecg_ihr_correlation_before"
        ),
        "median_ihr_correlation_after": _series_median(
            sessions, "ecg_ihr_correlation_after"
        ),
    }


def _diagnostic_beat_summary(sessions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        selected = sessions[
            sessions["sensor_channel"].eq(channel)
            & sessions["status"].isin(["COMPUTED", "PARTIAL"])
        ]
        summary[channel] = {
            "session_count": len(selected),
            "median_beat_f1": _series_median(selected, "diagnostic_beat_f1"),
            "median_ibi_mae_ms": _series_median(selected, "diagnostic_ibi_mae_ms"),
            "median_ecg_peak_count": _series_median(selected, "ecg_peak_count"),
            "median_ppg_peak_count": _series_median(selected, "ppg_peak_count"),
            "median_selected_lag_seconds": _series_median(
                selected, "ecg_ihr_selected_lag_seconds"
            ),
        }
    return summary


def _series_median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else None


def _implementation_checklist(
    sessions: pd.DataFrame,
    segments: pd.DataFrame,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "exact_192_session_rows": len(sessions) == 192,
        "only_id_01_to_id_32": set(sessions["participant_id"]) == set(PARTICIPANTS),
        "only_relax_and_task": set(sessions["phase"]) == set(PHASES),
        "three_independent_ppg_channels": set(sessions["sensor_channel"])
        == set(CHANNELS),
        "cubic_green_25_to_100_implemented": bool(
            sessions["waveform_upsampling_method"]
            .fillna("")
            .eq("natural_cubic_25_to_100")
            .any()
        ),
        "thirty_second_sqa_pooling_implemented": len(segments) > 0,
        "compressed_adjacency_audited": "ppg_bridged_rejected_segment_count"
        in sessions.columns,
        "tachogram_extrapolation_enabled_and_actual_count_separated": comparison[
            "transformation_audit"
        ]["actual_extrapolated_tachogram_point_count"]
        == 0,
        "same_recording_ihr_shift_audited": "ecg_ihr_selected_lag_seconds"
        in sessions.columns,
        "canonical_status_not_affected": not sessions["canonical_status_affected"]
        .astype(bool)
        .any(),
        "not_training_eligible": not sessions["model_eligible"].astype(bool).any(),
        "historical_report_marked_unverifiable": comparison["historical_report"][
            "status"
        ]
        == "historical_unverifiable",
    }
    return {
        "schema_version": "legacy-reproduction-checklist-v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "pipeline_signal_method_parity_scope": "ppg_green_only_on_current_240s_silver",
        "derived_channel_ablation_scope": ["ppg_ir", "ppg_red"],
    }


def _checklist_markdown(checklist: Mapping[str, Any]) -> str:
    lines = [
        "# Legacy reproduction implementation checklist",
        "",
        f"Overall: `{checklist['status']}`",
        "",
    ]
    for name, passed in checklist["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "PipelineSignal method parity on current 240 s Silver applies only to Green. "
            "IR and Red are explicit legacy-derived channel ablations; the historical "
            "five-minute source tables remain unavailable.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_historical_pearson(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    section = text.split("4.3 Analisis Kesalahan", maxsplit=1)[0]
    parsed: dict[str, float] = {}
    for label, feature in LEGACY_REPORT_FEATURES.items():
        match = re.search(
            rf"{re.escape(label)}:\s*\n\s*Pearson r = ([+-]?\d+(?:\.\d+)?)",
            section,
        )
        if match:
            parsed[feature] = float(match.group(1))
    return parsed


def _markdown_report(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Legacy PipelineSignal estimand reproduction",
        "",
        "Status: `LEGACY_REPRODUCTION_ONLY` / `NOT_CANONICAL` / `NOT_MODEL_ELIGIBLE`.",
        "",
        "Lima transformasi historis dijalankan pada Silver aktual. Hasil tidak mengubah ",
        "GO/NO_GO canonical, Gold, holdout, training, atau ONNX.",
        "",
        "## Coverage",
        "",
        "| Cohort | Channel | Computed | Partial | Excluded |",
        "|---|---|---:|---:|---:|",
    ]
    for cohort, channels in comparison["cohorts"].items():
        for channel, summary in channels.items():
            lines.append(
                f"| {cohort} | {channel} | {summary['computed_session_count']} | "
                f"{summary['partial_session_count']} | {summary['excluded_session_count']} |"
            )
    lines.extend(
        [
            "",
            "## Green Pearson parity",
            "",
            "| Feature | Historical 31 | Reproduction 31 | Delta | Extension 32 | Canonical |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["green_historical_parity"]:
        lines.append(
            "| {feature} | {historical_report_pearson} | {reproduction_31_pearson} | "
            "{reproduction_31_delta} | {extension_32_pearson} | {canonical_pearson} |".format(
                **{key: _display_number(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Extended-32 pooled HRV agreement",
            "",
            "| Channel | Feature | Pearson | Composite |",
            "|---|---|---:|---:|",
        ]
    )
    for channel, summary in comparison["cohorts"]["extended_32"].items():
        for row in summary["pooled_metrics"]:
            lines.append(
                f"| {channel} | {row['feature']} | "
                f"{_display_number(row.get('pearson_r'))} | "
                f"{_display_number(row.get('composite_score'))} |"
            )
    lines.extend(
        [
            "",
            "## Beat diagnostic (not a decision gate)",
            "",
            "| Channel | Sessions | Median F1 | Median IBI MAE ms | Median ECG/PPG peaks | Median selected lag s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for channel, summary in comparison["diagnostic_beat_summary"].items():
        lines.append(
            f"| {channel} | {summary['session_count']} | "
            f"{_display_number(summary['median_beat_f1'])} | "
            f"{_display_number(summary['median_ibi_mae_ms'])} | "
            f"{_display_number(summary['median_ecg_peak_count'])}/"
            f"{_display_number(summary['median_ppg_peak_count'])} | "
            f"{_display_number(summary['median_selected_lag_seconds'])} |"
        )
    audit = comparison["transformation_audit"]
    lines.extend(
        [
            "",
            "## Transformation audit",
            "",
            f"- Derived waveform samples: {audit['derived_waveform_sample_count']}",
            f"- Accepted/rejected 30 s segments: {audit['accepted_segment_count']} / "
            f"{audit['rejected_segment_count']}",
            f"- Median IHR correlation before/after lag selection: "
            f"{_display_number(audit['median_ihr_correlation_before'])} / "
            f"{_display_number(audit['median_ihr_correlation_after'])}",
            f"- Tachogram extrapolation enabled; actual out-of-support points: "
            f"{audit['actual_extrapolated_tachogram_point_count']}",
            "",
            "## Interpretation boundary",
            "",
            "Cubic upsampling adds derived samples, accepted-segment pooling changes time support, "
            "compressed adjacency bridges rejected blocks, tachogram interpolation bridges hidden "
            "gaps, and same-recording IHR lag selection optimizes alignment on evaluated data. "
            "Higher correlation therefore demonstrates estimand sensitivity, not detector truth.",
        ]
    )
    return "\n".join(lines) + "\n"


def _display_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}" if np.isfinite(number) else "NA"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
