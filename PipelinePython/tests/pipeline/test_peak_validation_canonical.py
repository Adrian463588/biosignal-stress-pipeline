from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.peak_detection import PpgPeakDetector
from PipelinePython.common.peak_validation import (
    DetectedPeaks,
    GapEvent,
    PpgEcgValidator,
    PreparedSignal,
    apply_morphology_status_policy,
    assert_canonical_validation_keyset,
    evaluate_morphology_policy_ablation,
)
from PipelinePython.common.settings import PipelinePaths, load_pipeline_config


def _paths(root: Path) -> PipelinePaths:
    artifacts = root / "data" / "artifacts"
    return PipelinePaths(
        project_root=root,
        raw_data_root=root / "data" / "raw",
        anchor_csv_path=root / "anchors.csv",
        artifacts_root=artifacts,
        bronze_dir=root / "data" / "bronze",
        silver_dir=root / "data" / "silver",
        gold_dir=root / "data" / "gold",
        gold_features_path=root / "data" / "gold" / "features.parquet",
        eda_dir=artifacts / "eda",
        validation_dir=artifacts / "validation",
        models_dir=artifacts / "models",
        export_dir=artifacts / "export",
    )


def test_validation_keyset_rejects_duplicate_and_missing_combination() -> None:
    rows = [
        {
            "participant_id": f"ID-{participant:02d}",
            "phase": phase,
            "sensor_channel": channel,
        }
        for participant in range(1, 33)
        for phase in ("relax", "task")
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    ]
    valid = pd.DataFrame(rows)
    assert_canonical_validation_keyset(valid)

    invalid = valid.copy()
    invalid.loc[191, ["participant_id", "phase", "sensor_channel"]] = invalid.loc[
        0, ["participant_id", "phase", "sensor_channel"]
    ]
    with pytest.raises(RuntimeError, match="duplicate_keys"):
        assert_canonical_validation_keyset(invalid)


def test_holdout_requires_checksum_bound_frozen_detector_policy(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    loaded = load_pipeline_config()
    config = replace(
        loaded,
        artifacts={**loaded.artifacts, "run_id": "TEST-RUN"},
    )
    config_fingerprint = hashlib.sha256(
        json.dumps(
            asdict(config),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    paths.artifacts_root.mkdir(parents=True)
    contract_path = paths.artifacts_root / "run_contract_fingerprint.json"
    contract_path.write_text(
        json.dumps(
            {
                "run_id": "TEST-RUN",
                "source_tree_sha256": "source-sha",
                "config_fingerprint": config_fingerprint,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    selection_dir = paths.validation_dir / "detector_selection"
    selection_dir.mkdir(parents=True)
    selected = {
        channel: {
            "candidate_method": config.peak_detection.selected_method_for(channel),
            "prominence_mad_multiplier": config.peak_detection.for_channel(
                channel
            ).prominence_mad_multiplier,
        }
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    }
    selection_path = selection_dir / "detector_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "run_id": "TEST-RUN",
                "source_tree_sha256": "source-sha",
                "config_fingerprint": config_fingerprint,
                "run_contract_sha256": contract_sha,
                "frozen_config_verified": True,
                "selection_contract_frozen": True,
                "selection_evidence_sufficient": False,
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )

    validator = PpgEcgValidator(paths, config)

    assert validator._frozen_detector_policy_verified()
    assert not validator._full_cohort_contracts_verified()
    paths.models_dir.mkdir(parents=True)
    report_root = tmp_path / "reports" / "model_eval" / config.model_version
    report_root.mkdir(parents=True)
    channels = {}
    for channel in ("ppg_green", "ppg_ir", "ppg_red"):
        report_path = report_root / f"{channel}_full.json"
        report_path.write_text(
            json.dumps(
                {
                    "run_id": "TEST-RUN",
                    "source_tree_sha256": "source-sha",
                    "config_fingerprint": config_fingerprint,
                    "release_status": "blocked_insufficient_ppg_coverage",
                    "requires_dl": False,
                }
            ),
            encoding="utf-8",
        )
        channels[channel] = {
            "terminal_status": "COVERAGE_BLOCKED",
            "training_admission_status": ("blocked_insufficient_development_coverage"),
            "selected_model": None,
            "release_status": "blocked_insufficient_ppg_coverage",
            "requires_dl": False,
            "holdout_evaluated": False,
            "model_contract_fingerprint": None,
            "report_path": str(report_path.resolve()),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    stress_payload = {
        "run_id": "TEST-RUN",
        "source_tree_sha256": "source-sha",
        "config_fingerprint": config_fingerprint,
        "run_contract_sha256": contract_sha,
        "selection_contract_frozen": True,
        "full_cohort_physiology_may_open": True,
        "selection_participants": list(config.training.development_participant_ids),
        "channels": channels,
    }
    stress_payload["selection_fingerprint"] = hashlib.sha256(
        json.dumps(
            stress_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (paths.models_dir / "stress_selection_contract.json").write_text(
        json.dumps(stress_payload),
        encoding="utf-8",
    )
    assert validator._full_cohort_contracts_verified()
    first_report = report_root / "ppg_green_full.json"
    first_report.write_text("{}", encoding="utf-8")
    assert not validator._full_cohort_contracts_verified()
    tampered = json.loads(selection_path.read_text(encoding="utf-8"))
    tampered["config_fingerprint"] = "wrong"
    selection_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert not validator._frozen_detector_policy_verified()


def test_pat_matching_uses_physiological_delay_without_shifting_ppg(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.arange(0.0, 10_000.0, 1_000.0)
    ppg = ecg + 220.0

    matches, pat = validator._match_with_pat(ecg, ppg)

    assert pat == 220.0
    assert len(matches) == len(ecg)
    assert np.array_equal(ppg, ecg + 220.0)
    metrics = validator._metrics(matches, ecg, ppg)
    assert metrics["beat_f1"] == 1.0
    assert metrics["ibi_mae_ms"] == 0.0
    assert metrics["ibi_rmse_ms"] == 0.0


def test_ppg_source_prefers_silver_and_only_falls_back_when_absent(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    silver_path = paths.silver_dir / "ID-01" / "relax" / "ppg_ir.parquet"
    silver_path.parent.mkdir(parents=True)
    silver_path.touch()
    sync_session = tmp_path / "data" / "sync" / "ID-01" / "relax"
    sync_session.mkdir(parents=True)
    sync_path = sync_session / "ppg_ir.parquet"
    sync_path.touch()

    validator = PpgEcgValidator(paths)

    assert validator._preferred_ppg_path("ID-01", "relax", "ppg_ir") == silver_path

    silver_path.unlink()
    assert validator._preferred_ppg_path("ID-01", "relax", "ppg_ir") is None


def test_canonical_ppg_peak_detection_does_not_clean_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = np.arange(0.0, 20_000.0, 10.0)
    values = np.sin(2.0 * np.pi * timestamps / 1_000.0)

    def fail_if_called(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("canonical Silver PPG must not be cleaned twice")

    monkeypatch.setattr(
        "PipelinePython.common.peak_validation.nk.ppg_clean", fail_if_called
    )
    monkeypatch.setattr(
        "PipelinePython.common.peak_validation.nk.ppg_findpeaks",
        lambda signal, sampling_rate, method: {
            "PPG_Peaks": np.arange(100, len(signal), 100)
        },
    )

    detected = validator._detect_ppg(timestamps, values, 100.0)

    assert detected.method == "neurokit2_ppg_elgendi"
    assert np.array_equal(detected.signal, values)


def test_ecg_reference_marks_consensus_when_both_detectors_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = np.arange(0.0, 20_000.0, 10.0)
    values = np.sin(2.0 * np.pi * timestamps / 1_000.0)
    expected = np.arange(100, len(values), 100, dtype=np.int64)

    def agree(signal: np.ndarray, sampling_rate: int, method: str):
        del sampling_rate, method
        return signal, {"ECG_R_Peaks": expected}

    monkeypatch.setattr("PipelinePython.common.peak_validation.nk.ecg_peaks", agree)

    detected = validator._detect_ecg(
        PreparedSignal(
            timestamps,
            values,
            100.0,
            np.zeros(values.shape, dtype=bool),
            np.zeros(values.shape, dtype=bool),
            (),
            False,
            False,
        )
    )

    assert detected.reference_status == "CONSENSUS"
    assert np.array_equal(detected.sample_indices, expected)


def test_short_gap_fragment_is_skipped_without_excluding_full_ppg_window(
    tmp_path: Path,
) -> None:
    config = load_pipeline_config()
    detector = PpgPeakDetector(config.peak_detection)
    validator = PpgEcgValidator(_paths(tmp_path), config, ppg_peak_detector=detector)
    timestamps = np.arange(0.0, 20_000.0, 10.0)
    values = np.sin(2.0 * np.pi * timestamps / 1_000.0)
    gap_mask = np.zeros(values.shape, dtype=bool)
    gap_mask[300:305] = True
    prepared = PreparedSignal(
        timestamps,
        values,
        100.0,
        gap_mask,
        np.zeros(values.shape, dtype=bool),
        (),
        False,
        False,
    )

    detected = validator._detect_ppg(prepared, channel="ppg_green")

    assert len(detected.sample_indices) >= 15
    assert "segment_skipped_too_short" in detected.fallback_reason


def test_pat_matching_is_one_to_one_and_rejects_outside_physiology(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.asarray([0.0, 1_000.0, 2_000.0])
    ppg = np.asarray([50.0, 220.0, 1_220.0, 2_220.0, 2_500.0])

    matches, pat = validator._match_with_pat(ecg, ppg)

    assert pat == 220.0
    assert len(matches) == 3
    assert len({ppg_index for _, ppg_index, _ in matches}) == len(matches)
    assert all(100.0 <= delay <= 400.0 for _, _, delay in matches)


def test_pat_tolerance_is_loaded_from_synchronization_config(tmp_path: Path) -> None:
    config = load_pipeline_config()
    config = replace(
        config,
        synchronization=replace(config.synchronization, pat_tolerance_ms=15.0),
    )
    validator = PpgEcgValidator(_paths(tmp_path), config)
    ecg = np.asarray([0.0, 1_000.0])
    ppg = np.asarray([200.0, 1_240.0])

    matches, pat = validator._match_with_pat(ecg, ppg)

    assert pat == 220.0
    assert matches == []


def test_missed_beat_does_not_hide_ibi_error(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.asarray([0.0, 1_000.0, 2_000.0, 3_000.0])
    ppg = np.asarray([220.0, 2_220.0, 3_220.0])
    matches = [(0, 0, 220.0), (2, 1, 220.0), (3, 2, 220.0)]

    metrics = validator._metrics(matches, ecg, ppg)

    assert metrics["beat_f1"] < 1.0
    assert metrics["ibi_pairs"] == 1
    assert metrics["ibi_mae_ms"] == 0.0
    assert metrics["ibi_rmse_ms"] == 0.0
    assert metrics["ci_status"] == "insufficient_pairs"
    assert metrics["ci_iterations"] == 0


def test_disjoint_adjacent_match_blocks_never_create_cross_block_ibi(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.arange(0.0, 6_000.0, 1_000.0)
    ppg = np.asarray([220.0, 1_220.0, 2_220.0, 3_220.0, 4_420.0, 5_420.0])
    matches = [
        (0, 0, 220.0),
        (1, 1, 220.0),
        (4, 4, 420.0),
        (5, 5, 420.0),
    ]

    metrics = validator._metrics(matches, ecg, ppg)

    assert metrics["ibi_pairs"] == 2
    assert metrics["ibi_mae_ms"] == 0.0
    assert metrics["ibi_rmse_ms"] == 0.0


def test_gap_between_matched_beats_is_not_used_for_ibi_metrics(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.asarray([0.0, 1_000.0, 2_000.0, 3_000.0])
    ppg = ecg + 220.0
    matches = [(index, index, 220.0) for index in range(4)]
    gap = GapEvent(
        start_ms=900.0,
        end_ms=1_900.0,
        duration_ms=1_000.0,
        sample_count=100,
        gap_type="long_dropout",
        action="mask_and_exclude_affected_beats",
        affects_peak_region=True,
    )

    metrics = validator._metrics(matches, ecg, ppg, ecg_gap_events=(gap,))

    assert metrics["ibi_pairs"] == 1
    assert metrics["ibi_mae_ms"] == 0.0


def test_nonphysiological_intervals_are_excluded_from_primary_ibi_metrics(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.asarray([0.0, 100.0, 1_100.0])
    ppg = ecg + 220.0
    matches = [(index, index, 220.0) for index in range(3)]

    metrics = validator._metrics(matches, ecg, ppg)

    assert metrics["ibi_pairs"] == 1
    assert metrics["ibi_mae_ms"] == 0.0


def test_bootstrap_confidence_intervals_are_seeded_and_reproducible(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.arange(0.0, 31_000.0, 1_000.0)
    ppg = ecg + 220.0 + np.resize(np.asarray([0.0, 10.0, -10.0]), ecg.size)
    matches = [
        (index, index, float(ppg[index] - ecg[index])) for index in range(len(ecg))
    ]

    first = validator._metrics(matches, ecg, ppg)
    second = validator._metrics(matches, ecg, ppg)

    assert first["ci_status"] == "ok"
    assert first["ci_iterations"] == 2_000
    for metric in (
        "beat_f1",
        "ibi_mae_ms",
        "ibi_rmse_ms",
        "hr_mae_bpm",
        "bland_altman_bias_ms",
    ):
        assert first[f"{metric}_ci_low"] == second[f"{metric}_ci_low"]
        assert first[f"{metric}_ci_high"] == second[f"{metric}_ci_high"]


def test_ibi_rmse_reports_root_mean_square_error(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    ecg = np.asarray([0.0, 1_000.0, 2_000.0, 3_000.0])
    ppg = np.asarray([220.0, 1_220.0, 2_320.0, 3_320.0])
    matches = [(index, index, 220.0) for index in range(4)]

    metrics = validator._metrics(matches, ecg, ppg)

    assert metrics["ibi_mae_ms"] == pytest.approx(100.0 / 3.0)
    assert metrics["ibi_rmse_ms"] == pytest.approx((10_000.0 / 3.0) ** 0.5)


def test_canonical_ppg_gap_is_not_interpolated_again(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = pd.date_range("2026-01-01", periods=2_000, freq="10ms", tz="UTC")
    values = np.sin(np.arange(2_000, dtype=float) / 10.0)
    values[500:503] = np.interp(np.arange(500, 503), [499, 503], values[[499, 503]])
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": timestamps,
            "clean_value": values,
            "interpolated_mask": np.r_[
                np.zeros(500, dtype=bool),
                np.ones(3, dtype=bool),
                np.zeros(1_497, dtype=bool),
            ],
            "sampling_rate_hz": 100.0,
        }
    )

    prepared = validator._prepare_frame(frame, "ppg_green")

    assert not prepared.long_gap_detected
    assert prepared.short_gap_interpolated
    assert np.isfinite(prepared.values).all()
    assert int(prepared.interpolated_mask.sum()) == 3


def test_timestamp_jitter_with_contiguous_raw_indices_is_not_a_gap(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    offsets_ms = np.asarray([0, 10, 20, 30, 70, 80, 90, 100], dtype=float)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": start + pd.to_timedelta(offsets_ms, unit="ms"),
            "raw_sample_index": np.arange(len(offsets_ms)),
            "clean_value": np.sin(np.arange(len(offsets_ms), dtype=float)),
            "sampling_rate_hz": 100.0,
        }
    )

    prepared = validator._prepare_frame(frame, "ppg_ir")

    assert prepared.timestamp_jitter_count == 1
    assert prepared.raw_index_gap_count == 0
    assert not prepared.gap_mask.any()
    assert not prepared.gap_events
    assert validator._detection_segments(prepared) == [(0, len(frame))]


def test_raw_index_jump_creates_a_true_gap(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    offsets_ms = np.asarray([0, 10, 20, 30, 70, 80, 90, 100], dtype=float)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": start + pd.to_timedelta(offsets_ms, unit="ms"),
            "raw_sample_index": np.asarray([0, 1, 2, 3, 7, 8, 9, 10]),
            "clean_value": np.sin(np.arange(len(offsets_ms), dtype=float)),
            "sampling_rate_hz": 100.0,
        }
    )

    prepared = validator._prepare_frame(frame, "ppg_ir")

    assert prepared.timestamp_jitter_count == 0
    assert prepared.raw_index_gap_count == 3
    assert prepared.gap_mask[3:5].all()
    assert len(prepared.gap_events) == 1
    assert prepared.gap_events[0].sample_count == 3


def test_heavy_motion_is_a_local_peak_mask_not_a_session_exclusion(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = pd.date_range("2026-01-01", periods=2_000, freq="10ms", tz="UTC")
    motion_mask = np.zeros(len(timestamps), dtype=bool)
    motion_mask[500:550] = True
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": timestamps,
            "clean_value": np.sin(np.arange(len(timestamps), dtype=float) / 10.0),
            "sampling_rate_hz": 100.0,
            "motion_heavy_mask": motion_mask,
            "madc_action": "MOTION_HEAVY_ANNOTATED",
        }
    )

    prepared = validator._prepare_frame(frame, "ppg_ir")

    np.testing.assert_array_equal(prepared.gap_mask, motion_mask)
    assert not prepared.long_gap_detected
    assert [event.gap_type for event in prepared.gap_events] == ["motion_heavy"]
    assert validator._motion_quality(frame) == "masked"
    segments = validator._detection_segments(prepared)
    assert segments == [(0, 500), (550, 2_000)]


def test_legacy_heavy_motion_without_mask_remains_fail_closed(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    legacy = pd.DataFrame({"madc_action": ["MOTION_HEAVY_ANNOTATED"]})

    assert validator._motion_quality(legacy) == "heavy"


def test_morphology_composite_score_is_diagnostic_not_a_hidden_gate(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    component_pass = {
        "morphology_valid_ratio": 0.80,
        "morphology_score": 0.20,
        "flags": (),
    }

    assert validator._morphology_exclusion_reason(component_pass) is None
    amplitude_failure = {
        **component_pass,
        "flags": ("morphology_amplitude_inconsistent",),
    }
    assert (
        validator._morphology_exclusion_reason(amplitude_failure)
        == "ppg_morphology_amplitude_invalid"
    )
    assert (
        validator._morphology_exclusion_reason(amplitude_failure, policy="diagnostic")
        is None
    )
    assert (
        validator._morphology_exclusion_reason(
            {
                **component_pass,
                "flags": ("morphology_insufficient_peaks",),
            },
            policy="diagnostic",
        )
        == "ppg_morphology_insufficient_peaks"
    )


def _morphology_ablation_candidate(*, warning_rows_per_channel: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    warning_counts = {channel: 0 for channel in ("ppg_ir", "ppg_red", "ppg_green")}
    for participant in range(1, 29):
        for phase in ("relax", "task"):
            for channel in ("ppg_ir", "ppg_red", "ppg_green"):
                warning = warning_counts[channel] < warning_rows_per_channel
                if warning:
                    warning_counts[channel] += 1
                rows.append(
                    {
                        "participant_id": f"ID-{participant:02d}",
                        "phase": phase,
                        "sensor_channel": channel,
                        "status": "ok",
                        "structural_status": "pass",
                        "metric_status": "computed",
                        "quality_status": "warning" if warning else "pass",
                        "strict_quality_eligible": not warning,
                        "validation_status": "valid",
                        "quality_flags": (
                            "morphology_amplitude_inconsistent" if warning else ""
                        ),
                        "exclusion_reason": "",
                        "ecg_peak_count": 240,
                        "ppg_peak_count": 240,
                        "matched_beat_count": 220,
                        "beat_f1": 0.70 if warning else 0.80,
                        "ibi_mae_ms": 30.0 if warning else 28.0,
                        "hr_mae_bpm": 3.2 if warning else 3.0,
                    }
                )
    return pd.DataFrame(rows)


def test_morphology_policy_changes_status_only() -> None:
    candidate = _morphology_ablation_candidate(warning_rows_per_channel=1)

    diagnostic = apply_morphology_status_policy(candidate, "diagnostic")
    hard = apply_morphology_status_policy(diagnostic, "hard")
    warning_key = (
        (diagnostic["participant_id"] == "ID-01")
        & (diagnostic["phase"] == "relax")
        & (diagnostic["sensor_channel"] == "ppg_green")
    )

    assert diagnostic.loc[warning_key, "validation_status"].item() == "valid"
    assert diagnostic.loc[warning_key, "quality_status"].item() == "warning"
    assert not bool(diagnostic.loc[warning_key, "strict_quality_eligible"].item())
    assert hard.loc[warning_key, "validation_status"].item() == "excluded"
    assert hard.loc[warning_key, "quality_status"].item() == "fail"
    for column in (
        "ecg_peak_count",
        "ppg_peak_count",
        "matched_beat_count",
        "beat_f1",
        "ibi_mae_ms",
        "hr_mae_bpm",
    ):
        pd.testing.assert_series_equal(hard[column], diagnostic[column])


def test_interpolation_provenance_masks_are_complete_and_hashed() -> None:
    frame = pd.DataFrame(
        {
            "clean_value": [1.0, 1.5, np.nan, 2.0],
            "observed_mask": [True, False, False, True],
            "interpolated_mask": [False, True, False, False],
            "gap_mask": [False, False, True, False],
            "unresolved_mask": [False, False, False, False],
            "interpolation_provenance": [
                "",
                "bounded_linear_waveform",
                "",
                "",
            ],
        }
    )

    result = PpgEcgValidator._interpolation_mask_provenance(frame)

    assert result["ppg_interpolation_mask_status"] == "explicit_complete"
    assert result["ppg_observed_sample_count"] == 2
    assert result["ppg_gap_provenance_sample_count"] == 1
    assert result["ppg_unresolved_sample_count"] == 0
    assert result["ppg_interpolation_method"] == "bounded_linear_waveform"
    assert len(result["ppg_interpolation_mask_sha256"]) == 64


def test_interpolation_provenance_rejects_overlapping_masks() -> None:
    frame = pd.DataFrame(
        {
            "clean_value": [1.0, 2.0],
            "observed_mask": [True, True],
            "interpolated_mask": [False, True],
            "gap_mask": [False, False],
            "unresolved_mask": [False, False],
        }
    )

    result = PpgEcgValidator._interpolation_mask_provenance(frame)

    assert result["ppg_interpolation_mask_status"] == "invalid_partition"


def test_diagnostic_morphology_never_overwrites_objective_quality_failure() -> None:
    candidate = (
        _morphology_ablation_candidate(warning_rows_per_channel=1).iloc[:1].copy()
    )
    candidate.loc[:, "status"] = "excluded"
    candidate.loc[:, "validation_status"] = "excluded"
    candidate.loc[:, "quality_status"] = "fail"
    candidate.loc[:, "strict_quality_eligible"] = True
    candidate.loc[:, "exclusion_reason"] = "ppg_sqa_failed:pre=poor;post=poor"

    diagnostic = apply_morphology_status_policy(candidate, "diagnostic")

    assert diagnostic["status"].item() == "excluded"
    assert diagnostic["validation_status"].item() == "excluded"
    assert diagnostic["quality_status"].item() == "fail"
    assert not bool(diagnostic["strict_quality_eligible"].item())
    assert diagnostic["exclusion_reason"].item() == "ppg_sqa_failed:pre=poor;post=poor"


def test_development_morphology_ablation_promotes_only_noninferior_policy() -> None:
    candidate = _morphology_ablation_candidate(warning_rows_per_channel=1)
    diagnostic = apply_morphology_status_policy(candidate, "diagnostic")
    hard = apply_morphology_status_policy(diagnostic, "hard")

    result = evaluate_morphology_policy_ablation(hard, diagnostic)

    assert result["complete_development_ledger"]
    assert result["peak_stream_unchanged"]
    assert result["coverage_increased"]
    assert result["promotion_eligible"]
    assert result["holdout_participants_accessed"] == []

    degraded_candidate = _morphology_ablation_candidate(warning_rows_per_channel=30)
    degraded_diagnostic = apply_morphology_status_policy(
        degraded_candidate, "diagnostic"
    )
    degraded_hard = apply_morphology_status_policy(degraded_diagnostic, "hard")
    degraded = evaluate_morphology_policy_ablation(degraded_hard, degraded_diagnostic)
    assert not degraded["promotion_eligible"]
    assert all(not channel["noninferior"] for channel in degraded["channels"])


def test_morphology_promotion_is_independent_per_channel() -> None:
    candidate = _morphology_ablation_candidate(warning_rows_per_channel=1)
    red_indices = candidate[candidate["sensor_channel"].eq("ppg_red")].index[:30]
    candidate.loc[red_indices, "quality_status"] = "warning"
    candidate.loc[red_indices, "strict_quality_eligible"] = False
    candidate.loc[red_indices, "quality_flags"] = "morphology_amplitude_inconsistent"
    red_warning = candidate.index.isin(red_indices)
    candidate.loc[red_warning, "beat_f1"] = 0.0
    candidate.loc[red_warning, "ibi_mae_ms"] = 500.0
    candidate.loc[red_warning, "hr_mae_bpm"] = 20.0
    diagnostic = apply_morphology_status_policy(candidate, "diagnostic")
    hard = apply_morphology_status_policy(diagnostic, "hard")

    result = evaluate_morphology_policy_ablation(hard, diagnostic)
    eligibility = {
        row["sensor_channel"]: row["promotion_eligible"] for row in result["channels"]
    }

    assert eligibility == {
        "ppg_green": True,
        "ppg_ir": True,
        "ppg_red": False,
    }


def test_development_morphology_ablation_rejects_holdout_rows() -> None:
    candidate = _morphology_ablation_candidate(warning_rows_per_channel=1)
    candidate.loc[candidate.index[0], "participant_id"] = "ID-29"

    with pytest.raises(ValueError, match="ablation_forbidden_participants:ID-29"):
        evaluate_morphology_policy_ablation(candidate, candidate)


def test_ecg_gap_is_preserved_and_not_interpolated(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = pd.date_range("2026-01-01", periods=2_000, freq="10ms", tz="UTC")
    values = np.sin(np.arange(2_000, dtype=float) / 10.0)
    values[500:503] = np.nan
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": timestamps,
            "clean_value": values,
            "sampling_rate_hz": 100.0,
        }
    )

    prepared = validator._prepare_frame(frame, "ecg")

    assert np.isnan(prepared.values[500:503]).all()
    assert not prepared.short_gap_interpolated
    assert not prepared.interpolated_mask.any()
    assert prepared.gap_mask[499:504].all()
    assert prepared.gap_events[0].action == "mask_and_exclude_affected_beats"


def test_ppg_without_canonical_clean_value_is_rejected(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = pd.date_range("2026-01-01", periods=100, freq="10ms", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": timestamps,
            "raw_value": np.zeros(len(timestamps)),
        }
    )

    with pytest.raises(KeyError, match="silver_missing_clean_value"):
        validator._prepare_frame(frame, "ppg_ir")


def test_long_ppg_gap_is_not_interpolated(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = pd.date_range("2026-01-01", periods=3_000, freq="10ms", tz="UTC")
    values = np.sin(np.arange(3_000, dtype=float) / 10.0)
    values[1_200:1_450] = np.nan
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": timestamps,
            "clean_value": values,
            "sampling_rate_hz": 100.0,
        }
    )

    prepared = validator._prepare_frame(frame, "ppg_green")

    assert prepared.long_gap_detected
    assert not prepared.short_gap_interpolated
    assert np.isnan(prepared.values[1_200:1_450]).all()
    assert prepared.gap_events[0].gap_type == "long_dropout"


def test_validation_crop_uses_the_exact_240_second_core_interval(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = pd.date_range("2026-01-01", periods=40_001, freq="10ms", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": timestamps,
            "clean_value": np.zeros(len(timestamps)),
        }
    )
    start = timestamps[5_000]
    end = start + pd.Timedelta(seconds=240)

    cropped, coverage = validator._crop_to_window(frame, start, end)

    assert 239.99 <= coverage <= 240.0
    assert (end - start).total_seconds() == 240.0
    assert cropped["timestamp_raw_utc"].min() == start
    assert cropped["timestamp_raw_utc"].max() < end


def test_validator_window_is_anchor_plus_30_to_plus_270(tmp_path: Path) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    anchor_start = pd.Timestamp("2026-01-01T00:00:00Z")
    validator._anchors_by_participant_phase = {
        ("ID-01", "relax"): pd.Series(
            {
                "start_timestamp": anchor_start,
                "end_timestamp": anchor_start + pd.Timedelta(seconds=300),
            }
        )
    }

    start, end = validator._validation_window("ID-01", "relax")

    assert start == anchor_start + pd.Timedelta(seconds=30)
    assert end == anchor_start + pd.Timedelta(seconds=270)
    assert (end - start).total_seconds() == 240


def test_all_ppg_channels_reuse_one_participant_phase_sync_row(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    sync_dir = tmp_path / "data" / "sync"
    sync_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "participant_id": "ID-01",
                "recording_id": "ID-01_relax_raw",
                "alpha": 1.001,
                "beta_seconds": 1.25,
                "method": "phase_anchor+ppg_timebase+physiological_pat",
                "excluded": False,
                "clock_excluded": False,
                "ppg_diagnostic_reason": "insufficient_ppg_peaks:4",
                "diagnostic_reference_channel": "ppg_ir",
            }
        ]
    ).to_parquet(sync_dir / "synchronization_manifest_v3.parquet", index=False)
    validator = PpgEcgValidator(paths)

    rows = [{}, {}, {}]
    resolved = [validator._add_sync_metadata(row, "ID-01", "relax") for row in rows]

    assert resolved == [True, True, True]
    assert {row["sync_alpha"] for row in rows} == {1.001}
    assert {row["sync_beta_seconds"] for row in rows} == {1.25}

    audit = validator._timebase_candidate_audit()
    selected = audit[
        (audit["participant_id"] == "ID-01") & (audit["phase"] == "relax")
    ].iloc[0]
    assert len(audit) == 32 * 2
    assert selected["recording_id"] == "ID-01_relax_raw"
    assert selected["alpha"] == 1.001
    assert selected["candidate_status"] == "accepted"
    assert selected["ppg_diagnostic_reason"] == "insufficient_ppg_peaks:4"
    assert selected["diagnostic_reference_channel"] == "ppg_ir"
    assert int((audit["candidate_status"] == "missing").sum()) == 63


def test_run_emits_all_whitelisted_combinations_with_exclusions(tmp_path: Path) -> None:
    outputs = PpgEcgValidator(_paths(tmp_path)).run()

    metrics = pd.read_csv(outputs.metrics_path)
    assert len(metrics) == 32 * 2 * 3
    assert outputs.output_dir.name == "ppg_ecg_240s"
    assert set(metrics["participant_id"]) == {
        f"ID-{index:02d}" for index in range(1, 33)
    }
    assert set(metrics["phase"]) == {"relax", "task"}
    assert set(metrics["sensor_channel"]) == {"ppg_ir", "ppg_red", "ppg_green"}
    assert set(metrics["validation_window_seconds"]) == {240}
    assert (metrics["status"] == "excluded").all()
    assert set(metrics["structural_status"]) == {"unavailable"}
    assert set(metrics["metric_status"]) == {"unavailable"}
    assert set(metrics["quality_status"]) == {"unavailable"}
    assert not metrics["strict_quality_eligible"].any()
    assert set(metrics["validation_status"]) == {"excluded"}
    assert metrics["exclusion_reason"].notna().all()
    audit_plots = list(
        (outputs.output_dir / "plots").glob("ID-*/**/peak_detection_240s.png")
    )
    assert len(audit_plots) == 32 * 2 * 3
    plots = list((outputs.output_dir / "plots").rglob("peak_detection_240s.png"))
    assert len(plots) == 32 * 2 * 3
    recommendation = json.loads(
        (outputs.output_dir / "channel_recommendation.json").read_text(encoding="utf-8")
    )
    assert recommendation["recommended_channel"] is None
    assert {row["decision"] for row in recommendation["channels"]} == {
        "insufficient_evidence"
    }

    participant_phase = pd.read_csv(
        outputs.output_dir / "participant_phase_summary_240s.csv"
    )
    timebase = pd.read_csv(outputs.output_dir / "timebase_candidate_audit_240s.csv")
    exclusions = pd.read_csv(outputs.output_dir / "exclusion_breakdown_240s.csv")
    baseline = pd.read_csv(outputs.output_dir / "baseline_vs_rebuild_240s.csv")

    assert len(participant_phase) == 32 * 2
    assert set(participant_phase["session_validation_status"]) == {"excluded"}
    assert set(participant_phase["observed_channel_count"]) == {3}
    assert len(timebase) == 32 * 2
    assert set(timebase["candidate_status"]) == {"missing"}
    assert len(exclusions) == 6
    assert "full_cohort_not_evaluated_unfrozen_contract" in set(
        exclusions["exclusion_reason"]
    )
    assert set(exclusions["combination_count"]) == {8, 56}
    assert len(baseline) == 32 * 2 * 3
    assert set(baseline["comparison_status"]) == {"baseline_file_missing"}
    ablation = json.loads(
        (outputs.output_dir / "morphology_policy_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    assert ablation["holdout_participants_accessed"] == []
    assert ablation["effective_policy"] == "hard"
    summary = json.loads(
        (outputs.output_dir / "validation_summary_240s.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["metric_computable_combinations"] == 0
    assert summary["strict_quality_combinations"] == 0
    assert summary["quality_warning_combinations"] == 0


def test_reference_style_plot_uses_participant_phase_channel_directory(
    tmp_path: Path,
) -> None:
    validator = PpgEcgValidator(_paths(tmp_path))
    timestamps = np.arange(0.0, 300_000.0, 1_000.0)
    signal = np.sin(np.arange(len(timestamps), dtype=float) / 4.0)
    peak_indices = np.arange(10, len(timestamps), 10)
    ecg = DetectedPeaks(
        timestamps,
        signal,
        peak_indices,
        fs_hz=1.0,
        method="test_ecg",
    )
    ppg = DetectedPeaks(
        timestamps + 220.0,
        signal,
        peak_indices,
        fs_hz=1.0,
        method="test_ppg",
    )
    matches, _ = validator._match_with_pat(ecg.timestamps_ms, ppg.timestamps_ms)

    validator._plot_audit("ID-01", "relax", "ppg_green", ecg, ppg, matches)

    assert (
        validator.audit_dir
        / "ID-01"
        / "relax"
        / "ppg_green"
        / "peak_detection_240s.png"
    ).is_file()
