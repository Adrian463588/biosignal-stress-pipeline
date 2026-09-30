from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.contracts import (
    DetectedPeaks,
    GapEvent,
    ObservedIntervalBlock,
)
from PipelinePython.common.hrv_agreement import (
    FEATURES,
    TIME_DOMAIN_FEATURES,
    HrvAgreementAnalyzer,
    _bootstrap_composite,
    _hrv_feature_result,
    _icc_a1,
    _metric_components,
    _scope_frame,
    _observed_intervals,
    _run_comparison,
    _spectral_features,
    _spectral_features_from_intervals,
    _successive_differences,
    write_hrv_agreement_artifacts,
)
from PipelinePython.common.settings import HrvAgreementConfig


def _interval(
    index: int,
    value_ms: float,
    *,
    block_id: int | None,
    eligible: bool = True,
    start_ms: float | None = None,
    spectral_support_id: int | None = 0,
) -> ObservedIntervalBlock:
    start = float(index * value_ms if start_ms is None else start_ms)
    return ObservedIntervalBlock(
        participant_id="ID-01",
        phase="relax",
        sensor_channel="ppg_green",
        signal_role="ppg",
        interval_index=index,
        block_id=block_id,
        start_peak_timestamp_ms=start,
        end_peak_timestamp_ms=start + value_ms,
        interval_ms=value_ms,
        physiological=eligible,
        gap_crossed=False,
        moving_median_valid=eligible,
        mad_valid=eligible,
        eligible=eligible,
        exclusion_reason=None if eligible else "rejected",
        spectral_support_id=spectral_support_id,
    )


def test_rejected_interval_breaks_successive_hrv_pairs() -> None:
    rows = _observed_intervals(
        "ID-01",
        "relax",
        "ppg_green",
        "ppg",
        np.asarray([0.0, 1_000.0, 2_000.0, 2_100.0, 3_100.0, 4_200.0]),
        (),
        HrvAgreementConfig(),
    )

    assert [row.block_id for row in rows] == [0, 0, None, 1, 1]
    differences = _successive_differences([row for row in rows if row.eligible])
    np.testing.assert_array_equal(differences, np.asarray([0.0, 100.0]))


def test_gap_crossing_interval_breaks_block() -> None:
    gap = GapEvent(1_400.0, 1_600.0, 200.0, 1, "dropout", "mask", True)
    rows = _observed_intervals(
        "ID-01",
        "relax",
        "ppg_green",
        "ecg",
        np.asarray([0.0, 1_000.0, 2_000.0, 3_000.0]),
        (gap,),
        HrvAgreementConfig(),
    )

    assert [row.block_id for row in rows] == [0, None, 1]
    assert rows[1].exclusion_reason == "interval_crosses_gap"
    assert _successive_differences([row for row in rows if row.eligible]).size == 0


def test_two_short_blocks_do_not_satisfy_spectral_support() -> None:
    first = [_interval(index, 1_000.0, block_id=0) for index in range(100)]
    second = [
        _interval(index + 101, 1_000.0, block_id=1, start_ms=101_000 + index * 1_000)
        for index in range(100)
    ]

    with pytest.raises(ValueError, match="insufficient_contiguous_spectral_support"):
        _spectral_features({0: first, 1: second}, HrvAgreementConfig())


def test_spectral_pchip_disables_extrapolation(monkeypatch: pytest.MonkeyPatch) -> None:
    import PipelinePython.common.hrv_agreement as module

    calls: list[bool] = []
    real = module.PchipInterpolator

    def tracked(*args: object, **kwargs: object) -> object:
        calls.append(bool(kwargs.get("extrapolate")))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "PchipInterpolator", tracked)
    block = [
        _interval(index, 1_000.0 + (index % 5), block_id=0) for index in range(190)
    ]

    _spectral_features({0: block}, HrvAgreementConfig())

    assert calls == [False]


def test_time_domain_features_survive_missing_spectral_support() -> None:
    intervals = [
        _interval(index, 1_000.0 + (index % 5), block_id=0) for index in range(130)
    ]

    features, reasons, diagnostics = _hrv_feature_result(
        intervals, HrvAgreementConfig()
    )

    assert all(np.isfinite(features[feature]) for feature in FEATURES[:3])
    assert all(np.isnan(features[feature]) for feature in FEATURES[3:])
    assert set(reasons[feature] for feature in FEATURES[3:]) == {
        "insufficient_contiguous_spectral_support"
    }
    assert diagnostics["spectral_corrected_interval_count"] == 0


def test_ecg_reference_features_do_not_require_a_ppg_stream() -> None:
    intervals = 1_000.0 + 30.0 * np.sin(np.arange(220, dtype=float) / 10.0)
    timestamps = np.concatenate(([0.0], np.cumsum(intervals)))
    ecg = DetectedPeaks(
        timestamps_ms=timestamps,
        signal=np.zeros(timestamps.size, dtype=float),
        sample_indices=np.arange(timestamps.size, dtype=np.int64),
        fs_hz=100.0,
        method="test_ecg",
    )

    columns = HrvAgreementAnalyzer(HrvAgreementConfig()).ecg_reference_columns(
        participant_id="ID-01",
        phase="relax",
        sensor_channel="ppg_green",
        ecg=ecg,
    )

    assert columns["ecg_reference_hrv_rmssd_ms_status"] == "COMPUTED"
    assert columns["ecg_reference_hrv_sdnn_ms_status"] == "COMPUTED"
    assert np.isfinite(columns["ecg_reference_hrv_rmssd_ms"])


def test_spectral_correction_is_bounded_and_provenanced() -> None:
    intervals = []
    block_id = 0
    for index in range(190):
        eligible = index not in {50, 100}
        if not eligible:
            current_block = None
            block_id += 1
        else:
            current_block = block_id
        intervals.append(
            _interval(
                index,
                1_000.0 + 30.0 * np.sin(index / 10.0),
                block_id=current_block,
                eligible=eligible,
                start_ms=index * 1_000.0,
                spectral_support_id=0,
            )
        )

    lf, hf, diagnostics = _spectral_features_from_intervals(
        intervals, HrvAgreementConfig()
    )

    assert lf > 0.0
    assert hf > 0.0
    assert diagnostics["spectral_corrected_interval_count"] == 2
    assert diagnostics["spectral_correction_fraction"] == pytest.approx(2 / 190)
    assert diagnostics["spectral_support_seconds"] >= 180.0


def test_spectral_correction_rejects_excessive_artifact_fraction() -> None:
    intervals = [
        _interval(
            index,
            1_000.0 + 30.0 * np.sin(index / 10.0),
            block_id=None if index % 10 == 0 else 0,
            eligible=index % 10 != 0,
            start_ms=index * 1_000.0,
            spectral_support_id=0,
        )
        for index in range(190)
    ]

    with pytest.raises(ValueError, match="artifact_fraction_above_limit"):
        _spectral_features_from_intervals(intervals, HrvAgreementConfig())


def test_pooled_bootstrap_relabels_duplicate_participant_draws() -> None:
    rows = []
    for participant in range(1, 9):
        for phase_index, phase in enumerate(("relax", "task")):
            value = float(participant + phase_index * 10)
            rows.append(
                {
                    "participant_id": f"ID-{participant:02d}",
                    "phase": phase,
                    "ecg_hrv_rmssd_ms": value,
                    "ppg_hrv_rmssd_ms": value + participant * 0.01,
                }
            )

    low, high, failures = _bootstrap_composite(
        pd.DataFrame(rows),
        "rmssd_ms",
        17.0,
        scope="pooled",
        iterations=100,
        seed=42,
    )

    assert np.isfinite([low, high]).all()
    assert low <= high
    assert failures < 50


def test_pooled_scope_counts_available_sessions_not_only_complete_pairs() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["ID-01", "ID-01", "ID-02"],
            "phase": ["relax", "task", "relax"],
        }
    )

    pooled = _scope_frame(frame, "pooled")

    assert len(pooled) == 3
    assert set(pooled["participant_id"]) == {"ID-01", "ID-02"}


def test_pooled_composite_survives_unavailable_repeated_bland_altman() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": [f"ID-{index:02d}" for index in range(1, 7)],
            "phase": ["task"] * 6,
            "ecg_hrv_rmssd_ms": np.arange(1.0, 7.0),
            "ppg_hrv_rmssd_ms": np.arange(1.0, 7.0),
        }
    )

    components = _metric_components(frame, "rmssd_ms", 5.0, "pooled")

    assert components["composite_score"] == pytest.approx(1.0)
    assert components["bland_altman_status"] == "INSUFFICIENT_EVIDENCE"
    assert np.isnan(components["bias"])


def test_icc_a1_known_cases_and_zero_variation() -> None:
    np.testing.assert_allclose(_icc_a1(np.array([1, 2, 3]), np.array([1, 2, 3])), 1.0)
    np.testing.assert_allclose(
        _icc_a1(np.array([1, 2, 3]), np.array([2, 3, 4])), 2.0 / 3.0
    )
    np.testing.assert_allclose(_icc_a1(np.array([1, 2, 3]), np.array([3, 2, 1])), -3.0)
    assert np.isnan(_icc_a1(np.ones(3), np.ones(3)))


def test_composite_includes_zero_mae_as_one() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": [f"ID-{index:02d}" for index in range(1, 7)],
            "phase": ["relax"] * 6,
            "ecg_hrv_rmssd_ms": np.arange(1.0, 7.0),
            "ppg_hrv_rmssd_ms": np.arange(1.0, 7.0),
        }
    )

    components = _metric_components(frame, "rmssd_ms", 5.0, "relax")

    assert components["accuracy_component"] == 1.0
    assert components["composite_score"] == pytest.approx(1.0)


def test_high_correlation_with_large_bias_fails_composite() -> None:
    values = np.arange(1.0, 7.0)
    frame = pd.DataFrame(
        {
            "participant_id": [f"ID-{index:02d}" for index in range(1, 7)],
            "phase": ["relax"] * 6,
            "ecg_hrv_rmssd_ms": values,
            "ppg_hrv_rmssd_ms": values + 10.0,
        }
    )

    components = _metric_components(frame, "rmssd_ms", 5.0, "relax")

    assert components["pearson_r"] == pytest.approx(1.0)
    assert components["spearman_rho"] == pytest.approx(1.0)
    assert components["composite_score"] < 0.70


def _agreement_rows(
    *, confounded_task: bool = False, participant_count: int = 32
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant in range(1, participant_count + 1):
        for phase in ("relax", "task"):
            ecg = float(participant + (100 if phase == "task" else 0))
            ppg = (
                float(129 - participant) if confounded_task and phase == "task" else ecg
            )
            if confounded_task and phase == "task":
                ppg += 100.0
            row: dict[str, object] = {
                "participant_id": f"ID-{participant:02d}",
                "phase": phase,
                "sensor_channel": "ppg_green",
                "hrv_feature_status": "COMPUTED",
            }
            for feature_index, feature in enumerate(FEATURES, start=1):
                row[f"ecg_hrv_{feature}"] = ecg * feature_index
                row[f"ppg_hrv_{feature}"] = ppg * feature_index
                row[f"ecg_reference_hrv_{feature}"] = ecg * feature_index
                row[f"ecg_reference_hrv_{feature}_status"] = "COMPUTED"
            rows.append(row)
    return pd.DataFrame(rows)


def test_missing_spectral_support_does_not_block_time_domain_go() -> None:
    frame = _agreement_rows()
    frame["ecg_hrv_lf_hf_ratio"] = 1.0
    frame["ppg_hrv_lf_hf_ratio"] = 1.0

    payload, _ = HrvAgreementAnalyzer(HrvAgreementConfig()).evaluate(
        frame,
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )

    decision = payload["channels"]["ppg_green"]
    assert decision["decision"] == "GO"
    assert decision["time_domain_decision"] == "GO"
    assert decision["spectral_validation_status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["primary_features"] == list(TIME_DOMAIN_FEATURES)


def test_each_time_domain_feature_is_required_for_primary_go() -> None:
    frame = _agreement_rows()
    frame.loc[frame.index[:13], "ppg_hrv_rmssd_ms"] = np.nan

    payload, _ = HrvAgreementAnalyzer(HrvAgreementConfig()).evaluate(
        frame,
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )

    assert payload["channels"]["ppg_green"]["decision"] == "INSUFFICIENT_EVIDENCE"


def test_phase_gate_rejects_phase_confounded_pooled_correlation() -> None:
    payload, metrics = HrvAgreementAnalyzer(HrvAgreementConfig()).evaluate(
        _agreement_rows(confounded_task=True),
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )

    pooled = metrics[
        metrics["scope"].eq("pooled") & metrics["feature"].eq("rmssd_ms")
    ].iloc[0]
    assert pooled["pearson_r"] > 0.9
    assert payload["channels"]["ppg_green"]["decision"] == "NO_GO"


def test_full_cohort_quorum_boundaries_are_fifty_two_and_twenty_six() -> None:
    frame = _agreement_rows()
    removed = {
        (f"ID-{participant:02d}", phase)
        for participant in range(1, 7)
        for phase in ("relax", "task")
    }
    fifty_two = frame[
        ~frame[["participant_id", "phase"]].apply(tuple, axis=1).isin(removed)
    ]
    fifty_one = fifty_two.drop(fifty_two.index[0])
    analyzer = HrvAgreementAnalyzer(HrvAgreementConfig())

    passing, passing_metrics = analyzer.evaluate(
        fifty_two,
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )
    failing, _ = analyzer.evaluate(
        fifty_one,
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )

    passing_channel = passing["channels"]["ppg_green"]
    assert passing_channel["decision"] == "GO"
    assert passing_channel["valid_time_domain_pooled_rows"] == 52
    assert passing_channel["valid_time_domain_relax_rows"] == 26
    assert passing_channel["valid_time_domain_task_rows"] == 26
    assert set(
        passing_metrics.loc[
            passing_metrics["feature"].isin(TIME_DOMAIN_FEATURES), "sample_count"
        ]
    ) == {26, 52}
    assert failing["channels"]["ppg_green"]["decision"] == "INSUFFICIENT_EVIDENCE"


def test_full_cohort_metrics_include_all_thirty_two_participants() -> None:
    payload, metrics = HrvAgreementAnalyzer(HrvAgreementConfig()).evaluate(
        _agreement_rows(),
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )

    time_metrics = metrics[metrics["feature"].isin(TIME_DOMAIN_FEATURES)]
    assert set(
        time_metrics.loc[time_metrics["scope"].eq("pooled"), "sample_count"]
    ) == {64}
    assert set(time_metrics.loc[time_metrics["scope"].eq("relax"), "sample_count"]) == {
        32
    }
    assert set(time_metrics.loc[time_metrics["scope"].eq("task"), "sample_count"]) == {
        32
    }
    assert payload["validation_participants"] == [
        f"ID-{participant:02d}" for participant in range(1, 33)
    ]


def test_any_channel_go_preserves_independent_channel_decisions() -> None:
    green = _agreement_rows()
    infrared = green.assign(sensor_channel="ppg_ir")
    red = green.assign(sensor_channel="ppg_red").copy()
    for feature in TIME_DOMAIN_FEATURES:
        red[f"ppg_hrv_{feature}"] = -red[f"ecg_hrv_{feature}"]
    payload, _ = HrvAgreementAnalyzer(HrvAgreementConfig()).evaluate(
        pd.concat([green, infrared, red], ignore_index=True),
        evaluation_scope="full_cohort",
        full_cohort_access_authorized=True,
    )

    assert payload["hrv_validation_decision"] == "GO"
    assert payload["channels"]["ppg_green"]["decision"] == "GO"
    assert payload["channels"]["ppg_ir"]["decision"] == "GO"
    assert payload["channels"]["ppg_red"]["decision"] == "NO_GO"


def test_full_cohort_access_requires_frozen_contract() -> None:
    with pytest.raises(RuntimeError, match="full_cohort_access_not_authorized"):
        HrvAgreementAnalyzer(HrvAgreementConfig()).evaluate(
            _agreement_rows(),
            evaluation_scope="full_cohort",
            full_cohort_access_authorized=False,
        )


def test_ecg_reference_range_is_invariant_across_ppg_candidates() -> None:
    first = _agreement_rows(participant_count=28)
    second = _agreement_rows(participant_count=28)
    second["sensor_channel"] = "ppg_ir"
    second.loc[
        second["participant_id"].isin({"ID-01", "ID-02"}), "ppg_hrv_rmssd_ms"
    ] = np.nan
    analyzer = HrvAgreementAnalyzer(HrvAgreementConfig())
    ranges = analyzer.development_reference_ranges(pd.concat([first, second]))

    first_payload, _ = analyzer.evaluate(
        first, evaluation_scope="selection", development_ranges=ranges
    )
    second_payload, _ = analyzer.evaluate(
        second, evaluation_scope="selection", development_ranges=ranges
    )

    assert (
        first_payload["ecg_reference_range_fingerprint"]
        == second_payload["ecg_reference_range_fingerprint"]
    )
    assert (
        first_payload["ecg_reference_development_ranges"]
        == second_payload["ecg_reference_development_ranges"]
    )


def test_internal_test_extremes_do_not_change_development_range() -> None:
    baseline = _agreement_rows()
    mutated = baseline.copy()
    internal_test = mutated["participant_id"].isin({"ID-29", "ID-30", "ID-31", "ID-32"})
    for feature in FEATURES:
        mutated.loc[internal_test, f"ecg_reference_hrv_{feature}"] = 1e12
    analyzer = HrvAgreementAnalyzer(HrvAgreementConfig())

    assert analyzer.development_reference_ranges(
        baseline
    ) == analyzer.development_reference_ranges(mutated)


def test_writer_emits_exact_192_feature_pair_rows(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for participant in range(1, 33):
        for phase in ("relax", "task"):
            for channel in ("ppg_green", "ppg_ir", "ppg_red"):
                row: dict[str, object] = {
                    "participant_id": f"ID-{participant:02d}",
                    "phase": phase,
                    "sensor_channel": channel,
                    "recording_id": f"ID-{participant:02d}-{phase}",
                    "source_file": "ppg",
                    "source_checksum": "a",
                    "ecg_source_file": "ecg",
                    "ecg_source_checksum": "b",
                    "ppg_source_file": "ppg",
                    "ppg_source_checksum": "a",
                    "hrv_feature_status": "INSUFFICIENT_EVIDENCE",
                    "hrv_feature_exclusion_reason": "unit_fixture",
                    "hrv_common_observed_seconds": 0.0,
                    "hrv_method_version": "block_safe_hrv_prv_v3",
                    "hrv_analysis_duration_label": "short_term_240s",
                    "hrv_pnn50_unit": "fraction",
                    "ecg_spectral_corrected_interval_count": 0,
                    "ppg_spectral_corrected_interval_count": 0,
                    "ecg_spectral_correction_fraction": np.nan,
                    "ppg_spectral_correction_fraction": np.nan,
                    "ecg_spectral_support_seconds": np.nan,
                    "ppg_spectral_support_seconds": np.nan,
                    "hrv_tachogram_interpolation_method": "none",
                    "run_id": "UNIT",
                    "source_tree_sha256": "tree",
                    "run_contract_sha256": "contract",
                    "config_fingerprint": "config",
                    "metric_status": "unavailable",
                }
                for feature in FEATURES:
                    row[f"ecg_hrv_{feature}"] = np.nan
                    row[f"ppg_hrv_{feature}"] = np.nan
                    row[f"ecg_reference_hrv_{feature}"] = np.nan
                    row[f"ecg_reference_hrv_{feature}_status"] = "INSUFFICIENT_EVIDENCE"
                    row[f"hrv_{feature}_status"] = "INSUFFICIENT_EVIDENCE"
                    row[f"hrv_{feature}_exclusion_reason"] = "unit_fixture"
                rows.append(row)
    output = tmp_path / "validation" / "hrv_agreement"

    write_hrv_agreement_artifacts(
        project_root=tmp_path,
        run_id="UNIT",
        validation_metrics=pd.DataFrame(rows),
        interval_rows=(),
        output_dir=output,
        analyzer=HrvAgreementAnalyzer(HrvAgreementConfig()),
        bootstrap_iterations=0,
        bootstrap_seed=42,
        run_identity={"run_contract_sha256": "contract"},
        evaluation_scope="selection",
        full_cohort_access_authorized=False,
    )

    assert len(pd.read_csv(output / "hrv_feature_pairs_240s.csv")) == 192
    assert (output / "observed_interval_blocks_240s.parquet").is_file()
    assert (output / "implementation_checklist.json").is_file()
    assert (output / "implementation_checklist.md").is_file()

    duplicated = pd.DataFrame(rows)
    duplicated.loc[191, ["participant_id", "phase", "sensor_channel"]] = duplicated.loc[
        0, ["participant_id", "phase", "sensor_channel"]
    ]
    with pytest.raises(RuntimeError, match="duplicate_keys"):
        write_hrv_agreement_artifacts(
            project_root=tmp_path,
            run_id="UNIT-DUPLICATE",
            validation_metrics=duplicated,
            interval_rows=(),
            output_dir=tmp_path / "validation" / "duplicate",
            analyzer=HrvAgreementAnalyzer(HrvAgreementConfig()),
            bootstrap_iterations=0,
            bootstrap_seed=42,
            run_identity={"run_contract_sha256": "contract"},
            evaluation_scope="selection",
            full_cohort_access_authorized=False,
        )


def test_tachogram_contract_rejects_invalid_nyquist_rate() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        replace(HrvAgreementConfig(), tachogram_rate_hz=0.8)


def test_run_comparison_reads_legacy_validation_status(tmp_path: Path) -> None:
    model_dir = (
        tmp_path
        / "data"
        / "artifacts"
        / "runs"
        / "current-run"
        / "models"
        / "stress"
        / "ppg_green"
        / "model-v1"
    )
    model_dir.mkdir(parents=True)
    (model_dir / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "sensor_channel": "ppg_green",
                "model_name": "logistic_regression",
                "model_version": "model-v1",
                "performance_tier": "research_usable",
                "release_status": "release_candidate",
                "feature_order": ["rmssd_ms"],
                "development_metrics": {"balanced_accuracy": 0.75},
                "holdout_metrics": {"balanced_accuracy": 0.7},
            }
        ),
        encoding="utf-8",
    )
    legacy_dir = (
        tmp_path
        / "data"
        / "artifacts"
        / "runs"
        / "20260809T193919Z"
        / "validation"
        / "ppg_ecg_240s"
    )
    legacy_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "participant_id": "ID-01",
                "sensor_channel": "ppg_green",
                "status": "ok",
                "validation_status": "valid",
                "beat_f1": 0.4,
                "ibi_mae_ms": 30.0,
                "hr_mae_bpm": 3.0,
            },
            {
                "participant_id": "ID-02",
                "sensor_channel": "ppg_green",
                "status": "excluded",
                "validation_status": "excluded",
                "beat_f1": np.nan,
                "ibi_mae_ms": np.nan,
                "hr_mae_bpm": np.nan,
            },
        ]
    ).to_csv(legacy_dir / "ppg_ecg_metrics_240s.csv", index=False)

    comparison = _run_comparison(
        tmp_path,
        "current-run",
        pd.DataFrame(
            [
                {
                    "participant_id": "ID-01",
                    "sensor_channel": "ppg_green",
                    "metric_status": "computed",
                    "hrv_feature_status": "INSUFFICIENT_EVIDENCE",
                    "beat_f1": 0.3,
                    "ibi_mae_ms": 40.0,
                    "hr_mae_bpm": 4.0,
                }
            ]
        ),
        current_hrv_payload={"hrv_validation_decision": "INSUFFICIENT_EVIDENCE"},
        current_agreement_metrics=pd.DataFrame(),
    )

    legacy = comparison["runs"]["20260809T193919Z"]["channels"]["ppg_green"]
    assert legacy["metric_computable"] == 1
    assert legacy["median_beat_f1"] == pytest.approx(0.4)
    model = comparison["runs"]["current-run"]["model_metrics"]
    assert model["status"] == "AVAILABLE"
    assert model["channels"]["ppg_green"]["spectral_features_promoted"] is False
