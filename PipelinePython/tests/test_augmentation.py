"""Tests for canonical train-fold-only waveform augmentation."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from PipelinePython.common.augmentation.pipeline import AugmentationPipeline
from PipelinePython.common.augmentation.transforms import (
    AmplitudeInversion,
    AmplitudeScaling,
    BaselineWanderInjection,
    EdgeSafeCropShift,
    GaussianNoiseInjection,
    TimeCutMix,
    TimeReversal,
)


@pytest.fixture
def example_ppg_signal():
    """Deterministic unit-contract waveform; never used as research evidence."""
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    return signal.astype(np.float32)


def test_amplitude_scaling_stays_within_canonical_range(example_ppg_signal):
    transform = AmplitudeScaling(
        scale_min=0.8, scale_max=1.2, rng=np.random.default_rng(42)
    )
    augmented, parameters = transform.apply_with_metadata(example_ppg_signal)
    assert augmented.shape == example_ppg_signal.shape
    assert 0.8 <= parameters["scale"] <= 1.2
    assert np.allclose(augmented, example_ppg_signal * parameters["scale"])

    with pytest.raises(ValueError, match="within 0.8-1.2"):
        AmplitudeScaling(scale_min=0.7, scale_max=1.2)


def test_gaussian_noise_snr_is_at_least_20_db(example_ppg_signal):
    transform = GaussianNoiseInjection(
        min_snr_db=20, max_snr_db=30, rng=np.random.default_rng(42)
    )
    augmented, parameters = transform.apply_with_metadata(example_ppg_signal)
    noise = augmented.astype(np.float64) - example_ppg_signal.astype(np.float64)
    realized_snr = 10 * np.log10(
        np.mean(example_ppg_signal.astype(np.float64) ** 2) / np.mean(noise**2)
    )
    assert parameters["snr_db"] >= 20.0
    assert realized_snr >= 19.5

    with pytest.raises(ValueError, match="SNR >= 20 dB"):
        GaussianNoiseInjection(min_snr_db=19.9)


def test_crop_shift_generic_apply_fails_without_phase_context():
    transform = EdgeSafeCropShift(max_shift=5, rng=np.random.default_rng(42))
    with pytest.raises(ValueError, match="requires explicit phase context"):
        transform.apply(np.arange(32, dtype=np.float32))


def test_phase_context_crop_shift_returns_contiguous_observed_slice():
    phase_signal = np.arange(64, dtype=np.float32)
    observed = np.ones_like(phase_signal, dtype=bool)
    transform = EdgeSafeCropShift(max_shift=5, rng=np.random.default_rng(42))

    augmented, parameters = transform.apply_with_phase_context(
        phase_signal,
        observed_mask=observed,
        window_start=20,
        window_length=16,
        phase_start=0,
        phase_end=64,
        shift_samples=5,
    )

    assert np.array_equal(augmented, phase_signal[25:41])
    assert len(augmented) == 16
    assert np.all(np.diff(augmented) == 1)
    assert parameters["observed_only"] is True
    assert parameters["target_start"] == 25


def test_phase_context_crop_shift_rejects_boundary_and_unobserved_samples():
    phase_signal = np.arange(32, dtype=np.float32)
    observed = np.ones_like(phase_signal, dtype=bool)
    transform = EdgeSafeCropShift(max_shift=5, rng=np.random.default_rng(42))

    with pytest.raises(ValueError, match="outside observed phase bounds"):
        transform.apply_with_phase_context(
            phase_signal,
            observed_mask=observed,
            window_start=2,
            window_length=10,
            shift_samples=-5,
        )

    observed[22] = False
    with pytest.raises(ValueError, match="contains unobserved samples"):
        transform.apply_with_phase_context(
            phase_signal,
            observed_mask=observed,
            window_start=10,
            window_length=10,
            shift_samples=5,
        )


@pytest.mark.parametrize(
    "transform",
    [
        AmplitudeScaling(),
        GaussianNoiseInjection(),
        BaselineWanderInjection(),
        TimeCutMix(),
    ],
)
def test_waveform_transforms_reject_integer_dtype(transform):
    with pytest.raises(ValueError, match="floating-point waveform"):
        transform.apply(np.arange(32, dtype=np.int32))


def test_contextual_crop_shift_rejects_integer_dtype():
    signal = np.arange(32, dtype=np.int32)
    with pytest.raises(ValueError, match="floating-point waveform"):
        EdgeSafeCropShift().apply_with_phase_context(
            signal,
            observed_mask=np.ones_like(signal, dtype=bool),
            window_start=5,
            window_length=10,
            shift_samples=1,
        )


def test_forbidden_time_reversal_raises_error(example_ppg_signal):
    transform = TimeReversal()
    with pytest.raises(ValueError, match="TimeReversal is STRICTLY FORBIDDEN"):
        transform.apply(example_ppg_signal)


def test_forbidden_amplitude_inversion_raises_error(example_ppg_signal):
    transform = AmplitudeInversion()
    with pytest.raises(ValueError, match="AmplitudeInversion is STRICTLY FORBIDDEN"):
        transform.apply(example_ppg_signal)


@pytest.mark.parametrize(
    "transform",
    [
        BaselineWanderInjection(),
        EdgeSafeCropShift(),
        TimeCutMix(),
        TimeReversal(),
        AmplitudeInversion(),
    ],
)
def test_canonical_pipeline_rejects_non_allowlisted_transforms(transform):
    with pytest.raises(ValueError, match="Forbidden transform detected"):
        AugmentationPipeline(transforms=[transform])


def test_default_pipeline_is_reproducible_and_returns_provenance(example_ppg_signal):
    X = np.stack([example_ppg_signal, example_ppg_signal * 1.1])
    y = np.array([0, 1])
    participant_ids = ["ID-01", "ID-28"]
    pipeline = AugmentationPipeline(seed=42, prob=1.0)
    assert [type(transform).__name__ for transform in pipeline.transforms] == [
        "AmplitudeScaling",
        "GaussianNoiseInjection",
    ]
    first = pipeline.augment_dataset(
        X,
        y,
        multiplier=2,
        dataset_role="train",
        participant_ids=participant_ids,
        return_provenance=True,
    )
    second = AugmentationPipeline(seed=42, prob=1.0).augment_dataset(
        X,
        y,
        multiplier=2,
        dataset_role="train",
        participant_ids=participant_ids,
        return_provenance=True,
    )
    X_aug, y_aug, provenance = first

    assert len(X_aug) == len(X) * 3
    assert len(y_aug) == len(y) * 3
    assert np.array_equal(X_aug, second[0])
    assert np.array_equal(y_aug, second[1])
    assert provenance == second[2]
    assert {record["source_participant_id"] for record in provenance} == set(
        participant_ids
    )
    assert all(record["seed"] == 42 for record in provenance)
    assert all(record["dataset_role"] == "train" for record in provenance)


def test_pipeline_rejects_integer_waveforms():
    with pytest.raises(ValueError, match="floating-point dtype"):
        AugmentationPipeline().augment_dataset(
            np.arange(16, dtype=np.int32).reshape(1, -1), np.array([0])
        )


@pytest.mark.parametrize(
    "dataset_role", ["validation", "holdout", "reused_internal_test"]
)
def test_validation_and_holdout_fail_closed_without_mutation(
    example_ppg_signal, dataset_role
):
    X = np.stack([example_ppg_signal])
    y = np.array([0])
    X_before = X.copy()
    y_before = y.copy()

    with pytest.raises(ValueError, match="only for dataset_role='train'"):
        AugmentationPipeline(seed=42).augment_dataset(X, y, dataset_role=dataset_role)

    assert np.array_equal(X, X_before)
    assert np.array_equal(y, y_before)


def test_reused_internal_test_participant_is_rejected(example_ppg_signal):
    X = np.stack([example_ppg_signal])
    y = np.array([0])
    with pytest.raises(ValueError, match="ID-01..ID-28"):
        AugmentationPipeline(seed=42).augment_dataset(X, y, participant_ids=["ID-29"])


def test_training_config_locks_canonical_policy():
    config_path = Path(__file__).parents[1] / "configs" / "training.yaml"
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["augmentation"]

    assert policy["enabled"] is False
    assert policy["dataset_role"] == "train"
    assert policy["canonical_allowlist"]["amplitude_scaling"] == {
        "scale_min": 0.8,
        "scale_max": 1.2,
    }
    assert policy["canonical_allowlist"]["gaussian_noise"]["min_snr_db"] >= 20
    crop_shift = policy["contextual_candidates"]["edge_safe_crop_shift"]
    assert crop_shift["enabled"] is False
    assert crop_shift["status"] == "unavailable_until_phase_context_wired"
    assert "edge_safe_crop_shift" not in policy["canonical_allowlist"]
    assert "time_cutmix" in policy["forbidden_default"]
