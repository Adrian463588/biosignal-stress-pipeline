from __future__ import annotations

import numpy as np
import pytest

from PipelinePython.common.feature_policy import DEFAULT_PREDICTORS
from PipelinePython.common.features_v4 import extract_ppg_features, feature_order
from PipelinePython.common.peak_detection import PpgPeakDetection


def _synthetic_window(seconds: int, sampling_rate_hz: int = 25):
    time = np.arange(seconds * sampling_rate_hz, dtype=np.float64) / sampling_rate_hz
    signal = np.sin(2.0 * np.pi * time)
    peaks = np.arange(
        sampling_rate_hz // 4,
        signal.size,
        sampling_rate_hz,
        dtype=np.int64,
    )
    troughs = np.arange(
        3 * sampling_rate_hz // 4,
        signal.size,
        sampling_rate_hz,
        dtype=np.int64,
    )
    detection = PpgPeakDetection(peaks, troughs, "fixture", "", 1.0)
    return signal, detection


@pytest.mark.parametrize(
    ("seconds", "sequence_length"),
    ((60, 240), (120, 480), (180, 720), (240, 960)),
)
def test_sequence_contract_follows_selected_window(
    seconds: int, sequence_length: int
) -> None:
    signal, detection = _synthetic_window(seconds)

    features, sequence, mask = extract_ppg_features(
        signal,
        25.0,
        detection=detection,
        window_seconds=seconds,
    )

    assert feature_order([features]) == DEFAULT_PREDICTORS
    assert sequence.shape == mask.shape == (sequence_length,)
    assert mask.any()
    assert not {"sample_entropy", "lf_hf_ratio"} & set(features)


def test_window_duration_is_enforced_without_padding() -> None:
    signal, detection = _synthetic_window(60)

    with pytest.raises(ValueError, match="insufficient_ppg_window"):
        extract_ppg_features(
            signal,
            25.0,
            detection=detection,
            window_seconds=120,
        )
