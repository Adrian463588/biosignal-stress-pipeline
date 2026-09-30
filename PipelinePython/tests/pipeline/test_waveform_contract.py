from __future__ import annotations

import numpy as np
import pytest

from PipelinePython.common.local_pipeline import _normalize_observed_waveform_batch
from PipelinePython.common.sqi import RawSignalSQA


def test_waveform_normalization_uses_observed_samples_only() -> None:
    values = np.asarray([[1.0, 2.0, 100.0, 4.0]], dtype=np.float32)
    interpolated_mask = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)

    normalized, observed_mask, means, stds = _normalize_observed_waveform_batch(
        values,
        interpolated_mask,
    )

    expected_mean = np.asarray([[7.0 / 3.0]], dtype=np.float32)
    expected_std = np.asarray(
        [
            [
                np.sqrt(
                    (
                        (1.0 - 7.0 / 3.0) ** 2
                        + (2.0 - 7.0 / 3.0) ** 2
                        + (4.0 - 7.0 / 3.0) ** 2
                    )
                    / 3.0
                )
            ]
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(observed_mask, [[1, 1, 0, 1]])
    np.testing.assert_allclose(means, expected_mean)
    np.testing.assert_allclose(stds, expected_std)
    assert np.isfinite(normalized).all()
    assert normalized[0, 2] > 50.0


def test_waveform_normalization_rejects_missing_observed_samples() -> None:
    values = np.asarray([[1.0, 2.0]], dtype=np.float32)
    interpolated_mask = np.ones((1, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="no_observed"):
        _normalize_observed_waveform_batch(values, interpolated_mask)


def test_waveform_normalization_rejects_nonfinite_values() -> None:
    values = np.asarray([[1.0, np.inf]], dtype=np.float32)
    interpolated_mask = np.zeros((1, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        _normalize_observed_waveform_batch(values, interpolated_mask)


def test_raw_sqa_counts_infinite_samples_as_unresolved_nonfinite() -> None:
    values = np.ones(100, dtype=np.float32)
    values[:11] = np.inf

    _, tier, flags = RawSignalSQA(diagnostic_neurokit=False).score(values)

    assert tier == "invalid"
    assert "unresolved_nonfinite" in flags
