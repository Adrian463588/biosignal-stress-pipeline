from __future__ import annotations

import numpy as np
import pytest

from PipelinePython.common.peak_detection import PpgPeakDetector
from PipelinePython.common.settings import load_pipeline_config


def _detector() -> PpgPeakDetector:
    return PpgPeakDetector(load_pipeline_config().peak_detection)


def test_candidate_registry_exposes_explicit_methods() -> None:
    assert _detector().candidate_methods() == (
        "neurokit2_ppg_elgendi",
        "neurokit2_ppg_bishop",
        "neurokit2_ppg_charlton_msptdfast",
        "scipy_find_peaks_adaptive",
        "scipy_legacy_green_native",
    )


@pytest.mark.parametrize(
    ("method", "neurokit_method"),
    [
        ("neurokit2_ppg_elgendi", "elgendi"),
        ("neurokit2_ppg_bishop", "bishop"),
        ("neurokit2_ppg_charlton_msptdfast", "charlton"),
    ],
)
def test_neurokit_candidates_keep_actual_sampling_rate(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    neurokit_method: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_ppg_findpeaks(
        values: np.ndarray, *, sampling_rate: float, method: str
    ) -> dict[str, np.ndarray]:
        captured.update(
            values=values,
            sampling_rate=sampling_rate,
            method=method,
        )
        return {"PPG_Peaks": np.arange(20, 420, 20)}

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", fake_ppg_findpeaks
    )

    signal = np.sin(np.arange(600, dtype=float) / 10.0)
    sampling_rate = 40.0
    result = _detector().evaluate_candidate(signal, sampling_rate, "ppg_green", method)

    assert result.status == "AVAILABLE"
    assert result.method == method
    assert captured["sampling_rate"] == sampling_rate
    assert captured["method"] == neurokit_method
    assert np.shares_memory(captured["values"], signal)


def test_msptdfast_rounds_near_integer_api_rate_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_ppg_findpeaks(
        values: np.ndarray, *, sampling_rate: float, method: str
    ) -> dict[str, np.ndarray]:
        captured.update(values=values, sampling_rate=sampling_rate, method=method)
        return {"PPG_Peaks": np.arange(20, 420, 20)}

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", fake_ppg_findpeaks
    )
    signal = np.sin(np.arange(600, dtype=float) / 10.0)

    result = _detector().evaluate_candidate(
        signal,
        24.9189,
        "ppg_green",
        "neurokit2_ppg_charlton_msptdfast",
    )

    assert result.status == "AVAILABLE"
    assert captured["sampling_rate"] == 25
    assert captured["method"] == "charlton"
    assert np.shares_memory(captured["values"], signal)


def test_msptdfast_rejects_rate_not_close_to_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def should_not_run(*_: object, **__: object) -> dict[str, np.ndarray]:
        nonlocal called
        called = True
        return {"PPG_Peaks": np.array([], dtype=np.int64)}

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", should_not_run
    )

    result = _detector().evaluate_candidate(
        np.sin(np.arange(600, dtype=float) / 10.0),
        24.4,
        "ppg_green",
        "neurokit2_ppg_charlton_msptdfast",
    )

    assert result.status == "UNAVAILABLE"
    assert "near the observed rate" in result.status_reason
    assert not called


def test_adaptive_scipy_uses_channel_config_and_actual_sampling_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_find_peaks(
        values: np.ndarray, **kwargs: object
    ) -> tuple[np.ndarray, dict]:
        calls.append({"values": values, **kwargs})
        return np.array([20, 80, 140], dtype=np.int64), {}

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.find_peaks", fake_find_peaks
    )

    sampling_rate = 37.5
    signal = np.sin(np.arange(600, dtype=float) / 10.0)
    result = _detector().evaluate_candidate(
        signal, sampling_rate, "ppg_ir", "scipy_find_peaks_adaptive"
    )
    channel_config = _detector().config.for_channel("ppg_ir")

    assert result.status == "AVAILABLE"
    assert result.method == "scipy_find_peaks_adaptive"
    assert len(calls) == 2  # one peak pass, one trough pass
    assert all(
        call["distance"] == round(channel_config.min_distance_seconds * sampling_rate)
        for call in calls
    )
    assert all(np.isfinite(call["values"]).all() for call in calls)


def test_unavailable_candidate_is_explicit_and_does_not_fabricate_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(*_: object, **__: object) -> dict[str, np.ndarray]:
        raise ValueError("method not found")

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", unsupported
    )

    result = _detector().evaluate_candidate(
        np.sin(np.arange(600, dtype=float) / 10.0),
        40.0,
        "ppg_red",
        "neurokit2_ppg_charlton_msptdfast",
    )

    assert result.status == "UNAVAILABLE"
    assert result.peaks.size == 0
    assert result.troughs.size == 0
    assert "method not found" in result.status_reason


def test_non_finite_signal_is_rejected_before_any_candidate_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def should_not_run(*_: object, **__: object) -> dict[str, np.ndarray]:
        nonlocal called
        called = True
        return {"PPG_Peaks": np.array([], dtype=np.int64)}

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", should_not_run
    )

    with pytest.raises(ValueError, match="non-finite"):
        _detector().evaluate_candidate(
            np.array([0.0, np.nan] + [0.0] * 598),
            37.5,
            "ppg_green",
            "neurokit2_ppg_elgendi",
        )

    assert not called
