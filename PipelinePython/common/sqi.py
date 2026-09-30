"""
sqi.py
------
Signal Quality Index (SQI) validation gate.

Single Responsibility:
    Accept or reject a feature-dictionary produced by FeatureExtractor
    based on physiological plausibility rules.

Design:
    - Pure functions; no state, no I/O.
    - Rules are centralized in _RULES (Open/Closed: add new rules without
      touching the gate logic).
    - Each rule is a callable (dict → bool) annotated with a reason string.

Usage:
    from .sqi import SignalQualityGate

    gate = SignalQualityGate()
    is_valid, reason = gate.check(feature_dict)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .config import (
    SQI_MAX_HR_BPM,
    SQI_MAX_RR_STD_MS,
    SQI_MIN_HR_BPM,
    SQI_MIN_PEAKS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule type alias
# ---------------------------------------------------------------------------

_FeatureDict = Dict[str, float | int | str]
_Rule = Tuple[str, Callable[[_FeatureDict], bool]]


# ---------------------------------------------------------------------------
# Individual SQI rules
# (Each rule returns True when the data PASSES, i.e., is physiologically valid)
# ---------------------------------------------------------------------------


def _hr_in_range(f: _FeatureDict) -> bool:
    """Heart rate must be within physiological limits."""
    hr = f.get("mean_hr_bpm")
    if hr is None:
        return False
    return SQI_MIN_HR_BPM <= float(hr) <= SQI_MAX_HR_BPM


def _sdnn_not_excessive(f: _FeatureDict) -> bool:
    """SDNN (std_rr_ms) must be within physiologically plausible range.

    A very high SDNN (> 400 ms) almost always indicates a noisy peak
    detection or signal artefact rather than genuine HRV.
    """
    sdnn = f.get("std_rr_ms")
    if sdnn is None:
        return True  # If feature absent, don't penalise (may be NaN window)
    return float(sdnn) < SQI_MAX_RR_STD_MS


def _enough_peaks(f: _FeatureDict) -> bool:
    """Minimum number of detected peaks required for reliable HRV."""
    n = f.get("num_peaks")
    if n is None:
        return False
    return int(n) >= SQI_MIN_PEAKS


def _rr_mean_physiological(f: _FeatureDict) -> bool:
    """Mean RR interval must map to a plausible HR (cross-check)."""
    rr = f.get("mean_rr_ms")
    if rr is None:
        return False
    # Equivalent HR must be in [SQI_MIN, SQI_MAX] bpm
    hr_from_rr = 60_000 / float(rr) if float(rr) > 0 else 0
    return SQI_MIN_HR_BPM <= hr_from_rr <= SQI_MAX_HR_BPM


# ---------------------------------------------------------------------------
# Registry — add new rules here (Open for Extension)
# ---------------------------------------------------------------------------

_RULES: List[_Rule] = [
    ("hr_not_in_physiological_range", _hr_in_range),
    ("sdnn_excessive_noise", _sdnn_not_excessive),
    ("insufficient_peaks", _enough_peaks),
    ("mean_rr_non_physiological", _rr_mean_physiological),
]


# ---------------------------------------------------------------------------
# SignalQualityGate
# ---------------------------------------------------------------------------


class SignalQualityGate:
    """
    Evaluates a feature dictionary against a set of physiological rules.

    Usage:
        gate = SignalQualityGate()
        passed, reason = gate.check(feature_dict)
        if not passed:
            # discard this window
    """

    def check(self, features: _FeatureDict) -> Tuple[bool, Optional[str]]:
        """
        Run all SQI rules against *features*.

        Args:
            features: Dictionary produced by FeatureExtractor.extract().

        Returns:
            (True, None)           — all rules passed; window is clean.
            (False, reason_str)    — a rule failed; window should be discarded.
        """
        for reason, rule in _RULES:
            if not rule(features):
                logger.debug(
                    "SQI FAIL [%s] — participant=%s, window=%s, hr=%.1f, sdnn=%.1f",
                    reason,
                    features.get("participant", "?"),
                    features.get("window_idx", "?"),
                    features.get("mean_hr_bpm", float("nan")),
                    features.get("std_rr_ms", float("nan")),
                )
                return False, reason
        return True, None


# ---------------------------------------------------------------------------
# RawSignalSQA — wrangling-time quality assessment on raw signal arrays
# ---------------------------------------------------------------------------

import numpy as _np  # noqa: E402

from .config import (  # noqa: E402
    QUALITY_TIER_THRESHOLDS,
    SQA_CLIP_RATIO_THRESHOLD,
    SQA_FLATLINE_STD_THRESHOLD,
    SQA_GAP_RATIO_THRESHOLD,
    SQA_LOW_VARIANCE_THRESHOLD,
    SQA_MIN_SAMPLES,
    SQA_MISSING_RATIO_THRESHOLD,
)


class RawSignalSQA:
    """
    Wrangling-time Signal Quality Assessment on raw signal arrays.

    Runs on the cleaned waveform BEFORE peak detection and BEFORE the silver
    layer is written.  NeuroKit2 quality is diagnostic-only: it can annotate
    a window, but it cannot turn an otherwise usable signal into ``invalid``.
    Conforms to AGENTS.md §10 and PRD §9.

    Usage:
        sqa = RawSignalSQA()
        score, tier, flags = sqa.score(values, timestamps_utc=ts)
        # tier in {"excellent", "good", "acceptable", "poor", "invalid"}
    """

    def __init__(self, *, diagnostic_neurokit: bool = True) -> None:
        # NeuroKit2 template matching is useful for diagnostics but is much
        # more expensive than the hard physiological checks. The canonical
        # lightweight rerun can disable it without changing clean_value.
        self.diagnostic_neurokit = diagnostic_neurokit

    def score(
        self,
        values: "pd.Series | _np.ndarray",
        timestamps_utc: "pd.Series | None" = None,
        *,
        cleaned_values: "pd.Series | _np.ndarray | None" = None,
        channel: str = "ppg",
        sampling_rate_hz: float | None = None,
        interpolation_mask: "pd.Series | _np.ndarray | None" = None,
    ) -> Tuple[float, str, List[str]]:
        """
        Compute quality score, tier, and flag list for a raw signal segment.

        Args:
            values:          1-D numeric series / array of raw signal samples.
            timestamps_utc:  Optional UTC timestamp series for gap/jitter checks.

        Returns:
            (quality_score, quality_tier, flags)
            quality_score : float [0.0 – 1.0], fraction of checks passed.
            quality_tier  : str — "excellent" | "good" | "acceptable" | "poor" | "invalid"
            flags         : list of failed check names (empty = all passed).
        """
        import pandas as pd  # local import keeps module-level deps clean

        vals = pd.Series(pd.to_numeric(values, errors="coerce"))
        quality_values = pd.Series(
            pd.to_numeric(
                cleaned_values if cleaned_values is not None else values,
                errors="coerce",
            )
        ).replace([_np.inf, -_np.inf], _np.nan)
        flags: List[str] = []
        total_checks = 8
        hard_flags = {
            "unresolved_nonfinite",
            "all_missing",
            "flatline",
            "clipping",
            "long_dropout",
        }

        # ── 1. Minimum length ──────────────────────────────────────────────
        if len(vals) < SQA_MIN_SAMPLES:
            return 0.0, "invalid", ["too_short"]

        # ── 2. Missing ratio ───────────────────────────────────────────────
        # The cleaned signal is authoritative for this gate.  A bounded gap
        # may remain NaN in the raw column while its clean value is finite and
        # carries an interpolation mask/provenance.
        missing_ratio = float(quality_values.isna().mean())
        if missing_ratio > SQA_MISSING_RATIO_THRESHOLD:
            flags.append("unresolved_nonfinite")

        clean = quality_values.replace([_np.inf, -_np.inf], _np.nan).dropna()
        if clean.empty:
            return 0.0, "invalid", ["all_missing"]

        if interpolation_mask is not None:
            mask = _np.asarray(interpolation_mask, dtype=bool)
            if mask.size == len(quality_values) and mask.any():
                flags.append("short_gap_interpolated")

        # ── 3. Flatline detection ──────────────────────────────────────────
        if float(clean.std()) < SQA_FLATLINE_STD_THRESHOLD:
            flags.append("flatline")

        # ── 4. Clipping / saturation ───────────────────────────────────────
        vmin, vmax = float(clean.min()), float(clean.max())
        if vmax > vmin:  # skip if flatline already flagged
            sat_ratio = float(((clean == vmin) | (clean == vmax)).mean())
            if sat_ratio > SQA_CLIP_RATIO_THRESHOLD:
                flags.append("clipping")

        # ── 5. Low variance (non-flat but near-constant) ───────────────────
        if float(clean.std()) < SQA_LOW_VARIANCE_THRESHOLD and "flatline" not in flags:
            flags.append("low_variance")

        # ── 6. Extreme amplitude jumps (artifact proxy) ────────────────────
        if len(clean) > 5:
            diffs = clean.diff().dropna().abs()
            q99 = float(diffs.quantile(0.99))
            q50 = float(diffs.quantile(0.50))
            if q50 > 0 and q99 > 20 * q50:
                flags.append("extreme_jumps")

        # ── 7. Timestamp gap / jitter ──────────────────────────────────────
        ts = pd.Series(dtype="datetime64[ns, UTC]")
        median = 0.0
        if timestamps_utc is not None:
            ts = (
                pd.Series(pd.to_datetime(timestamps_utc, utc=True, errors="coerce"))
                .dropna()
                .sort_values()
            )
            if len(ts) > 5:
                diffs_ms = ts.diff().dropna().dt.total_seconds() * 1000.0
                median = float(diffs_ms.median())
                if median > 0:
                    gap_ratio = float((diffs_ms > 1.5 * median).mean())
                    # The cleaning contract permits only bounded internal
                    # gaps up to 0.5 s.  A larger gap is a hard dropout, while
                    # ordinary jitter remains a soft diagnostic flag.
                    if bool((diffs_ms > max(500.0, 3.0 * median)).any()):
                        flags.append("long_dropout")
                    if gap_ratio > SQA_GAP_RATIO_THRESHOLD:
                        flags.append("timestamp_gap")

        # ── 8. NeuroKit2 PPG quality on the cleaned waveform ────────────────
        if (
            self.diagnostic_neurokit
            and channel.startswith("ppg_")
            and "flatline" not in flags
            and len(clean) > 50
        ):
            import neurokit2 as nk

            try:
                if sampling_rate_hz is not None and sampling_rate_hz > 0:
                    fs_est = float(sampling_rate_hz)
                elif timestamps_utc is not None and len(ts) > 5:
                    fs_est = 1000.0 / median if median > 0 else 100.0
                else:
                    fs_est = 100.0
                if not _np.isfinite(fs_est) or fs_est <= 0:
                    raise ValueError("sampling_rate_unavailable")
                quality_input = clean.to_numpy(dtype=float)
                quality = nk.ppg_quality(
                    quality_input,
                    sampling_rate=max(1, int(round(fs_est))),
                    method="templatematch",
                )
                mean_quality = float(_np.nanmean(quality))
                if not _np.isfinite(mean_quality):
                    raise ValueError("neurokit_quality_nonfinite")
                if mean_quality < 0.5:
                    flags.append("nk_low_quality")
            except Exception as exc:
                logger.warning("NeuroKit2 PPG quality failed: %s", exc)
                flags.append(f"ppg_quality_error:{type(exc).__name__}")

        if any(flag in hard_flags for flag in flags):
            return 0.0, "invalid", flags

        passed = total_checks - len(flags)
        quality_score = round(max(0.0, passed / total_checks), 4)
        quality_tier = self._score_to_tier(quality_score)

        logger.debug(
            "RawSignalSQA: score=%.2f tier=%s flags=%s",
            quality_score,
            quality_tier,
            flags,
        )
        return quality_score, quality_tier, flags

    @staticmethod
    def _score_to_tier(score: float) -> str:
        for tier, threshold in QUALITY_TIER_THRESHOLDS.items():
            if score >= threshold:
                return tier
        return "invalid"


@dataclass(frozen=True)
class PrePeakQuality:
    """Window-level quality result before peak detection."""

    score: float
    tier: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PostPeakQuality:
    """Window-level quality result after peak/IBI plausibility checks."""

    score: float
    tier: str
    flags: tuple[str, ...]
    peak_count: int
    physiological_ibi_ratio: float


class PrePeakSQA:
    """Assess a cleaned PPG window before peak detection."""

    def __init__(self, signal_sqa: RawSignalSQA | None = None) -> None:
        self._signal_sqa = signal_sqa or RawSignalSQA()

    def assess(
        self,
        clean_window: Sequence[float],
        *,
        timestamps_utc: Sequence[object] | None = None,
        channel: str = "ppg",
        sampling_rate_hz: float | None = None,
        interpolation_mask: Sequence[bool] | None = None,
    ) -> PrePeakQuality:
        score, tier, flags = self._signal_sqa.score(
            clean_window,
            timestamps_utc,
            cleaned_values=clean_window,
            channel=channel,
            sampling_rate_hz=sampling_rate_hz,
            interpolation_mask=interpolation_mask,
        )
        return PrePeakQuality(score, tier, tuple(flags))


class PostPeakSQA:
    """Apply peak count and IBI plausibility checks to a cleaned window."""

    def __init__(
        self,
        signal_sqa: RawSignalSQA | None = None,
        *,
        minimum_peak_count: int = SQI_MIN_PEAKS,
        minimum_physiological_ibi_ratio: float = 0.90,
        physiological_ibi_min_ms: float = 300.0,
        physiological_ibi_max_ms: float = 1500.0,
    ) -> None:
        self._signal_sqa = signal_sqa or RawSignalSQA()
        self._minimum_peak_count = minimum_peak_count
        self._minimum_physiological_ibi_ratio = minimum_physiological_ibi_ratio
        self._physiological_ibi_min_ms = physiological_ibi_min_ms
        self._physiological_ibi_max_ms = physiological_ibi_max_ms

    def assess(
        self,
        clean_window: Sequence[float],
        peaks: Sequence[int],
        *,
        sampling_rate_hz: float,
        timestamps_utc: Sequence[object] | None = None,
        interpolation_mask: Sequence[bool] | None = None,
        pre_quality: PrePeakQuality | None = None,
    ) -> PostPeakQuality:
        pre = pre_quality or PrePeakSQA(self._signal_sqa).assess(
            clean_window,
            timestamps_utc=timestamps_utc,
            channel="ppg",
            sampling_rate_hz=sampling_rate_hz,
            interpolation_mask=interpolation_mask,
        )
        flags = list(pre.flags)
        peak_indices = _np.asarray(peaks, dtype=int)
        peak_count = int(peak_indices.size)
        if peak_count < self._minimum_peak_count:
            flags.append("insufficient_peaks")
        if peak_count > 1:
            ibi_ms = self._observed_ibi_ms(
                peak_indices,
                sampling_rate_hz,
                timestamps_utc,
            )
            physiological = (ibi_ms >= self._physiological_ibi_min_ms) & (
                ibi_ms <= self._physiological_ibi_max_ms
            )
            ibi_ratio = float(_np.mean(physiological))
            if ibi_ratio < self._minimum_physiological_ibi_ratio:
                flags.append("implausible_ibi_ratio")
        else:
            ibi_ratio = 0.0
            flags.append("insufficient_ibi")
        # Post-peak checks are hard only when there is no usable physiological
        # beat sequence.  Soft diagnostics inherited from the clean waveform
        # remain visible without invalidating an otherwise usable window.
        hard_peak_failure = (
            peak_count < self._minimum_peak_count
            or ibi_ratio < self._minimum_physiological_ibi_ratio
        )
        if hard_peak_failure:
            if peak_count < self._minimum_peak_count:
                flags.append("insufficient_peaks")
            return PostPeakQuality(
                0.0, "invalid", tuple(dict.fromkeys(flags)), peak_count, ibi_ratio
            )
        score = max(0.0, min(1.0, pre.score - 0.10 * len(set(flags) - set(pre.flags))))
        tier = RawSignalSQA._score_to_tier(score)
        return PostPeakQuality(score, tier, tuple(flags), peak_count, ibi_ratio)

    @staticmethod
    def _observed_ibi_ms(
        peak_indices: _np.ndarray,
        sampling_rate_hz: float,
        timestamps_utc: Sequence[object] | None,
    ) -> _np.ndarray:
        """Calculate IBI from observed peak timestamps when available.

        The sample-rate product is retained only as a compatibility fallback
        for callers that do not provide timestamps.  Canonical validation
        passes timestamps, so timestamp jitter and short gaps are not silently
        converted into uniform synthetic time.
        """

        if timestamps_utc is not None:
            import pandas as pd

            timestamps = pd.to_datetime(
                pd.Series(timestamps_utc), utc=True, errors="coerce"
            )
            if (
                peak_indices.size > 1
                and int(peak_indices.min()) >= 0
                and int(peak_indices.max()) < len(timestamps)
            ):
                selected = timestamps.iloc[peak_indices]
                if selected.notna().all():
                    return (
                        selected.astype("int64").to_numpy(dtype=float) / 1_000_000.0
                    )[1:] - (
                        selected.astype("int64").to_numpy(dtype=float) / 1_000_000.0
                    )[:-1]
        return _np.diff(peak_indices).astype(float) * 1000.0 / sampling_rate_hz


if __name__ == "__main__":
    from PipelinePython.common.cli import main

    main(["sqa"])
