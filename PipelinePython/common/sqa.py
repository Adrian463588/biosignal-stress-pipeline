"""
sqa.py
------
Signal Quality Assessment (SQA) module.

Conforms to AGENTS.md §9 Tahap E (SQA/SQI).
Implements a two-layer SQA approach:
1. Pre-peak SQA: flatline, clipping, missing ratio, sampling jitter, motion score.
2. Post-peak SQA: plausible beat count, IBI outlier ratio, morphology consistency.

Assigns a quality tier: `excellent`, `good`, `acceptable`, `poor`, `invalid`.
Discards `poor` and `invalid` from training.

Design:
    - Clean Architecture / RTK AI Caveman guidelines.
    - Pure, single-responsibility functions/classes.
    - No direct side effects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_sqa_config() -> dict[str, Any]:
    """Load SQA configuration from PipelinePython/configs/sqa.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "sqa.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class PrePeakSQA:
    """
    Layer 1: Pre-peak SQA.
    Validates signal quality BEFORE peak detection.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or load_sqa_config()
        self.tiers = self.config.get(
            "tiers",
            {
                "excellent": 0.95,
                "good": 0.80,
                "acceptable": 0.60,
                "poor": 0.40,
            },
        )

    def evaluate(
        self,
        values: pd.Series | np.ndarray,
        timestamps_utc: Optional[pd.Series | np.ndarray] = None,
        motion_score: float = 0.0,
    ) -> Tuple[float, str, List[str]]:
        """
        Evaluate raw signal array.

        Args:
            values: Numeric array of raw signal.
            timestamps_utc: Corresponding timestamps for jitter/gap checks.
            motion_score: External motion score from MADC (0.0 to 1.0).

        Returns:
            (score, tier, flags)
        """
        vals = pd.to_numeric(pd.Series(values), errors="coerce")
        flags: List[str] = []
        total_checks = 5

        # 1. Missing ratio
        missing_ratio = float(vals.isna().mean())
        if missing_ratio > self.config.get("missing_ratio_max", 0.10):
            flags.append("high_missing")

        clean = vals.dropna()
        if len(clean) < self.config.get("minimum_samples", 10):
            return 0.0, "invalid", ["too_short"]

        # 2. Flatline
        if float(clean.std()) < self.config.get("flatline_std_max", 1e-9):
            flags.append("flatline")

        # 3. Clipping / saturation
        vmin, vmax = float(clean.min()), float(clean.max())
        if vmax > vmin:
            sat_ratio = float(((clean == vmin) | (clean == vmax)).mean())
            if sat_ratio > self.config.get("clip_ratio_max", 0.20):
                flags.append("clipping")

        # 4. Sampling Jitter / Timestamp gaps
        if timestamps_utc is not None:
            ts = (
                pd.to_datetime(pd.Series(timestamps_utc), utc=True, errors="coerce")
                .dropna()
                .sort_values()
            )
            if len(ts) > 5:
                diffs_ms = ts.diff().dropna().dt.total_seconds() * 1000.0
                median = float(diffs_ms.median())
                if median > 0:
                    gap_ratio = float((diffs_ms > 1.5 * median).mean())
                    # Using timestamp_gap_ratio_max or sampling_jitter_cv_max
                    if gap_ratio > self.config.get("timestamp_gap_ratio_max", 0.05):
                        flags.append("sampling_jitter")

        # 5. Motion Score
        if motion_score > self.config.get("artifact_ratio_max", 0.20):
            flags.append("high_motion_score")

        passed = total_checks - len(flags)
        quality_score = float(max(0.0, passed / total_checks))
        quality_tier = self._score_to_tier(quality_score)

        return quality_score, quality_tier, flags

    def _score_to_tier(self, score: float) -> str:
        sorted_tiers = sorted(self.tiers.items(), key=lambda x: x[1], reverse=True)
        for tier, threshold in sorted_tiers:
            if score >= threshold:
                return tier
        return "invalid"


class PostPeakSQA:
    """
    Layer 2: Post-peak SQA.
    Validates signal quality AFTER peak detection based on IBI and morphology.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or load_sqa_config()
        self.tiers = self.config.get(
            "tiers",
            {
                "excellent": 0.95,
                "good": 0.80,
                "acceptable": 0.60,
                "poor": 0.40,
            },
        )

    def evaluate(
        self,
        peaks_idx: np.ndarray | List[int],
        ibis_ms: np.ndarray | List[float],
        morphology_score: float = 1.0,
    ) -> Tuple[float, str, List[str]]:
        """
        Evaluate extracted peaks and IBIs.

        Args:
            peaks_idx: Array of peak indices.
            ibis_ms: Array of inter-beat intervals in milliseconds.
            morphology_score: Consistency of pulse morphology (0.0 to 1.0).

        Returns:
            (score, tier, flags)
        """
        flags: List[str] = []
        total_checks = 3

        ibis = np.array(ibis_ms, dtype=float)

        # 1. Plausible beat count
        if len(peaks_idx) < self.config.get("min_peaks", 15):
            flags.append("insufficient_beats")

        # 2. IBI outlier ratio
        if len(ibis) > 0:
            median_ibi = float(np.median(ibis))
            if median_ibi > 0:
                # Basic outlier definition: outside 0.7x to 1.3x median IBI
                outliers = np.sum((ibis < 0.7 * median_ibi) | (ibis > 1.3 * median_ibi))
                outlier_ratio = float(outliers / len(ibis))
                if outlier_ratio > self.config.get("ibi_outlier_ratio_max", 0.20):
                    flags.append("high_ibi_outliers")
            else:
                flags.append("invalid_ibi_median")
        else:
            flags.append("no_ibis")

        # 3. Morphology consistency
        if morphology_score < self.config.get("morphology_min_score", 0.60):
            flags.append("poor_morphology")

        passed = total_checks - len(flags)
        quality_score = float(max(0.0, passed / total_checks))
        quality_tier = self._score_to_tier(quality_score)

        return quality_score, quality_tier, flags

    def _score_to_tier(self, score: float) -> str:
        sorted_tiers = sorted(self.tiers.items(), key=lambda x: x[1], reverse=True)
        for tier, threshold in sorted_tiers:
            if score >= threshold:
                return tier
        return "invalid"


def is_usable_for_training(tier: str) -> bool:
    """
    Gate for ML training.
    Discards `poor` and `invalid` from training.
    """
    return tier not in ("poor", "invalid")
