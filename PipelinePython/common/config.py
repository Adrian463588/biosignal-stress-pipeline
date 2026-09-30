"""
config.py
---------
Centralized configuration constants for the data wrangling pipeline.

All magic numbers and tunable parameters live here (Single Responsibility, DRY).
Path configuration is intentionally absent — paths are owned by settings.py
(PipelinePaths / DEFAULT_PATHS) to allow runtime overrides via CLI or JSON config.
"""

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------
# Number of metadata header lines to skip in each sensor CSV
ECG_SKIP_ROWS = 13
PPG_SKIP_ROWS = 6

# Column names in order after skipping headers
# ECG:  ecg_mv | is_peak | unix_ms | datetime_str | acc_x | acc_y | acc_z
ECG_COLUMNS = [
    "ecg_mv",
    "is_peak",
    "unix_ms",
    "datetime_str",
    "acc_x",
    "acc_y",
    "acc_z",
]

# Samsung PPG exports contain three trailing positional fields, but the raw
# files provide no column header or manifest proving that they are acceleration
# axes. Preserve them for audit without assigning physiological semantics.
PPG_AUXILIARY_COLUMNS = ["ppg_aux_1", "ppg_aux_2", "ppg_aux_3"]
PPG_COLUMNS = ["ppg_raw", "unix_ms", "datetime_str", *PPG_AUXILIARY_COLUMNS]

# Filename patterns that identify each signal type (case-insensitive match)
ECG_PATTERN = "_ECG_"
PPG_GREEN_PATTERN = "PPG Green"
PPG_IR_PATTERN = "PPG IR"
PPG_RED_PATTERN = "PPG Red"

# Condition folder names used inside each participant directory
CONDITION_RELAX = "Relaks"
CONDITION_STRESS = "Stress"
CONDITIONS: dict[str, int] = {CONDITION_RELAX: 0, CONDITION_STRESS: 1}

# ---------------------------------------------------------------------------
# Sampling rates (Hz)
# ---------------------------------------------------------------------------
ECG_FS = 130.0  # native ECG sampling rate from Polar H10
PPG_FS = 25.0  # native PPG Green sampling rate
PPG_IR_FS = 100.0  # native PPG IR sampling rate
PPG_RED_FS = 100.0  # native PPG Red sampling rate
TARGET_FS = 100.0  # common target after resampling

# ---------------------------------------------------------------------------
# Signal filtering
# ---------------------------------------------------------------------------
# Butterworth bandpass — keeps 0.5–40 Hz; removes drift & HF noise
BUTTER_LOWCUT = 0.5
BUTTER_HIGHCUT = 40.0
BUTTER_ORDER = 4

# PPG-specific canonical bandpasses.  The YAML files are authoritative; these
# values are kept as safe defaults for standalone library use.
PPG_LOWCUT = 0.5
PPG_HIGHCUT = 8.0
PPG_FILTER_DEFAULTS: dict[str, dict[str, float | int | bool | str]] = {
    "ppg_ir": {
        "lowcut_hz": 0.5,
        "highcut_hz": 8.0,
        "order": 4,
        "filter_type": "butterworth",
        "zero_phase": True,
    },
    "ppg_red": {
        "lowcut_hz": 0.5,
        "highcut_hz": 8.0,
        "order": 4,
        "filter_type": "butterworth",
        "zero_phase": True,
    },
    "ppg_green": {
        "lowcut_hz": 0.5,
        "highcut_hz": 5.0,
        "order": 4,
        "filter_type": "butterworth",
        "zero_phase": True,
    },
}

# Powerline notch frequency (Indonesia = 50 Hz)
POWERLINE_FREQ = 50.0

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 180  # 3-minute segment length for HRV stress model extraction
WINDOW_STEP_SECONDS = 10  # step/overlap between windows

# Minimum number of detected peaks required per window to be included
MIN_PEAKS_PER_WINDOW = 15

# ---------------------------------------------------------------------------
# HRV Frequency bands (Task Force standards)
# ---------------------------------------------------------------------------
HRV_LF_LOW = 0.04  # Hz  (Low-Frequency band lower bound)
HRV_LF_HIGH = 0.15  # Hz  (Low-Frequency band upper bound)
HRV_HF_LOW = 0.15  # Hz  (High-Frequency band lower bound)
HRV_HF_HIGH = 0.40  # Hz  (High-Frequency band upper bound)

# ---------------------------------------------------------------------------
# Signal Quality Index (SQI) thresholds
# ---------------------------------------------------------------------------
SQI_MIN_HR_BPM = 40  # Physiological lower bound
SQI_MAX_HR_BPM = 200  # Physiological upper bound
SQI_MAX_RR_STD_MS = 400  # Max allowable SDNN (ms) — catches noise bursts
SQI_MIN_PEAKS = MIN_PEAKS_PER_WINDOW

# ---------------------------------------------------------------------------
# Feature version (bump when feature schema changes)
# ---------------------------------------------------------------------------
FEATURE_VERSION = "v4.0.0_robust_channel_independent"

# ---------------------------------------------------------------------------
# Raw-signal SQA thresholds (wrangling-time, before peak detection)
# ---------------------------------------------------------------------------
# Flatline: signal is considered flat if std < this value
SQA_FLATLINE_STD_THRESHOLD = 1e-9

# Clipping/saturation: fraction of samples at exact min or max allowed
SQA_CLIP_RATIO_THRESHOLD = 0.20

# Missing values: max fraction of NaN/non-numeric samples allowed
SQA_MISSING_RATIO_THRESHOLD = 0.10

# Timestamp gap: max fraction of intervals > 1.5x median allowed
SQA_GAP_RATIO_THRESHOLD = 0.05

# Low variance: min std required for a non-flat signal
SQA_LOW_VARIANCE_THRESHOLD = 1e-6

# Minimum segment length (samples) to run any SQA check
SQA_MIN_SAMPLES = 10

# Quality tier score thresholds (raw_score is fraction of checks passed)
QUALITY_TIER_THRESHOLDS: dict[str, float] = {
    "excellent": 0.95,  # >= 95% checks passed
    "good": 0.80,  # >= 80%
    "acceptable": 0.60,  # >= 60%
    "poor": 0.40,  # >= 40%
    # below 40% → "invalid"
}


def _config_root() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def load_cleaning_config(path: Path | None = None) -> dict:
    """Load canonical sensor filters and gap policy from local YAML."""
    selected = path or (_config_root() / "cleaning.yaml")
    if not selected.exists():
        return {key: dict(value) for key, value in PPG_FILTER_DEFAULTS.items()}
    payload = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Cleaning config must be a mapping: {selected}")
    return payload


def load_sqa_config(path: Path | None = None) -> dict:
    """Load the lightweight SQA policy from the canonical local YAML."""
    selected = path or (_config_root() / "sqa.yaml")
    if not selected.exists():
        return {"neurokit_quality_diagnostic": False}
    payload = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"SQA config must be a mapping: {selected}")
    return payload


def load_madc_config(path: Path | None = None) -> dict:
    """Load annotation-only MADC settings from local YAML."""
    selected = path or (_config_root() / "madc.yaml")
    if not selected.exists():
        return {"accelerometer_scale_status": "uncalibrated"}
    payload = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"MADC config must be a mapping: {selected}")
    return payload


def filter_config_for_channel(
    channel: str, cleaning_config: dict | None = None
) -> dict[str, float | int | bool | str]:
    """Return one canonical filter specification for a sensor channel."""
    config = cleaning_config or load_cleaning_config()
    if channel in config and isinstance(config[channel], dict):
        return dict(config[channel])
    if channel in PPG_FILTER_DEFAULTS:
        return dict(PPG_FILTER_DEFAULTS[channel])
    if channel == "ecg":
        return {
            "lowcut_hz": BUTTER_LOWCUT,
            "highcut_hz": BUTTER_HIGHCUT,
            "order": BUTTER_ORDER,
            "filter_type": "butterworth",
            "zero_phase": True,
        }
    raise ValueError(f"Unsupported signal channel: {channel}")


# ---------------------------------------------------------------------------
# Performance gates. PipelinePython/configs/training.yaml is the only Python policy source.
# Legacy aliases below are retained solely for import compatibility.
# ---------------------------------------------------------------------------
def get_tiered_performance_gates(
    training_path: Path | None = None,
) -> dict[str, dict[str, float] | str]:
    """Read and validate the canonical two-tier policy."""

    training_path = training_path or (
        Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    )
    try:
        with training_path.open("r", encoding="utf-8") as stream:
            training = yaml.safe_load(stream)
        policy = {
            "version": str(training["performance_policy_version"]),
            "minimum_research": {
                "balanced_accuracy": float(
                    training["minimum_research_balanced_accuracy"]
                ),
                "f1": float(training["minimum_research_f1"]),
                "roc_auc": float(training["minimum_research_roc_auc"]),
            },
            "preferred": {
                "balanced_accuracy": float(training["preferred_balanced_accuracy"]),
                "f1": float(training["preferred_f1"]),
                "roc_auc": float(training["preferred_roc_auc"]),
            },
        }
        if policy["minimum_research"] != {
            "balanced_accuracy": 0.70,
            "f1": 0.70,
            "roc_auc": 0.70,
        }:
            raise ValueError("minimum research gate must be 0.70/0.70/0.70")
        if policy["preferred"] != {
            "balanced_accuracy": 0.80,
            "f1": 0.78,
            "roc_auc": 0.83,
        }:
            raise ValueError("preferred gate must be 0.80/0.78/0.83")
        return policy
    except Exception as exc:
        raise RuntimeError(
            f"FATAL: Failed to validate tiered performance policy: {exc}"
        ) from exc


TIERED_PERFORMANCE_GATES = get_tiered_performance_gates()


def get_locked_balanced_accuracy_gate(_: Path | None = None) -> float:
    """Return the canonical minimum research BA gate.

    The optional path is deliberately ignored.  Reading ``frozen_gates.yaml``
    here would create a second Python source of truth and could silently make
    the old 0.80 threshold the minimum release gate again.
    """

    return float(TIERED_PERFORMANCE_GATES["minimum_research"]["balanced_accuracy"])


MINIMUM_RESEARCH_BALANCED_ACCURACY_GATE = get_locked_balanced_accuracy_gate()
MINIMUM_RESEARCH_F1_GATE = float(TIERED_PERFORMANCE_GATES["minimum_research"]["f1"])
MINIMUM_RESEARCH_ROC_AUC_GATE = float(
    TIERED_PERFORMANCE_GATES["minimum_research"]["roc_auc"]
)
PREFERRED_BALANCED_ACCURACY_GATE = float(
    TIERED_PERFORMANCE_GATES["preferred"]["balanced_accuracy"]
)
PREFERRED_F1_GATE = float(TIERED_PERFORMANCE_GATES["preferred"]["f1"])
PREFERRED_ROC_AUC_GATE = float(TIERED_PERFORMANCE_GATES["preferred"]["roc_auc"])

# Deprecated import aliases retain the preferred tier for legacy consumers.
# Canonical training must use ``PerformancePolicy.minimum_research`` instead.
MIN_BALANCED_ACCURACY_GATE = PREFERRED_BALANCED_ACCURACY_GATE
MIN_F1_GATE = PREFERRED_F1_GATE
MIN_ROC_AUC_GATE = PREFERRED_ROC_AUC_GATE
