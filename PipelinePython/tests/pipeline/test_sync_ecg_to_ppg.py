from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PipelinePython.common.ingestion import _load_ecg

from importlib.util import module_from_spec, spec_from_file_location

_SYNC_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "02_peak_to_peak"
    / "tools"
    / "sync_ecg_to_ppg.py"
)
_SYNC_SPEC = spec_from_file_location("pipeline_sync_ecg_to_ppg", _SYNC_MODULE_PATH)
if _SYNC_SPEC is None or _SYNC_SPEC.loader is None:
    raise ImportError(f"Unable to load {_SYNC_MODULE_PATH}")
_SYNC_MODULE = module_from_spec(_SYNC_SPEC)
sys.modules[_SYNC_SPEC.name] = _SYNC_MODULE
_SYNC_SPEC.loader.exec_module(_SYNC_MODULE)

PeakSeries = _SYNC_MODULE.PeakSeries
SyncCandidate = _SYNC_MODULE.SyncCandidate
_append_candidate_detail = _SYNC_MODULE._append_candidate_detail
choose_best_candidate = _SYNC_MODULE.choose_best_candidate
coarse_shift_to_ppg = _SYNC_MODULE.coarse_shift_to_ppg
estimate_fine_lag_seconds = _SYNC_MODULE.estimate_fine_lag_seconds
regularize_ecg_timestamps = _SYNC_MODULE.regularize_ecg_timestamps
write_synced_ecg = _SYNC_MODULE.write_synced_ecg


def test_raw_ecg_parser_keeps_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "ID-01Stress_ECG.csv"
    path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=ECG\n"
        "samplingrate=130.0\n"
        "0.10,1,1000,21 Apr 2026 09:00:00:000 +0700,1,2,3\n"
        "0.20,0,1008,21 Apr 2026 09:00:00:008 +0700,4,5,6\n",
        encoding="utf-8",
    )

    df = _load_ecg(path)

    assert list(df.columns) == [
        "ecg_mv",
        "is_peak",
        "unix_ms",
        "datetime_str",
        "acc_x",
        "acc_y",
        "acc_z",
    ]
    assert len(df) == 2
    assert df.attrs["sampling_rate_hz"] == 125.0


def test_write_synced_ecg_preserves_row_count_and_shifts_time(tmp_path: Path) -> None:
    ecg = pd.DataFrame(
        {
            "ecg_mv": [0.1, 0.2],
            "is_peak": [1, 0],
            "unix_ms": [1000, 1008],
            "datetime_str": ["old", "old"],
            "acc_x": [1, 2],
            "acc_y": [3, 4],
            "acc_z": [5, 6],
        }
    )

    out = write_synced_ecg(ecg, tmp_path / "ecg_synced.csv", total_shift_ms=5000)

    assert len(out) == len(ecg)
    assert out["unix_ms"].tolist() == [6000, 6008]
    assert out["ecg_mv"].tolist() == ecg["ecg_mv"].tolist()
    assert (tmp_path / "ecg_synced.csv").exists()


def test_coarse_shift_aligns_non_overlapping_ecg_to_ppg_start() -> None:
    ecg = pd.DataFrame({"unix_ms": [1000, 2000], "ecg_mv": [0.1, 0.2]})
    ppg = pd.DataFrame({"unix_ms": [10_000, 11_000], "ppg_raw": [1.0, 2.0]})

    assert coarse_shift_to_ppg(ecg, [ppg]) == 9000


def test_regularize_ecg_timestamps_rebuilds_from_sampling_rate() -> None:
    ecg = pd.DataFrame({"unix_ms": [1000, 5000, 9000], "ecg_mv": [0.1, 0.2, 0.3]})
    ecg.attrs["sampling_rate_hz"] = 100.0

    regular = regularize_ecg_timestamps(ecg)

    assert regular["unix_ms"].tolist() == [1000, 1010, 1020]
    assert regular["ecg_mv"].tolist() == ecg["ecg_mv"].tolist()


def test_estimate_fine_lag_recovers_known_lag() -> None:
    ecg_t = np.arange(10, 80, 1.0)
    ibi = 800 + 80 * np.sin(ecg_t / 5.0)
    known_lag = 2.5
    ecg = PeakSeries(
        peak_times_ms=np.array([0, 1000, 2000, 3000]),
        ibi_times_s=ecg_t,
        ibi_ms=ibi,
        clean_signal=np.zeros(10),
        peak_indices=np.array([1, 2, 3]),
    )
    ppg = PeakSeries(
        peak_times_ms=np.array([0, 1000, 2000, 3000]),
        ibi_times_s=ecg_t + known_lag,
        ibi_ms=ibi,
        clean_signal=np.zeros(10),
        peak_indices=np.array([1, 2, 3]),
    )

    lag, corr, before = estimate_fine_lag_seconds(ecg, ppg, max_lag_s=5, step_s=0.05)

    assert abs(lag - known_lag) <= 0.05
    assert corr > 0.99
    assert before < corr


def test_choose_best_candidate_prefers_high_score() -> None:
    weaker = SyncCandidate(
        "ppg_green", "ok", 0.0, 0.1, 0.7, 0.7, 100, 100, 1.0, "acceptable", ""
    )
    stronger = SyncCandidate(
        "ppg_ir", "ok", 0.0, 0.1, 0.9, 0.9, 100, 95, 0.95, "good", ""
    )
    failed = SyncCandidate(
        "ppg_red",
        "failed",
        0.0,
        np.nan,
        np.nan,
        np.nan,
        100,
        0,
        np.nan,
        "failed",
        "bad",
    )

    assert choose_best_candidate([weaker, stronger, failed]).channel == "ppg_ir"


def test_candidate_detail_writer_preserves_lineage(tmp_path: Path) -> None:
    candidate = SyncCandidate(
        "ppg_ir", "ok", 1.0, 0.1, 0.9, 0.9, 100, 95, 0.95, "good", ""
    )

    _append_candidate_detail(tmp_path, "ID-01", "Stress", candidate)
    detail = pd.read_csv(tmp_path / "sync_candidate_details.csv")

    assert detail.loc[0, "participant_code"] == "ID-01"
    assert detail.loc[0, "phase"] == "Stress"
    assert detail.loc[0, "candidate_channel"] == "ppg_ir"
    assert detail.loc[0, "sync_quality"] == "good"
