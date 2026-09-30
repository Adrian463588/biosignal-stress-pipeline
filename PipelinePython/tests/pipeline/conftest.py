"""
conftest.py
-----------
Shared pytest fixtures for the pipeline test suite.

Fixtures provided:
  - tmp_anchor_csv   : a minimal, valid anchor CSV file in a tmp_path directory
  - sample_bronze_df : a minimal bronze-layer DataFrame covering one segment
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Anchor CSV fixture
# ---------------------------------------------------------------------------

ANCHOR_CSV_CONTENT = (
    "participant_code,phase,start_timestamp,end_timestamp\n"
    "TEST01,Relax,2026-04-23T09:45:00+07:00,2026-04-23T09:46:00+07:00\n"
    "TEST01,Task,2026-04-23T10:00:00+07:00,2026-04-23T10:01:00+07:00\n"
)


@pytest.fixture()
def tmp_anchor_csv(tmp_path: Path) -> Path:
    """Return the path to a minimal, valid anchor CSV file."""
    csv = tmp_path / "anchors.csv"
    csv.write_text(ANCHOR_CSV_CONTENT, encoding="utf-8")
    return csv


# ---------------------------------------------------------------------------
# Bronze DataFrame fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_bronze_df() -> pd.DataFrame:
    """Return a minimal bronze-layer DataFrame with 61 UTC seconds of PPG_GREEN data.

    The timestamps are chosen to fall inside the Relax anchor window defined by
    ``tmp_anchor_csv`` (after converting from Asia/Jakarta to UTC).
    """
    ts = pd.date_range(
        start="2026-04-23T02:45:00Z",  # 09:45 WIB = 02:45 UTC
        periods=61,
        freq="1s",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "participant_code": "ID-01",
            "session_id": "S1",
            "signal_type": "PPG_GREEN",
            "timestamp_utc": ts,
            "timestamp_local": ts.tz_convert("Asia/Jakarta"),
            "sample_index": range(len(ts)),
            "raw_value": 1.0,
            "sampling_rate_hz": 1,
            "phase": pd.NA,
            "anchor_start": pd.NaT,
            "anchor_end": pd.NaT,
            "quality_flag": "UNKNOWN",
            "preprocessing_version": "v2.0.0",
        }
    )
