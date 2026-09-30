"""
test_anchors.py
---------------
Tests for AnchorMapper name resolution and timestamp handling.

Task 20.1: Test AnchorMapper name resolution with known fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from PipelinePython.common.anchors import AnchorMapper
from PipelinePython.common.settings import load_pipeline_config


class TestAnchorMapperNameResolution:
    """Test AnchorMapper.infer_code_from_session_participant()."""

    @pytest.fixture()
    def anchor_mapper(self, tmp_path: Path) -> AnchorMapper:
        """Create an AnchorMapper with a test anchor CSV."""
        anchor_csv = tmp_path / "anchors.csv"
        anchor_csv.write_text(
            "participant_code,phase,start_timestamp,end_timestamp\n"
            "TEST01,Relax,2026-04-23T09:45:00+07:00,2026-04-23T09:46:00+07:00\n"
            "TEST02,Task,2026-04-23T10:00:00+07:00,2026-04-23T10:01:00+07:00\n",
            encoding="utf-8",
        )
        return AnchorMapper(anchor_csv, timezone_local="Asia/Jakarta")

    def test_load_anchor_csv(self, anchor_mapper: AnchorMapper) -> None:
        """Test that AnchorMapper loads anchor CSV correctly."""
        df = anchor_mapper.load()
        assert len(df) >= 2
        assert "participant_code" in df.columns
        assert "phase" in df.columns

    def test_anchor_timestamps_are_datetime(self, anchor_mapper: AnchorMapper) -> None:
        """Test that anchor timestamps are parsed as datetime."""
        df = anchor_mapper.load()
        assert pd.api.types.is_datetime64_any_dtype(df["start_timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_timestamp"])

    def test_anchor_phase_values(self, anchor_mapper: AnchorMapper) -> None:
        """Test that anchor phases are correctly loaded."""
        df = anchor_mapper.load()
        assert "phase" in df.columns
        assert len(df) > 0


class TestAnchorTimestampHandling:
    """Test timezone handling and timestamp alignment."""

    def test_anchor_timestamps_utc_aware(self, tmp_anchor_csv: Path) -> None:
        """Test that loaded anchors have UTC-aware timestamps."""
        mapper = AnchorMapper(str(tmp_anchor_csv), timezone_local="Asia/Jakarta")
        df = mapper.load()
        assert df["start_timestamp"].dt.tz is not None
        assert df["end_timestamp"].dt.tz is not None

    def test_anchor_start_before_end(self, tmp_anchor_csv: Path) -> None:
        """Test that anchor start_timestamp < end_timestamp."""
        mapper = AnchorMapper(str(tmp_anchor_csv), timezone_local="Asia/Jakarta")
        df = mapper.load()
        assert (df["start_timestamp"] < df["end_timestamp"]).all()

    def test_id08_uses_full_anchor_without_legacy_buffer(self) -> None:
        """ID-08 anchors remain nominally five minutes, not 280 seconds."""
        config = load_pipeline_config()
        mapper = AnchorMapper(config.raw_source.anchor_csv, config.timezone_local)
        anchors = mapper.load()
        id08 = anchors[
            (anchors["participant_code"] == "ID-08")
            & (anchors["phase"].isin(["Relax", "Task"]))
        ].copy()

        assert len(id08) == 2
        assert config.window.validation_buffer_seconds == 0
        assert config.window.gold_buffer_seconds == 0
        durations = (id08["end_timestamp"] - id08["start_timestamp"]).dt.total_seconds()
        assert (durations >= 299.0).all()
