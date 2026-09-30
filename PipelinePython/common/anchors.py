from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorRecord:
    participant_code: str
    phase: str
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


class AnchorWindowCoverageError(ValueError):
    """Raised when a frame cannot cover the requested core window."""


@dataclass(frozen=True)
class AnchorWindowMetadata:
    """Immutable provenance for a nominal 300-second anchor and 240-second core."""

    anchor_start: pd.Timestamp
    anchor_end: pd.Timestamp
    core_start: pd.Timestamp
    core_end: pd.Timestamp
    source_phase_duration_seconds: int
    edge_trim_start_seconds: int
    edge_trim_end_seconds: int
    analysis_phase_duration_seconds: int
    observed_start: pd.Timestamp
    observed_end: pd.Timestamp
    observed_duration_seconds: float
    row_count: int


@dataclass(frozen=True)
class AnchorWindowSlice:
    """Result of an immutable timestamp crop and its boundary metadata."""

    frame: pd.DataFrame
    metadata: AnchorWindowMetadata


class AnchorWindowSlicer:
    """Purely crop a signal frame to the canonical 240-second core window.

    The slicer performs no sorting, deduplication, interpolation, filtering,
    padding, or value modification. Those responsibilities remain in their
    respective pipeline stages. A frame must therefore already be monotonic
    and contain at least ``minimum_coverage_seconds`` of observed timestamps
    inside the core interval.
    """

    _TIMESTAMP_CANDIDATES = (
        "timestamp_corrected_utc",
        "timestamp_utc",
        "timestamp_raw_utc",
        "timestamp",
    )

    def __init__(
        self,
        *,
        source_phase_duration_seconds: int = 300,
        edge_trim_start_seconds: int = 30,
        edge_trim_end_seconds: int = 30,
        analysis_phase_duration_seconds: int = 240,
        minimum_coverage_seconds: float = 239.0,
    ) -> None:
        if source_phase_duration_seconds != 300:
            raise ValueError("source phase duration must remain 300 seconds")
        if edge_trim_start_seconds != 30 or edge_trim_end_seconds != 30:
            raise ValueError("canonical edge trim must be 30 seconds on both sides")
        if analysis_phase_duration_seconds != 240:
            raise ValueError("analysis phase duration must remain 240 seconds")
        if (
            edge_trim_start_seconds
            + edge_trim_end_seconds
            + analysis_phase_duration_seconds
            != source_phase_duration_seconds
        ):
            raise ValueError("core duration and edge trims must equal the source phase")
        if minimum_coverage_seconds < 239.0:
            raise ValueError("minimum core coverage cannot be below 239 seconds")

        self._source_phase_duration_seconds = source_phase_duration_seconds
        self._edge_trim_start_seconds = edge_trim_start_seconds
        self._edge_trim_end_seconds = edge_trim_end_seconds
        self._analysis_phase_duration_seconds = analysis_phase_duration_seconds
        self._minimum_coverage_seconds = minimum_coverage_seconds

    def bounds(
        self,
        anchor_start: pd.Timestamp | str,
        *,
        mode: Literal["source", "core", "validation", "gold"] = "core",
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return deterministic half-open bounds without touching signal values."""

        start = self._as_utc_timestamp(anchor_start)
        if mode == "source":
            return start, start + pd.Timedelta(
                seconds=self._source_phase_duration_seconds
            )
        if mode in {"core", "validation", "gold"}:
            core_start = start + pd.Timedelta(seconds=self._edge_trim_start_seconds)
            return core_start, core_start + pd.Timedelta(
                seconds=self._analysis_phase_duration_seconds
            )
        raise ValueError(f"unsupported anchor slicing mode: {mode!r}")

    def slice(
        self,
        frame: pd.DataFrame,
        anchor_start: pd.Timestamp | str,
        anchor_end: pd.Timestamp | str | None = None,
        *,
        timestamp_column: str | None = None,
        mode: Literal["source", "core", "validation", "gold"] = "core",
    ) -> AnchorWindowSlice:
        """Return the observed half-open interval ``[start+30, start+270)``.

        The input frame is never modified. Coverage is measured from the first
        to last observed timestamp in the returned frame; internal gaps are
        deliberately not filled or hidden by this component.
        """

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            raise AnchorWindowCoverageError("frame is empty; core window unavailable")

        column = timestamp_column or self._resolve_timestamp_column(frame)
        timestamps = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError(f"timestamp column contains invalid values: {column}")
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"timestamp column must be monotonic: {column}")

        start = self._as_utc_timestamp(anchor_start)
        source_end = start + pd.Timedelta(seconds=self._source_phase_duration_seconds)
        anchor_end_timestamp = (
            self._as_utc_timestamp(anchor_end) if anchor_end is not None else source_end
        )
        window_start, window_end = self.bounds(start, mode=mode)
        core_start, core_end = self.bounds(start, mode="core")
        if mode == "source":
            window_start, window_end = self.bounds(start, mode="source")
        mask = (timestamps >= window_start) & (timestamps < window_end)
        cropped = frame.loc[mask].copy(deep=True)
        if cropped.empty:
            raise AnchorWindowCoverageError("core window contains no observed samples")

        observed_timestamps = timestamps.loc[mask]
        observed_start = observed_timestamps.iloc[0]
        observed_end = observed_timestamps.iloc[-1]
        observed_duration = (observed_end - observed_start).total_seconds()
        if observed_duration < self._minimum_coverage_seconds:
            raise AnchorWindowCoverageError(
                f"core coverage is below 239 seconds: {observed_duration:.6f} seconds"
            )

        metadata = AnchorWindowMetadata(
            anchor_start=start,
            anchor_end=anchor_end_timestamp,
            core_start=core_start,
            core_end=core_end,
            source_phase_duration_seconds=self._source_phase_duration_seconds,
            edge_trim_start_seconds=self._edge_trim_start_seconds,
            edge_trim_end_seconds=self._edge_trim_end_seconds,
            analysis_phase_duration_seconds=self._analysis_phase_duration_seconds,
            observed_start=observed_start,
            observed_end=observed_end,
            observed_duration_seconds=observed_duration,
            row_count=len(cropped),
        )
        return AnchorWindowSlice(frame=cropped, metadata=metadata)

    def _resolve_timestamp_column(self, frame: pd.DataFrame) -> str:
        for candidate in self._TIMESTAMP_CANDIDATES:
            if candidate in frame.columns:
                return candidate
        raise ValueError(
            "frame must contain one of: " + ", ".join(self._TIMESTAMP_CANDIDATES)
        )

    @staticmethod
    def _as_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")


class AnchorMapper:
    def __init__(
        self,
        anchor_csv_path: Path,
        timezone_local: str,
        mapping_csv_path: Optional[Path] = None,
    ) -> None:
        self._anchor_csv_path = Path(anchor_csv_path)
        self._timezone_local = timezone_local
        self._mapping_csv_path = Path(mapping_csv_path) if mapping_csv_path else None
        self._aliases = {
            "ARYOO05": "ARYO05",
            "RAIIHAN11": "RAIHAN11",
            "DANISH-15": "DANISH15",
            "NABIL21": "ID-22",
            "SYAHRIAL22": "ID-23",
            "BINTANG23": "ID-24",
            "BAGAS24": "ID-25",
            "ALPHONS25": "ID-26",
            "RAFINDA26": "ID-27",
        }
        self._participant_map = self._load_participant_map()

    def _load_participant_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if self._mapping_csv_path is None or not self._mapping_csv_path.exists():
            return mapping
        df = pd.read_csv(self._mapping_csv_path)
        for _, row in df.iterrows():
            pid = str(row.get("participant_id", "")).strip().upper()
            original = str(row.get("original_name", "")).strip().upper()
            if pid:
                mapping[pid] = pid
                if original:
                    key = "".join(ch for ch in original if ch.isalnum())
                    mapping[key] = pid
        return mapping

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._anchor_csv_path)
        df["participant_code"] = (
            df["participant_code"].astype(str).str.upper().str.strip()
        )
        df["participant_code"] = df["participant_code"].replace(self._aliases)
        df["participant_code"] = df["participant_code"].map(self._to_participant_id)
        df["phase"] = df["phase"].astype(str).str.strip()

        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
        df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], utc=True)

        bad = df[df["end_timestamp"] <= df["start_timestamp"]]
        if not bad.empty:
            logger.warning("Found %d invalid anchor rows where end<=start", len(bad))

        # Check for suspiciously similar codes (fuzzy duplicates)
        self._check_fuzzy_duplicates(df["participant_code"].unique())

        return df

    def _check_fuzzy_duplicates(self, codes: list[str]) -> None:
        """Log warnings if any two participant codes are suspiciously similar (edit distance 1)."""

        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if not s2:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        unique_codes = sorted([c for c in codes if c])
        # Canonical anonymised IDs intentionally differ by one digit. Fuzzy
        # duplicate warnings are useful only before anonymisation.
        unique_codes = [
            code for code in unique_codes if not re.fullmatch(r"ID-\d{2}", code)
        ]
        for i in range(len(unique_codes)):
            for j in range(i + 1, len(unique_codes)):
                c1, c2 = unique_codes[i], unique_codes[j]
                if levenshtein(c1, c2) == 1:
                    logger.warning(
                        "Suspiciously similar participant codes detected: '%s' and '%s'",
                        c1,
                        c2,
                    )

    def infer_code_from_session_participant(self, participant_name: str) -> str:
        candidate = "".join(ch for ch in participant_name.upper() if ch.isalnum())
        candidate = self._aliases.get(candidate, candidate)
        if candidate in self._participant_map:
            return self._participant_map[candidate]
        if candidate.startswith("ID") and re.search(r"\d+", candidate):
            num = re.search(r"\d+", candidate).group(0)
            return f"ID-{int(num):02d}"
        return candidate

    def _to_participant_id(self, raw_code: str) -> str:
        code = self._aliases.get(raw_code, raw_code)
        cleaned = "".join(ch for ch in code if ch.isalnum())
        # 1) Direct map by cleaned name / known id key
        if cleaned in self._participant_map:
            return self._participant_map[cleaned]
        # 2) Deterministic map by numeric suffix e.g. ALBI03 -> ID-03
        match = re.search(r"(\d+)$", cleaned)
        if match:
            return f"ID-{int(match.group(1)):02d}"
        return cleaned
