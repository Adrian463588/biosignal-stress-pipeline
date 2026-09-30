from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    CONDITION_RELAX,
    CONDITION_STRESS,
    CONDITIONS,
    ECG_COLUMNS,
    ECG_PATTERN,
    ECG_SKIP_ROWS,
    PPG_COLUMNS,
    PPG_AUXILIARY_COLUMNS,
    PPG_GREEN_PATTERN,
    PPG_IR_PATTERN,
    PPG_RED_PATTERN,
    PPG_SKIP_ROWS,
)

logger = logging.getLogger(__name__)

_CANONICAL_PHASE = {
    CONDITION_RELAX: "relax",
    CONDITION_STRESS: "task",
}


@dataclass
class ParticipantSession:
    participant: str
    condition: str
    label: int
    ecg: Optional[pd.DataFrame] = None
    ppg_green: Optional[pd.DataFrame] = None
    ppg_ir: Optional[pd.DataFrame] = None
    ppg_red: Optional[pd.DataFrame] = None
    signal_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    lineage_ambiguous_channels: set[str] = field(default_factory=set)

    def is_valid(self) -> bool:
        return self.ecg is not None and self.ppg_green is not None


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _is_int(x: str) -> bool:
    try:
        int(float(x))
        return True
    except Exception:
        return False


def _is_ppg_data_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    # Samsung exports may omit accelerometer columns.  The signal, unix-ms,
    # and human-readable timestamp are still a complete PPG sample.
    if len(parts) < 3:
        return False
    return _is_float(parts[0]) and _is_int(parts[1])


def _is_ecg_data_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    # Polar exports may omit accelerometer columns.  ECG value, peak flag,
    # unix-ms, and datetime are the required source-order fields.
    if len(parts) < 4:
        return False
    return _is_float(parts[0]) and _is_int(parts[1]) and _is_int(parts[2])


def _parse_metadata_lines(lines: List[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    known_keys = (
        "application",
        "datatype",
        "stoptime",
        "duration",
        "samplescount",
        "samplingrate",
        "stopdevicehr",
        "stopcalculatedhr",
        "peakscount",
        "devicename",
        "deviceid",
        "battery",
        "firmware",
    )
    key_pattern = re.compile(rf"(?i)({'|'.join(map(re.escape, known_keys))})=")
    for line in lines:
        text = line.strip()
        if "=" not in text:
            continue
        matches = list(key_pattern.finditer(text))
        if not matches:
            matches = list(re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=", text))
        if not matches:
            continue
        for i, m in enumerate(matches):
            key = m.group(1).strip().lower()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[start:end].strip()
            if key and value:
                metadata[key] = value
    return metadata


def _split_header_and_data(
    path: Path, signal_type: str
) -> tuple[List[str], List[tuple[int, str]]]:
    lines = [
        ln.rstrip("\n\r")
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    detector = _is_ecg_data_line if signal_type == "ecg" else _is_ppg_data_line

    data_start = None
    for i, line in enumerate(lines):
        if detector(line.strip()):
            data_start = i
            break

    skip = ECG_SKIP_ROWS if signal_type == "ecg" else PPG_SKIP_ROWS
    if data_start is None:
        data_start = min(skip, len(lines))
    elif data_start < skip:
        # Small contract fixtures can have fewer metadata lines than the
        # nominal export.  Preserve their first observed data/malformed row.
        pass
    else:
        # Some real exports contain additional key=value metadata after the
        # nominal header.  Move past metadata, but retain the first non-metadata
        # row even when it is malformed so it can be audited.
        nominal_start = min(skip, len(lines))
        for index in range(nominal_start, data_start + 1):
            text = lines[index].strip()
            if text and not ("=" in text and "," not in text):
                data_start = index
                break

    header_lines = lines[:data_start]
    data_lines = [
        (line_number, line)
        for line_number, line in enumerate(lines[data_start:], start=data_start + 1)
        if line.strip()
    ]
    return header_lines, data_lines


def _extract_sampling_rate(meta: Dict[str, str], default: float) -> float:
    raw = meta.get("samplingrate") or meta.get("sampling_rate")
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _compute_actual_fs(df: pd.DataFrame, default_fs: float) -> float:
    if len(df) > 1:
        timestamps = pd.to_numeric(df["unix_ms"], errors="coerce").to_numpy()
        deltas = np.diff(timestamps)
        positive_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        if positive_deltas.size:
            return 1000.0 / float(np.median(positive_deltas))
    return default_fs


_COLON_MILLISECOND_TIMESTAMP = re.compile(
    r"\d{1,2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2}:\d{1,6} [+-]\d{4}"
)


def _parse_timestamp_series(values: pd.Series) -> pd.Series:
    """Parse known export timestamp formats without scalar date parsing."""

    text_values = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    colon_millisecond = text_values.str.fullmatch(_COLON_MILLISECOND_TIMESTAMP)
    if colon_millisecond.any():
        parsed.loc[colon_millisecond] = pd.to_datetime(
            text_values.loc[colon_millisecond],
            format="%d %b %Y %H:%M:%S:%f %z",
            utc=True,
            errors="coerce",
        )
    remaining = ~colon_millisecond & text_values.notna() & text_values.ne("")
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            text_values.loc[remaining],
            format="mixed",
            utc=True,
            errors="coerce",
        )
    return parsed


def _parse_signal_rows(
    data_lines: List[tuple[int, str]],
    columns: List[str],
    signal_column: str,
    include_provenance: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    required_field_count = 4 if signal_column == "ecg_mv" else 3
    optional_missing_count = 0
    optional_columns = columns[required_field_count:]

    for raw_sample_index, (row_index, line) in enumerate(data_lines):
        row: dict[str, object] = {column: None for column in columns}
        try:
            values = next(csv.reader([line]))
        except csv.Error as exc:
            values = []
            issues.append(
                {
                    "row_index": row_index,
                    "raw_sample_index": raw_sample_index,
                    "issue_type": "parse_error",
                    "column": "",
                    "issues": f"CSV parse error: {exc}",
                }
            )

        if len(values) < required_field_count or len(values) > len(columns):
            issues.append(
                {
                    "row_index": row_index,
                    "raw_sample_index": raw_sample_index,
                    "issue_type": "malformed_row",
                    "column": "",
                    "issues": (
                        f"Expected {len(columns)} fields, received {len(values)}"
                    ),
                }
            )
        elif len(values) < len(columns):
            optional_missing_count += 1

        for column, value in zip(columns, values[: len(columns)]):
            row[column] = value.strip()
        rows.append(row)

    raw_frame = pd.DataFrame(rows, columns=columns)
    df = raw_frame.copy()
    numeric_columns = [column for column in columns if column != "datetime_str"]
    for column in numeric_columns:
        numeric = pd.to_numeric(raw_frame[column], errors="coerce")
        df[column] = numeric

    for column in (signal_column, "unix_ms"):
        numeric = df[column]
        invalid = (
            raw_frame[column].isna()
            | raw_frame[column].astype(str).eq("")
            | numeric.isna()
            | ~np.isfinite(numeric.to_numpy(dtype=float))
        )
        for raw_sample_index in np.flatnonzero(invalid.to_numpy(dtype=bool)):
            row_index = data_lines[raw_sample_index][0]
            value = raw_frame.iloc[raw_sample_index][column]
            issues.append(
                {
                    "row_index": row_index,
                    "raw_sample_index": int(raw_sample_index),
                    "issue_type": "invalid_value",
                    "column": column,
                    "issues": f"Invalid numeric value for {column}: {value!r}",
                }
            )

    timestamp_values = raw_frame["datetime_str"]
    parsed_timestamps = _parse_timestamp_series(timestamp_values)
    invalid_timestamps = (
        timestamp_values.isna()
        | timestamp_values.astype(str).eq("")
        | parsed_timestamps.isna()
    )
    for raw_sample_index in np.flatnonzero(invalid_timestamps.to_numpy(dtype=bool)):
        row_index = data_lines[raw_sample_index][0]
        timestamp_value = timestamp_values.iloc[raw_sample_index]
        issues.append(
            {
                "row_index": row_index,
                "raw_sample_index": int(raw_sample_index),
                "issue_type": "invalid_timestamp",
                "column": "datetime_str",
                "issues": f"Invalid timestamp for datetime_str: {timestamp_value!r}",
            }
        )

    # Keep duplicate timestamps in the returned frame for source-order
    # traceability, but emit one issue for every repeated observed timestamp.
    # The downstream canonical stages decide whether/how to deduplicate; raw
    # ingestion must not silently discard a sample.
    timestamps = pd.to_numeric(df["unix_ms"], errors="coerce")
    seen_timestamps: dict[float, int] = {}
    for raw_sample_index, timestamp in enumerate(timestamps):
        if not np.isfinite(timestamp):
            continue
        timestamp_value = float(timestamp)
        first_index = seen_timestamps.get(timestamp_value)
        if first_index is None:
            seen_timestamps[timestamp_value] = raw_sample_index
            continue
        row_index = data_lines[raw_sample_index][0]
        issues.append(
            {
                "row_index": row_index,
                "raw_sample_index": raw_sample_index,
                "issue_type": "duplicate_timestamp",
                "column": "unix_ms",
                "issues": (
                    f"Duplicate timestamp {timestamp_value:g}; "
                    f"first_raw_sample_index={first_index}"
                ),
            }
        )

    df.attrs["raw_sample_index"] = np.arange(len(df), dtype=np.int64)
    df.attrs["optional_columns_missing_count"] = optional_missing_count
    df.attrs["optional_columns_missing"] = tuple(optional_columns)
    if optional_missing_count:
        issues.append(
            {
                "row_index": None,
                "raw_sample_index": None,
                "issue_type": "optional_columns_missing",
                "column": ",".join(optional_columns),
                "issues": (
                    f"{optional_missing_count} observed rows omit optional columns: "
                    f"{','.join(optional_columns)}"
                ),
            }
        )
    if include_provenance:
        df.insert(0, "raw_sample_index", np.arange(len(df), dtype=np.int64))
    df.attrs["ingestion_issues"] = issues
    return df, issues


def _load_ecg(path: Path, *, include_provenance: bool = False) -> pd.DataFrame:
    header_lines, data_lines = _split_header_and_data(path, signal_type="ecg")
    meta = _parse_metadata_lines(header_lines)

    df, _ = _parse_signal_rows(
        data_lines,
        ECG_COLUMNS,
        signal_column="ecg_mv",
        include_provenance=include_provenance,
    )
    df.attrs["metadata"] = meta
    default_fs = _extract_sampling_rate(meta, default=130.0)
    df.attrs["sampling_rate_hz"] = _compute_actual_fs(df, default_fs)
    return df


def _load_ppg(
    path: Path, default_fs: float, *, include_provenance: bool = False
) -> pd.DataFrame:
    header_lines, data_lines = _split_header_and_data(path, signal_type="ppg")
    meta = _parse_metadata_lines(header_lines)

    df, issues = _parse_signal_rows(
        data_lines,
        PPG_COLUMNS,
        signal_column="ppg_raw",
        include_provenance=include_provenance,
    )
    auxiliary_present = any(
        df[column].notna().any() for column in PPG_AUXILIARY_COLUMNS
    )
    if auxiliary_present:
        issues.append(
            {
                "row_index": None,
                "raw_sample_index": None,
                "issue_type": "unverified_auxiliary_fields",
                "column": ",".join(PPG_AUXILIARY_COLUMNS),
                "issues": (
                    "Trailing PPG fields are preserved without accelerometer semantics; "
                    "a verified AccelerometerSet source is required for MADC"
                ),
            }
        )
    for axis in ("acc_x", "acc_y", "acc_z"):
        df[axis] = np.nan
    df["accelerometer_provenance_status"] = "UNAVAILABLE"
    df.attrs["accelerometer_provenance_status"] = "UNAVAILABLE"
    df.attrs["ingestion_issues"] = issues
    df.attrs["metadata"] = meta
    default_fs_extracted = _extract_sampling_rate(meta, default=default_fs)
    df.attrs["sampling_rate_hz"] = _compute_actual_fs(df, default_fs_extracted)
    return df


def _load_ppg_green(path: Path, *, include_provenance: bool = False) -> pd.DataFrame:
    return _load_ppg(path, default_fs=25.0, include_provenance=include_provenance)


def _load_ppg_ir(path: Path, *, include_provenance: bool = False) -> pd.DataFrame:
    return _load_ppg(path, default_fs=100.0, include_provenance=include_provenance)


def _load_ppg_red(path: Path, *, include_provenance: bool = False) -> pd.DataFrame:
    return _load_ppg(path, default_fs=100.0, include_provenance=include_provenance)


_FILE_LOADERS = {
    ECG_PATTERN: _load_ecg,
    PPG_GREEN_PATTERN: _load_ppg_green,
    PPG_IR_PATTERN: _load_ppg_ir,
    PPG_RED_PATTERN: _load_ppg_red,
}

_CHANNEL_LOADERS = {
    "ecg": _load_ecg,
    "ppg_green": _load_ppg_green,
    "ppg_ir": _load_ppg_ir,
    "ppg_red": _load_ppg_red,
}


def load_signal_file(
    path: Path, sensor_channel: str, *, include_provenance: bool = True
) -> pd.DataFrame:
    """Load one known signal file without loading any sibling recording."""

    try:
        loader = _CHANNEL_LOADERS[sensor_channel]
    except KeyError as exc:
        raise ValueError(f"Unsupported sensor channel: {sensor_channel}") from exc
    return loader(Path(path), include_provenance=include_provenance)


def get_sampling_rate_hz(df: pd.DataFrame, default: float) -> float:
    try:
        return float(df.attrs.get("sampling_rate_hz", default))
    except Exception:  # noqa: BLE001
        return float(default)


def _detect_signal_type(filename: str) -> Optional[str]:
    for pattern in _FILE_LOADERS:
        if pattern.lower() in filename.lower():
            return pattern
    return None


class DataLoader:
    def __init__(self, data_root: Path, *, run_id: str = "") -> None:
        self._root = Path(data_root)
        self._allowed_ids = {f"ID-{i:02d}" for i in range(1, 33)}
        self._run_id = run_id

    def load_all(self, manifest_dir: Optional[Path] = None) -> List[ParticipantSession]:
        sessions: List[ParticipantSession] = []
        manifest_records = []
        issue_records = []

        for participant_dir in sorted(self._root.iterdir()):
            if not participant_dir.is_dir():
                continue
            participant = participant_dir.name

            if participant not in self._allowed_ids:
                self._audit_unknown_participant(participant_dir, issue_records)
                logger.warning(
                    "Skipping participant outside allowed range: %s", participant
                )
                continue

            for condition_dir in sorted(participant_dir.iterdir()):
                if not condition_dir.is_dir():
                    continue
                condition = condition_dir.name
                if condition not in CONDITIONS:
                    self._audit_unknown_phase(participant, condition_dir, issue_records)
                    continue
                sessions.append(
                    self._load_session(
                        participant,
                        condition,
                        condition_dir,
                        manifest_records,
                        issue_records,
                    )
                )

        if manifest_dir:
            manifest_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(manifest_records).to_csv(
                manifest_dir / "manifest.csv", index=False
            )
            issue_frame = pd.DataFrame(issue_records)
            issue_columns = [
                "run_id",
                "participant_id",
                "session_id",
                "phase_source",
                "phase",
                "phase_canonical",
                "sensor_channel",
                "source_file",
                "source_checksum",
                "row_index",
                "raw_sample_index",
                "issue_type",
                "issue_code",
                "severity",
                "column",
                "issues",
            ]
            for column in issue_columns:
                if column not in issue_frame:
                    issue_frame[column] = ""
            issue_frame["run_id"] = self._run_id
            issue_frame["phase_source"] = issue_frame["phase_source"].where(
                issue_frame["phase_source"].notna()
                & issue_frame["phase_source"].astype(str).ne(""),
                issue_frame["phase"],
            )
            phase_fallback = (
                issue_frame["phase"]
                .map(
                    {
                        "Relaks": "relax",
                        "Stress": "task",
                        "relax": "relax",
                        "task": "task",
                    }
                )
                .fillna("")
            )
            issue_frame["phase_canonical"] = issue_frame["phase_canonical"].where(
                issue_frame["phase_canonical"].notna()
                & issue_frame["phase_canonical"].astype(str).ne(""),
                phase_fallback,
            )
            issue_frame["issue_code"] = issue_frame["issue_code"].where(
                issue_frame["issue_code"].notna()
                & issue_frame["issue_code"].astype(str).ne(""),
                issue_frame["issue_type"],
            )
            error_types = {
                "parse_error",
                "malformed_row",
                "invalid_value",
                "invalid_timestamp",
                "lineage_ambiguous",
                "unknown_phase",
                "unknown_participant",
                "duplicate_timestamp",
                "non_monotonic_timestamp",
                "load_error",
            }
            issue_frame["severity"] = issue_frame["issue_type"].map(
                lambda value: "error" if value in error_types else "warning"
            )
            issue_frame.loc[:, issue_columns].to_csv(
                manifest_dir / "manifest_issues.csv", index=False
            )

        logger.info("Loaded %d sessions from %s", len(sessions), self._root)
        return sessions

    def load_participant(self, participant_name: str) -> List[ParticipantSession]:
        if participant_name not in self._allowed_ids:
            raise ValueError(
                f"Participant {participant_name} is outside the allowed range ID-01 to ID-32"
            )

        participant_dir = self._root / participant_name
        if not participant_dir.exists():
            raise FileNotFoundError(
                f"Participant directory not found: {participant_dir}"
            )
        sessions = []
        for condition_dir in sorted(participant_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name
            if condition not in CONDITIONS:
                self._audit_unknown_phase(participant_name, condition_dir, None)
                continue
            sessions.append(
                self._load_session(participant_name, condition, condition_dir)
            )
        return sessions

    @staticmethod
    def _file_checksum(path: Path) -> str:
        sha256 = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4096), b""):
                    sha256.update(chunk)
        except OSError as exc:
            return f"ERROR: {exc}"
        return sha256.hexdigest()

    def _audit_unknown_phase(
        self,
        participant: str,
        folder: Path,
        issue_records: Optional[list] = None,
    ) -> None:
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            records = [
                {
                    "participant_id": participant,
                    "session_id": participant,
                    "phase": folder.name,
                    "sensor_channel": "unknown",
                    "source_file": str(folder.resolve()),
                    "source_checksum": "",
                    "row_index": None,
                    "raw_sample_index": None,
                    "issue_type": "unknown_phase",
                    "column": "",
                    "issues": f"Unknown phase directory: {folder.name}",
                }
            ]
        else:
            records = []
            channel_map = {
                ECG_PATTERN: "ecg",
                PPG_GREEN_PATTERN: "ppg_green",
                PPG_IR_PATTERN: "ppg_ir",
                PPG_RED_PATTERN: "ppg_red",
            }
            for path in csv_files:
                signal_type = _detect_signal_type(path.name)
                records.append(
                    {
                        "participant_id": participant,
                        "session_id": participant,
                        "phase": folder.name,
                        "sensor_channel": channel_map.get(signal_type, "unknown"),
                        "source_file": str(path.resolve()),
                        "source_checksum": self._file_checksum(path),
                        "row_index": None,
                        "raw_sample_index": None,
                        "issue_type": "unknown_phase",
                        "column": "",
                        "issues": f"Unknown phase directory: {folder.name}",
                    }
                )

        if issue_records is not None:
            issue_records.extend(records)
        logger.warning(
            "Auditing files under unknown phase %s for participant %s",
            folder.name,
            participant,
        )

    def _audit_unknown_participant(
        self,
        participant_dir: Path,
        issue_records: list[dict[str, object]],
    ) -> None:
        """Record out-of-contract participant files without loading them."""

        files = sorted(path for path in participant_dir.rglob("*") if path.is_file())
        if not files:
            files = [participant_dir]
        channel_map = {
            ECG_PATTERN: "ecg",
            PPG_GREEN_PATTERN: "ppg_green",
            PPG_IR_PATTERN: "ppg_ir",
            PPG_RED_PATTERN: "ppg_red",
        }
        for path in files:
            phase = path.parent.name if path.is_file() else "unknown"
            signal_type = _detect_signal_type(path.name) if path.is_file() else None
            issue_records.append(
                {
                    "participant_id": participant_dir.name,
                    "session_id": participant_dir.name,
                    "phase": phase,
                    "sensor_channel": channel_map.get(signal_type, "unknown"),
                    "source_file": str(path.resolve()),
                    "source_checksum": (
                        self._file_checksum(path) if path.is_file() else ""
                    ),
                    "row_index": None,
                    "raw_sample_index": None,
                    "issue_type": "unknown_participant",
                    "column": "",
                    "issues": (
                        "Participant directory is outside ID-01..ID-32: "
                        f"{participant_dir.name}"
                    ),
                }
            )

    def _load_session(
        self,
        participant: str,
        condition: str,
        folder: Path,
        manifest_records: list = None,
        issue_records: list = None,
    ) -> ParticipantSession:
        session = ParticipantSession(
            participant=participant, condition=condition, label=CONDITIONS[condition]
        )
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            session.errors.append(f"No CSV files found in {folder}")
            return session
        for csv_path in csv_files:
            self._assign_file_to_session(
                session, csv_path, manifest_records, issue_records
            )
        if not session.is_valid():
            session.errors.append(
                f"Missing required signals (ECG or PPG Green) in {folder}"
            )
        return session

    def _assign_file_to_session(
        self,
        session: ParticipantSession,
        path: Path,
        manifest_records: list = None,
        issue_records: list = None,
    ) -> None:
        signal_type = _detect_signal_type(path.name)
        if signal_type is None:
            logger.debug("Skipping unrecognised file: %s", path.name)
            return

        loader = _FILE_LOADERS[signal_type]
        channel_map = {
            ECG_PATTERN: "ecg",
            PPG_GREEN_PATTERN: "ppg_green",
            PPG_IR_PATTERN: "ppg_ir",
            PPG_RED_PATTERN: "ppg_red",
        }
        sensor_channel = channel_map.get(signal_type, "unknown")

        file_hash = self._file_checksum(path)
        existing_attribute = {
            ECG_PATTERN: "ecg",
            PPG_GREEN_PATTERN: "ppg_green",
            PPG_IR_PATTERN: "ppg_ir",
            PPG_RED_PATTERN: "ppg_red",
        }[signal_type]
        if getattr(session, existing_attribute) is not None:
            reason = (
                "LINEAGE_AMBIGUOUS:multiple source files for "
                f"{sensor_channel} in {session.participant}/{session.condition}"
            )
            session.errors.append(reason)
            session.lineage_ambiguous_channels.add(sensor_channel)
            if issue_records is not None:
                issue_records.append(
                    {
                        "participant_id": session.participant,
                        "session_id": session.participant,
                        "phase": _CANONICAL_PHASE[session.condition],
                        "sensor_channel": sensor_channel,
                        "source_file": str(path.resolve()),
                        "source_checksum": file_hash,
                        "row_index": None,
                        "raw_sample_index": None,
                        "issue_type": "lineage_ambiguous",
                        "column": "",
                        "issues": reason,
                    }
                )
            logger.error("%s", reason)
            return

        try:
            df = loader(path, include_provenance=True)
            df.attrs["source_file"] = str(path.resolve())
            df.attrs["source_checksum"] = file_hash
            self._attach_df(session, signal_type, df)
            session.signal_meta[signal_type] = dict(df.attrs.get("metadata", {}))
            logger.debug("Loaded %s -> %s (%d rows)", path.name, signal_type, len(df))

            if manifest_records is not None and issue_records is not None:
                for parser_issue in df.attrs.get("ingestion_issues", []):
                    issue_records.append(
                        {
                            "participant_id": session.participant,
                            "session_id": session.participant,
                            "phase": _CANONICAL_PHASE[session.condition],
                            "phase_source": session.condition,
                            "phase_canonical": _CANONICAL_PHASE[session.condition],
                            "sensor_channel": sensor_channel,
                            "source_file": str(path.resolve()),
                            "source_checksum": file_hash,
                            **parser_issue,
                        }
                    )

                issues = []
                actual_fs = float(df.attrs.get("sampling_rate_hz", 0.0))
                start_ts = df["unix_ms"].min() if not df.empty else None
                end_ts = df["unix_ms"].max() if not df.empty else None
                duration = (
                    (end_ts - start_ts) / 1000.0
                    if (start_ts is not None and end_ts is not None)
                    else 0.0
                )

                sig_col = "ecg_mv" if signal_type == ECG_PATTERN else "ppg_raw"

                if df.empty:
                    issues.append("Empty dataframe")
                else:
                    if not df["unix_ms"].is_monotonic_increasing:
                        issues.append("Non-monotonic timestamps")

                    dups = df["unix_ms"].duplicated().sum()
                    if dups > 0:
                        issues.append(f"{dups} duplicate timestamps")

                    invalid = df[sig_col].isna() | np.isinf(df[sig_col])
                    if invalid.sum() > 0:
                        issues.append(f"{invalid.sum()} NaN/Inf values")

                    delta_ms = df["unix_ms"].diff()
                    med_delta = delta_ms.median()
                    if pd.notna(med_delta) and med_delta > 0:
                        gaps = (delta_ms > 1.5 * med_delta).sum()
                        if gaps > 0:
                            issues.append(f"{gaps} large timestamp gaps")

                manifest_records.append(
                    {
                        "participant_id": session.participant,
                        "session_id": session.participant,
                        "phase": _CANONICAL_PHASE[session.condition],
                        "sensor_channel": sensor_channel,
                        "source_file": str(path.resolve()),
                        "source_checksum": file_hash,
                        "sampling_rate_hz": actual_fs,
                        "start_timestamp_utc": start_ts,
                        "end_timestamp_utc": end_ts,
                        "row_count": len(df),
                        "duration_sec": duration,
                    }
                )

                if issues:
                    issue_records.append(
                        {
                            "participant_id": session.participant,
                            "session_id": session.participant,
                            "phase": _CANONICAL_PHASE[session.condition],
                            "sensor_channel": sensor_channel,
                            "source_file": str(path.resolve()),
                            "source_checksum": file_hash,
                            "issues": "; ".join(issues),
                        }
                    )

        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load {path.name}: {exc}"
            session.errors.append(msg)
            logger.warning(msg)
            if issue_records is not None:
                issue_records.append(
                    {
                        "participant_id": session.participant,
                        "session_id": session.participant,
                        "phase": _CANONICAL_PHASE[session.condition],
                        "sensor_channel": sensor_channel,
                        "source_file": str(path.resolve()),
                        "source_checksum": file_hash,
                        "issues": f"Load error: {exc}",
                    }
                )

    @staticmethod
    def _attach_df(
        session: ParticipantSession, signal_type: str, df: pd.DataFrame
    ) -> None:
        if signal_type == ECG_PATTERN:
            session.ecg = df
        elif signal_type == PPG_GREEN_PATTERN:
            session.ppg_green = df
        elif signal_type == PPG_IR_PATTERN:
            session.ppg_ir = df
        elif signal_type == PPG_RED_PATTERN:
            session.ppg_red = df


if __name__ == "__main__":
    from PipelinePython.common.cli import main

    main(["ingest"])
