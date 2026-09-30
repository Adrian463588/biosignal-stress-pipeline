from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from PipelinePython.common.ingestion import DataLoader


def _write_ecg(path: Path) -> None:
    path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=ECG\n"
        "samplingrate=130.0\n"
        "0.10,1,1000,2026-04-21T09:00:00.000+07:00,1,2,3\n"
        "malformed,row\n"
        "nan,0,1020,2026-04-21T09:00:00.020+07:00,4,5,6\n"
        "0.20,0,1010,2026-04-21T09:00:00.010+07:00,7,8,9\n",
        encoding="utf-8",
    )


def test_data_loader_preserves_raw_order_and_audits_row_issues(
    tmp_path: Path,
) -> None:
    relax = tmp_path / "ID-01" / "Relaks"
    unknown = tmp_path / "ID-01" / "Routine"
    relax.mkdir(parents=True)
    unknown.mkdir()

    ecg_path = relax / "ID-01Relaks_ECG_2026.csv"
    unknown_path = unknown / "ID-01Routine_ECG_2026.csv"
    _write_ecg(ecg_path)
    _write_ecg(unknown_path)

    manifest_dir = tmp_path / "manifest"
    sessions = DataLoader(tmp_path).load_all(manifest_dir)

    assert len(sessions) == 1
    ecg = sessions[0].ecg
    assert ecg is not None
    assert ecg["raw_sample_index"].tolist() == [0, 1, 2, 3]
    assert ecg["unix_ms"].iloc[[0, 2, 3]].tolist() == [1000.0, 1020.0, 1010.0]
    assert pd.isna(ecg.loc[1, "unix_ms"])
    assert pd.isna(ecg.loc[1, "ecg_mv"])
    assert pd.isna(ecg.loc[2, "ecg_mv"])

    issues = pd.read_csv(manifest_dir / "manifest_issues.csv")
    assert {
        "source_file",
        "source_checksum",
        "row_index",
        "raw_sample_index",
        "issue_type",
    }.issubset(issues.columns)
    assert {"malformed_row", "invalid_value", "unknown_phase"}.issubset(
        set(issues["issue_type"])
    )

    expected_checksum = hashlib.sha256(ecg_path.read_bytes()).hexdigest()
    row_issues = issues[issues["source_file"] == str(ecg_path.resolve())]
    assert (row_issues["source_checksum"] == expected_checksum).all()
    assert (
        row_issues.loc[row_issues["issue_type"] == "malformed_row", "row_index"]
        .notna()
        .any()
    )
    assert (
        issues.loc[issues["issue_type"] == "unknown_phase", "source_file"]
        .eq(str(unknown_path.resolve()))
        .any()
    )


def test_data_loader_emits_row_level_duplicate_timestamp_issue(tmp_path: Path) -> None:
    relax = tmp_path / "ID-01" / "Relaks"
    relax.mkdir(parents=True)
    path = relax / "ID-01Relaks_ECG_2026.csv"
    path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=ECG\n"
        "samplingrate=130.0\n"
        "0.10,1,1000,2026-04-21T09:00:00.000+07:00,1,2,3\n"
        "0.20,0,1000,2026-04-21T09:00:00.000+07:00,4,5,6\n",
        encoding="utf-8",
    )

    manifest_dir = tmp_path / "manifest"
    sessions = DataLoader(tmp_path).load_all(manifest_dir)

    assert len(sessions) == 1
    assert sessions[0].ecg is not None
    assert sessions[0].ecg["raw_sample_index"].tolist() == [0, 1]
    issues = pd.read_csv(manifest_dir / "manifest_issues.csv")
    duplicate = issues[issues["issue_type"] == "duplicate_timestamp"]
    assert len(duplicate) == 1
    assert duplicate.iloc[0]["raw_sample_index"] == 1
    assert duplicate.iloc[0]["row_index"] == 5


def test_optional_accelerometer_columns_are_unavailable_not_malformed(
    tmp_path: Path,
) -> None:
    relax = tmp_path / "ID-01" / "Relaks"
    relax.mkdir(parents=True)
    ecg_path = relax / "ID-01Relaks_ECG_2026.csv"
    ecg_path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=ECG\n"
        "samplingrate=130.0\n"
        "0.10,1,1000,2026-04-21T09:00:00.000+07:00\n"
        "0.20,0,1008,2026-04-21T09:00:00.008+07:00\n",
        encoding="utf-8",
    )
    ppg_path = relax / "ID-01Relaks-PPG Green-2026.csv"
    ppg_path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=PPG Green\n"
        "samplingrate=25.0\n"
        "100,0,2026-04-21T09:00:00.000+07:00\n"
        "110,0,2026-04-21T09:00:00.040+07:00\n",
        encoding="utf-8",
    )

    manifest_dir = tmp_path / "manifest"
    sessions = DataLoader(tmp_path).load_all(manifest_dir)
    assert sessions[0].ecg is not None
    assert sessions[0].ppg_green is not None
    assert sessions[0].ecg["raw_sample_index"].tolist() == [0, 1]
    assert sessions[0].ppg_green["raw_sample_index"].tolist() == [0, 1]
    assert sessions[0].ecg[["acc_x", "acc_y", "acc_z"]].isna().all().all()
    assert sessions[0].ppg_green[["acc_x", "acc_y", "acc_z"]].isna().all().all()

    issues = pd.read_csv(manifest_dir / "manifest_issues.csv")
    assert "optional_columns_missing" in set(issues["issue_type"])
    assert "malformed_row" not in set(issues["issue_type"])


def test_ppg_trailing_fields_are_audit_only_without_accelerometer_schema(
    tmp_path: Path,
) -> None:
    relax = tmp_path / "ID-01" / "Relaks"
    relax.mkdir(parents=True)
    path = relax / "ID-01Relaks-PPG Green-2026.csv"
    path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=PPG Green\n"
        "samplingrate=25.0\n"
        "100,1000,2026-04-21T09:00:00.000+07:00,0,-9,-696\n"
        "110,1040,2026-04-21T09:00:00.040+07:00,0,-9,-656\n",
        encoding="utf-8",
    )

    sessions = DataLoader(tmp_path).load_all(tmp_path / "manifest")

    ppg = sessions[0].ppg_green
    assert ppg is not None
    assert ppg[["acc_x", "acc_y", "acc_z"]].isna().all().all()
    assert ppg["ppg_aux_3"].tolist() == [-696.0, -656.0]
    assert set(ppg["accelerometer_provenance_status"]) == {"UNAVAILABLE"}
    issues = pd.read_csv(tmp_path / "manifest" / "manifest_issues.csv")
    unverified = issues[issues["issue_type"] == "unverified_auxiliary_fields"]
    assert len(unverified) == 1
    assert unverified.iloc[0]["column"] == "ppg_aux_1,ppg_aux_2,ppg_aux_3"


def test_data_loader_audits_invalid_timestamp_without_dropping_raw_row(
    tmp_path: Path,
) -> None:
    relax = tmp_path / "ID-01" / "Relaks"
    relax.mkdir(parents=True)
    path = relax / "ID-01Relaks_ECG_2026.csv"
    path.write_text(
        "application=SamplingApp Version: 1.1\n"
        "datatype=ECG\n"
        "samplingrate=130.0\n"
        "0.10,1,1000,not-a-timestamp\n"
        "0.20,0,1008,2026-04-21T09:00:00.008+07:00\n",
        encoding="utf-8",
    )

    manifest_dir = tmp_path / "manifest"
    sessions = DataLoader(tmp_path).load_all(manifest_dir)

    assert len(sessions) == 1
    assert sessions[0].ecg is not None
    assert len(sessions[0].ecg) == 2
    issues = pd.read_csv(manifest_dir / "manifest_issues.csv")
    invalid_timestamp = issues[issues["issue_type"] == "invalid_timestamp"]
    assert len(invalid_timestamp) == 1
    assert invalid_timestamp.iloc[0]["raw_sample_index"] == 0


def test_data_loader_audits_unknown_participant_files(tmp_path: Path) -> None:
    unknown = tmp_path / "ID-99" / "Relaks"
    unknown.mkdir(parents=True)
    path = unknown / "ID-99Relaks_ECG_2026.csv"
    path.write_text("not loaded\n", encoding="utf-8")

    manifest_dir = tmp_path / "manifest"
    sessions = DataLoader(tmp_path).load_all(manifest_dir)

    assert sessions == []
    issues = pd.read_csv(manifest_dir / "manifest_issues.csv")
    row = issues[issues["issue_type"] == "unknown_participant"].iloc[0]
    assert row["participant_id"] == "ID-99"
    assert row["source_file"] == str(path.resolve())
    assert len(row["source_checksum"]) == 64
