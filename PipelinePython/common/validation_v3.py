"""Release-gate aggregation for local data, ML, ONNX, and Android evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class V3ReleaseValidator:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve()

    def _cleanup_baseline(self) -> dict[str, Any] | None:
        reference_path = (
            self.root / "reports" / "migration" / "cleanup_reference_v3.json"
        )
        candidates: list[Path] = []
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            manifest_path = Path(str(reference["cleanup_manifest"]))
            if manifest_path.exists():
                expected = str(reference["cleanup_manifest_sha256"])
                actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if actual != expected:
                    return None
                candidates.append(manifest_path)
        default_backup_root = self.root.parent / "_pipeline_backups"
        if default_backup_root.exists():
            candidates.extend(
                default_backup_root.glob(f"{self.root.name}_*/cleanup_manifest.json")
            )
        if not candidates:
            return None
        selected = max(set(candidates), key=lambda path: path.stat().st_mtime_ns)
        payload = json.loads(selected.read_text(encoding="utf-8"))
        return payload if payload.get("applied") is True else None

    def run(self, sensor_channel: str | None = None) -> dict[str, Any]:
        blockers: list[str] = []
        raw_index = pd.read_parquet(
            self.root / "reports" / "data_quality" / "raw_input_index_v3.parquet"
        )
        ingestion = pd.read_parquet(
            self.root / "data" / "bronze" / "ingestion_manifest_v3.parquet"
        )
        sync = pd.read_parquet(
            self.root / "data" / "sync" / "synchronization_manifest_v3.parquet"
        )
        segmented = pd.read_parquet(
            self.root / "data" / "segmented" / "segmentation_manifest_v3.parquet"
        )
        gold = pd.read_parquet(self.root / "data" / "gold" / "gold_features_v3.parquet")
        gold_manifest = json.loads(
            (self.root / "data" / "gold" / "manifest_v3.json").read_text(
                encoding="utf-8"
            )
        )
        raw_hash_unchanged = sorted(raw_index["sha256"]) == sorted(
            ingestion["source_sha256"]
        )
        cleanup_baseline = self._cleanup_baseline()
        protected_hashes = (
            cleanup_baseline.get("protected_hashes", {})
            if cleanup_baseline is not None
            else {}
        )
        raw_hash_matches_cleanup_baseline = bool(protected_hashes) and all(
            protected_hashes.get(str(row.source_file)) == str(row.sha256)
            for row in raw_index.itertuples()
        )
        ppg_segments = segmented[
            segmented["sensor_channel"].isin(["ppg_ir", "ppg_red", "ppg_green"])
        ]
        excluded_recordings = set(
            sync.loc[sync["excluded"], "recording_id"].astype(str)
        )
        ecg_monotonic_or_excluded = True
        for ecg_path in sorted(
            (self.root / "data" / "sync").glob("ID-*/*/ecg.parquet")
        ):
            ecg = pd.read_parquet(
                ecg_path,
                columns=[
                    "recording_id",
                    "timestamp_corrected_utc",
                ],
            )
            recording_id = str(ecg["recording_id"].iloc[0])
            corrected = pd.to_datetime(ecg["timestamp_corrected_utc"], utc=True)
            if (
                not corrected.is_monotonic_increasing
                and recording_id not in excluded_recordings
            ):
                ecg_monotonic_or_excluded = False
                break
        lineage_columns = {
            "participant_id",
            "session_id",
            "recording_id",
            "phase",
            "sensor_channel",
            "window_id",
            "source_file",
        }
        data_gates = {
            "raw_hash_unchanged": raw_hash_unchanged,
            "raw_hash_matches_cleanup_baseline": raw_hash_matches_cleanup_baseline,
            "raw_index_rows": len(raw_index),
            "raw_index_complete": bool(
                len(raw_index) == 32 * 2 * 4 and (raw_index["status"] == "ok").all()
            ),
            "ppg_segments_complete": bool(
                len(ppg_segments) == 32 * 2 * 3
                and (ppg_segments["status"] == "ok").all()
            ),
            "gold_participants": int(gold["participant_id"].nunique()),
            "gold_windows": len(gold),
            "feature_count": len(gold_manifest["feature_order"]),
            "routine_rows": int(
                gold["phase"].astype(str).str.lower().eq("routine").sum()
            ),
            "ecg_recordings_excluded": int(sync["excluded"].sum()),
            "ecg_exclusions_documented": bool(
                sync.loc[sync["excluded"], "exclusion_reason"].str.len().gt(0).all()
            ),
            "ecg_monotonic_or_excluded": ecg_monotonic_or_excluded,
            "gold_lineage_complete": bool(
                lineage_columns <= set(gold.columns)
                and gold.loc[:, sorted(lineage_columns)].notna().all().all()
            ),
            "development_participants": int(
                gold.loc[gold["split"] == "train", "participant_id"].nunique()
            ),
            "holdout_participants": int(
                gold.loc[gold["split"] == "parity_pool", "participant_id"].nunique()
            ),
        }
        if not all(
            [
                data_gates["raw_hash_unchanged"],
                data_gates["raw_hash_matches_cleanup_baseline"],
                data_gates["raw_index_complete"],
                data_gates["ppg_segments_complete"],
                data_gates["gold_participants"] == 32,
                data_gates["routine_rows"] == 0,
                data_gates["ecg_monotonic_or_excluded"],
                data_gates["gold_lineage_complete"],
                data_gates["development_participants"] == 28,
                data_gates["holdout_participants"] == 4,
            ]
        ):
            blockers.append("data_gate_failed")

        model_evidence: dict[str, Any] = {}
        if sensor_channel:
            report_path = (
                self.root
                / "reports"
                / "model_eval"
                / "stress_v3.0.0"
                / f"{sensor_channel}_full.json"
            )
            if not report_path.exists():
                report_path = (
                    self.root
                    / "reports"
                    / "model_eval"
                    / "stress_v4.0.0-dl"
                    / f"{sensor_channel}_full.json"
                )
            if report_path.exists():
                model_evidence = json.loads(report_path.read_text(encoding="utf-8"))
                if model_evidence.get("release_status") != "candidate_ready":
                    blockers.append("nested_model_gate_failed")
            else:
                blockers.append("full_model_report_missing")

        deployment: dict[str, Any] = {
            "python_vectors_passed": 0,
            "android_instrumentation_status": "pending",
            "android_ready": False,
        }
        if sensor_channel:
            manifest_path = (
                self.root
                / "data"
                / "artifacts"
                / "export"
                / f"stress_{sensor_channel}_manifest_v3.json"
            )
            if not manifest_path.exists():
                manifest_path = (
                    self.root
                    / "data"
                    / "artifacts"
                    / "export"
                    / "stress_v4.0.0-dl"
                    / f"model_manifest_{sensor_channel}.json"
                )
            if manifest_path.exists():
                onnx_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                deployment.update(
                    {
                        "onnx_parity": onnx_manifest.get("parity_status") == "ok",
                        "deployment_gate_passed": onnx_manifest.get(
                            "deployment_gate_passed", True
                        ),
                    }
                )
                if (
                    not onnx_manifest.get("deployment_gate_passed", True)
                    and not onnx_manifest.get("parity_status") == "ok"
                ):
                    blockers.append("onnx_deployment_gate_failed")
            else:
                blockers.append("onnx_manifest_missing")
            vector_path = (
                self.root
                / "tests"
                / "fixtures"
                / "android_parity"
                / "test_vectors.json"
            )
            if vector_path.exists():
                vectors = json.loads(vector_path.read_text(encoding="utf-8"))
                deployment["python_vectors_passed"] = int(
                    vectors.get("python_vectors_passed", 0)
                )
                deployment["android_instrumentation_status"] = vectors.get(
                    "android_instrumentation_status", "pending"
                )
            if deployment["python_vectors_passed"] != 100:
                blockers.append("python_vector_gate_failed")
        if deployment["android_instrumentation_status"] != "passed":
            blockers.append("physical_android_parity_pending")

        result = {
            "status": "blocked" if blockers else "passed",
            "data_gates": data_gates,
            "model_evidence": model_evidence,
            "deployment": deployment,
            "blockers": sorted(set(blockers)),
        }
        output = self.root / "reports" / "release_validation_v3.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
