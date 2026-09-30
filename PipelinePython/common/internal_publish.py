"""One-time internal-test publication for the run-scoped Android bundle.

This module deliberately sits after development training.  It never tunes a
model and it does not reuse the historical holdout ledger: its only authority
is one frozen candidate and one immutable ID-29..ID-32 evaluation record.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .calibration import PlattCalibration
from .settings import PipelineConfig, PipelinePaths, TRAINING_CHANNELS
from .training_canonical import (
    HOLDOUT,
    CanonicalClassicalTrainer,
    _sha256,
    _write_immutable_text,
    load_hrv_training_admission,
    validate_gold_input,
)

INTERNAL_SCHEMA = "internal-android-v2"
INTERNAL_EVIDENCE_SCHEMA = "internal-test-evidence-v1"


class InternalTestEvidenceStore:
    """Persist exactly one immutable ID-29..ID-32 evaluation per contract."""

    def __init__(self, governance_root: Path) -> None:
        self.root = Path(governance_root).resolve() / "internal_test_evidence"
        self.claim_root = self.root.parent / "internal_test_claims"

    def path_for(self, sensor_channel: str, fingerprint: str) -> Path:
        return self.root / sensor_channel / f"{fingerprint}.json"

    def claim(
        self, sensor_channel: str, fingerprint: str, payload: dict[str, Any]
    ) -> None:
        """Claim before reading an internal-test row from Gold."""

        path = self.claim_root / f"{sensor_channel}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            {
                "schema_version": "internal-test-claim-v1",
                "sensor_channel": sensor_channel,
                "internal_test_contract_fingerprint": fingerprint,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            with path.open("x", encoding="utf-8") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError as exc:
            raise RuntimeError("internal_test_access_already_claimed") from exc

    def load(self, sensor_channel: str, fingerprint: str) -> dict[str, Any] | None:
        path = self.path_for(sensor_channel, fingerprint)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def write_once(
        self, sensor_channel: str, fingerprint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        claim_path = self.claim_root / f"{sensor_channel}.json"
        if not claim_path.is_file():
            raise RuntimeError("internal_test_evidence_write_without_claim")
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        if claim.get("internal_test_contract_fingerprint") != fingerprint:
            raise RuntimeError("internal_test_claim_fingerprint_mismatch")
        path = self.path_for(sensor_channel, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, indent=2, sort_keys=True)
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing != content:
                raise RuntimeError("internal_test_evidence_immutability_violation")
            return json.loads(existing)
        temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8")
        try:
            with temporary.open("r+b") as target:
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return json.loads(content)


def internal_test_evidence_path(
    paths: PipelinePaths, config: PipelineConfig, sensor_channel: str
) -> Path:
    """Return the candidate-local receipt consumed by the internal exporter."""

    return (
        paths.models_dir
        / "stress"
        / sensor_channel
        / config.model_version
        / "development_candidate"
        / "internal_test"
        / "internal_test_evidence.json"
    )


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_paths(
    paths: PipelinePaths, config: PipelineConfig, sensor_channel: str
) -> tuple[Path, Path, Path, Path]:
    bundle = (
        paths.models_dir
        / "stress"
        / sensor_channel
        / config.model_version
        / "development_candidate"
    )
    return (
        bundle,
        bundle / "candidate_manifest.json",
        bundle / "model.joblib",
        bundle / "calibration.json",
    )


def _development_report_path(
    project_root: Path, config: PipelineConfig, sensor_channel: str
) -> Path:
    reports_root = Path(config.artifacts.get("reports_dir", "reports"))
    if not reports_root.is_absolute():
        reports_root = project_root / reports_root
    return (
        reports_root
        / "model_eval"
        / config.model_version
        / f"{sensor_channel}_full.json"
    )


def _read_json(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{code}:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{code}_must_be_object")
    return payload


def _calibration(payload: dict[str, Any]) -> PlattCalibration:
    required = {
        "method",
        "version",
        "slope",
        "intercept",
        "fitted_cohort",
        "fitted_on",
        "calibration_status",
        "brier_score",
        "log_loss",
        "ece",
    }
    if set(payload) != required:
        raise ValueError("internal_test_calibration_contract_mismatch")
    return PlattCalibration(
        method=str(payload["method"]),
        version=str(payload["version"]),
        slope=float(payload["slope"]),
        intercept=float(payload["intercept"]),
        fitted_cohort=tuple(str(value) for value in payload["fitted_cohort"]),
        fitted_on=str(payload["fitted_on"]),
        calibration_status=str(payload["calibration_status"]),
        brier_score=float(payload["brier_score"]),
        log_loss=float(payload["log_loss"]),
        ece=float(payload["ece"]),
    )


def _selected_quality_tiers(
    report: dict[str, Any], config: PipelineConfig
) -> tuple[str, ...]:
    selected = str(report.get("selected_quality_cohort", ""))
    if selected == "strict":
        return tuple(config.training.strict_quality_tiers)
    if selected == "expanded":
        return tuple(config.training.expanded_quality_tiers)
    raise ValueError("internal_test_selected_quality_cohort_missing")


def _validate_frozen_candidate(
    project_root: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    sensor_channel: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, str, dict[str, Any]]:
    """Validate all non-holdout inputs before the one-time access claim."""

    bundle, manifest_path, model_path, calibration_path = _candidate_paths(
        paths, config, sensor_channel
    )
    manifest = _read_json(manifest_path, "internal_test_candidate_manifest_missing")
    report = _read_json(
        _development_report_path(project_root, config, sensor_channel),
        "internal_test_development_report_missing",
    )
    expected_run_id = str(config.artifacts.get("run_id", ""))
    if not expected_run_id or manifest.get("run_id") != expected_run_id:
        raise RuntimeError("internal_test_candidate_run_id_mismatch")
    if manifest.get("schema_version") != "development-candidate-v1":
        raise RuntimeError("internal_test_candidate_schema_mismatch")
    if manifest.get("artifact_status") != "development_candidate":
        raise RuntimeError("internal_test_candidate_status_mismatch")
    if manifest.get("release_status") != "development_candidate":
        raise RuntimeError("internal_test_candidate_not_deferred")
    if manifest.get("sensor_channel") != sensor_channel:
        raise RuntimeError("internal_test_candidate_channel_mismatch")
    if not model_path.is_file() or _sha256(model_path) != manifest.get("model_sha256"):
        raise RuntimeError("internal_test_model_checksum_mismatch")
    if not calibration_path.is_file() or _sha256(calibration_path) != manifest.get(
        "calibration_sha256"
    ):
        raise RuntimeError("internal_test_calibration_checksum_mismatch")
    calibration_payload = _read_json(
        calibration_path, "internal_test_calibration_missing"
    )
    if calibration_payload != manifest.get("probability_calibration"):
        raise RuntimeError("internal_test_calibration_manifest_mismatch")
    _calibration(calibration_payload)
    feature_order = tuple(str(value) for value in manifest.get("feature_order", []))
    if not feature_order or len(feature_order) != len(set(feature_order)):
        raise RuntimeError("internal_test_feature_order_invalid")
    if report.get("candidate_artifact_fingerprint") != manifest.get(
        "candidate_artifact_fingerprint"
    ):
        raise RuntimeError("internal_test_report_candidate_fingerprint_mismatch")
    if tuple(str(value) for value in report.get("feature_order", [])) != feature_order:
        raise RuntimeError("internal_test_report_feature_order_mismatch")
    policy = CanonicalClassicalTrainer(project_root, config, paths).performance_policy
    if not policy.assess(dict(manifest.get("development_metrics", {}))).passes_minimum:
        raise RuntimeError("internal_test_development_gate_failed")
    gold_validation = validate_gold_input(
        project_root, config, paths, sensor_channel=sensor_channel
    )
    if gold_validation.get("status") != "ready":
        raise RuntimeError("internal_test_gold_admission_failed")
    if int(gold_validation.get("observed_training_groups", 0)) < 51:
        raise RuntimeError("internal_test_development_group_coverage_below_51_of_56")
    admission = load_hrv_training_admission(project_root, config, paths)
    if admission is None:
        raise RuntimeError("internal_test_hrv_prv_admission_missing")
    admission_row = admission.get("channels", {}).get(sensor_channel)
    if not isinstance(admission_row, dict) or not (
        admission_row.get("eligible") is True and admission_row.get("decision") == "GO"
    ):
        raise RuntimeError("internal_test_hrv_prv_admission_failed")
    candidate_gold = Path(
        str(manifest.get("gold_contract", {}).get("features_path", ""))
    )
    if not candidate_gold.is_absolute():
        candidate_gold = project_root / candidate_gold
    candidate_gold = candidate_gold.resolve()
    try:
        candidate_gold.relative_to(paths.gold_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("internal_test_candidate_gold_outside_current_run") from exc
    if not candidate_gold.is_file() or _sha256(candidate_gold) != manifest.get(
        "gold_contract", {}
    ).get("features_sha256"):
        raise RuntimeError("internal_test_candidate_gold_checksum_mismatch")
    contract_payload = {
        "schema_version": INTERNAL_EVIDENCE_SCHEMA,
        "run_id": expected_run_id,
        "sensor_channel": sensor_channel,
        "model_version": config.model_version,
        "candidate_artifact_fingerprint": manifest.get(
            "candidate_artifact_fingerprint"
        ),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "model_sha256": _sha256(model_path),
        "calibration_sha256": _sha256(calibration_path),
        "feature_order": list(feature_order),
        "probability_threshold": manifest.get("probability_threshold"),
        "gold_contract": manifest.get("gold_contract"),
        "run_contract_sha256": manifest.get("run_contract_sha256"),
        "performance_policy_sha256": manifest.get("performance_policy_sha256"),
        "selected_quality_cohort": report.get("selected_quality_cohort"),
        "hrv_training_admission_sha256": _sha256(
            paths.validation_dir
            / "hrv_agreement"
            / "selection"
            / "hrv_training_admission.json"
        ),
    }
    return (
        manifest,
        report,
        model_path,
        calibration_path,
        candidate_gold,
        _canonical_fingerprint(contract_payload),
        contract_payload,
    )


def _candidate_receipt(
    paths: PipelinePaths,
    config: PipelineConfig,
    sensor_channel: str,
    payload: dict[str, Any],
) -> Path:
    path = internal_test_evidence_path(paths, config, sensor_channel)
    _write_immutable_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def _evaluate_once(
    project_root: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    sensor_channel: str,
    *,
    manifest: dict[str, Any],
    report: dict[str, Any],
    model_path: Path,
    calibration_path: Path,
    candidate_gold: Path,
    fingerprint: str,
    contract_payload: dict[str, Any],
) -> dict[str, Any]:
    store = InternalTestEvidenceStore(
        project_root / "data" / "artifacts" / "governance"
    )
    existing = store.load(sensor_channel, fingerprint)
    if existing is not None:
        _candidate_receipt(paths, config, sensor_channel, existing)
        return {**existing, "internal_test_access": "reused_internal_test_once"}
    store.claim(
        sensor_channel,
        fingerprint,
        {
            "run_id": str(config.artifacts.get("run_id", "")),
            "model_version": config.model_version,
            "status": "claimed_before_read",
        },
    )
    try:
        trainer = CanonicalClassicalTrainer(project_root, config, paths)
        gold = trainer._load_gold(
            sensor_channel,
            source_override=candidate_gold,
            participant_ids=HOLDOUT,
        )
        gold = gold[
            gold["quality_tier"].isin(_selected_quality_tiers(report, config))
        ].copy()
        features = tuple(str(value) for value in manifest["feature_order"])
        finite = np.isfinite(gold.loc[:, features].to_numpy(dtype=np.float64)).all(
            axis=1
        )
        gold = gold.loc[finite].copy()
        if not trainer._has_complete_group_coverage(gold, HOLDOUT):
            raise ValueError("selected_quality_cohort_missing_8_test_groups")
        model = joblib.load(model_path)
        calibration = _calibration(
            _read_json(calibration_path, "internal_test_calibration_missing")
        )
        raw_probability = model.predict_proba(gold.loc[:, features])[:, 1]
        probability = calibration.apply(raw_probability)
        metrics, sessions = trainer._session_metrics(
            gold, probability, float(manifest["probability_threshold"])
        )
        raw_metrics, _ = trainer._session_metrics(
            gold, raw_probability, float(manifest["probability_threshold"])
        )
        assessment = trainer.performance_policy.assess(metrics)
        status = "GO" if assessment.passes_minimum else "NO_GO"
        payload = {
            "schema_version": INTERNAL_EVIDENCE_SCHEMA,
            "validation_scope": "internal_32_participants",
            "claims_policy": "decision_support_only",
            "run_id": str(config.artifacts.get("run_id", "")),
            "sensor_channel": sensor_channel,
            "model_version": config.model_version,
            "internal_test_contract_fingerprint": fingerprint,
            "frozen_contract": contract_payload,
            "internal_test_participants": list(HOLDOUT),
            "internal_test_session_count": int(len(sessions)),
            "internal_test_status": status,
            "internal_test_metrics": metrics,
            "internal_test_raw_metrics": raw_metrics,
            "internal_test_assessment": assessment.as_dict(),
            "internal_test_access": "evaluated_once",
        }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        payload = {
            "schema_version": INTERNAL_EVIDENCE_SCHEMA,
            "validation_scope": "internal_32_participants",
            "claims_policy": "decision_support_only",
            "run_id": str(config.artifacts.get("run_id", "")),
            "sensor_channel": sensor_channel,
            "model_version": config.model_version,
            "internal_test_contract_fingerprint": fingerprint,
            "frozen_contract": contract_payload,
            "internal_test_participants": list(HOLDOUT),
            "internal_test_status": "INSUFFICIENT_EVIDENCE",
            "reason": str(exc),
            "internal_test_access": "evaluated_once",
        }
    evidence = store.write_once(sensor_channel, fingerprint, payload)
    _candidate_receipt(paths, config, sensor_channel, evidence)
    return evidence


def publish_internal_tests(
    project_root: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    sensor: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate frozen development candidates on the internal cohort once."""

    if sensor is not None and sensor not in TRAINING_CHANNELS:
        raise ValueError(f"internal_test_sensor_not_allowed:{sensor}")
    results: dict[str, dict[str, Any]] = {}
    for sensor_channel in (sensor,) if sensor else TRAINING_CHANNELS:
        try:
            details = _validate_frozen_candidate(
                project_root, config, paths, sensor_channel
            )
            evidence = _evaluate_once(
                project_root,
                config,
                paths,
                sensor_channel,
                manifest=details[0],
                report=details[1],
                model_path=details[2],
                calibration_path=details[3],
                candidate_gold=details[4],
                fingerprint=details[5],
                contract_payload=details[6],
            )
            results[sensor_channel] = evidence
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            results[sensor_channel] = {
                "sensor_channel": sensor_channel,
                "internal_test_status": "NO_GO",
                "reason": str(exc),
            }
    return results
