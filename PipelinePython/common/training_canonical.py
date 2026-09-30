"""Canonical participant-safe classical training for the local stress pipeline.

This module is deliberately independent from the legacy random-split trainer.
It owns the fixed ID-01..ID-28 development / ID-29..ID-32 holdout protocol and
never evaluates the holdout until the development candidate has passed all
locked gates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .calibration import fit_platt_from_nested_oof
from .feature_policy import correlation_pruned_features
from .feature_policy import DEFAULT_PREDICTORS
from .model_selection import ClassicalModelFactory, GroupedHyperparameterOptimizer
from .performance_policy import policy_from_training_config
from .settings import PipelineConfig, PipelinePaths, TRAINING_CHANNELS
from .training_protocol import (
    aggregate_session_probabilities,
    channel_group_readiness,
    outer_participant_folds,
    rank_model_candidates,
    select_quality_cohort,
    select_session_threshold,
)

logger = logging.getLogger(__name__)

DEVELOPMENT = tuple(f"ID-{index:02d}" for index in range(1, 29))
HOLDOUT = tuple(f"ID-{index:02d}" for index in range(29, 33))
THRESHOLD = 0.5
SPECTRAL_ABLATION_FEATURES = (
    "log_lf_power",
    "log_hf_power",
    "lf_hf_ratio",
)


def _conservative_gate_metrics(
    development_metrics: dict[str, Any], holdout_metrics: dict[str, Any]
) -> dict[str, float]:
    """Return the weaker gated evidence across development and internal test.

    A preferred release tier requires every gated metric to meet its preferred
    threshold on both cohorts. Taking the per-metric minimum prevents a strong
    reused internal test from masking a development result that is only suitable
    for the research tier.
    """

    evidence: dict[str, float] = {}
    for metric in ("balanced_accuracy", "f1", "roc_auc"):
        try:
            development_value = float(development_metrics[metric])
            holdout_value = float(holdout_metrics[metric])
        except (KeyError, TypeError, ValueError):
            evidence[metric] = float("nan")
            continue
        evidence[metric] = min(development_value, holdout_value)
    return evidence


def _configured_path(
    project_root: Path, config: PipelineConfig, key: str, default: Path
) -> Path:
    """Resolve an artifact path without changing the configured source."""

    configured = Path(config.artifacts.get(key, default))
    return configured if configured.is_absolute() else project_root / configured


def participant_phase_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every participant-phase group one total training weight."""

    required = {"participant_id", "phase"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weight_groups_missing:{','.join(missing)}")
    keys = frame["participant_id"].astype(str) + "|" + frame["phase"].astype(str)
    counts = keys.value_counts(sort=False)
    return keys.map(lambda key: 1.0 / float(counts[key])).to_numpy(dtype=float)


def _configured_predictors(config: object) -> tuple[str, ...]:
    """Return the locked predictor order while keeping protocol helpers tiny."""

    features = getattr(config, "features", None)
    configured = getattr(features, "predictor_default", DEFAULT_PREDICTORS)
    return tuple(configured)


def validate_gold_input(
    project_root: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    sensor_channel: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable Gold input before local ML reads any rows.

    The returned hashes are evidence for the isolated ML run. No source Gold
    artifact is written by this function.
    """

    input_root = _configured_path(
        project_root, config, "gold_input_root", paths.gold_dir
    )
    feature_path = _configured_path(
        project_root,
        config,
        "gold_input_features",
        input_root / "gold_features_v3.parquet",
    )
    ibi_path = _configured_path(
        project_root,
        config,
        "gold_input_ibi_sequences",
        input_root / "gold_ibi_sequences_v3.parquet",
    )
    readiness_path = _configured_path(
        project_root,
        config,
        "gold_input_readiness",
        input_root / "data_readiness.json",
    )
    if sensor_channel is not None and sensor_channel not in TRAINING_CHANNELS:
        raise ValueError(f"unsupported_training_channel:{sensor_channel}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Gold feature artifact does not exist: {feature_path}")
    if not ibi_path.is_file():
        raise FileNotFoundError(f"Gold IBI artifact does not exist: {ibi_path}")
    if not readiness_path.is_file():
        raise FileNotFoundError(
            f"Gold readiness artifact does not exist: {readiness_path}"
        )

    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    manifest_path = input_root / "manifest_v3.json"
    strict_run_scope = bool(config.artifacts.get("run_id"))
    if strict_run_scope and not manifest_path.is_file():
        raise FileNotFoundError(f"Gold manifest does not exist: {manifest_path}")
    if strict_run_scope:
        try:
            input_root.resolve().relative_to(paths.gold_dir.resolve())
        except ValueError as exc:
            raise ValueError("gold_input_root_must_be_current_run") from exc
        try:
            feature_path.resolve().relative_to(input_root.resolve())
            ibi_path.resolve().relative_to(input_root.resolve())
            readiness_path.resolve().relative_to(input_root.resolve())
        except ValueError as exc:
            raise ValueError("gold_artifact_must_be_inside_current_run") from exc
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    expected_run_id = str(config.artifacts.get("run_id", ""))
    manifest_run_id = str(manifest.get("run_id", ""))
    readiness_run_id = str(readiness.get("run_id", ""))
    if strict_run_scope and manifest_run_id != expected_run_id:
        raise ValueError("gold_run_id_mismatch")
    if strict_run_scope and readiness_run_id != expected_run_id:
        raise ValueError("gold_readiness_run_id_mismatch")
    predictors = _configured_predictors(config)
    if strict_run_scope and manifest.get("feature_order") != list(predictors):
        raise ValueError("gold_feature_order_mismatch")
    if strict_run_scope and int(manifest.get("feature_count", 0)) != len(predictors):
        raise ValueError("gold_feature_count_mismatch")

    def digest(path: Path) -> str:
        return _sha256(path)

    feature_sha256 = digest(feature_path)
    if strict_run_scope:
        manifest_checksum = str(
            manifest.get("artifact_checksums", {}).get("gold_features_v3.parquet", "")
        )
        readiness_checksum = str(
            readiness.get("gold_artifact_checksums", {}).get(
                "gold_features_v3.parquet", ""
            )
        )
        if manifest_checksum != feature_sha256:
            raise ValueError("gold_feature_checksum_mismatch")
        if readiness_checksum != feature_sha256:
            raise ValueError("gold_readiness_feature_checksum_mismatch")
        contract_path = paths.artifacts_root / "run_contract_fingerprint.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"run_contract_fingerprint_missing:{contract_path}")
        contract_sha256 = digest(contract_path)
        if str(manifest.get("run_contract_sha256", "")) != contract_sha256:
            raise ValueError("gold_manifest_run_contract_mismatch")
        if str(readiness.get("run_contract_sha256", "")) != contract_sha256:
            raise ValueError("gold_readiness_run_contract_mismatch")
    expected = int(
        readiness.get(
            "expected_training_groups",
            readiness.get(
                "expected_ppg_groups", len(config.training.training_channels) * 64
            ),
        )
    )
    observed = int(
        readiness.get(
            "observed_training_groups", readiness.get("observed_ppg_groups", 0)
        )
    )
    channel_status = readiness.get("training_channel_status", {})

    development_coverage = readiness.get("training_development_coverage", {})
    selected_development = (
        development_coverage.get(sensor_channel, {})
        if sensor_channel is not None and isinstance(development_coverage, dict)
        else {}
    )
    selected_expected = (
        int(selected_development.get("expected_groups", 56))
        if sensor_channel is not None
        else expected
    )
    selected_observed = (
        int(selected_development.get("observed_groups", 0))
        if sensor_channel is not None
        else observed
    )
    selected_status = (
        str(channel_status.get(sensor_channel, "blocked"))
        if sensor_channel is not None
        else str(readiness.get("training_status", "blocked"))
    )
    coverage_threshold = float(
        getattr(
            config.training,
            "minimum_training_development_group_coverage",
            0.90,
        )
    )
    coverage_fraction = selected_observed / max(selected_expected, 1)
    coverage_passed = coverage_fraction >= coverage_threshold
    coverage_complete = selected_observed >= selected_expected
    status = "ready" if selected_status == "ready" and coverage_passed else "blocked"

    return {
        "features_path": str(feature_path.resolve()),
        "features_sha256": feature_sha256,
        "ibi_sequences_path": str(ibi_path.resolve()),
        "ibi_sequences_sha256": digest(ibi_path),
        "readiness_path": str(readiness_path.resolve()),
        "readiness_sha256": digest(readiness_path),
        "manifest_path": str(manifest_path.resolve())
        if manifest_path.is_file()
        else None,
        "manifest_sha256": digest(manifest_path) if manifest_path.is_file() else None,
        "ppg_gold_status": readiness.get("ppg_gold_status"),
        "training_status": selected_status,
        "expected_training_groups": selected_expected,
        "observed_training_groups": selected_observed,
        "training_group_coverage_fraction": coverage_fraction,
        "training_group_coverage_threshold": coverage_threshold,
        "training_group_coverage_passed": coverage_passed,
        "gold_group_coverage_status": readiness.get(
            "gold_group_coverage_status", readiness.get("ppg_gold_status")
        ),
        "gold_completeness_status": readiness.get(
            "gold_completeness_status", "unknown"
        ),
        "training_channels": list(config.training.training_channels),
        "training_channel_status": channel_status,
        "sensor_channel": sensor_channel,
        "status": status,
        "run_id": manifest_run_id or expected_run_id or input_root.name,
        "source_tree_sha256": manifest.get("source_tree_sha256", "UNAVAILABLE"),
        "config_fingerprint": manifest.get("config_fingerprint", "UNAVAILABLE"),
        "run_contract_sha256": manifest.get("run_contract_sha256", "UNAVAILABLE"),
        "blocking_reason": (
            None
            if status == "ready"
            else (
                f"training_channel_status:{selected_status};"
                f"observed_groups={selected_observed}/{selected_expected};"
                f"coverage_passed={str(coverage_passed).lower()};"
                f"coverage_complete={str(coverage_complete).lower()}"
            )
        ),
    }


def load_hrv_training_admission(
    project_root: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
) -> dict[str, Any] | None:
    """Load and verify the development-only HRV/PRV admission contract.

    A run-scoped training call must consume the selection artifact produced by
    the preceding physiology stage. Unscoped compatibility callers retain
    their historical behavior and do not create research evidence.
    """

    del project_root
    run_id = str(config.artifacts.get("run_id", ""))
    if not run_id:
        return None
    admission_path = (
        paths.validation_dir
        / "hrv_agreement"
        / "selection"
        / "hrv_training_admission.json"
    )
    if not admission_path.is_file():
        raise FileNotFoundError(
            f"HRV/PRV training admission is required before training: {admission_path}"
        )
    payload = json.loads(admission_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hrv_training_admission_must_be_an_object")
    if payload.get("schema_version") != "hrv-training-admission-v1":
        raise ValueError("hrv_training_admission_schema_mismatch")
    if payload.get("evaluation_scope") != "selection":
        raise ValueError("hrv_training_admission_scope_mismatch")
    if payload.get("selection_participants") != list(DEVELOPMENT):
        raise ValueError("hrv_training_admission_participant_scope_mismatch")
    if payload.get("primary_domain") != "time_domain":
        raise ValueError("hrv_training_admission_domain_mismatch")
    channels = payload.get("channels")
    if not isinstance(channels, dict) or set(channels) != set(TRAINING_CHANNELS):
        raise ValueError("hrv_training_admission_channel_scope_mismatch")
    if str(payload.get("run_id", "")) != run_id:
        raise ValueError("hrv_training_admission_run_id_mismatch")

    contract_path = paths.artifacts_root / "run_contract_fingerprint.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"run_contract_fingerprint_missing:{contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_identity = {
        "source_tree_sha256": contract.get("source_tree_sha256"),
        "config_fingerprint": contract.get("config_fingerprint"),
        "run_contract_sha256": _sha256(contract_path),
    }
    for key, expected in expected_identity.items():
        if str(payload.get(key, "")) != str(expected):
            raise ValueError(f"hrv_training_admission_{key}_mismatch")
    return payload


def _metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float = THRESHOLD
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = 0.0
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": auc,
        "stress_sensitivity": float(
            true_positive / sensitivity_denominator if sensitivity_denominator else 0.0
        ),
        "relax_specificity": float(
            true_negative / specificity_denominator if specificity_denominator else 0.0
        ),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": matrix.tolist(),
    }


def _passes(metrics: dict[str, Any], config: PipelineConfig) -> bool:
    """Return whether all minimum-research metrics pass."""

    return policy_from_training_config(config.training).assess(metrics).passes_minimum


def spectral_ablation_is_eligible(
    default_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    default_row_count: int,
    candidate_row_count: int,
) -> bool:
    """Apply the locked development-only spectral promotion rule."""

    return bool(
        candidate_row_count == default_row_count
        and float(candidate_metrics["balanced_accuracy"])
        >= float(default_metrics["balanced_accuracy"]) + 0.02
        and float(candidate_metrics["fold_balanced_accuracy_std"])
        <= float(default_metrics["fold_balanced_accuracy_std"])
    )


def _fingerprint(frame: pd.DataFrame, feature_order: tuple[str, ...]) -> str:
    columns = (
        "participant_id",
        "phase",
        "sensor_channel",
        "label",
        *feature_order,
    )
    canonical = frame.loc[:, columns].sort_values(list(columns), kind="mergesort")
    payload = canonical.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_text(path: Path, content: str) -> None:
    """Create one versioned artifact, allowing only byte-identical reruns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"artifact_immutability_violation:{path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dump_joblib_immutable(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    if path.exists():
        if (
            hashlib.sha256(path.read_bytes()).digest()
            != hashlib.sha256(temporary.read_bytes()).digest()
        ):
            temporary.unlink()
            raise RuntimeError(f"artifact_immutability_violation:{path}")
        temporary.unlink()
        return
    temporary.replace(path)


def _canonical_json_cell(value: Any) -> Any:
    """Make nested report values deterministic and Arrow-compatible."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        normalized = value.tolist() if isinstance(value, np.ndarray) else value
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: (
                item.item() if isinstance(item, np.generic) else str(item)
            ),
        )
    return value


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(_canonical_json_cell)
    return result


class HoldoutEvidenceStore:
    """Atomic immutable evidence keyed by one frozen holdout contract."""

    def __init__(self, governance_root: Path) -> None:
        self.root = Path(governance_root).resolve() / "holdout_evidence"

    def path_for(self, sensor_channel: str, contract_fingerprint: str) -> Path:
        return self.root / sensor_channel / f"{contract_fingerprint}.json"

    def claim_path_for(self, sensor_channel: str) -> Path:
        return self.root.parent / "holdout_claims" / f"{sensor_channel}.json"

    def claim(
        self,
        sensor_channel: str,
        contract_fingerprint: str,
        payload: dict[str, Any],
    ) -> None:
        """Claim the channel before any reused-test value is read."""

        path = self.claim_path_for(sensor_channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        claim = json.dumps(
            {
                "schema_version": "holdout-claim-v1",
                "sensor_channel": sensor_channel,
                "contract_fingerprint": contract_fingerprint,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            with path.open("x", encoding="utf-8") as target:
                target.write(claim)
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError as exc:
            raise RuntimeError("holdout_access_already_claimed") from exc

    def load(
        self, sensor_channel: str, contract_fingerprint: str
    ) -> dict[str, Any] | None:
        path = self.path_for(sensor_channel, contract_fingerprint)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def write_once(
        self,
        sensor_channel: str,
        contract_fingerprint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        path = self.path_for(sensor_channel, contract_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        canonical = json.dumps(payload, indent=2, sort_keys=True)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != canonical:
                raise RuntimeError("holdout_evidence_immutability_violation")
            return json.loads(existing)
        temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(canonical, encoding="utf-8")
        try:
            # Windows requires a writable descriptor for a reliable flush to
            # stable storage.  The file contents are already complete here;
            # opening r+b only provides a descriptor accepted by fsync.
            with temporary.open("r+b") as source:
                source.flush()
                os.fsync(source.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return json.loads(canonical)


class CanonicalClassicalTrainer:
    """Train exportable classical baselines under the locked local protocol."""

    def __init__(
        self,
        project_root: Path,
        config: PipelineConfig,
        paths: PipelinePaths,
        *,
        defer_reused_internal_test: bool = False,
    ) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.paths = paths
        self.performance_policy = policy_from_training_config(config.training)
        self._strict_run_scope = bool(getattr(config, "artifacts", {}).get("run_id"))
        self._gold_contract: dict[str, Any] = {}
        self.holdout_evidence = HoldoutEvidenceStore(
            self.root / "data" / "artifacts" / "governance"
        )
        self.defer_reused_internal_test = defer_reused_internal_test
        self._report_context: dict[str, Any] = {}

    def _optimizer(self) -> GroupedHyperparameterOptimizer:
        storage_path = self.paths.artifacts_root / "optuna" / "training.db"
        storage_uri = f"sqlite:///{storage_path.resolve().as_posix()}"
        return GroupedHyperparameterOptimizer(
            self.config.training,
            ClassicalModelFactory(self.config.training.random_seed),
            storage_uri=storage_uri,
        )

    def _release_candidate_status(self) -> str:
        """Use the canonical status only for an explicitly isolated run."""

        return "release_candidate" if self._strict_run_scope else "candidate_ready"

    def _blocked_coverage_report(
        self,
        sensor_channel: str,
        reason: str,
        *,
        feature_order: tuple[str, ...] = (),
        gold_input_validation: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = {
            "schema_version": "1.1.0",
            "model_version": self.config.model_version,
            "sensor_channel": sensor_channel,
            "model_family": "classical",
            "model_name": None,
            "feature_order": list(feature_order),
            "development_participants": list(DEVELOPMENT),
            "expected_development_participants": list(DEVELOPMENT),
            "represented_development_participants": [],
            "missing_development_participants": list(DEVELOPMENT),
            "holdout_participants": list(HOLDOUT),
            "holdout_evaluated": False,
            "channel_status": "not_evaluated",
            "requires_dl": False,
            "trigger_reason": None,
            "release_status": "blocked_insufficient_ppg_coverage",
            "blocking_reason": reason,
        }
        if gold_input_validation is not None:
            report["gold_input_validation"] = gold_input_validation
        if extra is not None:
            report.update(extra)
        self._write_report(sensor_channel, report)
        return report

    def _holdout_contract_fingerprint(
        self,
        *,
        sensor_channel: str,
        selected_model: str,
        selected_parameters: dict[str, Any],
        feature_order: tuple[str, ...],
        data_fingerprint: str,
        decision_threshold: float,
    ) -> str:
        payload = {
            "sensor_channel": sensor_channel,
            "selected_model": selected_model,
            "selected_parameters": selected_parameters,
            "feature_order": list(feature_order),
            "data_fingerprint": data_fingerprint,
            "preprocessing_version": self.config.preprocessing_version,
            "feature_version": self.config.feature_version,
            "threshold": decision_threshold,
            "random_seed": self.config.training.random_seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _legacy_holdout_contract_fingerprint(
        self,
        *,
        sensor_channel: str,
        selected_model: str,
        selected_parameters: dict[str, Any],
        feature_order: tuple[str, ...],
        data_fingerprint: str,
        decision_threshold: float,
    ) -> str:
        """Recreate the pre-tier fingerprint to reuse immutable evidence once."""

        payload = {
            "sensor_channel": sensor_channel,
            "selected_model": selected_model,
            "selected_parameters": selected_parameters,
            "feature_order": list(feature_order),
            "data_fingerprint": data_fingerprint,
            "preprocessing_version": self.config.preprocessing_version,
            "feature_version": self.config.feature_version,
            "threshold": decision_threshold,
            "random_seed": self.config.training.random_seed,
            "gates": self.performance_policy.preferred.as_dict(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _release_policy_fingerprint(self) -> str:
        return policy_from_training_config(self.config.training).fingerprint()

    def _freeze_development_candidate(
        self,
        sensor_channel: str,
        model: object,
        report: dict[str, Any],
        gold_contract: dict[str, Any],
    ) -> Path:
        """Persist a development-only model without reading reused-test data."""

        calibration = report.get("probability_calibration")
        if not isinstance(calibration, dict) or not calibration:
            raise RuntimeError("development_candidate_calibration_missing")
        bundle = (
            self.paths.models_dir
            / "stress"
            / sensor_channel
            / self.config.model_version
            / "development_candidate"
        )
        model_path = bundle / "model.joblib"
        calibration_path = bundle / "calibration.json"
        _dump_joblib_immutable(model, model_path)
        _write_immutable_text(
            calibration_path,
            json.dumps(calibration, indent=2, sort_keys=True),
        )
        model_sha256 = _sha256(model_path)
        calibration_sha256 = _sha256(calibration_path)
        run_contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
        run_contract = (
            json.loads(run_contract_path.read_text(encoding="utf-8"))
            if run_contract_path.is_file()
            else {}
        )
        artifact_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "holdout_contract_fingerprint": report.get(
                        "holdout_contract_fingerprint"
                    ),
                    "model_sha256": model_sha256,
                    "calibration_sha256": calibration_sha256,
                    "probability_threshold": report.get("threshold"),
                    "feature_order": report.get("feature_order", []),
                    "gold_contract": gold_contract,
                    "preprocessing_version": self.config.preprocessing_version,
                    "feature_version": self.config.feature_version,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        metadata = {
            "schema_version": "development-candidate-v1",
            "artifact_status": "development_candidate",
            "evaluation_scope": "development_only",
            "release_status": "development_candidate",
            "deployment_scope": "none",
            "android_ready": False,
            "holdout_evaluated": False,
            "model_version": self.config.model_version,
            "sensor_channel": sensor_channel,
            "model_family": "classical",
            "model_name": report.get("model_name"),
            "selected_parameters": report.get("selected_parameters", {}),
            "feature_order": list(report.get("feature_order", [])),
            "predictor_columns": list(report.get("feature_order", [])),
            "canonical_predictors": list(_configured_predictors(self.config)),
            "model_input_feature_order": list(report.get("feature_order", [])),
            "calibration_input_feature_order": list(
                report.get("calibration_input_feature_order", [])
            ),
            "feature_version": self.config.feature_version,
            "preprocessing_version": self.config.preprocessing_version,
            "native_model_path": str(model_path.relative_to(self.root)).replace(
                "\\", "/"
            ),
            "model_sha256": model_sha256,
            "calibration_path": str(calibration_path.relative_to(self.root)).replace(
                "\\", "/"
            ),
            "calibration_sha256": calibration_sha256,
            "candidate_artifact_fingerprint": artifact_fingerprint,
            "analysis_seconds": int(self.config.window.analysis_phase_duration_seconds),
            "selected_window_seconds": int(
                report.get(
                    "selected_window_seconds",
                    self.config.training.window_baseline_seconds,
                )
            ),
            "selected_stride_seconds": int(
                report.get(
                    "selected_stride_seconds",
                    self.config.training.window_baseline_stride_seconds,
                )
            ),
            "development_participants": list(DEVELOPMENT),
            "expected_development_participants": list(DEVELOPMENT),
            "represented_development_participants": list(
                report.get("represented_development_participants", [])
            ),
            "missing_development_participants": list(
                report.get("missing_development_participants", [])
            ),
            "probability_threshold": float(report["threshold"]),
            "threshold_source": report.get("threshold_source"),
            "development_metrics": report.get("development_metrics", {}),
            "development_raw_metrics": report.get("development_raw_metrics", {}),
            "probability_calibration": calibration,
            "channel_data_fingerprint": report.get("channel_data_fingerprint"),
            "fingerprint_scope": report.get("fingerprint_scope"),
            "holdout_contract_fingerprint": report.get("holdout_contract_fingerprint"),
            "model_contract_fingerprint": report.get("model_contract_fingerprint"),
            "release_policy_fingerprint": self._release_policy_fingerprint(),
            "performance_policy": self.performance_policy.as_dict(),
            "performance_policy_version": self.performance_policy.version,
            "performance_policy_sha256": self.performance_policy.fingerprint(),
            "performance_tier": report.get("performance_tier"),
            "research_disclaimer": self.performance_policy.research_disclaimer,
            "gold_contract": gold_contract,
            "run_id": str(self.config.artifacts.get("run_id", "")),
            "run_contract_sha256": (
                _sha256(run_contract_path) if run_contract_path.is_file() else None
            ),
            "source_tree_sha256": run_contract.get("source_tree_sha256"),
            "config_fingerprint": run_contract.get("config_fingerprint"),
            "split_fingerprint": run_contract.get("split_fingerprint"),
            "source_checksums": run_contract.get("source_checksums", {}),
            "converter_preflight": report.get("converter_preflight", {}),
            "test_cohort_status": self.config.training.reused_test_cohort_status,
            "clinical_validation_status": self.config.training.clinical_validation_status,
            "cv_gates_passed": True,
        }
        _write_immutable_text(
            bundle / "candidate_manifest.json",
            json.dumps(metadata, indent=2, sort_keys=True),
        )
        report["development_candidate_bundle"] = str(
            bundle.relative_to(self.root)
        ).replace("\\", "/")
        report["candidate_artifact_fingerprint"] = artifact_fingerprint
        return bundle

    def _holdout_report_payloads(
        self, sensor_channel: str
    ) -> list[tuple[Path, dict[str, Any]]]:
        """Load historical channel reports once for governance classification."""

        if self._strict_run_scope:
            return []
        patterns = (
            self.root / "reports" / "runs",
            self.root / "reports" / "model_eval",
        )
        candidates = list(
            patterns[0].glob(f"*/model_eval/*/{sensor_channel}_full.json")
        )
        candidates.extend(patterns[1].glob(f"*/{sensor_channel}_full.json"))
        payloads: list[tuple[Path, dict[str, Any]]] = []
        seen: set[Path] = set()
        for path in sorted(candidates, key=lambda candidate: candidate.as_posix()):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payloads.append((resolved, payload))
        return payloads

    @staticmethod
    def _is_explicit_contract_fingerprint(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @classmethod
    def _is_fingerprinted_holdout_evidence(cls, payload: dict[str, Any]) -> bool:
        return payload.get("holdout_evaluated") is True and any(
            cls._is_explicit_contract_fingerprint(payload.get(key))
            for key in ("model_contract_fingerprint", "holdout_contract_fingerprint")
        )

    def _find_prior_holdout_evidence(
        self,
        sensor_channel: str,
        contract_fingerprint: str,
        legacy_contract_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        stored = self.holdout_evidence.load(sensor_channel, contract_fingerprint)
        if stored is not None:
            return stored
        if legacy_contract_fingerprint is not None:
            stored = self.holdout_evidence.load(
                sensor_channel, legacy_contract_fingerprint
            )
            if stored is not None:
                return stored
        for _, payload in self._holdout_report_payloads(sensor_channel):
            fingerprints = {contract_fingerprint}
            if legacy_contract_fingerprint is not None:
                fingerprints.add(legacy_contract_fingerprint)
            if (
                payload.get("model_contract_fingerprint") in fingerprints
                or payload.get("holdout_contract_fingerprint") in fingerprints
            ) and payload.get("holdout_evaluated") is True:
                return payload
        return None

    def _has_any_prior_holdout_evidence(self, sensor_channel: str) -> bool:
        """Return only governed or explicitly fingerprinted evidence.

        A historical report with ``holdout_evaluated=true`` but no frozen
        contract fingerprint is preserved for audit, yet cannot be reused or
        block exactly one current canonical evaluation.
        """

        if any((self.holdout_evidence.root / sensor_channel).glob("*.json")):
            return True
        if self.holdout_evidence.claim_path_for(sensor_channel).is_file():
            return True
        return any(
            self._is_fingerprinted_holdout_evidence(payload)
            for _, payload in self._holdout_report_payloads(sensor_channel)
        )

    def _historical_unverifiable_holdout_reports(
        self, sensor_channel: str
    ) -> tuple[str, ...]:
        """Return retained holdout reports that predate contract fingerprinting."""

        reports: list[str] = []
        for path, payload in self._holdout_report_payloads(sensor_channel):
            if payload.get("holdout_evaluated") is not True:
                continue
            if self._is_fingerprinted_holdout_evidence(payload):
                continue
            try:
                reports.append(path.relative_to(self.root).as_posix())
            except ValueError:
                reports.append(path.as_posix())
        return tuple(reports)

    def _record_holdout_access(
        self,
        sensor_channel: str,
        contract_fingerprint: str,
        report: dict[str, Any],
        historical_unverifiable_reports: tuple[str, ...] = (),
    ) -> None:
        evidence = {
            "schema_version": "holdout-evidence-v2",
            "sensor_channel": sensor_channel,
            "contract_fingerprint": contract_fingerprint,
            "model_contract_fingerprint": contract_fingerprint,
            "release_policy_fingerprint": self._release_policy_fingerprint(),
            "model_version": self.config.model_version,
            "holdout_evaluated": True,
            "release_status": report.get("release_status"),
            "holdout_metrics": report.get("holdout_metrics"),
            "holdout_raw_metrics": report.get("holdout_raw_metrics"),
            "internal_test_session_count": report.get("internal_test_session_count"),
            "holdout_access_classification": "current_canonical_once",
            "historical_unverifiable_report_count": len(
                historical_unverifiable_reports
            ),
            "historical_unverifiable_reports": list(historical_unverifiable_reports),
        }
        self.holdout_evidence.write_once(sensor_channel, contract_fingerprint, evidence)
        ledger_path = self.holdout_evidence.root.parent / "holdout_access_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = (
            json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_path.is_file()
            else {"schema_version": "holdout-access-ledger-v1", "accesses": []}
        )
        if any(
            item.get("contract_fingerprint") == contract_fingerprint
            for item in ledger.get("accesses", [])
        ):
            return
        ledger.setdefault("accesses", []).append(
            {
                "sensor_channel": sensor_channel,
                "contract_fingerprint": contract_fingerprint,
                "model_version": self.config.model_version,
                "release_status": report.get("release_status"),
                "holdout_metrics": report.get("holdout_metrics"),
                "holdout_access_classification": "current_canonical_once",
                "historical_unverifiable_report_count": len(
                    historical_unverifiable_reports
                ),
                "historical_unverifiable_reports": list(
                    historical_unverifiable_reports
                ),
            }
        )
        temporary = ledger_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(ledger_path)

    def _load_gold(
        self,
        sensor_channel: str,
        source_override: Path | None = None,
        participant_ids: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        input_root = _configured_path(
            self.root, self.config, "gold_input_root", self.paths.gold_dir
        ).resolve()
        source = (
            Path(source_override).resolve()
            if source_override is not None
            else _configured_path(
                self.root,
                self.config,
                "gold_input_features",
                input_root / "gold_features_v3.parquet",
            ).resolve()
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"Current-run Gold feature table is missing: {source}"
            )
        if self._strict_run_scope and input_root != self.paths.gold_dir.resolve():
            raise ValueError("gold_input_root_must_be_current_run")
        try:
            source.relative_to(input_root)
        except ValueError as exc:
            raise ValueError("gold_source_must_be_inside_current_run") from exc
        manifest_path = input_root / "manifest_v3.json"
        readiness_path = input_root / "data_readiness.json"
        if not manifest_path.is_file() or not readiness_path.is_file():
            raise FileNotFoundError("current_run_gold_contract_missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        expected_run_id = str(self.config.artifacts.get("run_id", ""))
        manifest_run_id = str(manifest.get("run_id", ""))
        if expected_run_id and manifest_run_id != expected_run_id:
            raise ValueError("gold_run_id_mismatch")
        if expected_run_id and str(readiness.get("run_id", "")) != expected_run_id:
            raise ValueError("gold_readiness_run_id_mismatch")
        predictors = _configured_predictors(self.config)
        if manifest.get("feature_order") != list(predictors):
            raise ValueError("gold_feature_order_mismatch")
        if int(manifest.get("feature_count", 0)) != len(predictors):
            raise ValueError("gold_feature_count_mismatch")
        contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
        if self._strict_run_scope:
            if not contract_path.is_file():
                raise FileNotFoundError(
                    f"run_contract_fingerprint_missing:{contract_path}"
                )
            contract_sha256 = _sha256(contract_path)
            if str(manifest.get("run_contract_sha256", "")) != contract_sha256:
                raise ValueError("gold_manifest_run_contract_mismatch")
            if str(readiness.get("run_contract_sha256", "")) != contract_sha256:
                raise ValueError("gold_readiness_run_contract_mismatch")
        self._gold_contract = {
            "run_id": manifest_run_id or input_root.name,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "readiness_path": str(readiness_path),
            "readiness_sha256": _sha256(readiness_path),
            "feature_order": list(predictors),
            "features_path": str(source),
            "features_sha256": _sha256(source),
            "run_contract_sha256": (
                _sha256(contract_path) if contract_path.is_file() else None
            ),
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "config_fingerprint": manifest.get("config_fingerprint"),
        }
        if source == input_root / "gold_features_v3.parquet":
            expected_features_sha256 = str(
                manifest.get("artifact_checksums", {}).get(
                    "gold_features_v3.parquet", ""
                )
            )
            if expected_features_sha256 != self._gold_contract["features_sha256"]:
                raise ValueError("gold_feature_checksum_mismatch")
        if source.suffix == ".csv":
            frame = pd.read_csv(source)
        elif participant_ids is None:
            frame = pd.read_parquet(source)
        else:
            frame = pd.read_parquet(
                source,
                filters=[
                    ("sensor_channel", "==", sensor_channel),
                    ("participant_id", "in", list(participant_ids)),
                ],
            )
        if "participant_id" not in frame and "participant_code" in frame:
            frame = frame.rename(columns={"participant_code": "participant_id"})
        required = {
            "participant_id",
            "phase",
            "sensor_channel",
            "label",
            "quality_tier",
            "training_eligible",
            "session_id",
            "recording_id",
            "window_id",
            "window_start_utc",
            "window_end_utc",
            "sampling_rate_hz",
            "source_file",
            "source_checksum",
            "preprocessing_version",
            "feature_version",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Gold table is missing required columns: {missing}")
        phases = set(frame["phase"].astype(str).str.lower())
        if not phases <= {"relax", "task"}:
            raise ValueError(
                f"noncanonical_gold_phase:{sorted(phases - {'relax', 'task'})}"
            )
        eligibility = frame["training_eligible"]
        if not pd.api.types.is_bool_dtype(eligibility):
            normalized = eligibility.astype(str).str.strip().str.lower()
            if not normalized.isin({"true", "false"}).all():
                raise ValueError("invalid_training_eligible_contract")
            eligibility = normalized.eq("true")
        frame = frame[
            (frame["sensor_channel"].astype(str) == sensor_channel)
            & frame["phase"].astype(str).str.lower().isin({"relax", "task"})
            & frame["quality_tier"]
            .astype(str)
            .isin(self.config.training.allowed_quality_tiers)
            & eligibility
        ].copy()
        if participant_ids is not None:
            frame = frame[
                frame["participant_id"].astype(str).isin(participant_ids)
            ].copy()
        frame["participant_id"] = frame["participant_id"].astype(str)
        observed = set(frame["participant_id"])
        expected = set(participant_ids or (*DEVELOPMENT, *HOLDOUT))
        if not observed <= expected:
            raise ValueError(
                "Gold participant contract mismatch: "
                f"unexpected={sorted(observed - expected)}"
            )
        frame["phase"] = frame["phase"].astype(str).str.lower()
        frame["label"] = frame["label"].astype(int)
        expected_features = _configured_predictors(self.config)
        missing_features = sorted(set(expected_features) - set(frame.columns))
        if missing_features:
            raise ValueError(
                f"Gold is missing canonical predictors: {missing_features}"
            )
        observed_groups = {
            (participant, phase, sensor_channel)
            for participant, phase in frame[["participant_id", "phase"]].itertuples(
                index=False, name=None
            )
        }
        if participant_ids == HOLDOUT:
            if not self._has_complete_group_coverage(frame, HOLDOUT):
                raise ValueError("selected_quality_cohort_missing_8_test_groups")
        else:
            coverage = channel_group_readiness(
                observed_groups,
                minimum_training_development_group_coverage=(
                    self.config.training.minimum_training_development_group_coverage
                ),
                minimum_development_groups_per_phase=(
                    self.config.training.minimum_development_groups_per_phase
                ),
                minimum_development_participants=(
                    self.config.training.minimum_development_participants
                ),
            )
            development = coverage["development_coverage"][sensor_channel]
            development_ready = (
                coverage["training_channel_status"][sensor_channel] == "ready"
            )
            if not development_ready:
                raise ValueError(
                    "channel_training_coverage_below_policy:"
                    + json.dumps(development, sort_keys=True)
                )
        return frame

    def _feature_order(self, frame: pd.DataFrame) -> tuple[str, ...]:
        features = _configured_predictors(self.config)
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"canonical_predictors_missing:{missing}")
        non_numeric = [
            name for name in features if not pd.api.types.is_numeric_dtype(frame[name])
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric PPG predictors: {non_numeric}")
        return features

    @staticmethod
    def _session_metrics(
        frame: pd.DataFrame,
        probability: np.ndarray,
        threshold: float = THRESHOLD,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        sessions = aggregate_session_probabilities(frame, probability)
        metrics = _metrics(
            sessions["label"].to_numpy(dtype=int),
            sessions["probability"].to_numpy(dtype=float),
            threshold,
        )
        participant_ids = sessions["participant_id"].drop_duplicates().to_numpy()
        if participant_ids.size >= 2:
            generator = np.random.default_rng(42)
            scores: list[float] = []
            for _ in range(1_000):
                sampled = generator.choice(
                    participant_ids, size=participant_ids.size, replace=True
                )
                bootstrap = pd.concat(
                    [
                        sessions[sessions["participant_id"] == participant]
                        for participant in sampled
                    ],
                    ignore_index=True,
                )
                if bootstrap["label"].nunique() < 2:
                    continue
                prediction = (
                    bootstrap["probability"].to_numpy(dtype=float) >= threshold
                ).astype(int)
                scores.append(
                    float(balanced_accuracy_score(bootstrap["label"], prediction))
                )
            if scores:
                metrics["balanced_accuracy_bootstrap_ci95"] = [
                    float(np.quantile(scores, 0.025)),
                    float(np.quantile(scores, 0.975)),
                ]
        return metrics, sessions

    @staticmethod
    def _has_complete_group_coverage(
        frame: pd.DataFrame,
        participant_ids: tuple[str, ...] = DEVELOPMENT,
    ) -> bool:
        expected = {
            (participant, phase)
            for participant in participant_ids
            for phase in ("relax", "task")
        }
        observed = set(
            frame[["participant_id", "phase"]].itertuples(index=False, name=None)
        )
        return observed == expected

    def _has_development_group_coverage(self, frame: pd.DataFrame) -> bool:
        if frame.empty or frame["sensor_channel"].nunique() != 1:
            return False
        channel = str(frame["sensor_channel"].iloc[0])
        groups = {
            (participant, phase, channel)
            for participant, phase in frame[["participant_id", "phase"]].itertuples(
                index=False, name=None
            )
        }
        readiness = channel_group_readiness(
            groups,
            minimum_training_development_group_coverage=(
                self.config.training.minimum_training_development_group_coverage
            ),
            minimum_development_groups_per_phase=(
                self.config.training.minimum_development_groups_per_phase
            ),
            minimum_development_participants=(
                self.config.training.minimum_development_participants
            ),
        )
        return readiness["training_channel_status"][channel] == "ready"

    def _candidate_models(self) -> dict[str, Pipeline]:
        """Build untuned defaults for discovery and compatibility callers."""

        factory = ClassicalModelFactory(self.config.training.random_seed)
        return {
            name: factory.build(name, factory.defaults(name))
            for name in self.config.training.candidate_order
        }

    def _outer_evaluate(
        self,
        frame: pd.DataFrame,
        feature_order: tuple[str, ...],
        model_name: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        folds: list[dict[str, Any]] = []
        factory = ClassicalModelFactory(self.config.training.random_seed)
        optimizer = self._optimizer()
        model_offset = self.config.training.candidate_order.index(model_name) * 100
        window_seconds = int(
            frame["window_seconds"].iloc[0]
            if "window_seconds" in frame
            else self.config.training.window_baseline_seconds
        )
        stride_seconds = int(
            frame["stride_seconds"].iloc[0]
            if "stride_seconds" in frame
            else self.config.training.window_baseline_stride_seconds
        )
        for fold_index, validation_participants in enumerate(
            outer_participant_folds(DEVELOPMENT)
        ):
            validation_mask = frame["participant_id"].isin(validation_participants)
            train = frame[~validation_mask]
            validation = frame[validation_mask]
            selected_features = correlation_pruned_features(
                train,
                feature_order,
                self.config.training.correlation_pruning_threshold,
                self.config.features.promoted_experimental,
            )
            tuning = optimizer.optimize(
                train,
                selected_features,
                model_name,
                seed_offset=model_offset + fold_index,
                study_context={
                    "sensor_channel": str(frame["sensor_channel"].iloc[0]),
                    "quality_cohort": str(frame.attrs.get("quality_cohort", "unknown")),
                    "window_seconds": window_seconds,
                    "stride_seconds": stride_seconds,
                    "outer_fold": fold_index,
                    "stage": "nested_outer",
                },
                promoted_experimental=self.config.features.promoted_experimental,
            )
            model = factory.build(model_name, tuning.parameters)
            model.fit(
                train.loc[:, selected_features],
                train["label"].to_numpy(),
                classifier__sample_weight=participant_phase_weights(train),
            )
            probability = model.predict_proba(validation.loc[:, selected_features])[
                :, 1
            ]
            fold, session_predictions = self._session_metrics(validation, probability)
            fold["participant_ids"] = list(validation_participants)
            fold["session_count"] = int(len(session_predictions))
            fold["inner_cv_best_balanced_accuracy"] = tuning.balanced_accuracy
            fold["inner_cv_best_parameters"] = tuning.parameters
            fold["inner_group_splits"] = tuning.group_splits
            fold["optuna_trials"] = tuning.trials
            fold["feature_order"] = list(selected_features)
            window_predictions = validation.loc[
                :, ["participant_id", "phase", "sensor_channel", "label"]
            ].copy()
            if "window_id" in validation:
                window_predictions["window_id"] = validation["window_id"].to_numpy()
            else:
                window_predictions["window_id"] = np.arange(len(validation))
            window_predictions["probability"] = probability
            window_predictions["outer_fold"] = fold_index
            session_predictions = session_predictions.copy()
            session_predictions["outer_fold"] = fold_index
            fold["window_predictions"] = window_predictions.to_dict(orient="records")
            fold["session_predictions"] = session_predictions.to_dict(orient="records")
            folds.append(fold)
            labels.append(session_predictions["label"].to_numpy(dtype=int))
            probabilities.append(
                session_predictions["probability"].to_numpy(dtype=float)
            )
        aggregate = _metrics(np.concatenate(labels), np.concatenate(probabilities))
        aggregate["fold_balanced_accuracy_mean"] = float(
            np.mean([fold["balanced_accuracy"] for fold in folds])
        )
        aggregate["fold_balanced_accuracy_std"] = float(
            np.std([fold["balanced_accuracy"] for fold in folds])
        )
        return aggregate, folds

    @staticmethod
    def _nested_outer_oof(
        folds: list[dict[str, Any]],
        *,
        expected_participants: set[str],
    ) -> pd.DataFrame:
        """Return each development window's single nested-outer prediction."""

        rows = [row for fold in folds for row in fold.get("window_predictions", [])]
        frame = pd.DataFrame(rows)
        required = {
            "participant_id",
            "phase",
            "sensor_channel",
            "label",
            "window_id",
            "probability",
            "outer_fold",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("nested_outer_oof_missing_columns:" + ",".join(missing))
        if frame.empty or frame["window_id"].duplicated().any():
            raise ValueError("nested_outer_oof_window_contract")
        if set(frame["participant_id"].astype(str)) != expected_participants:
            raise ValueError("nested_outer_oof_participant_contract")
        probability = pd.to_numeric(frame["probability"], errors="coerce")
        if not np.isfinite(probability).all():
            raise ValueError("nested_outer_oof_nonfinite_probability")
        return frame.sort_values(
            ["participant_id", "phase", "window_id"], kind="mergesort"
        ).reset_index(drop=True)

    def _baseline_evaluate(
        self,
        frame: pd.DataFrame,
        feature_order: tuple[str, ...],
        model_name: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Evaluate fixed defaults before allowing only the strongest two into HPO."""

        factory = ClassicalModelFactory(self.config.training.random_seed)
        session_probabilities: list[np.ndarray] = []
        session_labels: list[np.ndarray] = []
        folds: list[dict[str, Any]] = []
        for validation_participants in outer_participant_folds(DEVELOPMENT):
            validation_mask = frame["participant_id"].isin(validation_participants)
            train, validation = frame[~validation_mask], frame[validation_mask]
            selected_features = correlation_pruned_features(
                train,
                feature_order,
                self.config.training.correlation_pruning_threshold,
                self.config.features.promoted_experimental,
            )
            model = factory.build(model_name, factory.defaults(model_name))
            model.fit(
                train.loc[:, selected_features],
                train["label"].to_numpy(),
                classifier__sample_weight=participant_phase_weights(train),
            )
            probability = model.predict_proba(validation.loc[:, selected_features])[
                :, 1
            ]
            metrics, sessions = self._session_metrics(validation, probability)
            metrics["participant_ids"] = list(validation_participants)
            metrics["session_count"] = int(len(sessions))
            metrics["feature_order"] = list(selected_features)
            folds.append(metrics)
            session_probabilities.append(sessions["probability"].to_numpy(dtype=float))
            session_labels.append(sessions["label"].to_numpy(dtype=int))
        aggregate = _metrics(
            np.concatenate(session_labels), np.concatenate(session_probabilities)
        )
        aggregate["fold_balanced_accuracy_mean"] = float(
            np.mean([fold["balanced_accuracy"] for fold in folds])
        )
        aggregate["fold_balanced_accuracy_std"] = float(
            np.std([fold["balanced_accuracy"] for fold in folds])
        )
        return aggregate, folds

    def _spectral_feature_ablation(
        self,
        development: pd.DataFrame,
        default_feature_order: tuple[str, ...],
        default_results: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        missing = sorted(set(SPECTRAL_ABLATION_FEATURES) - set(development.columns))
        if missing:
            return {
                "status": "UNAVAILABLE",
                "selected": False,
                "features": list(SPECTRAL_ABLATION_FEATURES),
                "reason": "spectral_features_missing:" + ",".join(missing),
            }
        finite = np.isfinite(
            development.loc[:, SPECTRAL_ABLATION_FEATURES].to_numpy(dtype=float)
        ).all(axis=1)
        candidate = development.loc[finite].copy()
        if len(candidate) != len(
            development
        ) or not self._has_development_group_coverage(candidate):
            return {
                "status": "NOT_PROMOTED",
                "selected": False,
                "features": list(SPECTRAL_ABLATION_FEATURES),
                "reason": "spectral_features_reduce_development_coverage",
                "default_row_count": len(development),
                "candidate_row_count": len(candidate),
            }
        candidate_order = (*default_feature_order, *SPECTRAL_ABLATION_FEATURES)
        candidate_results: dict[str, Any] = {}
        for model_name in self._candidate_models():
            metrics, folds = self._baseline_evaluate(
                candidate, candidate_order, model_name
            )
            candidate_results[model_name] = {"metrics": metrics, "folds": folds}
        default_name = rank_model_candidates(dict(default_results))[0]
        candidate_name = rank_model_candidates(candidate_results)[0]
        default_metrics = default_results[default_name]["metrics"]
        candidate_metrics = candidate_results[candidate_name]["metrics"]
        eligible = spectral_ablation_is_eligible(
            default_metrics,
            candidate_metrics,
            default_row_count=len(development),
            candidate_row_count=len(candidate),
        )
        return {
            "status": "PROMOTION_ELIGIBLE" if eligible else "NOT_PROMOTED",
            "selected": False,
            "features": list(SPECTRAL_ABLATION_FEATURES),
            "default_best_model": default_name,
            "candidate_best_model": candidate_name,
            "default_metrics": default_metrics,
            "candidate_metrics": candidate_metrics,
            "balanced_accuracy_gain": float(
                candidate_metrics["balanced_accuracy"]
                - default_metrics["balanced_accuracy"]
            ),
            "coverage_unchanged": True,
            "participant_stability_not_worse": bool(
                float(candidate_metrics["fold_balanced_accuracy_std"])
                <= float(default_metrics["fold_balanced_accuracy_std"])
            ),
            "candidate_results": candidate_results,
        }

    def run(self, sensor_channel: str) -> dict[str, Any]:
        if sensor_channel not in self.config.training.training_channels:
            raise ValueError(
                f"PPG sensor is not in the training allowlist: {sensor_channel}"
            )
        try:
            gold_input_validation = validate_gold_input(
                self.root,
                self.config,
                self.paths,
                sensor_channel=sensor_channel,
            )
        except (FileNotFoundError, ValueError) as exc:
            return self._blocked_coverage_report(sensor_channel, str(exc))
        if gold_input_validation.get("status") != "ready":
            return self._blocked_coverage_report(
                sensor_channel,
                str(
                    gold_input_validation.get(
                        "blocking_reason", "channel_gold_input_not_ready"
                    )
                ),
                gold_input_validation=gold_input_validation,
            )
        try:
            frame = self._load_gold(sensor_channel, participant_ids=DEVELOPMENT)
        except (FileNotFoundError, ValueError) as exc:
            return self._blocked_coverage_report(sensor_channel, str(exc))
        feature_order = self._feature_order(frame)
        finite_mask = np.isfinite(
            frame.loc[:, feature_order].to_numpy(dtype=np.float64)
        ).all(axis=1)
        excluded_nonfinite_window_count = int((~finite_mask).sum())
        frame = frame.loc[finite_mask].copy()
        if not self._has_development_group_coverage(
            frame[frame["participant_id"].isin(DEVELOPMENT)]
        ):
            report = self._blocked_coverage_report(
                sensor_channel,
                "non_finite_window_exclusion_removed_development_group",
                feature_order=feature_order,
                extra={
                    "excluded_nonfinite_window_count": excluded_nonfinite_window_count
                },
            )
            return report
        cohort_frames = {
            "strict": frame[
                frame["quality_tier"].isin(self.config.training.strict_quality_tiers)
            ].copy(),
            "expanded": frame[
                frame["quality_tier"].isin(self.config.training.expanded_quality_tiers)
            ].copy(),
        }
        quality_cohort_results: dict[str, Any] = {}
        for cohort_name, cohort_frame in cohort_frames.items():
            if (
                cohort_name == "expanded"
                and cohort_frame.index.equals(cohort_frames["strict"].index)
                and "strict" in quality_cohort_results
            ):
                quality_cohort_results[cohort_name] = {
                    **quality_cohort_results["strict"],
                    "reused_from": "strict",
                }
                continue
            cohort_development = cohort_frame[
                cohort_frame["participant_id"].isin(DEVELOPMENT)
            ].copy()
            if not self._has_development_group_coverage(cohort_development):
                quality_cohort_results[cohort_name] = {
                    "available": False,
                    "reason": "development_group_coverage_below_policy",
                    "row_count": int(len(cohort_frame)),
                }
                continue
            cohort_baselines: dict[str, Any] = {}
            for name in self._candidate_models():
                metrics, folds = self._baseline_evaluate(
                    cohort_development, feature_order, name
                )
                cohort_baselines[name] = {"metrics": metrics, "folds": folds}
            best_baseline = rank_model_candidates(cohort_baselines)[0]
            quality_cohort_results[cohort_name] = {
                "available": True,
                "row_count": int(len(cohort_frame)),
                "balanced_accuracy": cohort_baselines[best_baseline]["metrics"][
                    "balanced_accuracy"
                ],
                "best_fixed_baseline": best_baseline,
                "fixed_baseline_results": cohort_baselines,
            }
        try:
            selected_quality_cohort = select_quality_cohort(
                quality_cohort_results,
                tie_tolerance=self.config.training.quality_cohort_tie_tolerance,
            )
        except ValueError as exc:
            report = {
                "schema_version": "1.1.0",
                "model_version": self.config.model_version,
                "sensor_channel": sensor_channel,
                "model_family": "classical",
                "feature_order": list(feature_order),
                "predictor_columns": list(feature_order),
                "development_participants": list(DEVELOPMENT),
                "holdout_participants": list(HOLDOUT),
                "holdout_evaluated": False,
                "channel_status": "not_evaluated",
                "requires_dl": False,
                "release_status": "blocked_quality_cohort",
                "blocking_reason": str(exc),
                "quality_cohort_ablation": quality_cohort_results,
                "excluded_nonfinite_window_count": excluded_nonfinite_window_count,
            }
            self._write_report(sensor_channel, report)
            return report
        frame = cohort_frames[selected_quality_cohort]
        development = frame[frame["participant_id"].isin(DEVELOPMENT)].copy()
        development.attrs["quality_cohort"] = selected_quality_cohort
        baseline_results = quality_cohort_results[selected_quality_cohort][
            "fixed_baseline_results"
        ]
        selected_window = (
            self.config.training.window_baseline_seconds,
            self.config.training.window_baseline_stride_seconds,
        )
        selected_fixed_name = rank_model_candidates(baseline_results)[0]
        selected_fixed_metrics = baseline_results[selected_fixed_name]["metrics"]
        window_ablation: list[dict[str, Any]] = [
            {
                "window_seconds": int(selected_window[0]),
                "stride_seconds": int(selected_window[1]),
                "status": "evaluated",
                "selected": True,
                "quality_cohort": selected_quality_cohort,
                "best_fixed_baseline": selected_fixed_name,
                **{
                    key: selected_fixed_metrics.get(key)
                    for key in (
                        "balanced_accuracy",
                        "f1",
                        "roc_auc",
                        "fold_balanced_accuracy_std",
                    )
                },
                "development_group_count": int(
                    development[["participant_id", "phase"]].drop_duplicates().shape[0]
                ),
            }
        ]
        selected_variant = {
            "frame": frame,
            "development": development,
            "source_override": None,
            "gold_contract": dict(self._gold_contract),
            "quality_results": quality_cohort_results,
            "quality_cohort": selected_quality_cohort,
            "baseline_results": baseline_results,
            "excluded_nonfinite_window_count": excluded_nonfinite_window_count,
        }
        gold_input_root = _configured_path(
            self.root, self.config, "gold_input_root", self.paths.gold_dir
        )
        for window_seconds, stride_seconds in self.config.training.window_candidates:
            candidate_window = (int(window_seconds), int(stride_seconds))
            if candidate_window == selected_window:
                continue
            variant_key = f"w{candidate_window[0]:03d}_s{candidate_window[1]:03d}"
            source = (
                gold_input_root
                / "training_variants"
                / variant_key
                / "expanded"
                / "gold_features.parquet"
            )
            try:
                variant_frame = self._load_gold(
                    sensor_channel,
                    source,
                    participant_ids=DEVELOPMENT,
                )
                variant_feature_order = self._feature_order(variant_frame)
                finite = np.isfinite(
                    variant_frame.loc[:, variant_feature_order].to_numpy(
                        dtype=np.float64
                    )
                ).all(axis=1)
                variant_nonfinite = int((~finite).sum())
                variant_frame = variant_frame.loc[finite].copy()
                variant_development_all = variant_frame[
                    variant_frame["participant_id"].isin(DEVELOPMENT)
                ]
                if not self._has_development_group_coverage(variant_development_all):
                    raise ValueError("development_group_coverage_below_policy")
                variant_cohort_frames = {
                    "strict": variant_frame[
                        variant_frame["quality_tier"].isin(
                            self.config.training.strict_quality_tiers
                        )
                    ].copy(),
                    "expanded": variant_frame[
                        variant_frame["quality_tier"].isin(
                            self.config.training.expanded_quality_tiers
                        )
                    ].copy(),
                }
                variant_quality_results: dict[str, Any] = {}
                for cohort_name, cohort_frame in variant_cohort_frames.items():
                    if (
                        cohort_name == "expanded"
                        and cohort_frame.index.equals(
                            variant_cohort_frames["strict"].index
                        )
                        and "strict" in variant_quality_results
                    ):
                        variant_quality_results[cohort_name] = {
                            **variant_quality_results["strict"],
                            "reused_from": "strict",
                        }
                        continue
                    cohort_development = cohort_frame[
                        cohort_frame["participant_id"].isin(DEVELOPMENT)
                    ].copy()
                    if not self._has_development_group_coverage(cohort_development):
                        variant_quality_results[cohort_name] = {
                            "available": False,
                            "reason": "development_group_coverage_below_policy",
                            "row_count": int(len(cohort_frame)),
                        }
                        continue
                    fixed_results: dict[str, Any] = {}
                    for name in self._candidate_models():
                        metrics, folds = self._baseline_evaluate(
                            cohort_development, variant_feature_order, name
                        )
                        fixed_results[name] = {"metrics": metrics, "folds": folds}
                    best_name = rank_model_candidates(fixed_results)[0]
                    variant_quality_results[cohort_name] = {
                        "available": True,
                        "row_count": int(len(cohort_frame)),
                        "balanced_accuracy": fixed_results[best_name]["metrics"][
                            "balanced_accuracy"
                        ],
                        "best_fixed_baseline": best_name,
                        "fixed_baseline_results": fixed_results,
                    }
                variant_quality = select_quality_cohort(
                    variant_quality_results,
                    tie_tolerance=self.config.training.quality_cohort_tie_tolerance,
                )
                variant_selected_frame = variant_cohort_frames[variant_quality]
                variant_development = variant_selected_frame[
                    variant_selected_frame["participant_id"].isin(DEVELOPMENT)
                ].copy()
                variant_development.attrs["quality_cohort"] = variant_quality
                fixed_results = variant_quality_results[variant_quality][
                    "fixed_baseline_results"
                ]
                best_name = rank_model_candidates(fixed_results)[0]
                metrics = fixed_results[best_name]["metrics"]
                candidate = {
                    "window_seconds": candidate_window[0],
                    "stride_seconds": candidate_window[1],
                    "status": "evaluated",
                    "selected": False,
                    "quality_cohort": variant_quality,
                    "best_fixed_baseline": best_name,
                    **{
                        key: metrics.get(key)
                        for key in (
                            "balanced_accuracy",
                            "f1",
                            "roc_auc",
                            "fold_balanced_accuracy_std",
                        )
                    },
                    "development_group_count": int(
                        variant_development[["participant_id", "phase"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                }
                window_ablation.append(candidate)
                improves = (
                    float(metrics["balanced_accuracy"])
                    >= float(selected_fixed_metrics["balanced_accuracy"])
                    + self.config.training.window_minimum_ba_gain
                )
                stable = float(metrics["fold_balanced_accuracy_std"]) <= float(
                    selected_fixed_metrics["fold_balanced_accuracy_std"]
                )
                if improves and stable and _passes(metrics, self.config):
                    for row in window_ablation:
                        row["selected"] = False
                    window_ablation[-1]["selected"] = True
                    selected_window = candidate_window
                    selected_fixed_metrics = metrics
                    selected_variant = {
                        "frame": variant_selected_frame,
                        "development": variant_development,
                        "source_override": source,
                        "gold_contract": dict(self._gold_contract),
                        "quality_results": variant_quality_results,
                        "quality_cohort": variant_quality,
                        "baseline_results": fixed_results,
                        "excluded_nonfinite_window_count": variant_nonfinite,
                    }
            except (FileNotFoundError, ValueError) as exc:
                window_ablation.append(
                    {
                        "window_seconds": candidate_window[0],
                        "stride_seconds": candidate_window[1],
                        "status": "not_evaluated",
                        "selected": False,
                        "reason": str(exc),
                    }
                )

        development = selected_variant["development"]
        selected_source_override = selected_variant["source_override"]
        selected_gold_contract = selected_variant["gold_contract"]
        quality_cohort_results = selected_variant["quality_results"]
        selected_quality_cohort = selected_variant["quality_cohort"]
        baseline_results = selected_variant["baseline_results"]
        excluded_nonfinite_window_count = selected_variant[
            "excluded_nonfinite_window_count"
        ]
        feature_group_ablation = self._spectral_feature_ablation(
            development,
            feature_order,
            baseline_results,
        )
        configured_promotions = tuple(self.config.features.promoted_experimental)
        if configured_promotions:
            if configured_promotions != SPECTRAL_ABLATION_FEATURES:
                raise ValueError(
                    "unsupported_promoted_experimental_contract:"
                    + ",".join(configured_promotions)
                )
            if feature_group_ablation["status"] != "PROMOTION_ELIGIBLE":
                raise ValueError("spectral_feature_promotion_not_supported_by_ablation")
            feature_order = (*feature_order, *SPECTRAL_ABLATION_FEATURES)
            baseline_results = feature_group_ablation["candidate_results"]
            feature_group_ablation["status"] = "PROMOTED"
            feature_group_ablation["selected"] = True
        top_candidates = rank_model_candidates(baseline_results)[
            : self.config.training.hpo_top_candidates
        ]
        model_results: dict[str, Any] = {}
        for name in top_candidates:
            metrics, folds = self._outer_evaluate(development, feature_order, name)
            model_results[name] = {
                "metrics": metrics,
                "folds": folds,
                "development_passed": _passes(metrics, self.config),
            }
            logger.info("%s/%s development metrics: %s", sensor_channel, name, metrics)

        passing = [
            name
            for name, result in model_results.items()
            if result["development_passed"]
        ]
        eligible_results = {
            name: model_results[name] for name in passing or model_results.keys()
        }
        selected = rank_model_candidates(eligible_results)[0]
        selected_dev = model_results[selected]
        factory = ClassicalModelFactory(self.config.training.random_seed)
        optimizer = self._optimizer()
        feature_order = correlation_pruned_features(
            development,
            feature_order,
            self.config.training.correlation_pruning_threshold,
            self.config.features.promoted_experimental,
        )
        final_tuning = optimizer.optimize(
            development,
            feature_order,
            selected,
            seed_offset=1_000 + self.config.training.candidate_order.index(selected),
            study_context={
                "sensor_channel": sensor_channel,
                "quality_cohort": selected_quality_cohort,
                "window_seconds": selected_window[0],
                "stride_seconds": selected_window[1],
                "stage": "final_development",
            },
            promoted_experimental=self.config.features.promoted_experimental,
        )
        data_fingerprint = _fingerprint(development, feature_order)
        quality_cohort_summary = {
            name: {
                key: value
                for key, value in result.items()
                if key != "fixed_baseline_results"
            }
            for name, result in quality_cohort_results.items()
        }
        report: dict[str, Any] = {
            "schema_version": "1.0.0",
            "model_version": self.config.model_version,
            "sensor_channel": sensor_channel,
            "model_family": "classical",
            "model_name": selected,
            "selected_parameters": final_tuning.parameters,
            "final_inner_cv_balanced_accuracy": final_tuning.balanced_accuracy,
            "final_inner_group_splits": final_tuning.group_splits,
            "final_optuna_trials": final_tuning.trials,
            "feature_order": list(feature_order),
            "predictor_columns": list(feature_order),
            "canonical_predictors": list(_configured_predictors(self.config)),
            "model_input_feature_order": list(feature_order),
            "development_participants": list(DEVELOPMENT),
            "expected_development_participants": list(DEVELOPMENT),
            "holdout_participants": list(HOLDOUT),
            "development_metrics": selected_dev["metrics"],
            "development_folds": selected_dev["folds"],
            "fixed_baseline_results": baseline_results,
            "hpo_candidates": top_candidates,
            "candidate_results": model_results,
            "selected_quality_cohort": selected_quality_cohort,
            "quality_cohort_ablation": quality_cohort_summary,
            "window_ablation": window_ablation,
            "selected_window_seconds": selected_window[0],
            "selected_stride_seconds": selected_window[1],
            "gold_contract": selected_gold_contract,
            "feature_group_ablation": {
                "default": {
                    "status": ("reference" if configured_promotions else "selected"),
                    "features": list(_configured_predictors(self.config)),
                },
                "spectral": {
                    key: value
                    for key, value in feature_group_ablation.items()
                    if key != "candidate_results"
                },
            },
            "performance_policy": self.performance_policy.as_dict(),
            "performance_policy_version": self.performance_policy.version,
            "release_policy_fingerprint": self._release_policy_fingerprint(),
            "development_gates": self.performance_policy.minimum_research.as_dict(),
            "preferred_gates": self.performance_policy.preferred.as_dict(),
            "threshold": THRESHOLD,
            "threshold_source": "participant_grouped_development_validation",
            "data_fingerprint": data_fingerprint,
            "channel_data_fingerprint": data_fingerprint,
            "fingerprint_scope": "development_rows_and_predictors",
            "holdout_contract_fingerprint": None,
            "holdout_evaluated": False,
            "test_cohort_status": self.config.training.reused_test_cohort_status,
            "clinical_validation_status": self.config.training.clinical_validation_status,
            "requires_dl": False,
            "trigger_reason": None,
            "release_status": "blocked_development_gate",
            "channel_status": "development_failed",
            "excluded_nonfinite_window_count": excluded_nonfinite_window_count,
        }
        raw_development_assessment = self.performance_policy.assess(
            selected_dev["metrics"]
        )
        report["raw_development_assessment"] = raw_development_assessment.as_dict()
        report["performance_tier"] = raw_development_assessment.performance_tier.value
        report["deployment_scope"] = raw_development_assessment.deployment_scope.value
        report["research_disclaimer"] = (
            self.performance_policy.research_disclaimer
            if raw_development_assessment.disclaimer_required
            else None
        )
        report["development_metrics_source"] = "uncalibrated_nested_outer_oof"

        model = factory.build(selected, final_tuning.parameters)
        model.fit(
            development.loc[:, feature_order],
            development["label"].to_numpy(),
            classifier__sample_weight=participant_phase_weights(development),
        )
        calibration_feature_order = self._feature_order(development)
        represented_development = set(development["participant_id"].astype(str))
        nested_oof = self._nested_outer_oof(
            selected_dev["folds"],
            expected_participants=represented_development,
        )
        report["represented_development_participants"] = sorted(represented_development)
        report["missing_development_participants"] = sorted(
            set(DEVELOPMENT) - represented_development
        )
        development_oof_probability = nested_oof["probability"].to_numpy(dtype=float)
        calibration = fit_platt_from_nested_oof(
            labels=nested_oof["label"].to_numpy(),
            probabilities=development_oof_probability,
            groups=nested_oof["participant_id"].to_numpy(),
            phase_groups=(
                nested_oof["participant_id"].astype(str)
                + "|"
                + nested_oof["phase"].astype(str)
            ).to_numpy(),
            fitted_cohort=sorted(represented_development),
        )
        calibrated_development_oof = calibration.apply(development_oof_probability)
        selected_threshold = select_session_threshold(
            nested_oof, calibrated_development_oof
        )
        report["threshold"] = selected_threshold
        report["calibration_input_feature_order"] = list(calibration_feature_order)
        report["threshold_source"] = "nested_outer_participant_grouped_oof"
        calibrated_development_metrics, _ = self._session_metrics(
            nested_oof, calibrated_development_oof, selected_threshold
        )
        report["development_raw_metrics"] = report["development_metrics"]
        report["development_metrics"] = calibrated_development_metrics
        report["development_metrics_source"] = "calibrated_nested_outer_oof"
        development_assessment = self.performance_policy.assess(
            calibrated_development_metrics
        )
        report["development_assessment"] = development_assessment.as_dict()
        report["performance_tier"] = development_assessment.performance_tier.value
        report["deployment_scope"] = development_assessment.deployment_scope.value
        report["research_disclaimer"] = (
            self.performance_policy.research_disclaimer
            if development_assessment.disclaimer_required
            else None
        )
        if not development_assessment.passes_minimum:
            report["requires_dl"] = True
            report["trigger_reason"] = (
                "local_ml_development_metrics_below_minimum_research_gate"
            )
            report["failing_minimum_metrics"] = list(
                development_assessment.failing_minimum_metrics
            )
            report["probability_calibration"] = calibration.to_dict()
            report["development_oof_calibration_metrics"] = calibration.to_dict()
            report["development_oof_probability_count"] = int(
                development_oof_probability.size
            )
            report["release_status"] = "blocked_development_gate"
            report["channel_status"] = "development_failed"
            self._write_report(sensor_channel, report)
            return report

        report["channel_status"] = "development_passed"
        from .onnx_v3 import OnnxBundleExporter

        try:
            converter_preflight = OnnxBundleExporter(
                self.root,
                self.config,
            ).preflight_estimator(
                estimator=model,
                feature_order=feature_order,
                development_tensor=development.loc[:, feature_order]
                .head(16)
                .to_numpy(),
                sensor_channel=sensor_channel,
            )
        except (RuntimeError, ValueError) as exc:
            preflight_failures = {
                selected: f"{type(exc).__name__}:{exc}",
            }
            frozen_ranking = rank_model_candidates(
                {name: model_results[name] for name in passing}
            )
            fallback_selected = False
            for alternate in frozen_ranking:
                if alternate == selected:
                    continue
                alternate_tuning = optimizer.optimize(
                    development,
                    feature_order,
                    alternate,
                    seed_offset=1_000
                    + self.config.training.candidate_order.index(alternate),
                    study_context={
                        "sensor_channel": sensor_channel,
                        "quality_cohort": selected_quality_cohort,
                        "window_seconds": selected_window[0],
                        "stride_seconds": selected_window[1],
                        "stage": "final_development_export_fallback",
                    },
                    promoted_experimental=self.config.features.promoted_experimental,
                )
                alternate_model = factory.build(alternate, alternate_tuning.parameters)
                alternate_model.fit(
                    development.loc[:, feature_order],
                    development["label"].to_numpy(),
                    classifier__sample_weight=participant_phase_weights(development),
                )
                alternate_nested_oof = self._nested_outer_oof(
                    model_results[alternate]["folds"],
                    expected_participants=represented_development,
                )
                alternate_oof_probability = alternate_nested_oof[
                    "probability"
                ].to_numpy(dtype=float)
                alternate_calibration = fit_platt_from_nested_oof(
                    labels=alternate_nested_oof["label"].to_numpy(),
                    probabilities=alternate_oof_probability,
                    groups=alternate_nested_oof["participant_id"].to_numpy(),
                    phase_groups=(
                        alternate_nested_oof["participant_id"].astype(str)
                        + "|"
                        + alternate_nested_oof["phase"].astype(str)
                    ).to_numpy(),
                    fitted_cohort=sorted(represented_development),
                )
                alternate_threshold = select_session_threshold(
                    alternate_nested_oof,
                    alternate_calibration.apply(alternate_oof_probability),
                )
                alternate_calibrated_oof = alternate_calibration.apply(
                    alternate_oof_probability
                )
                alternate_development_metrics, _ = self._session_metrics(
                    alternate_nested_oof,
                    alternate_calibrated_oof,
                    alternate_threshold,
                )
                if not self.performance_policy.assess(
                    alternate_development_metrics
                ).passes_minimum:
                    preflight_failures[alternate] = (
                        "development_gate_failed_after_calibration"
                    )
                    continue
                try:
                    alternate_preflight = OnnxBundleExporter(
                        self.root,
                        self.config,
                    ).preflight_estimator(
                        estimator=alternate_model,
                        feature_order=feature_order,
                        development_tensor=development.loc[:, feature_order]
                        .head(16)
                        .to_numpy(),
                        sensor_channel=sensor_channel,
                    )
                except (RuntimeError, ValueError) as fallback_exc:
                    preflight_failures[alternate] = (
                        f"{type(fallback_exc).__name__}:{fallback_exc}"
                    )
                    continue
                report["converter_fallback_from"] = selected
                selected = alternate
                selected_dev = model_results[selected]
                final_tuning = alternate_tuning
                model = alternate_model
                calibration = alternate_calibration
                development_oof_probability = alternate_oof_probability
                calibrated_development_oof = alternate_calibrated_oof
                calibrated_development_metrics = alternate_development_metrics
                selected_threshold = alternate_threshold
                converter_preflight = alternate_preflight
                report.update(
                    {
                        "model_name": selected,
                        "selected_parameters": final_tuning.parameters,
                        "final_inner_cv_balanced_accuracy": final_tuning.balanced_accuracy,
                        "final_inner_group_splits": final_tuning.group_splits,
                        "final_optuna_trials": final_tuning.trials,
                        "development_metrics": calibrated_development_metrics,
                        "development_raw_metrics": selected_dev["metrics"],
                        "development_folds": selected_dev["folds"],
                        "threshold": selected_threshold,
                    }
                )
                fallback_selected = True
                break
            if not fallback_selected:
                report["converter_preflight"] = {
                    "status": "failed",
                    "failures": preflight_failures,
                }
                report["probability_calibration"] = calibration.to_dict()
                report["development_oof_calibration_metrics"] = calibration.to_dict()
                report["development_oof_probability_count"] = int(
                    development_oof_probability.size
                )
                report["release_status"] = "blocked_unsupported_converter"
                report["channel_status"] = "development_failed"
                report["requires_dl"] = False
                report["trigger_reason"] = "no_exportable_development_candidate"
                self._write_report(sensor_channel, report)
                return report
            report["converter_preflight_failures"] = preflight_failures
        report["converter_preflight"] = converter_preflight
        holdout_contract_fingerprint = self._holdout_contract_fingerprint(
            sensor_channel=sensor_channel,
            selected_model=selected,
            selected_parameters=final_tuning.parameters,
            feature_order=feature_order,
            data_fingerprint=data_fingerprint,
            decision_threshold=selected_threshold,
        )
        legacy_holdout_contract_fingerprint = self._legacy_holdout_contract_fingerprint(
            sensor_channel=sensor_channel,
            selected_model=selected,
            selected_parameters=final_tuning.parameters,
            feature_order=feature_order,
            data_fingerprint=data_fingerprint,
            decision_threshold=selected_threshold,
        )
        report["model_contract_fingerprint"] = holdout_contract_fingerprint
        report["holdout_contract_fingerprint"] = holdout_contract_fingerprint
        report["legacy_holdout_contract_fingerprint"] = (
            legacy_holdout_contract_fingerprint
        )
        report["probability_calibration"] = calibration.to_dict()
        report["development_oof_calibration_metrics"] = calibration.to_dict()
        report["development_oof_probability_count"] = int(
            development_oof_probability.size
        )
        report["converter_preflight"] = converter_preflight
        self._freeze_development_candidate(
            sensor_channel,
            model,
            report,
            selected_gold_contract,
        )
        if self.defer_reused_internal_test:
            # `publish-internal` owns the only ID-29..ID-32 access.  Training
            # must return immediately after freezing the full candidate so the
            # later command can bind one immutable internal-test contract.
            report["holdout_evaluated"] = False
            report["holdout_evidence_reused"] = False
            report["holdout_evidence_classification"] = "deferred_publish_internal"
            report["internal_test_status"] = "PENDING"
            report["release_status"] = "development_candidate"
            report["channel_status"] = "development_candidate"
            report["deployment_scope"] = "none"
            report["evaluation_provenance"] = {
                "candidate_frozen_before_internal_test": True,
                "threshold_frozen_before_internal_test": True,
                "internal_test_access_count": 0,
            }
            self._write_report(sensor_channel, report)
            return report
        historical_unverifiable_reports = self._historical_unverifiable_holdout_reports(
            sensor_channel
        )
        report["historical_unverifiable_holdout_reports"] = list(
            historical_unverifiable_reports
        )
        report["historical_unverifiable_holdout_report_count"] = len(
            historical_unverifiable_reports
        )
        prior_evidence = self._find_prior_holdout_evidence(
            sensor_channel,
            holdout_contract_fingerprint,
            legacy_holdout_contract_fingerprint,
        )
        if prior_evidence is None and self._has_any_prior_holdout_evidence(
            sensor_channel
        ):
            report["holdout_evaluated"] = False
            report["holdout_evidence_reused"] = False
            report["release_status"] = "blocked_existing_holdout_contract_mismatch"
            report["channel_status"] = "internal_test_failed"
            report["holdout_evidence_classification"] = (
                "blocked_mismatched_fingerprinted"
            )
            report["blocking_reason"] = (
                "existing_reused_test_evidence_does_not_match_frozen_model_contract"
            )
            self._write_report(sensor_channel, report)
            return report
        if prior_evidence is None:
            self.holdout_evidence.claim(
                sensor_channel,
                holdout_contract_fingerprint,
                {
                    "run_id": str(self.config.artifacts.get("run_id", "")),
                    "model_version": self.config.model_version,
                    "status": "claimed_before_read",
                },
            )
            holdout = self._load_gold(
                sensor_channel,
                selected_source_override,
                participant_ids=HOLDOUT,
            )
            selected_tiers = (
                self.config.training.strict_quality_tiers
                if selected_quality_cohort == "strict"
                else self.config.training.expanded_quality_tiers
            )
            holdout = holdout[holdout["quality_tier"].isin(selected_tiers)].copy()
            if not self._has_complete_group_coverage(holdout, HOLDOUT):
                report["channel_status"] = "internal_test_failed"
                report["release_status"] = "blocked_incomplete_reused_test_cohort"
                report["blocking_reason"] = (
                    "selected_quality_cohort_missing_8_test_groups"
                )
                self._write_report(sensor_channel, report)
                return report
            holdout_raw_probability = model.predict_proba(
                holdout.loc[:, feature_order]
            )[:, 1]
            holdout_probability = calibration.apply(holdout_raw_probability)
            holdout_metrics, holdout_sessions = self._session_metrics(
                holdout, holdout_probability, selected_threshold
            )
            report["holdout_evaluated"] = True
            report["holdout_evidence_reused"] = False
            report["holdout_evidence_classification"] = "current_canonical_once"
            report["holdout_metrics"] = holdout_metrics
            report["holdout_raw_metrics"], _ = self._session_metrics(
                holdout, holdout_raw_probability, selected_threshold
            )
            report["internal_test_session_count"] = int(len(holdout_sessions))
            holdout_assessment = self.performance_policy.assess(holdout_metrics)
            report["release_status"] = (
                self._release_candidate_status()
                if holdout_assessment.passes_minimum
                else "blocked_holdout_gate"
            )
            self._record_holdout_access(
                sensor_channel,
                holdout_contract_fingerprint,
                report,
                historical_unverifiable_reports,
            )
        else:
            holdout_metrics = dict(prior_evidence["holdout_metrics"])
            report["holdout_evaluated"] = False
            report["holdout_evidence_reused"] = True
            report["holdout_evidence_classification"] = "reused_fingerprinted"
            report["holdout_evidence_model_version"] = prior_evidence.get(
                "model_version"
            )
            report["holdout_metrics"] = holdout_metrics
            report["holdout_raw_metrics"] = prior_evidence.get(
                "holdout_raw_metrics", {}
            )
        holdout_assessment = self.performance_policy.assess(holdout_metrics)
        conservative_evidence_metrics = _conservative_gate_metrics(
            calibrated_development_metrics, holdout_metrics
        )
        final_assessment = self.performance_policy.assess(conservative_evidence_metrics)
        report["combined_evidence_metrics"] = conservative_evidence_metrics
        report["final_assessment"] = final_assessment.as_dict()
        report["holdout_assessment"] = holdout_assessment.as_dict()
        report["performance_tier"] = final_assessment.performance_tier.value
        report["deployment_scope"] = final_assessment.deployment_scope.value
        report["research_disclaimer"] = (
            self.performance_policy.research_disclaimer
            if final_assessment.disclaimer_required
            else None
        )
        report["probability_calibration"] = calibration.to_dict()
        report["development_oof_calibration_metrics"] = calibration.to_dict()
        report["development_oof_probability_count"] = int(
            development_oof_probability.size
        )
        report["release_status"] = (
            self._release_candidate_status()
            if final_assessment.passes_minimum
            else "blocked_holdout_gate"
        )
        report["channel_status"] = (
            "model_candidate"
            if report["release_status"] == self._release_candidate_status()
            else "internal_test_failed"
        )
        report["evaluation_provenance"] = {
            "candidate_frozen_before_holdout": True,
            "threshold_frozen_before_holdout": True,
            "holdout_access_count": 0 if prior_evidence is not None else 1,
            "holdout_evidence_reused": prior_evidence is not None,
        }
        if report["release_status"] == self._release_candidate_status():
            bundle = (
                self.paths.models_dir
                / "stress"
                / sensor_channel
                / self.config.model_version
            )
            bundle.mkdir(parents=True, exist_ok=True)
            model_path = bundle / "model.joblib"
            _dump_joblib_immutable(model, model_path)
            run_contract_path = (
                self.paths.artifacts_root / "run_contract_fingerprint.json"
            )
            run_contract_sha256 = (
                _sha256(run_contract_path) if run_contract_path.is_file() else None
            )
            metadata = {
                "model_version": self.config.model_version,
                "sensor_channel": sensor_channel,
                "run_id": str(getattr(self.config, "artifacts", {}).get("run_id", "")),
                "release_status": report.get("release_status"),
                "model_family": "classical",
                "model_name": selected,
                "selected_parameters": final_tuning.parameters,
                "feature_order": list(feature_order),
                "canonical_predictors": list(_configured_predictors(self.config)),
                "model_input_feature_order": list(feature_order),
                "calibration_input_feature_order": list(calibration_feature_order),
                "feature_version": self.config.feature_version,
                "preprocessing_version": self.config.preprocessing_version,
                "native_model_path": str(model_path.relative_to(self.root)).replace(
                    "\\", "/"
                ),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "analysis_seconds": int(
                    self.config.window.analysis_phase_duration_seconds
                ),
                "window_seconds": int(selected_window[0]),
                "stride_seconds": int(selected_window[1]),
                "expected_windows_per_session": int(
                    (
                        self.config.window.analysis_phase_duration_seconds
                        - selected_window[0]
                    )
                    // selected_window[1]
                    + 1
                ),
                "development_participants": list(DEVELOPMENT),
                "expected_development_participants": list(DEVELOPMENT),
                "represented_development_participants": sorted(represented_development),
                "missing_development_participants": sorted(
                    set(DEVELOPMENT) - represented_development
                ),
                "holdout_participants": list(HOLDOUT),
                "probability_threshold": selected_threshold,
                "development_metrics": calibrated_development_metrics,
                "development_raw_metrics": report.get("development_raw_metrics"),
                "holdout_metrics": holdout_metrics,
                "holdout_raw_metrics": report["holdout_raw_metrics"],
                "probability_calibration": calibration.to_dict(),
                "channel_data_fingerprint": data_fingerprint,
                "fingerprint_scope": "development_rows_and_predictors",
                "holdout_contract_fingerprint": holdout_contract_fingerprint,
                "model_contract_fingerprint": holdout_contract_fingerprint,
                "release_policy_fingerprint": self._release_policy_fingerprint(),
                "performance_policy_version": self.performance_policy.version,
                "performance_tier": final_assessment.performance_tier.value,
                "deployment_scope": final_assessment.deployment_scope.value,
                "minimum_research_gates": self.performance_policy.minimum_research.as_dict(),
                "preferred_gates": self.performance_policy.preferred.as_dict(),
                "research_disclaimer": (
                    self.performance_policy.research_disclaimer
                    if final_assessment.disclaimer_required
                    else None
                ),
                "holdout_evidence_reused": prior_evidence is not None,
                "gold_contract": selected_gold_contract,
                "run_contract_sha256": run_contract_sha256,
                "source_tree_sha256": (
                    json.loads(run_contract_path.read_text(encoding="utf-8")).get(
                        "source_tree_sha256"
                    )
                    if run_contract_path.is_file()
                    else None
                ),
                "config_fingerprint": (
                    json.loads(run_contract_path.read_text(encoding="utf-8")).get(
                        "config_fingerprint"
                    )
                    if run_contract_path.is_file()
                    else None
                ),
                "converter_preflight": converter_preflight,
                "test_cohort_status": self.config.training.reused_test_cohort_status,
                "clinical_validation_status": self.config.training.clinical_validation_status,
                "cv_gates_passed": development_assessment.passes_minimum,
                "holdout_gates_passed": holdout_assessment.passes_minimum,
            }
            _write_immutable_text(
                bundle / "candidate_manifest.json",
                json.dumps(metadata, indent=2, sort_keys=True),
            )
            report["candidate_bundle"] = str(bundle.relative_to(self.root)).replace(
                "\\", "/"
            )
        self._write_report(sensor_channel, report)
        return report

    def _write_report(self, sensor_channel: str, report: dict[str, Any]) -> None:
        # Context is attached before ``run`` starts so every return path is
        # serialized once with the same run-scoped admission metadata.
        for key, value in self._report_context.items():
            report.setdefault(key, value)
        run_contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
        if run_contract_path.is_file():
            run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
            report.setdefault("run_id", run_contract.get("run_id"))
            report.setdefault(
                "source_tree_sha256", run_contract.get("source_tree_sha256")
            )
            report.setdefault(
                "config_fingerprint", run_contract.get("config_fingerprint")
            )
            report.setdefault(
                "source_checksums", run_contract.get("source_checksums", {})
            )
            report.setdefault(
                "package_versions", run_contract.get("package_versions", {})
            )
            report.setdefault("run_contract_sha256", _sha256(run_contract_path))
        reports_root = Path(self.config.artifacts.get("reports_dir", "reports"))
        reports_root = (
            reports_root if reports_root.is_absolute() else self.root / reports_root
        )
        target = (
            reports_root
            / "model_eval"
            / self.config.model_version
            / f"{sensor_channel}_full.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        channel_dir = (
            reports_root
            / "model_eval"
            / "ml"
            / self.config.model_version
            / sensor_channel
        )
        channel_dir.mkdir(parents=True, exist_ok=True)
        if report.get("requires_dl"):
            trigger_path = channel_dir / "local_ml_dl_trigger.json"
            trigger = {
                "schema_version": "1.0.0",
                "sensor_channel": sensor_channel,
                "requires_dl": True,
                "trigger_reason": report.get("trigger_reason"),
                "best_classical_model": report.get("model_name"),
                "best_classical_metrics": report.get("development_metrics"),
                "failing_minimum_metrics": report.get("failing_minimum_metrics", []),
                "minimum_research_gates": (
                    self.performance_policy.minimum_research.as_dict()
                ),
                "preferred_gates": self.performance_policy.preferred.as_dict(),
                "performance_policy_version": self.performance_policy.version,
                "release_policy_fingerprint": self._release_policy_fingerprint(),
                "predictor_columns": report.get("predictor_columns", []),
                "development_participants": list(DEVELOPMENT),
                "holdout_participants": list(HOLDOUT),
                "holdout_evaluated": False,
                "next_owner": "parent_cli_combined_colab_handoff",
            }
            _write_immutable_text(
                trigger_path,
                json.dumps(trigger, indent=2, sort_keys=True),
            )
            report["local_ml_dl_trigger"] = str(trigger_path)
        serialized_report = json.dumps(report, indent=2, sort_keys=True)
        _write_immutable_text(target, serialized_report)
        if not self._strict_run_scope:
            canonical_target = (
                self.root
                / "reports"
                / "model_eval"
                / self.config.model_version
                / f"{sensor_channel}_full.json"
            )
            if canonical_target != target:
                canonical_target.parent.mkdir(parents=True, exist_ok=True)
                _write_immutable_text(canonical_target, serialized_report)
        _write_immutable_text(
            channel_dir / "candidate_metrics.json",
            json.dumps(report, indent=2, sort_keys=True),
        )
        folds = [
            {
                "candidate": candidate,
                **{
                    key: value
                    for key, value in fold.items()
                    if key not in {"window_predictions", "session_predictions"}
                },
            }
            for candidate, result in report.get("candidate_results", {}).items()
            for fold in result.get("folds", [])
        ]
        fold_frame = pd.DataFrame(folds)
        parquet_frame = _parquet_safe_frame(fold_frame)
        parquet_frame.to_parquet(
            channel_dir / "participant_fold_metrics.parquet", index=False
        )
        # Kept as a human-readable compatibility artifact; the parquet file is
        # the canonical typed output consumed by analysis tools.
        parquet_frame.to_csv(channel_dir / "participant_fold_metrics.csv", index=False)
        selected_folds = (
            report.get("candidate_results", {})
            .get(report.get("model_name"), {})
            .get("folds", [])
        )
        window_predictions = [
            row for fold in selected_folds for row in fold.get("window_predictions", [])
        ]
        session_predictions = [
            row
            for fold in selected_folds
            for row in fold.get("session_predictions", [])
        ]
        pd.DataFrame(
            window_predictions,
            columns=[
                "participant_id",
                "phase",
                "label",
                "window_id",
                "probability",
                "outer_fold",
            ],
        ).to_csv(channel_dir / "nested_oof_window_predictions.csv", index=False)
        session_frame = pd.DataFrame(
            session_predictions,
            columns=[
                "participant_id",
                "phase",
                "label",
                "probability",
                "participant_phase_weight",
                "outer_fold",
            ],
        )
        session_frame.to_csv(
            channel_dir / "nested_oof_session_predictions.csv", index=False
        )
        participant_rows: list[dict[str, Any]] = []
        if not session_frame.empty:
            for participant_id, participant in session_frame.groupby(
                "participant_id", sort=True
            ):
                participant_rows.append(
                    {
                        "participant_id": participant_id,
                        "probability_source": "uncalibrated_nested_outer_oof",
                        "decision_threshold": THRESHOLD,
                        **_metrics(
                            participant["label"].to_numpy(dtype=int),
                            participant["probability"].to_numpy(dtype=float),
                            THRESHOLD,
                        ),
                    }
                )
        pd.DataFrame(participant_rows).to_csv(
            channel_dir / "participant_metrics.csv", index=False
        )
        feature_occurrence: dict[str, int] = {}
        for fold in selected_folds:
            for feature in fold.get("feature_order", []):
                feature_occurrence[feature] = feature_occurrence.get(feature, 0) + 1
        pd.DataFrame(
            [
                {
                    "feature": feature,
                    "selected_outer_folds": count,
                    "outer_fold_count": len(selected_folds),
                    "selection_fraction": count / max(len(selected_folds), 1),
                }
                for feature, count in sorted(feature_occurrence.items())
            ],
            columns=[
                "feature",
                "selected_outer_folds",
                "outer_fold_count",
                "selection_fraction",
            ],
        ).to_csv(channel_dir / "feature_stability.csv", index=False)
        pd.DataFrame(report.get("window_ablation", [])).to_csv(
            channel_dir / "window_ablation.csv", index=False
        )
        pd.DataFrame(
            [
                {"cohort": name, **payload}
                for name, payload in report.get("quality_cohort_ablation", {}).items()
            ]
        ).to_csv(channel_dir / "quality_cohort_ablation.csv", index=False)
        pd.DataFrame(
            [
                {"feature_group": name, **payload}
                for name, payload in report.get("feature_group_ablation", {}).items()
            ]
        ).to_csv(channel_dir / "feature_group_ablation.csv", index=False)
        _write_immutable_text(
            channel_dir / "calibration_report.json",
            json.dumps(
                {
                    "sensor_channel": sensor_channel,
                    "method": self.config.training.probability_calibration,
                    "threshold": report.get("threshold"),
                    "threshold_source": report.get("threshold_source"),
                    "calibration": report.get("probability_calibration"),
                    "status": (
                        "evaluated"
                        if report.get("probability_calibration") is not None
                        else "not_evaluated"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        confusion_dir = channel_dir / "confusion_matrices"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        for candidate, result in report.get("candidate_results", {}).items():
            _write_immutable_text(
                confusion_dir / f"{candidate}.json",
                json.dumps(
                    {
                        "candidate": candidate,
                        "confusion_matrix": result.get("metrics", {}).get(
                            "confusion_matrix", []
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
        selection = {
            "sensor_channel": sensor_channel,
            "training_channels": list(self.config.training.training_channels),
            "selected_model": report.get("model_name"),
            "predictor_columns": report.get("predictor_columns", []),
            "development_metrics": report.get("development_metrics"),
            "development_gates": report.get("development_gates"),
            "holdout_evaluated": report.get("holdout_evaluated", False),
            "release_status": report.get("release_status"),
            "performance_policy_version": report.get("performance_policy_version"),
            "performance_tier": report.get("performance_tier"),
            "deployment_scope": report.get("deployment_scope"),
            "requires_dl": report.get("requires_dl", False),
            "trigger_reason": report.get("trigger_reason"),
            "local_ml_dl_trigger": report.get("local_ml_dl_trigger"),
        }
        _write_immutable_text(
            channel_dir / "model_selection.json",
            json.dumps(selection, indent=2, sort_keys=True),
        )


def run_baseline_ml(
    project_root: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    sensor: str | None = None,
    hrv_admission: Mapping[str, Any] | None = None,
    defer_reused_internal_test: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run only the configured independent PPG baselines without partial fitting."""

    channels = tuple(config.training.training_channels)
    if channels != TRAINING_CHANNELS:
        raise ValueError(
            "Canonical local ML run requires independent Green, IR, and Red"
        )
    if sensor is not None and sensor not in channels:
        raise ValueError(f"Training sensor is not allowed: {sensor}")
    selected_channels = (sensor,) if sensor is not None else channels
    if hrv_admission is None:
        hrv_admission = load_hrv_training_admission(project_root, config, paths)
    if hrv_admission is not None:
        admitted_channels = hrv_admission.get("channels")
        if not isinstance(admitted_channels, dict) or set(admitted_channels) != set(
            channels
        ):
            raise ValueError("hrv_training_admission_channel_scope_mismatch")
    input_readiness_configured = "gold_input_readiness" in config.artifacts
    readiness_path = _configured_path(
        project_root,
        config,
        "gold_input_readiness",
        paths.gold_dir / "data_readiness.json",
    )
    readiness = (
        json.loads(readiness_path.read_text(encoding="utf-8"))
        if readiness_path.is_file()
        else {}
    )
    trainer = CanonicalClassicalTrainer(
        project_root,
        config,
        paths,
        defer_reused_internal_test=defer_reused_internal_test,
    )
    try:
        gold_validation = validate_gold_input(project_root, config, paths)
    except (FileNotFoundError, ValueError) as exc:
        gold_validation = {"status": "blocked", "blocking_reason": str(exc)}
    else:
        gold_validation.setdefault("status", "ready")

    reports: dict[str, dict[str, Any]] = {}
    channel_gold_validations: dict[str, dict[str, Any]] = {}
    for sensor_channel in selected_channels:
        try:
            channel_validation = validate_gold_input(
                project_root,
                config,
                paths,
                sensor_channel=sensor_channel,
            )
        except (FileNotFoundError, ValueError) as exc:
            channel_validation = {
                "status": "blocked",
                "sensor_channel": sensor_channel,
                "blocking_reason": str(exc),
            }
        channel_gold_validations[sensor_channel] = channel_validation
        admission_row = (
            hrv_admission.get("channels", {}).get(sensor_channel)
            if hrv_admission is not None
            else None
        )
        if hrv_admission is not None and not (
            isinstance(admission_row, Mapping) and admission_row.get("eligible") is True
        ):
            status = (
                str(
                    admission_row.get("status", "blocked_hrv_prv_insufficient_evidence")
                )
                if isinstance(admission_row, Mapping)
                else "blocked_hrv_prv_insufficient_evidence"
            )
            reason = (
                str(admission_row.get("reason"))
                if isinstance(admission_row, Mapping) and admission_row.get("reason")
                else "hrv_prv_training_admission_failed"
            )
            reports[sensor_channel] = trainer._blocked_coverage_report(
                sensor_channel,
                f"{reason};gold={channel_validation.get('blocking_reason', 'ready')}",
                gold_input_validation=channel_validation,
                extra={
                    "training_channels": list(channels),
                    "channel_status": "blocked_hrv_prv_admission",
                    "training_admission_status": status,
                    "hrv_training_admission": dict(admission_row or {}),
                    "trigger_reason": "hrv_prv_admission_failed",
                    "requires_dl": False,
                    "release_status": "blocked_hrv_prv_admission",
                },
            )
            continue
        if channel_validation.get("status") != "ready":
            reports[sensor_channel] = {
                "schema_version": "1.1.0",
                "model_version": config.model_version,
                "sensor_channel": sensor_channel,
                "model_family": "classical",
                "training_channels": list(channels),
                "model_name": None,
                "holdout_evaluated": False,
                "channel_status": "not_evaluated",
                "requires_dl": False,
                "trigger_reason": None,
                "release_status": "blocked_insufficient_ppg_coverage",
                "blocking_reason": channel_validation.get(
                    "blocking_reason", "channel_gold_input_not_ready"
                ),
                "gold_input_validation": channel_validation,
            }
            if hrv_admission is not None:
                reports[sensor_channel]["training_admission_status"] = (
                    "ready"
                    if channel_validation.get("status") == "ready"
                    else "blocked_insufficient_ppg_coverage"
                )
                reports[sensor_channel]["hrv_training_admission"] = dict(
                    admission_row or {}
                )
            trainer._write_report(sensor_channel, reports[sensor_channel])
        else:
            trainer._report_context = {
                "training_channels": list(channels),
                "gold_input_validation": channel_validation,
                "training_admission_status": "ready",
                "hrv_training_admission": dict(admission_row or {}),
            }
            reports[sensor_channel] = trainer.run(sensor_channel)

    candidate_status = trainer._release_candidate_status()
    ready_channels = [
        channel
        for channel, report in reports.items()
        if report.get("release_status") == candidate_status
        or bool(report.get("development_candidate_bundle"))
    ]
    training_ready = bool(ready_channels)
    reports_root = _configured_path(
        project_root,
        config,
        "reports_dir",
        project_root / "reports",
    )
    summary = {
        "schema_version": "1.0.0",
        "model_version": config.model_version,
        "training_channels": list(selected_channels),
        "all_training_channels": list(channels),
        "hrv_training_admission": hrv_admission,
        "gold_input_validation": gold_validation,
        "channel_gold_input_validation": channel_gold_validations,
        "channel_status": {
            channel: {
                "selected_model": report.get("model_name"),
                "release_status": report.get("release_status"),
                "performance_tier": report.get("performance_tier"),
                "deployment_scope": report.get("deployment_scope"),
                "requires_dl": bool(report.get("requires_dl")),
                "trigger_reason": report.get("trigger_reason"),
            }
            for channel, report in reports.items()
        },
        "colab_handoff_channels": [
            channel for channel, report in reports.items() if report.get("requires_dl")
        ],
        "ready_channels": [
            *ready_channels,
        ],
    }
    summary_path = (
        reports_root / "model_eval" / config.model_version / "local_ml_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not trainer._strict_run_scope:
        canonical_summary_path = (
            project_root
            / "reports"
            / "model_eval"
            / config.model_version
            / "local_ml_summary.json"
        )
        if canonical_summary_path != summary_path:
            canonical_summary_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )

    # A historical in-place readiness file is updated only when this function
    # owns the output Gold directory. Immutable input Gold is never modified.
    if readiness_path.is_file() and not input_readiness_configured:
        # training_status describes data eligibility for fitting and is set by
        # Gold construction. Model/release evidence belongs in a separate key.
        readiness.setdefault("training_status", "blocked")
        readiness["baseline_ml_status"] = (
            candidate_status if training_ready else "blocked"
        )
        readiness["model_performance_status"] = (
            "passed" if training_ready else "below_minimum_or_not_evaluated"
        )
        readiness["baseline_ml_reports"] = {
            sensor: report.get("release_status", "blocked")
            for sensor, report in reports.items()
        }
        readiness["model_release_status_by_channel"] = dict(
            readiness["baseline_ml_reports"]
        )
        readiness["baseline_ml_training_channels"] = list(channels)
        readiness["baseline_ml_dl_handoff_channels"] = summary["colab_handoff_channels"]
        readiness["model_release_status"] = "blocked"
        readiness["model_release_reason"] = "onnx_export_not_verified"
        readiness["status"] = (
            "ready"
            if readiness.get("ppg_gold_status") == "ready" and training_ready
            else "blocked"
        )
        serialized_readiness = json.dumps(
            readiness, indent=2, sort_keys=True, default=str
        )
        temporary = readiness_path.with_name(
            f".{readiness_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(serialized_readiness, encoding="utf-8")
        try:
            temporary.replace(readiness_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return reports
