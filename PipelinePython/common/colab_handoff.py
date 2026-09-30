"""Fail-closed local handoff contracts for the conditional Colab DL run.

This module never uploads data and never starts a remote runtime.  It creates
and validates the small manifest that a researcher manually copies alongside
the already-built Gold IBI parquet artifact in Google Drive.  The manifest
binds a Colab resume checkpoint to one Gold checksum, one fixed participant
contract, and the local-ML channels that actually require DL escalation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .settings import TRAINING_CHANNELS

PPG_CHANNELS = TRAINING_CHANNELS
DEVELOPMENT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(1, 29))
HOLDOUT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(29, 33))
DL_ARCHITECTURES = ("1d_cnn", "tcn")
IBI_SEQUENCE_LENGTH = 720
HANDOFF_SCHEMA_VERSION = "colab-dl-handoff-v1"
FORBIDDEN_PREDICTOR_PREFIXES = ("ecg_", "pat_", "motion_")
DL_TRIGGER_REASONS = frozenset(
    {
        "local_ml_development_metrics_below_minimum_research_gate",
        # Read-only compatibility for artifacts produced before tiered policy.
        "local_ml_balanced_accuracy_below_gate",
    }
)


class HandoffContractError(ValueError):
    """Raised when a handoff or resume manifest violates the DL contract."""


@dataclass(frozen=True)
class FoldContract:
    """The closed participant split used by the Colab outer LOSO loop."""

    development_participants: tuple[str, ...] = DEVELOPMENT_PARTICIPANTS
    holdout_participants: tuple[str, ...] = HOLDOUT_PARTICIPANTS
    outer_cv: str = "leave_one_participant_out"

    def validate(self) -> None:
        if self.development_participants != DEVELOPMENT_PARTICIPANTS:
            raise HandoffContractError(
                "development participant contract must be ID-01..ID-28"
            )
        if self.holdout_participants != HOLDOUT_PARTICIPANTS:
            raise HandoffContractError(
                "holdout participant contract must be ID-29..ID-32"
            )
        if self.outer_cv != "leave_one_participant_out":
            raise HandoffContractError("outer_cv must be leave_one_participant_out")


@dataclass(frozen=True)
class ColabDlHandoff:
    """Manifest payload accepted by the DL-only Colab notebook."""

    run_id: str
    gold_ibi_relative_path: str
    gold_sha256: str
    training_config_sha256: str
    local_ml_handoff: Mapping[str, Mapping[str, Any]]
    fold_contract: FoldContract = FoldContract()
    schema_version: str = HANDOFF_SCHEMA_VERSION
    sequence_length: int = IBI_SEQUENCE_LENGTH
    value_column: str = "ibi_values"
    mask_column: str = "ibi_mask"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest with an explicit channel list."""

        payload = asdict(self)
        payload["dl_handoff_channels"] = list(self.dl_handoff_channels)
        return payload

    @property
    def dl_handoff_channels(self) -> tuple[str, ...]:
        """Return only channels explicitly failed over by local ML."""

        return tuple(
            channel
            for channel, result in self.local_ml_handoff.items()
            if bool(result.get("requires_dl"))
        )

    def validate(self) -> None:
        if self.schema_version != HANDOFF_SCHEMA_VERSION:
            raise HandoffContractError("unsupported handoff schema version")
        if not self.run_id.strip():
            raise HandoffContractError("run_id is required")
        if Path(self.gold_ibi_relative_path).name != "gold_ibi_sequences_v3.parquet":
            raise HandoffContractError(
                "handoff must reference the canonical Gold IBI artifact"
            )
        if len(self.gold_sha256) != 64 or len(self.training_config_sha256) != 64:
            raise HandoffContractError(
                "Gold and training-config checksums must be SHA-256"
            )
        if self.sequence_length != IBI_SEQUENCE_LENGTH:
            raise HandoffContractError("DL input must contain exactly 720 IBI values")
        if (self.value_column, self.mask_column) != ("ibi_values", "ibi_mask"):
            raise HandoffContractError(
                "only ibi_values plus ibi_mask are valid DL inputs"
            )
        self.fold_contract.validate()
        _validate_local_ml_handoff(self.local_ml_handoff)
        if not self.dl_handoff_channels:
            raise HandoffContractError("no local-ML channels were handed off to DL")


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 without loading Gold data into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_handoff(
    *,
    run_id: str,
    gold_ibi_path: Path,
    training_config_sha256: str,
    local_ml_handoff: Mapping[str, Mapping[str, Any]],
) -> ColabDlHandoff:
    """Build a validated manifest; this function performs no network I/O."""

    artifact = gold_ibi_path.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Gold IBI artifact does not exist: {artifact}")
    handoff = ColabDlHandoff(
        run_id=run_id,
        gold_ibi_relative_path=artifact.name,
        gold_sha256=sha256_file(artifact),
        training_config_sha256=training_config_sha256,
        local_ml_handoff=local_ml_handoff,
    )
    handoff.validate()
    return handoff


def write_handoff_manifest(handoff: ColabDlHandoff, destination: Path) -> Path:
    """Atomically write a local manifest for manual transfer to Google Drive."""

    handoff.validate()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(handoff.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def load_handoff_manifest(path: Path) -> ColabDlHandoff:
    """Load and validate a manifest created by :func:`write_handoff_manifest`."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    fold_payload = payload.get("fold_contract", {})
    handoff = ColabDlHandoff(
        run_id=str(payload.get("run_id", "")),
        gold_ibi_relative_path=str(payload.get("gold_ibi_relative_path", "")),
        gold_sha256=str(payload.get("gold_sha256", "")),
        training_config_sha256=str(payload.get("training_config_sha256", "")),
        local_ml_handoff=payload.get("local_ml_handoff", {}),
        fold_contract=FoldContract(
            development_participants=tuple(
                fold_payload.get("development_participants", ())
            ),
            holdout_participants=tuple(fold_payload.get("holdout_participants", ())),
            outer_cv=str(fold_payload.get("outer_cv", "")),
        ),
        schema_version=str(payload.get("schema_version", "")),
        sequence_length=int(payload.get("sequence_length", 0)),
        value_column=str(payload.get("value_column", "")),
        mask_column=str(payload.get("mask_column", "")),
    )
    handoff.validate()
    declared_channels = tuple(payload.get("dl_handoff_channels", ()))
    if declared_channels and declared_channels != handoff.dl_handoff_channels:
        raise HandoffContractError(
            "declared DL handoff channels do not match local ML results"
        )
    return handoff


def validate_gold_artifact(path: Path, handoff: ColabDlHandoff) -> None:
    """Fail closed when the manually transferred Gold artifact changed."""

    if path.name != Path(handoff.gold_ibi_relative_path).name:
        raise HandoffContractError("Gold artifact filename does not match handoff")
    if sha256_file(path) != handoff.gold_sha256:
        raise HandoffContractError("Gold artifact checksum does not match handoff")


def fold_contract_sha256(contract: FoldContract) -> str:
    """Fingerprint only the frozen CV contract used by resume validation."""

    contract.validate()
    encoded = json.dumps(
        asdict(contract), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_resume_manifest(
    resume_manifest: Mapping[str, Any], handoff: ColabDlHandoff
) -> None:
    """Reject stale checkpoints before Colab restores model/optimizer state."""

    handoff.validate()
    expected = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "gold_sha256": handoff.gold_sha256,
        "training_config_sha256": handoff.training_config_sha256,
        "fold_contract_sha256": fold_contract_sha256(handoff.fold_contract),
    }
    for key, value in expected.items():
        if resume_manifest.get(key) != value:
            raise HandoffContractError(f"resume manifest mismatch for {key}")
    if resume_manifest.get("channel") not in handoff.dl_handoff_channels:
        raise HandoffContractError("resume channel was not handed off by local ML")
    if resume_manifest.get("architecture") not in DL_ARCHITECTURES:
        raise HandoffContractError("resume architecture is not supported")
    completed = tuple(resume_manifest.get("completed_folds", ()))
    if not set(completed).issubset(DEVELOPMENT_PARTICIPANTS):
        raise HandoffContractError("resume manifest contains a non-development fold")


def _validate_local_ml_handoff(results: Mapping[str, Mapping[str, Any]]) -> None:
    if not results:
        raise HandoffContractError("local ML handoff results are required")
    unexpected = sorted(set(results) - set(PPG_CHANNELS))
    if unexpected:
        raise HandoffContractError(f"unexpected local ML channels: {unexpected}")
    for channel, result in results.items():
        if not isinstance(result, Mapping):
            raise HandoffContractError(
                f"local ML result for {channel} must be a mapping"
            )
        predictors = tuple(
            str(item).lower() for item in result.get("predictor_columns", ())
        )
        forbidden = [
            column
            for column in predictors
            if column.startswith(FORBIDDEN_PREDICTOR_PREFIXES)
        ]
        if forbidden:
            raise HandoffContractError(
                f"{channel} local ML handoff contains forbidden predictors: {forbidden}"
            )
        requires_dl = bool(result.get("requires_dl"))
        reason = str(result.get("trigger_reason", ""))
        if requires_dl and reason not in DL_TRIGGER_REASONS:
            raise HandoffContractError(
                f"{channel} DL handoff lacks the local-ML trigger reason"
            )
