"""Checksum-bound orchestration for reproducible local pipeline runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import RunContractFingerprint
from .settings import PipelineConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    """Hash only canonical source/config files used by the local pipeline."""

    root = project_root.resolve()
    files: list[dict[str, str]] = []
    for relative_root in (
        Path("PipelinePython/common"),
        Path("PipelinePython/configs"),
        Path("PipelinePython/01_data_preprocess"),
        Path("PipelinePython/02_peak_to_peak"),
        Path("PipelinePython/03_stress_onnx/tools"),
    ):
        source_root = root / relative_root
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256": _sha256_file(path),
                }
            )
    for relative_path in (
        Path("PipelinePython/__init__.py"),
        Path("PipelinePython/__main__.py"),
        Path(
            "PipelinePython/03_stress_onnx/notebooks/BiosignalStressTraining_Colab.ipynb"
        ),
    ):
        path = root / relative_path
        if path.is_file():
            files.append(
                {
                    "path": str(relative_path).replace("\\", "/"),
                    "sha256": _sha256_file(path),
                }
            )
    if not files:
        # Compatibility fixtures exercise orchestration with a temporary
        # project that has no checkout.  Keep the missing identity explicit;
        # release stages must reject this value before publishing evidence.
        return "UNAVAILABLE:canonical_source_tree_missing"
    return _sha256_payload(files)


def _git_identity(project_root: Path) -> tuple[str, str]:
    """Read source Git identity without hashing generated run artifacts."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--short",
                "--untracked-files=all",
                "--",
                "PipelinePython",
                "pyproject.toml",
                "requirements.lock",
                "uv.lock",
                "PRD.md",
                "AGENTS.md",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return head or "UNAVAILABLE", _sha256_payload(status)
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE", "UNAVAILABLE"


def build_run_contract_fingerprint(
    context: "PipelineRunContext",
    *,
    raw_manifest: Mapping[str, Any] | None = None,
    artifact_checksums: Mapping[str, str] | None = None,
) -> RunContractFingerprint:
    """Build the immutable identity shared by raw, Gold, training and export."""

    raw = raw_manifest or raw_corpus_manifest(context.config)
    source_checksums = {
        f"{record.get('role', 'unknown')}:{record['path']}": str(record["sha256"])
        for record in raw.get("records", [])
        if "path" in record and "sha256" in record
    }
    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "neurokit2",
        "onnx",
        "onnxruntime",
        "skl2onnx",
        "onnxmltools",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "UNAVAILABLE"
    git_head, git_state = _git_identity(context.project_root)
    split_fingerprint = _sha256_payload(
        {
            "development": [f"ID-{index:02d}" for index in range(1, 29)],
            "holdout": [f"ID-{index:02d}" for index in range(29, 33)],
            "outer_group_splits": context.config.training.outer_group_splits,
            "outer_fold_size": context.config.training.outer_fold_size,
        }
    )
    return RunContractFingerprint(
        run_id=context.run_id,
        source_checksums=source_checksums,
        source_tree_sha256=source_tree_sha256(context.project_root),
        git_head=git_head,
        git_state=git_state,
        config_fingerprint=context.config_sha256,
        preprocessing_version=context.config.preprocessing_version,
        feature_version=context.config.feature_version,
        split_fingerprint=split_fingerprint,
        package_versions=packages,
        artifact_checksums=dict(artifact_checksums or {}),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _path_fingerprint(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        entries = [
            {
                "path": str(item.relative_to(path)).replace("\\", "/"),
                "sha256": _sha256_file(item),
            }
            for item in sorted(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
        ]
        return _sha256_payload(entries)
    raise FileNotFoundError(f"Pipeline output does not exist: {path}")


def raw_corpus_manifest(config: PipelineConfig) -> dict[str, Any]:
    """Fingerprint immutable production inputs without writing beside raw data."""

    roots = (
        ("renamed", config.raw_source.renamed_dir),
        ("trimmed", config.raw_source.trimmed_dir),
    )
    records: list[dict[str, str]] = []
    for role, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Raw source directory is missing: {root}")
        for source in sorted(root.rglob("*.csv")):
            records.append(
                {
                    "role": role,
                    "path": str(source.relative_to(root)).replace("\\", "/"),
                    "sha256": _sha256_file(source),
                }
            )
    anchor = config.raw_source.anchor_csv
    if not anchor.is_file():
        raise FileNotFoundError(f"Anchor CSV is missing: {anchor}")
    records.append(
        {
            "role": "anchor",
            "path": anchor.name,
            "sha256": _sha256_file(anchor),
        }
    )
    return {
        "schema_version": "raw-integrity-v1",
        "file_count": len(records),
        "records": records,
        "corpus_sha256": _sha256_payload(records),
    }


@dataclass(frozen=True)
class PipelineRunContext:
    """Immutable dependencies shared by every stage in one local run."""

    project_root: Path
    config: PipelineConfig
    config_path: Path
    run_id: str
    participant: str | None = None

    @property
    def artifacts_root(self) -> Path:
        value = Path(self.config.artifacts["artifacts_root"])
        return value if value.is_absolute() else self.project_root / value

    @property
    def state_path(self) -> Path:
        return self.artifacts_root / "run_state.json"

    @property
    def config_sha256(self) -> str:
        return _sha256_payload(asdict(self.config))

    @property
    def contract_fingerprint(self) -> RunContractFingerprint:
        return build_run_contract_fingerprint(self)


@dataclass(frozen=True)
class StageResult:
    """Traceable outputs produced by one stage."""

    outputs: tuple[Path, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    """Small dependency-inversion boundary for orchestration stages."""

    name: str

    def declared_inputs(self, context: PipelineRunContext) -> tuple[Path, ...]:
        """Return durable stage inputs that participate in resume hashing."""

    def execute(self, context: PipelineRunContext) -> StageResult:
        """Execute the stage and return its durable outputs."""


@dataclass(frozen=True)
class CallablePipelineStage:
    name: str
    action: Callable[[PipelineRunContext], StageResult]
    dependencies: tuple[Path, ...] = ()

    def declared_inputs(self, context: PipelineRunContext) -> tuple[Path, ...]:
        del context
        return self.dependencies

    def execute(self, context: PipelineRunContext) -> StageResult:
        return self.action(context)


class PipelineOrchestrator:
    """Run injected stages with checksum-aware, fail-fast resume semantics."""

    def __init__(
        self,
        context: PipelineRunContext,
        stages: Sequence[PipelineStage],
    ) -> None:
        self.context = context
        self._stages = {stage.name: stage for stage in stages}
        if len(self._stages) != len(stages):
            raise ValueError("Pipeline stage names must be unique")

    def run(
        self, stage_names: Sequence[str], *, resume: bool = False
    ) -> dict[str, Any]:
        unknown = set(stage_names) - set(self._stages)
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")

        raw_before = raw_corpus_manifest(self.context.config)
        self.context.artifacts_root.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        if state and state.get("run_id") != self.context.run_id:
            raise ValueError("Existing run state belongs to another run_id")
        # Check the durable run state before comparing the immutable contract.
        # This preserves the actionable failure reason for a changed raw corpus
        # or configuration instead of masking it as a generic contract clash.
        if state:
            previous_raw = state.get("raw_corpus_sha256")
            if previous_raw != raw_before["corpus_sha256"]:
                raise RuntimeError("Raw corpus changed; refusing immutable run reuse")
            if state.get("config_sha256") != self.context.config_sha256:
                raise RuntimeError(
                    "Pipeline config changed; refusing immutable run reuse"
                )
        raw_pre_path = self.context.artifacts_root / "raw_integrity_pre.json"
        _atomic_json(raw_pre_path, raw_before)
        contract = build_run_contract_fingerprint(
            self.context,
            raw_manifest=raw_before,
            artifact_checksums={
                "raw_integrity_pre.json": _sha256_file(raw_pre_path),
                "raw_corpus_sha256": raw_before["corpus_sha256"],
            },
        )
        if self.context.config.artifacts.get(
            "run_id"
        ) and contract.source_tree_sha256.startswith("UNAVAILABLE:"):
            raise RuntimeError("canonical_source_tree_fingerprint_unavailable")
        contract_path = self.context.artifacts_root / "run_contract_fingerprint.json"
        contract_payload = contract.as_dict()
        if contract_path.is_file():
            existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing_contract != contract_payload:
                raise RuntimeError("run_contract_fingerprint_immutability_violation")
        else:
            _atomic_json(contract_path, contract_payload)
        if state:
            if state.get("source_tree_sha256") != contract.source_tree_sha256:
                raise RuntimeError(
                    "Canonical source tree changed; refusing immutable run reuse"
                )
            if state.get("run_contract_fingerprint") != _sha256_payload(
                contract_payload
            ):
                raise RuntimeError("Run contract changed; refusing immutable run reuse")

        state = state or {
            "schema_version": "pipeline-run-state-v1",
            "run_id": self.context.run_id,
            "participant": self.context.participant,
            "created_at_utc": _utc_now(),
            "stages": {},
        }
        state.update(
            {
                "config_path": str(self.context.config_path),
                "config_sha256": self.context.config_sha256,
                "raw_corpus_sha256": raw_before["corpus_sha256"],
                "source_tree_sha256": contract.source_tree_sha256,
                "run_contract_fingerprint": _sha256_payload(contract.as_dict()),
                "updated_at_utc": _utc_now(),
            }
        )
        self._write_state(state)

        upstream_fingerprints: dict[str, str] = {}
        for stage_name in stage_names:
            stage = self._stages[stage_name]
            dependency_paths = stage.declared_inputs(self.context)
            try:
                declared_inputs = [
                    {
                        "path": str(path.resolve()),
                        "sha256": _path_fingerprint(path.resolve()),
                    }
                    for path in dependency_paths
                ]
            except Exception as exc:
                state["stages"][stage_name] = {
                    "status": "failed",
                    "declared_inputs": [
                        {"path": str(path.resolve())} for path in dependency_paths
                    ],
                    "failed_at_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                self._write_state(state)
                self._write_raw_integrity_post(raw_before, state)
                raise
            input_sha256 = _sha256_payload(
                {
                    "config": self.context.config_sha256,
                    "raw": raw_before["corpus_sha256"],
                    "upstream": upstream_fingerprints,
                    "declared_inputs": declared_inputs,
                }
            )
            previous = state["stages"].get(stage_name, {})
            if resume and self._can_resume(previous, input_sha256):
                upstream_fingerprints[stage_name] = str(previous["output_sha256"])
                continue

            state["stages"][stage_name] = {
                "status": "running",
                "input_sha256": input_sha256,
                "declared_inputs": declared_inputs,
                "started_at_utc": _utc_now(),
            }
            self._write_state(state)
            try:
                result = stage.execute(self.context)
                output_records = [
                    {
                        "path": str(path.resolve()),
                        "sha256": _path_fingerprint(path.resolve()),
                    }
                    for path in result.outputs
                ]
                output_sha256 = _sha256_payload(output_records)
                state["stages"][stage_name] = {
                    "status": "completed",
                    "input_sha256": input_sha256,
                    "declared_inputs": declared_inputs,
                    "output_sha256": output_sha256,
                    "outputs": output_records,
                    "metadata": dict(result.metadata),
                    "started_at_utc": state["stages"][stage_name]["started_at_utc"],
                    "completed_at_utc": _utc_now(),
                }
                upstream_fingerprints[stage_name] = output_sha256
                self._write_state(state)
            except (Exception, KeyboardInterrupt) as exc:
                state["stages"][stage_name].update(
                    {
                        "status": "failed",
                        "failed_at_utc": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                self._write_state(state)
                try:
                    self._write_raw_integrity_post(raw_before, state)
                except RuntimeError as integrity_error:
                    raise integrity_error from exc
                raise

        self._write_raw_integrity_post(raw_before, state)
        state["completed_at_utc"] = _utc_now()
        self._write_state(state)
        return state

    def _load_state(self) -> dict[str, Any] | None:
        if not self.context.state_path.is_file():
            return None
        payload = json.loads(self.context.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Pipeline run state must be a JSON object")
        return payload

    def _write_state(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.context.state_path, state)

    def _write_raw_integrity_post(
        self, raw_before: Mapping[str, Any], state: dict[str, Any]
    ) -> None:
        """Persist the raw post-check even when a downstream stage fails."""

        raw_after = raw_corpus_manifest(self.context.config)
        _atomic_json(self.context.artifacts_root / "raw_integrity_post.json", raw_after)
        if raw_after["corpus_sha256"] != raw_before["corpus_sha256"]:
            raise RuntimeError("Raw corpus changed during pipeline execution")
        state["raw_integrity_verified"] = True
        self._write_state(state)

    @staticmethod
    def _can_resume(previous: Mapping[str, Any], input_sha256: str) -> bool:
        if previous.get("status") != "completed":
            return False
        if previous.get("input_sha256") != input_sha256:
            return False
        outputs = previous.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        current = []
        for record in outputs:
            path = Path(str(record.get("path", "")))
            if not path.exists():
                return False
            current.append(
                {"path": str(path.resolve()), "sha256": _path_fingerprint(path)}
            )
        return _sha256_payload(current) == previous.get("output_sha256")
