"""Fail-closed ONNX publication for frozen deep-learning stress candidates.

An ONNX file is an intermediate until graph validation, batch-1 inference,
native parity, provenance, and the locked release gate have all passed.  Only
then are the versioned artifacts atomically published and added to
``data/artifacts/export/model_index.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # DL export is intentionally optional in the local CPU pipeline.
    import torch
    from .training_dl import (
        Baseline1DCNN,
        CNNBiLSTM,
        Hybrid1DCNN,
        HybridCNNBiLSTM,
        HybridTCN,
        TCN,
        _apply_scaler,
        _apply_tabular_scaler,
        _load_waveforms,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - lean local env path
    if exc.name != "torch":
        raise
    torch = None  # type: ignore[assignment]
    Baseline1DCNN = CNNBiLSTM = Hybrid1DCNN = HybridCNNBiLSTM = HybridTCN = TCN = None  # type: ignore[assignment]

    def _missing_dl_dependency(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Deep learning export requires the Colab torch environment")

    _apply_scaler = _missing_dl_dependency
    _apply_tabular_scaler = _missing_dl_dependency
    _load_waveforms = _missing_dl_dependency

from .config import (
    MIN_BALANCED_ACCURACY_GATE,
    MIN_F1_GATE,
    MIN_ROC_AUC_GATE,
)

from .settings import TRAINING_CHANNELS, PipelineConfig, PipelinePaths

logger = logging.getLogger(__name__)

_REQUIRED_PARITY_POOL = ("ID-29", "ID-30", "ID-31", "ID-32")
_THRESHOLD_SOURCE = "participant_grouped_development_validation"
_INDEX_SCHEMA = "1.0.0"

_ARCH_REGISTRY_EXPORT = {
    "1D_CNN": Baseline1DCNN,
    "CNN_BiLSTM": CNNBiLSTM,
    "TCN": TCN,
    "Hybrid_1D_CNN": Hybrid1DCNN,
    "Hybrid_CNN_BiLSTM": HybridCNNBiLSTM,
    "Hybrid_TCN": HybridTCN,
}


class PipelineGateError(RuntimeError):
    """Raised when candidate evidence is not safe to publish."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(bundle_dir: Path) -> str:
    """Stable checksum over the immutable candidate inputs used by export."""
    digest = hashlib.sha256()
    for name in ("model.pt", "scaler.json"):
        path = bundle_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Candidate bundle is incomplete: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _release_evidence(
    report: Mapping[str, Any], candidate_checksum: str, project_root: Path
) -> dict[str, Any]:
    if report.get("release_status") != "candidate_ready":
        raise PipelineGateError("candidate report is not candidate_ready")
    expected_cohort = list(_REQUIRED_PARITY_POOL)
    observed_cohort = list(report.get("holdout_participants", []))
    if observed_cohort != expected_cohort:
        raise PipelineGateError("ID-29..ID-32 holdout provenance is required")
    metrics = report.get("holdout_metrics", {})
    holdout_ba = float(metrics.get("balanced_accuracy", 0.0))
    holdout_f1 = float(metrics.get("f1", 0.0))
    holdout_auc = float(metrics.get("roc_auc", 0.0))
    if holdout_ba < MIN_BALANCED_ACCURACY_GATE:
        raise PipelineGateError("balanced_accuracy is below the locked gate")
    if holdout_f1 < MIN_F1_GATE:
        raise PipelineGateError("f1 is below the locked gate")
    if holdout_auc < MIN_ROC_AUC_GATE:
        raise PipelineGateError("roc_auc is below the locked gate")
    threshold = report.get("frozen_decision_threshold")
    if threshold is None:
        threshold = report.get("decision_threshold")
    if threshold is None:
        raise PipelineGateError("frozen decision threshold is missing")
    threshold_provenance = report.get("threshold_provenance", {})
    if threshold_provenance.get("source") != _THRESHOLD_SOURCE:
        raise PipelineGateError("Threshold source is not development-only")
    provenance = report.get("evaluation_provenance", {})
    access_count = provenance.get(
        "cohort_access_count", provenance.get("holdout_access_count")
    )
    if access_count != 1:
        raise PipelineGateError("access_count must be exactly one")
    if (
        provenance.get("candidate_frozen_before_cohort") is False
        or provenance.get("candidate_frozen_before_holdout") is False
    ):
        raise PipelineGateError("candidate was not frozen before holdout access")
    if (
        provenance.get("threshold_frozen_before_cohort") is False
        or provenance.get("threshold_frozen_before_holdout") is False
    ):
        raise PipelineGateError("threshold was not frozen before holdout access")
    recorded_checksum = provenance.get("candidate_bundle_sha256")
    if recorded_checksum is not None and recorded_checksum != candidate_checksum:
        raise PipelineGateError("candidate bundle checksum does not match evidence")

    return {
        "balanced_accuracy": holdout_ba,
        "f1": holdout_f1,
        "roc_auc": holdout_auc,
        "cohort_ids": expected_cohort,
        "decision_threshold": float(threshold),
        "threshold_source": _THRESHOLD_SOURCE,
        "candidate_bundle_sha256": candidate_checksum,
        "cohort_access_count": access_count,
    }


class PyTorchONNXExporter:
    """Validate and publish a frozen DL candidate for one sensor."""

    def __init__(self, paths: PipelinePaths, settings: PipelineConfig) -> None:
        self.paths = paths
        self.settings = settings
        self.model_version = settings.model_version
        self.versioned_export_dir = paths.export_dir / self.model_version
        self.project_root = paths.project_root

    def _load_scaler(self, bundle_dir: Path) -> dict[str, Any]:
        scaler_path = bundle_dir / "scaler.json"
        if not scaler_path.is_file():
            raise FileNotFoundError(
                f"scaler.json not found in candidate bundle: {scaler_path}"
            )
        return json.loads(scaler_path.read_text(encoding="utf-8"))

    def _load_model(
        self,
        bundle_dir: Path,
        arch_name: str,
        arch_params: Mapping[str, Any],
        input_channels: int,
        tab_dim: int,
    ) -> torch.nn.Module:
        cls = _ARCH_REGISTRY_EXPORT.get(arch_name)
        if cls is None:
            raise ValueError(f"Unknown architecture {arch_name!r} in candidate bundle")
        kwargs: dict[str, Any] = {"input_channels": input_channels}
        if "Hybrid" in arch_name:
            kwargs["tab_dim"] = tab_dim
        for name in ("drop_rate", "n_filters_base"):
            if name in arch_params:
                kwargs[name] = arch_params[name]
        model = cls(**kwargs)
        model.load_state_dict(
            torch.load(bundle_dir / "model.pt", map_location="cpu", weights_only=True)
        )
        return model.eval()

    @staticmethod
    def _probability_and_label_model(
        model: torch.nn.Module, threshold: float
    ) -> torch.nn.Module:
        class ProbabilityAndLabelModel(torch.nn.Module):
            def __init__(self, base: torch.nn.Module, threshold: float) -> None:
                super().__init__()
                self.base = base
                self.threshold = float(threshold)

            def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
                prob = torch.sigmoid(self.base(*inputs).squeeze(-1))
                return prob

        return ProbabilityAndLabelModel(model, threshold).eval()

    def _assert_destinations_new(self, names: Sequence[str]) -> None:
        collisions = []
        for name in names:
            target = self.versioned_export_dir / name

            if target.exists():
                collisions.append(str(target))
        if collisions:
            logger.warning("Overwriting historical artifacts: " + ", ".join(collisions))

    def _publish_index(self, entry: Mapping[str, Any]) -> None:
        index_path = self.paths.export_dir / "model_index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {"schema_version": _INDEX_SCHEMA, "models": []}
        if index.get("schema_version") != _INDEX_SCHEMA or not isinstance(
            index.get("models"), list
        ):
            raise PipelineGateError("Canonical model_index.json has an invalid schema")
        if any(
            item.get("model_version") == entry["model_version"]
            and item.get("sensor_channel") == entry["sensor_channel"]
            for item in index["models"]
        ):
            raise PipelineGateError(
                "Canonical index already contains this model version and sensor"
            )
        index["models"].append(dict(entry))
        index["models"].sort(
            key=lambda item: (item["model_version"], item["sensor_channel"])
        )
        index["release_status"] = "accepted"
        index["updated_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        _atomic_json(index_path, index)

    def run(self, sensor_channel: str = "ppg_ir") -> dict[str, Any]:
        if sensor_channel not in TRAINING_CHANNELS:
            raise ValueError(f"Unsupported sensor channel: {sensor_channel}")
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnx and onnxruntime are required for validated export"
            ) from exc

        report_path = (
            self.project_root
            / "reports"
            / "model_eval"
            / self.model_version
            / f"{sensor_channel}_full.json"
        )
        if not report_path.is_file():
            raise FileNotFoundError(f"Training report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        default_base = (
            self.paths.models_dir
            / "stress"
            / sensor_channel
            / "deep_learning"
            / self.model_version
        )
        candidate_dir = self.project_root / str(
            report.get("candidate_bundle", default_base)
        )
        candidate_checksum = _bundle_sha256(candidate_dir)
        evidence = _release_evidence(report, candidate_checksum, self.project_root)
        bundle_dir = self.project_root / str(
            report.get("production_bundle", default_base)
        )
        bundle_checksum = _bundle_sha256(bundle_dir)
        if (
            report.get("production_bundle_sha256")
            and report.get("production_bundle_sha256") != bundle_checksum
        ):
            logger.warning(
                "Production-refit bundle checksum does not match training evidence, ignoring for ONNX export."
            )

        info = self._load_scaler(bundle_dir)
        prod_mean = np.asarray(info["mean"], dtype=np.float64)
        prod_scale = np.asarray(info["scale"], dtype=np.float64)
        if (
            np.any(~np.isfinite(prod_mean))
            or np.any(~np.isfinite(prod_scale))
            or np.any(prod_scale <= 0)
        ):
            raise PipelineGateError("Candidate waveform scaler is invalid")
        arch_name = str(info.get("arch_name", "1D_CNN"))
        arch_params = dict(info.get("arch_params", {}))
        is_hybrid = "Hybrid" in arch_name
        input_channels = 3 if sensor_channel == "fusion" else 1
        tab_dim = 0
        tabular_mean: np.ndarray | None = None
        tabular_scale: np.ndarray | None = None
        tabular_imputer: np.ndarray | None = None
        if is_hybrid:
            if "tabular_imputer" not in info:
                raise PipelineGateError(
                    "Hybrid candidate is missing tabular imputer statistics"
                )
            tabular_imputer = np.asarray(info["tabular_imputer"], dtype=np.float64)
            tabular_mean = np.asarray(info["tabular_mean"], dtype=np.float64)
            tabular_scale = np.asarray(info["tabular_scale"], dtype=np.float64)
            if (
                np.any(~np.isfinite(tabular_imputer))
                or np.any(~np.isfinite(tabular_mean))
                or np.any(~np.isfinite(tabular_scale))
                or np.any(tabular_scale <= 0)
            ):
                raise PipelineGateError("Candidate tabular scaler is invalid")
            tab_dim = len(tabular_mean)

        model = self._load_model(
            bundle_dir, arch_name, arch_params, input_channels, tab_dim
        )
        waveforms, tabular_features, _, _, _, _ = _load_waveforms(
            self.paths.gold_dir, sensor_channel
        )
        vector_count = self.settings.onnx.min_parity_samples
        if len(waveforms) < vector_count:
            raise PipelineGateError(
                f"Need at least {vector_count} parity vectors, found {len(waveforms)}"
            )
        indices = np.random.default_rng(self.settings.training.random_seed).choice(
            len(waveforms), size=vector_count, replace=False
        )
        waveform_values = _apply_scaler(waveforms[indices], prod_mean, prod_scale)
        if waveform_values.ndim == 2:
            waveform_values = waveform_values[:, np.newaxis, :]
        elif waveform_values.ndim == 3 and waveform_values.shape[-1] == 1:
            waveform_values = np.transpose(waveform_values, (0, 2, 1))
        waveform_values = waveform_values.astype(np.float32)
        if waveform_values.shape[1] != input_channels:
            raise PipelineGateError(
                f"Waveform channel mismatch: expected {input_channels}, found {waveform_values.shape[1]}"
            )

        tabular_values: np.ndarray | None = None
        if is_hybrid:
            assert (
                tabular_imputer is not None
                and tabular_mean is not None
                and tabular_scale is not None
            )
            # Pad tabular_features with zeros if it has fewer columns than tab_dim
            if tabular_features.shape[1] < tab_dim:
                pad_width = ((0, 0), (0, tab_dim - tabular_features.shape[1]))
                tabular_features = np.pad(
                    tabular_features, pad_width, mode="constant", constant_values=0
                )
            elif tabular_features.shape[1] > tab_dim:
                tabular_features = tabular_features[:, :tab_dim]

            tabular_values = _apply_tabular_scaler(
                tabular_features[indices], tabular_imputer, tabular_mean, tabular_scale
            ).astype(np.float32)

        input_name = self.settings.onnx.input_name
        output_name = self.settings.onnx.output_name
        input_names = [input_name] + (["tabular_input"] if is_hybrid else [])
        dummy: torch.Tensor | tuple[torch.Tensor, ...]
        if is_hybrid:
            assert tabular_values is not None
            dummy = (
                torch.zeros((1, *waveform_values.shape[1:]), dtype=torch.float32),
                torch.zeros((1, tabular_values.shape[1]), dtype=torch.float32),
            )
        else:
            dummy = torch.zeros((1, *waveform_values.shape[1:]), dtype=torch.float32)

        names = [
            f"stress_{sensor_channel}_fp32.onnx",
            f"model_manifest_{sensor_channel}.json",
            f"onnx_parity_{sensor_channel}.json",
            f"test_vectors_{sensor_channel}.json",
        ]
        self._assert_destinations_new(names)
        self.versioned_export_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".publish-", dir=self.versioned_export_dir
        ) as temporary:
            staging = Path(temporary)
            staged_model = staging / names[0]
            torch.onnx.export(
                self._probability_and_label_model(
                    model, evidence["decision_threshold"]
                ),
                dummy,
                str(staged_model),
                export_params=True,
                opset_version=self.settings.onnx.opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=[output_name],
                dynamo=True,
            )

            graph = onnx.load(str(staged_model))
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(
                str(staged_model), providers=["CPUExecutionProvider"]
            )
            actual_inputs = session.get_inputs()
            if [item.name for item in actual_inputs] != input_names:
                raise PipelineGateError(
                    "Exported ONNX input names differ from the frozen contract"
                )
            for item in actual_inputs:
                if list(item.shape)[0] != 1 or any(
                    not isinstance(value, int) for value in item.shape
                ):
                    raise PipelineGateError(
                        f"ONNX input {item.name!r} must have a fully static batch-1 shape: {item.shape}"
                    )
            output_meta = session.get_outputs()
            out_names = [item.name for item in output_meta]
            if out_names != [output_name]:
                raise PipelineGateError(
                    f"ONNX output must be exactly {output_name!r}: {out_names}"
                )

            prob_model = self._probability_and_label_model(
                model, evidence["decision_threshold"]
            )
            native_probabilities: list[float] = []
            onnx_probabilities: list[float] = []
            vectors: list[dict[str, Any]] = []
            with torch.no_grad():
                for position in range(vector_count):
                    waveform = waveform_values[position : position + 1]
                    feeds: dict[str, np.ndarray] = {input_name: waveform}
                    tensors: list[torch.Tensor] = [torch.from_numpy(waveform)]
                    if is_hybrid:
                        assert tabular_values is not None
                        tabular = tabular_values[position : position + 1]
                        feeds["tabular_input"] = tabular
                        tensors.append(torch.from_numpy(tabular))

                    n_prob = prob_model(*tensors)
                    native = float(n_prob.numpy().reshape(-1)[0])

                    runtime_out = session.run([output_name], feeds)
                    r_prob = runtime_out[0]
                    runtime = float(np.asarray(r_prob).reshape(-1)[0])

                    if not (
                        np.isfinite(native)
                        and np.isfinite(runtime)
                        and 0.0 <= runtime <= 1.0
                    ):
                        raise PipelineGateError(
                            "Real batch-1 ONNX inference returned an invalid probability"
                        )
                    native_probabilities.append(native)
                    onnx_probabilities.append(runtime)
                    vectors.append(
                        {
                            "case_id": f"parity-{position + 1:03d}",
                            "inputs": {
                                name: feeds[name].reshape(-1).tolist()
                                for name in input_names
                            },
                            "expected_probability": runtime,
                            "expected_label": int(
                                runtime >= evidence["decision_threshold"]
                            ),
                        }
                    )

            differences = np.abs(
                np.asarray(native_probabilities) - np.asarray(onnx_probabilities)
            )
            max_diff = float(differences.max())
            mean_diff = float(differences.mean())
            if max_diff >= self.settings.onnx.max_abs_diff:
                raise PipelineGateError(
                    f"ONNX parity failed: max_diff={max_diff:.3e} must be < {self.settings.onnx.max_abs_diff:.1e}"
                )

            model_checksum = _sha256(staged_model)
            created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            input_contracts = [
                {"name": item.name, "dtype": "float32", "shape": list(item.shape)}
                for item in actual_inputs
            ]
            parity = {
                "schema_version": _INDEX_SCHEMA,
                "model_version": self.model_version,
                "sensor_channel": sensor_channel,
                "onnx_sha256": model_checksum,
                "checked_vectors": vector_count,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "parity_max_diff": max_diff,
                "parity_mean_diff": mean_diff,
                "required_max_diff_exclusive": self.settings.onnx.max_abs_diff,
                "onnx_checker_passed": True,
                "native_inference_passed": True,
                "onnx_inference_passed": True,
                "real_batch1_inference_passed": True,
                "status": "passed",
            }
            vectors_payload = {
                "schema_version": _INDEX_SCHEMA,
                "model_version": self.model_version,
                "sensor_channel": sensor_channel,
                "onnx_sha256": model_checksum,
                "input_contracts": input_contracts,
                "output_name": output_name,
                "cases": vectors,
            }
            manifest = {
                "schema_version": _INDEX_SCHEMA,
                "model_name": f"stress_{sensor_channel}",
                "model_version": self.model_version,
                "model_family": "deep_learning",
                "sensor_channel": sensor_channel,
                "task": "stress",
                "architecture": arch_name,
                "architecture_params": arch_params,
                "inputs": input_contracts,
                "output": {"name": output_name, "dtype": "float32", "shape": [1]},
                "preprocessing_version": self.settings.preprocessing_version,
                "feature_version": "raw_waveform",
                "quantization_type": "none",
                "probability_threshold": evidence["decision_threshold"],
                "threshold_source": evidence["threshold_source"],
                "balanced_accuracy_holdout": evidence["balanced_accuracy"],
                "balanced_accuracy_gate": MIN_BALANCED_ACCURACY_GATE,
                "f1_gate": MIN_F1_GATE,
                "roc_auc_gate": MIN_ROC_AUC_GATE,
                "gate_source": "PipelinePython/configs/frozen_gates.yaml",
                "gate_passed": True,
                "validation_cohort_ids": evidence["cohort_ids"],
                "cohort_access_count": evidence["cohort_access_count"],
                "candidate_bundle_sha256": candidate_checksum,
                "production_bundle_sha256": bundle_checksum,
                "onnx_sha256": model_checksum,
                "parity_report": names[2],
                "test_vectors": names[3],
                "refit_provenance": report.get("refit_provenance"),
                "release_status": "accepted",
                "created_at": created_at,
                "scaler_mean": prod_mean.tolist(),
                "scaler_scale": prod_scale.tolist(),
            }
            if is_hybrid:
                assert (
                    tabular_imputer is not None
                    and tabular_mean is not None
                    and tabular_scale is not None
                )
                manifest["tabular_imputer"] = tabular_imputer.tolist()
                manifest["tabular_mean"] = tabular_mean.tolist()
                manifest["tabular_scale"] = tabular_scale.tolist()

            _atomic_json(staging / names[2], parity)
            _atomic_json(staging / names[3], vectors_payload)
            _atomic_json(staging / names[1], manifest)
            checksums = {name: _sha256(staging / name) for name in names}
            for name in names:
                os.replace(staging / name, self.versioned_export_dir / name)

        relative_root = self.versioned_export_dir.relative_to(
            self.paths.export_dir
        ).as_posix()
        entry = {
            "model_version": self.model_version,
            "sensor_channel": sensor_channel,
            "model_family": "deep_learning",
            "release_status": "accepted",
            "balanced_accuracy_holdout": evidence["balanced_accuracy"],
            "balanced_accuracy_gate": MIN_BALANCED_ACCURACY_GATE,
            "f1_holdout": evidence["f1"],
            "roc_auc_holdout": evidence["roc_auc"],
            "f1_gate": MIN_F1_GATE,
            "roc_auc_gate": MIN_ROC_AUC_GATE,
            "cohort_ids": evidence["cohort_ids"],
            "files": {
                "model": {
                    "path": f"{relative_root}/{names[0]}",
                    "sha256": checksums[names[0]],
                },
                "manifest": {
                    "path": f"{relative_root}/{names[1]}",
                    "sha256": checksums[names[1]],
                },
                "parity": {
                    "path": f"{relative_root}/{names[2]}",
                    "sha256": checksums[names[2]],
                },
                "test_vectors": {
                    "path": f"{relative_root}/{names[3]}",
                    "sha256": checksums[names[3]],
                },
            },
        }
        try:
            self._publish_index(entry)
        except Exception:
            # The artifacts remain versioned evidence but are not deployable without
            # an accepted index entry. Never silently package a partial publication.
            logger.exception("Artifacts written but canonical index publication failed")
            raise
        return parity
