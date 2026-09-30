"""Strict FP32 ONNX export and local parity validation for candidate bundles."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
import tempfile
import time
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .contracts import (
    OnnxParityResult,
    assert_feature_tensor,
    assert_stress_probability,
)
from .performance_policy import PerformancePolicy, policy_from_training_config
from .settings import TRAINING_CHANNELS, PipelineConfig, pipeline_paths_from_config


EXPORT_CHANNELS = TRAINING_CHANNELS
PPG_SOURCE_RATE_HZ: dict[str, float] = {
    "ppg_green": 25.0,
    "ppg_ir": 100.0,
    "ppg_red": 100.0,
}


class OnnxExportMode(str, Enum):
    PRODUCTION = "production"
    RESEARCH = "research"
    INTERNAL = "internal"


def _ppg_filter_contract(sensor_channel: str) -> dict[str, Any]:
    """Return the single manifest-owned runtime filter contract."""

    if sensor_channel == "ppg_green":
        high_cut_hz = 5.0
    elif sensor_channel in {"ppg_ir", "ppg_red"}:
        high_cut_hz = 8.0
    else:
        raise RuntimeError(f"unsupported PPG filter channel: {sensor_channel}")
    contract = {
        "algorithm": "butterworth_sosfiltfilt",
        "order": 4,
        "low_cut_hz": 0.5,
        "high_cut_hz": high_cut_hz,
        "zero_phase": True,
        "pad_policy": "scipy_odd_sosfiltfilt_default",
        "design_version": "sos_zpk_bilinear_v1",
    }

    def canonical_number(value: float) -> str:
        return f"{value:.15f}".rstrip("0").rstrip(".")

    canonical = "\n".join(
        (
            contract["algorithm"],
            str(contract["order"]),
            canonical_number(contract["low_cut_hz"]),
            canonical_number(contract["high_cut_hz"]),
            str(contract["zero_phase"]).lower(),
            contract["pad_policy"],
            contract["design_version"],
        )
    )
    contract["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON document atomically within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash a manifest without its self-referential digest field."""

    canonical = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_window_contract(
    candidate_manifest: dict[str, Any], *, analysis_seconds: int
) -> dict[str, int]:
    """Resolve the selected channel window without reverting to a global default."""

    window_seconds = candidate_manifest.get(
        "window_seconds", candidate_manifest.get("selected_window_seconds")
    )
    stride_seconds = candidate_manifest.get(
        "stride_seconds", candidate_manifest.get("selected_stride_seconds")
    )
    if window_seconds is None or stride_seconds is None:
        raise RuntimeError("candidate manifest is missing selected window/stride")
    window_seconds = int(window_seconds)
    stride_seconds = int(stride_seconds)
    if not 0 < window_seconds <= analysis_seconds:
        raise RuntimeError("candidate window_seconds is outside the analysis core")
    if not 0 < stride_seconds <= window_seconds:
        raise RuntimeError("candidate stride_seconds is invalid")
    return {
        "analysis_seconds": int(analysis_seconds),
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "expected_windows_per_session": (
            (int(analysis_seconds) - window_seconds) // stride_seconds + 1
        ),
    }


def _positive_probability_output(model: onnx.ModelProto) -> str:
    candidates = [
        output.name
        for output in model.graph.output
        if output.type.tensor_type.elem_type in {TensorProto.FLOAT, TensorProto.DOUBLE}
        and len(output.type.tensor_type.shape.dim) == 2
    ]
    if not candidates:
        raise RuntimeError("Converted classifier has no rank-2 probability output")
    return candidates[-1]


def _enforce_public_io(
    model: onnx.ModelProto,
    probability_matrix: str,
    *,
    batch_size: int | None = None,
) -> onnx.ModelProto:
    index_name = "positive_class_index"
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), name=index_name)
    )
    model.graph.node.append(
        helper.make_node(
            "Gather",
            inputs=[probability_matrix, index_name],
            outputs=["stress_probability"],
            axis=1,
            name="select_task_probability",
        )
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(
            "stress_probability",
            TensorProto.FLOAT,
            [batch_size],
        )
    )
    onnx.checker.check_model(model)
    return model


def _convert_logistic_pipeline_to_standard_onnx(
    estimator: Any,
    *,
    input_name: str,
    n_features: int,
    opset: int,
) -> onnx.ModelProto:
    """Convert binary logistic regression using portable tensor operators.

    ``ai.onnx.ml.LinearClassifier`` is mathematically suitable, but its
    device behaviour is not consistent with the host runtime for the current
    Android target.  This graph preserves the fitted imputer, scaler, and
    classifier parameters while using operators implemented by the regular
    ONNX execution path.
    """

    classifier = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if classifier.__class__.__name__ != "LogisticRegression":
        raise RuntimeError("standard_logistic_converter_requires_logistic_regression")
    classes = np.asarray(getattr(classifier, "classes_", []))
    coefficients = np.asarray(getattr(classifier, "coef_", []), dtype=np.float32)
    intercept = np.asarray(getattr(classifier, "intercept_", []), dtype=np.float32)
    if classes.tolist() != [0, 1] or coefficients.shape != (1, n_features):
        raise RuntimeError("standard_logistic_converter_requires_binary_classifier")
    if (
        intercept.shape != (1,)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
    ):
        raise RuntimeError("standard_logistic_converter_requires_finite_parameters")

    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    current = input_name

    def add_initializer(name: str, values: np.ndarray) -> None:
        initializers.append(
            numpy_helper.from_array(values.astype(np.float32), name=name)
        )

    named_steps = dict(getattr(estimator, "steps", ()))
    imputer = named_steps.get("imputer")
    if imputer is not None:
        statistics = np.asarray(getattr(imputer, "statistics_", []), dtype=np.float32)
        if statistics.shape != (n_features,) or not np.isfinite(statistics).all():
            raise RuntimeError("standard_logistic_converter_requires_finite_imputer")
        add_initializer("imputer_statistics", statistics.reshape(1, n_features))
        nodes.append(
            helper.make_node(
                "IsNaN",
                inputs=[current],
                outputs=["input_is_nan"],
                name="input_is_nan",
            )
        )
        nodes.append(
            helper.make_node(
                "Where",
                inputs=["input_is_nan", "imputer_statistics", current],
                outputs=["imputed_features"],
                name="impute_missing_features",
            )
        )
        current = "imputed_features"

    scaler = named_steps.get("scaler")
    if scaler is not None:
        mean = np.asarray(getattr(scaler, "mean_", []), dtype=np.float32)
        scale = np.asarray(getattr(scaler, "scale_", []), dtype=np.float32)
        if mean.shape != (n_features,) or scale.shape != (n_features,):
            raise RuntimeError("standard_logistic_converter_requires_standard_scaler")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0.0).any()
        ):
            raise RuntimeError("standard_logistic_converter_requires_finite_scaler")
        add_initializer("scaler_mean", mean.reshape(1, n_features))
        add_initializer("scaler_inverse_scale", (1.0 / scale).reshape(1, n_features))
        nodes.extend(
            (
                helper.make_node(
                    "Sub",
                    inputs=[current, "scaler_mean"],
                    outputs=["centered_features"],
                    name="center_features",
                ),
                helper.make_node(
                    "Mul",
                    inputs=["centered_features", "scaler_inverse_scale"],
                    outputs=["scaled_features"],
                    name="scale_features",
                ),
            )
        )
        current = "scaled_features"

    add_initializer("classifier_coefficients", coefficients.reshape(n_features, 1))
    add_initializer("classifier_intercept", intercept.reshape(1, 1))
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1], dtype=np.int64), name="probability_shape"
        )
    )
    nodes.extend(
        (
            helper.make_node(
                "MatMul",
                inputs=[current, "classifier_coefficients"],
                outputs=["decision_without_intercept"],
                name="linear_decision",
            ),
            helper.make_node(
                "Add",
                inputs=["decision_without_intercept", "classifier_intercept"],
                outputs=["positive_logit"],
                name="add_intercept",
            ),
            helper.make_node(
                "Sigmoid",
                inputs=["positive_logit"],
                outputs=["positive_probability_matrix"],
                name="positive_class_probability",
            ),
            helper.make_node(
                "Clip",
                inputs=[
                    "positive_probability_matrix",
                    "probability_min",
                    "probability_max",
                ],
                outputs=["bounded_probability_matrix"],
                name="bound_probability",
            ),
            helper.make_node(
                "Reshape",
                inputs=["bounded_probability_matrix", "probability_shape"],
                outputs=["stress_probability"],
                name="stress_probability",
            ),
        )
    )
    add_initializer("probability_min", np.asarray(0.0, dtype=np.float32))
    add_initializer("probability_max", np.asarray(1.0, dtype=np.float32))
    graph = helper.make_graph(
        nodes,
        "portable_logistic_regression",
        [
            helper.make_tensor_value_info(
                input_name,
                TensorProto.FLOAT,
                [1, n_features],
            )
        ],
        [helper.make_tensor_value_info("stress_probability", TensorProto.FLOAT, [1])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="PipelinePython",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    # Keep the graph readable by the pinned host/runtime combination.  The
    # graph uses opset 17 operators but needs no newer IR features.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _parity_payload(parity: OnnxParityResult) -> dict[str, Any]:
    """Serialize parity evidence with names used by current and legacy readers."""

    payload = asdict(parity)
    payload.update(
        {
            "parity_max_diff": parity.max_abs_diff,
            "parity_mean_diff": parity.mean_abs_diff,
            "onnx_checker_passed": True,
            "real_batch1_inference_passed": True,
        }
    )
    return payload


def _android_contract_payload(manifest: dict[str, Any]) -> str:
    return "\n".join(
        (
            str(manifest["model_version"]),
            str(manifest["preprocessing_version"]),
            str(manifest["feature_version"]),
            str(manifest["sensor_channel"]),
            str(int(manifest["analysis_seconds"])),
            str(int(manifest["window_seconds"])),
            str(int(manifest["stride_seconds"])),
            str(int(manifest["expected_windows_per_session"])),
            repr(float(manifest["source_rate_hz"])),
            repr(float(manifest["model_rate_hz"])),
            json.dumps(
                manifest.get("filter_contract", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            str(
                manifest.get(
                    "android_ready", manifest.get("internal_android_status", "")
                )
            ),
            ",".join(str(name) for name in manifest["feature_order"]),
            str(float(manifest["probability_threshold"])),
            str(manifest["onnx_sha256"]),
            str(manifest["onnx_input"]["name"]),
            str(manifest["onnx_output"]["name"]),
            str(manifest.get("performance_policy_version", "")),
            str(manifest.get("performance_policy_sha256", "")),
            str(manifest.get("performance_tier", "")),
            str(manifest.get("deployment_scope", "")),
            str(manifest.get("research_disclaimer", "")),
            str(manifest.get("device_parity_receipt_sha256", "")),
        )
    )


def _register_xgboost_converter() -> None:
    """Register onnxmltools' XGBoost converter with skl2onnx.

    ``skl2onnx`` deliberately does not ship the XGBoost converter.  The
    registration is explicit so an unsupported converter remains a hard
    export failure instead of silently exporting only part of a pipeline.
    """

    try:
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )
        from onnxmltools.convert.xgboost.shape_calculators.Classifier import (
            calculate_xgboost_classifier_output_shapes,
        )
        from xgboost import XGBClassifier
        from skl2onnx import update_registered_converter
    except ImportError as exc:
        raise RuntimeError(
            "unsupported_converter:XGBClassifier:onnxmltools is required"
        ) from exc
    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_xgboost_classifier_output_shapes,
        convert_xgboost,
    )


class OnnxBundleExporter:
    def __init__(
        self,
        project_root: Path,
        config: PipelineConfig,
        *,
        model_version: str | None = None,
        models_root: Path | None = None,
        gold_path: Path | None = None,
        export_root: Path | None = None,
        mode: OnnxExportMode | str = OnnxExportMode.PRODUCTION,
    ) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.paths = pipeline_paths_from_config(self.root, config)
        self.model_version = model_version or config.model_version
        self.gold_path = (gold_path or self.paths.gold_features_path).resolve()
        self.models_dir = (models_root or self.paths.models_dir / "stress").resolve()
        self.export_dir = (export_root or self.paths.export_dir).resolve()
        self.mode = OnnxExportMode(mode)
        self._strict_run_scope = bool(config.artifacts.get("run_id"))
        if self._strict_run_scope:
            try:
                self.gold_path.relative_to(self.paths.gold_dir.resolve())
            except ValueError as exc:
                raise ValueError("onnx_gold_source_must_be_current_run") from exc
        self.performance_policy: PerformancePolicy = policy_from_training_config(
            config.training
        )

    def _candidate_paths(self, sensor_channel: str) -> tuple[Path, Path, Path]:
        bundle_dir = self.models_dir / sensor_channel / self.model_version
        if self.mode in {OnnxExportMode.RESEARCH, OnnxExportMode.INTERNAL}:
            bundle_dir /= "development_candidate"
        return (
            bundle_dir,
            bundle_dir / "candidate_manifest.json",
            bundle_dir / "model.joblib",
        )

    def _published_export_dir(self, sensor_channel: str) -> Path:
        root = {
            OnnxExportMode.RESEARCH: self.export_dir / "research",
            OnnxExportMode.INTERNAL: self.export_dir / "internal",
            OnnxExportMode.PRODUCTION: self.export_dir,
        }[self.mode]
        return root / self.model_version / sensor_channel

    def _load_candidate_manifest(
        self, sensor_channel: str
    ) -> tuple[dict[str, Any], Path]:
        self._validate_channel(sensor_channel)
        _, manifest_path, _ = self._candidate_paths(sensor_channel)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"candidate gate has not produced {sensor_channel}: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_run_id = str(self.config.artifacts.get("run_id", ""))
        if self._strict_run_scope and manifest.get("run_id") != expected_run_id:
            raise RuntimeError("candidate_run_id_mismatch")
        return manifest, manifest_path

    def _load_development_report(self, sensor_channel: str) -> dict[str, Any]:
        """Load the run-scoped report, then the immutable canonical report.

        Export is intentionally bound to a training run, but reports created by
        the canonical trainer are stored below ``reports/model_eval`` rather
        than the per-run pipeline report directory.  The fallback is read-only
        and does not select a different model version.
        """

        reports_root = Path(self.config.artifacts.get("reports_dir", "reports"))
        if not reports_root.is_absolute():
            reports_root = self.root / reports_root
        candidates = (
            (
                reports_root
                / "model_eval"
                / self.model_version
                / f"{sensor_channel}_full.json",
            )
            if self._strict_run_scope
            else (
                reports_root
                / "model_eval"
                / self.model_version
                / f"{sensor_channel}_full.json",
                self.root
                / "reports"
                / "model_eval"
                / self.model_version
                / f"{sensor_channel}_full.json",
            )
        )
        for report_path in candidates:
            if report_path.is_file():
                return json.loads(report_path.read_text(encoding="utf-8"))
        return {}

    def _conservative_assessment(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Assess the weaker value of development and internal-test evidence.

        A channel is a release candidate only when every gated metric clears
        the minimum on both disjoint cohorts.  This avoids promoting IR as
        ``preferred`` merely because its internal-test result exceeds the
        preferred gate while its development result does not.
        """

        development = dict(manifest.get("development_metrics", {}))
        holdout = dict(manifest.get("holdout_metrics", {}))
        conservative_metrics = {
            metric: min(float(development[metric]), float(holdout[metric]))
            for metric in ("balanced_accuracy", "f1", "roc_auc")
        }
        assessment = self.performance_policy.assess(conservative_metrics)
        return {
            "conservative_metrics": conservative_metrics,
            "assessment": assessment.as_dict(),
        }

    def _window_contract(
        self, manifest: dict[str, Any], sensor_channel: str
    ) -> dict[str, int]:
        source = dict(manifest)
        report = self._load_development_report(sensor_channel)
        source.setdefault(
            "selected_window_seconds", report.get("selected_window_seconds")
        )
        source.setdefault(
            "selected_stride_seconds", report.get("selected_stride_seconds")
        )
        return _manifest_window_contract(
            source,
            analysis_seconds=self.config.window.analysis_phase_duration_seconds,
        )

    def _load_native_estimator(self, manifest: dict[str, Any]) -> tuple[Any, Path]:
        native_value = manifest.get("native_model_path")
        if not native_value:
            raise RuntimeError("candidate manifest is missing native_model_path")
        native_path = (self.root / str(native_value)).resolve()
        if self.root not in native_path.parents or not native_path.is_file():
            raise FileNotFoundError(
                f"candidate native model is unavailable: {native_path}"
            )
        if _sha256(native_path) != manifest.get("model_sha256"):
            raise RuntimeError("native model hash mismatch")
        return joblib.load(native_path), native_path

    def _convert_estimator(
        self,
        estimator: Any,
        feature_order: tuple[str, ...],
        finite_features: np.ndarray,
    ) -> onnx.ModelProto:
        """Convert a complete estimator or fail closed on unsupported converters."""

        classifier = (
            estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        )
        portable_logistic = classifier.__class__.__name__ == "LogisticRegression"
        try:
            if portable_logistic:
                converted = _convert_logistic_pipeline_to_standard_onnx(
                    estimator,
                    input_name=self.config.onnx.input_name,
                    n_features=len(feature_order),
                    opset=self.config.onnx.opset,
                )
            elif classifier.__class__.__name__ == "XGBClassifier":
                from onnxmltools import convert_xgboost
                from onnxmltools.convert.common.data_types import (
                    FloatTensorType as OnnxFloatTensorType,
                )

                if not np.isfinite(finite_features).all():
                    raise RuntimeError(
                        "XGBoost ONNX export requires finite Gold feature inputs"
                    )
                converted = convert_xgboost(
                    classifier,
                    initial_types=[
                        (
                            self.config.onnx.input_name,
                            OnnxFloatTensorType(
                                [self.config.onnx.static_batch_size, len(feature_order)]
                            ),
                        )
                    ],
                    target_opset=15,
                )
                default_import = next(
                    (item for item in converted.opset_import if item.domain == ""),
                    None,
                )
                if default_import is None:
                    default_import = converted.opset_import.add()
                default_import.version = self.config.onnx.opset
            else:
                converted = convert_sklearn(
                    estimator,
                    initial_types=[
                        (
                            self.config.onnx.input_name,
                            FloatTensorType(
                                [self.config.onnx.static_batch_size, len(feature_order)]
                            ),
                        )
                    ],
                    target_opset=self.config.onnx.opset,
                    options={id(classifier): {"zipmap": False}},
                )
            onnx.checker.check_model(converted)
        except Exception as exc:
            raise RuntimeError(
                f"unsupported_converter:{type(classifier).__name__}:{exc}"
            ) from exc
        if not portable_logistic:
            converted = _enforce_public_io(
                converted,
                _positive_probability_output(converted),
                batch_size=self.config.onnx.static_batch_size,
            )
        onnx.checker.check_model(converted)
        return converted

    def _validate_channel(self, sensor_channel: str) -> None:
        if sensor_channel not in EXPORT_CHANNELS:
            raise ValueError(
                f"ONNX publication is restricted to {EXPORT_CHANNELS}; "
                f"received {sensor_channel!r}"
            )

    def _candidate_gold_path(self, manifest: dict[str, Any]) -> Path:
        """Resolve the exact run-scoped Gold variant used to fit a candidate."""

        if not self._strict_run_scope:
            return self.gold_path
        gold_contract = manifest.get("gold_contract")
        if not isinstance(gold_contract, dict):
            raise RuntimeError("candidate_gold_contract_missing")
        declared_path = gold_contract.get("features_path")
        if not isinstance(declared_path, str) or not declared_path.strip():
            raise RuntimeError("candidate_gold_path_missing")
        candidate_path = Path(declared_path)
        if not candidate_path.is_absolute():
            candidate_path = self.root / candidate_path
        candidate_path = candidate_path.resolve()
        try:
            candidate_path.relative_to(self.paths.gold_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("candidate_gold_path_outside_current_run") from exc
        if not candidate_path.is_file():
            raise RuntimeError("candidate_gold_features_missing")
        if gold_contract.get("features_sha256") != _sha256(candidate_path):
            raise RuntimeError("candidate_gold_contract_mismatch")
        return candidate_path

    def _validate_development_manifest(
        self, manifest: dict[str, Any], sensor_channel: str
    ) -> None:
        self._validate_channel(sensor_channel)
        if manifest.get("model_version") != self.model_version:
            raise RuntimeError("candidate model_version does not match requested run")
        if manifest.get("sensor_channel") != sensor_channel:
            raise RuntimeError("candidate sensor_channel does not match export target")
        configured_predictors = tuple(self.config.features.predictor_default)
        raw_feature_order = manifest.get("feature_order")
        # Private non-scoped callers from the legacy test boundary may omit
        # feature_order. Canonical run-scoped export remains strict below.
        feature_order = tuple(
            str(name)
            for name in (
                raw_feature_order
                if raw_feature_order is not None or self._strict_run_scope
                else configured_predictors
            )
        )
        allowed_predictors = set(configured_predictors) | set(
            self.config.features.promoted_experimental
        )
        if not feature_order or len(set(feature_order)) != len(feature_order):
            raise RuntimeError("candidate feature_order is empty or duplicated")
        if not set(feature_order) <= allowed_predictors:
            raise RuntimeError("candidate feature_order contains forbidden predictors")
        if manifest.get("canonical_predictors") not in (
            None,
            list(configured_predictors),
        ):
            raise RuntimeError("candidate canonical predictor contract mismatch")
        if manifest.get("predictor_columns") not in (None, list(feature_order)):
            raise RuntimeError("candidate predictor_columns mismatch")
        if self._strict_run_scope:
            if manifest.get("run_id") != self.config.artifacts.get("run_id"):
                raise RuntimeError("candidate_run_id_mismatch")
            expected_status = (
                "development_candidate"
                if self.mode in {OnnxExportMode.RESEARCH, OnnxExportMode.INTERNAL}
                else "release_candidate"
            )
            if manifest.get("release_status") != expected_status:
                raise RuntimeError("candidate_release_status_is_not_frozen")
            contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
            if not contract_path.is_file() or manifest.get(
                "run_contract_sha256"
            ) != _sha256(contract_path):
                raise RuntimeError("candidate_run_contract_fingerprint_mismatch")
            run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            source_tree_sha256 = str(run_contract.get("source_tree_sha256", ""))
            if not source_tree_sha256 or source_tree_sha256.startswith("UNAVAILABLE:"):
                raise RuntimeError("source_tree_fingerprint_unavailable")
            if manifest.get("source_tree_sha256") != source_tree_sha256:
                raise RuntimeError("candidate_source_tree_fingerprint_mismatch")
            gold_contract = manifest.get("gold_contract", {})
            self._candidate_gold_path(manifest)
            gold_manifest_path = self.paths.gold_dir / "manifest_v3.json"
            if not gold_manifest_path.is_file():
                raise RuntimeError("current_run_gold_manifest_missing")
            gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
            if gold_manifest.get("run_id") != self.config.artifacts.get("run_id"):
                raise RuntimeError("current_run_gold_manifest_run_id_mismatch")
            if gold_contract.get("manifest_sha256") != _sha256(gold_manifest_path):
                raise RuntimeError("candidate_gold_manifest_fingerprint_mismatch")
        metrics = manifest.get("development_metrics", {})
        assessment = self.performance_policy.assess(metrics)
        if not assessment.passes_minimum:
            raise RuntimeError(
                f"ONNX export blocked for {sensor_channel}: development metrics "
                f"failed minimum policy {self.performance_policy.version}: "
                f"{', '.join(assessment.failing_minimum_metrics)}"
            )
        expected = [f"ID-{index:02d}" for index in range(1, 29)]
        declared_expected = sorted(
            manifest.get(
                "expected_development_participants",
                manifest.get("development_participants", []),
            )
        )
        if declared_expected != expected:
            raise RuntimeError("ONNX export blocked: invalid development provenance")
        represented_raw = manifest.get("represented_development_participants")
        if not isinstance(represented_raw, list):
            raise RuntimeError(
                "ONNX export blocked: represented development provenance missing"
            )
        represented = sorted(str(value) for value in represented_raw)
        if (
            len(represented) != len(set(represented))
            or not set(represented).issubset(expected)
            or len(represented) < self.config.training.minimum_development_participants
        ):
            raise RuntimeError(
                "ONNX export blocked: represented development provenance invalid"
            )
        declared_missing = sorted(
            str(value) for value in manifest.get("missing_development_participants", [])
        )
        if declared_missing != sorted(set(expected) - set(represented)):
            raise RuntimeError(
                "ONNX export blocked: missing development provenance mismatch"
            )

    def _validate_holdout_manifest(
        self, manifest: dict[str, Any], sensor_channel: str
    ) -> None:
        self._validate_development_manifest(manifest, sensor_channel)
        metrics = manifest.get("holdout_metrics", {})
        assessment = self.performance_policy.assess(metrics)
        if not assessment.passes_minimum:
            raise RuntimeError(
                f"ONNX export blocked for {sensor_channel}: internal-test metrics "
                f"failed minimum policy {self.performance_policy.version}: "
                f"{', '.join(assessment.failing_minimum_metrics)}"
            )
        if sorted(manifest.get("holdout_participants", [])) != [
            f"ID-{index:02d}" for index in range(29, 33)
        ]:
            raise RuntimeError("ONNX export blocked: invalid holdout provenance")

    def _internal_test_evidence(
        self, manifest: dict[str, Any], sensor_channel: str
    ) -> dict[str, Any]:
        """Read the immutable one-time internal-test result for this candidate."""

        from .internal_publish import (
            INTERNAL_EVIDENCE_SCHEMA,
            internal_test_evidence_path,
        )

        self._validate_development_manifest(manifest, sensor_channel)
        path = internal_test_evidence_path(self.paths, self.config, sensor_channel)
        if not path.is_file():
            raise RuntimeError("internal_test_evidence_missing")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise RuntimeError("internal_test_evidence_invalid")
        expected = {
            "schema_version": INTERNAL_EVIDENCE_SCHEMA,
            "validation_scope": "internal_32_participants",
            "claims_policy": "decision_support_only",
            "run_id": str(self.config.artifacts.get("run_id", "")),
            "sensor_channel": sensor_channel,
            "model_version": self.model_version,
            "internal_test_participants": [
                f"ID-{index:02d}" for index in range(29, 33)
            ],
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise RuntimeError("internal_test_evidence_contract_mismatch")
        frozen = evidence.get("frozen_contract")
        if not isinstance(frozen, dict):
            raise RuntimeError("internal_test_frozen_contract_missing")
        if (
            frozen.get("candidate_artifact_fingerprint")
            != manifest.get("candidate_artifact_fingerprint")
            or frozen.get("model_sha256") != manifest.get("model_sha256")
            or frozen.get("calibration_sha256") != manifest.get("calibration_sha256")
            or frozen.get("feature_order") != manifest.get("feature_order")
            or frozen.get("run_contract_sha256") != manifest.get("run_contract_sha256")
        ):
            raise RuntimeError("internal_test_candidate_fingerprint_mismatch")
        if evidence.get("internal_test_status") != "GO":
            raise RuntimeError(
                "internal_test_gate_failed:"
                + str(evidence.get("reason", evidence.get("internal_test_status")))
            )
        metrics = evidence.get("internal_test_metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("internal_test_metrics_missing")
        assessment = self.performance_policy.assess(metrics)
        if not assessment.passes_minimum:
            raise RuntimeError("internal_test_metrics_below_minimum")
        return evidence

    def _internal_conservative_assessment(
        self, manifest: dict[str, Any], evidence: dict[str, Any]
    ) -> dict[str, Any]:
        development = dict(manifest.get("development_metrics", {}))
        internal = dict(evidence.get("internal_test_metrics", {}))
        conservative = {
            metric: min(float(development[metric]), float(internal[metric]))
            for metric in ("balanced_accuracy", "f1", "roc_auc")
        }
        return {
            "conservative_metrics": conservative,
            "assessment": self.performance_policy.assess(conservative).as_dict(),
        }

    def _validate_manifest(self, manifest: dict[str, Any], sensor_channel: str) -> None:
        """Compatibility boundary for callers requiring full release validation."""

        self._validate_holdout_manifest(manifest, sensor_channel)

    def _validation_channel_decision(self, sensor_channel: str) -> dict[str, Any]:
        recommendation_path = (
            self.paths.validation_dir
            / "hrv_agreement"
            / "physiological_validation.json"
        )
        if not recommendation_path.is_file():
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "reason": "channel_recommendation_missing",
                "source": str(recommendation_path),
            }
        payload = json.loads(recommendation_path.read_text(encoding="utf-8"))
        expected_run_id = str(self.config.artifacts.get("run_id", ""))
        if expected_run_id and str(payload.get("run_id", "")) != expected_run_id:
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "reason": "physiological_validation_run_mismatch",
                "source": str(recommendation_path),
            }
        channels = payload.get("channels", {})
        channel = channels.get(sensor_channel) if isinstance(channels, dict) else None
        if channel is None:
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "reason": "channel_recommendation_row_missing",
                "source": str(recommendation_path),
            }
        return {
            **channel,
            "decision": channel.get("decision", "INSUFFICIENT_EVIDENCE"),
            "recommended_channel": payload.get("recommended_channel"),
            "source": str(recommendation_path),
            "source_sha256": _sha256(recommendation_path),
        }

    def preflight_sensor(self, sensor_channel: str) -> dict[str, Any]:
        """Prove converter support on development data before ONNX promotion."""

        manifest, manifest_path = self._load_candidate_manifest(sensor_channel)
        self._validate_development_manifest(manifest, sensor_channel)
        candidate_gold_path = self._candidate_gold_path(manifest)
        estimator, native_path = self._load_native_estimator(manifest)
        feature_order = tuple(str(name) for name in manifest.get("feature_order", []))
        if not feature_order:
            raise RuntimeError("candidate manifest has an empty feature_order")
        gold = pd.read_parquet(candidate_gold_path)
        development_ids = {f"ID-{index:02d}" for index in range(1, 29)}
        development = gold[
            (gold["sensor_channel"].astype(str) == sensor_channel)
            & gold["participant_id"].astype(str).isin(development_ids)
        ].sort_values(["participant_id", "window_start_utc"])
        missing = sorted(set(feature_order) - set(development.columns))
        if missing:
            raise RuntimeError(f"Gold is missing candidate features: {missing}")
        if development.empty:
            raise RuntimeError("converter preflight has no development rows")
        tensor = assert_feature_tensor(
            development.loc[:, feature_order].head(16).to_numpy(),
            n_features=len(feature_order),
        )
        report = self.preflight_estimator(
            estimator=estimator,
            feature_order=feature_order,
            development_tensor=tensor,
            sensor_channel=sensor_channel,
        )
        report.update(
            {
                "native_model_sha256": _sha256(native_path),
                "candidate_manifest_sha256": _sha256(manifest_path),
                "gold_sha256": _sha256(candidate_gold_path),
            }
        )
        destination = (
            self.export_dir
            / self.model_version
            / sensor_channel
            / "converter_preflight.json"
        )
        if not self._strict_run_scope:
            _atomic_write_json(destination, report)
        return report

    def preflight_estimator(
        self,
        *,
        estimator: Any,
        feature_order: tuple[str, ...],
        development_tensor: np.ndarray,
        sensor_channel: str,
    ) -> dict[str, Any]:
        """Validate one frozen development estimator without touching holdout data."""

        self._validate_channel(sensor_channel)
        if not feature_order:
            raise RuntimeError("converter preflight requires a non-empty feature_order")
        tensor = assert_feature_tensor(
            development_tensor,
            n_features=len(feature_order),
        )
        converted = self._convert_estimator(estimator, feature_order, tensor)
        session = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        native = assert_stress_probability(
            estimator.predict_proba(tensor)[:, 1], batch_size=len(tensor)
        )
        exported = np.asarray(
            [
                session.run(
                    [self.config.onnx.output_name],
                    {self.config.onnx.input_name: row.reshape(1, -1)},
                )[0]
                for row in tensor
            ],
            dtype=np.float32,
        ).reshape(-1)
        max_abs_diff = float(
            np.max(np.abs(native.astype(np.float64) - exported.astype(np.float64)))
        )
        if max_abs_diff >= self.config.onnx.max_abs_diff:
            raise RuntimeError(
                "unsupported_converter:development_preflight_parity:"
                f"max_abs_diff={max_abs_diff}"
            )
        classifier = (
            estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        )
        report = {
            "schema_version": "1.0.0",
            "status": "passed",
            "sensor_channel": sensor_channel,
            "model_version": self.model_version,
            "model_family": type(classifier).__name__,
            "checked_development_rows": int(len(tensor)),
            "max_abs_diff": max_abs_diff,
            "max_abs_diff_gate": self.config.onnx.max_abs_diff,
            "onnx_checker_passed": True,
        }
        return report

    def export_sensor(self, sensor_channel: str) -> dict[str, Any]:
        physiological = self._validation_channel_decision(sensor_channel)
        preflight = self.preflight_sensor(sensor_channel)
        manifest, manifest_path = self._load_candidate_manifest(sensor_channel)
        development_report = self._load_development_report(sensor_channel)
        internal_evidence: dict[str, Any] | None = None
        if self.mode is OnnxExportMode.RESEARCH:
            self._validate_development_manifest(manifest, sensor_channel)
            release_blocker = str(
                development_report.get("release_status", "not_evaluated")
            )
            if release_blocker == "release_candidate":
                raise RuntimeError("research_export_requires_nonrelease_candidate")
            development_metrics = dict(manifest.get("development_metrics", {}))
            development_assessment = self.performance_policy.assess(development_metrics)
            conservative = {
                "conservative_metrics": development_metrics,
                "assessment": development_assessment.as_dict(),
            }
        elif self.mode is OnnxExportMode.INTERNAL:
            internal_evidence = self._internal_test_evidence(manifest, sensor_channel)
            release_blocker = "internal_test_passed"
            conservative = self._internal_conservative_assessment(
                manifest, internal_evidence
            )
        else:
            self._validate_holdout_manifest(manifest, sensor_channel)
            release_blocker = "release_candidate"
            conservative = self._conservative_assessment(manifest)
        candidate_gold_path = self._candidate_gold_path(manifest)
        estimator, native_path = self._load_native_estimator(manifest)
        gold = pd.read_parquet(candidate_gold_path)
        development_ids = {f"ID-{index:02d}" for index in range(1, 29)}
        development_gold = gold[
            (gold["sensor_channel"].astype(str) == sensor_channel)
            & gold["participant_id"].astype(str).isin(development_ids)
        ].sort_values(["participant_id", "window_start_utc"])
        feature_order = tuple(manifest["feature_order"])
        if development_gold.empty:
            raise RuntimeError(f"Gold has no rows for {sensor_channel}")
        if len(development_gold) < self.config.onnx.min_parity_samples:
            raise RuntimeError(
                "insufficient_development_parity_vectors:"
                f"{len(development_gold)}<{self.config.onnx.min_parity_samples}"
            )
        if "sampling_rate_hz" not in development_gold:
            raise RuntimeError("Gold is missing sampling_rate_hz")
        sampling_rates = pd.to_numeric(
            development_gold["sampling_rate_hz"], errors="coerce"
        )
        if sampling_rates.isna().any() or not np.isfinite(sampling_rates).all():
            raise RuntimeError("Gold sampling_rate_hz contains non-finite values")
        source_rate_hz = float(sampling_rates.median())
        model_rate_hz = float(manifest.get("model_rate_hz", source_rate_hz))
        if source_rate_hz <= 0.0 or model_rate_hz <= 0.0:
            raise RuntimeError("candidate actual sampling rate is invalid")
        window_contract = self._window_contract(manifest, sensor_channel)
        converted = self._convert_estimator(
            estimator,
            feature_order,
            development_gold.loc[:, feature_order].to_numpy(dtype=np.float32),
        )
        published_export_dir = self._published_export_dir(sensor_channel)
        staging_export_dir: Path | None = None
        if self._strict_run_scope:
            if published_export_dir.exists():
                raise RuntimeError(f"onnx_export_collision:{published_export_dir}")
            published_export_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_export_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{sensor_channel}-", dir=published_export_dir.parent
                )
            )
            versioned_export_dir = staging_export_dir
        else:
            versioned_export_dir = published_export_dir
            versioned_export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = versioned_export_dir / f"stress_{sensor_channel}_fp32.onnx"
        onnx.save(converted, onnx_path)

        sample_count = min(
            len(development_gold), max(100, self.config.onnx.min_parity_samples)
        )
        if sample_count < self.config.onnx.min_parity_samples:
            raise RuntimeError("insufficient_development_parity_vectors")
        sample = development_gold.iloc[:sample_count]
        tensor = assert_feature_tensor(
            sample.loc[:, feature_order].to_numpy(),
            n_features=len(feature_order),
        )
        native = assert_stress_probability(
            estimator.predict_proba(tensor)[:, 1],
            batch_size=sample_count,
        )
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        if session.get_inputs()[0].name != self.config.onnx.input_name:
            raise RuntimeError("onnx_input_name_mismatch")
        if session.get_outputs()[0].name != self.config.onnx.output_name:
            raise RuntimeError("onnx_output_name_mismatch")
        exported = np.concatenate(
            [
                assert_stress_probability(
                    session.run(
                        [self.config.onnx.output_name],
                        {self.config.onnx.input_name: tensor[index : index + 1]},
                    )[0],
                    batch_size=self.config.onnx.static_batch_size,
                )
                for index in range(sample_count)
            ]
        )
        difference = np.abs(native.astype(np.float64) - exported.astype(np.float64))
        vectors = [
            {
                "case_id": f"parity-{index + 1:03d}",
                "participant_id": str(sample.iloc[index]["participant_id"]),
                "phase": str(sample.iloc[index].get("phase", "")),
                "window_id": str(sample.iloc[index].get("window_id", "")),
                "input": tensor[index].tolist(),
                "native_probability": float(native[index]),
                "onnx_probability": float(exported[index]),
            }
            for index in range(sample_count)
        ]
        single = tensor[:1]
        timings: list[float] = []
        for _ in range(110):
            started = time.perf_counter_ns()
            session.run(
                [self.config.onnx.output_name],
                {self.config.onnx.input_name: single},
            )
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        latency = float(np.median(timings[10:]))
        model_size = onnx_path.stat().st_size
        parity_passed = (
            sample_count >= self.config.onnx.min_parity_samples
            and float(difference.max()) < self.config.onnx.max_abs_diff
        )
        parity = OnnxParityResult(
            checked_samples=sample_count,
            max_abs_diff=float(difference.max()),
            mean_abs_diff=float(difference.mean()),
            model_size_bytes=model_size,
            batch1_latency_ms=latency,
            passed=parity_passed,
        )
        deployment_gate = (
            parity.passed
            and model_size <= self.config.onnx.max_model_size_bytes
            and latency < self.config.onnx.max_batch1_latency_ms
        )
        deployment_gate = deployment_gate and bool(
            conservative["assessment"]["passes_minimum"]
        )
        filter_contract = _ppg_filter_contract(sensor_channel)
        run_contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
        run_contract_sha256 = (
            _sha256(run_contract_path) if run_contract_path.is_file() else None
        )
        run_contract = (
            json.loads(run_contract_path.read_text(encoding="utf-8"))
            if run_contract_path.is_file()
            else {}
        )
        if self._strict_run_scope and run_contract_sha256 is None:
            if staging_export_dir is not None:
                shutil.rmtree(staging_export_dir, ignore_errors=True)
            raise RuntimeError("run_contract_fingerprint_missing")
        gold_manifest_path = self.paths.gold_dir / "manifest_v3.json"
        if self._strict_run_scope and not gold_manifest_path.is_file():
            if staging_export_dir is not None:
                shutil.rmtree(staging_export_dir, ignore_errors=True)
            raise RuntimeError("gold_manifest_missing_for_onnx_export")
        if self.mode is OnnxExportMode.RESEARCH:
            development_metrics = dict(manifest["development_metrics"])
            evaluation_fields = {
                "schema_version": "onnx-research-v1",
                "artifact_status": "research_candidate",
                "evaluation_scope": "development_only",
                "release_status": release_blocker,
                "gate_passed": deployment_gate,
                "research_gate_passed": deployment_gate,
                "performance_tier": conservative["assessment"]["performance_tier"],
                "deployment_scope": "research",
                "disclaimer_required": True,
                "balanced_accuracy_development": float(
                    development_metrics["balanced_accuracy"]
                ),
                "f1_development": float(development_metrics["f1"]),
                "roc_auc_development": float(development_metrics["roc_auc"]),
                "validation_cohort_ids": list(
                    manifest["represented_development_participants"]
                ),
                "cohort_access_count": 0,
                "holdout_evaluated": False,
                "clinical_validation_status": "external_validation_required",
                "acceptance_metrics": {
                    "development": development_metrics,
                    "onnx_parity": _parity_payload(parity),
                },
            }
        elif self.mode is OnnxExportMode.INTERNAL:
            if internal_evidence is None:
                raise RuntimeError("internal_test_evidence_missing")
            from .internal_publish import internal_test_evidence_path

            internal_metrics = dict(internal_evidence["internal_test_metrics"])
            development_metrics = dict(manifest["development_metrics"])
            evaluation_fields = {
                "schema_version": "internal-android-v2",
                "artifact_status": "internal_candidate",
                "evaluation_scope": "internal_32_participants",
                "validation_scope": "internal_32_participants",
                "claims_policy": "decision_support_only",
                "release_status": "internal_candidate",
                "gate_passed": deployment_gate,
                "research_gate_passed": deployment_gate,
                "performance_tier": conservative["assessment"]["performance_tier"],
                "deployment_scope": "internal",
                "disclaimer_required": False,
                "balanced_accuracy_development": float(
                    development_metrics["balanced_accuracy"]
                ),
                "f1_development": float(development_metrics["f1"]),
                "roc_auc_development": float(development_metrics["roc_auc"]),
                "balanced_accuracy_internal_test": float(
                    internal_metrics["balanced_accuracy"]
                ),
                "accuracy_internal_test": float(internal_metrics["accuracy"]),
                "f1_internal_test": float(internal_metrics["f1"]),
                "roc_auc_internal_test": float(internal_metrics["roc_auc"]),
                "development_participants": [
                    f"ID-{index:02d}" for index in range(1, 29)
                ],
                "internal_test_participants": [
                    f"ID-{index:02d}" for index in range(29, 33)
                ],
                "validation_cohort_ids": [f"ID-{index:02d}" for index in range(1, 29)],
                "cohort_access_count": 1,
                "hrv_prv_status": "GO",
                "training_admission_status": "GO",
                "development_model_status": "GO",
                "internal_test_status": "GO",
                "onnx_export_status": "GO",
                "android_device_parity_status": "PENDING",
                "internal_android_status": "PENDING_DEVICE_PARITY",
                "internal_test_contract_fingerprint": internal_evidence[
                    "internal_test_contract_fingerprint"
                ],
                "internal_test_evidence_sha256": _sha256(
                    internal_test_evidence_path(self.paths, self.config, sensor_channel)
                ),
                "acceptance_metrics": {
                    "development": development_metrics,
                    "internal_test": internal_metrics,
                    "onnx_parity": _parity_payload(parity),
                },
            }
        else:
            evaluation_fields = {
                "schema_version": "1.0.0",
                "artifact_status": "release_candidate",
                "evaluation_scope": "development_and_reused_internal_test",
                "release_status": (
                    "release_candidate" if deployment_gate else "blocked"
                ),
                "gate_passed": deployment_gate,
                "performance_tier": conservative["assessment"]["performance_tier"],
                "deployment_scope": conservative["assessment"]["deployment_scope"],
                "disclaimer_required": conservative["assessment"][
                    "disclaimer_required"
                ],
                "balanced_accuracy_holdout": float(
                    manifest["holdout_metrics"]["balanced_accuracy"]
                ),
                "accuracy_holdout": float(manifest["holdout_metrics"]["accuracy"]),
                "f1_holdout": float(manifest["holdout_metrics"]["f1"]),
                "roc_auc_holdout": float(manifest["holdout_metrics"]["roc_auc"]),
                "validation_cohort_ids": list(manifest["holdout_participants"]),
                "cohort_access_count": 1,
                "acceptance_metrics": {
                    "development": manifest["development_metrics"],
                    "holdout": manifest["holdout_metrics"],
                    "onnx_parity": _parity_payload(parity),
                },
            }
        manifest.update(
            {
                "run_id": str(self.config.artifacts.get("run_id", "")),
                "source_tree_sha256": (run_contract.get("source_tree_sha256")),
                "config_fingerprint": (run_contract.get("config_fingerprint")),
                "split_fingerprint": (run_contract.get("split_fingerprint")),
                "run_contract_sha256": run_contract_sha256,
                "source_checksums": run_contract.get("source_checksums", {}),
                "gold_manifest_sha256": (
                    _sha256(gold_manifest_path)
                    if gold_manifest_path.is_file()
                    else None
                ),
                **window_contract,
                "onnx_model_path": str(
                    (published_export_dir / onnx_path.name).relative_to(self.root)
                ).replace("\\", "/"),
                "onnx_sha256": _sha256(onnx_path),
                "onnx_opset": self.config.onnx.opset,
                "export_version": self.config.onnx.export_version,
                "static_batch_size": self.config.onnx.static_batch_size,
                "onnx_input": {
                    "name": self.config.onnx.input_name,
                    "dtype": "float32",
                    "shape": [
                        self.config.onnx.static_batch_size,
                        len(feature_order),
                    ],
                },
                "onnx_output": {
                    "name": self.config.onnx.output_name,
                    "dtype": "float32",
                    "shape": [self.config.onnx.static_batch_size],
                },
                "label_order": ["relax", "task"],
                "onnx_parity": _parity_payload(parity),
                "onnx_checker_passed": True,
                "converter_preflight": preflight,
                "int8": {
                    "status": "not_published",
                    "reason": "FP32-first; operator and accuracy gates not yet validated",
                },
                "deployment_gate_passed": deployment_gate,
                "android_ready": False,
                "physiological_validation": physiological,
                "physiological_validation_status": physiological.get(
                    "decision", "INSUFFICIENT_EVIDENCE"
                ),
                "physiological_validation_controls_model_export": False,
                "source_rate_hz": source_rate_hz,
                "model_rate_hz": model_rate_hz,
                **{
                    "filter_contract": filter_contract,
                    "filter_algorithm": filter_contract["algorithm"],
                    "filter_order": filter_contract["order"],
                    "filter_low_cut_hz": filter_contract["low_cut_hz"],
                    "filter_high_cut_hz": filter_contract["high_cut_hz"],
                    "filter_zero_phase": filter_contract["zero_phase"],
                    "filter_pad_policy": filter_contract["pad_policy"],
                    "filter_design_version": filter_contract["design_version"],
                    "filter_contract_fingerprint": filter_contract["fingerprint"],
                },
                "sampling_rate_contract_source": "gold_observed_timestamp_median",
                "sampling_rate_hz_actual_median": source_rate_hz,
                "sampling_rate_hz_actual_min": float(sampling_rates.min()),
                "sampling_rate_hz_actual_max": float(sampling_rates.max()),
                **evaluation_fields,
                "model_name": f"stress_{sensor_channel}",
                "model_family": str(manifest.get("model_name", "classical")),
                "task": "stress_classification",
                "inputs": [
                    {
                        "name": self.config.onnx.input_name,
                        "dtype": "float32",
                        "shape": [
                            self.config.onnx.static_batch_size,
                            len(feature_order),
                        ],
                    }
                ],
                "output": {
                    "name": self.config.onnx.output_name,
                    "dtype": "float32",
                    "shape": [self.config.onnx.static_batch_size],
                },
                "performance_policy": self.performance_policy.as_dict(),
                "performance_policy_version": self.performance_policy.version,
                "performance_policy_sha256": self.performance_policy.fingerprint(),
                "conservative_metrics": conservative["conservative_metrics"],
                "research_disclaimer": self.performance_policy.research_disclaimer,
                "balanced_accuracy_gate": self.performance_policy.minimum_research.balanced_accuracy,
                "accuracy_gate": None,
                "accuracy_is_release_gate": False,
                "f1_gate": self.performance_policy.minimum_research.f1,
                "roc_auc_gate": self.performance_policy.minimum_research.roc_auc,
                "threshold_source": "participant_grouped_development_validation",
                "candidate_bundle_sha256": manifest.get(
                    "candidate_artifact_fingerprint",
                    manifest.get("model_sha256", ""),
                ),
                "native_model_sha256": _sha256(native_path),
                "source_candidate_manifest_sha256": _sha256(manifest_path),
                "source_candidate_manifest": str(
                    manifest_path.relative_to(self.root)
                ).replace("\\", "/"),
                "source_gold_features": str(
                    candidate_gold_path.relative_to(self.root)
                ).replace("\\", "/"),
                "source_gold_sha256": _sha256(candidate_gold_path),
                "source_gold_features_sha256": _sha256(candidate_gold_path),
                "calibration_version": str(
                    manifest.get("probability_calibration", {}).get(
                        "version", "unversioned"
                    )
                ),
                "calibration": manifest.get("probability_calibration", {}),
                "parity_vector_count": sample_count,
                "validation_channel_decision": self._validation_channel_decision(
                    sensor_channel
                ),
            }
        )
        if self.mode is OnnxExportMode.INTERNAL:
            # v2 has its own scoped readiness vocabulary.  Re-emitting legacy
            # production flags would make a pending device receipt look like a
            # model failure, so they are forbidden from this artifact.
            for legacy_key in (
                "android_ready",
                "clinical_validation_status",
                "holdout_evaluated",
                "holdout_metrics",
                "holdout_raw_metrics",
                "research_disclaimer",
            ):
                manifest.pop(legacy_key, None)
        manifest["android_contract_sha256"] = hashlib.sha256(
            _android_contract_payload(manifest).encode("utf-8")
        ).hexdigest()
        manifest["manifest_sha256"] = _manifest_sha256(manifest)
        output_manifest = versioned_export_dir / "model_manifest.json"
        _atomic_write_json(output_manifest, manifest)
        _atomic_write_json(
            versioned_export_dir / "onnx_parity.json", _parity_payload(parity)
        )
        _atomic_write_json(
            versioned_export_dir / "onnx_feature_parity_vectors.json",
            {
                "schema_version": "1.0.0",
                "evaluation_scope": "development_only",
                "synthetic_data_used": False,
                "model_version": self.model_version,
                "sensor_channel": sensor_channel,
                "input_name": self.config.onnx.input_name,
                "output_name": self.config.onnx.output_name,
                "count": len(vectors),
                "source_gold_sha256": _sha256(candidate_gold_path),
                "cases": vectors,
            },
        )
        if not parity_passed:
            if staging_export_dir is not None:
                shutil.rmtree(staging_export_dir, ignore_errors=True)
            raise RuntimeError(
                f"ONNX parity failed for {sensor_channel}: max_abs_diff={parity.max_abs_diff}"
            )
        if not deployment_gate:
            if staging_export_dir is not None:
                shutil.rmtree(staging_export_dir, ignore_errors=True)
            raise RuntimeError(
                f"ONNX {self.mode.value} gate failed for {sensor_channel}: "
                f"parity={parity.passed};model_size={model_size};latency_ms={latency}"
            )
        if staging_export_dir is not None:
            staging_export_dir.replace(published_export_dir)
        return manifest

    def _write_research_index(self, manifests: list[dict[str, Any]]) -> Path:
        """Publish a run-scoped non-production index for the research flavor."""

        index_path = self.export_dir / "research" / "research_model_index.json"
        existing_models: list[dict[str, Any]] = []
        if index_path.is_file():
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            if existing.get("run_id") != self.config.artifacts.get("run_id"):
                raise RuntimeError("research_index_run_mismatch")
            existing_models = list(existing.get("models", []))
        by_channel = {str(model["sensor_channel"]): model for model in existing_models}
        for manifest in manifests:
            sensor_channel = str(manifest["sensor_channel"])
            export_dir = self._published_export_dir(sensor_channel)
            files = {
                "model": export_dir / f"stress_{sensor_channel}_fp32.onnx",
                "manifest": export_dir / "model_manifest.json",
                "parity": export_dir / "onnx_parity.json",
                "test_vectors": export_dir / "onnx_feature_parity_vectors.json",
            }
            by_channel[sensor_channel] = {
                "model_version": manifest["model_version"],
                "sensor_channel": sensor_channel,
                "artifact_status": "research_candidate",
                "evaluation_scope": "development_only",
                "release_status": manifest["release_status"],
                "android_ready": False,
                "clinical_validation_status": "external_validation_required",
                "calibration_version": manifest["calibration_version"],
                "balanced_accuracy_development": manifest[
                    "balanced_accuracy_development"
                ],
                "f1_development": manifest["f1_development"],
                "roc_auc_development": manifest["roc_auc_development"],
                "performance_policy_version": manifest["performance_policy_version"],
                "performance_policy_sha256": manifest["performance_policy_sha256"],
                "performance_tier": manifest["performance_tier"],
                "deployment_scope": "research",
                "research_disclaimer": manifest["research_disclaimer"],
                "disclaimer_required": True,
                "analysis_seconds": manifest["analysis_seconds"],
                "window_seconds": manifest["window_seconds"],
                "stride_seconds": manifest["stride_seconds"],
                "expected_windows_per_session": manifest[
                    "expected_windows_per_session"
                ],
                "source_rate_hz": manifest["source_rate_hz"],
                "model_rate_hz": manifest["model_rate_hz"],
                "candidate_artifact_fingerprint": manifest.get(
                    "candidate_artifact_fingerprint"
                ),
                "files": {
                    kind: {
                        "path": str(path.relative_to(self.root)).replace("\\", "/"),
                        "sha256": _sha256(path),
                    }
                    for kind, path in files.items()
                },
            }
        payload = {
            "schema_version": "1.0.0",
            "artifact_status": "research_candidate",
            "evaluation_scope": "development_only",
            "release_status": "research_candidate",
            "deployment_scope": "research",
            "android_ready": False,
            "clinical_validation_status": "external_validation_required",
            "run_id": str(self.config.artifacts.get("run_id", "")),
            "model_version": self.model_version,
            "models": [by_channel[channel] for channel in sorted(by_channel)],
        }
        _atomic_write_json(index_path, payload)
        return index_path

    def _write_internal_index(self, manifests: list[dict[str, Any]]) -> Path:
        """Publish the v2 Android index without legacy production flags."""

        index_path = self.export_dir / "internal" / "internal_model_index.json"
        existing_models: list[dict[str, Any]] = []
        if index_path.is_file():
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            if existing.get("schema_version") != "internal-android-v2" or existing.get(
                "run_id"
            ) != self.config.artifacts.get("run_id"):
                raise RuntimeError("internal_index_run_or_schema_mismatch")
            existing_models = list(existing.get("models", []))
        by_channel = {str(entry["sensor_channel"]): entry for entry in existing_models}
        for manifest in manifests:
            sensor_channel = str(manifest["sensor_channel"])
            export_dir = self._published_export_dir(sensor_channel)
            files = {
                "model": export_dir / f"stress_{sensor_channel}_fp32.onnx",
                "manifest": export_dir / "model_manifest.json",
                "parity": export_dir / "onnx_parity.json",
                "test_vectors": export_dir / "onnx_feature_parity_vectors.json",
            }
            entry = {
                "model_version": manifest["model_version"],
                "sensor_channel": sensor_channel,
                "artifact_status": "internal_candidate",
                "evaluation_scope": "internal_32_participants",
                "validation_scope": "internal_32_participants",
                "claims_policy": "decision_support_only",
                "release_status": "internal_candidate",
                "hrv_prv_status": manifest["hrv_prv_status"],
                "training_admission_status": manifest["training_admission_status"],
                "development_model_status": manifest["development_model_status"],
                "internal_test_status": manifest["internal_test_status"],
                "onnx_export_status": manifest["onnx_export_status"],
                "android_device_parity_status": manifest[
                    "android_device_parity_status"
                ],
                "internal_android_status": manifest["internal_android_status"],
                "calibration_version": manifest["calibration_version"],
                "balanced_accuracy_development": manifest[
                    "balanced_accuracy_development"
                ],
                "f1_development": manifest["f1_development"],
                "roc_auc_development": manifest["roc_auc_development"],
                "balanced_accuracy_internal_test": manifest[
                    "balanced_accuracy_internal_test"
                ],
                "f1_internal_test": manifest["f1_internal_test"],
                "roc_auc_internal_test": manifest["roc_auc_internal_test"],
                "performance_policy_version": manifest["performance_policy_version"],
                "performance_policy_sha256": manifest["performance_policy_sha256"],
                "performance_tier": manifest["performance_tier"],
                "deployment_scope": manifest["deployment_scope"],
                "analysis_seconds": manifest["analysis_seconds"],
                "window_seconds": manifest["window_seconds"],
                "stride_seconds": manifest["stride_seconds"],
                "expected_windows_per_session": manifest[
                    "expected_windows_per_session"
                ],
                "source_rate_hz": manifest["source_rate_hz"],
                "model_rate_hz": manifest["model_rate_hz"],
                "candidate_artifact_fingerprint": manifest[
                    "candidate_artifact_fingerprint"
                ],
                "internal_test_contract_fingerprint": manifest[
                    "internal_test_contract_fingerprint"
                ],
                "files": {
                    kind: {
                        "path": str(path.relative_to(self.root)).replace("\\", "/"),
                        "sha256": _sha256(path),
                    }
                    for kind, path in files.items()
                },
            }
            prior = by_channel.get(sensor_channel)
            if prior is not None and prior != entry:
                raise RuntimeError("internal_index_channel_collision")
            by_channel[sensor_channel] = entry
        payload = {
            "schema_version": "internal-android-v2",
            "run_id": str(self.config.artifacts.get("run_id", "")),
            "model_version": self.model_version,
            "validation_scope": "internal_32_participants",
            "development_participants": [f"ID-{index:02d}" for index in range(1, 29)],
            "internal_test_participants": [
                f"ID-{index:02d}" for index in range(29, 33)
            ],
            "claims_policy": "decision_support_only",
            "internal_android_status": "PENDING_DEVICE_PARITY",
            "eligible_channels": sorted(by_channel),
            "models": [by_channel[channel] for channel in sorted(by_channel)],
        }
        _atomic_write_json(index_path, payload)
        return index_path

    def run(self, sensor_channel: str | None = None) -> dict[str, Any]:
        mode = getattr(self, "mode", OnnxExportMode.PRODUCTION)
        sensors = (sensor_channel,) if sensor_channel else EXPORT_CHANNELS
        if sensor_channel is not None:
            manifest = self.export_sensor(sensor_channel)
            if mode is OnnxExportMode.RESEARCH:
                self._write_research_index([manifest])
            elif mode is OnnxExportMode.INTERNAL:
                self._write_internal_index([manifest])
            return {sensor_channel: manifest}
        manifests: list[dict[str, Any]] = []
        results: dict[str, Any] = {}
        for sensor in sensors:
            try:
                manifest = self.export_sensor(sensor)
            except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as exc:
                results[sensor] = {
                    "sensor_channel": sensor,
                    "status": "blocked",
                    "reason": str(exc),
                }
                continue
            manifests.append(manifest)
            results[sensor] = manifest
        if mode is OnnxExportMode.RESEARCH and manifests:
            self._write_research_index(manifests)
        elif mode is OnnxExportMode.INTERNAL and manifests:
            self._write_internal_index(manifests)
        elif sensor_channel is None:
            self._mark_readiness_model_release(manifests)
        return results

    def _mark_readiness_model_release(self, manifests: list[dict[str, Any]]) -> None:
        """Record verified ONNX evidence without changing broader data gates."""

        readiness_path = self.paths.gold_dir / "data_readiness.json"
        if not readiness_path.is_file():
            return
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        readiness["model_release_status"] = "blocked"
        readiness["model_release_reason"] = (
            "android_physical_validation_evidence_pending"
        )
        readiness["model_release_channels"] = [
            manifest["sensor_channel"] for manifest in manifests
        ]
        readiness["model_release_channel_status"] = {
            channel: (
                "release_candidate"
                if any(manifest["sensor_channel"] == channel for manifest in manifests)
                else "blocked"
            )
            for channel in EXPORT_CHANNELS
        }
        readiness["model_release_version"] = self.model_version
        readiness["model_release_max_abs_diff"] = (
            max(
                float(manifest["onnx_parity"]["max_abs_diff"]) for manifest in manifests
            )
            if manifests
            else None
        )
        _atomic_write_json(readiness_path, readiness)
