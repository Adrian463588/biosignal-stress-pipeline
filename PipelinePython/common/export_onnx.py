from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .contracts import ModelManifest
from .settings import PipelinePaths, PipelineSettings
from .config import TIERED_PERFORMANCE_GATES

MINIMUM_RESEARCH_GATES = dict(TIERED_PERFORMANCE_GATES["minimum_research"])
# Compatibility name for callers that only display the BA component.  The
# export decision below always checks all three policy metrics.
BALANCED_ACC_THRESHOLD = float(MINIMUM_RESEARCH_GATES["balanced_accuracy"])
ALLOWED_PHASES = ("relax", "task", "Relax", "Task")
ALLOWED_QUALITY_TIERS = ("excellent", "good", "Excellent", "Good")

logger = logging.getLogger(__name__)

PARITY_THRESHOLD = 1e-5
PARITY_SAMPLE_SIZE = 100


class ONNXExporter:
    def __init__(self, paths: PipelinePaths, settings: PipelineSettings) -> None:
        self.paths = paths
        self.settings = settings

    def run(self, sensor_channel: str = "ppg_ir") -> dict:
        """Export an accepted stress model for one PPG channel to ONNX.

        Supports three export backends (tried in order):
        1. skl2onnx  — Logistic Regression, Random Forest (sklearn pipelines)
        2. onnxmltools.convert.xgboost — XGBoost within a sklearn Pipeline
        3. Raises RuntimeError if no backend can handle the model type.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            ) from exc

        self.paths.export_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.paths.models_dir / "stress" / sensor_channel
        joblib_candidates = [
            model_dir / "best_model.joblib",
            model_dir / "classical" / "candidate_model.joblib",
            model_dir / "stress_v3.0.0" / "best_model.joblib",
            model_dir / "stress_v3.0.0" / "model.joblib",
        ]
        pt_candidates = [
            model_dir / "model.pt",
            model_dir / "deep_learning" / "model.pt",
        ]

        model = None
        chosen_dir = model_dir
        for p in joblib_candidates:
            if p.exists():
                model = joblib.load(p)
                chosen_dir = p.parent
                logger.info("Found joblib model at %s", p)
                break

        if model is None:
            for p in pt_candidates:
                if p.exists():
                    import torch

                    model = torch.load(p, map_location="cpu")
                    if hasattr(model, "eval"):
                        model.eval()
                    chosen_dir = p.parent
                    logger.info("Found PyTorch model at %s", p)
                    break

        if model is None:
            raise FileNotFoundError(
                f"No trained model found for sensor_channel={sensor_channel} under {model_dir}. Run training first."
            )

        feature_path = chosen_dir / "selected_features.json"
        if not feature_path.exists():
            feature_path = model_dir / "selected_features.json"

        metrics_path = chosen_dir / "split_and_metrics.json"
        if not metrics_path.exists():
            metrics_path = chosen_dir / "candidate_manifest.json"
        if not metrics_path.exists():
            metrics_path = model_dir / "split_and_metrics.json"

        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
        feature_cols = self._selected_feature_names(feature_payload)
        manifest_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = manifest_data.get(
            "test_metrics", manifest_data.get("holdout_metrics", manifest_data)
        )
        model_name = manifest_data.get("model_name", "stress_classifier")
        feature_version = manifest_data.get(
            "feature_version", self.settings.feature_version
        )

        if "dummy" in str(model_name).lower():
            return self._write_skipped(
                sensor_channel,
                metrics,
                feature_version,
                "skipped_dummy_single_class_model",
            )
        if metrics.get("below_threshold") or any(
            float(metrics.get(metric, 0.0)) < float(minimum)
            for metric, minimum in MINIMUM_RESEARCH_GATES.items()
        ):
            return self._write_skipped(
                sensor_channel,
                metrics,
                feature_version,
                "skipped_below_minimum_research_gate",
            )

        n_features = len(feature_cols)
        sample = self._load_parity_sample(sensor_channel, feature_cols)

        # ── Attempt ONNX conversion ──────────────────────────────────────────
        onnx_bytes, is_bare_clf = self._convert_to_onnx(model, model_name, n_features)
        if onnx_bytes is None:
            return self._write_skipped(
                sensor_channel,
                metrics,
                feature_version,
                "skipped_onnx_conversion_failed",
            )

        onnx_path = self.paths.export_dir / f"stress_{sensor_channel}_model.onnx"
        onnx_path.write_bytes(onnx_bytes)

        # ── Parity check ─────────────────────────────────────────────────────
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        py_prob = self._predict_positive_probability(model, sample)

        sample_tensor = sample.to_numpy(dtype=np.float32)
        scaler_mean = None
        scaler_scale = None
        if (
            hasattr(model, "named_steps")
            and "imputer" in model.named_steps
            and "scaler" in model.named_steps
        ):
            imp = model.named_steps["imputer"]
            scl = model.named_steps["scaler"]
            scaler_mean = scl.mean_.tolist() if hasattr(scl, "mean_") else None
            scaler_scale = scl.scale_.tolist() if hasattr(scl, "scale_") else None
            if is_bare_clf:
                sample_tensor = scl.transform(imp.transform(sample_tensor)).astype(
                    np.float32
                )

        ort_out = sess.run(None, {input_name: sample_tensor})
        ort_prob = self._extract_positive_probability(ort_out)
        diff = np.abs(py_prob - ort_prob)
        parity_max_abs_diff = float(np.max(diff))
        parity_mean_abs_diff = float(np.mean(diff))

        parity_status = "ok" if parity_max_abs_diff <= PARITY_THRESHOLD else "mismatch"
        if parity_status == "mismatch":
            logger.warning(
                "ONNX parity FAILED for %s: max_abs_diff=%.2e > threshold=%.2e; removing ONNX",
                sensor_channel,
                parity_max_abs_diff,
                PARITY_THRESHOLD,
            )
            onnx_path.unlink(missing_ok=True)
        else:
            logger.info(
                "ONNX export OK: %s | max_abs_diff=%.2e | model=%s",
                onnx_path.name,
                parity_max_abs_diff,
                model_name,
            )

        parity = {
            "onnx_path": str(onnx_path) if parity_status == "ok" else None,
            "sensor_channel": sensor_channel,
            "feature_version": feature_version,
            "model_window_seconds": manifest_data.get(
                "model_window_seconds",
                getattr(self.settings, "model_window_seconds", 180),
            ),
            "model_window_step_seconds": manifest_data.get(
                "model_window_step_seconds",
                getattr(self.settings, "model_window_step_seconds", 10),
            ),
            "parity_probability_max_abs_diff": parity_max_abs_diff,
            "parity_probability_mae": parity_mean_abs_diff,
            "checked_rows": int(len(py_prob)),
            "parity_status": parity_status,
            "threshold": PARITY_THRESHOLD,
        }
        self._write_json(self.paths.export_dir / "onnx_parity.json", parity)
        self._write_json(
            self.paths.export_dir / f"onnx_parity_{sensor_channel}.json", parity
        )
        export_version = "onnx-v2" if parity_status == "ok" else "onnx-v2-mismatch"
        self._write_manifest(
            sensor_channel,
            metrics,
            feature_version,
            export_version,
            parity,
            feature_order=feature_cols,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            probability_calibration=manifest_data.get("probability_calibration"),
        )

        return parity

    # ── ONNX conversion ───────────────────────────────────────────────────────

    def _convert_to_onnx(
        self,
        model: Any,
        model_name: str,
        n_features: int,
    ) -> tuple[bytes | None, bool]:
        """Dispatch ONNX conversion based on model type.

        Priority:
        1. PyTorch → torch.onnx.export(..., dynamo=True)
        2. LightGBM → onnxmltools.convert_lightgbm
        3. XGBoost → onnxmltools
        4. sklearn Pipeline (RF / LR) → skl2onnx
        """
        if "torch" in str(type(model)).lower() or hasattr(model, "forward"):
            result = self._convert_pytorch(model, n_features)
            if result is not None:
                logger.info("PyTorch→ONNX conversion succeeded via dynamo=True")
                return result, False
            logger.warning("PyTorch conversion failed")

        if "lightgbm" in model_name.lower() or "lgbm" in str(model).lower():
            result = self._convert_lightgbm_pipeline(model, n_features)
            if result is not None:
                logger.info("LightGBM→ONNX conversion succeeded via onnxmltools")
                return result, True
            logger.warning("onnxmltools LightGBM conversion failed; trying skl2onnx")

        if "xgboost" in model_name.lower():
            result = self._convert_xgboost_pipeline(model, n_features)
            if result is not None:
                logger.info("XGBoost→ONNX conversion succeeded via onnxmltools")
                return result, True
            logger.warning(
                "onnxmltools XGBoost conversion failed; falling back to skl2onnx"
            )

        res = self._convert_sklearn_pipeline(model, n_features)
        return res, False

    def _convert_pytorch(self, model: Any, n_features: int) -> bytes | None:
        """Convert PyTorch model to ONNX using dynamo=True."""
        try:
            import torch
            import io

            dummy_input = torch.randn(1, n_features)
            f = io.BytesIO()
            # As per requirement, use dynamo=True to ensure correct probability & label output formatting
            torch.onnx.export(
                model,
                dummy_input,
                f,
                export_params=True,
                dynamo=True,
                input_names=["input"],
                output_names=["label", "probability"],
            )
            return f.getvalue()
        except Exception as exc:
            logger.error("PyTorch dynamo conversion failed: %s", exc)
            return None

    def _convert_lightgbm_pipeline(
        self, pipeline: Any, n_features: int
    ) -> bytes | None:
        """Convert LightGBM model to ONNX via onnxmltools."""
        try:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType

            clf = (
                pipeline.named_steps.get("clf")
                if hasattr(pipeline, "named_steps")
                else pipeline
            )
            if hasattr(clf, "best_estimator_"):
                clf = clf.best_estimator_

            onnx_model = onnxmltools.convert_lightgbm(
                clf,
                initial_types=[("float_input", FloatTensorType([None, n_features]))],
                zipmap=False,
            )
            return onnx_model.SerializeToString()
        except Exception as exc:  # noqa: BLE001
            logger.error("onnxmltools LightGBM conversion failed: %s", exc)
            return None

    def _convert_sklearn_pipeline(self, model: Any, n_features: int) -> bytes | None:
        """Convert a sklearn Pipeline to ONNX via skl2onnx."""
        try:
            from skl2onnx import to_onnx
            from skl2onnx.common.data_types import FloatTensorType
            from sklearn.pipeline import Pipeline as SKPipeline

            steps = []
            for name, step in model.steps:
                if name == "clf" and hasattr(step, "best_estimator_"):
                    steps.append((name, step.best_estimator_))
                else:
                    steps.append((name, step))
            unpacked_model = SKPipeline(steps)
            try:
                from lightgbm import LGBMClassifier
                from onnxmltools.convert.common.shape_calculator import (
                    calculate_linear_classifier_output_shapes,
                )
                from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
                    convert_lightgbm,
                )
                from skl2onnx import update_registered_converter

                update_registered_converter(
                    LGBMClassifier,
                    "LightGbmLGBMClassifier",
                    calculate_linear_classifier_output_shapes,
                    convert_lightgbm,
                    options={"zipmap": [True, False, "columns"]},
                )
            except Exception as e:
                logger.debug("LGBM registration warning: %s", e)

            onnx_model = to_onnx(
                unpacked_model,
                initial_types=[("input", FloatTensorType([None, n_features]))],
                options={id(unpacked_model): {"zipmap": False}},
                target_opset=15,
            )
            return onnx_model.SerializeToString()
        except Exception as exc:  # noqa: BLE001
            logger.error("skl2onnx conversion failed: %s", exc)
            return None

    def _convert_xgboost_pipeline(self, pipeline: Any, n_features: int) -> bytes | None:
        """Convert an XGBoost model (possibly inside sklearn Pipeline) to ONNX.

        Strategy:
        1. Extract XGBClassifier from Pipeline step 'clf'.
        2. Build a pre-processing array using the pipeline's imputer + scaler.
        3. Convert XGBClassifier alone via onnxmltools, then wrap with skl2onnx
           pre-processing operators — OR use skl2onnx's XGBoost converter directly.
        """
        try:
            from skl2onnx import to_onnx
            from skl2onnx.common.data_types import FloatTensorType

            # skl2onnx 1.17+ supports XGBoost natively within sklearn Pipelines.
            onnx_model = to_onnx(
                pipeline,
                initial_types=[("input", FloatTensorType([None, n_features]))],
                options={id(pipeline): {"zipmap": False}},
                target_opset=15,
            )
            logger.info("XGBoost converted via skl2onnx (native support)")
            return onnx_model.SerializeToString()

        except Exception as skl2onnx_exc:
            logger.warning(
                "skl2onnx XGBoost failed (%s); trying onnxmltools path", skl2onnx_exc
            )

        # Fallback: onnxmltools on the raw XGBClassifier + wrap with pre-processing.
        try:
            from sklearn.pipeline import Pipeline as SKPipeline
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import (
                FloatTensorType as OmlFloatType,
            )
            from skl2onnx import to_onnx as skl_to_onnx
            from skl2onnx.common.data_types import FloatTensorType

            # Extract the RandomizedSearchCV or bare XGBClassifier
            clf_step = pipeline.named_steps.get("clf")
            if clf_step is None:
                return None
            # If wrapped in RandomizedSearchCV, get the best estimator
            if hasattr(clf_step, "best_estimator_"):
                xgb_clf = clf_step.best_estimator_
            else:
                xgb_clf = clf_step

            # Convert the XGBClassifier to ONNX with onnxmltools
            xgb_onnx = convert_xgboost(
                xgb_clf,
                initial_types=[("input", OmlFloatType([None, n_features]))],
            )

            # Build a preprocessing-only sklearn Pipeline (imputer + scaler)
            pre_steps = [(k, v) for k, v in pipeline.steps if k != "clf"]
            if pre_steps:
                pre_pipe = SKPipeline(pre_steps)
                # Fit a dummy sample to make it exportable (already fitted from training)
                np.zeros((1, n_features), dtype=np.float32)
                skl_to_onnx(
                    pre_pipe,
                    initial_types=[("input", FloatTensorType([None, n_features]))],
                    options={id(pre_pipe): {"zipmap": False}},
                    target_opset=15,
                )
                # Chain: pre_onnx → xgb_onnx via onnx graph merge
                # For simplicity: just return the XGBClassifier ONNX
                # (parity check will catch errors; pre-processing already done by pipeline)
                logger.info(
                    "XGBoost converted via onnxmltools (raw XGBClassifier only)"
                )
                return xgb_onnx.SerializeToString()

            return xgb_onnx.SerializeToString()

        except Exception as exc:  # noqa: BLE001
            logger.error("onnxmltools XGBoost conversion also failed: %s", exc)
            return None

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_parity_sample(
        self, sensor_channel: str, feature_cols: list[str]
    ) -> pd.DataFrame:
        gold_path = getattr(self.paths, "gold_features_path", None)
        if gold_path is None:
            gold_path = self.paths.gold_dir / "gold_features.csv"
        if not Path(gold_path).exists():
            raise FileNotFoundError(
                f"Gold features not found for ONNX parity: {gold_path}"
            )

        sample_df = pd.read_csv(gold_path)
        sample_df = sample_df[sample_df["phase"].isin(ALLOWED_PHASES)].copy()
        sample_df = sample_df[sample_df["sensor_channel"] == sensor_channel].copy()
        if "quality_tier" in sample_df.columns:
            filtered_q = sample_df[
                sample_df["quality_tier"].isin(ALLOWED_QUALITY_TIERS)
            ].copy()
            if not filtered_q.empty:
                sample_df = filtered_q
        if sample_df.empty:
            raise ValueError(
                f"No gold rows for ONNX parity check (sensor_channel={sensor_channel})"
            )
        if len(sample_df) < PARITY_SAMPLE_SIZE:
            raise ValueError(
                f"ONNX parity requires at least {PARITY_SAMPLE_SIZE} eligible rows; "
                f"found {len(sample_df)} for sensor_channel={sensor_channel}"
            )

        random_seed = getattr(self.settings, "random_seed", 42)
        sample_df = sample_df.sample(
            n=PARITY_SAMPLE_SIZE,
            random_state=random_seed,
        )
        X = sample_df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").clip(lower=-1e9, upper=1e9)
        return X.astype(np.float32)

    # ── Skipped export ────────────────────────────────────────────────────────

    def _write_skipped(
        self,
        sensor_channel: str,
        metrics: dict,
        feature_version: str,
        status: str,
    ) -> dict:
        parity = {
            "onnx_path": None,
            "sensor_channel": sensor_channel,
            "feature_version": feature_version,
            "model_window_seconds": getattr(self.settings, "model_window_seconds", 180),
            "model_window_step_seconds": getattr(
                self.settings, "model_window_step_seconds", 10
            ),
            "parity_probability_max_abs_diff": None,
            "parity_probability_mae": None,
            "checked_rows": 0,
            "parity_status": status,
            "threshold": PARITY_THRESHOLD,
        }
        self._write_json(self.paths.export_dir / "onnx_parity.json", parity)
        self._write_json(
            self.paths.export_dir / f"onnx_parity_{sensor_channel}.json", parity
        )
        self._write_manifest(
            sensor_channel, metrics, feature_version, "onnx-v2-skipped", parity
        )
        logger.warning("Skipping ONNX export for %s: %s", sensor_channel, status)
        return parity

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _write_manifest(
        self,
        sensor_channel: str,
        metrics: dict,
        feature_version: str,
        export_version: str,
        parity: dict,
        feature_order: list[str] | None = None,
        scaler_mean: list[float] | None = None,
        scaler_scale: list[float] | None = None,
        probability_calibration: dict[str, Any] | None = None,
    ) -> None:
        channel_decision = self._read_channel_recommendation()
        ba = metrics.get("balanced_accuracy", 0.0)
        gate_passed = bool(
            ba >= BALANCED_ACC_THRESHOLD and parity.get("parity_status") == "ok"
        )

        manifest = ModelManifest(
            preprocessing_version=getattr(
                self.settings, "preprocessing_version", "v3.0.0"
            ),
            feature_version=feature_version,
            validation_channel_decision=channel_decision,
            model_version=getattr(self.settings, "model_version", "stress_v3.0.0"),
            export_version=export_version,
            acceptance_metrics=metrics,
        ).__dict__
        manifest.update(
            {
                "task": "stress_classification",
                "sensor_channel": sensor_channel,
                "onnx_model_path": f"data/artifacts/export/stress_{sensor_channel}_model.onnx",
                "holdout_participants": ["ID-29", "ID-30", "ID-31", "ID-32"],
                "model_window_seconds": getattr(
                    self.settings, "model_window_seconds", 180
                ),
                "model_window_step_seconds": getattr(
                    self.settings, "model_window_step_seconds", 10
                ),
                "balanced_accuracy_gate": BALANCED_ACC_THRESHOLD,
                "balanced_accuracy_holdout": ba,
                "deployment_gate_passed": gate_passed,
                "feature_order": feature_order or [],
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "parity": parity,
            }
        )
        if probability_calibration is not None:
            manifest["probability_calibration"] = probability_calibration

        # Calculate SHA256 checksum
        onnx_path = self.paths.export_dir / f"stress_{sensor_channel}_model.onnx"
        if onnx_path.exists():
            import hashlib

            manifest["onnx_sha256"] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        else:
            manifest["onnx_sha256"] = ""

        self._write_json(self.paths.export_dir / "model_manifest.json", manifest)
        self._write_json(
            self.paths.export_dir / f"model_manifest_{sensor_channel}.json", manifest
        )
        self._write_json(
            self.paths.export_dir / f"stress_{sensor_channel}_manifest_v3.json",
            manifest,
        )

    # ── Probability helpers ───────────────────────────────────────────────────

    @staticmethod
    def _selected_feature_names(payload: Any) -> list[str]:
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict) and isinstance(
            payload.get("selected_features"), list
        ):
            return [str(x) for x in payload["selected_features"]]
        raise ValueError(
            "selected_features.json must be a list or an object with selected_features"
        )

    @staticmethod
    def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return prob[:, 1]
            return prob.ravel()
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    @staticmethod
    def _extract_positive_probability(outputs) -> np.ndarray:
        """Parse ONNX Runtime output to extract class-1 probability.

        Handles all known skl2onnx / onnxmltools output shapes:
        - [tensor(int64), seq(map(int64, tensor(float)))]   ← skl2onnx sklearn
        - [tensor(int64), tensor(float)[N,2]]               ← some XGBoost paths
        - [tensor(int64), list[dict]]                       ← onnxmltools XGBoost
        """
        if len(outputs) == 1:
            arr = np.asarray(outputs[0])
            return arr[:, 1] if arr.ndim == 2 and arr.shape[1] >= 2 else arr.ravel()

        prob_output = outputs[1]

        # seq(map(int64, tensor(float))) → List[Dict[int/long, float]]
        if isinstance(prob_output, list):
            probs = []
            for row in prob_output:
                if isinstance(row, dict):
                    prob1 = row.get(1, row.get(1.0, row.get("1", None)))
                    if prob1 is None:
                        # take second value if keys are unexpected
                        values = list(row.values())
                        prob1 = values[1] if len(values) >= 2 else values[0]
                    probs.append(float(prob1))
                else:
                    probs.append(float(row))
            return np.asarray(probs, dtype=np.float32)

        arr = np.asarray(prob_output)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1].astype(np.float32)
        return arr.ravel().astype(np.float32)

    def _read_channel_recommendation(self) -> str:
        p = (
            self.paths.validation_dir
            / "hrv_agreement"
            / "physiological_validation.json"
        )
        if not p.exists():
            return "UNAVAILABLE"
        payload = json.loads(p.read_text(encoding="utf-8"))
        return str(payload.get("recommended_channel") or "NO_GO")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @classmethod
    def _write_json(cls, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cls._json_safe(payload), indent=2), encoding="utf-8")

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
